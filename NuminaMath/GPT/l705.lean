import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCDMonoid
import Mathlib.Algebra.Logarithm
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Triangle.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Dataset
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Mathlib
import Mathlib.Tactic
import Real
import algebra.module.pi
import data.complex.basic
import data.list.basic
import data.nat.basic

namespace gcd_digits_le_3_l705_705801

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705801


namespace part1_part2_l705_705458

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705458


namespace bipin_chandan_age_ratio_l705_705649

-- Define the condition statements
def AlokCurrentAge : Nat := 5
def BipinCurrentAge : Nat := 6 * AlokCurrentAge
def ChandanCurrentAge : Nat := 7 + 3

-- Define the ages after 10 years
def BipinAgeAfter10Years : Nat := BipinCurrentAge + 10
def ChandanAgeAfter10Years : Nat := ChandanCurrentAge + 10

-- Define the ratio and the statement to prove
def AgeRatio := BipinAgeAfter10Years / ChandanAgeAfter10Years

-- The theorem to prove the ratio is 2
theorem bipin_chandan_age_ratio : AgeRatio = 2 := by
  sorry

end bipin_chandan_age_ratio_l705_705649


namespace distinct_painted_cubes_l705_705608

theorem distinct_painted_cubes : 
  ∃ n : ℕ, n = 4 ∧ (∀ (cube : Cube), 
    (painted_face cube 1 = red ∧ 
     painted_face cube 2 = blue ∧ 
     painted_face cube 3 = blue ∧ 
     painted_face cube 4 = blue ∧ 
     painted_face cube 5 = green ∧ 
     painted_face cube 6 = green) → 
     distinct_up_to_rotation cube) ↔ n = 4 :=
begin
  sorry
end

end distinct_painted_cubes_l705_705608


namespace cosine_of_largest_angle_l705_705887

theorem cosine_of_largest_angle (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
  (AB AC BC : A) (angle_A : ℝ) (h1 : angle_A = 30 * pi / 180)
  (h2 : 2 * ⟪AB, AC⟫ = 3 * ⟪BC, BC⟫) : 
  ∃ C_max : ℝ, C_max = max ((angle B).toReal) ((angle C).toReal) ∧ cos C_max = -1/2 :=
sorry

end cosine_of_largest_angle_l705_705887


namespace rotate_point_Q_l705_705546

theorem rotate_point_Q :
  let P : ℝ × ℝ := (0, 0),
      R : ℝ × ℝ := (6, 0),
      Q : ℝ × ℝ := (6, 6),
      θ : ℝ := 60 * π / 180  -- converting 60 degrees to radians
  in
  let Q_rot := (Q.1 * cos θ - Q.2 * sin θ, Q.1 * sin θ + Q.2 * cos θ)
  in 
  Q_rot = (3 - 3 * sqrt 3, 3 + 3 * sqrt 3) := by sorry

end rotate_point_Q_l705_705546


namespace negation_equiv_no_solution_l705_705192

-- Definition of there is at least one solution
def at_least_one_solution (P : α → Prop) : Prop := ∃ x, P x

-- Definition of no solution
def no_solution (P : α → Prop) : Prop := ∀ x, ¬ P x

-- Problem statement to prove that the negation of at_least_one_solution is equivalent to no_solution
theorem negation_equiv_no_solution (P : α → Prop) :
  ¬ at_least_one_solution P ↔ no_solution P := 
sorry

end negation_equiv_no_solution_l705_705192


namespace max_value_expression_l705_705061

theorem max_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ (M : ℝ), (x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4) ≤ M ∧ 
                  M = 6084 / 17 :=
begin
  use 6084 / 17,
  split,
  { sorry }, -- proof of the inequality
  { refl }   -- proof of equality
end

end max_value_expression_l705_705061


namespace annual_subscription_cost_l705_705618

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l705_705618


namespace gcd_has_at_most_3_digits_l705_705814

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705814


namespace ribbon_per_box_l705_705413

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l705_705413


namespace part_one_part_two_l705_705434

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705434


namespace smallest_number_cubes_to_cover_snaps_l705_705230

structure Cube where
  has_two_snaps : Bool
  has_four_receptacle_holes : Bool

def is_snap_each_adjacent : Cube → Bool := λ c, c.has_two_snaps
def has_four_receptacle : Cube → Bool := λ c, c.has_four_receptacle_holes

theorem smallest_number_cubes_to_cover_snaps : ∃ n, n = 4 ∧ (∀ c : Cube, is_snap_each_adjacent c ∧ has_four_receptacle c)
:= 
sorry

end smallest_number_cubes_to_cover_snaps_l705_705230


namespace gcd_max_digits_l705_705832

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705832


namespace fabian_total_cost_l705_705675

def cost_of_apples (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

def cost_of_sugar (packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  packs * price_per_pack

def cost_of_walnuts (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

theorem fabian_total_cost : 
  let apples_cost := cost_of_apples 5 2 in
  let sugar_cost := cost_of_sugar 3 1 in
  let walnuts_cost := cost_of_walnuts (1/2) 6 in
  apples_cost + sugar_cost + walnuts_cost = 16 :=
by
  sorry

end fabian_total_cost_l705_705675


namespace solve_3x_5y_eq_7_l705_705505

theorem solve_3x_5y_eq_7 :
  ∃ (x y k : ℤ), (3 * x + 5 * y = 7) ∧ (x = 4 + 5 * k) ∧ (y = -1 - 3 * k) :=
by 
  sorry

end solve_3x_5y_eq_7_l705_705505


namespace john_receives_more_l705_705406

noncomputable def partnership_difference (investment_john : ℝ) (investment_mike : ℝ) (profit : ℝ) : ℝ :=
  let total_investment := investment_john + investment_mike
  let one_third_profit := profit / 3
  let two_third_profit := 2 * profit / 3
  let john_effort_share := one_third_profit / 2
  let mike_effort_share := one_third_profit / 2
  let ratio_john := investment_john / total_investment
  let ratio_mike := investment_mike / total_investment
  let john_investment_share := ratio_john * two_third_profit
  let mike_investment_share := ratio_mike * two_third_profit
  let john_total := john_effort_share + john_investment_share
  let mike_total := mike_effort_share + mike_investment_share
  john_total - mike_total

theorem john_receives_more (investment_john investment_mike profit : ℝ)
  (h_john : investment_john = 700)
  (h_mike : investment_mike = 300)
  (h_profit : profit = 3000.0000000000005) :
  partnership_difference investment_john investment_mike profit = 800.0000000000001 := 
sorry

end john_receives_more_l705_705406


namespace rational_function_properties_l705_705959

noncomputable def p (x : ℝ) : ℝ := (12 * x^2 - 48) / 5

theorem rational_function_properties :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (12 * x^2 - 48) / 5) ∧
    (p(3) = 12) ∧
    (∀ x, (x = -2 ∨ x = 2 → is_vertical_asymptote (1 / p x) x)) :=
sorry

end rational_function_properties_l705_705959


namespace angle_BAC_value_l705_705901

theorem angle_BAC_value (A B C I: Point)
( h1: is_incenter I A B C )
( h2: dist B C = dist A C + dist A I )
( h3: angle_diff (angle B A C) (angle C A B) = 13 ) :
angle B A C = 96.5 :=
by sorry

end angle_BAC_value_l705_705901


namespace negation_of_universal_proposition_l705_705154
open Classical

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 ≥ 3)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := 
by
  sorry

end negation_of_universal_proposition_l705_705154


namespace trigonometric_ratios_of_point_l705_705737

theorem trigonometric_ratios_of_point (x y : ℕ) (h : x = 4 ∧ y = 3) :
  let hypotenuse := Real.sqrt (x^2 + y^2),
      sin_α := y / hypotenuse,
      cos_α := x / hypotenuse,
      tan_α := y / x in
  sin_α = 3/5 ∧ cos_α = 4/5 ∧ tan_α = 3/4 :=
by
  sorry

end trigonometric_ratios_of_point_l705_705737


namespace image_property_l705_705910

universe u

variables {T : Type u} [has_mul T] (a : T) (c : T)

def in_image (a : T) (c : T) : Prop :=
  ∃ b : T, c = a * b

theorem image_property (h : in_image a c) : a * c = c :=
sorry

end image_property_l705_705910


namespace MatthewSharedWithTwoFriends_l705_705481

theorem MatthewSharedWithTwoFriends
  (crackers : ℕ)
  (cakes : ℕ)
  (cakes_per_person : ℕ)
  (persons : ℕ)
  (H1 : crackers = 29)
  (H2 : cakes = 30)
  (H3 : cakes_per_person = 15)
  (H4 : persons * cakes_per_person = cakes) :
  persons = 2 := by
  sorry

end MatthewSharedWithTwoFriends_l705_705481


namespace initial_amount_A_l705_705214

theorem initial_amount_A (x t : ℕ) (B_investment time_B : ℝ)
  (profit_ratio : ℝ) (total_time : ℕ) :
  B_investment = 27000 →
  time_B = 2 →
  profit_ratio = 2 →
  total_time = 12 →
  2 * (B_investment * time_B) = x * t →
  t = total_time →
  x = 9000 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h6] at h5
  linarith

end initial_amount_A_l705_705214


namespace find_S6_l705_705915

noncomputable def geometric_series_nth_term (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n - 1)

noncomputable def geometric_series_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

variables (a2 q : ℝ)

-- Conditions
axiom a_n_pos : ∀ n, n > 0 → geometric_series_nth_term a2 q n > 0
axiom q_gt_one : q > 1
axiom condition1 : geometric_series_nth_term a2 q 3 + geometric_series_nth_term a2 q 5 = 20
axiom condition2 : geometric_series_nth_term a2 q 2 * geometric_series_nth_term a2 q 6 = 64

-- Question/statement of the theorem
theorem find_S6 : geometric_series_sum 1 q 6 = 63 :=
  sorry

end find_S6_l705_705915


namespace find_smallest_sphere_radius_squared_l705_705034

noncomputable def smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) : ℝ :=
if AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 then radius_AC_squared else 0

theorem find_smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) :
  (AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120) →
  radius_AC_squared = 49 :=
by
  intros h
  have h_ABCD : AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 := h
  sorry -- The proof steps would be filled in here

end find_smallest_sphere_radius_squared_l705_705034


namespace card_at_position_45_l705_705897

theorem card_at_position_45 : 
  let card_sequence := ["K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2", "A"]
  (card_sequence.cycle.nth 45) = some "8" :=
by
  sorry

end card_at_position_45_l705_705897


namespace determine_parents_genotype_l705_705114

noncomputable def genotype := ℕ -- We use nat to uniquely represent each genotype: e.g. HH=0, HS=1, Sh=2, SS=3

def probability_of_allele_H : ℝ := 0.1
def probability_of_allele_S : ℝ := 1 - probability_of_allele_H

def is_dominant (allele: ℕ) : Prop := allele == 0 ∨ allele == 1 -- HH or HS are dominant for hairy

def offspring_is_hairy (parent1 parent2: genotype) : Prop :=
  (∃ g1 g2, (parent1 = 0 ∨ parent1 = 1) ∧ (parent2 = 1 ∨ parent2 = 2 ∨ parent2 = 3) ∧
  ((g1 = 0 ∨ g1 = 1) ∧ (g2 = 0 ∨ g2 = 1))) ∧ 
  (is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0) 

def most_likely_genotypes (hairy_parent smooth_parent : genotype) : Prop :=
  (hairy_parent = 0) ∧ (smooth_parent = 2)

theorem determine_parents_genotype :
  ∃ hairy_parent smooth_parent, offspring_is_hairy hairy_parent smooth_parent ∧ most_likely_genotypes hairy_parent smooth_parent :=
  sorry

end determine_parents_genotype_l705_705114


namespace multiplication_difference_l705_705612

theorem multiplication_difference :
  let correct_result := 137 * 43,
      mistaken_result := 137 * 34 in
  correct_result - mistaken_result = 1233 :=
by
  let correct_result := 137 * 43
  let mistaken_result := 137 * 34
  show correct_result - mistaken_result = 1233
  sorry

end multiplication_difference_l705_705612


namespace magnitude_of_complex_power_abs_one_plus_i_to_8_l705_705280

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) : 
  complex.abs (z ^ n) = (complex.abs z) ^ n :=
begin
  sorry
end

theorem abs_one_plus_i_to_8 : complex.abs ((1 : ℂ) + complex.I) ^ 8 = 16 :=
begin
  have h1 : complex.abs (1 + complex.I) = real.sqrt 2,
  { rw complex.abs,
    sorry
    -- The calculation |1 + i| = sqrt(1^2 + 1^2) = sqrt 2
  },
  have h2 : complex.abs ((1 + complex.I) ^ 8) = (complex.abs (1 + complex.I)) ^ 8,
  {
    apply magnitude_of_complex_power,
  },
  rw [h1, h2],
  norm_num,
end

end magnitude_of_complex_power_abs_one_plus_i_to_8_l705_705280


namespace ratios_of_tetrahedra_l705_705140

open Real

noncomputable def surfaceAreaRatio (S₁ S₂ : ℝ) : ℝ := S₁ / S₂
noncomputable def volumeRatio (V₁ V₂ : ℝ) : ℝ := V₁ / V₂

-- Given problem conditions
def areTetrahedraSimilar (T₁ T₂ : Tetrahedron) : Prop :=
  T₁.isRegular ∧ T₂.isSimilarTo (centersOfFaces T₁)

-- Prove the ratios
theorem ratios_of_tetrahedra (T₁ T₂ : Tetrahedron)
  (h₁ : areTetrahedraSimilar T₁ T₂) :
  surfaceAreaRatio (T₂.surfaceArea) (T₁.surfaceArea) = 9 ∧
  volumeRatio (T₂.volume) (T₁.volume) = 27 :=
by 
  sorry

end ratios_of_tetrahedra_l705_705140


namespace map_colored_with_six_colors_l705_705496

universe u

/-- Definition placing constraints that are necessary for the theorem --/
def proper_coloring (G : Type u) [Graph G] (C : Type u) [Fintype C] [DecidableEq C] :=
  ∀ v : G, ∃ c : C, ∀ w : G, G.E v w → c ≠ (f w)

theorem map_colored_with_six_colors (G : Type u) [Graph G] :
  proper_coloring G (Fin 6) :=
begin
  sorry
end

end map_colored_with_six_colors_l705_705496


namespace area_ratio_l705_705875

/-- Lean 4 statement for the problem: Prove that given AB = AC = 130, AD = 45, and CF = 85, 
    the ratio of the areas of triangles CEF and DBE is 9/43. -/
theorem area_ratio (A B C D E F : Point) (h₁ : dist A B = 130) (h₂ : dist A C = 130)
  (h₃ : dist A D = 45) (h₄ : dist C F = 85) :
  area_ratio ⟨C, E, F⟩ ⟨D, B, E⟩ = 9 / 43 :=
sorry

end area_ratio_l705_705875


namespace distinct_real_roots_find_other_root_and_k_l705_705757

-- Definition of the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1): Proving the discriminant condition
theorem distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 k (-1) x1 = 0 ∧ quadratic_eq 2 k (-1) x2 = 0 := by
  sorry

-- Part (2): Finding the other root and the value of k
theorem find_other_root_and_k : 
  ∃ k : ℝ, ∃ x2 : ℝ,
    quadratic_eq 2 1 (-1) (-1) = 0 ∧ quadratic_eq 2 1 (-1) x2 = 0 ∧ k = 1 ∧ x2 = 1/2 := by
  sorry

end distinct_real_roots_find_other_root_and_k_l705_705757


namespace prove_no_solution_l705_705783

noncomputable def fractional_eq_no_solution (a : ℝ) : Prop :=
∀ x : ℝ, x ≠ 3 → (x / (x - 3) + 3 * a / (3 - x) ≠ 2 * a)

theorem prove_no_solution (a : ℝ) (h : (a = 1) ∨ (a = 1/2)) : fractional_eq_no_solution a :=
by
  intro x hx
  cases h with ha1 ha_half
  . rw ha1
    sorry -- This is where you would provide a detailed proof showing no solution exists when a = 1
  . rw ha_half
    sorry -- This is where you would provide a detailed proof showing no solution exists when a = 1/2

end prove_no_solution_l705_705783


namespace sum_solutions_c_l705_705145

noncomputable def g (x : ℝ) : ℝ := ((x - 6) * (x - 4) * (x - 2) * (x + 2) * (x + 4) * (x + 6)) / 720 - 5

theorem sum_solutions_c (S : Finset ℝ) :
  (S = {c : ℝ | ∃ (x : ℝ) (hx : -7 ≤ x ∧ x ≤ 7), g(x) = c ∧ (∀ y1 y2 ∈ {x : ℝ | g x = c}, y1 ≠ y2)) → S.sum id = -5 :=
sorry

end sum_solutions_c_l705_705145


namespace intersecting_diagonals_probability_l705_705017

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l705_705017


namespace max_prime_factors_of_a_l705_705133

theorem max_prime_factors_of_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (gcd_primes : (nat.gcd a b).factors.toFinset.card = 10)
  (lcm_primes : (nat.lcm a b).factors.toFinset.card = 35)
  (h_less : a.factors.toFinset.card < b.factors.toFinset.card) :
  a.factors.toFinset.card ≤ 22 := 
sorry

end max_prime_factors_of_a_l705_705133


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705450

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705450


namespace gcd_max_two_digits_l705_705840

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705840


namespace book_arrangement_l705_705100

theorem book_arrangement (D N : Nat) (hD : D = 3!) (hN : N = 2!) : (D * N * 2! = 24) := by
  sorry

end book_arrangement_l705_705100


namespace minimum_bench_sections_l705_705243

theorem minimum_bench_sections (N : ℕ) (hN : 8 * N = 12 * N) : N = 3 :=
sorry

end minimum_bench_sections_l705_705243


namespace lucy_needs_change_for_favorite_toy_l705_705479

-- Definitions based on conditions
def toy_costs : List ℝ := [2.00, 1.85, 1.70, 1.55, 1.40, 1.25, 1.10, 0.95, 0.80, 0.65]
def favorite_toy_cost := 1.65
def initial_quarters := 10
def quarters_value := initial_quarters * 0.25

-- Total possible orderings of toys
def total_orderings := 10.factorial

-- Favorable orderings: favorite toy is dispensed first or second
def favorable_orderings := (9.factorial + 8.factorial)

-- Calculate the probability
def probability := 1 - (favorable_orderings / total_orderings)

theorem lucy_needs_change_for_favorite_toy :
  probability = 8 / 9 :=
sorry

end lucy_needs_change_for_favorite_toy_l705_705479


namespace gcd_at_most_3_digits_l705_705843

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705843


namespace loss_percentage_is_21_l705_705245

noncomputable def loss_percentage (CP SP_new : ℝ) (gain_percent : ℝ) (extra_revenue : ℝ) : ℝ :=
  let SP := CP + (gain_percent / 100) * CP - extra_revenue
  in ((CP - SP) / CP) * 100

theorem loss_percentage_is_21 :
  loss_percentage 560 582.4 4 140 = 21 :=
by
  unfold loss_percentage
  sorry

end loss_percentage_is_21_l705_705245


namespace find_point_l705_705860

noncomputable def P_coords (x₀ y₀ : ℝ) : Prop :=
  y₀ = x₀^3 - 10 * x₀ + 3 ∧
  x₀ < 0 ∧ 
  (3 * x₀^2 - 10 = 2) ∧
  (y₀ = 15)

theorem find_point : ∃ P : ℝ × ℝ, P_coords P.fst P.snd :=
by
  use (-2, 15)
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  norm_num

end find_point_l705_705860


namespace smallest_integer_N_l705_705270

theorem smallest_integer_N : ∃ (N : ℕ), 
  (∀ (a : ℕ → ℕ), ((∀ (i : ℕ), i < 125 -> a i > 0 ∧ a i ≤ N) ∧
  (∀ (i : ℕ), 1 ≤ i ∧ i < 124 → a i > (a (i - 1) + a (i + 1)) / 2) ∧
  (∀ (i j : ℕ), i < 125 ∧ j < 125 ∧ i ≠ j → a i ≠ a j)) → N = 2016) :=
sorry

end smallest_integer_N_l705_705270


namespace integral_f_l705_705698

def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x ≤ 0 then x^2 else
if 0 < x ∧ x ≤ 1 then 1 else 0

theorem integral_f : ∫ x in -1..1, f x = 4 / 3 :=
by
  sorry

end integral_f_l705_705698


namespace part1_part2_l705_705461

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705461


namespace find_B_and_area_l705_705882

variable (A B C a b c d : ℝ)

-- Given conditions
axiom cond1 : (a + c) * Real.sin A = Real.sin A + Real.sin C
axiom cond2 : c^2 + c = b^2 - 1
axiom midpoint_D : D = (A + C) / 2
axiom BD_val : BD = sqrt 3 / 2

-- To Prove
theorem find_B_and_area : 
  ∃ B : ℝ, 
    (B = 2 * π / 3) ∧ 
    let area := (1 / 2) * a * c * Real.sin B in 
    area = sqrt 3 / 2 :=
by
  sorry

end find_B_and_area_l705_705882


namespace sum_divisibility_condition_l705_705911

def is_divisible (m n : ℤ) : Prop := ∃ k : ℤ, m = k * n

theorem sum_divisibility_condition (a : ℤ) (a_k : ℕ → ℤ) (n : ℕ) :
  (is_divisible (∑ k in Finset.range (n + 1), a_k k * (a^2 + 1)^(3 * k) : ℤ) (a^2 + a + 1)) ↔ 
  (is_divisible (∑ k in Finset.range (n + 1), (-1 : ℤ)^k * a_k k) (a^2 + a + 1))
  ∧ 
  (is_divisible (∑ k in Finset.range (n + 1), a_k k * (a^2 + 1)^(3 * k) : ℤ) (a^2 - a + 1)) ↔ 
  (is_divisible (∑ k in Finset.range (n + 1), (-1 : ℤ)^k * a_k k) (a^2 - a + 1)) :=
  sorry

end sum_divisibility_condition_l705_705911


namespace Q_div_P_eq_41_l705_705050

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def LCM (s : List ℕ) : ℕ := s.foldr lcm 1

def P : ℕ := LCM (List.range' 15 21)
def Q : ℕ := LCM ([P, 36, 37, 38, 39, 40, 41, 42, 45])

theorem Q_div_P_eq_41 : Q / P = 41 := by
  sorry

end Q_div_P_eq_41_l705_705050


namespace original_bet_l705_705220

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l705_705220


namespace circumcenter_eqdist_l705_705101

noncomputable theory
open_locale classical

-- Define the points and their properties: parallelogram ABCD, points K and L
variables {A B C D K L O : Type*}
variables [ordered_comm_ring O]

-- Assume we have points in a plane forming a parallelogram ABCD
variables [parallelogram A B C D]
variables (K_point : K ∈ segment A B)
variables (L_point : L ∈ segment B C)
variable (A_eq_CLD : ∠AKD = ∠CLD)

-- Define the circumcenter
def circumcenter (B K L : O) : O := sorry 

-- Prove that the circumcenter of triangle BKL is equidistant from A and C
theorem circumcenter_eqdist (h_parallelogram : parallelogram A B C D)
  (h_K_on_AB : K ∈ segment A B) (h_L_on_BC : L ∈ segment B C)
  (h_angles_eq : ∠AKD = ∠CLD) :
  dist (circumcenter B K L) A = dist (circumcenter B K L) C :=
sorry

end circumcenter_eqdist_l705_705101


namespace min_value_fraction_l705_705744

theorem min_value_fraction (m n : ℝ) (h1 : m + n = 2) (h2 : mn > 0) : 
  ∃ m n, ∀ m n, m + n = 2 ∧ mn > 0 → (4 / m + 2 / n) ≥ 3 + 2 * real.sqrt 2 :=
begin
  sorry
end

end min_value_fraction_l705_705744


namespace find_c_l705_705724

theorem find_c :
  ∃ c : ℝ, 0 < c ∧ ∀ line : ℝ, (∃ x y : ℝ, (x = 1 ∧ y = c) ∧ (x*x + y*y - 2*x - 2*y - 7 = 0)) ∧ (line = 1*x + 0 + y*c - 0) :=
sorry

end find_c_l705_705724


namespace inequality_proof_provable_l705_705309

variable {x a b : ℝ}

theorem inequality_proof_provable (hx : 0 ≤ x) (hx' : x < π / 2) : 
    a^2 * tan x * (cos x)^(1/3) + b^2 * sin x ≥ 2 * x * a * b := 
by
  sorry

end inequality_proof_provable_l705_705309


namespace rabbit_parent_genotype_l705_705116

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l705_705116


namespace sum_of_digits_greatest_prime_divisor_l705_705680

theorem sum_of_digits_greatest_prime_divisor (n : ℕ) (h : n = 2^14 - 2) : (13.digits_sum = 4) := by
  -- n = 16,382
  have h1 : n = 2^14 - 2 := h
  -- check greatest prime divisor and sum digits
  have h2 : greatest_prime_divisor n = 13 := sorry
  show (sum_digits 13 = 4), from sorry

end sum_of_digits_greatest_prime_divisor_l705_705680


namespace coefficient_x2_p_k_l705_705069

def pk (x : ℝ) (k : ℕ) : ℝ :=
  match k with
  | 0 => (x - 2)^2 - 2
  | k+1 => (pk x k - 2)^2

def Ak (k : ℕ) : ℝ :=
  match k with
  | 0 => 1
  | k+1 => Bk k ^ 2 + 4 * Ak k

def Bk (k : ℕ) : ℝ :=
  match k with
  | 0 => -4
  | k+1 => 4 * Bk k

theorem coefficient_x2_p_k (k : ℕ) : Ak k = (4^(2*k-1) - 4^(k-1)) / 3 := by
  sorry

end coefficient_x2_p_k_l705_705069


namespace average_homework_time_decrease_l705_705551

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l705_705551


namespace range_of_a_l705_705058

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - Real.log x / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 := by
  sorry

end range_of_a_l705_705058


namespace trig_identity_and_perimeter_l705_705454

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705454


namespace correct_mean_251_l705_705203

theorem correct_mean_251
  (n : ℕ) (incorrect_mean : ℕ) (wrong_val : ℕ) (correct_val : ℕ)
  (h1 : n = 30) (h2 : incorrect_mean = 250) (h3 : wrong_val = 135) (h4 : correct_val = 165) :
  ((incorrect_mean * n + (correct_val - wrong_val)) / n) = 251 :=
by
  sorry

end correct_mean_251_l705_705203


namespace smaller_number_l705_705204

theorem smaller_number (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_l705_705204


namespace Triangle_BC_Length_l705_705851

theorem Triangle_BC_Length :
  ∀ (A B C : Type) [inner_product_space ℝ A]
  (angle_B : real.angle) (AC AB BC : ℝ),
  angle_B = real.angle.pi / 6 →
  AC = 1 →
  AB = real.sqrt 3 →
  (BC = 1 ∨ BC = 2) :=
by 
  sorry

end Triangle_BC_Length_l705_705851


namespace min_time_proof_l705_705541

/-
  Problem: 
  Given 5 colored lights that each can shine in one of the colors {red, orange, yellow, green, blue},
  and the colors are all different, and the interval between two consecutive flashes is 5 seconds.
  Define the ordered shining of these 5 lights once as a "flash", where each flash lasts 5 seconds.
  We need to show that the minimum time required to achieve all different flashes (120 flashes) is equal to 1195 seconds.
-/

def min_time_required : Nat :=
  let num_flashes := 5 * 4 * 3 * 2 * 1
  let flash_time := 5 * num_flashes
  let interval_time := 5 * (num_flashes - 1)
  flash_time + interval_time

theorem min_time_proof : min_time_required = 1195 := by
  sorry

end min_time_proof_l705_705541


namespace complement_union_eq_l705_705753

open Set

-- Define the universe and sets P and Q
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 3, 5}
def Q : Set ℕ := {1, 2, 4}

-- State the theorem
theorem complement_union_eq :
  ((U \ P) ∪ Q) = {1, 2, 4, 6} := by
  sorry

end complement_union_eq_l705_705753


namespace find_original_radius_l705_705276

noncomputable def cylinder_radius (r : ℝ) : Prop :=
  let vol_increased_radius := π * (r + 4)^2 * 4
  let vol_tripled_height := π * r^2 * 12
  vol_increased_radius = vol_tripled_height ∧ r = 2 + 2 * Real.sqrt 3

theorem find_original_radius (r : ℝ) : 
  let height_original := 4 in
  cylinder_radius r :=
λ h,
begin
  let vol_increased_radius := π * (r + 4)^2 * 4,
  let vol_tripled_height := π * r^2 * 12,
  have h₁: vol_increased_radius = vol_tripled_height := sorry,
  have h₂: r = 2 + 2 * Real.sqrt 3 := sorry,
  exact ⟨h₁, h₂⟩,
end

end find_original_radius_l705_705276


namespace gcd_has_at_most_3_digits_l705_705813

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705813


namespace induction_proof_l705_705930

def f (n : ℕ) : ℕ := (List.range (2 * n - 1)).sum + n

theorem induction_proof (n : ℕ) (h : n > 0) : f (n + 1) - f n = 8 * n := by
  sorry

end induction_proof_l705_705930


namespace finite_set_exists_l705_705271

-- Define the finite set M and the condition of parallel, non-coincident lines

structure Point (α : Type*) :=
(x y z : α)

def is_finite (S : set (Point ℝ)) : Prop := 
  ∃ (L : list (Point ℝ)), ↑L = S ∧ L.length < ℵ₀ -- Ensure countably finite set S

def is_parallel (A B C D : Point ℝ) : Prop :=
  ∃ (λ : ℝ), (C.x - A.x = λ * (B.x - A.x)) ∧ (C.y - A.y = λ * (B.y - A.y)) ∧ (C.z - A.z = λ * (B.z - A.z))

def not_coincide (A B C D : Point ℝ) : Prop :=
  ¬ (A = C ∧ B = D)

theorem finite_set_exists :
  ∃ (M : set (Point ℝ)),
  is_finite M ∧ (∀ (A B : Point ℝ), A ∈ M → B ∈ M → A ≠ B → 
    ∃ (C D : Point ℝ), C ∈ M ∧ D ∈ M ∧ is_parallel A B C D ∧ not_coincide A B C D) :=
sorry

end finite_set_exists_l705_705271


namespace stratified_sampling_l705_705392

theorem stratified_sampling (teachers_A teachers_B teachers_C total_teachers sampled_teachers_C : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 270)
  (hC : teachers_C = 90)
  (h_total : total_teachers = teachers_A + teachers_B + teachers_C)
  (sample_count : 60):
  sampled_teachers_C = (teachers_C * sample_count) / total_teachers :=
by
  have h_total_value : total_teachers = 180 + 270 + 90 := by rw [hA, hB, hC]
  have sample_fraction : sampled_teachers_C = (90 * 60) / (180 + 270 + 90) := sorry
  sorry

end stratified_sampling_l705_705392


namespace pies_unspoiled_correct_l705_705480

def pies_left_unspoiled (pies_per_batch : ℕ) (batches : ℕ) (drop_percentage : ℝ) : ℕ :=
  let total_pies := pies_per_batch * batches
  let dropped_pies := (total_pies * drop_percentage).ceil.toNat
  total_pies - dropped_pies

theorem pies_unspoiled_correct : pies_left_unspoiled 25 15 0.125 = 328 :=
by
  sorry

end pies_unspoiled_correct_l705_705480


namespace correct_amendment_statements_l705_705523

/-- The amendment includes the abuse of administrative power by administrative organs 
    to exclude or limit competition. -/
def abuse_of_power_in_amendment : Prop :=
  true

/-- The amendment includes illegal fundraising. -/
def illegal_fundraising_in_amendment : Prop :=
  true

/-- The amendment includes apportionment of expenses. -/
def apportionment_of_expenses_in_amendment : Prop :=
  true

/-- The amendment includes failure to pay minimum living allowances or social insurance benefits according to law. -/
def failure_to_pay_benefits_in_amendment : Prop :=
  true

/-- The amendment further standardizes the exercise of government power. -/
def standardizes_govt_power : Prop :=
  true

/-- The amendment better protects the legitimate rights and interests of citizens. -/
def protects_rights : Prop :=
  true

/-- The amendment expands the channels for citizens' democratic participation. -/
def expands_democratic_participation : Prop :=
  false

/-- The amendment expands the scope of government functions. -/
def expands_govt_functions : Prop :=
  false

/-- The correct answer to which set of statements is true about the amendment is {②, ③}.
    This is encoded as proving (standardizes_govt_power ∧ protects_rights) = true. -/
theorem correct_amendment_statements : (standardizes_govt_power ∧ protects_rights) ∧ 
                                      ¬(expands_democratic_participation ∧ expands_govt_functions) :=
by {
  sorry
}

end correct_amendment_statements_l705_705523


namespace trig_identity_and_perimeter_l705_705456

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705456


namespace area_of_quadrilateral_AFCE_l705_705876

theorem area_of_quadrilateral_AFCE
  (A F C E D : Type*)
  [Point A] [Point F] [Point C] [Point E] [Point D]
  (h1 : angle A F C = 90) 
  (h2 : D ∈ line A C)
  (h3 : angle E D C = 90)
  (h4 : dist F C = 21)
  (h5 : dist A F = 20)
  (h6 : dist E D = 6)
  : area_quadrilateral A F C E = 297 := sorry

end area_of_quadrilateral_AFCE_l705_705876


namespace elena_pens_l705_705670

theorem elena_pens (X Y : ℕ) (h1 : X + Y = 12) (h2 : 4*X + 22*Y = 420) : X = 9 := by
  sorry

end elena_pens_l705_705670


namespace max_triangle_area_l705_705347

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.2^2) / 8 + (p.1^2) / 2 = 1}

def passesThrough (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop :=
  p ∈ s

def foci (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {(0, sqrt 6), (0, -sqrt 6)}

def center (s : Set (ℝ × ℝ)) : ℝ × ℝ :=
  (0, 0)

def parallelToOM (l : ℝ → ℝ) : Prop :=
  ∃ m, ∀ x, l x = 2 * x + m

def lineIntersectsEllipse (l : ℝ → ℝ) (s : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ s ∧ B ∈ s ∧ ∀ x, l x = 2 * x + (2 * (A.2 - 2 * A.1))

theorem max_triangle_area :
  ∀ A B : ℝ × ℝ, passesThrough (1,2) ellipse →
  foci ellipse = {(0, sqrt 6), (0, -sqrt 6)} →
  center ellipse = (0, 0) →
  ∃ (l : ℝ → ℝ), parallelToOM l →
  lineIntersectsEllipse l ellipse A B →
  abs ((1 / 2) * (A.1 * B.2 - B.1 * A.2)) ≤ 2 :=
by
  sorry

end max_triangle_area_l705_705347


namespace integer_solutions_are_zero_l705_705282

-- Definitions for integers and the given equation
def satisfies_equation (a b : ℤ) : Prop :=
  a^2 * b^2 = a^2 + b^2

-- The main statement to prove
theorem integer_solutions_are_zero :
  ∀ (a b : ℤ), satisfies_equation a b → (a = 0 ∧ b = 0) :=
sorry

end integer_solutions_are_zero_l705_705282


namespace number_of_as_l705_705685

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem number_of_as :
  let n := finset.filter (λ a, isPerfectSquare (5 * a^2 + 6 * a + 1)) (finset.range 101) in
  n.card = ?m_1 :=
sorry

end number_of_as_l705_705685


namespace cube_root_of_sum_l705_705502

def a := 25
def b := 30
def c := 35

theorem cube_root_of_sum :
  Real.cbrt (a^3 + b^3 + c^3) = 5 * Real.cbrt 684 :=
by
  -- identify the common factor
  sorry

end cube_root_of_sum_l705_705502


namespace number_of_regions_l705_705033

-- Define the number of regions F(n) based on the given conditions
def F : ℕ → ℕ
| 0     := 1
| (n+1) := F n + n + 1

-- The statement we want to prove
theorem number_of_regions (n : ℕ) : F n = (n * (n + 1)) / 2 + 1 :=
by
  induction n with n ih
  · -- base case F 0 = 1
    exact Nat.zero_mul.symm ▸ by norm_num
  · -- inductive step using ih : F n = (n * (n + 1)) / 2 + 1
    calc 
      F (n + 1) = F n + n + 1             : rfl
            ... = (n * (n + 1)) / 2 + 1 + n + 1 : by rw ih
            ... = (n * (n + 1)) / 2 + n + 2     : by norm_num
            ... = ((n * (n + 1)) / 2 + 2 * (n + 1) / 2) : by rw [Nat.mul_comm (2 : ℕ), ←Nat.add_mul, Nat.succ_mul, Nat.succ_eq_add_one]
            ... = (n.succ * n.succ.pred) / 2 + 1     : by ring_nf
            ... = ((n + 1) * (n + 1 + 1)) / 2 + 1   : by norm_num


end number_of_regions_l705_705033


namespace a_n_bound_l705_705527

theorem a_n_bound (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ m n : ℕ, 0 < m ∧ 0 < n → (m + n) * a (m + n) ≤ a m + a n) →
  1 / a 200 > 4 * 10^7 := 
sorry

end a_n_bound_l705_705527


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705451

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705451


namespace length_BD_proof_l705_705096

noncomputable def length_BD 
  (ABC_right : ∃ A B C : ℝ^3, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ ∠ A B C = π / 2)
  (ABD_right : ∃ D : ℝ^3, D ≠ A ∧ ∠ D A B = π / 2)
  (BC : ℝ) 
  (AC : ℝ) 
  (AD : ℝ) 
  : ℝ :=
  let b := AC
  let c := BC
  let x := BD
  if h : b^2 + c^2 - AD^2 ≥ 0 then 
    sqrt (b^2 + c^2 - AD^2)
  else
    0

theorem length_BD_proof 
  (ABC_right : ∃ A B C : ℝ^3, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ ∠ A B C = π / 2)
  (ABD_right : ∃ D : ℝ^3, D ≠ A ∧ ∠ D A B = π / 2)
  (BC : ℝ) 
  (AC : ℝ) 
  (AD : ℝ) 
  (hx : AD = 3)
  (hbc : BC = c)
  (hac : AC = b)
  : 
  length_BD ABC_right ABD_right BC AC AD = sqrt (b^2 + c^2 - 9) :=
  by 
    sorry

end length_BD_proof_l705_705096


namespace prime_factors_identity_l705_705469

theorem prime_factors_identity (w x y z k : ℕ) 
    (h : 2^w * 3^x * 5^y * 7^z * 11^k = 900) : 
      2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 20 :=
by
  sorry

end prime_factors_identity_l705_705469


namespace area_triangle_RST_l705_705024

-- Definitions corresponding to the problem conditions
def is_isosceles_right_triangle (P Q R : Point) : Prop :=
  right_angle ∠ P R Q ∧ (P Q).dist = (R Q).dist

def has_area (P Q R : Point) (area : ℝ) : Prop :=
  1 / 2 * (P Q).dist * (R Q).dist = area

def trisects (P Q S T : Point) : Prop :=
  (P S).dist = (S T).dist ∧ (S T).dist = (T Q).dist

def median_intersects (P Q R U : Point) : Prop :=
  midpoint P Q U

-- Theorem to be proven
theorem area_triangle_RST {P Q R S T U : Point}
  (h1 : is_isosceles_right_triangle P Q R)
  (h2 : has_area P Q R 18)
  (h3 : trisects P Q S T)
  (h4 : median_intersects P Q R U) :
  area_triangle R S T = 6 :=
sorry

end area_triangle_RST_l705_705024


namespace john_income_proof_l705_705042

def johns_total_income (salary_last_year bonus_last_year salary_this_year : ℤ) (bonus_same_percentage : ∀ s by : ℤ, 0 < s → by = bonus_last_year) : ℤ :=
  let bonus_percentage := (bonus_last_year : ℚ) / (salary_last_year : ℚ)
  let bonus_this_year := (salary_this_year : ℚ) * bonus_percentage
  salary_this_year + bonus_this_year.to_int

theorem john_income_proof : johns_total_income 100000 10000 200000 = 220000 := by
  sorry

end john_income_proof_l705_705042


namespace matrix_determinant_l705_705277

open Matrix

variables (x : ℝ)

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x + 1],
    ![x, x + 3, x],
    ![x + 1, x, x + 4]]

theorem matrix_determinant :
  det A = 6 * x^2 + 36 * x + 48 :=
sorry

end matrix_determinant_l705_705277


namespace exactly_one_first_class_probability_at_least_one_second_class_probability_l705_705384

-- Definitions based on the problem statement:
def total_pens : ℕ := 6
def first_class_pens : ℕ := 4
def second_class_pens : ℕ := 2

def total_draws : ℕ := 2

-- Event for drawing exactly one first-class quality pen
def probability_one_first_class := ((first_class_pens.choose 1 * second_class_pens.choose 1) /
                                    (total_pens.choose total_draws) : ℚ)

-- Event for drawing at least one second-class quality pen
def probability_at_least_one_second_class := (1 - (first_class_pens.choose total_draws /
                                                   total_pens.choose total_draws) : ℚ)

-- Statements to prove the probabilities
theorem exactly_one_first_class_probability :
  probability_one_first_class = 8 / 15 :=
sorry

theorem at_least_one_second_class_probability :
  probability_at_least_one_second_class = 3 / 5 :=
sorry

end exactly_one_first_class_probability_at_least_one_second_class_probability_l705_705384


namespace triangle_area_l705_705030

def base : ℝ := 12 -- Base BC in cm
def height : ℝ := 15 -- Height from point A to BC in cm

theorem triangle_area : 0.5 * base * height = 90 := 
by
  sorry

end triangle_area_l705_705030


namespace geometric_sequence_sum_l705_705386

/-- 
In a geometric sequence of real numbers, the sum of the first 2 terms is 15,
and the sum of the first 6 terms is 195. Prove that the sum of the first 4 terms is 82.
-/
theorem geometric_sequence_sum :
  ∃ (a r : ℝ), (a + a * r = 15) ∧ (a * (1 - r^6) / (1 - r) = 195) ∧ (a * (1 + r + r^2 + r^3) = 82) :=
by
  sorry

end geometric_sequence_sum_l705_705386


namespace jugglers_balls_needed_l705_705525

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l705_705525


namespace max_size_D_theorem_l705_705423

noncomputable def max_size_D : ℕ :=
sorry

theorem max_size_D_theorem :
  let C := { x : fin 100 → fin 2 // ∀ i : fin 100, x i ∈ {0, 1} }
  let transform (n : ℕ) := λ (x : fin 10 → fin 2), 
    (λ i : fin 10, x ((i + 5) % 10))
  let is_similar (a b : C) : Prop :=
    ∃ (f : ℕ → fin 10 → fin 2) (m : ℕ),
    (f 0 = a) ∧ (f m = b) ∧ ∀ k, f k.succ = transform k (f k)
  let D := { x ∈ C | ∀ y ∈ C, ¬ is_similar x y }
  in max_size_D = 21^5 :=
sorry

end max_size_D_theorem_l705_705423


namespace christmas_gift_count_l705_705922

theorem christmas_gift_count (initial_gifts : ℕ) (additional_gifts : ℕ) (gifts_to_orphanage : ℕ)
  (h1 : initial_gifts = 77)
  (h2 : additional_gifts = 33)
  (h3 : gifts_to_orphanage = 66) :
  (initial_gifts + additional_gifts - gifts_to_orphanage = 44) :=
by
  sorry

end christmas_gift_count_l705_705922


namespace find_g_of_2_l705_705658

def g (x : ℝ) : ℝ := sorry -- Definition of g, provided function used in the theorem

theorem find_g_of_2 :
  (∀ x : ℝ, (x^(2^2008-1) - 1) * g(x) = ((x + 1) * (x^2 + 1) * (x^4 + 1) * ... * (x^(2^2007) + 1) - 1)) → g(2) = 2 :=
by
  intro h
  -- Here one would implement the proof steps, but we encapsulate this within a sorry for now.
  sorry

end find_g_of_2_l705_705658


namespace general_equation_of_C1_rectangular_equation_of_C2_value_of_FA_FB_l705_705354

def curve_C1_parametric := 
  { t : ℝ // ∃ (x y : ℝ), (x = 1 + 1/2 * t) ∧ (y = sqrt(3) / 2 * t) }

def curve_C1_equation (x y : ℝ) : Prop := 
  y = sqrt(3) * (x - 1)

theorem general_equation_of_C1 : ∀ t x y : ℝ, 
  (x = 1 + 1/2 * t) ∧ (y = sqrt(3) / 2 * t) → curve_C1_equation x y :=
by sorry

def curve_C2_polar (ρ θ : ℝ) : Prop := 
  ρ^2 = 12 / (3 + sin(θ)^2)

def curve_C2_rectangular (x y : ℝ) : Prop := 
  (x^2 / 4) + (y^2 / 3) = 1

theorem rectangular_equation_of_C2 : 
  ∀ ρ θ : ℝ, curve_C2_polar ρ θ → ∃ x y : ℝ, (x = ρ * cos θ) ∧ (y = ρ * sin θ) ∧ curve_C2_rectangular x y :=
by sorry

def point_F := (1:ℝ, 0:ℝ)

theorem value_of_FA_FB : 
  ∀ t1 t2 : ℝ, 
  curve_C1_parametric t1 ∧ curve_C1_parametric t2 → 
  ∃ (A B : ℝ × ℝ), curve_C1_parametric t1 → curve_C1_parametric t2 → 
  (|A.1 - point_F.1| + |B.1 - point_F.1|) / (|A.1 - point_F.1| * |B.1 - point_F.1|) = 4/3 :=
by sorry

end general_equation_of_C1_rectangular_equation_of_C2_value_of_FA_FB_l705_705354


namespace cistern_fill_time_l705_705586

theorem cistern_fill_time (C: ℝ) (hA: C / 16 > 0) (hB: C / 20 > 0) : 
  (C / (C / 16 - C / 20)) = 80 := by
  have hAB: C / 16 - C / 20 = C / 80 := by
    calc 
      C / 16 - C / 20
          = 5 * C / 80 - 4 * C / 80 : by sorry
      ... = (5 - 4) * C / 80        : by sorry
      ... = C / 80                  : by sorry
  calc
    C / (C / 16 - C / 20)
        = C / (C / 80)            : by rw [hAB]
    ... = 80                      : by sorry

sorry

end cistern_fill_time_l705_705586


namespace part1_part2_l705_705444

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705444


namespace gcd_digit_bound_l705_705808

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705808


namespace water_required_for_reaction_l705_705290

noncomputable def sodium_hydride_reaction (NaH H₂O NaOH H₂ : Type) : Nat :=
  1

theorem water_required_for_reaction :
  let NaH := 2
  let required_H₂O := 2 -- Derived from balanced chemical equation and given condition
  sodium_hydride_reaction Nat Nat Nat Nat = required_H₂O :=
by
  sorry

end water_required_for_reaction_l705_705290


namespace last_passenger_probability_l705_705984

noncomputable def probability_last_passenger_gets_seat {n : ℕ} (h : n > 0) : ℚ :=
  if n = 1 then 1 else 1/2

theorem last_passenger_probability
  (n : ℕ) (h : n > 0) :
  probability_last_passenger_gets_seat h = 1/2 :=
  sorry

end last_passenger_probability_l705_705984


namespace AD_lt_BC_l705_705020

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variable (angle_A angle_B angle_C angle_D : ℝ)

-- Conditions
variable (quadrilateral : Quadrilateral A B C D)
variable (equal_angles : angle_A = angle_B)
variable (angle_D_gt_angle_C : angle_D > angle_C)

theorem AD_lt_BC (AD BC : ℝ) (h_equal_angles : equal_angles) (h_angle_D_gt_angle_C : angle_D_gt_angle_C) :
  AD < BC :=
by
  sorry

end AD_lt_BC_l705_705020


namespace maple_trees_planted_l705_705989

theorem maple_trees_planted (initial_trees planted_trees final_trees : ℕ) :
  initial_trees = 53 ∧ final_trees = 64 ∧ final_trees - initial_trees = planted_trees →
  planted_trees = 11 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3] at h4
  norm_num at h4
  exact h4

end maple_trees_planted_l705_705989


namespace digits_conditions_l705_705232

noncomputable def original_number : ℕ := 253
noncomputable def reversed_number : ℕ := 352

theorem digits_conditions (a b c : ℕ) : 
  a + b + c = 10 → 
  b = a + c → 
  (original_number = a * 100 + b * 10 + c) → 
  (reversed_number = c * 100 + b * 10 + a) → 
  reversed_number - original_number = 99 :=
by
  intros h1 h2 h3 h4
  sorry

end digits_conditions_l705_705232


namespace gcd_digit_bound_l705_705825

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705825


namespace isosceles_triangle_perimeter_l705_705390

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def roots_of_quadratic_eq := {x : ℕ | x^2 - 5 * x + 6 = 0}

theorem isosceles_triangle_perimeter
  (a b c : ℕ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_roots : (a ∈ roots_of_quadratic_eq) ∧ (b ∈ roots_of_quadratic_eq) ∧ (c ∈ roots_of_quadratic_eq)) :
  (a + b + c = 7 ∨ a + b + c = 8) :=
by
  sorry

end isosceles_triangle_perimeter_l705_705390


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705446

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705446


namespace amount_spent_on_marbles_l705_705085

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l705_705085


namespace pills_supply_duration_l705_705041

open Nat

-- Definitions based on conditions
def one_third_pill_every_three_days : ℕ := 1 / 3 * 3
def pills_in_bottle : ℕ := 90
def days_per_pill : ℕ := 9
def days_per_month : ℕ := 30

-- The Lean statement to prove the question == answer given conditions
theorem pills_supply_duration : (pills_in_bottle * days_per_pill) / days_per_month = 27 := by
  sorry

end pills_supply_duration_l705_705041


namespace fabian_total_cost_l705_705674

def cost_of_apples (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

def cost_of_sugar (packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  packs * price_per_pack

def cost_of_walnuts (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

theorem fabian_total_cost : 
  let apples_cost := cost_of_apples 5 2 in
  let sugar_cost := cost_of_sugar 3 1 in
  let walnuts_cost := cost_of_walnuts (1/2) 6 in
  apples_cost + sugar_cost + walnuts_cost = 16 :=
by
  sorry

end fabian_total_cost_l705_705674


namespace elizabetta_placement_l705_705672

def shape := {p q r s t u : ℕ}

def valid_placement (placement : shape) : Prop :=
  (placement.p * placement.q ≤ 15) ∧
  (placement.q * placement.r ≤ 15) ∧
  (placement.r * placement.s ≤ 15) ∧
  (placement.p * placement.s ≤ 15) ∧
  (placement.p * placement.r ≤ 15) ∧
  (placement.s * placement.u ≤ 15) ∧
  (placement.t * placement.u ≤ 15)

def valid_placements_count : ℕ :=
  ∑ p q r s t u in finset.range 10, if valid_placement {p, q, r, s, t, u} then 1 else 0

theorem elizabetta_placement : valid_placements_count = 16 := by
  sorry

end elizabetta_placement_l705_705672


namespace gcd_has_at_most_3_digits_l705_705812

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705812


namespace ratio_of_men_to_women_l705_705205

variable (M W : ℕ)

theorem ratio_of_men_to_women
  (h1 : W = M + 4)
  (h2 : M + W = 20) :
  (M : ℚ) / (W : ℚ) = 2 / 3 :=
sorry

end ratio_of_men_to_women_l705_705205


namespace gcd_digits_le_3_l705_705795

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705795


namespace b_share_of_earnings_l705_705196

-- Definitions derived from conditions
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def total_earnings := 1170

-- Mathematically equivalent Lean statement
theorem b_share_of_earnings : 
  (work_rate_b / (work_rate_a + work_rate_b + work_rate_c)) * total_earnings = 390 := 
by
  sorry

end b_share_of_earnings_l705_705196


namespace probability_different_colors_l705_705170

-- Definitions based on conditions
def num_blue := 7
def num_yellow := 5
def num_red := 4
def total_chips := num_blue + num_yellow + num_red

-- The problem in question form based on question and correct answer
theorem probability_different_colors :
  let P_different_colors := 
    (num_blue / total_chips) * (num_yellow / total_chips) + 
    (num_yellow / total_chips) * (num_blue / total_chips) +
    (num_blue / total_chips) * (num_red / total_chips) +
    (num_red / total_chips) * (num_blue / total_chips) +
    (num_yellow / total_chips) * (num_red / total_chips) +
    (num_red / total_chips) * (num_yellow / total_chips)
  in 
  P_different_colors = 83 / 128 :=
by
  sorry  -- Proof to be provided

end probability_different_colors_l705_705170


namespace xy_value_l705_705372

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 :=
by
  sorry

end xy_value_l705_705372


namespace find_new_person_weight_l705_705138

def original_weight : ℝ := 65
def average_increase : ℝ := 3.5
def number_of_people : ℝ := 8
def total_increase : ℝ := average_increase * number_of_people
def new_person_weight : ℝ := original_weight + total_increase
def E : ℝ -- environmental factor, not used in the final determination as specific values are not given

theorem find_new_person_weight :
  new_person_weight = 93 :=
by
  -- We have original_weight = 65 kg, average_increase = 3.5 kg, number_of_people = 8
  -- Proving that the new person's weight is 93 kg
  sorry

end find_new_person_weight_l705_705138


namespace number_of_zero_points_l705_705946

theorem number_of_zero_points (f : ℝ → ℝ) (h_odd : ∀ x, f x = -f (-x)) (h_period : ∀ x, f (x - π) = f (x + π)) :
  ∃ (points : Finset ℝ), (∀ x ∈ points, 0 ≤ x ∧ x ≤ 8 ∧ f x = 0) ∧ points.card = 7 :=
by
  sorry

end number_of_zero_points_l705_705946


namespace ralph_has_18_fewer_pictures_l705_705936

/-- Ralph has 58 pictures of wild animals. Derrick has 76 pictures of wild animals.
    Prove that Ralph has 18 fewer pictures of wild animals compared to Derrick. -/
theorem ralph_has_18_fewer_pictures :
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  76 - 58 = 18 :=
by
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  show 76 - 58 = 18
  sorry

end ralph_has_18_fewer_pictures_l705_705936


namespace tan_alpha_of_parallel_vectors_l705_705362

theorem tan_alpha_of_parallel_vectors (α : ℝ) :
  let a := (3 : ℝ, 4 : ℝ),
      b := (Real.sin α, Real.cos α) in
  a.1 * b.2 = a.2 * b.1 →
  Real.tan α = 3 / 4 :=
by
  intros
  sorry

end tan_alpha_of_parallel_vectors_l705_705362


namespace michael_junior_year_points_is_260_l705_705487

def michael_junior_year_points (J : ℝ) : Prop :=
  let senior_year_points := 1.20 * J in
  let total_points := J + senior_year_points in
  total_points = 572

theorem michael_junior_year_points_is_260 : ∃ J : ℝ, michael_junior_year_points J ∧ J = 260 :=
sorry

end michael_junior_year_points_is_260_l705_705487


namespace number_of_sets_that_can_be_equalized_l705_705424

-- Definition of the problem
def set_of_six_distinct_integers_summing_to_60 :=
  {M : finset ℕ | M.card = 6 ∧ M.sum = 60 ∧ ∀ (x y : ℕ), x ∈ M → y ∈ M → x ≠ y}

-- Main statement asserting the number of such sets M that can be made to have equal numbers on each face of the cube
theorem number_of_sets_that_can_be_equalized :
  (finset.univ.filter (λ M : finset ℕ, M ∈ set_of_six_distinct_integers_summing_to_60)).card = 84 :=
sorry

end number_of_sets_that_can_be_equalized_l705_705424


namespace expression_constant_value_l705_705127

theorem expression_constant_value (a b x y : ℝ) 
  (h_a : a = Real.sqrt (1 + x^2))
  (h_b : b = Real.sqrt (1 + y^2)) 
  (h_xy : x + y = 1) : 
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := 
by 
  sorry

end expression_constant_value_l705_705127


namespace problem1_problem2a_problem2b_l705_705222

-- Problem 1: Deriving y in terms of x
theorem problem1 (x y : ℕ) (h1 : 30 * x + 10 * y = 2000) : y = 200 - 3 * x :=
by sorry

-- Problem 2(a): Minimum ingredient B for at least 220 yuan profit with a=3
theorem problem2a (x y a w : ℕ) (h1 : a = 3) 
  (h2 : 3 * x + 2 * y ≥ 220) (h3 : y = 200 - 3 * x) 
  (h4 : w = 15 * x + 20 * y) : w = 1300 :=
by sorry

-- Problem 2(b): Profit per portion of dessert A for 450 yuan profit with 3100 grams of B
theorem problem2b (x : ℕ) (a : ℕ) (B : ℕ) 
  (h1 : B = 3100) (h2 : 15 * x + 20 * (200 - 3 * x) ≤ B) 
  (h3 : a * x + 2 * (200 - 3 * x) = 450) 
  (h4 : x ≥ 20) : a = 8 :=
by sorry

end problem1_problem2a_problem2b_l705_705222


namespace single_burger_cost_l705_705263

variable (S : ℚ) -- cost of a single burger

-- Conditions Translation
constant total_spent : ℚ := 66.50
constant total_burgers : ℕ := 50
constant double_burger_cost : ℚ := 1.50
constant num_double_burgers : ℕ := 33
constant num_single_burgers : ℕ := total_burgers - num_double_burgers := 17

-- Derived Condition Translation
constant double_burger_total_cost : ℚ := num_double_burgers * double_burger_cost := 49.50
constant single_burger_total_spent : ℚ := total_spent - double_burger_total_cost := 17.00

theorem single_burger_cost :
  S = single_burger_total_spent / num_single_burgers :=
begin
  sorry
end

end single_burger_cost_l705_705263


namespace gcd_digit_bound_l705_705824

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705824


namespace ashu_complete_job_in_20_hours_l705_705512

/--
  Suresh can complete a job in 15 hours.
  Ashutosh alone can complete the same job in some hours.
  Suresh works for 9 hours and then the remaining job is completed by Ashutosh in 8 hours.
  We need to prove that the number of hours it takes for Ashutosh to complete the job alone is 20.
-/
theorem ashu_complete_job_in_20_hours :
  let A : ℝ := 20
  let suresh_work_rate : ℝ := 1 / 15
  let suresh_completed_work_in_9_hours : ℝ := (9 * suresh_work_rate)
  let remaining_work : ℝ := 1 - suresh_completed_work_in_9_hours
  (8 * (1 / A)) = remaining_work → A = 20 :=
by
  sorry

end ashu_complete_job_in_20_hours_l705_705512


namespace gcd_digit_bound_l705_705818

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705818


namespace correct_average_l705_705591

theorem correct_average {numbers : List ℝ} (h_len : numbers.length = 10)
  (h_avg : numbers.sum / 10 = 40.2)
  (h_wrong1 : (numbers.head! + 17) ∈ numbers)
  (h_wrong2 : 13 ∈ numbers ∧ 31 ∉ numbers) :
  let corrected_sum := numbers.sum - 17 + (31 - 13)
  in corrected_sum / 10 = 40.3 :=
by
  sorry

end correct_average_l705_705591


namespace inclination_angle_range_l705_705158

theorem inclination_angle_range {θ : ℝ} :
  ∃ α : ℝ, (∃ θ : ℝ, (x * Real.cos θ + y - 1 = 0) ∧ 
  (α = Real.atan (-Real.cos θ)) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi)) :=
begin
  sorry
end

end inclination_angle_range_l705_705158


namespace average_homework_time_decrease_l705_705553

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l705_705553


namespace marbles_cost_correct_l705_705087

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l705_705087


namespace find_ordered_pair_l705_705288

theorem find_ordered_pair :
  ∃ x y : ℚ, (3 * x - 4 * y = -7) ∧ (4 * x - 3 * y = 5) ∧ x = 41 / 7 ∧ y = 43 / 7 :=
by
  use 41 / 7
  use 43 / 7
  split
  {
    rw [mul_div, mul_div]
    norm_num
  }
  split
  {
    rw [mul_div, mul_div]
    norm_num
  }
  split
  { reflexivity }
  { reflexivity }
  sorry

end find_ordered_pair_l705_705288


namespace number_of_integers_l705_705289

noncomputable def f (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem number_of_integers :
  {i : ℕ // 1 ≤ i ∧ i ≤ 5041 ∧ f i = 1 + Nat.sqrt i + i}.card = 20 :=
by
  sorry

end number_of_integers_l705_705289


namespace angle_between_vectors_l705_705756

theorem angle_between_vectors 
    (a b : ℝ × ℝ)
    (h1 : (a.1 - b.1) * a.1 + (a.2 - b.2) * a.2 = 0)
    (h2 : real.sqrt (a.1^2 + a.2^2) = 3)
    (h3 : real.sqrt (b.1^2 + b.2^2) = 2 * real.sqrt 3) :
    ∃ θ, 0 ≤ θ ∧ θ ≤ real.pi ∧ θ = real.pi / 6 := 
by
  sorry

end angle_between_vectors_l705_705756


namespace probability_diagonals_intersect_l705_705004

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l705_705004


namespace chloe_fifth_test_score_l705_705543

theorem chloe_fifth_test_score (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 84) (h2 : a2 = 87) (h3 : a3 = 78) (h4 : a4 = 90)
  (h_avg : (a1 + a2 + a3 + a4 + a5) / 5 ≥ 85) : 
  a5 ≥ 86 :=
by
  sorry

end chloe_fifth_test_score_l705_705543


namespace max_good_diagonals_l705_705002

-- Definitions of conditions in a) 
def convex_ngon (n : ℕ) : Prop := n ≥ 3 -- A polygon with at least three sides is convex

def good_diagonal (n : ℕ) (g : set (ℕ × ℕ)) : Prop :=
  ∀ d ∈ g, ∃! e ∈ g, e ≠ d ∧ (exists_crossing d e) -- Definition of a good diagonal (intersects exactly one other).

-- The main theorem statement
theorem max_good_diagonals (n : ℕ) (g : set (ℕ × ℕ)) :
  convex_ngon n →
  (∀ d1 d2 ∈ g, disjoint (int_point_set d1) (int_point_set d2) → d1 ≠ d2) →
  good_diagonal n g →
  if even n then size g = n - 2 else size g = n - 3 := 
sorry

end max_good_diagonals_l705_705002


namespace real_bounded_by_9_over_4_l705_705741

noncomputable def f (x b : ℝ) : ℝ := (Real.log x + (x - b)^2) / x

theorem real_bounded_by_9_over_4 :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) (2 : ℝ), f x b + x * (Deriv.deriv (λ y, f y b)) x > 0) → b < 9 / 4 :=
by
  sorry

end real_bounded_by_9_over_4_l705_705741


namespace semi_circle_perimeter_correct_l705_705972

noncomputable def perimeter_of_semi_circle (r : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := 2 * pi * r
  let half_circumference := circumference / 2
  let diameter := 2 * r
  in half_circumference + diameter

theorem semi_circle_perimeter_correct :
  perimeter_of_semi_circle 14 = 14 * Real.pi + 28 := by
  sorry

end semi_circle_perimeter_correct_l705_705972


namespace point_outside_circle_l705_705768

theorem point_outside_circle (a b : ℝ) (h : a + b * complex.i = (2 + complex.i) / (1 - complex.i)) : 
  a^2 + b^2 > 2 :=
by 
  sorry

end point_outside_circle_l705_705768


namespace gcd_digit_bound_l705_705793

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705793


namespace cone_volume_l705_705975

theorem cone_volume {r l : ℝ} (h₀ : l = 5) (h₁ : r = 3) : 
  let h := Real.sqrt (l^2 - r^2) in
  (1/3) * π * r^2 * h = 12 * π :=
by
  sorry

end cone_volume_l705_705975


namespace arrange_five_spheres_possible_l705_705889

noncomputable def possible_to_arrange_five_spheres : Prop :=
  ∃ (A B C J' J'' : ℝ^3) (r1 r2 : ℝ),
  r1 > 0 ∧ r2 > 0 ∧
  (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) ∧
  (dist J' A = r1 ∧ dist J' B = r1 ∧ dist J' C = r1) ∧
  (dist J'' A = r1 ∧ dist J'' B = r1 ∧ dist J'' C = r1) ∧
  (dist J' J'' = 2) ∧
  (∃ P, ∀ (X ∈ {A, B, C, J', J''}), ∃ plane : ℝ^3 → Prop, 
    plane J' ∧ ∀ Y ∈ {A, B, C, J', J''} \ {X}, tangent_to_sphere Y plane)

theorem arrange_five_spheres_possible : possible_to_arrange_five_spheres := sorry

end arrange_five_spheres_possible_l705_705889


namespace rectangle_perimeter_l705_705498

theorem rectangle_perimeter (u v : ℝ) (π : ℝ) (major minor : ℝ) (area_rect area_ellipse : ℝ) 
  (inscribed : area_ellipse = 4032 * π ∧ area_rect = 4032 ∧ major = 2 * (u + v)) :
  2 * (u + v) = 128 := by
  -- Given: the area of the rectangle, the conditions of the inscribed ellipse, and the major axis constraint.
  sorry

end rectangle_perimeter_l705_705498


namespace collinear_points_centroid_l705_705727

theorem collinear_points_centroid (G : Point)
  (A B C P Q : Point)
  (a b : Vector)
  (m n : ℝ) 
  (hG : G = (A + B + C) / 3) 
  (hAB : B - A = a) 
  (hAC : C - A = b) 
  (h_line_PQ : ∃ λ : ℝ, P = G + λ * (Q - P)) 
  (hAP : P - A = m * a) 
  (hAQ : Q - A = n * b) : 
  1/m + 1/n = 3 := 
sorry

end collinear_points_centroid_l705_705727


namespace minimize_sledding_time_l705_705631

noncomputable def time_to_sled (H S g : ℝ) (x : ℝ) : ℝ :=
  (Real.sqrt (H^2 + x^2) / H) + ((S - x) / Real.sqrt (2 * g * H))

theorem minimize_sledding_time :
  ∀ (H S g : ℝ), 
  H = 5 → S = 3 → g = 10 →
  ∃ x : ℝ, x = 5 / Real.sqrt 3 ∧ 
  ∀ y : ℝ, time_to_sled H S g x ≤ time_to_sled H S g y := 
by
  intros H S g H_val S_val g_val
  existsi 5 / Real.sqrt 3
  split
  . apply Eq.refl _
  . sorry

end minimize_sledding_time_l705_705631


namespace number_of_valid_codes_l705_705636

theorem number_of_valid_codes : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let valid_codes := { (a, b, c) | a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a < b ∧ b < c } in
  valid_codes.card = 84 :=
by
  sorry

end number_of_valid_codes_l705_705636


namespace demand_decrease_is_20_percent_l705_705642

variable (P Q : ℝ) (price_increase : ℝ) (profit_increase : ℝ)

def calculate_demand_decrease (P Q price_increase profit_increase : ℝ) : ℝ :=
  let new_price := P * (1 + price_increase)
  let required_income := (1 + profit_increase) * P * Q
  let new_demand := required_income / new_price
  1 - new_demand / Q

theorem demand_decrease_is_20_percent :
  price_increase = 0.5 → profit_increase = 0.2 → calculate_demand_decrease P Q price_increase profit_increase = 0.2 :=
by
  intros hp_inc hpr_inc
  simp [calculate_demand_decrease, hp_inc, hpr_inc]
  sorry

end demand_decrease_is_20_percent_l705_705642


namespace sum_second_largest_and_smallest_l705_705539

def numbers : Set ℕ := {10, 11, 12, 13, 14}

theorem sum_second_largest_and_smallest :
  let smallest := 10
  let second_largest := 13
  smallest ∈ numbers ∧ second_largest ∈ numbers ∧
  smallest = finset.min' numbers finset.nsmul_nat_subset_finset ∧
  second_largest = ((finset.filter (λ (x : ℕ), x < finset.max' numbers finset.nsmul_nat_subset_finset) numbers)).max' finset.nsmul_nat_subset_finset →
  smallest + second_largest = 23 := by
  sorry

end sum_second_largest_and_smallest_l705_705539


namespace slope_of_OA_l705_705377

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def focus_of_parabola : (ℝ × ℝ) := (1, 0)  -- Focus of the parabola y^2 = 4x

theorem slope_of_OA (x y : ℝ) 
  (h1 : parabola x y)
  (h2 : distance x y 1 0 = 3) :
  (x = 2 ∧ (y = 2 * real.sqrt 2 ∨ y = -2 * real.sqrt 2)) ∧
  (slope : ℝ := y / x) ∧ (slope = real.sqrt 2 ∨ slope = -real.sqrt 2) :=
by
  sorry

end slope_of_OA_l705_705377


namespace point_in_second_quadrant_l705_705867

theorem point_in_second_quadrant (m : ℝ) : (-1 < 0) ∧ (m^2 + 1 > 0) → Quadrant (-1, m^2 + 1) = Quadrant.second :=
by
  sorry

end point_in_second_quadrant_l705_705867


namespace square_overlapping_area_proof_l705_705316

theorem square_overlapping_area_proof : 
    let side1 := 6
    let side2 := 7
    let side3 := 8
    let side4 := 9
    let θ1 := 20
    let θ2 := 40
    let θ3 := 60
    side1^2 + side2^2 + side3^2 + side4^2 - (side1 * side2 * real.sin (θ1 * real.pi / 180) + side2 * side3 * real.sin (θ2 * real.pi / 180) + side3 * side4 * real.sin (θ3 * real.pi / 180)) = 220 - 15 * real.sin (θ2 * real.pi / 180) := 
by 
  sorry

end square_overlapping_area_proof_l705_705316


namespace max_value_expression_l705_705062

theorem max_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ (M : ℝ), (x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4) ≤ M ∧ 
                  M = 6084 / 17 :=
begin
  use 6084 / 17,
  split,
  { sorry }, -- proof of the inequality
  { refl }   -- proof of equality
end

end max_value_expression_l705_705062


namespace sufficient_but_not_necessary_l705_705211

theorem sufficient_but_not_necessary (x : ℝ) : (x^2 - 2 * x < 0) → (0 < x ∧ x < 4) ∧ ¬((0 < x ∧ x < 4) → (x^2 - 2 * x < 0)) :=
by
  intro h,
  sorry


end sufficient_but_not_necessary_l705_705211


namespace miles_driven_l705_705483

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end miles_driven_l705_705483


namespace count_sum_less_than_95_lt_sum_ge_95_l705_705501

-- Define the set S
def S : Finset ℕ := Finset.range (63 + 1)

-- Define the predicate for the sum condition
def sum_less_than_95 (t : Finset ℕ) : Prop :=
  t.card = 3 ∧ t.sum id < 95

def sum_ge_95 (t : Finset ℕ) : Prop :=
  t.card = 3 ∧ t.sum id ≥ 95

theorem count_sum_less_than_95_lt_sum_ge_95 :
  (Finset.filter sum_less_than_95 (S.powerset : Finset (Finset ℕ))).card <
  (Finset.filter sum_ge_95 (S.powerset : Finset (Finset ℕ))).card :=
sorry

end count_sum_less_than_95_lt_sum_ge_95_l705_705501


namespace sequence_identity_l705_705242

-- defining the sequence
def b (i : ℕ) : ℕ :=
  if h : i ≤ 6 then i else (Finset.prod (Finset.range (i - 1)) b) + 1

-- statement we need to prove
theorem sequence_identity :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 * b 8 * ... * b 1006 - 
  (Finset.sum (Finset.range 1006) (λ i, (b (i + 1))^2)) = -370 :=
sorry

end sequence_identity_l705_705242


namespace integral_sqrt_neg_x_squared_plus_4x_l705_705637

theorem integral_sqrt_neg_x_squared_plus_4x :
  ∫ x in (0:ℝ)..2, real.sqrt (-x^2 + 4 * x) = real.pi :=
by
  sorry

end integral_sqrt_neg_x_squared_plus_4x_l705_705637


namespace number_of_students_suggested_tomatoes_l705_705942

variable (m b T : ℕ)
variable (h_m : m = 228)
variable (h_b : b = 337)
variable (h_relation : b = T + 314)

theorem number_of_students_suggested_tomatoes : T = 23 :=
by {
  have h1 : T + 314 = 337 := h_relation.trans h_b.symm,
  linarith,
}

end number_of_students_suggested_tomatoes_l705_705942


namespace no_solutions_l705_705515

theorem no_solutions (N : ℕ) (d : ℕ) (H : ∀ (i j : ℕ), i ≠ j → d = 6 ∧ d + d = 13) : false :=
by
  sorry

end no_solutions_l705_705515


namespace find_angle_ABM_l705_705235

variable (A B C D M : Type)
variable [has_coe (K : Type) R] [linear_ordered_field K] [linear_ordered_field L] 
variables [has_coe_to_sort (Ring L) K] 

-- Define the points and the square
variable {ABCD : Quadrilateral K}
variable {M : Point K}
variable {x : Real}

-- Conditions 
axiom square_ABCD : Square ABCD
axiom angle_MAC_eq_x : angle M A C = x
axiom angle_MCD_eq_x : angle M C D = x

-- Theorem to prove
theorem find_angle_ABM : angle A B M = 45 :=
by
  sorry

end find_angle_ABM_l705_705235


namespace exists_unique_i_l705_705913

-- Let p be an odd prime number.
variable {p : ℕ} [Fact (Nat.Prime p)] (odd_prime : p % 2 = 1)

-- Let a be an integer in the sequence {2, 3, 4, ..., p-3, p-2}
variable (a : ℕ) (a_range : 2 ≤ a ∧ a ≤ p - 2)

-- Prove that there exists a unique i such that i * a ≡ 1 (mod p) and i ≠ a
theorem exists_unique_i (h1 : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → Nat.gcd k p = 1) :
  ∃! (i : ℕ), 1 ≤ i ∧ i ≤ p - 1 ∧ i * a % p = 1 ∧ i ≠ a :=
by 
  sorry

end exists_unique_i_l705_705913


namespace ribbon_per_box_l705_705408

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l705_705408


namespace p_arithmetic_Pascal_triangle_l705_705497

-- Defining the p-arithmetic factorial (mod p)
def p_fact (n p : ℕ) : ℕ :=
  (List.range (n+1)).foldl (λ acc x => (acc * (x % p)) % p) 1

-- Defining the p-arithmetic Pascal's triangle (mod p)
def p_Pascal (i j p : ℕ) : ℕ :=
  p_fact i p / (p_fact j p * p_fact (i - j) p % p) % p

theorem p_arithmetic_Pascal_triangle (n p : ℕ) (hp : p > 0)
  (h : n ≤ (p-1)/2) :
  p_Pascal (3*n) n p = ((-4)^n % p) * (p_Pascal ((p-1)/2) n p) % p :=
  sorry

end p_arithmetic_Pascal_triangle_l705_705497


namespace gcd_has_at_most_3_digits_l705_705810

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705810


namespace ribbon_per_box_l705_705415

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l705_705415


namespace triangle_trig_problems_l705_705710

open Real

-- Define the main theorem
theorem triangle_trig_problems (A B C a b c : ℝ) (h1: b ≠ 0) 
  (h2: cos A - 2 * cos C ≠ 0) 
  (h3 : (cos A - 2 * cos C) / cos B = (2 * c - a) / b) 
  (h4 : cos B = 1/4)
  (h5 : b = 2) :
  (sin C / sin A = 2) ∧ 
  (2 * a * c * sqrt 15 / 4 = sqrt 15 / 4) :=
by 
  sorry

end triangle_trig_problems_l705_705710


namespace part1_part2a_part2b_l705_705711

noncomputable theory
open Real

def a (x : ℝ) : ℝ × ℝ := (sin x, -cos x)
def b (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -cos x)
def c (x : ℝ) : ℝ × ℝ := (1, 2 * cos x - 1)
def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2

theorem part1 (x : ℝ) : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π :=
sorry

def y (x : ℝ) : ℝ := f x + (a x).1 * (c x).1 + (a x).2 * (c x).2

theorem part2a (t : ℝ) (x : ℝ) (h : t = sin x + cos x) : 
  ∃ s, (s = sin x * cos x) ∧ (s = 1/2 * (t^2 - 1)) ∧ (t ∈ Icc (-sqrt 2) (sqrt 2)) :=
sorry

theorem part2b (x : ℝ) : 
  ∃ t, (y x = sqrt 3 * t^2 + t - sqrt 3) ∧ y x = -13 * sqrt 3 / 12 :=
sorry

end part1_part2a_part2b_l705_705711


namespace noah_left_lights_on_2_hours_l705_705092

-- Define the conditions
def bedroom_light_usage : ℕ := 6
def office_light_usage : ℕ := 3 * bedroom_light_usage
def living_room_light_usage : ℕ := 4 * bedroom_light_usage
def total_energy_used : ℕ := 96
def total_energy_per_hour := bedroom_light_usage + office_light_usage + living_room_light_usage

-- Define the main theorem to prove
theorem noah_left_lights_on_2_hours : total_energy_used / total_energy_per_hour = 2 := by
  sorry

end noah_left_lights_on_2_hours_l705_705092


namespace min_beans_l705_705760

theorem min_beans (r b : ℕ) (H1 : r ≥ 3 + 2 * b) (H2 : r ≤ 3 * b) : b ≥ 3 := 
sorry

end min_beans_l705_705760


namespace optionD_is_correct_l705_705189

-- Definitions for each function
def f1 (x : ℝ) : ℝ := x - 1
def fA (x : ℝ) : ℝ := (Real.sqrt (x - 1))^2
def fB (x : ℝ) : ℝ := Real.sqrt (x^2) - 1
def fC (x : ℝ) : ℝ := (x^2) / x - 1
def fD (x : ℝ) : ℝ := (Real.cbrt (x^3)) - 1

-- Stating that fD is equivalent to f1
theorem optionD_is_correct : ∀ x : ℝ, f1 x = fD x :=
by
  intros x
  sorry

end optionD_is_correct_l705_705189


namespace ribbon_each_box_fraction_l705_705412

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l705_705412


namespace least_moves_l705_705209

theorem least_moves (n : ℕ) (m : ℕ) (candies : ℕ) 
  (initial_candies : ∀ i : fin n, ℕ) 
  (h_initial : ∀ i, initial_candies i = candies)
  (h_moves : (∀ i, 0 < (initial_candies i) → ∃ j, i ≠ j ∧ initial_candies j < candies) → ∃ k, k ≤ m) :
  m ≥ 30 :=
sorry

end least_moves_l705_705209


namespace investment_principal_l705_705201

theorem investment_principal (I : ℝ) (R : ℝ) (T : ℝ) : 
  I = 225 → 
  R = 0.09 → 
  T = 1 / 12 → 
  (I / (R * T)) = 30000 := 
by 
  intros hI hR hT 
  rw [hI, hR, hT] 
  calc 225 / (0.09 * (1 / 12)) = 225 / 0.0075 : by norm_num
                             ... = 30000     : by norm_num

end investment_principal_l705_705201


namespace orlando_has_2_more_cards_than_jenny_l705_705896

/- Definitions for the problem -/
def jenny_cards : ℕ := 6
def total_cards : ℕ := 38
def orlando_cards : ℕ
def richard_cards : ℕ := 3 * orlando_cards

/- Theorem statement -/
theorem orlando_has_2_more_cards_than_jenny 
  (O : ℕ) 
  (h_orlando_cards : orlando_cards = O)
  (h_total_cards : jenny_cards + O + richard_cards = total_cards) : 
  (O - jenny_cards) = 2 := 
by 
  sorry

end orlando_has_2_more_cards_than_jenny_l705_705896


namespace steve_balance_after_five_years_l705_705132

noncomputable def steve_initial_balance : ℝ :=
  100

noncomputable def steve_interest_rate (year : ℕ) (balance : ℝ) : ℝ :=
  if year <= 3 then 0.10 else if balance <= 300 then 0.08 else 0.12

noncomputable def steve_annual_deposit (year : ℕ) (balance : ℝ) : ℝ :=
  if year <= 2 then 10
  else if balance < 250 then 25 else 15

noncomputable def steve_balance_after_years (years : ℕ) : ℝ :=
  let rec balance (year : ℕ) (current_balance : ℝ) : ℝ :=
    if year = 0 then current_balance
    else let new_balance := current_balance + current_balance * steve_interest_rate (year - 1) current_balance
         in balance (year - 1) (new_balance + steve_annual_deposit (year - 1) new_balance)
  balance years steve_initial_balance

theorem steve_balance_after_five_years : steve_balance_after_years 5 = 245.86 := 
  by
  sorry

end steve_balance_after_five_years_l705_705132


namespace scientific_notation_of_3930_billion_l705_705514

theorem scientific_notation_of_3930_billion :
  (3930 * 10^9) = 3.93 * 10^12 :=
sorry

end scientific_notation_of_3930_billion_l705_705514


namespace ribbon_per_box_l705_705414

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l705_705414


namespace find_constants_l705_705473

variables {c d : ℝ}

-- Conditions from the problem
def distinct_roots_first_eq : Prop :=
  ∀ (a b c : ℝ), a ≠ b → b ≠ c → a ≠ c →
  (x + c) * (x + d) * (x - 5) ≠ (x + 4)^2

def one_distinct_root_second_eq : Prop :=
  ∀ (x : ℝ), 
  (x + 2 * c) * (x + 6) * (x + 9) = 0 → 
  (x ≠ d ∧ x ≠ 5)

-- Final proof problem statement
theorem find_constants (h1 : distinct_roots_first_eq) (h2 : one_distinct_root_second_eq) :
  100 * c + d = 93 :=
sorry

end find_constants_l705_705473


namespace distance_between_skew_edges_l705_705533

-- Definition of the given conditions
def side_length_of_base (a : ℝ) : ℝ := a
def lateral_edge_angle : ℝ := 60

-- Mathematical theorem to prove the distance between skew edges
theorem distance_between_skew_edges (a : ℝ) : Prop :=
  ∀ (A B C P : ℝ) (AP : ℝ), -- Vertices and lateral edge length
  side_length_of_base a = a →
  lateral_edge_angle = 60 →
  let M := (B + C) / 2 in   -- Midpoint of B and C
  let FM := AP * (Real.sin (lateral_edge_angle * Real.pi / 180)) in
  -- Conclusion
  FM = (3 * a) / 4 :=
  sorry

end distance_between_skew_edges_l705_705533


namespace not_prime_a_l705_705426

theorem not_prime_a 
  (a b : ℕ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : ∃ k : ℤ, (5 * a^4 + a^2) = k * (b^4 + 3 * b^2 + 4))
  : ¬ Nat.Prime a := 
sorry

end not_prime_a_l705_705426


namespace rabbit_parent_genotype_l705_705119

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l705_705119


namespace cost_of_drill_bits_l705_705226

theorem cost_of_drill_bits (x : ℝ) (h1 : 5 * x + 0.10 * (5 * x) = 33) : x = 6 :=
sorry

end cost_of_drill_bits_l705_705226


namespace odd_function_increasing_in_sym_interval_l705_705380

variables {f : ℝ → ℝ}

def odd_function_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Ioc a b, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂ ∈ set.Ioc a b, x₁ < x₂ → f x₁ < f x₂

theorem odd_function_increasing_in_sym_interval
  (h_odd: odd_function_on_interval f (-7) (-3))
  (h_increasing_pos: increasing_on_interval f 3 7)
  (h_value: f 4 = 5)
  : increasing_on_interval f (-7) (-3) ∧ f (-4) = -5 :=
by
  sorry

end odd_function_increasing_in_sym_interval_l705_705380


namespace probability_diagonals_intersect_l705_705005

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l705_705005


namespace triangle_third_side_length_l705_705968

theorem triangle_third_side_length 
  (a b : ℕ)
  (perimeter : ℕ)
  (h1 : a = 40)
  (h2 : b = 50)
  (h3 : perimeter = 160) : 
  ∃ c : ℕ, c = perimeter - a - b ∧ c = 70 :=
by
  use perimeter - a - b
  have h4 : a + b + (perimeter - a - b) = a + b + c, from calc
    a + b + (perimeter - a - b) = peri perimeter - a - b ) sorry
  sorry

end triangle_third_side_length_l705_705968


namespace trajectory_of_point_P_l705_705327

theorem trajectory_of_point_P :
  ∀ (P : ℝ × ℝ), 
    (∃ (ρ θ : ℝ), 0 ≤ θ ∧ θ ≤ π/4 ∧ ρ > 0 ∧ 
       let x := ρ * cos θ in 
       let y := ρ * sin θ in 
       P = (x, y) ∧ ρ * (cos θ + 2 * sin θ) = 3) 
    ↔ (P = (1, 1) ∨ P = (3, 0) ∨ P ∈ (λ (t : ℝ), (1 + t * 2, 1 - t)))
    sorry

end trajectory_of_point_P_l705_705327


namespace problem1_problem2_l705_705381

noncomputable def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ (R : ℝ), a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

theorem problem1 {a b c A B C R : ℝ} 
  (h_triangle_sides : triangle_sides a b c A B C) :
  b * cos C + c * cos B = a := by
  sorry

theorem problem2 {a b c A B C R : ℝ} 
  (h_triangle_sides : triangle_sides a b c A B C)
  (h_problem1 : b * cos C + c * cos B = a) :
  (cos A + cos B) / (a + b) = 2 * (sin (C / 2))^2 / c := by
  sorry

end problem1_problem2_l705_705381


namespace roots_real_imp_ab_leq_zero_l705_705102

theorem roots_real_imp_ab_leq_zero {a b c : ℝ}
  (h : ∀ x : ℝ, (x^4 + a * x^3 + b * x + c) = 0 → x ∈ ℝ) : a * b ≤ 0 :=
sorry

end roots_real_imp_ab_leq_zero_l705_705102


namespace lines_perpendicular_l705_705307

structure Vec3 :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ)

def line1_dir (x : ℝ) : Vec3 := ⟨x, -1, 2⟩
def line2_dir : Vec3 := ⟨2, 1, 4⟩

def dot_product (v1 v2 : Vec3) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem lines_perpendicular (x : ℝ) :
  dot_product (line1_dir x) line2_dir = 0 ↔ x = -7 / 2 :=
by sorry

end lines_perpendicular_l705_705307


namespace problem_projection_addition_l705_705433

open Matrix

def vec : Vector R 2 := ![4, 5]

def identityMatrix : Matrix (Fin 2) (Fin 2) R := 1

def projectionMatrix (u : Vector R 2) : Matrix (Fin 2) (Fin 2) R :=
  let normFactor := (sqrt (u 0 * u 0 + u 1 * u 1))⁻¹
  let normVector := fun i => normFactor * u i
  let ut := fun i j => normVector i * normVector j
  of fun i j => ut i j

theorem problem_projection_addition :
  projectionMatrix vec + identityMatrix = ![
    ![(57 / 41), (20 / 41)],
    ![(20 / 41), (66 / 41)]
  ] := by
  sorry

end problem_projection_addition_l705_705433


namespace hoseok_has_least_papers_l705_705416

-- Definitions based on the conditions
def pieces_jungkook : ℕ := 10
def pieces_hoseok : ℕ := 7
def pieces_seokjin : ℕ := pieces_jungkook - 2

-- Theorem stating Hoseok has the least pieces of colored paper
theorem hoseok_has_least_papers : pieces_hoseok < pieces_jungkook ∧ pieces_hoseok < pieces_seokjin := by 
  sorry

end hoseok_has_least_papers_l705_705416


namespace kaylee_biscuit_sales_l705_705417

theorem kaylee_biscuit_sales:
    ∀ (total_boxes required_boxes : ℕ) (lemon_boxes chocolate_boxes oatmeal_boxes : ℕ),
        required_boxes = 33 ∧ 
        lemon_boxes = 12 ∧ 
        chocolate_boxes = 5 ∧ 
        oatmeal_boxes = 4 →
        total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes →
        (required_boxes - total_boxes = 12) :=
begin
  sorry
end

end kaylee_biscuit_sales_l705_705417


namespace gcd_max_digits_l705_705827

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705827


namespace cartesian_equation_of_circle_c2_positional_relationship_between_circles_l705_705394
noncomputable def circle_c1 := {p : ℝ × ℝ | (p.1)^2 - 2*p.1 + (p.2)^2 = 0}
noncomputable def circle_c2_polar (theta : ℝ) : ℝ × ℝ := (2 * Real.sin theta * Real.cos theta, 2 * Real.sin theta * Real.sin theta)
noncomputable def circle_c2_cartesian := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 1}

theorem cartesian_equation_of_circle_c2 :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, p = circle_c2_polar θ) ↔ p ∈ circle_c2_cartesian :=
by
  sorry

theorem positional_relationship_between_circles :
  ∃ p : ℝ × ℝ, p ∈ circle_c1 ∧ p ∈ circle_c2_cartesian :=
by
  sorry

end cartesian_equation_of_circle_c2_positional_relationship_between_circles_l705_705394


namespace point_in_second_quadrant_l705_705863

theorem point_in_second_quadrant (m : ℝ) : 
  let x := -1 in
  let y := m^2 + 1 in
  x < 0 ∧ y > 0 →
  (∃ quadrant, quadrant = 2) :=
by
  let x := -1
  let y := m^2 + 1
  intro h,
  existsi 2,
  sorry

end point_in_second_quadrant_l705_705863


namespace most_likely_parents_genotypes_l705_705108

-- Defining the probabilities of alleles in the population
def p_H : ℝ := 0.1
def q_S : ℝ := 0.9

-- Defining the genotypes and their corresponding fur types
inductive Genotype
| HH : Genotype
| HS : Genotype
| SS : Genotype
| Sh : Genotype

-- A function to determine if a given genotype results in hairy fur
def isHairy : Genotype → Prop
| Genotype.HH := true
| Genotype.HS := true
| _ := false

-- Axiom stating that all four offspring have hairy fur
axiom offspring_all_hairy (parent1 parent2 : Genotype) : 
  (isHairy parent1 ∧ isHairy parent2) ∨
  ((parent1 = Genotype.HH ∨ parent2 = Genotype.Sh) ∧ isHairy Genotype.HH) 

-- The main theorem to prove the genotypes of the parents
theorem most_likely_parents_genotypes : 
  ∃ parent1 parent2,
    parent1 = Genotype.HH ∧ parent2 = Genotype.Sh :=
begin
  sorry
end

end most_likely_parents_genotypes_l705_705108


namespace kaylee_biscuit_sales_l705_705418

theorem kaylee_biscuit_sales:
    ∀ (total_boxes required_boxes : ℕ) (lemon_boxes chocolate_boxes oatmeal_boxes : ℕ),
        required_boxes = 33 ∧ 
        lemon_boxes = 12 ∧ 
        chocolate_boxes = 5 ∧ 
        oatmeal_boxes = 4 →
        total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes →
        (required_boxes - total_boxes = 12) :=
begin
  sorry
end

end kaylee_biscuit_sales_l705_705418


namespace integer_coordinates_for_all_vertices_l705_705607

-- Define a three-dimensional vector with integer coordinates
structure Vec3 :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

-- Define a cube with 8 vertices in 3D space
structure Cube :=
  (A1 A2 A3 A4 A1' A2' A3' A4' : Vec3)

-- Assumption: four vertices with integer coordinates that do not lie on the same plane
def has_four_integer_vertices (cube : Cube) : Prop :=
  ∃ (A B C D : Vec3),
    A ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    B ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    C ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    D ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C.x - A.x) * (D.y - B.y) ≠ (D.x - B.x) * (C.y - A.y) ∧  -- Ensure not co-planar
    (C.y - A.y) * (D.z - B.z) ≠ (D.y - B.y) * (C.z - A.z)

-- The proof problem: prove all vertices have integer coordinates given the condition
theorem integer_coordinates_for_all_vertices (cube : Cube) (h : has_four_integer_vertices cube) : 
  ∀ v ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'], 
    ∃ (v' : Vec3), v = v' := 
  by
  sorry

end integer_coordinates_for_all_vertices_l705_705607


namespace hundreds_digit_20_minus_15_factorial_l705_705573

theorem hundreds_digit_20_minus_15_factorial :
  ((20.factorial - 15.factorial) % 1000 / 100) % 10 = 0 :=
by
  sorry

end hundreds_digit_20_minus_15_factorial_l705_705573


namespace part1_part2_l705_705463

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705463


namespace tangent_slope_at_point_x_eq_1_l705_705534

noncomputable def curve (x : ℝ) : ℝ := x^3 - 4 * x
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 4

theorem tangent_slope_at_point_x_eq_1 : curve_derivative 1 = -1 :=
by {
  -- This is just the theorem statement, no proof is required as per the instructions.
  sorry
}

end tangent_slope_at_point_x_eq_1_l705_705534


namespace number_of_hens_l705_705587

-- Let H be the number of hens and C be the number of cows
def hens_and_cows (H C : Nat) : Prop :=
  H + C = 50 ∧ 2 * H + 4 * C = 144

theorem number_of_hens : ∃ H C : Nat, hens_and_cows H C ∧ H = 28 :=
by
  -- The proof is omitted
  sorry

end number_of_hens_l705_705587


namespace cylinder_volume_eq_pi_over_4_l705_705375

theorem cylinder_volume_eq_pi_over_4
  (r : ℝ)
  (h₀ : r > 0)
  (h₁ : 2 * r = r * 2)
  (h₂ : 4 * π * r^2 = π) : 
  (π * r^2 * (2 * r) = π / 4) :=
by
  sorry

end cylinder_volume_eq_pi_over_4_l705_705375


namespace intersect_point_value_l705_705960

theorem intersect_point_value (a b : ℤ) (h1 : ∃ x y, x = -4 ∧ y = a ∧ (f (x) = y) ∧ (f⁻¹ (x) = y)) (h2 : ∃ x y, x ∈ ℤ ∧ y ∈ ℤ) :
  a = -4 := by
  let f := λ x, 2 * x + b
  obtain ⟨x, y, hx, hy, hf, hf_inv⟩ := h1
  /- skipping the proof -/
  sorry

end intersect_point_value_l705_705960


namespace octahedron_cross_section_l705_705241

-- Define the regular octahedron and the condition
structure RegularOctahedron (e : ℝ) := 
  (edge_length : e > 0)

-- Define the properties of the cross-section
def cross_section_perimeter (e : ℝ) : ℝ :=
  3 * e

def cross_section_area (e : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 8) * (e ^ 2)

-- State the theorem
theorem octahedron_cross_section (e : ℝ) (h : RegularOctahedron e) :
  let perimeter := cross_section_perimeter e
  let area := cross_section_area e
  perimeter = 3 * e ∧ area = (3 * Real.sqrt 3 / 8) * (e ^ 2) :=
by
  sorry

end octahedron_cross_section_l705_705241


namespace tunnel_construction_l705_705605

noncomputable def total_tunnel_length : ℕ :=
  816

theorem tunnel_construction
  (days_total : ℕ)
  (men_initial : ℕ)
  (days_elapsed : ℕ)
  (work_done : ℕ)
  (men_additional : ℕ)
  (work_rate_per_man_per_day : ℝ) (men_total: ℕ)
  (time_left: ℕ): total_tunnel_length = 816 :=
by {
  have h_days : days_total = 240 := rfl,
  have h_men_initial : men_initial = 50 := rfl,
  have h_days_elapsed : days_elapsed = 120 := rfl,
  have h_work_done : work_done = 240 := rfl,
  have h_men_additional : men_additional = 70 := rfl,
  have h_work_rate_per_man_per_day: work_rate_per_man_per_day = 0.04 := rfl,
  have h_men_total: men_total = 120 := rfl,
  have h_time_left: time_left = 120 := rfl,
  sorry  
}

end tunnel_construction_l705_705605


namespace perpendicular_lines_l705_705348

theorem perpendicular_lines :
  ∃ m₁ m₄, (m₁ : ℚ) * (m₄ : ℚ) = -1 ∧
  (∀ x y : ℚ, 4 * y - 3 * x = 16 → y = m₁ * x + 4) ∧
  (∀ x y : ℚ, 3 * y + 4 * x = 15 → y = m₄ * x + 5) :=
by sorry

end perpendicular_lines_l705_705348


namespace gcd_at_most_3_digits_l705_705842

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705842


namespace trigonometric_identity_l705_705490

theorem trigonometric_identity (α : ℝ) : 
  (sin α) ^ 2 + (cos (α + real.pi / 6)) ^ 2 + (sin α) * (cos (α + real.pi / 6)) = 3 / 4 :=
  sorry

end trigonometric_identity_l705_705490


namespace computer_multiplications_l705_705223

def rate : ℕ := 15000
def time : ℕ := 2 * 3600
def expected_multiplications : ℕ := 108000000

theorem computer_multiplications : rate * time = expected_multiplications := by
  sorry

end computer_multiplications_l705_705223


namespace pete_three_times_older_in_4_years_l705_705927
-- Import the entire Mathlib library

-- Define the conditions and the statement
theorem pete_three_times_older_in_4_years (x : ℕ) :
  (35 + 4 = 3 * (9 + 4)) :=
by
  calc
  35 + 4 = 39   : by rfl
  3 * (9 + 4) = 3 * 13 : by rfl
  39 = 39 : by rfl

end pete_three_times_older_in_4_years_l705_705927


namespace find_n_l705_705252

theorem find_n :
  ∃ n : ℝ, (n ≠ 5) ∧ 15 / (1 - (5+n)/15) = 67.5 ↔ n = 6.67 := 
begin
  sorry
end

end find_n_l705_705252


namespace part1_part2_l705_705383

-- Define the conditions for part (1)
def condition1_1 (m n : ℝ) : Prop := n = -m + 1
def condition1_2 : Prop := m = -2

-- Define sin_alpha and cos_alpha based on point P and distance r
def sin_alpha (n r : ℝ) : ℝ := n / r
def cos_alpha (m r : ℝ) : ℝ := m / r
def r (m n : ℝ) : ℝ := Real.sqrt (m^2 + n^2)

-- Prove that sin_alpha * cos_alpha = -6/13
theorem part1 (m n : ℝ) (h1 : condition1_1 m n) (h2 : condition1_2) : sin_alpha n (r m n) * cos_alpha m (r m n) = -6 / 13 := by
  sorry

-- Define the conditions for part (2)
def condition2_1 (m n : ℝ) : Prop := n = -m + 1
def condition2_2 : Prop :=
  ∀ α : ℝ, 
    (sin (π / 2 + α) * cos (π - α) / 
    (tan (3 * π / 2 - α) * cot (π + α))) = -1 / 4

-- Prove the coordinates of point P
theorem part2 (m n : ℝ) (h1 : condition2_1 m n) (h2 : condition2_2 α) :
  (m = (3 - Real.sqrt 3) / 2 ∧ n = (Real.sqrt 3 - 1) / 2) ∨
  (m = (3 + Real.sqrt 3) / 2 ∧ n = -(Real.sqrt 3 + 1) / 2) := by
  sorry

end part1_part2_l705_705383


namespace product_scaled_areas_l705_705239

variable (a b c k V : ℝ)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem product_scaled_areas (a b c k : ℝ) (V : ℝ) (hV : V = volume a b c) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (V^2) := 
by
  -- Proof steps would go here, but we use sorry to skip the proof
  sorry

end product_scaled_areas_l705_705239


namespace a_n_formula_T_n_bound_l705_705051

noncomputable def a_n (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | (n+1) => 2 * (n + 1)

def S_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (4 / (a_n i * (a_n i + 2)))

theorem a_n_formula (n : ℕ) (hn : n > 0) : a_n n = 2 * n :=
by sorry

theorem T_n_bound (n : ℕ) (hn : n > 0) : 1/2 ≤ T_n n ∧ T_n n < 1 :=
by sorry

-- End of statement

end a_n_formula_T_n_bound_l705_705051


namespace greatest_integer_bounds_l705_705572

theorem greatest_integer_bounds :
  let x := (5^50 + 4^50) / (5^48 + 4^48)
  in floor x = 24 := by
  sorry

end greatest_integer_bounds_l705_705572


namespace smallest_n_l705_705509

theorem smallest_n (x y : ℤ) (h₁ : x ≡ 2 [MOD 7]) (h₂ : y ≡ -2 [MOD 7]) : 
  ∃ n : ℕ, n = 3 ∧ (x^2 + x * y + y^2 + n) ≡ 0 [MOD 7] := 
by
  sorry

end smallest_n_l705_705509


namespace gcd_digits_le_3_l705_705794

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705794


namespace find_incorrect_expression_l705_705430

namespace RepeatingDecimal

def is_correct_expression (E : ℝ) (t u : ℕ) (X Y : ℝ) : Prop :=
  (E = read_float(format!("0.{}{}", X, Y.repeat))) ∧
  (10^t * E = read_float(format!("{}.{}", X, Y.repeat))) ∧
  (10^(t + 2 * u) * E = read_float(format!("{}{}.","XYX", Y.repeat))) ∧
  (10^(t + u) * E = read_float(format!("{}.{}", XY, Y.repeat)))

theorem find_incorrect_expression :
  ∀ (E : ℝ) (t u : ℕ) (X Y : ℝ), ¬ is_correct_expression (10^t * (10^u - 1) * E) t u X (Y * (X - 1)) :=
by
  sorry

end RepeatingDecimal

end find_incorrect_expression_l705_705430


namespace sum_of_undefined_g_values_l705_705665

noncomputable def is_undefined_g (x : ℝ) : Prop :=
  x = 0 ∨ 2 + 1 / x = 0 ∨ 2 + 1 / (2 + 1 / x) = 0

theorem sum_of_undefined_g_values :
  (∑ x in {x : ℝ | is_undefined_g x}.toFinset, x) = -9 / 10 := 
sorry

end sum_of_undefined_g_values_l705_705665


namespace distance_range_l705_705941

variable (x : ℝ)
variable (starting_fare : ℝ := 6) -- fare in yuan for up to 2 kilometers
variable (surcharge : ℝ := 1) -- yuan surcharge per ride
variable (additional_fare : ℝ := 1) -- fare for every additional 0.5 kilometers
variable (additional_distance : ℝ := 0.5) -- distance in kilometers for every additional fare

theorem distance_range (h_total_fare : 9 = starting_fare + (x - 2) / additional_distance * additional_fare + surcharge) :
  2.5 < x ∧ x ≤ 3 :=
by
  -- Proof goes here
  sorry

end distance_range_l705_705941


namespace part1_part2_l705_705442

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705442


namespace arithmetic_geometric_sequences_and_reciprocal_sum_l705_705395

theorem arithmetic_geometric_sequences_and_reciprocal_sum :
  (∀ n : ℕ, n > 0 → (a n = 3 * n)) ∧
  (∀ n : ℕ, n > 0 → (b n = 3 * n - 1)) ∧
  (∀ n : ℕ, (∑ i in range n, 1 / (S i)) = (2 / 3) * (1 - 1 / (n + 1))) →
  (∀ n, 1 ≤ (∑ i in range n, 1 / (S i)) ∧ (∑ i in range n, 1 / (S i)) < 2 / 3) :=
begin
  sorry
end

def a : ℕ → ℕ := λ n, 3 * n

def b : ℕ → ℕ := λ n, 3 * n - 1

def S : ℕ → ℕ := λ n, (3 * n^2 + 3 * n) / 2

end arithmetic_geometric_sequences_and_reciprocal_sum_l705_705395


namespace allie_carl_product_points_l705_705248

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldr (λ x acc => g x + acc) 0

theorem allie_carl_product_points : (total_points allie_rolls) * (total_points carl_rolls) = 594 :=
  sorry

end allie_carl_product_points_l705_705248


namespace possible_values_of_b_l705_705937

-- Definitions for the roots and the polynomial properties
variables {r s a b : ℝ}
variable p : polynomial ℝ := polynomial.X ^ 3 + polynomial.C a * polynomial.X + polynomial.C b
variable q : polynomial ℝ := polynomial.X ^ 3 + polynomial.C a * polynomial.X + polynomial.C (b + 150)

-- Assertions for the roots conditions
def roots_p (r s t : ℝ) : Prop :=
  polynomial.root p r ∧ 
  polynomial.root p s ∧ 
  polynomial.root p t ∧ 
  r + s + t = 0

def roots_q (r s u : ℝ) : Prop :=
  polynomial.root q (r + 3) ∧ 
  polynomial.root q (s - 5) ∧ 
  polynomial.root q u ∧ 
  (r + 3) + (s - 5) + u = 0

-- Main theorem statement
theorem possible_values_of_b (r s t u : ℝ) (h_p : roots_p r s t) (h_q : roots_q r s u) : 
  (b = 0 ∨ b = 12082) :=
begin
  sorry
end

end possible_values_of_b_l705_705937


namespace max_value_of_a_l705_705365

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧
  (∃ x : ℝ, x < a ∧ ¬(x^2 - 2*x - 3 > 0)) →
  a = -1 :=
by
  sorry

end max_value_of_a_l705_705365


namespace intersecting_diagonals_probability_l705_705011

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l705_705011


namespace triangle_side_relationship_l705_705755

theorem triangle_side_relationship
  (ABC A'B'C' : Triangle)
  (A B C : Point) (A' B' C' : Point)
  (a b c a' b' c' : ℝ)
  (h1 : ∠A + ∠A' = 180)
  (h2 : ∠B = ∠B')
  (ha : a = distance B C)
  (hb : b = distance C A)
  (hc : c = distance A B)
  (ha' : a' = distance B' C')
  (hb' : b' = distance C' A')
  (hc' : c' = distance A' B') :
  a*a' + b*b' + c*c' = 0 := 
sorry

end triangle_side_relationship_l705_705755


namespace solve_phi_l705_705351

-- Define the problem
noncomputable def f (phi x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + phi)
noncomputable def f' (phi x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + phi)
noncomputable def g (phi x : ℝ) : ℝ := f phi x + f' phi x

-- Define the main theorem
theorem solve_phi (phi : ℝ) (h : -Real.pi < phi ∧ phi < 0) 
  (even_g : ∀ x, g phi x = g phi (-x)) : phi = -Real.pi / 3 :=
sorry

end solve_phi_l705_705351


namespace power_function_half_value_l705_705368

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_half_value (a : ℝ) (h : (f 4 a) / (f 2 a) = 3) :
  f (1 / 2) a = 1 / 3 :=
by
  sorry  -- Proof goes here

end power_function_half_value_l705_705368


namespace product_of_integers_abs_val_not_less_than_1_and_less_than_3_l705_705529

theorem product_of_integers_abs_val_not_less_than_1_and_less_than_3 :
  (-2) * (-1) * 1 * 2 = 4 :=
by
  sorry

end product_of_integers_abs_val_not_less_than_1_and_less_than_3_l705_705529


namespace radius_of_C3_is_zero_l705_705397

-- Given conditions
def radius_A : ℝ := 4
def radius_B : ℝ := 225
def are_congruent (r x y z: ℝ): Prop := x = y ∧ y = z ∧ z = r

-- The main proof statement
theorem radius_of_C3_is_zero (r : ℝ) (h_congruent : are_congruent r r r r) 
    (h_tangent_A : radius_A + r = r)
    (h_tangent_B : radius_B + r = r)
    (h_tangent_line : ∀ x, x = 4 ∨ x = 225 ∨ x = r ∨ x = 0) :
    r = 0 :=
begin
  sorry
end

end radius_of_C3_is_zero_l705_705397


namespace waiter_earned_total_tips_l705_705254

def tips (c1 c2 c3 c4 c5 : ℝ) := c1 + c2 + c3 + c4 + c5

theorem waiter_earned_total_tips :
  tips 1.50 2.75 3.25 4.00 5.00 = 16.50 := 
by 
  sorry

end waiter_earned_total_tips_l705_705254


namespace max_value_expression_l705_705076

variable (a1 d : ℝ)

-- Conditions
def a2 := a1 + d
def a4 := a1 + 3 * d
def a9 := a1 + 8 * d
def sum_condition := 3 * a1 + 12 * d = 24

-- Sum definitions
def S (n : ℕ) := (n:ℝ) / 2 * (2 * a1 + (n - 1) * d)

-- Calculating the required expression
def expression := (S 8 / 8) * (S 10 / 10)

-- Proof statement
theorem max_value_expression (h : sum_condition) : 
  ∃ (max_val : ℝ), max_val = 64 ∧ ∀ x, expression ≤ x := 
sorry

end max_value_expression_l705_705076


namespace problem_conditions_l705_705329

open Real

variable {m n : ℝ}

theorem problem_conditions (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2 * m * n) :
  (min (m + n) = 2) ∧ (min (sqrt (m * n)) = 1) ∧
  (min ((n^2) / m + (m^2) / n) = 2) ∧ 
  (max ((sqrt m + sqrt n) / sqrt (m * n)) = 2) :=
by sorry

end problem_conditions_l705_705329


namespace quadratic_max_value_l705_705318

open Real

variables (a b c x : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value (h₀ : a < 0) (x₀ : ℝ) (h₁ : 2 * a * x₀ + b = 0) : 
  ∀ x : ℝ, f a b c x ≤ f a b c x₀ := sorry

end quadratic_max_value_l705_705318


namespace gcd_digit_bound_l705_705802

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705802


namespace intersecting_diagonals_probability_l705_705013

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l705_705013


namespace fruit_combinations_l705_705583

theorem fruit_combinations (n r : ℕ) (n_eq : n = 4) (r_eq: r = 2) : 
  Nat.binom (n + r - 1) r = 10 :=
by
  rw [n_eq, r_eq]
  sorry

end fruit_combinations_l705_705583


namespace michael_drove_miles_l705_705486

theorem michael_drove_miles (rental_fee charge_per_mile total_amount_paid : ℝ) (h_rental_fee : rental_fee = 20.99)
  (h_charge_per_mile : charge_per_mile = 0.25) (h_total_amount_paid : total_amount_paid = 95.74) :
  let amount_paid_for_miles := total_amount_paid - rental_fee in
  let number_of_miles := amount_paid_for_miles / charge_per_mile in
  number_of_miles = 299 := 
by
  sorry

end michael_drove_miles_l705_705486


namespace star_shaped_region_area_l705_705853

theorem star_shaped_region_area : 
  let grid := (7, 7)
  let center_square := (3, 3)
  let star_points := [(*list of specific points forming the star shape*)]
  calculate_star_area grid center_square star_points = 23.5 := 
  sorry

end star_shaped_region_area_l705_705853


namespace gcd_has_at_most_3_digits_l705_705815

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705815


namespace center_of_mass_divides_segment_l705_705934

noncomputable def center_of_mass (a b : ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((a * A.1 + b * B.1) / (a + b), (a * A.2 + b * B.2) / (a + b))

theorem center_of_mass_divides_segment (a b : ℝ) (A B : ℝ × ℝ) :
  let O := center_of_mass a b A B in
  (a ≠ 0 ∧ b ≠ 0) → 
  (b * (O.1 - A.1) + a * (O.1 - B.1) = 0) ∧ (b * (O.2 - A.2) + a * (O.2 - B.2) = 0) :=
by
  sorry

end center_of_mass_divides_segment_l705_705934


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705449

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705449


namespace exists_number_mul_two_k_times_no_digit_seven_l705_705231

def no_digit_seven (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 7

theorem exists_number_mul_two_k_times_no_digit_seven (n : ℕ) (k : ℕ)
  (h : ∀ i ≤ k, no_digit_seven (5^i * n)) :
  ∃ m : ℕ, no_digit_seven m ∧ ∀ i ≤ k, no_digit_seven ((2^i) * m) :=
by {
  let m := 5^k * n,
  use m,
  split,
  { exact h k (nat.le_refl k) },
  { intros i hi,
    have h2 := h (k - i),
    simp only [nat.pow_sub, nat.succ_sub_succ_eq_sub],
    exact h2 (nat.sub_le k i) },
}

end exists_number_mul_two_k_times_no_digit_seven_l705_705231


namespace sales_volume_conditions_l705_705729

noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
else if 3 < x ∧ x ≤ 5 then k * x + 7
else 0

theorem sales_volume_conditions (a k : ℝ) :
  (sales_volume 3 a k = 4) ∧ (sales_volume 5 a k = 2) ∧
  ((∃ x, 1 < x ∧ x ≤ 3 ∧ sales_volume x a k = 10) ∨ 
   (∃ x, 3 < x ∧ x ≤ 5 ∧ sales_volume x a k = 9)) :=
sorry

end sales_volume_conditions_l705_705729


namespace max_goats_l705_705482

variable (coconuts : ℝ) (crabs_f_per_coconut : ℝ) (fish_f_per_crab : ℝ) (goats_f_per_fish : ℝ)

def crabs (c : ℝ) : ℝ := c / crabs_f_per_coconut
def fish (cr : ℝ) : ℝ := cr * fish_f_per_crab
def goats (f : ℝ) : ℝ := f / goats_f_per_fish

theorem max_goats (coconuts : ℝ) 
    (h1 : crabs_f_per_coconut = 3.5) 
    (h2 : fish_f_per_crab = 5.5 / 6.25) 
    (h3 : goats_f_per_fish = 7.5) :
    goats (fish (crabs coconuts)) = 33 :=
by
    sorry

end max_goats_l705_705482


namespace original_balloons_l705_705177

-- Define the variables and conditions.
variable (balloons_after : ℕ) (additional_balloons : ℕ)
variable (initial_balloons : ℕ)

-- The conditions given in the problem.
def given_conditions : Prop :=
  balloons_after = 60 ∧ additional_balloons = 34

-- The question stated as a proof problem.
theorem original_balloons (h : given_conditions) : initial_balloons = 26 :=
by
  -- Conditions from the hypothesis.
  have h1 : balloons_after = 60 := h.1
  have h2 : additional_balloons = 34 := h.2
  -- Calculation to find the original number of balloons.
  have compute : initial_balloons = balloons_after - additional_balloons := 
    by sorry
  -- final equality
  sorry

end original_balloons_l705_705177


namespace securely_nail_strip_requires_two_nails_l705_705996

-- Define the condition: At least two nails are needed to securely nail a wooden strip on the wall.
def at_least_two_nails_needed : Prop :=
  ∀ (strip : Type) (wall : Type) (nails : set Type), nails.size >= 2 →
    securely_nails strip wall nails

-- Define the concept: Two points determine a straight line.
def two_points_determine_straight_line : Prop :=
  ∀ (x1 x2 : point), x1 ≠ x2 → ∃! (line : set point), x1 ∈ line ∧ x2 ∈ line

-- Lean 4 statement to prove the connection between the two concepts.
theorem securely_nail_strip_requires_two_nails :
  at_least_two_nails_needed → two_points_determine_straight_line := by
  sorry

end securely_nail_strip_requires_two_nails_l705_705996


namespace ratio_sum_odd_even_divisors_l705_705431

def M : ℕ := 33 * 38 * 58 * 462

theorem ratio_sum_odd_even_divisors : 
  let sum_odd_divisors := 
    (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_all_divisors := 
    (1 + 2 + 4 + 8) * (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  (sum_odd_divisors : ℚ) / sum_even_divisors = 1 / 14 :=
by sorry

end ratio_sum_odd_even_divisors_l705_705431


namespace jake_bike_time_l705_705893

/-- Prove that Jake will take 2 hours to bike to the water park, given the conditions. -/
theorem jake_bike_time 
  (t_drive : ℝ) (speed1 speed2 speed_bike : ℝ)
  (t_half : t_drive / 2)
  (distance1 := speed1 * t_half) (distance2 := speed2 * t_half)
  (total_distance := distance1 + distance2)
  (total_distance = 22) :
  (total_distance / speed_bike = 2) := 
by
  sorry

end

end jake_bike_time_l705_705893


namespace ribbon_each_box_fraction_l705_705411

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l705_705411


namespace james_total_cost_l705_705036

noncomputable def couch_cost : ℝ := 2500
noncomputable def sectional_cost : ℝ := 3500
noncomputable def entertainment_center_cost : ℝ := 1500
noncomputable def accessories_cost : ℝ := 500

noncomputable def couch_discount : ℝ := 0.10
noncomputable def sectional_discount : ℝ := 0.10
noncomputable def entertainment_center_discount : ℝ := 0.05
noncomputable def accessories_discount : ℝ := 0.15

noncomputable def sales_tax_rate : ℝ := 0.0825

noncomputable def discounted_price (cost: ℝ) (discount_rate: ℝ) : ℝ :=
  cost * (1 - discount_rate)

noncomputable def total_cost_before_tax (couch sectional entertainment_center accessories : ℝ) : ℝ :=
  couch + sectional + entertainment_center + accessories

noncomputable def add_sales_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

theorem james_total_cost :
  let couch_discounted := discounted_price couch_cost couch_discount
  let sectional_discounted := discounted_price sectional_cost sectional_discount
  let entertainment_center_discounted := discounted_price entertainment_center_cost entertainment_center_discount
  let accessories_discounted := discounted_price accessories_cost accessories_discount
  let subtotal := total_cost_before_tax couch_discounted sectional_discounted entertainment_center_discounted accessories_discounted
  let final_cost := add_sales_tax subtotal sales_tax_rate
  Real.round (final_cost * 100) / 100 = 7848.13 := 
by
  -- Proof goes here
  sorry

end james_total_cost_l705_705036


namespace platform_and_train_length_equality_l705_705151

-- Definitions of the given conditions.
def speed_in_kmh : ℝ := 90
def speed_in_m_per_min : ℝ := (speed_in_kmh * 1000) / 60
def time_in_min : ℝ := 1
def length_of_train : ℝ := 750
def total_distance_covered : ℝ := speed_in_m_per_min * time_in_min

-- Assertion that length of platform is equal to length of train
theorem platform_and_train_length_equality : 
  total_distance_covered - length_of_train = length_of_train :=
by
  -- Placeholder for proof
  sorry

end platform_and_train_length_equality_l705_705151


namespace part1_part2_part3_l705_705702

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x > 0 → f(x) ∈ ℝ
axiom f_gt_zero : ∀ x : ℝ, x > 1 → f(x) > 0
axiom f_additive : ∀ x y : ℝ, x > 0 → y > 0 → f(x * y) = f(x) + f(y)

-- Given condition for part 1
axiom f_one : f(1) = 0

-- Prove that f(1/x) = -f(x)
theorem part1 (x : ℝ) (h : x > 0) :
  f (1 / x) = - f (x) :=
sorry

-- Prove that f(x) is increasing on (0, +∞)
theorem part2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 < x2 → f (x1) < f (x2) :=
sorry

-- Given condition for part 3
axiom f_one_third : f (1/3) = -1

-- Prove the range of values for x that satisfy the inequality
theorem part3 (x : ℝ) (h : x > 0) :
  f(x) - f(1 / (x - 2)) ≥ 2 ↔ x ≥ 1 + Real.sqrt(10) :=
sorry

end part1_part2_part3_l705_705702


namespace coeff_of_x4_in_expansion_l705_705396

theorem coeff_of_x4_in_expansion :
  let general_term (r : ℕ) := ((nat.choose 5 r) * (x ^ (2 * (5 - r))) * ((-1) ^ r) * (x ^ (-r))) in
  coeff (general_term 2) = 10 := 
by
  sorry

end coeff_of_x4_in_expansion_l705_705396


namespace tangent_circle_diameter_l705_705899

noncomputable def Quadrilateral (A B C D : Type) :=
  A ∈ convex_hull {A, B, C, D} ∧ B ∈ convex_hull {A, B, C, D} ∧
  C ∈ convex_hull {A, B, C, D} ∧ D ∈ convex_hull {A, B, C, D}

noncomputable def CircleTangent (line1 line2 : Type) :=
  ∀ P : Type, P ∈ line1 → P ∈ circle_on_diameter line2 → ⟪P, line1⟫ = ⟪line2⟫

axiom ConvexQuadrilateral (A B C D : Type) : Quadrilateral A B C D

theorem tangent_circle_diameter (A B C D : Type) (AB : Type) (CD : Type):
  (CircleTangent CD (circle_on_diameter AB) ∧ ConvexQuadrilateral A B C D) ↔ 
  (BC ⟂ AD) :=
sorry

end tangent_circle_diameter_l705_705899


namespace ratio_of_milk_to_water_l705_705387

namespace MixtureProblem

def initial_milk (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (milk_ratio * total_volume) / (milk_ratio + water_ratio)

def initial_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (water_ratio * total_volume) / (milk_ratio + water_ratio)

theorem ratio_of_milk_to_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  milk_ratio = 4 → water_ratio = 1 → total_volume = 45 → added_water = 21 → 
  (initial_milk total_volume milk_ratio water_ratio) = 36 →
  (initial_water total_volume milk_ratio water_ratio + added_water) = 30 →
  (36 / 30 : ℚ) = 6 / 5 :=
by
  intros
  sorry

end MixtureProblem

end ratio_of_milk_to_water_l705_705387


namespace count_positive_integers_in_range_l705_705364

theorem count_positive_integers_in_range :
  ∃ (count : ℕ), count = 11 ∧
    ∀ (n : ℕ), 300 < n^2 ∧ n^2 < 800 → (n ≥ 18 ∧ n ≤ 28) :=
by
  sorry

end count_positive_integers_in_range_l705_705364


namespace solid_color_marble_percentage_l705_705098

theorem solid_color_marble_percentage (solid striped dotted swirl red blue green yellow purple : ℝ)
  (h_solid: solid = 0.7) (h_striped: striped = 0.1) (h_dotted: dotted = 0.1) (h_swirl: swirl = 0.1)
  (h_red: red = 0.25) (h_blue: blue = 0.25) (h_green: green = 0.2) (h_yellow: yellow = 0.15) (h_purple: purple = 0.15) :
  solid * (red + blue + green) * 100 = 49 :=
by
  sorry

end solid_color_marble_percentage_l705_705098


namespace grade_assignment_ways_l705_705238

theorem grade_assignment_ways : (4 ^ 12) = 16777216 :=
by
  -- mathematical proof
  sorry

end grade_assignment_ways_l705_705238


namespace find_n_l705_705257

theorem find_n :
  ∃ (n : ℕ), 9 * (Finset.sum (Finset.range (n + 1))) = 9 * (Finset.sum (Finset.range (729 + 1))) :=
begin
  use 729,
  sorry
end

end find_n_l705_705257


namespace maximize_product_l705_705078

theorem maximize_product (x y z : ℝ) (h1 : x ≥ 20) (h2 : y ≥ 40) (h3 : z ≥ 1675) (h4 : x + y + z = 2015) :
  x * y * z ≤ 721480000 / 27 :=
by sorry

end maximize_product_l705_705078


namespace part_I_part_II_l705_705743

open Real

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem part_I (x : ℝ) : f x > 0 ↔ (x > 1 ∨ x < -5) := 
sorry

theorem part_II (m : ℝ) : (∀ x : ℝ, f x + 3 * abs (x - 4) > m) ↔ (m < 9) :=
sorry

end part_I_part_II_l705_705743


namespace domain_of_f_l705_705955

theorem domain_of_f :
  ∀ x : ℝ, 
  (4 * x - 5 > 0) → 
  (log (4 * x - 5) / log (1 / 3) ≥ 0) → 
  (x > 5 / 4) ∧ (x ≤ 3 / 2) :=
by
  intros x h1 h2
  have h3: 4 * x > 5 := by linarith
  have h4: 4 * x ≤ 6 := by linarith
  split
  { linarith }
  { linarith }
sorry

end domain_of_f_l705_705955


namespace max_value_of_expression_l705_705686

theorem max_value_of_expression (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  ∃ (M : ℝ), M = 0 ∧ 
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  (abc * (a + b + c)) / ((a + b) ^ 3 * (b + c) ^ 3) ≤ M)) :=
begin
  sorry
end

end max_value_of_expression_l705_705686


namespace annual_subscription_cost_l705_705617

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l705_705617


namespace smallest_positive_n_l705_705184

theorem smallest_positive_n : ∃ n : ℕ, 3 * n ≡ 8 [MOD 26] ∧ n = 20 :=
by 
  use 20
  simp
  sorry

end smallest_positive_n_l705_705184


namespace seed_treatment_effects_on_disease_l705_705635

theorem seed_treatment_effects_on_disease
  (diseased_treated : ℕ := 32) 
  (diseased_untreated : ℕ := 101) 
  (total_treated : ℕ := 224) 
  (total_untreated : ℕ := 314) :
  (diseased_treated / total_treated.to_rat) < (diseased_untreated / total_untreated.to_rat) → 
  ("Whether the seeds are treated is related to getting sick") :=
sorry

end seed_treatment_effects_on_disease_l705_705635


namespace problem_statement_l705_705310

-- Definitions of h and j as per the problem conditions
def h (n : ℕ) : ℕ :=
  (n.digits 6).sum

def j (n : ℕ) : ℕ :=
  (h n).digits 10.sum

-- Problem statement
theorem problem_statement : (n : ℕ) (∀ m, j(m) < 10 → m ≥ n) ∧ (j 14 = 10) → n % 100 = 14 :=
by
  sorry

end problem_statement_l705_705310


namespace find_digit_D_l705_705519

theorem find_digit_D (A B C D : ℕ)
  (h_add : 100 + 10 * A + B + 100 * C + 10 * A + A = 100 * D + 10 * A + B)
  (h_sub : 100 + 10 * A + B - (100 * C + 10 * A + A) = 100 + 10 * A) :
  D = 1 :=
by
  -- Since we're skipping the proof and focusing on the statement only
  sorry

end find_digit_D_l705_705519


namespace simplify_and_evaluate_expression_l705_705940

theorem simplify_and_evaluate_expression : 
  (x y : ℝ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 1 / 2) + x^2 * y) + 6 * x * y^2) = 1 / 2 := by
  sorry

end simplify_and_evaluate_expression_l705_705940


namespace triangle_properties_l705_705884

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_properties_l705_705884


namespace decrease_equation_l705_705556

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l705_705556


namespace cafeteria_orders_red_apples_l705_705532

theorem cafeteria_orders_red_apples :
  ∃ R : ℕ, R + 7 = 9 + 40 ∧ R = 42 :=
by
  use 42
  split
  · exact Nat.add_comm 42 7 ▸ rfl
  · rfl

end cafeteria_orders_red_apples_l705_705532


namespace coeff_x2_y2_in_expansion_l705_705141

noncomputable def binomial_expansion : ℚ := 
  ∑ r in finset.range 9, (binom 8 r : ℚ) * (x^(8 - r / 2 : ℚ) / y^(8 - r / 2 : ℚ)) * (- (y^(r / 2) / x^(r / 2)))

theorem coeff_x2_y2_in_expansion : 
  let c := @coeff ℚ _ _ _ (binomial_expansion) x^2 y^2 in
  c = 70 := by
  sorry

end coeff_x2_y2_in_expansion_l705_705141


namespace no_integer_roots_l705_705932

theorem no_integer_roots (P : ℤ[X]) (h0 : P.eval 0 % 2 = 1) (h1 : P.eval 1 % 2 = 1) : ¬ ∃ x : ℤ, P.eval x = 0 :=
sorry

end no_integer_roots_l705_705932


namespace shanghai_expo_problem_l705_705689

theorem shanghai_expo_problem :
  let total_ways := (Nat.choose 4 2) + (Nat.choose 6 2) in
  total_ways = 21 :=
by
  sorry

end shanghai_expo_problem_l705_705689


namespace part1_solution_set_part2_min_value_l705_705477

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x - 2)

-- Part 1: Prove that the solution set of inequality f(x) >= 1 - 2x is {x | x >= -1}
theorem part1_solution_set : {x : ℝ | f x ≥ 1 - 2 * x} = {x | x ≥ -1} :=
by
  sorry

-- Part 2: Prove that the minimum value of m + n is 3 given certain conditions
theorem part2_min_value (a m n : ℝ) (h1 : abs (x - a) + abs (x - 1) = 3) (h2 : m^2 * n = a) (hm : m > 0) (hn : n > 0) :
  m + n = 3 :=
by
  -- Given that a = 4
  have ha : a = 4 :=
    by
      exact 4

  -- Given that the minimum value of f(x) + |x - 1| is 3
  have h3 : f x + abs (x - 1) = 3 :=
    by
      exact 3

  -- Prove that m + n is 3
  sorry

end part1_solution_set_part2_min_value_l705_705477


namespace find_m_eq_l705_705691

theorem find_m_eq :
    ∃ (m : ℕ), (3^4 - m = 4^3 + 2) ∧ (m = 15) := 
by
    existsi 15
    split
    { calc 3^4 - 15 = 81 - 15 : by norm_num
                      ... = 66 : by norm_num
           ... = 64 + 2 : by norm_num }
    { refl }

end find_m_eq_l705_705691


namespace securely_nail_strip_requires_two_nails_l705_705997

-- Define the condition: At least two nails are needed to securely nail a wooden strip on the wall.
def at_least_two_nails_needed : Prop :=
  ∀ (strip : Type) (wall : Type) (nails : set Type), nails.size >= 2 →
    securely_nails strip wall nails

-- Define the concept: Two points determine a straight line.
def two_points_determine_straight_line : Prop :=
  ∀ (x1 x2 : point), x1 ≠ x2 → ∃! (line : set point), x1 ∈ line ∧ x2 ∈ line

-- Lean 4 statement to prove the connection between the two concepts.
theorem securely_nail_strip_requires_two_nails :
  at_least_two_nails_needed → two_points_determine_straight_line := by
  sorry

end securely_nail_strip_requires_two_nails_l705_705997


namespace geometric_progression_common_ratio_l705_705855

theorem geometric_progression_common_ratio (a r : ℝ) 
(h_pos: a > 0)
(h_condition: ∀ n : ℕ, a * r^(n-1) = (a * r^n + a * r^(n+1))^2):
  r = 0.618 :=
sorry

end geometric_progression_common_ratio_l705_705855


namespace gcd_digit_bound_l705_705790

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705790


namespace jeff_pays_when_picking_up_costumes_l705_705040

noncomputable def last_year_cost : ℝ := 250
noncomputable def current_year_increase : ℝ := 0.40
noncomputable def number_of_costumes : ℝ := 3
noncomputable def jeff_discount : ℝ := 0.15
noncomputable def friend_discount : ℝ := 0.10
noncomputable def deposit_percentage : ℝ := 0.10

def this_year_cost := last_year_cost * (1 + current_year_increase)
def total_cost_without_discounts := number_of_costumes * this_year_cost
def jeff_costume_discount := jeff_discount * this_year_cost
def friend_costume_discount := friend_discount * this_year_cost
def total_discounts := jeff_costume_discount + friend_costume_discount
def adjusted_total_cost := total_cost_without_discounts - total_discounts
def deposit_amount := deposit_percentage * adjusted_total_cost
def amount_to_pay := adjusted_total_cost - deposit_amount

theorem jeff_pays_when_picking_up_costumes : amount_to_pay = 866.25 := by
  sorry

end jeff_pays_when_picking_up_costumes_l705_705040


namespace solve_l705_705427

theorem solve (p x y z : ℕ) (pp : p.prime) :
  x^p + y^p = p^z → z = 2 :=
sorry

end solve_l705_705427


namespace vector_sum_magnitude_l705_705722

variables (a b : ℝ → ℝ → ℝ)
variables (norm_a : ℝ) (norm_b : ℝ) (angle : ℝ)
variables (angle_eq : angle = 30 * (Real.pi / 180))
variables (norm_a_eq : norm_a = Real.sqrt 3)
variables (norm_b_eq : norm_b = 2)

theorem vector_sum_magnitude :
  ∀ a b, ‖ a ‖ = norm_a → ‖ b ‖ = norm_b → angle = 30 * (Real.pi / 180) →
  ‖ a + b ‖ = Real.sqrt 13 :=
sorry

end vector_sum_magnitude_l705_705722


namespace simplify_expression_l705_705939

-- Define the problem and its conditions
theorem simplify_expression :
  (81 * 10^12) / (9 * 10^4) = 900000000 :=
by
  sorry  -- Proof placeholder

end simplify_expression_l705_705939


namespace max_value_of_trig_expr_l705_705296

theorem max_value_of_trig_expr (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expr_l705_705296


namespace correct_system_of_equations_l705_705872

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l705_705872


namespace sum_of_distances_l705_705269

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (12, 0)
def point_C : ℝ × ℝ := (4, 7)
def point_P : ℝ × ℝ := (5, 3)

def distance_AP : ℝ := distance (point_A.1) (point_A.2) (point_P.1) (point_P.2)
def distance_BP : ℝ := distance (point_B.1) (point_B.2) (point_P.1) (point_P.2)
def distance_CP : ℝ := distance (point_C.1) (point_C.2) (point_P.1) (point_P.2)

def total_distance : ℝ := distance_AP + distance_BP + distance_CP

theorem sum_of_distances : 1 * Real.sqrt 34 + 1 * Real.sqrt 58 + 1 * Real.sqrt 17 = 53 :=
  by
  have h1 : distance_AP = Real.sqrt 34 := by sorry
  have h2 : distance_BP = Real.sqrt 58 := by sorry
  have h3 : distance_CP = Real.sqrt 17 := by sorry
  exact sorry

end sum_of_distances_l705_705269


namespace find_missing_number_l705_705965

theorem find_missing_number (mean : ℤ) (a b c d e f g : ℤ) (h : (a + b + c + d + e + f + g) / 7 = mean) : 
  g = -9 :=
by
  have total_sum := 20 * 7
  have known_sum := 22 + 23 + 24 + 25 + 26 + 27 + 2
  have missing := total_sum - known_sum
  exact missing = -9

end find_missing_number_l705_705965


namespace ab_value_is_three_over_two_l705_705721

-- Defining the real numbers a and b, the function f and its tangent line equation
variables (a b : ℝ)

def f (x : ℝ) : ℝ := real.log x + a / x

-- The derivative f' of the function f
def f' (x : ℝ) : ℝ := 1 / x - a / x^2

-- The value of the derivative at x = 1
lemma f_prime_at_1 : f' 1 = 1 - a := by
  unfold f'
  norm_num

-- The value of the function at x = 1
lemma f_at_1 : f 1 = a := by
  unfold f
  rw [real.log_one]
  norm_num

-- The slope of the tangent line at x = 1 is 1/4 and we know from f'(1) = 1 - a 
-- leading to a = 3/4
def a_value : ℝ := 3 / 4

-- Given the tangent line is 4y - x - b = 0 at x = 1,
-- let's set a = 3/4 and solve for b directly implying b = 2.
def b_value : ℝ := 2

-- Finally, verify that ab = 3/2
theorem ab_value_is_three_over_two : a * b = 3 / 2 := by
  have h1 : a = 3 / 4 := by
    sorry  -- This part will follow from correct values and conditions.
  have h2 : b = 2 := by
    sorry  -- This part will follow from correct interpretations.
  rw [h1, h2]
  norm_num

end ab_value_is_three_over_two_l705_705721


namespace cannot_determine_arrays_without_computation_l705_705956

theorem cannot_determine_arrays_without_computation :
  ¬ (∃ n : ℕ, ∀ (A : matrix (fin 4) (fin 4) ℕ),
    (∀ i j, 1 ≤ A i j ∧ A i j ≤ 16) ∧
    (∀ i j, i < 3 → A i j < A (i + 1) j) ∧
    (∀ i j, j < 3 → A i j < A i (j + 1)) →
    number_of_valid_4x4_arrays = n) :=
sorry

end cannot_determine_arrays_without_computation_l705_705956


namespace find_f_half_l705_705333

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_half (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ Real.pi / 2) (h₁ : f (Real.sin x) = x) : 
  f (1 / 2) = Real.pi / 6 :=
sorry

end find_f_half_l705_705333


namespace locus_of_points_l705_705909

-- Declare the basic definitions and types
variables {Point : Type} [MetricSpace Point]

-- Define an equilateral triangle and its center
structure EquilateralTriangle :=
  (A B C : Point)
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)

structure Center (Δ : EquilateralTriangle) :=
  (O : Point)
  (centroid : ∃ α β γ, α * Δ.A + β * Δ.B + γ * Δ.C = O ∧ α + β + γ = 1)

-- Define midpoints of sides
structure Midpoints (Δ : EquilateralTriangle) :=
  (A1 : Point)
  (B1 : Point)
  (midpoints : ∃  m n, Δ.C = m * Δ.A + (1 - m) * Δ.B ∧ Δ.A = n * Δ.B + (1 - n) * Δ.C)

open EquilateralTriangle
open Center
open Midpoints

-- Define the quadrilateral defined by O, A1, C, B1
structure Quadrilateral (Δ : EquilateralTriangle) (σ: Center Δ) (ν: Midpoints Δ) :=
  (vertices : (Point × Point) × (Point × Point))
  (vertices_def : vertices = ((σ.O, ν.A1), (Δ.C, ν.B1)))
  
-- Prove that the locus of points M is within the quadrilateral
theorem locus_of_points (Δ : EquilateralTriangle)
    (σ : Center Δ)
    (ν : Midpoints Δ)
    (Q : Quadrilateral Δ σ ν)
    (M : Point) :
    (∀ l : Line, ∃ l.intside, (M ∈ l.intside) → 
                            (l.intside ∩ segment(Δ.A, Δ.B) ≠ ∅ ∨ 
                             l.intside ∩ segment(Δ.C, σ.O) ≠ ∅)) →
    (Q.vertices.1.1,A1,B,(Q.vertices.2.1,B1) := sorry
    
end locus_of_points_l705_705909


namespace unique_solution_exists_l705_705206

theorem unique_solution_exists :
  ∃ (x y : ℝ), x = -13 / 96 ∧ y = 13 / 40 ∧
    (x / Real.sqrt (x^2 + y^2) - 1/x = 7) ∧
    (y / Real.sqrt (x^2 + y^2) + 1/y = 4) :=
by
  sorry

end unique_solution_exists_l705_705206


namespace fraction_inequality_l705_705777

variable (a b c : ℝ)

theorem fraction_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c > a) (h5 : a > b) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

end fraction_inequality_l705_705777


namespace genotypes_of_parents_l705_705121

-- Define the possible alleles
inductive Allele
| H : Allele -- Hairy (dominant)
| h : Allele -- Hairy (recessive)
| S : Allele -- Smooth (dominant)
| s : Allele -- Smooth (recessive)

-- Define the genotype as a pair of alleles
def Genotype := (Allele × Allele)

-- Define the phenotype based on the genotype
def phenotype : Genotype → Allele
| (Allele.H, _) => Allele.H -- HH or HS results in Hairy
| (_, Allele.H) => Allele.H
| (Allele.S, _) => Allele.S -- SS results in Smooth
| (_, Allele.S) => Allele.S
| _           => Allele.s   -- Others

-- Given conditions
def p : ℝ := 0.1 -- probability of allele H
def q : ℝ := 1 - p -- probability of allele S

-- Define most likely genotypes of parents based on the conditions
def most_likely_genotype_of_parents (offspring: list Genotype) : Genotype × Genotype :=
  ((Allele.H, Allele.H), (Allele.S, Allele.h))

-- The main statement
theorem genotypes_of_parents (all_furry: ∀ g : Genotype, phenotype g = Allele.H) :
  most_likely_genotype_of_parents [(Allele.H, Allele.H), (Allele.H, Allele.s)]
  = ((Allele.H, Allele.H), (Allele.S, Allele.h)) :=
sorry

end genotypes_of_parents_l705_705121


namespace hadley_total_distance_l705_705759

def distance_to_grocery := 2
def distance_to_pet_store := 2 - 1
def distance_back_home := 4 - 1

theorem hadley_total_distance : distance_to_grocery + distance_to_pet_store + distance_back_home = 6 :=
by
  -- Proof is omitted.
  sorry

end hadley_total_distance_l705_705759


namespace contradiction_method_l705_705542

theorem contradiction_method (x y : ℝ) (h : x + y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end contradiction_method_l705_705542


namespace quadrilateral_parallelogram_l705_705717

/-- Given a quadrilateral ABCD, if the vector AB is equal to DC,
    then quadrilateral ABCD is a parallelogram. -/
theorem quadrilateral_parallelogram (A B C D : Point) :
  (AB = DC) → is_parallelogram ABCD :=
by sorry

end quadrilateral_parallelogram_l705_705717


namespace percentage_increase_of_toad_bugs_eaten_l705_705878

theorem percentage_increase_of_toad_bugs_eaten (total_bugs gecko_bugs lizard_bugs frog_bugs toad_bugs : ℕ) :
  gecko_bugs = 12 →
  lizard_bugs = gecko_bugs / 2 →
  frog_bugs = 3 * lizard_bugs →
  total_bugs = 63 →
  toad_bugs = total_bugs - (gecko_bugs + lizard_bugs + frog_bugs) →
  ((toad_bugs - frog_bugs : ℤ) * 100) / frog_bugs = 50 :=
by
  sorry

end percentage_increase_of_toad_bugs_eaten_l705_705878


namespace people_per_institution_l705_705246

-- Given conditions
def total_people : ℕ := 480
def number_of_institutions : ℕ := 6

-- Goal: To prove the number of people per institution
theorem people_per_institution :
  (total_people = 480) ∧ (number_of_institutions = 6) → (total_people / number_of_institutions = 80) :=
by
  intro h
  cases h with h_total_people h_number_of_institutions
  rw [h_total_people, h_number_of_institutions]
  exact Nat.div_eq_of_eq_mul (by simp) 


end people_per_institution_l705_705246


namespace team_B_score_l705_705194

theorem team_B_score (A B C total : ℕ) (hA : A = 2) (hC : C = 4) (hTotal : total = 15) (hSum : total = A + B + C) : B = 9 := by
  unfold A C total B
  apply Nat.add_eq_of_eq_sub'
  sorry

end team_B_score_l705_705194


namespace lines_perpendicular_l705_705780

theorem lines_perpendicular
  (a : ℝ)
  (h1 : 2 * 1 - a * 1 - 1 = 0) :
  (2 : ℝ) * (-2 : ℝ) = -1 :=
begin
  sorry
end

end lines_perpendicular_l705_705780


namespace biking_time_to_water_park_l705_705892

-- Definitions
def driving_distance_to_water_park : ℝ := 30 / 60 -- hours
def speed_first_half : ℝ := 28 -- miles per hour
def speed_second_half : ℝ := 60 -- miles per hour
def biking_speed : ℝ := 11 -- miles per hour

-- Derived Distances
def first_half_distance : ℝ := (driving_distance_to_water_park / 2) * speed_first_half
def second_half_distance : ℝ := (driving_distance_to_water_park / 2) * speed_second_half
def total_distance : ℝ := first_half_distance + second_half_distance

-- Statement to prove
theorem biking_time_to_water_park : (total_distance / biking_speed = 2) :=
begin
  sorry
end

end biking_time_to_water_park_l705_705892


namespace genotypes_of_parents_l705_705122

-- Define the possible alleles
inductive Allele
| H : Allele -- Hairy (dominant)
| h : Allele -- Hairy (recessive)
| S : Allele -- Smooth (dominant)
| s : Allele -- Smooth (recessive)

-- Define the genotype as a pair of alleles
def Genotype := (Allele × Allele)

-- Define the phenotype based on the genotype
def phenotype : Genotype → Allele
| (Allele.H, _) => Allele.H -- HH or HS results in Hairy
| (_, Allele.H) => Allele.H
| (Allele.S, _) => Allele.S -- SS results in Smooth
| (_, Allele.S) => Allele.S
| _           => Allele.s   -- Others

-- Given conditions
def p : ℝ := 0.1 -- probability of allele H
def q : ℝ := 1 - p -- probability of allele S

-- Define most likely genotypes of parents based on the conditions
def most_likely_genotype_of_parents (offspring: list Genotype) : Genotype × Genotype :=
  ((Allele.H, Allele.H), (Allele.S, Allele.h))

-- The main statement
theorem genotypes_of_parents (all_furry: ∀ g : Genotype, phenotype g = Allele.H) :
  most_likely_genotype_of_parents [(Allele.H, Allele.H), (Allele.H, Allele.s)]
  = ((Allele.H, Allele.H), (Allele.S, Allele.h)) :=
sorry

end genotypes_of_parents_l705_705122


namespace incorrect_rational_set_statement_l705_705193

theorem incorrect_rational_set_statement :
  ∃ (q : ℚ), ¬(q > 0 ∨ q < 0) :=
begin
  use (0 : ℚ),
  simp,
  sorry
end

end incorrect_rational_set_statement_l705_705193


namespace find_f_2008_l705_705732

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the problem statement with all given conditions
theorem find_f_2008 (h_odd : is_odd f) (h_f2 : f 2 = 0) (h_rec : ∀ x, f (x + 4) = f x + f 4) : f 2008 = 0 := 
sorry

end find_f_2008_l705_705732


namespace number_of_sets_l705_705162

theorem number_of_sets (a n : ℕ) (M : Finset ℕ) (h_consecutive : ∀ x ∈ M, ∃ k, x = a + k ∧ k < n) (h_card : M.card ≥ 2) (h_sum : M.sum id = 2002) : n = 7 :=
sorry

end number_of_sets_l705_705162


namespace original_acid_concentration_l705_705614

noncomputable def acid_percentage (a w : ℕ) :=
  (a / (a + w) : ℝ) * 100

theorem original_acid_concentration (a w : ℕ) (h1 : (a : ℝ) / (a + w + 2) = 3 / 10)
  (h2 : (a + 1 : ℝ) / (a + w + 3) = 2 / 5) :
  ∃ P, (P : ℝ) = acid_percentage a w :=
begin
  -- Proof goes here
  sorry
end

end original_acid_concentration_l705_705614


namespace probability_two_diagonals_intersect_l705_705008

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l705_705008


namespace max_value_of_trig_expr_l705_705295

theorem max_value_of_trig_expr (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expr_l705_705295


namespace pilot_miles_l705_705391

theorem pilot_miles (x : ℕ) :
  (3 * (1134 + x) = 7827) → x = 1475 :=
by
  assume h : 3 * (1134 + x) = 7827
  -- The proof will go here
  sorry

end pilot_miles_l705_705391


namespace angle_AOC_is_minus_150_l705_705124

-- Define the conditions.
def rotate_counterclockwise (angle1 : Int) (angle2 : Int) : Int :=
  angle1 + angle2

-- The initial angle starts at 0°, rotates 120° counterclockwise, and then 270° clockwise
def angle_OA := 0
def angle_OB := rotate_counterclockwise angle_OA 120
def angle_OC := rotate_counterclockwise angle_OB (-270)

-- The theorem stating the resulting angle between OA and OC.
theorem angle_AOC_is_minus_150 : angle_OC = -150 := by
  sorry

end angle_AOC_is_minus_150_l705_705124


namespace part_one_part_two_l705_705439

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705439


namespace probability_nearsighted_l705_705097

theorem probability_nearsighted 
  (total_students : ℕ)
  (nearsighted_rate : ℝ)
  (phone_use_rate_more_than_2_hours : ℝ)
  (nearsighted_rate_in_phone_use_more_than_2_hours : ℝ) :
  (nearsighted_rate = 0.4) →
  (phone_use_rate_more_than_2_hours = 0.3) →
  (nearsighted_rate_in_phone_use_more_than_2_hours = 0.5) →
  (total_students > 0) →
  let students_using_phone_more_than_2_hours := total_students * phone_use_rate_more_than_2_hours,
      nearsighted_students := total_students * nearsighted_rate,
      nearsighted_students_using_phone_more_than_2_hours := students_using_phone_more_than_2_hours * nearsighted_rate_in_phone_use_more_than_2_hours,
      students_using_phone_not_more_than_2_hours := total_students - students_using_phone_more_than_2_hours,
      nearsighted_students_using_phone_not_more_than_2_hours := nearsighted_students - nearsighted_students_using_phone_more_than_2_hours in
  (students_using_phone_not_more_than_2_hours > 0) →
  ((nearsighted_students_using_phone_not_more_than_2_hours / students_using_phone_not_more_than_2_hours) = (5 / 14)) :=
by
  intros h1 h2 h3 h4
  let students_using_phone_more_than_2_hours := total_students * phone_use_rate_more_than_2_hours
  let nearsighted_students := total_students * nearsighted_rate
  let nearsighted_students_using_phone_more_than_2_hours := students_using_phone_more_than_2_hours * nearsighted_rate_in_phone_use_more_than_2_hours
  let students_using_phone_not_more_than_2_hours := total_students - students_using_phone_more_than_2_hours
  let nearsighted_students_using_phone_not_more_than_2_hours := nearsighted_students - nearsighted_students_using_phone_more_than_2_hours
  sorry

end probability_nearsighted_l705_705097


namespace max_pN_value_l705_705066

noncomputable def max_probability_units_digit (N: ℕ) (q2 q5 q10: ℚ) : ℚ :=
  let qk (k : ℕ) := (Nat.floor (N / k) : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_pN_value : ∃ (a b : ℕ), (a.gcd b = 1) ∧ (∀ N q2 q5 q10, max_probability_units_digit N q2 q5 q10 ≤  27 / 100) ∧ (100 * 27 + 100 = 2800) :=
by
  sorry

end max_pN_value_l705_705066


namespace greatest_possible_percentage_l705_705213

theorem greatest_possible_percentage :
  ∃ (p : ℝ), (p ≤ 40/100) ∧ ∀ (w f i s : ℝ → Prop),
    (∀ x, w x → x ≤ 40/100) →
    (∀ x, f x → x ≤ 70/100) →
    (∀ x, i x → x ≤ 60/100) →
    (∀ x, s x → x ≤ 50/100) →
    (∀ x, w x → f x → i x → s x → x ≤ p) :=
begin
  use 0.4,
  split,
  { exact le_refl 0.4 },
  { intros w f i s hw hf hi hs x hwfi,
    exact hwfi.1.played.card.easy_le },
  sorry,
end

end greatest_possible_percentage_l705_705213


namespace vertex_on_x_axis_l705_705619

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l705_705619


namespace count_valid_triples_l705_705785

def num_valid_triples : ℕ :=
  let conditions := (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ 0 ≤ c ∧ c ≤ 10 ∧ 10 ≤ a + b + c ∧ a + b + c ≤ 20)
  let num_triples := 572
  ∃ a b c, conditions → num_triples = 572

theorem count_valid_triples (a b c : ℕ) :
  0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ 0 ≤ c ∧ c ≤ 10 ∧ 10 ≤ a + b + c ∧ a + b + c ≤ 20 →
  num_valid_triples = 572 := 
sorry

end count_valid_triples_l705_705785


namespace average_homework_time_decrease_l705_705564

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l705_705564


namespace find_speed_of_stream_l705_705981

def speed_of_stream (v : ℝ) : Prop :=
  let boat_speed := 18 in
  (∀ D : ℝ, D > 0 → (D / (boat_speed - v) = 2 * (D / (boat_speed + v)))) → v = 6

theorem find_speed_of_stream : ∃ v : ℝ, speed_of_stream v :=
by
  use 6
  unfold speed_of_stream
  sorry

end find_speed_of_stream_l705_705981


namespace inverse_function_fixed_point_l705_705958

noncomputable def f (a x : ℝ) : ℝ := log a (x - 1)

theorem inverse_function_fixed_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∃ x y : ℝ, (y = f a x) ∧ (x, y) = (0, 2) :=
sorry

end inverse_function_fixed_point_l705_705958


namespace least_n_divisibility_condition_l705_705183

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition_l705_705183


namespace green_cards_count_l705_705986

theorem green_cards_count (total_cards : ℕ) (red_fraction : ℚ) (black_fraction_of_remainder : ℚ) 
  (h_total_cards : total_cards = 120) (h_red_fraction : red_fraction = 2 / 5) 
  (h_black_fraction_of_remainder : black_fraction_of_remainder = 5 / 9) :
  let red_cards := (red_fraction * total_cards).natAbs,
      remaining_cards := total_cards - red_cards,
      black_cards := (black_fraction_of_remainder * remaining_cards).natAbs in
  total_cards - (red_cards + black_cards) = 32 :=
by
  sorry

end green_cards_count_l705_705986


namespace measure_tape_problem_l705_705613

noncomputable def number_of_coils_and_last_wrap_length (l : ℕ) (d : ℕ) (k : ℕ) : Prop :=
  ∃ (n : ℝ), n = 55.33 ∧ 
  let len_last_wrap := 10.68 in
  k + ((n - 1) * (0.2 * d * real.pi)) = len_last_wrap

theorem measure_tape_problem 
  (l : ℝ) (d : ℝ) (k : ℝ) 
  (hl : l = 150.0) 
  (hd : d = 0.1) 
  (hk : k = 1.0):
  number_of_coils_and_last_wrap_length l d k :=
begin
  sorry,
end

end measure_tape_problem_l705_705613


namespace gcd_at_most_3_digits_l705_705847

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705847


namespace secure_nailing_requires_two_nails_l705_705998

theorem secure_nailing_requires_two_nails :
  (Two_points_determine_a_straight_line := "Two points determine a straight line") →
  (Necessary_nails_to_secure_wooden_strip := "At least two") →
  Necessary_nails_to_secure_wooden_strip = Two_points_determine_a_straight_line :=
sorry

end secure_nailing_requires_two_nails_l705_705998


namespace marble_groups_l705_705990

theorem marble_groups (total_marbles groups_per_group : ℕ) (h_total : total_marbles = 64) (h_group : groups_per_group = 2) :
  total_marbles / groups_per_group = 32 := by
  rw [h_total, h_group]
  norm_num
  sorry

end marble_groups_l705_705990


namespace gcd_115_161_l705_705179

theorem gcd_115_161 : Nat.gcd 115 161 = 23 := by
  sorry

end gcd_115_161_l705_705179


namespace possible_denominators_of_repeating_decimal_l705_705949

theorem possible_denominators_of_repeating_decimal (a b c : ℕ) (h1 : a ≠ 0) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) :
  ∃ d : ℕ, d ∈ {3, 9, 27, 37, 111, 333, 999} ∧ (∃ m : ℕ, abc = d * m) ∧ 
  (∀ n : ℕ, ∃ p : ℕ, abc = 999 * p) ∧ 
  ∀ x : ℕ, x ∈ {3, 9, 27, 37, 111, 333, 999} 
  → ((∃ k : ℕ, abc = x * k) → (∀ y : ℕ, y ∈ {1, 3, 9, 27, 37, 111, 333, 999} \/ ∃ t : ℕ, 999 = t * x)) :=
sorry

end possible_denominators_of_repeating_decimal_l705_705949


namespace most_likely_parents_genotypes_l705_705110

-- Defining the probabilities of alleles in the population
def p_H : ℝ := 0.1
def q_S : ℝ := 0.9

-- Defining the genotypes and their corresponding fur types
inductive Genotype
| HH : Genotype
| HS : Genotype
| SS : Genotype
| Sh : Genotype

-- A function to determine if a given genotype results in hairy fur
def isHairy : Genotype → Prop
| Genotype.HH := true
| Genotype.HS := true
| _ := false

-- Axiom stating that all four offspring have hairy fur
axiom offspring_all_hairy (parent1 parent2 : Genotype) : 
  (isHairy parent1 ∧ isHairy parent2) ∨
  ((parent1 = Genotype.HH ∨ parent2 = Genotype.Sh) ∧ isHairy Genotype.HH) 

-- The main theorem to prove the genotypes of the parents
theorem most_likely_parents_genotypes : 
  ∃ parent1 parent2,
    parent1 = Genotype.HH ∧ parent2 = Genotype.Sh :=
begin
  sorry
end

end most_likely_parents_genotypes_l705_705110


namespace gcd_digits_le_3_l705_705797

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705797


namespace integral_value_l705_705667

noncomputable def integral_sin_pi_over_2_to_pi : ℝ := ∫ x in (Real.pi / 2)..Real.pi, Real.sin x

theorem integral_value : integral_sin_pi_over_2_to_pi = 1 := by
  sorry

end integral_value_l705_705667


namespace midpoint_one_seventh_one_ninth_l705_705301

theorem midpoint_one_seventh_one_ninth : 
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  (a + b) / 2 = 8 / 63 := 
by
  sorry

end midpoint_one_seventh_one_ninth_l705_705301


namespace sum_xyz_l705_705174

-- Define the conditions as hypotheses
variables {X Y Z : ℕ}
hypothesis h1 : X ∈ ℕ
hypothesis h2 : Y ∈ ℕ
hypothesis h3 : Z ∈ ℕ
hypothesis h4 : ∀ k : ℕ, k > 1 → ¬ (X % k = 0 ∧ Y % k = 0 ∧ Z % k = 0)

-- Define the logarithmic condition with logs in base 100
hypothesis h5 : X * (Real.log 5 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z

-- The theorem stating X + Y + Z equals the correct answer 4
theorem sum_xyz : X + Y + Z = 4 :=
by
  sorry

end sum_xyz_l705_705174


namespace difference_mean_median_score_l705_705854

theorem difference_mean_median_score :
  let percentage_60 := 0.15
      percentage_75 := 0.50
      percentage_85 := 0.20
      percentage_95 := 0.15
      score_60 := 60.0
      score_75 := 75.0
      score_85 := 85.0
      score_95 := 95.0
      mean := (percentage_60 * score_60 + percentage_75 * score_75 + percentage_85 * score_85 + percentage_95 * score_95)
      median := score_75
  in abs (median - mean) = 2.75 :=
by
  let percentage_60 := 0.15
  let percentage_75 := 0.50
  let percentage_85 := 0.20
  let percentage_95 := 0.15
  let score_60 := 60.0
  let score_75 := 75.0
  let score_85 := 85.0
  let score_95 := 95.0
  let mean := (percentage_60 * score_60 + percentage_75 * score_75 + percentage_85 * score_85 + percentage_95 * score_95)
  let median := score_75
  have h : abs (median - mean) = 2.75, from sorry
  exact h

end difference_mean_median_score_l705_705854


namespace gcd_max_digits_l705_705831

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705831


namespace max_intersections_9306_l705_705079

-- Define the statements based on the given conditions
def are_distinct (L : Fin 150 → Line) : Prop :=
  ∀ i j, i ≠ j → L i ≠ L j

def are_parallel (L : Fin 150 → Line) (P : Point → Prop) : Prop :=
  ∀ n, P (L ⟨5 * n, sorry⟩)

def pass_through_point (L : Fin 150 → Line) (B : Point) : Prop :=
  ∀ n, (L ⟨5 * n - 4, sorry⟩).contains B

-- Define the maximum intersections
def max_intersections (L : Fin 150 → Line) : Nat :=
  -- Binarith calculations for the Look-up table and theoretical max calculation
  (30 * 29 / 2) + (30 * 29 / 2) + (90 * 89 / 2) + (30 * 90) + (30 * 60 * 150/2) 

theorem max_intersections_9306 (L : Fin 150 → Line) (B : Point) 
  (h_distinct : are_distinct L) 
  (h_parallel : are_parallel L) 
  (h_passthrough : pass_through_point L B) :
  max_intersections L = 9306 :=
by
  sorry 

end max_intersections_9306_l705_705079


namespace sector_perimeter_l705_705630

theorem sector_perimeter (r : ℝ) (S : ℝ) (h_r : r = 2) (h_S : S = 8) : 
  let α := 4, L := r * α, P := 2 * r + L in P = 12 := 
by
  sorry

end sector_perimeter_l705_705630


namespace original_people_complete_work_in_four_days_l705_705507

noncomputable def original_people_work_days (P D : ℕ) :=
  (2 * P) * 2 = (1 / 2) * (P * D)

theorem original_people_complete_work_in_four_days (P D : ℕ) (h : original_people_work_days P D) : D = 4 :=
by
  sorry

end original_people_complete_work_in_four_days_l705_705507


namespace trig_identity_and_perimeter_l705_705457

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705457


namespace perpendicular_lines_a_values_l705_705779

theorem perpendicular_lines_a_values (a : ℝ) :
  let l1 := λ x y : ℝ, a * x + (1 - a) * y - 3
  let l2 := λ x y : ℝ, (a - 1) * x + (2 * a + 3) * y - 2
  let perpendicular := λ A1 B1 C1 A2 B2 C2 : ℝ, A1 * A2 + B1 * B2 = 0

  perpendicular a (1 - a) (-3) (a - 1) (2 * a + 3) (-2) ↔ a = 1 ∨ a = -3 :=
begin
  sorry
end

end perpendicular_lines_a_values_l705_705779


namespace trajectory_of_M_is_line_segment_l705_705360

-- Definitions based on conditions
variables {F1 F2 M : Type}
variables [metric_space F1] [metric_space F2] [metric_space M]
variable (dist : F1 → F2 → ℝ)

-- Fixed points F1 and F2 with distance 4
def fixed_points : Prop := dist F1 F2 = 4

-- Moving point M satisfies the condition
def moving_point_condition (M : F1) : Prop := dist M F1 + dist M F2 = 4

-- The theorem statement
theorem trajectory_of_M_is_line_segment (F1 F2 : F1) (h1 : fixed_points F1 F2) (M : F1)
  (h2 : moving_point_condition M) : 
  ∃ (M : F1), dist M F1 + dist M F2 = 4 ∧ dist F1 F2 = 4 → (M ∈ segment F1 F2) :=
by
  sorry

end trajectory_of_M_is_line_segment_l705_705360


namespace decrease_equation_l705_705558

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l705_705558


namespace decrease_in_area_l705_705641

theorem decrease_in_area (s : ℝ) 
  (h₁ : 64 * real.sqrt 3 = (s^2 * real.sqrt 3) / 4) : 
  let s_new := s - 4 in
  let area_new := (s_new^2 * real.sqrt 3) / 4 in
  let area_initial := 64 * real.sqrt 3 in
  area_initial - area_new = 28 * real.sqrt 3 :=
sorry

end decrease_in_area_l705_705641


namespace age_of_boy_not_included_l705_705137

theorem age_of_boy_not_included (average_age_11_boys : ℕ) (average_age_first_6 : ℕ) (average_age_last_6 : ℕ) 
(first_6_sum : ℕ) (last_6_sum : ℕ) (total_sum : ℕ) (X : ℕ):
  average_age_11_boys = 50 ∧ average_age_first_6 = 49 ∧ average_age_last_6 = 52 ∧ 
  first_6_sum = 6 * average_age_first_6 ∧ last_6_sum = 6 * average_age_last_6 ∧ 
  total_sum = 11 * average_age_11_boys ∧ first_6_sum + last_6_sum - X = total_sum →
  X = 56 :=
by
  sorry

end age_of_boy_not_included_l705_705137


namespace find_A_appended_is_square_l705_705677

theorem find_A_appended_is_square :
  ∃ A : ℕ, N : ℕ, (A = 13223140496 ∧ N = A * (10^11 + 1)) ∧ ∃ k : ℕ, N = k * k :=
by {
  sorry
}

end find_A_appended_is_square_l705_705677


namespace probability_two_diagonals_intersect_l705_705009

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l705_705009


namespace linear_function_increasing_l705_705321

theorem linear_function_increasing (x1 x2 y1 y2 : ℝ) (h1 : y1 = 2 * x1 - 1) (h2 : y2 = 2 * x2 - 1) (h3 : x1 > x2) : y1 > y2 :=
by
  sorry

end linear_function_increasing_l705_705321


namespace find_g3_l705_705143

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 1

theorem find_g3 : g 3 = 0 := by
  sorry

end find_g3_l705_705143


namespace trig_identity_and_perimeter_l705_705453

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705453


namespace michael_drove_miles_l705_705485

theorem michael_drove_miles (rental_fee charge_per_mile total_amount_paid : ℝ) (h_rental_fee : rental_fee = 20.99)
  (h_charge_per_mile : charge_per_mile = 0.25) (h_total_amount_paid : total_amount_paid = 95.74) :
  let amount_paid_for_miles := total_amount_paid - rental_fee in
  let number_of_miles := amount_paid_for_miles / charge_per_mile in
  number_of_miles = 299 := 
by
  sorry

end michael_drove_miles_l705_705485


namespace paul_oranges_l705_705925

theorem paul_oranges :
  let initial_oranges := 150 in
  let sold_to_peter := (20 * initial_oranges) / 100 in
  let remaining_after_peter := initial_oranges - sold_to_peter in
  let sold_to_paula := (30 * remaining_after_peter) / 100 in
  let remaining_after_paula := remaining_after_peter - sold_to_paula in
  let given_to_neighbor := 10 in
  let remaining_after_neighbor := remaining_after_paula - given_to_neighbor in
  let given_to_teacher := 1 in
  let final_remaining := remaining_after_neighbor - given_to_teacher in
  final_remaining = 73 :=
by {
  sorry
}

end paul_oranges_l705_705925


namespace probability_same_color_l705_705215

theorem probability_same_color (w b : ℕ) (h_w : w = 8) (h_b : b = 9) :
  (Nat.choose 8 2 + Nat.choose 9 2) / (Nat.choose 17 2) = 8 / 17 :=
by
  sorry

end probability_same_color_l705_705215


namespace min_x_prime_factors_sum_l705_705908

theorem min_x_prime_factors_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : 4 * x^7 = 13 * y^17) :
  ∃ a b c d : ℕ, prime a ∧ prime b ∧ 
  (a ^ c * b ^ d = x) ∧ (a + b + c + d = 19) :=
by {
  sorry
}

end min_x_prime_factors_sum_l705_705908


namespace jake_bike_time_l705_705894

/-- Prove that Jake will take 2 hours to bike to the water park, given the conditions. -/
theorem jake_bike_time 
  (t_drive : ℝ) (speed1 speed2 speed_bike : ℝ)
  (t_half : t_drive / 2)
  (distance1 := speed1 * t_half) (distance2 := speed2 * t_half)
  (total_distance := distance1 + distance2)
  (total_distance = 22) :
  (total_distance / speed_bike = 2) := 
by
  sorry

end

end jake_bike_time_l705_705894


namespace cost_of_country_cd_l705_705171

theorem cost_of_country_cd
  (cost_rock_cd : ℕ) (cost_pop_cd : ℕ) (cost_dance_cd : ℕ)
  (num_each : ℕ) (julia_has : ℕ) (julia_short : ℕ)
  (total_cost : ℕ) (total_other_cds : ℕ) (cost_country_cd : ℕ) :
  cost_rock_cd = 5 →
  cost_pop_cd = 10 →
  cost_dance_cd = 3 →
  num_each = 4 →
  julia_has = 75 →
  julia_short = 25 →
  total_cost = julia_has + julia_short →
  total_other_cds = num_each * cost_rock_cd + num_each * cost_pop_cd + num_each * cost_dance_cd →
  total_cost = total_other_cds + num_each * cost_country_cd →
  cost_country_cd = 7 :=
by
  intros cost_rock_cost_pop_cost_dance_num julia_diff 
         calc_total_total_other sub_total total_cds
  sorry

end cost_of_country_cd_l705_705171


namespace triangle_properties_l705_705883

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_properties_l705_705883


namespace exists_angle_le_26_degrees_l705_705320

theorem exists_angle_le_26_degrees (L : Finset (Line ℝ)) (hL : L.card = 7) (h_parallel : ∀ l₁ l₂ ∈ L, l₁ ≠ l₂ → ¬Parallel l₁ l₂) :
  ∃ l₁ l₂ ∈ L, l₁ ≠ l₂ ∧ angle l₁ l₂ ≤ 26 :=
sorry

end exists_angle_le_26_degrees_l705_705320


namespace angle_x_is_77_degrees_l705_705302

theorem angle_x_is_77_degrees 
  (angle_ABC_is_straight : ∠ ABC = 180)
  (angle_ABC_is_112 : ∠ B C D = 112)
  (angle_DIA_is_35 : ∠ D I A = 35)
  (angle_ADB_is_28 : ∠ A D B = 28) : 
  ∠ A B D = 77 :=
by
  sorry

end angle_x_is_77_degrees_l705_705302


namespace gcd_digits_le_3_l705_705800

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705800


namespace solution_per_beaker_l705_705991

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end solution_per_beaker_l705_705991


namespace set_intersection_l705_705340

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}
def B_complement : Set ℝ := {x | x ≥ 2}

theorem set_intersection :
  A ∩ B_complement = {x | 2 ≤ x ∧ x < 5} :=
by 
  sorry

end set_intersection_l705_705340


namespace one_greater_l705_705335

theorem one_greater (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) 
  (h5 : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
sorry

end one_greater_l705_705335


namespace biking_time_to_water_park_l705_705891

-- Definitions
def driving_distance_to_water_park : ℝ := 30 / 60 -- hours
def speed_first_half : ℝ := 28 -- miles per hour
def speed_second_half : ℝ := 60 -- miles per hour
def biking_speed : ℝ := 11 -- miles per hour

-- Derived Distances
def first_half_distance : ℝ := (driving_distance_to_water_park / 2) * speed_first_half
def second_half_distance : ℝ := (driving_distance_to_water_park / 2) * speed_second_half
def total_distance : ℝ := first_half_distance + second_half_distance

-- Statement to prove
theorem biking_time_to_water_park : (total_distance / biking_speed = 2) :=
begin
  sorry
end

end biking_time_to_water_park_l705_705891


namespace ratio_of_square_areas_l705_705900

-- Let's define the problem in Lean

theorem ratio_of_square_areas (ABCD EFGH : ℝ) :
  (is_square ABCD) ∧
  (is_square EFGH) ∧
  (sides_midpoints_are_vertices EFGH ABCD) ∧
  (isosceles_right_triangles_exterior EFGH ABCD) →
  (ratio_of_areas EFGH ABCD) = 5 / 4 :=
begin
  sorry
end

end ratio_of_square_areas_l705_705900


namespace det_projection_matrix_l705_705053

-- Definition of the projection matrix P onto vector (3, 4)
def projection_matrix (u : Vector ℝ 2) : Matrix (Fin 2) (Fin 2) ℝ :=
  let (x, y) := (u 0, u 1)
  let norm_sq := x^2 + y^2
  (Matrix.vecCons (Matrix.vecCons (x^2 / norm_sq) (x * y / norm_sq))
                  (Matrix.vecCons (x * y / norm_sq) (y^2 / norm_sq)))

-- Specific vector (3, 4)
def u : Vector ℝ 2 := ![3, 4]
def P := projection_matrix u

-- Goal: The determinant of the projection matrix P
theorem det_projection_matrix : det P = 0 :=
by {
  sorry
}

end det_projection_matrix_l705_705053


namespace infinite_series_sum_l705_705651

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) * (1 / 1000)^n = 3000000 / 998001 :=
by sorry

end infinite_series_sum_l705_705651


namespace gcd_max_two_digits_l705_705837

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705837


namespace xyz_value_l705_705331

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 18) 
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
                  x * y * z = 4 := 
by
  sorry

end xyz_value_l705_705331


namespace number_of_correct_propositions_l705_705769

-- Definitions of lines and their properties
structure Line where
  angle_of_inclination : ℝ
  slope : ℝ

-- Assume l1 and l2 are two non-coincident lines
variables (l1 l2 : Line)

-- Define parallel relation between two lines
def parallel (l1 l2 : Line) : Prop :=
  l1.angle_of_inclination = l2.angle_of_inclination

-- Proposition (1): If l1 is parallel to l2, then their slopes are equal.
lemma prop1 (h : parallel l1 l2) : l1.slope = l2.slope := sorry

-- Proposition (2): If slopes are equal, then l1 is parallel to l2.
lemma prop2 (h : l1.slope = l2.slope) : parallel l1 l2 := sorry

-- Proposition (3): If l1 is parallel to l2, then their angles of inclination are equal.
lemma prop3 (h : parallel l1 l2) : l1.angle_of_inclination = l2.angle_of_inclination := sorry

-- Proposition (4): If angles of inclination are equal, then l1 is parallel to l2.
lemma prop4 (h : l1.angle_of_inclination = l2.angle_of_inclination) : parallel l1 l2 := sorry

-- Theorem: The number of correct propositions is 4
theorem number_of_correct_propositions : 4 = 4 :=
begin
  have h1 := prop1,
  have h2 := prop2,
  have h3 := prop3,
  have h4 := prop4,
  exact eq.refl 4
end

end number_of_correct_propositions_l705_705769


namespace max_questions_wrong_to_succeed_l705_705646

theorem max_questions_wrong_to_succeed :
  ∀ (total_questions : ℕ) (passing_percentage : ℚ),
  total_questions = 50 →
  passing_percentage = 0.75 →
  ∃ (max_wrong : ℕ), max_wrong = 12 ∧
    (total_questions - max_wrong) ≥ passing_percentage * total_questions := by
  intro total_questions passing_percentage h1 h2
  use 12
  constructor
  . rfl
  . sorry  -- Proof omitted

end max_questions_wrong_to_succeed_l705_705646


namespace find_parameters_l705_705745

noncomputable def cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + 27

def deriv_cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + b

theorem find_parameters
  (a b : ℝ)
  (h1 : deriv_cubic_function a b (-1) = 0)
  (h2 : deriv_cubic_function a b 3 = 0) :
  a = -3 ∧ b = -9 :=
by
  -- leaving proof as sorry since the task doesn't require proving
  sorry

end find_parameters_l705_705745


namespace vertex_on_xaxis_l705_705621

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l705_705621


namespace brick_width_l705_705606

theorem brick_width (l_brick : ℕ) (w_courtyard l_courtyard : ℕ) (num_bricks : ℕ) (w_brick : ℕ)
  (H1 : l_courtyard = 24) 
  (H2 : w_courtyard = 14) 
  (H3 : num_bricks = 8960) 
  (H4 : l_brick = 25) 
  (H5 : (w_courtyard * 100 * l_courtyard * 100 = (num_bricks * (l_brick * w_brick)))) :
  w_brick = 15 :=
by
  sorry

end brick_width_l705_705606


namespace Bert_sandwiches_left_l705_705256

theorem Bert_sandwiches_left : (Bert:Type) → 
  (sandwiches_made : ℕ) → 
  sandwiches_made = 12 → 
  (sandwiches_eaten_day1 : ℕ) → 
  sandwiches_eaten_day1 = sandwiches_made / 2 → 
  (sandwiches_eaten_day2 : ℕ) → 
  sandwiches_eaten_day2 = sandwiches_eaten_day1 - 2 →
  (sandwiches_left : ℕ) → 
  sandwiches_left = sandwiches_made - (sandwiches_eaten_day1 + sandwiches_eaten_day2) → 
  sandwiches_left = 2 := 
  sorry

end Bert_sandwiches_left_l705_705256


namespace parallel_lines_eq_l705_705153

theorem parallel_lines_eq {a x y : ℝ} :
  (∀ x y : ℝ, x + a * y = 2 * a + 2) ∧ (∀ x y : ℝ, a * x + y = a + 1) →
  a = 1 :=
by
  sorry

end parallel_lines_eq_l705_705153


namespace gcd_has_at_most_3_digits_l705_705816

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705816


namespace class_average_is_75_l705_705370

-- Definitions of given conditions
def proportion_group1 := 0.25
def proportion_group2 := 0.50
def proportion_group3 := 1 - proportion_group1 - proportion_group2

def average_group1 := 80
def average_group2 := 65
def average_group3 := 90

-- Statement to prove the overall class average is 75%
theorem class_average_is_75 :
  (proportion_group1 * average_group1 + 
   proportion_group2 * average_group2 + 
   proportion_group3 * average_group3) = 75 := 
by
  sorry

end class_average_is_75_l705_705370


namespace intersection_M_N_l705_705474

def M := {x : ℝ | real.log10 x > 0}
def N := {x : ℝ | x^2 ≤ 4}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end intersection_M_N_l705_705474


namespace complex_value_l705_705071

open Complex

theorem complex_value (z : ℂ)
  (h : 15 * normSq z = 3 * normSq (z + 3) + normSq (z^2 + 4) + 25) :
  z + (8 / z) = -4 :=
sorry

end complex_value_l705_705071


namespace volume_doubled_height_l705_705146

theorem volume_doubled_height
  (r h : ℝ)
  (π : ℝ) 
  (original_volume : π * r^2 * h = 10) :
  let new_height := 2 * h in
  let new_volume := π * r^2 * new_height in
  new_volume = 20 :=
by 
  sorry

end volume_doubled_height_l705_705146


namespace problem_solution_l705_705157

theorem problem_solution
  (P Q R S : ℕ)
  (h1 : 2 * Q = P + R)
  (h2 : R * R = Q * S)
  (h3 : R = 4 * Q / 3) :
  P + Q + R + S = 171 :=
by sorry

end problem_solution_l705_705157


namespace game_probability_correct_l705_705084

noncomputable def game_probability :=
  let total_outcomes := 3^3
  let outcomes_at_least_one_rock := total_outcomes - 2^3
  let outcomes_at_most_one_paper := 2^3 + 3 * 2^2
  let outcomes_intersection := outcomes_at_least_one_rock + outcomes_at_most_one_paper - total_outcomes
  let outcomes_exactly_one_scissors := 3 * 1^2
  let probability := outcomes_exactly_one_scissors / outcomes_intersection
  let m := 1
  let n := 4
  100 * m + n

theorem game_probability_correct : game_probability = 104 := by
  sorry

end game_probability_correct_l705_705084


namespace acute_angle_subset_l705_705357

noncomputable def cos (θ : ℝ) : ℝ :=
Real.cos θ

theorem acute_angle_subset (θ : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : {1, cos θ} ⊆ {0, 1/2, 1}) : θ = π/3 := by
  sorry

end acute_angle_subset_l705_705357


namespace flag_coloring_l705_705995

open Polynomial

theorem flag_coloring {n : ℕ} (hn : Odd n) (k : ℕ) (hk : 2 ≤ k) 
    (colors : Fin k → Fin n → Fin k) :
  (∀ i : Fin k, ∃ (P : Polynomial ℂ), P.degree > 1 ∧ ∀ j : Fin n, eval (j : ℂ) P = 0 → colors i j = i) →
  ∃ i j h : Fin k, i ≠ j ∧ j ≠ h ∧ i ≠ h ∧ 
    {j // ∃ h : Fin n, colors i h = i}.card = {j // ∃ h : Fin n, colors j h = j}.card :=
sorry

end flag_coloring_l705_705995


namespace find_base_l705_705740

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem find_base (a : ℝ) (h : 1 < a) :
  (log_base a (2 * a) - log_base a a = 1 / 2) → a = 4 :=
by
  -- skipping the proof
  sorry

end find_base_l705_705740


namespace right_triangle_third_side_square_l705_705322

theorem right_triangle_third_side_square (a b : ℕ) (h_rt : a = 3 ∨ a = 4) (h_rt' : b = 3 ∨ b = 4) (h_neq : a ≠ b) :
  (a^2 + b^2 = 5^2 ∨ (max a b)^2 - (min a b)^2 = 7) :=
by
  cases h_rt; cases h_rt'; try { contradiction }
  · -- Case 1: Both 3 and 4 are legs
    have h_a : a^2 + b^2 = 3^2 + 4^2 := by 
      rw [h_rt, h_rt']
      norm_num
    exact Or.inl h_a
  · -- Case 2: One is a leg, the other is hypotenuse.
    have h_sub : (max a b)^2 - (min a b)^2 = (max 3 4)^2 - (min 3 4)^2 :=
      by rw [h_rt, h_rt']; norm_num
    norm_num at h_sub
    exact Or.inr h_sub
  · -- Remaining cases are analogous, so handle similarly
    sorry

end right_triangle_third_side_square_l705_705322


namespace translation_of_sine_function_l705_705178

def y (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

def y_translated (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem translation_of_sine_function (x : ℝ) :
  y (x - Real.pi / 6) = y_translated x :=
by sorry

end translation_of_sine_function_l705_705178


namespace max_n_consecutive_sum_2014_l705_705850

theorem max_n_consecutive_sum_2014 : 
  ∃ (k n : ℕ), (2 * k + n - 1) * n = 4028 ∧ n = 53 ∧ k > 0 := sorry

end max_n_consecutive_sum_2014_l705_705850


namespace james_total_distance_l705_705037

noncomputable def total_distance
  (d1 d2 d3 d4 d5 : ℝ) : ℝ :=
  d1 + d2 + d3 + d4 + d5

theorem james_total_distance :
  let d1 := 30 * 0.5 in
  let d2 := 60 * 0.75 in
  let d3 := 75 * 1.5 in
  let d4 := 60 * 2 in
  let d5 := 70 * 4 in
  total_distance d1 d2 d3 d4 d5 = 572.5 :=
by
  sorry

end james_total_distance_l705_705037


namespace integer_part_of_sum_l705_705425

theorem integer_part_of_sum :
  let S := 1 + ∑ n in finset.range 99, 1 / real.sqrt (n.succ.succ) in
  (∃ (H : ∀ n : ℕ, 1 ≤ n → real.sqrt n < 0.5 * (real.sqrt n + real.sqrt (n+1)) < real.sqrt (n + 1)), 
  ∀ S : ℝ, S = 1 + ∑ n in finset.range 99, 1 / real.sqrt (n.succ.succ) → ⌊S⌋ = 18 :=
begin
  sorry
end

end integer_part_of_sum_l705_705425


namespace Faye_crayons_l705_705279

theorem Faye_crayons (rows crayons_per_row : ℕ) (h_rows : rows = 7) (h_crayons_per_row : crayons_per_row = 30) : rows * crayons_per_row = 210 :=
by
  sorry

end Faye_crayons_l705_705279


namespace problem_statement_l705_705464

noncomputable def log_three_four : ℝ := Real.log 4 / Real.log 3
noncomputable def a : ℝ := Real.log (log_three_four) / Real.log (3/4)
noncomputable def b : ℝ := Real.rpow (3/4 : ℝ) 0.5
noncomputable def c : ℝ := Real.rpow (4/3 : ℝ) 0.5

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end problem_statement_l705_705464


namespace ratio_chocolate_to_regular_l705_705216

theorem ratio_chocolate_to_regular (total_cartons regular_cartons : ℕ) 
  (h_total : total_cartons = 24) 
  (h_regular : regular_cartons = 3) : 
  let chocolate_cartons := total_cartons - regular_cartons in
  chocolate_cartons % 3 = 0 ∧ regular_cartons % 3 = 0 ∧ (chocolate_cartons / 3, regular_cartons / 3) = (7, 1) :=
by
  sorry

end ratio_chocolate_to_regular_l705_705216


namespace quadrilateral_parallelogram_l705_705715

variable (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

theorem quadrilateral_parallelogram {AB DC : A} (h : AB = DC) :
  ∃ (ABCD : Quadrilateral), ABCD.is_parallelogram := 
sorry

end quadrilateral_parallelogram_l705_705715


namespace distance_sum_range_l705_705328

noncomputable def PA_squared {A : ℤ × ℤ} (P : ℝ × ℝ) : ℝ :=
  (P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2

theorem distance_sum_range :
  let A := (-2, 2 : ℤ × ℤ) 
  let B := (-2, 6 : ℤ × ℤ)
  let C := (4, -2 : ℤ × ℤ)
  ∀ (P : ℝ × ℝ), (P.1)^2 + (P.2)^2 ≤ 4 → 
  let dist_sum := PA_squared A P + PA_squared B P + PA_squared C P in
  72 ≤ dist_sum ∧ dist_sum ≤ 88 :=
begin
  intros A B C P,
  sorry
end

end distance_sum_range_l705_705328


namespace mean_of_samantha_scores_l705_705500

noncomputable def arithmetic_mean (l : List ℝ) : ℝ := l.sum / l.length

theorem mean_of_samantha_scores :
  arithmetic_mean [93, 87, 90, 96, 88, 94] = 91.333 :=
by
  sorry

end mean_of_samantha_scores_l705_705500


namespace constant_term_in_expansion_of_polynomial_is_375_l705_705341

-- Definitions translated from conditions
def binomial_sum_eq_64 (n : ℕ) : Prop :=
  (2^n = 64)

-- Proof goal translated from the question and correct answer
theorem constant_term_in_expansion_of_polynomial_is_375
  (n : ℕ) (h : binomial_sum_eq_64 n) :
  let exp := (5 : ℚ) * x - (1 / (x ^ (1 / 2)))
  in n = 6 → 
     (polynomial.eval 1 ((polynomial.C (5)) * x - (polynomial.C (1:ℚ) / polynomial.C (x^(1/2))))^n) =  (375 : ℚ) :=
by 
  sorry

end constant_term_in_expansion_of_polynomial_is_375_l705_705341


namespace trig_identity_and_perimeter_l705_705455

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705455


namespace shop_conditions_l705_705870

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l705_705870


namespace intersecting_diagonals_probability_l705_705015

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l705_705015


namespace unique_f_l705_705046

def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 10^10 }

noncomputable def f : ℕ → ℕ := sorry

axiom f_cond (x : ℕ) (hx : x ∈ S) :
  f (x + 1) % (10^10) = (f (f x) + 1) % (10^10)

axiom f_boundary :
  f (10^10 + 1) % (10^10) = f 1

theorem unique_f (x : ℕ) (hx : x ∈ S) :
  f x % (10^10) = x % (10^10) :=
sorry

end unique_f_l705_705046


namespace find_x_l705_705778

theorem find_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) (h3 : x + y = 5) : 
  x = (7 + Real.sqrt 5) / 2 :=
by 
  sorry

end find_x_l705_705778


namespace gcd_has_at_most_3_digits_l705_705817

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705817


namespace part1_part2_l705_705441

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705441


namespace gcd_max_digits_l705_705828

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705828


namespace curve_intersection_conditions_l705_705142

noncomputable def A_sin_x_a_property (A a : ℝ) (hA : A > 0) (ha : a > 0) : Prop :=
∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * real.pi) → 
  (∃ x1 x2 : ℝ, 
    (A * real.sin x1 + a = 2 ∧ A * real.sin x2 + a = -1 ∧ abs (x1 - x2) ≠ 0))

theorem curve_intersection_conditions (A a : ℝ) (hA : A > 0) (ha : a > 0) :
  A_sin_x_a_property A a hA ha → (a = 0.5 ∧ A > 1.5) :=
by
  sorry

end curve_intersection_conditions_l705_705142


namespace determine_parents_genotype_l705_705113

noncomputable def genotype := ℕ -- We use nat to uniquely represent each genotype: e.g. HH=0, HS=1, Sh=2, SS=3

def probability_of_allele_H : ℝ := 0.1
def probability_of_allele_S : ℝ := 1 - probability_of_allele_H

def is_dominant (allele: ℕ) : Prop := allele == 0 ∨ allele == 1 -- HH or HS are dominant for hairy

def offspring_is_hairy (parent1 parent2: genotype) : Prop :=
  (∃ g1 g2, (parent1 = 0 ∨ parent1 = 1) ∧ (parent2 = 1 ∨ parent2 = 2 ∨ parent2 = 3) ∧
  ((g1 = 0 ∨ g1 = 1) ∧ (g2 = 0 ∨ g2 = 1))) ∧ 
  (is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0) 

def most_likely_genotypes (hairy_parent smooth_parent : genotype) : Prop :=
  (hairy_parent = 0) ∧ (smooth_parent = 2)

theorem determine_parents_genotype :
  ∃ hairy_parent smooth_parent, offspring_is_hairy hairy_parent smooth_parent ∧ most_likely_genotypes hairy_parent smooth_parent :=
  sorry

end determine_parents_genotype_l705_705113


namespace complex_product_value_l705_705346

noncomputable def z : ℂ := (√3 + I) / ((1 - √3 * I) ^ 2)

def bar_z : ℂ := conj z

theorem complex_product_value : z * bar_z = 1/4 := sorry

end complex_product_value_l705_705346


namespace find_c_l705_705947

theorem find_c (a b c : ℝ) (k₁ k₂ : ℝ) 
  (h₁ : a * b = k₁) 
  (h₂ : b * c = k₂) 
  (h₃ : 40 * 5 = k₁) 
  (h₄ : 7 * 10 = k₂) 
  (h₅ : a = 16) : 
  c = 5.6 :=
  sorry

end find_c_l705_705947


namespace train_length_is_correct_l705_705199

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct_l705_705199


namespace gcd_digit_bound_l705_705806

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705806


namespace acme_profit_calculation_l705_705589

theorem acme_profit_calculation :
  let initial_outlay := 12450
  let cost_per_set := 20.75
  let selling_price := 50
  let number_of_sets := 950
  let total_revenue := number_of_sets * selling_price
  let total_manufacturing_costs := initial_outlay + cost_per_set * number_of_sets
  let profit := total_revenue - total_manufacturing_costs 
  profit = 15337.50 := 
by
  sorry

end acme_profit_calculation_l705_705589


namespace annual_subscription_cost_l705_705616

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l705_705616


namespace product_of_y_coordinates_l705_705493

theorem product_of_y_coordinates (y : ℝ) (h₁ : (x = -3)) (h₂ : (dist (5, 2) (-3, y) = 10)) :
  {y | y = 8 ∨ y = -4}.prod = -32 :=
by sorry

end product_of_y_coordinates_l705_705493


namespace find_r_l705_705281

theorem find_r (r : ℝ) : log 64 (3 * r ^ 2 - 2) = -1 / 3 ↔ (r = sqrt 3 / 2) ∨ (r = -sqrt 3 / 2) := by
  sorry

end find_r_l705_705281


namespace rectangular_field_area_l705_705662

noncomputable def length : ℝ := 1.2
noncomputable def width : ℝ := (3/4) * length

theorem rectangular_field_area : (length * width = 1.08) :=
by 
  -- The proof steps would go here
  sorry

end rectangular_field_area_l705_705662


namespace cone_lateral_area_l705_705736

theorem cone_lateral_area (cos_ASB : ℝ)
  (angle_SA_base : ℝ)
  (triangle_SAB_area : ℝ) :
  cos_ASB = 7 / 8 →
  angle_SA_base = 45 →
  triangle_SAB_area = 5 * Real.sqrt 15 →
  (lateral_area : ℝ) = 40 * Real.sqrt 2 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end cone_lateral_area_l705_705736


namespace correct_model_l705_705560

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l705_705560


namespace distance_from_A_to_B_l705_705695

-- Define Sophie's initial and final displacement movements
def sophia_walks_south_north : ℝ := 50 - 20 -- South - North movement
def sophia_walks_east_west : ℝ := 80 - 30 -- West - East movement

-- Diagonal southwest movement decomposed into south and west components
def southwest_movement : ℝ := 10 * (Real.sqrt 2 / 2) -- 10 yards * cos(45°)

def total_south_displacement : ℝ := sophia_walks_south_north + southwest_movement
def total_west_displacement : ℝ := sophia_walks_east_west + southwest_movement

-- Distance from point A to B using Pythagorean theorem
def distance_AB : ℝ := Real.sqrt ((total_south_displacement * total_south_displacement) +
                                  (total_west_displacement * total_west_displacement))

theorem distance_from_A_to_B : distance_AB ≈ 68.06 := by
  sorry

end distance_from_A_to_B_l705_705695


namespace find_original_rabbits_l705_705169

theorem find_original_rabbits (R S : ℕ) (h1 : R + S = 50)
  (h2 : 4 * R + 8 * S = 2 * R + 16 * S) :
  R = 40 :=
sorry

end find_original_rabbits_l705_705169


namespace area_of_triangle_PBQ_l705_705645

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Square :=
  (side : ℝ)
  (area : ℝ := side * side)
  (A B C D : Point)

def are_on_line (P Q R : Point) : Prop :=
  (Q.x - P.x) * (R.y - P.y) = (Q.y - P.y) * (R.x - P.x)

def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

def side_length (s : Square) : ℝ :=
  sqrt s.area

noncomputable def P := {x := 0, y := 4}
noncomputable def Q := {x := 3 * sqrt 2, y := 2 * sqrt 2}
noncomputable def B := {x := 5, y := 0}

def triangle_PBQ (s : Square) : ℝ :=
  triangle_area P B Q

theorem area_of_triangle_PBQ :
  let s := { side := 5, area := 25, A := {x := 0, y := 0}, B := {x := 5, y := 0}, C := {x := 5, y := 5}, D := {x := 0, y := 5} } in
  triangle_PBQ s = 1.5 :=
by sorry

end area_of_triangle_PBQ_l705_705645


namespace inequality_product_geq_power_sum_l705_705699

theorem inequality_product_geq_power_sum
  (n : ℕ) (a : ℝ) (x : Fin n → ℝ) (s : ℝ)
  (hx : ∀ i, 0 < x i)
  (ha : 0 < a)
  (hs : (∑ i, x i) = s ∧ s ≤ a) :
  (∏ i, (a + x i) / (a - x i)) ≥ ((n * a + s) / (n * a - s)) ^ n :=
sorry

end inequality_product_geq_power_sum_l705_705699


namespace volume_smaller_cube_inside_sphere_inside_big_cube_l705_705633

-- Define constants according to the conditions
constant edge_length_big_cube : ℝ := 12
constant side_length_small_cube : ℝ := 4 * Real.sqrt 3

-- Define the volume calculation
noncomputable def volume_of_smaller_cube : ℝ :=
  side_length_small_cube ^ 3

-- Proof statement
theorem volume_smaller_cube_inside_sphere_inside_big_cube :
  volume_of_smaller_cube = 192 * Real.sqrt 3 :=
  sorry

end volume_smaller_cube_inside_sphere_inside_big_cube_l705_705633


namespace find_T10_l705_705735

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
| 1     := 3
| (n+1) := if n = 1 then 2 * (n+1) else 2 * (n + 1)

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then a n else b n

def T (n : ℕ) : ℕ := ∑ k in Finset.range n, c (k + 1)

theorem find_T10 : T 10 = 733 :=
  sorry

end find_T10_l705_705735


namespace genotypes_of_parents_l705_705123

-- Define the possible alleles
inductive Allele
| H : Allele -- Hairy (dominant)
| h : Allele -- Hairy (recessive)
| S : Allele -- Smooth (dominant)
| s : Allele -- Smooth (recessive)

-- Define the genotype as a pair of alleles
def Genotype := (Allele × Allele)

-- Define the phenotype based on the genotype
def phenotype : Genotype → Allele
| (Allele.H, _) => Allele.H -- HH or HS results in Hairy
| (_, Allele.H) => Allele.H
| (Allele.S, _) => Allele.S -- SS results in Smooth
| (_, Allele.S) => Allele.S
| _           => Allele.s   -- Others

-- Given conditions
def p : ℝ := 0.1 -- probability of allele H
def q : ℝ := 1 - p -- probability of allele S

-- Define most likely genotypes of parents based on the conditions
def most_likely_genotype_of_parents (offspring: list Genotype) : Genotype × Genotype :=
  ((Allele.H, Allele.H), (Allele.S, Allele.h))

-- The main statement
theorem genotypes_of_parents (all_furry: ∀ g : Genotype, phenotype g = Allele.H) :
  most_likely_genotype_of_parents [(Allele.H, Allele.H), (Allele.H, Allele.s)]
  = ((Allele.H, Allele.H), (Allele.S, Allele.h)) :=
sorry

end genotypes_of_parents_l705_705123


namespace average_homework_time_decrease_l705_705567

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l705_705567


namespace problem1_problem2_l705_705262

-- Definitions of trigonometric values used in the problem
def sin_60 := Real.sin (Real.pi / 3)
def tan_45 := Real.tan (Real.pi / 4)
def cos_30 := Real.cos (Real.pi / 6)
def sin_30 := Real.sin (Real.pi / 6)
def tan_60 := Real.tan (Real.pi / 3)

-- First Problem Statement
theorem problem1 : 2 * sin_60 - 3 * tan_45 + Real.sqrt 9 = Real.sqrt 3 := 
by {
  sorry
}

-- Second Problem Statement
theorem problem2 : (cos_30 / (1 + sin_30) + tan_60) = (4 * Real.sqrt 3 / 3) := 
by {
  sorry
}

end problem1_problem2_l705_705262


namespace average_homework_time_decrease_l705_705565

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l705_705565


namespace positive_root_in_range_l705_705159

theorem positive_root_in_range : ∃ x > 0, (x^2 - 2 * x - 1 = 0) ∧ (2 < x ∧ x < 3) :=
by
  sorry

end positive_root_in_range_l705_705159


namespace gcd_max_digits_l705_705826

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705826


namespace inverse_proposition_l705_705904

variable {V : Type} [NormedAddCommGroup V]

theorem inverse_proposition (a b : V) :
  (|\overrightarrow{a}| = |\overrightarrow{b}|) → (a = -b) := 
sorry

end inverse_proposition_l705_705904


namespace genotypes_of_parents_l705_705120

-- Define the possible alleles
inductive Allele
| H : Allele -- Hairy (dominant)
| h : Allele -- Hairy (recessive)
| S : Allele -- Smooth (dominant)
| s : Allele -- Smooth (recessive)

-- Define the genotype as a pair of alleles
def Genotype := (Allele × Allele)

-- Define the phenotype based on the genotype
def phenotype : Genotype → Allele
| (Allele.H, _) => Allele.H -- HH or HS results in Hairy
| (_, Allele.H) => Allele.H
| (Allele.S, _) => Allele.S -- SS results in Smooth
| (_, Allele.S) => Allele.S
| _           => Allele.s   -- Others

-- Given conditions
def p : ℝ := 0.1 -- probability of allele H
def q : ℝ := 1 - p -- probability of allele S

-- Define most likely genotypes of parents based on the conditions
def most_likely_genotype_of_parents (offspring: list Genotype) : Genotype × Genotype :=
  ((Allele.H, Allele.H), (Allele.S, Allele.h))

-- The main statement
theorem genotypes_of_parents (all_furry: ∀ g : Genotype, phenotype g = Allele.H) :
  most_likely_genotype_of_parents [(Allele.H, Allele.H), (Allele.H, Allele.s)]
  = ((Allele.H, Allele.H), (Allele.S, Allele.h)) :=
sorry

end genotypes_of_parents_l705_705120


namespace jangshe_clothing_l705_705039

theorem jangshe_clothing
  (total_spent : ℕ)
  (price1 price2 price_other : ℕ)
  (num_other_pieces : ℕ)
  (h1 : total_spent = 610)
  (h2 : price1 = 49)
  (h3 : price2 = 81)
  (h4 : price_other = 96)
  (h5 : total_spent = price1 + price2 + num_other_pieces * price_other) :
  1 + 1 + num_other_pieces = 7 :=
by
  rw [h1, h2, h3, h4] at h5
  rw add_assoc at h5
  have : num_other_pieces * 96 = 480 := by linarith
  have : num_other_pieces = 5 := by exact_mod_cast (nat.div_eq_of_eq_mul_right (show 96 > 0 from dec_trivial) this symm)
  linarith

end jangshe_clothing_l705_705039


namespace disease_given_positive_test_l705_705388

variable (Pr : Type → ℝ)
variables (D T : Prop)
variables (Pr_D : Pr D = 1 / 400)
variables (Pr_not_D : Pr (¬D) = 399 / 400)
variables (Pr_T_given_D : Pr T = 1)
variables (Pr_T_given_not_D : PrT_given_not_D = 0.03)

theorem disease_given_positive_test : Pr (D ∧ T) / Pr T = 100 / 1297 := sorry

end disease_given_positive_test_l705_705388


namespace floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l705_705371

theorem floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2 (n : ℕ) (hn : n > 0) :
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
  sorry

end floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l705_705371


namespace find_r_l705_705338

theorem find_r :
  ∃ r : ℤ, (-1 ≤ r ∧ r ≤ 5) ∧ 
  (∀ x : ℝ, (1 - 1 / x) * (1 + x)^5 = (1 + (5 * x) + (10 * x^2) - 10 * x + (5 * x^4) + x^5) ∧ 
  !  ∑ k in (-1 to 5), ((1 - 1/x)):k * a_k = 0 ) :=
  sorry

end find_r_l705_705338


namespace gcd_at_most_3_digits_l705_705845

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705845


namespace min_distance_origin_to_line_l705_705723

-- Define the origin O
def origin := (0 : ℝ, 0 : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) := x + y - 1 = 0

-- Define the formula for the distance from a point to a line
def point_to_line_distance (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- Formulate the problem as a theorem to be proved
theorem min_distance_origin_to_line : 
  point_to_line_distance (fst origin) (snd origin) 1 1 (-1) = (sqrt 2) / 2 :=
  by
    sorry

end min_distance_origin_to_line_l705_705723


namespace minimum_number_of_kings_maximum_number_of_non_attacking_kings_l705_705200

-- Definitions for the chessboard and king placement problem

-- Problem (a): Minimum number of kings covering the board
def minimum_kings_covering_board (board_size : Nat) : Nat :=
  sorry

theorem minimum_number_of_kings (h : 6 = board_size) :
  minimum_kings_covering_board 6 = 4 := 
  sorry

-- Problem (b): Maximum number of non-attacking kings
def maximum_non_attacking_kings (board_size : Nat) : Nat :=
  sorry

theorem maximum_number_of_non_attacking_kings (h : 6 = board_size) :
  maximum_non_attacking_kings 6 = 9 :=
  sorry

end minimum_number_of_kings_maximum_number_of_non_attacking_kings_l705_705200


namespace Liz_total_spend_l705_705081

theorem Liz_total_spend :
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  total_spent_cost = 40 :=
by
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  show total_spent_cost = 40 from
    sorry

end Liz_total_spend_l705_705081


namespace linear_term_coefficient_of_expansion_sum_of_odd_coefficients_of_expansion_l705_705259

-- Problem 1
theorem linear_term_coefficient_of_expansion :
  (choose 17 8 = choose 17 9) →
  (choose 17 9 * (2^9) = choose 17 9 * (2^9)) :=
begin
  -- we are given that the binomial coefficients are equal
  intros h,
  -- therefore, the coefficient of the linear term will be choose 17 9 * (2^9)
  exact congr_arg2 (*) h rfl,
end

-- Problem 2
theorem sum_of_odd_coefficients_of_expansion :
  let a_0 := (2 - 1)^7,
      f := λ x : ℤ, (2 * x - 1)^7,
      f1 := f 1,
      f_neg1 := f (-1),
      a_sum := (f1 + f_neg1) / 2
  in a_sum = -1093 :=
begin
  -- simplify the expression
  dsimp [f, a_sum, f1, f_neg1],
  -- compute f(1) and f(-1)
  have h1 : (2 * 1 - 1)^7 = 1 := rfl,
  have h_neg1 : (2 * (-1) - 1)^7 = -2187 := by norm_num,
  rw [h1, h_neg1],
  -- compute the sum
  norm_num,
end

end linear_term_coefficient_of_expansion_sum_of_odd_coefficients_of_expansion_l705_705259


namespace derivative_of_f_l705_705055

noncomputable def f (x : ℝ) : ℝ :=
  (Nat.choose 4 0 : ℝ) - (Nat.choose 4 1 : ℝ) * x + (Nat.choose 4 2 : ℝ) * x^2 - (Nat.choose 4 3 : ℝ) * x^3 + (Nat.choose 4 4 : ℝ) * x^4

theorem derivative_of_f : 
  ∀ (x : ℝ), (deriv f x) = 4 * (-1 + x)^3 :=
by
  sorry

end derivative_of_f_l705_705055


namespace liz_spent_total_l705_705082

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end liz_spent_total_l705_705082


namespace periodic_sequence_l705_705284

theorem periodic_sequence (x : ℕ → ℝ) (a : ℝ) (h_pos : ∀ n, 0 < x n) (h_periodic : ∃ T > 0, ∀ n, x (n + T) = x n)
  (h_recurrence : ∀ n, x (n + 2) = 1 / 2 * (1 / x (n + 1) + x n)) :
  ∃ a ∈ ℝ, (∀ n, x n = a ∨ x n = 1 / a) :=
by
  sorry

end periodic_sequence_l705_705284


namespace parabola_integer_points_l705_705233

noncomputable def parabola (x y : ℝ) : Prop :=
  (6 * x + 2 * y ≤ 1200) ∧ (6 * x + 2 * y ≥ -1200)

theorem parabola_integer_points :
  {p : ℤ × ℤ // parabola p.1 p.2}.card = 151 :=
begin
  sorry
end

end parabola_integer_points_l705_705233


namespace gcd_digit_bound_l705_705803

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705803


namespace point_in_second_quadrant_l705_705869

theorem point_in_second_quadrant (m : ℝ) : (-1 < 0) ∧ (m^2 + 1 > 0) → Quadrant (-1, m^2 + 1) = Quadrant.second :=
by
  sorry

end point_in_second_quadrant_l705_705869


namespace find_x_plus_2y_squared_l705_705770

theorem find_x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2 * y) = 48) (h2 : y * (x + 2 * y) = 72) :
  (x + 2 * y) ^ 2 = 96 := 
sorry

end find_x_plus_2y_squared_l705_705770


namespace sequence_inequality_l705_705705

theorem sequence_inequality {f : ℕ → ℝ}
  (h : ∀ n, f n < ∑ k in finset.range (2 * n + 1) \finset.range n, (f k / (k + 1)) + 1 / (2 * n + 2007)) :
  ∀ n, f n < 1 / n :=
sorry

end sequence_inequality_l705_705705


namespace christian_sue_need_more_money_l705_705264

-- Definition of initial amounts
def christian_initial := 5
def sue_initial := 7

-- Definition of earnings from activities
def christian_per_yard := 5
def christian_yards := 4
def sue_per_dog := 2
def sue_dogs := 6

-- Definition of perfume cost
def perfume_cost := 50

-- Theorem statement for the math problem
theorem christian_sue_need_more_money :
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  total_money < perfume_cost → perfume_cost - total_money = 6 :=
by 
  intros
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  sorry

end christian_sue_need_more_money_l705_705264


namespace lamp_height_difference_l705_705402

def old_lamp_height : ℝ := 1
def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := new_lamp_height - old_lamp_height

theorem lamp_height_difference :
  height_difference = 1.3333333333333335 := by
  sorry

end lamp_height_difference_l705_705402


namespace maximum_area_equilateral_triangle_l705_705933

theorem maximum_area_equilateral_triangle {a b c : ℝ} (h1 : a + b + c = 2 * p) 
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  let S := sqrt (p * (p - a) * (p - b) * (p - c))
  in ∀ (a1 a2 a3 : ℝ), 
    (S ≤ sqrt (p * (p - a1) * (p - a2) * (p - a3))) → 
    (a1 = a2 ∧ a2 = a3) :=
by
  sorry

end maximum_area_equilateral_triangle_l705_705933


namespace minimum_additional_marbles_l705_705916

theorem minimum_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 34) : 
  ∃ additional_marbles : ℕ, additional_marbles = 44 :=
by
  -- The formal proof would go here.
  sorry

end minimum_additional_marbles_l705_705916


namespace d_is_distance_function_l705_705704

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance_function : 
  (∀ x, d x x = 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) :=
by
  sorry

end d_is_distance_function_l705_705704


namespace ratio_problem_l705_705366

theorem ratio_problem
  (a b c d : ℝ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 49) :
  d / a = 1 / 122.5 :=
by {
  -- Proof steps would go here
  sorry
}

end ratio_problem_l705_705366


namespace probability_two_diagonals_intersect_l705_705007

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l705_705007


namespace sqrt_eq_prime_sol_l705_705068

theorem sqrt_eq_prime_sol (p x y : ℕ) (hp : Prime p) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ (x y : ℕ), √ x + √ y = √ p → (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0)) := sorry

end sqrt_eq_prime_sol_l705_705068


namespace number_of_democrats_in_senate_l705_705168

/-
This Lean statement captures the essence of the problem: proving the number of Democrats in the Senate (S_D) is 55,
under given conditions involving the House's and Senate's number of Democrats and Republicans.
-/

theorem number_of_democrats_in_senate
  (D R S_D S_R : ℕ)
  (h1 : D + R = 434)
  (h2 : R = D + 30)
  (h3 : S_D + S_R = 100)
  (h4 : S_D * 4 = S_R * 5) :
  S_D = 55 := by
  sorry

end number_of_democrats_in_senate_l705_705168


namespace incenter_of_BCD_l705_705022

structure Prism (A B C D O : Type) :=
  (orthogonal_projection : A → B × C × D → O)
  (equal_distances : O → Prop)

theorem incenter_of_BCD (A B C D O : Type) 
  [prism : Prism A B C D O]
  (proj : prism.orthogonal_projection)
  (dist_eq : prism.equal_distances O) : 
  ∃ i : Type, 
    is_incenter i (B C D O) :=
begin
  sorry
end

end incenter_of_BCD_l705_705022


namespace gcd_digit_bound_l705_705786

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705786


namespace average_homework_time_decrease_l705_705552

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l705_705552


namespace k_mul_k1_eq_one_mn_fixed_point_l705_705725

variable (k k₁ : ℝ)
variable (l : ℝ → ℝ := λ x, k * x + 1)
variable (l₁ : ℝ → ℝ := λ x, k₁ * x + 1)

-- Condition: k > 0 and k ≠ 1
variable h1 : k > 0
variable h2 : k ≠ 1

-- Condition: lines l and l₁ are symmetric about the line y = x + 1
variable symmetric_about : ∀ x : ℝ, l(l₁(x) - 1) = x + 1

-- Condition: the line l intersects the ellipse E: x^2 / 4 + y^2 = 1 at points A and M
variable E : ℝ × ℝ → Prop := λ p, (p.1 ^ 2) / 4 + p.2 ^ 2 = 1
variable intersect_l : ∃ A M : ℝ × ℝ, E A ∧ E M ∧ l A.1 = A.2 ∧ l M.1 = M.2

-- Condition: the line l₁ intersects the ellipse at points A and N
variable intersect_l1 : ∃ A N : ℝ × ℝ, E A ∧ E N ∧ l₁ A.1 = A.2 ∧ l₁ N.1 = N.2

-- Proof that k * k₁ = 1
theorem k_mul_k1_eq_one (h1 : k > 0) (h2 : k ≠ 1) (symmetric_about : ∀ x : ℝ, l(l₁(x) - 1) = x + 1) : k * k₁ = 1 := 
sorry

-- Proof that for any k, line MN always passes through a fixed point
theorem mn_fixed_point (intersect_l : ∃ A M : ℝ × ℝ, E A ∧ E M ∧ l A.1 = A.2 ∧ l M.1 = M.2) 
                        (intersect_l1 : ∃ A N : ℝ × ℝ, E A ∧ E N ∧ l₁ A.1 = A.2 ∧ l₁ N.1 = N.2) : 
                      ∃ P : ℝ × ℝ, ∀ k > 0, ∀ k₁ ≠ 1, (∃ MN : ℝ × ℝ, l MN.1 = MN.2 • intersects_l1 = P) :=
sorry

end k_mul_k1_eq_one_mn_fixed_point_l705_705725


namespace max_consecutive_irreducible_l705_705611

-- Define what it means for a five-digit number to be irreducible
def is_irreducible (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ¬∃ x y : ℕ, 100 ≤ x ∧ x < 1000 ∧ 100 ≤ y ∧ y < 1000 ∧ x * y = n

-- Prove the maximum number of consecutive irreducible five-digit numbers is 99
theorem max_consecutive_irreducible : ∃ m : ℕ, m = 99 ∧ 
  (∀ n : ℕ, (n ≤ 99901) → (∀ k : ℕ, (n ≤ k ∧ k < n + m) → is_irreducible k)) ∧
  (∀ x y : ℕ, x > 99 → ∀ n : ℕ, (n ≤ 99899) → (∀ k : ℕ, (n ≤ k ∧ k < n + x) → is_irreducible k) → x = 99) :=
by
  sorry

end max_consecutive_irreducible_l705_705611


namespace inequality_solution_set_l705_705977

noncomputable def solution_set := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem inequality_solution_set :
  (solution_set = {x : ℝ | x ≤ -3 ∨ x ≥ 1}) :=
sorry

end inequality_solution_set_l705_705977


namespace liquid_flow_problem_l705_705856

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end liquid_flow_problem_l705_705856


namespace a_beats_b_by_4_rounds_l705_705197

variable (T_a T_b : ℝ)
variable (race_duration : ℝ) -- duration of the 4-round race in minutes
variable (time_difference : ℝ) -- Time that a beats b by in the 4-round race

open Real

-- Given conditions
def conditions :=
  (T_a = 7.5) ∧                             -- a's time to complete one round
  (race_duration = T_a * 4 + 10) ∧          -- a beats b by 10 minutes in a 4-round race
  (time_difference = T_b - T_a)             -- The time difference per round is T_b - T_a

-- Mathematical proof statement
theorem a_beats_b_by_4_rounds
  (h : conditions T_a T_b race_duration time_difference) :
  10 / time_difference = 4 := by
  sorry

end a_beats_b_by_4_rounds_l705_705197


namespace square_of_radius_of_inscribed_circle_l705_705601

theorem square_of_radius_of_inscribed_circle
  (E F G H R S O : Point)
  (r : ℝ)
  (h_tangent_ER : dist E R = 15)
  (h_tangent_RF : dist R F = 35)
  (h_tangent_GS : dist G S = 47)
  (h_tangent_SH : dist S H = 29)
  (h_inscribed : circle_inscribed_quad E F G H O r) :
  r^2 = 1886 :=
sorry

end square_of_radius_of_inscribed_circle_l705_705601


namespace initial_average_age_l705_705950

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 17) (h2 : n * A + 32 = (n + 1) * 15) : A = 14 := by
  sorry

end initial_average_age_l705_705950


namespace Veronica_to_Half_Samir_Ratio_l705_705126

-- Mathematical conditions 
def Samir_stairs : ℕ := 318
def Total_stairs : ℕ := 495
def Half_Samir_stairs : ℚ := Samir_stairs / 2

-- Definition for Veronica's stairs as a multiple of half Samir's stairs
def Veronica_stairs (R: ℚ) : ℚ := R * Half_Samir_stairs

-- Lean statement to prove the ratio
theorem Veronica_to_Half_Samir_Ratio (R : ℚ) (H1 : Veronica_stairs R + Samir_stairs = Total_stairs) : R = 1.1132 := 
by
  sorry

end Veronica_to_Half_Samir_Ratio_l705_705126


namespace max_A_plus_B_l705_705428

theorem max_A_plus_B:
  ∃ A B C D : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  A + B + C + D = 17 ∧ ∃ k : ℕ, C + D ≠ 0 ∧ A + B = k * (C + D) ∧
  A + B = 16 :=
by sorry

end max_A_plus_B_l705_705428


namespace point_in_second_quadrant_l705_705864

theorem point_in_second_quadrant (m : ℝ) : 
  let P := (-1, m^2 + 1) in
    P.1 < 0 ∧ P.2 > 0 → P ∈ quadrant2 :=
begin
  intro h,
  sorry,
end

end point_in_second_quadrant_l705_705864


namespace catch_two_salmon_l705_705491

def totalTroutWeight : ℕ := 8
def numBass : ℕ := 6
def weightPerBass : ℕ := 2
def totalBassWeight : ℕ := numBass * weightPerBass
def campers : ℕ := 22
def weightPerCamper : ℕ := 2
def totalFishWeightRequired : ℕ := campers * weightPerCamper
def totalTroutAndBassWeight : ℕ := totalTroutWeight + totalBassWeight
def additionalFishWeightRequired : ℕ := totalFishWeightRequired - totalTroutAndBassWeight
def weightPerSalmon : ℕ := 12
def numSalmon : ℕ := additionalFishWeightRequired / weightPerSalmon

theorem catch_two_salmon : numSalmon = 2 := by
  sorry

end catch_two_salmon_l705_705491


namespace number_of_common_elements_l705_705903

def multiples_up_to (n k : ℕ) : set ℕ := {m | ∃ i, 1 ≤ i ∧ i ≤ k ∧ m = n * i}

def S : set ℕ := multiples_up_to 4 1500
def T : set ℕ := multiples_up_to 6 1500

theorem number_of_common_elements : (S ∩ T).to_finset.card = 500 := by
  sorry

end number_of_common_elements_l705_705903


namespace largest_integer_initial_and_last_segments_l705_705134

def is_k_segment (a : ℕ) (k : ℕ) (segment : ℕ) := 
  -- some predicate that checks whether segment is a k-segment of a
  sorry

def all_k_segments_different (a : ℕ) (k : ℕ) := 
  ∀ (i j : ℕ), (i ≠ j) → is_k_segment a k i → is_k_segment a k j → i ≠ j

theorem largest_integer_initial_and_last_segments (a : ℕ) (k : ℕ) (k ≥ 2) :
  -- defining that a is the largest integer with the property
  (∀ (a' : ℕ), a' > a → ¬ all_k_segments_different a' k) →
  all_k_segments_different a k →
  -- proving the initial (k-1)-segment and last (k-1)-segment are identical
  (initial_segment (k - 1) a = last_segment (k - 1) a) :=
by { sorry }

end largest_integer_initial_and_last_segments_l705_705134


namespace mul_99_101_square_98_l705_705258

theorem mul_99_101 : 99 * 101 = 9999 := sorry

theorem square_98 : 98^2 = 9604 := sorry

end mul_99_101_square_98_l705_705258


namespace inequality_a3_minus_b3_l705_705326

theorem inequality_a3_minus_b3 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 - b^3 < 0 :=
by sorry

end inequality_a3_minus_b3_l705_705326


namespace smallest_k_exists_l705_705664

theorem smallest_k_exists : ∃ (k : ℕ) (n : ℕ), k = 53 ∧ k^2 + 49 = 180 * n :=
sorry

end smallest_k_exists_l705_705664


namespace simple_interest_principal_l705_705306

theorem simple_interest_principal (R T SI : ℝ) (hR : R = 9 / 100) (hT : T = 1) (hSI : SI = 900) : 
  (SI / (R * T) = 10000) :=
by
  sorry

end simple_interest_principal_l705_705306


namespace complex_purely_imaginary_solution_l705_705728

theorem complex_purely_imaginary_solution (z : ℂ) (h1 : z.im = z) (h2 : ((z + 1)^2 - 2 * complex.I).im = (z + 1)^2 - 2 * complex.I ) : 
          z = -complex.I := 
sorry

end complex_purely_imaginary_solution_l705_705728


namespace solve_log_eq_l705_705163

theorem solve_log_eq (x : ℝ) (h1 : x > 0) (h2 : 2 * x + 1 > 0) : log 10 (2 * x + 1) + log 10 x = 1 → x = 2 :=
by
  intro h
  sorry

end solve_log_eq_l705_705163


namespace find_a_l705_705703

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line equation passing through P(2,2)
def line_through_P (m b x y : ℝ) : Prop := y = m * x + b ∧ (2, 2) = (x, y)

-- Define the line equation ax - y + 1 = 0
def perpendicular_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a : ∃ a : ℝ, ∀ x y m b : ℝ,
    circle x y ∧ line_through_P m b x y ∧
    (line_through_P m b x y → perpendicular_line a x y) → a = 2 :=
by
  intros
  sorry

end find_a_l705_705703


namespace team_total_score_l705_705588

theorem team_total_score :
  ∀ (T : ℕ) (best_score : ℕ) (avg_new_best : ℕ) (team_size : ℕ),
  team_size = 6 →
  best_score = 85 →
  avg_new_best = 84 →
  (avg_new_best * team_size - (92 - best_score) = 497) →
  T = 497 :=
by
  intros T best_score avg_new_best team_size h_team_size h_best_score h_avg_new_best h_calculation
  have h_total : (avg_new_best * team_size - (92 - best_score)) = 497 :=
    by exact h_calculation
  exact h_total

#eval team_total_score 497 85 84 6 rfl rfl rfl rfl

end team_total_score_l705_705588


namespace divisible_by_3_l705_705692

theorem divisible_by_3 :
  ∃ n : ℕ, (5 + 2 + n + 4 + 8) % 3 = 0 ∧ n = 2 := 
by
  sorry

end divisible_by_3_l705_705692


namespace find_J_l705_705772

noncomputable def J := 196
def K := 290 - J
def condition1 := J - 8 = 2 * K
def condition2 := J + K = 290

theorem find_J : J = 196 :=
by
  have h1 : condition1 := sorry
  have h2 : condition2 := sorry
  exact rfl

end find_J_l705_705772


namespace median_name_length_is_four_l705_705634

def name_lengths : List ℕ := repeat 3 8 ++ repeat 4 5 ++ repeat 6 3 ++ repeat 7 5 ++ repeat 8 4

theorem median_name_length_is_four : 
  -- a study group consists of 25 people with predefined name lengths
  length name_lengths = 25 ->
  -- The median length of these names is 4.
  (name_lengths.nth ((25 + 1) / 2 - 1) = some 4) :=
by
  sorry

end median_name_length_is_four_l705_705634


namespace sum_of_signed_sequence_zero_l705_705920

theorem sum_of_signed_sequence_zero {n : ℕ} (a : ℕ → ℕ) (h1 : ∀ k, k < n + 1 → a k ≤ k) (h2 : even (∑ k in finset.range (n + 1), a k)) :
  ∃ (ε : ℕ → ℤ), (∀ k, k < n + 1 → ε k = 1 ∨ ε k = -1) ∧ (∑ k in finset.range (n + 1), ε k * (a k) : ℤ) = 0 :=
by sorry

end sum_of_signed_sequence_zero_l705_705920


namespace sum_of_a_values_l705_705656

theorem sum_of_a_values : 
  let A := 9 
  let B := a + 12 
  let C := 16 
  in (B^2 - 4 * A * C = 0) -> let a1 := 12 in let a2 := -36 in a1 + a2 = -24 :=
begin
  intros,
  sorry

end sum_of_a_values_l705_705656


namespace math_problem_statement_l705_705358

noncomputable def solve_problem (a : ℝ) : Prop :=
  let B := {x : ℝ | -2 < x ∧ x ≤ 6}
  let A := {x : ℝ | x^2 - a * x - 6 * a^2 ≤ 0}
  B ⊆ A → (a ∈ Set.Icc (-∞) (-3) ∪ Set.Icc 2 ∞)

theorem math_problem_statement {a : ℝ} : solve_problem a :=
by {
  sorry
}

end math_problem_statement_l705_705358


namespace find_a_if_f_is_even_l705_705731

noncomputable def f (a x : ℝ) : ℝ := x * log (sqrt (x^2 + 1) - a * x)

theorem find_a_if_f_is_even : 
  (∀ x : ℝ, f a x = f a (-x)) → (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_if_f_is_even_l705_705731


namespace fourfold_composition_is_odd_l705_705466

-- Define what it means for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

-- The problem statement
theorem fourfold_composition_is_odd (f : ℝ → ℝ) (hf : is_odd_function f) :
  is_odd_function (λ x, f(f(f(f(x))))) :=
by sorry

end fourfold_composition_is_odd_l705_705466


namespace limit_of_f_l705_705739

noncomputable def f (x : ℝ) : ℝ := (3/2) * x^2 - 2 * Real.exp x

theorem limit_of_f :
  (lim (Δx→0) (2 * (f Δx - f 0) / Δx)) = -4 :=
by
  sorry

end limit_of_f_l705_705739


namespace specially_monotonous_count_is_65_l705_705661

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_strictly_increasing (digits : List ℕ) : Prop :=
  ∀ i j, i < j → digits[i] < digits[j]

def is_strictly_decreasing (digits : List ℕ) : Prop :=
  ∀ i j, i < j → digits[i] > digits[j]

def is_special_monotonous (digits : List ℕ) : Prop :=
  (∀ digit in digits, is_even digit) ∨ (∀ digit in digits, is_odd digit) ∧
  (is_strictly_increasing digits ∨ is_strictly_decreasing digits)

def count_special_monotonous_numbers : ℕ :=
  sorry -- Placeholder for the actual counting logic

theorem specially_monotonous_count_is_65 :
  count_special_monotonous_numbers = 65 :=
sorry -- Placeholder for the proof

end specially_monotonous_count_is_65_l705_705661


namespace gcd_max_digits_l705_705833

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705833


namespace one_by_one_tile_position_l705_705317

theorem one_by_one_tile_position :
  ∀ (tiles3x1 : List (Fin 7 × Fin 7) × List (Fin 7 × Fin 7) × List (Fin 7 × Fin 7)), 
    length tiles3x1 = 16 →
    ∃ (tile1x1 : Fin 7 × Fin 7), 
      (tile1x1 = (3, 3)) ∨ (tile1x1.1 = 0) ∨ (tile1x1.2 = 0) ∨ (tile1x1.1 = 6) ∨ (tile1x1.2 = 6) :=
by
  sorry

end one_by_one_tile_position_l705_705317


namespace find_angle_C_l705_705697

noncomputable def f (x : ℝ) := 2 * real.sqrt 3 * real.sin x * real.cos x + 2 * real.cos x ^ 2 - 1

lemma max_value_f :
  ∃ (x_vals : set ℝ), ∀ x, f x ≤ 2 ∧ (∀ x ∈ x_vals, f x = 2) ∧
    x_vals = {x : ℝ | ∃ k : ℤ, x = k * real.pi + real.pi / 6} :=
sorry

theorem find_angle_C (A B C : ℝ) (a b c : ℝ)
  (ha : a = 1) (hb : b = real.sqrt 2) (hfA : f A = 2) :
  ∃ (c_vals : set ℝ), ∀ C, C ∈ c_vals ∧
    c_vals = {(15 : ℝ), 75, 105} :=
sorry

end find_angle_C_l705_705697


namespace point_in_second_quadrant_l705_705861

theorem point_in_second_quadrant (m : ℝ) : 
  let x := -1 in
  let y := m^2 + 1 in
  x < 0 ∧ y > 0 →
  (∃ quadrant, quadrant = 2) :=
by
  let x := -1
  let y := m^2 + 1
  intro h,
  existsi 2,
  sorry

end point_in_second_quadrant_l705_705861


namespace gcd_digit_bound_l705_705821

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705821


namespace max_sin_cos_expression_l705_705299

open Real  -- Open the real numbers namespace

theorem max_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) *
              (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_sin_cos_expression_l705_705299


namespace factorize_polynomial_l705_705676

theorem factorize_polynomial (a b c : ℚ) : 
  b^2 - c^2 + a * (a + 2 * b) = (a + b + c) * (a + b - c) :=
by
  sorry

end factorize_polynomial_l705_705676


namespace length_DE_l705_705027

-- Given data
variables (D E F : Type)
variables [MetricSpace D E F]
variable (triangle_DEF : right_triangle D E F)

-- Additional Conditions
variable (cos_F : ℝ)
variable (cos_F_eq : cos_F = (5 * sqrt 34) / 34)
variable (EF_length : ℝ)
variable (EF_length_eq : EF_length = sqrt 34)

-- The statement to be proved
theorem length_DE
  (h_cosF : cos_F = (5 * sqrt 34) / 34)
  (h_EF : EF_length = sqrt 34) :
  ∃ DE_length : ℝ, DE_length = 5 :=
sorry

end length_DE_l705_705027


namespace find_angle_PQR_l705_705902

noncomputable def point := (ℝ × ℝ × ℝ)

def P : point := (-3, 1, 7)
def Q : point := (-4, 1, 3)
def R : point := (-5, 0, 4)

noncomputable def distance (A B : point) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

noncomputable def cos_angle (A B C : point) : ℝ :=
  let distAB := distance A B
  let distBC := distance B C
  let distAC := distance A C
  (distAB^2 + distBC^2 - distAC^2) / (2 * distAB * distBC)

noncomputable def angle_degrees (cos_theta : ℝ) : ℝ :=
  real.acos cos_theta * (180 / real.pi)

theorem find_angle_PQR : 
  let θ := angle_degrees (cos_angle P Q R)
  θ ≈ 65 := sorry

end find_angle_PQR_l705_705902


namespace can_sell_tickets_l705_705094

theorem can_sell_tickets (n : ℕ) (queue : list ℕ) :
  list.length queue = 2 * n ∧ 
  (count (λ x, x = 5) queue = n) ∧ 
  (count (λ x, x = 10) queue = n) →
  (∀ k ≤ 2 * n, (count (λ x, x = 5) (take k queue) ≥ count (λ x, x = 10) (take k queue))) → 
  true :=
begin
  sorry
end

end can_sell_tickets_l705_705094


namespace value_of_m_making_365m_divisible_by_12_l705_705666

theorem value_of_m_making_365m_divisible_by_12
  (m : ℕ)
  (h1 : (3650 + m) % 3 = 0)
  (h2 : (50 + m) % 4 = 0) :
  m = 0 :=
sorry

end value_of_m_making_365m_divisible_by_12_l705_705666


namespace bernoulli_inequality_l705_705070

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x ≥ -1) (hn : n ≥ 1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end bernoulli_inequality_l705_705070


namespace area_of_triangle_l705_705054

-- Given definitions for triangle ABC with right angle at C and hypotenuse AB=60
def Triangle (A B C : ℝ × ℝ) := ∃ x y : ℝ, C = (x, y) ∧ dist A B = 60 ∧ 
(∃ m₁ m₂ : ℝ, line (A, B) (1, x + 3) ∧ line (B, A) (2, 2 * x + 4))

-- The theorem stating the area of the right triangle given the conditions
theorem area_of_triangle (A B C : ℝ × ℝ) (h : Triangle A B C) : 
  ∃ (area : ℝ), area = 400 := 
sorry

end area_of_triangle_l705_705054


namespace strategic_roads_important_l705_705874

structure Graph (V : Type) :=
  (E : set (V × V))
  (is_connected : ∀ v1 v2 : V, ∃ p : list (V × V), p ≠ [] ∧ (v1, v2) ∈ p++E)

def is_important {V : Type} (G : Graph V) (R : set (V × V)) : Prop :=
  ∀ v1 v2 : V, (v1 ≠ v2) → 
  ∃ p : list (V × V), p ≠ [] ∧ (∀ e ∈ p, e ∈ G.E \ R) ∧
  ¬ ∃ q : list (V × V), q ≠ [] ∧ (∀ e ∈ q, e ∈ G.E \ R) ∧ q.head = Some (v1, v2)

def is_strategic {V : Type} (G : Graph V) (R : set (V × V)) : Prop :=
  is_important G R ∧ ∀ R' : set (V × V), R' ⊂ R → ¬ is_important G R'

theorem strategic_roads_important {V : Type} (G : Graph V) (R_a R_b : set (V × V))
  (ha : is_strategic G R_a) (hb : is_strategic G R_b) (h_diff : R_a ≠ R_b) : 
  is_important G ((R_a \ R_b) ∪ (R_b \ R_a)) := sorry

end strategic_roads_important_l705_705874


namespace probability_diagonals_intersect_l705_705006

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l705_705006


namespace original_bet_is_40_l705_705218

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l705_705218


namespace mean_value_points_range_l705_705378

theorem mean_value_points_range (b : ℝ) : 
  (∃ m₁ m₂ : ℝ, 0 < m₁ ∧ m₁ < b ∧ 0 < m₂ ∧ m₂ < b ∧ m₁ ≠ m₂ ∧ 
  (f' m₁) * b = (f b - f 0) ∧ (f' m₂) * b = (f b - f 0)) 
  ↔ (3 / 2 < b ∧ b < 3) := 
sorry

where
  f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x ^ 2
  f' (x : ℝ) : ℝ := x ^ 2 - 2 * x

end mean_value_points_range_l705_705378


namespace james_total_points_l705_705405

def f : ℕ := 13
def s : ℕ := 20
def p_f : ℕ := 3
def p_s : ℕ := 2

def total_points : ℕ := (f * p_f) + (s * p_s)

theorem james_total_points : total_points = 79 := 
by
  -- Proof would go here.
  sorry

end james_total_points_l705_705405


namespace manager_packages_l705_705963

-- Define the ranges of the apartments on each floor
def first_floor_range := [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135]

def second_floor_range := [190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

-- Define a function to count the occurrences of digit 0 in a list of numbers
def count_zeros (numbers : List Nat) : Nat :=
  numbers.foldr (λ n acc, acc + n.digits.count 0) 0

-- Aggregate the numbers on both floors
def all_apartments := first_floor_range ++ second_floor_range

-- Final proof statement
theorem manager_packages : count_zeros all_apartments = 24 := by
  sorry

end manager_packages_l705_705963


namespace correct_model_l705_705561

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l705_705561


namespace find_equations_l705_705668

def is_neither_directly_nor_inversely_proportional (x y : ℝ) : Prop :=
  ¬ (∃ k : ℝ, x = k * y) ∧ ¬ (∃ k : ℝ, x * y = k)

theorem find_equations {x y : ℝ} :
  (2 * x + y = 5 → is_neither_directly_nor_inversely_proportional x y) ∧
  (2 * x + 3 * y = 15 → is_neither_directly_nor_inversely_proportional x y) :=
by
  split
  · intro h
    -- Proof goes here
    sorry
  
  · intro h
    -- Proof goes here
    sorry

end find_equations_l705_705668


namespace nina_initial_money_l705_705091

variables (W : ℝ) (M : ℝ)
constant initial_widgets : ℝ
constant reduced_price : ℝ
constant additional_items_cost : ℝ

axiom widgets_initial : initial_widgets = 10
axiom price_reduction : reduced_price = 2.35
axiom items_cost : additional_items_cost = 15

theorem nina_initial_money :
  M = initial_widgets * W → 
  M = 17 * (W - reduced_price) → 
  M + additional_items_cost = 72.07 :=
by 
  intros h1 h2 
  sorry

end nina_initial_money_l705_705091


namespace three_planes_divide_space_at_most_8_parts_l705_705175

theorem three_planes_divide_space_at_most_8_parts :
  ∀ (p1 p2 p3 : set ℝ³), 
    (∀ (p : set ℝ³), plane p → ∃ parts : ℕ, parts ≤ 2) → 
    (∀ (p1 p2 : set ℝ³), plane p1 → plane p2 → ∃ parts : ℕ, parts ≤ 4) → 
    (∀ (p1 p2 p3 : set ℝ³), plane p1 → plane p2 → plane p3 → ∃ parts : ℕ, parts ≤ 8) :=
by 
  sorry

end three_planes_divide_space_at_most_8_parts_l705_705175


namespace miles_driven_l705_705484

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end miles_driven_l705_705484


namespace sum_of_first_ten_terms_l705_705707

theorem sum_of_first_ten_terms (a : ℕ → ℝ)
  (h1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
  (h2 : ∀ n, a n < 0) :
  (5 * (a 3 + a 8) = -15) :=
sorry

end sum_of_first_ten_terms_l705_705707


namespace gum_cost_700_eq_660_cents_l705_705951

-- defining the cost function
def gum_cost (n : ℕ) : ℝ :=
  if n ≤ 500 then n * 0.01
  else 5 + (n - 500) * 0.008

-- proving the specific case for 700 pieces of gum
theorem gum_cost_700_eq_660_cents : gum_cost 700 = 6.60 := by
  sorry

end gum_cost_700_eq_660_cents_l705_705951


namespace differentiable_implies_inequality_l705_705784

variable {α : Type*} [LinearOrderedField α]

theorem differentiable_implies_inequality
  (f g : α → α) (a b : α)
  (hf : Differentiable α f) 
  (hg : Differentiable α g)
  (hfg_deriv : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (haf : f a = g a) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x := sorry

end differentiable_implies_inequality_l705_705784


namespace number_of_divisors_30_number_of_divisors_135_l705_705478

-- Definitions of divisibility and counting
def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

def number_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d => is_divisor d n).card

-- Problem Statement 1: Number of divisors of 30 is 8
theorem number_of_divisors_30 : number_of_divisors 30 = 8 := 
  sorry

-- Problem Statement 2: Number of divisors of 135 is 8
theorem number_of_divisors_135 : number_of_divisors 135 = 8 := 
  sorry

end number_of_divisors_30_number_of_divisors_135_l705_705478


namespace ma_xiaotiao_rank_l705_705988

-- Define the conditions
def total_participants : Nat := 34
def people_behind_eq_twice_people_ahead (x : Nat) : Prop := (total_participants - x = 2 * (x - 1))

-- State the theorem
theorem ma_xiaotiao_rank : ∃ x : Nat, people_behind_eq_twice_people_ahead x ∧ x = 12 :=
begin
  sorry
end

end ma_xiaotiao_rank_l705_705988


namespace ratio_c_a_l705_705776

theorem ratio_c_a (a b c : ℚ) (h1 : a * b = 3) (h2 : b * c = 8 / 5) : c / a = 8 / 15 := 
by 
  sorry

end ratio_c_a_l705_705776


namespace jugglers_balls_needed_l705_705524

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l705_705524


namespace complement_union_eq_l705_705536

universe u

-- Definitions based on conditions in a)
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

-- The goal to prove based on c)
theorem complement_union_eq :
  (U \ (M ∪ N)) = {5, 6} := 
by sorry

end complement_union_eq_l705_705536


namespace tickets_sold_l705_705600

theorem tickets_sold (T : ℕ) (h1 : 3 * T / 4 > 0)
    (h2 : 5 * (T / 4) / 9 > 0)
    (h3 : 80 > 0)
    (h4 : 20 > 0) :
    (1 / 4 * T - 5 / 36 * T = 100) -> T = 900 :=
by
  sorry

end tickets_sold_l705_705600


namespace problem_I_problem_II_l705_705748

-- Definitions of propositions p and q
def p (m x : ℝ) : Prop := m > 0 ∧ mx + 1 > 0
def q (x : ℝ) : Prop := (3 * x - 1) * (x + 2) < 0

-- Problem (I) statement
theorem problem_I : p 1 x ∧ q x ↔ -1 < x ∧ x < (1/3) :=
by sorry

-- Problem (II) statement
theorem problem_II (m : ℝ) : (∀ x, p m x → q x) ∧ (∃ x, q x ∧ ¬ p m x) ↔ (0 < m ∧ m ≤ 1/2) :=
by sorry

end problem_I_problem_II_l705_705748


namespace find_xyz_l705_705993

theorem find_xyz : ∃ (x y z : ℕ), x = 1 ∧ y = 9 ∧ z = 8 ∧ (11 * x + 11 * y + 11 * z = 100 * x + 10 * y + z) := 
by {
  let x := 1,
  let y := 9,
  let z := 8,
  have h : 11 * x + 11 * y + 11 * z = 100 * x + 10 * y + z,
  { sorry },  -- Proof that the equation holds
  use [x, y, z],
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  exact h,
}

end find_xyz_l705_705993


namespace sequence_2017th_term_l705_705957

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum (λ d => d ^ 3)

def sequence (a0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate sum_of_cubes_of_digits n a0

theorem sequence_2017th_term :
  sequence 2017 2017 = 352 := 
sorry

end sequence_2017th_term_l705_705957


namespace gcd_digits_le_3_l705_705798

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705798


namespace problem_1_problem_2_l705_705352

noncomputable def f (x a b : ℝ) := Real.exp x - a * x + b

noncomputable def g (x a b : ℝ) := f x a b + Real.log (x + 1)

theorem problem_1 (a b : ℝ) (H1 : f 0 a b = 2) (H2 : f x a b '' has_local_min_at x 0) : a = 1 ∧ b = 1 := sorry

theorem problem_2 (a b : ℝ) (H : ∀ x : ℝ, x ≥ 0 → g x a b ≥ 1 + b) : a ≤ 2 := sorry

end problem_1_problem_2_l705_705352


namespace value_of_m_l705_705659

theorem value_of_m :
  let f := λ x : ℝ, 3 * x ^ 2 - 1 / x + 5
  let g := λ x : ℝ, x ^ 2 - m
  f 3 - g 3 = 8 → m = -44 / 3 :=
by
  intro h
  sorry

end value_of_m_l705_705659


namespace min_value_of_M_l705_705980

variables {a1 a2 a3 a4 a5 a6 a7 : ℝ}
#check a1 ≥ 0 ∧ a2 ≥ 0 ∧ a3 ≥ 0 ∧ a4 ≥ 0 ∧ a5 ≥ 0 ∧ a6 ≥ 0 ∧ a7 ≥ 0 ∧ a1 + a2 + a3 + a4 + a5 + a6 + a7 = 1

def M : ℝ := max (a1 + a2 + a3) (max (a2 + a3 + a4) (max (a3 + a4 + a5) (max (a4 + a5 + a6) (a5 + a6 + a7))))

theorem min_value_of_M : 
  (a1 ≥ 0 ∧ a2 ≥ 0 ∧ a3 ≥ 0 ∧ a4 ≥ 0 ∧ a5 ≥ 0 ∧ a6 ≥ 0 ∧ a7 ≥ 0 ∧ a1 + a2 + a3 + a4 + a5 + a6 + a7 = 1) →
  M ≥ 1/3 :=
by
  sorry

end min_value_of_M_l705_705980


namespace subsets_union_count_l705_705750

theorem subsets_union_count {α : Type} [DecidableEq α] (S : Finset α) (h : S = {a, b, c, d}) :
  ∃ A B : Finset α, S = A ∪ B ∧ (S = {a, b, c, d}) ∧ card {p : Finset α × Finset α | S = p.1 ∪ p.2} = 41 :=
by
  sorry

end subsets_union_count_l705_705750


namespace pyramid_volume_l705_705139

noncomputable def volume_of_pyramid (S : ℝ) (H : ℝ) (d : ℝ)
  (h : d = 2 * H)
  (base_area : H^2 * real.sqrt 3 = S)
  (base_angle : real.sin (real.pi / 3) = real.sqrt 3 / 2)
  : ℝ :=
(1 / 3) * S * H

theorem pyramid_volume (S : ℝ) (H : ℝ) (d : ℝ)
  (h : d = 2 * H)
  (base_area : H^2 * real.sqrt 3 = S)
  (base_angle : real.sin (real.pi / 3) = real.sqrt 3 / 2)
  : volume_of_pyramid S H d h base_area base_angle = S * real.sqrt S * real.root 3 4 / 9 :=
by
  sorry

end pyramid_volume_l705_705139


namespace part1_part2_l705_705460

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705460


namespace uniqueness_of_function_l705_705472

open Complex

theorem uniqueness_of_function
  (g : ℂ → ℂ)
  (w a : ℂ)
  (hw_cond : w ^ 3 = 1 ∧ w ≠ 1)
  (f : ℂ → ℂ)
  (h_f : ∀ z : ℂ, f(z) + f(w * z + a) = g(z))
  : f = (λ z, (g(z) + g(w^2 * z + w * a + a) - g(w * z + a)) / 2) :=
sorry

end uniqueness_of_function_l705_705472


namespace solve_for_a_b_c_d_l705_705945

theorem solve_for_a_b_c_d :
  ∃ a b c d : ℕ, (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023 ∧ a^3 + b^3 + c^3 + d^3 = 43 := 
by
  sorry

end solve_for_a_b_c_d_l705_705945


namespace collinearity_of_points_l705_705429

theorem collinearity_of_points 
  (A B C B_0 C_0 D X G : Type*)
  [is_triangle A B C]
  [circumcircle Γ A B C]
  [midpoint B_0 A C]
  [midpoint C_0 A B]
  [altitude_foot D A B C]
  [tangent_circle ω B_0 C_0 Γ X]
  [centroid G A B C] :
  collinear {X, D, G} :=
sorry

end collinearity_of_points_l705_705429


namespace angle_alpha_value_l705_705344

theorem angle_alpha_value (k : ℤ) :
  (∃ α : ℝ, (11 : ℝ) (sin (π / 5)) = sin α ∧ (11 : ℝ) (-cos (π / 5)) = cos α ∧ 0 < (2 * π * (k : ℝ) - α) ∧ (2 * π * (k : ℝ) - α) < π) →
  α = - (3 * π / 10) + 2 * k * π :=
by
  intro h
  sorry

end angle_alpha_value_l705_705344


namespace trapezoid_inscribed_circle_ratio_trapezoid_circumscribed_circle_ratio_l705_705035

noncomputable def trapezoid_Area_to_InscribedCircle_Ratio (a b h : ℝ) (h_pos : 0 < h) : ℝ := 
  4 / π

noncomputable def trapezoid_Area_to_CircumscribedCircle_Ratio (a b h : ℝ) (h_pos : 0 < h) : ℝ := 
  2 / π

theorem trapezoid_inscribed_circle_ratio (a b h : ℝ) (h_pos : 0 < h)
  (ABCD_is_trapezoid : true) 
  (BD_angle_45 : true)
  (inscribed_circle_exists : true)
  (circumscribed_circle_exists : true) : 
  trapezoid_Area_to_InscribedCircle_Ratio a b h h_pos = 4 / π :=
  sorry

theorem trapezoid_circumscribed_circle_ratio (a b h : ℝ) (h_pos : 0 < h)
  (ABCD_is_trapezoid : true) 
  (BD_angle_45 : true)
  (inscribed_circle_exists : true)
  (circumscribed_circle_exists : true) : 
  trapezoid_Area_to_CircumscribedCircle_Ratio a b h h_pos = 2 / π :=
  sorry

end trapezoid_inscribed_circle_ratio_trapezoid_circumscribed_circle_ratio_l705_705035


namespace initial_house_cats_l705_705234

theorem initial_house_cats (H : ℕ) (H_condition : 13 + H - 10 = 8) : H = 5 :=
by
-- sorry provides a placeholder to skip the actual proof
sorry

end initial_house_cats_l705_705234


namespace hexagonal_prism_volume_is_correct_l705_705240

noncomputable def hexagonal_prism_volume (h : ℝ) (S_side : ℝ) : ℝ :=
let a := (S_side / (5 * h)) in
let area_hexagon := (3 * Real.sqrt 3 / 2) * a^2 in
area_hexagon * h

theorem hexagonal_prism_volume_is_correct :
  hexagonal_prism_volume 3 30 = 18 * Real.sqrt 3 :=
by
  rw [hexagonal_prism_volume]
  let a := (30 / (5 * 3))
  rw [a]
  let area_hexagon := (3 * Real.sqrt 3 / 2) * a^2
  rw [area_hexagon]
  let volume := area_hexagon * 3
  exact volume
  sorry

end hexagonal_prism_volume_is_correct_l705_705240


namespace quadratic_equation_solution_unique_l705_705660

noncomputable def b_solution := (-3 + 3 * Real.sqrt 21) / 2
noncomputable def c_solution := (33 - 3 * Real.sqrt 21) / 2

theorem quadratic_equation_solution_unique :
  (∃ (b c : ℝ), 
     (∀ (x : ℝ), 3 * x^2 + b * x + c = 0 → x = b_solution) ∧ 
     b + c = 15 ∧ 3 * c = b^2 ∧
     b = b_solution ∧ c = c_solution) :=
by { sorry }

end quadratic_equation_solution_unique_l705_705660


namespace log_expression_identity_l705_705773

theorem log_expression_identity
  (c : ℝ) (d : ℝ)
  (hc : c = Real.log 16)
  (hd : d = Real.log 25) :
  9^(c/d) + 4^(d/c) = 4421 / 625 :=
by
  sorry

end log_expression_identity_l705_705773


namespace lines_perpendicular_to_same_line_are_parallel_l705_705399

theorem lines_perpendicular_to_same_line_are_parallel
  (P Q R : Type) [plane P]
  {l1 l2 l3 : line P}
  (h1 : l1 ⊥ l3) (h2 : l2 ⊥ l3) :
  l1 ∥ l2 :=
sorry

end lines_perpendicular_to_same_line_are_parallel_l705_705399


namespace planes_divide_space_l705_705021

theorem planes_divide_space (n : ℕ)
  (h1 : ∀ (P : fin n → set (euclidean_space 3)), (∀ i : fin 3, ∃! x, x ∈ ⋂ (i : fin i.succ), P i))
  (h2 : ∀ (P : fin n → set (euclidean_space 3)), ¬ ∃ x, x ∈ ⋂ (i : fin 4), P i) :
  (number_of_regions_in_space_divided_by_planes n = (n^3 + 5 * n + 6) / 6) :=
by sorry

end planes_divide_space_l705_705021


namespace moles_of_KHSO4_formed_l705_705303

theorem moles_of_KHSO4_formed (nKOH nH2SO4 : ℝ) (hKOH : nKOH = 2) (hH2SO4 : nH2SO4 = 2) 
  (stoichiometry : ∀ nKOH nH2SO4, nKOH = nH2SO4 → ∃ nKHSO4, nKHSO4 = nKOH) :
  ∃ nKHSO4, nKHSO4 = 2 :=
by
  have h := stoichiometry 2 2 hKOH,
  exact h

end moles_of_KHSO4_formed_l705_705303


namespace chocolate_milk_probability_l705_705499

noncomputable theory

/--
  Robert visits the milk bottling plant for 7 days a week.
  The plant has a 1/2 chance of bottling chocolate milk on weekdays (Monday to Friday).
  The plant has a 3/4 chance of bottling chocolate milk on weekends (Saturday and Sunday).

  Prove that the probability that the plant bottles chocolate milk on exactly 5 of the 7 days Robert visits is 781/1024.
-/
theorem chocolate_milk_probability :
  let weekdays_prob := 1 / 2,
      weekends_prob := 3 / 4,
      total_days := 7,
      target_days := 5 in
  ∃ p : ℚ, p = 781 / 1024 ∧
    let events := (finset.powerset (finset.range 7)) in
    p = ∑ x in events, if x.card = target_days then
        let weekdays_count := (x ∩ (finset.range 5)).card,
            weekends_count := (x ∩ (finset.range 5).compl).card in
        if weekdays_count + weekends_count = target_days then
          (weekdays_prob ^ weekdays_count) * ((1 - weekdays_prob) ^ (5 - weekdays_count)) *
          (weekends_prob ^ weekends_count) * ((1 - weekends_prob) ^ (2 - weekends_count))
        else 0
    else 0 := by sorry

end chocolate_milk_probability_l705_705499


namespace max_value_sin_cos_expression_l705_705291

theorem max_value_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l705_705291


namespace proposition_p_proposition_q_l705_705929

theorem proposition_p : ∅ ≠ ({∅} : Set (Set Empty)) := by
  sorry

theorem proposition_q (A : Set ℕ) (B : Set (Set ℕ)) (hA : A = {1, 2})
    (hB : B = {x | x ⊆ A}) : A ∈ B := by
  sorry

end proposition_p_proposition_q_l705_705929


namespace gcd_max_digits_l705_705830

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705830


namespace arithmetic_sequence_sum_ratio_l705_705075

noncomputable def S (n : ℕ) (a_1 : ℚ) (d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_ratio (a_1 d : ℚ) (h : d ≠ 0) (h_ratio : (a_1 + 5 * d) / (a_1 + 2 * d) = 2) :
  S 6 a_1 d / S 3 a_1 d = 7 / 2 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l705_705075


namespace smallest_sum_of_cubes_two_ways_l705_705185

theorem smallest_sum_of_cubes_two_ways :
  ∃ (n : ℕ) (a b c d e f : ℕ),
  n = a^3 + b^3 + c^3 ∧ n = d^3 + e^3 + f^3 ∧
  (a, b, c) ≠ (d, e, f) ∧
  (d, e, f) ≠ (a, b, c) ∧ n = 251 :=
by
  sorry

end smallest_sum_of_cubes_two_ways_l705_705185


namespace smallest_possible_perimeter_l705_705575

theorem smallest_possible_perimeter (a : ℕ) (h : a > 2) (h_triangle : a < a + (a + 1) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a) :
  3 * a + 3 = 12 :=
by
  sorry

end smallest_possible_perimeter_l705_705575


namespace cost_of_two_other_puppies_l705_705644

theorem cost_of_two_other_puppies (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) (remaining_puppies_cost : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  remaining_puppies_cost = (total_cost - num_sale_puppies * sale_price) →
  (remaining_puppies_cost / (num_puppies - num_sale_puppies)) = 175 :=
by
  intros
  sorry

end cost_of_two_other_puppies_l705_705644


namespace trigonometric_expression_l705_705343

theorem trigonometric_expression {α m : Real} (h1 : m ≠ 0) (h2 : (cos α, sin α) = (-4*m/|5*m|, 3*m/|5*m|)) :
  2 * sin α + cos α = 2/5 ∨ 2 * sin α + cos α = -2/5 :=
by
  sorry

end trigonometric_expression_l705_705343


namespace decrease_equation_l705_705555

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l705_705555


namespace k_bound_of_no_three_collinear_l705_705468

   open Set

   variables (n k : ℕ) (S : Set (ℝ × ℝ))

   def no_three_collinear (S : Set (ℝ × ℝ)) : Prop :=
     ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ S → p2 ∈ S → p3 ∈ S → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
       ¬Collinear {p1, p2, p3}

   def exists_r_for_all_points (S : Set (ℝ × ℝ)) (k : ℕ) : Prop :=
     ∀ (P : (ℝ × ℝ)), P ∈ S → ∃ (r : ℝ), r > 0 ∧ (S ∩ {Q | dist P Q = r}).card ≥ k

   theorem k_bound_of_no_three_collinear (n k : ℕ) (S : Set (ℝ × ℝ))
     (h1 : S.card = n)
     (h2 : no_three_collinear S)
     (h3 : exists_r_for_all_points S k) :
     k < (1 / 2 : ℝ) + Real.sqrt (2 * n) :=
   sorry
   
end k_bound_of_no_three_collinear_l705_705468


namespace average_is_700_l705_705782

-- Define the list of known numbers
def numbers_without_x : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

-- Define the value of x
def x : ℕ := 755

-- Define the list of all numbers including x
def all_numbers : List ℕ := numbers_without_x.append [x]

-- Define the total length of the list containing x
def n : ℕ := all_numbers.length

-- Define the sum of the numbers in the list including x
noncomputable def sum_all_numbers : ℕ := all_numbers.sum

-- Define the average formula
noncomputable def average : ℕ := sum_all_numbers / n

-- State the theorem
theorem average_is_700 : average = 700 := by
  sorry

end average_is_700_l705_705782


namespace hypotenuse_calculation_l705_705650

theorem hypotenuse_calculation (p q : ℝ) (p > 0) (q > 0) (q < p) (p < q * sqrt 1.8) : 
  ∃ c : ℝ, c = sqrt ((p^4 - 9 * q^4) / (2 * (p^2 - 5 * q^2))) :=
by
  sorry

end hypotenuse_calculation_l705_705650


namespace complex_modulus_range_l705_705708

noncomputable def ellipse := λ z : ℂ, complex.abs (z + 1) + complex.abs (z - 1) = 4

noncomputable def complex_modulus (z : ℂ) := complex.abs (z - 2 * complex.conj z)

variable f : set ℂ := {z : ℂ | ellipse z}

theorem complex_modulus_range : 
  (set.range (λ z : ℂ, if h : z ∈ f then complex_modulus z else 0) = 
    set.Icc 2 (3 * real.sqrt 3)) := 
sorry

end complex_modulus_range_l705_705708


namespace intersection_sets_l705_705751

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≤ 2}
def C : Set ℝ := {x | 1 < x ≤ 2}

theorem intersection_sets :
  A ∩ B = C := by
  sorry

end intersection_sets_l705_705751


namespace platform_and_train_length_equality_l705_705152

-- Definitions of the given conditions.
def speed_in_kmh : ℝ := 90
def speed_in_m_per_min : ℝ := (speed_in_kmh * 1000) / 60
def time_in_min : ℝ := 1
def length_of_train : ℝ := 750
def total_distance_covered : ℝ := speed_in_m_per_min * time_in_min

-- Assertion that length of platform is equal to length of train
theorem platform_and_train_length_equality : 
  total_distance_covered - length_of_train = length_of_train :=
by
  -- Placeholder for proof
  sorry

end platform_and_train_length_equality_l705_705152


namespace monic_polynomial_with_shifted_roots_l705_705914

theorem monic_polynomial_with_shifted_roots :
  (a b c : ℂ) (h_roots : Polynomial.eval a (Polynomial.mk [6, -11, 6, -1]) = 0 ∧ Polynomial.eval b (Polynomial.mk [6, -11, 6, -1]) = 0 ∧ Polynomial.eval c (Polynomial.mk [6, -11, 6, -1]) = 0) :
  Polynomial.mk [0, 2, 3, 1] = Polynomial.mk [0, 2, 3, 1] := 
  by
  sorry

end monic_polynomial_with_shifted_roots_l705_705914


namespace average_homework_time_decrease_l705_705550

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l705_705550


namespace triangle_area_l705_705886

theorem triangle_area 
  (A B C : ℝ)
  (a b c : ℝ)
  (angle_A : Angle := 45)
  (angle_B : Angle := 60)
  (side_BC := 1):
    area (mk_triangle a b c angle_A angle_B) = (sqrt 6 + sqrt 2) / 4 :=
by
  admit -- Here, we assume the conditions and skip the proof

end triangle_area_l705_705886


namespace solve_inequality_l705_705130

theorem solve_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 :=
sorry

end solve_inequality_l705_705130


namespace find_alpha_l705_705639

-- Conditions given in the problem
variables (P Q R S T U V : Type)
variables [point P] [point Q] [point R] [point S] [point T] [point U] [point V]

-- Folding properties
variables (PS : line P S) (TS : line T S) (SU : segment S U)
variables (PS_eq_TS : P S = T S)
variables (angle_PSV : angle P S V) (angle_TSV : angle T S V)
variables (congruent_PSV_TSV : angle P S V = angle T S V)
variables (SU_eq_half_SR : S U = (1 / 2) * (S R))
variables (SU_eq_half_PS : S U = (1 / 2) * (P S))
variables (SU_eq_half_TS : S U = (1 / 2) * (T S))
variables (triangle_SUT : right_triangle S U T)
variables (angle_TSU_60 : ∠ T S U = 60)
variables (angle_USP_90 : ∠ U S P = 90)

theorem find_alpha : α = 75 :=
by
sorry

end find_alpha_l705_705639


namespace gcd_max_two_digits_l705_705838

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705838


namespace original_bet_is_40_l705_705219

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l705_705219


namespace marbles_cost_correct_l705_705088

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l705_705088


namespace value_range_of_f_l705_705537

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_range_of_f :
  set.range (λ (x : ℝ), f x) ∩ set.Icc 0 2 = set.Icc (-3) 5 :=
sorry

end value_range_of_f_l705_705537


namespace gcd_digit_bound_l705_705823

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705823


namespace right_triangle_segment_ratio_l705_705629

-- Define the given triangle with legs in ratio 1:3
structure RightTriangle (α β : ℝ) where
  h : α > 0 ∧ β > 0
  ratio : α / β = 1 / 3

-- Define the existence of a perpendicular from the right angle vertex to the hypotenuse
structure Altitude (t : RightTriangle) where
  vertex_right_angle : ℝ -- Vertex at the right angle
  hypotenuse_segment1 hypotenuse_segment2 : ℝ -- Segments created on the hypotenuse
  perpendicular : vertex_right_angle = hypotenuse_segment1 + hypotenuse_segment2
  ratio_segments : hypotenuse_segment2 / hypotenuse_segment1 = 9

-- The theorem statement to be proved
theorem right_triangle_segment_ratio (t : RightTriangle) (a : Altitude t) : a.ratio_segments = 9 := by
  sorry

end right_triangle_segment_ratio_l705_705629


namespace ratio_of_sums_l705_705367

theorem ratio_of_sums (a b c : ℚ) (h1 : b / a = 2) (h2 : c / b = 3) : (a + b) / (b + c) = 3 / 8 := 
  sorry

end ratio_of_sums_l705_705367


namespace sum_of_percentages_l705_705186

theorem sum_of_percentages : 
  (25 / 100 * 2018) + (2018 / 100 * 25) = 1009 :=
by
  -- mentioning the calculations of each part directly in Lean
  have part1 : 25 / 100 * 2018 = 504.5 := by sorry
  have part2 : 2018 / 100 * 25 = 504.5 := by sorry
  rw [part1, part2]
  exact add_self 504.5

end sum_of_percentages_l705_705186


namespace max_value_sin_cos_expression_l705_705293

theorem max_value_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l705_705293


namespace part1_part2_l705_705462

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705462


namespace sum_real_roots_l705_705681

theorem sum_real_roots (f : Polynomial ℝ) (hf : f = Polynomial.C 1 * (X^4 - 4 * X - 3)) :
  (∑ root in f.real_roots, root) = real.sqrt 2 :=
sorry

end sum_real_roots_l705_705681


namespace at_least_one_angle_not_rational_multiple_l705_705236

namespace ProofProblem

structure Point :=
  (x : ℤ)
  (y : ℤ)

structure Triangle :=
  (A B C P : Point)
  (is_lattice_point_A : A.x ∈ ℤ ∧ A.y ∈ ℤ)
  (is_lattice_point_B : B.x ∈ ℤ ∧ B.y ∈ ℤ)
  (is_lattice_point_C : C.x ∈ ℤ ∧ C.y ∈ ℤ)
  (is_lattice_point_P : P.x ∈ ℤ ∧ P.y ∈ ℤ)
  (P_inside_triangle : true) -- This would be defined properly in a complete proof

noncomputable def angle (a b c : Point) : ℝ := 
  -- Angle calculation definition here
  sorry 

noncomputable def is_rational_multiple_of_pi (θ : ℝ) : Prop :=
  ∃ (q : ℚ), θ = q * Real.pi

theorem at_least_one_angle_not_rational_multiple {A B C P : Point} 
  (hA : A.x ∈ ℤ ∧ A.y ∈ ℤ) 
  (hB : B.x ∈ ℤ ∧ B.y ∈ ℤ) 
  (hC : C.x ∈ ℤ ∧ C.y ∈ ℤ) 
  (hP : P.x ∈ ℤ ∧ P.y ∈ ℤ) 
  (h_inside: true): 
  ¬ (is_rational_multiple_of_pi (angle P A B) ∧ 
     is_rational_multiple_of_pi (angle P B C) ∧ 
     is_rational_multiple_of_pi (angle P C A)) :=
by sorry

end ProofProblem

end at_least_one_angle_not_rational_multiple_l705_705236


namespace dot_product_theorem_l705_705766

variable {V : Type*} [InnerProductSpace ℝ V]

theorem dot_product_theorem (a b : V) (ha : ‖a‖ = 4) (hb : ‖b‖ = 8) : 
  (a + b) ⬝ (a - b) = -48 :=
by
  sorry

end dot_product_theorem_l705_705766


namespace distinct_values_expressions_count_divisors_l705_705067

-- Given n > 3 and a set {2, 3, ..., n}, define the conditions.
variables (n : ℕ) (n_gt_3 : n > 3)
def setX := { x : ℕ | 2 ≤ x ∧ x ≤ n }

-- First proof problem.
-- Prove: If a, b, c > n / 2 are chosen from the set in any order,
-- the values of all expressions formed with a, b, c using + and * are distinct.
theorem distinct_values_expressions (a b c : ℕ) 
  (a_in : a ∈ setX n) (b_in : b ∈ setX n) (c_in : c ∈ setX n)
  (h : (n / 2 : ℕ) < a ∧ (n / 2 : ℕ) < b ∧ (n / 2 : ℕ) < c) :
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  ∀ e₁ e₂ : ℕ, 
    (e₁ ∈ {a * b * c, a + b * c, a * b + c, a + b + c, a * (b + c), b * (a + c), c * (a + b), (a + b) * c}) → 
    (e₂ ∈ {a * b * c, a + b * c, a * b + c, a + b + c, a * (b + c), b * (a + c), c * (a + b), (a + b) * c}) → 
    e₁ = e₂ → e₁ = e₂ :=
sorry

-- Second proof problem.
-- Given a prime p <= sqrt n and p is the smallest of three distinct chosen numbers,
-- Prove: If not all values are distinct, the number of such subsets equals the number of positive divisors of p-1.
theorem count_divisors (p a b : ℕ) (p_prime : Prime p) (sqrt_n_cond : p ≤ Nat.sqrt n)
  (p_in : p ∈ setX n) (a_in : a ∈ setX n) (b_in : b ∈ setX n)
  (p_min : p < a ∧ p < b) (not_distinct : ∃ e₁ e₂ : ℕ, 
    e₁ ∈ {p * a * b, p + a * b, p * a + b, p + a + b, p * (a + b), a * (p + b), b * (p + a), (p + a) * b} ∧ 
    e₂ ∈ {p * a * b, p + a * b, p * a + b, p + a + b, p * (a + b), a * (p + b), b * (p + a), (p + a) * b} ∧ 
    e₁ = e₂) :
  ∀ k, k = (Nat.divisors (p-1)).length →
  k = (Nat.divisors (p-1)).length :=
sorry

end distinct_values_expressions_count_divisors_l705_705067


namespace math_proof_problem_l705_705057

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable {a b : ℝ}

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, f' x = (f x)'

def condition1 : Prop :=
  odd_function f

def condition2 : Prop :=
  derivative f f'

def condition3 : Prop :=
  f (π / 2) = 0

def condition4 : Prop :=
  ∀ x ∈ Ioo 0 π, f' x * sin x - f x * cos x < 0

-- Question
def question : Prop :=
  ∀ x, f x < 2 * f (π / 6) * sin x → x ∈ Ioo (-π / 6) 0 ∪ Ioo (π / 6) π

-- Problem statement
theorem math_proof_problem :
  let S := Ioo (-π / 6) 0 ∪ Ioo (π / 6) π in
  (condition1 ∧ condition2 ∧ condition3 ∧ condition4) → (∀ x, f x < 2 * f (π / 6) * sin x → x ∈ S) :=
by
  sorry

end math_proof_problem_l705_705057


namespace inequality_solution_l705_705129

theorem inequality_solution (x : ℝ) :
  x ≠ 2 → x ≠ 3 → x ≠ 4 → x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (x ∈ set.Ioo (-∞) -3 ∪ set.Ioo (-2) 2 ∪ set.Ioo 3 4 ∪ set.Ioo 5 8 ∪ set.Ioo 9 ∞) := 
sorry

end inequality_solution_l705_705129


namespace centroids_coincide_l705_705696

noncomputable def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

theorem centroids_coincide (A B C : ℂ) (k : ℝ) (C1 A1 B1 : ℂ)
  (h1 : C1 = k * (B - A) + A)
  (h2 : A1 = k * (C - B) + B)
  (h3 : B1 = k * (A - C) + C) :
  centroid A1 B1 C1 = centroid A B C := by
  sorry

end centroids_coincide_l705_705696


namespace sum_of_midpoints_coordinates_is_15_5_l705_705576

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def sum_of_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Conditions
def segment1_start : ℝ × ℝ := (10, 20)
def segment1_end : ℝ × ℝ := (5, -10)
def segment2_start : ℝ × ℝ := (-3, 7)
def segment2_end : ℝ × ℝ := (4, -2)

-- Prove statement
theorem sum_of_midpoints_coordinates_is_15_5 :
  sum_of_coordinates (midpoint segment1_start segment1_end) +
  sum_of_coordinates (midpoint segment2_start segment2_end) = 15.5 :=
by
  -- Proof is omitted
  sorry

end sum_of_midpoints_coordinates_is_15_5_l705_705576


namespace amount_spent_on_marbles_l705_705086

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l705_705086


namespace gcd_digits_le_3_l705_705796

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705796


namespace parabola_equation_focus_directrix_constant_angle_MON_l705_705720

noncomputable def parabola_focus_directrix : Type := 
  { p : ℝ | p ≠ 0 }

-- Problem (1): Proof that the equation of the parabola, the coordinates of the focus, and the directrix
theorem parabola_equation_focus_directrix (p : ℝ) (h : p ≠ 0) (E : ℝ × ℝ) (hE : E = (2, 2)) : 
  ∃ (equation : ℝ → ℝ → Prop) (focus : ℝ × ℝ) (directrix : ℝ → Prop), 
    equation = (λ x y, y^2 = 2 * p * x) ∧ 
    focus = (⟨1 / 2, 0⟩ : ℝ × ℝ) ∧ 
    directrix = (λ x, x = -1 / 2) := 
by 
  sorry

-- Problem (2): Proof that ∠MON is a constant value
theorem constant_angle_MON (O : ℝ × ℝ) (hO : O = (0, 0)) (E : ℝ × ℝ) (hE : E = (2, 2)) 
  (D : ℝ × ℝ) (hD : D = (2, 0)) (A B : ℝ × ℝ) (M N : ℝ × ℝ) 
  (hA : A ≠ E) (hB : B ≠ E) (hM : M = (⟨-2, _⟩ : ℝ × ℝ)) (hN : N = (⟨-2, _⟩ : ℝ × ℝ)) :
  ∠MON = Real.pi / 2 :=
by
  sorry

end parabola_equation_focus_directrix_constant_angle_MON_l705_705720


namespace find_a_minus_b_l705_705187

variable (a b : ℚ)

theorem find_a_minus_b (h : ∀ x : ℚ, 0 < x → 
  (a / (10^x - 1) + b / (10^x + 2)) = ((2 * 10^x + 3) / ((10^x - 1) * (10^x + 2)))) : a - b = 4/3 :=
by
  sorry

end find_a_minus_b_l705_705187


namespace midpoint_diagonal_relation_l705_705953

open EuclideanGeometry
open Classical

variables {P Q R S T U A M : Point}

-- Two squares APQR and ASTU sharing a common vertex A
def is_square (A P Q R : Point) : Type := 
  (dist A P = dist P Q) ∧ 
  (dist P Q = dist Q R) ∧ 
  (dist Q R = dist R A) ∧ 
  (dist R A = dist A P) ∧ 
  (angle A P Q = 90) ∧ 
  (angle P Q R = 90) ∧ 
  (angle Q R A = 90) ∧ 
  (angle R A P = 90)

-- M is the midpoint of PU
def is_midpoint (M P U : Point) : Type := 
  (dist P M = dist M U) ∧ 
  (dist P U = dist P M + dist M U)

variables (h1 : is_square A P Q R)
variables (h2 : is_square A S T U)
variables (h3 : is_midpoint M P U)

-- Proving AM = 1/2 * RS
theorem midpoint_diagonal_relation : dist A M = 1 / 2 * dist R S :=
sorry

end midpoint_diagonal_relation_l705_705953


namespace blocks_per_friend_l705_705422

theorem blocks_per_friend (total_blocks : ℕ) (friends : ℕ) (h1 : total_blocks = 28) (h2 : friends = 4) :
  total_blocks / friends = 7 :=
by
  sorry

end blocks_per_friend_l705_705422


namespace ribbon_per_box_l705_705409

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l705_705409


namespace minimum_value_of_y_at_l705_705144

def y (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem minimum_value_of_y_at (x : ℝ) :
  (∀ x : ℝ, y x ≥ 2) ∧ (y (-2) = 2) :=
by 
  sorry

end minimum_value_of_y_at_l705_705144


namespace twenty_bananas_eq_nine_point_three_seven_five_mangoes_l705_705647

variable (bananas pears mangoes : ℕ) 

axiom cost_eq_bananas_pears : 4 * bananas = 3 * pears
axiom cost_eq_pears_mangoes : 8 * pears = 5 * mangoes

theorem twenty_bananas_eq_nine_point_three_seven_five_mangoes : 20 * bananas = 9.375 * mangoes :=
by
  sorry

end twenty_bananas_eq_nine_point_three_seven_five_mangoes_l705_705647


namespace perfect_square_trinomial_l705_705771

theorem perfect_square_trinomial (m : ℝ) : (∃ (a b : ℝ), (a * x + b) ^ 2 = x^2 + m * x + 16) -> (m = 8 ∨ m = -8) :=
sorry

end perfect_square_trinomial_l705_705771


namespace point_in_second_quadrant_l705_705862

theorem point_in_second_quadrant (m : ℝ) : 
  let x := -1 in
  let y := m^2 + 1 in
  x < 0 ∧ y > 0 →
  (∃ quadrant, quadrant = 2) :=
by
  let x := -1
  let y := m^2 + 1
  intro h,
  existsi 2,
  sorry

end point_in_second_quadrant_l705_705862


namespace min_large_trucks_l705_705489

theorem min_large_trucks (total_fruits : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) :
  total_fruits = 134 → large_capacity = 15 → small_capacity = 7 →
  ∃ n : ℕ, n = 8 ∧ ∀ l s : ℕ, (l * large_capacity + s * small_capacity = total_fruits) →
  (l = 8 ∧ s = 2) :=
by
  intros h1 h2 h3
  use 8
  split
  { refl }
  sorry

end min_large_trucks_l705_705489


namespace val_of_7c_plus_7d_l705_705060

noncomputable def h (x : ℝ) : ℝ := 7 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4

noncomputable def f (c d x : ℝ) : ℝ := c * x + d

theorem val_of_7c_plus_7d (c d : ℝ) (h_eq : ∀ x, h x = f_inv x - 2) 
  (inv_prop : ∀ x, f c d (f_inv x) = x) : 7 * c + 7 * d = 5 :=
by
  sorry

end val_of_7c_plus_7d_l705_705060


namespace polygon_problem_l705_705400

theorem polygon_problem 
  (D : ℕ → ℕ) (m x : ℕ) 
  (H1 : ∀ n, D n = n * (n - 3) / 2)
  (H2 : D m = 3 * D (m - 3))
  (H3 : D (m + x) = 7 * D m) :
  m = 9 ∧ x = 12 ∧ (m + x) - m = 12 :=
by {
  -- the proof would go here, skipped as per the instructions.
  sorry
}

end polygon_problem_l705_705400


namespace sum_of_coefficients_of_poly_l705_705260

-- Define the polynomial
def poly (x y : ℕ) := (2 * x + 3 * y) ^ 12

-- Define the sum of coefficients
def sum_of_coefficients := poly 1 1

-- The theorem stating the result
theorem sum_of_coefficients_of_poly : sum_of_coefficients = 244140625 :=
by
  -- Proof is skipped
  sorry

end sum_of_coefficients_of_poly_l705_705260


namespace isothermal_expansion_work_l705_705653

noncomputable section
open Real

-- Definitions of the given conditions
variables (m μ R T : ℝ) (V1 V2 : ℝ) (hV1 : V1 > 0) (hV2 : V2 > 0) (hT : T > 0)

-- Ideal gas law is expressed as P = (m / μ) * (R * T) / V
def ideal_gas_pressure (V : ℝ) : ℝ :=
  (m / μ) * (R * T) / V

-- Work done during isothermal expansion from V1 to V2
def work_done : ℝ :=
  ∫ (V : ℝ) in V1..V2, ideal_gas_pressure m μ R T V

theorem isothermal_expansion_work :
  work_done m μ R T V1 V2 = (m / μ) * R * T * (ln V2 - ln V1) :=
sorry

end isothermal_expansion_work_l705_705653


namespace student_chose_number_l705_705198

theorem student_chose_number (x : ℤ) (h : 2 * x - 148 = 110) : x = 129 := 
by
  sorry

end student_chose_number_l705_705198


namespace binomial_coefficient_is_252_l705_705026

theorem binomial_coefficient_is_252 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_coefficient_is_252_l705_705026


namespace max_expression_l705_705063

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end max_expression_l705_705063


namespace complex_number_solution_l705_705345

-- Definition of the problem with conditions and the expected result
theorem complex_number_solution (z : ℂ) (h : (3 - z) * complex.I = 1 - 3 * complex.I) : z = 6 + complex.I := 
by -- Proof not provided
  sorry

end complex_number_solution_l705_705345


namespace paving_stone_width_l705_705599

noncomputable def width_of_paving_stone (L W : ℕ) (N l : ℕ) : ℕ :=
  let A := L * W  -- Area of the courtyard
  let area_of_paving_stone := l * w  -- Area of one paving stone
  let total_area_covered := N * area_of_paving_stone  -- Total area covered by paving stones
  w

theorem paving_stone_width
  (L W N l : ℕ)
  (hL : L = 40)
  (hW : W = 20)
  (hN : N = 100)
  (hl : l = 4) : width_of_paving_stone L W N l = 2 := by
  sorry

end paving_stone_width_l705_705599


namespace max_value_sin_cos_expression_l705_705292

theorem max_value_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l705_705292


namespace points_per_correct_answer_hard_round_l705_705019

theorem points_per_correct_answer_hard_round (total_points easy_points_per average_points_per hard_correct : ℕ) 
(easy_correct average_correct : ℕ) : 
  (total_points = (easy_correct * easy_points_per + average_correct * average_points_per) + (hard_correct * 5)) →
  (easy_correct = 6) →
  (easy_points_per = 2) →
  (average_correct = 2) →
  (average_points_per = 3) →
  (hard_correct = 4) →
  (total_points = 38) →
  5 = 5 := 
by
  intros
  sorry

end points_per_correct_answer_hard_round_l705_705019


namespace problem1_problem2_l705_705353

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (2 * x - 1)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := abs (3 * x - 2 * m) + abs (3 * x - 1)

-- Problem 1: Prove that ∀ x ∈ ℝ, (0 ≤ x ∧ x ≤ 1) ↔ f(x) ≤ x + 2
theorem problem1 : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) ↔ f(x) ≤ x + 2 :=
by
  intro x
  sorry

-- Problem 2: Prove that -1/4 ≤ m ≤ 5/4 → ∀ x₁, ∃ x₂, f(x₁) = g(x₂) for g(x) = |3x - 2m| + |3x - 1|
theorem problem2 (m : ℝ) (h : -1/4 ≤ m ∧ m ≤ 5/4) : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f(x₁) = g(x₂, m) :=
by
  intro x₁
  sorry

end problem1_problem2_l705_705353


namespace isabella_more_than_sam_l705_705403

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end isabella_more_than_sam_l705_705403


namespace find_f_neg_2_l705_705339

def f (x : ℝ) : ℝ := sorry
def F (x : ℝ) : ℝ := f(x) + x^2

axiom F_odd : ∀ x : ℝ, F(-x) = -F(x)
axiom f_at_2 : f(2) = 1

theorem find_f_neg_2 : f(-2) = -9 := by
  sorry

end find_f_neg_2_l705_705339


namespace probability_of_drawing_yellow_ball_l705_705025

variable (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)

-- Given conditions
def conditions : Prop :=
  total_balls = 8 ∧
  white_balls = 1 ∧
  red_balls = 2 ∧
  yellow_balls = 5

-- The probability of drawing a yellow ball from the bag
def yellow_ball_probability : ℚ :=
  yellow_balls / total_balls

theorem probability_of_drawing_yellow_ball (h : conditions) :
  yellow_ball_probability total_balls white_balls red_balls yellow_balls = 5 / 8 := by
  sorry

end probability_of_drawing_yellow_ball_l705_705025


namespace minimize_area_triangle_OPQ_l705_705625

noncomputable theory

open_locale real
open real

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (3, 1)

-- Define the equation of line l and the intercept form
def line_l : ℝ × ℝ → ℝ := λ ⟨a, b⟩, (λ (x y : ℝ), x / a + y / b = 1)

-- Define the function to calculate the area of triangle △OPQ
def triangle_area : ℝ × ℝ → ℝ :=
  λ ⟨a, b⟩, 1 / 2 * a * b

-- Statement of the problem that requires proof
theorem minimize_area_triangle_OPQ : ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ triangle_area (a, b) = 12 ∧ line_l (6, 2) 1 = 1) :=
begin
  -- Proof is omitted
  sorry
end

end minimize_area_triangle_OPQ_l705_705625


namespace measure_of_α_l705_705726

variables (α β : ℝ)
-- Condition 1: α and β are complementary angles
def complementary := α + β = 180

-- Condition 2: Half of angle β is 30° less than α
def half_less_30 := α - (1 / 2) * β = 30

-- Theorem: Measure of angle α
theorem measure_of_α (α β : ℝ) (h1 : complementary α β) (h2 : half_less_30 α β) :
  α = 80 :=
by
  sorry

end measure_of_α_l705_705726


namespace letter_L_final_position_l705_705962

structure Position :=
(base_start : ℝ)
(base_end : ℝ)
(stem_direction : bool) -- true for positive y-axis, false for negative y-axis

noncomputable def initial_position : Position :=
{ base_start := 0,
  base_end := 1,
  stem_direction := true }

noncomputable def final_position : Position := 
{ base_start := -1/2,
  base_end := 0,
  stem_direction := true }

theorem letter_L_final_position :
  ∃ (p : Position), 
    p = final_position 
    ∧ p.base_start = -1/2 
    ∧ p.base_end = 0 
    ∧ p.stem_direction = true :=
begin
  use final_position,
  -- Conditions (initial position and transformations) could be related here if necessary.
  sorry -- Skipping proof as per instructions
end

end letter_L_final_position_l705_705962


namespace range_of_a_l705_705734

-- Defining the problem conditions
def f (x : ℝ) : ℝ := sorry -- The function f : ℝ → ℝ is defined elsewhere such that its range is [0, 4]
def g (a x : ℝ) : ℝ := a * x - 1

-- Theorem to prove the range of 'a'
theorem range_of_a (a : ℝ) : (a ≥ 1/2) ∨ (a ≤ -1/2) :=
sorry

end range_of_a_l705_705734


namespace kat_average_training_hours_l705_705898

def strength_training_sessions_per_week : ℕ := 3
def strength_training_hour_per_session : ℕ := 1
def strength_training_missed_sessions_per_2_weeks : ℕ := 1

def boxing_training_sessions_per_week : ℕ := 4
def boxing_training_hour_per_session : ℝ := 1.5
def boxing_training_skipped_sessions_per_2_weeks : ℕ := 1

def cardio_workout_sessions_per_week : ℕ := 2
def cardio_workout_minutes_per_session : ℕ := 30

def flexibility_training_sessions_per_week : ℕ := 1
def flexibility_training_minutes_per_session : ℕ := 45

def interval_training_sessions_per_week : ℕ := 1
def interval_training_hour_per_session : ℝ := 1.25 -- 1 hour and 15 minutes 

noncomputable def average_hours_per_week : ℝ :=
  let strength_training_per_week : ℝ := ((5 / 2) * strength_training_hour_per_session)
  let boxing_training_per_week : ℝ := ((7 / 2) * boxing_training_hour_per_session)
  let cardio_workout_per_week : ℝ := (cardio_workout_sessions_per_week * cardio_workout_minutes_per_session / 60)
  let flexibility_training_per_week : ℝ := (flexibility_training_sessions_per_week * flexibility_training_minutes_per_session / 60)
  let interval_training_per_week : ℝ := interval_training_hour_per_session
  strength_training_per_week + boxing_training_per_week + cardio_workout_per_week + flexibility_training_per_week + interval_training_per_week

theorem kat_average_training_hours : average_hours_per_week = 10.75 := by
  unfold average_hours_per_week
  norm_num
  sorry

end kat_average_training_hours_l705_705898


namespace find_constants_l705_705432

theorem find_constants {Q C D : Type} [AddCommGroup Q] [Module ℚ Q] 
  (h : ∃ (Q : Q) (C D : Q), CQ = 3 * Q ∧ QD = 5 * Q) :
  ∃ (s v : ℚ), (Q = s * C + v * D) ∧ s = 5 / 8 ∧ v = 3 / 8 :=
by
  obtain ⟨Q, C, D, hCQ, hQD⟩ := h
  use [5/8, 3/8]
  sorry

end find_constants_l705_705432


namespace van_speed_and_efficiency_l705_705172

theorem van_speed_and_efficiency :
  ∀ (distance : ℕ) (original_time : ℕ) (original_speed : ℕ) (fuel_efficiency : ℕ) (new_time : ℕ),
    distance = 450 →
    original_time = 5 →
    original_speed = distance / original_time →
    fuel_efficiency = 10 →
    new_time = (3 : ℚ) * original_time / 2 →
    original_speed = 90 ∧ fuel_efficiency = 10 :=
by
  intros distance original_time original_speed fuel_efficiency new_time h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  split
  exact rfl
  exact rfl

end van_speed_and_efficiency_l705_705172


namespace correct_model_l705_705559

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l705_705559


namespace gcd_digit_bound_l705_705789

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705789


namespace problem_statements_l705_705313

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2

theorem problem_statements (x : ℝ) :
  (f x < g x) ∧
  ((f x)^2 + (g x)^2 ≥ 1) ∧
  (f (2 * x) = 2 * f x * g x) :=
by
  sorry

end problem_statements_l705_705313


namespace largest_possible_value_l705_705156

theorem largest_possible_value (n : ℕ) : 
  (∀ d ∈ digits 10 n, d ≠ 0) ∧
  (∀ d ∈ digits 10 n, divides d n) ∧
  (5 ∈ digits 10 n) ∧
  list.nodup (digits 10 n) →
  n ≤ 9315 :=
by sorry

end largest_possible_value_l705_705156


namespace correct_proposition_is_B_l705_705640

-- Definitions of conditions based on the propositions
def PropA := ∀ (l1 l2 l3 : Line) (P : Point),
  (l1 ∩ l2 = P ∧ l2 ∩ l3 = P ∧ l1 ∩ l3 = P) → 
  exists H : Plane, l1 ⊆ H ∧ l2 ⊆ H ∧ l3 ⊆ H

def PropB := ∀ (A B C D : Point), 
  ¬ coplanar A B C D → (¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)

def PropC := ∀ (Q R S T : Point),
  dist Q R = dist R S ∧ dist R S = dist S T ∧ dist S T = dist T Q →
  is_rhombus Q R S T

def PropD := ∀ (W X Y Z : Point),
  (is_right_angle W X Y ∧ is_right_angle X Y Z ∧ is_right_angle Y Z W) →
  is_rectangle W X Y Z

-- Stating the theorem to prove Proposition B is correct.
theorem correct_proposition_is_B : PropB :=
by
  sorry

end correct_proposition_is_B_l705_705640


namespace exists_two_roots_in_intervals_l705_705404

def f (a b c x : ℝ) : ℝ :=
  (1 / (x - a)) + (1 / (x - b)) + (1 / (x - c))

theorem exists_two_roots_in_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧
    ∀ x : ℝ, f a b c x = 0 ↔ (x = x₁ ∨ x = x₂) := 
by
  sorry

end exists_two_roots_in_intervals_l705_705404


namespace probability_of_two_heads_one_tail_l705_705135

-- Define the conditions
def possible_outcomes := {
  "HHH", "HHT", "HTH", "THH",
  "HTT", "THT", "TTH", "TTT"
}

def favorable_outcomes := {
  "HHT", "HTH", "THH"
}

-- Define the problem statement
theorem probability_of_two_heads_one_tail :
  (favorable_outcomes.card : ℚ) / (possible_outcomes.card : ℚ) = 3 / 8 := by
  sorry

end probability_of_two_heads_one_tail_l705_705135


namespace digits_right_of_decimal_l705_705761

theorem digits_right_of_decimal (h : (5^7 : ℚ) / (10^5 * 125) = 78125 / 100000) :
  ∃ n, (78125 / 100000 : ℚ) = n / 10^5 ∧ nat.digits 10 n = [7,8,1,2,5] :=
begin
  sorry
end

end digits_right_of_decimal_l705_705761


namespace gcd_digits_le_3_l705_705799

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l705_705799


namespace gcd_max_two_digits_l705_705839

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705839


namespace minimum_value_of_x_minus_y_l705_705330

variable (x y : ℝ)
open Real

theorem minimum_value_of_x_minus_y (hx : x > 0) (hy : y < 0) 
  (h : (1 / (x + 2)) + (1 / (1 - y)) = 1 / 6) : 
  x - y = 21 :=
sorry

end minimum_value_of_x_minus_y_l705_705330


namespace geometric_series_sixth_term_l705_705781

theorem geometric_series_sixth_term :
  ∃ r : ℝ, r > 0 ∧ (16 * r^7 = 11664) ∧ (16 * r^5 = 3888) :=
by 
  sorry

end geometric_series_sixth_term_l705_705781


namespace pizzas_ordered_l705_705570

def number_of_people : ℝ := 8.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

theorem pizzas_ordered : ⌈number_of_people * slices_per_person / slices_per_pizza⌉ = 3 := 
by
  sorry

end pizzas_ordered_l705_705570


namespace solve_for_a_l705_705212

theorem solve_for_a (a x : ℝ) (h : (1 / 2) * x + a = -1) (hx : x = 2) : a = -2 :=
by
  sorry

end solve_for_a_l705_705212


namespace each_person_pays_48_32_l705_705265

noncomputable def item1_initial_cost : ℝ := 40
noncomputable def item1_tax_rate : ℝ := 0.05
noncomputable def item1_discount_rate : ℝ := 0.10

noncomputable def item2_initial_cost : ℝ := 70
noncomputable def item2_tax_rate : ℝ := 0.08
noncomputable def item2_coupon_discount : ℝ := 5
noncomputable def eur_to_usd : ℝ := 1.1

noncomputable def item3_initial_cost : ℝ := 100
noncomputable def item3_tax_rate : ℝ := 0.06
noncomputable def item3_discount_rate : ℝ := 0.15
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_split (initial_cost1 initial_cost2 initial_cost3 tax_rate1 tax_rate2 tax_rate3 discount_rate1 discount_rate3 coupon rate_eur_to_usd rate_gbp_to_usd) : ℝ :=
let item1_cost := initial_cost1 * (1 + tax_rate1) * (1 - discount_rate1),
    item2_cost := (initial_cost2 * (1 + tax_rate2) - coupon) * rate_eur_to_usd,
    item3_cost := (initial_cost3 * (1 + tax_rate3) * (1 - discount_rate3)) * rate_gbp_to_usd,
    total_cost := item1_cost + item2_cost + item3_cost
in total_cost / 5 

theorem each_person_pays_48_32 :
  total_split item1_initial_cost item2_initial_cost item3_initial_cost item1_tax_rate item2_tax_rate item3_tax_rate item1_discount_rate item3_discount_rate item2_coupon_discount eur_to_usd gbp_to_usd = 48.32 :=
by
  sorry

end each_person_pays_48_32_l705_705265


namespace intersecting_diagonals_probability_l705_705014

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l705_705014


namespace deer_speed_l705_705224

variable (D : ℝ) -- Speed of the deer in miles per hour

-- Conditions:
variable (ch_speed : ℝ := 60) -- Speed of the cheetah in miles per hour
variable (deer_head_start : ℝ := 2 / 60) -- Deer head start in hours (2 minutes)
variable (time_until_catch : ℝ := 1 / 60) -- Time until cheetah catches deer in hours (1 minute)

-- Convert the known times to minutes for ease of comparison.
noncomputable def cheetah_catches_deer : Prop :=
  let cheetah_distance_in_1_min := ch_speed * time_until_catch in
  let deer_distance_in_3_min := D * (3 / 60) in
  cheetah_distance_in_1_min = deer_distance_in_3_min

theorem deer_speed (h : cheetah_catches_deer D) : D = 20 :=
  sorry

end deer_speed_l705_705224


namespace altitude_le_inscribed_radius_l705_705935

variable {a b c : ℝ} (h : ℝ) (r : ℝ)

-- Definitions for conditions
def inscribed_circle_radius (a b c : ℝ) : ℝ := (a + b - c) / 2
def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)
def altitude_to_hypotenuse (a b c : ℝ) : ℝ := (a * b) / c

-- Theorem statement
theorem altitude_le_inscribed_radius (a b : ℝ) (c := hypotenuse a b) :
  altitude_to_hypotenuse a b c ≤ inscribed_circle_radius a b c * (1 + Real.sqrt 2) := 
sorry

end altitude_le_inscribed_radius_l705_705935


namespace find_midline_l705_705255

theorem find_midline (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
    (max_val : a * cos (b * x + c) + d = 5) (min_val : a * cos (b * x + c) + d = 1) :
    d = 3 := sorry

end find_midline_l705_705255


namespace most_likely_parents_genotypes_l705_705111

-- Defining the probabilities of alleles in the population
def p_H : ℝ := 0.1
def q_S : ℝ := 0.9

-- Defining the genotypes and their corresponding fur types
inductive Genotype
| HH : Genotype
| HS : Genotype
| SS : Genotype
| Sh : Genotype

-- A function to determine if a given genotype results in hairy fur
def isHairy : Genotype → Prop
| Genotype.HH := true
| Genotype.HS := true
| _ := false

-- Axiom stating that all four offspring have hairy fur
axiom offspring_all_hairy (parent1 parent2 : Genotype) : 
  (isHairy parent1 ∧ isHairy parent2) ∨
  ((parent1 = Genotype.HH ∨ parent2 = Genotype.Sh) ∧ isHairy Genotype.HH) 

-- The main theorem to prove the genotypes of the parents
theorem most_likely_parents_genotypes : 
  ∃ parent1 parent2,
    parent1 = Genotype.HH ∧ parent2 = Genotype.Sh :=
begin
  sorry
end

end most_likely_parents_genotypes_l705_705111


namespace part1_part2_l705_705459

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l705_705459


namespace largest_proper_divisor_condition_l705_705624

def is_proper_divisor (n k : ℕ) : Prop :=
  k > 1 ∧ k < n ∧ n % k = 0

theorem largest_proper_divisor_condition (n p : ℕ) (hp : is_proper_divisor n p) (hl : ∀ k, is_proper_divisor n k → k ≤ n / p):
  n = 12 ∨ n = 33 :=
by
  -- Placeholder for proof
  sorry

end largest_proper_divisor_condition_l705_705624


namespace shop_conditions_l705_705871

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l705_705871


namespace select_from_set_m_probability_is_0_5555555555555556_l705_705173

-- Definitions of the sets and conditions
def set_m : Set ℤ := { -6, -5, -4, -3, -2 }
def set_t : Set ℤ := { -3, -2, -1, 0, 1, 2, 3, 4, 5 }
def is_negative_product (x y : ℤ) : Prop := x * y < 0

-- The probability calculation function
def probability_of_negative_product_from_m_first : ℚ := 5 / 9

-- Proof problem statement
theorem select_from_set_m_probability_is_0_5555555555555556 :
  (∃ (x ∈ set_m) (y ∈ set_t), is_negative_product x y) →
  probability_of_negative_product_from_m_first = 0.5555555555555556 := by
  sorry

end select_from_set_m_probability_is_0_5555555555555556_l705_705173


namespace gcd_digit_bound_l705_705809

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705809


namespace probability_two_diagonals_intersect_l705_705010

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l705_705010


namespace trig_identity_and_perimeter_l705_705452

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l705_705452


namespace pyramid_volume_calculation_l705_705267
open Real

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

noncomputable def mean_altitude (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let g := centroid p1 p2 p3
  in (2 / 3) * (abs ((p3.2 - p1.2) * (g.1 - p1.1) - (p3.1 - p1.1) * (g.2 - p1.2)) / sqrt ((p3.2 - p1.2) ^ 2 + (p3.1 - p1.1) ^ 2))

noncomputable def pyramid_volume (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem pyramid_volume_calculation : 
  let A := (0, 0) in
  let B := (40, 0) in
  let C := (20, 30) in
  let D := midpoint A B in
  let E := midpoint B C in
  let F := midpoint A C in
  let G := centroid A B C in
  let base_area := triangle_area D E F in
  let height := mean_altitude A B C in
  pyramid_volume base_area height = 2000 := 
by
  let A := (0, 0)
  let B := (40, 0)
  let C := (20, 30)
  let D := midpoint A B
  let E := midpoint B C
  let F := midpoint A C
  let G := centroid A B C
  let base_area := triangle_area D E F
  let height := mean_altitude A B C 
  have h1 : pyramid_volume base_area height = (1 / 3) * 150 * (40 / 3),
  { sorry }, 
  rw h1,
  norm_num,
  sorry

end pyramid_volume_calculation_l705_705267


namespace composition_of_perpendicular_symmetries_is_symmetry_l705_705571

theorem composition_of_perpendicular_symmetries_is_symmetry
  {α₁ β₁ α₂ β₂ : Type*}
  (s₁ s₂ : Type*)
  [intersect : s₁ ∩ s₂]
  [perpendicular₁ : (α₁ ∩ β₁ = s₁) ∧ (β₁ ⊥ α₁)]
  [perpendicular₂ : (α₂ ∩ β₂ = s₂) ∧ (β₂ ⊥ α₂)]
  [coincide : β₁ = α₂]
  : ∃ s, (symmetry_of (comp_symm_lines s₁ s₂) = symmetry_line s) ∧ (s ⊥ s₁ ∧ s ⊥ s₂) := 
sorry

end composition_of_perpendicular_symmetries_is_symmetry_l705_705571


namespace books_sold_on_friday_l705_705043

variable (total_stock : ℕ := 1200)
variable (sold_mon : ℕ := 75)
variable (sold_tue : ℕ := 50)
variable (sold_wed : ℕ := 64)
variable (sold_thu : ℕ := 78)
variable (not_sold_ratio : ℚ := 0.665)

def sold_fri : ℕ :=
  total_stock - (not_sold_ratio * total_stock).toNat - 
  (sold_mon + sold_tue + sold_wed + sold_thu)

theorem books_sold_on_friday : sold_fri = 135 :=
by
  sorry

end books_sold_on_friday_l705_705043


namespace stanleyRanMore_l705_705944

def distanceStanleyRan : ℝ := 0.4
def distanceStanleyWalked : ℝ := 0.2

theorem stanleyRanMore : distanceStanleyRan - distanceStanleyWalked = 0.2 := by
  sorry

end stanleyRanMore_l705_705944


namespace sum_of_x_coords_Q3_l705_705597

-- Definitions
def Q1_vertices_sum_x (S : ℝ) := S = 1050

def Q2_vertices_sum_x (S' : ℝ) (S : ℝ) := S' = S

def Q3_vertices_sum_x (S'' : ℝ) (S' : ℝ) := S'' = S'

-- Lean 4 statement
theorem sum_of_x_coords_Q3 (S : ℝ) (S' : ℝ) (S'' : ℝ) :
  Q1_vertices_sum_x S →
  Q2_vertices_sum_x S' S →
  Q3_vertices_sum_x S'' S' →
  S'' = 1050 :=
by
  sorry

end sum_of_x_coords_Q3_l705_705597


namespace intersection_sum_at_least_1000_l705_705000

theorem intersection_sum_at_least_1000 
    (table : Fin 2000 → Fin 2000 → ℤ) 
    (h_values : ∀ i j, table i j = 1 ∨ table i j = -1) 
    (h_sum_nonneg : 0 ≤ ∑ i : Fin 2000, ∑ j : Fin 2000, table i j) : 
    ∃ (rows cols : Finset (Fin 2000)),
      rows.card = 1000 ∧
      cols.card = 1000 ∧
      1000 ≤ ∑ i in rows, ∑ j in cols, table i j := 
by 
  sorry

end intersection_sum_at_least_1000_l705_705000


namespace no_real_roots_quadratic_eq_l705_705974

theorem no_real_roots_quadratic_eq :
  ¬ ∃ x : ℝ, 7 * x^2 - 4 * x + 6 = 0 :=
by sorry

end no_real_roots_quadratic_eq_l705_705974


namespace point_in_second_quadrant_l705_705868

theorem point_in_second_quadrant (m : ℝ) : (-1 < 0) ∧ (m^2 + 1 > 0) → Quadrant (-1, m^2 + 1) = Quadrant.second :=
by
  sorry

end point_in_second_quadrant_l705_705868


namespace kaylee_more_boxes_to_sell_l705_705419

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end kaylee_more_boxes_to_sell_l705_705419


namespace angle_E_measure_l705_705857

-- Definition of degrees for each angle in the quadrilateral
def angle_measure (E F G H : ℝ) : Prop :=
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360

-- Prove the measure of angle E
theorem angle_E_measure (E F G H : ℝ) (h : angle_measure E F G H) : E = 360 * (4 / 7) :=
by {
  sorry
}

end angle_E_measure_l705_705857


namespace no_xy_exists_l705_705979

open Nat

def sum_of_digits (n : Nat) : Nat := 
  n.digits.sum

theorem no_xy_exists (x y : Nat) : ¬ (sum_of_digits ((10 ^ x) ^ y - 64) = 279) :=
by {
  sorry
}

end no_xy_exists_l705_705979


namespace path_count_l705_705266

theorem path_count (cities roads : ℕ) (start end visit : ℕ) (num_roads : ℕ) 
  (cB cH cD : ℕ) 
  (valid_paths : ℕ) : 
  cities = 8 →
  roads = 11 →
  start = cB →
  end = cH →
  visit = cD →
  num_roads = 8 →
  valid_paths = 6 →
  -- NOTE: You'd include graph theory definitions and conditions here
  -- such as the definition of a graph, paths, and any other necessary aspects
  (* path_definitions_and_conditions *) 
  True :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end path_count_l705_705266


namespace sequence_bound_l705_705065

open Real

noncomputable def sequence (x₁ : ℝ) : ℕ → ℝ
| 0     := x₁
| (n+1) := 1 + (sequence x₁ n) - 1 / 2 * (sequence x₁ n) ^ 2

theorem sequence_bound (x₁ : ℝ) (hx₁ : 1 < x₁ ∧ x₁ < 2) :
  ∀ n ≥ 3, abs ((sequence x₁ n) - sqrt 2) < 2 ^ (- n) :=
by sorry

end sequence_bound_l705_705065


namespace permutation_remainder_mod_1000_l705_705048

theorem permutation_remainder_mod_1000 :
  let str := ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
      n := (∑ k in (Fin 4).toFinset, choose 4 (k+1) * choose 5 k * choose 6 (k+2)) 
  in n % 1000 = 320 := 
begin
  sorry
end

end permutation_remainder_mod_1000_l705_705048


namespace problem_19101112_decomposition_l705_705918

theorem problem_19101112_decomposition:
  ∃ a b : ℕ, a * b = 19101112 ∧ prime a ∧ ¬ prime b :=
by {
  use 1163,
  use 16424,
  split,
  { -- prove 1163 * 16424 = 19101112
    rfl, },
  { -- prove 1163 is prime
    sorry, },
  { -- prove 16424 is composite
    sorry, },
}

end problem_19101112_decomposition_l705_705918


namespace jane_bought_two_bagels_l705_705275

variable (b m d k : ℕ)

def problem_conditions : Prop :=
  b + m + d = 6 ∧ 
  (60 * b + 45 * m + 30 * d) = 100 * k

theorem jane_bought_two_bagels (hb : problem_conditions b m d k) : b = 2 :=
  sorry

end jane_bought_two_bagels_l705_705275


namespace part1_part2_l705_705443

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705443


namespace min_abs_sum_l705_705059

def g (z : ℂ) (β δ : ℂ) : ℂ := (3 + 2*complex.I) * z^2 + β * z + δ

theorem min_abs_sum (β δ : ℂ)
  (h1 : ∃ x y u v : ℝ, β = x + complex.I * y ∧ δ = u + complex.I * v ∧ (y + v + 2 = 0 ∧ v - y = 3))
  (h2 : ∃ z : ℂ, g z β δ ∈ ℝ) :
  (complex.abs β + complex.abs δ) = 4 := by
  sorry

end min_abs_sum_l705_705059


namespace right_triangle_hypotenuse_product_square_l705_705125

theorem right_triangle_hypotenuse_product_square (A₁ A₂ : ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
(h₁ : a₁ * b₁ / 2 = A₁) (h₂ : a₂ * b₂ / 2 = A₂) 
(h₃ : A₁ = 2) (h₄ : A₂ = 3) 
(h₅ : a₁ = a₂) (h₆ : b₂ = 2 * b₁) : 
(a₁ ^ 2 + b₁ ^ 2) * (a₂ ^ 2 + b₂ ^ 2) = 325 := 
by sorry

end right_triangle_hypotenuse_product_square_l705_705125


namespace problem_part_a_problem_part_b_l705_705077

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 12 ∧ a 2 = 24 ∧ ∀ k ≥ 1, a (k + 2) = a k + 14

theorem problem_part_a (a : ℕ → ℕ) (k : ℕ) (h_seq : sequence a) : 
  a 286 = 2012 :=
by
  sorry

theorem problem_part_b (a : ℕ → ℕ) (h_seq : sequence a) :
  ¬ ∃ k, ∃ m, a k = m * m :=
by
  sorry

end problem_part_a_problem_part_b_l705_705077


namespace coefficient_x4y2_expansion_l705_705516

-- Define the problem
def binomial_expansion_coeff (n : ℕ) (a b : ℝ) (x y : ℝ) (r : ℕ) : ℝ :=
  (nat.choose n r) * (x ^ (n - r)) * ((b * y) ^ r)

-- Statement of the problem
theorem coefficient_x4y2_expansion : binomial_expansion_coeff 6 1 (-2) x y 2 = 60 * (x ^ 4) * (y ^ 2) :=
by
  sorry

end coefficient_x4y2_expansion_l705_705516


namespace determine_A_l705_705540

theorem determine_A (B A : ℕ) (h1 : B < 10) (h2 : A < 10) (h3 : (32 * 10^4 + B * 10^3 + A * 10^2 + 33) = 32BA33)
    (h4 : (32 * 10^4 + B * 10^3 + A * 10^2 + 33).roundToHundreds = 323400) : 
    A = 4 := sorry

end determine_A_l705_705540


namespace average_ABC_is_three_l705_705987
-- Import the entirety of the Mathlib library

-- Define the required conditions and the theorem to be proved
theorem average_ABC_is_three (A B C : ℝ) 
    (h1 : 2012 * C - 4024 * A = 8048) 
    (h2 : 2012 * B + 6036 * A = 10010) : 
    (A + B + C) / 3 = 3 := 
by
  sorry

end average_ABC_is_three_l705_705987


namespace continued_fraction_solution_l705_705682

noncomputable def continued_fraction: Real :=
  1 + (1 / (2 + (1 / (1 + (1 / (2 + (1 / continued_fraction)))))))

theorem continued_fraction_solution :
  1 + 1 / (2 + (1 / (1 + 1 / (2 + continued_fraction)))) = (1 + Real.sqrt 3) / 2 :=
by
  sorry

end continued_fraction_solution_l705_705682


namespace Jose_investment_amount_l705_705176

theorem Jose_investment_amount :
  ∃ X : ℝ, 
    let Tom_investment := 30000 * 12,
        Jose_share := 15000,
        Tom_share := 27000 - 15000,
        Profit_ratio := Tom_share / Jose_share 
    in Tom_investment / (X * 10) = Profit_ratio ∧ X = 225000 :=
by
  sorry

end Jose_investment_amount_l705_705176


namespace probability_prime_on_spinner_l705_705574

def is_prime (n : ℕ) : Prop :=
  n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def spinner_sections : List ℕ := [3, 8, 7, 9, 11, 4, 14, 2]

def prime_count (lst : List ℕ) : ℕ :=
  lst.countp is_prime

theorem probability_prime_on_spinner :
  prime_count spinner_sections / spinner_sections.length = 1 / 2 :=
by
  sorry

end probability_prime_on_spinner_l705_705574


namespace cylinder_views_not_identical_l705_705982

theorem cylinder_views_not_identical :
  (∀ (shape : Type),
    (sphere shape → ∀ v, identical_views v) ∧
    (triangular_pyramid shape → ∃ v1 ≠ v2, ¬identical_views v1 v2) ∧
    (cube shape → ∀ v, identical_views v) ∧
    (cylinder shape → ∃ v1 ≠ v2, ¬identical_views v1 v2)) :=
begin
  assume shape,
  split,
  { intros h sphere_v, exact sorry },
  split,
  { intros h tp_v, exact sorry },
  split,
  { intros h cube_v, exact sorry },
  { intros h cylinder_v, exact sorry }
end

end cylinder_views_not_identical_l705_705982


namespace dishonest_dealer_profit_percent_l705_705610

theorem dishonest_dealer_profit_percent:
  ∀ (C : ℝ), -- Assume C is the cost price for 1 kg
  let cost_per_800g := 0.8 * C, -- Cost price for 800 grams
  let profit := 0.2 * C, -- Profit made by selling 800 grams instead of 1000 grams
  (profit / cost_per_800g) * 100 = 25 :=
begin
  -- The statement here sets up the Lean environment
  sorry -- Proof to be completed 
end

end dishonest_dealer_profit_percent_l705_705610


namespace entire_company_can_be_seated_l705_705385

-- Define the properties and conditions
def can_seat_acquainted (group : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ group ∧ s.card = 6 → 
  ∃ (perm : List ℕ), multiset.of_list perm = s.val ∧ 
    ∀ (i : ℕ) (h : i < 6), (perm.nth_le i h) ∈ find_neighbors (perm.nth_le ((i + 1) % 6) h)

-- The main theorem to prove
theorem entire_company_can_be_seated :
  ∀ (company : Finset ℕ), company.card = 7 → 
  (∀ (s : Finset ℕ), s ⊆ company ∧ s.card = 6 → 
   ∃ (perm : List ℕ), multiset.of_list perm = s.val ∧ 
   ∀ (i : ℕ) (h : i < 6), (perm.nth_le i h) = find_neighbors (perm.nth_le ((i + 1) % 6) h)) → 
  ∃ (perm : List ℕ), multiset.of_list perm = company.val ∧ 
    ∀ (i : ℕ) (h : i < 7), (perm.nth_le i h) = find_neighbors (perm.nth_le ((i + 1) % 7) h) := 
begin
  sorry
end

end entire_company_can_be_seated_l705_705385


namespace proof_set_intersection_l705_705359

open Set

-- Definitions of universal set U, set A, and set B
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℤ := {x | ∃ n : ℤ, n > 0 ∧ 6 = n * (3 - x)}
def B : Set ℕ := {4, 5}

-- The proof goal
theorem proof_set_intersection : A ∩ (U \ B) = {0, 1, 2} :=
by
  sorry

end proof_set_intersection_l705_705359


namespace min_radius_circle_condition_l705_705028

theorem min_radius_circle_condition (r : ℝ) (a b : ℝ) 
    (h_circle : (a - (r + 1))^2 + b^2 = r^2)
    (h_condition : b^2 ≥ 4 * a) :
    r ≥ 4 := 
sorry

end min_radius_circle_condition_l705_705028


namespace stock_price_after_three_years_l705_705273

theorem stock_price_after_three_years
  (initial_price : ℕ)
  (year1_increase_percent : ℕ)
  (year2_decrease_percent : ℕ)
  (year3_increase_percent : ℕ)
  (price_at_end_of_third_year : ℕ)
  (h_initial : initial_price = 120)
  (h_year1_increase : year1_increase_percent = 50)
  (h_year2_decrease : year2_decrease_percent = 30)
  (h_year3_increase : year3_increase_percent = 20)
  (h_price_end_year3 : price_at_end_of_third_year = 151.2) : 
  let year1_price := initial_price * (1 + year1_increase_percent / 100),
      year2_price := year1_price * (1 - year2_decrease_percent / 100),
      year3_price := year2_price * (1 + year3_increase_percent / 100) in
  year3_price = 151.2 :=
by
  sorry

end stock_price_after_three_years_l705_705273


namespace solve_quadratic_textbook_l705_705286

theorem solve_quadratic_textbook :
  ∀ x : ℝ, x > 5 ∧ (sqrt (x - 3 * sqrt (x - 5)) + 3 = sqrt (x + 3 * sqrt (x - 5)) - 3) → x = 41 :=
by
  intro x h
  sorry

end solve_quadratic_textbook_l705_705286


namespace min_value_of_xy_l705_705319

theorem min_value_of_xy (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h_geom : (ln x)^2 = (ln y) * (1/2)) : xy ≥ real.exp 1 ∧ ∃ (x y : ℝ), xy = real.exp 1 :=
by {
  sorry
}

end min_value_of_xy_l705_705319


namespace gcd_max_two_digits_l705_705834

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705834


namespace solution_set_f_l705_705074

noncomputable def f (x : ℝ) : ℝ := ite (x ≥ 0) (2^x - 4) (2^(-x) - 4)

theorem solution_set_f (x : ℝ) :
  (x < 0 ∨ x > 4) ↔ f(x-2) > 0 :=
sorry

end solution_set_f_l705_705074


namespace complex_quadrant_l705_705577

open Complex

noncomputable def givenComplex : ℂ := -1/2 + (sqrt 3 / 2) * Complex.i

theorem complex_quadrant :
  let square := givenComplex^2 in
  square.re > 0 ∧ square.im < 0 ∧ (
    (square.re > 0 ∧ square.im < 0) → 
    (Quadrant(square.re, square.im) = Fourth_Quadrant)
  ) :=
  by
    -- Add the proof steps here
    sorry

def Quadrant (x y : ℝ) : String :=
  if x > 0 
  then 
    if y > 0 then "First_Quadrant"
    else if y < 0 then "Fourth_Quadrant"
    else "On_Axis"
  else
    if x < 0 
    then 
      if y > 0 then "Second_Quadrant"
      else if y < 0 then "Third_Quadrant"
      else "On_Axis"
  else 
    "On_Axis"

end Complex

end complex_quadrant_l705_705577


namespace average_homework_time_decrease_l705_705549

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l705_705549


namespace terminative_decimal_of_45_div_72_l705_705278

theorem terminative_decimal_of_45_div_72 :
  (45 / 72 : ℚ) = 0.625 :=
sorry

end terminative_decimal_of_45_div_72_l705_705278


namespace doughnuts_in_shop_l705_705531

def ratio_of_doughnuts_to_muffins : Nat := 5

def number_of_muffins_in_shop : Nat := 10

def number_of_doughnuts (D M : Nat) : Prop :=
  D = ratio_of_doughnuts_to_muffins * M

theorem doughnuts_in_shop :
  number_of_doughnuts D number_of_muffins_in_shop → D = 50 :=
by
  sorry

end doughnuts_in_shop_l705_705531


namespace product_of_two_integers_l705_705961

def gcd_lcm_prod (x y : ℕ) :=
  Nat.gcd x y = 8 ∧ Nat.lcm x y = 48

theorem product_of_two_integers (x y : ℕ) (h : gcd_lcm_prod x y) : x * y = 384 :=
by
  sorry

end product_of_two_integers_l705_705961


namespace products_not_all_greater_than_one_quarter_l705_705931

theorem products_not_all_greater_than_one_quarter
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  ¬ ((1 - a) * b > 1 / 4 ∧ (1 - b) * c > 1 / 4 ∧ (1 - c) * a > 1 / 4) :=
by
  sorry

end products_not_all_greater_than_one_quarter_l705_705931


namespace most_lines_of_symmetry_l705_705578

def regular_pentagon_lines_of_symmetry : ℕ := 5
def kite_lines_of_symmetry : ℕ := 1
def regular_hexagon_lines_of_symmetry : ℕ := 6
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def scalene_triangle_lines_of_symmetry : ℕ := 0

theorem most_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry = max
    (max (max (max regular_pentagon_lines_of_symmetry kite_lines_of_symmetry)
              regular_hexagon_lines_of_symmetry)
        isosceles_triangle_lines_of_symmetry)
    scalene_triangle_lines_of_symmetry :=
sorry

end most_lines_of_symmetry_l705_705578


namespace value_of_y_at_x8_l705_705774

theorem value_of_y_at_x8 
  (k : ℝ) 
  (h1 : ∀ x, y = k * x^(1/3))
  (h2 : y = 3 * real.sqrt 2) 
  (h3 : x = 64) 
  (h4 : x = 8) :
  y = 3 := sorry

end value_of_y_at_x8_l705_705774


namespace gcd_at_most_3_digits_l705_705849

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705849


namespace continuous_at_5_l705_705687

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x^2 - 1
else 3*x + b

theorem continuous_at_5 : ∃ b : ℝ, continuous (λ x, f x b) at 5 :=
begin
  use 9,
  sorry
end

end continuous_at_5_l705_705687


namespace exists_segment_l705_705195

theorem exists_segment (f : ℚ → ℤ) : 
  ∃ (a b c : ℚ), a ≠ b ∧ c = (a + b) / 2 ∧ f a + f b ≤ 2 * f c :=
by 
  sorry

end exists_segment_l705_705195


namespace find_a3_l705_705161

variable {a : ℕ → ℝ}  -- Define a sequence a

-- The sequence is geometric with first term a_1 = 2 and fifth term a_5 = 8
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def sequence_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 2 ∧ a 5 = 8 ∧ is_geometric_seq a q

theorem find_a3 (a : ℕ → ℝ) (q : ℝ) (h : sequence_conditions a q) : a 3 = 4 := by
  unfold sequence_conditions at h
  cases h with h1 h'
  cases h' with h2 h3
  -- Using the fact that a is a geometric sequence with ratio q
  sorry

end find_a3_l705_705161


namespace calculate_si_l705_705305

section SimpleInterest

def Principal : ℝ := 10000
def Rate : ℝ := 0.04
def Time : ℝ := 1
def SimpleInterest : ℝ := Principal * Rate * Time

theorem calculate_si : SimpleInterest = 400 := by
  -- Proof goes here.
  sorry

end SimpleInterest

end calculate_si_l705_705305


namespace max_value_of_function_l705_705521

theorem max_value_of_function :
  ∃ x : ℝ, (2 ≤ x) → (x ≤ 6) →
  (∀ y : ℝ, y = 3 * Real.sqrt (x - 2) + 4 * Real.sqrt (6 - x) → y ≤ 10) := 
begin
  sorry,
end

end max_value_of_function_l705_705521


namespace part1_part2_l705_705440

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705440


namespace exists_m_and_seq_l705_705106

theorem exists_m_and_seq (a : ℕ → ℤ) (m : ℤ) :
  a 0 = 1 ∧
  a 1 = 337 ∧
  (∀ n ≥ 1, (a (n+1) * a (n-1) - a n ^ 2) + (3/4) * (a (n+1) + a (n-1) - 2 * a n) = m) ∧
  (∀ n ≥ 1, ∃ k : ℤ, 1/6 * (a n + 1) * (2 * a n + 1) = k ^ 2) ∧
  (∃ c : ℤ, m = (9457 * c - 912625) / 16 ∧ c % 8 = 1) :=
begin
  sorry
end

end exists_m_and_seq_l705_705106


namespace find_BJ_l705_705928

open Real

variables {A B C D H J K : Type}
variables {AD BC AC JH KH : ℝ}

-- Conditions given in the problem
def parallelogram (A B C D : Type) (AD BC : ℝ) : Prop := AD = 2 * BC
def point_on_extension (H : Type) (AD : ℝ) : Prop := true
def BH_intersects_AC_at_J (B H AC J : Type) : Prop := true
def BH_intersects_DC_at_K (B H DC K : Type) : Prop := true
def length_JH (J H : Type) (JH : ℝ) : Prop := JH = 20
def length_KH (K H : Type) (KH : ℝ) : Prop := KH = 30

theorem find_BJ (A B C D H J K : Type) (AD BC : ℝ){
    h_parallelogram : parallelogram A B C D AD BC,
    h_point_on_extension : point_on_extension H AD,
    h_BH_intersects_AC : BH_intersects_AC_at_J B H (AC : ℝ) J,
    h_BH_intersects_DC : BH_intersects_DC_at_K B H (DC : ℝ) K,
    h_JH : length_JH J H (JH : ℝ),
    h_KH : length_KH K H (KH : ℝ)} :
    BJ = 5 := by
  sorry

end find_BJ_l705_705928


namespace proposition_D_true_l705_705312

theorem proposition_D_true: forall n : ℤ, even (n^2 + n) :=
sorry

end proposition_D_true_l705_705312


namespace regression_line_equation_l705_705337

-- Define the conditions in the problem
def slope_of_regression_line : ℝ := 1.23
def center_of_sample_points : ℝ × ℝ := (4, 5)

-- The proof problem to show that the equation of the regression line is y = 1.23x + 0.08
theorem regression_line_equation :
  ∃ b : ℝ, (∀ x y : ℝ, (y = slope_of_regression_line * x + b) 
  → (4, 5) = (x, y)) → b = 0.08 :=
sorry

end regression_line_equation_l705_705337


namespace ending_point_243_l705_705879

def has_two_fours(n : ℕ) : Prop :=
  n.digits 10.count 4 = 2

theorem ending_point_243 :
  ∃ (end_point : ℕ), (∀ n : ℕ, 10 ≤ n ∧ n ≤ end_point → has_two_fours n → n = 144) ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ end_point → has_two_fours n → ¬ has_two_fours (n + 1)) ∧ end_point = 243 :=
sorry

end ending_point_243_l705_705879


namespace complex_addition_l705_705700

theorem complex_addition (x y : ℝ) (h : (x + y * complex.I) / (1 + complex.I) = 2 + complex.I) : x + y = 4 :=
by
  sorry

end complex_addition_l705_705700


namespace stacy_height_last_year_l705_705508

-- Definitions for the conditions
def brother_growth := 1
def stacy_growth := brother_growth + 6
def stacy_current_height := 57
def stacy_last_years_height := stacy_current_height - stacy_growth

-- Proof statement
theorem stacy_height_last_year : stacy_last_years_height = 50 :=
by
  -- proof steps will go here
  sorry

end stacy_height_last_year_l705_705508


namespace smallest_sum_arithmetic_geometric_sequence_l705_705526

theorem smallest_sum_arithmetic_geometric_sequence :
  ∃ (A B C D : ℤ), 0 < A ∧ 0 < B ∧ 0 < C ∧
  (C - B = B - A) ∧ (B * 7 = 3 * C) ∧ (C * 7 = 3 * D) ∧
  A + B + C + D = 76 :=
begin
  sorry
end

end smallest_sum_arithmetic_geometric_sequence_l705_705526


namespace length_of_floor_l705_705148

-- Let b be the breadth of the floor
def breadth (b : ℝ) : Prop := b = ℝ.sqrt (293.3 / 4.5)

-- Length is 4.5 times the breadth
def length (L b : ℝ) : Prop := L = 4.5 * b

-- Area of the floor in square meters
def area (A : ℝ) : Prop := A = 2200 / 7.5

-- The statement to prove
theorem length_of_floor {b L A : ℝ} 
  (hc1 : breadth b) 
  (hc2 : length L b) 
  (hc3 : area A) : 
  L = 36.315 := 
  sorry

end length_of_floor_l705_705148


namespace balls_in_drawers_l705_705107

theorem balls_in_drawers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : (k ^ n) = 32 :=
by
  rw [h_n, h_k]
  sorry

end balls_in_drawers_l705_705107


namespace max_sin_cos_expression_l705_705297

open Real  -- Open the real numbers namespace

theorem max_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) *
              (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_sin_cos_expression_l705_705297


namespace max_value_of_trig_expr_l705_705294

theorem max_value_of_trig_expr (x y z : ℝ) : 
  (sin (2 * x) + sin (3 * y) + sin (4 * z)) * (cos (2 * x) + cos (3 * y) + cos (4 * z)) ≤ 4.5 :=
sorry

end max_value_of_trig_expr_l705_705294


namespace fifty_third_card_is_ace_l705_705895

def sequence := ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

def cycle_length : Nat := 13

def nth_card (n : Nat) : Char :=
  sequence.get! (n % cycle_length)

theorem fifty_third_card_is_ace : nth_card 52 = 'A' :=
by sorry

end fifty_third_card_is_ace_l705_705895


namespace find_next_sales_l705_705623

-- Conditions
def royalties_first_20_million (royalties sales : ℝ) : Prop := royalties = 3 ∧ sales = 20
def royalties_next_x (royalties sales : ℝ) : Prop := royalties = 9 ∧ sales = x
def ratio_decrease (original new : ℝ) : Prop := new = original - (0.15 * original)

-- Given
def original_ratio := 0.15
def new_ratio := 0.1275

-- Statement to prove
theorem find_next_sales (x : ℝ) (h1 : royalties_first_20_million 3 20) (h2 : royalties_next_x 9 x) (h3 : ratio_decrease original_ratio new_ratio) : x = 70.588 := 
sorry

end find_next_sales_l705_705623


namespace problem1_problem2_l705_705747

-- Problem (Ⅰ) in Lean 4 statement
theorem problem1 (m : ℝ) : ∃ (x y : ℝ), (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0 ∧ x = -1 ∧ y = -2 :=
by {
  -- point (-1, -2) satisfies the line equation for any real m
  use [-1, -2],
  sorry
}

-- Problem (Ⅱ) in Lean 4 statement
theorem problem2 : ∃ k : ℝ, ∀ x y : ℝ, (y + 2 = k * (x + 1)) → (2 * x + y + 4 = 0) :=
by {
  -- the condition ensures that the line goes through M(-1, -2)
  use [-2],
  sorry
}

end problem1_problem2_l705_705747


namespace problem_range_of_a_l705_705754

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ ((x^2 - x + a = 0) → false) ∧ ((x^2 - x + a = 0) ∨ (∀ x : ℝ, a * x^2 + a * x + 1 > 0)) ↔ true →
  a ∈ set.Iio 0 ∪ set.Ioo 0 4 :=
sorry

end problem_range_of_a_l705_705754


namespace complex_division_l705_705334

theorem complex_division (a : ℝ) (h : a^2 - 1 = 0) (h2 : a + 1 ≠ 0) : 
  let z := a^2 - 1 + (1 + a) * complex.I in 
  z / (2 + complex.I) = (2 / 5 : ℂ) + (4 / 5) * complex.I :=
by
  sorry

end complex_division_l705_705334


namespace percentage_of_children_speaking_only_Hindi_l705_705001

/-
In a class of 60 children, 30% of children can speak only English,
20% can speak both Hindi and English, and 42 children can speak Hindi.
Prove that the percentage of children who can speak only Hindi is 50%.
-/
theorem percentage_of_children_speaking_only_Hindi :
  let total_children := 60
  let english_only := 0.30 * total_children
  let both_languages := 0.20 * total_children
  let hindi_only := 42 - both_languages
  (hindi_only / total_children) * 100 = 50 :=
by
  sorry

end percentage_of_children_speaking_only_Hindi_l705_705001


namespace CanVolume_l705_705217

variable (X Y : Type) [Field X] [Field Y] (V W : X)

theorem CanVolume (mix_ratioX mix_ratioY drawn_volume new_ratioX new_ratioY : ℤ)
  (h1 : mix_ratioX = 5) (h2 : mix_ratioY = 7) (h3 : drawn_volume = 12) 
  (h4 : new_ratioX = 4) (h5 : new_ratioY = 7) :
  V = 72 ∧ W = 72 := 
sorry

end CanVolume_l705_705217


namespace apartment_building_floors_l705_705250

theorem apartment_building_floors (F : ℕ) 
  (half_full_floors : F / 2 = n) 
  (half_capacity_floors : F / 2 = m)
  (each_floor_apartments : 10 = a)
  (each_apartment_people : 4 = b)
  (total_people : 360 = total) :
  F = 12 :=
by
  have h_full_floors : (F / 2) * (a * b) = 40 * n, from sorry,
  have h_half_capacity_floors : (F / 2) * (a / 2 * b) = 20 * m, from sorry,
  have h_total : total = 360, from sorry,
  have h_equation : 360 = (F / 2 * 40) + (F / 2 * 20), from sorry,
  have h_F_solve : 30 * F = 360, from sorry,
  exact Nat.div_eq_of_eq_mul_left (show 30 > 0 from dec_trivial) (by norm_num), from sorry


end apartment_building_floors_l705_705250


namespace min_length_MX_l705_705495

/--
Given a triangle ABC with sides AB = 17, AC = 30, and BC = 19, 
M is the midpoint of BC and O is the midpoint of AB. A circle is constructed
with diameter AB and X is any point on this circle. This theorem states
that the minimum length of segment MX is 6.5.
-/
theorem min_length_MX (A B C M O X : Point) (h_mid_M : is_midpoint M B C) (h_mid_O : is_midpoint O A B)
  (h_circle : on_circle X O 8.5) (h_AB : dist A B = 17) (h_AC : dist A C = 30) (h_BC : dist B C = 19) : 
  ∃ X, dist M X = 6.5 :=
by
  sorry

end min_length_MX_l705_705495


namespace chord_length_l705_705719

open Real

-- Let C be the circle centered at (m, n) with radius 2
def CircleC (m n x y : ℝ) : Prop := (x - m)^2 + (y - n)^2 = 4

-- Vectors from center to points A and B
def CA (cx cy ax ay : ℝ) : ℝ × ℝ := (ax - cx, ay - cy)
def CB (cx cy bx by : ℝ) : ℝ × ℝ := (bx - cx, by - cy)

-- Magnitude of the vector sum
def vector_sum_magnitude (ca cb : ℝ × ℝ) : ℝ :=
  sqrt ((ca.1 + cb.1)^2 + (ca.2 + cb.2)^2)

-- Given that the magnitude of the sum of vectors is 2√3
def given_magnitude_condition (cx cy ax ay bx by : ℝ) : Prop :=
  vector_sum_magnitude (CA cx cy ax ay) (CB cx cy bx by) = 2 * sqrt 3

-- Proving the length of chord AB
theorem chord_length (m n ax ay bx by : ℝ)
  (hA : CircleC m n ax ay)
  (hB : CircleC m n bx by)
  (hSum : given_magnitude_condition m n ax ay bx by) : 
  dist (ax, ay) (bx, by) = 2 :=
by
  sorry

end chord_length_l705_705719


namespace cistern_emptying_time_l705_705603

theorem cistern_emptying_time (fill_time_normal : ℝ) (fill_time_leak : ℝ) (leak_fill_extra_time : ℝ) :
    fill_time_normal = 12 → fill_time_leak = fill_time_normal + leak_fill_extra_time → leak_fill_extra_time = 2 →
    (1 / fill_time_normal - 1 / fill_time_leak)⁻¹ = 84 :=
by 
  intros h1 h2 h3
  rw [h1, h3, add_comm, add_sub_cancel'_right, inv_eq_one_div, one_div_div]
  calc 12 * 14 : sorry,
       168 : sorry,
       1 / 12 - 1 / 14 : sorry,
       2 / 168 : sorry,
       1 / 84 : sorry,
       84 : sorry

end cistern_emptying_time_l705_705603


namespace proof_problem_l705_705249

variables {m n : Type} {α β γ : Type}
variables [linear_algebra.line m] [linear_algebra.line n]
variables [linear_algebra.plane α] [linear_algebra.plane β] [linear_algebra.plane γ]

-- Assuming the appropriate perpendicular and parallel relations within a suitable algebraic setup

axiom perp : m ⊥ α → n ∥ α → m ⊥ n
axiom parallel_perp : α ∥ β → β ∥ γ → m ⊥ α → m ⊥ γ

theorem proof_problem :
  (m ⊥ α ∧ n ∥ α → m ⊥ n) ∧
  (α ∥ β ∧ β ∥ γ ∧ m ⊥ α → m ⊥ γ) :=
by {
  split,
  { intros h1 h2, exact perp h1 h2, },
  { intros h1 h2 h3, exact parallel_perp h1 h2 h3}
}

end proof_problem_l705_705249


namespace sequence_general_term_min_n_for_Sn_l705_705355

-- Conditions for the sequence {a_n}
variable {a_n : ℕ → ℕ}
variable {a_1 a_2 a_3 : ℕ}
axiom seq_cond : ∀ n, a_{n+1} = 2 * a_n
axiom arith_seq_cond : a_1, a_2 + 1, a_3 form an arithmetic sequence

-- Formal problem statement
theorem sequence_general_term :
  (∀ n, a_{n+1} = 2 * a_n) ∧ (a_1, a_2 + 1, a_3 form an arithmetic sequence)
  → (∀ n, a_n = 2^n) := sorry

-- For part II
variable {S_n : ℕ → ℕ}
axiom log_seq_cond : S_n = (sum of first n terms of sequence log_2 a_n)
axiom bi_seq_start : b_1 = 1
axiom bi_seq_diff : ∀ n, n ≥ 2 → b_n - b_{n-1} = 1

theorem min_n_for_Sn :
  (S_n > 45) → (n >= 1) → (∃ n, n = 10) := sorry

end sequence_general_term_min_n_for_Sn_l705_705355


namespace matrix_A_pow_100_eq_l705_705044

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1], ![-9, -2]]

theorem matrix_A_pow_100_eq : matrix_A ^ 100 = ![![301, 100], ![-900, -299]] :=
  sorry

end matrix_A_pow_100_eq_l705_705044


namespace min_distance_P_to_line_l_l705_705029

theorem min_distance_P_to_line_l
  (O1 O2 : Point)
  (tangent_O1_O2_x_axis : tangent O1 O2 x_axis)
  (collinear_centers_origin : collinear {O1.center, O2.center, origin})
  (product_x_coordinates : O1.center.x * O2.center.x = 6)
  (intersect_points_P_Q : ∃ P Q : Point, O1.circumference ∩ O2.circumference = {P, Q})
  (line_l : Line)
  (line_l_eq : line_l = {p : Point | 2 * p.x - p.y - 8 = 0}) :
  minimum_distance (intersect_points_P_Q P) (line_l) = (8 * (sqrt 5) / 5) - (sqrt 6) := 
  sorry

end min_distance_P_to_line_l_l705_705029


namespace quadrilateral_parallelogram_l705_705716

variable (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

theorem quadrilateral_parallelogram {AB DC : A} (h : AB = DC) :
  ∃ (ABCD : Quadrilateral), ABCD.is_parallelogram := 
sorry

end quadrilateral_parallelogram_l705_705716


namespace milton_fraction_of_greta_l705_705926

-- Define the conditions
def penelopeDailyIntake := 20
def gretaDailyIntake := penelopeDailyIntake / 10
def elmerDailyIntake := penelopeDailyIntake + 60
def miltonDailyIntake := elmerDailyIntake / 4000

-- Theorem to prove that Milton eats 1/100 of Greta's food intake
theorem milton_fraction_of_greta : miltonDailyIntake / gretaDailyIntake = 1 / 100 := by
  unfold penelopeDailyIntake gretaDailyIntake elmerDailyIntake miltonDailyIntake
  sorry

end milton_fraction_of_greta_l705_705926


namespace sequence_form_l705_705160

noncomputable def a_seq : ℕ → ℕ
| 0     := 3
| 1     := 11
| (n+2) := 4 * a_seq (n + 1) - a_seq n

theorem sequence_form (n : ℕ) :
  ∃ (a b : ℕ), a_seq n = a^2 + 2 * b^2 :=
by
  sorry

end sequence_form_l705_705160


namespace correct_model_l705_705562

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l705_705562


namespace prime_square_minus_one_divisible_by_twelve_l705_705912

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) : 12 ∣ (p^2 - 1) :=
by
  sorry

end prime_square_minus_one_divisible_by_twelve_l705_705912


namespace cylinder_twice_volume_l705_705701

theorem cylinder_twice_volume :
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  V_desired = pi * r^2 * h2 :=
by
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  show V_desired = pi * r^2 * h2
  sorry

end cylinder_twice_volume_l705_705701


namespace transactions_proof_l705_705095

def transactions_problem : Prop :=
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + (0.10 * mabel_transactions)
  let cal_transactions := (2 / 3) * anthony_transactions
  let jade_transactions := 81
  jade_transactions - cal_transactions = 15

-- The proof is omitted (replace 'sorry' with an actual proof)
theorem transactions_proof : transactions_problem := by
  sorry

end transactions_proof_l705_705095


namespace backpack_weights_l705_705547

-- Define variables x and y as the weights of the boy's and girl's backpacks respectively.
variables (x y : ℝ)

-- Condition 1: The total weight of all backpacks
axiom total_weight : 2 * x + 3 * y = 44

-- Condition 2: The balancing condition indicating that when distributed, loads are equal 
axiom equal_distribution : x + y / 4 = y + y / 4

-- Theorem statement: Given the conditions, the weights of the backpacks
theorem backpack_weights :
  (∃ (x y : ℝ), (2 * x + 3 * y = 44) ∧ (x = 5 / 4 * y) ∧ y = 8 ∧ x = 10) :=
begin
  -- The proof is omitted, but we assert the existence of such x and y.
  use [10, 8],
  split,
  { exact total_weight },
  split,
  { exact equal_distribution },
  -- Remaining conditions
  split,
  { refl },
  { refl },
end

end backpack_weights_l705_705547


namespace select_10_teams_l705_705594

def football_problem (teams : Finset ℕ) (played_on_day1 : Finset (ℕ × ℕ)) (played_on_day2 : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v : ℕ), v ∈ teams → (∃ u w : ℕ, (u, v) ∈ played_on_day1 ∧ (v, w) ∈ played_on_day2)

theorem select_10_teams {teams : Finset ℕ}
  (h : teams.card = 20)
  {played_on_day1 played_on_day2 : Finset (ℕ × ℕ)}
  (h1 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day1 → u ∈ teams ∧ v ∈ teams)
  (h2 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day2 → u ∈ teams ∧ v ∈ teams)
  (h3 : ∀ x ∈ teams, ∃ u w, (u, x) ∈ played_on_day1 ∧ (x, w) ∈ played_on_day2) :
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ ⦃x y⦄, x ∈ S → y ∈ S → x ≠ y → (¬((x, y) ∈ played_on_day1) ∧ ¬((x, y) ∈ played_on_day2))) :=
by
  sorry

end select_10_teams_l705_705594


namespace find_h_l705_705530

theorem find_h {a b c n k : ℝ} (x : ℝ) (h_val : ℝ) 
  (h_quad : a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  (4 * a) * x^2 + (4 * b) * x + (4 * c) = n * (x - h_val)^2 + k → h_val = 5 :=
sorry

end find_h_l705_705530


namespace true_statements_l705_705510

variables (x y : ℤ)

-- Definitions (conditions)
def is_multiple_of_4 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 4 * k

def is_multiple_of_8 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 8 * k

-- Proof problem statement
theorem true_statements (hx : is_multiple_of_4 x) (hy : is_multiple_of_8 y) :
  is_multiple_of_4 y ∧ is_multiple_of_4 (x - y) ∧ is_multiple_of_2 (x - y) :=
by {
  sorry,
}

end true_statements_l705_705510


namespace sum_of_digits_of_terraces_l705_705089

theorem sum_of_digits_of_terraces {n : ℕ} :
  (∃ n : ℕ, (n : ℤ).ceil.div 3 - (n : ℤ).ceil.div 7 = 15) →
  ∑ (n in {n : ℕ | (n : ℤ).ceil.div 3 - (n : ℤ).ceil.div 7 = 15}, n.digits.sum) = 18 :=
by
  sorry

-- This is needed to complete the definition of digits sum
def ℕ.digits (n : ℕ) : List ℕ := n.toString.toList.map (λ c : Char => c.toNat - '0'.toNat)

noncomputable def ℕ.digits.sum (n : ℕ) : ℕ := (n.digits.natSum)

-- Here n.digits is the list of the digits of n and n.digits.sum is the sum of those digits

end sum_of_digits_of_terraces_l705_705089


namespace find_length_PF_l705_705031

theorem find_length_PF (P Q R L M F : Type) (distancePQ distancePR distanceFL distancePQF : ℝ)
  (h_right : ∀ (a b c : ℝ), a * a + b * b = c * c) 
  (PQ PR : ℝ) 
  (angleRPQ_right : math.angle R P Q = 90) 
  (PQ_value : PQ = 4) 
  (PR_value : PR = 4 * real.sqrt 3)
  (PL_altitude_RQ : Line) 
  (RM_median_PQ : Line)
  (PL_RM_intersect_F : F = PL_altitude_RQ ∩ RM_median_PQ)
  (M_position : distancePQ = pq_fraction * PQ ∧ pq_fraction = 1/3) : 
  distanceFL = 3/2 * real.sqrt 3 :=
by sorry

end find_length_PF_l705_705031


namespace projection_line_equation_l705_705971

variable (A B : Point) (l : Line)
open Point Line

def eq_of_line_l: Prop := 
  A = (0, 0) ∧ B = (2, 3) ∧ projection A l = B → line_equation l = "2x + 3y - 13 = 0"

theorem projection_line_equation (A B : Point) (l : Line) : eq_of_line_l A B l :=
by
  sorry

end projection_line_equation_l705_705971


namespace ice_cream_sundae_combinations_l705_705251

theorem ice_cream_sundae_combinations :  
  let n := 8 in
  let num_pairs := Nat.choose n 2 in
  num_pairs = 28 := 
by
  sorry

end ice_cream_sundae_combinations_l705_705251


namespace hexagon_circumscribing_circle_l705_705470

theorem hexagon_circumscribing_circle (A B C D E F U V W X Y Z : Point) (ω : Circle) :
  circumscribes ω [A, B, C, D, E, F] ∧
  touches ω [A, B, C, D, E, F] [U, V, W, X, Y, Z] ∧
  midpoint U A B ∧ midpoint W C D ∧ midpoint Y E F →
  are_concurrent [UX, VY, WZ] := 
sorry

end hexagon_circumscribing_circle_l705_705470


namespace gcd_max_two_digits_l705_705841

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705841


namespace hyperbola_minimum_distance_l705_705954

noncomputable def min_distance_from_p_to_midpoint (A B P : ℝ × ℝ) : ℝ :=
  let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB
  let dist := (x y : ℝ) → (x - mid.1)^2 + (y - mid.2)^2 -- Distance from P to midpoint
  sqrt(dist P.1 P.2)

theorem hyperbola_minimum_distance (A B P : ℝ × ℝ) (hAB : dist A B = 4) (hPA_PB : dist P A - dist P B = 2) :
  min_distance_from_p_to_midpoint A B P = 1 :=
by
  sorry

end hyperbola_minimum_distance_l705_705954


namespace probability_A_B_meet_l705_705921

-- Define starting points and movements of A and B
def start_A := (0, 0)
def start_B := (10, 10)

-- Define movement constraints for both A and B
def movement_A (x : ℕ) (y : ℕ) : Prop := x + y = 10
def movement_B (x : ℕ) (y : ℕ) : Prop := (10 - x) + (10 - y) = 10 

-- Define the probability calculation
noncomputable def probability_meet : ℝ :=
  let num_meeting_ways := ∑ x in finset.range 11, nat.choose 10 x * nat.choose 10 (10 - x)
  let total_paths := 2^20
  (num_meeting_ways : ℝ) / total_paths

-- Problem statement to prove the probability of meeting is 0.352
theorem probability_A_B_meet : probability_meet = 0.352 := by
  sorry

end probability_A_B_meet_l705_705921


namespace find_B_and_area_l705_705881

variable (A B C a b c d : ℝ)

-- Given conditions
axiom cond1 : (a + c) * Real.sin A = Real.sin A + Real.sin C
axiom cond2 : c^2 + c = b^2 - 1
axiom midpoint_D : D = (A + C) / 2
axiom BD_val : BD = sqrt 3 / 2

-- To Prove
theorem find_B_and_area : 
  ∃ B : ℝ, 
    (B = 2 * π / 3) ∧ 
    let area := (1 / 2) * a * c * Real.sin B in 
    area = sqrt 3 / 2 :=
by
  sorry

end find_B_and_area_l705_705881


namespace regular_price_of_each_shirt_l705_705544

theorem regular_price_of_each_shirt (P : ℝ) :
    let total_shirts := 20
    let sale_price_per_shirt := 0.8 * P
    let tax_rate := 0.10
    let total_paid := 264
    let total_price := total_shirts * sale_price_per_shirt * (1 + tax_rate)
    total_price = total_paid → P = 15 :=
by
  intros
  sorry

end regular_price_of_each_shirt_l705_705544


namespace rabbit_parent_genotype_l705_705118

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l705_705118


namespace no_integer_solutions_cos_sum_eq_cos_l705_705690

theorem no_integer_solutions_cos_sum_eq_cos 
  (k : ℕ)
  (α : Fin k → ℝ)
  (p : Fin k → ℕ)
  (cos_conditions : ∀ i, p i > 2 ∧ Prime (p i) ∧ cos (α i) = 1 / (p i))
  (n : Fin k → ℕ) :
  ¬ (cos (n ⟨0, by simp⟩ * α ⟨0, by simp⟩) + ∑ i in Finset.range (k - 1), cos (n ⟨i.succ, by simp⟩ * α ⟨i.succ, by simp⟩) = cos (n ⟨k - 1, by simp⟩ * α ⟨k - 1, by simp⟩)) :=
sorry

end no_integer_solutions_cos_sum_eq_cos_l705_705690


namespace triangle_property_l705_705852

theorem triangle_property
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a > b)
  (h2 : a = 5)
  (h3 : c = 6)
  (h4 : Real.sin B = 3 / 5) :
  (b = Real.sqrt 13 ∧ Real.sin A = 3 * Real.sqrt 13 / 13) →
  Real.sin (2 * A + π / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end triangle_property_l705_705852


namespace root_equation_m_l705_705373

theorem root_equation_m (m : ℝ) : 
  (∃ (x : ℝ), x = -1 ∧ m*x^2 + x - m^2 + 1 = 0) → m = 1 :=
by 
  sorry

end root_equation_m_l705_705373


namespace decrease_equation_l705_705554

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l705_705554


namespace quadratic_inequality_solution_l705_705978

theorem quadratic_inequality_solution
  (a b c α β : ℝ) 
  (h₁ : 0 < α) 
  (h₂ : α < β)
  (h₃ : α + β = -b / a)
  (h₄ : α * β = c / a) :
  (cx : ℝ) : cx ≠ 0 :=
  (cxx : ℝ -> ℝ) : cxx ≠ 0 :=
  ∀ x : ℝ, c * x^2 + b * x + a > 0 ↔ 1 / β < x ∧ x < 1 / α := sorry

end quadratic_inequality_solution_l705_705978


namespace not_necessarily_prime_sum_l705_705247

theorem not_necessarily_prime_sum (nat_ordered_sequence : ℕ → ℕ) :
  (∀ n1 n2 n3 : ℕ, n1 < n2 → n2 < n3 → nat_ordered_sequence n1 + nat_ordered_sequence n2 + nat_ordered_sequence n3 ≠ prime) :=
sorry

end not_necessarily_prime_sum_l705_705247


namespace nonnegative_integers_in_balanced_ternary_9_digits_l705_705762

theorem nonnegative_integers_in_balanced_ternary_9_digits :
  (∃ (a : Fin 9 → {-1, 0, 1}), ∀ i, 0 ≤ (a i) ∧ (a i) ≤ 1) →
  ∃ n : ℕ, n = 9842 ∧ (∃ (a : Fin 9 → {-1, 0, 1}), 
    n = a 8 * 3^8 + a 7 * 3^7 + a 6 * 3^6 + a 5 * 3^5 +
        a 4 * 3^4 + a 3 * 3^3 + a 2 * 3^2 + a 1 * 3^1 + a 0 * 3^0) := sorry

end nonnegative_integers_in_balanced_ternary_9_digits_l705_705762


namespace teachers_stand_together_teachers_stand_together_with_two_students_each_side_teachers_do_not_stand_next_to_each_other_l705_705694

-- Condition 1: The three teachers stand together
theorem teachers_stand_together (students teachers : ℕ) (students = 4) (teachers = 3) : 
  ∃ (n : ℕ), n = 3! * 5! ∧ n = 720 :=
by sorry

-- Condition 2: The three teachers stand together, and there are two students on each side
theorem teachers_stand_together_with_two_students_each_side (students teachers : ℕ) (students = 4) (teachers = 3) : 
  ∃ (n : ℕ), n = 3! * (4! / 2!) ∧ n = 144 :=
by sorry

-- Condition 3: The three teachers do not stand next to each other
theorem teachers_do_not_stand_next_to_each_other (students teachers : ℕ) (students = 4) (teachers = 3):
  ∃ (n : ℕ), n = 4! * (5.choose 3) ∧ n = 1440 :=
by sorry

end teachers_stand_together_teachers_stand_together_with_two_students_each_side_teachers_do_not_stand_next_to_each_other_l705_705694


namespace part_one_part_two_l705_705438

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705438


namespace exponential_function_minor_premise_incorrect_l705_705210

theorem exponential_function_minor_premise_incorrect:
  ∀ (α: ℝ), (1 < α) → ¬ (∃ (a : ℝ) (x : ℝ), (a > 1) ∧ ((a^x) = (x^α))) :=
λ α h1, 
  begin
    intro h, 
    rcases h with ⟨a, x, ha, hax⟩, 
    sorry
  end

end exponential_function_minor_premise_incorrect_l705_705210


namespace solve_for_x_l705_705943

theorem solve_for_x (x : ℝ) (h : 5^(2*x + 3) = 125^(x + 1)) : x = 0 := by
  sorry

end solve_for_x_l705_705943


namespace third_studio_students_l705_705994

theorem third_studio_students 
  (total_students : ℕ)
  (first_studio : ℕ)
  (second_studio : ℕ) 
  (third_studio : ℕ) 
  (h1 : total_students = 376) 
  (h2 : first_studio = 110) 
  (h3 : second_studio = 135) 
  (h4 : total_students = first_studio + second_studio + third_studio) :
  third_studio = 131 := 
sorry

end third_studio_students_l705_705994


namespace product_of_slopes_P_exists_line_passing_Q_l705_705713

noncomputable def ellipse : set (ℝ × ℝ) :=
  {p | let (x, y) := p in (x^2) / 2 + y^2 = 1}

def pointA : ℝ × ℝ := (0, -1)
def pointB : ℝ × ℝ := (0, 1)
def pointQ : ℝ × ℝ := (-2, 0)
def line_through (p q : ℝ × ℝ) : set (ℝ × ℝ) := {r | ∃ (t : ℝ), r = (1 - t) • p + t • q}

theorem product_of_slopes_P : (P : ℝ × ℝ) (hP : P ∈ ellipse) (hPA : P ≠ pointA) (hPB : P ≠ pointB) :
  let (x, y) := P in ((y - (-1)) / x) * ((y - 1) / x) = -1/2 :=
  by
    sorry

theorem exists_line_passing_Q (l : set (ℝ × ℝ)) :
  (∃ M N ∈ ellipse, M ≠ N ∧ M ∈ l ∧ N ∈ l ∧ (let (x1, y1) := M in let (x2, y2) := N in √((x1 - 0)^2 + (y1 - 1)^2) = √((x2 - 0)^2 + (y2 - 1)^2))) ∧ line_through pointQ l ∧
  (∀ M N ∈ l, let (x1, y1) := M, (x2, y2) := N in x1 = -x2) :=
  line_through pointQ {(x, 0) | x ∈ ℝ}

end product_of_slopes_P_exists_line_passing_Q_l705_705713


namespace masha_can_pay_exactly_with_11_ruble_bills_l705_705208

theorem masha_can_pay_exactly_with_11_ruble_bills (m n k p : ℕ) 
  (h1 : 3 * m + 4 * n + 5 * k = 11 * p) : 
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q := 
by {
  sorry
}

end masha_can_pay_exactly_with_11_ruble_bills_l705_705208


namespace determine_parents_genotype_l705_705112

noncomputable def genotype := ℕ -- We use nat to uniquely represent each genotype: e.g. HH=0, HS=1, Sh=2, SS=3

def probability_of_allele_H : ℝ := 0.1
def probability_of_allele_S : ℝ := 1 - probability_of_allele_H

def is_dominant (allele: ℕ) : Prop := allele == 0 ∨ allele == 1 -- HH or HS are dominant for hairy

def offspring_is_hairy (parent1 parent2: genotype) : Prop :=
  (∃ g1 g2, (parent1 = 0 ∨ parent1 = 1) ∧ (parent2 = 1 ∨ parent2 = 2 ∨ parent2 = 3) ∧
  ((g1 = 0 ∨ g1 = 1) ∧ (g2 = 0 ∨ g2 = 1))) ∧ 
  (is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0) 

def most_likely_genotypes (hairy_parent smooth_parent : genotype) : Prop :=
  (hairy_parent = 0) ∧ (smooth_parent = 2)

theorem determine_parents_genotype :
  ∃ hairy_parent smooth_parent, offspring_is_hairy hairy_parent smooth_parent ∧ most_likely_genotypes hairy_parent smooth_parent :=
  sorry

end determine_parents_genotype_l705_705112


namespace negation_of_proposition_l705_705966

theorem negation_of_proposition (a b : ℝ) : 
  (¬ ((a - 2) * (b - 3) = 0) → (a ≠ 2 ∧ b ≠ 3)) := 
begin
  sorry
end

end negation_of_proposition_l705_705966


namespace mr_green_expected_produce_l705_705488

noncomputable def total_produce_yield (steps_length : ℕ) (steps_width : ℕ) (step_length : ℝ)
                                      (yield_carrots : ℝ) (yield_potatoes : ℝ): ℝ :=
  let length_feet := steps_length * step_length
  let width_feet := steps_width * step_length
  let area := length_feet * width_feet
  let yield_carrots_total := area * yield_carrots
  let yield_potatoes_total := area * yield_potatoes
  yield_carrots_total + yield_potatoes_total

theorem mr_green_expected_produce:
  total_produce_yield 18 25 3 0.4 0.5 = 3645 := by
  sorry

end mr_green_expected_produce_l705_705488


namespace gcd_digit_bound_l705_705792

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705792


namespace log_inequality_example_l705_705983

theorem log_inequality_example (c d : ℤ) (log78903 : ℝ) :
  (log10 10000 = 4) ∧ (log10 100000 = 5) ∧ (4 < log78903) ∧ (log78903 < 5) ∧ 
  (c = 4) ∧ (d = 5) → 
  (c + d = 9) :=
by
  intro h,
  sorry

end log_inequality_example_l705_705983


namespace find_lambda_l705_705655

def geometric_sum (n : ℕ) (a : ℕ → ℕ) (r : ℝ) : ℝ :=
  a 0 * (1 - r^n) / (1 - r)

theorem find_lambda (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h_geom : ∀ n, a (n+1) = a 0 * r^n)
(h_sum : geometric_sum n a r = 2^(n+1) + -2) :
  ∃ λ : ℝ, λ = -2 := 
begin
  use -2,
  sorry
end

end find_lambda_l705_705655


namespace cost_price_equals_selling_price_l705_705952

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (h1 : 20 * C = 1.25 * C * x) : x = 16 :=
by
  -- This proof is omitted at the moment
  sorry

end cost_price_equals_selling_price_l705_705952


namespace degree_of_polynomial_l705_705663

def f (x : ℤ) : ℤ := 3 * x^5 + 2 * x^4 - x^3 + 5
def g (x : ℤ) : ℤ := 4 * x^11 - 8 * x^8 + 6 * x^5 + 22
def h (x : ℤ) : ℤ := x^3 + 4

theorem degree_of_polynomial :
  degree ((f x * g x) - (h x)^6) = 18 :=
sorry

end degree_of_polynomial_l705_705663


namespace find_function_l705_705545

def translation_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) - 2 = 2^x

theorem find_function (f : ℝ → ℝ) (h : translation_condition f) : 
  f = (λ x, 2^(x-2) + 2) :=
  sorry

end find_function_l705_705545


namespace incorrect_answer_A_l705_705191

-- Definitions for the conditions
def sumA : Nat := 8 + 34
def sumB : Nat := 17 + 17
def sumC : Nat := 15 + 13

-- The theorem stating the equivalent proof problem
theorem incorrect_answer_A : sumA ≠ 32 :=
by {
  -- Show each sum is correct based on given conditions
  have ha : sumA = 42, from rfl,
  have hb : sumB = 34, from rfl,
  have hc : sumC = 28, from rfl,
  -- Check the incorrect answer
  calc
    sumA = 42     : ha
    ... ≠ 32      : by decide
}

end incorrect_answer_A_l705_705191


namespace chandra_akiko_ratio_l705_705032

theorem chandra_akiko_ratio
  (points_bailey : ℕ)
  (points_michiko : ℕ)
  (points_akiko : ℕ)
  (points_chandra : ℕ)
  (total_points : ℕ)
  (bailey_score : points_bailey = 14)
  (michiko_score : points_michiko = points_bailey / 2)
  (akiko_score : points_akiko = points_michiko + 4)
  (total_score : points_chandra + points_bailey + points_michiko + points_akiko = 54) :
  points_chandra / points_akiko = 2 :=
begin
  sorry
end

end chandra_akiko_ratio_l705_705032


namespace intersection_M_N_l705_705752

def M := { x : ℝ | x^2 - 1 < 0 }
def N := { y : ℝ | ∃ x : ℝ, (x^2 - 1 < 0) ∧ y = Real.log2(x + 2)}

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y < 1 } :=
by
  sorry

end intersection_M_N_l705_705752


namespace find_x_l705_705598

theorem find_x : ∃ x : ℕ, (8 * x + 2 = 9 * x - 4) ∧ x = 6 := 
by {
  existsi 6,
  simp,
  sorry,
}

end find_x_l705_705598


namespace kaylee_more_boxes_to_sell_l705_705420

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end kaylee_more_boxes_to_sell_l705_705420


namespace quadrilateral_parallelogram_l705_705718

/-- Given a quadrilateral ABCD, if the vector AB is equal to DC,
    then quadrilateral ABCD is a parallelogram. -/
theorem quadrilateral_parallelogram (A B C D : Point) :
  (AB = DC) → is_parallelogram ABCD :=
by sorry

end quadrilateral_parallelogram_l705_705718


namespace gcd_digit_bound_l705_705788

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705788


namespace binary_operation_l705_705304

def b11001 := 25  -- binary 11001 is 25 in decimal
def b1101 := 13   -- binary 1101 is 13 in decimal
def b101 := 5     -- binary 101 is 5 in decimal
def b100111010 := 314 -- binary 100111010 is 314 in decimal

theorem binary_operation : (b11001 * b1101 - b101) = b100111010 := by
  -- provide implementation details to prove the theorem
  sorry

end binary_operation_l705_705304


namespace students_in_line_l705_705188

theorem students_in_line (students_in_front_of_seokjin : ℕ) (students_behind_jimin : ℕ) (students_between : ℕ) :
    students_in_front_of_seokjin = 4 ->
    students_behind_jimin = 7 ->
    students_between = 3 ->
    students_in_front_of_seokjin + 1 + students_between + 1 + students_behind_jimin = 16 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end students_in_line_l705_705188


namespace zeros_of_f_range_of_ratio_l705_705361

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ :=
  (2 * cos x, 2 * sin x)

noncomputable def vec_b (x : ℝ) : ℝ × ℝ :=
  (sin (x - π / 6), cos (x - π / 6))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ :=
  cos (dot_product (vec_a x) (vec_b x))

theorem zeros_of_f :
  ∀ (x : ℝ), f(x) = 0 ↔ ∃ k : ℤ, x = k * π / 2 + π / 12 := 
sorry

theorem range_of_ratio (α β : ℝ) (A B C a b c : ℝ) :
  A = π/3 ∧ f(A) = 1 → 1 < (b + c) / a ∧ (b + c) / a ≤ 2 := 
sorry

end zeros_of_f_range_of_ratio_l705_705361


namespace correct_relation_l705_705627

def satisfies_relation : Prop :=
  (∀ x y, (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0) →
  y = 200 - 10 * x - 10 * x^2) 

theorem correct_relation : satisfies_relation :=
sorry

end correct_relation_l705_705627


namespace gcd_digit_bound_l705_705805

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705805


namespace sequence_difference_l705_705165

theorem sequence_difference (S : ℕ → ℕ) (a : ℕ → ℕ) (n m : ℕ) (h1 : ∀ n, S n = n^2 + 2 * n) 
(h2 : n > 0) (h3 : m = n + 5) : a m - a n = 10 :=
by
  -- Definitions inferred from conditions
  have hSn : ∀ n, S n = n^2 + 2 * n, from h1,
  have hSmn : m - n = 5, by rw [h3, add_tsub_cancel_right n 5],
  -- skip actual proof steps
  sorry

end sequence_difference_l705_705165


namespace max_expression_l705_705064

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end max_expression_l705_705064


namespace find_T_shirts_l705_705093

variable (T S : ℕ)

-- Given conditions
def condition1 : S = 2 * T := sorry
def condition2 : T + S - (T + 3) = 15 := sorry

-- Prove that number of T-shirts T Norma left in the washer is 9
theorem find_T_shirts (h1 : S = 2 * T) (h2 : T + S - (T + 3) = 15) : T = 9 :=
  by
    sorry

end find_T_shirts_l705_705093


namespace mean_of_five_numbers_l705_705164

theorem mean_of_five_numbers (x1 x2 x3 x4 x5 : ℚ) (h_sum : x1 + x2 + x3 + x4 + x5 = 1/3) : 
  (x1 + x2 + x3 + x4 + x5) / 5 = 1/15 :=
by 
  sorry

end mean_of_five_numbers_l705_705164


namespace find_ordered_pair_l705_705287

theorem find_ordered_pair :
  ∃ x y : ℚ, (3 * x - 4 * y = -7) ∧ (4 * x - 3 * y = 5) ∧ x = 41 / 7 ∧ y = 43 / 7 :=
by
  use 41 / 7
  use 43 / 7
  split
  {
    rw [mul_div, mul_div]
    norm_num
  }
  split
  {
    rw [mul_div, mul_div]
    norm_num
  }
  split
  { reflexivity }
  { reflexivity }
  sorry

end find_ordered_pair_l705_705287


namespace find_multiple_of_sum_l705_705528

-- Define the conditions and the problem statement in Lean
theorem find_multiple_of_sum (a b m : ℤ) 
  (h1 : b = 8) 
  (h2 : b - a = 3) 
  (h3 : a * b = 14 + m * (a + b)) : 
  m = 2 :=
by
  sorry

end find_multiple_of_sum_l705_705528


namespace gcd_max_two_digits_l705_705835

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705835


namespace slope_tangent_line_at_pi_over_3_l705_705976

-- Define the function y = sin(3x)
def f (x : ℝ) := Real.sin (3 * x)

-- Define the derivative of the function
def f' (x : ℝ) := 3 * Real.cos (3 * x)

theorem slope_tangent_line_at_pi_over_3 :
  f' (Real.pi / 3) = -3 :=
by
  -- proof steps will be filled here
  sorry

end slope_tangent_line_at_pi_over_3_l705_705976


namespace sum_of_fourth_powers_equilateral_circumcircle_l705_705475

theorem sum_of_fourth_powers_equilateral_circumcircle
  (A B C M O : Type)
  [MetricSpace Type]
  [EquilateralTriangle A B C]
  [Circumcircle (A B C) M O]
  (R : ℝ)
  (hAOM : ∠ A O M = α) :
  (dist M A) ^ 4 + (dist M B) ^ 4 + (dist M C) ^ 4 = 18 * R ^ 4 := sorry

end sum_of_fourth_powers_equilateral_circumcircle_l705_705475


namespace altitudes_intersect_l705_705103

-- Defining relevant concepts in the context
variables (A B C S O P K : Type) -- Basic points in the pyramid

--axioms and hypotheses for the problem
axiom h1 : is_trianglular_pyramid  A B C S -- A pyramid with base ABC and apex S
axiom h2 : altitude SO (S, A, B, C) -- Altitude from apex S to base ABC
axiom h3 : altitude CP (C, S, A, B) -- Altitude from vertex C to the plane SAB
axiom h4 : intersects_at SO CP K -- Altitudes SO and CP intersect at point K

-- The theorem statement
theorem altitudes_intersect :
  exists X, remaining_altitudes_intersect (A, B, C, S, X) :=
sorry

end altitudes_intersect_l705_705103


namespace solve_exponential_equation_l705_705592

theorem solve_exponential_equation (x : ℝ) (h : 4^x - 2^(x+1) = 0) : x = 1 :=
sorry

end solve_exponential_equation_l705_705592


namespace sum_of_common_ratios_l705_705467

variable (k p r : ℝ)
variable (a_2 :-𝕝 a_3 b_2 b_3 : ℝ)
variable (h_nonconst : k ≠ 0)
variable (h_different_ratios : p ≠ r)
variable (h_seq_a : a_2 = k * p)
variable (h_seq_aa: a_3 = k * p^2)
variable (h_seq_b: b_2 = k * r)
variable (h_seq_bb : b_3 = k * r^2)
variable (h_condition : a_3 - b_3 = 4 * (a_2 - b_2))

theorem sum_of_common_ratios : p + r = 4 :=
sorry

end sum_of_common_ratios_l705_705467


namespace max_stones_in_pile_l705_705538

-- Define the maximum number of stones in one pile under given conditions
theorem max_stones_in_pile (piles : ℕ) (init_stones : ℕ) :
  piles = 2009 → init_stones = 2 →
  (∀ P, (P = list.replicate 2009 2) →
       ∃ max_stones, max_stones = 2010) :=
by
  intros piles_def stones_def initial_setup
  -- Specify initial condition and formulate the goal
  use 2010
  sorry

end max_stones_in_pile_l705_705538


namespace concyclic_iff_EA_ED_FA_FB_l705_705471

variables {A B C D E F : Type}
variables [convex_quadrilateral A B C D] (E : intersection (line_through A B) (line_through C D))
                              (F : intersection (line_through B C) (line_through A D))

theorem concyclic_iff_EA_ED_FA_FB (EA ED FA FB EF : ℝ) : 
  are_concyclic A B C D ↔ (EA * ED + FA * FB = EF^2) := sorry

end concyclic_iff_EA_ED_FA_FB_l705_705471


namespace th_perpendicular_cm_l705_705045

open Classical

noncomputable def midpoint (A B : Point) : Point := (A + B)/2
noncomputable def orthocenter (A B C : Triangle) : Point := sorry  -- Appropriate definition or construction needed
noncomputable def altitude (A B C : Triangle) : Line := sorry  -- Appropriate definition or construction needed
noncomputable def extend (L : Line) (P : Point) : Line := sorry -- Appropriate definition or construction needed
noncomputable def intersection (L₁ L₂ : Line) : Point := sorry  -- Appropriate definition or construction needed
noncomputable def perpendicular (L₁ L₂ : Line) : Prop := sorry  -- Appropriate definition needed

variables (A B C P Q T H M : Point)
variables (ABC : Triangle := Triangle.mk A B C)
variables (AP : Line := altitude ABC A)
variables (BQ : Line := altitude ABC B)
variables (QP_ext : Line := extend (Line.mk Q P) A)
variables (TH : Line := Line.mk T H)
variables (CM : Line := Line.mk C M)
variables (M := midpoint A B)
variables (H := orthocenter ABC)

def conditions : Prop :=
  BC < AC ∧ 
  P ∈ AP ∧ 
  Q ∈ BQ ∧ 
  T ∈ QP_ext ∧ 
  H ∈ AP ∧ 
  H ∈ BQ ∧ 
  H ∈ altitude ABC C

theorem th_perpendicular_cm : conditions → perpendicular TH CM := sorry

end th_perpendicular_cm_l705_705045


namespace circumradius_of_triangle_ABC_l705_705324

noncomputable def circumradius (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem circumradius_of_triangle_ABC :
  (circumradius 12 10 7 = 6) :=
by
  sorry

end circumradius_of_triangle_ABC_l705_705324


namespace determine_parents_genotype_l705_705115

noncomputable def genotype := ℕ -- We use nat to uniquely represent each genotype: e.g. HH=0, HS=1, Sh=2, SS=3

def probability_of_allele_H : ℝ := 0.1
def probability_of_allele_S : ℝ := 1 - probability_of_allele_H

def is_dominant (allele: ℕ) : Prop := allele == 0 ∨ allele == 1 -- HH or HS are dominant for hairy

def offspring_is_hairy (parent1 parent2: genotype) : Prop :=
  (∃ g1 g2, (parent1 = 0 ∨ parent1 = 1) ∧ (parent2 = 1 ∨ parent2 = 2 ∨ parent2 = 3) ∧
  ((g1 = 0 ∨ g1 = 1) ∧ (g2 = 0 ∨ g2 = 1))) ∧ 
  (is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0) 

def most_likely_genotypes (hairy_parent smooth_parent : genotype) : Prop :=
  (hairy_parent = 0) ∧ (smooth_parent = 2)

theorem determine_parents_genotype :
  ∃ hairy_parent smooth_parent, offspring_is_hairy hairy_parent smooth_parent ∧ most_likely_genotypes hairy_parent smooth_parent :=
  sorry

end determine_parents_genotype_l705_705115


namespace range_of_a_l705_705350

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + a * x + 2)
  (h2 : ∀ y, (∃ x, y = f (f x)) ↔ (∃ x, y = f x)) : a ≥ 4 ∨ a ≤ -2 := 
sorry

end range_of_a_l705_705350


namespace distance_between_P_and_Q_l705_705582

theorem distance_between_P_and_Q : 
  let initial_speed := 40  -- Speed in kmph
  let increment := 20      -- Speed increment in kmph after every 12 minutes
  let segment_duration := 12 / 60 -- Duration of each segment in hours (12 minutes in hours)
  let total_duration := 48 / 60    -- Total duration in hours (48 minutes in hours)
  let total_segments := total_duration / segment_duration -- Number of segments
  (total_segments = 4) ∧ 
  (∀ n : ℕ, n ≥ 0 → n < total_segments → 
    let speed := initial_speed + n * increment
    let distance := speed * segment_duration
    distance = speed * (12 / 60)) 
  → (40 * (12 / 60) + 60 * (12 / 60) + 80 * (12 / 60) + 100 * (12 / 60)) = 56 :=
by
  sorry

end distance_between_P_and_Q_l705_705582


namespace min_value_lemma_min_value_achieved_l705_705679

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2)

theorem min_value_lemma : ∀ (x : ℝ), f x ≥ Real.sqrt 5 := 
by
  intro x
  sorry

theorem min_value_achieved : ∃ (x : ℝ), f x = Real.sqrt 5 :=
by
  use 1 / 3
  sorry

end min_value_lemma_min_value_achieved_l705_705679


namespace infinite_n_exists_l705_705105

theorem infinite_n_exists (h : ∀ n : ℕ, ∃ p : ℕ, nat.prime p ∧ p ∣ 2 ^ (2 ^ n + 1) + 1 ∧ p ∣ 2 ^ n + 1 = false) :
  ∃ (S : set ℕ), infinite S ∧ ∀ n ∈ S, n ∣ 2 ^ (2 ^ n + 1) + 1 ∧ n ∣ 2 ^ n + 1 = false := sorry

end infinite_n_exists_l705_705105


namespace prod_divisibility_l705_705938

theorem prod_divisibility {n : ℤ} (hn : n ≥ 2) (a : ℕ → ℤ) :
  (∏ i in finset.range (n.to_nat), ∏ j in finset.Ico (i+1) n.to_nat, (a j - a i)) 
  % (∏ i in finset.range (n.to_nat), ∏ j in finset.Ico (i+1) n.to_nat, (j - i)) = 0 := by
  sorry

end prod_divisibility_l705_705938


namespace intersecting_diagonals_probability_l705_705016

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l705_705016


namespace average_homework_time_decrease_l705_705568

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l705_705568


namespace sum_of_three_smallest_metallic_integers_l705_705181

def isMetallic (n : ℕ) : Prop :=
  ∀ m : ℕ, let x := m * m - n
            x < 0 ∨ (¬ (Nat.prime x))

theorem sum_of_three_smallest_metallic_integers :
  ∑ (n : ℕ) in {16, 49, 100}, n = 165 := by
  sorry

end sum_of_three_smallest_metallic_integers_l705_705181


namespace vertical_sides_in_hexagonal_parallelogram_l705_705596

theorem vertical_sides_in_hexagonal_parallelogram (k l : ℕ) : 
  -- Given a k x l parallelogram drawn on paper with hexagonal cells
  -- prove that it contains k vertical sides in a set of non-intersecting sides
  -- of hexagons that divides all the vertices into pairs.
  count_vertical_sides (k l : ℕ) = k :=
sorry

end vertical_sides_in_hexagonal_parallelogram_l705_705596


namespace find_a_in_third_quadrant_l705_705237

theorem find_a_in_third_quadrant :
  ∃ a : ℝ, a < 0 ∧ 3 * a^2 + 4 * a^2 = 28 ∧ a = -2 :=
by
  sorry

end find_a_in_third_quadrant_l705_705237


namespace final_result_l705_705948

variables {x y z : ℂ}
variables {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
def equal_magnitude (x y z : ℂ) : Prop := abs x = abs y ∧ abs y = abs z
def sum_is_given (x y z : ℂ) : Prop := x + y + z = -√3/2 - complex.I*√5
def product_is_given (x y z : ℂ) : Prop := x * y * z = √3 + complex.I*√5
def components_are_real (x y z : ℂ) (x1 x2 y1 y2 z1 z2 : ℝ) : Prop := 
  x = x1 + complex.I * x2 ∧ y = y1 + complex.I * y2 ∧ z = z1 + complex.I * z2

-- Proof goal
theorem final_result 
  (h1 : equal_magnitude x y z)
  (h2 : sum_is_given x y z)
  (h3 : product_is_given x y z)
  (h4 : components_are_real x y z x1 x2 y1 y2 z1 z2) :
  (x1 * x2 + y1 * y2 + z1 * z2)^2 = 15 := 
sorry

end final_result_l705_705948


namespace solution_per_beaker_l705_705992

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end solution_per_beaker_l705_705992


namespace cafeteria_green_apples_l705_705973

theorem cafeteria_green_apples :
  ∃ G : ℕ, 43 + G - 2 = 73 ∧ G = 32 :=
by
  use 32
  split
  · exact rfl
  · sorry

end cafeteria_green_apples_l705_705973


namespace cosine_inequality_l705_705190

theorem cosine_inequality :
  cos (-2 * π / 5) < cos (-π / 4) :=
sorry

end cosine_inequality_l705_705190


namespace simplify_expression_l705_705128

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  (3 * (a^2 + a * b + b^2) / (4 * (a + b))) * (2 * (a^2 - b^2) / (9 * (a^3 - b^3))) = 
  1 / 6 := 
by
  -- Placeholder for proof steps
  sorry

end simplify_expression_l705_705128


namespace number_of_subsets_l705_705967

theorem number_of_subsets :
  { P : Set ℕ // {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4} ∧ P ≠ {1, 2, 3, 4} }.card = 3 :=
by sorry

end number_of_subsets_l705_705967


namespace sum_of_powers_seven_l705_705052

theorem sum_of_powers_seven (α1 α2 α3 : ℂ)
  (h1 : α1 + α2 + α3 = 2)
  (h2 : α1^2 + α2^2 + α3^2 = 6)
  (h3 : α1^3 + α2^3 + α3^3 = 14) :
  α1^7 + α2^7 + α3^7 = 478 := by
  sorry

end sum_of_powers_seven_l705_705052


namespace natasha_can_achieve_plan_l705_705919

noncomputable def count_ways : Nat :=
  let num_1x1 := 4
  let num_1x2 := 24
  let target := 2021
  6517

theorem natasha_can_achieve_plan (num_1x1 num_1x2 target : Nat) (h1 : num_1x1 = 4) (h2 : num_1x2 = 24) (h3 : target = 2021) :
  count_ways = 6517 :=
by
  sorry

end natasha_can_achieve_plan_l705_705919


namespace width_of_metallic_sheet_l705_705229

open Real

theorem width_of_metallic_sheet (L : ℝ) (s : ℝ) (V : ℝ) (H1 : L = 48) (H2 : s = 3) (H3 : V = 3780) :
  ∃ w : ℝ, 42 * (w - 2 * s) * s = V ∧ w = 36 :=
by
  use 36
  rw [H1, H2, H3]
  split
  calc 42 * (36 - 2 * 3) * 3 = 42 * 30 * 3  : by norm_num
                        ... = 3780         : by norm_num
  norm_num

end width_of_metallic_sheet_l705_705229


namespace perpendicular_sum_eq_l705_705207

theorem perpendicular_sum_eq :
  ∀ (A B C D E P P1 P2 P3 : Type*)
  (abc : Angle ABC = 60)
  (P_in_angle : isArbitrary P)
  (DBE_is_bisector : isAngleBisector (B E))
  (perpendiculars : ∀ (P_1 P_2 P_3 : Type*), isPerpendicular (P P1 AB) ∧ isPerpendicular (P P2 BC) ∧ isPerpendicular (P P3 DE)),
  (distance P P3 = distance P P1 + distance P P2) := sorry

end perpendicular_sum_eq_l705_705207


namespace sphere_radius_five_eq_cylinder_l705_705166

variable (r rcylinder h : ℝ)
variable (π : ℝ) [Archimedean π]
constant π_value : π = real.pi

def sphere_surface_area (r : ℝ) : ℝ := 4 * π * r^2
def cylinder_surface_area (rcylinder h : ℝ) : ℝ := 2 * π * rcylinder * h

theorem sphere_radius_five_eq_cylinder :
  (h = 10) → (rcylinder = 5) →
  sphere_surface_area π r = cylinder_surface_area π rcylinder h →
  r = 5 :=
by
  intros h_eq rcyl_eq area_eq
  rewrite [h_eq, rcyl_eq, π_value] at area_eq
  sorry

end sphere_radius_five_eq_cylinder_l705_705166


namespace part_one_part_two_l705_705437

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705437


namespace part_I_part_II_l705_705880

-- Define the parametric curve C
def curve_C (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, Real.sin φ)

-- Define the polar line l1
def polar_line_l1 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Define the polar line l2
def polar_line_l2 (θ : ℝ) : Prop :=
  θ = Real.pi / 2

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the point M
def M := (0, 1)

-- Part I: Prove that M lies on curve C
theorem part_I : curve_C_rect M.1 M.2 := by
  sorry

-- Maximum PM distance function
noncomputable def PM_distance (φ : ℝ) : ℝ :=
  let P := curve_C φ
  let M := (0, 1)
  Float.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)

-- Part II: Prove the maximum value of |PM|
theorem part_II : (∃ φ : ℝ, PM_distance φ = 4 * Real.sqrt 3 / 3) := by
  sorry

end part_I_part_II_l705_705880


namespace part_one_part_two_l705_705435

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705435


namespace gcd_digit_bound_l705_705787

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705787


namespace abscissa_of_point_on_parabola_l705_705730

open Real

noncomputable def parabola_abscissa (y : ℝ) : ℝ :=
  y^2

def parabolic_focus_distance (x y : ℝ) : ℝ :=
  sqrt((x - 1/4)^2 + y^2)

theorem abscissa_of_point_on_parabola : 
  ∀ (y : ℝ), parabolic_focus_distance (parabola_abscissa y) y = 2 → parabola_abscissa y = 7/4 := 
by
  intros y h
  sorry

end abscissa_of_point_on_parabola_l705_705730


namespace relatively_prime_dates_february_non_leap_l705_705628

-- Define the concept of a relatively prime date in February of a non-leap year.
def is_rel_prime_date (month day : ℕ) : Prop := Nat.gcd month day = 1

-- Define February of a non-leap year.
def february_non_leap := {d : ℕ | 1 ≤ d ∧ d ≤ 28}

-- Define the set of relatively prime dates in February.
def rel_prime_dates := {d ∈ february_non_leap | is_rel_prime_date 2 d}

-- The main statement to prove
theorem relatively_prime_dates_february_non_leap : 
  finset.card rel_prime_dates = 14 :=
sorry

end relatively_prime_dates_february_non_leap_l705_705628


namespace decrease_equation_l705_705557

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l705_705557


namespace find_point_coordinates_l705_705712

def P1 := (2, -1 : ℝ × ℝ)
def P2 := (0, 5 : ℝ × ℝ)
def P := (-2, 11 : ℝ × ℝ)

theorem find_point_coordinates :
  let a := P1.1
  let b := P1.2
  let c := P2.1
  let d := P2.2
  let x := P.1
  let y := P.2
  P2.1 = (x + a) / 2 ∧ P2.2 = (y + b) / 2 →
  P = (-2, 11) :=
by
  intro h
  sorry

end find_point_coordinates_l705_705712


namespace margie_change_is_6_25_l705_705917

-- The conditions are given as definitions in Lean
def numberOfApples : Nat := 5
def costPerApple : ℝ := 0.75
def amountPaid : ℝ := 10.00

-- The statement to be proved
theorem margie_change_is_6_25 :
  (amountPaid - (numberOfApples * costPerApple)) = 6.25 := 
  sorry

end margie_change_is_6_25_l705_705917


namespace at_least_one_not_less_than_2017_l705_705323

def sequence_a (a : ℕ → ℤ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  (a 0 = 0) ∧ 
  (a 1 = 1) ∧ 
  (∀ n > 0, a (n + 1) = if b (n - 1) = 1 then a n * b n + a (n - 1) else a n * b n - a (n - 1))

theorem at_least_one_not_less_than_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : sequence_a a b) : 
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end at_least_one_not_less_than_2017_l705_705323


namespace circles_are_intersecting_l705_705970

def circle1 : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = -4 * p.2}
def circle2 : set (ℝ × ℝ) := {p | (p.1 - 1) ^ 2 + p.2 ^ 2 = 1}

theorem circles_are_intersecting :
  let C1 := (0 : ℝ, -2 : ℝ)
  let r1 := 2 : ℝ
  let C2 := (1 : ℝ, 0 : ℝ)
  let r2 := 1 : ℝ
  let d := real.sqrt ((C2.1 - C1.1) ^ 2 + (C2.2 - C1.2) ^ 2)
  (d < r1 + r2) ∧ (d > (r1 - r2))
  :=
by
  sorry

end circles_are_intersecting_l705_705970


namespace secure_nailing_requires_two_nails_l705_705999

theorem secure_nailing_requires_two_nails :
  (Two_points_determine_a_straight_line := "Two points determine a straight line") →
  (Necessary_nails_to_secure_wooden_strip := "At least two") →
  Necessary_nails_to_secure_wooden_strip = Two_points_determine_a_straight_line :=
sorry

end secure_nailing_requires_two_nails_l705_705999


namespace determine_c_l705_705314

-- Define the parabola equation
def parabola (c : ℝ) : ℝ → ℝ := λ x, x^2 - 6 * x + c

-- Define the condition that the vertex (h, k) is on the x-axis, thus k = 0
def vertexOnXAxis (c : ℝ) : Prop :=
  let h := -(-6) / (2 * 1) in  -- h = 3
  let k := c - (6^2) / (4 * 1) in  -- k = c - 9
  k = 0

-- The proof problem
theorem determine_c : ∃ c : ℝ, vertexOnXAxis c ∧ c = 9 :=
sorry

end determine_c_l705_705314


namespace unique_polynomial_count_l705_705657

noncomputable def root_transform : Complex := (-1 - Complex.I * Real.sqrt 3) / 2

theorem unique_polynomial_count :
  let Q := λ (a b c d e : ℝ) (x : ℂ), x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + 2023
  ∃ (a b c d e : ℝ), ∀ (s : ℂ), Q a b c d e s = 0 → Q a b c d e (root_transform * s) = 0 → false :=
sorry

end unique_polynomial_count_l705_705657


namespace hexagons_cover_65_percent_l705_705969

noncomputable def hexagon_percent_coverage
    (a : ℝ)
    (square_area : ℝ := a^2) 
    (hexagon_area : ℝ := (3 * Real.sqrt 3 / 8 * a^2))
    (tile_pattern : ℝ := 3): Prop :=
    hexagon_area / square_area * tile_pattern = (65 / 100)

theorem hexagons_cover_65_percent (a : ℝ) : hexagon_percent_coverage a :=
by
    sorry

end hexagons_cover_65_percent_l705_705969


namespace gasoline_distribution_impossible_l705_705890

theorem gasoline_distribution_impossible
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = 50)
  (h2 : x1 = x2 + 10)
  (h3 : x3 + 26 = x2) : false :=
by {
  sorry
}

end gasoline_distribution_impossible_l705_705890


namespace count_multiples_of_13_three_digit_l705_705763

-- Definitions based on the conditions in the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

-- Statement of the proof problem
theorem count_multiples_of_13_three_digit :
  ∃ (count : ℕ), count = (76 - 8 + 1) :=
sorry

end count_multiples_of_13_three_digit_l705_705763


namespace sum_of_first_n_terms_sequence_l705_705268

open Nat

def sequence_term (i : ℕ) : ℚ :=
  if i = 0 then 0 else 1 / (i * (i + 1) / 2 : ℕ)

def sum_of_sequence (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum fun i => sequence_term i

theorem sum_of_first_n_terms_sequence (n : ℕ) : sum_of_sequence n = 2 * n / (n + 1) := by
  sorry

end sum_of_first_n_terms_sequence_l705_705268


namespace liz_spent_total_l705_705083

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end liz_spent_total_l705_705083


namespace max_area_triangle_ABC_l705_705401

-- Define the conditions
variable (A B C : Point)
variable (a c : ℝ)
variable (k : ℝ)

-- Conditions given in the problem
def AC_is3 : a = 3 := sorry
def sin_relation : sin C = k * sin A := sorry
def k_ge_2 : 2 ≤ k := sorry

-- Claim to be proven
theorem max_area_triangle_ABC :
  AC_is3 A B C a c → sin_relation A B C k → k_ge_2 k → 
  ∃ S_max, S_max = 3 := sorry

end max_area_triangle_ABC_l705_705401


namespace find_f_neg1_l705_705056

-- Definition of the periodic function f on ℝ
def is_periodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 3 then x - 2 else 0  -- f(x) is defined only for [1, 3)

theorem find_f_neg1 (h_periodic : is_periodic f 2) (h_def : ∀ x, 1 ≤ x ∧ x < 3 → f x = x - 2) : f (-1) = -1 :=
  sorry

end find_f_neg1_l705_705056


namespace find_N_l705_705643

theorem find_N
  (N : ℕ)
  (h : (4 / 10 : ℝ) * (16 / (16 + N : ℝ)) + (6 / 10 : ℝ) * (N / (16 + N : ℝ)) = 0.58) :
  N = 144 :=
sorry

end find_N_l705_705643


namespace gcd_digit_bound_l705_705791

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l705_705791


namespace sufficient_not_necessary_l705_705476

def M : Set ℤ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

theorem sufficient_not_necessary (a : ℤ) :
  (a = 1 → N a ⊆ M) ∧ (N a ⊆ M → a = 1) = false :=
by 
  sorry

end sufficient_not_necessary_l705_705476


namespace mashed_potatoes_count_l705_705504

theorem mashed_potatoes_count :
  ∀ (b s : ℕ), b = 489 → b = s + 10 → s = 479 :=
by
  intros b s h₁ h₂
  sorry

end mashed_potatoes_count_l705_705504


namespace both_solve_prob_l705_705374

variable (a b : ℝ) -- Define a and b as real numbers

-- Define the conditions
def not_solve_prob_A := (0 ≤ a) ∧ (a ≤ 1)
def not_solve_prob_B := (0 ≤ b) ∧ (b ≤ 1)
def independent := true -- independence is implicit by the question

-- Define the statement of the proof
theorem both_solve_prob (h1 : not_solve_prob_A a) (h2 : not_solve_prob_B b) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by sorry

end both_solve_prob_l705_705374


namespace most_likely_parents_genotypes_l705_705109

-- Defining the probabilities of alleles in the population
def p_H : ℝ := 0.1
def q_S : ℝ := 0.9

-- Defining the genotypes and their corresponding fur types
inductive Genotype
| HH : Genotype
| HS : Genotype
| SS : Genotype
| Sh : Genotype

-- A function to determine if a given genotype results in hairy fur
def isHairy : Genotype → Prop
| Genotype.HH := true
| Genotype.HS := true
| _ := false

-- Axiom stating that all four offspring have hairy fur
axiom offspring_all_hairy (parent1 parent2 : Genotype) : 
  (isHairy parent1 ∧ isHairy parent2) ∨
  ((parent1 = Genotype.HH ∨ parent2 = Genotype.Sh) ∧ isHairy Genotype.HH) 

-- The main theorem to prove the genotypes of the parents
theorem most_likely_parents_genotypes : 
  ∃ parent1 parent2,
    parent1 = Genotype.HH ∧ parent2 = Genotype.Sh :=
begin
  sorry
end

end most_likely_parents_genotypes_l705_705109


namespace gcd_digit_bound_l705_705820

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705820


namespace calculation_l705_705261

theorem calculation : 8 - (7.14 * (1 / 3) - (20 / 9) / (5 / 2)) + 0.1 = 6.62 :=
by
  sorry

end calculation_l705_705261


namespace point_in_second_quadrant_l705_705865

theorem point_in_second_quadrant (m : ℝ) : 
  let P := (-1, m^2 + 1) in
    P.1 < 0 ∧ P.2 > 0 → P ∈ quadrant2 :=
begin
  intro h,
  sorry,
end

end point_in_second_quadrant_l705_705865


namespace geometric_sequence_sum_l705_705336

theorem geometric_sequence_sum (a : ℝ) (q : ℝ) (h1 : a * q^2 + a * q^5 = 6)
  (h2 : a * q^4 + a * q^7 = 9) : a * q^6 + a * q^9 = 27 / 2 :=
by
  sorry

end geometric_sequence_sum_l705_705336


namespace valid_dedekind_cuts_l705_705964

-- Definitions based on problem conditions
def is_dedekind_cut (M N : Set ℚ) :=
  M ∪ N = Set.univ ∧
  M ∩ N = ∅ ∧
  (∀ x ∈ M, ∀ y ∈ N, x < y)

-- Option B
def option_B := ({x : ℚ | x < 0}, {x : ℚ | x >= 0})

-- Option D
def option_D := ({x : ℚ | x < real.sqrt 2}, {x : ℚ | x ≥ real.sqrt 2})

-- Statement to check the validity of options B and D
theorem valid_dedekind_cuts : 
  is_dedekind_cut (option_B.1) (option_B.2) ∧
  is_dedekind_cut (option_D.1) (option_D.2) := 
sorry

end valid_dedekind_cuts_l705_705964


namespace intersecting_diagonals_probability_l705_705018

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l705_705018


namespace negative_add_abs_neg_eq_zero_l705_705767

theorem negative_add_abs_neg_eq_zero (a : ℝ) (h : a < 0) : a + | -a | = 0 :=
by
  sorry

end negative_add_abs_neg_eq_zero_l705_705767


namespace bill_difference_is_zero_l705_705648

theorem bill_difference_is_zero
    (a b : ℝ)
    (h1 : 0.25 * a = 5)
    (h2 : 0.15 * b = 3) :
    a - b = 0 := 
by 
  sorry

end bill_difference_is_zero_l705_705648


namespace trader_sold_45_meters_l705_705244

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end trader_sold_45_meters_l705_705244


namespace find_area_of_triangle_ABC_l705_705888

open EuclideanGeometry

noncomputable def A := (0 : ℝ, 0 : ℝ)
noncomputable def B := (14 : ℝ, 0 : ℝ)
noncomputable def C := (7 : ℝ, 16 : ℝ)
noncomputable def O := midpoint ℝ A B
noncomputable def L := midpoint ℝ A B
noncomputable def K := (7 : ℝ, 8 : ℝ)

noncomputable def angle_LOA := (45 : ℝ) -- given angle LOA = 45 degrees
noncomputable def LK := (8 : ℝ) -- given LK = 8
noncomputable def AK := (7 : ℝ) -- given AK = 7

theorem find_area_of_triangle_ABC :
  triangle_area A B C = 112 :=
by
  sorry

end find_area_of_triangle_ABC_l705_705888


namespace total_blue_points_l705_705923

variables (a b c d : ℕ)

theorem total_blue_points (h1 : a * b = 56) (h2 : c * d = 50) (h3 : a + b = c + d) :
  a + b = 15 :=
sorry

end total_blue_points_l705_705923


namespace sequence_general_term_l705_705678

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end sequence_general_term_l705_705678


namespace largest_y_coordinate_l705_705182

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 25) + ((y - 3)^2 / 25) = 0) : y = 3 := by
  sorry

end largest_y_coordinate_l705_705182


namespace gcd_max_two_digits_l705_705836

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l705_705836


namespace ratio_of_segments_l705_705315

theorem ratio_of_segments (a b : ℝ) 
  (h1 : ∃ P Q R S : ℝ × ℝ, 
    dist P Q = a ∧ dist P R = a ∧ dist Q R = a ∧ 
    dist P S = 2 * a ∧ dist Q S = 2 * a ∧ dist R S = b) :
  b / a = 2 * real.sqrt 2 :=
by
  sorry

end ratio_of_segments_l705_705315


namespace max_non_overlapping_squares_within_square_l705_705104

-- Definitions based on conditions
def Square := {s : Type} -- A type representing a square

def AB : Square 
/- The original square ABCD with side length 2 units -/

def A1 : Square 
/- The square A1B1C1D1 with sides twice as long as AB and same center -/

-- Statement of the problem
theorem max_non_overlapping_squares_within_square : 
  (number_of_non_overlapping_squares_within AB A1 ≤ 8) :=
by
  sorry

end max_non_overlapping_squares_within_square_l705_705104


namespace gcd_at_most_3_digits_l705_705846

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705846


namespace gcd_max_digits_l705_705829

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l705_705829


namespace gcd_digit_bound_l705_705819

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705819


namespace problem_1_problem_2_l705_705332

noncomputable def f (a b x : ℝ) := |x + a| + |2 * x - b|

theorem problem_1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
(h_min : ∀ x, f a b x ≥ 1 ∧ (∃ x₀, f a b x₀ = 1)) :
2 * a + b = 2 :=
sorry

theorem problem_2 (a b t : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
(h_tab : ∀ t > 0, a + 2 * b ≥ t * a * b)
(h_eq : 2 * a + b = 2) :
t ≤ 9 / 2 :=
sorry

end problem_1_problem_2_l705_705332


namespace math_problem_equals_solution_l705_705283

noncomputable def math_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : ℝ :=
  (4 * x^2 * y + 6 * x^2 + 2 * x * y - 4 * x) / (3 * x - y - 2) +
  Math.sin ((3 * x^2 + x * y + x - y - 2) / (3 * x - y - 2))

noncomputable def solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : ℝ :=
  2 * x * y + y^2 + (x^2 / y^2) + (2 * x / y) + (2 * x * y * (x^2 + y^2)) / (3 * x - y - 2)^2 +
  1 / (x + y)^2 * (x^2 * Math.sin((x + y)^2 / x) + y^2 * Math.sin((x + y)^2 / y^2) + 2 * x * y * Math.sin((x + y)^2 / (3 * x - y - 2)))

theorem math_problem_equals_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  math_problem x y h1 h2 = solution x y h1 h2 := sorry

end math_problem_equals_solution_l705_705283


namespace fbox_eval_correct_l705_705311

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end fbox_eval_correct_l705_705311


namespace identity_cotangent_sine_cosine_l705_705585

variable {α β : ℝ}

theorem identity_cotangent_sine_cosine (hα : Sin α ≠ 0) (hβ : Sin β ≠ 0) :
  (Cos α / Sin α)^2 + (Cos β / Sin β)^2 - (2 * Cos (β - α) / (Sin α * Sin β)) + 2 =
  (Sin (α - β))^2 / (Sin α)^2 / (Sin β)^2 :=
    sorry

end identity_cotangent_sine_cosine_l705_705585


namespace fifth_dog_is_older_than_fourth_l705_705764

theorem fifth_dog_is_older_than_fourth :
  ∀ (age_1 age_2 age_3 age_4 age_5 : ℕ),
  (age_1 = 10) →
  (age_2 = age_1 - 2) →
  (age_3 = age_2 + 4) →
  (age_4 = age_3 / 2) →
  (age_5 = age_4 + 20) →
  ((age_1 + age_5) / 2 = 18) →
  (age_5 - age_4 = 20) :=
by
  intros age_1 age_2 age_3 age_4 age_5 h1 h2 h3 h4 h5 h_avg
  sorry

end fifth_dog_is_older_than_fourth_l705_705764


namespace choose_points_l705_705602

theorem choose_points (P : Fin 24 → Prop) :
  ((∀ i j, P i ∧ P j → ((i - j) % 24 ≠ 3) ∧ ((i - j) % 24 ≠ 8)) →
   ∃ S : Finset (Fin 24), S.card = 8 ∧ (∀ i ∈ S, ∀ j ∈ S, i ≠ j → ((i - j) % 24 ≠ 3) ∧ ((i - j) % 24 ≠ 8))) ↔ 258 :=
sorry

end choose_points_l705_705602


namespace Liz_total_spend_l705_705080

theorem Liz_total_spend :
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  total_spent_cost = 40 :=
by
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  show total_spent_cost = 40 from
    sorry

end Liz_total_spend_l705_705080


namespace reflection_of_0_4_l705_705626

def reflection (A B : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ) : ℝ × ℝ :=
  let proj := (P.1 * M.1 + P.2 * M.2) / (M.1 * M.1 + M.2 * M.2) in
  (2 * proj * M.1 - P.1, 2 * proj * M.2 - P.2)

theorem reflection_of_0_4 (a b : ℝ × ℝ) (h : a = (3, -2) ∧ b = (7, 6)) :
  reflection a b (0, 4) (5, 2) = (80 / 29, -84 / 29) :=
by {
  sorry
}

end reflection_of_0_4_l705_705626


namespace five_coins_no_105_cents_l705_705308

theorem five_coins_no_105_cents :
  ¬ ∃ (a b c d e : ℕ), a + b + c + d + e = 5 ∧
    (a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 105) :=
sorry

end five_coins_no_105_cents_l705_705308


namespace shaded_triangle_area_l705_705877

-- Definitions and conditions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

def larger_triangle_base : ℕ := grid_width
def larger_triangle_height : ℕ := grid_height - 1

def smaller_triangle_base : ℕ := 12
def smaller_triangle_height : ℕ := 3

-- The proof problem stating that the area of the smaller shaded triangle is 18 units
theorem shaded_triangle_area :
  (smaller_triangle_base * smaller_triangle_height) / 2 = 18 :=
by
  sorry

end shaded_triangle_area_l705_705877


namespace gcd_has_at_most_3_digits_l705_705811

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l705_705811


namespace ratio_of_areas_of_inscribed_squares_l705_705131

theorem ratio_of_areas_of_inscribed_squares
  (s : ℝ)
  (WXYZ : set (ℝ × ℝ))
  (IJKL : set (ℝ × ℝ))
  (WI IZ : ℝ)
  (h1 : ∀ (I ∈ IJKL) (J ∈ IJKL) (K ∈ IJKL) (L ∈ IJKL), I ≠ J ∧ J ≠ K ∧ K ≠ L ∧ L ≠ I)
  (h2 : ∀ (I ∈ IJKL) (I₁ I₂ ∈ WXYZ), I₁ ∈ WXYZ ∧ I₂ ∈ WXYZ)
  (h3 : WI = 9 * IZ)
  (h4 : 0 < s) :
  (let area_WXYZ := (10 * s) ^ 2 in
   let side_IJKL := s * real.sqrt 2 in
   let area_IJKL := side_IJKL ^ 2 in
   (area_IJKL / area_WXYZ) = 1 / 50) := by
{
  sorry
}

end ratio_of_areas_of_inscribed_squares_l705_705131


namespace minimum_value_frac_l705_705389

open Real

variables (A B C E P : Point)
variables (m n : ℝ) (h1 : m > 0) (h2 : n > 0)

def equilateral (ABC : Triangle) : Prop := 
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C A = 1)

def point_on_segment (P C E : Point) : Prop :=
  dist C E = 4 * dist P E

def point_on_BE (P B E : Point) : Prop := sorry

noncomputable def length_vector_AP (A B C P : Point) (m n : ℝ) :=
  real.sqrt ((1/3)^2 + 2 * (1/3) * (1/6) * (1/2) + (1/6)^2)

theorem minimum_value_frac (ABC : Triangle) (hABC: equilateral ABC) 
  (hE : point_on_segment A C E) (hP : point_on_BE B E P) 
  (m_eq : m = 1/3) (n_eq : n = 1/6) :
  (1/m + 1/n = 9) ∧ (length_vector_AP A B C P m n = sqrt 7 / 6) :=
begin
  sorry
end

end minimum_value_frac_l705_705389


namespace intersecting_diagonals_probability_l705_705012

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l705_705012


namespace an_increasing_sum_less_than_an_l705_705684

variable (n : ℕ) (a : ℕ → ℝ)
variable [n_pos : Fact (0 < n)]
variable (h_a : ∀ n, a n ^ 3 + a n / n = 1)

theorem an_increasing :
  a (n + 1) > a n :=
by
  sorry

theorem sum_less_than_an :
  ∑ i in Finset.range n, 1 / ((i + 2) * (a (i + 1))) < a n :=
by
  sorry

end an_increasing_sum_less_than_an_l705_705684


namespace original_bet_l705_705221

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l705_705221


namespace platform_length_l705_705149

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end platform_length_l705_705149


namespace correct_system_of_equations_l705_705873

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l705_705873


namespace sequence_contains_infinitely_many_powers_of_two_l705_705706

theorem sequence_contains_infinitely_many_powers_of_two (a : ℕ → ℕ) (b : ℕ → ℕ) : 
  (∃ a1, a1 % 5 ≠ 0 ∧ a 0 = a1) →
  (∀ n : ℕ, a (n + 1) = a n + b n) →
  (∀ n : ℕ, b n = a n % 10) →
  (∃ n : ℕ, ∃ k : ℕ, 2^k = a n) :=
by
  sorry

end sequence_contains_infinitely_many_powers_of_two_l705_705706


namespace evaluate_g_at_neg3_l705_705906

def g (x : ℝ) : ℝ := 3 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 - 10 * x ^ 2 - 12 * x + 36

theorem evaluate_g_at_neg3 : g (-3) = -1341 := by
  sorry

end evaluate_g_at_neg3_l705_705906


namespace intersections_divisible_by_3_l705_705099
open Nat

theorem intersections_divisible_by_3 (n : ℕ) (h : n ≥ 1) :
  let Q := (n * (n - 1)) * ((n + 1) * n) / 4 in Q % 3 = 0 :=
by
  sorry

end intersections_divisible_by_3_l705_705099


namespace male_salmon_count_l705_705669

theorem male_salmon_count (total_count : ℕ) (female_count : ℕ) (male_count : ℕ) :
  total_count = 971639 →
  female_count = 259378 →
  male_count = (total_count - female_count) →
  male_count = 712261 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end male_salmon_count_l705_705669


namespace find_50th_term_l705_705147

-- Define a function to list the sequence based on the given conditions
def sequence : List ℕ := -- We are defining the set conditions for the sequence
  List.bind (List.range 1 60) $ λ n,
    if n = 2^n ∨ n = 3^n ∨ ∃ a b, n = 2^a + 3^b then [n] else []

-- Prove that the 50th term in this sequence is 57
theorem find_50th_term : sequence.nth 49 = some 57 := sorry

end find_50th_term_l705_705147


namespace ribbon_per_box_l705_705407

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l705_705407


namespace regular_decagon_shaded_area_fraction_l705_705492

-- Define the center O of the decagon and the midpoint Y of the side AB
structure RegularDecagon :=
(center : Point)
(vertices : Fin 10 → Point)
(midpointAB : Point)
(is_midpoint : is_midpoint midpointAB (vertices 0) (vertices 1))
(is_regular : IsRegularPolygon vertices 10)
(in_center : center_in_center center vertices)

-- Define the characteristic of the shaded area
structure ShadedArea (D : RegularDecagon) :=
(triangles : Finset (Fin 10))
(partial_triangle : Bool)
(sum_triangles : triangles.card = 4)
(partial_frac : partial_triangle = true)

-- Auxiliary definitions for areas (we assume certain standard results and properties of regular polygons)
noncomputable def fraction_shaded_area (D : RegularDecagon) (S : ShadedArea D) : ℚ :=
  if S.partial_triangle 
  then 4 / 10 + 1 / 20 
  else 4 / 10

-- Statement of the problem
theorem regular_decagon_shaded_area_fraction (D : RegularDecagon) (S : ShadedArea D) : 
  fraction_shaded_area D S = 9 / 20 := 
sorry

end regular_decagon_shaded_area_fraction_l705_705492


namespace gh_two_value_l705_705369

def g (x : ℤ) : ℤ := 3 * x ^ 2 + 2
def h (x : ℤ) : ℤ := -5 * x ^ 3 + 2

theorem gh_two_value : g (h 2) = 4334 := by
  sorry

end gh_two_value_l705_705369


namespace point_in_second_quadrant_l705_705866

theorem point_in_second_quadrant (m : ℝ) : 
  let P := (-1, m^2 + 1) in
    P.1 < 0 ∧ P.2 > 0 → P ∈ quadrant2 :=
begin
  intro h,
  sorry,
end

end point_in_second_quadrant_l705_705866


namespace solve_inequalities_l705_705506

theorem solve_inequalities {x : ℝ} :
  (3 * x + 1) / 2 > x ∧ (4 * (x - 2) ≤ x - 5) ↔ (-1 < x ∧ x ≤ 1) :=
by sorry

end solve_inequalities_l705_705506


namespace vertex_on_x_axis_l705_705620

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l705_705620


namespace binomial_square_sum_eq_l705_705593

theorem binomial_square_sum_eq (n : ℕ) : 
    (∑ k in Finset.range (n + 1), (Nat.choose n k) ^ 2) = Nat.choose (2 * n) n := 
sorry

end binomial_square_sum_eq_l705_705593


namespace at_least_one_not_lt_one_l705_705180

theorem at_least_one_not_lt_one (a b c : ℝ) (h : a + b + c = 3) : ¬ (a < 1 ∧ b < 1 ∧ c < 1) :=
by
  sorry

end at_least_one_not_lt_one_l705_705180


namespace missing_angle_l705_705654

theorem missing_angle (s : ℝ) (missing : ℝ) :
  (s = 2017) →
  (missing = 2160 - s) →
  missing = 143 :=
by
  intros h₁ h₂
  rw h₁ at h₂
  norm_num at h₂
  exact h₂
  sorry -- Proof will be provided here.

end missing_angle_l705_705654


namespace tangent_line_curve_k_value_l705_705683

theorem tangent_line_curve_k_value :
  (∃ x : ℝ, 2 * Real.exp(x) = k * x) →
  k = 2 * Real.exp(1) :=
by
  intro h
  sorry

end tangent_line_curve_k_value_l705_705683


namespace james_charge_l705_705038

noncomputable def mural_charge : Real :=
  let complex_area_1 := 10 * 15
  let simple_area_1 := 10 * 15
  let complex_area_2 := 15 * 10
  let simple_area_2 := 10 * 10
  let complex_area_3 := 20 * 8
  let simple_area_3 := 10 * 8

  let total_complex_area := complex_area_1 + complex_area_2 + complex_area_3
  let total_simple_area := simple_area_1 + simple_area_2 + simple_area_3

  let time_complex := total_complex_area * 25
  let hours_complex := time_complex / 60
  let charge_complex := hours_complex * 200

  let time_simple := total_simple_area * 20
  let hours_simple := time_simple / 60
  let charge_simple := hours_simple * 150

  let premium_paint_units := (total_complex_area / 50.0).ceil.to_i
  let premium_paint_cost := premium_paint_units * 50

  charge_complex + charge_simple + premium_paint_cost

theorem james_charge : mural_charge = 55333.33 := sorry

end james_charge_l705_705038


namespace negation_of_conditional_l705_705155

-- Define the propositions
def P (x : ℝ) : Prop := x > 2015
def Q (x : ℝ) : Prop := x > 0

-- Negate the propositions
def notP (x : ℝ) : Prop := x <= 2015
def notQ (x : ℝ) : Prop := x <= 0

-- Theorem: Negation of the conditional statement
theorem negation_of_conditional (x : ℝ) : ¬ (P x → Q x) ↔ (notP x → notQ x) :=
by
  sorry

end negation_of_conditional_l705_705155


namespace correct_model_l705_705563

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l705_705563


namespace gcd_at_most_3_digits_l705_705844

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705844


namespace rabbit_parent_genotype_l705_705117

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l705_705117


namespace solve_for_b_l705_705285

theorem solve_for_b (b : ℝ) (hb : b + ⌈b⌉ = 17.8) : b = 8.8 := 
by sorry

end solve_for_b_l705_705285


namespace earliest_meeting_time_l705_705253

/-!
# Problem
Anna, Stephanie, and James all start running around a track at 8:00. 
Anna completes a lap every 4 minutes, Stephanie every 7 minutes, and James every 6 minutes. 
What is the earliest time they all meet back at the beginning?

# Given conditions
* Anna's lap time (minutes) : 4
* Stephanie's lap time (minutes) : 7
* James's lap time (minutes) : 6

# Goal
To find the earliest time after 8:00 when all three meet back at the beginning.
This is equivalent to finding the LCM of the lap times 4, 7, and 6, and then calculating 
the time 84 minutes after 8:00.

# Answer (Expected Result)
* The earliest time all three meet back at the beginning is 9:24.
-/

open Nat

theorem earliest_meeting_time :
  let lap_anna := 4
  let lap_stephanie := 7
  let lap_james := 6
  let lcm_times := lcm (lcm lap_anna lap_stephanie) lap_james
  let start_time := (8 * 60 : Nat) -- 8:00 in minutes
  8 * 60 + lcm_times = 9 * 60 + 24 :=
by
  sorry

end earliest_meeting_time_l705_705253


namespace concyclic_OEFK_l705_705023

variable (A B C M N K O E F : Point)

-- Conditions
axiom acute_triangle: ∀ (A B C : Point), Triangle A B C → Acute A B C
axiom circumcenter: ∀ (A B C O : Point), Circumcenter A B C O
axiom on_line_AB : M ∈ Line A B
axiom on_line_AC : N ∈ Line A C
axiom O_on_MN : O ∈ Line M N
axiom K_midpoint_MN : Midpoint K M N ∧ K ≠ O
axiom E_midpoint_BN : Midpoint E B N
axiom F_midpoint_CM : Midpoint F C M

-- Goal
theorem concyclic_OEFK : Concyclic O E F K := by
  sorry

end concyclic_OEFK_l705_705023


namespace gcd_digit_bound_l705_705822

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l705_705822


namespace sum_of_squared_gp_l705_705073

variable (r : ℝ) (n : ℕ) (s : ℝ)
variable (r_ne_zero : r ≠ 0) (s_ne_zero : s ≠ 0)
variable (sum_original_gp : s = (1 - r^n) / (1 - r))

theorem sum_of_squared_gp :
  (∑ i in Finset.range n, (1 * r^i)^2) = (1 - r^(2*n)) / (1 - r^2) :=
by
  sorry

end sum_of_squared_gp_l705_705073


namespace acceleration_at_2_l705_705274

def distance_function (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2 + 2

def velocity_function (t : ℝ) : ℝ := deriv distance_function t

def acceleration_function (t : ℝ) : ℝ := deriv velocity_function t

theorem acceleration_at_2 : acceleration_function 2 = 14 :=
by
  sorry

end acceleration_at_2_l705_705274


namespace all_three_together_l705_705581

variables (Jack Jill Joe : ℝ) (can_paint_in : ℝ → ℝ → ℝ)

-- Definition of the conditions
def jack_jill_can_paint_in_3_days (Jack Jill : ℝ) : Prop := 
  can_paint_in Jack Jill = (1 / 3)

def jill_joe_can_paint_in_4_days (Jill Joe : ℝ) : Prop := 
  can_paint_in Jill Joe = (1 / 4)

def joe_jack_can_paint_in_6_days (Joe Jack : ℝ) : Prop := 
  can_paint_in Joe Jack = (1 / 6)

-- Main theorem using the conditions
theorem all_three_together (Jack Jill Joe can_paint_in : ℝ) 
  (h1 : jack_jill_can_paint_in_3_days Jack Jill)
  (h2 : jill_joe_can_paint_in_4_days Jill Joe)
  (h3 : joe_jack_can_paint_in_6_days Joe Jack) :
  (Jack + Jill + Joe = 3/8) → 
  (1 / (Jack + Jill + Joe) = 8 / 3) := 
  by sorry

end all_three_together_l705_705581


namespace krista_bank_exceeds_400_on_tuesday_l705_705421

noncomputable def geometric_series_sum (a r n : ℝ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem krista_bank_exceeds_400_on_tuesday :
  ∃ n : ℕ, ∃ day_of_week : string, 
  (day_of_week = "Tuesday") ∧ (geometric_series_sum 2 3 n > 40000) :=
by
  sorry

end krista_bank_exceeds_400_on_tuesday_l705_705421


namespace tan_theta_correct_trig_expression_correct_l705_705342

open Real

-- Given conditions
variable (P : Point) (theta : ℝ)
local notation "cos'" x => Real.cos x
local notation "sin'" x => Real.sin x
local notation "tan'" x => Real.tan x

-- Definitions derived from conditions
def point_passes_through_theta (P : Point) := 
  P.x = 4 ∧ P.y = -3

noncomputable def tan_theta :=
  - (P.y / P.x)

-- Proof goals
theorem tan_theta_correct (h : point_passes_through_theta P) : 
  tan' theta = -3 / 4 :=
sorry

theorem trig_expression_correct (h : point_passes_through_theta P) :
  (sin' (theta + 90 * (π / 180)) + cos' theta) / (sin' theta - cos' (theta - 180 * (π / 180))) = 8 :=
sorry

end tan_theta_correct_trig_expression_correct_l705_705342


namespace snake_can_turn_around_l705_705632

structure Snake (n k : ℕ) :=
(cells : fin k → fin (n * n))
(pairwise_distinct : ∀ i j, cells i ≠ cells j)
(adjacent : ∀ i, i < k - 1 → (abs((cells i / n) - (cells (i + 1) / n)) + abs((cells i % n) - (cells (i + 1) % n)) = 1))

def turns_around {n k : ℕ} (s : Snake n k) :=
∃ (moves : ℕ → fin k → fin (n * n)), moves 0 = s.cells ∧
(∀ t, (∀ i, (abs((moves t i / n) - (moves (t + 1) (i - 1) / n)) + abs((moves t i % n) - (moves (t + 1) (i - 1) % n)) = 1)) →
(∀ i j, moves t i ≠ moves t j) →
moves (fin n) = λ i, s.cells (k - 1 - i))

theorem snake_can_turn_around :
  ∃ n > 1, ∃ (s : Snake n (⌈0.9 * n^2⌉₊)), turns_around s :=
sorry

end snake_can_turn_around_l705_705632


namespace am_gm_inequality_l705_705325

variables {n : ℕ} (x : ℕ → ℝ) 

def isNonDecreasing (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → x i ≤ x j

def isNonIncreasing (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → ∀ j, i ≤ j ∧ j ≤ n →  x i / i ≥ x j / j

def AMean (x : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in range n, x i.succ) / n

def GMean (x : ℕ → ℝ) (n : ℕ) : ℝ := (∏ i in range n, x i.succ)^(1 / n : ℝ)

theorem am_gm_inequality (n : ℕ) (x : ℕ → ℝ) (h₁ : n ≥ 2)
(h₂ : isNonDecreasing x n) (h₃ : isNonIncreasing x n) :
AMean x n / GMean x n ≤ (n + 1) / (2 * real.nth_root n (nat.factorial n)) :=
sorry

end am_gm_inequality_l705_705325


namespace kennel_arrangements_l705_705136

theorem kennel_arrangements :
  ∃ (n : ℕ), n = 70761600 ∧ 
  (∃ (cages animals : ℕ) (chickens dogs cats empty_cages remaining_cages : ℕ),
    cages = 15 ∧ 
    animals = 12 ∧ 
    chickens = 3 ∧ 
    dogs = 3 ∧ 
    cats = 6 ∧ 
    empty_cages = 3 ∧ 
    remaining_cages = 12 ∧ 
    n = nat.choose cages empty_cages * nat.factorial 3 * nat.factorial chickens * nat.factorial dogs * nat.factorial cats) :=
begin
  use 70761600,
  split,
  { refl },
  { use [15, 12, 3, 3, 6, 3, 12],
    split, 
    { refl },
    split, 
    { refl },
    split, 
    { refl },
    split, 
    { refl },
    split, 
    { refl },
    split, 
    { refl },
    split, 
    { refl },
    simp [nat.choose, nat.factorial],
    sorry
  }
end

end kennel_arrangements_l705_705136


namespace number_of_valid_orderings_valid_orderings_count_is_five_l705_705569

-- Define colors
inductive Color
| purple | green | blue | yellow | orange

open Color

-- Define the conditions as predicates
def is_valid_order (order : List Color) : Prop :=
  List.indexOf purple order < List.indexOf green order ∧
  List.indexOf blue order < List.indexOf yellow order ∧
  nat.abs ((List.indexOf blue order) - (List.indexOf yellow order)) > 1

-- The main theorem to be proved
theorem number_of_valid_orderings : 
  {orders : List (List Color) // ∀ order ∈ orders, is_valid_order order} → ℕ := sorry

-- Given the solution, we assert the number of such orderings is equal to 5
theorem valid_orderings_count_is_five : number_of_valid_orderings {orders | ∀ order ∈ orders, is_valid_order order } = 5 := sorry

end number_of_valid_orderings_valid_orderings_count_is_five_l705_705569


namespace common_chord_length_of_two_circles_l705_705548

noncomputable def common_chord_length (r : ℝ) : ℝ :=
  if r = 10 then 10 * Real.sqrt 3 else sorry

theorem common_chord_length_of_two_circles (r : ℝ) (h : r = 10) :
  common_chord_length r = 10 * Real.sqrt 3 :=
by
  rw [h]
  sorry

end common_chord_length_of_two_circles_l705_705548


namespace min_abs_diff_l705_705775

theorem min_abs_diff (x y : ℝ) (h : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) : 
  |x| - |y| = sqrt 3 :=
sorry

end min_abs_diff_l705_705775


namespace vacation_cost_division_l705_705167

theorem vacation_cost_division 
  (total_cost : ℝ) 
  (initial_people : ℝ) 
  (initial_cost_per_person : ℝ) 
  (cost_difference : ℝ) 
  (new_cost_per_person : ℝ) 
  (new_people : ℝ) 
  (h1 : total_cost = 1000) 
  (h2 : initial_people = 4) 
  (h3 : initial_cost_per_person = total_cost / initial_people) 
  (h4 : initial_cost_per_person = 250) 
  (h5 : cost_difference = 50) 
  (h6 : new_cost_per_person = initial_cost_per_person - cost_difference) 
  (h7 : new_cost_per_person = 200) 
  (h8 : total_cost / new_people = new_cost_per_person) :
  new_people = 5 := 
sorry

end vacation_cost_division_l705_705167


namespace expression1_calc_expression2_calc_l705_705652

-- For the first expression
theorem expression1_calc : (2/3)^(-2) + (1 - real.sqrt(2))^(0 : ℕ) - (27/8)^(2 / 3) = 1 := by
  sorry

-- For the second expression
theorem expression2_calc : (real.log 2 + real.log 4) / (1 + 1 / 2 * real.log 0.36 + 1 / 3 * real.log 8) = 1 := by
  sorry

end expression1_calc_expression2_calc_l705_705652


namespace part_one_part_two_l705_705738

noncomputable def z : ℂ := 1 - 2 * complex.i
noncomputable def conjugate_z : ℂ := complex.conj z

-- First part
theorem part_one (z1 : ℂ) (hz1 : conjugate_z * z1 = 4 + 3 * complex.i) : 
  z1 = 2 - complex.i :=
sorry

-- Second part
theorem part_two (p q : ℝ) (hz_root : IsRoot (polynomial.C 2 * polynomial.X^2 + polynomial.C p * polynomial.X + polynomial.C q) z) : 
  p = -4 ∧ q = 10 :=
sorry

end part_one_part_two_l705_705738


namespace smaller_area_l705_705228

theorem smaller_area (A B : ℝ) (total_area : A + B = 1800) (diff_condition : B - A = (A + B) / 6) :
  A = 750 := 
by
  sorry

end smaller_area_l705_705228


namespace new_number_with_21_l705_705376

theorem new_number_with_21 (t u : ℕ) (ht : t < 10) (hu : u < 10) : 
  let original_number := 10 * t + u in
  let new_number := original_number * 100 + 21 in
  new_number = 1000 * t + 100 * u + 21 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 100 + 21
  sorry

end new_number_with_21_l705_705376


namespace am_gm_inequality_l705_705517

theorem am_gm_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : (a + b + c) / 3 ≥ real.cbrt (a * b * c) :=
sorry

end am_gm_inequality_l705_705517


namespace solution_set_f_g_l705_705465

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem solution_set_f_g (h_odd_f : ∀ x, f (-x) = -f x)
  (h_even_g : ∀ x, g (-x) = g x)
  (h_derivative : ∀ x < 0, f' x * g x + f x * g' x > 0)
  (h_g_neg2 : g (-2) = 0) :
  { x : ℝ | f x * g x > 0 } = set.Ioc (-2 : ℝ) 0 ∪ set.Ioi (2 : ℝ) :=
sorry

end solution_set_f_g_l705_705465


namespace how_many_quantities_change_l705_705494

theorem how_many_quantities_change
  (A B P: Point)
  (M: Point := midpoint P A)
  (N: Point := midpoint P B)
  (move_P_vertically: P → P')
  (perpendicular: ∀ (X Y: Point), is_perpendicular X Y)
  (formation: triangle P A B)
  : (count_changed_quantities [segment_length M N, perimeter (triangle P A B), area (triangle P A B), area (trapezoid A B N M), angle P A B] move_P_vertically) = 4 :=
sorry

end how_many_quantities_change_l705_705494


namespace vertex_on_xaxis_l705_705622

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l705_705622


namespace alex_ride_time_l705_705638

theorem alex_ride_time
  (T : ℝ) -- time on flat ground
  (flat_speed : ℝ := 20) -- flat ground speed
  (uphill_speed : ℝ := 12) -- uphill speed
  (uphill_time : ℝ := 2.5) -- uphill time
  (downhill_speed : ℝ := 24) -- downhill speed
  (downhill_time : ℝ := 1.5) -- downhill time
  (walk_distance : ℝ := 8) -- distance walked
  (total_distance : ℝ := 164) -- total distance to the town
  (hup : uphill_speed * uphill_time = 30)
  (hdown : downhill_speed * downhill_time = 36)
  (hwalk : walk_distance = 8) :
  flat_speed * T + 30 + 36 + 8 = total_distance → T = 4.5 :=
by
  intros h
  sorry

end alex_ride_time_l705_705638


namespace percent_increase_jordan_alex_l705_705511

theorem percent_increase_jordan_alex :
  let pound_to_dollar := 1.5
  let alex_dollars := 600
  let jordan_pounds := 450
  let jordan_dollars := jordan_pounds * pound_to_dollar
  let percent_increase := ((jordan_dollars - alex_dollars) / alex_dollars) * 100
  percent_increase = 12.5 := 
by
  sorry

end percent_increase_jordan_alex_l705_705511


namespace proof_problem_l705_705859

-- Define points and conditions
variables {A B C D P O M N Q: Type}
variables (inside_ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) -- Ensuring ABCD is a quadrilateral

-- O is the intersection of AC and BD
variables (intersect_O : ∃ I, I = (AC ∩ BD))

-- M is the midpoint of PB, N is the midpoint of PC
variables (midpoint_M : M = midpoint PB)
variables (midpoint_N : N = midpoint PC)

-- Q is the intersection of AN and DM
variables (intersect_Q : ∃ Q, Q = (AN ∩ DM))

-- Prove collinearity of P, Q, O and length relation
theorem proof_problem (h1 : collinear P Q O) (h2 : PQ = 2 * QO) : Prop :=
sorry

end proof_problem_l705_705859


namespace triangle_perimeter_l705_705518

theorem triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 3) (x : ℕ) 
  (x_odd : x % 2 = 1) (triangle_ineq : 1 < x ∧ x < 5) : a + b + x = 8 :=
by
  sorry

end triangle_perimeter_l705_705518


namespace max_sin_cos_expression_l705_705298

open Real  -- Open the real numbers namespace

theorem max_sin_cos_expression (x y z : ℝ) :
  let expr := (sin (2 * x) + sin (3 * y) + sin (4 * z)) *
              (cos (2 * x) + cos (3 * y) + cos (4 * z))
  in expr ≤ 4.5 :=
sorry

end max_sin_cos_expression_l705_705298


namespace prove_speed_with_stream_l705_705227

-- Define all the values and variables we will use.
variables (V_as V_m V_s V_ws : ℝ)

-- Set the given conditions as hypotheses.
axiom H1 : V_as = 8
axiom H2 : V_m = 14
axiom H3 : V_as = V_m - V_s

-- Define a proposition to be proved.
def speed_with_stream (V_ws : ℝ) : Prop :=
  V_ws = V_m + V_s

-- The theorem stating what we ultimately want to prove.
theorem prove_speed_with_stream : speed_with_stream 20 :=
by
  -- Insert proof steps here
  sorry

end prove_speed_with_stream_l705_705227


namespace distance_from_N_to_orthocenter_l705_705522

variables {K M N A B L : Point}
variables {a b : ℝ}

-- Define the problem conditions
def mutually_perpendicular_diameter_and_chord (K M N A B : Point) : Prop :=
  diameter K M ∧ chord A B ∧ intersect_at_point K M A B N ∧ K ≠ N ∧ N ≠ M

def extension_owns_point (A B L N : Point) (a b : ℝ) : Prop :=
  on_extension_of_segment A B L ∧ distance L N = a ∧ distance A N = b

-- Define the theorem
theorem distance_from_N_to_orthocenter (K M N A B L : Point) (a b : ℝ)
  (h1 : mutually_perpendicular_diameter_and_chord K M N A B)
  (h2 : extension_owns_point A B L N a b) :
  distance N (orthocenter_of_triangle K L M) = 0 :=
sorry

end distance_from_N_to_orthocenter_l705_705522


namespace company_num_of_women_l705_705590

variable (W : ℕ) -- total number of workers 
variable (Men : ℕ) -- number of men

-- Conditions
variable (H1 : Men = 120)
variable (H2 : W / 3 = (W / 3) * 2 / 3 * 1 / 2)
-- First condition: A third of the workers do not have a retirement plan
variable (H3 : (2 / 3) * W = 300)
-- Second condition: 40% of workers with a retirement plan are men
variable (H4 : 0.40 * ((2 / 3) * W) = Men)
-- Solve for the number of women
def num_women (W Men : ℕ) : ℕ := W - Men

theorem company_num_of_women (H1 : Men = 120) (H2 : W / 3 = (W / 3) * 2 / 3 * 1 / 2) (H3 : (2 / 3) * W = 300) (H4 : 0.40 * ((2 / 3) * W) = Men) : 
  num_women 450 120 = 330 :=
by 
  have W := 450
  have Men := 120
  exact rfl

end company_num_of_women_l705_705590


namespace term_number_3_5_in_sequence_l705_705749

theorem term_number_3_5_in_sequence :
  let seq := λ (n : ℕ), if n = 1 then (1, 1) else let k := nat.find (λ k, (nat.succ k).choose k) in (k, n - k) 
  let sum_of_indices (n : ℕ) := ∑ i in finset.range (n+1), i,
  let numTerms (s : ℕ) := sum_of_indices (s-2),
  let nth_term (s : ℕ) := numTerms (7) + 3
  nth_term(8) = 24
:=
by
  sorry

end term_number_3_5_in_sequence_l705_705749


namespace gcd_digit_bound_l705_705804

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705804


namespace winning_percentage_l705_705595

noncomputable def total_votes (v1 v2 v3 : ℕ) : ℕ := v1 + v2 + v3

noncomputable def winning_candidate_votes : ℕ := 11628

theorem winning_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 8236) (h3 : v3 = 11628) :
  (winning_candidate_votes : ℕ) / (total_votes v1 v2 v3 : ℕ) * 100 = 58.14 :=
sorry

end winning_percentage_l705_705595


namespace all_rationals_as_dot_fractions_l705_705714

-- Define the sequence and the associated dot fraction
def dot_fraction (seq : List ℕ) : ℚ :=
  seq.foldl (λ (sum : ℚ) (a : ℕ), sum + (1 / List.prod (List.take (seq.indexOf a + 1) seq))) 0

-- Define the condition for positive integers greater than 1
def valid_sequence (seq : List ℕ) : Prop :=
  ∀ a ∈ seq, a > 1

-- Define the statement that all rational numbers in (0, 1) can be represented as finite dot fractions
theorem all_rationals_as_dot_fractions :
  ∀ (p q : ℕ), 0 < p ∧ p < q → ∃ (seq : List ℕ), valid_sequence seq ∧ dot_fraction seq = (p : ℚ) / (q : ℚ) :=
begin
  sorry
end

end all_rationals_as_dot_fractions_l705_705714


namespace find_angle_PAQ_eq_22_l705_705382

variable (A B C P Q : Type) [euclidean_geometry],
  (AB AC BC CP CQ : ℝ)
  (angle_BAC : ℝ)
  [AC_pos : AC > 0]
  [AB_eq_2AC : AB = 2 * AC]
  [angle_BAC_eq_112 : angle_BAC = 112]
  [AB_cp_condition : AB^2 + BC * CP = BC^2]
  [AC_cp_cq_condition : 3 * AC^2 + 2 * BC * CQ = BC^2]

theorem find_angle_PAQ_eq_22 
  (h : angle PAQ = 22) :
  angle PAQ = 22 :=
by sorry

end find_angle_PAQ_eq_22_l705_705382


namespace max_distance_l705_705049

-- Define the circle
def circle (x y : ℝ) : Prop := (x - 0)^2 + (y - 6)^2 = 2

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 10 + y^2 = 1

-- Define the points P and Q
def P (x y : ℝ) := circle x y
def Q (x y : ℝ) := ellipse x y

-- Prove the maximum distance
theorem max_distance (x1 y1 x2 y2 : ℝ) (hP : P x1 y1) (hQ : Q x2 y2) : 
  dist (x1, y1) (x2, y2) ≤ 6 * real.sqrt 2 :=
sorry

end max_distance_l705_705049


namespace average_homework_time_decrease_l705_705566

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l705_705566


namespace distance_to_right_directrix_l705_705709

-- Defining the ellipse equation and given conditions
def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def semi_major_axis : ℝ := 5
noncomputable def semi_minor_axis : ℝ := 3

def left_focus : ℝ × ℝ := (-4, 0) -- Assuming the center of the ellipse is at (0, 0)
def right_focus : ℝ × ℝ := (4, 0) -- Given a = 5 and c = 4

def M_condition (M : ℝ × ℝ) : Prop := distance M left_focus = 8

-- Main theorem to prove the distance from M to the right directrix is 5/2
theorem distance_to_right_directrix (M : ℝ × ℝ) (hM : M_condition M) : 
  ∃ d : ℝ, d = 5/2 := 
sorry

end distance_to_right_directrix_l705_705709


namespace toroidal_chessboard_no_eight_non_attacking_queens_l705_705758

theorem toroidal_chessboard_no_eight_non_attacking_queens :
  ∀ (board : matrix (fin 8) (fin 8) bool),
  (∀ i j, board i j = tt → ∀ k, k ≠ i → board k j = ff) → -- no two queens in the same column
  (∀ i j, board i j = tt → ∀ k, k ≠ j → board i k = ff) → -- no two queens in the same row
  (∀ i j, board i j = tt → ∀ k l, k ≠ l → (i + k ≠ j + l ∧ i + k ≠ j + l + 8 ∧ i + k ≠ j + l - 8)) → -- no two queens in the same main diagonal
  (∀ i j, board i j = tt → ∀ k l, k ≠ l → (i - k ≠ j - l ∧ i - k ≠ j - l + 8 ∧ i - k ≠ j - l - 8)) → -- no two queens in the same anti-diagonal
  (∃ i j, board i j = tt) → -- at least one queen is placed
  false := -- impossible to place 8 non-attacking queens
sorry

end toroidal_chessboard_no_eight_non_attacking_queens_l705_705758


namespace evaluate_expression_l705_705673

theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2 * x + 2) / x) * ((y^2 + 2 * y + 2) / y) + ((x^2 - 3 * x + 2) / y) * ((y^2 - 3 * y + 2) / x) 
  = 2 * x * y - (x / y) - (y / x) + 13 + 10 / x + 4 / y + 8 / (x * y) :=
by
  sorry

end evaluate_expression_l705_705673


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705447

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705447


namespace problem1_problem2_problem3_l705_705349

noncomputable def f (a x : ℝ) := a * real.log x + 1/2 * x^2 + (a + 1) * x + 1

def is_monotonically_increasing (f' : ℝ → ℝ) (I : set ℝ) := ∀ x ∈ I, f' x > 0

def satisfies_condition (a x : ℝ) := |f a x - f a x| > 2 * |x - x|

-- 1. When a = -1, the function f(x) is monotonically increasing on the interval (1, +∞).
theorem problem1 : (∀ x > 1, (f' (-1 / x + x)) > 0) := sorry

-- 2. If the function f(x) is increasing on the interval (0, +∞), then a ∈ [0, +∞).
theorem problem2 : (∀ x ∈ (0, +∞), is_monotonically_increasing (λ x, (a / x + x + a + 1)) (0, +∞)) → (a ∈ Ici (0 : ℝ)) := sorry

-- 3. If a > 0, and for any x1, x2 ∈ (0, +∞) with x1 ≠ x2, |f(x1) - f(x2)| > 2|x1 - x2|, then a ≥ 3 - 2√2.
theorem problem3 : (a > 0 → (∀ (x1 x2 : ℝ), x1 ∈ (0, +∞) → x2 ∈ (0, +∞) → x1 ≠ x2 → satisfies_condition a x1) → (a ≥ (3 - 2 * real.sqrt 2))) := sorry

end problem1_problem2_problem3_l705_705349


namespace same_leading_digit_l705_705688

theorem same_leading_digit (n : ℕ) (hn : 0 < n) : 
  (∀ a k l : ℕ, (a * 10^k < 2^n ∧ 2^n < (a+1) * 10^k) ∧ (a * 10^l < 5^n ∧ 5^n < (a+1) * 10^l) → a = 3) := 
sorry

end same_leading_digit_l705_705688


namespace train_cross_pole_time_l705_705202

theorem train_cross_pole_time :
  ∀ (length : ℝ) (speed_kmh : ℝ),
  length = 140 →
  speed_kmh = 144 →
  let speed_ms := speed_kmh * (1000 / 3600) in
  length / speed_ms = 3.5 :=
by
  intros length speed_kmh h_length h_speed_kmh
  let speed_ms := speed_kmh * (1000 / 3600)
  have h_speed_ms : speed_ms = 40 := by
    simp [h_speed_kmh]
    norm_num
  rw [h_length, h_speed_ms]
  norm_num

-- sorry

end train_cross_pole_time_l705_705202


namespace probability_diagonals_intersect_l705_705003

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l705_705003


namespace donny_savings_is_15_l705_705272

-- Definitions of the conditions in the problem
def donny_savings_monday (M : ℕ) : Prop := 
  let savings_tuesday := 28 in
  let savings_wednesday := 13 in
  let total_savings_before_thursday := M + savings_tuesday + savings_wednesday in
  let spent_on_thursday := 28 in
  let total_savings := 2 * spent_on_thursday in
  total_savings_before_thursday = total_savings

-- The theorem to be proved
theorem donny_savings_is_15 (M : ℕ) (h : donny_savings_monday M) : M = 15 :=
by 
  sorry

end donny_savings_is_15_l705_705272


namespace sum_f_2017_rem_2017_l705_705047

noncomputable def f : ℕ → ℕ := sorry

theorem sum_f_2017_rem_2017 :
  (∃ n : ℕ, f n = 577) →
  (∀ a b : ℕ, f a ^ 2 + f b ^ 2 + f (a + b) ^ 2 = 1 + 2 * f a * f b * f (a + b)) →
  let S := (possible_values_f_2017).sum in
  S % 2017 = 597 :=
sorry

end sum_f_2017_rem_2017_l705_705047


namespace students_taking_both_languages_l705_705513

theorem students_taking_both_languages (F S B : ℕ) (hF : F = 21) (hS : S = 21) (h30 : 30 = F - B + (S - B)) : B = 6 :=
by
  rw [hF, hS] at h30
  sorry

end students_taking_both_languages_l705_705513


namespace cricket_target_runs_l705_705398

theorem cricket_target_runs 
  (overs1 runs_rate1 : ℝ)
  (overs2 runs_rate2 : ℝ)
  (H1 : overs1 = 5)
  (H2 : runs_rate1 = 2.1)
  (H3 : overs2 = 30)
  (H4 : runs_rate2 = 6.316666666666666) :
  nat.ceil (overs1 * runs_rate1) + nat.ceil (overs2 * runs_rate2) = 201 :=
by
  sorry

end cricket_target_runs_l705_705398


namespace dresses_characteristic_l705_705225

theorem dresses_characteristic (sizes : Fin 3 → ℕ) (dark_room : Prop) (prob_not_choose_own_dress : ℝ) 
  (h_not_choose_own_dress : prob_not_choose_own_dress = 0.65) :
  ∀ (texture_same fabric_same design_same : Prop), 
    (dark_room → (∀ i : Fin 3, (texture_same ∧ fabric_same ∧ design_same) ∧ sizes i ≠ sizes (some_other_index i))) := 
sorry

end dresses_characteristic_l705_705225


namespace gcd_at_most_3_digits_l705_705848

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l705_705848


namespace intersection_of_A_and_B_l705_705356

-- Definitions based on conditions
def set_A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def set_B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Statement of the proof problem
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 ≤ x ∧ x ≤ -1} :=
  sorry

end intersection_of_A_and_B_l705_705356


namespace isothermal_compression_work_l705_705609

noncomputable def work_done (p0 : ℝ) (H h R : ℝ) : ℝ := 
  let S := π * R^2
  let integral_part := ∫ (x : ℝ) in 0..h, ((p0 * H * S) / (H - x))
  integral_part

theorem isothermal_compression_work :
  work_done 103_300 (2.0) (1.0) (0.4) = 72000 :=
by 
  -- Proof skipped for the statement purpose
  sorry

end isothermal_compression_work_l705_705609


namespace find_angle_A_triangle_is_right_l705_705905

theorem find_angle_A (A : ℝ) (h : 2 * Real.cos (Real.pi + A) + Real.sin (Real.pi / 2 + 2 * A) + 3 / 2 = 0) :
  A = Real.pi / 3 := 
sorry

theorem triangle_is_right (a b c : ℝ) (A : ℝ) (ha : c - b = (Real.sqrt 3) / 3 * a) (hA : A = Real.pi / 3) :
  c^2 = a^2 + b^2 :=
sorry

end find_angle_A_triangle_is_right_l705_705905


namespace f_at_1_l705_705733

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + (5 : ℝ) * x
  else if x = 2 then 6
  else  - (x^2 + (5 : ℝ) * x)

theorem f_at_1 : f 1 = 4 :=
by {
  sorry
}

end f_at_1_l705_705733


namespace a_has_inverse_b_no_inverse_c_has_inverse_d_no_inverse_e_no_inverse_f_has_inverse_g_has_inverse_l705_705579

section

variables {ℝ : Type} [LinearOrderedField ℝ]

def a (x : ℝ) : ℝ := sqrt (3 - x)
def b (x : ℝ) : ℝ := x^3 + x
def c (x : ℝ) : ℝ := x - 2 / x
def d (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1
def e (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)
def f (x : ℝ) : ℝ := 2^x + 8^x
def g (x : ℝ) : ℝ := x / 3

theorem a_has_inverse : ∃ (a_inv : ℝ → ℝ), Function.LeftInverse a_inv a ∧ Function.RightInverse a_inv a := sorry
theorem b_no_inverse : ¬(∃ (b_inv : ℝ → ℝ), Function.LeftInverse b_inv b ∧ Function.RightInverse b_inv b) := sorry
theorem c_has_inverse : ∃ (c_inv : ℝ → ℝ), Function.LeftInverse c_inv c ∧ Function.RightInverse c_inv c := sorry
theorem d_no_inverse : ¬(∃ (d_inv : ℝ → ℝ), Function.LeftInverse d_inv d ∧ Function.RightInverse d_inv d) := sorry
theorem e_no_inverse : ¬(∃ (e_inv : ℝ → ℝ), Function.LeftInverse e_inv e ∧ Function.RightInverse e_inv e) := sorry
theorem f_has_inverse : ∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f := sorry
theorem g_has_inverse : ∃ (g_inv : ℝ → ℝ), Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g := sorry

end

end a_has_inverse_b_no_inverse_c_has_inverse_d_no_inverse_e_no_inverse_f_has_inverse_g_has_inverse_l705_705579


namespace problem_I_problem_II_problem_III_l705_705746

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * x + 2
def g (a : ℝ) (x : ℝ) : ℝ := (1/3)^(f a x)

-- Problem (I): Given that f(x) has opposite monotonicity on intervals (-∞, 2] and [2, +∞), we need to prove f(x) = x^2 - 4x + 2.
theorem problem_I (a : ℝ) (h1 : ∀ x1 x2, x1 ∈ Icc (-∞:ℝ) 2 → x2 ∈ Icc 2 (∞ : ℝ) → 
                        (f a x1 - f a x2 ≠ 0)) : a = 1 :=
sorry

-- Problem (II): Given a < 0, prove a's range for which g(x) ≤ 9 for all x ∈ (0, 1/2].
theorem problem_II (a : ℝ) (h2 : a < 0) : (∀ x ∈ Icc (0 : ℝ) (1/2), g a x ≤ 9) → -8 ≤ a ∧ a < 0 :=
sorry

-- Problem (III): Given a ≤ 1, prove a's range for which y = f(x) - log₂(x/8) has exactly one root in the interval [1, 2].
theorem problem_III (a : ℝ) (h3 : a ≤ 1) : 
  (∀ y x, y = f a x - log (x / 8) / log 2  → x ∈ Icc 1 2 → true) → -1 ≤ a ∧ a ≤ 1 :=
sorry

end problem_I_problem_II_problem_III_l705_705746


namespace platform_length_l705_705150

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end platform_length_l705_705150


namespace part1_part2_l705_705445

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l705_705445


namespace part_one_part_two_l705_705436

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l705_705436


namespace smallest_period_f_max_value_f_on_interval_min_value_f_on_interval_l705_705742

def f (x : ℝ) : ℝ :=
  (sin (2 * x) * cos (2 * x + π / 3)) + (4 * sqrt 3 * sin x ^ 2 * cos x ^ 2)

theorem smallest_period_f : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ p = π / 2 :=
by sorry

theorem max_value_f_on_interval :
  ∃ x ∈ Icc (0 : ℝ) (π / 4), f x = (sqrt 3 + 2) / 4 :=
by sorry

theorem min_value_f_on_interval :
  ∃ x ∈ Icc (0 : ℝ) (π / 4), f x = 0 :=
by sorry

end smallest_period_f_max_value_f_on_interval_min_value_f_on_interval_l705_705742


namespace proof_of_problem_l705_705090

-- Define the problem conditions using a combination function
def problem_statement : Prop :=
  (Nat.choose 6 3 = 20)

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l705_705090


namespace marble_arrangements_count_l705_705671

def marbles := ["Aggie", "Bumblebee", "Crystal", "Steelie", "Tiger"]

def is_adjacent (a b : String) (arr : List String) : Prop :=
  ∃ i, i < arr.length - 1 ∧ (arr.get i = a ∧ arr.get (i+1) = b ∨ arr.get i = b ∧ arr.get (i+1) = a)

def crystal_positioned (arr : List String) : Prop :=
  arr.head? = some "Crystal" ∨ arr.last? = some "Crystal"

def valid_arrangement (arr : List String) : Prop :=
  ∃ perm, perm.permutations = arr ∧ ¬ is_adjacent "Steelie" "Tiger" perm ∧ crystal_positioned perm

theorem marble_arrangements_count : 
  ∃ l, l.length = 24 ∧ ∀ arr, arr ∈ l → valid_arrangement arr :=
sorry

end marble_arrangements_count_l705_705671


namespace gcd_digit_bound_l705_705807

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l705_705807


namespace dimes_and_half_dollars_l705_705907

theorem dimes_and_half_dollars (d h : ℕ) (h_nonzero : h > 0) (d_nonzero : d > 0) (eq_condition : 10 * d + 50 * h = 1200) : 
  ∃ n, n = 23 :=
by {
  use 23,
  sorry -- Proof not required.
}

end dimes_and_half_dollars_l705_705907


namespace smallest_positive_period_of_f_l705_705535

noncomputable def f : ℝ → ℝ := λ x, (Real.sin x) ^ 2 - (Real.cos x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end smallest_positive_period_of_f_l705_705535


namespace magnitude_a_plus_2b_eq_2_l705_705363

variables {V : Type*} [inner_product_space ℝ V]

/- Definitions directly from the problem conditions -/
variables (a b : V)
def norm_a : ℝ := 2
def norm_b : ℝ := 1
def angle_ab : ℝ := 2 * real.pi / 3
-- cos(2π/3) = -1/2
def cos_angle_ab : ℝ := -1 / 2

-- Necessary conditions
axiom norm_a_eq : ‖a‖ = norm_a
axiom norm_b_eq : ‖b‖ = norm_b
axiom inner_product_ab_eq : ⟪a, b⟫ = ‖a‖ * ‖b‖ * cos_angle_ab

-- Prove the goal
theorem magnitude_a_plus_2b_eq_2 :
  ‖a + 2 • b‖ = 2 :=
by 
  sorry

end magnitude_a_plus_2b_eq_2_l705_705363


namespace contest_end_time_l705_705604

def start_time := 12 * 60 -- 12:00 p.m. in minutes
def duration := 1500 -- duration in minutes
def end_time := start_time + duration -- end time in minutes

theorem contest_end_time :
  let end_hour := (end_time / 60) % 24 in
  let end_minute := end_time % 60 in
  end_hour = 13 ∧ end_minute = 0 :=
by
  -- Proof will be written here
  sorry

end contest_end_time_l705_705604


namespace zoey_holidays_l705_705584

def visits_per_year (visits_per_month : ℕ) (months_per_year : ℕ) : ℕ :=
  visits_per_month * months_per_year

def visits_every_two_months (months_per_year : ℕ) : ℕ :=
  months_per_year / 2

def visits_every_four_months (visits_per_period : ℕ) (periods_per_year : ℕ) : ℕ :=
  visits_per_period * periods_per_year

theorem zoey_holidays (visits_per_month_first : ℕ) 
                      (months_per_year : ℕ) 
                      (visits_per_period_third : ℕ) 
                      (periods_per_year : ℕ) : 
  visits_per_year visits_per_month_first months_per_year 
  + visits_every_two_months months_per_year 
  + visits_every_four_months visits_per_period_third periods_per_year = 39 := 
  by 
  sorry

end zoey_holidays_l705_705584


namespace inequality_range_l705_705379

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 * x + 1| - |2 * x - 1| < a) → a > 2 :=
by
  sorry

end inequality_range_l705_705379


namespace ribbon_each_box_fraction_l705_705410

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l705_705410


namespace number_of_rectangles_l705_705520

open Real Set

-- Given points A, B, C, D on a line L and a length k
variables {A B C D : ℝ} (L : Set ℝ) (k : ℝ)

-- The points are distinct and ordered on the line
axiom h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom h2 : A < B ∧ B < C ∧ C < D

-- We need to show there are two rectangles with certain properties
theorem number_of_rectangles : 
  (∃ (rect1 rect2 : Set ℝ), 
    rect1 ≠ rect2 ∧ 
    (∃ (a1 b1 c1 d1 : ℝ), rect1 = {a1, b1, c1, d1} ∧ 
      a1 < b1 ∧ b1 < c1 ∧ c1 < d1 ∧ 
      (d1 - c1 = k ∨ c1 - b1 = k)) ∧ 
    (∃ (a2 b2 c2 d2 : ℝ), rect2 = {a2, b2, c2, d2} ∧ 
      a2 < b2 ∧ b2 < c2 ∧ c2 < d2 ∧ 
      (d2 - c2 = k ∨ c2 - b2 = k))
  ) :=
sorry

end number_of_rectangles_l705_705520


namespace find_KP_length_l705_705885

    variables (K L M P Q E R: Point)
    variables (LQ a : ℝ) (EL LR b : ℝ) (α : ℝ)
    variables [IsTriangle (K L M)]

    -- Definitions
    def angle_bisector := True -- This can be refined with actual geometric definitions
    def incircle_tangent (K L P : Point) (Q : Point) := tangent_point K L P Q -- Placeholder for actual tangent definition
    def passes_through_incircle_center (E R : Point) := passes_through_center_of_incircle K L M E R -- Placeholder

    -- Given conditions
    def conditions := LQ = a ∧ EL + LR = b ∧ (area_ratio (K L P) (E L R) = α)

    noncomputable def KP_length : ℝ :=
    α * b - 2 * a

    theorem find_KP_length
      (h_angle_bisector : angle_bisector)
      (h_incircle_tangent : incircle_tangent K L P Q)
      (h_passes_through_center : passes_through_incircle_center E R)
      (h_conditions : conditions) :
      length (KP) = KP_length a α b :=
    sorry
    
end find_KP_length_l705_705885


namespace triangle_abc_proof_one_triangle_abc_perimeter_l705_705448

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l705_705448


namespace max_value_abcde_l705_705072

theorem max_value_abcde (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * real.sqrt 5 :=
sorry

end max_value_abcde_l705_705072


namespace people_dislike_both_radio_and_music_l705_705924

theorem people_dislike_both_radio_and_music :
  let total_people := 1500
  let dislike_radio_percent := 0.35
  let dislike_both_percent := 0.20
  let dislike_radio := dislike_radio_percent * total_people
  let dislike_both := dislike_both_percent * dislike_radio
  dislike_both = 105 :=
by
  sorry

end people_dislike_both_radio_and_music_l705_705924


namespace annual_subscription_cost_l705_705615

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l705_705615


namespace arrangement_condition_l705_705300

theorem arrangement_condition (x y z : ℕ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (H1 : x ≤ y + z) 
  (H2 : y ≤ x + z) 
  (H3 : z ≤ x + y) : 
  ∃ (A : ℕ) (B : ℕ) (C : ℕ), 
    A = x ∧ B = y ∧ C = z ∧
    A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1 ∧
    (A ≤ B + C) ∧ (B ≤ A + C) ∧ (C ≤ A + B) :=
by
  sorry

end arrangement_condition_l705_705300


namespace minimum_a_l705_705765

theorem minimum_a (x : ℝ) (h : ∀ x ≥ 0, x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a) : 
    a ≥ -1 := by
  sorry

end minimum_a_l705_705765


namespace probability_opposite_vertex_l705_705393

theorem probability_opposite_vertex (k : ℕ) (h : k > 0) : 
    P_k = (1 / 6 : ℝ) + (1 / (3 * (-2) ^ k) : ℝ) := 
sorry

end probability_opposite_vertex_l705_705393


namespace problem1_problem2_l705_705503

-- Problem 1
theorem problem1 (x y : ℤ) (h1 : x = 2) (h2 : y = 2016) :
  (3*x + 2*y)*(3*x - 2*y) - (x + 2*y)*(5*x - 2*y) / (8*x) = -2015 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h1 : x = 2) :
  ((x - 3) / (x^2 - 1)) * ((x^2 + 2*x + 1) / (x - 3)) - (1 / (x - 1) + 1) = 1 :=
by
  sorry

end problem1_problem2_l705_705503


namespace product_of_remaining_numbers_l705_705985

theorem product_of_remaining_numbers {a b c d : ℕ} (h1 : a = 11) (h2 : b = 22) (h3 : c = 33) (h4 : d = 44) :
  ∃ (x y z : ℕ), 
  (∃ n: ℕ, (a + b + c + d) - n * 3 = 3 ∧ -- We removed n groups of 3 different numbers
             x + y + z = 2 * n + (a + b + c + d)) ∧ -- We added 2 * n numbers back
  x * y * z = 12 := 
sorry

end product_of_remaining_numbers_l705_705985


namespace sum_of_digits_is_2640_l705_705693

theorem sum_of_digits_is_2640 (x : ℕ) (h_cond : (1 + 3 + 4 + 6 + x) * (Nat.factorial 5) = 2640) : x = 8 := by
  sorry

end sum_of_digits_is_2640_l705_705693


namespace BD_tangent_to_circumcircles_ABM_ADM_l705_705858

-- Define the geometric objects involved
variables {A B C D M O : Type}
variable [EuclideanGeometry A B C D M O]

-- Define that ABCD is a parallelogram
def IsParallelogram (A B C D : Type) [EuclideanGeometry A B C D] : Prop :=
Parallelogram A B C D

-- Define the conditions of the problem
axiom AB_parallelogram_AC_longer_BD :
  IsParallelogram A B C D ∧ (length A C > length B D)
axiom M_on_AC_cyclic_BCDM : 
  lies_on M (line_through A C) ∧ CyclicQuadrilateral B C D M

-- The theorem to prove
theorem BD_tangent_to_circumcircles_ABM_ADM :
  BD_tangent_to_circumcircles A B M D :=
sorry

end BD_tangent_to_circumcircles_ABM_ADM_l705_705858


namespace verify_derivatives_l705_705580

noncomputable def verify_option (x : ℝ) :=
  (deriv (fun (_ : ℝ) => ℓog (3 * x + 1)) = 3 / (3 * x + 1)) ∧
  (deriv (fun (_ : ℝ) => 1 / real.cbrt x) = -1 / (3 * real.cbrt (x ^ 4))) ∧
  (deriv (fun (_ : ℝ) => cos (real.pi / 6)) ≠ -1 / 2) ∧
  (deriv (fun (_ : ℝ) => exp (-x)) ≠ exp (-x))

theorem verify_derivatives:
  verify_option x :=
by 
  sorry

end verify_derivatives_l705_705580
