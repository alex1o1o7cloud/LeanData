import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Complex.ReIm
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinatorics
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Game.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace cost_of_purchase_l287_287760

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l287_287760


namespace max_value_at_one_l287_287320

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287320


namespace tangent_intersection_point_l287_287892

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287892


namespace incenter_inside_triangle_BOH_l287_287799

variables {A B C : ℝ} (hC_gt_B : C > B) (hB_gt_A : B > A)

noncomputable def incenter (A B C : ℝ) := sorry
noncomputable def circumcenter (A B C : ℝ) := sorry
noncomputable def orthocenter (A B C : ℝ) := sorry

theorem incenter_inside_triangle_BOH (hC_gt_B : C > B) (hB_gt_A : B > A)
    (I := incenter A B C) (O := circumcenter A B C) (H := orthocenter A B C) :
    ∃ (a b c : ℝ), I ∈ triangle (line B O) (line O H) (line H B) :=
sorry

end incenter_inside_triangle_BOH_l287_287799


namespace opposite_of_neg_five_l287_287795

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l287_287795


namespace erased_number_l287_287478

theorem erased_number (a : ℤ) (b : ℤ) (h : -4 ≤ b ∧ b ≤ 4) (h_sum : 8 * a - b = 1703) : a + b = 214 := 
by 
  have ha : 1699 / 8 ≤ a, from sorry
  have hb : a ≤ 1707 / 8, from sorry
  have ha_int : a = 213, from sorry
  have hb_int : b = 1, by { rw [ha_int] at h_sum, linarith }
  exact sorry 

-- The proof steps are not provided here, as only the theorem statement is requested.

end erased_number_l287_287478


namespace erased_number_l287_287470

theorem erased_number (a b : ℤ) (h1 : ∀ n : ℤ, n ∈ set.range (λ i, a + i) ↔ n ∈ set.range (λ i, a - 4 + i)) 
                      (h2 : 8 * a - b = 1703)
                      (h3 : -4 ≤ b ∧ b ≤ 4) : a + b = 214 := 
by
    sorry

end erased_number_l287_287470


namespace length_major_axis_l287_287946

-- Definitions of points and conditions
def F1 : ℝ × ℝ := (15, 10)
def F2 : ℝ × ℝ := (35, 40)
def reflected_F1 : ℝ × ℝ := (-15, 10)

-- Distance calculation function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove the length of the major axis
theorem length_major_axis :
  distance reflected_F1 F2 = 10 * Real.sqrt 34 :=
by
  sorry

end length_major_axis_l287_287946


namespace find_constant_l287_287388

theorem find_constant (f : ℝ → ℝ) (c : ℝ)
  (h1 : ∀ x : ℝ, f(x) + c * f(8 - x) = x)
  (h2 : f(2) = 2) :
  c = 3 :=
by
  sorry

end find_constant_l287_287388


namespace number_of_DVDs_sold_l287_287657

theorem number_of_DVDs_sold (C D: ℤ) (h₁ : D = 16 * C / 10) (h₂ : D + C = 273) : D = 168 := 
sorry

end number_of_DVDs_sold_l287_287657


namespace expansion_contains_one_odd_term_l287_287152

theorem expansion_contains_one_odd_term (m n : ℤ) (h1 : m % 2 = 1) (h2 : n % 2 = 0) : 
  let expansion := (m + n)^8 in
  ∃ t : ℤ, (∃ i : ℕ, 0 ≤ i ∧ i ≤ 8 ∧ t = nat.choose 8 i * m^(8-i) * n^i) ∧ (t % 2 = 1) ∧
  ∀ t' : ℤ, (∃ i : ℕ, 0 ≤ i ∧ i ≤ 8 ∧ t' = nat.choose 8 i * m^(8-i) * n^i ∧ (t' ≠ t)) → t' % 2 = 0 := sorry

end expansion_contains_one_odd_term_l287_287152


namespace tangent_circles_x_intersect_l287_287874

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287874


namespace apples_not_ripe_l287_287983

theorem apples_not_ripe (total_apples good_apples : ℕ) (h1 : total_apples = 14) (h2 : good_apples = 8) : total_apples - good_apples = 6 :=
by {
  sorry
}

end apples_not_ripe_l287_287983


namespace monotonic_increasing_interval_l287_287270

noncomputable def function_y (x : ℝ) : ℝ := (3 - x^2) * Real.exp x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, -3 < x ∧ x < 1 → deriv (function_y x) > 0 :=
by
  intro x
  intro h
  have deriv_y : deriv (function_y x) = (-x^2 - 2 * x + 3) * Real.exp x := sorry
  have inequality : (-x^2 - 2 * x + 3) > 0 := sorry
  exact sorry

end monotonic_increasing_interval_l287_287270


namespace ratio_of_shaded_to_unshaded_l287_287460

-- Definitions based on conditions
def is_regular_hexagon (hex : Hexagram) : Prop :=
  -- Hexagram vertices are the same as that of a regular hexagon
  hex.is_regular_hexagon

def forms_hexagram (hex : Hexagram) : Prop :=
  -- Hexagram is formed by overlapping two equilateral triangles
  hex.forms_hexagram_by_overlapping_equilateral_triangles

def shaded_region_triangles (hex : Hexagram) : ℕ :=
  -- Shaded region forms 18 smaller triangles
  hex.shaded_region_triangles

def unshaded_region_triangles (hex : Hexagram) : ℕ :=
  -- Unshaded region forms 6 smaller triangles
  hex.unshaded_region_triangles

-- Main statement
theorem ratio_of_shaded_to_unshaded (hex : Hexagram)
  (h1 : is_regular_hexagon hex) 
  (h2 : forms_hexagram hex)
  (h3 : shaded_region_triangles hex = 18) 
  (h4 : unshaded_region_triangles hex = 6) :
  shaded_region_triangles hex / unshaded_region_triangles hex = 3 :=
by
  -- Proof will go here
  sorry

end ratio_of_shaded_to_unshaded_l287_287460


namespace find_k_l287_287921

variables {a b : Type} [AddGroup a] [AddGroup b]

def line_through (a b : a) (t : b) : a := a + t * (b - a)

theorem find_k (a b : a) : ∃ k : b, k • a + (2/3 : b) • b = (1/3 : b) • a + (2/3 : b) • b :=
by
  use (1/3 : b)
  simp
  sorry

end find_k_l287_287921


namespace smallest_non_palindrome_product_l287_287549

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287549


namespace factor_x12_minus_1024_l287_287035

-- The mathematical proof problem stating that x^12 - 1024 can be completely factored as (x^6 + 32)(x^3 + 4√2)(x^3 - 4√2)
theorem factor_x12_minus_1024 : 
  ∀ x : ℝ, x^12 - 1024 = (x^6 + 32) * (x^3 + 4 * real.sqrt 2) * (x^3 - 4 * real.sqrt 2) :=
by
  sorry

end factor_x12_minus_1024_l287_287035


namespace sum_of_roots_of_quadratic_l287_287303

theorem sum_of_roots_of_quadratic :
  let a := 2
  let b := -8
  let c := 6
  let sum_of_roots := (-b / a)
  2 * (sum_of_roots) * sum_of_roots - 8 * sum_of_roots + 6 = 0 :=
by
  sorry

end sum_of_roots_of_quadratic_l287_287303


namespace pie_contest_total_pies_l287_287182

theorem pie_contest_total_pies 
  (bill_adam : ℕ -> ℕ) (bill_sierra : ℕ -> ℕ) (average_taylor : ℕ -> ℕ -> ℕ -> ℕ) :
  (∀ b, bill_sierra b = 2 * b) ->                        -- Sierra eats twice as many pies as Bill
  (∀ b, bill_adam b = b + 3) ->                          -- Adam eats three more pies than Bill
  (∀ a b s, average_taylor a b s = (a + b + s) / 3) ->   -- Taylor eats the average number of pies eaten by Adam, Bill, and Sierra
  (∀ B S b, S = 12 -> bill_sierra B = S -> bill_adam B = B + 3
          -> average_taylor (B + 3) B S = (B + 3 + B + S) / 3
          -> let t := (B + 3 + B + S) / 3 in B + 3 + B + S + t <= 50
          -> B + 3 + B + S + t = 36)
:= by {
   intro h1 h2 h3 h4 B S b H_S_eq h5 h6 h7;
   sorry
}

end pie_contest_total_pies_l287_287182


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287551

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287551


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287558

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287558


namespace tangent_line_value_of_a_l287_287605

noncomputable def is_tangent (a : ℝ) : Prop :=
  let d := |(-2 * a)| / Real.sqrt 5 in
  d = 1

theorem tangent_line_value_of_a (a : ℝ) : is_tangent a ↔ a = Real.sqrt 5 / 2 ∨ a = -Real.sqrt 5 / 2 :=
by
  sorry

end tangent_line_value_of_a_l287_287605


namespace derivative_at_2_l287_287369

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287369


namespace derivative_y_l287_287988

noncomputable def y (x : ℝ) : ℝ :=
  (1 / 4) * Real.log ((x - 1) / (x + 1)) - (1 / 2) * Real.arctan x

theorem derivative_y (x : ℝ) : deriv y x = 1 / (x^4 - 1) :=
  sorry

end derivative_y_l287_287988


namespace number_of_squares_in_figure_100_l287_287058

theorem number_of_squares_in_figure_100 :
  (∃ (f : ℕ → ℤ), 
     f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25 ∧ ∀ n, f n = 2 * n^2 + 2 * n + 1) →
  f 100 = 20201 :=
by
  intro h
  rcases h with ⟨f, f0, f1, f2, f3, hf⟩
  have key : f = λ n, 2 * n^2 + 2 * n + 1 := by
    apply funext
    intro n
    exact hf n
  rw [key]
  norm_num
  exact 20201

end number_of_squares_in_figure_100_l287_287058


namespace validate_financial_position_l287_287226

noncomputable def financial_position_start_of_year := 86588
noncomputable def financial_position_end_of_year := 137236
noncomputable def total_tax := 8919
noncomputable def remaining_funds_after_tour := 38817

variables
(father_income monthly: ℝ := 50000)
(mother_income monthly: ℝ := 28000)
(grandmother_pension monthly: ℝ := 15000)
(mikhail_scholarship monthly: ℝ := 3000)
(father_tax_deduction monthly: ℝ := 2800)
(mother_tax_deduction monthly: ℝ := 2800)
(tax_rate: ℝ := 0.13)
(np_father: ℝ := father_income - father_tax_deduction)
(np_mother: ℝ := mother_income - mother_tax_deduction)

def net_father_tax (monthly:ℝ) := np_father * tax_rate
def net_mother_tax (monthly:ℝ) := np_mother * tax_rate

def father_monthly_income_after_tax (monthly:=ℝ) := father_income - net_father_tax
def mother_monthly_income_after_tax (monthly:=ℝ) := mother_income - net_mother_tax

def net_monthly_income (monthly:ℝ) := father_monthly_income_after_tax + mother_monthly_income_after_tax + grandmother_pension + mikhail_scholarship
def annual_net_income (yearly:=ℝ) := net_monthly_income * 12

variables
(financial_safety_cushion: ℝ := 10000 * 12)
(household_expenses: ℝ := (50000 + 15000) * 12)

def net_disposable_income_per_year (net_yearly:= ℝ) := annual_net_income - financial_safety_cushion - household_expenses

variables
(cadastral_value:ℝ := 6240000)
(sq_m:ℝ := 78)
(sq_m_reduction: ℝ := 20)
(rate_property: ℝ := 0.001)

def property_tax := (cadastral_value - sq_m_reduction * (cadastral_value / sq_m)) * rate_property

variables
(lada_prior_hp: ℝ := 106)
(lada_xray_hp: ℝ := 122)
(car_tax_rate: ℝ := 35)
(months_prior:ℝ := 3/12)
(months_xray: ℝ := 8/12)

def total_transport_tax := lada_prior_hp * car_tax_rate * months_prior + lada_xray_hp * car_tax_rate * months_xray

variables
(cadastral_value_land: ℝ := 420300)
(land_are: ℝ := 10)
(tax_rate_land: ℝ := 0.003)
(deducted_land_area:= 6)
 
def land_tax := (cadastral_value_land - (cadastral_value_land / land_are) * deducted_land_area) * tax_rate_land

def total_tax_liability := property_tax + total_transport_tax + land_tax

def after_tax_liquidity (total_tax_yearly:=ℝ) := net_disposable_income_per_year - total_tax_liability

variables
(tour_cost := 17900)
(participants := 5)

def remaining_after_tour := after_tax_liquidity - tour_cost * participants

theorem validate_financial_position :
  financial_position_start_of_year = 86588 ∧
  financial_position_end_of_year = 137236 ∧
  total_tax = 8919 ∧
  remaining_funds_after_tour = 38817 :=
by
  sorry 

end validate_financial_position_l287_287226


namespace find_f_prime_at_two_l287_287329

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287329


namespace inequalities_correct_l287_287088

open Real

-- Define the variables and conditions
variables (a b : ℝ) (h_pos : a * b > 0)

-- Define the inequalities
def inequality_B : Prop := 2 * (a^2 + b^2) >= (a + b)^2
def inequality_C : Prop := (b / a) + (a / b) >= 2
def inequality_D : Prop := (a + 1 / a) * (b + 1 / b) >= 4

-- The Lean statement
theorem inequalities_correct : inequality_B a b h_pos ∧ inequality_C a b h_pos ∧ inequality_D a b h_pos :=
by
  sorry

end inequalities_correct_l287_287088


namespace octal_sum_l287_287994

open Nat

def octal_to_decimal (oct : ℕ) : ℕ :=
  match oct with
  | 0 => 0
  | n => let d3 := (n / 100) % 10
         let d2 := (n / 10) % 10
         let d1 := n % 10
         d3 * 8^2 + d2 * 8^1 + d1 * 8^0

def decimal_to_octal (dec : ℕ) : ℕ :=
  let rec aux (n : ℕ) (mul : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 8) (mul * 10) (acc + (n % 8) * mul)
  aux dec 1 0

theorem octal_sum :
  let a := 451
  let b := 167
  octal_to_decimal 451 + octal_to_decimal 167 = octal_to_decimal 640 := sorry

end octal_sum_l287_287994


namespace tangent_circles_l287_287458

theorem tangent_circles 
  {A B C H E F X Y T : Type*}
  [is_orthocenter H A B C]
  (hE : ∥B - E∥ = ∥B - H∥)
  (hF : ∥C - F∥ = ∥C - H∥)
  (hX : line_through E H ∩ line_through B C = X)
  (hY : line_through F H ∩ line_through B C = Y)
  (hT : perpendicular (line_through H T) (line_through E F))
  : tangent (circumcircle (triangle T X Y)) (circle_diameter B C) :=
sorry

end tangent_circles_l287_287458


namespace tangent_line_intersection_x_axis_l287_287896

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287896


namespace area_ABQCDP_is_112_5_l287_287194

-- Define the conditions of the problem using Lean notation
noncomputable def A := (0 : ℝ, 0 : ℝ)
noncomputable def B := (13 : ℝ, 0 : ℝ)
noncomputable def C := (𝑥_C : ℝ, 𝑦_C : ℝ)
noncomputable def D := (𝑥_D : ℝ, 𝑦_D : ℝ)

def P := ((0 + 𝑥_C) / 2, (0 + 𝑦_C) / 2)
def Q := ((13 + 𝑥_D) / 2, (0 + 𝑦_D) / 2)

-- Define the areas based on the given conditions
axiom area_ABCD : ℝ := 150

-- Prove that the area of the new quadrilateral ABQCDP is 112.5 square units
theorem area_ABQCDP_is_112_5 : 
  (area_of_quad A B Q C D P) = 112.5 := 
sorry

end area_ABQCDP_is_112_5_l287_287194


namespace irrational_number_among_options_l287_287380

theorem irrational_number_among_options :
  ¬ (is_rational (sqrt 2)) ∧ is_rational (1 / 3) ∧ is_rational (0.8) ∧ is_rational (-6) :=
by
  -- Conditions according to the problem statement
  have h1 : is_rational (1 / 3) := sorry,
  have h2 : is_rational (0.8) := sorry,
  have h3 : is_rational (-6) := sorry,
  have h4 : ¬ (is_rational (sqrt 2)) := sorry,
  exact ⟨h4, h1, h2, h3⟩

end irrational_number_among_options_l287_287380


namespace derivative_at_2_l287_287362

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287362


namespace erased_number_is_214_l287_287471

theorem erased_number_is_214 {a b : ℤ} 
  (h1 : 9 * a = sum (a - 4 :: a - 3 :: a - 2 :: a - 1 :: a :: a + 1 :: a + 2 :: a + 3 :: a + 4 :: []))
  (h2 : ∑ n in {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4} \ erase_val (a + b) {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4}, n = 1703)
  (h3 : -4 ≤ b ∧ b ≤ 4)
  : a + b = 214 :=
begin
  -- proof to be filled in
  sorry,
end

end erased_number_is_214_l287_287471


namespace find_angle_APB_l287_287175

noncomputable def vector (α : Type*) := α → α
variable {V : Type*} [inner_product_space ℝ V]

def triangle_ABC (A B C : V) : Prop :=
  let angle_C := 90
  is_right_triangle A B C ∧ is_isosceles_right_triangle A B C ∧
  ∃ P, 
    (P ∈ triangle_interior A B C) ∧ 
    (vector PA + sqrt 2 * vector PB + (2 * sqrt 2 + 2) * vector PC = 0)

theorem find_angle_APB (A B C P : V) :
  triangle_ABC A B C → 
  ∀ P, 
    (P ∈ triangle_interior A B C) → 
    (vector PA + sqrt 2 * vector PB + (2 * sqrt 2 + 2) * vector PC = 0) →
    angle_APB A P B = 5 * π / 8 := 
  by
    sorry

end find_angle_APB_l287_287175


namespace cost_price_of_article_l287_287945

noncomputable def cost_price (M : ℝ) : ℝ := 98.68 / 1.25

theorem cost_price_of_article (M : ℝ)
    (h1 : 0.95 * M = 98.68)
    (h2 : 98.68 = 1.25 * cost_price M) :
    cost_price M = 78.944 :=
by sorry

end cost_price_of_article_l287_287945


namespace tan_A_correct_l287_287690

-- Define basic properties of the triangle
structure Triangle :=
  (A B C : Type)
  (right_angle_at_B : B)
  (AC : ℝ)
  (AB : ℝ)
  (BC : ℝ)

-- Ensure given conditions
def right_triangle (T : Triangle) : Prop :=
  T.right_angle_at_B ∧ T.AC = 5 ∧ T.AB = 4 ∧ T.BC = Real.sqrt (T.AC^2 - T.AB^2)

-- Define the tangent function for angle A
def tan_A (T : Triangle) : ℝ :=
  T.BC / T.AB

-- The theorem we want to prove
theorem tan_A_correct (T : Triangle) (h : right_triangle T) :
  tan_A T = 3 / 4 :=
  sorry

end tan_A_correct_l287_287690


namespace measure_of_angle_4_l287_287024

theorem measure_of_angle_4 
  (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) : 
  angle4 = 110 :=
by
  sorry

end measure_of_angle_4_l287_287024


namespace smallest_number_of_club_members_l287_287836

theorem smallest_number_of_club_members :
  ∃ n, n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4 ∧ ∀ m, (m % 6 = 2 ∧ m % 8 = 3 ∧ m % 9 = 4) → m ≥ n :=
begin
  use 404,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    sorry,
  }
end

end smallest_number_of_club_members_l287_287836


namespace derivative_at_2_l287_287357

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287357


namespace length_DE_in_triangle_l287_287684

theorem length_DE_in_triangle
  (A B C D E : Type)
  (dist_BC : ℝ)
  (angle_C : ℝ)
  (midpoint_D : D = midpoint B C)
  (perpendicular_bisector : is_perpendicular_bisector D B C E)
  (BC_eq : BC = 30)
  (angle_C_eq : angle_C = 45)
  (legs_equal : ∀ {x y z}, angle_CA B C y z = 45 → dist y z = dist y x)
  : dist D E = 15 := 
by
  sorry

end length_DE_in_triangle_l287_287684


namespace max_value_at_one_l287_287319

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287319


namespace derivative_at_2_l287_287364

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287364


namespace length_of_segment_l287_287255

theorem length_of_segment (a : ℝ) (M : ℝ) (hAM : 3 * (AD : ℝ) = 4 * M) (hSegment : ((edges L: set ℝ) = {AD, BC, CD, DA}) ∧ ( ∃ K : ℝ, H= K - ((lateral edges: set ℝ) = 2a)) ) : 
(segment_in_plane : ℝ) = (personal_segment: ℝ) := 
by 
sorry

end length_of_segment_l287_287255


namespace complement_of_intersection_l287_287138

open Set

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}
def S : Set ℝ := univ -- S is the set of all real numbers

theorem complement_of_intersection :
  S \ (A ∩ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 3 < x } :=
by
  sorry

end complement_of_intersection_l287_287138


namespace tangent_point_value_l287_287877

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287877


namespace sum_of_two_primes_l287_287144

theorem sum_of_two_primes (k : ℕ) (n : ℕ) (h : n = 1 + 10 * k) :
  (n = 1 ∨ ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ n = p1 + p2) :=
by
  sorry

end sum_of_two_primes_l287_287144


namespace line_intersection_l287_287135

-- Definitions of the circles in polar coordinates
def circle1_polar (ρ θ : ℝ) := ρ = 2

def circle2_polar (ρ θ : ℝ) := ρ^2 - 2 * sqrt 2 * ρ * cos (θ - π / 4) = 2

-- Cartesian equations derived from the polar equations
def circle1_cartesian (x y : ℝ) := x^2 + y^2 = 4

def circle2_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x - 2 * y - 2 = 0

-- Conversion of line equation from Cartesian to Polar
def line_cartesian (x y : ℝ) := x + y = 1

def line_polar (ρ θ : ℝ) := ρ * sin (θ + π / 4) = sqrt 2 / 2

-- Proof statement
theorem line_intersection (ρ θ : ℝ) : 
  (∃ (x y : ℝ), circle1_cartesian x y ∧ circle2_cartesian x y) → 
  line_polar ρ θ :=
  sorry

end line_intersection_l287_287135


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287905

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287905


namespace find_abc_sum_l287_287767

def polynomials_gcd_lcm (a b c : ℤ) : Prop :=
  Polynomial.gcd (Polynomial.X^2 + (a : ℤ) * Polynomial.X + b)
                 (Polynomial.X^2 + (b : ℤ) * Polynomial.X + c) = (Polynomial.X + 1)^2 ∧
  Polynomial.lcm (Polynomial.X^2 + (a : ℤ) * Polynomial.X + b)
                 (Polynomial.X^2 + (b : ℤ) * Polynomial.X + c) = Polynomial.X^4 - 5 * Polynomial.X^3 + 9 * Polynomial.X^2 - 5 * Polynomial.X + 6

theorem find_abc_sum : ∃ a b c : ℤ, polynomials_gcd_lcm a b c ∧ a + b + c = -11 :=
by
  sorry

end find_abc_sum_l287_287767


namespace sum_first_six_terms_geometric_seq_l287_287997

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l287_287997


namespace percentage_milk_replaced_each_time_l287_287572

-- Define the initial percentage of milk.
def initial_percentage_milk : ℝ := 100.0

-- Define the percentage of milk remaining after three replacements.
def final_percentage_milk : ℝ := 51.2

-- Define the process of replacement for three operations.
def replacement_process (x : ℝ) : ℝ := ((initial_percentage_milk - x) / initial_percentage_milk) ^ 3 * initial_percentage_milk

-- Theorem: Prove the percentage of milk replaced by water each time.
theorem percentage_milk_replaced_each_time (x : ℝ) : replacement_process x = final_percentage_milk → x = 20 :=
by
  sorry

end percentage_milk_replaced_each_time_l287_287572


namespace negation_of_exists_l287_287791

theorem negation_of_exists (h : ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) : ∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0 :=
sorry

end negation_of_exists_l287_287791


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287534

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287534


namespace find_f_prime_at_2_l287_287337

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287337


namespace kids_played_on_monday_l287_287699

theorem kids_played_on_monday (total : ℕ) (tuesday : ℕ) (monday : ℕ) (h_total : total = 16) (h_tuesday : tuesday = 14) :
  monday = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end kids_played_on_monday_l287_287699


namespace min_trips_correct_l287_287395

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l287_287395


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287553

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287553


namespace sum_of_areas_of_circles_l287_287186

variable (a b c : ℝ)

noncomputable def p : ℝ := (a + b + c) / 2
noncomputable def area_sum : ℝ := (π * (p - a) * (p - b) * (p - c) * (a^2 + b^2 + c^2)) / (p^3)

theorem sum_of_areas_of_circles (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : p > a) (hb : p > b) (hc : p > c) :
  area_sum a b c = π * (p - a) * (p - b) * (p - c) * (a^2 + b^2 + c^2) / (p^3) :=
by
  sorry

end sum_of_areas_of_circles_l287_287186


namespace gcd_of_items_l287_287268

def numPens : ℕ := 891
def numPencils : ℕ := 810
def numNotebooks : ℕ := 1080
def numErasers : ℕ := 972

theorem gcd_of_items :
  Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numNotebooks) numErasers = 27 :=
by
  sorry

end gcd_of_items_l287_287268


namespace three_kids_savings_l287_287250

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end three_kids_savings_l287_287250


namespace square_cells_count_l287_287665

-- Define the conditions mentioned in the problem.
def large_square_cells (n : ℕ) : ℕ := n ^ 2
def small_square_cells (m : ℕ) : ℕ := m ^ 2
def remaining_cells (n m : ℕ) : ℕ := large_square_cells n - small_square_cells m

-- State the main problem.
theorem square_cells_count (n m : ℕ) (h1 : large_square_cells n > small_square_cells m)
  (h2 : remaining_cells n m = 209) : large_square_cells n = 225 :=
begin
  -- This will contain the proof, which is not required for now.
  sorry
end

end square_cells_count_l287_287665


namespace sum_of_consecutive_odds_eq_169_l287_287851

theorem sum_of_consecutive_odds_eq_169 (n : ℕ) (h_odd : n % 2 = 1) :
  (∑ k in finset.range ((n + 1) / 2), 2 * k + 1) = 169 → n = 25 :=
by
  sorry

end sum_of_consecutive_odds_eq_169_l287_287851


namespace petya_cannot_achieve_goal_l287_287442

theorem petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ ∀ (glasses : Fin (2 * n) → bool),
      (∃ i : Fin (2 * n),
        glasses i = glasses (i + 1) ∧ glasses (i + 1) = glasses (i + 2))
:= sorry

end petya_cannot_achieve_goal_l287_287442


namespace snow_probability_l287_287271

noncomputable def probability_at_least_three_snow_days_in_five (p : ℚ) (n : ℕ): ℚ :=
  ∑ k in (finset.range (n+1)).filter (λ k, k ≥ 3), 
  nat.choose n k * p^k * (1 - p)^(n - k)

theorem snow_probability (p : ℚ) (n : ℕ) : 
  p = 1/2 → n = 5 → 
  probability_at_least_three_snow_days_in_five p n = 1/2 :=
sorry

end snow_probability_l287_287271


namespace eval_expression_eq_neg431_l287_287509

theorem eval_expression_eq_neg431 (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^b - (b^a)^a) = -431 := by
  sorry

end eval_expression_eq_neg431_l287_287509


namespace sum_of_possible_A_plus_B_l287_287003

theorem sum_of_possible_A_plus_B (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9)
  (divisible_by_3 : (16 + A + B) % 3 = 0) :
  ∑ (k : ℕ) in { k | ∃ A B : ℕ, 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ (16 + A + B) % 3 = 0 ∧ k = A + B }, k = 57 := by
  sorry

end sum_of_possible_A_plus_B_l287_287003


namespace cylinder_surface_area_l287_287928

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area (h r : ℕ) (h_eq : h = 8) (r_eq : r = 3) :
  2 * Real.pi * r * h + 2 * Real.pi * r ^ 2 = 66 * Real.pi := by
  sorry

end cylinder_surface_area_l287_287928


namespace pow_eq_one_of_imaginary_eq_l287_287148

theorem pow_eq_one_of_imaginary_eq (x y : ℝ) (h : (x + y) * complex.I = x - 1) : 2^(x + y) = 1 := by
  sorry

end pow_eq_one_of_imaginary_eq_l287_287148


namespace average_price_of_goat_l287_287408

theorem average_price_of_goat (total_cost : ℕ) (number_of_goats : ℕ) (number_of_hens : ℕ) (average_price_of_hen : ℕ) :
  total_cost = 2500 → number_of_goats = 5 → number_of_hens = 10 → average_price_of_hen = 50 → 
  (total_cost - (number_of_hens * average_price_of_hen)) / number_of_goats = 400 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp only [Nat.sub_add_eq_add_sub, add_assoc, add_comm, add_left_comm, Nat.mul, Nat.div, Nat.succ]
  sorry

end average_price_of_goat_l287_287408


namespace hanging_spheres_ratio_l287_287073

theorem hanging_spheres_ratio (m1 m2 g T_B T_H : ℝ)
  (h1 : T_B = 3 * T_H)
  (h2 : T_H = m2 * g)
  (h3 : T_B = m1 * g + T_H)
  : m1 / m2 = 2 :=
by
  sorry

end hanging_spheres_ratio_l287_287073


namespace evaluate_xx_at_x_eq_3_l287_287507

theorem evaluate_xx_at_x_eq_3 :
  (3^3)^(Real.sqrt (3^3)) = 27^(3 * Real.sqrt 3) := 
by
  sorry

end evaluate_xx_at_x_eq_3_l287_287507


namespace selling_price_ratio_l287_287938

-- Define the conditions
def cost_price : ℝ := 100.0  -- Assume the cost price is 100
def profit_percentage_1 : ℝ := 0.40  -- 40% profit
def profit_percentage_2 : ℝ := 1.80  -- 180% profit

-- Define the selling prices based on the given conditions
def selling_price_1 (CP : ℝ) : ℝ := CP + profit_percentage_1 * CP
def selling_price_2 (CP : ℝ) : ℝ := CP + profit_percentage_2 * CP

-- Prove the ratio of the new selling price to the previous selling price is 2:1
theorem selling_price_ratio : 
  let SP1 := selling_price_1 cost_price in
  let SP2 := selling_price_2 cost_price in
  SP2 / SP1 = 2 :=
by
  sorry

end selling_price_ratio_l287_287938


namespace intersection_complement_eq_l287_287139

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 })
variable (B : Set ℝ := { x | x > -1 })

theorem intersection_complement_eq :
  A ∩ (U \ B) = { x | -2 ≤ x ∧ x ≤ -1 } :=
by {
  sorry
}

end intersection_complement_eq_l287_287139


namespace smallest_number_l287_287943

theorem smallest_number (x1 x2 x3 x4 : ℝ) 
  (h1 : x1 = -real.pi)
  (h2 : x2 = -3)
  (h3 : x3 = -real.sqrt 2)
  (h4 : x4 = -5 / 2) : 
  x1 < x2 ∧ x1 < x3 ∧ x1 < x4 :=
by
  sorry

end smallest_number_l287_287943


namespace bees_15feet_apart_move_north_south_l287_287290

def Bee := ℕ → ℝ × ℝ × ℝ

def initial_position : ℝ × ℝ × ℝ := (0, 0, 0)

def beeA_position : Bee := λ n =>
  let (x, y, z) := initial_position
  let k := n / 3
  let r := n % 3
  match r with
  | 0 => (x + k + 1, y + k, z + k)
  | 1 => (x + k + 1, y + k + 1, z + k)
  | 2 => (x + k + 1, y + k + 1, z + k + 1)
  | _ => (x, y, z)

def beeB_position : Bee := λ n =>
  let (x, y, z) := initial_position
  let k := n / 3
  let r := n % 3
  match r with
  | 0 => (x - k - 1, y - k, z + k)
  | 1 => (x - k - 1, y - k - 1, z + k)
  | 2 => (x - k - 1, y - k - 1, z + k + 1)
  | _ => (x, y, z)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def direction (n : ℕ) : string :=
  if n % 3 = 0 then "north, south"
  else if n % 3 = 1 then "east, west"
  else "up, up"

theorem bees_15feet_apart_move_north_south :
  ∃ n, distance (beeA_position n) (beeB_position n) = 15 ∧
    direction n = "north, south" :=
by
  sorry

end bees_15feet_apart_move_north_south_l287_287290


namespace students_juice_count_l287_287464

theorem students_juice_count (students chose_water chose_juice : ℕ) 
  (h1 : chose_water = 140) 
  (h2 : (25 : ℚ) / 100 * (students : ℚ) = chose_juice)
  (h3 : (70 : ℚ) / 100 * (students : ℚ) = chose_water) : 
  chose_juice = 50 :=
by 
  sorry

end students_juice_count_l287_287464


namespace ratio_of_areas_of_squares_l287_287177

-- Given square ABCD with side length 10 units and point E on AB such that AE = 3 * EB
def square_ABCD_side : ℝ := 10
def point_E_ratio : ℝ := 3 / 4
theorem ratio_of_areas_of_squares :
  let ABCD_area := (square_ABCD_side * square_ABCD_side)
      EFGH_side := square_ABCD_side * (1 - 2 * (1 / (1 + 3)))
      EFGH_area := EFGH_side * EFGH_side in
  (EFGH_area / ABCD_area) = (1 / 4) :=
by
  sorry

end ratio_of_areas_of_squares_l287_287177


namespace solve_problem_l287_287717

noncomputable def find_z_values (x : ℝ) : ℝ :=
  (x - 3)^2 * (x + 4) / (2 * x - 4)

theorem solve_problem (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  find_z_values x = 64.8 ∨ find_z_values x = -10.125 :=
by
  sorry

end solve_problem_l287_287717


namespace paul_crayons_left_l287_287231

-- Define the initial number of crayons.
def initial_crayons : ℕ := 253

-- Define the percentage of crayons lost or given away.
def percentage_lost : ℝ := 35.5 / 100

-- Define the number of crayons Paul had left by the end of the school year.
def crayons_left_by_end (initial : ℕ) (perc_lost : ℝ) : ℕ :=
  initial - (perc_lost * initial).round

theorem paul_crayons_left : crayons_left_by_end initial_crayons percentage_lost = 163 := 
by
  sorry

end paul_crayons_left_l287_287231


namespace geom_symbols_l287_287057

-- Define types for Point, Line, and Plane
constant Point : Type
constant Line : Type
constant Plane : Type

-- Define incidence relation of a point belonging to a line and a line belonging to a plane
constant OnLine : Point → Line → Prop
constant InPlane : Line → Plane → Prop

-- Define specific point, line, and plane
constant A : Point
constant m : Line
constant α : Plane

-- State the theorem representing the problem
theorem geom_symbols :
  OnLine A m ∧ InPlane m α :=
sorry

end geom_symbols_l287_287057


namespace abs_complex_calc_l287_287959

theorem abs_complex_calc :
  abs ((2 : ℝ) + 2 * real.sqrt 3 * complex.I) ^ 6 = 4096 := sorry

end abs_complex_calc_l287_287959


namespace valid_paths_in_grid_l287_287045

theorem valid_paths_in_grid : 
  let grid_size := (11, 4)
  let start_point := (0, 0)
  let end_point := (11, 4)
  let obstructions := [
      ((6, 0), (6, 2)), ((7, 0), (7, 2)),
      ((6, 3), (6, 4)), ((7, 3), (7, 4))
    ]
  number_of_paths grid_size start_point end_point obstructions = 189 :=
by
  sorry

end valid_paths_in_grid_l287_287045


namespace expression_for_f_l287_287645

variable (f g : ℝ → ℝ)
variable (x : ℝ)

theorem expression_for_f (H1 : ∀ x, f (g x) = 6 * x + 3)
                         (H2 : ∀ x, g x = 2 * x + 1) :
    f = (λ x, 3 * x) := by
sor
 
end expression_for_f_l287_287645


namespace fourth_person_knight_l287_287853

-- Let P1, P2, P3, and P4 be the statements made by the four people respectively.
def P1 := ∀ x y z w : Prop, x = y ∧ y = z ∧ z = w ∧ w = ¬w
def P2 := ∃! x y z w : Prop, x = true
def P3 := ∀ x y z w : Prop, (x = true ∧ y = true ∧ z = false) ∨ (x = true ∧ y = false ∧ z = true) ∨ (x = false ∧ y = true ∧ z = true)
def P4 := ∀ x : Prop, x = true → x = true

-- Now let's express the requirement of proving that the fourth person is a knight
theorem fourth_person_knight : P4 := by
  sorry

end fourth_person_knight_l287_287853


namespace count_of_valid_n_is_two_l287_287081

def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

def target_property (n : ℤ) : Prop :=
  is_square (n / (25 - n))

def count_valid_n (min_n max_n : ℤ) : ℕ :=
  finset.card (finset.filter target_property (finset.range (max_n - min_n + 1)).map (λ k => k + min_n))

theorem count_of_valid_n_is_two : count_valid_n 0 24 = 2 :=
  sorry

end count_of_valid_n_is_two_l287_287081


namespace route_comparison_l287_287218

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l287_287218


namespace maximize_triangle_area_l287_287746

-- Definitions and prerequisites
variable (a b c : ℝ) -- semi-major axis a, semi-minor axis b, distance from center to focus c
variable (A B F : EuclideanSpace ℝ (Fin 2)) -- points A, B, and F in 2D space

-- Defining the ellipse and points
-- Assuming the ellipse equation: x^2 / a^2 + y^2 / b^2 = 1
def is_ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Assuming A, B, F as points on the ellipse with specific roles
axiom major_axis (A : EuclideanSpace ℝ (Fin 2)) : is_ellipse a b A
axiom minor_axis (B : EuclideanSpace ℝ (Fin 2)) : is_ellipse a b B
axiom focus_position (F : EuclideanSpace ℝ (Fin 2)) : is_ellipse a b F

-- Moving point M along arc AB
variable (M : EuclideanSpace ℝ (Fin 2))
axiom M_on_arc_AB (M : EuclideanSpace ℝ (Fin 2)) : is_ellipse a b M

-- Statement to prove the maximization of triangle area MBF
theorem maximize_triangle_area : ∃ M_1 : EuclideanSpace ℝ (Fin 2), 
  (tangent_at_ellipse a b M_1 ∥ line_through B F) ∧
  ∀ M : EuclideanSpace ℝ (Fin 2), 
  (is_ellipse a b M ∧ M_on_arc_AB M) →
  area_triangle M B F ≤ area_triangle M_1 B F := 
sorry

-- Definitions for tangent and area calculations
def tangent_at_ellipse (a b : ℝ) (P : EuclideanSpace ℝ (Fin 2)) : Vector ℝ (Fin 2) := 
-- Implement the tangent vector calculation specific to the ellipse
sorry

def line_through (P Q : EuclideanSpace ℝ (Fin 2)) : Vector ℝ (Fin 2) :=
-- Define vector calculation between points P and Q
sorry

def area_triangle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ :=
-- Compute area of the triangle given vertices P, Q, and R
sorry

end maximize_triangle_area_l287_287746


namespace jenna_more_than_four_times_martha_l287_287949

noncomputable def problems : ℝ := 20
noncomputable def martha_problems : ℝ := 2
noncomputable def angela_problems : ℝ := 9
noncomputable def jenna_problems : ℝ := 6  -- We calculated J = 6 from the conditions
noncomputable def mark_problems : ℝ := jenna_problems / 2

theorem jenna_more_than_four_times_martha :
  (jenna_problems - 4 * martha_problems = 2) :=
by
  sorry

end jenna_more_than_four_times_martha_l287_287949


namespace blueberries_per_pint_l287_287694

theorem blueberries_per_pint (total_blueberries : ℕ) (total_pies : ℕ) (pints_per_quart : ℕ) 
  (blueberries_per_pie : ℕ) (blueberries_per_pint : ℕ) :
  total_blueberries = 2400 →
  total_pies = 6 →
  pints_per_quart = 2 →
  blueberries_per_pie = total_blueberries / total_pies →
  blueberries_per_pint = blueberries_per_pie / pints_per_quart →
  blueberries_per_pint = 200 :=
begin
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  intro h5,
  -- We state the assumptions and conclude the proof
  sorry
end

end blueberries_per_pint_l287_287694


namespace triangle_type_l287_287304

/-- The angles of a triangle are denoted by α, β, and γ. -/
variables (α β γ : ℝ)

/-- The sum of the angles in any triangle is 180 degrees. -/
def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

/-- Type of a triangle based on its angles. -/
inductive TriangleType
| right_triangle : TriangleType
| obtuse_triangle : TriangleType

/-- Determine the type of triangle based on given conditions. -/
theorem triangle_type (α β γ : ℝ) (h_sum : angle_sum α β γ) :
  (γ > α + β → TriangleType) × (γ = α + β → TriangleType) :=
by
  sorry

end triangle_type_l287_287304


namespace find_f_prime_at_2_l287_287342

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287342


namespace total_ceremony_duration_l287_287861

theorem total_ceremony_duration (n : ℕ) (time_first : ℕ) (increment : ℕ) :
  n = 16 ∧ time_first = 10 ∧ increment = 10 ->
  let rounds := nat.log2 n in
  let total_time := (finset.range rounds).sum
      (λ r, ((n / (bit0 1^r) / 2) : ℕ) * (time_first + r * increment)) in
  total_time = 260 :=
begin
  intros h,
  rcases h with ⟨hn, ht, hi⟩,
  sorry
end

end total_ceremony_duration_l287_287861


namespace christine_savings_l287_287489

/-- Christine's commission rate as a percentage. -/
def commissionRate : ℝ := 0.12

/-- Total sales made by Christine this month in dollars. -/
def totalSales : ℝ := 24000

/-- Percentage of commission allocated to personal needs. -/
def personalNeedsRate : ℝ := 0.60

/-- The amount Christine saved this month. -/
def amountSaved : ℝ := 1152

/--
Given the commission rate, total sales, and personal needs rate,
prove the amount saved is correctly calculated.
-/
theorem christine_savings :
  (1 - personalNeedsRate) * (commissionRate * totalSales) = amountSaved :=
by
  sorry

end christine_savings_l287_287489


namespace angle_C_in_acute_triangle_value_of_c_l287_287667

-- Part (1)
theorem angle_C_in_acute_triangle
  (a b c : ℝ) (A B C : ℝ) [has_angle_eq : decidable_eq ℝ] (h_triangle : 0 < A ∧ A + B + C = π ∧ 0 < B ∧ C = π - (A + B))
  (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_sine : √3 * a = 2 * c * sin A) :
  C = π / 3 :=
sorry

-- Part (2)
theorem value_of_c
  (a b c : ℝ) (A B C : ℝ) [has_angle_eq : decidable_eq ℝ]
  (h_triangle : 0 < A ∧ A + B + C = π ∧ 0 < B ∧ 0 < C ∧ A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_a : a = 2)
  (h_area : (1/2) * a * b * sin C = (3 * √3) / 2) :
  c = √7 :=
sorry

end angle_C_in_acute_triangle_value_of_c_l287_287667


namespace ratio_of_radii_l287_287677

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l287_287677


namespace sum_sqrt_identity_l287_287188

theorem sum_sqrt_identity :
  (∑ n in Finset.range 24, 1 / (Real.sqrt (n + 1 + Real.sqrt ((n + 1)^2 - 1)))) = 12 * Real.sqrt 2 + 12 * Real.sqrt 3 :=
sorry

end sum_sqrt_identity_l287_287188


namespace cross_section_area_of_tetrahedron_l287_287098

noncomputable def area_of_cross_section (P A B C L M N : Point) (h1 : is_tetrahedron P A B C)
  (h2 : edge_length_eq P A 1) (h3 : edge_length_eq P B 1) (h4 : edge_length_eq P C 1)
  (h5 : midpoint L P A) (h6 : midpoint M P B) (h7 : midpoint N P C) : ℝ :=
  have h8 : is_parallel_plane LMN ABC := sorry,
  have h9 : circumradius ABC = (1 : ℝ) / (Real.sqrt 3) := sorry,
  have h10 : area_of_cross_section_circle ((1 : ℝ) / (Real.sqrt 3)) := Real.pi / (3),
  h10

theorem cross_section_area_of_tetrahedron (P A B C L M N : Point) 
  (h1 : is_tetrahedron P A B C) (h2 : edge_length_eq P A 1) (h3 : edge_length_eq P B 1)
  (h4 : edge_length_eq P C 1) (h5 : midpoint L P A) (h6 : midpoint M P B)
  (h7 : midpoint N P C)
  : area_of_cross_section P A B C L M N h1 h2 h3 h4 h5 h6 h7 = Real.pi / 3 := 
sorry

end cross_section_area_of_tetrahedron_l287_287098


namespace area_triangle_PTR_l287_287754

theorem area_triangle_PTR (PQ QR : ℝ) (PQ_pos : PQ = 5) (QR_pos : QR = 12) :
  ∃ (PR QT PT : ℝ), PR = Real.sqrt (PQ^2 + QR^2) ∧ 
  QT = 60 / PR ∧ 
  PT = (5 / 12) * PR ∧ 
  (1 / 2) * PT * QT = 325 / 31 :=
  by
  -- Given conditions
  have PQ := PQ_pos
  have QR := QR_pos

  -- Step 1: Calculate diagonal PR
  let PR := Real.sqrt (PQ^2 + QR^2)
  
  -- Step 2: Calculate QT using area relation
  have QT := 60 / PR
  
  -- Step 3: Calculate PT using similarity
  have PT := (5 / 12) * PR
  
  -- Step 4: Compute the area of triangle PTR
  let area_PTR := (1 / 2) * PT * QT
  
  -- Existence statement
  use [PR, QT, PT, PR == Real.sqrt (PQ^2 + QR^2), QT == 60 / PR, PT == (5 / 12) * PR, area_PTR == 325 / 31]
  exact sorry

end area_triangle_PTR_l287_287754


namespace binom_19_10_l287_287490

theorem binom_19_10 (h₁ : nat.choose 17 7 = 19448) (h₂ : nat.choose 17 9 = 24310) : nat.choose 19 10 = 92378 := by
  sorry

end binom_19_10_l287_287490


namespace trapezoid_area_l287_287454

theorem trapezoid_area {a b c d e : ℝ} (h1 : a = 40) (h2 : b = 40) (h3 : c = 50) (h4 : d = 50) (h5 : e = 60) : 
  (a + b = 80) → (c * c = 2500) → 
  (50^2 - 30^2 = 1600) → ((50^2 - 30^2).sqrt = 40) → 
  (((e - 2 * ((a ^ 2 - (30) ^ 2).sqrt)) * 40) / 2 = 1336) :=
sorry

end trapezoid_area_l287_287454


namespace line_intersects_ellipse_l287_287267

theorem line_intersects_ellipse (b : ℝ) : (∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + 1 → ((x^2 / 5) + (y^2 / b) = 1))
  ↔ b ∈ (Set.Ico 1 5 ∪ Set.Ioi 5) := by
sorry

end line_intersects_ellipse_l287_287267


namespace size_of_smaller_package_l287_287033

theorem size_of_smaller_package :
  ∃ (small large : ℕ), small = 5 ∧ large = 10 ∧ (∃ (n m : ℕ), m = 3 ∧ n = m + 2 ∧ (m * large + n * small = 55)) :=
by {
  -- small: size of the smaller package
  -- large: size of the larger package
  -- n: number of smaller packages
  -- m: number of larger packages
  existsi [5, 10],
  split,
  { refl },
  split,
  { refl },
  existsi [2 + 1 + 1, 3],
  split,
  { refl },
  split,
  { refl },
  norm_num,
}

end size_of_smaller_package_l287_287033


namespace derivative_at_2_l287_287356

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287356


namespace max_value_at_one_l287_287322

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287322


namespace ratio_of_base_areas_l287_287141

theorem ratio_of_base_areas (S1 S2 V1 V2 : ℝ) (R r H h: ℝ)
  (hV_ratio : V1 / V2 = 3 / 2)
  (h_lateral_areas_equal : 2 * real.pi * R * H = 2 * real.pi * r * h) :
  (S1 / S2 = 9 / 4) :=
by 
  sorry

end ratio_of_base_areas_l287_287141


namespace solution_l287_287842

noncomputable def problem_statement : Prop :=
  (deriv (λ x : ℝ, x) = 1) ∧ (deriv (λ x : ℝ, Real.sqrt x) = (1 / (2 * Real.sqrt x)))

theorem solution : problem_statement :=
  by
    split
    exact deriv_id
    sorry

end solution_l287_287842


namespace constant_ratio_of_inscribed_polygons_l287_287982

-- Definitions and conditions
def isInscribedPolygon (P : ℕ → Point) (n : ℕ) := 
  ∃ (O : Point) (r : ℝ), ∀ i, 1 ≤ i ∧ i ≤ n → dist O (P i) = r

def areParallelSegments (A B : ℕ → Point) (n : ℕ) :=
  ∀ i, 1 ≤ i ∧ i ≤ n → is_parallel (segment (A i) (A (i + 1))) (segment (B i) (B (i + 1)))

-- Statement of the problem
theorem constant_ratio_of_inscribed_polygons
  (A B : ℕ → Point) (n : ℕ)
  (hA : isInscribedPolygon A n)
  (hB : isInscribedPolygon B n)
  (hParallel : areParallelSegments A B n) :
  ∃ k : ℝ, ∀ i, 1 ≤ i ∧ i ≤ n → (dist (A i) (A (i + 1)) / dist (B i) (B (i + 1))) = k :=
by sorry

end constant_ratio_of_inscribed_polygons_l287_287982


namespace inequality_2n_1_lt_n_plus_1_sq_l287_287080

theorem inequality_2n_1_lt_n_plus_1_sq (n : ℕ) (h : 0 < n) : 2 * n - 1 < (n + 1) ^ 2 := 
by 
  sorry

end inequality_2n_1_lt_n_plus_1_sq_l287_287080


namespace tangent_intersection_point_l287_287890

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287890


namespace variance_of_given_data_set_l287_287586

def data_set : List ℝ := [3, 6, 9, 8, 4]

def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length : ℝ)

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ)^2)).sum / (l.length : ℝ)

theorem variance_of_given_data_set :
  variance data_set = 5.2 :=
by
  sorry

end variance_of_given_data_set_l287_287586


namespace tangent_lines_through_point_l287_287524

theorem tangent_lines_through_point (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) ∧ (x = -2 ∨ (15*x + 8*y - 10 = 0)) ↔ 
  (x = -2 ∨ (15*x + 8*y - 10 = 0)) :=
by
  sorry

end tangent_lines_through_point_l287_287524


namespace cathy_can_win_l287_287720

theorem cathy_can_win (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (f : ℕ → ℕ) (hf : ∀ i, f i < n + 1), (∀ i j, (i < j) → (f i < f j) → (f j = f i + 1)) → n ≤ 2^(k-1)) :=
sorry

end cathy_can_win_l287_287720


namespace real_solutions_fx_eq_f_negx_l287_287264

noncomputable def f : ℝ → ℝ := sorry

theorem real_solutions_fx_eq_f_negx :
  (∀ x : ℝ, x ≠ 0 → f(x) * 2 * f(1 / x) = x^2 + 4) →
  (∀ x : ℝ, f(x) = f(-x) → (x = 1 ∨ x = -1)) :=
by
  intro hfx_eq
  intro hfx_eq_fnegx
  sorry

end real_solutions_fx_eq_f_negx_l287_287264


namespace number_of_buyers_l287_287809

theorem number_of_buyers 
  (today yesterday day_before : ℕ) 
  (h1 : today = yesterday + 40) 
  (h2 : yesterday = day_before / 2) 
  (h3 : day_before + yesterday + today = 140) : 
  day_before = 67 :=
by
  -- skip the proof
  sorry

end number_of_buyers_l287_287809


namespace erased_number_l287_287469

theorem erased_number (a b : ℤ) (h1 : ∀ n : ℤ, n ∈ set.range (λ i, a + i) ↔ n ∈ set.range (λ i, a - 4 + i)) 
                      (h2 : 8 * a - b = 1703)
                      (h3 : -4 ≤ b ∧ b ≤ 4) : a + b = 214 := 
by
    sorry

end erased_number_l287_287469


namespace total_amount_l287_287933

noncomputable def x_share : ℝ := 60
noncomputable def y_share : ℝ := 27
noncomputable def z_share : ℝ := 0.30 * x_share

theorem total_amount (hx : y_share = 0.45 * x_share) : x_share + y_share + z_share = 105 :=
by
  have hx_val : x_share = 27 / 0.45 := by
  { -- Proof that x_share is indeed 60 as per the given problem
    sorry }
  sorry

end total_amount_l287_287933


namespace fD_range_l287_287017

noncomputable def fA (x : ℝ) : ℝ := (2/3)^(-|x|)
noncomputable def fB (x : ℝ) : ℝ := x^2 - x + 1
noncomputable def fC (x : ℝ) : ℝ := (1 - x) / (1 + x)
noncomputable def fD (x : ℝ) : ℝ := abs (Real.logBase 2 (x + 1))

theorem fD_range :
  ∀ y, y ≥ 0 ↔ ∃ x : ℝ, fD x = y := 
sorry

end fD_range_l287_287017


namespace tangent_line_intersection_l287_287883

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287883


namespace one_eq_one_of_ab_l287_287628

variable {a b : ℝ}

theorem one_eq_one_of_ab (h : a * b = a^2 - a * b + b^2) : 1 = 1 := by
  sorry

end one_eq_one_of_ab_l287_287628


namespace min_colors_painting_problem_l287_287193

def paintNumbersWithMinColors (n : ℕ) : Prop :=
  ∀ (f : ℕ → ℕ),
    (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 100 ∧ 1 ≤ y ∧ y ≤ 100 ∧ x ≠ y ∧ (x + y) % 4 = 0 → f x ≠ f y) →
    (∃ (g : ℕ → ℕ), (∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → g x < n ∧ f x = g x))

theorem min_colors_painting_problem : ∃ n, (n ≥ 25 ∧ paintNumbersWithMinColors n) :=
begin
  existsi 25,
  split,
  { linarith, },
  { sorry, }
end

end min_colors_painting_problem_l287_287193


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287542

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287542


namespace tangent_line_intersection_l287_287885

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287885


namespace smallest_non_palindrome_product_l287_287548

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287548


namespace min_value_l287_287528

theorem min_value (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < real.pi / 2) :
  (∃ θ, θ ∈ set.Ioo 0 (real.pi / 2) ∧ ((a / real.sin θ) + (b / real.cos θ)) = (real.sqrt (real.sqrt (a^2) ^ 3 + real.sqrt (b^2) ^ 3)) ^ 2) :=
by sorry

end min_value_l287_287528


namespace geo_seq_a3_equals_one_l287_287602

theorem geo_seq_a3_equals_one (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_T5 : a 1 * a 2 * a 3 * a 4 * a 5 = 1) : a 3 = 1 :=
sorry

end geo_seq_a3_equals_one_l287_287602


namespace rolls_to_neighbor_l287_287079

theorem rolls_to_neighbor (total_needed rolls_to_grandmother rolls_to_uncle rolls_needed : ℕ) (h1 : total_needed = 45) (h2 : rolls_to_grandmother = 1) (h3 : rolls_to_uncle = 10) (h4 : rolls_needed = 28) :
  total_needed - rolls_needed - (rolls_to_grandmother + rolls_to_uncle) = 6 := by
  sorry

end rolls_to_neighbor_l287_287079


namespace sin_pi_div_three_l287_287808

theorem sin_pi_div_three : Real.sin (π / 3) = Real.sqrt 3 / 2 := 
sorry

end sin_pi_div_three_l287_287808


namespace division_by_fraction_l287_287031

theorem division_by_fraction (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (h : a / (1 / b) = c) : c = a * b := by
  sorry

example : 12 / (1 / 12) = 144 := by
  show 12 * 12 = 144
  sorry

end division_by_fraction_l287_287031


namespace polygon_angle_arithmetic_progression_l287_287917

theorem polygon_angle_arithmetic_progression
  (h1 : ∀ {n : ℕ}, n ≥ 3)   -- The polygon is convex and n-sided
  (h2 : ∀ (angles : Fin n → ℝ), (∀ i j, i < j → angles i + 5 = angles j))   -- The interior angles form an arithmetic progression with a common difference of 5°
  (h3 : ∀ (angles : Fin n → ℝ), (∃ i, angles i = 160))  -- The largest angle is 160°
  : n = 9 := sorry

end polygon_angle_arithmetic_progression_l287_287917


namespace sup_good_l287_287826

def is_good (d : ℝ) : Prop :=
  ∃ (a : ℕ → ℝ), (∀ n, 0 < a n ∧ a n < d) ∧ (∀ n, ∃ (p : Π i, set.Icc 0 d), (∀ i, i < n → set.Icc 0 d ⊆ ⋃ j, set.Ioo (p j).lower (p j).upper) ∧ ∀ i j, i < j → (p i ∩ p j = ∅) ∧ (∃ k, ∀ i j, p i ≠ p j → p i.upper - p i.lower ≤ 1 / (k : ℝ)))

theorem sup_good : supr (λ d, is_good d) = Real.log 2 :=
sorry

end sup_good_l287_287826


namespace greatest_m_value_l287_287042

theorem greatest_m_value (x y z u : ℕ) (hx : x ≥ y) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ m ≤ x / y :=
sorry

end greatest_m_value_l287_287042


namespace circle_radius_order_l287_287034

theorem circle_radius_order (r_X r_Y r_Z : ℝ)
  (hX : r_X = π)
  (hY : 2 * π * r_Y = 8 * π)
  (hZ : π * r_Z^2 = 9 * π) :
  r_Z < r_X ∧ r_X < r_Y :=
by {
  sorry
}

end circle_radius_order_l287_287034


namespace cost_per_ticket_l287_287466

/-- Adam bought 13 tickets and after riding the ferris wheel, he had 4 tickets left.
    He spent 81 dollars riding the ferris wheel, and we want to determine how much each ticket cost. -/
theorem cost_per_ticket (initial_tickets : ℕ) (tickets_left : ℕ) (total_cost : ℕ) (used_tickets : ℕ) 
    (ticket_cost : ℕ) (h1 : initial_tickets = 13) 
    (h2 : tickets_left = 4) 
    (h3 : total_cost = 81) 
    (h4 : used_tickets = initial_tickets - tickets_left) 
    (h5 : ticket_cost = total_cost / used_tickets) : ticket_cost = 9 :=
by {
    sorry
}

end cost_per_ticket_l287_287466


namespace derivative_at_2_l287_287358

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287358


namespace y_intercept_of_parallel_line_l287_287728

def Line (m c : ℝ) : (ℝ → ℝ) := λ x, m * x + c

theorem y_intercept_of_parallel_line
  (m : ℝ) (c₁ c₂ : ℝ)
  (p : ℝ × ℝ)
  (h_parallel : m = -3)
  (h_line_b_contains_p : p = (3, -1))
  (h_line_b_parallel_to : ∀ b : Line m c₁, b(0) = 6) :
  ∃ c₂ : ℝ, Line m c₂ 0 = 8 :=
by
  sorry

end y_intercept_of_parallel_line_l287_287728


namespace ratio_of_radii_l287_287678

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l287_287678


namespace max_sides_with_four_obtuse_l287_287053

-- Define the given conditions for the polygon
def obtuse_angle (o : ℝ) : Prop := 90 < o ∧ o < 180
def acute_angle (a : ℝ) : Prop := 0 < a ∧ a < 90

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem max_sides_with_four_obtuse :
  ∀ (n : ℕ), (∀ o1 o2 o3 o4 a1 a2 ... an-4: ℝ,
  (obtuse_angle o1) ∧ (obtuse_angle o2) ∧ (obtuse_angle o3) ∧ (obtuse_angle o4) ∧
  (∀ a i, i ∈ finset.range (n - 4) -> acute_angle a) ∧
  (sum_of_interior_angles n = o1 + o2 + o3 + o4 + ∑ i in finset.range (n - 4), a i)) -> n ≤ 7 :=
sorry

end max_sides_with_four_obtuse_l287_287053


namespace example_problem_l287_287417

-- Define the numbers of students in each grade
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300

-- Define the total number of spots for the trip
def total_spots : ℕ := 40

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the fraction of sophomores relative to the total number of students
def fraction_sophomores : ℚ := sophomores / total_students

-- Define the number of spots allocated to sophomores
def spots_sophomores : ℚ := fraction_sophomores * total_spots

-- The theorem we need to prove
theorem example_problem : spots_sophomores = 13 :=
by 
  sorry

end example_problem_l287_287417


namespace total_cost_shoes_and_jerseys_l287_287695

theorem total_cost_shoes_and_jerseys 
  (shoes : ℕ) (jerseys : ℕ) (cost_shoes : ℕ) (cost_jersey : ℕ) 
  (cost_total_shoes : ℕ) (cost_per_shoe : ℕ) (cost_per_jersey : ℕ) 
  (h1 : shoes = 6)
  (h2 : jerseys = 4) 
  (h3 : cost_per_jersey = cost_per_shoe / 4)
  (h4 : cost_total_shoes = 480)
  (h5 : cost_per_shoe = cost_total_shoes / shoes)
  (h6 : cost_per_jersey = cost_per_shoe / 4)
  (total_cost : ℕ) 
  (h7 : total_cost = cost_total_shoes + cost_per_jersey * jerseys) :
  total_cost = 560 :=
sorry

end total_cost_shoes_and_jerseys_l287_287695


namespace derivative_at_2_l287_287371

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287371


namespace concurrency_of_perpendiculars_l287_287496

theorem concurrency_of_perpendiculars 
  (A B C A_B A_B B_A B_C A_C C_B A_C A_B_C B_A_C C_B B_A_B A_B_A : Point)
  (α : ℝ)
  (h_acute : ∀ (P Q R : Point), acute_angle P Q R)
  (h_isosceles_A : isosceles_triangle A B A_B ∧ isosceles_triangle A B B_A)
  (h_isosceles_B : isosceles_triangle A C A_C ∧ isosceles_triangle A C C_A)
  (h_isosceles_C : isosceles_triangle B C B_C ∧ isosceles_triangle B C C_B)
  (h_equal_sides_A : dist A B = dist A B_A ∧ dist A B = dist B_A A)
  (h_equal_sides_B : dist A C = dist A C_A ∧ dist A C = dist C_A A)
  (h_equal_sides_C : dist B C = dist B C_B ∧ dist B C = dist C_B B)
  (h_equal_angles : ∀ (P Q R S : Point), ∠ P Q R = α ∧ ∠ Q P S = α)
  (h_angle_lt_90 : α < 90) :
  concurrent (perpendicular_from A (line_through B_A C_A)) 
             (perpendicular_from B (line_through A_B C_B)) 
             (perpendicular_from C (line_through A_C B_C)) :=
sorry

end concurrency_of_perpendiculars_l287_287496


namespace evaluate_expression_l287_287512

theorem evaluate_expression (a b : ℕ) (h_a : a = 3) (h_b : b = 2) : (a^b)^b - (b^a)^a = -431 := by
  sorry

end evaluate_expression_l287_287512


namespace proof_problem_l287_287126

open Classical

variable (a b : ℝ)
variable (A B : RealAngle)
variable (ABC : Triangle)

-- Definition of proposition p1
def p1 : Prop := ∃ (a b : ℝ), a^2 - a * b + b^2 < 0

-- Definition of proposition p2
def p2 : Prop := ∀ (A B : RealAngle) (ABC : Triangle), ∠A > ∠B → sin A > sin B

-- Main proof problem
theorem proof_problem : (¬p1) ∧ p2 := by
  sorry

end proof_problem_l287_287126


namespace min_trips_correct_l287_287396

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l287_287396


namespace investment_revenue_n_year_company_profit_starts_from_8th_year_l287_287915

noncomputable def investment (n : ℕ) : ℝ :=
  if n = 1 then 10
  else if n ≤ 6 then 10 * (1 / 2)^(n - 1)
  else 0.2

noncomputable def revenue (n : ℕ) : ℝ :=
  if n = 1 then 0.4
  else if n ≤ 6 then 40 + 0.8 * n - 1
  else 0.44

def company_start_profit (n : ℕ) : ℕ :=
  if n < 8 then 6
  else n

theorem investment_revenue_n_year (n : ℕ) :
  investment n = if n ≤ 6 then 10 * (1 / 2)^(n - 1) else 0.2 ∧
  revenue n = if n ≤ 6 then 40 + 0.8 * n - 1 else 0.44 :=
sorry

theorem company_profit_starts_from_8th_year :
  company_start_profit 8 = 8 :=
sorry

end investment_revenue_n_year_company_profit_starts_from_8th_year_l287_287915


namespace find_lambda_l287_287576

theorem find_lambda
  (λ : ℝ)
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (h_a : a = (λ + 1, 0, 2 * λ))
  (h_b : b = (6, 0, 2))
  (h_parallel : ∃ m : ℝ, a = (m * b.1, m * b.2, m * b.3)) :
  λ = 1 / 5 :=
by
  sorry

end find_lambda_l287_287576


namespace range_of_m_l287_287127

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) : -3 < m ∧ m < 1 := 
sorry

end range_of_m_l287_287127


namespace inequality_a_plus_b_plus_d_le_3c_l287_287594

theorem inequality_a_plus_b_plus_d_le_3c
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (h1 : a^2 + ab + b^2 = 3c^2)
  (h2 : a^3 + a^2b + ab^2 + b^3 = 4d^3)
  : a + b + d ≤ 3c := 
  sorry

end inequality_a_plus_b_plus_d_le_3c_l287_287594


namespace sample_size_l287_287411

theorem sample_size (N1 N2 N3 : ℕ) (p : ℝ) (hN1 : N1 = 280) (hN2 : N2 = 320) (hN3 : N3 = 400) (hp : p = 0.2) : 
  let n := N1 * p + N2 * p + N3 * p in n = 200 :=
by
  sorry

end sample_size_l287_287411


namespace axis_of_symmetry_l287_287617

noncomputable def f (x ϕ : ℝ) : ℝ := Real.sin (x - ϕ)

theorem axis_of_symmetry (ϕ : ℝ) (h : ∫ x in 0 .. (2 * Real.pi / 3), f x ϕ = 0) :
  ∃ k : ℤ, ∀ x : ℝ, f x ϕ = f (2 * k * Real.pi + (5 * Real.pi / 6) - x) ϕ :=
sorry

end axis_of_symmetry_l287_287617


namespace total_chairs_l287_287168

theorem total_chairs (tables : ℕ) (half_tables_chairs : ℕ) (fifth_tables_chairs : ℕ) (tenth_tables_chairs : ℕ) (rest_tables_chairs : ℕ) :
  tables = 50 →
  (∃ half_tables, half_tables = 25 ∧ half_tables_chairs = half_tables * 2) →
  (∃ fifth_tables, fifth_tables = 10 ∧ fifth_tables_chairs = fifth_tables * 3) →
  (∃ tenth_tables, tenth_tables = 5 ∧ tenth_tables_chairs = tenth_tables * 5) →
  (∃ rest_tables, rest_tables = 10 ∧ rest_tables_chairs = rest_tables * 4) →
  half_tables_chairs + fifth_tables_chairs + tenth_tables_chairs + rest_tables_chairs = 145 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  cases h₂ with h₆ h₇
  cases h₇ with h₈ h₉
  cases h₃ with h₁₀ h₁₁
  cases h₁₁ with h₁₂ h₁₃
  cases h₄ with h₁₄ h₁₅
  cases h₁₅ with h₁₆ h₁₇
  cases h₅ with h₁₈ h₁₉
  cases h₁₉ with h₂₀ h₂₁
  have ht: (25 + 10 + 5 + 10) = 50 := by norm_num
  rw ht at h₁
  rw [h₈, h₁₂, h₁₆, h₂₀]
  norm_num
  sorry

end total_chairs_l287_287168


namespace Bill_wins_at_least_once_in_three_games_l287_287029

noncomputable def probability_win_at_least_once : ℚ :=
  let P_win_single_game : ℚ := 4 / 36
  let P_lose_single_game := 1 - P_win_single_game
  let P_lose_all_three_games := P_lose_single_game * P_lose_single_game * P_lose_single_game
  let P_win_at_least_once := 1 - P_lose_all_three_games
  P_win_at_least_once

theorem Bill_wins_at_least_once_in_three_games :
  probability_win_at_least_once = 217 / 729 :=
by
  unfold probability_win_at_least_once
  sorry

end Bill_wins_at_least_once_in_three_games_l287_287029


namespace work_days_for_C_l287_287866

theorem work_days_for_C (x : ℝ) (hx : 0 < x) :
  (1/11 * 5 + 1/5 * 2.5 + 1/x * 2.5 = 1) -> x = 2.5 :=
by
  intro h
  -- setting up the equation based on the given problem
  have h_eq : (1 / 11) * 5 + (1 / 5) * 2.5 + (1 / x) * 2.5 = 1 := h
  sorry

-- Example instance to check if the statement is correct
noncomputable def x_value := (1 / 11) * 5 + (1 / 5) * 2.5 = 1

#eval work_days_for_C 2.5 (by norm_num) x_value


end work_days_for_C_l287_287866


namespace compare_p_q_l287_287578

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : a * b = 1)
variable (h4 : 0 < c)
variable (h5 : c < 1)

noncomputable def p := log c ((a^2 + b^2) / 2)
noncomputable def q := log c ((1 / (sqrt a + sqrt b))^2)

theorem compare_p_q : p < q :=
by sorry

end compare_p_q_l287_287578


namespace trapezoid_area_l287_287448

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l287_287448


namespace length_of_DE_l287_287687

theorem length_of_DE (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  [has_dist A ℝ] [has_dist B ℝ] [has_dist C ℝ] [has_dist D ℝ] [has_dist E ℝ] 
  (BC : ℝ) (angle_C : ℝ) (CD : ℝ) (DE : ℝ)
  (hBC : BC = 30)
  (hAngleC : angle_C = π / 4)
  (hDmidpoint : CD = BC / 2)
  (hPerpBisector : DE = CD)
  : DE = 15 :=
sorry

end length_of_DE_l287_287687


namespace small_palindrome_check_l287_287564

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287564


namespace repeating_decimal_fraction_l287_287516

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end repeating_decimal_fraction_l287_287516


namespace limit_comb_sum_eq_one_l287_287032

theorem limit_comb_sum_eq_one :
  (tendsto (λ n : ℕ, ((n * (n - 1) / 2 : ℝ) / (n * (n + 1) / 2)) : ℝ) atTop (𝓝 1)) :=
by sorry

end limit_comb_sum_eq_one_l287_287032


namespace modulus_of_complex_number_l287_287114

def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem modulus_of_complex_number (a : ℝ) (ha : pure_imaginary ((2 - complex.I) / (a + complex.I))) :
  complex.abs (2 * a + real.sqrt 3 * complex.I) = 2 :=
sorry

end modulus_of_complex_number_l287_287114


namespace chord_EF_length_l287_287673

theorem chord_EF_length :
  ∀ (A B C D E F G : Point) (O N P : Circle) 
  (rO rN rP : ℝ), 
  rO = 10 → rN = 20 → rP = 25 →
  A ≠ B → B ≠ C → C ≠ D →
  diameter O AB → diameter N BC → diameter P CD →
  tangent AG P G →
  intersects AG N E F →
  length_chord N E F = 25.56 :=
begin
  intros A B C D E F G O N P rO rN rP h_rO h_rN h_rP h_neq1 h_neq2 h_neq3 h_diamO h_diamN h_diamP h_tangent h_intersect,
  sorry
end

end chord_EF_length_l287_287673


namespace a_gt_b_iff_a_ln_a_gt_b_ln_b_l287_287592

theorem a_gt_b_iff_a_ln_a_gt_b_ln_b {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (a > b) ↔ (a + Real.log a > b + Real.log b) :=
by sorry

end a_gt_b_iff_a_ln_a_gt_b_ln_b_l287_287592


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287552

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287552


namespace bounded_sequence_eq_two_l287_287521

theorem bounded_sequence_eq_two (a : ℕ → ℕ) (H_bounded : ∃ M, ∀ n, a n ≤ M)
  (H_nat : ∀ n, a n ∈ ℕ)
  (H_recurrence : ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))) :
  ∀ i, a i = 2 := sorry

end bounded_sequence_eq_two_l287_287521


namespace correct_statements_l287_287749

-- Definitions
def p_A : ℚ := 1 / 2
def p_B : ℚ := 1 / 3

-- Statements to be verified
def statement1 := (p_A * (1 - p_B) + (1 - p_A) * p_B) = (1 / 2 + 1 / 3)
def statement2 := (p_A * p_B) = (1 / 2 * 1 / 3)
def statement3 := (p_A * (1 - p_B) + p_A * p_B) = (1 / 2 * 2 / 3 + 1 / 2 * 1 / 3)
def statement4 := (1 - (1 - p_A) * (1 - p_B)) = (1 - 1 / 2 * 2 / 3)

-- Theorem stating the correct sequence of statements
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬(statement1 ∨ statement3) :=
by
  sorry

end correct_statements_l287_287749


namespace savings_sum_l287_287253

-- Define the values assigned to each coin type
def penny := 0.01
def nickel := 0.05
def dime := 0.10

-- Define the number of coins each person has
def teaganPennies := 200
def rexNickels := 100
def toniDimes := 330

-- Calculate the total amount saved by each person
def teaganSavings := teaganPennies * penny
def rexSavings := rexNickels * nickel
def toniSavings := toniDimes * dime

-- Calculate the total savings of all three persons together
def totalSavings := teaganSavings + rexSavings + toniSavings

theorem savings_sum : totalSavings = 40 := by
  -- the actual proof is omitted, indicated by sorry
  sorry

end savings_sum_l287_287253


namespace math_problem_l287_287111

noncomputable def sin_alpha_and_cos_alpha (α : ℝ) (sin_2α : ℝ) (h1 : \sin 2α = 4 / 5) 
  (h2 : α ∈ (0 : ℝ, π / 4)) : ℝ × ℝ :=
⟨sqrt(5) / 5, 2 * sqrt(5) / 5⟩

noncomputable def tan_alpha_plus_2beta (α β : ℝ) (h1 : \sin 2α = 4 / 5)
  (h2 : α ∈ (0 : ℝ, π / 4)) (h3 : \sin (β - π / 4) = 3 / 5) 
  (h4 : β ∈ (π / 4, π / 2)) : ℝ :=
2 / 11

theorem math_problem 
  (α β : ℝ)
  (h1 : \sin 2α = 4 / 5) 
  (h2 : α ∈ (0 : ℝ, π / 4)) 
  (h3 : \sin (β - π / 4) = 3 / 5) 
  (h4 : β ∈ (π / 4, π / 2)) : 
  (sin_alpha_and_cos_alpha α (sin 2α) h1 h2 = (sqrt(5) / 5, 2 * sqrt(5) / 5)) ∧
  (tan_alpha_plus_2beta α β h1 h2 h3 h4 = 2 / 11) :=
sorry

end math_problem_l287_287111


namespace problem_statement_l287_287196

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

def sum_n : ℤ :=
  (List.sum $ List.filter (λ n, is_perfect_square (n^2 + 12 * n - 1947)) (List.range 10000))

theorem problem_statement :
  sum_n % 1000 = 312 :=
by
  sorry

end problem_statement_l287_287196


namespace cindy_correct_answer_l287_287965

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l287_287965


namespace smallest_m_in_interval_l287_287048

def seq (x : ℕ → ℝ) (x0 : ℝ) (rec : ∀ n, x (n + 1) = (x n ^ 2 + 4 * x n + 3) / (x n + 7)) := x

noncomputable def x : ℕ → ℝ := seq (λ x n, by sorry) 7 (by sorry)

theorem smallest_m_in_interval :
  let m := Nat.find (λ m, x m ≤ 3 + 1 / 2 ^ 30) in
  201 ≤ m ∧ m ≤ 300 :=
by
  sorry

end smallest_m_in_interval_l287_287048


namespace derivative_at_2_l287_287359

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287359


namespace train_length_l287_287014

open Real

def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def length_of_train (speed_train_kmph speed_man_kmph time_sec : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := kmph_to_mps relative_speed_kmph
  relative_speed_mps * time_sec

theorem train_length  : 
  length_of_train 116.99 3 9 = 299.97 :=
by
  sorry

end train_length_l287_287014


namespace factor_poly_PQ_sum_l287_287153

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l287_287153


namespace income_of_person_l287_287265

theorem income_of_person (x: ℝ) (h : 9 * x - 8 * x = 2000) : 9 * x = 18000 :=
by
  sorry

end income_of_person_l287_287265


namespace part1_part2_part3_l287_287049

-- Part (1): Prove that y = 5x + 2 is the composite function of y1 and y2 with m = 3 and n = 1
theorem part1 (x : ℝ) :
  let y1 := x + 1
  let y2 := 2 * x - 1
  let y := 3 * y1 + y2
  y = 5 * x + 2 :=
by
  let y1 := x + 1
  let y2 := 2 * x - 1
  let y := 3 * y1 + y2
  have h1 : y1 = x + 1 := rfl
  have h2 : y2 = 2 * x - 1 := rfl
  have h3 : y = 3 * (x + 1) + (2 * x - 1) := rfl
  have h4 : y = 3 * x + 3 + 2 * x - 1 := by rw [h3]; ring
  have h5 : y = 5 * x + 2 := by rw [h4]
  exact eq.refl (5 * x + 2)

-- Part (2): Prove the coordinates of intersection point P
theorem part2 (p : ℝ) :
  let x := 2 * p + 1
  let y := p - 1
  (x, y) = ((2 * p + 1 : ℝ), p - 1) :=
by
  let y1 := (2 * p + 1) - p - 2
  let y2 := - (2 * p + 1) + 3 * p
  have h1 : y1 = p - 1 := by ring
  have h2 : y2 = p - 1 := by ring
  show (2 * p + 1, p - 1) = (2 * p + 1, p - 1)
  exact eq.refl (2 * p + 1, p - 1)

-- Part (3): Given m + n > 1, prove that p < 1
theorem part3 (m n p : ℝ) (h : m + n > 1) :
  let P := (2 * p + 1, p - 1)
  P.2 > (m - n) * P.1 + (3 * p * n - m * p - 2 * m) → p < 1 :=
by
  let x := 2 * p + 1
  let y := p - 1
  let comp := (m - n) * x + (3 * p * n - m * p - 2 * m)
  have h1 : y = p - 1 := rfl
  have h2 : y > comp := λ hxy => hxy
  have h3 : p - 1 > (m + n) * (p - 1) := by sorry -- simplifying the inequality
  show p < 1
  exact sorry -- final steps for the proof

end part1_part2_part3_l287_287049


namespace minimum_xy_value_l287_287406

noncomputable def minimumValueOfXY (x y : ℝ) : ℝ :=
  if 1 + cos (2 * x + 3 * y - 1) * cos (2 * x + 3 * y - 1) 
     = (x ^ 2 + y ^ 2 + 2 * (x + 1) * (1 - y)) / (x - y + 1) then
    x * y
  else 
    1 / 0  -- This guard ensures that the conditional is correctly enforced.

theorem minimum_xy_value : ∃ (x y : ℝ), 
    1 + cos (2 * x + 3 * y - 1) * cos (2 * x + 3 * y - 1)
    = (x ^ 2 + y ^ 2 + 2 * (x + 1) * (1 - y)) / (x - y + 1) 
    ∧ xy = 1 / 25 := 
begin
  sorry
end

end minimum_xy_value_l287_287406


namespace find_z_plus_inverse_y_l287_287769

theorem find_z_plus_inverse_y
  (x y z : ℝ)
  (h1 : x * y * z = 1)
  (h2 : x + 1/z = 10)
  (h3 : y + 1/x = 5) :
  z + 1/y = 17 / 49 :=
by
  sorry

end find_z_plus_inverse_y_l287_287769


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287535

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287535


namespace find_m_l287_287041

theorem find_m (m : ℤ) : (∀ x : ℝ, x ≠ 0 → x^(m^2 - 2*m - 3) ≠ 0) ∧ (∀ x : ℝ, f(x) = x^(m^2 - 2*m - 3) → f x = f (-x)) → m = 1 := by
  sorry

end find_m_l287_287041


namespace pipeA_filling_time_l287_287936

noncomputable def pipeA_fill_time := 60

def pipeB_fill_time := 40
def total_fill_time := 30
def half_fill_time := total_fill_time / 2

theorem pipeA_filling_time :
  let t := pipeA_fill_time in
  let rateB := 1 / pipeB_fill_time in
  let rateA := 1 / t in
  let rateA_B := rateA + rateB in
  rateB * half_fill_time + rateA_B * half_fill_time = 1 →
  t = 60 :=
by
  sorry

end pipeA_filling_time_l287_287936


namespace find_x_for_sin_minus_cos_eq_sqrt2_l287_287986

theorem find_x_for_sin_minus_cos_eq_sqrt2 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
by
  sorry

end find_x_for_sin_minus_cos_eq_sqrt2_l287_287986


namespace lucy_money_l287_287159

variable (L : ℕ) -- Value for Lucy's original amount of money

theorem lucy_money (h1 : ∀ (L : ℕ), L - 5 = 10 + 5 → L = 20) : L = 20 :=
by sorry

end lucy_money_l287_287159


namespace estimate_probability_of_seedling_survival_l287_287913

def survival_rates : List ℚ := [0.800, 0.870, 0.923, 0.883, 0.890, 0.915, 0.905, 0.897, 0.902]

-- Define a function to compute the average of a list of rational numbers
def average (l : List ℚ) : ℚ :=
  l.sum / l.length

-- Estimate the probability of seedling survival after transplantation
def estimated_survival_probability : ℚ :=
  (Float.ofRat (average survival_rates)).round.toRat / 10

theorem estimate_probability_of_seedling_survival : estimated_survival_probability = 0.9 := by
  sorry

end estimate_probability_of_seedling_survival_l287_287913


namespace tangent_point_value_l287_287880

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287880


namespace petya_cannot_achieve_goal_l287_287441

theorem petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ ∀ (glasses : Fin (2 * n) → bool),
      (∃ i : Fin (2 * n),
        glasses i = glasses (i + 1) ∧ glasses (i + 1) = glasses (i + 2))
:= sorry

end petya_cannot_achieve_goal_l287_287441


namespace tangent_intersection_point_l287_287894

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287894


namespace balls_in_boxes_l287_287635

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
    balls = 5 ∧ boxes = 4 ∧ (∀ b, 1 ≤ b ∧ b ≤ balls) ∧ (∀ bx, 1 ≤ bx ∧ bx ≤ boxes)
    → (number_of_ways balls boxes (λ b bx, b ≠ 0 ∧ bx ≠ 0) = 480) :=
by
  sorry

end balls_in_boxes_l287_287635


namespace product_value_l287_287305

-- Define the term used in the product
def term (n : ℕ) : ℚ := (n * (n + 3)) / ((n + 1) * (n + 1))

-- Define the product of terms from 1 to 98
def product := ∏ n in (Finset.range 98).map (Finset.natAntidiagonal 1), term n

-- State the theorem to prove: the product is equal to 101 / 198
theorem product_value : product = 101 / 198 :=
by
  sorry

end product_value_l287_287305


namespace function_even_l287_287787

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x, sqrt 2 * sin (x - π / 4) - sin x

-- State the theorem to prove that f is an even function
theorem function_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end function_even_l287_287787


namespace half_angle_in_second_quadrant_l287_287647

theorem half_angle_in_second_quadrant 
  {θ : ℝ} (k : ℤ)
  (hθ_quadrant4 : 2 * k * Real.pi + (3 / 2) * Real.pi ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi)
  (hcos : abs (Real.cos (θ / 2)) = - Real.cos (θ / 2)) : 
  ∃ m : ℤ, (m * Real.pi + (Real.pi / 2) ≤ θ / 2 ∧ θ / 2 ≤ m * Real.pi + Real.pi) :=
sorry

end half_angle_in_second_quadrant_l287_287647


namespace derivative_at_2_l287_287377

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287377


namespace erased_number_l287_287475

theorem erased_number (a : ℤ) (b : ℤ) (h : -4 ≤ b ∧ b ≤ 4) (h_sum : 8 * a - b = 1703) : a + b = 214 := 
by 
  have ha : 1699 / 8 ≤ a, from sorry
  have hb : a ≤ 1707 / 8, from sorry
  have ha_int : a = 213, from sorry
  have hb_int : b = 1, by { rw [ha_int] at h_sum, linarith }
  exact sorry 

-- The proof steps are not provided here, as only the theorem statement is requested.

end erased_number_l287_287475


namespace scouts_attended_l287_287860

def chocolate_bar_cost : ℝ := 1.50
def total_spent : ℝ := 15
def sections_per_bar : ℕ := 3
def smores_per_scout : ℕ := 2

theorem scouts_attended (bars : ℝ) (sections : ℕ) (smores : ℕ) (scouts : ℕ) :
  bars = total_spent / chocolate_bar_cost →
  sections = bars * sections_per_bar →
  smores = sections →
  scouts = smores / smores_per_scout →
  scouts = 15 :=
by
  intro h1 h2 h3 h4
  sorry

end scouts_attended_l287_287860


namespace raise_percentage_to_original_l287_287393

-- Let original_salary be a variable representing the original salary.
-- Since the salary was reduced by 50%, the reduced_salary is half of the original_salary.
-- We need to prove that to get the reduced_salary back to the original_salary, 
-- it must be increased by 100%.

noncomputable def original_salary : ℝ := sorry
noncomputable def reduced_salary : ℝ := original_salary * 0.5

theorem raise_percentage_to_original :
  (original_salary - reduced_salary) / reduced_salary * 100 = 100 :=
sorry

end raise_percentage_to_original_l287_287393


namespace max_rectangles_is_5_l287_287260

-- Define the conditions of the problem as hypotheses.
def figure : Type := sorry -- Abstract representation of the figure
def is_unit_square (cell : figure) : Prop := sorry -- Predicate defining unit squares in the figure
def is_1x2_rectangle (rect : figure) : Prop := sorry -- Predicate defining 1x2 rectangles in the figure

-- Define the necessary hypotheses
axiom unit_squares_and_rectangles_only : ∀ (x : figure), is_unit_square x ∨ is_1x2_rectangle x
axiom rectangles_cover_adj_cells : ∀ (r : figure), is_1x2_rectangle r → r = sorry -- Abstract representation of rectangles covering two adjacent cells
axiom num_black_cells : ℕ := 5 -- Given number of black cells in the figure
axiom white_cells_equal_black_cells : ∃ (W : ℕ), W = num_black_cells -- Ensure equal or more white cells

-- Define what we need to prove
theorem max_rectangles_is_5 : ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), m ≤ n :=
by
  sorry

end max_rectangles_is_5_l287_287260


namespace ordering_exists_l287_287707

theorem ordering_exists (n : ℕ) (S : set ℕ) (h1 : n ≥ 3)
  (h2 : ∀ s ∈ S, ∀ s1 s2 ∈ S, s ≠ s1 + s2) :
  ∃ (a : fin n → ℕ), (∀ i : fin (n - 1), a ⟨i.1 + 1, sorry⟩ ∈ S) ∧
  (∀ i : fin (n - 2), ¬((a ⟨i.1 + 1, sorry⟩) ∣ (a ⟨i.1, sorry⟩ + a ⟨i.1 + 2, sorry⟩))) := sorry

end ordering_exists_l287_287707


namespace conference_room_arrangement_l287_287916

theorem conference_room_arrangement :
  let seats := 12
  let rocking_chairs := 8
  let stools := 4
  finset.card (finset.range seats).choose stools = 495 := 
by
  sorry

end conference_room_arrangement_l287_287916


namespace concentration_pn_qn_limit_concentration_pn_limit_concentration_qn_l287_287497

variable (a b p q m : ℕ)
variable (ha : a > m)
variable (hb : b > m)
variable (hpq : p ≠ q)

-- Define the concentration after n iterations
noncomputable def concentration_n (n : ℕ) : ℕ :=
  (a * p + b * q) / (a + b)

-- Define the statements to be proved
theorem concentration_pn_qn (n : ℕ) :
  ∀ (a b p q m : ℕ) (ha : a > m) (hb : b > m) (hpq : p ≠ q),
  concentration_n a b p q m n = (a * p + b * q) / (a + b) := sorry

theorem limit_concentration_pn :
  ∀ (a b p q m : ℕ) (ha : a > m) (hb : b > m) (hpq : p ≠ q),
  lim (λ n, concentration_n a b p q m n) (n → ∞) = (a * p + b * q) / (a + b) := sorry

theorem limit_concentration_qn :
  ∀ (a b p q m : ℕ) (ha : a > m) (hb : b > m) (hpq : p ≠ q),
  lim (λ n, concentration_n a b p q m n) (n → ∞) = (a * p + b * q) / (a + b) := sorry

end concentration_pn_qn_limit_concentration_pn_limit_concentration_qn_l287_287497


namespace sum_lent_is_1100_l287_287923

variables (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)

-- Given conditions
def interest_formula := I = P * r * t
def interest_difference := I = P - 572

-- Values
def rate := r = 0.06
def time := t = 8

theorem sum_lent_is_1100 : P = 1100 :=
by
  -- Definitions and axioms
  sorry

end sum_lent_is_1100_l287_287923


namespace angles_sum_to_two_pi_l287_287931

-- Define the angles in terms of a common factor
def angle_ratio_conditions (x : ℝ) : Prop :=
  let α := x in
  let β := 2 * x in
  let γ := 4 * x in
  let δ := 2 * x in
  α + β + γ + δ = 2 * real.pi

-- The statement we need to prove in Lean
theorem angles_sum_to_two_pi (x : ℝ) (h : angle_ratio_conditions x) :
  x = (2 * real.pi) / 9 :=
by
  sorry

end angles_sum_to_two_pi_l287_287931


namespace projectile_reaches_100_feet_first_time_l287_287783

noncomputable def height (t : ℝ) : ℝ := -16 * t^2 + 96 * t

theorem projectile_reaches_100_feet_first_time :
  ∃ t : ℝ, 0 < t ∧ height t = 100 ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → height t' ≠ 100) :=
sorry

end projectile_reaches_100_feet_first_time_l287_287783


namespace max_sum_of_prices_l287_287276

theorem max_sum_of_prices (R P : ℝ) 
  (h1 : 4 * R + 5 * P ≥ 27) 
  (h2 : 6 * R + 3 * P ≤ 27) : 
  3 * R + 4 * P ≤ 36 :=
by 
  sorry

end max_sum_of_prices_l287_287276


namespace gain_percent_correct_l287_287384

variables (CP SP : ℝ)

def gain (CP SP : ℝ) := SP - CP

def gainPercent (CP SP : ℝ) := (gain CP SP / CP) * 100

theorem gain_percent_correct (h1 : CP = 1000) (h2 : SP = 2000) :
  gainPercent CP SP = 100 :=
by
  sorry

end gain_percent_correct_l287_287384


namespace route_difference_is_18_l287_287215

theorem route_difference_is_18 :
  let t_uphill := 6 in
  let t_path1 := 2 * t_uphill in
  let t_final1 := (t_uphill + t_path1)/3 in
  let t_route1 := t_uphill + t_path1 + t_final1 in
  let t_flat := 14 in
  let t_final2 := 2 * t_flat in
  let t_route2 := t_flat + t_final2 in
  t_route2 - t_route1 = 18 :=
by
  let t_uphill := 6
  let t_path1 := 2 * t_uphill
  let t_final1 := (t_uphill + t_path1)/3
  let t_route1 := t_uphill + t_path1 + t_final1
  let t_flat := 14
  let t_final2 := 2 * t_flat
  let t_route2 := t_flat + t_final2
  have h : t_route2 - t_route1 = 18
  sorry

end route_difference_is_18_l287_287215


namespace domino_covering_l287_287235

theorem domino_covering (n m k : ℕ) : (∃ fn : Array (Fin n × Fin m) → bool, ∀ (i : Fin n) (j : Fin m), fn i j = fn (i + k) j ∨ fn i j = fn i (j + k)) ↔ (k ∣ n ∨ k ∣ m) :=
by
  sorry

end domino_covering_l287_287235


namespace ratio_of_segments_of_tangency_l287_287867

theorem ratio_of_segments_of_tangency
  {A B C L J K : Type} 
  (hA : Triangle A B C)
  (hB : TangentPoint L B C)
  (hC : TangentPoint J A C)
  (hD : TangentPoint K A B)
  (h1 : side_length A B = 17)
  (h2 : side_length A C = 13)
  (h3 : side_length B C = 8)
  (h4 : Segment r B C)
  (h5 : Segment s B C)
  (h6 : r + s = 8)
  (h7 : r < s) :
  r / s = 1 / 3 :=
sorry

end ratio_of_segments_of_tangency_l287_287867


namespace problem_example_l287_287201

theorem problem_example (a : ℕ) (H1 : a ∈ ({a, b, c} : Set ℕ)) (H2 : 0 ∈ ({x | x^2 ≠ 0} : Set ℕ)) :
  a ∈ ({a, b, c} : Set ℕ) ∧ 0 ∈ ({x | x^2 ≠ 0} : Set ℕ) :=
by
  sorry

end problem_example_l287_287201


namespace tangent_line_intersection_x_axis_l287_287901

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287901


namespace erased_number_l287_287477

theorem erased_number (a : ℤ) (b : ℤ) (h : -4 ≤ b ∧ b ≤ 4) (h_sum : 8 * a - b = 1703) : a + b = 214 := 
by 
  have ha : 1699 / 8 ≤ a, from sorry
  have hb : a ≤ 1707 / 8, from sorry
  have ha_int : a = 213, from sorry
  have hb_int : b = 1, by { rw [ha_int] at h_sum, linarith }
  exact sorry 

-- The proof steps are not provided here, as only the theorem statement is requested.

end erased_number_l287_287477


namespace area_PQR_is_3_l287_287288

-- The Line and Point Definitions
def P : (ℝ × ℝ) := (2, 1)
def Q : (ℝ × ℝ) := (1, 4)
def line_R (x y : ℝ) : Prop := x - y = 3

-- Formula to Calculate Area of Triangle
def triangle_area (p1 p2 p3 : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Statement to Prove
theorem area_PQR_is_3 (x y : ℝ) (h : line_R x y) : triangle_area P Q (x, y) = 3 := by
  -- Proof goes here
  sorry

end area_PQR_is_3_l287_287288


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287543

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287543


namespace prove_m_and_n_range_of_k_l287_287582

noncomputable def f (x : ℝ) (n : ℝ) (m : ℝ) := (n - 2^x) / (2^(x + 1) + m)

theorem prove_m_and_n (h_odd : ∀ x, f x 1 2 = -f (-x) 1 2) : 
  (∃ m n, m = 2 ∧ n = 1) :=
begin
  use [2, 1],
  simp,
end

theorem range_of_k (h_odd : ∀ x, f x 1 2 = -f (-x) 1 2) :
  (∀ x ∈ set.Icc (1/2 : ℝ) 3, f (λ k, k * x^2) 1 2 + f (2*x - 1) 1 2 > 0) → 
  (∀ k, k ∈ set.Iio (-1)) :=
sorry

end prove_m_and_n_range_of_k_l287_287582


namespace hypotenuse_length_l287_287790

theorem hypotenuse_length :
  ∀ (x : ℝ), 
  (3*x - 2)^2 + x^2 = 292 ↔ (1/2) * x * (3*x - 2) = 72 → 
  sqrt((3*x - 2)^2 + x^2) = sqrt(292) :=
begin
  intros x,
  sorry
end

end hypotenuse_length_l287_287790


namespace circle_segment_length_l287_287967

noncomputable def radius (C_circumference: ℝ) : ℝ :=
  C_circumference / (2 * Real.pi)

noncomputable def length_CB 
  (C_circumference: ℝ) 
  (angle_ACB: ℝ) : ℝ :=
let r := radius C_circumference in
  let AC := r in
  let AB := r in
  Real.sqrt (AC ^ 2 + AB ^ 2 - 2 * AC * AB * Real.cos angle_ACB)

theorem circle_segment_length 
  (C_circumference: ℝ)
  (h1: C_circumference = 18 * Real.pi)
  (angle_ACB: ℝ)
  (h2: angle_ACB = Real.pi / 6) :
  length_CB C_circumference angle_ACB = 9 * Real.sqrt (2 - Real.sqrt 3) :=
by
  sorry

end circle_segment_length_l287_287967


namespace frequency_of_fourth_group_l287_287505

theorem frequency_of_fourth_group (f₁ f₂ f₃ f₄ f₅ f₆ : ℝ) (h1 : f₁ + f₂ + f₃ = 0.65) (h2 : f₅ + f₆ = 0.32) (h3 : f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = 1) :
  f₄ = 0.03 :=
by 
  sorry

end frequency_of_fourth_group_l287_287505


namespace incenter_distance_sum_eq_side_product_l287_287718

theorem incenter_distance_sum_eq_side_product 
  (a b c l m n : ℝ) 
  (I : Type*) 
  (incenter : I)
  (ABC : Type*) 
  (triangle : ABC) 
  (side_lengths : triangle → ℝ := λ T, if T = BC then a else if T = AC then b else c) 
  (distances_from_incenter : I → ABC → ℝ := λ I T, if T = A then l else if T = B then m else n) 
  : a * l^2 + b * m^2 + c * n^2 = a * b * c :=
sorry

end incenter_distance_sum_eq_side_product_l287_287718


namespace quadrilateral_tessellation_exists_tessellating_pentagon_no_parallel_sided_pentagon_tessellation_l287_287846

-- Part (a)
theorem quadrilateral_tessellation (A B C D : Type) [quadrilateral A B C D] : 
  ∃ tessellation plane (quad : Type), quadrilateral quad ∧ congruent quad A B C D := sorry

-- Part (b)
theorem exists_tessellating_pentagon : 
  ∃ pentagon : Type, ∃ tessellation plane (p : Type), pentagon p := sorry

-- Part (c)
theorem no_parallel_sided_pentagon_tessellation : 
  ∃ pentagon : Type, (no_parallel_sides pentagon) → ∃ tessellation plane (p : Type), pentagon p := sorry

end quadrilateral_tessellation_exists_tessellating_pentagon_no_parallel_sided_pentagon_tessellation_l287_287846


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287533

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287533


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287907

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287907


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287532

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287532


namespace find_crayons_in_pack_l287_287457

variables (crayons_in_locker : ℕ) (crayons_given_by_bobby : ℕ) (crayons_given_to_mary : ℕ) (crayons_final_count : ℕ) (crayons_in_pack : ℕ)

-- Definitions from the conditions
def initial_crayons := 36
def bobby_gave := initial_crayons / 2
def mary_crayons := 25
def final_crayons := initial_crayons + bobby_gave - mary_crayons

-- The theorem to prove
theorem find_crayons_in_pack : initial_crayons = 36 ∧ bobby_gave = 18 ∧ mary_crayons = 25 ∧ final_crayons = 29 → crayons_in_pack = 29 :=
by
  sorry

end find_crayons_in_pack_l287_287457


namespace five_cubes_face_sharing_possible_six_cubes_face_sharing_possible_l287_287386

-- Define the problem statement and proof for five cubes
theorem five_cubes_face_sharing_possible :
  ∃ (cubes : Fin 5 → Cube), (∀ i j, i ≠ j → shares_face (cubes i) (cubes j)) :=
sorry

-- Define the problem statement and proof for six cubes
theorem six_cubes_face_sharing_possible :
  ∃ (cubes : Fin 6 → Cube), (∀ i j, i ≠ j → shares_face (cubes i) (cubes j)) :=
sorry

end five_cubes_face_sharing_possible_six_cubes_face_sharing_possible_l287_287386


namespace polynomial_must_be_196_l287_287815

-- Define the polynomial P(x) with the additional constant m
def P (x m : ℝ) : ℝ := (x - 1)*(x + 3)*(x - 4)*(x - 8) + m

-- Statement to prove that for P(x) to be a perfect square polynomial, m must be 196
theorem polynomial_must_be_196 (m : ℝ) : 
  (∀ x : ℝ, ∃ g : ℝ → ℝ, (P x m = (g x) ^ 2)) ↔ (m = 196) :=
begin
  sorry
end

end polynomial_must_be_196_l287_287815


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287537

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287537


namespace correct_option_l287_287838

noncomputable def option_A (x : ℝ) : Prop :=
  deriv (λ x : ℝ, 1 / (Real.log x)) = x

noncomputable def option_B (x : ℝ) : Prop :=
  deriv (λ x : ℝ, x * Real.exp x) = Real.exp x + 1

noncomputable def option_C (x : ℝ) : Prop :=
  deriv (λ x : ℝ, x^2 * Real.cos x) = -2 * x * Real.sin x

noncomputable def option_D (x : ℝ) : Prop :=
  deriv (λ x : ℝ, x - (1 / x)) = 1 + (1 / x^2)

theorem correct_option (x : ℝ) : option_D x ∧ ¬option_A x ∧ ¬option_B x ∧ ¬option_C x :=
by 
  -- The proof should go here
  sorry

end correct_option_l287_287838


namespace isosceles_triangle_perimeter_15_or_18_l287_287803

/-- Definition of isosceles triangle sides --/
def isosceles_triangle (a b : ℝ) :=
  ∃ c : ℝ, (a = c ∨ b = c) ∧ (a = c ∨ b = c) ∧ c ≠ a ∧ c ≠ b ∧ triangle_inequality a b c

/-- Definition of triangle inequality --/
def triangle_inequality (a b c : ℝ) :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Perimeter calculation --/
def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter_15_or_18 :
  (isosceles_triangle 4 7) → (perimeter 4 4 7 = 15 ∨ perimeter 4 7 7 = 18) :=
by
  intros h
  cases h with c hc
  sorry

end isosceles_triangle_perimeter_15_or_18_l287_287803


namespace find_10_digit_number_with_digit_frequencies_l287_287063

/--
Proof problem: Find a 10-digit number \( a_0 a_1 \ldots a_9 \) such that \( a_0 + a_1 + \ldots + a_9 = 10 \)
and for each \( k \), \( a_k \) is the number of times that the digit \( k \) appears in the number.
--/
theorem find_10_digit_number_with_digit_frequencies :
  ∃ (a : Fin 10 → Fin 10), (∀ k, a k = (Nat.digits 10 (Nat.ofDigits 10 (List.ofFn a))).count k) ∧ (∑ k, a k = 10) :=
by
  sorry

end find_10_digit_number_with_digit_frequencies_l287_287063


namespace pipe_A_fill_time_l287_287935

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end pipe_A_fill_time_l287_287935


namespace sequence_300th_term_eq_358_l287_287828

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def isMultipleOf10 (n : ℕ) : Prop := n % 10 = 0

def isOmitted (n : ℕ) : Prop := isPerfectSquare n ∨ isMultipleOf10 n

def sequenceTerm (n : ℕ) : ℕ :=
  let seq := (List.filter (λ x, ¬ isOmitted x) (List.range (2 * n)))
  seq.getD n 0

theorem sequence_300th_term_eq_358 : sequenceTerm 299 = 358 := by
  sorry

end sequence_300th_term_eq_358_l287_287828


namespace percent_defective_units_shipped_l287_287181

theorem percent_defective_units_shipped :
  let total_units_defective := 6 / 100
  let defective_units_shipped := 4 / 100
  let percent_defective_units_shipped := (total_units_defective * defective_units_shipped) * 100
  percent_defective_units_shipped = 0.24 := by
  sorry

end percent_defective_units_shipped_l287_287181


namespace monthly_phone_contract_cost_l287_287730

def cost_of_iphone : ℕ := 1000
def cost_case (c_phone : ℕ) : ℕ := (20 * c_phone) / 100
def cost_headphones (c_case : ℕ) : ℕ := c_case / 2
def total_first_year_spend : ℕ := 3700
def num_months_in_year : ℕ := 12

theorem monthly_phone_contract_cost :
  let c_case := cost_case cost_of_iphone,
      c_headphones := cost_headphones c_case,
      total_cost_items := cost_of_iphone + c_case + c_headphones,
      total_cost_contract_year := total_first_year_spend - total_cost_items,
      monthly_cost := total_cost_contract_year / num_months_in_year
  in monthly_cost = 200 := by
  sorry

end monthly_phone_contract_cost_l287_287730


namespace down_payment_calculation_l287_287229

theorem down_payment_calculation 
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (n : ℕ)
  (interest_rate : ℝ)
  (down_payment : ℝ) :
  purchase_price = 127 ∧ 
  monthly_payment = 10 ∧ 
  n = 12 ∧ 
  interest_rate = 0.2126 ∧
  down_payment + (n * monthly_payment) = purchase_price * (1 + interest_rate) 
  → down_payment = 34 := 
sorry

end down_payment_calculation_l287_287229


namespace derivative_at_2_l287_287355

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287355


namespace primes_between_70_and_100_l287_287634

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_prime (n : ℕ) : Prop :=
  if n <= 1 then false 
  else ∀ (m : ℕ), m > 1 ∧ m < n → n % m ≠ 0

theorem primes_between_70_and_100 :
  (count (λ p, 70 ≤ p ∧ p ≤ 100 ∧ is_prime p ∧ (p / 2 < 50))
    [71, 73, 79, 83, 89, 97]) = 6 :=
by
  sorry

end primes_between_70_and_100_l287_287634


namespace correct_mark_l287_287008

theorem correct_mark (x : ℕ) (S_Correct S_Wrong : ℕ) (n : ℕ) :
  n = 26 →
  S_Wrong = S_Correct + (83 - x) →
  (S_Wrong : ℚ) / n = (S_Correct : ℚ) / n + 1 / 2 →
  x = 70 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l287_287008


namespace second_pedestrian_speed_l287_287294

theorem second_pedestrian_speed 
  (S₀ : ℝ) (v₁ : ℝ) (t : ℝ) (S : ℝ) 
  (h₁ : S₀ = 100) (h₂ : v₁ = 8) (h₃ : t = 300) (h₄ : S = 50) :
  ∃ v₂ : ℝ, v₂ = 7.50 ∨ v₂ ≈ 7.83 :=
by
  sorry

end second_pedestrian_speed_l287_287294


namespace ordered_pair_sqrt_problem_l287_287530

theorem ordered_pair_sqrt_problem
  (a b : ℕ) 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a < b)
  (h4 : (1 + √(45 + 20 * √5)).sqrt = a.sqrt + b.sqrt) :
  (a, b) = (1, 5) :=
  sorry

end ordered_pair_sqrt_problem_l287_287530


namespace closest_integer_to_cube_root_of_sum_l287_287299

theorem closest_integer_to_cube_root_of_sum 
  (a : ℕ) (b : ℕ) (c : ℕ) (h1 : a = 343) (h2 : b = 729) (h3 : c = 10) :
  Int.closest (Real.cbrt (a + b + c)) = 10 :=
by
  sorry

end closest_integer_to_cube_root_of_sum_l287_287299


namespace quadratic_inequality_solution_l287_287206

theorem quadratic_inequality_solution (x a : ℝ) (h : x ∈ Iio 1 ∪ Ioi 5) :
    (x^2 - 2*(a-2)*x + a > 0) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end quadratic_inequality_solution_l287_287206


namespace num_sets_A_l287_287585

theorem num_sets_A : 
  (finset.filter (λ A, ({1} ⊆ A ∧ A ⊆ {1,2,3,4})) (finset.powerset {1,2,3,4})).card = 7 :=
by
  sorry

end num_sets_A_l287_287585


namespace parabola_eq_l287_287807

theorem parabola_eq (F : ℝ × ℝ) (hF : F = (0, -1)) : ∃ (a : ℝ), (a = 1 ∧ x^2 = -4 * y) :=
by 
  have h : F = (0, -1) := hF
  use 1
  split
  { refl },
  sorry

end parabola_eq_l287_287807


namespace shooter_prob_3_hits_in_4_shooter_prob_at_least_1_hit_in_4_l287_287929

open ProbabilityTheory

def prob_hit_target_exactly_3_times_4_shots : ℚ := 1 / 4
def prob_hit_target_at_least_once_4_shots : ℚ := 15 / 16

theorem shooter_prob_3_hits_in_4 :
  let p := 0.5 in
  binomial 4 3 * (p ^ 3) * ((1 - p) ^ (4 - 3)) = prob_hit_target_exactly_3_times_4_shots :=
by
  sorry

theorem shooter_prob_at_least_1_hit_in_4 :
  let p := 0.5 in
  1 - (p ^ 4) = prob_hit_target_at_least_once_4_shots :=
by
  sorry

end shooter_prob_3_hits_in_4_shooter_prob_at_least_1_hit_in_4_l287_287929


namespace cos_theta_range_l287_287263

theorem cos_theta_range:
  ∀ (θ : ℝ), (0 < θ) → (θ ≤ π/2) → 
  (∀ (φ : ℝ), (0 < φ) → (φ < π/2) → 
    (∀ (x : ℝ), (-π / 4 ≤ x) → (x ≤ θ) → 
      (let f := (λ x : ℝ, 2 * sin (x + φ)) in 
        (f x ≥ -√3) ∧ (f x ≤ 2))) → 
    symmetric_about (λ x : ℝ, 2 * sin (x + φ)) π/6) → 
  (π/6 ≤ θ ∧ θ ≤ 7 * π / 12) → 
  (let a := (λ θ : ℝ, √2 - √6 / 4) in 
   let b := (λ θ : ℝ, √3 / 2) in 
   (a θ ≤ cos θ ∧ cos θ ≤ b θ)) :=
sorry

end cos_theta_range_l287_287263


namespace restaurant_equidistant_from_barracks_l287_287941

theorem restaurant_equidistant_from_barracks
    (distance_G_from_road : ℝ)
    (distance_B_from_G_on_road : ℝ)
    (equidistant_point : ℝ)
    (h_distance_G : distance_G_from_road = 300)
    (h_distance_B : distance_B_from_G_on_road = 500) :
    equidistant_point = 200 :=
by
  -- Problem conditions:
  have OG_eq : distance_G_from_road = 300 := h_distance_G
  have OB_eq : distance_B_from_G_on_road = 500 := h_distance_B

  -- Calculations:
  let x := distance_G_from_road
  let y := distance_B_from_G_on_road

  -- Apply Pythagorean theorem:
  have G_to_C : ℝ := sqrt(equidistant_point^2 + x^2)
  have B_from_O : ℝ := sqrt(y^2 - x^2)
  have B_to_C : ℝ := sqrt((y - equidistant_point)^2 + x^2)

  -- Set up the equation G_to_C == B_to_C:
  have : G_to_C = B_to_C
  sorry

-- state the final goal:
#check restaurant_equidistant_from_barracks

end restaurant_equidistant_from_barracks_l287_287941


namespace simplify_expression_l287_287249

theorem simplify_expression (c : ℤ) : (3 * c + 6 - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l287_287249


namespace unfriendly_subset_count_l287_287023

theorem unfriendly_subset_count (n k : ℕ) : 
  ∃ (C : ℕ → ℕ → ℕ), 
    C (n - k + 1) k = nat.choose (n - k + 1) k ∧
    (∀ S ⊆ finset.range n, finset.card S = k → 
      (∀ {i j : ℕ}, i ∈ S → j ∈ S → i ≠ j → abs (i - j) ≥ 2)) → 
    C (n - k + 1) k = nat.choose (n - k + 1) k := sorry

end unfriendly_subset_count_l287_287023


namespace find_f_prime_at_two_l287_287327

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287327


namespace minimum_value_of_m_n_l287_287593

theorem minimum_value_of_m_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + b) = 1) :
  ∃ (m n : ℝ), m = a + 1/a ∧ n = b + 1/b ∧ m + n = 5 :=
begin
  let m := a + 1/a,
  let n := b + 1/b,
  have h4 : m + n = a + b + 1/a + 1/b := by ring,
  sorry
end

end minimum_value_of_m_n_l287_287593


namespace find_d_l287_287715

theorem find_d (a b c d : ℤ) (h_poly : ∃ s1 s2 s3 s4 : ℤ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧ 
  ( ∀ x, (Polynomial.eval x (Polynomial.C d + Polynomial.X * Polynomial.C c + Polynomial.X^2 * Polynomial.C b + Polynomial.X^3 * Polynomial.C a + Polynomial.X^4)) =
    (x + s1) * (x + s2) * (x + s3) * (x + s4) ) ) 
  (h_sum : a + b + c + d = 2013) : d = 0 :=
by
  sorry

end find_d_l287_287715


namespace length_of_second_train_l287_287437

theorem length_of_second_train :
  let l1 := 200 -- Length of the first train (in meters)
  let v1 := 72 / 3.6 -- Speed of the first train (in m/s)
  let v2 := 36 / 3.6 -- Speed of the second train (in m/s)
  let t := 49.9960003199744 -- Time to cross each other (in seconds)
  let relative_speed := v1 - v2
  let total_distance := relative_speed * t
  let l2 := total_distance - l1 -- Length of the second train (in meters)
  in l2 = 299.960003199744 :=
by
  sorry

end length_of_second_train_l287_287437


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287559

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287559


namespace smallest_non_palindrome_product_l287_287547

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287547


namespace small_palindrome_check_l287_287563

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287563


namespace tangent_circles_x_intersect_l287_287873

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287873


namespace find_f_prime_at_2_l287_287313

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287313


namespace min_trips_required_l287_287397

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l287_287397


namespace sum_of_exponents_divisors_l287_287244

theorem sum_of_exponents_divisors (n : ℕ) (p : ℕ → ℕ) (α : ℕ → ℕ) (N : ℕ) 
  (hN : N = ∏ i in finset.range n, p i ^ α i)
  (hp_prime : ∀ i, nat.prime (p i)):
  ∑ d in (nat.divisors N), ∑ i in finset.range n, i = 
  1/2 * (∏ i in finset.range n, (α i + 1)) * (∑ i in finset.range n, α i) :=
sorry

end sum_of_exponents_divisors_l287_287244


namespace product_of_tangents_l287_287710

noncomputable def S : set (ℕ × ℕ) := { p | p.1 ∈ ({0, 1, 2, 3, 4, 5} : set ℕ) ∧ p.2 ∈ ({0, 1, 2, 3, 4, 5} : set ℕ) ∧ p ≠ (0, 0) }
noncomputable def is_right_triangle (A B C : ℕ × ℕ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

noncomputable def T : set ((ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) :=
  { t | t.1.1 ∈ S ∧ t.1.2 ∈ S ∧ t.2 ∈ S ∧ is_right_triangle t.1.1 t.1.2 t.2 }

noncomputable def f (t : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) : ℝ :=
  real.tan (real.arctan2 (t.1.1.2 - t.2.2) (t.1.1.1 - t.2.1) —
            real.arctan2 (t.1.2.2 - t.2.2) (t.1.2.1 - t.2.1))

theorem product_of_tangents :
  ∏ t in T, f(t) = 1 :=
sorry

end product_of_tangents_l287_287710


namespace smallest_pos_int_gcd_gt_one_l287_287832

theorem smallest_pos_int_gcd_gt_one : ∃ n: ℕ, n > 0 ∧ (Nat.gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 121 :=
by
  sorry

end smallest_pos_int_gcd_gt_one_l287_287832


namespace confidence_association_l287_287668

-- Definitions of variables
variable (X Y : Type)
noncomputable def H₀ : Prop := ¬ (X ∈ Y)
noncomputable def p_value : ℝ := 0.010
noncomputable def test_statistic : ℝ := 6.635

-- Main theorem statement
theorem confidence_association :
  H₀ → p_value ≤ 0.010 → test_statistic ≥ 6.635 → (confidence_level : ℝ) > 0.99 := 
by 
  sorry

end confidence_association_l287_287668


namespace part1_sinx_over_x_part2_harmonic_interval_l287_287734

-- Part 1: Prove that for x in (0, π/2), sin(x)/x > 1/2
theorem part1_sinx_over_x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
  sin x / x > 1 / 2 := by sorry

-- Part 2: Find all harmonic intervals [a, b] for f(x) = -2sin(x), where the domain and range of f(x) are both [a, b]
theorem part2_harmonic_interval :
  ∃ (a b : ℝ), (∀ x, a ≤ x ∧ x ≤ b → -2 * sin x ≥ a ∧ -2 * sin x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ -2 * sin x = y) ∧
  a < b ∧ (∃ a, ∃ b, a = -2 ∧ b = 2 ∨ ¬ ∃ (a ≠ b, ∀ c d : ℝ, c ≠ d → (c, d) ≠ (-2, 2)) ) := by 
    sorry

end part1_sinx_over_x_part2_harmonic_interval_l287_287734


namespace tangent_circles_x_intersect_l287_287869

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287869


namespace find_f_prime_at_2_l287_287347

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287347


namespace tangent_intersection_point_l287_287889

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287889


namespace f_increasing_solve_inequality_l287_287132

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

-- Lean statement for proving the function is increasing on the interval (-1, 1)
theorem f_increasing : ∀ x1 x2 : ℝ, x1 ∈ Ioo (-1 : ℝ) 1 → x2 ∈ Ioo (-1) 1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

-- Lean statement for solving the inequality f(x-1) + f(x) < 0
theorem solve_inequality (x : ℝ) : x ∈ Ioo (0 : ℝ) (1/2) → f (x - 1) + f x < 0 :=
by
  sorry

end f_increasing_solve_inequality_l287_287132


namespace largest_S_size_l287_287192

-- Define the set T according to the problem statement
def T : Finset ℕ := 
  (Finset.Icc 1 6).bUnion (λ a, (Finset.Icc (a+1) 6).image (λ b, 10 * a + b))

-- Define a predicate to check if a set S of two-digit numbers contains all the digits 1 through 6
def contains_all_digits (S : Finset ℕ) : Prop :=
  ∀ d ∈ Finset.range 7 \ {0}, ∃ x ∈ S, d = x / 10 ∨ d = x % 10

-- Define a predicate to check if a set S does not contain any three numbers using all six digits
def no_three_elements_together_use_all_digits (S : Finset ℕ) : Prop :=
  ∀ s1 s2 s3 ∈ S, (s1 / 10 ∪ s1 % 10 ∪ s2 / 10 ∪ s2 % 10 ∪ s3 / 10 ∪ s3 % 10) ≠ Finset.range 7 \ {0}

-- The proof problem statement
theorem largest_S_size :
  ∃ S ⊆ T, contains_all_digits S ∧ no_three_elements_together_use_all_digits S ∧ S.card = 9 :=
sorry

end largest_S_size_l287_287192


namespace ratio_area_II_to_III_l287_287755

-- Define the properties of the squares as given in the conditions
def perimeter_region_I : ℕ := 16
def perimeter_region_II : ℕ := 32
def side_length_region_I := perimeter_region_I / 4
def side_length_region_II := perimeter_region_II / 4
def side_length_region_III := 2 * side_length_region_II
def area_region_II := side_length_region_II ^ 2
def area_region_III := side_length_region_III ^ 2

-- Prove that the ratio of the area of region II to the area of region III is 1/4
theorem ratio_area_II_to_III : (area_region_II : ℚ) / (area_region_III : ℚ) = 1 / 4 := 
by sorry

end ratio_area_II_to_III_l287_287755


namespace gcd_polynomial_eq_one_l287_287598

theorem gcd_polynomial_eq_one (b : ℤ) (hb : Even b) (hmb : 431 ∣ b) : 
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end gcd_polynomial_eq_one_l287_287598


namespace additional_kgs_l287_287426

variables (P R A : ℝ)
variables (h1 : R = 0.80 * P) (h2 : R = 34.2) (h3 : 684 = A * R)

theorem additional_kgs :
  A = 20 :=
by
  sorry

end additional_kgs_l287_287426


namespace range_of_g_l287_287503

def g (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.cos x) ^ 4

theorem range_of_g : set.range g = set.Icc (3 / 4) 1 :=
by
  sorry

end range_of_g_l287_287503


namespace derivative_at_2_l287_287365

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287365


namespace train_speed_l287_287939

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) :
  distance_meters = 180 →
  time_seconds = 17.998560115190784 →
  ((distance_meters / 1000) / (time_seconds / 3600)) = 36.00360072014403 :=
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end train_speed_l287_287939


namespace mark_total_cans_l287_287220

theorem mark_total_cans (p1 p2 p3 p4 p5 p6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ)
  (h1 : p1 = 30) (h2 : p2 = 25) (h3 : p3 = 35) (h4 : p4 = 40) 
  (h5 : p5 = 28) (h6 : p6 = 32) (hc1 : c1 = 12) (hc2 : c2 = 10) 
  (hc3 : c3 = 15) (hc4 : c4 = 14) (hc5 : c5 = 11) (hc6 : c6 = 13) :
  p1 * c1 + p2 * c2 + p3 * c3 + p4 * c4 + p5 * c5 + p6 * c6 = 2419 := 
by 
  sorry

end mark_total_cans_l287_287220


namespace repeating_decimal_example_l287_287407

theorem repeating_decimal_example : 
  ∀ (d : ℝ), d = 4.3535 → 
    repeating_decimal d ∧ 
    repeating_abbreviation d "4.\overline{35}" ∧ 
    repeating_cycle d 35 :=
by
  assume d h,
  sorry

end repeating_decimal_example_l287_287407


namespace line_through_point_has_given_equation_l287_287784

def line_equation (m x1 y1 : ℝ) : ℝ → ℝ := 
  fun x => y1 + m * (x - x1)

theorem line_through_point_has_given_equation :
  ∃ (m x1 y1 : ℝ), m = 2 ∧ x1 = 0 ∧ y1 = 3 ∧ 
  ∀ (x y : ℝ), y = line_equation m x1 y1 x ↔ 2 * x - y + 3 = 0 :=
by {
  let m := 2,
  let x1 := 0,
  let y1 := 3,
  use [m, x1, y1],
  split, 
  { refl },
  split,
  { refl },
  split,
  { refl },
  intros x y,
  split,
  { intro hxy,
    rw [line_equation, hxy],
    linarith },
  { intro hxy,
    simp [line_equation],
    linarith }
}

end line_through_point_has_given_equation_l287_287784


namespace count_integers_satisfying_conditions_l287_287143

theorem count_integers_satisfying_conditions :
  {n : ℤ | 200 < n ∧ n < 300 ∧ ∃ r : ℤ, 0 ≤ r ∧ r < 7 ∧ (n ≡ r [MOD 7]) ∧ (n ≡ r [MOD 9])}.finite.toFinset.card = 7 :=
by
  sorry

end count_integers_satisfying_conditions_l287_287143


namespace least_two_multiples_of_15_gt_450_l287_287298

-- Define a constant for the base multiple
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a constant for being greater than 450
def is_greater_than_450 (n : ℕ) : Prop :=
  n > 450

-- Two least positive multiples of 15 greater than 450
theorem least_two_multiples_of_15_gt_450 :
  (is_multiple_of_15 465 ∧ is_greater_than_450 465 ∧
   is_multiple_of_15 480 ∧ is_greater_than_450 480) :=
by
  sorry

end least_two_multiples_of_15_gt_450_l287_287298


namespace probability_point_satisfies_inequality_l287_287926

/-- 
  A rectangle with vertices at points K(-1,0), L(-1,5), M(2,5), and N(2,0). 
  We want to prove that the probability that a randomly thrown point (x, y) 
  inside this rectangle satisfies the inequality x² + 1 ≤ y ≤ x + 3 is 0.3.
-/
theorem probability_point_satisfies_inequality :
  let K := (-1, 0),
      L := (-1, 5),
      M := (2, 5),
      N := (2, 0),
      area_rectangle := 3 * 5,
      area_region := (29 / 6)
  in area_region / area_rectangle = 0.3 :=
by
  sorry

end probability_point_satisfies_inequality_l287_287926


namespace probability_same_color_l287_287146

-- Define the number of plates of each color
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_green_plates : ℕ := 3

-- Total number of plates
def total_plates := num_red_plates + num_blue_plates + num_green_plates

-- Function to calculate the number of ways to choose 2 out of n elements
def choose2 (n : ℕ) := n.choose 2

-- Number of ways to choose 2 red plates
def red_pairs := choose2 num_red_plates

-- Number of ways to choose 2 blue plates
def blue_pairs := choose2 num_blue_plates

-- Number of ways to choose 2 green plates
def green_pairs := choose2 num_green_plates

-- Total number of ways to choose any 2 plates
def total_pairs := choose2 total_plates

-- Total number of same-color pairs
def same_color_pairs := red_pairs + blue_pairs + green_pairs

-- Probability that the two plates selected are of the same color
def same_color_probability := (same_color_pairs : ℚ) / total_pairs

-- The theorem to prove
theorem probability_same_color : same_color_probability = 34/105 := sorry

end probability_same_color_l287_287146


namespace triangle_expression_l287_287655

open Real

variable (D E F : ℝ)
variable (DE DF EF : ℝ)

-- conditions
def triangleDEF : Prop := DE = 7 ∧ DF = 9 ∧ EF = 8

theorem triangle_expression (h : triangleDEF DE DF EF) :
  (cos ((D - E)/2) / sin (F/2) - sin ((D - E)/2) / cos (F/2)) = 81/28 :=
by
  have h1 : DE = 7 := h.1
  have h2 : DF = 9 := h.2.1
  have h3 : EF = 8 := h.2.2
  sorry

end triangle_expression_l287_287655


namespace derivative_at_2_l287_287370

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287370


namespace math_problem_l287_287778

theorem math_problem (a b c : ℝ) (h1 : (a + b) / 2 = 30) (h2 : (b + c) / 2 = 60) (h3 : c - a = 60) : c - a = 60 :=
by
  -- Insert proof steps here
  sorry

end math_problem_l287_287778


namespace find_PA_times_PB_l287_287609

-- Define the given circle and its properties
def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 9
def midpoint_AB : ℝ × ℝ := (1, 1)
def chord_intersects_x_axis_at : ℝ × ℝ := (2, 0)

-- Statement of the problem
theorem find_PA_times_PB :
  let P := (2:ℝ, 0:ℝ),
      A := (sqrt(8) + sqrt(2), sqrt(2) - sqrt(2)),  -- Coordinates can be deduced based on circle and line eq
      B := (-sqrt(8) + sqrt(2), sqrt(2) - sqrt(2))  -- Coordinates can be deduced based on circle and line eq
  in |P.1 - A.1| * |P.1 - B.1| = 7 :=
by sorry

end find_PA_times_PB_l287_287609


namespace find_f_prime_at_2_l287_287307

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287307


namespace max_value_frac_l287_287770

theorem max_value_frac (a b c d : ℕ) (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hd : d < 10) :
  (∃ a b c d, 0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 10 ∧ ∀ x y z w, 0 < x < y < z < w < 10 -> (a, b, c, d) = (x, y, z, w))
  → ∃ a b c d, 0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 10 ∧ ∀ x y z w, 0 < x < y < z < w < 10 → (x = 1 ∧ y = 7 ∧ z = 8 ∧ w = 9) →
   max_value_frac = ((∃ a b c d, 6: ℕ)) sorry

end max_value_frac_l287_287770


namespace tangent_intersection_point_l287_287891

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287891


namespace polynomial_F_eq_180_l287_287195

noncomputable def F (x : ℝ) : ℝ := sorry -- Polynomial definition to be found

theorem polynomial_F_eq_180 :
  (∀ x : ℝ, (x + 4) ≠ 0 ∧ (x^2 + 8 * x + 16) ≠ 0 → 
  (F(4 * x) / F(x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16))) 
  ∧ (F 8 = 21) →
  F 16 = 180 := sorry

end polynomial_F_eq_180_l287_287195


namespace find_smaller_number_l287_287277

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : y = 28.5 :=
by
  sorry

end find_smaller_number_l287_287277


namespace running_time_and_speed_l287_287421

-- Define the given conditions
def swimming_distance : ℝ := 1
def cycling_distance : ℝ := 25
def running_distance : ℝ := 4
def total_competition_time : ℝ := 5 / 4

def swimming_time_test : ℝ := 1 / 16
def cycling_time_test : ℝ := 1 / 49
def running_time_test : ℝ := 1 / 49
def total_test_distance : ℝ := 5 / 4

-- Define the problem to be proved
theorem running_time_and_speed :
  ∃ v_1 v_2 v_3 : ℝ,
    (1 / v_1 + 25 / v_2 + 4 / v_3 = 5 / 4) ∧
    (swimming_time_test * v_1 + cycling_time_test * v_2 + running_time_test * v_3 = total_test_distance) ∧
    ((running_distance / v_3) = 2 / 7) ∧
    (v_3 = 14) :=
begin
  sorry
end

end running_time_and_speed_l287_287421


namespace equivalent_proposition_l287_287800

variable (M : Set α) (m n : α)

theorem equivalent_proposition :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end equivalent_proposition_l287_287800


namespace derivative_at_2_l287_287352

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287352


namespace distance_to_supermarket_l287_287957

-- Define the conditions
constants (D : ℝ) -- distance to the supermarket in miles
constants (farm_distance : ℝ := 6) -- distance to the farm in miles
constants (initial_gas : ℝ := 12) -- initial gas in gallons
constants (remaining_gas : ℝ := 2) -- remaining gas in gallons
constants (gas_consumption_rate : ℝ := 2) -- consumption rate in miles per gallon

-- Define the total distance traveled
def total_distance : ℝ := (initial_gas - remaining_gas) * gas_consumption_rate

-- Define the distance driven to the farm
def farm_trips_distance : ℝ := (2 + 2) + 6

-- Define the distance for the trip to the supermarket and back
def supermarket_trip_distance : ℝ := total_distance - farm_trips_distance

-- The theorem to prove where D is the distance to the supermarket
theorem distance_to_supermarket : D = supermarket_trip_distance / 2 :=
sorry

end distance_to_supermarket_l287_287957


namespace exists_seq_unbounded_bounded_no_seq_unbounded_bounded_with_inverse_b_l287_287495

noncomputable def seq_unbounded_bounded (a b : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, n ≤ m → a n ≥ a m ∧ b n ≥ b m) ∧
  (∀ n, ∃ m, m ≥ n ∧ (a 1) + (a 2) + ... + (a m) ≥ m) ∧
  (∀ n, ∃ m, m ≥ n ∧ (b 1) + (b 2) + ... + (b m) ≥ m) ∧
  (∀ n, (c 1) + (c 2) + ... + (c n) < N) -- for some large N
where
  c := λ i, min (a i) (b i)

noncomputable def seq_unbounded_bounded_with_inverse_b (a : ℕ → ℝ) : Prop :=
  let b := λ i, 1 / i in
  (∀ n m : ℕ, n ≤ m → a n ≥ a m ∧ b n ≥ b m) ∧
  (∀ n, ∃ m, m ≥ n ∧ (a 1) + (a 2) + ... + (a m) ≥ m) ∧
  (∀ n, ∃ m, m ≥ n ∧ (b 1) + (b 2) + ... + (b m) ≥ m) ∧
  (¬ ∃ N, ∀ n, (a 1) + (a 2) + ... + (c n) < N) -- contradiction statement
where
  c := λ i, min (a i) (b i)

theorem exists_seq_unbounded_bounded : ∃ a b : ℕ → ℝ, seq_unbounded_bounded a b :=
sorry 

theorem no_seq_unbounded_bounded_with_inverse_b (a : ℕ → ℝ) : ¬ seq_unbounded_bounded_with_inverse_b a :=
sorry

end exists_seq_unbounded_bounded_no_seq_unbounded_bounded_with_inverse_b_l287_287495


namespace derivative_at_2_l287_287353

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287353


namespace eval_expression_eq_neg431_l287_287510

theorem eval_expression_eq_neg431 (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^b - (b^a)^a) = -431 := by
  sorry

end eval_expression_eq_neg431_l287_287510


namespace num_triples_l287_287529

theorem num_triples (a b c : ℕ) :
  number_of_triples (λ a b c, gcd a b c = 35 ∧ lcm a b c = 5^18 * 7^16) = 9180 :=
sorry

end num_triples_l287_287529


namespace number_of_months_to_fully_pay_off_car_l287_287740

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l287_287740


namespace coefficient_fourth_term_l287_287674

-- Let's state the binomial expansion definition for the given expression
def binomial_expansion (x : ℝ) : ℝ :=
  (2 * x^2 - 1 / x)^5

-- Now, let's state the theorem about the coefficient of the fourth term
theorem coefficient_fourth_term : 
  ∃ (a : ℝ), binomial_expansion x = (2 * x^2 - 1 / x)^5 ∧
  a = -40 ∧
  (binomial_expansion x - a * x) = - (40 / x) :=
sorry -- proof goes here

end coefficient_fourth_term_l287_287674


namespace farmer_land_l287_287742

variable (A C G P T : ℝ)
variable (h1 : C = 0.90 * A)
variable (h2 : G = 0.10 * C)
variable (h3 : P = 0.80 * C)
variable (h4 : T = 450)
variable (h5 : C = G + P + T)

theorem farmer_land (A : ℝ) (h1 : C = 0.90 * A) (h2 : G = 0.10 * C) (h3 : P = 0.80 * C) (h4 : T = 450) (h5 : C = G + P + T) : A = 5000 := by
  sorry

end farmer_land_l287_287742


namespace problem_solution_l287_287025

open Real

def point (α : Type*) := (α × α)

def sym_pt_xaxis (A A' : point ℝ) : Prop :=
  A.fst = -A'.fst ∧ A.snd = A'.snd

def parabola (x y : ℝ) : Prop :=
  y^2 = 2 * x

def on_line (A B : point ℝ) (λ : ℝ) :=
  ∃ D : point ℝ, D = (λ * A.fst + (1 - λ) * B.fst, λ * A.snd + (1 - λ) * B.snd)

def ratio_condition (A B C : point ℝ) (D E : point ℝ) : Prop :=
  abs E.fst / abs C.fst = abs D.fst / abs B.fst

noncomputable def has_one_parabola_intersection (A A' B C D E : point ℝ) : Prop :=
  ∀ (DE : point ℝ → point ℝ → Prop), DE D E → ¬ ∃ x y : ℝ, parabola x y ∧ DE (x, y) (0, 0)

noncomputable def area (A B C : point ℝ) : ℝ :=
  abs ((B.fst - A.fst) * (C.snd - A.snd) - (C.fst - A.fst) * (B.snd - A.snd)) / 2

noncomputable def ratio_of_areas (A B C D E F : point ℝ) : ℝ :=
  area B C F / area A D E

theorem problem_solution (A A' B C D E F : point ℝ) (λ : ℝ):
  (sym_pt_xaxis A A') →
  (on_line A B λ) →
  (on_line A C λ) →
  (ratio_condition A B C D E) →
  (has_one_parabola_intersection A A' B C D E) →
  (parabola F.fst F.snd) →
  (ratio_of_areas A B C D E F = 2) :=
begin
  -- Proof goes here
  sorry
end

end problem_solution_l287_287025


namespace find_fraction_l287_287245

-- Define the given variables and conditions
variables (x y : ℝ)
-- Assume x and y are nonzero
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
-- Assume the given condition
variable (h : (4*x + 2*y) / (2*x - 8*y) = 3)

-- Define the theorem to be proven
theorem find_fraction (h : (4*x + 2*y) / (2*x - 8*y) = 3) : (x + 4 * y) / (4 * x - y) = 1 / 3 := 
by
  sorry

end find_fraction_l287_287245


namespace find_extrema_on_interval_l287_287065
open Real

noncomputable def extrema_function (x : ℝ) : ℝ := (sin (3 * x)) ^ 2

theorem find_extrema_on_interval :
  ∃ x, (0 < x ∧ x < 0.6) ∧ extremum extrema_function (0 : ℝ) 0.6 x := 
begin
  use [π / 6],
  split,
  { split,
    { linarith [Real.pi_pos],  -- Proving 0 < π / 6
      exact Real.pi_div_six_lt_six,
    },
  },
  sorry
end

end find_extrema_on_interval_l287_287065


namespace total_people_on_hike_l287_287797

theorem total_people_on_hike
  (cars : ℕ) (cars_people : ℕ)
  (taxis : ℕ) (taxis_people : ℕ)
  (vans : ℕ) (vans_people : ℕ)
  (buses : ℕ) (buses_people : ℕ)
  (minibuses : ℕ) (minibuses_people : ℕ)
  (h_cars : cars = 7) (h_cars_people : cars_people = 4)
  (h_taxis : taxis = 10) (h_taxis_people : taxis_people = 6)
  (h_vans : vans = 4) (h_vans_people : vans_people = 5)
  (h_buses : buses = 3) (h_buses_people : buses_people = 20)
  (h_minibuses : minibuses = 2) (h_minibuses_people : minibuses_people = 8) :
  cars * cars_people + taxis * taxis_people + vans * vans_people + buses * buses_people + minibuses * minibuses_people = 184 :=
by
  sorry

end total_people_on_hike_l287_287797


namespace find_f_prime_at_2_l287_287336

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287336


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287561

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287561


namespace lambda_range_l287_287584

theorem lambda_range (λ : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = λ * n^2 + n) (mono_dec : ∀ n : ℕ, a n > a (n + 1)) :
  λ < -1/3 :=
by sorry

end lambda_range_l287_287584


namespace problem_solution_l287_287643

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_solution : f(g(2)) = -19 :=
by
  -- proof omitted
  sorry

end problem_solution_l287_287643


namespace sum_first_six_terms_geometric_seq_l287_287998

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l287_287998


namespace find_f_prime_at_2_l287_287306

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287306


namespace zeros_of_f_l287_287280

def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

theorem zeros_of_f :
  (f (-4) = 0) ∧ (f 2 = 0) :=
by
  -- Define the function
  let f := λ x : ℝ, if x < 0 then -x - 4 else x^2 - 4 in
  -- Evaluate f at -4 and 2
  have h1 : f (-4) = 0 := sorry,
  have h2 : f 2 = 0 := sorry,
  -- Conclude
  exact ⟨h1, h2⟩

end zeros_of_f_l287_287280


namespace steamship_encounter_count_l287_287011

/-- 
Question: How many ships from the same company traveling in the opposite direction will a steamship 
meet on its way from Le Havre to New York?

Conditions: 
1. A steamship departs from Le Havre to New York across the Atlantic Ocean every day at noon.
2. At the same time, a steamship from the same company departs from New York to Le Havre.
3. The journey in each direction takes 7 days.
-/
theorem steamship_encounter_count 
  (departure_LH_NY : ∀ t : ℕ, t >= 0 → t < 24 → t = 12 → true)
  (departure_NY_LH : ∀ t : ℕ, t >= 0 → t < 24 → t = 12 → true)
  (journey_duration : ∀ d : ℕ, d = 7 → true) : 
  ∀ t : ℕ, t mod 24 = 12 → (number_of_ships_on_journey 7 t) = 7 :=
sorry

end steamship_encounter_count_l287_287011


namespace trapezoid_area_l287_287453

theorem trapezoid_area {a b c d e : ℝ} (h1 : a = 40) (h2 : b = 40) (h3 : c = 50) (h4 : d = 50) (h5 : e = 60) : 
  (a + b = 80) → (c * c = 2500) → 
  (50^2 - 30^2 = 1600) → ((50^2 - 30^2).sqrt = 40) → 
  (((e - 2 * ((a ^ 2 - (30) ^ 2).sqrt)) * 40) / 2 = 1336) :=
sorry

end trapezoid_area_l287_287453


namespace cone_volume_ratio_l287_287927

theorem cone_volume_ratio (r h : ℝ) :
  let V_A := (1/3) * real.pi * r^2 * h,
      V_B := (1/3) * real.pi * (2 * r)^2 * (2 * h),
      V_C := (1/3) * real.pi * (3 * r)^2 * (3 * h),
      V_D := (1/3) * real.pi * (4 * r)^2 * (4 * h),
      V_E := (1/3) * real.pi * (5 * r)^2 * (5 * h),
      V1 := V_E - V_D,
      V2 := V_D - V_C
  in V2 / V1 = 37 / 61 :=
by
  sorry

end cone_volume_ratio_l287_287927


namespace inequality_proof_l287_287721

noncomputable def problem_statement (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (n ≥ 3) ∧
  (∀ i : Fin n, i.val ≥ 2 → x i > 0) ∧
  (x 1 * x 2 * ... * x (n - 1) = 1) ∧
  ((1 + x 1)^2 * (1 + x 2)^3 * ... * (1 + x (n - 1))^n > n^n)

theorem inequality_proof {n : ℕ} {x : Fin n → ℝ}
  (h : problem_statement n x) : 
  ((1 + x 1)^2 * (1 + x 2)^3 * ... * (1 + x (n - 1))^n > n^n) :=
sorry

end inequality_proof_l287_287721


namespace find_f_prime_at_2_l287_287349

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287349


namespace binary_110101_is_53_l287_287498

-- Define the conversion from binary to decimal
def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum_from 0 |>.sum (λ (p : ℕ × ℕ), (p.1 * (2 ^ p.2)))

-- Define the specific binary number 110101_(2)
def binary_110101 : List ℕ := [1, 1, 0, 1, 0, 1]

-- The target decimal number
def target_decimal : ℕ := 53

-- The main statement to be proved
theorem binary_110101_is_53 : binary_to_decimal binary_110101 = target_decimal := by
  sorry

end binary_110101_is_53_l287_287498


namespace order_of_a_b_c_l287_287091

noncomputable def a : ℝ := (real.sqrt 2 / 2) * (real.sin (real.pi / 180 * 17) + real.cos (real.pi / 180 * 17))
noncomputable def b : ℝ := 2 * (real.cos (real.pi / 180 * 13))^2 - 1
noncomputable def c : ℝ := (real.sin (real.pi / 180 * 37) * real.sin (real.pi / 180 * 67)) + (real.sin (real.pi / 180 * 53) * real.sin (real.pi / 180 * 23))

theorem order_of_a_b_c : c < a ∧ a < b :=
by
  -- The proof will go here.
  sorry

end order_of_a_b_c_l287_287091


namespace seq_is_arithmetic_sum_of_bn_l287_287624

-- Define the sequence {a_n}
def a_1 : ℚ := 1 / 2
def a (n : ℕ) (hn : n > 0) : ℚ := 
  if h : n = 1 then 
    a_1 
  else 
    let an := a (n - 1) (nat.lt_of_succ_lt hn) in
    1 / (2 - an)

-- Define the sequence {1 / (1 - a_n)}
def arithmetic_sequence_property (n : ℕ) (hn : n > 0) : Prop :=
  (1 / (1 - a (n + 1) (nat.succ_pos _))) = 1 + (1 / (1 - a n hn))

-- Prove that {1 / (1 - a_n)} is an arithmetic sequence
theorem seq_is_arithmetic (n : ℕ) (hn : n > 0) : arithmetic_sequence_property n hn := 
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) (hn : n > 0) : ℚ := a n hn / (n^2)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℚ := (finset.range n).sum (λ k, b (k + 1) (nat.succ_pos k))

-- Prove the sum of the first n terms of the sequence {b_n}
theorem sum_of_bn (n : ℕ) : S n = n / (n + 1) :=
  sorry

end seq_is_arithmetic_sum_of_bn_l287_287624


namespace pos_diff_is_multiple_of_9_l287_287391

theorem pos_diff_is_multiple_of_9 
  (q r : ℕ) 
  (h_qr : 10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 ∧ (q % 10) * 10 + (q / 10) = r)
  (h_max_diff : q - r = 63) : 
  ∃ k : ℕ, q - r = 9 * k :=
by
  sorry

end pos_diff_is_multiple_of_9_l287_287391


namespace percentage_of_watermelon_juice_l287_287919

theorem percentage_of_watermelon_juice
  (total_drink_ounces : ℕ)
  (orange_juice_percentage : ℝ)
  (grape_juice_ounces : ℕ)
  (orange_juice_ounces : ℕ)
  (watermelon_juice_ounces : ℕ)
  (W_percentage : ℝ) :
  total_drink_ounces = 120 ∧
  orange_juice_percentage = 15 ∧
  grape_juice_ounces = 30 ∧
  orange_juice_ounces = (orange_juice_percentage / 100) * total_drink_ounces ∧
  watermelon_juice_ounces = total_drink_ounces - orange_juice_ounces - grape_juice_ounces ∧
  W_percentage = (watermelon_juice_ounces / total_drink_ounces) * 100 →
  W_percentage = 60 :=
begin
  sorry
end

end percentage_of_watermelon_juice_l287_287919


namespace students_left_in_final_year_l287_287956

variable (s10 s_next s_final x : Nat)

-- Conditions
def initial_students : Prop := s10 = 150
def students_after_joining : Prop := s_next = s10 + 30
def students_final_year : Prop := s_final = s_next - x
def final_year_students : Prop := s_final = 165

-- Theorem to prove
theorem students_left_in_final_year (h1 : initial_students s10)
                                     (h2 : students_after_joining s10 s_next)
                                     (h3 : students_final_year s_next s_final x)
                                     (h4 : final_year_students s_final) :
  x = 15 :=
by
  sorry

end students_left_in_final_year_l287_287956


namespace average_goals_is_92_l287_287156

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l287_287156


namespace find_f_prime_at_2_l287_287338

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287338


namespace volume_ratio_CO2_O2_l287_287465

def A : Type := ℝ -- type for gas A, e.g., CO2
def B : Type := ℝ -- type for gas B, e.g., O2
def M_A (a: A) : ℝ := sorry -- molecular mass of gas A
def M_B (b: B) : ℝ := sorry -- molecular mass of gas B

axiom mass_ratio_CO_in_CO2 : ℝ := 3 / 8 -- mass ratio of C to O in CO2
axiom mass_ratio_limit : ℝ := 1 / 8 -- given mass ratio of C to O in the mixed gas

variables (a : A) (b : B)

theorem volume_ratio_CO2_O2 : M_A a > M_B b → mass_ratio_limit = 1 / 8 →
  ∀ (x y : ℝ), mass_ratio_CO_in_CO2 = 3 / 4 →
  (12 * x) / (16 * (2 * x + 2 * y)) = 1 / 8 → 
  2 * x = y :=
sorry

end volume_ratio_CO2_O2_l287_287465


namespace calculate_expression_l287_287843

theorem calculate_expression :
  427 / 2.68 * 16 * 26.8 / 42.7 * 16 = 25600 :=
sorry

end calculate_expression_l287_287843


namespace rectangle_area_given_perimeter_l287_287167

theorem rectangle_area_given_perimeter (x : ℝ) (h_perim : 8 * x = 160) : (2 * x) * (2 * x) = 1600 := by
  -- Definitions derived from conditions
  let length := 2 * x
  let width := 2 * x
  -- Proof transformed to a Lean statement
  have h1 : length = 40 := by sorry
  have h2 : width = 40 := by sorry
  have h_area : length * width = 1600 := by sorry
  exact h_area

end rectangle_area_given_perimeter_l287_287167


namespace magnitude_product_l287_287508

def z1 := Complex.mk 7 (-24)
def z2 := Complex.mk 3 4

theorem magnitude_product : abs (z1 * z2) = 125 := 
by sorry

end magnitude_product_l287_287508


namespace apple_juice_percentages_l287_287813

variable {α β : ℝ}

def apple_juice_percentage (v1 v2 : ℝ) (m1 m2 : ℝ) (p1 p2 : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    v1 = 1 ∧
    v2 = 2 ∧
    m1 = 0.5 ∧
    p1 = 0.4 ∧
    m2 = 2.5 ∧
    p2 = 0.88 ∧
    α + β = m2 ∧
    0.4 * α + 1 * β = m2 * p2 ∧
    0.4 = x ∧
    1 = y

theorem apple_juice_percentages :
  apple_juice_percentage 1 2 0.5 2.5 0.4 0.88 40 100 :=
sorry

end apple_juice_percentages_l287_287813


namespace count_valid_arrangements_l287_287055

namespace ChairRearrangement

-- Definition of the problem conditions
def initialSeating := [1, 2, 3, 4, 5, 6, 7, 8]

def isValidMove (initial current : Fin 8) : Prop :=
  initial ≠ current ∧ initial ≠ current.pred ∧ initial ≠ current.succ

def validArrangement (arrangement : List (Fin 8)) : Prop :=
  arrangement.length = 8 ∧ ∀ (i : Fin 8), isValidMove (initialSeating.nthLe i i.is_lt) (arrangement.nthLe i i.is_lt)

-- Theorem statement
theorem count_valid_arrangements : {arrangements : List (List (Fin 8)) // ∀ arr ∈ arrangements, validArrangement arr}.length = 100 :=
sorry

end ChairRearrangement

end count_valid_arrangements_l287_287055


namespace sqrt_a_add_one_eq_combined_8_l287_287112

theorem sqrt_a_add_one_eq_combined_8 (a : ℝ) : 
  (sqrt (a + 1) = sqrt 8) -> a = 1 := 
by
  intro h
  have h1 : a + 1 = 2 := sorry
  exact sorry

end sqrt_a_add_one_eq_combined_8_l287_287112


namespace solution_l287_287841

noncomputable def problem_statement : Prop :=
  (deriv (λ x : ℝ, x) = 1) ∧ (deriv (λ x : ℝ, Real.sqrt x) = (1 / (2 * Real.sqrt x)))

theorem solution : problem_statement :=
  by
    split
    exact deriv_id
    sorry

end solution_l287_287841


namespace age_difference_in_decades_l287_287392

-- Declare the ages of x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the condition
def age_condition (x y z : ℝ) : Prop := x + y = y + z + 18

-- The proof problem statement
theorem age_difference_in_decades (h : age_condition x y z) : (x - z) / 10 = 1.8 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end age_difference_in_decades_l287_287392


namespace range_of_a_l287_287623

variable (a : ℝ)

def f (x : ℝ) := (2 * a - 6) ^ x
def g (x : ℝ) := x^2 - 3 * a * x + 2 * a^2 + 1

theorem range_of_a :
  ¬(∀ x, (2 * a - 6) ^ x < (2 * a - 6) ^ (x + 1)) ∧
  ∀ x, x^2 - 3 * a * x + 2 * a^2 + 1 > 0 →
  a > 7 / 2 :=
sorry

end range_of_a_l287_287623


namespace trapezoid_construction_l287_287046

-- We start by defining Axioms and Definitions

def A (x : ℝ) (y : ℝ) : Type := sorry -- Define point A
def B (x : ℝ) (y : ℝ) : Type := sorry -- Define point B 
def C (x : ℝ) (y : ℝ) : Type := sorry -- Define point C
def D (x : ℝ) (y : ℝ) : Type := sorry -- Define point D
def E (x : ℝ) (y : ℝ) : Type := sorry -- Define point E
def S (x : ℝ) (y : ℝ) : Type := sorry -- Define point S

-- Define lengths as given in the problem
def length_AB (AB : ℝ) := 2 * (length_CD CD)
def length_AD (AD : ℝ) := d
def length_BC (BC : ℝ) := b

-- Define the angle between diagonals
def angle_ASB (AS : ℝ) (SB : ℝ) := φ

-- Define the reflection relationships for point E
def reflection_A_over_D (A : Type) (D : Type) := E
def reflection_B_over_C (B : Type) (C : Type) := E

-- Define centroids and diagonal relationships
def centroid_S_triangle_ABE (A : Type) (B : Type) (E : Type) := S

-- Define circle passing through points and angles condition
def circle_passing_AD (A : Type) (D : Type) (θ : ℝ) := φ = 180 - θ

-- Define scaled circle conditions for point C
def scaled_circle_C (C : Type) (A : Type) := sorry
def circle_centered_E (C : Type) (E : Type) (BC_length : ℝ) := sorry

-- Define the intersection points for locating C
def intersection_points_Circle (C : Type) := sorry

-- Define the reflection for point B
def reflection_E_over_C (E : Type) (C : Type) := B

-- Using all above definitions to create a statement that encapuslates the given condition:
theorem trapezoid_construction :
  ∃ A B C D E S : Type,
  (A (0, 0)) ∧ -- Utilizing one initial point for simplicity, point placement will vary
  (B (x₁, y₁)) ∧ -- To be placed based on derived conditions
  (C (x₂, y₂)) ∧ -- To be placed based on derived conditions
  (D (0, d)) ∧ -- D placed a distance d away from A
  (E (x₃, y₃)) ∧ -- E needs proper placement based on reflections
  (S (x₄, y₄)) ∧ -- S obtained as the centroid and more precise location thru calculations
  (length_AB (AB) = 2 * (length_CD CD)) ∧ 
  (length_AD (AD) = d) ∧ 
  (length_BC (BC) = b) ∧
  (angle_ASB (AS) (SB) = φ) ∧
  (reflection_A_over_D (A) (D) = E) ∧
  (reflection_B_over_C (B) (C) = E) ∧
  (centroid_S_triangle_ABE (A) (B) (E) = S) ∧
  (circle_passing_AD (A) (D) (θ)) ∧ 
  (scaled_circle_C (C) (A)) ∧
  (circle_centered_E (C) (E) (length_BC BC)) ∧
  (intersection_points_Circle (C)) ∧
  (reflection_E_over_C (E) (C) = B) := sorry

end trapezoid_construction_l287_287046


namespace number_of_months_to_fully_pay_off_car_l287_287739

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l287_287739


namespace date_stats_2020_l287_287735

theorem date_stats_2020 (values : Finset ℕ) 
  (h1 : ∀ k ∈ (Finset.range 30), values.count k = 12) 
  (h2 : values.count 31 = 7) :
  let μ := (12 * (30 * 31) / 2 + 7 * 31) / 366 in
  let M := 16 in
  let d := 15.5 in
  d < μ ∧ μ < M :=
by
  let μ := 15.84
  let M := 16
  let d := 15.5
  sorry

end date_stats_2020_l287_287735


namespace find_f_prime_at_two_l287_287324

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287324


namespace alice_wins_game_l287_287439

theorem alice_wins_game :
  ∀ (p : ℕ → Nat.Prime), ∃ (have : ℕ → Prop),
  let play_step := λ n, n + p n in
  let game_over := λ n, n > 2023^2023 in
  let alice_turn := true in -- Assuming true for Alice's turn
  let initial_number := 2 in
  ∃ next_number, 
  (play_step initial_number = next_number ∧ ¬ game_over next_number ∧ alice_turn) ∧
  (∀ bob_next_number, play_step next_number = bob_next_number → ¬ game_over bob_next_number →
  ∃ next_alice_number, play_step bob_next_number = next_alice_number ∧ ¬ game_over next_alice_number) :=
sorry

end alice_wins_game_l287_287439


namespace sum_of_integers_is_96_l287_287747

theorem sum_of_integers_is_96 (x y : ℤ) (h1 : x = 32) (h2 : y = 2 * x) : x + y = 96 := 
by
  sorry

end sum_of_integers_is_96_l287_287747


namespace filter_kit_savings_l287_287865

theorem filter_kit_savings:
  let kit_cost := 170.00,
  let individual_cost_tier1 := 2 * 12.45 + 2 * 14.05 + 2 * 22.30 + 11.50 + 18.30 + 15.75 + 16.95,
  let individual_cost_tier2 := individual_cost_tier1 * 0.90,
  let individual_cost_tier3 := individual_cost_tier1 * 0.85,
  let savings_tier1 := individual_cost_tier1 - kit_cost,
  let savings_tier2 := individual_cost_tier2 - kit_cost,
  let savings_tier3 := individual_cost_tier3 - kit_cost,
  let savings_percentage_tier1 := (savings_tier1 / individual_cost_tier1) * 100,
  let savings_percentage_tier2 := (savings_tier2 / individual_cost_tier2) * 100,
  let savings_percentage_tier3 := (savings_tier3 / individual_cost_tier3) * 100
  in
  savings_percentage_tier1 = -6.18 /- Prove it! -/ ∧
  savings_percentage_tier2 ≈ 17.98 ∧
  savings_percentage_tier3 ≈ 24.92 := by
    sorry

end filter_kit_savings_l287_287865


namespace tangent_line_intersection_l287_287888

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287888


namespace erased_number_l287_287468

theorem erased_number (a b : ℤ) (h1 : ∀ n : ℤ, n ∈ set.range (λ i, a + i) ↔ n ∈ set.range (λ i, a - 4 + i)) 
                      (h2 : 8 * a - b = 1703)
                      (h3 : -4 ≤ b ∧ b ≤ 4) : a + b = 214 := 
by
    sorry

end erased_number_l287_287468


namespace bank_deposit_years_l287_287924

-- Definitions of given conditions
def initial_deposit : ℕ := 5600
def annual_interest_rate : ℚ := 0.07
def ending_amount : ℕ := 6384

-- Lean statement to prove the number of years the money was kept in the bank
theorem bank_deposit_years: 
  ∃ n : ℕ, 
  ending_amount = initial_deposit * (1 + n * annual_interest_rate) ∧ 
  n = 2 :=
begin
  -- sorry indicates missing proof
  sorry
end

end bank_deposit_years_l287_287924


namespace intersecting_chords_theorem_l287_287817

theorem intersecting_chords_theorem
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18)
  (c d k : ℝ) (h3 : c = 3 * k) (h4 : d = 8 * k) :
  (a * b = c * d) → (k = 3) → (c + d = 33) :=
by 
  sorry

end intersecting_chords_theorem_l287_287817


namespace solve_cos_YXW_l287_287185

noncomputable def triangle_XYZ (XY XZ YZ : ℝ) (hXY : XY = 4) (hXZ : XZ = 5) (hYZ : YZ = 7) : Prop :=
  let X := XY
  let Y := XZ
  let Z := YZ
  let cos_X := (X^2 + Y^2 - Z^2) / (2 * X * Y)
  ∃ W : ℝ, (W ∈ [0, 1]) ∧ (cos_X = -1 / 5) ∧ (cos (1 / 2 * real.arccos (-1 / 5)) = real.sqrt (2 / 5))

theorem solve_cos_YXW :
  ∀ (XY XZ YZ : ℝ), (XY = 4) → (XZ = 5) → (YZ = 7) → ∃ W, (W ∈ [0, 1]) ∧ (cos (1 / 2 * real.arccos (-1 / 5)) = real.sqrt (2 / 5)) := by
  intros XY XZ YZ hXY hXZ hYZ
  sorry

end solve_cos_YXW_l287_287185


namespace point_on_right_branch_l287_287637

noncomputable def on_hyperbola_right_branch (a b m : ℝ) :=
  (∀ a b m : ℝ, (a - 2 * b > 0) → (a + 2 * b > 0) → (a ^ 2 - 4 * b ^ 2 = m) → (m ≠ 0) → a > 0)

theorem point_on_right_branch (a b m : ℝ) (h₁ : a - 2 * b > 0) (h₂ : a + 2 * b > 0) (h₃ : a ^ 2 - 4 * b ^ 2 = m) (h₄ : m ≠ 0) :
  a > 0 := 
by 
  sorry

end point_on_right_branch_l287_287637


namespace find_extrema_on_interval_l287_287064
open Real

noncomputable def extrema_function (x : ℝ) : ℝ := (sin (3 * x)) ^ 2

theorem find_extrema_on_interval :
  ∃ x, (0 < x ∧ x < 0.6) ∧ extremum extrema_function (0 : ℝ) 0.6 x := 
begin
  use [π / 6],
  split,
  { split,
    { linarith [Real.pi_pos],  -- Proving 0 < π / 6
      exact Real.pi_div_six_lt_six,
    },
  },
  sorry
end

end find_extrema_on_interval_l287_287064


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287550

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287550


namespace intersect_complementA_B_l287_287595

namespace ProofProblem

-- Define set A using the given quadratic inequality
def A : Set ℝ := {x | x^2 - 3 * x + 2 < 0}

-- Define set B using the given exponential inequality
def B : Set ℝ := {x | 3^x > 9}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}

-- Prove the intersection of complementA and B
theorem intersect_complementA_B :
  (complementA ∩ B) = {x | x > 2} :=
by 
  sorry

end ProofProblem

end intersect_complementA_B_l287_287595


namespace grapefruit_oranges_same_plane_l287_287282

def Sphere (center : Point3D) (radius : ℝ) : Prop := sorry

def Tangent (sphere1 sphere2 : Sphere) (point : Point3D) : Prop := sorry

theorem grapefruit_oranges_same_plane
  (v V g G a A1 A2 A3 A4 : Point3D)
  (P1 P2 P3 P4 K1 K2 K3 K4 : Point3D)
  (hemispherical_vase : Sphere V v)
  (orange1 orange2 orange3 orange4 : Sphere)
  (grapefruit : Sphere G g)
  (tangency_1 : Tangent grapefruit orange1 K1)
  (tangency_2 : Tangent grapefruit orange2 K2)
  (tangency_3 : Tangent grapefruit orange3 K3)
  (tangency_4 : Tangent grapefruit orange4 K4)
  (touches_vase1 : Tangent hemispherical_vase orange1 P1)
  (touches_vase2 : Tangent hemispherical_vase orange2 P2)
  (touches_vase3 : Tangent hemispherical_vase orange3 P3)
  (touches_vase4 : Tangent hemispherical_vase orange4 P4):
  Collinear K1 K2 K3 K4 :=
sorry

end grapefruit_oranges_same_plane_l287_287282


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287540

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287540


namespace tangent_line_intersection_l287_287886

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287886


namespace value_of_2a_minus_b_l287_287646

theorem value_of_2a_minus_b 
  (a b : ℝ)
  (h1 : ∀ x : ℝ, g (g x) = x)
  (h2 : a ≠ 0)
  (h3 : b ≠ 0)
  (g : ℝ → ℝ := λ x, (2 * a * x - b) / (b * x + 2 * a)) :
  2 * a - b = 0 := 
    sorry

end value_of_2a_minus_b_l287_287646


namespace problem_statement_l287_287234

theorem problem_statement (n : ℕ) (h : 2 ≤ n) :
  (∏ (k : ℕ) in finset.range n, (1 + 1 / (2 * k + 1))) > (real.sqrt (2 * n + 1)) / 2 :=
sorry

end problem_statement_l287_287234


namespace smallest_non_palindrome_product_l287_287546

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287546


namespace derivative_at_2_l287_287373

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287373


namespace ratio_of_shaded_to_empty_l287_287462

-- Definitions based on conditions
def vertices_of_regular_hexagon : Prop := sorry
def hexagram (hexagon : Prop) : Prop := 
    hexagon ∧ (formed_by_two_equilateral_triangles hexagon)

def formed_by_two_equilateral_triangles (hexagon : Prop) : Prop := sorry

def shaded_area_to_empty_space_ratio (shaded empty : Nat) : Nat :=
  shaded / empty

-- Given conditions as premises
axiom hexagon_vertices : vertices_of_regular_hexagon
axiom hexagram_form : hexagram hexagon_vertices
axiom shaded_triangles : Nat := 18
axiom empty_triangles : Nat := 6

-- Main theorem statement
theorem ratio_of_shaded_to_empty : shaded_area_to_empty_space_ratio shaded_triangles empty_triangles = 3 :=
by 
  sorry

end ratio_of_shaded_to_empty_l287_287462


namespace ratio_of_new_average_to_original_average_is_one_to_one_l287_287942

theorem ratio_of_new_average_to_original_average_is_one_to_one 
  (scores : List ℝ)
  (h_len : scores.length = 50)
  (h_arith_prog : ∀ i j, 0 ≤ i → i < j → j < 50 → scores.nth i + scores.nth j = scores.head + scores.back) :
  let true_average := (scores.head + scores.back) / 2
  let median := (scores.nth 24 + scores.nth 25) / 2 in
  let new_average := (scores.sum - scores.head + median) / 50 in
  new_average = true_average :=
sorry

end ratio_of_new_average_to_original_average_is_one_to_one_l287_287942


namespace tangent_circles_x_intersect_l287_287868

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287868


namespace rudy_robi_ratio_l287_287239

variables (R R_rudi R_robi x : ℝ)
variables (contribution_robi : ℝ := 4000)
variables (profit_percentage : ℝ := 0.20)
variables (total_profit : ℝ := 1800)
variables (share_profit : ℝ := 900)

-- Rudy's contribution
def Rudy_contribution (R_robi x : ℝ) : ℝ := R_robi + x

-- Total contributions
def Total_contribution (R_robi x : ℝ) : ℝ := R_robi + Rudy_contribution R_robi x

-- Profit calculation
def Calculate_profit (R_robi x : ℝ) : ℝ := profit_percentage * Total_contribution R_robi x

theorem rudy_robi_ratio (h1 : contribution_robi = 4000)
                        (h2 : Calculate_profit contribution_robi x = total_profit) :
  R = 5000 →
  R / contribution_robi = 5 / 4 :=
begin
  sorry
end


end rudy_robi_ratio_l287_287239


namespace min_sum_x8y4z_l287_287722

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l287_287722


namespace route_difference_is_18_l287_287214

theorem route_difference_is_18 :
  let t_uphill := 6 in
  let t_path1 := 2 * t_uphill in
  let t_final1 := (t_uphill + t_path1)/3 in
  let t_route1 := t_uphill + t_path1 + t_final1 in
  let t_flat := 14 in
  let t_final2 := 2 * t_flat in
  let t_route2 := t_flat + t_final2 in
  t_route2 - t_route1 = 18 :=
by
  let t_uphill := 6
  let t_path1 := 2 * t_uphill
  let t_final1 := (t_uphill + t_path1)/3
  let t_route1 := t_uphill + t_path1 + t_final1
  let t_flat := 14
  let t_final2 := 2 * t_flat
  let t_route2 := t_flat + t_final2
  have h : t_route2 - t_route1 = 18
  sorry

end route_difference_is_18_l287_287214


namespace mathematically_equivalent_proof_l287_287151

noncomputable def proof_problem (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a^x = 2 ∧ a^y = 3 → a^(x - 2 * y) = 2 / 9

theorem mathematically_equivalent_proof (a : ℝ) (x y : ℝ) :
  proof_problem a x y :=
by
  sorry  -- Proof steps will go here

end mathematically_equivalent_proof_l287_287151


namespace length_DE_in_triangle_l287_287686

theorem length_DE_in_triangle
  (A B C D E : Type)
  (dist_BC : ℝ)
  (angle_C : ℝ)
  (midpoint_D : D = midpoint B C)
  (perpendicular_bisector : is_perpendicular_bisector D B C E)
  (BC_eq : BC = 30)
  (angle_C_eq : angle_C = 45)
  (legs_equal : ∀ {x y z}, angle_CA B C y z = 45 → dist y z = dist y x)
  : dist D E = 15 := 
by
  sorry

end length_DE_in_triangle_l287_287686


namespace route_comparison_l287_287217

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l287_287217


namespace betty_picked_16_l287_287028

noncomputable theory

variables (B M N : ℕ)

-- Given conditions:
def condition1 := M = B + 20
def condition2 := M = 2 * N
def condition3 := 4 * 10 = 40
def condition4 := 7 * 10 = 70

-- Total strawberries used for 10 jars of jam
def total_strawberries := 10 * 7

-- Total strawberries is sum of strawberries picked
def total_picked := B + M + N

-- Using the given conditions prove that Betty picked 16 strawberries
theorem betty_picked_16 :
  (condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ total_picked = total_strawberries) → B = 16 :=
by sorry

end betty_picked_16_l287_287028


namespace not_eq_a_and_not_eq_b_l287_287822

theorem not_eq_a_and_not_eq_b {x a b : ℝ} : x^2 - (a + b) * x - a * b ≠ 0 → (x ≠ a ∧ x ≠ b) :=
by
  intro h
  by_contradiction H
  have : x = a ∨ x = b := by
    apply or_iff_not_and_not.mpr
    exact H
  sorry

end not_eq_a_and_not_eq_b_l287_287822


namespace value_of_y_when_x_is_9_l287_287773

-- Define the given conditions and desired outcome
theorem value_of_y_when_x_is_9 
  (x y : ℝ)
  (h1 : ∀ k : ℝ, (4 * x - 5) / (2 * y + 20) = k)
  (h2 : (4 * 4 - 5) / (2 * 5 + 20) = 11 / 30) :
  ∃ y : ℝ, x = 9 → y = 355 / 11 :=
begin
  sorry
end

end value_of_y_when_x_is_9_l287_287773


namespace minoxidil_percentage_l287_287925

-- Define the given conditions
def volume1 : ℝ := 70
def concentration1 (x : ℝ) : ℝ := x / 100
def volume2 : ℝ := 35
def concentration2 : ℝ := 5 / 100
def total_volume : ℝ := 105
def desired_concentration : ℝ := 3 / 100

-- Define the Lean theorem statement
theorem minoxidil_percentage (x : ℝ) (h1 : volume1 * concentration1 x + volume2 * concentration2 = total_volume * desired_concentration) : x = 2 :=
by 
  -- Proof omitted.
  sorry

end minoxidil_percentage_l287_287925


namespace minimum_numbers_to_form_triangle_l287_287830

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem minimum_numbers_to_form_triangle :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 1001) →
    16 ≤ S.card →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ {a, b, c} ⊆ S ∧ is_triangle a b c :=
by
  sorry

end minimum_numbers_to_form_triangle_l287_287830


namespace original_number_of_employees_l287_287002

theorem original_number_of_employees (E : ℝ) :
  (E - 0.125 * E) - 0.09 * (E - 0.125 * E) = 12385 → E = 15545 := 
by  -- Start the proof
  sorry  -- Placeholder for the proof, which is not required

end original_number_of_employees_l287_287002


namespace exists_thick_set_with_finite_sum_l287_287401

def is_thick (S : Set ℝ) : Prop :=
  ∀ (n : ℕ), ∃ (x y : ℝ), x ∈ S ∧ y ∈ S ∧ (1 / (n + 1) ≤ |x - y| ∧ |x - y| ≤ 1 / n)

def is_finite_sum (S : Set ℝ) : Prop :=
  (S.to_finset.sum id).finite

theorem exists_thick_set_with_finite_sum :
  ∃ (S : Set ℝ), (is_thick S) ∧ (is_finite_sum S) :=
sorry

end exists_thick_set_with_finite_sum_l287_287401


namespace length_DE_in_triangle_l287_287685

theorem length_DE_in_triangle
  (A B C D E : Type)
  (dist_BC : ℝ)
  (angle_C : ℝ)
  (midpoint_D : D = midpoint B C)
  (perpendicular_bisector : is_perpendicular_bisector D B C E)
  (BC_eq : BC = 30)
  (angle_C_eq : angle_C = 45)
  (legs_equal : ∀ {x y z}, angle_CA B C y z = 45 → dist y z = dist y x)
  : dist D E = 15 := 
by
  sorry

end length_DE_in_triangle_l287_287685


namespace erased_number_is_214_l287_287472

theorem erased_number_is_214 {a b : ℤ} 
  (h1 : 9 * a = sum (a - 4 :: a - 3 :: a - 2 :: a - 1 :: a :: a + 1 :: a + 2 :: a + 3 :: a + 4 :: []))
  (h2 : ∑ n in {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4} \ erase_val (a + b) {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4}, n = 1703)
  (h3 : -4 ≤ b ∧ b ≤ 4)
  : a + b = 214 :=
begin
  -- proof to be filled in
  sorry,
end

end erased_number_is_214_l287_287472


namespace triangle_similarity_ratio_l287_287382

theorem triangle_similarity_ratio (A B C D E F : Type*) [HasRightAngle B D] (AB BC AD EF ED : ℝ)
  (h_AB : AB = 3) (h_BC : BC = 5) (h_AD : AD = 13)
  (h_EF_parallel_AB : ∥AB∥ ∥ ∥EF∥) (h_right_angle_ABD : HasRightAngle AB D)
  (h_right_angle_EFD : HasRightAngle EF D) :
  ∃ p q : ℕ, p + q = 16 ∧ gcd p q = 1 ∧ EF / ED = p / q :=
by
  sorry

end triangle_similarity_ratio_l287_287382


namespace expected_value_l287_287163

-- Definitions for the problem
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def total_balls : ℕ := red_balls + white_balls
def prob_red : ℚ := red_balls / total_balls
def trials : ℕ := 4
def ξ : dist Binomial trials prob_red := sorry

-- Expected value for binomial distribution
theorem expected_value : E(ξ) = 12/5 := 
by solve_by_elim [ξ]

end expected_value_l287_287163


namespace Niklaus_walked_distance_l287_287729

noncomputable def MilesToFeet (miles : ℕ) : ℕ := miles * 5280
noncomputable def YardsToFeet (yards : ℕ) : ℕ := yards * 3

theorem Niklaus_walked_distance (n_feet : ℕ) :
  MilesToFeet 4 + YardsToFeet 975 + n_feet = 25332 → n_feet = 1287 := by
  sorry

end Niklaus_walked_distance_l287_287729


namespace derivative_at_2_l287_287354

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287354


namespace correctly_differentiated_l287_287839

-- Define the functions and their expected derivatives
def f1 := (fun x => x)
def f2 := (fun x => sin 2)
def f3 := (fun x => 1 / x)
def f4 := (fun x => sqrt x)

-- Define expected derivatives
def df1 := (fun x' => 1)
def df2 := (fun x' => 0)
def df3 := (fun x' => - 1 / x^2)
def df4 := (fun x' => 1 / (2 * sqrt x))

theorem correctly_differentiated :
  (derivative f1 == df1) ∧
  (derivative f4 == df4) ∧
  ¬ (derivative f2 == df2) ∧
  ¬ (derivative f3 == df3) := by
  sorry

end correctly_differentiated_l287_287839


namespace solution_set_of_quadratic_inequality_l287_287653

theorem solution_set_of_quadratic_inequality 
  (a b c x₁ x₂ : ℝ)
  (h1 : a > 0) 
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  : {x : ℝ | a * x^2 + b * x + c > 0} = ({x : ℝ | x > x₁} ∩ {x : ℝ | x > x₂}) ∪ ({x : ℝ | x < x₁} ∩ {x : ℝ | x < x₂}) :=
sorry

end solution_set_of_quadratic_inequality_l287_287653


namespace sequence_eventually_constant_iff_perfect_square_l287_287570

def S (n : ℕ) : ℕ :=
  let m := nat.sqrt n
  n - m^2

def sequence (A : ℕ) : ℕ → ℕ
| 0     := A
| (k+1) := sequence k + S (sequence k)

theorem sequence_eventually_constant_iff_perfect_square (A : ℕ) :
  (∃ m : ℕ, A = m^2) ↔ ∃ N, ∀ n, N ≤ n → sequence A n = sequence A N :=
sorry

end sequence_eventually_constant_iff_perfect_square_l287_287570


namespace some_value_correct_l287_287612

theorem some_value_correct (w x y : ℝ) (some_value : ℝ)
  (h1 : 3 / w + some_value = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  some_value = 6 := by
  sorry

end some_value_correct_l287_287612


namespace magnitude_of_b_cos_theta_l287_287575

def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_of_b : real.sqrt ((-1)^2 + 2^2) = real.sqrt 5 := 
by
  show real.sqrt (1 + 4) = real.sqrt 5
  sorry

theorem cos_theta :
  let a_dot_b := (4 * (-1) + 3 * 2)
  let magnitude_a := real.sqrt (4^2 + 3^2)
  let magnitude_b := real.sqrt ((-1)^2 + 2^2)
  cos_theta :=
    a_dot_b / (magnitude_a * magnitude_b) = 2 * real.sqrt 5 / 25 :=
by
  show 2 / (5 * real.sqrt 5) = 2 * real.sqrt 5 / 25
  sorry

end magnitude_of_b_cos_theta_l287_287575


namespace find_natural_number_with_common_divisor_l287_287520

def commonDivisor (a b : ℕ) (d : ℕ) : Prop :=
  d > 1 ∧ d ∣ a ∧ d ∣ b

theorem find_natural_number_with_common_divisor :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k ≤ 20 →
    ∃ d : ℕ, commonDivisor (n + k) 30030 d) ∧ n = 9440 :=
by
  sorry

end find_natural_number_with_common_divisor_l287_287520


namespace tangent_circles_x_intersect_l287_287872

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287872


namespace value_of_six_prime_prime_l287_287613

-- Define the function q' 
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Stating the main theorem we want to prove
theorem value_of_six_prime_prime : prime (prime 6) = 42 :=
by
  sorry

end value_of_six_prime_prime_l287_287613


namespace lambda_value_l287_287597

noncomputable def vector (α : Type*) [Add α] := α

variables {α : Type*} 
variables [Add α] [Neg α] [Mul α ℝ] [Sub α] [HasSmul ℝ α]

variables (A B C D : vector α) (λ : ℝ)

theorem lambda_value (h : D = A + (λ • (B - A))) 
  (hCD : C - D = (1/3 : ℝ) • (C - A) + λ • (B - C)) : 
  λ = -4/3 :=
  sorry

end lambda_value_l287_287597


namespace selling_price_l287_287428

theorem selling_price (purchase_price : ℝ) (overhead_expenses : ℝ) (profit_percent : ℝ) : (purchase_price = 225) → (overhead_expenses = 15) → (profit_percent = 45.833333333333314) → 
  let total_cost_price := purchase_price + overhead_expenses in
  let profit_amount := (profit_percent / 100) * total_cost_price in
  let selling_price := total_cost_price + profit_amount in
  selling_price = 350 := by
  sorry

end selling_price_l287_287428


namespace simplify_trig_expression_l287_287765

theorem simplify_trig_expression (x z : ℝ) :
  sin x ^ 2 + sin (x + z) ^ 2 - 2 * sin x * sin z * sin (x + z) = sin z ^ 2 :=
by sorry

end simplify_trig_expression_l287_287765


namespace alternating_sum_cubes_eval_l287_287980

noncomputable def alternating_sum_cubes : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum_cubes n + (-1)^(n / 4) * (n + 1)^3

theorem alternating_sum_cubes_eval :
  alternating_sum_cubes 99 = S :=
by
  sorry

end alternating_sum_cubes_eval_l287_287980


namespace distance_between_parallel_sides_l287_287987

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end distance_between_parallel_sides_l287_287987


namespace minimum_fence_posts_needed_l287_287425

theorem minimum_fence_posts_needed 
  (playground : Type)
  (dimensions : ℕ × ℕ)
  (rock_wall_length : ℕ)
  (post_spacing : ℕ) 
  (rock_wall_on_long_side : bool) 
  (corner_posts_included : bool) :
  dimensions = (50, 100) →
  rock_wall_length = 100 →
  post_spacing = 10 →
  rock_wall_on_long_side = true →
  corner_posts_included = true →
  ∑ p in {11, 5, 5}, p = 21 :=
by
  intros dim100 rwlen100 spacing10 rwl true crn true
  sorry

end minimum_fence_posts_needed_l287_287425


namespace min_value_of_z_l287_287831

theorem min_value_of_z : ∃ (x y : ℝ), ∀ z : ℝ, z = x^2 + 2 * y^2 + 6 * x - 4 * y + 22 → z ≥ 11 :=
begin
  sorry
end

end min_value_of_z_l287_287831


namespace grapes_and_pineapple_cost_l287_287704

variable (o g p s : ℕ)

-- Given conditions
def condition1 : Prop := o + g + p + s = 24
def condition2 : Prop := 2 * o = s
def condition3 : Prop := o - g = p

-- The theorem to prove
theorem grapes_and_pineapple_cost :
  condition1 o g p s → 
  condition2 o g p s → 
  condition3 o g p s → 
  g + p = 6 :=
by
  intro hc1 hc2 hc3
  sorry

end grapes_and_pineapple_cost_l287_287704


namespace find_f_at_1_l287_287579

noncomputable def f : ℝ → ℝ :=
λ x, if x < 2 then f (x + 2) else 2^(-x)

theorem find_f_at_1 : f 1 = 1 / 8 :=
by
  -- Proof goes here
  sorry

end find_f_at_1_l287_287579


namespace shadow_length_is_multiple_of_table_height_l287_287948

noncomputable def shadow_length_multiple (h : ℝ) (α β : ℝ) (tan_alpha : ℝ) (tan_alpha_minus_beta : ℝ) : ℝ :=
  h * (tan_beta tan_alpha tan_alpha_minus_beta)

lemma tan_beta (tan_alpha : ℝ) (tan_alpha_minus_beta : ℝ) : ℝ :=
  (tan_alpha - tan_alpha_minus_beta) / (1 + tan_alpha * tan_alpha_minus_beta)

theorem shadow_length_is_multiple_of_table_height
  (h : ℝ) (α β : ℝ)
  (h_gt_zero : h > 0)
  (tan_alpha : ℝ)
  (tan_alpha_minus_beta_reciprocal : ℝ)
  (h_eq_tan : ∀ θ, tan θ = sin θ / cos θ)
  (tan_alpha_eq_3 : tan α = 3) 
  (tan_alpha_minus_beta_eq_one_third : tan (α - β) = 1 / 3) :
  shadow_length_multiple h α β tan_alpha tan_alpha_minus_beta_reciprocal = 4 / 3 :=
by
  sorry

end shadow_length_is_multiple_of_table_height_l287_287948


namespace sum_first_six_terms_geometric_sequence_l287_287995

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l287_287995


namespace spending_correct_l287_287774

def terry_total : ℝ :=
  let monday := 6
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  let thursday := (monday + tuesday + wednesday) / 3
  let friday := thursday - 4
  let saturday := 1.5 * friday
  let sunday := tuesday + saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

def maria_total : ℝ :=
  let monday := 6 / 2
  let tuesday := 10
  let wednesday := 2 * 36
  let thursday := 8
  let friday := 14
  let saturday := 12
  let sunday := 33
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

def raj_total : ℝ :=
  let monday := 6 * 1.25
  let tuesday := 10
  let wednesday := 3 * 72
  let thursday := wednesday / 2
  let friday := 14
  let saturday := 21
  let sunday := 4 * saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

def total_spending : ℝ :=
  terry_total + maria_total + raj_total

theorem spending_correct : total_spending = 752.50 := by
  sorry  -- Proof goes here

end spending_correct_l287_287774


namespace minimize_distance_achieved_l287_287588

noncomputable def minimize_distance (ABC : Type) [triangle ABC] 
  (A B C D E F: ABC)
  (h₁ : acute_angle B C A)
  (h₂ : on_segment D A B)
  (h₃ : perpendicular D E A C)
  (h₄ : perpendicular D F B C) : Prop :=
  minimizes_distance E F (altitude D C A B)

theorem minimize_distance_achieved (ABC : Type) [triangle ABC] 
  (A B C D E F: ABC) :
  ∀ 
    (h₁ : acute_angle B C A) 
    (h₂ : on_segment D A B) 
    (h₃ : perpendicular D E A C) 
    (h₄ : perpendicular D F B C),
    minimize_distance ABC A B C D E F h₁ h₂ h₃ h₄ :=
by sorry

end minimize_distance_achieved_l287_287588


namespace positive_numbers_correct_fractions_correct_negative_integers_correct_negative_rational_numbers_correct_l287_287062

-- Define the number set
def number_set : Set ℚ := {
  22 / 7, -1 / 3, -1, -1.04, 2, 5, 3, 3.1415, -8
}

-- Define each target set
def positive_numbers_target : Set ℚ := {
  22 / 7, 2, 5, 3, 3.1415
}

def fractions_target : Set ℚ := {
  22 / 7, -1 / 3, -1.04, 3.1415
}

def negative_integers_target : Set ℤ := {
  -1, -8
}

def negative_rational_numbers_target : Set ℚ := {
  -1 / 3, -1, -1.04, -8
}

-- Proofs
theorem positive_numbers_correct :
  { x ∈ number_set | x > 0 } = positive_numbers_target := 
sorry

theorem fractions_correct :
  { x ∈ number_set | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b } = fractions_target := 
sorry

theorem negative_integers_correct :
  { x ∈ number_set | ∃ (n : ℤ), x = n ∧ n < 0 } = negative_integers_target := 
sorry

theorem negative_rational_numbers_correct :
  { x ∈ number_set | x < 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b } = negative_rational_numbers_target :=
sorry

end positive_numbers_correct_fractions_correct_negative_integers_correct_negative_rational_numbers_correct_l287_287062


namespace minimum_value_of_expression_l287_287200

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ c : ℝ, (c = a^2 + 1 / (b * (a - b))) ∧ c = 4 :=
by {
  have h_pos : b * (a - b) > 0 := sorry, -- This acknowledges b > 0 and a - b > 0
  have min_val : ∀ x y : ℝ, (x > 0 → y > 0 → x + 1 / y ≥ 4) := sorry,
  use a^2 + 1 / (b * (a - b)),
  split,
  { refl, },
  { exact min_val a b h1 h2 },
}

end minimum_value_of_expression_l287_287200


namespace problem1_eval_problem2_eval_l287_287856

-- Problem 1
theorem problem1_eval : 
  - (-2 : ℝ) ^ 4 + (-2) ^ (-3 : ℝ) + (- (1/2) : ℝ) ^ (-3 : ℝ) - (- (1/2)) ^ 3 = -24 := by
  sorry

-- Problem 2
theorem problem2_eval : 
  real.log 14 - 2 * real.log (7 / 3) + real.log 7 - real.log 18 = 0 := by
  sorry

end problem1_eval_problem2_eval_l287_287856


namespace regular_pyramid_l287_287430

variables {P A B C O : Type*} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]

noncomputable theory
def radius : ℝ := sorry -- given radius r

def distance_center_to_apex : ℝ := radius * Real.sqrt 3 -- given distance r * sqrt(3)

-- Given conditions
axiom sphere_touches_all_edges (P A B C O : Type*) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] (pyramid : Triangle P A B C) (sphere : Sphere O radius) : touches_all_edges sphere pyramid

axiom center_on_height (P O : Type*) [MetricSpace P] [MetricSpace O] (height : Line P O) (sphere : Sphere O radius) : on_height sphere height

axiom center_distance (O P : Type*) [MetricSpace O] [MetricSpace P] (center : Point O) (apex : Point P) : dist center apex = distance_center_to_apex

-- Objectives to prove
theorem regular_pyramid (P A B C O : Type*) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] (sphere : Sphere O radius) (pyramid : Triangle P A B C) (center : Point O) (height : Line P O) :
  touches_all_edges sphere pyramid →
  on_height sphere height →
  dist center (Point P) = distance_center_to_apex →
  is_regular_tetrahedron pyramid ∧ height_length pyramid = (1 / 3) * radius * Real.sqrt 3 :=
begin
  sorry
end

end regular_pyramid_l287_287430


namespace angle_C_pi_div_3_l287_287691

theorem angle_C_pi_div_3 (a b c : ℝ) (h : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) 
  (h_area : (√3 / 4) * (a^2 + b^2 - c^2) = 1 / 2 * a * b * real.sin (real.arcsin (real.sqrt 3 / 2))) :
  real.arccos (1 / 2) = π / 3 :=
by
  sorry

end angle_C_pi_div_3_l287_287691


namespace route_difference_is_18_l287_287216

theorem route_difference_is_18 :
  let t_uphill := 6 in
  let t_path1 := 2 * t_uphill in
  let t_final1 := (t_uphill + t_path1)/3 in
  let t_route1 := t_uphill + t_path1 + t_final1 in
  let t_flat := 14 in
  let t_final2 := 2 * t_flat in
  let t_route2 := t_flat + t_final2 in
  t_route2 - t_route1 = 18 :=
by
  let t_uphill := 6
  let t_path1 := 2 * t_uphill
  let t_final1 := (t_uphill + t_path1)/3
  let t_route1 := t_uphill + t_path1 + t_final1
  let t_flat := 14
  let t_final2 := 2 * t_flat
  let t_route2 := t_flat + t_final2
  have h : t_route2 - t_route1 = 18
  sorry

end route_difference_is_18_l287_287216


namespace tangent_line_intersection_l287_287882

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287882


namespace max_value_at_one_l287_287317

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287317


namespace varlimsup_char_func_l287_287854

-- Define the characteristic function for an arbitrary discrete random variable ξ
noncomputable def char_func (ξ : ℕ → ℝ) (p : ℕ → ℝ) (t : ℝ) : ℂ :=
  ∑ k in finset.range (ξ.length), p k * complex.exp (complex.I * (ξ k * t))

-- Define the Fejér kernel
noncomputable def fejér_kernel (n : ℕ) (t : ℝ) : ℂ :=
  (1 / n) * ∑ k in finset.range n, ∑ j in finset.Icc (-k) k, complex.exp (complex.I * (j * t))

-- Define the product K_N(t, n)
noncomputable def K_N (t : ℝ) (n N : ℕ) (x : ℕ → ℝ) : ℂ :=
  ∏ k in finset.range N, fejér_kernel n (x k * t)

-- Hypotheses of the problem
variables 
  (ξ : ℕ → ℝ) (p : ℕ → ℝ) 
  (p_nonneg : ∀ k, p k ≥ 0)
  (p_sum : ∑ k in finset.range (ξ.length), p k = 1)

-- Main theorem statement
theorem varlimsup_char_func :
  ∀ (t : ℝ) (N : ℕ) (n : ℕ), 
    (∃ T : ℝ, ∫ t in (0 : ℝ)..T, char_func ξ p t * conj (K_N t n N ξ) / T ≥ p 0 + (n - 1) / n * ∑ k in finset.range N, p k) →
    ∃ f : ℝ → ℂ, filter.limsup (filter.at_top.map f) = 1 := 
sorry

end varlimsup_char_func_l287_287854


namespace dark_squares_more_than_light_l287_287863

/--
A 9 × 9 board is composed of alternating dark and light squares, with the upper-left square being dark.
Prove that there is exactly 1 more dark square than light square.
-/
theorem dark_squares_more_than_light :
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  dark_squares - light_squares = 1 :=
by
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  show dark_squares - light_squares = 1
  sorry

end dark_squares_more_than_light_l287_287863


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287555

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287555


namespace vova_last_grades_l287_287297

theorem vova_last_grades (grades : Fin 19 → ℕ) 
  (first_four_2s : ∀ i : Fin 4, grades i = 2)
  (all_combinations_once : ∀ comb : Fin 4 → ℕ, 
    (∃ (start : Fin (19-3)), ∀ j : Fin 4, grades (start + j) = comb j) ∧
    (∀ i j : Fin (19-3), 
      (∀ k : Fin 4, grades (i + k) = grades (j + k)) → i = j)) :
  ∀ i : Fin 4, grades (15 + i) = if i = 0 then 3 else 2 :=
by
  sorry

end vova_last_grades_l287_287297


namespace series_sum_eq_l287_287969

noncomputable def sum_series : ℝ :=
  ∑ n in finset.range 50, (1 + 3 * (n + 1)) / (9^((50 - n) + 1))

theorem series_sum_eq : 
  sum_series = 18.75 :=
sorry

end series_sum_eq_l287_287969


namespace cubic_parabola_has_one_x_intercept_l287_287502

theorem cubic_parabola_has_one_x_intercept :
  let f (y : ℝ) := -3 * y^3 + 2 * y^2 - y + 2 in
  set.count (set_of (λ y, f y = 0)) = 1 :=
by
  sorry

end cubic_parabola_has_one_x_intercept_l287_287502


namespace range_of_a_l287_287093

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := -(x + 1) ^ 2 + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 ∈ Set.Icc (-2:ℝ) 0, f x2 ≤ g x1 a) → a ≥ -1 / Real.exp 1 :=
by
  -- Definitions for clarity
  let f_min := f (-1)
  let g_max := g (-1) a
  have h1 : f_min = -1 / Real.exp 1 := by
    sorry
  have h2 : g_max = a := by
    sorry
  intros h
  -- Use the given existential condition
  rcases h with ⟨x1, hx1, x2, hx2, hfg⟩
  -- Use the calculated minimum and maximum
  rw [← h1, ← h2] at hfg
  exact hfg

end range_of_a_l287_287093


namespace factors_of_2550_have_more_than_3_factors_l287_287633

theorem factors_of_2550_have_more_than_3_factors :
  ∃ n: ℕ, n = 5 ∧
    ∃ d: ℕ, d = 2550 ∧
    (∀ x < n, ∃ y: ℕ, y ∣ d ∧ (∃ z, z ∣ y ∧ z > 3)) :=
sorry

end factors_of_2550_have_more_than_3_factors_l287_287633


namespace toys_left_after_two_weeks_l287_287456

theorem toys_left_after_two_weeks
  (initial_stock : ℕ)
  (sold_first_week : ℕ)
  (sold_second_week : ℕ)
  (total_stock : initial_stock = 83)
  (first_week_sales : sold_first_week = 38)
  (second_week_sales : sold_second_week = 26) :
  initial_stock - (sold_first_week + sold_second_week) = 19 :=
by
  sorry

end toys_left_after_two_weeks_l287_287456


namespace range_x_for_p_range_a_for_q_l287_287106

variable p : Prop
variable q : Prop

def p_def (x : ℝ) := x^2 ≤ 5 * x - 4
def q_def (x a : ℝ) := x^2 - (a + 2) * x + 2 * a ≤ 0

theorem range_x_for_p : (∀ x, p_def x → 1 ≤ x ∧ x ≤ 4) := sorry

theorem range_a_for_q (h : ∀ x, q_def x a → p_def x) : 1 ≤ a ∧ a ≤ 4 := sorry

end range_x_for_p_range_a_for_q_l287_287106


namespace percentage_of_boys_is_60_percent_l287_287955

-- Definition of the problem conditions
def totalPlayers := 50
def juniorGirls := 10
def half (n : ℕ) := n / 2
def girls := 2 * juniorGirls
def boys := totalPlayers - girls
def percentage_of_boys := (boys * 100) / totalPlayers

-- The theorem stating the proof problem
theorem percentage_of_boys_is_60_percent : percentage_of_boys = 60 := 
by 
  -- Proof omitted
  sorry

end percentage_of_boys_is_60_percent_l287_287955


namespace playground_total_l287_287283

def boys : ℕ := 44
def girls : ℕ := 53

theorem playground_total : boys + girls = 97 := by
  sorry

end playground_total_l287_287283


namespace train_problem_l287_287844

def length_of_other_train (length_first : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  let speed1_mps := (speed1 * 1000) / 3600
  let speed2_mps := (speed2 * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  total_distance - length_first

theorem train_problem
  (length_first : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (time : ℝ)
  (length_other : ℝ)
  (h1 : length_first = 220)
  (h2 : speed1 = 120)
  (h3 : speed2 = 80)
  (h4 : time = 9)
  (h5 : length_other = 279.95) :
  length_of_other_train length_first speed1 speed2 time = length_other :=
by
  rw [h1, h2, h3, h4, h5]
  dsimp [length_of_other_train]
  sorry

end train_problem_l287_287844


namespace min_dot_product_l287_287604

-- Define the conditions of the ellipse and focal points
variables (P : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define vectors
def OP (P : ℝ × ℝ) : ℝ × ℝ := P
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Prove that the minimum value of the dot product is 2
theorem min_dot_product (hP : ellipse P.1 P.2) : 
  ∃ (P : ℝ × ℝ), dot_product (OP P) (FP P) = 2 := sorry

end min_dot_product_l287_287604


namespace irrational_number_l287_287378

-- Definitions
def sqrt_6 : ℝ := Real.sqrt 6
def cbrt_27 : ℝ := Real.cbrt 27
def frac_22_7 : ℝ := 22 / 7
def decimal_pi : ℝ := 3.1415926

-- Proposition
theorem irrational_number (h₁: ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 6 = p / q)
  (h₂: ∃ (p' q' : ℤ), q' ≠ 0 ∧ Real.cbrt 27 = p' / q')
  (h₃: ∃ (p'' q'' : ℤ), q'' ≠ 0 ∧ 22 / 7 = p'' / q'')
  (h₄: ∃ (p''' q''' : ℤ), q''' ≠ 0 ∧ 3.1415926 = p''' / q''') :
  ∀ x ∈ ({sqrt_6, cbrt_27, frac_22_7, decimal_pi} : set ℝ),
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q ↔ x = sqrt_6 :=
by
  sorry

end irrational_number_l287_287378


namespace line_length_limit_l287_287492

noncomputable def geometric_sum (a r : Real) (n : ℕ): Real :=
  if r = 1 then a * (n + 1) else a * (1 - r ^ (n + 1)) / (1 - r)

noncomputable def infinite_geometric_sum (a r : Real): Real :=
  if r < 1 then a / (1 - r) else 0

theorem line_length_limit : 
  lim_seq (λ n : ℕ, 2 + ∑ i in range n, (1 / 5) ^ i * sqrt 3 + ∑ i in range n, (1 / 5) ^ i) = (10 + sqrt 3) / 4 :=
by
  sorry

end line_length_limit_l287_287492


namespace x_divisible_by_5_l287_287591

theorem x_divisible_by_5 (x y : ℕ) (hx : x > 1) (h : 2 * x^2 - 1 = y^15) : 5 ∣ x := 
sorry

end x_divisible_by_5_l287_287591


namespace min_length_XY_ratio_one_l287_287183

theorem min_length_XY_ratio_one 
  (C1 C2 C3 : circle)
  (A B O : point)
  (AB : length A B = 1)
  (k : ℝ)
  (k_pos : 1 < k ∧ k < 3)
  (O_eq_center_C1 : O = center C1)
  (O_eq_center_C2 : O = center C2)
  (diam_C2 : diameter C2 = k)
  (A_eq_center_C3 : A = center C3)
  (diam_C3 : diameter C3 = 2 * k)
  (X Y : point)
  (X_on_C2 : X ∈ circle_points C2)
  (Y_on_C3 : Y ∈ circle_points C3)
  (B_on_XY : B ∈ line_segment X Y) :
  ∃ r : ℝ, r = 1 ∧ length (line_segment X Y) = min_length_XY k

end min_length_XY_ratio_one_l287_287183


namespace common_point_exists_l287_287424

-- Definitions and conditions based on the given problem
def A : Point := sorry
def ray1 : Ray := sorry
def ray2 : Ray := sorry
def a : ℝ := sorry
def B : Circle.intersect_point := sorry
def C : Circle.intersect_point := sorry

-- Given conditions
lemma given_positive_real (h : a > 0) : ℝ := 
sorry

lemma given_intersect_point (hB : B ≠ A) (hC : C ≠ A) : Point :=
sorry

lemma given_circle_relation (hR : (circle B A C).radius = (|B - A| + |C - A|) / 2) : ℝ :=
sorry

-- Proof that all such circles pass through a common point other than \( A \)
theorem common_point_exists (ha : given_positive_real a) (hpoints : given_intersect_point A B C) (h_relation : given_circle_relation B A C) :
∃ Z : Point, Z ≠ A ∧ ∀ (circle : Circle), circle B A C :=
sorry

end common_point_exists_l287_287424


namespace population_reaches_capacity_l287_287176

def max_capacity (total_acres : ℕ) (acres_per_person : ℕ) : ℕ :=
  total_acres / acres_per_person

def population_growth (initial_population years elapsed time_period multiplier : ℕ) : ℕ :=
  initial_population * multiplier^(years elapsed / time_period)

theorem population_reaches_capacity :
  ∀ (total_acres : ℕ) (acres_per_person : ℕ) (initial_population : ℕ) (time_period : ℕ) (multiplier : ℕ) (start_year : ℕ) (target_year : ℕ),
  total_acres = 36000 →
  acres_per_person = 2 →
  initial_population = 300 →
  time_period = 30 →
  multiplier = 4 →
  start_year = 2040 →
  target_year = start_year + 90 →
  population_growth initial_population (target_year - start_year) time_period multiplier ≥ max_capacity total_acres acres_per_person :=
by
  intros total_acres acres_per_person initial_population time_period multiplier start_year target_year
  sorry

end population_reaches_capacity_l287_287176


namespace common_difference_arithmetic_sequence_l287_287589

noncomputable def a_n (n : ℕ) : ℤ := 5 - 4 * n

theorem common_difference_arithmetic_sequence :
  ∀ n ≥ 1, a_n n - a_n (n - 1) = -4 :=
by
  intros n hn
  unfold a_n
  sorry

end common_difference_arithmetic_sequence_l287_287589


namespace max_element_l287_287018

theorem max_element (-5 0 3 (1 / 3) : ℝ) : 
  max (max (max (-5) 0) (1 / 3)) 3 = 3 :=
sorry

end max_element_l287_287018


namespace polyhedron_volume_l287_287491

-- Define the polygons
structure Polygon :=
  (shape : Type)
  (size : ℕ)

-- Define specific instances for polygons
def H : Polygon := ⟨isoscelesRightTriangle, 2⟩
def I : Polygon := ⟨isoscelesRightTriangle, 2⟩
def J : Polygon := ⟨isoscelesRightTriangle, 2⟩

def K : Polygon := ⟨square, 2⟩
def L : Polygon := ⟨square, 2⟩
def M : Polygon := ⟨square, 2⟩

def N : Polygon := ⟨regularHexagon, 2⟩

-- The polyhedron formed by the folding process
noncomputable def polyhedron := ⟨ [H, I, J, K, L, M], N ⟩

-- The theorem to prove the volume of the polyhedron
theorem polyhedron_volume : volume polyhedron = 20 / 3 := 
  sorry

end polyhedron_volume_l287_287491


namespace find_f_prime_at_2_l287_287341

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287341


namespace simplify_radical_l287_287958

theorem simplify_radical (p : ℝ) (hp1 : 0 ≤ 15 * p^2) (hp2 : 0 ≤ 8 * p) (hp3 : 0 ≤ 27 * p^5) :
  sqrt (15 * p^2) * sqrt (8 * p) * sqrt (27 * p^5) = 18 * p^4 * sqrt 10 := 
sorry

end simplify_radical_l287_287958


namespace ribbon_cost_l287_287568

variable (c_g c_m s : ℝ)

theorem ribbon_cost (h1 : 5 * c_g + s = 295) (h2 : 7 * c_m + s = 295) (h3 : 2 * c_m + c_g = 102) : s = 85 :=
sorry

end ribbon_cost_l287_287568


namespace infinite_continued_fraction_eq_l287_287999

noncomputable def infinite_continued_fraction : ℝ :=
  1 + Real.continued_fraction 2 [1, 2].cycle  -- This definition assumes the proper utility functions for continued fractions are in Mathlib.

theorem infinite_continued_fraction_eq :
  infinite_continued_fraction = (Real.sqrt 3 + 1) / 2 := sorry

end infinite_continued_fraction_eq_l287_287999


namespace cost_price_is_50_l287_287922

-- Define the conditions
def selling_price : ℝ := 80
def profit_rate : ℝ := 0.6

-- The cost price should be proven to be 50
def cost_price (C : ℝ) : Prop :=
  selling_price = C + (C * profit_rate)

theorem cost_price_is_50 : ∃ C : ℝ, cost_price C ∧ C = 50 := by
  sorry

end cost_price_is_50_l287_287922


namespace coeff_x3_in_2x_plus_sqrt_x_pow_5_l287_287975

theorem coeff_x3_in_2x_plus_sqrt_x_pow_5 : 
  let f := (λ x : ℝ, (2 * x + real.sqrt x)^5) in 
  polynomial.coeff (f polynomial.X) 3 = 10 :=
by
  sorry

end coeff_x3_in_2x_plus_sqrt_x_pow_5_l287_287975


namespace value_c_plus_d_squared_l287_287273

theorem value_c_plus_d_squared : 
  let a := 5
  let b := -2
  let c := 8
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := real.sqrt (-discriminant)
  let x1 := (-b + sqrt_discriminant * complex.i) / (2 * a)
  let x2 := (-b - sqrt_discriminant * complex.i) / (2 * a)
  let c := 1 / 5
  let d := (real.sqrt 156) / 10
  (c + d^2 = 44 / 25) := sorry

end value_c_plus_d_squared_l287_287273


namespace tangent_circles_x_intersect_l287_287870

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287870


namespace number_of_correct_conclusions_l287_287140

-- Definitions and assumptions
variables {r a b x y : ℝ} (h_r : r > 0)
variables {x1 y1 x2 y2 : ℝ}
def C1 (x y : ℝ) := x^2 + y^2 = r^2
def C2 (x y : ℝ) := (x - a)^2 + (y - b)^2 = r^2
def A := C1 x1 y1 ∧ C2 x1 y1
def B := C1 x2 y2 ∧ C2 x2 y2

theorem number_of_correct_conclusions
  (hA : A)
  (hB : B) :
  (a * (x1 - x2) + b * (y1 - y2) = 0) ∧
  (2 * a * x1 + 2 * b * y1 = a^2 + b^2) ∧
  (x1 + x2 = a) ∧ (y1 + y2 = b) :=
begin
  sorry
end

end number_of_correct_conclusions_l287_287140


namespace erased_number_is_214_l287_287473

theorem erased_number_is_214 {a b : ℤ} 
  (h1 : 9 * a = sum (a - 4 :: a - 3 :: a - 2 :: a - 1 :: a :: a + 1 :: a + 2 :: a + 3 :: a + 4 :: []))
  (h2 : ∑ n in {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4} \ erase_val (a + b) {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4}, n = 1703)
  (h3 : -4 ≤ b ∧ b ≤ 4)
  : a + b = 214 :=
begin
  -- proof to be filled in
  sorry,
end

end erased_number_is_214_l287_287473


namespace range_of_a_l287_287650

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, sin x^2 + cos x + a = 0) → a ∈ Icc (-5/4 : ℝ) 1 :=
by
  sorry

end range_of_a_l287_287650


namespace small_palindrome_check_l287_287566

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287566


namespace distance_is_18_l287_287000

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  let faster := (x + 1) * (3 * t / 4) = d
  let slower := (x - 1) * (t + 3) = d
  let normal := x * t = d
  faster ∧ slower ∧ normal

theorem distance_is_18 : 
  ∃ (x t : ℝ), distance_walked x t 18 :=
by
  sorry

end distance_is_18_l287_287000


namespace smaller_rectangle_perimeter_l287_287970

variables (a b c : ℝ)

-- Conditions
def larger_rectangle_dimensions (a b : ℝ) : Prop := a = 3 * c ∧ b = b
def smaller_rectangle_dimensions (c b : ℝ) : Prop := c = c ∧ b = b
def side_of_square (c : ℝ) : Prop := c = c

theorem smaller_rectangle_perimeter 
  (h1 : larger_rectangle_dimensions a b)
  (h2 : smaller_rectangle_dimensions c b)
  (h3 : side_of_square c) :
  2 * (c + b) = 2 * (c + b) :=
by
  sorry

end smaller_rectangle_perimeter_l287_287970


namespace valid_program_combinations_l287_287932

theorem valid_program_combinations :
  let courses := ['English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science]
  let math_courses := ['Algebra, 'Geometry]
  (comb_count : Nat → Nat → Nat) =
    (choose 5 3 - choose 3 3) → 9 := 
by
  sorry

end valid_program_combinations_l287_287932


namespace hyperbola_eccentricity_l287_287209

theorem hyperbola_eccentricity 
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c^2 = a^2 + b^2)
  (h4 : ∀ l: ℝ, ∀ l: ℝ, dist (0, 0) l = (sqrt 3 / 4) * c) :
  (∃ e : ℝ, e = 2 ∨ e = 2 * sqrt 3 / 3) := 
sorry

end hyperbola_eccentricity_l287_287209


namespace p_implies_q_l287_287105

variables {m : ℝ} {a x : ℝ}

def p : Prop := (m + 1) * (m - 2) = 0
def q : Prop := m = 2 ∧ a > 0 ∧ a ≠ 1

theorem p_implies_q : p → q :=
by
    sorry

end p_implies_q_l287_287105


namespace rail_elevation_correct_angle_l287_287013

noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) (g : ℝ) : ℝ :=
  Real.arctan (v^2 / (R * g))

theorem rail_elevation_correct_angle :
  rail_elevation_angle (60 * (1000 / 3600)) 200 9.8 = 8.09 := by
  sorry

end rail_elevation_correct_angle_l287_287013


namespace find_initial_flour_l287_287733

def initialFlour (x : ℕ) : Prop :=
  x + 2 = 10

theorem find_initial_flour : ∃ x : ℕ, initialFlour x ∧ x = 8 :=
by
  use 8
  split
  . exact rfl
  . exact rfl

end find_initial_flour_l287_287733


namespace ratio_of_cost_to_marked_price_l287_287001

variables (x : ℝ)
def marked_price := x
def discount := (2 / 5) * x
def selling_price := x - discount
def actual_selling_price := (3 / 5) * x
def cost_price := (4 / 5) * actual_selling_price
def ratio := cost_price / marked_price

theorem ratio_of_cost_to_marked_price : ratio = 12 / 25 := by 
  sorry

end ratio_of_cost_to_marked_price_l287_287001


namespace radius_ratio_eq_inv_sqrt_5_l287_287679

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l287_287679


namespace find_f_prime_at_two_l287_287330

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287330


namespace probability_of_at_least_one_three_l287_287020

noncomputable def probability_at_least_one_three_tossed : ℚ :=
  10 / 21

theorem probability_of_at_least_one_three 
  (X1 X2 X3 : ℕ) 
  (h1 : 1 ≤ X1 ∧ X1 ≤ 8)
  (h2 : 1 ≤ X2 ∧ X2 ≤ 8)
  (h3 : 1 ≤ X3 ∧ X3 ≤ 8)
  (cond : X1 + X2 = X3 + 1) :
  let event_has_three := (X1 = 3 ∨ X2 = 3 ∨ X3 = 3) in
  (event_has_three ↔ true) → 
  probability_at_least_one_three_tossed = 10 / 21 :=
by sorry

end probability_of_at_least_one_three_l287_287020


namespace Joe_catches_l287_287696

variable (J : ℝ)

def Derek_catches : ℝ := 2 * J - 4

def Tammy_catches : ℝ := (1 / 3) * Derek_catches J + 16

theorem Joe_catches (h : Tammy_catches J = 30) : J = 23 := by
  sorry

end Joe_catches_l287_287696


namespace max_value_at_one_l287_287316

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287316


namespace min_games_to_achieve_98_percent_l287_287254

-- Define initial conditions
def initial_games : ℕ := 5
def initial_sharks_wins : ℕ := 2
def initial_tigers_wins : ℕ := 3

-- Define the total number of games and the total number of wins by the Sharks after additional games
def total_games (N : ℕ) : ℕ := initial_games + N
def total_sharks_wins (N : ℕ) : ℕ := initial_sharks_wins + N

-- Define the Sharks' winning percentage
def sharks_winning_percentage (N : ℕ) : ℚ := total_sharks_wins N / total_games N

-- Define the minimum number of additional games needed
def minimum_N : ℕ := 145

-- Theorem: Prove that the Sharks' winning percentage is at least 98% when N = 145
theorem min_games_to_achieve_98_percent :
  sharks_winning_percentage minimum_N ≥ 49 / 50 :=
sorry

end min_games_to_achieve_98_percent_l287_287254


namespace value_of_p_l287_287103

theorem value_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) : p = -2278 :=
by
  sorry

end value_of_p_l287_287103


namespace complex_number_equation_l287_287649

theorem complex_number_equation
  (f : ℂ → ℂ)
  (z : ℂ)
  (h : f (i - z) = 2 * z - i) :
  (1 - i) * f (2 - i) = -1 + 7 * i := by
  sorry

end complex_number_equation_l287_287649


namespace max_value_at_one_l287_287321

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287321


namespace amoeba_growth_one_week_l287_287768

theorem amoeba_growth_one_week :
  (3 ^ 7 = 2187) :=
by
  sorry

end amoeba_growth_one_week_l287_287768


namespace determine_function_l287_287050

noncomputable def functional_solution (f : ℝ → ℝ) : Prop := 
  ∃ (C₁ C₂ : ℝ), ∀ (x : ℝ), 0 < x → f x = C₁ * x + C₂ / x 

theorem determine_function (f : ℝ → ℝ) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + 1 / x) * f y = f (x * y) + f (y / x)) →
  functional_solution f :=
sorry

end determine_function_l287_287050


namespace average_speed_car_l287_287849

theorem average_speed_car : 
  ∀ (d1 d2 t : ℕ), 
    d1 = 98 ∧ d2 = 70 ∧ t = 2 →
    (d1 + d2) / t = 84 :=
by
  intros d1 d2 t h
  cases h with h1 h_tail
  cases h_tail with h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end average_speed_car_l287_287849


namespace validate_financial_position_l287_287225

noncomputable def financial_position_start_of_year := 86588
noncomputable def financial_position_end_of_year := 137236
noncomputable def total_tax := 8919
noncomputable def remaining_funds_after_tour := 38817

variables
(father_income monthly: ℝ := 50000)
(mother_income monthly: ℝ := 28000)
(grandmother_pension monthly: ℝ := 15000)
(mikhail_scholarship monthly: ℝ := 3000)
(father_tax_deduction monthly: ℝ := 2800)
(mother_tax_deduction monthly: ℝ := 2800)
(tax_rate: ℝ := 0.13)
(np_father: ℝ := father_income - father_tax_deduction)
(np_mother: ℝ := mother_income - mother_tax_deduction)

def net_father_tax (monthly:ℝ) := np_father * tax_rate
def net_mother_tax (monthly:ℝ) := np_mother * tax_rate

def father_monthly_income_after_tax (monthly:=ℝ) := father_income - net_father_tax
def mother_monthly_income_after_tax (monthly:=ℝ) := mother_income - net_mother_tax

def net_monthly_income (monthly:ℝ) := father_monthly_income_after_tax + mother_monthly_income_after_tax + grandmother_pension + mikhail_scholarship
def annual_net_income (yearly:=ℝ) := net_monthly_income * 12

variables
(financial_safety_cushion: ℝ := 10000 * 12)
(household_expenses: ℝ := (50000 + 15000) * 12)

def net_disposable_income_per_year (net_yearly:= ℝ) := annual_net_income - financial_safety_cushion - household_expenses

variables
(cadastral_value:ℝ := 6240000)
(sq_m:ℝ := 78)
(sq_m_reduction: ℝ := 20)
(rate_property: ℝ := 0.001)

def property_tax := (cadastral_value - sq_m_reduction * (cadastral_value / sq_m)) * rate_property

variables
(lada_prior_hp: ℝ := 106)
(lada_xray_hp: ℝ := 122)
(car_tax_rate: ℝ := 35)
(months_prior:ℝ := 3/12)
(months_xray: ℝ := 8/12)

def total_transport_tax := lada_prior_hp * car_tax_rate * months_prior + lada_xray_hp * car_tax_rate * months_xray

variables
(cadastral_value_land: ℝ := 420300)
(land_are: ℝ := 10)
(tax_rate_land: ℝ := 0.003)
(deducted_land_area:= 6)
 
def land_tax := (cadastral_value_land - (cadastral_value_land / land_are) * deducted_land_area) * tax_rate_land

def total_tax_liability := property_tax + total_transport_tax + land_tax

def after_tax_liquidity (total_tax_yearly:=ℝ) := net_disposable_income_per_year - total_tax_liability

variables
(tour_cost := 17900)
(participants := 5)

def remaining_after_tour := after_tax_liquidity - tour_cost * participants

theorem validate_financial_position :
  financial_position_start_of_year = 86588 ∧
  financial_position_end_of_year = 137236 ∧
  total_tax = 8919 ∧
  remaining_funds_after_tour = 38817 :=
by
  sorry 

end validate_financial_position_l287_287225


namespace balls_in_boxes_l287_287232

/-- Given 4 white balls, 5 black balls, and 6 red balls, the number of ways to distribute them into 
3 of 4 distinct boxes such that one box remains empty and each of the 3 boxes contains at least one of each color -/
theorem balls_in_boxes : 
  let white : ℕ := 4,
      black : ℕ := 5,
      red : ℕ := 6,
      boxes : ℕ := 4 in 
  (∃ xs : Finset (Fin 4), xs.card = 3) ∧ 
  let chosen_boxes : Finset (Fin 4) := {0, 1, 2} in -- Example box selection
  (chosen_boxes.card = 3 ∧ 
  (∀ b ∈ chosen_boxes, ∃ w b r : ℕ, 
    w ≥ 1 ∧ b ≥ 1 ∧ r ≥ 1 ∧ w + b + r = 5 ∧ w + b + r = 6 ∧ w + b + r = 7)) →
  ∃ ways : ℕ, ways = 720 :=
sorry

end balls_in_boxes_l287_287232


namespace opposite_of_neg_five_l287_287794

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l287_287794


namespace tangent_intersection_point_l287_287893

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287893


namespace largest_angle_in_triangle_l287_287269

theorem largest_angle_in_triangle (k : ℕ) (h : 3 * k + 4 * k + 5 * k = 180) : 5 * k = 75 :=
  by
  -- This is a placeholder for the proof, which is not required as per instructions
  sorry

end largest_angle_in_triangle_l287_287269


namespace Elek_latitude_proof_Matthias_latitude_proof_l287_287423

noncomputable def calculate_latitude_elek : ℝ := 
  let φ := Real.arccos (1 / 4)
  in 90 - φ

def calculate_latitude_matthias : ℝ :=
  -- Placeholder for detailed route calculation based on problem conditions.
  let φ := some_calculations_based_on_path()
  in φ

theorem Elek_latitude_proof : 
  calculate_latitude_elek = 14.5 :=
sorry

theorem Matthias_latitude_proof : 
  calculate_latitude_matthias = 84.8 :=
sorry

end Elek_latitude_proof_Matthias_latitude_proof_l287_287423


namespace max_possible_number_under_operations_l287_287162

theorem max_possible_number_under_operations :
  ∀ x : ℕ, x < 17 →
    ∀ n : ℕ, (∃ k : ℕ, k < n ∧ (x + 17 * k) % 19 = 0) →
    ∃ m : ℕ, m = (304 : ℕ) :=
sorry

end max_possible_number_under_operations_l287_287162


namespace teacher_allocation_l287_287463

theorem teacher_allocation (teachers schools : ℕ) (h_teachers : teachers = 4) (h_schools : schools = 3) :
  ∃ f : fin teachers → fin schools, (∀ s : fin schools, ∃ t : fin teachers, f t = s) ∧ fintype.card (fin teachers → fin schools) = 36 :=
by {
  sorry
}

end teacher_allocation_l287_287463


namespace problem1_problem2_l287_287096

noncomputable def geometricSequence (a : ℕ → ℝ) : Prop :=
∃ q > 0, ∀ n : ℕ, a (n + 1) = q * a n

def arithmeticSequence (b : ℕ → ℝ) : Prop :=
∃ d, ∀ n : ℕ, b (n + 1) - b n = d

theorem problem1 (a b : ℕ → ℝ)
  (h_a : ∀ n, 0 < a n)  -- Condition that a_n is positive.
  (h_geom : geometricSequence a)  -- Condition that {a_n} is geometric sequence.
  (h_log : ∀ n, log 2 (a n) = (b 1 + b n) / 2)  -- Condition that log2(a_n) = (b1 + bn) / 2.
  : arithmeticSequence b := sorry

theorem problem2 (T a b : ℕ → ℝ) (c : ℕ → ℝ)
  (h_prod : ∀ n, T n = (sqrt 2)^(n^2 + n))  -- Condition that T_n = (sqrt(2))^(n^2+n).
  (h_geom : geometricSequence a)  -- Condition that {a_n} is geometric sequence.
  (h_seq_a : ∀ n, a n = 2^n)  -- Condition derived from the problem.
  (h_arith : arithmeticSequence b)  -- Condition that {b_n} is arithmetic sequence.
  (h_b : ∀ n, b n = 2*n - 1)  -- Condition derived from the problem.
  (h_c : ∀ n, c n = if n % 2 = 1 then a n else b n)  -- Condition defining {c_n}.
  : (finset.range 20).sum (λ i, c (i + 1)) = (628 + 2^21) / 3 := sorry

end problem1_problem2_l287_287096


namespace least_faces_dice_l287_287285

noncomputable theory

def dice_problem (a b c : Nat) : Prop :=
  let P_sum_18 := 1 / 16
  let P_sum_12 := 2 * P_sum_15
  ∃ (ni : Nat), ni = ((a * b * c) / 16) ∧ a ≥ 6 ∧ b ≥ 6 ∧ c ≥ 6 ∧ 
                P_sum_12 = 2 * P_sum_15 ∧ P_sum_18 = 1 / 16 --
                (a + b + c) = 24
  
theorem least_faces_dice : ∃ (a b c : Nat), dice_problem a b c := 
sorry

end least_faces_dice_l287_287285


namespace min_value_of_f_l287_287618

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : 
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y :=
sorry

end min_value_of_f_l287_287618


namespace last_part_length_l287_287763

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end last_part_length_l287_287763


namespace general_term_formula_l287_287137

noncomputable def a : ℕ → ℝ
| 1     := 1
| 2     := 1
| (n+3) := (a (n+2) ^ 2 + 2) / a (n+1)

theorem general_term_formula (n : ℕ) : a n = (1 / (2 * real.sqrt 3)) * ((5 - 3 * real.sqrt 3) * (2 + real.sqrt 3) ^ n + (5 + 3 * real.sqrt 3) * (2 - real.sqrt 3) ^ n) := sorry

end general_term_formula_l287_287137


namespace parabola_focus_directrix_l287_287420

-- Define the conditions of the problem
def focus : ℝ × ℝ := (4, -2)
def directrix (x y : ℝ) : Prop := 4 * x + 6 * y = 24

-- The equation of the parabola to be proven
def parabola_equation (x y : ℝ) : Prop := 9 * x^2 - 12 * x * y - 23 * y^2 - 56 * x + 196 * y + 64 = 0

-- Proof problem statement
theorem parabola_focus_directrix :
  ∀ (x y : ℝ), (parabola_equation x y ↔ 
                (sqrt ((x - 4)^2 + (y + 2)^2) = abs (4 * x + 6 * y - 24) / (2 * sqrt 13)) ∧
                directrix x y) :=
sorry

end parabola_focus_directrix_l287_287420


namespace route_time_difference_l287_287211

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l287_287211


namespace trapezoid_area_l287_287450

-- Define the conditions for the trapezoid
variables (legs : ℝ) (diagonals : ℝ) (longer_base : ℝ)
variables (h : ℝ) (b : ℝ)
hypothesis (leg_length : legs = 40)
hypothesis (diagonal_length : diagonals = 50)
hypothesis (base_length : longer_base = 60)
hypothesis (altitude : h = 100 / 3)
hypothesis (shorter_base : b = 60 - (40 * (Real.sqrt 11)) / 3)

-- The statement to prove the area of the trapezoid
theorem trapezoid_area : 
  ∃ (A : ℝ), 
  (A = ((b + longer_base) * h) / 2) →
  A = (10000 - 2000 * (Real.sqrt 11)) / 9 :=
by
  -- placeholder for the proof
  sorry

end trapezoid_area_l287_287450


namespace derivative_at_2_l287_287360

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287360


namespace satellite_visibility_l287_287284

theorem satellite_visibility (n : ℕ) (k : ℕ) (h₁ : n = 37) (h₂ : k = 17) :
  ∃ p : ℝ × ℝ × ℝ, ∀ t : set (ℝ × ℝ × ℝ), finite t ∧ t.card = n → ∃ s : set (ℝ × ℝ × ℝ), s.card ≤ k ∧ ∀ q ∈ s, q ∈ t ∧ visible_from p q :=
sorry

end satellite_visibility_l287_287284


namespace find_f_prime_at_2_l287_287311

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287311


namespace daily_production_rate_l287_287415

theorem daily_production_rate (P : ℕ) (D : ℕ) 
  (h : 0.90 * (P * D) = 3285) : P = 3650 :=
sorry

end daily_production_rate_l287_287415


namespace min_solutions_f_eq_zero_l287_287599

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 3) = f x)
variable (h_zero_at_2 : f 2 = 0)

theorem min_solutions_f_eq_zero : ∃ S : Finset ℝ, (∀ x ∈ S, f x = 0) ∧ 7 ≤ S.card ∧ (∀ x ∈ S, x > 0 ∧ x < 6) := 
sorry

end min_solutions_f_eq_zero_l287_287599


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287560

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287560


namespace shaded_area_summation_l287_287038

theorem shaded_area_summation (PQ QR : ℝ) (hPQ : PQ = 8) (hQR : QR = 8) : 
  let initial_area := (1 / 2) * PQ * QR in
  let total_shade_area := sum_geometric_series initial_area (1 / 4) 100 in
  total_shade_area ≈ 10.67 := by
  sorry

def sum_geometric_series (a r n : ℝ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

lemma approx_eq (x y : ℝ) : x ≈ y ↔ abs (x - y) < 1e-2 := by
  sorry

end shaded_area_summation_l287_287038


namespace extreme_points_inequality_l287_287133

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - a * x

theorem extreme_points_inequality (x₁ x₂ a : ℝ)
  (h₀ : 0 < x₁) (h₁ : x₁ ≤ 1) 
  (h_domain : ∀ x > 0, x ∈ set.univ) -- The domain of f is (0, +∞)
  (h_extreme : ∀ x, (differentiable ℝ (λ x, f x a)) ∧ ((differentiable ℝ (λ x, (deriv (λ x, f x a))) x)) ∧ ((deriv (λ x, f x a) x = 0 → x = x₁ ∨ x = x₂))) : 
  f x₁ a - f x₂ a ≥ Real.log 2 - (3 / 4) :=
sorry

end extreme_points_inequality_l287_287133


namespace largest_n_l287_287792

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_n (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) ∧
  100 ≤ x * y * (10 * y + x) ∧ x * y * (10 * y + x) < 1000

theorem largest_n : ∃ x y : ℕ, valid_n x y ∧ x * y * (10 * y + x) = 777 := by
  sorry

end largest_n_l287_287792


namespace staffing_ways_l287_287488

theorem staffing_ways (total_candidates unsuitable_candidates : ℕ) (positions : ℕ)
  (suitable_candidates_left : total_candidates - unsuitable_candidates = 15) :
  positions = 5 →
  let candidates := total_candidates - unsuitable_candidates in
  (candidates * (candidates - 1) * (candidates - 2) * (candidates - 3) * (candidates - 4) = 360360) :=
by 
  intros h1 h2; 
  simp [h2]; 
  sorry

end staffing_ways_l287_287488


namespace sale_price_of_sarees_after_discounts_l287_287272

theorem sale_price_of_sarees_after_discounts :
  let original_price := 400.0
  let discount_1 := 0.15
  let discount_2 := 0.08
  let discount_3 := 0.07
  let discount_4 := 0.10
  let price_after_first_discount := original_price * (1 - discount_1)
  let price_after_second_discount := price_after_first_discount * (1 - discount_2)
  let price_after_third_discount := price_after_second_discount * (1 - discount_3)
  let final_price := price_after_third_discount * (1 - discount_4)
  final_price = 261.81 := by
    -- Sorry is used to skip the proof
    sorry

end sale_price_of_sarees_after_discounts_l287_287272


namespace integer_points_in_intersection_l287_287044

theorem integer_points_in_intersection :
  (∃ (p: ℤ × ℤ × ℤ), 
    let (x, y, z) := p in 
    x^2 + y^2 + (z - (21 / 2))^2 ≤ 36 ∧ 
    x^2 + y^2 + (z - 1)^2 ≤ (81 / 4)) -> 
  13 :=
sorry

end integer_points_in_intersection_l287_287044


namespace locus_X_is_perpendicular_bisector_l287_287095

open EuclideanGeometry

noncomputable def geometric_locus (O M : Point) (R : ℝ) : Set Point :=
  {X | dist O X ^ 2 - dist X M ^ 2 = R ^ 2}

theorem locus_X_is_perpendicular_bisector (O M : Point) (R : ℝ) (S : Circle O R) (M_outside_S : ¬ (dist O M ≤ R)) :
  ∀ (X : Point), (∃ S1 : Circle (center S) (radius S), M ∈ S1 ∧ intersects S S1 ∧ tangent_at S1 M X) →
  (X ∈ geometric_locus O M R) ↔ (perpendicular_bisector O M X) :=
sorry

end locus_X_is_perpendicular_bisector_l287_287095


namespace ab_value_l287_287432

theorem ab_value (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by {
  sorry
}

end ab_value_l287_287432


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287541

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287541


namespace zachary_initial_money_l287_287571

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_football_shoes : ℝ := 11.85
def money_needed : ℝ := 8.00

def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_football_shoes
def initial_money : ℝ := total_cost - money_needed

theorem zachary_initial_money : initial_money = 9.00 :=
by
  have h : total_cost = 17.00 := by
    unfold total_cost cost_of_football cost_of_shorts cost_of_football_shoes
    norm_num
  unfold initial_money total_cost money_needed
  rw h
  norm_num

end zachary_initial_money_l287_287571


namespace algebraic_expression_value_l287_287638

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) :
  x^2 - 4 * y^2 + 1 = -3 := by
  sorry

end algebraic_expression_value_l287_287638


namespace expression_value_l287_287835

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (4 * a⁻¹ - 2 * a⁻¹ / 3) / a^2 = 90 := by
  sorry

end expression_value_l287_287835


namespace number_of_squares_in_figure_100_l287_287059

theorem number_of_squares_in_figure_100 :
  (∃ (f : ℕ → ℤ), 
     f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25 ∧ ∀ n, f n = 2 * n^2 + 2 * n + 1) →
  f 100 = 20201 :=
by
  intro h
  rcases h with ⟨f, f0, f1, f2, f3, hf⟩
  have key : f = λ n, 2 * n^2 + 2 * n + 1 := by
    apply funext
    intro n
    exact hf n
  rw [key]
  norm_num
  exact 20201

end number_of_squares_in_figure_100_l287_287059


namespace probability_three_dice_less_than_seven_l287_287483

open Nat

def probability_of_exactly_three_less_than_seven (dice_count : ℕ) (sides : ℕ) (target_faces : ℕ) : ℚ :=
  let p : ℚ := target_faces / sides
  let q : ℚ := 1 - p
  (Nat.choose dice_count (dice_count / 2)) * (p^(dice_count / 2)) * (q^(dice_count / 2))

theorem probability_three_dice_less_than_seven :
  probability_of_exactly_three_less_than_seven 6 12 6 = 5 / 16 := by
  sorry

end probability_three_dice_less_than_seven_l287_287483


namespace route_comparison_l287_287219

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l287_287219


namespace yellow_area_is_1_5625_percent_l287_287930

def square_flag_area (s : ℝ) : ℝ := s ^ 2

def cross_yellow_occupies_25_percent (s : ℝ) (w : ℝ) : Prop :=
  4 * w * s - 4 * w ^ 2 = 0.25 * s ^ 2

def yellow_area (s w : ℝ) : ℝ := 4 * w ^ 2

def percent_of_flag_area_is_yellow (s w : ℝ) : Prop :=
  yellow_area s w = 0.015625 * s ^ 2

theorem yellow_area_is_1_5625_percent (s w : ℝ) (h1: cross_yellow_occupies_25_percent s w) : 
  percent_of_flag_area_is_yellow s w :=
by sorry

end yellow_area_is_1_5625_percent_l287_287930


namespace find_m_l287_287404

theorem find_m (x1 x2 m : ℝ) (h1 : 2 * x1^2 - 3 * x1 + m = 0) (h2 : 2 * x2^2 - 3 * x2 + m = 0) (h3 : 8 * x1 - 2 * x2 = 7) :
  m = 1 :=
sorry

end find_m_l287_287404


namespace range_of_a_l287_287147

theorem range_of_a (a : ℝ) (h : (1 - a) * (1 - (a + 2)) ≤ 0) : -1 ≤ a ∧ a ≤ 1 := 
begin
  -- Proof is not required
  sorry
end

end range_of_a_l287_287147


namespace correctFunctionIsOptionB_l287_287016

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def isMonotonicallyDecreasingOnPositiveReal (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (0 < x ∧ x < y) → f y ≤ f x

def candidateFunctions : List (ℝ → ℝ) :=
  [ (λ x, x ^ (-1/2)),
    (λ x, x ^ (-2)),
    (λ x, x ^ (1/2)),
    (λ x, x ^ 2) ]

theorem correctFunctionIsOptionB :
  ∀ f ∈ candidateFunctions,
    (isEvenFunction f ∧ isMonotonicallyDecreasingOnPositiveReal f) ↔ f = (λ x, x ^ (-2)) := 
by
  sorry

end correctFunctionIsOptionB_l287_287016


namespace points_concyclic_l287_287672

-- Definitions and conditions as assumptions
variables {A B C H D E : Type*}
variables [AffineSpace A]
variables [AcuteTriangle ABC : Triangle A B C]
variables [Orthocenter H : Orthocenter A B C]
variables [OnLine D AC : OnLine D A C]
variables [Equal HA HD : (distance H A) = (distance H D)]
variables [Parallelogram ABEH : Parallelogram A B E H]

-- Theorem statement to prove concyclicity
theorem points_concyclic (A B C H D E : Point) [h1 : AcuteTriangle A B C] [h2 : Orthocenter H A B C]
  [h3 : OnLine D A C] [h4 : (distance H A) = (distance H D)] [h5 : Parallelogram A B E H] :
  Concyclic [B, E, C, D, H] := by
  sorry

end points_concyclic_l287_287672


namespace derivative_at_2_l287_287361

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287361


namespace expected_value_of_winnings_l287_287748

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop :=
  n = 4 ∨ n = 6 ∨ n = 8

def winnings (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if is_prime n then n
  else if n = 8 then -16
  else 0

def probabilities : list (ℕ × ℚ) :=
  [(1, 1/8), (2, 1/8), (3, 1/8), (4, 1/8),
   (5, 1/8), (6, 1/8), (7, 1/8), (8, 1/8)]

def expected_value : ℚ :=
  probabilities.sum (λ (outcome : ℕ × ℚ), (outcome.snd * winnings outcome.fst))

theorem expected_value_of_winnings :
  expected_value = 0.375 :=
by
  sorry

end expected_value_of_winnings_l287_287748


namespace emily_did_not_sell_bars_l287_287981

-- Definitions based on conditions
def cost_per_bar : ℕ := 4
def total_bars : ℕ := 8
def total_earnings : ℕ := 20

-- The statement to be proved
theorem emily_did_not_sell_bars :
  (total_bars - (total_earnings / cost_per_bar)) = 3 :=
by
  sorry

end emily_did_not_sell_bars_l287_287981


namespace cost_of_purchase_l287_287759

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l287_287759


namespace sum_of_coefficients_l287_287078

-- Define the polynomial
def polynomial (x : ℝ) : ℝ :=
  2 * (4 * x ^ 8 + 7 * x ^ 6 - 9 * x ^ 3 + 3) + 6 * (x ^ 7 - 2 * x ^ 4 + 8 * x ^ 2 - 2)

-- State the theorem to prove the sum of the coefficients
theorem sum_of_coefficients : polynomial 1 = 40 :=
by
  sorry

end sum_of_coefficients_l287_287078


namespace compacted_space_of_all_cans_l287_287224

def compacted_space_per_can (original_space: ℕ) (compaction_rate: ℕ) : ℕ :=
  original_space * compaction_rate / 100

def total_compacted_space (num_cans: ℕ) (compacted_space: ℕ) : ℕ :=
  num_cans * compacted_space

theorem compacted_space_of_all_cans :
  ∀ (num_cans original_space compaction_rate : ℕ),
  num_cans = 100 →
  original_space = 30 →
  compaction_rate = 35 →
  total_compacted_space num_cans (compacted_space_per_can original_space compaction_rate) = 1050 :=
by
  intros num_cans original_space compaction_rate h1 h2 h3
  rw [h1, h2, h3]
  dsimp [compacted_space_per_can, total_compacted_space]
  norm_num
  sorry

end compacted_space_of_all_cans_l287_287224


namespace find_f_prime_at_2_l287_287345

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287345


namespace solve_inequality_l287_287242

-- Define conditions
def valid_x (x : ℝ) : Prop := x ≠ -3 ∧ x ≠ -8/3

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < -8/3) ∨ ((1 - Real.sqrt 89) / 4 < x ∧ x < (1 + Real.sqrt 89) / 4)

-- Prove the equivalence
theorem solve_inequality (x : ℝ) (h : valid_x x) : inequality x ↔ solution_set x :=
by
  sorry

end solve_inequality_l287_287242


namespace tangent_line_intersection_x_axis_l287_287900

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287900


namespace magnitude_addition_l287_287110

variables (a b : ℝ^3)
variable  (θ : ℝ)

-- Assuming the given conditions
def unit_vectors (a b : ℝ^3) : Prop := |a| = 1 ∧ |b| = 1
def angle_60_degrees (a b : ℝ^3) : Prop := θ = π / 3 ∧ a ∷ b = (|a| * |b|) * (Real.cos θ)

-- This is the theorem to prove
theorem magnitude_addition (h₁ : unit_vectors a b) (h₂ : angle_60_degrees a b) : |a + 2 • b| = √7 :=
sorry

end magnitude_addition_l287_287110


namespace find_f_prime_at_2_l287_287340

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287340


namespace length_of_DE_l287_287689

theorem length_of_DE (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  [has_dist A ℝ] [has_dist B ℝ] [has_dist C ℝ] [has_dist D ℝ] [has_dist E ℝ] 
  (BC : ℝ) (angle_C : ℝ) (CD : ℝ) (DE : ℝ)
  (hBC : BC = 30)
  (hAngleC : angle_C = π / 4)
  (hDmidpoint : CD = BC / 2)
  (hPerpBisector : DE = CD)
  : DE = 15 :=
sorry

end length_of_DE_l287_287689


namespace range_of_m_l287_287727

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x-1)^2
  else if x > 0 then -(x+1)^2
  else 0

theorem range_of_m (m : ℝ) (h : f (m^2 + 2*m) + f m > 0) : -3 < m ∧ m < 0 := 
by {
  sorry
}

end range_of_m_l287_287727


namespace total_bird_families_l287_287811

-- Declare the number of bird families that flew to Africa
def a : Nat := 47

-- Declare the number of bird families that flew to Asia
def b : Nat := 94

-- Condition that Asia's number of bird families matches Africa + 47 more
axiom h : b = a + 47

-- Prove the total number of bird families is 141
theorem total_bird_families : a + b = 141 :=
by
  -- Insert proof here
  sorry

end total_bird_families_l287_287811


namespace oldest_person_jane_babysat_age_l287_287189

-- Definitions based on conditions
def jane_started_babysitting (jane_started_age : ℕ) : Prop :=
  jane_started_age = 18

def jane_current_age (jane_age : ℕ) : Prop :=
  jane_age = 32

def jane_stopped_babysitting_years_ago (years_ago : ℕ) : Prop :=
  years_ago = 12

def max_child_age_when_babysitting (jane_age child_age : ℕ) : Prop :=
  child_age ≤ jane_age / 2

-- Theorem based on question and answer
theorem oldest_person_jane_babysat_age (jane_started_age jane_current_age jane_stopped_babysitting_years_ago : ℕ) : 
  jane_started_babysitting jane_started_age →
  jane_current_age jane_current_age →
  jane_stopped_babysitting_years_ago jane_stopped_babysitting_years_ago →
  ∃ child_current_age, max_child_age_when_babysitting (jane_current_age - jane_stopped_babysitting_years_ago) (child_current_age - jane_stopped_babysitting_years_ago) ∧
                        child_current_age = 22 :=
by {
    -- Proof omitted
    sorry
}

end oldest_person_jane_babysat_age_l287_287189


namespace trapezoid_height_l287_287068

variable (a b : ℝ)
variable (h : ℝ)
variable (angle_diag angle_ext : ℝ)

-- Conditions
axiom bases_distinct (h_bases : a < b)
axiom diagonals_perpendicular (h_angle_diag : angle_diag = 90)
axiom extensions_angle (h_angle_ext : angle_ext = 45)

-- Proof problem
theorem trapezoid_height (h_bases : a < b) (h_angle_diag : angle_diag = 90) (h_angle_ext : angle_ext = 45) : 
  h = a * b / (b - a) := 
sorry

end trapezoid_height_l287_287068


namespace first_number_with_three_common_factors_with_45_l287_287812

theorem first_number_with_three_common_factors_with_45 (n : ℕ) (h_pos : 0 < n) (h_factors : (Finset.filter (λ d, d ∣ 45 ∧ d ∣ n) (Finset.range (n + 1))).card = 3) : n = 15 := 
  sorry

end first_number_with_three_common_factors_with_45_l287_287812


namespace aq_ae_in_terms_of_cd_l287_287968

-- Define the circle and its properties
variables {O A B C D E Q : Type} [MetricSpace O]

-- Define diameters AB and CD of the circle O and that they are perpendicular to each other.
axiom diameters_are_perpendicular : ∀ {O : Type} [MetricSpace O] {A B C D : O}, Diameter O A B → Diameter O C D → Perpendicular O A B C D

-- Define that AQ is triple the length of QE.
axiom length_ratio_AQ_QE : ∀ {Q E : ℝ}, (AQ = 3 * QE)

-- Prove that AQ * AE = 3CD^2 / 4 under the given conditions
theorem aq_ae_in_terms_of_cd (h1 : Diameter O A B) (h2 : Diameter O C D) (h3 : Perpendicular O A B C D)
  (h4 : AQ = 3 * QE) : AQ * AE = (3 * CD^2) / 4 :=
  by
  sorry

end aq_ae_in_terms_of_cd_l287_287968


namespace find_f_prime_at_2_l287_287314

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287314


namespace range_of_s_l287_287301

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x^2) ^ 2

theorem range_of_s :
  set.range s = set.Ioo 0 (1 / 4) ∪ { (1 / 4 : ℝ) } :=
sorry

end range_of_s_l287_287301


namespace base_5_to_base_10_l287_287414

theorem base_5_to_base_10 : 
  let n : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0
  n = 194 :=
by 
  sorry

end base_5_to_base_10_l287_287414


namespace magnitude_sum_b1_b2_l287_287629

-- Define the given vectors for the proof
variables (e1 e2 : ℝ^3) (h_unit_e1 : ∥e1∥ = 1) (h_unit_e2 : ∥e2∥ = 1)
variable (h_angle : e1 ⬝ e2 = 1/2) -- Angle between e1 and e2 is π/3, thus cos(π/3) = 1/2

-- Define b1 and b2 using e1 and e2
def b1 := e1 - 2 • e2
def b2 := e1 + 4 • e2

-- Statement of the problem
theorem magnitude_sum_b1_b2 : ∥b1 + b2∥ = 2 * Real.sqrt 3 :=
by
  sorry

end magnitude_sum_b1_b2_l287_287629


namespace variance_eta_l287_287772

/-
  Given:
  ξ follows a normal distribution N(2, 9)
  η = 2ξ - 1

  Prove:
  D(η) = 36
-/

variable (ξ : ℝ)
variable (η : ℝ)

-- Given: ξ follows a normal distribution N(2, 9)
axiom normal_dist_xi : ∀ E D : ℝ, (E ξ = 2 ∧ D ξ = 9)

-- Given: η = 2ξ - 1
axiom eta_def : η = 2 * ξ - 1

-- Variance formula for a linear transformation: D(aξ + b) = a^2 * D(ξ)
axiom linear_variance : ∀ a b Dξ : ℝ, D (aξ + b) = a^2 * Dξ

-- Prove: D(η) = 36
theorem variance_eta : D η = 36 :=
sorry

end variance_eta_l287_287772


namespace professional_pay_per_hour_l287_287703

def professionals : ℕ := 2
def hours_per_day : ℕ := 6
def days : ℕ := 7
def total_cost : ℕ := 1260

theorem professional_pay_per_hour :
  (total_cost / (professionals * hours_per_day * days) = 15) :=
by
  sorry

end professional_pay_per_hour_l287_287703


namespace new_planet_volume_eq_l287_287170

noncomputable def volume_of_new_planet (V_earth : ℝ) (scaling_factor : ℝ) : ℝ :=
  V_earth * (scaling_factor^3)

theorem new_planet_volume_eq 
  (V_earth : ℝ)
  (scaling_factor : ℝ)
  (hV_earth : V_earth = 1.08 * 10^12)
  (h_scaling_factor : scaling_factor = 10^4) :
  volume_of_new_planet V_earth scaling_factor = 1.08 * 10^24 :=
by
  sorry

end new_planet_volume_eq_l287_287170


namespace arrangement_count_l287_287950

def totalArrangements (A B C D E : Type) (jobs : Type) : ℕ :=
  let possibleJobs := {translation, tour_guide, etiquette, driver}
  let students := {A, B, C, D, E}
  if A ∈ possibleJobs ∧ B ∈ possibleJobs ∧ C ∈ possibleJobs ∧ D ∈ possibleJobs ∧ E ∈ possibleJobs then
    126 -- the proven result
  else
    0  -- placeholder since we know students must be able to do their jobs

theorem arrangement_count : 
  totalArrangements 
    (λ x : Type, x != driver → ∃ y : ℕ, y = 3 ∧ x ∈ {translation, tour_guide, etiquette})  
    (λ z : Type, z = driver → ∃ w : ℕ, w = 4 ∧ z ∈ {translation, tour_guide, etiquette, driver}) = 126 :=
sorry

end arrangement_count_l287_287950


namespace erased_number_l287_287467

theorem erased_number (a b : ℤ) (h1 : ∀ n : ℤ, n ∈ set.range (λ i, a + i) ↔ n ∈ set.range (λ i, a - 4 + i)) 
                      (h2 : 8 * a - b = 1703)
                      (h3 : -4 ≤ b ∧ b ≤ 4) : a + b = 214 := 
by
    sorry

end erased_number_l287_287467


namespace derivative_at_2_l287_287375

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287375


namespace max_third_side_l287_287771

theorem max_third_side (P Q R : ℝ) (a b c : ℝ)
  (h1 : a = 8) (h2 : b = 15) 
  (h3 : cos (4 * P) + cos (4 * Q) + cos (4 * R) = 1)
  (h4 : P + Q + R = π) : c ≤ 17 :=
by
  sorry

end max_third_side_l287_287771


namespace divisor_is_five_l287_287745

theorem divisor_is_five (n d : ℕ) (h1 : ∃ k, n = k * d + 3) (h2 : ∃ l, n^2 = l * d + 4) : d = 5 :=
sorry

end divisor_is_five_l287_287745


namespace tangent_point_value_l287_287878

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287878


namespace diff_sum_even_odd_l287_287829

theorem diff_sum_even_odd (n : ℕ) (hn : n = 1500) :
  let sum_odd := n * (2 * n - 1)
  let sum_even := n * (2 * n + 1)
  sum_even - sum_odd = 1500 :=
by
  sorry

end diff_sum_even_odd_l287_287829


namespace train_speed_l287_287821

theorem train_speed (length_of_train time_to_cross : ℝ) (equal_speeds : Bool) (total_distance := 2 * length_of_train) (relative_speed := total_distance / time_to_cross) (v := relative_speed / 2) (speed_in_kmph := v * 3.6) : 
  length_of_train = 120 → 
  time_to_cross = 24 → 
  equal_speeds → 
  speed_in_kmph = 18 := 
by
  intros h1 h2 h3
  unfold total_distance relative_speed v speed_in_kmph
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l287_287821


namespace function_periodicity_l287_287974

variable {R : Type*} [Ring R]

def periodic_function (f : R → R) (k : R) : Prop :=
  ∀ x : R, f (x + 4*k) = f x

theorem function_periodicity {f : ℝ → ℝ} {k : ℝ} (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) (hk : k ≠ 0) : 
  periodic_function f k :=
sorry

end function_periodicity_l287_287974


namespace polar_line_eqn_correct_l287_287184

-- Define the given point and condition
def given_point : ℝ × ℝ := (2, real.pi / 3)
def line_parallel_to_polar_axis (p : ℝ × ℝ) : Prop := p.1 = given_point.1 ∧ p.2 = given_point.2

-- Define the corresponding line equation in polar coordinates
def line_eqn_polar (ρ θ : ℝ) : Prop := ρ * real.sin θ = real.sqrt 3

-- The main theorem to prove
theorem polar_line_eqn_correct (ρ θ : ℝ) : 
  line_parallel_to_polar_axis (ρ, θ) → line_eqn_polar ρ θ := 
begin
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  sorry
end

end polar_line_eqn_correct_l287_287184


namespace savings_sum_l287_287252

-- Define the values assigned to each coin type
def penny := 0.01
def nickel := 0.05
def dime := 0.10

-- Define the number of coins each person has
def teaganPennies := 200
def rexNickels := 100
def toniDimes := 330

-- Calculate the total amount saved by each person
def teaganSavings := teaganPennies * penny
def rexSavings := rexNickels * nickel
def toniSavings := toniDimes * dime

-- Calculate the total savings of all three persons together
def totalSavings := teaganSavings + rexSavings + toniSavings

theorem savings_sum : totalSavings = 40 := by
  -- the actual proof is omitted, indicated by sorry
  sorry

end savings_sum_l287_287252


namespace octagon_side_length_eq_l287_287670

theorem octagon_side_length_eq (AB BC : ℝ) (AE FB s : ℝ) :
  AE = FB → AE < 5 → AB = 10 → BC = 12 →
  s = -11 + Real.sqrt 242 →
  EF = (10.5 - (Real.sqrt 242) / 2) :=
by
  -- Identified parameters and included all conditions from step a)
  intros h1 h2 h3 h4 h5
  -- statement of the theorem to be proven
  let EF := (10.5 - (Real.sqrt 242) / 2)
  sorry  -- placeholder for proof

end octagon_side_length_eq_l287_287670


namespace find_f_prime_at_2_l287_287308

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287308


namespace derivative_at_2_l287_287367

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287367


namespace problem_statement_l287_287654

theorem problem_statement
  (m n y : ℝ)
  (h : (m * x^2 + 3 * x - y) - (4 * x^2 - (2 * n + 3) * x + 3 * y - 2) = c for all x where c does not depend on x) :
  (m - n) + |m * n| = 19 :=
by
  sorry

end problem_statement_l287_287654


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287557

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287557


namespace erin_watching_time_l287_287984

theorem erin_watching_time (minutes_per_episode : ℕ) (num_episodes : ℕ) (minutes_in_hour : ℕ)
  (h1 : minutes_per_episode = 50)
  (h2 : num_episodes = 6)
  (h3 : minutes_in_hour = 60) :
  (minutes_per_episode * num_episodes) / minutes_in_hour = 5 :=
by
  rw [h1, h2, h3]
  -- Here you will proceed to calculate: (50 * 6) / 60 = 5
  sorry

end erin_watching_time_l287_287984


namespace time_difference_180_div_vc_l287_287661

open Real

theorem time_difference_180_div_vc
  (V_A V_B V_C : ℝ)
  (h_ratio : V_A / V_C = 5 ∧ V_B / V_C = 4)
  (start_A start_B start_C : ℝ)
  (h_start_A : start_A = 100)
  (h_start_B : start_B = 80)
  (h_start_C : start_C = 0)
  (race_distance : ℝ)
  (h_race_distance : race_distance = 1200) :
  (race_distance - start_A) / V_A - race_distance / V_C = 180 / V_C := 
sorry

end time_difference_180_div_vc_l287_287661


namespace find_f_prime_at_2_l287_287348

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287348


namespace repeating_decimal_fraction_l287_287517

theorem repeating_decimal_fraction :
  let x : ℚ := 75 / 99,
      z : ℚ := 25 / 99,
      y : ℚ := 2 + z 
  in (x / y) = 2475 / 7339 :=
by
  sorry

end repeating_decimal_fraction_l287_287517


namespace optimal_production_distribution_l287_287825

noncomputable def min_production_time (unitsI_A unitsI_B unitsII_B : ℕ) : ℕ :=
let rateI_A := 30
let rateII_B := 40
let rateI_B := 50
let initial_days_B := 20
let remaining_units_I := 1500 - (rateI_A * initial_days_B)
let combined_rateI_AB := rateI_A + rateI_B
let days_remaining_I := remaining_units_I / combined_rateI_AB
initial_days_B + days_remaining_I

theorem optimal_production_distribution :
  ∃ (unitsI_A unitsI_B unitsII_B : ℕ),
    unitsI_A + unitsI_B = 1500 ∧ unitsII_B = 800 ∧
    min_production_time unitsI_A unitsI_B unitsII_B = 31 := sorry

end optimal_production_distribution_l287_287825


namespace book_store_sold_total_copies_by_saturday_l287_287418

def copies_sold_on_monday : ℕ := 15
def copies_sold_on_tuesday : ℕ := copies_sold_on_monday * 2
def copies_sold_on_wednesday : ℕ := copies_sold_on_tuesday + (copies_sold_on_tuesday / 2)
def copies_sold_on_thursday : ℕ := copies_sold_on_wednesday + (copies_sold_on_wednesday / 2)
def copies_sold_on_friday_pre_promotion : ℕ := copies_sold_on_thursday + (copies_sold_on_thursday / 2)
def copies_sold_on_friday_post_promotion : ℕ := copies_sold_on_friday_pre_promotion + (copies_sold_on_friday_pre_promotion / 4)
def copies_sold_on_saturday : ℕ := copies_sold_on_friday_pre_promotion * 7 / 10

def total_copies_sold_by_saturday : ℕ :=
  copies_sold_on_monday + copies_sold_on_tuesday + copies_sold_on_wednesday +
  copies_sold_on_thursday + copies_sold_on_friday_post_promotion + copies_sold_on_saturday

theorem book_store_sold_total_copies_by_saturday : total_copies_sold_by_saturday = 357 :=
by
  -- Proof here
  sorry

end book_store_sold_total_copies_by_saturday_l287_287418


namespace closest_point_on_line_eq_l287_287531

noncomputable def closest_point_to_line : ℝ × ℝ :=
  let line := λ x : ℝ, (-2 * x + 3) in
  let point := (3, 1) : ℝ × ℝ in
  let direction_vector := (1, -2) : ℝ × ℝ in
  let vec_to_point := (3 - 0, 1 - 3) : ℝ × ℝ in
  let projection := (7 / 5, -14 / 5) in
  (0 + 7 / 5, 3 - 14 / 5)

theorem closest_point_on_line_eq :
  closest_point_to_line = (7 / 5, 1 / 5) :=
sorry

end closest_point_on_line_eq_l287_287531


namespace ranking_emily_olivia_nicole_l287_287741

noncomputable def Emily_score : ℝ := sorry
noncomputable def Olivia_score : ℝ := sorry
noncomputable def Nicole_score : ℝ := sorry

theorem ranking_emily_olivia_nicole :
  (Emily_score > Olivia_score) ∧ (Emily_score > Nicole_score) → 
  (Emily_score > Olivia_score) ∧ (Olivia_score > Nicole_score) := 
by sorry

end ranking_emily_olivia_nicole_l287_287741


namespace repeating_decimal_fraction_l287_287518

theorem repeating_decimal_fraction :
  let x : ℚ := 75 / 99,
      z : ℚ := 25 / 99,
      y : ℚ := 2 + z 
  in (x / y) = 2475 / 7339 :=
by
  sorry

end repeating_decimal_fraction_l287_287518


namespace evaluate_expression_l287_287511

theorem evaluate_expression (a b : ℕ) (h_a : a = 3) (h_b : b = 2) : (a^b)^b - (b^a)^a = -431 := by
  sorry

end evaluate_expression_l287_287511


namespace minimum_perimeter_l287_287820

noncomputable def minimum_perimeter_of_triangle_APR (A B C P R Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace R] [MetricSpace Q] : ℝ :=
  ∑ d in [dist A P, dist P R, dist R A], d

theorem minimum_perimeter (A B C P R Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace R] [MetricSpace Q] (h1 : dist A B = 25) (h2 : dist A B = dist A C) (h3 : dist B P = dist P Q) (h4 : dist C R = dist R Q) :
  minimum_perimeter_of_triangle_APR A B C P R Q = 50 :=
sorry 

end minimum_perimeter_l287_287820


namespace trapezoid_area_l287_287452

-- Define the conditions for the trapezoid
variables (legs : ℝ) (diagonals : ℝ) (longer_base : ℝ)
variables (h : ℝ) (b : ℝ)
hypothesis (leg_length : legs = 40)
hypothesis (diagonal_length : diagonals = 50)
hypothesis (base_length : longer_base = 60)
hypothesis (altitude : h = 100 / 3)
hypothesis (shorter_base : b = 60 - (40 * (Real.sqrt 11)) / 3)

-- The statement to prove the area of the trapezoid
theorem trapezoid_area : 
  ∃ (A : ℝ), 
  (A = ((b + longer_base) * h) / 2) →
  A = (10000 - 2000 * (Real.sqrt 11)) / 9 :=
by
  -- placeholder for the proof
  sorry

end trapezoid_area_l287_287452


namespace sum_first_six_terms_geometric_sequence_l287_287996

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l287_287996


namespace fertilization_changes_respiration_synthesis_rate_l287_287519

theorem fertilization_changes_respiration_synthesis_rate 
  (restore : ∀ cells, fertilized cells = (cells / 2) → original cells) :
  ∀ egg_cells, 
  rate_of(cellular_respiration egg_cells fertilized) ≠ rate_of(cellular_respiration egg_cells unfertilized)
  ∧ rate_of(substance_synthesis egg_cells fertilized) ≠ rate_of(substance_synthesis egg_cells unfertilized) :=
sorry

end fertilization_changes_respiration_synthesis_rate_l287_287519


namespace optimal_additional_plates_l287_287658

-- Define the original set sizes
def original_first_set := 5
def original_second_set := 3
def original_third_set := 4

-- Define the new choices in optimal scenarios
def new_first_set_opt1 := 6
def new_second_set_opt1 := 3
def new_third_set_opt1 := 5

def new_first_set_opt2 := 5
def new_second_set_opt2 := 3
def new_third_set_opt2 := 6

-- Define the original and new number of plates
def original_plates := original_first_set * original_second_set * original_third_set
def new_plates_opt1 := new_first_set_opt1 * new_second_set_opt1 * new_third_set_opt1
def new_plates_opt2 := new_first_set_opt2 * new_second_set_opt2 * new_third_set_opt2

-- Define the increase in plates for optimal placements
def additional_plates := 30

-- The theorem that states that adding letters optimally produces the correct additional plates
theorem optimal_additional_plates :
  (new_plates_opt1 - original_plates = additional_plates) ∧ 
  (new_plates_opt2 - original_plates = additional_plates) :=
by simp [original_plates, new_plates_opt1, new_plates_opt2, additional_plates]

end optimal_additional_plates_l287_287658


namespace trigonometric_identity_solution_l287_287118

theorem trigonometric_identity_solution (x : ℝ) (k n : ℤ) :
  (sin(3 * x) ≠ 0) ∧ (∀ k : ℤ, x ≠ (π * k / 3)) → 
  (sin(2 * x))^2 = (cos(x))^2 + cos(3 * x) / sin(3 * x) ↔ 
  (∃ n : ℤ, x = π / 2 + π * n) ∨ (∃ k : ℤ, x = π / 6 + π * k ∨ x = -π / 6 + π * k) := 
by 
sorry

end trigonometric_identity_solution_l287_287118


namespace bond_yield_correct_l287_287227

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end bond_yield_correct_l287_287227


namespace sequence_8123_appears_consecutively_l287_287682

theorem sequence_8123_appears_consecutively :
  (∃ n : ℕ, ∀ m : ℕ, (sequence m = 8) ∧ (sequence (m + 1) = 1) ∧ (sequence (m + 2) = 2) ∧ (sequence (m + 3) = 3)) :=
sorry

noncomputable def sequence (n : ℕ) : ℕ :=
if n < 4 then [1, 2, 3, 4].nth_le (n + 1) (by linarith)
else (sequence (n - 1) + sequence (n - 2) + sequence (n - 3) + sequence (n - 4)) % 10

end sequence_8123_appears_consecutively_l287_287682


namespace largest_c_constant_l287_287526

noncomputable def max_c : ℝ :=
  sqrt(6) / 6

theorem largest_c_constant
  (n : ℕ) (h : n ≥ 3)
  (A : Fin n → Set ℝ)
  (H : ∃ T : Finset (Fin n) × Finset (Fin n) × Finset (Fin n), T.1 ≠ ∅ ∧ T.1.card = (n.choose 3) / 2 ∧
       ∀ (i j k : Fin n), i ∈ T.1 ∧ j ∈ T.2 ∧ k ∈ T.3 → 1 ≤ i.val ∧ i.val < j.val ∧ j.val < k.val ∧ (A i ∩ A j ∩ A k).nonempty) :
  ∃ I : Finset (Fin n), I.card > (sqrt(6) / 6) * n ∧ (⋂ i in I, A i).nonempty :=
sorry

end largest_c_constant_l287_287526


namespace find_f_prime_at_2_l287_287309

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287309


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287903

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287903


namespace range_of_F_l287_287621

noncomputable def f (x : ℝ) : ℝ := sqrt (x + 2) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := x - 1
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem range_of_F :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ -2 ∧ x ≠ 1 ∧ F x = y) ↔ (y ∈ [0, sqrt 3) ∪ (sqrt 3, +∞)) :=
sorry

end range_of_F_l287_287621


namespace ratio_of_time_l287_287410

theorem ratio_of_time (T_A T_B : ℝ) (h1 : T_A = 8) (h2 : 1 / T_A + 1 / T_B = 0.375) :
  T_B / T_A = 1 / 2 :=
by 
  sorry

end ratio_of_time_l287_287410


namespace tangent_line_intersection_l287_287887

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287887


namespace trigonometric_identity_l287_287124

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  1 + Real.sin α * Real.cos α = 7 / 5 :=
by
  sorry

end trigonometric_identity_l287_287124


namespace find_divisor_l287_287067

variable (Dividend : ℕ) (Quotient : ℕ) (Divisor : ℕ)
variable (h1 : Dividend = 64)
variable (h2 : Quotient = 8)
variable (h3 : Dividend = Divisor * Quotient)

theorem find_divisor : Divisor = 8 := by
  sorry

end find_divisor_l287_287067


namespace problem_solution_l287_287412

noncomputable def P_conditional (A B : Set Ω) := P (A ∩ B) / P B

variables (Ω : Type) [Fintype Ω] (questions : Set Ω)
variables (mcqs fillintheblanks : Set Ω)
variables (A B : Set Ω)
variables [DecidableEq Ω]

def is_mcq (q : Ω) : Prop := q ∈ mcqs
def is_fill_in_the_blank (q : Ω) : Prop := q ∈ fillintheblanks

-- Conditions
axiom condition1 : mcqs.card = 3
axiom condition2 : fillintheblanks.card = 2
axiom condition3 : A = {q | is_mcq q}
axiom condition4 : B = {q | is_fill_in_the_blank q}

theorem problem_solution : P_conditional Ω A B = 3 / 4 := sorry

end problem_solution_l287_287412


namespace kai_ice_plate_division_l287_287702

-- Define the "L"-shaped ice plate with given dimensions
structure LShapedIcePlate (a : ℕ) :=
(horiz_length : ℕ)
(vert_length : ℕ)
(horiz_eq_vert : horiz_length = a ∧ vert_length = a)

-- Define the correctness of dividing the L-shaped plate into four equal parts
def can_be_divided_into_four_equal_parts (a : ℕ) (piece : LShapedIcePlate a) : Prop :=
∃ cut_points_v1 cut_points_v2 cut_points_h1 cut_points_h2,
  -- The cut points for vertical and horizontal cuts to turn the large "L" shape into four smaller "L" shapes
  piece.horiz_length = cut_points_v1 + cut_points_v2 ∧
  piece.vert_length = cut_points_h1 + cut_points_h2 ∧
  cut_points_v1 = a / 2 ∧ cut_points_v2 = a - a / 2 ∧
  cut_points_h1 = a / 2 ∧ cut_points_h2 = a - a / 2

-- Prove the main theorem
theorem kai_ice_plate_division (a : ℕ) (h : a > 0) (plate : LShapedIcePlate a) : 
  can_be_divided_into_four_equal_parts a plate :=
sorry

end kai_ice_plate_division_l287_287702


namespace total_cost_of_purchase_l287_287758

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l287_287758


namespace number_of_irrationals_l287_287019

def is_irrational (x : ℝ) : Prop := ¬∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem number_of_irrationals : 
  let a : ℝ := 2 / 3 in
  let b : ℝ := Real.sqrt 8 in
  let c : ℝ := Real.pi / 3 in
  let d : ℝ := 3.14159 in
  (if is_irrational a then 1 else 0) +
  (if is_irrational b then 1 else 0) +
  (if is_irrational c then 1 else 0) +
  (if is_irrational d then 1 else 0)
  = 2 :=
by sorry

end number_of_irrationals_l287_287019


namespace crayons_left_l287_287484

theorem crayons_left (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) :
  initial_crayons - eaten_crayons = 10 :=
by
  rw [h1, h2]
  exact rfl

end crayons_left_l287_287484


namespace percentage_of_students_in_75_to_84_range_l287_287416

noncomputable def tally_95_100 := 3
noncomputable def tally_85_94 := 6
noncomputable def tally_75_84 := 8
noncomputable def tally_65_74 := 4
noncomputable def tally_55_64 := 3
noncomputable def tally_below_55 := 4

noncomputable def total_students := tally_95_100 + tally_85_94 + tally_75_84 + tally_65_74 + tally_55_64 + tally_below_55
noncomputable def percentage_75_84 := (tally_75_84.to_float / total_students.to_float) * 100

theorem percentage_of_students_in_75_to_84_range :
  percentage_75_84 ≈ 28.57 := sorry

end percentage_of_students_in_75_to_84_range_l287_287416


namespace find_a_b_find_m_range_l287_287130

-- Defining the function f(x) in Lean
def f (x a b : ℝ) : ℝ := x^2 + x - Real.log (x + a) + 3 * b

-- Statements for proving the value of a and b
theorem find_a_b : 
  ∃ a b : ℝ, 
    a = 1 ∧ b = 0 ∧ 
    (∀ x : ℝ, f x a b = x^2 + x - Real.log(x + 1) + 3 * 0) ∧
    f 0 a b = 0 ∧ 
    (deriv (λ (x : ℝ), f x a b)) 0 = 0 :=
sorry
  
-- Statements for proving the range of m
theorem find_m_range :
  ∃ (m : ℝ), 
    0 < m ∧ m ≤ -1/4 + Real.log 2 ∧ 
    (∀ x : ℝ, ∃ y : ℝ, x ∈ set.Icc (-1/2) 2 → 
    (f x 1 0 = m) → 
    (f x 1 0 ≠ f y 1 0 → x ≠ y)) :=
sorry

end find_a_b_find_m_range_l287_287130


namespace batsman_average_after_17th_inning_l287_287383

-- Define the conditions and prove the required question.
theorem batsman_average_after_17th_inning (A : ℕ) (h1 : 17 * (A + 10) = 16 * A + 300) :
  (A + 10) = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l287_287383


namespace sin_alpha_value_sin_alpha_minus_pi_over_4_value_l287_287573

noncomputable def alpha (h₁ : Real) (h₂ : 0 < Real.pi / 2) : Prop :=
  ∃ α : Real, α ∈ Set.Ioo 0 (Real.pi / 2) ∧ (Real.tan (α / 2) + Real.cot (α / 2) = 5 / 2) 

theorem sin_alpha_value :
  ∀ α : Real, α ∈ Set.Ioo 0 (Real.pi / 2) →
  Real.tan (α / 2) + Real.cot (α / 2) = 5 / 2 →
  Real.sin α = 4 / 5 := 
by
  intros
  sorry

theorem sin_alpha_minus_pi_over_4_value :
  ∀ α : Real, α ∈ Set.Ioo 0 (Real.pi / 2) →
  Real.tan (α / 2) + Real.cot (α / 2) = 5 / 2 →
  Real.sin (α - Real.pi / 4) = (Real.sqrt 2) / 10 := 
by
  intros
  sorry

end sin_alpha_value_sin_alpha_minus_pi_over_4_value_l287_287573


namespace minimum_teachers_needed_l287_287845

theorem minimum_teachers_needed
  (math_teachers : ℕ) (physics_teachers : ℕ) (chemistry_teachers : ℕ)
  (max_subjects_per_teacher : ℕ) :
  math_teachers = 7 →
  physics_teachers = 6 →
  chemistry_teachers = 5 →
  max_subjects_per_teacher = 3 →
  ∃ t : ℕ, t = 5 ∧ (t * max_subjects_per_teacher ≥ math_teachers + physics_teachers + chemistry_teachers) :=
by
  repeat { sorry }

end minimum_teachers_needed_l287_287845


namespace cindy_correct_answer_l287_287966

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l287_287966


namespace race_last_part_length_l287_287761

theorem race_last_part_length (total_len first_part second_part third_part last_part : ℝ) 
  (h1 : total_len = 74.5) 
  (h2 : first_part = 15.5) 
  (h3 : second_part = 21.5) 
  (h4 : third_part = 21.5) :
  last_part = total_len - (first_part + second_part + third_part) → last_part = 16 :=
by {
  intros,
  sorry
}

end race_last_part_length_l287_287761


namespace object_speed_approx_13_64_mph_l287_287390

-- Definitions for the conditions given in the problem
def distance_feet : ℕ := 80
def time_seconds : ℕ := 4
def feet_per_mile : ℕ := 5280
def seconds_per_hour : ℕ := 3600

-- The mathematical proof problem represented as a Lean 4 statement
theorem object_speed_approx_13_64_mph :
  distance_feet / feet_per_mile / (time_seconds / seconds_per_hour) ≈ 13.64 :=
sorry

end object_speed_approx_13_64_mph_l287_287390


namespace rectangle_perimeter_l287_287257

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) : 2 * (L + B) = 186 :=
by
  sorry

end rectangle_perimeter_l287_287257


namespace sum_of_valid_b_l287_287504

theorem sum_of_valid_b :
  let discriminant (a b c : ℝ) := b^2 - 4 * a * c in
  let equation_has_one_solution (a b c : ℝ) := discriminant a b c = 0 in
  ∑ b in {b : ℝ | equation_has_one_solution 3 (b + 12) 4}, b = -24 :=
by
  sorry

end sum_of_valid_b_l287_287504


namespace sum_abs_diff_ge_two_n_minus_two_l287_287238

open Int

theorem sum_abs_diff_ge_two_n_minus_two (n : ℕ) (h : n ≥ 2) (circle: Fin n → ℕ) (h_circle : ∀ i, circle i ∈ Finset.range (n + 1) ∧ (Finset.range (n + 1) \ Finset.singleton 0).card = n ∧ (∑' i, circle i) = (∑ i in Finset.range (n+1), i)) :

  (∑ i in Finset.range n, Int.natAbs ((circle i) - (circle ((i + 1) % n)))) ≥ 2 * n - 2 := 
begin
  sorry
end

end sum_abs_diff_ge_two_n_minus_two_l287_287238


namespace smallest_positive_period_of_f_l287_287203

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem smallest_positive_period_of_f (ω φ : ℝ) (h1 : 0 < ω) (h2 : 0 < φ) (h3 : φ < Real.pi)
  (h4 : ∀ x : ℝ, f ω φ x = f ω φ (2 * Real.pi / 3 - x))
  (h5 : ∀ x : ℝ, f ω φ x = -f ω φ (Real.pi - x))
  (h6 : ∃ p > 0, ∀ x : ℝ, f ω φ (x + p) = f ω φ x ∧ p > Real.pi / 2) :
  ∃ p > 0, (∀ x : ℝ, f ω φ (x + p) = f ω φ x) ∧ p = 2 * Real.pi / 3 := 
begin
  sorry,
end

end smallest_positive_period_of_f_l287_287203


namespace rhombus_area_l287_287172

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem rhombus_area :
  let p1 := (0, 7.5)
      p2 := (8, 0)
      p3 := (0, -7.5)
      p4 := (-8, 0)
      d1 := distance p1 p3
      d2 := distance p2 p4
  in (d1 * d2) / 2 = 120 :=
by
  let p1 := (0, 7.5)
  let p2 := (8, 0)
  let p3 := (0, -7.5)
  let p4 := (-8, 0)
  let d1 := distance p1 p3
  let d2 := distance p2 p4
  sorry

end rhombus_area_l287_287172


namespace erased_number_is_214_l287_287482

def middle_value (a : ℤ) : list ℤ := [a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4]

theorem erased_number_is_214 (a b : ℤ) (h_sum : 8 * a - b = 1703)
  (h_bounds : -4 ≤ b ∧ b ≤ 4) (h_a : a = 213) (h_b : b = 8 * 213 - 1703) : 
  (middle_value a).sum - (a + b) = 1703 ∧ a + b = 214 :=
by { sorry }

end erased_number_is_214_l287_287482


namespace smallest_n_for_divisibility_l287_287074

theorem smallest_n_for_divisibility (n : ℕ) : 
  (∀ m, m > 0 → (315^2 - m^2) ∣ (315^3 - m^3) → m ≥ n) → 
  (315^2 - n^2) ∣ (315^3 - n^3) → 
  n = 90 :=
by
  sorry

end smallest_n_for_divisibility_l287_287074


namespace rate_of_interest_l287_287389

def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest (P : ℝ) (T : ℝ) (h : P * 2 = P + simple_interest P (100 / 15) T) :
  100 / 15 = 6.67 := by
  sorry

end rate_of_interest_l287_287389


namespace sum_of_integer_solutions_l287_287302

theorem sum_of_integer_solutions :
  let n_values := {n | |n-2| < |n-5| ∧ |n-5| < 7 ∧ n ∈ Int}
  (∑ n in n_values, n) = 3 :=
by
  sorry

end sum_of_integer_solutions_l287_287302


namespace largest_term_in_decomposition_of_five_cubed_is_29_l287_287082

theorem largest_term_in_decomposition_of_five_cubed_is_29 :
  let a := 1
  let d := 2
  let m := 5
  let middle_term_square (n : ℕ) : ℕ := a + (n - 1) * d + (n - 1) * d
  let middle_term := middle_term_square m / 2 - 1 / 2 (if even m then 0 else d)
  (middle_term_square m).div 2 = m -> 
  let first_term := middle_term
  first_term + 2 * (m - 1) = 29 := 
by
  let a := 1
  let d := 2
  let m := 5
  let middle_term_square (n : ℕ) : ℕ := a + (n - 1) * d + (n - 1) * d
  let middle_term := middle_term_square m / 2 - 1 / 2 (if even m then 0 else d)
  have h : first_term = 21 := by sorry 
  have h1: first_term + 2 * (m - 1) = 29 := by sorry
  exact (h, h1)

end largest_term_in_decomposition_of_five_cubed_is_29_l287_287082


namespace degree_of_g_l287_287712

def f (x : ℝ) : ℝ := 5 * x ^ 5 - 2 * x ^ 3 + x ^ 2 - 7

theorem degree_of_g (g : ℝ → ℝ) (h : ∀ x, degree (f x + g x) = 2) : degree g = 5 :=
sorry

end degree_of_g_l287_287712


namespace problem_b_problem_c_problem_d_l287_287090

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end problem_b_problem_c_problem_d_l287_287090


namespace track_length_l287_287030

theorem track_length (x : ℝ) (b_speed s_speed : ℝ) (b_dist1 s_dist1 s_dist2 : ℝ)
  (h1 : b_dist1 = 80)
  (h2 : s_dist1 = x / 2 - 80)
  (h3 : s_dist2 = s_dist1 + 180)
  (h4 : x / 4 * b_speed = (x / 2 - 80) * s_speed)
  (h5 : x / 4 * ((x / 2) - 100) = (x / 2 + 100) * s_speed) :
  x = 520 := 
sorry

end track_length_l287_287030


namespace relationship_among_ys_l287_287121

-- Define the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ :=
  -2 * x + b

-- Define the points on the graph
def y1 (b : ℝ) : ℝ :=
  linear_function (-2) b

def y2 (b : ℝ) : ℝ :=
  linear_function (-1) b

def y3 (b : ℝ) : ℝ :=
  linear_function 1 b

-- Theorem to prove the relation among y1, y2, y3
theorem relationship_among_ys (b : ℝ) : y1 b > y2 b ∧ y2 b > y3 b :=
by
  sorry

end relationship_among_ys_l287_287121


namespace sugar_cost_l287_287964

noncomputable def totalCostForBlueberryPie : ℚ := 13.5 + 2 + 1.5
noncomputable def totalCostForCherryPie : ℚ := 14 + 2 + 1.5
noncomputable def totalBudget : ℚ := 18

theorem sugar_cost : ∃ (sugar_cost : ℚ), sugar_cost = 1 ∧ 
  (totalCostForBlueberryPie < totalCostForCherryPie → 
  totalBudget - totalCostForBlueberryPie = sugar_cost) ∧ 
  (totalCostForCherryPie < totalCostForBlueberryPie → 
  totalBudget - totalCostForCherryPie = sugar_cost) ∧ 
  (totalCostForBlueberryPie = totalCostForCherryPie → 
  totalBudget - totalCostForBlueberryPie = sugar_cost) :=
begin
  sorry
end

end sugar_cost_l287_287964


namespace find_f_prime_at_2_l287_287333

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287333


namespace geometric_sequence_sum_sequence_terms_l287_287099

-- Define the sequence a_n given the initial condition and recurrence relation.
def sequence (a : ℤ) : ℕ → ℤ
| 0     := a
| (n+1) := (4 * n + 6) * sequence n + 4 * n + 10 / (2 * n + 1)

-- Problem 1: Show that the sequence { (a_n + 2) / (2n + 1) } is geometric with ratio 2 if a ≠ -2.
theorem geometric_sequence (a : ℤ) (h : a ≠ -2) :
  ∃ b_1 : ℤ, ∀ n : ℕ, (sequence a n + 2) / (2 * n + 1) = b_1 * 2^n - 2 :=
sorry

-- Problem 2: Show that for a = 1, the sum of the first n terms of a_n is (2n-1) * 2^(n-1).
theorem sum_sequence_terms (a : ℤ) (h : a = 1) (S_n : ℕ → ℤ) :
  ∀ n : ℕ, S_n n = (2 * n - 1) * 2^(n-1) :=
sorry

end geometric_sequence_sum_sequence_terms_l287_287099


namespace melting_point_ice_F_l287_287823

/-- Constants representing the boiling point and melting point in different temperature scales. -/
constant boiling_point_C : ℕ := 100 
constant boiling_point_F : ℕ := 212
constant melting_point_C : ℕ := 0

/-- Constants representing known temperatures in different temperature scales. -/
constant temp_pot_C : ℕ := 50
constant temp_pot_F : ℕ := 122

/-- Definition to find the melting point of ice in Fahrenheit given the conditions provided. -/
theorem melting_point_ice_F {F : ℕ} : F = 32 := by
  -- Proof steps would go here, but they are not required
  sorry

end melting_point_ice_F_l287_287823


namespace smallest_three_digit_divisible_l287_287076

theorem smallest_three_digit_divisible :
  ∃ (A B C : Nat), A ≠ 0 ∧ 100 ≤ (100 * A + 10 * B + C) ∧ (100 * A + 10 * B + C) < 1000 ∧
  (10 * A + B) > 9 ∧ (10 * B + C) > 9 ∧ 
  (100 * A + 10 * B + C) % (10 * A + B) = 0 ∧ (100 * A + 10 * B + C) % (10 * B + C) = 0 ∧
  (100 * A + 10 * B + C) = 110 :=
by
  sorry

end smallest_three_digit_divisible_l287_287076


namespace players_per_group_l287_287801

theorem players_per_group (new_players : ℕ) (returning_players : ℕ) (groups : ℕ) 
  (h1 : new_players = 48) 
  (h2 : returning_players = 6) 
  (h3 : groups = 9) : 
  (new_players + returning_players) / groups = 6 :=
by
  sorry

end players_per_group_l287_287801


namespace sum_of_digits_power_product_l287_287833

theorem sum_of_digits_power_product :
  ∑ (d ∈ Int.toDigits 10 (2^2009 * 5^2010 * 7)), d = 8 := 
by
  sorry

end sum_of_digits_power_product_l287_287833


namespace trajectory_eq_C_l287_287590

-- Define vertices A and B
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨3, 20⟩
def B : Point := ⟨3, 5⟩

-- Distance between two points
def dist (P₁ P₂ : Point) : ℝ :=
  Real.sqrt ((P₁.x - P₂.x) ^ 2 + (P₁.y - P₂.y) ^ 2)

-- Definition of isosceles triangle condition
def isIsoscelesAt (A B C : Point) : Prop :=
  dist A B = dist A C

-- Trajectory equation for point C
theorem trajectory_eq_C (A B : Point) (x y : ℝ) (h₃ : A.x = 3 ∧ A.y = 20) (h₅ : B.x = 3 ∧ B.y = 5) :
  (dist A B = dist A ⟨x, y⟩) → (x ≠ 3) → (x - 3) ^ 2 + (y - 20) ^ 2 = 225 :=
by
  sorry

end trajectory_eq_C_l287_287590


namespace tracy_dog_food_l287_287816

theorem tracy_dog_food
(f : ℕ) (c : ℝ) (m : ℕ) (d : ℕ)
(hf : f = 4) (hc : c = 2.25) (hm : m = 3) (hd : d = 2) :
  (f * c / m) / d = 1.5 :=
by
  sorry

end tracy_dog_food_l287_287816


namespace history_students_count_l287_287953

noncomputable theory
open_locale classical

def total_students := 70
def both_subjects := 10
def hist_geog_relation (hist_class geog_class : ℕ) := hist_class = 2 * (geog_class + both_subjects) - both_subjects

theorem history_students_count (hist_class geog_class only_geog : ℕ) 
  (h_total : total_students = hist_class + geog_class + both_subjects) 
  (h_relation : hist_geog_relation hist_class geog_class) : 
  52 = hist_class :=
begin
  sorry
end

end history_students_count_l287_287953


namespace original_proposition_true_negation_of_original_proposition_l287_287279

theorem original_proposition_true :
  (∀ x : ℝ, x > 2 → (log (x - 1) + x^2 + 4 > 4 * x)) :=
by
  sorry

theorem negation_of_original_proposition :
  (∃ x : ℝ, x > 2 ∧ (log (x - 1) + x^2 + 4 <= 4 * x)) :=
by
  sorry

end original_proposition_true_negation_of_original_proposition_l287_287279


namespace min_days_required_l287_287427

theorem min_days_required (n : ℕ) (h1 : n ≥ 1) (h2 : 2 * (2^n - 1) ≥ 100) : n = 6 :=
sorry

end min_days_required_l287_287427


namespace math_proof_problem_l287_287713

noncomputable def first_problem_statement : Prop :=
  let f : ℝ → ℝ := λ x, real.exp x - x + 1 in
  ∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), (2 : ℝ) ≤ f x ∧ f x ≤ real.exp 2 - 1

noncomputable def second_problem_statement : Prop :=
  ∀ (a b : ℝ), (∀ x : ℝ, real.exp x - a * x + b ≥ 0) → (a - b ≤ real.exp 1)

theorem math_proof_problem : first_problem_statement ∧ second_problem_statement :=
  sorry

end math_proof_problem_l287_287713


namespace range_of_x_l287_287104

variable (a b : ℝ) (t : ℝ)

def f (x : ℝ) : ℝ := |x - t| + |x + (1 / t)|

theorem range_of_x (a_pos : 0 < a) (b_pos : 0 < b) (a_plus_b : a + b = 2) (t_ne_zero : t ≠ 0) :
  (∃ (x : ℝ), f t x = 2) →
  ∀ (x : ℝ), (f t x = 2 → -1 ≤ x ∧ x ≤ 1) :=
sorry

end range_of_x_l287_287104


namespace compute_cross_product_l287_287639

variable (a b c : ℝ^3)

axiom cross_product_a_b : (a.cross b) = ⟨2, -3, 1⟩
axiom cross_product_a_c : (a.cross c) = ⟨1, 1, -1⟩

theorem compute_cross_product :
  2 • (a.cross (4 • b + 3 • c)) = ⟨22, -18, 2⟩ := by
  sorry

end compute_cross_product_l287_287639


namespace efficiency_increase_l287_287221

theorem efficiency_increase:
  ∀ (days_mary days_rosy : ℕ), 
  days_mary = 28 → 
  days_rosy = 20 → 
  let efficiency_mary := (1 : ℚ) / days_mary 
  in let efficiency_rosy := (1 : ℚ) / days_rosy 
  in (efficiency_rosy - efficiency_mary) / efficiency_mary * 100 = 40 := 
by
  intros days_mary days_rosy h_mary h_rosy
  have h_eff_mary : (1 : ℚ) / days_mary = 1 / 28, from h_mary ▸ rfl
  have h_eff_rosy : (1 : ℚ) / days_rosy = 1 / 20, from h_rosy ▸ rfl
  simp [h_eff_mary, h_eff_rosy]
  norm_num
  sorry

end efficiency_increase_l287_287221


namespace find_f_prime_at_2_l287_287335

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287335


namespace estimate_white_balls_l287_287669

theorem estimate_white_balls :
  (∃ x : ℕ, (6 / (x + 6) : ℝ) = 0.2 ∧ x = 24) :=
by
  use 24
  sorry

end estimate_white_balls_l287_287669


namespace problem_I_problem_II_l287_287128

noncomputable def f (a x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

theorem problem_I (a : ℝ) (h_a : a ≥ 0) :
  (a = 0 → (∀ x ∈ Ioo 0 1, 0 ≤ deriv (f a) x) ∧ (∀ x ∈ Ioo 1 real_top, deriv (f a) x ≤ 0)) ∧
  (0 < a ∧ a < 1 → (∀ x ∈ Ioo 0 1, 0 ≤ deriv (f a) x) ∧ (∀ x ∈ Ioo 1 (1/a), deriv (f a) x ≤ 0) ∧
   (∀ x ∈ Ioo (1/a) real_top, 0 ≤ deriv (f a) x)) ∧
  (a = 1 → ∀ x > 0, 0 ≤ deriv (f a) x) ∧
  (a > 1 → (∀ x ∈ Ioo 0 (1/a), 0 ≤ deriv (f a) x) ∧ (∀ x ∈ Ioo (1/a) 1, deriv (f a) x ≤ 0) ∧
   (∀ x ∈ Ioo 1 real_top, 0 ≤ deriv (f a) x)) :=
sorry

theorem problem_II :
  2 * Real.exp x > Real.exp (5 / 2) * (f (-2) x + 2 * x) :=
sorry

end problem_I_problem_II_l287_287128


namespace determine_lambda_mu_l287_287750

theorem determine_lambda_mu (A B C : ℝ × ℝ) (λ μ : ℝ) :
    A = (1, 0) ∧
    B = (0, 1) ∧
    (C.1 < 0 ∧ C.2 > 0) ∧
    (∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧ (C.1, C.2) = (2 * Real.cos θ, 2 * Real.sin θ)) ∧
    ((C.1, C.2) = (λ * A.1 + μ * B.1, λ * A.2 + μ * B.2)) →
    λ = -Real.sqrt 3 ∧ μ = 1 :=
by
  sorry

end determine_lambda_mu_l287_287750


namespace smallest_x_y_values_l287_287636

theorem smallest_x_y_values : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 0.75 = y / (242 + x) ∧ x = 2 ∧ y = 183 :=
by
  sorry

end smallest_x_y_values_l287_287636


namespace infinite_benelux_couples_l287_287022

def prime_divisors (n : ℕ) : Set ℕ :=
  { p | nat.prime p ∧ p ∣ n }

def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧ prime_divisors m = prime_divisors n ∧ prime_divisors (m + 1) = prime_divisors (n + 1)

example : is_benelux_couple 2 8 :=
by sorry

example : is_benelux_couple 6 48 :=
by sorry

example : is_benelux_couple 14 224 :=
by sorry

theorem infinite_benelux_couples : ∀ k ≥ 2, is_benelux_couple (2^k - 2) (2^k * (2^k - 2)) :=
by sorry

end infinite_benelux_couples_l287_287022


namespace price_after_reductions_l287_287433

theorem price_after_reductions (P : ℝ) : 
  let first_day_price := 0.9 * P 
  let second_day_price := 0.85 * first_day_price 
  let third_day_price := 0.95 * second_day_price 
  (third_day_price / P) * 100 = 72.675 := by 
  let first_day_price := 0.9 * P 
  let second_day_price := 0.85 * first_day_price
  let third_day_price := 0.95 * second_day_price
  have : third_day_price = 0.72675 * P := calc
    third_day_price
      = 0.95 * (0.85 * (0.9 * P)) : rfl
    ... = 0.95 * 0.85 * 0.9 * P : by ring
    ... = 0.72675 * P : by norm_num
  calc
    (third_day_price / P) * 100
      = (0.72675 * P / P) * 100 : by rw this
    ... = 0.72675 * 100 : by field_simp
    ... = 72.675 : by norm_num

end price_after_reductions_l287_287433


namespace solution_set_f_l287_287122

noncomputable def f (x : ℝ) : ℝ := sorry -- The differentiable function f

axiom f_deriv_lt (x : ℝ) : deriv f x < x -- Condition on the derivative of f
axiom f_at_2 : f 2 = 1 -- Given f(2) = 1

theorem solution_set_f : ∀ x : ℝ, f x < (1 / 2) * x^2 - 1 ↔ x > 2 :=
by sorry

end solution_set_f_l287_287122


namespace tangent_line_intersection_x_axis_l287_287899

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287899


namespace tom_age_ratio_l287_287287

variable (T N : ℕ)

theorem tom_age_ratio (h_sum : T = T) (h_relation : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end tom_age_ratio_l287_287287


namespace find_uv_solution_l287_287246

theorem find_uv_solution :
  ∃ (u v : ℝ), 
    (\begin{pmatrix} 3 \\ -1 \end{pmatrix} + u • \begin{pmatrix} 9 \\ -6 \end{pmatrix}) = 
    (\begin{pmatrix} 4 \\ 0 \end{pmatrix} + v • \begin{pmatrix} -3 \\ 4 \end{pmatrix} + \begin{pmatrix} 1 \\ 1 \end{pmatrix}) ∧ 
    (u = 7 / 9) ∧ 
    (v = -5 / 3) :=
begin
  use 7 / 9,
  use -5 / 3,
  simp,
  split,
  simp [Matrix.add, Matrix.smul, Matrix.row_const],
  split;
  ring,
  sorry
end

end find_uv_solution_l287_287246


namespace seedling_survival_rate_estimate_l287_287910

theorem seedling_survival_rate_estimate :
  let rates := [0.800, 0.870, 0.923, 0.883, 0.890, 0.915, 0.905, 0.897, 0.902] in
  let avg := (rates.foldl (λ acc r, acc + r) 0) / (rates.length : ℝ) in
  Float.roundDecimal avg 1 = 0.9 :=
by
  sorry

end seedling_survival_rate_estimate_l287_287910


namespace median_score_interval_l287_287043

theorem median_score_interval :
  let scores := [15, 15, 25, 10, 15, 20]
  ∑ i in scores, i = 100 →
  let cumulative := (List.scanl (+) 0 scores).drop 1
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 45 < cumulative.getD (i - 1) 0 ∧ cumulative.getD i 0 ≥ 50 →
  i - 1 = 2 :=
by
  sorry

end median_score_interval_l287_287043


namespace monika_bags_of_beans_l287_287736

theorem monika_bags_of_beans :
  ∃ (b : ℕ), let mall_spent := 250
             let movie_cost_per_movie := 24
             let num_movies := 3
             let movie_spent := num_movies * movie_cost_per_movie
             let total_before_beans := mall_spent + movie_spent
             let total_spent := 347
             let cost_per_bag := 1.25
             let beans_spent := total_spent - total_before_beans
             let bags := beans_spent / cost_per_bag
             bags = 20 := 
begin
  use 20,
  sorry
end

end monika_bags_of_beans_l287_287736


namespace hexagon_inscribed_in_square_area_l287_287596

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * side_length^2

theorem hexagon_inscribed_in_square_area (AB BC : ℝ) (BDEF_square : BDEF_is_square) (hAB : AB = 2) (hBC : BC = 2) :
  hexagon_area (2 * Real.sqrt 2) = 12 * Real.sqrt 3 :=
by
  sorry

-- Definitions to assume the necessary conditions in the theorem (placeholders)
-- Assuming a structure of BDEF_is_square to represent the property that BDEF is a square
structure BDEF_is_square :=
(square : Prop)

end hexagon_inscribed_in_square_area_l287_287596


namespace trapezoid_area_l287_287449

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l287_287449


namespace intersection_points_of_circles_l287_287751

-- Variables and assumptions
variable {C : Type} [metric_space C] {O P: C} {r r₁ r₂: ℝ}

-- Conditions
def valid_setup (P O: C) (r r₁: ℝ) (d: ℝ) := 
  dist P O = d ∧ r = r₁

-- Statement
theorem intersection_points_of_circles (P O: C) (r1 r2 d: ℝ) :
  valid_setup P O 4 7 →
  r2 = 5 →
  1 < d ∧ d < r1 + r2 →
  ∃ (X1 X2: C), dist X1 P = r2 ∧ dist X2 P = r2 ∧ dist X1 O = 4 ∧ dist X2 O = 4 ∧ X1 ≠ X2 :=
by
  sorry

end intersection_points_of_circles_l287_287751


namespace score_comparison_l287_287164

theorem score_comparison :
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  combined_score - opponent_score = 143 :=
by
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  sorry

end score_comparison_l287_287164


namespace petya_cannot_ensure_three_consecutive_same_l287_287444

theorem petya_cannot_ensure_three_consecutive_same (
  n : ℕ,
  h : n ≥ 2
) : ¬ (∀ (f : Fin (2*n) → (Fin (2*n) → ℕ)) 
        (h1 : ∀ i, f i 0 ≠ f (i+1) % (2*n) 0 ∨ f i 1 ≠ f (i+1) % (2*n) 1), 
      ∃ i, f i 0 = f (i+1) % (2*n) 0 ∧ f i 0 = f (i+2) % (2*n) 0) := 
sorry

end petya_cannot_ensure_three_consecutive_same_l287_287444


namespace iterative_average_difference_is_8_875_l287_287947

noncomputable def iterative_average_difference : ℝ :=
  let avg := λ a b, (a + b) / 2
  let max_iter_avg := avg (avg (avg (avg (-1) 0) 5) 10) 15
  let min_iter_avg := avg (avg (avg (avg 15 10) 5) 0) (-1)
  max_iter_avg - min_iter_avg

theorem iterative_average_difference_is_8_875 :
  iterative_average_difference = 8.875 :=
by
  sorry

end iterative_average_difference_is_8_875_l287_287947


namespace specific_heat_capacity_proof_l287_287274

-- Define the conditions
variables (p1 p2 V1 V2 : ℝ)
variables (h1 : p1 > p2) (h2 : V1 < V2)

-- Slope of the line in the P-V diagram
def a : ℝ := (p2 - p1) / (V2 - V1)

-- Specific heat capacity as given in the solution
def specific_heat_capacity (C_V R V : ℝ) : ℝ :=
  C_V + R * (p1 + a * (V - V1)) / (p1 + a * (2 * V - V1))

-- The hypothesis to be proved
theorem specific_heat_capacity_proof (C_V R V : ℝ) :
  specific_heat_capacity C_V R V = C_V + R * (p1 + a * (V - V1)) / (p1 + a * (2 * V - V1)) :=
sorry

end specific_heat_capacity_proof_l287_287274


namespace prove_BE_EQ_EZ_EQ_ZC_ratio_areas_BDE_ABC_l287_287705

-- Problem statement: Let ABC be an equilateral triangle, with D as described
-- and E, Z as the perpendicular bisectors' intersecting points on BC.
variable {A B C D E Z : Point}
variable [equilateral ABC]
variable [is_angle_bisector D B C]
variable [perpendicular_bisector E B D]
variable [perpendicular_bisector Z C D]

-- Prove that BE = EZ = ZC
theorem prove_BE_EQ_EZ_EQ_ZC : BE = EZ ∧ EZ = ZC ∧ BE = ZC :=
by sorry

-- Prove the ratio of areas of triangles BDE to ABC is 1/9
theorem ratio_areas_BDE_ABC : area (triangle B D E) / area (triangle A B C) = 1 / 9 :=
by sorry

end prove_BE_EQ_EZ_EQ_ZC_ratio_areas_BDE_ABC_l287_287705


namespace max_sum_real_part_l287_287725

-- This would likely be useful for complex number root calculations
noncomputable def max_real_part (w : ℕ → ℂ) : ℝ :=
∑ j in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, (w j).re

theorem max_sum_real_part :
  ∃ w_j : ℕ → ℂ, (∀ j, w_j j = z_j j ∨ w_j j = -complex.I * z_j j) ∧ max_real_part w_j = 64 * real.sqrt 5 :=
sorry

end max_sum_real_part_l287_287725


namespace periodicity_of_f_l287_287040

noncomputable def f : ℕ+ → ℕ+ := sorry  -- Placeholder for the function from positive integers to positive integers.

def periodic_sequence (s : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ n, s (n + p) = s n

theorem periodicity_of_f (h1 : ∀ (m n : ℕ+), ∃ k : ℕ+, ((f^[n] m).val - m.val) / n.val = k.val)
    (h2 : (∃ S : finset ℕ+, ∀ n : ℕ+, n ∉ S → (∃ m : ℕ+, f m = n))) :
  periodic_sequence (λ n, (f n) - n) :=
by
  sorry

end periodicity_of_f_l287_287040


namespace general_term_formula_l287_287802

def a : ℕ → ℝ
def S : ℕ → ℝ

axiom a_initial : a 1 = 1
axiom a_recurrence (n : ℕ) (hn : n ≥ 2) : a n = (1 / 2) + (1 / 2) * sqrt (1 + 8 * S (n - 1))
axiom S_definition (n : ℕ) : S n = ∑ i in Finset.range (n + 1), a i

theorem general_term_formula (n : ℕ) : a n = n := by
  sorry

end general_term_formula_l287_287802


namespace num_symmetric_patterns_l287_287675

-- Define the grid and rectangles
def grid : Type := array (fin 4) (array (fin 4) (option (fin 4)))

-- Define what it means for a grid to have the required properties.
def is_valid_pattern (pattern : grid) : Prop :=
  -- Every 2x1 rectangle should be represented exactly 8 times
  (count pattern = 8) ∧
  -- Assume 2 lines of symmetry condition without proof details.
  (has_two_lines_of_symmetry pattern)

-- Define counting distinct configurations.
def count_valid_patterns : nat :=
  count {pattern : grid // is_valid_pattern pattern}

-- Theorem stating the count of valid symmetrical patterns.
theorem num_symmetric_patterns : count_valid_patterns = 6 := by
  sorry

end num_symmetric_patterns_l287_287675


namespace percentage_of_diameter_l287_287782

variable (d_R d_S r_R r_S : ℝ)
variable (A_R A_S : ℝ)
variable (pi : ℝ) (h1 : pi > 0)

theorem percentage_of_diameter 
(h_area : A_R = 0.64 * A_S) 
(h_radius_R : r_R = d_R / 2) 
(h_radius_S : r_S = d_S / 2)
(h_area_R : A_R = pi * r_R^2) 
(h_area_S : A_S = pi * r_S^2) 
: (d_R / d_S) * 100 = 80 := by
  sorry

end percentage_of_diameter_l287_287782


namespace smaller_tv_diagonal_l287_287805

noncomputable def diagonal_length_smaller_tv (L S D : ℝ) : Prop :=
  (L^2 = S^2 + 40) ∧ (2 * L^2 = 21^2) → D = 19

theorem smaller_tv_diagonal : ∃ L S D : ℝ, diagonal_length_smaller_tv L S D :=
begin
  use [√(21^2 / 2), √180.5, 19],
  split; intros h; finish,
  sorry -- fill in the proof later
end

end smaller_tv_diagonal_l287_287805


namespace longer_diagonal_of_rhombus_l287_287256

theorem longer_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (h₁ : d1 = 12) (h₂ : area = 120) :
  d2 = 20 :=
by
  sorry

end longer_diagonal_of_rhombus_l287_287256


namespace fred_remaining_money_l287_287086

-- Define Fred's weekly allowance
def allowance : ℕ := 16

-- Define the money spent on movies as half of the allowance
def spent_on_movies (a : ℕ) : ℕ := a / 2

-- Define the money earned from washing the car
def earned (e : ℕ) : ℕ := e

-- Define Fred's remaining money after spending on movies and earning from washing the car
def remaining_money (a m e : ℕ) : ℕ := a - m + e

-- The theorem to be proven
theorem fred_remaining_money (a : ℕ) (e : ℕ) : remaining_money a (spent_on_movies a) e = 14 :=
by
  -- Use the given conditions: allowance = 16 and earned = 6
  have h_allowance : a = 16 := rfl
  have h_earned : e = 6 := rfl
  -- Substitute known values into the remaining_money definition
  rw [h_allowance, h_earned]
  -- Calculations to simplify and arrive at the final result
  calc
    remaining_money 16 (spent_on_movies 16) 6
      = 16 - (16 / 2) + 6 : rfl
      ... = 16 - 8 + 6 : by norm_num
      ... = 8 + 6 : rfl
      ... = 14 : by norm_num

end fred_remaining_money_l287_287086


namespace total_ticket_sales_l287_287814

theorem total_ticket_sales 
  (tickets_A : ℕ := 2900)
  (tickets_B : ℕ := 1600)
  (price_A : ℝ := 8)
  (price_B : ℝ := 4.25)
  (total_tickets : ℕ := 4500)
  (h_total : tickets_A + tickets_B = total_tickets) :
  tickets_A * price_A + tickets_B * price_B = 30000 :=
by
  rw [mul_comm tickets_A price_A, mul_comm tickets_B price_B]
  sorry

end total_ticket_sales_l287_287814


namespace typing_and_proofreading_time_is_19_78_l287_287698

noncomputable def collective_typing_and_proofreading_time : ℝ := 
  let pages := 10
  let jonathan_speed := pages / 40
  let susan_speed := pages / 30
  let abel_speed := pages / 35
  let jack_speed := pages / 24
  let total_speed := jonathan_speed + susan_speed + abel_speed + jack_speed
  let typing_time := pages / total_speed
  let jonathan_errors := (typing_time / 5).ceil
  let susan_errors := (typing_time / 7).ceil
  let abel_errors := (typing_time / 10).ceil
  let jack_errors := (typing_time / 4).ceil
  let correction_time := 2 * (jonathan_errors + susan_errors + abel_errors + jack_errors)
  typing_time + correction_time

theorem typing_and_proofreading_time_is_19_78 :
  collective_typing_and_proofreading_time ≈ 19.78 := sorry

end typing_and_proofreading_time_is_19_78_l287_287698


namespace min_number_knights_l287_287666

theorem min_number_knights (h1 : ∃ n : ℕ, n = 7) (h2 : ∃ s : ℕ, s = 42) (h3 : ∃ l : ℕ, l = 24) :
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ k * (7 - k) = 12 ∧ k = 3 :=
by
  sorry

end min_number_knights_l287_287666


namespace complex_number_solution_l287_287207

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (hz : z * (i - 1) = 2 * i) : 
z = 1 - i :=
by 
  sorry

end complex_number_solution_l287_287207


namespace sin_cos_diff_eq_neg_cos_sum_l287_287858

theorem sin_cos_diff_eq_neg_cos_sum (α β : ℝ) :
  sin α * sin β - cos α * cos β = - cos (α + β) :=
by
  sorry

end sin_cos_diff_eq_neg_cos_sum_l287_287858


namespace train_crosses_second_platform_in_20_seconds_l287_287940

-- Define the conditions for the problem

def length_of_train : ℝ := 150
def length_of_first_platform : ℝ := 150
def time_to_cross_first_platform : ℝ := 15
def length_of_second_platform : ℝ := 250

-- Define the speeds based on given conditions
def speed_of_train : ℝ := (length_of_train + length_of_first_platform) / time_to_cross_first_platform

-- Define the total distance for the second platform crossing
def total_distance_second_platform : ℝ := length_of_train + length_of_second_platform

-- Define the expected time to cross the second platform
def expected_time_to_cross_second_platform : ℝ := total_distance_second_platform / speed_of_train

-- Lean theorem to state the proof goal
theorem train_crosses_second_platform_in_20_seconds : 
  expected_time_to_cross_second_platform = 20 := by
  sorry

end train_crosses_second_platform_in_20_seconds_l287_287940


namespace max_min_values_l287_287527

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  (∀ x ∈ (Set.Icc 0 2), f x ≤ 5) ∧ (∃ x ∈ (Set.Icc 0 2), f x = 5) ∧
  (∀ x ∈ (Set.Icc 0 2), f x ≥ -15) ∧ (∃ x ∈ (Set.Icc 0 2), f x = -15) :=
by
  sorry

end max_min_values_l287_287527


namespace math_proof_problem_l287_287651

variable (a : ℝ) (f : ℝ → ℝ)

noncomputable def problem_statement : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (∀ x, f (a^x) = x) ∧ (f (sqrt a) = a) → f = λ x, log (1/2) x

theorem math_proof_problem : problem_statement a f :=
sorry

end math_proof_problem_l287_287651


namespace calculate_g_2_3_neg1_l287_287714

def g (m n p : ℝ) : ℝ := (m + p^2) / (m - n)

theorem calculate_g_2_3_neg1 : g 2 3 (-1) = -3 := 
by
  sorry

end calculate_g_2_3_neg1_l287_287714


namespace circumcircle_no_other_integer_points_l287_287706

theorem circumcircle_no_other_integer_points :
  let C := {P : ℝ × ℝ | (P.fst - 989)^2 + (P.snd - 989)^2 = 2 * (989:ℝ)^2} in
  ∀ (x y : ℤ), (⟨(x : ℝ), y⟩ ∈ C) → (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 1978) ∨ (x = 1978 ∧ y = 0) ∨ (x = 1978 ∧ y = 1978) :=
by {
  sorry
}

end circumcircle_no_other_integer_points_l287_287706


namespace opposite_of_neg_five_l287_287793

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l287_287793


namespace segment_arc_sum_l287_287179

noncomputable def sqrt_real (x : ℝ) : ℝ := Real.sqrt x

def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def line (x y : ℝ) : Prop := y = sqrt_real 3 * x - 4

theorem segment_arc_sum 
  (A B : ℝ × ℝ) 
  (hAcircle : circle A.1 A.2) 
  (hBcircle : circle B.1 B.2) 
  (hAline : line A.1 A.2) 
  (hBline : line B.1 B.2) :
  let distance_AB := (2 * sqrt_real 3)^2 + (2 - -4)^2
  let angle_aob := Real.pi
  let major_arc_length := angle_aob * 4
  sqrt_real distance_AB + major_arc_length = 4 * sqrt_real 3 + 4 * Real.pi :=
sorry

end segment_arc_sum_l287_287179


namespace hephaestus_guaranteed_victory_l287_287399

theorem hephaestus_guaranteed_victory (α : ℝ) (h_α : α ≥ 1) :
  (∃ (n : ℕ), ∀ (initial_flooded : set (ℤ × ℤ)),
    ∃ (levees : list (ℤ × ℤ) → list (ℤ × ℤ)),
    (∀ (k : ℕ), (length (levees k) ≤ α * k) ∧ 
                ((flooded_after_turn k initial_flooded levees) ⊆
                 (interior (path_to_set (levees k))))) = (α > 2) :=
sorry

-- Assume some utility functions and auxiliary definitions (like flooded_after_turn, path_to_set, and interior) exist.

end hephaestus_guaranteed_victory_l287_287399


namespace route_time_difference_l287_287212

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l287_287212


namespace paths_from_C_to_D_l287_287506

theorem paths_from_C_to_D :
  let total_moves := 10 in
  let right_moves := 6 in
  let up_moves := 4 in
  total_moves = right_moves + up_moves →
  @Finset.card _ _ (Finset.range total_moves).powerset (fun s => s.card = up_moves) = 210 := 
by sorry

end paths_from_C_to_D_l287_287506


namespace oranges_after_eating_l287_287222

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0
def final_oranges : ℝ := 75.0

theorem oranges_after_eating :
  initial_oranges - eaten_oranges = final_oranges := by
  sorry

end oranges_after_eating_l287_287222


namespace total_rental_cost_of_remaining_4_DVDs_l287_287210

theorem total_rental_cost_of_remaining_4_DVDs
  (total_cost_7_DVDs : ℤ) (cost_per_DVD : ℤ) (num_cheap_DVDs : ℕ) (cost_remaining_4_DVDs : ℤ) :
  total_cost_7_DVDs = 1260 →
  cost_per_DVD = 150 →
  num_cheap_DVDs = 3 →
  cost_remaining_4_DVDs = 810 :=
begin
  sorry
end

end total_rental_cost_of_remaining_4_DVDs_l287_287210


namespace divisor_in_second_division_l287_287744

theorem divisor_in_second_division 
  (n : ℤ) 
  (h1 : (68 : ℤ) * 269 = n) 
  (d q : ℤ) 
  (h2 : n = d * q + 1) 
  (h3 : Prime 18291):
  d = 18291 := by
  sorry

end divisor_in_second_division_l287_287744


namespace distinct_flavors_count_l287_287248

-- Define the number of available candies
def red_candies := 3
def green_candies := 2
def blue_candies := 4

-- Define what it means for a flavor to be valid: includes at least one candy of each color.
def is_valid_flavor (x y z : Nat) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x ≤ red_candies ∧ y ≤ green_candies ∧ z ≤ blue_candies

-- Define what it means for two flavors to have the same ratio
def same_ratio (x1 y1 z1 x2 y2 z2 : Nat) : Prop :=
  x1 * y2 * z2 = x2 * y1 * z1

-- Define the proof problem: the number of distinct flavors
theorem distinct_flavors_count :
  ∃ n, n = 21 ∧ ∀ (x y z : Nat), is_valid_flavor x y z ↔ (∃ x' y' z', is_valid_flavor x' y' z' ∧ ¬ same_ratio x y z x' y' z') :=
sorry

end distinct_flavors_count_l287_287248


namespace radius_ratio_eq_inv_sqrt_5_l287_287680

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l287_287680


namespace correctly_differentiated_l287_287840

-- Define the functions and their expected derivatives
def f1 := (fun x => x)
def f2 := (fun x => sin 2)
def f3 := (fun x => 1 / x)
def f4 := (fun x => sqrt x)

-- Define expected derivatives
def df1 := (fun x' => 1)
def df2 := (fun x' => 0)
def df3 := (fun x' => - 1 / x^2)
def df4 := (fun x' => 1 / (2 * sqrt x))

theorem correctly_differentiated :
  (derivative f1 == df1) ∧
  (derivative f4 == df4) ∧
  ¬ (derivative f2 == df2) ∧
  ¬ (derivative f3 == df3) := by
  sorry

end correctly_differentiated_l287_287840


namespace sum_of_perimeters_is_270_l287_287021

-- Define the original side length of T1
def side_length_T1 : ℝ := 45

-- Define the function to calculate the side length of the i-th triangle
def side_length (n : ℕ) : ℝ :=
  side_length_T1 * (1/2) ^ n

-- Define the function to calculate the perimeter of the i-th triangle
def perimeter (n : ℕ) : ℝ :=
  3 * side_length n

-- Define the sum of the perimeters of all triangles
def sum_perimeters : ℝ :=
  ∑' n, perimeter n

theorem sum_of_perimeters_is_270 :
  sum_perimeters = 270 := by
  sorry

end sum_of_perimeters_is_270_l287_287021


namespace pipeA_filling_time_l287_287937

noncomputable def pipeA_fill_time := 60

def pipeB_fill_time := 40
def total_fill_time := 30
def half_fill_time := total_fill_time / 2

theorem pipeA_filling_time :
  let t := pipeA_fill_time in
  let rateB := 1 / pipeB_fill_time in
  let rateA := 1 / t in
  let rateA_B := rateA + rateB in
  rateB * half_fill_time + rateA_B * half_fill_time = 1 →
  t = 60 :=
by
  sorry

end pipeA_filling_time_l287_287937


namespace derivative_at_2_l287_287372

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287372


namespace find_difference_between_larger_and_fraction_smaller_l287_287069

theorem find_difference_between_larger_and_fraction_smaller
  (x y : ℝ) 
  (h1 : x + y = 147)
  (h2 : x - 0.375 * y = 4) : x - 0.375 * y = 4 :=
by
  sorry

end find_difference_between_larger_and_fraction_smaller_l287_287069


namespace figure_100_squares_l287_287060

def figure_squares (n : ℕ) : ℕ :=
  2 * n ^ 2 + 2 * n + 1

theorem figure_100_squares : figure_squares 100 = 20201 := 
  by simp [figure_squares] -- Simplify the expression
    sorry -- Proof omitted

end figure_100_squares_l287_287060


namespace magical_stack_example_l287_287779

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def belongs_to_pile_A (card : ℕ) (n : ℕ) : Prop :=
  card <= n

def belongs_to_pile_B (card : ℕ) (n : ℕ) : Prop :=
  n < card

def magical_stack (cards : ℕ) (n : ℕ) : Prop :=
  ∀ (card : ℕ), (belongs_to_pile_A card n ∨ belongs_to_pile_B card n) → 
  (card + n) % (2 * n) = 1

-- The theorem to prove
theorem magical_stack_example :
  ∃ (n : ℕ), magical_stack 482 n ∧ (2 * n = 482) :=
by
  sorry

end magical_stack_example_l287_287779


namespace find_f_prime_at_2_l287_287350

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287350


namespace relationship_M_N_l287_287574

variable (a : ℝ)

def M := a^2 + 4 * a + 1
def N := 2 * a - 1 / 2

theorem relationship_M_N : M a > N a := by
  sorry

end relationship_M_N_l287_287574


namespace monotonic_intervals_minimum_value_1_minimum_value_2_minimum_value_3_l287_287616

noncomputable def f (x a : ℝ) : ℝ := 
  Real.log x + (1 - x) / (a * x)

theorem monotonic_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x → x < 1 → f' a x < 0) ∧ 
  (∀ x, 1 < x → f' a x > 0) := 
sorry

theorem minimum_value_1 (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ∈ set.Icc 2 3, f a x ≥ f a 2 := 
sorry

theorem minimum_value_2 (a : ℝ) (h1 : 1 / 3 < a) (h2 : a < 1 / 2) :
  ∀ x ∈ set.Icc 2 3, f a x ≥ f a (1 / a) := 
sorry

theorem minimum_value_3 (a : ℝ) (h : 0 < a ∧ a ≤ 1 / 3) :
  ∀ x ∈ set.Icc 2 3, f a x ≥ f a 3 := 
sorry

end monotonic_intervals_minimum_value_1_minimum_value_2_minimum_value_3_l287_287616


namespace range_of_g_l287_287991

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) 
  - 3 * (Real.arcsin (x / 3))^2 + (Real.pi^2 / 4) * (x^2 - 3 * x + 9)

theorem range_of_g : 
  ∀ x, -3 ≤ x ∧ x ≤ 3 → ∃ y, y = g x ∧ y ∈ set.Icc ((Real.pi^2) / 4) (37 * (Real.pi^2) / 4) :=
  sorry

end range_of_g_l287_287991


namespace martha_savings_l287_287732

-- Definitions based on conditions
def weekly_latte_spending : ℝ := 4.00 * 5
def weekly_iced_coffee_spending : ℝ := 2.00 * 3
def total_weekly_coffee_spending : ℝ := weekly_latte_spending + weekly_iced_coffee_spending
def annual_coffee_spending : ℝ := total_weekly_coffee_spending * 52
def savings_percentage : ℝ := 0.25

-- The theorem to be proven
theorem martha_savings : annual_coffee_spending * savings_percentage = 338.00 := by
  sorry

end martha_savings_l287_287732


namespace option_A_option_B_option_C_option_D_l287_287379

-- Option A
theorem option_A (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 
  (x-1)^2 + x*(x-4) + (x-2)*(x+2) ≠ 0 := 
sorry

-- Option B
theorem option_B (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^3 + (1/x)^3 - 3 = 15 := 
sorry

-- Option C
theorem option_C (x : ℝ) (a b c : ℝ) (h_a : a = 1 / 20 * x + 20) (h_b : b = 1 / 20 * x + 19) (h_c : c = 1 / 20 * x + 21) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := 
sorry

-- Option D
theorem option_D (x m n : ℝ) (h : 2*x^2 - 8*x + 7 = 0) (h_roots : m + n = 4 ∧ m * n = 7/2) : 
  Real.sqrt (m^2 + n^2) = 3 := 
sorry

end option_A_option_B_option_C_option_D_l287_287379


namespace min_sum_x8y4z_l287_287723

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l287_287723


namespace probability_bons_wins_even_rolls_l287_287985
noncomputable def probability_of_Bons_winning (p6 : ℚ) (p_not6 : ℚ) : ℚ := 
  let r := p_not6^2
  let a := p_not6 * p6
  a / (1 - r)

theorem probability_bons_wins_even_rolls : 
  let p6 := (1 : ℚ) / 6
  let p_not6 := (5 : ℚ) / 6
  probability_of_Bons_winning p6 p_not6 = (5 : ℚ) / 11 := 
  sorry

end probability_bons_wins_even_rolls_l287_287985


namespace find_f_prime_at_2_l287_287310

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287310


namespace foot_of_perpendicular_on_line_segment_l287_287261

-- Definitions related to the foot of the perpendicular drawn to a line segment.
def foot_of_perpendicular (P : Point) (A B : LineSegment) : Point :=
sorry -- Assume this function exists and is defined correctly

-- Points defining the line segment.
variables {A B : Point}
variables {P : Point} -- Point where the perpendicular intersects

-- Conditions for the location of the foot of the perpendicular.
theorem foot_of_perpendicular_on_line_segment (A B P : Point) :
  (P lies_on_line_segment_AB A B) ∨ 
  (P = A) ∨ 
  (P = B) ∨ 
  (P lies_on_extension_of_line_segment_AB A B) :=
sorry

end foot_of_perpendicular_on_line_segment_l287_287261


namespace inverse_proportion_relationship_l287_287603

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_relationship (h1 : x1 < 0) (h2 : 0 < x2) 
  (hy1 : y1 = 3 / x1) (hy2 : y2 = 3 / x2) : y1 < 0 ∧ 0 < y2 :=
by
  sorry

end inverse_proportion_relationship_l287_287603


namespace complex_num_count_l287_287990

noncomputable def num_solutions : ℕ :=
  let count := 5040 in
  count

theorem complex_num_count (z : ℂ) (hz : |z| = 1) (hz_real : z ^ 5040 - z ^ 720 ∈ ℝ) : 
  num_solutions = 5040 :=
sorry

end complex_num_count_l287_287990


namespace value_of_a_solution_set_f_x_gt_x_l287_287580

noncomputable def f (a x : ℝ) := 2 * |x - a| - x + a

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : 
  (1 / 2) * a * (| -(4 / 3) * a |) = (8 / 3)) : 
  a = 2 := 
sorry

theorem solution_set_f_x_gt_x (a x : ℝ) (h₁ : a > 0) : 
  f a x > x ↔ x < (3 * a / 4) :=
sorry

end value_of_a_solution_set_f_x_gt_x_l287_287580


namespace ratio_y_to_x_l287_287608

theorem ratio_y_to_x (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : y / x = 13 / 2 :=
by
  sorry

end ratio_y_to_x_l287_287608


namespace rachel_speed_painting_video_time_l287_287753

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end rachel_speed_painting_video_time_l287_287753


namespace smallest_palindrome_produces_non_palindromic_111_product_l287_287556

def is_palindrome (n : ℕ) : Prop := n.toString.reverse = n.toString

def smallest_non_palindromic_product : ℕ :=
  if 3 <= (nat.log10 171 * 111).ceil then
    171
  else
    sorry

theorem smallest_palindrome_produces_non_palindromic_111_product :
  smallest_non_palindromic_product = 171 :=
begin
  sorry
end

end smallest_palindrome_produces_non_palindromic_111_product_l287_287556


namespace last_three_digits_l287_287989

theorem last_three_digits (n : ℕ) : 7^106 % 1000 = 321 :=
by
  sorry

end last_three_digits_l287_287989


namespace equal_segments_l287_287119

theorem equal_segments 
  (O1 O2 C A B D T : Point)
  (circle_O1 : Circle O1)
  (circle_O2 : Circle O2)
  (tangent_ext : IsTangentExternally circle_O1 circle_O2 C)
  (intersects_O1 : IntersectsLineAtTwoPoints line l circle_O1 A B)
  (tangent_O2 : IsTangentLine circle_O2 l D)
  (second_intersection : SecondIntersectionPointOfLineCircle line CD circle_O1 C T) :
  dist A T = dist T B := 
sorry

end equal_segments_l287_287119


namespace find_m_value_l287_287973

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
    (hf : ∀ x, f x = 3 * x ^ 2 - 1 / x + 4)
    (hg : ∀ x, g x = x ^ 2 - m)
    (hfg : f 3 - g 3 = 5) :
    m = -50 / 3 :=
  sorry

end find_m_value_l287_287973


namespace find_f_prime_at_2_l287_287312

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l287_287312


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287536

-- Definitions based on the conditions
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ isPalindrome n

def isNotFiveDigitPalindrome (n : ℕ) : Prop :=
  let prod := n * 111
  prod.toString.length = 5 ∧ ¬isPalindrome prod

-- Lean 4 statement for the proof problem
theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  ∃ n, isThreeDigitPalindrome n ∧ isNotFiveDigitPalindrome n ∧ n = 505 := by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287536


namespace Dan_reaches_Cate_in_25_seconds_l287_287500

theorem Dan_reaches_Cate_in_25_seconds
  (d : ℝ) (v_d : ℝ) (v_c : ℝ)
  (h1 : d = 50)
  (h2 : v_d = 8)
  (h3 : v_c = 6) :
  (d / (v_d - v_c) = 25) :=
by
  sorry

end Dan_reaches_Cate_in_25_seconds_l287_287500


namespace general_term_l287_287202

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x + 2)

def seq (x1 : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x1
  else f (seq x1 (n - 1))

theorem general_term (x1 : ℝ) (h1 : x1 ≠ 1) : ∀ n : ℕ, seq x1 n = 2 / (n + 1) :=
by
  intros
  induction n with
  | zero => 
    simp [seq]
    sorry -- Base case proof
  | succ n ih => 
    simp [seq]
    rw ih
    sorry -- Inductive step proof

end general_term_l287_287202


namespace tangent_point_value_l287_287879

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287879


namespace valid_x_for_sqrt_l287_287652

theorem valid_x_for_sqrt (x : ℝ) (hx : x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) : x ≥ 2 ↔ x = 3 := 
sorry

end valid_x_for_sqrt_l287_287652


namespace smallest_integer_solution_l287_287071

theorem smallest_integer_solution (x : ℤ) (h : 2 * (x : ℝ)^2 + 2 * |(x : ℝ)| + 7 < 25) : x = -2 :=
by
  sorry

end smallest_integer_solution_l287_287071


namespace small_palindrome_check_l287_287567

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287567


namespace trapezoid_area_l287_287455

theorem trapezoid_area {a b c d e : ℝ} (h1 : a = 40) (h2 : b = 40) (h3 : c = 50) (h4 : d = 50) (h5 : e = 60) : 
  (a + b = 80) → (c * c = 2500) → 
  (50^2 - 30^2 = 1600) → ((50^2 - 30^2).sqrt = 40) → 
  (((e - 2 * ((a ^ 2 - (30) ^ 2).sqrt)) * 40) / 2 = 1336) :=
sorry

end trapezoid_area_l287_287455


namespace tangent_line_intersection_l287_287884

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l287_287884


namespace exists_club_with_at_least_11_girls_and_11_boys_l287_287173

/-- In a school with 2007 girls and 2007 boys, 
each student belongs to at most 100 clubs, 
and every pair consisting of a boy and a girl 
belongs to exactly one club. 
Show that there is a club having at least 11 girls and 11 boys. -/
theorem exists_club_with_at_least_11_girls_and_11_boys :
  ∃ (clubs : Type) (members : clubs → set (fin 2007 × bool)),
    (∀ student, fintype.card (image (λ c, c ∈ clubs) {c | (student, c) ∈ members c}) ≤ 100) ∧
    (∀ (g : fin 2007) (b : fin 2007),
       ∃! (c : clubs), ((g, true), c) ∈ members c ∧ ((b, false), c) ∈ members c) ->
  ∃ (c : clubs),
    11 ≤ fintype.card ({s | s ∈ members c ∧ s.2 = true}) ∧
    11 ≤ fintype.card ({s | s ∈ members c ∧ s.2 = false}) :=
sorry

end exists_club_with_at_least_11_girls_and_11_boys_l287_287173


namespace initial_term_exists_l287_287776

def hailstone_seq (a : ℕ) : ℕ → ℕ
| 0       => a
| (n + 1) => if (((hailstone_seq n) % 2) = 0) then (hailstone_seq n) / 2 else 3 * (hailstone_seq n) + 1

theorem initial_term_exists :
  ∃ a1 : ℕ, a1 < 50 ∧ ∀ n : ℕ, n < 10 → (hailstone_seq a1 n ≠ 1) ∧ (hailstone_seq a1 9 = 1) :=
sorry

end initial_term_exists_l287_287776


namespace area_triangle_ABF_l287_287622

/-- Define the hyperbola E by the equation x^2 - y^2/3 = 1 and its left focus F, and the line x = 2. 
  Prove that the area of the triangle ABF, where A and B are the points of intersection of the line 
  with the hyperbola, is 12. -/
theorem area_triangle_ABF :
  let F : ℝ×ℝ := (-2, 0)
  let A : ℝ×ℝ := (2, 3)
  let B : ℝ×ℝ := (2, -3)
  let area : ℝ := abs (1/2 * ((A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2) + F.1 * (A.2 - B.2))))
  in area = 12 := 
by
  sorry

end area_triangle_ABF_l287_287622


namespace robert_salary_loss_l287_287756

theorem robert_salary_loss (S : ℝ) : 
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  100 * (1 - increased_salary / S) = 9 :=
by
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  sorry

end robert_salary_loss_l287_287756


namespace trapezoid_not_necessarily_parallelogram_l287_287692

-- Definitions of a trapezoid and the condition of the midline EF segment
structure Trapezoid (A B C D E F : Type) :=
(base1 : A)
(base2 : B)
(side1 : C)
(side2 : D)
(is_parallel_to_bases : E)
(perim_half : F)
(area_half : F)

-- Statement of the problem in Lean 4
theorem trapezoid_not_necessarily_parallelogram
  {A B C D E F : Type}
  [Trapezoid A B C D E F]
  (trapezoid : Trapezoid A B C D E F)
  (parallel_line_ef_half_perim : trapezoid.perim_half)
  (parallel_line_ef_half_area : trapezoid.area_half) :
  ∃ b d a c : Type, (b ≠ d ∧ b ≠ c ∧ a ≠ d) :=
sorry

end trapezoid_not_necessarily_parallelogram_l287_287692


namespace small_palindrome_check_l287_287562

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287562


namespace find_interest_rate_l287_287525

-- Definitions for the given conditions
def principal : ℝ := 1300
def time_period : ℝ := 2.4
def amount : ℝ := 1456

-- Define the simple interest calculation
def interest (P A : ℝ) : ℝ := A - P

-- Define the rate calculation based on the simple interest formula
def rate (I P T : ℝ) : ℝ := (I * 100) / (P * T)

-- Interest and Rate using the conditions
def interest_earned : ℝ := interest principal amount
def interest_rate : ℝ := rate interest_earned principal time_period

-- The statement that needs to be proved
theorem find_interest_rate : interest_rate = 5 := by
  sorry

end find_interest_rate_l287_287525


namespace max_n_leq_V_l287_287160

theorem max_n_leq_V (n : ℤ) (V : ℤ) (h1 : 102 * n^2 <= V) (h2 : ∀ k : ℤ, (102 * k^2 <= V) → k <= 8) : V >= 6528 :=
sorry

end max_n_leq_V_l287_287160


namespace solve_equation_l287_287992

noncomputable def equation (x : ℝ) : ℝ :=
(13 * x - x^2) / (x + 1) * (x + (13 - x) / (x + 1))

theorem solve_equation :
  equation 1 = 42 ∧ equation 6 = 42 ∧ equation (3 + Real.sqrt 2) = 42 ∧ equation (3 - Real.sqrt 2) = 42 :=
by
  sorry

end solve_equation_l287_287992


namespace sum_even_range_l287_287850

-- Define the sum of the first 50 positive even integers as a given fact
def sum_first_50_even : ℕ := 2550

-- Define the sum of even integers from 102 to 200 inclusive
def sum_even_102_to_200 : ℕ := 7550

-- Proof statement that sum_even_102_to_200 is 7550 given sum_first_50_even
theorem sum_even_range :
  (∑ i in (Finset.range 50).map (λ x => 102 + 2 * x), i) = sum_even_102_to_200 := by
  sorry

end sum_even_range_l287_287850


namespace sum_factorial_formula_l287_287993

def S (n : ℕ) : ℕ := (finset.range (n+1)).sum (λ i, (i + 1) * (i + 1).fact)

theorem sum_factorial_formula (n : ℕ) (h : n ≥ 1) : 
  S n = (n + 1).fact - 1 :=
sorry

end sum_factorial_formula_l287_287993


namespace find_f_prime_at_2_l287_287344

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287344


namespace midpoint_count_bounds_l287_287101

theorem midpoint_count_bounds (n : ℕ) (h : n ≥ 2) :
  2 * n - 3 ≤ ∑ i in (Finset.range (n)).powerset.filter (λ s, s.card = 2), (by simp : ℤ) ≤ (n * (n - 1)) / 2 :=
sorry

end midpoint_count_bounds_l287_287101


namespace lines_parallel_to_same_plane_l287_287292

-- Define the concept of a line being parallel to a plane in a 3D space
structure Line3D (α : Type) := (x1 y1 z1 x2 y2 z2 : α) -- arbitrary structure for line
structure Plane3D (α : Type) := (a b c d : α) -- arbitrary structure for plane (ax + by + cz + d = 0)

-- Define when a line is parallel to a plane
def is_line_parallel_to_plane {α : Type} [field α] (L : Line3D α) (P : Plane3D α) : Prop :=
(L.x2 - L.x1) * P.a + (L.y2 - L.y1) * P.b + (L.z2 - L.z1) * P.c = 0

-- Define the positional relationships between two lines
inductive PositionalRelationship
| Parallel
| Intersecting
| Skew

-- The main theorem
theorem lines_parallel_to_same_plane {α : Type} [field α] 
  (L1 L2 : Line3D α) (P : Plane3D α)
  (h1 : is_line_parallel_to_plane L1 P) (h2 : is_line_parallel_to_plane L2 P) :
  ∃ r : PositionalRelationship, r = PositionalRelationship.Parallel ∨ 
                               r = PositionalRelationship.Intersecting ∨ 
                               r = PositionalRelationship.Skew :=
begin
  sorry
end

end lines_parallel_to_same_plane_l287_287292


namespace erased_number_l287_287476

theorem erased_number (a : ℤ) (b : ℤ) (h : -4 ≤ b ∧ b ≤ 4) (h_sum : 8 * a - b = 1703) : a + b = 214 := 
by 
  have ha : 1699 / 8 ≤ a, from sorry
  have hb : a ≤ 1707 / 8, from sorry
  have ha_int : a = 213, from sorry
  have hb_int : b = 1, by { rw [ha_int] at h_sum, linarith }
  exact sorry 

-- The proof steps are not provided here, as only the theorem statement is requested.

end erased_number_l287_287476


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287538

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287538


namespace avg_goals_l287_287157

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l287_287157


namespace train_length_at_constant_acceleration_l287_287436

variables (u : ℝ) (t : ℝ) (a : ℝ) (s : ℝ)

theorem train_length_at_constant_acceleration (h₁ : u = 16.67) (h₂ : t = 30) : 
  s = u * t + 0.5 * a * t^2 :=
sorry

end train_length_at_constant_acceleration_l287_287436


namespace range_of_a_l287_287134

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| - |x - 3| ≤ a) → a ≥ -5 := by
  sorry

end range_of_a_l287_287134


namespace quincy_sold_more_than_jake_l287_287237

theorem quincy_sold_more_than_jake :
  ∀ (T Jake : ℕ), Jake = 2 * T + 15 → 4000 = 100 * (T + Jake) → 4000 - Jake = 3969 :=
by
  intros T Jake hJake hQuincy
  sorry

end quincy_sold_more_than_jake_l287_287237


namespace smallest_non_palindrome_product_l287_287545

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287545


namespace sum_of_integers_k_l287_287077

theorem sum_of_integers_k :
  (∑ k in {k | nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k}, k) = 26 := by
  sorry

end sum_of_integers_k_l287_287077


namespace trapezoid_area_l287_287447

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l287_287447


namespace relationship_a_b_l287_287123

theorem relationship_a_b
  (m a b : ℝ)
  (h1 : ∃ m, ∀ x, -2 * x + m = y)
  (h2 : ∃ x₁ y₁, (x₁ = -2) ∧ (y₁ = a) ∧ (-2 * x₁ + m = y₁))
  (h3 : ∃ x₂ y₂, (x₂ = 2) ∧ (y₂ = b) ∧ (-2 * x₂ + m = y₂)) :
  a > b :=
sorry

end relationship_a_b_l287_287123


namespace figure_100_squares_l287_287061

def figure_squares (n : ℕ) : ℕ :=
  2 * n ^ 2 + 2 * n + 1

theorem figure_100_squares : figure_squares 100 = 20201 := 
  by simp [figure_squares] -- Simplify the expression
    sorry -- Proof omitted

end figure_100_squares_l287_287061


namespace tangent_point_value_l287_287881

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287881


namespace alex_journey_time_l287_287438

-- Definitions for the conditions
variables (v : ℝ) (t_mountain_pass t_highway t_total : ℝ)

-- Conditions in Lean
def travel_conditions :=
  (v > 0) ∧
  (t_mountain_pass = 40) ∧
  (20 / v = t_mountain_pass) ∧
  (t_highway = 60 / (4 * v))

-- The theorem statement
theorem alex_journey_time 
  (h : travel_conditions v t_mountain_pass t_highway) :
  t_total = t_mountain_pass + t_highway :=
begin
  sorry,
end

end alex_journey_time_l287_287438


namespace insects_total_l287_287281

def total_insects (n_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
                  (n_stones : ℕ) (ants_per_stone : ℕ) 
                  (total_bees : ℕ) (n_flowers : ℕ) : ℕ :=
  let num_ladybugs := n_leaves * ladybugs_per_leaf
  let num_ants := n_stones * ants_per_stone
  let num_bees := total_bees -- already given as total_bees
  num_ladybugs + num_ants + num_bees

theorem insects_total : total_insects 345 267 178 423 498 6 = 167967 :=
  by unfold total_insects; sorry

end insects_total_l287_287281


namespace ratio_of_shaded_to_empty_l287_287461

-- Definitions based on conditions
def vertices_of_regular_hexagon : Prop := sorry
def hexagram (hexagon : Prop) : Prop := 
    hexagon ∧ (formed_by_two_equilateral_triangles hexagon)

def formed_by_two_equilateral_triangles (hexagon : Prop) : Prop := sorry

def shaded_area_to_empty_space_ratio (shaded empty : Nat) : Nat :=
  shaded / empty

-- Given conditions as premises
axiom hexagon_vertices : vertices_of_regular_hexagon
axiom hexagram_form : hexagram hexagon_vertices
axiom shaded_triangles : Nat := 18
axiom empty_triangles : Nat := 6

-- Main theorem statement
theorem ratio_of_shaded_to_empty : shaded_area_to_empty_space_ratio shaded_triangles empty_triangles = 3 :=
by 
  sorry

end ratio_of_shaded_to_empty_l287_287461


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287908

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287908


namespace find_f_prime_at_two_l287_287328

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287328


namespace directrix_of_parabola_l287_287258

theorem directrix_of_parabola (x y : ℝ) : (x ^ 2 = y) → (4 * y + 1 = 0) :=
sorry

end directrix_of_parabola_l287_287258


namespace perpendicular_MN_A1B_D1B1_l287_287664

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨1, 0, 0⟩
def B : Point3D := ⟨1, 1, 0⟩
def A1 : Point3D := ⟨1, 0, 1⟩
def B1 : Point3D := ⟨1, 1, 1⟩
def D : Point3D := ⟨0, 0, 0⟩
def D1 : Point3D := ⟨0, 0, 1⟩

def M : Point3D := ⟨1, 1/3, 2/3⟩
def N : Point3D := ⟨2/3, 2/3, 1⟩

def vector (P Q : Point3D) : Point3D :=
⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def dot_product (P Q : Point3D) : ℝ :=
P.x * Q.x + P.y * Q.y + P.z * Q.z

def A1B := vector A1 B
def D1B1 := vector D1 B1
def MN := vector M N

theorem perpendicular_MN_A1B_D1B1 :
  dot_product MN A1B = 0 ∧ dot_product MN D1B1 = 0 :=
sorry

end perpendicular_MN_A1B_D1B1_l287_287664


namespace sum_of_interior_angles_at_R_l287_287676

-- Define the properties of regular polygons and the sum of their angles
def regular_polygon_interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

-- Define specific polygons
def triangle_interior_angle := regular_polygon_interior_angle 3
def rectangle_interior_angle := 90 -- since all angles in a rectangle are 90 degrees
def pentagon_interior_angle := regular_polygon_interior_angle 5

-- Define the sum of angles at R
def sum_of_angles_at_R := triangle_interior_angle + rectangle_interior_angle + pentagon_interior_angle

-- Prove the sum of the angles at R equals 258 degrees
theorem sum_of_interior_angles_at_R : sum_of_angles_at_R = 258 := by
  sorry

end sum_of_interior_angles_at_R_l287_287676


namespace erased_number_is_214_l287_287479

def middle_value (a : ℤ) : list ℤ := [a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4]

theorem erased_number_is_214 (a b : ℤ) (h_sum : 8 * a - b = 1703)
  (h_bounds : -4 ≤ b ∧ b ≤ 4) (h_a : a = 213) (h_b : b = 8 * 213 - 1703) : 
  (middle_value a).sum - (a + b) = 1703 ∧ a + b = 214 :=
by { sorry }

end erased_number_is_214_l287_287479


namespace find_all_functions_form_l287_287522

noncomputable def math_problem (f : ℝ → ℝ) : Prop :=
∀ (a b c d : ℝ), a > b ∧ b > c ∧ c > d ∧ d > 0 ∧ a * d = b * c →
f(a + d) + f(b - c) = f(a - d) + f(b + c)

theorem find_all_functions_form (f : ℝ → ℝ)
  (h : math_problem f) :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 
  ∀ (u : ℝ), u > 0 → f(u) = c * u^2 + d :=
sorry

end find_all_functions_form_l287_287522


namespace task1_on_time_task2_not_on_time_l287_287852

/-- Define the probabilities for task 1 and task 2 -/
def P_A : ℚ := 3 / 8
def P_B : ℚ := 3 / 5

/-- The probability that task 1 will be completed on time but task 2 will not is 3 / 20. -/
theorem task1_on_time_task2_not_on_time (P_A : ℚ) (P_B : ℚ) : P_A = 3 / 8 → P_B = 3 / 5 → P_A * (1 - P_B) = 3 / 20 :=
by
  intros hPA hPB
  rw [hPA, hPB]
  norm_num

end task1_on_time_task2_not_on_time_l287_287852


namespace expand_polynomial_product_l287_287513

theorem expand_polynomial_product :
  3 * (λ x : ℝ, (x^2 - 5*x + 6) * (x^2 + 8*x - 10)) = (λ x : ℝ, 3*x^4 + 9*x^3 - 132*x^2 + 294*x - 180) :=
by sorry

end expand_polynomial_product_l287_287513


namespace max_value_at_one_l287_287318

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287318


namespace raffle_distribution_l287_287818

noncomputable def distribution_law : List (ℕ × ℚ) :=
  [(0, 0.94), (5, 0.04), (30, 0.02)]

theorem raffle_distribution :
  ∃ (X : List (ℕ × ℚ)),
    (X = [(0, 0.94), (5, 0.04), (30, 0.02)]) ∧
    ∑ x in X, x.2 = 1 ∧
    ∃ (n : ℕ), n = 50 ∧
    ∃ (a b c : ℕ), (a = 2 ∧ b = 1 ∧ c = 47) ∧
    (X = [(0, c/50), (5, a/50), (30, b/50)]) :=
begin
  sorry
end

end raffle_distribution_l287_287818


namespace original_number_is_400_l287_287004

theorem original_number_is_400 (x y N : ℝ) :
  1.20 * N = 480 →
  (480 - 0.15 * 480) * x^2 = 5 * x^3 + 24 * x - 50 →
  (N / y) * 0.75 = x * y →
  N = 400 :=
by {
  intros h1 h2 h3,
  sorry
}

end original_number_is_400_l287_287004


namespace symmetric_point_correct_l287_287072

noncomputable def symmetric_point (M : ℝ × ℝ × ℝ) (a b c d : ℝ) : ℝ × ℝ × ℝ :=
  let normal_vector := (a, b, c)
  let t := -(a * M.1 + b * M.2 + c * M.3 + d) / (a^2 + b^2 + c^2)
  let M₀ := (M.1 + a * t, M.2 + b * t, M.3 + c * t)
  (2 * fst M₀ - M.1, 2 * snd M₀ - M.2, 2 * M₀.2 - M.3)

theorem symmetric_point_correct : symmetric_point (1, 1, 1) 1 4 3 5 = (0, -3, -2) := by
  sorry

end symmetric_point_correct_l287_287072


namespace derivative_at_2_l287_287376

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287376


namespace continuity_necessity_not_sufficiency_l287_287405

theorem continuity_necessity_not_sufficiency (f : ℝ → ℝ) (x₀ : ℝ) :
  ((∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) → f x₀ = f x₀) ∧ ¬ ((f x₀ = f x₀) → (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε)) := 
sorry

end continuity_necessity_not_sufficiency_l287_287405


namespace hcl_reaction_l287_287631

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction_l287_287631


namespace sum_of_coeffs_l287_287960

theorem sum_of_coeffs (x y : ℤ) : (x - 3 * y) ^ 20 = 2 ^ 20 := by
  sorry

end sum_of_coeffs_l287_287960


namespace proportion_proof_l287_287648

-- Define the problem conditions and the target theorem statement
def proportion := ∀ (n : ℝ), (x : ℝ), x = 1.2 → (n / x = 5 / 8) → n = 0.75

-- Theorem statement for the problem
theorem proportion_proof : proportion := by
  intro n x h_x h_proportion
  sorry

end proportion_proof_l287_287648


namespace num_club_member_allocations_l287_287015

-- Defining the problem conditions
def num_students : Nat := 6
def num_clubs : Nat := 5
def max_per_club : Nat := 2
def max_speech_team : Nat := 1

-- The main theorem to prove the number of ways to allocate students to clubs
theorem num_club_member_allocations :
  ∃ (count : Nat), 
    count = 5040 ∧
    ∀ (choices : Finset (Fin num_students)) 
      (allocation : choices → Fin num_clubs), 
      let allocation_counts := finmap.counts choices allocation in
      (allocation_counts (Fin.mk 0 sorry) ≤ 2) ∧
      (allocation_counts (Fin.mk 1 sorry) ≤ 2) ∧ 
      (allocation_counts (Fin.mk 2 sorry) ≤ 2) ∧ 
      (allocation_counts (Fin.mk 3 sorry) ≤ 1) ∧ 
      (allocation_counts (Fin.mk 4 sorry) ≤ 2) in 
  count = 5040 := 
sorry

end num_club_member_allocations_l287_287015


namespace seating_arrangements_l287_287810

theorem seating_arrangements :
  let total_seats := 8
  let people := 3
  let empty_surrounding (arr : List ℕ) := ∀ p ∈ arr, (p > 1) ∧ (p < total_seats - 1) ∧ (¬ (p + 1 ∈ arr)) ∧ (¬ (p - 1 ∈ arr))
  ∃ arr : List ℕ, (arr.length = people) ∧ empty_surrounding arr ∧ (arr.countp (λ p, true) = 24) := by
  sorry

end seating_arrangements_l287_287810


namespace kaashish_problem_l287_287701

theorem kaashish_problem (x y : ℤ) (h : 2 * x + 3 * y = 100) (k : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 :=
by
  sorry

end kaashish_problem_l287_287701


namespace match_graph_l287_287381

theorem match_graph (x : ℝ) (h : x ≤ 0) : 
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by
  sorry

end match_graph_l287_287381


namespace not_perfect_square_l287_287402

open Nat

-- Define the necessary conditions for m, n being positive integers such that m - n is odd
def conditions (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ odd (m - n)

-- The main theorem stating the problem
theorem not_perfect_square (m n : ℕ) (h : conditions m n) : ¬ ∃ k : ℕ, k * k = (m + 3 * n) * (5 * m + 7 * n) :=
sorry

end not_perfect_square_l287_287402


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287904

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287904


namespace b_120_is_1_l287_287419

noncomputable def b : ℕ → ℚ
| 1       := 1
| 2       := -1
| (n+1+1) := (1 - b (n+1) - 3 * b n) / (2 * b n)

theorem b_120_is_1 : b 120 = 1 := 
sorry

end b_120_is_1_l287_287419


namespace combined_std_dev_l287_287054

noncomputable def group1 : List ℝ := sorry -- Assume some list of real numbers representing the first group
noncomputable def group2 : List ℝ := sorry -- Assume some list of real numbers representing the second group
noncomputable def combined_group : List ℝ := group1 ++ group2

def average (l: List ℝ) : ℝ := (l.sum) / (l.length)
def variance (l: List ℝ) : ℝ := (l.map (λ x => x^2)).sum / (l.length) - (average l)^2
def std_dev (l: List ℝ) : ℝ := Real.sqrt (variance l)

axiom group1_average : average group1 = 50
axiom group1_variance : variance group1 = 33
axiom group2_average : average group2 = 40
axiom group2_variance : variance group2 = 45

-- The theorem we need to prove
theorem combined_std_dev :
  std_dev combined_group = 8 :=
sorry

end combined_std_dev_l287_287054


namespace download_rate_first_60_l287_287786

theorem download_rate_first_60 (file_size last_download size last_rate total_time : ℝ)
  (h_file_size : file_size = 90) 
  (h_last_download : last_download = 30) 
  (h_last_rate : last_rate = 10) 
  (h_total_time : total_time = 15) 
  (h_last_time : last_download / last_rate = 3) 
  (h_first_time : total_time - last_download / last_rate = 12) 
  (h_first_download : size = 60) : 
  size / h_first_time = 5 := 
sorry

end download_rate_first_60_l287_287786


namespace result_after_subtraction_l287_287434

theorem result_after_subtraction (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 :=
by
  sorry

end result_after_subtraction_l287_287434


namespace distance_between_parallel_lines_l287_287259

theorem distance_between_parallel_lines (a d : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 3 = 0 ∧ a * x - y + 4 = 0 → (2 = a ∧ d = |(3 - 4)| / Real.sqrt (2 ^ 2 + (-1) ^ 2))) → 
  (a = 2 ∧ d = Real.sqrt 5 / 5) :=
by 
  sorry

end distance_between_parallel_lines_l287_287259


namespace find_angle_AOB_l287_287610

noncomputable def pointA := (2 : ℝ, 1 : ℝ)
noncomputable def pointB := (3 / 10 : ℝ, -1 / 10 : ℝ)

theorem find_angle_AOB : 
  let A := pointA in
  let B := pointB in
  angle (0 : ℂ) (A.1 + A.2 * I) (B.1 + B.2 * I) = π / 4 :=
sorry

end find_angle_AOB_l287_287610


namespace ratio_of_shaded_to_unshaded_l287_287459

-- Definitions based on conditions
def is_regular_hexagon (hex : Hexagram) : Prop :=
  -- Hexagram vertices are the same as that of a regular hexagon
  hex.is_regular_hexagon

def forms_hexagram (hex : Hexagram) : Prop :=
  -- Hexagram is formed by overlapping two equilateral triangles
  hex.forms_hexagram_by_overlapping_equilateral_triangles

def shaded_region_triangles (hex : Hexagram) : ℕ :=
  -- Shaded region forms 18 smaller triangles
  hex.shaded_region_triangles

def unshaded_region_triangles (hex : Hexagram) : ℕ :=
  -- Unshaded region forms 6 smaller triangles
  hex.unshaded_region_triangles

-- Main statement
theorem ratio_of_shaded_to_unshaded (hex : Hexagram)
  (h1 : is_regular_hexagon hex) 
  (h2 : forms_hexagram hex)
  (h3 : shaded_region_triangles hex = 18) 
  (h4 : unshaded_region_triangles hex = 6) :
  shaded_region_triangles hex / unshaded_region_triangles hex = 3 :=
by
  -- Proof will go here
  sorry

end ratio_of_shaded_to_unshaded_l287_287459


namespace bela_wins_optimal_strategy_l287_287027

theorem bela_wins_optimal_strategy : 
  ∀ (interval : set ℝ) (x : ℝ), interval = Icc 0 10 → 0 ≤ x ∧ x ≤ 10 →
  (∀ (y : ℝ), y ∈ interval → |y - x| ≥ 1.5) → 
  ∃ (will_win : ℝ → Prop), will_win x → (∀ y ∈ interval, will_win y → will_win x) :=
by
  sorry

end bela_wins_optimal_strategy_l287_287027


namespace proof_of_ratio_l287_287711

def f (x : ℤ) : ℤ := 3 * x + 4

def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_of_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 :=
by
  sorry

end proof_of_ratio_l287_287711


namespace double_theta_acute_l287_287113

theorem double_theta_acute (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_theta_acute_l287_287113


namespace bond_yield_correct_l287_287228

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end bond_yield_correct_l287_287228


namespace find_ending_number_l287_287523

theorem find_ending_number (n : ℕ) 
  (h1 : n ≥ 7) 
  (h2 : ∀ m, 7 ≤ m ∧ m ≤ n → m % 7 = 0)
  (h3 : (7 + n) / 2 = 15) : n = 21 := 
sorry

end find_ending_number_l287_287523


namespace find_f_prime_at_two_l287_287332

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287332


namespace sequence_general_formula_triangle_side_b_triangle_angle_C_max_n_geometric_sequence_l287_287859

-- Problem 1
theorem sequence_general_formula (S_n : ℕ → ℚ) (a_n : ℕ → ℚ)
  (hS : ∀ n, S_n n = (2/3) * a_n n + (1/3)) :
  ∀ n, a_n n = (-2)^(n-1) :=
sorry

-- Problem 2
theorem triangle_side_b (a c : ℝ) (B : ℝ) (b : ℝ)
  (h1 : a = 2) (h2 : B = π/6) (h3 : c = 2 * sqrt 3)
  (h4 : b^2 = a^2 + c^2 - 2 * a * c * cos B) :
  b = 2 :=
sorry

-- Problem 3
theorem triangle_angle_C (a b c A B : ℝ) (C : ℝ)
  (h1 : b + c = 2 * a) (h2 : 3 * sin A = 5 * sin B)
  (h3 : cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  C = 2 * π / 3 :=
sorry

-- Problem 4
theorem max_n_geometric_sequence (a_n : ℕ → ℝ)
  (h1 : a_n 5 = 1/2) (h2 : a_n 6 + a_n 7 = 3)
  (h3 : ∀ n, a_n n = 2^(n-6))
  (h4 : ∑ i in finset.range (n+1), a_n i > (finset.range (n+1)).prod a_n) :
  n ≤ 12 :=
sorry

end sequence_general_formula_triangle_side_b_triangle_angle_C_max_n_geometric_sequence_l287_287859


namespace sufficient_x_ge_1_and_y_ge_2_not_necessary_x_ge_1_and_y_ge_2_l287_287094

variable {x y : ℝ}

theorem sufficient_x_ge_1_and_y_ge_2 (h1 : x ≥ 1) (h2 : y ≥ 2) : x + y ≥ 3 := 
by
  exact add_le_add h1 h2

theorem not_necessary_x_ge_1_and_y_ge_2 (h : x + y ≥ 3) : ¬ (x ≥ 1 ∧ y ≥ 2) :=
by
  intro hxy
  have h1 : x < 1 := by linarith
  have h2 : y < 2 := by linarith
  sorry -- complete with an explicit counterexample

end sufficient_x_ge_1_and_y_ge_2_not_necessary_x_ge_1_and_y_ge_2_l287_287094


namespace cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l287_287440

def cube (n : ℕ) : Type := ℕ × ℕ × ℕ

-- Define a 4x4x4 cube and the painting conditions
def four_by_four_cube := cube 4

-- Determine the number of small cubes with exactly one face painted
theorem cubes_with_one_face_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Determine the number of small cubes with exactly two faces painted
theorem cubes_with_two_faces_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Given condition and find the size of the new cube
theorem size_of_new_cube (n : ℕ) : 
  (n - 2) ^ 3 = 3 * 12 * (n - 2) → n = 8 :=
by
  -- proof goes here
  sorry

end cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l287_287440


namespace derivative_at_2_l287_287363

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287363


namespace total_cost_of_purchase_l287_287757

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l287_287757


namespace every_positive_integer_expressed_l287_287708

-- Definitions of the sequence elements
def is_valid_term (a : ℕ) : Prop := ∃ α β : ℕ, a = 2^α * 3^β

-- Main theorem statement
theorem every_positive_integer_expressed :
  ∀ n : ℕ, 0 < n → ∃ (i : ℕ → ℕ) (k : ℕ), (∀ j < k, is_valid_term (i j)) ∧
  (∀ j < k, ∀ l < k, j ≠ l → ¬ (i j ∣ i l)) ∧ (n = ∑ j in finset.range k, i j) :=
by
  sorry

end every_positive_integer_expressed_l287_287708


namespace tangent_line_intersection_x_axis_l287_287897

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287897


namespace average_goals_is_92_l287_287155

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l287_287155


namespace sin_diff_l287_287115

variable (θ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5)

theorem sin_diff
  (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5) :
  Real.sin (θ - π / 4) = Real.sqrt 10 / 10 :=
sorry

end sin_diff_l287_287115


namespace walkway_area_correct_l287_287659

-- Define the dimensions and conditions
def bed_width : ℝ := 4
def bed_height : ℝ := 3
def walkway_width : ℝ := 2
def num_rows : ℕ := 4
def num_columns : ℕ := 3
def num_beds : ℕ := num_rows * num_columns

-- Total dimensions of garden including walkways
def total_width : ℝ := (num_columns * bed_width) + ((num_columns + 1) * walkway_width)
def total_height : ℝ := (num_rows * bed_height) + ((num_rows + 1) * walkway_width)

-- Areas
def total_garden_area : ℝ := total_width * total_height
def total_bed_area : ℝ := (bed_width * bed_height) * num_beds

-- Correct answer we want to prove
def walkway_area : ℝ := total_garden_area - total_bed_area

theorem walkway_area_correct : walkway_area = 296 := by
  sorry

end walkway_area_correct_l287_287659


namespace identity_product_eq_neg_288_l287_287198

theorem identity_product_eq_neg_288 (N₁ N₂ : ℤ) :
    (∀ x ≠ 1, x ≠ 2, 42*x - 36 = N₁*(x - 2) + N₂*(x - 1)) → N₁*N₂ = -288 :=
by
  sorry -- proof to be written

end identity_product_eq_neg_288_l287_287198


namespace max_value_at_one_l287_287323

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287323


namespace problem_solution_l287_287204

open Real

def system_satisfied (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    (a (2 * k + 1) = (1 / a (2 * (k + n) - 1) + 1 / a (2 * k + 2))) ∧ 
    (a (2 * k + 2) = a (2 * k + 1) + a (2 * k + 3))

theorem problem_solution (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ k, 0 ≤ k → k < 2 * n → a k > 0)
  (h3 : system_satisfied a n) :
  ∀ k, 0 ≤ k ∧ k < n → a (2 * k + 1) = 1 ∧ a (2 * k + 2) = 2 :=
sorry

end problem_solution_l287_287204


namespace number_of_paths_avoiding_point_C_l287_287660

def binom : ℕ → ℕ → ℕ
| n, k := if h : k ≤ n then nat.choose n k else 0

def paths (n m : ℕ) : ℕ :=
  binom (n + m) m

def avoid_point (n m : ℕ) (p q : ℕ) : ℕ :=
  paths n m - (paths p q * paths (n - p) (m - q))

theorem number_of_paths_avoiding_point_C :
  avoid_point 10 5 5 3 = 1827 :=
by
  unfold avoid_point paths binom
  have h1 : binom 15 5 = 3003 := by sorry
  have h2 : binom 8 3 = 56 := by sorry
  have h3 : binom 7 2 = 21 := by sorry
  have h4 : 56 * 21 = 1176 := by sorry
  rw [h1, h2, h3, h4]
  exact nat.sub_eq_iff_eq_add.mpr rfl

end number_of_paths_avoiding_point_C_l287_287660


namespace derivative_value_at_1_l287_287129

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_value_at_1 :
  (derivative f 1) = 2 + Real.exp 1 := 
sorry

end derivative_value_at_1_l287_287129


namespace find_m_symmetric_points_l287_287102

theorem find_m_symmetric_points (m : ℝ) :
  (C : ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0) →
  (l : ∀ x y : ℝ, x + m * y + 1 = 0) →
  (l_center : ∀ x y : ℝ, l 1 2) →
  m = -1 :=
by
  sorry

end find_m_symmetric_points_l287_287102


namespace max_a_value_l287_287619

def f (a x : ℝ) : ℝ := x^3 - a*x^2 + (a^2 - 2)*x + 1

theorem max_a_value (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ f a m ≤ 0) → a ≤ 1 :=
by
  intro h
  sorry

end max_a_value_l287_287619


namespace red_balls_count_after_game_l287_287165

structure BagState :=
  (red : Nat)         -- Number of red balls
  (green : Nat)       -- Number of green balls
  (blue : Nat)        -- Number of blue balls
  (yellow : Nat)      -- Number of yellow balls
  (black : Nat)       -- Number of black balls
  (white : Nat)       -- Number of white balls)

def initialBallCount (totalBalls : Nat) : BagState :=
  let totalRatio := 15 + 13 + 17 + 9 + 7 + 23
  { red := totalBalls * 15 / totalRatio
  , green := totalBalls * 13 / totalRatio
  , blue := totalBalls * 17 / totalRatio
  , yellow := totalBalls * 9 / totalRatio
  , black := totalBalls * 7 / totalRatio
  , white := totalBalls * 23 / totalRatio
  }

def finalBallCount (initialState : BagState) : BagState :=
  { red := initialState.red + 400
  , green := initialState.green - 250
  , blue := initialState.blue
  , yellow := initialState.yellow - 100
  , black := initialState.black + 200
  , white := initialState.white - 500
  }

theorem red_balls_count_after_game :
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  final.red = 2185 :=
by
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  sorry

end red_balls_count_after_game_l287_287165


namespace find_cylinder_height_l287_287781

def volume_of_cylinder (r h : ℝ) : ℝ := π * (r ^ 2) * h

theorem find_cylinder_height :
  ∀ (d V h : ℝ), d = 4 → V = 20 → volume_of_cylinder (d / 2) h = V → h = 5 / π :=
by
  intro d V h hd hV hVolume
  rw [hd, hV] at hVolume
  sorry

end find_cylinder_height_l287_287781


namespace small_palindrome_check_l287_287565

def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def is_three_digit_palindrome (n : Nat) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (n / 100 = n % 10)

def smallest_non_palindrome_product : Nat :=
  515

theorem small_palindrome_check :
  ∃ n : Nat, is_three_digit_palindrome n ∧ 111 * n < 100000 ∧ ¬ is_palindrome (111 * n) ∧ n = smallest_non_palindrome_product :=
by
  have h₁ : is_three_digit_palindrome 515 := by
  -- Proof that 515 is in the form of 100a + 10b + a with single digits a and b
  sorry

  have h₂ : 111 * 515 < 100000 := by
  -- Proof that the product of 515 and 111 is a five-digit number
  sorry

  have h₃ : ¬ is_palindrome (111 * 515) := by
  -- Proof that 111 * 515 is not a palindrome
  sorry

  have h₄ : 515 = smallest_non_palindrome_product := by
  -- Proof that 515 is indeed the predefined smallest non-palindrome product
  sorry

  exact ⟨515, h₁, h₂, h₃, h₄⟩

end small_palindrome_check_l287_287565


namespace proof_problem_l287_287178

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 3 / 2) * t, (1 / 2) * t)

def circle_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 + 3 * real.cos θ, 3 * real.sin θ)

def F := (1, 0)

noncomputable def general_equation_line :=
  ∀ x y : ℝ, (∃ t : ℝ, x = 1 + (real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t) ↔ x - real.sqrt 3 * y - 1 = 0

noncomputable def general_equation_circle :=
  ∀ x y : ℝ, (∃ θ : ℝ, x = 2 + 3 * real.cos θ ∧ y = 3 * real.sin θ) ↔ (x - 2)^2 + y^2 = 9

noncomputable def length_FA_FB :=
  |((line_parametric t_1).1 - F.1)^2 + ((line_parametric t_1).2 - F.2)^2| 
  + |((line_parametric t_2).1 - F.1)^2 + ((line_parametric t_2).2 - F.2)^2|

theorem proof_problem:
  general_equation_line ∧ general_equation_circle ∧ length_FA_FB = real.sqrt 35 := by
  sorry

end proof_problem_l287_287178


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287554

def is_palindrome (n : ℕ) : Prop := n = (n.to_string.reverse).to_nat

def smallest_non_palindromic_tripled_palindrome : ℕ :=
  if h : ∃ n, (n >= 100) ∧ (n < 1000) ∧ is_palindrome n ∧ ¬ is_palindrome (n * 111) then 
    classical.some h 
  else 
    0

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_non_palindromic_tripled_palindrome = 111 :=
begin
  sorry
end

end smallest_three_digit_palindrome_not_five_digit_palindrome_l287_287554


namespace FemaleEmployeesAtLeastSixty_l287_287166

variable {TotalEmployees : ℕ}
variable {AdvancedDegreeEmployees : ℕ}
variable {CollegeDegreeOnlyEmployees : ℕ}
variable {MaleCollegeOnlyEmployees : ℕ}

-- Define the number of female employees F
def FemaleEmployees :=
  let CollegeDegreeOnlyFemales := CollegeDegreeOnlyEmployees - MaleCollegeOnlyEmployees
  CollegeDegreeOnlyFemales

-- The statement to prove the number of female employees is at least 60
theorem FemaleEmployeesAtLeastSixty
  (h1 : TotalEmployees = 200)
  (h2 : AdvancedDegreeEmployees = 100)
  (h3 : CollegeDegreeOnlyEmployees = 100)
  (h4 : MaleCollegeOnlyEmployees = 40) :
  FemaleEmployees ≥ 60 := by
  sorry

end FemaleEmployeesAtLeastSixty_l287_287166


namespace problem_I_tangent_problem_II_monotonicity_l287_287614

-- Define the function f(x) = x - a * log x
def f (x : ℝ) (a : ℝ) : ℝ := x - a * Real.log x

-- Statement for Problem I
theorem problem_I_tangent (x : ℝ) (y : ℝ) : 
  let a := 2 in 
  let f_1 : ℝ := f 1 a in 
  let f_prime_1 : ℝ := (1 - 2 / 1) in
  y - f_1 = f_prime_1 * (x - 1) → x + y - 2 = 0 :=
sorry

-- Define the function h(x) = x - log x + 2 / x
def h (x : ℝ) : ℝ := f x 1 + 2 / x

-- Statement for Problem II
theorem problem_II_monotonicity (x : ℝ) : 
  (∀ x, 0 < x ∧ x < 2 → deriv h x < 0) ∧ 
  (∀ x, 2 < x → deriv h x > 0) :=
sorry

end problem_I_tangent_problem_II_monotonicity_l287_287614


namespace pie_difference_l287_287295

theorem pie_difference:
  ∀ (a b c d : ℚ), a = 6 / 7 → b = 3 / 4 → (a - b) = c → c = 3 / 28 :=
by
  sorry

end pie_difference_l287_287295


namespace derivative_at_2_l287_287366

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287366


namespace no_valid_operation_for_equation_l287_287493

def equivalent_op (op: Char) (a b : Float) :=
  match op with
  | '/' => a / b
  | '*' => a * b
  | '+' => a + b
  | '-' => a - b
  | _   => 0 -- Invalid operation case

theorem no_valid_operation_for_equation :
  ∀ op : Char, (8 ≠ equivalent_op op 8 2 / 3) :=
by {
  intro op,
  fin_cases op;
  simp [equivalent_op];
  -- Verification of each case.
  sorry
}

end no_valid_operation_for_equation_l287_287493


namespace price_of_boxes_l287_287400

variable (x p_m p_p : ℕ) -- Define variables for number of boxes and prices per box

-- Conditions stated in the problem
def condition1 : Prop :=
  ∃ (money_a money_b : ℕ), (x - 4) * p_m + 6 = money_a ∧ x * p_p = money_b ∧ money_a = money_b

def condition2 : Prop :=
  ∃ (initial_money : ℕ), initial_money = x * p_m ∧ initial_money = x * p_p + 6

def condition3 : Prop :=
  ∀ (initial_money_a : ℕ), initial_money_a = 3 * ((x - 4) * p_m + 6) ∧ 
  initial_money_a = p_m * (x + 31) + 6

-- The theorem to be proved
theorem price_of_boxes :
  condition1 x p_m p_p ∧ condition2 x p_m p_p ∧ condition3 x p_m p_p → p_m = 12 ∧ p_p = 10 :=
by
  intros
  sorry

end price_of_boxes_l287_287400


namespace race_last_part_length_l287_287762

theorem race_last_part_length (total_len first_part second_part third_part last_part : ℝ) 
  (h1 : total_len = 74.5) 
  (h2 : first_part = 15.5) 
  (h3 : second_part = 21.5) 
  (h4 : third_part = 21.5) :
  last_part = total_len - (first_part + second_part + third_part) → last_part = 16 :=
by {
  intros,
  sorry
}

end race_last_part_length_l287_287762


namespace circumcircles_touch_at_A_find_BE_l287_287857

variables {A B C D E F : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

def points_on_side (A B C D E: Point) (BC : Line) : Prop :=
  D ∈ BC ∧ E ∈ BC

def point_on_extension (A B F: Point) (BA : Line) : Prop :=
  F ∉ BA

def equal_angles (D A E C A F: Point) (DAE CAF: Angle) : Prop :=
  DAE = CAF

def circumcircle_radius_relation (R1 R2 : ℝ) : Prop :=
  R1 = 2 * R2

theorem circumcircles_touch_at_A
  (A B C D E : Point) {BC : Line} (F : Point) 
  (h1 : points_on_side A B C D E BC)
  (h2 : point_on_extension A B F (line B A))
  (h3 : equal_angles D A E C A F (angle D A E) (angle C A F)) :
  touches (circumcircle A B D) (circumcircle A E C) A :=
sorry

theorem find_BE
  (A B C D E : Point) {BC : Line} (h : BC.length = 6) (h' : AB.length = 4)
  (circumcircle_A_B_D A B D : Circle) (circumcircle_A_E_C A E C : Circle)
  (R1 R2: ℝ)
  (h4 : circumcircle_radius_relation R1 R2) 
  (h5 : circumcircle_A_B_D.radius = R1)
  (h6 : circumcircle_A_E_C.radius = R2 ) :
  BE.length = 4 :=
sorry

end circumcircles_touch_at_A_find_BE_l287_287857


namespace trapezoid_ratio_l287_287587

theorem trapezoid_ratio (A B C D M N K : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup K]
  (CM MD CN NA AD BC : ℝ)
  (h1 : CM / MD = 4 / 3)
  (h2 : CN / NA = 4 / 3) 
  : AD / BC = 7 / 12 :=
by
  sorry

end trapezoid_ratio_l287_287587


namespace base_arithmetic_equation_l287_287052

theorem base_arithmetic_equation (b : ℕ) (h : b > 7) :
  let f : ℕ → ℕ := fun x => x
  in 452 * b^2 + 316 * b = 770 * b^2 - 452 * b^2 + (770 - 316 - 1) * b + (0 + 1) :=
b = 8 :=
sorry

end base_arithmetic_equation_l287_287052


namespace leanna_leftover_money_l287_287191

noncomputable def price_of_cd : ℝ := 14
noncomputable def total_money : ℝ := 37

noncomputable def price_of_cassette : ℝ :=
  total_money - 2 * price_of_cd

noncomputable def cost_of_second_option : ℝ :=
  price_of_cd + 2 * price_of_cassette

theorem leanna_leftover_money : total_money - cost_of_second_option = 5 :=
by
  unfold total_money price_of_cd price_of_cassette cost_of_second_option
  sorry

end leanna_leftover_money_l287_287191


namespace ABFCDE_perimeter_l287_287499

-- Definitions of the problem conditions
def is_square (s : ℝ) : Prop := s > 0

def is_isosceles_right_triangle (a : ℝ) : Prop := a > 0

-- Given the perimeter of square ABCD is 40 inches
def square_perimeter_condition : Prop := (4 * side_length = 40)

-- The side length of the square, deduced from its perimeter
def side_length_of_square : ℝ := 40 / 4

-- Prove that the perimeter of ABFCDE is 20 + 20√2
theorem ABFCDE_perimeter (side_length : ℝ) (triangle_side : ℝ) :
  is_square side_length ∧ is_isosceles_right_triangle triangle_side ∧ (4 * side_length = 40) →
  (2 * side_length + 2 * triangle_side * sqrt 2 = 20 + 20 * sqrt 2) :=
by
  intros h
  rw [←real.mul_assoc, real.mul_div_cancel' 40]
  rw [←real.mul_assoc (square_hypotenuse * sqrt 2)]
  sorry

end ABFCDE_perimeter_l287_287499


namespace derivative_at_2_l287_287351

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l287_287351


namespace correct_sampling_method_l287_287429

-- Definitions based on conditions
def number_of_classes : ℕ := 16
def sampled_classes : ℕ := 2
def sampling_method := "Lottery then Stratified"

-- The theorem statement based on the proof problem
theorem correct_sampling_method :
  (number_of_classes = 16) ∧ (sampled_classes = 2) → (sampling_method = "Lottery then Stratified") :=
sorry

end correct_sampling_method_l287_287429


namespace route_time_difference_l287_287213

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l287_287213


namespace loom_weaving_rate_l287_287446

theorem loom_weaving_rate :
  ∀ (cloth : ℝ) (time : ℝ), cloth = 24 → time = 187.5 → (cloth / time) = 0.128 :=
by
  intros cloth time h_cloth h_time
  rw [h_cloth, h_time]
  norm_num
  sorry

end loom_weaving_rate_l287_287446


namespace circumcenters_concyclic_l287_287007

-- Define the problem conditions
variables {E A B C D : Type} [AffineGeometry E]
variables [InParallelogram ABCD : Parallelogram A B C D]
variables [InParallelogramProperties : ∀ {E A B C D}, InParallelogram ABCD → InInteriorParallelogram E ABCD]

-- Main theorem
theorem circumcenters_concyclic (h : ∠ A B E = ∠ B C E): 
     @Concyclic E _ _ _ (circumcenter (triangle A B E)) (circumcenter (triangle B C E))
                         (circumcenter (triangle C D E)) (circumcenter (triangle D A E)) :=
sorry

end circumcenters_concyclic_l287_287007


namespace probability_second_marble_purple_correct_l287_287954

/-!
  Bag A has 5 red marbles and 5 green marbles.
  Bag B has 8 purple marbles and 2 orange marbles.
  Bag C has 3 purple marbles and 7 orange marbles.
  Bag D has 4 purple marbles and 6 orange marbles.
  A marble is drawn at random from Bag A.
  If it is red, a marble is drawn at random from Bag B;
  if it is green, a marble is drawn at random from Bag C;
  but if it is neither (an impossible scenario in this setup), a marble would be drawn from Bag D.
  Prove that the probability of the second marble drawn being purple is 11/20.
-/

noncomputable def probability_second_marble_purple : ℚ :=
  let p_red_A := 5 / 10
  let p_green_A := 5 / 10
  let p_purple_B := 8 / 10
  let p_purple_C := 3 / 10
  (p_red_A * p_purple_B) + (p_green_A * p_purple_C)

theorem probability_second_marble_purple_correct :
  probability_second_marble_purple = 11 / 20 := sorry

end probability_second_marble_purple_correct_l287_287954


namespace sin_cos_product_150_30_l287_287961

theorem sin_cos_product_150_30 : sin (150 * (Real.pi / 180)) * cos (30 * (Real.pi / 180)) = sqrt 3 / 4 :=
by
  sorry

end sin_cos_product_150_30_l287_287961


namespace finite_non_overlapping_crosses_l287_287487

theorem finite_non_overlapping_crosses (r : ℝ) (h : r = 100) : 
  ∃ (n : ℕ), ∀ (c₁ c₂ : ℝ), 
    (radius c₁ = 1/2 * real.sqrt 2 ∧ radius c₂ = 1/2 * real.sqrt 2) →
    (dist c₁ c₂ ≥ real.sqrt 2) →
    n < 100 :=
sorry

end finite_non_overlapping_crosses_l287_287487


namespace problem_statement_l287_287581

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f'' (x : ℝ) : ℝ := -Real.sin x - Real.cos x

theorem problem_statement (a : ℝ) (h : f'' a = 3 * f a) : 
  (Real.sin a)^2 - 3 / (Real.cos a)^2 + 1 = -14 / 9 := 
sorry

end problem_statement_l287_287581


namespace value_correct_l287_287848

def mean := 12
def std_dev := 1.2
def value_two_std_dev_less_than_mean := mean - 2 * std_dev

theorem value_correct : value_two_std_dev_less_than_mean = 9.6 := 
by 
  -- Sorry, the proof follows directly from the definitions
  sorry

end value_correct_l287_287848


namespace factor_poly_PQ_sum_l287_287154

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l287_287154


namespace find_f_prime_at_two_l287_287325

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287325


namespace distance_centers_circumcircle_incircle_l287_287205

variables {A B C : Type} [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C]

-- Define the isosceles triangle with the given properties
noncomputable def is_isosceles (A B C : point) : Prop :=
  dist A B = dist A C

-- Define the radii of the circumcircle and incircle
variables (r ϱ : ℝ)
variable (d : ℝ)

-- The distance between the centers of the circumcircle and the incircle
theorem distance_centers_circumcircle_incircle
  (ABC_isosceles : is_isosceles A B C)
  (circumcircle_radius : ℝ)
  (incircle_radius : ℝ) :
  d = sqrt (r * (r - 2 * ϱ)) := sorry

end distance_centers_circumcircle_incircle_l287_287205


namespace seedling_survival_rate_estimate_l287_287911

theorem seedling_survival_rate_estimate :
  let rates := [0.800, 0.870, 0.923, 0.883, 0.890, 0.915, 0.905, 0.897, 0.902] in
  let avg := (rates.foldl (λ acc r, acc + r) 0) / (rates.length : ℝ) in
  Float.roundDecimal avg 1 = 0.9 :=
by
  sorry

end seedling_survival_rate_estimate_l287_287911


namespace fg_of_two_l287_287641

theorem fg_of_two : (f : ℝ → ℝ) = (λ x, 5 - 4 * x) → (g : ℝ → ℝ) = (λ y, y^2 + 2) → f (g 2) = -19 :=
by
  sorry

end fg_of_two_l287_287641


namespace equivalent_single_discount_l287_287039

theorem equivalent_single_discount (P : ℝ) (hP : 0 ≤ P) : 
  let first_discount := 0.20 
  let second_discount := 0.10 
  let final_discount := 0.28
  ∀ (P : ℝ), (1 - final_discount) * P = (1 - second_discount) * (1 - first_discount) * P :=
by 
  intros P hP 
  let first_discount := 0.20
  let second_discount := 0.10
  let final_discount := 0.28
  rw [mul_assoc, mul_assoc, ←one_mul (1 - final_discount), mul_sub, one_mul, sub_mul, sub_mul]
  sorry

end equivalent_single_discount_l287_287039


namespace number_of_students_in_class_l287_287663

theorem number_of_students_in_class
  (G : ℕ) (E_and_G : ℕ) (E_only: ℕ)
  (h1 : G = 22)
  (h2 : E_and_G = 12)
  (h3 : E_only = 23) :
  ∃ S : ℕ, S = 45 :=
by
  sorry

end number_of_students_in_class_l287_287663


namespace arithmetic_geometric_sum_l287_287275

noncomputable def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop :=
  ∀ n, a_n 1 = 3 ∧ 
       (a_n 5 - 2 * b_n 2 = a_n 3) ∧
       (S_n n = n * (a_n 1 + a_n n) / 2)

noncomputable def geometric_sequence (b_n : ℕ → ℕ) : Prop :=
  ∀ n, b_n 1 = 1 ∧
       (b_n 2 + S_n 2 = 10)

noncomputable def cn_sequence (c_n : ℕ → ℕ) (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) : Prop :=
  ∀ n, c_n n = a_n n * b_n n

noncomputable def Tn_sum (T_n : ℕ → ℕ) (c_n : ℕ → ℕ) : Prop :=
  ∀ n, T_n n = ∑ k in range (n + 1), c_n k

theorem arithmetic_geometric_sum :
  ∃ a_n b_n S_n c_n T_n,
    arithmetic_sequence a_n S_n ∧
    geometric_sequence b_n ∧
    cn_sequence c_n a_n b_n ∧
    Tn_sum T_n c_n ∧
    (∀ n, a_n n = 2n + 1) ∧
    (∀ n, b_n n = 2 ^ (n - 1)) ∧
    (∀ n, T_n n = (2 * n - 1) * 2 ^ n + 1) :=
by
  sorry

end arithmetic_geometric_sum_l287_287275


namespace recurrence_relation_limit_f_n_no_limit_f_n_neg_l287_287047

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  let num := ∑ k in finset.range(n+1).filter(λ k, even k), (nat.choose n k) * (x ^ (k / 2))
  let denom := ∑ k in finset.range(n+1).filter(λ k, odd k), (nat.choose n k) * (x ^ (k / 2))
  num / denom

theorem recurrence_relation (n : ℕ) (x : ℝ) : 
  f_n (n + 1) x = (f_n n x + x) / (f_n n x + 1) :=
sorry

theorem limit_f_n (x : ℝ) : 
  (∀ n: ℕ, f_n n x) → (∀ y: ℝ, 0 ≤ x → tendsto (f_n n x) at_top (nhds y) → y = sqrt x) :=
sorry

theorem no_limit_f_n_neg (x : ℝ) :
  (x < 0) → (¬ (∃ y, tendsto (f_n n x) at_top (nhds y))) :=
sorry

end recurrence_relation_limit_f_n_no_limit_f_n_neg_l287_287047


namespace petya_cannot_ensure_three_consecutive_same_l287_287443

theorem petya_cannot_ensure_three_consecutive_same (
  n : ℕ,
  h : n ≥ 2
) : ¬ (∀ (f : Fin (2*n) → (Fin (2*n) → ℕ)) 
        (h1 : ∀ i, f i 0 ≠ f (i+1) % (2*n) 0 ∨ f i 1 ≠ f (i+1) % (2*n) 1), 
      ∃ i, f i 0 = f (i+1) % (2*n) 0 ∧ f i 0 = f (i+2) % (2*n) 0) := 
sorry

end petya_cannot_ensure_three_consecutive_same_l287_287443


namespace find_pq_l287_287862

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pq (p q : ℕ) 
(hp : is_prime p) 
(hq : is_prime q) 
(h : is_prime (q^2 - p^2)) : 
  p * q = 6 :=
by sorry

end find_pq_l287_287862


namespace integral_evaluation_eq_pi_minus_4_l287_287037

noncomputable def integral_expression : ℝ := 
  ∫ x in 0..2, (sqrt (4 - x^2) - 2 * x)

theorem integral_evaluation_eq_pi_minus_4 : 
  integral_expression = (Real.pi - 4) :=
by
  sorry

end integral_evaluation_eq_pi_minus_4_l287_287037


namespace john_trip_duration_l287_287190

noncomputable def clock_trip_duration (start_time end_time : Nat) : Nat :=
  if start_time > end_time then 12 * 60 - start_time + end_time else end_time - start_time

theorem john_trip_duration :
  let start_time := 10 * 60 + 54 -- 10:54 a.m. in minutes
  let end_time := 16 * 60 + 50 -- 4:50 p.m. in minutes
  clock_trip_duration start_time end_time = 5 * 60 + 55 := -- 5 hours and 55 minutes
by
  unfold clock_trip_duration
  rw [if_pos]
  sirius


end john_trip_duration_l287_287190


namespace max_speed_of_cart_l287_287010

noncomputable def vmax (R a : ℝ) : ℝ :=
  ((16 * Real.pi ^ 2 * R ^ 2 * a ^ 2) / (16 * Real.pi ^ 2 + 1)) ^ (1/4 : ℝ)

theorem max_speed_of_cart :
  let R := 4
  let a := 2
  vmax R a ≈ 2.8 :=
by
  let R := 4
  let a := 2
  have h : vmax R a = ((16 * Real.pi ^ 2 * R ^ 2 * a ^ 2) / (16 * Real.pi ^ 2 + 1))^(1/4 : ℝ) := rfl
  rw [h, Real.sqrt, Real.sqrt, Real.sqrt, Real.sqrt]
  sorry

end max_speed_of_cart_l287_287010


namespace domain_myFunction_l287_287300

noncomputable def myFunction (x : ℝ) : ℝ :=
  (x^3 - 125) / (x + 125)

theorem domain_myFunction :
  {x : ℝ | ∀ y, y = myFunction x → x ≠ -125} = { x : ℝ | x ≠ -125 } := 
by
  sorry

end domain_myFunction_l287_287300


namespace outer_squares_equal_three_times_inner_squares_l287_287972

theorem outer_squares_equal_three_times_inner_squares
  (a b c m_a m_b m_c : ℝ) 
  (h : m_a^2 + m_b^2 + m_c^2 = 3 / 4 * (a^2 + b^2 + c^2)) :
  a^2 + b^2 + c^2 = 3 * (m_a^2 + m_b^2 + m_c^2) := 
by 
  sorry

end outer_squares_equal_three_times_inner_squares_l287_287972


namespace angle_BAC_measure_l287_287174

theorem angle_BAC_measure 
  (A B C D : Type) 
  [AffineSpace ℝ A B]
  (h1 : 3 * B.angle x = A.angle B (Affine.rotation x)) 
  (h2 : 4 * B.angle x = A.angle B (Affine.rotation x)) 
  (A_eq_B : A = B)  
  (A_eq_C : A = C)
  (on_side_BC_D : B.onSide C (B.angle).rotation D)
  (AD_bisects_ABC: bisects D (A.angle B x)) :
  measure (A.angle B x) = 72 :=
begin 
  exact sorry
end

end angle_BAC_measure_l287_287174


namespace number_of_valid_three_digit_numbers_l287_287951

theorem number_of_valid_three_digit_numbers : 
  (∃ A B C : ℕ, 
      (100 * A + 10 * B + C + 297 = 100 * C + 10 * B + A) ∧ 
      (0 ≤ A ∧ A ≤ 9) ∧ 
      (0 ≤ B ∧ B ≤ 9) ∧ 
      (0 ≤ C ∧ C ≤ 9)) 
    ∧ (number_of_such_valid_numbers = 70) :=
by
  sorry

def number_of_such_valid_numbers : ℕ := 
  sorry

end number_of_valid_three_digit_numbers_l287_287951


namespace trajectory_of_P_l287_287627

-- Define the points A and B
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def dist (M N : Point2D) : ℝ :=
  real.sqrt ((M.x - N.x) ^ 2 + (M.y - N.y) ^ 2)

-- Define the coordinates of points A and B
def A : Point2D := {x := 0, y := 2}
def B : Point2D := {x := 0, y := -2}

-- Define the condition that the moving point P satisfies
def satisfies_condition (P : Point2D) : Prop :=
  dist P A + dist P B = 4

-- Define the trajectory of P
def lies_on_segment (P : Point2D) : Prop :=
  (P.y = 0) ∧ (P.x = 0)

-- The theorem to be proved
theorem trajectory_of_P (P : Point2D) (h : satisfies_condition P) : lies_on_segment P :=
  sorry

end trajectory_of_P_l287_287627


namespace smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287539

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

-- Define the property of being a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property of being a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_three_digit_palindrome_not_five_digit_product_with_111 :
  ∃ n, is_three_digit n ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) ∧ n = 191 :=
sorry  -- The proof steps are not required here

end smallest_three_digit_palindrome_not_five_digit_product_with_111_l287_287539


namespace minimum_possible_cells_minimum_certain_cells_l287_287243

-- Definitions of the problem conditions:
def condition (n t : ℕ) : Prop :=
  ∀ (cube : ℕ → ℕ → ℕ → ℕ), ∀ i j k : ℕ,
  (1 ≤ t ∧ t ≤ n) ∧
  (∀ x, cube i j x ≥ t ∨ cube i x k ≥ t ∨ cube x j k ≥ t → 
  (∀ y, y < n → cube i j y = n ∧ cube i y k = n ∧ cube y j k = n))

-- Theorem to prove the minimum number of cells to possibly infect the whole cube:
theorem minimum_possible_cells (n t: ℕ) (h_cond: condition n t) : 
  ∃ initially_infected_cells, initially_infected_cells = t^3 := 
begin 
  sorry
end

-- Theorem to prove the number of cells to ensure to infect the whole cube:
theorem minimum_certain_cells (n t : ℕ) (h_cond: condition n t) : 
  ∃ initially_infected_cells, 
  initially_infected_cells = n^3 - (n - (t - 1))^3 + 1 := 
begin 
  sorry
end

end minimum_possible_cells_minimum_certain_cells_l287_287243


namespace negation_equiv_l287_287837

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop := 
  (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ is_even c)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop := 
  (is_even a ∧ is_even b) ∨ 
  (is_even a ∧ is_even c) ∨ 
  (is_even b ∧ is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ ¬is_even c)
  
theorem negation_equiv (a b c : ℕ) : 
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c := 
sorry

end negation_equiv_l287_287837


namespace car_payment_months_l287_287737

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l287_287737


namespace dealer_purchased_articles_l287_287918

/-
The dealer purchases some articles for Rs. 25 and sells 12 articles for Rs. 38. 
The dealer has a profit percentage of 90%. Prove that the number of articles 
purchased by the dealer is 14.
-/

theorem dealer_purchased_articles (x : ℕ) 
    (total_cost : ℝ) (group_selling_price : ℝ) (group_size : ℕ) (profit_percentage : ℝ) 
    (h1 : total_cost = 25)
    (h2 : group_selling_price = 38)
    (h3 : group_size = 12)
    (h4 : profit_percentage = 90 / 100) :
    x = 14 :=
by
  sorry

end dealer_purchased_articles_l287_287918


namespace three_digit_numbers_count_natural_numbers_count_l287_287625

-- Define sets A and B
def A : Set Nat := { x | 1 < log 2 x ∧ log 2 x < 3 ∧ x ∈ Nat }
def B : Set Nat := { 4, 5, 6, 7, 8 }

-- Union of sets A and B
def A_union_B : Set Nat := A ∪ B

-- Main theorem statements

-- Theorem 1: Number of different three-digit numbers
theorem three_digit_numbers_count :
  (A_union_B.card).choose 3 = 120 :=
sorry

-- Definition for the second problem
def valid_numbers_greater_4000 (A : Set Nat) (B : Set Nat) : Set Nat :=
  { n : Nat | ∃ a ∈ A, ∃ b1 ∈ B, ∃ b2 ∈ B, ∃ b3 ∈ B, 
    (n > 4000 ∧ -- Ensure n > 4000
    (a, b1, b2, b3).2.AreDistinct()) } -- Ensure distinct digits

-- Theorem 2: Number of distinct natural numbers greater than 4000
theorem natural_numbers_count :
  (valid_numbers_greater_4000 A B).card = 564 :=
sorry

end three_digit_numbers_count_natural_numbers_count_l287_287625


namespace cos_5_deg_approx_l287_287962

theorem cos_5_deg_approx :
  |Float.cos (5 * Float.pi / 180) - 0.996195| ≤ 10^(-6) :=
by
  sorry

end cos_5_deg_approx_l287_287962


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287909

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287909


namespace intersect_centroid_l287_287681

variable {A B C P : Point}

-- Centroids of the triangles ABP, BCP, and CAP
variable (G1 : Point)
variable (G2 : Point)
variable (G3 : Point)

-- Lines drawn through the centroids parallel to CP, AP, BP respectively
variable (lineG3parallelCP : Line)
variable (lineG1parallelAP : Line)
variable (lineG2parallelBP : Line)

-- The centroid of triangle ABC
variable (G : Point)

-- Conditions
axiom centroid_ABP : is_centroid A B P G1
axiom centroid_BCP : is_centroid B C P G2
axiom centroid_CAP : is_centroid C A P G3
axiom line_parallel_G3_CP : line_parallel G3 P C lineG3parallelCP
axiom line_parallel_G1_AP : line_parallel G1 P A lineG1parallelAP
axiom line_parallel_G2_BP : line_parallel G2 P B lineG2parallelBP

-- Required proof: to show that the lines intersect at G, the centroid of triangle ABC
theorem intersect_centroid : 
  centroid_ABC A B C G →
  exists (P : Point), is_intersect_point_three_lines lineG3parallelCP lineG1parallelAP lineG2parallelBP P ∧ P = G :=
sorry

end intersect_centroid_l287_287681


namespace fibonacci_rabbits_l287_287777

theorem fibonacci_rabbits : 
  ∀ (F : ℕ → ℕ), 
    (F 0 = 1) ∧ 
    (F 1 = 1) ∧ 
    (∀ n, F (n + 2) = F n + F (n + 1)) → 
    F 12 = 233 := 
by 
  intro F h; sorry

end fibonacci_rabbits_l287_287777


namespace find_a_values_l287_287083

theorem find_a_values (a t t₁ t₂ : ℝ) :
  (t^2 + (a - 6) * t + (9 - 3 * a) = 0) ∧
  (t₁ = 4 * t₂) ∧
  (t₁ + t₂ = 6 - a) ∧
  (t₁ * t₂ = 9 - 3 * a)
  ↔ (a = -2 ∨ a = 2) := sorry

end find_a_values_l287_287083


namespace tangent_point_value_l287_287876

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287876


namespace number_of_real_roots_of_cubic_l287_287724

-- Define the real number coefficients
variables (a b c d : ℝ)

-- Non-zero condition on coefficients
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Statement of the problem: The cubic polynomial typically has 3 real roots
theorem number_of_real_roots_of_cubic :
  ∃ (x : ℝ), (x ^ 3 + x * (c ^ 2 - d ^ 2 - b * d) - (b ^ 2) * c = 0) := by
  sorry

end number_of_real_roots_of_cubic_l287_287724


namespace circle_equation_l287_287413

theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = r ↔ (x = 0 ∧ y = 0) → ((x - 3) ^ 2 + (y - 1) ^ 2 = 10) :=
by
  sorry

end circle_equation_l287_287413


namespace tangent_point_value_l287_287875

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l287_287875


namespace min_trips_required_l287_287398

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l287_287398


namespace trapezoid_area_l287_287451

-- Define the conditions for the trapezoid
variables (legs : ℝ) (diagonals : ℝ) (longer_base : ℝ)
variables (h : ℝ) (b : ℝ)
hypothesis (leg_length : legs = 40)
hypothesis (diagonal_length : diagonals = 50)
hypothesis (base_length : longer_base = 60)
hypothesis (altitude : h = 100 / 3)
hypothesis (shorter_base : b = 60 - (40 * (Real.sqrt 11)) / 3)

-- The statement to prove the area of the trapezoid
theorem trapezoid_area : 
  ∃ (A : ℝ), 
  (A = ((b + longer_base) * h) / 2) →
  A = (10000 - 2000 * (Real.sqrt 11)) / 9 :=
by
  -- placeholder for the proof
  sorry

end trapezoid_area_l287_287451


namespace opposite_of_neg_five_l287_287796

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l287_287796


namespace problem_statement_l287_287785

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement
  (even_f : ∀ x, f x = f (-x))
  (deriv_eq : ∀ x, deriv f x = deriv f (2 - x))
  (monotonic_decreasing : ∀ x y, 2023 ≤ x ∧ x ≤ y ∧ y ≤ 2024 → f x ≥ f y)
  (a_def : a = 4 ^ (-0.8))
  (b_def : b = 1 / 3)
  (c_def : c = Real.log (3 / 2)) :
  f a < f b ∧ f b < f c := sorry

end problem_statement_l287_287785


namespace proof_triangle_ABC_l287_287656

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
B = real.pi / 3 ∧ 
cos A = 4 / 5 ∧
b = real.sqrt 3 ∧
sin C = (3 + 4 * real.sqrt 3) / 10 ∧ 
(triangle_area a b C = (36 + 9 * real.sqrt 3) / 50)

theorem proof_triangle_ABC : 
  ∃ A B C a b c, triangle_ABC A B C a b c := 
by {
  sorry
}

end proof_triangle_ABC_l287_287656


namespace value_of_y_l287_287394

theorem value_of_y (x y : ℕ) (h1 : x % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 :=
sorry

end value_of_y_l287_287394


namespace estimate_probability_of_seedling_survival_l287_287912

def survival_rates : List ℚ := [0.800, 0.870, 0.923, 0.883, 0.890, 0.915, 0.905, 0.897, 0.902]

-- Define a function to compute the average of a list of rational numbers
def average (l : List ℚ) : ℚ :=
  l.sum / l.length

-- Estimate the probability of seedling survival after transplantation
def estimated_survival_probability : ℚ :=
  (Float.ofRat (average survival_rates)).round.toRat / 10

theorem estimate_probability_of_seedling_survival : estimated_survival_probability = 0.9 := by
  sorry

end estimate_probability_of_seedling_survival_l287_287912


namespace problem_solution_l287_287644

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_solution : f(g(2)) = -19 :=
by
  -- proof omitted
  sorry

end problem_solution_l287_287644


namespace coplanarity_condition_l287_287293

def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := 
  (-2 + r, 5 - 3 * k * r, k * r)

def line2 (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 * t, 2 + 2 * t, -2 * t)

def lines_coplanar (k : ℝ) : Prop :=
  ∃ (r t : ℝ), line1 r k = line2 t ∨ 
  (∃ (a b : ℝ), 
    line1 a k = 
    (line2 b).1 * 2 - (line2 b).2 * 3 * k + (line2 b).3 * k)

theorem coplanarity_condition (k : ℝ) : 
  k = -1 ∨ k = -1 / 3 ↔ lines_coplanar k := 
sorry

end coplanarity_condition_l287_287293


namespace find_f_prime_at_2_l287_287334

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287334


namespace plane_through_point_parallel_to_another_l287_287036

def point := ℝ × ℝ × ℝ
def normalVector := ℝ × ℝ × ℝ

-- Given conditions:
def M : point := (-2, 0, 3)
def n : normalVector := (2, -1, -3)
def parallel_plane_equation (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 5 = 0

-- Assertion to prove:
def desired_plane_equation (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 13 = 0

theorem plane_through_point_parallel_to_another :
  ∃ (d : ℝ), (λ x y z, 2 * x - y - 3 * z + d = 0) (-2) 0 3 ∧
  (∀ x y z, ∃ d', (λ x y z, 2 * x - y - 3 * z + d' = 0) x y z ↔ parallel_plane_equation x y z)
  → (λ x y z, 2 * x - y - 3 * z + 13 = 0) (-2) 0 3
  :=
  sorry

end plane_through_point_parallel_to_another_l287_287036


namespace pencils_currently_have_l287_287963

-- Defining the initial conditions of the problem.
def initial_pencils : Nat := 50

-- Defining the pencils lost while moving to school.
def pencils_lost_moving : Nat := 8

-- Defining the fraction of pencils misplaced in the first week.
def fraction_misplaced_first_week : ℚ := 1 / 3

-- Defining the extra pencils bought after feeling frustrated.
def extra_pencils_bought : Nat := 10

-- Defining the fraction of pencils lost in the second week.
def fraction_lost_second_week : ℚ := 1 / 4

-- Defining the pencils found by her friend.
def pencils_found_by_friend : Nat := 5

-- Formalizing the problem to prove the final number of pencils Charley has.
theorem pencils_currently_have : 
  let pencils_after_moving := 50 - 8,
      pencils_after_first_week := pencils_after_moving - (1/3 : ℚ) * pencils_after_moving,
      pencils_after_buying := pencils_after_first_week + 10,
      pencils_after_second_week := pencils_after_buying - (1/4 : ℚ) * pencils_after_buying,
      pencils_after_finding := pencils_after_second_week + 5 in
  pencils_after_finding = 34 := sorry

end pencils_currently_have_l287_287963


namespace sqrt_inequality_solution_l287_287125

theorem sqrt_inequality_solution (x : ℝ) (hx : x ≠ 1) :
  (sqrt (x^2 - 2*x + 2) ≥ -sqrt 5 * x) ↔ (x ∈ Icc (-1 : ℝ) 1 ∪ Ioi 1) :=
sorry

end sqrt_inequality_solution_l287_287125


namespace intervals_of_monotonicity_of_f_l287_287208

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem intervals_of_monotonicity_of_f (a b c d : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ d = P.2 ∧ (12 * P.1 - P.2 - 4 = 0))
  (h2 : ∃ x : ℝ, x = 2 ∧ (f a b c d x = 0) ∧ (∃ x : ℝ, x = 0 ∧ (3 * a * x^2 + 2 * b * x + c = 12))) 
  : ( ∃ a b c d : ℝ , (f a b c d) = (2 * x^3 - 9 * x^2 + 12 * x -4)) := 
  sorry

end intervals_of_monotonicity_of_f_l287_287208


namespace odd_function_sufficient_but_not_necessary_l287_287116

theorem odd_function_sufficient_but_not_necessary (ϕ : ℝ) :
  (∃ (h : ϕ = π / 2), ∀ x : ℝ, f(x) = cos (2 * x + ϕ) → f(-x) = -f(x)) ∧ 
  (∀ x : ℝ, f(x) = cos (2 * x + ϕ) → f(-x) = -f(x) → ϕ ≠ π / 2)  :=
by
  sorry

end odd_function_sufficient_but_not_necessary_l287_287116


namespace cos_pow_sum_cos_binom_l287_287236

open Real

theorem cos_pow_sum_cos_binom (n : ℕ) (hn : 0 < n) (θ : ℝ) : 
  cos θ ^ n = (1 / 2^n) * ∑ k in Finset.range (n + 1), Nat.choose n k * cos ((n - 2 * k) * θ) :=
by
  sorry

end cos_pow_sum_cos_binom_l287_287236


namespace exists_natural_n_l287_287223

theorem exists_natural_n (a b : ℕ) (h1 : b ≥ 2) (h2 : Nat.gcd a b = 1) : ∃ n : ℕ, (n * a) % b = 1 :=
by
  sorry

end exists_natural_n_l287_287223


namespace John_work_rate_l287_287697

-- Define the conditions
def Rose_work_rate := 1 / 16
def Combined_work_rate := 1 / 5.33

-- Define the statement to be proved
theorem John_work_rate:
  ∃ J : ℝ, Rose_work_rate + 1 / J = Combined_work_rate ∧ J = 8 :=
by
  have h : Rose_work_rate + 1 / 8 = Combined_work_rate := by sorry
  use 8
  exact ⟨h, rfl⟩

end John_work_rate_l287_287697


namespace cos_angle_focus_points_l287_287109

def hyperbola : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 - y^2 = 2 }

def focus_1 := (2, 0) -- assuming specific coordinates based on the problem
def focus_2 := (-2, 0) -- assuming specific coordinates based on the problem

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

noncomputable def cos_angle (F1 P F2 : ℝ × ℝ) : ℝ :=
  (distance P F1)^2 + (distance P F2)^2 - (distance F1 F2)^2 /
  (2 * distance P F1 * distance P F2)

theorem cos_angle_focus_points (P : ℝ × ℝ) (hP : P ∈ hyperbola)
  (h_dist : distance P focus_1 = 2 * distance P focus_2) :
  cos_angle focus_1 P focus_2 = 3 / 4 :=
sorry

end cos_angle_focus_points_l287_287109


namespace solve_for_k_in_quadratic_l287_287097

theorem solve_for_k_in_quadratic :
  ∃ k : ℝ, (∀ x1 x2 : ℝ,
    x1 + x2 = 3 ∧
    x1 * x2 + 2 * x1 + 2 * x2 = 1 ∧
    (x1^2 - 3*x1 + k = 0) ∧ (x2^2 - 3*x2 + k = 0)) →
  k = -5 :=
sorry

end solve_for_k_in_quadratic_l287_287097


namespace total_molecular_weight_calc_l287_287834

theorem total_molecular_weight_calc :
  let Ca_molecular_weight := 40.08
  let I_molecular_weight := 126.90
  let Na_molecular_weight := 22.99
  let Cl_molecular_weight := 35.45
  let K_molecular_weight := 39.10
  let S_molecular_weight := 32.06
  let O_molecular_weight := 16.00
  let CaI2_molecular_weight := Ca_molecular_weight + 2 * I_molecular_weight
  let NaCl_molecular_weight := Na_molecular_weight + Cl_molecular_weight
  let K2SO4_molecular_weight := 2 * K_molecular_weight + S_molecular_weight + 4 * O_molecular_weight
  let CaI2_weight := 10 * CaI2_molecular_weight
  let NaCl_weight := 7 * NaCl_molecular_weight
  let K2SO4_weight := 15 * K2SO4_molecular_weight
in CaI2_weight + NaCl_weight + K2SO4_weight = 5961.78 :=
by
  sorry

end total_molecular_weight_calc_l287_287834


namespace derivative_at_2_l287_287374

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l287_287374


namespace P_n_diff_winning_probability_l287_287422

def face_A : ℝ := 2
def face_BCD : ℝ := 1

def start_position : ℕ := 0
def win_position : ℕ := 99
def lose_position : ℕ := 100

def P : ℕ → ℝ 
| 0       := 1
| 1       := 3/4
| 2       := 13/16
| n       := if (n >= 3) then 3/4 * P (n-1) + 1/4 * P (n-2) else 0

theorem P_n_diff (n : ℕ) (hn : 2 ≤ n ≤ 99) : P n - P (n-1) = - (1 / 4) * (P (n-1) - P (n-2)) :=
by {
    sorry
}

theorem winning_probability : P 99 = (4 / 5) * (1 - (-1 / 4) ^ 100) :=
by {
    sorry
}

end P_n_diff_winning_probability_l287_287422


namespace problem_b_problem_c_problem_d_l287_287089

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end problem_b_problem_c_problem_d_l287_287089


namespace repeating_decimal_fraction_l287_287514

theorem repeating_decimal_fraction: 
  let x := (428 : ℚ) / 999 in
  let y := (2856 : ℚ) / 999 in
  (x / y = (1 : ℚ) / 6) :=
by { sorry }

end repeating_decimal_fraction_l287_287514


namespace area_of_largest_circle_l287_287006

theorem area_of_largest_circle (L W : ℝ) (hL : L = 18) (hW : W = 8) : 
  let P := 2 * (L + W) in
  P = 52 → 
  let r := 52 / (2 * Real.pi) in
  let area := Real.pi * r ^ 2 in
  area = 676 / Real.pi := 
by 
  intros hP hr harea 
  sorry

end area_of_largest_circle_l287_287006


namespace log_magnitude_comparison_l287_287577

open Real

theorem log_magnitude_comparison (a : ℝ) (x : ℝ) (h₁ : 1 < a) (h₂ : 0 < x) (h₃ : x < 1) :
  abs (log a (1 - x)) > abs (log a (1 + x)) :=
sorry

end log_magnitude_comparison_l287_287577


namespace gold_rod_weight_l287_287944

theorem gold_rod_weight
  (length : ℕ)
  (wt_end_1 wt_end_5 : ℝ)
  (h_length : length = 5)
  (h_wt_end_1 : wt_end_1 = 4)
  (h_wt_end_5 : wt_end_5 = 2)
  (h_uniform : ∀ n, 1 ≤ n ∧ n ≤ length → wt_end_1 + (n - 1) * (wt_end_5 - wt_end_1) / (length - 1) = 4 - (n - 1) * 0.5) :
  (∑ n in finset.range length, wt_end_1 + (n) * (wt_end_5 - wt_end_1) / (length - 1)) = 15 :=
by
  sorry


end gold_rod_weight_l287_287944


namespace limit_of_lower_derivatives_l287_287709

noncomputable def condition1 (f : ℝ → ℝ) (l : ℝ) :=
  tendsto f at_top (nhds l)

noncomputable def condition2 (f : ℝ → ℝ) (n : ℕ) :=
  tendsto (λ x, iterated_deriv n f x) at_top (nhds 0)

noncomputable def prove_statement (f : ℝ → ℝ) (n : ℕ) :=
  ∀ k : ℕ, k ∈ finset.range n → k > 0 → tendsto (λ x, iterated_deriv k f x) at_top (nhds 0)

theorem limit_of_lower_derivatives (f : ℝ → ℝ) (l : ℝ) (n : ℕ) (h₁ : condition1 f l) (h₂ : condition2 f n) :
  prove_statement f n :=
by
  -- Insert proof here
  sorry

end limit_of_lower_derivatives_l287_287709


namespace day_after_1999_cubed_days_is_tuesday_l287_287286

theorem day_after_1999_cubed_days_is_tuesday : 
    let today := "Monday"
    let days_in_week := 7
    let target_days := 1999 ^ 3
    ∃ remaining_days, remaining_days = (target_days % days_in_week) ∧ today = "Monday" ∧ remaining_days = 1 → 
    "Tuesday" = "Tuesday" := 
by
  sorry

end day_after_1999_cubed_days_is_tuesday_l287_287286


namespace part1_solution_part2_solution_l287_287620

def f (x : ℝ) : ℝ := |2 * x - 3| + 3
def g (x : ℝ) (a : ℝ) : ℝ := 2 * x + a / x

theorem part1_solution (x : ℝ) : x ≥ 6 / 5 → f x ≤ 3 * x :=
by
  sorry

theorem part2_solution (a : ℝ) : (∃ x : ℝ, f x ∈ Set.Ici 3) ∧ 
  (∀ y ∈ Set.Ici 3, ∃ x : ℝ, g x a = y) → 0 < a ∧ a ≤ 9 / 8 :=
by
  sorry

end part1_solution_part2_solution_l287_287620


namespace sector_area_is_120_l287_287775

-- Definitions based on the problem conditions.
def sector_area (r l : ℝ) : ℝ := 0.5 * r * l

def fan_shaped_field_area (d l : ℝ) : ℝ :=
  let r := d / 2 in
  sector_area r l

-- Given conditions.
def arc_length : ℝ := 30
def diameter : ℝ := 16

-- Statement to prove.
theorem sector_area_is_120 : fan_shaped_field_area diameter arc_length = 120 :=
by
  -- Definitions ensure types are correctly used and understood in Lean.
  sorry

end sector_area_is_120_l287_287775


namespace a_general_formula_T_sum_l287_287100

-- Define the sequence {a_n}
def a (n : ℕ) : ℝ := if n = 1 then 2 else 3^(n - 1)

-- Define the sum of the first n terms {S_n}
def S (n : ℕ) := (a n + 2 * n + 2) / 2

-- Define the first problem: Prove that the general formula a_n = 3^(n-1)
theorem a_general_formula (n : ℕ) (h : ∀ n, S n = (a (n + 1)) / 2 - n - 1) :
  a n = 3^(n - 1) :=
sorry

-- Define the sequence given in the second problem
def b (n : ℕ) := 2 * 3^n / (a n * a (n + 1))

-- Define the sum of the first n terms {T_n}
def T (n : ℕ) := (1/2) - 1 / (3^(n + 1) - 1)

-- Define the second problem: Prove that T_n = (1/2) - (1/(3^(n+1)-1))
theorem T_sum (n : ℕ) :
  ∑ i in Finset.range n, b i = T n :=
sorry

end a_general_formula_T_sum_l287_287100


namespace evaluate_expression_l287_287056

theorem evaluate_expression :
  ((3.5 / 0.7) * (5 / 3) + (7.2 / 0.36) - ((5 / 3) * (0.75 / 0.25))) = 23.3335 :=
by
  sorry

end evaluate_expression_l287_287056


namespace retailer_should_focus_on_mode_l287_287914

-- Define the conditions as options.
inductive ClothingModels
| Average
| Mode
| Median
| Smallest

-- Define a function to determine the best measure to focus on in the market share survey.
def bestMeasureForMarketShareSurvey (choice : ClothingModels) : Prop :=
  match choice with
  | ClothingModels.Average => False
  | ClothingModels.Mode => True
  | ClothingModels.Median => False
  | ClothingModels.Smallest => False

-- The theorem stating that the mode is the best measure to focus on.
theorem retailer_should_focus_on_mode : bestMeasureForMarketShareSurvey ClothingModels.Mode :=
by
  -- This proof is intentionally left blank.
  sorry

end retailer_should_focus_on_mode_l287_287914


namespace cannot_obtain_zero_2022_can_obtain_zero_2023_l287_287266

theorem cannot_obtain_zero_2022 : ∀ (board : List ℕ), 
  (board = List.range 2022) → 
  (∀ (a b : ℕ), a ∈ board → b ∈ board → 
   board = (board.erase a).erase b ++ [b - a]) →
  ¬ (List.foldl (λ x y, abs (x - y)) 0 board = 0) :=
by
  intros board h_board h_step
  sorry

theorem can_obtain_zero_2023 : ∀ (board : List ℕ), 
  (board = List.range 2023) → 
  (∀ (a b : ℕ), a ∈ board → b ∈ board → 
   board = (board.erase a).erase b ++ [b - a]) →
  (∃ final_board, List.foldl (λ x y, abs (x - y)) 0 final_board = 0) :=
by
  intros board h_board h_step
  sorry

end cannot_obtain_zero_2022_can_obtain_zero_2023_l287_287266


namespace find_a_l287_287671

open Real

theorem find_a (a : ℝ) : 
  let M := (1, 1) in
  let circle_center := (-1, 2) in
  let circle_radius := sqrt 5 in
  let line_l_slope : ℝ := -a in
  -- condition that l is tangent to the circle
  let line_l_Tangent := (line_l_slope * (fst M - fst circle_center) + (snd M - snd circle_center)) ^ 2 = circle_radius ^ 2 * (1 + line_l_slope ^ 2) in
  -- condition that l is perpendicular to ax + y - 1 = 0
  let perp_line_slope := -a in
  line_l_Tangent ∧ (line_l_slope * perp_line_slope = -1) → a = 1 / 2 :=
sorry

end find_a_l287_287671


namespace area_less_than_perimeter_probability_l287_287005

-- Define the side length s as the sum of a pair of dice rolls (ranging from 2 to 12)
noncomputable def sum_of_dice_rolls : ℕ := sorry

-- Define the probability function for a given outcome of dice rolls
noncomputable def probability_of_sum (s : ℕ) : ℚ := sorry

-- Define the probability that the side length s is less than 4
noncomputable def probability_s_less_than_4 : ℚ :=
  probability_of_sum 2 + probability_of_sum 3

-- State the theorem to prove the probability is 1/12
theorem area_less_than_perimeter_probability : probability_s_less_than_4 = 1/12 :=
by
  sorry

end area_less_than_perimeter_probability_l287_287005


namespace problem_statement_l287_287719

-- Definitions and assumptions for the problem
variables {n : ℕ} (α : ℝ) (a : ℕ → ℝ) (c : ℕ → ℝ)
variable {y : ℝ}
variable {x : ℝ}

-- Conditions
def conditions (n : ℕ) (α : ℝ) (a c : ℕ → ℝ) (y x : ℝ) : Prop :=
  (n > 1) ∧ (0 < α ∧ α < 2) ∧ (∀ i, 0 < a i) ∧ (∀ i, 0 < c i) ∧ (y > 0) ∧ (x > 0)

-- Function f(y)
noncomputable def f (a c : ℕ → ℝ) (y : ℝ) (α : ℝ) : ℝ :=
  (∑ i in Finset.range n, if a i ≤ y then c i * (a i)^2 else 0) ^ (1/2) +
  (∑ i in Finset.range n, if a i > y then c i * (a i)^α else 0) ^ (1/α)

-- The theorem stating the required inequality
theorem problem_statement
  (hc : conditions n α a c y x)
  (h : x ≥ f a c y α) : f a c x α ≤ 8^(1/α) * x :=
sorry

end problem_statement_l287_287719


namespace sqrt_inequality_l287_287445

theorem sqrt_inequality (x : ℝ) (hx : x > 0) : 
  sqrt x + (1 / sqrt x) ≥ 2 :=
sorry

end sqrt_inequality_l287_287445


namespace length_of_room_l287_287789

/-- Given the width of a room is 3.75 meters and the cost of paving the floor at the rate of $800 per square meter is $16,500,
    prove that the length of the room is 5.5 meters. -/
theorem length_of_room (w : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
  (hw : w = 3.75) (hcost_per_sqm : cost_per_sqm = 800) (htotal_cost : total_cost = 16500) : (16500 / 800) / 3.75 = 5.5 :=
by
  rw [hcost_per_sqm, htotal_cost]
  norm_num
  rw hw
  norm_num
  sorry

end length_of_room_l287_287789


namespace sufficient_not_necessary_condition_l287_287117

-- Definition of the conditions
def Q (x : ℝ) : Prop := x^2 - x - 2 > 0
def P (x a : ℝ) : Prop := |x| > a

-- Main statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, P x a → Q x) → a ≥ 2 :=
by
  sorry

end sufficient_not_necessary_condition_l287_287117


namespace students_both_l287_287435

noncomputable def students_total : ℕ := 32
noncomputable def students_go : ℕ := 18
noncomputable def students_chess : ℕ := 23

theorem students_both : students_go + students_chess - students_total = 9 := by
  sorry

end students_both_l287_287435


namespace tangent_intersection_point_l287_287895

theorem tangent_intersection_point (x : ℝ) :
  (∀ x, 
    (∀ y, (sqrt(x^2 + y^2) = 3 ∧ y = 0) → y = 0) 
    ∧ (∀ y, (sqrt((12 - x)^2 + y^2) = 5 ∧ y = 0) 
    → y = 0)) 
  →  x = 9 / 2 :=
begin
  sorry
end

end tangent_intersection_point_l287_287895


namespace integer_solutions_for_even_ratio_l287_287051

theorem integer_solutions_for_even_ratio (a : ℤ) (h : ∃ k : ℤ, (a = 2 * k * (1011 - k))): 
  a = 1010 ∨ a = 1012 ∨ a = 1008 ∨ a = 1014 ∨ a = 674 ∨ a = 1348 ∨ a = 0 ∨ a = 2022 :=
sorry

end integer_solutions_for_even_ratio_l287_287051


namespace b_minus_c_eq_neg_log_2002_715_l287_287569

noncomputable def a (n : ℕ) : ℝ := 1 / Real.log 2002 / Real.log n

def b : ℝ := a 2 + a 3 + a 5 + a 6

def c : ℝ := a 10 + a 11 + a 13 + a 15

theorem b_minus_c_eq_neg_log_2002_715 : b - c = -Real.log 715 2002 := by
  sorry

end b_minus_c_eq_neg_log_2002_715_l287_287569


namespace longest_line_segment_l287_287743

theorem longest_line_segment (total_length_cm : ℕ) (h : total_length_cm = 3000) :
  ∃ n : ℕ, 2 * (n * (n + 1) / 2) ≤ total_length_cm ∧ n = 54 :=
by
  use 54
  sorry

end longest_line_segment_l287_287743


namespace det_B_squared_minus_3B_l287_287601

theorem det_B_squared_minus_3B (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B = ![![2, 4], ![3, 2]]) : 
  Matrix.det (B * B - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l287_287601


namespace sin_squared_sum_equiv_l287_287387

theorem sin_squared_sum_equiv (α β γ : ℝ) (a b c p r R : ℝ) 
  (h1 : sin α * sin α + sin β * sin β + sin γ * sin γ = (a^2 + b^2 + c^2) / (4 * R^2))
  (h2 : a^2 + b^2 + c^2 = 4 * p^2 - 2 * r^2 - 2 * (a * b + b * c + c * a))
  (h3 : a * b + b * c + c * a = 4 * R * r + p^2) :
  sin α * sin α + sin β * sin β + sin γ * sin γ = (p^2 - r^2 - 4 * r * R) / (2 * R^2) :=
sorry

end sin_squared_sum_equiv_l287_287387


namespace max_right_angle_triangles_l287_287611

open Real

theorem max_right_angle_triangles (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x y : ℝ, x^2 + a^2 * y^2 = a^2) :
  ∃n : ℕ, n = 3 := 
by
  sorry

end max_right_angle_triangles_l287_287611


namespace probability_at_least_one_correct_l287_287171

-- Define the probability of missing a single question
def prob_miss_one : ℚ := 3 / 4

-- Define the probability of missing all six questions
def prob_miss_six : ℚ := prob_miss_one ^ 6

-- Define the probability of getting at least one correct answer
def prob_at_least_one : ℚ := 1 - prob_miss_six

-- The problem statement
theorem probability_at_least_one_correct :
  prob_at_least_one = 3367 / 4096 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_one_correct_l287_287171


namespace rachel_speed_painting_video_time_l287_287752

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end rachel_speed_painting_video_time_l287_287752


namespace erased_number_is_214_l287_287481

def middle_value (a : ℤ) : list ℤ := [a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4]

theorem erased_number_is_214 (a b : ℤ) (h_sum : 8 * a - b = 1703)
  (h_bounds : -4 ≤ b ∧ b ≤ 4) (h_a : a = 213) (h_b : b = 8 * 213 - 1703) : 
  (middle_value a).sum - (a + b) = 1703 ∧ a + b = 214 :=
by { sorry }

end erased_number_is_214_l287_287481


namespace max_value_of_f_on_interval_l287_287262

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f_on_interval :
  ∃ (a : ℝ), (∀ x ∈ set.Icc (0 : ℝ) 1, f x a ≥ -2) ∧ (∀ x ∈ set.Icc (0 : ℝ) 1, -2 ≤ f x a) ∧ (f 1 (-2) = 1) :=
by {
  sorry
}

end max_value_of_f_on_interval_l287_287262


namespace avg_goals_l287_287158

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l287_287158


namespace length_of_DE_l287_287688

theorem length_of_DE (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  [has_dist A ℝ] [has_dist B ℝ] [has_dist C ℝ] [has_dist D ℝ] [has_dist E ℝ] 
  (BC : ℝ) (angle_C : ℝ) (CD : ℝ) (DE : ℝ)
  (hBC : BC = 30)
  (hAngleC : angle_C = π / 4)
  (hDmidpoint : CD = BC / 2)
  (hPerpBisector : DE = CD)
  : DE = 15 :=
sorry

end length_of_DE_l287_287688


namespace carpet_cost_calculation_l287_287009

theorem carpet_cost_calculation
  (length_feet : ℕ)
  (width_feet : ℕ)
  (feet_to_yards : ℕ)
  (cost_per_square_yard : ℕ)
  (h_length : length_feet = 15)
  (h_width : width_feet = 12)
  (h_convert : feet_to_yards = 3)
  (h_cost : cost_per_square_yard = 10) :
  (length_feet / feet_to_yards) *
  (width_feet / feet_to_yards) *
  cost_per_square_yard = 200 := by
  sorry

end carpet_cost_calculation_l287_287009


namespace factorial_divisors_l287_287977

theorem factorial_divisors :
  let n := 20 
  let k2 := 2
  let k5 := 5
  let k10 := 10
  let Legendre (n p : Nat) : Nat :=
    Nat.div n p + 
    Nat.div n (p ^ 2) + 
    Nat.div n (p ^ 3) + 
    Nat.div n (p ^ 4)
  in
  (Legendre n k10 = 4) ∧
  (Legendre n k2 = 18) ∧
  (Legendre n k5 = 4) := by
    sorry

end factorial_divisors_l287_287977


namespace two_parabolas_intersect_528_points_l287_287971

def parabola_focus : Point := (0, 0)

def is_directrix (a b : ℤ) : Prop := a ∈ {-1, 0, 1} ∧ b ∈ {-2, -1, 0, 1, 2}

def parabola (a b: ℤ) (p: Point) : Prop := is_directrix a b ∧ p ≠ parabola_focus ∧ p lies_on_parabola_with_directrix a b

theorem two_parabolas_intersect_528_points :
  (no_three_parabolas_intersect (λ a b, is_directrix a b)) →
  (count_points_on_exactly_two_parabolas 25 (λ a b, parabola a b)) = 528 :=
begin
  sorry
end

end two_parabolas_intersect_528_points_l287_287971


namespace leak_draining_time_l287_287385

theorem leak_draining_time (P_rate : ℝ) (L_rate : ℝ) (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) :
  fill_time_no_leak = 2 ∧ fill_time_with_leak = 7 / 3 ∧ P_rate = 1 / fill_time_no_leak ∧ P_rate - L_rate = 1 / fill_time_with_leak
  → 1 / L_rate = 14 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  have h7 := h6.symm
  unfold P_rate at h7
  unfold L_rate
  sorry

end leak_draining_time_l287_287385


namespace digits_in_4pow20_5pow28_3pow10_l287_287976

theorem digits_in_4pow20_5pow28_3pow10 : 
  Nat.digits 10 (4^20 * 5^28 * 3^10) = 37 := 
by 
  sorry

end digits_in_4pow20_5pow28_3pow10_l287_287976


namespace car_payment_months_l287_287738

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l287_287738


namespace employee_earnings_l287_287278

theorem employee_earnings
  (total_employees : ℕ)
  (laid_off_fraction : ℚ)
  (total_payment : ℚ)
  (total_employees_eq : total_employees = 450)
  (laid_off_fraction_eq : laid_off_fraction = 1 / 3)
  (total_payment_eq : total_payment = 600_000) :
  let remaining_employees := total_employees * (1 - laid_off_fraction)
  in total_payment / remaining_employees = 2000 :=
by
  intros total_employees_eq laid_off_fraction_eq total_payment_eq
  let remaining_employees := 450 * (1 - (1/3))
  have rem_eq : remaining_employees = 300 := sorry
  have pay_eq : 600_000 / 300 = 2000 := sorry
  rw [total_employees_eq, laid_off_fraction_eq, total_payment_eq, rem_eq, pay_eq]
  exact pay_eq

end employee_earnings_l287_287278


namespace tangent_line_intersection_x_axis_l287_287898

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287898


namespace tangent_line_intersection_x_axis_l287_287902

theorem tangent_line_intersection_x_axis :
  let r1 := 3
  let r2 := 5
  let c1 := (0 : ℝ, 0 : ℝ)
  let c2 := (12 : ℝ, 0 : ℝ)
  let x := (9 / 2 : ℝ)
  tangent_intersects_at_x r1 r2 c1 c2 (x : ℝ, 0) → 
  x = 9 / 2 := 
by
  sorry

def tangent_intersects_at_x (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧
  ∃ l : ℝ, (c1.1 - l * √(r1^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c1.2 - l * √(r1^2 / (r1^2 + r2^2))) = p.2 ∧ 
            (c2.1 - l * √(r2^2 / (r1^2 + r2^2))) = p.1 ∧ 
            (c2.2 - l * √(r2^2 / (r1^2 + r2^2))) = p.2

end tangent_line_intersection_x_axis_l287_287902


namespace add_pure_water_l287_287145

theorem add_pure_water (w : ℝ) : 
  (w + 40) ≠ 0 → 
  let orig_salt_amt := 0.25 * 40 in 
  let final_vol := 40 + w in
  let target_concentration := 0.15 in
  (orig_salt_amt / final_vol) = target_concentration → 
  w = 400 / 15 :=
by
  intros h orig_salt_amt final_vol target_concentration h_conc_eq
  sorry

end add_pure_water_l287_287145


namespace francis_violin_count_l287_287085

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l287_287085


namespace tan_x_plus_pi_over_4_eq_neg_2_l287_287149

theorem tan_x_plus_pi_over_4_eq_neg_2 (x : ℝ) 
  (h : cos (3 * π - x) - 3 * cos (x + π / 2) = 0) :
  tan (x + π / 4) = -2 := 
by 
  sorry

end tan_x_plus_pi_over_4_eq_neg_2_l287_287149


namespace negation_of_proposition_l287_287136

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)) ↔ (∃ x : ℝ, x^2 + 2 * x + 3 < 0) :=
by sorry

end negation_of_proposition_l287_287136


namespace area_of_triangle_l287_287788

noncomputable def hyperbola (n : ℝ) (n_gt_1 : n > 1) : set (ℝ × ℝ) :=
  {p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / n - y^2 = 1)}

theorem area_of_triangle (n : ℝ) (n_gt_1 : n > 1)
  (P F1 F2 : ℝ × ℝ)
  (hP_on_hyperbola : P ∈ hyperbola n n_gt_1)
  (hP_condition : dist P F1 + dist P F2 = 2 * real.sqrt (n + 2))
  (hF1F2 : dist F1 F2 = 2 * real.sqrt (n + 1)) :
  real.abs ((P.1 * F1.2 + F1.1 * F2.2 + F2.1 * P.2) -
            (P.2 * F1.1 + F1.2 * F2.1 + F2.2 * P.1)) / 2 = 1 :=
by
  sorry

end area_of_triangle_l287_287788


namespace work_duration_l287_287847

theorem work_duration (W : ℕ) 
  (p q : ℕ) 
  (hp : p = 80) 
  (hq : q = 48) 
  (hp_work : W / 80) 
  (hq_work : W / 48) 
  (days_p : 16) 
  (work_p : days_p * (W / p) = W / 5)
  (combined_rate : (W / p) + (W / q))
  (common_denom : (W / 80) = 3 * W / 240)
  (common_denom2 : (W / 48) = 5 * W / 240)
  (total_combined_rate : 3 * W / 240 + 5 * W / 240 = W / 30)
  (remaining_work : W - W / 5)
  (needed_days : 4 * 6) :
  (16 + 24 = 40) :=
by
  sorry

end work_duration_l287_287847


namespace divisors_large_than_8_fact_count_l287_287632

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem divisors_large_than_8_fact_count :
  let n := 9
  let factorial_n := factorial n
  let factorial_n_minus_1 := factorial (n - 1)
  ∃ (num_divisors : ℕ), num_divisors = 8 ∧
    (∀ d, d ∣ factorial_n → d > factorial_n_minus_1 ↔ ∃ k, k ∣ factorial_n ∧ k < 9) :=
by
  sorry

end divisors_large_than_8_fact_count_l287_287632


namespace f_f_neg3_l287_287615

def f (x : ℝ) : ℝ :=
  if x < 3 then |x + 3| else 4^(x + 1)

theorem f_f_neg3 : f (f (-3)) = 3 :=
by
  sorry

end f_f_neg3_l287_287615


namespace vertical_asymptote_condition_l287_287501

-- Defined the function g(x)
def g (x c : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - 4*x + 3)

-- Statement to prove: g(x) has exactly one vertical asymptote if and only if c = 0 or c = 2
theorem vertical_asymptote_condition (c : ℝ) :
  (∃ x : ℝ, (x^2 - 4*x + 3 = 0 ∧ (x^2 - 3*x + c) / (x^2 - 4*x + 3) ≠ 0)) ↔ (c = 0 ∨ c = 2) :=
by
  sorry

end vertical_asymptote_condition_l287_287501


namespace part1_solution_part2_solution_l287_287485

noncomputable def part1 : ℝ :=
  sin (real.pi / 3) - real.sqrt 3 * cos (real.pi / 3) + (1 / 2) * tan (real.pi / 4)

theorem part1_solution : part1 = 1 / 2 := by
  sorry

variables (a c : ℝ)
def mean_proportional (a c : ℝ) : Prop :=
  ∃ b : ℝ, b * b = a * c

theorem part2_solution (h1 : a = 9) (h2 : c = 4) : mean_proportional a c := by
  use 6
  sorry

end part1_solution_part2_solution_l287_287485


namespace smallest_prime_perimeter_l287_287075

def is_prime (n : ℕ) := Nat.Prime n
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a
def is_scalene (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ a ≥ 5
  ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
by
  sorry

end smallest_prime_perimeter_l287_287075


namespace set_union_A_B_l287_287107

-- Define the set A
def A : Set ℝ := { x | log 2 x < 1 }

-- Define the set B
def B : Set ℝ := { x | x > 1 }

-- The theorem that we want to prove
theorem set_union_A_B : (A ∪ B) = { x : ℝ | 0 < x } := 
  sorry

end set_union_A_B_l287_287107


namespace range_of_c_over_a_l287_287150

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + 2 * b + c = 0) :
    -3 < c / a ∧ c / a < -(1 / 3) := 
sorry

end range_of_c_over_a_l287_287150


namespace find_sin_BAC_l287_287187

theorem find_sin_BAC
  (A B C M : Type) 
  (hC : angle(C) = π / 4) 
  (h_area : area(⟨A, B, C⟩) = 3) 
  (h_midpoint : midpoint(M) = ⟨B, C⟩) 
  (h_AM : distance(A, M) = sqrt(5)) 
  (h_AC_GT_BC : distance(A, C) > distance(B, C)) :
  sin (angle BAC) = 2 * sqrt(5) / 5 := 
sorry

end find_sin_BAC_l287_287187


namespace process_repetition_count_l287_287920

noncomputable def numberOfRepetitions (H H' : Float) : Nat :=
  (Int.ofFloat (Float.log (512 / H') / Float.log 0.8)).toNat

theorem process_repetition_count :
  numberOfRepetitions 1249.9999999999998 512 = 4 := by
  sorry

end process_repetition_count_l287_287920


namespace largest_two_digit_number_from_set_l287_287070

theorem largest_two_digit_number_from_set (x y z : ℕ) (h1 : x ∈ {1, 2, 4}) (h2 : y ∈ {1, 2, 4}) (h3 : z ∈ {1, 2, 4}) (hxne : x ≠ y) (hyne : y ≠ z) (hzne : x ≠ z) : 
  (10 * max (max x y) z + max (min x y) (min y z)) = 42 :=
by 
  sorry

end largest_two_digit_number_from_set_l287_287070


namespace geometric_sequence_tenth_term_l287_287494

theorem geometric_sequence_tenth_term :
  let a := 5
  let r := 3 / 2
  let a_n (n : ℕ) := a * r ^ (n - 1)
  a_n 10 = 98415 / 512 :=
by
  sorry

end geometric_sequence_tenth_term_l287_287494


namespace scientific_notation_3050000_l287_287026

def scientific_notation (n : ℕ) : String :=
  "3.05 × 10^6"

theorem scientific_notation_3050000 :
  scientific_notation 3050000 = "3.05 × 10^6" :=
by
  sorry

end scientific_notation_3050000_l287_287026


namespace find_f_prime_at_2_l287_287346

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287346


namespace arithmetic_sequence_common_difference_l287_287199

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_a2 : a 2 = 3)
  (h_a7 : a 7 = 13) : 
  ∃ d, ∀ n, a n = a 1 + (n - 1) * d ∧ d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l287_287199


namespace erased_number_is_214_l287_287480

def middle_value (a : ℤ) : list ℤ := [a-4, a-3, a-2, a-1, a, a+1, a+2, a+3, a+4]

theorem erased_number_is_214 (a b : ℤ) (h_sum : 8 * a - b = 1703)
  (h_bounds : -4 ≤ b ∧ b ≤ 4) (h_a : a = 213) (h_b : b = 8 * 213 - 1703) : 
  (middle_value a).sum - (a + b) = 1703 ∧ a + b = 214 :=
by { sorry }

end erased_number_is_214_l287_287480


namespace determine_a_l287_287092

noncomputable def f (a x : ℝ) : ℝ := a * real.log x + x^2

theorem determine_a {a : ℝ} (h : 0 < a) (H : ∀ {x1 x2 : ℝ}, x1 ≠ x2 → 0 < x1 → 0 < x2 → f a x1 - f a x2 > 2) :
  1 ≤ a :=
begin
  sorry
end

end determine_a_l287_287092


namespace tangent_line_intersects_x_axis_at_9_div_2_l287_287906

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (12, 0)
def radius2 : ℝ := 5

-- Define the target x-coordinate of the tangent line intersection
def target_x : ℝ := 9 / 2

-- Lean theorem to prove the question
theorem tangent_line_intersects_x_axis_at_9_div_2 :
  ∃ x : ℝ, (line_tangent_to_two_circles center1 radius1 center2 radius2 x) ∧ x = target_x :=
sorry

-- Hypothetical function to check tangent property
noncomputable def line_tangent_to_two_circles
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (x : ℝ) : Prop :=
  -- Dummy implementation of the tangent condition
  sorry

end tangent_line_intersects_x_axis_at_9_div_2_l287_287906


namespace seq_irrational_term_l287_287824

noncomputable def seq (a0: ℝ) (n : ℕ) : ℝ :=
  Nat.rec a0 (λ n an, Real.sqrt (an + 1)) n

theorem seq_irrational_term (a0 : ℝ) (h : a0 > 0) : ∃ n : ℕ, ¬ (seq a0 n).is_rat := 
sorry

end seq_irrational_term_l287_287824


namespace sqrt_int_eq_l287_287241

theorem sqrt_int_eq (n : ℕ) (h : n > 0) : 
  floor (Real.sqrt n + Real.sqrt (n + 1)) = floor (Real.sqrt (4 * n + 2)) :=
sorry

end sqrt_int_eq_l287_287241


namespace five_pow_neg_two_neg_one_fourth_pow_neg_two_l287_287486

theorem five_pow_neg_two : 5 ^ -2 = 1 / 25 :=
by sorry

theorem neg_one_fourth_pow_neg_two : (- (1 / 4)) ^ -2 = 16 :=
by sorry

end five_pow_neg_two_neg_one_fourth_pow_neg_two_l287_287486


namespace length_of_diagonal_l287_287233

open Real

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, -a^2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (-a, -a^2)
noncomputable def C (a : ℝ) : ℝ × ℝ := (a, -a^2)
def O : ℝ × ℝ := (0, 0)

noncomputable def is_square (A B O C : ℝ × ℝ) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O C ∧ dist O C = dist C A

theorem length_of_diagonal (a : ℝ) (h_square : is_square (A a) (B a) O (C a)) : 
  dist (A a) (C a) = 2 * abs a :=
sorry

end length_of_diagonal_l287_287233


namespace champion_is_D_l287_287012

-- Define the competitors
inductive Competitor
| A | B | C | D

open Competitor

-- Define the predictions as boolean (propositional) variables
def pred1 (champ : Competitor) : Prop := champ = A ∨ champ = B
def pred2 (champ : Competitor) : Prop := champ = A
def pred3 (champ : Competitor) : Prop := champ = D
def pred4 (champ : Competitor) : Prop := ¬ (champ = B ∨ champ = C)

-- State that exactly two of the four predictions are correct
def exactly_two_correct (champ : Competitor) : Prop :=
(p1 : pred1 champ).toBool + (p2 : pred2 champ).toBool + (p3 : pred3 champ).toBool + (p4 : pred4 champ).toBool = 2

-- The problem to prove
theorem champion_is_D : ∃ champ : Competitor, champ = D ∧ exactly_two_correct champ :=
sorry

end champion_is_D_l287_287012


namespace max_value_at_one_l287_287315

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l287_287315


namespace find_f_prime_at_2_l287_287339

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l287_287339


namespace find_f_prime_at_2_l287_287343

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l287_287343


namespace Julie_total_earnings_correct_l287_287700

def Julie_landscaping_rate_h : ℕ → ℕ
  | 1 => 4
  | 2 => 6
  | 3 => 8
  | 4 => 10
  | 5 => 10
  | 6 => 15
  | 7 => 12
  | _ => 0

def Julie_hours_sep : ℕ → ℕ
  | 1 => 10
  | 2 => 15
  | 3 => 2
  | 4 => 1
  | 5 => 5
  | 6 => 5
  | 7 => 5
  | _ => 0

def Julie_earnings_sep := 
  (Julie_landscaping_rate_h 1 * Julie_hours_sep 1) + 
  (Julie_landscaping_rate_h 2 * Julie_hours_sep 2) + 
  (Julie_landscaping_rate_h 3 * Julie_hours_sep 3) + 
  (Julie_landscaping_rate_h 4 * Julie_hours_sep 4) + 
  (Julie_landscaping_rate_h 5 * Julie_hours_sep 5) + 
  (Julie_landscaping_rate_h 6 * Julie_hours_sep 6) + 
  (Julie_landscaping_rate_h 7 * Julie_hours_sep 7)

def Julie_hours_oct : ℕ → ℕ 
  | n => 3 * Julie_hours_sep n / 2

def Julie_earnings_oct := 
  (Julie_landscaping_rate_h 1 * Julie_hours_oct 1) + 
  (Julie_landscaping_rate_h 2 * Julie_hours_oct 2) + 
  (Julie_landscaping_rate_h 3 * Julie_hours_oct 3) + 
  (Julie_landscaping_rate_h 4 * Julie_hours_oct 4) + 
  (Julie_landscaping_rate_h 5 * Julie_hours_oct 5) + 
  (Julie_landscaping_rate_h 6 * Julie_hours_oct 6) + 
  (Julie_landscaping_rate_h 7 * Julie_hours_oct 7)

def Julie_total_earnings := Julie_earnings_sep + Julie_earnings_oct

theorem Julie_total_earnings_correct : Julie_total_earnings = 852.5 := by sorry

end Julie_total_earnings_correct_l287_287700


namespace race_winner_distance_l287_287169

theorem race_winner_distance (meters_per_kilometer : ℕ := 1000) 
  (time_A time_B : ℕ) (hA : time_A = 235) (hB : time_B = 250) : 
  let v_A := meters_per_kilometer / time_A in
  let v_B := meters_per_kilometer / time_B in
  v_B * time_A = 940 :=
by
  sorry -- Proof is to be provided.

end race_winner_distance_l287_287169


namespace erased_number_is_214_l287_287474

theorem erased_number_is_214 {a b : ℤ} 
  (h1 : 9 * a = sum (a - 4 :: a - 3 :: a - 2 :: a - 1 :: a :: a + 1 :: a + 2 :: a + 3 :: a + 4 :: []))
  (h2 : ∑ n in {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4} \ erase_val (a + b) {a - 4, a - 3, a - 2, a - 1, a, a + 1, a + 2, a + 3, a + 4}, n = 1703)
  (h3 : -4 ≤ b ∧ b ≤ 4)
  : a + b = 214 :=
begin
  -- proof to be filled in
  sorry,
end

end erased_number_is_214_l287_287474


namespace simplify_expression_l287_287804

open Real

-- Assuming lg refers to the common logarithm log base 10
noncomputable def problem_expression : ℝ :=
  log 4 + 2 * log 5 + 4^(-1/2:ℝ)

theorem simplify_expression : problem_expression = 5 / 2 :=
by
  -- Placeholder proof, actual steps not required
  sorry

end simplify_expression_l287_287804


namespace problem1_problem2_l287_287583

open Nat

-- Given sequence condition
def sequence_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (finite n ∧ n > 0) → (finset.range n).sum a = n - a n

-- Problem (1): Prove {a_n - 1} is a geometric sequence
theorem problem1 (a : ℕ → ℝ) (h : ∀ n : ℕ, sequence_condition a n) : 
  ∃ (r : ℝ), ∃ (b : ℝ), ∀ n : ℕ, n > 0 → a n - 1 = b * (r ^ n) :=
begin
  sorry
end

-- Problem (2): Find the range of t based on the given condition
theorem problem2 (a : ℕ → ℝ) (t : ℝ) (h : ∀ n : ℕ, n > 0 → n * (1 - a n) ≤ t) :
  t ∈ set.Ici (1 / 2) :=
begin
  sorry
end

end problem1_problem2_l287_287583


namespace limit_difference_quotient_l287_287161

noncomputable def f (x : ℝ) : ℝ := x^2

theorem limit_difference_quotient : 
  (filter.tendsto (λ Δx : ℝ, (f (1 + Δx) - f 1) / Δx) (nhds 0) (nhds 2)) := 
by
  sorry

end limit_difference_quotient_l287_287161


namespace valid_numbers_correct_l287_287979

-- Define the conditions
def valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  n % 13 = 0 ∧
  let a := n / 100,
      b := (n / 10) % 10,
      c := n % 10 in
    (2 * b = a + c)

-- Define the set of valid numbers
def valid_numbers_set : set ℕ := {n | valid_number n}

-- Define the expected set of valid numbers
def expected_set : set ℕ := {741, 234, 975, 468}

-- Prove that the valid numbers set is exactly the expected set
theorem valid_numbers_correct : valid_numbers_set = expected_set :=
by
  sorry

end valid_numbers_correct_l287_287979


namespace variance_comparison_l287_287230

theorem variance_comparison (s s1 : ℝ) (hx : ∃ ε > 0, ε = s) (hs1 : s1 = 0) : s > s1 :=
by
  obtain ⟨ε, ε_pos, hes⟩ := hx
  rw [hes, hs1]
  exact ε_pos

end variance_comparison_l287_287230


namespace triangle_angles_l287_287683

/-
In triangle ABC, angle bisectors of angles ABC and ACB intersect sides AC and AB at points
D and E, respectively. Given:
    (1) ∠BDE = 24°
    (2) ∠CED = 18°
Prove:
    (3) ∠ABC = 12°
    (4) ∠ACB = 72°
    (5) ∠BAC = 96°
-/
theorem triangle_angles (A B C D E : Type) [triangle A B C] 
  (hD : is_angle_bisector A B C D) 
  (hE : is_angle_bisector A C B E) 
  (angle_BDE : angle D B E = 24)
  (angle_CED : angle C E D = 18) 
  : (angle B A C = 12) ∧ (angle C A B = 72) ∧ (angle A B C = 96) :=
sorry

end triangle_angles_l287_287683


namespace smallest_non_palindrome_product_l287_287544

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def smallest_three_digit_palindrome : ℕ :=
  (List.range' 100 900).filter (λ n, is_palindrome n) |>.min

theorem smallest_non_palindrome_product :
  ∃ n : ℕ, n = 606 ∧ is_palindrome n ∧ ¬is_palindrome (111 * n) :=
by {
  use 606,
  sorry
}

end smallest_non_palindrome_product_l287_287544


namespace terminal_side_in_second_or_fourth_l287_287640

/-- If sin(alpha) * cos(alpha) < 0 and sin(alpha) * tan(alpha) > 0, 
then the terminal side of the angle alpha / 2 lies in the second or fourth quadrant. -/
theorem terminal_side_in_second_or_fourth (α : ℝ)
    (h1 : sin α * cos α < 0)
    (h2 : sin α * tan α > 0) :
    ∃ k : ℤ, (k % 2 = 0 ∧ π / 2 + k * π < α / 2 ∧ α / 2 < π * (k + 1))
          ∨ (k % 2 = 1 ∧ π + k * π < α / 2 ∧ α / 2 < π * (k + 1)) := sorry

end terminal_side_in_second_or_fourth_l287_287640


namespace tan_double_angle_l287_287607

theorem tan_double_angle {θ : ℝ} (h₁ : ∀ θ, terminalSide θ = (λ x, x ≥ 0 → y = (1 / 2) * x)) : 
  tan(2 * θ) = 4 / 3 :=
by 
  -- using the definition of tan
  sorry

end tan_double_angle_l287_287607


namespace length_MD_l287_287766

-- Definitions from conditions
def A : Point := (0, 0)  -- Assume coordinates for simplicity
def B : Point := (10, 0)  -- Length of AB is 10
def E : Point := (5, 0)  -- Midpoint of AB and the center of the semicircle
def M : Point := (5, 5)  -- Midpoint of arc AB
def D : Point := (0, -10)  -- One corner of the square ABCD

-- The length of AB is 10 units
def AB_length : ℝ := 10

-- Define function to calculate Euclidean distance between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((Q.fst - P.fst)^2 + (Q.snd - P.snd)^2)

-- Question with answer: What is the length of segment MD?
theorem length_MD : distance M D = 5 * real.sqrt 5 :=
by
  -- Instead of providing the proof, use 'sorry'
  sorry


end length_MD_l287_287766


namespace repeating_decimal_fraction_l287_287515

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end repeating_decimal_fraction_l287_287515


namespace modified_determinant_l287_287600

variables {p q r s : ℝ}

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- Given conditions
axiom original_determinant : determinant p q r s = 7

-- Theorem to prove
theorem modified_determinant : determinant (p + 2 * r) (q + 2 * s) r s = 7 :=
by {
  have h1 : determinant p q r s = 7 := original_determinant,
  sorry
}

end modified_determinant_l287_287600


namespace calculate_water_added_l287_287409

def original_volume : ℝ := 40
def initial_percentage_alcohol : ℝ := 0.05
def added_alcohol : ℝ := 6.5
def final_percentage_alcohol : ℝ := 0.17

def initial_alcohol_content : ℝ := initial_percentage_alcohol * original_volume
def final_alcohol_content : ℝ := initial_alcohol_content + added_alcohol

def water_added (W : ℝ) :=
  W = (final_alcohol_content / final_percentage_alcohol) - original_volume - added_alcohol

theorem calculate_water_added : ∃ W : ℝ, water_added W ∧ W = 3.5 :=
by
  use 3.5
  unfold water_added
  simp only [initial_alcohol_content, final_alcohol_content]
  have h1 : initial_alcohol_content = 2 := by norm_num [initial_alcohol_content, initial_percentage_alcohol, original_volume]
  have h2 : final_alcohol_content = 8.5 := by norm_num [final_alcohol_content, initial_alcohol_content, added_alcohol]
  have h3 : (8.5 / 0.17) = 50 := by norm_num
  rw [h1, h2, h3]
  norm_num

end calculate_water_added_l287_287409


namespace derivative_at_2_l287_287368

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l287_287368


namespace polynomial_inequality_l287_287247

variable (n : ℕ)
variable (a : Fin n → ℝ)
variable (r : Fin n → ℝ)

theorem polynomial_inequality (h_poly : ∀ x : ℝ, (∏ i in Finset.finRange n, (x + r i)) = x^n + (a (Fin.ofNat' 1)) * x^(n-1) + (a (Fin.ofNat' 2)) * x^(n-2) + ∑ i in Finset.finRange (n-3), (a (Fin.ofNat' (i + 3))) * x^i) :
  (n - 1) * (a (Fin.ofNat' 1))^2 ≥ 2 * n * (a (Fin.ofNat' 2)) := sorry

end polynomial_inequality_l287_287247


namespace select_non_overlapping_circles_l287_287403

open Finset

theorem select_non_overlapping_circles (N : ℕ) (circles : Finset ℝ) 
  (h_len : circles.card = N) (h_area : (∑ circle in circles, circle^2 * π) = 1) :
  ∃ (non_overlapping : Finset ℝ), non_overlapping ⊆ circles ∧
    (∑ circle in non_overlapping, circle^2 * π) > 1 / 9 := by
  sorry

end select_non_overlapping_circles_l287_287403


namespace price_of_gift_l287_287731

def lisa_savings : ℕ := 1200
def mother_contribution (savings : ℕ) : ℕ := 3 * savings / 5
def brother_contribution (mother_contrib : ℕ) : ℕ := 2 * mother_contrib
def additional_amount : ℕ := 400

theorem price_of_gift :
  let total_savings := lisa_savings + mother_contribution lisa_savings + brother_contribution (mother_contribution lisa_savings) in
  total_savings + additional_amount = 3760 :=
by
  sorry

end price_of_gift_l287_287731


namespace fg_of_two_l287_287642

theorem fg_of_two : (f : ℝ → ℝ) = (λ x, 5 - 4 * x) → (g : ℝ → ℝ) = (λ y, y^2 + 2) → f (g 2) = -19 :=
by
  sorry

end fg_of_two_l287_287642


namespace notebooks_have_50_pages_l287_287693

theorem notebooks_have_50_pages (notebooks : ℕ) (total_dollars : ℕ) (page_cost_cents : ℕ) 
  (total_cents : ℕ) (total_pages : ℕ) (pages_per_notebook : ℕ)
  (h1 : notebooks = 2) 
  (h2 : total_dollars = 5) 
  (h3 : page_cost_cents = 5) 
  (h4 : total_cents = total_dollars * 100) 
  (h5 : total_pages = total_cents / page_cost_cents) 
  (h6 : pages_per_notebook = total_pages / notebooks) 
  : pages_per_notebook = 50 :=
by
  sorry

end notebooks_have_50_pages_l287_287693


namespace max_leap_years_l287_287952

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) :
  leap_interval = 5 ∧ total_years = 200 → (years = total_years / leap_interval) :=
by
  sorry

end max_leap_years_l287_287952


namespace number_of_even_integers_between_25_div_4_and_47_div_2_l287_287142

def floor (x : ℝ) : ℤ := Int.ofNat ⌊x⌋₊

def count_even_integers (a b : ℝ) : ℕ :=
  let lower := floor a + 1
  let upper := floor b
  (Nat.antidiag upper).filter (fun n => n % 2 = 0).length

theorem number_of_even_integers_between_25_div_4_and_47_div_2 :
  count_even_integers (25 / 4) (47 / 2) = 8 :=
by
  simp [count_even_integers, floor]
  sorry

end number_of_even_integers_between_25_div_4_and_47_div_2_l287_287142


namespace min_fraction_sum_l287_287197

/-- Given W, X, Y, and Z are distinct digits from the set {1, 2, 3, 4, 5, 6, 7, 8}, 
prove that the minimum value of (W / X) + (Y / Z) is 15 / 56. -/
theorem min_fraction_sum (W X Y Z : ℕ) (h : {W, X, Y, Z} ⊆ finset.range 9 \ {0}) (h_distinct : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) : 
  (↑W / ↑X + ↑Y / ↑Z) ≥ (15 / 56 : ℚ) :=
sorry

end min_fraction_sum_l287_287197


namespace find_x_l287_287108

open BigOperators

theorem find_x (ε : ℝ) (x : ℝ) (h : ℝ) : 
  3^x + 3^x + 3^x = 243 → x = 4 := 
by
  sorry

end find_x_l287_287108


namespace three_kids_savings_l287_287251

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end three_kids_savings_l287_287251


namespace coefficient_x3_in_binom_expansion_l287_287066

theorem coefficient_x3_in_binom_expansion : 
  let binom_coeff := λ (n k : ℕ), Nat.choose n k in
  binom_coeff 7 4 = 35 :=
by
  sorry

end coefficient_x3_in_binom_expansion_l287_287066


namespace number_of_five_digit_integers_with_ten_thousand_digit_3_l287_287630

theorem number_of_five_digit_integers_with_ten_thousand_digit_3 : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ digit 10000 n = 3) :=
begin
  -- Let d3 set be the set of nearest valid integers (30000 to 39999)
  let d3 := {n : ℕ | 30000 ≤ n \land n ≤ 39999 \land digit 10000 n = 3 },
  -- prove cardinality of this set
  have hd3 : Fintype.card d3 = 10000, sorry
end

end number_of_five_digit_integers_with_ten_thousand_digit_3_l287_287630


namespace overlapping_black_area_l287_287291

def chessboard_area := 64
def overlap_area : ℝ := chessboard_area * (Real.sqrt 2 - 1)

-- Assuming each chessboard is colored such that half of the squares are black and half are white
def sections_area : ℝ := overlap_area / 4

-- Calculating the total area of black-black overlapping squares
def total_black_overlap_area : ℝ := 2 * sections_area

theorem overlapping_black_area :
  total_black_overlap_area = 32 * (Real.sqrt 2 - 1) := 
by
  sorry

end overlapping_black_area_l287_287291


namespace bookstore_discount_l287_287864

theorem bookstore_discount (P MP price_paid : ℝ) (h1 : MP = 0.80 * P) (h2 : price_paid = 0.60 * MP) :
  price_paid / P = 0.48 :=
by
  sorry

end bookstore_discount_l287_287864


namespace triangle_abe_angle_l287_287798

theorem triangle_abe_angle
  (A B C D E F : Point)
  (square : Square A B C D)
  (inside_square : Inside E square)
  (outside_square : Outside F square)
  (congruent_triangles : Congruent (Triangle A B E) (Triangle B C F))
  (eq_ef_side_square : Distance E F = side square)
  (right_angle_bfd : Angle B F D = 90) :
  Angle A B E = 15 :=
sorry

end triangle_abe_angle_l287_287798


namespace ratio_green_red_l287_287662

-- Define the variables for green marbles, red marbles, yellow marbles, and other marbles
def G : ℕ := 60 -- Derived from the problem conditions
def R : ℕ := 20
def Y : ℕ := nat.ceil (0.2 * G)
def O : ℕ := 88

-- Condition for total number of marbles
axiom total_marbles : 3 * G = G + Y + R + O

-- Prove the ratio of green marbles to red marbles is 3:1
theorem ratio_green_red : G / R = 3 / 1 :=
by sorry

end ratio_green_red_l287_287662


namespace max_value_g_of_symmetric_axis_l287_287131

open Real

theorem max_value_g_of_symmetric_axis (a : ℝ)
  (h_symm : ∃ t : ℝ, ∀ x, sin x + a * cos x = sqrt (a^2 + 1) * sin (x + t) ∧ tan t = a)
  (h_axis : ∃ x, x = (5 / 3) * π ∧ ∀ x, sin x + a * cos x = sin ((5 / 3) * π) + a * cos ((5 / 3) * π)) :
  ∃ M, M = (2 * sqrt 3) / 3 ∧ (∀ x, g x = a * sin x + cos x ≤ M) :=
sorry

end max_value_g_of_symmetric_axis_l287_287131


namespace at_least_2_boys_and_1_girl_l287_287780

noncomputable def probability_at_least_2_boys_and_1_girl (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_with_0_boys := Nat.choose girls committee_size
  let ways_with_1_boy := Nat.choose boys 1 * Nat.choose girls (committee_size - 1)
  let ways_with_fewer_than_2_boys := ways_with_0_boys + ways_with_1_boy
  1 - (ways_with_fewer_than_2_boys / total_ways)

theorem at_least_2_boys_and_1_girl :
  probability_at_least_2_boys_and_1_girl 32 14 18 6 = 767676 / 906192 :=
by
  sorry

end at_least_2_boys_and_1_girl_l287_287780


namespace mod_1234_eq_5_l287_287827

theorem mod_1234_eq_5 : ∃ n : ℤ, 0 ≤ n ∧ n < 7 ∧ (-1234 ≡ n [MOD 7]) :=
by
  use 5
  split
  repeat { split }
  sorry -- proof that 0 ≤ 5
  sorry -- proof that 5 < 7
  sorry -- proof that -1234 ≡ 5 [MOD 7]

end mod_1234_eq_5_l287_287827


namespace francis_violin_count_l287_287084

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l287_287084


namespace intersection_eq_l287_287626

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℝ := {x : ℝ | x > 2 ∨ x < -1}

theorem intersection_eq : (setA ∩ setB) = {x : ℝ | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l287_287626


namespace slower_ball_speed_l287_287289

open Real

variables (v u C : ℝ)

theorem slower_ball_speed :
  (20 * (v - u) = C) → (4 * (v + u) = C) → ((v + u) * 3 = 75) → u = 10 :=
by
  intros h1 h2 h3
  sorry

end slower_ball_speed_l287_287289


namespace inequalities_correct_l287_287087

open Real

-- Define the variables and conditions
variables (a b : ℝ) (h_pos : a * b > 0)

-- Define the inequalities
def inequality_B : Prop := 2 * (a^2 + b^2) >= (a + b)^2
def inequality_C : Prop := (b / a) + (a / b) >= 2
def inequality_D : Prop := (a + 1 / a) * (b + 1 / b) >= 4

-- The Lean statement
theorem inequalities_correct : inequality_B a b h_pos ∧ inequality_C a b h_pos ∧ inequality_D a b h_pos :=
by
  sorry

end inequalities_correct_l287_287087


namespace no_perfect_square_nine_digit_ending_in_5_l287_287240

theorem no_perfect_square_nine_digit_ending_in_5 :
  ∀ n : ℕ, (n > 0 ∧ n < 10^9 ∧ (n % 10 = 5) ∧ (list.nodup n.digits 10) ∧ (∀ d ∈ n.digits 10, d ∈ [1,2,3,4,5,6,7,8,9] )) →
  ¬ (∃ m : ℕ, n = m^2) :=
begin
  sorry
end

end no_perfect_square_nine_digit_ending_in_5_l287_287240


namespace part_I_equation_of_line_l_part_II_equation_of_line_m_l287_287120

def point_a : ℝ × ℝ := ⟨1, -3⟩
def line1 := λ (x y : ℝ), 2 * x - y + 4 = 0

noncomputable def line_l := λ (x y : ℝ), 2 * x - y - 5 = 0
noncomputable def line_m := λ (x y : ℝ), x + 2 * y - 6 = 0

-- The theorem for part (I)
theorem part_I_equation_of_line_l
  (A : ℝ × ℝ)
  (hl : ∃ m b, ∀ (x y : ℝ), y = m * x + b ↔ line_l x y)
  (hp : A = point_a) :
  line_l 1 (-3) :=
sorry

-- The theorem for part (II)
theorem part_II_equation_of_line_m
  (A : ℝ × ℝ)
  (hl : ∃ m b, ∀ (x y : ℝ), y = m * x + b ↔ line_l x y)
  (hm : ∃ m b, ∀ (x y : ℝ), y = m * x + b ↔ line_m x y)
  (perpendicular : ∀ m₁ m₂ : ℝ, (y : ℝ) = m₁ * x ↔ y = - (1 / m₂) * x)
  (y_intercept_m : ∀ b, b = 3 → line_m 0 b)
  (hp : A = point_a) :
  line_m 0 3 :=
sorry

end part_I_equation_of_line_l_part_II_equation_of_line_m_l287_287120


namespace tangent_circles_x_intersect_l287_287871

open Real

theorem tangent_circles_x_intersect :
  ∀ (x : ℝ), (R1 R2 : ℝ) (C1 C2 : PPoint),
  C1 = PPoint.mk 0 0 ∧ R1 = 3 ∧ C2 = PPoint.mk 12 0 ∧ R2 = 5 ∧
  (tangent_intersects_x x C1 R1 C2 R2 (0, 0)) →
  x = 18 :=
by
  intros x R1 R2 C1 C2 h,
  sorry

end tangent_circles_x_intersect_l287_287871


namespace find_a9_a10_l287_287180

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem find_a9_a10 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 2) :
  a 9 + a 10 = 16 := 
sorry

end find_a9_a10_l287_287180


namespace intersection_empty_l287_287726

def setA : Set ℝ := { x | x^2 - 2 * x > 0 }
def setB : Set ℝ := { x | |x + 1| < 0 }

theorem intersection_empty : setA ∩ setB = ∅ :=
by
  sorry

end intersection_empty_l287_287726


namespace rectangles_in_square_chain_l287_287431

theorem rectangles_in_square_chain (n : ℕ) (h : 4 ≤ n^2) :
  ∃ (chain : list (ℕ × ℕ)), chain.length = 2 * n ∧ 
  (∀ (i j : ℕ) (hi : i < chain.length) (hj : j < chain.length), 
     i ≤ j → (fst (chain.nth_le i hi) ≤ fst (chain.nth_le j hj) ∧ 
               snd (chain.nth_le i hi) ≤ snd (chain.nth_le j hj))) :=
sorry

end rectangles_in_square_chain_l287_287431


namespace polar_line_segment_l287_287978

theorem polar_line_segment :
  {p : ℝ × ℝ | ∃ r θ, p = (r * real.cos θ, r * real.sin θ) ∧ θ = π / 4 ∧ 0 ≤ r ∧ r ≤ 5} = 
  {p : ℝ × ℝ | ∃ r, 0 ≤ r ∧ r ≤ 5 ∧ p = (r * real.cos (π / 4), r * real.sin (π / 4))} :=
by sorry

end polar_line_segment_l287_287978


namespace max_sum_n_l287_287606

def a_n (n : ℕ) : ℤ := 97 - 3 * n

def b_n (n : ℕ) : ℤ := a_n n * a_n (n + 1) * a_n (n + 2)

def sum_n (n : ℕ) : ℤ := (List.range n).sum_by (λ k, b_n k)

theorem max_sum_n : (∀ m : ℕ, m ≤ 32 → sum_n m ≤ sum_n 32) ∧ (∃ m : ℕ, m = 32) :=
sorry

end max_sum_n_l287_287606


namespace car_average_speed_l287_287806

theorem car_average_speed (s1 s2 : ℕ) (h1 : s1 = 98) (h2 : s2 = 60) : 
  let total_distance := s1 + s2 in
  let total_time := 2 in 
  let average_speed := total_distance / total_time in
  average_speed = 79 := 
by
  sorry

end car_average_speed_l287_287806


namespace digits_for_words_l287_287855

-- Definitions of digits for the given words
def digits (T W E N Y : ℕ) := 0 ≤ T ∧ T ≤ 9 ∧
                              0 ≤ W ∧ W ≤ 9 ∧
                              0 ≤ E ∧ E ≤ 9 ∧
                              0 ≤ N ∧ N ≤ 9 ∧
                              0 ≤ Y ∧ Y ≤ 9

-- The main proof statement: proving the digits represent correctly in tallying valid sums
theorem digits_for_words (T W E N Y : ℕ) :
  digits T W E N Y → 
  (T + W + E + N + T + Y + T + W + E + N + T + Y + 
   T + W + E + N + T + Y + T + E + N + T + E + N = 80) →
  ∃ T W E N Y, T ≠ W ∧ W ≠ E ∧ E ≠ N ∧ N ≠ Y :=
by {
  intros, 
  sorry
}

end digits_for_words_l287_287855


namespace find_f_prime_at_two_l287_287331

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287331


namespace find_f_prime_at_two_l287_287326

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l287_287326


namespace total_pieces_of_bread_correct_l287_287819

-- Define the constants for the number of bread pieces needed per type of sandwich
def pieces_per_regular_sandwich : ℕ := 2
def pieces_per_double_meat_sandwich : ℕ := 3

-- Define the quantities of each type of sandwich
def regular_sandwiches : ℕ := 14
def double_meat_sandwiches : ℕ := 12

-- Define the total pieces of bread calculation
def total_pieces_of_bread : ℕ := pieces_per_regular_sandwich * regular_sandwiches + pieces_per_double_meat_sandwich * double_meat_sandwiches

-- State the theorem
theorem total_pieces_of_bread_correct : total_pieces_of_bread = 64 :=
by
  -- Proof goes here (using sorry for now)
  sorry

end total_pieces_of_bread_correct_l287_287819


namespace find_n_plus_N_n_plus_N_equals_neg_412_l287_287716

def min_max_expression (p q r s : ℝ) :=
  5 * (p^3 + q^3 + r^3 + s^3) - 3 * (p^4 + q^4 + r^4 + s^4)

theorem find_n_plus_N : ∃ (n N : ℝ), n = -224 ∧ N = -188 ∧ (∀ p q r s : ℝ, 
  p + q + r + s = 8 → 
  p^2 + q^2 + r^2 + s^2 = 20 → 
  n ≤ min_max_expression p q r s ∧ 
  min_max_expression p q r s ≤ N) :=
begin
  sorry,
end

theorem n_plus_N_equals_neg_412 : (∃ (n N : ℝ), n = -224 ∧ N = -188 ∧ (∀ p q r s : ℝ, 
  p + q + r + s = 8 → 
  p^2 + q^2 + r^2 + s^2 = 20 → 
  n ≤ min_max_expression p q r s ∧ 
  min_max_expression p q r s ≤ N)) → ∑ (n N), n + N = -412 :=
begin
  sorry,
end

end find_n_plus_N_n_plus_N_equals_neg_412_l287_287716


namespace combined_platforms_length_is_correct_l287_287296

noncomputable def combined_length_of_platforms (lengthA lengthB speedA_kmph speedB_kmph timeA_sec timeB_sec : ℝ) : ℝ :=
  let speedA := speedA_kmph * (1000 / 3600)
  let speedB := speedB_kmph * (1000 / 3600)
  let distanceA := speedA * timeA_sec
  let distanceB := speedB * timeB_sec
  let platformA := distanceA - lengthA
  let platformB := distanceB - lengthB
  platformA + platformB

theorem combined_platforms_length_is_correct :
  combined_length_of_platforms 650 450 115 108 30 25 = 608.32 := 
by 
  sorry

end combined_platforms_length_is_correct_l287_287296


namespace pipe_A_fill_time_l287_287934

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end pipe_A_fill_time_l287_287934


namespace last_part_length_l287_287764

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end last_part_length_l287_287764
