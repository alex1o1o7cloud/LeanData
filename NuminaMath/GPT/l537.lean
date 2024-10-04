import Analysis.Limits
import Data.Rat.Basic
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.Geometry.Circle.Incircle
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Cos
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Inscription
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Stats.Basic
import Mathlib.Tactic
import mathlib

namespace log_sum_geometric_sequence_l537_537020

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (r : ℝ) 
  (a_pos : ∀ n, 0 < a n)
  (geometric_seq : ∀ n, a (n + 1) = a n * r)
  (h : a 5 * a 6 = 81) :
  ∑ i in finset.range 10, real.log (a (i + 1)) / real.log 3 = 20 :=
sorry

end log_sum_geometric_sequence_l537_537020


namespace product_of_fractions_is_25_div_324_l537_537187

noncomputable def product_of_fractions : ℚ := 
  (10 / 6) * (4 / 20) * (20 / 12) * (16 / 32) * 
  (40 / 24) * (8 / 40) * (60 / 36) * (32 / 64)

theorem product_of_fractions_is_25_div_324 : product_of_fractions = 25 / 324 := 
  sorry

end product_of_fractions_is_25_div_324_l537_537187


namespace part1_purely_imaginary_part2_root_of_polynomial_l537_537600

variable (a : ℝ)
def z : ℂ := (a^2 - 5*a + 6 : ℂ) + (a - 3 : ℂ) * complex.I

-- Part 1: If z is purely imaginary, then a = 2
theorem part1_purely_imaginary (h1 : z.re = 0) : a = 2 := sorry

-- Part 2: If z is a root of the polynomial x^2 - 4x + 8 = 0, then |z| = 2 * sqrt(2)
def polynomial : polynomial ℂ := polynomial.C 8 - polynomial.C 4 * polynomial.X + polynomial.X^2
theorem part2_root_of_polynomial (h2 : polynomial.is_root z) : complex.abs z = 2 * real.sqrt 2 := sorry

end part1_purely_imaginary_part2_root_of_polynomial_l537_537600


namespace f_x_squared_l537_537668

def f : ℝ → ℝ := sorry  -- Define the function f

theorem f_x_squared (x : ℝ) (h : f(x - 1) = 2 * x + 5) : f(x^2) = 2 * x^2 + 7 := sorry

end f_x_squared_l537_537668


namespace angle_sum_proof_l537_537690

variables {A B C D E F : Type}
variables {angle_BAC angle_EDF : ℝ}

def isosceles_triangle (a b c : Type) (h1 h2 : ℝ) :=
  h1 = h2

def level_support (AD CE : Type) :=
  parallel AD CE

theorem angle_sum_proof :
  let angle_DAC := (180 - angle_BAC) / 2
  let angle_ADE := (180 - angle_EDF) / 2
  angle_BAC = 25 → angle_EDF = 40 → isosceles_triangle B A C angle_BAC angle_BAC → isosceles_triangle E D F angle_EDF angle_EDF → level_support D E →
  angle_DAC + angle_ADE = 147.5 :=
by
  intros
  sorry

end angle_sum_proof_l537_537690


namespace regular_polygon_sides_l537_537980

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537980


namespace sum_largest_smallest_l537_537083

def is_largest (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, n ≥ m

def is_smallest (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, n ≤ m

theorem sum_largest_smallest :
  let l := [25, 41, 13, 32] in
  let largest := 41 in
  let smallest := 13 in
  (is_largest largest l) ∧ (is_smallest smallest l) → largest + smallest = 54 := 
by
  intro l largest smallest h
  sorry

end sum_largest_smallest_l537_537083


namespace simple_interest_rate_l537_537861

theorem simple_interest_rate
  (A5 A8 : ℝ) (years_between : ℝ := 3) (I3 : ℝ) (annual_interest : ℝ)
  (P : ℝ) (R : ℝ)
  (h1 : A5 = 9800) -- Amount after 5 years is Rs. 9800
  (h2 : A8 = 12005) -- Amount after 8 years is Rs. 12005
  (h3 : I3 = A8 - A5) -- Interest for 3 years
  (h4 : annual_interest = I3 / years_between) -- Annual interest
  (h5 : P = 9800) -- Principal amount after 5 years
  (h6 : R = (annual_interest * 100) / P) -- Rate of interest formula revised
  : R = 7.5 := 
sorry

end simple_interest_rate_l537_537861


namespace mark_baseball_cards_gcd_l537_537038

theorem mark_baseball_cards_gcd :
  Nat.gcd (Nat.gcd 1080 1620) 540 = 540 :=
by
  sorry

end mark_baseball_cards_gcd_l537_537038


namespace car_total_distance_l537_537881

  noncomputable def distance_traveled (a₁ : ℕ) (d : ℤ) : ℕ :=
    Nat.sum (List.filter (λ n, n > 0) (List.range ((a₁ : ℤ) / (-d)) .map (λ n, (a₁ : ℤ) + n * d).map (λ n, Int.toNat n)))

  theorem car_total_distance :
    distance_traveled 40 (-10) = 100 :=
  by
    sorry
  
end car_total_distance_l537_537881


namespace find_angle_A_area_over_dot_product_l537_537354

-- Condition and definition for problem (1)
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (sin_A sin_B sin_C : ℝ)
hypothesis (h₁ : a ≠ c)
hypothesis (h₂ : b ≠ c)
hypothesis (h₃ : sin_A = Real.sin A)
hypothesis (sin_B = Real.sin B)
hypothesis (sin_C = Real.sin C)
hypothesis (h₄ : ((a - c) / (b - c)) = (sin_B / (sin_A + sin_C)))

-- Proof of question (1)
theorem find_angle_A (h : A = Real.arccos (1 / 2)) : A = Real.pi / 3 := by
  sorry

-- Condition and definition for problem (2)
variable (S : ℝ)
variable (AB AC : ℝ)
variable (dot_AB_AC : ℝ)
hypothesis (hS : S = 1 / 2 * AB * AC * Real.sin A)
hypothesis (h5 : dot_AB_AC = AB * AC * Real.cos A)

-- Proof of question (2)
theorem area_over_dot_product (hA : A = Real.pi / 3) : (S / dot_AB_AC) = Real.sqrt 3 / 2 := by
  sorry

end find_angle_A_area_over_dot_product_l537_537354


namespace smallest_n_condition_l537_537425

-- Conditions
variables (x y : ℤ)
variables (hx : x ≡ 4 [MOD 7]) (hy : y ≡ -4 [MOD 7])

-- Statement to prove
theorem smallest_n_condition : ∃ n, 0 < n ∧ (x^2 - x * y + y^2 + n) ≡ 0 [MOD 7] ∧ (∀ m, (x^2 - x * y + y^2 + m) ≡ 0 [MOD 7] → 0 < m → n ≤ m) :=
by
  -- Proof will go here
  sorry

end smallest_n_condition_l537_537425


namespace volleyball_height_30_l537_537908

theorem volleyball_height_30 (t : ℝ) : (60 - 9 * t - 4.5 * t^2 = 30) → t = 1.77 :=
by
  intro h_eq
  sorry

end volleyball_height_30_l537_537908


namespace regular_polygon_sides_l537_537994

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537994


namespace animal_legs_total_l537_537822

theorem animal_legs_total (total_animals ducks : ℕ) (duck_legs dog_legs : ℕ) :
  total_animals = 11 → ducks = 6 → duck_legs = 2 → dog_legs = 4 → 
  2 * ducks + 4 * (total_animals - ducks) = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    2 * 6 + 4 * (11 - 6) = 2 * 6 + 4 * 5 : by rfl
                    ... = 12 + 20         : by rfl
                    ... = 32              : by rfl

end animal_legs_total_l537_537822


namespace regular_polygon_sides_l537_537988

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537988


namespace diametrically_opposite_points_exist_l537_537884

theorem diametrically_opposite_points_exist (k : ℕ) :
  ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ p1 + π = p2 ∧ (∃ arcs : list ℝ, 
  list.length arcs = 3 * k ∧
  (∀ arc ∈ arcs, arc = 1 ∨ arc = 2 ∨ arc = 3) ∧
  (list.count 1 arcs = k) ∧ 
  (list.count 2 arcs = k) ∧ 
  (list.count 3 arcs = k)) :=
sorry

end diametrically_opposite_points_exist_l537_537884


namespace smallest_base10_integer_l537_537102

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end smallest_base10_integer_l537_537102


namespace unique_tuple_l537_537170

theorem unique_tuple :
  ∃ a b c d : ℕ, 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
    ({a * b + c * d, a * c + b * d, a * d + b * c} = {40, 70, 100}) ∧ 
    (a, b, c, d) = (1, 4, 6, 16) := by
  sorry

end unique_tuple_l537_537170


namespace eighth_odd_multiple_of_5_l537_537474

theorem eighth_odd_multiple_of_5 : 
  (∃ n : ℕ, n = 8 ∧ ∃ k : ℤ, k = (10 * n - 5) ∧ k > 0 ∧ k % 2 = 1) → 75 := 
by {
  sorry
}

end eighth_odd_multiple_of_5_l537_537474


namespace product_of_chords_l537_537383

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 18)

noncomputable def chord_length_A (k : ℕ) : ℝ :=
  3 * Complex.abs (1 - omega ^ k)

noncomputable def chord_length_B (k : ℕ) : ℝ :=
  3 * Complex.abs (1 + omega ^ k)

theorem product_of_chords :
  (Finset.prod (Finset.range 8) (λ k, chord_length_A (k + 1))) *
  (Finset.prod (Finset.range 8) (λ k, chord_length_B (k + 1))) = 472392 := by
  sorry

end product_of_chords_l537_537383


namespace reflection_y_axis_correct_l537_537785

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l537_537785


namespace sum_of_digits_is_2015_l537_537714

def prime_p : ℕ := 2017

def order (A : matrix (fin n) (fin n) (zmod prime_p)) 
  (I : matrix (fin n) (fin n) (zmod prime_p)) : ℕ := 
  min {d | 0 < d ∧ A^d = I}

def a_n (n : ℕ) : ℕ := prime_p^n - 1

def sum_a_k (p : ℕ) : ℕ :=
  (finset.range (p + 1)).sum (λ k, a_n k)

def sum_of_digits_in_base_p (n p : ℕ) : ℕ :=
  let digits := nat.digits p n in digits.sum

theorem sum_of_digits_is_2015 :
  sum_of_digits_in_base_p (sum_a_k prime_p) prime_p = 2015 :=
by
  -- proof here
  sorry

end sum_of_digits_is_2015_l537_537714


namespace john_profit_calculation_l537_537364

-- Definitions for initial sales figures before discounts and taxes
def woodburnings_sales := 20 * 15
def metal_sculptures_sales := 15 * 25
def paintings_sales := 10 * 40
def glass_figurines_sales := 5 * 30

-- Definitions for costs
def wood_cost := 100
def metal_cost := 150
def paint_cost := 120
def glass_cost := 90

-- Definitions for discounts
def woodburnings_discount := woodburnings_sales * 0.10
def glass_figurines_discount := glass_figurines_sales * 0.15

-- Sales after applying discounts
def woodburnings_sales_after_discount := woodburnings_sales - woodburnings_discount
def glass_figurines_sales_after_discount := glass_figurines_sales - glass_figurines_discount

-- Total sales after discounts
def total_sales_after_discounts := woodburnings_sales_after_discount + metal_sculptures_sales + paintings_sales + glass_figurines_sales_after_discount

-- Sales tax
def sales_tax := total_sales_after_discounts * 0.05

-- Final sales amount including tax
def final_sales_amount := total_sales_after_discounts + sales_tax

-- Total costs
def total_costs := wood_cost + metal_cost + paint_cost + glass_cost

-- Profit calculation
def profit := final_sales_amount - total_costs

-- Theorem
theorem john_profit_calculation : profit = 771.13 := by
  sorry

end john_profit_calculation_l537_537364


namespace max_red_dragons_at_table_l537_537347

noncomputable def max_red_dragons (total_dragons : ℕ) : ℕ :=
  if h : total_dragons % 3 = 0 
  then total_dragons / 3 
  else total_dragons / 3

theorem max_red_dragons_at_table : max_red_dragons 530 = 176 :=
by
  simp [max_red_dragons]
  sorry

end max_red_dragons_at_table_l537_537347


namespace cosine_60_degrees_l537_537950

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537950


namespace smallest_value_is_2_5_l537_537724

noncomputable def smallest_value_z_plus_i (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) : ℝ :=
  Inf { w : ℝ | ∃ z : ℂ, |z^2 + 9| = |z * (z + 3 * complex.I) ∧ w = complex.abs (z + complex.I) }

theorem smallest_value_is_2_5 (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) :
  smallest_value_z_plus_i z h = 2.5 :=
sorry

end smallest_value_is_2_5_l537_537724


namespace cone_volume_is_correct_l537_537813

-- Given conditions
def slant_height : ℝ := 15
def height : ℝ := 9

-- Derived quantities
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- Volume formula for a cone
def volume_cone (r h : ℝ) := (1 / 3) * π * r^2 * h

theorem cone_volume_is_correct : volume_cone radius height = 432 * π := by
  sorry

end cone_volume_is_correct_l537_537813


namespace determine_function_l537_537566

theorem determine_function (f : ℕ → ℕ) :
  (∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) →
  ∃ k : ℕ, ∀ n : ℕ, f n = k * n^2 :=
by
  sorry

end determine_function_l537_537566


namespace regular_polygon_sides_l537_537981

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537981


namespace cos_relation_in_cyclic_quadrilateral_l537_537508

theorem cos_relation_in_cyclic_quadrilateral
  {A B C D E : Point}
  (h_circle : inscribed_in_circle A B C D E)
  (h_AB_CD_DE : dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D E = 4)
  (h_AE : dist A E = 2) :
  (1 - (cos (angle B))) * (1 - (cos (angle ACE))) = 1 / 16 := by
  sorry

end cos_relation_in_cyclic_quadrilateral_l537_537508


namespace investor_initial_deposit_l537_537549

theorem investor_initial_deposit (A r : ℚ) (n t : ℕ) (hA : A = 5202) (hr : r = 0.08) (hn : n = 4) (ht : t = 1/2) :
  let P := A / (1 + r / n)^((n : ℚ) * t)
  P ≈ 5000 :=
by
  sorry

end investor_initial_deposit_l537_537549


namespace Sabrina_cookies_left_l537_537411

theorem Sabrina_cookies_left (start : ℕ) (brother_given mom_given_sister_given : ℕ) : start = 20 → brother_given = 10 → mom_given_sister_given = 5 → (start - brother_given + mom_given_sister_given) = 15 →
(sister_given = 2 / 3 * (start - brother_given + mom_given_sister_given)) → (start - brother_given + mom_given_sister_given - sister_given) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Sabrina_cookies_left_l537_537411


namespace javier_baked_200_cookies_l537_537363

variable (X : ℕ)

theorem javier_baked_200_cookies 
  (h : (0.70 * X - 40) / 2 = 50) : 
  X = 200 := 
sorry

end javier_baked_200_cookies_l537_537363


namespace correct_operation_l537_537854

theorem correct_operation :
  (∀ a : ℕ, ∀ b : ℕ, ¬(a^2 + a = a^3)) ∧
  (∀ a : ℕ, (a^6 / a^2 = a^4)) ∧
  (∀ a b : ℕ, ¬((-2 * a * b^2)^3 = -6 * a^3 * b^6)) ∧
  (∀ a : ℕ, (a^2 * a^3 = a^5)) :=
by
  sorry

end correct_operation_l537_537854


namespace min_coins_to_pay_up_to_149_l537_537094

-- Define the types for each coin denomination
def pennies : ℕ := 1
def nickels : ℕ := 5
def dimes : ℕ := 10
def quarters : ℕ := 25
def halfDollars : ℕ := 50

-- Define a function for the total number of coins
def min_coins (amount : ℕ) : ℕ :=
  if amount < 5 then amount
  else if amount < 10 then (amount / 5 + min_coins (amount % 5))
  else if amount < 25 then (amount / 10 + min_coins (amount % 10))
  else if amount < 50 then (amount / 25 + min_coins (amount % 25))
  else (amount / 50 + min_coins (amount % 50))

-- The assertion to prove: minimum coins needed to cover any amount from 1 to 149 cents is 10
theorem min_coins_to_pay_up_to_149 : ∀ amount ∈ {1, 2, ..., 149}, min_coins amount ≤ 10 :=
begin
  sorry
end

end min_coins_to_pay_up_to_149_l537_537094


namespace find_s_l537_537721

def isMonicCubic (p : ℝ[X]) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def hasRoots (p : ℝ[X]) (roots : set ℝ) : Prop :=
  ∀ r ∈ roots, p.isRoot r

theorem find_s (f g : ℝ[X]) (s : ℝ) :
  isMonicCubic f ∧ isMonicCubic g ∧
  hasRoots f {s + 2, s + 8} ∧ hasRoots g {s + 5, s + 11} ∧
  (∀ x, f.eval x - g.eval x = 2 * s) →
  s = 81 / 4 :=
by
  intros
  sorry

end find_s_l537_537721


namespace angle_between_diagonals_of_cube_l537_537699

def cube_face_angle : real := sorry

theorem angle_between_diagonals_of_cube :
  ∀ (a : ℝ) (A B C D E F : Type)
  (faces_are_square : ∀ (f : Type), (f = A ∨ f = B ∨ f = C ∨ f = D ∨ f = E ∨ f = F) → true)
  (BM : A) (CN : C)
  (BM_is_diagonal : BM = some_diagonal A B F E)
  (CN_is_diagonal : CN = some_diagonal A D C B)
  (AB_is_common_edge : AB_edges_share A B F E A D C B)
  (ABFE_ADCB_perpendicular : perpendicular_faces A B F E A D C B),
  has_angle BM CN 90 :=
sorry

end angle_between_diagonals_of_cube_l537_537699


namespace interest_calculation_l537_537837

/-- Define the initial deposit in thousands of yuan (50,000 yuan = 5 x 10,000 yuan) -/
def principal : ℕ := 5

/-- Define the annual interest rate as a percentage in decimal form -/
def annual_interest_rate : ℝ := 0.04

/-- Define the number of years for the deposit -/
def years : ℕ := 3

/-- Calculate the total amount after 3 years using compound interest -/
def total_amount_after_3_years : ℝ :=
  principal * (1 + annual_interest_rate) ^ years

/-- Calculate the interest earned after 3 years -/
def interest_earned : ℝ :=
  total_amount_after_3_years - principal

theorem interest_calculation :
  interest_earned = 5 * (1 + 0.04) ^ 3 - 5 :=
by 
  sorry

end interest_calculation_l537_537837


namespace weight_of_B_l537_537433

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l537_537433


namespace volume_pyramid_ABC_l537_537564

structure Point where
  x : ℝ
  y : ℝ

def triangle_volume (A B C : Point) : ℝ :=
  -- The implementation would calculate the volume of the pyramid formed
  -- by folding along the midpoint sides.
  sorry

theorem volume_pyramid_ABC :
  let A := Point.mk 0 0
  let B := Point.mk 30 0
  let C := Point.mk 20 15
  triangle_volume A B C = 900 :=
by
  -- To be filled with the proof
  sorry

end volume_pyramid_ABC_l537_537564


namespace find_doodads_produced_in_four_hours_l537_537080

theorem find_doodads_produced_in_four_hours :
  ∃ (n : ℕ),
    (∀ (workers hours widgets doodads : ℕ),
      (workers = 150 ∧ hours = 2 ∧ widgets = 800 ∧ doodads = 500) ∨
      (workers = 100 ∧ hours = 3 ∧ widgets = 750 ∧ doodads = 600) ∨
      (workers = 80  ∧ hours = 4 ∧ widgets = 480 ∧ doodads = n)
    ) → n = 640 :=
sorry

end find_doodads_produced_in_four_hours_l537_537080


namespace sum_f_k_l537_537719

noncomputable def f (n : ℕ) : ℕ :=
  let m := (Real.sqrt (Real.sqrt (n.toReal))).toNat
  if m - 1 % 2 = 0 then m - 1 else m

theorem sum_f_k (n : ℕ) : 
  (∑ k in Finset.range 2995, (1 : ℚ) / f (k + 1)) = 768 := 
by
  sorry

end sum_f_k_l537_537719


namespace lesis_keeps_rice_fraction_l537_537745

-- Define the conditions
def total_rice : ℝ := 50
def fraction_kept (f : ℝ) : ℝ := f
def fraction_given (f : ℝ) : ℝ := (1 - f)
def rice_lesis_kept (f : ℝ) : ℝ := fraction_kept f * total_rice
def rice_everest_received (f : ℝ) : ℝ := fraction_given f * total_rice
def rice_difference_condition (f : ℝ) : Prop := rice_lesis_kept f = rice_everest_received f + 20

-- The statement to prove
theorem lesis_keeps_rice_fraction : ∃ (f : ℝ), rice_difference_condition f ∧ f = 7 / 10 :=
by {
  have h := rice_difference_condition,
  existsi 7 / 10,
  split,
  sorry,
  refl
}

end lesis_keeps_rice_fraction_l537_537745


namespace find_sequences_range_of_lambda_l537_537257

-- Define the arithmetic sequence {a_n} with initial term a1 and common difference d.
def arithmetic_seq (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ := a1 + d * (n - 1)

-- Define the geometric sequence {b_n} with initial term b1 and common ratio q.
def geometric_seq (n : ℕ) (b1 : ℕ) (q : ℕ) : ℕ := b1 * q ^ (n - 1)

-- Define the sequence {c_n}
def c_seq (n : ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) (λ : ℝ) : ℝ :=
  3 * b n - 2 * λ * (a n / 3)

-- Prove part (1): a_n = 3n and b_n = 3^(n-1)
theorem find_sequences (n : ℕ) : 
  (∀ n, arithmetic_seq n 3 3 = 3 * n) ∧
  (∀ n, geometric_seq n 1 3 = 3^(n - 1)) := 
by 
  split;
  sorry

-- Prove part (2): if {c_n} is increasing, find the range of λ
theorem range_of_lambda (λ : ℝ) 
  (c_n : ℕ → ℝ)
  (h : ∀ n, c_seq n (geometric_seq 1 3) (arithmetic_seq 3 3) λ < 
    c_seq (n + 1) (geometric_seq 1 3) (arithmetic_seq 3 3) λ):
  λ < 3 :=
by 
  sorry

end find_sequences_range_of_lambda_l537_537257


namespace product_third_fourth_term_l537_537065

theorem product_third_fourth_term (a d : ℝ) : 
  (a + 7 * d = 20) → (d = 2) → 
  ( (a + 2 * d) * (a + 3 * d) = 120 ) := 
by 
  intros h1 h2
  sorry

end product_third_fourth_term_l537_537065


namespace properties_of_g_x_squared_l537_537635

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

noncomputable def g (x : ℝ) : ℝ := Real.logBase (1 / 2) x

theorem properties_of_g_x_squared :
  (∀ x : ℝ, g (x * x) = Real.logBase (1 / 2) (x * x)) ∧
  (∀ x : ℝ, g (x * x) = g ((-x) * (-x))) ∧
  (∀ x y : ℝ, x < y → g (x * x) < g (y * y)) ∧
  even (λ x, g (x * x)) ∧
  (∀ x y : ℝ, x < y → y < 0 → g (x * x) < g (y * y)) :=
by sorry

end properties_of_g_x_squared_l537_537635


namespace simplify_polynomial_sum_l537_537773

/- Define the given polynomials -/
def polynomial1 (x : ℝ) : ℝ := (5 * x^10 + 8 * x^9 + 3 * x^8)
def polynomial2 (x : ℝ) : ℝ := (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9)
def resultant_polynomial (x : ℝ) : ℝ := (2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9)

theorem simplify_polynomial_sum (x : ℝ) :
  polynomial1 x + polynomial2 x = resultant_polynomial x :=
by
  sorry

end simplify_polynomial_sum_l537_537773


namespace polar_to_cartesian_min_distance_l537_537339

theorem polar_to_cartesian_min_distance :
  (∀ (ρ θ : ℝ), ρ^2 = 2 / (1 + sin θ^2) → ∃ x y : ℝ, x^2 + 2 * y^2 = 2)
  ∧ (∀ (ρ θ : ℝ), ρ = 4 / (sqrt 2 * sin θ + cos θ) → ∃ x y : ℝ, x + sqrt 2 * y = 4)
  ∧ ∃ (f : ℝ → ℝ → ℝ), 
    (∀ θ : ℝ, x = sqrt 2 * cos θ y = sin θ → f x (sqrt 2 * y) = |sqrt 2 * sin θ + sqrt 2 * cos θ - 4| / sqrt 3)
    → (∀ (d : ℝ), d = |2 * sin (θ + π / 4) - 4| / sqrt 3 → d ≥ 2 / sqrt 3) := sorry

end polar_to_cartesian_min_distance_l537_537339


namespace length_of_first_two_songs_l537_537742

noncomputable def first_two_song_lengths (total_minutes : ℕ) (days : ℕ) (gigs_per_week : ℕ) : ℕ :=
  let gigs := days / (2 * gigs_per_week) in
  let songs_per_gig := 3 in
  let total_songs := gigs * songs_per_gig in
  /- Let x represent the length of the first two songs in minutes -/
  let x := total_minutes / (28 * gigs_per_week) in
  /- In this problem 28 = 7 * 4 since there are 7 gigs and each gig has 4x minutes -/
  x

theorem length_of_first_two_songs (total_minutes : ℕ) (days : ℕ) (gigs_per_week : ℕ) : 
  first_two_song_lengths total_minutes days gigs_per_week = 10 := 
by
  /- We assume the input conditions from the problem: -/
  /- total_minutes = 280, days = 14, gigs_per_week = 7 -/
  have h1 : gigs_per_week = 7 := sorry
  have h2 : days = 14 := sorry
  have h3 : total_minutes = 280 := sorry
  have h4 : 2 * gigs_per_week = 14 := sorry /- For 2 weeks, work gig every other day so gigs are 14/2 = 7 gigs -/
  have h5 : (14 / (2 * gigs_per_week)) = 1 := sorry
  have h6 : (280 / (28 * 7)) = 10 := sorry
  /- Hence proved that first_two_song_lengths = 10 -/
  sorry

end length_of_first_two_songs_l537_537742


namespace trig_identity_l537_537215

theorem trig_identity :
  sin (50 * π / 180) * (1 + sqrt 3 * tan (10 * π / 180)) = 1 :=
by
  sorry

end trig_identity_l537_537215


namespace sin_x1_plus_x2_l537_537267

variable (x_1 x_2 m : ℝ)

-- Assuming x1 and x2 are roots of the function within the specified interval
axiom root1 : 0 ≤ x_1 ∧ x_1 ≤ π / 2 ∧ 2 * sin (2 * x_1) + cos (2 * x_1) = m
axiom root2 : 0 ≤ x_2 ∧ x_2 ≤ π / 2 ∧ 2 * sin (2 * x_2) + cos (2 * x_2) = m

theorem sin_x1_plus_x2 : sin (x_1 + x_2) = (2 * Real.sqrt 5) / 5 :=
sorry

end sin_x1_plus_x2_l537_537267


namespace sarahs_score_l537_537763

variable (s g : ℕ)

theorem sarahs_score :
  (s = g + 50) ∧ ((s + g) / 2 = 95) → s = 120 :=
by
  intro h,
  rcases h with ⟨h_sg, h_avg⟩,
  sorry

end sarahs_score_l537_537763


namespace simplify_vector_expression_l537_537052

universe u

section
variables {V : Type u} [AddCommGroup V] [VectorSpace ℝ V]

-- Definitions of the vectors
variables (OP PQ MQ OQ QM MO OM : V)

-- Conditions
axiom head_to_tail_rule : OP + PQ = OQ
axiom changing_order_direction : QM + (-MO) = OM

-- Proof statement
theorem simplify_vector_expression :
  OP + PQ - MQ = OM :=
by 
  have h1 : OQ - MQ = QM + (-MO), from sorry,
  sorry
end

end simplify_vector_expression_l537_537052


namespace derivative_at_pi_l537_537637

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.cos x)

theorem derivative_at_pi : deriv f π = -2 * π :=
by
  sorry

end derivative_at_pi_l537_537637


namespace chord_lengths_equal_l537_537342

theorem chord_lengths_equal (D E F : ℝ) (hcond_1 : D^2 ≠ E^2) (hcond_2 : E^2 > 4 * F) :
  ∀ x y, (x^2 + y^2 + D * x + E * y + F = 0) → 
  (abs x = abs y) :=
by
  sorry

end chord_lengths_equal_l537_537342


namespace incorrect_height_is_151_l537_537061

def incorrect_height (average_initial correct_height average_corrected : ℝ) : ℝ :=
  (30 * average_initial) - (30 * average_corrected) + correct_height

theorem incorrect_height_is_151 :
  incorrect_height 175 136 174.5 = 151 :=
by
  sorry

end incorrect_height_is_151_l537_537061


namespace sum_of_fractions_as_decimal_l537_537216

theorem sum_of_fractions_as_decimal : (3 / 8 : ℝ) + (5 / 32) = 0.53125 := by
  sorry

end sum_of_fractions_as_decimal_l537_537216


namespace ratio_KM_DB1_l537_537348
-- importing all necessary libraries

-- defining the properties of geometrical objects
variables (A B C D A1 B1 C1 D1 K M : Point)
variable parallelepiped : Parallelepiped A B C D A1 B1 C1 D1
variable K_on_AC : OnLine K (Line AC)
variable M_on_BA1 : OnLine M (Line BA1)
variable KM_parallel_DB1 : Parallel (Line KM) (Line DB1)

theorem ratio_KM_DB1 : (∃ r : ℚ, r = 1 / 3 ∧ KM.length = r * DB1.length) := by
  sorry

end ratio_KM_DB1_l537_537348


namespace petya_should_run_downwards_l537_537752

theorem petya_should_run_downwards 
  (v : ℝ)  -- Petya's speed
  (v_escalator : ℝ)  -- Speed of the escalator
  (h1 : v > v_escalator)  -- Condition that Petya's speed is greater than the escalator's speed
  : Petya_should_run_downwards :=
begin
  -- Assuming Petya_should_run_downwards is a proposition that returns True when it's faster for Petya to run downwards
  sorry
end

end petya_should_run_downwards_l537_537752


namespace area_of_region_a_area_of_region_b_area_of_region_c_l537_537223

-- Definition of regions and their areas
def area_of_square : Real := sorry
def area_of_diamond : Real := sorry
def area_of_hexagon : Real := sorry

-- Define the conditions for the regions
def region_a (x y : ℝ) := abs x ≤ 1 ∧ abs y ≤ 1
def region_b (x y : ℝ) := abs x + abs y ≤ 10
def region_c (x y : ℝ) := abs x + abs y + abs (x + y) ≤ 2020

-- Prove that the areas match the calculated solutions
theorem area_of_region_a : area_of_square = 4 := 
by sorry

theorem area_of_region_b : area_of_diamond = 200 := 
by sorry

theorem area_of_region_c : area_of_hexagon = 3060300 := 
by sorry

end area_of_region_a_area_of_region_b_area_of_region_c_l537_537223


namespace proof_problem_l537_537063

noncomputable def ellipse_equation_and_eccentricity (a b c e : ℝ) (x y : ℝ) : Prop :=
(center_origin : True) 
(minor_axis_len : b = sqrt 2)
(of_eq_2fa : c = 2 * sqrt 6) 
(eccentricity : e = c / a) 
(ellipse_eq : (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_pq_equation (k : ℝ) (x y : ℝ) : Prop :=
(intercept_at_a : x = 3)
(intersects_ellipse : ((x^2 / 6) + ((k * (x - 3))^2 / 2) = 1))
(orthogonality : -sqrt(5) / 3 < k ∧ k < sqrt(5) / 3)
(line_eq1 : x - sqrt(5) * y = 3 ∨ line_eq2 : x + sqrt(5) * y = 3)

theorem proof_problem :
    ∃ (a b c e : ℝ) (x y : ℝ), ellipse_equation_and_eccentricity a b c e x y ∧
    ∃ (k : ℝ) (x y : ℝ), line_pq_equation k x y :=
begin
    -- Placeholder for proof steps
    sorry
end

end proof_problem_l537_537063


namespace find_modulus_of_alpha_l537_537014

open Complex

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry

theorem find_modulus_of_alpha 
  (h0 : beta = conj α)
  (h1 : (alpha^2 / beta).im = 0)
  (h2 : abs (alpha - beta) = 4) :
  abs α = 4 * Real.sqrt 3 / 3 :=
sorry

end find_modulus_of_alpha_l537_537014


namespace problem_solution_l537_537108

theorem problem_solution : (101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3) = 1_092_727 := by
  sorry

end problem_solution_l537_537108


namespace abs_diff_eq_0_5_l537_537009

noncomputable def x : ℝ := 3.7
noncomputable def y : ℝ := 4.2

theorem abs_diff_eq_0_5 (hx : ⌊x⌋ + (y - ⌊y⌋) = 3.2) (hy : (x - ⌊x⌋) + ⌊y⌋ = 4.7) :
  |x - y| = 0.5 :=
by
  sorry

end abs_diff_eq_0_5_l537_537009


namespace solution_set_inequality_l537_537629

variable (f : ℝ → ℝ)

theorem solution_set_inequality (h1 : ∀ x, 0 < x → f(x) > -x * (deriv f(x)))
  (h2 : ∀ x, 0 < x): 
  {x | f(x + 1) > (x - 1) * f(x^2 - 1)} = {x | 1 < x ∧ x < 2} :=
sorry

end solution_set_inequality_l537_537629


namespace part1_part2_part3_l537_537640

variables (f g : ℝ → ℝ) (a λ : ℝ)

-- Part (I): 
def is_function_increasing : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2)

-- Part (II):
def minimum_value_on_interval (f : ℝ → ℝ) (I : set ℝ) (val : ℝ) : Prop :=
  ∃ x ∈ I, ∀ y ∈ I, f y ≥ f x ∧ f x = val

-- Part (III):
def inequality_holds1 (f g : ℝ → ℝ) (λ : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → x * f(x) ≤ λ * (g(x) + x)

axiom problem_conditions : 
  (f = λ x, ln x - a / x) ∧ 
  (g = λ x, (1 / 2) * (x - 1)^2 - 1)

-- Statement Part I
theorem part1 (h : a > 0) : is_function_increasing (λ x, ln x - a / x) :=
sorry

-- Statement Part II
theorem part2 (h : minimum_value_on_interval (λ x, ln x - a / x) (set.Icc 1 real.exp 1) (3/2)) :
  a = - real.sqrt real.exp 1 :=
sorry

-- Statement Part III
theorem part3 (h : a = 0) (h2 : inequality_holds1 (λ x, ln x) (λ x, (1 / 2) * (x - 1)^2 - 1) λ) :
  1 ≤ λ :=
sorry

end part1_part2_part3_l537_537640


namespace rhombus_area_l537_537337

variable (ABCD : Type) [linear_ordered_semiring ABCD] {A B C D O : ABCD} -- ABCD as a parameterized rhombus type
variable [has_add ABCD] [has_mul ABCD] [has_div ABCD] [has_sub ABCD] [has_sqrt ABCD] [has_pow ABCD]

def is_rhombus (ABCD : Type) [linear_ordered_semiring ABCD] (A B C D O : ABCD) : Prop :=
  true -- Assuming some properties and makeup of the rhombus, can be defined as needed

def is_perpendicular_bisector (x y : ABCD) : Prop :=
  x = 6 / 2 -- Intersection point and bisection by diagonals assumed true

theorem rhombus_area (ABCD : Type) [linear_ordered_semiring ABCD] {A B C D O : ABCD}
  (h1 : is_rhombus ABCD A B C D O)
  (h2 : 4 * 5 = 20)
  (h3 : 6 / 2 = 3)
  (h4 : 5 ^ 2 - 3 ^ 2 = 16)
  (h5 : sqrt 16 = 4)
  (h6 : 2 * 4 = 8) :
  (1 / 2) * 6 * 8 = 24 := by
  sorry

end rhombus_area_l537_537337


namespace arnold_danny_age_l537_537863

theorem arnold_danny_age:
  ∃ x : ℝ, (x + 1) * (x + 1) = x * x + 11 ∧ x = 5 :=
by
  sorry

end arnold_danny_age_l537_537863


namespace right_triangle_hypotenuse_l537_537073

theorem right_triangle_hypotenuse:
  ∀ (a b : ℝ) (m_a m_b : ℝ),
  m_a = sqrt (b^2 + (3 * a / 2)^2) →
  m_b = sqrt (a^2 + (3 * b / 2)^2) →
  m_a = sqrt 48 →
  m_b = 6 →
  sqrt (9 * a^2 + 9 * b^2) = 15.25 :=
by
  intros a b m_a m_b h1 h2 h3 h4
  sorry

end right_triangle_hypotenuse_l537_537073


namespace two_ships_distance_is_correct_l537_537467

-- Defining the conditions from the problem statement
def window_width : ℝ := 40 -- in cm
def window_height : ℝ := 25 -- in cm
def eye_distance_from_window : ℝ := 20 -- in cm
def plane_altitude : ℝ := 1030000 -- converted to cm (10.3 km * 100000)

-- Defining function to calculate the distance between two ships
def distance_between_ships (window_width : ℝ) (eye_distance_from_window : ℝ) (plane_altitude : ℝ) : ℝ :=
  let ratio := plane_altitude / eye_distance_from_window
  in ratio * window_width

-- Theorem statement: Verifying that calculated distance is as expected
theorem two_ships_distance_is_correct :
  distance_between_ships window_width eye_distance_from_window plane_altitude = 128750 := sorry

end two_ships_distance_is_correct_l537_537467


namespace cappuccino_cost_l537_537192

variable {C : ℝ} -- cost of a cappuccino

-- Given conditions
def iced_tea_cost : ℝ := 3
def cafe_latte_cost : ℝ := 1.5
def espresso_cost : ℝ := 1
def total_paid : ℝ := 20
def change_received : ℝ := 3

-- Sandy's order costs
def total_cost : ℝ :=
  3 * C + 2 * iced_tea_cost + 2 * cafe_latte_cost + 2 * espresso_cost

-- Total after change is subtracted
def total_after_change : ℝ :=
  total_paid - change_received

-- Main theorem: The cost of a cappuccino is $2
theorem cappuccino_cost : total_cost = total_after_change → C = 2 := by
  sorry

end cappuccino_cost_l537_537192


namespace area_of_circular_base_of_fountain_l537_537828

-- Definitions for the conditions
def Point : Type := ℝ × ℝ

variable (A B C D : Point)

def midpoint (A B : Point) : Point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def distance (P Q : Point) : ℝ := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def D_is_midpoint_of_AB (A B D : Point) : Prop :=
  D = midpoint A B

def length_AB (A B : Point) : Prop :=
  distance A B = 16

def length_DC (D C : Point) : Prop :=
  distance D C = 10

-- Proof statement
theorem area_of_circular_base_of_fountain
  (A B C D : Point)
  (h1 : D_is_midpoint_of_AB A B D)
  (h2 : length_AB A B)
  (h3 : length_DC D C) :
  ∃ R, Real.pi * R ^ 2 = 164 * Real.pi :=
  sorry

end area_of_circular_base_of_fountain_l537_537828


namespace natural_numbers_with_four_divisors_l537_537204

theorem natural_numbers_with_four_divisors (n : ℕ) (h₁ : 2 ≤ n) :
  (∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ ≠ d₂ → d₁ - d₂ ∣ n) →
  n = 4 ∨ (∀ p : ℕ, prime p → n = p) :=
by
  sorry

end natural_numbers_with_four_divisors_l537_537204


namespace triangle_area_bound_l537_537507

theorem triangle_area_bound 
  (S S1 S2 : ℝ) 
  (a b c a1 b1 c1 a2 b2 c2 : ℝ)
  (hAB : a = a1 + a2) 
  (hBC : b = b1 + b2) 
  (hCA : c = c1 + c2) 
  (HS : S = 4 * S1 * S2) :
  S ≤ 4 * real.sqrt (S1 * S2) := 
  sorry

end triangle_area_bound_l537_537507


namespace solve_system_of_equations_l537_537055

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), (5 * x + 3 * y = 65) ∧ (2 * y - z = 11) ∧ (3 * x + 4 * z = 57) ∧ x = 7 ∧ y = 10 ∧ z = 9 :=
by
  use 7, 10, 9
  repeat { split }
  { rw [mul_comm], norm_num }
  { norm_num }
  { norm_num }
  { norm_num }
  { norm_num }
  { norm_num }

end solve_system_of_equations_l537_537055


namespace range_of_a_perp_OA_OB_l537_537649

-- Problem 1: Range of a for intersection
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, y = a * x + 1 ∧ 3 * x^2 - y^2 = 1) →
  - real.sqrt 6 < a ∧ a < real.sqrt 6 ∧ a ≠ real.sqrt 3 ∧ a ≠ -real.sqrt 3 :=
sorry

-- Problem 2: Value of a if OA and OB are perpendicular
theorem perp_OA_OB (a : ℝ) :
  ((∃ x1 x2 : ℝ, (3 - a^2) * x1^2 - 2 * a * x1 - 2 = 0 ∧
                   (3 - a^2) * x2^2 - 2 * a * x2 - 2 = 0 ∧
                   x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) →
    (a = 1 ∨ a = -1)) :=
sorry

end range_of_a_perp_OA_OB_l537_537649


namespace problem_statement_l537_537033

def U : Set ℤ := {x | True}
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}
def complement (B : Set ℤ) : Set ℤ := {x | x ∉ B}

theorem problem_statement : (A ∩ (complement B)) = {1, 3, 9} :=
by {
  sorry
}

end problem_statement_l537_537033


namespace find_m_value_l537_537655

noncomputable def is_solution (p q m : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (x^2 - m*x + m^2 - 19 = 0)

theorem find_m_value :
  let A := { x : ℝ | x^2 + 2 * x - 8 = 0 }
  let B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
  ∀ (C : ℝ → Prop), 
  (∃ x, B x ∧ C x) ∧ (¬ ∃ x, A x ∧ C x) → 
  (∃ m, C = { x : ℝ | x^2 - m * x + m^2 - 19 = 0 } ∧ m = -2) :=
by
  sorry

end find_m_value_l537_537655


namespace cos_60_degrees_is_one_half_l537_537935

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537935


namespace cyclic_quad_area_l537_537028

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

theorem cyclic_quad_area (a b c d : ℝ) (ABCD : ConvexCyclicQuadrilateral a b c d) :
  let p := semiperimeter a b c d in
  area ABCD = Real.sqrt ((p - a) * (p - b) * (p - c) * (p - d)) := sorry

end cyclic_quad_area_l537_537028


namespace y_is_decreasing_l537_537799

open Real

-- Define the function u and y
def u (x : ℝ) : ℝ := 1 - 2 * cos (2 * x)
def y (x : ℝ) : ℝ := log (1 / 2) (u x)

-- Define the interval [π/6, π/2]
def I : Set ℝ := Set.Icc (π / 6) (π / 2)

-- Lean 4 statement to prove monotonicity
theorem y_is_decreasing : ∀ x1 x2 ∈ I, x1 ≤ x2 → y x2 ≤ y x1 :=
sorry

end y_is_decreasing_l537_537799


namespace semicircle_area_l537_537513

theorem semicircle_area (d : ℝ) (h : d = 3) : (π * (d / 2)^2) / 2 = 9 * π / 8 :=
by {
  have r := d / 2,
  rw h,
  have r_value : r = 3 / 2 := by { rw h, exact rfl },
  rw [r_value, pow_two, ←mul_div_assoc, mul_div, ←div_mul_eq_mul_div, ←div_div, mul_comm π, div_div_eq_mul_div],
  exact rfl
}

end semicircle_area_l537_537513


namespace compare_negative_fractions_l537_537927

theorem compare_negative_fractions : (-3/4 : ℚ) < (-2/3 : ℚ) :=
by sorry

end compare_negative_fractions_l537_537927


namespace triangle_is_isosceles_l537_537452

theorem triangle_is_isosceles 
  (a b c : ℝ)
  (h : a^2 - b^2 + a * c - b * c = 0)
  (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  : a = b := 
sorry

end triangle_is_isosceles_l537_537452


namespace sufficient_necessary_conditions_l537_537314

variable (A B C D : Prop)

#check
theorem sufficient_necessary_conditions :
  (A → B) ∧ ¬ (B → A) →
  (C → B) ∧ ¬ (B → C) →
  (D → C) ∧ ¬ (C → D) →
  ((B → A) ∧ ¬ (A → B)) ∧
  ((A → C) ∧ ¬ (C → A)) ∧
  (¬ (D → A) ∧ ¬ (A → D)) :=
by
  intros h1 h2 h3
  cases h1 with hA_implies_B hnB_implies_A
  cases h2 with hC_implies_B hnB_implies_C
  cases h3 with hD_implies_C hnC_implies_D
  split
  -- Proof for B → A and ¬A → B
  split
  { intros hB,
    apply false.elim,
    apply hnB_implies_A,
    exact hB, },
  { exact hnB_implies_A, },
  -- Proof for A → C and ¬C → A
  split
  { intros hA,
    apply hC_implies_B,
    apply hA_implies_B,
    exact hA, },
  { apply false.elim,
    intros hA,
    apply hnB_implies_A,
    exact hA, },
  -- Proof for ¬D → A and ¬A → D
  split
  { intros hD,
    apply false.elim,
    apply hnB_implies_A,
    exact hD, },
  { intros hA,
    apply false.elim,
    apply hnC_implies_D,
    exact hA, }

end sufficient_necessary_conditions_l537_537314


namespace coeff_sum_eq_neg_two_l537_537670

theorem coeff_sum_eq_neg_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^10 + x^4 + 1) = a + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 + a₄ * (x+1)^4 
   + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 + a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) 
  → (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2) := 
by sorry

end coeff_sum_eq_neg_two_l537_537670


namespace max_height_l537_537532

-- Definition of the height function
def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 10

-- Statement of the maximum height problem
theorem max_height : ∃ t : ℝ, height t = 30 :=
by
  sorry

end max_height_l537_537532


namespace closest_integer_sum_l537_537227

theorem closest_integer_sum :
  let S := 500 * (∑ n in finset.range (5000 - 4 + 1 + 1), 1 / ((n + 4)^2 - 9))
  in abs (S - 102) < 1 :=
by
  let S := 500 * (∑ n in finset.range (5000 - 4 + 1 + 1), 1 / ((n + 4)^2 - 9))
  sorry

end closest_integer_sum_l537_537227


namespace expected_value_equals_1_5_l537_537744

noncomputable def expected_value_win (roll : ℕ) : ℚ :=
  if roll = 1 then -1
  else if roll = 4 then -4
  else if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll
  else 0

noncomputable def expected_value_total : ℚ :=
  (1/8 : ℚ) * ((expected_value_win 1) + (expected_value_win 2) + (expected_value_win 3) +
               (expected_value_win 4) + (expected_value_win 5) + (expected_value_win 6) +
               (expected_value_win 7) + (expected_value_win 8))

theorem expected_value_equals_1_5 : expected_value_total = 1.5 := by
  sorry

end expected_value_equals_1_5_l537_537744


namespace tom_is_15_l537_537089

theorem tom_is_15 (T M : ℕ) (h1 : T + M = 21) (h2 : T + 3 = 2 * (M + 3)) : T = 15 :=
by {
  sorry
}

end tom_is_15_l537_537089


namespace max_area_convex_curve_diameter_1_max_area_convex_figure_diameter_width_l537_537501

-- Part (a)
theorem max_area_convex_curve_diameter_1 :
  ∀ (K : convex_shape) (d : ℝ), diameter(K) = 1 → enclosed_area(K) ≤ enclosed_area(circle 1) :=
by sorry

-- Part (b)
theorem max_area_convex_figure_diameter_width (D Δ : ℝ) :
  ∀ (K : convex_shape) (d w : ℝ), diameter(K) = D ∧ width(K) = Δ →
  enclosed_area(K) ≤ enclosed_area(figure_in_diagram_78 D Δ) :=
by sorry

end max_area_convex_curve_diameter_1_max_area_convex_figure_diameter_width_l537_537501


namespace first_problem_second_problem_l537_537115

theorem first_problem
  (a b c : ℝ) 
  (h : ∀ (x : ℝ), x > 0 → ∀ (y : Real), y > 0 → ∀ (z : Real), z > 0 →  
    ( ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ x = (a^2 - c^2)/(sqrtt (c * a)) ∧
                      y = (b^2 - c^2)/(sqrtt (b * c)) ∧ z = (a^2 - b^2) / (sqrtt (a * b) ) ) → 
                      ∃ (a b c: ℝ), 1/ sqrt c = 1/ sqrt a + 1/ sqrt b )): 
  ( c ∃ ( a b c : ℝ ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ( 1/( sqrt c ) = 1/ sqrt a + 1/ sqrt b )) excellent_proof) :
               c inverse_sqrt_ε ∑ c√a

 
theorem second_problem
  (a b c d : ℝ)
  (α β γ δ : ℝ := 1 / a 1 / b 1 /c finite_angle d<|broken_link|>a β^respect 
    4: 
    )--number_of_hours.avl_statement
  h3:
\[ ∀ { }

:=1

end first_problem_second_problem_l537_537115


namespace rectangle_diagonals_equiv_positive_even_prime_equiv_l537_537497

-- Definitions based on problem statement (1)
def is_rectangle (q : Quadrilateral) : Prop := sorry -- "q is a rectangle"
def diagonals_equal_and_bisect (q : Quadrilateral) : Prop := sorry -- "the diagonals of q are equal and bisect each other"

-- Problem statement (1)
theorem rectangle_diagonals_equiv (q : Quadrilateral) :
  (is_rectangle q → diagonals_equal_and_bisect q) ∧
  (diagonals_equal_and_bisect q → is_rectangle q) ∧
  (¬ is_rectangle q → ¬ diagonals_equal_and_bisect q) ∧
  (¬ diagonals_equal_and_bisect q → ¬ is_rectangle q) :=
sorry

-- Definitions based on problem statement (2)
def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def is_prime (n : ℕ) : Prop := sorry -- "n is a prime number"

-- Problem statement (2)
theorem positive_even_prime_equiv (n : ℕ) :
  (is_positive_even n → ¬ is_prime n) ∧
  ((¬ is_prime n → is_positive_even n) = False) ∧
  ((¬ is_positive_even n → is_prime n) = False) ∧
  ((is_prime n → ¬ is_positive_even n) = False) :=
sorry

end rectangle_diagonals_equiv_positive_even_prime_equiv_l537_537497


namespace certain_number_divisibility_l537_537672

-- Define the conditions and the main problem statement
theorem certain_number_divisibility (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % k = 0) (h4 : n = 1) : k = 11 :=
by
  sorry

end certain_number_divisibility_l537_537672


namespace orbit_composition_bound_l537_537007

def orbit {S : Type*} (f : S → S) (x : S) : set S :=
  {y | ∃ n : ℕ, (f^[n]) x = y}  -- The set of points in the orbit of x under f.

def num_orbits {S : Type*} (f : S → S) : ℕ :=
  (setoid.quotient.mk (λ x y, ∃ n : ℕ, (f^[n]) x = y)).to_finset.card  -- Number of distinct orbits.

variables {S : Type*} [fintype S] (n : ℕ) [decidable_eq S]

def c (f : S → S) : ℕ := num_orbits f

theorem orbit_composition_bound {f1 f2 : S → S} (k : ℕ) (fns : fin k → (S → S))
  (H_bij1 : ∀ i, function.bijective (fns i)) :
  let f := (λ x, finset.fold (λ acc i, (fns i) acc) x finset.univ) in
  c (f 0) ≤ n * (k - 1) + c (λ x, finset.fold (λ acc i, (fns i) acc) x finset.univ)
:=
sorry

end orbit_composition_bound_l537_537007


namespace percent_difference_l537_537460

theorem percent_difference
  (X : ℝ) -- third number
  (first_number : ℝ := 0.65 * X) -- 65% of the third number
  (second_number : ℝ := 0.58 * X) -- 58% of the third number) :
  (first_number - second_number) / first_number * 100 ≈ 10.77 := 
sorry

end percent_difference_l537_537460


namespace fred_initial_balloons_l537_537237

def green_balloons_initial (given: Nat) (left: Nat) : Nat := 
  given + left

theorem fred_initial_balloons : green_balloons_initial 221 488 = 709 :=
by
  sorry

end fred_initial_balloons_l537_537237


namespace three_legged_tables_count_l537_537766

theorem three_legged_tables_count (x y : ℕ) (h1 : 3 * x + 4 * y = 23) (h2 : 2 ≤ x) (h3 : 2 ≤ y) : x = 5 := 
sorry

end three_legged_tables_count_l537_537766


namespace no_C2Cl6_produced_l537_537605

theorem no_C2Cl6_produced
  (initial_Cl2 : ℝ) (initial_C2H2 : ℝ) (initial_C2H6 : ℝ)
  (reaction1 : 3 * initial_Cl2 + initial_C2H2 → (initial_C2H2/3) * 2 * (4 * initial_C2H2 / 3))
  (reaction2 : initial_C2H2 + initial_Cl2 → initial_C2H2 + initial_Cl2)
  (reaction3 : initial_C2H2 + 3 * initial_Cl2 → initial_C2H2 + 3 * initial_Cl2)
  (reaction4 : initial_C2H6 + initial_Cl2 → initial_C2H6 + initial_Cl2)
  (total_Cl2 : initial_Cl2 = 6)
  (total_C2H2 : initial_C2H2 = 2)
  (total_C2H6 : initial_C2H6 = 2):
  0 = 0 :=
begin
  sorry
end

end no_C2Cl6_produced_l537_537605


namespace compare_numbers_l537_537195

-- Define the numbers in the problem
def num1 : ℝ := 10 ^ (-49)
def num2 : ℝ := 2 * 10 ^ (-50)
def expected_difference : ℝ := 8 * 10 ^ (-50)

-- Theorem statement without proof
theorem compare_numbers :
  num1 - num2 = expected_difference := by
  sorry

end compare_numbers_l537_537195


namespace min_value_complex_l537_537436

noncomputable def min_value (x y : ℝ) : ℝ := 2 ^ x + 4 ^ y

theorem min_value_complex (x y : ℝ) (hz : |complex.mk x y - complex.I * 4| = |complex.mk x y + 2|) : 
    min_value x y = 4 * real.sqrt 2 :=
by
  sorry

end min_value_complex_l537_537436


namespace inverse_computation_l537_537288

-- Define the function f
def f : ℕ → ℕ
| 2 := 8
| 3 := 15
| 4 := 24
| 5 := 35
| 6 := 48
| _ := 0  -- Default case for completeness

-- Given inverse values from the table
noncomputable def f_inv : ℕ → ℕ
| 8 := 2
| 24 := 4
| 48 := 6
| _ := 0  -- Default case for completeness

-- The theorem to be proved
theorem inverse_computation : 
  f_inv (f_inv 48 * f_inv 8 - f_inv 24) = 2 :=
by
  -- Use 'sorry' since we're skipping the proof
  sorry

end inverse_computation_l537_537288


namespace equilateral_triangle_midpoints_dot_product_l537_537617

theorem equilateral_triangle_midpoints_dot_product 
(triangle_equilateral : ∀ {A B C : Point}, equilateral_triangle A B C → side_length AB = 2)
(midpoints : ∀ {A B C D E F : Point}, midpoint D A B ∧ midpoint E B C ∧ midpoint F C A) :
  (vector.de D E) • (vector.de D F) = 1 / 2 := 
sorry

end equilateral_triangle_midpoints_dot_product_l537_537617


namespace part_a_part_b_l537_537870

/-
Definitions for the elements of the problem
-/
variables {A B C : Type} [euclidean_space A]
variables {A' B' C' : A → A → A → A}
variables (midpoint_B' : A → A → A)
variables (area : ∀ {X Y Z : A}, ℝ)
variables (A_eq_midpoint : ∀ {a b : A}, midpoint_B' a b = a ↔ a = b)

/-
Part a) Theorem: Prove that the area of triangle A'B'C' is less than or equal to half the area of triangle ABC
-/
theorem part_a (h1 : C' = λ A B C, some_point_on_AB)
               (h2 : A' = λ A B C, some_point_on_BC)
               (h3 : B' = midpoint_B' A C)
            : area A' B' C' ≤ area A B C / 2 := sorry

/-
Part b) Theorem: Prove that the area of triangle A'B'C' is equal to one-fourth of the area of triangle ABC if and only if at least one of A' or C' is the midpoint of the corresponding side
-/
theorem part_b (h1 : C' = λ A B C, midpoint_B' A B)
               (h2 : A' = λ A B C, midpoint_B' B C)
               (h3 : B' = midpoint_B' A C)
: (area A' B' C' = area A B C / 4) ↔ 
  (A' = midpoint_B' B C ∨ C' = midpoint_B' A B) := sorry

end part_a_part_b_l537_537870


namespace find_x_y_l537_537035

def a : ℝ × ℝ × ℝ := (2, 4, x)
def b : ℝ × ℝ × ℝ := (2, y, 2)

theorem find_x_y (x y : ℝ) 
  (h1 : (a.fst)^2 + (a.snd).fst^2 + (a.snd.snd)^2 = 36)
  (h2 : a.fst * b.fst + (a.snd).fst * (b.snd).fst + (a.snd.snd) * (b.snd.snd) = 0) :
  x + y = -3 ∨ x + y = 1 :=
sorry

end find_x_y_l537_537035


namespace candies_left_to_share_l537_537709

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end candies_left_to_share_l537_537709


namespace winning_candidate_percentage_votes_l537_537331

theorem winning_candidate_percentage_votes
  (total_votes : ℕ) (majority_votes : ℕ) (P : ℕ) 
  (h1 : total_votes = 6500) 
  (h2 : majority_votes = 1300) 
  (h3 : (P * total_votes) / 100 - ((100 - P) * total_votes) / 100 = majority_votes) : 
  P = 60 :=
sorry

end winning_candidate_percentage_votes_l537_537331


namespace smallest_x_250_multiple_1080_l537_537846

theorem smallest_x_250_multiple_1080 : (∃ x : ℕ, x > 0 ∧ (250 * x) % 1080 = 0) ∧ ¬(∃ y : ℕ, y > 0 ∧ y < 54 ∧ (250 * y) % 1080 = 0) :=
by
  sorry

end smallest_x_250_multiple_1080_l537_537846


namespace cos_double_angle_l537_537306

theorem cos_double_angle (α : ℝ) (h : sin α = (real.sqrt 3) / 3) : cos (2 * α) = 1 / 3 :=
sorry

end cos_double_angle_l537_537306


namespace min_value_f2_eq_zero_range_of_m_l537_537603

-- Definition of the function f
def f (a : ℝ) (x : ℝ) (c : ℝ) := a * x^2 - 2 * x + c

-- Given constraints (conditions)
variables {a c : ℝ} (h_a_pos : a > 0) (h_c_pos : c > 0) (h_ac_one : a * c = 1)

-- Statement (1)
theorem min_value_f2_eq_zero : f a 2 c = 0 :=
  sorry

-- Explicit form when f(2) is minimized
def f_min (x : ℝ) := (1 / 2) * x^2 - 2 * x + 2

-- Statement (2)
theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 2 → f_min x + 4 ≥ m * (x - 2)) → m ≤ 2 * Real.sqrt 2 :=
  sorry

end min_value_f2_eq_zero_range_of_m_l537_537603


namespace ak_length_l537_537247

noncomputable def f : ℝ → ℝ := sorry -- We define it noncomputable for generality

theorem ak_length :
  (∀ x : ℝ, 0 < x → f (10 * x) = 10 * f x) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → f x = x - real.log x) →
  (∀ k : ℤ, let I_k := (set.Ioc (10^k : ℝ) (10^(k+1) : ℝ));
            let D_k := (set.Ioc (10^k : ℝ) (9 * 10^k : ℝ));
            let a_k := (10^k : ℝ);
            (interval_complement_length I_k D_k = a_k)) :=
begin
  intros h1 h2 k,
  let I_k := set.Ioc (10^k : ℝ) (10^(k+1) : ℝ),
  let D_k := set.Ioc (10^k : ℝ) (9 * 10^k : ℝ),
  let a_k := (10^k : ℝ),
  sorry
end

end ak_length_l537_537247


namespace part1_part2_l537_537639

noncomputable def f : ℝ → ℝ → ℝ
| x, a => (x + 1) / x + (a * log x) / (x^2)

def tangent_condition (a : ℝ) : Prop :=
  let df := deriv (λ x => f x a)
  df 1 = a - 1 ∧ (a - 1) * (1 - 1) + (f 1 a) = 0

def maximum_k_condition (k : ℝ) (t0 : ℝ) (h1 : a = 2) (h2 : 1 < t0 ∧ t0 < 5/4) : Prop :=
  (∃ t > 0, f t 2 > k) ∧
  k < (1/t0 + 1/4)^2 + 15/16

theorem part1 (h : tangent_condition 3) : 
  ∃ l : ℝ → ℝ, l 1 = 2 ∧ ∀ x, l x = 2*x :=
sorry

theorem part2 
  {a t0 : ℝ} (h1 : a = 2) (h2 : 1 < t0 ∧ t0 < 5/4) :
  ∃ k : ℝ, maximum_k_condition k t0 h1 h2 ∧ k = 2 :=
sorry

end part1_part2_l537_537639


namespace largest_four_digit_number_last_digit_l537_537522

theorem largest_four_digit_number_last_digit (n : ℕ) (n' : ℕ) (m r a b : ℕ) :
  (1000 * m + 100 * r + 10 * a + b = n) →
  (100 * m + 10 * r + a = n') →
  (n % 9 = 0) →
  (n' % 4 = 0) →
  b = 3 :=
by
  sorry

end largest_four_digit_number_last_digit_l537_537522


namespace find_overhead_expenses_l537_537901

noncomputable def overhead_expenses (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  let total_cost := purchase_price + O in
  let profit := selling_price - total_cost in
  let expected_profit := (profit_percent / 100) * total_cost in
  selling_price - total_cost = expected_profit

theorem find_overhead_expenses (purchase_price selling_price : ℝ) (profit_percent : ℝ) (O : ℝ) :
  purchase_price = 225 ∧ selling_price = 350 ∧ profit_percent = 45.833333333333314 → O ≈ 15 :=
sorry

end find_overhead_expenses_l537_537901


namespace sphere_radius_in_cube_with_tangent_spheres_l537_537997

theorem sphere_radius_in_cube_with_tangent_spheres :
  ∀ (r : ℝ), 
  (∀ sphere_center : ℝ × ℝ × ℝ, sphere_center = (0, 0, 0) -> 
   eight_spheres_tangent_to_cube_faces (2) sphere_center r) -> 
  (r = 1) :=
by
  sorry

def eight_spheres_tangent_to_cube_faces : ℝ -> ℝ × ℝ × ℝ -> ℝ -> Prop 
| side_length, center, radius := 
  let cube_side_half := side_length / 2
  ∧ all_spheres_fit_inside_cube side_length radius
  ∧ all_other_spheres_outer 7 radius
  ∧ center_sphere_at_cube_center center radius

def all_spheres_fit_inside_cube : ℝ -> ℝ -> Prop
| side_length, radius :=
  true -- represents the fact all spheres fit within the cube with given side length

def all_other_spheres_outer : ℕ -> ℝ -> Prop
| sphere_count, radius :=
  true -- represents the outer seven spheres being tangent to center sphere and cube faces

def center_sphere_at_cube_center : ℝ × ℝ × ℝ -> ℝ -> Prop
| center, radius :=
  center = (0, 0, 0)

end sphere_radius_in_cube_with_tangent_spheres_l537_537997


namespace index_card_area_l537_537913

theorem index_card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_shortened_length : (length - 2) * width = 21) : (length * (width - 2)) = 25 := by
  sorry

end index_card_area_l537_537913


namespace find_difference_l537_537021

-- Define the problem conditions in Lean
theorem find_difference (a b : ℕ) (hrelprime : Nat.gcd a b = 1)
                        (hpos : a > b) 
                        (hfrac : (a^3 - b^3) / (a - b)^3 = 73 / 3) :
    a - b = 3 :=
by
    sorry

end find_difference_l537_537021


namespace det_A_divisible_by_2015_fact_l537_537369

-- Define the sequences (a_i) and (k_i)
def a_seq (i : ℕ) : ℕ := sorry  -- We just need to define them formally; specifics aren't necessary for the statement
def k_seq (i : ℕ) : ℕ := sorry  -- We just need to define them formally; specifics aren't necessary for the statement

-- Define the matrix A
noncomputable def A : Matrix (Fin 2015) (Fin 2015) ℕ := 
  λ (i j : Fin 2015), (a_seq i.succ) ^ (k_seq j.succ)

-- State the theorem
theorem det_A_divisible_by_2015_fact : 2015! ∣ (Matrix.det A) :=
  sorry

end det_A_divisible_by_2015_fact_l537_537369


namespace solve_for_x_l537_537318

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 6 ∨ x = - (9 * Real.sqrt 6) :=
by
  sorry

end solve_for_x_l537_537318


namespace hyperbola_satisfies_conditions_l537_537914

-- Define the equations of the hyperbolas as predicates
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def hyperbola_B (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola_C (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- Define the conditions on foci and asymptotes
def foci_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop := 
  h = hyperbola_C ∨ h = hyperbola_D

def has_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)

-- The proof statement
theorem hyperbola_satisfies_conditions :
  foci_on_y_axis hyperbola_D ∧ has_asymptotes hyperbola_D ∧ 
    (¬ (foci_on_y_axis hyperbola_A ∧ has_asymptotes hyperbola_A)) ∧ 
    (¬ (foci_on_y_axis hyperbola_B ∧ has_asymptotes hyperbola_B)) ∧ 
    (¬ (foci_on_y_axis hyperbola_C ∧ has_asymptotes hyperbola_C)) := 
by
  sorry

end hyperbola_satisfies_conditions_l537_537914


namespace thief_speed_l537_537163

-- Definitions for initial conditions
def initial_gap : ℝ := 200 / 1000  -- converted to km
def policeman_speed : ℝ := 10  -- in km/hr
def thief_distance_before_overtaken : ℝ := 800 / 1000  -- converted to km

-- Theorem statement to prove that the speed of the thief is 8 km/hr
theorem thief_speed (V_t : ℝ) (h1 : initial_gap = 0.2) (h2 : policeman_speed = 10) (h3 : thief_distance_before_overtaken = 0.8) 
    (time_eq : (initial_gap + thief_distance_before_overtaken) / policeman_speed = thief_distance_before_overtaken / V_t) : 
  V_t = 8 :=
sorry

end thief_speed_l537_537163


namespace consecutive_negatives_product_sum_l537_537804

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l537_537804


namespace sum_of_consecutive_negative_integers_with_product_3080_l537_537806

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l537_537806


namespace sum_of_intervals_equal_one_l537_537235

theorem sum_of_intervals_equal_one :
  let f (x : ℝ) := (⌊x⌋₊ : ℝ) * (2015 ^ (x - (⌊x⌋₊ : ℝ)) - 1) in
  {x : ℝ | 1 ≤ x ∧ x < 2015 ∧ f x ≤ 1}.to_set.Ico_sum_length = 1 :=
by
  sorry

end sum_of_intervals_equal_one_l537_537235


namespace calc_expr_l537_537201

def diam (a b : ℕ) := (2 * a ^ b) / (b ^ a) * (b ^ a) / (a ^ b)

theorem calc_expr : diam (4, diam (2, 3)) 1 = 2 := by
  sorry

end calc_expr_l537_537201


namespace repeat_is_square_l537_537128

theorem repeat_is_square : ∃ n : ℕ, ∃ k : ℕ, (n = 10^27) ∧ (∃ x : ℕ, n * 10^k + n = x^2) :=
by 
  let n := 10^27
  use n
  use 27
  split
  {
    exact eq.refl n
  }
  {
    use n * 10^27 + n
    sorry
  }

end repeat_is_square_l537_537128


namespace sum_of_possible_values_of_a_l537_537071

theorem sum_of_possible_values_of_a :
  (∀ (a : ℝ), sqrt ((3 * a - 5) ^ 2 + (2 * a - 3) ^ 2) = 
     3 * sqrt 5 → (a = 3.474) ∨ (a = -0.243)) →
  3.474 + -0.243 = 3.231 :=
by
  intros h
  sorry

end sum_of_possible_values_of_a_l537_537071


namespace zhou_yu_age_equation_l537_537325

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation_l537_537325


namespace calc_panel_cost_l537_537911

theorem calc_panel_cost :
  let wall_area := 10 * 7 in 
  let roof_area := 2 * (10 * 6) in
  let total_area := wall_area + roof_area in
  let panel_area := 10 * 15 in
  let panels_needed := Int.toNat (Real.ceil (total_area / panel_area)) in
  let cost_per_panel := 35 in
  panels_needed * cost_per_panel = 70 :=
by
  sorry

end calc_panel_cost_l537_537911


namespace total_days_2001_2005_l537_537304

theorem total_days_2001_2005 : 
  let is_leap_year (y : ℕ) := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365 
  (days_in_year 2001) + (days_in_year 2002) + (days_in_year 2003) + (days_in_year 2004) + (days_in_year 2005) = 1461 :=
by
  sorry

end total_days_2001_2005_l537_537304


namespace min_magnitude_is_sqrt_2_l537_537300

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)
def b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, 2 * t)

-- Define the vector difference a - b
def a_minus_b (t : ℝ) : ℝ × ℝ × ℝ := 
  let (x1, y1, z1) := a t
  let (x2, y2, z2) := b t
  (x1 - x2, y1 - y2, z1 - z2)

-- Define the magnitude of a - b
def magnitude (t : ℝ) : ℝ :=
  let (dx, dy, dz) := a_minus_b t
  real.sqrt (dx^2 + dy^2 + dz^2)

-- The theorem to be proved: The minimum value of the magnitude is √2
theorem min_magnitude_is_sqrt_2 : ∃ t : ℝ, magnitude t = real.sqrt 2 :=
  sorry

end min_magnitude_is_sqrt_2_l537_537300


namespace matrix_inverses_solve_matrix_eq_l537_537259

open Matrix 

variables {A B X : Matrix (Fin 2) (Fin 2) ℚ}  

def A := ![![2, -1], ![-4, 3]]
def B := ![![2, -2], ![-4, 6]]
def X := ![![1, 0], ![0, 2]]
def A_inv := ![![3 / 2, 1 / 2], ![2, 1]]

theorem matrix_inverses (A_inv_correct : A⁻¹ = A_inv) : A_inv = ![![3 / 2, 1 / 2], ![2, 1]] :=
by
  exact A_inv_correct

theorem solve_matrix_eq (H: A⁻¹ = A_inv) : X = A_inv * B :=
by
  have : A⁻¹ = A_inv := H
  sorry

end matrix_inverses_solve_matrix_eq_l537_537259


namespace stork_count_l537_537511

theorem stork_count (B S : ℕ) (h1 : B = 7) (h2 : B = S + 3) : S = 4 := 
by 
  sorry -- Proof to be filled in


end stork_count_l537_537511


namespace A_20_equals_17711_l537_537158

def A : ℕ → ℕ
| 0     => 1  -- by definition, an alternating sequence on an empty set, counting empty sequence
| 1     => 2  -- base case
| 2     => 3  -- base case
| (n+3) => A (n+2) + A (n+1)

theorem A_20_equals_17711 : A 20 = 17711 := 
sorry

end A_20_equals_17711_l537_537158


namespace number_of_assignment_plans_l537_537573

theorem number_of_assignment_plans : 
  let students := ["A", "B", "C", "D"]
  let pavilions := ["B", "C", "D"]
  (∃ (assignments : {s // s = students}), 
     (∀ pavilion in pavilions, ∃ student in assignments, True) ∧ 
     assignments.size = 4 ∧ 
     "A" ∉ assignments("A")) -> 
  (number_of_different_assignment_plans students pavilions = 24) :=
sorry

end number_of_assignment_plans_l537_537573


namespace find_digits_l537_537092

def are_potential_digits (digits : Finset ℕ) : Prop :=
  digits.card = 4 ∧ 
  ∀ (a b c d : ℕ), 
    (a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  let largest := 1000 * a + 100 * b + 10 * c + d,
      smallest := 1000 * d + 100 * c + 10 * b + a in
  a > b ∧ b > c ∧ c > d ∧ largest + smallest = 10477

theorem find_digits : 
  are_potential_digits ({7, 4, 3, 0} : Finset ℕ) := 
sorry

end find_digits_l537_537092


namespace gcd_lcm_product_135_l537_537558

theorem gcd_lcm_product_135 (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 135 :=
by
  sorry

end gcd_lcm_product_135_l537_537558


namespace average_marks_l537_537120

/-- Shekar scored 76, 65, 82, 67, and 85 marks in Mathematics, Science, Social Studies, English, and Biology respectively.
    We aim to prove that his average marks are 75. -/

def marks : List ℕ := [76, 65, 82, 67, 85]

theorem average_marks : (marks.sum / marks.length) = 75 := by
  sorry

end average_marks_l537_537120


namespace domain_of_sqrt_sum_l537_537224

def is_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x = √(x - 3) + √(9 - x) → a ≤ x ∧ x ≤ b

theorem domain_of_sqrt_sum :
  is_domain (λ x => √(x - 3) + √(9 - x)) 3 9 :=
sorry

end domain_of_sqrt_sum_l537_537224


namespace distinct_s_average_is_28_over_3_l537_537279

theorem distinct_s_average_is_28_over_3 (a b : ℕ) (s : ℕ) :
  (∀ a b, a + b = 7 → a * b = s) →
  (∀ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ a < 7 ∧ b < 7) →
  let distinct_s_vals : finset ℕ := {6, 10, 12} in
  ↑(distinct_s_vals.sum id) / distinct_s_vals.card = 28 / 3 :=
by
  intros h1 h2
  sorry

end distinct_s_average_is_28_over_3_l537_537279


namespace find_PC_l537_537090

-- Define the main geometric entities and conditions
variables (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (pa pb pc : ℝ)
variables (angle_APB angle_BPC angle_CPA : ℝ)

-- Conditions for the problem
def is_right_triangle (angle_B : ℝ) : Prop := angle_B = 90
def distances := (pa = 10) ∧ (pb = 6)
def equal_angles := (angle_APB = 120) ∧ (angle_BPC = 120) ∧ (angle_CPA = 120)

-- The theorem to prove
theorem find_PC (h1 : is_right_triangle 90)
    (h2 : distances)
    (h3 : equal_angles) :
    pc = 33 :=
sorry

end find_PC_l537_537090


namespace common_region_area_of_triangles_l537_537095

noncomputable def area_of_common_region (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

theorem common_region_area_of_triangles (a : ℝ) (h : 0 < a) : 
  area_of_common_region a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 :=
by
  sorry

end common_region_area_of_triangles_l537_537095


namespace eq_solutions_if_and_only_if_m_eq_p_l537_537008

theorem eq_solutions_if_and_only_if_m_eq_p (p m : ℕ) (hp : p.prime) (hm : 2 ≤ m) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ 1 ∧ y ≠ 1 ∧ (x^p + y^p) / 2 = ((x + y) / 2)^m) ↔ m = p :=
by
  sorry

end eq_solutions_if_and_only_if_m_eq_p_l537_537008


namespace other_coin_denomination_l537_537823

theorem other_coin_denomination :
  ∀ (total_coins : ℕ) (value_rs : ℕ) (paise_per_rs : ℕ) (num_20_paise_coins : ℕ) (total_value_paise : ℕ),
  total_coins = 324 →
  value_rs = 71 →
  paise_per_rs = 100 →
  num_20_paise_coins = 200 →
  total_value_paise = value_rs * paise_per_rs →
  (∃ (denom_other_coin : ℕ),
    total_value_paise - num_20_paise_coins * 20 = (total_coins - num_20_paise_coins) * denom_other_coin
    → denom_other_coin = 25) :=
by
  sorry

end other_coin_denomination_l537_537823


namespace find_P_l537_537626

noncomputable def pointP_coordinates (N M P : ℝ × ℝ) : Prop := 
  SymmetricAboutX M P ∧ SymmetricAboutY N M ∧ N = (1, 2) ∧ P = (-1, -2)

theorem find_P (N M P : ℝ × ℝ) (h1 : SymmetricAboutX M P)
  (h2 : SymmetricAboutY N M) (h3 : N = (1, 2)) : 
  P = (-1, -2) :=
sorry

end find_P_l537_537626


namespace same_gender_probability_l537_537764

-- Define the total number of teachers in School A and their gender distribution.
def schoolA_teachers : Nat := 3
def schoolA_males : Nat := 2
def schoolA_females : Nat := 1

-- Define the total number of teachers in School B and their gender distribution.
def schoolB_teachers : Nat := 3
def schoolB_males : Nat := 1
def schoolB_females : Nat := 2

-- Calculate the probability of selecting two teachers of the same gender.
theorem same_gender_probability :
  (schoolA_males * schoolB_males + schoolA_females * schoolB_females) / (schoolA_teachers * schoolB_teachers) = 4 / 9 :=
by
  sorry

end same_gender_probability_l537_537764


namespace problem_a_problem_b_problem_c_problem_d_problem_e_problem_f_problem_g_l537_537733

variables (A B C M : Type)
variables (a b c x y z u v w S R r : ℝ)
variables (dist_from : M → A → ℝ)
variables (dist_to_side : M → A → ℝ)

variables (MA : M → A)
variables (MB : M → B)
variables (MC : M → C)

-- Conditions
-- distance from M to A, B, C denoted as x, y, z
axiom ax : dist_from M A = x
axiom ay : dist_from M B = y
axiom az : dist_from M C = z

-- distance from M to sides BC, CA, AB denoted as u, v, w
axiom au : dist_to_side M B = u
axiom av : dist_to_side M C = v
axiom aw : dist_to_side M A = w

-- sides a, b, c
axiom a_side : a = dist_to_side A B
axiom b_side : b = dist_to_side B C
axiom c_side : c = dist_to_side C A

-- area S
axiom area_S : S = some real number representing area of ABC

-- radii R and r
axiom circumscribed_R : R = some real number representing circumradius
axiom inscribed_r : r = some real number representing inradius

-- Proof statements
theorem problem_a : a * x + b * y + c * z ≥ 4 * S := sorry
theorem problem_b : x + y + z ≥ 2 * (u + v + w) := sorry
theorem problem_c : x * u + y * v + z * w ≥ 2 * (u * v + v * w + w * u) := sorry
theorem problem_d : 2 * (1/x + 1/y + 1/z) ≤ 1/u + 1/v + 1/w := sorry
theorem problem_e : x * y * z ≥ (R / (2 * r)) * (u + v) * (v + w) * (w + u) := sorry
theorem problem_f : x * y * z ≥ (4 * R / r) * u * v * w := sorry
theorem problem_g : x * y + y * z + z * x ≥ (2 * R / r) * (u * v + v * w + w * u) := sorry

end problem_a_problem_b_problem_c_problem_d_problem_e_problem_f_problem_g_l537_537733


namespace divide_figure_into_equal_parts_l537_537415

noncomputable theory
open_locale classical

def figure : Type := sorry -- Placeholder for the actual figure type

-- Condition: Each part consists of two squares and one triangle.
def is_equal_part (part : figure) : Prop :=
  (number_of_squares part = 2) ∧ (number_of_triangles part = 1)

-- Function to get parts by cutting the figure
def cut_figure_into_parts (f : figure) : list figure := sorry

-- Check if parts can be superimposed onto each other
def parts_are_superimposable (parts : list figure) : Prop :=
  ∀ p₁ p₂ ∈ parts, p₁ = p₂ ∨ (flip p₁ = p₂ ∨ rotate p₁ = p₂)

theorem divide_figure_into_equal_parts (f : figure) :
  ∃ parts : list figure, 
    (length parts = 5) ∧ 
    (∀ p ∈ parts, is_equal_part p) ∧ 
    (parts_are_superimposable parts) :=
begin
  sorry
end

end divide_figure_into_equal_parts_l537_537415


namespace book_arrangement_l537_537041

def arrangements_with_english_side_by_side (korean_workbooks english_workbooks : ℕ) : ℕ :=
  if korean_workbooks = 2 ∧ english_workbooks = 2 then
    3! * 2!
  else
    0

theorem book_arrangement : arrangements_with_english_side_by_side 2 2 = 12 :=
by
  simp [arrangements_with_english_side_by_side]
  sorry

end book_arrangement_l537_537041


namespace students_taking_all_three_classes_l537_537821

open Nat

theorem students_taking_all_three_classes 
  (total_students : ℕ) 
  (drawing_students : ℕ) 
  (chess_students : ℕ) 
  (music_students : ℕ) 
  (students_taking_at_least_two : ℕ) 
  (non_empty_classes : total_students ≥ drawing_students ∧ total_students ≥ chess_students ∧ total_students ≥ music_students ∧ total_students > 0)
  (all_students_taking_at_least_one : drawing_students + chess_students + music_students - students_taking_at_least_two = total_students) :
  let students_taking_all_three := 
    drawing_students + chess_students + music_students - students_taking_at_least_two - total_students in
  students_taking_all_three = 2 :=
by
  sorry

end students_taking_all_three_classes_l537_537821


namespace no_integer_roots_l537_537606

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem no_integer_roots 
  (f : ℤ → ℤ)
  (H_poly : ∃ n a, (λ x, a 0 * x ^ n + a 1 * x ^ (n - 1) + a (n - 1) * x + a n) = f)
  (α : ℤ)
  (H_odd_α : is_odd α)
  (β : ℤ)
  (H_even_β : is_even β)
  (H_odd_f_α : is_odd (f α))
  (H_odd_f_β : is_odd (f β)) :
  ∀ γ : ℤ, f γ ≠ 0 :=
by
  sorry

end no_integer_roots_l537_537606


namespace max_value_of_f_l537_537596

noncomputable def f (m x : ℝ) : ℝ := (4 - 3 * m) * x^2 - 2 * x + m

theorem max_value_of_f (m : ℝ) : 
  ∃ y_max : ℝ, (y_max = (if m < (2 / 3) then 2 - 2 * m else m) ∧ 
  ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f m x ≤ y_max) :=
by
  sorry

end max_value_of_f_l537_537596


namespace ellipse_hyperbola_foci_concide_eq_b_squared_l537_537791

theorem ellipse_hyperbola_foci_concide_eq_b_squared :
  let ellipse (b^2 : ℝ) := ∀ x y : ℝ, x^2 / 16 + y^2 / b^2 = 1 →
    (|x^2| + |y^2|)^2 / 16 + y^2 / b^2 - 1 = 0 -- Using normalized form.
  let hyperbola := ∀ x y : ℝ, x^2 / (144 / 25) - y^2 / (81 / 25) = 1 → 
    (|x^2| - |y^2|)^2 / (144 / 25) - y^2 / (81 / 25) - 1 = 0 -- Normalized form.
  in (∀ x y, ellipse x y → hyperbola x y ) → ellipse 7 sorry

end ellipse_hyperbola_foci_concide_eq_b_squared_l537_537791


namespace rectangle_perimeter_at_least_l537_537536

theorem rectangle_perimeter_at_least (m : ℕ) (m_pos : 0 < m) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a * b ≥ 1 / (m * m) ∧ 2 * (a + b) ≥ 4 / m) := sorry

end rectangle_perimeter_at_least_l537_537536


namespace price_of_coffee_increased_by_300_percent_l537_537437

theorem price_of_coffee_increased_by_300_percent
  (P : ℝ) -- cost per pound of milk powder and coffee in June
  (h1 : 0.20 * P = 0.20) -- price of a pound of milk powder in July
  (h2 : 1.5 * 0.20 = 0.30) -- cost of 1.5 lbs of milk powder in July
  (h3 : 6.30 - 0.30 = 6.00) -- cost of 1.5 lbs of coffee in July
  (h4 : 6.00 / 1.5 = 4.00) -- price per pound of coffee in July
  : ((4.00 - 1.00) / 1.00) * 100 = 300 := 
by 
  sorry

end price_of_coffee_increased_by_300_percent_l537_537437


namespace polynomial_degree_is_14_l537_537098

noncomputable def polynomial_degree (a b c d e f g h : ℝ) : ℕ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 then 14 else 0

theorem polynomial_degree_is_14 (a b c d e f g h : ℝ) (h_neq0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  polynomial_degree a b c d e f g h = 14 :=
by sorry

end polynomial_degree_is_14_l537_537098


namespace xy_zero_l537_537836

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 :=
by
  sorry

end xy_zero_l537_537836


namespace kindergarten_students_percentage_is_correct_l537_537179

-- Definitions based on conditions
def total_students_annville : ℕ := 150
def total_students_cleona : ℕ := 250
def percent_kindergarten_annville : ℕ := 14
def percent_kindergarten_cleona : ℕ := 10

-- Calculation of number of kindergarten students
def kindergarten_students_annville : ℕ := total_students_annville * percent_kindergarten_annville / 100
def kindergarten_students_cleona : ℕ := total_students_cleona * percent_kindergarten_cleona / 100
def total_kindergarten_students : ℕ := kindergarten_students_annville + kindergarten_students_cleona
def total_students : ℕ := total_students_annville + total_students_cleona
def kindergarten_percentage : ℚ := (total_kindergarten_students * 100) / total_students

-- The theorem to be proved using the conditions
theorem kindergarten_students_percentage_is_correct : kindergarten_percentage = 11.5 := by
  sorry

end kindergarten_students_percentage_is_correct_l537_537179


namespace emery_family_first_hour_distance_l537_537998

noncomputable def total_time : ℝ := 4
noncomputable def remaining_distance : ℝ := 300
noncomputable def first_hour_distance : ℝ := 100

theorem emery_family_first_hour_distance :
  (remaining_distance / (total_time - 1)) = first_hour_distance :=
sorry

end emery_family_first_hour_distance_l537_537998


namespace orthocenters_concyclic_l537_537024

-- Auxiliary assumptions and definitions
variables {A1 A2 A3 A4 O : Type} [has_vector_space A1] [has_vector_space A2]
          [has_vector_space A3] [has_vector_space A4] [has_vector_space O]

structure CyclicQuadrilateral (A1 A2 A3 A4 O : Type) :=
  (circle : is_circle O)
  (inscribed : is_inscribed A1 A2 A3 A4)

structure Orthocenter (A B C D : Type) :=
  (H1 : is_orthocenter A A(B C D))
  (H2 : is_orthocenter B B(C D A))
  (H3 : is_orthocenter C C(D A B))
  (H4 : is_orthocenter D D(A B C))

-- Problem statement
theorem orthocenters_concyclic
  (A1 A2 A3 A4 : Type) (O : Type)
  (H1 H2 H3 H4 : Type)
  [cyclic_quad : CyclicQuadrilateral A1 A2 A3 A4 O]
  [orthocenters : Orthocenter A1 A2 A3 A4] :
  ∃ (C : Type) (R : ℝ), is_concyclic H1 H2 H3 H4 C R := 
sorry

end orthocenters_concyclic_l537_537024


namespace ellipse_equation_l537_537634

open Real

theorem ellipse_equation :
  ∃ (a b : ℝ), (∃ (c : ℝ), a > 0 ∧ b > 0 ∧ 2 * c + 0 - 2 = 0 ∧ a^2 = b^2 + c^2 ∧ b = 2 ∧ c = 1)
  ∧ (∀ x y : ℝ, 2 * x + y - 2 = 0) 
  ∧ (\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1) 
  → (\frac{x^2}{5} + \frac{y^2}{4} = 1) :=
by
  sorry

end ellipse_equation_l537_537634


namespace length_of_path_CDE_l537_537059

theorem length_of_path_CDE (a b c : ℝ) (h : a^2 + b^2 = c^2) 
  (h_b : b < a) (h_angle_right : ∀ C (triangle ABC), ∠C = 90) :
  let D = point_on_circumcircle_angle_bisector (triangle ABC) C
  let E = point_on_circumcircle_altitude (triangle ABC) C
  path_length_CDE (triangle ABC) D E = b * Real.sqrt 2 :=
sorry

end length_of_path_CDE_l537_537059


namespace cos_60_eq_one_half_l537_537942

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537942


namespace at_least_one_not_less_than_one_l537_537597

variable (x : ℝ)
def a : ℝ := x^2 + 1/2
def b : ℝ := 2 - x
def c : ℝ := x^2 - x + 1

theorem at_least_one_not_less_than_one (x : ℝ) : ¬ (a x < 1 ∧ b x < 1 ∧ c x < 1) := by
  sorry

end at_least_one_not_less_than_one_l537_537597


namespace fraction_expression_equiv_l537_537112

theorem fraction_expression_equiv:
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := 
by 
  sorry

end fraction_expression_equiv_l537_537112


namespace regular_polygon_sides_l537_537979

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537979


namespace maximum_good_isosceles_triangles_l537_537385

theorem maximum_good_isosceles_triangles (P : Type) [polygon P] (n : ℕ) :
  is_regular_polygon P 2006 → 
  (∀ e : edge P, good_edge e ↔ divides_odd_parts e) →
  non_intersecting_diagonals P 2003 →
  maximum_isosceles_triangles_with_two_good_edges 2003 1003 :=
begin
  sorry
end

end maximum_good_isosceles_triangles_l537_537385


namespace candy_initial_count_l537_537455

theorem candy_initial_count :
  ∀ (T S R : ℕ), 
  T = 108 →
  S = 153 →
  R = 88 →
  (T + S + R = 349) := 
by
  intros T S R hT hS hR
  rw [hT, hS, hR]
  norm_num

end candy_initial_count_l537_537455


namespace tan_sum_pi_eighths_l537_537771

theorem tan_sum_pi_eighths : (Real.tan (Real.pi / 8) + Real.tan (3 * Real.pi / 8) = 2 * Real.sqrt 2) :=
by
  sorry

end tan_sum_pi_eighths_l537_537771


namespace arthur_walked_total_distance_l537_537550

def total_distance_walked : ℝ :=
  let blocks_east := 8
  let blocks_north := 10
  let blocks_west := 4
  let total_blocks := blocks_east + blocks_north + blocks_west
  let miles_per_block := 1 / 4
  total_blocks * miles_per_block

theorem arthur_walked_total_distance : total_distance_walked = 5.5 := by
  sorry

end arthur_walked_total_distance_l537_537550


namespace power_calculation_l537_537923

theorem power_calculation :
  ((8^5 / 8^3) * 4^6) = 262144 := by
  sorry

end power_calculation_l537_537923


namespace problem1_problem2_l537_537091

-- Part (1)
theorem problem1 (a b c d : ℕ) (h1 : a = 7) (h2 : b = 12) (h3 : c = 9) (h4 : d = 12) :
  c * b - a * b = 24 :=
by
  simp [h1, h2, h3, h4]
  sorry

-- Part (2)
theorem problem2 (a b c d : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 5) (h4 : d = 9) :
  (b - a) * (d - c) = 24 :=
by
  simp [h1, h2, h3, h4]
  sorry

end problem1_problem2_l537_537091


namespace cos_60_degrees_is_one_half_l537_537937

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537937


namespace simplify_and_evaluate_l537_537772

variable (x y : ℚ)
variable (expr : ℚ := 3 * x * y^2 - (x * y - 2 * (2 * x * y - 3 / 2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y)

theorem simplify_and_evaluate (h1 : x = 3) (h2 : y = -1 / 3) : expr = -3 :=
by
  sorry

end simplify_and_evaluate_l537_537772


namespace num_distinct_products_is_309_l537_537012

def divisor_set (n : ℕ) : set ℕ := {d | d > 0 ∧ d ∣ n}

def T : set ℕ := divisor_set 48000

def num_products_of_distinct_elements (s : set ℕ) : ℕ :=
  (s.to_finset.filter (λ pair, let (a, b) := pair in a ≠ b ∧ (a * b) ∈ s)).card

theorem num_distinct_products_is_309 :
  num_products_of_distinct_elements (diagonal_prod (T \ {(t : ℕ × ℕ) | t.1 = t.2})) = 309 := sorry

end num_distinct_products_is_309_l537_537012


namespace f_2007_2008_l537_537593

-- Define the conditions
def f : ℕ+ × ℕ+ → ℕ := sorry
axiom f_1_1 : f (1, 1) = 1
axiom f_nat_star : ∀ m n : ℕ+, f (m, n) ∈ ℕ+
axiom f_rec1 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2
axiom f_rec2 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1)

-- The goal to prove
theorem f_2007_2008 : f (2007, 2008) = 2 ^ 2006 + 4014 := sorry

end f_2007_2008_l537_537593


namespace irrational_option_C_l537_537494

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem irrational_option_C :
  ¬ is_rational (√2) ∧ is_rational 0.7 ∧ is_rational (1/2) ∧ is_rational (√9) := 
by {
  sorry
}

end irrational_option_C_l537_537494


namespace area_of_rectangle_l537_537502

noncomputable def area_rect (a c : ℝ) : ℝ := 
  let b := Real.sqrt (c^2 - a^2)
  a * b

theorem area_of_rectangle : area_rect 15 18 = 45 * Real.sqrt 11 := 
by
  unfold area_rect
  norm_num
  rw [Real.sqrt_sub (show 324 ≥ 225, by norm_num)]
  sorry

end area_of_rectangle_l537_537502


namespace arithmetic_progression_properties_range_of_lambda_l537_537608

open Nat

-- Given conditions in Lean 4:
def a (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℕ := n * (2 * a n)
def b (n : ℕ) : ℕ := 1 / (a n * a (n + 1))
def T (n : ℕ) : ℕ := (finset.Icc 1 n).sum fun k => b k

-- Statement to prove the arithmetic progression properties and T_n value.
theorem arithmetic_progression_properties (n : ℕ) (hn : n > 0) :
  a 1 = 1 ∧ ∀ n > 0, (a n)^2 = S (2*n-1) ∧ T n = n / (2*n+1) :=
by
  sorry

-- Statement to prove the range of λ.
theorem range_of_lambda (λ : ℝ) (n : ℕ) (hn : n > 0) :
  λ T n < n + 8*(-1)^n ↔ λ < -21 :=
by
  sorry

end arithmetic_progression_properties_range_of_lambda_l537_537608


namespace cos_60_eq_sqrt3_div_2_l537_537928

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537928


namespace area_of_union_12_8_8_l537_537898

noncomputable def area_of_union (l w r : ℝ) : ℝ :=
  let rect_area := l * w
  let circle_area := π * r^2
  let overlap_area := (1/4) * circle_area
  in rect_area + circle_area - overlap_area

theorem area_of_union_12_8_8 :
  area_of_union 12 8 8 = 96 + 48 * π := by
  sorry

end area_of_union_12_8_8_l537_537898


namespace maximize_profit_l537_537858

-- Defining necessary conditions
def buy_price := 0.60
def sell_price := 0.80
def return_price := 0.40

def days_in_month := 30
def high_demand_days := 20
def high_demand_copies := 100
def low_demand_days := 10
def low_demand_copies := 70

-- Defining maximum profit function
def monthly_profit (x : ℕ) : ℝ :=
  high_demand_days * (sell_price - buy_price) * x + 
  low_demand_days * (low_demand_copies * (sell_price - buy_price) - (buy_price - return_price) * (x - low_demand_copies))

-- Stating the theorem
theorem maximize_profit :
  ∃ x : ℕ, x = 100 ∧ monthly_profit 100 = 480 :=
by {
  sorry -- The proof is omitted as instructed
}

end maximize_profit_l537_537858


namespace tabitha_honey_nights_l537_537057

def servings_per_cup := 1
def cups_per_night := 2
def ounces_per_container := 16
def servings_per_ounce := 6
def total_servings := servings_per_ounce * ounces_per_container
def servings_per_night := servings_per_cup * cups_per_night
def number_of_nights := total_servings / servings_per_night

theorem tabitha_honey_nights : number_of_nights = 48 :=
by
  -- Proof to be provided.
  sorry

end tabitha_honey_nights_l537_537057


namespace blonde_hair_ratio_l537_537518

theorem blonde_hair_ratio (r b bl : ℕ) (hr hb hbl : ℕ) (hbnumber : ℕ) 
  (total_kids : ℕ)
  (h_ratio : r:b:bl = 3:6:7)
  (h_r : r = 9)
  (h_total : total_kids = 48)
  (h_bl : bl = hbnumber * b)
  (hb_eq : hbnumber = 6)
  : (hbnumber / total_kids) = (3 / 8) := 
by
  -- The full proof would start here, but for now, we use sorry.
  sorry

end blonde_hair_ratio_l537_537518


namespace apple_in_box_B_l537_537826

def note_on_A (box_with_apple : char) : Prop := box_with_apple = 'A'
def note_on_B (box_with_apple : char) : Prop := box_with_apple ≠ 'B'
def note_on_C (box_with_apple : char) : Prop := box_with_apple ≠ 'A'
def only_one_true (P Q R : Prop) : Prop := (P ∧ ¬ Q ∧ ¬ R) ∨ (¬ P ∧ Q ∧ ¬ R) ∨ (¬ P ∧ ¬ Q ∧ R)

theorem apple_in_box_B (box_with_apple : char) :
  (only_one_true (note_on_A box_with_apple) (note_on_B box_with_apple) (note_on_C box_with_apple)) →
  box_with_apple = 'B' :=
by
  sorry

end apple_in_box_B_l537_537826


namespace radius_circles_l537_537698

variables (d1 d2 y r : ℝ)
def r1 := d1 / 2
def r2 := d2 / 2

theorem radius_circles (h1 : (r + r1) ^ 2 = (r - 2 * r2 - r1) ^ 2 + y ^ 2)
                       (h2 : (r + r2) ^ 2 = (r - r2) ^ 2 + y ^ 2) :
  r = ((d1 + d2) * d2) / (2 * d1) ∨ r = ((d1 + d2) * d1) / (2 * d2) :=
sorry

end radius_circles_l537_537698


namespace smallest_base10_integer_l537_537103

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l537_537103


namespace appropriate_chart_for_decreasing_prices_l537_537037

open Classical

-- Definitions
def represents_percentage (chart : Type) : Prop :=
∀ data : List ℕ, chart.chart_type = "Pie" → 
(data.nonempty ∧ ∀ x, x ∈ data → 0 ≤ x ∧ x ≤ 100) → 
∃ total_percentage: ℕ, total_percentage = 100

def represents_changes (chart : Type) : Prop :=
∀ data : List ℕ, chart.chart_type = "Line" → 
(data.size ≥ 2 ∧ ∀ i, 0 < i ∧ i < (data.size - 1) → data.nth i - data.nth (i - 1) < 0) → 
True

def represents_numbers (chart : Type) : Prop :=
∀ data : List ℕ, chart.chart_type = "Bar" → 
data.nonempty → True

-- Theorem statement
theorem appropriate_chart_for_decreasing_prices 
  (chart : Type) 
  (facts : represents_percentage chart ∧ represents_changes chart ∧ represents_numbers chart) 
  (yearly_prices : List ℕ) 
  (decreasing_prices : ∀ i, 0 < i ∧ i < (yearly_prices.size - 1) → yearly_prices.nth i - yearly_prices.nth (i - 1) < 0) :
  chart.chart_type = "Line" :=
sorry

end appropriate_chart_for_decreasing_prices_l537_537037


namespace third_butcher_packages_l537_537443

theorem third_butcher_packages :
  ∀ (weight_per_package : ℕ) (first_packages : ℕ) (second_packages : ℕ) (total_weight : ℕ),
  weight_per_package = 4 →
  first_packages = 10 →
  second_packages = 7 →
  total_weight = 100 →
  let third_packages := (total_weight - (first_packages * weight_per_package + second_packages * weight_per_package)) / weight_per_package in
  third_packages = 8 := 
by
  intros weight_per_package first_packages second_packages total_weight
  assume h_wpp h_fp h_sp h_tw
  let third_packages := (total_weight - (first_packages * weight_per_package + second_packages * weight_per_package)) / weight_per_package
  sorry

end third_butcher_packages_l537_537443


namespace algebraic_expression_value_l537_537309

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) : a^2 - b^2 - 4*a = -4 := 
sorry

end algebraic_expression_value_l537_537309


namespace regular_polygon_sides_l537_537995

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537995


namespace graphs_with_inverses_l537_537961

-- Definitions of the graphs
def Graph_F : Prop := ∀ y : ℝ, ∃! x : ℝ, -4 ≤ x ∧ x ≤ 4 ∧ (x = -4 ∨ x = 4 ∨ y = 0)
def Graph_G : Prop := ∀ x : ℝ, -(x - 3)*(x + 3) 
def Graph_H : Prop := ∀ x : ℝ, x
def Graph_I : Prop := ∀ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ (x^2 + y^2 = 16)
def Graph_J : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ (x^2 + y^2 = 16)
def Graph_K : Prop := ∀ x : ℝ, (x^3/60 + x^2/30 - x/3 + 1)

-- Problem statement
theorem graphs_with_inverses : 
    Graph_F ∧ 
    ¬Graph_G ∧ 
    Graph_H ∧ 
    Graph_I ∧ 
    Graph_J ∧ 
    ¬Graph_K → 
    { Graph_F, Graph_H, Graph_I, Graph_J } :=
by { sorry }

end graphs_with_inverses_l537_537961


namespace hyperbola_equation_l537_537647

-- Definitions of the given conditions
def is_hyperbola (a b : ℝ) := a > 0 ∧ b > 0 ∧ 
  ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1)

-- Definition of the focal length and asymptotes conditions
def hyperbola_conditions (a b : ℝ) := 
  2 * real.sqrt (5: ℝ) = real.sqrt (a^2 + b^2) ∧
  b / a = 2

-- The target statement to prove
theorem hyperbola_equation (a b : ℝ) 
    (h_cond: hyperbola_conditions a b) 
    (h_hyper: is_hyperbola a b) :
  a ^ 2 = 4 ∧ b ^ 2 = 16 := 
sorry

end hyperbola_equation_l537_537647


namespace avg_distinct_s_l537_537274

theorem avg_distinct_s (s : ℤ) (hpq : ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p + q = 7 ∧ p * q = s) :
  (∑ (p q : ℤ) in {1, 2, 3, 4, 5, 6}, if p + q = 7 then p * q else 0).erase 0 = {6, 10, 12} →
  s = ∑ x in {6, 10, 12}, x / {6, 10, 12}.size := sorry

end avg_distinct_s_l537_537274


namespace cos_60_degrees_is_one_half_l537_537938

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537938


namespace closest_integer_sum_l537_537226

theorem closest_integer_sum : 
  let s := 500 * (∑ n in Finset.range (5000 - 3) \ Finset.range 1, 1 / (n + 4)^2 - 9)
  ∃ (k : ℤ), k = 174 ∧ abs (s - k) = abs (s - 174) := 
sorry

end closest_integer_sum_l537_537226


namespace reflect_y_axis_correct_l537_537787

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l537_537787


namespace triangle_incircle_tangency_angles_l537_537153

theorem triangle_incircle_tangency_angles (α β : ℝ) (h₁ : α + β = 90) :
  ∃ (D E F : Type), ∠D = 45 ∧ ∠F + ∠E = 135 ∧ (∠F = 90 ∧ ∠E > 45) :=
by
  sorry

end triangle_incircle_tangency_angles_l537_537153


namespace eighth_positive_odd_multiple_of_5_l537_537479

theorem eighth_positive_odd_multiple_of_5 : 
  let a := 5 in 
  let d := 10 in 
  let n := 8 in 
  a + (n - 1) * d = 75 :=
by
  let a := 5
  let d := 10
  let n := 8
  have : a + (n - 1) * d = 75 := by 
    calc
      a + (n - 1) * d = 5 + (8 - 1) * 10  : by rfl
      ... = 5 + 70                          : by rfl
      ... = 75                              : by rfl
  exact this

end eighth_positive_odd_multiple_of_5_l537_537479


namespace certain_number_is_63_l537_537149

theorem certain_number_is_63 (initial_number : ℕ) (h_initial : initial_number = 6) :
  let doubled := initial_number * 2,
      added := doubled + 9,
      trebled := added * 3
  in trebled = 63 :=
by
  have h_doubled : doubled = 6 * 2 := by rw [h_initial]; sorry,
  have h_added : added = 12 + 9 := by rw [h_doubled]; sorry,
  have h_trebled : trebled = 21 * 3 := by rw [h_added]; sorry,
  exact sorry

end certain_number_is_63_l537_537149


namespace truncated_pyramid_angle_l537_537797

def angle_between_height_and_lateral_edge (α : ℝ) : ℝ :=
  Real.arctan (2 * Real.cot α)

theorem truncated_pyramid_angle (α : ℝ) (hα : α < π / 2) : 
  ∃ β : ℝ, β = angle_between_height_and_lateral_edge α :=
by
  use Real.arctan (2 * Real.cot α)
  sorry

end truncated_pyramid_angle_l537_537797


namespace five_y_eq_45_over_7_l537_537305

theorem five_y_eq_45_over_7 (x y : ℚ) (h1 : 3 * x + 4 * y = 0) (h2 : x = y - 3) : 5 * y = 45 / 7 := by
  sorry

end five_y_eq_45_over_7_l537_537305


namespace new_avg_weight_l537_537824

-- Definition of the conditions
def original_team_avg_weight : ℕ := 94
def original_team_size : ℕ := 7
def new_player_weight_1 : ℕ := 110
def new_player_weight_2 : ℕ := 60
def total_new_team_size : ℕ := original_team_size + 2

-- Computation of the total weight
def total_weight_original_team : ℕ := original_team_avg_weight * original_team_size
def total_weight_new_team : ℕ := total_weight_original_team + new_player_weight_1 + new_player_weight_2

-- Statement of the theorem
theorem new_avg_weight : total_weight_new_team / total_new_team_size = 92 := by
  -- Proof is omitted
  sorry

end new_avg_weight_l537_537824


namespace least_three_digit_integer_with_factors_3_7_11_l537_537844

theorem least_three_digit_integer_with_factors_3_7_11 : ∃ n : ℕ, n = 231 ∧ 100 ≤ n ∧ n < 1000 ∧ (3 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) :=
by
  use 231
  split
  { refl }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  norm_num
  use sorry

end least_three_digit_integer_with_factors_3_7_11_l537_537844


namespace divisibility_by_29_and_29pow4_l537_537004

theorem divisibility_by_29_and_29pow4 (x y z : ℤ) (h : 29 ∣ (x^4 + y^4 + z^4)) : 29^4 ∣ (x^4 + y^4 + z^4) :=
by
  sorry

end divisibility_by_29_and_29pow4_l537_537004


namespace workers_problem_l537_537859

theorem workers_problem (W : ℕ) (A : ℕ) :
  (W * 45 = A) ∧ ((W + 10) * 35 = A) → W = 35 :=
by
  sorry

end workers_problem_l537_537859


namespace min_value_frac_sum_l537_537338

theorem min_value_frac_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 4) : 
  (4 / a^2 + 1 / b^2) ≥ 9 / 4 :=
by
  sorry

end min_value_frac_sum_l537_537338


namespace dispatch_plans_count_l537_537831

theorem dispatch_plans_count
  (students : Finset ℕ)
  (h_card : students.card = 6) :
  let sunday_selection := students.choose 2,
      friday_selection := students \ sunday_selection,
      saturday_selection := friday_selection \ (∅ : Finset ℕ)
  in sunday_selection.card * friday_selection.card * saturday_selection.card = 180 :=
by
  sorry

end dispatch_plans_count_l537_537831


namespace smallest_five_digit_palindrome_div_4_thm_l537_537750

def is_palindrome (n : ℕ) : Prop :=
  n = (n % 10) * 10000 + ((n / 10) % 10) * 1000 + ((n / 100) % 10) * 100 + ((n / 1000) % 10) * 10 + (n / 10000)

def smallest_five_digit_palindrome_div_4 : ℕ :=
  18881

theorem smallest_five_digit_palindrome_div_4_thm :
  is_palindrome smallest_five_digit_palindrome_div_4 ∧
  10000 ≤ smallest_five_digit_palindrome_div_4 ∧
  smallest_five_digit_palindrome_div_4 < 100000 ∧
  smallest_five_digit_palindrome_div_4 % 4 = 0 ∧
  ∀ n, is_palindrome n ∧ 10000 ≤ n ∧ n < 100000 ∧ n % 4 = 0 → n ≥ smallest_five_digit_palindrome_div_4 :=
by
  sorry

end smallest_five_digit_palindrome_div_4_thm_l537_537750


namespace find_a_l537_537682

-- Definitions of the known conditions.
def a (a c : ℝ) (C : ℝ) : Prop := 3 * a * (Real.cos C) = 4 * c * (Real.sin (Real.acos ((3 * a) / (4 * c))))
def S : ℝ := 10
def b : ℝ := 4

-- Main theorem to be proven.
theorem find_a (a c C : ℝ) (h1 : 3 * a * (Real.cos C) = 4 * c * (Real.sin (Real.acos ((3 * a) / (4 * c)))))
                  (h2 : 10 = 1/2 * a * b * (Real.sin C)) : 
                  a = 25 / 3 := 
by
  sorry -- Proof to be completed.

end find_a_l537_537682


namespace polynomial_triples_l537_537392

theorem polynomial_triples (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (A B C : Polynomial ℝ) :
  (A ^ n + B ^ n + C ^ n = 0) →
  ∃ (a b c : ℝ) (D : Polynomial ℝ), A = C * Polynomial.C a ∧ B = C * Polynomial.C b ∧ C = C * Polynomial.C c ∧ a ^ n + b ^ n + c ^ n = 0 :=
by sorry

end polynomial_triples_l537_537392


namespace xyz_final_stock_price_l537_537212

def initial_stock_price : ℝ := 120
def first_year_increase_rate : ℝ := 0.80
def second_year_decrease_rate : ℝ := 0.30

def final_stock_price_after_two_years : ℝ :=
  (initial_stock_price * (1 + first_year_increase_rate)) * (1 - second_year_decrease_rate)

theorem xyz_final_stock_price :
  final_stock_price_after_two_years = 151.2 := by
  sorry

end xyz_final_stock_price_l537_537212


namespace discount_calculation_l537_537798

noncomputable def other_discount_percentage (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price - (first_discount / 100 * list_price)
  in ((price_after_first_discount - final_price) / price_after_first_discount) * 100

theorem discount_calculation :
  other_discount_percentage 67 56.16 10 ≈ 6.86 :=
by
  sorry

end discount_calculation_l537_537798


namespace regular_polygon_sides_l537_537978

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537978


namespace root_of_quadratic_eq_l537_537208

theorem root_of_quadratic_eq (v : ℚ) (h : ∃ x : ℚ, (3 * x^2 + 15 * x + v = 0) ∧ x = (-15 - real.sqrt 469) / 6) : 
  v = -20.3333 := 
by
  sorry

end root_of_quadratic_eq_l537_537208


namespace problem_statement_l537_537461

def smallest_positive_prime : ℕ := 2
def largest_integer_with_three_positive_divisors_upto (n : ℕ) : ℕ :=
  let is_prime (p : ℕ) := p > 1 ∧ ∀ m, m > 1 ∧ m < p → p % m ≠ 0
  let has_three_divisors (k : ℕ) := ∃ p : ℕ, is_prime p ∧ k = p * p
  let candidates := (Finset.range n).filter has_three_divisors
  candidates.max' (by norm_num)

theorem problem_statement : smallest_positive_prime + largest_integer_with_three_positive_divisors_upto 150 = 123 := 
by
  let a := smallest_positive_prime
  let b := largest_integer_with_three_positive_divisors_upto 150
  have ha : a = 2 := rfl
  have hb : b = 121 := rfl
  rw [ha, hb]
  norm_num

end problem_statement_l537_537461


namespace ratio_of_ages_in_two_years_l537_537529

theorem ratio_of_ages_in_two_years
    (S : ℕ) (M : ℕ) 
    (h1 : M = S + 32)
    (h2 : S = 30) : 
    (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l537_537529


namespace trig_identity_proof_l537_537872

theorem trig_identity_proof :
  (√3 * real.sin (10 * real.pi / 180) - real.cos (10 * real.pi / 180)) / (real.cos (10 * real.pi / 180) * real.sin (10 * real.pi / 180)) = -4 :=
by sorry

end trig_identity_proof_l537_537872


namespace remaining_volume_is_half_or_less_surface_area_greater_than_100_l537_537160

noncomputable def unit_cube_volume := 1
noncomputable def division_segments (k : ℕ) := k
noncomputable def remaining_volume (k : ℕ) : ℚ :=
if even k then 1 / 2 else 1 / 2 + (3 * k ^ 2 - 1) / (4 * k ^ 3)

noncomputable def surface_area (k : ℕ) : ℚ :=
if even k then 3 * (k + 1) / 2 else 3 * (k + 1) ^ 2 / (2 * k)

theorem remaining_volume_is_half_or_less (k : ℕ) : even k → remaining_volume k = 1 / 2 :=
by sorry

theorem surface_area_greater_than_100 (k : ℕ) : k ≥ 65 → surface_area k > 100 :=
by sorry

end remaining_volume_is_half_or_less_surface_area_greater_than_100_l537_537160


namespace distance_between_points_l537_537924

theorem distance_between_points :
  let p1 := (1 : ℝ, -3 : ℝ, 4 : ℝ)
  let p2 := (-2 : ℝ, 2 : ℝ, -1 : ℝ)
  dist p1 p2 = Real.sqrt 59 := by
  sorry

end distance_between_points_l537_537924


namespace diagonal_through_center_l537_537686

variables {n : ℕ} (h_odd : n % 2 = 1)

/-- Given a 2n-gon inscribed around a circle, if n is odd and a set of diagonals passes through the circle's center,
    then the n-th diagonal from vertex A_n also passes through the center. -/
theorem diagonal_through_center
  (A : Fin 2n → Point)
  (O : Point)
  (inscribed : ∀ i, dist O (A i) = radius)
  (diagonals_pass_through_center : ∀ i, 1 ≤ i ∧ i < n → (line_through (A i) (A (i + n))).contains O) :
  (line_through (A n) (A 2n)).contains O :=
sorry

end diagonal_through_center_l537_537686


namespace sin_double_angle_l537_537591

theorem sin_double_angle (α : ℝ) (h : cos (α - π / 4) = sqrt 2 / 4) : sin (2 * α) = -3 / 4 :=
by
  -- sorry to skip the proof
  sorry

end sin_double_angle_l537_537591


namespace polynomial_coefficient_equality_l537_537445

noncomputable def polynomial_expansion_equality (p q : ℝ) : Prop :=
  (x + y)^10 = ∑ k in range(11), (choose 10 k) * x^(10-k) * y^k

theorem polynomial_coefficient_equality (p q : ℝ) (h1 : p + q = 2) 
  (h2 : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = 16 / 11 :=
by
  sorry

end polynomial_coefficient_equality_l537_537445


namespace volume_of_cone_l537_537809

theorem volume_of_cone (l h : ℝ) (l_pos : l = 15) (h_pos : h = 9) : 
  let r := Real.sqrt (l^2 - h^2) in
  let V := (1 / 3) * Real.pi * r^2 * h in
  V = 432 * Real.pi :=
by
  -- Definitions derived from the conditions
  have r_def : r = Real.sqrt (15^2 - 9^2), from sorry,
  have V_def : V = (1 / 3) * Real.pi * (Real.sqrt (15^2 - 9^2))^2 * 9, from sorry,
  -- Proof of the theorem
  sorry

end volume_of_cone_l537_537809


namespace factor_theorem_l537_537509

theorem factor_theorem (m : ℝ) : (∀ x : ℝ, x + 5 = 0 → x ^ 2 - m * x - 40 = 0) → m = 3 :=
by
  sorry

end factor_theorem_l537_537509


namespace candies_left_to_share_l537_537711

def initial_candies : ℕ := 100
def siblings : ℕ := 3
def candies_per_sibling : ℕ := 10
def candies_josh_eats : ℕ := 16

theorem candies_left_to_share : 
  let candies_given_to_siblings := siblings * candies_per_sibling in
  let candies_after_siblings := initial_candies - candies_given_to_siblings in
  let candies_given_to_friend := candies_after_siblings / 2 in
  let candies_after_friend := candies_after_siblings - candies_given_to_friend in
  candies_after_friend - candies_josh_eats = 19 :=
by 
  sorry

end candies_left_to_share_l537_537711


namespace determine_constants_l537_537367

-- Define the matrix B
def B : Matrix (fin 3) (fin 3) ℤ :=
  ![![0, 2, 1],
    ![2, 0, 2],
    ![1, 2, 0]]

-- Define the identity matrix
def I : Matrix (fin 3) (fin 3) ℤ :=
  1

-- Define the zero matrix
def Z : Matrix (fin 3) (fin 3) ℤ :=
  0

-- Define the constants (a, b, c)
def a : ℤ := 0
def b : ℤ := -10
def c : ℤ := -32

-- Statement to prove
theorem determine_constants :
  B ^ 3 + a • B ^ 2 + b • B + c • I = Z :=
by
  -- proof goes here, but we skip it with sorry
  sorry

end determine_constants_l537_537367


namespace min_degree_g_l537_537842

-- Definitions for the polynomials and their degrees.
def polynomial (x : Type) := finset x -> ℤ

variable {R : Type*} [comm_ring R]
variables {f g h : polynomial R}

-- Conditions given in the problem
def five_f_plus_seven_g_eq_h (f g h : polynomial R) : Prop :=
  (5 * f + 7 * g) = h

def degree_f_eq_six (f : polynomial R) : Prop :=
  f.degree = 6

def degree_h_eq_ten (h : polynomial R) : Prop :=
  h.degree = 10

-- The proof statement, no proof provided (using 'sorry')
theorem min_degree_g
  (f g h : polynomial R)
  (h₁ : five_f_plus_seven_g_eq_h f g h)
  (h₂ : degree_f_eq_six f)
  (h₃ : degree_h_eq_ten h) :
  g.degree ≥ 10 :=
sorry

end min_degree_g_l537_537842


namespace count_7s_in_1_to_1000_l537_537173

theorem count_7s_in_1_to_1000 : 
  (finset.range 1001).sum (λ n, (nat.digits 10 n).count 7) = 300 := 
sorry

end count_7s_in_1_to_1000_l537_537173


namespace correct_quadratic_equation_l537_537403

theorem correct_quadratic_equation :
  ∃ (b c : ℝ), (∀ α β : ℝ, (α + β = 9 ∧ α * β = c) → (α = 6 ∧ β = 3 ∨ α = 3 ∧ β = 6)) ∧
               (∀ γ δ : ℝ, (γ + δ = -b ∧ γ * δ = 36) → (γ = -12 ∧ δ = -3 ∨ γ = -3 ∧ δ = -12)) ∧
               (b = -9 ∧ c = 36) :=
begin
  use [-9, 36],
  split,
  { intros α β h,
    cases h with h1 h2,
    split; intros; finish,
    assumption },
  { intros γ δ h,
    cases h with h1 h2,
    split; intros; finish },
  split;
  finish
end

end correct_quadratic_equation_l537_537403


namespace find_f2009_l537_537966

noncomputable def f : ℝ → ℝ :=
sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (2 + x) = -f (2 - x)
axiom initial_condition : f (-3) = -2

theorem find_f2009 : f 2009 = 2 :=
sorry

end find_f2009_l537_537966


namespace number_of_2_1_designs_l537_537174

theorem number_of_2_1_designs (S : Finset (Fin 8)) :
  let designs := { design : Set (Finset (Fin 8)) // ∀ s ∈ design, s.card = 2 ∧ ∀ s₁ s₂ ∈ design, s₁ ≠ s₂ → s₁ ∩ s₂ ⊆ ∅ ∨ s₁ ∩ s₂.card = 1 }
  (designs.toFinset.card = 2 ^ 28) :=
by
  let designs : Set (Set (Finset (Fin 8))) := { design | ∀ s ∈ design, s.card = 2 ∧ ∀ s₁ s₂ ∈ design, s₁ ≠ s₂ → s₁ ∩ s₂ ⊆ ∅ ∨ s₁ ∩ s₂.card = 1 }
  have : designs.toFinset.card = 2 ^ 28
  apply sorry

end number_of_2_1_designs_l537_537174


namespace line_translation_units_l537_537543

theorem line_translation_units
  (original_line : ℝ → ℝ) (translated_line : ℝ → ℝ)
  (h_original : original_line = λ x, 3 * x - 1)
  (h_translated : translated_line = λ x, 3 * x + 6) :
  (∃ t : ℝ, ∀ x, translated_line x = original_line x + t) ∧ (t = 7) :=
by
  sorry

end line_translation_units_l537_537543


namespace focal_chords_unique_length_l537_537891

theorem focal_chords_unique_length {p : ℝ} (hp : 0 < p) :
  ∃ (A B : ℝ × ℝ), (y1 - y2) * (y1 + y2) = 6 ∧ (y1, x1 = -y2, x2 = 2p (x2)^2 3 (y1) (sqrt(3))?(x-(p)^6{}) = 

  sorry

end focal_chords_unique_length_l537_537891


namespace solve_a₃_l537_537345

noncomputable def geom_seq (a₁ a₅ a₃ : ℝ) : Prop :=
a₁ = 1 / 9 ∧ a₅ = 9 ∧ a₁ * a₅ = a₃^2

theorem solve_a₃ : ∃ a₃ : ℝ, geom_seq (1/9) 9 a₃ ∧ a₃ = 1 :=
by
  sorry

end solve_a₃_l537_537345


namespace profit_percentage_l537_537547

theorem profit_percentage (cost_price marked_price : ℝ)
  (h1 : cost_price = 47.5)
  (h2 : marked_price = 64.54) :
  let discount := 0.08 * marked_price,
      selling_price := marked_price - discount,
      profit := selling_price - cost_price,
      profit_percentage := (profit / cost_price) * 100
  in profit_percentage ≈ 25 :=
by
  sorry

end profit_percentage_l537_537547


namespace least_num_distinct_values_l537_537892

theorem least_num_distinct_values
  (n : ℕ) (a b c : ℕ)
  (h1 : n = 2023)
  (h2 : a = 15)
  (h3 : b + a * c = 2023)
  : n ≥ 2023 → a = 15 → b < 15 → c ≥ 145 :=
by
  intros,
  have h4 : 13 * c + 15 ≥ 2023,
  sorry

end least_num_distinct_values_l537_537892


namespace zeros_in_expansion_l537_537662

def x : ℕ := 10^12 - 3
def num_zeros (n : ℕ) : ℕ := (n.toString.filter (· == '0')).length

theorem zeros_in_expansion :
  num_zeros (x^2) = 20 := sorry

end zeros_in_expansion_l537_537662


namespace xiaolin_homework_points_l537_537184

theorem xiaolin_homework_points
  {a b : ℝ}
  (q1 : 2 * a * b + 3 * a * b = 5 * a * b)
  (q2 : 2 * a * b - 3 * a * b = - a * b)
  (q3 : 2 * a * b - 3 * a * b ≠ 6 * a * b)
  (q4 : 2 * a * b / (3 * a * b) = 2 / 3) :
  let correct_answers_points := 3 * 2 in
  correct_answers_points = 6 :=
sorry

end xiaolin_homework_points_l537_537184


namespace rectangle_area_error_l537_537176

theorem rectangle_area_error (A B : ℝ) :
  let A' := 1.08 * A
  let B' := 1.08 * B
  let actual_area := A * B
  let measured_area := A' * B'
  let percentage_error := ((measured_area - actual_area) / actual_area) * 100
  percentage_error = 16.64 :=
by
  sorry

end rectangle_area_error_l537_537176


namespace vasya_rectangles_l537_537465

theorem vasya_rectangles (s : ℝ) (h : ℝ) :
  let area_triangle := (1 / 2) * s * (s * (sqrt 3 / 2))
  let total_triangle_area := 3 * area_triangle
  let rectangle_area := 2 * h ^ 2 in
  total_triangle_area = rectangle_area → 2 = h / (h / (sqrt (3 * sqrt 3 / 8)))
:= 
begin
  sorry
end

end vasya_rectangles_l537_537465


namespace Debby_photographs_after_deletion_l537_537211

theorem Debby_photographs_after_deletion (N : ℝ) (hN : 0 ≤ N) :
  let zoo_photos := 0.60 * N
  let museum_photos := 0.25 * N
  let gallery_photos := 0.15 * N
  let kept_zoo_photos := 0.70 * zoo_photos
  let kept_museum_photos := 0.50 * museum_photos
  let kept_gallery_photos := gallery_photos
  in kept_zoo_photos + kept_museum_photos + kept_gallery_photos = 0.695 * N :=
by
  let zoo_photos := 0.60 * N
  let museum_photos := 0.25 * N
  let gallery_photos := 0.15 * N
  let kept_zoo_photos := 0.70 * zoo_photos
  let kept_museum_photos := 0.50 * museum_photos
  let kept_gallery_photos := gallery_photos
  have h1 : kept_zoo_photos = 0.42 * N := by sorry
  have h2 : kept_museum_photos = 0.125 * N := by sorry
  have h3 : kept_gallery_photos = 0.15 * N := by sorry
  show kept_zoo_photos + kept_museum_photos + kept_gallery_photos = 0.695 * N, from
    by sorry

end Debby_photographs_after_deletion_l537_537211


namespace both_selected_prob_l537_537834

noncomputable def prob_X : ℚ := 1 / 3
noncomputable def prob_Y : ℚ := 2 / 7
noncomputable def combined_prob : ℚ := prob_X * prob_Y

theorem both_selected_prob :
  combined_prob = 2 / 21 :=
by
  unfold combined_prob prob_X prob_Y
  sorry

end both_selected_prob_l537_537834


namespace age_of_new_person_l537_537783

-- Definitions based on conditions
def initial_avg : ℕ := 15
def new_avg : ℕ := 17
def n : ℕ := 9

-- Statement to prove
theorem age_of_new_person : 
    ∃ (A : ℕ), (initial_avg * n + A) / (n + 1) = new_avg ∧ A = 35 := 
by {
    -- Proof steps would go here, but since they are not required, we add 'sorry' to skip the proof
    sorry
}

end age_of_new_person_l537_537783


namespace find_sum_of_all_possible_values_l537_537665

noncomputable def given_condition (θ φ : ℝ) :=
  ( (sin θ ^ 4 / sin φ ^ 2) + (cos θ ^ 4 / cos φ ^ 2) = 1 )

noncomputable def target_expression (θ φ : ℝ) :=
  (cos φ ^ 4 / cos θ ^ 2) + (sin φ ^ 4 / sin θ ^ 2)

theorem find_sum_of_all_possible_values (θ φ : ℝ) (h : given_condition θ φ) :
  target_expression θ φ = 1 :=
sorry

end find_sum_of_all_possible_values_l537_537665


namespace max_trigonometric_expression_le_max_trigonometric_expression_ex_l537_537389

section
variables {a b c : ℝ} (θ : ℝ)

noncomputable def trigonometric_expression : ℝ := a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ)

theorem max_trigonometric_expression_le (a b c : ℝ) : ∀ θ : ℝ, trigonometric_expression a b c θ ≤ Real.sqrt (a^2 + b^2 + 4 * c^2) :=
by
  sorry
  
theorem max_trigonometric_expression_ex (a b c : ℝ) : ∃ θ : ℝ, trigonometric_expression a b c θ = Real.sqrt (a^2 + b^2 + 4 * c^2) :=
by
  sorry

end

end max_trigonometric_expression_le_max_trigonometric_expression_ex_l537_537389


namespace regular_15_gon_sum_of_lengths_l537_537900

noncomputable def sum_of_lengths_of_sides_and_diagonals
  (radius : ℝ)
  (a b c d : ℕ) :
  ℝ :=
  let α := (2 : ℕ) * radius * real.sin (real.pi / 30)
      β := (2 : ℕ) * radius * real.sin ((2 * real.pi) / 15)
  in (15 * α + 15 * β)

theorem regular_15_gon_sum_of_lengths
  (radius : ℝ)
  (a b c d : ℕ) :
  radius = 10 →
  sum_of_lengths_of_sides_and_diagonals radius a b c d = ↑a + ↑b * real.sqrt 2 + ↑c * real.sqrt 3 + ↑d * real.sqrt 5 →
  a + b + c + d = 120 :=
by
  intros h_radius h_sum
  -- proof goes here
  sorry

end regular_15_gon_sum_of_lengths_l537_537900


namespace average_gas_mileage_round_trip_l537_537138

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l537_537138


namespace tim_runs_more_than_sarah_l537_537330

-- Definitions based on the conditions
def street_width : ℕ := 25
def side_length : ℕ := 450

-- Perimeters of the paths
def sarah_perimeter : ℕ := 4 * side_length
def tim_perimeter : ℕ := 4 * (side_length + 2 * street_width)

-- The theorem to prove
theorem tim_runs_more_than_sarah : tim_perimeter - sarah_perimeter = 200 := by
  -- The proof will be filled in here
  sorry

end tim_runs_more_than_sarah_l537_537330


namespace abc_geq_inequality_l537_537768

open Real

theorem abc_geq_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end abc_geq_inequality_l537_537768


namespace joan_balloons_l537_537706

theorem joan_balloons (m t j : ℕ) (h1 : m = 41) (h2 : t = 81) : j = t - m → j = 40 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end joan_balloons_l537_537706


namespace green_eyes_count_l537_537453

noncomputable def people_count := 100
noncomputable def blue_eyes := 19
noncomputable def brown_eyes := people_count / 2
noncomputable def black_eyes := people_count / 4
noncomputable def green_eyes := people_count - (blue_eyes + brown_eyes + black_eyes)

theorem green_eyes_count : green_eyes = 6 := by
  sorry

end green_eyes_count_l537_537453


namespace max_coprime_set_cardinality_l537_537229

theorem max_coprime_set_cardinality :
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 50) ∧
                    (∀ n m ∈ S, n ≠ m → Nat.gcd n m = 1) ∧
                    S.card = 16 :=
begin
  sorry
end

end max_coprime_set_cardinality_l537_537229


namespace max_squares_fitting_l537_537843

theorem max_squares_fitting (L S : ℕ) (hL : L = 8) (hS : S = 2) : (L / S) * (L / S) = 16 := by
  -- Proof goes here
  sorry

end max_squares_fitting_l537_537843


namespace trains_meet_distance_l537_537464

theorem trains_meet_distance :
  let speed_train1 := 54 * 1000 / 3600 -- converting 54 km/h to m/s
  let speed_train2 := 72 * 1000 / 3600 -- converting 72 km/h to m/s
  let relative_speed := speed_train1 + speed_train2
  let time := 3.999680025597952
  let distance := relative_speed * time in
  distance ≈ 139.99 := -- using ≈ for approximately equal to 
by sorry

end trains_meet_distance_l537_537464


namespace smallest_n_is_7_l537_537106

noncomputable def smallest_n : ℕ :=
  Inf { n : ℕ | n > 0 ∧ ∃ d : ℕ, d > 1 ∧ d ∣ (8 * n - 3) ∧ d ∣ (5 * n + 4) }

theorem smallest_n_is_7 : smallest_n = 7 :=
by
  sorry

end smallest_n_is_7_l537_537106


namespace ellipse_equation_chord_line_through_point_max_area_triangle_line_eq_l537_537613

-- Problem (1)
theorem ellipse_equation {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a > b) (ecc : (sqrt 3) / 2 = sqrt (1 - (b / a) ^ 2)) (focus_slope : Abs ((2 * sqrt 3) / 3) = c) : 
  (a = 2) (b = 1) → ∀ {x y : ℝ},  y^2 + x^2 / 4 = 1 :=
sorry

-- Problem (2)
theorem chord_line_through_point (x1 y1 x2 y2 : ℝ)
  (ellipse_eq1 : y1^2 + x1^2 / 4 = 1)
  (ellipse_eq2 : y2^2 + x2^2 / 4 = 1) (P : (1, 1/2)) :
  y1 = 1/2 -> x1 = 1 -> IFF(y = (-(x/2)+2)){
 sorry
  
-- Problem (3)
theorem max_area_triangle_line_eq (k x1 y1 x2 y2 : ℝ) (b = 1) 
  (line_eq : k * x -2 = y)
  (e : POS(k) && k^2 > 3.4)
-- maximizing the area:
 let delta := 4k^2 - 3;
 let t := sqrt(delta) ;
 S = minimize k -> maximizing area :
  y = -((sqrt (7))/2) * x - 2 { 
  
    sorry


end ellipse_equation_chord_line_through_point_max_area_triangle_line_eq_l537_537613


namespace tan_arithmetic_geometric_l537_537283

noncomputable def a_seq : ℕ → ℝ := sorry -- Define a_n as an arithmetic sequence (details abstracted)
noncomputable def b_seq : ℕ → ℝ := sorry -- Define b_n as a geometric sequence (details abstracted)

axiom a_seq_is_arithmetic : ∀ n m : ℕ, a_seq (n + 1) - a_seq n = a_seq (m + 1) - a_seq m
axiom b_seq_is_geometric : ∀ n : ℕ, ∃ r : ℝ, b_seq (n + 1) = b_seq n * r
axiom a_seq_sum : a_seq 2017 + a_seq 2018 = Real.pi
axiom b_seq_square : b_seq 20 ^ 2 = 4

theorem tan_arithmetic_geometric : 
  (Real.tan ((a_seq 2 + a_seq 4033) / (b_seq 1 * b_seq 39)) = 1) :=
sorry

end tan_arithmetic_geometric_l537_537283


namespace arc_length_ratio_l537_537064

theorem arc_length_ratio
  (h_circ : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1)
  (h_line : ∀ x y : ℝ, x - y = 0) :
  let shorter_arc := (1 / 4) * (2 * Real.pi)
  let longer_arc := 2 * Real.pi - shorter_arc
  shorter_arc / longer_arc = 1 / 3 :=
by
  sorry

end arc_length_ratio_l537_537064


namespace average_distinct_s_l537_537281

theorem average_distinct_s (s : ℕ) :
  (∀ a b : ℕ, a + b = 7 → a * b = s → ∃ r : ℚ, r = 28 / 3) :=
begin
  sorry
end

end average_distinct_s_l537_537281


namespace reflect_y_axis_correct_l537_537786

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l537_537786


namespace apples_per_person_before_joining_l537_537515

theorem apples_per_person_before_joining :
  ∃ x : ℕ, let initial_people := x in
           let total_apples := 2750 in
           let extra_people := 60 in
           let apples_less := 12 in
           total_apples / x - total_apples / (x + extra_people) = apples_less ∧
           total_apples / x = 30 :=
begin
  sorry,
end

end apples_per_person_before_joining_l537_537515


namespace joan_games_last_year_l537_537707

theorem joan_games_last_year (games_year_this : ℕ) (games_last_year : ℕ) :
  (games_year_this = 4 + 3) → (games_year_this + games_last_year = 9) → games_last_year = 2 :=
by
  intros h₁ h₂
  rw h₁ at h₂
  sorry

end joan_games_last_year_l537_537707


namespace find_quadrilateral_area_l537_537964

noncomputable def cyclic_quadrilateral_K : ℚ :=
  let d1d2_A := arbitrary ℚ
  let d1d2_B := arbitrary ℚ
  let d1d2_C := arbitrary ℚ
  let sin_phi_A := 2 / 3
  let sin_phi_B := 3 / 5
  let sin_phi_C := 6 / 7
  let KA := (1 / 2) * d1d2_A * sin_phi_A
  let KB := (1 / 2) * d1d2_B * sin_phi_B
  let KC := (1 / 2) * d1d2_C * sin_phi_C
  let K_cube := (1 / 8) * (d1d2_A) * (d1d2_B) * (d1d2_C) * sin_phi_A * sin_phi_B * sin_phi_C
  KA = KB ∧ KB = KC ∧ 
  K_cube = (3 / 35) * d1d2_A * d1d2_B * d1d2_C →
  ∃ (K : ℚ), K = 16 / 35 ∧ 
  let m := 16 
  let n := 35 
  K = frac m n ∧ isRelativelyPrime m n

theorem find_quadrilateral_area : (cyclic_quadrilateral_K = 16 / 35 ∧ ¬ ∃ r : ℚ, 16 / 35 = r ∧ ¬ isRelativelyPrime 16 35) → 16 + 35 = 51 :=
by sorry

end find_quadrilateral_area_l537_537964


namespace isosceles_triangle_AD_perpendicular_BC_l537_537681

variable {A B C D : Type} [EuclideanGeometry A B C D] 
variables {AB AC BC AD BD CD : ℝ}
variables (h1 : AB = AC) (h2 : BD = CD)

theorem isosceles_triangle_AD_perpendicular_BC (h1 : AB = AC) (h2 : BD = CD) : 
    isPerpendicular AD BC := 
sorry

end isosceles_triangle_AD_perpendicular_BC_l537_537681


namespace find_pqr_l537_537167

-- Define the conditions.
def conditions (p q r : ℕ) (A B C : ℕ) : Prop :=
  p < q ∧ q < r ∧ A = 20 ∧ B = 10 ∧ C = 9 ∧ (∃ draws, 
   (last_draw_B : draws.pop.last r) ∧ 
    (B_sum_of_draws : ∑ d in draws, if r then r - p else 0) = 18)

-- Prove that, given the conditions, p = 3, q = 6, and r = 13.
theorem find_pqr : ∃ (p q r : ℕ), conditions p q r 20 10 9 ∧ p = 3 ∧ q = 6 ∧ r = 13 :=
by
  sorry

end find_pqr_l537_537167


namespace volume_of_cone_l537_537812

theorem volume_of_cone (l h : ℝ) (l_pos : l = 15) (h_pos : h = 9) : 
  let r := Real.sqrt (l^2 - h^2) in
  let V := (1 / 3) * Real.pi * r^2 * h in
  V = 432 * Real.pi :=
by
  -- Definitions derived from the conditions
  have r_def : r = Real.sqrt (15^2 - 9^2), from sorry,
  have V_def : V = (1 / 3) * Real.pi * (Real.sqrt (15^2 - 9^2))^2 * 9, from sorry,
  -- Proof of the theorem
  sorry

end volume_of_cone_l537_537812


namespace answered_q1_correctly_l537_537747

-- Define the context based on given conditions
def total_students : ℕ := 30
def answered_q2 : ℕ := 22
def did_not_take_test : ℕ := 5
def answered_both : ℕ := 22

-- Define the main proof that the number of students who answered question 1 correctly is 22
theorem answered_q1_correctly : (total_students - did_not_take_test = (25 : ℕ)) ∧ answered_both = 22 → answered_both = 22 := 
by 
  intro h,
  sorry

end answered_q1_correctly_l537_537747


namespace fraction_of_yard_occupied_by_flower_beds_is_one_fifth_l537_537151

-- Definitions for yard and flower beds
def Yard : Type := { length: ℝ, width: ℝ }
def Triangle : Type := { leg: ℝ }

-- Given conditions
def is_rectangular (y: Yard) := y.length > 0 ∧ y.width > 0
def flower_beds_are_congruent_isosceles_right (t1 t2: Triangle) := t1.leg = t2.leg
def trapezoid_parallel_sides (a b: ℝ) (x y: Yard) := a = 18 ∧ b = 30

-- Yard and triangles instance
def yard : Yard := { length := 30, width := 6 } -- assuming the same height as triangles
def triangle : Triangle := { leg := 6 }

-- Problem statement
theorem fraction_of_yard_occupied_by_flower_beds_is_one_fifth :
  is_rectangular yard ∧
  flower_beds_are_congruent_isosceles_right triangle triangle ∧
  trapezoid_parallel_sides 18 30 yard ∧ 
  (2 * (1 / 2 * (triangle.leg)^2)) / (yard.length * yard.width) = 1 / 5 := 
sorry

end fraction_of_yard_occupied_by_flower_beds_is_one_fifth_l537_537151


namespace solve_equation_l537_537053

theorem solve_equation :
  ∃! (x y z : ℝ), 2 * x^4 + 2 * y^4 - 4 * x^3 * y + 6 * x^2 * y^2 - 4 * x * y^3 + 7 * y^2 + 7 * z^2 - 14 * y * z - 70 * y + 70 * z + 175 = 0 ∧ x = 0 ∧ y = 0 ∧ z = -5 :=
by
  sorry

end solve_equation_l537_537053


namespace regular_polygon_sides_l537_537986

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l537_537986


namespace g_of_minus_two_eq_two_l537_537377

def f(x : ℝ) : ℝ := 2 * x - 4
def g(z : ℝ) : ℝ := 3 * (z / 2 + 2) ^ 2 + 4 * (z / 2 + 2) - 5

theorem g_of_minus_two_eq_two : g (-2) = 2 := by
  have h1 : f 1 = -2 := by
    show f 1 = 2 * 1 - 4
    show 2 * 1 - 4 = -2
  have h2 : f x = 2 * x - 4 := rfl
  have h3 : g (f x) = 3 * x * x + 4 * x - 5 := rfl
  show g (f 1) = 2
  calc
    g (f 1) = 3 * 1 * 1 + 4 * 1 - 5 : by rw [h2, h3]
         ... = 3 + 4 - 5 : by ring
         ... = 2 : rfl

end g_of_minus_two_eq_two_l537_537377


namespace exists_representation_of_77_using_fewer_sevens_l537_537362

-- Definition of the problem
def represent_77 (expr : String) : Prop :=
  ∀ n : ℕ, expr = "77" ∨ 
             expr = "(77 - 7) + 7" ∨ 
             expr = "(10 * 7) + 7" ∨ 
             expr = "(70 + 7)" ∨ 
             expr = "(7 * 11)" ∨ 
             expr = "7 + 7 * 7 + (7 / 7)"

-- The proof statement
theorem exists_representation_of_77_using_fewer_sevens : ∃ expr : String, represent_77 expr ∧ String.length expr < 3 := 
sorry

end exists_representation_of_77_using_fewer_sevens_l537_537362


namespace sum_of_first_6_primes_over_10_l537_537504

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def first_n_primes_greater_than (n : ℕ) (k : ℕ) : List ℕ :=
  -- A function that generates the first k primes greater than n
  List.filter is_prime (List.range (n + k + 100)).filter (λ x, x > n).take k

theorem sum_of_first_6_primes_over_10 : 
  (List.sum (first_n_primes_greater_than 10 6)) = 112 := sorry

end sum_of_first_6_primes_over_10_l537_537504


namespace number_of_correct_statements_l537_537801

def input_statement (s : String) : Prop :=
  s = "INPUT a; b; c"

def output_statement (s : String) : Prop :=
  s = "A=4"

def assignment_statement1 (s : String) : Prop :=
  s = "3=B"

def assignment_statement2 (s : String) : Prop :=
  s = "A=B=-2"

theorem number_of_correct_statements :
    input_statement "INPUT a; b; c" = false ∧
    output_statement "A=4" = false ∧
    assignment_statement1 "3=B" = false ∧
    assignment_statement2 "A=B=-2" = false :=
sorry

end number_of_correct_statements_l537_537801


namespace class_album_l537_537561

theorem class_album (
  (x y m : ℕ) 
  (h1: 38 ≤ x) 
  (h2: x < 50) 
  (h3: (x > 0)) 
  (h4: x = 42)
  (h5: y = (x / 6) + 2)
  (h6: x * y = m)
  (h7: y = (( x + 12 ) * ( y - 2 )) / x)
) : x = 42 ∧ y = 9 := sorry

end class_album_l537_537561


namespace Monge_Circle_Slope_Ratio_l537_537203

noncomputable def MongeCircleSlope (x y : ℝ) : ℝ :=
  if h : (x + 2 * y = 3) ∧ (x^2 + y^2 = 9) then y / x else 0

theorem Monge_Circle_Slope_Ratio :
  ∃ (x y : ℝ), 
    (x + 2 * y = 3) ∧ 
    (x^2 + y^2 = 9) ∧ 
    (MongeCircleSlope x y = -4/3 ∨ MongeCircleSlope x y = 0) :=
begin
  sorry
end

end Monge_Circle_Slope_Ratio_l537_537203


namespace pencils_more_than_pens_l537_537807

theorem pencils_more_than_pens (pencils pens : ℕ) (h_ratio : 5 * pencils = 6 * pens) (h_pencils : pencils = 48) : 
  pencils - pens = 8 :=
by
  sorry

end pencils_more_than_pens_l537_537807


namespace range_of_t_l537_537289

noncomputable def f (t x : ℝ) : ℝ := (Real.log x + (x - t)^2) / x

noncomputable def f' (t x : ℝ) : ℝ := (1 + 2*x*(x - t) - Real.log x - (x - t)^2) / (x^2)

theorem range_of_t (t : ℝ) : 
  (∀ x ∈ Icc 1 2, f'(t, x) * x + f(t, x) > 0) ↔ t < 3/2 :=
by
  sorry

end range_of_t_l537_537289


namespace cookies_left_l537_537408

theorem cookies_left 
  (brother_cookies : ℕ)
  (mother_factor : ℚ)
  (sister_factor : ℚ)
  (initial_cookies : ℕ) :
  brother_cookies = 10 →
  mother_factor = 1 / 2 →
  sister_factor = 2 / 3 →
  initial_cookies = 20 →
  let cookies_after_brother := initial_cookies - brother_cookies,
      mother_cookies := mother_factor * brother_cookies,
      cookies_after_mother := cookies_after_brother + mother_cookies,
      cookies_to_sister := sister_factor * cookies_after_mother,
      final_cookies := cookies_after_mother - cookies_to_sister
  in
  final_cookies = 5 :=
by
  intro h1 h2 h3 h4,
  simp only [h1, h2, h3, h4, sub_eq_add_neg, int.cast_of_nat, mul_assoc],
  norm_num,
  sorry

end cookies_left_l537_537408


namespace calc_remainder_mult_l537_537925

theorem calc_remainder_mult (n : ℕ) (m : ℕ) (k : ℕ) (q : ℕ) (r : ℕ) (h₁ : n = 972345) (h₂ : m = 145) (h₃ : k = 7) 
  (h₄ : q = n / m) (h₅ : r = n % m) : r * k = 840 :=
by 
  simp [h₁, h₂, h₃, h₄, h₅]
  sorry

end calc_remainder_mult_l537_537925


namespace regular_polygon_sides_l537_537987

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537987


namespace set_cardinality_bound_l537_537391

theorem set_cardinality_bound (n : ℕ) (A : Finset ℕ)
  (h_subset : ∀ x ∈ A, x ≤ n)
  (h_lcm_bound : ∀ a b ∈ A, Nat.lcm a b ≤ n) :
  A.card ≤ floor (1.9 * Real.sqrt n) + 5 := 
sorry

end set_cardinality_bound_l537_537391


namespace car_total_distance_l537_537880

  noncomputable def distance_traveled (a₁ : ℕ) (d : ℤ) : ℕ :=
    Nat.sum (List.filter (λ n, n > 0) (List.range ((a₁ : ℤ) / (-d)) .map (λ n, (a₁ : ℤ) + n * d).map (λ n, Int.toNat n)))

  theorem car_total_distance :
    distance_traveled 40 (-10) = 100 :=
  by
    sorry
  
end car_total_distance_l537_537880


namespace inequality_l537_537241

noncomputable def a : ℝ := 2 / 5
noncomputable def b : ℝ := real.sqrt 2
noncomputable def c : ℝ := real.log (1 / 2) / real.log 3

theorem inequality : b > a ∧ a > c :=
by
  -- a = 2 / 5 implied from problem statement
  -- b = sqrt 2 > 1 implied from problem statement
  -- c = log_base 3 (1 / 2) < 0 implied from problem statement
  sorry

end inequality_l537_537241


namespace problem_statement_l537_537915

noncomputable def function1 (x : Real) : Real := cos (abs (2 * x))
noncomputable def function2 (x : Real) : Real := abs (sin (x + π))
noncomputable def function3 (x : Real) : Real := abs (sin (2 * x + π / 2))
noncomputable def function4 (x : Real) : Real := tan (abs x)

theorem problem_statement : 
  (∀ x, function1 x = function1 (-x) ∧ ∀ k : Int, function1 (x + k * π) = function1 x) ∧
  (∀ x, function2 x = function2 (-x) ∧ ∀ k : Int, function2 (x + k * π) = function2 x) ∧
  ¬ (∀ x, function3 x = function3 (-x) ∧ ∀ k : Int, function3 (x + k * π) = function3 x) ∧
  ¬ (∀ x, function4 x = function4 (-x) ∧ ∀ k : Int, function4 (x + k * π) = function4 x) 
  :=
sorry

end problem_statement_l537_537915


namespace cos_of_60_degrees_is_half_l537_537953

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537953


namespace paint_grid_l537_537129

theorem paint_grid (paint : Fin 3 × Fin 3 → Bool) (no_adjacent : ∀ i j, (paint (i, j) = true) → (paint (i+1, j) = false) ∧ (paint (i-1, j) = false) ∧ (paint (i, j+1) = false) ∧ (paint (i, j-1) = false)) : 
  ∃! (count : ℕ), count = 8 :=
sorry

end paint_grid_l537_537129


namespace grasshopper_probability_2022_l537_537143

noncomputable theory

def a : ℕ → ℝ
def b : ℕ → ℝ
def c : ℕ → ℝ
def d : ℕ → ℝ
def e : ℕ → ℝ

axiom recurrence_relations (n : ℕ) :
  a (n + 1) = 1/2 * (b n + e n)
  ∧ b (n + 1) = 1/2 * (a n + c n)
  ∧ c (n + 1) = 1/2 * (b n + d n)
  ∧ d (n + 1) = 1/2 * (c n + e n)
  ∧ e (n + 1) = 1/2 * (d n + a n)

axiom initial_conditions :
  a 0 = 1 ∧ b 0 = 0 ∧ c 0 = 0 ∧ d 0 = 0 ∧ e 0 = 0

theorem grasshopper_probability_2022 :
  a 2022 > 1 / 5 :=
by {
  apply sorry,
}

end grasshopper_probability_2022_l537_537143


namespace find_n_l537_537031

def exp (m n : ℕ) : ℕ := m ^ n

-- Now we restate the problem formally
theorem find_n 
  (m n : ℕ) 
  (h1 : exp 10 m = n * 22) : 
  n = 10^m / 22 := 
sorry

end find_n_l537_537031


namespace conjugate_of_z_l537_537970

open Complex -- Open the Complex namespace to use complex number functionalities

theorem conjugate_of_z : (conj (2 * I / (1 + I)) = (1 - I)) :=
  by sorry  -- The proof goes here

end conjugate_of_z_l537_537970


namespace tangent_line_property_midpoint_trajectory_min_area_triangle_l537_537246

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

noncomputable def line_tangent_to_circle (a b x y : ℝ) : Prop :=
  let l := b * x + a * y - a * b in
  l = 0 ∧ a > 2 ∧ b > 2

noncomputable def distance_from_center_to_line (a b : ℝ) : ℝ :=
  abs (a + b - a * b) / real.sqrt (a^2 + b^2)

theorem tangent_line_property (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  circle_equation 1 1 → line_tangent_to_circle a b 1 1 → distance_from_center_to_line a b = 1 → 
  (a - 2) * (b - 2) = 2 := by
  sorry

theorem midpoint_trajectory (a b x y : ℝ) (ha : a = 2*x) (hb : b = 2*y) 
  (h : (a - 2) * (b - 2) = 2) : (x - 1) * (y - 1) = 1/2 := by
  sorry

theorem min_area_triangle (a b : ℝ) (ha : a > 2) (hb : b > 2)
  (h : (a - 2) * (b - 2) = 2) : (\(O A B : ℝ), a*b/2) ≥ 3 + 2*real.sqrt 2 := by
  sorry

end tangent_line_property_midpoint_trajectory_min_area_triangle_l537_537246


namespace solution_set_of_inequality_l537_537079

open Set Real

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 2 * x < 3) ↔ x ∈ Ioo (-3 : ℝ) 1 := by
  sorry

end solution_set_of_inequality_l537_537079


namespace eighth_positive_odd_multiple_of_5_l537_537481

theorem eighth_positive_odd_multiple_of_5 : 
  let a := 5 in 
  let d := 10 in 
  let n := 8 in 
  a + (n - 1) * d = 75 :=
by
  let a := 5
  let d := 10
  let n := 8
  have : a + (n - 1) * d = 75 := by 
    calc
      a + (n - 1) * d = 5 + (8 - 1) * 10  : by rfl
      ... = 5 + 70                          : by rfl
      ... = 75                              : by rfl
  exact this

end eighth_positive_odd_multiple_of_5_l537_537481


namespace Sabrina_cookies_left_l537_537410

theorem Sabrina_cookies_left (start : ℕ) (brother_given mom_given_sister_given : ℕ) : start = 20 → brother_given = 10 → mom_given_sister_given = 5 → (start - brother_given + mom_given_sister_given) = 15 →
(sister_given = 2 / 3 * (start - brother_given + mom_given_sister_given)) → (start - brother_given + mom_given_sister_given - sister_given) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Sabrina_cookies_left_l537_537410


namespace correctness_of_statements_l537_537317

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end correctness_of_statements_l537_537317


namespace average_gas_mileage_round_trip_l537_537139

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l537_537139


namespace percentage_error_calculation_l537_537114

theorem percentage_error_calculation :
  let incorrect_factor := (3 : ℚ) / 5
  let correct_factor   := (5 : ℚ) / 3
  let ratio := incorrect_factor / correct_factor
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 := by
    let incorrect_factor := (3 : ℚ) / 5
    let correct_factor := (5 : ℚ) / 3
    let ratio := incorrect_factor / correct_factor
    let percentage_error := (1 - ratio) * 100
    have h1 : ratio = (3 / 5) / (5 / 3) := by rfl
    rw [h1, div_eq_mul_inv, mul_comm, mul_assoc]
    norm_num
    have h2 : (1 - (incorrect_factor * incorrect_factor)) * 100 = 64 := by norm_num
    exact h2

end percentage_error_calculation_l537_537114


namespace largest_k_proof_l537_537740

noncomputable def a_sequence : ℕ → ℝ
| 0     := 1/3
| (n+1) := a_sequence n / (sqrt (1 + 13 * (a_sequence n) ^ 2))

def largest_k_satisfying_condition : ℕ :=
  (Nat.floor ((2504 : ℝ) / 13).ceil : ℕ)

theorem largest_k_proof : ∀ (k : ℕ), (a_sequence k < 1/50) ↔ (k ≤ 193) :=
by 
  have h_base : (a_sequence 0 < 1/50) := sorry
  have h_recurse : ∀ k, (a_sequence k < 1/50) → (a_sequence (k + 1) / (sqrt (1 + 13 * (a_sequence k)^2)) < 1/50) := sorry
  exact ⟨λ h, sorry, λ h, sorry⟩

end largest_k_proof_l537_537740


namespace car_total_distance_l537_537879

def arithmetic_sequence_distance (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem car_total_distance : 
  let a := 40 in let d := -10 in let n := 5 in
  (arithmetic_sequence_distance a d n) = 100 :=
by
  sorry

end car_total_distance_l537_537879


namespace describe_T_correctly_l537_537013

def T (x y : ℝ) : Prop :=
(x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2)

theorem describe_T_correctly :
  (∀ x y : ℝ, T x y ↔
    ((x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2))) :=
by
  sorry

end describe_T_correctly_l537_537013


namespace max_double_when_digit_erased_l537_537230

theorem max_double_when_digit_erased : 
  ∃ n : ℝ, (0 < n) ∧ ((∃ d : ℕ, d ∈ n.digits 10 ∧ 2 * n = n.erase_digit d) ∧ (n = 0.375))
:= 
sorry

end max_double_when_digit_erased_l537_537230


namespace min_abs_alpha_gamma_l537_537381

noncomputable def f (z : ℂ) (α γ : ℂ) : ℂ :=
  (2 + complex.i) * z^3 + (4 + complex.i) * z^2 + α * z + γ

theorem min_abs_alpha_gamma (α γ : ℂ)
  (h1 : (f 1 α γ).im = 0)
  (h2 : (f complex.i α γ).im = 0) :
  complex.abs α + complex.abs γ = sqrt 13 := sorry

end min_abs_alpha_gamma_l537_537381


namespace razorback_tshirt_shop_profit_l537_537780

theorem razorback_tshirt_shop_profit :
  let price_per_tshirt := 98
  let tshirts_sold_arkansas_game := 89
  let total_money_made := price_per_tshirt * tshirts_sold_arkansas_game
  total_money_made = 8722 := by
  -- each t-shirt is sold for $98
  have price : price_per_tshirt = 98 := rfl
  -- they sold 89 t-shirts during the Arkansas game
  have tshirts_sold : tshirts_sold_arkansas_game = 89 := rfl
  -- calculation of total amount: 98 * 89
  have calculation : total_money_made = 98 * 89 := rfl
  -- 98 * 89 = 8722 by numeric evaluation
  have numeric_evaluation : 98 * 89 = 8722 := by norm_num
  show total_money_made = 8722 from numeric_evaluation

end razorback_tshirt_shop_profit_l537_537780


namespace distinct_s_average_is_28_over_3_l537_537278

theorem distinct_s_average_is_28_over_3 (a b : ℕ) (s : ℕ) :
  (∀ a b, a + b = 7 → a * b = s) →
  (∀ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ a < 7 ∧ b < 7) →
  let distinct_s_vals : finset ℕ := {6, 10, 12} in
  ↑(distinct_s_vals.sum id) / distinct_s_vals.card = 28 / 3 :=
by
  intros h1 h2
  sorry

end distinct_s_average_is_28_over_3_l537_537278


namespace eighth_odd_multiple_of_5_is_75_l537_537477

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0) ∧ (n % 2 = 1) ∧ (n % 5 = 0) ∧ (nat.find_greatest (λ m, m % 2 = 1 ∧ m % 5 = 0) 75 = 75) :=
    sorry

end eighth_odd_multiple_of_5_is_75_l537_537477


namespace magnitude_of_sum_of_vectors_l537_537263

-- Definitions of given conditions
variables {R : Type*} [normed_field R] [inner_product_space R (R × R)]

noncomputable def unit_vector_a := (1 : R, 0 : R)  -- any unit vector
noncomputable def unit_vector_b := (1/2 : R, sqrt (3 / 4) : R)  -- any unit vector making 60° angle with a

-- The norm of a unit vector is 1
lemma norm_unit_vector_a : ∥unit_vector_a∥ = 1 := by simp [unit_vector_a]
lemma norm_unit_vector_b : ∥unit_vector_b∥ = 1 := by simp [unit_vector_b]

-- The dot product of unit vectors with angle 60 degrees is 1/2
lemma dot_product_unit_vectors : inner unit_vector_a unit_vector_b = 1 / 2 := 
begin
  simp [unit_vector_a, unit_vector_b],
  norm_num,
end

-- The main statement to be proved
theorem magnitude_of_sum_of_vectors : ∥(unit_vector_a + 3 • unit_vector_b)∥ = sqrt 13 := 
by sorry

end magnitude_of_sum_of_vectors_l537_537263


namespace vershoks_in_arshin_l537_537303

theorem vershoks_in_arshin (length_board_ar: ℝ) (width_board_ver: ℝ) (num_boards: ℕ) (side_room_ar: ℝ) :
  length_board_ar = 6 ∧ width_board_ver = 6 ∧ num_boards = 64 ∧ side_room_ar = 12 →
  (∃ (vershoks_in_arshin: ℝ), vershoks_in_arshin = 16) :=
by
  intro h,
  cases h,
  sorry

end vershoks_in_arshin_l537_537303


namespace cosine_60_degrees_l537_537948

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537948


namespace min_value_xy_yz_xz_thm_l537_537614

noncomputable def min_value_xy_yz_xz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : x + y + z = 3 * x * y * z) : ℝ := 
  xy + yz + xz

theorem min_value_xy_yz_xz_thm (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3 * x * y * z) :
  min_value_xy_yz_xz x y z h1 h2 h3 h4 = 3 :=
begin
  sorry
end

end min_value_xy_yz_xz_thm_l537_537614


namespace packages_calculation_l537_537496

theorem packages_calculation :
  ∀ (total_shirts packages_per_unit : ℕ),
  total_shirts = 426 →
  packages_per_unit = 6 →
  (total_shirts / packages_per_unit) = 71 :=
by
  intros total_shirts packages_per_unit h1 h2
  rw [h1, h2]
  exact Nat.div_eq_of_eq (by norm_num)

end packages_calculation_l537_537496


namespace probation_inequalities_l537_537717

variable {α : Type*} [LinearOrderedField α] (S : ℕ → α)

-- Formalizing the conditions
def is_arithmetic_sum (S : ℕ → α) : Prop :=
  ∃ (a d : α), 0 < a ∧ ∀ n, S n = n * (2 * a + (n - 1) * d) / 2

def a1_geq_1 (a : α) : Prop :=
  a ≥ 1

-- Defining the actual proof problem
theorem probation_inequalities (a d : α) (h1 : a > 0) (h2 : is_arithmetic_sum S) (h3 : a1_geq_1 a) (m n : ℕ) :
  S (2 * m) * S (2 * n) ≤ S (m + n) * S (m + n) ∧ 
  log S (2 * m) * log S (2 * n) ≤ (log S (m + n)) ^ 2 := 
sorry

end probation_inequalities_l537_537717


namespace loci_of_P_l537_537652

noncomputable theory

variables {A B C D P : Point}
variables (hAB : Line A B)
variables (hC : Point)
variables (hNotIso : ¬isosceles_triangle A B C)
variables (hD : external_angle_bisector A C B D)
variables (K : circle A D C)
variables (hP_tangent : tangent A P K)

theorem loci_of_P (hP_tangent : tangent A P K) (hExtBisector : external_angle_bisector A C B D)
  (hCircle : circle A D C) (hProps : properties_of_circle A D C) :
  lies_on_perpendicular_bisector P A B ∧ P ≠ midpoint A B :=
  sorry

end loci_of_P_l537_537652


namespace cos_double_angle_l537_537592

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 := 
sorry

end cos_double_angle_l537_537592


namespace limit_X_n_l537_537728

noncomputable theory

def S (n : ℕ) : Set (Finset ℕ) := {s | s.card = 6 ∧ s ⊆ Finset.range (n + 1)}

def P (n : ℕ) (s : Finset ℕ) : Polynomial ℤ := 
  ∑ i in s, Polynomial.monomial i (1 : ℤ)

def Q (p : Polynomial ℤ) : Prop := 
  ∃ (q : Polynomial ℤ) (hq : q.degree ≤ 3), (q ≠ 0) ∧ (q.coeff 0 ≠ 0) ∧ (q ∣ p)

def X_n (n : ℕ) : ℚ := 
  nat.size {s ∈ S n | Q (P n s)}.to_finset / nat.size (S n).to_finset

theorem limit_X_n : filter.tendsto (λ n, X_n n) filter.at_top (𝓝 (10015 / 20736 : ℚ)) :=
sorry

end limit_X_n_l537_537728


namespace problem_1_problem_2_l537_537126

-- Problem 1
theorem problem_1 : 
  (log 3 2 + log 9 2) * (log 4 3 + log 8 3) + (2 : ℝ) ^ log 2 5 = 25 / 4 :=
begin
  sorry
end

-- Problem 2
theorem problem_2 (x : ℝ) (h : x - x⁻¹ = -7 / 2) :
  x^3 - (x⁻¹)^3 = -371 / 8 :=
begin
  sorry
end

end problem_1_problem_2_l537_537126


namespace arithmetic_sequence_sum_l537_537341

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
def problem_conditions (a : ℕ → ℝ) : Prop :=
  (a 3 + a 8 = 3) ∧ is_arithmetic_sequence a

-- State the theorem to be proved
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : problem_conditions a) : a 1 + a 10 = 3 :=
sorry

end arithmetic_sequence_sum_l537_537341


namespace percentage_of_games_lost_l537_537448

theorem percentage_of_games_lost (games_won games_lost games_tied total_games : ℕ)
  (h_ratio : 5 * games_lost = 3 * games_won)
  (h_tied : games_tied * 5 = total_games) :
  (games_lost * 10 / total_games) = 3 :=
by sorry

end percentage_of_games_lost_l537_537448


namespace min_omega_value_for_50_maxima_l537_537205

theorem min_omega_value_for_50_maxima (ω : ℝ) (hω : ω > 0) :
  (∃ T : ℝ, T = 2 * Real.pi / ω ∧ ∑(i : ℕ) in (range 50), real.sin (ω * i) = 50 * T) → ω ≥ 197 * Real.pi / 2 :=
begin
  sorry
end

end min_omega_value_for_50_maxima_l537_537205


namespace transformed_function_is_neg_cos_l537_537833

def f (x : ℝ) : ℝ := sin((1/2) * x - (π / 6))

def transformation_shrink (h : ℝ → ℝ) (x : ℝ) : ℝ :=
  h (2 * x)

def transformation_shift_right (h : ℝ → ℝ) (shift : ℝ) (x : ℝ) : ℝ :=
  h (x - shift)

theorem transformed_function_is_neg_cos :
  transformation_shift_right (transformation_shrink f) (π / 3) = -cos := 
sorry

end transformed_function_is_neg_cos_l537_537833


namespace range_of_a_l537_537615

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, real.log (x^2 + 2 * x + a) = real.log (0.5) x)
  (p_iff : ∃ x : ℝ, p x ↔ ((x^2 + 2*x + a > 0) <-> a ≤ 1))
  (q : ∀ x : ℝ, -(5 - 2*a)^(x) < -(5 - 2*a)^(x + 1) )
  (q_iff : ∃ x : ℝ, q x ↔ (a < 2))
  (either_p_or_q : (∃ x : ℝ, p x) ∨ (∃ x : ℝ, q x))
  (p_and_q_false : ¬(∃ x : ℝ, p x ∧ q x)) :
  1 < a ∧ a < 2 := by
  sorry

end range_of_a_l537_537615


namespace average_distinct_s_l537_537282

theorem average_distinct_s (s : ℕ) :
  (∀ a b : ℕ, a + b = 7 → a * b = s → ∃ r : ℚ, r = 28 / 3) :=
begin
  sorry
end

end average_distinct_s_l537_537282


namespace find_line_from_points_l537_537273

-- Defining the conditions in terms of Lean structures and hypotheses

variables {x y a1 b1 a2 b2 : ℝ}
variables (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)

-- Consider P as the intersection point of the lines
def line1 := (a1 : ℝ) * x + (b1 : ℝ) * y + 1 = 0
def line2 := (a2 : ℝ) * x + (b2 : ℝ) * y + 1 = 0

-- Point of intersection given
def P_point := (2, 3)

-- Equations from substituting P into line1 and line2
def eq1 := (2 * a1 + 3 * b1 + 1 = 0)
def eq2 := (2 * a2 + 3 * b2 + 1 = 0)

-- Points A and B
def A_point := (a1, b1)
def B_point := (a2, b2)

-- Given A and B are distinct
axiom distinct_points : a1 ≠ a2

-- The line we need to verify
theorem find_line_from_points : eq1 ∧ eq2 ∧ distinct_points → ∀ x y, 2 * x + 3 * y + 1 = 0 :=
by
  intro h
  sorry

end find_line_from_points_l537_537273


namespace decimal_between_0_996_and_0_998_ne_0_997_l537_537519

theorem decimal_between_0_996_and_0_998_ne_0_997 :
  ∃ x : ℝ, 0.996 < x ∧ x < 0.998 ∧ x ≠ 0.997 :=
by
  sorry

end decimal_between_0_996_and_0_998_ne_0_997_l537_537519


namespace correct_factorization_l537_537853

variable (x y : ℝ)

theorem correct_factorization :
  x^2 - 2 * x * y + x = x * (x - 2 * y + 1) :=
by sorry

end correct_factorization_l537_537853


namespace compute_a_l537_537616

noncomputable def a_value : ℚ := 239 / 71

theorem compute_a
  (a b : ℚ)
  (h1 : root (Polynomial.C 45 + Polynomial.X * (Polynomial.C b + Polynomial.X * (Polynomial.C a + Polynomial.X))) (-2 - 5 * Real.sqrt 3)) : 
  a = a_value :=
sorry

end compute_a_l537_537616


namespace number_of_sides_of_regular_polygon_l537_537972

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l537_537972


namespace problem1_problem2_l537_537295

-- Sequence definition
def a (n : ℕ) : ℚ := 1 / (2 * n)

-- Sequence b definition
def b (n : ℕ) : ℚ :=
if n % 2 = 1 
then 1 / (Real.sqrt (n - 1) + Real.sqrt (n + 1))
else a (n / 2) * a (n / 2 + 1)

-- Sum of first n terms of b
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i, b i)

-- Sequence T definition
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i, 1 / a (i + 1))

-- Prove the sum S_{64}
theorem problem1 : S 64 = 140 / 33 := sorry

-- Prove the existence of a constant c such that {T_n / (n + c)} is an arithmetic sequence
theorem problem2 : ∃ c : ℚ, (∀ n : ℕ, (T n) / (n + c) = n + 1) :=
begin
  use 0,
  sorry
end

end problem1_problem2_l537_537295


namespace solution_set_of_inequality_l537_537737

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f'' x < f x) (h2 : f 0 = 1) :
  {x | f x / Real.exp x < 1} = set.Ioi 0 :=
sorry

end solution_set_of_inequality_l537_537737


namespace B_spends_85_percent_l537_537131

theorem B_spends_85_percent {S_A S_B savings_A savings_B : ℝ} :
  S_A + S_B = 14000 ∧ S_B = 8000 ∧ S_A * 0.2 = savings_A ∧ ∀ (S_A S_B : ℝ), savings_A = savings_B → S_B * (1 - 0.85) = savings_B → savings_B = S_B - S_B * 0.85 :=
begin
  sorry,
end

end B_spends_85_percent_l537_537131


namespace angle_sum_of_midpoints_of_altitudes_l537_537444

theorem angle_sum_of_midpoints_of_altitudes 
  (A B C A1 B1 C1 A2 B2 C2 : Type) 
  [acute_triangle A B C]
  (midpoint_altitudes : 
    midpoint A2 (altitude A A1)
    ∧ midpoint B2 (altitude B B1)
    ∧ midpoint C2 (altitude C C1)) : 
  ∠ B2 A1 C2 + ∠ C2 B1 A2 + ∠ A2 C1 B2 = 180 := 
sorry

end angle_sum_of_midpoints_of_altitudes_l537_537444


namespace major_premise_wrong_l537_537839

-- Definition of the problem conditions and the proof goal
theorem major_premise_wrong :
  (∀ a : ℝ, |a| > 0) ↔ false :=
by {
  sorry  -- the proof goes here but is omitted as per the instructions
}

end major_premise_wrong_l537_537839


namespace cosine_60_degrees_l537_537951

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537951


namespace crossed_out_digit_l537_537040

theorem crossed_out_digit (N S S' x : ℕ) (hN : N % 9 = 3) (hS : S % 9 = 3) (hS' : S' % 9 = 7)
  (hS'_eq : S' = S - x) : x = 5 :=
by
  sorry

end crossed_out_digit_l537_537040


namespace harsha_travel_time_l537_537867

theorem harsha_travel_time
  (track_length : ℕ)
  (round_trip_time : ℕ)
  (forest_grove_distance_fraction : ℝ)
  (scottsdale_to_sherbourne : ℕ)
  (forest_grove_distance : ℝ) :
  track_length = 200 ∧ round_trip_time = 5 ∧ forest_grove_distance_fraction = 1 / 5 ∧ forest_grove_distance = (1 / 5 : ℝ) * (200 : ℝ) →
  let speed := (2 * track_length) / round_trip_time in
  let remaining_distance := track_length - forest_grove_distance * ↑track_length in
  let travel_time := remaining_distance / speed in
  travel_time = 2 :=
by
  intros h
  simp at h
  sorry

end harsha_travel_time_l537_537867


namespace sum_of_variables_l537_537619

theorem sum_of_variables (a b c d : ℕ) (h1 : ac + bd + ad + bc = 1997) : a + b + c + d = 1998 :=
sorry

end sum_of_variables_l537_537619


namespace cos_60_eq_one_half_l537_537940

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537940


namespace clock_hands_indistinguishable_l537_537819

def minute_units := 60

def position_mod (α β : ℕ) : Prop :=
  ∃ k : ℕ, α = 12 * β - k * minute_units

def ambiguous_positions (α β : ℕ) : Prop :=
  ∃ (m n : ℕ), α = m * minute_units / 143 ∧ β = n * minute_units / 143

theorem clock_hands_indistinguishable (α β : ℕ) (h : position_mod α β) :
  ∃ moments : ℕ, 0 < moments ∧ moments = 143 ∧ ∀ m n : ℕ, ambiguous_positions α β → (n - m) * minute_units ≈ 5 + 21 / 60 := sorry

end clock_hands_indistinguishable_l537_537819


namespace estimate_total_fish_l537_537829

-- Defining conditions
variable (L : Type) -- Type for the lake
variable [Fintype L] -- Assuming the lake has a finite number of fish

def is_marked (f : L) : Prop := -- Property for fish being marked
  sorry

-- Given conditions
constant num_caught_first : Nat := 100
constant num_marked_first : Nat := 100
constant num_caught_second : Nat := 200
constant num_marked_second : Nat := 25

-- Proportion setup
constant total_fish : Nat -- Total number of fish in the lake

theorem estimate_total_fish : 
  let prop1 := (num_caught_second : ℝ) / (total_fish : ℝ)
  let prop2 := (num_marked_second : ℝ) / (num_marked_first : ℝ)
  (prop1 = prop2) → total_fish = 800 :=
by
  sorry

end estimate_total_fish_l537_537829


namespace p_div_q_is_12_l537_537574

-- Definition of binomials and factorials required for the proof
open Nat

/-- Define the number of ways to distribute balls for configuration A -/
def config_A : ℕ :=
  @choose 5 1 * @choose 4 2 * @choose 2 1 * (factorial 20) / (factorial 2 * factorial 4 * factorial 4 * factorial 3 * factorial 7)

/-- Define the number of ways to distribute balls for configuration B -/
def config_B : ℕ :=
  @choose 5 2 * @choose 3 3 * (factorial 20) / (factorial 3 * factorial 3 * factorial 4 * factorial 4 * factorial 4)

/-- The ratio of probabilities p/q for the given distributions of balls into bins is 12 -/
theorem p_div_q_is_12 : config_A / config_B = 12 :=
by
  sorry

end p_div_q_is_12_l537_537574


namespace not_monotonic_f4_l537_537546

open Real

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f x ≥ f y)

def f1 (x : ℝ) := -2 * x + 1
def f2 (x : ℝ) := x ^ 3
def f3 (x : ℝ) := log x
def f4 (x : ℝ) := 1 / x

theorem not_monotonic_f4 : ¬ is_monotonic f4 :=
by
  sorry

end not_monotonic_f4_l537_537546


namespace proof_problem_l537_537535

open Real

variables {a : ℝ}
def average (l : list ℝ) : ℝ := l.sum / l.length

def range (l : list ℝ) : ℝ := l.foldr max (-∞) - l.foldr min ∞

noncomputable def mode (l : list ℝ) : list ℝ :=
  let occurrences := l.foldl (λ counts x, counts.insert x (count x l)) ∅
  let max_occ := occurrences.fold (λ _ count max_count, max count max_count) 0
  (occurrences.to_finset.filter (λ x, count x l = max_occ)).to_list

noncomputable def variance (l : list ℝ) : ℝ :=
  let mean := average l
  l.foldr (λ x acc, acc + (x - mean) ^ 2) 0 / l.length

theorem proof_problem (h : average [3, 6, 8, a, 5, 9] = 6) : 
  a = 5 ∧ range [3, 6, 8, 5, 5, 9] = 6 ∧ mode [3, 6, 8, 5, 5, 9] = [5] ∧ variance [3, 6, 8, 5, 5, 9] = 4 :=
by sorry

end proof_problem_l537_537535


namespace determine_P_l537_537254

variables {A B C : Type} [topological_space A] [topological_space B] [topological_space C]
variables {A' B' C' : A → B} {α β : ℝ} 
variables (P : A) [topological_space P] 

-- Assume hypothesis for angles
variable (hα : ∠ A P C = α)
variable (hβ : ∠ B P C = β)

-- Constructs
def P' (A' B' C' : A → B) (α β : ℝ) : B := 
-- the intersection of the arcs
sorry 

def Q' (A' B' C' : A → B) (α β : ℝ) : B := 
-- the intersection point after rotation
sorry 

def P'' (A' B' C' : A → B) (Q' : B) : B := 
-- the circumcircle intersection
sorry 

-- The theorem statement
theorem determine_P' (A' B' C' : A → B) (α β : ℝ) : 
  P' A' B' C' α β = P'' A' B' C' (Q' A' B' C' α β) := 
sorry

end determine_P_l537_537254


namespace remaining_days_to_finish_l537_537000

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l537_537000


namespace avg_distinct_s_l537_537276

theorem avg_distinct_s (s : ℤ) (hpq : ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p + q = 7 ∧ p * q = s) :
  (∑ (p q : ℤ) in {1, 2, 3, 4, 5, 6}, if p + q = 7 then p * q else 0).erase 0 = {6, 10, 12} →
  s = ∑ x in {6, 10, 12}, x / {6, 10, 12}.size := sorry

end avg_distinct_s_l537_537276


namespace equal_numbers_l537_537373

theorem equal_numbers {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + da) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end equal_numbers_l537_537373


namespace weight_of_B_l537_537432

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l537_537432


namespace percentage_change_profits_1998_2000_l537_537684

variables (R : ℝ)
def profits_1998 := 0.10 * R
def revenue_1999 := 0.70 * R
def profits_1999 := 0.15 * revenue_1999
def revenue_2000 := revenue_1999 + 0.20 * revenue_1999
def profits_2000 := 0.18 * revenue_2000

theorem percentage_change_profits_1998_2000 : 
  let change := ((profits_2000 R - profits_1998 R) / profits_1998 R) * 100 in
  change = 51.2 := by
  sorry

end percentage_change_profits_1998_2000_l537_537684


namespace smallest_n_l537_537489

theorem smallest_n (n : ℕ) (hn1 : ∃ k, 5 * n = k^4) (hn2: ∃ m, 4 * n = m^3) : n = 2000 :=
sorry

end smallest_n_l537_537489


namespace min_weights_l537_537857

theorem min_weights (n : ℕ) (masses : List ℕ) (weights : List ℕ) 
  (h1 : n = 20) 
  (h2 : masses = List.range' 1 20)
  (h3 : ∀ m ∈ masses, ∃ v1 v2 ∈ weights, m = v1 + v2) : weights.length ≥ 6 := 
sorry

end min_weights_l537_537857


namespace volume_of_cone_l537_537811

theorem volume_of_cone (l h : ℝ) (l_pos : l = 15) (h_pos : h = 9) : 
  let r := Real.sqrt (l^2 - h^2) in
  let V := (1 / 3) * Real.pi * r^2 * h in
  V = 432 * Real.pi :=
by
  -- Definitions derived from the conditions
  have r_def : r = Real.sqrt (15^2 - 9^2), from sorry,
  have V_def : V = (1 / 3) * Real.pi * (Real.sqrt (15^2 - 9^2))^2 * 9, from sorry,
  -- Proof of the theorem
  sorry

end volume_of_cone_l537_537811


namespace num_ways_128_as_sum_of_four_positive_perfect_squares_l537_537333

noncomputable def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, 0 < m ∧ m * m = n

noncomputable def four_positive_perfect_squares_sum (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    is_positive_perfect_square a ∧
    is_positive_perfect_square b ∧
    is_positive_perfect_square c ∧
    is_positive_perfect_square d ∧
    a + b + c + d = n

theorem num_ways_128_as_sum_of_four_positive_perfect_squares :
  (∃! (a b c d : ℕ), four_positive_perfect_squares_sum 128) :=
sorry

end num_ways_128_as_sum_of_four_positive_perfect_squares_l537_537333


namespace proof_problem_l537_537310

noncomputable def arithmetic_mean (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem proof_problem (a b c x y z m : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (m_pos : 0 < m) (m_ne_one : m ≠ 1) 
  (h_b : b = arithmetic_mean a c) (h_y : y = geometric_mean x z) :
  (b - c) * Real.logb m x + (c - a) * Real.logb m y + (a - b) * Real.logb m z = 0 := by
  sorry

end proof_problem_l537_537310


namespace summation_identity_l537_537754

theorem summation_identity (n : ℕ) : 
  (∑ i in Finset.range n, (i + 1) / ((2 * (i + 1) - 1) * (2 * (i + 1) + 1) * (2 * (i + 1) + 3))) = 
  n * (n + 1) / (2 * (2 * n + 1) * (2 * n + 3)) :=
sorry

end summation_identity_l537_537754


namespace cone_volume_is_correct_l537_537816

-- Given conditions
def slant_height : ℝ := 15
def height : ℝ := 9

-- Derived quantities
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- Volume formula for a cone
def volume_cone (r h : ℝ) := (1 / 3) * π * r^2 * h

theorem cone_volume_is_correct : volume_cone radius height = 432 * π := by
  sorry

end cone_volume_is_correct_l537_537816


namespace total_pizzas_eaten_l537_537124

-- Definitions for the conditions
def pizzasA : ℕ := 8
def pizzasB : ℕ := 7

-- Theorem stating the total number of pizzas eaten by both classes
theorem total_pizzas_eaten : pizzasA + pizzasB = 15 := 
by
  -- Proof is not required for the task, so we use sorry
  sorry

end total_pizzas_eaten_l537_537124


namespace eighth_positive_odd_multiple_of_5_l537_537482

theorem eighth_positive_odd_multiple_of_5 : 
  let a := 5 in 
  let d := 10 in 
  let n := 8 in 
  a + (n - 1) * d = 75 :=
by
  let a := 5
  let d := 10
  let n := 8
  have : a + (n - 1) * d = 75 := by 
    calc
      a + (n - 1) * d = 5 + (8 - 1) * 10  : by rfl
      ... = 5 + 70                          : by rfl
      ... = 75                              : by rfl
  exact this

end eighth_positive_odd_multiple_of_5_l537_537482


namespace calculate_expression_l537_537190

theorem calculate_expression : -1 ^ 4 + 16 / (-2) ^ 3 * | -3 - 1 | = -9 := 
by 
  sorry

end calculate_expression_l537_537190


namespace SpyIsA_l537_537086

namespace LogicPuzzle

inductive Role
| Knight -- always tells the truth
| Liar -- always lies
| Spy -- can either lie or tell the truth

open Role

variables (A B C : Role)
variable (AJudgedCIsNotSpy : Bool) -- Judge's assertion on C not being the spy
variable (ACallsCNotSpy : Bool) -- A says "C is not the spy"
variable (AJudgesTruthfulnessTruth : Bool) -- Judge concludes who the spy is based on their logic reasoning

-- Define A says "Yes" or "No" on being the spy and B's response to A's truthfulness
variables (ASaysItsTheSpy : Bool) (BConfirmsATruthfulness : Bool)

-- Translate the conditions into Lean
def is_spy : Role → Bool
| Spy := true
| _ := false

def tells_truth (r : Role) (statement : Bool) : Bool :=
match r with
| Knight => statement
| Liar => ¬statement
| Spy => sorry -- spy can either tell the truth or lie, need more context
end

-- Theorem: Given the conditions, determine that the spy is A
theorem SpyIsA (hA : tells_truth A ASaysItsTheSpy = is_spy A)
              (hB : tells_truth B (tells_truth A ASaysItsTheSpy) = BConfirmsATruthfulness)
              (hADisclaimsC : tells_truth A ACallsCNotSpy = ¬ is_spy C)
              (hJudgeAffirms : AJudgedCIsNotSpy = ¬ is_spy C)
              (hJudgeKnows : AJudgesTruthfulnessTruth = true) :
  is_spy A = true :=
sorry -- Proof omitted
end LogicPuzzle

end SpyIsA_l537_537086


namespace trigonometric_identity_l537_537308

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
sorry

end trigonometric_identity_l537_537308


namespace every_prime_appears_in_seq_l537_537731

-- Definitions based on conditions
def primeDivisor (n : ℕ) : ℕ := smallestPrimeDivisor (n + 1)

def sequence : ℕ → ℕ 
| 0     := 2
| (n+1) := primeDivisor 
  (n * (sequence 0)^(1!) * (sequence 1)^(2!) * (sequence n)^(n!))

theorem every_prime_appears_in_seq (p : ℕ) (hp : Prime p) : 
  ∃ n : ℕ, p = sequence n := 
sorry

end every_prime_appears_in_seq_l537_537731


namespace gina_good_tipper_l537_537239

noncomputable def calculate_tip_difference (bill_in_usd : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (low_tip_rate : ℝ) (high_tip_rate : ℝ) (conversion_rate : ℝ) : ℝ :=
  let discounted_bill := bill_in_usd * (1 - discount_rate)
  let taxed_bill := discounted_bill * (1 + tax_rate)
  let low_tip := taxed_bill * low_tip_rate
  let high_tip := taxed_bill * high_tip_rate
  let difference_in_usd := high_tip - low_tip
  let difference_in_eur := difference_in_usd * conversion_rate
  difference_in_eur * 100

theorem gina_good_tipper : calculate_tip_difference 26 0.08 0.07 0.05 0.20 0.85 = 326.33 := 
by
  sorry

end gina_good_tipper_l537_537239


namespace intersection_of_lines_l537_537580

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (8*x - 5*y = 10) ∧ (6*x + 3*y = 21) ∧ (x = 5/2) ∧ (y = 2) :=
by
  use 5/2
  use 2
  split
  { ring_nf
    norm_num
    ring_nf }
  split
  { ring_nf
    norm_num }
  split
  { refl }
  { refl }

end intersection_of_lines_l537_537580


namespace part_a_part_b_l537_537862

-- Part (a) statement
theorem part_a (a x : ℝ) (ha : 0 < a ∧ a ≠ 1) (hx : 0 < x) :
  (3 * log a x + 6) / (log a x ^ 2 + 2) > 1 →
  ((0 < a ∧ a < 1 ∧ a ^ 4 < x ∧ x < 1 / a) ∨ (1 < a ∧ 1 / a < x ∧ x < a ^ 4)) :=
sorry

-- Part (b) statement
theorem part_b (x : ℝ) (hx : 0 < x) :
  log 2 (log 4 x) + log 4 (log 2 x) ≤ -4 →
  (1 < x ∧ x < real.root 4 2) :=
sorry

end part_a_part_b_l537_537862


namespace cos_60_eq_sqrt3_div_2_l537_537932

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537932


namespace angles_of_triangle_in_geometric_sequence_l537_537441

theorem angles_of_triangle_in_geometric_sequence (α β γ : ℝ) (a q : ℝ) (hpos : 0 < a) (hq : 1 < q) (cos_rule : ∃ (a q : ℝ), (cos α = (q^4 + q^2 - 1) / (2 * q^3)) ∧ (cos β = (1 - q^2 + q^4) / (2 * q^2)) ∧ (cos γ = (1 + q^2 - q^4) / (2 * q))) :
    cos α = (q^4 + q^2 - 1) / (2 * q^3) ∧
    cos β = (1 - q^2 + q^4) / (2 * q^2) ∧
    cos γ = (1 + q^2 - q^4) / (2 * q) := sorry

end angles_of_triangle_in_geometric_sequence_l537_537441


namespace sum_of_integers_l537_537438

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 54) : x + y = 21 :=
by
  sorry

end sum_of_integers_l537_537438


namespace final_problem_l537_537716

noncomputable def problem_statement (O : ℝ × ℝ × ℝ) (A B C : ℝ × ℝ × ℝ) (p q r : ℝ) : Prop :=
  let O := (0, 0, 0) in
  let d := 2 in
  let plane_eq := ∀ (x y z : ℝ), (x / (A.1) + y / (B.2) + z / (C.3) = 1) in
  let distance := 1 / sqrt ((1 / (A.1 ^ 2)) + (1 / (B.2 ^ 2)) + (1 / (C.3 ^ 2))) in
  let centroid := (p, q, r) = ((A.1 / 3), (B.2 / 3), (C.3 / 3)) in
  centroid ∧ distance = d → (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25

theorem final_problem : problem_statement (0, 0, 0) (a, 0, 0) (0, b, 0) (0, 0, c) (a / 3) (b / 3) (c / 3) :=
  sorry

end final_problem_l537_537716


namespace sheila_hourly_wage_l537_537414

-- Definition of conditions
def hours_per_day_mon_wed_fri := 8
def days_mon_wed_fri := 3
def hours_per_day_tue_thu := 6
def days_tue_thu := 2
def weekly_earnings := 432

-- Variables derived from conditions
def total_hours_mon_wed_fri := hours_per_day_mon_wed_fri * days_mon_wed_fri
def total_hours_tue_thu := hours_per_day_tue_thu * days_tue_thu
def total_hours_per_week := total_hours_mon_wed_fri + total_hours_tue_thu

-- Proof statement
theorem sheila_hourly_wage : (weekly_earnings / total_hours_per_week) = 12 := 
sorry

end sheila_hourly_wage_l537_537414


namespace sin_B_value_l537_537018

variable {A B C : Type} -- Define variables to represent the angles
variable {a b c : ℝ} -- Define variables to represent the sides

theorem sin_B_value (hA : Real.sin A = Real.sqrt 5 / 6) 
                    (ha : a = Real.sqrt 5) 
                    (hb : b = 2) 
                    (law_of_sines : Real.sin A / a = Real.sin B / b) : 
                    Real.sin B = 1 / 3 :=
by 
  sorry -- Placeholder for the actual proof

end sin_B_value_l537_537018


namespace gcd_g_x_l537_537269

def g (x : ℕ) : ℕ := (5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)

theorem gcd_g_x (x : ℕ) (hx : 17280 ∣ x) : Nat.gcd (g x) x = 120 :=
by sorry

end gcd_g_x_l537_537269


namespace locus_S_parallel_E_l537_537034

-- Definition of inputs and variables
variable {E : Type} [plane E]
variable {A B C : E} -- Three non-collinear points on one side of E
variable {A' B' C' : E} -- Arbitrary points on E
variable (L M N : E) -- Midpoints of AA', BB', and CC' respectively
variable (R T S : E) -- Centers of mass and centroid
hL : L = midpoint A A'
hM : M = midpoint B B'
hN : N = midpoint C C'
hS : S = centroid L M N

-- Problem statement: Prove the locus of S is parallel to E
theorem locus_S_parallel_E (hA : A ≠ B ∧ B ≠ C ∧ C ≠ A) (hA' : A' ∈ E) (hB' : B' ∈ E) (hC' : C' ∈ E) :
  ∃ (P : plane E), P ∥ E ∧ ∀ (A' B' C' : E), A' ∈ E → B' ∈ E → C' ∈ E → 
  S ∈ P :=
by
sorry

end locus_S_parallel_E_l537_537034


namespace sqrt_inequality_equality_condition_l537_537005

noncomputable def sqrt (x : Real) := Real.sqrt x

theorem sqrt_inequality (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_eq : x^2 + y^2 + z^2 + 3 = 2 * (x * y + y * z + z * x)) :
    sqrt (x * y) + sqrt (y * z) + sqrt (z * x) ≥ 3 :=
begin
  sorry
end

theorem equality_condition (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_eq : x^2 + y^2 + z^2 + 3 = 2 * (x * y + y * z + z * x)) :
    (sqrt (x * y) + sqrt (y * z) + sqrt (z * x) = 3) ↔ (x = 1 ∧ y = 1 ∧ z = 1) :=
begin
  sorry
end

end sqrt_inequality_equality_condition_l537_537005


namespace number_of_correct_conclusions_l537_537244

def f (ω x : ℝ) : ℝ := 2 * (Real.sin (ω * x + Real.pi / 3)) ^ 2 - 1

theorem number_of_correct_conclusions (ω : ℝ)
  (ω_pos : ω > 0)
  (c1 : ∀ (x1 x2 : ℝ), f ω x1 = 1 → f ω x2 = -1 → Real.abs (x1 - x2) = Real.pi → ω ≠ 1)
  (c2 : ∃ ω, 0 < ω ∧ ω < 2 ∧ ∀ x, f ω (x + Real.pi / 6) = f ω (-x + Real.pi / 6))
  (c3 : ∀ (x : ℝ), ω = 1 → -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → ¬(∀ x1 x2 : ℝ, x1 ≤ x2 → x1 ≤ x -> x ≤ x2 -> (f ω x1 ≤ f ω x2)))
  (c4 : ∀ (x : ℝ), f ω x = 0 → 0 ≤ x ∧ x < Real.pi → 5 ∃ ω, 29 / 12 < ω ∧ ω ≤ 35 / 12) :
  True := sorry

end number_of_correct_conclusions_l537_537244


namespace curve_is_circle_shortest_distance_l537_537700
noncomputable theory
open real

def curve_eq_polar (θ : ℝ) : ℝ := 2 * sqrt 2 * sin (θ - π / 4)

def curve_eq_cartesian (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

def Q : ℝ × ℝ := (sqrt 2 / 2, sqrt 2 / 2)

theorem curve_is_circle :
  ∀ (x y : ℝ), (∃ θ : ℝ, (x = (curve_eq_polar θ) * cos θ) ∧ (y = (curve_eq_polar θ) * sin θ)) ↔ curve_eq_cartesian x y := sorry

theorem shortest_distance (P : ℝ × ℝ)
    (hP : ∃ θ : ℝ, P = ((curve_eq_polar θ) * cos θ, (curve_eq_polar θ) * sin θ)) :
  dist P Q = sqrt 3 - sqrt 2 := sorry

end curve_is_circle_shortest_distance_l537_537700


namespace math_proof_l537_537188

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end math_proof_l537_537188


namespace xiao_ming_brother_age_l537_537081

def year := 2013
def x := 1995  -- This is calculated from the conditions as explained in the solution

/-- Prove that Xiao Ming's brother's age in 2013 is 18 given that his birth year is a multiple of 19 and the current year is the first with unique digits -/
theorem xiao_ming_brother_age (h1 : year = 2013)
    (h2: ∃ n : ℕ, x = 19 * n)
    (h3: ∀ y < year, y.digits.nodup) :
    year - x = 18 :=
by sorry

end xiao_ming_brother_age_l537_537081


namespace transport_safely_l537_537521

inductive Item
| Farmer
| Wolf
| Goat
| Cabbage

inductive Location
| Initial
| Opposite

open Item Location

def move_across_river (farmer : Location) (wolf : Location) (goat : Location) (cabbage : Location) (item : Item) : (Location × Location × Location × Location) :=
  match item, farmer with
  | Farmer, Initial     => (Opposite, wolf, goat, cabbage)
  | Farmer, Opposite    => (Initial, wolf, goat, cabbage)
  | Wolf, Initial       => (Opposite, Opposite, goat, cabbage)
  | Wolf, Opposite      => (Initial, Initial, goat, cabbage)
  | Goat, Initial       => (Opposite, wolf, Opposite, cabbage)
  | Goat, Opposite      => (Initial, wolf, Initial, cabbage)
  | Cabbage, Initial    => (Opposite, wolf, goat, Opposite)
  | Cabbage, Opposite   => (Initial, wolf, goat, Initial)
  | _, _ => sorry

def safe_condition (wolf : Location) (goat : Location) (cabbage : Location) (farmer : Location) : Prop :=
  (wolf ≠ goat ∨ wolf = farmer) ∧ (goat ≠ cabbage ∨ goat = farmer)

theorem transport_safely :
  ∃ (moves : List (Item × Location × Location × Location)), 
  (∀ (wolf goat cabbage farmer : Location), safe_condition wolf goat cabbage farmer) ∧
  (moves.head = (Farmer, Initial, Initial, Initial, Initial)) ∧
  (moves.last = some (Farmer, Opposite, Opposite, Opposite, Opposite)) :=
sorry

end transport_safely_l537_537521


namespace probability_prime_multiple_of_11_l537_537424

theorem probability_prime_multiple_of_11 : 
  let cards := finset.range 61
  let prime_multiple_of_11 := finset.filter (λ n, nat.prime n ∧ n % 11 = 0) cards
  let selected_card_probability := (prime_multiple_of_11.card : rat) / (cards.card : rat)
  selected_card_probability = 1 / 60 := 
by
  sorry

end probability_prime_multiple_of_11_l537_537424


namespace find_a_b_of_tangency_l537_537642

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a*x^2 + b*x

def tangent_line (x : ℝ) : ℝ := -3 * x + 8

-- Main statement
theorem find_a_b_of_tangency (a b : ℝ) (h_tangent : ∀ x, tangent_line x = f x a b ∧ deriv (f x a b) 2 = -3) :
  a = -6 ∧ b = 9 := sorry

end find_a_b_of_tangency_l537_537642


namespace determine_f_2010_l537_537380

open Real

variables {f : ℝ → ℝ} 

-- Define conditions
axiom f_pos (x : ℝ) (h : x > 0) : f x > 0
axiom f_eqn (x y : ℝ) (h : x > y) : f (x - y) = sqrt (f (x * y) + 2)

-- State the theorem
theorem determine_f_2010 (hf : ∀ x > 0, f x > 0) (hf' : ∀ x y > 0, x > y → f (x - y) = sqrt (f (x * y) + 2)) : f 2010 = 2 :=
by {
  sorry
}

end determine_f_2010_l537_537380


namespace find_tourism_function_find_peak_season_l537_537328

noncomputable def tourism_function (n : ℕ) : ℝ :=
  200 * Real.cos (Real.pi / 6 * n + 2 * Real.pi / 3) + 300

-- Condition 1: Period of 12 months and specific cosine function parameters
def tourism_function_periodic : Prop :=
  ∀ n, tourism_function n = 200 * Real.cos (Real.pi / 6 * n + 2 * Real.pi / 3) + 300

-- Condition 2: The difference between the number of workers in August and February is 400.
def tourism_diff_aug_feb : Prop :=
  tourism_function 8 - tourism_function 2 = 400

-- Condition 3: The number of workers in February is 100, and it increases till August.
def tourism_feb : Prop := 
  tourism_function 2 = 100

def tourism_increasing_feb_aug : Prop :=
  ∀ n, 2 ≤ n ∧ n ≤ 8 → tourism_function n.succ ≥ tourism_function n

-- Question 1: Prove the function is correct
theorem find_tourism_function :
  tourism_function_periodic ∧ tourism_diff_aug_feb ∧ tourism_feb ∧ tourism_increasing_feb_aug :=
sorry

-- Question 2: Prove peak season months are June to October
def peak_season (n : ℕ) : Prop :=
  tourism_function n ≥ 400

theorem find_peak_season (n : ℕ) :
  n ∈ {6, 7, 8, 9, 10} ↔ peak_season n :=
sorry

end find_tourism_function_find_peak_season_l537_537328


namespace prism_volume_l537_537849

theorem prism_volume (x y z : ℝ) (h1 : x * y = 18) (h2 : y * z = 12) (h3 : x * z = 8) :
  x * y * z = 24 * real.sqrt 3 :=
by
  sorry

end prism_volume_l537_537849


namespace cos_60_degrees_is_one_half_l537_537939

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537939


namespace g_at_seven_equals_92_l537_537311

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_at_seven_equals_92 : g 7 = 92 :=
by
  sorry

end g_at_seven_equals_92_l537_537311


namespace find_point_B_l537_537533

variable (A C : Point)
variable (plane : Plane)

def A : Point := { x := -4, y := 10, z := 13 }
def C : Point := { x := 5, y := 11, z := 13 }
def plane := { normal := { x := 2, y := 1, z := 1 }, constant := 16 }

theorem find_point_B : ∃ B : Point, 
  (reflects_on_plane A plane B) ∧ 
  (line_through B C) ∧
  (B = { x := 55/17, y := 193/17, z := 229/17 }) :=
sorry

end find_point_B_l537_537533


namespace necessary_but_not_sufficient_condition_l537_537067

variables (f : ℝ → ℝ) (x₀ : ℝ)
def p := deriv f x₀ = 0
def q := ∀ x, f x₀ ≤ f x ∨ f x₀ ≥ f x

theorem necessary_but_not_sufficient_condition :
  (∃ p q, deriv f x₀ = 0 ∧ (∀ x, f x₀ ≤ f x ∨ f x₀ ≥ f x)) →
  (p → q) ∧ ¬(q → p) :=
sorry

end necessary_but_not_sufficient_condition_l537_537067


namespace proposition_C_true_proposition_A_false_proposition_B_false_proposition_D_false_l537_537855

theorem proposition_C_true (a b : ℝ) : (1 / a = 1 / b) → (a = b) := 
begin
  intro h,
  have h1 : a * (1 / a) = b * (1 / b), by rw h,
  rw [mul_one, mul_one] at h1,
  assumption
end

-- Helper theorems for conditions A, B, and D (for completeness)
theorem proposition_A_false (a : ℝ) : (a^2 = 4) → (a = 2 ∨ a = -2) := sorry

theorem proposition_B_false (a b : ℝ) : (a = b) → (a ≥ 0) → (sqrt a = sqrt b) := sorry

theorem proposition_D_false (a b : ℝ) : (a < b) → (0 ≤ a) → (0 ≤ b) → ¬(a^2 < b^2) := sorry

end proposition_C_true_proposition_A_false_proposition_B_false_proposition_D_false_l537_537855


namespace closest_integer_to_series_sum_l537_537236

def primes (n : ℕ) : List ℕ := sorry -- function to list all prime factors of n
def count_primes_ge (n k : ℕ) : ℕ := 
  (primes n).filter (λ p, p ≥ k).length

noncomputable def series_sum : ℝ := 
  ∑' n, ∑' k, (count_primes_ge n k : ℝ) / 3 ^ (n + k - 7)

theorem closest_integer_to_series_sum : 
  ∃ (m : ℕ), series_sum = m ∧ m = 167 := sorry

end closest_integer_to_series_sum_l537_537236


namespace cos_60_eq_sqrt3_div_2_l537_537933

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537933


namespace geometric_series_an_sum_bn_l537_537248

noncomputable def a_n (n : ℕ) : ℝ := 3 ^ n

def b_n (n : ℕ) : ℝ := (2 * n - 1) * a_n n

def T_n (n : ℕ) : ℝ := (n - 1) * 3 ^ (n + 1) + 3

theorem geometric_series_an (n : ℕ) (h1 : ∀ n, a_n n > 0) (h2 : a_n 2 * a_n 3 = a_n 5) (h3 : (a_n 1 + a_n 2 + a_n 3 + a_n 4) = 10 * (a_n 1 + a_n 2)) :
  a_n n = 3 ^ n :=
sorry

theorem sum_bn (n : ℕ) (h1 : ∀ n, a_n n > 0) (h2 : a_n 2 * a_n 3 = a_n 5) (h3 : (a_n 1 + a_n 2 + a_n 3 + a_n 4) = 10 * (a_n 1 + a_n 2)):
  let S_n := (λ n, b_n n) in
  ∑ i in Finset.range n, S_n i = T_n n :=
sorry

end geometric_series_an_sum_bn_l537_537248


namespace count_valid_squares_on_6_by_6_checkerboard_l537_537130

/-- A 6 by 6 checkerboard has alternating black and white squares. -/
def alternating_checkerboard (n : ℕ) : Prop := n = 6 ∧ ∀ i j : ℕ, (i < 6 ∧ j < 6) → (i + j) % 2 = 0 ↔ (i, j) ∈ set_of (id)

/-- Count the distinct squares of various sizes on a checkerboard that contain at least 3 black squares. -/
def count_valid_squares (n : ℕ) : ℕ :=
  let one_by_one := 0 in
  let two_by_two := 0 in
  let three_by_three := 8 in
  let four_by_four := 9 in
  let five_by_five := 4 in
  let six_by_six := 1 in
  one_by_one + two_by_two + three_by_three + four_by_four + five_by_five + six_by_six

/-- There are exactly 22 distinct squares on a 6 by 6 checkerboard that contain at least 3 black squares. -/
theorem count_valid_squares_on_6_by_6_checkerboard : 
  alternating_checkerboard 6 → count_valid_squares 6 = 22 :=
by
  intros h_checkerboard
  simp only [count_valid_squares]
  exact add_zero (add_zero (add_zero (add 0 (add 8 (add 9 (add 4 1))))))

print axioms count_valid_squares_on_6_by_6_checkerboard

end count_valid_squares_on_6_by_6_checkerboard_l537_537130


namespace probability_of_drawing_3_black_and_2_white_l537_537516

noncomputable def total_ways_to_draw_5_balls : ℕ := Nat.choose 27 5
noncomputable def ways_to_choose_3_black : ℕ := Nat.choose 10 3
noncomputable def ways_to_choose_2_white : ℕ := Nat.choose 12 2
noncomputable def favorable_outcomes : ℕ := ways_to_choose_3_black * ways_to_choose_2_white
noncomputable def desired_probability : ℚ := favorable_outcomes / total_ways_to_draw_5_balls

theorem probability_of_drawing_3_black_and_2_white :
  desired_probability = 132 / 1345 := by
  sorry

end probability_of_drawing_3_black_and_2_white_l537_537516


namespace average_distinct_s_l537_537280

theorem average_distinct_s (s : ℕ) :
  (∀ a b : ℕ, a + b = 7 → a * b = s → ∃ r : ℚ, r = 28 / 3) :=
begin
  sorry
end

end average_distinct_s_l537_537280


namespace cone_volume_is_correct_l537_537815

-- Given conditions
def slant_height : ℝ := 15
def height : ℝ := 9

-- Derived quantities
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- Volume formula for a cone
def volume_cone (r h : ℝ) := (1 / 3) * π * r^2 * h

theorem cone_volume_is_correct : volume_cone radius height = 432 * π := by
  sorry

end cone_volume_is_correct_l537_537815


namespace sequence_formula_max_k_l537_537253

noncomputable def a_sequence (n : ℕ) : ℕ :=
  if n = 1 then 2 else (a_sequence (n - 1) + 2)

theorem sequence_formula (n : ℕ) (h : n ≥ 1) : 
  a_sequence n = 2 ^ n := 
sorry

noncomputable def b_sequence (n : ℕ) : ℝ :=
  1 / Real.log (2 ^ (a_sequence n))

noncomputable def T_n_sequence (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n) \ Finset.range (n+1), b_sequence (i : ℕ + 1)

theorem max_k (k : ℕ) (h : ∀ n : ℕ, T_n_sequence n > k / 12) : 
  k ≤ 5 := 
sorry

end sequence_formula_max_k_l537_537253


namespace fraction_equation_solution_l537_537817

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) : 
  (1 / (x - 2) = 3 / x) → x = 3 := 
by 
  sorry

end fraction_equation_solution_l537_537817


namespace coloring_problem_l537_537198

-- Define the coloring condition that no two connected vertices can have the same color
def valid_coloring (dots : ℕ) (colors : ℕ) (coloring : Fin dots → Fin colors) (connections : Fin dots → Fin dots → Prop) :=
  ∀ (v₁ v₂ : Fin dots), connections v₁ v₂ → coloring v₁ ≠ coloring v₂

-- Define the geometric problem conditions
def central_triangle_conditions (A B C : Fin 12) (coloring : Fin 12 → Fin 4) :=
  coloring A ≠ coloring B ∧ coloring B ≠ coloring C ∧ coloring A ≠ coloring C

def hexagon_conditions (A B C D E F G H I J K L : Fin 12) (coloring : Fin 12 → Fin 4) :=
  valid_coloring 12 4 coloring (λ (v₁ v₂), 
    (v₁ = A ∧ v₂ = B) ∨ (v₁ = B ∧ v₂ = C) ∨ (v₁ = C ∧ v₂ = D) ∨ (v₁ = D ∧ v₂ = E) ∨ 
    (v₁ = E ∧ v₂ = F) ∨ (v₁ = F ∧ v₂ = G) ∨ (v₁ = G ∧ v₂ = H) ∨ (v₁ = H ∧ v₂ = I) ∨ 
    (v₁ = I ∧ v₂ = J) ∨ (v₁ = J ∧ v₂ = K) ∨ (v₁ = K ∧ v₂ = L) ∨ (v₁ = L ∧ v₂ = A)
  )

-- Define the theorem for the problem
theorem coloring_problem (A B C D E F G H I J K L : Fin 12) : 
  ∃ (coloring : Fin 12 → Fin 4), central_triangle_conditions A B C coloring ∧ hexagon_conditions A B C D E F G H I J K L coloring :=
  sorry

end coloring_problem_l537_537198


namespace packets_needed_l537_537404

noncomputable def ounces_in_ton := 2200 * 16
noncomputable def gunny_bag_capacity_ounces := 13.5 * ounces_in_ton
noncomputable def grams_to_ounces := 0.035274 * 350
noncomputable def total_packet_weight_ounces := 16 * 16 + 4 + grams_to_ounces

theorem packets_needed : 
  let packets : ℝ := gunny_bag_capacity_ounces / total_packet_weight_ounces in 
  int.ceil packets = 1745 :=
by
  sorry

end packets_needed_l537_537404


namespace angle_between_vectors_l537_537624

open Real

theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 2)
  (h3 : (a - b) ⬝ a = 2) :
  real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) = π * 3 / 4 := sorry

end angle_between_vectors_l537_537624


namespace cos_diff_identity_sin_sum_identity_l537_537125

-- Problem 1
theorem cos_diff_identity (α : ℝ) :
  (cos (3 * α) - cos α) / (2 * sin α * sin (2 * α)) = -1 := 
by 
  sorry

-- Problem 2
theorem sin_sum_identity (α : ℝ) :
  (sin (3 * α) + sin α) / (2 * cos α * sin (2 * α)) = 1 := 
by 
  sorry

end cos_diff_identity_sin_sum_identity_l537_537125


namespace number_of_sides_of_regular_polygon_l537_537975

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l537_537975


namespace find_value_of_a_l537_537609

theorem find_value_of_a (a : ℝ) (h : (3 + a + 10) / 3 = 5) : a = 2 := 
by {
  sorry
}

end find_value_of_a_l537_537609


namespace max_min_magnitude_m_l537_537350

theorem max_min_magnitude_m (z1 z2 m : ℂ) (α β : ℂ) 
    (h_equation : ∀ x : ℂ, x^2 + z1 * x + z2 + m = 0 ↔ (x - α) * (x - β) = 0)
    (h_discriminant : z1^2 - 4 * z2 = 16 + 20 * complex.I)
    (h_root_diff : complex.abs (α - β) = 2 * real.sqrt 7) :
    real.abs m = 7 + real.sqrt 41 ∨ real.abs m = 7 - real.sqrt 41 :=
sorry

end max_min_magnitude_m_l537_537350


namespace arnold_protein_intake_l537_537180

theorem arnold_protein_intake :
  (∀ p q s : ℕ,  p = 18 / 2 ∧ q = 21 ∧ s = 56 → (p + q + s = 86)) := by
  sorry

end arnold_protein_intake_l537_537180


namespace sequence_general_term_l537_537032

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 3 * (Finset.range (n + 1)).sum a = (n + 2) * a n) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l537_537032


namespace minimum_value_of_f_l537_537738

noncomputable def f (x m : ℝ) := (1 / 3) * x^3 - x + m

theorem minimum_value_of_f (m : ℝ) (h_max : f (-1) m = 1) : 
  f 1 m = -1 / 3 :=
by
  sorry

end minimum_value_of_f_l537_537738


namespace consecutive_negatives_product_sum_l537_537803

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l537_537803


namespace solution_set_of_inequality_l537_537451

theorem solution_set_of_inequality (x : ℝ) : {x | x * (x - 1) > 0} = { x | x < 0 } ∪ { x | x > 1 } :=
sorry

end solution_set_of_inequality_l537_537451


namespace exists_single_shortest_arc_tetrahedron_l537_537918

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def isosceles_triangle (A B T : Point) : Prop :=
  ⦃P Q R : Point⦄, angle A T B = 30 ∧ angle B T C = 30 ∧ angle C T A = 30

noncomputable def distance (A B : Point) : ℝ := minimal_length_path A B

theorem exists_single_shortest_arc_tetrahedron (M T : Point) :
  ∃ (A B C T : Point), 
  equilateral_triangle A B C ∧ 
  isosceles_triangle A B T ∧ 
  isosceles_triangle B C T ∧
  isosceles_triangle C A T ∧ 
  ∀ (Q : Point), distance M Q = distance M T → Q ≠ T → ∃! (arc : Path M Q), length arc = distance M Q :=
begin
  sorry  
end

end exists_single_shortest_arc_tetrahedron_l537_537918


namespace lines_tangent_to_parabola_at_one_point_l537_537524

theorem lines_tangent_to_parabola_at_one_point :
  let A : ℝ × ℝ := (-1, 0)
  let parabola (x : ℝ) : ℝ := x^2
  ∃ (lines : List (ℝ × ℝ → ℝ)), 
    lines.length = 3 ∧ 
    ∀ line ∈ lines, 
      let b := line.1
      let m := line.2
      ( ∀ x : ℝ, m * (x + 1) ≠ parabola x ) →
      ( −(parabola (-1)) = b ∨ parabola (-1) + m * (-1) = b ∨ (∀ x : ℝ, x = -1 → m * (x + 1) = parabola x) )
:= sorry

end lines_tangent_to_parabola_at_one_point_l537_537524


namespace solve_problem_l537_537659

-- Definitions from the conditions
def is_divisible_by (n k : ℕ) : Prop :=
  ∃ m, k * m = n

def count_divisors (limit k : ℕ) : ℕ :=
  Nat.div limit k

def count_numbers_divisible_by_neither_5_nor_7 (limit : ℕ) : ℕ :=
  let total := limit - 1
  let divisible_by_5 := count_divisors limit 5
  let divisible_by_7 := count_divisors limit 7
  let divisible_by_35 := count_divisors limit 35
  total - (divisible_by_5 + divisible_by_7 - divisible_by_35)

-- The statement to be proved
theorem solve_problem : count_numbers_divisible_by_neither_5_nor_7 1000 = 686 :=
by
  sorry

end solve_problem_l537_537659


namespace find_omega_l537_537644

theorem find_omega (ω : Real) (h : ∀ x : Real, (1 / 2) * Real.cos (ω * x - (Real.pi / 6)) = (1 / 2) * Real.cos (ω * (x + Real.pi) - (Real.pi / 6))) : ω = 2 ∨ ω = -2 :=
by
  sorry

end find_omega_l537_537644


namespace matching_sum_l537_537883

def row_num (i j : ℕ) : ℕ := 23 * (i - 1) + j
def col_num (i j : ℕ) : ℕ := 15 * (j - 1) + i

theorem matching_sum :
  (row_num 4 4 + row_num 7 11 + row_num 10 18) = 629 :=
by
  have h1 : row_num 4 4 = col_num 4 4 := by simp [row_num, col_num]
  have h2 : row_num 7 11 = col_num 7 11 := by simp [row_num, col_num]
  have h3 : row_num 10 18 = col_num 10 18 := by simp [row_num, col_num]
  calc
    row_num 4 4 + row_num 7 11 + row_num 10 18 
    = col_num 4 4 + col_num 7 11 + col_num 10 18 : by simp [h1, h2, h3]
    ... = 78 + 192 + 359 : by simp 
    ... = 629 : by simp

end matching_sum_l537_537883


namespace cauchy_schwarz_inequality_l537_537723

variable {n : ℕ} (x : Fin n → ℝ)

theorem cauchy_schwarz_inequality (h : ∀ i, x i > 0) :
    (∑ i, x i) * (∑ i, 1 / x i) ≥ n ^ 2 :=
by 
  sorry

end cauchy_schwarz_inequality_l537_537723


namespace inscribable_quadrilateral_l537_537418

theorem inscribable_quadrilateral
  (a b c d : ℝ)
  (A : ℝ)
  (circumscribable : Prop)
  (area_condition : A = Real.sqrt (a * b * c * d))
  (A := Real.sqrt (a * b * c * d)) : 
  circumscribable → ∃ B D : ℝ, B + D = 180 :=
sorry

end inscribable_quadrilateral_l537_537418


namespace cosine_60_degrees_l537_537947

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537947


namespace complete_the_square_l537_537109

theorem complete_the_square (x : ℝ) :
  x^2 + 6 * x - 4 = 0 → (x + 3)^2 = 13 :=
by
  sorry

end complete_the_square_l537_537109


namespace proof_l537_537261

-- Defining propositions
def p : Prop := ∀ (parallelogram : Type) (a b : parallelogram), bisects (diagonals a b)
def q : Prop := ∀ (parallelogram : Type) (a b : parallelogram), equals (diagonals a b)

-- Given conditions
axiom hp : p  -- p is true
axiom hnotq : ¬ q  -- q is false

-- Goal: Prove that ¬p ∨ ¬q is true given the conditions
theorem proof : (¬p ∨ ¬q) :=
by {
  right,
  exact hnotq,
}

end proof_l537_537261


namespace mutually_exclusive_A_B_l537_537327

variable (cards : Set (Fin 4)) (select_two : Finset (Fin 4) → Set (Finset (Fin 4)))
variable (A B : Set (Finset (Fin 4)))

def sample_space : Set (Finset (Fin 4)) :=
  {s | s ⊆ cards ∧ s.card = 2}

def event_A : Set (Finset (Fin 4)) :=
  {s | s = {⟨0, by norm_num⟩, ⟨2, by norm_num⟩}}

def event_B : Set (Finset (Fin 4)) :=
  {s | ∃ a b ∈ cards, s = {a, b} ∧ (a.val + b.val + 2 = 5)}

def mutually_exclusive_events (A B : Set (Finset (Fin 4))) : Prop :=
  A ∩ B = ∅

theorem mutually_exclusive_A_B : mutually_exclusive_events event_A event_B :=
  by
    sorry

end mutually_exclusive_A_B_l537_537327


namespace sin_870_degree_cos_870_degree_l537_537958

theorem sin_870_degree (d : ℝ) (h : d = 870) : Real.sin (d * Real.pi / 180) = 1 / 2 := 
by
  rw [h, Real.sin_eq_sin_pi_sub_pi_180]
  norm_num

theorem cos_870_degree (d : ℝ) (h : d = 870) : Real.cos (d * Real.pi / 180) = - (Real.sqrt 3 / 2) := 
by
  rw [h, Real.cos_eq_neg_cos_pi_180_sub_pi_180]
  norm_num

end sin_870_degree_cos_870_degree_l537_537958


namespace find_third_number_l537_537121

theorem find_third_number
  (A B C : ℤ)
  (hA : A = 24)
  (hB : B = 36)
  (h_hcf : HCF(A, B, C) = 42)
  (h_lcm : LCM(A, B, C) = 5616) :
  C = 273 :=
by 
  sorry

end find_third_number_l537_537121


namespace no_bounded_sequence_a1_gt_2015_l537_537449

theorem no_bounded_sequence_a1_gt_2015 (a1 : ℚ) (h_a1 : a1 > 2015) : 
  ∀ (a_n : ℕ → ℚ), a_n 1 = a1 → 
  (∀ (n : ℕ), ∃ (p_n q_n : ℕ), p_n > 0 ∧ q_n > 0 ∧ (p_n.gcd q_n = 1) ∧ (a_n n = p_n / q_n) ∧ 
  (a_n (n + 1) = (p_n^2 + 2015) / (p_n * q_n))) → 
  ∃ (M : ℚ), ∀ (n : ℕ), a_n n ≤ M → 
  False :=
sorry

end no_bounded_sequence_a1_gt_2015_l537_537449


namespace car_distance_on_highway_l537_537877

theorem car_distance_on_highway (x : ℝ) : 
  let local_road_distance := 90
  let local_road_speed := 30
  let highway_speed := 60
  let average_speed := 38.82
  let total_distance := local_road_distance + x
  let total_time := (local_road_distance / local_road_speed) + (x / highway_speed)
  average_speed = total_distance / total_time →
  x = 74.95 :=
by 
  intros 
  have h1 : local_road_distance / local_road_speed = 3 := by sorry
  have h2 : x / highway_speed = x / 60 := by sorry
  have h3 : total_time = 3 + (x / 60) := by sorry
  have h4 : total_distance = local_road_distance + x := by sorry
  sorry

end car_distance_on_highway_l537_537877


namespace tom_candy_pieces_l537_537457

def total_boxes : ℕ := 14
def give_away_boxes : ℕ := 8
def pieces_per_box : ℕ := 3

theorem tom_candy_pieces : (total_boxes - give_away_boxes) * pieces_per_box = 18 := 
by 
  sorry

end tom_candy_pieces_l537_537457


namespace area_of_triangle_correct_l537_537097

noncomputable def area_of_triangle
  (A B C D : Type) 
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (AC AD DC AB : ℝ)
  (hAC : AC = 17)
  (hAB : AB = 19)
  (hDC : DC = 8)
  (right_angle_D : ∃ (AD AC DC : ℝ), Real.sqrt ((AC)^2 - (DC)^2) = AD) : ℝ :=
  30 * Real.sqrt 34 + 60

theorem area_of_triangle_correct 
  (A B C D : Type) 
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] :
  ∀ (AC AD DC AB : ℝ)
  (hAC : AC = 17)
  (hAB : AB = 19)
  (hDC : DC = 8)
  (right_angle_D : ∃ (AD AC DC : ℝ), Real.sqrt ((AC)^2 - (DC)^2) = AD),
  area_of_triangle A B C D AC AD DC AB hAC hAB hDC right_angle_D = 30 * Real.sqrt 34 + 60 :=
sorry

end area_of_triangle_correct_l537_537097


namespace eighth_odd_multiple_of_5_is_75_l537_537478

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0) ∧ (n % 2 = 1) ∧ (n % 5 = 0) ∧ (nat.find_greatest (λ m, m % 2 = 1 ∧ m % 5 = 0) 75 = 75) :=
    sorry

end eighth_odd_multiple_of_5_is_75_l537_537478


namespace volume_ratio_l537_537100

-- Define the edge lengths
def edge_length_cube1 : ℝ := 4 -- in inches
def edge_length_cube2 : ℝ := 2 * 12 -- 2 feet converted to inches

-- Define the volumes
def volume_cube (a : ℝ) : ℝ := a ^ 3

-- Statement asserting the ratio of the volumes is 1/216
theorem volume_ratio : volume_cube edge_length_cube1 / volume_cube edge_length_cube2 = 1 / 216 :=
by
  -- This is the placeholder to skip the proof
  sorry

end volume_ratio_l537_537100


namespace eventually_constant_set_of_integers_l537_537765

-- Definitions for gcd and lcm
def gcd (a b : Nat) := Nat.gcd a b
def lcm (a b : Nat) := a * b / gcd a b

theorem eventually_constant_set_of_integers (S : List Nat) (h : ∀ n ∈ S, n > 0): 
  ∃ S', S' = S ∧ (∀ {a b : Nat}, a ∈ S' ∧ b ∈ S' → 
    gcd a b ∈ S' ∧ lcm a b ∈ S' ∧ S = S') := 
sorry

end eventually_constant_set_of_integers_l537_537765


namespace number_of_sides_of_regular_polygon_l537_537974

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l537_537974


namespace arithmetic_sequence_l537_537790

-- Given conditions
variables {a x b : ℝ}

-- Statement of the problem in Lean 4
theorem arithmetic_sequence (h1 : x - a = b - x) (h2 : b - x = 2 * x - b) : a / b = 1 / 3 :=
sorry

end arithmetic_sequence_l537_537790


namespace problem_proof_l537_537427

noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, x ≥ 1 → h (k x) = x^3
axiom cond2 : ∀ x : ℝ, x ≥ 1 → k (h x) = x^4
axiom cond3 : k 81 = 81

theorem problem_proof : [k 9]^4 = 43046721 :=
by
  sorry

end problem_proof_l537_537427


namespace proof_problem_l537_537789

variables {A B : ℤ}

def condition1 (x : ℤ) : Prop := (10 * x^2 - 31 * x + 21) = (A * x - 7) * (B * x - 3)
def condition2 : Prop := A * B = 10
def condition3 : Prop := -7 * B - 3 * A = -31
def condition4 : Prop := 7 * 3 = 21

theorem proof_problem (h1 : ∀ x, condition1 x) (h2 : condition2) (h3 : condition3) (h4 : condition4) : A * B + A = 15 :=
sorry

end proof_problem_l537_537789


namespace sum_of_cos_series_l537_537512

theorem sum_of_cos_series :
  6 * Real.cos (18 * Real.pi / 180) + 2 * Real.cos (36 * Real.pi / 180) + 
  4 * Real.cos (54 * Real.pi / 180) + 6 * Real.cos (72 * Real.pi / 180) + 
  8 * Real.cos (90 * Real.pi / 180) + 10 * Real.cos (108 * Real.pi / 180) + 
  12 * Real.cos (126 * Real.pi / 180) + 14 * Real.cos (144 * Real.pi / 180) + 
  16 * Real.cos (162 * Real.pi / 180) + 18 * Real.cos (180 * Real.pi / 180) + 
  20 * Real.cos (198 * Real.pi / 180) + 22 * Real.cos (216 * Real.pi / 180) + 
  24 * Real.cos (234 * Real.pi / 180) + 26 * Real.cos (252 * Real.pi / 180) + 
  28 * Real.cos (270 * Real.pi / 180) + 30 * Real.cos (288 * Real.pi / 180) + 
  32 * Real.cos (306 * Real.pi / 180) + 34 * Real.cos (324 * Real.pi / 180) + 
  36 * Real.cos (342 * Real.pi / 180) + 38 * Real.cos (360 * Real.pi / 180) = 10 :=
by
  sorry

end sum_of_cos_series_l537_537512


namespace value_of_bc_l537_537671

theorem value_of_bc (a b c d : ℝ) (h1 : a + b = 14) (h2 : c + d = 3) (h3 : a + d = 8) : b + c = 9 :=
sorry

end value_of_bc_l537_537671


namespace sequence_limit_l537_537370

noncomputable def sequence_converges (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n > 1 ∧ a (n + 1) ^ 2 ≥ a n * a (n + 2)

theorem sequence_limit (a : ℕ → ℝ) (h : sequence_converges a) : 
  ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (Real.log (a (n + 1)) / Real.log (a n) - l) < ε := 
sorry

end sequence_limit_l537_537370


namespace area_of_sector_l537_537902

-- Let's start by defining the given conditions and the target statement. 
variables (R : ℝ) (l : ℝ)

-- Given conditions
def perimeter_condition (R l : ℝ) : Prop := 2 * R + l = 4 * R
def arc_length (l R : ℝ) : ℝ := 2 * R

-- The statement to be proven (the area of the sector)
def sector_area (R l : ℝ) : ℝ := (1 / 2) * l * R

-- The final proof problem statement
theorem area_of_sector (R : ℝ) (h : perimeter_condition R (arc_length R)) :
  sector_area R (arc_length R) = R^2 :=
sorry

end area_of_sector_l537_537902


namespace angle_ACA1_eq_angle_BDB1_l537_537727

variables {A B C D O A1 B1 : Type}
(h_trap : AB ∥ CD)
(h_inter : ∃ O, O = (diagonal_intersection ABCD))
(h_symm : (∀ A B A1 B1 : Type, symmetric_points A1 B1 O))

theorem angle_ACA1_eq_angle_BDB1 
  (h_O : O = diagonal_intersection ABCD)
  (h_AB_parallel_CD : AB ∥ CD)
  (h_symmetry_A1 : symmetric_point A A1 (angle_bisector A O B))
  (h_symmetry_B1 : symmetric_point B B1 (angle_bisector A O B)) :
  angle A C A1 = angle B D B1 :=
sorry

end angle_ACA1_eq_angle_BDB1_l537_537727


namespace hyperbola_range_m_l537_537439

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m + 2 > 0 ∧ m - 2 < 0) ∧ (x^2 / (m + 2) + y^2 / (m - 2) = 1)) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l537_537439


namespace seating_arrangement_l537_537691

theorem seating_arrangement :
  ∃ (C R Y : ℕ), 
    C = 4 ∧ 
    R = 4 ∧ 
    Y = 2 ∧ 
    (3! * 4! * 4! * 2! = 6912) := 
begin
  use [4, 4, 2],
  simp,
  split,
  refl,
  split,
  refl,
  split,
  refl,
  calc 3! * 4! * 4! * 2! = 6 * 24 * 24 * 2 : by refl
  ... = 6912 : by norm_num
end

end seating_arrangement_l537_537691


namespace function_minimum_value_no_maximum_l537_537618

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.sin x + a) / Real.sin x

theorem function_minimum_value_no_maximum (a : ℝ) (h_a : 0 < a) : 
  ∃ x_min, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≥ x_min ∧ 
           (∀ x ∈ Set.Ioo 0 Real.pi, f a x ≠ x_min) ∧ 
           ¬ (∃ x_max, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≤ x_max) :=
by
  let t := Real.sin
  have h : ∀ x ∈ Set.Ioo 0 Real.pi, t x ∈ Set.Ioo 0 1 := sorry -- Simple property of sine function in (0, π)
  -- Exact details skipped to align with the conditions from the problem, leveraging the property
  sorry -- Full proof not required as per instructions

end function_minimum_value_no_maximum_l537_537618


namespace regular_polygon_sides_l537_537996

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537996


namespace regular_polygon_sides_l537_537992

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537992


namespace intersection_complement_eq_l537_537298

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set M within U
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N within U
def N : Set ℕ := {5, 6, 7}

-- Define the complement of M in U
def CU_M : Set ℕ := U \ M

-- Define the complement of N in U
def CU_N : Set ℕ := U \ N

-- Mathematically equivalent proof problem
theorem intersection_complement_eq : CU_M ∩ CU_N = {2, 4, 8} := by
  sorry

end intersection_complement_eq_l537_537298


namespace find_n_l537_537219

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

theorem find_n (n : ℕ) (h : n * factorial (n + 1) + factorial (n + 1) = 5040) : n = 5 :=
sorry

end find_n_l537_537219


namespace g_is_odd_function_l537_537357

def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1/3

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g(x) := by
  intro x
  have h : g(-x) = -(g(x)) -- Show intermediate steps
  sorry -- Proof steps

end g_is_odd_function_l537_537357


namespace prime_fraction_sum_l537_537658

theorem prime_fraction_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
    (h : a + b + c + a * b * c = 99) :
    |(1 / a : ℚ) - (1 / b : ℚ)| + |(1 / b : ℚ) - (1 / c : ℚ)| + |(1 / c : ℚ) - (1 / a : ℚ)| = 9 / 11 := 
sorry

end prime_fraction_sum_l537_537658


namespace cover_points_with_non_overlapping_circles_l537_537755

theorem cover_points_with_non_overlapping_circles (n : ℕ) (n_pos : 0 < n)
  (points : fin n → ℝ × ℝ) :
  ∃ (k : ℕ) (circles : fin k → (ℝ × ℝ) × ℝ), k > 0 ∧
  (∀ i j, i ≠ j → dist (circles i).fst (circles j).fst > 1) ∧
  (∑ i, 2 * (circles i).snd < n) ∧
  (∀ p, p ∈ points → ∃ i, dist p (circles i).fst ≤ (circles i).snd) :=
sorry

end cover_points_with_non_overlapping_circles_l537_537755


namespace total_hours_l537_537047

variables (K P M : ℕ) (pat_kate_eq : P = 2 * K) (pat_mark_eq : P = (1/3 : ℚ) * M) (mark_kate_eq : M = K + 75)

theorem total_hours : 
  ∃ K P M, (P = 2 * K) ∧ (P = (1/3 : ℚ) * M) ∧ (M = K + 75) ∧ (P + K + M = 135) :=
by
  -- Existential quantification
  use 15, 30, 90
  split
  -- Showing P = 2 * K
  { exact rfl }
  split
  -- Showing P = (1/3 : ℚ) * M
  { exact rfl }
  split
  -- Showing M = K + 75
  { exact rfl }
  -- Showing P + K + M = 135
  norm_num
  exact rfl

end total_hours_l537_537047


namespace same_graphs_at_x_eq_1_l537_537852

theorem same_graphs_at_x_eq_1 :
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  y2 = 3 ∧ y3 = 3 ∧ y1 ≠ y2 := 
by
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  sorry

end same_graphs_at_x_eq_1_l537_537852


namespace weight_of_B_l537_537430

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l537_537430


namespace tank_length_is_six_l537_537466

noncomputable def length_of_tank (rate time : ℝ) (width depth : ℝ) : ℝ :=
  (rate * time) / (width * depth)

theorem tank_length_is_six
  (rate : ℝ) (time : ℝ) (width : ℝ) (depth : ℝ) 
  (rate_eq : rate = 4) (time_eq : time = 18)
  (width_eq : width = 4) (depth_eq : depth = 3) :
  length_of_tank rate time width depth = 6 :=
by
  rw [rate_eq, time_eq, width_eq, depth_eq]
  simp [length_of_tank]
  norm_num

#eval tank_length_is_six 4 18 4 3 rfl rfl rfl rfl -- This evaluates to true and validates our theorem.

end tank_length_is_six_l537_537466


namespace divisor_is_31_l537_537675

-- Definition of the conditions.
def condition1 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 62 * k + 7

def condition2 (x y : ℤ) : Prop :=
  ∃ m : ℤ, x + 11 = y * m + 18

-- Main statement asserting the divisor y.
theorem divisor_is_31 (x y : ℤ) (h₁ : condition1 x) (h₂ : condition2 x y) : y = 31 :=
sorry

end divisor_is_31_l537_537675


namespace flowers_died_l537_537565

theorem flowers_died : 
  let initial_flowers := 2 * 5
  let grown_flowers := initial_flowers + 20
  let harvested_flowers := 5 * 4
  grown_flowers - harvested_flowers = 10 :=
by
  sorry

end flowers_died_l537_537565


namespace volume_difference_times_pi_l537_537169

noncomputable def alice_height := 9
noncomputable def alice_circumference := 7
noncomputable def bob_height := 7
noncomputable def bob_circumference := 9

noncomputable def pi := Real.pi

noncomputable def volume (height circumference : ℝ) : ℝ :=
  let radius := circumference / (2 * pi)
  π * (radius^2) * height

theorem volume_difference_times_pi :
  let V_A := volume alice_height alice_circumference
  let V_B := volume bob_height bob_circumference
  π * (Float.abs (V_A - V_B)) = 850.5 :=
by
  let V_A := volume alice_height alice_circumference
  let V_B := volume bob_height bob_circumference
  sorry

end volume_difference_times_pi_l537_537169


namespace unique_determination_of_B1_l537_537544

-- Define a convex polygon with 2020 vertices and angles
variables {A : Type} [AddCommGroup A] [VectorSpace ℂ A]
variable (vertices : Fin 2020 → A)
variables (θ : Fin 2020 → ℝ)
variable (B : Fin 2020 → A)

-- Define the conditions
def is_convex_2020_gon (vertices : Fin 2020 → A) := 
  ∀ (i : Fin 2020), True  -- placeholder for the actual condition of convexity

def angles_sum_to_1010pi (θ : Fin 2020 → ℝ) :=
  ∑ i, θ i = 1010 * Real.pi

def isosceles_triangles (vertices : Fin 2020 → A) (B : Fin 2020 → A) (θ : Fin 2020 → ℝ) :=
  ∀ i, ‖B i - vertices i‖ = ‖B i - vertices (Fin.add i 1)‖ ∧ 
       (angle (B i) (vertices i) (vertices (Fin.add i 1)) = θ i)

-- Main theorem statement
theorem unique_determination_of_B1 (vertices : Fin 2020 → A) (θ : Fin 2020 → ℝ) (B : Fin 2020 → A)
  (h1 : is_convex_2020_gon vertices)
  (h2 : angles_sum_to_1010pi θ)
  (h3 : isosceles_triangles vertices B θ) : 
  ∃! B1, 
    ∀i : Fin 2019, 
    B1 ∈ range B := by
  sorry

end unique_determination_of_B1_l537_537544


namespace number_of_paychecks_l537_537302

theorem number_of_paychecks (P : ℕ) (contribution_per_paycheck : ℕ) (company_match : ℝ)
  (total_contribution : ℕ) (h_contribution : contribution_per_paycheck = 100)
  (h_company_match : company_match = 0.06)
  (h_total_contribution : total_contribution = 2756):
  P = 26 :=
by
  have total_contribution_per_paycheck : ℕ := contribution_per_paycheck + (contribution_per_paycheck * company_match).toNat
  have total_contribution_calculated : ℕ := total_contribution_per_paycheck * P
  have h_total_contribution_calc_eq : total_contribution_calculated = 2756
  rw [←h_total_contribution, total_contribution_calculated] at h_total_contribution_calc_eq
  sorry

end number_of_paychecks_l537_537302


namespace probability_of_black_hen_in_daytime_l537_537779

noncomputable def probability_black_hen_in_daytime_program : ℝ → ℝ 
| 5 -> 4 == 0.922 where 
  daytime_spots == 2,
  evening_spots == 3,
  total_hens == 4,
  black_hens == 3,
  white_hens == 1,
  random_submission := true,
  equal_probability_program_choice := true,
  precise_probability := 0.922

theorem probability_of_black_hen_in_daytime : 
  probability_black_hen_in_daytime_program(2,3,4,3,1) == 0.922 :=
begin
  -- proof omitted (not required)
sorry
end

end probability_of_black_hen_in_daytime_l537_537779


namespace problem_statement_l537_537906

open Real

def f (x : ℝ) : ℝ := x / (1 + abs x)

theorem problem_statement :
  (∀ x : ℝ, f (-x) + f x = 0) ∧
  (∀ y : ℝ, y ∈ Set.range f ↔ y ∈ Ioo (-1 : ℝ) (1 : ℝ)) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
  ¬ (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0) :=
by
  sorry

end problem_statement_l537_537906


namespace log_sum_gt_four_l537_537407

theorem log_sum_gt_four :
  (log 5 6) + (log 6 7) + (log 7 8) + (log 8 5) > 4 := by
  sorry

end log_sum_gt_four_l537_537407


namespace distinct_s_average_is_28_over_3_l537_537277

theorem distinct_s_average_is_28_over_3 (a b : ℕ) (s : ℕ) :
  (∀ a b, a + b = 7 → a * b = s) →
  (∀ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ a < 7 ∧ b < 7) →
  let distinct_s_vals : finset ℕ := {6, 10, 12} in
  ↑(distinct_s_vals.sum id) / distinct_s_vals.card = 28 / 3 :=
by
  intros h1 h2
  sorry

end distinct_s_average_is_28_over_3_l537_537277


namespace least_possible_value_of_D_l537_537865

-- Defining the conditions as theorems
theorem least_possible_value_of_D :
  ∃ (A B C D : ℕ), 
  (A + B + C + D) / 4 = 18 ∧
  A = 3 * B ∧
  B = C - 2 ∧
  C = 3 / 2 * D ∧
  (∀ x : ℕ, x ≥ 10 → D = x) := 
sorry

end least_possible_value_of_D_l537_537865


namespace definite_integral_sin_cos_l537_537209

open Real

theorem definite_integral_sin_cos :
  ∫ x in - (π / 2)..(π / 2), (sin x + cos x) = 2 :=
sorry

end definite_integral_sin_cos_l537_537209


namespace surface_area_of_sphere_l537_537782

-- Definitions for the given conditions
def face_area1 : ℝ := 2
def face_area2 : ℝ := 3
def face_area3 : ℝ := 6
def all_vertices_on_sphere : Prop := True

-- Main theorem statement with the specified conditions
theorem surface_area_of_sphere (h1 : face_area1 = 2) (h2 : face_area2 = 3) (h3 : face_area3 = 6) (h4 : all_vertices_on_sphere) :
    4 * π * ((sqrt 14 / 2) ^ 2) = 14 * π :=
by
  sorry

end surface_area_of_sphere_l537_537782


namespace stack_height_l537_537459

theorem stack_height (d : ℝ) (h_eq : d = 12) : 
  let h := 12 + 6 * real.sqrt 3 in 
  h = 12 + 6 * real.sqrt 3 :=
by
  sorry

end stack_height_l537_537459


namespace equilateral_triangle_point_condition_l537_537589

theorem equilateral_triangle_point_condition 
  {A B C D P Q P2 P3 : ℝ} : 
  ∀ {ABC : Triangle ℝ} (h_eq : equilateral ABC)
  (h_mid : midpoint D B C)
  (h_perp_1 : perpendicular P2 P CA)
  (h_perp_2 : perpendicular P3 P AB)
  (h_parallel : parallel (line P2 Q) BC),
  distance P3 Q = distance P D :=
by sorry

end equilateral_triangle_point_condition_l537_537589


namespace part1_part2_l537_537356

-- Definitions starting with the basic triangle and conditions given
variables {A B C : Real} (a b c : Real)

-- Conditions provided in the problem
def condition1 (A B C : Real) (a b c : Real) : Prop := 
  (a / b + b / a = 4 * Real.cos C)

def condition2 (A B C : Real): Prop :=
  (1 / Real.tan B = 1 / Real.tan A + 1 / Real.tan C)

-- Theorems derived from the conditions
theorem part1 (h1 : condition1 A B C a b c) : (a^2 + b^2) / c^2 = 2 :=
  sorry

theorem part2 (h2 : condition2 A B C) (a b c : Real):
  cos A = (Real.sqrt 3) / 6 :=
  sorry

end part1_part2_l537_537356


namespace least_8_heavy_three_digit_l537_537909

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 6

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_8_heavy_three_digit : ∃ n : ℕ, is_three_digit n ∧ is_8_heavy n ∧ ∀ m : ℕ, is_three_digit m ∧ is_8_heavy m → n ≤ m := 
sorry

end least_8_heavy_three_digit_l537_537909


namespace eighth_odd_multiple_of_5_is_75_l537_537476

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0) ∧ (n % 2 = 1) ∧ (n % 5 = 0) ∧ (nat.find_greatest (λ m, m % 2 = 1 ∧ m % 5 = 0) 75 = 75) :=
    sorry

end eighth_odd_multiple_of_5_is_75_l537_537476


namespace find_modulus_of_alpha_l537_537015

open Complex

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry

theorem find_modulus_of_alpha 
  (h0 : beta = conj α)
  (h1 : (alpha^2 / beta).im = 0)
  (h2 : abs (alpha - beta) = 4) :
  abs α = 4 * Real.sqrt 3 / 3 :=
sorry

end find_modulus_of_alpha_l537_537015


namespace find_a_l537_537729

variables (a b c d k m : ℤ)

-- Conditions
def odd_numbers (x : ℤ) : Prop := x % 2 = 1

theorem find_a (h1: odd_numbers a) (h2: odd_numbers b) (h3: odd_numbers c) (h4: odd_numbers d) 
  (h5: 0 < a ∧ a < b ∧ b < c ∧ c < d) (h6: a * d = b * c) 
  (h7: a + d = 2^k) (h8: b + c = 2^m) : a = 1 :=
sorry

end find_a_l537_537729


namespace cost_of_paint_per_kilogram_l537_537788

theorem cost_of_paint_per_kilogram:
  (∀ (coverage_per_kg : ℕ) (cost_to_paint_cube : ℕ) (side_length : ℕ),
    (coverage_per_kg = 20) →
    (cost_to_paint_cube = 1800) →
    (side_length = 10) →
    (cost_to_paint_cube * 1 / ((6 * side_length ^ 2) / coverage_per_kg) = 60)) :=
by
  intros coverage_per_kg cost_to_paint_cube side_length h_coverage h_cost h_side
  rw [h_coverage, h_cost, h_side]
  have surface_area : ℕ := 6 * side_length ^ 2
  have kilograms_paint_needed : ℕ := surface_area / coverage_per_kg
  have cost_per_kg : ℕ := cost_to_paint_cube / kilograms_paint_needed
  show cost_per_kg = 60 from sorry


end cost_of_paint_per_kilogram_l537_537788


namespace max_k_for_simple_graph_inequality_l537_537231

theorem max_k_for_simple_graph_inequality :
  ∃ k, k = 9 / 2 ∧ (∀ (G : SimpleGraph) (n : ℕ), n ≥ 3 → G.numVertices = n → 
                    let x := G.numEdges in
                    let y := G.numTriangles in
                    x ^ 3 ≥ k * y ^ 2) :=
sorry

end max_k_for_simple_graph_inequality_l537_537231


namespace grocer_purchased_108_lbs_l537_537860

-- Define the cost price in terms of dollars per pound.
def cost_price_per_lb : ℝ := 0.50 / 3

-- Define the selling price in terms of dollars per pound.
def selling_price_per_lb : ℝ := 1.00 / 4

-- Define the profit per pound as the difference between selling price and cost price.
def profit_per_lb : ℝ := selling_price_per_lb - cost_price_per_lb

-- Define the total profit made by the grocer.
def total_profit : ℝ := 9.00

-- Define the number of pounds purchased by the grocer.
def pounds_purchased : ℝ := 108

-- Prove that the total profit is equivalent to the profit per pound times the pounds purchased.
theorem grocer_purchased_108_lbs : pounds_purchased = total_profit / profit_per_lb :=
by
  sorry

end grocer_purchased_108_lbs_l537_537860


namespace prove_n_equals_3_l537_537003

noncomputable def f (x : ℤ) (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  ∏ k in finset.range n, (x - a k) - 2

theorem prove_n_equals_3
  {n : ℕ} (h1 : n ≥ 3)
  (a : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (g h : ℤ → ℤ) (h_nonconstant_g : degree g ≥ 1)(h_nonconstant_h : degree h ≥ 1)
  (h_f_product : ∀ x, f x n a = g x * h x) :
  n = 3 :=
sorry

end prove_n_equals_3_l537_537003


namespace maximum_OB_length_exists_l537_537462

-- Define the parameters and conditions
def angle_OAB (α : ℝ) (hα : α ≤ 60) : Prop := α ∈ Icc (0 : ℝ) 60
def AB : ℝ := 1
def angle_AOB : ℝ := 45

-- Statement of the problem
theorem maximum_OB_length_exists {O A B : Point} {α : ℝ} 
  (h1 : AB = 1)
  (h2 : ∠AOB = 45)
  (h3 : angle_OAB α hα)
  (hα : α ≤ 60) :
  max_length_OB = sqrt 6 / 2 :=
sorry 

end maximum_OB_length_exists_l537_537462


namespace cross_section_of_cut_cylinder_is_rectangle_l537_537141

theorem cross_section_of_cut_cylinder_is_rectangle 
  (D H : ℝ) (hcylinder : D = 3 ∧ H = 9 ) :
  (cut_cylinder_is_rectangle :  ∃ rectangle : Set (ℝ × ℝ), 
                                let cross_section_shape := if hcylinder then rectangle else sorry,
                                cross_section_shape = rectangle) :=
by
  sorry

end cross_section_of_cut_cylinder_is_rectangle_l537_537141


namespace solution_l537_537344

-- Definitions
def equation1 (x y z : ℝ) : Prop := 2 * x + y + z = 17
def equation2 (x y z : ℝ) : Prop := x + 2 * y + z = 14
def equation3 (x y z : ℝ) : Prop := x + y + 2 * z = 13

-- Theorem to prove
theorem solution (x y z : ℝ) (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : x = 6 :=
by
  sorry

end solution_l537_537344


namespace ceil_floor_difference_is_3_l537_537218

noncomputable def ceil_floor_difference : ℤ :=
  Int.ceil ((14:ℚ) / 5 * (-31 / 3)) - Int.floor ((14 / 5) * Int.floor ((-31:ℚ) / 3))

theorem ceil_floor_difference_is_3 : ceil_floor_difference = 3 :=
  sorry

end ceil_floor_difference_is_3_l537_537218


namespace has_zero_in_interval_l537_537082

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem has_zero_in_interval :
  ∃ c : ℝ, c ∈ set.Ioo 2 3 ∧ f c = 0 :=
by
  sorry

end has_zero_in_interval_l537_537082


namespace min_value_fraction_range_of_x_l537_537620

theorem min_value_fraction (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem range_of_x (x : ℝ) (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : 1 / a + 4 / b ≥ |2 * x - 1| - | x + 1 |) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end min_value_fraction_range_of_x_l537_537620


namespace find_original_cost_of_chips_l537_537705

def original_cost_chips (discount amount_spent : ℝ) : ℝ :=
  discount + amount_spent

theorem find_original_cost_of_chips :
  original_cost_chips 17 18 = 35 := by
  sorry

end find_original_cost_of_chips_l537_537705


namespace servant_received_amount_l537_537528

def annual_salary := 900
def uniform_price := 100
def fraction_of_year_served := 3 / 4

theorem servant_received_amount :
  annual_salary * fraction_of_year_served + uniform_price = 775 := by
  sorry

end servant_received_amount_l537_537528


namespace cos_60_eq_one_half_l537_537941

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537941


namespace num_undefined_values_l537_537585

theorem num_undefined_values :
  ∃! x : Finset ℝ, (∀ y ∈ x, (y + 5 = 0) ∨ (y - 1 = 0) ∨ (y - 4 = 0)) ∧ (x.card = 3) := sorry

end num_undefined_values_l537_537585


namespace graph_passes_through_point_l537_537586

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, (y = a^(x-2) + 2) → (x = 2) → (y = 3) :=
by
    intros x y hxy hx
    rw [hx] at hxy
    simp at hxy
    sorry

end graph_passes_through_point_l537_537586


namespace abs_val_inequality_solution_l537_537078

theorem abs_val_inequality_solution (x : ℝ) : |x - 2| + |x + 3| ≥ 4 ↔ x ≤ - (5 / 2) :=
by
  sorry

end abs_val_inequality_solution_l537_537078


namespace number_of_members_l537_537113

theorem number_of_members (n : ℕ) (H : n * n = 5776) : n = 76 :=
by
  sorry

end number_of_members_l537_537113


namespace problem1_problem2_l537_537657

-- Definitions of the sets
def U : Set ℕ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | 3 ≤ x ∧ x ≤ 7 }

-- Problems to prove (statements only, no proofs provided)
theorem problem1 : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
by
  sorry

theorem problem2 : U \ A ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)} :=
by
  sorry

end problem1_problem2_l537_537657


namespace ratio_of_triangle_BFD_to_square_ABCE_l537_537696

def is_square (ABCE : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a b c e : ℝ, ABCE a b c e → a = b ∧ b = c ∧ c = e

def ratio_of_areas (AF FE CD DE : ℝ) (ratio : ℝ) : Prop :=
  AF = 3 * FE ∧ CD = 3 * DE ∧ ratio = 1 / 2

theorem ratio_of_triangle_BFD_to_square_ABCE (AF FE CD DE ratio : ℝ) (ABCE : ℝ → ℝ → ℝ → ℝ → Prop)
  (h1 : is_square ABCE)
  (h2 : AF = 3 * FE) (h3 : CD = 3 * DE) : ratio_of_areas AF FE CD DE (1 / 2) :=
by
  sorry

end ratio_of_triangle_BFD_to_square_ABCE_l537_537696


namespace least_three_digit_7_heavy_l537_537166

def is_7_heavy (n : ℕ) : Prop :=
  n % 7 ≥ 5

theorem least_three_digit_7_heavy : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ is_7_heavy n ∧ ∀ m, 100 ≤ m < n → ¬ is_7_heavy m :=
  by
  sorry

end least_three_digit_7_heavy_l537_537166


namespace find_integer_pairs_l537_537353

theorem find_integer_pairs (m n : ℤ) (h1 : m * n ≥ 0) (h2 : m^3 + n^3 + 99 * m * n = 33^3) :
  (m = -33 ∧ n = -33) ∨ ∃ k : ℕ, k ≤ 33 ∧ m = k ∧ n = 33 - k ∨ m = 33 - k ∧ n = k :=
by
  sorry

end find_integer_pairs_l537_537353


namespace mystery_number_addition_l537_537680

theorem mystery_number_addition (mystery_number : ℕ) (h : mystery_number = 47) : mystery_number + 45 = 92 :=
by
  -- Proof goes here
  sorry

end mystery_number_addition_l537_537680


namespace cos_60_eq_sqrt3_div_2_l537_537931

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537931


namespace greatest_ratio_integer_coords_circle_l537_537210

theorem greatest_ratio_integer_coords_circle
  (A B C D : ℤ × ℤ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 16)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 16)
  (hC : C.1 ^ 2 + C.2 ^ 2 = 16)
  (hD : D.1 ^ 2 + D.2 ^ 2 = 16)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_irrational_AB : ¬ ∃ r : ℚ, r^2 = ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2))
  (h_irrational_CD : ¬ ∃ r : ℚ, r^2 = ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2)) :
  ∃ A B C D, 
    (A.1 ^ 2 + A.2 ^ 2 = 16) ∧
    (B.1 ^ 2 + B.2 ^ 2 = 16) ∧
    (C.1 ^ 2 + C.2 ^ 2 = 16) ∧
    (D.1 ^ 2 + D.2 ^ 2 = 16) ∧
    (¬ ∃ r : ℚ, r^2 = ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧
    (¬ ∃ r : ℚ, r^2 = ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2)) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) ∧
    (real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) / real.sqrt ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2) = 1) :=
sorry

end greatest_ratio_integer_coords_circle_l537_537210


namespace number_of_functions_l537_537792

-- Define the set of conditions
variables (x y : ℝ)

def relation1 := x - y = 0
def relation2 := y^2 = x
def relation3 := |y| = 2 * x
def relation4 := y^2 = x^2
def relation5 := y = 3 - x
def relation6 := y = 2 * x^2 - 1
def relation7 := y = 3 / x

-- Prove that there are 4 unambiguous functions of y with respect to x
theorem number_of_functions : 4 = 4 := sorry

end number_of_functions_l537_537792


namespace digit_equality_solution_exists_l537_537221

theorem digit_equality_solution_exists :
  ∃ (x y z : ℕ), (∃ (n1 n2 : ℕ), n1 ≠ n2 ∧
    (sqrt (x * (10^(2 * n1) - 1) / 9 - y * (10^n1 - 1) / 9) = z * (10^n1 - 1) / 9) ∧
    (sqrt (x * (10^(2 * n2) - 1) / 9 - y * (10^n2 - 1) / 9) = z * (10^n2 - 1) / 9)) ∧
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 9 ∧ y = 8 ∧ z = 9)) := by
  sorry

end digit_equality_solution_exists_l537_537221


namespace smallest_value_is_2_5_l537_537725

noncomputable def smallest_value_z_plus_i (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) : ℝ :=
  Inf { w : ℝ | ∃ z : ℂ, |z^2 + 9| = |z * (z + 3 * complex.I) ∧ w = complex.abs (z + complex.I) }

theorem smallest_value_is_2_5 (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) :
  smallest_value_z_plus_i z h = 2.5 :=
sorry

end smallest_value_is_2_5_l537_537725


namespace g_at_2_l537_537720

-- Definition of the function f and its inverse
def f (x : ℝ) : ℝ := 4 / (3 - x)
def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

-- Definition of the function g
def g (x : ℝ) : ℝ := 1 / f_inv x + 7

-- Statement of the theorem to prove
theorem g_at_2 : g 2 = 8 :=
by
  -- Proof would go here
  sorry

end g_at_2_l537_537720


namespace solution_l537_537386

def S : Set ℝ := { x : ℝ | x > -1 }

def f (x : ℝ) := - x / (1 + x)

theorem solution (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) :
  (f (x + f y + x * f y) = y + f x + y * f x) ∧ 
  (StrictMonoOn (λ x, f x / x) { x | x ∈ S ∧ x < 0 } ∧ StrictMonoOn (λ x, f x / x) { x | x > 0 }) :=
by
  sorry

end solution_l537_537386


namespace B_pow_six_l537_537016

variable {B : Matrix (Fin 2) (Fin 2) ℝ}
variable {v : Vector ℝ 2}

theorem B_pow_six {B : Matrix (Fin 2) (Fin 2) ℝ} {v : Vector ℝ 2} (h : B ⬝ v = 3 • v) :
  B^6 ⬝ v = 2187 • v :=
sorry

end B_pow_six_l537_537016


namespace determine_n_l537_537967

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem determine_n (a b c : ℕ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
    let n := lcm a b + lcm b c + lcm c a
    M (xy yz xz : ℕ) := 
    (∃ d p q r x y z, 
      a = d * p * q * x ∧ 
      b = d * p * r * y ∧ 
      c = d * q * r * z ∧ 
      M = d * p * q * r ∧ 
      xy = x * y ∧ 
      yz = y * z ∧ 
      xz = x * z ∧ 
      (xy + yz + xz) % 2 = 1) 
    → n = M * (xy + yz + xz) := 
sorry

end determine_n_l537_537967


namespace crossing_time_approx_10_8_l537_537503

def length_train1 : ℝ := 140  -- Length of the first train in meters
def length_train2 : ℝ := 160  -- Length of the second train in meters
def speed_train1 : ℝ := 60    -- Speed of the first train in km/hr
def speed_train2 : ℝ := 40    -- Speed of the second train in km/hr

def km_per_hr_to_m_per_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * (1000 / 3600)

def relative_speed_m_per_s : ℝ :=
  km_per_hr_to_m_per_s (speed_train1 + speed_train2) 

def total_length : ℝ := length_train1 + length_train2

def crossing_time : ℝ :=
  total_length / relative_speed_m_per_s

theorem crossing_time_approx_10_8 :
  abs (crossing_time - 10.8) < 0.1 :=
  sorry

end crossing_time_approx_10_8_l537_537503


namespace evaluate_expression_l537_537847

theorem evaluate_expression : 
  let a := 4 
  let b := -3 
  -a - b^2 + a*b = -25 := 
by 
  intro a b 
  have h1 : a = 4 := rfl 
  have h2 : b = -3 := rfl 
  rw [h1, h2] 
  -- simplifying the expression -a - b^2 + ab
  simp only [one_mul, neg_mul_eq_neg_mul_symm, pow_two, add_left_neg, zero_add, add_neg_eq_sub, neg_neg]
  sorry

end evaluate_expression_l537_537847


namespace total_mail_delivered_l537_537527

theorem total_mail_delivered (junk_mail magazines : ℕ) (hjunk : junk_mail = 6) (hmag : magazines = 5) :
  junk_mail + magazines = 11 :=
by {
  rw [hjunk, hmag],
  norm_num,
  done
  sorry
}

end total_mail_delivered_l537_537527


namespace avg_distinct_s_l537_537275

theorem avg_distinct_s (s : ℤ) (hpq : ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p + q = 7 ∧ p * q = s) :
  (∑ (p q : ℤ) in {1, 2, 3, 4, 5, 6}, if p + q = 7 then p * q else 0).erase 0 = {6, 10, 12} →
  s = ∑ x in {6, 10, 12}, x / {6, 10, 12}.size := sorry

end avg_distinct_s_l537_537275


namespace regular_polygon_sides_l537_537990

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537990


namespace find_y_l537_537869

open Nat

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h₁ : x % y = 12) (h₂ : x / y + (x % y) / y = 75.12) : y = 100 :=
by
  sorry

end find_y_l537_537869


namespace zeros_in_expansion_l537_537661

def x : ℕ := 10^12 - 3
def num_zeros (n : ℕ) : ℕ := (n.toString.filter (· == '0')).length

theorem zeros_in_expansion :
  num_zeros (x^2) = 20 := sorry

end zeros_in_expansion_l537_537661


namespace main_theorem_l537_537026

theorem main_theorem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end main_theorem_l537_537026


namespace problem_1_problem_2_l537_537636

noncomputable def f (x : ℝ) : ℝ :=
  (Real.logb 3 (x / 27)) * (Real.logb 3 (3 * x))

theorem problem_1 (h₁ : 1 / 27 ≤ x)
(h₂ : x ≤ 1 / 9) :
    (∀ x, f x ≤ 12) ∧ (∃ x, f x = 5) := 
sorry

theorem problem_2
(m α β : ℝ)
(h₁ : f α + m = 0)
(h₂ : f β + m = 0) :
    α * β = 9 :=
sorry

end problem_1_problem_2_l537_537636


namespace algebraic_notation_correct_l537_537493

def exprA : String := "a * 5"
def exprB : String := "a7"
def exprC : String := "3 1/2 x"
def exprD : String := "-7/8 x"

theorem algebraic_notation_correct :
  exprA ≠ "correct" ∧
  exprB ≠ "correct" ∧
  exprC ≠ "correct" ∧
  exprD = "correct" :=
by
  sorry

end algebraic_notation_correct_l537_537493


namespace minimum_value_of_E_l537_537488

theorem minimum_value_of_E (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 12) : |E| = 11 :=
sorry

end minimum_value_of_E_l537_537488


namespace find_extrema_l537_537240

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) - 1

theorem find_extrema :
  (∀ x : ℝ, (1/2)^x ≤ 4 ∧ log (real.sqrt 3) x ≤ 2 →
   -13 / 4 ≤ f x ∧ f x ≤ 647) :=
begin
  sorry
end

end find_extrema_l537_537240


namespace factor_theorem_l537_537116

theorem factor_theorem (h : ℤ) : (∀ m : ℤ, (m - 8) ∣ (m^2 - h * m - 24) ↔ h = 5) :=
  sorry

end factor_theorem_l537_537116


namespace range_of_eccentricity_l537_537242

noncomputable def ellipse_eccentricity_range
  (a b : ℝ) (h : a > b ∧ b > 0)
  (f : ℝ → ℝ → ℝ → ℝ) -- Function representing foci locations
  (p : ℝ → ℝ → Prop) -- Predicate that P(x, y) satisfies the given condition
  : Set ℝ :=
  { e | e = real.sqrt (2 - real.sqrt 2) } ∪ (set.Ioo (real.sqrt 6 / 3) 1)

theorem range_of_eccentricity
  (a b : ℝ) (h : a > b ∧ b > 0)
  (f : ℝ → ℝ → ℝ → ℝ)
  (p : Π x y : ℝ, Prop)  -- Predicate that P(x, y) satisfies the given condition
  : ellipse_eccentricity_range a b h f p =
    { e | e = real.sqrt (2 - real.sqrt 2) } ∪ (set.Ioo (real.sqrt 6 / 3) 1) :=
by
  sorry

end range_of_eccentricity_l537_537242


namespace sum_of_floors_l537_537959

/-- Defining the sequence and stating the target sum of the floors -/
def arith_sequence : ℕ → ℝ 
| 0 := 0.5
| (n + 1) := arith_sequence n + 0.6

def floor_sum (n : ℕ) : ℝ :=
  finset.sum (finset.range (n+1)) (λ i, ⌊arith_sequence i⌋)

theorem sum_of_floors : 
  (n : ℕ), (1 + (99.9 - 0.5) / 0.6).nat_ceil = n → floor_sum n = 8146 :=
by
  sorry

end sum_of_floors_l537_537959


namespace profit_calculation_l537_537708

theorem profit_calculation (investment_john investment_mike profit_john profit_mike: ℕ) 
  (total_profit profit_shared_ratio profit_remaining_profit: ℚ)
  (h_investment_john : investment_john = 700)
  (h_investment_mike : investment_mike = 300)
  (h_total_profit : total_profit = 3000)
  (h_shared_ratio : profit_shared_ratio = total_profit / 3 / 2)
  (h_remaining_profit : profit_remaining_profit = 2 * total_profit / 3)
  (h_profit_john : profit_john = profit_shared_ratio + (7 / 10) * profit_remaining_profit)
  (h_profit_mike : profit_mike = profit_shared_ratio + (3 / 10) * profit_remaining_profit)
  (h_profit_difference : profit_john = profit_mike + 800) :
  total_profit = 3000 := 
by
  sorry

end profit_calculation_l537_537708


namespace find_m_for_monotonically_decreasing_l537_537651

theorem find_m_for_monotonically_decreasing:
  ∃ m, (∀ x : ℝ, x > 0 → deriv (λ x, (m^2 - 2 * m - 2) * x^m) x < 0) ↔ m = -1 :=
sorry

end find_m_for_monotonically_decreasing_l537_537651


namespace smallest_base10_integer_l537_537101

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end smallest_base10_integer_l537_537101


namespace minimum_value_of_f_l537_537669

noncomputable def f (x : ℝ) : ℝ := 4 * x + 9 / x

theorem minimum_value_of_f : 
  (∀ (x : ℝ), x > 0 → f x ≥ 12) ∧ (∃ (x : ℝ), x > 0 ∧ f x = 12) :=
by {
  sorry
}

end minimum_value_of_f_l537_537669


namespace parabola_ratio_l537_537650

theorem parabola_ratio (F M N : Point) (parabola : ∀ (P : Point), P.y^2 = 4 * P.x)
    (line : ∀ (P : Point), P ∈ line ↔ ∃ k : ℝ, P.y = k * (P.x - F.x))
    (F : Point := { x := 1, y := 0 })
    (M : Point := { x := 2, y := 2 * Real.sqrt 2 })
    (N : Point := { x := 1 / 2, y := -Real.sqrt 2 })
    (NF : ℝ := Real.sqrt ((N.x - F.x)^2 + (N.y - F.y)^2))
    (FM : ℝ := Real.sqrt ((M.x - F.x)^2 + (M.y - F.y)^2))
: NF / FM = 1 / 2 := by
  sorry

end parabola_ratio_l537_537650


namespace workman_problem_l537_537145

theorem workman_problem (x : ℝ) (h : (1 / x) + (1 / (2 * x)) = 1 / 32): x = 48 :=
sorry

end workman_problem_l537_537145


namespace eighth_odd_multiple_of_5_is_75_l537_537485

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l537_537485


namespace fit_nine_cross_pentominoes_on_chessboard_l537_537428

def cross_pentomino (A B C D E : Prop) :=
  A ∧ B ∧ C ∧ D ∧ E -- A cross pentomino is five connected 1x1 squares

def square1x1 : Prop := sorry -- a placeholder for a 1x1 square

def eight_by_eight_chessboard := Fin 8 × Fin 8 -- an 8x8 chessboard using finitely indexed squares

noncomputable def can_cut_nine_cross_pentominoes : Prop := sorry -- a placeholder proof verification

theorem fit_nine_cross_pentominoes_on_chessboard : can_cut_nine_cross_pentominoes  :=
by 
  -- Assume each cross pentomino consists of 5 connected 1x1 squares
  let cross := cross_pentomino square1x1 square1x1 square1x1 square1x1 square1x1
  -- We need to prove that we can cut out nine such crosses from the 8x8 chessboard
  sorry

end fit_nine_cross_pentominoes_on_chessboard_l537_537428


namespace sqrt_14_lt_4_l537_537194

theorem sqrt_14_lt_4 : real.sqrt 14 < 4 := 
by
  -- We know that 4 = sqrt 16
  have h1 : 4 = real.sqrt 16 := by sorry
  -- Given that 14 < 16
  have h2 : 14 < 16 := by sorry
  -- With 14 < 16, we know sqrt 14 < sqrt 16
  have h3 : real.sqrt 14 < real.sqrt 16 := by 
    apply real.sqrt_lt_sqrt_of_lt h2
  -- Finally, we combine h1 and h3 to get sqrt 14 < 4
  exact lt_of_lt_of_eq h3 h1.symm

end sqrt_14_lt_4_l537_537194


namespace programs_same_result_l537_537761

def program_A_sum : ℕ :=
  let rec loop (i S : ℕ) :=
    if i > 1000 then S
    else loop (i + 1) (S + i)
  loop 1 0

def program_B_sum : ℕ :=
  let rec loop (i S : ℕ) :=
    if i < 1 then S
    else loop (i - 1) (S + i)
  loop 1000 0

theorem programs_same_result : program_A_sum = program_B_sum :=
by sorry

end programs_same_result_l537_537761


namespace similar_triangles_of_cyclic_quadrilateral_l537_537602

-- Definitions (conditions) based on the given problem:
def cyclic_quadrilateral (A B C D : Type) [Euclidean_plane A] [Euclidean_plane B]
  [Euclidean_plane C] [Euclidean_plane D] : Prop :=
cyclic A B C D

def intersection (A B C D X : Type) [Euclidean_plane A] [Euclidean_plane B]
  [Euclidean_plane C] [Euclidean_plane D] [Euclidean_plane X] : Prop :=
is_intersection A B C D X

-- Problem statement in Lean 4:
theorem similar_triangles_of_cyclic_quadrilateral
  {A B C D X : Type} [Euclidean_plane A] [Euclidean_plane B]
  [Euclidean_plane C] [Euclidean_plane D] [Euclidean_plane X]
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : intersection A B C D X)
  : similar (triangle X B C) (triangle X D A) :=
sorry

end similar_triangles_of_cyclic_quadrilateral_l537_537602


namespace sum_binom_eq_binom_l537_537390

open BigOperators

theorem sum_binom_eq_binom (k n : ℕ) (hk : 0 < k) (hn : 0 < n) :
  ∑ (a : Fin k → Fin (n+1)) in {a | (∀ i j, i ≤ j → a i ≤ a j) ∧ (∑ i, a i.val = k)}, 
    (∏ i in Finset.range k, Nat.choose (if i = 0 then n else a ⟨i - 1, _⟩.val) (a ⟨i, _⟩.val)) = 
  Nat.choose (k + n - 1) k :=
by
  sorry

end sum_binom_eq_binom_l537_537390


namespace identify_incorrect_proposition_l537_537111

-- Definitions based on problem conditions
def propositionA : Prop :=
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))

def propositionB : Prop :=
  (¬ (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0)

def propositionD (x : ℝ) : Prop :=
  (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬(x > 2) → ¬(x^2 - 3*x + 2 > 0))

-- Proposition C is given to be incorrect in the problem
def propositionC (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q

theorem identify_incorrect_proposition (p q : Prop) : 
  (propositionA ∧ propositionB ∧ (∀ x : ℝ, propositionD x)) → 
  ¬ (propositionC p q) :=
by
  intros
  -- We know proposition C is false based on the problem's solution
  sorry

end identify_incorrect_proposition_l537_537111


namespace range_ab_plus_a_plus_b_l537_537638

theorem range_ab_plus_a_plus_b {a b : ℝ} (h1 : a < b) (h2 : b < -1) 
  (h3 : |a^2 + 2 * a - 1| = |b^2 + 2 * b - 1|): 
  ∃ (r : Set ℝ), r = Ioo (-∞) (-1) ∧ (ab : ℝ) (ab ∈ r) :=
by
  sorry

end range_ab_plus_a_plus_b_l537_537638


namespace trader_profit_l537_537165

theorem trader_profit (profit donation goal : ℕ) (h1 : profit = 960) (h2 : donation = 310) (h3 : goal = 610) :
  (profit / 2 + donation - goal = 180) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end trader_profit_l537_537165


namespace log_base_8_solution_l537_537666

theorem log_base_8_solution (x : ℝ) (h : logb 8 x = 3.5) : x = 512 :=
by sorry

end log_base_8_solution_l537_537666


namespace original_average_l537_537062

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average_l537_537062


namespace part1_part2_l537_537291

-- Define the hyperbola and the line in Lean
def hyperbola (x y : ℝ) := x^2 - y^2 = 1
def line (k x y : ℝ) := y = k * x - 1

-- Proof statement for part (1)
theorem part1 (k : ℝ) :
  (∃ x y, hyperbola x y ∧ line k x y) → 
  k ∈ Ioo (-Real.sqrt 2) (-1) :=
sorry

-- Proof statement for part (2)
theorem part2 (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  (hyperbola x1 y1 ∧ line k x1 y1) ∧
  (hyperbola x2 y2 ∧ line k x2 y2) ∧
  x1 ≠ x2 ∧
  (1/2) * abs (x1 - x2) = Real.sqrt 2 →
  k = 0 ∨ k = Real.sqrt 6 / 2 ∨ k = -Real.sqrt 6 / 2 :=
sorry

end part1_part2_l537_537291


namespace sequence_next_term_l537_537048

theorem sequence_next_term (n m : ℕ) :
  (n = 6) → (sequence 1 = 3) → (sequence 2 = 15) → (sequence 3 = 35) → 
  (sequence 4 = 63) → (sequence 5 = 99) → sequence n = 143 :=
by
  sorry

end sequence_next_term_l537_537048


namespace closest_integer_sum_l537_537225

theorem closest_integer_sum : 
  let s := 500 * (∑ n in Finset.range (5000 - 3) \ Finset.range 1, 1 / (n + 4)^2 - 9)
  ∃ (k : ℤ), k = 174 ∧ abs (s - k) = abs (s - 174) := 
sorry

end closest_integer_sum_l537_537225


namespace volume_of_solid_T_l537_537202

noncomputable def volume_of_T : ℝ :=
  let T := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| ≤ 2 ∧ |p.1| + |p.3| ≤ 2 ∧ |p.2| + |p.3| ≤ 2} in
  sorry

theorem volume_of_solid_T : volume_of_T = 16 / 3 :=
sorry

end volume_of_solid_T_l537_537202


namespace gcd_lcm_ordering_l537_537388

theorem gcd_lcm_ordering (a b p q : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_gt_b : a > b) 
    (h_p_gcd : p = Nat.gcd a b) (h_q_lcm : q = Nat.lcm a b) : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end gcd_lcm_ordering_l537_537388


namespace crease_length_right_triangle_l537_537352

theorem crease_length_right_triangle (BC AC AB : ℝ) (h1 : BC = 3) (h2 : AC = 4) (h3 : AB = 5) : 
  let DE := 3 * (5 / 8) in DE = 15 / 8 := 
by
  sorry

end crease_length_right_triangle_l537_537352


namespace find_k_l537_537697

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℕ) := ∀ n m, a (n + m) - a n = (a (n + 1) - a n) * m

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℕ} {S : ℕ → ℕ}

-- Conditions from the problem
axiom arith_seq : arithmetic_sequence a
axiom sum_def : ∀ n, S n = sum_of_first_n_terms a n
axiom S3_eq_S8 : S 3 = S 8
axiom S7_eq_Sk : ∃ k, S 7 = S k

-- Theorem to prove the correct value of k
theorem find_k (k : ℕ) (S3_eq_S8 S7_eq_Sk) : k = 4 :=
sorry

end find_k_l537_537697


namespace only_rational_root_l537_537968

noncomputable def P : ℚ[X] := 3 * X^4 - 4 * X^3 - 10 * X^2 + 6 * X + 3

theorem only_rational_root :
  ∀ x : ℚ, P.eval x = 0 → x = 1/3 :=
by
  sorry

end only_rational_root_l537_537968


namespace polynomial_factor_l537_537222

theorem polynomial_factor (x : ℝ) : (x^2 - 4*x + 4) ∣ (x^4 + 16) :=
sorry

end polynomial_factor_l537_537222


namespace percent_flowers_are_carnations_l537_537142

-- Define the conditions
def one_third_pink_are_roses (total_flower pink_flower pink_roses : ℕ) : Prop :=
  pink_roses = (1/3) * pink_flower

def three_fourths_red_are_carnations (total_flower red_flower red_carnations : ℕ) : Prop :=
  red_carnations = (3/4) * red_flower

def six_tenths_are_pink (total_flower pink_flower : ℕ) : Prop :=
  pink_flower = (6/10) * total_flower

-- Define the proof problem statement
theorem percent_flowers_are_carnations (total_flower pink_flower pink_roses red_flower red_carnations : ℕ) :
  one_third_pink_are_roses total_flower pink_flower pink_roses →
  three_fourths_red_are_carnations total_flower red_flower red_carnations →
  six_tenths_are_pink total_flower pink_flower →
  (red_flower = total_flower - pink_flower) →
  (pink_flower - pink_roses + red_carnations = (4/10) * total_flower) →
  ((pink_flower - pink_roses) + red_carnations) * 100 / total_flower = 40 := 
sorry

end percent_flowers_are_carnations_l537_537142


namespace no_valid_f2_l537_537376

open Function

-- Define the functional equation given in the problem
axiom functional_eq :
  ∀ (f : ℝ → ℝ) (x y : ℝ), 2 * f (x^2 - y^2) = (x - y) * (f x + f y)

-- Define the specific function value given
axiom f_value_one : ∀ (f : ℝ → ℝ), f 1 = 2

-- Prove that there are no valid values for f(2) given the functional equation condition.
theorem no_valid_f2 : ∀ (f : ℝ → ℝ),
  (∀ (x y : ℝ), 2 * f (x^2 - y^2) = (x - y) * (f x + f y)) →
  f 1 = 2 →
  ∃ (n s : ℕ), n = 0 ∧ s = 0 ∧ n * s = 0 :=
by
  intros f functional_eq f_value_one
  existsi 0
  existsi 0
  simp
  sorry

end no_valid_f2_l537_537376


namespace minimum_students_lost_all_items_l537_537575

def smallest_number (N A B C : ℕ) (x : ℕ) : Prop :=
  N = 30 ∧ A = 26 ∧ B = 23 ∧ C = 21 → x ≥ 10

theorem minimum_students_lost_all_items (N A B C : ℕ) : 
  smallest_number N A B C 10 := 
by {
  sorry
}

end minimum_students_lost_all_items_l537_537575


namespace solution_set_of_inequality_l537_537632

variables {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def increasing_on (f : R → R) (S : Set R) : Prop :=
∀ ⦃x y⦄, x ∈ S → y ∈ S → x < y → f x < f y

theorem solution_set_of_inequality
  {f : R → R}
  (h_odd : odd_function f)
  (h_neg_one : f (-1) = 0)
  (h_increasing : increasing_on f {x : R | x > 0}) :
  {x : R | x * f x > 0} = {x : R | x < -1} ∪ {x : R | x > 1} :=
sorry

end solution_set_of_inequality_l537_537632


namespace probability_of_sum_five_l537_537845

def is_fair_six_sided_die (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 6

theorem probability_of_sum_five (d1 d2 : ℕ) (h1 : is_fair_six_sided_die d1) (h2 : is_fair_six_sided_die d2) :
  (d1 + d2 = 5) → (∑ x in finset.range 6, ∑ y in finset.range 6, if (x + 1) + (y + 1) = 5 then 1 else 0) / 36 = 1 / 9 := 
sorry

end probability_of_sum_five_l537_537845


namespace cos_of_60_degrees_is_half_l537_537952

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537952


namespace remaining_days_to_finish_l537_537001

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l537_537001


namespace sample_size_survey_l537_537830

def sample_size {students : Type*} (survey_sample : ℕ) : ℕ :=
survey_sample

theorem sample_size_survey (survey_sample : ℕ) (h : survey_sample = 1500) : sample_size survey_sample = 1500 :=
by
  rw h
  rfl

end sample_size_survey_l537_537830


namespace find_principal_l537_537850

-- Definitions of given conditions and variables
variables (P r : ℝ)
def A : ℕ → ℝ := λ n, P * (1 + r)^n

-- Given conditions
axiom condition1 : A 2 = 8840
axiom condition2 : A 3 = 9261

-- The goal to prove
theorem find_principal : P = 8056.57 :=
by
  sorry

end find_principal_l537_537850


namespace union_symmetry_containment_l537_537049

variable {M : Set Point} {L : Set Line}
variable (l0 l1 : Line)
variable (S : Line → Point → Point)

-- conditions
axiom reflection_involutive : ∀ (l : Line) (A : Point), S l (S l A) = A
axiom reflection_commute : ∀ (l m : Line) (A : Point), S l (S m A) = S m (S l A)

-- The statement to be proved
theorem union_symmetry_containment (h0 : ∀ A, S l0 A ∈ M) 
                                   (h1 : ∀ A, S l1 A ∈ M) 
                                   (h : ∀ A, A ∈ M → S l0 (S l1 A) ∈ M) : 
  (⋃ l ∈ {l0, l1}, {A | S l A ∈ M}) ⊆ ⋃ l ∈ L, {A | S l A ∈ L} := 
sorry

end union_symmetry_containment_l537_537049


namespace insulation_cost_calculation_l537_537885

noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ :=
  2 * Real.pi * r * (h + r)

noncomputable def insulationCost (diameter height cost_per_sqft : ℝ) : ℝ :=
  let r := diameter / 2
  let surface_area := cylinderSurfaceArea r height
  surface_area * cost_per_sqft

theorem insulation_cost_calculation : 
  insulationCost 3.7 5.2 20 ≈ 1639.28 := by
  sorry

end insulation_cost_calculation_l537_537885


namespace projection_inequality_l537_537371

variables {A B C D O P Q R S : Point}
variables {AB BC CD DA : ℝ} -- lengths of the sides
variables {OP OQ OR OS : ℝ} -- projections of O on the sides

-- Conditions
axiom convex_quadrilateral (h : convex_quadrilateral A B C D) : true
axiom diagonals_intersect (h : ∃ O, is_intersection O (diagonal A C) (diagonal B D)) : true
axiom projections 
  (h₁ : P = projection O A B) 
  (h₂ : Q = projection O B C) 
  (h₃ : R = projection O C D) 
  (h₄ : S = projection O D A) : true

-- Inequality to prove
theorem projection_inequality : 
  2 * (OP + OQ + OR + OS) ≤ AB + BC + CD + DA :=
sorry

end projection_inequality_l537_537371


namespace cannot_transform_to_target_l537_537447

namespace PolynomialTransform

open Polynomial

-- Definitions based on conditions in a)
def initial_poly : Polynomial ℚ := X^17 + 2*X^15 + 4*X^9 + X^6 + 4*X^3 + 2*X + 1
def target_poly : Polynomial ℚ := 3*X + 1

def transform1 (P : Polynomial ℚ) : Polynomial ℚ :=
  X * P.derivative

def transform2 (P : Polynomial ℚ) (k : ℕ) : Polynomial ℚ :=
  P.mapCoeff (λ c => if c.head.1 = k then (c.head.2 / k) else c.head.2)

def transform3 (P : Polynomial ℚ) (c : ℚ) : Polynomial ℚ :=
  P + c

def transform4 (P : Polynomial ℚ) : Polynomial ℚ :=
  P.erase P.natDegree

-- The proof statement
theorem cannot_transform_to_target : ¬(∃ (f : Polynomial ℚ → Polynomial ℚ), f initial_poly = target_poly) :=
sorry

end PolynomialTransform

end cannot_transform_to_target_l537_537447


namespace savings_of_person_l537_537795

theorem savings_of_person (income expenditure : ℕ) (h_ratio : 3 * expenditure = 2 * income) (h_income : income = 21000) :
  income - expenditure = 7000 :=
by
  sorry

end savings_of_person_l537_537795


namespace probability_one_new_ball_l537_537889

noncomputable def balls_prob : ℚ :=
  let new_balls : ℕ := 4 in
  let old_balls : ℕ := 4 in
  let total_balls : ℕ := new_balls + old_balls in
  let C := nat.choose in
  let P := λ n r => (C n r) / (C total_balls 2) in
  let P_A0 := (C old_balls 2) / (C total_balls 2) in
  let P_A1 := (C new_balls 1 * C old_balls 1) / (C total_balls 2) in
  let P_A2 := (C new_balls 2) / (C total_balls 2) in
  let P_B_given_A0 := (C new_balls 1 * C old_balls 1) / (C total_balls 2) in
  let P_B_given_A1 := (C (new_balls - 1) 1 * C (old_balls + 1) 1) / (C total_balls 2) in
  let P_B_given_A2 := (C (new_balls - 2) 1 * C (old_balls + 2) 1) / (C total_balls 2) in
  P_A0 * P_B_given_A0 + P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2

theorem probability_one_new_ball :
  balls_prob = 51 / 98 :=
by
  sorry

end probability_one_new_ball_l537_537889


namespace minInterestingSum_existInterestingConfig_l537_537841

def isInteresting (floors : List ℕ) (idx : ℕ) : Prop :=
  let n := floors.length
  n > 2 ∧ ∃i j, 0 ≤ i < n ∧ 0 ≤ j < n ∧ i ≠ j ∧ 
  (idx = (i + 1) % n ∧ idx = (j - 1) % n) ∧ 
  floors[(i + 1) % n] < floors[idx] ∧ floors[(j - 1) % n] > floors[idx]

def buildings : List (List ℕ) := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

theorem minInterestingSum :
  (∀ (floors : List ℕ), (floors.permutations.contains buildings) →
    (∃ (interesting_buildings: List ℕ),
      interesting_buildings.length = 6 ∧ (∀ x : ℕ, x ∈ interesting_buildings → isInteresting floors x) ∧ 
      (interesting_buildings.sum ≥ 27))) sorry

theorem existInterestingConfig :
  (∃ (config: List ℕ),
     (config.permutations.contains buildings) ∧ 
     (∃ (interesting_buildings: List ℕ),
      interesting_buildings.length = 6 ∧ (∀ x : ℕ, x ∈ interesting_buildings → isInteresting config x) ∧ 
      interesting_buildings.sum = 27 )) sorry

end minInterestingSum_existInterestingConfig_l537_537841


namespace eighth_odd_multiple_of_5_l537_537473

theorem eighth_odd_multiple_of_5 : 
  (∃ n : ℕ, n = 8 ∧ ∃ k : ℤ, k = (10 * n - 5) ∧ k > 0 ∧ k % 2 = 1) → 75 := 
by {
  sorry
}

end eighth_odd_multiple_of_5_l537_537473


namespace sequence_7th_term_l537_537793

theorem sequence_7th_term :
  let left_term (n : ℕ) := 1 + (n - 1) * 3,
      right_term (n : ℕ) := ((2 * n - 1) : ℕ) ^ 2 in
  left_term 7 = 19 ∧ right_term 7 = 169 :=
by
  sorry

end sequence_7th_term_l537_537793


namespace music_students_count_l537_537134

open Nat

theorem music_students_count (total_students : ℕ) (art_students : ℕ) (both_music_art : ℕ) 
      (neither_music_art : ℕ) (M : ℕ) :
    total_students = 500 →
    art_students = 10 →
    both_music_art = 10 →
    neither_music_art = 470 →
    (total_students - neither_music_art) = 30 →
    (M + (art_students - both_music_art)) = 30 →
    M = 30 :=
by
  intros h_total h_art h_both h_neither h_music_art_total h_music_count
  sorry

end music_students_count_l537_537134


namespace area_of_given_sector_l537_537251

noncomputable def area_of_sector (alpha l : ℝ) : ℝ :=
  let r := l / alpha
  (1 / 2) * l * r

theorem area_of_given_sector :
  let alpha := Real.pi / 9
  let l := Real.pi / 3
  area_of_sector alpha l = Real.pi / 2 :=
by
  sorry

end area_of_given_sector_l537_537251


namespace cosine_60_degrees_l537_537946

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537946


namespace length_of_PC_l537_537322

noncomputable def AB : ℝ := 9
noncomputable def BC : ℝ := 8
noncomputable def CA : ℝ := 7
noncomputable def PC : ℝ := 28

theorem length_of_PC (PAB_sim_PCA : similar (triangle P A B) (triangle P C A)) : PC = 28 := by
  sorry

end length_of_PC_l537_537322


namespace yellow_fraction_after_changes_l537_537687

theorem yellow_fraction_after_changes (y : ℕ) :
  let green_initial := (4 / 7 : ℚ) * y
  let yellow_initial := (3 / 7 : ℚ) * y
  let yellow_new := 3 * yellow_initial
  let green_new := green_initial + (1 / 2) * green_initial
  let total_new := green_new + yellow_new
  yellow_new / total_new = (3 / 5 : ℚ) :=
by
  sorry

end yellow_fraction_after_changes_l537_537687


namespace percentage_transactions_anthony_handled_more_l537_537398

theorem percentage_transactions_anthony_handled_more (M A C J : ℕ) (P : ℚ)
  (hM : M = 90)
  (hJ : J = 83)
  (hCJ : J = C + 17)
  (hCA : C = (2 * A) / 3)
  (hP : P = ((A - M): ℚ) / M * 100) :
  P = 10 := by
  sorry

end percentage_transactions_anthony_handled_more_l537_537398


namespace integral_solution_l537_537185

noncomputable def integral_problem := 
  ∫ x in -real.pi / 2..real.pi / 2, (real.sin x + 2) = 2 * real.pi

theorem integral_solution : integral_problem := sorry

end integral_solution_l537_537185


namespace first_student_can_ensure_one_real_root_l537_537401

noncomputable def can_first_student_ensure_one_real_root : Prop :=
  ∀ (b c : ℝ), ∃ a : ℝ, ∃ d : ℝ, ∀ (e : ℝ), 
    (d = 0 ∧ (e = b ∨ e = c)) → 
    (∀ x : ℝ, x^3 + d * x^2 + e * x + (if e = b then c else b) = 0)

theorem first_student_can_ensure_one_real_root :
  can_first_student_ensure_one_real_root := sorry

end first_student_can_ensure_one_real_root_l537_537401


namespace intersection_eq_l537_537736

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_eq : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_eq_l537_537736


namespace cos_60_eq_one_half_l537_537943

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537943


namespace max_trees_l537_537468

theorem max_trees (interval distance road_length number_of_intervals add_one : ℕ) 
  (h_interval: interval = 4) 
  (h_distance: distance = 28) 
  (h_intervals: number_of_intervals = distance / interval)
  (h_add: add_one = number_of_intervals + 1) :
  add_one = 8 :=
sorry

end max_trees_l537_537468


namespace sum_and_count_divisors_of_117_l537_537107

theorem sum_and_count_divisors_of_117 :
  ∑ d in (finset.filter (λ d, 117 % d = 0) (finset.range 118)), d = 182 ∧ 
  (finset.filter nat.prime (finset.filter (λ d, 117 % d = 0) (finset.range 118))).card = 2 :=
by
  sorry

end sum_and_count_divisors_of_117_l537_537107


namespace probability_of_event_A_l537_537868

/-- The events A and B are independent, and it is given that:
  1. P(A) > 0
  2. P(A) = 2 * P(B)
  3. P(A or B) = 8 * P(A and B)

We need to prove that P(A) = 1/3. 
-/
theorem probability_of_event_A (P_A P_B : ℝ) (hP_indep : P_A * P_B = P_A) 
  (hP_A_pos : P_A > 0) (hP_A_eq_2P_B : P_A = 2 * P_B) 
  (hP_or_eq_8P_and : P_A + P_B - P_A * P_B = 8 * P_A * P_B) : 
  P_A = 1 / 3 := 
by
  sorry

end probability_of_event_A_l537_537868


namespace angle_BAC_is_45_degrees_l537_537181

theorem angle_BAC_is_45_degrees
  (ABC : Triangle)
  (I : Point)
  (O : Point)
  (X : Point)
  (M : Point)
  (hI : I = ABC.incenter)
  (hO : O = ABC.circumcenter)
  (hOI : LineThrough O I)
  (hX : X ∈ (line_through O I) ∧ X ∈ ABC.BC)
  (hM : M = midpoint_minor_arc_BC_not_containing_A ABC)
  (hConcyclic : concyclic [A, O, M, X]) :
  ∠BAC = 45 :=
sorry

end angle_BAC_is_45_degrees_l537_537181


namespace problem_1_problem_2_l537_537285

-- Problem 1
theorem problem_1 (n : ℕ) (hn : n > 0) :
  let a_n := λ n : ℕ, -2 + 3 * (n - 1) 
  let c_n := λ n : ℕ, 1 / (a_n(n) * a_n(n + 1)) 
  let T_n := ∑ i in finset.range n, c_n(i) 
  in T_n = - (n : ℝ) / (2 * (3 * (n : ℝ) - 2)) :=
sorry

-- Problem 2
theorem problem_2 (n : ℕ) (hn : n > 0) :
  let S_n : ℕ → ℝ := λ n, 1 - 1 / (2 ^ n)
  let b_n : ℕ → ℝ := λ n, 
    if n = 1 then S_n(1)
    else S_n(n) - S_n(n - 1)
  in (b_n(1) = 1/2) ∧ (∀ k > 0, b_n(k + 1) = (1 / 2) * b_n(k)) :=
sorry

end problem_1_problem_2_l537_537285


namespace symmetry_axis_g_l537_537767

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3)

theorem symmetry_axis_g :
  ∃ k : ℤ, (x = k * Real.pi / 2 + Real.pi / 4) := sorry

end symmetry_axis_g_l537_537767


namespace expression_value_l537_537197

noncomputable def complicated_expression : ℝ :=
  (1 - real.sqrt 3) ^ 0 + real.sqrt 2 - 2 * real.cos (real.pi / 4) + (1 / 4)⁻¹

theorem expression_value : complicated_expression = 5 := by
  sorry

end expression_value_l537_537197


namespace a3_is_correct_l537_537296

-- Define the sequence by its sum of the first n terms
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence terms using the provided conditions
def a : ℕ → ℚ
| 0       := S 0
| (n + 1) := S (n + 1) - S n

-- The proof goal
theorem a3_is_correct : a 3 = 1 / 20 :=
by
  sorry

end a3_is_correct_l537_537296


namespace arrangements_of_books_in_boxes_l537_537312

theorem arrangements_of_books_in_boxes :
  ∀ (books boxes : ℕ), (books = 6) ∧ (boxes = 5) → (boxes ^ books = 15625) :=
by
  intros books boxes h
  cases h with h_books h_boxes
  rw [h_books, h_boxes]
  sorry

end arrangements_of_books_in_boxes_l537_537312


namespace eighth_positive_odd_multiple_of_5_l537_537480

theorem eighth_positive_odd_multiple_of_5 : 
  let a := 5 in 
  let d := 10 in 
  let n := 8 in 
  a + (n - 1) * d = 75 :=
by
  let a := 5
  let d := 10
  let n := 8
  have : a + (n - 1) * d = 75 := by 
    calc
      a + (n - 1) * d = 5 + (8 - 1) * 10  : by rfl
      ... = 5 + 70                          : by rfl
      ... = 75                              : by rfl
  exact this

end eighth_positive_odd_multiple_of_5_l537_537480


namespace option_A_false_option_B_false_option_C_true_option_D_false_l537_537110

theorem option_A_false : 2^(-2) ≠ -4 := sorry

theorem option_B_false : 2^0 + 3^2 ≠ 9 := sorry

theorem option_C_true : ∀ a : ℝ, a^2 * a^3 = a^(2 + 3) := sorry

theorem option_D_false : 8 / |(-8)| ≠ -1 := sorry

end option_A_false_option_B_false_option_C_true_option_D_false_l537_537110


namespace regular_polygon_sides_l537_537985

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l537_537985


namespace sum_of_first_8_terms_geometric_sequence_l537_537346

theorem sum_of_first_8_terms_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : 2 * a 4 + a 6 = 48) 
  (h2 : a 3 * a 5 = 64) :
  (sum (finset.range 8) a = 255) ∨ (sum (finset.range 8) a = 85) := 
sorry

end sum_of_first_8_terms_geometric_sequence_l537_537346


namespace longest_side_similar_triangle_l537_537072

theorem longest_side_similar_triangle 
  (a b c : ℕ) (p : ℕ) (longest_side : ℕ)
  (h1 : a = 6) (h2 : b = 7) (h3 : c = 9) (h4 : p = 110) 
  (h5 : longest_side = 45) :
  ∃ x : ℕ, (6 * x + 7 * x + 9 * x = 110) ∧ (9 * x = longest_side) :=
by
  sorry

end longest_side_similar_triangle_l537_537072


namespace digit_sum_property_l537_537802

noncomputable def sum_of_digits (x : ℚ) : ℕ :=
  x.to_string.fold 0 (λ c sum, if c.is_digit then sum + c.to_nat - '0'.to_nat else sum)

theorem digit_sum_property (x y : ℚ) (h_xy : x = 2.6) (h_y : y = 5) :
  x * y = 13 ∧ sum_of_digits x + sum_of_digits y = 13 :=
by {
  have h1 : x * y = 13,
  { sorry },
  have h2 : sum_of_digits x + sum_of_digits y = 13,
  { sorry },
  exact ⟨h1, h2⟩
}

end digit_sum_property_l537_537802


namespace max_value_of_expression_l537_537017

variables {V : Type*} [InnerProductSpace ℝ V]

def norm_squared (v : V) : ℝ := ∥v∥^2

theorem max_value_of_expression
  (a b c : V)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 3) :
  ∥3 • a - b∥^2 + ∥b - 3 • c∥^2 + ∥c - 3 • a∥^2 ≤ 170 :=
sorry

end max_value_of_expression_l537_537017


namespace tan_addition_example_l537_537307

theorem tan_addition_example (x : ℝ) (h : Real.tan x = 1/3) : 
  Real.tan (x + π/3) = 2 + 5 * Real.sqrt 3 / 3 := 
by 
  sorry

end tan_addition_example_l537_537307


namespace perimeter_of_triangle_A1DM_distance_from_D1_to_plane_A1DM_l537_537601

theorem perimeter_of_triangle_A1DM (a : ℝ) :
  let A1 := (a, 0, 0)
  let D := (0, 0, a)
  let D1 := (0, 0, 0)
  let M := (a, a / 2, 0)
  let A1M := Real.sqrt (a^2 + (a/2)^2)
  let DM := Real.sqrt (a^2 + (a/2)^2)
  let A1D := Real.sqrt (a^2 + a^2)
  (A1M + DM + A1D) = a * (Real.sqrt 5 + Real.sqrt 2) :=
sorry

theorem distance_from_D1_to_plane_A1DM (a : ℝ) :
  let A1 := (a, 0, 0)
  let D := (0, 0, a)
  let D1 := (0, 0, 0)
  let M := (a, a / 2, 0)
  let normal_vector := (a, 0, a^2 / 2)
  let plane_equation := (p : ℝ × ℝ × ℝ) → a * p.1 - (a^2 / 2) * p.3 = 0
  let distance := |a * 0 + 0 * 0 + (a^2 / 2) * 0 - a| / Real.sqrt (a^2 + 0 + (a^2 / 2)^2)
  distance = a / Real.sqrt 6 :=
sorry

end perimeter_of_triangle_A1DM_distance_from_D1_to_plane_A1DM_l537_537601


namespace regular_polygon_sides_l537_537989

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537989


namespace minimum_tests_for_connected_lattice_3x3_l537_537505

/-- A power grid with a 3x3 lattice of 16 nodes, ensures connectivity with at least 15 tests -/
theorem minimum_tests_for_connected_lattice_3x3 (n : ℕ) (h_n : n = 16) (connected_graph : ∀ (u v : ℕ), u ≠ v → ∃ (path : u = v), true) : ∃ (t : ℕ), t = 15 := by
  have tests_needed : t = n - 1 := sorry -- Minimal spanning tree condition
  rw [h_n] at tests_needed
  exact ⟨15, tests_needed⟩
  sorry

end minimum_tests_for_connected_lattice_3x3_l537_537505


namespace race_outcome_l537_537365

-- Defining the conditions
def track_length : ℝ := 100
def race1_katie_distance : ℝ := 100
def race1_sarah_distance : ℝ := 95
def race2_katie_extra_distance : ℝ := 5

-- Statement that Katie ran 105 meters in the second race
def race2_katie_distance : ℝ := race1_katie_distance + race2_katie_extra_distance

-- Defining the speed ratio
def speed_ratio : ℝ := race1_katie_distance / race1_sarah_distance

-- Defining the distance Sarah runs in the same time Katie runs 105 meters
def race2_sarah_distance (katie_distance : ℝ) : ℝ := katie_distance * (race1_sarah_distance / race1_katie_distance)

-- Defining the proof problem
theorem race_outcome : 
  race2_sarah_distance race2_katie_distance = 99.75 → 
  (track_length - race2_sarah_distance race2_katie_distance = 0.25) :=
by 
  intros h1 
  unfold track_length 
  unfold race2_katie_distance 
  unfold race1_katie_distance 
  unfold race1_sarah_distance 
  unfold race2_katie_extra_distance
  unfold speed_ratio 
  unfold race2_sarah_distance at h1
  exact sorry

end race_outcome_l537_537365


namespace distance_between_points_l537_537186

-- Definition of the two points
def point1 : (ℝ × ℝ) := (2, 3)
def point2 : (ℝ × ℝ) := (7, 11)

-- Statement of the theorem to prove the distance between the points
theorem distance_between_points :
  let dx := (point2.1 - point1.1)
  let dy := (point2.2 - point1.2)
  real.sqrt (dx^2 + dy^2) = real.sqrt 89 :=
by
  sorry

end distance_between_points_l537_537186


namespace sum_of_solutions_eq_zero_l537_537583

def f (x : ℝ) : ℝ := 2^(|x|) + 3 * |x| + x^2

theorem sum_of_solutions_eq_zero :
  (∑ x in {x | f x = 24}, x) = 0 :=
sorry

end sum_of_solutions_eq_zero_l537_537583


namespace spending_difference_l537_537963

def chocolate_price : ℝ := 7
def candy_bar_price : ℝ := 2
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def gum_price : ℝ := 3

def discounted_chocolate_price : ℝ := chocolate_price * (1 - discount_rate)
def total_before_tax : ℝ := candy_bar_price + gum_price
def tax_amount : ℝ := total_before_tax * sales_tax_rate
def total_after_tax : ℝ := total_before_tax + tax_amount

theorem spending_difference : 
  discounted_chocolate_price - candy_bar_price = 3.95 :=
by 
  -- Apply the necessary calculations
  have discount_chocolate : ℝ := discounted_chocolate_price
  have candy_bar : ℝ := candy_bar_price
  calc
    discounted_chocolate_price - candy_bar_price = _ := sorry

end spending_difference_l537_537963


namespace sin_cos_identity_l537_537873

-- Define the mathematical constants
def deg_to_rad (d : ℝ) := d * Real.pi / 180 

theorem sin_cos_identity :
  Real.sin (deg_to_rad 18) * Real.cos (deg_to_rad 12) + Real.cos (deg_to_rad 18) * Real.sin (deg_to_rad 12) = 1 / 2 :=
  sorry

end sin_cos_identity_l537_537873


namespace overtime_rate_is_90_cents_l537_537910

noncomputable def hourly_overtime_rate (ordinary_rate : ℝ) (total_hours : ℕ) (overtime_hours : ℕ) (weekly_pay : ℝ) : ℝ :=
  let ordinary_hours := total_hours - overtime_hours in
  let ordinary_pay := ordinary_hours * ordinary_rate in
  let overtime_pay := weekly_pay - ordinary_pay in
  overtime_pay / overtime_hours

theorem overtime_rate_is_90_cents :
  hourly_overtime_rate 0.60 50 8 32.40 = 0.90 := 
by
  unfold hourly_overtime_rate
  norm_num
  sorry

end overtime_rate_is_90_cents_l537_537910


namespace number_of_cars_with_fuel_consumption_greater_than_9_is_180_l537_537685

noncomputable def cars_with_fuel_consumption_greater_than_nine {ξ : Type*}
    [probability_space ξ] (surveyed_cars : ℕ) (mu : ℝ) (sigma : ℝ) : ℝ :=
let num_greater_than_nine := surveyed_cars * (1 - (cdf normal_distribution.mk 9 mu sigma - cdf normal_distribution.mk 8 mu sigma)) in
num_greater_than_nine

theorem number_of_cars_with_fuel_consumption_greater_than_9_is_180 :
    ∀ (surveyed_cars : ℕ) (mu : ℝ) (sigma : ℝ),
    surveyed_cars = 1200 →
    mu = 8 →
    cdf normal_distribution.mk 9 mu sigma - cdf normal_distribution.mk 7 mu sigma = 0.7 →
    cars_with_fuel_consumption_greater_than_nine surveyed_cars mu sigma = 180 :=
by
    intros
    sorry

end number_of_cars_with_fuel_consumption_greater_than_9_is_180_l537_537685


namespace probability_angle_APB_lt_90_l537_537695

theorem probability_angle_APB_lt_90 (A B C D P : Point)
  (h_square : is_square A B C D)
  (h_random : P ∈ interior(ABCD)) :
  Pr(∠APB < 90°) = 1 - (π / 8) :=
sorry

end probability_angle_APB_lt_90_l537_537695


namespace trip_average_efficiency_l537_537905

def average_fuel_efficiency (distance1 distance2 distance3 : ℕ) 
                            (efficiency1 efficiency2 efficiency3 : ℕ) : ℕ :=
  let total_distance := distance1 + distance2 + distance3
  let total_gallons := distance1 / efficiency1 + distance2 / efficiency2 + distance3 / efficiency3
  total_distance / total_gallons

theorem trip_average_efficiency :
  average_fuel_efficiency 150 150 50 25 15 10 = 16 := 
begin
  sorry
end

end trip_average_efficiency_l537_537905


namespace area_of_shaded_region_l537_537336

open Set

def is_parallelogram (A B C D : Point) : Prop :=
  (A.y = B.y ∧ C.y = D.y) ∧
  (A.x - B.x = D.x - C.x)

def midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

variables (A B C D E : Point)
variables (hAB : ℝ) (hheight : ℝ) (hDE : ℝ)

-- Conditions
axiom h1 : is_parallelogram A B C D
axiom h2 : E = midpoint D C
axiom h3 : hAB = 12
axiom h4 : hheight = 10
axiom hDE : hDE = 6

-- Proof statement
theorem area_of_shaded_region : 
  area_of_shaded BEDC E midpoint D C = 30 :=
sorry

end area_of_shaded_region_l537_537336


namespace probability_of_perfect_square_l537_537896

theorem probability_of_perfect_square :
  let p : ℚ := 1 / 350 in
  let prob_squares_le_100 : ℚ := 10 * p in
  let prob_squares_gt_100 : ℚ := 2 * 5 * p in
  prob_squares_le_100 + prob_squares_gt_100 = 0.05714 :=
by
  let p : ℚ := 1 / 350
  let prob_squares_le_100 := 10 * p
  let prob_squares_gt_100 := 2 * 5 * p
  have h : prob_squares_le_100 + prob_squares_gt_100 = 0.05714
  sorry

end probability_of_perfect_square_l537_537896


namespace cos_E_of_convex_quadrilateral_l537_537332

theorem cos_E_of_convex_quadrilateral (u v : ℝ) (h1 : u ≠ v) (h2 : u + v + 300 = 580) : 
  let β := (u + v) / 300 in
  cos β = 14 / 15 :=
by
  sorry

end cos_E_of_convex_quadrilateral_l537_537332


namespace area_of_R_sum_m_n_l537_537523

theorem area_of_R_sum_m_n  (s : ℕ) 
  (square_area : ℕ) 
  (rectangle1_area : ℕ)
  (rectangle2_area : ℕ) :
  square_area = 4 → rectangle1_area = 8 → rectangle2_area = 2 → s = 6 → 
  36 - (square_area + rectangle1_area + rectangle2_area) = 22 :=
by
  intros
  sorry

end area_of_R_sum_m_n_l537_537523


namespace weight_of_B_l537_537431

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l537_537431


namespace find_f_f_half_l537_537243

noncomputable def f : ℝ → ℝ :=
λ x, if |x| ≤ 1 then |x - 1| else 1 / (1 + x^2)

theorem find_f_f_half : f (f (1/2)) = 1/2 := 
by 
  sorry

end find_f_f_half_l537_537243


namespace log_sum_geometric_sequence_l537_537019

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (r : ℝ) 
  (a_pos : ∀ n, 0 < a n)
  (geometric_seq : ∀ n, a (n + 1) = a n * r)
  (h : a 5 * a 6 = 81) :
  ∑ i in finset.range 10, real.log (a (i + 1)) / real.log 3 = 20 :=
sorry

end log_sum_geometric_sequence_l537_537019


namespace smallest_possible_beta_l537_537375

theorem smallest_possible_beta
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 1)
  (h3 : ‖c‖ = 1)
  (h4 : ∃ β : ℝ, arccos (a ⬝ b) = β)
  (h5 : ∃ β : ℝ, arccos (c ⬝ (a × b)) = β)
  (h6 : b ⬝ (c × a) = 1 / 2) :
  ∃ β : ℝ, β = 45 :=
by
  sorry

end smallest_possible_beta_l537_537375


namespace integral_negative_of_negative_function_l537_537426

theorem integral_negative_of_negative_function {f : ℝ → ℝ} 
  (hf_cont : Continuous f) 
  (hf_neg : ∀ x, f x < 0) 
  {a b : ℝ} 
  (hab : a < b) 
  : ∫ x in a..b, f x < 0 := 
sorry

end integral_negative_of_negative_function_l537_537426


namespace problem_c_and_sinA_l537_537692

noncomputable def find_c (b : ℝ) (sin_B : ℝ) (sin_C : ℝ) : ℝ :=
  b / sin_B * sin_C

noncomputable def find_sin_A (sin_B : ℝ) (cos_C : ℝ) (cos_B : ℝ) (sin_C : ℝ) : ℝ :=
  sin_B * cos_C + cos_B * sin_C

theorem problem_c_and_sinA (a b c : ℝ) (A B C : ℝ)
    (sin_B : ℝ) (sin_C : ℝ) (cos_C : ℝ) (cos_B : ℝ) :
   C = 45 ∧
   b = 4 * Real.sqrt 5 ∧
   sin_B = 2 * Real.sqrt 5 / 5 ∧
   cos_C = Real.sqrt 2 / 2 ∧
   cos_B = Real.sqrt 5 / 5 →
   c = find_c b sin_B sin_C ∧
   sin A = find_sin_A sin_B cos_C cos_B sin_C :=
by
  intros
  dsimp [find_c, find_sin_A]
  sorry

end problem_c_and_sinA_l537_537692


namespace eighth_odd_multiple_of_5_l537_537471

theorem eighth_odd_multiple_of_5 : 
  (∃ n : ℕ, n = 8 ∧ ∃ k : ℤ, k = (10 * n - 5) ∧ k > 0 ∧ k % 2 = 1) → 75 := 
by {
  sorry
}

end eighth_odd_multiple_of_5_l537_537471


namespace log_proportional_l537_537406

noncomputable theory

open Real

theorem log_proportional (a b P K : ℝ) (ha : a > 0) (hb : b > 0) (hPa : a ≠ 1) (hPb : b ≠ 1) (hP : P > 0) (hK : K > 0) :
  (log P / log a) / (log P / log b) = (log K / log a) / (log K / log b) :=
sorry

end log_proportional_l537_537406


namespace percentage_decrease_is_40_percent_l537_537396

variable (R : ℝ) -- revenue from last year
def projected_revenue (R : ℝ) : ℝ := 1.25 * R
def actual_revenue (projected_revenue : ℝ) : ℝ := 0.60 * projected_revenue
def revenue_decrease (projected_revenue actual_revenue : ℝ) : ℝ := projected_revenue - actual_revenue
def percentage_decrease (revenue_decrease projected_revenue : ℝ) : ℝ := (revenue_decrease / projected_revenue) * 100

theorem percentage_decrease_is_40_percent (R : ℝ) : 
    percentage_decrease (revenue_decrease (projected_revenue R) (actual_revenue (projected_revenue R))) (projected_revenue R) = 40 := 
by 
  sorry

end percentage_decrease_is_40_percent_l537_537396


namespace satisfies_inequality_l537_537172

theorem satisfies_inequality : 
  ∀ x : ℚ, x ∈ {-3, -1/2, 1/3, 2} → (2 * (x - 1) + 3 < 0 ↔ x = -3) :=
by 
  intro x h 
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h
  cases h <;> simp [h, show -3 - 1 = -4 from rfl, show -1/2 - 1 = -3/2 from rfl, show 1/3 - 1 = -2/3 from rfl, show 2 - 1 = 1 from rfl]
  sorry

end satisfies_inequality_l537_537172


namespace third_meeting_point_l537_537148

-- Define the constants and the main theorem
def distance_AB : ℕ := 100
def first_meet_distance : ℕ := 20
def second_meet_distance_from_B : ℕ := 20
def third_meet_distance_from_A : ℕ := 45

-- We state the theorem that needs to be proved
theorem third_meeting_point :
  ∀ (distance_AB = 100) (first_meet_distance = 20) (second_meet_distance_from_B = 20), 
   third_meet_distance_from_A = 45 := 
by
  sorry

end third_meeting_point_l537_537148


namespace value_range_of_f_l537_537820

noncomputable def f (x : ℝ) : ℝ := sin^2 (3 * π / 2 - x) + sin (x + π)

theorem value_range_of_f : set.image f set.univ = set.Icc (-1) (5/4) :=
by
  sorry

end value_range_of_f_l537_537820


namespace intersection_of_A_and_B_l537_537029

/-- The set A is defined as {0, 1, 2}. --/
def A : Set ℕ := {0, 1, 2}

/-- The set B is defined as {x | -1 < x < 2}. --/
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

/-- Intersection of A and B is {0, 1}. --/
theorem intersection_of_A_and_B : A ∩ {x | -1 < x ∧ x < 2} = {0, 1} := by
  sorry

end intersection_of_A_and_B_l537_537029


namespace t_shirts_sold_l537_537778

theorem t_shirts_sold
  (minutes : ℕ)
  (black_price : ℕ)
  (white_price : ℕ)
  (revenue_per_minute : ℕ)
  (total_revenue : ℕ)
  (T : ℕ) :
  minutes = 25 →
  black_price = 30 →
  white_price = 25 →
  revenue_per_minute = 220 →
  (T / 2) * black_price + (T / 2) * white_price = total_revenue →
  total_revenue = minutes * revenue_per_minute →
  T = 200 :=
by
  intros h_minutes h_black_price h_white_price h_revenue_per_minute h_revenue_eq h_total_revenue_eq
  rw [h_minutes, h_black_price, h_white_price, h_revenue_per_minute, h_total_revenue_eq] at h_revenue_eq
  sorry

end t_shirts_sold_l537_537778


namespace train_speed_approximation_l537_537539

theorem train_speed_approximation :
  let distance_meters := 300
  let time_seconds := 11.999040076793857
  let distance_kilometers := distance_meters / 1000.0
  let speed_km_per_sec := distance_kilometers / time_seconds
  let speed_km_per_hr := speed_km_per_sec * 3600.0
  speed_km_per_hr ≈ 90.002 :=
by
  sorry

end train_speed_approximation_l537_537539


namespace symmetric_point_x_axis_l537_537249

variable (M : ℝ × ℝ × ℝ)
variable (M_coords : M = (2, 1, 3))

def symmetric_with_respect_to_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

theorem symmetric_point_x_axis (M : ℝ × ℝ × ℝ) (h : M = (2, 1, 3)) :
    symmetric_with_respect_to_x M = (2, -1, 3) := by
  sorry

end symmetric_point_x_axis_l537_537249


namespace cover_points_with_circles_l537_537757

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ) :
  ∃ (radii : Fin n → ℝ) (centers : Fin n → ℝ × ℝ),
  (∀ i j, i ≠ j → (dist (centers i) (centers j) > radii i + radii j + 1)) ∧
  (∑ i, 2 * radii i < n) ∧
  (∀ i, ∃ j, points i = centers j) :=
by
  sorry

end cover_points_with_circles_l537_537757


namespace reading_club_mean_days_l537_537746

theorem reading_club_mean_days :
  let students_days := [(2, 1), (4, 2), (5, 3), (10, 4), (7, 5), (3, 6), (2, 7)]
  let total_days := (2 * 1) + (4 * 2) + (5 * 3) + (10 * 4) + (7 * 5) + (3 * 6) + (2 * 7)
  let total_students := 2 + 4 + 5 + 10 + 7 + 3 + 2
  (total_days / total_students : ℚ) = 4.0 :=
by
  let students_days := [(2, 1), (4, 2), (5, 3), (10, 4), (7, 5), (3, 6), (2, 7)]
  let total_days := (2 * 1) + (4 * 2) + (5 * 3) + (10 * 4) + (7 * 5) + (3 * 6) + (2 * 7)
  let total_students := 2 + 4 + 5 + 10 + 7 + 3 + 2
  have h1 : total_days = 132 := by sorry
  have h2 : total_students = 33 := by sorry
  have h3 : (132 / 33 : ℚ) = 4 := by sorry
  exact h3

end reading_club_mean_days_l537_537746


namespace total_cost_is_correct_l537_537207

-- Define the constants and conditions
def first_tire_cost : ℝ := 0.50
def remaining_tire_cost : ℝ := 0.40
def sales_tax_rate : ℝ := 0.05

-- Number of tires in each category
def num_first_tires : ℕ := 4
def num_remaining_tires : ℕ := 4

-- Calculate the costs
def cost_first_tires : ℝ := num_first_tires * first_tire_cost
def cost_remaining_tires : ℝ := num_remaining_tires * remaining_tire_cost
def total_cost_before_tax : ℝ := cost_first_tires + cost_remaining_tires
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate
def total_cost : ℝ := total_cost_before_tax + sales_tax

-- The theorem to be proved
theorem total_cost_is_correct : total_cost = 3.78 :=
by
  unfold cost_first_tires cost_remaining_tires total_cost_before_tax sales_tax total_cost
  -- Simplify the arithmetic (actual arithmetic details can be skipped here)
  sorry

end total_cost_is_correct_l537_537207


namespace circle_properties_l537_537628

def circle_center_line (x y : ℝ) : Prop := x + y - 1 = 0

def point_A_on_circle (x y : ℝ) : Prop := (x, y) = (-1, 4)
def point_B_on_circle (x y : ℝ) : Prop := (x, y) = (1, 2)

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

def slope_range_valid (k : ℝ) : Prop :=
  k ≤ 0 ∨ k ≥ 4 / 3

theorem circle_properties
  (x y : ℝ)
  (center_x center_y : ℝ)
  (h_center_line : circle_center_line center_x center_y)
  (h_point_A_on_circle : point_A_on_circle x y)
  (h_point_B_on_circle : point_B_on_circle x y)
  (h_circle_equation : circle_equation x y)
  (k : ℝ) :
  circle_equation center_x center_y ∧ slope_range_valid k :=
sorry

end circle_properties_l537_537628


namespace store_earned_11740_dollars_l537_537144

theorem store_earned_11740_dollars :
  let graphics_cards_sold := 10
  let graphics_card_price := 600
  let hard_drives_sold := 14
  let hard_drive_price := 80
  let cpus_sold := 8
  let cpus_returned := 2
  let cpu_price := 200
  let ram_pairs_sold := 4
  let ram_pair_price := 60
  let psus_sold := 12
  let psu_price := 90
  let monitors_sold := 6
  let monitor_price := 250
  let keyboards_sold := 18
  let keyboard_price := 40
  let mice_sold := 24
  let mouse_price := 20
  let discount := 0.10
  let total_sales := graphics_cards_sold * graphics_card_price
                  + hard_drives_sold * hard_drive_price
                  + cpus_sold * cpu_price
                  + ram_pairs_sold * ram_pair_price
                  + psus_sold * psu_price
                  + monitors_sold * monitor_price
                  + keyboards_sold * keyboard_price
                  + mice_sold * mouse_price
  let discount_amount := discount * (graphics_cards_sold * graphics_card_price)
  let sales_after_discount := (graphics_cards_sold * graphics_card_price) - discount_amount
  let sales_after_returns := (cpus_sold * cpu_price) - (cpus_returned * cpu_price)
  let adjusted_total_sales := sales_after_discount
                              + (hard_drives_sold * hard_drive_price)
                              + sales_after_returns
                              + (ram_pairs_sold * ram_pair_price)
                              + (psus_sold * psu_price)
                              + (monitors_sold * monitor_price)
                              + (keyboards_sold * keyboard_price)
                              + (mice_sold * mouse_price)
  in adjusted_total_sales = 11740 := sorry

end store_earned_11740_dollars_l537_537144


namespace minimum_value_problem_l537_537382

noncomputable def minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : Prop :=
  (2 * x + 3 * y) * (2 * y + 3 * z) * (2 * x * z + 1) ≥ 24

theorem minimum_value_problem 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : 
  minimum_value_inequality x y z hx hy hz hxyz :=
by sorry

end minimum_value_problem_l537_537382


namespace remainder_276_l537_537492

theorem remainder_276 (y : ℤ) (k : ℤ) (hk : y = 23 * k + 19) : y % 276 = 180 :=
sorry

end remainder_276_l537_537492


namespace line_within_plane_correct_l537_537123

-- Definitions of sets representing a line and a plane
variable {Point : Type}
variable (l α : Set Point)

-- Definition of the statement
def line_within_plane : Prop := l ⊆ α

-- Proof statement (without the actual proof)
theorem line_within_plane_correct (h : l ⊆ α) : line_within_plane l α :=
by
  sorry

end line_within_plane_correct_l537_537123


namespace area_code_permutations_l537_537213

theorem area_code_permutations : 
  ∀ (digits : Multiset ℕ), digits = {5, 5, 1} → Multiset.countP digits = 3 :=
by
  sorry

end area_code_permutations_l537_537213


namespace incenter_x_coord_eq_2_l537_537895

noncomputable def equidistant_point (x y : ℝ) : Prop :=
  abs y = abs x ∧ abs (x + y - 4) / real.sqrt 2 = abs x

theorem incenter_x_coord_eq_2 : ∃ y : ℝ, equidistant_point 2 y :=
by {
  sorry
}

end incenter_x_coord_eq_2_l537_537895


namespace ellipse_param_eq_l537_537612

theorem ellipse_param_eq : 
  ∃ a b : ℝ, (a = 2) ∧ (b = 1) ∧ 
             (∀ θ : ℝ, (∃ e : ℝ, e = sqrt 3 / 2) ∧ 
                       | (2 * cos θ - cos θ, sin θ - sin θ) |^2 = (1 + sqrt 7)^2 ∧ 
                       (x = a * cos θ ∧ y = b * sin θ)) := 
by
  sorry

end ellipse_param_eq_l537_537612


namespace absentees_in_morning_session_is_three_l537_537044

theorem absentees_in_morning_session_is_three
  (registered_morning : ℕ)
  (registered_afternoon : ℕ)
  (absent_afternoon : ℕ)
  (total_students : ℕ)
  (total_registered : ℕ)
  (attended_afternoon : ℕ)
  (attended_morning : ℕ)
  (absent_morning : ℕ) :
  registered_morning = 25 →
  registered_afternoon = 24 →
  absent_afternoon = 4 →
  total_students = 42 →
  total_registered = registered_morning + registered_afternoon →
  attended_afternoon = registered_afternoon - absent_afternoon →
  attended_morning = total_students - attended_afternoon →
  absent_morning = registered_morning - attended_morning →
  absent_morning = 3 :=
by
  intros
  sorry

end absentees_in_morning_session_is_three_l537_537044


namespace rationalize_sum_l537_537760

theorem rationalize_sum {A B C D E : ℤ} (h₁ : B < D)
  (h₂ : is_simplest_form (12 * Real.sqrt 6 - 15 * Real.sqrt 7) 79)
  (h₃ : 3 / (4 * Real.sqrt 6 + 5 * Real.sqrt 7) = (A * Real.sqrt B + C * Real.sqrt D) / E) :
  A + B + C + D + E = 89 :=
by
  sorry

-- Assuming is_simplest_form is a predicate you could prove or define elsewhere specifying the simplest form condition.

end rationalize_sum_l537_537760


namespace parabola_through_intersection_points_l537_537402

noncomputable theory

-- Definitions from conditions in part a
def parabola_eqn (x y : ℝ) : Prop := x = y^2

def circle_eqn (x y : ℝ) : Prop := (x - 11)^2 + (y - 1)^2 = 25

-- The proof problem in Lean 4 statement
theorem parabola_through_intersection_points :
  ∀ (A B C D : ℝ × ℝ),
    parabola_eqn A.1 A.2 ∧ circle_eqn A.1 A.2 ∧
    parabola_eqn B.1 B.2 ∧ circle_eqn B.1 B.2 ∧
    parabola_eqn C.1 C.2 ∧ circle_eqn C.1 C.2 ∧
    parabola_eqn D.1 D.2 ∧ circle_eqn D.1 D.2 →
    ∃ (a b c : ℝ), a = 1/2 ∧ b = -21/2 ∧ c = 97/2 ∧
      (∀ (x : ℝ), A.2 = a * A.1^2 + b * A.1 + c ∧
                 B.2 = a * B.1^2 + b * B.1 + c ∧
                 C.2 = a * C.1^2 + b * C.1 + c ∧
                 D.2 = a * D.1^2 + b * D.1 + c) :=
begin
  sorry
end

end parabola_through_intersection_points_l537_537402


namespace sum_largest_smallest_f_l537_537023

def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2 * x - 6|

theorem sum_largest_smallest_f :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 8 → f x = 0 ∨ (f x = 2))
  → 2 :=
begin
  sorry -- Proof skipped
end

end sum_largest_smallest_f_l537_537023


namespace closest_integer_sum_l537_537228

theorem closest_integer_sum :
  let S := 500 * (∑ n in finset.range (5000 - 4 + 1 + 1), 1 / ((n + 4)^2 - 9))
  in abs (S - 102) < 1 :=
by
  let S := 500 * (∑ n in finset.range (5000 - 4 + 1 + 1), 1 / ((n + 4)^2 - 9))
  sorry

end closest_integer_sum_l537_537228


namespace cos_60_eq_sqrt3_div_2_l537_537929

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537929


namespace cookies_left_l537_537409

theorem cookies_left 
  (brother_cookies : ℕ)
  (mother_factor : ℚ)
  (sister_factor : ℚ)
  (initial_cookies : ℕ) :
  brother_cookies = 10 →
  mother_factor = 1 / 2 →
  sister_factor = 2 / 3 →
  initial_cookies = 20 →
  let cookies_after_brother := initial_cookies - brother_cookies,
      mother_cookies := mother_factor * brother_cookies,
      cookies_after_mother := cookies_after_brother + mother_cookies,
      cookies_to_sister := sister_factor * cookies_after_mother,
      final_cookies := cookies_after_mother - cookies_to_sister
  in
  final_cookies = 5 :=
by
  intro h1 h2 h3 h4,
  simp only [h1, h2, h3, h4, sub_eq_add_neg, int.cast_of_nat, mul_assoc],
  norm_num,
  sorry

end cookies_left_l537_537409


namespace sum_of_integers_l537_537835

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := 
by
  sorry

end sum_of_integers_l537_537835


namespace investment_duration_l537_537578

theorem investment_duration (P A : ℝ) (R : ℝ) (hP : P = 1100) (hA : A = 1232) (hR : R = 5) :
  ∃ T : ℝ, T = 2.4 := by
  have I : ℝ := A - P
  have hI : I = 132 := by
    rw [hA, hP]
    sorry
  have hI_eq : I = P * R * 2.4 / 100 := by
    rw [hI, hP, hR]
    sorry
  use 2.4
  exact eq_of_mul_eq_mul_right (by norm_num) (by
    rw [←mul_assoc, ←div_eq_inv_mul, mul_comm (P * R) _, ←mul_assoc]
    sorry)

end investment_duration_l537_537578


namespace remainder_is_linear_l537_537374

theorem remainder_is_linear (Q : ℚ[X]) (h1 : Q.eval 25 = 50) (h2 : Q.eval 50 = 25) :
  ∃ a b, (Q - ((X - 25) * (X - 50) * 1)).degree ≤ 1 ∧ (Q - ((X - 25) * (X - 50) * 1)).coeff 1 = a
   ∧ (Q - ((X - 25) * (X - 50) * 1)).coeff 0 = b ∧ a = -1 ∧ b = 75 := by
  sorry

end remainder_is_linear_l537_537374


namespace probability_first_qualified_on_third_test_l537_537554

definition pass_rate := (3:ℝ) / 4
definition fail_rate := (1:ℝ) / 4

theorem probability_first_qualified_on_third_test : 
  let P := (fail_rate) * (fail_rate) * (pass_rate)
  P = (1 / 4)^2 * (3 / 4) :=
by
  sorry

end probability_first_qualified_on_third_test_l537_537554


namespace regular_polygon_sides_l537_537993

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537993


namespace add_base8_l537_537542

/-- Define the numbers in base 8 --/
def base8_add (a b : Nat) : Nat := 
  sorry

theorem add_base8 : base8_add 0o12 0o157 = 0o171 := 
  sorry

end add_base8_l537_537542


namespace distinct_possible_values_of_c_l537_537734

theorem distinct_possible_values_of_c {c r s t : ℂ}
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_condition : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c^2 * s) * (z - c^4 * t)):
  ∃ cs : Finset ℂ, cs.card = 7 ∧ ∀ c' ∈ cs, c'^7 = 1 :=
begin
  sorry
end

end distinct_possible_values_of_c_l537_537734


namespace cover_points_with_non_overlapping_circles_l537_537756

theorem cover_points_with_non_overlapping_circles (n : ℕ) (n_pos : 0 < n)
  (points : fin n → ℝ × ℝ) :
  ∃ (k : ℕ) (circles : fin k → (ℝ × ℝ) × ℝ), k > 0 ∧
  (∀ i j, i ≠ j → dist (circles i).fst (circles j).fst > 1) ∧
  (∑ i, 2 * (circles i).snd < n) ∧
  (∀ p, p ∈ points → ∃ i, dist p (circles i).fst ≤ (circles i).snd) :=
sorry

end cover_points_with_non_overlapping_circles_l537_537756


namespace maura_classroom_students_l537_537087

theorem maura_classroom_students (T : ℝ) (h1 : Tina_students = T) (h2 : Maura_students = T) (h3 : Zack_students = T / 2) (h4 : Tina_students + Maura_students + Zack_students = 69) : T = 138 / 5 := by
  sorry

end maura_classroom_students_l537_537087


namespace median_A_is_86_variance_A_is_11_point_2_mean_B_is_85_l537_537463

/-- Definitions for student A and student B test scores -/
def scores_A : List ℝ := [86, 83, 90, 80, 86]
def scores_B : List ℝ := [78, 82, 84, 89, 92]

/-- Median calculation for a list of scores -/
def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (<=)
  if sorted.length % 2 = 1 then -- odd number of elements
    sorted.get (sorted.length / 2)
  else -- even number of elements
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

/-- Mean calculation for a list of scores -/
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

/-- Variance calculation for a list of scores, given the mean -/
def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

/-- Proof that the median of scores_A is 86 -/
theorem median_A_is_86 : median scores_A = 86 := by
  sorry

/-- Proof that the variance of scores_A with mean 85 is 11.2 -/
theorem variance_A_is_11_point_2 : variance scores_A 85 = 11.2 := by
  sorry

/-- Proof that the mean of scores_B is 85 -/
theorem mean_B_is_85 : mean scores_B = 85 := by
  sorry

end median_A_is_86_variance_A_is_11_point_2_mean_B_is_85_l537_537463


namespace negation_of_universal_prop_l537_537075

open Classical

variable {R : Type} [LinearOrderedField R]

theorem negation_of_universal_prop {P : R → Prop} :
  (¬ (∀ x : R, x^2 ≤ 1)) = (∃ x : R, x^2 > 1) :=
by
  simp
  sorry

end negation_of_universal_prop_l537_537075


namespace weight_of_second_square_l537_537907

-- Define the problem constants and conditions
def side_length_original := 4
def weight_original := 16
def side_length_second := 6

-- Define the condition for uniform density and thickness
def uniform_density_thickness := true

-- Problem statement (theorem)
theorem weight_of_second_square :
  uniform_density_thickness →
  let A1 := side_length_original ^ 2 in
  let A2 := side_length_second ^ 2 in
  let proportion := (A2 : ℚ) / A1 in
  let weight_second := weight_original * proportion in
  weight_second = 36 := 
begin
  intro uniform_density_thickness,
  unfold uniform_density_thickness A1 A2 proportion weight_second,
  norm_num, -- Automatically simplifies numeric expressions
end

end weight_of_second_square_l537_537907


namespace arrival_at_same_time_l537_537556

-- Define the speeds of the actors
def motorcycle_speed : ℝ := 20
def baldwin_speed : ℝ := 5
def clark_speed : ℝ := 4

-- Define the total distance
def total_distance : ℝ := 52

-- Define the time it takes for all to arrive at the destination together
def desired_time : ℝ := 5

-- Proof that it is possible for Atkinson, Baldwin, and Clark to arrive at the destination at the same time
theorem arrival_at_same_time :
  ∃ (travel_plan : list ℝ), 
  travel_plan.length = 3 ∧
  ∀ (i : ℕ), i < 3 → 
    let distance := [40,  16, 52].nth i in 
    let time := (total_distance / desired_time) in
    travel_plan[i] = time := 
by
  sorry

end arrival_at_same_time_l537_537556


namespace series_sum_is_neg50_l537_537563

theorem series_sum_is_neg50 :
  ∑ k in Finset.range 100, (-1)^k * (k + 1) = -50 :=
sorry

end series_sum_is_neg50_l537_537563


namespace days_worked_prove_l537_537517

/-- Work rate of A is 1/15 work per day -/
def work_rate_A : ℚ := 1/15

/-- Work rate of B is 1/20 work per day -/
def work_rate_B : ℚ := 1/20

/-- Combined work rate of A and B -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B

/-- Fraction of work left after some days -/
def fraction_work_left : ℚ := 8/15

/-- Fraction of work completed after some days -/
def fraction_work_completed : ℚ := 1 - fraction_work_left

/-- Number of days A and B worked together -/
def days_worked_together : ℚ := fraction_work_completed / combined_work_rate

theorem days_worked_prove : 
    days_worked_together = 4 := 
by 
    sorry

end days_worked_prove_l537_537517


namespace sixth_oak_placement_l537_537912

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_aligned (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

noncomputable def intersection_point (p1 p2 p3 p4 : Point) : Point := 
  let m1 := (p2.y - p1.y) / (p2.x - p1.x)
  let m2 := (p4.y - p3.y) / (p4.x - p3.x)
  let c1 := p1.y - (m1 * p1.x)
  let c2 := p3.y - (m2 * p3.x)
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  ⟨x, y⟩

theorem sixth_oak_placement 
  (A1 A2 A3 B1 B2 B3 : Point) 
  (hA : ¬ is_aligned A1 A2 A3)
  (hB : ¬ is_aligned B1 B2 B3) :
  ∃ P : Point, (∃ (C1 C2 : Point), C1 = A1 ∧ C2 = B1 ∧ is_aligned C1 C2 P) ∧ 
               (∃ (C3 C4 : Point), C3 = A2 ∧ C4 = B2 ∧ is_aligned C3 C4 P) := by
  sorry

end sixth_oak_placement_l537_537912


namespace moon_speed_conversion_l537_537074

def moon_speed_km_sec : ℝ := 1.04
def seconds_per_hour : ℝ := 3600

theorem moon_speed_conversion :
  (moon_speed_km_sec * seconds_per_hour) = 3744 := by
  sorry

end moon_speed_conversion_l537_537074


namespace exam_scores_symmetric_normal_distribution_l537_537137

noncomputable def number_of_students_with_score_geq_120 : ℕ :=
  50

def normal_distribution_X : ProbabilityMeasure ℝ :=
  ProbabilityMeasure.normal 110 10

def P_100_110 : ℝ :=
  0.34

theorem exam_scores_symmetric_normal_distribution 
  (num_students : ℕ) (hx : normal_distribution_X)
  (hprob : P (Icc 100 110) = P_100_110) :
  num_students_with_geq_120 = 8 := 
  sorry

end exam_scores_symmetric_normal_distribution_l537_537137


namespace math_proof_l537_537189

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end math_proof_l537_537189


namespace sum_fifth_powers_divisible_by_15_l537_537818

theorem sum_fifth_powers_divisible_by_15
  (A B C D E : ℤ) 
  (h : A + B + C + D + E = 0) : 
  (A^5 + B^5 + C^5 + D^5 + E^5) % 15 = 0 := 
by 
  sorry

end sum_fifth_powers_divisible_by_15_l537_537818


namespace regular_polygon_sides_l537_537984

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l537_537984


namespace probability_diff_colors_l537_537888

theorem probability_diff_colors :
  let total_marbles := 24
  let prob_diff_colors := 
    (4 / 24) * (5 / 23) + 
    (4 / 24) * (12 / 23) + 
    (4 / 24) * (3 / 23) + 
    (5 / 24) * (12 / 23) + 
    (5 / 24) * (3 / 23) + 
    (12 / 24) * (3 / 23)
  prob_diff_colors = 191 / 552 :=
by sorry

end probability_diff_colors_l537_537888


namespace divisible_by_7_tail_cutting_l537_537405

theorem divisible_by_7_tail_cutting (A : ℕ) (hA : A > 0) :
  (∃ n : ℕ, tail_cutting_iter n A % 7 = 0) ↔ (A % 7 = 0) :=
sorry

/-
  Definitions necessary for the theorem.
  "tail_cutting" function is the operation described, and "tail_cutting_iter" applies this "n" times.
-/

-- Helper function: Remove the last digit, double it, and subtract from the truncated number.
def tail_cutting (n : ℕ) : ℕ :=
  n / 10 - 2 * (n % 10)

-- Helper function: Apply tail_cutting function iteratively.
def tail_cutting_iter : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := tail_cutting_iter k (tail_cutting n)

end divisible_by_7_tail_cutting_l537_537405


namespace find_n_value_l537_537118

theorem find_n_value : 
  let n := (1 / 3) + (1 / 4) in 
  n = 7 / 12 :=
by
  sorry

end find_n_value_l537_537118


namespace removed_volume_percentage_l537_537903

theorem removed_volume_percentage :
  let original_volume := 20 * 14 * 12 in
  let cube_side := 4 in
  let cube_volume := cube_side^3 in
  let total_removed_volume := 8 * cube_volume in
  (total_removed_volume / original_volume : ℚ) * 100 = 15.24 :=
begin
  sorry
end

end removed_volume_percentage_l537_537903


namespace count_pairs_l537_537394

-- Definition of the set S
inductive S : Type
| A0 | A1 | A2 | A3

open S

-- Definition of the operation ⊕ on S
def op (a b : S) : S :=
match a, b with
| A0, A0 => A0
| A0, A1 => A1
| A0, A2 => A2
| A0, A3 => A3
| A1, A0 => A1
| A1, A1 => A2
| A1, A2 => A3
| A1, A3 => A0
| A2, A0 => A2
| A2, A1 => A3
| A2, A2 => A0
| A2, A3 => A1
| A3, A0 => A3
| A3, A1 => A0
| A3, A2 => A1
| A3, A3 => A2

-- The mathematical statement to be proven
theorem count_pairs : 
  (finset.univ.product finset.univ).filter (λ (p : S × S), op (op p.1 p.1) p.2 = A0).card = 4 :=
by 
  sorry

end count_pairs_l537_537394


namespace number_of_sides_of_regular_polygon_l537_537976

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l537_537976


namespace people_left_line_l537_537555

theorem people_left_line (initial new final L : ℕ) 
  (h1 : initial = 30) 
  (h2 : new = 5) 
  (h3 : final = 25) 
  (h4 : initial - L + new = final) : L = 10 := by
  sorry

end people_left_line_l537_537555


namespace inverse_proposition_l537_537070

theorem inverse_proposition :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ y : ℝ, y^2 > 0 → y < 0) :=
by
  sorry

end inverse_proposition_l537_537070


namespace annual_population_increase_l537_537446

theorem annual_population_increase 
  (P : ℕ) (A : ℕ) (t : ℕ) (r : ℚ)
  (hP : P = 10000)
  (hA : A = 14400)
  (ht : t = 2)
  (h_eq : A = P * (1 + r)^t) :
  r = 0.2 :=
by
  sorry

end annual_population_increase_l537_537446


namespace fault_line_in_domino_grid_l537_537199

theorem fault_line_in_domino_grid :
  ∀ (grid : fin 6 × fin 6),
  ∀ (cover : list (fin 6 × fin 6) × (fin 6 × fin 6)),
  (∀ pos ∈ grid, ∃ d ∈ cover, pos = d.1 ∨ pos = d.2) →
  (∃ line : fin 6, ¬ ∃ d ∈ cover, (d.1.1 = line ∧ d.2.1 = line) ∨ (d.1.2 = line ∧ d.2.2 = line)) :=
by sorry

end fault_line_in_domino_grid_l537_537199


namespace even_function_value_at_2_l537_537631

theorem even_function_value_at_2 :
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x < 0 → f x = x + 1) →
  f 2 = -1 :=
by
  intros h_even h_neg
  have h1: f (-2) = -1 := by
    have h_neg_eval : f -2 = -2 + 1 := h_neg (-2) (by linarith)
    exact h_neg_eval
  have h2 : f 2 = f (-2) := h_even 2
  rw h1 at h2
  exact h2

end even_function_value_at_2_l537_537631


namespace number_of_ants_proof_l537_537150

-- Define the conditions
def width_ft := 500
def length_ft := 600
def ants_per_sq_inch := 4
def inches_per_foot := 12

-- Define the calculation to get the number of ants
def number_of_ants (width_ft : ℕ) (length_ft : ℕ) (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :=
  let width_inch := width_ft * inches_per_foot
  let length_inch := length_ft * inches_per_foot
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

-- Prove the number of ants is approximately 173 million
theorem number_of_ants_proof :
  number_of_ants width_ft length_ft ants_per_sq_inch inches_per_foot = 172800000 :=
by
  sorry

end number_of_ants_proof_l537_537150


namespace find_abc_l537_537800

theorem find_abc (a b c : ℝ) 
  (h1 : a = 0.8 * b) 
  (h2 : c = 1.4 * b) 
  (h3 : c - a = 72) : 
  a = 96 ∧ b = 120 ∧ c = 168 := 
by
  sorry

end find_abc_l537_537800


namespace expected_winnings_correct_l537_537161

-- Definition of probabilities and winnings in different scenarios
def p6 : ℝ := 1 / 4
def p2to5 : ℝ := 1 / 2
def p1 : ℝ := 1 / 4

def w6 : ℝ := 4
def w2to5 : ℝ := 2
def l1 : ℝ := -6

-- Definition of expected winnings
def expected_winnings (p6 p2to5 p1 w6 w2to5 l1 : ℝ) : ℝ :=
  (p6 * w6) + (p2to5 * w2to5) + (p1 * l1)

-- Theorem stating the expected winnings
theorem expected_winnings_correct :
  expected_winnings p6 p2to5 p1 w6 w2to5 l1 = 0.5 :=
by
  sorry

end expected_winnings_correct_l537_537161


namespace solve_for_A_l537_537378

variable (A C D x : ℝ)
variable (hC : C ≠ 0)

def f (x : ℝ) : ℝ := A * x - 3 * C^2
def g (x : ℝ) : ℝ := C * x + D

theorem solve_for_A (h : f g 2 = 0) : A = (3 * C^2) / (2 * C + D) :=
by
  sorry

end solve_for_A_l537_537378


namespace area_of_triangle_circumcircle_l537_537177

-- Defining the isosceles triangle ABC with AB = AC = 4√6
def ab := 4 * Real.sqrt 6
def ac := 4 * Real.sqrt 6

-- Defining the circle with radius 6√2 tangent to AB at B and AC at C
def radius := 6 * Real.sqrt 2

-- Function to compute the area of the circumcircle passing through A, B, and C
noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BC := Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos (Math.PI / 3)) -- Calculate BC using law of cosines
  let R := BC / (2 * Real.sin (Math.PI / 3)) -- Calculate circumradius using extended law of sines
  Real.pi * R^2 -- Area of the circle

theorem area_of_triangle_circumcircle :
  ∃ A B C : ℝ, A = ab ∧ B = ac ∧ area_of_circumcircle A B C = 48 * Real.pi :=
by
  sorry

end area_of_triangle_circumcircle_l537_537177


namespace expected_balls_in_original_positions_after_transpositions_l537_537413

theorem expected_balls_in_original_positions_after_transpositions :
  let num_balls := 7
  let first_swap_probability := 2 / 7
  let second_swap_probability := 1 / 7
  let third_swap_probability := 1 / 7
  let original_position_probability := (2 / 343) + (125 / 343)
  let expected_balls := num_balls * original_position_probability
  expected_balls = 889 / 343 := 
sorry

end expected_balls_in_original_positions_after_transpositions_l537_537413


namespace stickers_distributed_correctly_l537_537039

theorem stickers_distributed_correctly :
  ∀ (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) (leftover_stickers : ℕ) (total_students : ℕ), 
  total_stickers = 50 →
  friends = 5 →
  stickers_per_friend = 4 →
  leftover_stickers = 8 →
  total_students = 17 →
  let total_given_away := total_stickers - leftover_stickers in
  let friends_total := friends * stickers_per_friend in
  let other_students := total_students - 1 - friends in
  let other_given_away := total_given_away - friends_total in
  other_students > 0 → 
  other_given_away / nat.pred other_students = 2 :=
begin
  intros total_stickers friends stickers_per_friend leftover_stickers total_students ht hf hspc hl htst,
  let total_given_away := total_stickers - leftover_stickers,
  let friends_total := friends * stickers_per_friend,
  let other_students := total_students - 1 - friends,
  let other_given_away := total_given_away - friends_total,
  intro hp,
  rw [← nat.sub_sub, nat.succ_pred_eq_of_pos hp, nat.div_eq_of_lt],
  sorry
end

end stickers_distributed_correctly_l537_537039


namespace regular_polygon_sides_l537_537977

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537977


namespace number_of_ants_is_approximately_216_million_l537_537899

-- Given conditions
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 5
def inches_per_foot : ℕ := 12

-- Proof statement
theorem number_of_ants_is_approximately_216_million :
  let width_inches := width_feet * inches_per_foot in
  let length_inches := length_feet * inches_per_foot in
  let area_square_inches := width_inches * length_inches in
  let number_of_ants := ants_per_square_inch * area_square_inches in
  number_of_ants = 216000000 :=
by
  sorry

end number_of_ants_is_approximately_216_million_l537_537899


namespace find_m_l537_537085

noncomputable def m_does_not_have_solutions : ℝ :=
  let v1 := ⟨1, 3⟩
  let d1 := ⟨2, -3⟩
  let v2 := ⟨-1, 4⟩
  let d2 := λ (m : ℝ), ⟨3, m⟩
  -4.5

theorem find_m (m : ℝ) :
  (∀ t s : ℝ, v1 + t•d1 ≠ v2 + s•d2 m) ↔ m = m_does_not_have_solutions :=
by
  sorry

end find_m_l537_537085


namespace log_div_simplifies_l537_537851

theorem log_div_simplifies :
  (log (16 ^ (-2)) / log 16) = -2 :=
by
  sorry

end log_div_simplifies_l537_537851


namespace sheela_savings_l537_537051

theorem sheela_savings
  (I : ℝ)
  (h1 : 0.4 * I = 6000)
  (h2 : 0.3 * I)
  (h3 : 0.2 * I)
  : (I = 15000) ∧ ((I - ((0.3 * I) + (0.2 * I))) = 7500) := by
  sorry

end sheela_savings_l537_537051


namespace infinite_disconnected_prime_l537_537584

theorem infinite_disconnected_prime :
  ∃ᶠ (p : ℕ) in at_top, ∃ (k : ℕ → Prop),
    (∀ p, p.prime → (∃ m n : ℕ, 1 ≤ m ∧ m ≤ p ∧ 1 ≤ n ∧ n ≤ p ∧ 
    ¬ (p ∣ ((m^2 - n + 1) * (n^2 - m + 1)))) → 
    ¬ (∃ p : ℕ, (p.prime → ∃ m n : ℕ, 1 ≤ m ∧ m ≤ p ∧ 1 ≤ n ∧ n ≤ p ∧ 
    (p ∣ ((m^2 - n + 1) * (n^2 - m + 1))) → 
    ¬ (same_component p m n)))) := sorry

end infinite_disconnected_prime_l537_537584


namespace daily_earning_r_l537_537119

theorem daily_earning_r :
  exists P Q R : ℝ, 
    (P + Q + R = 220) ∧
    (P + R = 120) ∧
    (Q + R = 130) ∧
    (R = 30) := 
by
  sorry

end daily_earning_r_l537_537119


namespace find_multiplier_l537_537894

variable (x y m : ℝ)

def bigger_part := y = 30.333333333333332
def sum_parts := x + y = 52
def weighted_sum := m * x + 22 * y = 780

theorem find_multiplier (h1 : bigger_part) (h2 : sum_parts) (h3 : weighted_sum) : 
  m ≈ 5.1 := 
by
  sorry

end find_multiplier_l537_537894


namespace problem_solution_l537_537848

theorem problem_solution :
  (2200 - 2089)^2 / 196 = 63 :=
sorry

end problem_solution_l537_537848


namespace total_players_is_139_l537_537127

def num_kabadi := 60
def num_kho_kho := 90
def num_soccer := 40
def num_basketball := 70
def num_volleyball := 50
def num_badminton := 30

def num_k_kh := 25
def num_k_s := 15
def num_k_b := 13
def num_k_v := 20
def num_k_ba := 10
def num_kh_s := 35
def num_kh_b := 16
def num_kh_v := 30
def num_kh_ba := 12
def num_s_b := 20
def num_s_v := 18
def num_s_ba := 7
def num_b_v := 15
def num_b_ba := 8
def num_v_ba := 10

def num_k_kh_s := 5
def num_k_b_v := 4
def num_s_b_ba := 3
def num_v_ba_kh := 2

def num_all_sports := 1

noncomputable def total_players : Nat :=
  (num_kabadi + num_kho_kho + num_soccer + num_basketball + num_volleyball + num_badminton) 
  - (num_k_kh + num_k_s + num_k_b + num_k_v + num_k_ba + num_kh_s + num_kh_b + num_kh_v + num_kh_ba + num_s_b + num_s_v + num_s_ba + num_b_v + num_b_ba + num_v_ba)
  + (num_k_kh_s + num_k_b_v + num_s_b_ba + num_v_ba_kh)
  - num_all_sports

theorem total_players_is_139 : total_players = 139 := 
  by 
    sorry

end total_players_is_139_l537_537127


namespace second_carpenter_days_to_complete_job_l537_537132

-- Definitions
def job := 1
def first_carpenter_days_to_complete := 8
def combined_work_days := 4

-- First carpenter's work rate
def r₁ := 1 / first_carpenter_days_to_complete

-- Equation involving both carpenters' work rates when combined work equals one job completed in 4 days
def combined_work_rate (x : ℝ) := r₁ + (1 / x)

-- Proof statement: find x such that combined work rate equals the required combined work rate
theorem second_carpenter_days_to_complete_job :
  ∃ x : ℝ, combined_work_rate x = 1 / combined_work_days ∧ x = 8 :=
by
  sorry

end second_carpenter_days_to_complete_job_l537_537132


namespace gray_area_l537_537890

noncomputable def radius_smaller_circle : ℝ := 6 / 2
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle
noncomputable def area_circle (radius : ℝ) : ℝ := Real.pi * radius ^ 2

theorem gray_area :
  let gray_area := area_circle radius_larger_circle - area_circle radius_smaller_circle in
  gray_area = 72 * Real.pi := by
  sorry

end gray_area_l537_537890


namespace eighth_odd_multiple_of_5_is_75_l537_537484

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l537_537484


namespace even_comp_even_l537_537726

/-- 
Let f be a function such that f(x) is an even function.
Prove that f(f(x)) is an even function.
-/
theorem even_comp_even {f : ℝ → ℝ} (h : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f (f (-x)) = f (f x) :=
by
  intros x
  have h_even := h (f x)
  rw h 
  rw h
  exact h_even

end even_comp_even_l537_537726


namespace find_f_99_l537_537633

noncomputable def f : ℤ → ℝ

axiom f_periodic : ∀ x : ℤ, f x * f (x + 2) = 1
axiom f_at_1 : f 1 = 2

theorem find_f_99 : f 99 = 1 / 2 :=
by
  sorry

end find_f_99_l537_537633


namespace no_perfect_square_from_1986_nums_l537_537469

theorem no_perfect_square_from_1986_nums :
  let S := (1986 * 1987) / 2 in
  (S % 9 = 3) →
  ∀ (A : ℕ), (A ≡ S % 9 [MOD 9]) → ¬ ∃ k, k * k = A := 
by
  sorry

end no_perfect_square_from_1986_nums_l537_537469


namespace five_digit_even_numbers_l537_537840

theorem five_digit_even_numbers (digits : set ℕ) (n : ℕ) : 
  digits = {0, 1, 2, 3, 4, 5} →
  20000 < n →
  (∀ d1 d2 : ℕ, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 → n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) →
  (∀ d : ℕ, d ∈ digits → 0 ≤ d ∧ d < 10) →
  (∃ count : ℕ, count = 240) :=
by
  -- The proof should be written below this line.
  sorry

end five_digit_even_numbers_l537_537840


namespace convex_polyhedron_in_inscribed_sphere_l537_537349

-- Definitions based on conditions
variables (S c r : ℝ) (S' V R : ℝ)

-- The given relationship for a convex polygon.
def poly_relationship := S = (1 / 2) * c * r

-- The desired relationship for a convex polyhedron.
def polyhedron_relationship := V = (1 / 3) * S' * R

-- Proof statement
theorem convex_polyhedron_in_inscribed_sphere (S c r S' V R : ℝ) 
  (poly : S = (1 / 2) * c * r) : V = (1 / 3) * S' * R :=
sorry

end convex_polyhedron_in_inscribed_sphere_l537_537349


namespace min_value_ineq_l537_537643

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem min_value_ineq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end min_value_ineq_l537_537643


namespace no_such_cuboid_exists_l537_537360

theorem no_such_cuboid_exists (a b c : ℝ) :
  a + b + c = 12 ∧ ab + bc + ca = 1 ∧ abc = 12 → false :=
by
  sorry

end no_such_cuboid_exists_l537_537360


namespace minimum_radius_of_third_sphere_l537_537808

noncomputable def cone_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 3

noncomputable def radius_identical_spheres : ℝ := 4 / 3  -- derived from the conditions

theorem minimum_radius_of_third_sphere
    (h r1 r2 : ℝ) -- heights and radii one and two
    (R1 R2 Rb : ℝ) -- radii of the common base
    (cond_h : h = 4)
    (cond_Rb : Rb = 3)
    (cond_radii_eq : r1 = r2) 
  : r2 = 27 / 35 :=
by
  sorry

end minimum_radius_of_third_sphere_l537_537808


namespace exists_line_perpendicular_to_skew_lines_l537_537625

-- Define the conditions where a and b are skew lines
variables {a b : Type} [Line a] [Line b]

-- Define a point P in 3D space
variables {P : Point3D}

-- Define what it means for lines to be skew
def skew_lines (a b : Line) : Prop :=
  ¬ (∃ (P : Point3D), P ∈ a ∧ P ∈ b) ∧
  ¬ (∃ (l : Line), l ∥ a ∧ l ∥ b)

-- The proof statement: for any point P in space, there exists a line passing through P that is perpendicular to both a and b.
theorem exists_line_perpendicular_to_skew_lines (h_skew : skew_lines a b) :
  ∃! l : Line, (P ∈ l) ∧ (l ⊥ a) ∧ (l ⊥ b) :=
sorry

end exists_line_perpendicular_to_skew_lines_l537_537625


namespace bluegrass_percentage_l537_537412

-- Define the problem conditions
def seed_mixture_X_ryegrass_percentage : ℝ := 40
def seed_mixture_Y_ryegrass_percentage : ℝ := 25
def seed_mixture_Y_fescue_percentage : ℝ := 75
def mixture_X_Y_ryegrass_percentage : ℝ := 30
def mixture_weight_percentage_X : ℝ := 33.33333333333333

-- Prove that the percentage of bluegrass in seed mixture X is 60%
theorem bluegrass_percentage (X_ryegrass : ℝ) (Y_ryegrass : ℝ) (Y_fescue : ℝ) (mixture_ryegrass : ℝ) (weight_percentage_X : ℝ) :
  X_ryegrass = seed_mixture_X_ryegrass_percentage →
  Y_ryegrass = seed_mixture_Y_ryegrass_percentage →
  Y_fescue = seed_mixture_Y_fescue_percentage →
  mixture_ryegrass = mixture_X_Y_ryegrass_percentage →
  weight_percentage_X = mixture_weight_percentage_X →
  (100 - X_ryegrass) = 60 :=
by
  intro hX_ryegrass hY_ryegrass hY_fescue hmixture_ryegrass hweight_X
  rw [hX_ryegrass]
  sorry

end bluegrass_percentage_l537_537412


namespace kody_half_mohamed_years_ago_l537_537776

-- Definitions of initial conditions
def current_age_mohamed : ℕ := 2 * 30
def current_age_kody : ℕ := 32

-- Proof statement
theorem kody_half_mohamed_years_ago : ∃ x : ℕ, (current_age_kody - x) = (1 / 2 : ℕ) * (current_age_mohamed - x) ∧ x = 4 := 
by 
  sorry

end kody_half_mohamed_years_ago_l537_537776


namespace function_value_bounds_l537_537416

theorem function_value_bounds (x : ℝ) : 
  (x^2 + x + 1) / (x^2 + 1) ≤ 3 / 2 ∧ (x^2 + x + 1) / (x^2 + 1) ≥ 1 / 2 := 
sorry

end function_value_bounds_l537_537416


namespace cover_points_with_circles_l537_537758

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ) :
  ∃ (radii : Fin n → ℝ) (centers : Fin n → ℝ × ℝ),
  (∀ i j, i ≠ j → (dist (centers i) (centers j) > radii i + radii j + 1)) ∧
  (∑ i, 2 * radii i < n) ∧
  (∀ i, ∃ j, points i = centers j) :=
by
  sorry

end cover_points_with_circles_l537_537758


namespace cos_conditions_implies_obtuse_triangle_l537_537323

theorem cos_conditions_implies_obtuse_triangle
  {A B C : ℝ}
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (triangle_inequality : A + B + C = 180)
  (cos_condition : cos A * cos B > sin A * sin B) : 
  90 < C := 
by
 sorry

end cos_conditions_implies_obtuse_triangle_l537_537323


namespace cos_60_degrees_is_one_half_l537_537936

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537936


namespace equal_distances_l537_537552

-- Given definitions
variables (A B C D M K L S : Point)
variable [triangle : Triangle ABC]
variable [altitude : AltitudeFrom C D]
variable [midpoint : Midpoint M A B]
variables [KA : LineThrough M A]
variable [KB : LineThrough M B]
variable [circumcenter : Circumcenter S C K L]
variable [equality : CK = CL]

-- To Prove
theorem equal_distances (S D M K L : Point) [Circumcenter S L K M D] : S D = S M := sorry

#print equal_distances

end equal_distances_l537_537552


namespace coefficient_of_x_cube_l537_537343

-- Definitions used in Lean 4 statement
def binomial_term (n : ℕ) (r : ℕ) (a b : ℚ) := 
  (Nat.choose n r) * a^(n-r) * b^r

def expansion_term (x : ℚ) (r : ℕ) := 
  binomial_term 10 r (x^(-(1/4))) (x^(2/3))

-- Lean 4 statement
theorem coefficient_of_x_cube :
  (Nat.choose 10 2 = 45) →
  ∃ c : ℚ, expansion_term x 6 = c * x^3 ∧ c = 210 :=
by
  sorry

end coefficient_of_x_cube_l537_537343


namespace simplify_f_value_of_f_neg_1860_deg_value_of_f_given_conditions_l537_537594

-- Problem 1: Simplify f(α) to -cos(α)
theorem simplify_f (α : ℝ) : 
  f(α) = -cos(α) := sorry

-- Problem 2: Prove the value of f(-1860°)
theorem value_of_f_neg_1860_deg : 
  f(-1860 * π / 180) = -1 / 2 := sorry

-- Problem 3: Given α in (0, π/2) and sin(α - π/6) = 1/3, find value of f(α)
theorem value_of_f_given_conditions (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin(α - π / 6) = 1 / 3) : 
  f(α) = (1 - 2 * sqrt(6)) / 6 := sorry

end simplify_f_value_of_f_neg_1860_deg_value_of_f_given_conditions_l537_537594


namespace neither_outstanding_nor_young_pioneers_is_15_l537_537454

-- Define the conditions
def total_students : ℕ := 87
def outstanding_students : ℕ := 58
def young_pioneers : ℕ := 63
def both_outstanding_and_young_pioneers : ℕ := 49

-- Define the function to calculate the number of students who are neither
def neither_outstanding_nor_young_pioneers
: ℕ :=
total_students - (outstanding_students - both_outstanding_and_young_pioneers) - (young_pioneers - both_outstanding_and_young_pioneers) - both_outstanding_and_young_pioneers

-- The theorem to prove
theorem neither_outstanding_nor_young_pioneers_is_15
: neither_outstanding_nor_young_pioneers = 15 :=
by
  sorry

end neither_outstanding_nor_young_pioneers_is_15_l537_537454


namespace isosceles_right_triangle_area_l537_537781

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = Real.sqrt 8) : 
  let leg := h
  (2 * leg * leg) * 0.5 = 4 :=
begin
  -- use the given condition h = sqrt(8)
  rw h_eq,
  -- calculate the leg
  let leg := Real.sqrt 8,
  have leg_eq : leg = 2 * Real.sqrt 2,
  { rw Real.sqrt_mul, rw (Real.sqrt_eq_iff_sqrt_eq _).1 rfl, norm_num,
    rw Real.sqrt_mul_self_eq_abs,
    rw abs_eq_self.mpr (le_of_lt Real.sqrt_r_pos),
    all_goals { exact (Real.sqrt_pos.2 (by norm_num)).ne' } },
  -- calculate the area
  calc
    (2 * leg * leg) * 0.5 = _ : by sorry
end

end isosceles_right_triangle_area_l537_537781


namespace quadratic_solution_l537_537678

def quadratic_ineq (a x : ℝ) : Prop :=
  a * x^2 + (a - 2) * x - 2 > 0

theorem quadratic_solution :
  (∃ a : ℝ, a ∈ set.Icc (1 : ℝ) 3 ∧ quadratic_ineq a x) → 
  (x ∈ set.Iio (-1) ∪ set.Ioi (2/3)) :=
sorry

end quadratic_solution_l537_537678


namespace sphere_center_result_l537_537715

noncomputable def A := (2 : ℝ, 0 : ℝ, 0 : ℝ)
noncomputable def B := (0 : ℝ, 4 : ℝ, 0 : ℝ)
noncomputable def C := (0 : ℝ, 0 : ℝ, 6 : ℝ)
noncomputable def O := (0 : ℝ, 0 : ℝ, 0 : ℝ)

theorem sphere_center_result : 
  ∃ (p q r : ℝ), 
  (p, q, r) = sphere_center {A, B, C, O} ∧ 
  (1 / p) + (1 / q) + (1 / r) = 49 / 72 := 
sorry

end sphere_center_result_l537_537715


namespace average_mileage_on_highway_l537_537548

-- Define the given conditions
def CityMileage := 7.6
def TotalDistance := 292.8
def TotalGallons := 24

-- State the proof goal
theorem average_mileage_on_highway : (TotalDistance / TotalGallons) = 12.2 :=
by
  -- proof will go here
  sorry

end average_mileage_on_highway_l537_537548


namespace family_members_count_l537_537495

-- Defining the conditions given in the problem
variables (cyrus_bites_arms_legs : ℕ) (cyrus_bites_body : ℕ) (total_bites_family : ℕ)
variables (family_bites_per_person : ℕ) (cyrus_total_bites : ℕ)

-- Given conditions
def condition1 : cyrus_bites_arms_legs = 14 := sorry
def condition2 : cyrus_bites_body = 10 := sorry
def condition3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body := sorry
def condition4 : total_bites_family = cyrus_total_bites / 2 := sorry
def condition5 : ∀ n : ℕ, total_bites_family = n * family_bites_per_person := sorry

-- The theorem to prove: The number of people in the rest of Cyrus' family is 12
theorem family_members_count (n : ℕ) (h1 : cyrus_bites_arms_legs = 14)
    (h2 : cyrus_bites_body = 10) (h3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body)
    (h4 : total_bites_family = cyrus_total_bites / 2)
    (h5 : ∀ n, total_bites_family = n * family_bites_per_person) : n = 12 :=
sorry

end family_members_count_l537_537495


namespace maximum_value_S_l537_537730

noncomputable def max_value_of_S : Real := sorry

theorem maximum_value_S :
  (∃ a : Fin 2018 → ℝ, 
    (∀ i, 0 ≤ a i) ∧ 
    (∑ i, a i = 1) ∧ 
    (max_value_of_S = 
      ∑ i in Finset.finRange 2018, ∑ j in Finset.finRange 2018, if i ≠ j ∧ i ∣ j then a i * a j else 0)) =
    (max_value_of_S = 5 / 11) := 
sorry

end maximum_value_S_l537_537730


namespace arch_height_at_10_feet_from_center_l537_537530

noncomputable def parabolic_height (a x k : ℝ) : ℝ := a * x ^ 2 + k

theorem arch_height_at_10_feet_from_center :
  ∃ (a : ℝ), let k := 20 in
  let height_at_10 := parabolic_height a 10 k in
  (parabolic_height a 25 k = 0) → (height_at_10 = 16.8) :=
begin
  sorry
end

end arch_height_at_10_feet_from_center_l537_537530


namespace main_theorem_l537_537676

-- Define the complex number z and the imaginary unit i
def z := Complex
def i := Complex.I

-- State the condition
def condition (z : Complex) : Prop :=
  z = (z + 1) * i

-- Define the conjugate of z
def conjugate (z : Complex) : Complex :=
  Complex.conj z

-- Define the point corresponding to the conjugate of z
def point (z : Complex) : ℂ × ℂ :=
  (conjugate z).re, (conjugate z).im

-- Define the quadrant function (for a visual quadrant, you could use a different data type)
def quadrant (p : ℂ × ℂ) : ℕ :=
  if p.1 > 0 ∧ p.2 > 0 then 1
  else if p.1 < 0 ∧ p.2 > 0 then 2
  else if p.1 < 0 ∧ p.2 < 0 then 3
  else if p.1 > 0 ∧ p.2 < 0 then 4
  else 0  -- this could represent a point on the axis

-- The main theorem
theorem main_theorem (z : Complex) (h : condition z) : quadrant (point z) = 3 :=
sorry

end main_theorem_l537_537676


namespace calculate_expression_l537_537191

theorem calculate_expression : -1 ^ 4 + 16 / (-2) ^ 3 * | -3 - 1 | = -9 := 
by 
  sorry

end calculate_expression_l537_537191


namespace no_equal_edge_sum_cube_labeling_l537_537359

theorem no_equal_edge_sum_cube_labeling :
  ¬ ∃ (f : Fin 12 → ℕ), (∀ v : Fin 8, (∑ e in {e | adj v e}, f e) = 19) :=
by
  sorry

end no_equal_edge_sum_cube_labeling_l537_537359


namespace perfect_square_representation_l537_537510

theorem perfect_square_representation :
  29 - 12*Real.sqrt 5 = (2*Real.sqrt 5 - 3*Real.sqrt 5 / 5)^2 :=
sorry

end perfect_square_representation_l537_537510


namespace find_principal_amount_l537_537919

-- Define the parameters
def R : ℝ := 11.67
def T : ℝ := 5
def A : ℝ := 950

-- State the theorem
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + (R/100) * T) :=
by { 
  use 600, 
  -- Skip the proof 
  sorry 
}

end find_principal_amount_l537_537919


namespace last_person_standing_l537_537777

theorem last_person_standing (initial_positions : list string) (elimination_rule : ℕ → bool) :
  ∃ last_person : string, 
  let circle := initial_positions
  let elimination_index_1 := (5 % 4)
  let circle_1 := circle.remove_nth elimination_index_1

  let elimination_index_2 := (10 % 3)
  let circle_2 := circle_1.remove_nth elimination_index_2

  let elimination_index_3 := (15 % 2)
  let circle_3 := circle_2.remove_nth elimination_index_3
        
  circle_3.head = "Dave" := 
begin
  let initial_positions := ["Alice", "Ben", "Carl", "Dave"],
  let elimination_rule := λ n, n % 5 = 0 ∨ list.elem '5' (to_digits 10 n),
  use "Dave",
  sorry
end

end last_person_standing_l537_537777


namespace neg_pi_lt_neg_three_l537_537562

theorem neg_pi_lt_neg_three (h : Real.pi > 3) : -Real.pi < -3 :=
sorry

end neg_pi_lt_neg_three_l537_537562


namespace domain_f_l537_537567

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 - 3 * x - 10)

theorem domain_f :
  {x : ℝ | x^2 - 3 * x - 10 ≠ 0} = set.Ioo (5 : ℝ) ∞ ∪ set.Ioo ⊥ (-2) := 
begin
  sorry
end

end domain_f_l537_537567


namespace relationship_S_T_l537_537654

-- Definitions based on the given conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n

def seq_b (n : ℕ) : ℕ :=
  2 ^ (n - 1) + 1

def S (n : ℕ) : ℕ :=
  (n * (n + 1))

def T (n : ℕ) : ℕ :=
  (2^n) + n - 1

-- The conjecture and proofs
theorem relationship_S_T (n : ℕ) : 
  if n = 1 then T n = S n
  else if (2 ≤ n ∧ n < 5) then T n < S n
  else n ≥ 5 → T n > S n :=
by sorry

end relationship_S_T_l537_537654


namespace angle_RS_CA_48_l537_537324

-- Definitions of points, angles, midpoints, and lines
variables {T C A V H R S : Type*}
variables [triangle T C A]
variables (CA : T → ℝ) (angleC : T → ℝ) (angleA : T → ℝ)
variables (CV VH: T → ℝ) 
variables (midpoint_CA midpoint_VH R S: T → Prop)

-- Given conditions
def conditions := 
  (angleC C A T = 48) ∧
  (angleA C A T = 58) ∧
  (CA C A T = 12) ∧
  (CV C A T = 1) ∧
  (VH C A T = 1) ∧
  (midpoint_CA R CA) ∧
  (midpoint_VH S VH)

-- Theorem to prove the degree measure of the acute angle is 48
theorem angle_RS_CA_48 : 
  conditions T C A V H CA angleC angleA CV VH midpoint_CA midpoint_VH R S →
  acute_angle (line R S) (line C A) = 48 := 
by
  sorry

end angle_RS_CA_48_l537_537324


namespace stratified_sampling_young_employees_l537_537971

-- Given conditions
def total_young : Nat := 350
def total_middle_aged : Nat := 500
def total_elderly : Nat := 150
def total_employees : Nat := total_young + total_middle_aged + total_elderly
def representatives_to_select : Nat := 20
def sampling_ratio : Rat := representatives_to_select / (total_employees : Rat)

-- Proof goal
theorem stratified_sampling_young_employees :
  (total_young : Rat) * sampling_ratio = 7 := 
by
  sorry

end stratified_sampling_young_employees_l537_537971


namespace equation_of_line_parallel_to_x_axis_l537_537440

theorem equation_of_line_parallel_to_x_axis (x: ℝ) :
  ∃ (y: ℝ), (y-2=0) ∧ ∀ (P: ℝ × ℝ), (P = (1, 2)) → P.2 = 2 := 
by
  sorry

end equation_of_line_parallel_to_x_axis_l537_537440


namespace find_numbers_between_1000_and_4000_l537_537232

theorem find_numbers_between_1000_and_4000 :
  ∃ (x : ℤ), 1000 ≤ x ∧ x ≤ 4000 ∧
             (x % 11 = 2) ∧
             (x % 13 = 12) ∧
             (x % 19 = 18) ∧
             (x = 1234 ∨ x = 3951) :=
sorry

end find_numbers_between_1000_and_4000_l537_537232


namespace erwan_spending_l537_537214

def discount (price : ℕ) (percent : ℕ) : ℕ :=
  price - (price * percent / 100)

theorem erwan_spending (shoe_original_price : ℕ := 200) 
  (shoe_discount : ℕ := 30)
  (shirt_price : ℕ := 80)
  (num_shirts : ℕ := 2)
  (pants_price : ℕ := 150)
  (second_store_discount : ℕ := 20)
  (jacket_price : ℕ := 250)
  (tie_price : ℕ := 40)
  (hat_price : ℕ := 60)
  (watch_price : ℕ := 120)
  (wallet_price : ℕ := 49)
  (belt_price : ℕ := 35)
  (belt_discount : ℕ := 25)
  (scarf_price : ℕ := 45)
  (scarf_discount : ℕ := 10)
  (rewards_points_discount : ℕ := 5)
  (sales_tax : ℕ := 8)
  (gift_card : ℕ := 50)
  (shipping_fee : ℕ := 5)
  (num_shipping_stores : ℕ := 2) :
  ∃ total : ℕ,
    total = 85429 :=
by
  have first_store := discount shoe_original_price shoe_discount
  have second_store_total := pants_price + (shirt_price * num_shirts)
  have second_store := discount second_store_total second_store_discount
  have tie_half_price := tie_price / 2
  have hat_half_price := hat_price / 2
  have third_store := jacket_price + (tie_half_price + hat_half_price)
  have fourth_store := watch_price
  have fifth_store := discount belt_price belt_discount + discount scarf_price scarf_discount
  have subtotal := first_store + second_store + third_store + fourth_store + fifth_store
  have after_rewards_points := subtotal - (subtotal * rewards_points_discount / 100)
  have after_gift_card := after_rewards_points - gift_card
  have after_shipping_fees := after_gift_card + (shipping_fee * num_shipping_stores)
  have total := after_shipping_fees + (after_shipping_fees * sales_tax / 100)
  use total / 100 -- to match the monetary value in cents
  sorry

end erwan_spending_l537_537214


namespace calculation_result_l537_537196

theorem calculation_result :
  3 * 3^3 + 4^7 / 4^5 = 97 :=
by
  sorry

end calculation_result_l537_537196


namespace consumer_installment_credit_l537_537920

theorem consumer_installment_credit (A C : ℝ) (h1 : A = 0.36 * C) (h2 : 35 = (1 / 3) * A) :
  C = 291.67 :=
by 
  -- The proof should go here
  sorry

end consumer_installment_credit_l537_537920


namespace find_lambda_l537_537299

-- Definitions for non-zero vectors and perpendicularity
variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b : V)

-- Conditions
def are_perpendicular (a b : V) : Prop := inner a b = 0
def is_nonzero (v : V) : Prop := v ≠ 0
def are_collinear (a b : V) : Prop := ∃ μ : ℝ, b = μ • a

-- Given conditions
variables (h1 : are_perpendicular a b) (h2 : is_nonzero a) (h3 : is_nonzero b)
variables (m : V) (n : V)
def m_def : V := 4 • a + 5 • b
def n_def : V := 2 • a + λ • b
def collinear_condition : are_collinear m n := ∃ μ : ℝ, n = μ • m

-- Theorem to be proved
theorem find_lambda (h_collinear : collinear_condition) : λ = 2.5 :=
sorry

end find_lambda_l537_537299


namespace right_angled_triangle_sides_l537_537250

theorem right_angled_triangle_sides :
  ∃ a b c : ℝ, (a^2 + b^2 = c^2) ∧ ((1/2) * a * b = 7) ∧ (a + b + c = 14) ∧ (a = 4 - Real.sqrt 2) ∧ (b = 4 + Real.sqrt 2) ∧ (c = 6) :=
begin
  sorry
end

end right_angled_triangle_sides_l537_537250


namespace range_of_a_l537_537260

variable (p : Prop) (x a : ℝ)

-- The given conditions
axiom cond1 : 1 / 2 ≤ x ∧ x ≤ 1
axiom cond2 : (x - a) * (x - a - 1) ≤ 0
axiom cond3 : ¬p → ¬\ ( (x - a) * (x - a - 1) ≤ 0)

-- The proof statement
theorem range_of_a : 0 ≤ a ∧ a ≤ 1 / 2 := 
  by
    sorry

end range_of_a_l537_537260


namespace probability_A_B_same_group_l537_537570

-- Variables to represent the people
axiom A : Type
axiom B : Type
axiom C : Type
axiom D : Type

-- Definitions and assumptions
def people := {A, B, C, D}
def groups := Set.Set (Set.Set people)

-- Defining the conditions of the problem
def validGroups : groups :=
  { S | S ∈ groups ∧ (∃ S1 S2, S = {S1, S2} ∧ S1 ∪ S2 = people ∧ S1 ∩ S2 = ∅ ∧ (S1.card = 2 ∧ S2.card = 2 ∨ S1.card = 1 ∧ S2.card = 3)) }

-- Main theorem statement
theorem probability_A_B_same_group :
  (∃ S ∈ validGroups, A ∈ S1 ∧ B ∈ S1) →
  (∃ S ∈ validGroups, A ∈ S1 ∧ B ∈ S1) / (∃ S ∈ validGroups, A ∈ S1) = 5 / 6 :=
by
  sorry

end probability_A_B_same_group_l537_537570


namespace fraction_calculation_l537_537234

-- Define the initial values of x and y
def x : ℚ := 4 / 6
def y : ℚ := 8 / 10

-- Statement to prove
theorem fraction_calculation : (6 * x^2 + 10 * y) / (60 * x * y) = 11 / 36 := by
  sorry

end fraction_calculation_l537_537234


namespace car_total_distance_l537_537878

def arithmetic_sequence_distance (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem car_total_distance : 
  let a := 40 in let d := -10 in let n := 5 in
  (arithmetic_sequence_distance a d n) = 100 :=
by
  sorry

end car_total_distance_l537_537878


namespace platform_length_l537_537540

theorem platform_length (train_length : ℕ) (speed_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : train_length = 360) 
  (h2 : speed_kmph = 45) 
  (h3 : time_seconds = 40.8) : 
  ∃ (platform_length : ℕ), platform_length = 150 := 
by
  sorry

end platform_length_l537_537540


namespace coin_toss_sequences_l537_537329

theorem coin_toss_sequences :
  ∃ (seqs : Finset (List Char)), seqs.card = 3920 ∧
  (∀ seq ∈ seqs, 
    has_subseq_count seq ['H', 'H'] 3 ∧
    has_subseq_count seq ['H', 'T'] 3 ∧
    has_subseq_count seq ['T', 'H'] 5 ∧
    has_subseq_count seq ['T', 'T'] 4) :=
sorry

end coin_toss_sequences_l537_537329


namespace circle_radius_5_l537_537569

theorem circle_radius_5 (k x y : ℝ) : x^2 + 8 * x + y^2 + 10 * y - k = 0 → (x + 4) ^ 2 + (y + 5) ^ 2 = 25 → k = -16 :=
by
  sorry

end circle_radius_5_l537_537569


namespace problem_part_1_problem_part_2_problem_part_3_l537_537882

noncomputable def f (n : ℕ) := (n * (n + 10)) / 10 + 16.9

def maintenance_cost (n : ℕ) : ℕ := 2000 * n
def insurance_cost (n : ℕ) : ℕ := 9000 * n
def initial_cost : ℕ := 169000

def total_cost (n : ℕ) :=
initial_cost + insurance_cost(n) + ∑ i in finset.range(n), maintenance_cost(i + 1)

theorem problem_part_1 :
total_cost 3 = 20800000 :=
sorry

theorem problem_part_2 (n : ℕ) :
f n = initial_cost / 10000 + (insurance_cost n / 10000) + (200 * n * (n + 1) / 2 / 10000) :=
sorry

theorem problem_part_3 :
(∀ n m : ℕ, (n > 0) → (m > 0) → n ≠ 13 → (f(n) / n * 1.0 ≤ f(m) / m * 1.0)) :=
sorry


end problem_part_1_problem_part_2_problem_part_3_l537_537882


namespace potato_difference_l537_537036

def x := 8 * 13
def k := (67 - 13) / 2
def z := 20 * k
def d := z - x

theorem potato_difference : d = 436 :=
by
  sorry

end potato_difference_l537_537036


namespace diamond_evaluation_l537_537965

def diamond (A B : ℝ) : ℝ := (A^2 + B^2) / 5

theorem diamond_evaluation : diamond (diamond 3 7) 4 = 30.112 := by
  sorry

end diamond_evaluation_l537_537965


namespace cost_price_percentage_l537_537320

variable (SP CP : ℝ)

-- Assumption that the profit percent is 25%
axiom profit_percent : 25 = ((SP - CP) / CP) * 100

-- The statement to prove
theorem cost_price_percentage : CP / SP = 0.8 := by
  sorry

end cost_price_percentage_l537_537320


namespace probability_of_at_least_one_black_ball_l537_537876

noncomputable def probability_at_least_one_black_ball := 
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := (Nat.choose 2 1) * (Nat.choose 2 1) + (Nat.choose 2 2)
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_black_ball :
  probability_at_least_one_black_ball = 5 / 6 :=
by
  sorry

end probability_of_at_least_one_black_ball_l537_537876


namespace magic_square_consistency_l537_537689

variable (S : ℝ)
variable (x d e f g h : ℝ)
variable (top_left top_middle top_right middle_left middle_middle middle_right bottom_left bottom_middle bottom_right : ℝ)

/-- In a 3x3 magic square, the sum of numbers in any row, column, or diagonal is the same.
    Prove that x can be any real number maintaining consistent sums given the initial values. -/
theorem magic_square_consistency :
  (∀ (a b c : ℝ), a + b + c = S) →
  (top_middle = 35) →
  (top_right = 58) →
  (middle_left = 8) →
  (top_left + top_middle + top_right = S) →
  (top_left + middle_left + bottom_left = S) →
  (top_left = x) →
  (top_left + middle_middle + bottom_right = S) →
  (f = 85 - d) →
  (27 = e + h) →
  ∃ (x : ℝ), true :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  use x
  exact true.intro

end magic_square_consistency_l537_537689


namespace geometric_sequence_sum_inv_l537_537645

theorem geometric_sequence_sum_inv
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 :=
by
  sorry

end geometric_sequence_sum_inv_l537_537645


namespace min_distance_between_inverse_exponential_and_logarithm_curves_l537_537270

noncomputable def min_distance_is_sqrt_2 : ℝ :=
  let P := λ x : ℝ, (x, Real.exp x)
  let Q := λ x : ℝ, (Real.log x, x)
  let dist := λ (p q : ℝ × ℝ), Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  Real.sqrt 2

theorem min_distance_between_inverse_exponential_and_logarithm_curves :
  ∃ P Q : ℝ × ℝ, (P.2 = Real.exp P.1) ∧ (Q.2 = Real.log Q.1) ∧
  (dist P Q = min_distance_is_sqrt_2) :=
sorry

end min_distance_between_inverse_exponential_and_logarithm_curves_l537_537270


namespace parts_purchased_l537_537175

noncomputable def price_per_part : ℕ := 80
noncomputable def total_paid_after_discount : ℕ := 439
noncomputable def total_discount : ℕ := 121

theorem parts_purchased : 
  ∃ n : ℕ, price_per_part * n - total_discount = total_paid_after_discount → n = 7 :=
by
  sorry

end parts_purchased_l537_537175


namespace projection_correct_l537_537233

/-
  Define the given vectors a and b.
  Vector a is the one being projected.
  Vector b is the direction vector of the line.
-/

def vector_a : ℝ × ℝ × ℝ := (4, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (3, -2, 1)

/-
  Define a function to calculate the dot product of two vectors.
-/
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

/-
  Define the projection of vector a onto vector b.
-/
def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := (dot_product a b) / (dot_product b b)
  (scalar * b.1, scalar * b.2, scalar * b.3)

/-
  Prove that the projection of vector_a onto vector_b is (3/2, -1, 1/2).
-/
theorem projection_correct :
  projection vector_a vector_b = (3/2 : ℝ, -1 : ℝ, 1/2 : ℝ) :=
by
  sorry

end projection_correct_l537_537233


namespace max_squared_sum_of_sides_l537_537171

variable {R : ℝ}
variable {O A B C : EucSpace} -- O is the center, A, B, and C are vertices
variable (a b c : ℝ)  -- Position vectors corresponding to vertices A, B, C

-- Hypotheses based on the problem conditions:
variable (h1 : ‖a‖ = R)
variable (h2 : ‖b‖ = R)
variable (h3 : ‖c‖ = R)
variable (hSumZero : a + b + c = 0)

theorem max_squared_sum_of_sides 
  {AB BC CA : ℝ} -- Side lengths
  (hAB : AB = ‖a - b‖)
  (hBC : BC = ‖b - c‖)
  (hCA : CA = ‖c - a‖) :
  AB^2 + BC^2 + CA^2 = 9 * R^2 :=
sorry

end max_squared_sum_of_sides_l537_537171


namespace caffeine_consumption_over_goal_l537_537395

def caffeine_content (bev : String) : ℕ :=
  match bev with
  | "coffee" => 95
  | "soda" => 45
  | "tea" => 55
  | "energy_drink" => 120
  | _ => 0

def beverages_consumed : List (String × ℕ) :=
  [ ("coffee", 3),
    ("soda", 1),
    ("tea", 2),
    ("energy_drink", 1) ]

def caffeine_goal : ℕ := 200

theorem caffeine_consumption_over_goal :
  let total_caffeine := List.sum (beverages_consumed.map (λ (b : (String × ℕ)), (caffeine_content b.1) * b.2))
  total_caffeine = 560 ∧ (total_caffeine - caffeine_goal) = 360 :=
by sorry

end caffeine_consumption_over_goal_l537_537395


namespace original_amount_of_cooking_oil_l537_537042

theorem original_amount_of_cooking_oil (X : ℝ) (H : (2 / 5 * X + 300) + (1 / 2 * (X - (2 / 5 * X + 300)) - 200) + 800 = X) : X = 2500 :=
by simp at H; linarith

end original_amount_of_cooking_oil_l537_537042


namespace total_length_of_set_T_l537_537387

def is_in_set_T (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2

noncomputable def total_length_set_T : ℝ := 16 * Real.sqrt 2

theorem total_length_of_set_T :
  (∑ x y, if is_in_set_T x y then 1 else 0) = total_length_set_T :=
sorry

end total_length_of_set_T_l537_537387


namespace simplify_polynomial_l537_537421

theorem simplify_polynomial (x : ℤ) :
  (3 * x - 2) * (6 * x^12 + 3 * x^11 + 5 * x^9 + x^8 + 7 * x^7) =
  18 * x^13 - 3 * x^12 + 15 * x^10 - 7 * x^9 + 19 * x^8 - 14 * x^7 :=
by
  sorry

end simplify_polynomial_l537_537421


namespace class_avg_difference_l537_537157

theorem class_avg_difference :
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (class_sizes.sum : ℝ) / (teachers : ℝ)
  let s := (class_sizes.map (λ n, n * (n / students : ℝ))).sum
  in 
  t - s = -13.5 :=
by
  -- Definitions
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (class_sizes.sum : ℝ) / (teachers : ℝ)
  let s := (class_sizes.map (λ n, n * (n / students : ℝ))).sum
  -- Equation to prove
  show t - s = -13.5,
  sorry

end class_avg_difference_l537_537157


namespace smallest_prime_not_diff_pow_2_3_l537_537582

theorem smallest_prime_not_diff_pow_2_3 (p : ℕ) (prime : nat.prime p) :
  (∀ x y : ℕ, p ≠ 2^x - 3^y ∧ p ≠ 3^y - 2^x) ↔ p = 41 :=
by {
  sorry
}

end smallest_prime_not_diff_pow_2_3_l537_537582


namespace verify_lines_planes_statements_l537_537206

-- Define the conditions and the problem
def lines_planes_statements_correct : Prop :=
  let m, n, l : Type := sorry; -- Placeholder for Line type
  let alpha, beta, gamma : Type := sorry; -- Placeholder for Plane type
  
  let statement_1 : Prop := (parallel m l) ∧ (parallel n l) → (parallel m n)
  let statement_2 : Prop := (perpendicular m l) ∧ (perpendicular n l) → (parallel m n)
  let statement_3 : Prop := (parallel m l) ∧ (parallel m alpha) → (parallel l alpha)
  let statement_4 : Prop := (parallel l m) ∧ (lies_in l alpha) ∧ (lies_in m beta) → (parallel alpha beta)
  let statement_5 : Prop := (lies_in m alpha) ∧ (parallel m beta) ∧ (lies_in l beta) ∧ (parallel l alpha) → (parallel alpha beta)
  let statement_6 : Prop := (parallel alpha gamma) ∧ (parallel beta gamma) → (parallel alpha beta)
  
  -- Count the correct statements
  let correct_statements : Nat := sorry -- We assume the correct solution derived earlier

  -- Verify that the number of correct statements is 2
  correct_statements = 2

-- The theorem to be proven
theorem verify_lines_planes_statements : lines_planes_statements_correct := by
  sorry

end verify_lines_planes_statements_l537_537206


namespace clerical_percentage_correct_l537_537748

noncomputable def companyX : Type :=
{
    total_employees : ℕ // total_employees = 6000
}

def initial_clerical : ℕ :=
    2 * companyX.total_employees / 10

def initial_technical : ℕ :=
    3 * companyX.total_employees / 10

def initial_managerial : ℕ :=
    5 * companyX.total_employees / 10

def reduced_clerical : ℕ :=
    initial_clerical - initial_clerical / 4

def reduced_technical : ℕ :=
    initial_technical - initial_technical / 5

def reduced_managerial : ℕ :=
    initial_managerial - initial_managerial / 10

def post_promotion_clerical : ℕ :=
    reduced_clerical - 50

def post_promotion_managerial : ℕ :=
    reduced_managerial + 50

def post_shift_technical : ℕ :=
    reduced_technical - 90

def post_shift_clerical : ℕ :=
    post_promotion_clerical + 90

def total_remaining_employees : ℕ :=
    post_shift_clerical + post_shift_technical + post_promotion_managerial

def percentage_clerical : ℚ :=
    (post_shift_clerical : ℚ) / (total_remaining_employees : ℚ) * 100

theorem clerical_percentage_correct : 
    percentage_clerical = 18.65 := 
sorry

end clerical_percentage_correct_l537_537748


namespace tiles_per_row_24_l537_537060

noncomputable def num_tiles_per_row (area : ℝ) (tile_size : ℝ) : ℝ :=
  let side_length_ft := Real.sqrt area
  let side_length_in := side_length_ft * 12
  side_length_in / tile_size

theorem tiles_per_row_24 :
  num_tiles_per_row 324 9 = 24 :=
by
  sorry

end tiles_per_row_24_l537_537060


namespace sum_of_coordinates_of_C_parallelogram_l537_537588

-- Definitions that encapsulate the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, 0⟩
def D : Point := ⟨5, -4⟩

-- The theorem we need to prove
theorem sum_of_coordinates_of_C_parallelogram :
  ∃ C : Point, C.x + C.y = 7 ∧
  ∃ M : Point, M = ⟨(A.x + D.x) / 2, (A.y + D.y) / 2⟩ ∧
  (M = ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩) :=
sorry

end sum_of_coordinates_of_C_parallelogram_l537_537588


namespace second_order_derivative_l537_537506

variable {t : ℝ}

def x (t : ℝ) := Real.sqrt (1 - t^2)

def y (t : ℝ) := 1 / t

theorem second_order_derivative :
  ∀ t, (y''_{xx} t) = (3 - 2 * t^2) / t^5 := by
  sorry

end second_order_derivative_l537_537506


namespace sum_a_b_eq_73_l537_537491

theorem sum_a_b_eq_73
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : (real.sqrt (a - 8 / b) = a * real.sqrt (8 / b))) :
  a + b = 73 :=
sorry

end sum_a_b_eq_73_l537_537491


namespace no_intersection_points_l537_537969

theorem no_intersection_points :
  ¬ ∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|2 * x + 1| :=
by
  sorry

end no_intersection_points_l537_537969


namespace length_of_ST_l537_537361

noncomputable def isosceles_triangle_area_given_base_height (PQR_base: ℝ) (height: ℝ) : ℝ :=
  1 / 2 * PQR_base * height

noncomputable def smaller_triangle_area (PQR_area: ℝ) (trapezoid_area: ℝ) : ℝ :=
  PQR_area - trapezoid_area

noncomputable def side_length_ratio (smaller_triangle_area: ℝ) (PQR_area: ℝ) : ℝ :=
  real.sqrt (smaller_triangle_area / PQR_area)

theorem length_of_ST
  (PQR_area : ℝ) (PQR_altitude : ℝ) (trapezoid_area: ℝ) (ST_length: ℝ) :
  PQR_area = 180 → PQR_altitude = 30 → trapezoid_area = 135 → 
  let PQR_base := (2 * PQR_area) / PQR_altitude in 
  let PST_area := smaller_triangle_area PQR_area trapezoid_area in
  let ratio := side_length_ratio PST_area PQR_area in 
  ST_length = ratio * PQR_base → 
  ST_length = 6 :=
by
  intros 
  let PQR_base := (2 * 180) / 30
  let PST_area := smaller_triangle_area 180 135
  let ratio := side_length_ratio PST_area 180
  have h1 : PQR_base = 12 := by sorry
  have h2 : ratio = 1 / 2 := by sorry
  change ST_length = 1 / 2 * 12 → ST_length = 6 
  intros
  exact h2

-- To be completed with detailed proof steps.

end length_of_ST_l537_537361


namespace describe_T_l537_537718

def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (common : ℝ), 
    (common = 5 ∧ p.1 + 3 = common ∧ p.2 - 6 ≤ common) ∨
    (common = 5 ∧ p.2 - 6 = common ∧ p.1 + 3 ≤ common) ∨
    (common = p.1 + 3 ∧ common = p.2 - 6 ∧ common ≤ 5)}

theorem describe_T :
  T = {(2, y) | y ≤ 11} ∪ { (x, 11) | x ≤ 2} ∪ { (x, x + 9) | x ≤ 2} :=
by
  sorry

end describe_T_l537_537718


namespace simplify_expression_l537_537420

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : y ^ (-2) - 2 = (1 - 2 * y ^ 2) / y ^ 2 :=
by
  sorry

end simplify_expression_l537_537420


namespace no_natural_k_l537_537220

-- Define the infinite sequence of prime numbers
def prime : ℕ → ℕ := λ n, (nth prime_seqs n)

-- Define a function to compute the product of the first k odd prime numbers
def product_odd_primes (k : ℕ) : ℕ :=
  (List.range k).foldl (λ acc i, if i = 0 then 1 else acc * (prime (i*2))) 1

-- Define the main statement
theorem no_natural_k :
  ¬ ∃ k n a : ℕ, 2 ≤ n ∧ a ≥ 1 ∧ (product_odd_primes k - 1 = a^n) :=
begin
  sorry
end

end no_natural_k_l537_537220


namespace proof_problem_l537_537622

-- Definitions of parallel and perpendicular relationships for lines and planes
def parallel (α β : Type) : Prop := sorry
def perpendicular (α β : Type) : Prop := sorry
def contained_in (m : Type) (α : Type) : Prop := sorry

-- Variables representing lines and planes
variables (l m n : Type) (α β : Type)

-- Assumptions from the conditions in step a)
variables 
  (h1 : m ≠ l)
  (h2 : α ≠ β)
  (h3 : parallel m n)
  (h4 : perpendicular m α)
  (h5 : perpendicular n β)

-- The goal is to prove that the planes α and β are parallel under the given conditions
theorem proof_problem : parallel α β :=
sorry

end proof_problem_l537_537622


namespace total_cost_of_fruits_l537_537423

noncomputable def cost_of_apples (n : ℕ) (cost_per_apple : ℝ) : ℝ :=
  n * cost_per_apple

noncomputable def cost_of_oranges (total_cost : ℝ) (cost_of_apples : ℝ) : ℝ :=
  total_cost - cost_of_apples

noncomputable def cost_of_one_orange (cost_of_three_oranges : ℝ) : ℝ :=
  cost_of_three_oranges / 3

noncomputable def total_cost_second_group (cost_of_two_apples : ℝ) (cost_of_five_oranges : ℝ) : ℝ :=
  cost_of_two_apples + cost_of_five_oranges

theorem total_cost_of_fruits :
  let cost_per_apple : ℝ := 0.21
  let total_cost_6_apples_3_oranges : ℝ := 1.77
  let cost_of_6_apples := cost_of_apples 6 cost_per_apple
  let cost_of_3_oranges := cost_of_oranges total_cost_6_apples_3_oranges cost_of_6_apples
  let cost_of_one_orange := cost_of_one_orange cost_of_3_oranges
  let cost_of_2_apples := cost_of_apples 2 cost_per_apple
  let cost_of_5_oranges := cost_per_apple * 5 * cost_of_one_orange
  let total_cost := total_cost_second_group cost_of_2_apples cost_of_5_oranges
  in total_cost = 1.27 :=
begin
  sorry,
end

end total_cost_of_fruits_l537_537423


namespace eighth_odd_multiple_of_5_l537_537472

theorem eighth_odd_multiple_of_5 : 
  (∃ n : ℕ, n = 8 ∧ ∃ k : ℤ, k = (10 * n - 5) ∧ k > 0 ∧ k % 2 = 1) → 75 := 
by {
  sorry
}

end eighth_odd_multiple_of_5_l537_537472


namespace measure_exactly_10_liters_l537_537358

theorem measure_exactly_10_liters (A B : ℕ) (A_cap B_cap : ℕ) (hA : A_cap = 11) (hB : B_cap = 9) :
  ∃ (A B : ℕ), A + B = 10 ∧ A ≤ A_cap ∧ B ≤ B_cap := 
sorry

end measure_exactly_10_liters_l537_537358


namespace line_inclination_angle_l537_537450

theorem line_inclination_angle (θ : ℝ) : 
  (∃ θ : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → θ = 3 * π / 4) := sorry

end line_inclination_angle_l537_537450


namespace angle_generatrix_height_is_30_deg_max_area_VAB_is_sqrt3_l537_537271

noncomputable def cone_base_radius := 1
noncomputable def cone_slant_height := 2

theorem angle_generatrix_height_is_30_deg : 
  (angle_between_generatrix_and_height cone_base_radius cone_slant_height) = 30 :=
sorry

theorem max_area_VAB_is_sqrt3 : 
  (max_area_triangle_VAB cone_base_radius cone_slant_height) = sqrt 3 :=
sorry

end angle_generatrix_height_is_30_deg_max_area_VAB_is_sqrt3_l537_537271


namespace density_transformation_l537_537384

variables {n : ℕ} (I : set (fin n → ℝ)) (ϕ : (fin n → ℝ) → (fin n → ℝ))
  (Jϕ : ((fin n → ℝ) → (fin n → ℝ) → ℝ)) (X : (fin n → ℝ) → ℝ) (ψ : (fin n → ℝ) → (fin n → ℝ))
  (fX : (fin n → ℝ) → ℝ) (fY : (fin n → ℝ) → ℝ)

-- Conditions
open_locale classical

axiom open_set_I : is_open I
axiom differentiable_ϕ : ∀ x ∈ I, ∀ i j, differentiable_at ℝ (λ x_j, ϕ i x_j) x
axiom continuous_partial_derivatives :
  ∀ x ∈ I, ∀ i j, continuous (λ x_j, (∂ ϕ i) j)
axiom positive_jacobian_det : ∀ x ∈ I, (det (jacobian ϕ x)) ≠ 0
axiom ψ_inverse_ϕ : ∀ x ∈ I, ψ (ϕ x) = x
axiom ϕ_inverse_ψ : ∀ y, (fY y) = (fX (ψ y)) * (abs (det (jacobian ψ y)))

-- The statement we want to prove:
theorem density_transformation :
  (∀ y, fY y = fX (ψ y) * abs (det (jacobian ψ y))) :=
sorry

end density_transformation_l537_537384


namespace circle_properties_l537_537598

-- Conditions
def passes_through (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 5 ∧ (x - 2)^2 + (y + 2)^2 = 5

def on_line (x y : ℝ) : Prop :=
  x + 3 * y + 3 = 0

def line_3x4y21 (x y : ℝ) : Prop :=
  3 * x + 4 * y - 21 = 0

def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 5

def distance_to_line (x y : ℝ) : ℝ :=
  abs (3 * x + 4 * y - 21) / 5

def area_function (t : ℝ) : ℝ :=
  sqrt 5 * sqrt (t^2 - 5)

-- Proof problem
theorem circle_properties :
  (∃ (x y : ℝ), passes_through x y ∧ on_line x y) ∧
  (∃ (P : ℝ × ℝ) (C : ℝ × ℝ), line_3x4y21 P.1 P.2 ∧ 
    |((P.1 - C.1), (P.2 - C.2))| = t ∧
    ∃ (S : ℝ), S = area_function t ∧ (∀ t ≥ 5, S ≥ 10)) :=
by
  sorry

end circle_properties_l537_537598


namespace find_smallest_a_l537_537200

open Nat

def f (n : ℕ) : ℕ := (list.range (n + 1)).lcm

theorem find_smallest_a : ∃ a : ℕ, f a = f (a + 2) ∧ a = 13 :=
by
  use 13
  have : f 13 = f 15 := sorry
  exact ⟨this, rfl⟩

end find_smallest_a_l537_537200


namespace fraction_of_one_third_l537_537470

theorem fraction_of_one_third (x : ℚ) (h: x * (3 / 7 : ℚ) = 0.12499999999999997) : 
  (x * 3 = 7 / 8) := by
  sorry

end fraction_of_one_third_l537_537470


namespace no_real_solution_l537_537571

-- Define the equation
def equation (a b : ℝ) : Prop := a^2 + 3 * b^2 + 2 = 3 * a * b

-- Prove that there do not exist real numbers a and b such that equation a b holds
theorem no_real_solution : ¬ ∃ a b : ℝ, equation a b :=
by
  -- Proof placeholder
  sorry

end no_real_solution_l537_537571


namespace herd_size_l537_537520

open Rat

theorem herd_size 
  (n : ℕ)
  (h1 : (3 / 7 : ℚ) * n + (1 / 3 : ℚ) * n + (1 / 6 : ℚ) * n ≤ n)
  (h2 : (1 - ((3 / 7 : ℚ) + (1 / 3 : ℚ) + (1 / 6 : ℚ))) * n = 16) :
  n = 224 := by
  sorry

end herd_size_l537_537520


namespace project_completion_time_l537_537045

theorem project_completion_time (t : ℕ) (t₂ : ℕ) (workers₁ workers₂ : ℕ) (days₁ : ℕ) (days₂ : ℕ)
  (h₁ : workers₁ = 60) (h₂ : days₁ = 7) (h₃ : workers₂ = 35) (h₄ : days₂ = 12)
  (h : (workers₁ * days₁) = (workers₂ * days₂)) : t₂ = 12 :=
by
  simp [h₁, h₂, h₃, h₄] at h
  exact h₂

# eval project_completion_time

end project_completion_time_l537_537045


namespace algebraic_expression_added_l537_537093

theorem algebraic_expression_added (k : ℕ) (hk : k > 1) :
  (∑ i in (finset.range (k+1)).map (finset.nat.antidiagonal_map (k+1)), (1 / (k + i + 1 : ℝ))) - 
  (∑ i in (finset.range k).map (finset.nat.antidiagonal_map (k+1)), (1 / (k + i + 1 : ℝ))) =
  (1 / (2*k + 1 : ℝ)) - (1 / (2*k + 2 : ℝ)) :=
by sorry

end algebraic_expression_added_l537_537093


namespace problem_solution_l537_537286

noncomputable def curve (x : ℝ) : ℝ := (x + 1) / (x - 1)

def tangent_slope_at (x : ℝ) : ℝ := 
  -2 / ((x - 1) ^ 2)

def line_slope (a : ℝ) : ℝ := -a

theorem problem_solution :
  (tangent_slope_at 3) * (line_slope a) = -1 → a = -2 :=
by
  intros h
  sorry

end problem_solution_l537_537286


namespace magnitude_of_z_l537_537599

theorem magnitude_of_z (z : ℂ) (h : z * (1 - 2 * I) = 4 + 2 * I) : complex.abs z = 2 :=
sorry

end magnitude_of_z_l537_537599


namespace proof_problem_l537_537193

-- Define the conditions based on Classmate A and Classmate B's statements
def classmateA_statement (x y : ℝ) : Prop := 6 * x = 5 * y
def classmateB_statement (x y : ℝ) : Prop := x = 2 * y - 40

-- Define the system of equations derived from the statements
def system_of_equations (x y : ℝ) : Prop := (6 * x = 5 * y) ∧ (x = 2 * y - 40)

-- Proof goal: Prove the system of equations if classmate statements hold
theorem proof_problem (x y : ℝ) :
  classmateA_statement x y ∧ classmateB_statement x y → system_of_equations x y :=
by
  sorry

end proof_problem_l537_537193


namespace age_of_20th_student_l537_537688

theorem age_of_20th_student :
  let n := 20 in
  let avg_total_age := 18 in
  let avg_age_6 := 16 in
  let avg_age_8 := 17 in
  let avg_age_5 := 21 in
  let total_age := n * avg_total_age in
  let total_age_6 := 6 * avg_age_6 in
  let total_age_8 := 8 * avg_age_8 in
  let total_age_5 := 5 * avg_age_5 in
  let known_total_age := total_age_6 + total_age_8 + total_age_5 in
  let age_20th_student := total_age - known_total_age in
  age_20th_student = 23 :=
by
  let n := 20
  let avg_total_age := 18
  let avg_age_6 := 16
  let avg_age_8 := 17
  let avg_age_5 := 21
  let total_age := n * avg_total_age
  let total_age_6 := 6 * avg_age_6
  let total_age_8 := 8 * avg_age_8
  let total_age_5 := 5 * avg_age_5
  let known_total_age := total_age_6 + total_age_8 + total_age_5
  let age_20th_student := total_age - known_total_age
  show age_20th_student = 23 from by
    sorry

end age_of_20th_student_l537_537688


namespace man_speed_correct_l537_537514

noncomputable def speed_of_man (L : ℝ) (t : ℝ) (V_t : ℝ) : ℝ := 
  V_t - (L / t)

theorem man_speed_correct :
  let L := 700
  let t := 41.9966402687785
  let V_t := 63 * 1000 / 3600
  speed_of_man L t V_t ≈ 0.832 := by
  sorry

end man_speed_correct_l537_537514


namespace sufficient_but_not_necessary_for_a_plus_b_gt_zero_l537_537287

theorem sufficient_but_not_necessary_for_a_plus_b_gt_zero (a b : ℝ):
  (a + b > 2) ∨ (a > 0 ∧ b > 0) → a + b > 0 :=
by
  intro h
  cases h
  case inl =>
    apply lt_of_lt_of_le h (le_add_of_nonneg_left (by norm_num))
  case inr =>
    apply add_pos h.1 h.2
  done

end sufficient_but_not_necessary_for_a_plus_b_gt_zero_l537_537287


namespace square_inscription_l537_537537

theorem square_inscription (a b : ℝ) (s1 s2 : ℝ)
  (h_eq_side_smaller : s1 = 4)
  (h_eq_side_larger : s2 = 3 * Real.sqrt 2)
  (h_sum_segments : a + b = s2)
  (h_eq_sum_squares : a^2 + b^2 = (4 * Real.sqrt 2)^2) :
  a * b = -7 := 
by sorry

end square_inscription_l537_537537


namespace t_minus_s_l537_537155

noncomputable def t_s_diff : ℝ :=
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (list.sum class_sizes) / teachers
  let s := (∑ size in class_sizes, size * (size / students))
  t - s

theorem t_minus_s : t_s_diff = -13.5 := 
  by
	sorry

end t_minus_s_l537_537155


namespace minimum_operations_to_sort_volumes_l537_537749

theorem minimum_operations_to_sort_volumes : 
  ∀ (n : ℕ), n = 30 → 
  (∀ (swap : ℕ → ℕ → Type), 
    (∀ (i j : ℕ), i = j + 1 ∨ i = j - 1 → swap i j) → 
    ∃ (min_operations : ℕ), min_operations = 435) :=
begin
  sorry
end

end minimum_operations_to_sort_volumes_l537_537749


namespace min_value_of_a_l537_537293

theorem min_value_of_a (a : ℝ) (h : ∃ x : ℝ, |x - 1| + |x + a| ≤ 8) : -9 ≤ a :=
by
  sorry

end min_value_of_a_l537_537293


namespace product_of_positive_real_solutions_l537_537321

theorem product_of_positive_real_solutions : 
  ∀ (x : ℂ), (x^6 = -216) → 0 < x.re → 
  ∏ (hx : x^6 = -216) in {x | 0 < x.re}, x = 6 :=
by
  sorry

end product_of_positive_real_solutions_l537_537321


namespace number_of_sides_of_regular_polygon_l537_537973

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l537_537973


namespace trader_profit_l537_537164

theorem trader_profit (profit donation goal : ℕ) (h1 : profit = 960) (h2 : donation = 310) (h3 : goal = 610) :
  (profit / 2 + donation - goal = 180) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end trader_profit_l537_537164


namespace product_lcm_gcd_eq_128_l537_537581

theorem product_lcm_gcd_eq_128 : (Int.gcd 8 16) * (Int.lcm 8 16) = 128 :=
by
  sorry

end product_lcm_gcd_eq_128_l537_537581


namespace angle_ACP_eq_angle_QCB_l537_537393

variables {A B C P Q : Point} {Γ Γ₁ : Circle}
variables {AC BC : Line}

-- Conditions
def circumcircle (Δ : Triangle) : Circle := sorry
def tangent_to_line (c : Circle) (l : Line) : Prop := sorry
def internally_tangent (c₁ c₂ : Circle) (p : Point) : Prop := sorry
def parallel_to (l₁ l₂ : Line) : Prop := sorry
def in_triangle (Δ : Triangle) (p : Point) : Prop := sorry

variables (ΔABC : Triangle)
variables (l_parallel : Line)
variables (AC BC : Line)

axiom circle_Γ_is_circumcircle (h : circumcircle ΔABC = Γ)
axiom circle_Γ₁_tangent_to_AC_and_BC (h : tangent_to_line Γ₁ AC ∧ tangent_to_line Γ₁ BC)
axiom Γ₁_internally_tangent_to_Γ_at_P (h : internally_tangent Γ₁ Γ P)
axiom line_parallel_to_AB_is_tangent_to_Γ₁_at_Q (h : parallel_to l_parallel (line_through A B) ∧ tangent_to_line Γ₁ l_parallel ∧ in_triangle ΔABC Q)

theorem angle_ACP_eq_angle_QCB
  (h1 : circumcircle ΔABC = Γ)
  (h2 : tangent_to_line Γ₁ AC ∧ tangent_to_line Γ₁ BC)
  (h3 : internally_tangent Γ₁ Γ P)
  (h4 : parallel_to l_parallel (line_through A B) ∧ tangent_to_line Γ₁ l_parallel ∧ in_triangle ΔABC Q) : 
  angle (line_through A C) (line_through C P) = angle (line_through Q C) (line_through C B) :=
sorry

end angle_ACP_eq_angle_QCB_l537_537393


namespace radius_of_base_of_cone_l537_537284

theorem radius_of_base_of_cone (S : ℝ) (hS : S = 9 * Real.pi)
  (H : ∃ (l r : ℝ), (Real.pi * l = 2 * Real.pi * r) ∧ S = Real.pi * r^2 + Real.pi * r * l) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_l537_537284


namespace millet_container_volume_l537_537871

theorem millet_container_volume :
  let C := 54 -- circumference in chi
  let h := 18 -- height in chi
  let pi := 3 -- approximate value of π
  
  -- converting hu to cubic chi
  let hu := 1.62
  
  -- calculating radius from circumference
  let r := C / (2 * pi)
  
  -- calculating volume of cylinder in cubic chi
  let volume_cylinder := pi * r^2 * h
  
  -- converting cubic chi to hu
  let volume_hu := volume_cylinder / hu
  
  volume_hu ≈ 2700 :=
by {
  let C := 54
  let h := 18
  let pi := 3
  let hu := 1.62
  let r := C / (2 * pi)
  let volume_cylinder := pi * r^2 * h
  let volume_hu := volume_cylinder / hu
  sorry
}

end millet_container_volume_l537_537871


namespace simple_interest_years_l537_537538

theorem simple_interest_years (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * ((R + 6) / 100) * T) = (P * (R / 100) * T + 90)) : 
  T = 5 := 
by 
  -- Necessary proof steps go here
  sorry

end simple_interest_years_l537_537538


namespace vector_magnitude_sum_l537_537301

variables (a b : ℝ)
variables (a_vec b_vec : EuclideanSpace ℝ (Fin 2))

axiom magnitude_a : ‖a_vec‖ = 6
axiom magnitude_b : ‖b_vec‖ = 5
axiom angle_ab : Real.angleCos (a_vec, b_vec) = -0.5

theorem vector_magnitude_sum :
  ‖a_vec + b_vec‖ = Real.sqrt 31 :=
by
  sorry

end vector_magnitude_sum_l537_537301


namespace cos_of_60_degrees_is_half_l537_537957

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537957


namespace A_inter_CUB_eq_l537_537656

noncomputable def U := Set.univ (ℝ)

noncomputable def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }

noncomputable def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x + 1 }

noncomputable def C_U (s : Set ℝ) := { x : ℝ | x ∉ s }

noncomputable def A_inter_CUB := A ∩ C_U B

theorem A_inter_CUB_eq : A_inter_CUB = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
  by sorry

end A_inter_CUB_eq_l537_537656


namespace ARML_value_l537_537011

theorem ARML_value 
  (A R M L : ℝ) 
  (h1 : log 10 (A * L) + log 10 (A * M) = 2)
  (h2 : log 10 (M * L) + log 10 (M * R) = 3)
  (h3 : log 10 (R * A) + log 10 (R * L) = 4) :
  A * R * M * L = 1000 :=
sorry

end ARML_value_l537_537011


namespace exists_parallelepiped_with_edges_on_skew_lines_l537_537498

theorem exists_parallelepiped_with_edges_on_skew_lines
    (f g h : ℝ → ℝ × ℝ × ℝ) 
    (hf_skew : ∀ t1 t2, f t1 ≠ g t2 ∧ f t1 ≠ h t2 ∧ g t1 ≠ h t2)
    (line : ℝ → ℝ × ℝ × ℝ → Set ℝ Untimed 3 [space])
    (h1 : ∀ t1, f t1 ∈ line t1 (f t1))
    (h2 : ∀ t2, g t2 ∈ line t2 (g t2))
    (h3 : ∀ t3, h t3 ∈ line t3 (h t3))
    : ∃ P : Set (ℝ × ℝ × ℝ), parallelepiped P ∧ 
      (line 1 ⟨P.left_sides⟩ = f(0)) ∧ 
      (line 2 ⟨P.right_sides⟩ = g(0)) ∧ 
      (line 3 ⟨P.front_sides⟩ = h(0)) :=
by
  sorry

end exists_parallelepiped_with_edges_on_skew_lines_l537_537498


namespace conference_duration_l537_537140

theorem conference_duration (hours minutes lunch_break total_minutes active_session : ℕ) 
  (h1 : hours = 8) 
  (h2 : minutes = 40) 
  (h3 : lunch_break = 15) 
  (h4 : total_minutes = hours * 60 + minutes)
  (h5 : active_session = total_minutes - lunch_break) :
  active_session = 505 := 
by {
  sorry
}

end conference_duration_l537_537140


namespace factorial_fraction_simplification_l537_537568

theorem factorial_fraction_simplification : 
  (10.factorial * 6.factorial * 2.factorial) / (9.factorial * 7.factorial) = 20 / 7 :=
by
  sorry

end factorial_fraction_simplification_l537_537568


namespace total_distance_of_journey_l537_537545

-- Definitions corresponding to conditions in the problem
def electric_distance : ℝ := 30 -- The first 30 miles were in electric mode
def gasoline_consumption_rate : ℝ := 0.03 -- Gallons per mile for gasoline mode
def average_mileage : ℝ := 50 -- Miles per gallon for the entire trip

-- Final goal: proving the total distance is 90 miles
theorem total_distance_of_journey (d : ℝ) :
  (d / (gasoline_consumption_rate * (d - electric_distance)) = average_mileage) → d = 90 :=
by
  sorry

end total_distance_of_journey_l537_537545


namespace regular_polygon_sides_l537_537991

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l537_537991


namespace zeros_in_expansion_l537_537663

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end zeros_in_expansion_l537_537663


namespace value_of_a_l537_537827

theorem value_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : {a - 2, b + 2, c^2} = {a, b, c}) : 
  a = 1003 :=
by
  sorry

end value_of_a_l537_537827


namespace simplify_and_evaluate_l537_537419

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -1) (hy : y = -1/3) :
  ((3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y)) = 8 := by
  sorry

end simplify_and_evaluate_l537_537419


namespace edge_length_of_cube_l537_537499

/-- Define the total paint volume, remaining paint and cube paint volume -/
def total_paint_volume : ℕ := 25 * 40
def remaining_paint : ℕ := 271
def cube_paint_volume : ℕ := total_paint_volume - remaining_paint

/-- Define the volume of the cube and the statement for edge length of the cube -/
theorem edge_length_of_cube (s : ℕ) : s^3 = cube_paint_volume → s = 9 :=
by
  have h1 : cube_paint_volume = 729 := by rfl
  sorry

end edge_length_of_cube_l537_537499


namespace concurrency_concyclicity_l537_537010

-- Definitions 
variables {α : Type} [EuclideanGeometry α]
variables {A B C D E F : α}

-- Conditions
def is_cyclic (P Q R S : α) : Prop := ∃ (γ : circle α), γ.contains P ∧ γ.contains Q ∧ γ.contains R ∧ γ.contains S
def not_parallel (l₁ l₂ : line α) : Prop := ¬ parallel l₁ l₂

-- Problem statement (lean statement only, no proof required)
theorem concurrency_concyclicity :
  is_cyclic A B C D ∧ is_cyclic C D E F ∧ 
  not_parallel (line_through A B) (line_through C D) ∧ 
  not_parallel (line_through C D) (line_through E F) ∧ 
  not_parallel (line_through E F) (line_through A B)
  ↔
  (concurrent (line_through A B) (line_through C D) (line_through E F)) 
  ↔
  is_cyclic A B E F :=
sorry

end concurrency_concyclicity_l537_537010


namespace inclination_angle_l537_537069

theorem inclination_angle (α : ℝ) (hα : α ∈ Ico 0 real.pi)
  (h_slope : real.tan α = 1) : α = real.pi / 4 :=
sorry

end inclination_angle_l537_537069


namespace mark_additional_inches_l537_537743

theorem mark_additional_inches
  (mark_feet : ℕ)
  (mark_inches : ℕ)
  (mike_feet : ℕ)
  (mike_inches : ℕ)
  (foot_to_inches : ℕ)
  (mike_taller_than_mark : ℕ) :
  mark_feet = 5 →
  mike_feet = 6 →
  mike_inches = 1 →
  mike_taller_than_mark = 10 →
  foot_to_inches = 12 →
  5 * 12 + mark_inches + 10 = 6 * 12 + 1 →
  mark_inches = 3 :=
by
  intros
  sorry

end mark_additional_inches_l537_537743


namespace largest_area_triangle_ABC_l537_537372

-- Definitions of the given conditions
variables
  (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (BC : Segment B C) (AM : Segment A M)
  (hMBC : M ∈ BC) -- M is on segment BC
  (hAM : dist A M = 3) -- AM = 3
  (hBM : dist B M = 4) -- BM = 4
  (hCM : dist C M = 5) -- CM = 5

-- Target statement to prove the area of triangle ABC is 27/2
theorem largest_area_triangle_ABC : 
  ∃ (A B C : Point) (M : Point) (BC : Segment B C) (AM : Segment A M),
    M ∈ BC ∧ dist A M = 3 ∧ dist B M = 4 ∧ dist C M = 5 ∧ 
    area_triangle A B C = 27 / 2 :=
by 
  sorry

end largest_area_triangle_ABC_l537_537372


namespace common_ratio_eq_l537_537677

variables {x y z r : ℝ}

theorem common_ratio_eq (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hgp : x * (y - z) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (x - y) ≠ 0 ∧ 
          (y * (z - x)) / (x * (y - z)) = r ∧ (z * (x - y)) / (y * (z - x)) = r) :
  r^2 + r + 1 = 0 :=
sorry

end common_ratio_eq_l537_537677


namespace points_lie_on_curve_g_divides_f_l537_537653

-- Define the sequence
def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1   -- This will represent a_1 (index starts at 0 in Lean)
  | 1 => 2   -- This will represent a_2
  | n + 2 => sequence (n + 1) + sequence n
  | _ => 0 -- This should never be needed. 

-- Prove (1)
theorem points_lie_on_curve :
  ∀ k : ℕ, (sequence (2 * k) ^ 2 + sequence (2 * k) * sequence (2 * k + 1) - sequence (2 * k + 1) ^ 2 + 1 = 0) := 
sorry

-- Define f(x) and g(x)
def f (x : ℤ) (n : ℕ) : ℤ :=
  x ^ n + x ^ (n - 1) - sequence n * x - sequence (n - 1)

def g (x : ℤ) : ℤ := 
  x ^ 2 - x - 1

-- Prove (2)
theorem g_divides_f (n : ℕ) :
  ∃ q : ℤ → ℤ, f = λ x, (g x) * (q x) :=
sorry

end points_lie_on_curve_g_divides_f_l537_537653


namespace hyperbola_properties_l537_537648

theorem hyperbola_properties
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_asymptote : 2 * x + y = 0)
  (h_distance : abs (2 * 0 + a) / sqrt (4 + 1) = 2 * sqrt 5 / 5) :
  ( ∃ (a b : ℝ), a = 2 ∧ b = 1 ∧
  (∀ x y : ℝ, (y^2 / 4 - x^2 = 1) ∧
  (let P := (m - n) / 2, m + n in
   let A := (m, 2 * m), B := (-n, 2 * n) in
   vec.AP = vec.PB → A ∈ first_quadrant → B ∈ second_quadrant → 
   area_triangle_OAB A B = 2))) := sorry

end hyperbola_properties_l537_537648


namespace monotonicity_f_minimum_k_l537_537641

def f (a x : ℝ) : ℝ := log x - a / (x + 1)

theorem monotonicity_f (a x : ℝ) (hₐ : a ≥ -4) : 
  ∀ x ≥ 0, deriv (f a) x ≥ 0 := 
sorry

theorem minimum_k (a k x₁ x₂ : ℝ) (hₐ : a < -4) (h₁ : x₁ + x₂ = -(a+2)) (h₂ : x₁ * x₂ = 1) 
  (ineq : k * exp (f a x₁ + f a x₂ - 4) + log (k / (x₁ + x₂ - 2)) ≥ 0) : 
  k ≥ 1/exp 1 := 
sorry

end monotonicity_f_minimum_k_l537_537641


namespace josh_marbles_earlier_l537_537002

-- Define the conditions
def marbles_lost : ℕ := 11
def marbles_now : ℕ := 8

-- Define the problem statement
theorem josh_marbles_earlier : marbles_lost + marbles_now = 19 :=
by
  sorry

end josh_marbles_earlier_l537_537002


namespace simplify_expression_l537_537422

variables (y : ℝ)

theorem simplify_expression : 
  3 * y + 4 * y^2 - 2 - (8 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 10 :=
by sorry

end simplify_expression_l537_537422


namespace probability_red_or_white_is_7_over_10_l537_537500

/-
A bag consists of 20 marbles, of which 6 are blue, 9 are red, and the remainder are white.
If Lisa is to select a marble from the bag at random, prove that the probability that the
marble will be red or white is 7/10.
-/
def num_marbles : ℕ := 20
def num_blue : ℕ := 6
def num_red : ℕ := 9
def num_white : ℕ := num_marbles - (num_blue + num_red)

def probability_red_or_white : ℚ :=
  (num_red + num_white) / num_marbles

theorem probability_red_or_white_is_7_over_10 :
  probability_red_or_white = 7 / 10 := 
sorry

end probability_red_or_white_is_7_over_10_l537_537500


namespace total_boys_in_camp_l537_537864

theorem total_boys_in_camp (T : ℕ) (h1 : 0.20 * T = 0.20 * T)
  (h2 : 0.30 * (0.20 * T) / (0.30 * (0.20 * T) / 0.70) = 42) :
  T = 300 := 
  sorry

end total_boys_in_camp_l537_537864


namespace mean_combined_set_l537_537442

noncomputable def mean (s : Finset ℚ) : ℚ :=
  (s.sum id) / s.card

theorem mean_combined_set :
  ∀ (s1 s2 : Finset ℚ),
  s1.card = 7 →
  s2.card = 8 →
  mean s1 = 15 →
  mean s2 = 18 →
  mean (s1 ∪ s2) = 249 / 15 :=
by
  sorry

end mean_combined_set_l537_537442


namespace smallest_non_square_product_of_four_primes_l537_537105

theorem smallest_non_square_product_of_four_primes :
  ∃ n, (∃ a b c d : ℕ, prime a ∧ prime b ∧ prime c ∧ prime d ∧ n = a * b * c * d) ∧ n > 0 ∧ ¬ (∃ k : ℕ, k^2 = n) ∧
      ∀ m, (∃ a b c d : ℕ, prime a ∧ prime b ∧ prime c ∧ prime d ∧ m = a * b * c * d) ∧ m > 0 ∧ ¬ (∃ k : ℕ, k^2 = m) → 24 ≤ m 
  :=
by
  existsi 24
  sorry

end smallest_non_square_product_of_four_primes_l537_537105


namespace probability_one_number_twice_the_other_l537_537238

theorem probability_one_number_twice_the_other : 
    (let S := {1, 2, 3, 4};
     let pairs := [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
     let valid_pairs := [(1, 2), (2, 4)];
     let total_pairs := 6;
     let total_valid_pairs := 2;
     (total_valid_pairs / total_pairs : ℝ)) = (1 / 3 : ℝ) :=
by
  sorry

end probability_one_number_twice_the_other_l537_537238


namespace hotel_manager_assignments_l537_537147

theorem hotel_manager_assignments :
  let rooms : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let friends : Finset ℕ := {1, 2, 3, 4, 5, 6}
  ∃ (assignments : friends → rooms),
    (∀ r ∈ rooms, 1 ≤ (friends.filter (λ f, assignments f = r)).card ∧ (friends.filter (λ f, assignments f = r)).card ≤ 3) ∧
    (∀ f1 f2 : friends, if f1 = 1 ∧ f2 = 2 then assignments f1 = assignments f2 else true) ∧
    (friends.card = 6 ∧ rooms.card = 6) →
    (number_of_assignments = 3600) := sorry

end hotel_manager_assignments_l537_537147


namespace work_days_l537_537117

variables (A B : Type) [has_div A] [has_div B]

def thrice_as_fast (a b : ℝ) : Prop := a = 3 * b

def combined_days (a b d : ℝ) : Prop := 1 / d = 1 / a + 1 / b

theorem work_days (a b : ℝ) (h1 : thrice_as_fast a b) 
  (h2 : combined_days a b 24) : a = 32 := by
  sorry

end work_days_l537_537117


namespace generating_function_set1_no_generating_function_set2_generating_function_problem_II_generating_function_problem_III_l537_537587

-- Proof theorems for Problem (Ⅰ)
theorem generating_function_set1 : 
  ∃ a b : ℝ, (λ x : ℝ, a * Real.sin x + b * Real.cos x) = (λ x : ℝ, Real.sin (x + Real.pi / 3)) := by
  sorry

theorem no_generating_function_set2 : 
  ¬∃ a b : ℝ, (λ x : ℝ, a * (x^2 - x) + b * (x^2 + x + 1)) = (λ x : ℝ, x^2 - x + 1) := by
  sorry

-- Proof theorems for Problem (Ⅱ)
theorem generating_function_problem_II (h : ℝ → ℝ) :
  (∀ x : ℝ, h x = 2 * Real.log 2 x + Real.log (1/2) x) ∧ 
  (∀ t : ℝ, (∃ x ∈ set.Icc (2 : ℝ) 4, 3 * (h x)^2 + 2 * (h x) + t < 0) → t < -5) := by
  sorry

-- Proof theorems for Problem (Ⅲ)
theorem generating_function_problem_III :
  ∃ h : ℝ → ℝ, 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → h x = x + b / x) ∧ 
  (∀ x ∈ set.Icc (1 : ℝ) 10, h x ≥ b) ↔ (0 < b ∧ b ≤ 4) := by
  sorry

end generating_function_set1_no_generating_function_set2_generating_function_problem_II_generating_function_problem_III_l537_537587


namespace skitties_remainder_l537_537999

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 :=
sorry

end skitties_remainder_l537_537999


namespace improper_integral_converges_l537_537770

theorem improper_integral_converges : 
  ∫ x in 0..∞, (1 / (1 + x^2)) = Real.pi / 2 := 
sorry

end improper_integral_converges_l537_537770


namespace no_hyperdeficient_numbers_l537_537722

-- Defining the function g(n) which sums the squares of the divisors of n
def g (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum (λ d, d * d)

-- Defining the property of a hyperdeficient number
def is_hyperdeficient (n : ℕ) : Prop :=
  g (g n) = n^2 + 2

-- Stating the theorem
theorem no_hyperdeficient_numbers : ∀ n : ℕ, ¬ is_hyperdeficient n :=
by sorry

end no_hyperdeficient_numbers_l537_537722


namespace negation_of_proposition_l537_537319

open Classical

variable (x : ℝ)

def proposition (x : ℝ) := ln x - x + 1 ≤ 0

theorem negation_of_proposition :
  ¬(∀ x > 0, proposition x) ↔ ∃ x > 0, ln x - x + 1 > 0 := by
  sorry

end negation_of_proposition_l537_537319


namespace sampling_method_D_is_the_correct_answer_l537_537856

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer_l537_537856


namespace cosine_60_degrees_l537_537949

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l537_537949


namespace min_abs_value_sum_l537_537623

theorem min_abs_value_sum (x : ℚ) : (min (|x - 1| + |x + 3|) = 4) :=
sorry

end min_abs_value_sum_l537_537623


namespace gergonne_point_l537_537136

theorem gergonne_point (A B C M N K : Type*) [DecidableEq A] [DecidableEq B] [DecidableEq C]
  (in_triangle : (M ∈ line B C) ∧ (N ∈ line A B) ∧ (K ∈ line A C))
  (tangency : is_tangency_point B C M ∧ is_tangency_point A B N ∧ is_tangency_point A C K)
  (segments : connect_to_opposite_vertices A B C M N K) :
  concurrent_segments (A, B, C) (M, N, K) := 
sorry

end gergonne_point_l537_537136


namespace championship_outcome_count_l537_537875

theorem championship_outcome_count (students championships : ℕ) (h_students : students = 8) (h_championships : championships = 3) : students ^ championships = 512 := by
  rw [h_students, h_championships]
  norm_num

end championship_outcome_count_l537_537875


namespace movie_theorem_l537_537541

variables (A B C D : Prop)

theorem movie_theorem 
  (h1 : (A → B))
  (h2 : (B → C))
  (h3 : (C → A))
  (h4 : (D → B)) 
  : ¬D := 
by
  sorry

end movie_theorem_l537_537541


namespace binding_cost_is_correct_l537_537926

-- Definitions for the conditions used in the problem
def total_cost : ℝ := 250      -- Total cost to copy and bind 10 manuscripts
def copy_cost_per_page : ℝ := 0.05   -- Cost per page to copy
def pages_per_manuscript : ℕ := 400  -- Number of pages in each manuscript
def num_manuscripts : ℕ := 10      -- Number of manuscripts

-- The target value we want to prove
def binding_cost_per_manuscript : ℝ := 5 

-- The theorem statement proving the binding cost per manuscript
theorem binding_cost_is_correct :
  let copy_cost_per_manuscript := pages_per_manuscript * copy_cost_per_page
  let total_copy_cost := num_manuscripts * copy_cost_per_manuscript
  let total_binding_cost := total_cost - total_copy_cost
  (total_binding_cost / num_manuscripts) = binding_cost_per_manuscript :=
by
  sorry

end binding_cost_is_correct_l537_537926


namespace cos_60_eq_one_half_l537_537945

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537945


namespace find_n_l537_537673

theorem find_n (n : ℚ) (h : 11^(5 * n) = (1 / 11)^(2 * n - 36)) : n = 36 / 7 :=
by
  sorry

end find_n_l537_537673


namespace validate_quad_propositions_l537_537183

/-- Given a quadrilateral, check the validity of four propositions about it, and prove that exactly one of them is true. -/
theorem validate_quad_propositions :
  let P1 := ∀ (ABCD : Type) [quadrilateral ABCD],
    (∃ (a b : ABCD), opposite_sides_equal ABCD a b) →
    (∃ (a b : ABCD), opposite_angles_equal ABCD a b) →
    parallelogram ABCD
  let P2 := ∀ (ABCD : Type) [quadrilateral ABCD],
    (∃ (a b : ABCD), opposite_sides_equal ABCD a b) →
    (∃ (d1 d2 : ABCD), diagonal_bisects_other ABCD d1 d2) →
    parallelogram ABCD
  let P3 := ∀ (ABCD : Type) [quadrilateral ABCD],
    (∃ (a b : ABCD), opposite_angles_equal ABCD a b) →
    (∃ (d1 d2 : ABCD), diagonal_bisects_other ABCD d1 d2) →
    parallelogram ABCD
  let P4 := ∀ (ABCD : Type) [quadrilateral ABCD],
    (∃ (a b : ABCD), opposite_angles_equal ABCD a b) →
    (∃ (d1 d2 : ABCD), diagonal_is_bisected ABCD d1 d2) →
    parallelogram ABCD
  in P1 = false ∧ P2 = false ∧ P3 = true ∧ P4 = false :=
begin
  sorry
end

end validate_quad_propositions_l537_537183


namespace even_cycle_exists_l537_537769

variable {V : Type*} [Fintype V] (G : SimpleGraph V)

-- Define the hypothesis that every vertex has at least three edges.
def at_least_three_edges (v : V) : Prop :=
  3 ≤ G.degree v

-- Define the hypothesis that there are finitely many vertices.
def finite_graph : Prop :=
  Fintype V

-- The theorem to prove based on the conditions
theorem even_cycle_exists (h1 : finite_graph) (h2 : ∀ v, at_least_three_edges G v) :
  ∃ (c : List V), G.IsCycle c ∧ c.length % 2 = 0 :=
  sorry

end even_cycle_exists_l537_537769


namespace find_ω_range_l537_537595

open Real

noncomputable def has_five_zeros_in_interval {ω : ℝ} (hω : ω > 0) :=
  let f := λ x : ℝ, sin (ω * x) + sqrt 3 * cos (ω * x) in
  ∃ l : List ℝ, List.length l = 5 ∧ List.forall (λ x, 0 < x ∧ x < 4 * π ∧ f x = 0) l

theorem find_ω_range (ω : ℝ) (hω : ω > 0) :
  has_five_zeros_in_interval hω ↔ (7/6 < ω ∧ ω ≤ 17/12) :=
sorry

end find_ω_range_l537_537595


namespace find_CY_l537_537217

theorem find_CY (A B C Y D : Type) (A B C : Point)
  (BY_bisects_ABC : ∃ B Y, angle_bisector B A B C Y)
  (D_is_midpoint_AC : midpoint D A C)
  (BY_length : length B Y = 36)
  (BC_length : length B C = 15)
  (AC_length : length A C = 20) :
  length C Y = 24 :=
sorry

end find_CY_l537_537217


namespace value_of_expression_l537_537621

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem value_of_expression (h : a = Real.log 3 / Real.log 4) : 2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end value_of_expression_l537_537621


namespace problem_inequality_l537_537262

variables {a b c x1 x2 x3 x4 x5 : ℝ} 

theorem problem_inequality
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x1: 0 < x1) (h_pos_x2: 0 < x2) (h_pos_x3: 0 < x3) (h_pos_x4: 0 < x4) (h_pos_x5: 0 < x5)
  (h_sum_abc : a + b + c = 1) (h_prod_x : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1^2 + b * x1 + c) * (a * x2^2 + b * x2 + c) * (a * x3^2 + b * x3 + c) * 
  (a * x4^2 + b * x4 + c) * (a * x5^2 + b * x5 + c) ≥ 1 :=
sorry

end problem_inequality_l537_537262


namespace passenger_rides_each_car_l537_537135

noncomputable def prob_riding_each_car (prob_car_A : ℚ) (prob_car_B : ℚ) (rides : ℕ) : ℚ :=
  if rides = 2 then prob_car_A * prob_car_B else 0

theorem passenger_rides_each_car :
  let prob := 1 / 2 in
  prob_riding_each_car prob prob 2 = 1 / 4 :=
by
  let prob := 1 / 2
  have h1 : prob_riding_each_car prob prob 2 = prob * prob := by rfl
  have h2 : prob * prob = 1 / 4 := by norm_num
  rw [h1, h2]
  exact eq.refl _


end passenger_rides_each_car_l537_537135


namespace plane_perpendicular_l537_537604

variables (l : Line) (α β : Plane)

axiom lies_in_plane : l ⊆ α
axiom perpendicular_to_plane : l ⊥ β

theorem plane_perpendicular :
  α ⊥ β := 
sorry

end plane_perpendicular_l537_537604


namespace range_of_m_l537_537679

theorem range_of_m (x m : ℝ) (h1 : (2 * x + m) / (x - 1) = 1) (h2 : x ≥ 0) : m ≤ -1 ∧ m ≠ -2 :=
sorry

end range_of_m_l537_537679


namespace cos_60_eq_sqrt3_div_2_l537_537930

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l537_537930


namespace coords_A_l537_537340

def A : ℝ × ℝ := (1, -2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

def A' : ℝ × ℝ := reflect_y_axis A

def A'' : ℝ × ℝ := move_up A' 3

theorem coords_A'' : A'' = (-1, 1) := by
  sorry

end coords_A_l537_537340


namespace concurrency_condition_l537_537693

-- Define the entities: triangle, points, segments, concurrency
variables (A B C Q M N L M1 N1 L1 : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq Q] 
          [DecidableEq M] [DecidableEq N] [DecidableEq L] [DecidableEq M1] [DecidableEq N1] [DecidableEq L1]
variables (triangle_ABC : Prop) (on_side_BC : M) (on_side_CA : N) (on_side_AB : L)
          (inside_triangle : Q) (intersect_M1 : Prop) (intersect_N1 : Prop) (intersect_L1 : Prop)

theorem concurrency_condition :
  (on_side_BC = M) → (on_side_CA = N) → (on_side_AB = L) → (inside_triangle = Q) →
  (intersect_M1) → (intersect_N1) → (intersect_L1) →
  (concurrent (M1, M) (N1, N) (L1, L) ↔ concurrent (M, A) (N, B) (L, C)) :=
sorry

end concurrency_condition_l537_537693


namespace angle_FHP_eq_angle_BAC_l537_537551

open_locale classical
noncomputable theory

variables {A B C O H F P : Type*}
variables [triangle A B C] [acute_triangle A B C]
variables [circumcenter O A B C] [orthocenter H A B C]
variables [foot F C (line_through A B)] [perpendicular (line_through F O) P (line_through A C)]

theorem angle_FHP_eq_angle_BAC
  (h_acute : ∀ (A B C : Point), acute_triangle A B C)
  (h_bc_gt_ac : ∀ (A B C : Point), side_length B C > side_length A C)
  (h_circumcenter : ∀ (A B C : Point), circumcenter O A B C)
  (h_orthocenter : ∀ (A B C : Point), orthocenter H A B C)
  (h_foot : ∀ (C : Point), foot F C (line_through A B))
  (h_perp : ∀ (F O : Point), perpendicular (line_through F O) P (line_through A C)) :
  angle F H P = angle A B C :=
sorry

end angle_FHP_eq_angle_BAC_l537_537551


namespace sum_arithmetic_sequence_sum_reciprocal_arith_geom_seq_l537_537960

-- First problem statement
theorem sum_arithmetic_sequence (n : ℕ) (n_pos : 0 < n) (a : ℕ → ℤ) (h1 : a 1 = -2) (d : ℤ) (h2 : d = 2)
  (common_diff : ∀ k, a (k + 1) = a k + d) : (∑ k in Finset.range n, a (k + 1)) = n^2 - 3 * n :=
sorry

-- Second problem statement
theorem sum_reciprocal_arith_geom_seq (n : ℕ) (n_pos : 0 < n) (a : ℕ → ℕ) (b : ℕ → ℝ)
  (h1 : a 1 = 1) (h2 : a 2 = 2) (arith_seq : ∀ k, a (k + 1) = a k + 1)
  (geom_seq : ∀ k, b k = 2 ^ a k) : (∑ k in Finset.range n, (a (k + 1) / b (k + 1))) = (2^(n + 1) - n - 2) / 2^n :=
sorry

end sum_arithmetic_sequence_sum_reciprocal_arith_geom_seq_l537_537960


namespace triangle_ABC_properties_l537_537683

variables {a b c : ℝ} -- lengths of sides opposite to angles A, B, and C respectively
variables {A B C : ℝ} -- angles A, B, and C respectively
variables {AM : ℝ} -- length of the median AM

-- Conditions
axiom condition1 : a^2 - (b - c)^2 = (2 - real.sqrt 3) * b * c
axiom condition2 : real.sin A * real.sin B = real.cos( C / 2) ^ 2
axiom condition3 : AM = real.sqrt 7

-- Proof that angles A and B are π / 6 and the area is √3
theorem triangle_ABC_properties (ha : A = real.pi / 6) (hb : B = real.pi / 6) 
                                 (area_eq : real.sqrt 3 = real.sqrt 3) : 
                                 A = real.pi / 6 ∧ B = real.pi / 6 ∧ area_eq = real.sqrt 3 :=
by {
  sorry
}

end triangle_ABC_properties_l537_537683


namespace exists_polynomial_in_closed_finite_codimensional_subspace_l537_537006

noncomputable def continuous_function_space := { f : ℝ → ℝ // continuous f ∧ ∀ x, x ∈ (-1:ℝ) .. 1 }

def is_finite_codimensional_subspace (V : Set continuous_function_space) :=
  ∃ n : ℕ, ∃ G : Fin n → continuous_function_space, Set.Finite { f | ∃ c : ℝ, c • G f = 0 } 

theorem exists_polynomial_in_closed_finite_codimensional_subspace
  (V : Set continuous_function_space)
  (hV_closed : is_closed V)
  (hV_finite_codim : is_finite_codimensional_subspace V) :
  ∃ p : continuous_function_space, p ∈ V ∧ ∥p∥ ≤ 1 ∧ deriv p.val 0 > 2023 := 
sorry

end exists_polynomial_in_closed_finite_codimensional_subspace_l537_537006


namespace center_of_circumcircle_of_BOD_lies_on_AC_l537_537607

-- Define the given conditions as mentioned in (a)
variable (ABC : Triangle)
variable (angleB : ∠ ABC = 135)
variable (M : Point)
variable (isMidpoint : M = midpoint AC)
variable (O : Point)
variable (isCircumcenter : O = circumcenter ABC)
variable (D : Point)
variable (BMD_intersects_Omega : ray BM intersects_circle Ω at D )

-- State the theorem to be proven
theorem center_of_circumcircle_of_BOD_lies_on_AC :
  let Γ := circumcircle BOD in
  center Γ ∈ line AC :=
sorry

end center_of_circumcircle_of_BOD_lies_on_AC_l537_537607


namespace problem_statement_l537_537022

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem problem_statement : g (f (g (f 1))) = 305 :=
by
  sorry

end problem_statement_l537_537022


namespace ratio_solution_l537_537897

theorem ratio_solution (x : ℚ) : (1 : ℚ) / 3 = 5 / 3 / x → x = 5 := 
by
  intro h
  sorry

end ratio_solution_l537_537897


namespace triangle_problem_part1_triangle_problem_part2_l537_537702

theorem triangle_problem_part1
    (a b c A B C : ℝ)
    (h1 : ∀ (A B C : ℝ), a = 3*c)
    (h2 : ∀ (A B C : ℝ), 2 * cos(B / 2) ^ 2 = sqrt(3) * sin(B))
    :
    (B = π / 3) ∧ (tan C = sqrt(3) / 5) :=
by
  sorry

theorem triangle_problem_part2
    (a b c A B C : ℝ)
    (h1 : ∀ (A B C : ℝ), a = 3*c)
    (h2 : ∀ (A B C : ℝ), 2 * cos(B / 2) ^ 2 = sqrt(3) * sin(B))
    (h3: b = 1)
    (h4: B = π / 3)
    :
    (1 / 2 * a * c * sin(B) = (3 * sqrt(3)) / 28) :=
by
  sorry

end triangle_problem_part1_triangle_problem_part2_l537_537702


namespace altitude_sum_eq_l537_537732

variables {A B C D P : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]

noncomputable def h1 (A B P : Type) : ℝ := sorry -- Length of the altitude from P to AB
noncomputable def h2 (B C P : Type) : ℝ := sorry -- Length of the altitude from P to BC
noncomputable def h3 (C D P : Type) : ℝ := sorry -- Length of the altitude from P to CD
noncomputable def h4 (D A P : Type) : ℝ := sorry -- Length of the altitude from P to DA

theorem altitude_sum_eq (A B C D P : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]
  (h1_altitude : ℝ) (h2_altitude : ℝ) (h3_altitude : ℝ) (h4_altitude : ℝ)
  (h_perpendicular: ∀x y : Type, x ≠ y → orthogonal x y )  -- Diagonals are perpendicular
  (h_heights : h1 A B P = h1_altitude ∧ h2 B C P = h2_altitude ∧ h3 C D P = h3_altitude ∧ h4 D A P = h4_altitude)
  : (1 / h1_altitude^2) + (1 / h3_altitude^2) = (1 / h2_altitude^2) + (1 / h4_altitude^2) :=
by sorry

end altitude_sum_eq_l537_537732


namespace reflection_y_axis_correct_l537_537784

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l537_537784


namespace correct_statements_l537_537258

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

noncomputable def a_n_sequence (n : ℕ) := a n
noncomputable def Sn_sum (n : ℕ) := S n

axiom Sn_2022_lt_zero : S 2022 < 0
axiom Sn_2023_gt_zero : S 2023 > 0

theorem correct_statements :
  (a 1012 > 0) ∧ ( ∀ n, S n >= S 1011 → n = 1011) :=
  sorry

end correct_statements_l537_537258


namespace angle_AKC_eq_30_l537_537703

variable (A B C K : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K]

constant AC : ℝ
constant AB : ℝ
constant BC : ℝ
constant AKC : ℝ
constant angle_KAC_obtuse : Prop

axiom AC_val : AC = 2 * Real.sqrt 3
axiom AB_val : AB = Real.sqrt 7
axiom BC_val : BC = 1
axiom AKC_similar : Triangle.similar_ (triangle K A C) (triangle A B C)
axiom KAC_obtuse : angle_KAC_obtuse

theorem angle_AKC_eq_30 :
  ∃ (angle_AKC : ℝ), angle_AKC = 30 ∧ angle_AKC = AKC := sorry

end angle_AKC_eq_30_l537_537703


namespace emily_subtracts_99_l537_537088

theorem emily_subtracts_99 : ∀ (a b : ℕ), (51 * 51 = a + 101) → (49 * 49 = b - 99) → b - 99 = 2401 := by
  intros a b h1 h2
  sorry

end emily_subtracts_99_l537_537088


namespace expression_inside_absolute_value_l537_537576

theorem expression_inside_absolute_value (E : ℤ) (x : ℤ) (h1 : x = 10) (h2 : 30 - |E| = 26) :
  E = 4 ∨ E = -4 := 
by
  sorry

end expression_inside_absolute_value_l537_537576


namespace probability_slope_le_1_3_l537_537025

def unit_square := {p : ℝ × ℝ | 0 ≤ p.fst ∧ p.fst ≤ 1 ∧ 0 ≤ p.snd ∧ p.snd ≤ 1}

def point_Q := unit_square

def P := (3/4, 1/4 : ℝ × ℝ)

def line_slope (Q P : ℝ × ℝ) := 
  (Q.snd - P.snd) / (Q.fst - P.fst)

theorem probability_slope_le_1_3 : 
  (measure_theory.measure_space.measure (measure_theory.measure_space.volume) 
    ({Q : ℝ × ℝ | Q ∈ unit_square ∧ line_slope Q P ≤ 1/3} : set (ℝ × ℝ))) =
  (1/2 : ℝ) := 
sorry

end probability_slope_le_1_3_l537_537025


namespace trigonometric_propositions_l537_537050

theorem trigonometric_propositions : 
  let prop1 := ∀ x, (0 < x ∧ x < π / 2) → (tan x) > (tan (x - 1))
  let prop2 := ∀ x, cos (2 * (π / 4 - x)) = cos (2 * (π / 4 + x))
  let prop3 := ∀ x, 4 * sin (2 * x - π / 3) = -4 * sin (2 * (π / 6 - x) - π / 3)
  let prop4 := ∀ x, (-π / 2 ≤ x ∧ x ≤ π / 2) → (sin (x + π / 4) > sin (x + π / 4 - 1))
  prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 :=
by
  let prop1 := ∀ x, (0 < x ∧ x < π / 2) → (tan x) > (tan (x - 1))
  let prop2 := ∀ x, cos (2 * (π / 4 - x)) = cos (2 * (π / 4 + x))
  let prop3 := ∀ x, 4 * sin (2 * x - π / 3) = -4 * sin (2 * (π / 6 - x) - π / 3)
  let prop4 := ∀ x, (-π / 2 ≤ x ∧ x ≤ π / 2) → (sin (x + π / 4) > sin (x + π / 4 - 1))
  sorry

end trigonometric_propositions_l537_537050


namespace highway_total_vehicles_l537_537399

theorem highway_total_vehicles (num_trucks : ℕ) (num_cars : ℕ) (total_vehicles : ℕ)
  (h1 : num_trucks = 100)
  (h2 : num_cars = 2 * num_trucks)
  (h3 : total_vehicles = num_cars + num_trucks) :
  total_vehicles = 300 :=
by
  sorry

end highway_total_vehicles_l537_537399


namespace interval_monotonicity_minimum_value_range_of_a_l537_537290

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x

theorem interval_monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x, 0 < x ∧ x < a → f x a > 0) ∧ (∀ x, x > a → f x a < 0) :=
sorry

theorem minimum_value (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f x a ≥ 1) ∧ (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = 1) → a = 1 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 1 → f x a < 1 / 2 * x) → a < 1 / 2 :=
sorry

end interval_monotonicity_minimum_value_range_of_a_l537_537290


namespace tom_purchased_8_kg_of_apples_l537_537458

noncomputable def number_of_apples_purchased (price_per_kg_apple : ℤ) (price_per_kg_mango : ℤ) (kg_mangoes : ℤ) (total_paid : ℤ) : ℤ :=
  let total_cost_mangoes := price_per_kg_mango * kg_mangoes
  total_paid - total_cost_mangoes / price_per_kg_apple

theorem tom_purchased_8_kg_of_apples : 
  number_of_apples_purchased 70 65 9 1145 = 8 := 
by {
  -- Expand the definitions and simplify
  sorry
}

end tom_purchased_8_kg_of_apples_l537_537458


namespace units_digit_of_sequence_l537_537490

def factorial : ℕ → ℕ
| 0 := 1
| n + 1 := (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ :=
if n % 2 = 1 then factorial n + n else factorial n - n

def units_digit (n : ℕ) : ℕ := n % 10

lemma units_digit_of_sum (terms : List ℕ) : ℕ :=
units_digit (terms.sum)

theorem units_digit_of_sequence : units_digit_of_sum (List.map sequence_term [1, 2, 3, 4, 5, 6, 7, 8, 9]) = 5 :=
by
  sorry

end units_digit_of_sequence_l537_537490


namespace t_minus_s_l537_537154

noncomputable def t_s_diff : ℝ :=
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (list.sum class_sizes) / teachers
  let s := (∑ size in class_sizes, size * (size / students))
  t - s

theorem t_minus_s : t_s_diff = -13.5 := 
  by
	sorry

end t_minus_s_l537_537154


namespace beginner_wins_by_first_move_l537_537904

def initial_positions : (ℕ × ℕ) := (1, 30)

def valid_move (pos₁ pos₂ : ℕ × ℕ) : Bool :=
  let (player_pos, opponent_pos) := (pos₁, pos₂)
  player_pos.1 < opponent_pos.2 -- Ensure the player does not pass the opponent

def first_move_wins_strategy : Prop :=
  ∃ (pos₁ pos₂ : ℕ × ℕ), 
    (pos₁.1 = 2 ∧ pos₁.2 = 30) ∧
    (∀ next_pos₁ next_pos₂,
      valid_move (pos₁, pos₂) → 
      (valid_move (next_pos₁, next_pos₂) → 
      (next_pos₂.2 < next_pos₁.1 + 1 ∨ next_pos₂.2 < next_pos₁.1 + 2)))

theorem beginner_wins_by_first_move : first_move_wins_strategy := 
begin
    -- Proof is required here
    sorry
end

end beginner_wins_by_first_move_l537_537904


namespace bunches_with_new_distribution_l537_537557

-- Given conditions
def bunches_initial := 8
def flowers_per_bunch_initial := 9
def total_flowers := bunches_initial * flowers_per_bunch_initial

-- New condition and proof requirement
def flowers_per_bunch_new := 12
def bunches_new := total_flowers / flowers_per_bunch_new

theorem bunches_with_new_distribution : bunches_new = 6 := by
  sorry

end bunches_with_new_distribution_l537_537557


namespace age_of_15th_student_l537_537866

theorem age_of_15th_student (avg_age_15 : ℕ) (avg_age_6 : ℕ) (avg_age_8 : ℕ) (num_students_15 : ℕ) (num_students_6 : ℕ) (num_students_8 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_6 : avg_age_6 = 14) 
  (h_avg_8 : avg_age_8 = 16) 
  (h_num_15 : num_students_15 = 15) 
  (h_num_6 : num_students_6 = 6) 
  (h_num_8 : num_students_8 = 8) : 
  ∃ age_15th_student : ℕ, age_15th_student = 13 := 
by
  sorry


end age_of_15th_student_l537_537866


namespace balls_count_l537_537076

theorem balls_count (w r b : ℕ) (h_ratio : 4 * r = 3 * w ∧ 2 * w = 4 * b ∧ w = 20) : r = 15 ∧ b = 10 :=
by
  sorry

end balls_count_l537_537076


namespace probability_yellow_last_marble_l537_537182

theorem probability_yellow_last_marble :
  let P_A_white := 5 / 11,
      P_A_black := 6 / 11,
      P_B_yellow := 8 / 14,
      P_C_blue := 7 / 10,
      P_C_yellow := 3 / 10,
      P_D_yellow := 1 / 5 in
  let P_from_B := P_A_white * (8 / 14),
      P_to_D := P_A_black * (7 / 10) * (1 / 5),
      P_from_C := P_A_black * (3 / 10) in
  (P_from_B + P_to_D + P_from_C) = (136 / 275) :=
by
  sorry

end probability_yellow_last_marble_l537_537182


namespace sufficient_necessary_conditions_l537_537315

variable (A B C D : Prop)

#check
theorem sufficient_necessary_conditions :
  (A → B) ∧ ¬ (B → A) →
  (C → B) ∧ ¬ (B → C) →
  (D → C) ∧ ¬ (C → D) →
  ((B → A) ∧ ¬ (A → B)) ∧
  ((A → C) ∧ ¬ (C → A)) ∧
  (¬ (D → A) ∧ ¬ (A → D)) :=
by
  intros h1 h2 h3
  cases h1 with hA_implies_B hnB_implies_A
  cases h2 with hC_implies_B hnB_implies_C
  cases h3 with hD_implies_C hnC_implies_D
  split
  -- Proof for B → A and ¬A → B
  split
  { intros hB,
    apply false.elim,
    apply hnB_implies_A,
    exact hB, },
  { exact hnB_implies_A, },
  -- Proof for A → C and ¬C → A
  split
  { intros hA,
    apply hC_implies_B,
    apply hA_implies_B,
    exact hA, },
  { apply false.elim,
    intros hA,
    apply hnB_implies_A,
    exact hA, },
  -- Proof for ¬D → A and ¬A → D
  split
  { intros hD,
    apply false.elim,
    apply hnB_implies_A,
    exact hD, },
  { intros hA,
    apply false.elim,
    apply hnC_implies_D,
    exact hA, }

end sufficient_necessary_conditions_l537_537315


namespace baker_cakes_l537_537921

theorem baker_cakes (C : ℕ) (h1 : 154 = 78 + 76) (h2 : C = 78) : C = 78 :=
sorry

end baker_cakes_l537_537921


namespace problem_statement_l537_537739

noncomputable def f : ℕ → ℝ
| 1     := 0.5
| (n+1) := Real.sin ((Math.pi / 2) * f n)

theorem problem_statement (x : ℕ) (h : x ≥ 2) 
  : 1 - f x < (Math.pi / 4) * (1 - f (x - 1)) := sorry

end problem_statement_l537_537739


namespace square_area_is_correct_l537_537916

noncomputable def find_area_of_square (x : ℚ) : ℚ :=
  let side := 6 * x - 27
  side * side

theorem square_area_is_correct (x : ℚ) (h1 : 6 * x - 27 = 30 - 2 * x) :
  find_area_of_square x = 248.0625 :=
by
  sorry

end square_area_is_correct_l537_537916


namespace number_of_ordered_triples_l537_537152

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 1536)
(h4 : ∃ (a' b' c' : ℕ), (a', b', c') ≠ (a, b, c) ∧ (a' ∣ a ∧ b' ∣ b ∧ c' ∣ c) ∧ a' * c' = a * c) :
  ((card {t : ℕ × ℕ × ℕ | a ≤ b ∧ b ≤ c ∧ b = 1536 ∧ (∃ (a' b' c' : ℕ), (a', b', c') ≠ (a, b, c) ∧ (a' ∣ a ∧ b' ∣ b ∧ c' ∣ c) ∧ a' * c' = a * c)}).to_nat = 14) := sorry

end number_of_ordered_triples_l537_537152


namespace total_sales_calculation_l537_537162

def average_price_per_pair : ℝ := 9.8
def number_of_pairs_sold : ℕ := 70
def total_amount : ℝ := 686

theorem total_sales_calculation :
  average_price_per_pair * (number_of_pairs_sold : ℝ) = total_amount :=
by
  -- proof goes here
  sorry

end total_sales_calculation_l537_537162


namespace polar_coordinate_of_circle_range_of_distances_l537_537351

noncomputable def circle_in_cartesian : set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + (p.2 - 2)^2 = 1 }

noncomputable def line_in_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + (sqrt 3)/2 * t, (1/2) * t)

noncomputable def polar_coordinate_equation (rho theta : ℝ) : Prop :=
  rho^2 - 2 * rho * real.cos theta - 4 * rho * real.sin theta + 4 = 0

theorem polar_coordinate_of_circle  :
  ∀ (p : ℝ × ℝ), p ∈ circle_in_cartesian ↔ ∃ (rho theta : ℝ), 
  polar_coordinate_equation rho theta ∧ p = (rho * real.cos theta, rho * real.sin theta) :=
begin
  sorry
end

theorem range_of_distances (C : ℝ × ℝ) :
  C = (1,2) →
  ∀ t : ℝ, let l := line_in_parametric t,
  ∃ d : ℝ, sqrt 3 - 1 ≤ d ∧ d ≤ sqrt 3 + 1 ↔ 
  ∃ d : ℝ, let l := line_in_parametric t in
  d = abs (1 - 2 * sqrt 3 - 1) / sqrt (1^2 + (-sqrt 3)^2) :=
begin
  sorry
end

end polar_coordinate_of_circle_range_of_distances_l537_537351


namespace angle_between_vectors_l537_537245

open Real

variables {a b : ℝ^2} 

theorem angle_between_vectors (h₁ : ∥a∥ = 1) (h₂ : ∥b∥ = sqrt 2) (h₃ : dot_product a (a - b) = 0) : angle a b = π / 4 :=
sorry

end angle_between_vectors_l537_537245


namespace zeros_in_expansion_l537_537664

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end zeros_in_expansion_l537_537664


namespace candies_left_to_share_l537_537712

def initial_candies : ℕ := 100
def siblings : ℕ := 3
def candies_per_sibling : ℕ := 10
def candies_josh_eats : ℕ := 16

theorem candies_left_to_share : 
  let candies_given_to_siblings := siblings * candies_per_sibling in
  let candies_after_siblings := initial_candies - candies_given_to_siblings in
  let candies_given_to_friend := candies_after_siblings / 2 in
  let candies_after_friend := candies_after_siblings - candies_given_to_friend in
  candies_after_friend - candies_josh_eats = 19 :=
by 
  sorry

end candies_left_to_share_l537_537712


namespace number_of_pens_bought_l537_537893

theorem number_of_pens_bought 
  (P : ℝ) -- Marked price of one pen
  (N : ℝ) -- Number of pens bought
  (discount : ℝ := 0.01)
  (profit_percent : ℝ := 29.130434782608695)
  (Total_Cost := 46 * P)
  (Selling_Price_per_Pen := P * (1 - discount))
  (Total_Revenue := N * Selling_Price_per_Pen)
  (Profit := Total_Revenue - Total_Cost)
  (actual_profit_percent := (Profit / Total_Cost) * 100) :
  actual_profit_percent = profit_percent → N = 60 := 
by 
  intro h
  sorry

end number_of_pens_bought_l537_537893


namespace tangent_line_exists_l537_537066

theorem tangent_line_exists :
  ∃ (k : ℝ), (∀ x y : ℝ, y = k * (x - 2) ∧ y = 1 / x → x + y - 2 = 0) :=
by
  -- Definitions based on the problem conditions
  let p := (2, 0) : ℝ × ℝ
  let curve := λ x : ℝ, 1 / x
  let tangent := λ k x : ℝ, k * (x - 2)

  -- The proof would go here
  sorry

end tangent_line_exists_l537_537066


namespace domain_of_f_l537_537099

def quadratic (x : ℝ) : ℝ := x^2 - 3 * x - 4

def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (quadratic x)

theorem domain_of_f :
  ∀ x : ℝ, 0 < quadratic x ↔ (x < -1 ∨ 4 < x) := sorry

end domain_of_f_l537_537099


namespace part1_part2_l537_537741

theorem part1 
(a : ℝ) (P Q : set ℝ) 
(hP : P = {x | (x - a) * (x + 1) > 0}) 
(hQ : Q = {x | |x - 1| ≤ 1})
(h_a_eq_1 : a = 1) :
  (set.compl P ∪ Q) = {x | -1 ≤ x ∧ x ≤ 2} := 
by 
  sorry

theorem part2 
(a : ℝ) (P Q : set ℝ) 
(hP : P = {x | (x - a) * (x + 1) > 0}) 
(hQ : Q = {x | 0 ≤ x ∧ x ≤ 2})
(hP_inter_Q_empty : ∀ x, x ∈ P → x ∉ Q) 
(h_a_gt_0 : a > 0) :
  a ∈ set.Ioi 2 :=
by 
  sorry

end part1_part2_l537_537741


namespace eighth_odd_multiple_of_5_is_75_l537_537483

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l537_537483


namespace fifth_plot_difference_l537_537159

-- Define the dimensions of the plots
def plot_width (n : Nat) : Nat := 3 + 2 * (n - 1)
def plot_length (n : Nat) : Nat := 4 + 3 * (n - 1)

-- Define the number of tiles in a plot
def tiles_in_plot (n : Nat) : Nat := plot_width n * plot_length n

-- The main theorem to prove the required difference
theorem fifth_plot_difference :
  tiles_in_plot 5 - tiles_in_plot 4 = 59 := sorry

end fifth_plot_difference_l537_537159


namespace learning_method_is_analogical_thinking_l537_537334

theorem learning_method_is_analogical_thinking
  (compare_objects : ∀ {α β : Type}, α → β → Prop)
  (find_similarities : ∀ {α : Type}, α → α → Prop)
  (deduce_similar_properties : ∀ {α : Type}, α → α → Prop) :
  ∃ (idea : Type), idea = "Analogical thinking" :=
by
  sorry

end learning_method_is_analogical_thinking_l537_537334


namespace eighth_odd_multiple_of_5_is_75_l537_537486

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l537_537486


namespace volume_of_cone_l537_537810

theorem volume_of_cone (l h : ℝ) (l_pos : l = 15) (h_pos : h = 9) : 
  let r := Real.sqrt (l^2 - h^2) in
  let V := (1 / 3) * Real.pi * r^2 * h in
  V = 432 * Real.pi :=
by
  -- Definitions derived from the conditions
  have r_def : r = Real.sqrt (15^2 - 9^2), from sorry,
  have V_def : V = (1 / 3) * Real.pi * (Real.sqrt (15^2 - 9^2))^2 * 9, from sorry,
  -- Proof of the theorem
  sorry

end volume_of_cone_l537_537810


namespace will_games_l537_537046

theorem will_games (earnings blades_percent game_cost tax_percent remaining_money num_games : ℝ) (h1 : earnings = 180) (h2 : blades_percent = 0.35) (h3 : game_cost = 12.50) (h4 : tax_percent = 0.05) (spent : ℝ) (cost_with_tax : ℝ) (h5 : spent = earnings * blades_percent) (h6 : remaining_money = earnings - spent) (h7 : cost_with_tax = game_cost * (1 + tax_percent)) :
  num_games = floor (remaining_money / cost_with_tax) → num_games = 8 :=
by
  intros h
  sorry

end will_games_l537_537046


namespace integral_f_l537_537735

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then x^2 else 
if 1 < x ∧ x ≤ 2 then 2 - x else 
0

theorem integral_f : ∫ x in 0..2, f x = 5 / 6 := by
  sorry

end integral_f_l537_537735


namespace arithmetic_sequence_f_multiple_of_18_l537_537252

open Nat

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (h : ∀ n, 3 * S n = 4 * a n - 4^(n+1) - 4):
  (b = λ n, 3 * n + 2) → (∀ n, b n = 3 * (n - 1) + b 1) := 
sorry

theorem f_multiple_of_18 (a : ℕ → ℕ) (f : ℕ → ℕ) (h : ∀ n, a n = (3 * n + 2) * 4^n):
  (f = λ n, (3 * n + 2) * 4^n - 2) → (∀ n, 18 ∣ f n) :=
sorry

end arithmetic_sequence_f_multiple_of_18_l537_537252


namespace candies_left_to_share_l537_537710

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end candies_left_to_share_l537_537710


namespace cos_of_angle_B_l537_537255

theorem cos_of_angle_B (A B C a b c : Real) (h₁ : A - C = Real.pi / 2) (h₂ : 2 * b = a + c) 
  (h₃ : 2 * a * Real.sin A = 2 * b * Real.sin B) (h₄ : 2 * c * Real.sin C = 2 * b * Real.sin B) :
  Real.cos B = 3 / 4 := by
  sorry

end cos_of_angle_B_l537_537255


namespace class_avg_difference_l537_537156

theorem class_avg_difference :
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (class_sizes.sum : ℝ) / (teachers : ℝ)
  let s := (class_sizes.map (λ n, n * (n / students : ℝ))).sum
  in 
  t - s = -13.5 :=
by
  -- Definitions
  let students := 100
  let teachers := 5
  let class_sizes := [50, 20, 20, 5, 5]
  let t := (class_sizes.sum : ℝ) / (teachers : ℝ)
  let s := (class_sizes.map (λ n, n * (n / students : ℝ))).sum
  -- Equation to prove
  show t - s = -13.5,
  sorry

end class_avg_difference_l537_537156


namespace houses_in_block_l537_537526

theorem houses_in_block (junk_per_house : ℕ) (total_junk : ℕ) (h_junk : junk_per_house = 2) (h_total : total_junk = 14) :
  total_junk / junk_per_house = 7 := by
  sorry

end houses_in_block_l537_537526


namespace matha_ends_with_79_cards_l537_537397

theorem matha_ends_with_79_cards (initial_cards : ℕ) (additional_cards : ℕ) : initial_cards = 3 → additional_cards = 76 → initial_cards + additional_cards = 79 :=
by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end matha_ends_with_79_cards_l537_537397


namespace min_obtuse_triangulation_l537_537838

def regular_polygon (n : ℕ) : Prop :=
  n ≥ 5

def obtuse_triangle (t : triangle) : Prop :=
  ∀ a b c : ℕ, a + b + c = 180 → (a > 90 ∨ b > 90 ∨ c > 90)

def triangulation (polygon : polygon) (m : ℕ) :=
  ∀ (triangles : list triangle), list.length triangles = m ∧ (∀ t ∈ triangles, obtuse_triangle t)

theorem min_obtuse_triangulation (n : ℕ) (h : regular_polygon n) : ∃ m : ℕ, triangulation (regular_polygon n) m ∧ m = n :=
by
  sorry

end min_obtuse_triangulation_l537_537838


namespace prime_divides_binom_l537_537417

-- We define that n is a prime number.
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Lean statement for the problem
theorem prime_divides_binom {n k : ℕ} (h₁ : is_prime n) (h₂ : 0 < k) (h₃ : k < n) :
  n ∣ Nat.choose n k :=
sorry

end prime_divides_binom_l537_537417


namespace math_problems_l537_537268

variable (a b : ℝ → ℝ)
variable (norm_a : ℝ) (norm_b : ℝ) (angle_ab : ℝ)

-- Given conditions
noncomputable def conditions : Prop :=
  norm_a = 3 ∧ norm_b = 4 ∧ angle_ab = Real.pi / 3

-- Proof statement for problem (1)
noncomputable def problem1 : Prop :=
  (3 * a - 2 * b) ⬝ (a - 2 * b) = 43

-- Proof statement for problem (2)
noncomputable def problem2 : Prop :=
  Real.norm (a - b) = Real.sqrt 13

-- Stating the theorem considering the given conditions
theorem math_problems
  (h : conditions a b norm_a norm_b angle_ab) :
  problem1 a b ∧ problem2 a b :=
sorry

end math_problems_l537_537268


namespace mappings_with_property_P_l537_537962

def vec := ℝ × ℝ

def f1 : vec → ℝ := λ (m : vec), m.1 - m.2
def f2 : vec → ℝ := λ (m : vec), m.1 * m.1 + m.2
def f3 : vec → ℝ := λ (m : vec), m.1 + m.2 + 1

def has_property_P (f : vec → ℝ) : Prop :=
  ∀ (a b : vec) (λ : ℝ), f (λ • a + (1 - λ) • b) = λ • f a + (1 - λ) • f b

theorem mappings_with_property_P :
  (has_property_P f1) ∧ ¬(has_property_P f2) ∧ (has_property_P f3) := by
  sorry

end mappings_with_property_P_l537_537962


namespace find_original_number_l537_537674

theorem find_original_number (c : ℝ) (h₁ : c / 12.75 = 16) (h₂ : 2.04 / 1.275 = 1.6) : c = 204 :=
by
  sorry

end find_original_number_l537_537674


namespace seventy_times_reciprocal_l537_537313

theorem seventy_times_reciprocal (x : ℚ) (hx : 7 * x = 3) : 70 * (1 / x) = 490 / 3 :=
by 
  sorry

end seventy_times_reciprocal_l537_537313


namespace correct_conclusions_l537_537272

variables {ℝ : Type*} [linear_ordered_field ℝ]

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * f(x + 1) - 2

lemma condition1 : ∀ x : ℝ, 2 * f(x) + g(x - 3) = 2 :=
sorry

lemma symmetric_f : ∀ x : ℝ, f(2 - x) = f(x + 2) :=
sorry

lemma value_at_1 : f(1) = 3 :=
sorry

theorem correct_conclusions : g(0) = 4 ∧ (∀ x : ℝ, g(x + 4) = g(x)) ∧ g(3) = 0 :=
by
  sorry

end correct_conclusions_l537_537272


namespace price_of_second_oil_l537_537874

theorem price_of_second_oil : 
  ∃ x : ℝ, 
    (10 * 50 + 5 * x = 15 * 56) → x = 68 := by
  sorry

end price_of_second_oil_l537_537874


namespace initial_velocity_is_three_l537_537178

-- Define the displacement function s(t)
def s (t : ℝ) : ℝ := 3 * t - t ^ 2

-- Define the initial time condition
def initial_time : ℝ := 0

-- State the main theorem about the initial velocity
theorem initial_velocity_is_three : (deriv s) initial_time = 3 :=
by
  sorry

end initial_velocity_is_three_l537_537178


namespace lego_set_cost_l537_537832

-- Define the cost per doll and number of dolls
def costPerDoll : ℝ := 15
def numberOfDolls : ℝ := 4

-- Define the total amount spent on the younger sister's dolls
def totalAmountOnDolls : ℝ := numberOfDolls * costPerDoll

-- Define the number of lego sets
def numberOfLegoSets : ℝ := 3

-- Define the total amount spent on lego sets (needs to be equal to totalAmountOnDolls)
def totalAmountOnLegoSets : ℝ := 60

-- Define the cost per lego set that we need to prove
def costPerLegoSet : ℝ := 20

-- Theorem to prove that the cost per lego set is $20
theorem lego_set_cost (h : totalAmountOnLegoSets = totalAmountOnDolls) : 
  totalAmountOnLegoSets / numberOfLegoSets = costPerLegoSet := by
  sorry

end lego_set_cost_l537_537832


namespace smallest_base10_integer_l537_537104

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l537_537104


namespace problem_statement_l537_537630

variable {f : ℝ → ℝ}

theorem problem_statement 
  (h1 : ∀ x ≥ 0, f' x < f x / (x + 1)) 
  : 2 * f 1 > f 3 := 
sorry

end problem_statement_l537_537630


namespace cos_60_eq_one_half_l537_537944

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l537_537944


namespace largest_possible_distance_l537_537487

def point : Type := ℝ × ℝ × ℝ

structure Sphere :=
  (center : point)
  (radius : ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

def sphere1 : Sphere := { center := (5, -3, 10), radius := 24 }
def sphere2 : Sphere := { center := (-7, 15, -20), radius := 44 }

theorem largest_possible_distance :
  ∃ (p1 p2 : point), distance sphere1.center p1 = sphere1.radius ∧ distance sphere2.center p2 = sphere2.radius ∧ distance p1 p2 = 105 :=
sorry

end largest_possible_distance_l537_537487


namespace probability_rain_90_percent_l537_537058

theorem probability_rain_90_percent (P : ℕ → Prop) (city : Type) (rain : city → Prop) : 
  (∀ t, 0.9 ≤ P t) → 
  (∀ p : city, (rain p → P 1) → (∃ p : city, rain p)) :=
sorry

end probability_rain_90_percent_l537_537058


namespace cubic_roots_l537_537774

variable (p q : ℝ)

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem cubic_roots (y z : ℂ) (h1 : -3 * y * z = p) (h2 : y^3 + z^3 = q) :
  ∃ (x1 x2 x3 : ℂ),
    (x^3 + p * x + q = 0) ∧
    (x1 = -(y + z)) ∧
    (x2 = -(ω * y + ω^2 * z)) ∧
    (x3 = -(ω^2 * y + ω * z)) :=
by
  sorry

end cubic_roots_l537_537774


namespace chord_length_eq_sqrt3_l537_537525

-- Given conditions and definitions
def line_through_point_with_inclination (a b : ℝ) (θ : ℝ) : ℝ → ℝ :=
  λ x, (Real.tan θ) * (x - a) + b

def circle_eq (h k r : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Specific problem statement
def line : ℝ → ℝ := line_through_point_with_inclination 1 0 (Real.pi / 6)
def circle : ℝ → ℝ → Prop := circle_eq 2 0 1

theorem chord_length_eq_sqrt3 : 
  ∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧ line A.1 = A.2 ∧ line B.1 = B.2 ∧ ∥A - B∥ = Real.sqrt 3 :=
sorry

end chord_length_eq_sqrt3_l537_537525


namespace solve_inequality_l537_537054

theorem solve_inequality :
  { x : ℝ // 10 * x^2 - 2 * x - 3 < 0 } =
  { x : ℝ // (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 } :=
by
  sorry

end solve_inequality_l537_537054


namespace geometric_locus_of_points_l537_537611

variable {A B C P : Point}

-- Define the height from C to AB (denoted by hC).
def height_from_C_to_AB (A B C : Point) : ℝ := distance (perpendicular_from C to_line (line_through A B)) C

-- Define the centroid of the triangle ABC.
def centroid (A B C : Point) : Point := 
  Point.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3)

-- Define the conditions for part (a)
def condition_a (A B C P : Point) : Prop := 
  distance (perpendicular_from P to_line (line_through A B)) (line_through A B) = height_from_C_to_AB A B C

-- Define the conditions for part (b)
def condition_b (A B C P : Point) : Prop :=
  line_through A P ∥ line_through B C ∨ 
  (P = Point.mk ((B.x + C.x) / 2) ((B.y + C.y) / 2))

-- Define the conditions for part (c)
def condition_c (A B C P : Point) : Prop :=
  P = centroid A B C ∨ P = A ∨ P = B ∨ P = C

-- Main Lean theorem statement
theorem geometric_locus_of_points (A B C : Point) :
  (∃ P : Point, condition_a A B C P) ∧ 
  (∃ P : Point, condition_b A B C P) ∧ 
  (∃ P : Point, condition_c A B C P) :=
by
  sorry

end geometric_locus_of_points_l537_537611


namespace find_length_of_side_c_l537_537355

noncomputable def length_of_side_c (a b A : ℝ) (hA : A = 30 * real.pi / 180) (ha : a = real.sqrt 2) (hb : b = real.sqrt 6) : ℝ :=
c

theorem find_length_of_side_c (a b c : ℝ) (A : ℝ) (hA : A = 30 * real.pi / 180) (ha : a = real.sqrt 2) (hb : b = real.sqrt 6) :
c = real.sqrt 2 ∨ c = 2 * real.sqrt 2 :=
sorry

end find_length_of_side_c_l537_537355


namespace range_of_g_l537_537264

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x ^ (k / 2)

theorem range_of_g (k : ℝ) (h : k < 0) : set.range (λ x : set.Icc (0 : ℝ) 1, g x k) = set.Ioo (1 : ℝ) ∞ :=
by
  sorry

end range_of_g_l537_537264


namespace shortest_time_to_complete_task_l537_537796

def wall_dimensions := (length : ℝ, width : ℝ, height : ℝ)
def normal_completion_time := 26.0
def break_time_per_break := 0.5
def number_of_breaks := 6
def increased_work_rate_factor := 1.25
def minimum_work_time_between_breaks := 0.75

noncomputable def total_break_time := number_of_breaks * break_time_per_break
noncomputable def effective_work_time_saved := (number_of_breaks + 1) * (increased_work_rate_factor - 1)
noncomputable def adjusted_work_time := normal_completion_time + total_break_time - effective_work_time_saved

theorem shortest_time_to_complete_task (T : ℝ) :
  wall_dimensions = (50, 0.25, 2) →
  T = 27.25 :=
by
  intros h_dim
  have h_normal_completion := normal_completion_time
  have h_total_break_time := total_break_time
  have h_effective_work_time_saved := effective_work_time_saved
  have h_adjusted_work_time := adjusted_work_time
  sorry

end shortest_time_to_complete_task_l537_537796


namespace clothing_probability_l537_537335

open Nat

theorem clothing_probability :
  let total_clothing := 5 + 7 + 8
  let total_ways := Nat.choose total_clothing 4
  let shirt_ways := Nat.choose 5 2
  let shorts_ways := Nat.choose 7 1
  let socks_ways := Nat.choose 8 1
  let desired_ways := shirt_ways * shorts_ways * socks_ways
  desired_ways / total_ways = 112 / 969 :=
by 
  let total_clothing := 20
  let total_ways := Nat.choose total_clothing 4
  let shirt_ways := Nat.choose 5 2
  let shorts_ways := Nat.choose 7 1
  let socks_ways := Nat.choose 8 1
  let desired_ways := shirt_ways * shorts_ways * socks_ways
  have : total_ways = 4845 := Nat.choose_eq 20 4 sorry
  have : desired_ways = 560 := sorry
  show desired_ways / total_ways = 112 / 969
  linarith

end clothing_probability_l537_537335


namespace variance_scaled_transformation_l537_537030

/-- Define a binomial random variable X with parameters n and p -/
def X : Type := ℕ → Type

/-- The variance of a binomial random variable given n and p -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ :=
  n * p * (1 - p)

/-- Prove that given X ~ Binomial(10, 0.8), D(2X + 1) = 6.4 -/
theorem variance_scaled_transformation :
  let n := 10 in
  let p := 0.8 in
  binomial_variance n p = 1.6 →
  4 * binomial_variance n p = 6.4 :=
by
  intros h
  sorry

end variance_scaled_transformation_l537_537030


namespace range_of_m_l537_537266

variable (x m : ℝ)

def p : Prop := -2 ≤ x ∧ x ≤ 10
def q : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m (h : p → q) : 9 ≤ m := 
sorry

end range_of_m_l537_537266


namespace cone_volume_is_correct_l537_537814

-- Given conditions
def slant_height : ℝ := 15
def height : ℝ := 9

-- Derived quantities
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- Volume formula for a cone
def volume_cone (r h : ℝ) := (1 / 3) * π * r^2 * h

theorem cone_volume_is_correct : volume_cone radius height = 432 * π := by
  sorry

end cone_volume_is_correct_l537_537814


namespace find_velocity_of_current_l537_537531

-- Define the conditions given in the problem
def rowing_speed_in_still_water : ℤ := 10
def distance_to_place : ℤ := 48
def total_travel_time : ℤ := 10

-- Define the primary goal, which is to find the velocity of the current given the conditions
theorem find_velocity_of_current (v : ℤ) 
  (h1 : rowing_speed_in_still_water = 10)
  (h2 : distance_to_place = 48)
  (h3 : total_travel_time = 10) 
  (h4 : rowing_speed_in_still_water * 2 + v * 0 = 
   rowing_speed_in_still_water - v) :
  v = 2 := 
sorry

end find_velocity_of_current_l537_537531


namespace regular_polygon_sides_l537_537982

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l537_537982


namespace units_digit_is_seven_l537_537794

-- Defining the structure of the three-digit number and its properties
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def four_times_original (a b c : ℕ) : ℕ := 4 * original_number a b c
def subtract_reversed (a b c : ℕ) : ℕ := four_times_original a b c - reversed_number a b c

-- Theorem statement: Given the condition, what is the units digit of the result?
theorem units_digit_is_seven (a b c : ℕ) (h : a = c + 3) : (subtract_reversed a b c) % 10 = 7 :=
by
  sorry

end units_digit_is_seven_l537_537794


namespace ratio_quadrilateral_to_triangle_l537_537694

noncomputable def midpoint (A B : ℝ×ℝ) : ℝ×ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def isosceles_triangle (A B C : ℝ×ℝ) : Prop :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def ratio_of_areas (A D E X B C : ℝ×ℝ) : ℝ :=
  (area A E X D) / (area B X C)

theorem ratio_quadrilateral_to_triangle (A B C D E X : ℝ×ℝ)
  (h_iso : isosceles_triangle A B C)
  (h_AB_AC : dist A B = 10 ∧ dist A C = 10)
  (h_BC : dist B C = 12)
  (h_midpoints : D = midpoint A B ∧ E = midpoint A C)
  (h_intersect : X = intersection (line_through C D) (line_through B E)) :
  ratio_of_areas A D E X B C = 1 :=
by
  sorry

end ratio_quadrilateral_to_triangle_l537_537694


namespace combinatorial_distance_problem_l537_537553

theorem combinatorial_distance_problem (n : ℕ) (x : ℝ) (x_i : ℕ → ℝ) :
  ∃ (S : finset ℝ), (S.card = nat.choose n (n/2)) ∧ (∀ y ∈ S, |x - y| < 1) :=
sorry

end combinatorial_distance_problem_l537_537553


namespace larry_wins_probability_l537_537366

noncomputable def probability_larry_wins (p_L : ℚ) (p_J : ℚ) : ℚ :=
  let q_L := 1 - p_L
  let q_J := 1 - p_J
  let r := q_L * q_J
  p_L / (1 - r)

theorem larry_wins_probability
  (p_L : ℚ) (p_J : ℚ) (h1 : p_L = 3 / 5) (h2 : p_J = 1 / 3) :
  probability_larry_wins p_L p_J = 9 / 11 :=
by 
  sorry

end larry_wins_probability_l537_537366


namespace cos_of_60_degrees_is_half_l537_537956

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537956


namespace smallest_possible_a_l537_537056

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + ↑c

theorem smallest_possible_a
  (a b c : ℕ)
  (r s : ℝ)
  (h_arith_seq : b - a = c - b)
  (h_order_pos : 0 < a ∧ a < b ∧ b < c)
  (h_distinct : r ≠ s)
  (h_rs_2017 : r * s = 2017)
  (h_fr_eq_s : f a b c r = s)
  (h_fs_eq_r : f a b c s = r) :
  a = 1 := sorry

end smallest_possible_a_l537_537056


namespace quadratic_properties_l537_537590

def quadratic_function (a b c x : ℝ) : ℝ :=
  a*x^2 + b*x + c

theorem quadratic_properties :
  let f := fun (x : ℝ) => quadratic_function (-6) 36 (-30) x in
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 24) :=
by {
  sorry
}

end quadratic_properties_l537_537590


namespace sum_of_consecutive_negative_integers_with_product_3080_l537_537805

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l537_537805


namespace non_parallel_planes_if_perpendicular_lines_and_planes_l537_537265

-- Definitions and assumptions based on the given problem
variables (m n : Line)
variables (α β : Plane)

-- Conditions
axiom lines_non_coincident : m ≠ n
axiom planes_non_coincident : α ≠ β

-- Statement to prove
theorem non_parallel_planes_if_perpendicular_lines_and_planes
  (H1 : m ⊥ α) (H2 : m ⊥ β) : α ∥ β := sorry

end non_parallel_planes_if_perpendicular_lines_and_planes_l537_537265


namespace minimum_cubes_l537_537887

-- Defining conditions as predicates
def shares_face_with_another_cube (figure : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∀ (x y z : ℕ), figure x y z → 
  ((figure (x+1) y z) ∨ (figure (x-1) y z) ∨ (figure x (y+1) z) ∨ (figure x (y-1) z) ∨ (figure x y (z+1)) ∨ (figure x y (z-1)))

def front_view_match (figure : ℕ → ℕ → ℕ → Prop) : Prop :=
  -- Assuming specific positions for the depiction of the front view
  ∀ (x z : ℕ), (figure 0 x z → x = 0 ∨ x = 1) ∧ (figure 1 x z → x = 1)

def side_view_match (figure : ℕ → ℕ → ℕ → Prop) : Prop :=
  -- Assuming specific positions for the depiction of the side view
  ∀ (z y : ℕ), (figure z 0 y → y = 0 ∨ y = 1) ∧ (figure z 1 y → y = 1)

-- Stating the main theorem
theorem minimum_cubes (figure : ℕ → ℕ → ℕ → Prop) : 
  shares_face_with_another_cube figure →
  front_view_match figure →
  side_view_match figure →
  (∃ n, (∀ (figure' : ℕ → ℕ → ℕ → Prop), shares_face_with_another_cube figure' →
   front_view_match figure' → side_view_match figure' → n ≤ (count figure')) ∧ count figure = 4) := sorry

end minimum_cubes_l537_537887


namespace find_extrema_of_f_l537_537579

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^2 + 5) / (x^2 + 1)^2

theorem find_extrema_of_f :
  (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) ∧ (∀ x : ℝ, f x ≥ 0.95) ∧ (∃ x : ℝ, f x = 0.95) :=
by {
  sorry
}

end find_extrema_of_f_l537_537579


namespace winning_configurations_for_blake_l537_537917

def isWinningConfigurationForBlake (config : List ℕ) := 
  let nimSum := config.foldl (xor) 0
  nimSum = 0

theorem winning_configurations_for_blake :
  (isWinningConfigurationForBlake [8, 2, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 3, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 5, 2]) :=
by {
  sorry
}

end winning_configurations_for_blake_l537_537917


namespace alpha_interval_l537_537627

open Real

theorem alpha_interval (α : ℝ) 
  (h1 : sin α - cos α > 0) 
  (h2 : tan α > 0) 
  (h3 : α ∈ Ico 0 (2 * π)) : 
  α ∈ Ioo (π / 4) (π / 2) ∪ Ioo π (5 * π / 4) :=
by
  sorry

end alpha_interval_l537_537627


namespace find_b2_l537_537077

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 15) (h9 : b 9 = 105)
  (h_avg : ∀ n, n ≥ 3 → b n = (∑ i in Finset.range (n - 1), b (i + 1)) / (n - 1)) :
  b 2 = 195 :=
by
  sorry

end find_b2_l537_537077


namespace hyperbola_eccentricity_l537_537292

noncomputable def eccentricity (x y a b : ℝ) [h_pos_a : 0 < a] [h_pos_b : 0 < b] : Prop :=
  let c := real.sqrt (a^2 + b^2)
  (𝔹 (a^2 b^2) ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 ->
    (a * real.sqrt (c^2 / a^2)) = a^2 + b^2)

theorem hyperbola_eccentricity (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_cond : ∀ (m n : ℝ), (∀ x y, x^2/a^2 - y^2/b^2 = 1) ->
    ∃ c, - (3/4) * c^2 ≤ a^2 - c^2 ∧ a^2 - c^2 ≤ - (1/2) * c^2) :
  let c := real.sqrt (a^2 + b^2)
  sqrt(2) ≤ c / a ∧ c / a ≤ 2 :=
begin 
  sorry
end

end hyperbola_eccentricity_l537_537292


namespace correctness_of_statements_l537_537316

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end correctness_of_statements_l537_537316


namespace a_minus_b_greater_than_one_l537_537753

open Real

theorem a_minus_b_greater_than_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (f_has_three_roots : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ (Polynomial.aeval r1 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r2 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r3 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0)
  (g_no_real_roots : ∀ (x : ℝ), (2*x^2 + 2*b*x + a) ≠ 0) :
  a - b > 1 := by
  sorry

end a_minus_b_greater_than_one_l537_537753


namespace raghu_investment_l537_537122

theorem raghu_investment (R : ℝ) 
  (h1 : ∀ T : ℝ, T = 0.9 * R) 
  (h2 : ∀ V : ℝ, V = 0.99 * R) 
  (h3 : R + 0.9 * R + 0.99 * R = 6069) : 
  R = 2100 := 
by
  sorry

end raghu_investment_l537_537122


namespace eccentricity_range_l537_537646

-- Definitions for the problem
def hyperbola_eq (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def circle_eq (x y : ℝ) : Prop :=
  (x + 4)^2 + y^2 = 8

def condition_asymptotes_not_intersect_circle (a b : ℝ) : Prop :=
  2 * b^2 > (a^2 + b^2) 

-- The statement we need to prove
theorem eccentricity_range (a b : ℝ) (h1: 0 < a) (h2: 0 < b)
    (h3: condition_asymptotes_not_intersect_circle a b) :
    sqrt 2 < (sqrt (a^2 + b^2) / a) :=
sorry

end eccentricity_range_l537_537646


namespace max_intersection_points_l537_537400

-- Definitions of the conditions
def plane : Type := ℝ × ℝ
def line (p1 p2 : plane) : set plane := {p : plane | ∃ λ : ℝ, p = (p1.1 + λ * (p2.1 - p1.1), p1.2 + λ * (p2.2 - p1.2))}

-- Assumption that three lines intersect pairwise (we use arbitrary points for the lines)
def lines_intersect_pairwise
  (p1 p2 p3 p4 p5 p6 : plane) : Prop :=
  ∃ lp1 lp2 lp3 : set plane, lp1 = line p1 p2 ∧ lp2 = line p3 p4 ∧ lp3 = line p5 p6 ∧
  (∃ a : plane, a ∈ lp1 ∧ a ∈ lp2) ∧
  (∃ b : plane, b ∈ lp2 ∧ b ∈ lp3) ∧
  (∃ c : plane, c ∈ lp1 ∧ c ∈ lp3)

-- The theorem we want to prove
theorem max_intersection_points 
  (p1 p2 p3 p4 p5 p6 : plane) 
  (h : lines_intersect_pairwise p1 p2 p3 p4 p5 p6) : 
  ∃ p1' p2' p3', 
    p1' ≠ p2' ∧ p2' ≠ p3' ∧ p1' ≠ p3' ∧ 
    (∀ l : set plane, l ∈ {line p1 p2, line p3 p4, line p5 p6} → 
                     ∃ p : plane, p ∈ l ∧ (p = p1' ∨ p = p2' ∨ p = p3')) :=
sorry

end max_intersection_points_l537_537400


namespace sequence_count_2048_l537_537762

namespace RectangleTransformations

def F (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def G (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def I (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2)

noncomputable def count_sequences := nat.factorial 12 / (nat.factorial 6 * nat.factorial 6)  -- This calculates 2^11 since we're interested in even counts

theorem sequence_count_2048 :
  count_sequences = 2048 := sorry

end RectangleTransformations

end sequence_count_2048_l537_537762


namespace total_girls_is_68_l537_537825

-- Define the initial conditions
def track_length : ℕ := 100
def student_spacing : ℕ := 2
def girls_per_cycle : ℕ := 2
def cycle_length : ℕ := 3

-- Calculate the number of students on one side
def students_on_one_side : ℕ := track_length / student_spacing + 1

-- Number of cycles of three students
def num_cycles : ℕ := students_on_one_side / cycle_length

-- Number of girls on one side
def girls_on_one_side : ℕ := num_cycles * girls_per_cycle

-- Total number of girls on both sides
def total_girls : ℕ := girls_on_one_side * 2

theorem total_girls_is_68 : total_girls = 68 := by
  -- proof will be provided here
  sorry

end total_girls_is_68_l537_537825


namespace cos_60_degrees_is_one_half_l537_537934

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l537_537934


namespace inverse_proposition_implies_necessary_condition_l537_537294

variables (p q : Prop)

theorem inverse_proposition_implies_necessary_condition
  (hpq : p → q)
  (hqp : q → p) : ∀ {p q}, (p → q) ∧ (q → p) → (¬q → ¬p) :=
begin
  intros p q h,
  rw ←@imp_iff_not_or (¬q) (¬p),
  exact λ hq, or.resolve_right (h.right hq) (mt h.left),
end

end inverse_proposition_implies_necessary_condition_l537_537294


namespace evaluate_fg_sum_at_1_l537_537379

def f (x : ℚ) : ℚ := (4 * x^2 + 3 * x + 6) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x + 1

theorem evaluate_fg_sum_at_1 : f (g 1) + g (f 1) = 497 / 104 :=
by
  sorry

end evaluate_fg_sum_at_1_l537_537379


namespace eighth_odd_multiple_of_5_is_75_l537_537475

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0) ∧ (n % 2 = 1) ∧ (n % 5 = 0) ∧ (nat.find_greatest (λ m, m % 2 = 1 ∧ m % 5 = 0) 75 = 75) :=
    sorry

end eighth_odd_multiple_of_5_is_75_l537_537475


namespace find_fraction_l537_537133

theorem find_fraction (f : ℝ) (h₁ : f * 50.0 - 4 = 6) : f = 0.2 :=
by
  sorry

end find_fraction_l537_537133


namespace crayons_initial_total_l537_537751

theorem crayons_initial_total 
  (lost_given : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : lost_given = 70) (h2 : left = 183) : 
  initial = lost_given + left := 
by
  sorry

end crayons_initial_total_l537_537751


namespace cos_of_60_degrees_is_half_l537_537954

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537954


namespace distinct_lines_single_intersection_distinct_planes_intersect_in_line_unique_plane_through_line_and_point_l537_537043

-- Theorem 1: Two distinct lines cannot have more than one common point.
theorem distinct_lines_single_intersection {A B : Type} [linear_order A] [linear_order B] :
  ∀ (l m : set (A × B)), (l ≠ m) → (∃ x ∈ l, x ∈ m) → (∀ y, y ∈ l → y ∈ m) → false := 
  sorry

-- Theorem 2: Two distinct planes either do not intersect or their intersection is a line.
theorem distinct_planes_intersect_in_line {A B C : Type} [linear_order A] [linear_order B] [linear_order C] :
  ∀ (P Q : set (A × B × C)), (P ≠ Q) → (∃ x ∈ P, x ∈ Q) → 
  (∃ L : set (A × B), ∀ y, y ∈ P ∩ Q ↔ ∃ z ∈ L, y ∈ z) :=
  sorry

-- Theorem 3: There is only one plane passing through a given line and a point not on that line.
theorem unique_plane_through_line_and_point {A B C : Type} [linear_order A] [linear_order B] [linear_order C] :
  ∀ (l : set (A × B)), ∀ (P : set (A × B × C)), 
  (∀ x ∈ l, ∃ z, z ∈ P ∧ x ∈ z) → (∃ A, (A ∉ l) ∧ (∀ B ∈ P, A ∈ B)) → 
  (∀ Q : set (A × B × C), (∀ x ∈ l, ∃ z, z ∈ Q ∧ x ∈ z) → (∃ A', (A' ∉ l) ∧ (∀ B ∈ Q, A' ∈ B)) → P = Q :=
  sorry

end distinct_lines_single_intersection_distinct_planes_intersect_in_line_unique_plane_through_line_and_point_l537_537043


namespace sample_variance_classA_twenty_five_percentile_classB_l537_537759

-- Define the height data for Class A
def classA_heights : List ℝ := [170, 179, 162, 168, 158, 182, 179, 168, 163, 171]

-- Define the height data for Class B
def classB_heights : List ℝ := [159, 173, 179, 178, 162, 181, 176, 168, 170, 165]

-- Calculate the sample mean for Class A
def mean_classA : ℝ := (List.sum classA_heights) / classA_heights.length

-- Define the sample variance function
def sample_variance (data : List ℝ) (mean : ℝ) : ℝ :=
  List.sum (data.map (λ x => (x - mean) ^ 2)) / data.length

-- Statement for Part 1
theorem sample_variance_classA : sample_variance classA_heights mean_classA = 57.2 :=
by sorry

-- Function to calculate the percentile
def percentile (data : List ℝ) (p : ℝ) : ℝ :=
  let sorted := List.qsort (· ≤ ·) data
  sorted.nthLe (Int.toNat (p * data.length).natAbs - 1) (by decide)

-- Statement for Part 2
theorem twenty_five_percentile_classB : percentile classB_heights 0.25 = 165 :=
by sorry

end sample_variance_classA_twenty_five_percentile_classB_l537_537759


namespace radius_of_tangent_circle_is_6_25_l537_537610

noncomputable def radius_of_circle (side_length : ℝ) : ℝ :=
  let x := (75:ℝ) / 20 in
  side_length - x

theorem radius_of_tangent_circle_is_6_25 :
  ∀ (A B C D : ℝ) (side_length : ℝ),
    side_length = 10 →
    radius_of_circle side_length = 6.25 :=
by
  intros A B C D side_length h
  have h1 : radius_of_circle 10 = 6.25 := sorry
  rw ← h at h1
  exact h1

end radius_of_tangent_circle_is_6_25_l537_537610


namespace regular_polygon_sides_l537_537983

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l537_537983


namespace max_colors_in_grid_l537_537326

def grid_size := 4
def rectangle_size := (1, 3)

def valid_coloring (colors : ℕ) (grid : list (list ℕ)) : Prop :=
  ∀ i j, (i < grid.length) → (j + 2 < grid.head.length) → 
    let rect := [grid[i][j], grid[i][j + 1], grid[i][j + 2]] in 
    rect.nth 0 = rect.nth 1 ∨ rect.nth 0 = rect.nth 2 ∨ rect.nth 1 = rect.nth 2

theorem max_colors_in_grid : 
  ∀ (colors : ℕ) (grid : list (list ℕ)),
    (grid.length = grid_size) → 
    (∀ row, row ∈ grid → row.length = grid_size) →
    valid_coloring colors grid → 
    colors ≤ 9 :=
by
  intros colors grid h_len h_row_lengths h_valid_coloring
  sorry

end max_colors_in_grid_l537_537326


namespace max_clique_size_even_split_l537_537775

-- Define the necessary conditions and the problem
theorem max_clique_size_even_split (Participants : Type) 
  (friendship : Participants → Participants → Prop)
  [∀ x y, Decidable (friendship x y)]
  (exists_max_clique :
    ∃ n, ∀ clique : set Participants,
      (∀ x y ∈ clique, friendship x y) → (finite clique) → size clique ≤ n ∧ even n) :
  ∃ (R1 R2 : set Participants), 
    (∀ x y ∈ R1, friendship x y) ∧ 
    (∀ x y ∈ R2, friendship x y) ∧ 
    (∃ n, ∀ clique1 : set Participants,
      (∀ x y ∈ (clique1 ∩ R1), friendship x y) → (finite clique1) → size clique1 ≤ n) ∧
    (∃ n, ∀ clique2 : set Participants,
      (∀ x y ∈ (clique2 ∩ R2), friendship x y) → (finite clique2) → size clique2 ≤ n) ∧
    R1 ≠ ∅ ∧ R2 ≠ ∅ ∧
    (∀ n1 n2 : ℕ, (∃ clique1 : set Participants,
      (∀ x y ∈ (clique1 ∩ R1), friendship x y) ∧ 
      (finite clique1) ∧
      size clique1 = n1) ∧
    (∃ clique2 : set Participants,
      (∀ x y ∈ (clique2 ∩ R2), friendship x y) ∧ 
      (finite clique2) ∧
      size clique2 = n2) → 
    n1 = n2) :=
sorry

end max_clique_size_even_split_l537_537775


namespace angle_B_side_b_max_area_triangle_eq_l537_537435

noncomputable def triangle_conditions (A B C a b c R : ℝ) :=
  (R = sqrt 3) ∧
  (sin (A + B + C) = 0) ∧
  (sin (2 * A) - sin (C) = cos (C) * sin (B) / cos (B))

theorem angle_B_side_b {A B C a b c R : ℝ} (h : triangle_conditions A B C a b c R) :
  B = π / 3 ∧ b = 3 :=
sorry

theorem max_area_triangle_eq {A B C a b c R : ℝ} (h : triangle_conditions A B C a b c R) :
  ∃ (a' c': ℝ), a' = a ∧ c' = c ∧ 
  S_triangle = 9 * sqrt 3 / 4 ∧ a = b ∧ b = c :=
sorry

end angle_B_side_b_max_area_triangle_eq_l537_537435


namespace simplify_abs_expression_l537_537667

theorem simplify_abs_expression (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2 * b + 5| + |-3 * a + 2 * b - 2| = 4 * a - 4 * b + 7 := by
  sorry

end simplify_abs_expression_l537_537667


namespace points_M_form_line_l537_537256

theorem points_M_form_line (B A C D E : Point) (M : Point) (circumcircle : Circle)
  (isosceles_trapezoid : IsoscelesTrapezoid A D E C)
  (lies_on_angle_sides : ∀ (P : Point), P ∈ {D, C} → angle_vertex_side P B)
  (tangent_DM : Tangent D M circumcircle)
  (tangent_CM : Tangent C M circumcircle) :
  locus_of_M = {M : Point | lies_on_line_through B ∧ parallel_to' M AC ∧ M ≠ B} :=
by
  sorry

end points_M_form_line_l537_537256


namespace determine_a_l537_537068

lemma even_exponent (a : ℤ) : (a^2 - 4*a) % 2 = 0 :=
sorry

lemma decreasing_function (a : ℤ) : a^2 - 4*a < 0 :=
sorry

theorem determine_a (a : ℤ) (h1 : (a^2 - 4*a) % 2 = 0) (h2 : a^2 - 4*a < 0) : a = 2 :=
sorry

end determine_a_l537_537068


namespace beta_exists_l537_537368

noncomputable theory
open Polynomial

-- Let n be a positive integer
variables (n : ℕ) (hn : 0 < n)

-- Let {a_i} and {b_i} be sequences of distinct real numbers
variables (a b : Fin n → ℝ)
variable (h_distinct : Function.Injective a)

-- There exists a real number α such that for each i, the product of (a_i + b_j) over j from 1 to n equals α
variable (α : ℝ)
variable (h_alpha : ∀ i, (Finset.univ : Finset (Fin n)).prod (λ j, a i + b j) = α)

-- Our goal is to prove that there is a real number β such that for each j, the product of (a_i + b_j) over i from 1 to n equals β
theorem beta_exists : ∃ β : ℝ, ∀ j, (Finset.univ : Finset (Fin n)).prod (λ i, a i + b j) = β := sorry

end beta_exists_l537_537368


namespace sequence_count_25_length_l537_537660

def count_all_zeros_consecutive : Nat :=
  ∑ k in Finset.range 26, 26 - k

def count_two_blocks_of_ones : Nat :=
  ∑ j in Finset.range 24 | j > 1, (j - 1) * (26 - j)

theorem sequence_count_25_length :
  count_all_zeros_consecutive + count_two_blocks_of_ones = 4458 := by
  sorry

end sequence_count_25_length_l537_537660


namespace Jeans_average_speed_up_to_meeting_point_l537_537560

theorem Jeans_average_speed_up_to_meeting_point (d : ℝ) (h_d_pos : d > 0) :
  let T := (3 * d / 8) + (7 * d / 24) + (1 / 4) in
  let v := (2 * d - 1) / T in
  d = 2 →
  v = 36 / 19 :=
by
  sorry

end Jeans_average_speed_up_to_meeting_point_l537_537560


namespace arrangements_not_next_to_each_other_l537_537084

theorem arrangements_not_next_to_each_other:
  let total_people := 6 in
  let total_arrangements := Nat.factorial total_people in
  let adjacent_A_B_arrangements := (Nat.factorial 5) * 2 in
  total_arrangements - adjacent_A_B_arrangements = 480 :=
by
  sorry

end arrangements_not_next_to_each_other_l537_537084


namespace divisor_is_ten_l537_537704

variable (x y : ℝ)

theorem divisor_is_ten
  (h : ((5 * x - x / y) / (5 * x)) * 100 = 98) : y = 10 := by
  sorry

end divisor_is_ten_l537_537704


namespace problem1_problem2_problem3_problem4_l537_537559

theorem problem1 : (sqrt 50 * sqrt 32 / sqrt 8 - 4 * sqrt 2) = 6 * sqrt 2 := 
by
  sorry

theorem problem2 : (sqrt 12 - 6 * sqrt (1 / 3) + sqrt 48) = 4 * sqrt 3 :=
by
  sorry

theorem problem3 : ((sqrt 5 + 3) * (3 - sqrt 5) - (sqrt 3 - 1)^2) = 2 * sqrt 3 :=
by
  sorry

theorem problem4 : ((sqrt 24 + sqrt 50) / sqrt 2 - 6 * sqrt (1 / 3)) = 5 :=
by
  sorry

end problem1_problem2_problem3_problem4_l537_537559


namespace men_in_luxury_coach_l537_537577

-- Definitions of given conditions
def total_passengers : ℕ := 300
def percent_men : ℝ := 0.55
def percent_men_in_luxury : ℝ := 0.15

-- Prove that the number of men in the luxury coach is 25
theorem men_in_luxury_coach : 
  let men := total_passengers * percent_men in
  let men_in_luxury := men * percent_men_in_luxury in
  round(men_in_luxury) = 25 :=
by
  sorry

end men_in_luxury_coach_l537_537577


namespace min_jugs_needed_to_fill_container_l537_537146

def min_jugs_to_fill (jug_capacity container_capacity : ℕ) : ℕ :=
  Nat.ceil (container_capacity / jug_capacity)

theorem min_jugs_needed_to_fill_container :
  min_jugs_to_fill 16 200 = 13 :=
by
  -- The proof is omitted.
  sorry

end min_jugs_needed_to_fill_container_l537_537146


namespace false_statement_must_be_digit_is_5_l537_537534

-- Defining the statements as propositions about a digit d
def is_digit (d : ℕ) : Prop := d < 10
def digit_is_1 (d : ℕ) : Prop := d = 1
def digit_is_not_2 (d : ℕ) : Prop := d ≠ 2
def digit_is_not_1 (d : ℕ) : Prop := d ≠ 1
def digit_is_5 (d : ℕ) : Prop := d = 5
def digit_is_not_3 (d : ℕ) : Prop := d ≠ 3

theorem false_statement_must_be_digit_is_5 :
  ∀ (d : ℕ),
  is_digit d →
  (digit_is_1 d ∨ digit_is_not_2 d ∨ digit_is_not_1 d ∨ digit_is_5 d ∨ digit_is_not_3 d) →
  (¬ (digit_is_1 d) ∨ ¬ (digit_is_not_2 d) ∨ ¬ (digit_is_not_1 d) ∨ ¬ (digit_is_5 d) ∨ ¬ (digit_is_not_3 d)) → 
  ¬ digit_is_5 d :=
by
  intro d,
  intro hd,
  intro h1,
  intro h2,
  sorry

end false_statement_must_be_digit_is_5_l537_537534


namespace exists_bijective_f_divisible_by_3_l537_537713

/-- Statement of problem:
Given a number represented as digits in base 10, 
prove there exists a bijective function f such that 
f(a1) != 0 and the transformed number is divisible by 3.
-/
theorem exists_bijective_f_divisible_by_3 
  (n : ℕ) 
  (a : Fin n → Fin 10) 
  (h_digits : ∀ i, a i < 10) : 
  ∃ (f : Fin 10 → Fin 10), 
    Function.Bijective f ∧ 
    f (a 0) ≠ 0 ∧ 
    (∑ i in Finset.range n, f (a i) : ℕ) % 3 = 0 :=
sorry

end exists_bijective_f_divisible_by_3_l537_537713


namespace surface_area_of_sphere_is_correct_l537_537434

def radius_of_intersecting_circle := 1
def distance_from_center_to_plane := Real.sqrt 3
def sphere_radius := Real.sqrt (radius_of_intersecting_circle^2 + distance_from_center_to_plane^2)
noncomputable def surface_area_of_sphere := 4 * Real.pi * sphere_radius^2

theorem surface_area_of_sphere_is_correct :
  surface_area_of_sphere = 16 * Real.pi :=
by
  sorry

end surface_area_of_sphere_is_correct_l537_537434


namespace area_quadrilateral_AEDC_eq_54_l537_537701

noncomputable def P := sorry  -- Define the point P, the centroid

variables {A B C D E : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AD CE : ℝ) (PE PD DE : ℝ)
variables (h1 : PE = 2) (h2 : PD = 6) (h3 : DE = 2 * real.sqrt 10)

def area_of_quadrilateral (AEDC : Type*) [metric_space AEDC] : ℝ := sorry -- Define the area of quadrilateral AEDC

theorem area_quadrilateral_AEDC_eq_54 :
  area_of_quadrilateral AEDC = 54 :=
begin
  -- Given conditions
  have PE_eq_2 : PE = 2 := h1,
  have PD_eq_6 : PD = 6 := h2,
  have DE_eq_2sqrt10 : DE = 2 * real.sqrt 10 := h3,
  sorry  -- Proof
end

end area_quadrilateral_AEDC_eq_54_l537_537701


namespace amplitude_and_period_correct_l537_537429

noncomputable def ampl_and_period : ℝ × ℝ :=
  let y := λ x: ℝ, (Real.sin ((Real.pi / 6) - 2 * x) + Real.cos (2 * x)) in
  let simplified_y := λ x: ℝ, (√3 * Real.cos (2 * x + (Real.pi / 6))) in
  -- Amplitude and smallest positive period
  (√3, Real.pi)

theorem amplitude_and_period_correct : ampl_and_period = (√3, Real.pi) :=
  sorry

end amplitude_and_period_correct_l537_537429


namespace smallest_distance_l537_537027

noncomputable def a : Complex := 2 + 4 * Complex.I
noncomputable def b : Complex := 5 + 2 * Complex.I

theorem smallest_distance 
  (z w : Complex) 
  (hz : Complex.abs (z - a) = 2) 
  (hw : Complex.abs (w - b) = 4) : 
  Complex.abs (z - w) ≥ 6 - Real.sqrt 13 :=
sorry

end smallest_distance_l537_537027


namespace total_action_figures_l537_537168

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := by
  sorry

end total_action_figures_l537_537168


namespace cookies_per_person_l537_537922

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) (h1 : total_cookies = 35) (h2 : num_people = 5) :
  total_cookies / num_people = 7 := 
by {
  sorry
}

end cookies_per_person_l537_537922


namespace cos_of_60_degrees_is_half_l537_537955

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l537_537955


namespace LeRoy_shares_costs_equally_l537_537572

variable (X Y Z : ℝ)

theorem LeRoy_shares_costs_equally :
  let T := X + Y + Z in
  let E := (X + Y + Z) / 3 in
  E - X = (Y + Z - 2 * X) / 3 :=
by
  sorry

end LeRoy_shares_costs_equally_l537_537572


namespace intersection_is_open_interval_l537_537297

open Set
open Real

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_is_open_interval :
  M ∩ N = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_is_open_interval_l537_537297


namespace bhart_derangements_l537_537096

def derangement : ℕ → ℕ
| 0       := 1
| 1       := 0
| (n + 2) := (n + 1) * (derangement n + derangement (n + 1))

def distinguishable_derangements (n k : ℕ) : ℕ :=
  derangement n / ((nat.factorial k) * (nat.factorial (n - k)))

theorem bhart_derangements : distinguishable_derangements 6 2 = 135 :=
by {
  -- Given the conditions:
  -- The word is BHARAT
  -- The letters in the word are B, H, A, R, A, T
  -- The two A's are considered identical
  -- The derangements are calculated,
  -- which leads us to the final result.
  sorry
}

end bhart_derangements_l537_537096


namespace pureGalaTrees_l537_537886

theorem pureGalaTrees {T F C : ℕ} (h1 : F + C = 204) (h2 : F = (3 / 4 : ℝ) * T) (h3 : C = (1 / 10 : ℝ) * T) : (0.15 * T : ℝ) = 36 :=
by
  sorry

end pureGalaTrees_l537_537886


namespace interval_for_systematic_sampling_l537_537456

-- Define the total population size
def total_population : ℕ := 1203

-- Define the sample size
def sample_size : ℕ := 40

-- Define the interval for systematic sampling
def interval (n m : ℕ) : ℕ := (n - (n % m)) / m

-- The proof statement that the interval \( k \) for segmenting is 30
theorem interval_for_systematic_sampling : interval total_population sample_size = 30 :=
by
  show interval 1203 40 = 30
  sorry

end interval_for_systematic_sampling_l537_537456
