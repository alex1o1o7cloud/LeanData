import Algebra.QuadraticDiscriminant
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Absolute
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Power
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Matrix.Invertible
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.ContinuousFunction.Basic

namespace ratio_u_v_l515_515014

-- Define the side lengths and the points of tangency
variables {a b c u v : ℕ}

-- Given a triangle with side lengths 9, 15, and 18 respectively
def triangle_sides : Prop := a = 9 ∧ b = 15 ∧ c = 18

-- Define the condition for inscribed circle and the side segments
def inscribed_circle_conditions : Prop :=
  u + v = 9 ∧ u < v ∧ (∀ x, (x = u ∨ x = v) → (x ≤ 15 ∧ x ≤ 18))

-- The proof statement
theorem ratio_u_v (h1 : triangle_sides) (h2 : inscribed_circle_conditions) : u = 3 ∧ v = 6 :=
sorry

end ratio_u_v_l515_515014


namespace fraction_equality_l515_515394
-- Import the necessary library

-- The proof statement
theorem fraction_equality : (16 + 8) / (4 - 2) = 12 := 
by {
  -- Inserting 'sorry' to indicate that the proof is omitted
  sorry
}

end fraction_equality_l515_515394


namespace mia_money_l515_515286

def darwin_has := 45
def mia_has (d : ℕ) := 2 * d + 20

theorem mia_money : mia_has darwin_has = 110 :=
by
  unfold mia_has darwin_has
  rw [←nat.mul_assoc]
  rw [nat.mul_comm 2 45]
  sorry

end mia_money_l515_515286


namespace sphere_surface_area_l515_515931

theorem sphere_surface_area
(Points : Type*)
[MetricSpace Points]
(A B C D O : Points)
(a : ℝ)
(h1 : dist A B = a)
(h2 : dist B C = a)
(h3 : dist A C = real.sqrt 2 * a)
(h4 : ∀ t : ℝ, t ∈ Icc 0 1 → dist (O) (lerp t A D) = dist O (lerp t A D))
(h5 : dist D C = real.sqrt 6 * a)
(h6 : Sphere := metric.sphere):
S = 8 * real.pi * (a ^ 2) :=
sorry

end sphere_surface_area_l515_515931


namespace compute_4375_l515_515246

-- Definitions of points and lines according to the given problem
variable {α : Type*} [MetricSpace α]
variables (A0 B C0 D U V : α) -- Points
variables (ω : circle α) -- Circle

-- Length conditions
def lengths : Prop := dist A0 B = 3 ∧ dist B C0 = 4 ∧ dist C0 D = 6 ∧ dist D A0 = 7

-- Proof problem
theorem compute_4375 (h : lengths A0 B C0 D) : 100 * 9 + 10 * 345 + 25 = 4375 :=
by sorry

end compute_4375_l515_515246


namespace triangle_angle_bisector_l515_515763

theorem triangle_angle_bisector (A B C : Type) [triangle A B C] (a b lc : ℝ)
  (angle_C_120 : ∠C = 120)
  (side_a : A.complementary_length = a)
  (side_b : B.complementary_length = b)
  (angle_bisector_lc : C.angle_bisector_length = lc) :
  (1 / a) + (1 / b) = (1 / lc) :=
sorry

end triangle_angle_bisector_l515_515763


namespace curve_equation_coefficients_l515_515016

variable (t : ℝ)

def x := 3 * Real.cos t + 2 * Real.sin t
def y := 3 * Real.sin t

theorem curve_equation_coefficients :
  ∃ a b c : ℝ, a = 1/9 ∧ b = -4/27 ∧ c = 13/81 ∧
  (a * x t * x t + b * x t * y t + c * y t * y t = 1) := by
  sorry

end curve_equation_coefficients_l515_515016


namespace sum_of_fraction_of_repeating_decimal_l515_515452

/-- Given the repeating decimal number 3.71717171..., 
    when written in fraction form and reduced to the lowest terms,
    the sum of the numerator and denominator is 467. --/
theorem sum_of_fraction_of_repeating_decimal :
  let decimal := (3 : ℚ) + (71 / 99) in
  let fraction := (368 / 99) in
  (fraction.num + fraction.denom) = 467 :=
by
  let decimal := (3 : ℚ) + (71 / 99)
  let fraction := (368 / 99)
  have h : fraction = decimal := sorry
  have h_num_denom_sum : (fraction.num + fraction.denom) = 467 := sorry
  exact h_num_denom_sum

end sum_of_fraction_of_repeating_decimal_l515_515452


namespace least_odd_prime_factor_2023_6_plus_1_is_13_l515_515478

noncomputable def least_odd_prime_factor_of_2023_6_plus_1 : ℕ := 
  13

theorem least_odd_prime_factor_2023_6_plus_1_is_13 :
  ∃ p, p.prime ∧ (p = least_odd_prime_factor_of_2023_6_plus_1) ∧ 
     (∀ q, q.prime ∧ q < p → 
       (q ≠ 1 ∨ q ∣ (2023^6 + 1)) ∨ 
       (2023^6 ≡ -1 [MOD q])) ∧
    (least_odd_prime_factor_of_2023_6_plus_1 ≡ 1 [MOD 12]) :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_is_13_l515_515478


namespace total_number_of_boys_l515_515047

theorem total_number_of_boys (n : ℕ) 
  (h1 : (40 - 10) % n = n / 2)
  (h2 : ∀ k, k = 2 * m ∨ k = 2 * m + 1 -> k < n) : 
  (n / 2 = 30) := 
begin
  sorry
end

end total_number_of_boys_l515_515047


namespace mia_has_110_l515_515281

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l515_515281


namespace min_adjacent_white_cells_8x8_grid_l515_515918

theorem min_adjacent_white_cells_8x8_grid (n_blacks : ℕ) (h1 : n_blacks = 20) : 
  ∃ w_cell_pairs, w_cell_pairs = 34 :=
by
  -- conditions are translated here for interpret
  let total_pairs := 112 -- total pairs in 8x8 grid
  let max_spoiled := 78  -- maximum spoiled pairs when placing 20 black cells
  let min_adjacent_white_pairs := total_pairs - max_spoiled
  use min_adjacent_white_pairs
  exact (by linarith)
  sorry

end min_adjacent_white_cells_8x8_grid_l515_515918


namespace distinct_points_count_l515_515522

-- Definitions of sets A, B, and C
def A : Set ℕ := {4}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 5}

-- Definition of the theorem to be proved
theorem distinct_points_count :
  (A.card * B.card * C.card) = 6 :=
sorry

end distinct_points_count_l515_515522


namespace point_on_x_axis_coord_l515_515223

theorem point_on_x_axis_coord (m : ℝ) (h : (m - 1, 2 * m).snd = 0) : (m - 1, 2 * m) = (-1, 0) :=
by
  sorry

end point_on_x_axis_coord_l515_515223


namespace ortho_sin2α_parallel_sin_cos_l515_515548

variables {α : ℝ}
variables (a b : ℝ → ℝ²)

def a (α : ℝ) : ℝ² := (1, Real.cos α)
def b (α : ℝ) : ℝ² := (1/3, Real.sin α)

theorem ortho_sin2α (h1 : α ∈ Set.Ioo 0 Real.pi)
    (h2 : a α ⬝ b α = 0) : Real.sin (2 * α) = -1/6 :=
  sorry

theorem parallel_sin_cos (h1 : α ∈ Set.Ioo 0 Real.pi)
    (h2 : ∃ k, (1, Real.cos α) = k • (1/3, Real.sin α)) : 
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -2 :=
  sorry

end ortho_sin2α_parallel_sin_cos_l515_515548


namespace total_profit_from_selling_30_necklaces_l515_515711

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l515_515711


namespace parabola_tangent_circle_radius_l515_515669

noncomputable def radius_of_tangent_circle : ℝ :=
  let r := 1 / 4
  r

theorem parabola_tangent_circle_radius :
  ∃ (r : ℝ), (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 1 - 4 * r) ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end parabola_tangent_circle_radius_l515_515669


namespace two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l515_515612

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hm_odd : odd m) (hn_odd : odd n) :
  ¬ ∃ k : ℤ, 2^m - 1 = k * (3^n - 1) :=
by
  sorry

end two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l515_515612


namespace find_m_l515_515623

theorem find_m (S : ℕ → ℝ) (m : ℕ) (h1 : S m = -2) (h2 : S (m+1) = 0) (h3 : S (m+2) = 3) : m = 4 :=
by
  sorry

end find_m_l515_515623


namespace find_p_l515_515542

noncomputable def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.2 ^ 2 = 2 * p * xy.1}
noncomputable def circle : Set (ℝ × ℝ) := {xy | (xy.1 - 3) ^ 2 + xy.2 ^ 2 = 16}

def directrix (p : ℝ) (x : ℝ) : Prop := x = -p / 2

def tangency_condition (p : ℝ) : Prop := abs (3 + p / 2) = 4

theorem find_p (p : ℝ) (h1: p > 0) (h2: ∀ xy, xy ∈ parabola p → xy ∈ circle → Directrix is tangent):
  tangency_condition p ↔ p = 2 :=
by
  sorry

end find_p_l515_515542


namespace min_adj_white_pairs_l515_515920

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l515_515920


namespace translation_of_polygon_gives_prism_l515_515698

variables (P : Type) [plane_polygon P] (d : direction)

def translated_solid_is_prism (P : Type) [plane_polygon P] (d : direction) : Prop :=
  solid_translating_polygon_in_direction_is_prism P d

theorem translation_of_polygon_gives_prism (P : Type) [plane_polygon P] (d : direction) :
  translated_solid_is_prism P d :=
sorry

end translation_of_polygon_gives_prism_l515_515698


namespace area_of_tangents_l515_515906

def radius := 3
def segment_length := 6

theorem area_of_tangents (r : ℝ) (l : ℝ) (h1 : r = radius) (h2 : l = segment_length) :
  let R := r * Real.sqrt 2 
  let annulus_area := π * (R ^ 2) - π * (r ^ 2)
  annulus_area = 9 * π :=
by
  sorry

end area_of_tangents_l515_515906


namespace mother_picked_38_carrots_l515_515968

theorem mother_picked_38_carrots
  (haley_carrots : ℕ)
  (good_carrots : ℕ)
  (bad_carrots : ℕ)
  (total_carrots_picked : ℕ)
  (mother_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : good_carrots = 64)
  (h3 : bad_carrots = 13)
  (h4 : total_carrots_picked = good_carrots + bad_carrots)
  (h5 : total_carrots_picked = haley_carrots + mother_carrots) :
  mother_carrots = 38 :=
by
  sorry

end mother_picked_38_carrots_l515_515968


namespace problem_statement_l515_515537

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then cos (Real.pi * x) else f (x - 1) + 1

theorem problem_statement : f (4 / 3) + f (-4 / 3) = 1 :=
  sorry

end problem_statement_l515_515537


namespace people_left_is_10_l515_515831

def initial_people : ℕ := 12
def people_joined : ℕ := 15
def final_people : ℕ := 17
def people_left := initial_people - final_people + people_joined

theorem people_left_is_10 : people_left = 10 :=
by sorry

end people_left_is_10_l515_515831


namespace b_geometric_a_general_term_l515_515965

-- Define the sequence a_n
def a : ℕ → ℚ
| 0     := 1 / 3
| (n+1) := 3 * a n + 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 1) - a n + 1

-- Prove that b_n is geometric with first term 11/3 and common ratio 3
theorem b_geometric : ∃ r : ℚ, r = 3 ∧ b 1 = 11 / 3 ∧ ∀ n, b (n + 1) = r * b n :=
sorry

-- Prove the general term formula for a_n
theorem a_general_term (n : ℕ₀) : a n = (11 / 2) * 3^(n - 2) - n - (5 / 6) :=
sorry

end b_geometric_a_general_term_l515_515965


namespace find_A_is_3_l515_515808

-- Declarations of the digits and the constraints as specified in the problem
variable (A B C D E F G H I J K : ℕ)

-- Ensure each digit is unique
axiom distinct_digits: Function.Injective (λ x : ℕ, x) -- simplifying uniqueness

-- Conditions based on the problem statement
axiom decreasing_order_ABC: A > B ∧ B > C
axiom decreasing_order_DEF: D > E ∧ E > F
axiom decreasing_order_GHIJK: G > H ∧ H > I ∧ I > J ∧ J > K
axiom consecutive_odd_DEF: ∃ n : ℕ, D = 2 * n + 1 ∧ E = 2 * (n - 1) + 1 ∧ F = 2 * (n - 2) + 1
axiom consecutive_even_GHIJK: ∃ m : ℕ, G = 2 * m ∧ H = 2 * (m - 1) ∧ I = 2 * (m - 2) ∧ J = 2 * (m - 3) ∧ K = 2 * (m - 4)
axiom sum_ABC: A + B + C = 7

theorem find_A_is_3 : A = 3 :=
by
  -- Proof involves formulating and checking each constraint given
  sorry

end find_A_is_3_l515_515808


namespace money_distribution_l515_515044

variable (A B C : ℝ)

theorem money_distribution
  (h₁ : A + B + C = 500)
  (h₂ : A + C = 200)
  (h₃ : C = 60) :
  B + C = 360 :=
by
  sorry

end money_distribution_l515_515044


namespace max_area_of_triangle_l515_515451

theorem max_area_of_triangle {z1 z2 z3 : ℂ}
  (hz1 : ∥z1∥ = 3)
  (hz2 : z2 = conj z1)
  (hz3 : z3 = 1 / z1) :
  max_area (triangle_area z1 z2 z3) = 4 :=
sorry

end max_area_of_triangle_l515_515451


namespace milk_amount_in_ounces_l515_515030

theorem milk_amount_in_ounces :
  ∀ (n_packets : ℕ) (ml_per_packet : ℕ) (ml_per_ounce : ℕ),
  n_packets = 150 →
  ml_per_packet = 250 →
  ml_per_ounce = 30 →
  (n_packets * ml_per_packet) / ml_per_ounce = 1250 :=
by
  intros n_packets ml_per_packet ml_per_ounce h_packets h_packet_ml h_ounce_ml
  rw [h_packets, h_packet_ml, h_ounce_ml]
  sorry

end milk_amount_in_ounces_l515_515030


namespace arrangement_count_l515_515440

noncomputable def arrangements (items : List ℕ) : ℕ := sorry

theorem arrangement_count :
  ∀ (items: List ℕ), 5 ∈ items.length →
  (∀ a b, a ∈ items → b ∈ items → a ≠ b → (∃ idx, items.nth idx = some a ∧ items.nth (idx + 1) = some b) →
  (∀ c d, c ∈ items → d ∈ items → c ≠ d → ¬(∃ idx, items.nth idx = some c ∧ items.nth (idx + 1) = some d)) →
  arrangements items = 48 :=
sorry

end arrangement_count_l515_515440


namespace A_equality_l515_515896

noncomputable def A (n k r : ℕ) : ℕ :=
  {x : Vector ℕ k | (∀ i j, i < j → x[i] ≥ x[j]) ∧ (x.toList.sum = n) ∧ (x[0] - x[k-1] ≤ r)}.card

theorem A_equality (s t : ℕ) (hs : s ≥ 2) (ht : t ≥ 2) :
  A (s * t) s t = A (s * (t - 1)) s t ∧ A (s * t) s t = A ((s - 1) * t) s t := by
  sorry

end A_equality_l515_515896


namespace smallest_n_divisor_lcm_gcd_l515_515203

theorem smallest_n_divisor_lcm_gcd :
  ∀ n : ℕ, n > 0 ∧ (∀ a b : ℕ, 60 = a ∧ n = b → (Nat.lcm a b / Nat.gcd a b = 50)) → n = 750 :=
by
  sorry

end smallest_n_divisor_lcm_gcd_l515_515203


namespace graph_of_equation_four_lines_l515_515685

theorem graph_of_equation_four_lines :
  (∀ (x y : ℝ), (x^2 - 9)^2 * (x^2 - y^2)^2 = 0 → 
    (x = 3 ∨ x = -3 ∨ y = x ∨ y = -x)) :=
begin
  sorry
end

end graph_of_equation_four_lines_l515_515685


namespace area_computation_l515_515833

noncomputable def areaOfBoundedFigure : ℝ :=
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4), 
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  (integral / 2) - rectArea

theorem area_computation :
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4),
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  ((integral / 2) - rectArea) = (5 * Real.pi - 10) :=
by
  sorry

end area_computation_l515_515833


namespace num_distinct_paths_l515_515419

theorem num_distinct_paths : 
  let right_steps := 3 in
  let total_steps := 11 in
  let diagonal_steps := 5 in 
  ∃ paths, paths = Nat.choose 11 5 ∧ paths = 462 :=
by
  have right_steps := 3
  have total_steps := 11
  have diagonal_steps := 5
  use (Nat.choose 11 5)
  split
  . rfl
  . simp [Nat.choose]
  { sorry }

end num_distinct_paths_l515_515419


namespace maria_savings_l515_515279

variable (S : ℝ) -- Define S as a real number (amount saved initially)

-- Conditions
def bike_cost : ℝ := 600
def additional_money : ℝ := 250 + 230

-- Theorem statement
theorem maria_savings : S + additional_money = bike_cost → S = 120 :=
by
  intro h -- Assume the hypothesis (condition)
  sorry -- Proof will go here

end maria_savings_l515_515279


namespace elvin_total_telephone_bill_first_month_l515_515872

-- Definitions
variables (F C : ℝ)
-- Conditions
axiom h1 : F + C = 40
axiom h2 : F + 2 * C = 76

-- Theorem statement
theorem elvin_total_telephone_bill_first_month : F + C = 40 := by
  exact h1
  sorry

end elvin_total_telephone_bill_first_month_l515_515872


namespace distribution_of_cousins_l515_515652

theorem distribution_of_cousins (cousins bedrooms : ℕ) (empty_bedroom : ℕ) (h1 : cousins = 5) (h2 : bedrooms = 3) (h3 : empty_bedroom ≥ 1) : 
  ∑ k in { (a, b, c) | a + b + c = 5 ∧ (a == 0 ∨ b == 0 ∨ c == 0) }, 1 = 26 :=
by 
  sorry

end distribution_of_cousins_l515_515652


namespace carrie_pays_94_l515_515841

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l515_515841


namespace determine_c_l515_515538

-- Given conditions and function definitions
def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

-- Main theorem stating the proof problem
theorem determine_c (c : ℝ) (h : ∀ x : ℝ, x ≠ -3/2 → (f c (f c x)) = x) : 
  c = -3 :=
sorry

end determine_c_l515_515538


namespace incorrect_option_C_l515_515137

-- Define noncomputable entities
noncomputable def Plane := Type
variable (α β γ : Plane) (a b : Plane)

-- Assumptions based on conditions from the problem
variables (h₁ : α ⊥ γ) (h₂ : β ⊥ γ)

-- The statement to be proved:
theorem incorrect_option_C : ¬ (α ∥ β) := sorry

end incorrect_option_C_l515_515137


namespace equation_of_line_l515_515696

def point : Type := ℝ × ℝ

def line_equation (m : ℝ) (P : point) : (ℝ → ℝ → Prop) :=
  λ x y, m * (x - P.1) = y - P.2

theorem equation_of_line
  (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 2)
  (h_point : (x1, y1) = (2, 3))
  : ∃ (a b c : ℝ), (a * x1 + b * y1 + c = 0) ∧ (a = 2 ∧ b = -1 ∧ c = -1) :=
sorry

end equation_of_line_l515_515696


namespace volunteer_assignment_schemes_count_l515_515224

-- Definitions of the conditions
def five_people := {A, B, P1, P2, P3}

def projects := {projA, projB, projC}

def is_valid_assignment (assignment : projects → five_people) : Prop :=
  (assignment projA ≠ A ∧ assignment projB ≠ A) ∧
  (assignment projB ≠ B ∧ assignment projC ≠ B)

-- Final statement leveraging the conditions defined
theorem volunteer_assignment_schemes_count :
  ∃ assignment_set : (projects → five_people) → Set 
    (assignment ∈ assignment_set 
      ∧ is_valid_assignment assignment), 
     #(assignment_set) = 21 :=
sorry

end volunteer_assignment_schemes_count_l515_515224


namespace area_triangle_ACE_l515_515792

-- Define the hexagon and triangle structure
structure RegularHexagon :=
  (sides : ℝ)
  (regular : ∀ (x y : ℝ), x = sides ∧ y = sides)

-- Define the conditions
def hexABCDEF : RegularHexagon := 
  { sides := 3, regular := by simp }

-- Define the problem to prove the area of $\triangle ACE$
theorem area_triangle_ACE (h : RegularHexagon) (h_sides : h.sides = 3) :
  area h.regular ACE = (9 * Real.sqrt 3) / 4 :=
sorry

end area_triangle_ACE_l515_515792


namespace final_price_correct_l515_515435

noncomputable def original_price : ℝ := 49.99
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.20

theorem final_price_correct :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount = 36.00 := by
    -- The proof would go here
    sorry

end final_price_correct_l515_515435


namespace max_S_n_l515_515947

theorem max_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) (h2 : d < 0) (h3 : S 6 = 5 * (a 1) + 10 * d) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, S m ≤ S n) :=
by
  sorry

end max_S_n_l515_515947


namespace selling_price_with_discount_l515_515326

variable (a : ℝ)

theorem selling_price_with_discount (h : a ≥ 0) : (a * 1.2 * 0.91) = (a * 1.2 * 0.91) :=
by
  sorry

end selling_price_with_discount_l515_515326


namespace sam_total_pennies_l515_515307

theorem sam_total_pennies : 
  ∀ (initial_pennies found_pennies total_pennies : ℕ),
  initial_pennies = 98 → 
  found_pennies = 93 → 
  total_pennies = initial_pennies + found_pennies → 
  total_pennies = 191 := by
  intros
  sorry

end sam_total_pennies_l515_515307


namespace exists_third_degree_poly_with_positive_and_negative_roots_l515_515239

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l515_515239


namespace petya_vasya_three_numbers_equal_l515_515754

theorem petya_vasya_three_numbers_equal (a b c : ℕ) :
  gcd a b = lcm a b ∧ gcd b c = lcm b c ∧ gcd a c = lcm a c → a = b ∧ b = c :=
by
  sorry

end petya_vasya_three_numbers_equal_l515_515754


namespace max_value_of_e_l515_515259

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (5^n - 1) / 4

-- Define e_n as the gcd of b_n and b_(n+1)
def e (n : ℕ) : ℚ := Int.gcd (b n) (b (n + 1))

-- The theorem we need to prove is that e_n is always 1
theorem max_value_of_e (n : ℕ) : e n = 1 :=
  sorry

end max_value_of_e_l515_515259


namespace Lapis_gets_correct_share_l515_515892

def Payment := ℕ

structure Payments :=
  (Fonzie  : Payment)
  (AuntBee : Payment)
  (Lapis   : Payment)

def total_paid (p : Payments) : Payment :=
  p.Fonzie + p.AuntBee + p.Lapis

def Lapis_fraction (p : Payments) : ℚ :=
  p.Lapis / total_paid p

def total_treasure : ℕ :=
  900000

def Lapis_share (p : Payments) : ℚ :=
  total_treasure * (Lapis_fraction p)

theorem Lapis_gets_correct_share : Lapis_share ⟨7000, 8000, 9000⟩ = 337500 := by
  sorry

end Lapis_gets_correct_share_l515_515892


namespace find_x_expected_value_X_l515_515220

-- Define conditions
def total_volunteers := 500
def selected_volunteers := 100
def frequency_distribution : List ℝ := [0.01, 0.02, 0.04, x, 0.07]
def under_35_in_sampling := 6
def over_35_in_sampling := 4

-- Given conditions about frequencies summation
lemma frequency_sum (x : ℝ) : (0.01 + 0.02 + 0.04 + x + 0.07) * 5 = 1 :=
begin
  sorry
end

-- Prove the value of x
theorem find_x : ∃ x : ℝ, (0.01 + 0.02 + 0.04 + x + 0.07) * 5 = 1 ∧ x = 0.06 :=
begin
  refine ⟨0.06, _, rfl⟩,
  sorry
end

-- Prove the expected value of X
theorem expected_value_X : ∃ P : Fin 4 → ℝ, 
  P 0 = 1 / 30 ∧ P 1 = 3 / 10 ∧ P 2 = 1 / 2 ∧ P 3 = 1 / 6 ∧
  ∑ i in Finset.univ, i * P i = 1.8 :=
begin
  refine ⟨λ i, match i with
    | 0 => 1 / 30
    | 1 => 3 / 10
    | 2 => 1 / 2
    | 3 => 1 / 6
    end,
    by norm_num,
    by norm_num,
    by norm_num,
    by norm_num,
    _⟩,
  sorry
end

end find_x_expected_value_X_l515_515220


namespace error_difference_l515_515026

noncomputable def total_income_without_error (T: ℝ) : ℝ :=
  T + 110000

noncomputable def total_income_with_error (T: ℝ) : ℝ :=
  T + 1100000

noncomputable def mean_without_error (T: ℝ) : ℝ :=
  (T + 110000) / 500

noncomputable def mean_with_error (T: ℝ) : ℝ :=
  (T + 1100000) / 500

theorem error_difference (T: ℝ) :
  mean_with_error T - mean_without_error T = 1980 :=
by
  sorry

end error_difference_l515_515026


namespace possible_to_position_guards_l515_515235

-- Define the conditions
def guard_sees (d : ℝ) : Prop := d = 100

-- Prove that it is possible to arrange guards around a point object so that neither the object nor the guards can be approached unnoticed
theorem possible_to_position_guards (num_guards : ℕ) (d : ℝ) (h : guard_sees d) : 
  (0 < num_guards) → 
  (∀ θ : ℕ, θ < num_guards → (θ * (360 / num_guards)) < 360) → 
  True :=
by 
  -- Details of the proof would go here
  sorry

end possible_to_position_guards_l515_515235


namespace cube_root_equality_l515_515382

theorem cube_root_equality (a b : ℝ) : (∛a = ∛b) → a = b := 
by 
    sorry

end cube_root_equality_l515_515382


namespace milk_amount_in_ounces_l515_515031

theorem milk_amount_in_ounces :
  ∀ (n_packets : ℕ) (ml_per_packet : ℕ) (ml_per_ounce : ℕ),
  n_packets = 150 →
  ml_per_packet = 250 →
  ml_per_ounce = 30 →
  (n_packets * ml_per_packet) / ml_per_ounce = 1250 :=
by
  intros n_packets ml_per_packet ml_per_ounce h_packets h_packet_ml h_ounce_ml
  rw [h_packets, h_packet_ml, h_ounce_ml]
  sorry

end milk_amount_in_ounces_l515_515031


namespace trapezoid_area_l515_515598

-- Define the geometric properties and distances given
variable (A B C D H K : Point)
variable (AC AD BC BH DK : ℝ)
variable (isosceles : IsoscelesTrapezoid A B C D)
variable (perpendicular_b : Perpendicular BH (Line A C))
variable (perpendicular_d : Perpendicular DK (Line A C))
variable (foot_b_on_ac : OnSegment (Foot BH (Line A C)) (Line A C))
variable (foot_d_on_ac : OnSegment (Foot DK (Line A C)))
variable (AC_eq : AC = 20)
variable (AK_eq : Distance A K = 19)
variable (AH_eq : Distance A H = 3)

-- Prove the area of the trapezoid is 120
theorem trapezoid_area : AreaOfTrapezoid A B C D = 120 := 
by
  sorry

end trapezoid_area_l515_515598


namespace sphere_pyramid_problem_l515_515765

theorem sphere_pyramid_problem (n m : ℕ) :
  (n * (n + 1) * (2 * n + 1)) / 6 + (m * (m + 1) * (m + 2)) / 6 = 605 → n = 10 ∧ m = 10 :=
by
  sorry

end sphere_pyramid_problem_l515_515765


namespace t_shirt_sale_revenue_per_minute_l515_515322

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l515_515322


namespace union_A_B_l515_515001

open Set Nat

def A : Set ℕ := {x | x % 2 = 1 ∧ x ≤ 5}
def B : Set ℤ := {-3, 2, 3}

theorem union_A_B : A ∪ B = {-3, 1, 2, 3, 5} := 
by {
  sorry -- skipping the proof as stipulated
}

end union_A_B_l515_515001


namespace money_made_per_minute_l515_515321

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l515_515321


namespace pyramid_total_area_l515_515742

noncomputable def pyramid_base_edge := 8
noncomputable def pyramid_lateral_edge := 10

/-- The total area of the four triangular faces of a right, square-based pyramid
with base edges measuring 8 units and lateral edges measuring 10 units is 32 * sqrt(21). -/
theorem pyramid_total_area :
  let base_edge := pyramid_base_edge,
      lateral_edge := pyramid_lateral_edge,
      height := sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2),
      area_of_one_face := 1 / 2 * base_edge * height
  in 4 * area_of_one_face = 32 * sqrt 21 :=
sorry

end pyramid_total_area_l515_515742


namespace milk_volume_in_ounces_l515_515029

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end milk_volume_in_ounces_l515_515029


namespace sample_size_is_100_l515_515368

/-- A sample size is defined to be the number of items selected from a larger population. -/
def sample_size (selected : ℕ) (population : ℕ) : ℕ := selected

theorem sample_size_is_100 (population selected : ℕ) (h1 : population = 5000) (h2 : selected = 100) :
  sample_size selected population = 100 :=
by {
  rw [sample_size, h2],
  exact rfl,
}

end sample_size_is_100_l515_515368


namespace mia_has_110_l515_515282

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l515_515282


namespace total_amount_l515_515427

theorem total_amount (x : ℝ) (hC : 2 * x = 70) :
  let B_share := 1.25 * x
  let C_share := 2 * x
  let D_share := 0.7 * x
  let E_share := 0.5 * x
  let A_share := x
  B_share + C_share + D_share + E_share + A_share = 190.75 :=
by
  sorry

end total_amount_l515_515427


namespace three_tangent_lines_l515_515181

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

def g (t : ℝ) : ℝ := t^2 / Real.exp t

theorem three_tangent_lines (a : ℝ) : 
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ g t1 = a ∧ g t2 = a ∧ g t3 = a) ↔ (0 < a ∧ a < 4 / Real.exp 2) :=
begin
  sorry
end

end three_tangent_lines_l515_515181


namespace mia_money_l515_515283

variable (DarwinMoney MiaMoney : ℕ)

theorem mia_money :
  (MiaMoney = 2 * DarwinMoney + 20) → (DarwinMoney = 45) → MiaMoney = 110 := by
  intros h1 h2
  rw [h2] at h1
  rw [h1]
  sorry

end mia_money_l515_515283


namespace equal_real_roots_of_quadratic_l515_515898

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ deriv (λ x : ℝ, x^2 + k*x + 1) x = 0) ↔ k = 2 ∨ k = -2 := 
by
  sorry

end equal_real_roots_of_quadratic_l515_515898


namespace track_circumference_l515_515000

theorem track_circumference 
  (uniform_speeds : ∀ t₁ t₂, A.travel t₁ t₂ = B.travel t₂ t₁)
  (diametrically_opposite_start : A.start_pos = B.start_pos + x)
  (first_meeting_100_yards : B.travel_time_dist 100 = A.travel_time_dist (x - 100))
  (second_meeting_60_yards_before_completion : A.travel_time_dist (2 * x - 60) = B.travel_time_dist (x + 60)) :
  2 * x = 480 :=
sorry

end track_circumference_l515_515000


namespace min_max_PM_minus_PN_l515_515157

theorem min_max_PM_minus_PN :
  let P := {p : ℝ × ℝ | p.1^2 / 9 - p.2^2 / 16 = 1},
      M := {m : ℝ × ℝ | (m.1 + 5)^2 + m.2^2 = 1},
      N := {n : ℝ × ℝ | (n.1 - 5)^2 + n.2^2 = 4},
      PM := λ p m, real.sqrt ((p.1 - m.1)^2 + (p.2 - m.2)^2),
      PN := λ p n, real.sqrt ((p.1 - n.1)^2 + (p.2 - n.2)^2) in
  ∃ (p ∈ P) (m ∈ M) (n ∈ N), (PM p m - PN p n) = 3 ∧ (PM p m - PN p n) = 9 := sorry

end min_max_PM_minus_PN_l515_515157


namespace min_value_of_expression_l515_515186

theorem min_value_of_expression (m n : ℝ) (h1 : m + 2 * n = 2) (h2 : m > 0) (h3 : n > 0) : 
  (1 / (m + 1) + 1 / (2 * n)) ≥ 4 / 3 :=
sorry

end min_value_of_expression_l515_515186


namespace four_digit_number_sum_is_correct_l515_515489

def valid_digits : Finset ℕ := {1, 2, 3, 4, 5}

def valid_four_digit_numbers : Finset (List ℕ) := 
  valid_digits.powerset.filter (λ s, s.card = 4) 
  >>= λ s, (s.val.to_list.permutations)

def four_digit_number_sum (numbers : Finset (List ℕ)) : ℕ :=
  numbers.sum (λ l, 
    match l with
    | [a, b, c, d] => 1000 * a + 100 * b + 10 * c + d
    | _           => 0
    end)

theorem four_digit_number_sum_is_correct :
  four_digit_number_sum valid_four_digit_numbers = 399960 :=
by
  sorry

end four_digit_number_sum_is_correct_l515_515489


namespace max_volume_formula_side_length_and_height_l515_515387

noncomputable def max_volume_of_pyramid (n S : ℝ) : ℝ :=
  (sqrt 2 / 12) * (S ^ (3 / 2) / sqrt (n * tan (Real.pi / n)))

theorem max_volume_formula (n : ℕ) (S : ℝ) :
  ∃ V, V = max_volume_of_pyramid n S :=
begin
  use (sqrt 2 / 12) * (S ^ (3 / 2) / sqrt (n * tan (Real.pi / n))),
  sorry
end

theorem side_length_and_height (S : ℝ) (V : ℝ) (n : ℕ) 
  (hn : n = 4) (hS : S = 144) (hV : V = 64) :
  ∃ (a1 a2 h1 h2 : ℝ),
    (a1 = 2 * sqrt 2 ∧ h1 = 24) ∧
    (a2 = 8 ∧ h2 = 3) :=
begin
  use [2 * sqrt 2, 8, 24, 3],
  split, 
  { split; refl }, 
  { split; refl }
end

end max_volume_formula_side_length_and_height_l515_515387


namespace raju_working_days_l515_515608

theorem raju_working_days (x : ℕ) 
  (h1: (1 / 10 : ℚ) + 1 / x = 1 / 8) : x = 40 :=
by sorry

end raju_working_days_l515_515608


namespace find_pq_equals_5_l515_515714

-- Definitions of the problem
def triangle (DE EF FD : ℝ) (angle_EDF :ℝ) (angle_DFE : ℝ): Prop :=
DE = 6 ∧ EF = 10 ∧ FD = 8 ∧ angle_EDF = 90 ∧ angle_DFE = 90

def circles_uw3_uw4 (E F D L : Point) (tangent_1 tangent_2 : Tangent)
(p q : ℕ) [RelPrime p q] : Prop :=
let DL := (p / q) in -- Define DL using given ratio
DL = 4

-- Problem statement
theorem find_pq_equals_5 (DE EF FD : ℝ) 
  (angle_EDF angle_DFE : ℝ)
  (E D F L : Point)
  (tangent_1 tangent_2 : Tangent)
  (p q : ℕ) 
  [RelPrime p q] :
  triangle DE EF FD angle_EDF angle_DFE →
  circles_uw3_uw4 E F D L tangent_1 tangent_2 p q → 
  p + q = 5 :=
by {
  intro h_triangle,
  intro h_circles,
  sorry
}

end find_pq_equals_5_l515_515714


namespace range_of_omega_l515_515957

theorem range_of_omega (ω : ℝ) (h₀ : 0 < ω) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → cos (ω * x) - 1 = 0 → x = 0 ∨ x = 2 * Real.pi) →
  2 <= ω ∧ ω < 3 :=
begin
  sorry
end

end range_of_omega_l515_515957


namespace balloons_remaining_intact_l515_515409

def initial_balloons : ℕ := 200
def blown_up_after_half_hour (n : ℕ) : ℕ := n / 5
def remaining_balloons_after_half_hour (n : ℕ) : ℕ := n - blown_up_after_half_hour n

def percentage_of_remaining_balloons_blow_up (remaining : ℕ) : ℕ := remaining * 30 / 100
def remaining_balloons_after_one_hour (remaining : ℕ) : ℕ := remaining - percentage_of_remaining_balloons_blow_up remaining

def durable_balloons (remaining : ℕ) : ℕ := remaining * 10 / 100
def non_durable_balloons (remaining : ℕ) (durable : ℕ) : ℕ := remaining - durable

def twice_non_durable (non_durable : ℕ) : ℕ := non_durable * 2

theorem balloons_remaining_intact : 
  (remaining_balloons_after_half_hour initial_balloons) - 
  (percentage_of_remaining_balloons_blow_up 
    (remaining_balloons_after_half_hour initial_balloons)) - 
  (twice_non_durable 
    (non_durable_balloons 
      (remaining_balloons_after_one_hour 
        (remaining_balloons_after_half_hour initial_balloons)) 
      (durable_balloons 
        (remaining_balloons_after_one_hour 
          (remaining_balloons_after_half_hour initial_balloons))))) = 
  0 := 
by
  sorry

end balloons_remaining_intact_l515_515409


namespace max_elements_subset_l515_515631

theorem max_elements_subset (M : Set ℕ) (hM : M = {1, 2, ..., 1995}) (A : Set ℕ)
  (hA_sub : A ⊆ M) (hA_cond : ∀ x ∈ A, 15 * x ∉ A) : ∃ n, n = 1870 ∧ #(A) ≤ n :=
by
  sorry

end max_elements_subset_l515_515631


namespace problem_statement_l515_515647

theorem problem_statement
  (a b c d : ℝ)
  (h1 : abs a > 1)
  (h2 : abs b > 1)
  (h3 : abs c > 1)
  (h4 : abs d > 1)
  (h5 : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) + (1 / (d - 1)) > 0 :=
begin
  sorry
end

end problem_statement_l515_515647


namespace slope_angle_l515_515697

theorem slope_angle (x y : ℝ) (h : x - √3 * y + 2 = 0) : 
  let θ := Real.arctan (1 / √3)
  in θ = Real.pi / 6 := by
sorry

end slope_angle_l515_515697


namespace find_line_equation_l515_515944

def passesThrough (line : ℝ → ℝ) (P : ℝ × ℝ) : Prop := 
  line P.1 = P.2

def chordLength (line : ℝ → ℝ) (C : ℝ × ℝ → Prop) : ℝ := 
  4 -- This is a placeholder, the actual implementation would compute the chord length

def circle (x y : ℝ) : Prop := 
  x^2 + y^2 = 25

def line_eq1 (x y : ℝ) : Prop := 
  2 * x - y - 5 = 0

def line_eq2 (x y : ℝ) : Prop := 
  x - 2 * y + 5 = 0

theorem find_line_equation : 
  ∃ line, passesThrough line (5, 5) ∧ chordLength line (circle) = 4 ∧ 
    (∀ (x y : ℝ), passesThrough line (x, y) ↔ line_eq1 x y ∨ line_eq2 x y) := 
sorry

end find_line_equation_l515_515944


namespace calculate_AE_length_l515_515312

noncomputable def square_side_length : ℝ := 2
noncomputable def AE (x : ℝ) : Prop := 3 * (x / 3) = x
noncomputable def AE_length : ℝ := sqrt 2 - 1

theorem calculate_AE_length 
  (square_side_length : ℝ)
  (AE : ℝ → Prop)
  (AE_length : ℝ) :
  AE AE_length ↔ AE_length = sqrt 2 - 1 :=
by
  sorry

end calculate_AE_length_l515_515312


namespace smallest_possible_intersections_l515_515781

theorem smallest_possible_intersections (n : ℕ) (hn : n = 2000) :
  ∃ N : ℕ, N ≥ 3997 :=
by
  sorry

end smallest_possible_intersections_l515_515781


namespace find_unique_integer_l515_515317

noncomputable def P (x : ℚ) : ℚ := x^2023 + ∑ i in range(2023), (a i) * x^i

def is_monic (P : ℚ → ℚ) : Prop :=
∀ x, P(x) = x^2023 + ∑ i in range(2023), (a i) * x^i

def satisfies_functional_eqn (P : ℚ → ℚ) : Prop :=
∀ k, 1 ≤ k ∧ k ≤ 2023 → P(k) = k^2023 * P(1 - 1/k)

def relatively_prime (a b : ℤ) : Prop :=
nat.gcd a.nat_abs b.nat_abs = 1

def unique_integer (a b : ℤ) : ℤ := sorry

theorem find_unique_integer :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 2027 ∧ ∃ a b : ℤ, relatively_prime a b ∧ unique_integer a b = 406 ∧ (b * n - a) % 2027 = 0 :=
begin
  sorry
end

end find_unique_integer_l515_515317


namespace proof_problem_l515_515526

variables {f g : ℝ → ℝ} {a : ℝ}

-- Conditions of the problem
def even_function (h : ℝ → ℝ) := ∀ x, h (-x) = h x
def monotonic_increasing_on_nonneg (h : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → h x ≤ h y
noncomputable def F (x : ℝ) := f x + g (1 - x) - |f x - g (1 - x)|
def a_pos := a > 0

-- Proof problem
theorem proof_problem (hf_even : even_function f) 
                      (hg_even : even_function g)
                      (hf_mono : monotonic_increasing_on_nonneg f) 
                      (hg_mono : monotonic_increasing_on_nonneg g) 
                      (ha : a_pos) : (F (-a) ≥ F a) ∧ (F (1 + a) ≥ F (1 - a)) :=
begin
  sorry
end

end proof_problem_l515_515526


namespace find_RS_length_l515_515211

-- Define the conditions and the problem in Lean

theorem find_RS_length
  (radius : ℝ)
  (P Q R S T : ℝ)
  (center_to_T : ℝ)
  (PT : ℝ)
  (PQ : ℝ)
  (RT TS : ℝ)
  (h_radius : radius = 7)
  (h_center_to_T : center_to_T = 3)
  (h_PT : PT = 8)
  (h_bisect_PQ : PQ = 2 * PT)
  (h_intersecting_chords : PT * (PQ / 2) = RT * TS)
  (h_perfect_square : ∃ k : ℝ, k^2 = RT * TS) :
  RS = 16 :=
by
  sorry

end find_RS_length_l515_515211


namespace find_m_l515_515967

-- Define vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, -m)
def b : ℝ × ℝ := (1, 3)

-- Define the condition for perpendicular vectors
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the problem
theorem find_m (m : ℝ) (h : is_perpendicular (a m + b) b) : m = 4 :=
sorry -- proof omitted

end find_m_l515_515967


namespace _l515_515113

open Matrix

noncomputable def matrix_2x2_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

@[simp] theorem inverse_correctness : 
  invOf (matrix_2x2_inverse) = ![![9/46, -5/46], ![2/46, 4/46]] :=
by
  sorry

end _l515_515113


namespace solution_of_equations_solution_of_inequalities_l515_515391

-- Part 1: System of equations
def system_of_equations (x y : ℝ) :=
  (3 * x + 2 * y = 12) ∧ (2 * x - y = 1)

theorem solution_of_equations : ∃ (x y : ℝ), system_of_equations x y :=
by
  use 2, 3
  dsimp [system_of_equations]
  split
  {
    norm_num
  }
  {
    norm_num
  }

-- Part 2: System of inequalities
def system_of_inequalities (x : ℝ) :=
  (-1 < x → x - 1 < 2 * x) ∧ (x ≤ 3 → 2 * (x - 3) ≤ 3 - x)

theorem solution_of_inequalities : ∀ (x : ℝ), system_of_inequalities x :=
by
  intro x
  dsimp [system_of_inequalities]
  split
  {
    intro h
    linarith
  }
  {
    intro h
    linarith
  }

#check solution_of_equations
#check solution_of_inequalities

end solution_of_equations_solution_of_inequalities_l515_515391


namespace KP_parallel_to_MN_l515_515848

open EuclideanGeometry

variables {k1 k2 : Circle} {A B K L M O P N : Point} (p : Line)

-- Given conditions from (a)
axiom intersect_circles : k1 ≠ k2 ∧ A ∈ k1 ∧ A ∈ k2 ∧ B ∈ k1 ∧ B ∈ k2
axiom center_k2_on_k1 : O ∈ k1 ∧ IsCenter O k2
axiom line_p_intersects :
  K ∈ k1 ∧ K ∈ p ∧ O ∈ p ∧
  L ∈ k2 ∧ L ∈ p ∧ M ∈ k2 ∧ M ∈ p ∧ IsBetween L K O
axiom orthogonal_projection :
  IsOrthoProjection L P AB ∧ ProjOnLine P AB

-- N is the midpoint of AB
def midpoint (A B : Point) : Point := (A + B) / 2

-- The main theorem to prove
theorem KP_parallel_to_MN
  (h1 : ∃ k1 k2 : Circle, A ∈ k1 ∧ B ∈ k1 ∧ A ∈ k2 ∧ B ∈ k2)
  (h2 : ∃ O : Point, IsCenter O k2 ∧ O ∈ k1)
  (h3 : ∃ p : Line, K ∈ k1 ∧ K ∈ p ∧ O ∈ p ∧ L ∈ k2 ∧ L ∈ p ∧ M ∈ k2 ∧ M ∈ p ∧ IsBetween L K O)
  (h4 : ∃ P : Point, IsOrthoProjection L P AB ∧ ProjOnLine P AB)
  : Parallel (lineThrough K P) (lineThrough M (midpoint A B))
  :=
sorry

end KP_parallel_to_MN_l515_515848


namespace magnitude_z10_l515_515454

noncomputable def seq_z : ℕ → ℂ
| 0     := 1 + complex.I
| (n+1) := (seq_z n)^2 - 1 + complex.I

theorem magnitude_z10 :
  complex.abs (seq_z 10) ≈ 9.607 * 10^7 := 
sorry

end magnitude_z10_l515_515454


namespace Isabel_initial_games_l515_515240

theorem Isabel_initial_games (games_given_away : ℕ) (games_left : ℕ) :
  games_given_away = 87 → games_left = 3 → 
  games_given_away + games_left = 90 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end Isabel_initial_games_l515_515240


namespace relationship_among_x_y_z_l515_515509

variable (a b c d : ℝ)

-- Conditions
variables (h1 : a < b)
variables (h2 : b < c)
variables (h3 : c < d)

-- Definitions of x, y, z
def x : ℝ := (a + b) * (c + d)
def y : ℝ := (a + c) * (b + d)
def z : ℝ := (a + d) * (b + c)

-- Theorem: Prove the relationship among x, y, z
theorem relationship_among_x_y_z (h1 : a < b) (h2 : b < c) (h3 : c < d) : x a b c d < y a b c d ∧ y a b c d < z a b c d := by
  sorry

end relationship_among_x_y_z_l515_515509


namespace q1_monotonic_increasing_intervals_q2_proof_l515_515184

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem q1_monotonic_increasing_intervals (a : ℝ) (h : a > 0) :
  (a > 1/2 ∧ (∀ x, (0 < x ∧ x < 1/a) ∨ (2 < x) → f a x > 0)) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → f a x ≥ 0)) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (0 < x ∧ x < 2) ∨ (1/a < x) → f a x > 0)) := sorry

theorem q2_proof (x : ℝ) :
  (a = 0 ∧ x > 0 → f 0 x < 2 * Real.exp x - x - 4) := sorry

end q1_monotonic_increasing_intervals_q2_proof_l515_515184


namespace percent_absent_is_correct_l515_515306

theorem percent_absent_is_correct (total_students boys girls absent_boys absent_girls : ℝ) 
(h1 : total_students = 100)
(h2 : boys = 50)
(h3 : girls = 50)
(h4 : absent_boys = boys * (1 / 5))
(h5 : absent_girls = girls * (1 / 4)):
  (absent_boys + absent_girls) / total_students * 100 = 22.5 :=
by 
  sorry

end percent_absent_is_correct_l515_515306


namespace increasing_interval_correct_l515_515576

noncomputable def increasing_interval_of_symmetric_sine_function 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (k : ℤ)
  (h1 : f = (λ x, 2 * sin(ω * x - π / 3)))
  (h2 : 0 < ω)
  (h3 : ω < 2 * π)
  (h4 : ∀ x, f(-1 / 6 - x) = f(x))
  : set ℝ :=
  {x | ∃ k : ℤ, -1 / 6 + 2 * k ≤ x ∧ x ≤ 5 / 6 + 2 * k}

theorem increasing_interval_correct 
  (f : ℝ → ℝ) 
  (ω : ℝ) 
  (k : ℤ)
  (h1 : f = (λ x, 2 * sin(ω * x - π / 3)))
  (h2 : 0 < ω)
  (h3 : ω < 2 * π)
  (h4 : ∀ x, f(-1 / 6 - x) = f(x))
  : increasing_interval_of_symmetric_sine_function f ω k h1 h2 h3 h4 = {x | ∃ (k : ℤ), -1 / 6 + 2 * k ≤ x ∧ x ≤ 5 / 6 + 2 * k} :=
by
  sorry

end increasing_interval_correct_l515_515576


namespace election_total_polled_votes_l515_515443

theorem election_total_polled_votes (V L W: ℕ) (H1: L = 0.45 * V) (H2: W = L + 9000) (H3: W + L = V) (H4: 83 = 83) : V + 83 = 90083 := 
sorry

end election_total_polled_votes_l515_515443


namespace maximize_S_n_l515_515856

variable (a_1 d : ℝ)
noncomputable def S (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d

theorem maximize_S_n {n : ℕ} (h1 : S 17 > 0) (h2 : S 18 < 0) : n = 9 := sorry

end maximize_S_n_l515_515856


namespace least_odd_prime_factor_2023_6_plus_1_l515_515481

theorem least_odd_prime_factor_2023_6_plus_1 : ∃ p : ℕ, prime p ∧ odd p ∧ p = 37 ∧ p ∣ (2023 ^ 6 + 1) ∧ ∀ q, prime q ∧ odd q ∧ q ∣ (2023 ^ 6 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_l515_515481


namespace radius_and_area_l515_515436

variables {R : ℚ} {S : ℚ}

-- Definitions based on given conditions
def isosceles_triangle (A B C : Prop) : Prop := B = C
def circle (O : Prop) (R : ℚ) : Prop := true
def chord_parallel_to_base (LM PQ BC : Prop) : Prop := true
def intersect_at_points (AB D T : Prop) : Prop := AD = DT ∧ DT = TB
def chord_lengths (LM PQ : ℚ) (L_value P_value : ℚ) : Prop := LM = L_value ∧ PQ = P_value
def center_between_lines (O LM PQ : Prop) : Prop := true

-- Given variables
axiom A B C O : Prop
axiom AD DT TB : ℚ
axiom L_value : ℚ := 10 / sqrt 3
axiom P_value : ℚ := 2 * sqrt 26 / sqrt 3

-- Given conditions
axiom h1 : isosceles_triangle A B C
axiom h2 : circle O R
axiom h3 : chord_parallel_to_base LM PQ BC
axiom h4 : intersect_at_points AB D T
axiom h5 : chord_lengths LM PQ L_value P_value
axiom h6 : center_between_lines O LM PQ

-- To prove
theorem radius_and_area : (R = 37 / 12) ∧ (S = 6) := by
  sorry

end radius_and_area_l515_515436


namespace hexagon_area_ratio_problem_solution_l515_515252

-- Define the regular hexagon and the given conditions
variable (A B C D E F G H I J K L M N O P Q R : Point) 
variable (mid_G_AB : Midpoint G A B) (mid_H_BC : Midpoint H B C) 
variable (mid_I_CD : Midpoint I C D) (mid_J_DE : Midpoint J D E) 
variable (mid_K_EF : Midpoint K E F) (mid_L_FA : Midpoint L F A)
variable (half_dist_MA : Dist M A = 1/2 * Dist G A)
variable (half_dist_NB : Dist N B = 1/2 * Dist H B)
variable (half_dist_OC : Dist O C = 1/2 * Dist I C)
variable (half_dist_PD : Dist P D = 1/2 * Dist J D)
variable (half_dist_QE : Dist Q E = 1/2 * Dist K E)
variable (half_dist_RF : Dist R F = 1/2 * Dist L F)
variable (side_length : ∀ x y, (x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∨ x = F) → 
                             (y = A ∨ y = B ∨ y = C ∨ y = D ∨ y = E ∨ y = F) → 
                             x ≠ y → Dist x y = 4)

theorem hexagon_area_ratio : 
  (area (hexagon M N O P Q R)) / (area (hexagon A B C D E F)) = 7 / 4 := 
sorry

theorem problem_solution : 
  m = 7 ∧ n = 4 ∧ m + n = 11 := 
sorry

end hexagon_area_ratio_problem_solution_l515_515252


namespace triangle_area_approx_l515_515579

theorem triangle_area_approx :
  ∀ (a b c : ℝ), a = 32 ∧ b = 27 ∧ c = 12 →
  let s := (a + b + c) / 2 in
  abs ((s * (s - a) * (s - b) * (s - c)).sqrt - 47.82) < 0.01 :=
by
  intros a b c h
  rcases h with ⟨ha, hb, hc⟩
  let s := (a + b + c) / 2
  sorry

end triangle_area_approx_l515_515579


namespace profit_is_1500_l515_515713

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l515_515713


namespace probability_second_term_3_l515_515622

open Finset
open Fintype

noncomputable def permutations_without_1_2 : Finset (Fin 6 → Fin 6) :=
  univ.filter (λ f, f 0 ≠ 0 ∧ f 0 ≠ 1)

noncomputable def favorable_permutations : Finset (Fin 6 → Fin 6) :=
  permutations_without_1_2.filter (λ f, f 1 = 2)

theorem probability_second_term_3 (a b : ℕ) (h : Nat.gcd a b = 1): 
  (a : ℚ) / b = (favorable_permutations.card : ℚ) / (permutations_without_1_2.card : ℚ) → a + b = 23 := by
  sorry

end probability_second_term_3_l515_515622


namespace find_constant_a_l515_515625

noncomputable theory

open Real

def f (x : ℝ) (a : ℝ) : ℝ := - (π / x) + sin x + a^2 * sin (x + π / 4)

theorem find_constant_a (a : ℝ) :
  (∃ x, -π / x + sin x + a^2 * sin (x + π / 4) = 3) →
  a = sqrt 3 ∨ a = - sqrt 3 :=
begin
  sorry
end

end find_constant_a_l515_515625


namespace arctan_tan_expr_is_75_degrees_l515_515072

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l515_515072


namespace burger_cost_l515_515046

theorem burger_cost 
    (b s : ℕ) 
    (h1 : 5 * b + 3 * s = 500) 
    (h2 : 3 * b + 2 * s = 310) :
    b = 70 := by
  sorry

end burger_cost_l515_515046


namespace find_m_if_circles_are_tangent_l515_515986

noncomputable def are_circles_externally_tangent (m : ℝ) : Prop := 
  let c1_center := (1 : ℝ, 1 : ℝ)
  let c1_radius := (2 : ℝ)
  let c2_center := (4 : ℝ, 5 : ℝ)
  let c2_radius := real.sqrt (35 - m)
  dist c1_center c2_center = c1_radius + c2_radius

theorem find_m_if_circles_are_tangent : 
  are_circles_externally_tangent m → m = 26 := 
sorry

end find_m_if_circles_are_tangent_l515_515986


namespace arithmetic_sequence_geometric_condition_l515_515151

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + 2) 
  (h_geom : (a 1)^2 = a 0 * a 3) : a 0 = 2 := 
by {
  -- Definitions based on given conditions
  let a1 := a 0,
  let a2 := a 1,
  let a4 := a 3,

  -- Rewriting conditions
  have ha2 : a2 = a1 + 2 := by rw h 0,
  have ha4 : a4 = a1 + 6 := by { rw h 2, rw h 1, rw h 0, simp },

  -- Applying the geometric condition
  rw [ha2, ha4] at h_geom,
  have h_simplified := h_geom.symm,

  -- This is where we would perform the algebraic steps to solve for a1,
  -- but since we're asked to only state the theorem, we conclude with
  -- the target value that needs to be proven as per the problem constraints

  -- Placeholder for the proof
  sorry
}

end arithmetic_sequence_geometric_condition_l515_515151


namespace mia_money_l515_515284

variable (DarwinMoney MiaMoney : ℕ)

theorem mia_money :
  (MiaMoney = 2 * DarwinMoney + 20) → (DarwinMoney = 45) → MiaMoney = 110 := by
  intros h1 h2
  rw [h2] at h1
  rw [h1]
  sorry

end mia_money_l515_515284


namespace stella_profit_l515_515315

theorem stella_profit 
    (dolls : ℕ) (doll_price : ℕ) 
    (clocks : ℕ) (clock_price : ℕ) 
    (glasses : ℕ) (glass_price : ℕ) 
    (cost : ℕ) :
    dolls = 3 →
    doll_price = 5 →
    clocks = 2 →
    clock_price = 15 →
    glasses = 5 →
    glass_price = 4 →
    cost = 40 →
    (dolls * doll_price + clocks * clock_price + glasses * glass_price - cost) = 25 := 
by 
  intros h_dolls h_doll_price h_clocks h_clock_price h_glasses h_glass_price h_cost
  rw [h_dolls, h_doll_price, h_clocks, h_clock_price, h_glasses, h_glass_price, h_cost]
  norm_num
  sorry

end stella_profit_l515_515315


namespace ratio_of_areas_l515_515823

noncomputable def radius_of_circle (r : ℝ) : ℝ := r

def equilateral_triangle_side_length (r : ℝ) : ℝ := r * Real.sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem ratio_of_areas (r : ℝ) : 
  ∃ K : ℝ, K = (3 * Real.sqrt 3) / (4 * Real.pi) → 
  (area_of_equilateral_triangle (equilateral_triangle_side_length r)) / (area_of_circle r) = K := 
by 
  sorry

end ratio_of_areas_l515_515823


namespace smallest_candy_distribution_cirle_l515_515666

theorem smallest_candy_distribution_cirle :
  ∃ (candies: Fin 7 → ℕ), (∀ i, candies i ≥ 1) 
  ∧ (Function.Injective candies)
  ∧ (∀ i, Nat.gcd (candies i) (candies ((i + 1) % 7)) > 1)
  ∧ (∀ p : ℕ, Nat.Prime p → ∃ i, ¬ p ∣ candies i)
  ∧ (Finset.univ.sum candies = 44) :=
by
  sorry

end smallest_candy_distribution_cirle_l515_515666


namespace arctan_tan_75_sub_2_tan_30_eq_l515_515074

theorem arctan_tan_75_sub_2_tan_30_eq :
  arctan (tan (75 * real.pi / 180) - 2 * tan (30 * real.pi / 180)) * 180 / real.pi = 75 :=
sorry

end arctan_tan_75_sub_2_tan_30_eq_l515_515074


namespace trapezoid_CM_l515_515230

theorem trapezoid_CM (ABCD : Trapezoid) (A B C D M K : Point) (AD BC : ℝ) (hAD : AD = 8) (hBC : BC = 4)
  (h_trap : is_trapezoid ABCD A B C D) (h_on_ext : on_extension C B M) 
  (h_AM : intersects_line A M CD K)
  (h_area_ratio : area_triangle A K D = (1 / 4) * area_trapezoid ABCD) :
  (length_segment C M) = (40 / 3) := sorry

end trapezoid_CM_l515_515230


namespace molecular_weight_correct_l515_515377

noncomputable def molecular_weight : ℚ :=
  let C_weight := 12.01 in
  let H_weight := 1.008 in
  let O_weight := 16.00 in
  let N_weight := 14.01 in
  let S_weight := 32.07 in
  let C_atoms := 10 in
  let H_atoms := 16 in
  let O_atoms := 12 in
  let N_atoms := 4 in
  let S_atoms := 2 in
  (C_atoms * C_weight) + (H_atoms * H_weight) + (O_atoms * O_weight) + (N_atoms * N_weight) + (S_atoms * S_weight)

theorem molecular_weight_correct : molecular_weight ≈ 448.41 := 
by 
  sorry

end molecular_weight_correct_l515_515377


namespace tan_45_degree_l515_515852

-- Definitions of the problem setup
def point_Q := (1 / Real.sqrt 2, 1 / Real.sqrt 2)

theorem tan_45_degree : Real.tan (Real.pi / 4) = 1 := by
  -- sorry to skip the proof
  sorry

end tan_45_degree_l515_515852


namespace distinct_units_digit_perfect_square_mod_7_l515_515195

theorem distinct_units_digit_perfect_square_mod_7 :
  let squares_mod_7 := {0^2 % 7, 1^2 % 7, 2^2 % 7, 3^2 % 7, 4^2 % 7, 5^2 % 7, 6^2 % 7}
  in squares_mod_7.size = 4 := 
by
  let squares_mod_7 := {0 % 7, (1 * 1) % 7, (2 * 2) % 7, (3 * 3) % 7, (4 * 4) % 7, (5 * 5) % 7, (6 * 6) % 7}
  have h : squares_mod_7 = {0, 1, 4, 2} := sorry
  rw h
  have h_distinct : ({0, 1, 4, 2} : Finset ℕ).card = 4 := sorry
  exact h_distinct

end distinct_units_digit_perfect_square_mod_7_l515_515195


namespace fuse_length_must_be_80_l515_515870

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80_l515_515870


namespace quadratic_root_l515_515163

theorem quadratic_root (m : ℝ) (h : m^2 + 2 * m - 1 = 0) : 2 * m^2 + 4 * m = 2 := by
  sorry

end quadratic_root_l515_515163


namespace cf_tangent_to_circle_l515_515247

open Triangle Angle Circle Point

-- Helper definitions, considering about the geometric figures and properties.
structure Triangle (A B C : Point) : Prop :=
  acute : ∀ α : Angle, α ∈ [A, B, C] → α < 90

structure IsFootOfAltitude (C D : Point) (ABC : Triangle) : Prop :=
  altitude_right_angle : Angle.mk (C - A) (C - D) = 90

structure IsAngleBisector (E : Point) (B : Point) (ABC : Triangle) : Prop :=
  bisects : Angle.bisects E B ABC

structure Circumcircle (A D E : Point) (ω : Circle) : Prop :=
  circumcircle : ω = Circle.mk (A, D, E)

structure IntersectsAt (B E F : Point) (ω : Circle) : Prop :=
  intersects : F ∈ ω -- Intersection point also part of the circle

-- Declare the theorem.
theorem cf_tangent_to_circle (A B C D E F : Point) (ABC : Triangle A B C) (hD : IsFootOfAltitude C D ABC)
  (hE : IsAngleBisector E B ABC) (ω : Circle) (hCircum : Circumcircle A D E ω) (hIntersects : IntersectsAt B E F ω)
  (hAngle : Angle.mk (A - D) (F - D) = 45) :
  IsTangent (C - F) ω :=
by
  sorry

end cf_tangent_to_circle_l515_515247


namespace max_available_is_sunday_l515_515348

def Alice_availability : List Bool := [false, true, true, false, true, false, true]
def Bob_availability : List Bool := [true, false, false, true, false, true, true]
def Cara_availability : List Bool := [false, false, true, false, true, false, false]
def Dave_availability : List Bool := [true, true, false, true, true, false, true]
def Ella_availability : List Bool := [false, true, true, true, true, false, false]

def team_availability : List (List Bool) := 
  [Alice_availability, Bob_availability, Cara_availability, Dave_availability, Ella_availability]

def count_available (day_index : Nat) : Nat :=
  team_availability.map (λ member_avail => if member_avail.get! day_index then 1 else 0).sum

theorem max_available_is_sunday :
  ∀ (day_index : Nat), count_available day_index ≤ count_available 6 :=
by
  sorry

end max_available_is_sunday_l515_515348


namespace set_A_cardinality_l515_515353

open Set Classical

theorem set_A_cardinality (U A B : Set α) [Fintype U] [Fintype A] [Fintype B]
  (hU : Fintype.card U = 193)
  (hB : Fintype.card B = 49)
  (hNeither : Fintype.card (U \ (A ∪ B)) = 59)
  (hAB : Fintype.card (A ∩ B) = 25) : Fintype.card A = 110 :=
by
  sorry

end set_A_cardinality_l515_515353


namespace division_and_subtraction_l515_515854

theorem division_and_subtraction :
  (-96) / (-24) - 3 = 1 :=
by
  have h1 : (-96) / (-24) = 96 / 24 := by sorry
  have h2 : 96 / 24 = 4 := by sorry
  rw [h1, h2]
  norm_num -- simplify to get the final answer

end division_and_subtraction_l515_515854


namespace arith_seq_formula_geom_seq_sum_l515_515518

-- Definitions for condition 1: Arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  (a 4 = 7) ∧ (a 10 = 19)

-- Definitions for condition 2: Sum of the first n terms of {a_n}
def sum_arith_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Definitions for condition 3: Geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop :=
  (b 1 = 2) ∧ (∀ n, b (n + 1) = b n * 2)

-- Definitions for condition 4: Sum of the first n terms of {b_n}
def sum_geom_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n, T n = (b 1 * (1 - (2 ^ n))) / (1 - 2)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula (a : ℕ → ℤ) (S : ℕ → ℤ) :
  arithmetic_seq a ∧ sum_arith_seq S a → 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, S n = n ^ 2) :=
sorry

-- Proving the sum of the first n terms for geometric sequence
theorem geom_seq_sum (b : ℕ → ℤ) (T : ℕ → ℤ) (S : ℕ → ℤ) :
  geometric_seq b ∧ sum_geom_seq T b ∧ b 4 = S 4 → 
  (∀ n, T n = 2 ^ (n + 1) - 2) :=
sorry

end arith_seq_formula_geom_seq_sum_l515_515518


namespace number_of_integers_satisfying_inequality_l515_515196

theorem number_of_integers_satisfying_inequality :
  {x : ℤ | |7 * x - 5| ≤ 3 * x + 4}.card = 3 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l515_515196


namespace proof_problem_l515_515748

theorem proof_problem 
  (mean_median : ∀ (X : list ℝ), (mean(X) reflects more information and is more sensitive to extreme values in X)
  (P60_data: list ℝ := [2, 3, 4, 5, 6, 7, 8, 9])
  (P60_value : 60th percentile of P60_data is 6)
  (prob_cond : ∀ (A B : Prop) (P : Prop → ℝ), P(A) > 0 → P(B) > 0 → P(B | A) = P(B) → P(A | B) = P(A))
  (abs_r_corr : ∀ (r : ℝ), |r| ∼ 1 → stronger linear correlation )
  : (A ∧ C ∧ D) :=
by   
  sorry

end proof_problem_l515_515748


namespace functional_equation_to_linear_l515_515667

-- Define that f satisfies the Cauchy functional equation
variable (f : ℕ → ℝ)
axiom cauchy_eq (x y : ℕ) : f (x + y) = f x + f y

-- The theorem we want to prove
theorem functional_equation_to_linear (h : ∀ n k : ℕ, f (n * k) = n * f k) : ∃ a : ℝ, ∀ n : ℕ, f n = a * n :=
by
  sorry

end functional_equation_to_linear_l515_515667


namespace tetrahedron_fourth_vertex_l515_515422

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ),
    (distance (1, 3, 2) (5, 4, 1) = 3 * real.sqrt 2) ∧
    (distance (1, 3, 2) (x, y, z) = 3 * real.sqrt 2) ∧
    (distance (5, 4, 1) (x, y, z) = 3 * real.sqrt 2) ∧
    (distance (4, 2, 6) (x, y, z) = 3 * real.sqrt 2) ∧
    ((x, y, z) = (4, 0, 8)) :=
by 
  sorry

end tetrahedron_fourth_vertex_l515_515422


namespace mahesh_worked_days_l515_515277

-- Definitions
def mahesh_work_days := 45
def rajesh_work_days := 30
def total_work_days := 54

-- Theorem statement
theorem mahesh_worked_days (maheshrate : ℕ := mahesh_work_days) (rajeshrate : ℕ := rajesh_work_days) (totaldays : ℕ := total_work_days) :
  ∃ x : ℕ, x = totaldays - rajesh_work_days := by
  apply Exists.intro (54 - 30)
  simp
  sorry

end mahesh_worked_days_l515_515277


namespace right_triangle_count_l515_515295

def point := ℝ × ℝ

def distance (p q : point) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def num_right_triangles_with_area (P Q : point) (area : ℝ) : ℕ :=
  if distance P Q = 10 ∧ area = 15 then 8 else 0

theorem right_triangle_count (P Q : point) (h₁ : distance P Q = 10) (h₂ : area = 15) :
  num_right_triangles_with_area P Q area = 8 :=
sorry

end right_triangle_count_l515_515295


namespace number_of_arithmetic_sequences_l515_515266

theorem number_of_arithmetic_sequences (n : ℕ) (S : Finset ℕ)
  (hS : S = Finset.range (n + 1)) :
  ∃ A : Finset (Finset ℕ), (∀ a ∈ A, (∃ d : ℕ, 1 ≤ d ∧ arithmetic_seq_with_diff d a ∧ a ⊆ S ∧ ∀ x ∈ S \ a, ¬arithmetic_seq_with_diff d (insert x a))) ∧ A.card = Nat.floor (n^2 / 4) := 
sorry

def arithmetic_seq_with_diff (d : ℕ) (s : Finset ℕ) : Prop :=
∃ a0 : ℕ, s = Finset.image (λ i, a0 + i * d) (Finset.range s.card)

end number_of_arithmetic_sequences_l515_515266


namespace fence_length_l515_515033

theorem fence_length (side_length : ℕ) (h : side_length = 15) : 4 * side_length = 60 :=
by
  rw [h]
  -- Solution steps can be skipped with 'sorry' as proof is not required.
  sorry

end fence_length_l515_515033


namespace minimum_distance_l515_515201

-- Define the function and conditions
def f (x : ℝ) : ℝ := -x^2 + 3 * Real.log x
def g (x : ℝ) : ℝ := x + 2

-- Define points P and Q
def P (a b : ℝ) : Prop := b = f a
def Q (c d : ℝ) : Prop := d = g c

-- Prove the minimum distance
theorem minimum_distance (a b c d : ℝ) (hP : P a b) (hQ : Q c d) 
  (ha_pos : 0 < a) (hc_pos : 0 < c) : (a - c)^2 + (b - d)^2 ≥ 8 :=
by
  sorry

end minimum_distance_l515_515201


namespace probability_jack_queen_queen_l515_515362

theorem probability_jack_queen_queen : 
  let deck_size := 52 in
  let jacks := 4 in
  let queens := 4 in
  let prob_jack := (jacks : ℚ) / deck_size in
  let prob_queen1 := (queens : ℚ) / (deck_size - 1) in
  let prob_queen2 := (queens - 1 : ℚ) / (deck_size - 2) in
  
  prob_jack * prob_queen1 * prob_queen2 = 2 / 5525 :=
by
  sorry

end probability_jack_queen_queen_l515_515362


namespace exists_special_cubic_polynomial_l515_515236

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l515_515236


namespace number_of_people_with_numbers_leq_threshold_l515_515244

def Jungkook := 0.8
def Yoongi := 1/2
def Yoojeong := 0.9
def threshold := 0.3

theorem number_of_people_with_numbers_leq_threshold : 
  (if Jungkook ≤ threshold then 1 else 0) + (if Yoongi ≤ threshold then 1 else 0) + (if Yoojeong ≤ threshold then 1 else 0) = 0 :=
by
  sorry

end number_of_people_with_numbers_leq_threshold_l515_515244


namespace f_monotonic_f_odd_function_range_of_a_l515_515143

variable {R : Type} [LinearOrderedField R] (f : R → R)
variable {a x x1 x2 : R}

-- Given conditions
axiom f_eq_fxy_fy_f0 : ∀ x y : R, f x = f (x - y) + f y + f 0
axiom f_lt_0 : ∀ x : R, x > 0 → f x < 0

-- Correct answers to be proven
theorem f_monotonic : ∀ x1 x2 : R, x1 > x2 → f x1 < f x2 := sorry

theorem f_odd_function : ∀ x : R, f (-x) = -f x := sorry

theorem range_of_a : (∃ x : R, (1 < x ∧ x < 2) ∧ f(x * x - a * x) + f(x - 3) > 0) → a ∈ Set.Ici (-1) := sorry

end f_monotonic_f_odd_function_range_of_a_l515_515143


namespace solvable_mod_p_l515_515668

theorem solvable_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (h100 : p ≤ 100) : 
  ∃ y x : ℕ, y^37 ≡ x^3 + 11 [ZMOD p] := 
begin
  sorry -- proof omitted
end

end solvable_mod_p_l515_515668


namespace ratio_of_areas_l515_515821

noncomputable def radius_of_circle (r : ℝ) : ℝ := r

def equilateral_triangle_side_length (r : ℝ) : ℝ := r * Real.sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem ratio_of_areas (r : ℝ) : 
  ∃ K : ℝ, K = (3 * Real.sqrt 3) / (4 * Real.pi) → 
  (area_of_equilateral_triangle (equilateral_triangle_side_length r)) / (area_of_circle r) = K := 
by 
  sorry

end ratio_of_areas_l515_515821


namespace number_of_elements_l515_515324

theorem number_of_elements
  (init_avg : ℕ → ℝ)
  (correct_avg : ℕ → ℝ)
  (incorrect_num correct_num : ℝ)
  (h1 : ∀ n : ℕ, init_avg n = 17)
  (h2 : ∀ n : ℕ, correct_avg n = 20)
  (h3 : incorrect_num = 26)
  (h4 : correct_num = 56)
  : ∃ n : ℕ, n = 10 := sorry

end number_of_elements_l515_515324


namespace value_of_a_minus_b_l515_515549

theorem value_of_a_minus_b (a b : ℝ) (h₁ : |a| = 2) (h₂ : |b| = 5) (h₃ : a < b) :
  a - b = -3 ∨ a - b = -7 := 
sorry

end value_of_a_minus_b_l515_515549


namespace optimal_play_bob_wins_l515_515431

-- Define the conditions: the game setting, rules and optimal play.
def game_condition (n : Nat) : Prop :=
  ∀ m : Nat, (m >= 0) ∧ (m < n) →
  ∃ k : Nat, (k divides n) ∧ (m = n - k)

-- The main theorem to prove based on the conditions stated.
theorem optimal_play_bob_wins : game_condition 2020 → ∃ winner : String, winner = "Bob" :=
  by
    sorry

end optimal_play_bob_wins_l515_515431


namespace min_value_of_na_l515_515206

def a (n : ℕ) : ℚ
| 0     := 0  -- Not actually used, since sequence starts at n = 1
| 1     := -1
| 2     := -1 / 2
| (n+3) := (4 * a (n + 2) - 2 * a (n + 1) + 3) / 2

def na (n : ℕ) : ℚ := n * a n

theorem min_value_of_na :
  ∃ n : ℕ, n ≥ 1 ∧ na n = -5 / 4 :=
sorry

end min_value_of_na_l515_515206


namespace maximum_magical_pairs_l515_515022

/-- Maximum magical pairs of adjacent numbers -/
theorem maximum_magical_pairs (nums : List ℕ) (h : ∀ n ∈ nums, 1 ≤ n ∧ n ≤ 30) 
  (adj_magical_pairs : ∀ (i : ℕ), i < nums.length - 1 → ((nums[i] + nums[i + 1]) % 7 = 0) → True) :
  ∃ (perm : List ℕ), perm ~ nums ∧ (finset.range (nums.length - 1)).card - 3 = 26 := 
by sorry

end maximum_magical_pairs_l515_515022


namespace equation_solution_unique_or_not_l515_515130

theorem equation_solution_unique_or_not (a b : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2) ↔ 
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
by
  sorry

end equation_solution_unique_or_not_l515_515130


namespace number_of_real_solutions_l515_515456

theorem number_of_real_solutions :
  set_of (λ x : ℝ, complex.abs (1 - 2 * x * complex.I) = 2).finite.card = 2 :=
by {
  -- Proof would go here
  sorry
}

end number_of_real_solutions_l515_515456


namespace min_value_sqrt_sum_l515_515256

variable {a b : ℝ}

theorem min_value_sqrt_sum (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  sqrt (a^2 + 1) + sqrt (b^2 + 4) ≥ sqrt 10 := 
by
  sorry

end min_value_sqrt_sum_l515_515256


namespace interest_rate_l515_515034

theorem interest_rate (
  (SI : ℚ) 
  (P : ℚ) 
  (T : ℕ) 
  (hSI : SI = 4016.25) 
  (hP : P = 7302.272727272727) 
  (hT : T = 5)
) : 
  let R := (SI * 100) / (P * T) 
  in 
  R = 11 := 
by 
  sorry

end interest_rate_l515_515034


namespace estimate_number_of_fish_in_pond_l515_515367

theorem estimate_number_of_fish_in_pond
  (marked_first_catch : ℕ)
  (total_second_catch : ℕ)
  (marked_second_catch : ℕ)
  (uniform_distribution : Prop) :
  marked_first_catch = 50 →
  total_second_catch = 200 →
  marked_second_catch = 2 →
  uniform_distribution →
  (marked_first_catch : ℚ) / (marked_second_catch / total_second_catch) = 5000 := 
by {
  intros h1 h2 h3 h4,
  simp [h1, h2, h3],
  norm_num,
  exact rfl,
}

end estimate_number_of_fish_in_pond_l515_515367


namespace ec_value_l515_515644

theorem ec_value (AB AD : ℝ) (EFGH1 EFGH2 : ℝ) (x : ℝ)
  (h1 : AB = 2)
  (h2 : AD = 1)
  (h3 : EFGH1 = 1 / 2 * AB)
  (h4 : EFGH2 = 1 / 2 * AD)
  (h5 : 1 + 2 * x = 1)
  : x = 1 / 3 :=
by sorry

end ec_value_l515_515644


namespace total_trolls_l515_515099

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l515_515099


namespace ellipse_equation_and_point_C_coordinates_l515_515520

theorem ellipse_equation_and_point_C_coordinates
  (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)
  (ecc : real.sqrt (a^2 - b^2) / a = real.sqrt 2 / 2)
  (chord_len_perp_x_axis : ∀ x, (b^2/a) = real.sqrt 2 / 2)
  (A B : ℝ × ℝ) (on_ellipse_A : A.1^2 / a^2 + A.2^2 / b^2 = 1)
  (on_ellipse_B : B.1^2 / a^2 + B.2^2 / b^2 = 1)
  (C : ℝ × ℝ) (on_parabola_C : C.2^2 = (1 / 2) * C.1)
  (centroid_origin : (A.1 + B.1 + C.1) / 3 = 0 ∧ (A.2 + B.2 + C.2) / 3 = 0)
  (area_triangle : real.abs ((A.1 - C.1) * (B.2 - B.2) - (B.1 - C.1) * (A.2 - B.2)) / 2 = (3 * real.sqrt 6) / 4) :
  (a = real.sqrt 2 ∧ b = 1) ∧
  (C = (1, real.sqrt 2 / 2) ∨ C = (1, - real.sqrt 2 / 2) ∨ C = (2, 1) ∨ C = (2, -1)) :=
sorry

end ellipse_equation_and_point_C_coordinates_l515_515520


namespace t_shirt_sale_revenue_per_minute_l515_515323

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l515_515323


namespace cafeteria_pies_l515_515389

theorem cafeteria_pies (total_apples handed_out_per_student apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_per_student = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_per_student) / apples_per_pie = 5 := by
  sorry

end cafeteria_pies_l515_515389


namespace exponent_equality_l515_515562

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l515_515562


namespace cyclic_tangential_quadrilateral_collinear_l515_515298

open EuclideanGeometry

-- Assume a tangential quadrilateral, which is also cyclic
-- Assume the center of the circumscribed circle (circumcenter)
-- Assume the center of the inscribed circle (incenter)
-- Assume the intersection point of the diagonals

variables {A B C D P Q R S M O I : Point}

-- Definitions of cyclic quadrilateral
def cyclic_quadrilateral (A B C D : Point) : Prop := 
CircumscribedCircleContains A B C D

-- Definitions of tangential quadrilateral
def tangential_quadrilateral (A B C D : Point) : Prop :=
InscribedCircleTangentTo A B C D

-- Definitions of circumcenter, incenter, and intersection point of diagonals in the context
def circumcenter (A B C D : Point) : Point :=
CircumcircleCenter A B C D

def incenter (A B C D : Point) : Point :=
IncircleCenter A B C D

def intersection_of_diagonals (A B C D : Point) : Point :=
IntersectionOfLines (LineThrough A C) (LineThrough B D)

theorem cyclic_tangential_quadrilateral_collinear 
  (A B C D : Point) 
  (H1 : cyclic_quadrilateral A B C D) 
  (H2 : tangential_quadrilateral A B C D) :
  Collinear (circumcenter A B C D) 
            (incenter A B C D) 
            (intersection_of_diagonals A B C D) :=
sorry

end cyclic_tangential_quadrilateral_collinear_l515_515298


namespace equilateral_triangle_area_from_hexagon_l515_515797

theorem equilateral_triangle_area_from_hexagon (side_length : ℝ) (h : side_length = 3) :
  (∃ hexagon : ℕ → ℤ×ℤ, 
    (∀ (i : ℕ), 0 ≤ i ∧ i < 6 → dist (hexagon (i)) (hexagon (i + 1)) = side_length) ∧
    dist (hexagon 0) (hexagon 2) = side_length ∧
    dist (hexagon 2) (hexagon 4) = side_length ∧
    ∃ triangle : ℕ → ℤ×ℤ, 
      triangle 0 = hexagon 0 ∧
      triangle 1 = hexagon 2 ∧
      triangle 2 = hexagon 4 ∧
      (dist (triangle 0) (triangle 1) = side_length ∧
      dist (triangle 1) (triangle 2) = side_length ∧
      dist (triangle 2) (triangle 0) = side_length)
  ) → ∃ (area : ℝ), area = (9 * real.sqrt 3) / 4 :=
by
  sorry

end equilateral_triangle_area_from_hexagon_l515_515797


namespace ratio_of_areas_l515_515826

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l515_515826


namespace sum_of_numbers_less_than_0_4_l515_515727

theorem sum_of_numbers_less_than_0_4 : 
  ∑ x in {0.8, 0.5, 0.9} ∩ set_of (λ x, x < 0.4), x = 0 := 
by
  sorry

end sum_of_numbers_less_than_0_4_l515_515727


namespace combined_weight_is_correct_l515_515605

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end combined_weight_is_correct_l515_515605


namespace max_prime_p_l515_515318

-- Define the variables and conditions
variable (a b : ℕ)
variable (p : ℝ)

-- Define the prime condition
def is_prime (n : ℝ) : Prop := sorry -- Placeholder for the prime definition

-- Define the equation condition
def p_eq (p : ℝ) (a b : ℕ) : Prop := 
  p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))

-- The theorem to prove
theorem max_prime_p (a b : ℕ) (p_max : ℝ) :
  (∃ p, is_prime p ∧ p_eq p a b) → p_max = 5 := 
sorry

end max_prime_p_l515_515318


namespace rectangles_in_grid_at_least_three_cells_l515_515330

theorem rectangles_in_grid_at_least_three_cells :
  let number_of_rectangles (n : ℕ) := (n + 1).choose 2 * (n + 1).choose 2
  let single_cell_rectangles (n : ℕ) := n * n
  let one_by_two_or_two_by_one_rectangles (n : ℕ) := n * (n - 1) * 2
  let total_rectangles (n : ℕ) := number_of_rectangles n - (single_cell_rectangles n + one_by_two_or_two_by_one_rectangles n)
  total_rectangles 6 = 345 :=
by
  sorry

end rectangles_in_grid_at_least_three_cells_l515_515330


namespace geometric_sequence_properties_l515_515144

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : ∀ n, S n = range (n + 1) |> List.map (a) |> List.sum)
  (h4 : 4 * a 3 = a 2 * a 4)
  (h5 : ∀ n, a (n + 1) = a 1 * q ^ n) :
  (q = 2 ∧ a 5 = 16) ∧ ∀ n, S n / a n < 2 :=
by
  sorry

end geometric_sequence_properties_l515_515144


namespace variables_in_context_l515_515401

variable (a t y : ℝ)

def is_fixed (a : ℝ) := True
def is_variable (x : ℝ) := True

theorem variables_in_context :
  is_fixed a →
  is_variable t →
  is_variable y →
  ({t, y} : set ℝ) = {x | is_variable x} :=
by
  intros ha ht hy
  sorry

end variables_in_context_l515_515401


namespace frank_position_l515_515504

theorem frank_position :
  let initial_position := 0 in
  let steps_back1 := -5 in
  let steps_forward1 := 10 in
  let steps_back2 := -2 in
  let steps_forward2 := 2 * (-steps_back2) in
  initial_position + steps_back1 + steps_forward1 + steps_back2 + steps_forward2 = 7 := by
  sorry

end frank_position_l515_515504


namespace total_area_of_pyramid_faces_l515_515737

theorem total_area_of_pyramid_faces (b l : ℕ) (hb : b = 8) (hl : l = 10) : 
  let h : ℝ := Math.sqrt (l^2 - (b / 2)^2) in
  let A : ℝ := 1 / 2 * b * h in
  let T : ℝ := 4 * A in
  T = 32 * Math.sqrt 21 := by
  -- Definitions
  have b_val : (b : ℝ) = 8 := by exact_mod_cast hb
  have l_val : (l : ℝ) = 10 := by exact_mod_cast hl

  -- Calculations
  have h_val : h = Math.sqrt (l^2 - (b / 2)^2) := rfl
  have h_simplified : h = 2 * Math.sqrt 21 := by
    rw [h_val, l_val, b_val]
    norm_num
    simp

  have A_val : A = 1 / 2 * b * h := rfl
  simp_rw [A_val, h_simplified, b_val]
  norm_num

  have T_val : T = 4 * A := rfl
  simp_rw [T_val]
  norm_num

  -- Final proof
  sorry

end total_area_of_pyramid_faces_l515_515737


namespace distance_NQ_l515_515656

theorem distance_NQ {A B C N Q : Type*} 
  (h_triangle : (A, B, C) ∈ Triangle) 
  (angle_A : angle A B C = 45)
  (side_AB : distance A B = 2 * √2)
  (side_AC : distance A C = 1)
  (rectangle_AMNB : Rectangle AM N B)
  (rectangle_APQC : Rectangle AP Q C)
  : distance N Q = 3 * √(2 + √2) :=
by sorry 

end distance_NQ_l515_515656


namespace find_a_value_l515_515566

theorem find_a_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq1 : a^b = b^a) (h_eq2 : b = 3 * a) : a = Real.sqrt 3 :=
  sorry

end find_a_value_l515_515566


namespace product_of_roots_l515_515865

theorem product_of_roots :
  ∀ (x : ℝ), (|x|^2 - 3 * |x| - 10 = 0) →
  (∃ a b : ℝ, a ≠ b ∧ (|a| = 5 ∧ |b| = 5) ∧ a * b = -25) :=
by {
  sorry
}

end product_of_roots_l515_515865


namespace Bons_wins_probability_l515_515670

theorem Bons_wins_probability :
  let p : ℚ := 5 / 11 in
  (∀ n : ℕ, n % 2 = 0 → valid_roll_sequence n → total_wins n = true) →
  (∀ k : ℕ, k % 2 = 1 → total_wins k = false) →
  (prob_roll_six = 1 / 6) →
  (p = 5 / 11) :=
begin
  sorry
end

end Bons_wins_probability_l515_515670


namespace sum_of_first_10_common_elements_l515_515124

open Nat

def arithmetic_progression (n : ℕ) : ℤ := 4 + 3 * n

def geometric_progression (k : ℕ) : ℤ := 20 * 2^k

def is_common_element (x : ℤ) : Prop :=
  ∃ n k : ℕ, x = arithmetic_progression n ∧ x = geometric_progression k

def common_elements (n : ℕ) : ℕ → list ℤ
| 0     := []
| (m+1) := match (list.find_x is_common_element (geometric_progression n)) with 
  | some x => x :: common_elements (n+1) m
  | none   => common_elements (n+1) (m+1)
  end

def sum_common_elements (n : ℕ) : ℤ :=
  (common_elements 0 n).sum
  
theorem sum_of_first_10_common_elements :
  sum_common_elements 10 = 13981000 := 
sorry

end sum_of_first_10_common_elements_l515_515124


namespace cosine_of_angle_between_vectors_l515_515194

open Real
open ComplexConjugate

variables (a b : ℝ × ℝ)
def cos_angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  let c := (a.1 + b.1, a.2 + b.2)
  let d := (a.1 - b.1, a.2 - b.2)
  (c.1 * d.1 + c.2 * d.2) / (sqrt (c.1 * c.1 + c.2 * c.2) * sqrt (d.1 * d.1 + d.2 * d.2))

theorem cosine_of_angle_between_vectors :
  cos_angle_between_vectors (2,1) (-1,1) = (sqrt 5) / 5 :=
by
  sorry

end cosine_of_angle_between_vectors_l515_515194


namespace sequence_sum_inequality_l515_515914

theorem sequence_sum_inequality (S : ℕ → ℚ) (a b : ℕ → ℚ) (T : ℕ → ℚ)
  (hS : ∀ n : ℕ, S n = n^2 / 2 + 3 * n / 2)
  (ha : ∀ n : ℕ, a n = S n - S (n - 1))
  (hb : ∀ n : ℕ, b n = a (n + 2) - a n + 1 / (a (n + 1) - a n))
  (hT : ∀ n : ℕ, T n = Σ i in Finset.range (n + 1), b i) :
  ∀ n : ℕ, T n < 2 * n + 5 / 12 := 
sorry

end sequence_sum_inequality_l515_515914


namespace chess_probabilities_l515_515506

noncomputable def probability_two_dark_or_different_colors : ℚ :=
  47 / 62

noncomputable def probability_one_bishop_one_pawn_or_different_colors : ℚ :=
  18 / 31

noncomputable def probability_two_rooks_different_colors_or_same_color_different_types : ℚ :=
  91 / 248

noncomputable def probability_one_king_one_knight_or_same_color : ℚ :=
  15 / 31

noncomputable def probability_two_same_type_or_same_color : ℚ :=
  159 / 248

theorem chess_probabilities :
  let total_pieces := 32 in
  let total_draws := 496 in
  let probability_two_dark_or_different_colors_correct := total_draws = 496 → 47 / 62 = probability_two_dark_or_different_colors in
  let probability_one_bishop_one_pawn_or_different_colors_correct := total_draws = 496 → 18 / 31 = probability_one_bishop_one_pawn_or_different_colors in
  let probability_two_rooks_different_colors_or_same_color_different_types_correct := total_draws = 496 → 91 / 248 = probability_two_rooks_different_colors_or_same_color_different_types in
  let probability_one_king_one_knight_or_same_color_correct := total_draws = 496 → 15 / 31 = probability_one_king_one_knight_or_same_color in
  let probability_two_same_type_or_same_color_correct := total_draws = 496 → 159 / 248 = probability_two_same_type_or_same_color in
  probability_two_dark_or_different_colors_correct ∧
  probability_one_bishop_one_pawn_or_different_colors_correct ∧
  probability_two_rooks_different_colors_or_same_color_different_types_correct ∧
  probability_one_king_one_knight_or_same_color_correct ∧
  probability_two_same_type_or_same_color_correct :=
by
  dsimp
  intros total_pieces total_draws
  sorry

end chess_probabilities_l515_515506


namespace tan_alpha_minus_beta_l515_515942

theorem tan_alpha_minus_beta :
  ∀ (α β : ℝ), (sin α = 3/5) ∧ (π/2 < α ∧ α < π) ∧ (tan (π - β) = 1/2) → (tan (α - β) = -2/5) :=
by
  intros α β h
  sorry

end tan_alpha_minus_beta_l515_515942


namespace dihedral_angle_A_BD_C_l515_515228

noncomputable def dihedral_angle_ABC (A B C D : Point) : ℝ :=
  -- function to compute the dihedral angle goes here
  sorry -- Placeholder for the proper function

theorem dihedral_angle_A_BD_C
  (A B C D : Point)
  (hAB : dist A B = 3)
  (hBD : dist B D = 3)
  (hCD : dist C D = 3)
  (hAD : dist A D = 4)
  (hBC : dist B C = 4)
  (hAC : dist A C = 5) :
  dihedral_angle_ABC A B C D = π - arccos (1 / 10) :=
sorry

end dihedral_angle_A_BD_C_l515_515228


namespace AI1_BI2_CI3_concurrent_l515_515615

open Triangle

variables (ABC : Triangle) (X : Point) (A B C : Point) 
variables (I1 I2 I3 : Point) (X_inside : X ∈ interior ABC)

-- Conditions
variables (h1 : XA * BC = XB * AC) (h2 : XB * AC = XC * AB) (h3 : XC * AB = XA * BC)
variables (I1_incenter : is_incenter XBC I1) (I2_incenter : is_incenter XCA I2) (I3_incenter : is_incenter XAB I3)

theorem AI1_BI2_CI3_concurrent (AI1 BI2 CI3 : Line):
  concurrent AI1 BI2 CI3 :=
begin
  sorry,
end

end AI1_BI2_CI3_concurrent_l515_515615


namespace range_of_a_l515_515510

theorem range_of_a
  (a : ℝ) 
  (h₀ : a ≠ 0)
  (f : ℝ → ℝ := λ x, 2 * (sqrt 3) * (Real.sin x * Real.cos x) + 2 * (Real.cos x)^2 - 1 - a)
  (g : ℝ → ℝ := λ x, a * (Real.log (x + 3) / Real.log 2) - 2)
  (h₁ : ∃ (x₁ : ℝ), x₁ ∈ Icc 0 (Real.pi / 2) ∧ ∀ (x₂ : ℝ), x₂ ∈ Icc 1 5 → f x₁ = g x₂) :
  a ∈ Set.Icc (1/3 : ℝ) 1 :=
  sorry

end range_of_a_l515_515510


namespace regular_hexagon_perimeter_l515_515174

theorem regular_hexagon_perimeter (r : ℝ) (h : r = real.sqrt 3) : 
  let a := r in 
  let P := 6 * a in 
  P = 6 * real.sqrt 3 :=
by sorry

end regular_hexagon_perimeter_l515_515174


namespace boat_speed_greater_than_current_l515_515683

theorem boat_speed_greater_than_current (U V : ℝ) (hU_gt_V : U > V)
  (h_equation : 1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_current_l515_515683


namespace limit_of_sequence_l515_515543

-- Define the quadratic function f with parameter t ∈ ℝ
def f (x t : ℝ) := x^2 - t*x + t

-- Define the sequence a_n
noncomputable def a_n (n : ℕ) := 2 * (n : ℝ) - 1

-- Define the limit expression.
noncomputable def limit_expr (n : ℕ) : ℝ :=
  (∑ i in finset.range n, 1 / (a_n i * a_n (i + 2)))

-- The theorem (question == answer given conditions)
theorem limit_of_sequence :
  (∀ t : ℝ, (∀ x : ℝ, f x t ≤ 0 → f x t = 0) ∧ 
            (∀ x1 x2 : ℝ, 0 < x2 < x1 → f x2 t < f x1 t) → 
   (a_n 1 = 1) ∧ 
   (∀ n : ℕ, a_n n = 2 * n - 1) ∧ 
   filter.at_top.tendsto (λ n, limit_expr n) (pure (1 / 3))) :=
by { sorry }

end limit_of_sequence_l515_515543


namespace circles_intersecting_l515_515089

open Real

def Circle (x y r : ℝ) := ∀ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = r^2

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem circles_intersecting :
  ∀ (x1 y1 r1 x2 y2 r2 : ℝ),
  Circle x1 y1 r1 →
  Circle x2 y2 r2 →
  distance (x1, y1) (x2, y2) = sqrt 13 →
  r1 = 4 →
  r2 = 3 →
  1 < sqrt 13 ∧ sqrt 13 < 7 :=
by
  intros x1 y1 r1 x2 y2 r2 H1 H2 hd hr1 hr2
  sorry

end circles_intersecting_l515_515089


namespace sequence_inequality_l515_515453

theorem sequence_inequality:
  ∀ (n : ℕ), 0 < n →
  ∀ (a : ℕ → ℝ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n ≥ 1, a (n + 2) = a (n + 1) + a n) →
  (real.sqrt n (a (n + 1)) ≥ 1 + 1 / real.sqrt n (a n)) :=
begin
  sorry
end

end sequence_inequality_l515_515453


namespace angle_difference_l515_515766

-- Definitions and conditions based on the problem statement: BK is the angle bisector and angle ratios
variables (A B C K : Type) [angle : has_angle A B C K]

-- Assume the angles and their ratios
variables (angleAKB angleCKB : ℝ) (h1 : angleAKB / angleCKB = 4 / 5)
variable (h2 : angleAKB + angleCKB = 180)

-- Difference between angles A and C is 20 degrees
theorem angle_difference (h1 : angleAKB / angleCKB = 4 / 5) (h2 : angleAKB + angleCKB = 180) :
  ∃ (angleA angleC : ℝ), angleA - angleC = 20 := 
by
  sorry

end angle_difference_l515_515766


namespace partial_fraction_sum_zero_l515_515447

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l515_515447


namespace trigonometric_identity_l515_515565

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) : 
  (Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α)) * Real.cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l515_515565


namespace longest_side_eq_24_l515_515334

noncomputable def x : Real := 19 / 3

def side1 (x : Real) : Real := x + 3
def side2 (x : Real) : Real := 2 * x - 1
def side3 (x : Real) : Real := 3 * x + 5

def perimeter (x : Real) : Prop :=
  side1 x + side2 x + side3 x = 45

theorem longest_side_eq_24 : perimeter x → max (max (side1 x) (side2 x)) (side3 x) = 24 :=
by
  sorry

end longest_side_eq_24_l515_515334


namespace number_of_valid_four_digit_integers_l515_515971

theorem number_of_valid_four_digit_integers : 
  let digits := {0, 1, 2, 3, 4, 6, 8, 9}
  in (1000 <= (10^3 * {1,2,3,4,6,8,9} + 10^2 * digits + 10^1 * digits + digits)) ∧
     (10^3 * {1,2,3,4,6,8,9} + 10^2 * digits + 10^1 * digits + digits <= 9999) ∧
     (10^3 * {1,2,3,4,6,8,9} + 10^2 * digits + 10^1 * digits + digits) ∉ {5, 7}
  → 7 * 8 * 8 * 8 = 3584 :=
by
  sorry

end number_of_valid_four_digit_integers_l515_515971


namespace numeral_9_occurrences_1_to_100_numeral_1_occurrences_1_to_100_l515_515802

def count_numeral_occurrences (n : ℕ) (digit : ℕ) : ℕ :=
  (List.range (n + 1)).map (λ x => x.digits 10).join.count digit

theorem numeral_9_occurrences_1_to_100 : count_numeral_occurrences 100 9 = 11 := by sorry

theorem numeral_1_occurrences_1_to_100 : count_numeral_occurrences 100 1 = 110 := by sorry

end numeral_9_occurrences_1_to_100_numeral_1_occurrences_1_to_100_l515_515802


namespace hexagon_area_l515_515794

theorem hexagon_area {s : ℝ} (h_eq : s = 3) :
  let A := (3 * s) * √3 / 4 in
  A = 9 * √3 := by
  sorry

end hexagon_area_l515_515794


namespace number_of_pieces_of_bubble_gum_l515_515358

theorem number_of_pieces_of_bubble_gum (cost_per_piece total_cost : ℤ) (h1 : cost_per_piece = 18) (h2 : total_cost = 2448) :
  total_cost / cost_per_piece = 136 :=
by
  rw [h1, h2]
  norm_num

end number_of_pieces_of_bubble_gum_l515_515358


namespace game_3_is_unfair_l515_515398

/-- 
Among the following three game rules, each involving drawing balls from a bag without replacement, 
prove that Game 3 is unfair.
Game 1: 
    3 black balls and 1 white ball
    Draw 1 ball, then draw another ball
    If the two balls drawn are the same color, Player A wins
    If the two balls drawn are different colors, Player B wins

Game 2:
    1 black ball and 1 white ball
    Draw 1 ball
    If a black ball is drawn, Player A wins
    If a white ball is drawn, Player B wins
    
Game 3:
    2 black balls and 2 white balls
    Draw 1 ball, then draw another ball
    If the two balls drawn are the same color, Player A wins
    If the two balls drawn are different colors, Player B wins
-/
theorem game_3_is_unfair :
  ∀ (n1 n2 n3 : nat) (nc1 : n1 = 3) (nc2 : n2 = 1) (nc3 : n3 = 2),
  ∀ (d1 d2 d3 : nat) (dc1 : d1 = 1) (dc2 : d2 = 1) (dc3 : d3 = 2),
  ∀ (g1_win_A g1_win_B g3_win_A g3_win_B : Prop)
    (gc1 : g1_win_A ↔ (n1 - d2 + 1) = (n2 - d2 + 1))
    (gc2 : g3_win_A ↔ (n3 - d3 + 1) = (n3 - d3 + 1)) (gc3 : g3_win_B ↔ (n3 - d3 + 1) ≠ (n3 - d3 + 1)),
  g3_win_A ↔ g3_win_B :=
sorry

end game_3_is_unfair_l515_515398


namespace age_difference_l515_515439

theorem age_difference :
  let x := 5
  let prod_today := x * x
  let prod_future := (x + 1) * (x + 1)
  prod_future - prod_today = 11 :=
by
  sorry

end age_difference_l515_515439


namespace max_distance_between_points_l515_515264

noncomputable def max_distance_complex (z : ℂ) (h : |z| = 2) : ℝ :=
  let a := 5 + 2 * complex.I in
  8 * (complex.abs (complex.abs a + h)) -- Redefine |a + h|

theorem max_distance_between_points (z : ℂ) (h : |z| = 2) :
  max_distance_complex z h = 8 * (Real.sqrt 29 + 2) :=
by
  -- Proof goes here
  sorry

end max_distance_between_points_l515_515264


namespace minimum_adjacent_white_pairs_l515_515925

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l515_515925


namespace sum_of_last_two_digits_l515_515728

theorem sum_of_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) : (a^15 + b^15) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l515_515728


namespace f_7_5_l515_515626

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_7_5 : f 7.5 = -0.5 := by
  sorry

end f_7_5_l515_515626


namespace problem_statement_l515_515905

noncomputable def sequence (x₀ : ℝ) : ℕ → ℝ
| 0       := x₀
| (n + 1) := (sequence x₀ n ^ 2 + 2) / (2 * sequence x₀ n)

theorem problem_statement :
  0 < sequence 10^9 36 - real.sqrt 2 ∧ sequence 10^9 36 - real.sqrt 2 < 10^(-9) :=
by
  sorry

end problem_statement_l515_515905


namespace parallel_planes_lines_relation_l515_515209

theorem parallel_planes_lines_relation (P₁ P₂ : Plane) (L₁ L₂ : Line) 
  (h₁ : P₁.parallel P₂) (h₂: L₁ ∈ P₁) (h₃: L₂ ∈ P₂) :
  L₁.parallel L₂ ∨ L₁.skew L₂ := 
sorry

end parallel_planes_lines_relation_l515_515209


namespace suff_not_necessary_x0_l515_515962

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else -2 / x

theorem suff_not_necessary_x0 (x0 : ℝ) (h : x0 = -2) : 
  f x0 = -1 ∧ ¬(∀ x : ℝ, f x = -1 → x = -2) := 
by
  sorry

end suff_not_necessary_x0_l515_515962


namespace mod_inverse_35_37_l515_515884

theorem mod_inverse_35_37 : ∃ a : ℤ, 0 ≤ a ∧ a < 37 ∧ (35 * a) % 37 = 1 :=
by
  use 18
  split
  · norm_num
  · split
  · norm_num
  · norm_num
  sorry

end mod_inverse_35_37_l515_515884


namespace number_of_distinct_numbers_l515_515970

/-- How many distinct four-digit numbers can be created using the digits {1, 2, 3, 4, 5} if no digit is repeated and the number must start with an even digit? -/
theorem number_of_distinct_numbers :
  ∃ n : ℕ, n = 48 ∧ (∀ a b c d : {x // x ∈ {1, 2, 3, 4, 5}}, 
  a.1 ≠ b.1 ∧ a.1 ≠ c.1 ∧ a.1 ≠ d.1 ∧ b.1 ≠ c.1 ∧ b.1 ≠ d.1 ∧ c.1 ≠ d.1 ∧ 
  a.1 ∈ {2, 4} → (a.1, b.1, c.1, d.1) = (a.1, b.1, c.1, d.1) → 
  (a.1 : ℕ) * 4 * 3 * 2 = 48) :=
sorry

end number_of_distinct_numbers_l515_515970


namespace golden_ratio_problem_l515_515593

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_problem (m : ℝ) (x : ℝ) :
  (1000 ≤ m) → (1000 ≤ x) → (x ≤ m) →
  ((m - 1000) / (x - 1000) = phi ∧ (x - 1000) / (m - x) = phi) →
  (m = 2000 ∨ m = 2618) :=
by
  sorry

end golden_ratio_problem_l515_515593


namespace stella_profit_l515_515314

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end stella_profit_l515_515314


namespace area_of_path_l515_515015

-- Define the conditions from the problem
def circumference : ℝ := 314
def path_width : ℝ := 2
def pi_approx : ℝ := 3.14

-- Calculate the radius of the flower bed
def radius_bed : ℝ := circumference / (2 * pi_approx)

-- Calculate the radius of the larger circle including the path
def radius_outer : ℝ := radius_bed + path_width

-- Calculate the area of the path
def area_path : ℝ := pi_approx * (radius_outer^2 - radius_bed^2)

-- Lean statement to prove the area of the path
theorem area_of_path : area_path = 640.56 := by
  sorry

end area_of_path_l515_515015


namespace wolf_wins_with_correct_play_l515_515752

/-- Wolf and Hare are playing a game where they take turns subtracting any of the non-zero digits from the current number on the board.
    The winner is the player to make the number zero. Initially, the number 1234 is written on the board and Wolf goes first.
    This proof verifies that with correct play, Wolf will win. -/
theorem wolf_wins_with_correct_play : ∀ (initial_number : ℕ), initial_number = 1234 ∧ 
  (∀ (n : ℕ), 0 < n → 
   (∃ d : ℕ, d ∈ {d | d ∈ finset.range 10 ∧ d ≠ 0 ∧ d ≤ n}) ∧ 
   (∃ next_number : ℕ, next_number = n - d)) → 
  ∃ (winner: string), winner = "Wolf" := 
begin
  sorry
end


end wolf_wins_with_correct_play_l515_515752


namespace xy_squared_value_l515_515568

theorem xy_squared_value (x y : ℝ) (h1 : x * (x + y) = 22) (h2 : y * (x + y) = 78 - y) :
  (x + y) ^ 2 = 100 :=
  sorry

end xy_squared_value_l515_515568


namespace largest_k_l515_515248

/-- Let \( n \geq 2 \) be an integer. 
  There is a field divided into \( n \times n \) squares. 
  The bottom-left square contains \( k \) Swiss gymnasts. 
  The top-right square contains \( k \) Liechtensteiner gymnasts. 
  Swiss gymnasts move either up or right. 
  Liechtensteiner gymnasts move either down or left. 
  A gymnast cannot enter a house until all gymnasts from the other nationality have left their house.
  Prove the largest \( k \) for which switching the gymnasts between the two houses is possible 
  is \( (n - 1)^2 \).
-/
theorem largest_k (n : ℕ) (hn : n ≥ 2) : 
  {k : ℕ // 
    ∀ (k ≥ 0) (k ≤ (n-1)^2), 
    k = (n-1)^2} :=
by
  sorry

end largest_k_l515_515248


namespace angle_with_same_terminal_side_in_range_l515_515599

theorem angle_with_same_terminal_side_in_range (θ : ℝ) :
  (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (∃ k : ℤ, θ = 2 * k * Real.pi + (-4 * Real.pi / 3)) → θ = 2 * Real.pi / 3 :=
by
  assume h,
  sorry

end angle_with_same_terminal_side_in_range_l515_515599


namespace max_elements_subset_l515_515630

theorem max_elements_subset (M : Set ℕ) (hM : M = {1, 2, ..., 1995}) (A : Set ℕ)
  (hA_sub : A ⊆ M) (hA_cond : ∀ x ∈ A, 15 * x ∉ A) : ∃ n, n = 1870 ∧ #(A) ≤ n :=
by
  sorry

end max_elements_subset_l515_515630


namespace fifth_term_geom_prog_l515_515878

noncomputable theory

def geom_prog (b : ℕ → ℝ) (q : ℝ) :=
∀ n : ℕ, b (n + 1) = (b 0) * q^(n + 1)

theorem fifth_term_geom_prog
  (b : ℕ → ℝ)
  (q : ℝ)
  (b1_condition : b 0 = 7 - 3 * Real.sqrt 5)
  (geom_condition : geom_prog b q)
  (q_condition : |q| > 1) :
  b 4 = 2 :=
sorry

end fifth_term_geom_prog_l515_515878


namespace find_y_l515_515594

-- Definitions
variables (A B C D E F G: Type)
variables (ABCD DEFG: set (set.mem)) 
variables (ADC CDE GDE: ℝ)

-- Conditions
axiom squares_with_equal_sides : (ABCD) = (DEFG)
axiom angle_DCE : (angle DCE) = 70

-- Theorem to Prove
theorem find_y {y : ℝ} :
  (angle ADC = 90) ∧ (angle CDE = 40) ∧ (angle GDE = 90) ∧ (y + 220 = 360) → y = 140 :=
by sorry

end find_y_l515_515594


namespace smallest_feared_sequence_l515_515837

def is_feared (n : ℕ) : Prop :=
  -- This function checks if a number contains '13' as a contiguous substring.
  sorry

def is_fearless (n : ℕ) : Prop := ¬is_feared n

theorem smallest_feared_sequence : ∃ (n : ℕ) (a : ℕ), 0 < n ∧ a < 100 ∧ is_fearless n ∧ is_fearless (n + 10 * a) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → is_feared (n + k * a)) ∧ n = 1287 := 
by
  sorry

end smallest_feared_sequence_l515_515837


namespace presents_cycle_l515_515993

-- Definitions for our condition
def makes_present_to {child : Type} (S : set (set child)) (A B C : child) : Prop :=
  {A, B, C} ∈ S

theorem presents_cycle (child : Type) (S : set (set child)) (n : ℕ) (h_odd : odd n)
    (h_S : ∀ (A B : child), ∃! (C : child), makes_present_to S A B C) :
  ∀ (A B C : child), makes_present_to S A B C → makes_present_to S A C B :=
  sorry

end presents_cycle_l515_515993


namespace area_of_f2_equals_7_l515_515963

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_of_f2_equals_7 : 
  (∫ x in (-3 : ℝ)..3, f2 x) = 7 :=
by
  sorry

end area_of_f2_equals_7_l515_515963


namespace sam_needs_change_l515_515359

/--
Machine has 10 toys ranging in cost from 50 cents to $2.50, priced in 25 cent increments.
Sam has 10 quarters and a twenty-dollar bill, and the machine only accepts quarters.
Prove that the probability Sam needs to get change for the twenty-dollar bill before he can buy his favorite toy, which costs $2.25, is 44/45.
-/
theorem sam_needs_change (toys : Fin 10 → ℕ) (quarters_count : ℕ) (favorite_toy_index : Fin 10) :
  (∀ i, 50 ≤ toys i ∧ toys i ≤ 250 ∧ (toys (i + 1) mod 25 = 0)) ∧
  quarters_count = 10 ∧ toys favorite_toy_index = 225 →
  let orders := (factorial 10) in
  let favorable_orders := (2 * (factorial 8)) in
  1 - (favorable_orders / orders) = 44 / 45 :=
begin
  intro h,
  sorry
end

end sam_needs_change_l515_515359


namespace pentagon_angle_T_l515_515999

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l515_515999


namespace roots_of_cos_poly_l515_515275

theorem roots_of_cos_poly (α β : ℂ) (h1 : 2 * α^2 + 3 * α + 5 = 0) (h2 : 2 * β^2 + 3 * β + 5 = 0)
  (h3 : α + β = -3 / 2) (h4 : α * β = 5 / 2) (h5 : complex.cos α + complex.cos β = -1)
  (h6 : complex.cos α * complex.cos β = 1) :
  ∀ x : ℂ, x^2 + x + 1 = 0 :=
by
  sorry

end roots_of_cos_poly_l515_515275


namespace gandalf_reachability_l515_515351

theorem gandalf_reachability :
  ∀ (k : ℕ), ∃ (s : ℕ → ℕ) (m : ℕ), (s 0 = 1) ∧ (s m = k) ∧ (∀ i < m, s (i + 1) = 2 * s i ∨ s (i + 1) = 3 * s i + 1) := 
by
  sorry

end gandalf_reachability_l515_515351


namespace residual_at_point_9_11_l515_515700

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def regress_line_slope (x_list y_list : List ℝ) : ℝ :=
  let mean_x := mean x_list
  let mean_y := mean y_list
  (mean_y - 40) / mean_x

def predicted_y (b x : ℝ) : ℝ :=
  b * x + 40

def residual (y_pred y_actual : ℝ) : ℝ :=
  y_actual - y_pred

theorem residual_at_point_9_11 :
  let x_values := [9, 9.5, 10, 10.5, 11]
  let y_values := [11, 10, 8, 6, 5]
  let point := (9 : ℝ, 11 : ℝ)
  let b := regress_line_slope x_values y_values
  let y_pred := predicted_y b point.1
  residual y_pred point.2 = -0.2 :=
by
  sorry

end residual_at_point_9_11_l515_515700


namespace find_initial_mean_l515_515688

/-- 
  The mean of 50 observations is M.
  One observation was wrongly taken as 23 but should have been 30.
  The corrected mean is 36.5.
  Prove that the initial mean M was 36.36.
-/
theorem find_initial_mean (M : ℝ) (h : 50 * 36.36 + 7 = 50 * 36.5) : 
  (500 * 36.36 - 7) = 1818 :=
sorry

end find_initial_mean_l515_515688


namespace triangle_angle_B_range_l515_515229

theorem triangle_angle_B_range (A B C : ℝ) (h₁ : A + B + C = π) 
    (h₂ : log (tan A) + log (tan C) = 2 * log (tan B)) : 
    π / 3 ≤ B ∧ B < π / 2 :=
sorry

end triangle_angle_B_range_l515_515229


namespace circle_equation_minimum_distance_mn_l515_515512

-- Define the circle's properties
def circle_passes_through_origin (x y : ℝ) : Prop := x^2 + y^2 = 5
def center_on_ray (a b : ℝ) : Prop := b = 2 * a ∧ a > 0

-- Define the line m
def line_m (x y : ℝ) : Prop := x + 2 * y + 5 = 0

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the problem for the circle's equation
theorem circle_equation :
  (∃ x y : ℝ, circle_passes_through_origin (x - 1) (y - 2)) ∧
  (∃ a b : ℝ, center_on_ray a b ∧ (x - a)^2 + (y - b)^2 = 5) :=
sorry

-- State the problem for the minimum distance |MN|
theorem minimum_distance_mn :
  ∀ M N : ℝ × ℝ, line_m M.1 M.2 → 
  (M.1, M.2) = (-1, -2) → N = (1, 2) → 
  distance M N = real.sqrt 5 :=
sorry

end circle_equation_minimum_distance_mn_l515_515512


namespace least_odd_prime_factor_of_2023_power6_plus_1_l515_515475

theorem least_odd_prime_factor_of_2023_power6_plus_1 :
  ∃ (p : ℕ), prime p ∧ odd p ∧ p ∣ (2023^6 + 1) ∧ p = 13 :=
begin
  sorry
end

end least_odd_prime_factor_of_2023_power6_plus_1_l515_515475


namespace find_divisor_l515_515385

theorem find_divisor (d : ℕ) (h1 : 127 = d * 5 + 2) : d = 25 :=
sorry

end find_divisor_l515_515385


namespace find_p_l515_515337

theorem find_p (a b p : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) 
  (h3: a^2 - 4 * b = 0) 
  (h4: a + b = 5 * p) 
  (h5: a * b = 2 * p^3) : p = 3 := 
sorry

end find_p_l515_515337


namespace product_of_common_divisors_eq_8000_l515_515888

theorem product_of_common_divisors_eq_8000 :
  let divisors_180 := [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180, -1, -2, -3, -4, -5, -6, -9, -10, -12, -15, -18, -20, -30, -36, -45, -60, -90, -180],
      divisors_20 := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20],
      common_divisors := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20] in
    (List.prod common_divisors) = 8000 :=
by
  let divisors_180 := [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180, -1, -2, -3, -4, -5, -6, -9, -10, -12, -15, -18, -20, -30, -36, -45, -60, -90, -180];
  let divisors_20 := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20];
  let common_divisors := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20];
  sorry

end product_of_common_divisors_eq_8000_l515_515888


namespace remainder_of_sum_of_squares_mod_n_l515_515450

theorem remainder_of_sum_of_squares_mod_n (a b n : ℤ) (hn : n > 1) 
  (ha : a * a ≡ 1 [ZMOD n]) (hb : b * b ≡ 1 [ZMOD n]) : 
  (a^2 + b^2) % n = 2 := 
by 
  sorry

end remainder_of_sum_of_squares_mod_n_l515_515450


namespace equilateral_triangle_area_from_hexagon_l515_515798

theorem equilateral_triangle_area_from_hexagon (side_length : ℝ) (h : side_length = 3) :
  (∃ hexagon : ℕ → ℤ×ℤ, 
    (∀ (i : ℕ), 0 ≤ i ∧ i < 6 → dist (hexagon (i)) (hexagon (i + 1)) = side_length) ∧
    dist (hexagon 0) (hexagon 2) = side_length ∧
    dist (hexagon 2) (hexagon 4) = side_length ∧
    ∃ triangle : ℕ → ℤ×ℤ, 
      triangle 0 = hexagon 0 ∧
      triangle 1 = hexagon 2 ∧
      triangle 2 = hexagon 4 ∧
      (dist (triangle 0) (triangle 1) = side_length ∧
      dist (triangle 1) (triangle 2) = side_length ∧
      dist (triangle 2) (triangle 0) = side_length)
  ) → ∃ (area : ℝ), area = (9 * real.sqrt 3) / 4 :=
by
  sorry

end equilateral_triangle_area_from_hexagon_l515_515798


namespace Carrie_pays_94_l515_515844

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l515_515844


namespace option_d_correct_l515_515646

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)
def M : Set ℝ := {x | f x = 0}

theorem option_d_correct : ({1, 3} ∪ {2, 3} : Set ℝ) = M := by
  sorry

end option_d_correct_l515_515646


namespace value_at_7_5_l515_515269

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = -f x
axiom interval_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem value_at_7_5 : f 7.5 = -0.5 := by
  sorry

end value_at_7_5_l515_515269


namespace max_value_of_e_l515_515260

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (5^n - 1) / 4

-- Define e_n as the gcd of b_n and b_(n+1)
def e (n : ℕ) : ℚ := Int.gcd (b n) (b (n + 1))

-- The theorem we need to prove is that e_n is always 1
theorem max_value_of_e (n : ℕ) : e n = 1 :=
  sorry

end max_value_of_e_l515_515260


namespace t_value_real_l515_515950

noncomputable def z1 : ℂ := 3 + 4 * complex.I
noncomputable def z2 (t : ℝ) : ℂ := t + complex.I

theorem t_value_real (t : ℝ) (h : (z1 * complex.conj (z2 t)).im = 0) : t = 3 / 4 :=
by sorry

end t_value_real_l515_515950


namespace incorrect_process_flowchart_judgment_l515_515381

theorem incorrect_process_flowchart_judgment :
    ∀ (A B C D : Prop),
    (A = "Drawing a process flowchart is similar to drawing an algorithm flowchart, where each process needs to be refined step by step, in a top-down or left-to-right order") →
    (B = "Loops can appear in a process flowchart, which is different from an algorithm flowchart") →
    (C = "The flow lines in a process flowchart represent the connection between two adjacent processes") →
    (D = "The flow lines in a process flowchart are directional") →
    (¬ B) :=
begin
    intros A B C D hA hB hC hD,
    -- Proof goes here
    sorry,
end

end incorrect_process_flowchart_judgment_l515_515381


namespace three_digit_number_formed_by_1198th_1200th_digits_l515_515812

def albertSequenceDigit (n : ℕ) : ℕ :=
  -- Define the nth digit in Albert's sequence
  sorry

theorem three_digit_number_formed_by_1198th_1200th_digits :
  let d1198 := albertSequenceDigit 1198
  let d1199 := albertSequenceDigit 1199
  let d1200 := albertSequenceDigit 1200
  (d1198 * 100 + d1199 * 10 + d1200) = 220 :=
by
  sorry

end three_digit_number_formed_by_1198th_1200th_digits_l515_515812


namespace exponent_equality_l515_515556

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l515_515556


namespace minimize_distance_sum_l515_515300

-- Define points A, B, C, D and O
variables {A B C D O : Point}

-- Define point P as any point in Euclidean space
variables {P : Point}

-- Distance function d between two points
def d (X Y : Point) : Real := EuclideanDistance X Y

-- The statement to prove
theorem minimize_distance_sum
  (h1 : IsRegularTetrahedron A B C D) 
  (h2 : IsCircumscribedSphereCenter O A B C D) :
  ∑ X in {A, B, C, D}, d X P >= ∑ X in {A, B, C, D}, d X O :=
by
  sorry

end minimize_distance_sum_l515_515300


namespace values_of_n_l515_515902

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l515_515902


namespace exists_six_equally_spaced_lines_in_space_l515_515234

theorem exists_six_equally_spaced_lines_in_space :
  ∃ (lines : fin 6 → ℝ^3), 
    (∀ i j, i ≠ j → lines i ≠ lines j) ∧ -- lines are non-parallel
    (∃ θ : ℝ, ∀ i j, i ≠ j → angle (lines i) (lines j) = θ) := -- equal angles
sorry

end exists_six_equally_spaced_lines_in_space_l515_515234


namespace sum_of_all_four_digit_numbers_using_1_to_5_l515_515495

open BigOperators

theorem sum_of_all_four_digit_numbers_using_1_to_5 : ∑ n in { n : ℕ | n >= 1000 ∧ n < 10000 ∧ ∀ d ∈ nat.digits 10 n, d ∈ {1, 2, 3, 4, 5} ∧ (λ x, list.count x (nat.digits 10 n)) d = 1 }, n = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_using_1_to_5_l515_515495


namespace number_of_elements_l515_515357

theorem number_of_elements (n : ℕ) (S : ℕ) (h1 : S = n * 60)
  (h2 : ∑ i in (range 6), a i = 342)
  (h3 : ∑ i in (range 8), b i = 488)
  (h4 : a 7 = b 0)
  (h5 : a 7 = 50) : n = 13 :=
  sorry

end number_of_elements_l515_515357


namespace range_of_x_l515_515980

theorem range_of_x (x : ℝ) : 
  ¬(x ∈ set.Icc 2 5 ∨ x ∈ {x | x < 1 ∨ x > 4}) → (x ∈ set.Ico 1 2) := by
sorry

end range_of_x_l515_515980


namespace max_speed_motorboat_in_still_water_l515_515463

theorem max_speed_motorboat_in_still_water :
  ∃ v : ℝ, let flood_speed := 10 in
    (2 / (v + flood_speed) = 1.2 / (v - flood_speed)) → v = 40 :=
by
  sorry

end max_speed_motorboat_in_still_water_l515_515463


namespace largest_stamps_per_page_l515_515242

theorem largest_stamps_per_page (a b k : ℕ) (h1 : a = 960) (h2 : b = 1200) (h3 : k divides a) (h4 : k divides b) (h5 : 2 ≤ a / k) (h6 : 2 ≤ b / k) : k = 240 := 
by
  sorry

end largest_stamps_per_page_l515_515242


namespace condition_1_condition_2_condition_3_condition_4_l515_515458

-- Defining the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := x^2 - m * x + 2 * m - 2

-- Condition 1: There is at least one solution in [0, 3/2]
theorem condition_1 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3/2 ∧ quadratic_eq m x = 0) ↔ m ∈ [-1/2, 4 - 2*sqrt 2] :=
sorry

-- Condition 2: There is at least one solution in (0, 3/2)
theorem condition_2 (m : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 3/2 ∧ quadratic_eq m x = 0) ↔ m ∈ (-1/2, 4 - 2*sqrt 2] :=
sorry

-- Condition 3: There is exactly one solution in (0, 3/2)
theorem condition_3 (m : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x < 3/2 ∧ quadratic_eq m x = 0) ↔ m ∈ (-1/2, 1] :=
sorry

-- Condition 4: There are two solutions in [0, 3/2]
theorem condition_4 (m : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 3/2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3/2 ∧ x₁ ≠ x₂ ∧ quadratic_eq m x₁ = 0 ∧ quadratic_eq m x₂ = 0) ↔ m ∈ [1, 4 - 2*sqrt 2] :=
sorry

end condition_1_condition_2_condition_3_condition_4_l515_515458


namespace area_trapezoid_ABCD_l515_515762

variable {AB CD : ℝ} (ACperpCD : AC ⊥ CD) (DBperpAB : DB ⊥ AB)
variable {S : ℝ} (areaAMND : Area AMND = S)
variable (similarBMNCtoABCD : Similar BMNC ABCD)
variable (sumAngles60 : ∠CAD + ∠BDA = 60°)

theorem area_trapezoid_ABCD : 
  Area ABCD = (4 * S) / 3 :=
by
  sorry

end area_trapezoid_ABCD_l515_515762


namespace lateral_surface_area_of_cylinder_l515_515171

def base_area (S : ℝ) := S
def lateral_surface_unfolds_into_square : Prop := true

theorem lateral_surface_area_of_cylinder (S : ℝ) 
  (h_base_area : base_area S = S)
  (h_square : lateral_surface_unfolds_into_square) : 
  lateral_surface_area (cylinder S) = 4 * π * S := 
sorry

end lateral_surface_area_of_cylinder_l515_515171


namespace number_of_n_with_prime_sum_of_divisors_l515_515268

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (list.range (n + 1)).filter (λ d, n % d = 0).sum

-- The statement to prove
theorem number_of_n_with_prime_sum_of_divisors :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ Nat.Prime (sum_of_divisors n)}.to_finset.card = m := sorry

end number_of_n_with_prime_sum_of_divisors_l515_515268


namespace monic_polynomial_root_equivalence_l515_515628

noncomputable def roots (p : Polynomial ℝ) : List ℝ := sorry

theorem monic_polynomial_root_equivalence :
  let r1 := roots (Polynomial.C (8:ℝ) + Polynomial.X^3 - 3 * Polynomial.X^2)
  let p := Polynomial.C (216:ℝ) + Polynomial.X^3 - 9 * Polynomial.X^2
  r1.map (fun r => 3*r) = roots p :=
by
  sorry

end monic_polynomial_root_equivalence_l515_515628


namespace angle_E_l515_515442

theorem angle_E (A B C D E : Type)
  (triangle_ABC : Triangular_shape A B C)
  (angle_rel : angle_CAB - angle_B = 90)
  (point_D : Extends_point line_BC D)
  (CE_bisects : Bisects_angle CE (angle_ACD))
  (intersection_E : Intersects_line CE line_BA E) :
  angle_E = 45 :=
sorry

end angle_E_l515_515442


namespace maximizing_beam_strength_l515_515505

noncomputable theory

open Real

variables (R : ℝ) (x y : ℝ)
-- Given conditions
def beam_condition (R x y : ℝ) := x^2 + y^2 = 4 * R^2

def beam_strength (k x y : ℝ) := k * x * y^2

-- Requirement to prove: the dimensions that maximize the strength
theorem maximizing_beam_strength (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ (x y : ℝ), beam_condition R x y ∧ x = 2 * R / sqrt(3) ∧ y = 2 * R * sqrt(2) / sqrt(3) := 
sorry

end maximizing_beam_strength_l515_515505


namespace correct_propositions_l515_515182

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 4))
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem correct_propositions : 
  (∃ k : ℤ, k ≠ 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 = -1 → f x2 = -1 → x1 - x2 = k * Real.pi) ∧
  (∀ x : ℝ, f (-Real.pi / 8) = f x → x = -Real.pi / 8) :=
sorry

end correct_propositions_l515_515182


namespace inverse_proportionality_square_l515_515345

theorem inverse_proportionality_square (a b k : ℝ) (h1 : ∀ a b, a^2 * sqrt b = k) (h2 : a = 3) (h3 : b = 36) (h4 : a * b = 54) : b = 18 * (4^(1/3)) :=
by 
  sorry

end inverse_proportionality_square_l515_515345


namespace problem_solution_l515_515897

variables {x y z k : ℝ}

-- Conditions
def matrix_non_invertible (A : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : Prop :=
  let ⟨a, b, c, d, e, f, g, h, i⟩ := A in
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) = 0

def condition1 : Prop := ∃ x y z k : ℝ, (matrix_non_invertible (x, y, z, z, x, y, y, z, x)) ∧ (x + y + z = k) ∧ (y + z ≠ k) ∧ (z + x ≠ k) ∧ (x + y ≠ k)

-- Target expression
def target_expression (x y z k : ℝ) : ℝ :=
  x / (y + z - k) + y / (z + x - k) + z / (x + y - k)

theorem problem_solution :
  condition1 →
  ∀ x y z k : ℝ,
  target_expression x y z k = -3 :=
by
  sorry

end problem_solution_l515_515897


namespace complex_number_real_integral_integral_zero_to_two_of_sqrt_4_minus_x_squared_eq_pi_l515_515142

noncomputable def integral_value (a : ℝ) : ℝ :=
  ∫ x in 0..a, sqrt (4 - x^2)

theorem complex_number_real_integral (a : ℝ) (h_imaginary_part_zero : a - 2 = 0) : integral_value a = π :=
by 
  rw [integral_value]
  rw [h_imaginary_part_zero] at *
  rw [integral_zero_to_two_of_sqrt_4_minus_x_squared_eq_pi]
  sorry

-- Auxiliary theorem to handle the integration part
theorem integral_zero_to_two_of_sqrt_4_minus_x_squared_eq_pi : ∫ x in 0..2, sqrt (4 - x^2) = π :=
by sorry

end complex_number_real_integral_integral_zero_to_two_of_sqrt_4_minus_x_squared_eq_pi_l515_515142


namespace equilateral_triangle_l515_515090

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ) (p R : ℝ)
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = p / (9 * R)) :
  a = b ∧ b = c ∧ a = c :=
sorry

end equilateral_triangle_l515_515090


namespace num_integer_solutions_y_lt_y_sq_l515_515895

theorem num_integer_solutions_y_lt_y_sq (y : ℤ) : 
  (finset.filter (λ y, y^2 < 9 * y) (finset.range 10)).card = 8 := 
by
  sorry

end num_integer_solutions_y_lt_y_sq_l515_515895


namespace values_of_a_b_f_monotonicity_on_0_1_l515_515959

-- Define the function f for a general a and b
def f (a b x : ℝ) := (a * x^2 + 1) / (x + b)

-- Conditions from the problem
variable (a b : ℝ)
variable (haodd : ∀ x, f a b (-x) = -f a b x)
variable (hf1 : f a b 1 = 2)

-- Result of the first question
theorem values_of_a_b : a = 1 ∧ b = 0 :=
by
  sorry

-- Defining the function after finding the values of a and b
def f_specific (x : ℝ) := x + 1 / x

-- Monotonicity condition of the resulting function f(x) on (0,1)
theorem f_monotonicity_on_0_1 : ∀ ⦃x1 x2 : ℝ⦄, 0 < x1 ∧ x1 < 1 → 0 < x2 ∧ x2 < 1 → x1 < x2 → f_specific x1 > f_specific x2 :=
by
  sorry

end values_of_a_b_f_monotonicity_on_0_1_l515_515959


namespace mean_and_variance_of_transformed_data_set_l515_515915

noncomputable theory

def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6
def increase_value : ℝ := 10

theorem mean_and_variance_of_transformed_data_set:
  (initial_mean + increase_value = 12.8) ∧ (initial_variance = 3.6) :=
by
  sorry

end mean_and_variance_of_transformed_data_set_l515_515915


namespace intersection_M_N_l515_515545

noncomputable def M : Set ℝ := {x | x^2 - x ≤ 0}
noncomputable def N : Set ℝ := {x | x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l515_515545


namespace base_7_minus_base_8_to_decimal_l515_515063

theorem base_7_minus_base_8_to_decimal : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) - (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 8190 :=
by sorry

end base_7_minus_base_8_to_decimal_l515_515063


namespace range_of_m_l515_515156

def p (x : ℝ) : Prop := x^2 - 5 * x - 6 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

theorem range_of_m (m : ℝ) (m_pos : m > 0) : 
  (∃ (h : ∀ x, ¬ p x → ¬ q x m), ¬ ∀ x, q x m → p x) ↔ m ∈ Ioo 0 3 := by
  sorry

end range_of_m_l515_515156


namespace min_adj_white_pairs_l515_515922

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l515_515922


namespace minimum_perimeter_ABC_l515_515916

theorem minimum_perimeter_ABC (a b c : ℝ) 
  (ABC_area : 1 / 2 * a * b * sin(acos((a^2 + b^2 - c^2) / (2 * a * b))) = 4)
  (sidelength_c_greater_than_b : c > b)
  (C_prime_reflection : ∀ P : ℝ, C - P = P - C')
  (similar_triangles : ∀ Q O : ℝ, (C' - P) / (B - D) = (A - B) / (C - B)) :
  (a + b + sqrt(a^2 + b^2) = 4 * sqrt(2) + 4) :=
begin
  sorry
end

end minimum_perimeter_ABC_l515_515916


namespace compute_product_bn_l515_515894

theorem compute_product_bn :
  (∏ n in Finset.range (101 - 4) + 4, (n^2 + 5 * n + 9) / (n^3 + 3 * n^2 + 3 * n)) = 41616 / (Nat.factorial 100) :=
by
  sorry

end compute_product_bn_l515_515894


namespace divisible_by_odd_number_of_primes_l515_515249

theorem divisible_by_odd_number_of_primes (k : ℕ) (p : ℕ → ℕ) (N : ℕ) (i : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ k → Nat.prime (p i))
  (h2 : ∀ j, j ∈ Finset.range(k) → p j = Nat.prime_seq (j+1))
  (h3 : N = ∏ i in Finset.range(k + 1) | (p i)): 
  ∃ s : Finset ℕ, s.card = N / 2 ∧ ∀ n ∈ s, odd (Finset.card (Finset.filter (λ j, (p j) ∣ n) (Finset.range(k + 1)))) :=
sorry

end divisible_by_odd_number_of_primes_l515_515249


namespace parabola_distance_to_focus_l515_515573

theorem parabola_distance_to_focus :
  ∀ (P : ℝ × ℝ), P.1 = 2 ∧ P.2^2 = 4 * P.1 → dist P (1, 0) = 3 :=
by
  intro P h
  have h₁ : P.1 = 2 := h.1
  have h₂ : P.2^2 = 4 * P.1 := h.2
  sorry

end parabola_distance_to_focus_l515_515573


namespace parabola_intersect_line_l515_515188

open Real

variables {p x1 y1 x2 y2 : ℝ} (h : 0 < p)
variables (A B : ℝ × ℝ) (λ : ℝ)

def is_on_parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

def lineAB (x y : ℝ) : Prop := y = (4 / 3) * (x - (p / 2))

def af_fb_relation (λ : ℝ) : Prop := λ > 1

theorem parabola_intersect_line (h₁ : is_on_parabola x1 y1)
                                 (h₂ : is_on_parabola x2 y2)
                                 (h₃ : lineAB A.1 A.2)
                                 (h₄ : lineAB B.1 B.2)
                                 (h₅ : af_fb_relation λ)
                                 (h₆ : A.1 = x1) (h₇ : A.2 = y1)
                                 (h₈ : B.1 = x2) (h₉ : B.2 = y2)
                                 (h₀ : (p / 2 - x1, -y1) = λ • (x2 - p / 2, y2)) : 
                                  λ = 4 := 
by sorry

end parabola_intersect_line_l515_515188


namespace range_of_x_l515_515175

theorem range_of_x (x a1 a2 y : ℝ) (d r : ℝ) (hx : x ≠ 0) 
  (h_arith : a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d)
  (h_geom : b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3) : 4 ≤ x :=
by
  -- Assume x ≠ 0 as given and the sequences are arithmetic and geometric
  have hx3d := h_arith.2.2
  have hx3r := h_geom.2.2
  -- Substituting y in both sequences
  simp only [hx3d, hx3r] at *
  -- Solving for d and determining constraints
  sorry

end range_of_x_l515_515175


namespace petya_coloring_failure_7_petya_coloring_failure_10_l515_515139

theorem petya_coloring_failure_7 :
  ¬ ∀ (points : Fin 200 → Fin 7) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

theorem petya_coloring_failure_10 :
  ¬ ∀ (points : Fin 200 → Fin 10) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

end petya_coloring_failure_7_petya_coloring_failure_10_l515_515139


namespace triangular_faces_area_of_pyramid_l515_515735

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l515_515735


namespace sequence_a_correct_l515_515544

open Nat -- Opening the natural numbers namespace

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => (1 / 2 : ℝ) * a n

theorem sequence_a_correct : 
  (∀ n, 0 < a n) ∧ 
  a 1 = 1 ∧ 
  (∀ n, a (n + 1) = a n / 2) ∧
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧
  ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_a_correct_l515_515544


namespace result_l515_515861

-- Defining the basic problem and conditions
variable (prob : ℚ) -- The probability we want to prove is 1/28

-- Conditions
def fair_coin : Prop := ∀ (x : ℕ), (x = 0) ∨ (x = 1)
def after_HT_TTH : Prop := ∀ (x y z : ℕ)
  (sequence : List ℕ), sequence = [x, y, z] 
  ∧ x = 1 ∧ y = 0 -- Representing "HT"
  ∧ z = 1 -- Representing the next flip can be either head or tail.

-- Question to prove
def probability_TTH : Prop :=
  fair_coin →
  after_HT_TTH →
  prob = 1 / 28

-- Final statement
theorem result : probability_TTH :=
  by
    sorry

end result_l515_515861


namespace expression_sum_of_coeffs_l515_515873

theorem expression_sum_of_coeffs (d : ℝ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d)) in
  Polynomial.sum (Polynomial.Coeff expr) = -30 := by
  sorry

end expression_sum_of_coeffs_l515_515873


namespace find_maximum_point_l515_515179

noncomputable def f (x : ℝ) : ℝ := (x * (1 - x)) / (x^3 - x + 1)

theorem find_maximum_point :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧
    (∀ x : ℝ, 0 < x ∧ x < x₀ → f x < f x₀) ∧
    (∀ x : ℝ, x₀ < x ∧ x < 1 → f x < f x₀) ∧
    x₀ = (√2 + 1 - sqrt (2*sqrt 2 - 1)) / 2 :=
by
  sorry

end find_maximum_point_l515_515179


namespace expression_evaluation_l515_515470

noncomputable def eval_expression : ℝ :=
  let log := Real.logb 3 in
  let expr := 2 * log (1/2) + log 12 - (0.7)^0 + (0.25)^(-1) in
  expr

theorem expression_evaluation : eval_expression = 4 :=
by
  sorry

end expression_evaluation_l515_515470


namespace greatest_divisor_of_3815_and_4521_l515_515879

theorem greatest_divisor_of_3815_and_4521 :
  let n := Int.gcd 3784 4488 in
  n = 64 :=
by
  let n := Int.gcd 3784 4488
  sorry

end greatest_divisor_of_3815_and_4521_l515_515879


namespace angle_DBQ_eq_angle_PAC_l515_515507

open_locale real

variables {P A B C D Q : Type*} [metric_space P]

-- Assume that we have points P, A, B, C, D, Q in some metric space ℝ^2
-- with specific properties as given in the problem statement:
-- P is a point outside the circle
def is_tangent (P A : P): Prop := sorry
def is_secant (P C D : P): Prop := sorry
def is_point_on_chord (Q C D : P): Prop := sorry
def angle_DAQ_eq_angle_PBC (DAQ PBC : ℝ): Prop := DAQ = PBC

-- Given the assumptions defined above
axiom P_outside_circle : ∀ {P : P} (C : P), ∃ (tangent PA PB : P),
is_tangent P A ∧ is_tangent P B

-- Secant intersects circle 
axiom secant_intersects : ∀ {P : P} (secant_line : P) (C D : P), is_secant P C D

-- Point Q on chord with angle condition
axiom Q_on_chord_with_angle : ∀ {P Q : P} (C : P) (D : ℝ), 
is_point_on_chord Q C D → angle_DAQ_eq_angle_PBC D (angle P B C)

-- Statement to prove
theorem angle_DBQ_eq_angle_PAC :
  ∀ {P A B C D Q : P} (D : ℝ),
    is_tangent P A →
    is_tangent P B →
    is_secant P C D →
    (∀ {DAQ : ℝ}, is_point_on_chord Q C D → angle_DAQ_eq_angle_PBC D (angle P B C)) →
    angle D B Q = angle P A C :=
begin
  -- Proof is here but we're omitting it as per the instructions
  sorry
end

end angle_DBQ_eq_angle_PAC_l515_515507


namespace unique_card_assignment_l515_515292

def is_divisible_or_divides (a b : ℕ) : Prop :=
  a ∣ b ∨ b ∣ a

def written_on_each_card (cards : List (ℕ × ℕ)) : Prop :=
  cards.length = 5 ∧
  ∀ (card : ℕ × ℕ), card ∈ cards → is_divisible_or_divides card.1 card.2 ∧ card.1 ≠ card.2

def sum_of_pairs (cards : List (ℕ × ℕ)) : Prop :=
  (List.sum (cards.map Prod.fst) + List.sum (cards.map Prod.snd)) = 55

noncomputable def is_unique_solution (cards : List (ℕ × ℕ)) : Prop :=
  cards = [(7, 1), (5, 10), (9, 3), (6, 2), (8, 4)] ∨
  cards = [(1, 7), (10, 5), (3, 9), (2, 6), (4, 8)] -- account for commutativity

theorem unique_card_assignment :
  ∃ (cards : List (ℕ × ℕ)),
    written_on_each_card(cards) ∧
    sum_of_pairs(cards) ∧
    is_unique_solution(cards) :=
sorry

end unique_card_assignment_l515_515292


namespace rectangle_length_is_16_l515_515384

-- Define the conditions
def side_length_square : ℕ := 8
def width_rectangle : ℕ := 4
def area_square : ℕ := side_length_square ^ 2  -- Area of the square
def area_rectangle (length : ℕ) : ℕ := width_rectangle * length  -- Area of the rectangle

-- Lean 4 statement
theorem rectangle_length_is_16 (L : ℕ) (h : area_square = area_rectangle L) : L = 16 :=
by
  /- Proof will be inserted here -/
  sorry

end rectangle_length_is_16_l515_515384


namespace isosceles_right_triangle_angle_l515_515591

-- Define the conditions given in the problem
def is_isosceles (a b c : ℝ) : Prop := 
(a = b ∨ b = c ∨ c = a)

def is_right_triangle (a b c : ℝ) : Prop := 
(a = 90 ∨ b = 90 ∨ c = 90)

def angles_sum_to_180 (a b c : ℝ) : Prop :=
a + b + c = 180

-- The Proof Problem
theorem isosceles_right_triangle_angle :
  ∀ (a b c x : ℝ), (is_isosceles a b c) → (is_right_triangle a b c) → (angles_sum_to_180 a b c) → (x = a ∨ x = b ∨ x = c) → x = 45 :=
by
  intros a b c x h_isosceles h_right h_sum h_x
  -- Proof is omitted with sorry
  sorry

end isosceles_right_triangle_angle_l515_515591


namespace election_results_l515_515595

theorem election_results :
  let total_members := 1600
  let votes_cast := 900
  let candidate_A_votes := 0.45 * votes_cast
  let candidate_B_votes := 0.35 * votes_cast
  let candidate_C_votes := 0.15 * votes_cast
  let candidate_D_votes := 0.05 * votes_cast
  let percent_A := (candidate_A_votes / total_members) * 100
  let percent_B := (candidate_B_votes / total_members) * 100
  let percent_C := (candidate_C_votes / total_members) * 100
  let percent_D := (candidate_D_votes / total_members) * 100
  let abstained_members := total_members - votes_cast
  let percent_abstained := (abstained_members / total_members) * 100
in
percent_A = 25.3125 ∧
percent_B = 19.6875 ∧
percent_C = 8.4375 ∧
percent_D = 2.8125 ∧
percent_abstained = 43.75 :=
by
  sorry

end election_results_l515_515595


namespace final_result_l515_515232

-- Define the sides of the triangle
def DE : ℝ := 20
def EF : ℝ := 21
def FD : ℝ := 29

-- Define the area of the triangle DEF using Heron's formula
noncomputable def semi_perimeter : ℝ := (DE + EF + FD) / 2
noncomputable def area : ℝ := real.sqrt (semi_perimeter * (semi_perimeter - DE) * (semi_perimeter - EF) * (semi_perimeter - FD))

-- Define the heights from vertices D, E, and F
noncomputable def h_d : ℝ := (2 * area) / FD
noncomputable def h_e : ℝ := (2 * area) / DE
noncomputable def h_f : ℝ := (2 * area) / EF

-- Define the maximum height k
noncomputable def k : ℝ := (h_f * h_d) / (h_f + h_d)

-- Define the values of x, y, z
def x : ℕ := 7
def y : ℕ := 210
def z : ℕ := 5

-- Finally, the proof statement that x + y + z = 222
theorem final_result : x + y + z = 222 := by
  -- We insert the proof here later.
  sorry

end final_result_l515_515232


namespace intersection_distance_l515_515455

-- Definitions for the curves
def curve1 (x y : ℝ) : Prop := x = y^4
def curve2 (x y : ℝ) : Prop := x + y^3 = 1

-- Proof statement for the distance between intersections
theorem intersection_distance : 
  ∃ (p q r : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
    curve1 x1 y1 ∧ curve2 x1 y1 ∧
    curve1 x2 y2 ∧ curve2 x2 y2 ∧ 
    (x1, y1) ≠ (x2, y2) →
    (distance (x1, y1) (x2, y2) = Real.sqrt (p + q * Real.sqrt r)) :=
sorry

end intersection_distance_l515_515455


namespace exponent_equality_l515_515558

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l515_515558


namespace min_expression_l515_515158

-- Define the expression whose minimum value we are interested in
def expression (a b: ℝ) : ℝ :=
  6 * real.sqrt (a * b) + 3 / a + 3 / b

-- Prove that the minimum value of the expression is 12 for a > 0 and b > 0
theorem min_expression (a b: ℝ) (h1: a > 0) (h2: b > 0) :
  expression a b ≥ 12 := 
sorry

end min_expression_l515_515158


namespace f2_plus_f_minus2_eq_zero_f_expression_solve_inequality_l515_515162

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_def (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x - 1 else -a^(-x) + 1

theorem f2_plus_f_minus2_eq_zero (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f_def a in
  is_odd_function f →
  f 2 + f (-2) = 0 :=
by
  sorry

theorem f_expression (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f_def a in
  f = (λ x, if x ≥ 0 then a^x - 1 else -a^(-x) + 1) :=
by
  sorry

theorem solve_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f_def a in
  (f x < 4) → (
    if a > 1 then x ∈ set.Iio (Real.log 5 / Real.log a)
    else x ∈ set.Iio 0) :=
by
  sorry

end f2_plus_f_minus2_eq_zero_f_expression_solve_inequality_l515_515162


namespace fourth_root_207360000_l515_515080

theorem fourth_root_207360000 :
  sqrt (207360000 ^ (1/4)) = 120 :=
by
  let a := 6^(4 : ℤ)
  let b := 20^(4 : ℤ)
  have h1 : 1296 = a := sorry
  have h2 : 160000 = b := sorry
  have h3 : 207360000 = 1296 * 160000 := sorry
  rw [h3, h1, h2]
  have h4 : (a * b) = (6 * 20)^( 4 : ℤ) := by {
    rw [← mul_pow],
    congr,
  }
  rw [h4, sqrt_pow (by norm_num : (1/4) / (1/4) = 1)]
  norm_num
  done

end fourth_root_207360000_l515_515080


namespace total_area_of_pyramid_faces_l515_515738

theorem total_area_of_pyramid_faces (b l : ℕ) (hb : b = 8) (hl : l = 10) : 
  let h : ℝ := Math.sqrt (l^2 - (b / 2)^2) in
  let A : ℝ := 1 / 2 * b * h in
  let T : ℝ := 4 * A in
  T = 32 * Math.sqrt 21 := by
  -- Definitions
  have b_val : (b : ℝ) = 8 := by exact_mod_cast hb
  have l_val : (l : ℝ) = 10 := by exact_mod_cast hl

  -- Calculations
  have h_val : h = Math.sqrt (l^2 - (b / 2)^2) := rfl
  have h_simplified : h = 2 * Math.sqrt 21 := by
    rw [h_val, l_val, b_val]
    norm_num
    simp

  have A_val : A = 1 / 2 * b * h := rfl
  simp_rw [A_val, h_simplified, b_val]
  norm_num

  have T_val : T = 4 * A := rfl
  simp_rw [T_val]
  norm_num

  -- Final proof
  sorry

end total_area_of_pyramid_faces_l515_515738


namespace original_sales_tax_percentage_l515_515343

-- Define the known quantities
def market_price : ℝ := 10800
def new_sales_tax_rate : ℝ := 10/3 / 100  -- 3 1/3% or 10/3%
def savings : ℝ := 18
def new_sales_tax_amount : ℝ := market_price * new_sales_tax_rate

-- Define the relationship expressing the original sales tax amount
def original_sales_tax_amount : ℝ := new_sales_tax_amount + savings

-- Define the equation relating the original sales tax amount to
-- the original sales tax percentage
def original_tax_rate (x : ℝ) : Prop :=
  original_sales_tax_amount = market_price * (x / 100)

theorem original_sales_tax_percentage : ∃ x, original_tax_rate x ∧ x = 3.5 :=
  by
    -- Here the proof would go, but we skip it with "sorry"
    sorry

end original_sales_tax_percentage_l515_515343


namespace graph_does_not_pass_second_quadrant_l515_515135

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h₀ : 1 < a) (h₁ : b < -1) : 
∀ x : ℝ, ¬ (y = a^x + b ∧ y > 0 ∧ x < 0) :=
by
  sorry

end graph_does_not_pass_second_quadrant_l515_515135


namespace parabola_focus_l515_515532

theorem parabola_focus (p : ℝ) (h : 4 = 2 * p * 1^2) : (0, 1 / (4 * 2 * p)) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l515_515532


namespace problem_solution_l515_515764

variables (A B C O G: Type)
variables [euclidean_geometry A B C O G]

-- Conditions
-- Triangle ABC is acute and scalene
axiom triangle_ABC_acute_scalene : acute_triangle A B C ∧ scalene_triangle A B C

-- O is the circumcenter
axiom O_is_circumcenter : circumcenter O A B C

-- G is the centroid
axiom G_is_centroid : centroid G A B C

-- AGO forms a right triangle
axiom AGO_right_triangle : right_triangle A G O

-- AO = 9
axiom AO_equals_9 : dist A O = 9

-- BC = 15
axiom BC_equals_15 : dist B C = 15

-- Proof that S^2 = 288
theorem problem_solution : ∃ S, (S = sum_of_possible_areas_of_triangle_AGO) ∧ S^2 = 288 :=
by sorry

end problem_solution_l515_515764


namespace carrie_pays_l515_515846

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l515_515846


namespace square_root_of_sum_l515_515138

def a := Real.sqrt 5 + 2
def b := Real.sqrt 5 - 2

theorem square_root_of_sum (a := Real.sqrt 5 + 2) (b := Real.sqrt 5 - 2) : Real.sqrt (a^2 + b^2 + 7) = 5 := by
  sorry

end square_root_of_sum_l515_515138


namespace frog_position_after_20_jumps_l515_515829

theorem frog_position_after_20_jumps :
  let position_after_n_jumps : ℕ → ℕ
      | 0     := 1
      | (n+1) := (position_after_n_jumps n + (n + 1)) % 3
  in position_after_n_jumps 20 = 1 :=
by {
  let position_after_n_jumps : ℕ → ℕ
      | 0     := 1
      | (n+1) := (position_after_n_jumps n + (n + 1)) % 3,
  sorry
}

end frog_position_after_20_jumps_l515_515829


namespace molecular_weight_correct_l515_515720

-- Define the atomic weights of the elements.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element in the compound.
def number_of_C : ℕ := 7
def number_of_H : ℕ := 6
def number_of_O : ℕ := 2

-- Define the molecular weight calculation.
def molecular_weight : ℝ := 
  (number_of_C * atomic_weight_C) +
  (number_of_H * atomic_weight_H) +
  (number_of_O * atomic_weight_O)

-- Step to prove that molecular weight is equal to 122.118 g/mol.
theorem molecular_weight_correct : molecular_weight = 122.118 := by
  sorry

end molecular_weight_correct_l515_515720


namespace trees_in_garden_l515_515213

theorem trees_in_garden (yard_length : ℕ) (distance_between_trees : ℕ) (H1 : yard_length = 400) (H2 : distance_between_trees = 16) : 
  (yard_length / distance_between_trees) + 1 = 26 :=
by
  -- Adding sorry to skip the proof
  sorry

end trees_in_garden_l515_515213


namespace op_exp_eq_l515_515187

-- Define the operation * on natural numbers
def op (a b : ℕ) : ℕ := a ^ b

-- The theorem to be proven
theorem op_exp_eq (a b n : ℕ) : (op a b)^n = op a (b^n) := by
  sorry

end op_exp_eq_l515_515187


namespace min_guests_at_banquet_l515_515759

theorem min_guests_at_banquet (total_food : ℕ) (max_food_per_guest : ℕ) : 
  total_food = 323 ∧ max_food_per_guest = 2 → 
  (∀ guests : ℕ, guests * max_food_per_guest >= total_food) → 
  (∃ g : ℕ, g = 162) :=
by
  -- Assuming total food and max food per guest
  intro h_cons
  -- Mathematical proof steps would go here, skipping with sorry
  sorry

end min_guests_at_banquet_l515_515759


namespace triangular_faces_area_of_pyramid_l515_515736

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l515_515736


namespace product_of_numerator_and_denominator_l515_515721

theorem product_of_numerator_and_denominator (x : ℚ) (h : x = 0.07)
  (hl : (7 : ℤ).gcd(99 : ℤ) = 1) : 
  (7 * 99 = 693) := by
  sorry

end product_of_numerator_and_denominator_l515_515721


namespace factorize_expression_l515_515104

variable (x : ℝ)

theorem factorize_expression : x^2 + x = x * (x + 1) :=
by
  sorry

end factorize_expression_l515_515104


namespace values_of_a_l515_515618

theorem values_of_a (a : ℝ) : 
  let A := {2, 1 - a, a^2 - a + 2} 
  in 4 ∈ A ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end values_of_a_l515_515618


namespace arithmetic_sequence_general_term_l515_515930

variable (a : ℕ → ℝ)

-- Conditions as given in the problem
axiom a1_neg : a 1 < 0
axiom a100_geq_74 : a 100 ≥ 74
axiom a200_lt_200 : a 200 < 200
axiom nums_in_interval : (∑ n in (1..100).filter (λ n, 1/2 < a n ∧ a n < 8), 1) + 2 = 
                         (∑ n in (1..200).filter (λ n, 14 ≤ a n ∧ a n ≤ 43/2), 1)

-- The general term of the sequence
def general_term (n : ℕ) := (3/4 : ℝ) * n - 1

-- Problem statement to prove
theorem arithmetic_sequence_general_term :
  (∀ n : ℕ, a n = general_term n) :=
sorry

end arithmetic_sequence_general_term_l515_515930


namespace find_n_that_satisfies_conditions_l515_515718

theorem find_n_that_satisfies_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-207 ≡ n [MOD 23]) ∧ n = 0 :=
by
  sorry

end find_n_that_satisfies_conditions_l515_515718


namespace snacks_displayed_at_dawn_l515_515830

variable (S : ℝ)
variable (SoldMorning : ℝ)
variable (SoldAfternoon : ℝ)

axiom cond1 : SoldMorning = (3 / 5) * S
axiom cond2 : SoldAfternoon = 180
axiom cond3 : SoldMorning = SoldAfternoon

theorem snacks_displayed_at_dawn : S = 300 :=
by
  sorry

end snacks_displayed_at_dawn_l515_515830


namespace remove_one_to_achieve_average_l515_515745

theorem remove_one_to_achieve_average :
  let S := (Finset.range 16).sum - 1 in
  S / 14 = 8.5 :=
by 
  have sum_15 : (Finset.range 16).sum = 120 := by 
    calc
      (Finset.range 16).sum = (15 * 16) / 2 := by exact Finset.sum_range_id (n := 15)
      _ = 120 := by norm_num

  let S := 120 - (1 : ℕ)
  calc
    S / 14 = 119 / 14 := by simp
    _ = 8.5 := by norm_num

end remove_one_to_achieve_average_l515_515745


namespace vans_have_8_people_l515_515017

theorem vans_have_8_people (v : ℕ) :
  let vans := 9 in
  let buses := 10 in
  let bus_capacity := 27 in
  let total_people := 342 in
  (total_people - (buses * bus_capacity)) / vans = v →
  v = 8 :=
by
  sorry

end vans_have_8_people_l515_515017


namespace inradius_of_right_triangle_l515_515297

-- Define the legs and hypotenuse of the right triangle
variables {a b c r : Real}

-- Define the conditions: a and b are legs, c is the hypotenuse
def is_right_triangle (a b c : Real) : Prop :=
  c ^ 2 = a ^ 2 + b ^ 2

-- Define the radius formula for the inscribed circle
def inradius (a b c : Real) : Real :=
  (a + b - c) / 2

-- The theorem we need to prove under the given conditions
theorem inradius_of_right_triangle
  (a b c : Real)
  (h : is_right_triangle a b c) :
  inradius a b c = (a + b - c) / 2 :=
sorry

end inradius_of_right_triangle_l515_515297


namespace min_value_frac_sum_geq_4_min_value_frac_sum_eq_4_l515_515159

noncomputable def min_value_frac_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  if h : (a = 1 / 2) ∧ (b = 1 / 2) then 4
  else Inf (set_of (λ z, z >= (1/a + 1/b)))

theorem min_value_frac_sum_geq_4 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  4 ≤ (1 / a + 1 / b) :=
sorry

theorem min_value_frac_sum_eq_4 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 1 / b) = 4 ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end min_value_frac_sum_geq_4_min_value_frac_sum_eq_4_l515_515159


namespace product_of_areas_eq_square_of_volume_l515_515912

theorem product_of_areas_eq_square_of_volume 
(x y z d : ℝ) 
(h1 : d^2 = x^2 + y^2 + z^2) :
  (x * y) * (y * z) * (z * x) = (x * y * z) ^ 2 :=
by sorry

end product_of_areas_eq_square_of_volume_l515_515912


namespace remainder_2021st_term_div_7_l515_515858

-- Define the sequence based on the given conditions
def seq_term_position (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem remainder_2021st_term_div_7 :
  (∃ n : ℕ, seq_term_position n ≥ 2021 ∧ seq_term_position (n - 1) < 2021) →
  let n := some (nat.find_spec ‹∃ n : ℕ, seq_term_position n ≥ 2021 ∧ seq_term_position (n - 1) < 2021›) in
  n % 7 = 1 :=
by
  sorry

end remainder_2021st_term_div_7_l515_515858


namespace two_digit_integer_plus_LCM_of_3_4_5_l515_515746

theorem two_digit_integer_plus_LCM_of_3_4_5 (x : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : ∃ k, x = 60 * k + 2) :
  x = 62 :=
by {
  sorry
}

end two_digit_integer_plus_LCM_of_3_4_5_l515_515746


namespace hexagon_area_l515_515795

theorem hexagon_area {s : ℝ} (h_eq : s = 3) :
  let A := (3 * s) * √3 / 4 in
  A = 9 * √3 := by
  sorry

end hexagon_area_l515_515795


namespace magnitude_a_plus_2b_l515_515168

variables (a b : EuclideanSpace ℝ (fin 2))
variable [normed_space ℝ (EuclideanSpace ℝ (fin 2))]

-- Define the conditions
def angle_between_a_b := real.angle a b = 2 * real.pi / 3
def norm_a := ∥a∥ = 3
def norm_b := ∥b∥ = 4

-- The problem statement
theorem magnitude_a_plus_2b :
  angle_between_a_b → 
  norm_a → 
  norm_b → 
  ∥a + 2 • b∥ = 7 :=
by
  intros
  sorry

end magnitude_a_plus_2b_l515_515168


namespace probability_five_digit_divisible_by_8_l515_515486

open set real

-- Definitions to capture the conditions and problem statement
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000
def has_unique_digits (n : ℕ) (s : finset ℕ) : Prop := (n.digits 10).to_finset = s
def composed_of_digits (n : ℕ) (s : finset ℕ) : Prop := ∀ d ∈ n.digits 10, d ∈ s
def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0
def probability_of_event (total favorable : ℕ) : ℝ := (favorable : ℝ) / (total : ℝ)

-- The finite set of digits
def digits_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Mathematical problem to the proof
theorem probability_five_digit_divisible_by_8 :
  probability_of_event
    (finset.card (finset.permutes (finset.range 8) 5).attach)
    (finset.card ((finset.range 100000).filter (λ n,
                 is_five_digit n ∧
                 has_unique_digits n digits_set ∧
                 composed_of_digits n digits_set ∧
                 divisible_by_8 n))) = 1 / 8 := 
sorry

end probability_five_digit_divisible_by_8_l515_515486


namespace values_of_n_l515_515901

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l515_515901


namespace problem_statement_l515_515642

variable (a b c : ℝ)

theorem problem_statement (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) ≥ 6 :=
sorry

end problem_statement_l515_515642


namespace goldbach_counterexample_l515_515550

theorem goldbach_counterexample (even_int : ℕ) (h_even : even_int > 2) (h_even_prime_sum : ¬ ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ even_int = p + q) : 
  ∃ (even_int : ℕ), even_int > 2 ∧ ¬ ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ even_int = p + q :=
by
  existsi even_int
  split
  { assumption }
  { assumption }

end goldbach_counterexample_l515_515550


namespace train_speed_approximation_l515_515407

noncomputable def speed_of_train (length_train length_tunnel time_min: ℕ) : ℚ :=
  let total_distance := length_train + length_tunnel
  let time_seconds := time_min * 60
  let speed_m_per_s := (total_distance : ℚ) / (time_seconds : ℚ)
  let speed_km_per_hr := speed_m_per_s * 3.6
  speed_km_per_hr

theorem train_speed_approximation :
  speed_of_train 800 500 1 ≈ 78.01 := sorry

end train_speed_approximation_l515_515407


namespace same_radicand_as_sqrt_3_l515_515051

-- Define similar quadratic radicals
def similar_quadratic_radical (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a ≠ 0 ∧ a * sqrt x = sqrt y

-- Given radicals to check
def sqrt_12 := sqrt 12
def sqrt_3 := sqrt 3
def sqrt_03 := sqrt 0.3
def sqrt_23 := sqrt (2 / 3)
def sqrt_18 := sqrt 18

-- The proof statement
theorem same_radicand_as_sqrt_3 : similar_quadratic_radical 12 3 :=
by {
  use 2,
  split,
  { norm_num },
  { rw sqrt_12, norm_num, exact sqrt_3 },
  sorry
}

end same_radicand_as_sqrt_3_l515_515051


namespace Carrie_pays_94_l515_515842

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l515_515842


namespace no_transition_possible_l515_515388

theorem no_transition_possible (A B : Matrix) (hA : ∃ j, ∀ i, A[i, j] = shaded ∧ ∀ j' ≠ j, ∃ i', A[i', j'] ≠ shaded)
  (hB : ∀ j, ∃ i, B[i, j] ≠ shaded) : ¬ exists (fRow fCol: Permutation), fRow • fCol • A = B := 
by
  sorry

end no_transition_possible_l515_515388


namespace normal_distribution_probability_example_l515_515530

noncomputable def normalDist (μ σ : ℝ) : probability_theory.ProbabilityMeasure ℝ :=
sorry    -- Normal distribution measure, implementation omitted for brevity

theorem normal_distribution_probability_example :
  let X := @measure_theory.Measure.measure_space.measure
    in
    (X (λ x, 4 - 2 * 1 < x ∧ x ≤ 4 + 2 * 1) = 0.9544)
    ∧ (X (λ x, 4 - 1 < x ∧ x ≤ 4 + 1) = 0.6826)
    ∧ (X (λ x, 5 < x ∧ x < 6) = 0.1359)
:= by
  let μ := 4
  let σ := 1
  let X := normalDist μ σ
  -- Bringing in the conditions
  have h1 : X (λ x, μ - 2 * σ < x ∧ x ≤ μ + 2 * σ) = 0.9544 := sorry,
  have h2 : X (λ x, μ - σ < x ∧ x ≤ μ + σ) = 0.6826 := sorry,
  -- Conclusion based on given conditions
  exact And.intro h1 (And.intro h2 sorry)

end normal_distribution_probability_example_l515_515530


namespace tina_sells_more_than_katya_l515_515245

noncomputable def katya_rev : ℝ := 8 * 1.5
noncomputable def ricky_rev : ℝ := 9 * 2.0
noncomputable def combined_rev : ℝ := katya_rev + ricky_rev
noncomputable def tina_target : ℝ := 2 * combined_rev
noncomputable def tina_glasses : ℝ := tina_target / 3.0
noncomputable def difference_glasses : ℝ := tina_glasses - 8

theorem tina_sells_more_than_katya :
  difference_glasses = 12 := by
  sorry

end tina_sells_more_than_katya_l515_515245


namespace total_trolls_l515_515101

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l515_515101


namespace number_of_mappings_correct_l515_515940

open Function

noncomputable def number_of_mappings : Nat :=
  let M := {a, b, c}
  let N := {-3, -2, -1, 0, 1, 2, 3}
  let f := M → N
  let condition (f : M → N) : Prop := f a + f b + f c = 0
  let mappings := {g : M → N // condition g}
  Fintype.card mappings

theorem number_of_mappings_correct : number_of_mappings = 37 := 
  by 
  sorry

end number_of_mappings_correct_l515_515940


namespace monotonically_decreasing_a_l515_515525

noncomputable def f (a x : ℝ) := (x^2 - 2*a*x) * Real.exp x

theorem monotonically_decreasing_a (a : ℝ) (x : ℝ) (h₁ : a ≥ 0) (h₂ : x ∈ Set.Icc (-1 : ℝ) (1 : ℝ))
  (h₃ : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), Deriv f a x ≤ 0) : a ≥ 3/4 :=
sorry

end monotonically_decreasing_a_l515_515525


namespace smallest_positive_period_max_min_values_range_of_m_l515_515533

def f (x : ℝ) : ℝ := 2 * (sin (π / 4 + x))^2 - sqrt 3 * cos (2 * x)

theorem smallest_positive_period (T : ℝ) : T = π → (∀ x, f (x + T) = f x) :=
sorry

theorem max_min_values : 
(∀ x, x ∈ Icc (π / 4) (π / 2) → 2 ≤ f x ∧ f x ≤ 3) :=
sorry

theorem range_of_m (m : ℝ) : (∀ x, x ∈ Icc (π / 4) (π / 2) → abs (f x - m) < 2)
→ 1 < m ∧ m < 4 :=
sorry

end smallest_positive_period_max_min_values_range_of_m_l515_515533


namespace work_together_days_l515_515776

theorem work_together_days (d : ℕ) (A_rate B_rate remaining_work : ℚ) :
  A_rate = 1 / 15 ∧ B_rate = 1 / 20 ∧ remaining_work = 8 / 15 →
  (d : ℚ) = 4 :=
by
  assume h,
  have A_rate := h.1.1,
  have B_rate := h.1.2,
  have remaining_work := h.2,
  sorry

end work_together_days_l515_515776


namespace Carrie_pays_94_l515_515843

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l515_515843


namespace Maria_height_in_meters_l515_515651

theorem Maria_height_in_meters :
  let inch_to_cm := 2.54
  let cm_to_m := 0.01
  let height_in_inch := 54
  let height_in_cm := height_in_inch * inch_to_cm
  let height_in_m := height_in_cm * cm_to_m
  let rounded_height_in_m := Float.round (height_in_m * 1000) / 1000
  rounded_height_in_m = 1.372 := 
by
  sorry

end Maria_height_in_meters_l515_515651


namespace checkerboard_7_strips_l515_515900

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l515_515900


namespace molly_age_l515_515758

variable (S M : ℕ)

theorem molly_age (h1 : S / M = 4 / 3) (h2 : S + 6 = 38) : M = 24 :=
by
  sorry

end molly_age_l515_515758


namespace commute_times_l515_515025

theorem commute_times (x y : ℝ) 
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) : |x - y| = 4 := 
sorry

end commute_times_l515_515025


namespace jill_total_tax_percentage_l515_515757

theorem jill_total_tax_percentage (total_spent : ℝ) 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ)
  (tax_clothing_rate : ℝ) (tax_food_rate : ℝ) (tax_other_rate : ℝ)
  (h_clothing : spent_clothing = 0.45 * total_spent)
  (h_food : spent_food = 0.45 * total_spent)
  (h_other : spent_other = 0.10 * total_spent)
  (h_tax_clothing : tax_clothing_rate = 0.05)
  (h_tax_food : tax_food_rate = 0.0)
  (h_tax_other : tax_other_rate = 0.10) :
  ((spent_clothing * tax_clothing_rate + spent_food * tax_food_rate + spent_other * tax_other_rate) / total_spent) * 100 = 3.25 :=
by
  sorry

end jill_total_tax_percentage_l515_515757


namespace area_of_region_R_l515_515371

noncomputable def area_of_R : ℝ :=
  let triangle_abe_area := (sqrt 3) / 4 
  let strip_area := (1 / 4) * 2 
  strip_area - triangle_abe_area

theorem area_of_region_R : area_of_R = 1 / 2 :=
by
  sorry

end area_of_region_R_l515_515371


namespace relationship_y1_y2_y3_l515_515155

theorem relationship_y1_y2_y3 :
  ∀ (y1 y2 y3 : ℝ), y1 = 6 ∧ y2 = 3 ∧ y3 = -2 → y1 > y2 ∧ y2 > y3 :=
by 
  intros y1 y2 y3 h
  sorry

end relationship_y1_y2_y3_l515_515155


namespace zero_in_interval_l515_515352

noncomputable def f (x : ℝ) : ℝ := 3^x + Real.log x - 5

theorem zero_in_interval :
  (∃ c ∈ set.Ioo 1 2, f c = 0) :=
by 
  have h_cont : continuous_on f (set.Icc 1 2) := sorry
  have h_monotonic : strict_mono_incr_on f (set.Icc 1 2) := sorry
  have h_f1 : f 1 < 0 := sorry
  have h_f2 : f 2 > 0 := sorry
  sorry

end zero_in_interval_l515_515352


namespace problem1_l515_515767

theorem problem1 {boys girls : ℕ} (h1 : boys = 5) (h2 : girls = 3) :
  (∃ x y, binomial girls 2 * binomial boys 3 * x.factorial = 3600) :=
by {
  have h3 : binomial girls 2 = 3 := by sorry,
  have h4 : binomial boys 3 = 10 := by sorry,
  existsi 5,
  simp [h3, h4],
  linarith,
  sorry
}

end problem1_l515_515767


namespace complex_number_coordinates_l515_515202

theorem complex_number_coordinates (z : ℂ) (hz : complex.I * z = 2 + 4 * complex.I) :
  z = 4 - 2 * complex.I := 
by
  sorry

end complex_number_coordinates_l515_515202


namespace num_turtles_on_sand_l515_515037

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l515_515037


namespace general_term_of_sequence_l515_515190

theorem general_term_of_sequence (a : ℕ → ℝ) (h₀ : a 1 = 1 / 2) (h₁ : ∀ n, n ≥ 1 → (∑ k in finset.range n, a k.succ) = n^2 * a n.succ) :
  ∀ n, n ≥ 1 → a n = 1 / (2 * n * (n + 1)) :=
begin
  sorry
end

end general_term_of_sequence_l515_515190


namespace least_odd_prime_factor_of_2023_power6_plus_1_l515_515476

theorem least_odd_prime_factor_of_2023_power6_plus_1 :
  ∃ (p : ℕ), prime p ∧ odd p ∧ p ∣ (2023^6 + 1) ∧ p = 13 :=
begin
  sorry
end

end least_odd_prime_factor_of_2023_power6_plus_1_l515_515476


namespace triangular_faces_area_of_pyramid_l515_515734

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l515_515734


namespace student_average_greater_l515_515805

variables (x y w z : ℝ)
variables (hxy : x < y) (hyw : y < w) (hwz : w < z)

def true_average : ℝ := (x + y + w + z) / 4
def student_average : ℝ := (x + y / 2 + w + z) / 3

theorem student_average_greater (hxy : x < y) (hyw : y < w) (hwz : w < z) : student_average x y w z > true_average x y w z :=
sorry

end student_average_greater_l515_515805


namespace positive_solution_is_perfect_square_l515_515251

theorem positive_solution_is_perfect_square
  (t : ℤ)
  (n : ℕ)
  (h : n > 0)
  (root_cond : (n : ℤ)^2 + (4 * t - 1) * n + 4 * t^2 = 0) :
  ∃ k : ℕ, n = k^2 :=
sorry

end positive_solution_is_perfect_square_l515_515251


namespace number_of_k_positive_integer_solutions_l515_515128

theorem number_of_k_positive_integer_solutions :
  (∀ k : ℕ, (∃ x y : ℕ, 9 * x + 4 * y = 600 ∧ k * x - 4 * y = 24)) → 
  (card {k : ℕ | ∃ x y : ℕ, 9 * x + 4 * y = 600 ∧ k * x - 4 * y = 24 ∧ x > 0 ∧ y > 0} = 7) :=
sorry

end number_of_k_positive_integer_solutions_l515_515128


namespace power_equivalence_l515_515559

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l515_515559


namespace area_of_triangle_l515_515602

variables {V : Type*} [inner_product_space ℝ V]

def length_v {v : V} : ℝ :=
  real.sqrt (⟪v, v⟫)

theorem area_of_triangle
  (A B C : V)
  (hAB : length_v (B - A) = 2)
  (hAC : length_v (C - A) = 3)
  (hDot : ⟪B - A, C - A⟫ = -3) :
  ∃ S : ℝ, S = (3 * real.sqrt 3) / 2 :=
sorry

end area_of_triangle_l515_515602


namespace sum_50_to_75_l515_515723

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end sum_50_to_75_l515_515723


namespace arctan_tan_75_minus_2_tan_30_eq_75_l515_515077

theorem arctan_tan_75_minus_2_tan_30_eq_75 : 
  arctan (tan 75 - 2 * tan 30) = 75 :=
by
  sorry

end arctan_tan_75_minus_2_tan_30_eq_75_l515_515077


namespace sum_of_arithmetic_sequence_l515_515125

theorem sum_of_arithmetic_sequence (n a l : ℕ) (h1 : n = 10) (h2 : a = 1) (h3 : l = 37) : 
  (\sum i in finset.range n, (a + (i * ((l - a) / (n - 1))))) = 190 := by
  sorry

end sum_of_arithmetic_sequence_l515_515125


namespace coefficient_x3_in_expansion_l515_515597

theorem coefficient_x3_in_expansion :
  let expansion := (x^2 + 1/x + 1)^6 in
  polynomial.coeff expansion 3 = 80 :=
sorry

end coefficient_x3_in_expansion_l515_515597


namespace mushrooms_weight_change_l515_515131

-- Conditions
variables (x W : ℝ)
variable (initial_weight : ℝ := 100 * x)
variable (dry_weight : ℝ := x)
variable (final_weight_dry : ℝ := 2 * W / 100)

-- Given fresh mushrooms have moisture content of 99%
-- and dried mushrooms have moisture content of 98%
theorem mushrooms_weight_change 
  (h1 : dry_weight = x) 
  (h2 : final_weight_dry = x / 0.02) 
  (h3 : W = x / 0.02) 
  (initial_weight : ℝ := 100 * x) : 
  2 * W = initial_weight / 2 :=
by
  -- This is a placeholder for the proof steps which we skip
  sorry

end mushrooms_weight_change_l515_515131


namespace marco_score_percentage_less_l515_515325

theorem marco_score_percentage_less
  (average_score : ℕ)
  (margaret_score : ℕ)
  (margaret_more_than_marco : ℕ)
  (h1 : average_score = 90)
  (h2 : margaret_score = 86)
  (h3 : margaret_more_than_marco = 5) :
  (average_score - (margaret_score - margaret_more_than_marco)) * 100 / average_score = 10 :=
by
  sorry

end marco_score_percentage_less_l515_515325


namespace min_colors_for_painting_triangles_on_grid_l515_515465

theorem min_colors_for_painting_triangles_on_grid : 
  ∀ (n : ℕ), n = 100 →
  ∀ (board : fin n × fin n → σ), 
  (∀ (cell : fin n × fin n), ∃ (triangles : fin 2 → ℕ), 
    (triangle1 ∈ triangles) → (triangle2 ∈ triangles) → 
    (triangle1 ≠ triangle2) → 
    color_triangle triangle1 ≠ color_triangle triangle2) →
  ∀ (a b : fin n × fin n) (ha : color_triangle (triangle a) = color_triangle (triangle b)),
  ∃ (colors : fin 8), 
  (∀ (x y : fin n × fin n), x ≠ y → color_triangle x ≠ color_triangle y)  
  :=
begin
  -- Sorry, skipping proof details
  sorry
end

end min_colors_for_painting_triangles_on_grid_l515_515465


namespace rectangle_altitude_decrease_l515_515571

theorem rectangle_altitude_decrease
  (b h : ℝ)
  (h₀ : b > 0)
  (h₁ : h > 0)
  (h_area : Real)
  (h_bnew : ∀ (b_new : ℝ), b_new = 1.1 * b)
  (h_area_unchanged : h_area = b * h) :
  let h_new := h * 10 / 11 in
  (h - h_new) / h * 100 = 9 + 1 / 11 :=
by 
  sorry

end rectangle_altitude_decrease_l515_515571


namespace daniel_task_time_l515_515083

-- Define the problem conditions and correct answer
def permutations_of_multiset (n : ℕ) (counts : List ℕ) : ℕ :=
  Nat.factorial n / List.foldl (λ acc x -> acc * Nat.factorial x) 1 counts

def time_to_write_rearrangements (rearrangements_per_minute : ℚ) (total_rearrangements : ℕ) : ℚ :=
  total_rearrangements / rearrangements_per_minute / 60

theorem daniel_task_time :
  let name := "Anna"
  let n := 4 -- Length of the name "Anna"
  let counts := [2, 1, 1] -- Frequency counts of each character in "Anna": ['a', 'n', 'n', 'a']
  let rearrangements_per_minute := 15
  let permutations := permutations_of_multiset n counts
  time_to_write_rearrangements rearrangements_per_minute permutations = 1 / 75 :=
by
  sorry

end daniel_task_time_l515_515083


namespace find_y_l515_515069

variable (A B C : Point)

def carla_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees clockwise about point B lands at point C
  sorry

def devon_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees counterclockwise about point B lands at point C
  sorry

theorem find_y
  (h1 : carla_rotate 690 A B C)
  (h2 : ∀ y, devon_rotate y A B C)
  (h3 : y < 360) :
  ∃ y, y = 30 :=
by
  sorry

end find_y_l515_515069


namespace volume_error_is_neg_2p728_l515_515997

variable (a b c : ℝ) -- Variables for the sides of the rectangular prism

def actual_volume (a b c : ℝ) : ℝ :=
  a * b * c

def erroneous_volume (a b c : ℝ) : ℝ :=
  (a * 1.08) * (b * 0.90) * (c * 0.94)

def volume_error_percentage (a b c : ℝ) : ℝ :=
  let V_actual := actual_volume a b c
  let V_erroneous := erroneous_volume a b c
  ((V_erroneous - V_actual) / V_actual) * 100

theorem volume_error_is_neg_2p728 (a b c : ℝ) : volume_error_percentage a b c = -2.728 :=
  sorry

end volume_error_is_neg_2p728_l515_515997


namespace stella_profit_l515_515316

theorem stella_profit 
    (dolls : ℕ) (doll_price : ℕ) 
    (clocks : ℕ) (clock_price : ℕ) 
    (glasses : ℕ) (glass_price : ℕ) 
    (cost : ℕ) :
    dolls = 3 →
    doll_price = 5 →
    clocks = 2 →
    clock_price = 15 →
    glasses = 5 →
    glass_price = 4 →
    cost = 40 →
    (dolls * doll_price + clocks * clock_price + glasses * glass_price - cost) = 25 := 
by 
  intros h_dolls h_doll_price h_clocks h_clock_price h_glasses h_glass_price h_cost
  rw [h_dolls, h_doll_price, h_clocks, h_clock_price, h_glasses, h_glass_price, h_cost]
  norm_num
  sorry

end stella_profit_l515_515316


namespace visiting_plans_count_l515_515462

-- Let's define the exhibitions
inductive Exhibition
| OperaCultureExhibition
| MingDynastyImperialCellarPorcelainExhibition
| AncientGreenLandscapePaintingExhibition
| ZhaoMengfuCalligraphyAndPaintingExhibition

open Exhibition

-- The condition is that the student must visit at least one painting exhibition in the morning and another in the afternoon
-- Proof that the number of different visiting plans is 10.
theorem visiting_plans_count :
  let exhibitions := [OperaCultureExhibition, MingDynastyImperialCellarPorcelainExhibition, AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  let painting_exhibitions := [AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  ∃ visits : List (Exhibition × Exhibition), (∀ (m a : Exhibition), (m ∈ painting_exhibitions ∨ a ∈ painting_exhibitions)) → visits.length = 10 :=
sorry

end visiting_plans_count_l515_515462


namespace expression_eval_l515_515392

theorem expression_eval :
  -14 - (-2) ^ 3 * (1 / 4) - 16 * (1 / 2 - 1 / 4 + 3 / 8) = -22 := by
  sorry

end expression_eval_l515_515392


namespace sphere_volume_expansion_l515_515426

theorem sphere_volume_expansion (r : ℝ) : 
  let V := (λ r, (4 * real.pi * r^3) / 3) in
  V (2 * r) = 8 * V r := 
by 
  -- skip the proof
  sorry

end sphere_volume_expansion_l515_515426


namespace remainder_21_l515_515760

theorem remainder_21 (y : ℤ) (k : ℤ) (h : y = 288 * k + 45) : y % 24 = 21 := 
  sorry

end remainder_21_l515_515760


namespace total_area_of_pyramid_faces_l515_515739

theorem total_area_of_pyramid_faces (b l : ℕ) (hb : b = 8) (hl : l = 10) : 
  let h : ℝ := Math.sqrt (l^2 - (b / 2)^2) in
  let A : ℝ := 1 / 2 * b * h in
  let T : ℝ := 4 * A in
  T = 32 * Math.sqrt 21 := by
  -- Definitions
  have b_val : (b : ℝ) = 8 := by exact_mod_cast hb
  have l_val : (l : ℝ) = 10 := by exact_mod_cast hl

  -- Calculations
  have h_val : h = Math.sqrt (l^2 - (b / 2)^2) := rfl
  have h_simplified : h = 2 * Math.sqrt 21 := by
    rw [h_val, l_val, b_val]
    norm_num
    simp

  have A_val : A = 1 / 2 * b * h := rfl
  simp_rw [A_val, h_simplified, b_val]
  norm_num

  have T_val : T = 4 * A := rfl
  simp_rw [T_val]
  norm_num

  -- Final proof
  sorry

end total_area_of_pyramid_faces_l515_515739


namespace ratio_triangle_circle_l515_515820

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let A_triangle := (sqrt 3 / 4) * (3 * r)^2
  let A_circle := π * r^2
  A_triangle / A_circle

theorem ratio_triangle_circle (r : ℝ) (h_r : r > 0) :
  ratio_of_areas r = 9 * sqrt 3 / (4 * π) :=
by
  sorry

end ratio_triangle_circle_l515_515820


namespace area_of_triangle_l515_515653

-- Define a structure representing a triangle
structure Triangle (α : Type) [Field α] :=
  (A B C : α × α)

-- Define the length of a median given the endpoints
def median_length {α : Type} [Field α] (p1 p2 : α × α) : α :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement of the problem
theorem area_of_triangle (α : Type) [Field α] (T : Triangle α)
  (D E : α × α) (G : α × α)
  (hmediansAD : D = ((2 : α) / 3, (2 : α) / 3))
  (hmediansBE : E = ((1 : α) / 3, (1 : α) / 3))
  (h_perpendicular : (D.1 - G.1) * (E.1 - G.1) + (D.2 - G.2) * (E.2 - G.2) = 0)
  (h_AD : median_length T.A D = 6)
  (h_BE : median_length T.B E = 9) :
  (1 / 2) * ((2 / 3) * 6) * ((1 / 3) * 9) * 6 = 36 := sorry

end area_of_triangle_l515_515653


namespace sum_four_digit_numbers_l515_515491

theorem sum_four_digit_numbers : ∑ n in {p | ∀ d ∈ p_digits n, d ∈ {1, 2, 3, 4, 5} ∧ nodup p_digits n ∧ p_digits n.length = 4}, n = 399960 :=
by
  sorry

end sum_four_digit_numbers_l515_515491


namespace gcd_gx_x_eq_144_l515_515164

theorem gcd_gx_x_eq_144 (x : ℕ) (h : 12096 ∣ x) : 
  gcd ((3*x + 8)*(5*x + 1)*(11*x + 6)*(2*x + 3)) x = 144 :=
sorry

end gcd_gx_x_eq_144_l515_515164


namespace cube_surface_area_increase_l515_515755

theorem cube_surface_area_increase (a : ℝ) (h : a > 0) : 
  let SA_original := 6 * a^2 in
  let a_new := 1.10 * a in
  let SA_new := 6 * (a_new)^2 in
  let Percentage_increase := ((SA_new - SA_original) / SA_original) * 100 in
  Percentage_increase = 21 :=
by
  let SA_original := 6 * a^2
  let a_new := 1.10 * a
  let SA_new := 6 * (a_new)^2
  let Percentage_increase := ((SA_new - SA_original) / SA_original) * 100
  have h1 : SA_new = 7.26 * a^2 := sorry
  have h2 : Percentage_increase = 21 := sorry
  exact h2

end cube_surface_area_increase_l515_515755


namespace maximum_magical_pairs_l515_515021

/-- Maximum magical pairs of adjacent numbers -/
theorem maximum_magical_pairs (nums : List ℕ) (h : ∀ n ∈ nums, 1 ≤ n ∧ n ≤ 30) 
  (adj_magical_pairs : ∀ (i : ℕ), i < nums.length - 1 → ((nums[i] + nums[i + 1]) % 7 = 0) → True) :
  ∃ (perm : List ℕ), perm ~ nums ∧ (finset.range (nums.length - 1)).card - 3 = 26 := 
by sorry

end maximum_magical_pairs_l515_515021


namespace pyramid_total_area_l515_515743

noncomputable def pyramid_base_edge := 8
noncomputable def pyramid_lateral_edge := 10

/-- The total area of the four triangular faces of a right, square-based pyramid
with base edges measuring 8 units and lateral edges measuring 10 units is 32 * sqrt(21). -/
theorem pyramid_total_area :
  let base_edge := pyramid_base_edge,
      lateral_edge := pyramid_lateral_edge,
      height := sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2),
      area_of_one_face := 1 / 2 * base_edge * height
  in 4 * area_of_one_face = 32 * sqrt 21 :=
sorry

end pyramid_total_area_l515_515743


namespace sum_cross_sectional_areas_of_tetrahedron_l515_515517

theorem sum_cross_sectional_areas_of_tetrahedron (ABCD : Type) [regular_tetrahedron ABCD (edge_length := 2)] :
  cross_sectional_areas_sum_tertrahedron ABCD = 3 + Real.sqrt 3 :=
by
  sorry

end sum_cross_sectional_areas_of_tetrahedron_l515_515517


namespace rearrangements_of_abcde_l515_515975

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 == 'a' ∧ c2 == 'b') ∨ 
  (c1 == 'b' ∧ c1 == 'a') ∨ 
  (c1 == 'b' ∧ c2 == 'c') ∨ 
  (c1 == 'c' ∧ c2 == 'b') ∨ 
  (c1 == 'c' ∧ c2 == 'd') ∨ 
  (c1 == 'd' ∧ c2 == 'c') ∨ 
  (c1 == 'd' ∧ c2 == 'e') ∨ 
  (c1 == 'e' ∧ c2 == 'd')

def is_valid_rearrangement (lst : List Char) : Bool :=
  match lst with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => 
    ¬is_adjacent c1 c2 ∧ is_valid_rearrangement (c2 :: rest)

def count_valid_rearrangements (chars : List Char) : Nat :=
  chars.permutations.filter is_valid_rearrangement |>.length

theorem rearrangements_of_abcde : count_valid_rearrangements ['a', 'b', 'c', 'd', 'e'] = 8 := 
by
  sorry

end rearrangements_of_abcde_l515_515975


namespace solve_real_solution_l515_515874

theorem solve_real_solution:
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔
           (x = 4 + Real.sqrt 57) ∨ (x = 4 - Real.sqrt 57) :=
by
  sorry

end solve_real_solution_l515_515874


namespace petya_obtains_11_triangles_l515_515761

noncomputable theory
open Classical

def breaking_triangle (angle_A : ℕ) (angle_B : ℕ) (angle_C : ℕ) : ℕ :=
sorry -- placeholder for the actual function

theorem petya_obtains_11_triangles :
  breaking_triangle 3 88 89 = 11 :=
sorry

end petya_obtains_11_triangles_l515_515761


namespace fractional_eq_solution_l515_515129

theorem fractional_eq_solution (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) →
  k ≠ -3 ∧ k ≠ 5 :=
by 
  sorry

end fractional_eq_solution_l515_515129


namespace number_of_adjacent_subsets_equals_1634_l515_515369

/-- Twelve chairs arranged in a circular formation. -/
def chairs : Finset ℕ := Finset.range 12

/-- Subsets of the set of chairs that include at least three adjacent chairs. -/
def adjacent_subsets (s : Finset ℕ) : Prop :=
  ∃ i : ℕ, s = finset.range 3 ∨ s = finset.range 4 ∨ s = finset.range 5 ∨
           s = finset.range 6 ∨ s.card >= 7

theorem number_of_adjacent_subsets_equals_1634 :
  (chairs.powerset.filter adjacent_subsets).card = 1634 :=
by
  sorry

end number_of_adjacent_subsets_equals_1634_l515_515369


namespace water_fee_20_water_fee_55_l515_515219

-- Define the water charge method as a function
def water_fee (a : ℕ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- Prove the specific cases
theorem water_fee_20 :
  water_fee 20 = 42.5 :=
by sorry

theorem water_fee_55 :
  (∃ a : ℕ, water_fee a = 55) ↔ (a = 25) :=
by sorry

end water_fee_20_water_fee_55_l515_515219


namespace willy_episodes_per_day_l515_515750

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def episodes_per_day (total_episodes : ℕ) (days : ℕ) : ℕ :=
  total_episodes / days

theorem willy_episodes_per_day :
  episodes_per_day (total_episodes 3 20) 30 = 2 :=
by
  sorry

end willy_episodes_per_day_l515_515750


namespace new_probability_of_blue_ball_l515_515410

theorem new_probability_of_blue_ball 
  (initial_total_balls : ℕ) (initial_blue_balls : ℕ) (removed_blue_balls : ℕ) :
  initial_total_balls = 18 →
  initial_blue_balls = 6 →
  removed_blue_balls = 3 →
  (initial_blue_balls - removed_blue_balls) / (initial_total_balls - removed_blue_balls) = 1 / 5 :=
by
  sorry

end new_probability_of_blue_ball_l515_515410


namespace ab_squared_l515_515185

noncomputable def trig_function_max_min (a b : ℝ) : Prop :=
  ∀ x : ℝ, 
  let y := (a * Real.cos x + b * Real.sin x) * Real.cos x in
  (∃ c d : ℝ, 
    (∀ x : ℝ, y ≤ c) ∧ 
    (∀ x : ℝ, y ≥ d) ∧ 
    (c = 2) ∧ 
    (d = -1))

theorem ab_squared (a b : ℝ) (h : trig_function_max_min a b) : (a * b)^2 = 8 :=
sorry

end ab_squared_l515_515185


namespace arrangement_count_l515_515430

theorem arrangement_count (A B C D E : Type) : 
  ∃ (a b c d e : A) (cond : (b = e ∧ a = e) ∨ (b = a ∧ a = e)), 
  true := 
begin
  sorry
end

end arrangement_count_l515_515430


namespace ellipse_c_equation_chord_length_max_l515_515178

noncomputable def ellipse_c_eq (a b : ℝ) (h1 : a > b) (h2 : a^2 + b^2 = 8) 
  (hb : b / a = Real.sqrt 3 / 3) : Prop :=
  (a^2 = 6) ∧ (b^2 = 2) ∧ (∀ x y, (x^2 / 6) + (y^2 / 2) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

theorem ellipse_c_equation (a b : ℝ) (h1 : a > b) (h2 : a^2 + b^2 = 8) 
  (hb : b / a = Real.sqrt 3 / 3) : 
    ∀ (ellipse_c_eq : Prop), ellipse_c_eq (a b h1 h2 hb) := 
sorry

def max_triangle_chord_len (a : ℝ) (F : ℝ → ℝ)
  (circle_eq : ∀ x y, x^2 + y^2 = a^2)
  (h3 : ∀ l : ℝ → ℝ, l = (x, y) -> y = mx + 2)
  (hyp_focus : F = 2)
  (line_F : ∀ x y, x = (my + 2))
  (max_area : ∀ S_triangle OAB -> S_triangle = sqrt 3)
  (d1 : ℝ = sqrt 2)
  (d2 : ℝ = 4) : Prop :=
  ((l = y -> x = 1) ∧ (l = y -> x = 1)
  ((d2 = 2 sqrt(r^2 - d1^2)) -> d2 = 4)
 
theorem chord_length_max (a : ℝ) (F : ℝ → ℝ)
  (circle_eq : ∀ x y, x^2 + y^2 = a^2)
  (h3 : ∀ l : ℝ → ℝ, l = (x, y) ↔ y = mx + 2)
  (hyp_focus : F = 2) 
  (line_F : ∀ x y, x = (my + 2))
  (max_area : ∀ S_triangle OAB -> S_triangle = sqrt 3)
  (d1 : ℝ = (sqrt 2))
  (d2 : ℝ = 4) :
  ∀ (max_triangle_chord_len : Prop), max_triangle_chord_len (a F circle_eq h3 hyp_focus line_F max_area d1 d2) := 
sorry

end ellipse_c_equation_chord_length_max_l515_515178


namespace shaded_percentage_of_grid_l515_515081

theorem shaded_percentage_of_grid :
  let grid_size := 6 * 6 in
  let odd_rows := [1, 3, 5].map (λ _ => 4) in
  let even_rows := [2, 4, 6].map (λ _ => 3) in
  let shaded_squares := odd_rows.sum + even_rows.sum in
  (shaded_squares : ℚ) / grid_size * 100 = 58.33 := sorry

end shaded_percentage_of_grid_l515_515081


namespace product_common_divisors_180_20_l515_515885

def integer_divisors (n : ℤ) : set ℤ :=
  {d | d ∣ n}

def common_divisors (a b : ℤ) : set ℤ :=
  integer_divisors a ∩ integer_divisors b

noncomputable def product_of_set (s : set ℤ) : ℤ :=
  s.to_finset.prod id
  -- Assuming id is an identity function over the integers

theorem product_common_divisors_180_20 :
  product_of_set (common_divisors 180 20) = 640000000 :=
by {
  sorry
}

end product_common_divisors_180_20_l515_515885


namespace total_trolls_l515_515097

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l515_515097


namespace line_parallel_or_skew_l515_515747

open_locale classical

variables {Point : Type*} {Line : Type*} {Plane : Type*}
variable parallel : Line → Line → Prop
variable parallel_plane : Line → Plane → Prop
variable inside_plane : Line → Plane → Prop
variable skew : Line → Line → Prop

theorem line_parallel_or_skew
  (a b : Line)
  (alpha : Plane)
  (h1 : parallel_plane a alpha)
  (h2 : inside_plane b alpha) :
  parallel a b ∨ skew a b :=
sorry

end line_parallel_or_skew_l515_515747


namespace area_triangle_MBC_constant_l515_515614

theorem area_triangle_MBC_constant
  (A B C D B' C' M : Point)
  (hABC : Triangle A B C)
  (hAB_eq_AC : AB = AC)
  (hD_in_BC : D ∈ Segment [B, C])
  (hBC_greater_than_BD : BC > BD)
  (hBD_greater_than_DC : BD > DC)
  (hDC_greater_than_zero : DC > 0)
  (circumcircle_ABD : Circle)
  (circumcircle_ADC : Circle)
  (hCirc_ABD : Circumcircle Triangle A B D circumcircle_ABD)
  (hCirc_ADC : Circumcircle Triangle A C D circumcircle_ADC)
  (hBBprime : Diameter circumcircle_ABD B B')
  (hCCprime : Diameter circumcircle_ADC C C')
  (hM_midpoint_BCprime : Midpoint M B' C') :
  area (Triangle M B C) = constant :=
sorry

end area_triangle_MBC_constant_l515_515614


namespace ellipse_equation_l515_515153

open Real

noncomputable def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def hyperbola : Prop :=
  ∀ x y : ℝ, (x^2 / 2 - y^2 / 2 = 1)

noncomputable def quadrilateral_area : ℝ := 16

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - b^2 / a^2)

-- The main theorem stating the problem in Lean
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b)
  (he : eccentricity a b = sqrt 3 / 2)
  (intersection_area : quadrilateral_area = 16):
  ellipse a b ha hb h → hyperbola → (∃ a b, a^2 = 20 ∧ b^2 = 5 ∧ ellipse 20 5 ha hb h) := 
by
  sorry

end ellipse_equation_l515_515153


namespace swimming_third_place_either_B_or_C_l515_515390

variable (a b c : ℕ)
variable (A_score B_score C_score : ℕ)
variable (event_results : ℕ → ℕ)
variable (eq B_won_equestrian : Prop)

-- Conditions
def valid_distribution := a > b ∧ b > c ∧ a + b + c = 8
def A_final_score := A_score = 22
def B_final_score := B_score = 9
def C_final_score := C_score = 9
def B_won_equestrian := B_won_equestrian = true ∧ event_results 1 = a ∧ a = 5  -- Assuming first event is equestrian

-- Main theorem: The third-place finisher in the swimming event is either B or C.
theorem swimming_third_place_either_B_or_C :
  valid_distribution a b c →
  A_final_score A_score →
  B_final_score B_score →
  C_final_score C_score →
  B_won_equestrian →
  (∃ x, x == "B_or_C") :=
by
  sorry  -- proof goes here

end swimming_third_place_either_B_or_C_l515_515390


namespace B_can_catch_A_l515_515708

variables (v v' : ℝ) (v_ge_v' : v' ≤ v) (x0 : ℝ)

def A_position (t : ℝ) : ℝ := v * t

def B_position (t x'_0 y'_t : ℝ) : ℝ × ℝ := 
  let x'_t := x'_0 + v' * t * (v / v')
  (x'_t, y'_t)

theorem B_can_catch_A (t x'_0 : ℝ) (h_x0 : x'_0 ≥ 0) : 
  ∃ t : ℝ, B_position v v' t x'_0 0 = (A_position v t, 0) :=
sorry

end B_can_catch_A_l515_515708


namespace arctan_tan_75_sub_2_tan_30_eq_l515_515075

theorem arctan_tan_75_sub_2_tan_30_eq :
  arctan (tan (75 * real.pi / 180) - 2 * tan (30 * real.pi / 180)) * 180 / real.pi = 75 :=
sorry

end arctan_tan_75_sub_2_tan_30_eq_l515_515075


namespace general_term_formula_sum_of_first_n_bn_l515_515929

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable {q : ℝ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  ∃ d : ℤ, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

theorem general_term_formula (h1 : sum_of_first_n_terms a_n 3 = -6)
    (h2 : sum_of_first_n_terms a_n 8 = 24)
    (h_arith : is_arithmetic_sequence a_n) :
  ∀ n : ℕ, a_n n = 2 * n - 6 := 
sorry

theorem sum_of_first_n_bn (h1 : sum_of_first_n_terms a_n 3 = -6)
    (h2 : sum_of_first_n_terms a_n 8 = 24)
    (h_arith : is_arithmetic_sequence a_n)
    (h_b : ∀ n : ℕ, b_n n = (a_n n + 6) * q^n) :
  ∀ n : ℕ,
  (if q = 1 then S_n n = n^2 + n
   else if q ≠ 0 then S_n n = -2 * (q^(n+1) - q) / (q - 1) + 2 * n * q^(n+1)
   else false) := 
sorry

end general_term_formula_sum_of_first_n_bn_l515_515929


namespace P_abs_X_eq_1_D_Y_ne_10_div_9_l515_515952

open ProbabilityTheory

variable {Ω : Type*} (p : pmf Ω)

-- Assumption on distribution of X
variable (X : Ω → ℤ) 
variable (pX : pmf (ℤ))
axiom hX_dist : pX (-1) = 1/2 ∧ pX 0 = 1/3 ∧ pX 1 = 1/6

-- Definition of Y
def Y (ω : Ω) : ℤ := 2 * X ω + 1

-- Statements to be proven
theorem P_abs_X_eq_1 : pX (1) + pX (-1) = 2/3 := by sorry

theorem D_Y_ne_10_div_9 : ¬(variance (Y p) = 10/9) := by sorry

end P_abs_X_eq_1_D_Y_ne_10_div_9_l515_515952


namespace smallest_positive_period_of_f_intervals_of_decrease_of_f_range_of_f_in_interval_l515_515954

noncomputable def f (x : ℝ) := -sin (2 * x) - sqrt (3) * (1 - 2 * (sin x)^2) + 1

theorem smallest_positive_period_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) := sorry

theorem intervals_of_decrease_of_f :
  (∀ k : ℤ, ∀ x, k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12 → ∃ x', f x ≤ f x') := sorry

theorem range_of_f_in_interval :
  ∀ x ∈ Icc (-π / 6) (π / 6), -1 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_of_f_intervals_of_decrease_of_f_range_of_f_in_interval_l515_515954


namespace height_of_model_logan_model_height_l515_515649

theorem height_of_model (h_original : ℝ) (V_original : ℝ) (V_model : ℝ) : ℝ :=
by 
  have V_ratio := V_original / V_model
  have size_ratio := V_ratio ** (1 / 3)
  have h_model := h_original / size_ratio
  exact h_model

theorem logan_model_height :
  height_of_model 50 100000 0.05 = 0.3968 :=
by
  unfold height_of_model
  have V_ratio := 100000 / 0.05
  have size_ratio := V_ratio ** (1 / 3)
  have h_model := 50 / size_ratio
  have approx_h_model : abs (h_model - 0.3968) < 0.0001, from by sorry
  linarith [approx_h_model]

end height_of_model_logan_model_height_l515_515649


namespace find_sin_2α_plus_β_find_β_l515_515527

variable (α β : Real)
variables (sin_α cos_α sin_α_plus_β cos_α_plus_β : Real)

noncomputable def sin_α_def : sin_α = (4 * Real.sqrt 3) / 7 := by sorry
noncomputable def cos_α_plus_β_def : cos_α_plus_β = -11 / 14 := by sorry
noncomputable def α_in_I : 0 < α ∧ α < Real.pi / 2 := by sorry
noncomputable def β_in_I : 0 < β ∧ β < Real.pi / 2 := by sorry

theorem find_sin_2α_plus_β :
  sin_2α_plus_β = -39 * Real.sqrt 3 / 98 :=
by
  have sin_2α_plus_β := sin (2 * α + β)
  sorry -- proof omitted

theorem find_β :
  β = Real.pi / 3 :=
by
  sorry -- proof omitted

end find_sin_2α_plus_β_find_β_l515_515527


namespace coffee_shop_error_l515_515212

variables (x y z : ℕ)

def half_dollar_overestimation (y : ℕ) : ℕ := 25 * y
def nickels_underestimation (z : ℕ) : ℕ := 4 * z
def quarters_overestimation_twice (x : ℕ) : ℕ := 50 * x

theorem coffee_shop_error (x y z : ℕ) :
  let net_error := half_dollar_overestimation y + quarters_overestimation_twice x - nickels_underestimation z in
  net_error = 25 * y + 50 * x - 4 * z := 
by
  -- The proof goes here.
  sorry

end coffee_shop_error_l515_515212


namespace range_of_a_l515_515133

-- Definitions of sets A and B
def A (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| < 2
def B (x a : ℝ) : Prop := x^2 - (a + 1) * x + a < 0

-- The condition A ∩ B ≠ ∅
def nonempty_intersection (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a

-- Proving the required range of a
theorem range_of_a : {a : ℝ | nonempty_intersection a} = {a : ℝ | a < 1 ∨ a > 3} := by
  sorry

end range_of_a_l515_515133


namespace number_of_integers_satisfying_condition_l515_515485

def condition (n : ℤ) : Prop :=
  1 + Int.floor (100 * n / 101) = Int.ceil (99 * n / 100)

theorem number_of_integers_satisfying_condition :
  { n : ℤ | condition n }.finite.card = 10100 :=
begin
  sorry
end

end number_of_integers_satisfying_condition_l515_515485


namespace sum_four_digit_numbers_l515_515492

theorem sum_four_digit_numbers : ∑ n in {p | ∀ d ∈ p_digits n, d ∈ {1, 2, 3, 4, 5} ∧ nodup p_digits n ∧ p_digits n.length = 4}, n = 399960 :=
by
  sorry

end sum_four_digit_numbers_l515_515492


namespace circle_AB_over_BC_l515_515658

-- Definitions for the problem conditions
variables {A B C O : Type} [MetricSpace O]
variable {r : ℝ}
variable {s : Real} -- s represents the length equivalence condition
variable (is_circle : MetricSpace.is_circle r) -- circle radius r

-- Point A, B, and C are on the circle, and AB = AC, AB > r.
variable (A B C : O) 
variable (AO BO CO AB BC : ℝ) (hab_eq_hac : AB = AO ∧ AB = CO)
variable (hab_gt_r : AB > r)
variable (hbc_minor : MetricSpace.arc_length B C < 2 * π * r / 2) -- minor arc length

theorem circle_AB_over_BC :
  AB / BC = (1/2) * (real.csc (1/4 * π)) :=
sorry

end circle_AB_over_BC_l515_515658


namespace Ariella_account_balance_l515_515438

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) :=
  P * (1 + r / (n : ℝ))^(n * t)

theorem Ariella_account_balance :
  let Daniella_amount := 400
  let Ariella_amount := Daniella_amount + 200
  let r := 0.10
  let n := 4
  let t := 3
  in compound_interest Ariella_amount r n t = 807.53 := by
  sorry

end Ariella_account_balance_l515_515438


namespace total_cost_of_one_pencil_and_eraser_l515_515588

/-- Lila buys 15 pencils and 7 erasers for 170 cents. A pencil costs less than an eraser, 
neither item costs exactly half as much as the other, and both items cost a whole number of cents. 
Prove that the total cost of one pencil and one eraser is 16 cents. -/
theorem total_cost_of_one_pencil_and_eraser (p e : ℕ) (h1 : 15 * p + 7 * e = 170)
  (h2 : p < e) (h3 : p ≠ e / 2) : p + e = 16 :=
sorry

end total_cost_of_one_pencil_and_eraser_l515_515588


namespace find_k_l515_515966

/--
Given vectors a = (1, 1, 0) and b = (-1, 0, 2),
and assuming k * a + b is perpendicular to 2 * a - b,
prove that k = 7 / 5.
-/
theorem find_k
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (ha : a = ![1, 1, 0]) 
  (hb : b = ![-1, 0, 2]) 
  (k : ℝ) 
  (perp : (k • a + b) ⬝ (2 • a - b) = 0) : 
  k = 7 / 5 :=
  sorry

end find_k_l515_515966


namespace abcd_inequality_l515_515932

theorem abcd_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_eq : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) + (d^2 / (1 + d^2)) = 1) :
  a * b * c * d ≤ 1 / 9 :=
sorry

end abcd_inequality_l515_515932


namespace num_divisors_of_two_b_plus_twelve_l515_515319

theorem num_divisors_of_two_b_plus_twelve (a b : ℤ)
  (h : 3 * b = 8 - 2 * a) :
  {n : ℕ | n ∈ finset.range 6 ∧ n ∣ (2 * b + 12)}.card = 3 :=
sorry

end num_divisors_of_two_b_plus_twelve_l515_515319


namespace right_triangle_of_ratio_l515_515207

theorem right_triangle_of_ratio (x : ℝ) (hx : x > 0) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x in
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_of_ratio_l515_515207


namespace smallest_m_l515_515501

-- Defining the remainder function
def r (m n : ℕ) : ℕ := m % n

-- Main theorem stating the problem needed to be proved
theorem smallest_m (m : ℕ) (h : m > 0) 
  (H : (r m 1 + r m 2 + r m 3 + r m 4 + r m 5 + r m 6 + r m 7 + r m 8 + r m 9 + r m 10) = 4) : 
  m = 120 :=
sorry

end smallest_m_l515_515501


namespace initial_water_amount_l515_515807

theorem initial_water_amount (E D R F I : ℕ) 
  (hE : E = 2000) 
  (hD : D = 3500) 
  (hR : R = 350 * (30 / 10))
  (hF : F = 1550) 
  (h : I - (E + D) + R = F) : 
  I = 6000 :=
by
  sorry

end initial_water_amount_l515_515807


namespace find_w_l515_515471

theorem find_w (w : ℝ) : 7^3 * 7^w = 81 → w = 4 * Real.log 3 / Real.log 7 - 3 :=
by
  assume h : 7^3 * 7^w = 81
  sorry

end find_w_l515_515471


namespace count_whole_numbers_in_interval_l515_515979

theorem count_whole_numbers_in_interval : 
  let a := 2 / 5
  let b := 2 * Real.sqrt 17
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, (a < x ∧ x < b) ↔ (1 <= x ∧ x <= 8) :=
by
  let a := 2 / 5
  let b := 2 * Real.sqrt 17
  exists 8
  split
  sorry

end count_whole_numbers_in_interval_l515_515979


namespace larger_number_eq_1599_l515_515681

theorem larger_number_eq_1599 (L S : ℕ) (h1 : L - S = 1335) (h2 : L = 6 * S + 15) : L = 1599 :=
by 
  sorry

end larger_number_eq_1599_l515_515681


namespace volume_rotation_l515_515065

-- Define the functions and their bounds
def f (x : ℝ) : ℝ := (x - 1) ^ 2
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- Define the volume of the solid of revolution about the y-axis
noncomputable def volume : ℝ := 
  let integral := ∫ y in lower_bound..upper_bound, 4 * real.sqrt y in
  π * integral

-- State the theorem
theorem volume_rotation (y : ℝ) (h₁ : y = f (1 + real.sqrt y)) (h₂ : y = f (1 - real.sqrt y)) : 
  volume = 8 * π / 3 := 
sorry

end volume_rotation_l515_515065


namespace vinegar_percentage_in_first_brand_l515_515778

variable (V : ℝ)  -- Percentage of vinegar in the first brand

-- Assumptions 
constant second_brand_percentage : ℝ := 13
constant total_mixture_percentage : ℝ := 11
constant volume_each_brand : ℝ := 128
constant total_volume_mixture : ℝ := 256

-- The proof problem
theorem vinegar_percentage_in_first_brand :
  (V / 100) * volume_each_brand + (second_brand_percentage / 100) * volume_each_brand =
  (total_mixture_percentage / 100) * total_volume_mixture →
  V = 9 := 
  by
  sorry

end vinegar_percentage_in_first_brand_l515_515778


namespace dogs_in_center_total_l515_515056

theorem dogs_in_center_total :
  let sit := 45
      stay := 35
      roll_over := 40
      sit_stay := 20
      stay_roll_over := 15
      sit_roll_over := 20
      all_three := 12
      none := 8
  in sit + stay + roll_over - (sit_stay + stay_roll_over + sit_roll_over) + all_three + none = 93 :=
by 
  let sit := 45
  let stay := 35
  let roll_over := 40
  let sit_stay := 20
  let stay_roll_over := 15
  let sit_roll_over := 20
  let all_three := 12
  let none := 8
  -- calculate mutually exclusive sets
  let only_sit := sit - sit_roll_over - sit_stay + all_three
  let only_stay := stay - sit_stay - stay_roll_over + all_three
  let only_roll_over := roll_over - sit_roll_over - stay_roll_over + all_three
  let sit_stay_only := sit_stay - all_three
  let stay_roll_over_only := stay_roll_over - all_three
  let sit_roll_over_only := sit_roll_over - all_three
  -- sum up all dogs
  let total := only_sit + only_stay + only_roll_over + sit_stay_only + stay_roll_over_only + sit_roll_over_only + all_three + none
  show total = 93, from sorry

end dogs_in_center_total_l515_515056


namespace probability_each_player_has_1_after_2024_rings_l515_515301

noncomputable def raashan_game : ℕ → (ℕ × ℕ × ℕ) := sorry

theorem probability_each_player_has_1_after_2024_rings :
  let initial_state := (1, 1, 1) in
  let final_state := (1, 1, 1) in
  let rings := 2024 in
  let probability := (5 : ℚ) / 27 in
  (raashan_game rings = final_state ↔ true) → true :=
sorry

end probability_each_player_has_1_after_2024_rings_l515_515301


namespace area_of_triangle_PF1F2_l515_515953

theorem area_of_triangle_PF1F2
  (a b : ℝ)                        -- semi-major and semi-minor axes
  (h1 : a = 4)
  (h2 : b = 3)
  (c : ℝ)                          -- the distance from center to foci
  (h3 : c = Real.sqrt (a^2 - b^2))
  (x y : ℝ)                        -- coordinates of point P
  (h4 : (x^2 / a^2) + (y^2 / b^2) = 1)  -- P is on the ellipse
  (F1 F2 : ℝ × ℝ)                  -- coordinates of foci
  (h5 : F1 = (c, 0))
  (h6 : F2 = (-c, 0))
  (P : ℝ × ℝ)                      -- point P
  (h7 : P = (x, y))
  (right_angle_at_F1 : ∠ F1 P F2 = 90)  -- PF1F2 is a right triangle with right angle at F1
  : 1/2 * (2*c) * (b^2 / a) = (9 * Real.sqrt 7) / 4 :=
sorry

end area_of_triangle_PF1F2_l515_515953


namespace sequence_arithmetic_condition_l515_515514

theorem sequence_arithmetic_condition {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α + β ≠ 0)
  (seq : ℕ → ℝ) (hseq : ∀ n, seq (n + 2) = (α * seq (n + 1) + β * seq n) / (α + β)) :
  ∃ α β : ℝ, (∀ a1 a2 : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α + β = 0 → seq (n + 1) - seq n = seq n - seq (n - 1)) :=
by sorry

end sequence_arithmetic_condition_l515_515514


namespace f_expression_min_f_is_3_range_of_m_l515_515134

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (2 * sqrt 3 * sin x, cos x)

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, 2 * cos x)

noncomputable def f (x m : ℝ) : ℝ :=
  2 * ((2 * sqrt 3 * sin x * cos x) + (cos x * 2 * cos x)) + 2 * m - 1

theorem f_expression (x m : ℝ) :
    f x m = 4 * sin (2 * x + π / 6) + 2 * m + 1 :=
  sorry

theorem min_f_is_3 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hmin : f x 2 = 3) : 
    m = 2 :=
  sorry

theorem range_of_m (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hmax : f x m ≤ 6) :
    m ≤ 1 / 2 :=
  sorry

end f_expression_min_f_is_3_range_of_m_l515_515134


namespace correct_conclusions_l515_515062

noncomputable theory

-- Define the conditions
variables {P Q : Type} [plane_space : P] [plane_space : Q]

-- Define the conclusions
def conclusion1 (l₁ l₂ : P) (L : Q) : Prop :=
  perpendicular l₁ L ∧ perpendicular l₂ L → parallel l₁ l₂

def conclusion2 (l₁ l₂ : P) (π : Q) : Prop :=
  perpendicular l₁ π ∧ perpendicular l₂ π → parallel l₁ l₂

def conclusion3 (π₁ π₂ : P) (L : Q) : Prop :=
  perpendicular π₁ L ∧ perpendicular π₂ L → parallel π₁ π₂

def conclusion4 (π₁ π₂ : P) (π : Q) : Prop :=
  perpendicular π₁ π ∧ perpendicular π₂ π → parallel π₁ π₂

-- Conditions translations
def cond1 (l₁ l₂ : P) (L : Q) : Prop :=
  perpendicular l₁ L ∧ perpendicular l₂ L → (∃ x, x ∈ l₁ ∧ x ∈ l₂ → intersecting l₁ l₂)

def cond2 (π₁ π₂ : P) (π : Q) : Prop :=
  perpendicular π₁ π ∧ perpendicular π₂ π → (∃ x, x ∈ π₁ ∧ x ∈ π₂ ∨ parallel π₁ π₂)

-- The proof statement
theorem correct_conclusions {P Q : Type} [plane_space : P] [plane_space : Q]
  (h1 : ∀ l₁ l₂ (L : Q), cond1 l₁ l₂ L)
  (h2 : ∀ π₁ π₂ (π : Q), cond2 π₁ π₂ π)
  : (∀ l₁ l₂ (π : Q), conclusion2 l₁ l₂ π) ∧ (∀ π₁ π₂ (L : Q), conclusion3 π₁ π₂ L) :=
  by sorry

end correct_conclusions_l515_515062


namespace dandelion_average_l515_515061

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end dandelion_average_l515_515061


namespace mans_speed_in_still_water_l515_515018

-- Definitions
def speed_of_current_kmh : ℝ := 3
def time_taken_seconds : ℝ := 19.99840012798976
def distance_covered_meters : ℝ := 110

-- Convert speed of the current from km/h to m/s
def speed_of_current_ms : ℝ := (speed_of_current_kmh * 1000) / 3600

-- Downstream speed in m/s
def downstream_speed_ms : ℝ := distance_covered_meters / time_taken_seconds

-- Man's speed in still water in m/s
def speed_in_still_water_ms : ℝ := downstream_speed_ms - speed_of_current_ms

-- Man's speed in still water in km/h
def speed_in_still_water_kmh : ℝ := speed_in_still_water_ms * 3600 / 1000

-- Theorems
theorem mans_speed_in_still_water : 
  speed_in_still_water_kmh ≈ 16.80098924870248 := sorry

end mans_speed_in_still_water_l515_515018


namespace pyramid_total_area_l515_515729

theorem pyramid_total_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (base_edge_eq : base_edge = 8)
  (lateral_edge_eq : lateral_edge = 10)
  : 4 * (1 / 2 * base_edge * sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 32 * sqrt 21 := by
  sorry

end pyramid_total_area_l515_515729


namespace min_z_value_l515_515933

noncomputable def min_value (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) : ℝ :=
  2 * x + (sqrt 3) * y

theorem min_z_value (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) : ∃ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * π ∧ min_value x y h = -5 :=
sorry

end min_z_value_l515_515933


namespace overall_gain_or_loss_percentage_l515_515674

-- Define the sets
def SetA := {costs: list ℝ // costs.length = 50} × {sellingPrices: list ℝ // sellingPrices.length = 60}
def SetB := {costs: list ℝ // costs.length = 30} × {sellingPrices: list ℝ // sellingPrices.length = 40}
def SetC := {costs: list ℝ // costs.length = 20} × {sellingPrices: list ℝ // sellingPrices.length = 25}

-- Define total cost price and selling price conditions
def totalCostPrice_eq_totalSellingPrice (s : {costs: list ℝ} × {sellingPrices: list ℝ}) : Prop :=
  (list.sum s.1.costs = list.sum s.2.sellingPrices)

-- Define sets and conditions
def sets_and_conditions : Prop :=
  totalCostPrice_eq_totalSellingPrice (SetA.val.val) ∧
  totalCostPrice_eq_totalSellingPrice (SetB.val.val) ∧
  totalCostPrice_eq_totalSellingPrice (SetC.val.val)

-- Prove that the overall gain or loss percentage is 0%
theorem overall_gain_or_loss_percentage : sets_and_conditions → (∀ s, s = SetA ∨ s = SetB ∨ s = SetC → 
(list.sum s.1.costs = list.sum s.2.sellingPrices) ∧ 0 = 0) :=
by
  intros h s hs
  sorry

end overall_gain_or_loss_percentage_l515_515674


namespace stella_profit_l515_515313

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end stella_profit_l515_515313


namespace sum_of_distances_l515_515770

def point := ℝ × ℝ 

def distance (a b : point) : ℝ := real.sqrt (((a.1 - b.1)^2) + ((a.2 - b.2)^2))

def Q : point := (3, 1)
def D : point := (1, 1)
def E : point := (7, -2)
def F : point := (4, 5)

theorem sum_of_distances (p q : ℕ) (r : ℝ) :
  distance Q D + distance Q E + distance Q F = p + q * real.sqrt r ∧ p + q = 8 :=
by {
  sorry
}

end sum_of_distances_l515_515770


namespace children_count_l515_515498

theorem children_count (C : ℕ) 
    (cons : ℕ := 12)
    (total_cost : ℕ := 76)
    (child_ticket_cost : ℕ := 7)
    (adult_ticket_cost : ℕ := 10)
    (num_adults : ℕ := 5)
    (adult_cost := num_adults * adult_ticket_cost)
    (cost_with_concessions := total_cost - adult_cost )
    (children_cost := cost_with_concessions - cons):
    C = children_cost / child_ticket_cost :=
by
    sorry

end children_count_l515_515498


namespace solution_set_l515_515172

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- f is a differentiable function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f' is the derivative of f
variable (f' : ℝ → ℝ)

-- Assumptions
axiom h1 : ∀ x : ℝ, deriv f x = f' x
axiom h2 : ∀ x : ℝ, f' x < 2
axiom h3 : f 3 = 7

-- The theorem statement
theorem solution_set (x : ℝ) : f x < 2 * x + 1 ↔ x > 3 := by
  sorry

end solution_set_l515_515172


namespace max_subset_size_l515_515633

-- Define the set M
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1995 }

-- Define the property of subset A
def valid_subset (A : Set ℕ) : Prop :=
  ∀ x ∈ A, 15 * x ∉ A

-- Define the maximum possible size of such subset
theorem max_subset_size : ∃ A : Set ℕ, A ⊆ M ∧ valid_subset A ∧ Finset.card (A.to_finset) = 1870 :=
sorry

end max_subset_size_l515_515633


namespace liam_mia_walk_upwards_l515_515218

theorem liam_mia_walk_upwards :
  let liam : ℝ × ℝ := (10, -15)
  let mia : ℝ × ℝ := (5, 20)
  let zoe : ℝ × ℝ := (7.5, 10)
  let midpoint : ℝ × ℝ := ((liam.1 + mia.1) / 2, (liam.2 + mia.2) / 2)
  (midpoint.1 = zoe.1) → midpoint.2 = 2.5 → (zoe.2 - midpoint.2 = 7.5) :=
by
  intros
  have midpoint : midpoint = (7.5, 2.5) := by
    -- Here, we would show calculations for midpoint
    sorry
  rw [midpoint] at *
  have h1 : midpoint.2 = 2.5 := by
    -- Here, we would verify the y-component of midpoint
    sorry
  have h2 : zoo.2 - midpoint.2 = 7.5 := by
    -- Here, we would demonstrate the vertical distance
    sorry
  exact h2
      

end liam_mia_walk_upwards_l515_515218


namespace quadratic_solution_is_unique_l515_515859

theorem quadratic_solution_is_unique (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 2 * p + q / 2 = -p)
  (h2 : 2 * p * (q / 2) = q) :
  (p, q) = (1, -6) :=
by
  sorry

end quadratic_solution_is_unique_l515_515859


namespace dice_sides_prob_l515_515785

theorem dice_sides_prob (n : ℕ) (h : 1 - (1 - (1 / n))^3 = 0.111328125) : n = 26 :=
sorry

end dice_sides_prob_l515_515785


namespace minimum_adjacent_white_pairs_l515_515923

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l515_515923


namespace nuts_left_l515_515701

theorem nuts_left (total_nuts : ℕ) (fraction_eaten : ℚ) (remaining_nuts : ℕ) (h1 : total_nuts = 30) (h2 : fraction_eaten = 5/6) (h3 : remaining_nuts = total_nuts - (total_nuts * fraction_eaten).toNat) : remaining_nuts = 5 :=
by
  simp [h1, h2, h3]
  sorry

end nuts_left_l515_515701


namespace similar_isosceles_right_triangles_l515_515749

-- Define properties of an isosceles right triangle in Lean
structure IsoscelesRightTriangle (a b c : ℝ) :=
  (side_ratio : a / b = 1 / sqrt 2) -- The ratio of the sides opposite the equal angles
  (angles : c = π / 2 ∧ b = a)

theorem similar_isosceles_right_triangles (a1 b1 c1 a2 b2 c2 : ℝ) :
  IsoscelesRightTriangle a1 b1 c1 → IsoscelesRightTriangle a2 b2 c2 → (a1 / a2 = b1 / b2) ∧ (c1 = c2) :=
by
  sorry

end similar_isosceles_right_triangles_l515_515749


namespace pyramid_total_area_l515_515731

theorem pyramid_total_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (base_edge_eq : base_edge = 8)
  (lateral_edge_eq : lateral_edge = 10)
  : 4 * (1 / 2 * base_edge * sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 32 * sqrt 21 := by
  sorry

end pyramid_total_area_l515_515731


namespace distance_between_L1_and_L2_l515_515682

-- Define the first line
def L1 (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the second line
def L2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the distance between the lines
def distance_between_lines (A B : ℝ) (C1 C2 : ℝ) : ℝ := 
  |C1 - C2| / Real.sqrt (A^2 + B^2)

-- Main theorem statement
theorem distance_between_L1_and_L2 : distance_between_lines 1 2 3 (-3) = 6 * Real.sqrt 5 / 5 := by
  sorry

end distance_between_L1_and_L2_l515_515682


namespace pyramid_total_area_l515_515744

noncomputable def pyramid_base_edge := 8
noncomputable def pyramid_lateral_edge := 10

/-- The total area of the four triangular faces of a right, square-based pyramid
with base edges measuring 8 units and lateral edges measuring 10 units is 32 * sqrt(21). -/
theorem pyramid_total_area :
  let base_edge := pyramid_base_edge,
      lateral_edge := pyramid_lateral_edge,
      height := sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2),
      area_of_one_face := 1 / 2 * base_edge * height
  in 4 * area_of_one_face = 32 * sqrt 21 :=
sorry

end pyramid_total_area_l515_515744


namespace simplify_expression_l515_515310

theorem simplify_expression (x : ℝ) (h1 : sqrt (1 - sin x ^ 2) = abs (cos x)) (h2 : cos (3 * π / 5) < 0) :
  sqrt (1 - sin (3 * π / 5) ^ 2) = - cos (3 * π / 5) :=
by
  sorry

end simplify_expression_l515_515310


namespace train_cross_time_l515_515042

def L_t : ℕ := 230
def L_p1 : ℕ := 130
def L_p2 : ℕ := 250
def time_2 : ℕ := 20

theorem train_cross_time : 
  let D_2 := L_t + L_p2,
      v := D_2 / time_2,
      D_1 := L_t + L_p1,
      time_1 := D_1 / v
  in time_1 = 15 := 
by
  sorry

end train_cross_time_l515_515042


namespace differential_of_y_l515_515876

variable (x : ℝ) (dx : ℝ)

noncomputable def y := x * (Real.sin (Real.log x) - Real.cos (Real.log x))

theorem differential_of_y : (deriv y x * dx) = 2 * Real.sin (Real.log x) * dx := by
  sorry

end differential_of_y_l515_515876


namespace households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l515_515214

namespace VehicleHouseholds

-- Definitions for the conditions
def totalHouseholds : ℕ := 250
def householdsNoVehicles : ℕ := 25
def householdsAllVehicles : ℕ := 36
def householdsCarOnly : ℕ := 62
def householdsBikeOnly : ℕ := 45
def householdsScooterOnly : ℕ := 30

-- Proof Statements
theorem households_with_two_types_of_vehicles :
  (totalHouseholds - householdsNoVehicles - householdsAllVehicles - 
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly)) = 52 := by
  sorry

theorem households_with_exactly_one_type_of_vehicle :
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly) = 137 := by
  sorry

theorem households_with_at_least_one_type_of_vehicle :
  (totalHouseholds - householdsNoVehicles) = 225 := by
  sorry

end VehicleHouseholds

end households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l515_515214


namespace Carla_total_students_l515_515068

theorem Carla_total_students : 
  let students_in_restroom := 2
  let students_absent := 3 * students_in_restroom - 1
  let rows := 4
  let desks_per_row := 6
  let desks_total := rows * desks_per_row
  let desks_filled := 2 * desks_total / 3
  students_filled := desks_filled in
  students_in_restroom + students_absent + students_filled = 23 := 
by
  sorry

end Carla_total_students_l515_515068


namespace distribution_problem_l515_515867

open Finset
open Fintype

noncomputable def num_distribution_methods : Nat :=
let students : Finset Nat := {1, 2, 3, 4, 5}
let universities : Finset Nat := {1, 2, 3}
let C n k : Nat := (univ.product (univ.filter (λ s: Finset Nat, s.card = k))).card / k!
let A n k : Nat := n! / (n - k)!
(C 5 2) * (C 3 2) * (A 3 3) / 2 + (C 5 3) * (C 2 1) * (A 3 3) / 2

theorem distribution_problem : num_distribution_methods = 150 := 
by
  -- placeholder for the proof
  sorry

end distribution_problem_l515_515867


namespace towel_bleach_area_decrease_l515_515428

theorem towel_bleach_area_decrease :
  ∀ (L B : ℝ), 
  let original_area := L * B in
  let new_area := (0.70 * L) * (0.60 * B) in
  (original_area - new_area) / original_area * 100 = 58 := 
by 
  intros L B
  let original_area := L * B 
  let new_area := (0.70 * L) * (0.60 * B)
  have : (original_area - new_area) / original_area * 100 = 58 := sorry
  exact this

end towel_bleach_area_decrease_l515_515428


namespace initial_spiders_correct_l515_515787

-- Define the initial number of each type of animal
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5

-- Conditions about the changes in the number of animals
def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

-- Number of animals left in the store
def total_animals_left : Nat := 25

-- Define the remaining animals after sales and adoptions
def remaining_birds : Nat := initial_birds - birds_sold
def remaining_puppies : Nat := initial_puppies - puppies_adopted
def remaining_cats : Nat := initial_cats

-- Define the remaining animals excluding spiders
def animals_without_spiders : Nat := remaining_birds + remaining_puppies + remaining_cats

-- Define the number of remaining spiders
def remaining_spiders : Nat := total_animals_left - animals_without_spiders

-- Prove the initial number of spiders
def initial_spiders : Nat := remaining_spiders + spiders_loose

theorem initial_spiders_correct :
  initial_spiders = 15 := by 
  sorry

end initial_spiders_correct_l515_515787


namespace relationship_between_cardinals_and_sparrows_l515_515838

theorem relationship_between_cardinals_and_sparrows:
  (C R B S : ℕ) (h₁ : C = 3) (h₂ : R = 4 * C) (h₃ : B = 2 * C) (h₄ : C + R + B + S = 31) :
  S = 10 ∧ (S = 10 * C / 3) :=
by {
  sorry,
}

end relationship_between_cardinals_and_sparrows_l515_515838


namespace rhombus_area_is_correct_l515_515370

def side_length (ABCD: Type) [has_sides ABCD] : ℝ := 4
def angle_DAB (ABCD: Type) [has_angles ABCD] : ℝ := 45

theorem rhombus_area_is_correct (ABCD: Type) [is_rhombus ABCD]
  (h1: side_length ABCD = 4)
  (h2: angle_DAB ABCD = 45) : rhombus_area ABCD = 8 * Real.sqrt 2 := 
by
  sorry

end rhombus_area_is_correct_l515_515370


namespace volume_truncated_pyramid_l515_515350
noncomputable def volume_of_truncated_pyramid (V : ℝ) (α : ℝ) : ℝ :=
  V / (8 * (cos (α / 2))^6)

theorem volume_truncated_pyramid
  (V : ℝ) (α : ℝ) : 
  ∃ V₁, V₁ = volume_of_truncated_pyramid V α := 
by
  let V₁ := volume_of_truncated_pyramid V α
  exact ⟨V₁, rfl⟩

end volume_truncated_pyramid_l515_515350


namespace integer_sum_satisfying_inequality_l515_515672

theorem integer_sum_satisfying_inequality : 
  let f := λ (x : ℝ), (2 + Real.sqrt 3) ^ x + 2 < 3 * (Real.sqrt (2 - Real.sqrt 3)) ^ (2 * x)
  let range := Set.Ico (-20 : ℝ) 53
  let sum_int_values := (Finset.filter (λ x : ℤ, f x) (Finset.Ico (-20) 53)).sum (λ x, x)
  sum_int_values = -190 :=
by
  sorry

end integer_sum_satisfying_inequality_l515_515672


namespace angle_B_in_range_l515_515226

theorem angle_B_in_range (A B C : ℝ) (h_nonright : A + B + C = π) (h_geometric : ∃ r : ℝ, tan A = r ∧ tan B = r^2 ∧ tan C = r^3) 
  : B ∈ set.Ico (π/3) (π/2) :=
sorry

end angle_B_in_range_l515_515226


namespace smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l515_515535

noncomputable def f (x : ℝ) : ℝ := 4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - sqrt 3

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = π :=
sorry

theorem intervals_where_f_is_monotonically_increasing :
  ∃ (k : ℤ), ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → f' x > 0 :=
sorry

end smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l515_515535


namespace line_passing_point_has_b_value_l515_515413

theorem line_passing_point_has_b_value (b : ℝ) :
  ∃ b : ℝ, (b = ( - 3 + Real.sqrt 33) / 2 ∨ b = ( - 3 - Real.sqrt 33) / 2) ∧
  (∃ x y : ℝ, x = 2 ∧ y = -5 ∧ b * x + (b - 1) * y = b ^ 2 - 1) :=
by
  use ( - 3 + Real.sqrt 33) / 2
  split
  · left
    rfl
  · use 2, -5
    exact ⟨rfl, rfl, by sorry⟩

end line_passing_point_has_b_value_l515_515413


namespace num_turtles_on_sand_l515_515038

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l515_515038


namespace total_profit_from_selling_30_necklaces_l515_515710

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l515_515710


namespace sum_sin_squares_l515_515420

theorem sum_sin_squares (θ : Fin 6 → ℝ) (h1 : ∑ i : Fin 3, cos (θ i) ^ 2 = 1) 
: ∑ i, sin (θ i) ^ 2 = 4 := by
  sorry

end sum_sin_squares_l515_515420


namespace number_of_correct_conclusions_l515_515327

-- Lean 4 statement for the proof problem
theorem number_of_correct_conclusions 
  (P F S : ℝ) 
  (h1 : P = F / S)
  (c1_correct : ∀ F_constant, (P = F_constant / S) → ∀ S1 S2, S1 ≠ S2 → P(S1) ≠ P(S2))
  (c2_incorrect : ∃ S_constant, ∀ (F1 F2 : ℝ), P = F1 / S_constant → P ≠ F2 / S_constant)
  (c3_incorrect : ∃ P_constant, ∀ (S1 S2 : ℝ), P_constant = F / S1 → P_constant = F / S2 → S1 ≠ S2 → F(S1) ≠ F(S2))
  (c4_partially_correct : ∃ S_constant, (P = F / S_constant) ∧ (S_constant > 0) ∧ (P > 0) ∧ ÿ(F Q P, F > 0)): 
  (c1_correct ∧ ¬c2_incorrect ∧ ¬c3_incorrect ∧ ¬c4_partially_correct) → (number_of_correct_conclusions = 1) :=
begin
  sorry
end

end number_of_correct_conclusions_l515_515327


namespace complex_sum_identity_l515_515982

def B : ℂ := 3 - 2 * complex.I
def Q : ℂ := -5 + 3 * complex.I
def R : ℂ := 2 * complex.I
def T : ℂ := -1 + 2 * complex.I

theorem complex_sum_identity : B - Q + R + T = 7 - complex.I := by
  sorry

end complex_sum_identity_l515_515982


namespace cupcake_ratio_l515_515304

theorem cupcake_ratio (C B : ℕ) (hC : C = 4) (hTotal : C + B = 12) : B / C = 2 :=
by
  sorry

end cupcake_ratio_l515_515304


namespace repeating_decimal_division_l515_515373

theorem repeating_decimal_division:
  let x := (54 / 99 : ℚ)
  let y := (18 / 99 : ℚ)
  (x / y) * (1 / 2) = (3 / 2 : ℚ) := by
    sorry

end repeating_decimal_division_l515_515373


namespace longest_side_of_triangle_l515_515810

open Real

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem longest_side_of_triangle :
  let A := (3, 3)
  let B := (7, 8)
  let C := (8, 3)
  max (distance A B) (max (distance A C) (distance B C)) = sqrt 41 :=
by
  sorry

end longest_side_of_triangle_l515_515810


namespace cylinder_lateral_area_l515_515677

-- Define the radius and height as constants
def radius : ℝ := 2
def height : ℝ := 2

-- Define the formula for the lateral surface area of a cylinder
def lateral_surface_area (r h : ℝ) : ℝ := 2 * real.pi * r * h

-- State the theorem we want to prove
theorem cylinder_lateral_area : 
  lateral_surface_area radius height = 8 * real.pi :=
by
  sorry

end cylinder_lateral_area_l515_515677


namespace solve_for_x_l515_515126

theorem solve_for_x :
  ∃ x : ℕ, (12 ^ 3) * (6 ^ x) / 432 = 144 ∧ x = 2 := by
  sorry

end solve_for_x_l515_515126


namespace range_of_y_l515_515086

noncomputable def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_y :
  (∀ x : ℝ, operation (x - y) (x + y) < 1) ↔ - (1 : ℝ) / 2 < y ∧ y < (3 : ℝ) / 2 :=
by
  sorry

end range_of_y_l515_515086


namespace bounded_sequence_l515_515800

def sequence (a : ℕ → ℝ) := 
  ∀ n : ℕ, n > 0 →
    a n >
    (∑ i in finset.range n.succ, a (n + i + 1)) / (n + 2016)

theorem bounded_sequence (a : ℕ → ℝ) (h : sequence a) :
  ∃ C : ℝ, ∀ n : ℕ, n > 0 → a n < C :=
sorry

end bounded_sequence_l515_515800


namespace erin_trolls_count_l515_515094

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l515_515094


namespace combined_weight_is_correct_l515_515606

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end combined_weight_is_correct_l515_515606


namespace probability_of_double_tile_is_one_fourth_l515_515035

noncomputable def probability_double_tile : ℚ :=
  let total_pairs := (7 * 7) / 2
  let double_pairs := 7
  double_pairs / total_pairs

theorem probability_of_double_tile_is_one_fourth :
  probability_double_tile = 1 / 4 :=
by
  sorry

end probability_of_double_tile_is_one_fourth_l515_515035


namespace timeToCrossStationaryTrain_l515_515043

-- Definitions based on conditions
def trainPassingPoleTime (train_speed_m_s length_train_m : ℝ → ℝ) : Prop :=
forall (t1 t2:ℝ), 108 * (1000 / 3600) * (t2 - t1) = length_train_m

def lengthOfMovingTrain (train_speed_m_s t : ℝ → ℝ) : Prop :=
forall (t1 t2:ℝ), trainPassingPoleTime 108 600 → (train_speed_m_s = 30) →
length_train_m = train_speed_m_s * (t2 - t1)

noncomputable def total_cross_length : ℝ := 300 + 600

theorem timeToCrossStationaryTrain :
  trainPassingPoleTime 108 600 →
  lengthOfMovingTrain 30 10 →
  total_cross_length / 30 = 30 := by
  sorry

end timeToCrossStationaryTrain_l515_515043


namespace calc_expression_value_l515_515835

open Real

theorem calc_expression_value :
  sqrt ((16: ℝ) ^ 12 + (8: ℝ) ^ 15) / ((16: ℝ) ^ 5 + (8: ℝ) ^ 16) = (3 * sqrt 2) / 4 := sorry

end calc_expression_value_l515_515835


namespace integer_sum_satisfying_inequality_l515_515671

theorem integer_sum_satisfying_inequality : 
  let f := λ (x : ℝ), (2 + Real.sqrt 3) ^ x + 2 < 3 * (Real.sqrt (2 - Real.sqrt 3)) ^ (2 * x)
  let range := Set.Ico (-20 : ℝ) 53
  let sum_int_values := (Finset.filter (λ x : ℤ, f x) (Finset.Ico (-20) 53)).sum (λ x, x)
  sum_int_values = -190 :=
by
  sorry

end integer_sum_satisfying_inequality_l515_515671


namespace log_product_computation_l515_515851

theorem log_product_computation : 
  (Real.log 32 / Real.log 2) * (Real.log 27 / Real.log 3) = 15 := 
by
  -- The proof content, which will be skipped with 'sorry'.
  sorry

end log_product_computation_l515_515851


namespace average_variance_add_mean_l515_515988

variables {a b c d e f : ℝ} 

def mean (x y z w v u : ℝ) : ℝ := (x + y + z + w + v + u) / 6

theorem average_variance_add_mean (x : ℝ) (h_avg : mean a b c d e f = x)
  (h_var : (a - x)^2 + (b - x)^2 + (c - x)^2 + (d - x)^2 + (e - x)^2 + (f - x)^2 = 1.2) :
  ((a - x)^2 + (b - x)^2 + (c - x)^2 + (d - x)^2 + (e - x)^2 + (f - x)^2 + (x - x)^2) / 7 = 6 / 35 :=
by 
  rw [h_avg, h_var]
  norm_num
  sorry

end average_variance_add_mean_l515_515988


namespace max_value_of_quadratic_at_2_l515_515379

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

theorem max_value_of_quadratic_at_2 : ∃ (x : ℝ), x = 2 ∧ ∀ y : ℝ, f y ≤ f x :=
by
  use 2
  sorry

end max_value_of_quadratic_at_2_l515_515379


namespace perimeter_triangle_distinct_9_units_l515_515553

open Nat

theorem perimeter_triangle_distinct_9_units :
  {t : Finset (Finset ℕ) | ∃ a b c : ℕ, a + b + c = 9 ∧
                        a + b > c ∧ b + c > a ∧ c + a > b ∧
                        {a, b, c} = t}.card = 2 := 
by 
  sorry

end perimeter_triangle_distinct_9_units_l515_515553


namespace volunteers_meet_again_in_360_days_l515_515664

theorem volunteers_meet_again_in_360_days :
  let Sasha := 5
  let Leo := 8
  let Uma := 9
  let Kim := 10
  Nat.lcm Sasha (Nat.lcm Leo (Nat.lcm Uma Kim)) = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l515_515664


namespace laptop_sticker_price_l515_515552

theorem laptop_sticker_price (x : ℝ)
  (h1 : 0.80 * x - 120 = 0.70 * x + 5)
  : x = 1250 :=
sorry

end laptop_sticker_price_l515_515552


namespace least_odd_prime_factor_2023_6_plus_1_l515_515483

theorem least_odd_prime_factor_2023_6_plus_1 : ∃ p : ℕ, prime p ∧ odd p ∧ p = 37 ∧ p ∣ (2023 ^ 6 + 1) ∧ ∀ q, prime q ∧ odd q ∧ q ∣ (2023 ^ 6 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_l515_515483


namespace unemployment_rate_next_year_l515_515045

theorem unemployment_rate_next_year (x : ℝ)
  (initial_unemployment_rate : ℝ) (population_growth_rate : ℝ)
  (unemployment_decrease_rate : ℝ):
  initial_unemployment_rate = 0.056 ∧
  population_growth_rate = 0.04 ∧
  unemployment_decrease_rate = 0.09 →
  ((0.91 * (initial_unemployment_rate * x)) / (1 + population_growth_rate) = 0.049) := 
by
  intro h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry

end unemployment_rate_next_year_l515_515045


namespace exists_third_degree_poly_with_positive_and_negative_roots_l515_515238

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l515_515238


namespace sin_double_angle_l515_515935

theorem sin_double_angle (θ : ℝ) (h : sin θ + cos θ = 1 / 3) : sin (2 * θ) = -8 / 9 :=
by
  sorry

end sin_double_angle_l515_515935


namespace min_area_triangle_OBX_l515_515393

theorem min_area_triangle_OBX :
  ∃ (x y : ℕ), (0 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 8) ∧ 
  (∀ x' y' : ℕ, (0 ≤ x' ∧ x' ≤ 9) ∧ (0 ≤ y' ∧ y' ≤ 8) → abs (9 * y - 8 * x) ≤ abs (9 * y' - 8 * x')) ∧
  ∃ x y : ℕ, (∃ x y : ℕ, (0 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 8)) ∧ 
  1/2 = abs (9 * y - 8 * x) / 2
  :=
sorry

end min_area_triangle_OBX_l515_515393


namespace probability_odd_heads_after_60_flips_l515_515437

theorem probability_odd_heads_after_60_flips :
  (∑ (n : Fin 61) in {x | (x % 2 = 1)}, 
     (60.choose n) * (3 / 4)^(n : ℕ) * (1 / 4)^(60 - n) )
  = 1 / 2 * (1 - 1 / 2^60) := 
  sorry

end probability_odd_heads_after_60_flips_l515_515437


namespace ArletteAge_l515_515291

/-- Define the ages of Omi, Kimiko, and Arlette -/
def OmiAge (K : ℕ) : ℕ := 2 * K
def KimikoAge : ℕ := 28   /- K = 28 -/
def averageAge (O K A : ℕ) : Prop := (O + K + A) / 3 = 35

/-- Prove Arlette's age given the conditions -/
theorem ArletteAge (A : ℕ) (h1 : A + OmiAge KimikoAge + KimikoAge = 3 * 35) : A = 21 := by
  /- Hypothesis h1 unpacks the third condition into equality involving O, K, and A -/
  sorry

end ArletteAge_l515_515291


namespace polyhedron_edges_l515_515803

theorem polyhedron_edges (F V E : ℕ) (h1 : F = 12) (h2 : V = 20) (h3 : F + V = E + 2) : E = 30 :=
by
  -- Additional details would go here, proof omitted as instructed.
  sorry

end polyhedron_edges_l515_515803


namespace find_a_l515_515534

theorem find_a (a : ℝ) (ha : a ≠ 0)
  (h_area : (1/2) * (a/2) * a^2 = 2) :
  a = 2 ∨ a = -2 :=
sorry

end find_a_l515_515534


namespace min_abs_val_l515_515627

noncomputable theory

def g (x : ℝ) : ℝ := x^4 + 16*x^3 + 80*x^2 + 128*x + 64

variables {z1 z2 z3 z4 : ℝ}

-- Given that z1, z2, z3, z4 are the roots of polynomial g
def roots_of_g : Prop :=
  z1^4 + 16*z1^3 + 80*z1^2 + 128*z1 + 64 = 0 ∧
  z2^4 + 16*z2^3 + 80*z2^2 + 128*z2 + 64 = 0 ∧
  z3^4 + 16*z3^3 + 80*z3^2 + 128*z3 + 64 = 0 ∧
  z4^4 + 16*z4^3 + 80*z4^2 + 128*z4 + 64 = 0

theorem min_abs_val (h : roots_of_g) : 
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  {a, b, c, d} = {1, 2, 3, 4} ∧ 
  | (nth [z1, z2, z3, z4] (a - 1)) * (nth [z1, z2, z3, z4] (b - 1)) + 
    (nth [z1, z2, z3, z4] (c - 1)) * (nth [z1, z2, z3, z4] (d - 1)) | = 16) := sorry

end min_abs_val_l515_515627


namespace floor_sqrt_product_l515_515850

theorem floor_sqrt_product :
  (∏ i in Finset.range 25, Int.floor (Real.sqrt (2 * i + 1))) / 
  (∏ i in Finset.Ico 1 26, Int.floor (Real.sqrt (2 * i))) = 48 / 105 :=
by
  sorry

end floor_sqrt_product_l515_515850


namespace area_triangle_ACE_l515_515790

-- Define the hexagon and triangle structure
structure RegularHexagon :=
  (sides : ℝ)
  (regular : ∀ (x y : ℝ), x = sides ∧ y = sides)

-- Define the conditions
def hexABCDEF : RegularHexagon := 
  { sides := 3, regular := by simp }

-- Define the problem to prove the area of $\triangle ACE$
theorem area_triangle_ACE (h : RegularHexagon) (h_sides : h.sides = 3) :
  area h.regular ACE = (9 * Real.sqrt 3) / 4 :=
sorry

end area_triangle_ACE_l515_515790


namespace pyramid_min_volume_l515_515027

noncomputable def pyramid_volume {V r α : ℝ} 
  (hV : 4 / 3 * π * r ^ 3 = V) : ℝ := 
  let a := r * (3 * (Math.tan (α / 2)) / sqrt 3) in
  let S_ABC := (sqrt 3 / 4) * a ^ 2 in
  let DM := 2 * r / sqrt (1 - Math.tan(α / 2) ^ 2) in
  (1 / 3) * S_ABC * DM

theorem pyramid_min_volume {V r α : ℝ} 
  (hV : 4 / 3 * π * r ^ 3 = V) :
  pyramid_volume hV = (6 * V * sqrt 3) / π :=
sorry

end pyramid_min_volume_l515_515027


namespace number_of_arrangements_l515_515309

theorem number_of_arrangements : 
  let letters := ['e', 'q', 'u', 'a', 't', 'i', 'o', 'n']
  let remaining_letters := ['e', 'a', 't', 'i', 'o', 'n']
  let qu := ('q', 'u')
  let num_ways_choose := Nat.choose 6 3
  let num_ways_arrange := 4!
  in (num_ways_choose * num_ways_arrange = 480) :=
by 
  sorry

end number_of_arrangements_l515_515309


namespace log2_denominator_eq_37_l515_515675

noncomputable def tournament_probability : ℤ := 
  let total_games := 45
  let factorial_10 := Nat.factorial 10
  let total_outcomes := 2 ^ total_games
  let probability_fraction := factorial_10 / total_outcomes
  let denominator := Nat.gcdA factorial_10 total_outcomes
  denominator -- this will calculate n
#eval numerator.gcd_2_power_exponent

theorem log2_denominator_eq_37 : tournament_probability = 2^37 := by
  sorry

end log2_denominator_eq_37_l515_515675


namespace probability_yellow_side_l515_515010

theorem probability_yellow_side :
  let num_cards := 8
  let num_blue_blue := 4
  let num_blue_yellow := 2
  let num_yellow_yellow := 2
  let total_yellow_sides := (2 * num_yellow_yellow + 1 * num_blue_yellow)
  let yellow_yellow_sides := (2 * num_yellow_yellow)
  (total_yellow_sides = 6) →
  (yellow_yellow_sides = 4) →
  (yellow_yellow_sides / total_yellow_sides = (2 : ℚ) / 3) := 
by 
  intros
  sorry

end probability_yellow_side_l515_515010


namespace ratio_of_areas_l515_515825

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l515_515825


namespace total_trolls_l515_515098

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l515_515098


namespace find_a_in_polynomial_l515_515596

theorem find_a_in_polynomial {a : ℝ} (h : (polynomial.coeff ((X - polynomial.C a) ^ 10) 7 = 15)) :
  a = -1 / 2 :=
sorry

end find_a_in_polynomial_l515_515596


namespace find_middle_number_l515_515347

theorem find_middle_number (x y z : ℤ) (h1 : x + y = 22) (h2 : x + z = 29) (h3 : y + z = 37) (h4 : x < y) (h5 : y < z) : y = 15 :=
by
  sorry

end find_middle_number_l515_515347


namespace min_adj_white_pairs_l515_515921

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l515_515921


namespace num_positive_chains_is_4951_l515_515434

-- Define the conditions specified in the problem
def circle_integers (n : ℕ) : Type := { l : List ℤ // l.length = n }

-- Define what it means for a list to be a chain
def chain (l : List ℤ) (start length : ℕ) : List ℤ :=
  let xs := List.cycle l
  List.take length (List.drop start xs)

-- Define the sum of a chain
def sum_of_chain (l : List ℤ) (start length : ℕ) : ℤ :=
  (chain l start length).sum

-- Given the number of integers (100) and their circular nature with total sum = 1
noncomputable def num_positive_sum_chains (ints : circle_integers 100) : ℕ :=
  let n := 100
  let total_sum := ints.val.sum
  let chain_count := (n * (n + 1)) / 2
  let all_chains := List.bind (List.range n) (λ start, List.map (λ length, (start, length)) (List.range (n + 1)))
  all_chains.count (λ (sl : ℕ × ℕ), sum_of_chain ints.val sl.fst sl.snd > 0)

-- The theorem to prove:
theorem num_positive_chains_is_4951 (ints : circle_integers 100) (h_sum : ints.val.sum = 1) :
  num_positive_sum_chains ints = 4951 :=
sorry -- Proof is omitted.

end num_positive_chains_is_4951_l515_515434


namespace problem_solution_l515_515183

def f (x : ℝ) : ℝ :=
  x^(1/3) + Real.logb (1/3) x

theorem problem_solution : f 27 = 0 :=
by sorry

end problem_solution_l515_515183


namespace midpoints_coplanar_inscribed_sphere_center_l515_515499

theorem midpoints_coplanar_inscribed_sphere_center
  (A B C D : Point)
  (H1 : face_area (A, B, C) + face_area (A, B, D) = face_area (C, D, A) + face_area (C, D, B)) :
  let K := midpoint A C,
      L := midpoint B C,
      M := midpoint B D,
      N := midpoint A D
  in coplanar {K, L, M, N} ∧ incircle_center A B C D ∈ plane K L M := 
sorry

end midpoints_coplanar_inscribed_sphere_center_l515_515499


namespace find_number_l515_515006

theorem find_number (N : ℝ) (h1 : (4/5) * (3/8) * N = some_number)
                    (h2 : 2.5 * N = 199.99999999999997) :
  N = 79.99999999999999 := 
sorry

end find_number_l515_515006


namespace max_tetrominoes_in_6x6_grid_l515_515376
open Mathlib

-- Define the 6x6 grid
def grid_6x6 := (Fin 6 × Fin 6)

-- Define a tetromino as a set of 4 positions in the grid
def Tetromino := Finset grid_6x6

-- Define the conditions of the problem
def is_tetromino (s : Tetromino) : Prop :=
  s.card = 4

-- Define the main theorem
theorem max_tetrominoes_in_6x6_grid : ∃ (ts : Finset Tetromino), ts.card = 8 ∧ 
(∀ t ∈ ts, is_tetromino t) ∧ 
(two_set_disjoint : ∀ (t1 t2 : Tetromino), t1 ≠ t2 → Disjoint t1 t2) ∧
(to_cover_set : (⋃₀ (ts : Set Tetromino)) = grid_6x6) := 
sorry

end max_tetrominoes_in_6x6_grid_l515_515376


namespace arithmetic_sequence_sum_l515_515936

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum_l515_515936


namespace sum_of_all_four_digit_numbers_using_1_to_5_l515_515493

open BigOperators

theorem sum_of_all_four_digit_numbers_using_1_to_5 : ∑ n in { n : ℕ | n >= 1000 ∧ n < 10000 ∧ ∀ d ∈ nat.digits 10 n, d ∈ {1, 2, 3, 4, 5} ∧ (λ x, list.count x (nat.digits 10 n)) d = 1 }, n = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_using_1_to_5_l515_515493


namespace pyramid_total_area_l515_515732

theorem pyramid_total_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (base_edge_eq : base_edge = 8)
  (lateral_edge_eq : lateral_edge = 10)
  : 4 * (1 / 2 * base_edge * sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 32 * sqrt 21 := by
  sorry

end pyramid_total_area_l515_515732


namespace sum_of_reciprocals_l515_515629

open Real

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x + y = 5 * x * y) (hx2y : x = 2 * y) : 
  (1 / x) + (1 / y) = 5 := 
  sorry

end sum_of_reciprocals_l515_515629


namespace range_of_a_l515_515205

theorem range_of_a (a : ℝ) : ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l515_515205


namespace norm_squared_sum_l515_515254

open Real EuclideanSpace
noncomputable theory

variable (a b : ℝ^3)
def m : ℝ^3 := (4, -2, 3)

axiom midpoint_condition : 2 * m = a + b
axiom dot_product_condition : ⟪a, b⟫ = 10

theorem norm_squared_sum : ‖a‖^2 + ‖b‖^2 = 96 := by
  sorry

end norm_squared_sum_l515_515254


namespace sum_a_for_exactly_one_solution_l515_515891

theorem sum_a_for_exactly_one_solution :
  (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 + (a + 6) * x + 7 = 0) →
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  sorry

end sum_a_for_exactly_one_solution_l515_515891


namespace exists_monic_polynomial_with_root_l515_515105

noncomputable def monic_polynomial_with_root (x : ℝ) : Polynomial ℚ :=
  have H_root : x = Real.sqrt 3 + Real.sqrt 5 := by 
  sorry
  have H_polynomial : Polynomial ℚ := Polynomial.Coeff (x^4 - 16*x^2 + 4) 
  have H_monic : _root_.Polynomial := by
  sorry
  H_polynomial

theorem exists_monic_polynomial_with_root :
  ∃ P : Polynomial ℚ, 
  (P.degree = 4) ∧ (P.isMonic) ∧ (P.eval (Real.sqrt 3 + Real.sqrt 5) = 0) :=
begin
  use Polynomial.Coeff ((x ^ 4) - (16 * x ^ 2) + 4),
  simp,
  split
  sorry
  sorry
  sorry
end

end exists_monic_polynomial_with_root_l515_515105


namespace complex_root_of_unity_product_l515_515567

theorem complex_root_of_unity_product :
  ∀ (ω : ℂ), (ω^3 = 1 ∧ 1 + ω + ω^2 = 0) → (1 - ω + ω^2) * (1 + ω - ω^2) = 4 :=
begin
  intros ω h,
  sorry
end

end complex_root_of_unity_product_l515_515567


namespace not_possible_to_form_triangle_l515_515417

-- Define the conditions
variables (a : ℝ)

-- State the problem in Lean 4
theorem not_possible_to_form_triangle (h : a > 0) :
  ¬ (a + a > 2 * a ∧ a + 2 * a > a ∧ a + 2 * a > a) :=
by
  sorry

end not_possible_to_form_triangle_l515_515417


namespace find_R_value_l515_515270

noncomputable def x (Q : ℝ) : ℝ := Real.sqrt (Q / 2 + Real.sqrt (Q / 2))
noncomputable def y (Q : ℝ) : ℝ := Real.sqrt (Q / 2 - Real.sqrt (Q / 2))
noncomputable def R (Q : ℝ) : ℝ := (x Q)^6 + (y Q)^6 / 40

theorem find_R_value (Q : ℝ) : R Q = 10 :=
sorry

end find_R_value_l515_515270


namespace function_decreasing_on_interval_l515_515299

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem function_decreasing_on_interval : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f x₁ ≥ f x₂ := by
  sorry

end function_decreasing_on_interval_l515_515299


namespace move_hole_to_any_corner_l515_515992

theorem move_hole_to_any_corner (m n : ℕ) (h_m_odd : odd m) (h_n_odd : odd n) :
  ∀ (start_corner end_corner : (Fin 2 × Fin 2)),
    possible_to_move_hole (m, n) start_corner end_corner := 
sorry

end move_hole_to_any_corner_l515_515992


namespace four_digit_number_sum_is_correct_l515_515488

def valid_digits : Finset ℕ := {1, 2, 3, 4, 5}

def valid_four_digit_numbers : Finset (List ℕ) := 
  valid_digits.powerset.filter (λ s, s.card = 4) 
  >>= λ s, (s.val.to_list.permutations)

def four_digit_number_sum (numbers : Finset (List ℕ)) : ℕ :=
  numbers.sum (λ l, 
    match l with
    | [a, b, c, d] => 1000 * a + 100 * b + 10 * c + d
    | _           => 0
    end)

theorem four_digit_number_sum_is_correct :
  four_digit_number_sum valid_four_digit_numbers = 399960 :=
by
  sorry

end four_digit_number_sum_is_correct_l515_515488


namespace three_heads_with_tail_probability_correct_l515_515084

noncomputable def prob_three_heads_with_at_least_one_tail : ℚ :=
  5 / 64

theorem three_heads_with_tail_probability_correct (P : ℚ) : 
  (∃ F : ℕ → Prop, 
    (∀ n, F n → (coin_flip_sequence n = 'H')) ∧
    (∃ k, k < n ∧ (coin_flip_sequence k = 'T'))) →
  P = prob_three_heads_with_at_least_one_tail :=
by simp [prob_three_heads_with_at_least_one_tail]; sorry

end three_heads_with_tail_probability_correct_l515_515084


namespace tangent_line_sum_l515_515204

theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, (e^(x₀ - 1) = 1) ∧ (x₀ + a = e^(x₀-1) * (1 - x₀) - b + 1)) → a + b = 1 :=
by
  sorry

end tangent_line_sum_l515_515204


namespace factorization_min_value_expr_l515_515303

-- Problem 1: Factorization
theorem factorization (x y : ℝ) :
    1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
by sorry

-- Problem 2: Minimum value for given expression
theorem min_value_expr :
  let n := 1 in
  ((n^2 - 2*n - 3) * (n^2 - 2*n + 5) + 17) = 1 :=
by sorry

end factorization_min_value_expr_l515_515303


namespace sum_irrational_mult_not_zero_l515_515500

theorem sum_irrational_mult_not_zero 
  (m : ℕ)
  (a : Fin m → ℕ)
  (b : Fin m → ℤ)
  (h_distinct: ∀ i j : Fin m, i ≠ j → a i ≠ a j)
  (h_no_square_div: ∀ i : Fin m, ∀ p : ℕ, p > 1 → p^2 ∣ a i → False)
  (h_nonzero: ∀ i : Fin m, b i ≠ 0) : 
  (Finset.univ.sum (λ i : Fin m, real.sqrt (a i) * b i) ≠ 0) :=
by
  sorry

end sum_irrational_mult_not_zero_l515_515500


namespace parameter_values_for_distinct_solutions_l515_515474

theorem parameter_values_for_distinct_solutions (a : ℝ) :
  (∀ t : ℝ,
    t ∈ Ioo (-π / 2) 0 →
    (4 * a * cos t ^ 2 + 4 * a * (2 * real.sqrt 2 - 1) * cos t + 
    4 * (a - 1) * sin t + a + 2) / (sin t + 2 * real.sqrt 2 * cos t) = 4 * a) ↔ 
  a ∈ set.Ioo (-∞) (-18 - 24 * real.sqrt 2) ∪ set.Ioo (-18 - 24 * real.sqrt 2) (-6) :=
by sorry

end parameter_values_for_distinct_solutions_l515_515474


namespace turtles_still_on_sand_l515_515039

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l515_515039


namespace particle_velocity_at_3_l515_515786

noncomputable def s (t : ℝ) : ℝ := real.sqrt (t + 1)

theorem particle_velocity_at_3 : 
  (deriv s 3) = (1 / 4) :=
by
  sorry

end particle_velocity_at_3_l515_515786


namespace cakes_bought_l515_515832

theorem cakes_bought (initial : ℕ) (left : ℕ) (bought : ℕ) :
  initial = 169 → left = 32 → bought = initial - left → bought = 137 :=
by
  intros h_initial h_left h_bought
  rw [h_initial, h_left] at h_bought
  exact h_bought

end cakes_bought_l515_515832


namespace rank_32_boxers_in_15_days_l515_515586

theorem rank_32_boxers_in_15_days :
  ∀ (boxers : Fin 32 → ℕ),
    (∀ i j, i < j → boxers i ≠ boxers j) → -- all boxers have distinct strength levels
    (∀ i j, i < j → boxers i < boxers j) → -- the strongest boxer always wins
    (∃ (days : ℕ), days ≤ 15 ∧ (ordered : Fin 32 → Fin 32) → -- within 15 days
      ∀ i j, i < j → boxers (ordered i) < boxers (ordered j)) -- ranking is determined
:= 
sorry

end rank_32_boxers_in_15_days_l515_515586


namespace main_theorem_l515_515253

-- Definition of a closed set
def isClosedSet (S : Set ℝ) : Prop :=
  (∀ x y ∈ S, (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S)

-- Proposition 1: The set of real numbers is a closed set
def Prop1 : Prop :=
  isClosedSet Set.univ

-- Proposition 4: If S is a closed set, then 0 ∈ S
def Prop4 (S : Set ℝ) (h : isClosedSet S) : Prop :=
  0 ∈ S

-- The theorem we aim to prove is that Prop1 and Prop4 are true
theorem main_theorem : Prop1 ∧ ∀ S (h : isClosedSet S), Prop4 S h := by
  sorry

end main_theorem_l515_515253


namespace cost_500_pencils_is_25_dollars_l515_515679

def cost_of_500_pencils (cost_per_pencil : ℕ) (pencils : ℕ) (cents_per_dollar : ℕ) : ℕ :=
  (cost_per_pencil * pencils) / cents_per_dollar

theorem cost_500_pencils_is_25_dollars : cost_of_500_pencils 5 500 100 = 25 := by
  sorry

end cost_500_pencils_is_25_dollars_l515_515679


namespace f_at_8_solution_set_l515_515166

noncomputable def f : ℝ → ℝ := sorry

axiom f_increasing : ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f(x) < f(y)
axiom f_functional_eq : ∀ (x y : ℝ), 0 < x → 0 < y → f(x / y) = f(x) - f(y) + 1
axiom f_at_2 : f 2 = 2

theorem f_at_8 : f 8 = 4 :=
by
  sorry

theorem solution_set : {x : ℝ | 2 < x ∧ x < 4} = {x : ℝ | f(x) + f(x - 2) < 5} :=
by
  sorry

end f_at_8_solution_set_l515_515166


namespace min_expression_value_l515_515882

theorem min_expression_value (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m : ℝ, (∀ x y : ℝ, x > 2 → y > 2 → (x^3 / (y - 2) + y^3 / (x - 2)) ≥ m) ∧ 
          (m = 64) :=
by
  sorry

end min_expression_value_l515_515882


namespace number_of_solutions_l515_515333

-- Given conditions
def positiveIntSolution (x y : ℤ) : Prop := x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 2001

-- Theorem statement
theorem number_of_solutions : ∃ (count : ℕ), 
  count = 71 ∧ ∃ f : Fin count → ℤ × ℤ,
    (∀ i, positiveIntSolution (f i).1 (f i).2) :=
by
  sorry

end number_of_solutions_l515_515333


namespace position_of_each_person_l515_515363

inductive Person
| left
| middle
| right

inductive Role
| truth_teller
| liar
| diplomat

def Question1 : Person → Prop
| Person.left => (Person.middle = Role.truth_teller)
| Person.middle => (Person.middle = Role.diplomat)
| Person.right => (Person.middle = Role.liar)
| _ => False

def Solution : (Person → Role) → Prop :=
λ roles, 
  roles Person.left = Role.diplomat ∧
  roles Person.middle = Role.liar ∧
  roles Person.right = Role.truth_teller

theorem position_of_each_person (roles : Person → Role) : 
  Question1 Person.left ∧ 
  Question1 Person.middle ∧ 
  Question1 Person.right → 
  Solution roles := 
sorry

end position_of_each_person_l515_515363


namespace count_true_propositions_l515_515167

theorem count_true_propositions (p q : Prop) (hp : p) (hq : ¬ q) :
  (if (p ∨ q) then 1 else 0) + (if (p ∧ q) then 1 else 0) + (if (¬ p) then 1 else 0) + (if (¬ q) then 1 else 0) = 2 := by
  sorry

end count_true_propositions_l515_515167


namespace power_equivalence_l515_515560

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l515_515560


namespace inequality_proof_l515_515511

open Real

theorem inequality_proof (x y z λ : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (hz : 0 < z) (hλ : 0 < λ) (hk : 1 ≤ k) (hxyz : x + y + z = 3) :
  (x^k / ((λ + y) * (λ + z)) + y^k / ((λ + z) * (λ + x)) + 
   z^k / ((λ + x) * (λ + y))) ≥ (3 / ((λ + 1) * (λ + 1))) :=
by
  sorry

end inequality_proof_l515_515511


namespace number_of_trapezoids_is_336_l515_515516

-- Definition for a regular 16-gon
def regular_16gon (M : Type) := ∀ (v : Fin 16), v ∈ M

-- Function to calculate the number of sets of four vertices that form trapezoids
noncomputable def number_of_trapezoids (M : Type) [regular_16gon M] : Nat := sorry

-- Theorem statement
theorem number_of_trapezoids_is_336 (M : Type) [regular_16gon M] 
  : number_of_trapezoids M = 336 := 
by 
  sorry

end number_of_trapezoids_is_336_l515_515516


namespace area_of_tangent_circle_centers_l515_515709

noncomputable def area_of_triangle_formed_by_centers (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 6) (h3 : r3 = 7) : ℝ :=
let d12 := r1 + r2 in
let d23 := r2 + r3 in
let d13 := r1 + r3 in
let s := (d12 + d23 + d13) / 2 in
Real.sqrt (s * (s - d12) * (s - d23) * (s - d13))

theorem area_of_tangent_circle_centers : 
area_of_triangle_formed_by_centers 5 6 7  rfl rfl rfl = 61.48 :=
sorry

end area_of_tangent_circle_centers_l515_515709


namespace negation_of_universal_statement_l515_515689

theorem negation_of_universal_statement:
  ¬ (∀ x : ℝ, x > 0 → exp x > x + 1) ↔ ∃ x : ℝ, x > 0 ∧ exp x ≤ x + 1 :=
by
  sorry

end negation_of_universal_statement_l515_515689


namespace distance_traveled_in_2_minutes_l515_515817

theorem distance_traveled_in_2_minutes (a r : ℝ) (hr : r ≠ 0) :
  let rate := (a / 3) / r in
  let rate_mps := rate * 0.3048 in
  let time_seconds := 120 in
  let distance := rate_mps * time_seconds in
  distance = (12.192 * a) / r :=
by
  sorry

end distance_traveled_in_2_minutes_l515_515817


namespace find_large_monkey_doll_cost_l515_515459

-- Define the conditions and the target property
def large_monkey_doll_cost (L : ℝ) (condition1 : 300 / (L - 2) = 300 / L + 25)
                           (condition2 : 300 / (L + 1) = 300 / L - 15) : Prop :=
  L = 6

-- The main theorem with the conditions
theorem find_large_monkey_doll_cost (L : ℝ)
  (h1 : 300 / (L - 2) = 300 / L + 25)
  (h2 : 300 / (L + 1) = 300 / L - 15) : large_monkey_doll_cost L h1 h2 :=
  sorry

end find_large_monkey_doll_cost_l515_515459


namespace integral_equivalence_l515_515853

variable (x : ℝ)

def integrand : ℝ → ℝ := λ x, (x^3 - 3 * x^2 - 12) / ((x - 4) * (x - 2) * x)

noncomputable def integral_result : ℝ → ℝ := λ x, 
  x + (1 / 2) * Real.log (abs (x - 4)) + 4 * Real.log (abs (x - 2)) - (3 / 2) * Real.log (abs x)

theorem integral_equivalence :
  ∫ (x : ℝ), integrand x = integral_result x + C :=
sorry

end integral_equivalence_l515_515853


namespace product_evaluation_l515_515985

theorem product_evaluation (a b c : ℕ) (h : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) :
  6 * 15 * 2 = 4 := by
  sorry

end product_evaluation_l515_515985


namespace smallest_angle_between_planes_proof_l515_515433

noncomputable def smallest_angle_between_planes (a b c d s : ℝ) : ℝ :=
  if (s > 0 ∧ a = b ∧ b = c ∧ c = d) then π / 6 else 0

theorem smallest_angle_between_planes_proof (a b c d s : ℝ)
  (h1 : s > 0)
  (h2 : a = b)
  (h3 : b = c)
  (h4 : c = d) :
  smallest_angle_between_planes a b c d s = π / 6 :=
by {
  sorry
}

end smallest_angle_between_planes_proof_l515_515433


namespace eccentricity_of_ellipse_l515_515177

-- Define the ellipse with semi-major axis 'a' and semi-minor axis 'b'
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define focus positions based on the ellipse parameters
def foci_positions (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := real.sqrt (a^2 - b^2) in ((-c, 0), (c, 0))

-- Define the circle passing through the center O of the ellipse
def circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  let (cx, cy) := center in (x - cx)^2 + (y - cy)^2 = radius^2

-- Prove that the eccentricity e of the given ellipse is sqrt(3) - 1 under the conditions
theorem eccentricity_of_ellipse (a b : ℝ) (h_ellipse : a > b) (h_ellipse_not_zero : a > 0)
  (F1 F2 : ℝ × ℝ) (h_foci : foci_positions a b = (F1, F2))
  (O : ℝ × ℝ) (M N : ℝ × ℝ) (h_center : O = (0, 0))
  (h_circle_intersect : ellipse a b M.1 M.2 ∧ ellipse a b N.1 N.2)
  (h_circle : circle F2 (real.sqrt (a^2 - b^2)) M.1 M.2)
  (h_tangent : (let c := real.sqrt (a^2 - b^2) in
                let e := c / a in
                e^2 + 2 * e - 2 = 0)) :
  let e := (real.sqrt (a^2 - b^2)) / a in
  e = real.sqrt 3 - 1 :=
sorry

end eccentricity_of_ellipse_l515_515177


namespace books_division_l515_515773

def num_ways_divide_books : ℕ :=
  let books := {1, 2, 3, 4, 5, 6}
  let group_size := (4, 1, 1)
  let combinations := Nat.choose 6 4
  combinations

theorem books_division : num_ways_divide_books = 15 := by
  sorry

end books_division_l515_515773


namespace total_trolls_l515_515102

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l515_515102


namespace combined_weight_l515_515604

def weight_in_tons := 3
def weight_in_pounds_per_ton := 2000
def weight_in_pounds := weight_in_tons * weight_in_pounds_per_ton
def donkey_weight_in_pounds := weight_in_pounds - (0.90 * weight_in_pounds)

theorem combined_weight :
  (weight_in_pounds + donkey_weight_in_pounds) = 6600 :=
by
  -- Proof goes here
  sorry

end combined_weight_l515_515604


namespace cyclic_quadrilateral_angle_l515_515665

open Real
open Set

variables {A B K D H L M : Point}

-- Definitions from conditions
def quadrilateral_MLDH := True -- We abstract this information since the object exists
def angle_sum_MHD_MLD := ∠MHD + ∠MLD = 180
def lengths_equal := KH = (1 / 2) * AB ∧ KH = AK ∧ KH = DL

theorem cyclic_quadrilateral_angle (quadrilateral_MLDH : True) 
    (angle_sum_MHD_MLD : ∠MHD + ∠MLD = 180) 
    (lengths_equal : KH = (1 / 2) * AB ∧ KH = AK ∧ KH = DL) : 
    ∠MKD = 90 :=
by
  sorry

end cyclic_quadrilateral_angle_l515_515665


namespace right_triangle_shorter_leg_l515_515216

theorem right_triangle_shorter_leg (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
sorry

end right_triangle_shorter_leg_l515_515216


namespace equation_of_ellipse_and_line_l515_515152

-- Define the conditions for the ellipse
def ellipse (a b : ℝ) (h : a > b > 0) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions for the maximum distance
def max_distance (a b : ℝ) (h1 : a = 2*b) (dist : ℝ) : Prop :=
  dist = 2 + √3

-- Define the conditions for line intersection and the area of the triangle
def intersection (a b m x1 x2 : ℝ) (h2 : b^2 = 1): Prop :=
  (5*x1^2 + 8*m*x1 + 4*m^2 - 4 = 0) ∧ 
  (5*x2^2 + 8*m*x2 + 4*m^2 - 4 = 0)

def area_of_triangle (a b m : ℝ) : Prop :=
  ∀ x1 x2 y1 y2 : ℝ, let AB := (4 * √2 * √(5 - m^2)) / 5 in 
  let d := m / √2 in
  (AB * d / 2) = 1

-- Main theorem 
theorem equation_of_ellipse_and_line :
  ∀ (a b : ℝ), (a > b > 0) → 
  (a = 2*b) → 
  (max_distance a b a (2 + √3)) →
  (ellipse a b (a > b > 0)) →
  (b^2 = 1) →
  (area_of_triangle a b (√(10) / 2)) →
  ellipse 2 1 (2 > 1 > 0) ∧ 
  ∀ m : ℝ, (m = √(10) / 2 ∨ m = -√(10) / 2) → (y = x + m) :=
by
  sorry

end equation_of_ellipse_and_line_l515_515152


namespace invertible_2x2_matrix_l515_515111

open Matrix

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ

def A : matrix_2x2 := ![![4, 5], ![-2, 9]]

def inv_A : matrix_2x2 := ![![9/46, -5/46], ![1/23, 2/23]]

theorem invertible_2x2_matrix :
  det A ≠ 0 → (inv A = inv_A) := 
by
  sorry

end invertible_2x2_matrix_l515_515111


namespace range_of_a_l515_515189

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := 
sorry

end range_of_a_l515_515189


namespace ratio_of_sums_l515_515934

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2
axiom a4_eq_2a3 : a 4 = 2 * a 3

theorem ratio_of_sums (a : ℕ → ℝ) (S : ℕ → ℝ)
                      (arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2)
                      (a4_eq_2a3 : a 4 = 2 * a 3) :
  S 7 / S 5 = 14 / 5 :=
by sorry

end ratio_of_sums_l515_515934


namespace analytical_expression_of_f_range_of_a_l515_515513

noncomputable def f (x : ℝ) : ℝ := (2^x + 2^(-x)) / 3

def g (x : ℝ) : ℝ := (x + 5) / 4

theorem analytical_expression_of_f (x : ℝ) : f(x) = (2^x + 2^(-x)) / 3 := by
  sorry

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ m ∈ set.Icc 1 2, ∃ n ∈ set.Icc (2*real.sqrt 2 - 5) (2/3), f(a * m) = g(n)) →
  a ∈ set.Icc (1/2) 1 := by
  sorry

end analytical_expression_of_f_range_of_a_l515_515513


namespace correct_conclusions_l515_515170

-- Definitions for conditions
variables (a b c m : ℝ)
variables (h_parabola : ∀ x, a * x^2 + b * x + c ≥ 0)
variables (h_a_pos : 0 < a)
variables (h_axis : b = 2 * a)
variables (h_intersect : 0 < m ∧ m < 1)
variables (h_point : a * (1 / 2)^2 + b * (1 / 2) + c = 2)
variables (x1 x2 : ℝ)

-- Correct conclusions to prove
theorem correct_conclusions :
  (4 * a + c > 0) ∧ (∀ t : ℝ, a - b * t ≤ a * t^2 + b) :=
sorry

end correct_conclusions_l515_515170


namespace erin_trolls_count_l515_515095

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l515_515095


namespace poly_divisible_by_seven_l515_515197

-- Define the given polynomial expression
def poly_expr (x n : ℕ) : ℕ := (1 + x)^n - 1

-- Define the proof statement
theorem poly_divisible_by_seven :
  ∀ x n : ℕ, x = 5 ∧ n = 4 → poly_expr x n % 7 = 0 :=
by
  intro x n h
  cases h
  sorry

end poly_divisible_by_seven_l515_515197


namespace payment_difference_correct_l515_515461

noncomputable def prove_payment_difference (x : ℕ) (h₀ : x > 0) : Prop :=
  180 / x - 180 / (x + 2) = 3

theorem payment_difference_correct (x : ℕ) (h₀ : x > 0) : prove_payment_difference x h₀ :=
  by
    sorry

end payment_difference_correct_l515_515461


namespace sum_fourth_sixth_eq_ten_l515_515994

noncomputable def seq : ℕ → ℕ
| 1 => 2
| n+1 => (n+1) * (n+1) / (n * seq n)

theorem sum_fourth_sixth_eq_ten :
  seq 4 + seq 6 = 10 := by
  sorry

end sum_fourth_sixth_eq_ten_l515_515994


namespace arctan_tan_expr_is_75_degrees_l515_515073

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l515_515073


namespace passed_boys_avg_marks_l515_515590

theorem passed_boys_avg_marks (total_boys : ℕ) (avg_marks_all_boys : ℕ) (avg_marks_failed_boys : ℕ) (passed_boys : ℕ) 
  (h1 : total_boys = 120)
  (h2 : avg_marks_all_boys = 35)
  (h3 : avg_marks_failed_boys = 15)
  (h4 : passed_boys = 100) : 
  (39 = (35 * 120 - 15 * (total_boys - passed_boys)) / passed_boys) :=
  sorry

end passed_boys_avg_marks_l515_515590


namespace min_value_S1_div_S2_l515_515005

noncomputable def min_S1_div_S2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  let S1 := (a + b)^3 / 6
  let S2 := 1 / 2 * a * b * (a + b)
  (S1 / S2)

theorem min_value_S1_div_S2 : ∀ (a b : ℝ), 0 < a → 0 < b → min_S1_div_S2 a b (λ ha, 0 < a) (λ hb, 0 < b) ≥ 4 / 3 :=
by
  intros a b ha hb
  -- Use the AM-GM inequality to show that (a + b)^2 / (3 * a * b) ≥ 4 / 3
  have am_gm := real.AM_GM_eq a b ha hb
  sorry

end min_value_S1_div_S2_l515_515005


namespace sum_integers_50_to_75_l515_515726

theorem sum_integers_50_to_75 : (Finset.range 26).sum (λ i, 50 + i) = 1625 :=
by
  sorry

end sum_integers_50_to_75_l515_515726


namespace total_cost_is_62300_l515_515706

-- Define the dimensions and costs for each section
def length_A := 8
def width_A := 4.75
def slab_cost_A := 900

def length_B := 6
def width_B := 3.25
def slab_cost_B := 800

def length_C := 5
def width_C := 2.5
def slab_cost_C := 1000

-- Define the areas for each section
def area_A := length_A * width_A
def area_B := length_B * width_B
def area_C := length_C * width_C

-- Define the costs for each section
def cost_A := area_A * slab_cost_A
def cost_B := area_B * slab_cost_B
def cost_C := area_C * slab_cost_C

-- Define the total cost
def total_cost := cost_A + cost_B + cost_C

-- Prove that the total cost is Rs. 62,300
theorem total_cost_is_62300 : total_cost = 62300 :=
by
  sorry

end total_cost_is_62300_l515_515706


namespace line_equation_l515_515684

theorem line_equation (x y : ℝ) (hx : ∃ t : ℝ, t ≠ 0 ∧ x = t * -3) (hy : ∃ t : ℝ, t ≠ 0 ∧ y = t * 4) :
  4 * x - 3 * y + 12 = 0 := 
sorry

end line_equation_l515_515684


namespace morgan_lowest_score_l515_515289

theorem morgan_lowest_score :
  let scores := [150, 180, 175, 160]
  let total_points_needed := 1020 in
  let max_score := 200 in
  let total_first_four := list.sum scores in
  let required_sum_last_two := total_points_needed - total_first_four in
  (200 + 155 = required_sum_last_two) →
  ∀ s1 s2, s1 + s2 = required_sum_last_two ∧ s1 ≤ max_score ∧ s2 ≤ max_score → (min s1 s2 = 155) :=
by
  sorry

end morgan_lowest_score_l515_515289


namespace coral_read_percentage_l515_515082

theorem coral_read_percentage (pages_total pages_first_week pages_third_week pages_second_week : ℕ)
    (h1 : pages_total = 600)
    (h2 : pages_first_week = pages_total / 2)
    (h3 : pages_third_week = 210)
    (h4 : pages_first_week + pages_second_week + pages_third_week = pages_total) :
    (pages_second_week : ℚ) / (pages_total / 2 : ℚ) * 100 = 30 :=
by 
  -- definitions based on conditions
  have h1_def : pages_total = 600 := h1
  have h2_def : pages_first_week = 300 := by rw [h1_def, Nat.div_self (Nat.succ_pos 1)]
  have h4_def : pages_second_week = pages_total / 2 - pages_third_week := by
    rw [h1_def, h2_def, h3, Nat.sub_self]
  -- proving the main statement
  rw [h2_def, h4_def] at h4
  simp at h4
  norm_num
  sorry

end coral_read_percentage_l515_515082


namespace find_hours_spent_l515_515871

/-- Let 
  h : ℝ := hours Ed stayed in the hotel last night
  morning_hours : ℝ := 4 -- hours Ed stayed in the hotel this morning
  
  conditions:
  night_cost_per_hour : ℝ := 1.50 -- the cost per hour for staying at night
  morning_cost_per_hour : ℝ := 2 -- the cost per hour for staying in the morning
  initial_amount : ℝ := 80 -- initial amount Ed had
  remaining_amount : ℝ := 63 -- remaining amount after stay
  
  Then the total cost calculated by Ed is:
  total_cost : ℝ := (night_cost_per_hour * h) + (morning_cost_per_hour * morning_hours)
  spent_amount : ℝ := initial_amount - remaining_amount

  We need to prove that h = 6 given the above conditions.
-/
theorem find_hours_spent {h morning_hours night_cost_per_hour morning_cost_per_hour initial_amount remaining_amount total_cost spent_amount : ℝ}
  (hc1 : night_cost_per_hour = 1.50)
  (hc2 : morning_cost_per_hour = 2)
  (hc3 : initial_amount = 80)
  (hc4 : remaining_amount = 63)
  (hc5 : morning_hours = 4)
  (hc6 : spent_amount = initial_amount - remaining_amount)
  (hc7 : total_cost = night_cost_per_hour * h + morning_cost_per_hour * morning_hours)
  (hc8 : spent_amount = 17)
  (hc9 : total_cost = spent_amount) :
  h = 6 :=
by 
  sorry

end find_hours_spent_l515_515871


namespace product_of_common_divisors_eq_8000_l515_515887

theorem product_of_common_divisors_eq_8000 :
  let divisors_180 := [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180, -1, -2, -3, -4, -5, -6, -9, -10, -12, -15, -18, -20, -30, -36, -45, -60, -90, -180],
      divisors_20 := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20],
      common_divisors := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20] in
    (List.prod common_divisors) = 8000 :=
by
  let divisors_180 := [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180, -1, -2, -3, -4, -5, -6, -9, -10, -12, -15, -18, -20, -30, -36, -45, -60, -90, -180];
  let divisors_20 := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20];
  let common_divisors := [1, 2, 4, 5, 10, 20, -1, -2, -4, -5, -10, -20];
  sorry

end product_of_common_divisors_eq_8000_l515_515887


namespace solve_for_a_l515_515536

theorem solve_for_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x ∈ set.interval (-3 / 2 : ℝ) (2 : ℝ), (a * x^2 + (2 * a - 1) * x - 3) ≤ 1)
  (h₂ : ∃ x ∈ set.interval (-3 / 2 : ℝ) (2 : ℝ), (a * x^2 + (2 * a - 1) * x - 3) = 1) :
  a = 3 / 4 ∨ a = 1 / 2 :=
sorry

end solve_for_a_l515_515536


namespace _l515_515114

open Matrix

noncomputable def matrix_2x2_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

@[simp] theorem inverse_correctness : 
  invOf (matrix_2x2_inverse) = ![![9/46, -5/46], ![2/46, 4/46]] :=
by
  sorry

end _l515_515114


namespace unique_solution_h_l515_515866

theorem unique_solution_h (h : ℝ) (hne_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 :=
by
  sorry

end unique_solution_h_l515_515866


namespace second_smallest_sum_l515_515354

theorem second_smallest_sum (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
                           (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
                           (h7 : a + b + c = 180) (h8 : a + c + d = 197)
                           (h9 : b + c + d = 208) (h10 : a + b + d = 222) :
  208 ≠ 180 ∧ 208 ≠ 197 ∧ 208 ≠ 222 := 
sorry

end second_smallest_sum_l515_515354


namespace probability_of_convex_number_l515_515809

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ c < b

def is_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def digits_set := {5, 6, 7, 8, 9}

theorem probability_of_convex_number :
  ∃ p : ℚ,
  (∑ a in digits_set, ∑ b in digits_set, ∑ c in digits_set, if is_distinct a b c ∧ is_convex_number a b c then 1 else 0) / 
  (∑ a in digits_set, ∑ b in digits_set, ∑ c in digits_set, if is_distinct a b c then 1 else 0) = 1 / 3 := sorry

end probability_of_convex_number_l515_515809


namespace postage_for_72_5g_is_3_20_l515_515502

/--
The postage for sending a letter weighing 72.5 g should be 3.20 yuan based on the given postage rules:
- 0.80 yuan for each letter not exceeding 20 g
- 1.60 yuan for each letter exceeding 20 g but not exceeding 40 g
- Additional 0.80 yuan for every additional 20 g (for letters up to 100 g)

Proof statement: The postage for a letter weighing 72.5 g is 3.20 yuan.
-/
theorem postage_for_72_5g_is_3_20 : 
  ∀ w, w = 72.5 → 
  (w ≤ 20 → 0.80) ∧ 
  (w > 20 ∧ w ≤ 40 → 1.60) ∧
  (w > 40 ∧ w ≤ 60 → 2.40) ∧
  (w > 60 ∧ w ≤ 80 → 3.20) → 
  ∀ p, p = 3.20 := 
sorry

end postage_for_72_5g_is_3_20_l515_515502


namespace domain_y_l515_515395

noncomputable def domain_part1 := {x : ℝ | 
  (x^2 - 4 ≥ 0) ∧ 
  (x^2 + 2x - 3 > 0) ∧ 
  (x^2 + 2x - 3 ≠ 1)
}

theorem domain_y :
  domain_part1 = {x : ℝ | 
    (x < -1-Real.sqrt 5) ∨ 
    (-1-Real.sqrt 5 < x ∧ x < -3) ∨ 
    (2 < x)
  } :=
sorry

end domain_y_l515_515395


namespace probability_y_intercept_greater_than_x_intercept_is_five_eighteenths_l515_515716

-- Define the events and probability space
def probability_y_intercept_greater_than_x_intercept : ℚ :=
  let outcomes := (finset.product (finset.range 1 7).erase_lead, finset.range 1 7) in
  let valid_outcomes := outcomes.filter (λ (ab : ℕ × ℕ), ab.1 - ab.2 > ab.2 / ab.1 - 1) in
  valid_outcomes.card / outcomes.card

theorem probability_y_intercept_greater_than_x_intercept_is_five_eighteenths :
  probability_y_intercept_greater_than_x_intercept = 5 / 18 := 
by 
  sorry

end probability_y_intercept_greater_than_x_intercept_is_five_eighteenths_l515_515716


namespace stable_marriage_exists_l515_515335

open Classical

-- Defining the terms used in the problem: children, toys, and preferences.
def children : Type := fin n
def toys : Type := fin n

-- Assuming preferences as relations from children to toys and from toys to children
def child_prefers (c : children) (t t' : toys) : Prop := sorry
def toy_prefers (t : toys) (c c' : children) : Prop := sorry

-- Defining stable matching
def is_stable_matching (matching : children → toys) : Prop :=
  ∀ c1 c2 : children, ∀ t1 t2 : toys,
    (matching c1 = t1) → (matching c2 = t2) →
    ¬ (child_prefers c1 t2 t1 ∧ toy_prefers t2 c1 c2)

-- The theorem statement stating the existence of a stable matching
theorem stable_marriage_exists (n : ℕ) :
  ∃ (matching : children → toys), is_stable_matching matching :=
sorry

end stable_marriage_exists_l515_515335


namespace verify_num_incorrect_propositions_l515_515816

def P1 := ∀ (Π₁ Π₂ : Plane) (l : Line), (Π₁ || l) → (Π₂ || l) → (Π₁ || Π₂)
def P2 := ∀ (Π₁ Π₂ Π₃ : Plane), (Π₁ || Π₂) → (Π₂ || Π₃) → (Π₁ || Π₃)
def P3 := ∀ (Π₁ Π₂ Π₃ : Plane), (Π₁ ∩ Π₂ ≠ ∅) → (Π₁ ∩ Π₃ ≠ ∅) → (Π₂ || Π₃) → ((Π₁ ∩ Π₂) || (Π₁ ∩ Π₃))
def P4 := ∀ (Π₁ Π₂ : Plane) (l : Line), (l ∩ Π₁ ≠ ∅) → (Π₁ || Π₂) → (l ∩ Π₂ ≠ ∅)

def num_incorrect_propositions := 
  (if (∀ (Π₁ Π₂ : Plane) (l : Line), (Π₁ || l) → (Π₂ || l) → (Π₁ || Π₂)) then 0 else 1) + 
  (if (∀ (Π₁ Π₂ Π₃ : Plane), (Π₁ || Π₂) → (Π₂ || Π₃) → (Π₁ || Π₃)) then 0 else 1) + 
  (if (∀ (Π₁ Π₂ Π₃ : Plane), (Π₁ ∩ Π₂ ≠ ∅) → (Π₁ ∩ Π₃ ≠ ∅) → (Π₂ || Π₃) → ((Π₁ ∩ Π₂) || (Π₁ ∩ Π₃))) then 0 else 1) + 
  (if (∀ (Π₁ Π₂ : Plane) (l : Line), (l ∩ Π₁ ≠ ∅) → (Π₁ || Π₂) → (l ∩ Π₂ ≠ ∅)) then 0 else 1)

theorem verify_num_incorrect_propositions : 
  num_incorrect_propositions = 1 := 
sorry

end verify_num_incorrect_propositions_l515_515816


namespace invertible_2x2_matrix_l515_515110

open Matrix

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ

def A : matrix_2x2 := ![![4, 5], ![-2, 9]]

def inv_A : matrix_2x2 := ![![9/46, -5/46], ![1/23, 2/23]]

theorem invertible_2x2_matrix :
  det A ≠ 0 → (inv A = inv_A) := 
by
  sorry

end invertible_2x2_matrix_l515_515110


namespace intersect_circle_at_two_distinct_points_chord_length_range_number_of_integer_length_chords_l515_515141

variable (m : ℝ)
variable (x y : ℝ)
def circle := x^2 + (y - 1)^2 = 25
def line := m * x - y + 1 - 4 * m = 0

theorem intersect_circle_at_two_distinct_points : 
  ∀ m ∈ ℝ, ∃ A B : ℝ × ℝ, A ≠ B ∧ (circle x y) ∧ (line x y) := by
  sorry

theorem chord_length_range :
  ∀ A B : ℝ × ℝ, 
  (circle A.1 A.2) ∧ (circle B.1 B.2) ∧ (line A.1 A.2) ∧ (line B.1 B.2) → 
  6 ≤ dist A B ∧ dist A B ≤ 10 := by
  sorry

theorem number_of_integer_length_chords :
  ∃ n : ℕ, n = 8 ∧ (
    ∑ l in (finset.Icc 6 10).filter (λ l => ∃ A B : ℝ × ℝ, 
    (circle A.1 A.2) ∧ (circle B.1 B.2) ∧ 
    (line A.1 A.2) ∧ (line B.1 B.2) ∧ 
    dist A B = l),
    1) = n := by
  sorry

end intersect_circle_at_two_distinct_points_chord_length_range_number_of_integer_length_chords_l515_515141


namespace find_first_number_l515_515012

theorem find_first_number (x : ℝ) : 
  x + 2.017 + 0.217 + 2.0017 = 221.2357 → x = 217 :=
by
  intros h
  linarith
  sorry

end find_first_number_l515_515012


namespace ratio_trumpet_to_flute_l515_515694

-- Given conditions
def flute_players : ℕ := 5
def trumpet_players (T : ℕ) : ℕ := T
def trombone_players (T : ℕ) : ℕ := T - 8
def drummers (T : ℕ) : ℕ := T - 8 + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (T : ℕ) : ℕ := T - 8 + 3
def total_seats_needed (T : ℕ) : ℕ := 
  flute_players + trumpet_players T + trombone_players T + drummers T + clarinet_players + french_horn_players T

-- Proof statement
theorem ratio_trumpet_to_flute 
  (T : ℕ) (h : total_seats_needed T = 65) : trumpet_players T / flute_players = 3 :=
sorry

end ratio_trumpet_to_flute_l515_515694


namespace square_paper_side_length_l515_515783

theorem square_paper_side_length :
  ∀ (edge_length : ℝ) (num_pieces : ℕ) (side_length : ℝ),
  edge_length = 12 ∧ num_pieces = 54 ∧ 6 * (edge_length ^ 2) = num_pieces * (side_length ^ 2)
  → side_length = 4 :=
by
  intros edge_length num_pieces side_length h
  sorry

end square_paper_side_length_l515_515783


namespace exponent_equality_l515_515563

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l515_515563


namespace ratio_of_areas_l515_515013

-- Define the conditions and the proof problem's goal in Lean 4
theorem ratio_of_areas (r : ℝ) (h1 : r > 0):
  let side_small_sq := 2 * r,
      diag_small_sq := 2 * Real.sqrt 2 * r,
      radius_large_circle := Real.sqrt 2 * r,
      side_large_sq := 2 * Real.sqrt 2 * r,
      area_small_circle := Real.pi * r^2,
      area_large_sq := (2 * Real.sqrt 2 * r)^2 in
  area_small_circle / area_large_sq = Real.pi / 8 := 
by
  sorry -- Proof not required

end ratio_of_areas_l515_515013


namespace value_of_a_l515_515085

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem value_of_a : ∃ a : ℤ, star a 3 = 63 ∧ a = 30 := by
  sorry

end value_of_a_l515_515085


namespace e_n_max_value_l515_515258

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end e_n_max_value_l515_515258


namespace base_sum_correct_l515_515103

theorem base_sum_correct :
  let C := 12
  let a := 3 * 9^2 + 5 * 9^1 + 7 * 9^0
  let b := 4 * 13^2 + C * 13^1 + 2 * 13^0
  a + b = 1129 :=
by
  sorry

end base_sum_correct_l515_515103


namespace problem_rational_density_l515_515634

def units_digit (n : ℕ) : ℕ :=
  n % 10

def a (n : ℕ) : ℕ :=
  units_digit (finset.sum (finset.range n) (λ k, (k + 1) ^ 2))

noncomputable def decimal_seq_is_rational : Prop :=
  ∃ (m n : ℕ), (∀ k, a (k + m * n) = a k)

theorem problem_rational_density : decimal_seq_is_rational :=
sorry

end problem_rational_density_l515_515634


namespace polynomial_roots_transformation_equivalence_l515_515981

theorem polynomial_roots_transformation_equivalence (a b c d : ℂ)
  (h1: a + b + c + d = 0)
  (h2: ab + ac + ad + bc + bd + cd = -6)
  (h3: abc + abd + acd + bcd = 0)
  (h4: abcd = -8) : 
  let s1 := (a^2 + b^2 + c^2 + d^2) / 2,
      s2 := (a^2 + b^2 + c^2 - d^2) / 2,
      s3 := (a^2 - b^2 + c^2 + d^2) / 2,
      s4 := (-a^2 + b^2 + c^2 + d^2) / 2 in
  (x : ℂ) → (x - s1) * (x - s2) * (x - s3) * (x - s4) = 
  x^4 - 24 * x^3 + 216 * x^2 - 864 * x + 1296 :=
sorry

end polynomial_roots_transformation_equivalence_l515_515981


namespace complex_add_conjugate_is_real_l515_515949

theorem complex_add_conjugate_is_real :
  ∀ (z : ℂ), z = 2 * complex.i * (1 - complex.i) → z + conj z = 4 :=
by
  intros z hz
  sorry

end complex_add_conjugate_is_real_l515_515949


namespace exists_value_in_segment_l515_515799

theorem exists_value_in_segment (segment : Set ℝ) (h_segment : ∀ x ∈ segment, 0 < x ∧ x < 1) (h_length : ∀ a b ∈ segment, |a - b| < 0.001) :
  ∃ a ∈ segment, (∀ ε > 0, ∃ n ∈ ℕ, abs (a - n * ε) < ε) :=
begin
  sorry
end

end exists_value_in_segment_l515_515799


namespace remainder_when_divided_by_x_minus_1_is_minus_2_l515_515889

def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

theorem remainder_when_divided_by_x_minus_1_is_minus_2 : (p 1) = -2 := 
by 
  -- Proof not required
  sorry

end remainder_when_divided_by_x_minus_1_is_minus_2_l515_515889


namespace Michael_points_l515_515989

theorem Michael_points (total_points : ℕ) (num_other_players : ℕ) (avg_points : ℕ) (Michael_points : ℕ) 
  (h1 : total_points = 75)
  (h2 : num_other_players = 5)
  (h3 : avg_points = 6)
  (h4 : Michael_points = total_points - num_other_players * avg_points) :
  Michael_points = 45 := by
  sorry

end Michael_points_l515_515989


namespace mia_money_l515_515285

variable (DarwinMoney MiaMoney : ℕ)

theorem mia_money :
  (MiaMoney = 2 * DarwinMoney + 20) → (DarwinMoney = 45) → MiaMoney = 110 := by
  intros h1 h2
  rw [h2] at h1
  rw [h1]
  sorry

end mia_money_l515_515285


namespace arctan_tan_expr_is_75_degrees_l515_515071

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l515_515071


namespace fraction_transformation_l515_515199

theorem fraction_transformation (a b x: ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = (x + 2) / (x - 2) :=
by sorry

end fraction_transformation_l515_515199


namespace fraction_of_interests_l515_515695

def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

def compound_interest (P R T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

theorem fraction_of_interests :
  simple_interest 2625 8 2 / compound_interest 4000 10 2 = 1 / 4 :=
by
  sorry

end fraction_of_interests_l515_515695


namespace tina_wins_before_first_loss_l515_515365

-- Definitions based on conditions
variable (W : ℕ) -- The number of wins before Tina's first loss

-- Conditions
def win_before_first_loss : W = 10 := by sorry

def total_wins (W : ℕ) := W + 2 * W -- After her first loss, she doubles her wins and loses again
def total_losses : ℕ := 2 -- She loses twice

def career_record_condition (W : ℕ) : Prop := total_wins W - total_losses = 28

-- Proof Problem (Statement)
theorem tina_wins_before_first_loss : career_record_condition W → W = 10 :=
by sorry

end tina_wins_before_first_loss_l515_515365


namespace greatest_four_digit_divisible_by_3_5_6_l515_515375

theorem greatest_four_digit_divisible_by_3_5_6 : 
  ∃ n, n ≤ 9999 ∧ n ≥ 1000 ∧ (∀ m, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n = 9990 :=
by 
  sorry

end greatest_four_digit_divisible_by_3_5_6_l515_515375


namespace carrie_pays_l515_515845

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l515_515845


namespace number_of_triangles_number_of_triangles_correct_l515_515868

theorem number_of_triangles (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | _ => if even n then
            let k := n / 2
            (finset.range k).sum (λ i, 6*(i + 1)^2 - (i + 1))
         else
            let k := (n + 1) / 2
            (finset.range (k - 1)).sum (λ i, 6*(i + 1)^2 - (i + 1)) + 3*k^2 - 2*k

theorem number_of_triangles_correct (n : ℕ) : 
  let fn := number_of_triangles n in
  ∃ k, (n = 2 * k → fn = (finset.range k).sum (λ i, 6*(i + 1)^2 - (i + 1))) ∧ 
        (n = 2 * k - 1 → fn = (finset.range (k - 1)).sum (λ i, 6*(i + 1)^2 - (i + 1)) + 3*k^2 - 2*k) := 
by {
  sorry
}

end number_of_triangles_number_of_triangles_correct_l515_515868


namespace sum_series_formula_l515_515890

noncomputable def sum_series (a : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (k + 1) * a^k

theorem sum_series_formula (a : ℝ) (n : ℕ) (h : a ≠ 1) :
  sum_series a n = (1 - a^n - n * a^n + n * a^(n+1)) / (1 - a)^2 :=
by
  sorry

end sum_series_formula_l515_515890


namespace length_of_AP_l515_515215

theorem length_of_AP 
  (A B C M P : Point)
  (r : ℝ)
  (h_triangle : Triangle A B C)
  (h_right : ∠ A = π / 2)
  (h_AC : dist A C = 1)
  (h_BC : dist B C = 1)
  (ω : Circle)
  (h_ω_inscribed : InscribedCircle ω h_triangle)
  (h_touch_AC : Touches ω AC at M)
  (h_touch_BC : Touches ω BC at N)
  (h_AM_intersects : LineIntersectsCircleAtTwoPoints (LineSegment A M) ω M P)
  : dist A P = (√2 - 1) / (2 * (1 + √2)) := 
sorry

end length_of_AP_l515_515215


namespace original_sphere_radius_l515_515976

-- Definitions
def radius_of_shot : ℝ := 1 -- radius of each shot
def number_of_shots : ℝ := 343 -- number of shots

-- Volumes
def volume_of_shot : ℝ := (4 / 3) * Real.pi * (radius_of_shot^3)
def total_volume_of_shots : ℝ := number_of_shots * volume_of_shot

-- Theorem
theorem original_sphere_radius (R : ℝ) (h : total_volume_of_shots = (4 / 3) * Real.pi * (R^3)) : R = 7 := by
  sorry

end original_sphere_radius_l515_515976


namespace base10_to_base7_l515_515719

theorem base10_to_base7 (n : ℕ) (h : n = 648) : 
  n = 1 * 7^3 + 6 * 7^2 + 1 * 7^1 + 4 * 7^0 :=
by
  rw h
  rfl

end base10_to_base7_l515_515719


namespace book_pages_l515_515294

theorem book_pages (P : ℝ) (h1 : 2/3 * P = 1/3 * P + 20) : P = 60 :=
by
  sorry

end book_pages_l515_515294


namespace coeff_x3_in_product_l515_515875

theorem coeff_x3_in_product :
  let p1 := 3 * (Polynomial.X ^ 3) + 4 * (Polynomial.X ^ 2) + 5 * Polynomial.X + 6
  let p2 := 7 * (Polynomial.X ^ 2) + 8 * Polynomial.X + 9
  (Polynomial.coeff (p1 * p2) 3) = 94 :=
by
  sorry

end coeff_x3_in_product_l515_515875


namespace perpendicular_lines_l515_515255

-- Definitions and conditions based on the problem statement
variables (a b : ℝ) -- Assume lines a and b are real numbers for simplicity
variables (α β : ℝ) -- Assume planes α and β are real numbers for simplicity

def condition1 : Prop := (a < α) ∧ (b ∥ β) ∧ (α ⊥ β)
def condition2 : Prop := (a ⊥ α) ∧ (b ⊥ β) ∧ (α ⊥ β)
def condition3 : Prop := (a < α) ∧ (b ⊥ β) ∧ (α ∥ β)
def condition4 : Prop := (a ⊥ α) ∧ (b ∥ β) ∧ (α ∥ β)

-- The main theorem statement
theorem perpendicular_lines (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  (h2 → a ⊥ b) ∧ (h3 → a ⊥ b) ∧ (h4 → a ⊥ b) := 
sorry

end perpendicular_lines_l515_515255


namespace girls_came_in_classroom_l515_515355

theorem girls_came_in_classroom (initial_boys initial_girls boys_left final_children girls_in_classroom : ℕ)
  (h1 : initial_boys = 5)
  (h2 : initial_girls = 4)
  (h3 : boys_left = 3)
  (h4 : final_children = 8)
  (h5 : girls_in_classroom = final_children - (initial_boys - boys_left)) :
  girls_in_classroom - initial_girls = 2 :=
by
  sorry

end girls_came_in_classroom_l515_515355


namespace count_perfect_cubes_between_bounds_l515_515973

theorem count_perfect_cubes_between_bounds :
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  -- the number of perfect cubes k^3 such that 3^6 + 1 < k^3 < 3^12 + 1 inclusive is 72
  (730 < k * k * k ∧ k * k * k <= 531442 ∧ 10 <= k ∧ k <= 81 → k = 72) :=
by
  let lower_bound : ℕ := 3^6 + 1
  let upper_bound : ℕ := 3^12 + 1
  sorry

end count_perfect_cubes_between_bounds_l515_515973


namespace lexie_paintings_distribution_l515_515276

theorem lexie_paintings_distribution (total_paintings : ℕ) (rooms : ℕ) (paintings_per_room : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : rooms = 4) 
  : paintings_per_room = total_paintings / rooms :=
by {
  rw [h1, h2],
  exact rfl,
}

end lexie_paintings_distribution_l515_515276


namespace number_of_common_terms_between_arithmetic_sequences_l515_515690

-- Definitions for the sequences
def seq1 (n : Nat) := 2 + 3 * n
def seq2 (n : Nat) := 4 + 5 * n

theorem number_of_common_terms_between_arithmetic_sequences
  (A : Finset Nat := Finset.range 673)  -- There are 673 terms in seq1 from 2 to 2015
  (B : Finset Nat := Finset.range 403)  -- There are 403 terms in seq2 from 4 to 2014
  (common_terms : Finset Nat := (A.image seq1) ∩ (B.image seq2)) :
  common_terms.card = 134 := by
  sorry

end number_of_common_terms_between_arithmetic_sequences_l515_515690


namespace product_square_preceding_div_by_12_l515_515660

theorem product_square_preceding_div_by_12 (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) :=
by
  sorry

end product_square_preceding_div_by_12_l515_515660


namespace minimum_expression_value_l515_515941

noncomputable def expr (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  (2 * (Real.sin x₁)^2 + 1 / (Real.sin x₁)^2) *
  (2 * (Real.sin x₂)^2 + 1 / (Real.sin x₂)^2) *
  (2 * (Real.sin x₃)^2 + 1 / (Real.sin x₃)^2) *
  (2 * (Real.sin x₄)^2 + 1 / (Real.sin x₄)^2)

theorem minimum_expression_value :
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  x₁ + x₂ + x₃ + x₄ = Real.pi →
  expr x₁ x₂ x₃ x₄ ≥ 81 := sorry

end minimum_expression_value_l515_515941


namespace geometric_sequence_common_ratio_l515_515909

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  -- Sum of geometric series
  (h2 : a 3 = S 3 + 1) : q = 3 :=
by sorry

end geometric_sequence_common_ratio_l515_515909


namespace sequence_general_term_l515_515600

theorem sequence_general_term :
  (∀ n : ℕ, a (n + 2) = 3 * a (n + 1) - 2 * a n) →
  (a 1 = 1) →
  (a 2 = 3) →
  (∀ n : ℕ, a n = 2^n - 1) :=
by
  intros h_rec h_a1 h_a2
  sorry

end sequence_general_term_l515_515600


namespace minimum_adjacent_white_pairs_l515_515924

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l515_515924


namespace sum_first_10_common_elements_eq_l515_515121

/-- 
Find the sum of the first 10 elements that appear both among 
the terms of the arithmetic progression {4, 7, 10, 13, ...} 
and the geometric progression {20, 40, 80, 160, ...}.
-/
theorem sum_first_10_common_elements_eq :
  let a_n : ℕ → ℕ := λ n, 4 + 3 * n,
      b_k : ℕ → ℕ := λ k, 20 * 2^k,
      c_n : ℕ → ℕ := λ n, 40 * 4^(n-1),
      S : ℕ := ∑ n in finset.range 10, c_n n
  in S = 13981000 :=
by {
  -- The full proof would go here,
  -- but we are omitting it as per the instructions.
  sorry
}

end sum_first_10_common_elements_eq_l515_515121


namespace tangent_seq_1_tangent_seq_2_l515_515911

noncomputable def tangent_seq (m : ℕ) (h : 1 < m) : ℕ → ℝ
| 0 => 1
| n + 1 => ((m : ℝ) / (m - 1)) * tangent_seq n

theorem tangent_seq_1 (m : ℕ) (h : 1 < m) :
  tangent_seq m h 1 = (m : ℝ) / (m - 1) :=
by sorry

theorem tangent_seq_2 (m : ℕ) (h : 1 < m) :
  tangent_seq m h 2 = ((m : ℝ) / (m - 1))^2 :=
by sorry

end tangent_seq_1_tangent_seq_2_l515_515911


namespace calculate_x_times_a_l515_515055

-- Define variables and assumptions
variables (a b x y : ℕ)
variable (hb : b = 4)
variable (hy : y = 2)
variable (h1 : a = 2 * b)
variable (h2 : x = 3 * y)
variable (h3 : a + b = x * y)

-- The statement to be proved
theorem calculate_x_times_a : x * a = 48 :=
by sorry

end calculate_x_times_a_l515_515055


namespace sum_of_integers_abs_val_range_l515_515497

theorem sum_of_integers_abs_val_range :
  ∑ n in ({-5, -4, 4, 5} : Finset ℤ), n = 0 := 
by
  sorry

end sum_of_integers_abs_val_range_l515_515497


namespace find_basketball_lovers_l515_515585

variable (B C B_inter_C B_union_C : Nat)

theorem find_basketball_lovers (hC : C = 8) (hB_inter_C : B_inter_C = 6) (hB_union_C : B_union_C = 11)
  (h_inclusion_exclusion : B_union_C = B + C - B_inter_C) : B = 9 :=
by
  rw [hC, hB_inter_C, hB_union_C, h_inclusion_exclusion]
  sorry

end find_basketball_lovers_l515_515585


namespace probability_of_10_correct_answers_l515_515372

theorem probability_of_10_correct_answers :
  let total_combinations := 4 ^ 14 in
  let choose_4_wrong := Nat.choose 14 4 in
  let incorrect_combinations := 3 ^ 4 in
  let successful_outcomes := choose_4_wrong * incorrect_combinations in
  (successful_outcomes : ℝ) / (total_combinations : ℝ) = 81081 / 268435456 :=
  sorry

end probability_of_10_correct_answers_l515_515372


namespace necessary_but_not_sufficient_l515_515547

theorem necessary_but_not_sufficient (a b : ℝ^3)
  (p : a • b = - (∥a∥^2)) (q : a = - b) :
  (p → q) ∧ (¬ (q → p)) :=
by
  sorry

end necessary_but_not_sufficient_l515_515547


namespace problem_statement_l515_515779

variables (P D E F : Type)
variables [normed_group P] [normed_space ℝ P]

-- Conditions
def circle_centered_at_P_with_radius_2 (c : P) (r : ℝ) (d : P) : Prop :=
  dist c d = r

def segment_DE_tangent_to_circle_at_D (d e : P) (c : P) (r : ℝ) : Prop :=
  dist c d = r ∧ (d - c) ⬝ (e - d) = 0

def angle_DPE (d p e : P) (phi : ℝ) : Prop :=
  ∃ θ : ℝ, θ = phi ∧ (angle (d - p) (e - p) = real.pi / 2 - theta)

def point_F_on_PD (f p d : P) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ f = t • d + (1 - t) • p

def EF_bisects_angle_DPE (e f : P) (phi : ℝ) : Prop :=
  ∃ theta : ℝ, theta = phi ∧ ∀ θ_1 θ_2, angle theta_1 theta + angle theta₂ theta = real.pi / 2

variables (u v : ℝ)
def sin_phi (phi : ℝ) := u = real.sin phi
def cos_phi (phi : ℝ) := v = real.cos phi

-- Correct answer
noncomputable def PF_value (u : ℝ) : ℝ := 2 / (1 + u)

theorem problem_statement (P D E F : P) (phi : ℝ)
  (h1 : circle_centered_at_P_with_radius_2 P 2 D)
  (h2 : segment_DE_tangent_to_circle_at_D D E P 2)
  (h3 : angle_DPE D P E phi)
  (h4 : point_F_on_PD F P D)
  (h5 : EF_bisects_angle_DPE E F phi)
  (h6 : sin_phi u phi)
  (h7 : cos_phi v phi) :
  PF_value u = 2 / (1 + u) := sorry

end problem_statement_l515_515779


namespace max_integer_solutions_l515_515788

theorem max_integer_solutions (p : ℤ[X]) (hp : p.coeff 100 = 100) 
  : ∃ k_set : Finset ℤ, k_set.card ≤ 9 ∧ ∀ k ∈ k_set, p.eval k = k^2 := 
sorry

end max_integer_solutions_l515_515788


namespace parallelogram_angle_A_l515_515661

-- Define the problem setting
def quadrilateral_is_parallelogram (ABCD : Quadrilateral) : Prop := parallelogram ABCD
def external_angle_DC_extension (ABCD : Quadrilateral) (C D_point : Point) (angle: Real) : Prop :=
  angle = 80

-- Define the angle measure to be proven
def angle_A_measure (A_angle: Real) : Prop :=
  A_angle = 100

-- The final theorem
theorem parallelogram_angle_A
  (ABCD : Quadrilateral) 
  (parallelogram_ABCD : quadrilateral_is_parallelogram ABCD) 
  (ext_angle_condition : external_angle_DC_extension ABCD C D 80) 
  : angle_A_measure 100 :=
begin
  sorry
end

end parallelogram_angle_A_l515_515661


namespace average_dandelions_picked_l515_515058

def Billy_initial_pick : ℕ := 36
def George_initial_ratio : ℚ := 1 / 3
def additional_picks : ℕ := 10

theorem average_dandelions_picked :
  let Billy_initial := Billy_initial_pick,
      George_initial := (George_initial_ratio * Billy_initial).toNat,
      Billy_total := Billy_initial + additional_picks,
      George_total := George_initial + additional_picks,
      total_picked := Billy_total + George_total in
  total_picked / 2 = 34 :=
  by
  let Billy_initial := Billy_initial_pick
  let George_initial := (George_initial_ratio * Billy_initial).toNat
  let Billy_total := Billy_initial + additional_picks
  let George_total := George_initial + additional_picks
  let total_picked := Billy_total + George_total
  sorry

end average_dandelions_picked_l515_515058


namespace diameter_large_circle_correct_l515_515311

noncomputable def diameter_of_large_circle : ℝ :=
  2 * (Real.sqrt 17 + 4)

theorem diameter_large_circle_correct :
  ∃ (d : ℝ), (∀ (r : ℝ), r = Real.sqrt 17 + 4 → d = 2 * r) ∧ d = diameter_of_large_circle := by
    sorry

end diameter_large_circle_correct_l515_515311


namespace hexagon_area_l515_515793

theorem hexagon_area {s : ℝ} (h_eq : s = 3) :
  let A := (3 * s) * √3 / 4 in
  A = 9 * √3 := by
  sorry

end hexagon_area_l515_515793


namespace parabola_equation_l515_515329

theorem parabola_equation (p : ℝ) (h1 : 0 < p) (h2 : p / 2 = 2) : ∀ y x : ℝ, y^2 = -8 * x :=
by
  sorry

end parabola_equation_l515_515329


namespace rectangle_length_l515_515421

theorem rectangle_length
  (width : ℕ) (area : ℕ) (length : ℕ)
  (h1 : width = 6)
  (h2 : area = 48)
  (h3 : area = length * width) :
  length = 8 :=
by 
  -- Given conditions
  have h4 : 48 = length * 6 := by rw [h2, h1, h3],
  -- Solve for length
  have h5 : length = 8 := by linarith,
  assumption

end rectangle_length_l515_515421


namespace sum_of_integer_solutions_l515_515328

theorem sum_of_integer_solutions :
  (∑ x in (finset.filter (λ x : ℤ, (4 : ℤ)^x - 5 * 2^(x+1) + 16 = 0) (finset.range 10)), x) = 4 :=
by
  -- proof placeholder
  sorry

end sum_of_integer_solutions_l515_515328


namespace sum_of_integers_l515_515496

theorem sum_of_integers (s : Finset ℕ) (h₀ : ∀ a ∈ s, 0 ≤ a ∧ a ≤ 124)
  (h₁ : ∀ a ∈ s, a^3 % 125 = 2) : s.sum id = 265 :=
sorry

end sum_of_integers_l515_515496


namespace range_of_a_l515_515659

noncomputable def check_conditions (a : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ∨ (∃ x : ℝ, x^2 - x + a = 0) ∧ 
  ¬ (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∧ ∃ x : ℝ, x^2 - x + a = 0) ↔ 
  a ∈ set.Iio (-2) ∨ a ∈ set.Ioo (1/4) 2 :=
begin
  sorry
end

end range_of_a_l515_515659


namespace compare_f_values_l515_515960

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem compare_f_values : 
  f (-π / 4) > f 1 ∧ f 1 > f (π / 3) := 
sorry

end compare_f_values_l515_515960


namespace least_odd_prime_factor_2023_6_plus_1_l515_515482

theorem least_odd_prime_factor_2023_6_plus_1 : ∃ p : ℕ, prime p ∧ odd p ∧ p = 37 ∧ p ∣ (2023 ^ 6 + 1) ∧ ∀ q, prime q ∧ odd q ∧ q ∣ (2023 ^ 6 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_l515_515482


namespace find_eccentricity_find_standard_eq_l515_515693

-- Given conditions for part (1)
def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def condition1 (a b : ℝ) : Prop := a > b ∧ b > 0
def ratio_BF_AB (a b : ℝ) : Prop := sqrt(3) / 2 = a / sqrt(a^2 + b^2)

-- Given conditions for part (2)
def condition2 (O M N : ℝ × ℝ) : Prop := abs (prod.fst O * prod.snd M - prod.fst O * prod.snd N) = sqrt(3)
def equal_dist (O M N : ℝ × ℝ) : Prop := dist O M = dist O N

theorem find_eccentricity (a b : ℝ) (h_cond1 : condition1 a b) (h_ratio : ratio_BF_AB a b) : 
  ∃ e : ℝ, e = sqrt(6) / 3 := sorry

theorem find_standard_eq (x y a b : ℝ) (O M N : ℝ × ℝ) (h_cond1 : condition1 a b) (h_cond2 : condition2 O M N) (h_equal_dist : equal_dist O M N) : 
  ellipse_eq x y 6 2 := sorry

end find_eccentricity_find_standard_eq_l515_515693


namespace sum_of_all_xi_l515_515274

-- Definitions of conditions
def cond1 (x y z : ℂ) : Prop := x + y * z = 8
def cond2 (x y z : ℂ) : Prop := y + x * z = 11
def cond3 (x y z : ℂ) : Prop := z + x * y = 12

-- Proposition we want to prove
theorem sum_of_all_xi : 
  (∃ (n : ℕ) (triples : Fin n → ℂ × ℂ × ℂ), 
      (∀ i, cond1 (triples i).1 (triples i).2.1 (triples i).2.2) ∧ 
      (∀ i, cond2 (triples i).1 (triples i).2.1 (triples i).2.2) ∧ 
      (∀ i, cond3 (triples i).1 (triples i).2.1 (triples i).2.2) ∧ 
      ∑ i, (triples i).1 = 7)
 :=
sorry

end sum_of_all_xi_l515_515274


namespace range_of_x_l515_515956

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 
  else 2 * x - 1

theorem range_of_x (x : ℝ) : f x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 := 
by
  -- Proof omitted
  sorry

end range_of_x_l515_515956


namespace min_value_expression_min_value_expression_achieved_at_1_l515_515261

noncomputable def min_value_expr (a b : ℝ) (n : ℕ) : ℝ :=
  (1 / (1 + a^n)) + (1 / (1 + b^n))

theorem min_value_expression (a b : ℝ) (n : ℕ) (h1 : a + b = 2) (h2 : 0 < a) (h3 : 0 < b) : 
  (min_value_expr a b n) ≥ 1 :=
sorry

theorem min_value_expression_achieved_at_1 (n : ℕ) :
  (min_value_expr 1 1 n = 1) :=
sorry

end min_value_expression_min_value_expression_achieved_at_1_l515_515261


namespace maximal_subsets_l515_515613

theorem maximal_subsets (A : Set ℕ) (hA : A = {i | i ≤ 2006}) :
  ∃ B : Finset (Finset ℕ), 
  (∀ S T ∈ B, S ≠ T → (S ∩ T).card = 2004) ∧ B.card = 2006 := 
sorry

end maximal_subsets_l515_515613


namespace product_of_total_points_l515_515049

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def f (n : ℕ) : ℕ :=
  if is_prime n then 7
  else if n % 3 = 0 then 3
  else 0

def allie_rolls := [2, 4, 5, 3, 6]
def betty_rolls := [1, 2, 3, 3, 4, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldl (λ acc x => acc + f x) 0

theorem product_of_total_points :
  total_points allie_rolls * total_points betty_rolls = 400 := by
  sorry

end product_of_total_points_l515_515049


namespace monkeys_each_get_banana_l515_515703

-- Definitions based on conditions
def num_monkeys : ℕ := 5
def num_ladders : ℕ := 5
def has_banana (ladder : ℕ) : Prop := ladder < num_ladders
def rope_connects (r1 r2 : ℕ) : Prop := r1 ≠ r2 ∧ r1 < num_rungs ∧ r2 < num_rungs
def distinct_starting_points (monkey : ℕ) : Prop := monkey < num_monkeys

-- Theorem Statement
theorem monkeys_each_get_banana :
  ∀ (monkeys : fin num_monkeys) (ladders : fin num_ladders)
    (ropes : list (ℕ × ℕ)),
    (∀ r1 r2, r1 ≠ r2 → r1 < num_rungs → r2 < num_rungs → (r1, r2) ∈ ropes) →
    (∀ monkey, distinct_starting_points monkey) →
    (∀ ladder, has_banana ladder) →
    (∀ monkey, ∃ ladder, has_banana ladder ∧ 
      (∀ (visited_ladders : list ℕ), 
        (∀ l ∈ visited_ladders, l < num_ladders) → 
        (ladder ∈ visited_ladders))) :=
by
  sorry

end monkeys_each_get_banana_l515_515703


namespace exists_real_x_y_l515_515091

theorem exists_real_x_y (a : ℝ) :
  (∃ (x y : ℝ), sqrt (2 * x * y + a) = x + y + 17) ↔ a ≥ -289 / 2 :=
by
  sorry

end exists_real_x_y_l515_515091


namespace alex_sweaters_l515_515432

def num_items (shirts : ℕ) (pants : ℕ) (jeans : ℕ) (total_cycle_time_minutes : ℕ)
  (cycle_time_minutes : ℕ) (max_items_per_cycle : ℕ) : ℕ :=
  total_cycle_time_minutes / cycle_time_minutes * max_items_per_cycle

def num_sweaters_to_wash (total_items : ℕ) (non_sweater_items : ℕ) : ℕ :=
  total_items - non_sweater_items

theorem alex_sweaters :
  ∀ (shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle : ℕ),
  shirts = 18 →
  pants = 12 →
  jeans = 13 →
  total_cycle_time_minutes = 180 →
  cycle_time_minutes = 45 →
  max_items_per_cycle = 15 →
  num_sweaters_to_wash
    (num_items shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle)
    (shirts + pants + jeans) = 17 :=
by
  intros shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle
    h_shirts h_pants h_jeans h_total_cycle_time_minutes h_cycle_time_minutes h_max_items_per_cycle
  
  sorry

end alex_sweaters_l515_515432


namespace rectangle_perimeter_l515_515680

-- Definition of side lengths of the squares
def square_sides : Type := {b : Fin 10 → ℕ // 
  b 1 + b 2 = b 3 ∧
  b 1 + b 3 = b 4 ∧
  b 3 + b 4 = b 5 ∧
  b 4 + b 5 = b 6 ∧
  b 2 + b 3 + b 5 = b 7 ∧
  b 2 + b 7 = b 8 ∧
  b 1 + b 4 + b 6 = b 9 ∧
  b 6 + b 9 = b 7 + b 8}

noncomputable def rectangle_dimensions (s : square_sides) : (ℕ × ℕ) := 
let b := s.val in
(2 + (4 + b 2) + (10 + 3 * b 2), (2 + b 2) + (6 + 2 * b 2) + (8 + 4 * b 2))

noncomputable def relatively_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Main theorem statement
theorem rectangle_perimeter : 
  ∀ s : square_sides, let (L, W) := rectangle_dimensions s in
  relatively_prime L W → 2*L + 2*W = 270 :=
by
  -- This is where a detailed proof would follow
  sorry

end rectangle_perimeter_l515_515680


namespace mia_money_l515_515288

def darwin_has := 45
def mia_has (d : ℕ) := 2 * d + 20

theorem mia_money : mia_has darwin_has = 110 :=
by
  unfold mia_has darwin_has
  rw [←nat.mul_assoc]
  rw [nat.mul_comm 2 45]
  sorry

end mia_money_l515_515288


namespace find_alpha_beta_l515_515910

-- defining Nat and Int subset as Natural Number and Integer
def alpha : Nat 
def beta : Nat 
def m : Nat 
def k : Int

-- defining constants
constant sum_digits_constant: Nat := 21
constant alternate_sum_constant: Int := 15

-- conditions
axiom multiple_of_9 : 21 + alpha + beta = 9 * m
axiom multiple_of_11 : alpha - beta + 15 = 11 * k

-- statement to be proven
theorem find_alpha_beta (h₁ : 21 + alpha + beta = 9 * m) (h₂ : alpha - beta + 15 = 11 * k) : (alpha = 2 ∧ beta = 4) := 
  sorry

end find_alpha_beta_l515_515910


namespace correct_conclusions_count_l515_515396

open Set Int

def class (k : ℤ) : Set ℤ := { n | ∃ m : ℤ, n = 5 * m + k }

theorem correct_conclusions_count : 
    ({2011} ⊆ class 1) ∧ 
    (¬({-3} ⊆ class 3)) ∧ 
    (univ = class 0 ∪ class 1 ∪ class 2 ∪ class 3 ∪ class 4) ∧ 
    (∀ a b : ℤ, (a - b) ∈ class 0 ↔ (∃ k, a = 5 * k + b)) := 
begin 
    sorry
end

end correct_conclusions_count_l515_515396


namespace PQR_positive_iff_P_Q_R_positive_l515_515624

noncomputable def P (a b c : ℝ) : ℝ := a + b - c
noncomputable def Q (a b c : ℝ) : ℝ := b + c - a
noncomputable def R (a b c : ℝ) : ℝ := c + a - b

theorem PQR_positive_iff_P_Q_R_positive (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c * Q a b c * R a b c > 0) ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end PQR_positive_iff_P_Q_R_positive_l515_515624


namespace intersection_of_sets_l515_515546

open Set

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  M ∩ N = {0, 4, 8} := 
by
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  sorry

end intersection_of_sets_l515_515546


namespace distance_comparison_l515_515241

def distance_mart_to_home : ℕ := 800
def distance_home_to_academy : ℕ := 1300
def distance_academy_to_restaurant : ℕ := 1700

theorem distance_comparison :
  (distance_mart_to_home + distance_home_to_academy) - distance_academy_to_restaurant = 400 :=
by
  sorry

end distance_comparison_l515_515241


namespace geometric_sequence_division_condition_l515_515225

variable {a : ℕ → ℝ}
variable {q : ℝ}

/-- a is a geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a 1 * q ^ (n - 1)

/-- 3a₁, 1/2a₅, and 2a₃ forming an arithmetic sequence -/
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * (a 1 * q ^ 2) = 2 * (1 / 2 * (a 1 * q ^ 4))

theorem geometric_sequence_division_condition
  (h1 : is_geometric_sequence a q)
  (h2 : arithmetic_sequence_condition a q) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
sorry

end geometric_sequence_division_condition_l515_515225


namespace field_trip_total_l515_515408

-- Define the conditions
def vans := 2
def buses := 3
def people_per_van := 8
def people_per_bus := 20

-- The total number of people
def total_people := (vans * people_per_van) + (buses * people_per_bus)

theorem field_trip_total : total_people = 76 :=
by
  -- skip the proof here
  sorry

end field_trip_total_l515_515408


namespace union_A_B_intersection_A_C_ne_empty_implies_a_lt_8_l515_515523

section

variables {α : Type*} [LinearOrder α] {x a : α}

def A := {x : α | 2 ≤ x ∧ x ≤ 8}
def B := {x : α | 1 < x ∧ x < 6}
def C (a : α) := {x : α | x > a}

theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem intersection_A_C_ne_empty_implies_a_lt_8 {a : α} (h : (A ∩ (C a)).Nonempty) : a < 8 :=
sorry

end

end union_A_B_intersection_A_C_ne_empty_implies_a_lt_8_l515_515523


namespace smallest_k_divides_l515_515119

-- Given Problem: z^{12} + z^{11} + z^8 + z^7 + z^6 + z^3 + 1 divides z^k - 1
theorem smallest_k_divides (
  k : ℕ
) : (∀ z : ℂ, (z ^ 12 + z ^ 11 + z ^ 8 + z ^ 7 + z ^ 6 + z ^ 3 + 1) ∣ (z ^ k - 1) ↔ k = 182) :=
sorry

end smallest_k_divides_l515_515119


namespace eccentricity_of_hyperbola_eq_l515_515541

-- Definitions based on the problem conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def focal_length (c : ℝ) : ℝ := 2 * c
def distance_foci_asymptote (c b : ℝ) : Prop := b = c / 2

-- Theorem statement
theorem eccentricity_of_hyperbola_eq : 
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  hyperbola a b 0 0 → -- This just serves to include a, b in the context of the hyperbola
  distance_foci_asymptote c b →
  (c^2 - a^2 = b^2) →
  (c / a = 2 * real.sqrt 3 / 3) := 
by
  intros a b c a_pos b_pos c_pos hyperbola_def dist_foci_asymp_def c2_a2_eq_b2
  sorry

end eccentricity_of_hyperbola_eq_l515_515541


namespace sin_alpha_value_l515_515531

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the length of the line segment OP
def OP_length : ℝ := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

-- Define the sine of the angle α passing through P
def sin_alpha : ℝ := P.2 / OP_length

-- The goal is to prove the following statement
theorem sin_alpha_value :
  sin_alpha = (2 * Real.sqrt 5) / 5 :=
by
  -- Proof would go here
  sorry

end sin_alpha_value_l515_515531


namespace sum_of_roots_of_cis_equation_l515_515342

theorem sum_of_roots_of_cis_equation 
  (cis : ℝ → ℂ)
  (phi : ℕ → ℝ)
  (h_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 0 ≤ phi k ∧ phi k < 360)
  (h_equation : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → (cis (phi k)) ^ 5 = (1 / Real.sqrt 2) + (Complex.I / Real.sqrt 2))
  : (phi 1 + phi 2 + phi 3 + phi 4 + phi 5) = 450 :=
by
  sorry

end sum_of_roots_of_cis_equation_l515_515342


namespace smallest_n_satisfying_conditions_l515_515645

-- defining the function f(x) = ⌊x⌋{x} where ⌊x⌋ is the floor of x and {x} is the fractional part of x
def f (x : ℝ) : ℝ := (x.floor : ℝ) * (x - x.floor)

-- proving the smallest positive integer n such that the graph of f(f(f(x))) on [0, n] 
-- is the union of 2017 or more line segments is 23
theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, (∀ x ∈ set.Icc 0 (n : ℝ), 
  let y := f(f(f(x))) in 
    -- the number of line segments in the graph of y on [0, n]
    -- implying n + 1 choose 3 is at least 2017
    (finset.nat.antidiagonal n).card * (n.choose 3) ≥ 2017) ∧ n = 23 := 
sorry

end smallest_n_satisfying_conditions_l515_515645


namespace math_club_team_selection_l515_515290

def total_students : ℕ := 22
def team_size : ℕ := 8

theorem math_club_team_selection : nat.choose total_students team_size = 319770 := 
by sorry

end math_club_team_selection_l515_515290


namespace sum_of_all_side_lengths_of_pentagon_l515_515360

def side_length (n : ℕ) := 15 -- length of one side is 15 cm
def num_sides_pentagon := 5 -- pentagon has 5 sides

theorem sum_of_all_side_lengths_of_pentagon : 
  ∀ (length : ℕ), (length = side_length 1) → (num_sides_pentagon * length = 75) := 
by
  intro length
  intro h
  calc
    num_sides_pentagon * length = 5 * 15 : by rw [h]
                          ... = 75     : by norm_num

end sum_of_all_side_lengths_of_pentagon_l515_515360


namespace sheets_needed_per_printer_l515_515775

def sheets_per_printer (total_sheets : ℤ) (num_printers : ℤ) : ℤ :=
  int.floor (total_sheets / num_printers)

theorem sheets_needed_per_printer (total_sheets : ℤ) (num_printers : ℤ)
  (h1 : total_sheets = 221)
  (h2 : num_printers = 31) :
  sheets_per_printer total_sheets num_printers = 7 :=
by
  -- import necessary libraries to perform the proof
  -- here we skip the proof with sorry
  sorry

end sheets_needed_per_printer_l515_515775


namespace stratified_sampling_medium_supermarkets_l515_515403

theorem stratified_sampling_medium_supermarkets 
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (sampling_ratio : ℚ)
  (medium_supermarkets_to_draw : ℕ) 
  (h_sampling_ratio : sampling_ratio = sample_size / (large_supermarkets + medium_supermarkets + small_supermarkets))
  (h_medium_supermarkets_to_draw : medium_supermarkets_to_draw = medium_supermarkets * sampling_ratio) :
  medium_supermarkets_to_draw = 40 :=
by
  -- Applying the conditions
  have h1 : large_supermarkets = 200 := rfl,
  have h2 : medium_supermarkets = 400 := rfl,
  have h3 : small_supermarkets = 1400 := rfl,
  have h4 : sample_size = 200 := rfl,
  have h5 : sampling_ratio = 1 / 10 := by sorry,
  have h6 : medium_supermarkets_to_draw = 400 / 10 := by sorry,
  exact h6

end stratified_sampling_medium_supermarkets_l515_515403


namespace common_difference_correct_l515_515528

noncomputable def common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 4) + (a 12) = 24 ∧ -- The sum condition
  a 1 = -2 ∧           -- The first term condition
  (∀ n, a n = a 1 + (n - 1) * d) -- Definition of the arithmetic sequence

theorem common_difference_correct (a : ℕ → ℤ) (d : ℤ) :
  common_difference a d → d = 2 :=
by
  intro h,
  cases h with h_sum h_rest,
  cases h_rest with h_first h_def,
  sorry

end common_difference_correct_l515_515528


namespace marching_band_formations_l515_515020

open Nat

theorem marching_band_formations :
  ∃ g, (g = 9) ∧ ∀ s t : ℕ, (s * t = 480 ∧ 15 ≤ t ∧ t ≤ 60) ↔ 
    (t = 15 ∨ t = 16 ∨ t = 20 ∨ t = 24 ∨ t = 30 ∨ t = 32 ∨ t = 40 ∨ t = 48 ∨ t = 60) :=
by
  -- Skipped proof.
  sorry

end marching_band_formations_l515_515020


namespace one_eighth_of_2_pow_33_eq_2_pow_x_l515_515570

theorem one_eighth_of_2_pow_33_eq_2_pow_x (x : ℕ) : (1 / 8) * (2 : ℝ) ^ 33 = (2 : ℝ) ^ x → x = 30 := by
  intro h
  sorry

end one_eighth_of_2_pow_33_eq_2_pow_x_l515_515570


namespace sqrt_eq_self_implies_a_squared_plus_one_l515_515169

theorem sqrt_eq_self_implies_a_squared_plus_one (a : ℝ) (h : sqrt a = a) : a^2 + 1 = 1 ∨ a^2 + 1 = 2 := by
  sorry

end sqrt_eq_self_implies_a_squared_plus_one_l515_515169


namespace problem_statement_l515_515002

def permutations (n r : ℕ) : ℕ := n.factorial / (n - r).factorial
def combinations (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : permutations 4 2 - combinations 4 3 = 8 := 
by 
  sorry

end problem_statement_l515_515002


namespace man_arrived_earlier_l515_515466

-- Definitions of conditions as Lean variables
variables
  (usual_arrival_time_home : ℕ)  -- The usual arrival time at home
  (usual_drive_time : ℕ) -- The usual drive time for the wife to reach the station
  (early_arrival_difference : ℕ := 16) -- They arrived home 16 minutes earlier
  (man_walk_time : ℕ := 52) -- The man walked for 52 minutes

-- The proof statement
theorem man_arrived_earlier
  (usual_arrival_time_home : ℕ)
  (usual_drive_time : ℕ)
  (H : usual_arrival_time_home - man_walk_time <= usual_drive_time - early_arrival_difference)
  : man_walk_time = 52 :=
sorry

end man_arrived_earlier_l515_515466


namespace scientific_notation_of_86000000_l515_515331

theorem scientific_notation_of_86000000 :
  ∃ (x : ℝ) (y : ℤ), 86000000 = x * 10^y ∧ x = 8.6 ∧ y = 7 :=
by
  use 8.6
  use 7
  sorry

end scientific_notation_of_86000000_l515_515331


namespace monotone_increasing_interval_l515_515336

def f (x : ℝ) := x^2 - 2

theorem monotone_increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y :=
by
  sorry

end monotone_increasing_interval_l515_515336


namespace turtles_still_on_sand_l515_515041

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l515_515041


namespace num_ways_to_choose_providers_l515_515610

theorem num_ways_to_choose_providers (n m : ℕ) (h : n = 4) (k : m = 25) :
  ∏ i in finset.range n, (m - i) = 303600 :=
by
  cases h  -- substitute n = 4
  cases k  -- substitute m = 25
  norm_num  -- compute the product ∏ i in finset.range 4, (25 - i)
  sorry    -- this proof will show 25 * 24 * 23 * 22 = 303600

end num_ways_to_choose_providers_l515_515610


namespace problem_statement_l515_515621

def setS : Set (ℝ × ℝ) := {p | p.1 * p.2 > 0}
def setT : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0}

theorem problem_statement : setS ∪ setT = setS ∧ setS ∩ setT = setT :=
by
  -- To be proved
  sorry

end problem_statement_l515_515621


namespace pentagon_area_calc_l515_515977

variables (a b c d e : ℝ)
variables (angle_ade : ℝ)
variables (right_triangle_area trapezoid_area total_area : ℝ)

-- Given side lengths
def side_lengths : Prop := 
  a = 18 ∧ b = 20 ∧ c = 27 ∧ d = 24 ∧ e = 20

-- Given one internal angle is 90 degrees
def internal_angle : Prop := angle_ade = 90

-- Calculate the area of a right triangle
def right_triangle_area_calc := 0.5 * a * b

-- Calculate the area of the trapezoid
def trapezoid_area_calc := 0.5 * (d + c) * b

-- Calculate the total area
def total_area_calc := right_triangle_area_calc + trapezoid_area_calc

theorem pentagon_area_calc (h1 : side_lengths) (h2 : internal_angle) : total_area_calc = 690 :=
by sorry

end pentagon_area_calc_l515_515977


namespace solve_inequality_l515_515673

theorem solve_inequality (a : ℝ) :
  (a = 0 → {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a > 0 → {x : ℝ | x ≥ 2 / a} ∪ {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | 2 / a ≤ x ∧ x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a = -2 → {x : ℝ | x = -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a < -2 → {x : ℝ | -1 ≤ x ∧ x ≤ 2 / a} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) :=
by 
  sorry

end solve_inequality_l515_515673


namespace expression_takes_many_different_values_l515_515469

theorem expression_takes_many_different_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) : 
  ∃ v : ℝ, ∀ x, x ≠ 3 → x ≠ -2 → v = (3*x^2 - 2*x + 3)/((x - 3)*(x + 2)) - (5*x - 6)/((x - 3)*(x + 2)) := 
sorry

end expression_takes_many_different_values_l515_515469


namespace percent_increase_l515_515446

variable (P : ℝ)
def firstQuarterPrice := 1.20 * P
def secondQuarterPrice := 1.50 * P

theorem percent_increase:
  ((secondQuarterPrice P - firstQuarterPrice P) / firstQuarterPrice P) * 100 = 25 := by
  sorry

end percent_increase_l515_515446


namespace fraction_of_arith_geo_seq_l515_515928

theorem fraction_of_arith_geo_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_seq_arith : ∀ n, a (n+1) = a n + d)
  (h_seq_geo : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by
  sorry

end fraction_of_arith_geo_seq_l515_515928


namespace curve_tangent_common_a_range_l515_515572

noncomputable def h (x : ℝ) : ℝ := -log (2 * x + 1) + x^2

theorem curve_tangent_common_a_range : 
  {a : ℝ | ∃ x: ℝ, x > -1/2 ∧ 3 * a = h x} = Ici ((1 - 4 * log 2) / 12) := sorry

end curve_tangent_common_a_range_l515_515572


namespace problem1_problem2_l515_515066

theorem problem1 : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := 
by 
  sorry
  
theorem problem2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1) ^ 2 = 14 + 4 * Real.sqrt 3 := 
by 
  sorry

end problem1_problem2_l515_515066


namespace unemployment_percentage_next_year_l515_515811

theorem unemployment_percentage_next_year (E U : ℝ) (h1 : E > 0) :
  ( (0.91 * (0.056 * E)) / (1.04 * E) ) * 100 = 4.9 := by
  sorry

end unemployment_percentage_next_year_l515_515811


namespace bricks_in_wall_l515_515404

-- Definitions of conditions based on the problem statement
def time_first_bricklayer : ℝ := 12 
def time_second_bricklayer : ℝ := 15 
def reduced_productivity : ℝ := 12 
def combined_time : ℝ := 6
def total_bricks : ℝ := 720

-- Lean 4 statement of the proof problem
theorem bricks_in_wall (x : ℝ) 
  (h1 : (x / time_first_bricklayer + x / time_second_bricklayer - reduced_productivity) * combined_time = x) 
  : x = total_bricks := 
by {
  sorry
}

end bricks_in_wall_l515_515404


namespace problem1_problem2_l515_515955

noncomputable def f (x a b : ℝ) : ℝ := 2 * x ^ 2 - 2 * a * x + b

noncomputable def set_A (a b : ℝ) : Set ℝ := {x | f x a b > 0 }

noncomputable def set_B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1 }

theorem problem1 (a b : ℝ) (h : f (-1) a b = -8) :
  (∀ x, x ∈ (set_A a b)ᶜ ∪ set_B 1 ↔ -3 ≤ x ∧ x ≤ 2) :=
  sorry

theorem problem2 (a b : ℝ) (t : ℝ) (h : f (-1) a b = -8) (h_not_P : (set_A a b) ∩ (set_B t) = ∅) :
  -2 ≤ t ∧ t ≤ 0 :=
  sorry

end problem1_problem2_l515_515955


namespace cost_price_computer_table_l515_515691

noncomputable def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cost_price_computer_table (SP : ℝ) (CP : ℝ) (h : SP = 7967) (h2 : SP = 1.24 * CP) : 
  approx_eq CP 6424 0.01 :=
by
  sorry

end cost_price_computer_table_l515_515691


namespace correct_statements_l515_515192

-- Definitions for the problem
def plane (P : Type) := {pt : P // ∃ a b c : ℝ, a*pt.1 + b*pt.2 + c*pt.3 = 0}
def perpendicular_planes (P Q : Type) [plane P] [plane Q] : Prop :=
  ∃ a1 b1 c1 a2 b2 c2 : ℝ, (a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2)

-- Conditions: given two perpendicular planes
variable {P Q : Type} [plane P] [plane Q]

-- Correct statements to be proven
theorem correct_statements (h : perpendicular_planes P Q) :
  (∀ (l : P), ∃ (k : Q), l ⊥ k) ∧
  (∀ (pt : P), ∃ (l : P), l ⊥ ∃ (m : Q), m ⊥ pt) :=
sorry

end correct_statements_l515_515192


namespace decrypt_nbui_is_math_l515_515617

-- Define the sets A and B as the 26 English letters
def A := {c : Char | c ≥ 'a' ∧ c ≤ 'z'}
def B := A

-- Define the mapping f from A to B
def f (c : Char) : Char :=
  if c = 'z' then 'a'
  else Char.ofNat (c.toNat + 1)

-- Define the decryption function g (it reverses the mapping f)
def g (c : Char) : Char :=
  if c = 'a' then 'z'
  else Char.ofNat (c.toNat - 1)

-- Define the decryption of the given ciphertext
def decrypt (ciphertext : String) : String :=
  ciphertext.map g

-- Prove that the decryption of "nbui" is "math"
theorem decrypt_nbui_is_math : decrypt "nbui" = "math" :=
  by
  sorry

end decrypt_nbui_is_math_l515_515617


namespace exists_special_cubic_polynomial_l515_515237

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l515_515237


namespace range_of_a_l515_515958

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / Real.exp x - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a ^ 2) ≤ 0) : 
  a ∈ Set.Iic (-1) ∪ Set.Ici (1 / 2) :=
sorry

end range_of_a_l515_515958


namespace joes_total_weight_l515_515995

theorem joes_total_weight (F S : ℕ) (h1 : F = 700) (h2 : 2 * F = S + 300) :
  F + S = 1800 :=
by
  sorry

end joes_total_weight_l515_515995


namespace product_inequality_l515_515635

-- Define the conditions
variable {I : Set ℝ} (f : ℝ → ℝ) (x : ℕ → ℝ) (p : ℕ → ℝ) (n : ℕ) (r : ℝ)

-- Given assumptions 
-- continuous positive function f on I ⊆ ℝ⁺
-- for any xi ∈ I
-- pi > 0 and Σ pi = 1
-- base case for n = 2
-- inductive hypothesis for n = k
-- inductive step for n = k+1
def problem_conditions := 
  (∀ x ∈ I, 0 < f x) ∧ 
  (∀ i, 0 < i → i ≤ n → x i ∈ I) ∧ 
  (∀ i, 0 < p i) ∧ 
  (∑ i in Finset.range (n + 1), p i = 1)

-- The final proof statement 
theorem product_inequality 
  (h : problem_conditions f x p n r) : 
  ∏ i in Finset.range (n + 1), f (x i) ≥ f ((∑ i in Finset.range (n + 1), p i * x i^r)^(1/r)) :=
sorry

end product_inequality_l515_515635


namespace cos_of_C_in_triangle_l515_515233

theorem cos_of_C_in_triangle 
  (A B C : ℝ)
  (sin_A : ℝ) (cos_B : ℝ)
  (h_sin_A : sin_A = 5 / 13)
  (h_cos_B : cos_B = 3 / 5)
  (h_angle_sum : A + B + C = π) :
  cos (π - (A + B)) = -16 / 65 :=
by
  sorry

end cos_of_C_in_triangle_l515_515233


namespace general_term_a_n_l515_515913

open BigOperators

variable {a : ℕ → ℝ}  -- The sequence a_n
variable {S : ℕ → ℝ}  -- The sequence sum S_n

-- Define the sum of the first n terms:
def seq_sum (a : ℕ → ℝ) (n : ℕ) := ∑ k in Finset.range (n + 1), a k

theorem general_term_a_n (h : ∀ n : ℕ, S n = 2 ^ n - 1) (n : ℕ) : a n = 2 ^ (n - 1) :=
by
  sorry

end general_term_a_n_l515_515913


namespace price_increase_to_restore_l515_515340

-- Definitions and conditions
def original_price (P : ℝ) := P
def reduced_price (P : ℝ) := 0.8 * P
def restored_price_factor (reduced_price : ℝ) := 100.0 / reduced_price
def percentage_increase (factor : ℝ) := (factor - 1) * 100

-- Main theorem statement
theorem price_increase_to_restore (P : ℝ > 0) :
  percentage_increase (restored_price_factor (reduced_price P)) = 25 := sorry

end price_increase_to_restore_l515_515340


namespace customer_buys_smartphones_l515_515801

theorem customer_buys_smartphones (total_smartphones defective_smartphones : ℕ) 
  (prob_all_defective : ℝ) (prob_defective : ℝ) (n : ℝ) :
  total_smartphones = 230 →
  defective_smartphones = 84 →
  prob_all_defective = 0.13237136890070247 →
  prob_defective = (defective_smartphones / total_smartphones) →
  (prob_defective ^ n) = prob_all_defective →
  n ≈ 2 := 
sorry

end customer_buys_smartphones_l515_515801


namespace find_initial_salt_concentration_l515_515806

noncomputable def initial_salt_concentration 
  (x : ℝ) (final_concentration : ℝ) (extra_water : ℝ) (extra_salt : ℝ) (evaporation_fraction : ℝ) : ℝ :=
  let initial_volume : ℝ := x
  let remaining_volume : ℝ := evaporation_fraction * initial_volume
  let mixed_volume : ℝ := remaining_volume + extra_water + extra_salt
  let target_salt_volume_fraction : ℝ := final_concentration / 100
  let initial_salt_volume_fraction : ℝ := (target_salt_volume_fraction * mixed_volume - extra_salt) / initial_volume * 100
  initial_salt_volume_fraction

theorem find_initial_salt_concentration :
  initial_salt_concentration 120 33.333333333333336 8 16 (3 / 4) = 18.333333333333332 :=
by
  sorry

end find_initial_salt_concentration_l515_515806


namespace linear_regression_equation_predict_2023_l515_515751

variable (x_vals : List ℝ := [1, 2, 3, 4, 5])
variable (y_vals : List ℝ := [4.9, 4.1, 3.9, 3.2, 3.5])
variable (sum_diff_x_y : ℝ := -3.7)

noncomputable def mean (lst : List ℝ) : ℝ :=
  lst.sum / lst.length

noncomputable def linear_regression_slope (x_vals y_vals : List ℝ) (mean_x mean_y : ℝ) : ℝ :=
  sum_diff_x_y / (x_vals.map (fun x => (x - mean_x) ^ 2)).sum

noncomputable def linear_regression_intercept (mean_x mean_y slope : ℝ) : ℝ :=
  mean_y - slope * mean_x

theorem linear_regression_equation :
  let mean_x := mean x_vals
  let mean_y := mean y_vals
  let slope := linear_regression_slope x_vals y_vals mean_x mean_y
  let intercept := linear_regression_intercept mean_x mean_y slope in
  slope = -0.37 ∧ intercept = 5.03 :=
by
  let mean_x := mean x_vals
  let mean_y := mean y_vals
  let slope := linear_regression_slope x_vals y_vals mean_x mean_y
  let intercept := linear_regression_intercept mean_x mean_y slope
  sorry

theorem predict_2023 :
  let slope := -0.37
  let intercept := 5.03
  let x := 6 in
  let y := slope * x + intercept in
  y = 2.81 :=
by
  let slope := -0.37
  let intercept := 5.03
  let x := 6
  let y := slope * x + intercept
  sorry

end linear_regression_equation_predict_2023_l515_515751


namespace find_angle_C_find_side_c_l515_515191

variable (A B C : ℝ) (CA CB : ℝ)
variable (a b c S : ℝ)

-- Condition for dot product of vectors CA and CB
axiom dot_product_condition : CA * CB = 1
-- Condition for the area of the triangle
axiom area_condition : S = 1 / 2
-- Condition for sin and cos of angle A
axiom sin_cos_A_condition : sin A * cos A = sqrt 3 / 4
-- Given length of side a
axiom side_a_condition : a = 2
-- Derived angle C
axiom angle_C_condition : C = π / 4

-- Question (1): Prove the value of angle C
theorem find_angle_C (CA CB : ℝ) (S : ℝ) : 
  (CA * CB = 1) ∧ (S = 1 / 2) → C = π / 4 := by
  intros h
  sorry

-- Question (2): Prove the possible values of c
theorem find_side_c (A B C : ℝ) (a : ℝ) :
  (sin A * cos A = sqrt 3 / 4) ∧ (a = 2) ∧ (C = π / 4) →
  c = sqrt 6 ∨ c = 2 * sqrt 6 / 3 := by
  intros h
  sorry

end find_angle_C_find_side_c_l515_515191


namespace original_total_cost_l515_515217

-- Definitions based on the conditions
def price_jeans : ℝ := 14.50
def price_shirt : ℝ := 9.50
def price_jacket : ℝ := 21.00

def jeans_count : ℕ := 2
def shirts_count : ℕ := 4
def jackets_count : ℕ := 1

-- The proof statement
theorem original_total_cost :
  (jeans_count * price_jeans) + (shirts_count * price_shirt) + (jackets_count * price_jacket) = 88 := 
by
  sorry

end original_total_cost_l515_515217


namespace circumcircle_radii_equal_l515_515052

noncomputable theory
open_locale classical

variables {A B C D E A' : Type*}
variables (ABC : Triangle A B C) [acute_triangle ABC]
variable (BC_longest : BC.is_longest_side_of ABC)
variables (D_on_AC : On_Line D AC) (E_on_AB : On_Line E AB)
variables (BD_eq_BA : BD = BA) (CE_eq_CA : CE = CA)
variable (A'_reflection_of_A : Reflection A BC A')

theorem circumcircle_radii_equal :
  ∀ (h : ∀ P Q : Point, ∃ (c : Circle), c.is_circumcircle_of ⟨A, B, C⟩ ∧ c.is_circumcircle_of ⟨A', D, E⟩),
  ∃ r1 r2 : ℝ, r1 = r2 :=
begin
  by_contradiction,
  sorry
end

end circumcircle_radii_equal_l515_515052


namespace minimal_value_n_plus_d_l515_515883

noncomputable def minimal_sum_n_d : ℕ :=
  let a_n (n : ℕ) (d : ℕ) := 1 + (n - 1) * d in
  let possible_pairs := [(11, 5), (6, 10)] in -- pairs (n, d) where n + d is minimized
  possible_pairs.foldl (λ min_val pair, min min_val ((pair.1) + (pair.2))) 16

-- The proof statement
theorem minimal_value_n_plus_d :
  ∃ n d : ℕ, a_n n d = 51 ∧ 1 + (n - 1) * d = 51 ∧ n + d = minimal_sum_n_d :=
sorry

end minimal_value_n_plus_d_l515_515883


namespace power_expression_l515_515938

variable {x : ℂ} -- Define x as a complex number

theorem power_expression (
  h : x - 1/x = 2 * Complex.I * Real.sqrt 2
) : x^(2187:ℕ) - 1/x^(2187:ℕ) = -22 * Complex.I * Real.sqrt 2 :=
by sorry

end power_expression_l515_515938


namespace germination_percentage_l515_515127

theorem germination_percentage (total_seeds_plot1 total_seeds_plot2 germinated_plot2_percentage total_germinated_percentage germinated_plot1_percentage : ℝ) 
  (plant1 : total_seeds_plot1 = 300) 
  (plant2 : total_seeds_plot2 = 200) 
  (germination2 : germinated_plot2_percentage = 0.35) 
  (total_germination : total_germinated_percentage = 0.23)
  (germinated_plot1 : germinated_plot1_percentage = 0.15) :
  (total_germinated_percentage * (total_seeds_plot1 + total_seeds_plot2) = 
    (germinated_plot2_percentage * total_seeds_plot2) + (germinated_plot1_percentage * total_seeds_plot1)) :=
by
  sorry

end germination_percentage_l515_515127


namespace cos_cubic_solution_count_l515_515088

theorem cos_cubic_solution_count : 
  (∃ x₀ x₁ x₂ x₃ ∈ Icc (0 : ℝ) (2 * Real.pi), 
    (3 * Real.cos x₀ ^ 3 - 7 * Real.cos x₀ ^ 2 + 3 * Real.cos x₀ = 0) ∧
    (3 * Real.cos x₁ ^ 3 - 7 * Real.cos x₁ ^ 2 + 3 * Real.cos x₁ = 0) ∧
    (3 * Real.cos x₂ ^ 3 - 7 * Real.cos x₂ ^ 2 + 3 * Real.cos x₂ = 0) ∧
    (3 * Real.cos x₃ ^ 3 - 7 * Real.cos x₃ ^ 2 + 3 * Real.cos x₃ = 0) ∧
    list.nodup [x₀, x₁, x₂, x₃] ∧ 
    list.chain (≤) x₀ [x₁, x₂, x₃])
  :=
sorry

end cos_cubic_solution_count_l515_515088


namespace convex_curve_line_segment_l515_515648

open Function Set

variables {α : Type*} [Add α] {A B : α} {a : ℝ}
variables {K K1 K2 : Set α}

-- The sum of the convex curves definition
def convex_sum (K1 K2 : Set α) : Set α :=
{P | ∃ P1 ∈ K1, ∃ P2 ∈ K2, P = P1 + P2}

-- The assumption that K is the sum of K1 and K2
axiom K_eq_convex_sum : K = convex_sum K1 K2

-- The assumption that K contains a straight line segment AB of length a
axiom K_contains_segment_AB : ∃ (t : ℝ → α), (∀ τ : ℝ, 0 ≤ τ ∧ τ ≤ 1 → t τ ∈ K) ∧ (t 1 - t 0 = B - A) ∧ (dist A B = a)

-- The property of the distance function
axiom dist_eq_length : dist A B = (B - A).norm

-- The theorem to prove
theorem convex_curve_line_segment (K1 K2 : Set α) (A B : α) (a : ℝ) :
  (K = convex_sum K1 K2) →
  (∃ (t : ℝ → α), (∀ τ : ℝ, 0 ≤ τ ∧ τ ≤ 1 → t τ ∈ K) ∧ (t 1 - t 0 = B - A) ∧ (dist A B = a)) →
  ∃ C D : α, (C - D = B - A ∧ dist C D = a ∧ (C ∈ K1 ∨ D ∈ K2)) ∨ 
  (∃ A1 B1 ∈ K1, ∃ A2 B2 ∈ K2, (A1 - B1 = B - A ∨ A2 - B2 = B - A) ∧ dist A1 B1 + dist A2 B2 = a) :=
sorry

end convex_curve_line_segment_l515_515648


namespace total_trolls_l515_515100

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l515_515100


namespace sum_four_digit_numbers_l515_515490

theorem sum_four_digit_numbers : ∑ n in {p | ∀ d ∈ p_digits n, d ∈ {1, 2, 3, 4, 5} ∧ nodup p_digits n ∧ p_digits n.length = 4}, n = 399960 :=
by
  sorry

end sum_four_digit_numbers_l515_515490


namespace angle_relationship_l515_515828

theorem angle_relationship (angle1 angle2 angle3 : ℝ)
  (h1 : ∠AOE = angle2)
  (h2 : ∠1 is an exterior angle to a triangle where ∠AOE = angle2)
  (h3 : angle2 > angle3) : angle1 > angle2 ∧ angle2 > angle3 :=
by
  sorry

end angle_relationship_l515_515828


namespace number_and_square_sum_eq_132_l515_515580

theorem number_and_square_sum_eq_132 : 
  let n := 11 in n + n^2 = 132 :=
by
  sorry

end number_and_square_sum_eq_132_l515_515580


namespace min_adjacent_white_cells_8x8_grid_l515_515919

theorem min_adjacent_white_cells_8x8_grid (n_blacks : ℕ) (h1 : n_blacks = 20) : 
  ∃ w_cell_pairs, w_cell_pairs = 34 :=
by
  -- conditions are translated here for interpret
  let total_pairs := 112 -- total pairs in 8x8 grid
  let max_spoiled := 78  -- maximum spoiled pairs when placing 20 black cells
  let min_adjacent_white_pairs := total_pairs - max_spoiled
  use min_adjacent_white_pairs
  exact (by linarith)
  sorry

end min_adjacent_white_cells_8x8_grid_l515_515919


namespace sum_f_values_l515_515273

def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem sum_f_values :
  (List.range' (-2016) 4033).sum (λ k, f k) = 4033 :=
begin
  let D := {x : ℝ | x ∈ ℝ},
  have h_f_sym := ∀ (x₁ x₂ : ℝ), x₁ + x₂ = 0 → f(x₁) + f(x₂) = 2,
  sorry
end

end sum_f_values_l515_515273


namespace ratio_of_areas_l515_515822

noncomputable def radius_of_circle (r : ℝ) : ℝ := r

def equilateral_triangle_side_length (r : ℝ) : ℝ := r * Real.sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem ratio_of_areas (r : ℝ) : 
  ∃ K : ℝ, K = (3 * Real.sqrt 3) / (4 * Real.pi) → 
  (area_of_equilateral_triangle (equilateral_triangle_side_length r)) / (area_of_circle r) = K := 
by 
  sorry

end ratio_of_areas_l515_515822


namespace angle_B_value_l515_515601

theorem angle_B_value (A B C D : ℝ)
  (h1: ∠A = 3 * ∠D)
  (h2: ∠C = 4 * ∠B)
  (h3: ∠B + ∠C = 180) : ∠B = 36 := 
by
  sorry

end angle_B_value_l515_515601


namespace intersection_M_N_l515_515272

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l515_515272


namespace least_odd_prime_factor_2023_6_plus_1_is_13_l515_515480

noncomputable def least_odd_prime_factor_of_2023_6_plus_1 : ℕ := 
  13

theorem least_odd_prime_factor_2023_6_plus_1_is_13 :
  ∃ p, p.prime ∧ (p = least_odd_prime_factor_of_2023_6_plus_1) ∧ 
     (∀ q, q.prime ∧ q < p → 
       (q ≠ 1 ∨ q ∣ (2023^6 + 1)) ∨ 
       (2023^6 ≡ -1 [MOD q])) ∧
    (least_odd_prime_factor_of_2023_6_plus_1 ≡ 1 [MOD 12]) :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_is_13_l515_515480


namespace ad_value_l515_515160

variable (a b c d : ℝ)

-- Conditions
def geom_seq := b^2 = a * c ∧ c^2 = b * d
def vertex_of_parabola := (b = 1 ∧ c = 2)

-- Question
theorem ad_value (h_geom : geom_seq a b c d) (h_vertex : vertex_of_parabola b c) : a * d = 2 := by
  sorry

end ad_value_l515_515160


namespace bucket_water_weight_l515_515412

theorem bucket_water_weight (W : ℕ) (h1 : W % 900 = 200) (h2 : W / 900 = 7) : W / 1000 = 6.5 :=
by
    sorry

end bucket_water_weight_l515_515412


namespace determine_p_n_plus_1_l515_515983

noncomputable def p (n : ℕ) (x : ℝ) : ℝ := sorry

theorem determine_p_n_plus_1 (p : ℕ → ℝ → ℝ)
  (h_poly : ∀ n : ℕ, polynomial (p n))
  (h_conditions : ∀ (n k : ℕ), k ≤ n → p n k = k / (k + 1)) 
  (n : ℕ) : 
  p n (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) :=
by
  sorry

end determine_p_n_plus_1_l515_515983


namespace arithmetic_seq_common_diff_zero_geometric_seq_minimum_k_l515_515964

/- Part 1: Arithmetic sequence problem -/

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_common_diff_zero (a : ℕ → ℝ) (h1 : a 1 = 10)
  (h2 : ∀ n : ℕ, a n - 10 ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 10)
  (h3 : ∀ n : ℕ, let S := λ n, (Finset.range n).sum a in S n - 10 ≤ S (n + 1) ∧ S (n + 1) ≤ S n + 10)
  (ha : arithmetic_seq a) : ∃ d, d = 0 :=
begin
  sorry
end

/- Part 2: Geometric sequence problem -/

def geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_seq_minimum_k (b : ℕ → ℝ) (h1 : b 1 = 10)
  (h2 : b 2 ≤ 20)
  (h3 : ∃ k : ℕ, (Finset.range k).sum b > 2017)
  (hb : geometric_seq b) : ∃ k, k = 8 :=
begin
  sorry
end

end arithmetic_seq_common_diff_zero_geometric_seq_minimum_k_l515_515964


namespace FD_closest_to_4_l515_515619

noncomputable def parallelogram_ABCD_condition 
  (AB BC AD CD DE BE AF FD : ℝ)
  (angle_ABC angle_ADC : ℝ) : Prop :=
    angle_ABC = 100 ∧
    AB = 12 ∧
    BC = 8 ∧
    angle_ADC = 100 ∧
    DE = 6 ∧
    BE = sqrt (BC^2 + (BC + DE)^2 + 2 * BC * (BC + DE) * (Real.cos 100)) ∧
    ∃ k : ℝ, k = DE / (BC + DE) ∧
    k = 3 / 7 ∧
    FD = k * AD ∧
    AD = BC

theorem FD_closest_to_4 : 
  ∀ (AB BC AD CD DE BE AF FD : ℝ) (angle_ABC angle_ADC : ℝ), 
  parallelogram_ABCD_condition AB BC AD CD DE BE AF FD angle_ABC angle_ADC → 
  FD = 4 :=
by
  intros AB BC AD CD DE BE AF FD angle_ABC angle_ADC h
  obtain ⟨angle_ABC_eq, AB_eq, BC_eq, angle_ADC_eq, DE_eq, BE_eq, k, ratio_eq, k_eq, FD_eq, AD_eq⟩ := h
  sorry

end FD_closest_to_4_l515_515619


namespace train_speed_proof_l515_515400

noncomputable def speedOfTrain (lengthOfTrain : ℝ) (timeToCross : ℝ) (speedOfMan : ℝ) : ℝ :=
  let man_speed_m_per_s := speedOfMan * 1000 / 3600
  let relative_speed := lengthOfTrain / timeToCross
  let train_speed_m_per_s := relative_speed + man_speed_m_per_s
  train_speed_m_per_s * 3600 / 1000

theorem train_speed_proof :
  speedOfTrain 100 5.999520038396929 3 = 63 := by
  sorry

end train_speed_proof_l515_515400


namespace sum_first_10_common_elements_eq_l515_515122

/-- 
Find the sum of the first 10 elements that appear both among 
the terms of the arithmetic progression {4, 7, 10, 13, ...} 
and the geometric progression {20, 40, 80, 160, ...}.
-/
theorem sum_first_10_common_elements_eq :
  let a_n : ℕ → ℕ := λ n, 4 + 3 * n,
      b_k : ℕ → ℕ := λ k, 20 * 2^k,
      c_n : ℕ → ℕ := λ n, 40 * 4^(n-1),
      S : ℕ := ∑ n in finset.range 10, c_n n
  in S = 13981000 :=
by {
  -- The full proof would go here,
  -- but we are omitting it as per the instructions.
  sorry
}

end sum_first_10_common_elements_eq_l515_515122


namespace hexagon_area_trapezoid_l515_515640

theorem hexagon_area_trapezoid (AB CD BC DA : ℝ)
  (h_parallel : AB ∥ CD)
  (h_AB : AB = 13)
  (h_BC : BC = 8)
  (h_CD : CD = 23)
  (h_DA : DA = 10)
  (P and Q : Point)
  (h_P : P = intersect_bisectors ∠A ∠D)
  (h_Q : Q = intersect_bisectors ∠B ∠C) :
  area_hexagon ABQCDP = 36 * sqrt 3 :=
by
  sorry

end hexagon_area_trapezoid_l515_515640


namespace total_area_of_pyramid_faces_l515_515740

theorem total_area_of_pyramid_faces (b l : ℕ) (hb : b = 8) (hl : l = 10) : 
  let h : ℝ := Math.sqrt (l^2 - (b / 2)^2) in
  let A : ℝ := 1 / 2 * b * h in
  let T : ℝ := 4 * A in
  T = 32 * Math.sqrt 21 := by
  -- Definitions
  have b_val : (b : ℝ) = 8 := by exact_mod_cast hb
  have l_val : (l : ℝ) = 10 := by exact_mod_cast hl

  -- Calculations
  have h_val : h = Math.sqrt (l^2 - (b / 2)^2) := rfl
  have h_simplified : h = 2 * Math.sqrt 21 := by
    rw [h_val, l_val, b_val]
    norm_num
    simp

  have A_val : A = 1 / 2 * b * h := rfl
  simp_rw [A_val, h_simplified, b_val]
  norm_num

  have T_val : T = 4 * A := rfl
  simp_rw [T_val]
  norm_num

  -- Final proof
  sorry

end total_area_of_pyramid_faces_l515_515740


namespace smallest_possible_value_n_l515_515704

noncomputable def min_value_n : ℕ :=
  Inf { n : ℕ | ∃ (a b c d : ℕ), gcd (gcd (gcd a b) c) d = 88 ∧
  lcm (lcm (lcm a b) c) d = n ∧ (count_quadruplets a b c d 88 77_000)}

axiom quadruplets_gcd_lcm_count : ∀ (a b c d n : ℕ), (gcd (gcd (gcd a b) c) d = 88 ∧ lcm (lcm (lcm a b) c) d = n ∧
  count_quadruplets a b c d 88 77_000) → (n = 31680)

theorem smallest_possible_value_n : min_value_n = 31680 :=
  by
    apply Inf_def
    apply quadruplets_gcd_lcm_count
    sorry -- Proof details are omitted

end smallest_possible_value_n_l515_515704


namespace arctan_tan_75_sub_2_tan_30_eq_l515_515076

theorem arctan_tan_75_sub_2_tan_30_eq :
  arctan (tan (75 * real.pi / 180) - 2 * tan (30 * real.pi / 180)) * 180 / real.pi = 75 :=
sorry

end arctan_tan_75_sub_2_tan_30_eq_l515_515076


namespace binomial_coeffs_not_arithmetic_l515_515637

open Nat

theorem binomial_coeffs_not_arithmetic (n r : ℕ) (h : r + 3 ≤ n) :
  ¬ (C(n, r) + C(n, r+2) = 2 * C(n, r+1) ∧
     C(n, r+1) + C(n, r+3) = 2 * C(n, r+2)) :=
sorry

end binomial_coeffs_not_arithmetic_l515_515637


namespace difference_in_balance_l515_515449

def cedric_balance (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem difference_in_balance 
  (principal : ℝ) (cedric_rate : ℝ) (cedric_compoundings : ℕ) (daniel_rate : ℝ) (time_years : ℕ) :
  abs ((daniel_balance principal daniel_rate time_years) - (cedric_balance principal cedric_rate cedric_compoundings time_years)) = 21514 :=
by
  sorry

end difference_in_balance_l515_515449


namespace consecutive_numbers_probability_l515_515117

theorem consecutive_numbers_probability :
  let total_ways := Nat.choose 20 5
  let non_consecutive_ways := Nat.choose 16 5
  let probability_of_non_consecutive := (non_consecutive_ways : ℚ) / (total_ways : ℚ)
  let probability_of_consecutive := 1 - probability_of_non_consecutive
  probability_of_consecutive = 232 / 323 :=
by
  sorry

end consecutive_numbers_probability_l515_515117


namespace johns_drawings_l515_515607

theorem johns_drawings (total_pictures : ℕ) (back_pictures : ℕ) 
  (h1 : total_pictures = 15) (h2 : back_pictures = 9) : total_pictures - back_pictures = 6 := by
  -- proof goes here
  sorry

end johns_drawings_l515_515607


namespace interval_of_increase_l515_515863

noncomputable def log_base_0_5 (x : ℝ) := log x / log (0.5 : ℝ)
def quadratic_formula (x : ℝ) : ℝ := 5 + 4 * x - x^2
def vertex_of_parabola := 2
def domain := set.Ioo (-1 : ℝ) 5

theorem interval_of_increase :
  {x : ℝ | (2 < x) ∧ (x < 5)} = 
  {x : ℝ | (f : ℝ → ℝ), f x = log_base_0_5 (quadratic_formula x) ∧ (x ∈ domain)} :=
sorry

end interval_of_increase_l515_515863


namespace minimize_profit_l515_515414

/-- Conditions definitions --/
def cost_to_manufacture (r : ℝ) : ℝ := 0.8 * Real.pi * r^2
def profit_per_ml : ℝ := 0.2

/-- Volume of a sphere --/
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Profit per bottle as a function of radius --/
def profit_per_bottle (r : ℝ) : ℝ := profit_per_ml * volume_sphere(r) - cost_to_manufacture(r)

/-- The problem statement follows --/
theorem minimize_profit (r_min : ℝ) (h_max_radius : r_min ≤ 6) : 
  ∀ (r : ℝ), 0 < r → r ≤ 6 →
  profit_per_bottle(r) ≤ profit_per_bottle(r_min) := 
by {
  sorry
}

end minimize_profit_l515_515414


namespace sum_of_remaining_digits_l515_515857

theorem sum_of_remaining_digits (m n : ℕ) :
  let nums := (List.replicate 200 [2, 3, 4, 5]).join in -- Create the 800 digit number
  -- Sum of 800 digits (200 blocks of "2345"), which is 2800
  let total_sum := 200 * 14 in 
  -- The sum of the digits crossed out is 2800 - 2345 = 455
  let crossed_sum := total_sum - 2345 in 
  -- The sum of crossed out digits must be 455
  total_sum - (List.sum (nums.drop m).take (800 - m - n)) = 2345 →
  -- If m + n = 130, this property holds
  m + n = 130 :=
sorry

end sum_of_remaining_digits_l515_515857


namespace HG_eq_BE_l515_515441

-- Define the points and related conditions
variables {A B C D E F G H : Type} [AddGroup A] [AddGroup B] [AddGroup C]
          [AddGroup D] [AddGroup E] [AddGroup F] [AddGroup G] [AddGroup H]

-- AD is the median of triangle ABC
def is_median (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] : Prop :=
  segment D A = segment B A + segment C A

-- EG parallel to AB
def is_parallel (x y : Type) [AddGroup x] [AddGroup y] : Prop := sorry

-- Define the equivalence to express FH parallel to AC
def is_parallel_to_AC (F H A C : Type) [AddGroup F] [AddGroup H] [AddGroup A] [AddGroup C] : Prop :=
  sorry

-- To formally state the proof
theorem HG_eq_BE {ABC : Type} [AddGroup ABC]
                 (h_median : is_median A B C D)
                 (h_parallel_EG_AB : is_parallel E G → is_parallel A B)
                 (h_parallel_FH_AC : is_parallel_to_AC F H A C) :
  segment H G = segment B E :=
begin
  sorry
end

end HG_eq_BE_l515_515441


namespace probability_P_in_D2_given_D1_l515_515193

noncomputable def region1_area : ℝ := 16
noncomputable def region2_intersection_area : ℝ := π

theorem probability_P_in_D2_given_D1 : 
  (π / region1_area) = (π / 16) :=
by
  have h1 : region1_area = 16 := rfl
  have h2 : region2_intersection_area = π := rfl
  rw [h1, h2]
  exact rfl

#eval probability_P_in_D2_given_D1 -- Expected output: ∎

end probability_P_in_D2_given_D1_l515_515193


namespace minimum_value_l515_515643

theorem minimum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_condition : 2 * a + 3 * b = 1) :
  ∃ min_value : ℝ, min_value = 65 / 6 ∧ (∀ c d : ℝ, (0 < c) → (0 < d) → (2 * c + 3 * d = 1) → (1 / c + 1 / d ≥ min_value)) :=
sorry

end minimum_value_l515_515643


namespace combined_weight_l515_515603

def weight_in_tons := 3
def weight_in_pounds_per_ton := 2000
def weight_in_pounds := weight_in_tons * weight_in_pounds_per_ton
def donkey_weight_in_pounds := weight_in_pounds - (0.90 * weight_in_pounds)

theorem combined_weight :
  (weight_in_pounds + donkey_weight_in_pounds) = 6600 :=
by
  -- Proof goes here
  sorry

end combined_weight_l515_515603


namespace ratio_triangle_circle_l515_515818

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let A_triangle := (sqrt 3 / 4) * (3 * r)^2
  let A_circle := π * r^2
  A_triangle / A_circle

theorem ratio_triangle_circle (r : ℝ) (h_r : r > 0) :
  ratio_of_areas r = 9 * sqrt 3 / (4 * π) :=
by
  sorry

end ratio_triangle_circle_l515_515818


namespace angle_B_45_l515_515583

theorem angle_B_45 (A B a b : ℝ)
  (h1 : A + B = 90)
  (h2 : sin A / a = cos B / b)
  (h3 : a = b) : B = 45 :=
sorry

end angle_B_45_l515_515583


namespace highest_annual_profit_l515_515676

def g (n : ℕ) : ℝ := 80 / Real.sqrt (n + 1)
def annual_profit (n : ℕ) : ℝ := (10 + n) * (100 - g n) - 100 * n

theorem highest_annual_profit : ∃ n, n = 8 ∧ annual_profit n = 520 :=
by
  sorry

end highest_annual_profit_l515_515676


namespace positive_solutions_l515_515106

theorem positive_solutions (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔
  x = 1 ∨ x = 3 :=
by
  sorry

end positive_solutions_l515_515106


namespace pants_to_shirts_ratio_l515_515057

-- Conditions
def shirts : ℕ := 4
def total_clothes : ℕ := 16

-- Given P as the number of pants and S as the number of shorts
variable (P S : ℕ)

-- State the conditions as hypotheses
axiom shorts_half_pants : S = P / 2
axiom total_clothes_condition : 4 + P + S = 16

-- Question: Prove that the ratio of pants to shirts is 2
theorem pants_to_shirts_ratio : P = 2 * shirts :=
by {
  -- insert proof steps here
  sorry
}

end pants_to_shirts_ratio_l515_515057


namespace valid_parameterizations_l515_515686

def parametrize_line (x0 y0 dx dy : ℝ) (t : ℝ) : ℝ × ℝ :=
  (x0 + t * dx, y0 + t * dy)

def lies_on_line (x y : ℝ) : Prop :=
  y = 3 * x + 5

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem valid_parameterizations :
  let points := [((0, 5), (3, 1)), ((-5 / 3, 0), (-1, -3)), ((2, 11), (9, 3)),
                 ((3, -2), (1 / 3, 1)), ((-5, -10), (1 / 15, 1 / 5))]
  let valid_points := points.filter (λ p, lies_on_line p.1.1 p.1.2)
  let valid_directions := valid_points.filter (λ p, is_scalar_multiple p.2 (1, 3))
  valid_directions = [((-5 / 3, 0), (-1, -3)), ((2, 11), (9, 3)), ((-5, -10), (1 / 15, 1 / 5))] :=
by sorry

end valid_parameterizations_l515_515686


namespace hamiltonian_cycle_l515_515620

theorem hamiltonian_cycle (G : Type) [graph G] (V : set G) (E : set (G × G))
  (h_card : ∀ v : G, v ∈ V → ∃! n : ℕ, cardinal.mk V = n ∧ 3 ≤ n)
  (h_deg : ∀ v : G, v ∈ V → degree v ≥ cardinal.mk V / 2) :
  ∃ cycle : list G, (∀ v ∈ V, v ∈ cycle) ∧ (∀ i, i < length cycle - 1 → (cycle.nth i, cycle.nth (i + 1)) ∈ E) :=
sorry

end hamiltonian_cycle_l515_515620


namespace reduced_rates_apply_two_days_l515_515777

-- Definition of total hours in a week
def total_hours_in_week : ℕ := 7 * 24

-- Given fraction of the week with reduced rates
def reduced_rate_fraction : ℝ := 0.6428571428571429

-- Total hours covered by reduced rates
def reduced_rate_hours : ℝ := reduced_rate_fraction * total_hours_in_week

-- Hours per day with reduced rates on weekdays (8 p.m. to 8 a.m.)
def hours_weekday_night : ℕ := 12

-- Total weekdays with reduced rates
def total_weekdays : ℕ := 5

-- Total reduced rate hours on weekdays
def reduced_rate_hours_weekdays : ℕ := total_weekdays * hours_weekday_night

-- Remaining hours for 24 hour reduced rates
def remaining_reduced_rate_hours : ℝ := reduced_rate_hours - reduced_rate_hours_weekdays

-- Prove that the remaining reduced rate hours correspond to exactly 2 full days
theorem reduced_rates_apply_two_days : remaining_reduced_rate_hours = 2 * 24 := 
by
  sorry

end reduced_rates_apply_two_days_l515_515777


namespace masha_can_generate_all_integers_up_to_1093_l515_515227

theorem masha_can_generate_all_integers_up_to_1093 :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n → n ≤ 1093 → f n ∈ {k | ∃ (a b c d e f g : ℤ), a * 1 + b * 3 + c * 9 + d * 27 + e * 81 + f * 243 + g * 729 = k}) :=
sorry

end masha_can_generate_all_integers_up_to_1093_l515_515227


namespace haleigh_needs_46_leggings_l515_515551

-- Define the number of each type of animal
def num_dogs : ℕ := 4
def num_cats : ℕ := 3
def num_spiders : ℕ := 2
def num_parrot : ℕ := 1

-- Define the number of legs each type of animal has
def legs_dog : ℕ := 4
def legs_cat : ℕ := 4
def legs_spider : ℕ := 8
def legs_parrot : ℕ := 2

-- Define the total number of legs function
def total_leggings (d c s p : ℕ) (ld lc ls lp : ℕ) : ℕ :=
  d * ld + c * lc + s * ls + p * lp

-- The statement to be proven
theorem haleigh_needs_46_leggings : total_leggings num_dogs num_cats num_spiders num_parrot legs_dog legs_cat legs_spider legs_parrot = 46 := by
  sorry

end haleigh_needs_46_leggings_l515_515551


namespace erin_trolls_count_l515_515096

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l515_515096


namespace most_stable_performance_l515_515903

theorem most_stable_performance :
  ∀ (σ2_A σ2_B σ2_C σ2_D : ℝ), 
  σ2_A = 0.56 → 
  σ2_B = 0.78 → 
  σ2_C = 0.42 → 
  σ2_D = 0.63 → 
  σ2_C ≤ σ2_A ∧ σ2_C ≤ σ2_B ∧ σ2_C ≤ σ2_D :=
by
  intros σ2_A σ2_B σ2_C σ2_D hA hB hC hD
  sorry

end most_stable_performance_l515_515903


namespace range_y_over_x_plus_1_l515_515521

theorem range_y_over_x_plus_1 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  ∃ δ, -real.sqrt 2 / 2 ≤ δ ∧ δ ≤ real.sqrt 2 / 2 ∧ δ = y / (x + 1) :=
sorry

end range_y_over_x_plus_1_l515_515521


namespace three_digit_palindromes_difference_l515_515341

theorem three_digit_palindromes_difference :
  ∃ (a b : ℕ), a * b = 589845 ∧ 
  (100 ≤ a ∧ a ≤ 999) ∧ (100 ≤ b ∧ b ≤ 999) ∧
  (∀ n : ℕ, n = a ∨ n = b → (to_string n).reverse = to_string n) ∧
  abs (a - b) = 304 :=
sorry

end three_digit_palindromes_difference_l515_515341


namespace day_of_week_2010_l515_515293

theorem day_of_week_2010 (h : ∀ n, (n % 7 = 0 → n = 7 * (n / 7)) := true) : 
  let days_between_2000_2010 := 2555 + 1098
  let day_of_week_2000 := 6 -- Saturday is the 6th day of the week
  let day_of_week_2010 := (day_of_week_2000 + (days_between_2000_2010 % 7)) % 7
  day_of_week_2010 = 0 := -- 0 represents Sunday in this case
by {
  have days_between_2000_2010_eq : days_between_2000_2010 = 3653 := by sorry,
  have mod_result_eq : (3653 % 7) = 1 := by sorry,
  have result_eq : ((6 + 1) % 7) = 0 := by sorry,
  show 0 = 0,
  exact eq.refl 0,
}

end day_of_week_2010_l515_515293


namespace startingNumberInRange_l515_515416

/-- Defining a function to calculate the sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Condition: The difference between a number and the sum of its digits is divisible by 9. -/
def divisibleBy9 (n : ℕ) : Prop :=
  (n - sumOfDigits n) % 9 = 0

/-- Given a natural number n, we subtract the sum of its digits successively 11 times. -/
def iterateSubtractDigits (n : ℕ) : ℕ :=
  (λ m, m - sumOfDigits m)^[11] n

theorem startingNumberInRange (n : ℕ)
  (h : iterateSubtractDigits n = 0) :
  100 ≤ n ∧ n ≤ 109 := 
sorry

end startingNumberInRange_l515_515416


namespace chocolates_bought_in_a_month_l515_515609

theorem chocolates_bought_in_a_month :
  ∀ (chocolates_for_her: ℕ)
    (chocolates_for_sister: ℕ)
    (chocolates_for_charlie: ℕ)
    (weeks_in_a_month: ℕ), 
  weeks_in_a_month = 4 →
  chocolates_for_her = 2 →
  chocolates_for_sister = 1 →
  chocolates_for_charlie = 10 →
  (chocolates_for_her * weeks_in_a_month + chocolates_for_sister * weeks_in_a_month + chocolates_for_charlie) = 22 :=
by
  intros chocolates_for_her chocolates_for_sister chocolates_for_charlie weeks_in_a_month
  intros h_weeks h_her h_sister h_charlie
  sorry

end chocolates_bought_in_a_month_l515_515609


namespace odd_function_iff_l515_515769

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x * abs (x + a) + b

theorem odd_function_iff (a b : α) : 
  (∀ x : α, f a b (-x) = -f a b x) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end odd_function_iff_l515_515769


namespace ratio_triangle_circle_l515_515819

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let A_triangle := (sqrt 3 / 4) * (3 * r)^2
  let A_circle := π * r^2
  A_triangle / A_circle

theorem ratio_triangle_circle (r : ℝ) (h_r : r > 0) :
  ratio_of_areas r = 9 * sqrt 3 / (4 * π) :=
by
  sorry

end ratio_triangle_circle_l515_515819


namespace change_received_l515_515650

theorem change_received (oranges : ℕ) (cost_per_orange : ℝ) (amount_paid : ℝ) : 
  oranges = 5 → 
  cost_per_orange = 0.30 → 
  amount_paid = 10.00 → 
  amount_paid - (oranges * cost_per_orange) = 8.50 :=
by
  intros h_oranges h_cost h_paid
  rw [h_oranges, h_cost, h_paid]
  norm_num
  sorry

end change_received_l515_515650


namespace milk_volume_in_ounces_l515_515028

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end milk_volume_in_ounces_l515_515028


namespace acute_scalene_never_isosceles_l515_515926

theorem acute_scalene_never_isosceles (T : Triangle) (h_acute : T.is_acute_angled) (h_scalene : T.is_scalene)
    (cut_along_median : Triangle → Triangle × Triangle) :
  ¬ (∀ T' : set Triangle, (∀ t ∈ T', ∃ (T1 T2 : Triangle), cut_along_median t = (T1, T2) → T1.is_isosceles ∧ T2.is_isosceles) :=
  sorry

end acute_scalene_never_isosceles_l515_515926


namespace find_a_l515_515943

noncomputable def center_radius_circle1 (x y : ℝ) := x^2 + y^2 = 16
noncomputable def center_radius_circle2 (x y a : ℝ) := (x - a)^2 + y^2 = 1
def centers_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

theorem find_a (a : ℝ) (h1 : center_radius_circle1 x y) (h2 : center_radius_circle2 x y a) : centers_tangent a :=
sorry

end find_a_l515_515943


namespace original_candies_shelly_had_l515_515361

-- Let's define the number of candies Shelly originally had in the bowl.
def candies_in_bowl_before_friend_came (C : ℕ) : Prop :=
  ∃ C : ℕ,
  let friend_candies_before_eating := 95 in -- Her friend had 85 candies after eating 10, so 95 before eating.
  let friend_brought := 2 * C in           -- Her friend brought twice as much candy as Shelly originally had.
  let total_candies := C + friend_brought in -- Total candies after her friend adds her candies.
  let divided_candies_each := 95 in        -- Each of them ended up with 95 candies after dividing equally.
  total_candies = 2 * divided_candies_each -- Total candies divided equally means 3C = 190.

-- The theorem that we want to prove
theorem original_candies_shelly_had : candies_in_bowl_before_friend_came 63 :=
begin
  -- Proof goes here
  sorry
end

end original_candies_shelly_had_l515_515361


namespace turtles_still_on_sand_l515_515040

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l515_515040


namespace maria_total_percentage_correct_l515_515278

theorem maria_total_percentage_correct :
  let num_correct_test1 := 0.75 * 32,
      num_correct_test2 := 0.85 * 20,
      num_correct_test3 := 0.80 * 16,
      total_correct := num_correct_test1 + num_correct_test2 + num_correct_test3,
      total_problems := 32 + 20 + 16,
      percentage_correct := total_correct / total_problems * 100 in
  percentage_correct ≈ 79 := 
sorry -- Proof omitted

end maria_total_percentage_correct_l515_515278


namespace minSubsets_l515_515265

open Set

-- Definitions for the set S and the family of sets A
def S: Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

-- Using options until elaboration on types
variable (n : ℕ)
variable (A : Fin n → Set ℕ)

-- Constraints
def condition1 := ∀ i, i < n → A i ⊆ S ∧ |A i| = 7
def condition2 := ∀ i j, i < j ∧ j < n → |A i ∩ A j| ≤ 3
def condition3 := ∀ M, M ⊆ S ∧ |M| = 3 → ∃ k, k < n ∧ M ⊆ A k

-- The theorem statement
theorem minSubsets : ∃ n, condition1 n A ∧ condition2 n A ∧ condition3 n A ∧ n = 15 := 
sorry

end minSubsets_l515_515265


namespace nested_fraction_equality_l515_515524

variables (m n : ℕ)

def is_irreducible (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def nested_fraction (k : ℕ) : ℚ :=
  Nat.recOn k (1 : ℚ) (λ n f, 1 / (1 + f))

theorem nested_fraction_equality
  (hn : is_irreducible m n)
  (heq : nested_fraction 1990 = m / n) :
  (1/2 + m / n) ^ 2 = 5 / 4 - 1 / n ^ 2 :=
sorry

end nested_fraction_equality_l515_515524


namespace general_term_sum_first_n_terms_l515_515519

-- Define the arithmetic sequence
def arith_sequence (a1 d : ℕ) (n : ℕ) : ℕ := a1 + n * d

-- Initial conditions
axiom a2_is_4 : arith_sequence a1 d 2 = 4
axiom a6_a8_is_18 : arith_sequence a1 d 6 + arith_sequence a1 d 8 = 18

-- The first theorem: find the general term
theorem general_term (a1 d : ℕ) (n : ℕ) : (arith_sequence 3 1 n) = n + 2 := by
  sorry

-- The second theorem: sum of the first n terms of the sequence {1/(na_n)}
noncomputable def a (n : ℕ) : ℚ := n + 2

noncomputable def seq (n : ℕ) : ℚ := 1 / (n * a n)

noncomputable def sum_seq (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, seq (k + 1))

theorem sum_first_n_terms (n : ℕ) :
    sum_seq n = 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2)) := by
  sorry

end general_term_sum_first_n_terms_l515_515519


namespace floor_inequality_l515_515296

theorem floor_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (Real.floor (5 * x) + Real.floor (5 * y)) ≥ (Real.floor (3 * x + y) + Real.floor (3 * y + x)) :=
by
  -- proof goes here
  sorry

end floor_inequality_l515_515296


namespace triangular_faces_area_of_pyramid_l515_515733

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l515_515733


namespace arithmetic_geometric_sequences_l515_515939

variable {R : Type*} [OrderedCommRing R] {a b : R}
variable {n : ℕ}
variables (x : ℕ → R) (y : ℕ → R)

def isArithmeticSeq (a b : R) (x : ℕ → R) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → x i = a + i * (b - a) / (n + 1)

def isGeometricSeq (a b : R) (y : ℕ → R) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → y i = a * (b / a) ^ (i / (n + 1))

theorem arithmetic_geometric_sequences (hab : a ≠ b) (ha : 0 < a) (hb : 0 < b)
  (arith : isArithmeticSeq a b x n) (geom : isGeometricSeq a b y n) :
  (1 / n * ∑ i in finset.range n, x (i + 1) > real.sqrt (a * b) + (real.sqrt a - real.sqrt b) ^ 2 / 4) ∧
  (1 / n * ∑ i in finset.range n, x (i + 1) > (a + b) / 2) ∧
  (real.sqrt (∏ i in finset.range n, y (i + 1)) < real.sqrt (a * b)) ∧
  (real.sqrt (∏ i in finset.range n, y (i + 1)) < (a + b) / 2 - (real.sqrt a - real.sqrt b) ^ 2 / 4) := sorry

end arithmetic_geometric_sequences_l515_515939


namespace no_intersection_abs_eq_l515_515974

theorem no_intersection_abs_eq (x : ℝ) : ∀ y : ℝ, y = |3 * x + 6| → y = -|2 * x - 4| → false := 
by
  sorry

end no_intersection_abs_eq_l515_515974


namespace abs_inequality_solution_set_l515_515699

theorem abs_inequality_solution_set :
  { x : ℝ | abs (2 - x) < 5 } = { x : ℝ | -3 < x ∧ x < 7 } :=
by
  sorry

end abs_inequality_solution_set_l515_515699


namespace tangent_line_to_curve_l515_515577

theorem tangent_line_to_curve {k : ℝ} (h : ∃ m n : ℝ, n = k * m ∧ n = m - exp m ∧ k = 1 - exp m) : k = 1 - exp 1 :=
by
  obtain ⟨m, n, h1, h2, h3⟩ := h
  have hm : m = 1, from sorry
  have hk : k = 1 - exp 1, from sorry
  exact hk

end tangent_line_to_curve_l515_515577


namespace max_magical_pairs_1_to_30_l515_515024

def is_magical_pair (a b : ℕ) : Prop :=
  (a + b) % 7 = 0

def max_magical_pairs (l : List ℕ) : ℕ :=
  l.zip l.tail |>.filter (λ (p : ℕ × ℕ), is_magical_pair p.1 p.2) |>.length

theorem max_magical_pairs_1_to_30 : 
  ∃ l : List ℕ, (∀ x ∈ l, 1 ≤ x ∧ x ≤ 30) ∧ l.nodup ∧ l.length = 30 ∧ max_magical_pairs l = 26 := 
sorry

end max_magical_pairs_1_to_30_l515_515024


namespace least_odd_prime_factor_of_2023_power6_plus_1_l515_515477

theorem least_odd_prime_factor_of_2023_power6_plus_1 :
  ∃ (p : ℕ), prime p ∧ odd p ∧ p ∣ (2023^6 + 1) ∧ p = 13 :=
begin
  sorry
end

end least_odd_prime_factor_of_2023_power6_plus_1_l515_515477


namespace least_odd_prime_factor_2023_6_plus_1_is_13_l515_515479

noncomputable def least_odd_prime_factor_of_2023_6_plus_1 : ℕ := 
  13

theorem least_odd_prime_factor_2023_6_plus_1_is_13 :
  ∃ p, p.prime ∧ (p = least_odd_prime_factor_of_2023_6_plus_1) ∧ 
     (∀ q, q.prime ∧ q < p → 
       (q ≠ 1 ∨ q ∣ (2023^6 + 1)) ∨ 
       (2023^6 ≡ -1 [MOD q])) ∧
    (least_odd_prime_factor_of_2023_6_plus_1 ≡ 1 [MOD 12]) :=
by
  sorry

end least_odd_prime_factor_2023_6_plus_1_is_13_l515_515479


namespace inequality_solution_set_l515_515120

theorem inequality_solution_set : {x : ℝ | |x + 1| + |x - 2| ≤ 5} = Icc (-2 : ℝ) 3 :=
by
  sorry

end inequality_solution_set_l515_515120


namespace difference_of_squares_36_l515_515346

theorem difference_of_squares_36 {x y : ℕ} (h₁ : x + y = 18) (h₂ : x * y = 80) (h₃ : x > y) : x^2 - y^2 = 36 :=
by
  sorry

end difference_of_squares_36_l515_515346


namespace find_m_value_l515_515221

-- Condition: P(-m^2, 3) lies on the axis of symmetry of the parabola y^2 = mx
def point_on_axis_of_symmetry (m : ℝ) : Prop :=
  let P := (-m^2, 3)
  let axis_of_symmetry := (-m / 4)
  P.1 = axis_of_symmetry

theorem find_m_value (m : ℝ) (h : point_on_axis_of_symmetry m) : m = 1 / 4 :=
  sorry

end find_m_value_l515_515221


namespace sheet_metal_required_is_lateral_surface_area_l515_515064

-- Definitions based on the problem statement
def isCylindricalVentilationDuct (duct : Type) := duct -- Assuming a placeholder type for duct

-- Proof statement to verify that the required metal sheet area is the lateral surface area
theorem sheet_metal_required_is_lateral_surface_area (duct : Type) (h : isCylindricalVentilationDuct duct) :
  ¬(requiredSheetMetalArea = cylinderVolume duct) := sorry

end sheet_metal_required_is_lateral_surface_area_l515_515064


namespace product_ge_power_l515_515639

theorem product_ge_power (n : ℕ) (k : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, 0 < x i) 
  (h2 : ∑ i, x i ^ k / (1 + x i ^ k) = n - 1) 
  (h3 : 3 ≤ n) 
  (h4 : 2 ≤ k) 
  (h5 : k ≤ n) : 
  ∏ i, x i ≥ (n - 1) ^ (n / k) := 
by
  sorry

end product_ge_power_l515_515639


namespace bike_ride_hours_energetic_l515_515444

-- Define the conditions
def hours_energetic (x : ℚ) : Prop :=
  let hours_fatigue := 9 - x - 1 in
  let distance := 25 * x + 15 * hours_fatigue in
  distance = 150

-- The proof problem statement
theorem bike_ride_hours_energetic (x : ℚ) :
  hours_energetic x → x = 3 := 
by 
  sorry

end bike_ride_hours_energetic_l515_515444


namespace find_z2_l515_515946

noncomputable def z1 : ℂ := 2 - complex.I

noncomputable def satisfies_condition (z : ℂ) : Prop :=
  (z - 2) * (1 + complex.I) = 1 - complex.I

def z2_imaginary_part (z : ℂ) : Prop :=
  z.im = 2

def product_is_real (z1 z2 : ℂ) : Prop :=
  (z1 * z2).im = 0

theorem find_z2 (z2 : ℂ) : 
  satisfies_condition z1 ∧ 
  z2_imaginary_part z2 ∧ 
  product_is_real z1 z2 → 
  z2 = 4 + 2 * complex.I :=
by 
  sorry

end find_z2_l515_515946


namespace asymptotes_of_C2_l515_515136

theorem asymptotes_of_C2 (a b : ℝ) (x y : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) -> True)) 
  (h4 : (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) -> True)) 
  (h5 : (sqrt (a^2 - b^2) / a * sqrt (a^2 + b^2) / a = sqrt 3 / 2)) :
  (x = sqrt 2 * y ∨ x = - sqrt 2 * y) :=
sorry

end asymptotes_of_C2_l515_515136


namespace randy_spent_10_l515_515302

def randy_spending (x : ℝ) : Prop := 
  30 - x - (1 / 4) * (30 - x) = 15

theorem randy_spent_10 : ∃ x, randy_spending x ∧ x = 10 := 
by
  constructor
  exists 10
  rw [randy_spending]
  sorry

end randy_spent_10_l515_515302


namespace sum_of_first_10_common_elements_l515_515123

open Nat

def arithmetic_progression (n : ℕ) : ℤ := 4 + 3 * n

def geometric_progression (k : ℕ) : ℤ := 20 * 2^k

def is_common_element (x : ℤ) : Prop :=
  ∃ n k : ℕ, x = arithmetic_progression n ∧ x = geometric_progression k

def common_elements (n : ℕ) : ℕ → list ℤ
| 0     := []
| (m+1) := match (list.find_x is_common_element (geometric_progression n)) with 
  | some x => x :: common_elements (n+1) m
  | none   => common_elements (n+1) (m+1)
  end

def sum_common_elements (n : ℕ) : ℤ :=
  (common_elements 0 n).sum
  
theorem sum_of_first_10_common_elements :
  sum_common_elements 10 = 13981000 := 
sorry

end sum_of_first_10_common_elements_l515_515123


namespace count_three_digit_integers_with_conditions_l515_515554

theorem count_three_digit_integers_with_conditions :
  ∃ (count : ℕ), count = 22 ∧ 
    ∀ n : ℕ, (100 ≤ n ∧ n < 1000) →
    (n % 7 = 3 ∧ n % 8 = 4 ∧ n % 10 = 6) ↔ (n.exists (k : ℕ), 3 ≤ k ∧ k ≤ 24 ∧ n = 280k + 166) :=
sorry

end count_three_digit_integers_with_conditions_l515_515554


namespace abc_divides_sum_pow_31_l515_515616

theorem abc_divides_sum_pow_31 (a b c : ℕ) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) : 
  abc ∣ (a + b + c) ^ 31 := 
sorry

end abc_divides_sum_pow_31_l515_515616


namespace biased_coin_probability_l515_515380

theorem biased_coin_probability (p : ℚ) 
  (h : nat.choose 4 1 * p * (1 - p)^3 = nat.choose 4 2 * p^2 * (1 - p)^2) :
  let probability := nat.choose 4 2 * (p)^2 * (1 - p)^2 in
  ∃ i j k : ℕ, ((↑i) * (↑j) = probability.denom * (probability.num .bor))
  ∧ (i + j + k = 841) := sorry

end biased_coin_probability_l515_515380


namespace fraction_of_field_planted_l515_515587

noncomputable def planted_fraction_of_field : ℚ := 
  let leg1 := 5
  let leg2 := 12
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let square_side := 39 / 169 -- From solving the quadratic equation derived in the solution.
  let total_area := (leg1 * leg2) / 2
  let square_area : ℚ := (square_side)^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem fraction_of_field_planted (h_leg1 : ∃ a : ℚ, a = 5)
                                  (h_leg2 : ∃ b : ℚ, b = 12)
                                  (h_distance_square_to_hypotenuse : ∃ d : ℚ, d = 3) :
  planted_fraction_of_field = 85611 / 85683 :=
by 
  sorry

end fraction_of_field_planted_l515_515587


namespace good_coloring_count_l515_515140

theorem good_coloring_count (n : ℕ) : 
  ∃ (c : ℕ → ℕ), 
  (c 1 = 5) ∧ 
  (c 2 = 13) ∧ 
  (∀ n, c (n+2) = 2 * c (n+1) + 3 * c n) ∧ 
  (c n = (3^(n+1) + (-1)^(n+1)) / 2) :=
sorry

end good_coloring_count_l515_515140


namespace sum_integers_50_to_75_l515_515725

theorem sum_integers_50_to_75 : (Finset.range 26).sum (λ i, 50 + i) = 1625 :=
by
  sorry

end sum_integers_50_to_75_l515_515725


namespace probability_participates_on_Wednesday_given_Tuesday_l515_515402

-- Define the universe of days from Monday to Friday.
inductive Day
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day

-- Define the events A and B
def participates_on (d : Day) (days : List Day) : Prop :=
  d ∈ days

theorem probability_participates_on_Wednesday_given_Tuesday (days : List Day) 
  (h_len : days.length = 2) 
  (h_Tuesday : participates_on Day.Tuesday days) : 
  (elems := [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]) 
  (p_Wednesday_given_Tuesday : ℚ) :=
  p_Wednesday_given_Tuesday = 1 / 4 :=
begin
  sorry
end

end probability_participates_on_Wednesday_given_Tuesday_l515_515402


namespace cos_double_angle_l515_515904

theorem cos_double_angle
  (θ : ℝ)
  (h1 : sin θ + cos θ = 1 / 5)
  (h2 : π / 2 ≤ θ ∧ θ ≤ 3 * π / 4) :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l515_515904


namespace smallest_n_unique_k_l515_515722

theorem smallest_n_unique_k (n : ℕ) (h1 : n > 0) (h2 : (∃ k : ℤ, (9 : ℚ) / 17 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (10 : ℚ) / 19)) : 
    n = 10 :=
begin
  -- Prove that there is no smaller n and that n = 10 is valid
  sorry
end

end smallest_n_unique_k_l515_515722


namespace domain_f1_f2_expr_eq_f2_eval_at_3_l515_515399

-- Definition of the function for Problem 1
def f1 (x : ℝ) : ℝ := (sqrt (4 - 2 * x)) + 1 + (1 / (x + 1))

-- Problem 1: Prove the domain of the function
theorem domain_f1 : {x : ℝ | 4 - 2 * x ≥ 0 ∧ x ≠ 1 ∧ x ≠ -1} = (-∞, -1) ∪ (-1, 1) ∪ (1, 2] := sorry

-- Definition of the function for Problem 2
def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Problem 2: Prove the function expression and its value at 3
theorem f2_expr_eq (x : ℝ) : f2 x = x^2 - 4 * x + 3 := 
  by sorry

theorem f2_eval_at_3 : f2 3 = 0 := 
  by sorry

end domain_f1_f2_expr_eq_f2_eval_at_3_l515_515399


namespace arctan_tan_75_minus_2_tan_30_eq_75_l515_515079

theorem arctan_tan_75_minus_2_tan_30_eq_75 : 
  arctan (tan 75 - 2 * tan 30) = 75 :=
by
  sorry

end arctan_tan_75_minus_2_tan_30_eq_75_l515_515079


namespace polynomial_equivalence_l515_515271

variable (a t x : ℝ)

def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x + a

def M (a t : ℝ) : ℝ :=
  if t < a then f a t else f a a

theorem polynomial_equivalence (a : ℝ) : M a (a - 1) + M a (a + 2) = 2 * a^2 + 2 * a - 1 :=
by
  sorry

end polynomial_equivalence_l515_515271


namespace rationalize_denominator_l515_515662

theorem rationalize_denominator :
  ∃ A B C D E : ℤ,
  B < D ∧
  (5 : ℚ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
  A + B + C + D + E = 20 :=
by
  use [-4, 7, 3, 13, 1]
  split
  · exact lt_of_lt_of_le (Int.lt_add_one_iff.mpr zero_lt_succ) (le_refl 13)
  split
  · have : (4 * Real.sqrt 7 - 3 * Real.sqrt 13) * (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
          (4 * Real.sqrt 7) ^ 2 - (3 * Real.sqrt 13) ^ 2 :=
      (Real.mul_add_distrib _ _ _).symm
    simp only [Real.sqrt_mul_self zero_lt_seven, Real.sqrt_mul_self zero_lt_thirteen] at this
    field_simp [this]
  rfl

end rationalize_denominator_l515_515662


namespace rachel_win_probability_is_63_div_64_l515_515308

-- Defining the initial setup and conditions
def initial_setup : (ℕ × ℕ) := (0, 0)
def rachel_initial_setup : (ℕ × ℕ) := (6, 8)

-- Defining the movement constraints
def sarah_move (pos : ℕ × ℕ) : set (ℕ × ℕ) :=
  { (x + 1, y) | (x, y) = pos } ∪ { (x, y + 1) | (x, y) = pos }

def rachel_move (pos : ℕ × ℕ) : set (ℕ × ℕ) :=
  { (x - 1, y) | (x, y) = pos } ∪ { (x, y - 1) | (x, y) = pos }

-- The win conditions
def sarah_catches_rachel (sarah_pos rachel_pos : ℕ × ℕ) : Prop :=
  sarah_pos = rachel_pos

def rachel_reaches_origin (rachel_pos : ℕ × ℕ) : Prop :=
  rachel_pos = (0, 0)

-- Parameters for optimal play
constant play_optimally : Prop

-- The probability calculation under optimal conditions
def prob_rachel_wins : ℚ :=
  63 / 64

-- The theorem statement:
theorem rachel_win_probability_is_63_div_64 :
  play_optimally →
  let sarah_pos := initial_setup in
  let rachel_pos := rachel_initial_setup in
  let rachel_wins : Prop :=
    sarah_catches_rachel sarah_pos rachel_pos → false ∧
    rachel_reaches_origin rachel_pos → true in
  ∃ p : ℚ, p = prob_rachel_wins :=
sorry

end rachel_win_probability_is_63_div_64_l515_515308


namespace sum_50_to_75_l515_515724

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end sum_50_to_75_l515_515724


namespace product_of_positive_integers_for_distinct_real_roots_l515_515457

theorem product_of_positive_integers_for_distinct_real_roots :
  ∀ c : ℕ, (10 * c + 24 * c + c > 0) → (∏ i in Finset.range 15, i) = 87178291200 := by
  sorry

end product_of_positive_integers_for_distinct_real_roots_l515_515457


namespace smallest_n_l515_515118

theorem smallest_n (h₁ : ∀ (a b c d : ℕ), gcd(gcd(a, b), gcd(c, d)) = 105 → 
                        lcm(lcm(a, b), lcm(c, d)) = 37800 → 
                        (∃! x : ℕ, x = 79000)) 
    : ∃ n, n = 37800 :=
by
  sorry

end smallest_n_l515_515118


namespace carrie_pays_94_l515_515840

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l515_515840


namespace batches_needed_nina_tom_bake_l515_515687

theorem batches_needed (students : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
(extra_students : ℕ) : ℕ :=
  let total_students := students + extra_students
  let total_cookies := total_students * cookies_per_student
  let batches := total_cookies / cookies_per_batch
  if total_cookies % cookies_per_batch = 0 then 
    batches 
  else 
    batches + 1

theorem nina_tom_bake : batches_needed 90 3 20 15 = 16 := by
  simp [batches_needed]
  sorry

end batches_needed_nina_tom_bake_l515_515687


namespace area_circumference_correct_l515_515107

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -4, y := 5 }
def B : Point := { x := 10, y := -2 }

-- Define the distance formula
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the radius of the circle passing through A and B
def radius : ℝ :=
  distance A B

-- Define the area of the circle
def area : ℝ :=
  Real.pi * (radius^2)

-- Define the circumference of the circle
def circumference : ℝ :=
  2 * Real.pi * radius

-- State the proof problem
theorem area_circumference_correct :
  area = 245 * Real.pi ∧ circumference = 2 * Real.pi * Real.sqrt 245 :=
by
  sorry

end area_circumference_correct_l515_515107


namespace angle_difference_proof_l515_515584

-- Define the angles A and B
def angle_A : ℝ := 65
def angle_B : ℝ := 180 - angle_A

-- Define the difference
def angle_difference : ℝ := angle_B - angle_A

theorem angle_difference_proof : angle_difference = 50 :=
by
  -- The proof goes here
  sorry

end angle_difference_proof_l515_515584


namespace rectangle_area_l515_515663

theorem rectangle_area
  (r s t : Type)
  [linear_ordered_field r]
  [comm_ring s]
  [linear_ordered_field t]
  (A B C D E F G : s) 
  (EG HF : t) 
  (AD AB : r) 
  (hEG : EG = 15)
  (hHF : HF = 12)
  (hAD_eq_3AB : AD = 3 * AB)
  (area_of_rect : AB * AD = 10800 / 289) :
  AB * AD = 10800 / 289 :=
by
  sorry

end rectangle_area_l515_515663


namespace arithmetic_progression_sum_l515_515927

variable {α : Type*} [LinearOrderedField α]

def arithmetic_progression (S : ℕ → α) :=
  ∃ (a d : α), ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_progression_sum :
  ∀ (S : ℕ → α),
  arithmetic_progression S →
  (S 4) / (S 8) = 1 / 7 →
  (S 12) / (S 4) = 43 :=
by
  intros S h_arith_prog h_ratio
  sorry

end arithmetic_progression_sum_l515_515927


namespace parabola_equation_l515_515108

-- Define the given conditions
def vertex : ℝ × ℝ := (3, 5)
def point_on_parabola : ℝ × ℝ := (4, 2)

-- Prove that the equation is as specified
theorem parabola_equation :
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x y : ℝ, (y = a * x^2 + b * x + c) ↔
     (y = -3 * x^2 + 18 * x - 22) ∧ (vertex.snd = -3 * (vertex.fst - 3)^2 + 5) ∧
     (point_on_parabola.snd = a * point_on_parabola.fst^2 + b * point_on_parabola.fst + c)) := 
sorry

end parabola_equation_l515_515108


namespace fractional_part_inequality_l515_515250

noncomputable def fractional_part (x : ℝ) : ℝ := x - real.floor x

theorem fractional_part_inequality (q : ℕ) (h_q : ¬ ∃ k : ℕ, q = k^3) :
  ∃ C > 0, ∀ n : ℕ, fractional_part (n * real.cbrt q) + fractional_part (n * real.cbrt (q ^ 2)) ≥ C * n^(-1/2:ℝ) :=
begin
  sorry
end

end fractional_part_inequality_l515_515250


namespace impossible_a_values_l515_515578

theorem impossible_a_values (a : ℝ) :
  ¬((1-a)^2 + (1+a)^2 < 4) → (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end impossible_a_values_l515_515578


namespace rectangular_box_proof_l515_515349

noncomputable def rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) : ℝ :=
2 * (a * b + b * c + c * a)

theorem rectangular_box_proof
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) :
  rectangular_box_surface_area a b c h1 h2 = 784 :=
by
  sorry

end rectangular_box_proof_l515_515349


namespace reduced_rates_end_at_8am_l515_515011

theorem reduced_rates_end_at_8am : 
  ∀ (fraction : ℝ) (weekend_hours : ℕ) (weekday_hours_evening : ℕ), 
    fraction = 0.6428571428571429 → 
    weekend_hours = 48 → 
    weekday_hours_evening = 4 → 
    let total_week_hours := 168 in
    let total_reduced_hours := total_week_hours * fraction in
    let weekday_reduced_hours := total_reduced_hours - weekend_hours in
    let weekday_count := 5 in
    let weekday_hours_total := weekday_reduced_hours / weekday_count in
    let morning_extra_hours := weekday_hours_total - weekday_hours_evening in
    morning_extra_hours = 8 :=
by
  intros fraction weekend_hours weekday_hours_evening 
         fraction_eq weekend_hours_eq weekday_hours_evening_eq
  let total_week_hours := 168
  let total_reduced_hours := total_week_hours * fraction
  let weekday_reduced_hours := total_reduced_hours - weekend_hours
  let weekday_count := 5
  let weekday_hours_total := weekday_reduced_hours / weekday_count
  let morning_extra_hours := weekday_hours_total - weekday_hours_evening
  sorry

end reduced_rates_end_at_8am_l515_515011


namespace jerry_field_hours_l515_515243

theorem jerry_field_hours :
  ∀ (daughters games_per_daughter hours_per_game practice_hours_per_game : ℕ),
    daughters = 4 →
    games_per_daughter = 12 →
    hours_per_game = 3 →
    practice_hours_per_game = 6 →
    (daughters * games_per_daughter * hours_per_game + daughters * games_per_daughter * practice_hours_per_game) = 432 :=
by
  intros daughters games_per_daughter hours_per_game practice_hours_per_game
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end jerry_field_hours_l515_515243


namespace pentagon_angle_T_l515_515998

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l515_515998


namespace flag_arrangement_remainder_l515_515705

theorem flag_arrangement_remainder : 
  let total_flags := 20
  let blue_flags := 10
  let green_flags := 10
  let valid_arrangements (N : ℕ) := 
    N = 9 * (nat.choose (blue_flags + 1) green_flags) - 2 * nat.choose (blue_flags + 1) green_flags
  valid_arrangements N → N % 1000 = 77 :=
by {
  let N := 9 * (nat.choose (blue_flags + 1) green_flags) - 2 * nat.choose (blue_flags + 1) green_flags;
  show N % 1000 = 77, 
  sorry
}

end flag_arrangement_remainder_l515_515705


namespace find_number_l515_515657

statement:

theorem find_number (N : ℝ) (h : (1 / 4 * 1 / 3 * 2 / 5 * N - 1 / 2 * 1 / 6 * N = 35)) : 0.40 * N = -280 := 
by 
  sorry

end find_number_l515_515657


namespace range_of_fx_less_than_zero_l515_515574

theorem range_of_fx_less_than_zero (f : ℝ → ℝ) 
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_monotonic : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x ≤ y → f y ≤ f x)
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x < 0} = set.Ioo (-1) 1 := 
begin
  sorry
end

end range_of_fx_less_than_zero_l515_515574


namespace ratio_of_areas_l515_515824

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l515_515824


namespace verify_problem_l515_515366

noncomputable def problem_data : List ℕ := [86, 82, 90, 99, 98, 96, 90, 100, 89, 83, 87, 88, 81, 90, 93, 100, 96, 100, 92, 100]
def sorted_data : List ℕ := List.qsort (· < ·) problem_data

def a : ℕ := List.countp (fun x => 90 <= x ∧ x < 95) problem_data
def b : ℚ := (sorted_data.get! 9 + sorted_data.get! 10) / 2  -- Assuming List is 0-indexed
def c : ℕ := problem_data.mode

def estimate_students := 2700 * (problem_data.countp (fun x => x >= 90)) / problem_data.length

theorem verify_problem :
  a = 5 ∧ b = 91 ∧ c = 100 ∧ estimate_students = 1755 := by
  sorry

end verify_problem_l515_515366


namespace initial_shirts_claimed_l515_515611

theorem initial_shirts_claimed:
  ∀ (pairs_trousers : ℕ) (cost_per_trouser : ℕ) (total_bill : ℕ) (cost_per_shirt : ℕ) (missing_shirts : ℕ),
  pairs_trousers = 10 →
  cost_per_trouser = 9 →
  total_bill = 140 →
  cost_per_shirt = 5 →
  missing_shirts = 8 →
  let total_cost_trousers := pairs_trousers * cost_per_trouser in
  let total_cost_shirts := total_bill - total_cost_trousers in
  let actual_number_shirts := total_cost_shirts / cost_per_shirt in
  actual_number_shirts - missing_shirts = 2 :=
by
  intros pairs_trousers cost_per_trouser total_bill cost_per_shirt missing_shirts 
  assume h_pairs triv h_trousers triv h_total triv h_shirt triv h_missing triv
  let total_cost_trousers := pairs_trousers * cost_per_trouser
  let total_cost_shirts := total_bill - total_cost_trousers
  let actual_number_shirts := total_cost_shirts / cost_per_shirt
  exact calc
    actual_number_shirts - missing_shirts = 2 : sorry

end initial_shirts_claimed_l515_515611


namespace sin_x_value_angle_sum_value_l515_515004

-- Problem 1
theorem sin_x_value (
  x : ℝ
) (
  h1 : cos (x - π / 4) = sqrt 2 / 10
) (
  h2 : x ∈ Set.Ioo (π / 2) (3 * π / 4)
) : sin x = 4 / 5 := sorry

-- Problem 2
theorem angle_sum_value (
  A B : ℝ
) (
  h1 : A < π / 2 ∧ A > 0
) (
  h2 : B < π / 2 ∧ B > 0
) (
  h3 : sin A = sqrt 5 / 5
) (
  h4 : sin B = sqrt 10 / 10
) : A + B = π / 4 := sorry

end sin_x_value_angle_sum_value_l515_515004


namespace longest_segment_in_cylinder_l515_515406

noncomputable def cylinder_diagonal (radius height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

theorem longest_segment_in_cylinder :
  cylinder_diagonal 4 10 = 2 * Real.sqrt 41 :=
by
  -- Proof placeholder
  sorry

end longest_segment_in_cylinder_l515_515406


namespace max_sin_prod_l515_515984

theorem max_sin_prod (x : Fin 2012 → ℝ) 
  (h : ∏ i, tan (x i) = 1) : 
  (∏ i, sin (x i)) ≤ 1 / 2^1006 := 
sorry

end max_sin_prod_l515_515984


namespace marble_arrangement_problem_l515_515093

def max_yellow_marbles (blue yellow : Nat) : Nat :=
  (blue - 1) + (blue - 1)  -- 2 * (blue - 1)

def num_ways (total_marbles boxes : Nat) : Nat :=
  Nat.choose (total_marbles + boxes - 1) (boxes - 1)

theorem marble_arrangement_problem :
  let blue := 6
  let yellow := max_yellow_marbles blue
  let total := blue + yellow
  let boxes := blue + 1
  let N := num_ways total boxes
  N % 1000 = 824 := sorry

end marble_arrangement_problem_l515_515093


namespace hyperbola_standard_equation_l515_515332

-- Definitions based on the provided conditions
def ellipse_focal_length : ℝ := sqrt (9 - 4)

def hyperbola_same_focal_length (c : ℝ) : Prop :=
  c = ellipse_focal_length

def hyperbola_asymptote : Prop :=
  ∃ a b : ℝ, a/b = 2

-- Main theorem statement
theorem hyperbola_standard_equation :
  ∀ (c : ℝ), hyperbola_same_focal_length c → 
  hyperbola_asymptote → 
  ( ∃ (λ : ℝ), (λ = 1 ∧ (λ > 0 → (∀ x y : ℝ, x^2 / 4 - y^2 = λ))) ∨ 
                 (λ = -1 ∧ (λ < 0 → (∀ x y : ℝ, y^2 - x^2 / 4 = λ))) ) :=
by
  intros c h1 h2
  use λ
  sorry

end hyperbola_standard_equation_l515_515332


namespace convert_base_7_to_binary_l515_515860

theorem convert_base_7_to_binary : 
  let n := 25
  let base7 := 2 * 7^1 + 5 * 7^0
  let base10 := 19
  let binary := 10011
  n = 25 → 
  base7 = base10 →
  base10 = 19 → 
  binary = 10011 → 
  convert_base_7_to_binary : Prop := base10.baseToBinary = binary :=
sorry

end convert_base_7_to_binary_l515_515860


namespace rectangle_side_ratios_l515_515717

noncomputable def possible_ratios : Set ℝ :=
  { sqrt 30 / 1, sqrt 30 / 2, sqrt 30 / 3, sqrt 30 / 5, sqrt 30 / 6, sqrt 30 / 10, sqrt 30 / 15, sqrt 30 / 30 }

theorem rectangle_side_ratios (a b : ℝ) (h : a > 0 ∧ b > 0) (congruent_rectangles : ∀ k, k = 30) :
  ∃ r, r ∈ possible_ratios ∧ (a / b = r) :=
by
  sorry

end rectangle_side_ratios_l515_515717


namespace obtuse_triangle_side_range_l515_515937

theorem obtuse_triangle_side_range (a : ℝ) (h1 : 0 < a)
  (h2 : a + (a + 1) > a + 2)
  (h3 : (a + 1) + (a + 2) > a)
  (h4 : (a + 2) + a > a + 1)
  (h5 : (a + 2)^2 > a^2 + (a + 1)^2) : 1 < a ∧ a < 3 :=
by
  -- proof omitted
  sorry

end obtuse_triangle_side_range_l515_515937


namespace alice_additional_cookies_proof_l515_515813

variable (alice_initial_cookies : ℕ)
variable (bob_initial_cookies : ℕ)
variable (cookies_thrown_away : ℕ)
variable (bob_additional_cookies : ℕ)
variable (total_edible_cookies : ℕ)

theorem alice_additional_cookies_proof 
    (h1 : alice_initial_cookies = 74)
    (h2 : bob_initial_cookies = 7)
    (h3 : cookies_thrown_away = 29)
    (h4 : bob_additional_cookies = 36)
    (h5 : total_edible_cookies = 93) :
  alice_initial_cookies + bob_initial_cookies - cookies_thrown_away + bob_additional_cookies + (93 - (74 + 7 - 29 + 36)) = total_edible_cookies :=
by
  sorry

end alice_additional_cookies_proof_l515_515813


namespace parallel_lines_have_equal_slopes_l515_515529

theorem parallel_lines_have_equal_slopes (m : ℝ) :
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → m = -1 / 2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l515_515529


namespace problem1_problem2_l515_515540

noncomputable def f (x: ℝ) : ℝ := abs (2 * x - 1) + 2

noncomputable def g (x: ℝ) : ℝ := -abs (x + 2) + 3

theorem problem1 (x : ℝ) : g(x) ≥ -2 → -7 ≤ x ∧ x ≤ 3 := by
  sorry

theorem problem2 (m : ℝ) : (∀ x : ℝ, f(x) - g(x) ≥ m + 2) → m ≤ -1 / 2 := by
  sorry

end problem1_problem2_l515_515540


namespace mia_money_l515_515287

def darwin_has := 45
def mia_has (d : ℕ) := 2 * d + 20

theorem mia_money : mia_has darwin_has = 110 :=
by
  unfold mia_has darwin_has
  rw [←nat.mul_assoc]
  rw [nat.mul_comm 2 45]
  sorry

end mia_money_l515_515287


namespace second_number_is_72_l515_515789

-- Define the necessary variables and conditions
variables (x y : ℕ)
variables (h_first_num : x = 48)
variables (h_ratio : 48 / 8 = x / y)
variables (h_LCM : Nat.lcm x y = 432)

-- State the problem as a theorem
theorem second_number_is_72 : y = 72 :=
by
  sorry

end second_number_is_72_l515_515789


namespace queue_problem_l515_515715

open Nat

def alwaysLies (n : ℕ) : Prop := odd n

theorem queue_problem :
  ∃ (S : Fin 25 → Prop),
  (∀ i : Fin 25, 
    (if i = 0 then
       ∀ j : Fin 25, j > i → alwaysLies j.index
     else
       S i → alwaysLies (Nat.pred i).index) 
    ∧ (∃ i : Fin 25, S i = true)
  ) ∧ (∃ L : ℕ, L = 13) :=
by {
  sorry
}

end queue_problem_l515_515715


namespace cyclic_inequality_l515_515893

theorem cyclic_inequality 
  (x y z : ℝ) 
  (h1x : 0 < x) 
  (h1y : 0 < y) 
  (h1z : 0 < z) 
  (hcond : x * y * z + x * y + y * z + z * x = 4):
  (sqrt ((x * y + x + y) / z) + sqrt ((y * z + y + z) / x) + sqrt ((z * x + z + x) / y)) ≥
  (3 * sqrt (3 * (x + 2) * (y + 2) * (z + 2) / ((2 * x + 1) * (2 * y + 1) * (2 * z + 1)))) :=
by 
  sorry

end cyclic_inequality_l515_515893


namespace contest_score_order_l515_515991

variables (E F G H : ℕ) -- nonnegative scores of Emily, Fran, Gina, and Harry respectively

-- Conditions
axiom cond1 : E - F = G + H + 10
axiom cond2 : G + E > F + H + 5
axiom cond3 : H = F + 8

-- Statement to prove
theorem contest_score_order : (H > E) ∧ (E > F) ∧ (F > G) :=
sorry

end contest_score_order_l515_515991


namespace canonical_equation_of_line_l515_515383

def point_on_line (x₀ y₀ z₀ : ℝ) : Prop :=
  x₀ + 5 * y₀ - z₀ - 5 = 0 ∧ 2 * x₀ - 5 * y₀ + 2 * z₀ + 5 = 0

noncomputable def line_canonical_equation : Prop :=
  ∀ (x y z : ℝ),
    (x, y, z) = (0, 1, 0) ∨ (∃ t : ℝ, (x, y, z) = (0, 1, 0) + t * (5, -4, -15)) ↔
    x / 5 = (y - 1) / -4 = z / -15

theorem canonical_equation_of_line :
  point_on_line 0 1 0 → line_canonical_equation :=
begin
  intros,
  sorry
end

end canonical_equation_of_line_l515_515383


namespace pyramid_total_area_l515_515730

theorem pyramid_total_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (base_edge_eq : base_edge = 8)
  (lateral_edge_eq : lateral_edge = 10)
  : 4 * (1 / 2 * base_edge * sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 32 * sqrt 21 := by
  sorry

end pyramid_total_area_l515_515730


namespace invertible_2x2_matrix_l515_515109

open Matrix

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ

def A : matrix_2x2 := ![![4, 5], ![-2, 9]]

def inv_A : matrix_2x2 := ![![9/46, -5/46], ![1/23, 2/23]]

theorem invertible_2x2_matrix :
  det A ≠ 0 → (inv A = inv_A) := 
by
  sorry

end invertible_2x2_matrix_l515_515109


namespace problem_statement_l515_515087

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ a b : ℝ, a ∈ s → b ∈ s → a < b → f a < f b

noncomputable def f := λ x : ℝ, x ^ 3

theorem problem_statement : is_odd_function f ∧ is_monotonically_increasing f {x : ℝ | 0 < x} :=
by sorry

end problem_statement_l515_515087


namespace total_miles_hiked_l515_515464

theorem total_miles_hiked : 
  (a b c d e : ℝ) (ha : a = 3.8) (hb : b = 1.75) (hc : c = 2.3) (hd : d = 0.45) (he : e = 1.92) :
  a + b + c + d + e = 10.22 :=
by
  rw [ha, hb, hc, hd, he]
  sorry

end total_miles_hiked_l515_515464


namespace positive_integer_solutions_count_3x_plus_4y_eq_1024_l515_515339

theorem positive_integer_solutions_count_3x_plus_4y_eq_1024 :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = 1024) ∧ 
  (∀ n, n = 85 → ∃! (s : ℕ × ℕ), s.fst > 0 ∧ s.snd > 0 ∧ 3 * s.fst + 4 * s.snd = 1024 ∧ n = 85) := 
sorry

end positive_integer_solutions_count_3x_plus_4y_eq_1024_l515_515339


namespace two_y_minus_three_x_l515_515582

variable (x y : ℝ)

noncomputable def x_val : ℝ := 1.2 * 98
noncomputable def y_val : ℝ := 0.9 * (x_val + 35)

theorem two_y_minus_three_x : 2 * y_val - 3 * x_val = -78.12 := by
  sorry

end two_y_minus_three_x_l515_515582


namespace quadratic_equation_properties_l515_515515

theorem quadratic_equation_properties (m : ℝ) (h : m < 4) (root_one : ℝ) (root_two : ℝ) 
  (eq1 : root_one + root_two = 4) (eq2 : root_one * root_two = m) (root_one_eq : root_one = -1) :
  m = -5 ∧ root_two = 5 ∧ (root_one ≠ root_two) :=
by
  -- Sorry is added to skip the proof because only the statement is needed.
  sorry

end quadratic_equation_properties_l515_515515


namespace f_is_constant_l515_515638

noncomputable def f : ℝ → ℝ
def t : ℝ

axiom f_zero : f 0 = 1/2
axiom f_eqn : ∀ x y : ℝ, f (x + y) = f x * f (t - y) + f y * f (t - x)

theorem f_is_constant : ∀ x : ℝ, f x = 1/2 := 
by
  sorry

end f_is_constant_l515_515638


namespace arc_length_correct_l515_515834

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, real.sqrt (1 + (deriv f x)^2)

noncomputable def function_y (x : ℝ) : ℝ := (real.exp (2 * x) + real.exp (-2 * x) + 3) / 4

theorem arc_length_correct :
  arc_length function_y 0 2 = (1/2) * (real.exp 4 - real.exp (-4)) :=
sorry

end arc_length_correct_l515_515834


namespace mia_has_110_l515_515280

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l515_515280


namespace no_m_for_P_eq_S_m_le_3_for_P_implies_S_l515_515508

namespace ProofProblem

def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def S (m x : ℝ) : Prop := |x - 1| ≤ m

theorem no_m_for_P_eq_S : ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S m x := sorry

theorem m_le_3_for_P_implies_S : ∀ (m : ℝ), (m ≤ 3) → (∀ x, S m x → P x) := sorry

end ProofProblem

end no_m_for_P_eq_S_m_le_3_for_P_implies_S_l515_515508


namespace num_consecutive_sets_l515_515116

theorem num_consecutive_sets (S : ℕ) (hS : S = 156) : 
  (∃ n a, n ≥ 2 ∧ a ≥ 1 ∧ n * (2 * a + n - 1) = 2 * S) ↔ 2 := 
by
  sorry

end num_consecutive_sets_l515_515116


namespace max_magical_pairs_1_to_30_l515_515023

def is_magical_pair (a b : ℕ) : Prop :=
  (a + b) % 7 = 0

def max_magical_pairs (l : List ℕ) : ℕ :=
  l.zip l.tail |>.filter (λ (p : ℕ × ℕ), is_magical_pair p.1 p.2) |>.length

theorem max_magical_pairs_1_to_30 : 
  ∃ l : List ℕ, (∀ x ∈ l, 1 ≤ x ∧ x ≤ 30) ∧ l.nodup ∧ l.length = 30 ∧ max_magical_pairs l = 26 := 
sorry

end max_magical_pairs_1_to_30_l515_515023


namespace amoeba_growth_l515_515008

theorem amoeba_growth : ∀ (n : ℕ), (3 ^ n = 59049) → (n = 10) :=
by {
  intro n,
  intros h,
  omega,
  sorry
}

end amoeba_growth_l515_515008


namespace number_verification_l515_515210

noncomputable def a : ℝ := 2.23606797749979

theorem number_verification :
  ∃ a : ℝ, ((a + a)^2 - (a - a)^2 = 20.000000000000004) ∧ (a = 2.23606797749979) :=
by
  use a
  split
  { -- first part of split: proving ((a + a)^2 - (a - a)^2 = 20.000000000000004) 
    sorry },
  { -- second part of split: proving a = 2.23606797749979
    exact rfl }

end number_verification_l515_515210


namespace part_i_part_ii_l515_515539

noncomputable def f (x a : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Part I: when a = 1, find the intervals of monotonicity for f(x)
theorem part_i :
  ( ∀ x, f x 1 = x - 1 - 2 * Real.log x) ∧
  ( ∀ x, 0 < x ∧ x ≤ 2 → deriv (λ x, f x 1) x < 0 ) ∧
  ( ∀ x, 2 ≤ x → deriv (λ x, f x 1) x > 0 ) :=
sorry

-- Part II: If the function f(x) has no zeros in the interval (0, 1/3), find the range of a
theorem part_ii :
  ( ∀ x ∈ Ioo (0 : ℝ) (1/3 : ℝ), ∀ a, f x a ≠ 0 ) ↔ 
  ( a ∈ Icc (2 - 3 * Real.log 3) +∞):=
sorry

end part_i_part_ii_l515_515539


namespace max_boats_race_l515_515423

theorem max_boats_race (w_river : ℕ) (w_boat : ℕ) (space_side : ℕ) (w_river = 42) (w_boat = 3) (space_side = 2) : 
  (w_river - 2 * space_side) / (w_boat + 2 * space_side) = 5 := 
by
  sorry

end max_boats_race_l515_515423


namespace find_a_l515_515173

-- Definitions of the circles and given condition
def circle1 (a : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

def circle2 (a : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + a * p.2 - 6 = 0}

def common_chord_length : ℝ :=
  2 * sqrt 3

-- Problem statement
theorem find_a (a : ℝ) (h1 : true) : a = 2 ∨ a = -2 :=
sorry

end find_a_l515_515173


namespace fourth_vertex_of_parallelogram_l515_515951

theorem fourth_vertex_of_parallelogram :
  ∃ (x y : ℤ), ((x = 5 ∧ y = -5) ∨ (x = -3 ∧ y = -5) ∨ (x = 1 ∧ y = 5)) ∧
  (x, y) = fourth_vertex ⟨-1, 0⟩ ⟨3, 0⟩ ⟨1, -5⟩ :=
sorry

end fourth_vertex_of_parallelogram_l515_515951


namespace num_turtles_on_sand_l515_515036

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l515_515036


namespace counterexamples_prime_numbers_l515_515864

theorem counterexamples_prime_numbers :
  ∃ (N : ℕ) (digits : List ℕ), (digits.sum = 9) ∧ (∀ d ∈ digits, 0 < d ∧ d ≤ 5) ∧ (count_non_prime N = 7) :=
begin
  sorry
end

end counterexamples_prime_numbers_l515_515864


namespace money_made_per_minute_l515_515320

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l515_515320


namespace semicircle_arcs_perimeter_l515_515032

theorem semicircle_arcs_perimeter (s : ℝ) (h : s = Real.sqrt 2) 
    (arc_length : ∀ {d : ℝ}, d = s → 0.5 * π * d) :
  4 * (arc_length (d := s) h / 2) = 2 * π * Real.sqrt 2 := 
  by 
    have d := s
    have arc_len := arc_length (d := s) h
    sorry

end semicircle_arcs_perimeter_l515_515032


namespace triangle_and_semicircle_ratio_l515_515589

noncomputable def ratio_of_areas (x : ℝ) : ℝ :=
let AB : ℝ := 2 * x,
    BC : ℝ := 3 * x,
    AC : ℝ := 4 * x,
    
    -- Cosine of angle BCA
    cos_BCA : ℝ := (AB^2 + BC^2 - AC^2) / (2 * AB * BC),

    -- Sine of angle BCA
    sin_BCA : ℝ := real.sqrt(1 - cos_BCA^2),

    -- Area of the triangle ABC
    area_triangle : ℝ := (1/2) * BC * AC * sin_BCA,

    -- Radius of the semicircle
    radius_semicircle : ℝ := AC / 2,

    -- Area of the semicircle
    area_semicircle : ℝ := (1/2) * real.pi * radius_semicircle^2
in
    
-- Ratio of the area of the semicircle to the area of the triangle
area_semicircle / area_triangle

theorem triangle_and_semicircle_ratio (x : ℝ) (h : x > 0) : ratio_of_areas x = real.pi * real.sqrt 15 / 12 :=
by
  sorry

end triangle_and_semicircle_ratio_l515_515589


namespace max_subset_size_l515_515632

-- Define the set M
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1995 }

-- Define the property of subset A
def valid_subset (A : Set ℕ) : Prop :=
  ∀ x ∈ A, 15 * x ∉ A

-- Define the maximum possible size of such subset
theorem max_subset_size : ∃ A : Set ℕ, A ⊆ M ∧ valid_subset A ∧ Finset.card (A.to_finset) = 1870 :=
sorry

end max_subset_size_l515_515632


namespace sin_double_angle_l515_515948

-- Define a point in 2D, assuming the standard Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- The terminal side of angle intersects at P(1, 2)
def P : Point := { x := 1, y := 2 }

-- The definition of sin 2α based on the given problem
theorem sin_double_angle (α : ℝ) (h : P.x^2 + P.y^2 = 5) : 2 * (P.y / (real.sqrt 5)) * (P.x / (real.sqrt 5)) = 4 / 5 :=
sorry

end sin_double_angle_l515_515948


namespace approximation_correct_l515_515827

def approximate_to_place (n : ℝ) : String :=
  if n = 4.02 then "hundredth place" else "undefined"

theorem approximation_correct :
  approximate_to_place 4.02 = "hundredth place" :=
by
  sorry

end approximation_correct_l515_515827


namespace company_KW_price_percentage_more_l515_515849

-- Defining parameters
variable {A B P : ℝ}

-- Conditions from the problem
def condition1 : Prop := P = 1.30 * A
def condition2 : Prop := P = 0.7878787878787878 * (A + B)

-- The main theorem statement
theorem company_KW_price_percentage_more (h1 : condition1) (h2 : condition2) : P = 2 * B :=
by
  -- Begin a proof block
  sorry

end company_KW_price_percentage_more_l515_515849


namespace identify_ideal_functions_l515_515575

-- Definition of an "ideal function" based on the given conditions
def is_ideal_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧ (∀ x1 x2, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0)

-- Problem statement: Prove that the second and fourth functions are "ideal functions"
theorem identify_ideal_functions :
  is_ideal_function (λ x:ℝ, -x^3) ∧ 
  is_ideal_function (λ x:ℝ, if x ≥ 0 then -x^2 else x^2) :=
by sorry

end identify_ideal_functions_l515_515575


namespace triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l515_515344

-- Definition of the sides according to Plato's rule
def triangle_sides (p : ℕ) : ℕ × ℕ × ℕ :=
  (2 * p, p^2 - 1, p^2 + 1)

-- Function to check if the given sides form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Theorems to verify the sides of the triangle for given p values
theorem triangle_sides_p2 : triangle_sides 2 = (4, 3, 5) ∧ is_right_triangle 4 3 5 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p3 : triangle_sides 3 = (6, 8, 10) ∧ is_right_triangle 6 8 10 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p4 : triangle_sides 4 = (8, 15, 17) ∧ is_right_triangle 8 15 17 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p5 : triangle_sides 5 = (10, 24, 26) ∧ is_right_triangle 10 24 26 :=
by {
  sorry -- Proof goes here
}

end triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l515_515344


namespace carrie_pays_94_l515_515839

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l515_515839


namespace e_n_max_value_l515_515257

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end e_n_max_value_l515_515257


namespace nail_trimming_sound_count_l515_515987

def nails_per_customer : ℕ := 20
def number_of_customers : ℕ := 6
def total_sounds : ℕ := nails_per_customer * number_of_customers

theorem nail_trimming_sound_count : total_sounds = 120 :=
by
  simp [total_sounds, nails_per_customer, number_of_customers]
  sorry

end nail_trimming_sound_count_l515_515987


namespace tiling_2x12_l515_515978

def d : Nat → Nat
| 0     => 0  -- Unused but for safety in function definition
| 1     => 1
| 2     => 2
| (n+1) => d n + d (n-1)

theorem tiling_2x12 : d 12 = 233 := by
  sorry

end tiling_2x12_l515_515978


namespace math_proof_problem_l515_515165

-- Define the function f and the condition for its extreme value point
def f (x : ℝ) (b : ℝ) : ℝ := 2 * x + b / x + Real.log x

-- Condition: x = 1 is an extreme value point of f
def is_extreme_value_point (x : ℝ) (f : ℝ → ℝ) : Prop :=
  has_deriv_at f (0 : ℝ) x

-- Goal 1: prove b = 3 given the condition
def prove_b_eq_3 : Prop :=
  ∃ b : ℝ, (is_extreme_value_point 1 (λ x, 2 * x + b / x + Real.log x)) → (b = 3)

-- Define the function g and the monotonicity condition
def g (x : ℝ) (a : ℝ) : ℝ := f(x) 3 - (3 + a) / x

-- g(x) is monotonically increasing in [1, 2]
def is_monotonically_increasing (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x ≤ y → g x ≤ g y

-- The interval [1, 2]
def interval : Set ℝ := set.Icc (1 : ℝ) (2 : ℝ)

-- Goal 2: prove the range of values for a is [-3, +∞) given the condition
def prove_range_of_a : Prop :=
  is_monotonically_increasing (λ x, f x 3 - (3 + a) / x) interval → ∀ a : ℝ, a ≥ -3

-- Combining both goals
def final_proof_problem : Prop :=
  prove_b_eq_3 ∧ prove_range_of_a

-- Statement only, no proof needed.
theorem math_proof_problem : final_proof_problem :=
  sorry

end math_proof_problem_l515_515165


namespace cube_painting_distinct_count_l515_515784

theorem cube_painting_distinct_count :
  (∃ (cubes : Fin 24 → Fin 6 → Fin 3),
    (∀ i, cubes i 0 = 1) ∧ (∃ j, cubes j 0 = 0) ∧ (∃ k, cubes k 1 = 1) ∧ (∃ l, cubes l 2 = 2)) :=
begin
  sorry
end

end cube_painting_distinct_count_l515_515784


namespace min_value_of_sum_of_reciprocals_l515_515263

open Real

theorem min_value_of_sum_of_reciprocals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  ∃ t : ℝ, t = add_reciprocals x y z ∧ t = 27 / 8 :=
by
  let add_reciprocals x y z := (1 / (x + 3 * y)) + (1 / (y + 3 * z)) + (1 / (z + 3 * x))
  use (add_reciprocals x y z)
  split
  {
    sorry -- Proving t = add_reciprocals x y z
  }
  {
    sorry -- Proving t = 27 / 8
  }

end min_value_of_sum_of_reciprocals_l515_515263


namespace utilities_cost_l515_515019

theorem utilities_cost
    (rent1 : ℝ) (utility1 : ℝ) (rent2 : ℝ) (utility2 : ℝ)
    (distance1 : ℝ) (distance2 : ℝ) 
    (cost_per_mile : ℝ) 
    (drive_days : ℝ) (cost_difference : ℝ)
    (h1 : rent1 = 800)
    (h2 : rent2 = 900)
    (h3 : utility2 = 200)
    (h4 : distance1 = 31)
    (h5 : distance2 = 21)
    (h6 : cost_per_mile = 0.58)
    (h7 : drive_days = 20)
    (h8 : cost_difference = 76)
    : utility1 = 259.60 := 
by
  sorry

end utilities_cost_l515_515019


namespace sum_of_geometric_sequence_l515_515176

/--
Given a geometric sequence {a_n} with a_1 = 1 and common ratio q = 2,
if the sum of the terms from the m-th to the n-th term is 112 (m < n),
then m + n = 12.
-/
theorem sum_of_geometric_sequence {a : ℕ → ℕ} (m n : ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ k, a (k + 1) = 2 * a k) 
  (h₃ : m < n) 
  (h₄ : (∑ k in finset.range (n - m + 1), a (m + k)) = 112) :
  m + n = 12 := 
sorry

end sum_of_geometric_sequence_l515_515176


namespace _l515_515641

-- Define the main objects and points in the problem.
variables (A B C D E F G H I : Type) [noncomputable] [inst: linear_ordered_field ℝ]

-- Assume these points are on a circumcircle \Gamma of an acute triangle ABC
def on_circumcircle (P : Type) := true 

-- Definitions of perpendicular lines and angle bisectors
def perpendicular (L M : Type) := true
def angle_bisector (P Q : Type) := true
def meets_at (L M : Type) (P : Type) := true

-- Given conditions as hypotheses
noncomputable def conditions : Prop :=
∀ (Γ : Type) (A B C D E F G H I : Type)
    [circumcircle : on_circumcircle Γ],
  (perpendicular C AB) → 
  (meets_at (perpendicular C AB) AB D) →
  (meets_at (perpendicular C AB) Γ E) →
  (angle_bisector Γ C) →
  (meets_at (angle_bisector Γ C) AB F) →
  (meets_at (angle_bisector Γ C) Γ G) →
  (meets_at GD Γ H) →
  (meets_at HF Γ I) →
  true

-- The main theorem to prove
noncomputable theorem AI_eq_EB : conditions A B C D E F G H I → ∀ [inst: linear_ordered_field ℝ], AI = EB := 
sorry

end _l515_515641


namespace sum_of_three_circles_l515_515814

theorem sum_of_three_circles :
  ∃ (a b : ℤ), 3 * a + 2 * b = 21 ∧ 2 * a + 3 * b = 19 ∧ 3 * b = 9 := 
by
  use (2 : ℤ) -- Placeholder for triangle variable a
  use (3 : ℤ) -- Placeholder for circle variable b
  split
  { -- proving 3a + 2b = 21
    exact (by norm_num) }
  split
  { -- proving 2a + 3b = 19
    exact (by norm_num) }
  { -- proving 3b = 9
    exact (by norm_num) }
  sorry

end sum_of_three_circles_l515_515814


namespace problem_statement_l515_515636

theorem problem_statement (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (g x + y) = g (x ^ 2 + y) + 4 * g x * y) :
  let m := 2 in let t := 4 in m * t = 8 :=
by
  let m := 2
  let t := 4
  show m * t = 8
  sorry

end problem_statement_l515_515636


namespace carrie_pays_l515_515847

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l515_515847


namespace omar_remaining_coffee_l515_515655

noncomputable def remaining_coffee : ℝ := 
  let initial_coffee := 12
  let after_first_drink := initial_coffee - (initial_coffee * 1/4)
  let after_office_drink := after_first_drink - (after_first_drink * 1/3)
  let espresso_in_ounces := 75 / 29.57
  let after_espresso := after_office_drink + espresso_in_ounces
  let after_lunch_drink := after_espresso - (after_espresso * 0.75)
  let iced_tea_addition := 4 * 1/2
  let after_iced_tea := after_lunch_drink + iced_tea_addition
  let after_cold_drink := after_iced_tea - (after_iced_tea * 0.6)
  after_cold_drink

theorem omar_remaining_coffee : remaining_coffee = 1.654 :=
by 
  sorry

end omar_remaining_coffee_l515_515655


namespace find_n_l515_515150

-- Define an arithmetic sequence and the relevant conditions
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

theorem find_n
    (a : ℕ → ℝ)
    (h_arith : is_arithmetic a)
    (h_a2 : a 1 = 2)
    (h_S_diff : ∀ n > 3, S a n - S a (n - 3) = 54)
    (h_Sn : S a 9 = 100) : 
  9 = 10 := 
sorry

end find_n_l515_515150


namespace exponent_equality_l515_515557

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l515_515557


namespace slope_range_of_intersecting_line_and_circle_l515_515145

theorem slope_range_of_intersecting_line_and_circle (k : ℝ) :
  (∃ (x y : ℝ), y = k*x + 2*k ∧ x^2 + y^2 = 2 * x) →
  - √2 / 4 < k ∧ k < √2 / 4 :=
sorry

end slope_range_of_intersecting_line_and_circle_l515_515145


namespace minimum_distance_on_parabola_l515_515397

-- Define the points and the parabola
def point (R : Type*) := prod R R
def A : point ℝ := (1, 0)
def B : point ℝ := (5, 5)

-- Define the parabola y^2 = 4x
def is_on_parabola (P : point ℝ) : Prop := (P.snd)^2 = 4 * P.fst

-- Define the distance function
def dist (P Q : point ℝ) : ℝ := real.sqrt ((Q.fst - P.fst)^2 + (Q.snd - P.snd)^2)

-- Define the minimum value of AP + BP
def min_value_AP_BP : ℝ := 6

-- The statement we need to prove
theorem minimum_distance_on_parabola (P : point ℝ)
  (hP_on_parabola : is_on_parabola P) : dist A P + dist B P ≥ min_value_AP_BP :=
sorry

end minimum_distance_on_parabola_l515_515397


namespace car_travel_distance_in_30_min_l515_515972

theorem car_travel_distance_in_30_min :
  ∀ {train_speed car_speed : ℝ} (h1 : train_speed = 96)
    (h2 : car_speed = (5/8) * train_speed),
    (car_speed * (1/2)) = 30 :=
by
  intros train_speed car_speed h1 h2
  rw [h1, h2]
  norm_num
  sorry

end car_travel_distance_in_30_min_l515_515972


namespace equilateral_triangle_area_from_hexagon_l515_515796

theorem equilateral_triangle_area_from_hexagon (side_length : ℝ) (h : side_length = 3) :
  (∃ hexagon : ℕ → ℤ×ℤ, 
    (∀ (i : ℕ), 0 ≤ i ∧ i < 6 → dist (hexagon (i)) (hexagon (i + 1)) = side_length) ∧
    dist (hexagon 0) (hexagon 2) = side_length ∧
    dist (hexagon 2) (hexagon 4) = side_length ∧
    ∃ triangle : ℕ → ℤ×ℤ, 
      triangle 0 = hexagon 0 ∧
      triangle 1 = hexagon 2 ∧
      triangle 2 = hexagon 4 ∧
      (dist (triangle 0) (triangle 1) = side_length ∧
      dist (triangle 1) (triangle 2) = side_length ∧
      dist (triangle 2) (triangle 0) = side_length)
  ) → ∃ (area : ℝ), area = (9 * real.sqrt 3) / 4 :=
by
  sorry

end equilateral_triangle_area_from_hexagon_l515_515796


namespace total_rods_required_l515_515782

-- Define the number of rods needed per unit for each type
def rods_per_sheet_A : ℕ := 10
def rods_per_sheet_B : ℕ := 8
def rods_per_sheet_C : ℕ := 12
def rods_per_beam_A : ℕ := 6
def rods_per_beam_B : ℕ := 4
def rods_per_beam_C : ℕ := 5

-- Define the composition per panel
def sheets_A_per_panel : ℕ := 2
def sheets_B_per_panel : ℕ := 1
def beams_C_per_panel : ℕ := 2

-- Define the number of panels
def num_panels : ℕ := 10

-- Prove the total number of metal rods required for the entire fence
theorem total_rods_required : 
  (sheets_A_per_panel * rods_per_sheet_A + 
   sheets_B_per_panel * rods_per_sheet_B +
   beams_C_per_panel * rods_per_beam_C) * num_panels = 380 :=
by 
  sorry

end total_rods_required_l515_515782


namespace oddly_powerful_count_l515_515067

def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem oddly_powerful_count : (finset.range 2010).filter is_oddly_powerful .card = 16 := 
sorry

end oddly_powerful_count_l515_515067


namespace quadratic_function_expression_range_of_a_l515_515768

-- Definitions of the conditions
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 1

-- Question 1: Expression of f(x)
theorem quadratic_function_expression : 
  f(0) = 3 ∧ f(2) = 3 ∧ ∀ x, f(x) ≥ 1 :=
  by 
    -- Provide the necessary steps to prove this theorem
    sorry

-- Question 2: Range of values for a
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ interval 2 * a (a + 1), ∃ x1 x2, x1 ≠ x2 ∧ f(x1) = f(x2)) ↔ 0 < a ∧ a < 0.5 :=
  by 
    -- Provide the necessary steps to prove this theorem
    sorry

end quadratic_function_expression_range_of_a_l515_515768


namespace wildflowers_contained_red_more_than_white_l515_515050

theorem wildflowers_contained_red_more_than_white :
  ∀ (total_wildflowers yellow_white red_yellow red_white : ℕ),
  total_wildflowers = 44 →
  yellow_white = 13 →
  red_yellow = 17 →
  red_white = 14 →
  let flowers_with_red := red_yellow + red_white in
  let flowers_with_white := yellow_white + red_white in
  flowers_with_red - flowers_with_white = 4 :=
by
  intros total_wildflowers yellow_white red_yellow red_white
  sorry

end wildflowers_contained_red_more_than_white_l515_515050


namespace isosceles_triangle_reflection_l515_515054

theorem isosceles_triangle_reflection
  (a m a' m' : ℝ) 
  (α : ℝ)
  (h1 : 0 < α ∧ α < 120)
  (h2 : m > 0)
  (h3 : reflect_isosceles_triangle a m α = (a', m')):
  a' / a + m' / m = 4 := 
sorry

end isosceles_triangle_reflection_l515_515054


namespace solve_vector_b_l515_515692

variables (a b c : ℝ × ℝ) (A B C : Type) [NormedSpace ℝ A]

def dot_product (u v : ℝ × ℝ) :=
  u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) :=
  real.sqrt (u.1 ^ 2 + u.2 ^ 2)

def angle_between_obtuse (u v : ℝ × ℝ) :=
  dot_product u v < 0

theorem solve_vector_b :
  let a := (6, -8) in
  let b := (-4, -3) in
  let c := (1, 0) in
  dot_product a b = 0 ∧
  magnitude b = 5 ∧
  angle_between_obtuse b c :=
begin
  sorry
end

end solve_vector_b_l515_515692


namespace perimeter_quadrilateral_DEBF_area_quadrilateral_DEBF_l515_515411

-- Define the setup and properties of the kite
variables (A B C D E F : Type)
variables (AB BC BD DC : ℝ) (angle_BCD : ℝ)
variables (E_is_foot_from_B_to_AD : E)
variables (F_is_foot_from_D_to_BC : F)

-- Assuming given conditions
axiom AB_length : AB = 5
axiom BC_length : BC = 3
axiom angle_BCD_measure : angle_BCD = 60
axiom kite_symmetry : ∀ X A D, X ∈ (∃ P, P = (foot_perpendicular X A D))

-- The problem to prove the perimeter of DEBF
theorem perimeter_quadrilateral_DEBF :
  let DE_length : ℝ := /* calculations considering all given details */
  let BE_length : ℝ := /* calculations considering all given details */
  let perimeter : ℝ := /* BF and FD calculations and so on */
  perimeter = 7.86 :=
sorry

-- The problem to prove the area of DEBF
theorem area_quadrilateral_DEBF :
  let area : ℝ := /* calculations considering all given details of sides */
  area = 3.24 :=
sorry

end perimeter_quadrilateral_DEBF_area_quadrilateral_DEBF_l515_515411


namespace pastries_sold_on_Monday_l515_515007

theorem pastries_sold_on_Monday (baker_works_7_days : Prop)
(sells_1_more_each_day : ∀ (x : ℕ), ∀ (d : ℕ), d < 6 → sells (x + d + 1))
(average_5_per_day : (∑ d in range 7, sells (d)) = 35)
: sells 0 = 2 :=
by
  sorry

end pastries_sold_on_Monday_l515_515007


namespace checkerboard_7_strips_l515_515899

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l515_515899


namespace molecular_weight_compound_l515_515378

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_Cl : ℝ := 35.453

def molecular_weight (nH nC nO nN nCl : ℕ) : ℝ :=
  nH * atomic_weight_H + nC * atomic_weight_C + nO * atomic_weight_O + nN * atomic_weight_N + nCl * atomic_weight_Cl

theorem molecular_weight_compound :
  molecular_weight 4 2 3 1 2 = 160.964 := by
  sorry

end molecular_weight_compound_l515_515378


namespace total_lives_correct_l515_515707

-- Define the initial number of friends
def initial_friends : ℕ := 16

-- Define the number of lives each player has
def lives_per_player : ℕ := 10

-- Define the number of additional players that joined
def additional_players : ℕ := 4

-- Define the initial total number of lives
def initial_lives : ℕ := initial_friends * lives_per_player

-- Define the additional lives from the new players
def additional_lives : ℕ := additional_players * lives_per_player

-- Define the final total number of lives
def total_lives : ℕ := initial_lives + additional_lives

-- The proof goal
theorem total_lives_correct : total_lives = 200 :=
by
  -- This is where the proof would be written, but it is omitted.
  sorry

end total_lives_correct_l515_515707


namespace circle_x_intercept_l515_515780

theorem circle_x_intercept (C D : ℝ × ℝ) (hx : C = (0, 0)) (hy : D = (10, 0)) :
  ∃ x : ℝ, (x, 0) ∈ ({p : ℝ × ℝ | (p.fst - 5)^2 + p.snd^2 = 25} : set (ℝ × ℝ)) ∧ x = 10 :=
by
  sorry

end circle_x_intercept_l515_515780


namespace jake_more_balloons_than_allan_l515_515048

-- Define the initial and additional balloons for Allan
def initial_allan_balloons : Nat := 2
def additional_allan_balloons : Nat := 3

-- Total balloons Allan has in the park
def total_allan_balloons : Nat := initial_allan_balloons + additional_allan_balloons

-- Number of balloons Jake has
def jake_balloons : Nat := 6

-- The proof statement
theorem jake_more_balloons_than_allan : jake_balloons - total_allan_balloons = 1 := by
  sorry

end jake_more_balloons_than_allan_l515_515048


namespace mike_spend_on_plants_l515_515654

def Mike_buys : Prop :=
  let rose_bushes_total := 6
  let rose_bush_cost := 75
  let friend_rose_bushes := 2
  let self_rose_bushes := rose_bushes_total - friend_rose_bushes
  let self_rose_bush_cost := self_rose_bushes * rose_bush_cost
  let tiger_tooth_aloe_total := 2
  let aloe_cost := 100
  let self_aloe_cost := tiger_tooth_aloe_total * aloe_cost
  self_rose_bush_cost + self_aloe_cost = 500

theorem mike_spend_on_plants :
  Mike_buys := by
  sorry

end mike_spend_on_plants_l515_515654


namespace vacation_total_cost_l515_515386

theorem vacation_total_cost 
  (C : ℝ)
  (h1 : ∀ (p : ℝ), p = C / 3)
  (h2 : ∀ (q : ℝ), q = C / 5)
  (h3 : ∀ (r : ℝ), r = 50)
  (h4 : h1 C - h2 C = h3 C) :
  C = 375 := by
  sorry

end vacation_total_cost_l515_515386


namespace dandelion_average_l515_515060

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end dandelion_average_l515_515060


namespace pyramid_total_area_l515_515741

noncomputable def pyramid_base_edge := 8
noncomputable def pyramid_lateral_edge := 10

/-- The total area of the four triangular faces of a right, square-based pyramid
with base edges measuring 8 units and lateral edges measuring 10 units is 32 * sqrt(21). -/
theorem pyramid_total_area :
  let base_edge := pyramid_base_edge,
      lateral_edge := pyramid_lateral_edge,
      height := sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2),
      area_of_one_face := 1 / 2 * base_edge * height
  in 4 * area_of_one_face = 32 * sqrt 21 :=
sorry

end pyramid_total_area_l515_515741


namespace hyperbola_equation_l515_515945

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_y_axis (x : ℝ) : Prop := x = 0
def focal_distance (d : ℝ) : Prop := d = 4
def point_on_hyperbola (x y : ℝ) : Prop := x = 1 ∧ y = -Real.sqrt 3

-- Final statement to prove
theorem hyperbola_equation :
  (center_at_origin 0 0) ∧
  (focus_on_y_axis 0) ∧
  (focal_distance 4) ∧
  (point_on_hyperbola 1 (-Real.sqrt 3))
  → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = Real.sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l515_515945


namespace tile_C_in_rectangle_Y_l515_515364

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := { top := 6, right := 4, bottom := 1, left := 3 }
def TileB : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileC : Tile := { top := 5, right := 6, bottom := 3, left := 4 }
def TileD : Tile := { top := 4, right := 5, bottom := 2, left := 1 }

inductive Rectangle
| W | X | Y | Z

def placement : Rectangle → Tile → Prop :=
λ r t,
  match r with
  | Rectangle.W => (t = TileA ∨ t = TileB ∨ t = TileC ∨ t = TileD)
  | Rectangle.X => (t = TileA ∨ t = TileB ∨ t = TileC ∨ t = TileD)
  | Rectangle.Y => (t = TileC) -- Based on the correct answer from the solution steps
  | Rectangle.Z => (t = TileB)
  end

theorem tile_C_in_rectangle_Y : placement Rectangle.Y TileC :=
by
  sorry

end tile_C_in_rectangle_Y_l515_515364


namespace power_equivalence_l515_515561

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l515_515561


namespace number_of_valid_subsets_l515_515146

def isValidSubset (M : Set ℕ) : Prop :=
  (M ⊆ {0, 1, 2, 3, 4, 5}) ∧
  M.Nonempty ∧
  (∀ x ∈ M, x^2 ∉ M ∧ x.sqrt ∉ M)

theorem number_of_valid_subsets :
  finset.card {M // isValidSubset M} = 11 := 
sorry

end number_of_valid_subsets_l515_515146


namespace median_of_trapezoid_l515_515429

-- Definition of side lengths of the equilateral triangles
def s1 := 4
def s2 := 3

-- Proof of the length of the median of the trapezoid formed by these two triangles
theorem median_of_trapezoid : (s1 + s2) / 2 = 3.5 :=
by sorry

end median_of_trapezoid_l515_515429


namespace satisfies_condition_l515_515862

theorem satisfies_condition (n : ℕ) (h : n ≥ 2) :
  (∀ a b : ℕ, nat.coprime a n → nat.coprime b n → (a ≡ b [MOD n] ↔ a * b ≡ 1 [MOD n])) ↔
    n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24 ∨ n = 72 :=
sorry

end satisfies_condition_l515_515862


namespace rational_numbers_integer_sum_pow_l515_515503

open Rat Nat

theorem rational_numbers_integer_sum_pow (n : ℕ) : 
  (∃ a b : ℚ, (¬ a ∈ Int) ∧ (¬ b ∈ Int) ∧ (a + b ∈ Int) ∧ (a^n + b^n ∈ Int)) ↔ Odd n :=
sorry

end rational_numbers_integer_sum_pow_l515_515503


namespace min_adjacent_white_cells_8x8_grid_l515_515917

theorem min_adjacent_white_cells_8x8_grid (n_blacks : ℕ) (h1 : n_blacks = 20) : 
  ∃ w_cell_pairs, w_cell_pairs = 34 :=
by
  -- conditions are translated here for interpret
  let total_pairs := 112 -- total pairs in 8x8 grid
  let max_spoiled := 78  -- maximum spoiled pairs when placing 20 black cells
  let min_adjacent_white_pairs := total_pairs - max_spoiled
  use min_adjacent_white_pairs
  exact (by linarith)
  sorry

end min_adjacent_white_cells_8x8_grid_l515_515917


namespace smallest_crate_dimension_l515_515405

theorem smallest_crate_dimension (x : ℝ) : 
  x * 8 * 12 = x * 96 → 
  ∀ (r : ℝ), r = 3 → 
  (∀ (d : ℝ), d = 2 * r → d ≤ x ∨ d ≤ 8 ∨ d ≤ 12) →
  x = 6 := 
begin
  intros h1 h2 h3,
  sorry,
end

end smallest_crate_dimension_l515_515405


namespace average_dandelions_picked_l515_515059

def Billy_initial_pick : ℕ := 36
def George_initial_ratio : ℚ := 1 / 3
def additional_picks : ℕ := 10

theorem average_dandelions_picked :
  let Billy_initial := Billy_initial_pick,
      George_initial := (George_initial_ratio * Billy_initial).toNat,
      Billy_total := Billy_initial + additional_picks,
      George_total := George_initial + additional_picks,
      total_picked := Billy_total + George_total in
  total_picked / 2 = 34 :=
  by
  let Billy_initial := Billy_initial_pick
  let George_initial := (George_initial_ratio * Billy_initial).toNat
  let Billy_total := Billy_initial + additional_picks
  let George_total := George_initial + additional_picks
  let total_picked := Billy_total + George_total
  sorry

end average_dandelions_picked_l515_515059


namespace total_votes_l515_515996

-- Definitions to set up the conditions
def election_winner_margin (V W L : ℕ) : Prop :=
  W = L + 0.1 * V

def vote_change_outcome (V W L : ℕ) : Prop :=
  W - 3000 = L + 3000 ∧ L + 3000 = W - 3000 + 0.1 * V

-- The main theorem statement
theorem total_votes (V W L : ℕ) 
  (h1 : election_winner_margin V W L) 
  (h2 : vote_change_outcome V W L) : 
  V = 30000 :=
by 
  sorry -- proof goes here

end total_votes_l515_515996


namespace goats_more_than_pigs_l515_515356

-- Defining the number of goats
def number_of_goats : ℕ := 66

-- Condition: there are twice as many chickens as goats
def number_of_chickens : ℕ := 2 * number_of_goats

-- Calculating the total number of goats and chickens
def total_goats_and_chickens : ℕ := number_of_goats + number_of_chickens

-- Condition: the number of ducks is half of the total number of goats and chickens
def number_of_ducks : ℕ := total_goats_and_chickens / 2

-- Condition: the number of pigs is a third of the number of ducks
def number_of_pigs : ℕ := number_of_ducks / 3

-- The statement we need to prove
theorem goats_more_than_pigs : number_of_goats - number_of_pigs = 33 := by
  -- The proof is omitted as instructed
  sorry

end goats_more_than_pigs_l515_515356


namespace area_triangle_ACE_l515_515791

-- Define the hexagon and triangle structure
structure RegularHexagon :=
  (sides : ℝ)
  (regular : ∀ (x y : ℝ), x = sides ∧ y = sides)

-- Define the conditions
def hexABCDEF : RegularHexagon := 
  { sides := 3, regular := by simp }

-- Define the problem to prove the area of $\triangle ACE$
theorem area_triangle_ACE (h : RegularHexagon) (h_sides : h.sides = 3) :
  area h.regular ACE = (9 * Real.sqrt 3) / 4 :=
sorry

end area_triangle_ACE_l515_515791


namespace maximum_real_roots_maximum_total_real_roots_l515_515267

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

def quadratic_discriminant (p q r : ℝ) : ℝ := q^2 - 4 * p * r

theorem maximum_real_roots (h1 : quadratic_discriminant a b c < 0)
  (h2 : quadratic_discriminant b c a < 0)
  (h3 : quadratic_discriminant c a b < 0) :
  ∀ (x : ℝ), (a * x^2 + b * x + c ≠ 0) ∧ 
             (b * x^2 + c * x + a ≠ 0) ∧ 
             (c * x^2 + a * x + b ≠ 0) :=
sorry

theorem maximum_total_real_roots :
    ∃ x : ℝ, ∃ y : ℝ, ∃ z : ℝ,
    (a * x^2 + b * x + c = 0) ∧
    (b * y^2 + c * y + a = 0) ∧
    (a * y ≠ x) ∧
    (c * z^2 + a * z + b = 0) ∧
    (b * z ≠ x) ∧
    (c * z ≠ y) :=
sorry

end maximum_real_roots_maximum_total_real_roots_l515_515267


namespace quadratic_points_relation_l515_515154

theorem quadratic_points_relation
  (k y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -((-1) - 1)^2 + k)
  (hB : y₂ = -(2 - 1)^2 + k)
  (hC : y₃ = -(4 - 1)^2 + k) : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end quadratic_points_relation_l515_515154


namespace infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l515_515771

open Nat

theorem infinite_solutions_2n_3n_square :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2 :=
sorry

theorem n_multiple_of_40 :
  ∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → (40 ∣ n) :=
sorry

theorem infinite_solutions_general (m : ℕ) (hm : 0 < m) :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, m * n + 1 = a^2 ∧ (m + 1) * n + 1 = b^2 :=
sorry

end infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l515_515771


namespace add_fractions_l515_515836

theorem add_fractions (x : ℝ) (h : x ≠ 1) : (1 / (x - 1) + 3 / (x - 1)) = (4 / (x - 1)) :=
by
  sorry

end add_fractions_l515_515836


namespace crayons_left_l515_515702

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
    (initial_eq : initial_crayons = 48)
    (kiley_eq : kiley_fraction = 1/4)
    (joe_eq : joe_fraction = 1/2) : 
     let kiley_take := initial_crayons * kiley_fraction
     let remaining_after_kiley := initial_crayons - kiley_take
     let joe_take := remaining_after_kiley * joe_fraction
     let final_crayons := remaining_after_kiley - joe_take
     in final_crayons = 18 :=
by
  simp [initial_eq, kiley_eq, joe_eq]
  sorry

end crayons_left_l515_515702


namespace max_distance_product_l515_515070

noncomputable def maxProductDistance (r1 r2 : ℝ) (CASH MONEY : ℂ → Prop) : ℝ :=
  let verticesCASH := {1, complex.I, -1, -complex.I}
  let verticesMONEY := {z | ∃ (k : ℕ), k < 5 ∧ z = 2 * complex.exp(complex.I * k * 2 * real.pi / 5)}
  let distances := { complex.abs(v1 - v2) | v1 ∈ verticesCASH, v2 ∈ verticesMONEY }
  distances.prod

theorem max_distance_product :
  let r1 := 1
  let r2 := 2
  let CASH := {1, complex.I, -1, -complex.I}
  let MONEY := {z | ∃ (k : ℕ), k < 5 ∧ z = 2 * complex.exp(complex.I * k * 2 * real.pi / 5)}
  maxProductDistance r1 r2 CASH MONEY = 2 ^ 20 + 1 :=
sorry

end max_distance_product_l515_515070


namespace smallest_positive_value_l515_515198

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℚ), k = (a^2 + b^2) / (a^2 - b^2) + (a^2 - b^2) / (a^2 + b^2) ∧ k = 2 :=
sorry

end smallest_positive_value_l515_515198


namespace frequency_rate_identity_l515_515424

theorem frequency_rate_identity (n : ℕ) : 
  (36 : ℕ) / (n : ℕ) = (0.25 : ℝ) → 
  n = 144 := by
  sorry

end frequency_rate_identity_l515_515424


namespace profit_is_1500_l515_515712

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l515_515712


namespace total_distance_correct_l515_515468

-- Define the times in hours
def time_Ethan := 25 / 60
def time_Frank := time_Ethan * 2
def time_Lucy := 45 / 60

-- Define the speeds in km/h
def speed_Ethan := 3
def speed_Frank := 4
def speed_Lucy := 2

-- Define the distances covered by each person
def distance_Ethan := speed_Ethan * time_Ethan
def distance_Frank := speed_Frank * time_Frank
def distance_Lucy := speed_Lucy * time_Lucy

-- Define the total distance covered
def total_distance := distance_Ethan + distance_Frank + distance_Lucy

-- Prove the total distance equals approximately 6.08 km
theorem total_distance_correct : total_distance = 6.08 :=
begin
  sorry -- Proof to be completed
end

end total_distance_correct_l515_515468


namespace incorrect_statement_A_l515_515555

theorem incorrect_statement_A (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := by
  intros h
  cases h with
  | inl hp => sorry
  | inr hq => sorry

end incorrect_statement_A_l515_515555


namespace arctan_tan_75_minus_2_tan_30_eq_75_l515_515078

theorem arctan_tan_75_minus_2_tan_30_eq_75 : 
  arctan (tan 75 - 2 * tan 30) = 75 :=
by
  sorry

end arctan_tan_75_minus_2_tan_30_eq_75_l515_515078


namespace value_of_f1_c_ge_3_find_expression_l515_515961

open Real

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

theorem value_of_f1 (b c : ℝ) (h1 : ∀ (α : ℝ), f (sin α) b c ≥ 0)
    (h2 : ∀ (β : ℝ), f (2 + cos β) b c ≤ 0) : 
    f 1 b c = 0 := 
sorry

theorem c_ge_3 (b c : ℝ) (h1 : ∀ (α : ℝ), f (sin α) b c ≥ 0) 
    (h2 : ∀ (β : ℝ), f (2 + cos β) b c ≤ 0) : 
    c ≥ 3 := 
sorry

theorem find_expression (b c : ℝ) (h1 : ∀ (α : ℝ), f (sin α) b c ≥ 0)
    (h2 : ∀ (β : ℝ), f (2 + cos β) b c ≤ 0) 
    (h3 : ∀ (α : ℝ), f (sin α) b c ≤ 8) : 
    f = (λ x, x^2 - 4 * x + 3) := 
sorry

end value_of_f1_c_ge_3_find_expression_l515_515961


namespace possible_values_of_x_l515_515208

theorem possible_values_of_x (x : ℝ) (h : (x^2 - 1) / x = 0) (hx : x ≠ 0) : x = 1 ∨ x = -1 :=
  sorry

end possible_values_of_x_l515_515208


namespace product_common_divisors_180_20_l515_515886

def integer_divisors (n : ℤ) : set ℤ :=
  {d | d ∣ n}

def common_divisors (a b : ℤ) : set ℤ :=
  integer_divisors a ∩ integer_divisors b

noncomputable def product_of_set (s : set ℤ) : ℤ :=
  s.to_finset.prod id
  -- Assuming id is an identity function over the integers

theorem product_common_divisors_180_20 :
  product_of_set (common_divisors 180 20) = 640000000 :=
by {
  sorry
}

end product_common_divisors_180_20_l515_515886


namespace prime_p_condition_l515_515473

theorem prime_p_condition (p : ℕ) (h_prime : Nat.prime p) (x y : ℕ) :
  p + 7 = 2 * x^2 ∧ p^2 + 7 = 2 * y^2 → p = 11 := by
  sorry

end prime_p_condition_l515_515473


namespace round_table_legs_l515_515592

theorem round_table_legs:
  ∀ (chairs tables disposed chairs_legs tables_legs : ℕ) (total_legs : ℕ),
  chairs = 80 →
  chairs_legs = 5 →
  tables = 20 →
  disposed = 40 * chairs / 100 →
  total_legs = 300 →
  total_legs - (chairs - disposed) * chairs_legs = tables * tables_legs →
  tables_legs = 3 :=
by 
  intros chairs tables disposed chairs_legs tables_legs total_legs
  sorry

end round_table_legs_l515_515592


namespace book_selling_price_l515_515009

def cost_price : ℕ := 225
def profit_percentage : ℚ := 0.20
def selling_price := cost_price + (profit_percentage * cost_price)

theorem book_selling_price :
  selling_price = 270 :=
by
  sorry

end book_selling_price_l515_515009


namespace complex_num_sum_l515_515907

def is_complex_num (a b : ℝ) (z : ℂ) : Prop :=
  z = a + b * Complex.I

theorem complex_num_sum (a b : ℝ) (z : ℂ) (h : is_complex_num a b z) :
  z = (1 - Complex.I) ^ 2 / (1 + Complex.I) → a + b = -2 :=
by
  sorry

end complex_num_sum_l515_515907


namespace square_area_eq_26_l515_515374

open Real
open EuclideanGeometry

structure Point where
  x : ℝ
  y : ℝ

definition P : Point := ⟨1, 1⟩
definition Q : Point := ⟨-4, 0⟩
definition R : Point := ⟨-3, -5⟩
definition S : Point := ⟨2, -4⟩

noncomputable def distance (p1 p2 : Point) : ℝ :=
  sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem square_area_eq_26 : 
  distance P Q = sqrt 26 ∧ 
  distance Q R = sqrt 26 ∧ 
  distance R S = sqrt 26 ∧ 
  distance S P = sqrt 26 ∧ 
  (P.x - Q.x) * (Q.x - R.x) + (P.y - Q.y) * (Q.y - R.y) = 0 ∧
  (Q.x - R.x) * (R.x - S.x) + (Q.y - R.y) * (R.y - S.y) = 0 → 
  let side := sqrt 26 in side * side = 26 := 
sorry

end square_area_eq_26_l515_515374


namespace monomial_sum_mn_l515_515581

theorem monomial_sum_mn (m n : ℤ) 
  (h1 : m + 6 = 1) 
  (h2 : 2 * n + 1 = 7) : 
  m * n = -15 := by
  sorry

end monomial_sum_mn_l515_515581


namespace difference_max_min_y_l515_515445

def initial_positive_percentage := 40
def initial_negative_percentage := 60
def final_positive_percentage := 80
def final_negative_percentage := 20

theorem difference_max_min_y : 
  let initial_positive := initial_positive_percentage in
  let initial_negative := initial_negative_percentage in
  let final_positive := final_positive_percentage in
  let final_negative := final_negative_percentage in
  let y_min := 40 in
  let y_max := 60 in
  (y_max - y_min) = 20 :=
by
  sorry

end difference_max_min_y_l515_515445


namespace greatest_number_of_roses_l515_515305

-- Define the prices and total budget.
def individual_price : ℝ := 6.30
def dozen_price : ℝ := 36
def two_dozen_price : ℝ := 50
def total_budget : ℝ := 680

-- Define the number of roses in a dozen and in two dozen.
def dozen_roses : ℕ := 12
def two_dozen_roses : ℕ := 24

-- Define the target number of roses to prove.
def max_roses : ℕ := 316

-- Formalize the proof problem.
theorem greatest_number_of_roses :
  ∃ n (individuals dozens two_dozens : ℕ),
      total_budget = (individual_price * individuals + dozen_price * dozens + two_dozen_price * two_dozens)
      ∧ max_roses = (individuals + dozen_roses * dozens + two_dozen_roses * two_dozens) :=
  sorry

end greatest_number_of_roses_l515_515305


namespace max_metro_lines_l515_515415

theorem max_metro_lines (lines : ℕ) 
  (stations_per_line : ℕ) 
  (max_interchange : ℕ) 
  (max_lines_per_interchange : ℕ) :
  (stations_per_line >= 4) → 
  (max_interchange <= 3) → 
  (max_lines_per_interchange <= 2) → 
  (∀ s_1 s_2, ∃ t_1 t_2, t_1 ≤ max_interchange ∧ t_2 ≤ max_interchange ∧
     (s_1 = t_1 ∨ s_2 = t_1 ∨ s_1 = t_2 ∨ s_2 = t_2)) → 
  lines ≤ 10 :=
by
  sorry

end max_metro_lines_l515_515415


namespace perpendicular_bisectors_condition_l515_515132

theorem perpendicular_bisectors_condition (O : Point) (n : ℕ) (lines : Fin 2n → Line) :
  (∃ k : ℤ, ∃ angles : Fin 2n → ℝ, 
    angles.sum (Finset.range (2 * n)) = 180 * k ∧ 
    (∀ i : Fin (2 * n), 
      lines i ≠ lines (i + 1) ∧
      (lines i).perpendicular (lines (i + 1)))
   →
  (∀ (polygon : Polygon (2 * n)),
    (∀ i : Fin (2 * n), 
      (lines i).perpendicular_bisector_of (polygon.side i)))
   :=
sorry

end perpendicular_bisectors_condition_l515_515132


namespace tan_addition_identity_l515_515003

theorem tan_addition_identity :
  tan 18 + tan 12 + (sqrt 3 / 3) * tan 18 * tan 12 = sqrt 3 / 3 := sorry

end tan_addition_identity_l515_515003


namespace mean_of_solutions_l515_515881

theorem mean_of_solutions (x : ℝ) (h : x^3 + x^2 - 14 * x = 0) : 
  let a := (0 : ℝ)
  let b := (-1 + Real.sqrt 57) / 2
  let c := (-1 - Real.sqrt 57) / 2
  (a + b + c) / 3 = -2 / 3 :=
sorry

end mean_of_solutions_l515_515881


namespace quadratic_inequality_solution_set_l515_515149

theorem quadratic_inequality_solution_set (m t : ℝ)
  (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - m*x + t < 0) : 
  m - t = -1 := sorry

end quadratic_inequality_solution_set_l515_515149


namespace largest_prime_factor_of_9883_l515_515115

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_9883 :
  ∃ p : ℕ, is_prime p ∧ p ∣ 9883 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 9883 → q ≤ p :=
begin
  let p := 109,
  have hp : is_prime p := by {
    sorry,  -- Here we would insert the detailed proof that 109 is a prime number
  },
  have divisor_9883 : 109 ∣ 9883 := by {
    sorry,  -- Here we would show that 109 is a factor of 9883
  },
  use p,
  split,
  { exact hp, },
  split,
  { exact divisor_9883, },
  intros q hq,
  have prime_q : is_prime q := hq.1,
  have divisor_q : q ∣ 9883 := hq.2,
  -- Here we would insert the detailed reasoning to show there is no larger prime factor
  sorry,
end

end largest_prime_factor_of_9883_l515_515115


namespace domain_of_f_smallest_positive_period_l515_515180

def f (x : Real) : Real := Real.tan (2 * x + Real.pi / 4)

noncomputable def domain_f : Set Real :=
  { x : Real | ∀ k : ℤ, x ≠ k * Real.pi / 2 + Real.pi / 8 }

theorem domain_of_f :
  ∀ x : Real, x ∈ domain_f :=
sorry

theorem smallest_positive_period :
  ∀ x : Real, f (x + Real.pi / 2) = f x :=
sorry

end domain_of_f_smallest_positive_period_l515_515180


namespace acute_angle_repeated_root_eq_l515_515262

theorem acute_angle_repeated_root_eq
  (q : ℝ) (h_acute : 0 < q ∧ q < π / 2)
  (h_repeated_root : ∃ x, x^2 + 4 * x * real.cos q + real.cot q = 0) :
  q = π / 6 ∨ q = 5 * π / 6 :=
sorry

end acute_angle_repeated_root_eq_l515_515262


namespace length_of_plot_l515_515756

theorem length_of_plot (b : ℕ) (h1 : l = b + 60) (h2 : 26.50 * (4 * b + 120) = 5300) : l = 80 := 
by
  sorry

end length_of_plot_l515_515756


namespace hypotenuse_length_l515_515855

-- Given conditions of the problem:

variables {P Q R S : Type*}
variables [metric_space P] -- Assume P is a point in a metric space for distance measurement
variables (θ : ℝ) (hyp_PQR : is_right_triangle P Q R) -- PQR is a right triangle
variables (PS_length_tan : dist P S = real.tan θ) -- PS has length tan θ
variables (angle_bounds : 0 < θ ∧ θ < real.pi / 2) -- θ is an angle such that 0 < θ < π/2

-- To prove:

theorem hypotenuse_length (hyp_PQR : is_right_triangle P Q R) (PS_length_tan : dist P S = real.tan θ)
(angle_bounds : 0 < θ ∧ θ < real.pi / 2) : 
  let QR := dist Q R in
  QR = 2 * real.tan θ := 
sorry

end hypotenuse_length_l515_515855


namespace find_A_for_diamond_eq_64_l515_515200

def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B * 2

theorem find_A_for_diamond_eq_64 : ∃ (A : ℝ), diamond A 5 = 64 ∧ A = 8.5 :=
by
  use 8.5
  show diamond 8.5 5 = 64
  show 4 * 8.5 + 3 * 5 * 2 = 64
  sorry

end find_A_for_diamond_eq_64_l515_515200


namespace _l515_515112

open Matrix

noncomputable def matrix_2x2_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

@[simp] theorem inverse_correctness : 
  invOf (matrix_2x2_inverse) = ![![9/46, -5/46], ![2/46, 4/46]] :=
by
  sorry

end _l515_515112


namespace exponent_equality_l515_515564

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l515_515564


namespace quadratic_minimum_value_l515_515147

theorem quadratic_minimum_value (p q : ℝ) (h_min_value : ∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) :
  q = 10 + p^2 / 8 :=
by
  sorry

end quadratic_minimum_value_l515_515147


namespace sum_of_all_four_digit_numbers_using_1_to_5_l515_515494

open BigOperators

theorem sum_of_all_four_digit_numbers_using_1_to_5 : ∑ n in { n : ℕ | n >= 1000 ∧ n < 10000 ∧ ∀ d ∈ nat.digits 10 n, d ∈ {1, 2, 3, 4, 5} ∧ (λ x, list.count x (nat.digits 10 n)) d = 1 }, n = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_using_1_to_5_l515_515494


namespace sum_of_interior_angles_l515_515053

theorem sum_of_interior_angles (x : ℝ) (h : x = 36) : 
  let n := 360 / x in
  (n - 2) * 180 = 1440 :=
by
  have h1 : n = 10 := by
    rw [h, Real.div_eq 360 36, Real.rat_cast_index_k 36 36]
  rw [h1]
  sorry

end sum_of_interior_angles_l515_515053


namespace avg_of_9_numbers_l515_515678

theorem avg_of_9_numbers (a b c d e f g h i : ℕ)
  (h1 : (a + b + c + d + e) / 5 = 99)
  (h2 : (e + f + g + h + i) / 5 = 100)
  (h3 : e = 59) : 
  (a + b + c + d + e + f + g + h + i) / 9 = 104 := 
sorry

end avg_of_9_numbers_l515_515678


namespace stick_segments_l515_515804

theorem stick_segments (L : ℕ) (L_nonzero : L > 0) :
  let red_segments := 8
  let blue_segments := 12
  let black_segments := 18
  let total_segments := (red_segments + blue_segments + black_segments) 
                       - (lcm red_segments blue_segments / blue_segments) 
                       - (lcm blue_segments black_segments / black_segments)
                       - (lcm red_segments black_segments / black_segments)
                       + (lcm red_segments (lcm blue_segments black_segments) / (lcm blue_segments black_segments))
  let shortest_segment_length := L / lcm red_segments (lcm blue_segments black_segments)
  (total_segments = 28) ∧ (shortest_segment_length = L / 72) := by
  sorry

end stick_segments_l515_515804


namespace consecutive_ints_product_div_6_l515_515772

theorem consecutive_ints_product_div_6 (n : ℤ) : (n * (n + 1) * (n + 2)) % 6 = 0 := 
sorry

end consecutive_ints_product_div_6_l515_515772


namespace four_digit_number_sum_is_correct_l515_515487

def valid_digits : Finset ℕ := {1, 2, 3, 4, 5}

def valid_four_digit_numbers : Finset (List ℕ) := 
  valid_digits.powerset.filter (λ s, s.card = 4) 
  >>= λ s, (s.val.to_list.permutations)

def four_digit_number_sum (numbers : Finset (List ℕ)) : ℕ :=
  numbers.sum (λ l, 
    match l with
    | [a, b, c, d] => 1000 * a + 100 * b + 10 * c + d
    | _           => 0
    end)

theorem four_digit_number_sum_is_correct :
  four_digit_number_sum valid_four_digit_numbers = 399960 :=
by
  sorry

end four_digit_number_sum_is_correct_l515_515487


namespace area_of_S3_l515_515092

theorem area_of_S3 (S1_area : ℝ) (h1 : S1_area = 16)
  (h2 : ∀ (s1 : ℝ), s1^2 = S1_area → 
        let S2_side := s1 * (Real.sqrt 2) / 2 in
        ∀ (s2 : ℝ), s2 = S2_side → 
          let S2_area := S2_side^2 in
          let S3_side := S2_side * (Real.sqrt 2) / 2 in
          ∀ (s3 : ℝ), s3 = S3_side → 
            S3_side^2 = 4) : 
  ∃ S3_area : ℝ, S3_area = 4 :=
by
  sorry

end area_of_S3_l515_515092


namespace function_bounded_in_interval_l515_515908

variables {f : ℝ → ℝ}

theorem function_bounded_in_interval (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end function_bounded_in_interval_l515_515908


namespace min_value_fraction_l515_515148

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b^2 - 4 * a * c ≤ 0) :
  (a + b + c) / (2 * a) ≥ 2 :=
  sorry

end min_value_fraction_l515_515148


namespace probability_prime_sum_l515_515467

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def prime_sum_probability : ℚ :=
  let balls := [10, 11, 13, 14, 17, 19, 21, 23]
  let prime_sums := balls.filter (λ b => is_prime (sum_of_digits b))
  prime_sums.length / balls.length

theorem probability_prime_sum : prime_sum_probability = 1 / 2 :=
  sorry

end probability_prime_sum_l515_515467


namespace second_plan_minutes_included_l515_515418

theorem second_plan_minutes_included 
  (monthly_fee1 : ℝ := 50) 
  (limit1 : ℝ := 500) 
  (cost_per_minute1 : ℝ := 0.35) 
  (monthly_fee2 : ℝ := 75) 
  (cost_per_minute2 : ℝ := 0.45) 
  (M : ℝ) 
  (usage : ℝ := 2500)
  (cost1 := monthly_fee1 + cost_per_minute1 * (usage - limit1))
  (cost2 := monthly_fee2 + cost_per_minute2 * (usage - M))
  (equal_costs : cost1 = cost2) : 
  M = 1000 := 
by
  sorry 

end second_plan_minutes_included_l515_515418


namespace lightning_distance_l515_515460

theorem lightning_distance :
  let t := 12 -- time in seconds
  let s := 1050 -- speed of sound in feet per second
  let feet_per_mile := 5280 -- conversion factor
  let distance_in_feet := s * t

  -- Convert feet to miles
  let distance_in_miles := distance_in_feet / feet_per_mile

  -- nearest_half_mile rounds the number to the nearest 0.5
  let round_to_half_mile (d : Real) : Real := Integer.round (d * 2) / 2

  -- rounding the distance in miles
  let rounded_distance := round_to_half_mile distance_in_miles

  rounded_distance = 2.5 :=
by
  sorry

end lightning_distance_l515_515460


namespace cartesian_eqn_and_min_AB_l515_515222

open Real

-- Definitions for the conditions
noncomputable def line_eqn (rho theta a : ℝ) := rho * cos theta + 2 * rho * sin theta + a

noncomputable def point_P := (sqrt 2, 7 * π / 4)

noncomputable def curve_C (t : ℝ) := (t, 1 / 4 * t^2)

-- Statement of the proof problem
theorem cartesian_eqn_and_min_AB :
  (∃ a : ℝ, line_eqn (sqrt 2) (7 * π / 4) a = 0 ∧
    ∀ θ : ℝ, line_eqn (sqrt 2) θ 1 = 0 → x + 2 * y + 1 = 0) ∧
  (∀ t : ℝ,
    let B := curve_C t in
    ∀ A : (ℝ × ℝ), A.1 + 2 * A.2 + 1 = 0 →
      (∥(A.1 - B.1, A.2 - B.2)∥ = ∥(t, 1 / 4 * t^2 - B.2)∥)) →
    ∃ min_val : ℝ, min_val = sqrt 5 / 10
:= 
sorry

end cartesian_eqn_and_min_AB_l515_515222


namespace tangent_line_equation_at_0_l515_515877

noncomputable def f (x : ℝ) : ℝ := exp x - cos x

theorem tangent_line_equation_at_0 :
  let m := (deriv f 0)
  let y1 := f 0
  let x1 := 0
  (λ (x : ℝ), m * (x - x1) + y1) = (λ (x : ℝ), x) :=
by
  sorry

end tangent_line_equation_at_0_l515_515877


namespace multiply_fractions_l515_515448

theorem multiply_fractions :
  (1 / 3) * (4 / 7) * (9 / 13) * (2 / 5) = 72 / 1365 :=
by sorry

end multiply_fractions_l515_515448


namespace all_n_geq_2_l515_515472

def satisfiesCondition (n : ℕ) (a : Fin n → ℝ) : Prop :=
  let A := {d | ∃ (i j : Fin n), i < j ∧ d = |a i - a j|}
  A.card = n * (n - 1) / 2

theorem all_n_geq_2 (n : ℕ) (h : n ≥ 2) : ∃ a : Fin n → ℝ, satisfiesCondition n a :=
by
  sorry

end all_n_geq_2_l515_515472


namespace greatest_prime_factor_X_plus_Y_l515_515569

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
noncomputable def double_factorial (n : ℕ) : ℕ := if n ≤ 1 then 1 else n * double_factorial (n - 2)

def prime_factors (n : ℕ) : List ℕ := 
  let rec factors (n : ℕ) (p : ℕ) (acc : List ℕ) : List ℕ := 
    if p * p > n then if n > 1 then n :: acc else acc
    else if n % p = 0 then factors (n / p) p (p :: acc)
    else factors n (p + 1) acc
  factors n 2 []

def greatest_prime_factor (n : ℕ) : ℕ := 
  match prime_factors n with
  | []     => 0
  | l      => l.maximum.getOrElse 0

theorem greatest_prime_factor_X_plus_Y :
  let X := factorial 20
  let Y := double_factorial 21
  greatest_prime_factor (X + Y) = 19 := 
by
  sorry

end greatest_prime_factor_X_plus_Y_l515_515569


namespace days_between_dates_l515_515969

-- Define the starting and ending dates
def start_date : Nat := 1990 * 365 + (19 + 2 * 31 + 28) -- March 19, 1990 (accounting for leap years before the start date)
def end_date : Nat   := 1996 * 365 + (23 + 2 * 31 + 29 + 366 * 2 + 365 * 3) -- March 23, 1996 (accounting for leap years)

-- Define the number of leap years between the dates
def leap_years : Nat := 2 -- 1992 and 1996

-- Total number of days
def total_days : Nat := (end_date - start_date + 1)

theorem days_between_dates : total_days = 2197 :=
by
  sorry

end days_between_dates_l515_515969


namespace total_travel_distance_l515_515774

theorem total_travel_distance
  (initial_height : ℝ)
  (bounce_coefficient : ℝ)
  (num_bounces : ℕ)
  (initial_height_eq : initial_height = 20)
  (bounce_coefficient_eq : bounce_coefficient = 0.6)
  (num_bounces_eq : num_bounces = 4) :
  let distances := (List.range num_bounces).map (λ n => initial_height * bounce_coefficient ^ n)
  in
  (initial_height + 2 * (distances.sum)) = 69.632 :=
by
  sorry

end total_travel_distance_l515_515774


namespace min_value_expression_l515_515484

theorem min_value_expression (x y : ℝ) (hx : |x| < 1) (hy : |y| < 2) (hxy : x * y = 1) : 
  ∃ k, k = 4 ∧ (∀ z, z = (1 / (1 - x^2) + 4 / (4 - y^2)) → z ≥ k) :=
sorry

end min_value_expression_l515_515484


namespace num_irrational_count_l515_515815

-- Definition of the numbers
def num1 : ℝ := Real.pi
def num2 : ℝ := 0.2
def num3 : ℚ := 22 / 7
def num4 : ℚ := 0
def num5 : ℝ := 0.13333...(repeating 3)
def num6 : ℝ := -1.121121112...(with pattern of additional 1 between every two 2's)

-- Helper functions to identify rational and irrational numbers
def is_rational (x : ℝ) : Prop := ∀ q : ℚ, x ≠ ↑q
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- The theorem to prove the number of irrational numbers is 2
theorem num_irrational_count : 
  (if is_irrational num1 then 1 else 0) + 
  (if is_irrational num2 then 1 else 0) +
  (if is_irrational (↑num3 : ℝ) then 1 else 0) + 
  (if is_irrational (↑num4 : ℝ) then 1 else 0) + 
  (if is_irrational num5 then 1 else 0) + 
  (if is_irrational num6 then 1 else 0) = 2 :=
by sorry

end num_irrational_count_l515_515815


namespace problem_statement_l515_515161

variables {α : Type*} [linear_ordered_field α]

/-- Conditions of the problem -/
def even_function (f : α → α) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : α → α) (p : α) : Prop :=
∀ x, f x = f (x + p)

def decreasing_on (f : α → α) (s : set α) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def increasing_on (f : α → α) (s : set α) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

/-- Problem Statement -/
theorem problem_statement (f : ℝ → ℝ) (A B : ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_decreasing : decreasing_on f (set.Ioo (-3) (-2)))
  (hA : 0 < A) (hAB : A + B < π/2) :
  f (Real.sin A) < f (Real.cos B) :=
begin
  sorry
end

end problem_statement_l515_515161


namespace part_a_part_b_l515_515425

section acquaintances

open Classical

-- Define the concept of acquaintance
variable {Person : Type} (knows : Person → Person → Prop)

-- Define the "knows" relation is symmetric and includes self-knowledge
axiom knows_symmetric : ∀ {x y : Person}, knows x y → knows y x
axiom knows_refl : ∀ x : Person, knows x x

-- A set of persons is divided into non-intersecting subsets
variable {subsets : Finset (Finset Person)}
axiom subsets_partition : ∀ {p : Person}, ∃ s ∈ subsets, p ∈ s

-- Conditions on subsets
axiom no_one_knows_all : ∀ s ∈ subsets, ∀ p ∈ s, ∃ q ∈ s, ¬knows p q
axiom not_all_three_mutual : ∀ s ∈ subsets, ∀ p q r ∈ s, ¬(knows p q ∧ knows q r ∧ knows r p)
axiom unique_mutual_knows : ∀ s ∈ subsets, ∀ p q ∈ s, ¬knows p q → ∃! r ∈ s, knows p r ∧ knows q r

-- Part (a)
theorem part_a : ∀ s ∈ subsets, ∀ p q ∈ s, ∃ n, (∀ r ∈ s, (Finset.filter (knows r) s).card = n) :=
by sorry

-- Part (b)
noncomputable def max_subsets : ℕ := (1990 / 5).natAbs

theorem part_b : subsets.card ≤ max_subsets :=
by sorry

end acquaintances

end part_a_part_b_l515_515425


namespace number_of_real_solutions_l515_515338

theorem number_of_real_solutions :
  (∃ x : ℝ, 3^(2*x+2) - 3^(x+3) - 3^x + 3 = 0) ∧
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    3^(2*x₁+2) - 3^(x₁+3) - 3^x₁ + 3 = 0 ∧ 
    3^(2*x₂+2) - 3^(x₂+3) - 3^x₂ + 3 = 0) ∧ 
  ¬(∃ x₃ : ℝ, x₃ ≠ x₁ ∧ x₃ ≠ x₂ ∧ 
    3^(2*x₃+2) - 3^(x₃+3) - 3^x₃ + 3 = 0) := sorry

end number_of_real_solutions_l515_515338


namespace time_for_B_l515_515753

noncomputable theory

def rate_A : ℝ := 1 / 5
def rate_BC : ℝ := 1 / 3
def rate_AC : ℝ := 1 / 2

theorem time_for_B :
  ∃ (time_B : ℝ), time_B = 30 :=
by
  let rate_C := rate_AC - rate_A
  let rate_B := rate_BC - rate_C
  let time_for_B := 1 / rate_B
  have : time_for_B = 30 := by sorry
  exact ⟨time_for_B, this⟩

end time_for_B_l515_515753


namespace diagonal_length_l515_515880

-- Provided: a circle ω with radius R
variables {R : ℝ} (circle : set (ℝ × ℝ)) (ω : (ℝ × ℝ) → ℝ)
def circle_center : (ℝ × ℝ) := (0, 0)
axiom circle_definition : ∀ (p : (ℝ × ℝ)), p ∈ circle ↔ (ω p = R)

-- Provided: Square ABCD with vertices A and D lying on ω
variables (A D : ℝ × ℝ) (B C : ℝ × ℝ)
axiom A_on_circle : ω A = R
axiom D_on_circle : ω D = R

-- Provided: BC tangent to ω
variables (tangent_plane : set (ℝ × ℝ))
axiom tangent_definition : tangent_plane = {p | (∀ (q : (ℝ × ℝ)), q ∈ circle → q ≠ p) ∧ (∀ (q : (ℝ × ℝ)), q ∈ circle → q ∈ tangent_plane)}

-- Provided: Diagonal of square ABCD
def length (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
def diag_AC : ℝ := length A C

-- To prove: |AC| = (8 * real.sqrt 2 / 5 ) * R
theorem diagonal_length : diag_AC = (8 * real.sqrt 2 / 5) * R :=
sorry

end diagonal_length_l515_515880


namespace value_of_certain_number_l515_515990

theorem value_of_certain_number (a b : ℕ) (h : 1 / 7 * 8 = 5) (h2 : 1 / 5 * b = 35) : b = 175 :=
by
  -- by assuming the conditions hold, we need to prove b = 175
  sorry

end value_of_certain_number_l515_515990


namespace angle_CBD_is_48_degrees_l515_515231

theorem angle_CBD_is_48_degrees :
  ∀ (A B D C : Type) (α β γ δ : ℝ), 
    α = 28 ∧ β = 46 ∧ C ∈ [B, D] ∧ γ = 30 → 
    δ = 48 := 
by 
  sorry

end angle_CBD_is_48_degrees_l515_515231


namespace cards_make_24_l515_515869

theorem cards_make_24 :
  ∃ (a b c d : ℤ), (a = 6) ∧ (b = -3) ∧ (c = -4) ∧ (d = 2) ∧ (a / b * c * d = 24) :=
begin
  use [6, -3, -4, 2],
  split, 
  { refl, },
  split,
  { refl, },
  split,
  { refl, },
  split,
  { refl, },
  norm_num,
end

end cards_make_24_l515_515869
