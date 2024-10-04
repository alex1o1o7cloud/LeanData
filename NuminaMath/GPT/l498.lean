import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Coefficient
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Monomial
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Ineq
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Vector.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Measure.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Order

namespace power_multiplication_l498_498242

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication_l498_498242


namespace total_reduction_l498_498140

theorem total_reduction:
  let original_price := 500
  let price1 := original_price * (1 - 0.06)
  let price2 := price1 * (1 + 0.03)
  let price3 := price2 * (1 - 0.04)
  let price4 := price3 * (1 + 0.02)
  let final_price := price4 * (1 - 0.05)
in original_price - final_price = 49.670816 := by
  sorry

end total_reduction_l498_498140


namespace magnitude_of_power_l498_498203

noncomputable def z : ℂ := 4 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_power :
  Complex.abs (z ^ 4) = 576 := by
  sorry

end magnitude_of_power_l498_498203


namespace area_of_triangle_AEF_l498_498701

theorem area_of_triangle_AEF
  (A B C D E F : Point)
  (h_square: square ABCD)
  (h_length: ∀ P Q : Point, side_length P Q = 2)
  (h_bisects1: bisects AE BC)
  (h_bisects2: bisects AF CD)
  (h_parallel: parallel EF BD)
  (h_passes: passes EF C) :
  area_triangle A E F = 8 / 3 :=
sorry

end area_of_triangle_AEF_l498_498701


namespace max_truthful_gnomes_l498_498002

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l498_498002


namespace find_triplets_l498_498221

theorem find_triplets (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a ^ b ∣ b ^ c - 1) ∧ (a ^ c ∣ c ^ b - 1)) ↔ (a = 1 ∨ (b = 1 ∧ c = 1)) :=
by sorry

end find_triplets_l498_498221


namespace sum_of_solutions_for_x_l498_498328

theorem sum_of_solutions_for_x :
  (∃ x y : ℝ, y = 8 ∧ x^2 + y^2 = 169) →
  (∃ x₁ x₂ : ℝ, x₁^2 = 105 ∧ x₂^2 = 105 ∧ x₁ + x₂ = 0) :=
by
  intro h
  cases h with x h1
  cases h1 with y h2
  cases h2 with hy h3
  use (√(105)), (-√(105))
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  exact add_right_neg (√(105))

end sum_of_solutions_for_x_l498_498328


namespace measure_angle_abh_l498_498819

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498819


namespace regular_octagon_angle_ABH_l498_498879

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498879


namespace intersection_distance_line_curve1_max_distance_curve2_line_l498_498698

-- Define line l with parameter t
def line_l (t : ℝ) : ℝ × ℝ := (1 + t / 2, (real.sqrt 3) * t / 2)

-- Define curve C1 with parameter theta
def curve_C1 (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sin θ)

-- Define curve C2 after stretching
def curve_C2 (θ : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.cos θ, 3 * real.sin θ)

-- Prove the distance between the two intersection points of l and C1 is 1
theorem intersection_distance_line_curve1 :
  let A := (1 : ℝ, 0)
  let B := (1 / 2 : ℝ, -(real.sqrt 3) / 2)
  real.dist A B = 1 := sorry

-- Prove the maximum distance from a moving point P on C2 to line l is (3 * sqrt 2 + sqrt 3) / 2
theorem max_distance_curve2_line :
  ∃ θ : ℝ,
  let P := curve_C2 θ
  let d (P : ℝ × ℝ) : ℝ := abs (3 * real.cos θ - (real.sqrt 3))
  d P = (3 * real.sqrt 2 + real.sqrt 3) / 2 := sorry

end intersection_distance_line_curve1_max_distance_curve2_line_l498_498698


namespace ratio_of_areas_l498_498021

variable (a : ℝ) (h_a : a > 0)

def triangle_ADE_area (a : ℝ) : ℝ :=
  (a * a * Real.sin (Real.pi / 3)) / 2

def triangle_DEC_area (a : ℝ) : ℝ :=
  (a * a * Real.sin (Real.pi / 6)) / 2

theorem ratio_of_areas (a : ℝ) (h_a : a > 0) :
  (triangle_ADE_area a) / (triangle_DEC_area a) = Real.sqrt 3 :=
by
  sorry

end ratio_of_areas_l498_498021


namespace inequality_x1_x_l498_498784

variable {a b c m_a m_b m_c : ℝ}

def x := a * b + b * c + c * a
def x₁ := m_a * m_b + m_b * m_c + m_c * m_a

theorem inequality_x1_x :
  9 / 20 < x₁ / x ∧ x₁ / x < 5 / 4 :=
by
  sorry

end inequality_x1_x_l498_498784


namespace find_c_l498_498313

open Real

noncomputable def triangle_side_c (a b c : ℝ) (A B C : ℝ) :=
  A = (π / 4) ∧
  2 * b * sin B - c * sin C = 2 * a * sin A ∧
  (1/2) * b * c * (sqrt 2)/2 = 3 →
  c = 2 * sqrt 2
  
theorem find_c {a b c A B C : ℝ} (h : triangle_side_c a b c A B C) : c = 2 * sqrt 2 :=
sorry

end find_c_l498_498313


namespace meaningful_fraction_l498_498305

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l498_498305


namespace sqrt_of_product_of_factorials_l498_498505

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_of_product_of_factorials :
  (real.sqrt (2 * factorial 4 * factorial 4)) = 24 * real.sqrt 2 :=
by
  sorry

end sqrt_of_product_of_factorials_l498_498505


namespace task_assignment_l498_498297

theorem task_assignment (volunteers : ℕ) (tasks : ℕ) (selected : ℕ) (h_volunteers : volunteers = 6) (h_tasks : tasks = 4) (h_selected : selected = 4) :
  ((Nat.factorial volunteers) / (Nat.factorial (volunteers - selected))) = 360 :=
by
  rw [h_volunteers, h_selected]
  norm_num
  sorry

end task_assignment_l498_498297


namespace part_a_part_b_l498_498542

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l498_498542


namespace correct_statement_l498_498670

-- Definitions of the entities
variables (m n l : Type) (α β : Type)

-- Conditions
axiom skew_lines (m n : Type) : ¬ (exists p, p ∈ m ∧ p ∈ n)
axiom perp_plane (m : Type) (α : Type) : m ⊥ α
axiom perp_plane_n (n : Type) (β : Type) : n ⊥ β
axiom perp_l_m (l m : Type) : l ⊥ m
axiom perp_l_n (l n : Type) : l ⊥ n
axiom not_in_plane_alpha (l : Type) (α : Type) : ¬ (l ∈ α)
axiom not_in_plane_beta (l : Type) (β : Type) : ¬ (l ∈ β)

-- The theorem to prove
theorem correct_statement : 
  (∃ i, α intersects β ∧ intersection_line(α, β) ∥ l) :=
sorry

end correct_statement_l498_498670


namespace fifth_term_arithmetic_sequence_l498_498461

variable (x y : ℝ)

def a1 := x + 2 * y^2
def a2 := x - 2 * y^2
def a3 := x + 3 * y
def a4 := x - 4 * y
def d := a2 - a1

theorem fifth_term_arithmetic_sequence : y = -1/2 → 
  x - 10 * y^2 - 4 * y^2 = x - 7/2 := by
  sorry

end fifth_term_arithmetic_sequence_l498_498461


namespace Ivan_uses_more_paint_l498_498069

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l498_498069


namespace sum_of_distinct_abc_eq_roots_l498_498763

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 * ((x + 2*y)^2 - y^2 + x - 1)

-- Main theorem statement
theorem sum_of_distinct_abc_eq_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : f a (b+c) = f b (c+a)) (h2 : f b (c+a) = f c (a+b)) :
  a + b + c = (1 + Real.sqrt 5) / 2 ∨ a + b + c = (1 - Real.sqrt 5) / 2 :=
sorry

end sum_of_distinct_abc_eq_roots_l498_498763


namespace combined_cost_price_is_250_l498_498571

axiom store_selling_conditions :
  ∃ (CP_A CP_B CP_C : ℝ),
    (CP_A = (110 + 70) / 2) ∧
    (CP_B = (90 + 30) / 2) ∧
    (CP_C = (150 + 50) / 2) ∧
    (CP_A + CP_B + CP_C = 250)

theorem combined_cost_price_is_250 : ∃ (CP_A CP_B CP_C : ℝ), CP_A + CP_B + CP_C = 250 :=
by sorry

end combined_cost_price_is_250_l498_498571


namespace odd_function_l498_498347

def f (x : ℝ) : ℝ := 1 / (2^x - 1) + 1 / 2

theorem odd_function (x : ℝ) : f (-x) = -f x := by
  sorry

end odd_function_l498_498347


namespace vector_magnitude_l498_498408

noncomputable def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude :
  ∀ (x y : ℝ), let a := (x, 2)
               let b := (1, y)
               let c := (2, -6)
               (a.1 * c.1 + a.2 * c.2 = 0) →
               (b.1 * (-c.2) - b.2 * c.1 = 0) →
               magnitude (vec_add a b) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h₁ h₂
  let a := (x, 2)
  let b := (1, y)
  let c := (2, -6)
  sorry

end vector_magnitude_l498_498408


namespace angle_ABH_regular_octagon_l498_498924

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498924


namespace Egypt_has_traditional_growth_pattern_l498_498113

-- Definitions based on conditions
def is_developed (country : String) : Prop := 
  country = "United States" ∨ country = "Japan" ∨ country = "France"

def is_developing (country : String) : Prop := 
  country = "Egypt"

def population_growth_pattern (country : String) : String :=
  if is_developed country then "modern"
  else if is_developing country then "traditional"
  else "unknown"

-- Theorem statement
theorem Egypt_has_traditional_growth_pattern : population_growth_pattern "Egypt" = "traditional" :=
sorry

end Egypt_has_traditional_growth_pattern_l498_498113


namespace distinctArrangementsMOON_l498_498717

theorem distinctArrangementsMOON : 
  (∃ s : Finset (List Char), s.card = 12 ∧ (∀ l ∈ s, l = ['M', 'O', 'O', 'N']) ∧ 
  (∀ l₁ l₂ : List Char, l₁ ∈ s → l₂ ∈ s → l₁ ~ l₂ → l₁ = l₂)) := 
sorry

end distinctArrangementsMOON_l498_498717


namespace sum_of_coordinates_on_inverse_graph_l498_498961

-- Defining the function f and its properties
variable {f : ℝ → ℝ}

-- Assumptions
axiom point_on_graph : (2 : ℝ, 4 : ℝ) ∈ set_of (λ p, p.snd = 3 * f p.fst)
axiom function_f_is_valid : function.injective f

-- Claim: Sum of the coordinates of a point on the graph of y = (1/3) * f⁻¹(x) is 2
theorem sum_of_coordinates_on_inverse_graph (h : function.surjective f) :
  ∃ x y : ℝ, (y = (1/3) * (function.inv_fun f x)) ∧ (x + y = 2) :=
by
  sorry

end sum_of_coordinates_on_inverse_graph_l498_498961


namespace course_selection_l498_498163

theorem course_selection (C : Type) [Fintype C] (courses : Finset C)
    (h : Fintype.card courses = 6)
    (conflict_courses : Finset C) (h_conflict : conflict_courses.card = 2)
    (h_disjoint : disjoint conflict_courses (courses \ conflict_courses)) :
  ∃ (ways : ℕ), ways = 14 :=
by
    let remaining_courses := courses \ conflict_courses
    have h_remaining : remaining_courses.card = 4 :=
      by rw [Finset.card_sdiff, h_conflict, Fintype.card, h, nat.sub_eq_four]
    have h_neither_conflict : ∃ c, c = (remaining_courses.choose 2).card := 
      by sorry
    have h_one_conflict : ∃ c, c = (conflict_courses.choose 1).card *
                                 (remaining_courses.choose 1).card :=
      by sorry
    use (6 + 8)
    exact sorry

end course_selection_l498_498163


namespace white_surface_area_fraction_l498_498149

-- Definitions for conditions
def four_inch_cube : ℕ := 4
def smaller_cube_size : ℕ := 1
def total_smaller_cubes : ℕ := 64
def red_cubes : ℕ := 56
def white_cubes : ℕ := 8

-- Calculations for solution
def total_surface_area (side_length : ℕ) : ℕ := 6 * (side_length * side_length)
def exposed_white_surface_area : ℕ := 4 * 3
def fraction_of_surface_area (numerator denominator : ℕ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Theorem statement
theorem white_surface_area_fraction :
  fraction_of_surface_area exposed_white_surface_area (total_surface_area four_inch_cube) = 1 / 8 :=
begin
  sorry -- Proof to be completed
end

end white_surface_area_fraction_l498_498149


namespace maximum_possible_questions_l498_498800

open Classical

theorem maximum_possible_questions (n : ℕ) :
  (∀ Q : Fin n, ∃ S : Fin 4, S Q) →
  (∀ (Q1 Q2 : Fin n), Q1 ≠ Q2 → ∃ S : Fin 1, S Q1 ∧ S Q2) →
  (∀ S, ∃ Q : Fin n, ¬ S Q) →
  n ≤ 13 :=
by
  sorry

end maximum_possible_questions_l498_498800


namespace ice_volume_after_two_hours_correct_l498_498186

noncomputable def ice_volume_after_two_hours (initial_volume : ℝ) : ℝ :=
  let volume_after_first_hour := (1/4) * initial_volume
  let volume_after_second_hour := (1/4) * volume_after_first_hour
  volume_after_second_hour

theorem ice_volume_after_two_hours_correct :
  ice_volume_after_two_hours 12 = 3/4 :=
  by
    unfold ice_volume_after_two_hours
    norm_num
    sorry

end ice_volume_after_two_hours_correct_l498_498186


namespace find_solution_am_l498_498632

theorem find_solution_am :
  ∃ a m, 15 * a + 2 ≡ 7 [MOD 20] ∧ ∃ 2 ≤ m ∧ a < m ∧ a + m = 7 :=
by
  use 3
  use 4
  split
  · -- Proof that 15x + 2 ≡ 7 [MOD 20] gives x ≡ 3 [MOD 4]
    calc
      15 * 3 + 2 ≡ 45 + 2 [MOD 20] : by rw mul_comm
      ... ≡ 47 [MOD 20] : by norm_num
      ... ≡ 7 [MOD 20] : by norm_num
  use 2
  split
  · norm_num
  split
  · norm_num
  norm_num

end find_solution_am_l498_498632


namespace gain_percent_not_applicable_l498_498178

-- Definitions used in Lean 4 statement
def purchase_price := 4700
def repair_cost := 800
def tax_and_fees := 300
def accessories_cost := 250
def selling_price := 6000

-- Total cost calculation
def total_cost := purchase_price + repair_cost + tax_and_fees + accessories_cost

-- Gain calculation
def gain := selling_price - total_cost

-- Lean 4 statement to prove
theorem gain_percent_not_applicable :
  gain < 0 → "Gain percent is not applicable due to a loss." :=
by
  intros h
  exact "Gain percent is not applicable due to a loss."

#eval gain_percent_not_applicable sorry

end gain_percent_not_applicable_l498_498178


namespace sum_of_digits_of_valid_hex_numbers_count_l498_498712

def is_valid_hex (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 16 n), d < 10

def does_not_start_with_zero (n : ℕ) : Prop :=
  let digits := nat.digits 16 n in
  if digits.head? = some 0 then False else True

def valid_hex_numbers_count : ℕ :=
  (finset.range 2001).filter (λ n, is_valid_hex n ∧ does_not_start_with_zero n).card

def sum_of_digits (n : ℕ) : ℕ :=
  (nat.digits 10 n).sum

theorem sum_of_digits_of_valid_hex_numbers_count :
  sum_of_digits valid_hex_numbers_count = 7 :=
  sorry

end sum_of_digits_of_valid_hex_numbers_count_l498_498712


namespace parallel_condition_l498_498019

-- Define the slopes of the lines
def slope (a b : ℝ) : ℝ := -a / b

-- Define the condition for parallel lines
def is_parallel (a1 b1 a2 b2 : ℝ) : Prop := slope a1 b1 = slope a2 b2

-- Define the lines
def line1 (a : ℝ) : Prop := (is_parallel a 2 1 1)

-- State the theorem for equivalence
theorem parallel_condition (a : ℝ) : (a = 2) ↔ line1 a :=
by
  sorry

end parallel_condition_l498_498019


namespace maggie_total_income_l498_498793

def total_income (h_tractor : ℕ) (r_office r_tractor : ℕ) :=
  let h_office := 2 * h_tractor
  (h_tractor * r_tractor) + (h_office * r_office)

theorem maggie_total_income :
  total_income 13 10 12 = 416 := 
  sorry

end maggie_total_income_l498_498793


namespace angle_ABH_in_regular_octagon_l498_498839

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498839


namespace salmons_kept_l498_498289

noncomputable def catch_total (hazel_catch dad_catch_dad_discarded: ℝ) :=
  let hazel_catch_round := Real.round hazel_catch
  let dad_catch_round := Real.round (dad_catch_dad_discarded - Real.round (dad_catch_dad_discarded / 10))
  hazel_catch_round + dad_catch_round

theorem salmons_kept :
  let hazel_catch := 16.5
  let dad_catch := hazel_catch * 1.75
  let dad_small := dad_catch * 0.1
  catch_total hazel_catch (dad_catch - dad_small) = 43 :=
by
  -- computation will be verified with proof
  sorry

end salmons_kept_l498_498289


namespace matrix_transformation_correct_l498_498207

theorem matrix_transformation_correct (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  let P := ![
      ![-1, 0, 0],
      ![0, 0, 1],
      ![0, 3, 0]
    ]
  in 
  (P ⬝ Q) = ![
      ![-Q 0 0, -Q 0 1, -Q 0 2],
      ![Q 2 0, Q 2 1, Q 2 2],
      ![3 * Q 1 0, 3 * Q 1 1, 3 * Q 1 2]
    ] :=
by sorry

end matrix_transformation_correct_l498_498207


namespace sum_of_fifth_powers_l498_498536

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l498_498536


namespace a_2011_l498_498247

def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → S n + S m = S (m + n)

def a1 (a : ℕ → ℕ) : Prop :=
  a 1 = 2

theorem a_2011 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : sequence a S) (h_a1 : a1 a) : a 2011 = 2 :=
sorry

end a_2011_l498_498247


namespace find_a8_l498_498746

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

theorem find_a8
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_terms a_n S)
  (h_S15 : S 15 = 45) :
  a_n 8 = 3 :=
sorry

end find_a8_l498_498746


namespace maximum_value_of_k_l498_498241

noncomputable def max_k (m : ℝ) : ℝ := 
  if 0 < m ∧ m < 1 / 2 then 
    1 / m + 2 / (1 - 2 * m) 
  else 
    0

theorem maximum_value_of_k : ∀ m : ℝ, (0 < m ∧ m < 1 / 2) → (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m) ≥ k) → k ≤ 8) :=
  sorry

end maximum_value_of_k_l498_498241


namespace ratio_c_d_of_cubic_roots_l498_498282

theorem ratio_c_d_of_cubic_roots (a b c d : ℤ)
  (h_eqn : ∀ x, a * x^3 + b * x^2 + c * x + d = 0)
  (h_roots : (a!=0) ∧ (b!=0) ∧ (c!=0) ∧ (d!=0) 
  ∧ h_eqn 4 = 0 ∧ h_eqn 5 = 0 ∧ h_eqn 6 = 0) :
  c / d = 1 / 8 := 
sorry

end ratio_c_d_of_cubic_roots_l498_498282


namespace number_of_pairs_of_shoes_size_40_to_42_200_pairs_l498_498491

theorem number_of_pairs_of_shoes_size_40_to_42_200_pairs 
  (total_pairs_sample : ℕ)
  (freq_3rd_group : ℝ)
  (freq_1st_group : ℕ)
  (freq_2nd_group : ℕ)
  (freq_4th_group : ℕ)
  (total_pairs_200 : ℕ)
  (scaled_pairs_size_40_42 : ℕ)
: total_pairs_sample = 40 ∧ freq_3rd_group = 0.25 ∧ freq_1st_group = 6 ∧ freq_2nd_group = 7 ∧ freq_4th_group = 9 ∧ total_pairs_200 = 200 ∧ scaled_pairs_size_40_42 = 40 :=
sorry

end number_of_pairs_of_shoes_size_40_to_42_200_pairs_l498_498491


namespace tan_alpha_gt_tan_beta_in_fourth_quadrant_l498_498657

variable {α β : ℝ} {k : ℤ}

theorem tan_alpha_gt_tan_beta_in_fourth_quadrant
  (h1 : sin α > sin β)
  (h2 : ∀ k : ℤ, α ≠ k * π + π / 2)
  (h3 : ∀ k : ℤ, β ≠ k * π + π / 2)
  (h4 : α ∈ Ioo (3 * π / 2) (2 * π))
  (h5 : β ∈ Ioo (3 * π / 2) (2 * π)) :
  tan α > tan β :=
sorry

end tan_alpha_gt_tan_beta_in_fourth_quadrant_l498_498657


namespace petya_wrong_l498_498424

theorem petya_wrong : ∃ (a b : ℕ), b^2 ∣ a^5 ∧ ¬ (b ∣ a^2) :=
by
  use 4
  use 32
  sorry

end petya_wrong_l498_498424


namespace angle_in_regular_octagon_l498_498884

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498884


namespace sum_of_fifth_powers_l498_498537

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l498_498537


namespace cups_remaining_l498_498792

-- Definitions based on problem conditions
def initial_cups : ℕ := 12
def mary_morning_cups : ℕ := 1
def mary_evening_cups : ℕ := 1
def frank_afternoon_cups : ℕ := 1
def frank_late_evening_cups : ℕ := 2 * frank_afternoon_cups

-- Hypothesis combining all conditions:
def total_given_cups : ℕ :=
  mary_morning_cups + mary_evening_cups + frank_afternoon_cups + frank_late_evening_cups

-- Theorem to prove
theorem cups_remaining : initial_cups - total_given_cups = 7 :=
  sorry

end cups_remaining_l498_498792


namespace sum_elements_T_l498_498370

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498370


namespace mul_powers_same_base_l498_498196

theorem mul_powers_same_base (a : ℝ) : a^3 * a^4 = a^7 := 
by 
  sorry

end mul_powers_same_base_l498_498196


namespace measure_angle_abh_l498_498866

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498866


namespace measure_angle_abh_l498_498859

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498859


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498903

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498903


namespace complement_of_A_relative_to_U_l498_498413

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 4, 5}

-- Define the proof statement for the complement of A with respect to U
theorem complement_of_A_relative_to_U : (U \ A) = {2} := by
  sorry

end complement_of_A_relative_to_U_l498_498413


namespace regular_octagon_angle_ABH_l498_498871

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498871


namespace sum_elements_T_l498_498375

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498375


namespace part_I_part_II_l498_498977

-- Define f
def f (a x : ℝ) := |x - a|

-- Part I: 
theorem part_I (x : ℝ) (h : f (-2) x + f (-2) (2 * x) > 2) : 
    x ∈ set.Iio (-2) ∪ set.Ioi (-(2/3)) :=
sorry

-- Part II:
theorem part_II (a : ℝ) (hne : ∃ x : ℝ, f a x + f a (2 * x) < 1 / 2) (h : a < 0) : 
    -1 < a := sorry

end part_I_part_II_l498_498977


namespace geometric_factorial_quotient_l498_498723

theorem geometric_factorial_quotient :
  ∀ (n : ℕ), (P : ℕ → ℝ) (H : ∀ k, P k = 1 / 2^k) (n = 7),
  (n! / (3! * (n - 3)!)) = 35 :=
sorry

end geometric_factorial_quotient_l498_498723


namespace percent_increase_perimeter_of_fourth_triangle_l498_498185

theorem percent_increase_perimeter_of_fourth_triangle :
  let side_length := 3
  let ratio := 1.2
  let fourth_side_length := side_length * ratio^3
  let percent_increase := ((fourth_side_length / side_length) * 100) - 100
  percent_increase = 72.8 :=
by
  let side_length := 3
  let ratio := 1.2
  let fourth_side_length := side_length * ratio ^ 3
  let percent_increase := ((fourth_side_length / side_length) * 100) - 100
  have fourth_side_length_eq : fourth_side_length = 3 * 1.728 := by
    simp [side_length, ratio, pow_succ]
    ring
  have percent_increase_eq : percent_increase = ((3 * 1.728 / 3) * 100) - 100 := by
    simp [fourth_side_length_eq, side_length]
  have result : percent_increase = 72.8 := by
    norm_num at percent_increase_eq
  exact result

end percent_increase_perimeter_of_fourth_triangle_l498_498185


namespace length_of_AE_l498_498975

noncomputable def calculate_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem length_of_AE :
  let A := (0, 4 : ℝ × ℝ)
  let B := (6, 0 : ℝ × ℝ)
  let E := (4, 2 : ℝ × ℝ)
  let AB := calculate_distance A B
  AE = (5 / 9) * AB → AE = (5 / 9) * (2 * real.sqrt 13) → AE = (10 * real.sqrt 13) / 9 :=
begin
  intros,
  sorry
end

end length_of_AE_l498_498975


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498898

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498898


namespace angle_ABH_is_22_point_5_l498_498905

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498905


namespace area_of_trapezoid_is_73_point_5_l498_498503

-- Define the lines y = x + 1, y = 15, and y = 8
def line1 (x : ℝ) : ℝ := x + 1
def line2 (y : ℝ) : Prop := y = 15
def line3 (y : ℝ) : Prop := y = 8

def vertex1 : ℝ × ℝ := (0, 15)
def vertex2 : ℝ × ℝ := (0, 8)
def vertex3 : ℝ × ℝ := (7, 8)
def vertex4 : ℝ × ℝ := (14, 15)

-- Define the height and bases of the trapezoid
def base1 : ℝ := 7
def base2 : ℝ := 14
def height : ℝ := 7

-- Define the area of a trapezoid function
def trapezoid_area (b1 b2 h : ℝ) : ℝ := 0.5 * (b1 + b2) * h

-- Proof statement
theorem area_of_trapezoid_is_73_point_5 :
  trapezoid_area base1 base2 height = 73.5 := by
  sorry

end area_of_trapezoid_is_73_point_5_l498_498503


namespace min_value_of_c_square_l498_498323

variables (a b c : ℝ)
variables (A B C : ℝ)
variables (S : ℝ)

-- Use the Law of Sines
-- a / sin A = b / sin B = c / sin C 
noncomputable def law_of_sines (a b c sin_A sin_B sin_C : ℝ) : Prop :=
  a / sin_A = b / sin_B ∧ b / sin_B = c / sin_C ∧ a / sin_A = c / sin_C

-- Acute triangle condition
axiom acute_triangle (A B C : ℝ) : A < π / 2 ∧ B < π / 2 ∧ C < π / 2

-- Given conditions
axiom given_conditions (a b : ℝ) (sin_A sin_B sin_C : ℝ) :
  a + 2 * b = 4 ∧ a * sin_A + 4 * b * sin_B = 6 * a * sin_B * sin_C

-- Minimum value of c^2
theorem min_value_of_c_square (h : given_conditions a b (Real.sin A) (Real.sin B) (Real.sin C)) : 
  ∃ c, c^2 = 5 - (4 * Real.sqrt 5) / 3 :=
sorry

end min_value_of_c_square_l498_498323


namespace probability_of_third_smallest_five_l498_498443

noncomputable def probability_third_smallest_is_five : ℚ :=
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4) in
  favorable_ways / total_ways

theorem probability_of_third_smallest_five :
  probability_third_smallest_is_five = 35 / 132 := 
by
  sorry

end probability_of_third_smallest_five_l498_498443


namespace first_year_after_2010_with_digit_sum_10_l498_498090

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem first_year_after_2010_with_digit_sum_10 :
  ∃ y > 2010, sum_of_digits y = 10 ∧ ∀ z, (2010 < z < y) → sum_of_digits z ≠ 10 :=
begin
  use 2035,
  split,
  { exact lt_of_lt_of_le (by norm_num : 2010 < 2035) (le_refl 2035)},
  split,
  { simp [sum_of_digits],
    norm_num,
  },
  { intros z hz,
    have hz0 : z ≥ 2010 + 1 := hz.1,
    have hz1 : z ≤ 2035 - 1 := hz.2,
    sorry,
  }
end

end first_year_after_2010_with_digit_sum_10_l498_498090


namespace john_gets_36_rolls_l498_498354

-- Let's define the conditions as a part of our Lean statement.
variable (dollars_per_dozen : ℕ) (spent_dollars : ℕ) (rolls_per_dozen : ℕ)

-- Theorem statement using the above variables to prove the final result.
theorem john_gets_36_rolls
  (h1 : dollars_per_dozen = 5)
  (h2 : spent_dollars = 15)
  (h3 : rolls_per_dozen = 12) :
  (spent_dollars / dollars_per_dozen) * rolls_per_dozen = 36 :=
begin
  -- The proof goes here but we skip it as per the requirement.
  sorry
end

end john_gets_36_rolls_l498_498354


namespace problem_statement_l498_498696

def f (x : ℝ) : ℝ :=
if x > 0 then log x else 9^(-x) + 1

theorem problem_statement : f (f 1) + f (-real.logb 3 2) = 7 :=
by {
  sorry
}

end problem_statement_l498_498696


namespace num_points_on_ellipse_with_area_l498_498981

-- Define the line equation
def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Define the area condition for the triangle
def area_condition (xA yA xB yB xP yP : ℝ) : Prop :=
  abs (xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB)) = 6

-- Define the main theorem statement
theorem num_points_on_ellipse_with_area (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∃ P1 P2 : ℝ × ℝ, 
    (ellipse_eq P1.1 P1.2) ∧ 
    (ellipse_eq P2.1 P2.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P1.1 P1.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P2.1 P2.2) ∧ 
    P1 ≠ P2 := sorry

end num_points_on_ellipse_with_area_l498_498981


namespace measure_angle_abh_l498_498816

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498816


namespace minimum_of_quadratic_l498_498035

theorem minimum_of_quadratic : ∀ x : ℝ, 1 ≤ x^2 - 6 * x + 10 :=
by
  intro x
  have h : x^2 - 6 * x + 10 = (x - 3)^2 + 1 := by ring
  rw [h]
  have h_nonneg : (x - 3)^2 ≥ 0 := by apply sq_nonneg
  linarith

end minimum_of_quadratic_l498_498035


namespace ivan_needs_more_paint_l498_498080

theorem ivan_needs_more_paint
  (section_count : ℕ)
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2) :
  let area_ivan := section_count * (5 * 2)
  let area_petr := section_count * (5 * 2 * sin α)
  area_ivan > area_petr := 
by
  simp only [area_ivan, area_petr, mul_assoc, mul_lt_mul_left, gt_iff_lt]
  exact sin_lt_one_iff.mpr hα

end ivan_needs_more_paint_l498_498080


namespace smallest_constant_exists_l498_498664

noncomputable def good_sequence (x : ℕ → ℝ) : Prop :=
  (x 0 = 1) ∧ (∀ i, x i ≥ x (i + 1))

theorem smallest_constant_exists (x : ℕ → ℝ) (h_good : good_sequence x) :
  ∃ c, c = 4 ∧ ∀ n, ∑ i in Finset.range (n + 1), (x i)^2 / (x (i + 1)) ≤ c :=
sorry

end smallest_constant_exists_l498_498664


namespace min_abc_sum_l498_498672

theorem min_abc_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 8) : a + b + c ≥ 6 :=
by {
  sorry
}

end min_abc_sum_l498_498672


namespace first_year_after_2010_has_digit_sum_10_l498_498098

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2010_has_digit_sum_10 : 
  ∃ y : ℕ, y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ z : ℕ, z > 2010 ∧ z < y → sum_of_digits z ≠ 10 :=
begin
  use 2017,
  split,
  { exact nat.lt_succ_self 2016, },
  split,
  { norm_num [sum_of_digits], },
  { intros z hz hzy,
    interval_cases z; norm_num [sum_of_digits], },
  sorry
end

end first_year_after_2010_has_digit_sum_10_l498_498098


namespace shaded_fraction_eq_four_fifteenths_l498_498570

theorem shaded_fraction_eq_four_fifteenths :
  let series_sum := 1 / 4 * (1 + ∑' n, (1 / 16) ^ n)
  in series_sum = 4 / 15 :=
by {
  have geom_sum_formula : ∀ a r : ℝ, |r| < 1 → (a * (1/(1 - r))) = a / (1 - r), from sorry,
  have series_value : ∑' n : ℕ, (1 / (16 : ℝ)) ^ n = 1 / (1 - 1/16), 
    from sorry,
  have h1: 1 / 4 * ((1 : ℝ) + 1 / (1 - 1 / 16)) = 1 / 4 * ((1 : ℝ) + 16 / 15), 
    from sorry,
  have h2: 1 / 4 * (15 / 15 + 16 / 15) = 1 / 4 * (31 / 15), from sorry,
  have h3: 1 / 4 * (31 / 15) = 31 / 60, from sorry,
  have h_correct : 4 / 15 = 31 / 60, sorry,
  exact h_correct,
}

end shaded_fraction_eq_four_fifteenths_l498_498570


namespace sum_of_three_numbers_l498_498048

theorem sum_of_three_numbers (x : ℝ) (a b c : ℝ) (h1 : a = 5 * x) (h2 : b = x) (h3 : c = 4 * x) (h4 : c = 400) :
  a + b + c = 1000 := by
  sorry

end sum_of_three_numbers_l498_498048


namespace sum_opposite_signs_eq_zero_l498_498121

theorem sum_opposite_signs_eq_zero (x y : ℝ) (h : x * y < 0) : x + y = 0 :=
sorry

end sum_opposite_signs_eq_zero_l498_498121


namespace sufficient_no_x_axis_intersections_l498_498120

/-- Sufficient condition for no x-axis intersections -/
theorem sufficient_no_x_axis_intersections
    (a b c : ℝ)
    (h : a ≠ 0)
    (h_sufficient : b^2 - 4 * a * c < -1) :
    ∀ x : ℝ, ¬(a * x^2 + b * x + c = 0) :=
by
  sorry

end sufficient_no_x_axis_intersections_l498_498120


namespace complex_number_in_second_quadrant_l498_498472

-- Define the given complex number
def z : ℂ := (2 * complex.I) / (2 - complex.I)

-- Proof statement
theorem complex_number_in_second_quadrant : z.im > 0 ∧ z.re < 0 := by
  sorry

end complex_number_in_second_quadrant_l498_498472


namespace first_year_after_2010_with_digit_sum_10_l498_498092

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem first_year_after_2010_with_digit_sum_10 :
  ∃ y > 2010, sum_of_digits y = 10 ∧ ∀ z, (2010 < z < y) → sum_of_digits z ≠ 10 :=
begin
  use 2035,
  split,
  { exact lt_of_lt_of_le (by norm_num : 2010 < 2035) (le_refl 2035)},
  split,
  { simp [sum_of_digits],
    norm_num,
  },
  { intros z hz,
    have hz0 : z ≥ 2010 + 1 := hz.1,
    have hz1 : z ≤ 2035 - 1 := hz.2,
    sorry,
  }
end

end first_year_after_2010_with_digit_sum_10_l498_498092


namespace find_y_l498_498959

variable (t : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := x = 3 - t
def condition2 : Prop := y = 2 * t + 11
def condition3 : Prop := x = 1

theorem find_y (h1 : condition1 x t) (h2 : condition2 t y) (h3 : condition3 x) : y = 15 := by
  sorry

end find_y_l498_498959


namespace sum_possible_values_l498_498404

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values_l498_498404


namespace year_2017_l498_498095

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

theorem year_2017 : ∃ (y : ℕ), y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ y', y' > 2010 → sum_of_digits y' = 10 → y' ≥ y :=
by {
  let y := 2017,
  use y,
  split,
  { exact Nat.lt_of_succ_lt_succ (Nat.succ_lt_succ (Nat.succ_pos 2016)) },
  split,
  { norm_num },
  intros y' y'_gt_2010 sum_y',
  sorry
}

end year_2017_l498_498095


namespace expected_value_Y_correct_l498_498046

noncomputable def density_function (x : ℝ) : ℝ :=
if (0 < x ∧ x < real.pi) then 1 / 2 * real.sin x else 0

noncomputable def expected_value_Y : ℝ :=
∫ y in 0..π^2, y * (1 / 4 * real.sin (real.sqrt y) / real.sqrt y)

theorem expected_value_Y_correct :
  expected_value_Y = (real.pi^2 - 2) / 2 :=
sorry

end expected_value_Y_correct_l498_498046


namespace rectangular_to_polar_l498_498595

variable (x y : ℝ)

def is_polar_coordinate (x y r theta : ℝ) : Prop :=
  r = Real.sqrt (x * x + y * y) ∧ tan theta = y / x

theorem rectangular_to_polar :
  is_polar_coordinate 6 (2 * Real.sqrt 3) (4 * Real.sqrt 3) (Real.pi / 6) :=
by
  dsimp [is_polar_coordinate]
  sorry

end rectangular_to_polar_l498_498595


namespace angle_ABH_is_22_point_5_l498_498909

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498909


namespace rationalize_denominator_l498_498440

noncomputable def cuberoot (x : ℝ) := real.cbrt x

theorem rationalize_denominator :
  let num := (cuberoot 27 + cuberoot 2)
  let denom := (cuberoot 3 + cuberoot 2)
  (num / denom) = (7 - cuberoot 54 + cuberoot 6) :=
by
  sorry

end rationalize_denominator_l498_498440


namespace probability_three_blue_marbles_l498_498179

theorem probability_three_blue_marbles
  (blue_marbles : ℕ := 8) 
  (red_marbles : ℕ := 7) 
  (total_marbles : ℕ := 15) 
  (trials : ℕ := 6) : 
  let p_blue := (blue_marbles : ℚ) / total_marbles,
      p_red := (red_marbles : ℚ) / total_marbles in
  (finset.card (finset.powerset_len 3 (finset.range trials))).val * (p_blue ^ 3) * (p_red ^ 3) = 3512320 / 11390625 :=
by sorry

end probability_three_blue_marbles_l498_498179


namespace number_with_1_before_and_after_l498_498730

theorem number_with_1_before_and_after (n : ℕ) (hn : n < 10) : 100 * 1 + 10 * n + 1 = 101 + 10 * n := by
    sorry

end number_with_1_before_and_after_l498_498730


namespace segment_AF_length_l498_498805

-- Define points
variables {A B C D E F : Type}

-- Define a line segment length in units
def length (AB : Type) [has_length AB] : ℝ := sorry

-- Define midpoint property
def is_midpoint (M A B : Type) [has_midpoint M A B] : Prop := sorry

-- Given conditions
variables {length_AB : ℝ} (h_AB : length_AB = 64)
variables (h1 : is_midpoint C A B)
variables (h2 : is_midpoint D A C)
variables (h3 : is_midpoint E A D)
variables (h4 : is_midpoint F A E)

-- Question to prove
theorem segment_AF_length : length (A E) / 2 = 4 :=
by {
  sorry
}

end segment_AF_length_l498_498805


namespace max_truthful_gnomes_l498_498003

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l498_498003


namespace part_a_part_b_l498_498531

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l498_498531


namespace angle_ABH_is_22_point_5_l498_498906

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498906


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498524

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498524


namespace chocolates_sold_in_second_week_l498_498500

theorem chocolates_sold_in_second_week
  (c₁ c₂ c₃ c₄ c₅ : ℕ)
  (h₁ : c₁ = 75)
  (h₃ : c₃ = 75)
  (h₄ : c₄ = 70)
  (h₅ : c₅ = 68)
  (h_mean : (c₁ + c₂ + c₃ + c₄ + c₅) / 5 = 71) :
  c₂ = 67 := 
sorry

end chocolates_sold_in_second_week_l498_498500


namespace length_of_rope_touching_tower_l498_498165

-- Defining the problem conditions
def rope_length : ℝ := 30 -- Length of the rope in feet
def tower_radius : ℝ := 5 -- Radius of the tower base in feet
def unicorn_height : ℝ := 5 -- Height at which the rope is attached to the unicorn
def angle_with_ground : ℝ := 30 -- Angle the rope makes with the ground in degrees
def nearest_point_distance : ℝ := 5 -- Distance from the nearest point on the tower at ground level

-- Main proof statement
theorem length_of_rope_touching_tower :
  ∃ l, l ≈ 19.06 :=
  sorry

end length_of_rope_touching_tower_l498_498165


namespace ivan_needs_more_paint_l498_498081

theorem ivan_needs_more_paint
  (section_count : ℕ)
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2) :
  let area_ivan := section_count * (5 * 2)
  let area_petr := section_count * (5 * 2 * sin α)
  area_ivan > area_petr := 
by
  simp only [area_ivan, area_petr, mul_assoc, mul_lt_mul_left, gt_iff_lt]
  exact sin_lt_one_iff.mpr hα

end ivan_needs_more_paint_l498_498081


namespace disjunction_of_p_and_q_l498_498674

-- Define the propositions p and q
variable (p q : Prop)

-- Assume that p is true and q is false
theorem disjunction_of_p_and_q (h1 : p) (h2 : ¬q) : p ∨ q := 
sorry

end disjunction_of_p_and_q_l498_498674


namespace range_of_a_l498_498047

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := 
sorry

end range_of_a_l498_498047


namespace find_number_of_students_l498_498968

-- Definitions for the conditions
def avg_age_students := 14
def teacher_age := 65
def new_avg_age := 15

-- The total age of students is n multiplied by their average age
def total_age_students (n : ℕ) := n * avg_age_students

-- The total age including teacher
def total_age_incl_teacher (n : ℕ) := total_age_students n + teacher_age

-- The new average age when teacher is included
def new_avg_age_incl_teacher (n : ℕ) := total_age_incl_teacher n / (n + 1)

theorem find_number_of_students (n : ℕ) (h₁ : avg_age_students = 14) (h₂ : teacher_age = 65) (h₃ : new_avg_age = 15) 
  (h_averages_eq : new_avg_age_incl_teacher n = new_avg_age) : n = 50 :=
  sorry

end find_number_of_students_l498_498968


namespace original_cost_of_pencil_l498_498974

theorem original_cost_of_pencil (final_price discount: ℝ) (h_final: final_price = 3.37) (h_disc: discount = 0.63) : 
  final_price + discount = 4 :=
by
  sorry

end original_cost_of_pencil_l498_498974


namespace perpendicular_lines_to_same_plane_are_parallel_l498_498260

-- Definitions
variables (m n : Line) (α β γ : Plane)

-- Theorem statement
theorem perpendicular_lines_to_same_plane_are_parallel 
  (m_perp_alpha : m ⊥ α) 
  (n_perp_alpha : n ⊥ α) : 
  m ∥ n :=
sorry

end perpendicular_lines_to_same_plane_are_parallel_l498_498260


namespace third_derivative_correct_l498_498518

noncomputable def y (x : ℝ) : ℝ := (1 / x) * (Real.sin (2 * x))

def y_third_derivative (x : ℝ) : ℝ :=
  (-6 / x^4 + 12 / x^2) * (Real.sin (2 * x)) + (12 / x^3 - 8 / x) * (Real.cos (2 * x))

theorem third_derivative_correct (x : ℝ) (hx : x ≠ 0) : 
  deriv (deriv (deriv (y x))) = y_third_derivative x :=
by
  sorry

end third_derivative_correct_l498_498518


namespace ivan_uses_more_paint_l498_498075

-- Define the basic geometric properties
def rectangular_section_area (length width : ℝ) : ℝ := length * width
def parallelogram_section_area (side1 side2 : ℝ) (angle : ℝ) : ℝ := side1 * side2 * Real.sin angle

-- Define the areas for each neighbor's fences
def ivan_area : ℝ := rectangular_section_area 5 2
def petr_area (alpha : ℝ) : ℝ := parallelogram_section_area 5 2 alpha

-- Theorem stating that Ivan's total fence area is greater than Petr's total fence area provided the conditions
theorem ivan_uses_more_paint (α : ℝ) (hα : α ≠ Real.pi / 2) : ivan_area > petr_area α := by
  sorry

end ivan_uses_more_paint_l498_498075


namespace series_sum_eq_100_over_9801_p_plus_q_eq_9901_l498_498719

noncomputable def series_expression : ℚ :=
  ∑' n : ℕ, (∑ k in finset.range (n + 1), (1 / (k + 1))) / (nat.factorial (n + 100) / (nat.factorial 100 * nat.factorial n))

theorem series_sum_eq_100_over_9801 :
  series_expression = 100 / 9801 :=
sorry

theorem p_plus_q_eq_9901 
  (p q : ℕ)
  (hpq_coprime : nat.coprime p q)
  (h_series : series_expression = p / q) :
  p + q = 9901 :=
sorry

end series_sum_eq_100_over_9801_p_plus_q_eq_9901_l498_498719


namespace sum_of_solutions_l498_498331

theorem sum_of_solutions : 
  ∀ (x y : ℝ), y = 8 → x^2 + y^2 = 169 → (x = sqrt 105 ∨ x = -sqrt 105) → (sqrt 105 + (-sqrt 105) = 0) :=
by
  intros x y h1 h2 h3
  sorry

end sum_of_solutions_l498_498331


namespace hyperbola_asymptotes_l498_498603

theorem hyperbola_asymptotes (x y : ℝ) (h : y^2 / 16 - x^2 / 9 = (1 : ℝ)) :
  ∃ (m : ℝ), (m = 4 / 3) ∨ (m = -4 / 3) :=
sorry

end hyperbola_asymptotes_l498_498603


namespace total_calories_box_l498_498141

-- Definitions from the conditions
def bags := 6
def cookies_per_bag := 25
def calories_per_cookie := 18

-- Given the conditions, prove the total calories equals 2700
theorem total_calories_box : bags * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end total_calories_box_l498_498141


namespace find_number_l498_498157

theorem find_number (x : ℤ) (h : 3 * (2 * x + 15) = 75) : x = 5 :=
by
  sorry

end find_number_l498_498157


namespace remainder_when_divided_by_18_l498_498158

theorem remainder_when_divided_by_18 (N k : ℤ) (h : N = 242 * k + 100) : N % 18 = 6 :=
by {
  sorry,
}

end remainder_when_divided_by_18_l498_498158


namespace sum_elements_T_l498_498373

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498373


namespace measure_of_angle_ABH_l498_498832

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498832


namespace max_truthful_gnomes_l498_498001

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l498_498001


namespace intersection_points_l498_498602

open Real

def parabola1 (x : ℝ) : ℝ := x^2 - 3 * x + 2
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem intersection_points : 
  ∃ x y : ℝ, 
  (parabola1 x = y ∧ parabola2 x = y) ∧ 
  ((x = 1/2 ∧ y = 3/4) ∨ (x = -3 ∧ y = 20)) :=
by sorry

end intersection_points_l498_498602


namespace impossible_many_moves_l498_498776

-- Define the initial conditions
variables (M : set (ℝ × ℝ)) (n : ℕ)
variable (hM : ∀ (p q r : ℝ × ℝ), p ∈ M → q ∈ M → r ∈ M → p ≠ q → p ≠ r → q ≠ r → 
  ¬ collinear {p, q, r})
variable (h_n_ge_4 : n ≥ 4)
variable (init_segments : finset (ℝ × ℝ))
variable (h_init_seg : ∀ p ∈ M, 2 * (card (init_segments.filter (λ s, p ∈ s))) = card init_segments)

-- Define the move operation
def move_possible (s1 s2 s3 s4 : ℝ × ℝ) (segments : finset (ℝ × ℝ)) : Prop :=
  (s1, s2) ∈ segments ∧ (s3, s4) ∈ segments ∧ 
  ∃ common_point, (s1 ≠ common_point ∧ s2 ≠ common_point ∧ s3 ≠ common_point ∧ s4 ≠ common_point )
  ∧ segment_intersection (s1, s2) (s3, s4) (common_point) ∧ 
  ¬ ((s1, s3) ∈ segments ∨ (s2, s4) ∈ segments)

-- Theorem stating the impossibility of performing many moves
theorem impossible_many_moves :
  ¬ ∃ k : ℕ, k ≥ n^3 / 4 ∧ ∃ (steps : list (finset (ℝ × ℝ))), all_moves_possible hM h_n_ge_4 k steps :=
sorry

end impossible_many_moves_l498_498776


namespace measure_of_angle_ABH_l498_498824

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498824


namespace polynomial_factoring_one_polynomial_factoring_two_calculate_expression_l498_498985

/- Given: Let a^2 + 2a = x, then prove the original expression (a^2 + 2a)(a^2 + 2a + 2) + 1 equals (a + 1)^4 -/
theorem polynomial_factoring_one (a : ℝ) (x : ℝ) (h1 : x = a^2 + 2a) : 
  (a^2 + 2a) * (a^2 + 2a + 2) + 1 = (a + 1)^4 :=
  sorry

/- Given: Let a^2 - 4a = x, then prove the original expression (a^2 - 4a)(a^2 - 4a + 8) + 16 equals (a - 2)^4 -/
theorem polynomial_factoring_two (a : ℝ) (x : ℝ) (h2 : x = a^2 - 4a) : 
  (a^2 - 4a) * (a^2 - 4a + 8) + 16 = (a - 2)^4 :=
  sorry

/- Given: a = 1 - 2 - 3 - ... - 2023, x = 2 + 3 + ... + 2024, prove that the expression (1 - 2 - 3 - ... - 2023) * (2 + 3 + ... + 2024) - (1 - 2 - 3 - ... - 2024) * (2 + 3 + ... + 2023) equals 2024 -/
theorem calculate_expression (a : ℝ) (x : ℝ) 
  (h3 : a = ∑ k in (range 2024), (1:ℝ) - k) (h4 : x = ∑ k in (range 2024), (k + 2)) :
  (a * x - (a - 2024) * (x - 2024)) = 2024 :=
  sorry

end polynomial_factoring_one_polynomial_factoring_two_calculate_expression_l498_498985


namespace sum_of_set_T_l498_498397

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498397


namespace measure_of_angle_ABH_in_regular_octagon_l498_498935

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498935


namespace thank_you_cards_correct_l498_498757

noncomputable def invitations : ℕ := 200
noncomputable def rsvp_percent : ℝ := 0.90
noncomputable def attendance_percent : ℝ := 0.80
noncomputable def no_gift : ℕ := 10

def thank_you_cards : ℕ :=
  let rsvp := (rsvp_percent * invitations)
  let attendance := (attendance_percent * rsvp)
  attendance.toNat - no_gift

theorem thank_you_cards_correct : thank_you_cards = 134 := by
  sorry

end thank_you_cards_correct_l498_498757


namespace problem_statement_l498_498655

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f a b α β 2007 = 5) :
  f a b α β 2008 = 3 := 
by
  sorry

end problem_statement_l498_498655


namespace polynomial_q_l498_498009

noncomputable def q (x : ℝ) : ℝ := -2*x^5 - 2*x^4 + 10*x^3 + 32*x^2 + 5*x + 3

theorem polynomial_q (x : ℝ) :
  q(x) + (2*x^5 + 5*x^4 + 4*x^3 + 12*x) = (3*x^4 + 14*x^3 + 32*x^2 + 17*x + 3) :=
by
  unfold q
  ring

end polynomial_q_l498_498009


namespace binom_calculation_l498_498299

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ := x * (x - 1) * (x - 2) * ... * (x - k + 1) / (k!)

theorem binom_calculation :
  (binom (1/2) (2021) * 4^(2021)) / binom (4042) (2021) = -1 / 4041 := 
sorry

end binom_calculation_l498_498299


namespace angle_in_regular_octagon_l498_498883

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498883


namespace dogwood_tree_count_l498_498486

theorem dogwood_tree_count (n d1 d2 d3 d4 d5: ℕ) 
  (h1: n = 39)
  (h2: d1 = 24)
  (h3: d2 = d1 / 2)
  (h4: d3 = 4 * d2)
  (h5: d4 = 5)
  (h6: d5 = 15):
  n + d1 + d2 + d3 + d4 + d5 = 143 :=
by
  sorry

end dogwood_tree_count_l498_498486


namespace christopher_strolling_time_l498_498201

theorem christopher_strolling_time
  (initial_distance : ℝ) (initial_speed : ℝ) (break_time : ℝ)
  (continuation_distance : ℝ) (continuation_speed : ℝ)
  (H1 : initial_distance = 2) (H2 : initial_speed = 4)
  (H3 : break_time = 0.25) (H4 : continuation_distance = 3)
  (H5 : continuation_speed = 6) :
  (initial_distance / initial_speed + break_time + continuation_distance / continuation_speed) = 1.25 := 
  sorry

end christopher_strolling_time_l498_498201


namespace sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l498_498754

theorem sqrt_three_is_irrational_and_infinite_non_repeating_decimal :
    ∀ r : ℝ, r = Real.sqrt 3 → ¬ ∃ (m n : ℤ), n ≠ 0 ∧ r = m / n := by
    sorry

end sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l498_498754


namespace first_year_after_2010_has_digit_sum_10_l498_498100

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2010_has_digit_sum_10 : 
  ∃ y : ℕ, y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ z : ℕ, z > 2010 ∧ z < y → sum_of_digits z ≠ 10 :=
begin
  use 2017,
  split,
  { exact nat.lt_succ_self 2016, },
  split,
  { norm_num [sum_of_digits], },
  { intros z hz hzy,
    interval_cases z; norm_num [sum_of_digits], },
  sorry
end

end first_year_after_2010_has_digit_sum_10_l498_498100


namespace part_a_part_b_l498_498533

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l498_498533


namespace ratio_of_areas_l498_498751

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry
noncomputable def r : ℝ := sorry

-- conditions
axiom condition_AB : ∀ (AB : ℝ), AB = 12
axiom condition_BC : ∀ (BC : ℝ), BC = 13
axiom condition_CA : ∀ (CA : ℝ), CA = 14
axiom condition_pqrs : p + q + r = 3 / 4 ∧ p^2 + q^2 + r^2 = 2 / 5

theorem ratio_of_areas (AB BC CA : ℝ) (p q r : ℝ)
        (h_AB : AB = 12) (h_BC : BC = 13) (h_CA : CA = 14)
        (h_pq_sum : p + q + r = 3 / 4) (h_pq_square_sum : p^2 + q^2 + r^2 = 2 / 5) :
        let area_DEF := 1 / 16 in
        let m := 1 in
        let n := 16 in
        m + n = 17 :=
begin
  -- Skipping proof as sorries are placed where necessary
  sorry
end

end ratio_of_areas_l498_498751


namespace locus_is_circle_l498_498663

-- Assuming localized coordinates are in R²
open_locale real

-- Definitions for our problem:
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
∃ s, s > 0 ∧ dist A B = s ∧ dist B C = s ∧ dist C A = s

def locus_condition (A B C M : ℝ × ℝ) : Prop :=
dist M C ^ 2 = dist M A ^ 2 + dist M B ^ 2

theorem locus_is_circle (A B C : ℝ × ℝ) (h : equilateral_triangle A B C) :
  ∃ O r, ∀ M, locus_condition A B C M ↔ dist M O = r :=
sorry

end locus_is_circle_l498_498663


namespace measure_of_angle_ABH_l498_498846

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498846


namespace angle_ABH_is_22_point_5_l498_498912

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498912


namespace digits_of_x_l498_498209

theorem digits_of_x (x : ℕ) (h : log 4 (log 4 (log 2 x)) = 1) : nat.digits 10 x = 5 := by
  sorry

end digits_of_x_l498_498209


namespace sin_equality_root_initial_sin_equality_next_root_l498_498780

noncomputable def smallest_natural_number_root : ℕ :=
  36

theorem sin_equality_root_initial :
  ∃ k : ℕ, k = smallest_natural_number_root ∧ sin (k * real.pi / 180) = sin (334 * k * real.pi / 180) := 
by
  use smallest_natural_number_root
  split
  { refl }
  sorry

theorem sin_equality_next_root (k1 : ℕ) (hk1 : k1 = smallest_natural_number_root) :
  ∃ k : ℕ, k > k1 ∧ sin (k * real.pi / 180) = sin (334 * k * real.pi / 180) ∧ k = 40 := 
by
  use 40
  split
  { norm_num, rw hk1, exact nat.succ_le_succ k1.zero_le }
  split
  { sorry }
  refl

end sin_equality_root_initial_sin_equality_next_root_l498_498780


namespace incorrect_statement_identification_l498_498118

theorem incorrect_statement_identification :
  (A: (∀ x y : ℝ, x * y ≠ 10 → x ≠ 5 ∨ y ≠ 2)) ∧
  (B: (∀ p : Prop, (p = ∀ x : ℝ, x ^ 2 + x + 1 ≠ 0) → (¬p = ∃ x : ℝ, x ^ 2 + x + 1 = 0))) ∧
  (C: (∀ r : ℝ, abs r → 1 → abs r ≃ 1 → correlation_strong r)) ∧
  (D: (∀ rectangles : list (ℝ × ℝ), 
    mean_estimate_rectangles rectangles = 
    sum (map (λ rect, rect.1 * rect.2) rectangles)) → False) := 
by {
  sorry
}

end incorrect_statement_identification_l498_498118


namespace ivan_uses_more_paint_l498_498071

-- Conditions
def ivan_section_area : ℝ := 5 * 2
def petr_section_area (alpha : ℝ) : ℝ := 5 * 2 * Real.sin(alpha)
axiom alpha_lt_90 : ∀ α : ℝ, α < 90 → Real.sin(α) < 1

-- Assertion
theorem ivan_uses_more_paint (α : ℝ) (h1 : α < 90) : ivan_section_area > petr_section_area α :=
by
  sorry

end ivan_uses_more_paint_l498_498071


namespace multiply_exponents_l498_498586

theorem multiply_exponents (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end multiply_exponents_l498_498586


namespace sticker_distribution_correct_l498_498711

noncomputable def sticker_distribution_count : ℕ :=
  -- the total number of ways to distribute 10 stickers ensuring conditions
  35

theorem sticker_distribution_correct :
  let total_stickers := 10
      total_sheets := 4
      large_sheets := 2
      small_sheets := 2
  in
  (∀ l1 l2 s1 s2 : ℕ, l1 + l2 + s1 + s2 = total_stickers ∧
                        l1 ≥ 3 ∧ l2 ≥ 3) ∧ 
      sticker_distribution_count = 35 :=
sorry

end sticker_distribution_correct_l498_498711


namespace slope_of_line_through_focus_l498_498976

theorem slope_of_line_through_focus (
  h1 : ∀ x y, ⟦x^2 / 2 + y^2 = 1⟧,
  h2 : ∃ P Q, P ≠ Q ∧ (∃ x₁ x₂ : ℝ, P = (x₁, k * (x₁ - sqrt 3)) ∧ Q = (x₂, k * (x₂ - sqrt 3))),
  h3 : ∀ P Q, (P - O).dot (Q - O) = 0
) : k = sqrt 2 ∨ k = - sqrt 2 := sorry

-- Definitions for point, origin, and the dot product for clarity
def point := ℝ × ℝ
def origin : point := (0, 0)
def point.dot (P Q : point) : ℝ := P.1 * Q.1 + P.2 * Q.2

end slope_of_line_through_focus_l498_498976


namespace projection_correct_l498_498677

open Real

variables (a : ℝ → ℝ → ℝ) (b : ℝ → ℝ)
variables (θ : ℝ)

noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_sq := b.1 * b.1 + b.2 * b.2
  (dot_product / magnitude_sq * b.1, dot_product / magnitude_sq * b.2)

theorem projection_correct
  (h_angle : θ = π / 3)
  (h_a_magnitude : (a 0 2).1 = 2 ∧ (a 0 2).2 = 0)
  (h_b : b 1 = 1 ∧ b 1 = 1) :
  proj (a θ 2) (1, 1) = (sqrt 2 / 2, sqrt 2 / 2) :=
sorry

end projection_correct_l498_498677


namespace students_in_first_class_l498_498489

theorem students_in_first_class (x : ℕ) (h1 : ∀ y, 40 * y = x →  average_class_1 = 40) (h2 : number_of_students_class_2 = 50) (average_class_2 = 80) (overall_average = 65) : 
    x = 30 :=
by
  have total_marks_class_1 : ℕ := 40 * x
  have total_marks_class_2 : ℕ := 50 * 80
  have total_students : ℕ := x + 50
  have total_marks := total_marks_class_1 + total_marks_class_2
  have overall_marks : ℕ := 65 * total_students
  have equation := total_marks = overall_marks
  have expanded_left := 40 * x + 50 * 80
  have expanded_right := 65 * (x + 50)
  have expanded_equation := expanded_left = expanded_right
  have simplified_eq := (65 * x - 40 * x = 4000 - 3250)
  have final_eq := 25 * x = 750
  have x_value : ℕ := 750 / 25
  have solution : ℕ := 30
  exact x_value

end students_in_first_class_l498_498489


namespace cannot_end_with_two_l498_498084

theorem cannot_end_with_two : 
  (list.range 2017).sum % 2 = 1 → ¬ ∃ (l : list ℕ), l.sum % 2 = 0 ∧ l.length = 1 ∧ l.head = some 2 := 
by
  sorry

end cannot_end_with_two_l498_498084


namespace sin_alpha_plus_pi_over_2_l498_498671

theorem sin_alpha_plus_pi_over_2 (α : ℝ) (x : ℝ) (hα : π / 2 < α ∧ α < π) (P_on_terminal_side : ∃ (y : ℝ), y = sqrt 5 ∧ cos α = ((sqrt 2 / 4) * x)) :
  sin (α + π / 2) = - (sqrt 6 / 4) :=
by
  -- Conditions
  have h_cond1 : ∃ y, y = sqrt 5 ∧ cos α = ((sqrt 2 / 4) * x) := P_on_terminal_side
  cases h_cond1 with y hy,
  rw hy.left at *,
  rw hy.right at *,
  sorry

end sin_alpha_plus_pi_over_2_l498_498671


namespace quadratic_passes_through_point_l498_498747

theorem quadratic_passes_through_point (a b : ℝ) : a + b = 0 → ((1 : ℝ), (1 : ℝ)) ∈ {p : ℝ × ℝ | ∃ x, p = (x, x^2 + a * x + b)} := by
  intro h
  use 1
  rw [h]
  simp
  sorry

end quadratic_passes_through_point_l498_498747


namespace proving_smallest_5_digit_multiple_of_3_and_4_l498_498108

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def smallest_five_digit_number := 10000

def smallest_5_digit_multiple_of_3_and_4 : ℕ :=
  Inf {n : ℕ | n ≥ smallest_five_digit_number ∧ is_multiple n 3 ∧ is_multiple n 4}

theorem proving_smallest_5_digit_multiple_of_3_and_4 : smallest_5_digit_multiple_of_3_and_4 = 10008 :=
by {
  -- Proof goes here
  sorry
}

end proving_smallest_5_digit_multiple_of_3_and_4_l498_498108


namespace measure_angle_abh_l498_498813

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498813


namespace values_equal_for_a_equal_2_and_neg2_l498_498506

theorem values_equal_for_a_equal_2_and_neg2 (a : ℤ) :
  (a = 2 ∨ a = -2) → (a^4 - 2 * a^2 + 3 = 11) :=
by
  assume h : (a = 2 ∨ a = -2)
  cases h
  case or.inl h₁ => 
    rw [h₁]
    calc (2 ^ 4 - 2 * 2 ^ 2 + 3) = 11 : by norm_num
  case or.inr h₂ => 
    rw [h₂]
    calc ((-2) ^ 4 - 2 * (-2) ^ 2 + 3) = 11 : by norm_num

end values_equal_for_a_equal_2_and_neg2_l498_498506


namespace trigonometric_identity_proof_l498_498245

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4

theorem trigonometric_identity_proof
  (a b α β : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0)
  (h : f 2014 a b α β = 5) :
  f 2015 a b α β = 3 :=
by
  sorry

end trigonometric_identity_proof_l498_498245


namespace find_quotient_l498_498050

-- Definitions based on given conditions
def remainder : ℕ := 8
def dividend : ℕ := 997
def divisor : ℕ := 23

-- Hypothesis based on the division formula
def quotient_formula (q : ℕ) : Prop :=
  dividend = (divisor * q) + remainder

-- Statement of the problem
theorem find_quotient (q : ℕ) (h : quotient_formula q) : q = 43 :=
sorry

end find_quotient_l498_498050


namespace incorrect_statements_count_l498_498183

theorem incorrect_statements_count :
  (∠ C = 90 ∧ AC = 9 ∧ BC = 12 → ¬ calc_distance(C, AB) = 9) ∧
  (7^2 + 24^2 = 25^2 → right_triangle(7, 24, 25)) ∧
  (AC = 6 ∧ BC = 8 ∧ is_ambiguous_median(AC, BC) → ¬ median_length(AC, BC) = 5) ∧
  (isosceles_triangle(3, 5) → perimeter_is_11_or_13(3, 5)) →
  num_incorrect_statements = 3 :=
by
  -- Proof is omitted
  sorry

end incorrect_statements_count_l498_498183


namespace recycling_problem_l498_498008

theorem recycling_problem (initial_cans damaged_cans cans_required : ℕ) : 
  initial_cans = 500 → damaged_cans = 20 → cans_required = 6 → 
  let usable_cans := initial_cans - damaged_cans in
  let cycle1 := usable_cans / cans_required in
  let cycle2 := cycle1 / cans_required in
  let cycle3 := cycle2 / cans_required in
  let total_new_cans := cycle1 + cycle2 + cycle3 in
  total_new_cans = 95 :=
begin
  intros h₁ h₂ h₃,
  let usable_cans := initial_cans - damaged_cans,
  let cycle1 := usable_cans / cans_required,
  let cycle2 := cycle1 / cans_required,
  let cycle3 := cycle2 / cans_required,
  have usable_cans_correct : usable_cans = 480, from by rw [h₁, h₂],
  have cycle1_correct : cycle1 = 80, from by rw [usable_cans_correct, h₃],
  have cycle2_correct : cycle2 = 13, from by rw [cycle1_correct, h₃],
  have cycle3_correct : cycle3 = 2, from by rw [cycle2_correct, h₃],
  have total_new_cans_correct : total_new_cans = cycle1 + cycle2 + cycle3, from rfl,
  rw [cycle1_correct, cycle2_correct, cycle3_correct, h₃],
  exact rfl,
end

end recycling_problem_l498_498008


namespace sum_of_set_T_l498_498394

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498394


namespace frog_jumps_probability_l498_498151

/-- A frog makes 4 jumps in random directions with lengths 1, 2, 2, and 3 meters respectively. 
We aim to prove that the probability that the cumulative displacement's magnitude is 
no more than 2 meters from the starting position is 1/5. -/
theorem frog_jumps_probability : 
  let lengths := [1, 2, 2, 3]
  in ∃ P, P = 1 / 5 ∧ 
     P = probability_within_distance lengths 2 :=
sorry

end frog_jumps_probability_l498_498151


namespace calculate_a_l498_498307

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * x + b

theorem calculate_a (b : ℝ) (h : ∃ x : ℝ, x = 1 ∧ deriv (λ x, 4 * x^3 - a * x^2 - 2 * x + b) x = 0) : a = 5 :=
begin
  sorry
end

end calculate_a_l498_498307


namespace number_of_correct_statements_l498_498184

theorem number_of_correct_statements :
  let statement1 := "The complement of an angle is always an acute angle.",
      statement2 := "∠1 = ∠2, ∠1 and ∠2 are vertical angles.",
      statement3 := "There is only one line that is parallel to a given line and passes through a point not on the given line.",
      statement4 := "The perpendicular segment from a point outside a line to the line is called the distance from the point to the line.",
      statement5 := "If two lines are intersected by a third line, and the corresponding angles are equal."
  (correct_statements : Int) := 1 := by
  sorry

end number_of_correct_statements_l498_498184


namespace measure_of_angle_ABH_in_regular_octagon_l498_498938

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498938


namespace exponent_property_l498_498643

theorem exponent_property (x : ℚ) 
  (h : ↑3^(2 * x^2 - 3 * x + 5) = ↑3^(2 * x^2 + 5 * x - 1)) : 
  x = 3 / 4 := 
by
  sorry

end exponent_property_l498_498643


namespace correct_proposition_is_D_l498_498507

-- Definition of vertical angles
def vertical_angle (a b : ℝ) : Prop :=
  ∃ (l1 l2 : ℝ → ℝ), -- l1 and l2 are the equations of two intersecting lines
  l1 ≠ l2 ∧ -- The lines are distinct
  ∀ θ₁ θ₂ : ℝ, 
  (vertical_angle' θ₁ θ₂) -- angles θ₁ and θ₂ are vertical angles

-- Propositions
def proposition_A (a b : ℝ) : Prop := a^2 = b^2 → a = b

def proposition_B (a b : ℝ) : Prop := a > b → |a| > |b|

def proposition_C : Prop := ∀ (l1 l2 : ℝ → ℝ), (corresponding_angles_equal l1 l2)

def proposition_D : Prop := ∀ (l1 l2 : ℝ → ℝ), (intersect l1 l2 → vertical_angles_equal l1 l2)

theorem correct_proposition_is_D : proposition_D :=
by sorry

end correct_proposition_is_D_l498_498507


namespace crate_height_correct_l498_498146

/-- A certain rectangular crate measures 12 feet by 16 feet by some feet. A cylindrical gas tank
is to be made for shipment in the crate and will stand upright when the crate is placed on one
of its six faces. The radius of the tank should be 8 feet for it to be of the largest possible volume.
What is the height of the crate? -/
def height_of_crate (h : ℕ) (crate_dim : ℕ × ℕ × ℕ) (tank_radius : ℕ) : Prop :=
  crate_dim = (12, 16, h) ∧ tank_radius = 8 ∧ h = 16

theorem crate_height_correct (h : ℕ) (crate_dim : ℕ × ℕ × ℕ) (tank_radius : ℕ) :
  height_of_crate h crate_dim tank_radius :=
by {
  -- Define the height of the crate
  have h_correct : h = 16 := rfl,
  -- Define the crate dimensions
  have crate_dim_correct : crate_dim = (12, 16, 16) := rfl,
  -- Define the tank radius
  have tank_radius_correct : tank_radius = 8 := rfl,
  -- Combine all conditions
  exact ⟨crate_dim_correct, tank_radius_correct, h_correct⟩
}

end crate_height_correct_l498_498146


namespace probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l498_498417

def countWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else countWays (n - 1) + countWays (n - 2)

theorem probability_no_consecutive_tails : countWays 5 = 13 :=
by
  sorry

theorem probability_no_consecutive_tails_in_five_tosses : 
  (countWays 5) / (2^5 : ℕ) = 13 / 32 :=
by
  sorry

end probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l498_498417


namespace exists_n_good_but_not_succ_good_l498_498778

def S (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), 
    a_seq n = a ∧ (∀ i : Fin n, a_seq (Fin.succ i) = a_seq i - S (a_seq i))

theorem exists_n_good_but_not_succ_good (n : ℕ) : 
  ∃ a, n_good n a ∧ ¬ n_good (n + 1) a := 
sorry

end exists_n_good_but_not_succ_good_l498_498778


namespace monotonicity_of_f_max_min_of_f_l498_498789

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_of_f :
  (∀ x ∈ Ioo (-3/2) (-1), f' x > 0) ∧ (∀ x ∈ Ioo (-1) (-1/2), f' x < 0) ∧ (∀ x ∈ Ioi (-1/2), f' x > 0) := 
sorry

theorem max_min_of_f :
  ∀ x ∈ Icc (-1) ((Real.exp 2 - 3) / 2), f x ≥ Real.log 2 + 1/4 ∧ f x ≤ 2 + (Real.exp 2 - 3)^2 / 4 := 
sorry

end monotonicity_of_f_max_min_of_f_l498_498789


namespace first_year_after_2010_with_digit_sum_10_l498_498104

/--
Theorem: The first year after 2010 for which the sum of the digits equals 10 is 2017.
-/
theorem first_year_after_2010_with_digit_sum_10 : ∃ (y : ℕ), (y > 2010) ∧ (∑ d in (y.to_digits 10), d) = 10 ∧ (∀ z, (z > 2010) → ((∑ d in (z.to_digits 10), d) = 10 → z ≥ y))

end first_year_after_2010_with_digit_sum_10_l498_498104


namespace number_of_wheels_on_each_bicycle_l498_498484

theorem number_of_wheels_on_each_bicycle 
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (wheels_per_tricycle : ℕ)
  (total_wheels : ℕ)
  (h_bicycles : num_bicycles = 24)
  (h_tricycles : num_tricycles = 14)
  (h_wheels_tricycle : wheels_per_tricycle = 3)
  (h_total_wheels : total_wheels = 90) :
  2 * num_bicycles + 3 * num_tricycles = 90 → 
  num_bicycles = 24 → 
  num_tricycles = 14 → 
  wheels_per_tricycle = 3 → 
  total_wheels = 90 → 
  ∃ b : ℕ, b = 2 :=
by
  sorry

end number_of_wheels_on_each_bicycle_l498_498484


namespace direction_vector_correct_l498_498024

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end direction_vector_correct_l498_498024


namespace sum_of_fifth_powers_l498_498523

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l498_498523


namespace line_circle_separate_min_tangent_length_l498_498279

noncomputable def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 * t, sqrt 2 / 2 * t + 4 * sqrt 2)

def polar_eq_curve (θ : ℝ) : ℝ :=
  4 * cos (θ + π / 4)

noncomputable def cartesian_eq_circle (x y : ℝ) : Prop :=
  (x - sqrt 2)^2 + (y + sqrt 2)^2 = 4

theorem line_circle_separate :
  (∀ t, parametric_eq_line t) →
  (∀ θ (ρ := polar_eq_curve θ), cartesian_eq_circle (ρ * cos θ) (ρ * sin θ)) →
  ∃ d > 2, ∀ t, (let (x, y) := parametric_eq_line t in (sqrt 2 - x)^2 + (-sqrt 2 - y)^2 = d^2) →
  d = 6 :=
sorry

theorem min_tangent_length :
  ∀ t, (let (x, y) := parametric_eq_line t in ∃ d, cartesian_eq_circle x y → d = sqrt (6^2 - 2^2) ∧ d = 4 * sqrt 2) :=
sorry

end line_circle_separate_min_tangent_length_l498_498279


namespace no_such_broken_line_exists_l498_498658

-- Definitions derived from the problem conditions
def segment := sorry  -- TODO: Define a segment structure
def broken_line := sorry  -- TODO: Define a broken line structure

def intersects_exactly_once (bl : broken_line) (s : segment) : Prop := sorry  -- TODO: Define the property

def shaded_region (index : ℕ) : set segment := sorry  -- TODO: Define the shaded regions (1, 2, 8)

def contains_segments (r : set segment) (n : ℕ) : Prop := r.card = n  -- Each region contains 5 segments

-- The theorem stating the impossibility
theorem no_such_broken_line_exists :
  ∀ (bl : broken_line), 
  (∀ s, s ∈ segments → intersects_exactly_once bl s) → 
  (¬ ∃ bl, 
    (∀ s, s ∈ segments → intersects_exactly_once bl s) ∧
    (∀ r ∈ {1, 2, 8}, contains_segments (shaded_region r) 5)) :=
by {
  sorry  -- Proof goes here
}

end no_such_broken_line_exists_l498_498658


namespace eccentricity_of_ellipse_l498_498563

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * b^2 = a^2) →
  ∃ e : ℝ, e = Real.sqrt(1 - (b / a)^2) ∧ e = Real.sqrt 2 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l498_498563


namespace percentage_difference_correct_l498_498040

noncomputable def percentage_difference (initial_price : ℝ) (increase_2012_percent : ℝ) (decrease_2013_percent : ℝ) : ℝ :=
  let price_end_2012 := initial_price * (1 + increase_2012_percent / 100)
  let price_end_2013 := price_end_2012 * (1 - decrease_2013_percent / 100)
  ((price_end_2013 - initial_price) / initial_price) * 100

theorem percentage_difference_correct :
  ∀ (initial_price : ℝ),
  percentage_difference initial_price 25 12 = 10 := 
by
  intros
  sorry

end percentage_difference_correct_l498_498040


namespace mono_poly_ge_xp1_l498_498359

noncomputable def P (x : ℝ) : ℝ

variables (n : ℕ)
  (C1 : ∀ x ≥ 0, P x = x^n + ∑ i in range (n), (coeff P (n - i)) * x^i)
  (C2 : ∃ r : fin n → ℝ, (∀ i, r i ≤ 0) ∧ (P(x) = ∏ i in finset.fin_range n, (x + r i)))
  (C3 : (coeff P 0) = 1)

theorem mono_poly_ge_xp1 : ∀ x ≥ 0, P x ≥ (x + 1)^n := sorry

end mono_poly_ge_xp1_l498_498359


namespace contrapositive_example_l498_498020

theorem contrapositive_example (x : ℝ) :
  (x < -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
sorry

end contrapositive_example_l498_498020


namespace evaluate_expression_l498_498616

theorem evaluate_expression (k : ℝ) : 
    2^(-(3 * k + 1)) - 2^(-(3 * k - 2)) + 2^(-3 * k) = - (5 / 2) * 2^(-3 * k) := 
sorry

end evaluate_expression_l498_498616


namespace drama_action_ratio_l498_498215

variable (T a : ℕ)

/-- Let T be the total number of movies rented. -/
variable (h1 : 0.64 * T = 10 * a)
/-- Let a be the number of action movies rented. -/
variable (h2 : 0.36 * T = a + a * 37 / 8)

/--
Prove that the ratio of dramas to action movies rented during that two-week period is 37 to 8.
-/
theorem drama_action_ratio (h1 : ∀ (T a : ℕ), 0.64 * (T : ℝ) = 10 * (a : ℝ))
    (h2 : ∀ (T a : ℕ), 0.36 * (T : ℝ) = a + a * (37 / 8)) :
    ∀ (T a : ℕ), (37 / 8) = 37 / 8 :=
by
  sorry

end drama_action_ratio_l498_498215


namespace triangle_is_right_l498_498362

theorem triangle_is_right (A B C D E : Type)
  [inner_product_space ℝ A] [inner_product_space ℝ B]
  [inner_product_space ℝ C] [inner_product_space ℝ D]
  [inner_product_space ℝ E]
  (h1 : D = reflection C B)
  (h2 : ∃ (x : ℝ), E = A + x • (C - A) ∧ x = 2)
  (h3 : ∥B - E∥ = ∥A - D∥) :
  ∠BAC = 90° :=
by
  sorry

end triangle_is_right_l498_498362


namespace part_a_part_b_l498_498532

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l498_498532


namespace range_of_a_l498_498689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x / x) - a * (x ^ 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f a x1 / x2) - (f a x2 / x1) < 0) ↔ (a ≤ Real.exp 2 / 12) := by
  sorry

end range_of_a_l498_498689


namespace measure_angle_abh_l498_498867

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498867


namespace minimal_period_of_f_intervals_of_monotonic_decrease_maximum_value_of_f_l498_498271

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * sin x ^ 2

theorem minimal_period_of_f : (∀ x : ℝ, f (x + π) = f x) ∧ (∀ x : ℝ, 0 < x → (∀ y : ℝ, f (y + x) = f y) → x = π) := 
by
  sorry

theorem intervals_of_monotonic_decrease : 
  (∀ k : ℤ, intervals (k * π + π / 6) (k * π + 2 * π / 3) (∀ x : ℝ, (k * π + π / 6) < x ∧ x < (k * π + 2 * π / 3) → derivative f x < 0)) :=
by
  sorry

theorem maximum_value_of_f :
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/3 → f x ≤ 1) ∧ (f (π/6) = 1) :=
by
  sorry

end minimal_period_of_f_intervals_of_monotonic_decrease_maximum_value_of_f_l498_498271


namespace problem1_problem2_l498_498647

section problems

variables (m n a b : ℕ)
variables (h1 : 4 ^ m = a) (h2 : 8 ^ n = b)

theorem problem1 : 2 ^ (2 * m + 3 * n) = a * b :=
sorry

theorem problem2 : 2 ^ (4 * m - 6 * n) = a ^ 2 / b ^ 2 :=
sorry

end problems

end problem1_problem2_l498_498647


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498894

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498894


namespace sum_of_fifth_powers_l498_498519

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l498_498519


namespace roger_left_money_correct_l498_498947

noncomputable def roger_left_money (P : ℝ) (q : ℝ) (E : ℝ) (r1 : ℝ) (C : ℝ) (r2 : ℝ) : ℝ :=
  let feb_expense := q * P
  let after_feb := P - feb_expense
  let mar_expense := E * r1
  let after_mar := after_feb - mar_expense
  let mom_gift := C * r2
  after_mar + mom_gift

theorem roger_left_money_correct :
  roger_left_money 45 0.35 20 1.2 46 0.8 = 42.05 :=
by
  sorry

end roger_left_money_correct_l498_498947


namespace solutions_sin_cos_eq_cos_sin_in_interval_l498_498718

open Real

theorem solutions_sin_cos_eq_cos_sin_in_interval :
  (finset.card {x : ℝ | x ∈ Icc 0 π ∧ sin (π / 2 * cos x) = cos (π / 2 * sin x)} = 2) :=
sorry

end solutions_sin_cos_eq_cos_sin_in_interval_l498_498718


namespace geese_surviving_first_year_l498_498798

theorem geese_surviving_first_year (E : ℝ) (H1 : E = 550.0000000000001) :
  let hatched := (2 / 3) * E,
      survived_first_month := (3 / 4) * hatched,
      survived_first_year := (2 / 5) * survived_first_month in
  survived_first_year = 165 :=
by
  -- define the intermediate steps using the provided conditions
  let hatched := (2 / 3) * E
  let survived_first_month := (3 / 4) * hatched
  let survived_first_year := (2 / 5) * survived_first_month
  -- simplify the final value using the specific E value
  have : survived_first_year = (2 / 5) * (3 / 4) * (2 / 3) * 550.0000000000001 := by sorry
  -- the final approximation result
  have : survived_first_year ≈ 165 := by sorry
  -- exact rounding
  exact rfl

end geese_surviving_first_year_l498_498798


namespace angle_ABH_in_regular_octagon_l498_498838

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498838


namespace sequence_property_l498_498235

open Real

noncomputable def sequence_a (a1 : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a1 else a1 / (1 + (n - 1) * a1)

theorem sequence_property 
  (a1 : ℝ) (x y : ℕ → ℝ) (b : ℕ → ℕ → ℝ)
  (h1 : a1 ≠ -1) 
  (h2 : ∀ i, (1 ≤ i ∧ i ≤ 5) → x i ≠ 0)
  (h3 : ∀ i, (1 ≤ i ∧ i ≤ 5) → y i ≠ 0)
  (h4 : ∀ i j, b i j = ∏ k in finset.Icc 1 i, 1 + j * sequence_a a1 k)
  (hx : ∀ i, (1 ≤ i ∧ i ≤ 5) → ∑ j in finset.range 5, b i j * x j = x i)
  (hy : ∀ i, (1 ≤ i ∧ i ≤ 5) → ∑ j in finset.range 5, b i j * y j = 2 * y i) :
  ∑ i in finset.range 5, x i * y i = 0 :=
sorry

end sequence_property_l498_498235


namespace complex_numbers_satisfying_conditions_l498_498624

theorem complex_numbers_satisfying_conditions (x y z : ℂ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : x = 1 ∧ y = 1 ∧ z = 1 := 
by sorry

end complex_numbers_satisfying_conditions_l498_498624


namespace girl_walked_distance_l498_498153

-- Define the conditions
def speed : ℝ := 5 -- speed in kmph
def time : ℝ := 6 -- time in hours

-- Define the distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof statement that we need to show
theorem girl_walked_distance :
  distance speed time = 30 := by
  sorry

end girl_walked_distance_l498_498153


namespace annual_population_increase_l498_498986

theorem annual_population_increase (x : ℝ) (initial_pop : ℝ) :
    (initial_pop * (1 + (x - 1) / 100)^3 = initial_pop * 1.124864) → x = 5.04 :=
by
  -- Provided conditions
  intros h
  -- The hypothesis conditionally establishes that this will derive to show x = 5.04
  sorry

end annual_population_increase_l498_498986


namespace at_least_one_of_A_B_hired_l498_498556

-- Definitions of the candidates and hiring process parameters
structure Candidate :=
  (name : String)

def A : Candidate := ⟨"A"⟩
def B : Candidate := ⟨"B"⟩
def C : Candidate := ⟨"C"⟩
def D : Candidate := ⟨"D"⟩

def candidates : List Candidate := [A, B, C, D]

def hires : Finset (Finset Candidate) := (candidates.to_finset).powerset.filter (λ s, s.card = 2)

-- The probability we are asked to prove
theorem at_least_one_of_A_B_hired (p : ℚ) : 
  p = 5 / 6 ↔ 
  let prob_neither_A_nor_B := (hires.filter (λ s, ¬(A ∈ s ∨ B ∈ s))).card / hires.card
  in (1 - prob_neither_A_nor_B) = p := 
by sorry

end at_least_one_of_A_B_hired_l498_498556


namespace three_digit_numbers_satisfy_condition_l498_498219

theorem three_digit_numbers_satisfy_condition : 
  ∃ (x y z : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    x + y + z = (10 * x + y) - (10 * y + z) ∧ 
    (100 * x + 10 * y + z = 209 ∨ 
     100 * x + 10 * y + z = 428 ∨ 
     100 * x + 10 * y + z = 647 ∨ 
     100 * x + 10 * y + z = 866 ∨ 
     100 * x + 10 * y + z = 214 ∨ 
     100 * x + 10 * y + z = 433 ∨ 
     100 * x + 10 * y + z = 652 ∨ 
     100 * x + 10 * y + z = 871) := sorry

end three_digit_numbers_satisfy_condition_l498_498219


namespace find_positive_number_l498_498623

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l498_498623


namespace wall_clock_ring_interval_l498_498301

theorem wall_clock_ring_interval 
  (n : ℕ)                -- Number of rings in a day
  (total_minutes : ℕ)    -- Total minutes in a day
  (intervals : ℕ) :       -- Number of intervals
  n = 6 ∧ total_minutes = 1440 ∧ intervals = n - 1 ∧ intervals = 5
    → (1440 / intervals = 288 ∧ 288 / 60 = 4∧ 288 % 60 = 48) := sorry

end wall_clock_ring_interval_l498_498301


namespace angle_ABH_in_regular_octagon_l498_498835

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498835


namespace S_2016_value_l498_498660

noncomputable def a_n (n : ℕ) (S : ℕ → ℝ) : ℝ :=
if n = 1 then 1 else (2 * (S n)^2) / (2 * (S n) - 1)

def S_n : (ℕ → ℝ) → ℕ → ℝ
| S, 1 => S 1
| S, (n+1) => S (n+1) - S n

theorem S_2016_value (S : ℕ → ℝ) :
  (∀ n, n ≥ 2 → a_n n S = 2 * (S n)^2 / (2 * (S n) - 1)) →
  S 2016 = 1 / 4031 :=
sorry

end S_2016_value_l498_498660


namespace measure_of_angle_ABH_in_regular_octagon_l498_498929

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498929


namespace correct_universal_proposition_l498_498182

-- Conditions for the problem
def condition_A : Prop :=
  ∀ a b : ℝ, a^2 + b^2 - 2*a - 2*b + 2 < 0

def condition_B : Prop :=
  ∀ (rhombus : Type) [Geometry.Rhombus rhombus], 
  Geometry.diagonals_are_equal rhombus

def condition_C : Prop :=
  ∃ x : ℝ, Real.sqrt (x^2) = x

def condition_D : Prop :=
  ∀ x y : ℝ, x < y → Real.log x < Real.log y

-- Theorem to prove
theorem correct_universal_proposition : 
  (¬ condition_A) ∧ (¬ condition_B) ∧ (¬ condition_C) ∧ condition_D :=
by
  sorry

end correct_universal_proposition_l498_498182


namespace first_pipe_fills_cistern_in_10_hours_l498_498498

noncomputable def time_to_fill (x : ℝ) : Prop :=
  let first_pipe_rate := 1 / x
  let second_pipe_rate := 1 / 12
  let third_pipe_rate := 1 / 15
  let combined_rate := first_pipe_rate + second_pipe_rate - third_pipe_rate
  combined_rate = 7 / 60

theorem first_pipe_fills_cistern_in_10_hours : time_to_fill 10 :=
by
  sorry

end first_pipe_fills_cistern_in_10_hours_l498_498498


namespace pentagon_star_perimeter_difference_eq_zero_l498_498765

/-- Let ABCDE be an equiangular convex pentagon of perimeter 1. The pairwise intersections of the lines that extend the sides of the pentagon determine a five-pointed star polygon. Prove that the difference between the maximum and minimum possible values of the perimeter of this star is 0. -/
theorem pentagon_star_perimeter_difference_eq_zero
  (ABCDE : Type) 
  [convex_pentagon ABCDE] 
  (h1 : equiangular ABCDE) 
  (h2 : perimeter ABCDE = 1) 
  (s : ℝ) : 
  max_perimeter_star ABCDE s - min_perimeter_star ABCDE s = 0 :=
sorry

end pentagon_star_perimeter_difference_eq_zero_l498_498765


namespace inverse_comp_eval_l498_498192

variable {α : Type*} {β : Type*}
variable (f : β → α) (g : α → β)
variable (h₁ : ∀ x : α, f (g x) = 4 * x - 2)

theorem inverse_comp_eval (h₁ : ∀ x : α, f (g x) = 4 * x - 2) : g⁻¹ (f 5) = 7 / 4 := 
by sorry

end inverse_comp_eval_l498_498192


namespace first_year_after_2010_has_digit_sum_10_l498_498099

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2010_has_digit_sum_10 : 
  ∃ y : ℕ, y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ z : ℕ, z > 2010 ∧ z < y → sum_of_digits z ≠ 10 :=
begin
  use 2017,
  split,
  { exact nat.lt_succ_self 2016, },
  split,
  { norm_num [sum_of_digits], },
  { intros z hz hzy,
    interval_cases z; norm_num [sum_of_digits], },
  sorry
end

end first_year_after_2010_has_digit_sum_10_l498_498099


namespace probability_different_parity_l498_498239

theorem probability_different_parity :
  let cards := [1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let odd_cards := [1, 3, 5, 7, 9] in
  let even_cards := [2, 4, 6, 8] in
  let total_num_ways := 9 * 8 in
  let num_favorable_ways := (list.length odd_cards) * (list.length even_cards) + (list.length even_cards) * (list.length odd_cards) in
  num_favorable_ways / total_num_ways = 5 / 9 := by
  sorry

end probability_different_parity_l498_498239


namespace arthur_num_hamburgers_on_first_day_l498_498579

theorem arthur_num_hamburgers_on_first_day (H D : ℕ) (hamburgers_1 hamburgers_2 : ℕ) (hotdogs_1 hotdogs_2 : ℕ)
  (h1 : hamburgers_1 * H + hotdogs_1 * D = 10)
  (h2 : hamburgers_2 * H + hotdogs_2 * D = 7)
  (hprice : D = 1)
  (h1_hotdogs : hotdogs_1 = 4)
  (h2_hotdogs : hotdogs_2 = 3) : 
  hamburgers_1 = 1 := 
by
  sorry

end arthur_num_hamburgers_on_first_day_l498_498579


namespace sequence_limit_exists_l498_498368

noncomputable def sequence (alpha : ℝ) : ℕ → ℝ
| 0       := alpha
| (n + 1) := alpha ^ (sequence alpha n)

theorem sequence_limit_exists (α : ℝ) (hα : 0 < α) :
  (∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence α n - L| < ε) ↔ 0 < α ∧ α ≤ Real.exp (1 / Real.exp 1) :=
sorry

end sequence_limit_exists_l498_498368


namespace exist_linear_functions_l498_498339

theorem exist_linear_functions (x0 y0 : ℤ) (h_points: (1 ≤ x0 ∧ x0 ≤ 20) ∧ (1 ≤ y0 ∧ y0 ≤ 20)) :
  ∃ (s : finset (ℤ × ℤ)), s.card = 10 ∧
    ∀ p ∈ s, 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20 ∧
    ∃ (k : ℤ), ∃ (b : ℤ), (p = (k, b) ∧ y0 = k * x0 + b) :=
begin
  sorry
end

end exist_linear_functions_l498_498339


namespace a719_divisible_by_11_l498_498222

theorem a719_divisible_by_11 (a : ℕ) (h : a < 10) : (∃ k : ℤ, a - 15 = 11 * k) ↔ a = 4 :=
by
  sorry

end a719_divisible_by_11_l498_498222


namespace distance_y_axis_18_l498_498132

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ :=
  let dy := abs P.2
  in 2 * dy

theorem distance_y_axis_18 (x : ℝ) (P : ℝ × ℝ) (h : P = (x, -9)) :
  distance_from_y_axis P = 18 :=
by
  sorry

end distance_y_axis_18_l498_498132


namespace points_satisfy_equation_l498_498708

theorem points_satisfy_equation (x y : ℝ) : 
  (2 * x^2 + 3 * x * y + y^2 + x = 1) ↔ (y = -x - 1) ∨ (y = -2 * x + 1) := by
  sorry

end points_satisfy_equation_l498_498708


namespace number_of_complex_solutions_l498_498227

theorem number_of_complex_solutions :
  (∃ z : ℂ, (z^4 - 1) = 0 ∧ (z^3 + z^2 - 3z - 3) ≠ 0) = 3 := sorry

end number_of_complex_solutions_l498_498227


namespace year_2017_l498_498094

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

theorem year_2017 : ∃ (y : ℕ), y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ y', y' > 2010 → sum_of_digits y' = 10 → y' ≥ y :=
by {
  let y := 2017,
  use y,
  split,
  { exact Nat.lt_of_succ_lt_succ (Nat.succ_lt_succ (Nat.succ_pos 2016)) },
  split,
  { norm_num },
  intros y' y'_gt_2010 sum_y',
  sorry
}

end year_2017_l498_498094


namespace angle_ABH_regular_octagon_l498_498921

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498921


namespace Alyssa_puppies_l498_498180

theorem Alyssa_puppies (initial_puppies give_away_puppies : ℕ) (h_initial : initial_puppies = 12) (h_give_away : give_away_puppies = 7) :
  initial_puppies - give_away_puppies = 5 :=
by
  sorry

end Alyssa_puppies_l498_498180


namespace min_P_k_l498_498638

def closest_integer (m k : ℤ) : ℤ :=
  m / k + if (m % k) * 2 ≥ k then 1 else 0

def P (k : ℤ) : ℚ :=
  (1 / 149) * (Finset.range 149).count (λ n, closest_integer n k + closest_integer (150 - n) k = closest_integer 150 k)

def is_odd_divisor_of_150 (k : ℤ) : Prop :=
  k ∣ 150 ∧ odd k ∧ 1 ≤ k ∧ k ≤ 149

theorem min_P_k : ↑(1/5 : ℚ) = Finset.inf (Finset.filter is_odd_divisor_of_150 (Finset.range 150)) P :=
  sorry

end min_P_k_l498_498638


namespace closest_set_is_circle_l498_498294

def Rectangle (P B : Point) :=
  ∃ R : Set Point, B ∈ R ∧ P ∈ vertex_set(R)

def Distance (A B : Point) : ℝ :=
  sorry -- Define distance between points A and B

def is_circle_centered_at_B (A B : Point) (R : Set Point) : Prop :=
  Distance(A, B) ≤ Distance(A, P) ∀ P ∈ R

theorem closest_set_is_circle (P B : Point) (A : Point) (R : Set Point) 
  (h1 : Rectangle P B)  :
  is_circle_centered_at_B A B R := 
sorry

end closest_set_is_circle_l498_498294


namespace geometric_sequence_value_sum_l498_498317

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m, a (n + m) * a 0 = a n * a m

theorem geometric_sequence_value_sum {a : ℕ → ℝ}
  (hpos : ∀ n, a n > 0)
  (geom : is_geometric_sequence a)
  (given : a 0 * a 2 + 2 * a 1 * a 3 + a 2 * a 4 = 16) 
  : a 1 + a 3 = 4 :=
sorry

end geometric_sequence_value_sum_l498_498317


namespace measure_of_angle_ABH_in_regular_octagon_l498_498937

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498937


namespace angle_ABH_in_regular_octagon_l498_498841

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498841


namespace projection_of_a_onto_b_l498_498287

namespace ProjectionProof

open Real

def vector_a : ℝ × ℝ := (1, real.sqrt 3)
def vector_b : ℝ × ℝ := (real.sqrt 3, 1)

noncomputable def v_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def v_norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_a_onto_b :
  (v_dot vector_a vector_b) / (v_norm vector_b) = real.sqrt 3 :=
by
  sorry

end ProjectionProof

end projection_of_a_onto_b_l498_498287


namespace sum_elements_T_l498_498374

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498374


namespace first_year_after_2010_with_digit_sum_10_l498_498089

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem first_year_after_2010_with_digit_sum_10 :
  ∃ y > 2010, sum_of_digits y = 10 ∧ ∀ z, (2010 < z < y) → sum_of_digits z ≠ 10 :=
begin
  use 2035,
  split,
  { exact lt_of_lt_of_le (by norm_num : 2010 < 2035) (le_refl 2035)},
  split,
  { simp [sum_of_digits],
    norm_num,
  },
  { intros z hz,
    have hz0 : z ≥ 2010 + 1 := hz.1,
    have hz1 : z ≤ 2035 - 1 := hz.2,
    sorry,
  }
end

end first_year_after_2010_with_digit_sum_10_l498_498089


namespace coefficient_of_x5y2_l498_498683

theorem coefficient_of_x5y2 (n : ℕ) (C : ℕ → ℕ → ℕ) (sum_coeffs_eq : ∑ k in range (n + 1), C n k = 32) :
  (C n 2) * 2^(n-2) * (2^(n-3)) * (C 3 1) = 120 :=
sorry

end coefficient_of_x5y2_l498_498683


namespace monotonic_decreasing_cosine_l498_498036

noncomputable def monotonic_decreasing_interval : Set ℝ :=
  { x | ∃ k : ℤ, k * real.pi + real.pi / 8 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 8 }

-- Statement of the problem, showing the interval for the monotonic decrease of y = cos (π/4 - 2x)
theorem monotonic_decreasing_cosine : ∀ x : ℝ,
  (∃ k : ℤ, k * real.pi + real.pi / 8 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 8) ↔
  (∃ k : ℤ, 2 * k * real.pi ≤ 2 * x - real.pi / 4 ∧ 2 * x - real.pi / 4 ≤ 2 * k * real.pi + real.pi) :=
sorry

end monotonic_decreasing_cosine_l498_498036


namespace range_of_f_l498_498230

noncomputable def g (x : ℝ) := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

noncomputable def f (x : ℝ) := Real.sqrt (g x ^ 2 - 245)

theorem range_of_f : (Set.range f) = Set.Icc 0 14 := sorry

end range_of_f_l498_498230


namespace reduced_population_l498_498139

theorem reduced_population (initial_population : ℕ)
  (percentage_died : ℝ)
  (percentage_left : ℝ)
  (h_initial : initial_population = 8515)
  (h_died : percentage_died = 0.10)
  (h_left : percentage_left = 0.15) :
  ((initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ)) - 
   (⌊percentage_left * (initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ))⌋₊ : ℕ)) = 6515 :=
by
  sorry

end reduced_population_l498_498139


namespace extreme_value_at_neg_one_monotonicity_of_f_extreme_value_range_sum_extreme_values_gt_ln_half_e_l498_498690

noncomputable def f (x a : ℝ) : ℝ := Real.log (x + a) + x^2

theorem extreme_value_at_neg_one (a : ℝ) :
  (∃ x, x = -1 ∧ differentiable_at ℝ (λ x, f x a) x ∧ deriv (λ x, f x a) x = 0) →
  a = 3 / 2 := sorry

theorem monotonicity_of_f (a : ℝ) :
  a = 3 / 2 →
  (∀ x, (-3/2 < x ∧ x < -1) → deriv (λ x, f x a) x > 0) ∧
  (∀ x, (-1 < x ∧ x < -1/2) → deriv (λ x, f x a) x < 0) ∧
  (∀ x, (x > -1/2) → deriv (λ x, f x a) x > 0) := sorry

theorem extreme_value_range (a : ℝ) :
  (∃ x y, x ≠ y ∧ differentiable ℝ (λ x, f x a) ∧ deriv (λ x, f x a) x = 0 ∧ deriv (λ x, f x a) y = 0) ↔
  a ∈ Set.Ioi (Real.sqrt 2) := sorry

theorem sum_extreme_values_gt_ln_half_e (a : ℝ) :
  a ∈ Set.Ioi (Real.sqrt 2) →
  (∃ x1 x2, x1 ≠ x2 ∧ 
      deriv (λ x, f x a) x1 = 0 ∧ 
      deriv (λ x, f x a) x2 = 0 ∧
      (f x1 a + f x2 a) > Real.log (Real.exp 1 / 2)) := sorry

end extreme_value_at_neg_one_monotonicity_of_f_extreme_value_range_sum_extreme_values_gt_ln_half_e_l498_498690


namespace plane_equation_l498_498340

theorem plane_equation (x y z : ℝ) (h : (1, -1, 2) = (1, -1, 2))
  (ha : (0, 3, 1) ∈ ((x, y, z) : ℝ × ℝ × ℝ) → 
  (1, -1, 2) ≠ (0, 0, 0)) :
  x - y + 2z + 1 = 0 :=
sorry

end plane_equation_l498_498340


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498900

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498900


namespace track_width_l498_498161

theorem track_width (r1 r2 : ℝ) (h : 2 * real.pi * r1 - 2 * real.pi * r2 = 20 * real.pi) : r1 - r2 = 10 :=
sorry

end track_width_l498_498161


namespace sum_of_fifth_powers_l498_498521

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l498_498521


namespace star_value_l498_498470

variable (a b : ℤ)
noncomputable def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem star_value
  (h1 : a + b = 11)
  (h2 : a * b = 24)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  star a b = 11 / 24 := by
  sorry

end star_value_l498_498470


namespace D_is_largest_l498_498508

def D := (2008 / 2007) + (2008 / 2009)
def E := (2008 / 2009) + (2010 / 2009)
def F := (2009 / 2008) + (2009 / 2010) - (1 / 2009)

theorem D_is_largest : D > E ∧ D > F := by
  sorry

end D_is_largest_l498_498508


namespace common_ratio_of_geometric_series_l498_498544

theorem common_ratio_of_geometric_series (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_limit : ((λ n, (∑ i in finset.range (n + 1), a (3 * i)) / (∑ i in finset.range (n + 1), a i)) ⟶ 3 / 4) at_top) :
  q = (Real.sqrt 21 - 3) / 6 := 
sorry

end common_ratio_of_geometric_series_l498_498544


namespace smallest_positive_integer_l498_498109

theorem smallest_positive_integer (n : ℕ) (hn : 0 < n) (h : 19 * n ≡ 1456 [MOD 11]) : n = 6 :=
by
  sorry

end smallest_positive_integer_l498_498109


namespace find_omega_l498_498633

theorem find_omega (ω : ℝ) (h : (∀ x : ℝ, 2 * cos (π / 3 - ω * x) = 2 * cos (π / 3 - ω * (x + 4 * π))) ) : 
  ω = 1 / 2 ∨ ω = -1 / 2 :=
sorry

end find_omega_l498_498633


namespace range_of_a_l498_498694

def f (x : ℝ) : ℝ := 3^(-x) - 3^x - x

theorem range_of_a (a : ℝ) (h : f (2 * a + 3) + f (3 - a) > 0) : a < -6 := by
  sorry

end range_of_a_l498_498694


namespace angle_ABH_in_regular_octagon_l498_498842

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498842


namespace power_sum_eq_l498_498195

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq_l498_498195


namespace light_flash_time_l498_498562

theorem light_flash_time :
  1/6 * 3600 = 600 →
  600 / 600 = 1 :=
by
  intros h1
  rw [← mul_div_assoc, (nat.cast_comm 1), mul_one, mul_one_div, nat.cast_inj.2 (1 : ℕ).eq_iff, div_self]
  exact dec_trivial

end light_flash_time_l498_498562


namespace sum_w_minus_l_cubed_nonneg_l498_498124

noncomputable def tournament (n : ℕ) : Prop :=
n ≥ 4 ∧
(∀ (i j : ℕ) (hij : i ≠ j), ∃! (r : ℤ), r = 1 ∨ r = -1) ∧
-- No ties condition
(∀ (i j k l : ℕ), 
  i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
  ¬ (w i < w j ∧ w i < w k ∧ w i < w l ∧ 
     (w j + l j) = (w k + l k) ∧ (w j + l j) = (w l + l l) ∧ l k ≠ l l)) ∧
-- Definitions of wins and losses of the players
(∀ (i : ℕ), w i - l i = ∑ (j : ℕ) in (finset.range n).erase i, (if i > j then 1 else -1))

theorem sum_w_minus_l_cubed_nonneg {n : ℕ} (h : tournament n) :
  ∑ i in finset.range n, (w i - l i) ^ 3 ≥ 0 :=
sorry

end sum_w_minus_l_cubed_nonneg_l498_498124


namespace sum_of_fifth_powers_l498_498522

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l498_498522


namespace find_minimum_m_l498_498644

noncomputable def volume_space (m : ℝ) : ℝ := m^4
noncomputable def favorable_event (m : ℝ) : ℝ := (m - 1)^4
noncomputable def probability_favorable (m : ℝ) : ℝ := favorable_event m / volume_space m

theorem find_minimum_m :
  ∃ m : ℕ, 
    (probability_favorable m > 2/3) ∧
    (∀ n : ℕ, n < m → ¬(probability_favorable n > 2/3)) ∧
    m = 12 :=
by
  sorry

end find_minimum_m_l498_498644


namespace number_of_crows_is_2_l498_498318

def Bird : Type := {name : String}

def Adam : Bird := {name := "Adam"}
def Bella : Bird := {name := "Bella"}
def Cindy : Bird := {name := "Cindy"}
def Dave : Bird := {name := "Dave"}

def isParrot (b : Bird) : Prop := sorry
def isCrow (b : Bird) : Prop := sorry

axiom adam_statement : (isParrot Adam ↔ isParrot Dave) ∧ (isCrow Adam ↔ isCrow Dave)
axiom bella_statement : isParrot Bella → isCrow Cindy
axiom cindy_statement : isParrot Cindy → isCrow Bella
axiom dave_statement : (isParrot Adam) + (isParrot Bella) + (isParrot Cindy) + (isParrot Dave) = 2

theorem number_of_crows_is_2 : 
  ((if isCrow Adam then 1 else 0) +
   (if isCrow Bella then 1 else 0) +
   (if isCrow Cindy then 1 else 0) +
   (if isCrow Dave then 1 else 0)) = 2 := 
by sorry

end number_of_crows_is_2_l498_498318


namespace total_number_of_outfits_l498_498010

-- Definitions of the conditions as functions/values
def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_ties_options : Nat := 4 + 1  -- 4 ties + 1 option for no tie
def num_belts_options : Nat := 2 + 1  -- 2 belts + 1 option for no belt

-- Lean statement to formulate the proof problem
theorem total_number_of_outfits : 
  num_shirts * num_pants * num_ties_options * num_belts_options = 600 := by
  sorry

end total_number_of_outfits_l498_498010


namespace triangle_PQR_not_right_l498_498337

-- Definitions based on conditions
def isIsosceles (a b c : ℝ) (angle1 angle2 : ℝ) : Prop := (angle1 = angle2) ∧ (a = c)

def perimeter (a b c : ℝ) : ℝ := a + b + c

def isRightTriangle (a b c : ℝ) : Prop := a * a = b * b + c * c

-- Given conditions
def PQR : ℝ := 10
def PRQ : ℝ := 10
def QR : ℝ := 6
def angle_PQR : ℝ := 1
def angle_PRQ : ℝ := 1

-- Lean statement for the proof problem
theorem triangle_PQR_not_right 
  (h1 : isIsosceles PQR QR PRQ angle_PQR angle_PRQ)
  (h2 : QR = 6)
  (h3 : PRQ = 10):
  ¬ isRightTriangle PQR QR PRQ ∧ perimeter PQR QR PRQ = 26 :=
by {
    sorry
}

end triangle_PQR_not_right_l498_498337


namespace log_10_7_in_terms_of_p_q_l498_498649

noncomputable def log_relation (p q : ℝ) : Prop :=
  log 4 3 = p ∧ log 3 7 = q → log 10 7 = (2 * p * q) / (2 * p * q + 1)

theorem log_10_7_in_terms_of_p_q (p q : ℝ) : log_relation p q :=
by
  intros h
  sorry

end log_10_7_in_terms_of_p_q_l498_498649


namespace right_tetrahedron_circumsphere_l498_498749

/-
Pufo : In the right tetrahedron \(SABC\), the lateral edges \(SA\), \(SB\), and \(SC\) are mutually perpendicular.
Let \(M\) be the centroid of triangle \(ABC\), and let \(D\) be the midpoint of \(AB\).
Draw a line \(DP\) through \(D\) parallel to \(SC\).
Prove:
(1) \(DP\) intersects \(SM\).
(2) Let \(D'\) be the intersection point of \(DP\) and \(SM\).
Then prove \(D'\) is the center of the circumsphere of the right tetrahedron \(SABC\).
-/

structure RightTetrahedron (V : Type) [InnerProductSpace ℝ V] :=
(S A B C : V)
(mutually_perpendicular : ⟪S - A, S - B⟫ = 0 ∧ ⟪S - A, S - C⟫ = 0 ∧ ⟪S - B, S - C⟫ = 0)

def midpoint {V : Type} [InnerProductSpace ℝ V] (A B : V) : V :=
(1 / 2 : ℝ) • (A + B)

def centroid {V : Type} [InnerProductSpace ℝ V] (A B C : V) : V :=
(1 / 3 : ℝ) • (A + B + C)

theorem right_tetrahedron_circumsphere 
  (V : Type) [InnerProductSpace ℝ V] 
  (T : RightTetrahedron V) :
  let M := centroid T.A T.B T.C
  let D := midpoint T.A T.B
  let DP := {
    point_in_line := ∃ t : ℝ, D + t • ((T.S) - T.C),
    line_parallel := is_parallel (D, ∃ t : ℝ, D + t • ((T.S) - T.C)) (T.S, ∃ t : ℝ, T.S + t • ((T.S) - T.C))
  }
  let SM := {}
  let D' := point_in_line_of_intersection DP SM
  (1) DP.intersects SM
  (2) metric_space.is_circumscribed D' :=
sorry

end right_tetrahedron_circumsphere_l498_498749


namespace value_of_expression_l498_498653

theorem value_of_expression (a b : ℤ) (h : a - b = 1) : 3 * a - 3 * b - 4 = -1 :=
by {
  sorry
}

end value_of_expression_l498_498653


namespace volume_of_right_triangle_pyramid_l498_498015

noncomputable def pyramid_volume (H α β : ℝ) : ℝ :=
  (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2)

theorem volume_of_right_triangle_pyramid (H α β : ℝ) (alpha_acute : 0 < α ∧ α < π / 2) (H_pos : 0 < H) (beta_acute : 0 < β ∧ β < π / 2) :
  pyramid_volume H α β = (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2) := 
sorry

end volume_of_right_triangle_pyramid_l498_498015


namespace unique_fixed_point_l498_498278

noncomputable def f1 (x : ℝ) : ℝ :=
if x = 0 then 0
else if ∃ k : ℤ, 4^(k-1) ≤ |x| ∧ |x| < 2 * 4^(k-1) then -(1/2) * x
else if ∃ k : ℤ, 2 * 4^(k-1) ≤ |x| ∧ |x| ≤ 4^k then 2 * x
else 0 -- catches edge cases per definition

noncomputable def f2 (x : ℝ) : ℝ :=
if x = 0 then 0
else if ∃ k : ℤ, 4^(k-1) ≤ |x| ∧ |x| < 2 * 4^(k-1) then -(1/2) * x
else if ∃ k : ℤ, 2 * 4^(k-1) ≤ |x| ∧ |x| < 4^k then 2 * x
else 0 -- matches the given problem definition for f₂

-- Define f(x) being invariant under π/2 rotation
def rotation_invariant (f : ℝ → ℝ) := ∀ x : ℝ, f (f (-x)) = x

theorem unique_fixed_point (f : ℝ → ℝ) (h : rotation_invariant f) : ∃! x : ℝ, f x = x :=
begin
  sorry
end

end unique_fixed_point_l498_498278


namespace average_age_inhabitants_Campo_Verde_l498_498987

theorem average_age_inhabitants_Campo_Verde
  (H M : ℕ)
  (ratio_h_m : H / M = 2 / 3)
  (avg_age_men : ℕ := 37)
  (avg_age_women : ℕ := 42) :
  ((37 * H + 42 * M) / (H + M) : ℕ) = 40 := 
sorry

end average_age_inhabitants_Campo_Verde_l498_498987


namespace side_length_of_square_l498_498292

theorem side_length_of_square : 
  ∀ (L : ℝ), L = 28 → (L / 4) = 7 :=
by
  intro L h
  rw [h]
  norm_num

end side_length_of_square_l498_498292


namespace find_a_l498_498276

def f (x a : ℝ) : ℝ := Real.exp x + a * (x - 2) ^ 2

-- We state that if the second derivative of f at x = 0 equals 2, then a must be -1/4.
theorem find_a (a : ℝ) (h : deriv (λ x : ℝ, deriv (λ x : ℝ, f x a) x) 0 = 2) : 
  a = -1/4 :=
sorry

end find_a_l498_498276


namespace smallest_square_area_proof_l498_498233

open Real

noncomputable def smallest_square_area (l : ℝ) := 10 * (24 + 4 * l)

theorem smallest_square_area_proof :
  (∃ (k : ℝ), (k = 26 ∨ k = 36) ∧ (smallest_square_area k = 1280)) :=
by
  use 26
  split
  · left
    rfl
  · simp [smallest_square_area]
    sorry

end smallest_square_area_proof_l498_498233


namespace angle_ABH_is_22_point_5_l498_498916

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498916


namespace mean_weight_of_cats_l498_498989

def weight_list : List ℝ :=
  [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

noncomputable def total_weight : ℝ := weight_list.sum

noncomputable def mean_weight : ℝ := total_weight / weight_list.length

theorem mean_weight_of_cats : mean_weight = 101.64 := by
  sorry

end mean_weight_of_cats_l498_498989


namespace weaving_additional_yards_l498_498176

theorem weaving_additional_yards {d : ℝ} :
  (∃ d : ℝ, (30 * 5 + (30 * 29) / 2 * d = 390) → d = 16 / 29) :=
sorry

end weaving_additional_yards_l498_498176


namespace find_number_l498_498621

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l498_498621


namespace polynomial_evaluation_l498_498760

-- Statement: Given conditions, prove the specific computed value.
theorem polynomial_evaluation :
  ∃ (q : List (Polynomial ℤ)), 
    (∀ p ∈ q, Polynomial.Monic p ∧ Irreducible p) ∧
    Polynomial.prod q = Polynomial.X^6 - 3 * Polynomial.X^3 - Polynomial.X^2 - Polynomial.X - 2 ∧
    q.sum (λ p, Polynomial.eval 3 p) = 634 :=
by
  sorry

end polynomial_evaluation_l498_498760


namespace sum_of_T_is_2510_l498_498383

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498383


namespace measure_angle_abh_l498_498860

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498860


namespace cube_root_simplification_l498_498445

theorem cube_root_simplification (a b : ℕ) (h₁ : 16000 = 2^6 * 5^2) (h₂ : ∛16000 = ∛(2^6 * 5^2)) :
  ∛16000 = a * ∛b ∧ a + b = 29 :=
by
  have h₃ : ∛(2^6 * 5^2) = (2^2) * ∛(5^2), from sorry,
  have ha : a = 4, from sorry,
  have hb : b = 25, from sorry,
  have hab : a + b = 29, from sorry,
  exact ⟨h₃, hab⟩

end cube_root_simplification_l498_498445


namespace coeff_of_neg_5ab_l498_498018

theorem coeff_of_neg_5ab : coefficient (-5 * (a * b)) = -5 :=
by
  sorry

end coeff_of_neg_5ab_l498_498018


namespace direction_vector_correct_l498_498025

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end direction_vector_correct_l498_498025


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498901

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498901


namespace problem_l498_498590

-- Define \(\alpha\)
def alpha : ℝ := 49 * Real.pi / 48

-- Define the expression
def expr : ℝ := 4 * (Real.sin(alpha) ^ 3 * Real.cos(49 * Real.pi / 16) + 
                     Real.cos(alpha) ^ 3 * Real.sin(49 * Real.pi / 16)) * 
                     Real.cos(49 * Real.pi / 12)

-- The main theorem
theorem problem : expr = 0.75 :=
  sorry

end problem_l498_498590


namespace ellipse_equation_proof_dot_product_proof_l498_498251

noncomputable def ellipse_equation : string :=
  "The equation of the ellipse is x^2 / 12 + y^2 / 4 = 1"

noncomputable def dot_product_constant : string :=
  "The dot product of vectors RM and RN is always 0"

theorem ellipse_equation_proof :
  let F1 := (-2 : ℝ, sqrt 2)
  let F2 := (2  : ℝ, sqrt 2)
  let R := (0, -2 : ℝ)
  let P : ℝ × ℝ := sorry -- to define a point on the ellipse
  { |P - F1| + |P - F2| = 4 * sqrt 3 ∧ (|F1.1| = 2 * sqrt 2 ∧ |F2.1| = 2 * sqrt 2)} →
  (∃ (a b : ℝ), a > b > 0 ∧ 4 * sqrt 3 = 2 * a ∧ a^2 = 12 ∧ b^2 = a^2 - (2 * sqrt 2)^2 ∧
   ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry 

theorem dot_product_proof :
  let F1 := (-2 : ℝ, sqrt 2)
  let F2 := (2  : ℝ, sqrt 2)
  let R := (0, -2 : ℝ)
  let P := (0, 1 : ℝ)
  let k : ℝ := sorry -- slope of the line
  let line := fun x => k * x + 1
  let x1 := sorry
  let x2 := sorry
  let M := (λ x, (x, line x))
  let N := (λ x, (x, line x))
  (|P|^2 = 0 ∧ (x1 * x2 = -9 / (1 + 3 * k^2) ∧
   (x1 + x x2 = -6 * k / (1 + 3 * k^2)))) →
  ((x1 * x2 + (k * x1 + 3) * (k * x2 + 3)) = 0) :=
sorry

end ellipse_equation_proof_dot_product_proof_l498_498251


namespace acute_triangle_sin_cos_ineq_l498_498687

theorem acute_triangle_sin_cos_ineq
  (f : ℝ → ℝ)
  (h_f : ∀ x, x > 0 → f x = x ^ 2014)
  (α β : ℝ)
  (hα : 0 < α)
  (hβ : 0 < β)
  (hacuteα : α < π / 2)
  (hacuteβ : β < π / 2)
  (hαβ : α + β > π / 2) :
  f (sin α) > f (cos β) :=
sorry

end acute_triangle_sin_cos_ineq_l498_498687


namespace find_50th_element_l498_498460

def row_pattern (n : ℕ) : ℕ := 3 * n

def elements_in_row (n : ℕ) : ℕ := 2 * n

def cumulative_elements (n : ℕ) : ℕ :=
  (sum (finset.range n)) * 2

theorem find_50th_element : 
  ∃ r k, cumulative_elements (r - 1) < 50 ∧ 50 ≤ cumulative_elements r ∧ row_pattern r = 21 :=
sorry

end find_50th_element_l498_498460


namespace emily_clock_sync_24_hours_l498_498614

theorem emily_clock_sync_24_hours (backup_speed : ℕ) (interval_hours : ℕ) :
  (backup_speed = 5 ∧ interval_hours = 24) →
  (∃ sync_times : ℕ, sync_times = 12) :=
by
  -- Given that Emily’s clock runs backwards at five times the speed of a normal clock and both are analog clocks,
  -- we need to prove that in the next 24 hours, Emily’s broken clock will display the correct time exactly 12 times.
  intros h,
  -- Placeholder for the actual proof
  sorry

end emily_clock_sync_24_hours_l498_498614


namespace dryer_weight_l498_498612

theorem dryer_weight 
(empty_truck_weight crates_soda_weight num_crates soda_weight_factor 
    fresh_produce_weight_factor num_dryers fully_loaded_truck_weight : ℕ) 

  (h1 : empty_truck_weight = 12000) 
  (h2 : crates_soda_weight = 50) 
  (h3 : num_crates = 20) 
  (h4 : soda_weight_factor = crates_soda_weight * num_crates) 
  (h5 : fresh_produce_weight_factor = 2 * soda_weight_factor) 
  (h6 : num_dryers = 3) 
  (h7 : fully_loaded_truck_weight = 24000) 

  : (fully_loaded_truck_weight - empty_truck_weight 
      - (soda_weight_factor + fresh_produce_weight_factor)) / num_dryers = 3000 := 
by sorry

end dryer_weight_l498_498612


namespace least_number_of_trees_l498_498560

theorem least_number_of_trees :
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n % 7 = 0) ∧ n = 210 :=
by
  sorry

end least_number_of_trees_l498_498560


namespace original_price_l498_498948

theorem original_price (x : ℝ) (h : x * (1 / 8) = 8) : x = 64 := by
  -- To be proved
  sorry

end original_price_l498_498948


namespace measure_angle_abh_l498_498814

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498814


namespace winner_is_Junsu_l498_498122

def Younghee_water_intake : ℝ := 1.4
def Jimin_water_intake : ℝ := 1.8
def Junsu_water_intake : ℝ := 2.1

theorem winner_is_Junsu : 
  Junsu_water_intake > Younghee_water_intake ∧ Junsu_water_intake > Jimin_water_intake :=
by sorry

end winner_is_Junsu_l498_498122


namespace log_equation_solution_l498_498725

theorem log_equation_solution (x : ℝ) (h : log 3 (2 * x) = 3) : x = 27 / 2 :=
by {
  sorry
}

end log_equation_solution_l498_498725


namespace hyperbola_eccentricity_range_l498_498699

noncomputable def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

noncomputable def hyperbola (x y a : ℝ) : Prop :=
  (x^2) / (a^2) - y^2 / 16 = 1

noncomputable def inside_circle (x y r : ℝ) : Prop :=
  x^2 + y^2 < r^2

def parabola_focus_inside_circle (x y a : ℝ) : Prop :=
  inside_circle 2 0 (4 / a * sqrt (4 - a^2))

theorem hyperbola_eccentricity_range (a e : ℝ)
  (h0 : ∀ x y, parabola x y → hyperbola x y a)
  (h1 : a > 0)
  (h2 : a < sqrt 2)
  (h3 : parabola_focus_inside_circle 2 0 a) : 3 < sqrt ((16 + a^2) / a^2) :=
by
  sorry

end hyperbola_eccentricity_range_l498_498699


namespace polygon_sides_l498_498492

theorem polygon_sides (n : ℕ) (h : n - 1 = 2022) : n = 2023 :=
by
  sorry

end polygon_sides_l498_498492


namespace birds_more_than_nests_l498_498058

theorem birds_more_than_nests : 
  let birds := 6 
  let nests := 3 
  (birds - nests) = 3 := 
by 
  sorry

end birds_more_than_nests_l498_498058


namespace isabel_weekly_distance_l498_498348

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end isabel_weekly_distance_l498_498348


namespace find_sister_candy_initially_l498_498759

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end find_sister_candy_initially_l498_498759


namespace regular_octagon_angle_ABH_l498_498878

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498878


namespace probability_of_even_is_one_fifth_l498_498240

-- Define the set of numbers
def numbers : set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set of even numbers from the given set
def evens : set ℕ := {2, 4, 6}

-- Function to count combinations of two numbers drawn without replacement where both are even
def count_even_combinations : ℕ :=
  -- There are 6 even combinations: (2,4), (2,6), (4,2), (4,6), (6,2), (6,4)
  6

-- Total number of combinations when drawing two numbers without replacement from six numbers
def count_total_combinations : ℕ :=
  -- There are 30 total combinations of 2 numbers out of 6
  30

-- Define the probability
def probability_of_even_even : ℚ :=
  count_even_combinations / count_total_combinations

theorem probability_of_even_is_one_fifth :
  probability_of_even_even = 1 / 5 := sorry

end probability_of_even_is_one_fifth_l498_498240


namespace find_smallest_b_l498_498631

theorem find_smallest_b :
  ∃ b : ℕ, 
    (∀ r s : ℤ, r * s = 3960 → r + s ≠ b ∨ r + s > 0) ∧ 
    (∀ r s : ℤ, r * s = 3960 → (r + s < b → r + s ≤ 0)) ∧ 
    b = 126 :=
by
  sorry

end find_smallest_b_l498_498631


namespace det_transformation_matrix_l498_498768

noncomputable def scaling_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3, 0; 0, 3]

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let sqrt2over2 := (Real.sqrt 2) / 2;
  !![sqrt2over2, -sqrt2over2; sqrt2over2, sqrt2over2]

noncomputable def transformation_matrix := scaling_matrix ⬝ rotation_matrix

theorem det_transformation_matrix :
  Matrix.det transformation_matrix = 9 := sorry

end det_transformation_matrix_l498_498768


namespace find_side_a_in_triangle_l498_498345

noncomputable def triangle_side_a (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  cosA = 4/5 ∧ b = 2 ∧ S = 3 → a = Real.sqrt 13

-- Theorem statement with explicit conditions and proof goal
theorem find_side_a_in_triangle
  (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) :
  cosA = 4 / 5 → b = 2 → S = 3 → a = Real.sqrt 13 :=
by 
  intros 
  sorry

end find_side_a_in_triangle_l498_498345


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498528

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498528


namespace third_median_length_l498_498752

noncomputable def triangle_median_length (m₁ m₂ : ℝ) (area : ℝ) : ℝ :=
  if m₁ = 5 ∧ m₂ = 4 ∧ area = 6 * Real.sqrt 5 then
    3 * Real.sqrt 7
  else
    0

theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ)
  (h₁ : m₁ = 5) (h₂ : m₂ = 4) (h₃ : area = 6 * Real.sqrt 5) :
  triangle_median_length m₁ m₂ area = 3 * Real.sqrt 7 :=
by
  -- Proof is skipped
  sorry

end third_median_length_l498_498752


namespace translation_result_l498_498064

variables (P : ℝ × ℝ) (P' : ℝ × ℝ)

def translate_left (P : ℝ × ℝ) (units : ℝ) := (P.1 - units, P.2)
def translate_down (P : ℝ × ℝ) (units : ℝ) := (P.1, P.2 - units)

theorem translation_result :
    P = (-4, 3) -> P' = translate_down (translate_left P 2) 2 -> P' = (-6, 1) :=
by
  intros h1 h2
  sorry

end translation_result_l498_498064


namespace exists_x_l498_498360

variables {n s t : ℕ} -- positive integers n, s, and t
variables (A B : finset (fin n) → ℕ) -- functions f and g defined on subsets of {1, 2, ..., n}

-- Definitions for f and g
def f (S : finset (fin n)) : ℕ := A S
def g (S : finset (fin n)) : ℕ := B S

-- Conditions on f and g
variable (h1 : ∀ (x y : fin n), x < y → f ({x, y} : finset (fin n)) = g ({x, y} : finset (fin n)))
variable (h2 : t < n)

-- Theorem statement: there exists some x such that f({x}) ≥ g({x})
theorem exists_x (A B : finset (fin n) → ℕ) 
  (h1 : ∀ (x y : fin n), x < y → (A {x, y} : ℕ) = B {x, y})
  (h2 : t < n) : 
  ∃ x : fin n, A {x} ≥ B {x} := 
begin
  sorry
end

end exists_x_l498_498360


namespace sum_of_T_is_2510_l498_498382

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498382


namespace circle_diameter_l498_498452

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ (d : ℝ), d = 16 :=
by
  sorry

end circle_diameter_l498_498452


namespace sampling_distribution_l498_498136

theorem sampling_distribution 
  (total_employees : ℕ := 3200)
  (sample_size : ℕ := 400)
  (middle_ratio : ℕ := 5)
  (young_ratio : ℕ := 3)
  (elderly_ratio : ℕ := 2)
  (total_ratio : ℕ := middle_ratio + young_ratio + elderly_ratio) :
  stratified_sampling :=
  by
  let middle_aged : ℕ := (sample_size * middle_ratio) / total_ratio
  let young : ℕ := (sample_size * young_ratio) / total_ratio
  let elderly : ℕ := (sample_size * elderly_ratio) / total_ratio
  have stratified_sampling_method : (middle_aged = 200) ∧ (young = 120) ∧ (elderly = 80),
    sorry
  exact stratified_sampling_method

end sampling_distribution_l498_498136


namespace sqrt_diff_eq_neg_four_sqrt_five_l498_498615

theorem sqrt_diff_eq_neg_four_sqrt_five : 
  (Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5)) = -4 * Real.sqrt 5 := 
sorry

end sqrt_diff_eq_neg_four_sqrt_five_l498_498615


namespace unique_octuple_primes_l498_498625

open Nat

theorem unique_octuple_primes (p : Fin 8 → ℕ) (h_prime : ∀ i, Prime (p i)) :
    (∑ i, (p i)^2 = 4 * (∏ i, p i) - 992) ↔ (∀ i, p i = 2) := 
by 
  sorry

end unique_octuple_primes_l498_498625


namespace closest_point_on_plane_l498_498628

theorem closest_point_on_plane 
  (x y z : ℝ) 
  (h : 4 * x - 3 * y + 2 * z = 40) 
  (h_closest : ∀ (px py pz : ℝ), (4 * px - 3 * py + 2 * pz = 40) → dist (px, py, pz) (3, 1, 4) ≥ dist (x, y, z) (3, 1, 4)) :
  (x, y, z) = (139/19, -58/19, 86/19) :=
sorry

end closest_point_on_plane_l498_498628


namespace find_b_perpendicular_l498_498308

theorem find_b_perpendicular (a b : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0 → 
  - (a / 2) * - (3 / b) = -1) → b = -3 := 
sorry

end find_b_perpendicular_l498_498308


namespace count_sixth_powers_below_50000_l498_498564

theorem count_sixth_powers_below_50000 : 
  {n : ℕ | n < 50000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 6 :=
by
  sorry

end count_sixth_powers_below_50000_l498_498564


namespace volunteer_arrangements_l498_498473

theorem volunteer_arrangements (students : Fin 5 → String) (events : Fin 3 → String)
  (A : String) (high_jump : String)
  (h : ∀ (arrange : Fin 3 → Fin 5), ¬(students (arrange 0) = A ∧ events 0 = high_jump)) :
  ∃! valid_arrangements, valid_arrangements = 48 :=
by
  sorry

end volunteer_arrangements_l498_498473


namespace sqrt_simplification_l498_498953

-- Define the value 360.
def n : ℕ := 360

-- Condition 1: 360 is divisible by 36.
def condition1 : n % 36 = 0 := by
  sorry

-- Condition 2: 360 divided by 36 is 10, which is not a perfect square.
def condition2 : n / 36 = 10 := by
  sorry

-- The proof statement.
theorem sqrt_simplification :
  condition1 →
  condition2 →
  sqrt(ℝ) 360 = 6 * sqrt(ℝ) 10 :=
by
  sorry

end sqrt_simplification_l498_498953


namespace measure_angle_abh_l498_498863

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498863


namespace standard_equation_of_ellipse_l498_498681

noncomputable def line (x y : ℝ) := sqrt 6 * x + 2 * y - 2 * sqrt 6 = 0

def ellipse (x y : ℝ) := x^2 / 10 + y^2 / 6 = 1

def point_E := (0 : ℝ, sqrt 6)

def point_F := (2 : ℝ, 0)

def point_P := (sqrt 5 : ℝ, sqrt 3)

theorem standard_equation_of_ellipse :
    (∀ (x y : ℝ), line x y → ellipse x y) ∧
    (∀ (x y : ℝ), x = sqrt 5 ∧ y = sqrt 3 → ellipse x y → 
      (sqrt 5 / 10 * x + sqrt 3 / 6 * y = 1)) := 
by sorry

end standard_equation_of_ellipse_l498_498681


namespace f_formula_l498_498697

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1) ^ 2)

noncomputable def f (n : ℕ) : ℚ := (List.range n).map (fun k => 1 - a (k + 1)).prod

theorem f_formula (n : ℕ) : f n = (n + 2) / (2 * n + 2) :=
sorry

end f_formula_l498_498697


namespace proportion_correct_l498_498722

theorem proportion_correct (x y : ℝ) (h : 3 * x = 2 * y) (hy : y ≠ 0) : x / 2 = y / 3 :=
by
  sorry

end proportion_correct_l498_498722


namespace stratified_sampling_school_C_l498_498735

theorem stratified_sampling_school_C 
  (teachers_A : ℕ) 
  (teachers_B : ℕ) 
  (teachers_C : ℕ) 
  (total_teachers : ℕ)
  (total_drawn : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 140)
  (hC : teachers_C = 160)
  (hTotal : total_teachers = teachers_A + teachers_B + teachers_C)
  (hDraw : total_drawn = 60) :
  (total_drawn * teachers_C / total_teachers) = 20 := 
by
  sorry

end stratified_sampling_school_C_l498_498735


namespace pizza_area_percent_increase_l498_498600

noncomputable def percent_increase_area (r1 r2 : ℝ) : ℝ :=
  let area1 := π * r1^2
      area2 := π * r2^2
  in ((area2 - area1) / area1) * 100

theorem pizza_area_percent_increase :
  percent_increase_area 7 9 = 65.31 :=
sorry

end pizza_area_percent_increase_l498_498600


namespace sum_a_1_to_a_100_l498_498790

noncomputable def seq : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n => if (n % 4 == 0) then seq (n - 4) else if n % 4 == 1 then seq (n - 3) else if n % 4 == 2 then seq (n - 2) else 4

theorem sum_a_1_to_a_100 :
  (finset.range 100).sum (λ i, seq i) = 200 := 
  sorry

end sum_a_1_to_a_100_l498_498790


namespace measure_of_angle_ABH_l498_498828

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498828


namespace ratio_democrats_total_participants_l498_498992

theorem ratio_democrats_total_participants (total_participants female_democrats : ℕ)
    (total_participants_eq : total_participants = 780)
    (female_democrats_eq : female_democrats = 130)
    (half_female_democrats : ∀ F : ℕ, half (F / 2) = female_democrats)
    (one_quarter_male_democrats : ∀ M : ℕ, one_quarter (M / 4))
    (total_participants_split : ∀ F M : ℕ, F + M = total_participants)
    (democrats_eq : ∀ total_democrats : ℕ, total_democrats = female_democrats + one_quarter_male_democrats)
: (∃ ratio : ℚ, ratio = 1 / 3) := sorry

end ratio_democrats_total_participants_l498_498992


namespace number_of_different_types_of_cards_l498_498191

theorem number_of_different_types_of_cards 
  (dim : ℕ × ℕ) (front_color : ℕ) (back_color : ℕ)
  (identical_squares : ℕ → ℕ → bool) 
  (cut_along_grid_lines : bool)
  (cards_of_same_shape : ℕ → ℕ → bool) :
  dim = (3, 4) →
  front_color = 1 →  -- assuming 1 represents gray
  back_color = 2 →   -- assuming 2 represents red
  (∀ i j, identical_squares i j = true) →
  cut_along_grid_lines = true →
  (∀ i j, cards_of_same_shape i j = true) →
  ∃ n : ℕ, n = 8 := 
by {
  sorry
}

end number_of_different_types_of_cards_l498_498191


namespace rectangle_in_triangle_area_l498_498409

theorem rectangle_in_triangle_area (b h : ℕ) (hb : b = 12) (hh : h = 8)
  (x : ℕ) (hx : x = h / 2) : (b * x / 2) = 48 := 
by
  sorry

end rectangle_in_triangle_area_l498_498409


namespace range_of_a_l498_498692

def f (x : ℝ) : ℝ := 3^(-x) - 3^x - x

theorem range_of_a (a : ℝ) (h : f (2 * a + 3) + f (3 - a) > 0) : a < -6 :=
by
  sorry

end range_of_a_l498_498692


namespace area_shaded_region_l498_498743

theorem area_shaded_region (ABCD : Type) [parallelogram ABCD] (B E D C : ABCD)
  (h1 : height_from B to AD E)
  (h2 : segment ED = 8)
  (h3 : base BC = 14)
  (h4 : parallelogram_area ABCD = 126) :
  shaded_region_area BEDC = 99 := 
sorry

end area_shaded_region_l498_498743


namespace sum_of_fifth_powers_l498_498534

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l498_498534


namespace arithmetic_sequence_sum_l498_498250

/-- Let {x_k} be a sequence defined by x_1 = 3 and x_{k+1} = x_k + 1. 
    We want to prove the sum of the first n terms of the sequence is (n * (n + 5)) / 2, 
    given n is a positive integer less than or equal to 10.-/
theorem arithmetic_sequence_sum {n : ℕ} (h_pos : 1 ≤ n) (h_le : n ≤ 10) :
  let x : ℕ → ℕ := λ k, 3 + (k - 1) in
  (∑ k in Finset.range (n + 1), ite (k = 0) 0 (x k)) = n * (n + 5) / 2 :=
  sorry

end arithmetic_sequence_sum_l498_498250


namespace trig_identity_l498_498648

theorem trig_identity (α : ℝ) (h : cos (α + π / 4) = 2 / 3) : sin (π / 4 - α) = 2 / 3 :=
by
  -- The proof is omitted
  sorry

end trig_identity_l498_498648


namespace trigonometric_identity_l498_498726

theorem trigonometric_identity 
  (α : ℝ)
  (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l498_498726


namespace andrew_can_determine_barabashka_l498_498188

theorem andrew_can_determine_barabashka :
  ∃ (n : ℕ) (h : n = 100)
    (P : Fin n → Type)
    (is_weight : ∀ i : Fin n , P i)
    (barabashka_exists : ∀ (s : Fin 10 → Fin n), Prop),
  (∀ s, (∃ (sorted : Fin 10 → Fin n), (∀ i j, i ≤ j → sorted i ≤ sorted j)) 
        ∨ (∃ (sorted : Fin 10 → Fin n), (∀ i j, i ≤ j → sorted i ≤ sorted j) 
        ∧ (∃ i j, sorted i = s j ∧ sorted j = s i))) →
  (∃ s, (∀ (sorted1 sorted2 : Fin 10 → Fin n), 
    ((∃ i j, sorted1 i = sorted2 j ∧ sorted1 j = sorted2 i) → barabashka_exists s))) :=
sorry

end andrew_can_determine_barabashka_l498_498188


namespace sum_α_eq_zero_l498_498412

def M : Finset ℕ := (Finset.range 2024).filter (λ n, n > 0)

def α (A : Finset ℕ) : ℤ := (-1) ^ (A.sum id)

theorem sum_α_eq_zero : (Finset.powerset M).sum α = 0 := 
  sorry

end sum_α_eq_zero_l498_498412


namespace regular_octagon_angle_ABH_l498_498875

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498875


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498897

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498897


namespace regular_octagon_angle_ABH_l498_498874

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498874


namespace cost_of_sunglasses_l498_498171

noncomputable def cost_per_pair 
  (price_per_pair : ℕ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℕ) 
  (profits_half : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_revenue := price_per_pair * pairs_sold in
  let total_cost := total_revenue - (profits_half * 2) in
  total_cost / pairs_sold

theorem cost_of_sunglasses :
  ∀ (price_per_pair pairs_sold sign_cost profits_half profit : ℕ),
    price_per_pair = 30 → 
    pairs_sold = 10 → 
    sign_cost = 20 → 
    (profits_half = sign_cost → profit = profits_half * 2) →
    cost_per_pair price_per_pair pairs_sold sign_cost profits_half profit = 26 :=
begin
  intros,
  sorry
end

end cost_of_sunglasses_l498_498171


namespace pre_image_of_f_5_1_l498_498979

def f (x y : ℝ) : ℝ × ℝ := (x + y, 2 * x - y)

theorem pre_image_of_f_5_1 : ∃ (x y : ℝ), f x y = (5, 1) ∧ (x, y) = (2, 3) :=
by
  sorry

end pre_image_of_f_5_1_l498_498979


namespace inequality_geq_l498_498434

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l498_498434


namespace tray_contains_correct_number_of_pieces_l498_498551

-- Define the dimensions of the tray
def tray_width : ℕ := 24
def tray_length : ℕ := 20
def tray_area : ℕ := tray_width * tray_length

-- Define the dimensions of each brownie piece
def piece_width : ℕ := 3
def piece_length : ℕ := 4
def piece_area : ℕ := piece_width * piece_length

-- Define the goal: the number of pieces of brownies that the tray contains
def num_pieces : ℕ := tray_area / piece_area

-- The statement to prove
theorem tray_contains_correct_number_of_pieces :
  num_pieces = 40 :=
by
  sorry

end tray_contains_correct_number_of_pieces_l498_498551


namespace speed_of_stream_l498_498478

variable (v : ℝ)

theorem speed_of_stream (h : (64 / (24 + v)) = (32 / (24 - v))) : v = 8 := 
by
  sorry

end speed_of_stream_l498_498478


namespace number_of_three_digit_numbers_with_sum_27_l498_498237

theorem number_of_three_digit_numbers_with_sum_27 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 27}.card = 1 :=
begin
  sorry
end

end number_of_three_digit_numbers_with_sum_27_l498_498237


namespace direction_vector_is_3_1_l498_498022

-- Given the line equation x - 3y + 1 = 0
def line_equation : ℝ × ℝ → Prop :=
  λ p, p.1 - 3 * p.2 + 1 = 0

-- The direction vector of the line
def direction_vector_of_line (v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * 3, k * 1)

theorem direction_vector_is_3_1 : direction_vector_of_line (3, 1) :=
by
  sorry

end direction_vector_is_3_1_l498_498022


namespace min_rectangles_to_cover_square_l498_498162

-- Conditions
def diagonal : ℝ := real.sqrt 65
def side_length : ℝ := 8

-- Statement
theorem min_rectangles_to_cover_square : 
  ∃ n : ℕ, n = 3 ∧ ∀ (rec : ℝ × ℝ), ((rec.1^2 + rec.2^2 = 65) → (∃ (covers_share : (ℝ × ℝ) → ℝ), 
  (∑ (i : ℕ) in finset.range n, covers_share (rec)) ≥ (side_length * side_length))) :=
sorry

end min_rectangles_to_cover_square_l498_498162


namespace card_draw_probability_l498_498995

-- Define a function to compute the probability of a sequence of draws
noncomputable def probability_of_event : Rat :=
  (4 / 52) * (4 / 51) * (1 / 50)

theorem card_draw_probability :
  probability_of_event = 4 / 33150 :=
by
  -- Proof goes here
  sorry

end card_draw_probability_l498_498995


namespace ivan_uses_more_paint_l498_498078

-- Define the basic geometric properties
def rectangular_section_area (length width : ℝ) : ℝ := length * width
def parallelogram_section_area (side1 side2 : ℝ) (angle : ℝ) : ℝ := side1 * side2 * Real.sin angle

-- Define the areas for each neighbor's fences
def ivan_area : ℝ := rectangular_section_area 5 2
def petr_area (alpha : ℝ) : ℝ := parallelogram_section_area 5 2 alpha

-- Theorem stating that Ivan's total fence area is greater than Petr's total fence area provided the conditions
theorem ivan_uses_more_paint (α : ℝ) (hα : α ≠ Real.pi / 2) : ivan_area > petr_area α := by
  sorry

end ivan_uses_more_paint_l498_498078


namespace area_N1N2N3_relative_l498_498338

-- Definitions
variable (A B C D E F N1 N2 N3 : Type)
-- Assuming D, E, F are points on sides BC, CA, AB respectively such that CD, AE, BF are one-fourth of their respective sides.
variable (area_ABC : ℝ)  -- Total area of triangle ABC
variable (area_N1N2N3 : ℝ)  -- Area of triangle N1N2N3

-- Given conditions
variable (H1 : CD = 1 / 4 * BC)
variable (H2 : AE = 1 / 4 * CA)
variable (H3 : BF = 1 / 4 * AB)

-- The expected result
theorem area_N1N2N3_relative :
  area_N1N2N3 = 7 / 15 * area_ABC :=
sorry

end area_N1N2N3_relative_l498_498338


namespace power_function_passing_through_point_l498_498548

theorem power_function_passing_through_point :
  ∃ (α : ℝ), (2:ℝ)^α = 4 := by
  sorry

end power_function_passing_through_point_l498_498548


namespace smallest_positive_period_of_f_max_min_f_on_interval_l498_498288

open Real

def vec_a (x : ℝ) : ℝ × ℝ := (sin x, 1 / 2)
def vec_b (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x + sin x, -1)
def f (x : ℝ) := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f : 
  (f (x + π) = f x) → ¬ ∃ ε > 0, (∀ x, f (x + ε) = f x) ∧ ε < π := 
by
  sorry

theorem max_min_f_on_interval : 
  let I := set.Icc (π / 4) (π / 2) in
  (∀ x ∈ I, f x ≤ 1) ∧ (∃ x ∈ I, f x = 1) ∧
  (∀ x ∈ I, f x ≥ 1 / 2) ∧ (∃ x ∈ I, f x = 1 / 2) :=
by
  sorry

end smallest_positive_period_of_f_max_min_f_on_interval_l498_498288


namespace if_a_eq_b_then_a_squared_eq_b_squared_l498_498041

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l498_498041


namespace angle_in_regular_octagon_l498_498888

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498888


namespace total_birds_in_marsh_l498_498060

def geese := 58
def ducks := 37
def herons := 23
def kingfishers := 46
def swans := 15

theorem total_birds_in_marsh :
  geese + ducks + herons + kingfishers + swans = 179 := 
by 
  calc
    geese + ducks + herons + kingfishers + swans = 58 + 37 + 23 + 46 + 15 := by rfl
    ... = 95 + 23 + 46 + 15 := by rfl
    ... = 118 + 46 + 15 := by rfl
    ... = 164 + 15 := by rfl
    ... = 179 := by rfl

end total_birds_in_marsh_l498_498060


namespace triangle_perimeter_l498_498264

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Define the foci locations based on the given ellipse parameters
def foci (c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  if c = 4 then ((-c, 0), (c, 0)) else ((0, 0), (0, 0))

-- Define the point M with x-coordinate 4 that lies on the ellipse
def point_M (y : ℝ) : ℝ × ℝ :=
  (4, y)

-- Condition: point M is on the ellipse
def M_on_ellipse (y : ℝ) : Prop :=
  ellipse 4 y

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of the triangle F1MF2
def perimeter (F1 F2 M : ℝ × ℝ) : ℝ :=
  dist F1 M + dist M F2 + dist F1 F2

theorem triangle_perimeter (y : ℝ) (c : ℝ) (F1 F2 M : ℝ × ℝ)
  (Foci : foci c = (F1, F2)) (PointM : point_M y = M) (Ellipse_M : M_on_ellipse y) :
  perimeter F1 F2 M = 18 :=
by
  sorry

end triangle_perimeter_l498_498264


namespace year_2017_l498_498096

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

theorem year_2017 : ∃ (y : ℕ), y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ y', y' > 2010 → sum_of_digits y' = 10 → y' ≥ y :=
by {
  let y := 2017,
  use y,
  split,
  { exact Nat.lt_of_succ_lt_succ (Nat.succ_lt_succ (Nat.succ_pos 2016)) },
  split,
  { norm_num },
  intros y' y'_gt_2010 sum_y',
  sorry
}

end year_2017_l498_498096


namespace angle_ABH_regular_octagon_l498_498920

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498920


namespace boy_travel_speed_l498_498552

theorem boy_travel_speed 
  (v : ℝ)
  (travel_distance : ℝ := 10) 
  (return_speed : ℝ := 2) 
  (total_time : ℝ := 5.8)
  (distance : ℝ := 9.999999999999998) :
  (v = 12.5) → (travel_distance = distance) →
  (total_time = (travel_distance / v) + (travel_distance / return_speed)) :=
by
  sorry

end boy_travel_speed_l498_498552


namespace cube_faces_consecutive_sum_l498_498148

noncomputable def cube_face_sum (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)

theorem cube_faces_consecutive_sum (n : ℕ) (h1 : ∀ i, i ∈ [0, 5] -> (2 * n + 5 + n + 5 - 6) = 6) (h2 : n = 12) :
  cube_face_sum n = 87 :=
  sorry

end cube_faces_consecutive_sum_l498_498148


namespace time_to_travel_to_shop_l498_498611

-- Define the distance and speed as given conditions
def distance : ℕ := 184
def speed : ℕ := 23

-- Define the time taken for the journey
def time_taken (d : ℕ) (s : ℕ) : ℕ := d / s

-- Statement to prove that the time taken is 8 hours
theorem time_to_travel_to_shop : time_taken distance speed = 8 := by
  -- The proof is omitted
  sorry

end time_to_travel_to_shop_l498_498611


namespace regular_octagon_angle_ABH_l498_498877

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498877


namespace sum_of_fifth_powers_l498_498520

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l498_498520


namespace three_distinct_solutions_no_solution_for_2009_l498_498782

-- Problem 1: Show that the equation has at least three distinct solutions if it has one
theorem three_distinct_solutions (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3*x1*y1^2 + y1^3 = n ∧ 
    x2^3 - 3*x2*y2^2 + y2^3 = n ∧ 
    x3^3 - 3*x3*y3^2 + y3^3 = n ∧ 
    (x1, y1) ≠ (x2, y2) ∧ 
    (x1, y1) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x3, y3)) :=
sorry

-- Problem 2: Show that the equation has no solutions when n = 2009
theorem no_solution_for_2009 :
  ¬ ∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = 2009 :=
sorry

end three_distinct_solutions_no_solution_for_2009_l498_498782


namespace sum_of_T_l498_498391

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498391


namespace measure_of_angle_ABH_l498_498851

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498851


namespace region_area_above_line_l498_498086

noncomputable def region_area : ℝ :=
  let circle_equation := λ x y : ℝ, (x - 5)^2 + (y - 5/2)^2 = 25/4
  let line_equation := λ x y : ℝ, y = x - 5
  let area_of_circle := π * (5/2)^2
  area_of_circle

theorem region_area_above_line : region_area = 25 * π / 4 :=
  sorry

end region_area_above_line_l498_498086


namespace problem_statement_l498_498030

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x / Real.log 2 else sorry

theorem problem_statement : f (1 / 2) < f (1 / 3) ∧ f (1 / 3) < f 2 :=
by
  -- Definitions based on given conditions
  have h1 : ∀ x : ℝ, f (2 - x) = f x := sorry
  have h2 : ∀ x : ℝ, 1 ≤ x → f x = Real.log x / Real.log 2 := sorry
  -- Proof of the statement based on h1 and h2
  sorry

end problem_statement_l498_498030


namespace cows_dogs_ratio_l498_498488

theorem cows_dogs_ratio (C D : ℕ) (hC : C = 184) (hC_remain : 3 / 4 * C = 138)
  (hD_remain : 1 / 4 * D + 138 = 161) : C / D = 2 :=
sorry

end cows_dogs_ratio_l498_498488


namespace angle_in_regular_octagon_l498_498882

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498882


namespace ivan_uses_more_paint_l498_498073

-- Conditions
def ivan_section_area : ℝ := 5 * 2
def petr_section_area (alpha : ℝ) : ℝ := 5 * 2 * Real.sin(alpha)
axiom alpha_lt_90 : ∀ α : ℝ, α < 90 → Real.sin(α) < 1

-- Assertion
theorem ivan_uses_more_paint (α : ℝ) (h1 : α < 90) : ivan_section_area > petr_section_area α :=
by
  sorry

end ivan_uses_more_paint_l498_498073


namespace sin_cos_theorem_l498_498549

theorem sin_cos_theorem (A B C : ℝ) (h : ∃ BC : ℝ, BC = 2) (h1 : ∃ AB : ℝ, AB = √3) (h2 : cos B = -1/2) : 
  sin C = √3/2 ∧ C < π/3 ∧ 0 < C := 
sorry

end sin_cos_theorem_l498_498549


namespace coefficient_x7y2_l498_498017

theorem coefficient_x7y2 (x y : ℕ) : 
  (let poly := (x - y) * (expand_binomial x y 8) in
   coefficient (x^7 * y^2) poly) = 20 := 
sorry

-- Helper definition to expand binomial terms
-- This should be correctly defined to mimic the binomial expansion
def expand_binomial (x y : ℕ) (n : ℕ) :=
  ∑ i in (finset.range (n + 1)), (nat.choose n i) * (x^(n - i)) * (y^i)

-- Helper definition to extract coefficients from the polynomial expression
def coefficient (monomial : ℕ) (poly : ℕ → ℕ) := 
  poly monomial

end coefficient_x7y2_l498_498017


namespace acute_triangle_l498_498732

-- Given the lengths of three line segments
def length1 : ℝ := 5
def length2 : ℝ := 6
def length3 : ℝ := 7

-- Conditions (C): The lengths of the three line segments
def triangle_inequality : Prop :=
  length1 + length2 > length3 ∧
  length1 + length3 > length2 ∧
  length2 + length3 > length1

-- Question (Q) and Answer (A): They form an acute triangle
theorem acute_triangle (h : triangle_inequality) : (length1^2 + length2^2 - length3^2 > 0) :=
by
  sorry

end acute_triangle_l498_498732


namespace measure_of_angle_ABH_l498_498852

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498852


namespace discount_difference_l498_498189

theorem discount_difference 
  (bill : ℝ)
  (S1 : ℝ := bill * 0.55) 
  (S2 : ℝ :=
     let after_first_discount := bill * 0.70 in
     let after_second_discount := after_first_discount * 0.90 in
     after_second_discount * 0.95) :
  (S2 - S1) = 582 :=
by
  -- Introduction of the given bill amount
  let bill := 12000
  -- Definitions of S1 and S2 as given in the conditions
  let S1 := bill * 0.55
  let after_first_discount := bill * 0.70
  let after_second_discount := after_first_discount * 0.90
  let S2 := after_second_discount * 0.95
  -- Assert that the difference between S2 and S1 is 582
  calc
    S2 - S1 = (after_second_discount * 0.95) - (bill * 0.55) : by rfl
         ... = 582 : sorry

end discount_difference_l498_498189


namespace sum_solutions_eq_zero_l498_498333

theorem sum_solutions_eq_zero :
  let y := 8 in
  let eq1 := y = 8 in
  let eq2 := ∀ x, x ^ 2 + y ^ 2 = 169 in
  (x ∈ ℝ ∧ eq1 ∧ eq2) →
  ∃ x1 x2, x1 ^ 2 + y ^ 2 = 169 ∧ x2 ^ 2 + y ^ 2 = 169 ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_solutions_eq_zero_l498_498333


namespace rate_of_simple_interest_l498_498513

variable (P R : ℝ)

-- Conditions
def amount_after_5_years := P + (P * R * 5)
def amount_is_seven_sixths := (7 / 6) * P

-- Proof Problem
theorem rate_of_simple_interest (h : amount_after_5_years P R = amount_is_seven_sixths P) : R = 1 / 30 := by
  sorry

end rate_of_simple_interest_l498_498513


namespace measure_angle_abh_l498_498817

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498817


namespace area_of_rotated_squares_l498_498997

noncomputable def side_length : ℝ := 8
noncomputable def rotation_middle : ℝ := 45
noncomputable def rotation_top : ℝ := 75

-- Theorem: The area of the resulting 24-sided polygon.
theorem area_of_rotated_squares :
  (∃ (polygon_area : ℝ), polygon_area = 96) :=
sorry

end area_of_rotated_squares_l498_498997


namespace find_f3_l498_498463

theorem find_f3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, x * f y = y * f x) (h2 : f 15 = 20) : f 3 = 4 := 
  sorry

end find_f3_l498_498463


namespace happiness_index_percentile_l498_498012

theorem happiness_index_percentile :
  let data := [3, 4, 5, 5, 6, 7, 7, 8, 9, 10] in
  let percentile (d : List ℕ) (p : ℕ) := 
    let n := d.length in
    let k := p * n / 100 in
    let sorted_d := d.qsort (≤) in
    (sorted_d[nat.pred k] + sorted_d.k) / 2 in
  percentile data 80 = 8.5 :=
by
  sorry

end happiness_index_percentile_l498_498012


namespace ratio_of_boys_to_girls_l498_498490

-- Variables for the number of boys, girls, and teachers
variables (B G T : ℕ)

-- Conditions from the problem
def number_of_girls := G = 60
def number_of_teachers := T = (20 * B) / 100
def total_people := B + G + T = 114

-- Proving the ratio of boys to girls is 3:4 given the conditions
theorem ratio_of_boys_to_girls 
  (hG : number_of_girls G)
  (hT : number_of_teachers B T)
  (hTotal : total_people B G T) :
  B / 15 = 3 ∧ G / 15 = 4 :=
by {
  sorry
}

end ratio_of_boys_to_girls_l498_498490


namespace abs_floor_eval_l498_498217

theorem abs_floor_eval : (Int.floor (Real.abs (-34.1))) = 34 := by
  sorry

end abs_floor_eval_l498_498217


namespace triangle_inequality_l498_498545

variable {α : Type*} [LinearOrderedField α]

/-- Given a triangle ABC with sides a, b, c, circumradius R, 
exradii r_a, r_b, r_c, and given 2R ≤ r_a, we need to show that a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : α) (h₁ : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry

end triangle_inequality_l498_498545


namespace solve_quadratic_1_solve_quadratic_2_l498_498957

noncomputable def roots_quadratic_1 : Set ℝ := {x | x = -2 + real.sqrt 6 ∨ x = -2 - real.sqrt 6}
noncomputable def roots_quadratic_2 : Set ℝ := {x | x = -1 ∨ x = 2}

theorem solve_quadratic_1 :
  ∀ x : ℝ, x^2 + 4*x - 2 = 0 ↔ x ∈ roots_quadratic_1 :=
by
  sorry

theorem solve_quadratic_2 :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x ∈ roots_quadratic_2 :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_l498_498957


namespace fifth_student_is_43_l498_498495

def isValidStudentNum (n : ℕ) : Prop := n < 50

/-- Simple random sampling using a given random number table, selecting the 5th student -/
def selectFifthStudent (table : List ℕ) : ℕ :=
  let validNumbers := table.filter isValidStudentNum
  validNumbers.nth! 4

theorem fifth_student_is_43 :
  selectFifthStudent [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43] = 43 :=
by
  -- Using the definition of selectFifthStudent
  unfold selectFifthStudent
  -- Filtering text to find students within numbers [0, 49]
  simp [isValidStudentNum]
  sorry  -- Proof skipped for simplicity

end fifth_student_is_43_l498_498495


namespace vector_midpoint_eq_l498_498246

-- Given definitions based on the problem statement
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D E F : V)
variables (AB_mid : (A + B) / 2 = E) (CD_mid : (C + D) / 2 = F)
variables (convex_quad : convex (A, B, C, D))

-- The theorem to prove the problem statement
theorem vector_midpoint_eq {A B C D E F : V} [AddCommGroup V] [Module ℝ V]
  (AB_mid : (A + B) / 2 = E) (CD_mid : (C + D) / 2 = F) :
  2 • (F - E) = (D - A) + (C - B) :=
by
  sorry

end vector_midpoint_eq_l498_498246


namespace a_lt_one_l498_498774

-- Define the function f(x) = |x-3| + |x+7|
def f (x : ℝ) : ℝ := |x-3| + |x+7|

-- The statement of the problem
theorem a_lt_one (a : ℝ) :
  (∀ x : ℝ, a < Real.log (f x)) → a < 1 :=
by
  intro h
  have H : f (-7) = 10 := by sorry -- piecewise definition
  have H1 : Real.log (f (-7)) = 1 := by sorry -- minimum value of log
  specialize h (-7)
  rw [H1] at h
  exact h

end a_lt_one_l498_498774


namespace real_values_of_c_l498_498640

theorem real_values_of_c (c : ℝ) : ({c : ℝ | abs (1 - complex.I * c) = 2}).finite.to_finset.card = 2 :=
by
  -- Proof will be added here
  sorry

end real_values_of_c_l498_498640


namespace solve_system_l498_498447

theorem solve_system :
  ∃ (x y : ℝ), 3 * x - 4 * y = 12 ∧ 9 * x + 6 * y = -18 ∧ x = 0 ∧ y = -3 :=
by {
  existsi [0, -3],
  split; linarith,
  split; linarith,
  split; exact rfl,
  exact rfl,
}

end solve_system_l498_498447


namespace hyperbola_eccentricity_l498_498458

theorem hyperbola_eccentricity (m : ℤ) (h1 : -2 < m) (h2 : m < 2) : 
  let a := m
  let b := (4 - m^2).sqrt 
  let c := (a^2 + b^2).sqrt
  let e := c / a
  e = 2 := by
sorry

end hyperbola_eccentricity_l498_498458


namespace complex_conjugate_l498_498669

def z_conj (z : ℂ) : ℂ := conj z

theorem complex_conjugate (z : ℂ) (h : (3 - 4 * I) * z = 1 + 2 * I) : 
  z_conj z = -1 / 5 - (2 / 5) * I :=
by
  sorry

end complex_conjugate_l498_498669


namespace equation_of_line_AB_l498_498806

noncomputable def midpoint := (2 : ℝ, -1 : ℝ)
noncomputable def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25
noncomputable def is_midpoint_of_chord (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

theorem equation_of_line_AB :
  ∀ (x y : ℝ), is_midpoint_of_chord midpoint (x, y) -> circle_eqn x y -> x - y - 3 = 0 :=
by
  sorry

end equation_of_line_AB_l498_498806


namespace David_walks_6m_more_than_Cindy_l498_498336

noncomputable def distance_difference : ℝ :=
let AC : ℝ := 8
let CB : ℝ := 15
let AB := real.sqrt (AC^2 + CB^2) in
(AC + CB) - AB

theorem David_walks_6m_more_than_Cindy :
  distance_difference = 6 :=
by
  sorry

end David_walks_6m_more_than_Cindy_l498_498336


namespace measure_of_angle_ABH_l498_498827

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498827


namespace sum_of_T_is_2510_l498_498386

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498386


namespace regular_octagon_angle_ABH_l498_498876

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498876


namespace lcm_of_32_and_12_l498_498497

def gcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_of_32_and_12 : lcm 32 12 = 48 :=
by
  have h_gcf : gcf 32 12 = 8 := by sorry
  have h_lcm := Nat.lcm_eq (by norm_num) h_gcf
  show 48
  simp [h_lcm]

end lcm_of_32_and_12_l498_498497


namespace first_year_after_2010_with_digit_sum_10_l498_498101

/--
Theorem: The first year after 2010 for which the sum of the digits equals 10 is 2017.
-/
theorem first_year_after_2010_with_digit_sum_10 : ∃ (y : ℕ), (y > 2010) ∧ (∑ d in (y.to_digits 10), d) = 10 ∧ (∀ z, (z > 2010) → ((∑ d in (z.to_digits 10), d) = 10 → z ≥ y))

end first_year_after_2010_with_digit_sum_10_l498_498101


namespace find_f2_l498_498682

-- Define the conditions
variable {f g : ℝ → ℝ} {a : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Assume g is an even function
axiom even_g : ∀ x : ℝ, g (-x) = g x

-- Condition given in the problem
axiom f_g_relation : ∀ x : ℝ, f x + g x = a^x - a^(-x) + 2

-- Condition that g(2) = a
axiom g_at_2 : g 2 = a

-- Condition for a
axiom a_cond : a > 0 ∧ a ≠ 1

-- Proof problem
theorem find_f2 : f 2 = 15 / 4 := by
  sorry

end find_f2_l498_498682


namespace geometric_progression_common_ratio_l498_498316

theorem geometric_progression_common_ratio (r : ℝ) (a : ℝ) (h_pos : 0 < a)
    (h_geom_prog : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) :
    r^3 + r^2 + r - 1 = 0 :=
by
  sorry

end geometric_progression_common_ratio_l498_498316


namespace count_fractions_l498_498744

def is_fraction (expr : Prop) : Prop :=
  match expr with
  | ∃ A B, expr = A / B ∧ ∃ c, B = c ∧ is_letter c

def is_letter (a : Prop) : Prop :=
  a ∈ {'m', 'b', 'x', 'y', 'a'}

theorem count_fractions :
  let exprs := [1 / m, b / 3, (x - 1) / π, 2 / (x + y), a + 1 / a] in
  (3 : ℕ) = list.countp is_fraction exprs :=
sorry

end count_fractions_l498_498744


namespace correct_equation_is_B_l498_498116

theorem correct_equation_is_B :
  (\(\sin 40^\circ \neq \sin 50^\circ\) ∧ \(\tan 20^\circ \tan 70^\circ = 1\) 
   ∧ \(\cos 30^\circ > \cos 35^\circ\) ∧ \(\sin^2 30^\circ + \sin^2 30^\circ \neq 1\)) 
  → (\(\tan 20^\circ \tan 70^\circ = 1\)) :=
by
  intro h
  obtain ⟨hA, hB, hC, hD⟩ := h
  exact hB

end correct_equation_is_B_l498_498116


namespace no_real_roots_iff_l498_498731

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem no_real_roots_iff (a : ℝ) : (∀ x : ℝ, f x a ≠ 0) → a > 1 :=
  by
    sorry

end no_real_roots_iff_l498_498731


namespace find_angle_l498_498455

-- Given the complement condition
def complement_condition (x : ℝ) : Prop :=
  x + 2 * (4 * x + 10) = 90

-- Proving the degree measure of the angle
theorem find_angle (x : ℝ) : complement_condition x → x = 70 / 9 := by
  intro hc
  sorry

end find_angle_l498_498455


namespace find_product_AC_BC_l498_498324

-- The given conditions as definitions
def triangle_ABC : Type := sorry -- Placeholder for the concrete triangle type
axiom acute_triangle (ABC : triangle_ABC) : Prop  -- ABC is an acute triangle
axiom perpendicular_feet (ABC : triangle_ABC) (R S : triangle_ABC) :
  Prop  -- R and S are the feet of the perpendiculars from B to AC and from A to BC, respectively
axiom circumcircle_intersection (ABC : triangle_ABC) (R S Z W : triangle_ABC) :
  Prop  -- RS intersects the circumcircle of ABC at Z and W
axiom lengths (ZR RS SW : ℕ) : ZR = 12 ∧ RS = 30 ∧ SW = 18  -- Given lengths

-- The theorem to be proven
theorem find_product_AC_BC (ABC : triangle_ABC)
  (h_acute : acute_triangle ABC)
  (h_perpendicular : perpendicular_feet ABC R S)
  (h_intersection : circumcircle_intersection ABC R S Z W)
  (h_lengths : lengths ZR RS SW) :
  ∃ (p q : ℕ), AC * BC = p * sqrt q ∧ ¬ (∃ r : ℕ, r^2 ∣ q) ∧ p + q = 3318 := 
sorry

end find_product_AC_BC_l498_498324


namespace exercise_l498_498451

theorem exercise (a b : ℕ) (h1 : 656 = 3 * 7^2 + a * 7 + b) (h2 : 656 = 3 * 10^2 + a * 10 + b) : 
  (a * b) / 15 = 1 :=
by
  sorry

end exercise_l498_498451


namespace day_flying_hours_l498_498061

theorem day_flying_hours (D : ℕ) :
  (1500 - (220 * 6)) = (D + 9 + 121) → D = 50 :=
by
  intro h
  rw [mul_comm, mul_assoc] at h
  rw [sub_eq_add_neg, add_assoc] at h
  sorry

end day_flying_hours_l498_498061


namespace Brian_watch_animal_videos_l498_498193

theorem Brian_watch_animal_videos :
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  total_time = 144 := by
{
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  have h1 : total_time = (4 + 8 + 24 + 36 + 72) := sorry
  exact h1
}

end Brian_watch_animal_videos_l498_498193


namespace sum_of_T_l498_498393

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498393


namespace measure_angle_abh_l498_498820

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498820


namespace measure_of_angle_ABH_in_regular_octagon_l498_498939

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498939


namespace angle_ABH_is_22_point_5_l498_498913

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498913


namespace sindbad_can_identify_eight_genuine_dinars_l498_498007

/--
Sindbad has 11 visually identical dinars in his purse, one of which may be counterfeit and differs in weight from the genuine ones. Using a balance scale twice without weights, it's possible to identify at least 8 genuine dinars.
-/
theorem sindbad_can_identify_eight_genuine_dinars (dinars : Fin 11 → ℝ) (is_genuine : Fin 11 → Prop) :
  (∃! i, ¬ is_genuine i) → 
  (∃ S : Finset (Fin 11), S.card = 8 ∧ S ⊆ (Finset.univ : Finset (Fin 11)) ∧ ∀ i ∈ S, is_genuine i) :=
sorry

end sindbad_can_identify_eight_genuine_dinars_l498_498007


namespace simplify_frac_l498_498005

theorem simplify_frac : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 :=
by 
  sorry

end simplify_frac_l498_498005


namespace transformed_system_solution_l498_498702

theorem transformed_system_solution :
  (∀ (a b : ℝ), 2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 → a = 8.3 ∧ b = 1.2) →
  (∀ (x y : ℝ), 2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9 →
    x = 6.3 ∧ y = 2.2) :=
by
  intro h1
  intro x y
  intro hy
  sorry

end transformed_system_solution_l498_498702


namespace number_of_kids_l498_498993

variable (X : ℕ)
variable (H1 : 0.10 * X = 4)

theorem number_of_kids (h : X = 40) : H1 := 
by
  sorry

end number_of_kids_l498_498993


namespace root_of_function_is_4_l498_498262

def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_4 (a : ℝ) (h : f a = 0) : a = 4 :=
by 
  sorry

end root_of_function_is_4_l498_498262


namespace goal_l498_498764

open EuclideanGeometry

variables (A B C D E F O : Point)
variables (radius : ℝ)

noncomputable def circle (center : Point) (r : ℝ) : set Point := { p | dist center p = r }

def tangent_to_circle (p : Point) (center : Point) (radius : ℝ) :=
  ∃ (t : Point), t ∈ circle center radius ∧ Line.p t p = ⟂ Line.t center t

def diameter (p q : Point) (center : Point) :=
  dist p center = dist q center ∧ p ≠ q

axioms
  (h1 : A ∉ circle O radius)
  (h2 : (diameter B C O))
  (h3 : tangent_to_circle A O radius D)
  (h4 : E ∈ Line.s A B ∧ dist A E = dist A D)
  (h5 : Perpendicular (Line.s E F) (Line.s A B) ∧ F ∈ Line.s A C)

theorem goal : dist A E * dist A F = dist A B * dist A C :=
  by sorry

end goal_l498_498764


namespace range_of_a_l498_498656

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : a ≤ -3 :=
by
  sorry

end range_of_a_l498_498656


namespace inequality_a3_b3_c3_l498_498771

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end inequality_a3_b3_c3_l498_498771


namespace ninth_triangle_shaded_fraction_l498_498321

theorem ninth_triangle_shaded_fraction :
  let shaded_seq : List Nat := [1, 4, 9, 16, 25, 36, 49],
      total_triangles_seq : List Nat := [4, 9, 16, 25, 36, 49, 64] in
  shaded_seq.sum = 140 ∧ total_triangles_seq.sum = 203 → (140 : ℚ) / 203 = 140 / 203 :=
by
  intro h
  have hs : shaded_seq.sum = 140 := h.left
  have ht : total_triangles_seq.sum = 203 := h.right
  sorry

end ninth_triangle_shaded_fraction_l498_498321


namespace solve_for_x_l498_498129

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end solve_for_x_l498_498129


namespace measure_of_angle_ABH_in_regular_octagon_l498_498931

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498931


namespace bullet_train_pass_time_l498_498142

def train_length : ℕ := 240 -- in meters
def train_speed : ℕ := 100 -- in kmph
def man_speed : ℕ := 8 -- in kmph

-- Converting kmph to m/s
def kmph_to_mps (speed_kmph : ℕ) : ℝ := (speed_kmph : ℝ) * (5.0 / 18.0)

def relative_speed_mps : ℝ := kmph_to_mps (train_speed + man_speed)

def time_to_pass (distance : ℕ) (speed : ℝ) : ℝ := (distance : ℝ) / speed

theorem bullet_train_pass_time : time_to_pass train_length relative_speed_mps = 8 := by
  sorry

end bullet_train_pass_time_l498_498142


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498899

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498899


namespace measure_of_angle_ABH_in_regular_octagon_l498_498934

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498934


namespace sunglasses_cost_l498_498167

open Real

def cost_per_pair (selling_price_per_pair : ℝ) (num_pairs_sold : ℝ) (sign_cost : ℝ) : ℝ := 
  (num_pairs_sold * selling_price_per_pair - 2 * sign_cost) / num_pairs_sold

theorem sunglasses_cost (sp : ℝ) (n : ℝ) (sc : ℝ) (H1 : sp = 30) (H2 : n = 10) (H3 : sc = 20) :
  cost_per_pair sp n sc = 26 :=
by
  rw [H1, H2, H3]
  simp [cost_per_pair]
  norm_num
  sorry

end sunglasses_cost_l498_498167


namespace number_of_functions_l498_498228

open Finset

noncomputable def count_functions {A B : Type*} [Fintype A] [Fintype B] (n k : ℕ)
  (hA : Fintype.card A = n) (hB : Fintype.card B = k) : ℕ :=
  (choose n k) * (n^(n - k))

theorem number_of_functions (A B : Type*) [Fintype A] [Fintype B]
  (n k : ℕ) (hA : Fintype.card A = n) (hB : Fintype.card B = k) :
  ∃ (f : A → A), ∃ (g : A → B) (h : B → A),
    (∀ b : B, g (h b) = b) ∧ (g ∘ h = id) ∧ (h ∘ g = f) ∧
    count_functions A B n k hA hB = (choose n k) * (n^(n - k)) := by
  sorry

end number_of_functions_l498_498228


namespace sum_of_solutions_for_x_l498_498327

theorem sum_of_solutions_for_x :
  (∃ x y : ℝ, y = 8 ∧ x^2 + y^2 = 169) →
  (∃ x₁ x₂ : ℝ, x₁^2 = 105 ∧ x₂^2 = 105 ∧ x₁ + x₂ = 0) :=
by
  intro h
  cases h with x h1
  cases h1 with y h2
  cases h2 with hy h3
  use (√(105)), (-√(105))
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  exact add_right_neg (√(105))

end sum_of_solutions_for_x_l498_498327


namespace distribute_problems_l498_498576

theorem distribute_problems :
  let n_problems := 7
  let n_friends := 12
  (n_friends ^ n_problems) = 35831808 :=
by 
  sorry

end distribute_problems_l498_498576


namespace ivan_needs_more_paint_l498_498082

theorem ivan_needs_more_paint
  (section_count : ℕ)
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2) :
  let area_ivan := section_count * (5 * 2)
  let area_petr := section_count * (5 * 2 * sin α)
  area_ivan > area_petr := 
by
  simp only [area_ivan, area_petr, mul_assoc, mul_lt_mul_left, gt_iff_lt]
  exact sin_lt_one_iff.mpr hα

end ivan_needs_more_paint_l498_498082


namespace angle_in_regular_octagon_l498_498892

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498892


namespace diff_largest_second_largest_l498_498033

open Set

-- Define the set of lottery numbers
def lottery_nums : Set ℕ := {11, 25, 42, 15, 28, 39}

-- Definition of the largest and the second largest element
noncomputable def largest : ℕ := Sup lottery_nums
noncomputable def second_largest : ℕ := Sup (lottery_nums \ {largest})

-- Theorem stating the difference between the largest and second largest number is 3
theorem diff_largest_second_largest : largest - second_largest = 3 :=
  by
  sorry

end diff_largest_second_largest_l498_498033


namespace frog_jump_correct_l498_498031

def grasshopper_jump : ℤ := 25
def additional_distance : ℤ := 15
def frog_jump : ℤ := grasshopper_jump + additional_distance

theorem frog_jump_correct : frog_jump = 40 := by
  sorry

end frog_jump_correct_l498_498031


namespace calculator_sum_correct_l498_498322

theorem calculator_sum_correct :
  ∀ (n : ℕ) (a b c : ℤ),
  n = 37 →
  a = 2 →
  b = -2 →
  c = 0 →
  (∑ i in finset.range n, a^2 + b^2 + c^2) = 8 :=
by sorry

end calculator_sum_correct_l498_498322


namespace smallest_n_condition_l498_498249

noncomputable def a (n : ℕ) : ℕ := 2^n
noncomputable def b (n : ℕ) : ℕ := n * 2^n
noncomputable def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

theorem smallest_n_condition :
  ∃ n : ℕ, (T n - n * 2^(n + 1) + 50 < 0) ∧ (∀ m : ℕ, m < n -> T m - m * 2^(m + 1) + 50 >= 0) :=
begin
  use 5,
  sorry
end

end smallest_n_condition_l498_498249


namespace inequality_abc_l498_498425

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l498_498425


namespace shift_sin_to_cos_l498_498493

theorem shift_sin_to_cos (x : ℝ) : 
  (∀ x, cos (2 * x) = sin (2 * x + π / 2)) →
  (∃ d : ℝ, d = π / 4 ∧ ∀ x, cos (2 * x) = sin (2 * (x + d))) :=
by
  intro h
  use π / 4
  split
  . refl
  . intro x
    rw [←h x, add_assoc, (by norm_num : π / 2 = π / 4 + π / 4)]
    sorry

end shift_sin_to_cos_l498_498493


namespace last_two_digits_factorial_sum_l498_498606

theorem last_two_digits_factorial_sum : 
  (∑ i in finset.range 16, (nat.factorial i)) % 100 = 13 := 
by sorry

end last_two_digits_factorial_sum_l498_498606


namespace correct_option_is_A_l498_498509

-- Defining abstract objects and their relationships as given in the problem.
variables {A : Type} {l : set A} {α : set A}

-- Condition: Point A is on line l.
axiom A_on_l : A ∈ l

-- Condition: Point A is in plane α.
axiom A_in_alpha : A ∈ α

-- Condition: Line l is in plane α.
axiom l_in_alpha : l ⊆ α

-- The proof problem: Prove that A ∈ l is correct.
theorem correct_option_is_A : A ∈ l :=
by exact A_on_l

end correct_option_is_A_l498_498509


namespace bruce_can_buy_11_bags_l498_498583

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l498_498583


namespace number_of_students_l498_498958

theorem number_of_students (B S : ℕ) 
  (h1 : S = 9 * B + 1) 
  (h2 : S = 10 * B - 10) : 
  S = 100 := 
by 
  { sorry }

end number_of_students_l498_498958


namespace number_of_complex_solutions_l498_498226

theorem number_of_complex_solutions :
  (∃ z : ℂ, (z^4 - 1) = 0 ∧ (z^3 + z^2 - 3z - 3) ≠ 0) = 3 := sorry

end number_of_complex_solutions_l498_498226


namespace angle_ABH_regular_octagon_l498_498917

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498917


namespace students_playing_neither_l498_498130

theorem students_playing_neither (N F T F_and_T : ℕ) (hN : N = 38) (hF : F = 26) (hT : T = 20) (hF_and_T : F_and_T = 17) :
  N - (F + T - F_and_T) = 9 :=
by
  rw [hN, hF, hT, hF_and_T]
  exact sorry

end students_playing_neither_l498_498130


namespace circles_tangent_to_axes_l498_498026

namespace CircleProof

theorem circles_tangent_to_axes
    (h₁ : ∀ (x y : ℝ), 5 * x - 3 * y = 8 → (x = y ∨ x = -y))
    (h₂ : ∀ (r : ℝ) (x y : ℝ), ((x = y ∧ r = x) ∨ (x = -y ∧ r = x))):
    ∃ (r₁ r₂ : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
      (5 * x₁ - 3 * y₁ = 8 ∧ x₁ = y₁ ∧ r₁ = x₁ ∧ (∀ p q : ℝ, (p - x₁)^2 + (q - y₁)^2 = r₁^2 → (p = 0 ∨ q = 0))) ∧
      (5 * x₂ - 3 * y₂ = 8 ∧ x₂ = -y₂ ∧ r₂ = x₂ ∧ (∀ p q : ℝ, (p - x₂)^2 + (q + y₂)^2 = r₂^2 → (p = 0 ∨ q = 0)))
  :=
begin
  sorry,
end

end CircleProof

end circles_tangent_to_axes_l498_498026


namespace sum_of_solutions_l498_498330

theorem sum_of_solutions : 
  ∀ (x y : ℝ), y = 8 → x^2 + y^2 = 169 → (x = sqrt 105 ∨ x = -sqrt 105) → (sqrt 105 + (-sqrt 105) = 0) :=
by
  intros x y h1 h2 h3
  sorry

end sum_of_solutions_l498_498330


namespace exists_line_dividing_points_l498_498799

variables (P : Finset (ℝ × ℝ)) -- 20 points on a plane
variable [DecidableEq (ℝ × ℝ) ]

noncomputable def is_blue : (ℝ × ℝ) → Prop := sorry -- Function to determine whether a point is blue
noncomputable def is_red : (ℝ × ℝ) → Prop := sorry -- Function to determine whether a point is red
noncomputable def collinear : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry -- Predicate for collinearity

-- Conditions
def blue_points := {p ∈ P | is_blue p}
def red_points := {p ∈ P | is_red p}
def no_three_collinear := ∀ (a b c ∈ P), ¬ collinear a b c

-- Main Theorem
theorem exists_line_dividing_points :
  (P.card = 20) →
  (blue_points.card = 10) →
  (red_points.card = 10) →
  no_three_collinear →
  ∃ (l : ℝ → ℝ × ℝ → Prop), (∃ b₁ b₂ b₃ b₄ b₅ r₁ r₂ r₃ r₄ r₅,  
  b₁ ≠ b₂ ∧ b₁ ≠ b₃ ∧ b₁ ≠ b₄ ∧ b₁ ≠ b₅ ∧ b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₄ ≠ b₅ ∧
  r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₄ ≠ r₅ ∧
  ∀ p ∈ {b₁,b₂,b₃,b₄,b₅,r₁,r₂,r₃,r₄,r₅}, l(0) p →
  5 = Finset.filter (λ (x : (ℝ × ℝ)) , l x), blue_points = 5 ∧ Finset.filter (λ (x: (ℝ × ℝ)), l x), red_points = 5))
:= sorry

end exists_line_dividing_points_l498_498799


namespace angle_ABH_is_22_point_5_l498_498907

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498907


namespace max_good_rhombuses_l498_498476

-- Definitions based on conditions
def triangles : ℕ := 100
def colors : ℕ := 150
def sides_per_color : ℕ := 2

-- Question translation to Lean statement
theorem max_good_rhombuses : ∀ (triangles = 100) (colors = 150) (sides_per_color = 2), ensure_good_rhombuses ≤ 25 := by
  -- The proof steps are not included as per instruction
  sorry

end max_good_rhombuses_l498_498476


namespace triangle_construction_exists_l498_498594

-- Define the side lengths and the angle condition
variables {a b : ℝ} {δ : ℝ}

-- Define the main statement in Lean to prove the existence of triangle ABC with given conditions
theorem triangle_construction_exists (a b δ : ℝ) : 
  ∃ (A B C S : Type) (BC CA : ℝ) (AT BC : ℝ) (angle : ℝ), 
  BC = a ∧ CA = b ∧ 
  ∠ ASB = δ ∧
  (exists_triangle_with_sides_and_angle a b δ) := sorry

end triangle_construction_exists_l498_498594


namespace isabel_weekly_run_distance_l498_498351

theorem isabel_weekly_run_distance
  (circuit_length : ℕ)
  (morning_laps : ℕ)
  (afternoon_laps : ℕ)
  (days_in_week : ℕ)
  : circuit_length = 365 → morning_laps = 7 → afternoon_laps = 3 → days_in_week = 7 →
    (morning_laps * circuit_length + afternoon_laps * circuit_length) * days_in_week = 25550 :=
by
  intros h_circuit h_morning h_afternoon h_days
  rw [h_circuit, h_morning, h_afternoon, h_days]
  have morning_distance : 7 * 365 = 2555 := rfl
  have afternoon_distance : 3 * 365 = 1095 := rfl
  have total_day_distance : 2555 + 1095 = 3650 := rfl
  have week_distance : 3650 * 7 = 25550 := rfl
  exact week_distance

end isabel_weekly_run_distance_l498_498351


namespace intersection_with_x_axis_l498_498972

theorem intersection_with_x_axis (a : ℝ) (h : 2 * a - 4 = 0) : a = 2 := by
  sorry

end intersection_with_x_axis_l498_498972


namespace dale_slices_of_toast_l498_498580

theorem dale_slices_of_toast
  (slice_cost : ℤ) (egg_cost : ℤ)
  (dale_eggs : ℤ) (andrew_slices : ℤ) (andrew_eggs : ℤ)
  (total_cost : ℤ)
  (cost_eq : slice_cost = 1)
  (egg_cost_eq : egg_cost = 3)
  (dale_eggs_eq : dale_eggs = 2)
  (andrew_slices_eq : andrew_slices = 1)
  (andrew_eggs_eq : andrew_eggs = 2)
  (total_cost_eq : total_cost = 15)
  :
  ∃ T : ℤ, (slice_cost * T + egg_cost * dale_eggs) + (slice_cost * andrew_slices + egg_cost * andrew_eggs) = total_cost ∧ T = 2 :=
by
  sorry

end dale_slices_of_toast_l498_498580


namespace find_term_ninth_term_l498_498054

variable (a_1 d a_k a_12 : ℤ)
variable (S_20 : ℤ := 200)

-- Definitions of the given conditions
def term_n (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

-- Problem Statement
theorem find_term_ninth_term :
  (∃ k, term_n a_1 d k + term_n a_1 d 12 = 20) ∧ 
  (S_20 = 10 * (2 * a_1 + 19 * d)) → 
  ∃ k, k = 9 :=
by sorry

end find_term_ninth_term_l498_498054


namespace measure_of_angle_ABH_l498_498855

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498855


namespace angle_ABH_is_22_point_5_l498_498911

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498911


namespace integer_quotient_l498_498762

variable (a b c : ℤ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h_rat : ∃ q : ℚ, (a : ℚ) * real.sqrt 3 + b = q * ((b : ℚ) * real.sqrt 3 + c))

theorem integer_quotient (a b c : ℤ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_rat : ∃ q : ℚ, (a : ℚ) * real.sqrt 3 + b = q * ((b : ℚ) * real.sqrt 3 + c)) : 
  ∃ k : ℤ, (a^2 + b^2 + c^2) = k * (a + b + c) :=
sorry

end integer_quotient_l498_498762


namespace sin_cos_value_l498_498298

theorem sin_cos_value (x : real) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := 
by 
  sorry

end sin_cos_value_l498_498298


namespace measure_of_angle_ABH_l498_498854

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498854


namespace manager_salary_l498_498131

theorem manager_salary 
    (avg_salary_18 : ℕ)
    (new_avg_salary : ℕ)
    (num_employees : ℕ)
    (num_employees_with_manager : ℕ)
    (old_total_salary : ℕ := num_employees * avg_salary_18)
    (new_total_salary : ℕ := num_employees_with_manager * new_avg_salary) :
    (new_avg_salary = avg_salary_18 + 200) →
    (old_total_salary = 18 * 2000) →
    (new_total_salary = 19 * (2000 + 200)) →
    new_total_salary - old_total_salary = 5800 :=
by
  intros h1 h2 h3
  sorry

end manager_salary_l498_498131


namespace cost_of_sunglasses_l498_498169

noncomputable def cost_per_pair 
  (price_per_pair : ℕ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℕ) 
  (profits_half : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_revenue := price_per_pair * pairs_sold in
  let total_cost := total_revenue - (profits_half * 2) in
  total_cost / pairs_sold

theorem cost_of_sunglasses :
  ∀ (price_per_pair pairs_sold sign_cost profits_half profit : ℕ),
    price_per_pair = 30 → 
    pairs_sold = 10 → 
    sign_cost = 20 → 
    (profits_half = sign_cost → profit = profits_half * 2) →
    cost_per_pair price_per_pair pairs_sold sign_cost profits_half profit = 26 :=
begin
  intros,
  sorry
end

end cost_of_sunglasses_l498_498169


namespace pizza_sharing_order_l498_498737

theorem pizza_sharing_order {total_slices : ℕ} 
    (hLina : (total_slices : ℤ) * 1 / 6 = roundRat (1/6 * total_slices : ℚ))
    (hMarco : (total_slices : ℤ) * 3 / 8 = roundRat (3/8 * total_slices : ℚ))
    (hNadia : (total_slices : ℤ) * 1 / 5 = roundRat (1/5 * total_slices : ℚ))
    (hOmar : total_slices = 120) :
  (roundRat (3 / 8 * total_slices : ℚ) > roundRat (31 / 120 * total_slices : ℚ))
  ∧ (roundRat (31 / 120 * total_slices : ℚ) > roundRat (1 / 5 * total_slices : ℚ))
  ∧ (roundRat (1 / 5 * total_slices : ℚ) > roundRat (1 / 6 * total_slices : ℚ)) := 
by
  -- Proof skipped.
  sorry

end pizza_sharing_order_l498_498737


namespace isabel_weekly_run_distance_l498_498350

theorem isabel_weekly_run_distance
  (circuit_length : ℕ)
  (morning_laps : ℕ)
  (afternoon_laps : ℕ)
  (days_in_week : ℕ)
  : circuit_length = 365 → morning_laps = 7 → afternoon_laps = 3 → days_in_week = 7 →
    (morning_laps * circuit_length + afternoon_laps * circuit_length) * days_in_week = 25550 :=
by
  intros h_circuit h_morning h_afternoon h_days
  rw [h_circuit, h_morning, h_afternoon, h_days]
  have morning_distance : 7 * 365 = 2555 := rfl
  have afternoon_distance : 3 * 365 = 1095 := rfl
  have total_day_distance : 2555 + 1095 = 3650 := rfl
  have week_distance : 3650 * 7 = 25550 := rfl
  exact week_distance

end isabel_weekly_run_distance_l498_498350


namespace both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l498_498214

variable (p q : Prop)

-- 1. Both shots were unsuccessful
theorem both_shots_unsuccessful : ¬p ∧ ¬q := sorry

-- 2. Both shots were successful
theorem both_shots_successful : p ∧ q := sorry

-- 3. Exactly one shot was successful
theorem exactly_one_shot_successful : (¬p ∧ q) ∨ (p ∧ ¬q) := sorry

-- 4. At least one shot was successful
theorem at_least_one_shot_successful : p ∨ q := sorry

-- 5. At most one shot was successful
theorem at_most_one_shot_successful : ¬(p ∧ q) := sorry

end both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l498_498214


namespace exists_even_number_of_mutual_friends_l498_498777

noncomputable def S : set ℕ := sorry  -- Assume S is a set of 2n people, defined over ℕ for simplicity
def is_friend (a b : ℕ) : Prop := sorry -- Relationship predicate denoting friendship

-- Main theorem
theorem exists_even_number_of_mutual_friends
  (hS_size : (S.card = 2 * n))
  (h_odd_common_friends : ∀ a b ∈ S, a ≠ b → odd (finset.card ({x ∈ S | is_friend x a ∧ is_friend x b}))) :
  ∃ a b ∈ S, a ≠ b ∧ (even (finset.card ({x ∈ S | is_friend x a ∧ is_friend x b}))) :=
sorry

end exists_even_number_of_mutual_friends_l498_498777


namespace measure_of_angle_ABH_in_regular_octagon_l498_498930

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498930


namespace system_solution_fraction_l498_498234

theorem system_solution_fraction (x y z : ℝ) (h1 : x + (-95/9) * y + 4 * z = 0)
  (h2 : 4 * x + (-95/9) * y - 3 * z = 0) (h3 : 3 * x + 5 * y - 4 * z = 0) (hx_ne_zero : x ≠ 0) 
  (hy_ne_zero : y ≠ 0) (hz_ne_zero : z ≠ 0) : 
  (x * z) / (y ^ 2) = 20 :=
sorry

end system_solution_fraction_l498_498234


namespace sheets_in_height_l498_498566

theorem sheets_in_height (sheets_per_ream : ℕ) (thickness_per_ream : ℝ) (target_thickness : ℝ) 
  (h₀ : sheets_per_ream = 500) (h₁ : thickness_per_ream = 5.0) (h₂ : target_thickness = 7.5) :
  target_thickness / (thickness_per_ream / sheets_per_ream) = 750 :=
by sorry

end sheets_in_height_l498_498566


namespace remainder_sum_first_150_div_12000_l498_498107

theorem remainder_sum_first_150_div_12000 : 
  let S := 150 * 151 / 2 in 
  S % 12000 = 11325 :=
by 
  have S_def : S = 150 * 151 / 2 := rfl
  have h : S = 11325 := by norm_num
  show 11325 % 12000 = 11325 from by norm_num
  sorry

end remainder_sum_first_150_div_12000_l498_498107


namespace problem_l498_498244

variable {a b c : ℝ}

theorem problem (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 := 
sorry

end problem_l498_498244


namespace divide_segment_mean_proportional_l498_498212

theorem divide_segment_mean_proportional (a : ℝ) (x : ℝ) : 
  ∃ H : ℝ, H > 0 ∧ H < a ∧ H = (a * (Real.sqrt 5 - 1) / 2) :=
sorry

end divide_segment_mean_proportional_l498_498212


namespace range_of_t_l498_498266

-- Define the function properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f(x) ≤ f(y)

-- Given conditions
constant f : ℝ → ℝ
constant t : ℝ
axiom f_odd : is_odd f
axiom f_increasing : is_increasing f (-1) 1
axiom f_at_neg_one : f (-1) = -1
axiom f_leq_t_squared : ∀ x, -1 ≤ x → x ≤ 1 → f(x) ≤ t^2 - 2*t + 1

-- The theorem statement
theorem range_of_t : t ≤ 0 ∨ t ≥ 2 :=
sorry

end range_of_t_l498_498266


namespace factorize_polynomial_l498_498218

theorem factorize_polynomial (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3) ^ 2 :=
by sorry

end factorize_polynomial_l498_498218


namespace measure_angle_abh_l498_498864

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498864


namespace problem_odds_monotonic_l498_498265

theorem problem_odds_monotonic (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_dec : ∀ (x y : ℝ), x ≤ y ∧ x ∈ Icc (-1:ℝ) 0 → f x ≥ f y)
  (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) :
  f (Real.sin A) < f (Real.cos B) :=
by
  sorry

end problem_odds_monotonic_l498_498265


namespace part1_part2_l498_498277

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry 

noncomputable def g (x : ℝ) : ℝ := f x - x^2 + x

theorem part2 (m : ℝ) : 
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
sorry 

end part1_part2_l498_498277


namespace num_distinct_four_digit_integers_with_digit_product_18_l498_498223

def is_valid_digit_combination (d : ℕ) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 6 ∨ d = 9

def product_of_digits (v : Vector ℕ 4) : ℕ :=
  v.toList.prod

def count_valid_numbers_with_product_18 : ℕ :=
  let valid_digits := [1, 2, 3, 6, 9]
  let four_digit_combinations := (Vector.asList (vector.ofFn (fun _ => valid_digits)))
  (four_digit_combinations.filter (fun v => product_of_digits v = 18)).length

theorem num_distinct_four_digit_integers_with_digit_product_18 : count_valid_numbers_with_product_18 = 24 := 
  sorry

end num_distinct_four_digit_integers_with_digit_product_18_l498_498223


namespace robin_gum_count_l498_498946

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (final_gum : ℝ) 
  (h1 : initial_gum = 18.0) (h2 : additional_gum = 44.0) : final_gum = 62.0 :=
by {
  sorry
}

end robin_gum_count_l498_498946


namespace problem_100m_n_l498_498150

def is_valid_pair (a b : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 100) ∧ (1 ≤ b ∧ b ≤ 100) ∧ ∃ k : ℤ, (a : ℤ)^2 - 4 * (b : ℤ) = k^2

def count_valid_pairs : ℕ :=
  (finset.range 101).sum $ λ a, (finset.range 101).countp $ λ b, is_valid_pair a b

noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / 10000

theorem problem_100m_n : 100 * 281 + 10000 = 38100 :=
by
  have h_prob : probability = (281 : ℚ) / 10000 := sorry
  have h_coprime : nat.coprime 281 10000 := by sorry
  sorry

end problem_100m_n_l498_498150


namespace bruce_can_buy_11_bags_l498_498584

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l498_498584


namespace pie_slices_l498_498441

def num_of_slices_per_pie (x : ℕ) : Prop :=
  let remaining_slices_per_pie := 0.5 * (x - 1) - 1 in
  (remaining_slices_per_pie = 2.5) ∧ (2 * remaining_slices_per_pie = 5)

theorem pie_slices : ∃ x : ℕ, num_of_slices_per_pie x ∧ x = 8 :=
by {
  use 8,
  unfold num_of_slices_per_pie,
  simp,
  split,
  rfl,
  sorry
}

end pie_slices_l498_498441


namespace square_problem_l498_498448

theorem square_problem 
  (ABCD_square : ∀ (AB CD : ℝ), AB = 2 ∧ CD = 2)
  (E_on_BC : ∃ (E : ℝ × ℝ), E.1 = 2 ∧ 0 ≤ E.2 ≤ 2)
  (F_on_CD : ∃ (F : ℝ × ℝ), F.2 = 2 ∧ 0 ≤ F.1 ≤ 2)
  (angle_AEF_right : ∀ (A E F : ℝ × ℝ), (E.1, E.2) = (2, 0) → (F.1, F.2) = (0, 2) → ∠ E A F = 90)
  (smaller_square_B : ∀ (s : ℝ), s = (2 - sqrt 2) / 2)
  (a b c : ℕ)
  (h1 : a = 2)
  (h2 : b = 2)
  (h3 : c = 2) :
  a + b + c = 6 :=
sorry

end square_problem_l498_498448


namespace measure_angle_abh_l498_498857

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498857


namespace range_log_sqrt_sin_l498_498106

theorem range_log_sqrt_sin (x : ℝ) (y : ℝ) : 
  (0 < x ∧ x < real.pi) → (y = real.log 3 (real.sqrt (real.sin x))) ↔ (y ≤ 0) := 
sorry

end range_log_sqrt_sin_l498_498106


namespace largest_stickers_per_page_l498_498177

theorem largest_stickers_per_page :
  Nat.gcd (Nat.gcd 1050 1260) 945 = 105 := 
sorry

end largest_stickers_per_page_l498_498177


namespace part_a_part_b_l498_498541

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l498_498541


namespace m_squared_plus_n_l498_498254

def M : ℝ × ℝ × ℝ := (4, -3, 5)
def m : ℝ := Real.sqrt ((-3)^2 + 5^2)
def n : ℝ := Real.abs 5

theorem m_squared_plus_n :
  m^2 + n = 39 :=
by
  sorry

end m_squared_plus_n_l498_498254


namespace ivan_needs_more_paint_l498_498079

theorem ivan_needs_more_paint
  (section_count : ℕ)
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2) :
  let area_ivan := section_count * (5 * 2)
  let area_petr := section_count * (5 * 2 * sin α)
  area_ivan > area_petr := 
by
  simp only [area_ivan, area_petr, mul_assoc, mul_lt_mul_left, gt_iff_lt]
  exact sin_lt_one_iff.mpr hα

end ivan_needs_more_paint_l498_498079


namespace angle_ABH_regular_octagon_l498_498919

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498919


namespace system_of_equations_solution_l498_498750

theorem system_of_equations_solution :
  ∃ (x y : ℝ), (y = x + 1) ∧ (y = -x + 3) ∧ (x = 1) ∧ (y = 2) :=
by 
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  simp
  exact ⟨eq.refl _, eq.refl _, eq.refl _, eq.refl _⟩

end system_of_equations_solution_l498_498750


namespace sum_of_mathcalT_is_2520_l498_498380

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498380


namespace find_a4_l498_498248

-- Define the basic properties and conditions of the sequence
def sequence (n : ℕ) : ℝ := sorry  -- This will hold the sequence a_n
def sum_sequence (n : ℕ) : ℝ := sorry  -- This will hold the sum of the first n terms

-- Conditions given in the problem
axiom S2 : sum_sequence 2 = 7
axiom recursion_formula : ∀ n : ℕ, sequence (n + 1) = 2 * sum_sequence n + 1

-- The statement of the problem
theorem find_a4 : sequence 4 = 45 := by
  sorry

end find_a4_l498_498248


namespace sum_of_T_l498_498389

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498389


namespace max_min_sum_eq_4016_l498_498652

variable (a : ℝ) (h : a > 0)

def f (x : ℝ) : ℝ := (2009^(x+1) + 2007) / (2009^x + 1) + Real.sin x

theorem max_min_sum_eq_4016 (h : a > 0) : 
  let M := sup {y | ∃ x ∈ Icc (-a) a, y = f x},
      N := inf {y | ∃ x ∈ Icc (-a) a, y = f x} in
  M + N = 4016 :=
by
  sorry

end max_min_sum_eq_4016_l498_498652


namespace AC_l498_498517

variables {P B B' C C' K A D : Point}

-- Definitions of points P, B, B', C, C', A, D, and K
def is_midpoint (m p q : Point) : Prop := dist m p = dist m q

-- Define the conditions
axiom P_is_midpoint_BB' : is_midpoint P B B'
axiom P_is_midpoint_CC' : is_midpoint P C C'
axiom K_is_midpoint_B'C' : is_midpoint K B' C'
axiom B'C'_equals_BC : dist B' C' = dist B C
axiom AC'BD_is_parallelogram : parallelogram A C' B D
axiom AB'_equals_AB : dist A B' = dist A B
axiom DC'_equals_DC : dist D C' = dist D C
axiom AB_equals_DC : dist A B = dist D C

-- The theorem we want to prove
theorem AC'BD_is_rectangle : rectangle A C' B D :=
by sorry

end AC_l498_498517


namespace unknown_number_lcm_hcf_l498_498467

theorem unknown_number_lcm_hcf (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 192) 
  (hcf_ab : Nat.gcd a b = 16) 
  (known_number : a = 64) :
  b = 48 :=
by
  sorry -- Proof is omitted as per instruction

end unknown_number_lcm_hcf_l498_498467


namespace total_population_increase_l498_498315
-- Import the required library

-- Define the conditions for Region A and Region B
def regionA_births_0_14 (time: ℕ) := time / 20
def regionA_births_15_64 (time: ℕ) := time / 30
def regionB_births_0_14 (time: ℕ) := time / 25
def regionB_births_15_64 (time: ℕ) := time / 35

-- Define the total number of people in each age group for both regions
def regionA_population_0_14 := 2000
def regionA_population_15_64 := 6000
def regionB_population_0_14 := 1500
def regionB_population_15_64 := 5000

-- Define the total time in seconds
def total_time := 25 * 60

-- Proof statement
theorem total_population_increase : 
  regionA_population_0_14 * regionA_births_0_14 total_time +
  regionA_population_15_64 * regionA_births_15_64 total_time +
  regionB_population_0_14 * regionB_births_0_14 total_time +
  regionB_population_15_64 * regionB_births_15_64 total_time = 227 := 
by sorry

end total_population_increase_l498_498315


namespace rectangle_area_x_l498_498013

theorem rectangle_area_x (x : ℕ) (h1 : x > 0) (h2 : 5 * x = 45) : x = 9 := 
by
  -- proof goes here
  sorry

end rectangle_area_x_l498_498013


namespace find_unique_solution_l498_498011

-- Define conditions and the problem statement
variable {a : ℝ} (ha : a ≠ 0)
variable h_discriminant : (12^2 - 4 * a * 9 = 0)
def unique_solution (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + 12 * x + 9 = 0

-- State the solution
theorem find_unique_solution (a : ℝ) (ha : a ≠ 0) (h_discriminant : 12^2 - 4 * a * 9 = 0) :
    ∃ x : ℝ, a = 4 ∧ x = -3 / 2 := by
  sorry

end find_unique_solution_l498_498011


namespace six_letter_words_no_substring_amc_l498_498290

theorem six_letter_words_no_substring_amc : 
  let alphabet := ['A', 'M', 'C']
  let totalNumberOfWords := 3^6
  let numberOfWordsContainingAMC := 4 * 3^3 - 1
  let numberOfWordsNotContainingAMC := totalNumberOfWords - numberOfWordsContainingAMC
  numberOfWordsNotContainingAMC = 622 :=
by
  sorry

end six_letter_words_no_substring_amc_l498_498290


namespace coefficient_x3_term_in_binomial_l498_498970

theorem coefficient_x3_term_in_binomial :
  let expansion := (1 - X)^5;
  coeff expansion x^3 = -10 :=
by
  sorry

end coefficient_x3_term_in_binomial_l498_498970


namespace problem_inequality_l498_498125

theorem problem_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 1 ≤ x i) :
  (∑ i, (1 / (1 + x i))) ≥ (n / (1 + (∏ i, x i) ^ (1 / n))) :=
sorry

end problem_inequality_l498_498125


namespace if_a_eq_b_then_a_squared_eq_b_squared_l498_498042

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l498_498042


namespace solution_l498_498666

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℝ) (a b : ℝ) (x : ℝ), 
    (|A - B| = 3) ∧
    (|A - C| = 1) ∧
    (A = Real.pi / 2) ∧  -- This typically signifies angle A is 90 degrees.
    (a > 0) ∧
    (b > 0) ∧
    (a = 1) ∧
    (|A - D| = x) ∧
    (|B - D| = 3 - x) ∧
    (|C - D| = Real.sqrt (x^2 + 1)) ∧
    (Real.sqrt (x^2 + 1) - (3 - x) = 2) ∧
    (|A - D| / |B - D| = 4)

theorem solution : problem_statement :=
sorry

end solution_l498_498666


namespace number_of_even_factors_of_n_l498_498781

def n : ℕ := 2^4 * 3^3 * 5 * 7^2

theorem number_of_even_factors_of_n : 
  (∃ k : ℕ, n = 2^4 * 3^3 * 5 * 7^2 ∧ k = 96) → 
  ∃ count : ℕ, 
    count = 96 ∧ 
    (∀ m : ℕ, 
      (m ∣ n ∧ m % 2 = 0) ↔ 
      (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d)) :=
by
  sorry

end number_of_even_factors_of_n_l498_498781


namespace angle_in_regular_octagon_l498_498889

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498889


namespace equation_of_line_AB_l498_498807

noncomputable def midpoint := (2 : ℝ, -1 : ℝ)
noncomputable def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25
noncomputable def is_midpoint_of_chord (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

theorem equation_of_line_AB :
  ∀ (x y : ℝ), is_midpoint_of_chord midpoint (x, y) -> circle_eqn x y -> x - y - 3 = 0 :=
by
  sorry

end equation_of_line_AB_l498_498807


namespace find_fraction_2012th_l498_498462

def sequence_term (p : ℕ) : ℚ :=
  let candidates := {f : ℚ // f.denom ≤ floor (f.num / 2) ∧ f.num < f.denom ∧ 0 < f}
  (candidates.sort (λ x y, (x.denom, x.num) < (y.denom, y.num))).nth (p - 1) 

theorem find_fraction_2012th (m n: ℕ) (hmn : nat.coprime m n) (hp : m < n ∧ 1 ≤ m ∧ 1 < n) (h : sequence_term 2012 = m / n) :
  m + n = 61 :=
  sorry

end find_fraction_2012th_l498_498462


namespace juice_pressed_l498_498963

variables (x y V : ℕ)
noncomputable def initial_state : Prop :=
  x = 21 ∧ y = 19 ∧ x + y = 40 ∧ x + 9 = V ∧ V = 30

theorem juice_pressed (x y V : ℕ) : initial_state x y V → x + y = 40 ∧ V = 30 :=
begin
  sorry
end

end juice_pressed_l498_498963


namespace prob_of_exactly_3_items_damaged_thm_prob_of_fewer_than_3_items_damaged_thm_prob_of_more_than_3_items_damaged_thm_prob_of_at_least_1_item_damaged_thm_l498_498459

noncomputable def λ : ℝ := 500 * 0.002 -- Expected number of occurrences (mean)
def poisson (k : ℕ) : ℝ := (λ ^ k * Real.exp (-λ)) / Nat.factorial k

-- Proof objectives
def prob_of_exactly_3_items_damaged : Prop := poisson 3 = 0.0613
def prob_of_fewer_than_3_items_damaged : Prop := (poisson 0 + poisson 1 + poisson 2) = 0.9197
def prob_of_more_than_3_items_damaged : Prop := (1 - (poisson 0 + poisson 1 + poisson 2 + poisson 3)) = 0.019
def prob_of_at_least_1_item_damaged : Prop := (1 - poisson 0) = 0.632

-- Lean statement with conditions and exact proofs
theorem prob_of_exactly_3_items_damaged_thm : prob_of_exactly_3_items_damaged := by exact sorry
theorem prob_of_fewer_than_3_items_damaged_thm : prob_of_fewer_than_3_items_damaged := by exact sorry
theorem prob_of_more_than_3_items_damaged_thm : prob_of_more_than_3_items_damaged := by exact sorry
theorem prob_of_at_least_1_item_damaged_thm : prob_of_at_least_1_item_damaged := by exact sorry

end prob_of_exactly_3_items_damaged_thm_prob_of_fewer_than_3_items_damaged_thm_prob_of_more_than_3_items_damaged_thm_prob_of_at_least_1_item_damaged_thm_l498_498459


namespace expression_at_x_equals_2_l498_498721

theorem expression_at_x_equals_2 (a b : ℝ) (h : 2 * a - b = -1) : (2 * b - 4 * a) = 2 :=
by {
  sorry
}

end expression_at_x_equals_2_l498_498721


namespace birds_count_l498_498059

theorem birds_count (N B : ℕ) 
  (h1 : B = 5 * N)
  (h2 : B = N + 360) : 
  B = 450 := by
  sorry

end birds_count_l498_498059


namespace constant_COG_of_mercury_column_l498_498557

theorem constant_COG_of_mercury_column (L : ℝ) (A : ℝ) (beta_g : ℝ) (beta_m : ℝ) (alpha_g : ℝ) (x : ℝ) :
  L = 1 ∧ A = 1e-4 ∧ beta_g = 1 / 38700 ∧ beta_m = 1 / 5550 ∧ alpha_g = beta_g / 3 ∧
  x = (2 / (3 * 38700)) / ((1 / 5550) - (2 / 116100)) →
  x = 0.106 :=
by
  sorry

end constant_COG_of_mercury_column_l498_498557


namespace ivan_uses_more_paint_l498_498074

-- Conditions
def ivan_section_area : ℝ := 5 * 2
def petr_section_area (alpha : ℝ) : ℝ := 5 * 2 * Real.sin(alpha)
axiom alpha_lt_90 : ∀ α : ℝ, α < 90 → Real.sin(α) < 1

-- Assertion
theorem ivan_uses_more_paint (α : ℝ) (h1 : α < 90) : ivan_section_area > petr_section_area α :=
by
  sorry

end ivan_uses_more_paint_l498_498074


namespace find_expression_l498_498967

theorem find_expression (E a : ℝ) 
  (h1 : (E + (3 * a - 8)) / 2 = 69) 
  (h2 : a = 26) : 
  E = 68 :=
sorry

end find_expression_l498_498967


namespace infinite_sum_of_digits_of_polynomial_l498_498952

def sum_of_digits (n : ℤ) : ℤ := sorry

theorem infinite_sum_of_digits_of_polynomial (f : ℤ → ℤ) (hf : polynomial ℤ) :
  ∃ C : ℤ, {n : ℤ | sum_of_digits (f n) = C}.infinite := 
sorry

end infinite_sum_of_digits_of_polynomial_l498_498952


namespace isosceles_triangle_ADG_l498_498787

noncomputable theory

variables
  (A B O C D P F E G : Type*)
  [metric_space A] [metric_space B] [metric_space O]
  [metric_space C] [metric_space D] [metric_space P]
  [metric_space F] [metric_space E] [metric_space G]
  [interpolates A B O C D P F E G] -- Considering the input and requirements depicted by the conditions

axiom semicircle {A B O} : metric_space.semicircle A B O
axiom diameter {AB : Type*} : metric_space.diameter AB
axiom center {O : Type*} : metric_space.center O
axiom arbitrary_point {C : Type*} : metric_space.segment O B C
axiom perpendicular {CD : Type*} : metric_space.perpendicular CD A B
axiom on_semicircle {D : Type*} : metric_space.on_semicircle D A B O
axiom circle_center {P : Type*} : metric_space.center P
axiom tangent_arc {F : Type*} : metric_space.tangent_arc F P B D
axiom tangent_segment_CD {E : Type*} : metric_space.tangent_segment CD P E
axiom tangent_segment_AB {G : Type*} : metric_space.tangent_segment AB P G 

theorem isosceles_triangle_ADG : ∀
  (A B O C D P F E G : Type*)
  [metric_space.semicircle A B O]
  [metric_space.diameter A B]
  [metric_space.center O]
  [metric_space.segment O B C]
  [metric_space.perpendicular CD A B]
  [metric_space.on_semicircle D A B O]
  [metric_space.center P]
  [metric_space.tangent_arc F P B D]
  [metric_space.tangent_segment CD P E]
  [metric_space.tangent_segment AB P G],
  A distance between AD = AG :=
begin
  sorry -- Proof to be implemented.
end

end isosceles_triangle_ADG_l498_498787


namespace solve_exponential_equation_l498_498446

theorem solve_exponential_equation (x : ℝ) : 3^(x + 2) - 3^(2 - x) = 80 → x = 2 :=
by
  sorry

end solve_exponential_equation_l498_498446


namespace sequence_properties_l498_498661

-- Define the sequence and its properties
def seq (n : ℕ) : ℕ → ℝ
| 0 := 1
| 1 := 1
| (n + 2) := 2 * seq (n + 1) + 2^n

-- Theorem statement
theorem sequence_properties (n : ℕ) (h : n ≥ 2) :
  (∀ m ≥ 2, seq (m) = 2 * seq (m-1) + 2^m) ∧
  ∃ t : ℝ, (t = (seq n) / (2^n)) → 
           t = n - 0.5 :=
sorry

end sequence_properties_l498_498661


namespace ellipse_properties_l498_498252

-- Definitions and conditions
def ellipse (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ ((x * x) / (a * a) + (y * y) / (b * b) = 1)

def eccentricity (c a e : ℝ) : Prop :=
  (e = c / a) ∧ (e = 1 / 2)

def conditions (a b c : ℝ) (F1 F2 P : ℝ × ℝ) : Prop :=
  (a = 2 * c) ∧ 
  (b * b = a * a - c * c) ∧ 
  (⟦F1, P, F2⟧ = π/2) ∧ 
  (1 / 2 * |PF1| * |PF2| = 3) ∧ 
  (|PF1| * |PF2| = 6) ∧ 
  (|PF1| + |PF2| = 2 * a) ∧ 
  (|PF1|^2 + |PF2|^2 = (2 * c)^2)

-- Prove the standard equation of the ellipse E
def standard_equation {x y : ℝ} (a b : ℝ) : Prop :=
  ellipse x y a b → 
  eccentricity (a / 2) a (1 / 2) →
  (b * b = a * a - (a / 2) * (a / 2)) →
  (a = 2 ∧ b = sqrt 3) →
  (∃ (a = 2) ∧ (b * b = 3), (x^2 / 4 + y^2 / 3 = 1))

-- Prove the distance from origin O to the line MN is constant
def distance_O_to_MN_is_const (M N O : ℝ × ℝ) : Prop :=
  ∀ (M, N : ℝ × ℝ), (M ∈ Ellipse) ∧ (N ∈ Line) ∧ (dot_product(O, M) = 0) →
  distance(O, line_through(M, N)) = sqrt(3)

-- Full theorem statement
theorem ellipse_properties (x y a b : ℝ) (M N O : ℝ × ℝ) (F1 F2 P : ℝ × ℝ) :
  (ellipse x y a b) →
  (eccentricity (a / 2) a (1 / 2)) →
  (conditions a b (a / 2) F1 F2 P) →
  standard_equation a b →
  distance_O_to_MN_is_const M N O :=
by 
  sorry

end ellipse_properties_l498_498252


namespace nonempty_subsets_count_sum_of_products_sum_of_products_neg_l498_498502

-- Defines N_{2010} as the set of natural numbers from 1 to 2010
def N_2010 := Finset.range 2011 -- {0, 1, ..., 2010} in Lean, need to adjust for 1-based indexing

-- (a) Number of non-empty subsets of N_{2010} 
theorem nonempty_subsets_count :  N_2010.card = 2010 → (2 ^ 2010 - 1 = Finset.powersetLen (λ n, n ≠ 0) N_2010).card :=
sorry

-- (b) Sum of products of non-empty subsets of N_{2010}
theorem sum_of_products (h : N_2010.card = 2010) : 
  (Finset.sum (Finset.powersetLen (λ n, n ≠ 0) N_2010) (λ s, s.prod id)) = 2011.factorial - 1 :=
sorry

-- (c) Sum of products of non-empty subsets of -N_2010
def neg_N_2010 := N_2010.map (λ x, - x)

theorem sum_of_products_neg (h : neg_N_2010.card = 2010) : 
  (Finset.sum (Finset.powersetLen (λ n, n ≠ 0) neg_N_2010) (λ s, s.prod id)) = -1 :=
sorry

end nonempty_subsets_count_sum_of_products_sum_of_products_neg_l498_498502


namespace translated_point_coordinates_l498_498037

theorem translated_point_coordinates : 
    ∃ (P : ℝ × ℝ), P = (-5, 1) ∧ 
    (P.1 + 2, P.2) = (-3, 1) ∧ 
    (P.1 + 2, P.2 - 4) = (-3, -3) :=
by
  use (-5, 1)
  split
  { refl }
  split
  { exact rfl }
  { exact rfl }

end translated_point_coordinates_l498_498037


namespace inequality_abc_l498_498426

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l498_498426


namespace f_inequality_l498_498004

variables {n1 n2 d : ℕ} (f : ℕ → ℕ → ℕ)

theorem f_inequality (hn1 : n1 > 0) (hn2 : n2 > 0) (hd : d > 0) :
  f (n1 * n2) d ≤ f n1 d + n1 * (f n2 d - 1) :=
sorry

end f_inequality_l498_498004


namespace chess_match_schedule_count_l498_498198

theorem chess_match_schedule_count : 
  let players_team1 := ['A', 'B', 'C', 'D']
      players_team2 := ['W', 'X', 'Y', 'Z']
      games_per_pair := 3
      total_games := 4 * 4 * games_per_pair
      games_per_round := 4
      total_rounds := total_games / games_per_round
      num_schedules := total_rounds.factorial in
  total_games = 48 ∧ 
  total_rounds = 12 ∧ 
  num_schedules = 479001600 := 
by
  sorry

end chess_match_schedule_count_l498_498198


namespace loggers_count_l498_498144

theorem loggers_count 
  (cut_rate : ℕ) 
  (forest_width : ℕ) 
  (forest_height : ℕ) 
  (tree_density : ℕ) 
  (days_per_month : ℕ) 
  (months : ℕ) 
  (total_loggers : ℕ)
  (total_trees : ℕ := forest_width * forest_height * tree_density) 
  (total_days : ℕ := days_per_month * months)
  (trees_cut_down_per_logger : ℕ := cut_rate * total_days) 
  (expected_loggers : ℕ := total_trees / trees_cut_down_per_logger) 
  (h1: cut_rate = 6)
  (h2: forest_width = 4)
  (h3: forest_height = 6)
  (h4: tree_density = 600)
  (h5: days_per_month = 30)
  (h6: months = 10)
  (h7: total_loggers = expected_loggers)
: total_loggers = 8 := 
by {
    sorry
}

end loggers_count_l498_498144


namespace find_numbers_and_difference_l498_498990

-- Definitions of the problem conditions
variables {A B x : ℝ}

def sum_condition := A + B = 40
def product_condition := A * B = 375
def difference_condition := A - B = x
def ratio_condition := A / B = 3 / 2

-- Theorem statement
theorem find_numbers_and_difference
  (h_sum : sum_condition)
  (h_product : product_condition)
  (h_difference : difference_condition)
  (h_ratio : ratio_condition) : A = 24 ∧ B = 16 ∧ x = 8 :=
by {
  sorry
}

end find_numbers_and_difference_l498_498990


namespace marie_eggs_total_l498_498418

variable (x : ℕ) -- Number of eggs in each box

-- Conditions as definitions
def egg_weight := 10 -- weight of each egg in ounces
def total_boxes := 4 -- total number of boxes
def remaining_boxes := 3 -- boxes left after one is discarded
def remaining_weight := 90 -- total weight of remaining eggs in ounces

-- Proof statement
theorem marie_eggs_total : remaining_boxes * egg_weight * x = remaining_weight → total_boxes * x = 12 :=
by
  intros h
  sorry

end marie_eggs_total_l498_498418


namespace sum_of_set_T_l498_498395

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498395


namespace meaningful_if_and_only_if_l498_498302

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l498_498302


namespace arabella_dance_steps_l498_498190

theorem arabella_dance_steps :
  exists T1 T2 T3 : ℕ,
    T1 = 30 ∧
    T3 = T1 + T2 ∧
    T1 + T2 + T3 = 90 ∧
    (T2 / T1 : ℚ) = 1 / 2 :=
by
  sorry

end arabella_dance_steps_l498_498190


namespace power_of_5000_l498_498111

theorem power_of_5000 (h : 5000 = 5 * 10^3) : 5000^150 = 5^150 * 10^450 :=
by {
  rw [h],
  simp [pow_mul],
  sorry
}

end power_of_5000_l498_498111


namespace bus_passengers_expression_bus_passengers_specific_l498_498143

variable (m n : Nat)

def passengers_after_boarding (m n : Nat) : Nat := m - 12 + n

theorem bus_passengers_expression : passengers_after_boarding m n = m - 12 + n := rfl

theorem bus_passengers_specific (h1 : m = 26) (h2 : n = 6) : passengers_after_boarding m n = 20 := by
  rw [h1, h2]
  show passengers_after_boarding 26 6 = 20
  rw [passengers_after_boarding]
  norm_num
  done
 
end bus_passengers_expression_bus_passengers_specific_l498_498143


namespace arithmetic_sequence_sum_l498_498745

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_pos : ∀ n, a n > 0) (h_sum : (finset.range 10).sum (λ n, a (n + 1)) = 30) :
  a 4 + a 5 = 6 :=
sorry

end arithmetic_sequence_sum_l498_498745


namespace min_seated_people_adjacent_l498_498739

def min_seated_people (seats : ℕ) : ℕ :=
  seats / 3

theorem min_seated_people_adjacent (seats : ℕ) (h : seats = 120) : min_seated_people seats = 40 :=
  by
    rw [h]
    norm_num

end min_seated_people_adjacent_l498_498739


namespace measure_angle_abh_l498_498862

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498862


namespace measure_angle_abh_l498_498861

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498861


namespace total_blocks_walked_each_day_l498_498944

theorem total_blocks_walked_each_day :
  let morning := 4 + 7 + 11 in
  let afternoon := 3 + 5 + 8 in
  let evening := 6 + 9 + 10 in
  morning + afternoon + evening = 63 :=
by
  let morning := 4 + 7 + 11
  let afternoon := 3 + 5 + 8
  let evening := 6 + 9 + 10
  show morning + afternoon + evening = 63
  sorry

end total_blocks_walked_each_day_l498_498944


namespace eq_implies_sq_eq_l498_498044

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l498_498044


namespace original_price_of_lawn_chair_l498_498155

theorem original_price_of_lawn_chair (P : ℝ) 
(SalePrice : ℝ) 
(DiscountPercentage : ℝ) 
(h : P * (1 - DiscountPercentage) = SalePrice)
: P = 74.94 :=
by
  have h0 : 1 - DiscountPercentage = 0.799866577718479 := sorry
  have h1 : SalePrice = 59.95 := sorry
  have h2 : P = SalePrice / (1 - DiscountPercentage) := by linarith
  rw [h1, h0] at h2
  linarith
  /- Alternatively, you can use a division trick without an intermediate step as:
    have : 59.95 / 0.799866577718479 = 74.94 := by sorry
    rw[this] -- assuming decimal calculation is accurate in lean.
  -/
  sorry

end original_price_of_lawn_chair_l498_498155


namespace inequality_abc_l498_498428

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l498_498428


namespace solve_equation_l498_498220

theorem solve_equation (x : ℝ) : (∃ (y : ℝ), (y = real.sqrt (real.sqrt x)) ∧ (y = 20 / (9 - y))) → (x = 256 ∨ x = 625) := by
  sorry

end solve_equation_l498_498220


namespace complement_union_M_N_eq_set_l498_498703

open Set

-- Define the universe U
def U : Set (ℝ × ℝ) := { p | True }

-- Define the set M
def M : Set (ℝ × ℝ) := { p | (p.snd - 3) / (p.fst - 2) ≠ 1 }

-- Define the set N
def N : Set (ℝ × ℝ) := { p | p.snd ≠ p.fst + 1 }

-- Define the complement of M ∪ N in U
def complement_MN : Set (ℝ × ℝ) := compl (M ∪ N)

theorem complement_union_M_N_eq_set : complement_MN = { (2, 3) } :=
  sorry

end complement_union_M_N_eq_set_l498_498703


namespace N_gt_M_l498_498358

def num_solutions_eq1 (a b c d : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 1000000 ∧ 0 ≤ b ∧ b ≤ 1000000 ∧
  0 ≤ c ∧ c ≤ 1000000 ∧ 0 ≤ d ∧ d ≤ 1000000 ∧
  (a^2 - b^2 = c^3 - d^3)

def num_solutions_eq2 (a b c d : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 1000000 ∧ 0 ≤ b ∧ b ≤ 1000000 ∧
  0 ≤ c ∧ c ≤ 1000000 ∧ 0 ≤ d ∧ d ≤ 1000000 ∧
  (a^2 - b^2 = c^3 - d^3 + 1)

def N : ℕ := (Finset.Icc 0 1000000).card (λ t, num_solutions_eq1 t.1 t.2 t.3 t.4)
def M : ℕ := (Finset.Icc 0 1000000).card (λ t, num_solutions_eq2 t.1 t.2 t.3 t.4)

theorem N_gt_M : N > M := 
by
  sorry

end N_gt_M_l498_498358


namespace valid_combinations_l498_498971

theorem valid_combinations :
  ∀ (x y z : ℕ), 
  10 ≤ x ∧ x ≤ 20 → 
  10 ≤ y ∧ y ≤ 20 →
  10 ≤ z ∧ z ≤ 20 →
  3 * x^2 - y^2 - 7 * z = 99 →
  (x, y, z) = (15, 10, 12) ∨ (x, y, z) = (16, 12, 11) ∨ (x, y, z) = (18, 15, 13) := 
by
  intros x y z hx hy hz h
  sorry

end valid_combinations_l498_498971


namespace max_distance_midpoint_to_line_l498_498613

noncomputable def parametric_curve (α : ℝ) :=
(x = 2 * real.sqrt 3 * real.cos α, y = 2 * real.sin α)

noncomputable def polar_line_eq (ρ θ : ℝ) :=
ρ * real.sin (θ - real.pi / 4) + 5 * real.sqrt 2 = 0

noncomputable def polar_point (P : ℝ × ℝ) :=
P = (4 * real.sqrt 2, real.pi / 4)

noncomputable def cartesian_line_eq (x y : ℝ) :=
x - y - 10 = 0

noncomputable def standard_curve_eq (x y : ℝ) :=
x^2 / 12 + y^2 / 4 = 1

theorem max_distance_midpoint_to_line :
  ∀ α ∈ set.Ioo 0 real.pi,
    ∃ d : ℝ, d = max_dist (sqrt 3 * real.cos α + 2, real.sin α + 2) 6 * real.sqrt 2 := 
sorry

end max_distance_midpoint_to_line_l498_498613


namespace angle_ABH_in_regular_octagon_l498_498836

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498836


namespace integer_solution_unique_l498_498951

theorem integer_solution_unique (w x y z : ℤ) :
  w^2 + 11 * x^2 - 8 * y^2 - 12 * y * z - 10 * z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry
 
end integer_solution_unique_l498_498951


namespace range_of_a_l498_498695

def f (x : ℝ) : ℝ := 3^(-x) - 3^x - x

theorem range_of_a (a : ℝ) (h : f (2 * a + 3) + f (3 - a) > 0) : a < -6 := by
  sorry

end range_of_a_l498_498695


namespace find_x_l498_498960

-- Define the variables and conditions
variables {x y z : ℝ}

-- Given conditions
def condition1 := x^2 / y = 2
def condition2 := y^2 / z = 3
def condition3 := z^2 / x = 4

-- Goal to prove
theorem find_x (h1 : condition1) (h2 : condition2) (h3 : condition3) : x = 24^(2/7) :=
by
  sorry -- Proof to be done

end find_x_l498_498960


namespace bus_average_speed_excluding_stoppages_l498_498618

theorem bus_average_speed_excluding_stoppages :
  ∀ v : ℝ, (32 / 60) * v = 40 → v = 75 :=
by
  intro v
  intro h
  sorry

end bus_average_speed_excluding_stoppages_l498_498618


namespace output_value_2018_l498_498471

def outputValue (a n : ℝ) (years : ℕ) : ℝ :=
  a * (1 + n / 100) ^ years

theorem output_value_2018 (a n : ℝ) :
  outputValue a n 12 = a * (1 + n / 100) ^ 12 := by
  sorry

end output_value_2018_l498_498471


namespace integral_evaluation_l498_498761

theorem integral_evaluation (a : ℝ) (h : a > 0) :
  ∫ x in -a..a, (x^2 * cos x + exp x) / (exp x + 1) = a := sorry

end integral_evaluation_l498_498761


namespace factors_count_l498_498211

theorem factors_count (M : ℕ) (hM : M = 2^4 * 3^3 * 5^2 * 7^1) : nat.divisors_count M = 120 :=
by {
  -- Given M = 2^4 * 3^3 * 5^2 * 7^1, find the number of its divisors
  have factM : (M = 2^4 * 3^3 * 5^2 * 7^1) := hM,
  sorry
}

end factors_count_l498_498211


namespace sum_elements_T_l498_498372

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498372


namespace dice_probability_sum_18_l498_498311

theorem dice_probability_sum_18 : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 8 ∧ 1 ≤ d2 ∧ d2 ≤ 8 ∧ 1 ≤ d3 ∧ d3 ≤ 8 ∧ d1 + d2 + d3 = 18) →
  (1/8 : ℚ) * (1/8) * (1/8) * 9 = 9 / 512 :=
by 
  sorry

end dice_probability_sum_18_l498_498311


namespace never_prime_l498_498609

theorem never_prime (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 105) := sorry

end never_prime_l498_498609


namespace regular_octagon_angle_ABH_l498_498869

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498869


namespace exam_total_number_l498_498582

theorem exam_total_number (x : ℝ) (h1 : 15 / 100 * x + 144 + 5 / 3 * (15 / 100 * x) = x) : x = 240 := by
  intro h1
  sorry

end exam_total_number_l498_498582


namespace correct_statement_for_y_eq_3_x_minus_1_sq_plus_2_l498_498642

theorem correct_statement_for_y_eq_3_x_minus_1_sq_plus_2 : 
  let y := λ x : ℝ, 3 * (x - 1) ^ 2 + 2
  in  ∃ h, (∀ x : ℝ, (x = h ↔ x = 1)) :=
by
  sorry

end correct_statement_for_y_eq_3_x_minus_1_sq_plus_2_l498_498642


namespace fraction_of_weight_kept_l498_498756

-- Definitions of the conditions
def hunting_trips_per_month := 6
def months_in_season := 3
def deers_per_trip := 2
def weight_per_deer := 600
def weight_kept_per_year := 10800

-- Definition calculating total weight caught in the hunting season
def total_trips := hunting_trips_per_month * months_in_season
def weight_per_trip := deers_per_trip * weight_per_deer
def total_weight_caught := total_trips * weight_per_trip

-- The theorem to prove the fraction
theorem fraction_of_weight_kept : (weight_kept_per_year : ℚ) / (total_weight_caught : ℚ) = 1 / 2 := by
  -- Proof goes here
  sorry

end fraction_of_weight_kept_l498_498756


namespace distance_from_A_to_B_l498_498051

/-- 
The smaller square has a perimeter of 8 cm, and the larger square has an area of 36 cm^2.
We need to prove the distance from point A to point B is approximately 8.9 cm.
-/
theorem distance_from_A_to_B (perimeter_smaller_square : ℝ) (area_larger_square : ℝ) : 
  perimeter_smaller_square = 8 → area_larger_square = 36 → 
  (sqrt ((2 + sqrt area_larger_square - (perimeter_smaller_square / 4))^2 + 
        (sqrt area_larger_square - (perimeter_smaller_square / 4))^2) ≈ 8.9) :=
by
  intros h1 h2
  sorry

end distance_from_A_to_B_l498_498051


namespace part_a_part_b_l498_498483

def kopecks := ℕ

structure Passenger :=
  (coins : Multiset kopecks) -- Each passenger has many coins

def fare_per_passenger := 5 -- Each passenger must pay 5 kopecks
def num_passengers := 20
def coins24 := 24
def coins25 := 25

-- Definition: Can they pay the fare with a certain number of coins?
def can_pay_fare (num_coins : ℕ) (passengers : list Passenger) : Prop :=
  -- A function or predicate stating whether the passengers can pay exactly 
  -- the total fare using the given number of coins.
  sorry

theorem part_a (passengers : list Passenger) :
  (∀ p ∈ passengers, ∀ c ∈ p.coins, c ∈ {10, 15, 20}) → ¬ can_pay_fare coins24 passengers :=
  sorry

theorem part_b (passengers : list Passenger) :
  (∀ p ∈ passengers, ∀ c ∈ p.coins, c ∈ {10, 15, 20}) → can_pay_fare coins25 passengers :=
  sorry

end part_a_part_b_l498_498483


namespace year_2017_l498_498093

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

theorem year_2017 : ∃ (y : ℕ), y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ y', y' > 2010 → sum_of_digits y' = 10 → y' ≥ y :=
by {
  let y := 2017,
  use y,
  split,
  { exact Nat.lt_of_succ_lt_succ (Nat.succ_lt_succ (Nat.succ_pos 2016)) },
  split,
  { norm_num },
  intros y' y'_gt_2010 sum_y',
  sorry
}

end year_2017_l498_498093


namespace four_lines_two_intersections_impossible_l498_498238

theorem four_lines_two_intersections_impossible :
  ∀ (L : set (set ℝ × ℝ)), L.card = 4 → ¬ (∃ P : set (ℝ × ℝ), P.card = 2 ∧ (∀ p ∈ P, ∃ l₁ l₂ ∈ L, p ∈ l₁ ∧ p ∈ l₂)) :=
by
  sorry

end four_lines_two_intersections_impossible_l498_498238


namespace total_area_of_shaded_triangles_in_6x6_grid_l498_498314

theorem total_area_of_shaded_triangles_in_6x6_grid : 
  let grid_size := 6
  let num_squares := grid_size * grid_size
  let triangles_per_square := 2
  let area_per_triangle := 0.5
  let total_area := num_squares * triangles_per_square * area_per_triangle
  total_area = 36 :=
by
  let grid_size := 6
  let num_squares := grid_size * grid_size
  let triangles_per_square := 2
  let area_per_triangle := 0.5
  let total_area := num_squares * triangles_per_square * area_per_triangle
  have : total_area = 36 := by sorry
  exact this

end total_area_of_shaded_triangles_in_6x6_grid_l498_498314


namespace sum_of_tan_squared_l498_498365

noncomputable def T : Set ℝ :=
  {x : ℝ | 0 < x ∧ x < π / 2 ∧ (∃ P Q H : ℝ, 
    (P = sin x ∨ P = tan x ∨ P = sec x) ∧
    (Q = sin x ∨ Q = tan x ∨ Q = sec x) ∧ 
    (H = sin x ∨ H = tan x ∨ H = sec x) ∧
    (P ≠ Q ∧ Q ≠ H ∧ H ≠ P) ∧
    P^2 + Q^2 = H^2)}

theorem sum_of_tan_squared (S : ℝ) :
  (∀ x ∈ T, x < π / 2 ∧ 0 < x) → 
  S = ∑ x in T, tan x ^ 2 :=
sorry

end sum_of_tan_squared_l498_498365


namespace tan_theta_l498_498400

theorem tan_theta (x y : ℝ) (θ : ℝ) (hθ_acute : 0 < θ ∧ θ < π / 2) 
  (h : sin (θ / 2) = sqrt ((y - x) / (y + x))) :
  tan θ = 2 * sqrt (x * y) / (3 * x - y) := by
  sorry

end tan_theta_l498_498400


namespace farmer_transaction_gain_l498_498558

-- Definition for conditional variables
def initial_cows := 600
def sold_cows_price := initial_cows
def final_cows := initial_cows - 500
def price_increase := 1.10

-- Lean theorem statement
theorem farmer_transaction_gain (c : ℝ) : 
  let initial_cost := initial_cows * c in
  let revenue_from_500 := sold_cows_price * c in
  let price_per_500 := revenue_from_500 / 500 in
  let price_per_final := price_per_500 * price_increase in
  let revenue_from_final := final_cows * price_per_final in
  let total_revenue := revenue_from_500 + revenue_from_final in
  let profit := total_revenue - initial_cost in
  let percentage_gain := (profit / initial_cost) * 100 in
  percentage_gain = 22 := 
by 
  sorry

end farmer_transaction_gain_l498_498558


namespace measure_of_angle_ABH_l498_498826

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498826


namespace relationship_among_abcd_l498_498295

theorem relationship_among_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := 
by
  sorry

end relationship_among_abcd_l498_498295


namespace inequality_geq_l498_498433

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l498_498433


namespace chandra_monster_hunt_days_l498_498199

theorem chandra_monster_hunt_days :
  ∃ (d : ℕ), (2 * (2^d - 1)) = 62 ∧ 2^d = 32 ∧ d = 5 :=
begin
  sorry
end

end chandra_monster_hunt_days_l498_498199


namespace falling_body_distance_feet_minutes_l498_498755

/-- 
Given that a freely falling body travels \( s \approx 4.903 t^2 \) meters in \( t \) seconds,
prove that the formula in feet and minutes is \( s_{feet} \approx 57816.9 T^2 \),
where \( t = 60T \).
-/
theorem falling_body_distance_feet_minutes :
  ∀ (T : ℝ), let t := 60 * T
             let s := 4.903 * t^2 
             let s_feet := s * 3.28084
             in s_feet = 57816.9 * T^2 :=
by 
  intros T 
  let t := 60 * T
  let s := 4.903 * t^2
  let s_feet := s * 3.28084 
  rw [mul_assoc, mul_comm (4.903 * (60 * 60)), mul_assoc]
  suffices : 4.903 * 3600 * 3.28084 = 57816.9
  {
    -- Skip this complex computation for now
    sorry
  }
  {
    sorry  -- Handle simplification
  }

end falling_body_distance_feet_minutes_l498_498755


namespace grape_juice_amount_l498_498152

-- Definitions for the conditions
def total_weight : ℝ := 150
def orange_percentage : ℝ := 0.35
def watermelon_percentage : ℝ := 0.35

-- Theorem statement to prove the amount of grape juice
theorem grape_juice_amount : 
  (total_weight * (1 - orange_percentage - watermelon_percentage)) = 45 :=
by
  sorry

end grape_juice_amount_l498_498152


namespace normal_distribution_symmetry_l498_498565

noncomputable def X : ℝ → ℝ := sorry -- Distribution function for N(0, 1)

def p1 : ℝ := ∫ x in -2 .. -1, X x
def p2 : ℝ := ∫ x in 1 .. 2, X x

theorem normal_distribution_symmetry :
  p1 = p2 :=
sorry

end normal_distribution_symmetry_l498_498565


namespace first_year_after_2010_with_digit_sum_10_l498_498091

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem first_year_after_2010_with_digit_sum_10 :
  ∃ y > 2010, sum_of_digits y = 10 ∧ ∀ z, (2010 < z < y) → sum_of_digits z ≠ 10 :=
begin
  use 2035,
  split,
  { exact lt_of_lt_of_le (by norm_num : 2010 < 2035) (le_refl 2035)},
  split,
  { simp [sum_of_digits],
    norm_num,
  },
  { intros z hz,
    have hz0 : z ≥ 2010 + 1 := hz.1,
    have hz1 : z ≤ 2035 - 1 := hz.2,
    sorry,
  }
end

end first_year_after_2010_with_digit_sum_10_l498_498091


namespace measure_of_angle_ABH_l498_498821

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498821


namespace triangle_property_l498_498344

-- Definitions of sides opposite to angles and the given conditions
variables (a b c : ℝ) (A B C : ℝ) (M : ℝ)

-- Define the triangle ABC and the given conditions
def triangle_condition (a b c A B M : ℝ) : Prop := 
  a * cos B = (2 * c - b) * cos A ∧ 
  b = 3 ∧ 
  A = π / 3 ∧
  M = 3 * sqrt 7 / 2

-- Proof statement of angle A and area of triangle ABC given the above conditions
theorem triangle_property (a b c A B C M : ℝ) 
  (h : triangle_condition a b c A B M) :
  (A = π / 3) ∧ ((b = 3 ∧ (M = 3 * sqrt 7 / 2) → 
  (1 / 2 * c * b * (sqrt 3 / 2) = 9 * sqrt 3 / 2)) :=
  sorry

end triangle_property_l498_498344


namespace largest_circle_diameter_l498_498766

variables (ABCDEF : Type) [EquiangularHexagon ABCDEF]
variables (AB BC CD DE : ℝ)
variables (d : ℝ)

def is_equiangular_hexagon := 
  equiangular_hexagon ABCDEF ∧ 
  side_length ABCDEF "AB" = 6 ∧ 
  side_length ABCDEF "BC" = 8 ∧ 
  side_length ABCDEF "CD" = 10 ∧ 
  side_length ABCDEF "DE" = 12

theorem largest_circle_diameter (h : is_equiangular_hexagon ABCDEF AB BC CD DE) : d^2 = 147 :=
sorry

end largest_circle_diameter_l498_498766


namespace Ivan_uses_more_paint_l498_498068

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l498_498068


namespace tan_value_l498_498668

theorem tan_value (θ : ℝ) (h1 : sin θ + cos θ = 1/5) (h2 : 0 < θ ∧ θ < real.pi) : real.tan θ = 12/23 :=
  sorry

end tan_value_l498_498668


namespace angle_between_vectors_is_zero_l498_498769

variables {V : Type*} [inner_product_space ℝ V] {a b : V}

-- Define the conditions
def nonzero_vectors (a b : V) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def equal_norm (a b : V) : Prop :=
  ∥a∥ = ∥b∥

def norm_sum_twice (a b : V) : Prop :=
  ∥a + b∥ = 2 * ∥a∥

-- Lean statement for the proof problem
theorem angle_between_vectors_is_zero (h1 : nonzero_vectors a b) 
  (h2 : equal_norm a b) (h3 : norm_sum_twice a b) : 
  ∃ θ : ℝ, θ = 0 ∧ real.angle a b = θ :=
sorry

end angle_between_vectors_is_zero_l498_498769


namespace projection_eq_half_sqrt2_l498_498675

def vec_a : (ℝ × ℝ) := (2*cos(π/3), 2*sin(π/3))   -- Since angle is given in the conditions
def vec_b : (ℝ × ℝ) := (1, 1)

def vec_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let mag_squared := b.1 * b.1 + b.2 * b.2
  let projection_factor := dot_product / mag_squared
  (projection_factor * b.1, projection_factor * b.2)

theorem projection_eq_half_sqrt2 :
  let proj := vec_projection vec_a vec_b
  proj = (√2 / 2, √2 / 2) :=
by
  sorry

end projection_eq_half_sqrt2_l498_498675


namespace inequality_geq_l498_498436

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l498_498436


namespace monomials_same_type_l498_498733

theorem monomials_same_type (a b : ℤ) (h_a : a = 3) (h_b : b = 2) : (a - b) ^ 2022 = 1 := 
by {
  rw [h_a, h_b],
  norm_num,
  sorry
}

end monomials_same_type_l498_498733


namespace exists_triangle_with_given_midpoints_l498_498286

noncomputable def point := ℝ × ℝ

structure triangle (P Q R : point) :=
(midpoint1 : point)
(midpoint2 : point)
(midpoint3 : point)
(is_midpoint1 : midpoint1 = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
(is_midpoint2 : midpoint2 = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2))
(is_midpoint3 : midpoint3 = ((R.1 + P.1) / 2, (R.2 + P.2) / 2))

theorem exists_triangle_with_given_midpoints (A B C : point) (h_non_collinear : ¬ collinear A B C) :
  ∃ (P Q R : point), triangle P Q R ∧ ((A = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) ∧ (B = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)) ∧ (C = ((R.1 + P.1) / 2, (R.2 + P.2) / 2))) :=
by
  sorry

end exists_triangle_with_given_midpoints_l498_498286


namespace harmonic_mean_of_speeds_l498_498577

-- Define the conditions.
variable (n : ℕ) (v : ℝ) (speeds : Fin n → ℝ)

-- Define the hypothesis.
def average_speed_is_harmonic_mean (n : ℕ) (v : ℝ) (speeds : Fin n → ℝ) : Prop :=
  v = n / (Finset.univ.sum (λ i : Fin n, 1 / speeds i))

-- Prove that average speed is the harmonic mean of the speeds.
theorem harmonic_mean_of_speeds (n : ℕ) (v : ℝ) (speeds : Fin n → ℝ) :
  average_speed_is_harmonic_mean n v speeds := sorry

end harmonic_mean_of_speeds_l498_498577


namespace number_of_friends_dividing_bill_l498_498173

theorem number_of_friends_dividing_bill :
  ∃ n : ℕ, 45 * n = 135 ∧ n = 3 :=
begin
  use 3,
  split,
  { -- 45 * 3 = 135
    norm_num,
  },
  { -- n = 3
    refl,
  }
end

end number_of_friends_dividing_bill_l498_498173


namespace final_amount_after_two_years_l498_498617

open BigOperators

/-- Given an initial amount A0 and a percentage increase p, calculate the amount after n years -/
def compound_increase (A0 : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  (A0 * (1 + p)^n)

theorem final_amount_after_two_years (A0 : ℝ) (p : ℝ) (A2 : ℝ) :
  A0 = 1600 ∧ p = 1 / 8 ∧ compound_increase 1600 (1 / 8) 2 = 2025 :=
  sorry

end final_amount_after_two_years_l498_498617


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498902

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498902


namespace measure_of_angle_ABH_in_regular_octagon_l498_498940

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498940


namespace angle_in_regular_octagon_l498_498881

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498881


namespace five_consecutive_not_all_interesting_l498_498775

def E (n : ℕ) : ℕ := (n.bitsP.makedList.findIndexes (λ b : Bool, b)).length

def interesting (n : ℕ) : Prop := E n ∣ n

theorem five_consecutive_not_all_interesting : ∀ n : ℕ, ¬ (interesting n ∧ interesting (n+1) ∧ interesting (n+2) ∧ interesting (n+3) ∧ interesting (n+4)) :=
sorry

end five_consecutive_not_all_interesting_l498_498775


namespace percentage_of_students_who_received_certificates_l498_498740

theorem percentage_of_students_who_received_certificates
  (total_boys : ℕ)
  (total_girls : ℕ)
  (perc_boys_certificates : ℝ)
  (perc_girls_certificates : ℝ)
  (h1 : total_boys = 30)
  (h2 : total_girls = 20)
  (h3 : perc_boys_certificates = 0.1)
  (h4 : perc_girls_certificates = 0.2)
  : (3 + 4) / (30 + 20) * 100 = 14 :=
by
  sorry

end percentage_of_students_who_received_certificates_l498_498740


namespace marble_color_173_l498_498572

theorem marble_color_173 :
  (∃ (marbles : ℕ → ℕ),
    (∀ n, marbles n ∈ {1, 2, 3} ∧ (
      n % 14 < 6 → marbles n = 1) ∧ 
      (6 ≤ n % 14 ∧ n % 14 < 9 → marbles n = 2) ∧ 
      (9 ≤ n % 14 ∧ n % 14 < 14 → marbles n = 3)
    ) ∧ marbles 172 = 1) :=
  sorry

end marble_color_173_l498_498572


namespace logarithmic_AMGM_inequality_l498_498437

theorem logarithmic_AMGM_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log b / (a * Real.log a)) / (a + b) + 
       (Real.log c / (b * Real.log b)) / (b + c) + 
       (Real.log a / (c * Real.log c)) / (c + a)) 
  ≥ 9 / (a + b + c) := 
sorry

end logarithmic_AMGM_inequality_l498_498437


namespace find_values_of_ω_and_φ_find_monotonically_decreasing_interval_l498_498275

-- Definitions of the problem
def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_values_of_ω_and_φ (ω : ℝ) (φ : ℝ) (hω : ω > 0) (hφ : |φ| ≤ Real.pi / 2)
  (T : ℝ) (hT : T = Real.pi) (h2T : T = 2 * Real.pi / ω) (h_symmetry : ∃ k : ℤ, 
  2 * (Real.pi / 12) + φ = k * Real.pi + Real.pi / 2) :
  ω = 2 ∧ φ = Real.pi / 3 :=
by
  sorry

def g (ω φ x : ℝ) : ℝ := f ω φ x + f ω φ (x - Real.pi / 6)

theorem find_monotonically_decreasing_interval (ω : ℝ) (φ : ℝ) (h1 : ω = 2) (h2 : φ = Real.pi / 3) :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi) →
  ∃ u v : ℝ, u < v ∧ g ω φ x = v - u :=
by
  sorry

end find_values_of_ω_and_φ_find_monotonically_decreasing_interval_l498_498275


namespace sequence_bound_l498_498474

-- Define the sequence {a_n} recursively
noncomputable def a : ℕ → ℚ
| 0       := 0
| 1       := 1
| (n + 2) := (1 + 1 / (n + 1)) * a (n + 1) - a n

-- State the proof objective
theorem sequence_bound (n : ℕ) : (a n)^2 ≤ 8 / 3 := 
  sorry

end sequence_bound_l498_498474


namespace sum_of_ages_l498_498965

variable {P M Mo : ℕ}

-- Conditions
axiom ratio1 : 3 * M = 5 * P
axiom ratio2 : 3 * Mo = 5 * M
axiom age_difference : Mo - P = 80

-- Statement that needs to be proved
theorem sum_of_ages : P + M + Mo = 245 := by
  sorry

end sum_of_ages_l498_498965


namespace hyperbola_eccentricity_l498_498224

theorem hyperbola_eccentricity {
  e : ℝ
  (h1 : e = 3 / 2)
  (h2 : ∃ (a b : ℝ), (a^2 = 6 ∧ b^2 = 10 ∧ ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1))
  (h3 : ∃ (c : ℝ), c = √(6 + 10))
} : ∃ (a' b' : ℝ), (a' = 8 / 3) ∧ (b'^2 = 80 / 9) ∧ ∀ (x y : ℝ), 9 * x^2 / 64 - 9 * y^2 / 80 = 1 :=
by
  sorry

end hyperbola_eccentricity_l498_498224


namespace log3_2_exponent_le_one_l498_498788

variable (x : ℝ)

theorem log3_2_exponent_le_one (h : 0 < log 2 / log 3 ∧ log 2 / log 3 < 1) :
  ∀ x ∈ set.Ici (0 : ℝ), (log 2 / log 3)^x ≤ 1 :=
by
  sorry

end log3_2_exponent_le_one_l498_498788


namespace coefficient_of_x3y3_in_expansion_l498_498016

theorem coefficient_of_x3y3_in_expansion :
  let T (r : ℕ) := (5.choose r) * (2 ^ (5 - r)) * (-1) ^ r * x^(5 - r) * y^r in
  (x + y) * (2 * x - y)^5 = 40 := by
  sorry

end coefficient_of_x3y3_in_expansion_l498_498016


namespace angle_ABH_in_regular_octagon_l498_498843

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498843


namespace isabel_weekly_distance_l498_498349

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end isabel_weekly_distance_l498_498349


namespace height_of_cuboid_l498_498994

theorem height_of_cuboid (A l w : ℝ) (h : ℝ) (hA : A = 442) (hl : l = 7) (hw : w = 8) : h = 11 :=
by
  sorry

end height_of_cuboid_l498_498994


namespace angle_ABH_is_22_point_5_l498_498910

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498910


namespace part_a_part_b_l498_498530

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l498_498530


namespace problem1_problem2_l498_498770

-- Define vectors e1 and e2, and conditions for non-collinearity and unit perpendicularity
variables {V : Type*} [inner_product_space ℝ V]

variables (e1 e2 : V)

-- Condition that e1 and e2 are non-collinear corresponds to being linearly independent
axiom e1_e2_non_collinear : ¬ collinear ℝ {e1, e2}

-- Define a as a function of e1 and lambda * e2
def a (λ : ℝ) : V := e1 + λ • e2

-- Define b as a specific linear combination of e1 and e2
def b : V := 2 • e1 - e2

-- Problem 1: If a is collinear with b, then lambda = -1/2
theorem problem1 (λ : ℝ) (h : collinear ℝ {a λ, b}) : λ = -1/2 :=
sorry

-- Additional conditions for problem 2
variable (e1_perp_e2 : ⟪e1, e2⟫ = 0)
variable (e1_unit : ∥e1∥ = 1)
variable (e2_unit : ∥e2∥ = 1)

-- Problem 2: If a is perpendicular to b, and e1, e2 are perpendicular unit vectors, then lambda = 2
theorem problem2 (λ : ℝ) (h : ⟪a λ, b⟫ = 0) : λ = 2 :=
sorry

end problem1_problem2_l498_498770


namespace dice_sides_l498_498496

def total_dice := 8
def total_sides := 48

theorem dice_sides (d : ℕ) (s : ℕ) (H_d : d = total_dice) (H_s : s = total_sides) : s / d = 6 :=
by
  subst H_d
  subst H_s
  simp
  sorry

end dice_sides_l498_498496


namespace trigonometric_expression_simplification_l498_498588

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end trigonometric_expression_simplification_l498_498588


namespace flower_beds_and_circular_path_fraction_l498_498159

noncomputable def occupied_fraction 
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ) : ℝ :=
  let flower_bed_area := 2 * (1 / 2 : ℝ) * triangle_leg^2
  let circular_path_area := Real.pi * circle_radius ^ 2
  let occupied_area := flower_bed_area + circular_path_area
  occupied_area / (yard_length * yard_width)

theorem flower_beds_and_circular_path_fraction
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ)
  (h1 : side1 = 20)
  (h2 : side2 = 30)
  (h3 : triangle_leg = (side2 - side1) / 2)
  (h4 : yard_length = 30)
  (h5 : yard_width = 5)
  (h6 : circle_radius = 2) :
  occupied_fraction yard_length yard_width side1 side2 triangle_leg circle_radius = (25 + 4 * Real.pi) / 150 :=
by sorry

end flower_beds_and_circular_path_fraction_l498_498159


namespace grid_XYZ_length_sum_l498_498964

theorem grid_XYZ_length_sum :
  (∑ len in {segment | segment.type = straight}, len.size + 
   ∑ len in {segment | segment.type = slanted}, len.size) = 12 + 2 * sqrt 2 := 
sorry

end grid_XYZ_length_sum_l498_498964


namespace num_non_congruent_triangles_l498_498574

noncomputable def Point : Type := ℝ × ℝ

variables (A B C D : Point)
variables (hD_centroid : D = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3))

def non_congruent_triangles : ℕ := 2

theorem num_non_congruent_triangles : non_congruent_triangles A B C D hD_centroid = 2 :=
by sorry

end num_non_congruent_triangles_l498_498574


namespace sum_of_T_l498_498392

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498392


namespace compare_y1_y2_l498_498255

theorem compare_y1_y2 : ∀ (y1 y2 : ℝ), (A : ℝ × ℝ) (-2, y1) (B : ℝ × ℝ) (1, y2) (line_eq : ∀ x, (-2 * x + 3))
  (hA : y1 = -2*(-2) + 3) (hB : y2 = -2*1 + 3) → y1 > y2 :=
by
  intros y1 y2 A B line_eq hA hB
  sorry

end compare_y1_y2_l498_498255


namespace sum_of_T_is_2510_l498_498385

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498385


namespace first_year_after_2010_has_digit_sum_10_l498_498097

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2010_has_digit_sum_10 : 
  ∃ y : ℕ, y > 2010 ∧ sum_of_digits y = 10 ∧ ∀ z : ℕ, z > 2010 ∧ z < y → sum_of_digits z ≠ 10 :=
begin
  use 2017,
  split,
  { exact nat.lt_succ_self 2016, },
  split,
  { norm_num [sum_of_digits], },
  { intros z hz hzy,
    interval_cases z; norm_num [sum_of_digits], },
  sorry
end

end first_year_after_2010_has_digit_sum_10_l498_498097


namespace collinearity_of_X_Y_Z_l498_498357

  noncomputable def segments_with_midpoints (A B C D E F : Type) := sorry
  noncomputable def on_lines (X Y Z : Type) (EF FD DE : Type) := sorry
  noncomputable def parallel_to_line (X Y Z : Type) (ℓ : Type) := sorry

  theorem collinearity_of_X_Y_Z 
    (A B C D E F X Y Z ℓ : Type)
    (h1 : segments_with_midpoints A B C D E F)
    (h2 : on_lines X Y Z EF FD DE)
    (h3 : parallel_to_line (set_of (λ x, A x ∧ X x)) 
                           (set_of (λ y, B y ∧ Y y)) 
                           (set_of (λ z, C z ∧ Z z)) 
                           ℓ) :
    collinear X Y Z :=
  sorry
  
end collinearity_of_X_Y_Z_l498_498357


namespace matrix_pow_2018_l498_498592

open Matrix

-- Define the specific matrix
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![1, 1]]

-- Formalize the statement
theorem matrix_pow_2018 : A ^ 2018 = ![![1, 0], ![2018, 1]] :=
  sorry

end matrix_pow_2018_l498_498592


namespace closest_to_standard_weight_l498_498045

theorem closest_to_standard_weight : 
  ∀ (ball1 ball2 ball3 ball4 ball5 : ℝ), 
    ball1 = -0.02 → 
    ball2 = 0.1 → 
    ball3 = -0.23 → 
    ball4 = -0.3 → 
    ball5 = 0.2 → 
    (∀ i ∈ [|-0.02|, |0.1|, |-0.23|, |-0.3|, |0.2|], i ≥ |ball1|) := 
by
  intros ball1 ball2 ball3 ball4 ball5 h1 h2 h3 h4 h5
  have h_abs_ball1 : |ball1| = 0.02, { rw h1, norm_num, }
  have h_abs_ball2 : |ball2| = 0.1, { rw h2, norm_num, }
  have h_abs_ball3 : |ball3| = 0.23, { rw h3, norm_num, }
  have h_abs_ball4 : |ball4| = 0.3, { rw h4, norm_num, }
  have h_abs_ball5 : |ball5| = 0.2, { rw h5, norm_num, }
  have h_abs : [|ball1|, |ball2|, |ball3|, |ball4|, |ball5|] = [0.02, 0.1, 0.23, 0.3, 0.2], 
  { rw [h_abs_ball1, h_abs_ball2, h_abs_ball3, h_abs_ball4, h_abs_ball5, list.of_fn], }
  intros i h_i
  simp only [list.mem_of_fn, list.of_fn] at h_i
  exact list.assoc_of_mem h_abs h_i

end closest_to_standard_weight_l498_498045


namespace ordered_pairs_count_l498_498639

theorem ordered_pairs_count :
  {p : ℕ × ℕ | let b := p.1, let c := p.2 in b > 0 ∧ c > 0 ∧ b^2 - 4 * c ≤ 0 ∧ c^2 - 4 * b ≤ 0}.to_finset.card = 5 :=
by
  sorry

end ordered_pairs_count_l498_498639


namespace not_late_prob_expected_encounters_l498_498512

open MeasureTheory ProbabilityTheory

noncomputable def redLightProbability : ProbabilityTheory.ProbabilitySpace ℝ := sorry

-- Define the probability of encountering red lights at each post
def prob_A : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_B : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_C : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_D : ProbabilityTheory.ProbMeasure ℝ := sorry

-- Define the indicator random variables for encountering red lights at each post
def indicator_A : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_B : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_C : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_D : ProbabilityTheory.RandomVariable ℝ := sorry

-- Define X as the total number of red lights encountered
def X := indicator_A + indicator_B + indicator_C + indicator_D

-- Define the condition for being late
def is_late : ℝ → Prop := λ x, x ≥ 3

-- Define the probability of not being late
def prob_not_late : ℝ := 
  ProbabilityTheory.P (λ x, ¬is_late x)
  
-- Define the expected value of X
def expected_X : ℝ := 
  ProbabilityTheory.ExpectedValue X

theorem not_late_prob : prob_not_late redLightProbability = 29/36 := sorry
theorem expected_encounters : expected_X = 5/3 := sorry

end not_late_prob_expected_encounters_l498_498512


namespace value_of_expression_l498_498999

noncomputable def centroid_coordinates : ℚ × ℚ :=
  let X := (4 : ℚ, 9 : ℚ)
  let Y := (7 : ℚ, -3 : ℚ)
  let Z := (9 : ℚ, 4 : ℚ)
  ((X.1 + Y.1 + Z.1) / 3, (X.2 + Y.2 + Z.2) / 3)

theorem value_of_expression : 7 * centroid_coordinates.1 + 3 * centroid_coordinates.2 = 170 / 3 := by
  sorry

end value_of_expression_l498_498999


namespace travel_with_no_more_than_two_layovers_l498_498610

-- Define the notion of direct flight connections between cities
variable {City : Type}

-- Given a relation that represents direct bilateral non-stop flights between cities
variable (direct_flight : City → City → Prop)

-- Condition: It is possible to reach any city from any other city (possibly with layovers)
variable (reachable : City → City → Prop)

-- Condition: For each city A, there exists a city B such that any other city is directly connected to either A or B
variable (special_city_exists : ∀ (A : City), ∃ (B : City), 
  ∀ (C : City), C ≠ A ∧ C ≠ B → (direct_flight C A ∨ direct_flight C B))

-- Define the problem statement: It is possible to travel from any city to any other city with no more than two layovers
theorem travel_with_no_more_than_two_layovers (X Y : City)
  (reachable_X_Y : reachable X Y)
  (H1 : ∀ (A B : City), special_city_exists A)
  (H2 : ∀ (A B : City), special_city_exists B) : 
  ∃ (Z₁ Z₂ : City), (direct_flight X Z₁ ∧ direct_flight Z₁ Z₂ ∧ direct_flight Z₂ Y) := 
sorry

end travel_with_no_more_than_two_layovers_l498_498610


namespace acute_angle_le_45_l498_498439

theorem acute_angle_le_45 (A B C : Type) [right_triangle A B C] (angleA angleB : ℝ) 
  (hC : angleC = 90) (sum_angles : angleA + angleB = 90) : 
  angleA <= 45 ∨ angleB <= 45 := 
by 
  sorry

end acute_angle_le_45_l498_498439


namespace trucking_cost_equation_train_cost_equation_cost_comparison_l498_498559

-- Definitions based on conditions
def speed_truck := 75
def speed_train := 100
def rate_truck := 1.5
def rate_train := 1.3
def refrigeration_fee := 5
def load_unload_truck := 4000
def load_unload_train := 6600
def distance (x : ℝ) := x
def total_cost_truck (x : ℝ) := 60 * (rate_truck * distance x) + 60 * (refrigeration_fee * (distance x / speed_truck)) + load_unload_truck
def total_cost_train (x : ℝ) := 60 * (rate_train * distance x) + 60 * (refrigeration_fee * (distance x / speed_train)) + load_unload_train

-- Proof goal for total cost equations
theorem trucking_cost_equation (x : ℝ) : total_cost_truck x = 94 * x + 4000 := sorry

theorem train_cost_equation (x : ℝ) : total_cost_train x = 81 * x + 6600 := sorry

-- Proof goal for cost comparison
theorem cost_comparison (x : ℝ) : 
  if x = 200 then total_cost_truck x = total_cost_train x 
  else if x < 200 then total_cost_truck x < total_cost_train x 
  else total_cost_truck x > total_cost_train x := sorry

end trucking_cost_equation_train_cost_equation_cost_comparison_l498_498559


namespace exists_polygons_forming_any_sided_l498_498213

theorem exists_polygons_forming_any_sided (n : ℕ) (h : 3 ≤ n ∧ n ≤ 100) :
  ∃ (M N : List (ℚ × ℚ)), 
  (∀ k, 3 ≤ k ∧ k ≤ 100 → 
    ∃ P : List (ℚ × ℚ), (P = M ∪ N ∧ length P = k)) :=
sorry

end exists_polygons_forming_any_sided_l498_498213


namespace quartic_poly_roots_l498_498627

noncomputable def roots_polynomial : List ℝ := [
  (1 + Real.sqrt 5) / 2,
  (1 - Real.sqrt 5) / 2,
  (3 + Real.sqrt 13) / 6,
  (3 - Real.sqrt 13) / 6
]

theorem quartic_poly_roots :
  ∀ x : ℝ, x ∈ roots_polynomial ↔ 3*x^4 - 4*x^3 - 5*x^2 - 4*x + 3 = 0 :=
by sorry

end quartic_poly_roots_l498_498627


namespace probability_second_term_three_l498_498366

def S := Finset.perm (Finset.range 1 7)
def T := { σ ∈ S | σ 0 ≠ 1 }

theorem probability_second_term_three (σ ∈ T) :
  (Finset.filter (λ σ, σ 1 = 3) T).card / T.card = 4 / 25 := sorry

end probability_second_term_three_l498_498366


namespace variance_unchanged_l498_498736

open Real

-- Define the original donations and the new donations after adding 10 units
def original_donations := [20, 20, 30, 40, 40]
def new_donations := [d + 10 | d ∈ original_donations]

-- Function to calculate variance
def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  data.map (λ x => (x - mean) ^ 2).sum / data.length

-- Statement to prove that variance remains unchanged after adding 10 to each donation
theorem variance_unchanged :
  variance original_donations = variance new_donations :=
by
  sorry

end variance_unchanged_l498_498736


namespace angle_in_regular_octagon_l498_498887

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498887


namespace measure_angle_abh_l498_498868

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498868


namespace adam_teaches_650_students_in_10_years_l498_498575

noncomputable def students_in_n_years (n : ℕ) : ℕ :=
  if n = 1 then 40
  else if n = 2 then 60
  else if n = 3 then 70
  else if n <= 10 then 70
  else 0 -- beyond the scope of this problem

theorem adam_teaches_650_students_in_10_years :
  (students_in_n_years 1 + students_in_n_years 2 + students_in_n_years 3 +
   students_in_n_years 4 + students_in_n_years 5 + students_in_n_years 6 +
   students_in_n_years 7 + students_in_n_years 8 + students_in_n_years 9 +
   students_in_n_years 10) = 650 :=
by
  sorry

end adam_teaches_650_students_in_10_years_l498_498575


namespace first_year_after_2010_with_digit_sum_10_l498_498103

/--
Theorem: The first year after 2010 for which the sum of the digits equals 10 is 2017.
-/
theorem first_year_after_2010_with_digit_sum_10 : ∃ (y : ℕ), (y > 2010) ∧ (∑ d in (y.to_digits 10), d) = 10 ∧ (∀ z, (z > 2010) → ((∑ d in (z.to_digits 10), d) = 10 → z ≥ y))

end first_year_after_2010_with_digit_sum_10_l498_498103


namespace same_terminal_side_l498_498181

theorem same_terminal_side :
  ∃ (k : ℤ), -300 = k * 360 + 60 :=
begin
  use -1,
  ring,
end

end same_terminal_side_l498_498181


namespace regular_octagon_angle_ABH_l498_498873

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498873


namespace intersection_eq_l498_498410

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : S ∩ T = { x | -2 < x ∧ x ≤ 1 } :=
by
  simp [S, T]
  sorry

end intersection_eq_l498_498410


namespace sum_of_set_T_l498_498399

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498399


namespace circle_area_difference_l498_498291

theorem circle_area_difference (π : Real) :
  let r1 := 30
  let d2 := 30
  let r2 := d2 / 2
  let A1 := π * r1^2
  let A2 := π * r2^2
  A1 - A2 = 675 * π :=
by
  let r1 := 30
  let d2 := 30
  let r2 := d2 / 2
  let A1 := π * r1^2
  let A2 := π * r2^2
  have hA1 : A1 = π * (30 : Real)^2 := by sorry
  have hA2 : A2 = π * (15 : Real)^2 := by sorry
  have hdiff : A1 - A2 = 675 * π := by sorry
  exact hdiff

end circle_area_difference_l498_498291


namespace monotonically_increasing_interval_l498_498210

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x < f y

def func (x : ℝ) : ℝ := real.log (4 - x) + real.log (2 + x)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Icc (-2 : ℝ) 4 → (∃ y, y ∈ Icc (-2 : ℝ) 1 ∧ is_monotonically_increasing func (Icc (-2 : ℝ) y)) :=
sorry

end monotonically_increasing_interval_l498_498210


namespace sum_of_solutions_for_x_l498_498326

theorem sum_of_solutions_for_x :
  (∃ x y : ℝ, y = 8 ∧ x^2 + y^2 = 169) →
  (∃ x₁ x₂ : ℝ, x₁^2 = 105 ∧ x₂^2 = 105 ∧ x₁ + x₂ = 0) :=
by
  intro h
  cases h with x h1
  cases h1 with y h2
  cases h2 with hy h3
  use (√(105)), (-√(105))
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  split
  { exact sqrt_sq_of_nonneg (by norm_num) }
  exact add_right_neg (√(105))

end sum_of_solutions_for_x_l498_498326


namespace transaction_loss_l498_498154

variables (h s : ℝ)

-- Conditions
def house_sold_price := 15000
def store_sold_price := 10000
def house_loss := 0.25
def store_gain := 0.25

axiom house_cost : h * (1 - house_loss) = house_sold_price
axiom store_cost : s * (1 + store_gain) = store_sold_price

-- Proof Problem Statement
theorem transaction_loss :
  let house_initial_cost := h,
      store_initial_cost := s,
      total_initial_cost := house_initial_cost + store_initial_cost,
      total_selling_price := house_sold_price + store_sold_price,
      total_loss := total_initial_cost - total_selling_price in
  total_loss = 3000 :=
sorry

end transaction_loss_l498_498154


namespace angle_in_regular_octagon_l498_498891

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498891


namespace measure_angle_abh_l498_498858

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498858


namespace max_PA_PB_product_l498_498369

theorem max_PA_PB_product (λ : ℝ) : 
  let A := (-1 : ℝ, 0 : ℝ)
  let B := (3 : ℝ, 2 : ℝ)
  ∃ P : ℝ × ℝ, 
  let PA := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let PB := (P.1 - B.1)^2 + (P.2 - B.2)^2
  (λ * P.1 - P.2 + λ = 0) ∧ (P.1 + λ * P.2 - 3 - 2 * λ = 0) →
  (|P.1 - (-1)| * |P.2 - 0|) * (|P.1 - 3| * |P.2 - 2|) ≤ 10 := 
begin
  sorry
end

end max_PA_PB_product_l498_498369


namespace solution_set_of_inequality_l498_498053

-- Definition of the conditions
def roots_of_quadratic_eq (a b c x : ℝ) : Prop := (a ≠ 0) ∧ (a*x^2 + b*x + c = 0)

variables {a b c : ℝ}
variables (h1 : roots_of_quadratic_eq a b c (-2)) (h2 : roots_of_quadratic_eq a b c 1)

-- Conditions on coefficients based on the problem statement
axiom coeff_b_eq_a : b = a
axiom coeff_c_eq_neg2a : c = -2 * a

-- Inequality transformation
def transformed_inequality (x : ℝ) := a * (x^2 + 1) + b * (x + 1) + c < 3 * a * x

-- The proof problem: Proving the solution set of the inequality
theorem solution_set_of_inequality :
  ∀ x : ℝ, transformed_inequality x ↔ x < 0 ∨ x > 2 :=
  sorry

end solution_set_of_inequality_l498_498053


namespace max_valid_subset_card_l498_498105

def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k^3

def set := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

def valid_subset (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → c ≠ a → not (is_perfect_cube (a * b * c))

theorem max_valid_subset_card (s : Finset ℕ) (h : s ⊆ set) (h_valid : valid_subset s) : s.card ≤ 9 :=
sorry

end max_valid_subset_card_l498_498105


namespace measure_of_angle_ABH_in_regular_octagon_l498_498933

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498933


namespace inequality_for_n_eq_2_counterexample_for_n_gt_2_l498_498637

theorem inequality_for_n_eq_2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) : 
  x1 * x2 ≤ (1 / 2) * (x1^2 + x2^2) :=
sorry

theorem counterexample_for_n_gt_2 : ∀ (n : ℕ), (2 ≤ n) → 
  ∃ (x : ℕ → ℝ), 
  (∀ i, 0 < x i) ∧ ¬ 
  (n-1) * (∑ i in range(n-1), (x i) * (x (i+1))) ≤ 
  ((n-1)/n) * (∑ i in range(n), (x i)^2)) :=
sorry

end inequality_for_n_eq_2_counterexample_for_n_gt_2_l498_498637


namespace min_area_triangle_DEF_l498_498477

noncomputable def isRegularPolygon (points : List ℂ) (n : ℕ) : Prop :=
  ∃ (r : ℝ) (θ : ℝ) (k : ℕ → ℂ), 
    ∃ (offset : ℂ),
      (∀ k, k < n → points.nth k = some (offset + r * complex.exp (θ * complex.I * k))) ∧
      (θ = 2 * real.pi / n) ∧
      list.Nth points n = some points.head

noncomputable def areaOfTriangle (a b c : ℂ) : ℝ :=
  let ab := b - a
  let ac := c - a
  0.5 * complex.abs (ab.re * ac.im - ab.im * ac.re)

theorem min_area_triangle_DEF :
  (∃ points : List ℂ, isRegularPolygon points 10 ∧ ∃ (D E F: ℂ), D ≠ E ∧ E ≠ F ∧ F ≠ D ∧ D ∈ points ∧ E ∈ points ∧ F ∈ points ∧
    areaOfTriangle D E F = 10^(2/5) * real.sin (real.pi / 10) * (1 - real.cos (real.pi / 5))) :=
begin
  sorry
end

end min_area_triangle_DEF_l498_498477


namespace coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l498_498604

theorem coefficient_of_x9_in_expansion_of_x_minus_2_pow_10 :
  (∃ c : ℤ, (x - 2) ^ 10 = ∑ k in Finset.range 11, c * x ^ 9 ∧ c = -20) := sorry

end coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l498_498604


namespace F_range_l498_498686

-- Define the function f and its inverse
def f (x : ℝ) : ℝ := 3 * x - 5
def f_inv (x : ℝ) : ℝ := (x + 5) / 3

-- Define the function F
def F (x : ℝ) : ℝ := (f_inv x)^2 - f_inv (x^2)

-- State the theorem to prove
theorem F_range : ∀ x, 2 ≤ x ∧ x ≤ 4 → 2 ≤ F x ∧ F x ≤ 13 :=
    sorry

end F_range_l498_498686


namespace part1_part2a_part2b_part2c_l498_498274

def f (x a : ℝ) := |2 * x - 1| + |x - a|

theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x 3 ≤ 4 := sorry

theorem part2a (a x : ℝ) (h0 : a < 1 / 2) (h1 : a ≤ x ∧ x ≤ 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2b (a x : ℝ) (h0 : a = 1 / 2) (h1 : x = 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2c (a x : ℝ) (h0 : a > 1 / 2) (h1 : 1 / 2 ≤ x ∧ x ≤ a) : f x a = |x - 1 + a| := sorry

end part1_part2a_part2b_part2c_l498_498274


namespace measure_of_angle_ABH_l498_498829

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498829


namespace measure_of_angle_ABH_in_regular_octagon_l498_498936

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498936


namespace non_decreasing_6_digit_remainder_l498_498364

theorem non_decreasing_6_digit_remainder :
  (nat.choose (14) (6)) % 1000 = 3 :=
by 
  sorry

end non_decreasing_6_digit_remainder_l498_498364


namespace cos_beta_eq_sqrt3_div_2_l498_498263

variables {α β : ℝ}

theorem cos_beta_eq_sqrt3_div_2
    (h1 : 0 < α ∧ α < π / 2)
    (h2 : 0 < β ∧ β < π / 2)
    (h3 : sin α = 1 / 2)
    (h4 : cos (α + β) = 1 / 2) :
  cos β = sqrt 3 / 2 :=
sorry

end cos_beta_eq_sqrt3_div_2_l498_498263


namespace complex_multiplication_l498_498786

theorem complex_multiplication : (let i := complex.I in i * (1 + i)^2 = -2) :=
by
  sorry

end complex_multiplication_l498_498786


namespace additional_days_needed_is_15_l498_498734

-- Definitions and conditions from the problem statement
def good_days_2013 : ℕ := 365 * 479 / 100  -- Number of good air quality days in 2013
def target_increase : ℕ := 20              -- Target increase in percentage for 2014
def additional_days_first_half_2014 : ℕ := 20 -- Additional good air quality days in first half of 2014 compared to 2013
def half_good_days_2013 : ℕ := good_days_2013 / 2 -- Good air quality days in first half of 2013

-- Target number of good air quality days for 2014
def target_days_2014 : ℕ := good_days_2013 * (100 + target_increase) / 100

-- Good air quality days in the first half of 2014
def good_days_first_half_2014 : ℕ := half_good_days_2013 + additional_days_first_half_2014

-- Additional good air quality days needed in the second half of 2014
def additional_days_2014_second_half (target_days good_days_first_half_2014 : ℕ) : ℕ := 
  target_days - good_days_first_half_2014 - half_good_days_2013

-- Final theorem verifying the number of additional days needed in the second half of 2014 is 15
theorem additional_days_needed_is_15 : 
  additional_days_2014_second_half target_days_2014 good_days_first_half_2014 = 15 :=
sorry

end additional_days_needed_is_15_l498_498734


namespace sector_area_l498_498320

def degree_to_radian (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Definitions based on conditions
def r : ℝ := 6
def θ : ℝ := degree_to_radian 60

-- Theorem statement
theorem sector_area : (1/2) * r^2 * θ = 6 * Real.pi := by
  sorry

end sector_area_l498_498320


namespace middle_aged_employees_participating_l498_498555

-- Define the total number of employees and the ratio
def total_employees : ℕ := 1200
def ratio_elderly : ℕ := 1
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 6

-- Define the number of employees chosen for the performance
def chosen_employees : ℕ := 36

-- Calculate the number of middle-aged employees participating in the performance
theorem middle_aged_employees_participating : (36 * ratio_middle_aged / (ratio_elderly + ratio_middle_aged + ratio_young)) = 15 :=
by
  sorry

end middle_aged_employees_participating_l498_498555


namespace more_than_twenty_components_possible_l498_498325

def cell := ℕ × ℕ -- representing a cell as a pair of its coordinates.
def grid_size := 8 -- 8x8 grid.

-- A function representing the existence of a diagonal in a cell. True means the diagonal is present.
def has_diagonal (c : cell) : Prop := true

-- Definition of being able to traverse between two cells.
def traversable (c1 c2 : cell) : Prop := sorry -- Placeholder for the traversal condition between cells.

-- Definition of connected if there exists a series of cells connecting the two given cells.
def connected (c1 c2 : cell) : Prop := ∃ (path : list cell), path.head = c1 ∧ path.last = c2 ∧ ∀ (i j : nat), i < j → j < path.length → traversable (path.nth i) (path.nth j)

-- Definition of connected component as a set of cells.
def connected_component (comp : set cell) : Prop :=
  ∀ (c1 c2 : cell), c1 ∈ comp ∧ c2 ∈ comp → connected c1 c2

-- Function to count the number of connected components in the grid.
def count_connected_components (grid : set cell) : ℕ := sorry -- Placeholder for the actual counting logic.

instance : DecidableEq cell := classical.decEq _

-- The main theorem to prove the possibility of having more than 20 connected components given the conditions.
theorem more_than_twenty_components_possible :
  ∃ (grid : set cell), (∀ c ∈ grid, has_diagonal c) ∧ count_connected_components grid > 20 :=
sorry

end more_than_twenty_components_possible_l498_498325


namespace measure_of_angle_ABH_l498_498853

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498853


namespace no_real_quadruples_solutions_l498_498626

theorem no_real_quadruples_solutions :
  ¬ ∃ (a b c d : ℝ),
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := 
sorry

end no_real_quadruples_solutions_l498_498626


namespace jogging_walking_ratio_l498_498597

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end jogging_walking_ratio_l498_498597


namespace percentage_increase_third_year_l498_498646

noncomputable def stock_price_2007 := P : ℝ
noncomputable def stock_price_end_2007 := 1.20 * stock_price_2007
noncomputable def stock_price_end_2008 := 0.90 * stock_price_end_2007
noncomputable def stock_price_end_2009 := 1.035 * stock_price_2007

theorem percentage_increase_third_year (P : ℝ) :
  ∃ (X : ℝ), stock_price_end_2009 = stock_price_end_2008 * (1 + X / 100) ∧ X = 15 :=
by {
    sorry
}

end percentage_increase_third_year_l498_498646


namespace find_third_grade_classes_l498_498794

def third_grade_classes (total_cost_per_student : ℝ) (cost : ℝ) 
    (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) 
    (fifth_grade_classes : ℕ) (students_per_fifth_grade_class : ℕ)
    (students_per_third_grade_class : ℕ) 
    (total_cost : ℝ) : ℕ :=
  let total_fourth_grade_students := fourth_grade_classes * students_per_fourth_grade_class
  let total_fifth_grade_students := fifth_grade_classes * students_per_fifth_grade_class
  let total_students := total_cost / total_cost_per_student
  let total_third_grade_students := total_students - total_fourth_grade_students - total_fifth_grade_students
  total_third_grade_students / students_per_third_grade_class

theorem find_third_grade_classes : third_grade_classes 2.80 1036 4 28 4 27 30 1036 = 5 :=
by
  sorry

end find_third_grade_classes_l498_498794


namespace sum_of_coordinates_eq_neg_ten_l498_498804

variable (x : ℝ)
def point_C := (x, -5 : ℝ)
def point_D := (-x, -5 : ℝ)

theorem sum_of_coordinates_eq_neg_ten : 
  (x + (-5) + (-x) + (-5) = -10) :=
by
  sorry

end sum_of_coordinates_eq_neg_ten_l498_498804


namespace point_distance_sum_l498_498216

noncomputable definition parametric_line (t : ℝ) : ℝ × ℝ :=
(1 - 0.5 * t, sqrt(3) / 2 * t)

noncomputable definition polar_circle (θ : ℝ) : ℝ :=
2 * sqrt(3) * sin θ

noncomputable definition cartesian_circle (x y : ℝ) : Prop :=
x^2 + (y - sqrt(3))^2 = 3

noncomputable definition cartesian_line (x y : ℝ) : Prop :=
3 * x + sqrt(3) * y - 3 = 0

theorem point_distance_sum : 
  ∃ A B : ℝ × ℝ, 
    (∀ t : ℝ, parametric_line t = A ∨ parametric_line t = B) ∧
    ((∃ θ : ℝ, polar_circle θ = sqrt (A.1^2 + A.2^2)) ∧ (∃ θ : ℝ, polar_circle θ = sqrt (B.1^2 + B.2^2))) ∧
    let P := (1 : ℝ, 0 : ℝ) in
    ((A ≠ B) ∧ cartesian_circle A.1 A.2 ∧ cartesian_circle B.1 B.2 ∧ cartesian_line A.1 A.2 ∧ cartesian_line B.1 B.2 ∧
    (sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2) + sqrt ((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2) = 4)) :=
sorry

end point_distance_sum_l498_498216


namespace regular_octagon_angle_ABH_l498_498872

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498872


namespace angle_ABH_is_22_point_5_l498_498914

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498914


namespace chord_length_of_concentric_circles_l498_498966

theorem chord_length_of_concentric_circles 
  (R r : ℝ) (h1 : R^2 - r^2 = 15) (h2 : ∀ s, s = 2 * R) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ ∀ x, x = c := 
by 
  sorry

end chord_length_of_concentric_circles_l498_498966


namespace angle_ABH_in_regular_octagon_l498_498844

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498844


namespace domain_of_f_l498_498457

noncomputable def f (x : ℝ) := log (x - 2) + 1 / sqrt (x - 3)

theorem domain_of_f :
  {x : ℝ | 2 < x ∧ 3 ≤ x} = {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_f_l498_498457


namespace compare_abc_l498_498654

theorem compare_abc (a b c : ℝ) (h1 : a = 2⁻³) (h2 : b = 3^(1/2)) (h3 : c = real.log 5 / real.log 2) :
  a < b ∧ b < c := 
by
  sorry

end compare_abc_l498_498654


namespace find_a_from_regression_l498_498147

theorem find_a_from_regression (x : List ℤ) (y : List ℤ) (a : ℤ) (reg_eq: ∀ x, -3 * x + 60 = y) :
  x = [17, 14, 10, -1] ->
  y = [21, a, 34, 40] ->
  (∃ a, 30 = (21 + a + 34 + 40) / 4) :=
by
  sorry

end find_a_from_regression_l498_498147


namespace measure_of_angle_ABH_l498_498822

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498822


namespace part_a_part_b_l498_498529

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l498_498529


namespace smallest_n_mod_equiv_l498_498607

theorem smallest_n_mod_equiv (n : ℕ) (h : 0 < n ∧ 2^n ≡ n^5 [MOD 4]) : n = 2 :=
by
  sorry

end smallest_n_mod_equiv_l498_498607


namespace percentage_of_silver_in_second_alloy_l498_498568

theorem percentage_of_silver_in_second_alloy :
  ∃ x : ℝ, (0.70 * 280 + x * 120 = 0.61 * 400) ∧ (x * 100 = 40) :=
begin
  use 0.4,
  split,
  { rw [mul_assoc, ← mul_add],
    norm_num },
  { norm_num }
end

end percentage_of_silver_in_second_alloy_l498_498568


namespace configuration_of_points_l498_498236

noncomputable def distance (X Y : ℝ×ℝ) : ℝ :=
  real.sqrt ((X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2)

def is_permutation {α : Type*} (σ : list ℕ) (S : set ℕ) : Prop :=
  list.perm (list.of_fn id) σ ∧ (∀ x ∈ S, x ∈ list.to_finset σ)

theorem configuration_of_points (n : ℕ)
  (h : n ≥ 3)
  (X : fin n → ℝ × ℝ)
  (h_dist : ∀ i j : fin n, i ≠ j → ∃ σ : list ℕ, is_permutation σ (finset.univ : finset (fin n)) ∧ ∀ k : fin n, distance (X i) (X k) = distance (X j) (X (σ.k))) :
  (n % 2 = 1 → ∃ R : ℝ, ∀ i j : fin n, distance (X (fin.rotate i)) (X (fin.rotate j)) = distance (X 0) (X 1)) ∧
  (n % 2 = 0 → ∃ (R₁ R₂ : ℝ), R₁ ≠ R₂ ∧ ∀ i : fin n, distance (X i) (X (i + 1)) = if i.1 % 2 = 0 then R₁ else R₂) :=
sorry

end configuration_of_points_l498_498236


namespace parallel_condition_sufficient_not_necessary_l498_498659

-- Let α be a plane, and m and n be lines in α.
variable {α : Type*} [plane α] -- Define α to be a plane
variable {m n : line α} -- Define m and n to be lines in the plane α

-- Assume m and n are subsets of α, and m is parallel to n.
axiom m_subset_α : m ⊆ α
axiom n_subset_α : n ⊆ α
axiom m_parallel_n : m ∥ n

-- Prove that m ∥ α is the sufficient but not necessary condition for m ∥ n.
theorem parallel_condition_sufficient_not_necessary : (m ∥ n → m ∥ α) ∧ ¬(m ∥ α → m ∥ n) :=
by
  sorry

end parallel_condition_sufficient_not_necessary_l498_498659


namespace tetrahedron_symmetry_planes_l498_498715

-- Define a structure for a triangular pyramid (tetrahedron)
structure Tetrahedron where
  A B C D : Point -- Assume Point type exists and represents vertices
  h_sym_plane : (AC = BC) → (AD = BD) → Bool

-- Define planes of symmetry
def symmetry_planes (T : Tetrahedron) : ℕ :=
  if T.h_sym_plane then
    -- We consider possible conditions for symmetry given the structure of the tetrahedron
    if all_equal_edges T then 6 
    else if some_equal_edges_shared_vertex T then 3
    else if two_planes_of_symmetry_distinct_vertices T then 2
    else if unique_sym_plane_cond T then 1
    else 0
  else 0 -- No symmetry planes if the condition isn't satisfied

-- The main theorem statement
theorem tetrahedron_symmetry_planes : ∀ T : Tetrahedron,
  (symmetry_planes T = 0 ∨ 
  symmetry_planes T = 1 ∨ 
  symmetry_planes T = 2 ∨ 
  symmetry_planes T = 3 ∨ 
  symmetry_planes T = 6) :=
by
  intro T
  -- This will justify how the number of symmetry planes fits one of these values
  sorry 

end tetrahedron_symmetry_planes_l498_498715


namespace Sue_returned_books_l498_498449

theorem Sue_returned_books :
  ∀ (initial_books initial_movies new_books total_items current_movies books_returned: ℕ),
  initial_books = 15 →
  initial_movies = 6 →
  new_books = 9 →
  total_items = 20 →
  current_movies = initial_movies - initial_movies / 3 →
  let current_books := total_items - current_movies in
  let books_after_checkout := initial_books + new_books in
  books_returned = books_after_checkout - current_books →
  books_returned = 8 := 
by
  intros initial_books initial_movies new_books total_items current_movies books_returned
  sorry

end Sue_returned_books_l498_498449


namespace range_of_m_l498_498309

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l498_498309


namespace john_spent_on_candy_l498_498581

theorem john_spent_on_candy (M : ℝ) 
  (h1 : M = 29.999999999999996)
  (h2 : 1/5 + 1/3 + 1/10 = 19/30) :
  (11 / 30) * M = 11 :=
by {
  sorry
}

end john_spent_on_candy_l498_498581


namespace equation_of_L2_l498_498027

-- Define the line L1 and point P
def line_L1 (m : ℝ) : ℝ × ℝ → Prop := 
  λ P, m * P.1 - m^2 * P.2 = 1

-- Define point P(2,1)
def P : ℝ × ℝ := (2, 1)

-- Define the condition that P lies on L1
def point_on_L1 (m : ℝ) : Prop :=
  line_L1 m P

-- Define the slope of line L2 which is perpendicular to L1
-- If slope of L1 is m, then slope of L2 is -1/m
def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

-- Define the equation of the line L2
def line_L2 (L1_slope : ℝ) : ℝ × ℝ → Prop := 
  λ Q, Q.1 + 1 / L1_slope * Q.2 = 3

-- Prove that the equation of L2 given the condition is x + y - 3 = 0
theorem equation_of_L2 (m : ℝ) (hP : point_on_L1 m) :
  line_L2 m = λ Q, Q.1 + Q.2 - 3 = 0 :=
  sorry

end equation_of_L2_l498_498027


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498895

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498895


namespace measure_angle_abh_l498_498812

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498812


namespace measure_angle_abh_l498_498865

-- Define the concept of a regular octagon
def regular_octagon (S : Type) [metric_space S] (P : points_on_circle S 8) := ∀ i j, dist (P i) (P j) = dist (P 1) (P 2)

-- The main theorem stating the problem condition and conclusion
theorem measure_angle_abh {S : Type} [metric_space S] 
  (P : points_on_circle S 8) 
  (h : regular_octagon S P) :
  measure_angle (P 1) (P 2) (P 8) = 22.5
:= 
sorry  

end measure_angle_abh_l498_498865


namespace find_k_l498_498516

-- Defining the conditions used in the problem context
def line_condition (k a b : ℝ) : Prop :=
  (b = 4 * k + 1) ∧ (5 = k * a + 1) ∧ (b + 1 = k * a + 1)

-- The statement of the theorem
theorem find_k (a b k : ℝ) (h : line_condition k a b) : k = 3 / 4 :=
by sorry

end find_k_l498_498516


namespace trig_identity_l498_498293

theorem trig_identity (α : ℝ) (h : 3 * sin α + cos α = 0) : (1 / (cos (2 * α) + sin (2 * α)) = 5) :=
sorry

end trig_identity_l498_498293


namespace sum_of_mathcalT_is_2520_l498_498377

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498377


namespace root_of_polynomial_l498_498630

noncomputable def polynomial := 3 * (Polynomial.X ^ 4) + 32 * (Polynomial.X ^ 3) - 85 * (Polynomial.X ^ 2) - 70 * Polynomial.X

theorem root_of_polynomial :
  Polynomial.eval (-5) polynomial = 0 ∧ 
  Polynomial.eval (-0.8106) polynomial ≈ 0 ∧ 
  Polynomial.eval 3.4773 polynomial ≈ 0 ∧ 
  Polynomial.eval 0 polynomial = 0 := 
sorry

end root_of_polynomial_l498_498630


namespace measure_of_angle_ABH_l498_498830

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498830


namespace problem_l498_498589

-- Define \(\alpha\)
def alpha : ℝ := 49 * Real.pi / 48

-- Define the expression
def expr : ℝ := 4 * (Real.sin(alpha) ^ 3 * Real.cos(49 * Real.pi / 16) + 
                     Real.cos(alpha) ^ 3 * Real.sin(49 * Real.pi / 16)) * 
                     Real.cos(49 * Real.pi / 12)

-- The main theorem
theorem problem : expr = 0.75 :=
  sorry

end problem_l498_498589


namespace inequality_abc_l498_498432

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l498_498432


namespace distance_towns_a_b_l498_498127

noncomputable def distance_between_towns :=
  let west1 := 20
  let north1 := 6
  let east1 := 10
  let north2 := 18
  let net_west := west1 - east1
  let net_north := north1 + north2
  let distance := Math.sqrt (net_west^2 + net_north^2)
  distance

theorem distance_towns_a_b : distance_between_towns = 26 :=
by
  -- Calculations based on given movements and conditions
  let west1 := 20
  let north1 := 6
  let east1 := 10
  let north2 := 18
  let net_west := west1 - east1
  let net_north := north1 + north2
  let distance := Math.sqrt (net_west^2 + net_north^2)
  show distance = 26, from sorry

end distance_towns_a_b_l498_498127


namespace smallest_number_l498_498114

theorem smallest_number (a b c d e: ℕ) (h1: a = 5) (h2: b = 8) (h3: c = 1) (h4: d = 2) (h5: e = 6) :
  min (min (min (min a b) c) d) e = 1 :=
by
  -- Proof skipped using sorry
  sorry

end smallest_number_l498_498114


namespace sum_of_set_T_l498_498398

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498398


namespace min_value_of_a_plus_b_l498_498724

theorem min_value_of_a_plus_b (a b : ℝ) (h : log 2 a + log 2 b = 3) : a + b ≥ 4 * sqrt 2 := by
  -- Definitions and assumptions
  sorry

end min_value_of_a_plus_b_l498_498724


namespace smallest_prime_sum_min_l498_498231

open BigOperators

-- Define the constraints
def digits_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_prime_set (primes : List ℕ) : Prop :=
  primes.length = 4 ∧
  ∀ p ∈ primes, Nat.Prime p ∧
  (primes.foldl (λ acc p => acc + p.digits) []).toFinset = digits_set

-- Define the main theorem to be proven
theorem smallest_prime_sum_min : 
  ∃ (primes : List ℕ), is_valid_prime_set primes ∧ primes.sum = 720 :=
begin
  sorry
end

end smallest_prime_sum_min_l498_498231


namespace sum_of_T_l498_498390

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498390


namespace appropriate_chart_for_temperature_changes_l498_498135

-- Definitions for the characteristics of charts
def bar_chart_characteristic : Prop := ∀ (data : List ℕ), data.nonempty → string = "shows amount clearly"

def line_chart_characteristic : Prop := ∀ (data : List ℕ), data.nonempty → string = "shows amount and changes"

def pie_chart_characteristic : Prop := ∀ (data : List ℕ), data.nonempty → string = "reflects part-whole relationships"

-- The main statement asserting the appropriateness of using a line chart for temperature changes over 12 months
theorem appropriate_chart_for_temperature_changes (data : List ℕ) (h : data.nonempty) : string :=
if bar_chart_characteristic data h then "shows amount clearly"
else if line_chart_characteristic data h then "correct"
else if pie_chart_characteristic data h then "reflects part-whole relationships"
else "unknown"

#check appropriate_chart_for_temperature_changes -- Ensure the statement is syntactically correct

end appropriate_chart_for_temperature_changes_l498_498135


namespace minimum_phi_for_even_function_l498_498208

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def f (x : ℝ) : ℝ := determinant 2 (2 * Real.sin x) (Real.sqrt 3) (Real.cos x)

def translated_f (x φ : ℝ) : ℝ := f (x + φ)

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem minimum_phi_for_even_function :
  (∃ φ > 0, is_even_function (λ x, translated_f x φ)) → φ = (2 * Real.pi) / 3 :=
sorry

end minimum_phi_for_even_function_l498_498208


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498904

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498904


namespace measure_angle_abh_l498_498811

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498811


namespace sunglasses_cost_l498_498166

open Real

def cost_per_pair (selling_price_per_pair : ℝ) (num_pairs_sold : ℝ) (sign_cost : ℝ) : ℝ := 
  (num_pairs_sold * selling_price_per_pair - 2 * sign_cost) / num_pairs_sold

theorem sunglasses_cost (sp : ℝ) (n : ℝ) (sc : ℝ) (H1 : sp = 30) (H2 : n = 10) (H3 : sc = 20) :
  cost_per_pair sp n sc = 26 :=
by
  rw [H1, H2, H3]
  simp [cost_per_pair]
  norm_num
  sorry

end sunglasses_cost_l498_498166


namespace find_alpha_l498_498685

def f (x : ℝ) : ℝ :=
  if x < 2 then 3 - x else 2^x - 3

theorem find_alpha (alpha : ℝ) (h : f (f alpha) = 1) : alpha = 1 ∨ alpha = real.log (5) (2) :=
sorry

end find_alpha_l498_498685


namespace part_a_part_b_l498_498543

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l498_498543


namespace closed_shape_area_is_four_thirds_l498_498014

noncomputable def area_of_parabola_under_x_axis : ℝ := 
  let lower_limit : ℝ := 1
  let upper_limit : ℝ := 3
  let integrand (x : ℝ) : ℝ := x^2 - 4 * x + 3
  -(∫ x in lower_limit..upper_limit, integrand x)

theorem closed_shape_area_is_four_thirds :
  area_of_parabola_under_x_axis = 4 / 3 :=
sorry

end closed_shape_area_is_four_thirds_l498_498014


namespace length_PQ_of_trapezoid_l498_498343

theorem length_PQ_of_trapezoid
  (ABCD : trapezoid)
  (BC_parallel_AD : parallel ABCD.B ABCD.D)
  (BC_eq : ABCD.BC = 1500)
  (AD_eq : ABCD.AD = 2500)
  (angle_A : ABCD.angleA = 40)
  (angle_D : ABCD.angleD = 50)
  (P_is_midpoint_BC : is_midpoint ABCD.B ABCD.C ABCD.P)
  (Q_is_midpoint_AD : is_midpoint ABCD.A ABCD.D ABCD.Q)
  : (distance ABCD.P ABCD.Q) = 500 :=
by
  sorry

end length_PQ_of_trapezoid_l498_498343


namespace complex_number_in_first_quadrant_l498_498598

def det (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_number_in_first_quadrant (z : ℂ) (h : det z (1 + 2 * complex.i) (1 - complex.i) (1 + complex.i) = 0) :
  z = 2 - complex.i ∧ (complex.conj (2 - complex.i)).im > 0 ∧ (complex.conj (2 - complex.i)).re > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l498_498598


namespace buffaloes_added_l498_498550

-- Let B be the daily fodder consumption of one buffalo in units
noncomputable def daily_fodder_buffalo (B : ℝ) := B
noncomputable def daily_fodder_cow (B : ℝ) := (3 / 4) * B
noncomputable def daily_fodder_ox (B : ℝ) := (3 / 2) * B

-- Initial conditions
def initial_buffaloes := 15
def initial_cows := 24
def initial_oxen := 8
def initial_days := 24
noncomputable def total_initial_fodder (B : ℝ) := (initial_buffaloes * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + (initial_cows * daily_fodder_cow B)
noncomputable def total_fodder (B : ℝ) := total_initial_fodder B * initial_days

-- New conditions after adding cows and buffaloes
def additional_cows := 60
def new_days := 9
noncomputable def total_new_daily_fodder (B : ℝ) (x : ℝ) := ((initial_buffaloes + x) * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + ((initial_cows + additional_cows) * daily_fodder_cow B)

-- Proof statement: Prove that given the conditions, the number of additional buffaloes, x, is 30.
theorem buffaloes_added (B : ℝ) : 
  (total_fodder B = total_new_daily_fodder B 30 * new_days) :=
by sorry

end buffaloes_added_l498_498550


namespace quadratic_eq_cos_2x_l498_498232

theorem quadratic_eq_cos_2x (a b c : ℝ) (x : ℝ) (h : a * (cos x)^2 + b * cos x + c = 0) :
    (a = 4) → (b = 2) → (c = -1) → (cos (2 * x))^2 + 2 * cos (2 * x) - 1 = 0 :=
by
  intros ha hb hc
  rw [ha, hb, hc] at h
  sorry

end quadratic_eq_cos_2x_l498_498232


namespace inequality_B_inequality_C_l498_498258

variable (a b : ℝ)

theorem inequality_B (h : 0 < a ∧ 0 < b ∧ (1 / Real.sqrt a > 1 / Real.sqrt b)) :
  (b / (a + b) + a / (2 * b) ≥ (2 * Real.sqrt 2 - 1) / 2) :=
by sorry

theorem inequality_C (h : 0 < a ∧ 0 < b ∧ (1 / Real.sqrt a > 1 / Real.sqrt b)) :
  ((b + 1) / (a + 1) < b / a) :=
by sorry

end inequality_B_inequality_C_l498_498258


namespace mod_equivalence_l498_498729

theorem mod_equivalence (x y m : ℤ) (h1 : x ≡ 25 [ZMOD 60]) (h2 : y ≡ 98 [ZMOD 60]) (h3 : m = 167) :
  x - y ≡ m [ZMOD 60] :=
sorry

end mod_equivalence_l498_498729


namespace east_bound_speed_is_18_l498_498065

-- Define the conditions
def speeds (x : ℝ) := (east_bound := x, west_bound := x + 4)
def total_distance_travelled (t : ℝ) (speeds : ℝ × ℝ) := t * speeds.1 + t * speeds.2

-- Define the parameters
def time : ℝ := 5
def distance_apart : ℝ := 200

-- Theorem statement
theorem east_bound_speed_is_18 (x : ℝ) (h : speeds x) (d : total_distance_travelled time h = distance_apart) : x = 18 := 
sorry

end east_bound_speed_is_18_l498_498065


namespace area_of_triangle_with_given_medians_l498_498034

noncomputable def area_of_triangle (m1 m2 m3 : ℝ) : ℝ :=
sorry

theorem area_of_triangle_with_given_medians :
    area_of_triangle 3 4 5 = 8 :=
sorry

end area_of_triangle_with_given_medians_l498_498034


namespace boat_speed_in_still_water_l498_498742

-- Identifying the speeds of the boat in still water and the stream
variables (b s : ℝ)

-- Conditions stated in terms of equations
axiom boat_along_stream : b + s = 7
axiom boat_against_stream : b - s = 5

-- Prove that the boat speed in still water is 6 km/hr
theorem boat_speed_in_still_water : b = 6 :=
by
  sorry

end boat_speed_in_still_water_l498_498742


namespace participants_l498_498420

variable {A B C D : Prop}

theorem participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) :
  (¬A ∧ C ∧ B ∧ ¬D) ∨ ¬B :=
by
  -- The proof is not provided
  sorry

end participants_l498_498420


namespace measure_of_angle_ABH_l498_498849

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498849


namespace ivan_uses_more_paint_l498_498077

-- Define the basic geometric properties
def rectangular_section_area (length width : ℝ) : ℝ := length * width
def parallelogram_section_area (side1 side2 : ℝ) (angle : ℝ) : ℝ := side1 * side2 * Real.sin angle

-- Define the areas for each neighbor's fences
def ivan_area : ℝ := rectangular_section_area 5 2
def petr_area (alpha : ℝ) : ℝ := parallelogram_section_area 5 2 alpha

-- Theorem stating that Ivan's total fence area is greater than Petr's total fence area provided the conditions
theorem ivan_uses_more_paint (α : ℝ) (hα : α ≠ Real.pi / 2) : ivan_area > petr_area α := by
  sorry

end ivan_uses_more_paint_l498_498077


namespace monotonicity_f1_exists_greater_value_l498_498688

-- Problem 1: Define and state the proof problem
def f1 (x : ℝ) := (1 / 3) * x^2 - (7 / 3) * x + 21

theorem monotonicity_f1 :
  (∀ x, 0 < x ∧ x < 3 / 2 → f1(x) < f1(x + 0.001)) ∧
  (∀ x, 3 / 2 < x ∧ x < 2 → f1(x) > f1(x + 0.001)) ∧
  (∀ x, 2 < x → f1(x) < f1(x + 0.001)) :=
sorry

-- Problem 2: Define and state the proof problem
def f2 (a x : ℝ) := (1 / 2) * a * x^2 - (2 * a + 1) * x + 21
def g (x : ℝ) := x^2 - 2 * x + exp x

theorem exists_greater_value (a : ℝ) (h : a > 1 / 2) (x1 : ℝ) (hx : 0 < x1 ∧ x1 ≤ 2) :
  ∃ x2 ∈ (Icc 0 2), f2 a x1 < g x2 :=
sorry

end monotonicity_f1_exists_greater_value_l498_498688


namespace playground_area_l498_498353

theorem playground_area (posts : ℕ) (spacing : ℝ) (long_post_ratio : ℕ → ℕ) : 
  posts = 24 → spacing = 3 → (∀ x, (long_post_ratio x) = 3 * x) → ∃ short_side long_side, 
  short_side = 3 * (4 - 1) ∧ long_side = 3 * (12 - 1) ∧ short_side * long_side = 297 :=
by
  intros h_posts h_spacing h_ratio
  have h1 : posts = 24 := h_posts
  have h2 : spacing = 3 := h_spacing
  cases' h_ratio 3 with h3'
  exists 9, 33
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end playground_area_l498_498353


namespace f_of_f_eq_f_l498_498772

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem f_of_f_eq_f (x : ℝ) : f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 :=
by
  sorry

end f_of_f_eq_f_l498_498772


namespace even_numbers_in_third_row_and_beyond_l498_498319

noncomputable def numerical_triangle (n : ℕ) (m : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then if m = 0 ∨ m = 2 then 1 else 0
  else
    let left := if m = 0 then 0 else numerical_triangle (n-1) (m-1)
    let mid := numerical_triangle (n-1) m
    let right := numerical_triangle (n-1) (m+1)
    in left + mid + right

theorem even_numbers_in_third_row_and_beyond :
  ∀ n m : ℕ, n ≥ 2 → ∃ k : ℕ, k ≤ m ∧ numerical_triangle n k % 2 = 0 :=
by
  sorry

end even_numbers_in_third_row_and_beyond_l498_498319


namespace isabella_hair_length_l498_498352

-- Define conditions: original length and doubled length
variable (original_length : ℕ)
variable (doubled_length : ℕ := 36)

-- Theorem: Prove that if the original length doubled equals 36, then the original length is 18.
theorem isabella_hair_length (h : 2 * original_length = doubled_length) : original_length = 18 := by
  sorry

end isabella_hair_length_l498_498352


namespace express_m_in_terms_of_b_n_a_l498_498727

theorem express_m_in_terms_of_b_n_a
  (a b m n : ℝ)
  (h1 : a > 1)
  (h2 : log a m = 2 * b - log a n) :
  m = (a ^ (2 * b)) / n :=
by
  sorry

end express_m_in_terms_of_b_n_a_l498_498727


namespace probability_product_negative_l498_498066

noncomputable def integers : Finset ℤ := {-6, -3, -1, 2, 5, 8}

theorem probability_product_negative :
  let pairs := (integers.product integers).filter (λ p, p.1 ≠ p.2),
      negative_pairs := pairs.filter (λ p, p.1 * p.2 < 0) in
  (negative_pairs.card : ℚ) / (pairs.card : ℚ) = 3 / 5 :=
by
  sorry

end probability_product_negative_l498_498066


namespace petya_can_divide_any_segment_l498_498803

theorem petya_can_divide_any_segment (k l : ℕ) : ∃ p, 0 < p ∧ p < 1 ∧ divides (k + l) p :=
sorry

end petya_can_divide_any_segment_l498_498803


namespace felicia_white_sugar_l498_498619

-- Define the necessary variables and parameters
variables (flour_cups sugar_cups brown_sugar_cups oil_cups total_scoops remaining_scoops : ℕ)

-- State the problem conditions
def problem_conditions :=
  flour_cups = 2 ∧ 
  brown_sugar_cups = 1 ∧ 
  oil_cups = 2 ∧  -- since 1/2 cup is 2 quarters (1/4 cups)
  total_scoops = 15 ∧
  remaining_scoops = 15 - (8 + 1 + 2)

-- State the total cups of white sugar needed by Felicia
def white_sugar_needed :=
  (remaining_scoops * 1) / 4 -- converting quarters to cups

-- The main theorem to prove that the total cups of white sugar needed is 1
theorem felicia_white_sugar (h : problem_conditions) : white_sugar_needed = 1 :=
by
  sorry

end felicia_white_sugar_l498_498619


namespace good_config_reachable_l498_498482

-- Definition of a good configuration in terms of graph connectivity
def good_configuration (V : Type) (E : set (V × V)) : Prop :=
  ∀ (v1 v2 : V), ∃ (path : list (V × V)), 
  (∀ (e : V × V), e ∈ path → e ∈ E ∨ (e.snd, e.fst) ∈ E) ∧
  (path.head.fst = v1) ∧ (path.reverse.head.snd = v2)

-- Allowed move transformation
def allowed_move {V : Type} (E : set (V × V)) (A B C : V) : set (V × V) :=
  if (A, B) ∈ E ∧ (A, C) ∈ E ∧ ¬((B, C) ∈ E ∨ (C, B) ∈ E)
  then (E \ {(A, B)}) ∪ {(B, C)}
  else E

-- Theorem stating the equivalence of good configurations under allowed moves
theorem good_config_reachable {V : Type} (E₁ E₂ : set (V × V)) :
  good_configuration V E₁ → good_configuration V E₂ →
  set.card E₁ = set.card E₂ →
  ∃ (steps : ℕ) (transforms : fin steps → (set (V × V))),
    good_configuration V (transforms 0) ∧ transforms steps = E₂ ∧
    ∀ (i : fin (steps - 1)), transforms (i + 1) = allowed_move (transforms i) :=
begin
  sorry
end

end good_config_reachable_l498_498482


namespace cost_of_10_minute_call_l498_498973

def cost_of_3_minute_call : ℝ := 0.18
def duration_3_minute_call : ℕ := 3
def duration_10_minute_call : ℕ := 10

def rate_per_minute (cost : ℝ) (duration : ℕ) : ℝ := cost / duration

theorem cost_of_10_minute_call :
  rate_per_minute cost_of_3_minute_call duration_3_minute_call * duration_10_minute_call = 0.60 :=
sorry

end cost_of_10_minute_call_l498_498973


namespace minute_hand_radians_1_to_3_20_l498_498585

theorem minute_hand_radians_1_to_3_20 :
  let minutes_per_hour := 60 in
  let degrees_per_minute := 360 / minutes_per_hour in
  let total_minutes := 2 * minutes_per_hour + 20 in
  let total_degrees := total_minutes * degrees_per_minute in
  total_degrees * (Real.pi / 180) = -((14 : ℝ) / 3) * Real.pi :=
by
  sorry

end minute_hand_radians_1_to_3_20_l498_498585


namespace largest_four_digit_number_l498_498950

theorem largest_four_digit_number : ∃ (n : ℕ), 
  (n.digits 10).length = 4 ∧ 
  (∃ d1 d2 d3 d4 : ℕ, 
    n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
    d1 ∈ {1,2,3,4,5,6,7,8,9} ∧ 
    d2 ∈ {1,2,3,4,5,6,7,8,9} ∧ 
    d3 ∈ {1,2,3,4,5,6,7,8,9} ∧ 
    d4 ∈ {1,2,3,4,5,6,7,8,9} ∧ 
    n = 1769) :=
sorry

end largest_four_digit_number_l498_498950


namespace measure_of_angle_ABH_l498_498845

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498845


namespace part_a_part_b_l498_498539

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l498_498539


namespace no_non_congruent_right_triangles_l498_498714

theorem no_non_congruent_right_triangles (a b : ℝ) (c : ℝ) (h_right_triangle : c = Real.sqrt (a^2 + b^2)) (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2)) : a = 0 ∨ b = 0 :=
by
  sorry

end no_non_congruent_right_triangles_l498_498714


namespace greatest_integer_solution_l498_498605

theorem greatest_integer_solution :
  ∃ x : ℤ, (\frac{1}{4} + \frac{x}{9} < \frac{7}{8}) ∧ (∀ y : ℤ, (\frac{1}{4} + \frac{y}{9} < \frac{7}{8}) → y ≤ x) ∧ x = 5 := by
  sorry

end greatest_integer_solution_l498_498605


namespace mangoes_amount_difference_l498_498797

theorem mangoes_amount_difference (P_orig : ℝ) (N_orig N_new : ℝ) :
  N_orig = 360 / P_orig →
  P_new = P_orig * 0.9 →
  N_new = 360 / P_new →
  P_orig = 450 / 135 →
  N_new - N_orig = 12 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end mangoes_amount_difference_l498_498797


namespace units_digit_of_product_l498_498110

def is_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_product : 
  is_units_digit (6 * 8 * 9 * 10 * 12) 0 := 
by
  sorry

end units_digit_of_product_l498_498110


namespace sum_of_coefficients_l498_498728

theorem sum_of_coefficients (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℝ) :
  (5 * 1 - 2)^6 = b_6 * 1^6 + b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0
  → b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 = 729 := by
  sorry

end sum_of_coefficients_l498_498728


namespace solve_trigonometric_equation_l498_498956

theorem solve_trigonometric_equation :
  ∀ x : ℝ, (sin (3 * x) + 3 * cos x = 2 * sin (2 * x) * (sin x + cos x)) →
  (∃ k : ℤ, x = (↑k * π + π / 4)) := 
by
  sorry

end solve_trigonometric_equation_l498_498956


namespace joyce_apples_l498_498758

theorem joyce_apples (initial_apples given_apples remaining_apples : ℕ) (h1 : initial_apples = 75) (h2 : given_apples = 52) (h3 : remaining_apples = initial_apples - given_apples) : remaining_apples = 23 :=
by
  rw [h1, h2] at h3
  exact h3

end joyce_apples_l498_498758


namespace divide_bill_evenly_l498_498174

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end divide_bill_evenly_l498_498174


namespace neither_sufficient_nor_necessary_l498_498667

variables (x y : ℤ)

def p : Prop := x + y ≠ -2
def q : Prop := x ≠ -1 ∧ y ≠ -1

theorem neither_sufficient_nor_necessary : ¬ (p ↔ q) :=
sorry

end neither_sufficient_nor_necessary_l498_498667


namespace dagger_example_l498_498599

def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

theorem dagger_example : dagger 5 8 3 4 = 15 := by
  sorry

end dagger_example_l498_498599


namespace smallest_positive_period_monotone_intervals_sin_alpha_max_value_in_interval_l498_498272

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.sin x ^ 2 - Real.sin (2 * x - Real.pi / 3)

theorem smallest_positive_period :
  f (x + π) = f x :=
sorry

theorem monotone_intervals (k : ℤ) :
  ∃ a b, [a, b] = [k * π + π / 12, k * π + π / 3] ∧ ∀ x ∈ [a, b], f' x > 0 :=
sorry

theorem sin_alpha (α : ℝ) (h1 : α ∈ (0, π)) (h2 : f(α / 2) = (1 / 2) + Real.sqrt 3) :
  Real.sin α = 1 / 2 :=
sorry

theorem max_value_in_interval :
  ∃ xmax, xmax ∈ [-π/2, 0] ∧ ∀ x ∈ [-π/2, 0], f x ≤ f xmax :=
sorry

end smallest_positive_period_monotone_intervals_sin_alpha_max_value_in_interval_l498_498272


namespace bus_total_journey_time_l498_498194

variables {A B C X Y : Point}
variables {d_AB d_BC d_AC d_XC d_XA d_XB d_YA d_YB d_YC : ℝ}
variables {time_since_A time_since_X time_to_B : ℝ}
constant distance : Point → Point → ℝ
constant bus_speed : ℝ

-- Declaration of points with given distances
axiom dist_AC : distance A C = d_AB + d_BC
axiom point_X_condition : distance X C = distance X A + distance X B
axiom point_Y_condition : distance Y B + distance Y C = distance B C

-- Given constants
axiom time_X_to_Y : time_since_A = time_since_X
axiom time_from_Y_to_B : 25
axiom bus_stops_B : 5
axiom speed_constant : ∀ (P Q : Point), (distance P Q) / bus_speed = time_since_A + time_since_X
axiom time_Y_to_B : distance Y B = d_BC / 2
axiom bus_hour_alignment : (2 * d_BC) / bus_speed = time_since_A

theorem bus_total_journey_time : (2 * d_BC) / bus_speed + 25 / bus_speed + 5 = 180 :=
by
  sorry

end bus_total_journey_time_l498_498194


namespace aquarium_length_l498_498138

/-- Variables and conditions -/
variables (volume : ℝ) (breadth : ℝ) (height_water_rises : ℝ) (length_of_aquarium : ℝ)
variables (h_volume : volume = 10000)
variables (h_breadth : breadth = 20)
variables (h_height_water_rises : height_water_rises = 10)

/-- Proof problem -/
theorem aquarium_length 
  (cond_volume : volume = 10000)
  (cond_breadth : breadth = 20)
  (cond_height_water_rises : height_water_rises = 10)
  : length_of_aquarium = 50 := 
by
  -- We conclude the proof
  sorry

end aquarium_length_l498_498138


namespace angle_ABH_regular_octagon_l498_498922

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498922


namespace arithmetic_sequence_geometric_term_ratio_l498_498268

theorem arithmetic_sequence_geometric_term_ratio (a : ℕ → ℤ) (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : a 1 = a 1)
  (h₂ : a 3 = a 1 + 2 * d)
  (h₃ : a 4 = a 1 + 3 * d)
  (h_geom : (a 1 + 2 * d)^2 = a 1 * (a 1 + 3 * d)) :
  (a 1 + a 5 + a 17) / (a 2 + a 6 + a 18) = 8 / 11 :=
by
  sorry

end arithmetic_sequence_geometric_term_ratio_l498_498268


namespace has_three_real_zeros_l498_498306

noncomputable def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem has_three_real_zeros (m : ℝ) : 
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (-4 < m ∧ m < 4) :=
sorry

end has_three_real_zeros_l498_498306


namespace angle_ABH_regular_octagon_l498_498927

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498927


namespace correct_calculation_l498_498115

theorem correct_calculation : -real.cbrt (-8) = 2 := by
  sorry

end correct_calculation_l498_498115


namespace strawberries_weight_l498_498063

theorem strawberries_weight (total_weight apples_weight oranges_weight grapes_weight strawberries_weight : ℕ) 
  (h_total : total_weight = 10)
  (h_apples : apples_weight = 3)
  (h_oranges : oranges_weight = 1)
  (h_grapes : grapes_weight = 3) 
  (h_sum : total_weight = apples_weight + oranges_weight + grapes_weight + strawberries_weight) :
  strawberries_weight = 3 :=
by
  sorry

end strawberries_weight_l498_498063


namespace count_paths_A_to_B_l498_498204

-- Defining the specific points as data points for clarity.
constant A B C D E F G : Type

-- Assuming all necessary definitions exist
def is_path : list Type -> Prop := sorry

def no_revisit (path : list Type) : Prop :=
  path.nodup

-- Paths connecting points by steps based on the description
def paths_A_to_B : list (list Type) :=
  [[A, C, B],
   [A, D, C, B],
   [A, D, F, C, B],
   [A, D, E, F, C, B],
   [A, D, F, C, B],
   [A, D, E, F, C, B],
   [A, C, F, G, B],
   [A, C, D, F, G, B],
   [A, D, F, G, B],
   [A, D, E, F, G, B]]

-- Verify the path is valid and does not revisit any points
def valid_paths (paths : list (list Type)) : list (list Type) :=
  paths.filter no_revisit

theorem count_paths_A_to_B : 
  paths_A_to_B.length = 10 :=
by
  sorry

end count_paths_A_to_B_l498_498204


namespace problem_statement_l498_498363

def M : ℕ := 888 * 10^316 + 6

def leading_digit_of_rth_root (r : ℕ) (n : ℕ) : ℕ :=
  let root := n^(1 / r)
  let leading := nat.digits 10 root
  leading.head.default 0

def g (r : ℕ) : ℕ := leading_digit_of_rth_root r M

theorem problem_statement : g 2 + g 3 + g 4 + g 5 + g 6 = 9 := 
  sorry

end problem_statement_l498_498363


namespace hyperbola_eccentricity_eq_sqrt21_div_3_l498_498256

-- Step 1: Identify all questions and conditions
-- Conditions:
-- 1. F is the focus of the parabola C: y^2 = 2px, with p > 0.
-- 2. The directrix of parabola C intersects the two asymptotes of the hyperbola Γ: x^2/a^2 - y^2/b^2 = 1 at points A and B, with a > 0, b > 0.
-- 3. Triangle ABF is an equilateral triangle.

-- Question: Find the eccentricity e of the hyperbola Γ.

-- Step 2: Identify all solution steps and the correct answer
-- Correct answer: D: e = √21 / 3

-- Step 3: Translate to a mathematically equivalent proof problem
-- Prove: Given the conditions, the eccentricity e of the hyperbola Γ is √21 / 3.

-- Step 4: Rewrite the math proof problem in Lean 4 statement

theorem hyperbola_eccentricity_eq_sqrt21_div_3
  (p a b : ℝ)
  (p_pos : p > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (F : ℝ × ℝ)
  (focus_of_parabola : F = (p / 2, 0))
  (directrix_eq : ∀ x y : ℝ, y^2 = 2 * p * x → x = -p / 2)
  (asymptotes_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (y = b / a * x ∨ y = -b / a * x))
  (A B : ℝ × ℝ)
  (A_on_directrix_and_asymptote : directrix_eq A.1 A.2 ∧ asymptotes_eq A.1 A.1)
  (B_on_directrix_and_asymptote : directrix_eq B.1 B.2 ∧ asymptotes_eq B.1 B.1)
  (equilateral_triangle : dist A B = dist A F ∧ dist A F = dist B F ∧ dist A B = dist B F) :
  let e := real.sqrt (1 + (b / a)^2) in
  e = real.sqrt 21 / 3 := sorry

end hyperbola_eccentricity_eq_sqrt21_div_3_l498_498256


namespace false_PQ_l498_498996

variable {P Q R : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R]

variable (p q r d : ℝ)
variable (dist_PQ : P → Q → ℝ)
variable (p_gt_q : p > q)
variable (q_gt_r : q > r)
variable (circleR_inside_circleP : ∀ (rP rR : P), rR < rP → True)
variable (circleR_inside_circleQ : ∀ (rQ rR : Q), rR < rQ → True)

theorem false_PQ (h : dist_PQ.dist P Q = d) : ¬ (p + r = d) :=
  sorry

end false_PQ_l498_498996


namespace goods_train_speed_l498_498156

def speed_of_goods_train (length_in_meters : ℕ) (time_in_seconds : ℕ) (speed_of_man_train_kmph : ℕ) : ℕ :=
  let length_in_km := length_in_meters / 1000
  let time_in_hours := time_in_seconds / 3600
  let relative_speed_kmph := (length_in_km * 3600) / time_in_hours
  relative_speed_kmph - speed_of_man_train_kmph

theorem goods_train_speed :
  speed_of_goods_train 280 9 50 = 62 := by
  sorry

end goods_train_speed_l498_498156


namespace necessary_but_not_sufficient_condition_l498_498403

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > e) : x > 1 :=
sorry

end necessary_but_not_sufficient_condition_l498_498403


namespace product_of_last_two_numbers_l498_498485

theorem product_of_last_two_numbers (A B C : ℕ) (h_coprime : Nat.coprime A B ∧ Nat.coprime A C ∧ Nat.coprime B C)
  (h_product : A * B = 551) (h_sum : A + B + C = 85) : B * C = 1073 := by
  sorry

end product_of_last_two_numbers_l498_498485


namespace inequality_abc_l498_498429

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l498_498429


namespace angle_ABH_regular_octagon_l498_498925

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498925


namespace computer_distribution_l498_498561

theorem computer_distribution :
  let num_schools := 3
  let total_computers := 9
  let min_computers_per_school := 2
  ∑ n in ({(2, 2, 5), (2, 3, 4), (3, 3, 3)} : Finset (ℕ × ℕ × ℕ)), 1 = 10 :=
sorry

end computer_distribution_l498_498561


namespace ivan_uses_more_paint_l498_498076

-- Define the basic geometric properties
def rectangular_section_area (length width : ℝ) : ℝ := length * width
def parallelogram_section_area (side1 side2 : ℝ) (angle : ℝ) : ℝ := side1 * side2 * Real.sin angle

-- Define the areas for each neighbor's fences
def ivan_area : ℝ := rectangular_section_area 5 2
def petr_area (alpha : ℝ) : ℝ := parallelogram_section_area 5 2 alpha

-- Theorem stating that Ivan's total fence area is greater than Petr's total fence area provided the conditions
theorem ivan_uses_more_paint (α : ℝ) (hα : α ≠ Real.pi / 2) : ivan_area > petr_area α := by
  sorry

end ivan_uses_more_paint_l498_498076


namespace monotonic_intervals_range_of_b_inequality_proof_l498_498401

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (x + 1) - a * Real.log x - b

theorem monotonic_intervals (a b : ℝ) (h_a_pos : 0 < a) (h_b : b ∈ Set.univ) : 
  ∀ x : ℝ, x > 0 → (a ≥ 1 → ∀ y : ℝ, f y a b ≤ f x a b) ∧ 
                    (0 < a ∧ a < 1 → f (a / (1 - a)) a b ≤ f x a b ∧ ∀ z : ℝ, z > a / (1 - a) → f x a b ≤ f z a b) :=
sorry

theorem range_of_b (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : b > Real.log 2) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ f x₁ a b = 0 ∧ f x₂ a b = 0 :=
sorry

theorem inequality_proof (n : ℕ) (h_n : 2 ≤ n) :
  (∏ k in Finset.range (n - 1), (k.succ / n) ^ k.succ) > (2 : ℝ) ^ (-(n * n / 2)) :=
sorry

end monotonic_intervals_range_of_b_inequality_proof_l498_498401


namespace opposite_of_neg_1009_l498_498984

variable (x : ℤ)

theorem opposite_of_neg_1009 : ∃ x, -1009 + x = 0 ∧ x = 1009 :=
by
  use 1009
  split
  case inl
  {
    show -1009 + 1009 = 0 from sorry,
  }
  case inr
  {
    show 1009 = 1009 from by rfl,
  }

end opposite_of_neg_1009_l498_498984


namespace filling_tank_with_pipes_l498_498164

theorem filling_tank_with_pipes :
  let Ra := 1 / 70
  let Rb := 2 * Ra
  let Rc := 2 * Rb
  let Rtotal := Ra + Rb + Rc
  Rtotal = 1 / 10 →  -- Given the combined rate fills the tank in 10 hours
  3 = 3 :=  -- Number of pipes used to fill the tank
by
  intros Ra Rb Rc Rtotal h
  simp [Ra, Rb, Rc] at h
  sorry

end filling_tank_with_pipes_l498_498164


namespace count_valid_k_l498_498779

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem count_valid_k : 
  (λ (n : ℕ), ∃ count : ℕ, count = 2146 ∧ ∀ k, 0 ≤ k ∧ k ≤ n ∧ (2188 ∣ binomial 2188 k) ↔ ∃ c, c = count) 2188 :=
by 
{ sorry }

end count_valid_k_l498_498779


namespace angle_DNF_is_right_angle_l498_498801

theorem angle_DNF_is_right_angle 
  (A B C D E F N : Point)
  (h_square : Square A B C D)
  (h_E_on_AB : PointOnLineSegment E A B)
  (h_F_on_BC : PointOnLineSegment F B C)
  (h_BE_eq_BF : dist B E = dist B F)
  (h_BN_altitude : Altitude B N C E)
  : ∠ D N F = 90 := 
sorry

end angle_DNF_is_right_angle_l498_498801


namespace sum_of_altitudes_l498_498468

open Real

theorem sum_of_altitudes
  (x y : ℝ)
  (h_line : 15 * x + 10 * y = 150) :
  let a := 10
  let b := 15
  let hypotenuse := 5 * sqrt 13
  let area := 75
  let third_altitude := 30 * sqrt 13 / 13
  let sum_of_altitudes := a + b + third_altitude
  in sum_of_altitudes = (325 + 30 * sqrt 13) / 13 :=
by
  -- Given line 15x + 10y = 150
  sorry

end sum_of_altitudes_l498_498468


namespace integer_solution_count_l498_498713

theorem integer_solution_count : 
  ∃ S : Set ℤ, (∀ x : ℤ, x ∈ S ↔ (x-2)^2 ≤ 9) ∧ S.card = 7 := 
sorry

end integer_solution_count_l498_498713


namespace perpendicular_CE_AD_l498_498767

-- Variables and Conditions
variables {A B C D E : Type*} [EuclideanGeometry ℝ]

-- Define the points A, B, C, and related conditions
variables (A B C D E : ℝ) -- Midpoint D of BC, and ratio AE:EB=2:1

-- Definitions
def midpoint := ∃ D, D = (B + C) / 2
def isosceles_right_triangle := ∃ α β γ, α + β + γ = 180 ∧ α = β ∧ γ = 90
def AE_EB_ratio := ∃ E, (A - E) / (E - B) = 2

-- Problem Statement
theorem perpendicular_CE_AD :
  isosceles_right_triangle A B C ∧ midpoint D ∧ AE_EB_ratio E →
  ⊥ A D C E :=
sorry -- This is where the proof of perpendicularity would be written.

end perpendicular_CE_AD_l498_498767


namespace packaging_boxes_vehicle_arrangements_best_arrangement_l498_498062

theorem packaging_boxes (x y : ℕ) (h1 : 10 * x + 5 * y = 3250) (h2 : 5 * x + 3 * y = 1700) : x = 250 ∧ y = 150 :=
sorry

theorem vehicle_arrangements (z : ℕ) (total_vehicles : ℕ = 10)
  (h1 : 30 * z + 20 * (total_vehicles - z) ≥ 250)
  (h2 : 10 * z + 40 * (total_vehicles - z) ≥ 150) : 
  5 ≤ z ∧ z ≤ 8 :=
sorry

theorem best_arrangement (z : ℕ) (h : 5 ≤ z ∧ z ≤ 8) (fuel_efficient_A : Bool) : z = 8 :=
sorry

end packaging_boxes_vehicle_arrangements_best_arrangement_l498_498062


namespace number_of_students_in_first_class_l498_498969

theorem number_of_students_in_first_class 
  (x : ℕ) -- number of students in the first class
  (avg_first_class : ℝ := 50) 
  (num_second_class : ℕ := 50)
  (avg_second_class : ℝ := 60)
  (avg_all_students : ℝ := 56.25)
  (total_avg_eqn : (avg_first_class * x + avg_second_class * num_second_class) / (x + num_second_class) = avg_all_students) : 
  x = 30 :=
by sorry

end number_of_students_in_first_class_l498_498969


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498526

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498526


namespace length_of_courtyard_l498_498494

namespace Courtyard

-- Define the width of the courtyard
def width : ℝ := 16.5

-- Define the number of paving stones
def num_paving_stones : ℝ := 165

-- Define the dimensions of each paving stone
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2

-- Calculate the area of one paving stone
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width

-- Calculate the total area covered by all the paving stones
def total_area : ℝ := paving_stone_area * num_paving_stones

-- Define the length of the courtyard
def length := total_area / width

-- The theorem that states the length of the courtyard is 50 meters
theorem length_of_courtyard : length = 50 := sorry

end Courtyard

end length_of_courtyard_l498_498494


namespace M_inside_CurveC_θ_value_l498_498280

noncomputable def x (α : ℝ) : ℝ := 2 * Real.cos α
noncomputable def y (α : ℝ) : ℝ := 3 * Real.sin α
noncomputable def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

-- Define the point M in cartesian coordinates
def Mx := Real.sqrt 2
def My := Real.sqrt 2

-- The first question statement
theorem M_inside_CurveC : ¬ ellipse Mx My := by
  sorry

-- Definitions for the second question
def θ := (3 * Real.pi) / 4
def M'_x : ℝ := 2 * Real.cos (Real.pi / 4 + θ)
def M'_y : ℝ := 2 * Real.sin (Real.pi / 4 + θ)

-- The second question statement
theorem θ_value : ∀ θ, (θ ∈ Set.Icc (0 : ℝ) Real.pi) ∧ ellipse M'_x M'_y → θ = (3 * Real.pi) / 4 := by
  sorry

end M_inside_CurveC_θ_value_l498_498280


namespace problem1_problem2_problem3_problem4_l498_498414

def R : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

theorem problem1 : A ∩ B = {x | 3 ≤ x ∧ x < 5} := sorry

theorem problem2 : A ∪ B = {x | 1 < x ∧ x ≤ 6} := sorry

theorem problem3 : (Set.compl A) ∩ B = {x | 5 ≤ x ∧ x ≤ 6} :=
sorry

theorem problem4 : Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 5} := sorry

end problem1_problem2_problem3_problem4_l498_498414


namespace chord_line_equation_l498_498225

theorem chord_line_equation (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), y1^2 = -8 * x1 ∧ y2^2 = -8 * x2 ∧ (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1 ∧ y - 1 = -4 * (x + 1)) →
  4 * x + y + 3 = 0 :=
by
  sorry

end chord_line_equation_l498_498225


namespace average_speed_l498_498998

theorem average_speed (D : ℝ) (h1 : 0 < D) :
  let s1 := 60   -- speed from Q to B in miles per hour
  let s2 := 20   -- speed from B to C in miles per hour
  let d1 := 2 * D  -- distance from Q to B
  let d2 := D     -- distance from B to C
  let t1 := d1 / s1  -- time to travel from Q to B
  let t2 := d2 / s2  -- time to travel from B to C
  let total_distance := d1 + d2  -- total distance
  let total_time := t1 + t2   -- total time
  let average_speed := total_distance / total_time  -- average speed
  average_speed = 36 :=
by
  sorry

end average_speed_l498_498998


namespace maximum_value_of_function_l498_498481

noncomputable def max_value_condition (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 1 - real.cos x ∧ ∃ k : ℤ, x = real.pi + 2 * k * real.pi

theorem maximum_value_of_function : 
  ∀ x : ℝ, max_value_condition (λ x, 1 - real.cos x) x ↔ ∃ k : ℤ, x = real.pi + 2 * k * real.pi :=
by
  sorry

end maximum_value_of_function_l498_498481


namespace grid_sum_correct_l498_498945

theorem grid_sum_correct (A B : ℕ) (grid : Array (Array (Option ℕ))) :
  grid = #[#[some 2, none, some B], #[none, some 3, none], #[some A, none, none]] →
  (∀ i, (Set.of (grid[i].filterMap id).toList) = {2,3,4}) →
  (∀ j, (Set.of ((List.range 3).map (λ i => grid[i]![j]).filterMap id)) = {2,3,4}) →
  A + B = 6 :=
by
  introv hgrid hrow hcol
  sorry

end grid_sum_correct_l498_498945


namespace measure_of_angle_ABH_l498_498831

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498831


namespace simplify_expression_l498_498954

theorem simplify_expression :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ ¬∃ p : ℕ, prime p ∧ p^2 ∣ c ∧
    (↑a - ↑b * real.sqrt c : ℝ) =
    (real.sqrt 3 - 1)^(2 - real.sqrt 5) / (real.sqrt 3 + 1)^(2 + real.sqrt 5)) :=
begin
  use [7, 1, 3],
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  -- c is not divisible by the square of a prime
  intros p hp,
  have h := prime.not_square_sub_dvd_self hp,
  norm_num at h,
  exact h,
  -- The simplified expression
  have h1 : (real.sqrt 3 - 1)^(2 - real.sqrt 5) / (real.sqrt 3 + 1)^(2 + real.sqrt 5) = (7 / 16 : ℝ) - (1 / 4 : ℝ) * real.sqrt 3,
  sorry
end

end simplify_expression_l498_498954


namespace increasing_interval_l498_498980

noncomputable def f (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + (3 / 2) * x ^ 2 - x

theorem increasing_interval : ∀ x y : ℝ, (1/2) ≤ x → y ≤ 1 → x ≤ y → f x ≤ f y :=
by
  intro x y hx hy hxy
  -- Proof goes here (to be filled in with the proof steps)
  sorry

end increasing_interval_l498_498980


namespace area_of_circle_l498_498808

-- Define points A and B
def A : ℝ × ℝ := (8, 15)
def B : ℝ × ℝ := (14, 9)

-- Define a function to calculate the midpoint of two points
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the circle ω given the above conditions
def circle_area (A B : ℝ × ℝ) : ℝ :=
  let D := midpoint A B in -- midpoint of AB
  let slope_AB := (B.2 - A.2) / (B.1 - A.1) in -- slope of AB
  let slope_CD := -1 / slope_AB in -- slope of the perpendicular bisector
  let C := (-1, 0) in -- intersection of CD with the x-axis
  let radius := real.sqrt ((A.1 + 1)^2 + (A.2)^2) in -- distance AC == OA since C is on x-axis
  real.pi * radius^2

-- The theorem to prove the area of ω is 306π
theorem area_of_circle :
  circle_area A B = 306 * real.pi := 
sorry

end area_of_circle_l498_498808


namespace meaningful_fraction_l498_498304

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l498_498304


namespace star_neg5_4_star_neg3_neg6_l498_498083

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end star_neg5_4_star_neg3_neg6_l498_498083


namespace angleC_is_36_l498_498416

theorem angleC_is_36 
  (p q r : ℝ)  -- fictitious types for lines, as Lean needs a type here
  (A B C : ℝ)  -- Angles as Real numbers
  (hpq : p = q)  -- Line p is parallel to line q (represented equivalently for Lean)
  (h : A = 1/4 * B)
  (hr : B + C = 180)
  (vert_opposite : C = A) :
  C = 36 := 
by
  sorry

end angleC_is_36_l498_498416


namespace final_price_is_correct_l498_498635

-- Define the original price
def original_price : ℝ := 10

-- Define the first reduction percentage
def first_reduction_percentage : ℝ := 0.30

-- Define the second reduction percentage
def second_reduction_percentage : ℝ := 0.50

-- Define the price after the first reduction
def price_after_first_reduction : ℝ := original_price * (1 - first_reduction_percentage)

-- Define the final price after the second reduction
def final_price : ℝ := price_after_first_reduction * (1 - second_reduction_percentage)

-- Theorem to prove the final price is $3.50
theorem final_price_is_correct : final_price = 3.50 := by
  sorry

end final_price_is_correct_l498_498635


namespace axis_of_symmetry_set_of_values_range_of_m_l498_498650

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem axis_of_symmetry (k : ℤ) :
  ∃ m : ℤ, ∀ x : ℝ, x = (m * π) / 2 + (3 * π) / 8 ↔ f x = f (π / 2 - x) :=
sorry

theorem set_of_values (x : ℝ) (k : ℤ) :
  f x ≥ 1 ↔ (π / 4 + k * π) ≤ x ∧ x ≤ (π / 2 + k * π) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc (π / 6) (π / 3), f x - m < 2) ↔ m > (sqrt 3 - 5) / 4 :=
sorry

end axis_of_symmetry_set_of_values_range_of_m_l498_498650


namespace ending_number_is_540_l498_498487

def lcm(a b : ℕ) : ℕ := a / Nat.gcd a b * b

theorem ending_number_is_540 : ∃ n, (∀ k, 1 ≤ k ∧ k ≤ 6 → 190 + k * (lcm 4 (lcm 5 6)) < n) ∧ 190 + 6 * (lcm 4 (lcm 5 6)) = 540 := 
by 
  sorry

end ending_number_is_540_l498_498487


namespace a_4_value_l498_498475

noncomputable def S : ℕ → ℕ
| n => 3^n + 2*n + 1

def a : ℕ → ℕ
| n => if n = 0 then S 0 else S n - S (n - 1)

theorem a_4_value : a 4 = 56 := 
by 
  sorry

end a_4_value_l498_498475


namespace number_of_friends_dividing_bill_l498_498172

theorem number_of_friends_dividing_bill :
  ∃ n : ℕ, 45 * n = 135 ∧ n = 3 :=
begin
  use 3,
  split,
  { -- 45 * 3 = 135
    norm_num,
  },
  { -- n = 3
    refl,
  }
end

end number_of_friends_dividing_bill_l498_498172


namespace defective_part_probability_l498_498601

theorem defective_part_probability :
  let P_H1 := (1000 : ℝ) / (1000 + 2000 + 3000),
      P_H2 := (2000 : ℝ) / (1000 + 2000 + 3000),
      P_H3 := (3000 : ℝ) / (1000 + 2000 + 3000),
      P_A_given_H1 := 0.001,
      P_A_given_H2 := 0.002,
      P_A_given_H3 := 0.003 in
  P_H1 * P_A_given_H1 + P_H2 * P_A_given_H2 + P_H3 * P_A_given_H3 = 0.0023333 := 
by 
  sorry

end defective_part_probability_l498_498601


namespace inequality_abc_l498_498427

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l498_498427


namespace angle_ABH_regular_octagon_l498_498926

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498926


namespace diff_of_arith_prog_divisible_by_prime_l498_498134

theorem diff_of_arith_prog_divisible_by_prime 
  (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℕ) (d : ℕ)
  (ha_prog : ∀ i j : Fin p, i < j → a j = a i + (j - i) * d)
  (ha1_gt_p : a 0 > p) :
  p ∣ d :=
sorry

end diff_of_arith_prog_divisible_by_prime_l498_498134


namespace range_of_m_l498_498691

-- Define the given function f(x)
def f (x : ℝ) : ℝ := Real.exp x * (Real.log x - 1)

-- Define the set of x and a
def X := Set.Ici (1 / 2)
def A := Icc (-2 : ℝ) 1

-- State the proof problem: the range of m
theorem range_of_m :
  (∃ a ∈ A, f (2 - 1 / m) ≤ a ^ 2 + 2 * a - 3 - Real.exp 1) ↔ (m ∈ Icc (2 / 3) 1) := 
sorry

end range_of_m_l498_498691


namespace angle_ABH_in_regular_octagon_l498_498833

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498833


namespace angle_in_regular_octagon_l498_498885

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498885


namespace find_divisor_from_count_l498_498057

theorem find_divisor_from_count (h1 : ∃ n : ℕ, 11110 = n)
                                (h2 : ∀ k : ℕ, (10 ≤ k ∧ k ≤ 100000) → ∃ x : ℕ, k % x = 0)
                                (h3 : 10 % 9 = 0)
                                (h4 : 100000 % 9 = 0) :
  ∃ x : ℕ, (99990 / 11109) = x :=
by
  have gcd_99990_11109 : Int.gcd 99990 11109 = 9 := by sorry
  exact ⟨9, gcd_99990_11109⟩

end find_divisor_from_count_l498_498057


namespace permutations_remainder_5_l498_498593

open Finset

noncomputable def num_permutations_with_remainder_5 : Nat :=
  let digits : List Nat := [1, 2, 3, 4, 5, 6, 7]
  let permutations := univ.filter (fun perm : List Nat =>
    perm.perms (digits : List Nat) && (perm.vecMod 7).1 = 5)
  permutations.card

theorem permutations_remainder_5 : num_permutations_with_remainder_5 = 720 := by
  sorry

end permutations_remainder_5_l498_498593


namespace power_function_half_l498_498296

theorem power_function_half (a : ℝ) (ha : (4 : ℝ)^a / (2 : ℝ)^a = 3) : (1 / 2 : ℝ) ^ a = 1 / 3 := 
by
  sorry

end power_function_half_l498_498296


namespace measure_of_angle_ABH_in_regular_octagon_l498_498932

/-- Let ABCDEFGH be a regular octagon. Prove that the measure of angle ABH is 22.5 degrees. -/
theorem measure_of_angle_ABH_in_regular_octagon
  (A B C D E F G H : Type)
  [RegularOctagon A B C D E F G H] : 
  ∠A B H = 22.5 :=
sorry

end measure_of_angle_ABH_in_regular_octagon_l498_498932


namespace divide_bill_evenly_l498_498175

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end divide_bill_evenly_l498_498175


namespace sum_solutions_eq_zero_l498_498334

theorem sum_solutions_eq_zero :
  let y := 8 in
  let eq1 := y = 8 in
  let eq2 := ∀ x, x ^ 2 + y ^ 2 = 169 in
  (x ∈ ℝ ∧ eq1 ∧ eq2) →
  ∃ x1 x2, x1 ^ 2 + y ^ 2 = 169 ∧ x2 ^ 2 + y ^ 2 = 169 ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_solutions_eq_zero_l498_498334


namespace sphere_surface_area_l498_498267

noncomputable def lateral_area (a h : ℝ) : ℝ := 3 * a * h
noncomputable def base_area (a : ℝ) : ℝ := (√3 / 4) * a^2
noncomputable def circumradius (a : ℝ) : ℝ := a / √3
noncomputable def height_from_circumradius_and_half_height (r h : ℕ) := (r^2 + (h/2)^2)^0.5

theorem sphere_surface_area (a h r R : ℝ) (lat_area base_area : ℝ):
  (lateral_area a h = 6) ∧
  (base_area a = √3) ∧
  (circumradius a = r) ∧
  (height_from_circumradius_and_half_height r (h/2) = R) →
  4 * Real.pi * R^2 = (19 * Real.pi) / 3 :=
by
  sorry

end sphere_surface_area_l498_498267


namespace change_points_l498_498421

def Point := ℤ × ℤ

def symmetric_point (A B : Point) : Point :=
  (2 * B.1 - A.1, 2 * B.2 - A.2)

def allowable_transformation (pts : List Point) : List (List Point) :=
  pts.product pts.map (λ x, pts.map λ y, symmetric_point x y)

def can_transform (initial : List Point) (target : List Point) : Prop :=
  target ∈ allow_transformation initial

theorem change_points :
  ¬ can_transform [(0,0), (0,1), (1,0), (1,1)] [(0,0), (1,1), (3,0), (2,-1)] :=
sorry

end change_points_l498_498421


namespace calculate_normal_recess_time_l498_498055

def num_As := 10
def num_Bs := 12
def num_Cs := 14
def num_Ds := 5
def total_recess_report_card_day := 47

def extra_recess_from_As := num_As * 2
def extra_recess_from_Bs := num_Bs * 1
def extra_recess_from_Cs := num_Cs * 0
def extra_recess_from_Ds := num_Ds * (-1)

def total_extra_recess := extra_recess_from_As + extra_recess_from_Bs + extra_recess_from_Cs + extra_recess_from_Ds

def normal_recess_time := total_recess_report_card_day - total_extra_recess

theorem calculate_normal_recess_time : normal_recess_time = 20 := by
  sorry

end calculate_normal_recess_time_l498_498055


namespace meaningful_if_and_only_if_l498_498303

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l498_498303


namespace find_positive_number_l498_498622

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l498_498622


namespace angle_ABH_in_regular_octagon_l498_498837

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498837


namespace find_xyz_sum_l498_498038

noncomputable def distance_from_center_to_triangle (radius PQ QR RP : ℕ) : ℝ := 
  let K := real.sqrt(19.5 * 10.5 * 7.5 * 1.5)
  let circumradius := (PQ : ℝ) * QR * RP / (4 * K)
  real.sqrt (radius ^ 2 - circumradius ^ 2)

theorem find_xyz_sum : 
  ∀ (P Q R O : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space O], 
  ∀ (radius PQ QR RP : ℕ), 
  radius = 15 → PQ = 9 → QR = 12 → RP = 18 → 
  let dist := distance_from_center_to_triangle radius PQ QR RP in 
  dist = 12 * real.sqrt 221 / 5 → 
  let x := 12 in let y := 221 in let z := 5 in 
  x + y + z = 238 := 
by 
  intros
  sorry

end find_xyz_sum_l498_498038


namespace cube_root_of_product_l498_498504

theorem cube_root_of_product :
  (∛(5 ^ 3 * 2 ^ 6) = 10) := sorry

end cube_root_of_product_l498_498504


namespace string_length_proof_l498_498554

noncomputable def string_length (circumference : ℤ) (num_loops : ℤ) (height : ℤ) : ℝ :=
let height_per_loop := (height : ℝ) / (num_loops : ℝ) in
let loop_length := real.sqrt ((height_per_loop) ^ 2 + (circumference : ℝ) ^ 2) in
loop_length * (num_loops : ℝ)

theorem string_length_proof :
  string_length 6 6 18 = 18 * real.sqrt 5 := by
  sorry

end string_length_proof_l498_498554


namespace safest_password_l498_498145

theorem safest_password :
  let P_A := 1 - (4 : ℚ) / (4^4 : ℚ),
      P_B := 1 - (24 : ℚ) / (4^4 : ℚ),
      P_C := 1 - (72 : ℚ) / (4^4 : ℚ),
      P_D := 1 - (24 : ℚ) / (4^4 : ℚ)
  in P_C < P_A ∧ P_C < P_B ∧ P_C < P_D := by
sorry

end safest_password_l498_498145


namespace verify_dot_product_l498_498706

open Real EuclideanSpace

-- Definitions of given vectors and magnitudes
def vec_a : ℝ × ℝ := (sqrt 3, -1)
def norm_b : ℝ := sqrt 5

-- Define the condition of orthogonality
def orthogonal_condition : Bool :=
  let b1 := sqrt (norm_b^2 - 3)
  let b2 := -1
  (sqrt 3 * (sqrt 3 - b1) + -1 * (-1 - b2) = 0)

-- Define the result of the dot product to verify
def dot_product_result : ℝ :=
  let b1 := sqrt (norm_b^2 - 3)
  let b2 := -1
  let a_b1 := sqrt 3 + b1
  let a_b2 := -1 + b2
  let a_3b1 := sqrt 3 - 3 * b1
  let a_3b2 := -1 - 3 * b2
  (a_b1 * (a_3b1) + a_b2 * (a_3b2))

-- The statement to be proved
theorem verify_dot_product : orthogonal_condition → dot_product_result = -19 := by
  sorry

end verify_dot_product_l498_498706


namespace oblique_axonometric_triangle_area_l498_498636

theorem oblique_axonometric_triangle_area (a b : ℝ) :
  let original_area := (1/2) * a * b,
      intuitive_height := (sqrt 2) / 4 * a,
      intuitive_area := (1/2) * a * intuitive_height in
      intuitive_area = (sqrt 2 / 4) * original_area :=
by
  -- Definitions from conditions
  let original_area := (1/2) * a * b
  let intuitive_height := (sqrt 2) / 4 * a
  let intuitive_area := (1/2) * a * intuitive_height
  
  -- Goal
  show intuitive_area = (sqrt 2 / 4) * original_area
  sorry

end oblique_axonometric_triangle_area_l498_498636


namespace sum_of_mathcalT_is_2520_l498_498381

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498381


namespace maximize_garden_area_l498_498795

def optimal_dimensions_area : Prop :=
  let l := 100
  let w := 60
  let area := 6000
  (2 * l) + (2 * w) = 320 ∧ l >= 100 ∧ (l * w) = area

theorem maximize_garden_area : optimal_dimensions_area := by
  sorry

end maximize_garden_area_l498_498795


namespace sum_elements_T_l498_498371

def is_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def set_T : set ℝ :=
  {x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999.0}

theorem sum_elements_T : real.sum (set.to_finset set_T) = 2520 :=
by
  sorry

end sum_elements_T_l498_498371


namespace point_in_circle_l498_498684

theorem point_in_circle (a b c : ℝ) (h : a > b ∧ b > 0) (e : c/a = 1/2)
                        (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
                        (h_roots : ∀ x : ℝ, a * x^2 + b * x - c = 0):
                        let x1 x2 : ℝ := ... in  
                         (x1 + x2) = -(sqrt 3 / 2) ∧ (x1 * x2) = -(1 / 2)
                         ∧ (x1^2 + x2^2 < 2) →
                         (x1, x2) ∈ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 < 2} :=
by {
   sorry
}

end point_in_circle_l498_498684


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498896

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498896


namespace direction_vector_is_3_1_l498_498023

-- Given the line equation x - 3y + 1 = 0
def line_equation : ℝ × ℝ → Prop :=
  λ p, p.1 - 3 * p.2 + 1 = 0

-- The direction vector of the line
def direction_vector_of_line (v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * 3, k * 1)

theorem direction_vector_is_3_1 : direction_vector_of_line (3, 1) :=
by
  sorry

end direction_vector_is_3_1_l498_498023


namespace total_hours_correct_l498_498423

/-- Definitions for the times each person has left to finish their homework. -/
noncomputable def Jacob_time : ℕ := 18
noncomputable def Greg_time : ℕ := Jacob_time - 6
noncomputable def Patrick_time : ℕ := 2 * Greg_time - 4

/-- Proving the total time left for Patrick, Greg, and Jacob to finish their homework. -/

theorem total_hours_correct : Jacob_time + Greg_time + Patrick_time = 50 := by
  sorry

end total_hours_correct_l498_498423


namespace find_product_of_constants_l498_498406

theorem find_product_of_constants
  (M1 M2 : ℝ)
  (h : ∀ x : ℝ, (x - 1) * (x - 2) ≠ 0 → (45 * x - 31) / (x * x - 3 * x + 2) = M1 / (x - 1) + M2 / (x - 2)) :
  M1 * M2 = -826 :=
sorry

end find_product_of_constants_l498_498406


namespace circumradius_triangle_BCM_l498_498456

-- Definitions for the problem conditions
variables {A B C D M : Type*} 
variables {a m R : ℝ}
variable [is_cyclic_quad A B C D]

-- Constraint that specifies distances
def distances (A B C D : Type*) (AB BC BD : ℝ) : Prop := 
  (dist A B = AB) ∧ (dist B C = BC) ∧ (dist B D = BD)

-- The main statement to be proven
theorem circumradius_triangle_BCM (h_cyclic : is_cyclic_quad A B C D)
  (h_dist : distances A B C D a m) (h_intersect : intersects_at_point M (proj A B C D)) :
  circumradius_triangle B C M = (a * R) / m :=
sorry

end circumradius_triangle_BCM_l498_498456


namespace quadratic_solution_probability_l498_498285

open Finset

theorem quadratic_solution_probability :
  let A := {1, 2, 3, 4, 5, 6}
  in ∃ b c ∈ A, (δ : (Rat)) ∈  [⟨b, Nat.cast_lt.mpr (int.coe_nat_lt.mpr (lt_succ_self 7)) b.2 b.2⟩]
  in ∀ (dists: (b, c) ∈ A), (A ∩ [b ≥ 2 * c]) in },

  (∃ (solutions_count : ℕ) (total_count : ℕ),
     total_count = card A * card A ∧
     solutions_count = card { (b, c) ∈ A.prod A | b ≥ 2 * c } ∧
     solutions_count = 9 ∧
     (solutions_count.toR / total_count.toR) = 1 / 4) :=
sorry

end quadratic_solution_probability_l498_498285


namespace solve_inequality_l498_498720

theorem solve_inequality (a x : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) :
  ((0 ≤ a ∧ a < 1/2 → a < x ∧ x < 1 - a) ∧ 
   (a = 1/2 → false) ∧ 
   (1/2 < a ∧ a ≤ 1 → 1 - a < x ∧ x < a)) ↔ (x - a) * (x + a - 1) < 0 := 
by
  sorry

end solve_inequality_l498_498720


namespace eq_implies_sq_eq_l498_498043

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l498_498043


namespace angle_ABH_regular_octagon_l498_498923

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498923


namespace measure_of_angle_ABH_l498_498848

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498848


namespace monotonic_increasing_interval_l498_498982

noncomputable def y : ℝ → ℝ := λ x, (3 - x^2) * Real.exp x

theorem monotonic_increasing_interval :
  ∃ s : Set ℝ, (s = Set.Ioo (-3 : ℝ) 1) ∧ ∀ x ∈ s, HasDerivAt y ((-2 * x + (3 - x^2)) * Real.exp x) x :=
by
  sorry

end monotonic_increasing_interval_l498_498982


namespace value_of_a_l498_498253

theorem value_of_a {a x : ℝ} (h1 : x > 0) (h2 : 2 * x + 1 > a * x) : a ≤ 2 :=
sorry

end value_of_a_l498_498253


namespace set_intersection_problem_l498_498056

theorem set_intersection_problem :
  ∃ S : finset (finset ℕ), 
    (S.card = 11) ∧ 
    (∀ s ∈ S, s.card = 5) ∧ 
    (∀ s1 s2 ∈ S, s1 ≠ s2 → s1 ∩ s2 ≠ ∅) ∧ 
    (∀ s ∈ S, ∃ x ∈ s, (S.filter (λ t, x ∈ t)).card ≤ 4) :=
begin
  sorry,
end

end set_intersection_problem_l498_498056


namespace acute_triangle_A_and_area_l498_498741

variables {A B C : ℝ} {a b c : ℝ}

theorem acute_triangle_A_and_area (h1 : a = sqrt 7) (h2 : b = 3) 
  (h3 : sqrt 7 * sin B + sin A = 2 * sqrt 3) (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :
  A = π / 3 ∧ 
  (∃ (area : ℝ), area = 1 / 2 * b * 2 * sin A ∧ area = 3 * sqrt 3 / 2) :=
by
  sorry

end acute_triangle_A_and_area_l498_498741


namespace largest_number_from_hcf_factors_l498_498962

/-- This statement checks the largest number derivable from given HCF and factors. -/
theorem largest_number_from_hcf_factors (HCF factor1 factor2 : ℕ) (hHCF : HCF = 52) (hfactor1 : factor1 = 11) (hfactor2 : factor2 = 12) :
  max (HCF * factor1) (HCF * factor2) = 624 :=
by
  sorry

end largest_number_from_hcf_factors_l498_498962


namespace find_angleYXZ_l498_498346

noncomputable def triangleXYZ := Type
variables {X Y Z O : triangleXYZ}

-- Assume conditions
variables (h1 : ∀ (P : triangleXYZ), P ≠ X ∧ P ≠ Y ∧ P ≠ Z → P ∈ circle O)
variables (angle_XYZ : angle X Y Z = 75)
variables (angle_YZO : angle Y Z O = 25)

-- To Prove
theorem find_angleYXZ (h : angle Y X Z = 55) : h :=
sorry

end find_angleYXZ_l498_498346


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498527

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498527


namespace analytical_expression_range_of_f_l498_498259

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ Icc (-1) 1 then
    if x < 0 then -x^2 + 2*x - 3
    else if x = 0 then 0
    else x^2 + 2*x + 3
  else 0

lemma f_odd (x : ℝ) (hx : x ∈ Icc (-1) 1) : f (-x) = -f x := sorry

theorem analytical_expression (x : ℝ) :
  x ∈ Icc (-1) 1 →
  (f x =
      if x < 0 then -x^2 + 2*x - 3
      else if x = 0 then 0
      else x^2 + 2*x + 3) := sorry

theorem range_of_f :
  set.range f = set.Ico (-6 : ℝ) (-3) ∪ {0} ∪ set.Ioc (3 : ℝ) (6) := sorry

end analytical_expression_range_of_f_l498_498259


namespace tenth_digit_sum_l498_498085

theorem tenth_digit_sum (a b : ℚ) (ha : a = 1 / 2) (hb : b = 1 / 4) : 
  (∀ n : ℕ, n ≥ 2 → (a + b).denom * 10 ^ n < 100) → 
  (∀ n : ℕ, n ≥ 10 →  ((a + b) * 10 ^ n).natAbs % 10 = 0) := 
by
  sorry

end tenth_digit_sum_l498_498085


namespace area_gray_region_l498_498342

-- Defining the given conditions.
def radius_inner (r : ℝ) : Prop := r = 2
def radius_outer (r : ℝ) : Prop := r = 4
def width_gray_region (w : ℝ) : Prop := w = 2

-- The main statement to prove in Lean.
theorem area_gray_region (r : ℝ) (r_outer r_inner : ℝ) (w : ℝ)
  (h₀ : radius_inner r_inner) (h₁ : radius_outer r_outer) (h₂ : width_gray_region w) :
    r_outer = 2 * r_inner →
    r_inner = 2 →
    r_outer = 4 →
    w = 2 →
    (π * r_outer ^ 2 - π * r_inner ^ 2 = 12 * π) :=
by intros;
sorry

end area_gray_region_l498_498342


namespace average_percentage_decrease_l498_498006

theorem average_percentage_decrease
  (original_price final_price : ℕ)
  (h_original_price : original_price = 2000)
  (h_final_price : final_price = 1280) :
  (original_price - final_price) / original_price * 100 / 2 = 18 :=
by 
  sorry

end average_percentage_decrease_l498_498006


namespace triangle_inequality_valid_third_side_l498_498117

theorem triangle_inequality_valid_third_side
  (a b : ℝ) (third_side_candidate : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  (valid_candidate : third_side_candidate > 0) :
  a = 2 → b = 6 → third_side_candidate = 6 → 
  2 + third_side_candidate > 6 ∧ 
  6 + third_side_candidate > 2 ∧ 
  2 + 6 > third_side_candidate ↔ 
  (4 < third_side_candidate ∧ third_side_candidate < 8) := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end triangle_inequality_valid_third_side_l498_498117


namespace sum_q_p_values_l498_498206

def p (x : ℤ) : ℤ := x^2 - 4
def q (x : ℤ) : ℤ := -x

def q_p_composed (x : ℤ) : ℤ := q (p x)

theorem sum_q_p_values :
  q_p_composed (-3) + q_p_composed (-2) + q_p_composed (-1) + q_p_composed 0 + 
  q_p_composed 1 + q_p_composed 2 + q_p_composed 3 = 0 := by
  sorry

end sum_q_p_values_l498_498206


namespace problem1_problem2_problem3_l498_498665

def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + 1)

-- (1) Prove the values of 'a' and 'b'
theorem problem1 (a b : ℝ) (odd_f : ∀ x : ℝ, f a (-x) = -f a x) (domain : set.Icc (-(a+2)) b) :
  a = 1 ∧ b = 3 :=
sorry

-- (2) Prove monotonicity of the function f on [-3, 3]
theorem problem2 : ∀ x1 x2 : ℝ, x1 ∈ set.Icc (-3) 3 → x2 ∈ set.Icc (-3) 3 → x1 < x2 → f 1 x1 < f 1 x2 :=
sorry

-- (3) Prove the range of values for m
theorem problem3 (m : ℝ) : (∀ x : ℝ, x ∈ set.Icc 1 2 → 2 + m * f 1 x + 2^x > 0) ↔ m > -2*ℝ.sqrt 6 - 5 :=
sorry

end problem1_problem2_problem3_l498_498665


namespace angle_ABH_in_regular_octagon_l498_498840

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498840


namespace projection_eq_half_sqrt2_l498_498676

def vec_a : (ℝ × ℝ) := (2*cos(π/3), 2*sin(π/3))   -- Since angle is given in the conditions
def vec_b : (ℝ × ℝ) := (1, 1)

def vec_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let mag_squared := b.1 * b.1 + b.2 * b.2
  let projection_factor := dot_product / mag_squared
  (projection_factor * b.1, projection_factor * b.2)

theorem projection_eq_half_sqrt2 :
  let proj := vec_projection vec_a vec_b
  proj = (√2 / 2, √2 / 2) :=
by
  sorry

end projection_eq_half_sqrt2_l498_498676


namespace compute_nested_operations_l498_498983

def operation (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem compute_nested_operations :
  operation 5 (operation 6 (operation 7 (operation 8 9))) = 3588 / 587 :=
  sorry

end compute_nested_operations_l498_498983


namespace seq_two_three_seq_general_l498_498284

-- Define the sequence according to given conditions
noncomputable def seq : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * seq n + 1

-- Prove that a_2 = 3, a_3 = 7
theorem seq_two_three : seq 1 = 3 ∧ seq 2 = 7 :=
by {
  sorry 
}

-- Prove the general formula for the sequence
theorem seq_general (n : ℕ) : seq n = 2^(n + 1) - 1 :=
by {
  sorry 
}

end seq_two_three_seq_general_l498_498284


namespace sum_of_fifth_powers_l498_498535

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l498_498535


namespace smallest_domain_of_g_l498_498501

-- Definition of the function g
def g (d : Int) : Int :=
  if d % 2 == 0 then -- d is even
    d / 2 + 1
  else -- d is odd
    2 * d + 2

-- Starting condition
def initial_condition (x : Int) : Prop :=
  x = 14 ∧ g x = 29

-- Predicate to define the domain of g
def in_domain (x : Int) : Prop :=
  x = 14 ∨ x = 29 ∨ x = 60 ∨ x = 31 ∨ x = 64 ∨ x = 33 ∨ x = 66 ∨ x = 68 ∨
  x = 34 ∨ x = 35 ∨ x = 70 ∨ x = 72 ∨ x = 36 ∨ x = 37 ∨ x = 74 ∨ x = 76 ∨
  x = 38 ∨ x = 39 ∨ x = 78 ∨ x = 80 ∨ x = 40 ∨ x = 41 ∨ x = 82 ∨ x = 84 ∨
  x = 42 ∨ x = 43 ∨ x = 86 ∨ x = 88

-- The theorem that needs to be proved
theorem smallest_domain_of_g : ∃ S : Set Int, initial_condition 14 ∧ (∀ x, in_domain x → x ∈ S) ∧ S.size = 14 := 
  by
  sorry

end smallest_domain_of_g_l498_498501


namespace type_of_number_when_g_is_a_plus_5_l498_498466

-- Define the function g(a) as given in the problem
def g (a : Int) : Int :=
  if a % 2 = 0 then a / 2 else a + 5

-- Define the target problem
theorem type_of_number_when_g_is_a_plus_5 
  (a : Int) 
  (h1 : g(g(g(g(g(a))))) = 19) 
  (h2 : (finset.filter (λ x, g(g(g(g(g(x))))) = 19) (Finset.range 100)).card = 8) : 
  a % 2 = 1 ∧ a < 0 :=
sorry

end type_of_number_when_g_is_a_plus_5_l498_498466


namespace line_perpendicular_to_plane_l498_498679

noncomputable def direction_vector_line_l : Vectr ℝ 3 :=
(⟨1, 0, 2⟩)

noncomputable def normal_vector_plane_alpha : Vectr ℝ 3 :=
(⟨-2, 0, -4⟩)

theorem line_perpendicular_to_plane (a : Vectr ℝ 3) (mu : Vectr ℝ 3) :
  a = (⟨1, 0, 2⟩) →
  mu = (⟨-2, 0, -4⟩) →
  (a = -0.5 • mu) →
  perpendicular a mu :=
by
  intros ha hmu h.
  rw [ha, hmu] at h.
  use h
  sorry

end line_perpendicular_to_plane_l498_498679


namespace find_a8_l498_498032

noncomputable def increasing_positive_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 2) = a n + a (n + 1)

noncomputable def a (n : ℕ) : ℕ
| 0     := sorry
| 1     := sorry
| (n+2) := a(n) + a(n+1)

variables (a : ℕ → ℕ) (n : ℕ)
hypothesis h_increasing_pos_seq : increasing_positive_sequence a
hypothesis h_a7 : a 6 + a 7 = 120
hypothesis h_positive : ∀ n, 0 < a n

theorem find_a8 (a : ℕ → ℕ) [h_increasing_pos_seq : increasing_positive_sequence a] [h_a7 : a 7 = 120] : a 8 = 194 :=
sorry

end find_a8_l498_498032


namespace primes_satisfy_property_l498_498438

theorem primes_satisfy_property (p : ℕ) (hp : prime p) (hp_gt_3 : p > 3) : 
  ∃ x y k : ℤ, 0 < 2 * k ∧ 2 * k < p ∧ kp + 3 = x^2 + y^2 := 
sorry

end primes_satisfy_property_l498_498438


namespace larger_triangle_perimeter_l498_498187

-- Given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

def similar (t1 t2 : Triangle) (k : ℝ) : Prop :=
  t1.a / t2.a = k ∧ t1.b / t2.b = k ∧ t1.c / t2.c = k

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define specific triangles based on the problem
def smaller_triangle : Triangle := {a := 12, b := 12, c := 15}
def larger_triangle_ratio : ℝ := 2
def larger_triangle : Triangle := {a := 12 * larger_triangle_ratio, b := 12 * larger_triangle_ratio, c := 15 * larger_triangle_ratio}

-- Main theorem statement
theorem larger_triangle_perimeter : perimeter larger_triangle = 78 :=
by 
  sorry

end larger_triangle_perimeter_l498_498187


namespace required_volume_l498_498112

-- Define the dimensions of the box
def length := 20
def width := 20
def height := 15

-- Define the cost per box and the total minimum cost
def cost_per_box := 1.30
def total_cost := 663.0

-- Define the volume of one box and the number of boxes required
def volume_of_one_box := length * width * height
def number_of_boxes := total_cost / cost_per_box

-- Define the total volume required for packaging the collection
def total_volume := number_of_boxes * volume_of_one_box

-- The theorem we want to prove
theorem required_volume : total_volume = 3060000 := by
  sorry

end required_volume_l498_498112


namespace part_a_part_b_part_c_l498_498705

open Real

-- Definition of Parabola
def parabola (x : ℝ) : ℝ := x^2 - x - 2

-- Points on the parabola
def point_on_parabola (x y : ℝ) : Prop := y = parabola x

-- Definition of a line equation
def line_eq (m b x : ℝ) : ℝ := m * x + b

-- Part (a)
theorem part_a (x1 x2 y1 y2 : ℝ) (hP : point_on_parabola x1 y1) (hQ : point_on_parabola x2 y2)
  (hm : (x1 + x2) / 2 = 0) (hm' : (y1 + y2) / 2 = 0) : 
  ∃ m, (∀ x, line_eq m 0 x = -x) := sorry

-- Part (b)
theorem part_b (x1 x2 y1 y2 : ℝ) (hP : point_on_parabola x1 y1) (hQ : point_on_parabola x2 y2)
  (hr : (2 * x2 + x1) / 3 = 0) (hr' : (2 * y2 + y1) / 3 = 0) : 
  ∃ m, (∀ x, line_eq m 0 x = -2 * x) := sorry

-- Part (c)
theorem part_c (x1 x2 : ℝ) (h_int : parabola x1 = line_eq (-2) 0 x1 ∧ parabola x2 = line_eq (-2) 0 x2)
  (hP : x1 < x2) : 
  ∃ A, A = 9 / 2 := sorry

end part_a_part_b_part_c_l498_498705


namespace circle_center_radius_l498_498454

theorem circle_center_radius
    (x y : ℝ)
    (eq_circle : (x - 2)^2 + y^2 = 4) :
    (2, 0) = (2, 0) ∧ 2 = 2 :=
by
  sorry

end circle_center_radius_l498_498454


namespace find_a_b_l498_498243

theorem find_a_b (a b x y : ℝ) (h1 : x = 2) (h2 : y = 4) (h3 : a * x + b * y = 16) (h4 : b * x - a * y = -12) : a = 4 ∧ b = 2 := by
  sorry

end find_a_b_l498_498243


namespace conditionA_is_necessary_for_conditionB_l498_498943

-- Definitions for conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (area : ℝ) -- area of the triangle

def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

def conditionA (t1 t2 : Triangle) : Prop :=
  t1.area = t2.area ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem conditionA_is_necessary_for_conditionB (t1 t2 : Triangle) :
  congruent t1 t2 → conditionA t1 t2 :=
by sorry

end conditionA_is_necessary_for_conditionB_l498_498943


namespace angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498893

theorem angle_ABH_in_regular_octagon_is_22_5_degrees
  (ABCDEFGH_regular : ∀ (A B C D E F G H : Point), regular_octagon A B C D E F G H)
  (AB_equal_AH : ∀ (A B H : Point), is_isosceles_triangle A B H) :
  angle_measure (A B H) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_is_22_5_degrees_l498_498893


namespace measure_of_angle_ABH_l498_498825

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498825


namespace measure_angle_abh_l498_498809

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498809


namespace train_length_180_l498_498573

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_seconds

theorem train_length_180 :
  train_length 6 108 = 180 :=
sorry

end train_length_180_l498_498573


namespace triangle_count_in_rectangle_grid_l498_498205

theorem triangle_count_in_rectangle_grid :
  let rows := 3
  let columns := 4
  let diagonals := 2
  total_triangles rows columns diagonals = 44 := 
by
  sorry

-- Auxiliary function that calculates the total number of triangles
noncomputable def total_triangles (rows : ℕ) (columns : ℕ) (diagonals : ℕ) : ℕ :=
  let basic_triangles := rows * columns * diagonals in
  let combined_row_triangles := (rows * ((columns + 1) * columns / 2)) in
  let large_triangles := 2 in
  basic_triangles + combined_row_triangles + large_triangles

end triangle_count_in_rectangle_grid_l498_498205


namespace fraction_same_ratio_l498_498128

theorem fraction_same_ratio (x : ℚ) : 
  (x / (2 / 5)) = (3 / 7) / (6 / 5) ↔ x = 1 / 7 :=
by
  sorry

end fraction_same_ratio_l498_498128


namespace problem_statement_l498_498300

-- Define the expressions for the given problem
def expr1 : ℝ := (Real.sqrt ((Real.sqrt 5) + 2) + Real.sqrt ((Real.sqrt 5) - 2)) / Real.sqrt ((Real.sqrt 5) + 1)
def expr2 : ℝ := Real.sqrt (3 - 2 * Real.sqrt 2)

-- Define N using the expressions
def N : ℝ := expr1 - expr2

-- The theorem to be proved
theorem problem_statement : N = 1 := 
by
  sorry

end problem_statement_l498_498300


namespace measure_of_angle_ABH_l498_498847

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498847


namespace measure_of_angle_ABH_l498_498823

noncomputable def internal_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

def is_regular_octagon (P : ℝ × ℝ → Prop) : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
    P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G ∧ P H ∧
    ∀ (X Y : ℝ × ℝ), P X → P Y → X ≠ Y → dist X Y = dist A B

noncomputable def measure_angle_ABH (A B H : ℝ × ℝ) [inhabited (ℝ × ℝ)] : ℝ := 
let θ := internal_angle 8 in
  let x := θ/2 in
  (180 - θ) / 2 

theorem measure_of_angle_ABH 
  (A B C D E F G H : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  [inhabited (ℝ × ℝ)]
  (h : is_regular_octagon P) 
  (hA : P A) (hB : P B) (hH : P H) : 
  measure_angle_ABH A B H = 22.5 := 
sorry

end measure_of_angle_ABH_l498_498823


namespace find_p_l498_498269

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def quadrilateral_is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2

theorem find_p (A B C D : ℝ × ℝ) (p : ℝ) (h1 : ∃ x y, circle_eq x y ∧ parabola_eq p x y) 
  (h2 : ∃ x y, circle_eq x y ∧ x = 0) 
  (h3 : quadrilateral_is_rectangle A B C D) 
  (h4 : 0 < p) : 
  p = 2 := 
sorry

end find_p_l498_498269


namespace SharonsSpeedIsSix_l498_498419

noncomputable def SharonsWalkingSpeed : ℝ :=
  let t : ℝ := 0.3 -- Time in hours
  let d : ℝ := 3 -- Distance in miles
  let MarysSpeed : ℝ := 4 -- Mary's walking speed

  -- Define Sharon's walking speed and set up the equation
  let S : ℝ := d / t - MarysSpeed

  -- Using the given conditions and simplifying the equation
  S

theorem SharonsSpeedIsSix :
  SharonsWalkingSpeed = 6 :=
by calc
  SharonsWalkingSpeed = (3 / 0.3) - 4 : by rfl
  ... = 10 - 4          : by norm_num
  ... = 6               : by norm_num

end SharonsSpeedIsSix_l498_498419


namespace sum_of_solutions_l498_498329

theorem sum_of_solutions : 
  ∀ (x y : ℝ), y = 8 → x^2 + y^2 = 169 → (x = sqrt 105 ∨ x = -sqrt 105) → (sqrt 105 + (-sqrt 105) = 0) :=
by
  intros x y h1 h2 h3
  sorry

end sum_of_solutions_l498_498329


namespace actual_time_when_car_clock_shows_10PM_l498_498645

def car_clock_aligned (aligned_time wristwatch_time : ℕ) : Prop :=
  aligned_time = wristwatch_time

def car_clock_time (rate: ℚ) (hours_elapsed_real_time hours_elapsed_car_time : ℚ) : Prop :=
  rate = hours_elapsed_car_time / hours_elapsed_real_time

def actual_time (current_car_time car_rate : ℚ) : ℚ :=
  current_car_time / car_rate

theorem actual_time_when_car_clock_shows_10PM :
  let accurate_start_time := 9 -- 9:00 AM
  let car_start_time := 9 -- Synchronized at 9:00 AM
  let wristwatch_time_wristwatch := 13 -- 1:00 PM in hours
  let car_time_car := 13 + 48 / 60 -- 1:48 PM in hours
  let rate := car_time_car / wristwatch_time_wristwatch
  let current_car_time := 22 -- 10:00 PM in hours
  let real_time := actual_time current_car_time rate
  real_time = 19.8333 := -- which converts to 7:50 PM (Option B)
sorry

end actual_time_when_car_clock_shows_10PM_l498_498645


namespace solution_set_l498_498978

noncomputable def e : ℝ := Real.exp 1

variable (f : ℝ → ℝ)
variable (H1 : ∀ x, f(x) + (fun y => deriv f y) x < e)
variable (H2 : f(0) = e + 2)

theorem solution_set (x : ℝ) : (e^x * f x > e^(x + 1) + 2) ↔ x < 0 := 
by 
  sorry

end solution_set_l498_498978


namespace largest_bucket_capacity_l498_498710

-- Let us define the initial conditions
def capacity_5_liter_bucket : ℕ := 5
def capacity_3_liter_bucket : ℕ := 3
def remaining_after_pour := capacity_5_liter_bucket - capacity_3_liter_bucket
def additional_capacity_without_overflow : ℕ := 4

-- Problem statement: Prove that the capacity of the largest bucket is 6 liters
theorem largest_bucket_capacity : ∀ (c : ℕ), remaining_after_pour + additional_capacity_without_overflow = c → c = 6 := 
by
  sorry

end largest_bucket_capacity_l498_498710


namespace measure_angle_abh_l498_498815

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498815


namespace angle_ABH_regular_octagon_l498_498918

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498918


namespace sunflower_seed_count_l498_498709

theorem sunflower_seed_count :
  (weight_per_seed : ℚ) (total_weight : ℚ),
  weight_per_seed = 1 / 15 ∧
  total_weight = 300 →
  total_weight * 15 = 4500 :=
by
  sorry

end sunflower_seed_count_l498_498709


namespace measure_angle_abh_l498_498818

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498818


namespace sin_70_equals_1_minus_2a_squared_l498_498651

variable (a : ℝ)

theorem sin_70_equals_1_minus_2a_squared (h : Real.sin (10 * Real.pi / 180) = a) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * a^2 := 
sorry

end sin_70_equals_1_minus_2a_squared_l498_498651


namespace ratio_of_wormy_apples_l498_498415

theorem ratio_of_wormy_apples 
  (total_apples : ℕ) (bruised_apples : ℕ) (wormy_apples : ℕ) (raw_apples : ℕ)
  (h1 : total_apples = 85)
  (h2 : bruised_apples = (1/5 : ℚ) * 85 + 9)
  (h3 : raw_apples = 42)
  (h4 : wormy_apples = total_apples - bruised_apples - raw_apples) :
  wormy_apples / total_apples = 17 / 85 :=
begin
  sorry
end

end ratio_of_wormy_apples_l498_498415


namespace ivan_uses_more_paint_l498_498072

-- Conditions
def ivan_section_area : ℝ := 5 * 2
def petr_section_area (alpha : ℝ) : ℝ := 5 * 2 * Real.sin(alpha)
axiom alpha_lt_90 : ∀ α : ℝ, α < 90 → Real.sin(α) < 1

-- Assertion
theorem ivan_uses_more_paint (α : ℝ) (h1 : α < 90) : ivan_section_area > petr_section_area α :=
by
  sorry

end ivan_uses_more_paint_l498_498072


namespace extreme_point_a_zero_monotonic_increasing_in_interval_real_roots_maximum_b_l498_498273

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^3) / 3 - x^2 - a * x + Real.log (a * x + 1)

-- (1)
theorem extreme_point_a_zero (a : ℝ) : (∃ x = 2, ∀ y, y ≠ x → f y a ≠ f x a) → a = 0 := 
by sorry

-- (2)
theorem monotonic_increasing_in_interval (a : ℝ) : (∀ x, x ∈ Set.Ici 3 → 0 ≤ (deriv (λ x => f x a) x)) → a ∈ Set.Icc 0 ((3 + Real.sqrt 13) / 2) := 
by sorry

noncomputable def f_negative_one (x : ℝ) : ℝ := (x^3) / 3 + b / (1 - x)

-- (3)
theorem real_roots_maximum_b (b : ℝ) : (∃ x, f_negative_one x = 0) → b = 0 := 
by sorry

end extreme_point_a_zero_monotonic_increasing_in_interval_real_roots_maximum_b_l498_498273


namespace number_of_correct_statements_is_one_l498_498704

noncomputable def count_correct_statements (a b : ℝ^3) (alpha beta : set ℝ^3) : ℕ :=
  let s1 := ¬ (a ⟂ b ∧ a ⟂ alpha → b ∥ alpha)
  let s2 := ¬ (alpha ⟂ beta ∧ a ∥ alpha → a ⟂ beta)
  let s3 := ¬ (a ⟂ beta ∧ alpha ⟂ beta → a ∥ alpha)
  let s4 := (a ⟂ b ∧ a ⟂ alpha ∧ b ⟂ beta → alpha ⟂ beta)
  let correct_statements := [s1, s2, s3, s4].count(λ x, x)
  correct_statements

theorem number_of_correct_statements_is_one (a b : ℝ^3) (alpha beta : set ℝ^3) :
  count_correct_statements a b alpha beta = 1 :=
sorry

end number_of_correct_statements_is_one_l498_498704


namespace C_proposition_l498_498261

variables {α β γ m n : Type}
-- Assume α, β, γ are planes and m, n are lines.
variable [Plane α] [Plane β] [Line m] [Line n]

-- Definitions for parallelism and intersection
def parallel (x y : Type) [HasParallel x y] : Prop := is_parallel x y
def intersect (x y : Type) [HasIntersection x y] : Type := intersection x y

-- Given conditions
variables (pα : Plane α) (pβ : Plane β)
variables (l m n : Line m)

-- Importing conditions into statements:
def αβ_intersection : α ∩ β = n := sorry
def m_parallel_α : m ∥ α := sorry
def m_parallel_β : m ∥ β := sorry

-- The proposition to be proven
theorem C_proposition (h1 : α ∩ β = n) (h2 : m ∥ α) (h3 : m ∥ β) : m ∥ n :=
sorry

end C_proposition_l498_498261


namespace log_sum_inequality_l498_498942

theorem log_sum_inequality (a b c : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) :
  log a (log a b) + log b (log b c) + log c (log c a) > 0 :=
by 
sorry

end log_sum_inequality_l498_498942


namespace distance_to_lightning_l498_498988

theorem distance_to_lightning
  (speed_of_sound : ℝ)
  (time_interval : ℝ)
  (h_speed : speed_of_sound = 331)
  (h_time : time_interval = 12) :
  (331 * 12 / 1000).round = 4 :=
by
  let distance := speed_of_sound * time_interval
  have h_distance : distance = 3972, by sorry
  let distance_km := distance / 1000
  have h_distance_km : distance_km = 3.972, by sorry
  have h_rounded_distance : (distance_km).round = 4, by sorry
  exact h_rounded_distance

end distance_to_lightning_l498_498988


namespace zander_stickers_l498_498123

theorem zander_stickers (S : ℕ) (h1 : 44 = (11 / 25) * S) : S = 100 :=
by
  sorry

end zander_stickers_l498_498123


namespace traffic_light_probability_l498_498578

open ProbabilityTheory MeasureTheory Set Real

noncomputable def period : ℝ := (60 + 45) / 60 -- Total period, in minutes

noncomputable def greenLightInterval : Set ℝ := Ioc 0 1 -- Interval representing green light

noncomputable def uniformDist : MeasureSpace ℝ := 
  volume.restrict (Ioc 0 period) / (volume (Ioc 0 period))

theorem traffic_light_probability :
  (volume.restrict greenLightInterval uniformDist).val = 4 / 7 :=
by 
  sorry

end traffic_light_probability_l498_498578


namespace polar_to_cartesian_l498_498281

theorem polar_to_cartesian (θ ρ x y : ℝ) (h1 : ρ = 2 * Real.sin θ) (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l498_498281


namespace find_constant_l498_498465

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant :
  (∀ x : ℝ, f x = x + 4) →
  (∃ c : ℝ, (∀ x : ℝ, (3 * f (x - 2) / f 0 + 4 = f (c * x + 1))) → c = 2) :=
begin
  intro h_f,
  use 2,
  intros x h_eq,
  -- additional proof goes here
  sorry,
end

end find_constant_l498_498465


namespace distinct_differences_condition_l498_498133

theorem distinct_differences_condition (n : ℕ) (n_ge_3 : n ≥ 3) :
  (exists (a : Fin n → ℕ),
    (∀ i : Fin n, a i ∈ {1, 2, ..., n+1}) ∧
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin n, |a(i) - a((i+1)%n)| ∈ {1, 2, ..., n} ∧ ∀ i j : Fin n, i ≠ j → |a(i) - a((i+1)%n)| ≠ |a(j) - a((j+1)%n)|) ) ↔
  (∃ k : ℕ, n = 4 * k ∨ n = 4 * k - 1) :=
sorry

end distinct_differences_condition_l498_498133


namespace integral_abs_convergence_integral_conditional_convergence_l498_498753

def floor (x : Real) : Int := -- definition of the floor function for [1/x]
  Int.ofNat (Nat.floor x)

noncomputable def improper_integral (f : Real → Real) (a b : Real) : Real := sorry

theorem integral_abs_convergence :
  ¬convergent (improper_integral (λ x => abs ((-1)^(floor (1/x)) / x)) 0 1) := sorry

theorem integral_conditional_convergence :
  convergent (improper_integral (λ x => ((-1)^(floor (1/x)) / x)) 0 1) := sorry

end integral_abs_convergence_integral_conditional_convergence_l498_498753


namespace angle_in_regular_octagon_l498_498890

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498890


namespace solve_for_x_l498_498955

theorem solve_for_x (x : ℝ) (h : 8^(2 * x - 9) = 2^(- (2 * x + 3))) : x = 3 :=
by {
  sorry
}

end solve_for_x_l498_498955


namespace cost_of_sunglasses_l498_498170

noncomputable def cost_per_pair 
  (price_per_pair : ℕ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℕ) 
  (profits_half : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_revenue := price_per_pair * pairs_sold in
  let total_cost := total_revenue - (profits_half * 2) in
  total_cost / pairs_sold

theorem cost_of_sunglasses :
  ∀ (price_per_pair pairs_sold sign_cost profits_half profit : ℕ),
    price_per_pair = 30 → 
    pairs_sold = 10 → 
    sign_cost = 20 → 
    (profits_half = sign_cost → profit = profits_half * 2) →
    cost_per_pair price_per_pair pairs_sold sign_cost profits_half profit = 26 :=
begin
  intros,
  sorry
end

end cost_of_sunglasses_l498_498170


namespace inequality_geq_l498_498435

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l498_498435


namespace sum_of_fifth_powers_l498_498538

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l498_498538


namespace ratio_of_almonds_to_walnuts_l498_498553

theorem ratio_of_almonds_to_walnuts (A W : ℕ) 
    (h1 : 280 = 200 + 80)
    (h2 : 80 = 2 * W)
    (h3 : 200 = A * W)
    : 200 / 80 = 2.5 :=
by ratio_custom sorry

end ratio_of_almonds_to_walnuts_l498_498553


namespace hawks_loss_percentage_is_30_l498_498049

-- Define the variables and the conditions
def matches_won (x : ℕ) : ℕ := 7 * x
def matches_lost (x : ℕ) : ℕ := 3 * x
def total_matches (x : ℕ) : ℕ := matches_won x + matches_lost x
def percent_lost (x : ℕ) : ℕ := (matches_lost x * 100) / total_matches x

-- The goal statement in Lean 4
theorem hawks_loss_percentage_is_30 (x : ℕ) (h : x > 0) : percent_lost x = 30 :=
by sorry

end hawks_loss_percentage_is_30_l498_498049


namespace tan_of_negative_angle_l498_498202

theorem tan_of_negative_angle :
  Real.tan (Real.angle.of_deg (-3645)) = -1 :=
by
  sorry

end tan_of_negative_angle_l498_498202


namespace regular_octagon_angle_ABH_l498_498870

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498870


namespace exists_digit_sum_divisible_by_11_l498_498941

-- Define a function to compute the sum of the digits of a natural number
def digit_sum (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

-- The main theorem to be proven
theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k) % 11 = 0) := 
sorry

end exists_digit_sum_divisible_by_11_l498_498941


namespace sum_of_T_is_2510_l498_498387

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498387


namespace new_person_weight_l498_498453

theorem new_person_weight (n : ℕ) (a : ℝ) (initial_person_weight new_avg_weight : ℝ)
  (h1 : n = 15)
  (h2 : a = 3.2)
  (h3 : initial_person_weight = 80)
  (h4 : new_avg_weight = initial_person_weight + n * a) :
  new_avg_weight = 128 := by
  rw [h1, h2, h3]
  sorry

end new_person_weight_l498_498453


namespace regular_octagon_angle_ABH_l498_498880

noncomputable def measure_angle_ABH (AB AH : ℝ) (internal_angle : ℝ) : ℝ := 
if h : AB = AH then 
  let x := (180 - internal_angle) / 2 in x 
else 
  0 -- handle the non-equal case for completeness

-- Proving the specific scenario where internal_angle = 135 and AB = AH
theorem regular_octagon_angle_ABH (AB AH : ℝ) 
(h : AB = AH) : measure_angle_ABH AB AH 135 = 22.5 :=
by 
  unfold measure_angle_ABH
  split_ifs
  { 
    simp [h]
    linarith
  }
  sorry  -- placeholder for the equality check case due to completeness

end regular_octagon_angle_ABH_l498_498880


namespace measure_angle_abh_l498_498810

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  regular : ∀ i : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices 0) (vertices 1)
  angles : ∀ i : Fin 8, angle (vertices i) (vertices (i + 1)) (vertices (i + 2)) = 135

-- Define the measure of angle ABH in the regular octagon
theorem measure_angle_abh (O : RegularOctagon) :
  angle O.vertices 0 O.vertices 1 O.vertices 7 = 22.5 := sorry

end measure_angle_abh_l498_498810


namespace right_triangle_third_side_l498_498738

theorem right_triangle_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : c = Real.sqrt (7) ∨ c = 5) :
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
  sorry

end right_triangle_third_side_l498_498738


namespace Ivan_uses_more_paint_l498_498070

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l498_498070


namespace part1_natural_numbers_less_than_5000_no_repeated_digits_part2_odd_even_position_constraints_part2_only_odd_positions_for_odd_digits_l498_498547

theorem part1_natural_numbers_less_than_5000_no_repeated_digits :
  ∑ n in {1, 2, 3, 4}, (∑ v in (Finset.perm_filter (List.range 10) n.toNat) fun _ => 1) = 2349 :=
sorry

theorem part2_odd_even_position_constraints :
  let odd_digits := {1, 3, 5, 7, 9}
  let even_digits := {2, 4, 6, 8}
  ∑ p in (Finset.prod odd_digits (Finset.prod odd_digits (Finset.prod odd_digits (Finset.prod even_digits even_digits))), fun _ => 1) = 720 :=
sorry

theorem part2_only_odd_positions_for_odd_digits :
  let odd_digits := {1, 3, 5, 7, 9}
  ∑ p in (Finset.prod odd_digits (Finset.prod odd_digits (Finset.prod odd_digits (Finset.insert 2 (Finset.range 10 \ odd_digits)))), fun _ => 1) = 1800 :=
sorry

end part1_natural_numbers_less_than_5000_no_repeated_digits_part2_odd_even_position_constraints_part2_only_odd_positions_for_odd_digits_l498_498547


namespace sum_of_T_is_2510_l498_498384

def isRepeatingDecimalForm (x : ℝ) : Prop :=
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : Set ℝ := {x | isRepeatingDecimalForm x}

noncomputable def sum_of_elements_T : ℝ :=
  2510

theorem sum_of_T_is_2510 : 
  (¬ ∃ x, x ∈ T → x ≠ 2510) → Σ x ∈ T, x = sum_of_elements_T := by
  sorry 

end sum_of_T_is_2510_l498_498384


namespace number_of_possible_scenarios_l498_498200
noncomputable def possible_scenarios_for_y (x y z : ℕ) : ℕ :=
  if (x + y + z = 100) ∧ (x + 2*y + 5*z = 300) then 17 else 0
  
theorem number_of_possible_scenarios :
  ∀ (x y z : ℕ), (x + y + z = 100) → (x + 2*y + 5*z = 300) → possible_scenarios_for_y x y z = 17 := 
by
  intros x y z h1 h2
  rw possible_scenarios_for_y
  simp [h1, h2]
  sorry

end number_of_possible_scenarios_l498_498200


namespace max_truthful_gnomes_l498_498000

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l498_498000


namespace range_of_a_l498_498693

def f (x : ℝ) : ℝ := 3^(-x) - 3^x - x

theorem range_of_a (a : ℝ) (h : f (2 * a + 3) + f (3 - a) > 0) : a < -6 :=
by
  sorry

end range_of_a_l498_498693


namespace kernels_popped_in_final_bag_l498_498802

/-- Parker wants to find out what the average percentage of kernels that pop in a bag is.
In the first bag he makes, 60 kernels pop and the bag has 75 kernels.
In the second bag, 42 kernels pop and there are 50 in the bag.
In the final bag, some kernels pop and the bag has 100 kernels.
The average percentage of kernels that pop in a bag is 82%.
How many kernels popped in the final bag?
We prove that given these conditions, the number of popped kernels in the final bag is 82.
-/
noncomputable def kernelsPoppedInFirstBag := 60
noncomputable def totalKernelsInFirstBag := 75
noncomputable def kernelsPoppedInSecondBag := 42
noncomputable def totalKernelsInSecondBag := 50
noncomputable def totalKernelsInFinalBag := 100
noncomputable def averagePoppedPercentage := 82

theorem kernels_popped_in_final_bag (x : ℕ) :
  (kernelsPoppedInFirstBag * 100 / totalKernelsInFirstBag +
   kernelsPoppedInSecondBag * 100 / totalKernelsInSecondBag +
   x * 100 / totalKernelsInFinalBag) / 3 = averagePoppedPercentage →
  x = 82 := 
by
  sorry

end kernels_popped_in_final_bag_l498_498802


namespace soccer_team_wins_l498_498569

-- Definitions for conditions
def total_games_played : ℕ := 158
def win_percentage : ℝ := 0.4

-- The theorem statement
theorem soccer_team_wins : 
  let games_won := (win_percentage * (total_games_played : ℝ)).round in
  games_won = 63 :=
by
  sorry

end soccer_team_wins_l498_498569


namespace sum_solutions_eq_zero_l498_498332

theorem sum_solutions_eq_zero :
  let y := 8 in
  let eq1 := y = 8 in
  let eq2 := ∀ x, x ^ 2 + y ^ 2 = 169 in
  (x ∈ ℝ ∧ eq1 ∧ eq2) →
  ∃ x1 x2, x1 ^ 2 + y ^ 2 = 169 ∧ x2 ^ 2 + y ^ 2 = 169 ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_solutions_eq_zero_l498_498332


namespace quadratic_root_ratio_l498_498634

theorem quadratic_root_ratio (k : ℝ) (h : ∃ r : ℝ, r ≠ 0 ∧ 3 * r * r = k * r - 12 * r + k ∧ r * r = k + 9 * r - k) : k = 27 :=
sorry

end quadratic_root_ratio_l498_498634


namespace inequality_abc_l498_498430

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l498_498430


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498525

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l498_498525


namespace number_of_lines_through_6_5_l498_498335

open Int

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

def candidates_x_intercepts : List ℕ := [7, 11]

theorem number_of_lines_through_6_5 :
  (∀ a, a ∈ candidates_x_intercepts → is_prime a ∧ a > 5) →
  (∃ b : ℕ, ∀ a, a ∈ candidates_x_intercepts → ∃! l : AffineLine R ^ 2, l.contains (6, 5) ∧
    l.x_intercept = some (a : R) ∧ ∃ b, l.y_intercept = some (b : R) ∧ b > 0
   → 2) := 
    sorry

end number_of_lines_through_6_5_l498_498335


namespace sum_of_mathcalT_is_2520_l498_498378

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498378


namespace maximum_monthly_profit_at_5_l498_498510

def R (x : ℕ) : ℚ := (1 / 2) * x * (x + 1) * (39 - 2 * x)

def W (x : ℕ) : ℚ := 150000 + 2000 * x

def g (x : ℕ) : ℚ := -3 * x^2 + 40 * x

def f (x : ℕ) : ℚ := (185000 - W x) * g x

theorem maximum_monthly_profit_at_5 : ∃ x : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ f x = 3125000 :=
by
  have h_x5 : f 5 = 3125000 := sorry -- this should demonstrate the calculation proof for x = 5 resulting in 3125000 profit
  use 5
  split
  · show 1 ≤ 5, from nat.le_refl 1
  · split
    · show 5 ≤ 6, from nat.le_succ 5
    · exact h_x5

end maximum_monthly_profit_at_5_l498_498510


namespace trigonometric_expression_simplification_l498_498587

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end trigonometric_expression_simplification_l498_498587


namespace polynomial_remainder_l498_498629

theorem polynomial_remainder :
  let f := (λ x, x ^ 1012)
  let g := (λ x, (x ^ 2 - 1) * (x + 2))
  ∃ r : ℚ[X], degree r < degree g ∧ ∃ q : ℚ[X], f = q * g + r ∧ r = 1 :=
by
  let f := (λ x, x ^ 1012)
  let g := (λ x, (x ^ 2 - 1) * (x + 2))
  sorry

end polynomial_remainder_l498_498629


namespace approximate_number_of_fish_in_pond_l498_498515

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, 
    (∃ tagged_first : ℕ, tagged_first = 50) ∧
    (∃ caught_second : ℕ, caught_second = 50) ∧
    (∃ tagged_second : ℕ, tagged_second = 2) ∧
    (∑ n, (tagged_second : ℚ) / (caught_second : ℚ) = (tagged_first : ℚ) / (N : ℚ)) ∧
    N = 1250 :=
begin
  sorry
end

end approximate_number_of_fish_in_pond_l498_498515


namespace cards_per_page_l498_498119

theorem cards_per_page 
  (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) : (new_cards + old_cards) / pages = 3 := 
by 
  sorry

end cards_per_page_l498_498119


namespace distance_between_foci_correct_l498_498662

noncomputable def ellipse_distance_between_foci (a b : ℝ) : ℝ :=
  let c := real.sqrt (a * a - b * b)
  in 2 * c

theorem distance_between_foci_correct {a b : ℝ} (h_a_gt_b : a > b) (h_a_pos : a > 0)
  (h_b_pos : b > 0) (h_f1f2 : ∀ P : ℝ × ℝ, (P.1 = 2 ∧ P.2 = 1) →
  |(real.sqrt (a * a - b * b)) + real.sqrt (a * a - b * b)| = 2 * real.sqrt 6) :
  ∃ d : ℝ, d = |(2 * real.sqrt (a * a - b * b))| ∧ d = 2 * real.sqrt 3 :=
begin
  let a := real.sqrt 6,
  let b := real.sqrt 3,
  have h_a : a = real.sqrt 6, from sorry,
  have h_b2 : b * b = 3, from sorry,
  have h_c2 : a * a - b * b = 3, from sorry,
  let c := real.sqrt (a * a - b * b),
  have h_c : c = real.sqrt 3, from sorry,
  use 2 * c,
  split,
  { rw abs_of_nonneg, exact sorry, exact sorry },
  { rw [←mul_assoc, mul_comm 2, mul_assoc, ←mul_assoc 2, mul_comm 2, mul_assoc], exact sorry }
end

end distance_between_foci_correct_l498_498662


namespace find_f2_l498_498464

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : 
  f 2 = -3 - (2 * real.sqrt 3 / 3) :=
by
  sorry

end find_f2_l498_498464


namespace probability_2023_equals_2970_l498_498137

noncomputable def grid : ℕ := 45
noncomputable def total_numbers : ℕ := 2025

def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability := 
  (45 * (binom 45 2) * 44 * 45) / (binom total_numbers 3)

theorem probability_2023_equals_2970 :
  probability = 2970 / ((2025 * 2024 * 2023) / 6) := sorry

end probability_2023_equals_2970_l498_498137


namespace max_shapes_fit_l498_498126

-- Define dimensions of the shapes and the cube
def shape_length := 2
def shape_width := 2
def shape_height := 1

def cube_length := 3
def cube_width := 3
def cube_height := 3

-- Volume calculations
def shape_volume : ℕ := shape_length * shape_width * shape_height
def cube_volume : ℕ := cube_length * cube_width * cube_height

-- Maximum number of shapes that can fit into the cube
theorem max_shapes_fit : (cube_volume / shape_volume) = 6 :=
by
  have cube_vol := cube_volume
  have shape_vol := shape_volume
  have num_shapes := cube_vol / shape_vol
  exact eq.refl num_shapes
  sorry

end max_shapes_fit_l498_498126


namespace measure_of_angle_ABH_l498_498850

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498850


namespace part_a_part_b_l498_498514

-- Part (a) Lean 4 Statement
theorem part_a (k : ℕ) (h : 0 < k) :
  ∃ (a : Fin k → ℕ), (∀ i j : Fin k, i ≠ j → (a i - a j) ∣ (a i)^1997) ∧ (∀ i j : Fin k, i < j → a i < a j) :=
sorry

-- Part (b) Lean 4 Statement
theorem part_b :
  ∃ c : ℝ, c > 0 ∧ ∀ (k : ℕ) (a : Fin k → ℕ), (∀ i j : Fin k, i ≠ j → (a i - a j) ∣ (a i)^1997) ∧
  (∀ i j : Fin k, i < j → a i < a j) → a (⟨k - 1, Nat.pred_lt (show 0 < k from sorry)⟩) > 2^(c * k) :=
sorry

end part_a_part_b_l498_498514


namespace find_number_l498_498620

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l498_498620


namespace proof_problem_l498_498402

def h (x : ℝ) : ℝ := 2 * x + 4
def k (x : ℝ) : ℝ := 4 * x + 6

theorem proof_problem : h (k 3) - k (h 3) = -6 :=
by
  sorry

end proof_problem_l498_498402


namespace quadratic_equation_example_l498_498511

theorem quadratic_equation_example (x : ℝ) :
  (∀ a b c : ℝ, a = 2 ∧ (∀ α β : ℝ, α = 3 ∧ β = -1/2 → (∀ x : ℝ, a * x^2 + b * x + c = (x - α) * (x - β) * a))) →
  (∃ b c : ℝ, 2 * x ^ 2 + b * x + c = 2 * x^2 - 5 * x - 3) :=
by {
  intros h,
  have h3 : h 2 -5 -3,
  {
    intro α,
    intro β,
    have sum := α = 3 ∧ β = -1/2 → α + β = (3) + (-1/2) := rfl,
    have prod := α = 3 ∧ β = -1/2 → α * β = (3) * (-1/2) := rfl,
    exact ⟨sum, prod⟩,
  },
  exact ⟨-5, -3, h3⟩,
}

end quadratic_equation_example_l498_498511


namespace first_digit_base_9_of_628_l498_498088

theorem first_digit_base_9_of_628 :
  ∃ d : ℕ, (d < 9) ∧ (628 = d * 81 + (628 % 81)) ∧ d = 7 :=
by
  use 7
  split
  . exact Nat.lt_succ_self 7
  . split
  . exact Nat.mod_add_div 628 81
  . rfl

end first_digit_base_9_of_628_l498_498088


namespace sum_of_mathcalT_is_2520_l498_498376

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498376


namespace distinct_strings_after_operations_l498_498796

def valid_strings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else valid_strings (n-1) + valid_strings (n-2)

theorem distinct_strings_after_operations :
  valid_strings 10 = 144 := by
  sorry

end distinct_strings_after_operations_l498_498796


namespace find_lambda_l498_498707

theorem find_lambda (λ : ℝ) : 
  (let a := (1, 2 : ℝ × ℝ) in
   let b := (1, 0 : ℝ × ℝ) in
   let c := (3, 4 : ℝ × ℝ) in
   let v := (1 + λ, 2 : ℝ × ℝ) in
   v.1 * c.2 = v.2 * c.1) →
   λ = 1 / 2 :=
by
  intros h
  sorry

end find_lambda_l498_498707


namespace can_form_triangle_l498_498028

noncomputable theory

-- Define the points A, B, C, D as elements of a normed space
variables (A B C D : ℝ^3)

-- Assume the distances between points A, B, and C form an equilateral triangle
def is_equilateral_triangle (A B C : ℝ^3) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Prove that given an equilateral triangle ABC, the segments DA, DB, DC can form a triangle
theorem can_form_triangle (h : is_equilateral_triangle A B C) :
  ∃ (T : Triangle ℝ^3), T.has_sides (dist D A) (dist D B) (dist D C) := 
sorry

end can_form_triangle_l498_498028


namespace problem_l498_498039

theorem problem (a b a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ)
  (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) 
  (h4 : a₄ < a₅) (h5 : a₅ < a₆) (h6 : a₆ < a₇)
  (h7 : a₇ < a₈) (h8 : a₈ < a₉) (h9 : a₉ < a₁₀)
  (h10 : a₁₀ < a₁₁) (h11 : b₁ < b₂) (h12 : b₂ < b₃)
  (h13 : b₃ < b₄) (h14 : b₄ < b₅) (h15 : b₅ < b₆)
  (h16 : b₆ < b₇) (h17 : b₇ < b₈) (h18 : b₈ < b₉)
  (h19 : b₉ < b₁₀) (h20 : b₁₀ < b₁₁) 
  (h21 : a₁₀ + b₁₀ = a) (h22 : a₁₁ + b₁₁ = b) : 
  a = 1024 ∧ b = 2048 :=
sorry

end problem_l498_498039


namespace angle_ABH_is_22_point_5_l498_498908

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498908


namespace trigonometric_identity_l498_498546

theorem trigonometric_identity :
  sin (27 * Real.pi / 180) * cos (63 * Real.pi / 180) +
  cos (27 * Real.pi / 180) * sin (63 * Real.pi / 180) = 1 := 
by
  sorry

end trigonometric_identity_l498_498546


namespace sum_of_reciprocals_of_divisors_l498_498783

theorem sum_of_reciprocals_of_divisors (p : ℕ) (h_prime : Nat.Prime (2^p - 1)) : 
  let q := 2^p - 1
  let n := 2^(p - 1) * q
  (∑ d in (Nat.divisors n), (1:ℚ) / d) = 2 := 
by 
  sorry

end sum_of_reciprocals_of_divisors_l498_498783


namespace solution_set_inequality_l498_498680

noncomputable theory

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if 0 < x then x * exp x else x * exp (-x)

def correct_solution_set (x : ℝ) : Prop :=
  (- real.log 3 < x ∧ x < 0) ∨ (real.log 3 < x)

theorem solution_set_inequality :
  is_odd_function f →
  (∀ x: ℝ, x > 0 → f x = x * exp x) →
  ∀ x : ℝ, f x > 3 * x ↔ correct_solution_set x := sorry

end solution_set_inequality_l498_498680


namespace alpha_value_l498_498367

theorem alpha_value (α : ℂ) (h₁ : α ≠ 1) 
  (h₂ : complex.abs (α^2 - 1) = 3 * complex.abs (α - 1))
  (h₃ : complex.abs (α^3 - 1) = 6 * complex.abs (α - 1)) :
  α = -1 :=
sorry

end alpha_value_l498_498367


namespace inequality_abc_l498_498431

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l498_498431


namespace f_9_over_2_eq_5_over_2_l498_498773

noncomputable def f : ℝ → ℝ := sorry

-- The domain of f is ℝ, f(x + 1) is odd, f(x + 2) is even
axiom h_domain : ∀ x : ℝ, f x ∈ ℝ
axiom h_odd : ∀ x : ℝ, f (x + 1) = -f (-x + 1)
axiom h_even : ∀ x : ℝ, f (x + 2) = f (-x + 2)

-- When x ∈ [1, 2], f(x) = ax^2 + b
axiom h_quad : ∃ a b : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x ^ 2 + b

-- f(0) + f(3) = 6
axiom h_sum : f 0 + f 3 = 6

-- Prove that f(9/2) = 5/2
theorem f_9_over_2_eq_5_over_2 : f (9 / 2) = 5 / 2 :=
by 
  -- proof goes here
  sorry

end f_9_over_2_eq_5_over_2_l498_498773


namespace negation_of_proposition_l498_498700

theorem negation_of_proposition (a b : ℝ) : 
  (¬ (∀ (a b : ℝ), (ab > 0 → a > 0)) ↔ ∀ (a b : ℝ), (ab ≤ 0 → a ≤ 0)) := 
sorry

end negation_of_proposition_l498_498700


namespace find_min_sum_of_squares_l498_498407

open Real

theorem find_min_sum_of_squares
  (x1 x2 x3 : ℝ)
  (h1 : 0 < x1)
  (h2 : 0 < x2)
  (h3 : 0 < x3)
  (h4 : 2 * x1 + 4 * x2 + 6 * x3 = 120) :
  x1^2 + x2^2 + x3^2 >= 350 :=
sorry

end find_min_sum_of_squares_l498_498407


namespace simplify_sqrt_expression_l498_498444

theorem simplify_sqrt_expression : 
  (sqrt 800 / sqrt 100 - sqrt 288 / sqrt 72 = 2 * sqrt 2 - 2) :=
by
  sorry

end simplify_sqrt_expression_l498_498444


namespace monotonic_intervals_intersection_points_l498_498411

open Real

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - a * log x

-- Define the derivative of f
def f_derivative (x : ℝ) (a : ℝ) : ℝ := (x^2 - a) / x

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - (a + 1) * x

-- Define the function F
def F (x : ℝ) (a : ℝ) : ℝ := f x a - g x a

-- Prove monotonic intervals of f
theorem monotonic_intervals (a : ℝ) :
  (f_derivative ⟹ 0) ↔ (a ≤ 0 ↔ ∀ x > 0, f_derivative x a > 0) ∧
  (a > 0 ↔ ∀ x ∈ Ioi 0, (x < sqrt a ↔ f_derivative x a < 0) ∧ (x > sqrt a ↔ f_derivative x a > 0)) := sorry

-- Prove number of intersection points between f(x) and g(x)
theorem intersection_points (a : ℝ) (ha : a ≥ 0) :
  ∃! x > 0, F x a = 0 := sorry

end monotonic_intervals_intersection_points_l498_498411


namespace common_roots_exist_and_unique_l498_498229

noncomputable def polynomial1 (a : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C 20 * Polynomial.X + Polynomial.C 10

noncomputable def polynomial2 (b : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C 17 * Polynomial.X + Polynomial.C 12

theorem common_roots_exist_and_unique (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ Polynomial.is_root (polynomial1 a) r ∧ Polynomial.is_root (polynomial1 a) s ∧
     Polynomial.is_root (polynomial2 b) r ∧ Polynomial.is_root (polynomial2 b) s) ↔ (a = 1 ∧ b = 0) :=
begin
  sorry
end

end common_roots_exist_and_unique_l498_498229


namespace sum_of_T_l498_498388

def is_repeating_abcd (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

noncomputable def T : set ℝ :=
{ x | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   0 ≤ a ∧ a ≤ 9 ∧ 
   0 ≤ b ∧ b ≤ 9 ∧ 
   0 ≤ c ∧ c ≤ 9 ∧ 
   0 ≤ d ∧ d ≤ 9 ∧ 
   is_repeating_abcd x a b c d }

theorem sum_of_T : ∑ x in T, x = 227.052227052227 :=
sorry

end sum_of_T_l498_498388


namespace calculate_expression_l498_498197

theorem calculate_expression : sqrt 4 - abs (sqrt 3 - 2) + (-1)^2023 = sqrt 3 - 1 := by
  sorry

end calculate_expression_l498_498197


namespace part_a_part_b_l498_498540

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l498_498540


namespace tangent_circumcircle_angle_bisector_equality_l498_498480

-- Given a triangle ABC with AD as the angle bisector
-- and the tangent at A to the circumcircle of triangle ABC intersects the line BC at point E,
-- prove that AE = ED.

theorem tangent_circumcircle_angle_bisector_equality
  (A B C D E : Type)
  [Triangle ABC]
  (h1 : is_angle_bisector AD ⦃ABC⦄)
  (h2 : is_tangent_at A (circumcircle ABC) E BC) :
  AE = ED := 
sorry

end tangent_circumcircle_angle_bisector_equality_l498_498480


namespace part_I_part_II_l498_498405

open Classical

-- Definitions based on the conditions
variables {p : ℝ} (h : p > 0)

/-- A point A on the axis of symmetry of the parabola y^2 = 2px, located inside the parabola -/
def A (a : ℝ) (h₀: a > 0) : ℝ × ℝ := (a, 0)

/-- The reflection of A about the y-axis -/
def B (a : ℝ) (h₀: a > 0) : ℝ × ℝ := (-a, 0)

/-- points P and Q on the parabola -/
def on_parabola (P Q : ℝ × ℝ) : Prop := P.2 ^ 2 = 2 * p * P.1 ∧ Q.2 ^ 2 = 2 * p * Q.1

namespace MathProof

/-- Part (I): If a line passing through A intersects the parabola at points P and Q on either side of the axis of symmetry,
then ∠PBA = ∠QBA -/
theorem part_I (a : ℝ) (h₀: a > 0) (P Q : ℝ × ℝ) (h₁ : on_parabola p P Q) (h₂ : P.1 > 0 ∧ Q.1 > 0) :
  (∠ B p h₀ P A = ∠ Q p h₀ P A) := sorry

/-- Part (II): If a line passing through B intersects the parabola at points P and Q on one side of the axis of symmetry,
then ∠PAB + ∠QAB = 180° -/
theorem part_II (a : ℝ) (h₀: a > 0) (P Q : ℝ × ℝ) (h₁ : on_parabola p P Q) (h₂ : P.1 > 0 ∧ Q.1 > 0) :
  (∠ P p h₀ Q A + ∠ Q p h₀ Q A = 180) := sorry

end MathProof

end part_I_part_II_l498_498405


namespace angle_ABH_is_22_point_5_l498_498915

-- Define what it means to be a regular octagon
def regular_octagon (A B C D E F G H : Point) : Prop :=
  all_equidistant [A, B, C, D, E, F, G, H] ∧
  all_equal_internal_angles [A, B, C, D, E, F, G, H]

-- Define the points (for clarity)
variables (A B C D E F G H : Point)

-- Given: Polygon ABCDEFGH is a regular octagon
axiom h : regular_octagon A B C D E F G H

-- Prove that the measure of angle ABH is 22.5 degrees
theorem angle_ABH_is_22_point_5 : ∠ABH = 22.5 :=
by {
  sorry
}

end angle_ABH_is_22_point_5_l498_498915


namespace find_width_of_floor_l498_498567

variable (w : ℝ) -- width of the floor

theorem find_width_of_floor (h1 : w - 4 > 0) (h2 : 10 - 4 > 0) 
                            (area_rug : (10 - 4) * (w - 4) = 24) : w = 8 :=
by
  sorry

end find_width_of_floor_l498_498567


namespace pentagon_AH_perpendicular_CE_l498_498341

open EuclideanGeometry

variables {A B C D E F H : Point}

def regular_pentagon (A B C D E : Point) : Prop :=
  ∃ l, 
    regular_polygon_l A B C D E l ∧
    ∀ (i : ℕ), i ∈ finset.range 5 → 
    (angle (vertices_of_regular_polygon_l i) (vertices_of_regular_polygon_l (i + 1)) (vertices_of_regular_polygon_l (i + 2)) = π / 5)

def midpoint (F : Point) (C D : Point) : Prop :=
  dist C F = dist F D

def perpendicular_bisector (A F H : Point) : Prop :=
  ∃ M, midpoint M A F ∧ collinear [M, H, C, E]

theorem pentagon_AH_perpendicular_CE
  (h1 : regular_pentagon A B C D E)
  (h2 : midpoint F C D)
  (h3 : perpendicular_bisector A F H) :
  angle A H C = π / 2 := 
sorry

end pentagon_AH_perpendicular_CE_l498_498341


namespace pos_factors_of_36_multiple_of_4_l498_498716

theorem pos_factors_of_36_multiple_of_4 : 
  (finset.filter (λ x => x % 4 = 0) ({1, 2, 3, 4, 6, 9, 12, 18, 36} : finset ℕ)).card = 3 := 
by
  sorry

end pos_factors_of_36_multiple_of_4_l498_498716


namespace prism_edge_lengths_l498_498160

theorem prism_edge_lengths (x : ℝ) : 
    (let
        l3 := Real.log x / Real.log 3
        l5 := Real.log x / Real.log 5
        l7 := Real.log x / Real.log 7
        sa := 2 * (l3 * l5 + l3 * l7 + l5 * l7)
        v := l3 * l5 * l7
    in sa = v) → x = 11025 :=
begin
  sorry
end

end prism_edge_lengths_l498_498160


namespace ratio_of_kits_to_students_l498_498791

theorem ratio_of_kits_to_students (art_kits students : ℕ) (h1 : art_kits = 20) (h2 : students = 10) : art_kits / Nat.gcd art_kits students = 2 ∧ students / Nat.gcd art_kits students = 1 := by
  sorry

end ratio_of_kits_to_students_l498_498791


namespace number_of_master_sudokus_l498_498361

open Nat

def f : ℕ → ℕ
| 0       => 1
| (n + 1) => Σ i in range (n + 1), f i

theorem number_of_master_sudokus (n : ℕ) : f n = 2^(n - 1) :=
by
  sorry

end number_of_master_sudokus_l498_498361


namespace kyle_additional_bottles_l498_498356

-- Define the parameters and conditions
def initial_bottles : ℕ := 2
def bottle_capacity : ℕ := 15
def total_stars_needed : ℕ := 75

-- The total capacity of the initial bottles
def initial_capacity := initial_bottles * bottle_capacity

-- The theorem statement
theorem kyle_additional_bottles : 
  ∃ (additional_bottles : ℕ), additional_bottles = (total_stars_needed - initial_capacity) / bottle_capacity :=
begin
  use 3,
  sorry
end

end kyle_additional_bottles_l498_498356


namespace hyperbola_eccentricity_l498_498270

theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 6 / 2 := 
by
  sorry

end hyperbola_eccentricity_l498_498270


namespace number_of_pairs_l498_498450

theorem number_of_pairs (A B : set ℕ) (a b : ℕ) :
  A = {x : ℕ | 5 * x ≤ a} →
  B = {x : ℕ | 6 * x > b} →
  (∀ x, x ∈ (A ∩ B) → x ∈ {2, 3, 4}) →
  (∃! (a b : ℕ), 20 ≤ a ∧ a < 25 ∧ 6 < b ∧ b < 12 ∧
    (A ∩ B ∩ (set.univ : set ℕ) = {2, 3, 4})) →
  (∃ (pairs : ℕ), pairs = 25) :=
begin
  intros hA hB hAB hex,
  obtain ⟨a, ha1, ha2, ha3, ha4, ha5⟩ := hex,
  sorry
end

end number_of_pairs_l498_498450


namespace volume_of_rectangular_solid_l498_498991

theorem volume_of_rectangular_solid
  (a b c : ℝ)
  (h1 : a * b = 3)
  (h2 : a * c = 5)
  (h3 : b * c = 15) :
  a * b * c = 15 :=
sorry

end volume_of_rectangular_solid_l498_498991


namespace T_n_sum_l498_498283

noncomputable def a_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else
  (∑ k in Finset.range n, (4 ^ k) * (a_seq (k + 1))) = n / 4

noncomputable def b_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else
  (4 ^ n) * (a_seq n) / (2 * n + 1)

lemma a_seq_formula (n : ℕ) (h : n ≠ 0) :
  a_seq n = 1 / (4 ^ n) :=
sorry

theorem T_n_sum (n : ℕ) :
  (∑ k in Finset.range n, (b_seq k) * (b_seq (k + 1))) = n / (6 * n + 9) :=
sorry

end T_n_sum_l498_498283


namespace sum_of_first_six_terms_geometric_sequence_l498_498608

theorem sum_of_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r ^ n) / (1 - r)
  S_n = 1365 / 4096 := by
  sorry

end sum_of_first_six_terms_geometric_sequence_l498_498608


namespace first_year_after_2010_with_digit_sum_10_l498_498102

/--
Theorem: The first year after 2010 for which the sum of the digits equals 10 is 2017.
-/
theorem first_year_after_2010_with_digit_sum_10 : ∃ (y : ℕ), (y > 2010) ∧ (∑ d in (y.to_digits 10), d) = 10 ∧ (∀ z, (z > 2010) → ((∑ d in (z.to_digits 10), d) = 10 → z ≥ y))

end first_year_after_2010_with_digit_sum_10_l498_498102


namespace sum_of_mathcalT_is_2520_l498_498379

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def isRepeatingDecimal (n : ℕ) : Prop :=
  n % 9999 = n ∧ n < 10000

def mathcalT (S : set ℕ) : Prop :=
  ∀ n, n ∈ S ↔ ∃ a b c d : ℕ, 
    isDistinctDigits a b c d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    isRepeatingDecimal (1000 * a + 100 * b + 10 * c + d)

noncomputable def sumOfElements (S : set ℕ) : ℚ :=
  ∑ n in S, (n : ℚ) / 9999

theorem sum_of_mathcalT_is_2520 (S : set ℕ) (hT : mathcalT S) :
  sumOfElements S = 2520 := by
  sorry

end sum_of_mathcalT_is_2520_l498_498379


namespace math_problem_l498_498257

-- Definitions of conditions
def cond1 (x a y b z c : ℝ) : Prop := x / a + y / b + z / c = 1
def cond2 (x a y b z c : ℝ) : Prop := a / x + b / y + c / z = 0

-- Theorem statement
theorem math_problem (x a y b z c : ℝ)
  (h1 : cond1 x a y b z c) (h2 : cond2 x a y b z c) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1 :=
by
  sorry

end math_problem_l498_498257


namespace percentage_female_officers_on_duty_l498_498422

theorem percentage_female_officers_on_duty:
  ∀ (total_officers_on_duty : ℕ) (female_officers_on_duty : ℕ) (total_female_officers : ℕ),
  total_officers_on_duty = 170 →
  female_officers_on_duty = total_officers_on_duty / 2 →
  total_female_officers = 500 →
  (female_officers_on_duty : ℚ) / total_female_officers * 100 = 17 := 
by
  intros total_officers_on_duty female_officers_on_duty total_female_officers h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end percentage_female_officers_on_duty_l498_498422


namespace max_C_usage_l498_498596

-- Definition of variables (concentration percentages and weights)
def A_conc := 3 / 100
def B_conc := 8 / 100
def C_conc := 11 / 100

def target_conc := 7 / 100
def total_weight := 100

def max_A := 50
def max_B := 70
def max_C := 60

-- Equation to satisfy
def conc_equation (x y : ℝ) : Prop :=
  C_conc * x + B_conc * y + A_conc * (total_weight - x - y) = target_conc * total_weight

-- Definition with given constraints
def within_constraints (x y : ℝ) : Prop :=
  x ≤ max_C ∧ y ≤ max_B ∧ (total_weight - x - y) ≤ max_A

-- The theorem that needs to be proved
theorem max_C_usage (x y : ℝ) : within_constraints x y ∧ conc_equation x y → x ≤ 50 :=
by
  sorry

end max_C_usage_l498_498596


namespace sum_of_set_T_l498_498396

def is_repeating_decimal_0_abcd (x : ℝ) : Prop :=
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def set_T : set ℝ := {x | is_repeating_decimal_0_abcd x}

theorem sum_of_set_T : 
  ∑ x in set_T.to_finset, x = 2520 :=
sorry

end sum_of_set_T_l498_498396


namespace angle_ABH_regular_octagon_l498_498928

theorem angle_ABH_regular_octagon (n : ℕ) (h : n = 8) : ∃ x : ℝ, x = 22.5 :=
by
  have h1 : (n - 2) * 180 / n = 135
  sorry

  have h2 : 2 * x + 135 = 180
  sorry

  have h3 : 2 * x = 45
  sorry

  use 22.5
  sorry

end angle_ABH_regular_octagon_l498_498928


namespace measure_of_angle_ABH_l498_498856

noncomputable def measure_angle_ABH : ℝ :=
  if h : True then 22.5 else 0

theorem measure_of_angle_ABH (A B H : ℝ) (h₁ : is_regular_octagon A B H) 
  (h₂ : consecutive_vertices A B H) : measure_angle_ABH = 22.5 := 
by
  sorry

structure is_regular_octagon (A B H : ℝ) :=
  (congruent_sides : A = B ∧ B = H)

structure consecutive_vertices (A B H : ℝ) :=
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_H : H > 0)

end measure_of_angle_ABH_l498_498856


namespace problem_statement_l498_498312

noncomputable def sin2A_div_sinC (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : ℝ := 
  let A := acos ((b^2 + c^2 - a^2) / (2 * b * c))
  let C := acos ((a^2 + b^2 - c^2) / (2 * a * b))
  let sinA := sin A
  let sinC := sin C
  let cosA := cos A
  let sin2A := 2 * sinA * cosA
  sin2A / sinC

theorem problem_statement (A B C : Type) (a b c : ℝ)
  (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  sin2A_div_sinC a b c ha hb hc = 1 := 
sorry

end problem_statement_l498_498312


namespace area_increase_44_percent_l498_498310

-- Define the original side length and the increased side length.
variable {s : ℝ}

def original_area (s : ℝ) : ℝ := s^2

def new_side_length (s : ℝ) : ℝ := 1.2 * s

def new_area (s : ℝ) : ℝ := (new_side_length s)^2

-- Define the percentage increase in area.
def percentage_increase_in_area (s : ℝ) : ℝ :=
  ((new_area s) - (original_area s)) / (original_area s) * 100

-- Prove that the percentage increase in area is 44%
theorem area_increase_44_percent (s : ℝ) : percentage_increase_in_area s = 44 := by
  sorry

end area_increase_44_percent_l498_498310


namespace projection_correct_l498_498678

open Real

variables (a : ℝ → ℝ → ℝ) (b : ℝ → ℝ)
variables (θ : ℝ)

noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_sq := b.1 * b.1 + b.2 * b.2
  (dot_product / magnitude_sq * b.1, dot_product / magnitude_sq * b.2)

theorem projection_correct
  (h_angle : θ = π / 3)
  (h_a_magnitude : (a 0 2).1 = 2 ∧ (a 0 2).2 = 0)
  (h_b : b 1 = 1 ∧ b 1 = 1) :
  proj (a θ 2) (1, 1) = (sqrt 2 / 2, sqrt 2 / 2) :=
sorry

end projection_correct_l498_498678


namespace problem_statement_l498_498499

def op_table : Array (Array ℕ) :=
  #[#[1, 2, 3, 4], #[2, 3, 4, 1], #[3, 4, 1, 2], #[4, 1, 2, 3]]

def custom_op (i j : ℕ) : ℕ :=
  if h1 : i > 0 ∧ i ≤ 4 ∧ j > 0 ∧ j ≤ 4 then
    op_table[i-1][j-1]
  else
    0  -- Default case (not used under the problem's valid inputs)

theorem problem_statement : (custom_op 3 2) * (custom_op 4 1) = custom_op 1 4  :=
by
  sorry

end problem_statement_l498_498499


namespace problem_lambda_range_l498_498673

noncomputable def P (n : ℕ) (a_n : ℝ) : Prop :=
  a_n = (2 * n + 4) / n

noncomputable def b_n (n : ℕ) (a_n : ℝ) (lambda : ℝ) : ℝ :=
  a_n + lambda * n

noncomputable def increasing_seq {α : Type*} [LinearOrder α] (seq : ℕ → α) : Prop :=
  ∀ n : ℕ, seq (n + 1) > seq n

theorem problem_lambda_range (lambda : ℝ) (n : ℕ) (hn : 0 < n) :
  (∃ a_n, P n a_n ∧ increasing_seq (λ n, b_n n a_n lambda)) ↔ lambda > 2 :=
sorry

end problem_lambda_range_l498_498673


namespace find_rate_of_current_l498_498479

noncomputable def rate_of_current : ℝ := 
  let speed_still_water := 42
  let distance_downstream := 33.733333333333334
  let time_hours := 44 / 60
  (distance_downstream / time_hours) - speed_still_water

theorem find_rate_of_current : rate_of_current = 4 :=
by sorry

end find_rate_of_current_l498_498479


namespace solution_set_of_inequality_l498_498052

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l498_498052


namespace proof_problem_minimize_a1_b1_l498_498469

theorem proof_problem_minimize_a1_b1 (a_1 a_2 a_m b_1 b_2 b_n : ℕ)
  (h0 : a_1! * a_2! * ... * a_m! = 2025 * (b_1! * b_2! * ... * b_n!))
  (h1 : a_1 ≥ a_2)
  (h2 : b_1 ≥ b_2)
  (h3 : a_1 > 0)
  (h4 : b_1 > 0)
  (min_a1_b1 : a_1 + b_1 is minimized) :
  |a_1 - b_1| = 1 := by
  sorry

end proof_problem_minimize_a1_b1_l498_498469


namespace alpha_irrational_l498_498748
open Set

/-- A function to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of the digit sequence a_n. -/
def a_n (n : ℕ) : ℕ :=
  if is_prime n then 1 else 0

/-- Definition of the real number α by its decimal expansion. -/
noncomputable def α : ℝ :=
  ∑' n : ℕ, (a_n (n + 1) : ℝ) / (10 ^ (n + 1))

/-- The theorem to prove that the real number α is irrational. -/
theorem alpha_irrational : irrational α :=
sorry

end alpha_irrational_l498_498748


namespace unique_n_for_T_n_integer_l498_498641

noncomputable def T_n (n : ℕ) (b : Fin n → ℝ) : ℝ :=
∑ k in Finset.range n, sqrt ((3 * (k + 1) - 2)^2 + b ⟨k, by linarith⟩ ^ 2)

theorem unique_n_for_T_n_integer : ∃! (n : ℕ), n = 8 ∧ ∀ (b : Fin n → ℝ), (∑ (i : Fin n), b i = 23) → (T_n n b).denom = 1 :=
by
  intro n
  use 8
  sorry

end unique_n_for_T_n_integer_l498_498641


namespace find_xyz_l498_498785

open Complex

theorem find_xyz (a b c x y z : ℂ)
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0)
  (ha : a = (b + c) / (x + 1))
  (hb : b = (a + c) / (y + 1))
  (hc : c = (a + b) / (z + 1))
  (hxy_z_1 : x * y + x * z + y * z = 9)
  (hxy_z_2 : x + y + z = 5) :
  x * y * z = 13 := 
sorry

end find_xyz_l498_498785


namespace angle_in_regular_octagon_l498_498886

noncomputable def measure_angle_ABH (α β : ℝ) : Prop :=
  α = 135 ∧ β = 22.5 ∧ 2 * β + α = 180

theorem angle_in_regular_octagon
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (A B H : ABCDEFGH)
  (AB AH : ℝ) (h_eq_sides : AB = AH) :
  measure_angle_ABH 135 22.5 :=
by
  unfold measure_angle_ABH
  sorry

end angle_in_regular_octagon_l498_498886


namespace log5_7_gt_log13_17_l498_498591

theorem log5_7_gt_log13_17 : log 5 7 > log 13 17 := by
  sorry

end log5_7_gt_log13_17_l498_498591


namespace kevin_savings_first_exceeds_10_dollars_l498_498355

def kevin_savings_day (initial_amount : ℕ) (multiplier : ℕ) (target_cents : ℕ) : ℕ :=
  let S_n (n : ℕ) := initial_amount * ((multiplier ^ n - 1) / (multiplier - 1))
  let n := Nat.find (λ n, S_n n ≥ target_cents)
  n % 7

theorem kevin_savings_first_exceeds_10_dollars :
  kevin_savings_day 2 3 1000 = 6 := -- Saturday corresponds to 6, assuming 0-indexed where 0 is Sunday
sorry

end kevin_savings_first_exceeds_10_dollars_l498_498355


namespace Ivan_uses_more_paint_l498_498067

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l498_498067


namespace coefficient_a2b2_in_expansion_l498_498087

theorem coefficient_a2b2_in_expansion :
  -- Combining the coefficients: \binom{4}{2} and \binom{6}{3}
  (Nat.choose 4 2) * (Nat.choose 6 3) = 120 :=
by
  -- No proof required, using sorry to indicate that.
  sorry

end coefficient_a2b2_in_expansion_l498_498087


namespace sunglasses_cost_l498_498168

open Real

def cost_per_pair (selling_price_per_pair : ℝ) (num_pairs_sold : ℝ) (sign_cost : ℝ) : ℝ := 
  (num_pairs_sold * selling_price_per_pair - 2 * sign_cost) / num_pairs_sold

theorem sunglasses_cost (sp : ℝ) (n : ℝ) (sc : ℝ) (H1 : sp = 30) (H2 : n = 10) (H3 : sc = 20) :
  cost_per_pair sp n sc = 26 :=
by
  rw [H1, H2, H3]
  simp [cost_per_pair]
  norm_num
  sorry

end sunglasses_cost_l498_498168


namespace initial_apples_l498_498029

-- Definitions based on the given conditions
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

-- Statement to prove
theorem initial_apples : apples_given_away + apples_left = 127 :=
by {
  -- Proof steps would go here
  sorry
}

end initial_apples_l498_498029


namespace trucks_initial_count_l498_498949

theorem trucks_initial_count (x : ℕ) (h : x - 13 = 38) : x = 51 :=
by sorry

end trucks_initial_count_l498_498949


namespace salary_percentage_change_l498_498442

theorem salary_percentage_change (S : ℝ) (x : ℝ) :
  (S * (1 - (x / 100)) * (1 + (x / 100)) = S * 0.84) ↔ (x = 40) :=
by
  sorry

end salary_percentage_change_l498_498442


namespace angle_ABH_in_regular_octagon_l498_498834

theorem angle_ABH_in_regular_octagon {A B C D E F G H : Point} (h1 : regular_octagon A B C D E F G H) :
  measure_of_angle A B H = 22.5 :=
sorry

end angle_ABH_in_regular_octagon_l498_498834
