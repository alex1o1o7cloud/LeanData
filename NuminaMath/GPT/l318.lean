import Mathlib
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.CombinatorialHierarchy
import Mathlib.Combinatorics.Complex4
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Real.Basic
import Mathlib.Mathlib.Probability.Theory.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.Tactic

namespace x_power_2023_zero_or_neg_two_l318_318373

variable {x : ℂ} -- Assuming x is a complex number to handle general roots of unity.

theorem x_power_2023_zero_or_neg_two 
  (h1 : (x - 1) * (x + 1) = x^2 - 1)
  (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
  (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
  (pattern : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) :
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 :=
by
  sorry

end x_power_2023_zero_or_neg_two_l318_318373


namespace rhombus_perimeter_l318_318026

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h3 : d1 / 2 ≠ 0) (h4 : d2 / 2 ≠ 0) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * s = 52 :=
by
  sorry

end rhombus_perimeter_l318_318026


namespace max_winners_at_least_three_matches_l318_318493

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end max_winners_at_least_three_matches_l318_318493


namespace find_coeff_sum_l318_318394

def parabola_eq (a b c : ℚ) (y : ℚ) : ℚ := a*y^2 + b*y + c

theorem find_coeff_sum 
  (a b c : ℚ)
  (h_eq : ∀ y, parabola_eq a b c y = - ((y + 6)^2) / 3 + 7)
  (h_pass : parabola_eq a b c 0 = 5) :
  a + b + c = -32 / 3 :=
by
  sorry

end find_coeff_sum_l318_318394


namespace coloring_ways_1128_l318_318102

theorem coloring_ways_1128 : ∃ (count : ℕ), count = 1128 ∧
  (∀ (n : ℕ), n = 50 → 
    ∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ (i j : ℕ), (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j ∧ abs (i - j) > 2) ∧
      pairs.card = count) :=
by {
  use 1128,
  split,
  { refl },
  { intros n hn,
    use {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n ∧ p.1 ≠ p.2 ∧ abs (p.1 - p.2) > 2}.to_finset,
    split,
    { intros i j h,
      simp at h,
      exact h, },
    { simp [hn],
      sorry -- Proof of the actual counting logic goes here
    }
  }
}

end coloring_ways_1128_l318_318102


namespace b_minus_a_l318_318041

def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflectAboutLineYEqNegX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

def transformations (a b : ℝ) : ℝ × ℝ :=
  let (x1, y1) := reflectAboutLineYEqNegX a b
  let (x2, y2) := rotate90 x1 y1 1 5
  (x2, y2)

theorem b_minus_a : let P := (12, 7 : ℝ)
    let final_image := (-6, 3 : ℝ)
    transformations P.1 P.2 = final_image → (P.2 - P.1) = -5 :=
by
  sorry

end b_minus_a_l318_318041


namespace cos_squared_difference_l318_318908

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318908


namespace quadrilateral_problem_l318_318354

-- Define a convex quadrilateral and the points M, N, P, and Q based on the problem's conditions
variables {A B C D M N P Q : Type} [convex_quadrilateral A B C D]

-- Define the existence of regular triangles ABM, CDP outward and BCN, ADQ inward
variables (regular_triangle_outward : regular_triangle A B M) 
          (regular_triangle_inward : regular_triangle B C N)
          (regular_triangle_inward' : regular_triangle A D Q)
          (regular_triangle_outward' : regular_triangle C D P)

-- State the problem in Lean 4
theorem quadrilateral_problem :
  MN = AC ∧ (parallelogram M N P Q ∨ collinear_points M N P Q) :=
sorry

end quadrilateral_problem_l318_318354


namespace max_m_value_l318_318199

theorem max_m_value : ∃ m : ℝ, (∀ a b : ℝ, (a / real.exp a - b) ^ 2 ≥ m - (a - b + 3) ^ 2) ∧ m = 9 / 2 :=
begin
  -- proof omitted
  sorry
end

end max_m_value_l318_318199


namespace problem_l318_318545

variable (x y : ℝ)

def op_star (x y : ℝ) : ℝ := 1/y - 1/x

theorem problem :
  x ≠ 0 →
  y ≠ 0 →
  op_star x y = 2 →
  (2022 * x * y) / (x - y) = 1011 :=
by
  intros
  sorry

end problem_l318_318545


namespace area_difference_of_circles_l318_318018

theorem area_difference_of_circles (C1 C2 : ℝ) (hC1 : C1 = 264) (hC2 : C2 = 704) : 
  let r1 := C1 / (2 * Real.pi),
      r2 := C2 / (2 * Real.pi),
      A1 := Real.pi * r1^2,
      A2 := Real.pi * r2^2,
      diff := A2 - A1
  in diff ≈ 33866.72 := 
by
  sorry

end area_difference_of_circles_l318_318018


namespace ratio_of_red_marbles_l318_318112

theorem ratio_of_red_marbles (total_marbles blue_marbles green_marbles yellow_marbles red_marbles : ℕ)
  (h1 : total_marbles = 164)
  (h2 : blue_marbles = total_marbles / 2)
  (h3 : green_marbles = 27)
  (h4 : yellow_marbles = 14)
  (h5 : red_marbles = total_marbles - (blue_marbles + green_marbles + yellow_marbles)) :
  (red_marbles : ℚ) / total_marbles = (1 : ℚ) / 4 :=
by {
  sorry
}

end ratio_of_red_marbles_l318_318112


namespace rhombus_perimeter_52_l318_318022

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l318_318022


namespace cos_squared_difference_l318_318917

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318917


namespace largest_integer_chosen_l318_318501

-- Define the sequence of operations and establish the resulting constraints
def transformed_value (x : ℤ) : ℤ :=
  2 * (4 * x - 30) - 10

theorem largest_integer_chosen : 
  ∃ (x : ℤ), (10 : ℤ) ≤ transformed_value x ∧ transformed_value x ≤ (99 : ℤ) ∧ x = 21 :=
by
  sorry

end largest_integer_chosen_l318_318501


namespace number_of_truthful_people_l318_318053

-- Definitions from conditions
def people := Fin 100
def tells_truth (p : people) : Prop := sorry -- Placeholder definition.

-- Conditions
axiom c1 : ∃ p : people, ¬ tells_truth p
axiom c2 : ∀ p1 p2 : people, p1 ≠ p2 → (tells_truth p1 ∨ tells_truth p2)

-- Goal
theorem number_of_truthful_people : 
  ∃ S : Finset people, S.card = 99 ∧ (∀ p ∈ S, tells_truth p) :=
sorry

end number_of_truthful_people_l318_318053


namespace total_charging_time_l318_318123

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l318_318123


namespace third_part_is_306_l318_318291

-- Define the total amount and ratio
def total_amount : ℝ := 782
def ratio1 : ℝ := 1 / 2
def ratio2 : ℝ := 2 / 3
def ratio3 : ℝ := 3 / 4

def common_denominator := 12
def ratio1_common := ratio1 * 12
def ratio2_common := ratio2 * 12
def ratio3_common := ratio3 * 12

def sum_of_ratios := ratio1_common + ratio2_common + ratio3_common

def third_part_value := (ratio3_common / sum_of_ratios) * total_amount

-- Prove that the value of the third part is Rs. 306
theorem third_part_is_306 : third_part_value = 306 :=
by
  sorry

end third_part_is_306_l318_318291


namespace max_value_of_S_l318_318565

variable (n : ℕ) (M : ℝ) (a_1 a_(n+1) a_(2n+1) : ℝ)

def max_S (a : ℕ → ℝ) : ℝ :=
  a (n + 1) + a (n + 2) + ... + a (2*n + 1)

theorem max_value_of_S (n : ℕ) (M : ℝ) (h₀ : 0 < n) (h₁ : 0 < M) (a_1 a_{n+1} : ℝ) (h₂ : a_1^2 + a_{n+1}^2 ≤ M) :
  max_S (a) = \frac{\sqrt{10}}{2}(n+1) \sqrt{M} :=
sorry

end max_value_of_S_l318_318565


namespace lisa_total_cost_l318_318688

def c_phone := 1000
def c_contract_per_month := 200
def c_case := 0.20 * c_phone
def c_headphones := 0.5 * c_case
def t_year := 12

theorem lisa_total_cost :
  c_phone + (c_case) + (c_headphones) + (c_contract_per_month * t_year) = 3700 :=
by
  sorry

end lisa_total_cost_l318_318688


namespace maria_luke_difference_in_nickels_l318_318361

theorem maria_luke_difference_in_nickels (q : ℕ) : 
  let maria_quarters := 2 * q^2 + 3 in
  let luke_quarters := 3^q + 4 in
  let difference_in_quarters := maria_quarters - luke_quarters in
  let difference_in_nickels := 5 * difference_in_quarters in
  difference_in_nickels = 10 * q^2 - 15^q - 5 :=
by
  sorry -- proof not required, assuming the focus is on the statement itself

end maria_luke_difference_in_nickels_l318_318361


namespace find_a_l318_318267

theorem find_a (x y a : ℝ) (h1 : x + 3 * y = 4 - a) 
  (h2 : x - y = -3 * a) (h3 : x + y = 0) : a = 1 :=
sorry

end find_a_l318_318267


namespace Kelly_gives_away_games_l318_318652

variable (N : ℕ) -- number of Nintendo games
variable (S T : ℕ) -- number of Sony games and target number of Sony games

theorem Kelly_gives_away_games 
  (h1 : N = 46) 
  (h2 : S = 132) 
  (h3 : T = 31) : ∃ x : ℕ, S - x = T ∧ x = 101 :=
by
  use 101
  split
  . rw [h2, h3]
    norm_num
  . rfl

#eval Kelly_gives_away_games 46 132 31

end Kelly_gives_away_games_l318_318652


namespace crayons_produced_l318_318988

theorem crayons_produced (colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) 
  (h_colors : colors = 4) (h_crayons_per_color : crayons_per_color = 2) 
  (h_boxes_per_hour : boxes_per_hour = 5) (h_hours : hours = 4) : 
  colors * crayons_per_color * boxes_per_hour * hours = 160 := 
by
  rw [h_colors, h_crayons_per_color, h_boxes_per_hour, h_hours]
  norm_num

end crayons_produced_l318_318988


namespace marie_ends_with_755_l318_318690

def erasers_end (initial lost packs erasers_per_pack : ℕ) : ℕ :=
  initial - lost + packs * erasers_per_pack

theorem marie_ends_with_755 :
  erasers_end 950 420 3 75 = 755 :=
by
  sorry

end marie_ends_with_755_l318_318690


namespace root_quad_eq_sum_l318_318666

theorem root_quad_eq_sum (a b : ℝ) (h1 : a^2 + a - 2022 = 0) (h2 : b^2 + b - 2022 = 0) (h3 : a + b = -1) : a^2 + 2 * a + b = 2021 :=
by sorry

end root_quad_eq_sum_l318_318666


namespace identity_implies_sum_l318_318283

theorem identity_implies_sum (a b : ℚ) (h : ∀ x : ℚ, 0 < x → 
    (a / (10 ^ x - 3) + b / (10 ^ x + 4) = (3 * 10 ^ x + 4) / ((10 ^ x - 3) * (10 ^ x + 4)))) :
    a + b = 3 := 
sorry

end identity_implies_sum_l318_318283


namespace cos_difference_identity_l318_318937

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318937


namespace trigonometric_identity_l318_318854

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318854


namespace centers_of_externally_constructed_equilateral_triangles_form_equilateral_centers_of_internally_constructed_equilateral_triangles_form_equilateral_difference_in_areas_of_internal_and_external_equilateral_triangles_equals_original_area_l318_318086

-- Problem (a)
theorem centers_of_externally_constructed_equilateral_triangles_form_equilateral
  (ABC : Triangle)
  (A_side_ext : EquilateralTriangle)
  (B_side_ext : EquilateralTriangle)
  (C_side_ext : EquilateralTriangle)
  (Centers_of_ext_eq_triangles_form_eq_triangle : EquilateralTriangle):
  Centers_of_ext_eq_triangles_form_eq_triangle := sorry

-- Problem (b)
theorem centers_of_internally_constructed_equilateral_triangles_form_equilateral
  (ABC : Triangle)
  (A_side_int : EquilateralTriangle)
  (B_side_int : EquilateralTriangle)
  (C_side_int : EquilateralTriangle)
  (Centers_of_int_eq_triangles_form_eq_triangle : EquilateralTriangle):
  Centers_of_int_eq_triangles_form_eq_triangle := sorry

-- Problem (c)
theorem difference_in_areas_of_internal_and_external_equilateral_triangles_equals_original_area
  (ABC : Triangle)
  (A_side_ext : EquilateralTriangle)
  (B_side_ext : EquilateralTriangle)
  (C_side_ext : EquilateralTriangle)
  (A_side_int : EquilateralTriangle)
  (B_side_int : EquilateralTriangle)
  (C_side_int : EquilateralTriangle):
  (Area A_side_ext + Area B_side_ext + Area C_side_ext) - (Area A_side_int + Area B_side_int + Area C_side_int) = Area ABC := sorry

end centers_of_externally_constructed_equilateral_triangles_form_equilateral_centers_of_internally_constructed_equilateral_triangles_form_equilateral_difference_in_areas_of_internal_and_external_equilateral_triangles_equals_original_area_l318_318086


namespace min_sum_permutations_eq_44_l318_318238

theorem min_sum_permutations_eq_44
  (a b c : Fin 4 → ℕ)
  (ha : Multiset.ofFn a = {1, 2, 3, 4})
  (hb : Multiset.ofFn b = {1, 2, 3, 4})
  (hc : Multiset.ofFn c = {1, 2, 3, 4}) :
  (∑ i : Fin 4, a i * b i * c i) = 44 :=
sorry

end min_sum_permutations_eq_44_l318_318238


namespace find_k_l318_318296

-- Definitions based on the conditions in the problem
def satisfies_arithmetic_progression (a b c: ℝ): Prop :=
  2 * b = a + c

-- Proving the main statement given the conditions and the correct answer
theorem find_k (k : ℤ) (h1 : satisfies_arithmetic_progression (√(49 + k)) (√(361 + k)) (√(784 + k))) : k = 3059 :=
by
  sorry

end find_k_l318_318296


namespace fraction_shaded_l318_318460

theorem fraction_shaded (length width : ℕ) (h_length : length = 15)
  (h_width : width = 20) (h_fraction : 1 / 3 = 1 / 3) (h_shaded : 1 / 2 = 1 / 2) :
  (length * width) * (1 / 3) * (1 / 2) / (length * width) = 1 / 6 :=
by
  -- Definitions from the conditions
  let area := length * width
  have h_area : area = 300 := by rw [h_length, h_width]; norm_num
  let one_third := area * (1 / 3)
  have h_one_third : one_third = 100 := by rw [←h_area]; norm_num
  let shaded_area := one_third * (1 / 2)
  have h_shaded_area : shaded_area = 50 := by rw [←h_one_third]; norm_num
  have fraction := shaded_area / area
  have h_fraction := (fraction = 1 / 6) := by rw [←h_shaded_area, ←h_area]; norm_num
  exact h_fraction

end fraction_shaded_l318_318460


namespace emily_initial_toys_l318_318521

theorem emily_initial_toys : ∃ (initial_toys : ℕ), initial_toys = 3 + 4 :=
by
  existsi 7
  sorry

end emily_initial_toys_l318_318521


namespace rectangle_area_l318_318020

variables (c d w : ℝ)
noncomputable def length := w + 3

theorem rectangle_area {c d w : ℝ} (h : w^2 + (w + 3)^2 = (c + d)^2) : w * (w + 3) = w * length :=
by
  -- Assuming the condition h, we want to prove the area formula.
  sorry

end rectangle_area_l318_318020


namespace evaluate_expression_l318_318190

theorem evaluate_expression : (18 ^ 12) / (54 ^ 6) = 46656 :=
by 
  have h₁ : 54 = 18 * 3 := by norm_num
  have h₂ : 54 ^ 6 = (18 * 3) ^ 6 := by rw [h₁]
  have h₃ : (18 * 3) ^ 6 = 18 ^ 6 * 3 ^ 6 := by norm_num
  rw [h₂, h₃]
  have h₄ : (18 ^ 12) / (18 ^ 6 * 3 ^ 6) = (18 ^ (12 - 6)) / 3 ^ 6 := by
    rw [pow_sub, ← div_div] ; norm_num
  rw [h₄]
  have h₅ : 18 ^ 6 / 3 ^ 6 = (18 / 3) ^ 6 := by norm_num
  rw [h₅]
  norm_num
  exact rfl

end evaluate_expression_l318_318190


namespace game_winner_l318_318065

theorem game_winner : 
  ∀ (a : ℤ), a = 2022 → 
    (∃ n : ℕ, (∃ f : fin n → ℤ, 
      (f 0 = a) ∧ 
      (∀ i < n-1, i.even ∨ i.odd →
        (f (i+1) = f i - 3 ∨
         f (i+1) = f i - (f i - 2) % 7 ∨ 
         f (i+1) = f i - (5 * (f i) - 1) % 7) ∧ (f n < 0) ∧ ∀ j < n, f j > 0))) since 
    the first player who moves can always force the sequence to a point where the second player loses :=
    sorry

end game_winner_l318_318065


namespace triangle_altitude_l318_318474

theorem triangle_altitude
  {A B C : Type*}
  {side1 side2 side3 : real}
  (h1 : side1 = 1)  -- side length of the square
  (K L M N : Type*)
  (h2 : inscribed_square K L M N A B C) -- the square KLMN is inscribed in triangle ABC
  (h3 : area_square K L M N = (1/2) * (area_triangle A B C)) -- the area of the square is half the area of the triangle
  : altitude B (line A C) A B C = 2 := 
sorry

end triangle_altitude_l318_318474


namespace cos_squared_difference_l318_318835

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318835


namespace domain_of_f_l318_318174

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 15))

theorem domain_of_f : {x : ℝ | f x ∈ ℝ} = {x : ℝ | x ≠ 9} :=
by sorry

end domain_of_f_l318_318174


namespace crayons_produced_l318_318989

theorem crayons_produced (colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) 
  (h_colors : colors = 4) (h_crayons_per_color : crayons_per_color = 2) 
  (h_boxes_per_hour : boxes_per_hour = 5) (h_hours : hours = 4) : 
  colors * crayons_per_color * boxes_per_hour * hours = 160 := 
by
  rw [h_colors, h_crayons_per_color, h_boxes_per_hour, h_hours]
  norm_num

end crayons_produced_l318_318989


namespace intersection_eq_singleton_zero_l318_318265

-- Definition of the sets M and N
def M : Set ℤ := {0, 1}
def N : Set ℤ := { x | ∃ n : ℤ, x = 2 * n }

-- The theorem stating that the intersection of M and N is {0}
theorem intersection_eq_singleton_zero : M ∩ N = {0} :=
by
  sorry

end intersection_eq_singleton_zero_l318_318265


namespace num_combinations_two_dresses_l318_318994

def num_colors : ℕ := 4
def num_patterns : ℕ := 5

def combinations_first_dress : ℕ := num_colors * num_patterns
def combinations_second_dress : ℕ := (num_colors - 1) * (num_patterns - 1)

theorem num_combinations_two_dresses :
  (combinations_first_dress * combinations_second_dress) = 240 := by
  sorry

end num_combinations_two_dresses_l318_318994


namespace coeff_a1_expansion_l318_318231

theorem coeff_a1_expansion (a : Fin 11 → ℤ) :
  (x^2 + 2 * x + 3) ^ 5 = ∑ i in Finset.range 11, a i * x^i →
  a 1 = 810 :=
by
  sorry

end coeff_a1_expansion_l318_318231


namespace rate_per_kg_of_grapes_l318_318273

theorem rate_per_kg_of_grapes : 
  ∀ (rate_per_kg_grapes : ℕ), 
    (10 * rate_per_kg_grapes + 9 * 55 = 1195) → 
    rate_per_kg_grapes = 70 := 
by
  intros rate_per_kg_grapes h
  sorry

end rate_per_kg_of_grapes_l318_318273


namespace correct_product_l318_318314

theorem correct_product (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : a.digits.reverse.toNat * b = 172) :
    a * b = 136 :=
  sorry

end correct_product_l318_318314


namespace bug_travel_distance_l318_318561

-- Define the vertices A and B in a lattice of regular hexagons
structure Point := (x : ℤ) (y : ℤ)

-- Define the distance between two points in the hexagonal lattice
def distance (A B : Point) : ℝ :=
  let dx := B.x - A.x
  let dy := B.y - A.y
  real.sqrt (3 * (dx^2 + dx * dy + dy^2))

-- Define the condition that a bug travels the shortest path from A to B
def shortest_path (A B : Point) : ℤ :=
  abs (B.x - A.x) + abs (B.y - A.y)

-- Definition of the problem
theorem bug_travel_distance (A B : Point) : 
  let AB := distance A B in
  AB / 2 ≤ shortest_path A B
  ∧ (AB / 2 = shortest_path A B → shortest_path A B = 3) :=
by
  sorry

end bug_travel_distance_l318_318561


namespace volume_remaining_proof_l318_318550

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof_l318_318550


namespace variance_of_data_l318_318050

theorem variance_of_data :
  let data := [3, 1, 0, -1, -3]
  let mean := (3 + 1 + 0 - 1 - 3) / (5:ℝ)
  let variance := (1 / 5:ℝ) * (3^2 + 1^2 + (-1)^2 + (-3)^2)
  variance = 4 := sorry

end variance_of_data_l318_318050


namespace max_value_inequality_l318_318622

open Complex

theorem max_value_inequality (z : ℂ) (h : abs z = 2) : 
  abs ((z^2 - z + 1) / (2*z - 1 - complex.I * (sqrt 3))) ≤ (3 / 2) := 
sorry

end max_value_inequality_l318_318622


namespace value_of_a_minus_3_l318_318152

-- We assume an invertible function f : ℝ → ℝ
variable {f : ℝ → ℝ}
-- Given conditions
hypothesis h1 : f 3 = a
hypothesis h2 : f a = 3

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = 0 :=
sorry

end value_of_a_minus_3_l318_318152


namespace area_triangle_PCB_correct_l318_318034

noncomputable def area_of_triangle_PCB (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) : ℝ :=
  6

theorem area_triangle_PCB_correct (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) :
  area_APB = 4 ∧ area_CPD = 9 → area_of_triangle_PCB ABCD A B C D P AB_parallel_CD diagonals_intersect_P area_APB area_CPD = 6 :=
by
  sorry

end area_triangle_PCB_correct_l318_318034


namespace convert_to_rectangular_form_l318_318169

theorem convert_to_rectangular_form :
  exp(13 * real.pi * complex.I / 3) = (1 / 2 : ℂ) + (complex.I * (real.sqrt 3 / 2)) :=
by
  sorry

end convert_to_rectangular_form_l318_318169


namespace cos_difference_identity_l318_318938

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318938


namespace combined_baseball_total_l318_318310

theorem combined_baseball_total :
  let 
    -- Day 1
    missesA1 := 60
    hitsA1 := missesA1 / 2
    homeRunsA1 := hitsA1 / 2
    singlesA1 := hitsA1 / 2

    missesB1 := 90
    hitsB1 := missesB1 / 3
    homeRunsB1 := hitsB1 / 2
    singlesB1 := hitsB1 / 2

    -- Day 2
    missesA2 := 68
    hitsA2 := missesA2 / 4
    doublesA2 := hitsA2 / 3
    singlesA2 := hitsA2 - doublesA2

    missesB2 := 56
    hitsB2 := missesB2 / 2
    doublesB2 := hitsB2 / 3
    singlesB2 := hitsB2 - doublesB2

    -- Day 3
    missesA3 := 100
    hitsA3 := missesA3 / 5
    triplesA3 := hitsA3 / 4
    homeRunsA3 := hitsA3 / 4
    singlesA3 := hitsA3 - (triplesA3 + homeRunsA3)

    missesB3 := 120
    hitsB3 := missesB3 / 5
    triplesB3 := hitsB3 / 4
    homeRunsB3 := hitsB3 / 4
    singlesB3 := hitsB3 - (triplesB3 + homeRunsB3)

    -- Calculate Combined Totals
    combined_misses := missesA1 + missesA2 + missesA3 + missesB1 + missesB2 + missesB3
    combined_singles := singlesA1 + singlesA2 + singlesA3 + singlesB1 + singlesB2 + singlesB3
    combined_doubles := doublesA2 + doublesB2
    combined_triples := triplesA3 + triplesB3
    combined_homeRuns := homeRunsA1 + homeRunsA3 + homeRunsB1 + homeRunsB3
    combined_total := combined_misses + combined_singles + combined_doubles + combined_triples + combined_homeRuns
  in 
    combined_total = 643 :=
sorry

end combined_baseball_total_l318_318310


namespace mozi_launch_indicates_respect_laws_and_unity_motion_stillness_l318_318374

-- Given Conditions
axiom quantum_communication (M : Prop) (O : Prop): M → O
axiom respect_objective_laws_and_exert_subjective_initiative : Prop
axiom matter_is_unity_of_motion_and_stillness : Prop

-- Theorem Statement
theorem mozi_launch_indicates_respect_laws_and_unity_motion_stillness :
  (quantum_communication (respect_objective_laws_and_exert_subjective_initiative)
  (matter_is_unity_of_motion_and_stillness)) →
  (respect_objective_laws_and_exert_subjective_initiative ∧ matter_is_unity_of_motion_and_stillness) :=
sorry

end mozi_launch_indicates_respect_laws_and_unity_motion_stillness_l318_318374


namespace trapezoid_properties_l318_318311

def a := 12.35 -- meters
def b := 11.2  -- meters

def angle1 := 52 + 9 / 60 -- degrees
def angle2 := 81 + 36 / 60 -- degrees
def angle3 := 98 + 24 / 60 -- degrees
def angle4 := 127 + 21 / 60 -- degrees

def area := 134.35 -- square meters

theorem trapezoid_properties:
  ∃ h: ℝ,
  let A := (1 / 2) * (a + b) * h in
  angle1 = 52 + 9 / 60 ∧ angle2 = 81 + 36 / 60 ∧ 
  angle3 = 98 + 24 / 60 ∧ angle4 = 127 + 21 / 60 ∧ 
  A = 134.35 :=
sorry

end trapezoid_properties_l318_318311


namespace cos_squared_difference_l318_318915

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318915


namespace evaluate_fraction_l318_318522

-- Let's restate the problem in Lean
theorem evaluate_fraction :
  (∃ q, (2024 / 2023 - 2023 / 2024) = 4047 / q) :=
by
  -- Substitute a = 2023
  let a := 2023
  -- Provide the value we expect for q to hold in the reduced fraction.
  use (a * (a + 1)) -- The expected denominator
  -- The proof for the theorem is omitted here
  sorry

end evaluate_fraction_l318_318522


namespace mutually_exclusive_but_not_complementary_pair_l318_318307

-- Define the finite types for boys and girls
inductive Boy: Type
| b1 | b2 | b3

inductive Girl: Type
| g1 | g2

def students := (Boy ⊕ Girl)

noncomputable def chosen : Finset students := 
  Finset.singleton (Sum.inl Boy.b1) ∪ Finset.singleton (Sum.inl Boy.b2)

-- Define events
def at_least_one_boy (s : Finset students) : Prop :=
  ∃ b : Boy, Sum.inl b ∈ s

def all_girls (s : Finset students) : Prop :=
  ∀ g : Girl, Sum.inr g ∈ s

-- Define mutual exclusiveness and complementarity
def mutually_exclusive (P Q : Finset students → Prop) : Prop :=
  ∀ s, ¬(P s ∧ Q s)

def complementary (P Q : Finset students → Prop) : Prop :=
  mutually_exclusive P Q ∧ ∀ s, P s ∨ Q s

-- Prove that the pair of events (at least 1 boy) and (all girls) are mutually exclusive but not complementary
theorem mutually_exclusive_but_not_complementary_pair :
  mutually_exclusive at_least_one_boy all_girls ∧ ¬complementary at_least_one_boy all_girls :=
by
  sorry

end mutually_exclusive_but_not_complementary_pair_l318_318307


namespace test_two_categorical_features_l318_318420

-- Definitions based on the problem conditions
def is_testing_method (method : String) : Prop :=
  method = "Three-dimensional bar chart" ∨
  method = "Two-dimensional bar chart" ∨
  method = "Contour bar chart" ∨
  method = "Independence test"

noncomputable def correct_method : String :=
  "Independence test"

-- Theorem statement based on the problem and solution
theorem test_two_categorical_features :
  ∀ m : String, is_testing_method m → m = correct_method :=
by
  sorry

end test_two_categorical_features_l318_318420


namespace range_of_a_l318_318032

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → quadratic_function a x ≥ quadratic_function a y ∧ y ≤ 4) →
  a ≤ -5 :=
by sorry

end range_of_a_l318_318032


namespace trigonometric_identity_l318_318856

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318856


namespace max_gcd_of_15m_plus_4_and_14m_plus_3_l318_318149

theorem max_gcd_of_15m_plus_4_and_14m_plus_3 (m : ℕ) (hm : 0 < m) :
  ∃ k : ℕ, k = gcd (15 * m + 4) (14 * m + 3) ∧ k = 11 :=
by {
  sorry
}

end max_gcd_of_15m_plus_4_and_14m_plus_3_l318_318149


namespace ratio_of_green_to_yellow_area_l318_318767

def radius_of_smaller_circle : ℝ := 1 -- diameter of 2 inches implies radius is 1
def radius_of_larger_circle : ℝ := 3 -- diameter of 6 inches implies radius is 3

def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

def area_yellow := area_of_circle radius_of_smaller_circle
def area_large := area_of_circle radius_of_larger_circle
def area_green := area_large - area_yellow

theorem ratio_of_green_to_yellow_area : area_green / area_yellow = 8 := 
by 
  sorry

end ratio_of_green_to_yellow_area_l318_318767


namespace cos_square_difference_l318_318866

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318866


namespace polynomial_value_at_zero_l318_318095

-- Define the polynomial of degree 2008
variable (P : ℕ → ℚ)

-- Conditions from the problem
def P_degree : Prop := ∀ k, k > 0 → k < 2010 → (P k = 1 / k)

-- Statement that needs to be proven
theorem polynomial_value_at_zero : 
  P_degree P →
  P 0 = ∑ k in (Finset.range 2009).map (nat.cast : ℕ → ℚ), (1 / k) := 
sorry

end polynomial_value_at_zero_l318_318095


namespace polynomial_factor_pair_l318_318511

theorem polynomial_factor_pair (a b : ℝ) :
  (∃ (c d : ℝ), 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 6)) →
  (a, b) = (-26.5, -40) :=
by
  sorry

end polynomial_factor_pair_l318_318511


namespace william_won_more_rounds_than_harry_l318_318435

def rounds_played : ℕ := 15
def william_won_rounds : ℕ := 10
def harry_won_rounds : ℕ := rounds_played - william_won_rounds
def william_won_more_rounds := william_won_rounds > harry_won_rounds

theorem william_won_more_rounds_than_harry : william_won_rounds - harry_won_rounds = 5 := 
by sorry

end william_won_more_rounds_than_harry_l318_318435


namespace symmetric_line_equation_l318_318399

-- Given conditions as definitions
def given_line := λ x y : ℝ, x + 2 * y - 3 = 0
def symmetry_line := λ x : ℝ, x = 1

-- Theorem statement
theorem symmetric_line_equation (x y : ℝ) : 
  (given_line x y) ∧ (symmetry_line x) → (x - 2 * y + 1 = 0) :=
by
  -- Proof will go here
  sorry

end symmetric_line_equation_l318_318399


namespace smallest_palindromic_prime_is_313_l318_318203

def is_palindrome (n : ℕ) : Prop :=
  Nat.digits 10 n = List.reverse (Nat.digits 10 n)

def has_hundreds_digit (n : ℕ) (d : ℕ) : Prop :=
  Nat.digits 10 n = d :: (Nat.digits 10 n).tail

def smallest_palindromic_prime_with_hundreds_digit_3 : ℕ :=
  313

theorem smallest_palindromic_prime_is_313 :
  ∃ n : ℕ, n = smallest_palindromic_prime_with_hundreds_digit_3 ∧ 
           (100 ≤ n ∧ n < 1000) ∧ 
           is_palindrome n ∧ 
           has_hundreds_digit n 3 ∧ 
           Nat.Prime n :=
by {
  use smallest_palindromic_prime_with_hundreds_digit_3,
  sorry
}

end smallest_palindromic_prime_is_313_l318_318203


namespace ab_value_l318_318316

theorem ab_value (a b : ℚ) (h1 : 3 * a - 8 = 0) (h2 : b = 3) : a * b = 8 :=
by
  sorry

end ab_value_l318_318316


namespace cos_A_minus_B_l318_318577

theorem cos_A_minus_B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = -1) 
  (h2 : Real.cos A + Real.cos B = 1/2) :
  Real.cos (A - B) = -3/8 :=
by
  sorry

end cos_A_minus_B_l318_318577


namespace average_infection_rate_l318_318118

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l318_318118


namespace orthogonal_pairs_count_eq_36_l318_318614

-- Define a cube and its properties
structure Cube where
  vertices : Finset (ℝ × ℝ × ℝ)
  edges : Finset (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)
  face_diagonals : Finset (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)

-- Conditions describing the cube
def is_cube (c : Cube) : Prop :=
  c.vertices.card = 8 ∧
  c.edges.card = 12 ∧
  c.face_diagonals.card = 12

-- Count the number of orthogonal line-plane pairs
noncomputable def count_orthogonal_pairs (c : Cube) : ℕ :=
  2 * c.edges.card + c.face_diagonals.card

-- Problem statement: the final proof statement
theorem orthogonal_pairs_count_eq_36 (c : Cube) (hcube : is_cube c) :
  count_orthogonal_pairs(c) = 36 := 
by
  sorry

end orthogonal_pairs_count_eq_36_l318_318614


namespace incorrect_permutations_of_word_l318_318298

theorem incorrect_permutations_of_word : ∀ (word : String), word.length = 4 → 
  (∃ incorrect_permutations : ℕ, incorrect_permutations = nat.factorial 4 - 1 ∧ incorrect_permutations = 23) :=
by
  intros word h_length
  use nat.factorial 4 - 1
  split
  rfl
  norm_num
  sorry

end incorrect_permutations_of_word_l318_318298


namespace monica_tiles_count_l318_318693

-- Definitions based on the problem's conditions
def room_length := 15 -- in feet
def room_width := 18 -- in feet
def border_tile_side := 2 -- in feet
def inner_tile_side := 3 -- in feet
def border_width := 2 -- in feet

-- Calculation of the total number of tiles
theorem monica_tiles_count : 
  let border_length := room_length - 2 * border_width
      border_width_tiles := (room_width - 2 * border_width) / border_tile_side
      total_border_tiles := 2 * (border_length / border_tile_side) + 2 * (border_width_tiles) + 4
      inner_length := room_length - 2 * border_width
      inner_width := room_width - 2 * border_width
      inner_area := inner_length * inner_width
      inner_tile_area := inner_tile_side * inner_tile_side
      inner_tiles := inner_area / inner_tile_area
  in total_border_tiles + inner_tiles = 45 := sorry

end monica_tiles_count_l318_318693


namespace correct_exponent_rule_l318_318077

variable (a : ℝ)

theorem correct_exponent_rule (h₁ : ¬ (a^4 + a^5 = a^9))
                             (h₂ : ¬ (a^3 * a^3 * a^3 = 3 * a^3))
                             (h₃ : a^4 * a^6 = a^{10})
                             (h₄ : ¬ ((-a^3)^6 = a^9)) : 
                             a^4 * a^6 = a^{10} := 
by 
  exact h₃

end correct_exponent_rule_l318_318077


namespace cosine_difference_identity_l318_318841

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318841


namespace cos_squared_difference_l318_318804

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318804


namespace proof_problem_l318_318569

open Nat

-- Definitions of arithmetic sequence and sum up to n terms
def a (n : ℕ) := 2 * n - 3
def S (n : ℕ) := (n * (a 1) + (n - 1) * (a (n - 1))) / 2

-- Sequence {b_n} definitions
def b (n : ℕ) := 4^(n - 1)
def T (n : ℕ) := (4^n - 1) / 3

-- The conditions and main theorem encapsulating the problem
theorem proof_problem (n : ℕ) : 
  (a 1 + a 2 = 0) → (S 5 = 15) → (b 1 = a 2) → 
  (∀ n, n * b (n + 1) + (a n + 2) * b n = a (3 * n + 1) * b n) →
    (T n = (4^n - 1) / 3) :=
by {
  intros,
  sorry
}

end proof_problem_l318_318569


namespace problem_statement_l318_318357

open Finset

-- Define the main variables and hypotheses
variables {B : Type*} [DecidableEq B] [Fintype B]
variables (n : ℕ) (A : Fin (2 * n + 1) → Finset B)

-- Conditions from the problem
def condition_1 : Prop := n > 0
def condition_2a : Prop := ∀ i, (A i).card = 2 * n

def condition_2b : Prop := ∀ (i j : Fin (2 * n + 1)), i ≠ j → (A i ∩ A j).card = 1

def condition_2c : Prop := ∀ b ∈ univ.sUnion (λ i => A i), 
  2 ≤ (univ.filter (λ i => b ∈ A i)).card

-- The statement to be proven
theorem problem_statement (h1 : condition_1 n)
                         (h2a : condition_2a A)
                         (h2b : condition_2b A)
                         (h2c : condition_2c A) : Even n :=
sorry

end problem_statement_l318_318357


namespace find_f_five_l318_318615

noncomputable def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

theorem find_f_five (y : ℝ) (h : f 2 y = 50) : f 5 y = 92 := by
  sorry

end find_f_five_l318_318615


namespace cos_squared_difference_l318_318829

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318829


namespace geometric_series_theorem_l318_318508

noncomputable def geometric_series_sum (a_1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1
  else a_1 * (q^n - 1) / (q - 1)

theorem geometric_series_theorem (a_1 q : ℝ) :
  geometric_series_sum a_1 q 200 + 1 = (geometric_series_sum a_1 q 100 + 1)^2 →
  geometric_series_sum a_1 q 600 + 1 = (geometric_series_sum a_1 q 300 + 1)^2 :=
begin
  sorry
end

end geometric_series_theorem_l318_318508


namespace calculate_monthly_installment_l318_318791

noncomputable def monthly_installment (cash_price : ℝ) (deposit_rate : ℝ) (num_months : ℕ) (annual_interest_rate : ℝ) : ℝ :=
  let deposit := cash_price * deposit_rate in
  let balance := cash_price - deposit in
  let monthly_interest_rate := annual_interest_rate / 12 in
  balance * (monthly_interest_rate * (1 + monthly_interest_rate)^num_months) / ((1 + monthly_interest_rate)^num_months - 1)

theorem calculate_monthly_installment :
  monthly_installment 25000 0.10 60 0.12 ≈ 499.30 :=
by
  -- proof goes here
  sorry

end calculate_monthly_installment_l318_318791


namespace lines_intersecting_sum_a_b_l318_318732

theorem lines_intersecting_sum_a_b 
  (a b : ℝ) 
  (hx : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x = 3 * y + a)
  (hy : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ y = 3 * x + b)
  : a + b = -10 :=
by
  sorry

end lines_intersecting_sum_a_b_l318_318732


namespace cos_diff_square_identity_l318_318823

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318823


namespace tangent_equation_at_origin_range_of_a_l318_318255

-- Condition definitions
def f (x : ℝ) : ℝ := Real.log x - x
def h (x : ℝ) (a : ℝ) : ℝ := (a * Real.exp x) / x + 1

-- Proof Problem 1
theorem tangent_equation_at_origin :
  ∀ x : ℝ, f x = Real.log x - x → ∃ l : ℝ, tangent_line f (0, 0) = l * x := 
sorry

-- Proof Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f x + h x a ≥ 0) → ∃ b : ℝ, b = 1 / (Real.exp 2) ∧ ∀ y : ℝ, y ≥ b :=
sorry

end tangent_equation_at_origin_range_of_a_l318_318255


namespace meal_combinations_l318_318632

theorem meal_combinations :
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  meats * vegetable_combinations * desserts = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  show meats * vegetable_combinations * desserts = 150
  sorry

end meal_combinations_l318_318632


namespace solution_set_inequality_l318_318750

theorem solution_set_inequality (x : ℝ) : 
  (-x^2 + 3 * x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_inequality_l318_318750


namespace negation_of_all_honest_l318_318738

-- Define the needed predicates
variable {Man : Type} -- Type for men
variable (man : Man → Prop)
variable (age : Man → ℕ)
variable (honest : Man → Prop)

-- Define the conditions and the statement we want to prove
theorem negation_of_all_honest :
  (∀ x, man x → age x > 30 → honest x) →
  (∃ x, man x ∧ age x > 30 ∧ ¬ honest x) :=
sorry

end negation_of_all_honest_l318_318738


namespace remaining_files_l318_318449

variable {files_initial_music : ℕ}
variable {files_initial_video : ℕ}
variable {files_deleted : ℕ}

theorem remaining_files (h1 : files_initial_music = 13) 
                        (h2 : files_initial_video = 30)
                        (h3 : files_deleted = 10) : 
                        files_initial_music + files_initial_video - files_deleted = 33 :=
by
  have total_files_initial : ℕ := files_initial_music + files_initial_video
  have remaining_files : ℕ := total_files_initial - files_deleted
  calc
    files_initial_music + files_initial_video - files_deleted
        = total_files_initial - files_deleted : by rfl
    ... = 43 - 10 : by rw [h1, h2, h3]
    ... = 33 : by norm_num

end remaining_files_l318_318449


namespace cos_diff_square_identity_l318_318825

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318825


namespace integer_ratio_condition_l318_318677

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end integer_ratio_condition_l318_318677


namespace cos_squared_difference_l318_318887

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318887


namespace max_winners_3_matches_200_participants_l318_318495

theorem max_winners_3_matches_200_participants (n : ℕ) (h_total : n = 200):
  ∃ (x : ℕ), (∀ y : ℕ, (3 * y ≤ 199) → y ≤ 66) ∧ (3 * 66 ≤ 199) :=
by
  use 66
  split
  · intro y h
    have : (n - 1) = 199 := by linarith
    suffices : 3 * 66 ≤ (n - 1) by linarith
    exact this
    sorry
  · linarith

end max_winners_3_matches_200_participants_l318_318495


namespace cos_squared_difference_l318_318914

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318914


namespace contradiction_proof_l318_318557

theorem contradiction_proof (x y : ℝ) (h1 : x + y < 2) (h2 : 1 < x) (h3 : 1 < y) : false := 
by 
  sorry

end contradiction_proof_l318_318557


namespace rational_product_nonpositive_l318_318235

open Classical

theorem rational_product_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 :=
by
  sorry

end rational_product_nonpositive_l318_318235


namespace journey_time_difference_l318_318105

theorem journey_time_difference :
  let speed := 40  -- mph
  let distance1 := 360  -- miles
  let distance2 := 320  -- miles
  (distance1 / speed - distance2 / speed) * 60 = 60 := 
by
  sorry

end journey_time_difference_l318_318105


namespace cos_squared_difference_l318_318891

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318891


namespace cos_squared_difference_l318_318880

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318880


namespace exists_answer_layout_l318_318162

-- Define the dimensions of the hall
def rows : ℕ := 8
def cols : ℕ := 8

-- Define the number of questions and possible answers
def questions : ℕ := 6
def answers : fin 2 := ⟨2, by norm_num⟩ -- fin 2 is 2 possible answers, 0 or 1 

-- A set of answers is represented as a bitvector of length 6
def AnswerSet := vector (fin 2) questions

-- A seating layout is a matrix of answer sets
def Layout := fin rows × fin cols → AnswerSet

-- Define adjacency in the layout
def adjacent (p q : fin rows × fin cols) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨ 
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1))

-- Function to count differing bits between two answer sets
def differing_bits (a b : AnswerSet) : ℕ :=
  vector.foldl_nat (+) 0 (vector.zip_with (λ x y, if x ≠ y then 1 else 0) a b)

-- Proof statement: We need to show that there exists a layout with the desired properties
theorem exists_answer_layout : ∃ (layout : Layout), 
  (∀ (p q : fin rows × fin cols), p ≠ q → layout p ≠ layout q) ∧ 
  (∀ (p q : fin rows × fin cols), adjacent p q → differing_bits (layout p) (layout q) ≤ 1) :=
by sorry

end exists_answer_layout_l318_318162


namespace sum_of_remainders_l318_318074

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l318_318074


namespace max_omega_for_increasing_l318_318295

noncomputable def sin_function (ω : ℕ) (x : ℝ) := Real.sin (ω * x + Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem max_omega_for_increasing : ∀ (ω : ℕ), (0 < ω) →
  is_monotonically_increasing_on (sin_function ω) (Real.pi / 6) (Real.pi / 4) ↔ ω ≤ 9 :=
sorry

end max_omega_for_increasing_l318_318295


namespace parabola_axis_l318_318590

theorem parabola_axis (f : ℝ → ℝ) (p : ℝ) 
    (hf : ∀ x, f x = x^3 + x^2 + x + 3)
    (h_tangent : ∀ x, (f' x) = 3 * x^2 + 2 * x + 1)
    (h_tangent_point : ∀ x, (x = -1) → (f x = 2)) 
    (h_tangent_line : ∀ x, (x = -1) → (2 * x + 4 ≥ 2px^2)) :
  axis_eqn : ∀ x, (y = 2 * p * x^2) → (axis_eqn y = 1) := 
by
  sorry

end parabola_axis_l318_318590


namespace odd_integer_in_list_l318_318434

theorem odd_integer_in_list : 
  let a := 6^2,
      b := 23 - 17,
      c := 9 * 24,
      d := 96 / 8,
      e := 9 * 41 in
  e % 2 = 1 :=
by
  let a : ℤ := 6^2
  let b : ℤ := 23 - 17
  let c : ℤ := 9 * 24
  let d : ℤ := 96 / 8
  let e : ℤ := 9 * 41
  sorry

end odd_integer_in_list_l318_318434


namespace exists_point_on_euler_line_l318_318383

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

structure triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
(A : A)
(B : B)
(C : C)

variables (T : triangle A B C)

noncomputable def centroid (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] : Type := 
sorry

noncomputable def euler_line (T : triangle A B C) : Type := sorry

noncomputable def distance (X Y : Type) [metric_space X] [metric_space Y] : ℝ := sorry

theorem exists_point_on_euler_line (T : triangle A B C) :
  ∃ (P : Type), P ∈ euler_line T ∧ 
  let G1 := centroid A B P,
      G2 := centroid B C P,
      G3 := centroid C A P in
  distance G1 C = distance G2 A ∧ distance G2 A = distance G3 B :=
sorry

end exists_point_on_euler_line_l318_318383


namespace isosceles_triangle_area_l318_318061

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_area :
  let a := 25
  let b := 25
  let c := 48
  triangle_area a b c = 168 := by
let a := 25
let b := 25
let c := 48
have h1 : triangle_area a b c = 168 := by sorry
exact h1

end isosceles_triangle_area_l318_318061


namespace crayon_production_correct_l318_318992

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l318_318992


namespace cube_cut_possible_l318_318645

theorem cube_cut_possible (a b : ℝ) (unit_a : a = 1) (unit_b : b = 1) : 
  ∃ (cut : ℝ → ℝ → Prop), (∀ x y, cut x y → (∃ q r : ℝ, q > 0 ∧ r > 0 ∧ q * r > 1)) :=
sorry

end cube_cut_possible_l318_318645


namespace sum_not_unchanged_l318_318412
-- Import the necessary Lean 4 and Mathlib modules

-- Define the problem statement
theorem sum_not_unchanged (A B : ℕ) (h : A + B = 2022) : (A % 2 = 0 → B % 3 = 0 → \(\frac{A}{2} + 3B ≠ A + B\))
| h evenA mod2 => by sorry

end sum_not_unchanged_l318_318412


namespace blot_is_circle_l318_318978

theorem blot_is_circle {A B : Point} (blot : set Point)
  (min_distance_to_boundary : Point → ℝ)
  (max_distance_to_boundary : Point → ℝ)
  (r_0 R_0 : ℝ) :
  (∀ p ∈ blot, min_distance_to_boundary p ≤ max_distance_to_boundary p) →
  (∀ p ∈ blot, r_0 = min_distance_to_boundary p) →
  (∀ p ∈ blot, R_0 = max_distance_to_boundary p) →
  (r_0 = R_0) →
  (∃ ρ : ℝ, ∀ p ∈ blot, min_distance_to_boundary p = ρ ∧ max_distance_to_boundary p = ρ) :=
sorry

end blot_is_circle_l318_318978


namespace proportion_solution_l318_318795

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 4.5 / (7 / 3)) : x = 0.3888888889 :=
by
  sorry

end proportion_solution_l318_318795


namespace geometric_series_smallest_b_l318_318667

theorem geometric_series_smallest_b (a b c : ℝ) (h_geometric : a * c = b^2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 216) : b = 6 :=
sorry

end geometric_series_smallest_b_l318_318667


namespace triangle_perimeter_is_16_l318_318534

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (6, 6)

-- Calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Function to calculate the perimeter of triangle ABC
def triangle_perimeter (A B C : ℝ × ℝ) : ℝ := 
  distance A B + distance B C + distance C A

-- The theorem statement that the perimeter of the triangle is 16
theorem triangle_perimeter_is_16 : triangle_perimeter A B C = 16 := 
sorry

end triangle_perimeter_is_16_l318_318534


namespace diameter_in_scientific_notation_l318_318398

def diameter : ℝ := 0.00000011
def scientific_notation (d : ℝ) : Prop := d = 1.1e-7

theorem diameter_in_scientific_notation : scientific_notation diameter :=
by
  sorry

end diameter_in_scientific_notation_l318_318398


namespace coeff_x4_in_expansion_l318_318317

open Nat

theorem coeff_x4_in_expansion : 
  let expr := fun x : ℚ => (x - 1 / (2 * x)) ^ 6 in
  (expr 0).coeff (Polynomial.X 4) = -3 :=
sorry

end coeff_x4_in_expansion_l318_318317


namespace slope_of_perpendicular_l318_318780

variable (x1 y1 x2 y2 : ℝ)

def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

noncomputable def perpendicular_slope := -1 / slope x1 y1 x2 y2

theorem slope_of_perpendicular (h : slope (-3) 5 6 (-4) = -1) :
  perpendicular_slope (-3) 5 6 (-4) = 1 :=
by
  unfold slope perpendicular_slope
  rw [h]
  norm_num
  -- place to insert additional 'sorry' if other additional unfolding is needed

end slope_of_perpendicular_l318_318780


namespace solve_xyz_integers_l318_318714

theorem solve_xyz_integers (x y z : ℤ) : x^2 + y^2 + z^2 = 2 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end solve_xyz_integers_l318_318714


namespace average_speed_of_car_l318_318798

theorem average_speed_of_car 
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (total_time : ℕ)
  (h1 : speed_first_hour = 90)
  (h2 : speed_second_hour = 40)
  (h3 : total_time = 2) : 
  (speed_first_hour + speed_second_hour) / total_time = 65 := 
by
  sorry

end average_speed_of_car_l318_318798


namespace total_carrots_l318_318712

theorem total_carrots (carrots_sandy carrots_mary : ℕ) (h1 : carrots_sandy = 8) (h2 : carrots_mary = 6) :
  carrots_sandy + carrots_mary = 14 :=
by
  sorry

end total_carrots_l318_318712


namespace no_tangent_line_l318_318043

-- Define the function f(x) = x^3 - 3ax
def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

-- Proposition stating no b exists in ℝ such that y = -x + b is tangent to f
theorem no_tangent_line (a : ℝ) (H : ∀ b : ℝ, ¬ ∃ x : ℝ, f' a x = -1) : a < 1 / 3 :=
by
  sorry

end no_tangent_line_l318_318043


namespace sum_of_remainders_l318_318075

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l318_318075


namespace decimal_to_fraction_l318_318080

theorem decimal_to_fraction (x : ℚ) (h : x = 3.675) : x = 147 / 40 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l318_318080


namespace cos_diff_square_identity_l318_318816

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318816


namespace decreasing_f_range_l318_318254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem decreasing_f_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end decreasing_f_range_l318_318254


namespace projection_matrix_correct_l318_318664

variable (u : ℝ^3) (Q : ℝ^3 → Prop)
variable (normal : ℝ^3 := ![2, -1, 3])

theorem projection_matrix_correct (Q_pass : Q 0) (Q_normal : ∀ v, Q v ↔ v ⋅ normal = 0) :
  let Q_matrix := λ (x : ℝ^3), ![
    [ 5/7,  3/14, -2/7],
    [ 2/7,  5/14, 11/14],
    [-2/7, 11/14,  5/14]
  ] in
  ∀ v : ℝ^3, Q (Q_matrix v) :=
by
  sorry

end projection_matrix_correct_l318_318664


namespace scientific_notation_of_probe_unit_area_l318_318078

def probe_unit_area : ℝ := 0.0000064

theorem scientific_notation_of_probe_unit_area :
  ∃ (mantissa : ℝ) (exponent : ℤ), probe_unit_area = mantissa * 10^exponent ∧ mantissa = 6.4 ∧ exponent = -6 :=
by
  sorry

end scientific_notation_of_probe_unit_area_l318_318078


namespace percentage_is_50_l318_318463

theorem percentage_is_50 (P : ℝ) (h1 : P = 0.20 * 15 + 47) : P = 50 := 
by
  -- skip the proof
  sorry

end percentage_is_50_l318_318463


namespace joan_books_l318_318649

theorem joan_books : 
  ∀ (initial_books yard_sale_books day1_books day2_books day3_books : ℕ), 
  initial_books = 75 → 
  yard_sale_books = 33 → 
  day1_books = 15 → 
  day2_books = 8 → 
  day3_books = 12 → 
  initial_books - yard_sale_books - day1_books - day2_books - day3_books = 7 := 
by 
  intros initial_books yard_sale_books day1_books day2_books day3_books 
  intros h_initial_books h_yard_sale_books h_day1_books h_day2_books h_day3_books 
  rw [h_initial_books, h_yard_sale_books, h_day1_books, h_day2_books, h_day3_books] 
  exact Nat.sub_sub_sub_sub_eq 75 33 15 8 12 7 sorry

end joan_books_l318_318649


namespace number_of_mappings_l318_318165

noncomputable def countMappings (n : ℕ) (X : Type) [Fintype X] [DecidableEq X]
  (f : X → X) (a : X) (h1 : n ≥ 2) 
  (h2 : ∀ x : X, f (f x) = a) (h3 : a ∈ X) : ℕ :=
∑ k in Finset.range (n - 1) \ {0}, Nat.choose (n - 1) k * k ^ (n - k - 1)

theorem number_of_mappings (n : ℕ) (X : Type) [Fintype X] [DecidableEq X] 
  (f : X → X) (a : X) (h1 : n ≥ 2)
  (h2 : ∀ x : X, f (f x) = a) (h3 : a ∈ X) :
  ∃ (k_set : Finset ℕ), k_set = Finset.range (n - 1) \ {0} 
  ∧ countMappings n X f a h1 h2 h3 = ∑ k in k_set, Nat.choose (n - 1) k * k ^ (n - k - 1) :=
sorry

end number_of_mappings_l318_318165


namespace lamps_all_off_eventually_l318_318416

theorem lamps_all_off_eventually (n : ℕ) : (∃ k : ℕ, n = 4 * k + 3) →
  (∃ T : ℕ, ∀ t ≥ T, ∀ (L : fin n → bool),
    (∀ i : fin n, t = 0 → (i.val = 0 → L i = tt) ∧ (i.val ≠ 0 → L i = ff)) →
    (∀ i : fin n, (L (i - 1) = L i ∧ L (i + 1) = L i) → L i = ff) ∧
    (L (i - 1) ≠ L i ∨ L (i + 1) ≠ L i) → L i = tt →
    ∀ i : fin n, L i = ff) :=
begin
  intros hn,
  -- the proof goes here
  sorry
end

end lamps_all_off_eventually_l318_318416


namespace prism_triples_count_l318_318130

/-- A right rectangular prism Q with integral sides a, b, c and a ≤ b ≤ c.
    A plane parallel to one of the faces of Q cuts Q into two prisms,
    one of which is similar to Q. 
    Given b = 500 and a is odd,
    we need to determine the number of ordered triples (a, b, c) that permit such a plane to exist. -/ 
theorem prism_triples_count :
  ∃ (n : ℕ), n = 3 ∧ ∀ (a b c : ℕ), 
    b = 500 ∧ a % 2 = 1 ∧ a ≤ b ∧ b ≤ c ∧ (a * c = 250000) ↔ a ∈ {1, 25, 125} :=
begin
  sorry
end

end prism_triples_count_l318_318130


namespace cos_squared_difference_l318_318839

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318839


namespace final_people_amount_l318_318151

def initial_people : ℕ := 250
def people_left1 : ℕ := 35
def people_joined1 : ℕ := 20
def percentage_left : ℕ := 10
def groups_joined : ℕ := 4
def group_size : ℕ := 15

theorem final_people_amount :
  let intermediate_people1 := initial_people - people_left1;
  let intermediate_people2 := intermediate_people1 + people_joined1;
  let people_left2 := (intermediate_people2 * percentage_left) / 100;
  let rounded_people_left2 := people_left2;
  let intermediate_people3 := intermediate_people2 - rounded_people_left2;
  let total_new_join := groups_joined * group_size;
  let final_people := intermediate_people3 + total_new_join;
  final_people = 272 :=
by sorry

end final_people_amount_l318_318151


namespace rhombus_perimeter_52_l318_318021

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l318_318021


namespace complex_sum_identity_l318_318673

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity_l318_318673


namespace cos_double_angle_l318_318344

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) :
  Real.cos (2 * θ) = -7 / 9 :=
sorry

end cos_double_angle_l318_318344


namespace infinitely_many_a_l318_318544

theorem infinitely_many_a (n : ℕ) : ∃ (a : ℕ), ∃ (k : ℕ), ∀ n : ℕ, n^6 + 3 * (3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3) = (n^2 + 3 * k)^3 :=
by
  sorry

end infinitely_many_a_l318_318544


namespace range_of_k_l318_318625

theorem range_of_k (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_ne : x ≠ 2) :
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) ↔ (k > -2 ∧ k ≠ 4) :=
by
  sorry

end range_of_k_l318_318625


namespace main_proof_l318_318670

namespace ProofProblem

-- Conditions and definitions
def f (x a : ℝ) : ℝ := x - (2 / x) - a * (Real.log x - 1 / (x * x))
def g (a : ℝ) : ℝ := a - a * Real.log a - 1 / a

-- Statement of the problem
theorem main_proof (a : ℝ) (h : a > 0) : g(a) < 1 := by
  sorry

end ProofProblem

end main_proof_l318_318670


namespace distance_A_B_general_distance_A_B_parallel_y_l318_318704

-- Proof problem for Question 1:
theorem distance_A_B_general 
  (x1 y1 x2 y2 : ℝ) 
  (hx1 : x1 = 2) (hy1 : y1 = 4)
  (hx2 : x2 = -3) (hy2 : y2 = -8) : 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 13 := 
by {
  rw [hx1, hy1, hx2, hy2],
  calc real.sqrt ((-3 - 2)^2 + (-8 - 4)^2)
      = real.sqrt ((-5)^2 + (-12)^2) : by norm_num
  ... = real.sqrt (25 + 144) : by norm_num
  ... = real.sqrt (169) : by norm_num
  ... = 13 : by norm_num }

-- Proof problem for Question 2:
theorem distance_A_B_parallel_y 
  (y1 y2 : ℝ) 
  (hy1 : y1 = 5) (hy2 : y2 = -1) : 
  |y2 - y1| = 6 := 
by {
  rw [hy1, hy2],
  calc |(-1) - 5|
      = |(-6)| : by norm_num 
  ... = 6 : by norm_num }

end distance_A_B_general_distance_A_B_parallel_y_l318_318704


namespace sequence_inequality_l318_318657

def sequence_u : ℕ → ℝ 
| 1       := 1
| (n + 1) := sequence_u n + 1 / sequence_u n

theorem sequence_inequality (n : ℕ) (hn : n ≥ 1) : 
  sequence_u n ≤ (3 * Real.sqrt n) / 2 :=
sorry

end sequence_inequality_l318_318657


namespace cosine_difference_identity_l318_318845

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318845


namespace men_in_first_group_l318_318983

theorem men_in_first_group (M : ℕ) (h1 : ∀ W, W = M * 30) (h2 : ∀ W, W = 10 * 36) : 
  M = 12 :=
by
  sorry

end men_in_first_group_l318_318983


namespace subtraction_verification_l318_318502

theorem subtraction_verification : 888888888888 - 111111111111 = 777777777777 :=
by
  sorry

end subtraction_verification_l318_318502


namespace problem_mean_minus_median_is_32_l318_318308

def contestant_scores : List (ℕ × ℤ) := 
  [ 
    (12, 60), -- 12% scored 60 points
    (20, 85), -- 20% scored 85 points
    (38, 95), -- 38% scored 95 points
    (30, 105) -- 30% scored 105 points
  ]

theorem problem_mean_minus_median_is_32 (hs : contestant_scores) : 
  let mean := (hs[0].fst * hs[0].snd + hs[1].fst * hs[1].snd + hs[2].fst * hs[2].snd + hs[3].fst * hs[3].snd) / 100 in
  let median := hs[2].snd in
  median - mean = 3.2 :=
  sorry

end problem_mean_minus_median_is_32_l318_318308


namespace auditorium_earnings_l318_318466

theorem auditorium_earnings
    (ticket_cost : ℕ)
    (rows : ℕ) 
    (seats_per_row : ℕ)
    (sold_ratio : ℚ) :
    ticket_cost = 10 →
    rows = 20 →
    seats_per_row = 10 →
    sold_ratio = 3 / 4 →
    let total_seats := rows * seats_per_row in
    let sold_seats := (sold_ratio * total_seats) in
    let earnings := (sold_seats.numerator * ticket_cost) / sold_seats.denominator in
    earnings = 1500 := by
  sorry

end auditorium_earnings_l318_318466


namespace triangle_area_correct_l318_318480

noncomputable def triangle_area : ℝ :=
  let A : ℝ×ℝ := (2, 8)
  let B : ℝ×ℝ := (1, 13)
  let C : ℝ×ℝ := (5, 10)
  in (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A := (2, 8)
  let B := (1, 13)
  let C := (5, 10)
  in triangle_area = 17 / 2 :=
by
  sorry

end triangle_area_correct_l318_318480


namespace find_m_for_parallel_lines_l318_318270

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
by
  sorry

end find_m_for_parallel_lines_l318_318270


namespace problem_solution_l318_318472

noncomputable def sequence (a : ℕ → ℂ) (n : ℕ) : ℂ :=
if h : n = 0 then a 0 else 2 * (a (n - 1)).re - (a (n - 1)).im + (2 * (a (n - 1)).im + (a (n - 1)).re) * complex.I

theorem problem_solution (a b : ℕ → ℂ) :
  (a 50, b 50) = (5, -3) →
  (a 0 + b 0) = -1 / 5^(47 / 2) :=
by
  intros h
  sorry

end problem_solution_l318_318472


namespace distance_AB_13_distance_AB_y_parallel_l318_318705

/-- Proof Problem 1: Distance between points A(2, 4) and B(-3, -8) is 13 -/
theorem distance_AB_13 (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = 4) (h3 : x2 = -3) (h4 : y2 = -8) : 
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 13 :=
by
  rw [h1, h2, h3, h4]
  sorry

/-- Proof Problem 2: Distance between points with coordinates (a, 5) and (b, -1) lying on a line parallel to the y-axis is 6 -/
theorem distance_AB_y_parallel (yA yB : ℝ) (hA : yA = 5) (hB : yB = -1) : 
  real.abs (yA - yB) = 6 :=
by
  rw [hA, hB]
  sorry

end distance_AB_13_distance_AB_y_parallel_l318_318705


namespace sport_formulation_water_quantity_l318_318089

theorem sport_formulation_water_quantity (flavoring : ℝ) (corn_syrup : ℝ) (water : ℝ)
    (hs : flavoring / corn_syrup = 1 / 12) 
    (hw : flavoring / water = 1 / 30) 
    (sport_fs_ratio : flavoring / corn_syrup = 3 * (1 / 12)) 
    (sport_fw_ratio : flavoring / water = (1 / 2) * (1 / 30)) 
    (cs_sport : corn_syrup = 1) : 
    water = 15 :=
by
  sorry

end sport_formulation_water_quantity_l318_318089


namespace exist_invertible_matrices_rank_of_power_matrices_l318_318655

-- Part (a)
theorem exist_invertible_matrices 
  (A : Matrix (Fin 4) (Fin 4) ℂ) (r : ℕ) (hr : r < 4) (rank_A : rank A = r) :
  ∃ (U V : Matrix (Fin 4) (Fin 4) ℂ), isInvertible U ∧ isInvertible V ∧ 
  (U ⬝ A ⬝ V = (Matrix.blockDiagonal (λ i, if i = 0 then 1 else 0) (Fin r) (Fin (4 - r)) : Matrix (Fin 4) (Fin 4) ℂ)) := 
sorry

-- Part (b)
theorem rank_of_power_matrices
  (A : Matrix (Fin 4) (Fin 4) ℂ) (k : ℕ) 
  (rank_A : rank A = k) (rank_A2 : rank (A ⬝ A) = k) (n : ℕ) (hn : n ≥ 3) :
  rank (A^n) = k :=
sorry

end exist_invertible_matrices_rank_of_power_matrices_l318_318655


namespace cos_difference_squared_l318_318967

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318967


namespace max_pons_l318_318154

-- Define variables and conditions
variables (pan pin pon : ℕ)
variables (h_pan : 1 ≤ pan) (h_pin : 1 ≤ pin) (h_pon : 1 ≤ pon)
variables (h_total : 3 * pan + 5 * pin + 10 * pon = 100)

-- The claim statement in Lean 4
theorem max_pons : (∃ pan pin, 3 * pan + 5 * pin + 10 * 9 = 100 ∧ 1 ≤ pan ∧ 1 ≤ pin) :=
begin
  use [1, 1], -- This sets pan = 1 and pin = 1 for simplicty
  split,
  { exact h_total },
  split,
  { exact h_pan },
  { exact h_pin }
end

end max_pons_l318_318154


namespace unique_integer_sequence_l318_318000

theorem unique_integer_sequence :
  ∃ a : ℕ → ℤ, a 1 = 1 ∧ a 2 > 1 ∧ ∀ n ≥ 1, (a (n + 1))^3 + 1 = a n * a (n + 2) :=
sorry

end unique_integer_sequence_l318_318000


namespace coefficient_of_x_squared_in_expansion_l318_318641

theorem coefficient_of_x_squared_in_expansion : 
  (∃ c : ℝ, (∑ r in Finset.range (5+1), (Nat.choose 5 r) * (-2)^r * x^((5-r)/2)) = c * x^2) := 
begin
  use -10,
  sorry
end

end coefficient_of_x_squared_in_expansion_l318_318641


namespace arithmetic_geometric_means_l318_318707

theorem arithmetic_geometric_means (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 125) : x^2 + y^2 = 1350 :=
by sorry

end arithmetic_geometric_means_l318_318707


namespace area_of_shape_l318_318378

def points := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]

theorem area_of_shape : 
  let I := 6 -- Number of interior points
  let B := 5 -- Number of boundary points
  ∃ (A : ℝ), A = I + B / 2 - 1 ∧ A = 7.5 := 
  by
    use 7.5
    simp
    sorry

end area_of_shape_l318_318378


namespace ratio_of_squares_l318_318748

theorem ratio_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a / b = 1 / 3) :
  (4 * a / (4 * b) = 1 / 3) ∧ (a * a / (b * b) = 1 / 9) :=
by
  sorry

end ratio_of_squares_l318_318748


namespace smallest_number_with_conditions_l318_318715

-- Define predicates for the conditions
def ends_in_37 (n : ℕ) : Prop := n % 100 = 37
def digit_sum_37 (n : ℕ) : Prop := (n.toDigits Nat).sum = 37
def divisible_by_37 (n : ℕ) : Prop := n % 37 = 0

-- Define the number 99937
def smallest_number : ℕ := 99937

-- The main proposition
theorem smallest_number_with_conditions :
  ends_in_37 smallest_number ∧ digit_sum_37 smallest_number ∧ divisible_by_37 smallest_number :=
by
  sorry

end smallest_number_with_conditions_l318_318715


namespace min_sum_x_y_exists_min_sum_x_y_l318_318207

variable {p x y : ℝ}

theorem min_sum_x_y (h : p > 1) (h_eq : (x + sqrt(1 + x ^ 2)) * (y + sqrt(1 + y ^ 2)) = p) :
  x + y ≥ (p - 1) / sqrt(p) :=
sorry

theorem exists_min_sum_x_y (h : p > 1) : 
  ∃ x y : ℝ, (x + sqrt(1 + x ^ 2)) * (y + sqrt(1 + y ^ 2)) = p ∧ x + y = (p - 1) / sqrt(p) :=
sorry

end min_sum_x_y_exists_min_sum_x_y_l318_318207


namespace jellybean_avg_increase_l318_318056

noncomputable def avg_increase_jellybeans 
  (avg_original : ℕ) (num_bags_original : ℕ) (num_jellybeans_new_bag : ℕ) : ℕ :=
  let total_original := avg_original * num_bags_original
  let total_new := total_original + num_jellybeans_new_bag
  let num_bags_new := num_bags_original + 1
  let avg_new := total_new / num_bags_new
  avg_new - avg_original

theorem jellybean_avg_increase :
  avg_increase_jellybeans 117 34 362 = 7 := by
  let total_original := 117 * 34
  let total_new := total_original + 362
  let num_bags_new := 34 + 1
  let avg_new := total_new / num_bags_new
  let increase := avg_new - 117
  have h1 : total_original = 3978 := by norm_num
  have h2 : total_new = 4340 := by norm_num
  have h3 : num_bags_new = 35 := by norm_num
  have h4 : avg_new = 124 := by norm_num
  have h5 : increase = 7 := by norm_num
  exact h5

end jellybean_avg_increase_l318_318056


namespace distance_between_towns_l318_318506

variables (x y z : ℝ)

theorem distance_between_towns
  (h1 : x / 24 + y / 16 + z / 12 = 2)
  (h2 : x / 12 + y / 16 + z / 24 = 2.25) :
  x + y + z = 34 :=
sorry

end distance_between_towns_l318_318506


namespace circle_representation_circle_intersects_line_l318_318591

-- Definitions and conditions
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0
def intersection_distance : ℝ := 4 / real.sqrt 5

-- Statements to prove
theorem circle_representation (m : ℝ) : (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
by sorry

theorem circle_intersects_line (m : ℝ) : (∃ x y : ℝ, circle_equation x y m)
  → (∃ (M N : ℝ × ℝ), line_equation M.1 M.2 ∧ line_equation N.1 N.2 ∧ dist M N = intersection_distance) 
  → m = 4 :=
by sorry

end circle_representation_circle_intersects_line_l318_318591


namespace probability_of_white_ball_l318_318973

variable (P_red P_black P_yellow P_white : ℚ)

-- Given Conditions
def condition_1 := P_red = 1/3
def condition_2 := P_black + P_yellow = 5/12
def condition_3 := P_yellow + P_white = 5/12
def condition_4 := P_red + P_black + P_yellow + P_white = 1

-- Theorem to Prove
theorem probability_of_white_ball (hc1 : condition_1) (hc2 : condition_2) (hc3 : condition_3) (hc4 : condition_4) : 
  P_white = 1/4 :=
by
  sorry

end probability_of_white_ball_l318_318973


namespace surface_area_of_cube_l318_318092

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 4 * a
  let face_area := edge_length ^ 2
  let total_surface_area := 6 * face_area
  total_surface_area = 96 * a^2 := by
  sorry

end surface_area_of_cube_l318_318092


namespace cos_difference_squared_l318_318969

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318969


namespace cos_squared_difference_l318_318910

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318910


namespace number_of_divisors_f_2010_l318_318513

def f (n : ℕ) : ℕ := 3 ^ n * 2 ^ n

theorem number_of_divisors_f_2010 : 
  (nat.divisors_count (f 2010)) = 4044121 :=
by
  -- Proof is omitted.
  sorry

end number_of_divisors_f_2010_l318_318513


namespace total_amount_is_16260_l318_318431

variable (A : ℝ)

def distributed_among_30_boys (A : ℝ) : ℝ :=
  A / 30

def distributed_among_40_boys (A : ℝ) : ℝ :=
  A / 40

def condition (A : ℝ) : Prop :=
  distributed_among_30_boys A = distributed_among_40_boys A + 135.50

theorem total_amount_is_16260 (h : condition A) : A = 16260 :=
  sorry

end total_amount_is_16260_l318_318431


namespace sector_area_l318_318395

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = π / 3) (hr : r = 6) : 
  1/2 * r^2 * α = 6 * π :=
by {
  sorry
}

end sector_area_l318_318395


namespace projections_of_Miquel_point_collinear_l318_318571

-- Let l1, l2, l3, l4 be four lines
variable (l1 l2 l3 l4 : Line)

-- Let P be Miquel's Point associated with these lines
variable (P : Point)

-- Let P1, P2, P3, P4 be the projections of P on l1, l2, l3, and l4
variable (P1 : Point) (P2 : Point) (P3 : Point) (P4 : Point)
variable (proj1 : projecting_on P l1 P1)
variable (proj2 : projecting_on P l2 P2)
variable (proj3 : projecting_on P l3 P3)
variable (proj4 : projecting_on P l4 P4)

-- The proof goal: the points P1, P2, P3, and P4 are collinear.
theorem projections_of_Miquel_point_collinear : 
  collinear {P1, P2, P3, P4} :=
sorry

end projections_of_Miquel_point_collinear_l318_318571


namespace highest_possible_value_of_x_l318_318175

theorem highest_possible_value_of_x :
  ∃ x : ℝ, (5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 18 ∧ x = 50/29 :=
by
  sorry

end highest_possible_value_of_x_l318_318175


namespace proof_problem_l318_318903

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318903


namespace triangle_perimeter_is_16_l318_318532

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (6, 6)

-- Calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Function to calculate the perimeter of triangle ABC
def triangle_perimeter (A B C : ℝ × ℝ) : ℝ := 
  distance A B + distance B C + distance C A

-- The theorem statement that the perimeter of the triangle is 16
theorem triangle_perimeter_is_16 : triangle_perimeter A B C = 16 := 
sorry

end triangle_perimeter_is_16_l318_318532


namespace cos_square_difference_l318_318877

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318877


namespace angles_equal_sixty_degrees_l318_318226

/-- Given a triangle ABC with sides a, b, c and respective angles α, β, γ, and with circumradius R,
if the following equation holds:
    (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R),
prove that α = β = γ = 60 degrees. -/
theorem angles_equal_sixty_degrees 
  (a b c R : ℝ) 
  (α β γ : ℝ) 
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R)) :
  α = 60 ∧ β = 60 ∧ γ = 60 := 
sorry

end angles_equal_sixty_degrees_l318_318226


namespace cos_difference_squared_l318_318957

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318957


namespace cos_squared_difference_l318_318949

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318949


namespace sum_of_first_fifty_terms_l318_318681

theorem sum_of_first_fifty_terms (c d : ℕ → ℕ) (ec ed : ℕ)
  (h1 : c 1 = 50)
  (h2 : d 1 = 150)
  (h3 : c 50 + d 50 = 300)
  (hc : ∀ n, c n = 50 + (n - 1) * ec)
  (hd : ∀ n, d n = 150 + (n - 1) * ed) :
  let c_d_sum := λ n, c n + d n
  in ∑ n in Finset.range 50, c_d_sum (n + 1) = 12500 :=
by
  let c_d_sum := λ n, 200 + (n - 1) * (ec + ed)
  have h4 : ∑ n in Finset.range 50, c_d_sum (n + 1) = 50 * 200 + ((∑ n in Finset.range 50, n) - (Finset.range 50).card) * (ec + ed) := _
  have h5 : Finset.card (Finset.range 50) = 50 := Finset.card_range 50
  have h6 : ∑ n in Finset.range 50, n = (50 * (50 - 1)) / 2 := _
  have h7 : (50 * 200) + ((50 * (50 - 1)) / 2 - 50) * (1 * (ec + ed)) = 12500 := _
  exact h7
  sorry

end sum_of_first_fifty_terms_l318_318681


namespace infection_average_l318_318120

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l318_318120


namespace irrational_power_rational_l318_318177

theorem irrational_power_rational :
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ ∃ r : ℚ, (a : ℝ) ^ (b : ℝ) = (r : ℝ) := 
sorry

end irrational_power_rational_l318_318177


namespace cos_difference_squared_l318_318959

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318959


namespace smallest_number_marked_cells_l318_318448

noncomputable def smallest_marked_cells (n : ℕ) : ℕ :=
  ⌈(n^2 : ℝ) / 2⌉.to_nat

theorem smallest_number_marked_cells (n : ℕ) : 
  ∃ N, (∀ (i j : ℕ) (h1 : i ≤ n-2) (h2 : j ≤ n-2),
    ∃ (k l : ℕ), (k = i ∨ k = i+1) ∧ (l = j ∨ l = j+1) ∧ is_marked i n ∧ is_marked j n) ∧
    N = smallest_marked_cells n :=
sorry  -- this proof needs to be constructed

end smallest_number_marked_cells_l318_318448


namespace pentagon_termination_l318_318496

theorem pentagon_termination
  (x : Fin 5 → ℤ)
  (h_sum_pos : (∑ i, x i) > 0)
  (operation : ∀ i : Fin 5, x (i + 1) < 0 → ∀ j, x (j + 2) = x j + x (j + 1) ∧ x (j + 1) = -x (i + 1) ∧ x (i + 1) = x (i + 1) + x (i + 2)): 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∀ i, x (i + n) ≥ 0 := sorry

end pentagon_termination_l318_318496


namespace trigonometric_identity_l318_318865

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318865


namespace episodes_per_wednesday_l318_318364

theorem episodes_per_wednesday :
  ∀ (W : ℕ), (∃ (n_episodes : ℕ) (n_mondays : ℕ) (n_weeks : ℕ), 
    n_episodes = 201 ∧ n_mondays = 67 ∧ n_weeks = 67 
    ∧ n_weeks * W + n_mondays = n_episodes) 
    → W = 2 :=
by
  intro W
  rintro ⟨n_episodes, n_mondays, n_weeks, h1, h2, h3, h4⟩
  -- proof would go here
  sorry

end episodes_per_wednesday_l318_318364


namespace seven_glasses_impossible_l318_318375

variable (n : Nat) (initial : Fin n → Bool)

/-- 
There are 7 glasses, all initially upside-down (True).
In each move, 4 glasses are flipped.
Prove that it is impossible to make all glasses upright (False)
-/
theorem seven_glasses_impossible 
  (h_init : ∀ i : Fin 7, initial i = true)
  (h_move : ∀ s : Finset (Fin 7), s.card = 4 → 
    let new_state := λ i, if i ∈ s then !initial i else initial i in
    ∀ j : Fin 7, initial j = true → new_state j = false) :
  ∀ m : ℕ, ∀ state : Fin 7 → Bool, (∀ i : Fin 7, state i = false) → False :=
by
  intros m state all_up
  sorry

end seven_glasses_impossible_l318_318375


namespace period_axis_of_symmetry_min_value_of_a_l318_318594

-- Definition for Question 1
def f (x : ℝ) : ℝ := (sqrt 3) * sin x * cos x - (cos x) ^ 2 + 1 / 2

-- Statement for Question 1: Period and Axis of Symmetry
theorem period_axis_of_symmetry :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ k : ℤ, f (π / 2 * k + π / 3) = f (π / 2 * k + π / 3)) :=
sorry

-- Definition for Question 2: Triangle conditions
variables {A B C a b c : ℝ}

def law_of_cosines (a b c A : ℝ) : Prop := a^2 = b^2 + c^2 - 2 * b * c * cos A

-- Given conditions
-- A/2 simplifies to A - π/6 equals π/6 or 5π/6, which resolves to A = π/3.
def angle_condition (A : ℝ) : Prop := A = π / 3

-- Minimum value of a with bc = 6 and A = π/3
theorem min_value_of_a (h : b * c = 6) (h2 : angle_condition A):
  (∀ b c : ℝ, 0 < b → 0 < c → law_of_cosines a b c (π / 3)) → a = sqrt 6 :=
sorry

end period_axis_of_symmetry_min_value_of_a_l318_318594


namespace range_f_l318_318069

def f (x : ℝ) : ℝ := 2 / (2 - x)^3

theorem range_f : set.range f = set.univ :=
by sorry

end range_f_l318_318069


namespace real_root_of_determinant_eq_zero_l318_318345

theorem real_root_of_determinant_eq_zero (a b c : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) :
  ∃! x : ℝ, det ![
    ![x, c, -b],
    ![-c, x, a],
    ![b, -a, x]
  ] = 0 :=
sorry

end real_root_of_determinant_eq_zero_l318_318345


namespace triangle_area_is_correct_l318_318421

noncomputable def triangle_area : ℝ :=
  let A := (3, 3)
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℝ)|

theorem triangle_area_is_correct : triangle_area = 9 := by
  sorry

end triangle_area_is_correct_l318_318421


namespace compare_apothems_l318_318392

noncomputable def cot (x : Real) : Real := 1 / tan x

def pentagon_side_length (p : Real) : Prop :=
  (5 / 4) * p^2 * cot (pi / 5) = 5 * p

def pentagon_apothem (a_p p : Real) : Prop :=
  a_p = p * (cot (pi / 5) / 2)

def rectangle_dimensions (w l : Real) : Prop :=
  l = 2 * w ∧ 2 * w^2 = 6 * w

def rectangle_apothem (a_r w : Real) : Prop :=
  a_r = w / 2

theorem compare_apothems (a_p a_r : Real) (p w l : Real) 
  (h1 : pentagon_side_length p) 
  (h2 : pentagon_apothem a_p p) 
  (h3 : rectangle_dimensions w l) 
  (h4 : rectangle_apothem a_r w) : 
  a_p = (40 / 3) * a_r :=
sorry

end compare_apothems_l318_318392


namespace trigonometric_identity_l318_318859

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318859


namespace intersection_point_a_l318_318035

-- Definitions for the given conditions 
def f (x : ℤ) (b : ℤ) : ℤ := 3 * x + b
def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 3 -- Considering that f is invertible for integer b

-- The problem statement
theorem intersection_point_a (a b : ℤ) (h1 : a = f (-3) b) (h2 : a = f_inv (-3)) (h3 : f (-3) b = -3):
  a = -3 := sorry

end intersection_point_a_l318_318035


namespace missing_student_number_in_sample_l318_318059

-- Definitions based on problem conditions
def total_students : ℕ := 52
def sample_size : ℕ := 4
def interval : ℕ := total_students / sample_size
def sample_set : set ℕ := {7, 33, 46, 13}

-- Theorem stating the missing student number in the sample is 13 given the conditions
theorem missing_student_number_in_sample :
  {n | n ∈ sample_set} = {7, 33, 46, 13} :=
sorry

end missing_student_number_in_sample_l318_318059


namespace lateral_area_theorem_l318_318322

noncomputable def lateral_area_of_prism {ABC A1 B1 C1 : Type*}
  [right_triangular_prism ABC A1 B1 C1]
  (hBAC : angle BAC = 90)
  (hBC : length BC = 2)
  (hCC1 : length CC1 = 1)
  (hangle : angle_between_line_and_plane BC1 (A1 ABB1) = 60): ℝ := 
  (2 + 1/2 + (sqrt 15) / 2) * 1

theorem lateral_area_theorem {ABC A1 B1 C1 : Type*}
  [right_triangular_prism ABC A1 B1 C1]
  (hBAC : angle BAC = 90)
  (hBC : length BC = 2)
  (hCC1 : length CC1 = 1)
  (hangle : angle_between_line_and_plane BC1 (A1 ABB1) = 60):
  lateral_area_of_prism hBAC hBC hCC1 hangle = (5 + sqrt 15) / 2 :=
by
  sorry

end lateral_area_theorem_l318_318322


namespace inequality_in_triangle_l318_318631

theorem inequality_in_triangle
(triangle: Type)
(A B C : triangle)
(hA : Prop)
(hB : Prop)
(hC : Prop) :
  (Real.tan (A / 2))^2 + (Real.tan (B / 2))^2 + (Real.tan (C / 2))^2 + 
  8 * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.sin (C / 2)) ≥ 2 :=
by sorry

end inequality_in_triangle_l318_318631


namespace sodium_diameter_scientific_notation_l318_318397

theorem sodium_diameter_scientific_notation :
  ∃ n : ℤ, 0.0000000599 = 5.99 * 10^n :=
by
  use -8
  rw ← mul_assoc
  norm_num
  sorry

end sodium_diameter_scientific_notation_l318_318397


namespace cos_difference_identity_l318_318943

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318943


namespace cos_squared_difference_l318_318886

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318886


namespace cosine_difference_identity_l318_318852

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318852


namespace stickers_distribution_l318_318609

/-- Henry's little brother has 10 identical stickers and 4 sheets of paper, each a different color.
    He wants to put all the stickers on the sheets of paper such that no single sheet has more 
    than 6 stickers. We need to prove the number of ways he can do this is 238. -/
theorem stickers_distribution : 
  (∑ (a b c d : ℕ) in finset.filter (λ x, (x.1 ≤ 6) ∧ (x.2 ≤ 6) ∧ (x.3 ≤ 6) ∧ (x.4 ≤ 6) 
      ∧ (x.1 + x.2 + x.3 + x.4 = 10)) finset.univ, 
      (4.choose (finset.card {a, b, c, d}))) = 238 :=
by sorry

end stickers_distribution_l318_318609


namespace cookie_sheet_perimeter_l318_318975

def width : ℕ := 10
def length : ℕ := 2

def perimeter (w l : ℕ) : ℕ := 2 * w + 2 * l

theorem cookie_sheet_perimeter : 
  perimeter width length = 24 := by
  sorry

end cookie_sheet_perimeter_l318_318975


namespace part_I_solution_part_II_solution_l318_318257

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x - a) + 1

-- Part (I): Solution set of the inequality f(x) > 0 for a = 3
theorem part_I_solution : { x : ℝ | f x 3 > 0 } = { x : ℝ | 2 < x ∧ x < 6 } :=
by sorry

-- Part (II): Range of real number a for area condition
theorem part_II_solution : { a : ℝ | a > 1 ∧ (let area := 
  if 1 ≤ a ∧ (λ x a : ℝ, abs (x - 1) - 2 * abs (x - a) + 1) x a 
  then 2 * (a - 1) ^ 2 + 1 < 6
  else 2 / 3 * a ^ 2 > 6)
  in area
} = { a : ℝ | a > 3 } :=
by sorry

end part_I_solution_part_II_solution_l318_318257


namespace cos_squared_difference_l318_318921

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318921


namespace solve_eq_f_x_plus_3_l318_318261

-- Define the function f with its piecewise definition based on the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 - 3 * x
  else -(x^2 - 3 * (-x))

-- Define the main theorem to find the solution set
theorem solve_eq_f_x_plus_3 (x : ℝ) :
  f x = x + 3 ↔ x = 2 + Real.sqrt 7 ∨ x = -1 ∨ x = -3 :=
by sorry

end solve_eq_f_x_plus_3_l318_318261


namespace loan_difference_l318_318651

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def monthly_compounding : ℝ :=
  future_value 8000 0.10 12 5

noncomputable def semi_annual_compounding : ℝ :=
  future_value 8000 0.10 2 5

noncomputable def interest_difference : ℝ :=
  monthly_compounding - semi_annual_compounding

theorem loan_difference (P : ℝ) (r : ℝ) (n_m n_s t : ℝ) :
    interest_difference = 745.02 := by sorry

end loan_difference_l318_318651


namespace find_C_coordinates_l318_318321

noncomputable def maximize_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (x : ℝ) : Prop :=
  ∀ C : ℝ × ℝ, C = (x, 0) → x = Real.sqrt (a * b)

theorem find_C_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  maximize_angle a b ha hb hab (Real.sqrt (a * b)) :=
by  sorry

end find_C_coordinates_l318_318321


namespace find_a_l318_318216

theorem find_a 
  (a : ℝ)
  (h : ∀ n : ℕ, (n.choose 2) * 2^(5-2) * a^2 = 80 → n = 5) :
  a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l318_318216


namespace cos_difference_identity_l318_318933

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318933


namespace cos_squared_difference_l318_318837

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318837


namespace gross_profit_value_l318_318749

-- Define the variables and constants
variables (C : ℝ) (sales_price : ℝ) (gross_profit_percentage : ℝ)

-- Assign the given values based on the conditions
def sales_price := 54
def gross_profit_percentage := 1.25

-- Calculate the gross profit based on conditions
def gross_profit := gross_profit_percentage * C

-- Define the relationship between cost, gross profit, and sales price
theorem gross_profit_value :
  ∃ (C : ℝ), sales_price = C + gross_profit_percentage * C ∧ gross_profit = 30 :=
by
  sorry

end gross_profit_value_l318_318749


namespace cosine_identity_l318_318578

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 4) : 
  Real.cos (2 * α - Real.pi / 3) = 7 / 8 := by 
  sorry

end cosine_identity_l318_318578


namespace find_b_l318_318318

-- Define the conditions as constants
def x := 36 -- angle a in degrees
def y := 44 -- given
def z := 52 -- given
def w := 48 -- angle b we need to find

-- Define the problem as a theorem
theorem find_b : x + w + y + z = 180 :=
by
  -- Substitute the given values and show the sum
  have h : 36 + 48 + 44 + 52 = 180 := by norm_num
  exact h

end find_b_l318_318318


namespace find_n_l318_318630

theorem find_n (x n : ℝ) (h : x > 0) 
  (h_eq : x / 10 + x / n = 0.14000000000000002 * x) : 
  n = 25 :=
by
  sorry

end find_n_l318_318630


namespace cos_squared_difference_l318_318912

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318912


namespace chord_length_l318_318729

noncomputable def parabola : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 8 * x }
noncomputable def chord_AB : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

theorem chord_length (A B : ℝ × ℝ) (F : ℝ × ℝ) (hF : F = (2, 0))
  (h_parabola : A ∈ parabola)
  (h_parabola : B ∈ parabola)
  (h_chord : ∃ θ, θ = 135 ∧ chord_AB F A = θ ∧ chord_AB F B = θ) :
  chord_AB A B = 16 :=
sorry

end chord_length_l318_318729


namespace proof_problem_l318_318897

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318897


namespace remainder_problem_l318_318799

theorem remainder_problem
  (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 :=
by
  sorry

end remainder_problem_l318_318799


namespace lisa_phone_spending_l318_318685

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end lisa_phone_spending_l318_318685


namespace monotonic_increasing_interval_l318_318736

def f (x : ℝ) : ℝ := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y: ℝ, 0 <= x -> x <= y -> f x <= f y := 
by
  -- proof would be here
  sorry

end monotonic_increasing_interval_l318_318736


namespace find_sum_of_squares_l318_318037

variable {x y z : ℝ}

def matrix_M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * y, z],
    ![x, y, -z],
    ![x, -y, z]]

def matrix_I : Matrix (Fin 3) (Fin 3) ℝ :=
  1 -- This is the identity matrix of appropriate size in Lean

lemma matrix_transpose_mul_eq_I :
  matrix_M.transpose ⬝ matrix_M = matrix_I :=
sorry

theorem find_sum_of_squares
  (h : matrix_transpose_mul_eq_I) :
  x^2 + y^2 + z^2 = 1 :=
sorry

end find_sum_of_squares_l318_318037


namespace find_n_divisors_l318_318040

theorem find_n_divisors (n : ℕ) (h1 : 2287 % n = 2028 % n)
                        (h2 : 2028 % n = 1806 % n) : n = 37 := 
by
  sorry

end find_n_divisors_l318_318040


namespace time_difference_between_Danny_and_Steve_l318_318171

theorem time_difference_between_Danny_and_Steve :
  let T_D := 29 / 2 in
  let T_S := (29 * 2) / 2 in
  T_S - T_D = 14.5 :=
by
  let T_D := 29 / 2
  let T_S := (29 * 2) / 2
  sorry

end time_difference_between_Danny_and_Steve_l318_318171


namespace count_mappings_l318_318661

theorem count_mappings (A : Type) (B : Type) (a b : A) (h : A = {a, b}) (hB : B = {-1, 0, 1}) :
  (∃ f : A → B, f a + f b = 0) → (fintype.card {f : A → B // f a + f b = 0} = 3) :=
begin
  -- Definitions
  have hA : finset A,
  { rw h, exact finset.insert (finset.singleton b) a},
  have hBB : finset B,
  { exact finset.insert (-1) (finset.insert 0 (finset.singleton 1)) },
  -- State the proof
  sorry
end

end count_mappings_l318_318661


namespace sam_correct_percentage_l318_318634

-- Definitions of conditions
def total_questions (y : ℕ) : ℕ := 7 * y
def questions_incorrect (y : ℕ) : ℕ := 2 * y

-- Proof of the claim that Sam answered approximately 71.43% of the questions correctly
theorem sam_correct_percentage (y : ℕ) : 
  (5 / 7) * (100 : ℝ) ≈ (71.43 : ℝ) := 
by
  sorry

end sam_correct_percentage_l318_318634


namespace cos_difference_squared_l318_318965

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318965


namespace general_term_formula_l318_318566

theorem general_term_formula (n : ℕ) :
  let S : ℕ → ℤ := λ n, 2 * n^2 - n + 1 in
  let a : ℕ → ℤ :=
    λ n, match n with
    | 0 => 2
    | n + 1 => 4 * (n + 1) - 3
    end in
  a n = 
  match n with
  | 1 => 2
  | n + 1 => if n = 0 then 2 else 4 * (n + 1) - 3
  end :=
sorry

end general_term_formula_l318_318566


namespace cos_squared_difference_l318_318890

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318890


namespace triangle_perimeter_l318_318535

def point := (ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def A : point := (2, 3)
def B : point := (2, 9)
def C : point := (6, 6)

noncomputable def perimeter (A B C : point) : ℝ :=
  dist A B + dist B C + dist C A

theorem triangle_perimeter : perimeter A B C = 16 :=
  sorry

end triangle_perimeter_l318_318535


namespace correct_three_digit_number_l318_318639

noncomputable def is_valid_path (path : List Nat) : Prop :=
  path.length = 8 ∧
  path = [8, 3, 4, 9, 5, 1, 6, 7]

noncomputable def first_three_digits (path : List Nat) : Nat :=
  100 * path.head! + 10 * path.nth! 1 + path.nth! 2

theorem correct_three_digit_number : ∃ path : List Nat, is_valid_path path ∧ first_three_digits path = 834 := 
by
  use [8, 3, 4, 9, 5, 1, 6, 7]
  split
  . dsimp [is_valid_path]
    simp
  . dsimp [first_three_digits]
    rfl
  sorry

end correct_three_digit_number_l318_318639


namespace minimum_positive_period_sin_cos_l318_318038

def sin_cos_period : ℝ := π

theorem minimum_positive_period_sin_cos :
  ∃ p > 0, ∀ x, sin (x + p) * cos (x + p) = sin x * cos x ∧ p = sin_cos_period := sorry

end minimum_positive_period_sin_cos_l318_318038


namespace sum_c_equals_binom_l318_318340

-- Define the recursive sequence c_n
def c : ℕ → ℕ
| 0       := 1
| (2*n+1) := c n
| (2*n)   := 
  let e := (Nat.find (λ k, 2^k ∣ n) : ℕ) in
  c n + (if n - 2^e < n then c (n - 2^e) else 0)

-- Define the sum u_n
def u (n : ℕ) : ℕ :=
  ∑ i in Finset.range (2^n), c i

-- Theorem statement to prove the desired equality
theorem sum_c_equals_binom (n : ℕ) : 
  u n = (1 / (n+2)) * Nat.choose (2*n+2) (n+1) := 
sorry 

end sum_c_equals_binom_l318_318340


namespace positive_value_of_m_l318_318209

variable {m : ℝ}

theorem positive_value_of_m (h : ∃ x : ℝ, (3 * x^2 + m * x + 36) = 0 ∧ (∀ y : ℝ, (3 * y^2 + m * y + 36) = 0 → y = x)) :
  m = 12 * Real.sqrt 3 :=
sorry

end positive_value_of_m_l318_318209


namespace volume_over_surface_area_of_sphere_equals_one_l318_318468

def regular_hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
def distance_from_center_to_plane : ℝ := 2 * Real.sqrt 2

noncomputable def side_length_of_hexagon (A : ℝ) : ℝ :=
  Real.sqrt ((2 * A) / (3 * Real.sqrt 3))

noncomputable def radius_of_sphere_from_hexagon (a d : ℝ) : ℝ :=
  Real.sqrt (d^2 + a ^ 2 / 3)

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

theorem volume_over_surface_area_of_sphere_equals_one (V S : ℝ) (a O r : ℝ)
    (hA : regular_hexagon_area = (3 * Real.sqrt 3) / 2)
    (hD : distance_from_center_to_plane = 2 * Real.sqrt 2)
    (ha : a = side_length_of_hexagon regular_hexagon_area)
    (hr : r = radius_of_sphere_from_hexagon a distance_from_center_to_plane)
    (hV : V = volume_of_sphere r)
    (hS : S = surface_area_of_sphere r) :
    V / S = 1 :=
begin
  sorry
end

end volume_over_surface_area_of_sphere_equals_one_l318_318468


namespace cos_difference_identity_l318_318931

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318931


namespace smallest_k_condition_exists_l318_318428

theorem smallest_k_condition_exists (k : ℕ) :
    k > 1 ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 3 = 1) → k = 313 :=
by
  sorry

end smallest_k_condition_exists_l318_318428


namespace ratio_w_to_y_l318_318044

variables {w x y z : ℝ}

theorem ratio_w_to_y
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 9) :
  w / y = 8 :=
by
  sorry

end ratio_w_to_y_l318_318044


namespace elroy_miles_difference_l318_318181

theorem elroy_miles_difference (earning_rate_last_year earning_rate_this_year total_collection_last_year : ℝ)
  (rate_last_year : earning_rate_last_year = 4)
  (rate_this_year : earning_rate_this_year = 2.75)
  (total_collected : total_collection_last_year = 44) :
  (total_collection_last_year / earning_rate_this_year) - (total_collection_last_year / earning_rate_last_year) = 5 :=
by
  rw [rate_last_year, rate_this_year, total_collected]
  norm_num
  sorry

end elroy_miles_difference_l318_318181


namespace train_pass_bridge_l318_318136

theorem train_pass_bridge :
  ∀ (train_length bridge_length : ℕ) (train_speed_kmph : ℕ),
    train_length = 360 →
    bridge_length = 140 →
    train_speed_kmph = 45 →
    let total_distance := train_length + bridge_length in
    let train_speed_mps := (train_speed_kmph * 1000) / 3600 in
    let time := total_distance / train_speed_mps in
    time = 40 := 
by
  intros train_length bridge_length train_speed_kmph h1 h2 h3
  dsimp only [total_distance, train_speed_mps, time]
  rw [h1, h2, h3]
  norm_num
  sorry

end train_pass_bridge_l318_318136


namespace value_of_k_l318_318012

theorem value_of_k (k t : ℝ) 
  (line_eq : ∀ (x y : ℝ), sqrt k * x + 4 * y = 10)
  (triangle_area : t = 20) :
  k = 25 / 64 :=
by sorry

end value_of_k_l318_318012


namespace cos_squared_difference_l318_318950

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318950


namespace problem_statement_l318_318678

variable (x y : ℝ)
def t : ℝ := x / y

theorem problem_statement (h : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 3) : t = 1 := by 
  sorry

end problem_statement_l318_318678


namespace find_n_l318_318797

noncomputable def f (n : ℝ) : ℝ :=
  n ^ (n / 2)

example : f 2 = 2 := sorry

theorem find_n : ∃ n : ℝ, f n = 12 ∧ abs (n - 3.4641) < 0.0001 := sorry

end find_n_l318_318797


namespace number_of_distinct_keys_l318_318405

theorem number_of_distinct_keys (n : ℕ) (h : n % 2 = 0) : 
  ∃ k : ℕ, k = 4 ^ (n * n / 4) :=
by
  use 4 ^ (n * n / 4)
  sorry

end number_of_distinct_keys_l318_318405


namespace sum_of_digits_T_l318_318052

def horse_running_times : List Int :=
  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

def lcm_of_six_times : Int :=
  List.lcm (horse_running_times.take 6)

def sum_of_digits (n : Int) : Int :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_T : sum_of_digits lcm_of_six_times = 18 := by
  sorry

end sum_of_digits_T_l318_318052


namespace xy_value_l318_318285

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l318_318285


namespace range_of_k_l318_318303

theorem range_of_k (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 2 → x^2 - y^2 = 4 →
  x*y <= 0 ∧ x ≠ 0 ∧ ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0) ↔ (1 < k ∧ k < real.sqrt 2) := 
sorry

end range_of_k_l318_318303


namespace number_of_sides_l318_318490

theorem number_of_sides (n : ℕ) : 
  let a_1 := 6 
  let d := 5
  let a_n := a_1 + (n - 1) * d
  a_n = 5 * n + 1 := 
by
  sorry

end number_of_sides_l318_318490


namespace expression_zero_denominator_nonzero_l318_318071

theorem expression_zero (x : ℝ) : 
  (2 * x - 6) = 0 ↔ x = 3 :=
by {
  sorry
  }

theorem denominator_nonzero (x : ℝ) : 
  x = 3 → (5 * x + 10) ≠ 0 :=
by {
  sorry
  }

end expression_zero_denominator_nonzero_l318_318071


namespace conic_is_pair_of_lines_l318_318518

-- Define the specific conic section equation
def conic_eq (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

-- State the theorem
theorem conic_is_pair_of_lines : ∀ x y : ℝ, conic_eq x y ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  -- Sorry is placed to denote that proof steps are omitted in this statement
  sorry

end conic_is_pair_of_lines_l318_318518


namespace range_of_m_l318_318660

noncomputable def A (m : ℝ) : Set ℝ :=
  { y | ∃ x : ℝ, y = sin x - cos (x + Real.pi / 6) + m }

noncomputable def B : Set ℝ :=
  { y | ∃ x : ℝ, x ∈ Icc 1 2 ∧ y = -x^2 + 2 * x }

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ A m → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A m) →
  1 - Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end range_of_m_l318_318660


namespace function_properties_l318_318529

theorem function_properties (f : ℤ → ℤ) (a b : ℤ) : 
  (∀ n : ℤ, f(f(n)) = n ∧ f(f(n + 2) + 2) = n) ↔ 
  (∀ k : ℤ, f(2 * k) = a - 2 * k ∧ f(2 * k + 1) = b - 2 * k ∧ (a % 2 = 0 ∧ b % 2 = 1 ∨ a = b + 1)) :=
sorry

end function_properties_l318_318529


namespace cos_squared_difference_l318_318801

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318801


namespace find_multiple_of_son_age_l318_318744

variable (F S k : ℕ)

theorem find_multiple_of_son_age
  (h1 : F = k * S + 4)
  (h2 : F + 4 = 2 * (S + 4) + 20)
  (h3 : F = 44) :
  k = 4 :=
by
  sorry

end find_multiple_of_son_age_l318_318744


namespace rhombus_perimeter_l318_318027

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l318_318027


namespace symmetry_center_proof_l318_318047

def symmetry_center_tangent_function (x : ℝ) : Prop :=
  y = (1/3) * tan (-7 * x + π / 3)

theorem symmetry_center_proof :
  symmetry_center_tangent_function (π / 21) = true :=
sorry

end symmetry_center_proof_l318_318047


namespace max_matching_pairs_l318_318689

theorem max_matching_pairs (total_pairs : ℕ) (lost_individual : ℕ) (left_pair : ℕ) : 
  total_pairs = 25 ∧ lost_individual = 9 → left_pair = 20 :=
by
  sorry

end max_matching_pairs_l318_318689


namespace shirts_made_today_l318_318486

def shirts_per_minute : ℕ := 6
def minutes_yesterday : ℕ := 12
def total_shirts : ℕ := 156
def shirts_yesterday : ℕ := shirts_per_minute * minutes_yesterday
def shirts_today : ℕ := total_shirts - shirts_yesterday

theorem shirts_made_today :
  shirts_today = 84 :=
by
  sorry

end shirts_made_today_l318_318486


namespace max_distance_S_l318_318743

open Complex

theorem max_distance_S (z : ℂ) 
  (hz1 : abs z = 1) 
  (hz2 : abs ((1 + I) * z) = abs (1 + I) * abs z)
  (hz3 : abs (2 * conj z) = 2 * abs z)
  (h_not_collinear : ¬ collinear ℂ ({z, (1 + I) * z, 2 * conj z} : set ℂ)) :
  let S := (1 + I) * z + 2 * conj z - z in 
  abs S ≤ 3 := sorry

end max_distance_S_l318_318743


namespace find_x_l318_318450

theorem find_x (x : ℝ) (h : 49 / x = 700) : x = 0.07 :=
sorry

end find_x_l318_318450


namespace cos_squared_difference_l318_318807

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318807


namespace group_division_l318_318406

-- defining the student and height parameters
def num_students : ℕ := 40
def max_height : ℕ := 175
def min_height : ℕ := 155
def interval : ℕ := 3

-- the number of groups is 7
theorem group_division :
  let range := max_height - min_height in
  let num_groups := (range + interval - 1) / interval in
  num_groups = 7 :=
by
  sorry

end group_division_l318_318406


namespace compare_sqrts_l318_318497

theorem compare_sqrts {a_1 a_2 ... a_n : ℝ} 
  (h₀ : ∀ i, 0 ≤ a_i) 
  (h₁ : ∃ i j, i ≠ j ∧ a_i ≠ 0 ∧ a_j ≠ 0) :
  (∑ i in finset.range n, a_i ^ 2002) ^ (1 / 2002) > (∑ i in finset.range n, a_i ^ 2003) ^ (1 / 2003) :=
sorry

end compare_sqrts_l318_318497


namespace find_length_RS_l318_318150

-- Define the known lengths and equal angles
def FD : ℝ := 4
def DR : ℝ := 6
def FR : ℝ := 5
def FS : ℝ := 7.5
def angle_RFS_eq_angle_FDR : Prop := true -- This will indicate the angle equality

-- Define the proof goal using these definitions:
theorem find_length_RS (h : angle_RFS_eq_angle_FDR) : RS = 6.25 := by
sorrry -- Placeholder for the proof

end find_length_RS_l318_318150


namespace geometric_sequence_a5_eq_2_l318_318559

-- Define geometric sequence and the properties
noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Roots of given quadratic equation
variables (h1 : a 3 = 1 ∨ a 3 = 4 / 1) (h2 : a 7 = 4 / a 3)
variables (h3 : q > 0) (h4 : geometric_seq a q)

-- Prove that a5 = 2
theorem geometric_sequence_a5_eq_2 : a 5 = 2 :=
sorry

end geometric_sequence_a5_eq_2_l318_318559


namespace range_of_f_l318_318247

open Interval

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem range_of_f :
  ∀ (x : ℝ), x ∈ Icc 1 5 → 4 ≤ f x ∧ f x ≤ 29/5 :=
by 
  sorry

end range_of_f_l318_318247


namespace smallest_n_satisfying_condition_l318_318085

noncomputable def min_even_n_divisible (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n else n+1

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 2 ∣ m) ∧
            (∃ k : ℕ, (2 * 4 * 6 * ... * k) % 3^3 = 0 ∧ (2 * 4 * 6 * ... * k) % 7^2 = 0) ∧
            n = 42 :=
begin
  sorry
end

end smallest_n_satisfying_condition_l318_318085


namespace harvey_runs_more_than_sam_l318_318274

theorem harvey_runs_more_than_sam :
  ∀ (miles_sam miles_total : ℕ), 
    miles_sam = 12 → 
    miles_total = 32 → 
    ∃ (x : ℕ), 
      miles_sam + (miles_sam + x) = miles_total ∧ 
      x = 8 := 
by 
  intros miles_sam miles_total h_sam h_total
  use 8
  split
  { rw [h_sam, h_total]
    simp
  }
  { refl }

end harvey_runs_more_than_sam_l318_318274


namespace solve_equation1_solve_equation2_l318_318388

-- Define the equations and the problem.
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = ((3 - x) / (x - 2))

-- Proof problem for the first equation: Prove that x = -4 is the solution.
theorem solve_equation1 : ∀ x : ℝ, equation1 x → x = -4 :=
by {
  sorry
}

-- Proof problem for the second equation: Prove that there are no solutions.
theorem solve_equation2 : ∀ x : ℝ, ¬equation2 x :=
by {
  sorry
}

end solve_equation1_solve_equation2_l318_318388


namespace cos_diff_square_identity_l318_318817

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318817


namespace union_of_sets_l318_318602

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l318_318602


namespace range_of_b_no_common_points_l318_318404

theorem range_of_b_no_common_points (b : ℝ) :
  ¬ (∃ x : ℝ, 2 ^ |x| - 1 = b) ↔ b < 0 :=
by
  sorry

end range_of_b_no_common_points_l318_318404


namespace cost_of_larger_container_l318_318477

noncomputable def volume_of_cylinder (diameter height : ℝ) : ℝ :=
  π * ((diameter / 2)^2) * height

noncomputable def price_per_cubic_inch (price volume : ℝ) : ℝ :=
  price / volume

noncomputable def discounted_price_per_cubic_inch (price_per_unit : ℝ) : ℝ :=
  price_per_unit * 0.9

noncomputable def cost_of_container (volume price_per_unit : ℝ) : ℝ :=
  volume * price_per_unit

theorem cost_of_larger_container
  (small_diameter small_height : ℝ) (small_price : ℝ)
  (large_diameter large_height : ℝ)
  (discount : ℝ = 0.9) :
  small_diameter = 5 ∧ small_height = 8 ∧ small_price = 1.5 ∧
  large_diameter = 10 ∧ large_height = 10 →
  cost_of_container 
    (volume_of_cylinder large_diameter large_height)
    (discounted_price_per_cubic_inch (price_per_cubic_inch small_price (volume_of_cylinder small_diameter small_height))) = 6.75 :=
  by
    intro h -- introduces all conditions
    cases h with hd hd1
    cases hd1 with hh hp
    cases hp with ld lh
    rw [hd, hh, hp, ld, lh]
    sorry -- proof omitted for now

end cost_of_larger_container_l318_318477


namespace perpendicular_implies_value_of_m_l318_318598

-- Define the lines l1 and l2
def l₁ (m : ℝ) : ℝ → ℝ → Prop := λ x y, (m + 3) * x + y - 1 = 0
def l₂ (m : ℝ) : ℝ → ℝ → Prop := λ x y, 4 * x + m * y + 3 * m - 4 = 0

-- Define perpendicular condition
def perpendicular (m : ℝ) : Prop :=
  let A₁ := m + 3
  let B₁ := 1
  let A₂ := 4
  let B₂ := m
  in A₁ * A₂ + B₁ * B₂ = 0

-- Prove that if l1 ⊥ l2, then m = -12/5
theorem perpendicular_implies_value_of_m (m : ℝ) : 
  perpendicular m → m = -12 / 5 :=
by
  sorry

end perpendicular_implies_value_of_m_l318_318598


namespace trigonometric_identity_l318_318862

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318862


namespace anthony_path_shortest_l318_318489

noncomputable def shortest_distance (A B C D M : ℝ) : ℝ :=
  4 + 2 * Real.sqrt 3

theorem anthony_path_shortest {A B C D : ℝ} (M : ℝ) (side_length : ℝ) (h : side_length = 4) : 
  shortest_distance A B C D M = 4 + 2 * Real.sqrt 3 :=
by 
  sorry

end anthony_path_shortest_l318_318489


namespace triangle_perimeter_l318_318536

def point := (ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def A : point := (2, 3)
def B : point := (2, 9)
def C : point := (6, 6)

noncomputable def perimeter (A B C : point) : ℝ :=
  dist A B + dist B C + dist C A

theorem triangle_perimeter : perimeter A B C = 16 :=
  sorry

end triangle_perimeter_l318_318536


namespace points_on_circle_at_distance_three_l318_318163

theorem points_on_circle_at_distance_three (x y : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 4 * y - 31 = 0) →
  abs(3 * x + 4 * y + 4) / sqrt(3^2 + 4^2) = 3) →
  (∃ P : ℝ × ℝ, (P ∈ { (x, y) : ℝ × ℝ | x^2 + y^2 - 2 * x - 4 * y - 31 = 0 ∧ abs(3 * x + 4 * y + 4) / sqrt(3^2 + 4^2) = 3 })) :=
begin 
  sorry
end

end points_on_circle_at_distance_three_l318_318163


namespace number_of_common_tangents_l318_318605

variable {r R d : ℝ}

theorem number_of_common_tangents (h_rR : r < R) :
  if d < R - r then 0 else
  if d = R - r then 1 else
  if R - r < d ∧ d < R + r then 2 else
  if d = R + r then 3 else
  4 := by
  sorry

end number_of_common_tangents_l318_318605


namespace domain_width_of_g_l318_318611

theorem domain_width_of_g (h : ℝ → ℝ) (domain_h : ∀ x, -8 ≤ x ∧ x ≤ 8 → h x = h x) :
  let g (x : ℝ) := h (x / 2)
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → ∃ y, g x = y) ∧ (b - a = 32) := 
sorry

end domain_width_of_g_l318_318611


namespace tan_alpha_value_l318_318576

theorem tan_alpha_value
  {α : ℝ}
  (h1 : sin (π - α) = -2/3)
  (h2 : α ∈ set.Ioo (-π/2) 0) :
  tan α = -2 * real.sqrt 5 / 5 :=
sorry

end tan_alpha_value_l318_318576


namespace find_f_determine_g0_l318_318269

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (exp x - exp (-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (exp x + exp (-x))

theorem find_f (f g : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, g (-x) = g x)
  (h3 : ∀ x, f x - g x = exp x) : f = (λ x, (1 / 2) * (exp x - exp (-x))) :=
by
  sorry

theorem determine_g0 (f g : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, g (-x) = g x)
  (h3 : ∀ x, f x - g x = exp x) : g 0 = 1 :=
by
  sorry

end find_f_determine_g0_l318_318269


namespace cos_square_difference_l318_318870

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318870


namespace arithmetic_sequence_general_formula_T_n_expression_and_monotonicity_T_n_greater_than_2_range_l318_318225

-- Definitions from the problem statement
def a (n : ℕ) : ℕ := n + 1
def S (n : ℕ) : ℕ := n * (n + 1) / 2  -- Sum of first n terms of the sequence
  
-- Sequence b_n based on the given sequence {a_n}
def b (n : ℕ) : ℝ := (n + 1) * 2^(-n)
def T (n : ℕ) : ℝ := ∑ i in range (n+1), b i  -- Sum of the first n terms of sequence {b_n}

-- Main theorem statements
theorem arithmetic_sequence_general_formula :
  ∀ n : ℕ, a n = n + 1 :=
by sorry

theorem T_n_expression_and_monotonicity :
  ∀ n : ℕ, T n = 3 - (n + 3) / 2^n ∧ monotonic_increasing (T n) :=
by sorry

theorem T_n_greater_than_2_range :
  ∀ n : ℕ, T n > 2 ↔ n > 3 :=
by sorry

end arithmetic_sequence_general_formula_T_n_expression_and_monotonicity_T_n_greater_than_2_range_l318_318225


namespace concentration_replacement_l318_318984

theorem concentration_replacement 
  (initial_concentration : ℝ)
  (new_concentration : ℝ)
  (fraction_replaced : ℝ)
  (replacing_concentration : ℝ)
  (h1 : initial_concentration = 0.45)
  (h2 : new_concentration = 0.35)
  (h3 : fraction_replaced = 0.5) :
  replacing_concentration = 0.25 := by
  sorry

end concentration_replacement_l318_318984


namespace find_BT_BM_ratio_l318_318636

-- Define the problem context
variable {A B C M P Q T : Type} [Geometry.AffineSpace A B C M P Q T]

-- Given conditions
variable (h_triangle_acute : acute_triangle A B C)
variable (h_midpoint : midpoint M A C)
variable (h_circle_passing : circle_passing_through B M)
variable (h_intersections : intersects_at_second_point P Q AB BC)
variable (h_parallelogram : parallelogram BP QT)

-- Prove the ratio
theorem find_BT_BM_ratio :
  on_circumcircle T A B C →
  BT = BM * sqrt 2 := 
sorry

end find_BT_BM_ratio_l318_318636


namespace find_k_l318_318669

noncomputable theory

def quadratic_function (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

theorem find_k :
  ∃ (k : ℤ),
    (∀ (f : ℤ → ℤ)
    (a b c : ℤ),
      (f = quadratic_function a b c) →
      (f 2 = 0) →
      (30 < f 5) → (f 5 < 40) →
      (50 < f 6) → (f 6 < 60) →
      (1000 * k < f 50) → (f 50 < 1000 * (k + 1))) 
    → k = 7 :=
begin
  sorry
end

end find_k_l318_318669


namespace area_of_sector_l318_318768

theorem area_of_sector (r1 r2 : ℝ) (h_r1 : r1 = 12) (h_r2 : r2 = 8) : 
  let area_ring := π * r1^2 - π * r2^2 in
  let area_sector := (1 / 6) * area_ring in
  area_sector = (40 * π) / 3 :=
by 
  sorry

end area_of_sector_l318_318768


namespace cos_squared_difference_l318_318946

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318946


namespace num_even_four_digit_numbers_num_greater_than_3125_numbers_l318_318778

-- Define the digits and conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def four_digit_numbers_without_repetition := {n : ℕ | 
  (1000 <= n) ∧ (n < 10000) ∧ (∀ d : ℕ, d ∈ digits → (multiplicity d n).count ≤ 1)}

-- Problem 1: Prove the number of even numbers
def even_four_digit_numbers := {n ∈ four_digit_numbers_without_repetition | n % 2 = 0}

theorem num_even_four_digit_numbers : even_four_digit_numbers.card = 636 := by
  sorry

-- Problem 2: Prove the number of numbers greater than 3125
def greater_than_3125_numbers := {n ∈ four_digit_numbers_without_repetition | n > 3125}

theorem num_greater_than_3125_numbers : greater_than_3125_numbers.card = 162 := by
  sorry

end num_even_four_digit_numbers_num_greater_than_3125_numbers_l318_318778


namespace max_triangles_convex_polygon_l318_318323

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end max_triangles_convex_polygon_l318_318323


namespace proof_problem_l318_318896

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318896


namespace cos_square_difference_l318_318876

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318876


namespace count_lines_through_point_l318_318315

open Set

def is_prime (n : ℕ) : Prop := n > 1 ∧ ¬ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem count_lines_through_point (p : ℕ) (q : ℕ) :
  p = 5 → q = 2 → 
  Nat.card { (a, b) | is_prime a ∧ is_prime b ∧ 
                     a > 0 ∧ b > 0 ∧ 
                     (5*b + 2*a = a*b) } = 2 := 
by {
  intros h₁ h₂,
  have : { (a, b) | is_prime a ∧ is_prime b ∧ 
                   a > 0 ∧ b > 0 ∧ 
                   (5 * b + 2 * a = a * b) } = 
         { (7, 3), (3, 7) }, from sorry,
  rw this,
  simp,
}

end count_lines_through_point_l318_318315


namespace parabola_vertex_l318_318051

theorem parabola_vertex : 
  (∃ (x y : ℝ), y = 2 * (x - 3) ^ 2 + 1 ∧ (x, y) = (3, 1)) :=
begin
  sorry
end

end parabola_vertex_l318_318051


namespace point_trajectory_is_ellipse_l318_318584

noncomputable def is_ellipse (p : ℝ × ℝ) : Prop :=
  10 * real.sqrt (p.1^2 + p.2^2) = abs (3 * p.1 + 4 * p.2 - 12)

theorem point_trajectory_is_ellipse (M : ℝ × ℝ) (h : is_ellipse M) : 
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧ 
  (∀ (x y : ℝ), (10 * real.sqrt (x^2 + y^2) = abs (3 * x + 4 * y - 12)) ↔ 
  ((x^2 / a^2) + (y^2 / b^2) = 1)) :=
sorry

end point_trajectory_is_ellipse_l318_318584


namespace cos_squared_difference_l318_318882

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318882


namespace point_on_x_axis_coordinates_l318_318616

theorem point_on_x_axis_coordinates 
  (m : ℝ) 
  (P_x P_y : ℝ)
  (hP : P_x = m ∧ P_y = m - 3) 
  (h : P_y = 0) : P_x = 3 ∧ P_y = 0 :=
by
  have h1 : m - 3 = 0 := by rwa [←h] at hP.right
  have h2 : m = 3 := by linarith
  rw [h2] at hP
  exact ⟨hP.left, h⟩

end point_on_x_axis_coordinates_l318_318616


namespace brenda_blisters_l318_318156

theorem brenda_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) (arms : ℕ) :
  blisters_per_arm = 60 → blisters_rest = 80 → arms = 2 → 
  blisters_per_arm * arms + blisters_rest = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end brenda_blisters_l318_318156


namespace main_l318_318385

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

-- Proposition ②
def prop_2 : Prop := ∀ x, f (-Real.pi / 8) = f (x + Real.pi / 4 - x - Real.pi / 4)

-- Proposition ④
def prop_4 : Prop := ∃ α ∈ Ioo 0 Real.pi, ∀ x, f (x + α) = f (x + 3 * α)

theorem main : prop_2 ∧ prop_4 := by sorry

end main_l318_318385


namespace solve_m_ellipse_line_l318_318585

noncomputable def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 16 + y^2 / m^2 = 1

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * x

def right_focus_x (m : ℝ) : ℝ := Real.sqrt (16 - m^2)

theorem solve_m_ellipse_line :
  ∀ (m : ℝ), m > 0 →
    ∃ (x y : ℝ), ellipse_equation x y m ∧ line_equation x y ∧ x = right_focus_x m →
    m = 2 * Real.sqrt 2 :=
by
  intros m hm
  have h := sorry
  exact h

end solve_m_ellipse_line_l318_318585


namespace proof_problem_l318_318894

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318894


namespace cos_diff_square_identity_l318_318814

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318814


namespace range_of_k_l318_318242

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x^2 - 2 * x + k^2 - 3 > 0) -> (k > 2 ∨ k < -2) :=
by
  sorry

end range_of_k_l318_318242


namespace smallest_positive_period_axis_of_symmetry_max_value_and_corresponding_x_l318_318553

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + π / 4) - 1

theorem smallest_positive_period : ∃ (T > 0), ∀ x, f (x + T) = f x := 
    sorry

theorem axis_of_symmetry : ∃ (k : ℤ), ∀ x, f x = f (1 / 2 * k * π + π / 8 + x) := 
    sorry

theorem max_value_and_corresponding_x : ∃ (k : ℤ), ∀ x, f (k * π + π / 8) = 2 := 
    sorry

end smallest_positive_period_axis_of_symmetry_max_value_and_corresponding_x_l318_318553


namespace quadratic_inequality_solution_empty_l318_318411

theorem quadratic_inequality_solution_empty (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - m * x + m - 1 < 0) → false) →
  (m ≥ (2 * Real.sqrt 3) / 3 ∨ m ≤ -(2 * Real.sqrt 3) / 3) :=
by
  sorry

end quadratic_inequality_solution_empty_l318_318411


namespace third_side_of_triangle_with_sides_2_and_6_l318_318730

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem third_side_of_triangle_with_sides_2_and_6 : ∃ x, (x^2 - 10 * x + 21 = 0) ∧ is_valid_triangle 2 6 x :=
begin
  use 7,
  split,
  { -- Prove 7 is a solution to x^2 - 10x + 21 = 0
    calc
      7^2 - 10 * 7 + 21 = 49 - 70 + 21 : by simp
                    ... = 0 : by simp,
  },
  { -- Prove (2, 6, 7) forms a valid triangle
    unfold is_valid_triangle,
    split,
    { calc
        2 + 6 = 8 : by simp
          ... > 7 : by norm_num,
    },
    split,
    { compute,
    },
    { compute,
    }
  }
  { unfold is_valid_triangle,
    split,
    { calc
        2 + 6 = 8 : by simp
          ... > 7 : by norm_num,
    },
    split,
    { calc
        2 + 7 = 9 : by simp
          ... > 6 : by simp,
    },
    { calc
        6 + 7 = 13 : by simp
          ... > 2 : by simp,
    }
  }
end

end third_side_of_triangle_with_sides_2_and_6_l318_318730


namespace repeating_decimal_denominator_l318_318019

theorem repeating_decimal_denominator (S : ℚ) (h : S = 36 / 99) : S.denom = 11 := by
  have h1 : S = 4/11 := by sorry -- This step is skipping proof for simplification
  rw h1
  exact Rat.num_denom_eq S

end repeating_decimal_denominator_l318_318019


namespace minimum_value_of_fraction_plus_variable_l318_318781

theorem minimum_value_of_fraction_plus_variable (a : ℝ) (h : a > 1) : ∃ m, (∀ b, b > 1 → (4 / (b - 1) + b) ≥ m) ∧ m = 5 :=
by
  use 5
  sorry

end minimum_value_of_fraction_plus_variable_l318_318781


namespace sports_day_condition_l318_318470

open Set

variable (U : Type) 
variable (A B C : Set U)

theorem sports_day_condition (h : ∀ x ∈ U, x ∈ A ∪ B ∪ C → x ∈ A ∧ x ∈ B → x ∉ C) : 
  (A ∩ B) ∩ C = ∅ :=
by
  sorry

end sports_day_condition_l318_318470


namespace crayons_produced_l318_318987

theorem crayons_produced (colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) 
  (h_colors : colors = 4) (h_crayons_per_color : crayons_per_color = 2) 
  (h_boxes_per_hour : boxes_per_hour = 5) (h_hours : hours = 4) : 
  colors * crayons_per_color * boxes_per_hour * hours = 160 := 
by
  rw [h_colors, h_crayons_per_color, h_boxes_per_hour, h_hours]
  norm_num

end crayons_produced_l318_318987


namespace seedlings_by_father_correct_l318_318708

def seedlings_planted_by_remi_father (first_day second_day third_day fourth_day : ℕ) 
  (total : ℕ) : ℕ :=
total - (first_day + second_day)

theorem seedlings_by_father_correct :
  seedlings_planted_by_remi_father 200 400 1200 1600 5000 = 4400 :=
by
  -- define each variable with given conditions
  let first_day := 200
  let second_day := 400
  let third_day := 1200
  let fourth_day := 1600
  let total := 5000
  
  -- apply the function to calculate the seedlings planted by Remi's father
  rw [seedlings_planted_by_remi_father, first_day, second_day, third_day, fourth_day, total]
  -- remind that this is just statement without proof, thus we add sorry
  sorry

end seedlings_by_father_correct_l318_318708


namespace problem_solution_l318_318346

noncomputable def a : ℕ → ℝ 
| 0       := 7/25
| (n + 1) := 3 * (a n)^2 - 2

def bounded_product (c : ℝ) : Prop :=
  ∀ n : ℕ, |List.prod (List.map a (List.range n))| ≤ c / 2^n

theorem problem_solution : 
  let c := Inf {c : ℝ | bounded_product c}
  (100 * c).round = 107 :=
sorry

end problem_solution_l318_318346


namespace bag_of_chips_weight_l318_318103

theorem bag_of_chips_weight (c : ℕ) : 
  (∀ (t : ℕ), t = 9) → 
  (∀ (b : ℕ), b = 6) → 
  (∀ (x : ℕ), x = 4 * 6) → 
  (21 * 16 = 336) →
  (336 - 24 * 9 = 6 * c) → 
  c = 20 :=
by
  intros ht hb hx h_weight_total h_weight_chips
  sorry

end bag_of_chips_weight_l318_318103


namespace rhombus_perimeter_l318_318028

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l318_318028


namespace work_days_together_l318_318091

theorem work_days_together (days_p: ℕ) (days_q : ℕ) (fraction_left : ℚ) 
  (hp : days_p = 15) (hq : days_q = 20) (hf : fraction_left = 8 / 15) :
  ∃ d : ℕ, 1 - (fraction_left) = d * ((1 / days_p) + (1 / days_q)) :=
begin
  use 4,
  rw [hp, hq, hf],
  norm_num,
end

end work_days_together_l318_318091


namespace pizza_cost_difference_l318_318113

theorem pizza_cost_difference :
  let p := 12 -- Cost of plain pizza
  let m := 3 -- Cost of mushrooms
  let o := 4 -- Cost of olives
  let s := 12 -- Total number of slices
  (m + o + p) / s * 10 - (m + o + p) / s * 2 = 12.67 :=
by
  sorry

end pizza_cost_difference_l318_318113


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l318_318350

variables (l m n : Type) [line l] [line m] [line n] (α : Type) [plane α]

def perpendicular_to_plane (l : Type) (α : Type) [line l] [plane α] : Prop := sorry
def perpendicular_to_line (l m : Type) [line l] [line m] : Prop := sorry

theorem sufficient_but_not_necessary_condition :
  (perpendicular_to_plane l α) → ((perpendicular_to_line l m) ∧ (perpendicular_to_line l n)) :=
begin
  sorry -- proof goes here
end

theorem not_necessary_condition :
  ((perpendicular_to_line l m) ∧ (perpendicular_to_line l n)) → (perpendicular_to_plane l α) :=
begin
  sorry -- proof goes here
end

end sufficient_but_not_necessary_condition_not_necessary_condition_l318_318350


namespace intersection_empty_l318_318663

def M : set (ℝ × ℝ) := { p | ∃ x, p = (x, 2 * x + 1) }
def N : set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem intersection_empty : M ∩ N = ∅ :=
by sorry

end intersection_empty_l318_318663


namespace problem_statement_l318_318555

noncomputable def f : ℕ → (ℝ → ℝ)
| 1 := λ x, Real.cos x
| (n+1) := λ x, (f n x).derivative

theorem problem_statement : f 2016 = λ x, Real.sin x :=
by sorry

end problem_statement_l318_318555


namespace choose_officers_count_l318_318377

theorem choose_officers_count :
  let members := 24 
  let boys := 12 
  let girls := 12 
  (∃ (president boys → ∃ vice_president boys → ∃ secretary ∉ {president, vice_president}, 
     12 * 11 * 22)) + 
  (∃ (president girls → ∃ vice_president girls → ∃ secretary ∉ {president, vice_president}, 
     12 * 11 * 22)) = 5808 := 
by 
sorry

end choose_officers_count_l318_318377


namespace proof_problem_l318_318899

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318899


namespace charging_time_is_correct_l318_318127

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l318_318127


namespace triangle_LCM_is_isosceles_l318_318239

variable (ABC : Triangle)
variable (O1 O2 : Circle)
variable (L : Point)
variable (K M : Point)

-- Conditions from the problem
axiom tangent_O1_AC : tangent (Circle.center O1) (Triangle.side AC ABC)
axiom tangent_O1_AB : tangent (Circle.center O1) (Triangle.side AB ABC)
axiom tangent_O2_AC : tangent (Circle.center O2) (Triangle.side AC ABC)
axiom tangent_O2_extAB : tangent (Circle.center O2) (lineExtension (Triangle.side AB ABC))
axiom tangent_O1_O2_at_L : tangentAt O1 O2 L
axiom L_on_BC : L ∈ (Triangle.side BC ABC)
axiom AL_intersects_O1_at_K : K ∈ (LineThrough (Triangle.vertex A ABC) L) ∧ K ≠ (Triangle.vertex A ABC) ∧ K ≠ L ∧ K ∈ O1
axiom AL_intersects_O2_at_M : M ∈ (LineThrough (Triangle.vertex A ABC) L) ∧ M ≠ (Triangle.vertex A ABC) ∧ M ≠ L ∧ M ∈ O2
axiom KB_parallel_CM : parallel (LineThrough K (Triangle.vertex B ABC)) (LineThrough M (Triangle.vertex C ABC))

-- Proving the result based on the conditions
theorem triangle_LCM_is_isosceles : is_isosceles (Triangle.mk L (Triangle.vertex C ABC) M) := sorry

end triangle_LCM_is_isosceles_l318_318239


namespace cos_squared_difference_l318_318813

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318813


namespace cos_diff_square_identity_l318_318821

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318821


namespace even_function_of_square_l318_318403

section
variable (x : ℝ)

def f (x : ℝ) := x^2

theorem even_function_of_square : f(-x) = f(x) :=
by
  unfold f
  simp
  exact congr_arg (f) (neg_sq x)
end

end even_function_of_square_l318_318403


namespace externally_tangent_circles_l318_318268

theorem externally_tangent_circles (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + y^2 = 4 → x^2 + (y - real.sqrt 5)^2 = a^2)
  ∧ (|a| = 1/4) :=
begin
  sorry
end

end externally_tangent_circles_l318_318268


namespace simplify_expression_l318_318782

-- Define the main condition
def a : ℝ := sqrt 3 - 1

-- Define the expression to be simplified
def expression (a : ℝ) : ℝ := (1 - 3 / (a + 2)) / ((a ^ 2 - 1) / (a + 2))

-- The theorem to be proved
theorem simplify_expression : expression a = sqrt 3 / 3 := 
by sorry

end simplify_expression_l318_318782


namespace angle_is_120_deg_l318_318607

open Real

variables (a b c : ℝ × ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  acos (dot_product u v / (magnitude u * magnitude v))

-- Conditions
axiom ha : a = (1, 2)
axiom hb : b = (-2, -4)
axiom hc : magnitude c = sqrt 5
axiom h_dot : dot_product (c - b) a = 15 / 2

-- Problem statement
theorem angle_is_120_deg : angle_between_vectors a c = (2/3) * π := sorry

end angle_is_120_deg_l318_318607


namespace tetrahedron_volume_tetrahedron_height_l318_318159

-- Define the points A1, A2, A3, and A4
def A1 : ℝ × ℝ × ℝ := (3, 10, -1)
def A2 : ℝ × ℝ × ℝ := (-2, 3, -5)
def A3 : ℝ × ℝ × ℝ := (-6, 0, -3)
def A4 : ℝ × ℝ × ℝ := (1, -1, 2)

-- Define the function to calculate the volume of the tetrahedron
def volume_tetrahedron (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ := 
  let (x1, y1, z1) := A₁ 
  let (x2, y2, z2) := A₂ 
  let (x3, y3, z3) := A₃ 
  let (x4, y4, z4) := A₄
  (1 / 6) * abs (
    (x2 - x1) * ((y3 - y1) * (z4 - z1) - (z3 - z1) * (y4 - y1)) -
    (y2 - y1) * ((x3 - x1) * (z4 - z1) - (z3 - z1) * (x4 - x1)) +
    (z2 - z1) * ((x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1))
  )

-- Define the function to calculate the height of the tetrahedron
def height_tetrahedron (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ := 
  let base_area : ℝ :=
    let (x1, y1, z1) := A₁ 
    let (x2, y2, z2) := A₂ 
    let (x3, y3, z3) := A₃
    (1/2) * real.sqrt (
      (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    ) ^ 2 +
    ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)) ^ 2 +
    ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) ^ 2
  
  let vol := volume_tetrahedron A₁ A₂ A₃ A₄
  (3 * vol) / base_area

-- Prove the volume and height of the tetrahedron
theorem tetrahedron_volume : volume_tetrahedron A1 A2 A3 A4 = 45.5 :=
  by 
    sorry

theorem tetrahedron_height : height_tetrahedron A1 A2 A3 A4 = 7 :=
  by 
    sorry

end tetrahedron_volume_tetrahedron_height_l318_318159


namespace minimum_possible_value_l318_318517

noncomputable def minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a / (3 * b) + b / (6 * c) + c / (9 * a))

theorem minimum_possible_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  minimum_value a b c ha hb hc ≥ 3 * (1 / real.cbrt 162) :=
sorry

end minimum_possible_value_l318_318517


namespace fixed_point_A_l318_318726

-- Definitions from the conditions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 1

-- The statement to prove
theorem fixed_point_A (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) : f (a) (-2) = -1 :=
  by sorry

end fixed_point_A_l318_318726


namespace xy_value_l318_318286

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l318_318286


namespace length_AB_l318_318997

open Real

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

theorem length_AB (x1 y1 x2 y2 : ℝ) 
  (hA : y1^2 = 4 * x1) (hB : y2^2 = 4 * x2) 
  (hLine: (y2 - y1) * 1 = (x2 - x1) *0)
  (hSum : x1 + x2 = 6) : 
  dist (x1, y1) (x2, y2) = 8 := 
sorry

end length_AB_l318_318997


namespace part_1_part_2_l318_318240

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {λ : ℝ}
variable (a_2 a_3 a_4 a_5 : ℝ) (increasing_geom_seq : ∀ n, a_n n ≤ a_n (n + 1))
variable (a2a5 : a_2 * a_5 = 32) (a3a4 : a_3 + a_4 = 12) (b1 : b_n 1 = 1)
variable (bn_recurrence : ∀ n, b_n (n+1) = 2 * b_n n + 2 * a_n n)

theorem part_1 : (∀ n, (b_n (n+1)) / (a_n (n+1)) = (b_n n) / (a_n n) + 1) →
  (∀ m n, ((b_n m) / (a_n m) - (b_n n) / (a_n n)) = (m - n)) :=
sorry  -- proof omitted

theorem part_2 : (∀ n, (n+2) * b_n n + 1 ≥ λ * b_n n) → λ ≤ 3 :=
sorry  -- proof omitted

end part_1_part_2_l318_318240


namespace contradiction_proof_l318_318066

theorem contradiction_proof {E : Prop} : 
  (∃ x1 x2, x1 ≠ x2 ∧ E x1 ∧ E x2) ↔ ¬(∀ x1 x2, E x1 ∧ E x2 → x1 = x2) :=
by
  sorry

end contradiction_proof_l318_318066


namespace circle_through_points_and_center_on_line_l318_318219

noncomputable def circle_equation (x y : ℝ) : ℝ := 
(x - 3)^2 + (y - 2)^2 - 13

theorem circle_through_points_and_center_on_line :
  (circle_equation 6 0 = 0) ∧ (circle_equation 1 5 = 0) ∧ (2 * 3 - 7 * 2 + 8 = 0) :=
by
  split
  case left =>
    -- Show first point satisfies the circle equation
    sorry
  case right =>
    split
    case left =>
      -- Show second point satisfies the circle equation
      sorry
    case right =>
      -- Show the center satisfies the line equation
      sorry

end circle_through_points_and_center_on_line_l318_318219


namespace option_A_cannot_determine_parallelogram_options_B_C_D_can_determine_parallelogram_l318_318145

def Quad (A B C D : Type) := A × B × C × D

def is_parallelogram {A B C D : Type} (q : Quad A B C D) : Prop :=
  -- Dummy definition for is_parallelogram, to be replaced with actual properties that define a parallelogram.
  sorry

def cond_A {A B C D : Type} (q : Quad A B C D) : Prop :=
  -- Assume q is (a, b, c, d)
  let (a, b, c, d) := q in (parallel a b) ∧ (equal_length d b)

def cond_B {A B C D : Type} (q : Quad A B C D) : Prop :=
  -- Assume q is (A, B, C, D)
  let (A, B, C, D) := q in (equal_angle A C) ∧ (equal_angle B D)

def cond_C {A B C D : Type} (q : Quad A B C D) : Prop :=
  -- Assume q is (a, b, c, d)
  let (a, b, c, d) := q in (parallel a b) ∧ (parallel d b)

def cond_D {A B C D : Type} (q : Quad A B C D) : Prop :=
  -- Assume q is (a, b, c, d)
  let (a, b, c, d) := q in (equal_length a b) ∧ (equal_length d b)

theorem option_A_cannot_determine_parallelogram (A B C D : Type) (q : Quad A B C D) :
  cond_A q → ¬ is_parallelogram q :=
by
  sorry

theorem options_B_C_D_can_determine_parallelogram (A B C D : Type) (q : Quad A B C D) :
  (cond_B q ∧ cond_C q ∧ cond_D q) → is_parallelogram q :=
by
  sorry

end option_A_cannot_determine_parallelogram_options_B_C_D_can_determine_parallelogram_l318_318145


namespace part1_part2_l318_318608

-- Define A and B according to given expressions
def A (a b : ℚ) : ℚ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) : ℚ := -a^2 + a * b - 1

-- Prove the first statement
theorem part1 (a b : ℚ) : 4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 :=
by sorry

-- Prove the second statement
theorem part2 (F : ℚ) (b : ℚ) : (∀ a, A a b + 2 * B a b = F) → b = 2 / 5 :=
by sorry

end part1_part2_l318_318608


namespace intersection_points_distance_l318_318196

noncomputable def parametric_line_eq : ℝ → ℝ × ℝ := λ t, (sqrt 3 / 2 * t, 2 + 1 / 2 * t)
noncomputable def circle_eq (x y : ℝ) : ℝ := (x - 3)^2 + (y - 2)^2 - 9

theorem intersection_points_distance (t₁ t₂ : ℝ) :
  let A := parametric_line_eq t₁
  let B := parametric_line_eq t₂
  let dist_P_A := |t₁|
  let dist_P_B := |t₂|
  (circle_eq A.1 A.2 = 0 ∧ circle_eq B.1 B.2 = 0 ∧ t₁ * t₂ = -7) → dist_P_A * dist_P_B = 7 := by
  sorry

end intersection_points_distance_l318_318196


namespace solve_x_l318_318194

theorem solve_x (x : ℝ) (h : x ≠ -1) : 
  (x^3 - 3 * x^2 + 2 * x) / (x^2 + 2 * x + 1) + 2 * x = -8 ↔ x = -4 / 3 :=
by
suffices 
  (∀ x : ℝ, x ≠ -1 → (x^3 - 3 * x^2 + 2 * x) / (x^2 + 2 * x + 1) + 2 * x = -8 → x = -4 / 3) 
  from sorry

end solve_x_l318_318194


namespace sum_of_legs_30sqrt5_l318_318771

noncomputable def sum_of_legs_of_larger_triangle
  (area_small : ℝ) 
  (area_large : ℝ) 
  (hypotenuse_small : ℝ) 
  (sum_of_legs_large : ℝ) : Prop :=
  ∀ (a b : ℝ), 
  (1/2 * a * b = area_small) → 
  (a^2 + b^2 = hypotenuse_small^2) → 
  (sqrt (area_large / area_small) * (a + b) = sum_of_legs_large)

theorem sum_of_legs_30sqrt5 :
  sum_of_legs_of_larger_triangle 10 250 10 (30 * sqrt 5) :=
by
  sorry

end sum_of_legs_30sqrt5_l318_318771


namespace S_eq_Z_l318_318674

noncomputable def S : Set ℤ := sorry -- Placeholder for definition of S

open Int

variables (a b x y : ℤ)

-- Conditions
axiom cond1 : a ∈ S ∧ b ∈ S
axiom cond2 : gcd a b = 1
axiom cond3 : gcd (a - 2) (b - 2) = 1
axiom cond4 : ∀ x y ∈ S, (x^2 - y) ∈ S

-- Question: Prove that S = ℤ
theorem S_eq_Z : S = Set.univ := 
sorry

end S_eq_Z_l318_318674


namespace region_R_area_is_correct_l318_318709

def angle := ℝ
def area := ℝ

structure Rhombus :=
(side : ℝ)
(angle : angle)

def region_area (r : Rhombus) : area :=
  if r.side = 3 ∧ r.angle = 110 then 2.16 else 0

theorem region_R_area_is_correct :
  ∀ r : Rhombus, r.side = 3 → r.angle = 110 → region_area r = 2.16 :=
by
  intro r
  intro h1 h2
  /-
    Assume:
    - r.side = 3
    - r.angle = 110
    Show:
    - region_area r = 2.16
  -/
  sorry

end region_R_area_is_correct_l318_318709


namespace find_digits_l318_318800

namespace MathPuzzle

def S : ℕ := 9
def O : ℕ := 0
def E : ℕ := 5

theorem find_digits : 
  ∃ M N R Y : ℕ, 
    (S = 9) ∧ 
    (O = 0) ∧ 
    (E = 5) ∧ 
    (S + M = 10) ∧ 
    (E + O + 1 = N) ∧ 
    (6 + R + 1 = 15) ∧ 
    (7 + E = 10 + Y) ∧ 
    (M ≠ N) ∧ (M ≠ R) ∧ (M ≠ Y) ∧ (M ≠ S) ∧ (M ≠ O) ∧ (M ≠ E) ∧ 
    (N ≠ R) ∧ (N ≠ Y) ∧ (N ≠ S) ∧ (N ≠ O) ∧ (N ≠ E) ∧ 
    (R ≠ Y) ∧ (R ≠ S) ∧ (R ≠ O) ∧ (R ≠ E) ∧
    (Y ≠ S) ∧ (Y ≠ O) ∧ (Y ≠ E)
  :=
  by 
    have M := 1
    have N := 6
    have R := 8
    have Y := 2
    use M, N, R, Y
    simp [S, O, E, M, N, R, Y]
    sorry

end MathPuzzle

end find_digits_l318_318800


namespace cos_difference_squared_l318_318964

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318964


namespace greatest_positive_multiple_of_4_l318_318391

theorem greatest_positive_multiple_of_4 {y : ℕ} (h1 : y % 4 = 0) (h2 : y > 0) (h3 : y^3 < 8000) : y ≤ 16 :=
by {
  -- The proof will go here
  -- Sorry is placed here to skip the proof for now
  sorry
}

end greatest_positive_multiple_of_4_l318_318391


namespace tan_alpha_solution_l318_318613

theorem tan_alpha_solution
  (α : ℝ)
  (h1 : π < α ∧ α < 3 * π / 2)
  (h2 : tan (2 * α) = -cos α / (2 + sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_solution_l318_318613


namespace number_of_ones_in_binary_representation_l318_318656

-- Define the polynomial and the key property to be proven
def poly (x : ℕ) : ℕ := (List.range 2011).foldr (λ n acc, acc * (x - 2^n)) 1 - 1

def S : ℕ := (List.range 2011).sum (λ k, 2^(k * 2011))

-- The theorem that encapsulates the problem statement
theorem number_of_ones_in_binary_representation :
  (S.to_digits 2).count 1 = 2011 :=
sorry

end number_of_ones_in_binary_representation_l318_318656


namespace elroy_miles_difference_l318_318182

theorem elroy_miles_difference (earning_rate_last_year earning_rate_this_year total_collection_last_year : ℝ)
  (rate_last_year : earning_rate_last_year = 4)
  (rate_this_year : earning_rate_this_year = 2.75)
  (total_collected : total_collection_last_year = 44) :
  (total_collection_last_year / earning_rate_this_year) - (total_collection_last_year / earning_rate_last_year) = 5 :=
by
  rw [rate_last_year, rate_this_year, total_collected]
  norm_num
  sorry

end elroy_miles_difference_l318_318182


namespace sector_angle_sector_max_area_l318_318099

-- Part (1)
theorem sector_angle (r l : ℝ) (α : ℝ) :
  2 * r + l = 10 → (1 / 2) * l * r = 4 → α = l / r → α = 1 / 2 :=
by
  intro h1 h2 h3
  sorry

-- Part (2)
theorem sector_max_area (r l : ℝ) (α S : ℝ) :
  2 * r + l = 40 → α = l / r → S = (1 / 2) * l * r →
  (∀ r' l' α' S', 2 * r' + l' = 40 → α' = l' / r' → S' = (1 / 2) * l' * r' → S ≤ S') →
  r = 10 ∧ α = 2 ∧ S = 100 :=
by
  intro h1 h2 h3 h4
  sorry

end sector_angle_sector_max_area_l318_318099


namespace cos_squared_difference_l318_318947

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318947


namespace arithmetic_mean_of_fractions_l318_318779

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 8 + (5 : ℚ) / 12 / 2 = 19 / 48 := by
  sorry

end arithmetic_mean_of_fractions_l318_318779


namespace quotient_is_zero_l318_318006

def square_mod_16 (n : ℕ) : ℕ :=
  (n * n) % 16

def distinct_remainders_in_range : List ℕ :=
  List.eraseDup $
    List.map square_mod_16 (List.range' 1 15)

def sum_of_distinct_remainders : ℕ :=
  distinct_remainders_in_range.sum

theorem quotient_is_zero :
  (sum_of_distinct_remainders / 16) = 0 :=
by
  sorry

end quotient_is_zero_l318_318006


namespace cosine_difference_identity_l318_318850

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318850


namespace cos_difference_squared_l318_318963

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318963


namespace ratio_of_doctors_to_nurses_l318_318055

theorem ratio_of_doctors_to_nurses (total_staff doctors nurses : ℕ) (h1 : total_staff = 456) (h2 : nurses = 264) (h3 : doctors + nurses = total_staff) :
  doctors = 192 ∧ (doctors : ℚ) / nurses = 8 / 11 :=
by
  sorry

end ratio_of_doctors_to_nurses_l318_318055


namespace cos_squared_difference_l318_318802

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318802


namespace hyperbola_equation_proof_line_intersects_proof_circle_contains_proof_l318_318597

def hyperbola_equation_and_eccentricity (a b c : ℝ) : Prop :=
  a^2 = 3 ∧ b = 1 ∧ c = 2 ∧ (c^2 = a^2 + b^2) ∧ ( ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 )
  ∧ ( ∀ e, e = c / a )

def line_intersects_hyperbola_at_two_points (a b c : ℝ) (k m : ℝ) : Prop :=
  a^2 = 3 ∧ b = 1 ∧ c = 2 ∧ (c^2 = a^2 + b^2) ∧
  (∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1) ∧
  ( ∀ l, l = k*x + m ∧ m ≠ 0 ) ∧
  ( ∃ k, k = 1 ∨ k = -1 ∨ k = 0 ∧
  (l = (λ x, x - 2) ∨ l = (λ x, -x + 2) ∨ l = (λ x, 0 )))

def circle_contains_both_points (a b c : ℝ) (k m : ℝ) (D : ℝ × ℝ) : Prop :=
  a^2 = 3 ∧ b = 1 ∧ c = 2 ∧ (c^2 = a^2 + b^2) ∧
  (∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1) ∧
  ( ∀ l, l = k*x + m ∧ m ≠ 0 ) ∧
  ( D = (0, -1) ∧ (m ∈ Set.Icc -0.25 0 ∨  ( 0 < m  ∧ 4 < m )))

-- Proofs
theorem hyperbola_equation_proof : (a b c : ℝ) → hyperbola_equation_and_eccentricity a b c :=
begin
  intros a b c,
  split,
  exact sorry,
end

theorem line_intersects_proof : (a b c : ℝ) → hyperbola_equation_and_eccentricity a b c → (k m : ℝ)  → line_intersects_hyperbola_at_two_points a b c k m :=
begin
  intros a b c habc k m,
  exact sorry,
end

theorem circle_contains_proof : (a b c : ℝ) → hyperbola_equation_and_eccentricity a b c → (k m : ℝ) → (D : ℝ × ℝ) →  circle_contains_both_points a b c k m D :=
begin
  intros a b c habc k m D,
  exact sorry,
end

end hyperbola_equation_proof_line_intersects_proof_circle_contains_proof_l318_318597


namespace alcohol_fraction_after_tripling_water_l318_318134

noncomputable def initial_volume (v : ℝ) : ℝ := v
noncomputable def alcohol_fraction : ℝ := 2 / 3
noncomputable def water_fraction : ℝ := 1 / 3

noncomputable def initial_alcohol_volume (v : ℝ) : ℝ := alcohol_fraction * v
noncomputable def initial_water_volume (v : ℝ) : ℝ := water_fraction * v

noncomputable def tripled_water_volume (v : ℝ) : ℝ := 3 * initial_water_volume v
noncomputable def new_mixture_volume (v : ℝ) : ℝ := initial_alcohol_volume v + tripled_water_volume v

theorem alcohol_fraction_after_tripling_water (v : ℝ) : (initial_alcohol_volume v) / (new_mixture_volume v) = 2 / 5 :=
by
  rw [initial_alcohol_volume, tripled_water_volume, new_mixture_volume, initial_water_volume, alcohol_fraction, water_fraction]
  field_simp
  norm_num
  sorry

end alcohol_fraction_after_tripling_water_l318_318134


namespace geom_series_sum_l318_318504

/-- The sum of the first six terms of the geometric series 
    with first term a = 1 and common ratio r = (1 / 4) is 1365 / 1024. -/
theorem geom_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1 / 4
  let n : ℕ := 6
  (a * (1 - r^n) / (1 - r)) = 1365 / 1024 :=
by
  sorry

end geom_series_sum_l318_318504


namespace duration_of_fourth_episode_l318_318332

theorem duration_of_fourth_episode :
    ∀ (ep1 ep2 ep3 ep4 : ℕ), 
    ep1 = 58 →
    ep2 = 62 →
    ep3 = 65 →
    (ep1 + ep2 + ep3 + ep4 = 4 * 60) →
    ep4 = 55 :=
by
intros ep1 ep2 ep3 ep4 h1 h2 h3 h4
rw [h1, h2, h3] at h4
simp at h4
exact h4

end duration_of_fourth_episode_l318_318332


namespace ferry_time_difference_l318_318088

-- Definitions for the given conditions
def speed_p := 8
def time_p := 3
def distance_p := speed_p * time_p
def distance_q := 3 * distance_p
def speed_q := speed_p + 1
def time_q := distance_q / speed_q

-- Theorem to be proven
theorem ferry_time_difference : (time_q - time_p) = 5 := 
by
  let speed_p := 8
  let time_p := 3
  let distance_p := speed_p * time_p
  let distance_q := 3 * distance_p
  let speed_q := speed_p + 1
  let time_q := distance_q / speed_q
  sorry

end ferry_time_difference_l318_318088


namespace rectangle_area_l318_318457

theorem rectangle_area (r : ℝ) (h_r : r = 7) 
  (h_ratio : ∀ w l : ℝ, l = 3 * w) : 
    let w := 2 * r,
        l := 3 * w
    in l * w = 588 :=
by
  sorry

end rectangle_area_l318_318457


namespace a_n_formula_T_n_formula_l318_318224

variables {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℕ} {T : ℕ → ℕ}

-- Given conditions:
-- a_1 = 2
axiom a_1 : a 1 = 2
-- S_n = 2 * a_n - 2
axiom Sn_def : ∀ n, S n = 2 * a n - 2
-- b_n = n * a_n
def b (n : ℕ) : ℕ := n * a n

-- Problem (1): Prove:
-- a_n = 2^n
theorem a_n_formula (n : ℕ) : a n = 2^n :=
sorry

-- Problem (2): Prove:
-- T_n = 2 + (n-1) * 2^(n+1)
theorem T_n_formula (n : ℕ) : T n = 2 + (n-1) * 2^(n+1) :=
sorry

end a_n_formula_T_n_formula_l318_318224


namespace limit_of_fraction_at_zero_l318_318157

theorem limit_of_fraction_at_zero :
  tendsto (λ x : ℝ, (exp (4 * x) - exp (-2 * x)) / (2 * arctan x - sin x)) (𝓝 0) (𝓝 6) :=
sorry

end limit_of_fraction_at_zero_l318_318157


namespace cosine_difference_identity_l318_318848

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318848


namespace snow_fall_time_l318_318619

theorem snow_fall_time (rate_mm_per_minute : ℕ) (time_minute : ℕ) (time_hour : ℕ) (meter_to_cm : ℕ) (cm_to_mm : ℕ) : 
(rate_mm_per_minute = 1 / 6) → 
(time_minute = 60) → 
(meter_to_cm = 100) → 
(cm_to_mm = 10) →
let rate_mm_per_hour := rate_mm_per_minute * time_minute in
let rate_cm_per_hour := rate_mm_per_hour / cm_to_mm in
let time_required_for_1m := meter_to_cm / rate_cm_per_hour in
time_required_for_1m = 100 := 
by
  intros rate_cond time_cond meter_cond cm_cond
  rw [rate_cond, time_cond, meter_cond, cm_cond]
  simp [rate_mm_per_hour, rate_cm_per_hour, time_required_for_1m]
  sorry

end snow_fall_time_l318_318619


namespace integral_value_l318_318140

noncomputable def integral_circle_quarter : ℝ :=
  ∫ x in 0..2, (Real.sqrt (-x^2 + 4 * x))

theorem integral_value :
  integral_circle_quarter = Real.pi :=
  sorry

end integral_value_l318_318140


namespace graphene_scientific_notation_l318_318272

def scientific_notation (n : ℝ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * 10 ^ exp ∧ 1 ≤ abs a ∧ abs a < 10

theorem graphene_scientific_notation :
  scientific_notation 0.00000000034 3.4 (-10) :=
by {
  sorry
}

end graphene_scientific_notation_l318_318272


namespace age_solution_l318_318115

theorem age_solution (M S : ℕ) (h1 : M = S + 16) (h2 : M + 2 = 2 * (S + 2)) : S = 14 :=
by sorry

end age_solution_l318_318115


namespace maximize_area_difference_l318_318461

def point_P : ℝ × ℝ := (1, 1)
def circle_radius : ℝ := 3
def circle_region (x y : ℝ) : Prop := x^2 + y^2 ≤ circle_radius^2

theorem maximize_area_difference (L : ℝ → ℝ → Prop) :
  (∀ x y, L x y ↔ x + y - 2 = 0) ∧
  (∀ x y, x^2 + y^2 = circle_radius^2 ∧ L x y = ∀ z w, z^2 + w^2 = circle_radius^2 ∧ ¬L z w → abs((π * circle_radius^2) - (π * 0)) ≥ abs((π * circle_radius^2) - (π / 2 * circle_radius^2)) :=
begin
  sorry
end

end maximize_area_difference_l318_318461


namespace largest_prime_divisor_of_13fact_plus_14fact_times_2_l318_318530

theorem largest_prime_divisor_of_13fact_plus_14fact_times_2 :
  ∃ p, prime p ∧ (p ∣ (13! + 14! * 2)) ∧ ∀ q, prime q ∧ (q ∣ (13! + 14! * 2)) → q ≤ p ∧ p = 29 :=
by
  sorry

end largest_prime_divisor_of_13fact_plus_14fact_times_2_l318_318530


namespace charging_time_is_correct_l318_318125

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l318_318125


namespace min_sum_fraction_l318_318356

theorem min_sum_fraction (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 4.5 / 11 < a / b ∧ a / b < 5 / 11) : 
  a = 3 ∧ b = 7 :=
by
  sorry

end min_sum_fraction_l318_318356


namespace galaxy_destruction_probability_l318_318096

theorem galaxy_destruction_probability :
  let m := 45853
  let n := 65536
  m + n = 111389 :=
by
  sorry

end galaxy_destruction_probability_l318_318096


namespace min_value_y_l318_318556

theorem min_value_y (x : ℝ) (hx : x > 2) : 
  ∃ x, x > 2 ∧ (∀ y, y = (x^2 - 4*x + 8) / (x - 2) → y ≥ 4 ∧ y = 4 ↔ x = 4) :=
sorry

end min_value_y_l318_318556


namespace cos_squared_difference_l318_318803

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318803


namespace find_smaller_integer_l318_318376

theorem find_smaller_integer (x : ℤ) (h1 : ∃ y : ℤ, y = 2 * x) (h2 : x + 2 * x = 96) : x = 32 :=
sorry

end find_smaller_integer_l318_318376


namespace trigonometric_identity_l318_318863

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318863


namespace fractional_eq_solution_range_l318_318751

theorem fractional_eq_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 3 ∧ x > 0) ↔ m < -3 :=
by
  sorry

end fractional_eq_solution_range_l318_318751


namespace length_of_platform_l318_318084

def len_train : ℕ := 300 -- length of the train in meters
def time_platform : ℕ := 39 -- time to cross the platform in seconds
def time_pole : ℕ := 26 -- time to cross the signal pole in seconds

theorem length_of_platform (L : ℕ) (h1 : len_train / time_pole = (len_train + L) / time_platform) : L = 150 :=
  sorry

end length_of_platform_l318_318084


namespace proof_problem_l318_318895

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318895


namespace minimum_perimeter_is_12_l318_318220

noncomputable def minimum_perimeter_upper_base_frustum
  (a b : ℝ) (h : ℝ) (V : ℝ) : ℝ :=
if h = 3 ∧ V = 63 ∧ (a * b = 9) then
  2 * (a + b)
else
  0 -- this case will never be used

theorem minimum_perimeter_is_12 :
  ∃ a b : ℝ, a * b = 9 ∧ 2 * (a + b) = 12 :=
by
  existsi 3
  existsi 3
  sorry

end minimum_perimeter_is_12_l318_318220


namespace patio_tiles_l318_318129

theorem patio_tiles (r c : ℕ) (h1 : r * c = 48) (h2 : (r + 4) * (c - 2) = 48) : r = 6 :=
sorry

end patio_tiles_l318_318129


namespace hoseok_position_from_back_l318_318717

theorem hoseok_position_from_back
    (n : ℕ)
    (Hoseok_pos : ℕ)
    (total_people : n = 9)
    (Hoseok_order : Hoseok_pos = 5)
    : Hoseok_pos = (n - 4) :=
by
  rw [total_people, Hoseok_order]
  exact rfl

end hoseok_position_from_back_l318_318717


namespace cos_squared_difference_l318_318911

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318911


namespace shortest_distance_l318_318487

theorem shortest_distance 
  (ΔX : ℝ) (ΔY : ℝ) 
  (hx1 : ΔX = 5 + 0 - 1)
  (hy1 : ΔY = 0 + 4 - 1) :
  sqrt (ΔX^2 + ΔY^2) = 5 := 
by
  sorry

end shortest_distance_l318_318487


namespace inequality_solution_set_correct_l318_318213

noncomputable def inequality_solution_set (a b c x : ℝ) : Prop :=
  (a > c) → (b + c > 0) → ((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0)

theorem inequality_solution_set_correct (a b c : ℝ) :
  a > c → b + c > 0 → ∀ x, ((a > c) → (b + c > 0) → (((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0))) :=
by
  intros h1 h2 x
  sorry

end inequality_solution_set_correct_l318_318213


namespace min_tangent_slope_l318_318039

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x + x^2 - b * x + a

def f_prime (x : ℝ) (b : ℝ) : ℝ := 1 / x + 2 * x - b

theorem min_tangent_slope (a b : ℝ) (hb : b > 0) : 
  ∃ x, f_prime b b = 2 :=
begin
  use 1,
  rw [f_prime, div_one, mul_one],
  norm_num,
end

end min_tangent_slope_l318_318039


namespace equivalent_systems_solution_and_value_l318_318629

-- Definitions for the conditions
def system1 (x y a b : ℝ) : Prop := 
  (2 * (x + 1) - y = 7) ∧ (x + b * y = a)

def system2 (x y a b : ℝ) : Prop := 
  (a * x + y = b) ∧ (3 * x + 2 * (y - 1) = 9)

-- The proof problem as a Lean 4 statement
theorem equivalent_systems_solution_and_value (a b : ℝ) :
  (∃ x y : ℝ, system1 x y a b ∧ system2 x y a b) →
  ((∃ x y : ℝ, x = 3 ∧ y = 1) ∧ (3 * a - b) ^ 2023 = -1) :=
  by sorry

end equivalent_systems_solution_and_value_l318_318629


namespace total_pets_remaining_l318_318499

def initial_counts := (7, 6, 4, 5, 3)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def morning_sales := (1, 2, 1, 0, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def afternoon_sales := (1, 1, 2, 3, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def returns := (0, 1, 0, 1, 1)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)

def calculate_remaining (initial_counts morning_sales afternoon_sales returns : Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (p0, k0, r0, g0, c0) := initial_counts
  let (p1, k1, r1, g1, c1) := morning_sales
  let (p2, k2, r2, g2, c2) := afternoon_sales
  let (p3, k3, r3, g3, c3) := returns
  let remaining_puppies := p0 - p1 - p2 + p3
  let remaining_kittens := k0 - k1 - k2 + k3
  let remaining_rabbits := r0 - r1 - r2 + r3
  let remaining_guinea_pigs := g0 - g1 - g2 + g3
  let remaining_chameleons := c0 - c1 - c2 + c3
  remaining_puppies + remaining_kittens + remaining_rabbits + remaining_guinea_pigs + remaining_chameleons

theorem total_pets_remaining : calculate_remaining initial_counts morning_sales afternoon_sales returns = 15 := 
by
  simp [initial_counts, morning_sales, afternoon_sales, returns, calculate_remaining]
  sorry

end total_pets_remaining_l318_318499


namespace ethanol_percentage_in_fuel_A_l318_318484

noncomputable def percent_ethanol_in_fuel_A : ℝ := 0.12

theorem ethanol_percentage_in_fuel_A
  (fuel_tank_capacity : ℝ)
  (fuel_A_volume : ℝ)
  (fuel_B_volume : ℝ)
  (fuel_B_ethanol_percent : ℝ)
  (total_ethanol : ℝ) :
  fuel_tank_capacity = 218 → 
  fuel_A_volume = 122 → 
  fuel_B_volume = 96 → 
  fuel_B_ethanol_percent = 0.16 → 
  total_ethanol = 30 → 
  (fuel_A_volume * percent_ethanol_in_fuel_A) + (fuel_B_volume * fuel_B_ethanol_percent) = total_ethanol :=
by
  sorry

end ethanol_percentage_in_fuel_A_l318_318484


namespace indispensable_structure_l318_318393

theorem indispensable_structure (S : Type) [Nonempty S] (P : S → Prop) :
  (∀ a, P a) → ∃ a, P a := 
by
  intro h
  apply exists.intro
  sorry

end indispensable_structure_l318_318393


namespace point_to_plane_distance_l318_318528

def point := ℝ × ℝ × ℝ

noncomputable def distance_from_point_to_plane (M0 M1 M2 M3 : point) : ℝ :=
  let (x0, y0, z0) := M0
  let (x1, y1, z1) := M1
  let (x2, y2, z2) := M2
  let (x3, y3, z3) := M3
  let A := (y1 - y2) * (z1 - z3) - (z1 - z2) * (y1 - y3)
  let B := (z1 - z2) * (x1 - x3) - (x1 - x2) * (z1 - z3)
  let C := (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
  let D := -(A * x1 + B * y1 + C * z1)
  (|A * x0 + B * y0 + C * z0 + D|) / sqrt (A^2 + B^2 + C^2)

noncomputable def M0 : point := (-1, -8, 7)
noncomputable def M1 : point := (14, 4, 5)
noncomputable def M2 : point := (-5, -3, 2)
noncomputable def M3 : point := (-2, -6, -3)

theorem point_to_plane_distance :
  distance_from_point_to_plane M0 M1 M2 M3 = 3 * sqrt (13 / 2) :=
sorry

end point_to_plane_distance_l318_318528


namespace xy_value_l318_318287

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l318_318287


namespace elroy_more_miles_l318_318189

-- Given conditions
def last_year_rate : ℝ := 4
def this_year_rate : ℝ := 2.75
def last_year_collection : ℝ := 44

-- Goals
def last_year_miles : ℝ := last_year_collection / last_year_rate
def this_year_miles : ℝ := last_year_collection / this_year_rate
def miles_difference : ℝ := this_year_miles - last_year_miles

theorem elroy_more_miles :
  miles_difference = 5 := by
  sorry

end elroy_more_miles_l318_318189


namespace mapping_cardinality_l318_318337

def mapping_condition (f : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i ∈ finset.range (n+1) ∧ j ∈ finset.range (i+1) → 
    (f i = f j) → (¬ (finset.filter (λ k, f k = i) (finset.range (n+1))).nonempty implies
    ¬ (finset.filter (λ k, f k = j) (finset.range (n+1))).nonempty)

def A_n (n : ℕ) : finset (ℕ → ℕ) :=
  (finset.range (n+1) → finset.range (n+1)).filter (λ f, mapping_condition f n)

theorem mapping_cardinality (n : ℕ) : 
  (A_n n).card = ∑ k in finset.range (n + 1), (k^n) / (2^(k + 1)) := 
sorry

end mapping_cardinality_l318_318337


namespace polygon_angle_ratio_unique_pair_l318_318747

theorem polygon_angle_ratio_unique_pair
    (r k : ℕ)
    (h_r : 3 ≤ r)
    (h_k : 3 ≤ k)
    (h : (180 * r - 360) * (180 * k - 360)⁻¹ = (5 / 3)) :
    ∃! (r k : ℕ), (h_r : 3 ≤ r)∧ (h_k : 3 ≤ k)∧ (h : ((180 * r - 360) * (180 * k - 360)⁻¹ = (5 / 3))) := 
sorry

end polygon_angle_ratio_unique_pair_l318_318747


namespace gardner_bakes_brownies_l318_318366

theorem gardner_bakes_brownies : 
  ∀ (cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes : ℕ),
  cookies = 20 →
  cupcakes = 25 →
  students = 20 →
  sweet_treats_per_student = 4 →
  total_sweet_treats = students * sweet_treats_per_student →
  total_cookies_and_cupcakes = cookies + cupcakes →
  brownies = total_sweet_treats - total_cookies_and_cupcakes →
  brownies = 35 :=
by
  intros cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end gardner_bakes_brownies_l318_318366


namespace xy_value_l318_318288

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l318_318288


namespace max_winners_at_least_three_matches_l318_318492

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end max_winners_at_least_three_matches_l318_318492


namespace sequence_sum_integer_part_l318_318601

theorem sequence_sum_integer_part :
  let a : ℕ → ℝ := λ n, if n = 0 then 1/4 else (a (n-1))^2 + (a (n-1))
  let sum_term := λ n, 1 / (a n + 1)
  (⌊∑ n in Finset.range 2016, sum_term (n+1)⌋ = 3) :=
by
  sorry

end sequence_sum_integer_part_l318_318601


namespace num_integers_g_eq_1_l318_318200

noncomputable def g (n : ℤ) : ℤ := 
  ⌈98 * n / 101⌉ - ⌊101 * n / 102⌋

theorem num_integers_g_eq_1 : 
  {n : ℤ | g n = 1}.to_finset.card = 10302 :=
sorry

end num_integers_g_eq_1_l318_318200


namespace range_of_a_analytical_expression_l318_318579

variables {f : ℝ → ℝ}

-- Problem 1
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ x y, x < y → f x ≥ f y)
  {a : ℝ} (h_ineq : f (1 - a) + f (1 - 2 * a) < 0) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

-- Problem 2
theorem analytical_expression 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = x^2 + x + 1)
  (h_zero : f 0 = 0) :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f x = 
    if x > 0 then x^2 + x + 1
    else if x = 0 then 0
    else -x^2 + x - 1 :=
sorry

end range_of_a_analytical_expression_l318_318579


namespace cos_square_difference_l318_318868

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318868


namespace regular_tetrahedron_l318_318338

theorem regular_tetrahedron
  {T : Type} [normed_group T] [normed_space ℝ T]
  (A B C D : T)
  (h1 : dist A B = dist C D)
  (h2 : dist A C = dist B D)
  (h3 : dist A D = dist B C) :
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  dist D A = dist A C ∧ dist A C = dist B D ∧ dist B D = dist A B :=
sorry

end regular_tetrahedron_l318_318338


namespace limit_of_fraction_at_zero_l318_318158

theorem limit_of_fraction_at_zero :
  tendsto (λ x : ℝ, (exp (4 * x) - exp (-2 * x)) / (2 * arctan x - sin x)) (𝓝 0) (𝓝 6) :=
sorry

end limit_of_fraction_at_zero_l318_318158


namespace cake_mix_buyers_l318_318107

theorem cake_mix_buyers :
  let total_buyers := 100
  let muffin_mix_buyers := 40
  let both_mix_buyers := 19
  let p_neither := 0.29
  let neither_buyers := total_buyers * p_neither := 29
  let at_least_one_mix := total_buyers - neither_buyers := 71
  let only_muffin_mix := muffin_mix_buyers - both_mix_buyers := 21
  let cake_mix_buyers := at_least_one_mix - only_muffin_mix := 50 in
  cake_mix_buyers = 50 :=
by
  -- sorry to skip the proof
  sorry

end cake_mix_buyers_l318_318107


namespace difference_between_percentages_l318_318445

def percent := (p : ℝ) → (x : ℝ) → ℝ := λ p x, (p / 100) * x
def diff (a b : ℝ) := a - b

theorem difference_between_percentages :
  diff (percent 80 170) (percent 35 300) = 31 := by 
  sorry

end difference_between_percentages_l318_318445


namespace midpoint_xy_zero_l318_318662

variables (x y : ℝ)
def A : ℝ × ℝ := (2, 6)
def C : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (x, y)

-- Midpoint condition
def is_midpoint (A B C : ℝ × ℝ) : Prop :=
  (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

theorem midpoint_xy_zero (h : is_midpoint A B C) : x * y = 0 :=
by
  sorry

end midpoint_xy_zero_l318_318662


namespace cos_squared_difference_l318_318889

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318889


namespace cos_squared_difference_l318_318948

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318948


namespace glove_selection_l318_318548

theorem glove_selection (pairs_gloves : Finset (Finset (Fin 12))) :
  (pairs_gloves.card = 6) →
  (∀ g ∈ pairs_gloves, g.card = 2) →
  ∃ selected_gloves : Finset (Fin 12),
    (selected_gloves.card = 4) ∧
    (∃ p ∈ selected_gloves.powerset, p.card = 2 ∧ (p ∈ pairs_gloves)) ∧
    (sum (λ p, if p.card = 2 ∧ ∃ g ∈ pairs_gloves, p ∈ g.powerset then 1 else 0) (powerset selected_gloves) = 240) :=
begin
  intro h_pairs_card,
  intro h_each_pair,
  -- Further proof steps should be provided here
  sorry
end

end glove_selection_l318_318548


namespace bug_crawl_distance_l318_318469

noncomputable def least_distance_on_cone (r h d1 d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let C := 2 * Real.pi * r
  let θ := C / R
  let start := (d1, 0)
  let end := (d2 * (Cos (θ / 2)), d2 * (Sin (θ / 2)))
  Real.sqrt ((end.1 - start.1)^2 + (end.2 - start.2)^2)

theorem bug_crawl_distance :
  least_distance_on_cone 500 (150 * Real.sqrt 7) 100 (300 * Real.sqrt 2) ≈ 778.497 :=
sorry

end bug_crawl_distance_l318_318469


namespace explicit_formula_of_odd_function_monotonicity_in_interval_l318_318249

-- Using Noncomputable because divisions are involved.
noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

theorem explicit_formula_of_odd_function (p q : ℝ) 
  (h_odd : ∀ x : ℝ, f x p q = - f (-x) p q) 
  (h_value : f 2 p q = -5/3) : 
  f x 2 0 = -2/3 * (x + 1/x) :=
by sorry

theorem monotonicity_in_interval {x : ℝ} (h_domain : 0 < x ∧ x < 1) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 -> f x1 2 0 < f x2 2 0 :=
by sorry

end explicit_formula_of_odd_function_monotonicity_in_interval_l318_318249


namespace minimum_number_of_different_numbers_l318_318756

theorem minimum_number_of_different_numbers (total_numbers : ℕ) (frequent_count : ℕ) (frequent_occurrences : ℕ) (less_frequent_occurrences : ℕ) (h1 : total_numbers = 2019) (h2 : frequent_count = 10) (h3 : less_frequent_occurrences = 9) : ∃ k : ℕ, k ≥ 225 :=
by {
  sorry
}

end minimum_number_of_different_numbers_l318_318756


namespace joe_video_game_months_l318_318650

theorem joe_video_game_months : 
  let max_expense := 75
  let min_income := 20
  let net_monthly_expense := max_expense - min_income
  let starting_amount := 240
  let n := starting_amount / net_monthly_expense
  floor n = 4 := 
by
  let max_expense := 75
  let min_income := 20
  let net_monthly_expense := max_expense - min_income
  let starting_amount := 240
  let n := starting_amount / net_monthly_expense
  sorry

end joe_video_game_months_l318_318650


namespace tangent_line_at_M_l318_318229

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define point A, where the circle is tangent to the x-axis
def point_A (x y : ℝ) : Prop := y = 0 ∧ circle x y

-- Define point B, where the circle is tangent to the y-axis
def point_B (x y : ℝ) : Prop := x = 0 ∧ circle x y

-- Define M, the midpoint of the minor arc \hat{AB}
def point_M (x y : ℝ) : Prop := y = -x ∧ circle x y

-- Define the correct equation of the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 2 - sqrt 2

-- The proof statement
theorem tangent_line_at_M :
  ∃ x y : ℝ, point_M x y ∧ tangent_line x y := sorry

end tangent_line_at_M_l318_318229


namespace students_prefer_dogs_l318_318179

def total_students : ℕ := 30
def percent_dog_video_games : ℕ := 50
def percent_dog_movies : ℕ := 10

theorem students_prefer_dogs 
    (total_students = 30)
    (percent_dog_video_games = 50)
    (percent_dog_movies = 10) 
  : (percent_dog_video_games * total_students / 100) + (percent_dog_movies * total_students / 100) = 18 := 
  sorry

end students_prefer_dogs_l318_318179


namespace sugar_more_than_flour_l318_318362

def flour_needed : Nat := 9
def sugar_needed : Nat := 11
def flour_added : Nat := 4
def sugar_added : Nat := 0

def flour_remaining : Nat := flour_needed - flour_added
def sugar_remaining : Nat := sugar_needed - sugar_added

theorem sugar_more_than_flour : sugar_remaining - flour_remaining = 6 :=
by
  sorry

end sugar_more_than_flour_l318_318362


namespace operation_38_to_3800_l318_318142

theorem operation_38_to_3800 :
  ∀ op, op = (λ x : ℕ, x / 3800) ∨ op = (λ x : ℕ, x * 100) ∨ op = (λ x : ℕ, x + 100) →
  op 38 = 3800 → op = (λ x : ℕ, x * 100) :=
by
  intros op h1 h2
  have h_mult : (λ x : ℕ, x * 100) 38 = 3800 := rfl
  sorry

end operation_38_to_3800_l318_318142


namespace highest_power_of_2_dividing_15_to_6_minus_9_to_6_l318_318198

theorem highest_power_of_2_dividing_15_to_6_minus_9_to_6 : ∃ k: ℕ, 2 ^ k ∣ ( 15^6 - 9^6 ) ∧ 2^k = 16 :=
begin
  sorry
end

end highest_power_of_2_dividing_15_to_6_minus_9_to_6_l318_318198


namespace proof_problem_l318_318901

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318901


namespace function_inequality_l318_318033

theorem function_inequality 
  (f : ℝ → ℝ) 
  (h₀ : ∀ x, x ∈ set.Icc 0 1 → f(x) = f(mod_floor x 1)) 
  (h₁ : ∀ x₁ x₂, x₁ ≠ x₂ → x₁ ∈ set.Icc 0 1 → x₂ ∈ set.Icc 0 1 → |f(x₂) - f(x₁)| < |x₁ - x₂|) : 
  ∀ x₁ x₂, x₁ ∈ set.Icc 0 1 → x₂ ∈ set.Icc 0 1 → |f(x₂) - f(x₁)| < 1 / 2 :=
by
  sorry

end function_inequality_l318_318033


namespace cos_squared_difference_l318_318920

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318920


namespace sin_neg_600_eq_sqrt3_div_2_l318_318414

theorem sin_neg_600_eq_sqrt3_div_2 :
  sin (-600 * (π / 180)) = sqrt 3 / 2 :=
by
  -- Proof that satisfies the condition and demonstrates the periodicity and the given angle properties.
  sorry

end sin_neg_600_eq_sqrt3_div_2_l318_318414


namespace area_rectangle_ABCD_l318_318319

-- Definitions
variables {A B C D E F : Point}
variables [rectangle ABCD]

-- Given Conditions
variables (h1 : AE ⊥ BD) (h2 : CF ⊥ BD)
variables (hE : E = foot_perpendicular BE BD) (hF : F = foot_perpendicular CF BD)
variables (hBE : BE = 1) (hEF : EF = 2)

-- Theorem Statement
theorem area_rectangle_ABCD : 
  area_rectangle ABCD = 4 * √3 :=
sorry

end area_rectangle_ABCD_l318_318319


namespace fundamental_period_of_sine_product_l318_318541

noncomputable def lcm (a b : ℝ) : ℝ := abs (a * b) / gcd (nat_abs a) (nat_abs b)

theorem fundamental_period_of_sine_product : 
  ∀ (f : ℝ → ℝ), 
  f = (λ x : ℝ, sin x * sin (2 * x) * sin (3 * x) * sin (4 * x)) → 
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = 2 * π :=
begin
  intro f,
  assume h,
  use 2 * π,
  split,
  { norm_num, },
  { split,
    { intro x,
      rw h,
      sorry, },
    { refl } }
end

end fundamental_period_of_sine_product_l318_318541


namespace product_base_8_units_digit_l318_318503

theorem product_base_8_units_digit :
  let sum := 324 + 73
  let product := sum * 27
  product % 8 = 7 :=
by
  let sum := 324 + 73
  let product := sum * 27
  have h : product % 8 = 7 := by
    sorry
  exact h

end product_base_8_units_digit_l318_318503


namespace map_distance_l318_318562

theorem map_distance (scale : ℝ) (d_actual_km : ℝ) (d_actual_m : ℝ) (d_actual_cm : ℝ) (d_map : ℝ) :
  scale = 1 / 250000 →
  d_actual_km = 5 →
  d_actual_m = d_actual_km * 1000 →
  d_actual_cm = d_actual_m * 100 →
  d_map = (1 * d_actual_cm) / (1 / scale) →
  d_map = 2 :=
by sorry

end map_distance_l318_318562


namespace cosine_difference_identity_l318_318840

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318840


namespace cosine_difference_identity_l318_318851

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318851


namespace min_value_sum_of_cubes_complex_l318_318583

theorem min_value_sum_of_cubes_complex (z1 z2 : ℂ)
    (h1 : |z1 + z2| = 20)
    (h2 : |z1^2 + z2^2| = 16) :
    |z1^3 + z2^3| = 3520 := sorry

end min_value_sum_of_cubes_complex_l318_318583


namespace systematic_sampling_number_8th_group_stratified_sampling_under_40_l318_318312

theorem systematic_sampling_number_8th_group 
  (num_employees : ℕ)
  (sample_size : ℕ)
  (starting_number : ℕ)
  (interval : ℕ)
  (group_5_number : ℕ) :
  (num_employees = 200) →
  (sample_size = 40) →
  (starting_number = 22) →
  (interval = 5) →
  (group_5_number = 5) →
  starting_number + (interval * 3) = 37 :=
by
  intros h1 h2 h3 h4 h5
  sorry

theorem stratified_sampling_under_40 
  (num_employees : ℕ)
  (sample_size : ℕ)
  (under_40_percentage : ℕ)
  (total_employees : ℕ)
  (age_group : string) :
  (num_employees = 200) →
  (sample_size = 40) →
  (under_40_percentage = 50) →
  (age_group = "under_40") →
  sample_size * (under_40_percentage / 100) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end systematic_sampling_number_8th_group_stratified_sampling_under_40_l318_318312


namespace secret_spread_reaches_3280_on_saturday_l318_318694

theorem secret_spread_reaches_3280_on_saturday :
  (∃ n : ℕ, 4 * ( 3^n - 1) / 2 + 1 = 3280 ) ∧ n = 7  :=
sorry

end secret_spread_reaches_3280_on_saturday_l318_318694


namespace min_value_l318_318735

theorem min_value (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by sorry

end min_value_l318_318735


namespace counterfeit_coin_identification_possible_l318_318144

noncomputable def identify_fake_coins (coins : Fin 9 → ℕ) : Prop :=
  -- 9 coins represented as coins(0) to coins(8)
  -- Genuine coin weighs 10 g, counterfeit coin weighs 11 g
  (∀ i : Fin 9, coins i = 10 ∨ coins i = 11) ∧
  -- There are exactly 2 counterfeit coins
  (∃ a b : Fin 9, a ≠ b ∧ coins a = 11 ∧ coins b = 11 ∧ ∀ c : Fin 9, c ≠ a → c ≠ b → coins c = 10) →
  -- We can identify the counterfeit coins in at most 5 weighings
  ∃ find_countefeit : (Fin 9 → ℕ) → list (Fin 9 × Fin 9),
  find_countefeit coins = [(a, b)] ∧ (find_countefeit coins).length ≤ 5

theorem counterfeit_coin_identification_possible :
  ∀ (coins : Fin 9 → ℕ), identify_fake_coins coins := sorry

end counterfeit_coin_identification_possible_l318_318144


namespace janet_freelancer_income_difference_l318_318331

theorem janet_freelancer_income_difference :
  let hours_per_week := 40
  let current_job_hourly_rate := 30
  let freelancer_hourly_rate := 40
  let fica_taxes_per_week := 25
  let healthcare_premiums_per_month := 400
  let weeks_per_month := 4
  
  let current_job_weekly_income := hours_per_week * current_job_hourly_rate
  let current_job_monthly_income := current_job_weekly_income * weeks_per_month
  
  let freelancer_weekly_income := hours_per_week * freelancer_hourly_rate
  let freelancer_monthly_income := freelancer_weekly_income * weeks_per_month
  
  let freelancer_monthly_fica_taxes := fica_taxes_per_week * weeks_per_month
  let freelancer_total_additional_costs := freelancer_monthly_fica_taxes + healthcare_premiums_per_month
  
  let freelancer_net_monthly_income := freelancer_monthly_income - freelancer_total_additional_costs
  
  freelancer_net_monthly_income - current_job_monthly_income = 1100 :=
by
  sorry

end janet_freelancer_income_difference_l318_318331


namespace num_good_n_values_l318_318514

theorem num_good_n_values : 
  (∃ n_values: set ℤ, 
    ∀ n ∈ n_values, 
      8000 * (2/5)^n ∈ ℤ ∧ 
      8000 = 2^6 * 5^3 ∧
      n_values.count = 10) := 
by
  sorry

end num_good_n_values_l318_318514


namespace cos_difference_squared_l318_318962

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318962


namespace proof_problem_l318_318902

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318902


namespace fg_of_3_is_2810_l318_318258

def f (x : ℕ) : ℕ := x^2 + 1
def g (x : ℕ) : ℕ := 2 * x^3 - 1

theorem fg_of_3_is_2810 : f (g 3) = 2810 := by
  sorry

end fg_of_3_is_2810_l318_318258


namespace solve_fraction_equation_l318_318004

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ x = -2 := by
sorry

end solve_fraction_equation_l318_318004


namespace radian_to_degree_equivalent_l318_318168

theorem radian_to_degree_equivalent : 
  (7 / 12) * (180 : ℝ) = 105 :=
by
  sorry

end radian_to_degree_equivalent_l318_318168


namespace cos_squared_difference_l318_318833

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318833


namespace total_students_in_class_l318_318418

theorem total_students_in_class : 
  ∀ (total_candies students_candies : ℕ), 
    total_candies = 901 → students_candies = 53 → 
    students_candies * (total_candies / students_candies) = total_candies ∧ 
    total_candies % students_candies = 0 → 
    total_candies / students_candies = 17 := 
by 
  sorry

end total_students_in_class_l318_318418


namespace BC_length_105_l318_318304

variable (A B C X : Type)
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space X]
variable (dist : A → B → ℝ)  -- This represents distance function

-- Given the conditions
def triangle_ABC (AB AC : ℝ) (h_AB : dist A B = 90) (h_AC : dist A C = 105)
    (h_circle : dist A X = 90) (int_lengths : ∃ (BX CX : ℕ), dist B X = BX ∧ dist C X = CX) : Prop :=
-- Prove that BC equals 105
dist B C = 105

-- The main theorem stating that given the above conditions, BC = 105
theorem BC_length_105 
    (A B C X : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space X] 
    (dist : A → B → ℝ)
    (h_AB : dist A B = 90) (h_AC : dist A C = 105)
    (h_circle : dist A X = 90) (int_lengths : ∃ (BX CX : ℕ), dist B X = BX ∧ dist C X = CX) :
    dist B C = 105 := 
sorry

end BC_length_105_l318_318304


namespace daily_calories_burned_l318_318647

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def days : ℕ := 35
def total_calories := pounds_to_lose * calories_per_pound

theorem daily_calories_burned :
  (total_calories / days) = 500 := 
  by 
    -- calculation steps
    sorry

end daily_calories_burned_l318_318647


namespace top_field_value_is_nine_l318_318166

open Real

variables {a b c d e f g h i s : ℝ}

theorem top_field_value_is_nine 
  (h1 : a + b + f = s)
  (h2 : a + c + g = s)
  (h3 : a + d + h = s)
  (h4 : a + e + i = s)
  (h5 : b + c + d + e = s)
  (h6 : f + g + h + i = s)
  (h7 : a + b + c + d + e + f + g + h + i = 45)
  : a = 9 :=
begin
  sorry
end

end top_field_value_is_nine_l318_318166


namespace largest_square_in_square_with_triangles_l318_318328

noncomputable def largest_inscribed_square_side_length : ℝ :=
  (15 - 5 * real.sqrt 3) / 3

theorem largest_square_in_square_with_triangles
  (s : ℝ)
  (side_length_of_squares : s = 15)
  (side_length_of_triangles : ∀ t : ℝ, t = let x := (15 * real.sqrt 2) / (2 * real.sqrt 3) in 2 * x)
  (largest_square : ∀ l : ℝ, l = (15 - 5 * real.sqrt 3) / 3) :
  largest_inscribed_square_side_length = (15 - 5 * real.sqrt 3) / 3 := 
by
  sorry

end largest_square_in_square_with_triangles_l318_318328


namespace arrangement_schemes_correct_l318_318176

-- Given conditions
def number_of_teachers : ℕ := 2
def number_of_students : ℕ := 4
def teachers_group_ways : ℕ := 2.choose 1
def students_groups_ways : ℕ := 4.choose 2

-- Total arrangement schemes
def total_arrangement_schemes : ℕ :=
  teachers_group_ways * students_groups_ways

-- Theorem that proves the computation
theorem arrangement_schemes_correct :
  total_arrangement_schemes = 12 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end arrangement_schemes_correct_l318_318176


namespace probability_one_piece_is_two_probability_both_pieces_longer_than_two_l318_318478

theorem probability_one_piece_is_two (l1 l2 : ℕ) (h_pos : l1 > 0 ∧ l2 > 0) 
    (h_sum : l1 + l2 = 6) (h_cases : {l1, l2} ⊆ {1,2,3,4,5}) :
    (1/5 : ℚ) = 2/5 :=
by
    sorry

theorem probability_both_pieces_longer_than_two (l1 l2 : ℕ) (h_pos : l1 > 0 ∧ l2 > 0) 
    (h_sum : l1 + l2 = 6) (h_cases : {l1, l2} ⊆ {1,2,3,4,5}) :
    (1/3 : ℚ) = 2/6 :=
by
    sorry

end probability_one_piece_is_two_probability_both_pieces_longer_than_two_l318_318478


namespace brenda_blisters_l318_318155

theorem brenda_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) (arms : ℕ) :
  blisters_per_arm = 60 → blisters_rest = 80 → arms = 2 → 
  blisters_per_arm * arms + blisters_rest = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end brenda_blisters_l318_318155


namespace cos_squared_difference_l318_318919

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318919


namespace fraction_option_c_l318_318785

def is_fraction (num denom : ℚ) : Prop :=
  ∃ (a b : ℚ), num = a ∧ denom = b ∧ b ≠ 0

theorem fraction_option_c (x : ℚ) (num := 5) (denom := x + 3) :
  denom ≠ 0 → is_fraction num denom :=
begin
  intro h,
  unfold is_fraction,
  use [5, x + 3],
  split,
  { refl },
  split,
  { refl },
  { exact h }
end

end fraction_option_c_l318_318785


namespace quadrilateral_is_parallelogram_l318_318633

-- Define the conditions that are present in the original problem
axiom convex_quadrilateral (A B C D: Type) : Prop
axiom distance_sum_equal (A B C D: Type) : Prop

-- The goal is to prove that under the given conditions, the figure is a parallelogram
theorem quadrilateral_is_parallelogram (A B C D: Type) 
  (convex_quad : convex_quadrilateral A B C D)
  (sum_dist_eq : distance_sum_equal A B C D) : 
  is_parallelogram A B C D :=
sorry

end quadrilateral_is_parallelogram_l318_318633


namespace brokerage_percentage_correct_l318_318719

def cash_realized : ℝ := 106.25
def total_amount_including_brokerage : ℝ := 106
def brokerage_amount : ℝ := cash_realized - total_amount_including_brokerage
def percentage_of_brokerage : ℝ := (brokerage_amount / total_amount_including_brokerage) * 100

theorem brokerage_percentage_correct : percentage_of_brokerage ≈ 0.236 := by
  sorry

end brokerage_percentage_correct_l318_318719


namespace distance_AB_13_distance_AB_y_parallel_l318_318706

/-- Proof Problem 1: Distance between points A(2, 4) and B(-3, -8) is 13 -/
theorem distance_AB_13 (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = 4) (h3 : x2 = -3) (h4 : y2 = -8) : 
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 13 :=
by
  rw [h1, h2, h3, h4]
  sorry

/-- Proof Problem 2: Distance between points with coordinates (a, 5) and (b, -1) lying on a line parallel to the y-axis is 6 -/
theorem distance_AB_y_parallel (yA yB : ℝ) (hA : yA = 5) (hB : yB = -1) : 
  real.abs (yA - yB) = 6 :=
by
  rw [hA, hB]
  sorry

end distance_AB_13_distance_AB_y_parallel_l318_318706


namespace geometry_problem_l318_318064

-- Definitions
def circles_intersect (K₁ K₂ : set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ K₁ ∧ A ∈ K₂ ∧ B ∈ K₁ ∧ B ∈ K₂ ∧ A ≠ B

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Conditions
variables {K₁ K₂ : set (ℝ × ℝ)}
variables {A B C D E F P : ℝ × ℝ}
variables {l₁ l₂ : set (ℝ × ℝ)}

-- Problem Statement
theorem geometry_problem
  (h1 : circles_intersect K₁ K₂ A B)
  (h2 : A = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  (h3 : C ∈ K₁)
  (h4 : D ∈ K₂)
  (h5 : ∃ E', E' ∈ K₁ ∧ E' ≠ D ∧ ∃ t, E' = D + t * (B - D) ∧ E = E')
  (h6 : ∃ F', F' ∈ K₂ ∧ F' ≠ C ∧ ∃ t, F' = C + t * (B - C) ∧ F = F')
  (h7 : ∀ M,
    (M ∈ l₁ ↔ ((M.1 - C.1)^2 + (M.2 - C.2)^2 = (M.1 - D.1)^2 + (M.2 - D.2)^2)) ∧
    (M ∈ l₂ ↔ ((M.1 - E.1)^2 + (M.2 - E.2)^2 = (M.1 - F.1)^2 + (M.2 - F.2)^2))) :
  (∃! P, P ∈ l₁ ∧ P ∈ l₂) ∧
  ((C.1 - A.1)^2 + (C.2 - A.2)^2 + (A.1 - P.1)^2 + (A.2 - P.2)^2 = (P.1 - E.1)^2 + (P.2 - E.2)^2) → 
  ((C.1 - A.1)^2 + (C.2 - A.2)^2 + (A.1 - P.1)^2 + (A.2 - P.2)^2 = (P.1 - E.1)^2)
  sorry

end geometry_problem_l318_318064


namespace rotation_preserves_measure_l318_318733

-- Define the initial angle before rotation and the result after rotation
variables (initial_angle : ℝ) (rotation : ℝ)

-- Defining the condition
def initial_measure (initial_angle : ℝ) : Prop := initial_angle = 30

-- The final angle after rotation
def final_measure (initial_angle rotation : ℝ) : ℝ := 
  let reduced_rotation := rotation - 360 * (rotation / 360).to_int
  in (initial_angle + reduced_rotation) % 360

-- The problem statement
theorem rotation_preserves_measure : 
  initial_measure initial_angle → 
  final_measure initial_angle rotation = 120 :=
by
  intros h
  -- Skipping the actual proof steps with sorry
  sorry

end rotation_preserves_measure_l318_318733


namespace isosceles_triangle_congruent_l318_318413

theorem isosceles_triangle_congruent (A B C C1 : ℝ) 
(h₁ : A = B) 
(h₂ : C = C1) 
: A = B ∧ C = C1 :=
by
  sorry

end isosceles_triangle_congruent_l318_318413


namespace sprinted_further_than_jogged_l318_318512

def sprint_distance1 := 0.8932
def sprint_distance2 := 0.7773
def sprint_distance3 := 0.9539
def sprint_distance4 := 0.5417
def sprint_distance5 := 0.6843

def jog_distance1 := 0.7683
def jog_distance2 := 0.4231
def jog_distance3 := 0.5733
def jog_distance4 := 0.625
def jog_distance5 := 0.6549

def total_sprint_distance := sprint_distance1 + sprint_distance2 + sprint_distance3 + sprint_distance4 + sprint_distance5
def total_jog_distance := jog_distance1 + jog_distance2 + jog_distance3 + jog_distance4 + jog_distance5

theorem sprinted_further_than_jogged :
  total_sprint_distance - total_jog_distance = 0.8058 :=
by
  sorry

end sprinted_further_than_jogged_l318_318512


namespace min_value_max_interval_domain_inequality_l318_318256

-- Problem (1)
theorem min_value (a : ℝ) (ha : 0 < a) (hmin : ∀ x ∈ Ioo (0 : ℝ) a, f x < 4) : a = 4 :=
sorry

-- Problem (2)
theorem max_interval (A : set ℝ) (hA_range : range f ∩ A = set.Icc 4 5): A = set.Icc 1 4 :=
sorry

-- Problem (3)
theorem domain_inequality (a : ℝ) (ha : 2 ≤ a) : f (a^2 - a) ≥ f (2a + 4) ↔ a ≥ 4 ∨ a = -1 :=
sorry

end min_value_max_interval_domain_inequality_l318_318256


namespace maximum_positive_factors_l318_318049

theorem maximum_positive_factors (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 15) :
  ∃ k, (k = b^n) ∧ (∀ m, m = b^n → m.factors.count ≤ 61) :=
sorry

end maximum_positive_factors_l318_318049


namespace perpendicular_vectors_implication_l318_318212

variable (m : ℝ)

def a : (ℝ × ℝ) := (1, -2)
def b : (ℝ × ℝ) := (m, m + 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_vectors_implication (h : dot_product a b = 0) : m = -4 := 
sorry

end perpendicular_vectors_implication_l318_318212


namespace monotonicity_of_f_range_of_a_if_f_ge_zero_l318_318250

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (Real.exp x - a) - (a ^ 2) * x

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂

def is_monotonic_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

theorem monotonicity_of_f (a : ℝ) : 
  (a = 0 → is_monotonic_increasing (f _ a)) ∧
  (a > 0 → 
    (∀ x, x < Real.log a → is_monotonic_decreasing (f _ a)) ∧ 
    (∀ x, x > Real.log a → is_monotonic_increasing (f _ a))) ∧ 
  (a < 0 → 
    (∀ x, x < Real.log (-a / 2) → is_monotonic_decreasing (f _ a)) ∧ 
    (∀ x, x > Real.log (-a / 2) → is_monotonic_increasing (f _ a))) :=
sorry

theorem range_of_a_if_f_ge_zero : 
  (∀ x a, f x a ≥ 0) → (-2 * Real.exp (3 / 4)) ≤ a ∧ a ≤ 1 :=
sorry

end monotonicity_of_f_range_of_a_if_f_ge_zero_l318_318250


namespace n_le_100_diagonals_l318_318178

theorem n_le_100_diagonals {n : ℕ} 
  (h : ∀ v, v ∈ set.range (λ x, x^2) → v.succ ≤ 100) :
  n ≤ 100 :=
sorry

end n_le_100_diagonals_l318_318178


namespace place_cards_l318_318008

noncomputable def card_set(n : ℕ) : Type := 
  {cards : list (ℕ × ℕ) // ∀ i ∈ cards, (i.fst ∈ list.range (n + 1)) ∧ (i.snd ∈ list.range (n + 1)) ∧
  (list.count (λ c => c.fst = i.fst) cards = 2) ∧ (list.count (λ c => c.snd = i.snd) cards = 2)}

theorem place_cards (n : ℕ) (cards : card_set n):
  ∃ placed_cards : list (ℕ × bool), ∀ i ∈ list.range (n + 1), i ∈ list.map prod.fst (list.filter (λ c : ℕ × bool, c.snd = tt) placed_cards) :=
sorry

end place_cards_l318_318008


namespace total_notes_proof_l318_318999

variable (x : Nat)

def total_money := 10350
def fifty_notes_count := 17
def fifty_notes_value := 850  -- 17 * 50
def five_hundred_notes_value := 500 * x
def total_value_proposition := fifty_notes_value + five_hundred_notes_value = total_money

theorem total_notes_proof :
  total_value_proposition -> (fifty_notes_count + x) = 36 :=
by
  intros h
  -- The proof steps would go here, but we use sorry for now.
  sorry

end total_notes_proof_l318_318999


namespace cevian_concurrency_l318_318985

noncomputable def is_concurrent (A B C M1 N1 K1 : Point) : Prop := sorry

theorem cevian_concurrency
(A B C M N K O' M1 N1 K1 : Point)
(h1 : CircleInscribedInTriangle A B C M N K)
(h2 : IsMidpoint M1 (Segment O' M))
(h3 : IsMidpoint N1 (Segment O' N))
(h4 : IsMidpoint K1 (Segment O' K))
(h5 : ConnectedTo C M1)
(h6 : ConnectedTo A N1)
(h7 : ConnectedTo B K1) :
is_concurrent A B C M1 N1 K1 :=
sorry

end cevian_concurrency_l318_318985


namespace snow_fall_time_l318_318618

theorem snow_fall_time (rate_mm_per_minute : ℕ) (time_minute : ℕ) (time_hour : ℕ) (meter_to_cm : ℕ) (cm_to_mm : ℕ) : 
(rate_mm_per_minute = 1 / 6) → 
(time_minute = 60) → 
(meter_to_cm = 100) → 
(cm_to_mm = 10) →
let rate_mm_per_hour := rate_mm_per_minute * time_minute in
let rate_cm_per_hour := rate_mm_per_hour / cm_to_mm in
let time_required_for_1m := meter_to_cm / rate_cm_per_hour in
time_required_for_1m = 100 := 
by
  intros rate_cond time_cond meter_cond cm_cond
  rw [rate_cond, time_cond, meter_cond, cm_cond]
  simp [rate_mm_per_hour, rate_cm_per_hour, time_required_for_1m]
  sorry

end snow_fall_time_l318_318618


namespace cos_squared_difference_l318_318945

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318945


namespace cook_one_potato_l318_318455

theorem cook_one_potato (total_potatoes cooked_potatoes remaining_potatoes remaining_time : ℕ) 
  (h1 : total_potatoes = 15) 
  (h2 : cooked_potatoes = 6) 
  (h3 : remaining_time = 72)
  (h4 : remaining_potatoes = total_potatoes - cooked_potatoes) :
  (remaining_time / remaining_potatoes) = 8 :=
by
  sorry

end cook_one_potato_l318_318455


namespace average_speed_of_pinedale_bus_line_l318_318009

-- Define the conditions
def stopping_time : ℕ := 5 -- minutes per stop
def num_stops : ℕ := 5
def distance : ℕ := 25 -- kilometers

-- Prove that the average speed is 27.27 km/h
theorem average_speed_of_pinedale_bus_line :
  let total_interval := num_stops + 1 in -- extra intervals of travel
  let total_travel_time_min := stopping_time * num_stops + stopping_time * total_interval in
  let total_travel_time_hr := (total_travel_time_min : ℝ) / 60 in
  let average_speed := (distance : ℝ) / total_travel_time_hr in
  average_speed = 300 / 11 :=
by
  sorry

end average_speed_of_pinedale_bus_line_l318_318009


namespace exists_good_polynomials_l318_318067

def good_polynomial (p : ℝ[X, X]) : Prop :=
  ∀ (x y : ℝ), y ≠ 0 → p.eval₂ x y = p.eval₂ (x * y) (1 / y)

theorem exists_good_polynomials :
  ∃ (r s : ℝ[X, X]), good_polynomial r ∧ good_polynomial s ∧
  ∀ (p : ℝ[X, X]) (hp : good_polynomial p), ∃ (f : ℝ[X, X]), f.eval₂ (r.eval₂ x y) (s.eval₂ x y) = p :=
by
  sorry

end exists_good_polynomials_l318_318067


namespace cos_squared_difference_l318_318925

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318925


namespace car_additional_mileage_after_modification_l318_318790

-- Definitions based on conditions

def miles_per_gallon : ℝ := 28
def fuel_efficiency_modifier : ℝ := 0.8
def fuel_tank_capacity : ℝ := 15

-- Proof problem statement
theorem car_additional_mileage_after_modification :
  let total_miles_before_mod := miles_per_gallon * fuel_tank_capacity,
      new_miles_per_gallon := miles_per_gallon / fuel_efficiency_modifier,
      total_miles_after_mod := new_miles_per_gallon * fuel_tank_capacity
  in 
    total_miles_after_mod - total_miles_before_mod = 84 := 
sorry

end car_additional_mileage_after_modification_l318_318790


namespace cos_difference_identity_l318_318942

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318942


namespace snow_fall_time_l318_318620

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end snow_fall_time_l318_318620


namespace maximize_ratio_l318_318675

-- Definitions of positive reals and their properties
variables (a b : ℝ)

-- Assumptions
variables (a_pos : 0 < a) (b_pos : 0 < b) (ab_condition : a > b ∧ b > a / 2)

-- Expression for the line passing through (0, a) and (a + b, 0)
def line_equation (x : ℝ) := a - (a / (a + b)) * x

-- Calculate the area above the line in both squares
-- Area above the line in the larger square
def area_larger_square := 1/2 * a * (a^2/(a+b))

-- Area above the line in the smaller square
def area_smaller_square := 1/2 * b * a

-- Total area above the line condition
variable (area_condition : area_larger_square + area_smaller_square = 2013)

-- Definition of the ratio t
def t := a / b

-- Assume (a, b) is the unique pair maximizing a + b
-- Thus, we need to show that t = a / b = 3^(1/5)
theorem maximize_ratio : t = real.sqrt (3^(1/5)) := sorry

end maximize_ratio_l318_318675


namespace triangle_perimeter_l318_318540

def point := ℝ × ℝ
def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perimeter (A B C : point) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  perimeter A B C = 16 := by
  -- define points
  let A : point := (2, 3)
  let B : point := (2, 9)
  let C : point := (6, 6)
  -- assert the statement and skip the proof using sorry
  show perimeter A B C = 16 from sorry

end triangle_perimeter_l318_318540


namespace range_of_a_l318_318543

noncomputable
def f (x a : ℝ) : ℝ := a - x - |Real.log x|

theorem range_of_a (h : ∀ x > 0, f x a ≤ 0) : 
  a ≤ Real.log (Real.exp 1) - Real.log (Real.log (Real.exp 1)) :=
begin
  sorry
end

end range_of_a_l318_318543


namespace cos_squared_difference_l318_318811

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318811


namespace cos_squared_difference_l318_318951

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318951


namespace xi_sin_xi_independent_iff_constant_l318_318774

noncomputable def xi_independence {Ω : Type*} [MeasureSpace Ω] 
    (xi : Ω → Real) (y : Fin n → Real) : Prop :=
∀ i j, 
  ((MeasureTheory.measureOf (λ ω, xi ω = y i ∧ Real.sin (xi ω) = Real.sin (y j)) > 0) = 
   (MeasureTheory.measureOf (λ ω, xi ω = y i) > 0) * (MeasureTheory.measureOf (λ ω, Real.sin (xi ω) = Real.sin (y j)) > 0))

theorem xi_sin_xi_independent_iff_constant {Ω : Type*} [MeasureSpace Ω] 
    (xi : Ω → Real) (y : Fin n → Real) :
  (∀ i, MeasureTheory.measureOf (λ ω, xi ω = y i) > 0) →
  (∀ i j, xi_independence xi y) ↔ ∃ c : Real, ∀ ω, Real.sin (xi ω) = c := 
sorry

end xi_sin_xi_independent_iff_constant_l318_318774


namespace cos_squared_difference_l318_318916

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318916


namespace probability_at_least_one_but_not_all_l318_318139

noncomputable def prob_A := 3 / 4
noncomputable def prob_B := 2 / 3
noncomputable def prob_C := 3 / 5

def prob_all := prob_A * prob_B * prob_C
def prob_none := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

def prob_at_least_one_but_not_all := 1 - prob_all - prob_none

theorem probability_at_least_one_but_not_all :
  prob_at_least_one_but_not_all = 1 / 3 :=
by
  sorry

end probability_at_least_one_but_not_all_l318_318139


namespace cos_squared_difference_l318_318827

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318827


namespace sqrt_six_ineq_l318_318788

theorem sqrt_six_ineq : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_six_ineq_l318_318788


namespace number_of_people_in_group_l318_318995

def total_dining_bill : ℝ := 139.00
def tip_percentage : ℝ := 0.10
def amount_each_person_paid : ℝ := 16.99

theorem number_of_people_in_group : 
  let total_bill_with_tip := total_dining_bill + (tip_percentage * total_dining_bill)
  let num_people := total_bill_with_tip / amount_each_person_paid
  num_people ≈ 9 :=
by
  have total_bill_with_tip := total_dining_bill + (tip_percentage * total_dining_bill)
  have num_people := total_bill_with_tip / amount_each_person_paid
  sorry

end number_of_people_in_group_l318_318995


namespace eight_digit_permutations_l318_318277

open Fin

-- Definition of the number of each digit occurrence
def digits_occurrences : List Nat := [2, 2, 2, 2]

-- Definition of the total number of digits
def total_digits : Nat := 8

-- The factorial function defined in Lean
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- The count of unique permutations considering repetitions of digits
def count_permutations : Nat :=
  factorial total_digits / (factorial 2 * factorial 2 * factorial 2 * factorial 2)

-- The main theorem: Proof that the number of different eight-digit integers is 2520
theorem eight_digit_permutations : count_permutations = 2520 :=
by
  sorry

end eight_digit_permutations_l318_318277


namespace complex_product_not_necessary_sufficient_l318_318558

theorem complex_product_not_necessary_sufficient (z1 z2 : ℂ) :
  (z1 * z2).im = 0 ↔ (z1 * z2 ∈ ℝ) ∧ (∀ (a b : ℝ), z1 = (a + b * complex.I) ∧ z2 = (a - b * complex.I) → (z1 * z2).im = 0) :=
by
  sorry

end complex_product_not_necessary_sufficient_l318_318558


namespace minimum_selling_price_l318_318792

theorem minimum_selling_price (units_per_month : ℕ) (cost_per_unit : ℕ) (desired_profit : ℕ) : 
  (units_per_month = 400) → 
  (cost_per_unit = 40) → 
  (desired_profit = 40000) → 
  let total_production_cost := cost_per_unit * units_per_month,
      total_revenue_required := total_production_cost + desired_profit,
      minimum_selling_price_per_unit := total_revenue_required / units_per_month
  in 
  minimum_selling_price_per_unit = 140 :=
by
  intros h_units h_cost h_profit
  let total_production_cost := 40 * 400
  let total_revenue_required := total_production_cost + 40000
  let minimum_selling_price_per_unit := total_revenue_required / 400
  have : minimum_selling_price_per_unit = 140 := sorry
  exact this

end minimum_selling_price_l318_318792


namespace affine_transformation_zero_vector_affine_transformation_additivity_affine_transformation_scalar_multiplication_l318_318697

variables {V W : Type*} [AddCommGroup V] [AddCommGroup W] [Module ℝ V] [Module ℝ W]
variable (L : V → W)
variable (H_affine : ∃ (f : V →ₗ[ℝ] W) (b : W), L = f + (λ _, b))

theorem affine_transformation_zero_vector (hL : ∃ (f : V →ₗ[ℝ] W) (b : W), L = f + (λ _, b)) :
  L 0 = 0 := by
  sorry

theorem affine_transformation_additivity (hL : ∃ (f : V →ₗ[ℝ] W) (b : W), L = f + (λ _, b))
  (a b : V) : L (a + b) = L a + L b := by
  sorry

theorem affine_transformation_scalar_multiplication (hL : ∃ (f : V →ₗ[ℝ] W) (b : W), L = f + (λ _, b)) 
  (a : V) (k : ℝ) : L (k • a) = k • L a := by
  sorry

end affine_transformation_zero_vector_affine_transformation_additivity_affine_transformation_scalar_multiplication_l318_318697


namespace problem_a_b_l318_318284

theorem problem_a_b (a b : ℝ) (h₁ : a + b = 10) (h₂ : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_a_b_l318_318284


namespace arithmetic_series_sum_l318_318389

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 50
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  S = 442 := by
  sorry

end arithmetic_series_sum_l318_318389


namespace lisa_phone_spending_l318_318686

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end lisa_phone_spending_l318_318686


namespace cos_squared_difference_l318_318954

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318954


namespace enchilada_taco_cost_l318_318701

theorem enchilada_taco_cost (e t : ℝ) 
  (h1 : 3 * e + 4 * t = 3.50) 
  (h2 : 4 * e + 3 * t = 3.90) : 
  4 * e + 5 * t = 4.56 := 
sorry

end enchilada_taco_cost_l318_318701


namespace complex_abs_value_l318_318580

theorem complex_abs_value :
  let i := Complex.I in
  abs ((3 - i) / (i + 2)) = Real.sqrt 2 := 
by
  sorry

end complex_abs_value_l318_318580


namespace green_sweets_count_l318_318758

def total_sweets := 285
def red_sweets := 49
def neither_red_nor_green_sweets := 177

theorem green_sweets_count : 
  (total_sweets - red_sweets - neither_red_nor_green_sweets) = 59 :=
by
  -- The proof will go here
  sorry

end green_sweets_count_l318_318758


namespace number_of_possible_values_of_a_l318_318382

theorem number_of_possible_values_of_a :
  ∃ a_count : ℕ, (∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2020 ∧ a^2 - b^2 + c^2 - d^2 = 2020 ∧ a_count = 501) :=
sorry

end number_of_possible_values_of_a_l318_318382


namespace even_sum_probability_l318_318772

-- Define the probabilities of even and odd outcomes for each wheel
def probability_even_first_wheel : ℚ := 2 / 3
def probability_odd_first_wheel : ℚ := 1 / 3
def probability_even_second_wheel : ℚ := 3 / 5
def probability_odd_second_wheel : ℚ := 2 / 5

-- Define the probabilities of the scenarios that result in an even sum
def probability_both_even : ℚ := probability_even_first_wheel * probability_even_second_wheel
def probability_both_odd : ℚ := probability_odd_first_wheel * probability_odd_second_wheel

-- Define the total probability of an even sum
def probability_even_sum : ℚ := probability_both_even + probability_both_odd

-- The theorem statement to be proven
theorem even_sum_probability :
  probability_even_sum = 8 / 15 :=
by
  sorry

end even_sum_probability_l318_318772


namespace dot_product_is_constant_l318_318228

open Real

noncomputable theory
def ellipse_eq (a x y : ℝ) : x^2 + 2*y^2 = a^2 := sorry

def triangle_area (a c : ℝ) : ℝ := 4

def equation_of_ellipse (x y: ℝ) : Prop := 
  (∃ a > 0, ellipse_eq a x y ∧ triangle_area a (sqrt (a^2 / 2)) = 4)

def intersects (k x y : ℝ) : Prop :=
  y = k * (x - 1) ∧ ellipse_eq (sqrt 8) x y

def point_M (x y : ℝ) : Prop :=
  x = 11 / 4 ∧ y = 0

def dot_product_constant (A B M : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  let ⟨xm, ym⟩ := M in
  (x1 - xm) * (x2 - xm) + y1 * y2 = -7/16

theorem dot_product_is_constant :
  ∀ k (x1 y1 x2 y2 : ℝ),
  (∃ a > 0, ellipse_eq a x1 y1) ∧
  (∃ a > 0, ellipse_eq a x2 y2) ∧
  intersects k x1 y1 ∧ intersects k x2 y2 ∧
  point_M (11 / 4) 0 →
  dot_product_constant (x1, y1) (x2, y2) (11 / 4, 0) := 
sorry

end dot_product_is_constant_l318_318228


namespace proportion_first_quarter_time_l318_318981

variable (D V : ℝ)

theorem proportion_first_quarter_time :
  let T1 := (D / 4) / (4 * V)
  let T2 := (D / 4) / V
  let T3 := (D / 4) / (V / 6)
  let T4 := (D / 4) / (V / 2)
  let T_total := T1 + T2 + T3 + T4 in
  (T1 / T_total) = (1 / 37) :=
by
  let T1 := (D / 4) / (4 * V)
  let T2 := (D / 4) / V
  let T3 := (D / 4) / (V / 6)
  let T4 := (D / 4) / (V / 2)
  let T_total := T1 + T2 + T3 + T4
  have h1 : T1 = D / (16 * V) := by sorry
  have h2 : T2 = D / (4 * V) := by sorry
  have h3 : T3 = (3 * D) / (2 * V) := by sorry
  have h4 : T4 = D / (2 * V) := by sorry
  have h_total : T_total = 37 * D / (16 * V) := by sorry
  show T1 / T_total = 1 / 37 from sorry

end proportion_first_quarter_time_l318_318981


namespace speed_man_l318_318302

noncomputable def speedOfMan : ℝ := 
  let d := 437.535 / 1000  -- distance in kilometers
  let t := 25 / 3600      -- time in hours
  d / t                    -- speed in kilometers per hour

theorem speed_man : speedOfMan = 63 := by
  sorry

end speed_man_l318_318302


namespace exists_infinitely_many_primes_dividing_fib_l318_318659

theorem exists_infinitely_many_primes_dividing_fib (u : ℕ → ℕ) (h0 : u 0 = 0) (h1 : u 1 = 1)
  (hn : ∀ n > 1, u n = u (n-1) + u (n-2)) :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p ∣ u (p - 1) := 
sorry

end exists_infinitely_many_primes_dividing_fib_l318_318659


namespace find_divisor_l318_318299

variable (r q d v : ℕ)
variable (h1 : r = 8)
variable (h2 : q = 43)
variable (h3 : d = 997)

theorem find_divisor : d = v * q + r → v = 23 :=
by
  sorry

end find_divisor_l318_318299


namespace cubes_with_paint_l318_318087

theorem cubes_with_paint (interior_cubes : ℕ) (unpainted_cubes : ℕ)
  (h1 : interior_cubes = 8) : 
  unpainted_cubes = 23 → 
  let larger_cube_side_length := 4 in
  let total_cubes := larger_cube_side_length ^ 3 in
  total_cubes - unpainted_cubes = 41 :=
by
  intro hunpainted
  rw hunpainted
  simp only [larger_cube_side_length, total_cubes, pow_succ]
  norm_num
  exact rfl

end cubes_with_paint_l318_318087


namespace concurrent_diagonals_l318_318764

structure Parallelogram (A B C D : Type) :=
  (AB : A → B → Prop)
  (AD : A → D → Prop)
  (BC : B → C → Prop)
  (CD : C → D → Prop)
  (parallelogram : AB ∧ CD ∧ AD ∧ BC)

structure PointOnDiagonal {A B D : Type} (O : Type) :=
  (on_diagonal : ∃ α β : ℝ, O = α * B + β * D)

structure ParallelSegments {A B D M N P Q : Type} :=
  (MN_parallel_AB : M ∥ N)
  (PQ_parallel_AD : P ∥ Q)
  (M_on_AD : M = (scalar : ℝ) * AD)
  (Q_on_AB : Q = (scalar : ℝ) * AB)

theorem concurrent_diagonals 
  {A B C D M N P Q O : Type}
  [Parallelogram A B C D] 
  [PointOnDiagonal O] 
  [ParallelSegments A B D M N P Q] : 
  ∃ k : ℝ, (AO ∥ k * AO) := 
sorry

end concurrent_diagonals_l318_318764


namespace initial_students_count_l318_318014

theorem initial_students_count (N T : ℕ) (h1 : T = N * 90) (h2 : (T - 120) / (N - 3) = 95) : N = 33 :=
by
  sorry

end initial_students_count_l318_318014


namespace horner_rule_V3_l318_318775

def poly : ℤ → ℤ := λ x, x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

theorem horner_rule_V3 :
  let V0 := 1,
      V1 := (-2) - 5 * V0,
      V2 := (-2) * V1 + 6,
      V3 := (-2) * V2 + 0
  in V3 = -40 :=
by
  sorry

end horner_rule_V3_l318_318775


namespace percentage_increase_l318_318626

-- Definitions

def original_price (P : ℝ) : ℝ := P

def decreased_price (P : ℝ) : ℝ := 0.80 * P

def increased_price (P : ℝ) (x : ℝ) : ℝ := 0.80 * P * (1 + x / 100)

def final_price (P : ℝ) : ℝ := 1.12 * P

-- Theorem Statement
theorem percentage_increase (P x : ℝ) : (increased_price P x = final_price P) ↔ x = 40 := by
  sorry

end percentage_increase_l318_318626


namespace final_composition_is_approx_83_193_percent_water_l318_318549

-- Define the initial conditions
def initial_milk_volume (V : ℝ) : ℝ := V
def replacement_ratio : ℝ := 0.7
def operations : ℕ := 5

-- Define the volume of milk after n operations
def milk_volume_after_n_operations (V : ℝ) (n : ℕ) : ℝ :=
V * (replacement_ratio ^ n)

-- Define the percentage of milk after n operations
def percentage_of_milk_after_n_operations (V : ℝ) (n : ℕ) : ℝ :=
(milk_volume_after_n_operations V n / V) * 100

-- Define the final operation and answer
def final_percentage_of_water (V : ℝ) (n : ℕ) : ℝ :=
100 - percentage_of_milk_after_n_operations V n

theorem final_composition_is_approx_83_193_percent_water (V : ℝ) :
  final_percentage_of_water V operations ≈ 83.193 :=
by
  -- Skipping the proof steps
  sorry

end final_composition_is_approx_83_193_percent_water_l318_318549


namespace find_k_l318_318230

theorem find_k (a b : ℕ) (k : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a^2 + b^2) = k * (a * b - 1)) :
  k = 5 :=
sorry

end find_k_l318_318230


namespace cos_squared_difference_l318_318923

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318923


namespace distance_from_starting_point_total_turnover_l318_318083

def movements := [+5, -3, -5, +4, -8, +6, -4]
def charge_per_km : ℝ := 2

theorem distance_from_starting_point :
  (List.sum movements) = -5 :=
by
  sorry

theorem total_turnover :
  (List.sum (List.map abs movements) * charge_per_km) = 70 :=
by
  sorry

end distance_from_starting_point_total_turnover_l318_318083


namespace total_original_cost_l318_318131

theorem total_original_cost (discounted_price1 discounted_price2 discounted_price3 : ℕ) 
  (discount_rate1 discount_rate2 discount_rate3 : ℚ)
  (h1 : discounted_price1 = 4400)
  (h2 : discount_rate1 = 0.56)
  (h3 : discounted_price2 = 3900)
  (h4 : discount_rate2 = 0.35)
  (h5 : discounted_price3 = 2400)
  (h6 : discount_rate3 = 0.20) :
  (discounted_price1 / (1 - discount_rate1) + discounted_price2 / (1 - discount_rate2) 
    + discounted_price3 / (1 - discount_rate3) = 19000) :=
by
  sorry

end total_original_cost_l318_318131


namespace AO_eq_OC_l318_318407

/-
Given a parallelogram ABCD such that:
1. P lies on the side CD of the parallelogram.
2. ∠DBA = ∠CBP.
3. O is the center of the circle passing through D and P and tangent to the line AD at D.

Prove that AO = OC.
-/

theorem AO_eq_OC (A B C D P O : Point)
  (parallelogram : Parallelogram A B C D)
  (P_on_CD : OnLineSegment P C D)
  (angle_eq : Angle D B A = Angle C B P)
  (O_center : CircleTangent D P AD O)
  : dist A O = dist O C :=
sorry

end AO_eq_OC_l318_318407


namespace bug_at_opposite_vertex_l318_318104

-- Definitions from conditions
def cube_vertices : Finset (Fin 8) := 
  {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent_edges (v : Fin 8) : Finset (Fin 8) :=
  match v with
  | 0 => {1, 2, 4}
  | 1 => {0, 3, 5}
  | 2 => {0, 3, 6}
  | 3 => {1, 2, 7}
  | 4 => {0, 5, 6}
  | 5 => {1, 4, 7}
  | 6 => {2, 4, 7}
  | 7 => {3, 5, 6}
  | _ => ∅

-- Proposition to prove
theorem bug_at_opposite_vertex :
  (∃ (count_paths : ℕ), count_paths = 60) ∧
  let total_moves := 729 in
  (count_paths / total_moves : ℚ) = 20 / 243 := 
by 
  sorry

end bug_at_opposite_vertex_l318_318104


namespace elroy_more_miles_l318_318187

-- Given conditions
def last_year_rate : ℝ := 4
def this_year_rate : ℝ := 2.75
def last_year_collection : ℝ := 44

-- Goals
def last_year_miles : ℝ := last_year_collection / last_year_rate
def this_year_miles : ℝ := last_year_collection / this_year_rate
def miles_difference : ℝ := this_year_miles - last_year_miles

theorem elroy_more_miles :
  miles_difference = 5 := by
  sorry

end elroy_more_miles_l318_318187


namespace lisa_total_cost_l318_318687

def c_phone := 1000
def c_contract_per_month := 200
def c_case := 0.20 * c_phone
def c_headphones := 0.5 * c_case
def t_year := 12

theorem lisa_total_cost :
  c_phone + (c_case) + (c_headphones) + (c_contract_per_month * t_year) = 3700 :=
by
  sorry

end lisa_total_cost_l318_318687


namespace range_of_m_l318_318438

theorem range_of_m (a c m : ℝ) (h_mono : ∀ x ∈ Icc 0 1, 2 * a * (x - 1) < 0)
  (h_le : a ≠ 0 ∧ f m ≤ f 0) :
  0 ≤ m ∧ m ≤ 2 :=
sorry

where { f : ℝ → ℝ := λ x, a * x^2 - 2 * a * x + c }


end range_of_m_l318_318438


namespace find_d_l318_318347

-- Definitions for the functions f and g
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

-- Statement to prove the value of d
theorem find_d (c d : ℝ) (h1 : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  -- inserting custom logic for proof
  sorry

end find_d_l318_318347


namespace find_m_value_l318_318595

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem find_m_value :
  ∃ m : ℝ, (∀ x ∈ (Set.Icc 0 3), f x m ≤ 1) ∧ (∃ x ∈ (Set.Icc 0 3), f x m = 1) ↔ m = -2 :=
by
  sorry

end find_m_value_l318_318595


namespace middle_term_expansion_l318_318592

theorem middle_term_expansion (n a : ℕ) (h1 : n > a) (h2 : 1 + a^n = 65) : 
  ∃ (t : ℕ), t = (nat.choose n (n / 2)) * (2 ^ (n / 2)) ∧ t = 160 :=
by
  have ha_pos : 0 < a := sorry, -- a > 0 as a ∈ ℕ* 
  have hn_pos : 0 < n := sorry, -- n > 0 as n ∈ ℕ*
  have ha_n_pos : 0 < a^n := sorry, -- a ^ n > 0 for positive a and n
  have := a^n,
  have h65 : 1 + a^n = 65 := h2,
  have h_an : a^n = 64 := sorry, -- derive a^n from the equation
  have h_a : a = 2 := sorry,  -- find a
  have h_n : n = 6 := sorry,  -- find n
    
  use (nat.choose 6 3) * (2 ^ 3),  -- middle term computation when n = 6 and a = 2
  have h_middle_term : (nat.choose 6 3) * (2 ^ 3) = 160 := sorry,
  exact ⟨h_middle_term, rfl⟩

end middle_term_expansion_l318_318592


namespace find_solution_l318_318524

theorem find_solution :
  ∃ x y z : ℝ, (x + y = 3 * x + 4) ∧ (2 * y + 3 + z = 6 * y + 6) ∧ (3 * z + 3 + x = 9 * z + 8) ∧ (x = 2) ∧ (y = 2) ∧ (z = 2) :=
by
  exists 2
  exists 2
  exists 2
  split
  calc 2 + 2 = 4 : by ring
  calc 3 * 2 + 4 = 10 : by ring
  calc 2 * 2 + 3 + 2 = 9 : by ring
  calc 6 * 2 + 6 = 18 : by ring
-- Additional verifications can be included by using additional sorry placeholders
  sorry

end find_solution_l318_318524


namespace complex_quadratic_solution_l318_318519

theorem complex_quadratic_solution (a b : ℝ) (h₁ : ∀ (x : ℂ), 5 * x ^ 2 - 4 * x + 20 = 0 → x = a + b * Complex.I ∨ x = a - b * Complex.I) :
 a + b ^ 2 = 394 / 25 := 
sorry

end complex_quadratic_solution_l318_318519


namespace fraction_value_l318_318415

theorem fraction_value : (3 - (-3)) / (2 - 1) = 6 := 
by
  sorry

end fraction_value_l318_318415


namespace remainder_of_base12_2563_mod_17_l318_318784

-- Define the base-12 number 2563 in decimal.
def base12_to_decimal : ℕ := 2 * 12^3 + 5 * 12^2 + 6 * 12^1 + 3 * 12^0

-- Define the number 17.
def divisor : ℕ := 17

-- Prove that the remainder when base12_to_decimal is divided by divisor is 1.
theorem remainder_of_base12_2563_mod_17 : base12_to_decimal % divisor = 1 :=
by
  sorry

end remainder_of_base12_2563_mod_17_l318_318784


namespace solve_log_eq_l318_318542

-- Conditions
variables {x : ℝ}
def log_eq (x : ℝ) := log 5 (x + 1) - log (1/5) (x - 3) = 1
def domain_condition := (x + 1 > 0) ∧ (x - 3 > 0)

-- Theorem
theorem solve_log_eq (h_domain : domain_condition) : log_eq 4 :=
sorry

end solve_log_eq_l318_318542


namespace rhombus_perimeter_52_l318_318023

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l318_318023


namespace evaluate_fx_plus_2_l318_318348

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem evaluate_fx_plus_2 (x : ℝ) (h : x ^ 2 ≠ 1) : 
  f (x + 2) = (x + 3) / (x + 1) :=
by
  sorry

end evaluate_fx_plus_2_l318_318348


namespace avg_english_dropped_students_l318_318013

/-- Given the conditions:
  1. The average score of 16 students' first quiz in an English class is 62.5.
  2. Three students dropped the English class.
  3. The new average quiz score of the remaining 13 students is 62.0.
  4. The dropped students have a combined average Math quiz score of 80.
  5. The total average grade of the dropped students is 72.
Prove that the combined average English quiz score of the three students who dropped 
the class is approximately 64.67. -/
theorem avg_english_dropped_students :
  let e_total := 16 * 62.5,
      e_remaining := 13 * 62.0,
      e_dropped := e_total - e_remaining,
      e_average_dropped := e_dropped / 3
  in e_average_dropped ≈ 64.67 :=
by {
  sorry
}

end avg_english_dropped_students_l318_318013


namespace unique_digit_sum_l318_318173

theorem unique_digit_sum (Y M E T : ℕ) (h1 : Y ≠ M) (h2 : Y ≠ E) (h3 : Y ≠ T)
    (h4 : M ≠ E) (h5 : M ≠ T) (h6 : E ≠ T) (h7 : 10 * Y + E = YE) (h8 : 10 * M + E = ME)
    (h9 : YE * ME = T * T * T) (hT_even : T % 2 = 0) : 
    Y + M + E + T = 10 :=
  sorry

end unique_digit_sum_l318_318173


namespace power_sum_l318_318754

theorem power_sum (n : ℕ) : (-2 : ℤ)^n + (-2 : ℤ)^(n+1) = 2^n := by
  sorry

end power_sum_l318_318754


namespace cos_squared_difference_l318_318918

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318918


namespace sum_of_fourth_powers_l318_318101

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1 / 2 :=
by sorry

end sum_of_fourth_powers_l318_318101


namespace smallest_integer_larger_than_sqrt3_sub_sqrt2_pow6_l318_318429

noncomputable def smallest_integer_larger_than (x : ℝ) : ℤ :=
  (floor x).to_int + 1

theorem smallest_integer_larger_than_sqrt3_sub_sqrt2_pow6 : smallest_integer_larger_than ((Real.sqrt 3 - Real.sqrt 2)^6) = 133 :=
by
  sorry

end smallest_integer_larger_than_sqrt3_sub_sqrt2_pow6_l318_318429


namespace minimum_positive_period_tan_l318_318734

theorem minimum_positive_period_tan (ω : ℝ) (hω : ω = 2) : 
  let T := π / |ω| in T = π / 2 :=
by
  sorry

end minimum_positive_period_tan_l318_318734


namespace cos_squared_difference_l318_318836

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318836


namespace cosine_difference_identity_l318_318849

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318849


namespace ordered_quadruples_count_l318_318201

theorem ordered_quadruples_count:
  ∃ (s : Finset (Real × Real × Real × Real)), 
    (∀ a b c d ∈ s, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 
      a^2 + b^2 + c^2 + d^2 = 4 ∧ a^5 + b^5 + c^5 + d^5 = 4) ∧ 
      s.card = 15 := 
by
  sorry

end ordered_quadruples_count_l318_318201


namespace exists_k_simplifies_expression_to_5x_squared_l318_318353

theorem exists_k_simplifies_expression_to_5x_squared :
  ∃ k : ℝ, (∀ x : ℝ, (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :=
by
  sorry

end exists_k_simplifies_expression_to_5x_squared_l318_318353


namespace base_conversion_least_sum_l318_318740

theorem base_conversion_least_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3) : a + b = 10 :=
sorry

end base_conversion_least_sum_l318_318740


namespace cos_squared_difference_l318_318930

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318930


namespace slope_range_intersection_l318_318572

theorem slope_range_intersection 
  (A B : ℝ × ℝ)
  (hA : A = (2, 4)) 
  (hB : B = (-3, 1)) : 
  ∃ k : ℝ, (k ≤ -1/3 ∨ k ≥ 2) ∧ ∃ P : ℝ × ℝ, P = (0, 0) ∧ intersects (line_through P A) (line_segment A B) :=
sorry

end slope_range_intersection_l318_318572


namespace triangle_perimeter_is_16_l318_318533

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (6, 6)

-- Calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Function to calculate the perimeter of triangle ABC
def triangle_perimeter (A B C : ℝ × ℝ) : ℝ := 
  distance A B + distance B C + distance C A

-- The theorem statement that the perimeter of the triangle is 16
theorem triangle_perimeter_is_16 : triangle_perimeter A B C = 16 := 
sorry

end triangle_perimeter_is_16_l318_318533


namespace elroy_more_miles_l318_318186

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l318_318186


namespace cos_squared_difference_l318_318883

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318883


namespace semicircle_area_difference_l318_318467

/-- A rectangle measures 8 meters by 12 meters. On each long side, a semicircle is drawn with the endpoints of its diameter on the vertices of the rectangle. On each short side, a semicircle is drawn with the endpoints of its diameter at the midpoints of the respective sides. The problem proves that the semicircles on the long sides have an area that is 125% larger than those on the short sides. --/
theorem semicircle_area_difference (length short_side long_side : ℝ) 
  (h_short_side : short_side = 8) (h_long_side : long_side = 12) :
  let r_long := long_side / 2,
      area_long := 2 * (1 / 2 * Real.pi * r_long ^ 2),
      r_short := short_side / 2,
      area_short := 2 * (1 / 2 * Real.pi * r_short ^ 2)
  in (area_long / area_short - 1) * 100 = 125 :=
by
  sorry

end semicircle_area_difference_l318_318467


namespace planes_parallel_if_lines_intersect_and_parallel_to_third_plane_l318_318671

variables {m n : Line} {α β : Plane}

def m_subset_alpha : Prop := m ⊆ α
def n_subset_alpha : Prop := n ⊆ α
def m_parallel_beta : Prop := m ∥ β
def n_parallel_beta : Prop := n ∥ β
def m_intersects_n : Prop := ∃ (p : Point), p ∈ m ∧ p ∈ n

theorem planes_parallel_if_lines_intersect_and_parallel_to_third_plane :
  m_subset_alpha →
  n_subset_alpha →
  m_intersects_n →
  m_parallel_beta →
  n_parallel_beta →
  α ∥ β :=
by
  -- Definitions are in place and need to be proven; proof omitted
  sorry

end planes_parallel_if_lines_intersect_and_parallel_to_third_plane_l318_318671


namespace cos_squared_difference_l318_318834

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318834


namespace hats_solution_l318_318426

-- Definitions for the number of hats of each color
variables (G B R Y : ℕ)

-- Conditions given in the problem
def B_eq_2G : Prop := B = 2 * G
def total_hats : Prop := B + G + R + Y = 150
def total_price : Prop := 8 * B + 10 * G + 12 * R + 15 * Y = 1280

-- The theorem we want to prove
theorem hats_solution (h1 : B_eq_2G) (h2 : total_hats) (h3 : total_price) : 
  G = 50 ∧ B = 100 ∧ R + Y = 0 := 
sorry

end hats_solution_l318_318426


namespace negation_equiv_l318_318737

-- Define the proposition that the square of all real numbers is positive
def pos_of_all_squares : Prop := ∀ x : ℝ, x^2 > 0

-- Define the negation of the proposition
def neg_pos_of_all_squares : Prop := ∃ x : ℝ, x^2 ≤ 0

theorem negation_equiv (h : ¬ pos_of_all_squares) : neg_pos_of_all_squares :=
  sorry

end negation_equiv_l318_318737


namespace cos_difference_identity_l318_318939

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318939


namespace relationship_y1_y2_l318_318573

theorem relationship_y1_y2
  (x1 y1 x2 y2 : ℝ)
  (hA : y1 = 3 * x1 + 4)
  (hB : y2 = 3 * x2 + 4)
  (h : x1 < x2) :
  y1 < y2 :=
sorry

end relationship_y1_y2_l318_318573


namespace vec_magnitude_sqrt_2_l318_318599

variable (a b : Type) [AddGroup a] [VectorSpace ℝ a]

def a_vec (m : ℝ) := (2 * m + 1, 3 : ℝ)
def b_vec (m : ℝ) := (2 : ℝ, m)

theorem vec_magnitude_sqrt_2 
  (m : ℝ)
  (collinear : ∃ k : ℝ, a_vec m = k • b_vec m)
  (dot_product_neg : 2 * (2 * m + 1) + 3 * m < 0) :
  ‖b_vec (-2)‖ = 2 * Real.sqrt 2 :=
  sorry

end vec_magnitude_sqrt_2_l318_318599


namespace liters_of_milk_l318_318974

def total_cost (flour eggs soda : Nat) : Nat :=
  3 * flour + 10 * eggs + 3 * soda

def milk_cost (total_cost paid : Nat) : Nat :=
  paid - total_cost

def milk_quantity (cost_per_liter total_milk_cost : Nat) : Nat :=
  total_milk_cost / cost_per_liter

theorem liters_of_milk (liters : Nat) (paid : Nat) (flour_qty : Nat) (eggs_qty : Nat) (soda_qty : Nat) :
  total_cost flour_qty eggs_qty soda_qty + milk_cost (total_cost flour_qty eggs_qty soda_qty) paid = paid 
  ∧ milk_quantity 5 (milk_cost (total_cost flour_qty eggs_qty soda_qty) paid) = liters ↔ liters = 7 :=
by
  let totalFlourCost := 3 * 3
  let totalEggsCost := 3 * 10
  let totalSodaCost := 2 * 3
  let combinedCost := totalFlourCost + totalEggsCost + totalSodaCost
  
  have totalCostsCorrect : totalCost 3 3 2 = combinedCost := sorry
  have amountPaid : paid = 80 := sorry
  have totalMilkCost : milk_cost combinedCost 80 = 35 := sorry
  have correctMilkQuantity : milk_quantity 5 35 = 7 := sorry
  
  exact ⟨totalCostsCorrect, amountPaid, totalMilkCost, correctMilkQuantity⟩

end liters_of_milk_l318_318974


namespace distance_A_B_general_distance_A_B_parallel_y_l318_318703

-- Proof problem for Question 1:
theorem distance_A_B_general 
  (x1 y1 x2 y2 : ℝ) 
  (hx1 : x1 = 2) (hy1 : y1 = 4)
  (hx2 : x2 = -3) (hy2 : y2 = -8) : 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 13 := 
by {
  rw [hx1, hy1, hx2, hy2],
  calc real.sqrt ((-3 - 2)^2 + (-8 - 4)^2)
      = real.sqrt ((-5)^2 + (-12)^2) : by norm_num
  ... = real.sqrt (25 + 144) : by norm_num
  ... = real.sqrt (169) : by norm_num
  ... = 13 : by norm_num }

-- Proof problem for Question 2:
theorem distance_A_B_parallel_y 
  (y1 y2 : ℝ) 
  (hy1 : y1 = 5) (hy2 : y2 = -1) : 
  |y2 - y1| = 6 := 
by {
  rw [hy1, hy2],
  calc |(-1) - 5|
      = |(-6)| : by norm_num 
  ... = 6 : by norm_num }

end distance_A_B_general_distance_A_B_parallel_y_l318_318703


namespace prob_none_given_not_A_l318_318691

-- Definitions based on the conditions
def prob_single (h : ℕ → Prop) : ℝ := 0.2
def prob_double (h1 h2 : ℕ → Prop) : ℝ := 0.1
def prob_triple_given_AB : ℝ := 0.5

-- Assume that h1, h2, and h3 represent the hazards A, B, and C respectively.
variables (A B C : ℕ → Prop)

-- The ultimate theorem we want to prove
theorem prob_none_given_not_A (P : ℕ → Prop) :
  ((1 - (0.2 * 3 + 0.1 * 3) + (prob_triple_given_AB * (prob_single A + prob_double A B))) / (1 - 0.2) = 11 / 9) :=
by
  sorry

end prob_none_given_not_A_l318_318691


namespace cos_squared_difference_l318_318955

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318955


namespace expected_value_ξ_l318_318760

noncomputable theory
open_locale big_operators

-- Definitions of the conditions used in the problem
def total_balls := 5
def new_balls := 3
def old_balls := 2
def balls_picked := 2

-- Definition of ξ (ξ is the number of new balls picked in the second match - formal introduction)
def ξ : ℕ → ℝ := λ k, if k = 0 then 0 else if k = 1 then 1 else 2

-- Main theorem
theorem expected_value_ξ : 
  (∑ x in ({0, 1, 2} : finset ℕ), (ξ x * ((choose new_balls x * choose old_balls (balls_picked - x))/choose total_balls balls_picked))) = 18 / 25 :=
sorry

end expected_value_ξ_l318_318760


namespace cross_section_area_ratio_squared_l318_318507

-- Define the points in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the coordinates of all points
def P : Point3D := ⟨0, 0, 0⟩
def Q : (s : ℝ) → Point3D := λ s, ⟨2 * s, 0, 0⟩
def R : (s : ℝ) → Point3D := λ s, ⟨2 * s, 2 * s, 0⟩
def S : (s : ℝ) → Point3D := λ s, ⟨0, 2 * s, 0⟩
def T : (s : ℝ) → Point3D := λ s, ⟨0, 0, 2 * s⟩
def U : (s : ℝ) → Point3D := λ s, ⟨2 * s, 0, 2 * s⟩
def V : (s : ℝ) → Point3D := λ s, ⟨2 * s, 2 * s, 2 * s⟩
def W : (s : ℝ) → Point3D := λ s, ⟨0, 2 * s, 2 * s⟩
def M : (s : ℝ) → Point3D := λ s, ⟨2 * s, 0, s / 2⟩
def N : (s : ℝ) → Point3D := λ s, ⟨0, 2 * s, 3 * s / 2⟩

-- Define a function that calculates the cross product of two vectors
def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y,
   u.z * v.x - u.x * v.z,
   u.x * v.y - u.y * v.x⟩

-- Define a function to calculate the magnitude of a vector
def magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

-- Prove the ratio squared K^2 of the area of the cross section to the area of one face of the cube
theorem cross_section_area_ratio_squared (s : ℝ) : let
  pm := Point3D.mk (2 * s) 0 (s / 2),
  pn := Point3D.mk 0 (2 * s) (3 * s / 2) in
  let cp := cross_product pm pn in
  let area_PMNP := 1/2 * magnitude cp in
  let area_face := 4 * s * s in
  (area_PMNP / area_face) ^ 2 = 339 / 256 :=
by {
  let pm := M s,
  let pn := N s,
  let pm := Point3D.mk pm.x pm.y pm.z,
  let pn := Point3D.mk pn.x pn.y pn.z,
  let cp := cross_product pm pn,
  sorry
}

end cross_section_area_ratio_squared_l318_318507


namespace downstream_speed_is_33_l318_318998

-- Definitions based on the given conditions
def upstream_speed : ℝ := 7 -- kmph
def still_water_speed : ℝ := 20 -- kmph

-- The goal is to prove that the downstream_speed is 33 kmph
theorem downstream_speed_is_33 : ∃ (downstream_speed : ℝ), 
  ((still_water_speed = (upstream_speed + downstream_speed) / 2) ∧ downstream_speed = 33) :=
by
  let downstream_speed := 33
  use downstream_speed
  split
  · -- Proof that still_water_speed equals the average of upstream_speed and downstream_speed
    calc
      still_water_speed = (upstream_speed + downstream_speed) / 2 : by
        have h : 20 = (7 + 33) / 2 := by
          rw [← add_div (7 : ℝ) (33 : ℝ) 2]
          norm_num
        exact h
  · -- Proof that downstream_speed is 33
    norm_num

end downstream_speed_is_33_l318_318998


namespace area_two_layers_l318_318058

def table_area : ℝ := 175
def combined_area_runners : ℝ := 224
def table_coverage_percentage : ℝ := 0.80
def three_layers_area : ℝ := 30
def two_layers_area : ℝ := 30

theorem area_two_layers :
  let total_table_coverage := table_coverage_percentage * table_area in
  let remaining_runner_area := combined_area_runners - 3 * three_layers_area in
  let one_or_two_layers_area := total_table_coverage - three_layers_area in
  two_layers_area + one_or_two_layers_area - one_or_two_layers_area = two_layers_area := 
  sorry

end area_two_layers_l318_318058


namespace cos_squared_difference_l318_318808

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318808


namespace average_speed_l318_318060

theorem average_speed (D : ℝ) (h_pos : 0 < D)
  (speed_WB : ℝ := 60) (speed_BC : ℝ := 20)
  (dist_WB : ℝ := 2 * D) (dist_BC : ℝ := D) :
  let total_distance := dist_WB + dist_BC
      time_WB := dist_WB / speed_WB
      time_BC := dist_BC / speed_BC
      total_time := time_WB + time_BC
  in (total_distance / total_time) = 36 := sorry

end average_speed_l318_318060


namespace find_m_and_n_profit_relation_find_max_a_l318_318773

-- Define the constants and their values
def seafood_skewer_cost := 3
def meat_skewer_cost := 2

-- Define the conditions from the problem
def first_time_cost (m n : ℕ) := 3000 * m + 4000 * n = 17000
def second_time_cost (m n : ℕ) := 4000 * m + 3000 * n = 18000

-- Part (1): Prove values of m and n
theorem find_m_and_n : 
  (∃ m n : ℕ, 3000 * m + 4000 * n = 17000 ∧ 4000 * m + 3000 * n = 18000 ∧ m = seafood_skewer_cost ∧ n = meat_skewer_cost) := 
by {
  use [3, 2],
  split, norm_num,
  split, norm_num,
  split, refl,
  refl
}

-- Part (2): Functional relationship between y and x
def profit_seafood_skewers (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 200 then 2 * x else if 200 < x ∧ x ≤ 400 then x + 200 else 0

theorem profit_relation : 
  ∀ x : ℕ, 0 < x ∧ x ≤ 400 -> 
    (if 0 < x ∧ x ≤ 200 then profit_seafood_skewers x = 2 * x 
     else if 200 < x ∧ x ≤ 400 then profit_seafood_skewers x = x + 200 
     else false) := 
by {
  intro x,
  intro h,
  cases h with h1 h2,
  split_ifs;
  norm_num
}

-- Part (3): Maximum value of a
theorem find_max_a : 
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ (∀ x : ℕ, 200 < x ∧ x ≤ 400 ->
    let profit_meat := (3.5 - a) * (1000 - x) - 2 * (1000 - x) in
    let profit_seafood := profit_seafood_skewers x in
    profit_meat ≥ profit_seafood)  ∧ a = 0.5 := 
by {
  use 0.5,
  split, norm_num,
  split, norm_num,
  intros, norm_num,
  split; linarith
}

end find_m_and_n_profit_relation_find_max_a_l318_318773


namespace squares_in_region_l318_318280

def is_square_contained (x1 y1 x2 y2 : ℕ) (y : ℕ → ℕ) : Prop :=
  x2 - x1 = y2 - y1 ∧ x2 - x1 = y x2 - y x1

theorem squares_in_region :
  let region_bounded := λ x y, y ≤ 2 * x ∧ y ≥ -1 ∧ x ≤ 6 in
  ∃ n : ℕ, n = 74 ∧ 
  (∀ (x1 y1 x2 y2 : ℕ), 
    region_bounded x1 y1 → region_bounded x2 y2 → 
    (x1, y1) ≠ (x2, y2) → 
    is_square_contained x1 y1 x2 y2 2 → n = 74) :=
sorry

end squares_in_region_l318_318280


namespace diagonal_elements_are_1_to_n_l318_318509

-- Define the problem
theorem diagonal_elements_are_1_to_n (n : ℕ) (A : Array (Array ℕ))
  (h_odd : n % 2 = 1)
  (h_size : A.size = n ∧ ∀ i, (A[i]).size = n)
  (h_rows : ∀ i, ∃ perm_i : List ℕ, perm_i ~ [1,2,⋯,n] ∧ perm_i = A[i].toList)
  (h_symmetry : ∀ i j, A[i][j] = A[j][i]) :
  ∃ perm_d : List ℕ, perm_d ~ [1,2,⋯,n] ∧ perm_d = List.ofFn (λ i, A[i][i]) :=
by
  sorry

end diagonal_elements_are_1_to_n_l318_318509


namespace num_entries_multiple_of_73_l318_318137

theorem num_entries_multiple_of_73 :
  let b : ℕ → ℕ → ℕ := λ n k, 2^(n-1) * (n + 2 * k) in
  (∑ n in finset.range 28, (nemultiple_of_73 n).card k ((n + 2 * k) = 73)) = 14 :=
begin
  sorry
end

end num_entries_multiple_of_73_l318_318137


namespace cos_squared_difference_l318_318953

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318953


namespace correct_option_l318_318237

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variable (h_cond : ∀ x : ℝ, f x > deriv f x)

theorem correct_option :
  e ^ 2016 * f (-2016) > f 0 ∧ f 2016 < e ^ 2016 * f 0 :=
sorry

end correct_option_l318_318237


namespace alice_average_speed_l318_318143

/-- Alice cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour. 
    The average speed for the entire trip --/
theorem alice_average_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  (total_distance / total_time) = (120 / 11) := 
by
  sorry -- proof steps would go here

end alice_average_speed_l318_318143


namespace total_difference_is_minus_3_total_weight_of_sampled_bags_is_1497_l318_318106

-- Define the differences and the number of bags according to the problem's conditions
def differences : List ℤ := [-2, -1, 0, 1, 2]
def number_of_bags : List ℕ := [2, 3, 2, 2, 1]
def standard_weight_per_bag : ℤ := 150
def total_number_of_bags : ℕ := 10

-- Part 1: Prove the total difference in weight is -3 grams
theorem total_difference_is_minus_3 :
  ∑ (i : ℕ) in (Finset.range 5), differences.nthLe i (by sorry) * number_of_bags.nthLe i (by sorry) = -3 :=
sorry

-- Part 2: Prove the total weight of the sampled bags is 1497 grams
theorem total_weight_of_sampled_bags_is_1497 :
  (total_number_of_bags * standard_weight_per_bag + ∑ (i : ℕ) in (Finset.range 5), differences.nthLe i (by sorry) * (number_of_bags.nthLe i (by sorry) : ℤ)) = 1497 :=
sorry

end total_difference_is_minus_3_total_weight_of_sampled_bags_is_1497_l318_318106


namespace compute_zeta_seventh_power_sum_l318_318098

noncomputable def complex_seventh_power_sum : Prop :=
  ∀ (ζ₁ ζ₂ ζ₃ : ℂ), 
    (ζ₁ + ζ₂ + ζ₃ = 1) ∧ 
    (ζ₁^2 + ζ₂^2 + ζ₃^2 = 3) ∧
    (ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) →
    (ζ₁^7 + ζ₂^7 + ζ₃^7 = 71)

theorem compute_zeta_seventh_power_sum : complex_seventh_power_sum :=
by
  sorry

end compute_zeta_seventh_power_sum_l318_318098


namespace cos_squared_difference_l318_318828

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318828


namespace triangle_is_right_triangle_l318_318628

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 10 * a - 6 * b - 8 * c + 50 = 0) :
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2 :=
sorry

end triangle_is_right_triangle_l318_318628


namespace A_winning_strategy_l318_318451

-- Definitions of the conditions and entities
def totalPeople : ℕ := 2003
def A_is_not_adjacent (A B : ℕ) : Prop := B > A + 1 ∨ A > B + 1

-- The main theorem stating that A has a winning strategy.
theorem A_winning_strategy (A B : ℕ) (H1 : A ≤ totalPeople ∧ B ≤ totalPeople) (H2 : A ≠ B) (H3 : A_is_not_adjacent A B) : 
  ∃ strategy : (ℕ → ℕ) → Prop, (strategy = (λ A_turn : ℕ → ℕ, ∀ turn, player_wins A_turn turn A B)) :=
sorry

end A_winning_strategy_l318_318451


namespace cos_diff_square_identity_l318_318824

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318824


namespace cos_diff_square_identity_l318_318819

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318819


namespace min_value_l318_318236

variables (a b c : ℝ)
variable (hpos : a > 0 ∧ b > 0 ∧ c > 0)
variable (hsum : a + b + c = 1)

theorem min_value (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) :
  9 * a^2 + 4 * b^2 + (1/4) * c^2 = 36 / 157 := 
sorry

end min_value_l318_318236


namespace min_bottles_needed_l318_318459

theorem min_bottles_needed (bottle_size : ℕ) (min_ounces : ℕ) (n : ℕ) 
  (h1 : bottle_size = 15) 
  (h2 : min_ounces = 195) 
  (h3 : 15 * n >= 195) : n = 13 :=
sorry

end min_bottles_needed_l318_318459


namespace proof_rewritten_eq_and_sum_l318_318380

-- Define the given equation
def given_eq (x : ℝ) : Prop := 64 * x^2 + 80 * x - 72 = 0

-- Define the rewritten form of the equation
def rewritten_eq (x : ℝ) : Prop := (8 * x + 5)^2 = 97

-- Define the correctness of rewriting the equation
def correct_rewrite (x : ℝ) : Prop :=
  given_eq x → rewritten_eq x

-- Define the correct value of a + b + c
def correct_sum : Prop :=
  8 + 5 + 97 = 110

-- The final theorem statement
theorem proof_rewritten_eq_and_sum (x : ℝ) : correct_rewrite x ∧ correct_sum :=
by
  sorry

end proof_rewritten_eq_and_sum_l318_318380


namespace cos_squared_difference_l318_318884

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318884


namespace number_of_correct_statements_l318_318483

theorem number_of_correct_statements :
  let s1 := (∀ (P Q R : ℝ^3), P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ 
              ∃ (α β : set (ℝ^3)), {P, Q, R} ⊆ α ∧ {P, Q, R} ⊆ β → α = β)
  let s2 := (∀ (a b : ℝ^3), a ≠ b → ∃ (α : set (ℝ^3)), a ∈ α ∧ b ∈ α)
  let s3 := (∀ (M : ℝ^3) (α β : set (ℝ^3)) (l : set (ℝ^3)), 
              M ∈ α ∧ M ∈ β ∧ (α ∩ β = l) → M ∈ l)
  let s4 := (∀ (l1 l2 l3 : set (ℝ^3)), 
              (∃ (P : ℝ^3), P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3) → 
              ∃ (α : set (ℝ^3)), l1 ⊆ α ∧ l2 ⊆ α ∧ l3 ⊆ α)
  s1 ∧ ¬s2 ∧ ¬s3 ∧ ¬s4 →
  (s1 = true ∧ s2 = false ∧ s3 = false ∧ s4 = false) ∧ 
  (s1 ↔ ¬s2 ∧ ¬s2 ↔ ¬s3 ∧ ¬s3 ↔ ¬s4) :=
begin
  sorry
end

end number_of_correct_statements_l318_318483


namespace min_value_sqrt_x2_y2_l318_318294

theorem min_value_sqrt_x2_y2 (x y : ℝ) (h : 3 * x^2 + 2 * (real.sqrt 3) * x * y + y^2 = 1) :
  ∃ (m : ℝ), m = real.sqrt (x^2 + y^2) ∧ m = 1 / 2 :=
by
  sorry

end min_value_sqrt_x2_y2_l318_318294


namespace common_ratio_is_half_l318_318642

variable {a₁ q : ℝ}

-- Given the conditions of the geometric sequence

-- First condition
axiom h1 : a₁ + a₁ * q ^ 2 = 10

-- Second condition
axiom h2 : a₁ * q ^ 3 + a₁ * q ^ 5 = 5 / 4

-- Proving that the common ratio q is 1/2
theorem common_ratio_is_half : q = 1 / 2 :=
by
  -- The proof details will be filled in here.
  sorry

end common_ratio_is_half_l318_318642


namespace selection_methods_count_l318_318417

theorem selection_methods_count (total_doctors female_doctors : ℕ) (doctor_count team_size : ℕ)
  (min_male max_female : ℕ) (N : ℕ) :
  total_doctors = 13 ∧ female_doctors = 6 ∧ doctor_count = total_doctors ∧ team_size = 5 ∧
  min_male = 2 ∧ max_female = 3 ∧
  N = (nat.choose 7 2 * nat.choose 6 3 + nat.choose 7 3 * nat.choose 6 2 +
       nat.choose 7 4 * nat.choose 6 1 + nat.choose 7 5 ∨
       N = (nat.choose 13 5 - nat.choose 7 1 * nat.choose 6 4 - nat.choose 6 5)) := by
  sorry

end selection_methods_count_l318_318417


namespace cos_difference_squared_l318_318960

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318960


namespace cos_difference_squared_l318_318958

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318958


namespace plane_can_fly_approx_40_minutes_l318_318121

-- The conditions given in the problem
def fuel_rate : ℝ := 9.5
def fuel_left : ℝ := 6.3333

-- The function to calculate the time of flight in minutes
def time_of_flight_in_minutes (fuel_left fuel_rate : ℝ) : ℝ :=
  (fuel_left / fuel_rate) * 60

-- The main theorem we want to prove
theorem plane_can_fly_approx_40_minutes :
  abs (time_of_flight_in_minutes fuel_left fuel_rate - 40) < 1e-6 :=
by
  sorry

end plane_can_fly_approx_40_minutes_l318_318121


namespace problem1_l318_318970

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
sorry

end problem1_l318_318970


namespace range_of_f_l318_318746

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1 else x^2 - 2 * x

theorem range_of_f : set.range f = set.Ioi (-1) := by
  sorry

end range_of_f_l318_318746


namespace circle_area_from_tangency_conditions_l318_318456

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - 20 * y^2 = 24

-- Tangency to the x-axis implies the circle's lowest point touches the x-axis
def tangent_to_x_axis (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ r y₀, circle 0 y₀ ∧ y₀ = r

-- The circle is given as having tangency conditions to derive from
theorem circle_area_from_tangency_conditions (circle : ℝ → ℝ → Prop) :
  (∀ x y, circle x y → (x = 0 ∨ hyperbola x y)) →
  tangent_to_x_axis circle →
  ∃ area, area = 504 * Real.pi :=
by
  sorry

end circle_area_from_tangency_conditions_l318_318456


namespace max_min_value_of_f_l318_318724

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_min_value_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f (Real.pi / 2) ≤ f x) :=
by
  sorry

end max_min_value_of_f_l318_318724


namespace roots_of_polynomial_l318_318359

-- Define the complex numbers a, b, c
variables {a b c : ℂ}

-- State the conditions
def condition_1 := a + b + c = 1
def condition_2 := a * b + a * c + b * c = 1
def condition_3 := a * b * c = -1

-- State the theorem that needs to be proven
theorem roots_of_polynomial :
  condition_1 ∧ condition_2 ∧ condition_3 → 
  (a = 1 ∨ a = complex.I ∨ a = -complex.I) ∧
  (b = 1 ∨ b = complex.I ∨ b = -complex.I) ∧
  (c = 1 ∨ c = complex.I ∨ c = -complex.I) :=
by { sorry }

end roots_of_polynomial_l318_318359


namespace find_abc_l318_318408

theorem find_abc (a b c : ℕ) 
(h1 : a < 10) (h2 : b < 10) (h3 : c < 10) 
(h4 : (0.ababab : ℚ) + (0.abcabc : ℚ) = 17 / 37) : 
100 * a + 10 * b + c = 270 := 
sorry

end find_abc_l318_318408


namespace probability_X_geq_4_l318_318600

def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem probability_X_geq_4 (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let X : ℕ → ℝ := λ k, binomial_prob 5 k p in
  (X 4 + X 5 = 5 * p^4 - 3 * p^5) :=
begin
  sorry
end

end probability_X_geq_4_l318_318600


namespace rhombus_perimeter_l318_318025

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h3 : d1 / 2 ≠ 0) (h4 : d2 / 2 ≠ 0) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * s = 52 :=
by
  sorry

end rhombus_perimeter_l318_318025


namespace hyperbola_eq_l318_318259

theorem hyperbola_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∃ c, 
    c = 3 ∧ 
    (∀ x y : ℝ, (x^2 + y^2 - 6*x + 5 = 0) → ((3*b)/(sqrt (a^2 + b^2)) = 2)) ∧ 
    ((3 (sqrt(a^2 + b^2))) = 3) ∧ 
    a^2 = 9 - b^2 ∧ 
    b = 2) → 
    (\dfrac{x^2}{5} - \dfrac{y^2}{4} = 1) :=
by 
  intros h; 
  rcases h with ⟨c, hc1, hc2, hc3, hc4, hc5⟩;
  sorry

end hyperbola_eq_l318_318259


namespace george_vs_lawrence_time_difference_l318_318653

theorem george_vs_lawrence_time_difference (d : ℕ) (h_d : d = 52) :
  let t_L := (d / 2) * 8
  let t_G := (d / 2) * 12
  t_G - t_L = 104 := 
by
  -- Calculation of time taken by Lawrence
  let t_L := (52 / 2) * 8
  -- Calculation of time taken by George
  let t_G := (52 / 2) * 12
  -- Calculate the difference
  have : t_G - t_L = 104 := by
    calc 
      t_G - t_L
      = ((52 / 2) * 12) - ((52 / 2) * 8) : by sorry
      = 104 : by sorry
  exact this

end george_vs_lawrence_time_difference_l318_318653


namespace average_infection_rate_l318_318117

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l318_318117


namespace find_m_l318_318243

-- Defining the points and conditions as described in the problem
variable {m : ℝ}

-- Define Line equation and their properties
def line1 (x y : ℝ) := 2 * x - y - 1 

-- Define point A and the point B
def pointA := (-2, m)
def pointB := (m, 10)

-- Define slope calculation between two points which should equal to 2
def slope (p1 p2 : ℝ × ℝ) := (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Problem statement: Prove that if the line through the points A and B is parallel to the given line, then m = 2.
theorem find_m (h : slope pointA pointB = 2) : m = 2 :=
by sorry

end find_m_l318_318243


namespace area_triangle_AKC_l318_318742

variables (A B C D A' K : Type)
variables (BC A'D : line)
variables [parallelogram ABCD]
variables [folds_along_BD : folds_along ABCD BD A' C]
variables [intersects_at : intersects BC A'D K]
variables [ratio_BK_KC : ratio BK KC 3 2]
variables [area_ABCD : area ABCD 27]

theorem area_triangle_AKC : area (triangle A' K C) = 3.6 :=
sorry

end area_triangle_AKC_l318_318742


namespace solve_for_k_l318_318301

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, (3 * x - 6 = 0) ∧ (2 * x - 5 * k = 11)) → k = -7/5 :=
by 
  intro h
  cases' h with x hx
  have hx1 : x = 2 := by linarith
  have hx2 : x = 11 / 2 + 5 / 2 * k := by linarith
  linarith

end solve_for_k_l318_318301


namespace ab_bc_ca_lt_sqrt_abc_div_2_add_1_div_4_l318_318682

theorem ab_bc_ca_lt_sqrt_abc_div_2_add_1_div_4 (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : abc > 0) :
  ab + bc + ca < sqrt abc / 2 + 1 / 4 := 
by sorry

end ab_bc_ca_lt_sqrt_abc_div_2_add_1_div_4_l318_318682


namespace greatest_drop_in_price_l318_318114

def jan_change : ℝ := -0.75
def feb_change : ℝ := 1.50
def mar_change : ℝ := -3.00
def apr_change : ℝ := 2.50
def may_change : ℝ := -0.25
def jun_change : ℝ := 0.80
def jul_change : ℝ := -2.75
def aug_change : ℝ := -1.20

theorem greatest_drop_in_price : 
  mar_change = min (min (min (min (min (min jan_change jul_change) aug_change) may_change) feb_change) apr_change) jun_change :=
by
  -- This statement is where the proof would go.
  sorry

end greatest_drop_in_price_l318_318114


namespace hexagon_area_equality_l318_318341

theorem hexagon_area_equality (b : ℝ) (u : ℝ) :
  let A := (0, 10 : ℝ)
  let B := (b, 12 : ℝ)
  let P : ℝ × ℝ := _
  let Q : ℝ × ℝ := _
  let R : ℝ × ℝ := _
  let S : ℝ × ℝ := _
  let T : ℝ × ℝ := _
  let U := (u, 14 : ℝ)
  let vertices := [A, B, P, Q, R, S, T, U]
  -- ensuring all distinct y-coordinates and given y-coordinate set
  (∀ v ∈ vertices, v.2 ∈ {10, 12, 14, 16, 18, 20}) ∧
  (∀ v1 v2 ∈ vertices, v1 ≠ v2 → v1.2 ≠ v2.2) ∧
  ∠UAP = 120 ∧
  -- ensuring all parallel conditions
  let A_B_Q_R_BC_RS_CD_UV := 
    (line_through A B) || (line_through Q R) ∧
    (line_through B R) || (line_through Q S) ∧
    (line_through B R) || (line_through Q T) ∧
    (line_through C D) || (line_through U P) ∧
  -- proving the area of hexagon is equal to the expected value
  area_of_hexagon P Q R S T U = (16 * 3^.half + 48) / 3 :=
sorry

end hexagon_area_equality_l318_318341


namespace land_plot_side_length_l318_318446

theorem land_plot_side_length (area : ℝ) (h : area = real.sqrt 100) : real.sqrt area = 10 :=
by
  rw h
  rw real.sqrt_sqrt
  norm_num
  exact real.sqrt_nonneg 100

end land_plot_side_length_l318_318446


namespace length_O1O2_constant_l318_318306

variables (A B C D E O1 O2 : Type)
variables [trapezoid A B C D] [parallel AD BC] [point_on E AB]
variables [circumcenter O1 (triangle A E D)] [circumcenter O2 (triangle B E C)]

theorem length_O1O2_constant :
  ∃ k : ℝ, ∀ E : AB, distance O1 O2 = k :=
begin
  sorry
end

end length_O1O2_constant_l318_318306


namespace proof_problem_l318_318904

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318904


namespace complex_number_calculation_l318_318589

theorem complex_number_calculation (z : ℂ) (hz : z = 1 - I) : (z^2 / (z - 1)) = 2 := by
  sorry

end complex_number_calculation_l318_318589


namespace average_abs_diff_sum_l318_318206

open Finset

-- Define the absolute difference sum over permutations
def abs_diff_sum (s : Permutations (Fin 8)) : ℝ :=
|b_1 - b_2| + |b_3 - b_4| + |b_5 - b_6| + |b_7 - b_8|

-- The main statement
theorem average_abs_diff_sum : 
  ∃ (p q : ℕ), Nat.coprime p q ∧ (p / q : ℝ) = 12 ∧ p + q = 13 := by
  sorry

end average_abs_diff_sum_l318_318206


namespace probability_x_gt_5y_in_rectangle_l318_318381

theorem probability_x_gt_5y_in_rectangle :
  let rect := set.univ.prod (set.Icc 0 4040) (set.Icc 0 2020) in
  let prob := (set_of (λ (p : ℝ × ℝ), p.1 > 5 * p.2 ∧ p ∈ rect) : set (ℝ × ℝ)) in
  (measure_theory.volume prob / measure_theory.volume rect) = (101 / 505) :=
sorry

end probability_x_gt_5y_in_rectangle_l318_318381


namespace cos_squared_difference_l318_318806

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318806


namespace cos_diff_square_identity_l318_318822

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318822


namespace parallelogram_area_l318_318526

theorem parallelogram_area (b s : ℝ) (θ : ℝ) (h : ℝ) (area : ℝ) (sin_θ : ℝ) 
  (hb : b = 24) (hs : s = 16) (hθ : θ = real.pi / 3) 
  (hsin : sin_θ = real.sin (real.pi / 3)) 
  (hh : h = s * sin_θ) 
  (ha : area = b * h) : 
  area ≈ 332.544 := 
by
  sorry

end parallelogram_area_l318_318526


namespace like_terms_mn_equal_l318_318232

theorem like_terms_mn_equal (a b : ℕ) (m n: ℕ) (h_like_terms : (-32) * a^(2*m) * b = b^(3-n) * a^4) : m^n = n^m :=
by {
  -- Using the condition that the terms are like terms to equate the exponents
  have h_exponents_a : 2 * m = 4, from (
    -- Extract the relationship of exponents for a from the like terms condition
  ),
  have h_exponents_b : 3 - n = 1, from (
    -- Extract the relationship of exponents for b from the like terms condition
  ),
  -- From h_exponents_a and h_exponents_b, solve for m and n
  have h_m : m = 2, from sorry,
  have h_n : n = 2, from sorry,
  -- Conclude that m^n = n^m
  rw [h_m, h_n],
  exact rfl,
}


end like_terms_mn_equal_l318_318232


namespace p_sufficient_not_necessary_q_l318_318617

def p (x : ℝ) : Prop := log x / log 2 < 0
def q (x : ℝ) : Prop := x < 1

theorem p_sufficient_not_necessary_q : ∀ (x : ℝ), p x → q x ∧ (¬ q x → ¬ p x) := 
by 
  -- proof goes here
  sorry

end p_sufficient_not_necessary_q_l318_318617


namespace total_charging_time_l318_318122

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l318_318122


namespace fraction_division_problem_l318_318702

theorem fraction_division_problem :
  (-1/42 : ℚ) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 :=
by
  -- Skipping the proof step as per the instructions
  sorry

end fraction_division_problem_l318_318702


namespace dancers_not_slow_dance_l318_318762

theorem dancers_not_slow_dance (total_kids : ℕ) (fraction_dancers : ℚ) (slow_dancers : ℕ) 
  (h1 : total_kids = 140) (h2 : fraction_dancers = 1/4) (h3 : slow_dancers = 25) : 
  let total_dancers := (total_kids : ℚ) * fraction_dancers in
  total_dancers - slow_dancers = 10 := 
by
  let total_dancers := (total_kids : ℚ) * fraction_dancers
  have total_dancers_nat : ℕ := total_dancers.toNat
  have : total_dancers_nat = 35 := by
    rw [h1, h2]
    simp only [Rat.cast_one, Rat.cast_inv, Rat.cast_bit1, Rat.cast_zero, Rat.cast_add, Fraction.ceil]
    norm_num
  have : total_dancers_nat - slow_dancers = 10 := by
    rw [this, h3]
    norm_num
  contra sorry

end dancers_not_slow_dance_l318_318762


namespace max_number_of_triangles_l318_318325

theorem max_number_of_triangles (num_sides : ℕ) (num_internal_points : ℕ) 
    (total_points : ℕ) (h1 : num_sides = 13) (h2 : num_internal_points = 200) 
    (h3 : total_points = num_sides + num_internal_points) 
    (h4 : ∀ (x y z : point), x ≠ y ∧ y ≠ z ∧ z ≠ x → ¬ collinear x y z) : 
    (total_points.choose 3) = 411 :=
by
  sorry

end max_number_of_triangles_l318_318325


namespace round_3_65_to_nearest_tenth_l318_318002

theorem round_3_65_to_nearest_tenth : (Real.floor (3.65 * 10) + 1) / 10 = 3.7 := by
  sorry

end round_3_65_to_nearest_tenth_l318_318002


namespace magnitude_of_sum_l318_318586

-- Given conditions
variables {a b : EuclideanSpace ℝ (fin 2)}  -- Vectors a and b in 2D space
variables (ha : ‖a‖ = 3)  -- Magnitude of vector a is 3
variables (hb : ‖b‖ = 4)  -- Magnitude of vector b is 4
variables (h_diff : a - b = ![real.sqrt 2, real.sqrt 7])  -- Difference of vectors a and b

-- Goal: Magnitude of (a + b) is sqrt(41)
theorem magnitude_of_sum : ‖a + b‖ = real.sqrt 41 := by
  sorry

end magnitude_of_sum_l318_318586


namespace cos_difference_squared_l318_318961

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318961


namespace intersection_point_l318_318516

def point := (ℝ × ℝ)

def line1 (p : point) : Prop :=
  3 * p.2 = -2 * p.1 + 6

def line2 (p : point) : Prop :=
  -2 * p.2 = 6 * p.1 - 4

theorem intersection_point :
  ∃ p : point, line1 p ∧ line2 p ∧ p = (0, 2) :=
by {
  use (0, 2),
  split,
  {
    -- Check the condition for line1
    unfold line1,
    norm_num,
  },
  split,
  {
    -- Check the condition for line2
    unfold line2,
    norm_num,
  },
  -- Check if the point is indeed (0, 2)
  refl,
}

end intersection_point_l318_318516


namespace necessary_but_not_sufficient_l318_318218

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a ≠ 0) → (ab ≠ 0) ↔ (a ≠ 0) :=
by sorry

end necessary_but_not_sufficient_l318_318218


namespace max_winners_3_matches_200_participants_l318_318494

theorem max_winners_3_matches_200_participants (n : ℕ) (h_total : n = 200):
  ∃ (x : ℕ), (∀ y : ℕ, (3 * y ≤ 199) → y ≤ 66) ∧ (3 * 66 ≤ 199) :=
by
  use 66
  split
  · intro y h
    have : (n - 1) = 199 := by linarith
    suffices : 3 * 66 ≤ (n - 1) by linarith
    exact this
    sorry
  · linarith

end max_winners_3_matches_200_participants_l318_318494


namespace cos_squared_difference_l318_318832

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318832


namespace remainder_div_150_by_4_eq_2_l318_318547

theorem remainder_div_150_by_4_eq_2 :
  (∃ k : ℕ, k > 0 ∧ 120 % k^2 = 24) → 150 % 4 = 2 :=
by
  intro h
  sorry

end remainder_div_150_by_4_eq_2_l318_318547


namespace solution_set_f_of_10_to_the_x_l318_318587

theorem solution_set_f_of_10_to_the_x (f : ℝ → ℝ)
  (h : ∀ x, f(x) < 0 ↔ x < -1 ∨ x > 1/2) :
  ∀ x, f(10^x) > 0 ↔ x < -Real.log 2 := 
by 
  sorry

end solution_set_f_of_10_to_the_x_l318_318587


namespace value_range_f_l318_318755

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * real.exp x * (real.sin x + real.cos x)

theorem value_range_f :
  (set.range (λ (x : ℝ), f x) ∩ set.Icc 0 (real.pi / 2)) = 
  set.Icc (1 / 2) ((1 / 2) * real.exp (real.pi / 2)) :=
sorry

end value_range_f_l318_318755


namespace factor_polynomial_l318_318523

theorem factor_polynomial (x y : ℝ) : 
  x^4 + 4 * y^4 = (x^2 - 2 * x * y + 2 * y^2) * (x^2 + 2 * x * y + 2 * y^2) :=
by
  sorry

end factor_polynomial_l318_318523


namespace A_worked_days_l318_318979

theorem A_worked_days 
  (W : ℝ)                              -- Total work in arbitrary units
  (A_work_days : ℕ)                    -- Days A can complete the work 
  (B_work_days_remaining : ℕ)          -- Days B takes to complete remaining work
  (B_work_days : ℕ)                    -- Days B can complete the work alone
  (hA : A_work_days = 15)              -- A can do the work in 15 days
  (hB : B_work_days_remaining = 12)    -- B completes the remaining work in 12 days
  (hB_alone : B_work_days = 18)        -- B alone can do the work in 18 days
  :
  ∃ (x : ℕ), x = 5                     -- A worked for 5 days before leaving the job
  := 
  sorry                                 -- Proof not provided

end A_worked_days_l318_318979


namespace ratio_of_sums_is_1_l318_318300

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ) -- The common difference of the arithmetic sequence.

-- Condition: The sequence is an arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: The sum of the first n terms.
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Condition: Given ratio.
def given_ratio :=
  a 5 / a 3 = 5 / 9

-- Question: Prove that the ratio of sums is 1.
theorem ratio_of_sums_is_1 (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
    (h1 : is_arithmetic_sequence a d) 
    (h2 : sum_of_first_n_terms a S) 
    (h3 : given_ratio a) : 
    S 9 / S 5 = 1 :=
by
  sorry

end ratio_of_sums_is_1_l318_318300


namespace sector_area_l318_318745

theorem sector_area (r : ℝ) (alpha : ℝ) (h1 : r = 6) (h2 : alpha = π / 6) : 
  (1 / 2) * r^2 * alpha = 3 * π :=
by {
  rw [h1, h2],
  sorry
}

end sector_area_l318_318745


namespace trigonometric_identity_l318_318861

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318861


namespace cos_squared_difference_l318_318810

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318810


namespace equidistant_from_lines_l318_318766

theorem equidistant_from_lines
  (circle1 circle2 : Circle)
  (pointD : Point)
  (tangent_line : Line)
  (A B C : Point)
  (H_touch : CirclesTouchExternallyAt circle1 circle2 pointD)
  (H_tangent : TangentToCircleAt tangent_line circle1 A)
  (H_intersects : TangentIntersectsCircleAt tangent_line circle2 B C) :
  EquidistantFromLines A (LineThrough B pointD) (LineThrough C pointD) :=
by
  sorry

end equidistant_from_lines_l318_318766


namespace cos_squared_difference_l318_318927

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318927


namespace pie_left_after_shares_l318_318505

-- Definitions from conditions
def carlos_share : ℝ := 0.60
def maria_share : ℝ := 1 / 4

-- The mathematically equivalent proof problem in Lean 4
theorem pie_left_after_shares (whole_pie : ℝ) (h : whole_pie = 1) : 
  (whole_pie - carlos_share) * (1 - maria_share) = 0.30 := 
sorry

end pie_left_after_shares_l318_318505


namespace cos_difference_identity_l318_318935

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318935


namespace walk_to_bus_stop_time_l318_318425

theorem walk_to_bus_stop_time 
  (S T : ℝ)   -- Usual speed and time
  (D : ℝ)        -- Distance to bus stop
  (T'_delay : ℝ := 9)   -- Additional delay in minutes
  (T_coffee : ℝ := 6)   -- Coffee shop time in minutes
  (reduced_speed_factor : ℝ := 4/5)  -- Reduced speed factor
  (h1 : D = S * T)
  (h2 : D = reduced_speed_factor * S * (T + T'_delay - T_coffee)) :
  T = 12 :=
by
  sorry

end walk_to_bus_stop_time_l318_318425


namespace irrational_of_sqrt_5_l318_318482

theorem irrational_of_sqrt_5 (r1 r2 r3 r4 : ℝ) (h1 : r1 = 0.333333333333333333) (h2 : r2 = -22/7) (h3 : r3 = sqrt 5) (h4 : r4 = 0) : 
  irrational r3 ∧ rational r1 ∧ rational r2 ∧ rational r4 :=
by
  sorry

end irrational_of_sqrt_5_l318_318482


namespace fuel_tank_capacity_l318_318485

theorem fuel_tank_capacity
  (ethanol_A_fraction : ℝ)
  (ethanol_B_fraction : ℝ)
  (ethanol_total : ℝ)
  (fuel_A_volume : ℝ)
  (C : ℝ)
  (h1 : ethanol_A_fraction = 0.12)
  (h2 : ethanol_B_fraction = 0.16)
  (h3 : ethanol_total = 28)
  (h4 : fuel_A_volume = 99.99999999999999)
  (h5 : 0.12 * 99.99999999999999 + 0.16 * (C - 99.99999999999999) = 28) :
  C = 200 := 
sorry

end fuel_tank_capacity_l318_318485


namespace binomial_expansion_const_term_l318_318588

theorem binomial_expansion_const_term (a : ℝ) (h : a > 0) 
  (A : ℝ) (B : ℝ) :
  (A = (15 * a ^ 4)) ∧ (B = 15 * a ^ 2) ∧ (A = 4 * B) → B = 60 := 
by 
  -- The actual proof is omitted
  sorry

end binomial_expansion_const_term_l318_318588


namespace quadratic_two_distinct_real_roots_l318_318262

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end quadratic_two_distinct_real_roots_l318_318262


namespace problem_correct_statements_l318_318401

theorem problem_correct_statements:
  let stmt1 := ∅ = ({0} : Set ℕ)
  let stmt2 := ({1, 2, 3} : Set ℕ) = {3, 2, 1}
  let stmt3 := ({x | (x - 1)^2 * (x - 2) = 0} : Set ℝ) = {1, 1, 2}
  let stmt4 := ({x | 4 < x ∧ x < 5} : Set ℝ).Finite
  (¬stmt1 ∧ stmt2 ∧ ¬stmt3 ∧ ¬stmt4) = true :=
by
  sorry

end problem_correct_statements_l318_318401


namespace work_completion_days_l318_318789

theorem work_completion_days (A B : Type) (work : A → B → ℝ) (a_days b_days together_days : ℝ) 
  (hb : b_days = 30) (htogether : together_days = 10) :
  ∃ a_days, (1 / a_days + 1 / b_days = 1 / together_days) → a_days = 15 :=
by {
  use 15,
  intros h,
  rw [←h, ←inv_inj],
  field_simp,
  linarith,
  sorry
}

end work_completion_days_l318_318789


namespace exists_subset_sum_fifty_l318_318217

theorem exists_subset_sum_fifty (a : Fin 35 → ℕ) (h_pos : ∀ i, 0 < a i) (h_sum : (∑ i, a i) = 100) (h_bound : ∀ i, a i ≤ 50) :
  ∃ (s : Finset (Fin 35)), (∑ i in s, a i) = 50 :=
by
  sorry

end exists_subset_sum_fifty_l318_318217


namespace jill_time_to_run_up_and_down_l318_318334

noncomputable def time_to_run_up_and_down_hill : ℝ :=
  let base := 900.0
  let incline_angle := 30.0
  let speed_up := 9.0
  let speed_down := 12.0
  let cos30 := Math.cos (Real.pi / 6)
  let hypotenuse := base / cos30
  let distance_up := hypotenuse
  let distance_down := hypotenuse
  let time_up := distance_up / speed_up
  let time_down := distance_down / speed_down
  time_up + time_down

theorem jill_time_to_run_up_and_down :
  abs (time_to_run_up_and_down_hill - 202.07) < 0.01 :=
by
  sorry

end jill_time_to_run_up_and_down_l318_318334


namespace permutations_equal_combinations_l318_318282

theorem permutations_equal_combinations (n : ℕ) (h : nat.fact 3 = n*(n-1)*(n-2) /\ nat.fact 4 = nat.div (6 * (n*(n-1)*(n-2)*(n-3))) (4*3*2*1)) : n = 7 := 
by
  sorry

end permutations_equal_combinations_l318_318282


namespace nathaniel_best_friends_proof_l318_318371

-- Define initial assumptions
variables (initial_tickets remaining_tickets tickets_per_friend : ℕ)
variables (tickets_given_away : ℕ)
variables (num_best_friends : ℕ)

-- Given conditions
axiom initial_tickets_is : initial_tickets = 11
axiom remaining_tickets_is : remaining_tickets = 3
axiom tickets_per_friend_is : tickets_per_friend = 2

-- Define the number of tickets given away
def calculate_tickets_given_away : ℕ := initial_tickets - remaining_tickets
axiom tickets_given_away_is : calculate_tickets_given_away = tickets_given_away

-- Define the number of best friends
def calculate_num_best_friends : ℕ := tickets_given_away / tickets_per_friend
axiom num_best_friends_is : calculate_num_best_friends = num_best_friends

-- The goal is to prove that Nathaniel has 4 best friends
theorem nathaniel_best_friends_proof :
  initial_tickets = 11 →
  remaining_tickets = 3 →
  tickets_per_friend = 2 →
  tickets_given_away = 8 →
  num_best_friends = 4 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2] at tickets_given_away_is,
  rw h4,
  have correct_given_away : calculate_tickets_given_away = 8 := by sorry,
  rw correct_given_away,
  rw tickets_per_friend_is at h3,
  have correct_num_friends : calculate_num_best_friends = 4 := by sorry,
  rw correct_num_friends,
  exact h1,
end

end nathaniel_best_friends_proof_l318_318371


namespace henry_needs_30_dollars_l318_318275

def henry_action_figures_completion (current_figures total_figures cost_per_figure : ℕ) : ℕ :=
  (total_figures - current_figures) * cost_per_figure

theorem henry_needs_30_dollars : henry_action_figures_completion 3 8 6 = 30 := by
  sorry

end henry_needs_30_dollars_l318_318275


namespace lattice_square_sets_equality_l318_318355

theorem lattice_square_sets_equality (S : Type) [fintype S] [decidable_eq S] (T : finset (finset S)) 
  (A B C: ℕ) (hA : A = (S.to_finset.powerset.filter (λ p, disjoint p.to_finset) \ S.to_finset).card)
  (hB : B = (S.to_finset.powerset.filter (λ p, 2 ≤ p.card)).card)
  (hC : C = (S.to_finset.powerset.filter (λ p, 3 ≤ p.card)).card) :
  A = B + 2 * C :=
sorry

end lattice_square_sets_equality_l318_318355


namespace cos_square_difference_l318_318874

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318874


namespace sides_relation_l318_318723

-- We state the conditions using Lean definitions
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Condition that the quadratic has two equal roots
def equalRoots (A B C : ℝ) : Prop := 
  (4 * (sin B)^2 - 4 * sin A * sin C = 0)

-- The theorem to be proved
theorem sides_relation (A B C a b c : ℝ) (h : equalRoots A B C) : b^2 = a * c := 
sorry

end sides_relation_l318_318723


namespace find_remainder_when_q_divided_by_x_plus_2_l318_318072

noncomputable def q (x : ℝ) (D E F : ℝ) := D * x^4 + E * x^2 + F * x + 5

theorem find_remainder_when_q_divided_by_x_plus_2 (D E F : ℝ) :
  q 2 D E F = 15 → q (-2) D E F = 15 :=
by
  intro h
  sorry

end find_remainder_when_q_divided_by_x_plus_2_l318_318072


namespace find_length_AV_l318_318108

-- Define the centers of the circles and their radii
variables {A B U V C : Point}
variables {radiusA radiusB : ℝ}
variables {AU BU : ℝ}
variables {AB UV AV : ℝ}

-- The conditions
def conditions : Prop :=
  radiusA = 10 ∧ radiusB = 3 ∧ AB = radiusA + radiusB ∧ UV = 13 ∧ 
  AU = radiusA ∧ AV^2 = AU^2 + UV^2 ∧ UV^2 = 109

-- The theorem statement
theorem find_length_AV (h : conditions) : AV = Real.sqrt 209 :=
sorry

end find_length_AV_l318_318108


namespace cos_squared_difference_l318_318907

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318907


namespace problem_statement_l318_318679

variable (x y : ℝ)
def t : ℝ := x / y

theorem problem_statement (h : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 3) : t = 1 := by 
  sorry

end problem_statement_l318_318679


namespace solve_equation_l318_318193

theorem solve_equation (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (a > 0) → (b > 0) → (n > 0) → (a ^ 2013 + b ^ 2013 = p ^ n) ↔ 
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ p = 2 ∧ n = 2013 * k + 1 :=
by
  sorry

end solve_equation_l318_318193


namespace ratio_MS_SN_l318_318644

theorem ratio_MS_SN {A B C: Type*} (a b c: ℝ) (x: ℝ)
  (AB BC AC: ℝ) 
  (A1 B1 C1 L K M N S: Type*) 
  (h1: AB = 2 * x) 
  (h2: BC = 3 * x) 
  (h3: AC = 4 * x)
  (condition1: is_bisector A A1)
  (condition2: is_bisector B B1)
  (condition3: is_bisector C C1)
  (condition4: intersection L B1 C1 A A1)
  (condition5: intersection K B1 A1 C C1)
  (condition6: intersection M B K A A1)
  (condition7: intersection N B L C C1)
  (condition8: intersection S B B1 M N) : 
  MS / SN = 16 / 15 :=
sorry

end ratio_MS_SN_l318_318644


namespace correct_result_l318_318419

def polynomial1 : ℕ → Polynomial ℤ := λ a, (Polynomial.C (2 * a ^ 2) - (Polynomial.C (3 * a)) + Polynomial.C 5)
def polynomial2 : ℕ → Polynomial ℤ := λ a, (Polynomial.C (a ^ 2) - (Polynomial.C (5 * a)) + Polynomial.C 7)

theorem correct_result (a : ℕ) : polynomial1 a + polynomial1 a = Polynomial.C (5 * a ^ 2) - Polynomial.C (11 * a) + Polynomial.C 17 :=
by sorry

end correct_result_l318_318419


namespace train_overtake_time_l318_318423

noncomputable def time_to_pass 
  (length: ℝ) (speed_faster: ℝ) (speed_slower: ℝ) (conversion_factor: ℝ) : ℝ :=
  let relative_speed := (speed_faster - speed_slower) * conversion_factor
  let distance := 2 * length
  distance / relative_speed

theorem train_overtake_time 
  (length: ℝ := 75)
  (speed_faster: ℝ := 46)
  (speed_slower: ℝ := 36)
  (conversion_factor: ℝ := 5/18) :
  time_to_pass length speed_faster speed_slower conversion_factor ≈ 53.96 :=
by
  sorry

end train_overtake_time_l318_318423


namespace distinct_pairs_of_books_l318_318610

-- Define the total number of distinct books in each genre as described in the problem
def num_mystery_books : ℕ := 4
def num_fantasy_books : ℕ := 4
def num_biography_books : ℕ := 4

-- Define a statement to calculate the total number of distinct pairs
theorem distinct_pairs_of_books : 
  (num_fantasy_books * (num_mystery_books + num_biography_books)) = 32 :=
by
  have h_fantasy : num_fantasy_books = 4 := rfl
  have h_others : num_mystery_books + num_biography_books = 8 := rfl
  rw [h_fantasy, h_others]
  exact rfl

end distinct_pairs_of_books_l318_318610


namespace ratio_HD_HA_l318_318410

noncomputable def triangle_ratio (a b c : ℝ) (h : Point) (D : Point) (A : Point) : ℝ :=
  if (a = 8 ∧ b = 15 ∧ c = 17)
  then 
    let s := (a + b + c) / 2 in
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
    let h_AD := (2 * area) / b in
    if (h = A)
    then 0
    else h_AD / h_AD -- this placeholder shows the logical construct; the actual conditions simplify to ratio 0 : 1.
  else 0

theorem ratio_HD_HA (a b c : ℝ) (h D A : Point) :
  (a = 8 ∧ b = 15 ∧ c = 17) →
  (let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let h_AD := (2 * area) / b in
  let h := A in
  (0 = 0) / (h_AD = h_AD)) = (0 : 1) :=
by sorry

end ratio_HD_HA_l318_318410


namespace sector_circumradius_l318_318471

theorem sector_circumradius
  (r : ℝ) (θ : ℝ) (A B : Type) [HasDist A B] (O : A)
  (h_radius : r = 9)
  (h_angle : θ = 60) :
  ∃ R : ℝ, R = 6 * Real.sqrt 3 :=
by
  sorry

end sector_circumradius_l318_318471


namespace constant_term_in_expansion_is_17_l318_318527

noncomputable def constantTermExpansion := 
  (x^2 + 2) * (1/x - 1)^6

theorem constant_term_in_expansion_is_17 : 
  let term := constantTermExpansion
  constantTermExpansion == 17 := by
  sorry

end constant_term_in_expansion_is_17_l318_318527


namespace integer_ratio_condition_l318_318676

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end integer_ratio_condition_l318_318676


namespace elroy_more_miles_l318_318185

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l318_318185


namespace area_of_triangle_PF1F2_l318_318564

noncomputable def hyperbola (x y : ℝ) := (x^2)/2 - y^2 = 1
noncomputable def distance (a b : ℝ × ℝ) := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_PF1F2 :
  ∀ (P F1 F2 : ℝ × ℝ),
  hyperbola P.1 P.2 →
  distance P F1 + distance P F2 = 4 * real.sqrt 2 →
  ∃ (A : ℝ), A = real.sqrt 5 ∧
    A = 1 / 2 * distance P F1 * distance P F2 * real.sqrt (1 - (2 / 3)^2) :=
by
  intros P F1 F2 hP hDist
  -- Insert proof here
  sorry

end area_of_triangle_PF1F2_l318_318564


namespace calculate_expr_l318_318160

theorem calculate_expr : 1 - Real.sqrt 9 = -2 := by
  sorry

end calculate_expr_l318_318160


namespace calculate_planes_l318_318276

noncomputable def expected_flight_number (n : ℕ) : ℝ :=
  1 + (∑ j in finset.range (n - 1), (n.factorial / ((n - j).factorial * n^j)))

def number_of_airplanes : ℕ :=
  let eX := implicit eX ʺexpected_flight_number nʺ in 
  find (fun n : ℕ => real.abs (eX - 15) < pfrac 1 100)

theorem calculate_planes : number_of_airplanes = 134 :=
  sorry

end calculate_planes_l318_318276


namespace eggs_per_basket_l318_318648

theorem eggs_per_basket (red_eggs : ℕ) (orange_eggs : ℕ) (min_eggs : ℕ) :
  red_eggs = 30 → orange_eggs = 45 → min_eggs = 5 →
  (∃ k, (30 % k = 0) ∧ (45 % k = 0) ∧ (k ≥ 5) ∧ k = 15) :=
by
  intros h1 h2 h3
  use 15
  sorry

end eggs_per_basket_l318_318648


namespace crayon_production_correct_l318_318991

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l318_318991


namespace side_length_of_inscribed_square_l318_318329

theorem side_length_of_inscribed_square :
  ∀ (square_side : ℝ) (triangle_side : ℝ), 
  square_side = 15 →
  let s := 2 * (15 * real.sqrt 2 / (2 * real.sqrt 3)) in
  let y := (15 * real.sqrt 2 / 2 - s) / real.sqrt 2 in
  y = 15 * real.sqrt 3 / 6 :=
by
  intros square_side triangle_side h_square_side hs hy
  sorry

end side_length_of_inscribed_square_l318_318329


namespace find_f_of_1_l318_318241

theorem find_f_of_1 (a b : ℝ) (h_sym : ∀ x : ℝ, (x, x^3 + a * x^2 + b * x + 2) = (2 - x, 0)) : 
  (f : ℝ → ℝ) (f_def : ∀ x : ℝ, f x = x^3 + a * x^2 + b * x + 2),
  let a_val : a = -6 := sorry,
      b_val : b = 7 := sorry
  in f 1 = 4 :=
sorry

end find_f_of_1_l318_318241


namespace sum_of_first_12_terms_l318_318227

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_12_terms (a d : ℤ) (h1 : a + d * 4 = 3 * (a + d * 2))
                             (h2 : a + d * 9 = 14) : Sn a d 12 = 84 := 
by
  sorry

end sum_of_first_12_terms_l318_318227


namespace graph_shift_left_pi_over_4_l318_318097

theorem graph_shift_left_pi_over_4 :
  (∀ x, sin (3 * (x - π / 4)) = sin 3x + cos 3x) :=
sorry

end graph_shift_left_pi_over_4_l318_318097


namespace blood_expiration_date_l318_318481

-- Conditions
def twelve_factorial : ℕ := 12!

def seconds_in_a_day : ℕ := 86400

-- Question (Theorem/Goal)
theorem blood_expiration_date : 
  ∀ (donation_date : ℕ × ℕ × ℕ),
  (donation_date = (2023, 1, 5) ∧ twelve_factorial = 479001600 ∧ seconds_in_a_day = 86400) 
  → ∃ (expiration_date : ℕ × ℕ × ℕ), expiration_date = (2038, 2, 6) := 
by 
  sorry

end blood_expiration_date_l318_318481


namespace cos_squared_difference_l318_318944

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318944


namespace num_ways_not_adjacent_correct_num_ways_A_not_first_B_not_last_correct_num_ways_video_before_current_correct_l318_318982

section problem1
variable (songs speeches: Finset ℕ)

def num_ways_not_adjacent (songs speeches: Finset ℕ) : ℕ :=
  if songs.card = 5 ∧ speeches.card = 3 then
    14400
  else
    0

theorem num_ways_not_adjacent_correct : num_ways_not_adjacent songs speeches = 14400 :=
  sorry
end problem1

section problem2
variable (events : Finset ℕ)

def num_ways_A_not_first_B_not_last (events: Finset ℕ) : ℕ :=
  if events.card = 8 then
    30960
  else
    0

theorem num_ways_A_not_first_B_not_last_correct : num_ways_A_not_first_B_not_last events = 30960 :=
  sorry
end problem2

section problem3
variable (video_speech current_speech : ℕ)
(variable (other_events : Finset ℕ)

def num_ways_video_before_current (video_speech current_speech: ℕ) (other_events : Finset ℕ) : ℕ :=
  if other_events.card = 6 then
    20160
  else
    0

theorem num_ways_video_before_current_correct : num_ways_video_before_current video_speech current_speech other_events = 20160 :=
  sorry
end problem3

end num_ways_not_adjacent_correct_num_ways_A_not_first_B_not_last_correct_num_ways_video_before_current_correct_l318_318982


namespace black_squares_covered_by_trominoes_l318_318437

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

noncomputable def min_trominoes (n : ℕ) : ℕ :=
  ((n + 1) ^ 2) / 4

theorem black_squares_covered_by_trominoes (n : ℕ) (h1 : n ≥ 7) (h2 : is_odd n):
  ∀ n : ℕ, ∃ k : ℕ, k = min_trominoes n :=
by
  sorry

end black_squares_covered_by_trominoes_l318_318437


namespace cos_difference_squared_l318_318968

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318968


namespace find_eccentricity_of_hyperbola_l318_318260

variables (a b : ℝ) (x y : ℝ)
variables (F_1 F P M : ℝ × ℝ)
variablues (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
variables (h_positive_a : 0 < a)
variables (h_positive_b : 0 < b)
variables (h_right_focus : F = (c, 0))
variables (h_point_P : P = (-x, y))
variables (h_tangent_circle : ∀ {x y : ℝ}, P = (x, y) → x^2 + y^2 = a^2)
variables (h_midpoint_M : M = ((P.1 + F.1) / 2, (P.2 + F.2) / 2))
variables (h_equiv : F_1F = 2 * c)

theorem find_eccentricity_of_hyperbola :
  let e := c / a in
  e = sqrt 5 := 
sorry

end find_eccentricity_of_hyperbola_l318_318260


namespace min_value_proof_l318_318215

noncomputable def min_value (x y : ℝ) : ℝ := 1 / x + 1 / (2 * y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) :
  min_value x y = 4 :=
sorry

end min_value_proof_l318_318215


namespace cos_difference_identity_l318_318932

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318932


namespace vector_dot_product_l318_318271

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  let a := (3 : ℝ, Real.sqrt 3)
  let b := (1 : ℝ, 0 : ℝ)
  dot_product a b = 3 :=
by
  sorry

end vector_dot_product_l318_318271


namespace collinear_A_F_K_l318_318695

variables {A B C D K F : Type}

noncomputable def rhombus (A B C D : Type) := 
  -- definition of rhombus properties
  sorry

noncomputable def point_on_line (K : Type) (L : Type) := 
  -- definition of point on line
  sorry

noncomputable def collinear (P Q R : Type) := 
  -- definition of collinearity
  sorry

def condition1 (A B C D : Type) [rhombus A B C D] : Prop :=
  -- condition that ABCD is a rhombus
  sorry

def condition2 (A B C D : Type) (K : Type) [point_on_line K C D] (ad_eq_bk : K = A) : Prop :=
  -- condition that K is on CD and AD = BK
  sorry

def condition3 (B D : Type) (F : Type) := 
  -- condition that F is intersection of BD and the perpendicular bisector of BC
  sorry

theorem collinear_A_F_K : 
  condition1 A B C D → 
  condition2 A B C D K (sorry) → 
  condition3 B D F → 
  collinear A F K :=
sorry

end collinear_A_F_K_l318_318695


namespace coloring_methods_390_l318_318424

def numColoringMethods (colors cells : ℕ) (maxColors : ℕ) : ℕ :=
  if colors = 6 ∧ cells = 4 ∧ maxColors = 3 then 390 else 0

theorem coloring_methods_390 :
  numColoringMethods 6 4 3 = 390 :=
by 
  sorry

end coloring_methods_390_l318_318424


namespace pyramid_base_edge_length_l318_318720

noncomputable def edge_length_of_pyramid_base : ℝ :=
  let R := 4 -- radius of the hemisphere
  let h := 12 -- height of the pyramid
  let base_length := 6 -- edge-length of the base of the pyramid to be proved
  -- assume necessary geometric configurations of the pyramid and sphere
  base_length

theorem pyramid_base_edge_length :
  ∀ R h base_length, R = 4 → h = 12 → edge_length_of_pyramid_base = base_length → base_length = 6 :=
by
  intros R h base_length hR hH hBaseLength
  have R_spec : R = 4 := hR
  have h_spec : h = 12 := hH
  have base_length_spec : edge_length_of_pyramid_base = base_length := hBaseLength
  sorry

end pyramid_base_edge_length_l318_318720


namespace percentageOrangeJuiceInBlend_l318_318692

def MikiBasket : Type := { oranges : ℕ, pears : ℕ }

def juiceExtractionRate (fruit : String) (amount : ℕ) (units : ℕ) : ℚ :=
  if fruit = "pear" then
    amount / 5
  else if fruit = "orange" then
    amount / 4
  else
    0

def totalJuice (fruit : String) (count : ℕ) : ℚ :=
  if fruit = "pear" then
    count * (juiceExtractionRate "pear" 10 5)
  else if fruit = "orange" then
    count * (juiceExtractionRate "orange" 12 4)
  else
    0

theorem percentageOrangeJuiceInBlend (basket : MikiBasket)
  (pearJuicePerPear : ℚ := juiceExtractionRate "pear" 10 5)
  (orangeJuicePerOrange : ℚ := juiceExtractionRate "orange" 12 4)
  (usedPears : ℕ := 9)
  (usedOranges : ℕ := 6)
  (totalPearJuice : ℚ := totalJuice "pear" usedPears)
  (totalOrangeJuice : ℚ := totalJuice "orange" usedOranges)
  (totalJuice : ℚ := totalPearJuice + totalOrangeJuice) :
  (totalOrangeJuice / totalJuice) * 100 = 50 :=
by
  sorry

end percentageOrangeJuiceInBlend_l318_318692


namespace max_halls_visited_l318_318793

theorem max_halls_visited (side_len large_tri small_tri: ℕ) 
  (h1 : side_len = 100)
  (h2 : large_tri = 100)
  (h3 : small_tri = 10)
  (div : large_tri = (side_len / small_tri) ^ 2) :
  ∃ m : ℕ, m = 91 → m ≤ large_tri - 9 := 
sorry

end max_halls_visited_l318_318793


namespace total_charging_time_l318_318124

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l318_318124


namespace f_g_minus_g_f_l318_318668

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 3 * x^2 + 5

-- Proving the given math problem
theorem f_g_minus_g_f :
  f (g 2) - g (f 2) = 140 := by
sorry

end f_g_minus_g_f_l318_318668


namespace area_of_shaded_rectangle_l318_318769

-- Definition of side length of the squares
def side_length : ℕ := 12

-- Definition of the dimensions of the overlapped rectangle
def rectangle_length : ℕ := 20
def rectangle_width : ℕ := side_length

-- Theorem stating the area of the shaded rectangle PBCS
theorem area_of_shaded_rectangle
  (squares_identical : ∀ (a b c d p q r s : ℕ),
    a = side_length → b = side_length →
    p = side_length → q = side_length →
    rectangle_width * (rectangle_length - side_length) = 48) :
  rectangle_width * (rectangle_length - side_length) = 48 :=
by sorry -- Proof omitted

end area_of_shaded_rectangle_l318_318769


namespace carts_needed_each_day_last_two_days_l318_318057

-- Define capacities as per conditions
def daily_capacity_large_truck : ℚ := 1 / (3 * 4)
def daily_capacity_small_truck : ℚ := 1 / (4 * 5)
def daily_capacity_cart : ℚ := 1 / (20 * 6)

-- Define the number of carts required each day in the last two days
def required_carts_last_two_days : ℚ :=
  let total_work_done_by_large_trucks := 2 * daily_capacity_large_truck * 2
  let total_work_done_by_small_trucks := 3 * daily_capacity_small_truck * 2
  let total_work_done_by_carts := 7 * daily_capacity_cart * 2
  let total_work_done := total_work_done_by_large_trucks + total_work_done_by_small_trucks + total_work_done_by_carts
  let remaining_work := 1 - total_work_done
  remaining_work / (2 * daily_capacity_cart)

-- Assertion of the number of carts required
theorem carts_needed_each_day_last_two_days :
  required_carts_last_two_days = 15 := by
  sorry

end carts_needed_each_day_last_two_days_l318_318057


namespace correct_answer_is_B_l318_318246

open Real

def statement1 (f : ℝ → ℝ) (θ : ℝ) : Prop :=
  (∀ x, x ∈ Icc (-1) 1 → f (-x) = f x) ∧
  (∀ x y, x ∈ Icc (-1) 0 → y ∈ Icc (-1) 0 → x < y → f x < f y) ∧
  θ ∈ Ioo (π/4) (π/2) ∧
  ¬ (f (sin θ) > f (cos θ))

def statement2 (α β : ℝ) : Prop :=
  0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ cos α > sin β ∧ α + β < π/2

def statement3 (f : ℝ → ℝ) (k : ℤ) : Prop :=
  f = (λ x, 2 * sin (π/3 - 2*x) + 1) ∧
  (∀ x y, x ∈ Icc (k*π - π/12) (k*π + 5*π/12) → y ∈ Icc (k*π - π/12) (k*π + 5*π/12) → x < y → f x < f y)

def statement4 (x : ℝ) (k : ℤ) : Prop :=
  ¬ (x ≥ 5*π/6 + 2*k*π ∧ x ≤ 7*π/6 + 2*k*π) ↔ ¬ (cos (x + π/6) ≥ -sqrt 3/2)

def is_true (s : Prop) : Prop :=
  s

def number_of_true_statements : ℕ :=
  if is_true (∃ f θ, statement1 f θ) then 1 else 0 +
  if is_true (∃ α β, statement2 α β) then 1 else 0 +
  if is_true (∃ f k, statement3 f k) then 1 else 0 +
  if is_true (∃ x k, statement4 x k) then 1 else 0

theorem correct_answer_is_B : number_of_true_statements = 1 := sorry

end correct_answer_is_B_l318_318246


namespace simplify_fraction_of_decimal_l318_318081

theorem simplify_fraction_of_decimal :
  let n        := 3675
  let d        := 1000
  let gcd      := Nat.gcd n d
  n / gcd = 147 ∧ d / gcd = 40 → 
  (3675 / 1000 : ℚ) = (147 / 40 : ℚ) :=
by {
  sorry
}

end simplify_fraction_of_decimal_l318_318081


namespace nancy_hardwood_flooring_l318_318368

theorem nancy_hardwood_flooring :
  let central_area := 10 * 10
  let hallway := 6 * 4
  let l_shaped_section := 5 * 2
  let triangular_section := (3 * 3 * (1/2 : ℝ))
  (central_area + hallway + l_shaped_section + triangular_section) = 138.5 := 
by
  -- given conditions
  let central_area := 10 * 10
  let hallway := 6 * 4
  let l_shaped_section := 5 * 2
  let triangular_section := (3 * 3 * (1/2 : ℝ))
  -- we need to prove the total area is 138.5 square feet
  have : central_area = 100 := by norm_num
  have : hallway = 24 := by norm_num
  have : l_shaped_section = 10 := by norm_num
  have : triangular_section = 4.5 := by norm_num
  have : (central_area + hallway + l_shaped_section + triangular_section) = 100 + 24 + 10 + 4.5 := by norm_num [central_area, hallway, l_shaped_section, triangular_section]
  show 134.5 = 138.5
  sorry

end nancy_hardwood_flooring_l318_318368


namespace correct_statements_l318_318787

theorem correct_statements (
    A B : Prop)
  (X : ℝ → ℝ)
  (σ : ℝ)
  (h_dist : ∃ (μ = 1), ∀ x, X x = PDF_normal μ σ)
  (h_prob_X_gt_2 : P(X > 2) = 0.2) 
  (r : ℝ) 
  (h_corr_bound : -1 ≤ r ∧ r ≤ 1)
  (h_corr_strong : abs r → ℝ → ℝ ∈ set.Icc -1 1)
  (is_complementary : A ∧ B → ¬ (A ∧ B) ∧ A ∨ B = univ)
  (is_mutually_exclusive : A ∧ B → ¬ (A ∧ B)) :
  (is_complementary → is_mutually_exclusive) ∧
  h_prob_X_gt_2 ∧ P(0 < X < 1) = 0.3 ∧
  (∀ x y, (strong_corr : corr(Pair(x, y)) = r) → ∃ c, abs(r) = c ∧ c ∈ (set.Ico 0 1)) := sorry

end correct_statements_l318_318787


namespace cos_square_difference_l318_318875

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318875


namespace inverse_function_of_exp_l318_318036

theorem inverse_function_of_exp : 
  ∀ (y : ℝ), (1 ≤ y ∧ y < 3) → (∃ x : ℝ, -1 ≤ x ∧ x < 0 ∧ (y = 3^(x + 1))) → (y = -1 + real.log x / real.log 3) :=
by
  sorry

end inverse_function_of_exp_l318_318036


namespace max_triangles_convex_polygon_l318_318324

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end max_triangles_convex_polygon_l318_318324


namespace elroy_miles_difference_l318_318183

theorem elroy_miles_difference (earning_rate_last_year earning_rate_this_year total_collection_last_year : ℝ)
  (rate_last_year : earning_rate_last_year = 4)
  (rate_this_year : earning_rate_this_year = 2.75)
  (total_collected : total_collection_last_year = 44) :
  (total_collection_last_year / earning_rate_this_year) - (total_collection_last_year / earning_rate_last_year) = 5 :=
by
  rw [rate_last_year, rate_this_year, total_collected]
  norm_num
  sorry

end elroy_miles_difference_l318_318183


namespace cos_squared_difference_l318_318929

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318929


namespace construct_point_l318_318776

theorem construct_point (
  (O₁ O₂ : Type) 
  (M A B : Type) 
  (r R a : ℝ) 
  (O₁F O₁O₂ : ℝ)
  (h_r_lt_R : r < R)
  (h_AB_eq_a : dist A B = a)
  (h_O₁F_eq_half_a : O₁F = 0.5 * a)
  (h_O₁O₂_assign : dist O₁ O₂ = O₁O₂)
):
  ∃ M : Type, ∃ (l : Type), 
  (is_line l) ∧ 
  (is_parallel l (O₁F)) ∧ 
  (is_point_between M A B) 
  :=
sorry

end construct_point_l318_318776


namespace cos_square_difference_l318_318872

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318872


namespace smaller_square_area_l318_318141

theorem smaller_square_area (a p q : ℝ) (h : a > 0 ∧ p > 0 ∧ q > 0) :
  ∃ (s : ℝ), s = (a / (p + q)) * sqrt(p^2 + q^2) ∧
  s^2 = a^2 * (p^2 + q^2) / (p + q)^2 :=
by
  sorry

end smaller_square_area_l318_318141


namespace no_term_in_range_l318_318409

noncomputable def a : ℕ → ℕ
| 0     := 2
| 1     := 3
| (n+2) := if n > 0 then 2 * a n else 3 * a (n+1) - 2 * a n

theorem no_term_in_range (n : ℕ) : ¬ (1612 ≤ a n ∧ a n ≤ 2012) :=
by sorry

end no_term_in_range_l318_318409


namespace find_b_if_continuous_l318_318360

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

theorem find_b_if_continuous (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 23 / 2 :=
by
  sorry

end find_b_if_continuous_l318_318360


namespace cos_diff_square_identity_l318_318815

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318815


namespace other_diagonal_possible_lengths_l318_318010

theorem other_diagonal_possible_lengths
  (d_1 a c θ : ℝ)
  (h_area : 0.5 * d_1 * (64 / (d_1 * sin θ)) * sin θ = 32)
  (h_sum : d_1 + a + c = 16) :
  ∃ d_2 : ℝ, d_2 = 64 / (d_1 * sin θ) :=
begin
  sorry
end

end other_diagonal_possible_lengths_l318_318010


namespace wind_velocity_l318_318042

theorem wind_velocity (P A V : ℝ) (k : ℝ := 1/200) :
  (P = k * A * V^2) →
  (P = 2) → (A = 1) → (V = 20) →
  ∀ (P' A' : ℝ), P' = 128 → A' = 4 → ∃ V' : ℝ, V'^2 = 6400 :=
by
  intros h1 h2 h3 h4 P' A' h5 h6
  use 80
  linarith

end wind_velocity_l318_318042


namespace sin_double_angle_cos_sum_product_l318_318211

noncomputable def α : ℝ := sorry -- α exists with the given conditions

theorem sin_double_angle:
  (cos α = -√5/5) ∧ (π/2 < α ∧ α < π) →
  sin (2 * α) = -4/5 := sorry

theorem cos_sum_product:
  (cos α = -√5/5) ∧ (π/2 < α ∧ α < π) →
  cos (π/4 + α) * cos (α - 3 * π / 2) = 3 * √2 / 5 := sorry

end sin_double_angle_cos_sum_product_l318_318211


namespace nathaniel_best_friends_proof_l318_318372

-- Define initial assumptions
variables (initial_tickets remaining_tickets tickets_per_friend : ℕ)
variables (tickets_given_away : ℕ)
variables (num_best_friends : ℕ)

-- Given conditions
axiom initial_tickets_is : initial_tickets = 11
axiom remaining_tickets_is : remaining_tickets = 3
axiom tickets_per_friend_is : tickets_per_friend = 2

-- Define the number of tickets given away
def calculate_tickets_given_away : ℕ := initial_tickets - remaining_tickets
axiom tickets_given_away_is : calculate_tickets_given_away = tickets_given_away

-- Define the number of best friends
def calculate_num_best_friends : ℕ := tickets_given_away / tickets_per_friend
axiom num_best_friends_is : calculate_num_best_friends = num_best_friends

-- The goal is to prove that Nathaniel has 4 best friends
theorem nathaniel_best_friends_proof :
  initial_tickets = 11 →
  remaining_tickets = 3 →
  tickets_per_friend = 2 →
  tickets_given_away = 8 →
  num_best_friends = 4 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2] at tickets_given_away_is,
  rw h4,
  have correct_given_away : calculate_tickets_given_away = 8 := by sorry,
  rw correct_given_away,
  rw tickets_per_friend_is at h3,
  have correct_num_friends : calculate_num_best_friends = 4 := by sorry,
  rw correct_num_friends,
  exact h1,
end

end nathaniel_best_friends_proof_l318_318372


namespace trigonometric_identity_l318_318860

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318860


namespace power_of_four_l318_318204

-- Definition of the conditions
def prime_factors (x: ℕ): ℕ := 2 * x + 5 + 2

-- The statement we need to prove given the conditions
theorem power_of_four (x: ℕ) (h: prime_factors x = 33) : x = 13 :=
by
  -- Proof goes here
  sorry

end power_of_four_l318_318204


namespace cos_squared_difference_l318_318952

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318952


namespace cos_diff_square_identity_l318_318820

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318820


namespace cos_squared_difference_l318_318913

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318913


namespace calculate_expression_l318_318161

noncomputable def sqrt (x : ℝ) := real.sqrt x
noncomputable def sin (x : ℝ) := real.sin x
noncomputable def reciprocal (x : ℝ) := x⁻¹

theorem calculate_expression :
  sqrt 2 * sqrt 2 - 4 * sin (real.pi / 6) + reciprocal (1 / 2) = 2 :=
by
  have h1 : sqrt 2 * sqrt 2 = 2 := by sorry
  have h2 : sin (real.pi / 6) = 1 / 2 := by sorry
  have h3 : reciprocal (1 / 2) = 2 := by sorry
  sorry

end calculate_expression_l318_318161


namespace circumcircle_radius_eq_m_l318_318654

-- Definitions of the geometric objects and their properties
def is_isosceles (A B C : Point) : Prop := dist A C = dist B C
def midpoint (M A C : Point) : Prop := 2 * dist A M = dist A C
def perpendicular_to (Z : Line) (A B C : Point) : Prop := is_perpendicular Z (line_through A B)

-- The given conditions
variables (A B C M Q : Point) (Z : Line) (m : ℝ)
hypothesis h_isosceles : is_isosceles A B C
hypothesis h_midpoint : midpoint M A C
hypothesis h_perpendicular : perpendicular_to Z A B C
hypothesis h_circle : circle_through B C M ∩ Z = {C, Q}
hypothesis h_CQ : dist C Q = m 

-- The statement to prove
theorem circumcircle_radius_eq_m : 
  let R := circumradius A B C in
  R = m :=
sorry

end circumcircle_radius_eq_m_l318_318654


namespace find_p_l318_318796

/-- Given conditions about the coordinates of points on a line, we want to prove p = 3. -/
theorem find_p (m n p : ℝ) 
  (h1 : m = n / 3 - 2 / 5)
  (h2 : m + p = (n + 9) / 3 - 2 / 5) 
  : p = 3 := by 
  sorry

end find_p_l318_318796


namespace ordered_pair_solution_l318_318525

theorem ordered_pair_solution :
  ∃ x y : ℚ, (4 * x = -10 - 3 * y) ∧ (6 * x = 5 * y - 32) ∧ x = -73 / 19 ∧ y = 34 / 19 :=
by
  use -73 / 19, 34 / 19
  split
  sorry
  split
  sorry
  split
  rfl
  rfl

end ordered_pair_solution_l318_318525


namespace cos_difference_identity_l318_318941

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318941


namespace tangent_secant_theorem_l318_318699

-- We define the data and conditions as abstractions
variables {A B C K : Type} 
           [line : LineSegment A B] 
           [line2 : LineSegment A C] 
           [tangent : Tangent A K] 
           [circle : Circle]

-- We state the theorem based on the conditions given and required proof
theorem tangent_secant_theorem (A B C K : Point) (AK : Segment) 
                               (circle : Circle) 
                               (h_tangent : Tangent AK circle) 
                               (h_ray_intersect : Ray A intersects circle at [B, C]) : 
  (length AK) ^ 2 = (length (Segment A B)) * (length (Segment A C)) :=
by
  sorry

end tangent_secant_theorem_l318_318699


namespace pascal_first_20_rows_sum_l318_318279

/-- The number of elements in the nth row of Pascal's Triangle is n + 1. -/
def pascalRowCount (n : ℕ) : ℕ := n + 1

/-- The sum of the first 20 rows (0th to 19th) of Pascal's Triangle. -/
def pascalSum20Rows : ℕ := ∑ n in Finset.range 20, pascalRowCount n

/-- The total number of numbers in the first 20 rows of Pascal's Triangle equals 210. -/
theorem pascal_first_20_rows_sum : pascalSum20Rows = 210 :=
by
  sorry

end pascal_first_20_rows_sum_l318_318279


namespace area_T_l318_318343

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1], ![8, 3]]

def area (region : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_T' (T T' : Set (ℝ × ℝ))
  (hT : area T = 9)
  (hT' : T' = {p | ∃ q, q ∈ T ∧ p = (matrix_A.mul_vec q)}) :
  area T' = 153 :=
  sorry

end area_T_l318_318343


namespace box_max_volume_l318_318170

theorem box_max_volume (x : ℝ) (h1 : 0 < x) (h2 : x < 5) :
    (10 - 2 * x) * (16 - 2 * x) * x ≤ 144 :=
by
  -- The proof will be filled here
  sorry

end box_max_volume_l318_318170


namespace fraction_equation_correct_l318_318379

theorem fraction_equation_correct : (1 / 2 - 1 / 6) / (1 / 6009) = 2003 := by
  sorry

end fraction_equation_correct_l318_318379


namespace ellipse_circle_inequality_l318_318245

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end ellipse_circle_inequality_l318_318245


namespace right_angled_triangle_with_prime_circumradius_l318_318698

theorem right_angled_triangle_with_prime_circumradius
    (a b c : ℕ) (R : ℕ) (h₁ : ∃ (T : Triangle ℕ), T.side_lengths = (a, b, c))
    (h₂ : nat.prime R) (h₃ : T.circumradius = R) :
    (T.angle = 90) :=
begin
    sorry
end

end right_angled_triangle_with_prime_circumradius_l318_318698


namespace shaded_area_of_square_l318_318384

theorem shaded_area_of_square (ABCD : Type) [square ABCD]
  (side_length_ABCD : ∀ A B C D : ABCD, dist A B = 2)
  (M : midpoint A D) (N : midpoint B C) :
  area_of_shaded_region ABCD M N = 8 / 3 := 
sorry

end shaded_area_of_square_l318_318384


namespace max_subset_S_l318_318339

theorem max_subset_S (S : Finset ℕ) (h₁ : ∀ (x ∈ S) (y ∈ S), x ≠ y → (x + y) % 7 ≠ 0 ∧ (x * y) % 7 ≠ 0) :
  S.card ≤ 865 :=
sorry

end max_subset_S_l318_318339


namespace cos_square_difference_l318_318871

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318871


namespace shortest_chord_through_point_on_circle_l318_318515

theorem shortest_chord_through_point_on_circle :
  ∀ (M : ℝ × ℝ) (x y : ℝ),
    M = (3, 0) →
    x^2 + y^2 - 8 * x - 2 * y + 10 = 0 →
    ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -3 :=
by
  sorry

end shortest_chord_through_point_on_circle_l318_318515


namespace cos_squared_difference_l318_318906

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318906


namespace jeff_total_distance_l318_318333

def monday_speed : ℝ := 6
def monday_time : ℝ := 1
def monday_distance : ℝ := monday_speed * monday_time

def tuesday_speed : ℝ := 5
def tuesday_time : ℝ := 1
def tuesday_distance : ℝ := tuesday_speed * tuesday_time

def wednesday_speed : ℝ := 4
def wednesday_time : ℝ := 1
def wednesday_distance : ℝ := wednesday_speed * wednesday_time

def thursday_speed : ℝ := 3
def thursday_time : ℝ := 40.0 / 60.0
def thursday_distance : ℝ := thursday_speed * thursday_time

def friday_speed : ℝ := 8
def friday_time : ℝ := 70.0 / 60.0
def friday_distance : ℝ := friday_speed * friday_time

def weekly_total_distance : ℝ :=
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance

theorem jeff_total_distance :
  weekly_total_distance = 26.33 :=
by
  sorry

end jeff_total_distance_l318_318333


namespace triangle_perimeter_l318_318538

def point := ℝ × ℝ
def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perimeter (A B C : point) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  perimeter A B C = 16 := by
  -- define points
  let A : point := (2, 3)
  let B : point := (2, 9)
  let C : point := (6, 6)
  -- assert the statement and skip the proof using sorry
  show perimeter A B C = 16 from sorry

end triangle_perimeter_l318_318538


namespace value_of_expression_l318_318612

theorem value_of_expression (m : ℝ) 
  (h : m^2 - 2 * m - 1 = 0) : 3 * m^2 - 6 * m + 2020 = 2023 := 
by 
  /- Proof is omitted -/
  sorry

end value_of_expression_l318_318612


namespace integer_solutions_l318_318192

theorem integer_solutions (m n : ℤ) (h1 : m * (m + n) = n * 12) (h2 : n * (m + n) = m * 3) :
  (m = 4 ∧ n = 2) :=
by sorry

end integer_solutions_l318_318192


namespace parabola_focus_chord_length_l318_318222

theorem parabola_focus_chord_length (p : ℝ) (h₀ : 0 < p) (h₁ : ∃ A B : ℝ × ℝ, A ≠ B ∧ (y¹⁰ = k(x - p/2) ∧ y² = 2px) ∧ dist A B = 4) : (0 < p ∧ p < 2) :=
sorry

end parabola_focus_chord_length_l318_318222


namespace car_initial_value_l318_318765

theorem car_initial_value (dep_rate : ℤ) (t : ℕ) (final_value : ℤ) (h_dep : dep_rate = -1000) (h_t : t = 6) (h_final : final_value = 14000) :
  let init_value := final_value - dep_rate * t in
  init_value = 20000 :=
by
  sorry

end car_initial_value_l318_318765


namespace cos_diff_square_identity_l318_318826

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318826


namespace cost_for_custom_hats_l318_318646

def head_circumference_jack := 12

def head_circumference_charlie := (head_circumference_jack / 2) + 9

def head_circumference_bill := (2 / 3) * head_circumference_charlie

def head_circumference_maya := (head_circumference_jack + head_circumference_charlie) / 2

def head_circumference_thomas := (2 * head_circumference_bill) - 3

def cost_of_hat : ℕ → ℕ
| n := if n ≤ 15 then 15 else if n ≤ 22 then 20 else 25

def total_cost : ℕ :=
  cost_of_hat head_circumference_jack +
  cost_of_hat head_circumference_charlie +
  cost_of_hat head_circumference_bill +
  cost_of_hat head_circumference_maya +
  cost_of_hat head_circumference_thomas

theorem cost_for_custom_hats : total_cost = 80 := by
  sorry

end cost_for_custom_hats_l318_318646


namespace probability_of_correct_match_l318_318110

def total_arrangements (n : ℕ) : ℕ := n!

def correct_probability : ℚ :=
  let total_possible_matches := total_arrangements 3
  let correct_matches := 1
  (correct_matches : ℚ) / total_possible_matches

theorem probability_of_correct_match :
  correct_probability = 1 / 6 := by
  sorry

end probability_of_correct_match_l318_318110


namespace area_of_quadrilateral_l318_318700

-- Definitions
variables (A B C D : Type) [EucDiv A] [EucDiv B] [EucDiv C] [EucDiv D]
variables (AB BC AD DC AC : ℕ)

-- Conditions
def right_angles_at_B_and_D (B D : ℕ) : Prop :=
  B = 90 ∧ D = 90

def diagonal_AC (AC : ℕ) : Prop :=
  AC = 5

def distinct_integer_lengths (AB BC AD DC : ℕ) : Prop :=
  AB ≠ BC ∧ AB ≠ AD ∧ AB ≠ DC ∧ BC ≠ AD ∧ BC ≠ DC ∧ AD ≠ DC

def one_side_length_3 (AB AD : ℕ) : Prop :=
  AB = 3 ∨ AD = 3

-- Theorem
theorem area_of_quadrilateral (A B C D : Type) [EucDiv A] [EucDiv B] [EucDiv C] [EucDiv D]
  (AB BC AD DC AC : ℕ) (h1 : right_angles_at_B_and_D 90 90) (h2 : diagonal_AC AC)
  (h3 : distinct_integer_lengths AB BC AD DC) (h4 : one_side_length_3 AB AD) :
  (AB = 3 ∧ BC = 4 ∧ AD = 3 ∧ DC = 4 ∧ AC = 5) →
  ∃ area : ℕ, area = 12 :=
begin
  intro h,
  existsi 12,
  sorry
end

end area_of_quadrilateral_l318_318700


namespace sum_of_first_half_of_numbers_l318_318718

theorem sum_of_first_half_of_numbers 
  (avg_total : ℝ) 
  (total_count : ℕ) 
  (avg_second_half : ℝ) 
  (sum_total : ℝ)
  (sum_second_half : ℝ)
  (sum_first_half : ℝ) 
  (h1 : total_count = 8)
  (h2 : avg_total = 43.1)
  (h3 : avg_second_half = 46.6)
  (h4 : sum_total = avg_total * total_count)
  (h5 : sum_second_half = 4 * avg_second_half)
  (h6 : sum_first_half = sum_total - sum_second_half)
  :
  sum_first_half = 158.4 := 
sorry

end sum_of_first_half_of_numbers_l318_318718


namespace fraction_fliers_afternoon_l318_318436

theorem fraction_fliers_afternoon :
  ∀ (initial_fliers remaining_fliers next_day_fliers : ℕ),
    initial_fliers = 2500 →
    next_day_fliers = 1500 →
    remaining_fliers = initial_fliers - initial_fliers / 5 →
    (remaining_fliers - next_day_fliers) / remaining_fliers = 1 / 4 :=
by
  intros initial_fliers remaining_fliers next_day_fliers
  sorry

end fraction_fliers_afternoon_l318_318436


namespace count_integer_values_of_x_l318_318278

theorem count_integer_values_of_x :
  ∃! (x : ℤ), 0 ≤ x ∧ x ≤ 3 ∧ ( ∀ y, 0 ≤ y ∧ y ≤ 3 → y = x ) :=
begin
  sorry
end

end count_integer_values_of_x_l318_318278


namespace find_x_l318_318672

theorem find_x (x y z : ℕ) (hxy : x ≥ y) (hyz : y ≥ z)
    (h1 : x^2 - y^2 - z^2 + x * y = 3007)
    (h2 : x^2 + 3 * y^2 + 3 * z^2 - 2 * x * y - 3 * x * z - 3 * y * z = -2013) : x = 25 :=
  by
  sorry

end find_x_l318_318672


namespace sports_league_game_count_l318_318473

theorem sports_league_game_count :
  ∀ (teams divisions : Nat) 
    (teams_per_division games_per_intra_division_game games_per_inter_division_game : Nat),
  teams = 12 →
  divisions = 3 →
  teams_per_division = 4 →
  games_per_intra_division_game = 3 →
  games_per_inter_division_game = 1 →
  (divisions * teams_per_division = teams) →
  let intra_division_games_per_team := (teams_per_division - 1) * games_per_intra_division_game in
  let inter_division_games_per_team := (teams - teams_per_division) * games_per_inter_division_game in
  let total_games_per_team := intra_division_games_per_team + inter_division_games_per_team in
  let initial_total_games := teams * total_games_per_team in
  let final_total_games := initial_total_games / 2 in
  final_total_games = 102 :=
by
  intros
  sorry

end sports_league_game_count_l318_318473


namespace oranges_group_count_l318_318054

theorem oranges_group_count (total_oranges groups_size number_of_groups : ℕ) 
    (h1 : total_oranges = 384) 
    (h2 : groups_size = 24) 
    (h3 : total_oranges = groups_size * number_of_groups) : 
    number_of_groups = 16 := 
by
  rw [h1, h2] at h3
  have : 384 = 24 * number_of_groups := h3
  have : number_of_groups = 384 / 24 := by
    rw [Nat.mul_div_cancel_left 384 (by dec_trivial : 0 < 24)]
  rw this
  norm_num
  done

end oranges_group_count_l318_318054


namespace smallest_is_C_l318_318761

def A : ℚ := 1/2
def B : ℚ := 9/10
def C : ℚ := 2/5

theorem smallest_is_C : min (min A B) C = C := 
by
  sorry

end smallest_is_C_l318_318761


namespace sugar_needed_for_one_third_batch_l318_318976

theorem sugar_needed_for_one_third_batch : (1/3 : ℚ) * (10/3 : ℚ) = 10/9 := by
  calc
    (1/3 : ℚ) * (10/3 : ℚ) = (10/9 : ℚ) : by sorry

lemma mixed_number_conversion (frac : ℚ) (hn : frac = 10/9) : 1 + 1/9 = frac := sorry

example : ∃ mixed : ℚ, mixed = 1 + 1 / 9 ∧ mixed = (1 / 3 : ℚ) * (3 + 1 / 3 : ℚ) := by
  use (10/9 : ℚ)
  constructor
  . sorry
  . sorry

end sugar_needed_for_one_third_batch_l318_318976


namespace square_perimeter_l318_318011

noncomputable def side_length_square (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_square (area : ℝ) : ℝ :=
  4 * side_length_square area

theorem square_perimeter (area : ℝ) (h : area = 200) : perimeter_square area = 40 * real.sqrt 2 := 
by
  rw [perimeter_square, side_length_square, h]
  norm_num
  sorry

end square_perimeter_l318_318011


namespace equilateral_triangle_l318_318031

variable (A B C A₀ B₀ C₀ : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]

variable (midpoint : ∀ (X₁ X₂ : Type), Type) 
variable (circumcircle : ∀ (X Y Z : Type), Type)

def medians_meet_circumcircle := ∀ (A A₁ B B₁ C C₁ : Type) 
  [AddGroup A] [AddGroup A₁] [AddGroup B] [AddGroup B₁] [AddGroup C] [AddGroup C₁], 
  Prop

def areas_equal := ∀ (ABC₀ AB₀C A₀BC : Type) 
  [AddGroup ABC₀] [AddGroup AB₀C] [AddGroup A₀BC], 
  Prop

theorem equilateral_triangle (A B C A₀ B₀ C₀ A₁ B₁ C₁ : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]
  [AddGroup A₁] [AddGroup B₁] [AddGroup C₁] 
  (midpoint_cond : ∀ (X Y Z : Type), Z = midpoint X Y)
  (circumcircle_cond : ∀ (X Y Z : Type), Z = circumcircle X Y Z)
  (medians_meet_circumcircle : Prop)
  (areas_equal: Prop) :
    A = B ∧ B = C ∧ C = A :=
  sorry

end equilateral_triangle_l318_318031


namespace cosine_difference_identity_l318_318843

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318843


namespace does_not_balance_l318_318777

variables (square odot circ triangle O : ℝ)

-- Conditions represented as hypothesis
def condition1 : Prop := 4 * square = odot + circ
def condition2 : Prop := 2 * circ + odot = 2 * triangle

-- Statement to be proved
theorem does_not_balance (h1 : condition1 square odot circ) (h2 : condition2 circ odot triangle)
 : ¬(2 * triangle + square = triangle + odot + square) := 
sorry

end does_not_balance_l318_318777


namespace charging_time_is_correct_l318_318126

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l318_318126


namespace complex_fraction_simplification_l318_318387

theorem complex_fraction_simplification : (5 : ℂ) / (2 - (1 : ℂ) * complex.I) = (2 : ℂ) + (1 : ℂ) * complex.I :=
by sorry

end complex_fraction_simplification_l318_318387


namespace find_functions_l318_318191

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (p q : ℝ), p ≠ q → (f q - f p) / (q - p) * 0 + f p - (f q - f p) / (q - p) * p = p * q

theorem find_functions (f : ℝ → ℝ) (c : ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = x * (c + x)) :=
by
  intros
  sorry

end find_functions_l318_318191


namespace findPerpendicularLine_l318_318100

-- Defining the condition: the line passes through point (-1, 2)
def pointOnLine (x y : ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Defining the condition: the line is perpendicular to 2x - 3y + 4 = 0
def isPerpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

-- The original line equation: 2x - 3y + 4 = 0
def originalLine (x y : ℝ) : Prop :=
  2 * x - 3 * y + 4 = 0

-- The target equation of the line: 3x + 2y - 1 = 0
def targetLine (x y : ℝ) : Prop :=
  3 * x + 2 * y - 1 = 0

theorem findPerpendicularLine :
  (pointOnLine (-1) 2 3 2 (-1)) ∧
  (isPerpendicular 3 2 2 (-3)) →
  (∀ x y, targetLine x y ↔ 3 * x + 2 * y - 1 = 0) :=
by
  sorry

end findPerpendicularLine_l318_318100


namespace sum_of_cubes_eq_twice_product_of_roots_l318_318195

theorem sum_of_cubes_eq_twice_product_of_roots (m : ℝ) :
  (∃ a b : ℝ, (3*a^2 + 6*a + m = 0) ∧ (3*b^2 + 6*b + m = 0) ∧ (a ≠ b)) → 
  (a^3 + b^3 = 2 * a * b) → 
  m = 6 :=
by
  intros h_exists sum_eq_twice_product
  sorry

end sum_of_cubes_eq_twice_product_of_roots_l318_318195


namespace find_z_l318_318390

theorem find_z (y z : ℝ) (k : ℝ) (h1 : y^3 * z^2 = k) (h2 : (3 : ℝ)^3 * (2 : ℝ)^2 = 108) (h3 : k = 108) (h4 : y = 6) : 
  z = sqrt(2) / 2 :=
by
  sorry

end find_z_l318_318390


namespace daily_serving_ratio_l318_318062

noncomputable def servings_in_bottle : ℕ := 6
noncomputable def price_per_bottle : ℝ := 3.00
noncomputable def total_cost : ℝ := 21.00
noncomputable def weeks : ℕ := 2
noncomputable def days_in_week : ℕ := 7
noncomputable def total_days : ℕ := weeks * days_in_week

theorem daily_serving_ratio :
  let bottles := total_cost / price_per_bottle,
      total_servings := bottles * servings_in_bottle,
      daily_servings := total_servings / total_days in
  daily_servings / servings_in_bottle = 1 / 2 :=
by sorry

end daily_serving_ratio_l318_318062


namespace pi_over_2_irrational_l318_318076

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem pi_over_2_irrational : is_irrational (Real.pi / 2) :=
by sorry

end pi_over_2_irrational_l318_318076


namespace sin_product_obtuse_cos_half_angle_product_obtuse_l318_318643

noncomputable def obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π/2 ∨ B > π/2 ∨ C > π/2)

theorem sin_product_obtuse (A B C : ℝ) (h : obtuse_triangle A B C) (hC : C > π/2) :
  sin A * sin B * sin C < 1/2 :=
sorry

theorem cos_half_angle_product_obtuse (A B C : ℝ) (h : obtuse_triangle A B C) :
  cos (A/2) * cos (B/2) * cos (C/2) < (1 + sqrt 2) / 4 :=
sorry

end sin_product_obtuse_cos_half_angle_product_obtuse_l318_318643


namespace rhombus_perimeter_l318_318024

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h3 : d1 / 2 ≠ 0) (h4 : d2 / 2 ≠ 0) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * s = 52 :=
by
  sorry

end rhombus_perimeter_l318_318024


namespace fraction_of_time_to_cover_distance_l318_318980

-- Definitions for the given conditions
def distance : ℝ := 540
def initial_time : ℝ := 12
def new_speed : ℝ := 60

-- The statement we need to prove
theorem fraction_of_time_to_cover_distance :
  ∃ (x : ℝ), (x = 3 / 4) ∧ (distance / (initial_time * x) = new_speed) :=
by
  -- Proof steps would go here
  sorry

end fraction_of_time_to_cover_distance_l318_318980


namespace positive_roots_sign_changes_negative_roots_sign_changes_l318_318439

-- Definitions for the problem parameters
def polynomial (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldr (λ a b => a + x * b) 0

def sign_changes (coeffs : List ℝ) : ℕ :=
  coeffs.sliding 2 |>.filter (λ L => L.head! * L.head!.tail != 0 ∧ L.head! * L.head!.tail < 0) |>.length

-- Statement for positive roots (Part a)
theorem positive_roots_sign_changes (n : ℕ) (coeffs : List ℝ) (h : coeffs.length = n + 1) (hn : coeffs.last! ≠ 0) :
  (count_positive_roots (polynomial coeffs) ≤ sign_changes (coeffs)) := sorry

-- Statement for negative roots (Part b)
theorem negative_roots_sign_changes (n : ℕ) (coeffs : List ℝ) (h : coeffs.length = n + 1) (hn : coeffs.last! ≠ 0) :
  (count_negative_roots (polynomial (coeffs.map_with_index (λ i a => (-1) ^ (n - i) * a))) ≤ sign_changes (coeffs.map_with_index (λ i a => (-1) ^ (n - i) * a))) := sorry

end positive_roots_sign_changes_negative_roots_sign_changes_l318_318439


namespace projection_of_a_onto_b_is_2_sqrt_5_l318_318552

noncomputable def vector_a : ℝ × ℝ := (-3, 4)
noncomputable def vector_b : ℝ × ℝ := (-2, 1)

def projection (a b : ℝ × ℝ) : ℝ := 
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_onto_b_is_2_sqrt_5 :
  projection vector_a vector_b = 2 * Real.sqrt 5 := 
sorry

end projection_of_a_onto_b_is_2_sqrt_5_l318_318552


namespace stickers_on_first_page_l318_318500

theorem stickers_on_first_page :
  ∀ (a b c d e : ℕ), 
    (b = 16) →
    (c = 24) →
    (d = 32) →
    (e = 40) →
    (b - a = 8) →
    (c - b = 8) →
    (d - c = 8) →
    (e - d = 8) →
    a = 8 :=
by
  intros a b c d e hb hc hd he h1 h2 h3 h4
  -- Proof would go here
  sorry

end stickers_on_first_page_l318_318500


namespace problem_proof_l318_318320

open Classical
open Real

noncomputable def isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = sqrt 5 ∧ dist A C = sqrt 5

-- Assume we have points satisfying the conditions. We don't specify coordinates because 
-- it's not necessary for defining the conditions, only their properties matter.
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions for the Lean statement
def conditions : Prop :=
  isosceles_triangle A B C ∧
  D ∈  set.Icc (min (B.1) (C.1)) (max (B.1) (C.1)) ∧ 
  D ≠ (B.1 + C.1) / 2 ∧
  E = reflect_point A D C ∧
  ∃ P, P ∈ line (E, B) ∧ intersection (line (A, D)) (line (E, B)) = some F

-- State the theorem
theorem problem_proof (h : conditions A B C D E F) : dist A D * dist A F = 5 := 
  sorry  -- proof is not required as per instructions

end problem_proof_l318_318320


namespace cos_squared_difference_l318_318926

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318926


namespace new_hours_per_week_l318_318281

/-- 
  Given that:
  - initial_hours: initially planned working 15 hours per week.
  - initial_weeks: initially planned 10 weeks of work.
  - missed_weeks: missed 3 weeks due to emergency.
  - total_money: total money to earn = 2250.

  Prove that the new required working hours per week is 21.
-/
theorem new_hours_per_week (initial_hours : ℕ) (initial_weeks : ℕ) (missed_weeks : ℕ) (total_money : ℕ) :
  initial_hours = 15 ∧ initial_weeks = 10 ∧ missed_weeks = 3 ∧ total_money = 2250 → 
  let weeks_left := initial_weeks - missed_weeks in
  let new_hours := (initial_weeks * initial_hours + weeks_left - 1) / weeks_left in
  new_hours = 21 :=
by
  intros
  sorry

end new_hours_per_week_l318_318281


namespace sin_alpha_sub_pi_div_six_tan_double_alpha_l318_318234

namespace MathProofs

theorem sin_alpha_sub_pi_div_six (α : ℝ) (h1 : Real.sin α = (4 / 5)) (h2 : α ∈ Ioo (π / 2) π) :
  Real.sin (α - π / 6) = (4 * Real.sqrt 3 + 3) / 10 :=
  sorry

theorem tan_double_alpha (α : ℝ) (h1 : Real.sin α = (4 / 5)) (h2 : α ∈ Ioo (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 :=
  sorry

end MathProofs

end sin_alpha_sub_pi_div_six_tan_double_alpha_l318_318234


namespace area_of_triangle_XYZ_l318_318637

noncomputable def hypotenuse := 8 * Real.sqrt 2
noncomputable def leg := 8

theorem area_of_triangle_XYZ (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (XY : ℝ) 
  (right_triangle : ∀ a b c d : ℝ, a^2 + b^2 = c^2)
  (isosceles_right_triangle : ∀ a b : ℝ, a = b)
  (hypotenuse_length : XY = hypotenuse)
  : 
  Θ :=
begin
  sorry
end

end area_of_triangle_XYZ_l318_318637


namespace tangent_circle_equality_l318_318109

variable (A B C D : Type)
variable [AffineSpace ℝ A]
variable [MetricSpace A]

noncomputable def isTangent (r : ℝ) (α β γ : A) : Prop := sorry

theorem tangent_circle_equality
  (ABCD_is_convex : ∀ A B C D : A, isConvexQuartrel ABCD)
  (r1 r2 r3 r4 : ℝ)
  (AB BC CD DA : ℝ)
  (h1 : isTangent r1 D A B)
  (h2 : isTangent r2 A B C)
  (h3 : isTangent r3 B C D)
  (h4 : isTangent r4 C D A)
  (hAB : AB = r1 * (cot (angle A (B - A)) / 2 + cot (angle B (A - B)) / 2))
  (hCD : CD = r3 * (cot (angle C (D - C)) / 2 + cot (angle D (C - D)) / 2))
  (hBC : BC = r2 * (cot (angle B (C - B)) / 2 + cot (angle C (D - C)) / 2))
  (hAD : AD = r4 * (cot (angle A (D - A)) / 2 + cot (angle D (A - D)) / 2))
  : (AB / r1 + CD / r3) = (BC / r2 + AD / r4) :=
  sorry

end tangent_circle_equality_l318_318109


namespace cosine_difference_identity_l318_318844

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318844


namespace cos_squared_difference_l318_318830

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318830


namespace cosine_difference_identity_l318_318842

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318842


namespace count_dracula_is_alive_l318_318716

variable (P Q : Prop)
variable (h1 : P)          -- I am human
variable (h2 : P → Q)      -- If I am human, then Count Dracula is alive

theorem count_dracula_is_alive : Q :=
by
  sorry

end count_dracula_is_alive_l318_318716


namespace max_x_add_inv_x_l318_318757

variable (x : ℝ) (y : Fin 2022 → ℝ)

-- Conditions
def sum_condition : Prop := x + (Finset.univ.sum y) = 2024
def reciprocal_sum_condition : Prop := (1/x) + (Finset.univ.sum (λ i => 1 / (y i))) = 2024

-- The statement we need to prove
theorem max_x_add_inv_x (h_sum : sum_condition x y) (h_rec_sum : reciprocal_sum_condition x y) : 
  x + (1/x) ≤ 2 := by
  sorry

end max_x_add_inv_x_l318_318757


namespace cos_squared_difference_l318_318879

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318879


namespace max_number_of_triangles_l318_318326

theorem max_number_of_triangles (num_sides : ℕ) (num_internal_points : ℕ) 
    (total_points : ℕ) (h1 : num_sides = 13) (h2 : num_internal_points = 200) 
    (h3 : total_points = num_sides + num_internal_points) 
    (h4 : ∀ (x y z : point), x ≠ y ∧ y ≠ z ∧ z ≠ x → ¬ collinear x y z) : 
    (total_points.choose 3) = 411 :=
by
  sorry

end max_number_of_triangles_l318_318326


namespace probability_of_heart_and_joker_l318_318063

-- Define a deck with 54 cards, including jokers
def total_cards : ℕ := 54

-- Define the count of specific cards in the deck
def hearts_count : ℕ := 13
def jokers_count : ℕ := 2
def remaining_cards (x: ℕ) : ℕ := total_cards - x

-- Define the probability of drawing a specific card
def prob_of_first_heart : ℚ := hearts_count / total_cards
def prob_of_second_joker (first_card_a_heart: Bool) : ℚ :=
  if first_card_a_heart then jokers_count / remaining_cards 1 else 0

-- Calculate the probability of drawing a heart first and then a joker
def prob_first_heart_then_joker : ℚ :=
  prob_of_first_heart * prob_of_second_joker true

-- Proving the final probability
theorem probability_of_heart_and_joker :
  prob_first_heart_then_joker = 13 / 1419 := by
  -- Skipping the proof
  sorry

end probability_of_heart_and_joker_l318_318063


namespace centroid_of_tetrahedron_is_correct_l318_318313

-- Define the points A, B, C, D
def A : ℝ × ℝ × ℝ := (1, 2, 3)
def B : ℝ × ℝ × ℝ := (5, 3, 1)
def C : ℝ × ℝ × ℝ := (3, 4, 5)
def D : ℝ × ℝ × ℝ := (4, 5, 2)

-- The expected centroid G
def G_expected : ℝ × ℝ × ℝ := (13/4, 7/2, 11/4)

-- Define a proof problem to prove the centroid is as expected
theorem centroid_of_tetrahedron_is_correct : 
  let G := ((A.1 + B.1 + C.1 + D.1) / 4,
            (A.2 + B.2 + C.2 + D.2) / 4,
            (A.3 + B.3 + C.3 + D.3) / 4) in
  G = G_expected :=
by
  let G := ((A.1 + B.1 + C.1 + D.1) / 4,
            (A.2 + B.2 + C.2 + D.2) / 4,
            (A.3 + B.3 + C.3 + D.3) / 4)
  show G = G_expected
  sorry

end centroid_of_tetrahedron_is_correct_l318_318313


namespace find_x_value_l318_318167

open Real

theorem find_x_value (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * log x) = c) : 
  x = 10^((5 / c - a^2) / b) := 
by 
  sorry

end find_x_value_l318_318167


namespace cos_squared_difference_l318_318922

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318922


namespace trigonometric_identity_l318_318857

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318857


namespace quadratic_real_roots_find_m_l318_318263

open Real

theorem quadratic_real_roots (m : ℝ) :
  ∃ α β : ℝ, (α + β = -(m + 3)) ∧ (α * β = m + 1) :=
begin
  let a := 1,
  let b := m + 3,
  let c := m + 1,
  have discriminant_nonneg : b^2 - 4 * a * c ≥ 0,
  { calc
      (m + 3)^2 - 4 * 1 * (m + 1)
        = m^2 + 2 * m + 5 :
          by ring,
    show m^2 + 2 * m + 5 ≥ 0,
    let delta := m^2 + 2 * m + 5,
    exact add_nonneg (add_nonneg (sq_nonneg m) (mul_nonneg zero_le_two (le_of_lt zero_lt_two))) (by norm_num)
  },
  existsi [(-b + sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - sqrt (b^2 - 4 * a * c)) / (2 * a)],
  split,
  { field_simp, ring },
  { field_simp, ring }
end

theorem find_m (α β : ℝ) (m : ℝ) (hαβ : (α - β = 2 * sqrt 2)) :
  (m = -3 ∨ m = 1) :=
begin
  have h1 : α + β = -(m + 3),
  have h2 : α * β = m + 1,
  let sum_prod := (α + β)^2 - 4 * (α * β),
  rw [h1, h2, hαβ] at sum_prod,
  calc
    (-(m + 3))^2 - 4 * (m + 1) = (2 * sqrt 2)^2 :
      by ring,
  show (m + 3)^2 - 4 * (m + 1) = 8,
  let lhs := (m + 3)^2 - 4 * (m + 1),
  have h3 : lhs = 8,
  { calc
      lhs
        = (m+3)^2 - 4 * (m+1) : by ring
        = m^2 + 2 * m - 3 : by ring_nf,
    solve_by_elim
  },
  solve_quadratic
end

end quadratic_real_roots_find_m_l318_318263


namespace f_x_add_1_is_odd_l318_318402

variables (ω : ℝ) (ϕ : ℝ)
def f (x : ℝ) : ℝ := Real.cos (ω * x + ϕ)

axiom ω_pos : ω > 0
axiom f_at_1 : f ω ϕ 1 = 0

theorem f_x_add_1_is_odd : ∀ x, f ω ϕ (x + 1) = - f ω ϕ (-x - 1) :=
sorry

end f_x_add_1_is_odd_l318_318402


namespace problem_conditions_l318_318253

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)

theorem problem_conditions (x : ℝ) (s t : ℝ) (hx : 0 < x) (hs : 0 < s) (ht : 0 < t) : 
  (f 0 = 0) ∧ 
  (f x > 0) ∧
  (∀ x : ℝ, 0 ≤ x → f'.x ≥ 0) ∧
  (f (s + t) > f s + f t) :=
by
  sorry

end problem_conditions_l318_318253


namespace rhombus_perimeter_l318_318029

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l318_318029


namespace cos_squared_difference_l318_318885

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318885


namespace average_math_score_of_class_l318_318759

theorem average_math_score_of_class (n : ℕ) (jimin_score jung_score avg_others : ℕ) 
  (h1 : n = 40) 
  (h2 : jimin_score = 98) 
  (h3 : jung_score = 100) 
  (h4 : avg_others = 79) : 
  (38 * avg_others + jimin_score + jung_score) / n = 80 :=
by sorry

end average_math_score_of_class_l318_318759


namespace area_of_rectangle_l318_318093

-- Definitions from the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l b : ℝ) : Prop := l = 3 * b
def perimeter (P l b : ℝ) : Prop := P = 2 * (l + b)

-- The main theorem we are proving
theorem area_of_rectangle (b l : ℝ) (P : ℝ) (h1 : breadth b) (h2 : length l b) (h3 : perimeter P l b) (h4 : P = 96) : l * b = 432 := 
by
  -- Proof steps will go here
  sorry

end area_of_rectangle_l318_318093


namespace cos_square_difference_l318_318873

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318873


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l318_318251

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp (2 * a * x)

noncomputable def g (x : ℝ) (k : ℝ) (b : ℝ) := k * x + b

variables {a b k x x1 x2 : ℝ}
variables (h₀ : f 1 a = Real.exp 1)
variables (h₁ : g x k b = k * x + b)
variables (h₂ : ∀ x > 0, f x a > g x k b)
variables (h₃ : Real.exp x1 = k * x1 ∧ Real.exp x2 = k * x2 ∧ x1 < x2)

axiom problem1 : a = 1 / 2
axiom problem2 : b = 0
axiom problem3 : k ∈ Set.Ioo (-Real.exp 0) (Real.exp 1)
axiom problem4 : x1 * x2 < 1

theorem proof_problem1 : f 1 a = Real.exp 1 → a = 1 / 2 :=
sorry

theorem proof_problem2 : (∀ x, g x k b = -g (-x) k b) → b = 0 :=
sorry

theorem proof_problem3 : (∀ x > 0, f x a > g x k b) → k < Real.exp 1 :=
sorry

theorem proof_problem4 : Real.exp x1 = k * x1 ∧ Real.exp x2 = k * x2 ∧ x1 < x2 → x1 * x2 < 1 :=
sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l318_318251


namespace trigonometric_identity_l318_318858

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318858


namespace alpha_value_l318_318233

noncomputable def alpha (x : ℝ) := Real.arccos x

theorem alpha_value (h1 : Real.cos α = -1/6) (h2 : 0 < α ∧ α < Real.pi) : 
  α = Real.pi - alpha (1/6) :=
by
  sorry

end alpha_value_l318_318233


namespace cos_squared_difference_l318_318928

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318928


namespace decimal_to_fraction_l318_318079

theorem decimal_to_fraction (x : ℚ) (h : x = 3.675) : x = 147 / 40 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l318_318079


namespace trapezoid_perimeter_l318_318016

theorem trapezoid_perimeter (BC AD AC : ℝ) (angle_AOB : ℝ) (h1 : BC = 3) (h2 : AD = 5) (h3 : AC = 8) (h4 : angle_AOB = 60) : 
  (BC + AD + (sqrt (AC^2 + (AD - BC)^2 - 2 * AC * (AD - BC) * cos (real.pi / 3))) + (sqrt (AC^2 + (AD - BC)^2 - 2 * AC * (AD - BC) * cos (real.pi / 3)))) = 22 :=
by
  sorry

end trapezoid_perimeter_l318_318016


namespace inequality1_inequality2_l318_318005

variable (x : ℝ)

-- Definition of conditions for the problem
def condition1 : Prop := x^2 - 5 * x - 6 < 0

def condition2 : Prop := (x - 1) / (x + 2) ≤ 0

-- Define the solution sets for given conditions
def solution1 : Prop := -1 < x ∧ x < 6

def solution2 : Prop := -2 < x ∧ x ≤ 1

-- Proof statements
theorem inequality1 : condition1 → solution1 :=
by
  intro h
  sorry

theorem inequality2 : condition2 → solution2 :=
by
  intro h
  sorry

end inequality1_inequality2_l318_318005


namespace find_m_l318_318581

theorem find_m (m : ℂ) (i : ℂ) (hi : i * i = -1) : (1 - m * i) / (i ^ 3) = 1 + i → m = 1 := by
  intros h1
  have h2 : i ^ 3 = -i, from by
    rw [pow_succ, pow_two, hi]
    ring
  rw [h2] at h1
  have h3 : (1 - m * i) / (-i) = 1 + i, from h1
  have h4 : (1 - m * i) * i / (-i ^ 2) = (m + i), from by
    have : -i ^ 2 = 1, from by
      rw [pow_two, hi, neg_neg]
    rw [neg_eq_neg_one_mul, mul_assoc, this, div_div_eq_div_mul]
    ring
  rw [h2] at h4
  have h5 : m + i = 1 + i := by
    rw [this]
  exact sub_eq_zero.mp (congr_arg (λ z, z - i) h5)

end find_m_l318_318581


namespace triangle_perimeter_l318_318539

def point := ℝ × ℝ
def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perimeter (A B C : point) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  perimeter A B C = 16 := by
  -- define points
  let A : point := (2, 3)
  let B : point := (2, 9)
  let C : point := (6, 6)
  -- assert the statement and skip the proof using sorry
  show perimeter A B C = 16 from sorry

end triangle_perimeter_l318_318539


namespace monthly_growth_rate_price_reduction_l318_318454

/-- Part 1: Monthly average growth rate for July and August --/
theorem monthly_growth_rate 
    (sales_june : ℕ) 
    (sales_august : ℕ) 
    (constant_price : ℝ) 
    : sales_june = 256 → sales_august = 400 → constant_price ≥ 0 →  
      ∃ (x : ℝ), x = 0.25 :=
begin 
  intros h1 h2 h3,
  use 0.25,
  sorry
end

/-- Part 2: Price reduction for profit of 4250 yuan in September --/
theorem price_reduction 
    (initial_price : ℝ) 
    (cost_price : ℝ) 
    (sales_august : ℕ) 
    (increment_per_yuan : ℕ) 
    (target_profit : ℝ) 
    : initial_price = 40 → cost_price = 25 → sales_august = 400 → increment_per_yuan = 5 → target_profit = 4250 →  
      ∃ (reduction : ℝ), reduction = 5 :=
begin
  intros h_initial h_cost h_sales h_increment h_profit,
  use 5,
  sorry
end

end monthly_growth_rate_price_reduction_l318_318454


namespace minimum_k_condition_l318_318551

def is_acute_triangle (a b c : ℕ) : Prop :=
  a * a + b * b > c * c

def any_subset_with_three_numbers_construct_acute_triangle (s : Finset ℕ) : Prop :=
  ∀ t : Finset ℕ, t.card = 3 → 
    (∃ a b c : ℕ, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ 
      is_acute_triangle a b c ∨
      is_acute_triangle a c b ∨
      is_acute_triangle b c a)

theorem minimum_k_condition (k : ℕ) :
  (∀ s : Finset ℕ, s.card = k → any_subset_with_three_numbers_construct_acute_triangle s) ↔ (k = 29) :=
  sorry

end minimum_k_condition_l318_318551


namespace proportion_of_face_cards_l318_318520

theorem proportion_of_face_cards (p : ℝ) (h : 1 - (1 - p)^3 = 19 / 27) : p = 1 / 3 :=
sorry

end proportion_of_face_cards_l318_318520


namespace inverse_function_less_than_zero_l318_318728

theorem inverse_function_less_than_zero (x : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2^x + 1) (h₂ : ∀ y, f (f⁻¹ y) = y) (h₃ : ∀ y, f⁻¹ (f y) = y) :
  {x | f⁻¹ x < 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end inverse_function_less_than_zero_l318_318728


namespace proof_problem_l318_318900

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318900


namespace turnover_increase_l318_318048

theorem turnover_increase (a : ℝ) : 
  let increase_rate := 0.15 in
  let new_turnover := (1 + increase_rate) * a in
  new_turnover = 1.15 * a :=
sorry

end turnover_increase_l318_318048


namespace factorial_simplification_l318_318070

theorem factorial_simplification :
  Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 728 := 
sorry

end factorial_simplification_l318_318070


namespace distance_between_intersections_is_6sqrt2_l318_318017

/-- Define the circle equation x^2 + y^2 = 25. -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- Define the line equation y = x + 3. -/
def line (x y : ℝ) : Prop := y = x + 3

/-- Prove that the distance between the intersection points of the line and the circle
    is 6 * sqrt(2). -/
theorem distance_between_intersections_is_6sqrt2 :
  ∀ x1 y1 x2 y2 : ℝ, circle x1 y1 → circle x2 y2 → line x1 y1 → line x2 y2 → 
  x1 ≠ x2 → y1 ≠ y2 → (real.dist (x1, y1) (x2, y2)) = 6 * real.sqrt 2 := 
by
  intros x1 y1 x2 y2 h_circle1 h_circle2 h_line1 h_line2 h_x h_y
  sorry

end distance_between_intersections_is_6sqrt2_l318_318017


namespace elroy_more_miles_l318_318188

-- Given conditions
def last_year_rate : ℝ := 4
def this_year_rate : ℝ := 2.75
def last_year_collection : ℝ := 44

-- Goals
def last_year_miles : ℝ := last_year_collection / last_year_rate
def this_year_miles : ℝ := last_year_collection / this_year_rate
def miles_difference : ℝ := this_year_miles - last_year_miles

theorem elroy_more_miles :
  miles_difference = 5 := by
  sorry

end elroy_more_miles_l318_318188


namespace y_intercept_is_correct_l318_318045

-- Define the slope of the line
def slope : ℝ := 3

-- Define the x-intercept of the line
def x_intercept : ℝ × ℝ := (2, 0)

-- The y-intercept proof problem
theorem y_intercept_is_correct (slope_eq : slope = 3) (x_int_eq : x_intercept = (2, 0)) : 
  ∃ y_intercept, y_intercept = (0, 6) :=
sorry

end y_intercept_is_correct_l318_318045


namespace log_ineq_min_a_l318_318624

noncomputable def min_a : ℝ := 1 / real.exp 2

theorem log_ineq_min_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → real.log x ≤ a * x + 1) ↔ a ≥ min_a :=
by
  sorry

end log_ineq_min_a_l318_318624


namespace number_of_solutions_congruence_l318_318531
open BigOperators

def odd_prime (p : ℕ) : Prop := p > 2 ∧ ∀ n, n ∣ p → n = 1 ∨ n = p

theorem number_of_solutions_congruence (p : ℕ) (a : ℤ) :
  odd_prime p →
  ∃ N, N = (p + (-1) ^ ((p - 1) / 2)) ^ 2 ∧
  ∀ x y z : ℤ, x * y * z ∈ finset.range p →
  (x^2 + y^2 + z^2) % p = (2 * a * x * y * z) % p →
  x ∈ finset.range p ∧ y ∈ finset.range p ∧ z ∈ finset.range p → (N = (p + (-1) ^ ((p - 1) / 2)) ^ 2) := 
sorry

end number_of_solutions_congruence_l318_318531


namespace train_speed_l318_318135

theorem train_speed (train_length bridge_length : ℕ) (time_sec : ℕ) 
  (h_train_length : train_length = 140)
  (h_bridge_length : bridge_length = 235)
  (h_time_sec : time_sec = 30) : 
  let total_distance := train_length + bridge_length in
  let speed_m_per_s := total_distance / time_sec in
  let conversion_factor := 3.6 in
  let speed_km_per_hr := speed_m_per_s * conversion_factor in
  speed_km_per_hr = 45 :=
by
  sorry

end train_speed_l318_318135


namespace digit_root_of_fib2012_l318_318208

def digit_root_mod_9 (n : ℕ) : ℕ := if n % 9 = 0 then 9 else n % 9

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

lemma fibonacci_modulo_9 (n : ℕ) : (fibonacci n) % 9 :=
nat.rec_on n
  0 
  (λ n ih, 
    if h : n ≤ 10 then 
      -- here you need to add computed fibonacci numbers modulo 9 up to 2012th
      sorry
    else 
      sorry 
  )

theorem digit_root_of_fib2012 : digit_root_mod_9 (fibonacci 2012) = 6 := 
  sorry

end digit_root_of_fib2012_l318_318208


namespace cos_square_difference_l318_318867

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318867


namespace cross_section_perimeter_l318_318297

-- Define the lengths of the diagonals AC and BD.
def length_AC : ℝ := 8
def length_BD : ℝ := 12

-- Define the perimeter calculation for the cross-section quadrilateral
-- that passes through the midpoint E of AB and is parallel to BD and AC.
theorem cross_section_perimeter :
  let side1 := length_AC / 2
  let side2 := length_BD / 2
  let perimeter := 2 * (side1 + side2)
  perimeter = 20 :=
by
  sorry

end cross_section_perimeter_l318_318297


namespace exists_monochromatic_triangle_in_K17_l318_318180

theorem exists_monochromatic_triangle_in_K17 :
  ∀ (K17 : Type) [fintype K17] [decidable_eq K17], (graph.complete K17) →
  (coloring : function K17 K17 (sym2 K17) (fin 3)) →
  ∃ (v1 v2 v3 : K17), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
  (coloring (v1, v2) = coloring (v2, v3) ∧ coloring (v2, v3) = coloring (v1, v3)) := sorry

end exists_monochromatic_triangle_in_K17_l318_318180


namespace triangle_circle_intersection_l318_318479

noncomputable def AB_length : ℝ :=
  sqrt 10 - 1

theorem triangle_circle_intersection (
  (XA YC ZE : ℝ) (r : ℝ) (AB CD EF : ℝ)) :
  r = 2 → 
  XA = 1 →
  YC = 2 →
  ZE = 3 →
  AB = CD →
  CD = EF →
  AB = AB_length :=
by sorry

end triangle_circle_intersection_l318_318479


namespace selling_price_of_cycle_l318_318993

theorem selling_price_of_cycle (CP : ℕ) (gain_percent : ℕ) (h1 : CP = 900) (h2 : gain_percent = 20) : 
  let profit := (gain_percent / 100) * CP in
  let SP := CP + profit in
  SP = 1080 :=
by
  sorry

end selling_price_of_cycle_l318_318993


namespace sum_of_four_cubes_l318_318386

theorem sum_of_four_cubes : 
  ∃ (a b c d : ℤ), a = 325 ∧ b = 323 ∧ c = -324 ∧ d = -324 ∧ 1944 = a^3 + b^3 + c^3 + d^3 := 
by {
  use [325, 323, -324, -324],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  sorry
}

end sum_of_four_cubes_l318_318386


namespace projection_matrix_pu2_to_pu1_l318_318665

variable (u0 : ℝ × ℝ)

def proj_matrix (a : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ x, let aaT := (Matrix.vecMul' ![a.1, a.2] ![a.1, a.2]) in
         let aTa := (a.1 * a.1 + a.2 * a.2 : ℝ) in
         aaT.map (λ e, e / aTa) • x

theorem projection_matrix_pu2_to_pu1 :
  let a := (2, 2) in
  let b := (2, -1) in
  let P1 := proj_matrix a in
  let P2 := proj_matrix b in
  P2 (P1 u0) = ((2 / 5, 2 / 5), (-1 / 5, -1 / 5)) :=
sorry

end projection_matrix_pu2_to_pu1_l318_318665


namespace percentage_left_l318_318971

theorem percentage_left (P0 : ℕ) (Pd : ℝ) (Pf : ℕ) : 
  P0 = 8515 → Pd = 0.10 → Pf = 6514 → 
  ((P0 - round (Pd * P0 : ℝ)) - Pf) / (P0 - round (Pd * P0 : ℝ) : ℝ) = 0.1501 := by
  intros h1 h2 h3
  have h4 : (round (Pd * P0) : ℕ) = 851 := by sorry
  have h5 : (P0 - round (Pd * P0) : ℕ) = 7664 := by 
    rw [h1, h4]
    sorry
  have h6 : ((P0 - round (Pd * P0)) - Pf) = 1150 := by 
    rw [h1, h4, h3]
    sorry
  have h7 : ((P0 - round (Pd * P0) - Pf) : ℝ) / (P0 - round (Pd * P0) : ℝ) = 0.1501:= by
    rw [h5, h6]
    sorry
  exact h7

end percentage_left_l318_318971


namespace b_share_is_correct_l318_318138

-- Definitions and conditions based on problem statement
def share_at_end_of_year (x : ℕ) (profit_6m : ℕ) (total_profit : ℕ) (d_fixed_share : ℕ) : ℕ :=
  let a_invest := 2.5 * x
  let c_invest := 1.5 * x
  let d_invest := 1.25 * x
  let total_invest := 5 * x
  let remaining_profit_6m := profit_6m - d_fixed_share
  let b_share_6m := (x / total_invest) * remaining_profit_6m
  let remaining_profit_year := total_profit - 2 * d_fixed_share
  let b_share_year := (x / total_invest) * remaining_profit_year
  b_share_6m + b_share_year

-- Given the conditions to be used later
def x : ℕ := 1000 -- Some example value as x isn't specified to compute share directly
def profit_6m : ℕ := 6000
def total_profit : ℕ := 16900
def d_fixed_share : ℕ := 500

-- Statement to prove
theorem b_share_is_correct : 
  share_at_end_of_year x profit_6m total_profit d_fixed_share = 3180 :=
by sorry

end b_share_is_correct_l318_318138


namespace probability_one_is_three_times_other_l318_318210

def set := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def total_selections := set.choose 2
def desired_selections := [(1, 3), (2, 6), (3, 9)] -- Representing the pairs directly

theorem probability_one_is_three_times_other :
  (desired_selections.length : ℚ) / (total_selections.length : ℚ) = 1 / 12 :=
by 
  sorry

end probability_one_is_three_times_other_l318_318210


namespace proper_k_coloring_of_graph_l318_318563

noncomputable def e : ℝ := Real.exp 1

theorem proper_k_coloring_of_graph 
  (G : Type) [Graph G] 
  (k : ℕ) (hk : k > 1) 
  (h_edge_cycles : ∀ (e : Edge G), simple_cycle_count G e < ⌊e * (Nat.factorial (k - 1))⌋) 
  : ∃ (coloring : G → Fin k), ∀ (u v : G), (u ∼ v) → coloring u ≠ coloring v := sorry

end proper_k_coloring_of_graph_l318_318563


namespace problem1_problem2_l318_318248

noncomputable def a_seq : ℕ → ℝ
| 0 := 1/3
| (n+1) := a_seq n + (a_seq n ^ 2) / (n + 1) ^ 2

def S (n : ℕ) : ℝ :=
(n + 1) * a_seq n

theorem problem1 (d : ℝ) (h1 : 2 * (1 / 3) + d = 3) : d = 7 / 3 :=
by sorry

theorem problem2 (n : ℕ) : n > 0 → (n : ℝ) / (2 * n + 1) ≤ a_seq n ∧ a_seq n < 1 :=
by sorry

end problem1_problem2_l318_318248


namespace crayon_production_correct_l318_318990

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l318_318990


namespace simplify_expression_l318_318713

-- Define general term for y
variable (y : ℤ)

-- Statement representing the given proof problem
theorem simplify_expression :
  4 * y + 5 * y + 6 * y + 2 = 15 * y + 2 := 
sorry

end simplify_expression_l318_318713


namespace sum_possible_cookies_l318_318363

theorem sum_possible_cookies : ∑ n in {n | n < 100 ∧ n % 9 = 3 ∧ n % 5 = 2}.toFinset, n = 69 :=
by
  sorry

end sum_possible_cookies_l318_318363


namespace proof_problem_l318_318898

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318898


namespace ellipse_ratio_sum_l318_318510

open Real

theorem ellipse_ratio_sum :
  let ellipse := λ x y,  3 * x ^ 2 + 2 * x * y + 4 * y ^ 2 - 15 * x - 25 * y + 55 = 0
  let ratio_values := {m : ℝ | ∃ x y : ℝ, ellipse x y ∧ y = m * x}
  ratio_values ≠ ∅ ∧ ratio_values ⊆ {m : ℝ | ∃ m_max m_min, m_max = Sup ratio_values ∧ m_min = inf ratio_values}
  → ∀ m_max m_min, (max ratio_values = m_max ∧ min ratio_values = m_min) → m_max + m_min = 26 / 51 :=
begin
  sorry
end

end ellipse_ratio_sum_l318_318510


namespace tan_double_angle_l318_318596

def f (x : ℝ) : ℝ := sin x - cos x

theorem tan_double_angle (x : ℝ) (h : deriv f x = 2 * f x) : 
  tan (2 * x) = - (3 / 4) := 
by 
  -- Start the problem from the hypotheses
  have h1 : deriv f x = cos x + sin x := by sorry
  have h2 : 2 * f x = 2 * (sin x - cos x) := by sorry
  have : cos x + sin x = 2 * (sin x - cos x) := by
    rw [<- h1, <- h]
    sorry
  have : 3 * cos x = sin x := by sorry
  have : tan x = 3 := by sorry
  have : tan (2 * x) = (2 * tan x) / (1 - tan x ^ 2) := by sorry
  rw this 
  have : (2 * 3) / (1 - 3 ^ 2) = - (3 / 4) := by
    norm_num 
    sorry
  some assumption  
  rw some_nil no clash case_eq.refl<src>none<endcopy>
  -- Complete the theorem
  sorry

end tan_double_angle_l318_318596


namespace parabola_equation_triangle_area_AOB_l318_318221

-- Define the parabola with given conditions
noncomputable def parabola_through_vertex_and_focus (A : ℝ × ℝ) (a b : ℝ) : Prop :=
  -- Define the vertex at the origin (0,0) and focus on x-axis
  A = (1, 2) ∧ a > 0 ∧ b > 0

-- Define the statement to prove the equation of the parabola
theorem parabola_equation (A : ℝ × ℝ) (a b : ℝ) :
  parabola_through_vertex_and_focus A a b →
  (∃ p : ℝ, (p > 0) ∧ ( ∀ (x y : ℝ), y^2 = 2*p*x )) :=
sorry

-- Define the statement to prove the area of ΔAOB
theorem triangle_area_AOB (A B O : ℝ × ℝ) (slope : ℝ) (F : ℝ × ℝ) (eq_C : ∀ (x y : ℝ), y^2 = 4*x) :
  A = (1, 2) ∧ B = (5, -2) ∧ O = (0, 0) ∧ slope = 1 ∧ F = (1, 0) ∧
  (∀ (x y : ℝ), eq_C x y) →
  let
    AB := 8
    d := sqrt(2) / 2
  in
  (area : ℝ) = (1/2) * AB * d :=
sorry

end parabola_equation_triangle_area_AOB_l318_318221


namespace rotate_graph_90_deg_l318_318725

theorem rotate_graph_90_deg (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  ∀ x, (∃ y, x = f(-y)) ↔ ∃ y, y = -Function.invFun f x :=
by
  sorry

end rotate_graph_90_deg_l318_318725


namespace find_rate_percent_l318_318427

def rate_percent (P SI T : ℝ) : ℝ :=
  (SI * 100) / (P * T)

theorem find_rate_percent (P SI T : ℝ) (hP : P = 1200) (hSI : SI = 400) (hT : T = 4) :
  rate_percent P SI T = 8.3333 :=
by
  rw [rate_percent, hP, hSI, hT]
  norm_num
  sorry

end find_rate_percent_l318_318427


namespace smallest_k_for_inequality_l318_318205

def num_prime_divisors (n : ℕ) : ℕ := 
  -- Use a placeholder implementation, actual implementation can be replaced with proper logic.
  sorry

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ n : ℕ, 2^(num_prime_divisors n) ≤ k * (n^(1/4 : ℝ))) ∧ k = 5 :=
begin
  sorry
end

end smallest_k_for_inequality_l318_318205


namespace KT_TM_ratio_LN_length_l318_318128

-- Define the context and conditions
variables (K L M N T : Point) -- Points making up the quadrilateral and the intersection
variables (KL LM MN NK : Line) -- Sides of the quadrilateral
variables (KM LN : Line) -- Diagonals of the quadrilateral
variable (circ : Circle) -- Circumcircle of the quadrilateral

-- Conditions
axiom quad_inscribed : isInscribedInCircle K L M N circ
axiom diag_intersect : intersect K M L N T
axiom perp_distances : distanceFromPointToLine T KL = 4 * sqrt 2 ∧ 
                       distanceFromPointToLine T LM = sqrt 2 ∧ 
                       distanceFromPointToLine T MN = 8 / sqrt 17 ∧ 
                       distanceFromPointToLine T NK = 8 / sqrt 17

-- Given KM length
variable (KM_len : Real)
axiom KM_length : KM_len = 10

-- Conclusion statements to be proved
theorem KT_TM_ratio : KT / TM = 4 := sorry
theorem LN_length : LN = 50 / sqrt 34 := sorry

end KT_TM_ratio_LN_length_l318_318128


namespace cos_squared_difference_l318_318881

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318881


namespace share_difference_l318_318147

variables {x : ℕ}

theorem share_difference (h1: 12 * x - 7 * x = 5000) : 7 * x - 3 * x = 4000 :=
by
  sorry

end share_difference_l318_318147


namespace multiples_of_4_between_5th_and_8th_l318_318367

noncomputable def count_multiples_between (m n : ℕ) : ℕ :=
  let multiples := List.range' 4 96 (λ x, x*4)
  let fifth_from_left := multiples[4]
  let eighth_from_right := multiples[multiples.length - 8]
  (eighth_from_right - fifth_from_left) / 4 - 1

theorem multiples_of_4_between_5th_and_8th : 
  count_multiples_between 5 8 = 11 := 
  by sorry

end multiples_of_4_between_5th_and_8th_l318_318367


namespace cosine_difference_identity_l318_318847

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318847


namespace quadrilateral_parallelogram_iff_l318_318696

variable (a b c d e f MN : ℝ)

-- Define a quadrilateral as a structure with sides and diagonals 
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the condition: sum of squares of diagonals equals sum of squares of sides
def sum_of_squares_condition (q : Quadrilateral) : Prop :=
  q.e ^ 2 + q.f ^ 2 = q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2

-- Define what it means for a quadrilateral to be a parallelogram:
-- Midpoints of the diagonals coincide (MN = 0)
def is_parallelogram (q : Quadrilateral) (MN : ℝ) : Prop :=
  MN = 0

-- Main theorem to prove
theorem quadrilateral_parallelogram_iff (q : Quadrilateral) (MN : ℝ) :
  is_parallelogram q MN ↔ sum_of_squares_condition q :=
sorry

end quadrilateral_parallelogram_iff_l318_318696


namespace simplify_fraction_of_decimal_l318_318082

theorem simplify_fraction_of_decimal :
  let n        := 3675
  let d        := 1000
  let gcd      := Nat.gcd n d
  n / gcd = 147 ∧ d / gcd = 40 → 
  (3675 / 1000 : ℚ) = (147 / 40 : ℚ) :=
by {
  sorry
}

end simplify_fraction_of_decimal_l318_318082


namespace cos_squared_difference_l318_318812

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318812


namespace segment_eq_quarter_perimeter_l318_318996

variable (A B C D O P Q : Point)
variable (AB CD AD BC : ℝ)
variable (P_perimeter : ℝ)

-- Given conditions
def isosceles_trapezoid (AB CD AD BC: ℝ) : Prop := 
  AB + CD = AD + BC

def inscribed_circle (ISO_T: Prop) : Prop := 
  ∃ O: Point, O.inside (circle A B C D) ∧ ISO_T

def midline_segment_eqn (PQ AB CD : ℝ) : Prop := 
  PQ = (AB + CD) / 2

def perimeter_eqn (P AB CD AD BC : ℝ) : Prop := 
  P = AB + CD + AD + BC

-- Prove PQ == P / 4 given conditions
theorem segment_eq_quarter_perimeter
  (iso_t: isosceles_trapezoid AB CD AD BC) 
  (inc: inscribed_circle iso_t) 
  (pq_eq: midline_segment_eqn PQ AB CD) 
  (p_eq: perimeter_eqn P_perimeter AB CD AD BC) :
  PQ = P_perimeter / 4 :=
sorry

end segment_eq_quarter_perimeter_l318_318996


namespace side_length_of_inscribed_square_l318_318330

theorem side_length_of_inscribed_square :
  ∀ (square_side : ℝ) (triangle_side : ℝ), 
  square_side = 15 →
  let s := 2 * (15 * real.sqrt 2 / (2 * real.sqrt 3)) in
  let y := (15 * real.sqrt 2 / 2 - s) / real.sqrt 2 in
  y = 15 * real.sqrt 3 / 6 :=
by
  intros square_side triangle_side h_square_side hs hy
  sorry

end side_length_of_inscribed_square_l318_318330


namespace proof_problem_l318_318892

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318892


namespace cos_squared_difference_l318_318909

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318909


namespace calories_per_serving_is_120_l318_318462

-- Define the conditions
def servings : ℕ := 3
def halfCalories : ℕ := 180
def totalCalories : ℕ := 2 * halfCalories

-- Define the target value
def caloriesPerServing : ℕ := totalCalories / servings

-- The proof goal
theorem calories_per_serving_is_120 : caloriesPerServing = 120 :=
by 
  sorry

end calories_per_serving_is_120_l318_318462


namespace isosceles_triangle_condition_D_l318_318432

-- Definitions and theorems regarding isosceles triangles and angle properties
theorem isosceles_triangle_condition_D :
  (∃ (A B C : ℝ), 2*A = 2*B ∧ A + B + C = 180 ∧ 2*A = B ∧ A = B ∧ (A / B) = (B / C)) → 
  (∃ (A B C : ℝ), A = B ∨ A = C ∨ B = C) :=
by
  -- Definitions
  assume h
  cases h with A h
  cases h with B h
  cases h with C h
  cases h with h1 h2
  cases h2 with h2a h2b
  existsi A
  existsi B
  existsi C
  use [rfl, h1, h2a, h2b]
  sorry -- proof needed

end isosceles_triangle_condition_D_l318_318432


namespace test_completion_ways_l318_318116

theorem test_completion_ways :
  ∀ (n_questions : ℕ) 
    (n_choices : ℕ) 
    (n_single_answer : ℕ) 
    (n_multi_select : ℕ)
    (n_correct_multi_select : ℕ),
  n_questions = 10 →
  n_choices = 8 →
  n_single_answer = 6 →
  n_multi_select = 4 →
  n_correct_multi_select = 2 →
  number_of_ways_to_complete_test_unanswered n_questions n_single_answer n_multi_select = 1 :=
by
  intro n_questions n_choices n_single_answer n_multi_select n_correct_multi_select
  intro h1 h2 h3 h4 h5
  sorry

-- Definition of number_of_ways_to_complete_test_unanswered
-- This should be mathematically defined given the constraints, but is placed here as a placeholder
-- for completeness.
def number_of_ways_to_complete_test_unanswered 
  (n_questions n_single_answer n_multi_select : ℕ) : ℕ := 1

end test_completion_ways_l318_318116


namespace triangle_inequality_l318_318568

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
by
  sorry

end triangle_inequality_l318_318568


namespace hillary_total_profit_l318_318491

def base_price := 12
def cost_day1 := 4
def cost_day2 := cost_day1 + 1
def cost_day3 := cost_day1 - 2
def discount_day3 := 2
def sales_tax := 0.1
def extra_money_day1 := 7

def revenue (crafts_sold : ℕ) (price : ℕ) : ℕ := crafts_sold * price

def profit_day1 := revenue 3 base_price + extra_money_day1 - (3 * cost_day1)
def taxed_profit_day1 := (1 - sales_tax) * profit_day1

def profit_day2 := revenue 4 base_price - (4 * cost_day2)
def taxed_profit_day2 := (1 - sales_tax) * profit_day2

def profit_day3 := revenue 5 (base_price - discount_day3) - (5 * cost_day3)
def taxed_profit_day3 := (1 - sales_tax) * profit_day3

def total_profit := taxed_profit_day1 + taxed_profit_day2 + taxed_profit_day3

theorem hillary_total_profit : total_profit = 89.10 := 
sorry

end hillary_total_profit_l318_318491


namespace michael_total_earnings_l318_318365

-- Define the cost of large paintings and small paintings
def large_painting_cost : ℕ := 100
def small_painting_cost : ℕ := 80

-- Define the number of large and small paintings sold
def large_paintings_sold : ℕ := 5
def small_paintings_sold : ℕ := 8

-- Calculate Michael's total earnings
def total_earnings : ℕ := (large_painting_cost * large_paintings_sold) + (small_painting_cost * small_paintings_sold)

-- Prove: Michael's total earnings are 1140 dollars
theorem michael_total_earnings : total_earnings = 1140 := by
  sorry

end michael_total_earnings_l318_318365


namespace prime_number_conditions_l318_318430

theorem prime_number_conditions :
  ∃ p n : ℕ, Prime p ∧ p = n^2 + 9 ∧ p = (n+1)^2 - 8 :=
by
  sorry

end prime_number_conditions_l318_318430


namespace elroy_more_miles_l318_318184

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l318_318184


namespace cos_squared_difference_l318_318809

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318809


namespace circle_area_increase_l318_318444

theorem circle_area_increase (r : ℝ) : 
  let r' := 1.01 * r,
      A := π * r^2,
      A' := π * r'^2
  in (A' - A) / A * 100 = 2.01 := 
by
  sorry

end circle_area_increase_l318_318444


namespace ellipse_equation_k_k_prime_constant_l318_318570

-- Define the conditions for the ellipse and its eccentricity
def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (sqrt (a^2 - b^2) / a = sqrt 2 / 2)

-- Define the condition for the circle tangent to the line
def circle_tangent_line (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (b = 6 / sqrt (a^2 + 1))

-- Proposition 1: Equation of the ellipse C
theorem ellipse_equation : ∃ (a b : ℝ), (a > b ∧ b > 0) ∧
  ellipse_eccentricity a b ∧
  circle_tangent_line a b ∧
  (a^2 = 8 ∧ b^2 = 4) :=
sorry

-- Define the condition relating the areas of triangles
def areas_relation (A B P : ℝ × ℝ) (S1 S2 : ℝ) : Prop :=
  |A - P| * S2 = |B - P| * S1

-- Proposition 2: Determine k' is a constant
theorem k_k_prime_constant (k k' : ℝ) (A B P : ℝ × ℝ) (S1 S2 : ℝ) :
  k ≥ 0 ∧
  areas_relation A B P S1 S2 →
  (k * k' = 1 / 2) :=
sorry

end ellipse_equation_k_k_prime_constant_l318_318570


namespace clothes_merchant_profit_loss_l318_318458

theorem clothes_merchant_profit_loss :
  ∃ (x y : ℝ), 
  let sale_price := 168 in
  let profit_factor := 1.2 in
  let loss_factor := 0.8 in
  x * profit_factor = sale_price ∧ y * loss_factor = sale_price ∧ (sale_price - x + y - sale_price) = 14 :=
begin
  use 140,
  use 210,
  dsimp [profit_factor, loss_factor, sale_price],
  split,
  { linarith },
  { split; linarith }
end

end clothes_merchant_profit_loss_l318_318458


namespace rectangle_laser_distance_l318_318335

noncomputable def distance_traveled (EF FG FP: ℝ) : ℝ :=
  let EP := Real.sqrt (EF^2 + FP^2)
  let E_to_P_H := EP + EF
  let P_H_to_E := 2 * E_to_P_H
  (2 * E_to_P_H) + (4 * P_H_to_E)

theorem rectangle_laser_distance:
  ∀ (EF FG FP: ℝ),
    EF = 3 →
    FG = 40 →
    FP = 4 →
    distance_traveled EF FG FP = 80 :=
by
  intros EF FG FP hEF hFG hFP
  unfold distance_traveled
  rw [hEF, hFG, hFP]
  sorry

end rectangle_laser_distance_l318_318335


namespace lily_typing_speed_l318_318684

-- Define the conditions
def wordsTyped : ℕ := 255
def totalMinutes : ℕ := 19
def breakTime : ℕ := 2
def typingInterval : ℕ := 10
def effectiveMinutes : ℕ := totalMinutes - breakTime

-- Define the number of words typed in effective minutes
def wordsPerMinute (words : ℕ) (minutes : ℕ) : ℕ := words / minutes

-- Statement to be proven
theorem lily_typing_speed : wordsPerMinute wordsTyped effectiveMinutes = 15 :=
by
  -- proof goes here
  sorry

end lily_typing_speed_l318_318684


namespace hyperbola_eccentricity_l318_318030

theorem hyperbola_eccentricity 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = (√3 / 2) * a)
  (hc : c = (√7 / 2) * a) :
  ∃ e : ℝ, e = c / a ∧ e = √7 / 2 :=
by
  sorry

end hyperbola_eccentricity_l318_318030


namespace sphere_surface_area_l318_318223

theorem sphere_surface_area (a b c : ℝ)
  (h1 : a * b * c = Real.sqrt 6)
  (h2 : a * b = Real.sqrt 2)
  (h3 : b * c = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (a^2 + b^2 + c^2) / 2) ^ 2 = 6 * Real.pi :=
sorry

end sphere_surface_area_l318_318223


namespace base_angle_of_isosceles_triangle_l318_318722

-- Definition of an isosceles triangle with a vertex angle of 50 degrees
def is_isosceles_triangle (triangle : Type) :=
  ∃ A B C : triangle, ∃ vertex_angle base_angle : ℝ,
  vertex_angle = 50 ∧
  base_angle = (180 - vertex_angle) / 2 ∧
  ∀ x y z : triangle, x + y + z = 180 ∧
  (x = B ∧ y = C ∧ z = A ∧ B = C)

-- Proof statement as Lean theorem
theorem base_angle_of_isosceles_triangle {triangle : Type} :
  is_isosceles_triangle triangle → 
  (∃ (A B C : triangle) (vertex_angle base_angle : ℝ), 
    vertex_angle = 50 ∧
    base_angle = (180 - vertex_angle) / 2 ∧
    base_angle = 65) :=
by
  sorry

end base_angle_of_isosceles_triangle_l318_318722


namespace item_prices_correct_l318_318464

def final_selling_price (cost_price : ℝ) (profit_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let profit := cost_price * (profit_percent / 100)
  let pre_tax_price := cost_price + profit
  let sales_tax := pre_tax_price * (tax_percent / 100)
  pre_tax_price + sales_tax

theorem item_prices_correct (cost_price_A cost_price_B cost_price_C : ℝ)
  (profit_percent_A profit_percent_B profit_percent_C : ℝ)
  (tax_percent : ℝ) :
  cost_price_A = 650 ∧ profit_percent_A = 10 ∧
  cost_price_B = 1200 ∧ profit_percent_B = 15 ∧
  cost_price_C = 800 ∧ profit_percent_C = 20 ∧
  tax_percent = 5 →
  final_selling_price cost_price_A profit_percent_A tax_percent = 750.75 ∧
  final_selling_price cost_price_B profit_percent_B tax_percent = 1449 ∧
  final_selling_price cost_price_C profit_percent_C tax_percent = 1008 :=
by
  intros
  sorry

end item_prices_correct_l318_318464


namespace find_theta_l318_318244

theorem find_theta 
  (θ : ℝ) 
  (z : ℂ := complex.of_real_real (cos θ) + complex.i * (sin θ)) 
  (ω : ℂ := (1 - (complex.conj z)^4) / (1 + z^4)) 
  (h₀ : 0 < θ ∧ θ < π) 
  (h₁ : complex.abs ω = (real.sqrt 3) / 3) 
  (h₂ : complex.arg ω < π / 2) : 
  θ = π / 12 ∨ θ = 7 * π / 12 := 
sorry

end find_theta_l318_318244


namespace line_pq_through_fixed_point_l318_318574

noncomputable def fixed_point (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : a + b ≠ 0) : ℝ × ℝ :=
(-d * (a - b) / (a + b), c / (a + b))

theorem line_pq_through_fixed_point (a b c d l m : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : a + b ≠ 0)
  (h_pts : a * l + b * m + c = 0) :
    ∃ A : ℝ × ℝ, A = fixed_point a b c d h₁ h₂ h₃ h₄ h₅ ∧ line_through (bounded_by_points (-d, l) (d, m)) A :=
sorry

end line_pq_through_fixed_point_l318_318574


namespace option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l318_318786

variable (x y: ℝ)

theorem option_A_is_incorrect : 5 - 3 * (x + 1) ≠ 5 - 3 * x - 1 := 
by sorry

theorem option_B_is_incorrect : 2 - 4 * (x + 1/4) ≠ 2 - 4 * x + 1 := 
by sorry

theorem option_C_is_correct : 2 - 4 * (1/4 * x + 1) = 2 - x - 4 := 
by sorry

theorem option_D_is_incorrect : 2 * (x - 2) - 3 * (y - 1) ≠ 2 * x - 4 - 3 * y - 3 := 
by sorry

end option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l318_318786


namespace cos_squared_difference_l318_318831

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318831


namespace optionB_is_a9_l318_318146

-- Definitions of the expressions
def optionA (a : ℤ) : ℤ := a^3 + a^6
def optionB (a : ℤ) : ℤ := a^3 * a^6
def optionC (a : ℤ) : ℤ := a^10 - a
def optionD (a α : ℤ) : ℤ := α^12 / a^2

-- Theorem stating which option equals a^9
theorem optionB_is_a9 (a α : ℤ) : optionA a ≠ a^9 ∧ optionB a = a^9 ∧ optionC a ≠ a^9 ∧ optionD a α ≠ a^9 :=
by
  sorry

end optionB_is_a9_l318_318146


namespace angle_between_vectors_is_60_degrees_l318_318266

noncomputable def vector_a : ℝ × ℝ × ℝ := (0, 1, 1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 0, 1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

theorem angle_between_vectors_is_60_degrees :
  cos_angle vector_a vector_b = 1 / 2 :=
by
  sorry

end angle_between_vectors_is_60_degrees_l318_318266


namespace find_original_price_l318_318741

-- Definitions based on Conditions
def original_price (P : ℝ) : Prop :=
  let increased_price := 1.25 * P
  let final_price := increased_price * 0.75
  final_price = 187.5

theorem find_original_price (P : ℝ) (h : original_price P) : P = 200 :=
  by sorry

end find_original_price_l318_318741


namespace geom_seq_sum_ratio_l318_318342

theorem geom_seq_sum_ratio
  (a1 q : ℝ)
  (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h : 27 * a1 * q + a1 * q^4 = 0) :
  S 4 / S 2 = 10 := 
by
  sorry

end geom_seq_sum_ratio_l318_318342


namespace cos_difference_identity_l318_318936

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318936


namespace find_a_l318_318293

theorem find_a (k : ℕ) (h1 : 18^k ∣ 624938) (h2 : k = 1) : ∃ a : ℕ, a^k - k^a = 1 ∧ a = 2 :=
by
  use 2
  split
  sorry
  rfl

end find_a_l318_318293


namespace integer_ratio_value_l318_318352

theorem integer_ratio_value {x y : ℝ} (h1 : 3 < (x^2 - y^2) / (x^2 + y^2)) (h2 : (x^2 - y^2) / (x^2 + y^2) < 4) (h3 : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = 2 :=
by
  sorry

end integer_ratio_value_l318_318352


namespace trigonometric_identity_l318_318853

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318853


namespace smallest_sphere_radius_l318_318068

theorem smallest_sphere_radius 
(A B C D : ℝ×ℝ×ℝ)
(hA : A = (1, 1, 1))
(hB : B = (-1, -1, 1))
(hC : C = (-1, 1, -1))
(hD : D = (1, -1, -1))
(hAB_dist : dist A B = 2 * real.sqrt 2)
(hAC_dist : dist A C = 2 * real.sqrt 2)
(hAD_dist : dist A D = 2 * real.sqrt 2)
(hBC_dist : dist B C = 2 * real.sqrt 2)
(hBD_dist : dist B D = 2 * real.sqrt 2)
(hCD_dist : dist C D = 2 * real.sqrt 2) :
    ∃ R : ℝ, R = real.sqrt 3 + 1 :=
by
  sorry

end smallest_sphere_radius_l318_318068


namespace complex_division_l318_318727

-- Defining the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Defining the complex numbers involved
def z1 : ℂ := 1 + 3 * i
def z2 : ℂ := 1 - i

-- Proving the desired equality
theorem complex_division : z1 / z2 = -1 + 2 * i := by
  sorry

end complex_division_l318_318727


namespace cos_diff_square_identity_l318_318818

theorem cos_diff_square_identity :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_diff_square_identity_l318_318818


namespace find_ab_sum_l318_318627

theorem find_ab_sum
  (a b : ℝ)
  (h₁ : a^3 - 3 * a^2 + 5 * a - 1 = 0)
  (h₂ : b^3 - 3 * b^2 + 5 * b - 5 = 0) :
  a + b = 2 := by
  sorry

end find_ab_sum_l318_318627


namespace cos_squared_difference_l318_318924

theorem cos_squared_difference :
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by
  sorry

end cos_squared_difference_l318_318924


namespace nathaniel_best_friends_l318_318369

theorem nathaniel_best_friends :
  ∀ (nathaniel_tickets total_tickets tickets_left tickets_per_friend : ℕ),
  nathaniel_tickets = 11 →
  tickets_left = 3 →
  tickets_per_friend = 2 →
  total_tickets = nathaniel_tickets - tickets_left →
  total_tickets / tickets_per_friend = 4 :=
by
  intros nathaniel_tickets total_tickets tickets_left tickets_per_friend htickets hleft hper heq
  rw [htickets, hleft, hper] at heq
  rw [htickets, hleft, hper]
  sorry

end nathaniel_best_friends_l318_318369


namespace maximal_subset_with_property_A_l318_318133

-- Define property A for a subset S ⊆ {0, 1, 2, ..., 99}
def has_property_A (S : Finset ℕ) : Prop := 
  ∀ a b c : ℕ, (a * 10 + b ∈ S) → (b * 10 + c ∈ S) → False

-- Define the set of integers {0, 1, 2, ..., 99}
def numbers_set := Finset.range 100

-- The main statement to be proven
theorem maximal_subset_with_property_A :
  ∃ S : Finset ℕ, S ⊆ numbers_set ∧ has_property_A S ∧ S.card = 25 := 
sorry

end maximal_subset_with_property_A_l318_318133


namespace sheets_per_pack_l318_318007

theorem sheets_per_pack (p d t : Nat) (total_sheets : Nat) (sheets_per_pack : Nat) 
  (h1 : p = 2) (h2 : d = 80) (h3 : t = 6) 
  (h4 : total_sheets = d * t)
  (h5 : sheets_per_pack = total_sheets / p) : sheets_per_pack = 240 := 
  by 
    sorry

end sheets_per_pack_l318_318007


namespace sum_of_perimeters_l318_318148

noncomputable def triangle_series_perimeters_sum (T1_side : ℕ) : ℕ :=
  let P1 := 3 * T1_side in
  let r := (1 : ℚ) / 2 in
  let a := (P1 : ℚ) in
  a / (1 - r)

theorem sum_of_perimeters (hT1_side : 80) : triangle_series_perimeters_sum hT1_side = 480 := 
by 
  -- mathematical proof will be provided here
  sorry

end sum_of_perimeters_l318_318148


namespace infection_average_l318_318119

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l318_318119


namespace solve_for_x_l318_318290

theorem solve_for_x (x : ℕ) (h1 : x > 0) (h2 : x % 6 = 0) (h3 : x^2 > 144) (h4 : x < 30) : x = 18 ∨ x = 24 :=
by
  sorry

end solve_for_x_l318_318290


namespace line_circle_chord_length_condition_l318_318731

theorem line_circle_chord_length_condition (k : ℝ) (M N : ℝ × ℝ) :
  let line_eq := ∀ x : ℝ, y = k * x + 3
      circle_eq := ∀ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4
      chord_length := ∀ (M N : ℝ × ℝ), (x M - x N)^2 + (y M - y N)^2 ≥ 12
  line_eq ∧ circle_eq ∧ chord_length → (-3/4 : ℝ) ≤ k ∧ k ≤ 0 :=
  by
  -- omitted proof
  sorry

end line_circle_chord_length_condition_l318_318731


namespace inter_A_B_l318_318264

variable (x : ℝ)

def A : Set ℝ := {x | abs (x - 1) < 1}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

theorem inter_A_B : A x ∩ B x = {x | x ∈ Ioo 1 2} :=
sorry

end inter_A_B_l318_318264


namespace prime_iff_exists_x_l318_318351

-- Define the problem conditions and the theorem to be proved
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_iff_exists_x (n : ℕ) (p : ℕ → Prop) :
  prime n ↔ ∃ (x : ℤ) (Hx : x ∈ zpowers_of_two (n - 1)), 
    x^(n - 1) ≡ 1 [MOD n] ∧ (∀ (i : ℕ), p i → (n - 1) ∣ i → x^((n - 1) / i) ≢ 1 [MOD n]) := by
  sorry

end prime_iff_exists_x_l318_318351


namespace find_n_pow_m_l318_318575

theorem find_n_pow_m (n m : ℤ) (h1 : |n| = 1) (h2 : n ≠ 1) (h3 : m = 2017) : n^m = -1 := 
sorry

end find_n_pow_m_l318_318575


namespace John_spending_l318_318441

theorem John_spending
  (X : ℝ)
  (h1 : (1/2) * X + (1/3) * X + (1/10) * X + 10 = X) :
  X = 150 :=
by
  sorry

end John_spending_l318_318441


namespace num_students_B_l318_318309

-- Define the given conditions
variables (x : ℕ) -- The number of students who get a B

noncomputable def number_of_A := 2 * x
noncomputable def number_of_C := (12 / 10 : ℤ) * x -- Using (12 / 10) to approximate 1.2 in integers

-- Given total number of students is 42 for integer result
def total_students := 42

-- Lean statement to show number of students getting B is 10
theorem num_students_B : 4.2 * (x : ℝ) = 42 → x = 10 :=
by
  sorry

end num_students_B_l318_318309


namespace points_on_fourth_board_l318_318447

-- Definition of the points scored on each dartboard
def points_board_1 : ℕ := 30
def points_board_2 : ℕ := 38
def points_board_3 : ℕ := 41

-- Statement to prove that points on the fourth board are 34
theorem points_on_fourth_board : (points_board_1 + points_board_2) / 2 = 34 :=
by
  -- Given points on first and second boards
  have h1 : points_board_1 + points_board_2 = 68 := by rfl
  sorry

end points_on_fourth_board_l318_318447


namespace nathaniel_best_friends_l318_318370

theorem nathaniel_best_friends :
  ∀ (nathaniel_tickets total_tickets tickets_left tickets_per_friend : ℕ),
  nathaniel_tickets = 11 →
  tickets_left = 3 →
  tickets_per_friend = 2 →
  total_tickets = nathaniel_tickets - tickets_left →
  total_tickets / tickets_per_friend = 4 :=
by
  intros nathaniel_tickets total_tickets tickets_left tickets_per_friend htickets hleft hper heq
  rw [htickets, hleft, hper] at heq
  rw [htickets, hleft, hper]
  sorry

end nathaniel_best_friends_l318_318370


namespace cost_per_square_foot_other_modules_l318_318172

-- Define given conditions
def kitchen_square_feet : ℕ := 400
def kitchen_cost : ℕ := 20000
def bathroom_square_feet : ℕ := 150
def bathroom_cost : ℕ := 12000
def total_home_square_feet : ℕ := 2000
def total_home_cost : ℕ := 174000

-- Calculate derived quantities
def num_bathrooms : ℕ := 2
def total_kitchen_and_bath_sq_ft : ℕ := kitchen_square_feet + num_bathrooms * bathroom_square_feet := 700
def total_kitchen_and_bath_cost : ℕ := kitchen_cost + num_bathrooms * bathroom_cost := 44000
def other_modules_square_feet : ℕ := total_home_square_feet - total_kitchen_and_bath_sq_ft := 1300
def other_modules_cost : ℕ := total_home_cost - total_kitchen_and_bath_cost := 130000

-- The cost per square foot for other modules
theorem cost_per_square_foot_other_modules :
  (other_modules_cost / other_modules_square_feet) = 100 :=
by
  -- Proof steps would be here, but we use 'sorry' as this is omitted
  sorry

end cost_per_square_foot_other_modules_l318_318172


namespace container_max_volume_l318_318476

noncomputable def volume (x : ℝ) : ℝ :=
  x * (x + 0.5) * (3.2 - 2 * x)

theorem container_max_volume (h : ¬(0 : ℝ) < x < 1.6) :
  let height := 3.2 - 2 * 1 in
  let max_volume := volume 1 in
  height = 1.2 ∧ max_volume = 1.8 := by
  sorry

end container_max_volume_l318_318476


namespace n_value_l318_318606

variable (e1 e2 : Type) [AddCommGroup e1] [Module ℝ e1]
variable (a b : e1)
variable (n : ℝ)
variable (scalar_mul : ℝ → e1 → e1)

def vector_a : e1 := scalar_mul 2 e1 - scalar_mul 3 e2
def vector_b : e1 := scalar_mul (1 + n) e1 + scalar_mul n e2

axiom a_parallel_b : ∀ λ : ℝ, vector_a = scalar_mul λ vector_b

theorem n_value (h₁ : vector_a = scalar_mul 2 e1 - scalar_mul 3 e2) 
                (h₂ : vector_b = scalar_mul (1 + n) e1 + scalar_mul n e2) 
                (h₃ : ∃ λ : ℝ, scalar_mul λ vector_a = vector_b) : 
                n = -3 / 5 :=
by
  sorry

end n_value_l318_318606


namespace base8_base6_positive_integer_l318_318783

theorem base8_base6_positive_integer (C D N : ℕ)
  (base8: N = 8 * C + D)
  (base6: N = 6 * D + C)
  (valid_C_base8: C < 8)
  (valid_D_base6: D < 6)
  (valid_C_D: 7 * C = 5 * D)
: N = 43 := by
  sorry

end base8_base6_positive_integer_l318_318783


namespace libby_igloo_bricks_l318_318683

theorem libby_igloo_bricks :
  let row1 := 14
  let row2 := row1 + 2
  let row3 := row2 + 2
  let row4 := row3 + 2
  let row5 := row4 + 2
  let row6 := row5 + 2
  let row7 := row6 - 3
  let row8 := row7 - 3
  let row9 := row8 - 3
  let row10 := row9 - 3
  row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9 + row10 = 170 :=
by
  let row1 := 14
  let row2 := row1 + 2
  let row3 := row2 + 2
  let row4 := row3 + 2
  let row5 := row4 + 2
  let row6 := row5 + 2
  let row7 := row6 - 3
  let row8 := row7 - 3
  let row9 := row8 - 3
  let row10 := row9 - 3
  have h1 : row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9 + row10 = 14 + 16 + 18 + 20 + 22 + 24 + 21 + 18 + 15 + 12,
    sorry
  have h2 : 14 + 16 + 18 + 20 + 22 + 24 + 21 + 18 + 15 + 12 = 170,
    sorry
  exact eq.trans h1 h2

end libby_igloo_bricks_l318_318683


namespace cos_of_angle_through_point_l318_318752

theorem cos_of_angle_through_point :
  let α : ℝ 
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2 in
  (∃ (x y : ℝ), x = 2 * sin_30 ∧ y = -2 * cos_30 ∧ (x^2 + y^2 = 4)) →
  cos α = 1 / 2 :=
by
  sorry

end cos_of_angle_through_point_l318_318752


namespace ratio_Jane_to_John_l318_318336

-- Define the conditions as given in the problem.
variable (J N : ℕ) -- total products inspected by John and Jane
variable (rJ rN rT : ℚ) -- rejection rates for John, Jane, and total

-- Setting up the provided conditions
axiom h1 : rJ = 0.005 -- John rejected 0.5% of the products he inspected
axiom h2 : rN = 0.007 -- Jane rejected 0.7% of the products she inspected
axiom h3 : rT = 0.0075 -- 0.75% of the total products were rejected

-- Prove the ratio of products inspected by Jane to products inspected by John is 5
theorem ratio_Jane_to_John : (rJ * J + rN * N) = rT * (J + N) → N = 5 * J :=
by 
  sorry

end ratio_Jane_to_John_l318_318336


namespace tangent_line_at_slope_neg1_correct_l318_318197

-- Definitions based on the conditions
def curve (x : ℝ) : ℝ := Real.log x - x ^ 2
def derivative_curve (x : ℝ) : ℝ := (1 / x) - 2 * x

-- The proof goal
theorem tangent_line_at_slope_neg1_correct :
  ∃ (x₀ y₀ : ℝ), derivative_curve x₀ = -1 ∧ x₀ > 0 ∧ y₀ = curve x₀ ∧ (∀ x y : ℝ, y = curve x → (y + 1 = -(x - 1)) ∨ y = -x) :=
by
  sorry


end tangent_line_at_slope_neg1_correct_l318_318197


namespace problem_statement_l318_318349

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem problem_statement
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 3)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  -1 < a ∧ a < 2 :=
sorry

end problem_statement_l318_318349


namespace union_sets_l318_318604

variable {α : Type*} [LinearOrder α]

def setP : Set α := { x : α | -1 < x ∧ x < 1 }
def setQ : Set α := { x : α | 0 < x ∧ x < 2 }

theorem union_sets : setP ∪ setQ = { x : α | -1 < x ∧ x < 2 } :=
by
  sorry

end union_sets_l318_318604


namespace largest_square_in_square_with_triangles_l318_318327

noncomputable def largest_inscribed_square_side_length : ℝ :=
  (15 - 5 * real.sqrt 3) / 3

theorem largest_square_in_square_with_triangles
  (s : ℝ)
  (side_length_of_squares : s = 15)
  (side_length_of_triangles : ∀ t : ℝ, t = let x := (15 * real.sqrt 2) / (2 * real.sqrt 3) in 2 * x)
  (largest_square : ∀ l : ℝ, l = (15 - 5 * real.sqrt 3) / 3) :
  largest_inscribed_square_side_length = (15 - 5 * real.sqrt 3) / 3 := 
by
  sorry

end largest_square_in_square_with_triangles_l318_318327


namespace simplify_expression_l318_318003

theorem simplify_expression (n : ℕ) (hn : 0 < n) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4) - 6) = 80 / 81 :=
by
  sorry

end simplify_expression_l318_318003


namespace cos_square_difference_l318_318878

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318878


namespace angle_between_a_and_a_plus_2b_l318_318582

variables (a b : ℝ^3)

def angle_between_vectors (u v : ℝ^3) : ℝ :=
  real.arccos ((u • v) / (∥u∥ * ∥v∥))

theorem angle_between_a_and_a_plus_2b 
  (h1 : angle_between_vectors a b = real.pi / 3)
  (h2 : ∥a∥ = 2)
  (h3 : ∥b∥ = 1) :
  angle_between_vectors a (a + 2 • b) = real.pi / 6 :=
sorry

end angle_between_a_and_a_plus_2b_l318_318582


namespace sum_of_transformed_set_l318_318567

theorem sum_of_transformed_set (n : ℕ) (s : ℕ) (x : Fin n → ℕ) (h_sum : ∑ i, x i = s) :
  ∑ i, (3 * (x i + 15) + 10) = 3 * s + 55 * n := by
sorry

end sum_of_transformed_set_l318_318567


namespace cos_difference_identity_l318_318934

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318934


namespace direction_vector_correct_b_l318_318400

-- Define the points
def point1 : Prod ℝ ℝ := (-3, 0)
def point2 : Prod ℝ ℝ := (0, 3)

-- Define the direction vector from point1 to point2
def direction_vector : Prod ℝ ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Prove that the direction vector is of the form (3, b) where b = 3
theorem direction_vector_correct_b : ∃ b : ℝ, direction_vector = (3, b) ∧ b = 3 := by
  use 3
  simp [direction_vector, point1, point2]
  sorry

end direction_vector_correct_b_l318_318400


namespace triangle_ABC_properties_l318_318252

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem triangle_ABC_properties
  (xA xB xC : ℝ)
  (h_seq : xA < xB ∧ xB < xC ∧ 2 * xB = xA + xC)
  : (f xB + (f xA + f xC) / 2 > f ((xA + xC) / 2)) ∧ (f xA ≠ f xB ∧ f xB ≠ f xC) := 
sorry

end triangle_ABC_properties_l318_318252


namespace num_incorrect_sequences_hello_l318_318739

theorem num_incorrect_sequences_hello :
  let total_permutations := Nat.factorial 5 / Nat.factorial 2 in
  total_permutations - 1 = 119 :=
by
  sorry

end num_incorrect_sequences_hello_l318_318739


namespace proof_problem_l318_318893

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ), θ = π / 12 ∧ 
              cos θ ^ 2 - cos (5 * θ) ^ 2 = (sqrt 3) / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l318_318893


namespace max_length_BD_min_length_BD_l318_318770

variables {a b : ℝ}
variables {A B C D : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [normed_group A] [normed_group B] [normed_group C] [normed_group D]
variables [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ D]

def is_equilateral_triangle (A C D : Type) [metric_space A] [metric_space C] [metric_space D] :=
  dist A C = dist C D ∧ dist C D = dist D A

def length (x y : Type) [metric_space x] [metric_space y] [normed_group x] [normed_group y] [normed_space ℝ x] [normed_space ℝ y] := dist x y

theorem max_length_BD (a b : ℝ) (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  [normed_group A] [normed_group B] [normed_group C] [normed_group D] [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ D]
  (h1 : length A B = a) (h2 : length B C = b) (h3 : is_equilateral_triangle A C D) (angle_ABC : ℝ) 
  (h_angle_ABC : angle_ABC = 120) : length B D = a + b := 
sorry

theorem min_length_BD (a b : ℝ) (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  [normed_group A] [normed_group B] [normed_group C] [normed_group D] [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ D]
  (h1 : length A B = a) (h2 : length B C = b) (h3 : is_equilateral_triangle A C D) (angle_ABC : ℝ) 
  (h_angle_ABC : angle_ABC = 0) : length B D = sqrt (a ^ 2 + b ^ 2 - a * b) := 
sorry

end max_length_BD_min_length_BD_l318_318770


namespace cosine_difference_identity_l318_318846

theorem cosine_difference_identity :
  (cos (π / 12)) ^ 2 - (cos (5 * π / 12)) ^ 2 = (√3 / 2) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end cosine_difference_identity_l318_318846


namespace cos_squared_difference_l318_318888

theorem cos_squared_difference:
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt 3 / 2 := sorry

end cos_squared_difference_l318_318888


namespace point_P_in_fourth_quadrant_l318_318721

-- Define the point P with given coordinates
def P : ℝ × ℝ := (8, -3)

-- Define a function that determines the quadrant based on coordinates
def quadrant (x y : ℝ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On an axis"

-- The theorem we want to prove
theorem point_P_in_fourth_quadrant : quadrant P.1 P.2 = "Fourth quadrant" :=
by
  sorry

end point_P_in_fourth_quadrant_l318_318721


namespace cos_difference_identity_l318_318940

theorem cos_difference_identity : 
  cos (π / 12) ^ 2 - cos (5 * π / 12) ^ 2 = sqrt 3 / 2 := 
sorry

end cos_difference_identity_l318_318940


namespace ratio_of_runs_l318_318440

theorem ratio_of_runs (A B C : ℕ) (h1 : B = C / 5) (h2 : A + B + C = 95) (h3 : C = 75) :
  A / B = 1 / 3 :=
by sorry

end ratio_of_runs_l318_318440


namespace trains_clear_each_other_in_35_seconds_l318_318763

theorem trains_clear_each_other_in_35_seconds 
  (length1 length2 length3 : ℕ) 
  (speed1_kmph speed2_kmph speed3_kmph : ℕ) 
  (h1 : length1 = 350) 
  (h2 : length2 = 450) 
  (h3 : length3 = 250) 
  (h4 : speed1_kmph = 60) 
  (h5 : speed2_kmph = 48) 
  (h6 : speed3_kmph = 36) :
  let speed1 := speed1_kmph * 1000 / 3600,
      speed2 := speed2_kmph * 1000 / 3600,
      speed3 := speed3_kmph * 1000 / 3600,
      relative_speed12 := speed1 + speed2,
      relative_speed13 := speed1 + speed3,
      relative_speed23 := speed2 + speed3,
      max_relative_speed := max (max relative_speed12 relative_speed13) relative_speed23,
      total_length := length1 + length2 + length3,
      time_to_clear := total_length / max_relative_speed
  in time_to_clear = 35 :=
by 
  sorry

end trains_clear_each_other_in_35_seconds_l318_318763


namespace distance_between_cities_l318_318094

theorem distance_between_cities (d : ℝ)
  (meeting_point1 : d - 437 + 437 = d)
  (meeting_point2 : 3 * (d - 437) = 2 * d - 237) :
  d = 1074 :=
by
  sorry

end distance_between_cities_l318_318094


namespace find_plot_width_l318_318453

theorem find_plot_width:
  let length : ℝ := 360
  let area_acres : ℝ := 10
  let square_feet_per_acre : ℝ := 43560
  let area_square_feet := area_acres * square_feet_per_acre
  let width := area_square_feet / length
  area_square_feet = 435600 ∧ length = 360 ∧ square_feet_per_acre = 43560
  → width = 1210 :=
by
  intro h
  sorry

end find_plot_width_l318_318453


namespace larger_cookie_raisins_l318_318396

theorem larger_cookie_raisins : ∃ n r, 5 ≤ n ∧ n ≤ 10 ∧ (n - 1) * r + (r + 1) = 100 ∧ r + 1 = 12 :=
by
  sorry

end larger_cookie_raisins_l318_318396


namespace laundry_lcm_l318_318710

theorem laundry_lcm :
  Nat.lcm (Nat.lcm 6 9) (Nat.lcm 12 15) = 180 :=
by
  sorry

end laundry_lcm_l318_318710


namespace geometric_mean_of_golden_ratio_split_l318_318292

noncomputable def golden_ratio_split {a b : ℝ} (h : b ≠ 0) (h' : a ≠ 0) : Prop :=
  let AC := a - b in
  let hypotenuse := a in
  let leg := b in
  let other_leg := Math.sqrt (a^2 - b^2) in
  AC / b = b / a

theorem geometric_mean_of_golden_ratio_split {a b : ℝ} (h : b ≠ 0) (h' : a ≠ 0) (h_ratio : golden_ratio_split h h') :
  Math.sqrt (a^2 - b^2) = a * b / Math.sqrt (a^2 - b^2) :=
sorry

end geometric_mean_of_golden_ratio_split_l318_318292


namespace last_locker_opened_l318_318132

theorem last_locker_opened (n : ℕ) (h : n = 729) : 
  let lockers := list.range n
      open_third : list ℕ → list ℕ
      | [] => []
      | (l :: ls) => l :: open_third (ls.drop 2)
      closed_locker_indices :=
        lockers.filter (λ i => open_third i ∉ [1, 4, 7, ..., 727])
  in
  closed_locker_indices.nth (closed_locker_indices.length - 1) = some 727 :=
by
  sorry

end last_locker_opened_l318_318132


namespace schedule_count_correct_l318_318986

-- Define the number of periods
def periods : List String := ["Chinese", "Mathematics", "English", "Physical Education"]

-- Define the constraints: PE can't be the first or fourth period
def valid_periods (p : List String) : Prop :=
  p.length = 4 ∧ 
  List.nth p 0 ≠ some "Physical Education" ∧ 
  List.nth p 3 ≠ some "Physical Education"

-- Define the total number of valid schedules
def count_valid_schedules : ℕ :=
  2 * Nat.factorial 3 -- 2 positions for PE times 3! permutations for other subjects

theorem schedule_count_correct : count_valid_schedules = 12 := by
  sorry

end schedule_count_correct_l318_318986


namespace problem_I_problem_II_l318_318214

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem problem_I (x : ℝ) : f(x) < 4 ↔ x ∈ Set.Ioo (-2 : ℝ) 2 :=
by
  sorry

theorem problem_II (a : ℝ) : (∃ x : ℝ, f(x) - abs(a - 1) < 0) ↔ a ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 3 (∞) :=
by
  sorry

end problem_I_problem_II_l318_318214


namespace find_larger_number_l318_318794

-- Definitions based on the conditions
def larger_number (L S : ℕ) : Prop :=
  L - S = 1365 ∧ L = 6 * S + 20

-- The theorem to prove
theorem find_larger_number (L S : ℕ) (h : larger_number L S) : L = 1634 :=
by
  sorry  -- Proof would go here

end find_larger_number_l318_318794


namespace cos_squared_difference_l318_318956

theorem cos_squared_difference :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = (√3 / 2) := 
by
  sorry

end cos_squared_difference_l318_318956


namespace sin_double_alpha_l318_318593

theorem sin_double_alpha :
  ∀ (ω α ϕ : ℝ),
    ω > 0 →
    0 ≤ ϕ ∧ ϕ ≤ π / 2 →
    (∃ δ > 0, ∀ x : ℝ, fmod (x + δ) (2 * π / ω) = x) →
    sin (ω * (π / 6) + ϕ) + 1 = 2 →
    (∃ α, α > π / 6 ∧ α < 2 * π / 3 ∧ sin (α + ϕ) + 1 = 9 / 5) →
    sin (2 * α + 2 * π / 3) = -24 / 25 :=
  sorry

end sin_double_alpha_l318_318593


namespace cyclic_quadrilateral_inequality_l318_318680

theorem cyclic_quadrilateral_inequality (A B C D : Point) (hAneqB : A ≠ B) (hAneqC : A ≠ C) 
(hAneqD : A ≠ D) (hBneqC : B ≠ C) (hBneqD : B ≠ D) (hCneqD : C ≠ D) 
(on_circle : ∀ P : Point, is_on_circle P [A, B, C, D]) 
(longest_side : AB > max BC (max CD DA)) :
  AB + BD > AC + CD :=
sorry

end cyclic_quadrilateral_inequality_l318_318680


namespace sin_product_inequality_l318_318635

-- Definitions related to the geometry problem
variable (A B C : Point)
variable (aA aB aC : ℝ)

-- Acute angled condition
variable (acute_tri : acute_angled_triangle A B C)
variable (altitude_BB' : is_altitude A B)
variable (altitude_CC' : is_altitude A C)
variable (C'B'_intersect : ray_intersects_at C' B' (circumcircle A B C) B'')
variable (alpha_Aangle : angle A B B'' = aA)
variable (alpha_Bangle : angle B C C'' = aB)
variable (alpha_Cangle : angle C A A'' = aC)

-- Main theorem statement
theorem sin_product_inequality : 
  (sin aA) * (sin aB) * (sin aC) ≤ (3 * sqrt 6) / 32 := 
sorry

end sin_product_inequality_l318_318635


namespace parking_lot_cars_l318_318488

-- Define the assumptions
variables (C : ℝ) -- total number of cars
variables (valid_tickets : ℝ) -- number of cars with valid tickets
variables (perm_passes : ℝ) -- number of cars with permanent passes
variables (not_paid : ℝ) -- number of cars that did not pay

-- Assign the values based on conditions
def valid_tickets := 0.75 * C
def perm_passes := (1/5) * valid_tickets
def not_paid := 30

-- Main theorem stating the problem validity
theorem parking_lot_cars : C = 300 :=
by
  have h1 : 0.75 * C + (1/5) * (0.75 * C) + 30 = C,
  { sorry },
  have h2 : 0.75 * C + 0.15 * C + 30 = C,
  { sorry },
  have h3 : 0.90 * C + 30 = C,
  { sorry },
  have h4 : 0.10 * C = 30,
  { sorry },
  have h5 : C = 300,
  { sorry },
  exact h5

end parking_lot_cars_l318_318488


namespace region_ratio_l318_318475

theorem region_ratio (side_length : ℝ) (s r : ℝ) 
  (h1 : side_length = 2)
  (h2 : s = (1 / 2) * (1 : ℝ) * (1 : ℝ))
  (h3 : r = (1 / 2) * (Real.sqrt 2) * (Real.sqrt 2)) :
  r / s = 2 :=
by
  sorry

end region_ratio_l318_318475


namespace complement_intersect_l318_318603

open Set

variable {α : Type*}

-- Define the sets P and Q according to the conditions
def P : Set ℝ := (Iic 0) ∪ (Ioi 3)
def Q : Set ℕ := {0, 1, 2, 3}

-- State the theorem that we want to prove
theorem complement_intersect :
  ((Pᶜ : Set ℝ) ∩ (Q : Set ℝ)) = {1, 2, 3} := by
  sorry

end complement_intersect_l318_318603


namespace percent_between_20000_and_150000_l318_318465

-- Define the percentages for each group of counties
def less_than_20000 := 30
def between_20000_and_150000 := 45
def more_than_150000 := 25

-- State the theorem using the above definitions
theorem percent_between_20000_and_150000 :
  between_20000_and_150000 = 45 :=
sorry -- Proof placeholder

end percent_between_20000_and_150000_l318_318465


namespace average_of_middle_two_numbers_l318_318015

theorem average_of_middle_two_numbers (a b c d : ℕ) 
  (h_dis_diff: a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 20)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_max_diff: ∀ x y w z : ℕ, x ≠ y ∧ y ≠ w ∧ w ≠ z ∧ x < y ∧ y < w ∧ w < z ∧ x + y + w + z = 20 → z - x ≤ d - a):
  (b + c) / 2 = 2.5 :=
by
  sorry

end average_of_middle_two_numbers_l318_318015


namespace ballet_class_members_l318_318977

theorem ballet_class_members (large_groups : ℕ) (members_per_large_group : ℕ) (total_members : ℕ) 
    (h1 : large_groups = 12) (h2 : members_per_large_group = 7) (h3 : total_members = large_groups * members_per_large_group) : 
    total_members = 84 :=
sorry

end ballet_class_members_l318_318977


namespace trigonometric_identity_l318_318855

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318855


namespace domain_f_monotonicity_f_range_f_on_interval_l318_318554

noncomputable def f (x : ℝ) : ℝ := log 4 (4^x - 1)

theorem domain_f : ∀ x : ℝ, f x = f x → x > 0 := sorry

theorem monotonicity_f : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 < f x2 := sorry

theorem range_f_on_interval : ∀ x : ℝ, (1/2) ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ log 4 15 := sorry

end domain_f_monotonicity_f_range_f_on_interval_l318_318554


namespace jill_investment_value_l318_318442

noncomputable def compoundInterest 
  (P : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 := 
by
  sorry

end jill_investment_value_l318_318442


namespace unique_digit_sum_to_2017_l318_318001

noncomputable def digits (n : ℕ) : Set ℕ := 
  ((to_digits 10 n).map (λ d, d.to_nat)).to_set

theorem unique_digit_sum_to_2017 :
  ∃ (a1 a2 a3 a4 a5 : ℕ),
  (a1 + a2 + a3 + a4 + a5 = 2017) ∧
  (digits a1 ∪ digits a2 ∪ digits a3 ∪ digits a4 ∪ digits a5 = digits a1 ∩ digits a2 ∩ digits a3 ∩ digits a4 ∩ digits a5) := 
sorry

end unique_digit_sum_to_2017_l318_318001


namespace cos_square_difference_l318_318869

theorem cos_square_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = √3 / 2 :=
by
  sorry

end cos_square_difference_l318_318869


namespace finite_x_for_perfect_square_l318_318289

theorem finite_x_for_perfect_square (x : ℕ) (y : ℕ) :
  (y = x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1) → 
  ∃ (S : set ℕ), (∀ x ∈ S, (∃ (k : ℕ), y = k^2)) ∧ 
  (∀ x ∉ S, ¬(∃ (k : ℕ), y = k^2)) ∧ 
  (set.finite S) := 
by
  sorry

end finite_x_for_perfect_square_l318_318289


namespace cos_squared_difference_l318_318838

theorem cos_squared_difference :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = sqrt(3) / 2 := by
  sorry

end cos_squared_difference_l318_318838


namespace number_of_roses_l318_318711

def interval : ℝ := 34.1
def road_length : ℝ := 852.5

theorem number_of_roses :
  ∃ (n : ℕ), n = (floor (road_length / interval)) ∧ n = 25 :=
by sorry

end number_of_roses_l318_318711


namespace original_triangle_area_proof_l318_318560

noncomputable def visual_triangle_area : ℝ :=
  (sqrt 3) / 4

noncomputable def original_triangle_area : ℝ :=
  2 * sqrt 2 * visual_triangle_area

theorem original_triangle_area_proof :
  original_triangle_area = sqrt 6 / 2 :=
by
  sorry

end original_triangle_area_proof_l318_318560


namespace li_number_points_on_curve_quadratic_li_number_quadratic_coefficient_ratio_l318_318640

noncomputable theory

-- Definition of Li numbers: Points where the horizontal and vertical coordinates are opposites.
def is_li_number (p : ℝ × ℝ) : Prop := p.2 = -p.1

-- Condition 1: Curve equation y = -16/x
def curve_eq (p : ℝ × ℝ) : Prop := p.2 = -16 / p.1

-- Part 1: Prove the only Li number points on the curve are (4, -4) and (-4, 4)
theorem li_number_points_on_curve :
  ∀ (p : ℝ × ℝ), is_li_number p ∧ curve_eq p ↔ p = (4, -4) ∨ p = (-4, 4) :=
by
  intros p,
  sorry

-- Part 2: Prove c/a = 16 given the specific conditions
theorem quadratic_li_number {a b c : ℝ} (h1 : ∀ (p : ℝ × ℝ), is_li_number p ∧ (p = (-4, 4) ↔ p = curve_eq p ∨ p = (4,)) :=
by
  intros p,
  sorry

theorem quadratic_coefficient_ratio {a b c : ℝ}
  (h1 : ∀ (p : ℝ × ℝ), is_li_number p ∧ (p = (-4, 4) → curve_eq p))
  (h2 : ∃! (p : ℝ × ℝ), is_li_number p ∧ (p.2 = a * p.1 ^ 2 + b * p.1 + c)) :
  c / a = 16 :=
by
  sorry

end li_number_points_on_curve_quadratic_li_number_quadratic_coefficient_ratio_l318_318640


namespace normal_level_short_of_capacity_l318_318498

noncomputable def total_capacity (water_amount : ℕ) (percentage : ℝ) : ℝ :=
  water_amount / percentage

noncomputable def normal_level (water_amount : ℕ) : ℕ :=
  water_amount / 2

theorem normal_level_short_of_capacity (water_amount : ℕ) (percentage : ℝ) (capacity : ℝ) (normal : ℕ) : 
  water_amount = 30 ∧ percentage = 0.75 ∧ capacity = total_capacity water_amount percentage ∧ normal = normal_level water_amount →
  (capacity - ↑normal) = 25 :=
by
  intros h
  sorry

end normal_level_short_of_capacity_l318_318498


namespace invalid_diagonal_sets_l318_318433

/-- A structure representing a box (rectangular prism) with side lengths a, b, and c. --/
structure Box where
  a b c : ℝ

/-- Definition to check if a given set of lengths can be the diagonals of a rectangular prism's faces. --/
def isValidDiagonalSet (d1 d2 d3 : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  d1 = Real.sqrt (a^2 + b^2) ∧ 
  d2 = Real.sqrt (b^2 + c^2) ∧ 
  d3 = Real.sqrt (a^2 + c^2) ∧
  d1^2 + d2^2 ≥ d3^2 ∧
  d2^2 + d3^2 ≥ d1^2 ∧
  d3^2 + d1^2 ≥ d2^2

/-- Theorem stating that {3,4,6} and {4,5,6} cannot be the diagonal lengths of a rectangular prism. --/
theorem invalid_diagonal_sets : 
  ¬ isValidDiagonalSet 3 4 6 ∧ 
  ¬ isValidDiagonalSet 4 5 6 :=
by
  sorry

end invalid_diagonal_sets_l318_318433


namespace first_day_speed_l318_318452

open Real

-- Define conditions
variables (v : ℝ) (t : ℝ)
axiom distance_home_school : 1.5 = v * (t - 7/60)
axiom second_day_condition : 1.5 = 6 * (t - 8/60)

theorem first_day_speed :
  v = 10 :=
by
  -- The proof will be provided here
  sorry

end first_day_speed_l318_318452


namespace triangle_perimeter_l318_318537

def point := (ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def A : point := (2, 3)
def B : point := (2, 9)
def C : point := (6, 6)

noncomputable def perimeter (A B C : point) : ℝ :=
  dist A B + dist B C + dist C A

theorem triangle_perimeter : perimeter A B C = 16 :=
  sorry

end triangle_perimeter_l318_318537


namespace smallest_good_number_is_5_good_number_2005_l318_318358

def is_good_number (n : ℕ) : Prop :=
  ∀ (P : Fin n → ℤ × ℤ),
    (∀ i j, i ≠ j → (rational_dist P i j ↔ ∃ k, k ≠ i ∧ k ≠ j ∧ irrational_dist P i k ∧ irrational_dist P j k)) ∧
    (∀ i j, i ≠ j → (irrational_dist P i j ↔ ∃ k, k ≠ i ∧ k ≠ j ∧ rational_dist P i k ∧ rational_dist P j k))

def rational_dist (P : Fin n → ℤ × ℤ) (i j : Fin n) : Prop :=
  let (xi, yi) := P i in
  let (xj, yj) := P j in
  ∃ (r : ℚ), r^2 = (xi - xj)^2 + (yi - yj)^2

def irrational_dist (P : Fin n → ℤ × ℤ) (i j : Fin n) : Prop :=
  let (xi, yi) := P i in
  let (xj, yj) := P j in
  ∃ (r : ℝ), irrational (r^2) ∧ r^2 = (xi - xj)^2 + (yi - yj)^2

theorem smallest_good_number_is_5 : is_good_number 5 :=
sorry

theorem good_number_2005 : is_good_number 2005 :=
sorry

end smallest_good_number_is_5_good_number_2005_l318_318358


namespace smallest_k_correct_l318_318202

noncomputable def smallest_k (a b t_a t_b : ℝ) [h1 : 0 < a] [h2 : 0 < b] : ℝ :=
  if h : a > 0 ∧ b > 0 then (4 / 3) else 0

theorem smallest_k_correct (a b t_a t_b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hta_lt : t_a < 2 * b * a / (a + b))
  (htb_lt : t_b < 2 * a * (a + b) / (2 * a + b)) :
  ∃ k, k = smallest_k a b t_a t_b ∧ (t_a + t_b) / (a + b) < k :=
begin
  use 4 / 3,
  split,
  { exact if_pos (⟨ha, hb⟩) },
  { sorry }
end

end smallest_k_correct_l318_318202


namespace this_year_winner_time_l318_318753

def length_town_square : ℝ := 3 / 4
def laps : ℝ := 7
def last_year_time : ℝ := 47.25
def time_difference_per_mile : ℝ := 1

def total_distance : ℝ := length_town_square * laps
def last_year_avg_time_per_mile : ℝ := last_year_time / total_distance
def this_year_avg_time_per_mile : ℝ := last_year_avg_time_per_mile - time_difference_per_mile
def this_year_time : ℝ := this_year_avg_time_per_mile * total_distance

theorem this_year_winner_time :
  this_year_time = 42 :=
by
  rw [this_year_time, this_year_avg_time_per_mile, last_year_avg_time_per_mile, total_distance]
  sorry

end this_year_winner_time_l318_318753


namespace football_player_goals_l318_318111

def total_goals_in_5_matches (A : ℝ) (initial_average : ℝ) (goals_in_fifth_match: ℝ) (average_increase: ℝ): Prop :=
  goals_in_fifth_match = 5 ∧ average_increase = 0.2 ∧ (4 * initial_average + goals_in_fifth_match) / 5 = initial_average + average_increase
  → 4 * initial_average + goals_in_fifth_match = 21

theorem football_player_goals (A : ℝ) (initial_average: ℝ) (goals_in_fifth_match: ℝ) (average_increase: ℝ) :
  total_goals_in_5_matches A initial_average goals_in_fifth_match average_increase :=
begin
  sorry
end

end football_player_goals_l318_318111


namespace cos_squared_difference_l318_318905

theorem cos_squared_difference :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = sqrt 3 / 2 :=
sorry

end cos_squared_difference_l318_318905


namespace coprime_nat_three_solutions_l318_318546

theorem coprime_nat_three_solutions {m : ℕ} (x y z : ℤ)
  (hxz : Int.gcd x z = 1) (hyz : Int.gcd y z = 1) (hxy : Int.gcd x y = 1) :
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ (x // y) + (y // z) + (z // x) = m) ↔ m = 3 :=
sorry

end coprime_nat_three_solutions_l318_318546


namespace snow_fall_time_l318_318621

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end snow_fall_time_l318_318621


namespace penAveragePrice_is_20_l318_318972

-- Define the conditions
def totalPens : ℕ := 30
def totalPencils : ℕ := 75
def totalCost : ℝ := 750.00
def pencilAveragePrice : ℝ := 2.00

-- Calculate the total cost of pencils
def pencilTotalCost : ℝ := totalPencils * pencilAveragePrice

-- Calculate the total cost of pens
def penTotalCost : ℝ := totalCost - pencilTotalCost

-- Calculate the average price of a pen
def penAveragePrice : ℝ := penTotalCost / totalPens

-- Theorem to prove the average price of a pen is 20.00
theorem penAveragePrice_is_20 : penAveragePrice = 20.00 :=
by
  sorry

end penAveragePrice_is_20_l318_318972


namespace cos_squared_difference_l318_318805

theorem cos_squared_difference :
  cos(π / 12) ^ 2 - cos(5 * π / 12) ^ 2 = (sqrt 3) / 2 :=
by sorry

end cos_squared_difference_l318_318805


namespace intersection_area_is_zero_l318_318422

open EuclideanGeometry

variable (A B C D : Point)
variable (ABC BAD : Triangle)

def congruent_triangles : Prop :=
  (Triangle.SideLength ABC A B = 12) ∧ (Triangle.SideLength ABC B C = 15) ∧ 
  (Triangle.SideLength ABC C A = 20) ∧ 
  (Triangle.SideLength BAD B A = 12) ∧ (Triangle.SideLength BAD A D = 15) ∧ 
  (Triangle.SideLength BAD D B = 20) ∧ 
  Triangle.Congruent ABC BAD

theorem intersection_area_is_zero (h : congruent_triangles A B C D ABC BAD) :
  Triangle.Area (Triangle.Intersection ABC BAD) = 0 := 
sorry

end intersection_area_is_zero_l318_318422


namespace uniquePlantsTotal_l318_318305

-- Define the number of plants in each bed
def numPlantsInA : ℕ := 600
def numPlantsInB : ℕ := 500
def numPlantsInC : ℕ := 400

-- Define the number of shared plants between beds
def sharedPlantsAB : ℕ := 60
def sharedPlantsAC : ℕ := 120
def sharedPlantsBC : ℕ := 80
def sharedPlantsABC : ℕ := 30

-- Prove that the total number of unique plants in the garden is 1270
theorem uniquePlantsTotal : 
  numPlantsInA + numPlantsInB + numPlantsInC 
  - sharedPlantsAB - sharedPlantsAC - sharedPlantsBC 
  + sharedPlantsABC = 1270 := 
by sorry

end uniquePlantsTotal_l318_318305


namespace new_student_weight_l318_318443

theorem new_student_weight : 
  ∀ (w_new : ℕ), 
    (∀ (sum_weight: ℕ), 80 + sum_weight - w_new = sum_weight - 18) → 
      w_new = 62 := 
by
  intros w_new h
  sorry

end new_student_weight_l318_318443


namespace divides_p_minus_one_l318_318658

theorem divides_p_minus_one {p a b : ℕ} {n : ℕ} 
  (hp : p ≥ 3) 
  (prime_p : Nat.Prime p )
  (gcd_ab : Nat.gcd a b = 1)
  (hdiv : p ∣ (a ^ (2 ^ n) + b ^ (2 ^ n))) : 
  2 ^ (n + 1) ∣ p - 1 := 
sorry

end divides_p_minus_one_l318_318658


namespace quadratic_inequality_solution_set_l318_318046

def quadratic_inequality_solution (a b : ℝ) : set ℝ :=
  {x : ℝ | a * x^2 + b * x - 2 < 0}

theorem quadratic_inequality_solution_set :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 ↔ x ∈ (-∞, -3) ∪ (1, ∞)) →
  quadratic_inequality_solution 2 (-3) = { x | (-(1/2) : ℝ) < x ∧ x < 2 } :=
by sorry

end quadratic_inequality_solution_set_l318_318046


namespace range_of_f_on_interval_l318_318623

def f (x : ℝ) : ℝ := x^2 + 4 * x + 6

theorem range_of_f_on_interval :
  (∀ x, x ∈ set.Ico (-3 : ℝ) 0 → 2 ≤ f x) ∧ (∀ y, (2 ≤ y ∧ y < 6) → ∃ x, x ∈ set.Ico (-3 : ℝ) 0 ∧ f x = y) := 
begin
  sorry
end

end range_of_f_on_interval_l318_318623


namespace nitin_borrowed_amount_l318_318090

theorem nitin_borrowed_amount (P : ℝ) (interest_paid : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) 
  (h_rates1 : rate1 = 0.06) (h_rates2 : rate2 = 0.09) 
  (h_rates3 : rate3 = 0.13) (h_time1 : time1 = 3) 
  (h_time2 : time2 = 5) (h_time3 : time3 = 3)
  (h_interest : interest_paid = 8160) :
  P * (rate1 * time1 + rate2 * time2 + rate3 * time3) = interest_paid → 
  P = 8000 := 
by 
  sorry

end nitin_borrowed_amount_l318_318090


namespace cos_difference_squared_l318_318966

theorem cos_difference_squared :
  cos^2 (π / 12) - cos^2 (5 * π / 12) = (√3) / 2 := 
  sorry

end cos_difference_squared_l318_318966


namespace tan_function_constants_l318_318153

theorem tan_function_constants (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_period : b ≠ 0 ∧ ∃ k : ℤ, b * (3 / 2) = k * π) 
(h_pass : a * Real.tan (b * (π / 4)) = 3) : a * b = 2 * Real.sqrt 3 :=
by 
  sorry

end tan_function_constants_l318_318153


namespace product_of_roots_of_cubic_l318_318164

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end product_of_roots_of_cubic_l318_318164


namespace trigonometric_identity_l318_318864

theorem trigonometric_identity :
  (cos (π / 12))^2 - (cos (5 * π / 12))^2 = (√3 / 2) :=
by
  sorry

end trigonometric_identity_l318_318864


namespace sum_quotient_remainder_17_l318_318073

def number_division (n : ℕ) (q1 r1 : ℕ) (d1 m1 : ℕ) (q2 r2 : ℕ) (d2 m2 : ℕ) : Prop :=
  n = d1 * q1 + r1 ∧ r1 < d1 ∧ n + m1 = d2 * q2 + r2 ∧ r2 < d2 ∧ q2 + r2 = m2

theorem sum_quotient_remainder_17 :
  ∃ n, number_division n 13 1 7 9 12 5 8 17 :=
by
  use 92
  dsimp [number_division]
  norm_num
  sorry

end sum_quotient_remainder_17_l318_318073


namespace mp_squared_l318_318638

/-
 In square ABCD, points K and L lie on line segments AB and DA, respectively, such that AK = AL.
 Points M and N lie on line segments BC and CD, respectively, and points P and Q lie on line segment KL such that MP is perpendicular to KL and NQ is perpendicular to KL.
 Triangle AKL, quadrilateral BMKP, quadrilateral DLNQ, and pentagon MCNQP each has area 2.
 Prove MP^2 = 16 * sqrt(2) - 8
-/

theorem mp_squared (ABCD : Type)
  (A B C D K L M N P Q : ABCD)
  (side : ℝ)
  (AK : ℝ)
  (KL MP NQ : ABCD → ABCD → Prop)
  (AK_eq_AL : AK = AL)
  (MP_perp_KL : MP P K)
  (NQ_perp_KL : NQ Q L)
  (area_AKL : side^2 / 2 = 2)
  (area_BMKP : 2)
  (area_DLNQ : 2)
  (area_MCNQP : 2) :
  MP^2 = 16 * √2 - 8 :=
begin
  sorry
end

end mp_squared_l318_318638
