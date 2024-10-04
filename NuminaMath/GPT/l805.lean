import Math
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Order
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binom
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Angle

namespace count_int_values_cube_bound_l805_805314

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805314


namespace parallism_of_lines_l805_805755

   variables {V : Type*} [inner_product_space ℝ V]
   variables {L : V →ᵃ[ℝ] V} {l : affine_subspace ℝ V} {M N : V}

   -- Conditions: L is not the identity, L maps every point on l to itself, M and N are not on l.
   axiom affine_not_identity (L : V →ᵃ[ℝ] V) : ¬ function.id L
   axiom maps_points_on_l (L : V →ᵃ[ℝ] V) (l : affine_subspace ℝ V) (x : V) (hx : x ∈ l) : L x = x
   axiom point_not_on_l (M N : V) (l : affine_subspace ℝ V) : M ∉ l ∧ N ∉ l

   -- Question: show that lines ML(M) and NL(N) are parallel for points M, N not on the line l
   theorem parallism_of_lines (L : V →ᵃ[ℝ] V) (l : affine_subspace ℝ V) (M N : V)
   (hL : ¬ function.id L)
   (hmap : ∀ x ∈ l, L x = x)
   (hM : M ∉ l) (hN : N ∉ l) :
   let M' := L M
       N' := L N
   in (∃ k : ℝ, ∀ (x y : V), L (x + k • y) = L x + k • L y) → (affine_subspace ℝ M M' ∥ affine_subspace ℝ N N') : 
   begin
     admit,
   end
   
end parallism_of_lines_l805_805755


namespace similar_triangle_DEF_pedal_triangle_area_triangle_DEF_l805_805446

-- Definitions for the given problem
variables {ABC : Type*} [triangle ABC]
variables (AA1 BB1 CC1 : line)
variables (A1 B1 C1 : point)

-- Further conditions specified
variables (circumcircle : circle)
variables (A2 B2 C2 : point)
variables (Simson_line : point → line)
variables (D E F : point)

-- Given points are intersections of altitudes with the circumcircle
axiom second_intersection_A2 : point_on_circumcircle A2 circumcircle
axiom second_intersection_B2 : point_on_circumcircle B2 circumcircle
axiom second_intersection_C2 : point_on_circumcircle C2 circumcircle

-- Definition of the pedal triangle and the Simson lines defining triangle DEF
variables pedal_triangle : triangle A1 B1 C1
variables triangle_DEF : triangle D E F

-- Simson lines for points A2, B2, and C2 forming the triangle DEF
axiom simson_line_A2 : triangle_DEF = Simson_line A2
axiom simson_line_B2 : triangle_DEF = Simson_line B2
axiom simson_line_C2 : triangle_DEF = Simson_line C2

-- Proving the questions
theorem similar_triangle_DEF_pedal_triangle :
  triangle_DEF ∼ pedal_triangle :=
sorry

theorem area_triangle_DEF :
  area triangle_DEF = 4 * area pedal_triangle :=
sorry

end similar_triangle_DEF_pedal_triangle_area_triangle_DEF_l805_805446


namespace Anja_wins_game_l805_805375

theorem Anja_wins_game (a b : ℕ) (ha : a > 0) (hb : b > 0) : ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), ∀ moves : list (ℕ × ℕ), strategy_wins strategy moves :=
begin
  sorry
end

end Anja_wins_game_l805_805375


namespace subway_train_speed_l805_805031

theorem subway_train_speed (s : ℕ) (h1 : 0 ≤ s ∧ s ≤ 7) (h2 : s^2 + 2*s = 63) : s = 7 :=
by
  sorry

end subway_train_speed_l805_805031


namespace total_components_is_900_l805_805038

theorem total_components_is_900 :
  ∃ n : ℕ, 
  (let a := 20
   let b := 300
   let c := 200
   let sample_size := 45
   let sample_C := 10
   in 
   sample_C / sample_size = c / n) → n = 900 :=
by
  sorry

end total_components_is_900_l805_805038


namespace max_kings_is_16_l805_805052

-- Define the Kings and Chessboard
def king : Type :=
{
  move : (ℕ × ℕ) → (ℕ × ℕ) → Prop
}

noncomputable def can_attack (k : king) (p q : ℕ × ℕ) : Prop :=
abs (p.1 - q.1) ≤ 1 ∧ abs (p.2 - q.2) ≤ 1

-- Define the 8x8 chessboard
def chessboard : Type := fin 8 × fin 8

-- Define the condition that no two kings can attack each other
def no_two_kings_attack_each_other (ks : set (fin 8 × fin 8)) : Prop :=
  ∀ k1 k2 ∈ ks, k1 ≠ k2 → ¬ can_attack king k1 k2

-- Define the maximum number of kings
def max_kings_on_chessboard (n : ℕ) : Prop :=
  ∃ (ks : set (fin 8 × fin 8)), no_two_kings_attack_each_other ks ∧ ks.card = n

-- The theorem statement
theorem max_kings_is_16 : max_kings_on_chessboard 16 :=
  sorry

end max_kings_is_16_l805_805052


namespace find_third_vertex_l805_805818

noncomputable section

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 6)
def B : ℝ × ℝ := (-3, 0)

-- Define the area condition
def area_eq_14 (C : ℝ × ℝ) : Prop := 
  (real.abs (1 * (0 - 0) + C.1 * (0 - 6) + (-3) * (6 - 0))) = 28

-- Define the condition that the third vertex is on the positive x-axis
def on_positive_x_axis (C : ℝ × ℝ) : Prop := 
  C.2 = 0 ∧ C.1 > 0

-- Prove that the third vertex satisfying the conditions is (5/3, 0)
theorem find_third_vertex : 
  ∃ C : ℝ × ℝ, area_eq_14 C ∧ on_positive_x_axis C ∧ C = (5 / 3, 0) :=
sorry

end find_third_vertex_l805_805818


namespace at_least_one_not_less_than_two_l805_805324

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (¬ (∀ (x ∈ {b + c / a, a + c / b, a + b / c}), x < 2)) :=
sorry

end at_least_one_not_less_than_two_l805_805324


namespace max_sum_arithmetic_sequence_terms_l805_805776

theorem max_sum_arithmetic_sequence_terms (d : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h0 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : d < 0)
  (h2 : a 1 ^ 2 = a 11 ^ 2) : 
  (n = 5) ∨ (n = 6) :=
sorry

end max_sum_arithmetic_sequence_terms_l805_805776


namespace minimize_tetrahedron_volume_l805_805463

theorem minimize_tetrahedron_volume
  {α β γ a b c : ℝ}
  {x y z : ℝ}
  (h1 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)
  (h3 : (x:ℝ) / a + (y:ℝ) / b + (z:ℝ) / c = 1)
  (h4 : α / a + β / b + γ / c = 1) :
  ∃ (k : ℝ), α / a = β / b ∧ β / b = γ / c ∧ γ / c = k ∧
    k = 1 / 3 ∧
    (a = 3 * α) ∧ (b = 3 * β) ∧ (c = 3 * γ) ∧
    (α * β * γ > 0) ∧
    (1 / 6 * a * b * c =
        (1 / 6 * (3 * α) * (3 * β) * (3 * γ)) ∧
        1 / 6 * α * β * γ = α * β * γ :=
    sorry

end minimize_tetrahedron_volume_l805_805463


namespace first_part_second_part_l805_805706

-- Definitions based on given conditions
def population := {x : ℕ // x < 1000}

def groups := {g : ℕ // g < 10}

noncomputable def sample (x : ℕ) : list ℕ :=
list.map (λ k, (x + 33 * k) % 1000) (list.range 10)

-- Theorem statements
theorem first_part (x : ℕ) (hx : x = 24) :
  sample x = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921] :=
sorry

theorem second_part (ys : list ℕ) (hys : list.any ys (λ y, y % 100 = 87)) :
  ∃ x, (x ∈ {21, 22, 23, 54, 55, 56, 87, 88, 89, 90}) ∧
    ys = sample x :=
sorry

end first_part_second_part_l805_805706


namespace part1_part2_l805_805970

noncomputable def quadratic_eq (k : ℝ) : Polynomial ℝ := polynomial.X^2 - (k + 2) * polynomial.X + 2 * k

-- Problem part 1
theorem part1 (k : ℝ) (h : (1 : ℝ) ∈ (quadratic_eq k).roots) : 
    k = 1 ∧ (roots (quadratic_eq 1)) = {1, 2} := 
  sorry

-- Problem part 2
theorem part2 (k : ℝ) : 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quadratic_eq k).roots = {x₁, x₂} := 
  sorry

end part1_part2_l805_805970


namespace median_first_twelve_integers_l805_805059

theorem median_first_twelve_integers : 
  let lst : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = (6.5 : ℤ) :=
by
  sorry

end median_first_twelve_integers_l805_805059


namespace inequality_conditions_l805_805230

noncomputable def f : ℝ → ℝ := λ x, log x / log 2 + 3^x

variables {a b c d : ℝ}

theorem inequality_conditions 
  (h_arith_seq: ∃ k > 0, b = a + k ∧ c = a + 2 * k)
  (ha: 0 < a) (hb: 0 < b) (hc: 0 < c)
  (h_order: a < b ∧ b < c)
  (h_sign: f(a) * f(b) * f(c) < 0)
  (h_zero: f d = 0) :
  d > b ∧ d < c ∧ d > a :=
sorry

end inequality_conditions_l805_805230


namespace solve_for_x_l805_805327

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 := 
by
  sorry

end solve_for_x_l805_805327


namespace problem_solution_l805_805386

def f(x : ℝ) : ℝ := (2 * x^2 + 3 * x + 4) / (x^2 - 2 * x + 5)
def g(x : ℝ) : ℝ := x^2 - 2

theorem problem_solution : f(g(2)) + g(f(2)) = 14.56 := 
by
  -- Define intermediate steps
  have g_2 : g(2) = 2 := by sorry
  have f_2 : f(2) = 18 / 5 := by sorry
  have g_f_2 : g(f(2)) = 274 / 25 := by sorry
  have f_g_2 : f(g(2)) = 18 / 5 := by sorry
  
  -- Calculate final result
  show f(g(2)) + g(f(2)) = 14.56 from by
    calc
      f(g(2)) + g(f(2)) = 18 / 5 + 274 / 25 : by rewrite [f_g_2, g_f_2]
                   ... = 90 / 25 + 274 / 25 : by rw [div_add_div_same]
                   ... = 364 / 25          : by norm_num
                   ... = 14.56             : by norm_num

end problem_solution_l805_805386


namespace intersection_of_M_and_complement_N_l805_805744

namespace Proof

-- Given definitions
def U := {0, 1, 2, 3, 4, 5}
def M := {0, 3, 5}
def N := {1, 4, 5}
def compl_N := U \ N  -- Complement of N with respect to U

-- Proof statement
theorem intersection_of_M_and_complement_N : M ∩ compl_N = {0, 3} := by
  sorry

end Proof

end intersection_of_M_and_complement_N_l805_805744


namespace find_min_value_l805_805900

theorem find_min_value : 
  ∃ x : ℝ, (∀ y, (y = x + 4 / x) → y ≠ 4) ∧
           (∀ x : ℝ, ∀ y, (y = -x^2 + 2 * x + 3) → y ≠ 4) ∧
           (∀ x : ℝ, (0 < x ∧ x < real.pi) → ∀ y, (y = real.sin x + 4 / real.sin x) → y ≠ 4) ∧
           ∃ x : ℝ, ∀ y, (y = real.exp x + 4 / real.exp x) → y = 4 := 
by 
  sorry

end find_min_value_l805_805900


namespace integer_count_satisfies_inequality_l805_805296

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805296


namespace right_triangle_legs_l805_805881

noncomputable def calculate_legs (m n : ℝ) : (ℝ × ℝ) :=
  (Real.sqrt (m * (m + n)), Real.sqrt (n * (m + n)))

theorem right_triangle_legs {A B C : Type} 
  (m n : ℝ)
  (h_right : ∠ ABC = 90°)
  (h_inscribed : ∃ (circle_center : A), ∀ P ∈ {A, B, C}, dist P circle_center = radius)
  (h_tangent_distances : dist A tangent_line = m ∧ dist B tangent_line = n):
  let (AC, BC) := calculate_legs m n in
  true := by
    sorry

end right_triangle_legs_l805_805881


namespace reciprocal_of_neg2019_l805_805451

theorem reciprocal_of_neg2019 : (1 / -2019) = - (1 / 2019) := 
by
  sorry

end reciprocal_of_neg2019_l805_805451


namespace messages_on_monday_l805_805523

theorem messages_on_monday (M : ℕ) (h0 : 200 + 500 + 1000 = 1700) (h1 : M + 1700 = 2000) : M = 300 :=
by
  -- Maths proof step here
  sorry

end messages_on_monday_l805_805523


namespace rest_area_location_l805_805010

theorem rest_area_location : 
  ∃ (rest_area_milepost : ℕ), 
    let first_exit := 23
    let seventh_exit := 95
    let distance := seventh_exit - first_exit
    let halfway_distance := distance / 2
    rest_area_milepost = first_exit + halfway_distance :=
by
  sorry

end rest_area_location_l805_805010


namespace train_crosses_platform_time_l805_805549

theorem train_crosses_platform_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) (speed_conversion : ℝ) :
  train_length = 160 →
  train_speed_kmph = 72 →
  platform_length = 340.04 →
  speed_conversion = 5 / 18 →
  let train_speed := train_speed_kmph * speed_conversion in
  let total_distance := train_length + platform_length in
  let time := total_distance / train_speed in
  time = 25.002 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have h_train_speed: train_speed = 72 * (5 / 18) := rfl
  rw h_train_speed
  have h_total_distance: total_distance = 160 + 340.04 := rfl
  rw h_total_distance
  have h_time: time = 500.04 / 20 := rfl
  rw h_time
  norm_num
  exact eq.refl 25.002

end train_crosses_platform_time_l805_805549


namespace integer_count_satisfies_inequality_l805_805295

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805295


namespace solve_inequality_l805_805767

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  -3 * (x^2 - 4 * x + 16) * (x^2 + 6 * x + 8) / ((x^3 + 64) * (Real.sqrt (x^2 + 4 * x + 4))) ≤ x^2 + x - 3

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ Set.Iic (-4) ∪ {x : ℝ | -4 < x ∧ x ≤ -3} ∪ {x : ℝ | -2 < x ∧ x ≤ -1} ∪ Set.Ici 0

-- The theorem statement, which we need to prove
theorem solve_inequality : ∀ x : ℝ, inequality x ↔ solution_set x :=
by
  intro x
  sorry

end solve_inequality_l805_805767


namespace percentage_decrease_l805_805448

theorem percentage_decrease (x : ℝ) 
  (h1 : 400 * (1 - x / 100) * 1.40 = 476) : 
  x = 15 := 
by 
  sorry

end percentage_decrease_l805_805448


namespace power_neg_two_thirds_eq_one_over_sqrt_a_cubed_l805_805980

variable (a : ℝ) (ha : a ≠ 0)

theorem power_neg_two_thirds_eq_one_over_sqrt_a_cubed : a^(-2/3) = 1 / real.sqrt (a^3) :=
sorry

end power_neg_two_thirds_eq_one_over_sqrt_a_cubed_l805_805980


namespace simplify_vectors_l805_805766

variables {V : Type*} [AddCommGroup V]

-- Assume the existence of vectors AB, BC, AC, AD, and DC
variables (AB BC AC AD DC : V)

-- Given conditions as relations
axiom H1 : AB + BC = AC
axiom H2 : AC - AD = DC

-- The theorem to prove
theorem simplify_vectors : AB + BC - AD = DC :=
by
  rw [H1],
  rw [H2],
  sorry

end simplify_vectors_l805_805766


namespace find_cos_A_l805_805623

variable {A : Real}

theorem find_cos_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.tan A = 2 / 3) : Real.cos A = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end find_cos_A_l805_805623


namespace total_spent_l805_805365

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l805_805365


namespace rectangle_diagonal_and_perimeter_l805_805909

theorem rectangle_diagonal_and_perimeter 
  (a b : ℝ) 
  (h₁ : a = 40 * Real.sqrt 3) 
  (h₂ : b = 30 * Real.sqrt 3) :
  (Real.sqrt (a^2 + b^2) = 50 * Real.sqrt 3) ∧ (2 * (a + b) = 140 * Real.sqrt 3) := 
by 
  split; 
  { sorry }

end rectangle_diagonal_and_perimeter_l805_805909


namespace sqrt_fraction_mult_sqrt_simp_l805_805148

theorem sqrt_fraction_mult_sqrt_simp :
  sqrt (2 / 3) * sqrt 6 = 2 := by
  sorry

end sqrt_fraction_mult_sqrt_simp_l805_805148


namespace median_eq_6point5_l805_805064
open Nat

def median_first_twelve_positive_integers (l : List ℕ) : ℝ :=
  (l !!! 5 + l !!! 6) / 2

theorem median_eq_6point5 : median_first_twelve_positive_integers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 6.5 :=
by
  sorry

end median_eq_6point5_l805_805064


namespace ellipse_properties_l805_805987

noncomputable def ellipse_equation {a b c : ℝ} (h1 : b = 1) (h2 : c = sqrt 2) (h3 : a^2 - b^2 = c^2) : Prop :=
  (x y : ℝ) → (x^2 / 3 + y^2 = 1)

noncomputable def max_chord {x y k : ℝ} : Prop :=
  (y = x + 1) ∨ (y = -x + 1)

theorem ellipse_properties (a b c : ℝ) (h1 : b = 1) (h2 : c = sqrt 2) (h3 : a^2 - b^2 = c^2)
    (h4 : ∀ (x y : ℝ), x^2 / 3 + y^2 = 1 ∧ y = x + 1 ∨ y = -x + 1 ∧ x ∈ E) :
  ellipse_equation h1 h2 h3 ∧ max_chord :=
sorry

end ellipse_properties_l805_805987


namespace min_value_distance_l805_805652

-- Definitions for the conditions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4
def tangent_line (M : ℝ × ℝ) : Prop := ∃ x y : ℝ, M = (x, y) ∧ x - y - 6 = 0

-- The proof problem
theorem min_value_distance : 
  ∀ (M N : ℝ × ℝ), 
  tangent_line M → 
  (∃ C : ℝ × ℝ, circle_eq (fst C) (snd C) ∧ C = (1,1) ∧ (∃ r : ℝ, r = 2)) →  -- center and radius are given
  (∀ x y : ℝ, (x = 1 ∧ y = 1) → (x - y - 6 = 0)) → -- Point and line relation
  ∃ d : ℝ, d = sqrt 14 :=
by sorry

end min_value_distance_l805_805652


namespace farmer_total_cows_l805_805117

theorem farmer_total_cows (cows : ℕ) 
  (h1 : 1 / 3 + 1 / 6 + 1 / 8 = 5 / 8) 
  (h2 : (3 / 8) * cows = 15) : 
  cows = 40 := by
  -- Given conditions:
  -- h1: The first three sons receive a total of 5/8 of the cows.
  -- h2: The fourth son receives 3/8 of the cows, which is 15 cows.
  sorry

end farmer_total_cows_l805_805117


namespace employee_price_l805_805510

theorem employee_price (wholesale_cost retail_markup employee_discount : ℝ) 
    (h₁ : wholesale_cost = 200) 
    (h₂ : retail_markup = 0.20) 
    (h₃ : employee_discount = 0.25) : 
    (wholesale_cost * (1 + retail_markup)) * (1 - employee_discount) = 180 := 
by
  sorry

end employee_price_l805_805510


namespace mean_and_variance_y_l805_805645

-- Define the conditions
def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) (mean_val : ℝ) : ℝ :=
  data.map (λ x => (x - mean_val) ^ 2).sum / data.length

-- Given conditions
axiom mean_x : mean [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] = 1
axiom variance_x : variance [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] 1 = 4
axiom non_zero_a : a ≠ 0

-- Definition of y_i
def y (x : ℝ) : ℝ := x + a

-- List of y values
def y_list : List ℝ := [y x1, y x2, y x3, y x4, y x5, y x6, y x7, y x8, y x9, y x10]

-- Proof problem
theorem mean_and_variance_y :
  mean y_list = 1 + a ∧ variance y_list (mean y_list) = 4 :=
by
  sorry

end mean_and_variance_y_l805_805645


namespace calculate_expression_l805_805573

theorem calculate_expression : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end calculate_expression_l805_805573


namespace difference_of_solutions_l805_805843

theorem difference_of_solutions (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ∃ a b : ℝ, a ≠ b ∧ (x = a ∨ x = b) ∧ abs (a - b) = 22 :=
by
  sorry

end difference_of_solutions_l805_805843


namespace sequence_term_general_sequence_sum_term_general_l805_805631

theorem sequence_term_general (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S (n + 1) = 2 * S n + 1) →
  a 1 = 1 →
  (∀ n ≥ 1, a n = 2^(n-1)) :=
  sorry

theorem sequence_sum_term_general (na : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ k, na k = k * 2^(k-1)) →
  (∀ n, T n = (n - 1) * 2^n + 1) :=
  sorry

end sequence_term_general_sequence_sum_term_general_l805_805631


namespace quartic_poly_exists_l805_805937

noncomputable def quartic_poly_with_given_roots (p : ℚ[X]) : Prop :=
  p.monic ∧ p.coeff 4 = 1 ∧
  (p.eval (3 + real.sqrt 5) = 0) ∧ (p.eval (3 - real.sqrt 5) = 0) ∧
  (p.eval (2 - real.sqrt 7) = 0) ∧ (p.eval (2 + real.sqrt 7) = 0)

theorem quartic_poly_exists :
  ∃ p : ℚ[X], quartic_poly_with_given_roots p ∧ p = (X^4 - 10*X^3 + 13*X^2 + 18*X - 12) :=
begin
  sorry
end

end quartic_poly_exists_l805_805937


namespace part1_part2_part3_l805_805996

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - 4

theorem part1 (a : ℝ) (ha : a = 3) : ∃ (m : ℝ), ∃ (b : ℝ), m = 3 ∧ b = -5 ∧ (∀ (x y : ℝ), y = f 3 x → y = m * x + b) := 
sorry

theorem part2 (a : ℝ) : 
  let crit_points := {0, (2 / 3) * a } in  -- Critical points
  (∀ x : ℝ, if a < 0 then (x < (2 / 3) * a ∨ (0 < x)) else (x < 0 ∨ x > (2 / 3) * a)) := 
sorry

theorem part3 : ∀ (a : ℝ), (∃ x₀ : ℝ, 0 < x₀ ∧ f a x₀ > 0) ↔ a > 3 :=
sorry

end part1_part2_part3_l805_805996


namespace sum_of_integers_a_l805_805333

theorem sum_of_integers_a :
  ∑ a in {-1, 1, 3}.toFinset, a = 3 :=
by
  have h1 : ∀ a, ∃ x : ℕ, x + 2 = 3 * (x - 1) + a ↔ ∃ (n : ℕ), n = (5 - a) / 2 ∧  0 < (5 - a) / 2 := 
    sorry
  
  have h2 : ∀ a, (6 + 4 * a) ≥ 0 :=
    sorry

  have h3 : ∀ a, ∀ y, (∃ x : ℕ, x + 2 = 3 * (x - 1) + a) →  
    (∀ y, (y - 4 * a) / 3 ≤ 2 → (y + 1) / 3 < 3 - y)  ->  
    ∃ y, y < 2 :=
    sorry

  have h4 : (∀ a : ℤ, (-1 ≤ a) ∧ (a ≤ 3)) ∧ (∀ n : ℕ , (n = ((5 - a) / 2 )) ∧  0 < ((5 - a) / 2 ) ) := 
    sorry  
  -- Now, let's prove the sum
  sorry

end sum_of_integers_a_l805_805333


namespace probability_2_le_ξ_lt_4_l805_805221

variables {μ δ : ℝ}

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ :=
MeasureTheory.ProbabilityMeasure.gaussian μ δ

theorem probability_2_le_ξ_lt_4 (h₁ : MeasureTheory.ProbabilityMeasure.probMeasure ξ {x | x < 2} = 0.15)
                               (h₂ : MeasureTheory.ProbabilityMeasure.probMeasure ξ {x | x > 6} = 0.15) :
                               MeasureTheory.ProbabilityMeasure.probMeasure ξ {x | 2 ≤ x ∧ x < 4} = 0.35 :=
by
  sorry

end probability_2_le_ξ_lt_4_l805_805221


namespace distance_symmetric_point_l805_805218

noncomputable def point (x y z : ℝ) := (x, y, z)
noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

theorem distance_symmetric_point :
  let M := point 2 (-3) 1
      N := point (-2) 3 (-1) in
  distance M N = real.sqrt 56 :=
by
  -- proof goes here
  sorry

end distance_symmetric_point_l805_805218


namespace points_on_same_circle_l805_805207

structure Point :=
(x : ℝ)
(y : ℝ)

def on_circle (A B C D : Point) : Prop :=
∃ (D E F : ℝ), (A.x^2 + A.y^2 + D * A.x + E * A.y + F = 0) ∧ 
               (B.x^2 + B.y^2 + D * B.x + E * B.y + F = 0) ∧
               (C.x^2 + C.y^2 + D * C.x + E * C.y + F = 0) ∧
               (D.x^2 + D.y^2 + D * D.x + E * D.y + F = 0)

def A := Point.mk (-1) 5
def B := Point.mk 5 5
def C := Point.mk (-3) 1
def D := Point.mk 6 (-2)

theorem points_on_same_circle : on_circle A B C D :=
sorry

end points_on_same_circle_l805_805207


namespace min_value_range_l805_805660

theorem min_value_range:
  ∀ (x m n : ℝ), 
    (y = (3 * x + 2) / (x - 1)) → 
    (∀ x ∈ Set.Ioo m n, y ≥ 3 + 5 / (x - 1)) → 
    (y = 8) → 
    n = 2 → 
    (1 ≤ m ∧ m < 2) := by
  sorry

end min_value_range_l805_805660


namespace quadratic_equation_in_x_l805_805687

theorem quadratic_equation_in_x (m : ℤ) (h1 : abs m = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
sorry

end quadratic_equation_in_x_l805_805687


namespace part_a_part_b_part_c_l805_805840

-- Part (a): Impossible to tile a 9x9 chessboard with 2x1 dominoes
theorem part_a : ¬ ∃ (f : ℕ × ℕ → ℕ × ℕ), (∀ (i j : ℕ), i < 9 ∧ j < 9 → (f (i, j)).fst - (i % 2) < 2 ∧ (f (i, j)).snd = j) := 
sorry

-- Part (b): Possible to tile a 9x9 chessboard with 3x1 triominoes
theorem part_b : ∃ (f : ℕ × ℕ → (ℕ × ℕ) × (ℕ × ℕ)), (∀ (i j : ℕ), i < 9 ∧ j < 9 → ((f (i, j)).fst.fst = i ∧ (f (i, j)).fst.snd = j ∧ 
  (((f (i, j)).snd.fst - (i % 3) < 3 ∧ (f (i, j)).snd.snd = j) ∨ 
   ((f (i, j)).snd.snd - (j % 3) < 3 ∧ (f (i, j)).snd.fst = i)))) := 
sorry

-- Part (c): Impossible to tile a 9x9 chessboard with L-shaped tetrominoes
theorem part_c : ¬ ∃ (f : ℕ × ℕ → (ℕ × ℕ) × (ℕ × ℕ × ℕ × ℕ)), (∀ (i j : ℕ), i < 9 ∧ j < 9 → 
  (let (a, b, c, d) := (f (i, j)).snd in (a.fst - i < 4 ∧ b.fst - i < 4 ∧ c.fst - i < 4 ∧ d.fst - i < 4 ∧ 
  a.snd - j < 4 ∧ b.snd - j < 4 ∧ c.snd - j < 4 ∧ d.snd - j < 4))) :=
sorry

end part_a_part_b_part_c_l805_805840


namespace remaining_area_of_square_l805_805192

theorem remaining_area_of_square :
  (let square_area := 6 * 6 in
   let dark_grey_area := 1 * 3 in
   let light_grey_area := 2 * 3 in
   square_area - dark_grey_area - light_grey_area = 27) :=
by
  let square_area := 6 * 6
  let dark_grey_area := 1 * 3
  let light_grey_area := 2 * 3
  show square_area - dark_grey_area - light_grey_area = 27
  sorry

end remaining_area_of_square_l805_805192


namespace integer_count_satisfies_inequality_l805_805294

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805294


namespace Antoinette_less_than_twice_Rupert_l805_805566

variable (R : ℕ) (A : ℕ)

-- Conditions
def condition1 : A + R = 98 := by
  sorry

def condition2 : A = 63 := by
  sorry

def condition3 : ∃ k, 2 * R - A = k := by
  sorry

-- The statement we want to prove
theorem Antoinette_less_than_twice_Rupert :
  ∃ k, k = 7 ∧ 2 * R - A = k :=
by
  -- Expanding the conditions directly into the proof problem
  unfold condition1
  unfold condition2
  unfold condition3
  sorry

end Antoinette_less_than_twice_Rupert_l805_805566


namespace kate_yellow_packs_l805_805369

def number_of_yellow_packs (red_packs yellow_balls_per_pack more_red_balls : ℕ) : ℕ :=
  let red_balls := red_packs * yellow_balls_per_pack
  (red_balls - more_red_balls) / yellow_balls_per_pack

theorem kate_yellow_packs : number_of_yellow_packs 7 18 18 = 6 :=
  by
  -- We claim that the number of yellow packs Y can be calculated as follows:
  -- red_packs = 7, yellow_balls_per_pack = 18, more_red_balls = 18
  have h1 : 7 * 18 = 126 := by norm_num
  have h2 : 126 - 18 = 108 := by norm_num
  have h3 : 108 / 18 = 6 := by norm_num
  rw [number_of_yellow_packs, h1, h2, h3]
  simp
  -- We conclude that the number of yellow packs Y is 6
  rfl

-- This statement can be checked in Lean using the above theorem.

end kate_yellow_packs_l805_805369


namespace num_int_values_satisfying_ineq_l805_805255

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805255


namespace problem_statement_l805_805231

noncomputable def transformed_function (x : ℝ) : ℝ :=
  - Math.sin (4 * x - (Real.pi / 6))

theorem problem_statement : transformed_function (Real.pi / 3) = 1 / 2 :=
by
  sorry

end problem_statement_l805_805231


namespace pyramid_surface_area_l805_805542

theorem pyramid_surface_area (s : ℝ) (h : ℝ) (area_base : ℝ) (area_triangles : ℝ) : 
  s = 6 → h = 15 → 
  area_base = 54 * Real.sqrt 3 → 
  area_triangles = 180 + 45 * Real.sqrt 3 →
  (area_base + area_triangles = 99 * Real.sqrt 3 + 135) :=
by
  intro s_eq six_eq
  intro h_eq fifteen_eq 
  intro base_eq triangle_eq 
  rw [s_eq, h_eq, base_eq, triangle_eq]
  -- Here we would normally prove it step by step, but for now we provide:
  sorry

end pyramid_surface_area_l805_805542


namespace count_int_values_cube_bound_l805_805309

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805309


namespace four_faucets_fill_time_correct_l805_805692

-- Define the parameters given in the conditions
def three_faucets_rate (volume : ℕ) (time : ℕ) := volume / time
def one_faucet_rate (rate : ℕ) := rate / 3
def four_faucets_rate (rate : ℕ) := 4 * rate
def fill_time (volume : ℕ) (rate : ℕ) := volume / rate

-- Given problem parameters
def volume_large_tub : ℕ := 100
def time_large_tub : ℕ := 6
def volume_small_tub : ℕ := 50

-- Theorem to be proven
theorem four_faucets_fill_time_correct :
  fill_time volume_small_tub (four_faucets_rate (one_faucet_rate (three_faucets_rate volume_large_tub time_large_tub))) * 60 = 135 :=
sorry

end four_faucets_fill_time_correct_l805_805692


namespace distance_between_stations_l805_805467

theorem distance_between_stations
  (time_start_train1 time_meet time_start_train2 : ℕ) -- time in hours (7 a.m., 11 a.m., 8 a.m.)
  (speed_train1 speed_train2 : ℕ) -- speed in kmph (20 kmph, 25 kmph)
  (distance_covered_train1 distance_covered_train2 : ℕ)
  (total_distance : ℕ) :
  time_start_train1 = 7 ∧ time_meet = 11 ∧ time_start_train2 = 8 ∧ speed_train1 = 20 ∧ speed_train2 = 25 ∧
  distance_covered_train1 = (time_meet - time_start_train1) * speed_train1 ∧
  distance_covered_train2 = (time_meet - time_start_train2) * speed_train2 ∧
  total_distance = distance_covered_train1 + distance_covered_train2 →
  total_distance = 155 := by
{
  sorry
}

end distance_between_stations_l805_805467


namespace closest_distance_course_l805_805418

variable (d v1 v2 : ℝ) (k : ℝ)
variable (k_pos : k > 0) (k_lt_one : k < 1)
variable (v2_pos : v2 > 0) (v1_eq_k_v2 : v1 = k * v2)

-- Condition 1: Ship O2 moves perpendicular to line connecting O1 and O2.
-- Condition 2: Ship O2 maintains its speed v2 and course.
-- Condition 3: Ship O1 moves at its maximum speed v1, attempting to approach O2.
-- Condition 4: Initial distance between O1 and O2 is d.
-- Condition 5: The speed ratio v1 / v2 is k and k < 1.

theorem closest_distance_course (α : ℝ) (sin_α_eq_k : sin α = k) (h_distance : d > 0):
  d * sqrt (1 - k^2) = d * (sqrt (1 - sin α^2)) := 
by sorry

end closest_distance_course_l805_805418


namespace expression_equals_two_l805_805422

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l805_805422


namespace number_of_frogs_is_160_l805_805705

def cats (dogs : ℕ) : ℕ := dogs - dogs / 5
def frogs (dogs : ℕ) : ℕ := 2 * dogs
def total_animals (dogs : ℕ) : ℕ := dogs + cats(dogs) + frogs(dogs)

theorem number_of_frogs_is_160 (dogs : ℕ) (h1 : total_animals dogs = 304) : frogs dogs = 160 :=
by
  sorry

end number_of_frogs_is_160_l805_805705


namespace points_symmetric_about_horizontal_axis_l805_805713

def is_symmetric (s : list (ℝ × ℝ)) : Prop :=
  s = s.map (λ p, (p.1, -p.2))

theorem points_symmetric_about_horizontal_axis 
  (n k : ℕ) 
  (M : fin n → ℝ × ℝ)
  (A : fin (n + 1) → ℝ)
  (h1 : ∀ i, i < k → (M i).2 > 0)
  (h2 : ∀ i, k ≤ i ∧ i < n → (M i).2 < 0)
  (h_condition : ∀ j : fin (n+1), 
      (∑ i in finset.range k, real.angle(M i, (A j, 0))) = 
      (∑ i in finset.range k, real.angle(M i, (A j, 0)))) :
  is_symmetric ((list.fin_range n).map M) :=
sorry

end points_symmetric_about_horizontal_axis_l805_805713


namespace merchant_boxes_fulfill_order_l805_805867

theorem merchant_boxes_fulfill_order :
  ∃ (a b c d e : ℕ), 16 * a + 17 * b + 23 * c + 39 * d + 40 * e = 100 := sorry

end merchant_boxes_fulfill_order_l805_805867


namespace min_x_in_triangle_l805_805222

theorem min_x_in_triangle (x : ℤ) (hx_pos : x > 2) (hx_bound : x < 14) :
  ∃ x : ℤ, 2 < x ∧ x < 14 ∧ x = 3 :=
by
  use 3
  split
  { linarith }
  split
  { linarith }
  { exact rfl }

end min_x_in_triangle_l805_805222


namespace ellipse_chords_constant_l805_805694

variables {R : ℝ} {M N P Q J : ℝ}
variables (MN PQ : ℝ) [IsEllipse E] {f : E.focus}

/-- If two chords MN and PQ passing through a focus of an ellipse are perpendicular, then 1/|MN| + 1/|PQ| is a constant value. -/
theorem ellipse_chords_constant (h1 : MN ≠ 0) (h2 : PQ ≠ 0)
  (h3 : (MN ⊥ PQ))
  (h4 : passes_through_focus MN PQ E f) :
  1 / |MN| + 1 / |PQ| = constant_value :=
sorry

end ellipse_chords_constant_l805_805694


namespace median_first_twelve_pos_integers_l805_805073

theorem median_first_twelve_pos_integers : 
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = 6.5 := by
  sorry

end median_first_twelve_pos_integers_l805_805073


namespace student_can_escape_l805_805714

def will_student_escape
  (s : ℝ) -- side length of the square
  (v : ℝ) -- swimming speed of the student
  (student_runs_faster : Prop) -- student can run faster than the teacher
  : Prop :=
  let o_to_shore_dist := s / real.sqrt 2 in
  let student_time := o_to_shore_dist / v in
  let teacher_time := (2 * s) / (4 * v) in
  student_time < teacher_time

theorem student_can_escape
  (s : ℝ)
  (v : ℝ)
  (student_runs_faster : Prop)
  (h_pos_s : 0 < s)
  (h_pos_v : 0 < v)
  : will_student_escape s v student_runs_faster :=
begin
  -- Proof is to be filled in
  sorry
end

end student_can_escape_l805_805714


namespace count_int_values_cube_bound_l805_805310

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805310


namespace field_trip_people_count_l805_805534

theorem field_trip_people_count :
  let standard_vans := 7 * 8
  let special_vans := 2 * 5
  let standard_buses := 9 * 27
  let special_bus := 30
  let minibuses := 4 * 15
  let standard_boats := 2 * 12
  let special_boat := 8
  standard_vans + special_vans + standard_buses + special_bus + minibuses + standard_boats + special_boat = 431 :=
by
  let standard_vans := 7 * 8
  let special_vans := 2 * 5
  let standard_buses := 9 * 27
  let special_bus := 30
  let minibuses := 4 * 15
  let standard_boats := 2 * 12
  let special_boat := 8
  show standard_vans + special_vans + standard_buses + special_bus + minibuses + standard_boats + special_boat = 431 from sorry

end field_trip_people_count_l805_805534


namespace angle_bisector_eq_angle_ratio_l805_805371

noncomputable def triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ (AB AC BC : ℝ), AB ≠ AC ∧ AB > 0 ∧ AC > 0 ∧ BC > 0

noncomputable def angle_bisector_intersection 
  (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]
  [Triangle : triangle A B C] : Prop :=
  ∃ (AB AC BC : ℝ), 
  let PQ := dist P Q in
  let AQ := dist A Q in
  let BC := dist B C in
  let sum := AB + AC in
  Triangle ∧ PQ / AQ = (BC / sum)^2

variable {A B C P Q : Type}

theorem angle_bisector_eq_angle_ratio 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]
  (h₁ : triangle A B C)
  (h₂ : angle_bisector_intersection A B C P Q) :
  ∃ (A B C P Q : Type), PQ / AQ = (BC / (AB + AC))^2 := 
sorry

end angle_bisector_eq_angle_ratio_l805_805371


namespace monic_polynomial_transformation_l805_805736

theorem monic_polynomial_transformation :
  ∀ (r1 r2 r3 : ℝ), (r1, r2, r3 ∈ {x | x^3 - 3*x^2 + 9 = 0}) → 
  ({3 * r1, 3 * r2, 3 * r3} = {x | x^3 - 9*x^2 + 243 = 0}) :=
by
  sorry

end monic_polynomial_transformation_l805_805736


namespace num_integer_solutions_l805_805287

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805287


namespace boyfriend_picks_up_correct_l805_805563

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end boyfriend_picks_up_correct_l805_805563


namespace objects_meet_probability_l805_805406

open Classical
open Finset
open Probability

noncomputable theory

def count_paths (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose n k

def total_paths (n : ℕ) : ℕ :=
  2 ^ n

def meeting_probability (n : ℕ) : ℝ :=
  (finset.sum (range (7)) (λ i, (count_paths n i : ℝ))) / (total_paths n * total_paths n)

theorem objects_meet_probability :
  meeting_probability 9 = 0.162 :=
by
  have : meeting_probability 9 =
    (Nat.choose 9 0 * Nat.choose 9 0 + Nat.choose 9 1 * Nat.choose 9 1 + Nat.choose 9 2 * Nat.choose 9 2 +
    Nat.choose 9 3 * Nat.choose 9 3 + Nat.choose 9 4 * Nat.choose 9 4 + Nat.choose 9 5 * Nat.choose 9 5 + 
    Nat.choose 9 6 * Nat.choose 9 6) / (2 ^ 18) := rfl
  calc
    meeting_probability 9 = (1 + 81 + 1296 + 6561 + 11664 + 9025 + 2916) / 262144 : by simp [this]
    ... = 42544 / 262144 : by norm_num
    ... = 0.162 : by norm_num

end objects_meet_probability_l805_805406


namespace largest_rectangle_area_l805_805130

theorem largest_rectangle_area (l w : ℕ) (hl : l > 0) (hw : w > 0) (hperimeter : 2 * l + 2 * w = 42)
  (harea_diff : ∃ (l1 w1 l2 w2 : ℕ), l1 > 0 ∧ w1 > 0 ∧ l2 > 0 ∧ w2 > 0 ∧ 2 * l1 + 2 * w1 = 42 
  ∧ 2 * l2 + 2 * w2 = 42 ∧ (l1 * w1) - (l2 * w2) = 90) : (l * w ≤ 110) :=
sorry

end largest_rectangle_area_l805_805130


namespace elements_start_with_one_l805_805381

theorem elements_start_with_one :
  let T := {x : ℤ | ∃ k : ℕ, (0 ≤ k ∧ k ≤ 100 ∧ x = 3^k)} in
  (∃ n : ℕ, length (to_digits 10 (3^100)) = 48) →
  card {x ∈ T | ∃ d : ℕ, (d > 0 ∧ d < 10 ∧ x / 10^(digit_length x - 1) = d ∧ d = 1)} = 53 :=
by
  intro T h
  sorry

end elements_start_with_one_l805_805381


namespace integer_count_satisfies_inequality_l805_805298

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805298


namespace fraction_of_milk_in_mug1_is_1_over_4_l805_805729

-- Define the initial amounts in Mug 1 and Mug 2
def initial_tea_in_mug1 : ℝ := 6
def initial_milk_in_mug2 : ℝ := 6

-- Define the transfers according to the problem statement
def transfer1 : ℝ := initial_tea_in_mug1 / 3
def mug1_after_transfer1_tea : ℝ := initial_tea_in_mug1 - transfer1
def mug2_after_transfer1_tea : ℝ := transfer1
def mug2_after_transfer1_milk : ℝ := initial_milk_in_mug2

def total_mug2_after_transfer1 : ℝ := mug2_after_transfer1_tea + mug2_after_transfer1_milk
def transfer2 : ℝ := total_mug2_after_transfer1 / 4
def transfer2_tea : ℝ := transfer2 * (mug2_after_transfer1_tea / total_mug2_after_transfer1)
def transfer2_milk : ℝ := transfer2 * (mug2_after_transfer1_milk / total_mug2_after_transfer1)

def mug1_after_transfer2_tea : ℝ := mug1_after_transfer1_tea + transfer2_tea
def mug1_after_transfer2_milk : ℝ := transfer2_milk

def total_mug1_after_transfer2 : ℝ := mug1_after_transfer2_tea + mug1_after_transfer2_milk
def transfer3 : ℝ := total_mug1_after_transfer2 / 3
def transfer3_tea : ℝ := transfer3 * (mug1_after_transfer2_tea / total_mug1_after_transfer2)
def transfer3_milk : ℝ := transfer3 * (mug1_after_transfer2_milk / total_mug1_after_transfer2)

def final_tea_in_mug1 : ℝ := mug1_after_transfer2_tea - transfer3_tea
def final_milk_in_mug1 : ℝ := mug1_after_transfer2_milk - transfer3_milk

def final_total_in_mug1 : ℝ := final_tea_in_mug1 + final_milk_in_mug1

-- Define the fraction of milk in the first mug at the end
def final_fraction_of_milk_in_mug1 : ℝ := final_milk_in_mug1 / final_total_in_mug1

-- The proof statement
theorem fraction_of_milk_in_mug1_is_1_over_4 :
  final_fraction_of_milk_in_mug1 = 1 / 4 :=
sorry

end fraction_of_milk_in_mug1_is_1_over_4_l805_805729


namespace greatest_large_chips_l805_805036

theorem greatest_large_chips (s l p : ℕ) (h1 : s + l = 80) (h2 : s = l + p) (hp : Nat.Prime p) : l ≤ 39 :=
by
  sorry

end greatest_large_chips_l805_805036


namespace number_of_terms_with_odd_coefficients_l805_805173

theorem number_of_terms_with_odd_coefficients (n : ℕ) (h : n ≥ 1):
  (∏ (i : ℕ) in Finset.range (n - 1), (∏ (j : ℕ) in Finset.Ico (i + 1) n, (λ x : ℕ → ℕ, x i + x j)))
  = n! := sorry

end number_of_terms_with_odd_coefficients_l805_805173


namespace find_common_difference_l805_805974

-- Define an arithmetic sequence
def arith_sequence (a₁ d : ℕ → ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arith_sequence (a₁ d : ℕ → ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

-- Given conditions
variables (a₁ d : ℝ)
axiom A4_A6 : arith_sequence a₁ d 4 + arith_sequence a₁ d 6 = 10
axiom S5 : sum_arith_sequence a₁ d 5 = 5

-- Prove the common difference d = 2
theorem find_common_difference : d = 2 :=
  sorry

end find_common_difference_l805_805974


namespace quadratic_solution_l805_805639

-- Definition of the quadratic function satisfying the given conditions
def quadraticFunc (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (f (-1) = 12 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12)

-- The proof goal: proving the function f(x) is 2x^2 - 10x
theorem quadratic_solution (f : ℝ → ℝ) (h : quadraticFunc f) : ∀ x, f x = 2 * x^2 - 10 * x :=
by
  sorry

end quadratic_solution_l805_805639


namespace addition_of_counts_l805_805558

-- Defining the range and the sets for prime, even, perfect square, and composite numbers
def integers := {n | 0 <= n ∧ n <= 9}
def primes := {n | n ∈ integers ∧ n ∈ {2, 3, 5, 7}}
def evens := {n | n ∈ integers ∧ n % 2 = 0}
def perfect_squares := {n | n ∈ integers ∧ n ∈ {0, 1, 4, 9}}
def composites := {n | n ∈ integers ∧ n ∈ {4, 6, 8, 9}}

-- Defining the counts of each category
def x := primes.to_finset.card
def y := evens.to_finset.card
def z := perfect_squares.to_finset.card
def u := composites.to_finset.card

-- Stating the problem as a theorem in Lean 4
theorem addition_of_counts : x + y + z + u = 17 := by
  -- This is where the proof would go; we'll use sorry for now
  sorry

end addition_of_counts_l805_805558


namespace ajay_total_gain_l805_805137

-- Definitions for the given conditions
def weightA := 15
def rateA := 14.50

def weightB := 10
def rateB := 13

def weightC := 12
def rateC := 16

def weightD := 8
def rateD := 18

def weightX := 20
def rateX := 17

def weightY := 15
def rateY := 17.50

def rateZ := 18

-- Calculation functions
def cost := (weightA * rateA) + (weightB * rateB) + (weightC * rateC) + (weightD * rateD)
def total_dal := weightA + weightB + weightC + weightD

def revenueX := weightX * rateX
def revenueY := weightY * rateY
def remaining_dal := total_dal - (weightX + weightY)
def revenueZ := remaining_dal * rateZ
def total_revenue := revenueX + revenueY + revenueZ

def total_gain := total_revenue - cost

-- Lean proof statement
theorem ajay_total_gain : total_gain = 99 := by
  sorry

end ajay_total_gain_l805_805137


namespace marble_problem_l805_805458

theorem marble_problem (R B : ℝ) 
  (h1 : R + B = 6000) 
  (h2 : (R + B) - |R - B| = 4800) 
  (h3 : B > R) : B = 3600 :=
sorry

end marble_problem_l805_805458


namespace any_nat_as_fraction_form_l805_805085

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l805_805085


namespace sum_of_solutions_l805_805166

theorem sum_of_solutions : 
  let a := 1 
  let b := -9 
  let c := 20 
  x^2 - 9*x + 20 = 0 -> 
  ∑ (x : ℝ) in {x | x^2 - 9*x + 20 = 0}.to_finset, x = 9 := 
by 
  sorry

end sum_of_solutions_l805_805166


namespace geometry_problem_l805_805716

open Set Function Topology

variable {A B C D E F G H : Point} -- Point is a placeholder for an appropriate geometric point type

-- Assuming these definitions
def is_square (A B C D : Point) : Prop := sorry -- Definition to be used, squares have specific properties
def midpoint (E : Point) (A B : Point) : Prop := sorry -- E is midpoint of AB
def on_line (F : Point) (B C : Point) : Prop := sorry -- F on line extending BC
def intersects (F G : Point) (C D : Point) : Prop := sorry -- F and G intersect on line CD
def parallel (GH AD : Line) : Prop := sorry -- GH parallel to AD
def intersect_on (BH FG : Line) (AD : Line) : Prop := sorry -- BH and FG intersect on line AD
def line_through (A : Point) (EF : Line) : Line := sorry -- Line through A parallel to EF

-- Assuming the conditions
axiom square_configuration : is_square A B C D
axiom midpoint_E : midpoint E A B
axiom arbitrary_F : on_line F B C
axiom G_line_through_A_parallel_EF : intersects (line_through A (some arbitrary_line_parallel_to_EF)) C D

-- Statement of the theorem, with conditions leading to two conclusions
theorem geometry_problem 
  (EF : Line) -- Line through E and F
  (midline_PARALLEL_AB : Line) 
  : parallel (line_through H G) AD ∧ intersect_on (line_through B H) (line_through F G) AD := 
sorry

end geometry_problem_l805_805716


namespace product_possible_values_N_l805_805906

theorem product_possible_values_N (N : ℤ) (D B : ℤ) (D6 B6 : ℤ) 
      (h1 : D = B + N) 
      (h2 : D6 = D - 8) 
      (h3 : B6 = B + 2) 
      (h4 : |D6 - B6| = 3) :
  N = 13 ∨ N = 7 → (13 * 7 = 91) := 
by 
  intro hN 
  cases hN 
  case or.inl h₁ => 
    have : N = 13 := h₁ 
    have : 13 * 7 = 91 := by norm_num 
    assumption 
  case or.inr h₂ => 
    have : N = 7 := h₂ 
    have : 7 * 13 = 91 := by norm_num 
    assumption

end product_possible_values_N_l805_805906


namespace right_triangle_bc_l805_805382

open Real

noncomputable def distance (x y : ℝ) : ℝ := (x^2 + y^2)^0.5
noncomputable def length (a b : ℝ) := distance (b - a) 0

theorem right_triangle_bc
  (A B C D E : ℝ)
  (midpoint_D : D = (A + B) / 2)
  (midpoint_E : E = (A + C) / 2)
  (BD : distance (B - D) 0 = 25)
  (EC : distance (E - C) 0 = 16)
  : distance ((B - A)^2 + (C - A)^2)^0.5 = 12 * sqrt 5 :=
sorry

end right_triangle_bc_l805_805382


namespace max_cables_cut_l805_805035

theorem max_cables_cut (computers cables clusters : ℕ) (h_computers : computers = 200) (h_cables : cables = 345) (h_clusters : clusters = 8) :
  ∃ k : ℕ, k = cables - (computers - clusters + 1) ∧ k = 153 :=
by
  sorry

end max_cables_cut_l805_805035


namespace triangle_area_40_l805_805545

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  base * height / 2

theorem triangle_area_40
  (a : ℕ) (P B Q : (ℕ × ℕ)) (PB_side : (P.1 = 0 ∧ P.2 = 0) ∧ (B.1 = 10 ∧ B.2 = 0))
  (Q_vert_aboveP : Q.1 = 0 ∧ Q.2 = 8)
  (PQ_perp_PB : P.1 = Q.1)
  (PQ_length : (Q.snd - P.snd) = 8) :
  area_of_triangle 10 8 = 40 := by
  sorry

end triangle_area_40_l805_805545


namespace fraction_division_l805_805486

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805486


namespace theta_quadrant_l805_805681

theorem theta_quadrant (θ : ℝ) (h : Real.sin (2 * θ) < 0) : 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) ∨ (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
sorry

end theta_quadrant_l805_805681


namespace monkey_climbing_time_l805_805537

-- Define the conditions
def tree_height : ℕ := 20
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2
def net_distance_per_hour : ℕ := hop_distance - slip_distance

-- Define the theorem statement
theorem monkey_climbing_time : ∃ (t : ℕ), t = 18 ∧ (net_distance_per_hour * (t - 1) + hop_distance) >= tree_height :=
by
  sorry

end monkey_climbing_time_l805_805537


namespace area_parallelogram_ABCD_l805_805850

-- Definitions for the conditions
variable (A B C D E G : Point)
variable (parallelogram_ABCD : Parallelogram A B C D)
variable (is_midpoint_E : Midpoint E B C)
variable (line_AE_intersects_BD_at_G : Intersects (Line A E) (Line B D) G)
variable (area_triangle_BEG : Area (Triangle B E G) = 1)

-- Theorem to prove
theorem area_parallelogram_ABCD :
  ParallelogramArea parallelogram_ABCD = 12 :=
sorry

end area_parallelogram_ABCD_l805_805850


namespace evaluate_f_f_1_l805_805965

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else log (x - 1)

theorem evaluate_f_f_1 : f (f 1) = 0 :=
by
  sorry

end evaluate_f_f_1_l805_805965


namespace jihye_wallet_total_l805_805341

-- Declare the amounts
def notes_amount : Nat := 2 * 1000
def coins_amount : Nat := 560

-- Theorem statement asserting the total amount
theorem jihye_wallet_total : notes_amount + coins_amount = 2560 := by
  sorry

end jihye_wallet_total_l805_805341


namespace seconds_in_minutes_l805_805319

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l805_805319


namespace minimize_broken_line_l805_805973

/-- Given an angle M O N and two points A and B, the points C on line MO and D on line NO
that minimize the length of the broken line ACDB are determined using the reflective symmetry 
method. -/
theorem minimize_broken_line (M O N A B C D : Point) 
  (h_MO : C ∈ line MO) 
  (h_NO : D ∈ line NO) : 
  -- The reflective symmetry principle gives the shortest broken line
  minimize_broken_line_acdb C D ↔ reflective_symmetry A B C D A' B' :=
sorry

end minimize_broken_line_l805_805973


namespace g_60_l805_805012

noncomputable def g : ℝ → ℝ :=
sorry

axiom g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

axiom g_45 : g 45 = 15

theorem g_60 : g 60 = 11.25 :=
by
  sorry

end g_60_l805_805012


namespace initial_animal_types_l805_805908

theorem initial_animal_types (x : ℕ) (h1 : 6 * (x + 4) = 54) : x = 5 := 
sorry

end initial_animal_types_l805_805908


namespace min_rental_cost_l805_805135

theorem min_rental_cost :
  ∃ (x y : ℕ), 36 * x + 60 * y ≥ 900 ∧ x + y ≤ 21 ∧ y - x ≤ 7 ∧ 1600 * x + 2400 * y = 36800 ∧ x = 5 ∧ y = 12 :=
by
  use [5, 12]
  -- All conditions need to be verified
  have h1 : 36 * 5 + 60 * 12 ≥ 900, by linarith,
  have h2 : 5 + 12 ≤ 21, by norm_num,
  have h3 : 12 - 5 ≤ 7, by norm_num,
  have cost_eq : 1600 * 5 + 2400 * 12 = 36800, by norm_num,
  exact ⟨h1, h2, h3, cost_eq, rfl, rfl⟩
 
end min_rental_cost_l805_805135


namespace range_of_a_for_inequality_l805_805331

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a > 0) → a > 1 :=
by
  sorry

end range_of_a_for_inequality_l805_805331


namespace median_first_twelve_integers_l805_805062

theorem median_first_twelve_integers : 
  let lst : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = (6.5 : ℤ) :=
by
  sorry

end median_first_twelve_integers_l805_805062


namespace ore_needed_approximation_l805_805904

theorem ore_needed_approximation :
  let wA : ℝ := 30 / 0.80
  let wB : ℝ := 20 / 0.50
  let wC : ℝ := 40 / 0.90
  (wA + wB + wC) ≈ 121.94 :=
by
  let wA : ℝ := 37.5
  let wB : ℝ := 40
  let wC : ℝ := 44.44
  have hA : wA = 30 / 0.80 := by rfl
  have hB : wB = 20 / 0.50 := by rfl
  have hC : wC = 40 / 0.90 := by rfl
  rw [hA, hB, hC]
  norm_num
  sorry

end ore_needed_approximation_l805_805904


namespace smallest_positive_period_of_f_triangle_area_l805_805656

def f (x : Real) : Real := sqrt 3 * sin x * sin x + sin x * cos x

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem triangle_area (A B C a b c : Real)
  (hA : f (A / 2) = sqrt 3 / 2)
  (ha : a = 4)
  (hb : b + c = 5) :
  let S := 1 / 2 * b * c * sin A
  S = 3 * sqrt 3 / 4 := sorry
 
end smallest_positive_period_of_f_triangle_area_l805_805656


namespace count_integers_in_range_num_of_integers_l805_805269

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805269


namespace circumcenter_invariance_l805_805411

open EuclideanGeometry

variable {ABC : Triangle} (I : Point) (A B C : Point) (B1 : Point) (A1 C1 : Point)

/-- Prove that the circumcenter of triangle A1B1C1 does not depend on the position of point B1 on side AC. -/
theorem circumcenter_invariance (hI : is_incenter I ABC)
  (hB1 : B1 ∈ segment A C)
  (hC1 : C1 ∈ circumcircle A B1 I ∩ segment A B)
  (hA1 : A1 ∈ circumcircle C B1 I ∩ segment B C) :
  is_circumcenter I ⟨A1, B1, C1⟩ := sorry

end circumcenter_invariance_l805_805411


namespace third_discount_percentage_l805_805024

-- Definitions based on the conditions
def initial_price : ℝ := 12000
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.15
def final_price : ℝ := 7752

-- The target statement we need to prove
theorem third_discount_percentage :
  let price_after_first_discount := initial_price * (1 - first_discount_rate),
      price_after_second_discount := price_after_first_discount * (1 - second_discount_rate),
      third_discount_rate := (price_after_second_discount - final_price) / price_after_second_discount in
  third_discount_rate * 100 = 5 :=
by
  sorry

end third_discount_percentage_l805_805024


namespace tax_rate_l805_805131

noncomputable def payroll_tax : Float := 300000
noncomputable def tax_paid : Float := 200
noncomputable def tax_threshold : Float := 200000

theorem tax_rate (tax_rate : Float) : 
  (payroll_tax - tax_threshold) * tax_rate = tax_paid → tax_rate = 0.002 := 
by
  sorry

end tax_rate_l805_805131


namespace perimeter_of_triangle_l805_805023

def triangle_inradius : ℝ := 2.5
def triangle_area : ℝ := 45

theorem perimeter_of_triangle (r : ℝ) (A : ℝ) (p : ℝ) (h1 : r = triangle_inradius) (h2 : A = triangle_area) : 
    A = r * p / 2 → p = 36 :=
by
  assume h : A = r * p / 2,
  sorry

end perimeter_of_triangle_l805_805023


namespace factor_w4_minus_16_l805_805601

theorem factor_w4_minus_16 (w : ℝ) : (w^4 - 16) = (w - 2) * (w + 2) * (w^2 + 4) :=
by
    sorry

end factor_w4_minus_16_l805_805601


namespace minnie_takes_longer_l805_805401

-- Definitions for the given problem conditions
def minnie_speed_flat : ℝ := 25 -- Minnie's speed on flat road in kph
def minnie_speed_uphill : ℝ := 10 -- Minnie's speed uphill in kph
def minnie_speed_downhill : ℝ := 35 -- Minnie's speed downhill in kph
def penny_speed_flat : ℝ := 35 -- Penny's speed on flat road in kph
def penny_speed_uphill : ℝ := 15 -- Penny's speed uphill in kph
def penny_speed_downhill : ℝ := 45 -- Penny's speed downhill in kph

def distance_AB : ℝ := 12 -- Distance from A to B in km (flat)
def distance_BC : ℝ := 18 -- Distance from B to C in km (uphill)
def distance_CA : ℝ := 25 -- Distance from C to A in km (downhill)

def minnie_time : ℝ := (distance_AB / minnie_speed_flat) + (distance_BC / minnie_speed_uphill) + (distance_CA / minnie_speed_downhill)
def penny_time : ℝ := (distance_CA / penny_speed_downhill) + (distance_BC / penny_speed_uphill) + (distance_AB / penny_speed_flat)
def time_difference_minutes : ℝ := (minnie_time - penny_time) * 60

-- The required proof statement
theorem minnie_takes_longer : time_difference_minutes = 54 :=
by
  sorry

end minnie_takes_longer_l805_805401


namespace sum_positive_real_solutions_eq_540pi_l805_805613

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos (2 * x) * (cos (2 * x) - cos (2024 * π^2 / x^2))

theorem sum_positive_real_solutions_eq_540pi :
  (∀ x : ℝ, x > 0 → f x = cos (4 * x) - 1) →
  let solutions := {x | x > 0 ∧ f x = cos (4 * x) - 1}
  ∑ x in solutions, x = 540 * π :=
sorry

end sum_positive_real_solutions_eq_540pi_l805_805613


namespace mean_inequality_l805_805157

theorem mean_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) :
  (a + b + c) / 3 > Real.cbrt (a * b * c) ∧ Real.cbrt (a * b * c) > (3 * a * b * c) / (a * b + b * c + c * a) :=
by
  sorry

end mean_inequality_l805_805157


namespace count_integers_in_range_num_of_integers_l805_805266

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805266


namespace michelle_gas_left_l805_805748

def gasLeft (initialGas: ℝ) (usedGas: ℝ) : ℝ :=
  initialGas - usedGas

theorem michelle_gas_left :
  gasLeft 0.5 0.3333333333333333 = 0.1666666666666667 :=
by
  -- proof goes here
  sorry

end michelle_gas_left_l805_805748


namespace smallest_x_with_24_factors_and_18_28_factors_l805_805016

theorem smallest_x_with_24_factors_and_18_28_factors : 
  ∃ x : ℕ, 
    (∀ d : ℕ, d ∣ x → d ∈ {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}) ∧ 
    ∏ m in {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}, (1 : ℤ / m) = 24 := 
  ∃ x : ℕ, nat.factors x = {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}.toFinset ∧ x = 504 := 
  sorry

end smallest_x_with_24_factors_and_18_28_factors_l805_805016


namespace calculate_expression_l805_805578

-- Definitions based on conditions
def step1 : Int := 12 - (-18)
def step2 : Int := step1 + (-7)
def final_result : Int := 23

-- Theorem to prove
theorem calculate_expression : step2 = final_result := by
  have h1 : step1 = 12 + 18 := by sorry
  have h2 : step2 = step1 - 7 := by sorry
  rw [h1, h2]
  norm_num
  sorry

end calculate_expression_l805_805578


namespace hyperbola_equation_l805_805644

theorem hyperbola_equation (x y : ℝ) (h1 : (4, real.sqrt 3) = (4, real.sqrt 3)) (h2 : ∀ x, y = x / 2 ∨ y = - x / 2) :
  (y^2 - x^2 / 4 = -1) ↔ (x^2 / 4 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l805_805644


namespace binary_addition_to_decimal_l805_805473

theorem binary_addition_to_decimal : nat.ofDigits 2 [1, 0, 1, 0, 1, 0, 1] + nat.ofDigits 2 [1, 1, 1, 0, 0, 0] = 141 := by
  sorry

end binary_addition_to_decimal_l805_805473


namespace division_of_fractions_l805_805489

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805489


namespace problem_a_problem_b_l805_805757

-- Statement for problem a)
theorem problem_a (n : ℕ) (h : n = 12) :
  (finset.range n).sum + 1 ≡ (finset.range n).sum (λ i, 2^i) + 1 [MOD 13] :=
sorry

-- Statement for problem b)
theorem problem_b (n : ℕ) (h : n = 12) :
  (finset.range n).sum (λ i, (i + 1)^2) + 1 ≡ (finset.range n).sum (λ i, 4^i) + 1 [MOD 13] :=
sorry

end problem_a_problem_b_l805_805757


namespace find_x_l805_805805

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l805_805805


namespace find_polynomial_l805_805604

def isSolution (u_n v_n : ℕ → ℤ) := 
  ∀ n, (2 + real.sqrt 3) ^ n = u_n n + v_n n * real.sqrt 3

def bounded_sequence (u_n : ℕ → ℤ) := 
  ∀ n, 0 < (2 - real.sqrt 3) ^ n < 2 ∧
  (u_n n = ⌈ ((2 + real.sqrt 3) ^ n) / 2 ⌉)

def diophantine_solution (u_n v_n : ℕ → ℤ) :=
  ∀ n, u_n n ^ 2 - 3 * v_n n ^ 2 = 1

theorem find_polynomial : 
  ∃ P : ℤ → ℤ → ℤ, 
    (∀ n, ∃ x y, x = u_n n → y = v_n n →  P x y = (⌈ ((2 + real.sqrt 3) ^ n) / 2 ⌉)) ∧ 
    (∀ x y, x² - 3 * y² - 1 ≠ 0 → P x y < 0) :=
begin
  let P := λ x y : ℤ, x - (x^2 + 1) * (x^2 - 3*y^2 - 1)^2,
  use P,
  sorry
end

end find_polynomial_l805_805604


namespace equivalent_G_F_l805_805738

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))
def h (x : ℝ) : ℝ := (3*x + x^3) / (1 + 3*x^2)
def G (x : ℝ) : ℝ := F (h x)

theorem equivalent_G_F (x : ℝ) : G x = 3 * (F x) :=
by
  -- Proof goes here
  sorry

end equivalent_G_F_l805_805738


namespace lines_parallel_l805_805851

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Triangle ABC with vertices A, B, and C
variables (A B C : Point)

-- Points A₁, A₂, B₁, B₂, C₁, C₂ defined as per the problem statement
variables (A₁ A₂ B₁ B₂ C₁ C₂ : Point)

-- Conditions given in the problem
-- Using distance for equality of segments
def dist (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

axiom cond_A1 : dist B A₁ = dist B C
axiom cond_A2 : dist C A₂ = dist B C
axiom cond_B1 : dist C B₁ = dist C A
axiom cond_B2 : dist A B₂ = dist C A
axiom cond_C1 : dist A C₁ = dist A B
axiom cond_C2 : dist B C₂ = dist A B

-- Lines A₁A₂, B₁B₂, C₁C₂ parallel
theorem lines_parallel (A B C A₁ A₂ B₁ B₂ C₁ C₂ : Point) 
  (hA₁ : dist B A₁ = dist B C) 
  (hA₂ : dist C A₂ = dist B C)
  (hB₁ : dist C B₁ = dist C A) 
  (hB₂ : dist A B₂ = dist C A) 
  (hC₁ : dist A C₁ = dist A B) 
  (hC₂ : dist B C₂ = dist A B) : 
  (∃ l : ℝ, ∀ x y z : Point, ∃ l₂ : ℝ, ∀ x y z : Point, (l // l₂)) :=
sorry

end lines_parallel_l805_805851


namespace power_function_point_l805_805991

variable {m n : ℕ}

theorem power_function_point (h : (m - 1) * m ^ n = 8) (h_m : m = 2) : n ^ -m = (3 : ℤ) ^ -2 := by
  have hm : m = 2 := h_m
  have hn : n = 3 := sorry
  have hn_pos : (3 : ℤ) ≠ 0 := by decide
  calc
    n ^ -m = (3 : ℤ)^ -2 : by rw [hn, hm]
        ... = 1 / (3 ^ 2) : by rw [zpow_neg, ←int.coe_nat_pow, zpow_two, inv_eq_one_div]
        ... = (1 / 9) : by norm_num


end power_function_point_l805_805991


namespace find_m_l805_805243

open Real

variables (m : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (m, -1)
def vec_c : ℝ × ℝ := (4, m)
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
def vec_sub (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 - y.1, x.2 - y.2)

theorem find_m
  (h : dot_product (vec_sub vec_a vec_b) vec_c = 0) : m = 4 :=
sorry

end find_m_l805_805243


namespace difference_of_squares_l805_805334

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := 
sorry

end difference_of_squares_l805_805334


namespace work_together_days_l805_805864

noncomputable def A_per_day := 1 / 78
noncomputable def B_per_day := 1 / 39

theorem work_together_days 
  (A : ℝ) (B : ℝ) 
  (hA : A = 1 / 78)
  (hB : B = 1 / 39) : 
  1 / (A + B) = 26 :=
by
  rw [hA, hB]
  sorry

end work_together_days_l805_805864


namespace smallest_x_with_24_factors_and_18_28_factors_l805_805015

theorem smallest_x_with_24_factors_and_18_28_factors : 
  ∃ x : ℕ, 
    (∀ d : ℕ, d ∣ x → d ∈ {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}) ∧ 
    ∏ m in {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}, (1 : ℤ / m) = 24 := 
  ∃ x : ℕ, nat.factors x = {1, 2, 3, 6, 9, 18, 28, 36, 42, 54, 63, 84, 108, 126, 168, 252, 504}.toFinset ∧ x = 504 := 
  sorry

end smallest_x_with_24_factors_and_18_28_factors_l805_805015


namespace g_of_36_l805_805387

theorem g_of_36 (g : ℕ → ℕ)
  (h1 : ∀ n, g (n + 1) > g n)
  (h2 : ∀ m n, g (m * n) = g m * g n)
  (h3 : ∀ m n, m ≠ n ∧ m ^ n = n ^ m → (g m = n ∨ g n = m))
  (h4 : ∀ n, g (n ^ 2) = g n * n) :
  g 36 = 36 :=
  sorry

end g_of_36_l805_805387


namespace cost_per_litre_of_mixture_l805_805587

-- Define the given volumes and prices per litre for each type of oil
def litres_A := 10
def price_A := 54
def litres_B := 5
def price_B := 66
def litres_C := 8
def price_C := 48

-- Define the total cost and total volume for the mixture
def total_cost_A := litres_A * price_A
def total_cost_B := litres_B * price_B
def total_cost_C := litres_C * price_C

def total_cost := total_cost_A + total_cost_B + total_cost_C
def total_volume := litres_A + litres_B + litres_C

-- Define the proof problem
theorem cost_per_litre_of_mixture :
    (total_cost.toReal / total_volume.toReal) = 54.52 := sorry

end cost_per_litre_of_mixture_l805_805587


namespace parabola_vertex_l805_805812

theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, t^2 + 2 * t - 2 ≥ y) ∧ (x^2 + 2 * x - 2 = y) ∧ (x = -1) ∧ (y = -3) :=
by sorry

end parabola_vertex_l805_805812


namespace sequence_eq_third_term_l805_805090

theorem sequence_eq_third_term 
  (p : ℤ → ℤ)
  (a : ℕ → ℤ)
  (n : ℕ) (h₁ : n > 2)
  (h₂ : a 2 = p (a 1))
  (h₃ : a 3 = p (a 2))
  (h₄ : ∀ k, 4 ≤ k ∧ k ≤ n → a k = p (a (k - 1)))
  (h₅ : a 1 = p (a n))
  : a 1 = a 3 :=
sorry

end sequence_eq_third_term_l805_805090


namespace area_of_hexagon_PQURTS_l805_805715

noncomputable def hexagon_area : ℝ := 16 * Real.sqrt 3

theorem area_of_hexagon_PQURTS (P Q R S T U : ℝ × ℝ)
  (hPQR : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 16 ∧ (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 16 ∧ (R.1 - P.1)^2 + (R.2 - P.2)^2 = 16)
  (hPQS : (P.1 - S.1)^2 + (P.2 - S.2)^2 = 16 ∧ (Q.1 - S.1)^2 + (Q.2 - S.2)^2 = 16)
  (hQRT : (Q.1 - T.1)^2 + (Q.2 - T.2)^2 = 16 ∧ (R.1 - T.1)^2 + (R.2 - T.2)^2 = 16)
  (hRUP : (R.1 - U.1)^2 + (R.2 - U.2)^2 = 16 ∧ (U.1 - P.1)^2 + (U.2 - P.2)^2 = 16) :
  let area_PQS := Real.sqrt 3 * 4^2 / 4,
      area_QRT := Real.sqrt 3 * 4^2 / 4,
      area_RUP := Real.sqrt 3 * 4^2 / 4,
      area_PQR := Real.sqrt 3 * 4^2 / 4 in
  area_PQS + area_QRT + area_RUP + area_PQR = hexagon_area :=
by
  sorry

end area_of_hexagon_PQURTS_l805_805715


namespace batsman_average_after_12_l805_805855

variable (A : ℝ) -- A is the average after 11 innings
constant h1 : 11 * A + 92 = 12 * (A + 2) -- Condition from problem
variable (average_after_12 : ℝ) -- The average after 12 innings

-- Assertion we will prove
theorem batsman_average_after_12 : average_after_12 = 70 :=
by
  -- Use the conditions to derive the proof
  sorry

end batsman_average_after_12_l805_805855


namespace min_sum_a1_a2_l805_805811

noncomputable def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 1 then a 1 else
  if n = 2 then a 2 else
  (a (n - 2) + 1540) / (1 + a (n - 1))

theorem min_sum_a1_a2 : 
  ∃ (a1 a2 : ℕ), a1 > 0 ∧ a2 > 0 ∧ (a1 + a2) % 5 = 0 ∧ 
  (∀ a : ℕ → ℕ, 
    sequence a 3 = (sequence a 1 + 1540) / (1 + sequence a 2) →
    sequence a 4 = (sequence a 2 + 1540) / (1 + sequence a 3) →
    sequence a 5 = (sequence a 3 + 1540) / (1 + sequence a 4) →
    a 1 = a1 ∧ a 2 = a2) ∧
  a1 + a2 = 164 :=
begin
  sorry
end

end min_sum_a1_a2_l805_805811


namespace range_of_x_l805_805241

open Set

noncomputable def M (x : ℝ) : Set ℝ := {x^2, 1}

theorem range_of_x (x : ℝ) (hx : M x) : x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end range_of_x_l805_805241


namespace median_first_twelve_pos_integers_l805_805074

theorem median_first_twelve_pos_integers : 
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = 6.5 := by
  sorry

end median_first_twelve_pos_integers_l805_805074


namespace annual_interest_rate_l805_805749

theorem annual_interest_rate (r : ℝ) (triple_time : r > 0 → 112 / r)
  (initial : ℝ) (final : ℝ) (years : ℝ) (H1 : initial > 0) (H2 : final > 0) (H3 : years > 0) 
  (triples_twice : initial * 3^2 = final) (time_condition : r > 0 → 2 * (112 / r) = years) :
  r = 8 := by
  sorry

end annual_interest_rate_l805_805749


namespace inscribed_square_area_l805_805903

theorem inscribed_square_area (area_first_square : ℝ) (area_first_square = 484) : ∃ area_second_square : ℝ, area_second_square = 430 :=
by
  sorry

end inscribed_square_area_l805_805903


namespace right_triangle_inscribed_circles_l805_805044

/-- Triangle ABC is right-angled at C and is inscribed in a circle with center O'.
    A circle with center O is inscribed in Triangle ABC.
    Side AC is along the radius of the bigger circle extending from the center O' to the point A.
    Line OA is extended to meet the larger circle at point D.
    The line O'B perpendicular to AC intersects AC at E.
    We need to prove that O'B = O'E. -/
theorem right_triangle_inscribed_circles (A B C O O' E D : Point)
  (h1 : ∠C = 90°)
  (h2 : is_circumcenter O' ABC)
  (h3 : is_incenter O ABC)
  (h4 : ∃ R, O'A = R ∧ O'C = R ∧ D lies_on_circle_radius (AC) extended)
  (h5 : perpendicular O'B AC ∧ intersection_point (O'B AC) = E):
  O'B = O'E :=
sorry

end right_triangle_inscribed_circles_l805_805044


namespace difference_between_avg_weight_and_Joe_l805_805833

-- Define the conditions given in the problem
variables (weightJoe : ℕ)
          (avgWeightOriginalGroup : ℕ)
          (increaseAfterJoeJoins : ℕ)
          (avgWeightReturnAfterLeave : ℕ)
          (n : ℕ)

-- Define the condition variables given in the problem
def conditions : Prop :=
  weightJoe = 40 ∧
  avgWeightOriginalGroup = 30 ∧
  increaseAfterJoeJoins = 1 ∧
  avgWeightReturnAfterLeave = 30

-- Define a function to determine the difference between the weight of Joe and the average weight of the two students who left
def difference_in_weight (n : ℕ) (weightJoe : ℕ) : ℕ :=
  let totalWeightBeforeJoe := avgWeightOriginalGroup * n in
  let totalWeightWithJoe := totalWeightBeforeJoe + weightJoe in
  let totalNumAfterJoe := n + 1 in
  let totalWeightAfterTwoLeave := avgWeightReturnAfterLeave * (totalNumAfterJoe - 2) in
  let totalWeightTwoLeft := totalWeightWithJoe - totalWeightAfterTwoLeave in
  let avgWeightTwoLeft := totalWeightTwoLeft / 2 in
  abs (int.nat_sub avgWeightTwoLeft weightJoe)

-- Prove the theorem statement
theorem difference_between_avg_weight_and_Joe : conditions →
  difference_in_weight n weightJoe = 5 :=
by
  sorry -- Proof is skipped

end difference_between_avg_weight_and_Joe_l805_805833


namespace sum_sequence_value_l805_805629

open Nat

def sequence (a : ℕ → ℕ) := 
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n : ℕ, n > 0 → a n * (a (n + 2) - 1) = a (n + 1) * (a (n + 1) - 1) 

noncomputable def sum_seq (a : ℕ → ℕ) : ℕ :=
  ∑ k in range 2024, choose 2023 k * a (k + 1)

theorem sum_sequence_value (a : ℕ → ℕ) (h : sequence a) :
  sum_seq a = 2 * 3^2023 - 2^2023 :=
sorry

end sum_sequence_value_l805_805629


namespace trajectory_of_P_l805_805975

-- Definitions and conditions 
def centerM : ℝ × ℝ := (-1, 0)
def radiusM : ℝ := 1
def centerN : ℝ × ℝ := (1, 0)
def radiusN : ℝ := 3

-- Definitions for Lean
def external_tangent (P : ℝ × ℝ) (r : ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = (r + radiusM)^2

def internal_tangent (P : ℝ × ℝ) (r : ℝ) : Prop :=
  (P.1 - 1)^2 + P.2^2 + (r - radiusN)^2 = 4

def ellipse_equation (P : ℝ × ℝ) : Prop :=
  P.1^2 / 4 + P.2^2 / 3 = 1

-- Statement of the theorem
theorem trajectory_of_P (P : ℝ × ℝ) (r : ℝ) :
  external_tangent P r → internal_tangent P r → ellipse_equation P :=
by {
  sorry,
}

end trajectory_of_P_l805_805975


namespace solve_for_r_l805_805000

variable (n : ℝ) (r : ℝ)

theorem solve_for_r (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n * (1 + Real.sqrt 3)) / 2 :=
by
  sorry

end solve_for_r_l805_805000


namespace find_x_l805_805804

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l805_805804


namespace midpoint_coordinates_l805_805209

theorem midpoint_coordinates (A B M : ℝ × ℝ) (hx : A = (2, -4)) (hy : B = (-6, 2)) (hm : M = (-2, -1)) :
  let (x1, y1) := A
  let (x2, y2) := B
  M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
  sorry

end midpoint_coordinates_l805_805209


namespace sufficient_but_not_necessary_l805_805976

variable (x : ℝ)

def p : Prop := x^2 ≥ 1
def q : Prop := 2^x ≤ 2
def not_p : Prop := ¬p

theorem sufficient_but_not_necessary : (not_p x → q x) ∧ ¬(q x → not_p x) := 
by sorry

end sufficient_but_not_necessary_l805_805976


namespace num_integer_solutions_l805_805288

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805288


namespace number_of_integers_satisfying_cubed_inequality_l805_805306

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805306


namespace inequality_proof_l805_805211

variable (n : ℕ)
variable (a : Fin n → ℝ)
hypothesis pos_a : ∀ i, 0 < a i
hypothesis sum_a : Finset.univ.sum a = 1

theorem inequality_proof :
  (Finset.univ.sum (λ i => a i * a ((i : ℕ + 1).mod n)))
  * (Finset.univ.sum (λ i => a i / (a ((i : ℕ + 1).mod n) ^ 2 + a ((i : ℕ + 1).mod n)))) 
  ≥ n / (n + 1) := sorry

end inequality_proof_l805_805211


namespace intersection_of_sets_l805_805328

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets : (setA ∩ { x | 1 - x^2 ∈ setB }) = Set.Icc (-1) 1 :=
by
  sorry

end intersection_of_sets_l805_805328


namespace max_expression_value_l805_805796

theorem max_expression_value :
  ∃ (a b c d : ℝ), a ∈ set.Icc (-10.5) 10.5 ∧ b ∈ set.Icc (-10.5) 10.5 ∧
  c ∈ set.Icc (-10.5) 10.5 ∧ d ∈ set.Icc (-10.5) 10.5 ∧
  (∀ (a' b' c' d' : ℝ), a' ∈ set.Icc (-10.5) 10.5 → b' ∈ set.Icc (-10.5) 10.5 →
                        c' ∈ set.Icc (-10.5) 10.5 → d' ∈ set.Icc (-10.5) 10.5 →
                        a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 462) ∧
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a = 462 :=
sorry

end max_expression_value_l805_805796


namespace walking_speed_is_correct_l805_805866

-- Define the conditions
def time_in_minutes : ℝ := 10
def distance_in_meters : ℝ := 1666.6666666666665
def speed_in_km_per_hr : ℝ := 2.777777777777775

-- Define the theorem to prove
theorem walking_speed_is_correct :
  (distance_in_meters / time_in_minutes) * 60 / 1000 = speed_in_km_per_hr :=
sorry

end walking_speed_is_correct_l805_805866


namespace smallest_six_digit_negative_integer_congruent_to_five_mod_17_l805_805831

theorem smallest_six_digit_negative_integer_congruent_to_five_mod_17 :
  ∃ x : ℤ, x < -100000 ∧ x ≥ -999999 ∧ x % 17 = 5 ∧ x = -100011 :=
by
  sorry

end smallest_six_digit_negative_integer_congruent_to_five_mod_17_l805_805831


namespace smallest_x_with_24_factors_and_factors_18_28_l805_805013

theorem smallest_x_with_24_factors_and_factors_18_28 :
  ∃ x : ℕ, (∀ n : ℕ, n > 0 → n ∣ x → (nat.factors x).length = 24) ∧ (18 ∣ x ∧ 28 ∣ x) ∧ x = 504 :=
  sorry

end smallest_x_with_24_factors_and_factors_18_28_l805_805013


namespace excircle_problem_l805_805552

-- Define the data structure for a triangle with incenter and excircle properties
structure TriangleWithIncenterAndExcircle (α : Type) [LinearOrderedField α] :=
  (A B C I X : α)
  (is_incenter : Boolean)  -- condition for point I being the incenter
  (is_excircle_center_opposite_A : Boolean)  -- condition for point X being the excircle center opposite A
  (I_A_I : I ≠ A)
  (X_A_X : X ≠ A)

-- Define the problem statement
theorem excircle_problem
  (α : Type) [LinearOrderedField α]
  (T : TriangleWithIncenterAndExcircle α)
  (h_incenter : T.is_incenter)
  (h_excircle_center : T.is_excircle_center_opposite_A)
  (h_not_eq_I : T.I ≠ T.A)
  (h_not_eq_X : T.X ≠ T.A)
  : 
    (T.I * T.X = T.A * T.B) ∧ 
    (T.I * (T.B * T.C) = T.X * (T.B * T.C)) :=
by
  sorry

end excircle_problem_l805_805552


namespace final_price_calculation_l805_805151

theorem final_price_calculation 
  (ticket_price : ℝ)
  (initial_discount : ℝ)
  (additional_discount : ℝ)
  (sales_tax : ℝ)
  (final_price : ℝ) 
  (h1 : ticket_price = 200) 
  (h2 : initial_discount = 0.25) 
  (h3 : additional_discount = 0.15) 
  (h4 : sales_tax = 0.07)
  (h5 : final_price = (ticket_price * (1 - initial_discount)) * (1 - additional_discount) * (1 + sales_tax)):
  final_price = 136.43 :=
by
  sorry

end final_price_calculation_l805_805151


namespace count_int_values_cube_bound_l805_805317

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805317


namespace abs_equation_interval_l805_805682

theorem abs_equation_interval (x : ℝ) (h : |x| + ||x| - 1| = 1) : (x + 1) * (x - 1) ≤ 0 :=
sorry

end abs_equation_interval_l805_805682


namespace contrapositive_equivalence_l805_805778

-- Define the original proposition and its contrapositive
def original_proposition (q p : Prop) := q → p
def contrapositive (q p : Prop) := ¬q → ¬p

-- The theorem to prove
theorem contrapositive_equivalence (q p : Prop) :
  (original_proposition q p) ↔ (contrapositive q p) :=
by
  sorry

end contrapositive_equivalence_l805_805778


namespace highest_numbered_street_l805_805677

/--
  Gretzky Street begins at Orr Street and runs directly east for 5600 meters until it meets Howe Street.
  Gretzky Street is intersected every 350 meters by a perpendicular street.
  Each of those streets other than Orr Street and Howe Street is given a number beginning at 1st Street and continuing consecutively.
  Prove that the highest-numbered street that intersects Gretzky Street is 14.
-/
theorem highest_numbered_street :
  ∀ (length_of_street : ℕ) (distance_between_streets : ℕ), length_of_street = 5600 → distance_between_streets = 350 → 
  (length_of_street / distance_between_streets) - 2 = 14 := by
  intros length_of_street distance_between_streets h_length h_distance
  rw [h_length, h_distance]
  norm_num
  sorry

end highest_numbered_street_l805_805677


namespace correct_statements_about_f_l805_805759

def f (x : ℝ) : ℝ := x + sin x

theorem correct_statements_about_f :
  (f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x ≤ f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0) ∧ (∀ x : ℝ, x < 0 → f x < 0) ∧
  ¬(∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x) ∧
  ¬(∀ x : ℝ, f (-x) = f x) :=
by
  sorry

end correct_statements_about_f_l805_805759


namespace box_volume_l805_805498

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l805_805498


namespace monthly_food_expense_l805_805167

-- Definitions based on the given conditions
def E : ℕ := 6000
def R : ℕ := 640
def EW : ℕ := E / 4
def I : ℕ := E / 5
def L : ℕ := 2280

-- Define the monthly food expense F
def F : ℕ := E - (R + EW + I) - L

-- The theorem stating that the monthly food expense is 380
theorem monthly_food_expense : F = 380 := 
by
  -- proof goes here
  sorry

end monthly_food_expense_l805_805167


namespace passing_thresholds_l805_805039

-- Define the total marks of the exam
def total_marks (T : ℕ) := T

-- Define the percentage of marks obtained by candidates
def marks_A (T : ℕ) := 0.25 * T
def marks_B (T : ℕ) := 0.35 * T
def marks_C (T : ℕ) := 0.40 * T

-- Define the lower passing threshold using Candidate A's information
def lower_passing_threshold_A (T : ℕ) := marks_A T + 30

-- Define the lower passing threshold using Candidate B's information
def lower_passing_threshold_B (T : ℕ) := marks_B T - 10

-- Define the higher passing threshold using Candidate C's information
def higher_passing_threshold (T : ℕ) := marks_C T

-- Prove the passing thresholds
theorem passing_thresholds (T : ℕ) :
  (lower_passing_threshold_A T = 130) ∧ (higher_passing_threshold T = 160) := by
  sorry

end passing_thresholds_l805_805039


namespace max_S_correctness_l805_805188

noncomputable def max_S_value (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum : a + b + c + d = 100) : ℝ := 
  let S := (real.cbrt (a / (b + 7)) + real.cbrt (b / (c + 7)) + real.cbrt (c / (d + 7)) + real.cbrt (d / (a + 7)))
  in if h_ne_0 : S = 4 * (25 / 32)^(1/3) then S else
  if h_le_0 : S ≤ 3.6924 then 3.6924 else 3.6924

theorem max_S_correctness (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum : a + b + c + d = 100) : 
  max_S_value a b c d h_nonneg h_sum = 3.6924 :=
  sorry

end max_S_correctness_l805_805188


namespace gather_all_candies_l805_805208

theorem gather_all_candies (n : ℕ) (a : Fin n → ℕ) :
  4 ≤ n →
  (∑ i, a i ≥ 4) →
  ∃ k, ∀ i, k ≠ i → a i = 0 :=
by
  sorry

end gather_all_candies_l805_805208


namespace integer_count_satisfies_inequality_l805_805291

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805291


namespace cost_large_bulb_l805_805049

def small_bulbs : Nat := 3
def cost_small_bulb : Nat := 8
def total_budget : Nat := 60
def amount_left : Nat := 24

theorem cost_large_bulb (cost_large_bulb : Nat) :
  total_budget - amount_left - small_bulbs * cost_small_bulb = cost_large_bulb →
  cost_large_bulb = 12 := by
  sorry

end cost_large_bulb_l805_805049


namespace find_kg_of_mangoes_l805_805244

-- Define the conditions
def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 965
def cost_of_mangoes (m : ℕ) : ℕ := 45 * m

-- Formalize the proof problem
theorem find_kg_of_mangoes (m : ℕ) :
  cost_of_grapes + cost_of_mangoes m = total_amount_paid → m = 9 :=
by
  intros h
  sorry

end find_kg_of_mangoes_l805_805244


namespace fraction_division_l805_805485

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805485


namespace coprime_exists_l805_805753

theorem coprime_exists (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
    ∃ d : ℕ, (1 < d ∧ d < 100) ∧ (Nat.coprime d a ∧ Nat.coprime d b ∧ Nat.coprime d c) :=
sorry

end coprime_exists_l805_805753


namespace john_total_spent_l805_805363

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l805_805363


namespace decimal_base7_equivalence_l805_805605

theorem decimal_base7_equivalence (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 343 * d + 49 * c + 7 * b + a) : 
  1000 * a + 100 * b + 10 * c + d = 2116 :=
begin
  sorry
end

end decimal_base7_equivalence_l805_805605


namespace emily_lucas_difference_l805_805927

/-- 
Condition: Emily computes correctly.
  E is defined as:
  E = 12 - (3 + 4 * 2)
Condition: Lucas computes incorrectly.
  L is defined as:
  L = 12 - 3 + 4 - 2
Proof: We want to show E - L = -10.
-/
theorem emily_lucas_difference :
  let E := 12 - (3 + 4 * 2)
  let L := 12 - 3 + 4 - 2
  E - L = -10 :=
by
  let E := 12 - (3 + 4 * 2)
  let L := 12 - 3 + 4 - 2
  have h : E - L = -10 := sorry
  exact h

end emily_lucas_difference_l805_805927


namespace dinner_seating_l805_805926

-- Let's define our conditions and problem statement in Lean 4.
theorem dinner_seating : 
  let n := 8 in  -- number of people
  let k := 7 in  -- number of seats
  (nat.choose n k * (nat.factorial k / k)) = 5760 :=
by
  -- sorry is used here as we are not required to provide the proof
  sorry

end dinner_seating_l805_805926


namespace solution_exists_l805_805186

theorem solution_exists (x : ℝ) :
  (|x - 10| + |x - 14| = |2 * x - 24|) ↔ (x = 12) :=
by
  sorry

end solution_exists_l805_805186


namespace Zoe_siblings_l805_805424

structure Child where
  eyeColor : String
  hairColor : String
  height : String

def Emma : Child := { eyeColor := "Green", hairColor := "Red", height := "Tall" }
def Zoe : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Short" }
def Liam : Child := { eyeColor := "Green", hairColor := "Brown", height := "Short" }
def Noah : Child := { eyeColor := "Gray", hairColor := "Red", height := "Tall" }
def Mia : Child := { eyeColor := "Green", hairColor := "Red", height := "Short" }
def Lucas : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Tall" }

def sibling (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

theorem Zoe_siblings : sibling Zoe Noah ∧ sibling Zoe Lucas ∧ ∃ x, sibling Noah x ∧ sibling Lucas x :=
by
  sorry

end Zoe_siblings_l805_805424


namespace pair_a_correct_pair_b_correct_pair_c_correct_pair_d_correct_l805_805845

section Pairs

variable (C x : Real)

noncomputable def pair_a := ∀ (x : Real), linear_independent ℝ (λ i, [x, x^2])
noncomputable def pair_b := ∀ (x : Real), linear_independent ℝ (λ i, [1, x])
noncomputable def pair_c := ∀ (x : Real), linear_dependent ℝ (λ i, [x, 2x])
noncomputable def pair_d := ∀ (x : Real), linear_dependent ℝ (λ i, [cos x, C * cos x])

#check pair_a
#check pair_b
#check pair_c
#check pair_d

end Pairs

-- Proofs placeholder
theorem pair_a_correct : pair_a := by 
  sorry

theorem pair_b_correct : pair_b := by 
  sorry

theorem pair_c_correct : pair_c := by 
  sorry

theorem pair_d_correct : pair_d := by 
  sorry

end pair_a_correct_pair_b_correct_pair_c_correct_pair_d_correct_l805_805845


namespace average_weight_with_D_l805_805003

open Real

theorem average_weight_with_D (A B C D E : ℝ) (hA : A = 95) 
  (h_avg_ABC : (A + B + C) / 3 = 80) 
  (h_avg_BCDE : (B + C + D + E) / 4 = 81)
  (hE : E = D + 3) 
  : ((A + B + C + D) / 4) = 82 := 
by
  have h_sum_ABC : A + B + C = 240 := by
    rw [hA]
    linarith
  have h_sum_BC : B + C = 145 := by
    rw [←h_sum_ABC, hA]
    linarith
  have h_avg_solution : 148 + 2 * D = 324 := by
    rw [h_sum_BC, h_avg_BCDE, hE]
    linarith
  have hD : D = 88 := by
    linarith
  nth_rewrite 0 h_sum_ABC
  rw [hD]
  linarith

end average_weight_with_D_l805_805003


namespace num_integer_solutions_l805_805282

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805282


namespace find_an_l805_805380

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l805_805380


namespace xiaoMing_better_performance_l805_805089

-- Definitions based on conditions
def xiaoMing_scores : List Float := [90, 67, 90, 92, 96]
def xiaoLiang_scores : List Float := [87, 62, 90, 92, 92]

-- Definitions of average and variance calculation
def average (scores : List Float) : Float :=
  (scores.sum) / (scores.length.toFloat)

def variance (scores : List Float) : Float :=
  let avg := average scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length.toFloat)

-- Prove that Xiao Ming's performance is better than Xiao Liang's.
theorem xiaoMing_better_performance :
  average xiaoMing_scores > average xiaoLiang_scores ∧ variance xiaoMing_scores < variance xiaoLiang_scores :=
by
  sorry

end xiaoMing_better_performance_l805_805089


namespace green_space_after_three_years_l805_805525

theorem green_space_after_three_years 
  (initial_green_space : ℕ) (k : ℝ) 
  (annual_increase : k = 0.1) :
  initial_green_space = 1000 → 
  let year1 := initial_green_space * (1 + k) in
  let year2 := year1 * (1 + k) in
  let year3 := year2 * (1 + k) in
  year3 = 1331 := 
begin
  sorry
end

end green_space_after_three_years_l805_805525


namespace average_speed_is_36_l805_805464

-- Define the scenario setup
def distance_BC (d : ℝ) := d
def distance_AB (d : ℝ) := 2 * d
def speed_AB : ℝ := 60
def speed_BC : ℝ := 20

-- Define time calculations
def time_AB (d : ℝ) := distance_AB d / speed_AB
def time_BC (d : ℝ) := distance_BC d / speed_BC

-- Define total distance and time
def total_distance (d : ℝ) := distance_AB d + distance_BC d
def total_time (d : ℝ) := time_AB d + time_BC d

-- Define average speed calculation
def average_speed (d : ℝ) := total_distance d / total_time d

-- The proof problem statement
theorem average_speed_is_36 (d : ℝ) (h : d > 0) : average_speed d = 36 :=
by
  -- The actual proof would go here.
  sorry

end average_speed_is_36_l805_805464


namespace count_int_values_cube_bound_l805_805311

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805311


namespace num_int_values_satisfying_ineq_l805_805260

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805260


namespace cookies_per_student_l805_805621

theorem cookies_per_student (students : ℕ) (percent : ℝ) (oatmeal_cookies : ℕ) 
                            (h_students : students = 40)
                            (h_percent : percent = 10 / 100)
                            (h_oatmeal : oatmeal_cookies = 8) :
                            (oatmeal_cookies / percent / students) = 2 := by
  sorry

end cookies_per_student_l805_805621


namespace cost_of_3000_pencils_l805_805109

-- Define the given conditions
def cost_per_box : ℝ := 40
def pencils_per_box : ℕ := 200
def discount_threshold : ℕ := 1000
def discount_rate : ℝ := 0.1
def pencils_ordered : ℕ := 3000

-- Define the functions to calculate total cost
def cost_per_pencil : ℝ := cost_per_box / pencils_per_box
def total_cost_without_discount : ℝ := pencils_ordered * cost_per_pencil
def discount : ℝ := if pencils_ordered > discount_threshold then total_cost_without_discount * discount_rate else 0
def total_cost_with_discount : ℝ := total_cost_without_discount - discount

-- The theorem to prove
theorem cost_of_3000_pencils : total_cost_with_discount = 540 :=
by
  -- Skip the proof details
  sorry

end cost_of_3000_pencils_l805_805109


namespace conditional_probability_l805_805701

def slips : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def P_A : ℚ := 5/9

def P_A_and_B : ℚ := 5/9 * 4/8

theorem conditional_probability :
  (5 / 18) / (5 / 9) = 1 / 2 :=
by
  sorry

end conditional_probability_l805_805701


namespace segments_intersect_l805_805737

-- Define points in a 3D space
structure Point3D := (x y z : ℝ)

-- Define segments in terms of their endpoints
structure Segment3D := (start end_ : Point3D)

-- Conditions
variables (A1 A2 A3 A4 : Point3D)
variables (A12 : Segment3D)
variables (A23 : Segment3D)
variables (A34 : Segment3D)
variables (A41 : Segment3D)

-- Assuming the Ceva's condition holds
axiom CevasConditionHolds :
  (dist A1 A12.start) / (dist A12.start A2) *
  (dist A2 A23.start) / (dist A23.start A3) *
  (dist A3 A34.start) / (dist A34.start A4) *
  (dist A4 A41.start) / (dist A41.start A1) = 1

-- To Prove: the segments intersect at a point
theorem segments_intersect :
  ∃ P : Point3D, ∃ (s1 s2 : ℝ), 
    (P = A12.start + s1 * (A34.start - A12.start)) ∧
    (P = A23.start + s2 * (A41.start - A23.start)) :=
sorry

end segments_intersect_l805_805737


namespace g_recursion_relation_l805_805394

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2)^n

theorem g_recursion_relation (n : ℕ) : g (n + 1) - 2 * g n + g (n - 1) = 0 :=
  sorry

end g_recursion_relation_l805_805394


namespace cost_of_one_photocopy_is_0_02_l805_805780

-- Definitions and conditions
def photocopy_cost : ℝ := sorry  -- The cost of one photocopy is a certain amount

def discount_rate := 0.25
def copies_per_individual := 80
def total_copies := 160
def individual_savings := 0.40  -- Savings per individual

-- The problem statement: prove that the cost of one photocopy is $0.02
theorem cost_of_one_photocopy_is_0_02 :
  (photocopy_cost * (1 : ℝ) * total_copies - photocopy_cost * (1 - discount_rate) * total_copies = 2 * individual_savings) →
  photocopy_cost = 0.02 :=
  sorry

end cost_of_one_photocopy_is_0_02_l805_805780


namespace vremyankin_arrives_first_l805_805050

variable (D T : ℝ)

-- Definitions based on given conditions
def Vremyankin_distance_first_half (T : ℝ) := 5 * (T / 2)
def Vremyankin_distance_second_half (T : ℝ) := 4 * (T / 2)
def Vremyankin_total_distance (T : ℝ) := Vremyankin_distance_first_half T + Vremyankin_distance_second_half T

def Puteykin_time_first_half (D : ℝ) := (D / 2) / 4
def Puteykin_time_second_half (D : ℝ) := (D / 2) / 5
def Puteykin_total_time (D : ℝ) := Puteykin_time_first_half D + Puteykin_time_second_half D

-- Main statement comparing both times
theorem vremyankin_arrives_first (D T : ℝ) (h : Vremyankin_total_distance T = D) :
  T < Puteykin_total_time D :=
sorry

end vremyankin_arrives_first_l805_805050


namespace find_f_2017_l805_805989

theorem find_f_2017 {f : ℤ → ℤ}
  (symmetry : ∀ x : ℤ, f (-x) = -f x)
  (periodicity : ∀ x : ℤ, f (x + 4) = f x)
  (f_neg_1 : f (-1) = 2) :
  f 2017 = -2 :=
sorry

end find_f_2017_l805_805989


namespace sum_of_first_3n_terms_l805_805348

variable {S : ℕ → ℝ}
variable {n : ℕ}
variable {a b : ℝ}

def arithmetic_sum (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, S (m + 1) = S m + (d * (m + 1))

theorem sum_of_first_3n_terms (h1 : S n = a) (h2 : S (2 * n) = b) 
  (h3 : arithmetic_sum S) : S (3 * n) = 3 * b - 2 * a :=
by
  sorry

end sum_of_first_3n_terms_l805_805348


namespace total_goals_not_2020_l805_805896

theorem total_goals_not_2020 (total_goals : ℕ)
  (h : ∀ match_goals, match_goals = 3 * ⌊match_goals / 3⌋) : total_goals ≠ 2020 :=
by {
  -- Suppose the total number of goals is 2020
  intro h1,
  -- Since total_goals must be a multiple of 3
  have h2: total_goals % 3 = 0 := sorry,
  -- But 2020 % 3 != 0, which is contradiction
  have h3: 2020 % 3 ≠ 0 := by norm_num,
  contradiction,
}

end total_goals_not_2020_l805_805896


namespace geometric_sequence_first_term_l805_805646

open_locale big_operators

theorem geometric_sequence_first_term (a : ℕ → ℝ) (q : ℝ) (h_sum_four : a 0 + a 1 + a 2 + a 3 = 240) 
  (h_sum_two_four : a 1 + a 3 = 180) (h_geometric : ∀ n, a (n + 1) = a n * q):
  a 0 = 6 :=
begin
  sorry
end

end geometric_sequence_first_term_l805_805646


namespace area_of_rectangle_l805_805437

-- Define the conditions
variable {S1 S2 S3 S4 : ℝ} -- side lengths of the four squares

-- The conditions:
-- 1. Four non-overlapping squares
-- 2. The area of the shaded square is 4 square inches
def conditions (S1 S2 S3 S4 : ℝ) : Prop :=
    S1^2 = 4 -- Given that one of the squares has an area of 4 square inches

-- The proof problem:
theorem area_of_rectangle (S1 S2 S3 S4 : ℝ) (h1 : 2 * S1 = S2) (h2 : 2 * S2 = S3) (h3 : conditions S1 S2 S3 S4) : 
    S1^2 + S2^2 + S3^2 = 24 :=
by
  sorry

end area_of_rectangle_l805_805437


namespace similar_triangles_PQR_STU_l805_805029

section
variables (P Q R S T U : Type) [has_zero P] [has_add P] [has_scalar ℝ P]
variables (PQ QR PR ST TU SU : ℝ)
variables (angleP : ℝ) (angleS : ℝ)

theorem similar_triangles_PQR_STU
  (hPQ : PQ = 9)
  (hQR : QR = 21)
  (hPR : PR = 15)
  (hST : ST = 4.5)
  (hTU : TU = 10.5)
  (hSU : SU = 15)
  (hAnglePS : angleP = 120 ∧ angleS = 120)
  (SAS_similar : PQ / ST = QR / TU) :
  PQ = 9 :=
sorry

end

end similar_triangles_PQR_STU_l805_805029


namespace max_m_plus_n_l805_805664

theorem max_m_plus_n (m n : ℝ) (h : n = -m^2 - 3*m + 3) : m + n ≤ 4 :=
by {
  sorry
}

end max_m_plus_n_l805_805664


namespace volume_dripped_time_for_300_mL_l805_805913

-- Define the conditions
def drops_per_second := 2
def volume_per_drop := 0.05 -- mL

-- Define the relationship between x (minutes) and y (mL)
def volume_per_minute := volume_per_drop * drops_per_second * 60 -- mL/minute

-- State the theorem for the functional relationship
theorem volume_dripped (x : ℝ) : (volume_per_minute * x) = (6 * x) :=
  by
    sorry -- This is where the proof steps would go

-- State the theorem for the time when 300 mL of water has dripped
theorem time_for_300_mL (y : ℝ) (hy : y = 300) : ∃ x : ℝ, (volume_per_minute * x) = y ∧ x = 50 :=
  by
    sorry -- This is where the proof steps would go

end volume_dripped_time_for_300_mL_l805_805913


namespace vector_parallel_k_value_l805_805675

-- Define the vectors
def a : ℝ × ℝ := (real.sqrt 3, 1)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (k, real.sqrt 3)

-- Define the condition for parallelism
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

-- The problem statement
theorem vector_parallel_k_value :
  is_parallel (a.1, a.2 + 2 * b.2) (c 1) → 1 = 1 :=
by
  sorry

end vector_parallel_k_value_l805_805675


namespace greatest_prime_divisor_digits_sum_l805_805789

theorem greatest_prime_divisor_digits_sum (h : 8191 = 2^13 - 1) : (1 + 2 + 7) = 10 :=
by
  sorry

end greatest_prime_divisor_digits_sum_l805_805789


namespace scheduling_methods_correct_l805_805693

-- Define the parameters and conditions
def number_of_schools : ℕ := 3
def total_days : ℕ := 7
def school_a_visits_days : ℕ := 2
def other_schools_visit_days : ℕ := 1
def total_scheduling_methods : ℕ := 120

-- The theorem stating that given the conditions, the total number of different scheduling methods is 120
theorem scheduling_methods_correct :
  (∃ (A B C : Set ℕ), 
    A.card = school_a_visits_days ∧ 
    B.card = other_schools_visit_days ∧ 
    C.card = other_schools_visit_days ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
    A ∪ B ∪ C = {i | i < total_days} ∧ 
    A = {i, i + 1} ∧ -- School A visits two consecutive days
    ((B ∪ C) ⊆ {i | i < total_days} - A)) →
  total_scheduling_methods = 120 :=
sorry

end scheduling_methods_correct_l805_805693


namespace parity_of_f_l805_805388

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ :=
  x * (x - 2) * (x - 1) * x * (x + 1) * (x + 2)

theorem parity_of_f :
  is_even_function f ∧ ¬ (∃ g : ℝ → ℝ, g = f ∧ (∀ x : ℝ, g (-x) = -g x)) :=
by
  sorry

end parity_of_f_l805_805388


namespace part1_part2_l805_805633

def a (n : ℕ) : ℕ := n

def S (n : ℕ) : ℕ := (list.range n).sum a

theorem part1 (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, 2 * S n = a n * (n + 1)) :
  a n = n :=
sorry

def b (n : ℕ) : ℝ := (3 * a n - 2) / 2^n

def T (n : ℕ) : ℝ := (list.range n).sum b

theorem part2 (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, 2 * S n = a n * (n + 1)) :
  T n < 4 :=
sorry

end part1_part2_l805_805633


namespace characterize_convex_polyhedra_l805_805502

-- Define a convex polyhedron
structure ConvexPolyhedron :=
  (vertices : Finset (ℝ × ℝ × ℝ))
  (is_convex : ConvexHull (vertices : Set (ℝ × ℝ × ℝ)) = set.Univ)

def same_volume_tetrahedrons (P : ConvexPolyhedron) : Prop :=
  ∃ v : ℝ, ∀ (A B C D : (ℝ × ℝ × ℝ)), 
    A ∈ P.vertices → B ∈ P.vertices → C ∈ P.vertices → D ∈ P.vertices →
    ¬Collinear ({A, B, C} : Set (ℝ × ℝ × ℝ)) → ¬Collinear ({A, B, D} : Set (ℝ × ℝ × ℝ)) →
    ¬Collinear ({A, C, D} : Set (ℝ × ℝ × ℝ)) → ¬Collinear ({B, C, D} : Set (ℝ × ℝ × ℝ)) →
    volume_tetrahedron A B C D = v

noncomputable def convex_polyhedra_with_same_volume_tetrahedrons :
  Set ConvexPolyhedron :=
  {P : ConvexPolyhedron | same_volume_tetrahedrons P}

def triangular_pyramids : Set ConvexPolyhedron := sorry
def dual_pyramids : Set ConvexPolyhedron := sorry
def triangular_prisms : Set ConvexPolyhedron := sorry

theorem characterize_convex_polyhedra
  (P : ConvexPolyhedron) :
  same_volume_tetrahedrons P ↔ P ∈ triangular_pyramids ∨ P ∈ dual_pyramids ∨ P ∈ triangular_prisms :=
sorry

end characterize_convex_polyhedra_l805_805502


namespace median_first_twelve_integers_l805_805061

theorem median_first_twelve_integers : 
  let lst : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = (6.5 : ℤ) :=
by
  sorry

end median_first_twelve_integers_l805_805061


namespace joanna_estimate_is_larger_l805_805770

theorem joanna_estimate_is_larger 
  (u v ε₁ ε₂ : ℝ) 
  (huv : u > v) 
  (hv0 : v > 0) 
  (hε₁ : ε₁ > 0) 
  (hε₂ : ε₂ > 0) : 
  (u + ε₁) - (v - ε₂) > u - v := 
sorry

end joanna_estimate_is_larger_l805_805770


namespace solve_for_x_l805_805594

theorem solve_for_x (x : ℝ) : 16^(x + 2) = 400 + 48 * 16^x → x = 25 / 13 := 
by
  intro h
  sorry

end solve_for_x_l805_805594


namespace domain_of_log_sqrt_l805_805008

noncomputable def domain_of_function := {x : ℝ | (2 * x - 1 > 0) ∧ (2 * x - 1 ≠ 1) ∧ (3 * x - 2 > 0)}

theorem domain_of_log_sqrt : domain_of_function = {x : ℝ | (2 / 3 < x ∧ x < 1) ∨ (1 < x)} :=
by sorry

end domain_of_log_sqrt_l805_805008


namespace area_R_correct_l805_805469

noncomputable def area_of_region_R (A B C W X Y Z : Point) (WXYZ : Square W X Y Z)
  (ABC : EquilateralTriangle A B C) (Hinside : A ∈ WXYZ ∧ B ∈ XY ∧ C ∈ WZ)
  (R : Region) (Hregion : ∀ p ∈ R, p ∈ WXYZ ∧ p ∉ ABC ∧ (1/4 ≤ dist p Y ∧ dist p Y ≤ 1/2)) : ℝ :=
1/4 - sqrt 3 / 4

theorem area_R_correct (A B C W X Y Z : Point) (WXYZ : Square W X Y Z)
  (ABC : EquilateralTriangle A B C) (Hinside : A ∈ WXYZ ∧ B ∈ XY ∧ C ∈ WZ)
  (R : Region) (Hregion : ∀ p ∈ R, p ∈ WXYZ ∧ p ∉ ABC ∧ (1/4 ≤ dist p Y ∧ dist p Y ≤ 1/2)) :
  area_of_region_R A B C W X Y Z WXYZ ABC Hinside R Hregion = (1 - sqrt 3) / 4 :=
sorry

end area_R_correct_l805_805469


namespace max_planes_from_15_points_l805_805874

-- Define the statement of the problem
theorem max_planes_from_15_points
  (points : Fin n → ℝ × ℝ × ℝ)
  (h_card : n = 15)
  (h_no_four_coplanar : ∀ (p1 p2 p3 p4 : Fin n), 
        ∃ d12 d13 d14 d23 d24 d34: ℝ,
        d12 ≠ 0 ∧ d13 ≠ 0 ∧ d14 ≠ 0 ∧ d23 ≠ 0 ∧ d24 ≠ 0 ∧ d34 ≠ 0) :
  ∃ m : ℕ, m = 455 :=
by
  have H := Nat.choose 15 3
  have H_eq := by linarith
  exact ⟨H, H_eq⟩

end max_planes_from_15_points_l805_805874


namespace scientific_notation_n_is_8_l805_805554

-- The main definition where 250,000,000 in scientific notation form is 2.5 * 10 ^ n
def scientific_notation (x : ℕ) : Prop := x = 250000000

-- The main theorem stating that 250,000,000 can be expressed as 2.5 * 10 ^ 8 in scientific notation
theorem scientific_notation_n_is_8 : ∃ n : ℕ, scientific_notation (2.5 * 10 ^ n) ∧ n = 8 := 
begin 
  sorry
end

end scientific_notation_n_is_8_l805_805554


namespace midpoint_of_translated_segment_l805_805798

def translation (p : ℝ × ℝ) (dx : ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

def midpoint (p₁ p₂ : ℝ × ℝ) : ℝ × ℝ :=
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)

theorem midpoint_of_translated_segment :
  let C := (2, 2)
      H := (6, 2)
      C' := translation C 6 (-3)
      H' := translation H 6 (-3)
  in midpoint C' H' = (10, -1) :=
by {
  let C := (2, 2),
  let H := (6, 2),
  let C' := translation C 6 (-3),
  let H' := translation H 6 (-3),
  have : midpoint C' H' = (10, -1), {
    sorry,
  },
  exact this,
}

end midpoint_of_translated_segment_l805_805798


namespace hunter_movies_count_l805_805919

theorem hunter_movies_count (H : ℕ) 
  (dalton_movies : ℕ := 7)
  (alex_movies : ℕ := 15)
  (together_movies : ℕ := 2)
  (total_movies : ℕ := 30)
  (all_different_movies : dalton_movies + alex_movies - together_movies + H = total_movies) :
  H = 8 :=
by
  -- The mathematical proof will go here
  sorry

end hunter_movies_count_l805_805919


namespace bob_more_than_alice_l805_805139

-- Definitions for conditions
def initial_investment_alice : ℕ := 10000
def initial_investment_bob : ℕ := 10000
def multiple_alice : ℕ := 3
def multiple_bob : ℕ := 7

-- Derived conditions based on the investment multiples
def final_amount_alice : ℕ := initial_investment_alice * multiple_alice
def final_amount_bob : ℕ := initial_investment_bob * multiple_bob

-- Statement of the problem
theorem bob_more_than_alice : final_amount_bob - final_amount_alice = 40000 :=
by
  -- Proof to be filled in
  sorry

end bob_more_than_alice_l805_805139


namespace area_triangle_XYZ_correct_l805_805390

noncomputable def areaXYZ : ℝ :=
  let AB : ℝ := 8
  let AD : ℝ := 11
  let angBAD : ℝ := real.pi / 3  -- 60 degrees in radians
  let ratioCX_XD : ℝ := 1 / 3
  let ratioAY_YD : ℝ := 1 / 2
  -- Calculation of the area using given conditions
  -- Area of parallelogram ABD = Area of triangle ABD + Area of triangle ABD (repeated)
  -- Area of triangle ABD = 1/2 * AB * AD * sin(angBAD)
  let areaABCD := AB * AD * real.sin(angBAD)
  let ratioXYZ_ABCD := 4 / 11
  ratioXYZ_ABCD * areaABCD

-- Proof this is the area of triangle XYZ
theorem area_triangle_XYZ_correct :
  let AB : ℝ := 8
  let AD : ℝ := 11
  let angBAD : ℝ := real.pi / 3  -- 60 degrees in radians
  let ratioCX_XD : ℝ := 1 / 3
  let ratioAY_YD : ℝ := 1 / 2
  let areaABCD := AB * AD * real.sin(angBAD)
  (4 / 11) * areaABCD = 16 * real.sqrt(3) :=
by
  sorry

end area_triangle_XYZ_correct_l805_805390


namespace how_much_did_B_invest_l805_805895

variable (A : ℝ) (B : ℝ) (C : ℝ)
variable (total_profit : ℝ) (A_profit : ℝ)

def investments := A + B + C

def total_investment := 6300 + B + 10500

def A_share_ratio := A / investments
def A_profit_ratio := A_profit / total_profit

theorem how_much_did_B_invest (A : ℝ := 6300) (C : ℝ := 10500) 
  (total_profit : ℝ := 12500) (A_profit : ℝ := 3750) :
  ∀ (B : ℝ), (A / (A + B + C)) = (A_profit / total_profit) → B = 13650 :=
by
  sorry

end how_much_did_B_invest_l805_805895


namespace cistern_wet_surface_area_l805_805115

def cistern_length : ℝ := 4
def cistern_width : ℝ := 8
def water_depth : ℝ := 1.25

def area_bottom (l w : ℝ) : ℝ := l * w
def area_pair1 (l h : ℝ) : ℝ := 2 * (l * h)
def area_pair2 (w h : ℝ) : ℝ := 2 * (w * h)
def total_wet_surface_area (l w h : ℝ) : ℝ := area_bottom l w + area_pair1 l h + area_pair2 w h

theorem cistern_wet_surface_area : total_wet_surface_area cistern_length cistern_width water_depth = 62 := 
by 
  sorry

end cistern_wet_surface_area_l805_805115


namespace ratio_nephews_l805_805138

variable (N : ℕ) -- The number of nephews Alden has now.
variable (Alden_had_50 : Prop := 50 = 50)
variable (Vihaan_more_60 : Prop := Vihaan = N + 60)
variable (Together_260 : Prop := N + (N + 60) = 260)

theorem ratio_nephews (N : ℕ) 
  (H1 : Alden_had_50)
  (H2 : Vihaan_more_60)
  (H3 : Together_260) :
  50 / N = 1 / 2 :=
by
  sorry

end ratio_nephews_l805_805138


namespace covering_radius_of_regular_ngon_l805_805630

-- Definitions of the conditions
variables (n : ℕ) (a : ℝ)
-- Non-negative side lengths and n-gon should have at least 3 sides
-- Note that in practice, we'd use these conditions to restrict; however, Lean doesn't require such domain considerations in the proposition
-- Hence we focus on the mathematical content for the equivalence.

-- The proof problem statement
theorem covering_radius_of_regular_ngon (h : n ≥ 3) :
  ∃ r : ℝ, r = a / (2 * sin (Real.pi / n)) :=
sorry

end covering_radius_of_regular_ngon_l805_805630


namespace integer_count_satisfies_inequality_l805_805297

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805297


namespace tangent_lines_through_point_l805_805608

theorem tangent_lines_through_point :
  ∃ k : ℚ, ((5  * k - 12 * (36 - k * 2) + 36 = 0) ∨ (2 = 0)) := sorry

end tangent_lines_through_point_l805_805608


namespace find_number_l805_805806

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l805_805806


namespace find_principal_l805_805842

-- Given definitions
def r : ℝ := 0.05
def t : ℕ := 6
def A : ℝ := 1120
def n : ℕ := 1
def compounding_factor : ℝ := (1 + r / n) ^ (n * t)

-- Statement to prove
theorem find_principal : 
  ∃ P : ℝ, A = P * compounding_factor ∧ P = 835.82 :=
by
  sorry

end find_principal_l805_805842


namespace count_integers_in_range_num_of_integers_l805_805264

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805264


namespace mn_is_one_third_l805_805436

variable (m n θ₁ θ₂ : Real)
variable (hθ : θ₁ = 3 * θ₂)
variable (slope_L1 : Real := 3 * m)
variable (slope_L2 : Real := n)
variable (tan_eq_3m : slope_L1 = tan θ₁)
variable (tan_eq_n : slope_L2 = tan θ₂)
variable (L1_not_vertical : slope_L1 ≠ ∞)

theorem mn_is_one_third : m * n = 1 / 3 := by
  sorry

end mn_is_one_third_l805_805436


namespace binary_addition_l805_805471

theorem binary_addition :
  let x := (1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) in
  let y := (1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0) in
  x + y = 141 :=
by
  sorry

end binary_addition_l805_805471


namespace reservoir_water_l805_805907

-- Conditions definitions
def total_capacity (C : ℝ) : Prop :=
  ∃ (x : ℝ), x = C

def normal_level (C : ℝ) : ℝ :=
  C - 20

def water_end_of_month (C : ℝ) : ℝ :=
  0.75 * C

def condition_equation (C : ℝ) : Prop :=
  water_end_of_month C = 2 * normal_level C

-- The theorem proving the amount of water at the end of the month is 24 million gallons given the conditions
theorem reservoir_water (C : ℝ) (hC : total_capacity C) (h_condition : condition_equation C) : water_end_of_month C = 24 :=
by
  sorry

end reservoir_water_l805_805907


namespace number_of_integers_satisfying_cubed_inequality_l805_805303

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805303


namespace division_of_fractions_l805_805488

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805488


namespace arithmetic_sequence_l805_805378

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l805_805378


namespace count_int_values_cube_bound_l805_805316

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805316


namespace negation_of_p_l805_805396

def proposition_p (n : ℕ) : Prop := 3^n ≥ n + 1

theorem negation_of_p : (∃ n0 : ℕ, 3^n0 < n0^2 + 1) :=
  by sorry

end negation_of_p_l805_805396


namespace simplified_value_of_exponent_product_l805_805829

theorem simplified_value_of_exponent_product :
  (2^0.5) * (2^{-0.3}) * (2^1.5) * (2^{-0.7}) * (2^0.9) = 2^1.9 :=
by sorry

end simplified_value_of_exponent_product_l805_805829


namespace inclination_angle_eq_135_l805_805454

theorem inclination_angle_eq_135 (h : ∀ x y : ℝ, x + y = 0 → (y = -x)) : 
  (∃ α : ℝ, α = 135 ∧ tan (α * π / 180) = -1) :=
sorry

end inclination_angle_eq_135_l805_805454


namespace max_min_P_l805_805374

theorem max_min_P (a b c : ℝ) (h : |a + b| + |b + c| + |c + a| = 8) :
  (a^2 + b^2 + c^2 = 48) ∨ (a^2 + b^2 + c^2 = 16 / 3) :=
sorry

end max_min_P_l805_805374


namespace sum_of_odd_binomial_coeffs_l805_805988

theorem sum_of_odd_binomial_coeffs (n : ℕ) (h : nat.choose n 3 = nat.choose n 7) : 
  (∑ k in finset.range (n + 1), if k % 2 = 1 then nat.choose n k else 0) = 2 ^ (n - 1) :=
by
  sorry

end sum_of_odd_binomial_coeffs_l805_805988


namespace division_of_fractions_l805_805481

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805481


namespace tan_A_l805_805351

def triangle (A B C : Type) := sorry

variables {A B C : Type} [triangle A B C]

variables (BAC : angle) (AB BC AC : ℝ)

axiom BAC_right : BAC = 90
axiom AB_length : AB = 15
axiom BC_length : BC = 17
axiom AC_length : AC = Real.sqrt (BC^2 - AB^2)

theorem tan_A : (AC / AB) = (8 / 15) := by
  sorry

end tan_A_l805_805351


namespace count_integers_in_range_num_of_integers_l805_805272

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805272


namespace total_pics_uploaded_l805_805751

-- Definitions of conditions
def pic_in_first_album : Nat := 14
def albums_with_7_pics : Nat := 3
def pics_per_album : Nat := 7

-- Theorem statement
theorem total_pics_uploaded :
  pic_in_first_album + albums_with_7_pics * pics_per_album = 35 := by
  sorry

end total_pics_uploaded_l805_805751


namespace value_of_a_l805_805330

theorem value_of_a (a : ℝ) (h : (2 : ℝ)^a = (1 / 2 : ℝ)) : a = -1 := 
sorry

end value_of_a_l805_805330


namespace markup_is_correct_l805_805800

def purchase_price : ℝ := 48
def overhead_percent : ℝ := 0.25
def net_profit : ℝ := 12

def overhead_cost := overhead_percent * purchase_price
def total_cost := purchase_price + overhead_cost
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_is_correct : markup = 24 := by sorry

end markup_is_correct_l805_805800


namespace hula_hoop_radius_l805_805837

theorem hula_hoop_radius (d : ℝ) (hd : d = 14) : d / 2 = 7 :=
by
  rw [hd]
  norm_num

end hula_hoop_radius_l805_805837


namespace vasya_guaranteed_win_l805_805461

def petya_moves := {1, 3, 4}
def vasya_moves := {1, 2, 3}
def initial_matches := 100

theorem vasya_guaranteed_win (N : ℕ) (hN : N = initial_matches) :
  (∃ (move : ℕ), (move ∈ vasya_moves) ∧ ((N - move) % 3 = 2)) :=
sorry

end vasya_guaranteed_win_l805_805461


namespace barbara_wins_l805_805898

theorem barbara_wins (n : ℕ) (h : n = 15) (num_winning_sequences : ℕ) :
  num_winning_sequences = 8320 :=
sorry

end barbara_wins_l805_805898


namespace square_area_l805_805886

theorem square_area :
  ∀ (x1 x2 : ℝ), (x1^2 + 2 * x1 + 1 = 8) ∧ (x2^2 + 2 * x2 + 1 = 8) ∧ (x1 ≠ x2) →
  (abs (x1 - x2))^2 = 36 :=
by
  sorry

end square_area_l805_805886


namespace correct_proposition_among_choices_l805_805006

theorem correct_proposition_among_choices :
  (∀ (P Q : Prop), 
   ((analogical_reasoning P Q ↔ P → Q) ∧        -- definition 1
    (deductive_reasoning P Q ↔ Q → P) ∧         -- definition 2
    (abductive_reasoning P ↔ ¬ P) ∧             -- definition 3
    (deductive_conclusion_correct P Q ↔ (P ∧ (Q → P))))) -- definition 4 
   → ((analogical_reasoning general specific → False) ∧
      (deductive_conclusion_correct general specific → True) ∧
      (abductive_reasoning specific → False) ∧
      (deductive_conclusion_correct (general ∧ specific) specific → True))
:=
sorry

end correct_proposition_among_choices_l805_805006


namespace tangent_circle_radius_is_4_l805_805204

noncomputable def radius_of_tangent_circle (r : ℝ) : Prop :=
let Q := (0, 0) in
let PQ := 2 in
let QR := 2 in
let PR := 2 * Real.sqrt 2 in
let S := (r, r) in
-- Hypotenuse can be expressed using geometric properties
(2 - r) ^ 2 + (2 - r) ^ 2 = PR ^ 2 ∧ -- distance from circle center to hypotenuse
r > 0 -- radius must be positive

theorem tangent_circle_radius_is_4 (r : ℝ) : radius_of_tangent_circle r → r = 4 :=
by
  sorry

end tangent_circle_radius_is_4_l805_805204


namespace black_tshirt_cost_l805_805773

/-- 
Given the conditions:
- Total t-shirts sold = 200
- Time taken to sell = 25 minutes
- Total revenue per minute = $220
- Half of the t-shirts were black
- The other half were white and cost $25 each 
Prove the cost of each black t-shirt is $30.
-/
theorem black_tshirt_cost (total_tshirts : ℕ) (time_minutes : ℕ) (revenue_per_minute : ℕ) (white_cost : ℕ) (half_total_tshirts : total_tshirts / 2 = 100) :
  let total_revenue := time_minutes * revenue_per_minute,
      black_revenue := total_revenue - (half_total_tshirts * white_cost),
      black_cost := black_revenue / half_total_tshirts
  in
  total_tshirts = 200 →
  time_minutes = 25 →
  revenue_per_minute = 220 →
  white_cost = 25 →
  black_cost = 30 :=
by
  intros _ _ _ _ _ _ _ _ _; -- stating that we are assuming these hypothesis
  sorry -- the actual proof would go here

end black_tshirt_cost_l805_805773


namespace triangle_ratio_l805_805508

theorem triangle_ratio
  (K : ℝ) -- Area of triangle ABC
  (CD AE BF : ℝ) -- Lengths of CD, AE and BF
  (hn1 : CD = AE = BF = K / 4) -- Each of them is one fourth of their respective sides
  (h_ratio : ∀ (N_1 N_2 N_3 : Point),
     LineCD N_1 N_2 N_3 →
     Area N_1 N_2 N_3 = K / 9)
: Area ΔN_1 ΔN_2 ΔN_3 = K / 9 :=
sorry

end triangle_ratio_l805_805508


namespace hemisphere_surface_area_cylinder_surface_area_l805_805032

-- Definitions for conditions
def radius (r : ℝ) : Prop := r = 8

def sphere_surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

-- Theorem that the total surface area of the hemisphere is 192π
theorem hemisphere_surface_area : ∀ (r : ℝ), radius r → 
  let base_area := real.pi * r^2 in
  let hemisphere_area := (sphere_surface_area r) / 2 in
  base_area + hemisphere_area = 192 * real.pi := by
  intros r hr
  unfold radius at hr
  rw hr
  unfold sphere_surface_area
  unfold real.pi
  sorry

-- Theorem that the total surface area of the cylinder is 256π
theorem cylinder_surface_area : ∀ (r : ℝ), radius r → 
  let h := r in
  let lateral_area := 2 * real.pi * r * h in
  let top_bottom_area := 2 * real.pi * r^2 in
  lateral_area + top_bottom_area = 256 * real.pi := by
  intros r hr
  unfold radius at hr
  rw hr
  unfold real.pi
  sorry

end hemisphere_surface_area_cylinder_surface_area_l805_805032


namespace find_k_l805_805674

theorem find_k (k : ℝ) :
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, 7)
  ((a.1 - c.1) * b.2 - (a.2 - c.2) * b.1 = 0) → k = 5 := 
by
  sorry

end find_k_l805_805674


namespace tetrahedron_volume_ratio_l805_805127

theorem tetrahedron_volume_ratio
  (a b : ℝ)
  (larger_tetrahedron : a = 6)
  (smaller_tetrahedron : b = a / 2) :
  (b^3 / a^3) = 1 / 8 := 
by 
  sorry

end tetrahedron_volume_ratio_l805_805127


namespace max_min_values_of_y_l805_805198

noncomputable def y (x : ℝ) : ℝ := (x - 2) * abs x

theorem max_min_values_of_y (a : ℝ) (h : a ≤ 2) :
  ∀ x ∈ set.Icc a 2, 
  ∃ (max min : ℝ), 
    max = 0 ∧ 
    (min = if 1 ≤ a ∧ a ≤ 2 then a^2 - 2 * a 
           else if 1 - real.sqrt 2 ≤ a ∧ a < 1 then -1 
           else -a^2 + 2 * a) :=
sorry

end max_min_values_of_y_l805_805198


namespace count_integer_values_l805_805250

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805250


namespace simplify_log_expression_l805_805419

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l805_805419


namespace minimum_square_side_length_l805_805546

theorem minimum_square_side_length (width height : ℕ) (h_w : width = 5) (h_h : height = 7) :
  ∃ (side_length : ℕ), side_length = 35 := by
  use 35
  sorry

end minimum_square_side_length_l805_805546


namespace most_economical_is_small_l805_805888

noncomputable def most_economical_size (c_S q_S c_M q_M c_L q_L : ℝ) :=
  c_M = 1.3 * c_S ∧
  q_M = 0.85 * q_L ∧
  q_L = 1.5 * q_S ∧
  c_L = 1.4 * c_M →
  (c_S / q_S < c_M / q_M) ∧ (c_S / q_S < c_L / q_L)

theorem most_economical_is_small (c_S q_S c_M q_M c_L q_L : ℝ) :
  most_economical_size c_S q_S c_M q_M c_L q_L := by 
  sorry

end most_economical_is_small_l805_805888


namespace find_A_l805_805942

variables {R : Type*} [Field R] [VectorSpace R (R × R)]

theorem find_A (A : R × R → R × R) (h : ∀ u : R × R, A u = (3 : R) • u) :
  A = fun u => (3 : R) • u :=
by
  sorry

end find_A_l805_805942


namespace green_pairs_count_l805_805344

variable (blueShirtedStudents : Nat)
variable (yellowShirtedStudents : Nat)
variable (greenShirtedStudents : Nat)
variable (totalStudents : Nat)
variable (totalPairs : Nat)
variable (blueBluePairs : Nat)

def green_green_pairs (blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs : Nat) : Nat := 
  greenShirtedStudents / 2

theorem green_pairs_count
  (h1 : blueShirtedStudents = 70)
  (h2 : yellowShirtedStudents = 80)
  (h3 : greenShirtedStudents = 50)
  (h4 : totalStudents = 200)
  (h5 : totalPairs = 100)
  (h6 : blueBluePairs = 30) : 
  green_green_pairs blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs = 25 := by
  sorry

end green_pairs_count_l805_805344


namespace multiplication_modulo_l805_805426

theorem multiplication_modulo :
  ∃ n : ℕ, (253 * 649 ≡ n [MOD 100]) ∧ (0 ≤ n) ∧ (n < 100) ∧ (n = 97) := 
by
  sorry

end multiplication_modulo_l805_805426


namespace find_k_l805_805514

def equation (k : ℝ) (x : ℝ) : Prop := 2 * x^2 + 3 * x - k = 0

theorem find_k (k : ℝ) (h : equation k 7) : k = 119 :=
by
  sorry

end find_k_l805_805514


namespace vector_norm_range_l805_805624

variable {V : Type*} [normed_add_comm_group V]

theorem vector_norm_range (A B C : V) (h₁ : ∥A - B∥ = 3) (h₂ : ∥A - C∥ = 6) : 
  3 ≤ ∥B - C∥ ∧ ∥B - C∥ ≤ 9 := 
sorry

end vector_norm_range_l805_805624


namespace number_of_subsets_congruent_2006_mod_2048_l805_805948

theorem number_of_subsets_congruent_2006_mod_2048 : 
  (∃ B : set ℕ, ∀ b ∈ B, b ∈ {1, 2, ..., 2005} ∧ (∑ b in B, b) % 2048 = 2006) → 
  (2^{1994}) :=
sorry

end number_of_subsets_congruent_2006_mod_2048_l805_805948


namespace part1_part2_part3_l805_805673

/- definition of vectors a and b -/
def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-1, k)

/- condition to prove k when a and b are orthogonal -/
theorem part1 (h : (2 * -1) + (1 * k) = 0) : k = 2 :=
sorry

/- condition to find dot product when a and b are parallel -/
theorem part2 (h : ∃ c : ℝ, b (-c * 2) = (2 * c, 1 * c)) : 
  (2 * -1) + (1 * (-1 / 2)) = -1 * (1 / 2) :=
sorry

/- condition to find k when angle between a and b is 135 degrees -/
theorem part3 {k : ℝ} 
  (h : (2 * -1 + 1 * k) / (Real.sqrt (4 + 1) * Real.sqrt (1 + k * k)) = -Real.sqrt(2) / 2) :
  k = -3 ∨ k = 1 / 3 :=
sorry

end part1_part2_part3_l805_805673


namespace coordinates_of_c_cosine_of_angle_l805_805668

variables (a b c : ℝ × ℝ)
variables (λ : ℝ)

def vector_a : (ℝ × ℝ) := (1, 2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Condition 1: |c| = 3√5 and a ∥ c
axiom magnitude_c : magnitude c = 3 * Real.sqrt 5
axiom parallel_ac : c = λ • vector_a

-- Condition 2: |b| = 3√5 and (4a - b) ⊥ (2a + b)
axiom magnitude_b : magnitude b = 3 * Real.sqrt 5
axiom perpendicular_condition : dot_product (4 • vector_a - b) (2 • vector_a + b) = 0

-- Question 1: Find the coordinates of c
theorem coordinates_of_c : c = (3, 6) ∨ c = (-3, -6) := sorry

-- Question 2: Find the cosine of the angle θ between a and b
theorem cosine_of_angle : dot_product vector_a b / (magnitude vector_a * magnitude b) = 1 / 6 := sorry

end coordinates_of_c_cosine_of_angle_l805_805668


namespace exists_interval_with_2012_crazy_numbers_l805_805822

def is_crazy (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 1 ∧ n = a^b + b

def count_crazy_in_interval (n : ℕ) : ℕ :=
  (List.range' n 2014).countp is_crazy

theorem exists_interval_with_2012_crazy_numbers :
  ∃ n : ℕ, count_crazy_in_interval n = 2012 :=
sorry

end exists_interval_with_2012_crazy_numbers_l805_805822


namespace eta_zero_ae_l805_805733

noncomputable section

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define the random variables ξ and η
variable (ξ η : Ω → ℝ)

-- Define independence of ξ and η
variable (h_ind : Independency ξ η)

-- Define the condition that the distribution of ξ + η is the same as the distribution of ξ
variable (h_dist : Distribution (ξ + η) = Distribution ξ)

-- Goal: Prove that η = 0 almost surely
theorem eta_zero_ae : η =ᵐ[ProbabilitySpace] (λ _ : Ω, 0) :=
sorry

end eta_zero_ae_l805_805733


namespace quadrilateral_is_rectangle_l805_805143

theorem quadrilateral_is_rectangle
  (A B C D : Point)
  (AB CD AC BD : ℝ)
  (h1 : AB ∥ CD)
  (h2 : AB = CD)
  (h3 : AC = BD) :
  is_rectangle A B C D :=
sorry

end quadrilateral_is_rectangle_l805_805143


namespace division_of_fractions_l805_805490

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805490


namespace volume_of_rotation_l805_805575

noncomputable def volume_of_solid_of_revolution : ℝ := 
  π * ∫ y in 0..4, (4 - y)

theorem volume_of_rotation (y : ℝ) :
  (∀ x : ℝ, y = x^2 ∨ x = 2 ∨ y = 0) → volume_of_solid_of_revolution = 8 * π := by
  sorry

end volume_of_rotation_l805_805575


namespace num_integer_solutions_l805_805285

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805285


namespace net_wealth_after_transactions_l805_805402

-- Define initial values and transactions
def initial_cash_A : ℕ := 15000
def initial_cash_B : ℕ := 20000
def initial_house_value : ℕ := 15000
def first_transaction_price : ℕ := 20000
def depreciation_rate : ℝ := 0.15

-- Post-depreciation house value
def depreciated_house_value : ℝ := initial_house_value * (1 - depreciation_rate)

-- Final amounts after transactions
def final_cash_A : ℝ := (initial_cash_A + first_transaction_price) - depreciated_house_value
def final_cash_B : ℝ := depreciated_house_value

-- Net changes in wealth
def net_change_wealth_A : ℝ := final_cash_A + depreciated_house_value - (initial_cash_A + initial_house_value)
def net_change_wealth_B : ℝ := final_cash_B - initial_cash_B

-- Our proof goal
theorem net_wealth_after_transactions :
  net_change_wealth_A = 5000 ∧ net_change_wealth_B = -7250 :=
by
  sorry

end net_wealth_after_transactions_l805_805402


namespace find_integer_n_l805_805178

theorem find_integer_n (n : ℤ) (hn : -150 < n ∧ n < 150) : (n = 80 ∨ n = -100) ↔ (Real.tan (n * Real.pi / 180) = Real.tan (1340 * Real.pi / 180)) :=
by 
  sorry

end find_integer_n_l805_805178


namespace seconds_in_12_5_minutes_l805_805321

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l805_805321


namespace area_after_shortening_l805_805905

theorem area_after_shortening (l w new_side_length : ℕ) (initial_area new_area : ℕ) :
  l = 5 → w = 7 → new_side_length = w - 2 → new_area = 25 → l * w = initial_area →
  l * (w - 2) = new_area :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h5
  have hw : w - 2 = new_side_length := h3
  rw hw
  rw mul_sub
  rw h1
  rw h2
  exact h4
  sorry

end area_after_shortening_l805_805905


namespace domain_of_h_l805_805161

noncomputable def h (x : ℝ) : ℝ :=
  (x^3 - 9*x^2 + 23*x - 15) / (|x - 4| + |x + 2|)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| ≠ 0 :=
by
  intro x
  have h₁ : x ≠ 4 := by
    intro h
    rw h at *
    -- Simplify, etc.
    sorry
  have h₂ : x ≠ -2 := by
    intro h
    rw h at *
    -- Simplify, etc.
    sorry
  exact sorry

end domain_of_h_l805_805161


namespace median_first_twelve_integers_l805_805060

theorem median_first_twelve_integers : 
  let lst : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = (6.5 : ℤ) :=
by
  sorry

end median_first_twelve_integers_l805_805060


namespace local_minimum_f_is_1_maximum_local_minimum_g_is_1_l805_805655

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

def local_minimum_value_f := 1

theorem local_minimum_f_is_1 : 
  ∃ x0 : ℝ, x0 > 0 ∧ (∀ x > 0, f x0 ≤ f x) ∧ f x0 = local_minimum_value_f :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f x - a * (x - 1)

def maximum_value_local_minimum_g := 1

theorem maximum_local_minimum_g_is_1 :
  ∃ a x0 : ℝ, a = 0 ∧ x0 > 0 ∧ (∀ x > 0, g a x0 ≤ g a x) ∧ g a x0 = maximum_value_local_minimum_g :=
sorry

end local_minimum_f_is_1_maximum_local_minimum_g_is_1_l805_805655


namespace int_sufficient_but_not_necessary_l805_805849

theorem int_sufficient_but_not_necessary (x : ℤ) : 
  (∃ x : ℤ, 2 * x + 1 ∈ ℤ) ∧ ¬(∃ x : ℤ, 2 * x + 1 ∈ ℤ → x ∈ ℤ) :=
by
  sorry

end int_sufficient_but_not_necessary_l805_805849


namespace integer_count_satisfies_inequality_l805_805293

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805293


namespace find_e_l805_805156

-- Definitions of the problem conditions
def Q (x : ℝ) (f d e : ℝ) := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) :
  (∀ x : ℝ, Q x f d e = 3 * x^3 + d * x^2 + e * x + f) →
  (f = 9) →
  ((∃ p q r : ℝ, p + q + r = - d / 3 ∧ p * q * r = - f / 3
    ∧ 1 / (p + q + r) = -3
    ∧ 3 + d + e + f = p * q * r) →
    e = -16) :=
by
  intros hQ hf hroots
  sorry

end find_e_l805_805156


namespace plane_eq_l805_805177

theorem plane_eq (P0 P1 P2 : ℝ × ℝ × ℝ) (h0 : P0 = (2, -1, 2)) (h1 : P1 = (4, 3, 0)) (h2 : P2 = (5, 2, 1)) :
  ∃ a b c d : ℝ, (a, b, c) ≠ (0, 0, 0) ∧ (∀ (x y z : ℝ), a * x + b * y + c * z + d = 0 ↔ x - 2 * y - 3 * z + 2 = 0) :=
by {
  use [1, -2, -3, 2],
  split,
  { simp, },
  { intro x,
    intro y,
    intro z,
    split,
    { intro h,
      assumption, },
    { intro h,
      assumption, } },
  sorry
}

end plane_eq_l805_805177


namespace count_integer_values_l805_805249

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805249


namespace count_n_integers_l805_805274

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805274


namespace count_n_integers_l805_805273

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805273


namespace select_and_swap_ways_l805_805814

theorem select_and_swap_ways :
  let n := 8
  let k := 3
  Nat.choose n k * 2 = 112 := 
by
  let n := 8
  let k := 3
  sorry

end select_and_swap_ways_l805_805814


namespace spinner_probability_l805_805532

-- Define the game board conditions
def total_regions : ℕ := 12  -- The triangle is divided into 12 smaller regions
def shaded_regions : ℕ := 3  -- Three regions are shaded

-- Define the probability calculation
def probability (total : ℕ) (shaded : ℕ): ℚ := shaded / total

-- State the proof problem
theorem spinner_probability :
  probability total_regions shaded_regions = 1 / 4 :=
by
  sorry

end spinner_probability_l805_805532


namespace min_total_sheep_l805_805462

variables (x y z : ℕ)

theorem min_total_sheep (h1 : 3 * x - 5 * y = -8) (h2 : 5 * x - 7 * y = -24) :
  x + y + z = 19 :=
sorry

end min_total_sheep_l805_805462


namespace john_total_spent_l805_805364

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l805_805364


namespace reciprocal_of_neg_4_l805_805028

theorem reciprocal_of_neg_4.5 : 1 / (-4.5 : ℝ) = -2 / 9 := 
by
  sorry

end reciprocal_of_neg_4_l805_805028


namespace probability_not_square_or_cube_probability_fraction_l805_805022

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k * k = n

noncomputable def num_not_square_or_cube (n : ℕ) : ℕ :=
  let sq_count := (Nat.floor (Real.sqrt n)).val
  let cub_count := (Nat.floor (Real.cbrt n)).val
  let overlap_count := (Nat.floor (Real.root 6 n)).val
  n - (sq_count + cub_count - overlap_count)

theorem probability_not_square_or_cube :
  num_not_square_or_cube 200 = 182 :=
by
  sorry

theorem probability_fraction :
  let p := probability_not_square_or_cube 200
  p / 200 = 91 / 100 :=
by
  sorry

end probability_not_square_or_cube_probability_fraction_l805_805022


namespace digit_1C3_multiple_of_3_l805_805955

theorem digit_1C3_multiple_of_3 :
  (∃ C : Fin 10, (1 + C.val + 3) % 3 = 0) ∧
  (∀ C : Fin 10, (1 + C.val + 3) % 3 = 0 → (C.val = 2 ∨ C.val = 5 ∨ C.val = 8)) :=
by
  sorry

end digit_1C3_multiple_of_3_l805_805955


namespace evaluate_expression_l805_805599

theorem evaluate_expression (a : ℝ) (h : a = 3) : (3 * a ^ (-2) + a ^ (-1) / 3) / a ^ 2 = 28 / 243 := by
  sorry

end evaluate_expression_l805_805599


namespace magnitude_of_z_l805_805992

namespace ComplexNumberProof

open Complex

noncomputable def z (b : ℝ) : ℂ := (3 - b * Complex.I) / Complex.I

theorem magnitude_of_z (b : ℝ) (h : (z b).re = (z b).im) : Complex.abs (z b) = 3 * Real.sqrt 2 :=
by
  sorry

end ComplexNumberProof

end magnitude_of_z_l805_805992


namespace area_ge_11n_squared_plus_one_div_12_l805_805392

open Complex -- Opening the Complex namespace 

variables (n : ℕ) (z : ℂ) -- Defining variables n as a positive integer and z as a complex number

def S (n : ℕ) : Set ℂ :=
  {z : ℂ | ∑ k in Finset.range (n + 1), (1 / |z - k|) ≥ 1 }

theorem area_ge_11n_squared_plus_one_div_12 (n : ℕ) (h : 0 < n) : 
  let S_area := (volume (S n)).toReal in 
  S_area ≥ (Real.pi * (11 * n^2 + 1) / 12) :=
sorry

end area_ge_11n_squared_plus_one_div_12_l805_805392


namespace problem_solution_l805_805228

noncomputable def f (x : ℝ) : ℝ := log (sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem problem_solution : 
  f (log 2) + f (log (1/2)) = 2 := 
begin
  sorry,
end

end problem_solution_l805_805228


namespace middle_integer_is_zero_l805_805946

-- Mathematical equivalent proof problem in Lean 4

theorem middle_integer_is_zero
  (n : ℤ)
  (h : (n - 2) + n + (n + 2) = (1 / 5) * ((n - 2) * n * (n + 2))) :
  n = 0 :=
by
  sorry

end middle_integer_is_zero_l805_805946


namespace fraction_division_l805_805484

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805484


namespace min_value_expr_min_value_achieved_l805_805947

theorem min_value_expr : ∀ x : ℝ, 4 ≤ (x^2 + 5) / Real.sqrt (x^2 + 1) :=
by
  sorry

-- State the equality condition for the minimum value
theorem min_value_achieved : (∃ x : ℝ, (x^2 + 5) / Real.sqrt (x^2 + 1) = 4) :=
by
  use [sqrt 3]
  sorry

end min_value_expr_min_value_achieved_l805_805947


namespace median_of_first_twelve_positive_integers_l805_805057

theorem median_of_first_twelve_positive_integers :
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth (5)).getD 0 + (lst.nth (6)).getD 0 / 2 = 6.5 :=
by
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let median := ((lst.nth (5)).getD 0 + (lst.nth (6)).getD 0) / 2
  show median = 6.5
  sorry

end median_of_first_twelve_positive_integers_l805_805057


namespace number_of_integers_satisfying_cubed_inequality_l805_805305

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805305


namespace sum_of_digits_of_N_l805_805910

theorem sum_of_digits_of_N :
  let N := (16 ^ 75 * 75 ^ 16)^(1/3)
  in N.digits.sum = 7 :=
by
  sorry

end sum_of_digits_of_N_l805_805910


namespace monotonic_decreasing_interval_l805_805592

def f (x : ℝ) := Real.exp x / x^2

theorem monotonic_decreasing_interval :
  ∀ x, (0 < x ∧ x ≤ 2) → (f' x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l805_805592


namespace total_students_college_l805_805515

-- Define the conditions
def ratio_boys_girls := 8 / 5
def num_girls := 160

-- Define the proof problem statement
theorem total_students_college : 
  (∃ (num_boys : ℕ), num_boys / num_girls.to_nat = ratio_boys_girls ∧  num_boys * 5 = 8 * num_girls) → 
  (∃ (total_students : ℕ), total_students = num_girls + num_boys) :=
begin
  sorry
end

end total_students_college_l805_805515


namespace angle_A_is_60_degrees_max_value_sinB_plus_sinC_l805_805635

variable {A B C : ℝ}
variable {a b c : ℝ}

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively.
If it satisfies the condition 1 - 2 * sin(B) * sin(C) = cos(2 * B) + cos(2 * C) - cos(2 * A), 
then the magnitude of angle A is 60 degrees. --/
theorem angle_A_is_60_degrees
  (h1 : 1 - 2 * Real.sin B * Real.sin C = Real.cos (2 * B) + Real.cos (2 * C) - Real.cos (2 * A)) :
  A = Real.pi / 3 :=
sorry

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively.
The maximum value of sin(B) + sin(C) is 1. --/
theorem max_value_sinB_plus_sinC
  (h1 : 1 - 2 * Real.sin B * Real.sin C = Real.cos (2 * B) + Real.cos (2 * C) - Real.cos (2 * A))
  (hA : A = Real.pi / 3)
  (hB : B > 0) (hC : C > 0) (h_angle_sum : A + B + C = Real.pi) :
  sin B + sin C ≤ 1 :=
sorry

end angle_A_is_60_degrees_max_value_sinB_plus_sinC_l805_805635


namespace greatest_value_x_l805_805923

theorem greatest_value_x (x: ℤ) : 
  (∃ k: ℤ, (x^2 - 5 * x + 14) = k * (x - 4)) → x ≤ 14 :=
sorry

end greatest_value_x_l805_805923


namespace routes_from_A_to_C_with_D_twice_l805_805914

def City := {A B C D E : Type}

def Road (c1 c2 : City) : Prop :=
  (c1 = A ∧ c2 = B) ∨ (c1 = A ∧ c2 = D) ∨ (c1 = A ∧ c2 = E) ∨
  (c1 = B ∧ c2 = C) ∨ (c1 = B ∧ c2 = D) ∨ (c1 = C ∧ c2 = D) ∨ (c1 = D ∧ c2 = E)

noncomputable def countRoutes (start end : City) (visitDAtLeastTwice : Bool) : Nat :=
  if start = A ∧ end = C ∧ visitDAtLeastTwice = true then 6 else 0

theorem routes_from_A_to_C_with_D_twice :
  countRoutes A C true = 6 := by
  sorry

end routes_from_A_to_C_with_D_twice_l805_805914


namespace brown_eyed_brunettes_is_20_l805_805168

def total_girls : Nat := 60
def blue_eyed_blondes : Nat := 20
def brunettes : Nat := 35
def brown_eyed : Nat := 25

def brown_eyed_brunettes : Nat :=
  brown_eyed - blue_eyed_blondes - (total_girls - brunettes - blue_eyed_blondes)

theorem brown_eyed_brunettes_is_20 :
  brown_eyed_brunettes = 20 :=
by
  unfold brown_eyed_brunettes
  -- Substitute known values and simplify
  rw [Nat.sub_sub, Nat.sub_sub, Nat.sub_sub]
  sorry

end brown_eyed_brunettes_is_20_l805_805168


namespace num_integer_solutions_l805_805284

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805284


namespace magnitude_of_angle_A_height_from_BC_l805_805698

noncomputable def findAngleA (a b c : ℝ) (cosB : ℝ) (h : a * cosB - c = b / 2) : ℝ :=
  ⟨A, hA⟩ 

def findHeightFromBC (b c : ℝ) (a : ℝ) (A : ℝ) (hA : A = 2 * Real.pi / 3) (hb_c : b - c = Real.sqrt 6) (ha : a = 3 + Real.sqrt 3) : ℝ :=
  ⟨h, hh⟩

theorem magnitude_of_angle_A (a b c : ℝ) (cosB : ℝ) (h : a * cosB - c = b / 2) : 
  findAngleA a b c cosB h = 2 * Real.pi / 3 :=
sorry

theorem height_from_BC (b c : ℝ) (a : ℝ) (A : ℝ) (hA : A = 2 * Real.pi / 3) (hb_c : b - c = Real.sqrt 6) (ha : a = 3 + Real.sqrt 3) : 
  findHeightFromBC b c a A hA hb_c ha = 1 :=
sorry

end magnitude_of_angle_A_height_from_BC_l805_805698


namespace base8_representation_1024_has_4_digits_l805_805078

theorem base8_representation_1024_has_4_digits :
  (nat.log 1024 8 + 1) = 4 := by
  have h: 8^3 <= 1024 ∧ 1024 < 8^4 := by
    split
    exact pow_le_pow_of_le_left (nat.succ_pos 8) (le_of_lt (by norm_num)) (by norm_num)
    exact pow_lt_pow_of_lt (by norm_num) (by norm_num)
  sorry

end base8_representation_1024_has_4_digits_l805_805078


namespace percentage_of_x_is_2x_minus_y_l805_805337

variable (x y : ℝ)
variable (h1 : x / y = 4)
variable (h2 : y ≠ 0)

theorem percentage_of_x_is_2x_minus_y :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end percentage_of_x_is_2x_minus_y_l805_805337


namespace general_term_formula_l805_805610

-- Define the sequence terms
def sequence (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | _ => (n * (n + 1)) / 2

-- State the theorem
theorem general_term_formula (n : ℕ) : sequence n = (n * (n + 1)) / 2 := 
  sorry

end general_term_formula_l805_805610


namespace find_scalars_l805_805961

variables (λ1 λ2 : ℝ)
def e1 : ℝ × ℝ := (2, 1)
def e2 : ℝ × ℝ := (1, 3)
def a : ℝ × ℝ := (-1, 2)

theorem find_scalars
  (h : a = (λ1 * e1.1 + λ2 * e2.1, λ1 * e1.2 + λ2 * e2.2)) : 
  (λ1, λ2) = (-1, 1) :=
sorry

end find_scalars_l805_805961


namespace veranda_area_correct_l805_805438

-- Definitions of the room dimensions and veranda width
def room_length : ℝ := 18
def room_width : ℝ := 12
def veranda_width : ℝ := 2

-- Definition of the total length including veranda
def total_length : ℝ := room_length + 2 * veranda_width

-- Definition of the total width including veranda
def total_width : ℝ := room_width + 2 * veranda_width

-- Definition of the area of the entire space (room plus veranda)
def area_entire_space : ℝ := total_length * total_width

-- Definition of the area of the room
def area_room : ℝ := room_length * room_width

-- Definition of the area of the veranda
def area_veranda : ℝ := area_entire_space - area_room

-- Theorem statement to prove the area of the veranda
theorem veranda_area_correct : area_veranda = 136 := 
by
  sorry

end veranda_area_correct_l805_805438


namespace P_9_plus_P_neg5_l805_805967

open Polynomial

noncomputable def P (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

variables {a b c d : ℝ}

axiom P_1 : P 1 = 2000
axiom P_2 : P 2 = 4000
axiom P_3 : P 3 = 6000

theorem P_9_plus_P_neg5 : P 9 + P (-5) = 12704 :=
sorry

end P_9_plus_P_neg5_l805_805967


namespace pn_not_five_l805_805452

theorem pn_not_five (p : ℕ → ℕ) 
  (hp1 : p 1 = 2)
  (hpn : ∀ n ≥ 2, p (n+1) = (Prime.factor (∏ i in Finset.range n, p i.succ + 1)).max') :
  ∀ n, p n ≠ 5 :=
by
  sorry

end pn_not_five_l805_805452


namespace zeroes_of_f_l805_805742

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x > 0 then exp(-x) - 1/2 else x^3 - 3 * m * x - 2

def has_three_distinct_zeros (m : ℝ) : Prop :=
∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f m x1 = 0 ∧ f m x2 = 0 ∧ f m x3 = 0

theorem zeroes_of_f (m : ℝ) : has_three_distinct_zeros m ↔ 1 < m :=
sorry

end zeroes_of_f_l805_805742


namespace geometric_seq_ab_ge_2_l805_805637

-- Define the problem statement
theorem geometric_seq_ab_ge_2 {a b : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h_geom : ∃ r : ℝ, r ≠ 0 ∧ 4 * a^2 + b^2 = r * b ∧ b = r * a) : ab a b ≥ 2 :=
sorry

end geometric_seq_ab_ge_2_l805_805637


namespace cross_to_square_l805_805158

-- Defining the problem of transforming a cross made of 5 squares into a 2x2 square.
theorem cross_to_square (cross : fin 5 → (ℕ × ℕ))
  (square : fin 5 → (ℕ × ℕ)) (h : ∀ i : fin 5, cross i ≠ square i) :
  ∃ (cut_positions : fin 5 → (ℕ × ℕ)),
  (cut_positions 0 ≠ cut_positions 1) ∧
  (cut_positions 1 ≠ cut_positions 2) ∧
  (cut_positions 2 ≠ cut_positions 3) ∧
  (cut_positions 3 ≠ cut_positions 4) ∧
  (cut_positions 4 ≠ cut_positions 0) ∧
  (∀ j : fin 4, cut_positions j) = 
  sorry

end cross_to_square_l805_805158


namespace probability_jacob_sequence_integer_l805_805362

noncomputable def jacob_next_term (previous: ℝ) (coin: bool): ℝ :=
if previous % 3 = 0 then 1 + (if coin then 2 * previous - 1 else previous / 2 - 2)
else if coin then 2 * previous - 1 else previous / 2 - 2

noncomputable def possible_terms (a: ℝ): list ℝ := 
let second_term := [jacob_next_term a tt, jacob_next_term a ff] in
let third_term := second_term.bind (λ t, [jacob_next_term t tt, jacob_next_term t ff]) in
third_term.bind (λ t, [jacob_next_term t tt, jacob_next_term t ff])

theorem probability_jacob_sequence_integer:
  let fourth_terms := possible_terms 10 in
  let num_integer_terms := fourth_terms.count (λ x, x.floor = x) in
  let total_terms := fourth_terms.length in
  (num_integer_terms : ℝ) / total_terms = 1 / 2 :=
by
  sorry

end probability_jacob_sequence_integer_l805_805362


namespace card_M_is_15_l805_805590

def posInt : Type := {n : ℕ // n > 0}

def op_⊕ (m n : posInt) : ℕ :=
  if m.val % 2 = n.val % 2 then m.val + n.val else m.val * n.val

def satisfies_condition (a b : posInt) : Prop :=
  (op_⊕ a b) = 12

def M : set (posInt × posInt) :=
  {p | satisfies_condition p.fst p.snd}

theorem card_M_is_15 :
  set.card M = 15 :=
sorry

end card_M_is_15_l805_805590


namespace choose_good_B_l805_805538

def capital := 100000 

def prob_profit_A_20000 := 0.4
def prob_profit_A_30000 := 0.3
def prob_loss_A_10000 := 0.3

def prob_profit_B_20000 := 0.6
def prob_profit_B_40000 := 0.2
def prob_loss_B_20000 := 0.2

def expected_profit_A : ℝ := (2 * prob_profit_A_20000 + 3 * prob_profit_A_30000 + (-1) * prob_loss_A_10000)
def expected_profit_B : ℝ := (2 * prob_profit_B_20000 + 4 * prob_profit_B_40000 + (-2) * prob_loss_B_20000)

theorem choose_good_B : expected_profit_B > expected_profit_A :=
by
  -- the formal proof goes here
  sorry

end choose_good_B_l805_805538


namespace line_equation_RS_line_MN_fixed_point_l805_805663

-- Definitions for hyperbola and points
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def pointP : ℝ × ℝ := (4, 0)

-- Problem Part 1: Equation of line RS
theorem line_equation_RS :
  ∀ x : ℝ, ∀ y : ℝ,
  hyperbola x y ∧ (y = (1 / 2) * (x - 4) ∨ y = - (1 / 2) * (x - 4))
  → x = 5 / 2 := 
sorry 

-- Problem Part 2: Existence of fixed point for line MN
theorem line_MN_fixed_point :
  ∃ (fixed_point : ℝ × ℝ),
  fixed_point = (1, 0) ∧
  ∀ t1 t2 : ℝ, t1 * t2 = 12 →
  let y_D := 4 * t1 / (t1^2 + 4) in
  let x_D := 2 * t1^2 / (t1^2 + 4) - 2 in
  let y_E := 4 * t2 / (t2^2 + 4) in
  let x_E := 2 * t2^2 / (t2^2 + 4) - 2 in
  let y_M := 2 * (x_D * t1^2 / (t1^2 + 4) - x_D) in
  let y_N := 2 * (x_E * t2^2 / (t2^2 + 4) - x_E) in
  let k := (y_N - y_M) / (x_D - x_E) in
  let m := y_N - k * x_E in
  m = -k ∨ m = 2 * k →
  (k, m) = (1, 0) := 
sorry

end line_equation_RS_line_MN_fixed_point_l805_805663


namespace eccentricity_of_hyperbola_l805_805214

variable {a b : ℝ}
variable (a_pos : a > 0) (b_pos : b > 0)

def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

theorem eccentricity_of_hyperbola (c e : ℝ) :
  (∀ x y : ℝ, hyperbola x y) → e = (c / a) → 
  \(|PF1| = 2 * |QF1| \) ∧
  ∀ m : ℝ,
  |F2Q| = sqrt (68 * a^2 / 9) → 
  c^2 / a^2 = 17 / 9 →
  e = sqrt (17) / 3 :=
begin
  -- Proof omitted
  sorry
end

end eccentricity_of_hyperbola_l805_805214


namespace select_m_sets_l805_805391

variable {X : Type} [Fintype X] [Nonempty X]
variable {A : Finset (Finset X)}
variable (h : ∀ A_i ∈ A, A_i.card ≤ 3)
variable (h2 : ∀ x ∈ (Fintype.elems X : Finset X), (A.filter (λ A_i, x ∈ A_i)).card ≥ 6)

theorem select_m_sets (k m : ℕ) (hk : A.card = k) (hm : m ≥ k / 3) :
  ∃ (B : Finset (Finset X)), B.card = m ∧ (B.sup id) = (Fintype.elems X : Finset X) :=
sorry

end select_m_sets_l805_805391


namespace correct_propositions_count_l805_805225

variable (Plane Point Line : Type)
variable (passes_through : ∀ {a b : Type}, a -> b -> Prop)
variable (perpendicular : ∀ {a b : Type}, a -> b -> Prop)
variable (parallel : ∀ {a b : Type}, a -> b -> Prop)

axiom prop1 (p : Point) (P Q : Plane) : (¬ passes_through p Q) → (∃! P, passes_through p P ∧ perpendicular P Q)
axiom prop2 (p : Point) (ℓ : Line) (P : Plane) : (¬ passes_through p ℓ) → (∃! P, passes_through p P ∧ parallel P ℓ)
axiom prop3 (p : Point) (ℓ m : Line) : (¬ passes_through p ℓ) → (∃! m, passes_through p m ∧ perpendicular m ℓ)
axiom prop4 (p : Point) (P : Plane) (ℓ : Line) : (¬ passes_through p P) → (∃! ℓ, passes_through p ℓ ∧ perpendicular ℓ P)

theorem correct_propositions_count : (∀ (p : Point) (P Q : Plane), ¬ passes_through p Q → ∃! P, passes_through p P ∧ perpendicular P Q) =
false ∧ (∀ (p : Point) (ℓ : Line) (P : Plane), ¬ passes_through p ℓ → ∃! P, passes_through p P ∧ parallel P ℓ) = false ∧ 
(∀ (p : Point) (ℓ m : Line), ¬ passes_through p ℓ → ∃! m, passes_through p m ∧ perpendicular m ℓ) = false ∧ 
(∀ (p : Point) (P : Plane) (ℓ : Line), ¬ passes_through p P → ∃! ℓ, passes_through p ℓ ∧ perpendicular ℓ P) = true → count (map (λ prop, prop == true) [prop1, prop2, prop3, prop4]) = 1 := sorry

end correct_propositions_count_l805_805225


namespace quadrilateral_side_length_l805_805878

theorem quadrilateral_side_length (r a b c x : ℝ) (h_radius : r = 100 * Real.sqrt 6) 
    (h_a : a = 100) (h_b : b = 200) (h_c : c = 200) :
    x = 100 * Real.sqrt 2 := 
sorry

end quadrilateral_side_length_l805_805878


namespace area_of_square_field_l805_805719

/-- Given it takes 2 minutes to travel the diagonal of a square field at 3 km/hour,
the area of the square field is 5000 square meters. -/
theorem area_of_square_field :
  let t := 2 -- time in minutes
  let v := 3 -- speed in km/hour
  let diagonal_in_meters := v * (t / 60) * 1000 -- convert speed to meters per minute and multiply by time
  let side := diagonal_in_meters / Math.sqrt 2 -- side length of the square
  side * side = 5000 := 
by {
  let t := 2
  let v := 3
  let diagonal_in_meters := v * (t / 60) * 1000
  let side := diagonal_in_meters / Math.sqrt 2
  sorry
}


end area_of_square_field_l805_805719


namespace median_eq_6point5_l805_805063
open Nat

def median_first_twelve_positive_integers (l : List ℕ) : ℝ :=
  (l !!! 5 + l !!! 6) / 2

theorem median_eq_6point5 : median_first_twelve_positive_integers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 6.5 :=
by
  sorry

end median_eq_6point5_l805_805063


namespace books_per_shelf_correct_l805_805408

-- Definitions based on conditions
def initial_books : ℝ := 46.0
def additional_books : ℝ := 10.0
def number_of_shelves : ℝ := 14.0

-- The total number of books
def total_books : ℝ := initial_books + additional_books

-- The number of books per shelf
def books_per_shelf : ℝ := total_books / number_of_shelves

-- Main statement to be proved
theorem books_per_shelf_correct : books_per_shelf = 4.0 := by
  sorry

end books_per_shelf_correct_l805_805408


namespace quadratics_common_root_square_sum_6_l805_805984

theorem quadratics_common_root_square_sum_6
  (a b c : ℝ)
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_common_root_1: ∃ x1, x1^2 + a * x1 + b = 0 ∧ x1^2 + b * x1 + c = 0)
  (h_common_root_2: ∃ x2, x2^2 + b * x2 + c = 0 ∧ x2^2 + c * x2 + a = 0)
  (h_common_root_3: ∃ x3, x3^2 + c * x3 + a = 0 ∧ x3^2 + a * x3 + b = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratics_common_root_square_sum_6_l805_805984


namespace eden_stuffed_bears_l805_805588

theorem eden_stuffed_bears 
  (initial_bears : ℕ) 
  (percentage_kept : ℝ) 
  (sisters : ℕ) 
  (eden_initial_bears : ℕ)
  (h1 : initial_bears = 65) 
  (h2 : percentage_kept = 0.40) 
  (h3 : sisters = 4) 
  (h4 : eden_initial_bears = 20) :
  ∃ eden_bears : ℕ, eden_bears = 29 :=
by
  sorry

end eden_stuffed_bears_l805_805588


namespace alan_can_obtain_340_from_3_number_remainder_1_mod4_obtain_43_from_3_not_obtain_43_from_5_l805_805846

-- Define the allowed operations
def add_four (n : ℕ) : ℕ := n + 4
def multiply_four (n : ℕ) : ℕ := n * 4
def square (n : ℕ) : ℕ := n * n

-- Part (a) & (b): Prove Alan can obtain 340 from 3 or 5
theorem alan_can_obtain_340_from_3 : ∃ f : ℕ → ℕ, 
  f (f (f (f 3))) = 340 ∧ 
  (f (f (f (f 5))) = 340) :=
by sorry

-- Part (c): Given a number with remainder 1 modulo 4, resultant number = 0 or 1 modulo 4
theorem number_remainder_1_mod4 (x : ℕ) (h : x % 4 = 1) :
  ∀ f, f x % 4 = 0 ∨ f x % 4 = 1 :=
by {
  intros f,
  simp [add_four, multiply_four, square],
  -- Case analysis and modulo arithmetic required here
  sorry
}

-- Part (d) (1): It is possible to obtain the number 43 from the number 3
theorem obtain_43_from_3 : ∃ f : ℕ → ℕ, f (f (f (f (f (f (f (f (f (f 3))))))))) = 43 :=
by {
  -- Define the function based on the transformation
  sorry
}

-- Part (d) (2): It is impossible to obtain the number 43 starting from 5
theorem not_obtain_43_from_5 : ¬∃ f : ℕ → ℕ, f (f (f (f (f (f (f (f (f (f 5))))))))) = 43 :=
by {
  -- A contradiction based on modulo 4 conditions
  sorry
}

end alan_can_obtain_340_from_3_number_remainder_1_mod4_obtain_43_from_3_not_obtain_43_from_5_l805_805846


namespace odd_function_l805_805191

noncomputable def f (x : ℝ) : ℝ := log 2 ((1 + x) / (1 - x))

theorem odd_function (x : ℝ) (h : x > -1 ∧ x < 1) : f (-x) = -f x := 
by sorry

end odd_function_l805_805191


namespace grid_two_colors_intersection_l805_805034

theorem grid_two_colors_intersection :
  ∀ (color : ℕ → ℕ → bool), 
  (∃ h₁ h₂ v₁ v₂ : ℕ, 
    h₁ ≠ h₂ ∧ v₁ ≠ v₂ ∧ 
    color h₁ v₁ = color h₁ v₂ ∧ 
    color h₁ v₂ = color h₂ v₁ ∧ 
    color h₂ v₁ = color h₂ v₂) :=
by
sor<SIL>Ty

end grid_two_colors_intersection_l805_805034


namespace train_pass_tree_in_16_seconds_l805_805892

-- Conditions
def train_length : ℝ := 280
def speed_kmph : ℝ := 63

-- Convert speed from km/hr to m/s
def speed_mps : ℝ := speed_kmph * (1000 / 3600)

-- Question: How many seconds to pass the tree?
def time_to_pass_tree (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Theorem to prove the mathematically equivalent statement
theorem train_pass_tree_in_16_seconds : time_to_pass_tree train_length speed_mps = 16 :=
by
  -- The proof steps will be included here
  sorry

end train_pass_tree_in_16_seconds_l805_805892


namespace prob_intersection_l805_805096

noncomputable def p (event : Type) : ℝ := sorry

def event_A : Type := sorry
def event_B : Type := sorry

axiom prob_A : p event_A = 1 / 5
axiom prob_B : p event_B = 2 / 5
axiom independent_events : independent event_A event_B

theorem prob_intersection :
  p (event_A ∩ event_B) = 2 / 25 :=
by
  sorry

end prob_intersection_l805_805096


namespace num_integer_solutions_l805_805286

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805286


namespace factorization1_factorization2_l805_805603

theorem factorization1 (x y : ℝ) : 4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3 * x + 3 * y)^2 :=
by
  sorry

theorem factorization2 (x : ℝ) (a : ℝ) : 2 * a * (x^2 + 1)^2 - 8 * a * x^2 = 2 * a * (x - 1)^2 * (x + 1)^2 :=
by
  sorry

end factorization1_factorization2_l805_805603


namespace simple_interest_rate_l805_805332

theorem simple_interest_rate (P : ℝ) (increase_time : ℝ) (increase_amount : ℝ) 
(hP : P = 2000) (h_increase_time : increase_time = 4) (h_increase_amount : increase_amount = 40) :
  ∃ R : ℝ, (2000 * R / 100 * (increase_time + 4) - 2000 * R / 100 * increase_time = increase_amount) ∧ (R = 0.5) := 
by
  sorry

end simple_interest_rate_l805_805332


namespace average_value_divisible_by_3_not_4_l805_805432

def is_divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def is_not_divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b ≠ 0

def filtered_average : Nat :=
  let numbers := filter (λ x => is_divisible_by x 3 ∧ is_not_divisible_by x 4) (List.range 61) 
  (List.foldl (λ acc x => acc + x) 0 numbers) / numbers.length

theorem average_value_divisible_by_3_not_4 :
  filtered_average = 30 :=
by sorry

end average_value_divisible_by_3_not_4_l805_805432


namespace fraction_division_l805_805483

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805483


namespace common_real_solution_for_y_l805_805595

theorem common_real_solution_for_y :
  ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 - 4*y + 4 = 0 ↔ (y = -4.44 ∨ y = -8.56) :=
by
  sorry -- Proof goes here

end common_real_solution_for_y_l805_805595


namespace area_of_rectangle_l805_805710

theorem area_of_rectangle (SN PM : ℝ) (PS QR RM : ℝ) :
  SN = 4 → PM = 3 →
  PS = 4 * sqrt 3 → QR = 4 * (sqrt 6 - 2) →
  RM = PS * QR - 48 * sqrt 2 + 32 * sqrt 3 := sorry

end area_of_rectangle_l805_805710


namespace arrangements_with_conditions_correct_l805_805952

noncomputable def arrangements_with_conditions : ℕ :=
  let students := ["A", "B", "C", "D", "E"]
  let end_positions := ["B", "C", "D", "E"]
  let valid_arrangements := (permutations_with_condition students (λ arr,
    let (left, middle, right) := split_at_positions arr in
    (left != "A" and right != "A") and adjacent "C" "D" middle
  )) 
  valid_arrangements.length

theorem arrangements_with_conditions_correct : arrangements_with_conditions = 24 :=
  sorry

end arrangements_with_conditions_correct_l805_805952


namespace arithmetic_sequence_sum_l805_805708

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ)     -- arithmetic sequence
  (d : ℝ)         -- common difference
  (h: ∀ n, a (n + 1) = a n + d)     -- definition of arithmetic sequence
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := 
  sorry

end arithmetic_sequence_sum_l805_805708


namespace hyperbola_find_a_b_l805_805234

def hyperbola_conditions (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (∃ e : ℝ, e = 2) ∧ (∃ c : ℝ, c = 4)

theorem hyperbola_find_a_b (a b : ℝ) : hyperbola_conditions a b → a = 2 ∧ b = 2 * Real.sqrt 3 := 
sorry

end hyperbola_find_a_b_l805_805234


namespace bridge_length_l805_805893

-- Definitions of conditions as given.
def train_length : ℝ := 800
def time_to_pass_tree : ℝ := 120
def time_to_pass_bridge : ℝ := 300

-- Speed of the train calculated based on given conditions.
def train_speed : ℝ := train_length / time_to_pass_tree

-- The mathematically equivalent proof problem
theorem bridge_length : ∃ x : ℝ, x = 1200 
  ∧ (train_length + x = train_speed * time_to_pass_bridge) := 
by
  -- The proof is not needed but indicated with sorry.
  sorry

end bridge_length_l805_805893


namespace min_value_frac_sum_l805_805732

theorem min_value_frac_sum (x y : ℝ) 
  (h1 : 1 + x + y = 2) 
  (h2 : 0 < x) 
  (h3 : 0 < y) :
  ∃ x y, x + y = 1 ∧ (1/x + 4/y = 9) :=
by {
  use [1/3, 2/3],
  split,
  { norm_num },
  { norm_num, },
  sorry
}

end min_value_frac_sum_l805_805732


namespace a_is_constant_l805_805848

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, a n ≥ (a (n+2) + a (n+1) + a (n-1) + a (n-2)) / 4)

theorem a_is_constant : ∀ n m, a n = a m :=
by
  sorry

end a_is_constant_l805_805848


namespace quartic_polynomial_with_roots_l805_805930

theorem quartic_polynomial_with_roots :
  ∃ p : Polynomial ℚ, p.monic ∧ p.degree = 4 ∧ (p.eval (3 + Real.sqrt 5) = 0) ∧ (p.eval (2 - Real.sqrt 7) = 0) :=
by
  sorry

end quartic_polynomial_with_roots_l805_805930


namespace probability_Alex_Mel_Chelsea_winning_sequence_l805_805345

theorem probability_Alex_Mel_Chelsea_winning_sequence :
  (∃ m c : ℚ, 
    (2 / 5 : ℚ) + m + c = 1 ∧ 
    m = 3 * c ∧ 
    (8.choose 3) * (3.choose 4) * (1.choose 1) * 
    ( (2 / 5 : ℚ)^3 * (m)^4 * (c) ) = (881 / 1000 : ℚ)) :=
begin
  -- Definitions for the variables and conditions
  let A : ℚ := 2 / 5,
  have total_rounds : ℕ := 8,
  have alex_wins : ℕ := 3,
  have mel_wins : ℕ := 4,
  have chelsea_wins : ℕ := 1,
  
  assume hA : A + 3 * (1 - (A + m + c) / 3) + ((1 - (A + m)) / 4) = 1,

  -- Correct answer verification
  have answer : (8.choose 3) * (3.choose 4) * (1.choose 1) * 
                ( (2 / 5)^3 * (3 * (c))^4 * (c) ) = (881 / 1000),
  sorry
end

end probability_Alex_Mel_Chelsea_winning_sequence_l805_805345


namespace P_1989_divisible_by_3_994_l805_805581

theorem P_1989_divisible_by_3_994 :
  ∃ (k : ℕ), P 1989 = 3^994 * k := sorry

end P_1989_divisible_by_3_994_l805_805581


namespace find_element_X_weight_l805_805611

theorem find_element_X_weight (H C O mw : ℕ) (X : ℕ) 
  (hH : H = 1) 
  (hC : C = 12) 
  (hO : O = 16) 
  (h_mw : mw = 60) 
  (h_calculation : 3 * H + 1 * C + 1 * X + 2 * O = mw) : X = 13 := 
by 
  rw [hH, hC, hO, h_mw] at h_calculation 
  apply nat.add_left_cancel, swap, apply nat.add_cancel_right(_ + _ + _), 
  have : 47 + X = _ := _, exact this
sorry

end find_element_X_weight_l805_805611


namespace derived_seq_div_count_geq_of_nat_set_l805_805772

variable {m n : ℕ} (A : Finset ℤ)

def derived_sequence_div_count (A : Finset ℤ) (m : ℕ) : ℕ :=
  (A.product A).filter (λ ⟨a, b⟩, a ≠ b ∧ (a - b) % m = 0).card

theorem derived_seq_div_count_geq_of_nat_set (hm : m ≥ 2) (hn : n ≥ 2) 
  (hA : A.card = n) (B : Finset ℤ) (hB : B = (Finset.range n).map (λ x => (x + 1 : ℤ))) :
  derived_sequence_div_count A m ≥ derived_sequence_div_count B m := 
sorry

end derived_seq_div_count_geq_of_nat_set_l805_805772


namespace boy_speed_kmph_l805_805092

theorem boy_speed_kmph (distance_to_turning_point : ℕ) (round_trips : ℕ) (total_time_min : ℕ) (distance_km : ℕ) (total_time_hr : ℕ) : Prop :=
  distance_to_turning_point = 350 ∧
  round_trips = 5 ∧
  total_time_min = 30 ∧
  distance_km = (distance_to_turning_point * 2 * round_trips) / 1000 / 1000 ∧
  total_time_hr = total_time_min / 60 →
  distance_km / total_time_hr = 7

-- Equivalent Lean 4 statement where specific definitions from the conditions are used
example (distance_to_turning_point : ℕ) (round_trips : ℕ) (total_time_min : ℕ) : Prop :=
  let distance_per_trip := distance_to_turning_point * 2 in
  let total_distance := distance_per_trip * round_trips in
  let total_time_hr := total_time_min / 60 in
  let speed_kmph := total_distance / 1000 / total_time_hr in
  distance_to_turning_point = 350 ∧
  round_trips = 5 ∧
  total_time_min = 30 →
  speed_kmph = 7

-- Sorry is used to skip the actual proof implementation
by
  sorry

end boy_speed_kmph_l805_805092


namespace count_integers_in_range_num_of_integers_l805_805268

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805268


namespace monic_quartic_polynomial_exists_l805_805934

theorem monic_quartic_polynomial_exists :
  ∃ p : polynomial ℚ, p.monic ∧ p.eval (3 + real.sqrt 5) = 0 ∧ p.eval (3 - real.sqrt 5) = 0 ∧ 
  p.eval (2 - real.sqrt 7) = 0 ∧ p.eval (2 + real.sqrt 7) = 0 ∧ 
  p = polynomial.monic_quotient (polynomial.X^4 - 10 * polynomial.X^3 + 25 * polynomial.X^2 + 2 * polynomial.X - 12) :=
sorry

end monic_quartic_polynomial_exists_l805_805934


namespace find_multiple_l805_805871

theorem find_multiple (x m : ℤ) (hx : x = 13) (h : x + x + 2 * x + m * x = 104) : m = 4 :=
by
  -- Proof to be provided
  sorry

end find_multiple_l805_805871


namespace contractor_initial_plan_l805_805861

theorem contractor_initial_plan (D : ℝ) (initial_people total_people : ℕ)
    (initial_work_percentage days elapsed_work_percentage : ℝ)
    (additional_people : ℕ) :
  initial_people = 60 →
  elapsed_work_percentage = 0.4 →
  days = 25 →
  additional_people = 90 →
  let total_work_remaining := 1 - elapsed_work_percentage in
  let rate := initial_work_percentage / elapsed_work_percentage in
  let additional_days := rate * days in
  let new_total_people := initial_people + additional_people in
  let new_rate := (new_total_people / initial_people.to_real) * elapsed_work_percentage in
  let new_total_days := days + additional_days in
  D = new_total_days →
  D = 62.5 :=
begin
  sorry
end

end contractor_initial_plan_l805_805861


namespace sequence_sum_of_geometric_progressions_l805_805810

theorem sequence_sum_of_geometric_progressions
  (u1 v1 q p : ℝ)
  (h1 : u1 + v1 = 0)
  (h2 : u1 * q + v1 * p = 0) :
  u1 * q^2 + v1 * p^2 = 0 :=
by sorry

end sequence_sum_of_geometric_progressions_l805_805810


namespace average_of_real_solutions_is_three_halves_l805_805877

-- Given
variables {a b : ℝ}

-- Condition: The quadratic equation ax^2 - 3ax + b = 0 has two real solutions
axiom has_two_real_solutions (h : ∃ x1 x2 : ℝ, a * x1^2 - 3 * a * x1 + b = 0 ∧ a * x2^2 - 3 * a * x2 + b = 0)

-- To prove: The average of the two real solutions is 3/2
theorem average_of_real_solutions_is_three_halves (h : ∃ x1 x2 : ℝ, a * x1^2 - 3 * a * x1 + b = 0 ∧ a * x2^2 - 3 * a * x2 + b = 0) : 
  (let x1, x2 := classical.some h, classical.some (classical.some_spec h) in 
  (x1 + x2) / 2) = 3 / 2 :=
by
  sorry

end average_of_real_solutions_is_three_halves_l805_805877


namespace expected_gain_of_peculiar_die_l805_805917

theorem expected_gain_of_peculiar_die 
  (P_heads P_tails P_edge : ℝ) 
  (H_prob : P_heads = 1/4) 
  (T_prob : P_tails = 1/4) 
  (E_prob : P_edge = 1/2)
  (gain_heads : ℝ) (gain_tails : ℝ) (loss_edge : ℝ) 
  (H_gain : gain_heads = 2)
  (T_gain : gain_tails = 4)
  (E_loss : loss_edge = 6) :
  let E := P_heads * gain_heads + P_tails * gain_tails - P_edge * loss_edge in
  E = -3/2 := 
by {
  sorry
}

end expected_gain_of_peculiar_die_l805_805917


namespace height_cylinder_l805_805902

variables (r_c h_c r_cy h_cy : ℝ)
variables (V_cone V_cylinder : ℝ)
variables (r_c_val : r_c = 15)
variables (h_c_val : h_c = 20)
variables (r_cy_val : r_cy = 30)
variables (V_cone_eq : V_cone = (1/3) * π * r_c^2 * h_c)
variables (V_cylinder_eq : V_cylinder = π * r_cy^2 * h_cy)

theorem height_cylinder : h_cy = 1.67 :=
by
  rw [r_c_val, h_c_val, r_cy_val] at *
  have V_cone := V_cone_eq
  have V_cylinder := V_cylinder_eq
  sorry

end height_cylinder_l805_805902


namespace domain_y_eq_f_of_xplus1_l805_805642

noncomputable def domainOfTwoXPlusOne : Set ℝ := {x | 2 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}

-- This is the main statement we need to prove
theorem domain_y_eq_f_of_xplus1 (f : ℝ → ℝ) :
  domainOfTwoXPlusOne = {5} → (domain (λ x, f (x + 1)) = {4 ≤ x ∧ x ≤ 6}) :=
by
  sorry

end domain_y_eq_f_of_xplus1_l805_805642


namespace variance_Y_eq_l805_805879

theorem variance_Y_eq (f : ℝ → ℝ) (Y : ℝ → ℝ) :
  (∀ x, 0 < x ∧ x < π → f x = 1/2 * real.sin x) ∧ (∀ x, x ≤ 0 ∨ x ≥ π → f x = 0) ∧
  (∀ x, Y x = x^2) →
  (let g (y : ℝ) := (if 0 < y ∧ y < π^2 then (f (real.sqrt y)) / (2 * real.sqrt y) else 0) in
   let M_Y := ∫ y in 0..π^2, y * g y in
   let D_Y := ∫ y in 0..π^2, y^2 * g y - (M_Y)^2 in
   D_Y = (π^4 - 16 * π^2 + 80) / 4) :=
begin
  sorry
end

end variance_Y_eq_l805_805879


namespace range_of_a_l805_805977

variable (a : ℝ)

def prop_p : Prop := ∀ x ∈ Icc 1 2, exp x - a ≥ 0

theorem range_of_a (h : ¬ ¬ prop_p) : a ≤ Real.exp 1 := sorry

end range_of_a_l805_805977


namespace count_integer_values_l805_805246

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805246


namespace least_possible_value_of_n_l805_805543

noncomputable def cost_per_ornament (d : ℕ) (n : ℕ) : ℚ := d / n

noncomputable def revenue_from_three_ornaments (d : ℕ) (n : ℕ) : ℚ :=
  3 * (cost_per_ornament d n) / 3

noncomputable def profit_from_remaining_ornaments (d : ℕ) (n : ℕ) : ℚ :=
  (n - 3) * (cost_per_ornament d n + 10)

noncomputable def total_total_revenue (d : ℕ) (n : ℕ) : ℚ :=
  revenue_from_three_ornaments d n + profit_from_remaining_ornaments d n

noncomputable def total_profit (d : ℕ) (n : ℕ) : ℚ := 
  total_total_revenue d n - d

theorem least_possible_value_of_n
  (d : ℕ) (n : ℕ)
  (hd : 0 < d)
  (profit_condition : total_profit d n = 150) :
  n = 18 :=
begin
  sorry
end

end least_possible_value_of_n_l805_805543


namespace taxi_fare_proportional_l805_805897

theorem taxi_fare_proportional (fare_per_80km fare_per_100km: ℝ) (fare_proportional : ∀ (d1 d2: ℝ), d2 = d1 * (100 / 80) → fare_per_100km = fare_per_80km * (d2 / d1)):
  fare_per_80km = 160 ∧ 100 * fare_per_80km / 80 = fare_per_100km → fare_per_100km = 200 :=
by
  intros h
  cases h
  rw h_left
  simp
  exact h_right

end taxi_fare_proportional_l805_805897


namespace no_three_natural_numbers_l805_805924

theorem no_three_natural_numbers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
    (h4 : b ∣ a^2 - 1) (h5 : a ∣ c^2 - 1) (h6 : b ∣ c^2 - 1) : false :=
by
  sorry

end no_three_natural_numbers_l805_805924


namespace divisible_if_and_only_if_l805_805741

theorem divisible_if_and_only_if
  (m n : ℕ)
  (a : ℕ → ℕ)
  (N : ℕ)
  (M : ℕ)
  (r : ℕ → ℕ)
  (hN : N = ∑ i in finset.range (n + 1), a i * 10^i)
  (hr : ∀ i ∈ finset.range (n + 1), r i = 10^i % m)
  (hM : M = ∑ i in finset.range (n + 1), a i * r i) :
  (N % m = 0 ↔ M % m = 0) :=
by
  sorry

end divisible_if_and_only_if_l805_805741


namespace alice_wins_second_attempt_l805_805555

-- Define the conditions
def card_deck : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 20 }

def initial_guess_probability := (1 : ℚ) / 20

def probability_increases (n : ℕ) := (1 : ℚ) / (20 - n)

-- Define the probabilities of guessing correctly for each attempt
def alice_first_guess_prob := initial_guess_probability

def bob_first_guess_prob := probability_increases 1

def alice_second_guess_prob := probability_increases 2

-- Define the proof problem
theorem alice_wins_second_attempt :
  let P_Alice_1st_incorrect := 1 - alice_first_guess_prob,
      P_Bob_incorrect := 1 - bob_first_guess_prob,
      P_Alice_2nd_correct := alice_second_guess_prob
  in P_Alice_1st_incorrect * P_Bob_incorrect * P_Alice_2nd_correct = (1 : ℚ) / 20 :=
by {
  sorry
}

end alice_wins_second_attempt_l805_805555


namespace equal_magnitudes_necessary_not_sufficient_l805_805442

variable {V : Type*} [InnerProductSpace ℝ V]

theorem equal_magnitudes_necessary_not_sufficient {u v : V} (hu : u ≠ 0) (hv : v ≠ 0) :
  (∥u∥ = ∥v∥) → (∥u∥ = ∥v∥) ∧ ¬(u = v) :=
by
  sorry

end equal_magnitudes_necessary_not_sufficient_l805_805442


namespace unique_root_of_quadratic_eq_l805_805027

theorem unique_root_of_quadratic_eq (a b c : ℝ) (d : ℝ) 
  (h_seq : b = a - d ∧ c = a - 2 * d) 
  (h_nonneg : a ≥ b ∧ b ≥ c ∧ c ≥ 0) 
  (h_discriminant : (-(a - d))^2 - 4 * a * (a - 2 * d) = 0) :
  ∃ x : ℝ, (ax^2 - bx + c = 0) ∧ x = 1 / 2 :=
by
  sorry

end unique_root_of_quadratic_eq_l805_805027


namespace find_p_q_l805_805393

def vector_a (p : ℚ) := ⟨4, p, -2⟩ : ℝ^3
def vector_b (q : ℚ) := ⟨3, 2, q⟩ : ℝ^3

/-- Prove that if vectors a and b are orthogonal and have the same magnitude,
    then p = -29/12 and q = 43/12 -/
theorem find_p_q (p q : ℚ) (h1 : (4 : ℝ) * 3 + (p : ℝ) * 2 + (-2 : ℝ) * (q : ℝ) = 0)
  (h2 : 4 * 4 + (p : ℝ) * (p : ℝ) + (-2) * (-2) = 3 * 3 + 2 * 2 + (q : ℝ) * (q : ℝ)) :
  p = -29 / 12 ∧ q = 43 / 12 :=
by {
  sorry
}

end find_p_q_l805_805393


namespace complement_A_in_U_l805_805242

namespace SetTheory

def U : set ℕ := {2, 0, 1, 5}
def A : set ℕ := {0, 2}

theorem complement_A_in_U : U \ A = {1, 5} := by
  sorry

end SetTheory

end complement_A_in_U_l805_805242


namespace count_int_values_cube_bound_l805_805313

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805313


namespace sum_ages_is_13_l805_805370

-- Define the variables for the ages
variables (a b c : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  a * b * c = 72 ∧ a < b ∧ c < b

-- State the theorem to be proved
theorem sum_ages_is_13 (h : conditions a b c) : a + b + c = 13 :=
sorry

end sum_ages_is_13_l805_805370


namespace sale_price_of_trouser_l805_805367

theorem sale_price_of_trouser 
    (original_price : ℝ) 
    (discount_percentage : ℝ)
    (discounted_price : ℝ) :
    original_price = 100 → 
    discount_percentage = 0.90 → 
    discounted_price = original_price - (original_price * discount_percentage) → 
    discounted_price = 10 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  linarith

-- Proof is skipped, adding sorry
-- theorem sale_price_of_trouser 
--    (original_price : ℝ) 
--    (discount_percentage : ℝ)
--    (discounted_price : ℝ) :
--    original_price = 100 → 
--    discount_percentage = 0.90 → 
--    discounted_price = original_price - (original_price * discount_percentage) → 
--    discounted_price = 10 := 
-- sorry

end sale_price_of_trouser_l805_805367


namespace find_integer_l805_805791

theorem find_integer (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 150)
  (h2 : n % 7 = 0)
  (h3 : n % 9 = 3)
  (h4 : n % 6 = 3) : 
  n = 63 := by 
  sorry

end find_integer_l805_805791


namespace find_x_l805_805834

variable (N x : ℕ)
variable (h1 : N = 500 * x + 20)
variable (h2 : 4 * 500 + 20 = 2020)

theorem find_x : x = 4 := by
  -- The proof code will go here
  sorry

end find_x_l805_805834


namespace players_taking_mathematics_l805_805569

def RiverdaleAcademy := Unit

noncomputable def players_on_team (r : RiverdaleAcademy) : ℕ := 15

-- All 15 players are taking at least one of physics or mathematics
def taking_physics_or_math (r : RiverdaleAcademy) : Prop :=
  players_on_team r = 15

-- There are 9 players taking physics.
noncomputable def taking_physics (r : RiverdaleAcademy) : ℕ := 9

-- There are 4 players taking both physics and mathematics.
noncomputable def taking_both (r : RiverdaleAcademy) : ℕ := 4

-- Prove the number of students taking mathematics is 10.
theorem players_taking_mathematics (r : RiverdaleAcademy) :
  taking_physics_or_math r →
  taking_physics r = 9 →
  taking_both r = 4 →
  ∃ m : ℕ, m = 10 :=
by
  intros h1 h2 h3
  use 10
  sorry

end players_taking_mathematics_l805_805569


namespace arrangement_count_l805_805457

theorem arrangement_count :
  let boys := {b₁, b₂, b₃, b₄} in
  let girls := {g₁, g₂, g₃} in
  let boys_and_girls := boys ∪ girls in
  let arrangements := {a : List boys_and_girls // ∀ g ∈ girls, g ∈ a ∧ (∃ b ∈ boys, b > g)} in
  (∃ a ∈ arrangements,  2 of 3 girls stand together ∧ all 3 girls cannot stand together) →
  arrangements.card = 2880 :=
by
  sorry

end arrangement_count_l805_805457


namespace any_nat_representation_as_fraction_l805_805084

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l805_805084


namespace min_L_shapes_for_8x8_grid_l805_805883

def LShape := {positions : set (ℕ × ℕ) // positions.card = 3}

def valid_placement (grid_size : ℕ) (LShapes : list LShape) : Prop :=
  (∀ shape ∈ LShapes, (∀ (x y : ℕ), (x, y) ∈ shape.1 → x < grid_size ∧ y < grid_size)) ∧
  (∀ s1 s2 ∈ LShapes, s1 ≠ s2 → disjoint s1.1 s2.1) ∧
  (grid_size = 8)

theorem min_L_shapes_for_8x8_grid : ∃ k LShapes, 
    (valid_placement 8 LShapes) ∧ LShapes.length = k ∧ k = 11 :=
by
  sorry

end min_L_shapes_for_8x8_grid_l805_805883


namespace pentagon_area_is_fixed_l805_805533

-- Define a structure for a convex pentagon with given vertices.
structure ConvexPentagon (α : Type*) :=
(A B C D E : α)
(area_ABC : ℝ)
(area_BCD : ℝ)
(area_CDE : ℝ)
(area_DEA : ℝ)
(area_EAB : ℝ)

-- Define a theorem that states the correct problem.
theorem pentagon_area_is_fixed {α : Type*} [EuclideanSpace α] (P : ConvexPentagon α)
  (h1 : P.area_ABC = 1)
  (h2 : P.area_BCD = 1)
  (h3 : P.area_CDE = 1)
  (h4 : P.area_DEA = 1)
  (h5 : P.area_EAB = 1) :
  let area_P := Geometric.area (convex_hull ℝ (coe <$> ([P.A, P.B, P.C, P.D, P.E] : list α))) in
  area_P = (5 + Real.sqrt 5) / 2 :=
sorry

end pentagon_area_is_fixed_l805_805533


namespace benches_count_l805_805775

theorem benches_count (num_people_base6 : ℕ) (people_per_bench : ℕ) (num_people_base10 : ℕ) (num_benches : ℕ) :
  num_people_base6 = 204 ∧ people_per_bench = 2 ∧ num_people_base10 = 76 ∧ num_benches = 38 →
  (num_people_base10 = 2 * 6^2 + 0 * 6^1 + 4 * 6^0) ∧
  (num_benches = num_people_base10 / people_per_bench) :=
by
  sorry

end benches_count_l805_805775


namespace arithmetic_sequence_l805_805377

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l805_805377


namespace find_second_offset_l805_805607

theorem find_second_offset 
  (diagonal : ℝ) (offset1 : ℝ) (area_quad : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 20) (h2 : offset1 = 9) (h3 : area_quad = 150) :
  offset2 = 6 :=
by
  sorry

end find_second_offset_l805_805607


namespace average_value_permutations_l805_805187

theorem average_value_permutations (b : Fin 6 → ℕ) (h : ∀ i, 1 ≤ b i ∧ b i ≤ 6) :
  let sum_abs := (|b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5|)
      avg_value := (∑ (σ : Sym (Fin 6)), sum_abs σ) / (Nat.factorial 6) in
  let p := 55
      q := 3 in
  p + q = 58 :=
by sorry

end average_value_permutations_l805_805187


namespace sum_z_eq_220_l805_805165

noncomputable def sum_valid_two_digit_z (k l : ℕ) : ℕ :=
  if l = 4 ∧ (10 * k + l) ∈ {4, 24, 44, 64, 84}
  then 10 * k + l
  else 0

theorem sum_z_eq_220 : (∑ k in {0, 2, 4, 6, 8}, sum_valid_two_digit_z k 4) = 220 :=
by
  sorry

end sum_z_eq_220_l805_805165


namespace max_area_of_triangle_l805_805355

-- Definitions based on conditions
def ellipse := {p : ℝ × ℝ // p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1}
def focus : ℝ × ℝ := (c, 0)

-- Lean theorem statement
theorem max_area_of_triangle (a b c : ℝ) (h1 : 0 < b) (h2 : b < a)
  (h3 : ∃ A B : ℝ × ℝ, A.1 ^ 2 / a ^ 2 + A.2 ^ 2 / b ^ 2 = 1 ∧ B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1 ∧
    (line p, p ∈ ellipse, intersects points A and B))
  : ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ 
  (let area_of_triangle := λ A B F : ℝ × ℝ, (F.1 * (A.2 - B.2)) - (F.2 * (A.1 - B.1)) / 2 in
  (area_of_triangle A B focus) ≤ c * 2 * b ∧ (area_of_triangle A B focus) = c * 2 * b) :=
sorry

end max_area_of_triangle_l805_805355


namespace racing_game_cost_l805_805721

theorem racing_game_cost (total_spent : ℝ) (basketball_game_cost : ℝ) (racing_game_cost : ℝ)
  (h1 : total_spent = 9.43) (h2 : basketball_game_cost = 5.20) :
  racing_game_cost = total_spent - basketball_game_cost :=
by
  -- Defining local variables
  let total_spent := 9.43
  let basketball_game_cost := 5.20
  let expected_racing_game_cost := 4.23
  
  -- The statement of the theorem
  have h1 : total_spent = 9.43 := rfl
  have h2 : basketball_game_cost = 5.20 := rfl
  have h3 : racing_game_cost = total_spent - basketball_game_cost := by
    rw [h1, h2]
    
  show racing_game_cost = expected_racing_game_cost
  exact h3
sorry

end racing_game_cost_l805_805721


namespace molecular_weight_BaSO4_l805_805494

-- Definitions for atomic weights of elements.
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_S : ℝ := 32.07
def atomic_weight_O : ℝ := 16.00

-- Defining the number of atoms in BaSO4
def num_Ba : ℕ := 1
def num_S : ℕ := 1
def num_O : ℕ := 4

-- Statement to be proved
theorem molecular_weight_BaSO4 :
  (num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O) = 233.40 := 
by
  sorry

end molecular_weight_BaSO4_l805_805494


namespace count_n_integers_l805_805275

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805275


namespace cos_of_tan_l805_805193

/-- Given a triangle ABC with angle A such that tan(A) = -5/12, prove cos(A) = -12/13. -/
theorem cos_of_tan (A : ℝ) (h : Real.tan A = -5 / 12) : Real.cos A = -12 / 13 := by
  sorry

end cos_of_tan_l805_805193


namespace compass_legs_cannot_swap_l805_805793

-- Define the problem conditions: compass legs on infinite grid, constant distance d.
def on_grid (p q : ℤ × ℤ) : Prop := 
  ∃ d : ℕ, d * d = (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) ∧ d > 0

-- Define the main theorem as a Lean 4 statement
theorem compass_legs_cannot_swap (p q : ℤ × ℤ) (h : on_grid p q) : 
  ¬ ∃ r s : ℤ × ℤ, on_grid r p ∧ on_grid s p ∧ p ≠ q ∧ r = q ∧ s = p :=
sorry

end compass_legs_cannot_swap_l805_805793


namespace number_of_integers_satisfying_cubed_inequality_l805_805308

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805308


namespace racing_game_cost_l805_805724

theorem racing_game_cost (total_spent : ℝ) (basketball_game_cost : ℝ) (racing_game_cost : ℝ) 
  (h_total_spent : total_spent = 9.43) 
  (h_basketball_game_cost : basketball_game_cost = 5.2) : 
  racing_game_cost = 4.23 :=
by
  rw [h_total_spent, h_basketball_game_cost]
  sorry

end racing_game_cost_l805_805724


namespace simplify_expression_l805_805474

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l805_805474


namespace emma_missing_coins_l805_805598

theorem emma_missing_coins (x : ℝ) : 
  (x - ((2/3) * x) + ((4/5) * (2/3) * x)) = (13/15) * x :=
begin
  have coins_lost := (2 / 3) * x,
  have coins_found := (4 / 5) * coins_lost,
  have total_after_retracing := x - coins_lost + coins_found,
  have total_after_retracing_simplified : total_after_retracing = (13 / 15) * x,
    -- simplification steps
  exact total_after_retracing_simplified,
end

end emma_missing_coins_l805_805598


namespace percentage_of_sikh_boys_l805_805707

theorem percentage_of_sikh_boys (total_boys muslim_percentage hindu_percentage other_boys : ℕ) 
  (h₁ : total_boys = 300) 
  (h₂ : muslim_percentage = 44) 
  (h₃ : hindu_percentage = 28) 
  (h₄ : other_boys = 54) : 
  (10 : ℝ) = 
  (((total_boys - (muslim_percentage * total_boys / 100 + hindu_percentage * total_boys / 100 + other_boys)) * 100) / total_boys : ℝ) :=
by
  sorry

end percentage_of_sikh_boys_l805_805707


namespace general_term_formula_l805_805786

theorem general_term_formula (a : ℕ → ℚ) (h₁ : ∀ n, a n = n / (2 * n - 1)) :
  ∀ n, a n = n / (2 * n - 1) :=
by sorry

end general_term_formula_l805_805786


namespace inequality_range_of_k_l805_805690

theorem inequality_range_of_k 
  (a b k : ℝ)
  (h : ∀ a b : ℝ, a^2 + b^2 ≥ 2 * k * a * b) : k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end inequality_range_of_k_l805_805690


namespace math_proof_problem_l805_805026

def sequence_a_terms_product (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = (∏ i in Finset.range n, a i)

-- Conditions: Given
def condition_b1 : ℝ := 1
def condition_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) = 2 * a n

def condition_bn (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2^(n*(n - 1)/2)

def condition_cn (a : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n, c n = Real.logb 4 (a (2*n - 1))

-- Define sequences and proof goals
def sequences (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :=
  sequence_a_terms_product a b ∧ condition_b b a ∧ condition_bn b ∧ condition_cn a c

def question1 (b : ℕ → ℝ) : Prop :=
  b 4 = 4

def question2 (c : ℕ → ℝ) : Prop :=
  (∑ k in Finset.range n, c (k + 1)) = (n * (n - 1)) / 2

-- Proof problem
theorem math_proof_problem (a b c : ℕ → ℝ) (n : ℕ) (hb : b 1 = condition_b1)
  (h_seq : sequences a b c) :
  question1 b ∧ question2 c :=
by sorry

end math_proof_problem_l805_805026


namespace number_of_integers_satisfying_cubed_inequality_l805_805307

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805307


namespace rachel_plants_lamps_l805_805415

-- Definitions based on conditions
def basil_plants : ℕ := 2
def aloe_plant : ℕ := 1
def white_lamps : ℕ := 2
def red_lamps : ℕ := 2

-- The theorem to prove the problem statement
theorem rachel_plants_lamps : 
  ∃ (ways : ℕ), ways = 14 ∧
  (ways = (cases_all_same_color basil_plants aloe_plant white_lamps red_lamps) + 
          (cases_different_color_for_aloe basil_plants aloe_plant white_lamps red_lamps) + 
          (cases_basil_different_color basil_plants aloe_plant white_lamps red_lamps)) := sorry

-- Helper definitions (as placeholders)
def cases_all_same_color (basil_plants aloe_plant white_lamps red_lamps : ℕ) : ℕ := sorry
def cases_different_color_for_aloe (basil_plants aloe_plant white_lamps red_lamps : ℕ) : ℕ := sorry
def cases_basil_different_color (basil_plants aloe_plant white_lamps red_lamps : ℕ) : ℕ := sorry

end rachel_plants_lamps_l805_805415


namespace system_solution_l805_805425

theorem system_solution:
  ∃ (x y z : ℝ), 
    x + y + z = 15 ∧ 
    x^2 + y^2 + z^2 = 81 ∧ 
    xy + xz = 3yz ∧ 
    (x = 6 ∧ y = 3 ∧ z = 6) ∨ (x = 6 ∧ y = 6 ∧ z = 3) :=
by {
  use [6, 3, 6],
  split,
  linarith,
  split,
  linarith,
  split,
  linarith,
  left,
  linarith,

  use [6, 6, 3],
  split,
  linarith,
  split,
  linarith,
  split,
  linarith,
  right,
  linarith,
}

end system_solution_l805_805425


namespace kelly_snacks_total_l805_805728

def pounds_of_snacks (ounces_to_pounds : ℝ → ℝ) (peanuts_pounds : ℝ) (raisins_ounces : ℝ) (almonds_pounds : ℝ) (almonds_ounces : ℝ) : ℝ :=
  peanuts_pounds + (ounces_to_pounds raisins_ounces) + almonds_pounds

theorem kelly_snacks_total :
  let ounces_to_pounds := (λ x: ℝ, x / 16)
  ∧ let peanuts_pounds := 0.1
  ∧ let raisins_ounces := 5
  ∧ let almonds_pounds := 0.3
  ∧ let almonds_ounces := 4.8
  in
    pounds_of_snacks ounces_to_pounds peanuts_pounds raisins_ounces almonds_pounds almonds_ounces = 0.7125 :=
by
  sorry

end kelly_snacks_total_l805_805728


namespace intersection_of_M_and_N_l805_805397

noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N :
    (M ∩ N) = {0, 1} := 
by
  sorry

end intersection_of_M_and_N_l805_805397


namespace median_eq_6point5_l805_805065
open Nat

def median_first_twelve_positive_integers (l : List ℕ) : ℝ :=
  (l !!! 5 + l !!! 6) / 2

theorem median_eq_6point5 : median_first_twelve_positive_integers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 6.5 :=
by
  sorry

end median_eq_6point5_l805_805065


namespace max_water_surface_area_parallel_l805_805530

theorem max_water_surface_area_parallel (e : ℝ) (h₀ : e = 1) :
  ∃ A, A = sqrt 2 ∧ A ≤ max (1 : ℝ) (max (1 : ℝ) ((3 * sqrt 3) / 4)) :=
by
  sorry

end max_water_surface_area_parallel_l805_805530


namespace num_int_values_satisfying_ineq_l805_805258

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805258


namespace general_formula_a_n_sum_terms_b_n_l805_805648

noncomputable def S (n : ℕ) : ℝ := 1 - (1 / 2)^n

noncomputable def a (n : ℕ) : ℝ :=
  if h : n = 0 then 0 else (1 / 2)^n

noncomputable def b (n : ℕ) : ℝ := 
  if h : n = 0 then 0 else n / a n

noncomputable def T (n : ℕ) : ℝ := 
  if h : n = 0 then 0 else (n - 1) * 2^(n+1) + 2

theorem general_formula_a_n (n : ℕ) (h : n > 0) : a n = (1 / 2)^n := by
  sorry

theorem sum_terms_b_n (n : ℕ) (h : n > 0) : (finset.range n).sum (λ i, b (i+1)) = T n := by
  sorry

end general_formula_a_n_sum_terms_b_n_l805_805648


namespace convince_jury_of_innocence_l805_805568

-- Define the types for Knights, Liars, and the innocence state.
inductive Role
| knight : Role
| liar : Role

-- Axioms defining the conditions given in the problem
axiom knight_tells_truth : ∀ (P : Prop), Role.knight → P → P
axiom liar_lies : ∀ (P : Prop), Role.liar → ¬P → P

-- Condition that the criminal is either a knight or a liar, and you are innocent.
axiom criminal_is_knight_or_liar : ∀ (r : Role), r = Role.knight ∨ r = Role.liar
axiom you_are_innocent : ¬ guilty

-- Define what it means to be guilty
def guilty : Prop := -- assuming guilty means the statement "You committed the crime"
sorry

-- Define the statement you make in court
def court_statement : Prop := "I am either an innocent knight or a guilty liar"

-- Prove that the statement is true given the conditions
theorem convince_jury_of_innocence (r : Role) (innocent : ¬ guilty) : court_statement :=
by {
  -- Proof steps would go here, but currently omitted with sorry
  sorry,
}


end convince_jury_of_innocence_l805_805568


namespace sound_together_time_l805_805134

-- Define the intervals in minutes
def chime_interval := 18
def alarm_interval := 24
def bell_interval := 30

-- Calculate the LCM of the intervals
def LCM_intervals (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

-- Define the time when they all sound together
def initial_time := 8 * 60  -- 8:00 AM in minutes

-- Calculate the next time they'll sound together
def next_sound_time := initial_time + LCM_intervals chime_interval alarm_interval bell_interval

-- Proof that the next sound time equals 2:00 PM
theorem sound_together_time : 
  next_sound_time = 14 * 60 :=  -- 2:00 PM in minutes
by 
  unfold next_sound_time LCM_intervals initial_time chime_interval alarm_interval bell_interval
  norm_num
  norm_num
  sorry

end sound_together_time_l805_805134


namespace general_term_formula_l805_805649

def Sn (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 2^(n + 1)

theorem general_term_formula (a_n : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → Sn a_n n = (2 * a_n n - 2^(n + 1))) :
  ∀ n : ℕ, n > 0 → a_n n = (n + 1) * 2^n :=
sorry

end general_term_formula_l805_805649


namespace find_constants_l805_805940

variable (x : ℝ)

/-- Restate the equation problem and the constants A, B, C, D to be found. -/
theorem find_constants 
  (A B C D : ℝ)
  (h : ∀ x, x^3 - 7 = A * (x - 3) * (x - 5) * (x - 7) + B * (x - 2) * (x - 5) * (x - 7) + C * (x - 2) * (x - 3) * (x - 7) + D * (x - 2) * (x - 3) * (x - 5)) :
  A = 1/15 ∧ B = 5/2 ∧ C = -59/6 ∧ D = 42/5 :=
  sorry

end find_constants_l805_805940


namespace max_min_of_cubic_function_l805_805439

theorem max_min_of_cubic_function :
  let f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - 9 * x 
  ∃ x_max x_min, (-2 < x_max ∧ x_max < 4) ∧ (-2 < x_min ∧ x_min < 4) ∧
  is_max (f x_max) ∧ is_min (f x_min) ∧ f x_max = 5 ∧ f x_min = -27 :=
by {
  -- Definitions/conditions
  let f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - 9 * x,
  -- sorry can be used to skip the proof here
  sorry
}

-- Define the helper functions for checking if the function values are indeed the max and min
def is_max (y : ℝ) : Prop := ∀ (x : ℝ), (-2 < x ∧ x < 4) → y ≥ x^3 - 3*x^2 - 9*x
def is_min (y : ℝ) : Prop := ∀ (x : ℝ), (-2 < x ∧ x < 4) → y ≤ x^3 - 3*x^2 - 9*x

end max_min_of_cubic_function_l805_805439


namespace exotic_meat_original_price_l805_805120

theorem exotic_meat_original_price (y : ℝ) :
  (0.75 * (y / 4) = 4.5) → y = 96 :=
by
  intro h
  sorry

end exotic_meat_original_price_l805_805120


namespace projection_is_correct_l805_805625

variables {α : Type*} [inner_product_space ℝ α] 
variables (a e : α) (θ : ℝ)

noncomputable def projection_vector (a e : α) : α :=
  (‖a‖ * real.cos θ) • (e / ‖e‖)

theorem projection_is_correct
  (h₁ : ‖a‖ = 2)
  (h₂ : ‖e‖ = 1)
  (h₃ : θ = 3 * real.pi / 4) :
  projection_vector a e = -real.sqrt 2 • e := 
sorry

end projection_is_correct_l805_805625


namespace find_a_l805_805998

noncomputable def f (x a : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 3) → a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l805_805998


namespace tablets_med_A_l805_805856

open Nat

theorem tablets_med_A (numA numB minTablets : ℕ) (hB : numB = 14) (hMin : minTablets = 16) (hCond : 14 + 2 = minTablets) : numA = 2 := 
begin
  sorry
end

end tablets_med_A_l805_805856


namespace city_division_into_1014_districts_possible_l805_805859

-- Define the necessary structures and properties
structure City (Square : Type) where
  traffic_routes : Square → Square → Prop
  outgoing_routes_count : Square → Nat
  has_exactly_two_outgoing_routes : ∀ sq, outgoing_routes_count sq = 2

-- Define the property we need to prove
def districts_property (Square : Type) (D : Fin 1014) (city : City Square) (d : Square → D) : Prop :=
  (∀ sq1 sq2, city.traffic_routes sq1 sq2 → d sq1 ≠ d sq2) ∧
  (∀ (dist1 dist2 : D), dist1 ≠ dist2 →
    (∀ sq1 sq2, (d sq1 = dist1 ∧ d sq2 = dist2) → city.traffic_routes sq1 sq2 ∨ city.traffic_routes sq2 sq1))

-- Target theorem
theorem city_division_into_1014_districts_possible {Square : Type} (city : City Square) :
  ∃ (d : Square → Fin 1014), districts_property Square (Fin 1014) city d :=
by 
  -- The proof is skipped
  sorry

end city_division_into_1014_districts_possible_l805_805859


namespace monic_quartic_polynomial_exists_l805_805933

theorem monic_quartic_polynomial_exists :
  ∃ p : polynomial ℚ, p.monic ∧ p.eval (3 + real.sqrt 5) = 0 ∧ p.eval (3 - real.sqrt 5) = 0 ∧ 
  p.eval (2 - real.sqrt 7) = 0 ∧ p.eval (2 + real.sqrt 7) = 0 ∧ 
  p = polynomial.monic_quotient (polynomial.X^4 - 10 * polynomial.X^3 + 25 * polynomial.X^2 + 2 * polynomial.X - 12) :=
sorry

end monic_quartic_polynomial_exists_l805_805933


namespace maximize_triangle_area_l805_805051

-- Definitions of points and tangents on a circle
variables {P1 P2 P3 A0 A1 A2 : Point}
variable {k : Circle}
variables {t1 t2 t3 : Line}

-- Conditions
axiom circle_points : PointsOnCircle [P1, P2, P3] k
axiom nonparallel_tangents : TangentsAtPointsNonParallel [t1, t2] [P1, P2] k
axiom tangents_intersect : IntersectAtPoint t1 t2 A0
axiom point_on_arc : PointOnSmallerArc P3 P1 P2 k
axiom tangent_at_P3 : TangentAtPoint t3 P3 k
axiom t3_intersects_t1_at_A1 : IntersectAtPoint t3 t1 A1
axiom t3_intersects_t2_at_A2 : IntersectAtPoint t3 t2 A2

-- Goal
theorem maximize_triangle_area : 
  (MidpointOfSmallerArc P3 P1 P2 k) → 
  (AreaOfTriangle A0 A1 A2 isMaximum) := 
sorry

end maximize_triangle_area_l805_805051


namespace amount_earned_from_notebooks_l805_805797

def total_revenue (a : ℝ) (b : ℝ) : ℝ :=
  (70 * (1 + 20 / 100) * a) + (30 * (a - b))

theorem amount_earned_from_notebooks
  (a b : ℝ) :
  total_revenue a b = (70 * 1.2 * a) + (30 * (a - b)) :=
by
  sorry

end amount_earned_from_notebooks_l805_805797


namespace polynomial_has_right_triangle_roots_l805_805102

theorem polynomial_has_right_triangle_roots (p : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 = c^2 ∧
   (a, b, c) = root_triple (x^3 - 2*p*(p+1)*x^2 + (p^4 + 4*p^3 - 1)*x - 3*p^3)) ↔ p = Real.sqrt 2 := 
sorry

end polynomial_has_right_triangle_roots_l805_805102


namespace function_range_l805_805227

def f (x : ℕ) : ℤ := 2 * ↑x - 3

theorem function_range : 
  (∃ y : ℤ, ∃ x : ℕ, 1 ≤ x ∧ x ≤ 5 ∧ y = f x) ↔ 
  (∀ y : ℤ, y ∈ {-1, 1, 3, 5, 7}) := 
sorry

end function_range_l805_805227


namespace count_integers_in_range_num_of_integers_l805_805267

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805267


namespace calculate_expression_l805_805580

-- Declare the mixed number to improper fraction conversions
def mixed_to_improper(num : ℚ, whole_part : ℤ, frac_part_num : ℚ, frac_part_den : ℚ) : ℚ :=
  whole_part + frac_part_num / frac_part_den

-- Define the mixed numbers given in the problem
def three_and_three_fourths := mixed_to_improper 3 3 3 4
def two_and_two_thirds := mixed_to_improper 2 2 2 3

-- Arithmetic operations
def one_point_three : ℚ := 1.3
def three : ℚ := 3

-- Problem statement in Lean 4
theorem calculate_expression : 
  (three_and_three_fourths * one_point_three + three / two_and_two_thirds) = 6 :=
by
  -- The proof would go here
  sorry

end calculate_expression_l805_805580


namespace division_of_fractions_l805_805479

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805479


namespace problem_solution_l805_805628

-- Define the conditions
variable (k : ℝ)
def line_l (x : ℝ) := k * (x + 2)

def circle_M (x y : ℝ) := (x - 2)^2 + y^2 = 1

-- Point P where line l intersects the y-axis
def P := (0, k * 2)

-- Tangents from point P to circle M, denoted by points A and B
def tangents (P : ℝ × ℝ) (circle : ℝ × ℝ → Prop) : Prop := sorry

-- Line AB intersects line MP at point C
def line_AB (A B : ℝ × ℝ) := sorry
def line_MP (M P : ℝ × ℝ) := sorry
def intersection_C (AB MP : ℝ → ℝ) := sorry

-- Define Q point
def Q := ((7:ℝ) / 4, 0)

-- Proof outline
-- Theorem to prove
theorem problem_solution : 
  (∀ {P : ℝ × ℝ}, line_l 0 = P.2 → 
     circle_M 2 1 → tangents P circle_M → 
     line_MP (2,1) P = sorry → P = (0, 4)) ∧ 
  (∃ A B : ℝ × ℝ, line_AB A B = sorry ∧ line_MP (2,1) P = sorry) →
  ((k = abs (\sqrt{15}/15) ∨ k = -abs (\sqrt{15}/15)) ∧ 
  (∃ N : ℝ × ℝ, ∀ A B : ℝ × ℝ, line_AB A B = sorry → 
  ∃ P : ℝ × ℝ, line_l 0 = P.2 ∧ P.2 = k * 2 ∧ 
  P.1 = 0 ∧ circle_M 2 1) ∧ 
  (∀ C : ℝ × ℝ, ∃ Q : ℝ × ℝ, |C - Q| = 1/4)) :=
sorry

end problem_solution_l805_805628


namespace finite_possibilities_T_l805_805756

theorem finite_possibilities_T :
  ∃ T ∈ ({(0, 0, 0), (0, -1, 1), (-1, 1, 0), (1, 0, -1)} : set (ℂ × ℂ × ℂ)),
  ∃ (x y z k : ℂ),
  (x (x - 1) + 2 * y * z = k) ∧
  (y (y - 1) + 2 * z * x = k) ∧
  (z (z - 1) + 2 * x * y = k) ∧
  T = (x - y, y - z, z - x) :=
by
  sorry

end finite_possibilities_T_l805_805756


namespace expression_expansion_l805_805172

noncomputable def expand_expression : Polynomial ℤ :=
 -2 * (5 * Polynomial.X^3 - 7 * Polynomial.X^2 + Polynomial.X - 4)

theorem expression_expansion :
  expand_expression = -10 * Polynomial.X^3 + 14 * Polynomial.X^2 - 2 * Polynomial.X + 8 :=
by
  sorry

end expression_expansion_l805_805172


namespace first_fourth_liar_distances_l805_805410

structure Islander (position : ℕ) :=
  (is_knight : Bool)

def lineup := List (Islander)

variables (islanders : lineup)
variables (distance : ℕ → ℕ → ℕ)

axiom stand_1_meter_apart {i j : ℕ} (h₁ : i < j) (h₂ : j < 4) : distance i j = 1
axiom knight_liar {i : ℕ} : (islanders[i] = 1) ↔ (islanders[3-i] = 1)
axiom trib_statement (i : ℕ) (knight: bool) : 
  (i = 1 ⊕ i = 2) → 
  (islanders[i].is_knight = knight) → 
  (if knight then distance i (i+1) = 1 else distance i (i+1) ≠ 1)

theorem first_fourth_liar_distances : 
  (λ d, d = 1 ∨ d = 2 ∨ d = 4) = (λ d, ∃ i j, 0 <= i < 4 ∧ 0 <= j < 4 ∧ distance i j = d) := 
begin
  sorry
end

end first_fourth_liar_distances_l805_805410


namespace range_of_a_l805_805968

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  real.log (1 + a * x) - (2 * x) / (x + 2)

def extreme_points (a : ℝ) : ℝ × ℝ :=
  let x1 := (2 * real.sqrt (a * (1 - a))) / a
  let x2 := -(2 * real.sqrt (a * (1 - a))) / a
  (x1, x2)

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : x1 = (2 * real.sqrt (a * (1 - a))) / a)
  (h4 : x2 = -(2 * real.sqrt (a * (1 - a))) / a)
  (h5 : f a x1 + f a x2 > 0) : 
  ∃ (a : ℝ), 1/2 < a ∧ a < 1 :=
begin
  sorry
end

end range_of_a_l805_805968


namespace like_terms_only_in_pair_D_l805_805836

-- Define like terms
def like_terms (term1 term2 : String) : Prop := 
  term1 = "3xy" ∧ term2 = "-2xy"

-- Problem conditions as pairs
def pair_A : List String := ["a^2b", "ab^2"]
def pair_B : List String := ["3x", "3y"]
def pair_C : List String := ["6abc", "6bc"]
def pair_D : List String := ["3xy", "-2xy"]

-- Theorem statement
theorem like_terms_only_in_pair_D :
  ((like_terms pair_A.head pair_A.getLast) = false) ∧
  ((like_terms pair_B.head pair_B.getLast) = false) ∧
  ((like_terms pair_C.head pair_C.getLast) = false) ∧
  ((like_terms pair_D.head pair_D.getLast) = true) :=
  by sorry

end like_terms_only_in_pair_D_l805_805836


namespace p_necessary_but_not_sufficient_for_q_l805_805966

noncomputable def p (x : ℝ) : Prop := abs x ≤ 2
noncomputable def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by 
  sorry

end p_necessary_but_not_sufficient_for_q_l805_805966


namespace common_chord_eqn_l805_805669

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 6 * y + 1 = 0) ∧
  (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) →
  3 * x - 4 * y + 6 = 0 :=
by
  intro h
  sorry

end common_chord_eqn_l805_805669


namespace probability_of_sum_18_when_four_dice_rolled_l805_805685

noncomputable def probability_sum_18 : ℝ :=
  sorry -- This is a placeholder for the actual calculation

theorem probability_of_sum_18_when_four_dice_rolled :
  probability_sum_18 = 1/72 :=
begin
  sorry -- This is a placeholder for the actual proof
end

end probability_of_sum_18_when_four_dice_rolled_l805_805685


namespace dozen_chocolate_bars_cost_l805_805794

theorem dozen_chocolate_bars_cost
  (cost_mag : ℕ → ℝ) (cost_choco_bar : ℕ → ℝ)
  (H1 : cost_mag 1 = 1)
  (H2 : 4 * (cost_choco_bar 1) = 8 * (cost_mag 1)) :
  12 * (cost_choco_bar 1) = 24 := 
sorry

end dozen_chocolate_bars_cost_l805_805794


namespace largest_interesting_number_l805_805821

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def interesting_number (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.nodup ∧ 
  ∀ (a b : ℕ), list.pairwise (λ a b, is_perfect_square (a + b)) digits

theorem largest_interesting_number : 
  ∃ n : ℕ, interesting_number n ∧ ∀ m : ℕ, interesting_number m → m ≤ 6310972 := 
by 
  sorry

end largest_interesting_number_l805_805821


namespace smallest_m_integral_roots_l805_805079

theorem smallest_m_integral_roots (m : ℕ) : 
  (∃ p q : ℤ, (10 * p * p - ↑m * p + 360 = 0) ∧ (p + q = m / 10) ∧ (p * q = 36) ∧ (p % q = 0 ∨ q % p = 0)) → 
  m = 120 :=
by
sorry

end smallest_m_integral_roots_l805_805079


namespace smallest_b_for_factorization_l805_805949

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ r s : ℕ, (r * s = 3258) → (b = r + s)) ∧ (∀ c : ℕ, (∀ r' s' : ℕ, (r' * s' = 3258) → (c = r' + s')) → b ≤ c) :=
sorry

end smallest_b_for_factorization_l805_805949


namespace part1_part2i_part2ii_l805_805202

noncomputable def f (x m : ℝ) : ℝ := (1 / 2) * Real.log (x^2 - m) + m * (x - 1) / (x^2 - m)

theorem part1 (x : ℝ) : (1 < x ∨ x < -1) →
  (let f := f x 1 in
   (∀ x, 1 < x → deriv (f x 1) x > 0 ∧
    (∀ x, x < -1 → deriv (f x 1) x < 0))) := by
  sorry

theorem part2i (x m : ℝ) (hm : m > 0)
  (h : ∀ x, (Real.sqrt m) < x → f x m ≥ 1) : m ≥ (1 + Real.sqrt 5) / 2 := by
  sorry

theorem part2ii (t : ℝ) (ht : t > 1) : 
  (1 / 2 * Real.log (t^2 - t) ≤ t * Real.sqrt (1 + t) - t - 1 ) := by
  sorry

end part1_part2i_part2ii_l805_805202


namespace general_term_formula_l805_805240

-- Definition of the sequence {a_n}
noncomputable def a : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := (3 * a n - 1) / (4 * a n + 7)

-- Proposition stating the correct answer
theorem general_term_formula :
  ∀ n : ℕ, a (n + 1) = (9 - 4 * (n + 1)) / (2 + 8 * (n + 1)) := by
  sorry

end general_term_formula_l805_805240


namespace division_of_fractions_l805_805492

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805492


namespace log_sum_simplification_l805_805571

open Real

theorem log_sum_simplification :
  (2 * log 2 10 + log 2 0.04) = 2 :=
by
  sorry

end log_sum_simplification_l805_805571


namespace volume_of_cube_l805_805815

theorem volume_of_cube (P : ℝ) (h : P = 32) : ∃ V : ℝ, V = 512 :=
by
  have edge_length : ℝ := P / 4
  have volume := edge_length^3
  use volume
  calc
    volume = (P / 4)^3       : by sorry
          ... = (32 / 4)^3   : by rw [h]
          ... = 8^3          : by simp
          ... = 512          : by norm_num

end volume_of_cube_l805_805815


namespace triangle_problem_l805_805559

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def has_same_area (a b : ℕ) (area : ℝ) : Prop :=
  let s := (2 * a + b) / 2
  let areaT := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  areaT = area

def has_same_perimeter (a b : ℕ) (perimeter : ℕ) : Prop :=
  2 * a + b = perimeter

def correct_b (b : ℕ) : Prop :=
  b = 5

theorem triangle_problem
  (a1 a2 b1 b2 : ℕ)
  (h1 : is_isosceles_triangle a1 a1 b1)
  (h2 : is_isosceles_triangle a2 a2 b2)
  (h3 : has_same_area a1 b1 (Real.sqrt 275))
  (h4 : has_same_perimeter a1 b1 22)
  (h5 : has_same_area a2 b2 (Real.sqrt 275))
  (h6 : has_same_perimeter a2 b2 22)
  (h7 : ¬(a1 = a2 ∧ b1 = b2)) : correct_b b2 :=
by
  sorry

end triangle_problem_l805_805559


namespace david_average_marks_l805_805920

-- Define the marks for each subject
def english_marks := 96
def mathematics_marks := 95
def physics_marks := 82
def chemistry_marks := 97
def biology_marks := 95

-- Define the weightage for each subject
def english_weight := 0.10
def mathematics_weight := 0.20
def physics_weight := 0.30
def chemistry_weight := 0.20
def biology_weight := 0.20

-- Define the weighted average calculation
def weighted_average : ℝ :=
  (english_marks * english_weight) +
  (mathematics_marks * mathematics_weight) +
  (physics_marks * physics_weight) +
  (chemistry_marks * chemistry_weight) +
  (biology_marks * biology_weight)

-- Prove that the calculated weighted average equals 91.6
theorem david_average_marks : weighted_average = 91.6 := by
  sorry

end david_average_marks_l805_805920


namespace five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l805_805047

theorem five_digit_numbers_greater_than_20314_and_formable_with_0_to_5 :
  (∃ (f : Fin 6 → Fin 5) (n : ℕ), 
    (n = 120 * 3 + 24 * 4 + 6 * 3 - 1) ∧
    (n = 473) ∧ 
    (∀ (x : Fin 6), f x = 0 ∨ f x = 1 ∨ f x = 2 ∨ f x = 3 ∨ f x = 4 ∨ f x = 5) ∧
    (∀ (i j : Fin 5), i ≠ j → f i ≠ f j)) :=
sorry

end five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l805_805047


namespace number_of_integers_satisfying_cubed_inequality_l805_805300

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805300


namespace FalseConverseD_l805_805088

/-- Proposition A: If two lines are parallel, then their corresponding angles are equal. -/
def PropositionA (l1 l2 : Line) (parallel : Parallel l1 l2) : CorrespondingAnglesEqual l1 l2 :=
sorry

/-- Converse of Proposition A: If corresponding angles are equal, then the two lines are parallel. -/
def ConverseA (l1 l2 : Line) (anglesEqual : CorrespondingAnglesEqual l1 l2) : Parallel l1 l2 :=
sorry

/-- Proposition B: The diagonals of a parallelogram bisect each other. -/
def PropositionB (p : Parallelogram) : DiagonalsBisectEachOther p :=
sorry

/-- Converse of Proposition B: If the diagonals of a quadrilateral bisect each other, then it is a parallelogram. -/
def ConverseB (q : Quadrilateral) (diagonalsBisect : DiagonalsBisectEachOther q) : Parallelogram q :=
sorry

/-- Proposition C: All four sides of a rhombus are equal. -/
def PropositionC (r : Rhombus) : AllSidesEqual r :=
sorry 

/-- Converse of Proposition C: If a quadrilateral has all sides equal, then it is a rhombus. -/
def ConverseC (q : Quadrilateral) (allSidesEqual : AllSidesEqual q) : Rhombus q :=
sorry 

/-- Proposition D: All four angles of a square are right angles. -/
def PropositionD (s : Square) : AllRightAngles s :=
sorry

/-- Converse of Proposition D: If a quadrilateral has all right angles, then it is a square. -/
def ConverseD (q : Quadrilateral) (allRightAngles : AllRightAngles q) : Square q :=
sorry

/-- The proposition with a false converse is proposition D. -/
theorem FalseConverseD : ConverseD = False :=
sorry

end FalseConverseD_l805_805088


namespace count_int_values_cube_bound_l805_805315

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805315


namespace problem_solution_l805_805638

noncomputable def f (x : ℝ) : ℝ :=
if h : 1 ≤ x ∧ x ≤ 2 then
  real.log x
else if h₁ : ((0 ≤ x ∧ x < 1) ∨ (2 < x ∧ x ≤ 3)) then
  real.log (4 - x)
else
  -- Assuming f is zero otherwise based on the given range and conditions
  0

lemma even_function (x : ℝ) : f x = f (-x) :=
begin
  sorry,
end

lemma functional_equation (x : ℝ) : f (x + 1) = f (1 - x) :=
begin
  sorry,
end

lemma range_of_a (a : ℝ) : 
  (0 < a ∧ a ≤ 1 / 5) → 
  ∃ x1 x2, 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 ∧ (f x1 + a * x1 - 1 = 0) ∧ (f x2 + a * x2 - 1 = 0) :=
begin
  sorry,
end

theorem problem_solution : 
  (\(0 < a ∧ a ≤ 1 / 5) ∧ (\(1 - real.log 2) / 4 < a) → 
  ∃ x1 x2, 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 ∧ (f x1 + a * x1 - 1 = 0) ∧ (f x2 + a * x2 - 1 = 0) :=
begin
  sorry,
end

end problem_solution_l805_805638


namespace line_general_eq_curve_cartesian_eq_intersection_value_l805_805352

-- Definitions and conditions
def line_parametric (t : ℝ) : ℝ × ℝ := (1 + t, t * real.sqrt 3)
def curve_equation (rho theta : ℝ) : Prop := rho^2 * (3 + (real.sin theta)^2) = 12

-- Part 1: Prove the general equation of the line
theorem line_general_eq (t : ℝ) : (line_parametric t).2 = real.sqrt 3 * (line_parametric t).1 - real.sqrt 3 :=
by sorry

-- Part 1: Prove the Cartesian equation of the curve
theorem curve_cartesian_eq (x y : ℝ) :
  (x^2 + y^2) * (3 + y^2 / (x^2 + y^2)) = 12 ↔ (x^2) / 4 + (y^2) / 3 = 1 :=
by sorry

-- Part 2: Calculate the value of (1 / |MA|) + (1 / |MB|)
theorem intersection_value (A B M : ℝ × ℝ) (hA : curve_cartesian_eq A.1 A.2) 
  (hB : curve_cartesian_eq B.1 B.2) (hlA : (A = line_parametric _)) 
  (hlB : (B = line_parametric _)) (hM : M = (1, 0)) : 
  1 / (dist M A) + 1 / (dist M B) = 4 / 3 :=
by sorry

end line_general_eq_curve_cartesian_eq_intersection_value_l805_805352


namespace inequality_l805_805734

theorem inequality (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a * b + 2 * a + b / 2 :=
sorry

end inequality_l805_805734


namespace division_of_fractions_l805_805493

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805493


namespace match_scheduling_ways_l805_805150

def different_ways_to_schedule_match (num_players : Nat) (num_rounds : Nat) : Nat :=
  (num_rounds.factorial * num_rounds.factorial)

theorem match_scheduling_ways : different_ways_to_schedule_match 4 4 = 576 :=
by
  sorry

end match_scheduling_ways_l805_805150


namespace sum_odd_coefficients_zero_l805_805627

def poly (a b c d e x : ℝ) : ℝ := (a * x^4 + b * x^3 + c * x^2 + d * x + e)^5 * (a * x^4 - b * x^3 + c * x^2 - d * x + e)^5

theorem sum_odd_coefficients_zero (a b c d e : ℝ) :
  let F := poly a b c d e in
  (∀ (coeff : ℕ → ℝ), ∑ i in finset.range 21, if i % 2 = 1 then coeff i else 0 = 0) :=
sorry

end sum_odd_coefficients_zero_l805_805627


namespace present_worth_is_120_l805_805099

theorem present_worth_is_120 (BG : ℝ) (Rate : ℝ) (Time : ℝ) (PW : ℝ) :
  BG = 36 → Rate = 10 → Time = 3 → 
  PW = 36 / ((Rate × Time) / 100) :=
by
  intros hBG hRate hTime
  rw [hBG, hRate, hTime]
  simp
  norm_num
  done

end present_worth_is_120_l805_805099


namespace number_of_distinct_intersection_points_l805_805163

theorem number_of_distinct_intersection_points :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
  let line := {p : ℝ × ℝ | p.1 = 4}
  let intersection_points := circle ∩ line
  ∃! p : ℝ × ℝ, p ∈ intersection_points :=
by
  sorry

end number_of_distinct_intersection_points_l805_805163


namespace number_of_triples_satisfying_gcd_and_lcm_l805_805180

theorem number_of_triples_satisfying_gcd_and_lcm :
  let count := (Cardinal.mk {p : ℕ × ℕ × ℕ // 
    let (a, b, c) := p
    (Nat.gcd (Nat.gcd a b) c = 33) ∧ 
    Nat.lcm (Nat.lcm a b) c = 3 ^ 19 * 11 ^ 15
  }) in
  count = 9072 :=
by
  sorry

end number_of_triples_satisfying_gcd_and_lcm_l805_805180


namespace original_number_is_0_2_l805_805409

theorem original_number_is_0_2 :
  ∃ x : ℝ, (1 / (1 / x - 1) - 1 = -0.75) ∧ x = 0.2 :=
by
  sorry

end original_number_is_0_2_l805_805409


namespace max_real_part_of_cubed_l805_805750

theorem max_real_part_of_cubed (z_1 z_2 z_3 z_4 z_5 : ℂ) :
  z_1 = -1 ∧
  z_2 = -complex.sqrt 2 + complex.I ∧
  z_3 = -1 + complex.sqrt 3 * complex.I ∧
  z_4 = 2 * complex.I ∧
  z_5 = -1 - complex.sqrt 3 * complex.I →
  ∃ w ∈ ({z_1, z_2, z_3, z_4, z_5} : set ℂ), 
  (∀ v ∈ ({z_1, z_2, z_3, z_4, z_5} : set ℂ), (v ^ 3).re ≤ (w ^ 3).re) ∧ w = z_4 :=
by
  sorry

end max_real_part_of_cubed_l805_805750


namespace count_integer_values_l805_805251

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805251


namespace largest_valid_n_l805_805155

open Set

-- Define the triangle property
def triangle_property (S : Set ℕ) : Prop := ∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → 
  (a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the condition that any twelve-element subset of a set has the triangle property
def has_triangle_property (S : Set ℕ) : Prop := ∀ T : Finset ℕ, T.card = 12 → T ⊆ S → triangle_property T

-- Define the main theorem
theorem largest_valid_n : ∃ n, n = 808 ∧ has_triangle_property (Ico 5 (809 : ℕ)) :=
begin
  use 808,
  split,
  { refl, },
  {
    intros T hT hSub,
    sorry
  }

end largest_valid_n_l805_805155


namespace complete_the_square_l805_805082

/-- Given the quadratic equation x^2 - 2x - 7 = 0,
    prove that it can be written in the form (x - 1)^2 = 8
    using the completing the square method. -/
theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 7 = 0 ↔ (x - 1)^2 = 8 :=
begin
  sorry
end

end complete_the_square_l805_805082


namespace arithmetic_sequence_a12_l805_805636

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_sequence a a1 d) :
  a 11 = 23 :=
by
  -- condtions
  let a1_val := 1
  let d_val := 2
  have ha1 : a1 = a1_val := sorry
  have hd : d = d_val := sorry
  
  -- proof
  rw [ha1, hd] at h
  
  sorry

end arithmetic_sequence_a12_l805_805636


namespace racing_game_cost_l805_805723

theorem racing_game_cost (total_spent : ℝ) (basketball_game_cost : ℝ) (racing_game_cost : ℝ) 
  (h_total_spent : total_spent = 9.43) 
  (h_basketball_game_cost : basketball_game_cost = 5.2) : 
  racing_game_cost = 4.23 :=
by
  rw [h_total_spent, h_basketball_game_cost]
  sorry

end racing_game_cost_l805_805723


namespace second_degree_polynomial_l805_805376

-- Defining concavity
def is_concave (f : ℝ → ℝ) : Prop := ∀ x y t, 0 ≤ t ∧ t ≤ 1 → f(t * x + (1 - t) * y) ≥ t * f(x) + (1 - t) * f(y)

-- Main theorem statement
theorem second_degree_polynomial (f g : ℝ → ℝ) (A B C : ℝ) :
  is_concave f ∧ continuous g ∧ (∀ x y, f(x + y) + f(x - y) - 2 * f(x) = g(x) * y^2) →
  ∃ A B C, ∀ x, f(x) = A * x + B * x^2 + C :=
begin
  sorry
end

end second_degree_polynomial_l805_805376


namespace pen_cost_price_l805_805551

-- Define the variables and assumptions
variable (x : ℝ)

-- Given conditions
def profit_one_pen (x : ℝ) := 10 - x
def profit_three_pens (x : ℝ) := 20 - 3 * x

-- Statement to prove
theorem pen_cost_price : profit_one_pen x = profit_three_pens x → x = 5 :=
by
  sorry

end pen_cost_price_l805_805551


namespace isosceles_triangle_largest_angle_l805_805350

theorem isosceles_triangle_largest_angle (α : ℝ) (β : ℝ)
  (h1 : 0 < α) (h2 : α = 30) (h3 : β = 30):
  ∃ γ : ℝ, γ = 180 - 2 * α ∧ γ = 120 := by
  sorry

end isosceles_triangle_largest_angle_l805_805350


namespace truck_mileage_l805_805136

theorem truck_mileage (distance : ℝ) (gas_in_tank : ℝ) (gas_needed: ℝ) (total_gas : ℝ):
  distance = 90 → gas_in_tank = 12 → gas_needed = 18 → total_gas = gas_in_tank + gas_needed →
  (distance / total_gas) = 3 :=
by
  intros h_distance h_gas_in_tank h_gas_needed h_total_gas
  have h1 : total_gas = 30 := by
    rw [h_gas_in_tank, h_gas_needed, h_total_gas]
  have h2 : distance = 90 := h_distance
  rw [h2, h1]
  norm_num
  sorry

end truck_mileage_l805_805136


namespace subset_implies_range_l805_805213

open Set

-- Definitions based on the problem statement
def A : Set ℝ := { x : ℝ | x < 5 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x < a }

-- Theorem statement
theorem subset_implies_range (a : ℝ) (h : A ⊆ B a) : a ≥ 5 :=
sorry

end subset_implies_range_l805_805213


namespace find_AB_length_l805_805783

def AB_length {A B C D : Point ℝ} (side_length : ℝ) (vectors_AB_AD : (Vector ℝ) × (Vector ℝ)) : ℝ :=
  let pentagon_side := 1
  let AB_vector := ⟨1, 0, 0⟩
  let AD_vector := ⟨0, 1, 0⟩
  -- all necessary conditions are set here
  if (side_length = pentagon_side) ∧ (vectors_AB_AD = (AB_vector, AD_vector)) then 2 else 0

theorem find_AB_length : ∀ (A B C D : Point ℝ) (side_length : ℝ) (vectors_AB_AD : (Vector ℝ) × (Vector ℝ)),
  side_length = 1 →
  vectors_AB_AD = (⟨1, 0, 0⟩, ⟨0, 1, 0⟩) →
  AB_length side_length vectors_AB_AD = 2 :=
by
  intros A B C D side_length vectors_AB_AD h₁ h₂
  rw [AB_length, h₁, h₂]
  simp
  sorry

end find_AB_length_l805_805783


namespace claire_profit_l805_805152

noncomputable def total_loaves := 60
noncomputable def price_morning := 3
noncomputable def price_afternoon := 2
noncomputable def price_evening := 1.5
noncomputable def cost_per_loaf := 1
noncomputable def fixed_cost := 10

noncomputable def loaves_sold_morning := total_loaves / 3
noncomputable def loaves_remaining_after_morning := total_loaves - loaves_sold_morning
noncomputable def loaves_sold_afternoon := loaves_remaining_after_morning / 2
noncomputable def loaves_remaining_after_afternoon := loaves_remaining_after_morning - loaves_sold_afternoon
noncomputable def loaves_sold_evening := loaves_remaining_after_afternoon

noncomputable def revenue_morning := loaves_sold_morning * price_morning
noncomputable def revenue_afternoon := loaves_sold_afternoon * price_afternoon
noncomputable def revenue_evening := loaves_sold_evening * price_evening
noncomputable def total_revenue := revenue_morning + revenue_afternoon + revenue_evening

noncomputable def total_production_cost := total_loaves * cost_per_loaf
noncomputable def total_cost := total_production_cost + fixed_cost

noncomputable def profit := total_revenue - total_cost

theorem claire_profit : profit = 60 := by
  sorry

end claire_profit_l805_805152


namespace num_int_values_satisfying_ineq_l805_805262

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805262


namespace accurate_scale_l805_805118

-- Definitions for the weights on each scale
variables (a b c d e x : ℝ)

-- Given conditions
def condition1 := c = b - 0.3
def condition2 := d = c - 0.1
def condition3 := e = a - 0.1
def condition4 := c = e - 0.1
def condition5 := 5 * x = a + b + c + d + e

-- Proof statement
theorem accurate_scale 
  (h1 : c = b - 0.3)
  (h2 : d = c - 0.1)
  (h3 : e = a - 0.1)
  (h4 : c = e - 0.1)
  (h5 : 5 * x = a + b + c + d + e) : e = x :=
by
  sorry

end accurate_scale_l805_805118


namespace line_through_point_parallel_to_given_l805_805435

open Real

theorem line_through_point_parallel_to_given (x y : ℝ) :
  (∃ (m : ℝ), (y - 0 = m * (x - 1)) ∧ x - 2*y - 1 = 0) ↔
  (x = 1 ∧ y = 0 ∧ ∃ l, x - 2*y - l = 0) :=
by sorry

end line_through_point_parallel_to_given_l805_805435


namespace flowerman_sale_number_of_flowers_l805_805011

theorem flowerman_sale_number_of_flowers (n : ℕ) (h1 : ∀ i j : ℕ, i ≠ j → prices i ≠ prices j)
    (h2 : ∃ k, nth_highest prices k = 17 ∧ nth_lowest prices k = 42) : n = 58 :=
by
  sorry

end flowerman_sale_number_of_flowers_l805_805011


namespace conversion_relation_l805_805771

theorem conversion_relation (a b c d e f g h : ℕ) :
  let hops_in_skips := b / a,
      jumps_in_hops := d / c,
      leaps_in_jumps := f / e,
      meters_in_leaps := g / h in 
  1 * (g / h) * (f / e) * (d / c) * (b / a) = (gbfd / aehc) := sorry

end conversion_relation_l805_805771


namespace exists_rectangle_same_color_l805_805506

-- Define the structure and predicates for the problem.
def colored_tile (r : ℕ) (c : ℕ) : Prop := r < 4 ∧ c < 19

def color := {white, blue, red}

def coloring (t : ℕ × ℕ) : color

axiom color_count (c : color) : (∑ r in finset.range 4, ∑ col in finset.range 19, (coloring (r, col) = c).to_add) ≥ 26

theorem exists_rectangle_same_color :
  ∃ r₁ r₂ c₁ c₂, 1 ≤ r₁ + 1 ∧ r₁ < r₂ ∧ r₂ ≤ 4 ∧ 1 ≤ c₁ + 1 ∧ c₁ < c₂ ∧ c₂ ≤ 19 ∧
  coloring (r₁, c₁) = coloring (r₁, c₂) ∧ 
  coloring (r₁, c₁) = coloring (r₂, c₁) ∧ 
  coloring (r₁, c₁) = coloring (r₂, c₂) :=
by {
  sorry,
}

end exists_rectangle_same_color_l805_805506


namespace coefficient_of_monomial_l805_805005

def monomial := (4 * Real.pi * a^2 * b) / 5

theorem coefficient_of_monomial :
  -- Define the coefficient concept and state what needs to be proved
  ∃ c : ℝ, monomial = c * (a^2 * b) ∧ c = (4 * Real.pi) / 5 :=
sorry

end coefficient_of_monomial_l805_805005


namespace median_of_first_twelve_positive_integers_l805_805055

theorem median_of_first_twelve_positive_integers :
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth (5)).getD 0 + (lst.nth (6)).getD 0 / 2 = 6.5 :=
by
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let median := ((lst.nth (5)).getD 0 + (lst.nth (6)).getD 0) / 2
  show median = 6.5
  sorry

end median_of_first_twelve_positive_integers_l805_805055


namespace digit_A_is_1_prime_l805_805164

theorem digit_A_is_1_prime :
  ∃ A : ℕ, A = 1 ∧ Prime (303101) :=
begin
  use 1,
  split,
  {
    refl,
  },
  {
    -- Prove that 303101 is prime
    sorry,
  }
end

end digit_A_is_1_prime_l805_805164


namespace sqrt_sum_equality_l805_805496

open Real

theorem sqrt_sum_equality :
  (sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8) :=
sorry

end sqrt_sum_equality_l805_805496


namespace number_of_sets_satisfying_union_condition_l805_805444

theorem number_of_sets_satisfying_union_condition :
  {A : set ℕ | ({0, 1} ∪ A = {0, 1}) ∧ A ⊆ {0, 1}}.finite.card = 4 := by
  sorry

end number_of_sets_satisfying_union_condition_l805_805444


namespace puppies_left_l805_805140

theorem puppies_left (initial_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : initial_puppies = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_puppies = initial_puppies - given_away) : 
  remaining_puppies = 5 :=
  by
  sorry

end puppies_left_l805_805140


namespace train_length_proof_l805_805045

def v1_kmh : ℤ := 60  -- speed of the first train in km/hr
def v2_kmh : ℤ := 40  -- speed of the second train in km/hr
def time_seconds : ℝ := 12.59899208063355  -- time to cross each other in seconds
def L2_m : ℝ := 210  -- length of the second train in meters

def kmh_to_ms (v_kmh : ℤ) : ℝ := (v_kmh * 1000) / 3600  -- conversion from km/hr to m/s

def v1_ms := kmh_to_ms v1_kmh  -- speed of the first train in m/s
def v2_ms := kmh_to_ms v2_kmh  -- speed of the second train in m/s
def relative_speed_ms := v1_ms + v2_ms  -- relative speed in m/s

def crossing_distance : ℝ := relative_speed_ms * time_seconds  -- total distance covered

theorem train_length_proof : 
  ∃ (L1 : ℝ), L1 + L2_m = crossing_distance ∧ L1 = 140 :=
by
  sorry

end train_length_proof_l805_805045


namespace quadratic_roots_real_probability_l805_805433

theorem quadratic_roots_real_probability :
  let I := set.Ioc (0:ℝ) 2 in
  ∃ (p q : ℝ), p ∈ I ∧ q ∈ I ∧ 
  (∀ p ∈ I, ∀ q ∈ I, p^2 - 4*q ≥ 0) → (∃ f : ℝ → ℝ, (∫ x in 0..2, ((x^2) / 4)) = (4 / 6) / 4) := 
begin
  sorry
end

end quadratic_roots_real_probability_l805_805433


namespace sixth_term_l805_805239

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 4 ∧
  ∀ n, a (n + 2) = a (n + 1) - a n

theorem sixth_term (a : ℕ → ℤ) (h : sequence a) : a 6 = -3 :=
by
  -- Proof goes here
  sorry

end sixth_term_l805_805239


namespace trajectory_of_complex_point_l805_805429

open Complex Topology

theorem trajectory_of_complex_point (z : ℂ) (hz : ‖z‖ ≤ 1) : 
  {w : ℂ | ‖w‖ ≤ 1} = {w : ℂ | w.re * w.re + w.im * w.im ≤ 1} :=
sorry

end trajectory_of_complex_point_l805_805429


namespace local_call_cost_proof_l805_805958

noncomputable def local_call_cost_per_minute : ℝ :=
  let x := 5 in
  x

theorem local_call_cost_proof (x : ℝ) 
  (dad_call_time : ℝ := 45) 
  (bro_call_time : ℝ := 31)
  (local_call_cost : ℝ := x)
  (int_call_cost : ℝ := 25)
  (total_cost : ℝ := 1000)
  (call_cost_eq : dad_call_time * local_call_cost + bro_call_time * int_call_cost = total_cost) : 
  local_call_cost = 5 := 
by 
  sorry

end local_call_cost_proof_l805_805958


namespace least_number_of_cubes_is_10_l805_805880

noncomputable def volume_of_block (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

noncomputable def volume_of_cube (side : ℕ) : ℕ :=
  side ^ 3

noncomputable def least_number_of_cubes (length width height : ℕ) : ℕ := 
  volume_of_block length width height / volume_of_cube (gcd_three_numbers length width height)

theorem least_number_of_cubes_is_10 : least_number_of_cubes 15 30 75 = 10 := by
  sorry

end least_number_of_cubes_is_10_l805_805880


namespace DE_parallel_AC_and_passes_through_incenter_l805_805112

-- Definitions based on conditions
variables {A B C D E X Y Z : Type*}

-- Assume these points are defined to satisfy the conditions
-- of midpoints of specified arcs and the intersections
-- Definitions of specific geometric properties and constructions
axiom circle_circumscribed (A B C : Type*) : Prop
axiom arc_midpoint (X : Type*) (A B C : Type*) : Prop
axiom arc_midpoint_opposite (X : Type*) (A B C : Type*) : Prop
axiom arc_midpoint (Y : Type*) (A B C : Type*) : Prop
axiom arc_midpoint (Z : Type*) (A B C : Type*) : Prop
axiom intersect_YZ_AB (Y Z : Type*) (A B : Type*) : Prop
axiom intersect_YX_BC (Y X : Type*) (B C : Type*) : Prop
axiom incenter (I : Type*) (A B C : Type*) : Prop

-- The problem statement to be proved in Lean
theorem DE_parallel_AC_and_passes_through_incenter
  (A B C D E X Y Z I : Type*)
  [circle_circumscribed A B C]
  [arc_midpoint_opposite X A B C]
  [arc_midpoint Y A C]
  [arc_midpoint Z A B]
  [intersect_YZ_AB Y Z A B]
  [intersect_YX_BC Y X B C]
  [incenter I A B C] :
  (parallel D E A C) ∧ (passes_through I D E) := 
  sorry

end DE_parallel_AC_and_passes_through_incenter_l805_805112


namespace chord_existence_l805_805042

theorem chord_existence (O M : ℝ × ℝ) (R l : ℝ) :
  let OM := dist O M
  let d := sqrt (R^2 - (l/2)^2)
  l ≤ 2*R → OM ≤ R →
  (OM < d → ¬ ∃ A B : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2) ∧
  (OM = d → ∃! A B : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2) ∧
  (OM > d → ∃ A B C D : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2 ∧ dist O C = R ∧ dist O D = R ∧ dist C D = l ∧ M = (C + D) / 2 ∧ A ≠ C) :=
by {
  intros O M R l OM d hl hOM,
  split,
  { intro h1,
    sorry },
  split,
  { intro h2,
    sorry },
  { intro h3,
    sorry }
}

end chord_existence_l805_805042


namespace shape_of_pentagon_AMNCD_is_D_l805_805567

-- Definitions based on the conditions of the problem
def paper_square := Type -- Define a type for the paper square
def midpoint (A B : ℝ) : ℝ := (A + B) / 2

variables (A B C D : ℝ)
variable (shape_D : Type) -- We assume shape_D corresponds to the description of option (D)

-- Given conditions
axiom fold_vertically : paper_square → paper_square
axiom fold_horizontally : paper_square → paper_square
axiom cut_triangle (M N B : ℝ) (AB BC : ℝ) : paper_square → paper_square
axiom unfold_shape (p : paper_square) : Type

-- Midpoints as per problem
def M := midpoint A B
def N := midpoint B C

-- Statement: The shape of the unfolded and flattened paper 
theorem shape_of_pentagon_AMNCD_is_D : 
  unfold_shape (cut_triangle M N B (midpoint A B) (midpoint B C) 
              (fold_horizontally (fold_vertically paper_square))) = shape_D := sorry

end shape_of_pentagon_AMNCD_is_D_l805_805567


namespace floor_calc_l805_805170

theorem floor_calc : (Int.floor (4 * (7 - 1 / 3))) = 26 := by
  sorry

end floor_calc_l805_805170


namespace find_p_q_r_l805_805460

theorem find_p_q_r : ∃ (p q r : ℕ), 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 
  4 * real.sqrt (real.cbrt 7 - real.cbrt 6) = real.cbrt p + real.cbrt q - real.cbrt r ∧ 
  p + q + r = 93 :=
begin
  sorry
end

end find_p_q_r_l805_805460


namespace find_a_b_l805_805185

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem find_a_b (a b : ℝ) (x : ℝ) (h : 5 * (log a x) ^ 2 + 2 * (log b x) ^ 2 = (10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) + (Real.log x) ^ 2) :
  b = a ^ (2 / (5 + Real.sqrt 17)) ∨ b = a ^ (2 / (5 - Real.sqrt 17)) :=
sorry

end find_a_b_l805_805185


namespace remainder_of_a_cubed_l805_805735

theorem remainder_of_a_cubed (n : ℕ) (hn : nat.prime n) (hn_odd : n % 2 = 1) (a : ℤ) 
  (ha_inverse : a * a ≡ 1 [MOD n]) : a^3 ≡ a [MOD n] := by
  sorry

end remainder_of_a_cubed_l805_805735


namespace p_plus_q_l805_805021

noncomputable def a : ℚ :=
  let p := 32
  let q := 1156
  p / q

theorem p_plus_q : 
  let p := 32
  let q := 1156
  let S := 504
  ∑ x in {x | ∃ (w : ℝ), floor x = w ∧ (x - w) = (a * (w + (x - w))^2)}, x = 504 → 
  p + q = 1188 :=
sorry

end p_plus_q_l805_805021


namespace parking_lot_arrangements_l805_805872

theorem parking_lot_arrangements :
  let num_spaces := 8
  let num_trucks := 2
  let num_cars := 2
  (∀ (adjacency_constraint : ∀ (vehicles : list string), 
    vehicles.count "truck" = num_trucks ∧ vehicles.count "car" = num_cars → 
    (∃ (start1 start2 : ℕ), 
      start1 < start2 ∧
      start1 + num_trucks ≤ num_spaces ∧ 
      start2 + num_cars ≤ num_spaces ∧ 
      list.all (list.range' start1 num_trucks) (λ i, vehicles.nth i = some "truck") ∧
      list.all (list.range' start2 num_cars) (λ i, vehicles.nth i = some "car")) +
      list.all (list.range' start2 num_cars) (λ i, vehicles.nth i = some "car") ∧
      list.all (list.range' start1 num_trucks) (λ i, vehicles.nth i = some "truck"))) →
  ∃ P : ℕ, P = 120

end parking_lot_arrangements_l805_805872


namespace population_net_increase_per_day_l805_805097

theorem population_net_increase_per_day (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) (net_increase : ℚ) :
  birth_rate = 7 / 2 ∧
  death_rate = 2 / 2 ∧
  seconds_per_day = 24 * 60 * 60 ∧
  net_increase = (birth_rate - death_rate) * seconds_per_day →
  net_increase = 216000 := 
by
  sorry

end population_net_increase_per_day_l805_805097


namespace circle_radius_l805_805858

theorem circle_radius (r : ℝ) (x y : ℝ) (h₁ : x = π * r ^ 2) (h₂ : y = 2 * π * r - 6) (h₃ : x + y = 94 * π) : 
  r = 10 :=
sorry

end circle_radius_l805_805858


namespace ellipse_eq_slopes_k_k1_k2_const_l805_805205

-- Define the conditions for the ellipse
def ellipse (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Provide additional conditions for the specific problem
def conditions (a b : ℝ) (h₁ : a > b) :=
  b = Real.sqrt 2 ∧ a + b = 3 * Real.sqrt 2

-- Translate the proof problem (I) about finding the ellipse equation
theorem ellipse_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  conditions a b h₃ →
  ellipse a b h₁ h₂ :=
sorry

-- Translate the proof problem (II.1) about the slopes k1 and k2
def line_through_vertex := ∀ x y : ℝ, y = (1/2) * x + Real.sqrt 2

-- Define k1 and k2 under given conditions
def slopes (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ( (y1 - 1) / (x1 - 2),
    (y2 - 1) / (x2 - 2) )

-- Define the proof problem for slopes k1 and k2 under the specified conditions
theorem slopes_k (a b x1 y1 x2 y2 : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  conditions a b h₃ →
  line_through_vertex x1 y1 →
  slopes x1 y1 x2 y2 = (-(Real.sqrt 2 - 1) / 2, (Real.sqrt 2 - 1) / 2) :=
sorry

-- Translate the proof problem (II.2) about the constance value of k1 + k2 being 0
theorem k1_k2_const (k1 k2 : ℝ) (h₁ : k1 + k2 = 0) :
  k1 + k2 = 0 :=
sorry

end ellipse_eq_slopes_k_k1_k2_const_l805_805205


namespace count_integer_values_l805_805247

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805247


namespace power_function_at_4_l805_805788

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_4 {α : ℝ} :
  power_function α 2 = (Real.sqrt 2) / 2 →
  α = -1/2 →
  power_function α 4 = 1 / 2 :=
by
  intros h1 h2
  rw [h2, power_function]
  sorry

end power_function_at_4_l805_805788


namespace monkey_rope_problem_l805_805536

variable (m M w_rope : ℕ) -- m: age of the monkey, M: age of the mother, w_rope: weight of the rope in pounds
variable (l : ℕ) -- l: length of the rope in feet

-- Conditions as definitions
theorem monkey_rope_problem
  (h1 : m + M = 4)
  (h2 : monkey_weight = M) -- weight of the monkey is the age of the mother in pounds
  (h3 : w_load = monkey_weight) -- weight of the load is the weight of the monkey  
  (h4 : M = 2 * (m - (M / 2 - (m + (m / 3 - M / 3) / 3)))) -- mother and monkey age relation
  (h5 : w_rope + w_load = 1.5 * (w_load - w_rope + monkey_weight)) -- weight relationship with rope
  : l = 5 := sorry

end monkey_rope_problem_l805_805536


namespace find_dividing_line_l805_805119

/--
A line passing through point P(1,1) divides the circular region \{(x, y) \mid x^2 + y^2 \leq 4\} into two parts,
making the difference in area between these two parts the largest. Prove that the equation of this line is x + y - 2 = 0.
-/
theorem find_dividing_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ (A B C : ℝ), A * 1 + B * 1 + C = 0 ∧
                 (∀ x y, x^2 + y^2 ≤ 4 → A * x + B * y + C = 0 → (x + y - 2) = 0) :=
sorry

end find_dividing_line_l805_805119


namespace correct_average_weight_l805_805098

theorem correct_average_weight (n : ℕ) (avg_incorrect : ℝ) (wrong_weight correct_weight : ℝ)
  (h_n : n = 20) (h_avg_incorrect : avg_incorrect = 58.4) (h_wrong_weight : wrong_weight = 56)
  (h_correct_weight : correct_weight = 66) :
  (avg_incorrect * n + (correct_weight - wrong_weight)) / n = 58.9 :=
by
  rw [h_n, h_avg_incorrect, h_wrong_weight, h_correct_weight]
  norm_num
  sorry

end correct_average_weight_l805_805098


namespace square_base_edge_length_l805_805615

noncomputable def radius : ℝ := 2
noncomputable def side_length : ℝ := 4
noncomputable def diagonal (a : ℝ) : ℝ := a * real.sqrt 2

theorem square_base_edge_length :
  let c := side_length in
  let d := diagonal c in
  d = 4 * real.sqrt 2 :=
by
  let c := side_length
  have h1 : c = 2 * radius := by sorry
  have h2 : d = diagonal c := by sorry
  have h3 : d = 4 * real.sqrt 2 := by sorry
  exact h3

end square_base_edge_length_l805_805615


namespace sum_of_3_digit_numbers_l805_805080

theorem sum_of_3_digit_numbers : 
  let digits := [2, 3, 4] in
  let numbers := list.permutations digits |>.map (λ l, 100 * l.head! + 10 * l.get! 1 + l.get! 2) in
  numbers.sum = 1998 :=
by sorry

end sum_of_3_digit_numbers_l805_805080


namespace num_real_solutions_l805_805686

theorem num_real_solutions (x : ℝ) (A B : Set ℝ) (hx : x ∈ A) (hx2 : x^2 ∈ A) :
  A = {0, 1, 2, x} → B = {1, x^2} → A ∪ B = A → 
  ∃! y : ℝ, y = -Real.sqrt 2 ∨ y = Real.sqrt 2 :=
by
  intro hA hB hA_union_B
  sorry

end num_real_solutions_l805_805686


namespace total_spent_l805_805366

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l805_805366


namespace f_decreasing_intervals_f_extreme_values_l805_805658

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x - Real.sin x

def decreases_on_interval (x : ℝ) (k : ℤ) : Prop :=
  (2 * k * Real.pi - Real.pi / 3) < x ∧ x < (2 * k * Real.pi + Real.pi / 3)

theorem f_decreasing_intervals :
  ∀ (k : ℤ), ∃ (a b : ℝ), decreases_on_interval a k ∧ decreases_on_interval b k :=
sorry

theorem f_extreme_values :
  let a_min := -Real.pi
  let a_max := Real.pi
  let f_a_min := -Real.pi / 2
  let f_a_max := Real.pi / 2 in
  ∃ (x_min x_max : ℝ),
    (x_min = a_min ∧ x_max = a_max ∧ f x_min = f_a_min ∧ f x_max = f_a_max) :=
sorry

end f_decreasing_intervals_f_extreme_values_l805_805658


namespace quotient_of_integers_l805_805835

theorem quotient_of_integers
  (a b : ℤ)
  (h : 1996 * a + b / 96 = a + b) :
  b / a = 2016 ∨ a / b = 2016 := 
sorry

end quotient_of_integers_l805_805835


namespace nested_sqrt_rational_count_l805_805956

theorem nested_sqrt_rational_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 2021 ∧ (∃ x : ℚ, x * x = n + x) }.to_finset.card = 44 :=
sorry

end nested_sqrt_rational_count_l805_805956


namespace area_of_circle_l805_805884

theorem area_of_circle (s : ℝ) (h₁ : s = 10) : 
  let d := s * Real.sqrt 2 in
  let r := d / 2 in
  let A := Real.pi * r^2 in
  A = 50 * Real.pi :=
by
  sorry

end area_of_circle_l805_805884


namespace symmetric_line_equation_x_axis_l805_805782

theorem symmetric_line_equation_x_axis (x y : ℝ) :
    let original_line := 3 * x + 4 * y - 5 = 0 in
    let symmetric_line := 3 * x - 4 * y + 5 = 0 in
    original_line → symmetric_line :=
by
  sorry

end symmetric_line_equation_x_axis_l805_805782


namespace complex_number_solution_l805_805223

theorem complex_number_solution (a b : ℝ) (z : ℂ) (hz : z = 1 + I) :
  (a * z + 2 * b * conj z = (a + 2 * z) ^ 2) →
  (a = -2 ∧ b = -1) ∨ (a = -4 ∧ b = 2) :=
by
  sorry

end complex_number_solution_l805_805223


namespace train_cross_platform_time_correct_l805_805107

def train_length : ℝ := 300
def signal_pole_time : ℝ := 18
def platform_length : ℝ := 500

def speed_of_train (distance time : ℝ) : ℝ := distance / time
def total_distance (train platform : ℝ) : ℝ := train + platform
def time_to_cross (distance speed : ℝ) : ℝ := distance / speed

theorem train_cross_platform_time_correct : 
  time_to_cross (total_distance train_length platform_length) 
                (speed_of_train train_length signal_pole_time) = 48 :=
by
  sorry

end train_cross_platform_time_correct_l805_805107


namespace trapezium_area_proof_l805_805176

-- Definitions given in the problem
def parallel_side_a : ℝ := 26 -- cm
def parallel_side_b : ℝ := 18 -- cm
def distance_between_sides : ℝ := 15 -- cm
def angle_theta : ℝ := 35 -- degrees

-- Conversion from degrees to radians for trigonometric functions
noncomputable def angle_theta_rad : ℝ := real.pi * angle_theta / 180

-- Area calculation based on given conditions
def trapezium_area : ℝ := (1 / 2) * (parallel_side_a + parallel_side_b) * distance_between_sides

-- Finding the length of one non-parallel side
noncomputable def non_parallel_side_1 : ℝ := distance_between_sides / real.tan angle_theta_rad

-- The proof problem statement
theorem trapezium_area_proof :
  trapezium_area = 330 ∧ non_parallel_side_1 ≈ 21.42 :=
by
  sorry

end trapezium_area_proof_l805_805176


namespace total_time_spent_on_goals_l805_805121

-- Definitions for time taken in years for each goal based on the conditions provided.
def get_in_shape : ℝ := 2
def learn_mountain_climbing : ℝ := 4
def learn_survival_skills : ℝ := 0.75 -- 9 months
def photography_course : ℝ := 0.25 -- 3 months
def downtime : ℝ := 1 / 12 -- 1 month

-- Time spent climbing the seven summits in years.
def climbing_summits : ℝ := (4 + 5 + 6 + 8 + 7 + 9 + 10) / 12 -- 49 months

def learn_diving : ℝ := 13 / 12 -- 13 months
def cave_diving : ℝ := 2

-- Time spent learning foreign languages in years.
def learn_languages : ℝ := (6 + 9 + 12) / 12 -- 27 months

-- Time spent visiting countries in years.
def visiting_countries : ℝ := (5 * 2) / 12 -- 10 months


-- Formalization of the problem statement in Lean.
theorem total_time_spent_on_goals :
  get_in_shape + learn_mountain_climbing + learn_survival_skills + photography_course + downtime + 
  climbing_summits + learn_diving + cave_diving + learn_languages + visiting_countries = 17.337 := 
by 
  -- We state the value, proof is omitted.
  sorry -- Placeholder for the actual proof.

end total_time_spent_on_goals_l805_805121


namespace count_even_sum_sequences_l805_805105

def balls : Finset ℕ := Finset.range 16 \ {0}
def is_even (n : ℕ) : Prop := n % 2 = 0
def drawing_even_sum (l : Finset ℕ) : Prop := l.card = 3 ∧ is_even (l.sum id)

theorem count_even_sum_sequences : 
  (Finset.filter drawing_even_sum (balls.powerset.filter (λ l, l.card = 3))).card = 259 := 
sorry

end count_even_sum_sequences_l805_805105


namespace total_flour_l805_805747

theorem total_flour (original_flour extra_flour : Real) (h_orig : original_flour = 7.0) (h_extra : extra_flour = 2.0) : original_flour + extra_flour = 9.0 :=
sorry

end total_flour_l805_805747


namespace cannot_form_triangle_l805_805144

theorem cannot_form_triangle (a b c : ℕ) (h : a = 2 ∧ b = 2 ∧ c = 6) : ¬(a + b > c) :=
by {
  cases h with ha hbc,
  cases hbc with hb hc,
  rw [ha, hb, hc],
  simp,
  sorry
}

end cannot_form_triangle_l805_805144


namespace relationship_among_a_b_c_l805_805964

noncomputable def a := Real.log 0.3 / Real.log 2
noncomputable def b := 2 ^ 0.3
noncomputable def c := 0.3 ^ 0.2

theorem relationship_among_a_b_c : b > c ∧ c > a := by
  have h_a : a < 0 := sorry
  have h_b : b > 1 := sorry
  have h_c : 0 < c ∧ c < 1 := sorry
  exact ⟨sorry, sorry⟩

end relationship_among_a_b_c_l805_805964


namespace max_value_trig_expression_l805_805945

-- Condition: defining the trigonometric expression f(x) = cos x + 3 sin x
def trig_expression (x : ℝ) : ℝ := Real.cos x + 3 * Real.sin x

-- Statement: proving the maximum value of the trigonometric expression
theorem max_value_trig_expression :
  ∃ x : ℝ, trig_expression x = sqrt 10 ∧ (∀ y : ℝ, trig_expression y ≤ sqrt 10) :=
sorry

end max_value_trig_expression_l805_805945


namespace a_7_value_l805_805985

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

-- Given conditions
def geometric_sequence_positive_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

def geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 0 * (1 - ((a (1 + n)) / a 0))) / (1 - (a 1 / a 0))

def S_4_eq_3S_2 (S : ℕ → ℝ) : Prop :=
S 4 = 3 * S 2

def a_3_eq_2 (a : ℕ → ℝ) : Prop :=
a 3 = 2

-- The statement to prove
theorem a_7_value (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  geometric_sequence_positive_terms a →
  geometric_sequence_sum a S →
  S_4_eq_3S_2 S →
  a_3_eq_2 a →
  a 7 = 8 :=
by
  sorry

end a_7_value_l805_805985


namespace problem_solution_l805_805577

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution_l805_805577


namespace final_amount_after_bets_l805_805873

theorem final_amount_after_bets :
  let initial_amount := 128
  let num_bets := 8
  let num_wins := 4
  let num_losses := 4
  let bonus_per_win_after_loss := 10
  let win_multiplier := 3 / 2
  let loss_multiplier := 1 / 2
  ∃ final_amount : ℝ,
    (final_amount =
      initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses) + 2 * bonus_per_win_after_loss) ∧
    final_amount = 60.5 :=
sorry

end final_amount_after_bets_l805_805873


namespace height_10_inches_from_center_l805_805124

noncomputable def height_of_parabolic_arch (h k x : ℝ) (a : ℝ) :=
  a * x^2 + h

theorem height_10_inches_from_center
  (h : ℝ) (span : ℝ) (k : ℝ) (a : ℝ)
  (hyp1 : h = 20) (hyp2 : span = 50) (hyp3 : k = -20 / 625) :
  height_of_parabolic_arch h k 10 a = 16.8 :=
by
  rw [←hyp1, ←hyp3]
  change (-4 / 125) * 10^2 + 20 = 16.8
  norm_num
  exact of_dec_eq_true (by norm_num : -4 / 125 * 100 + 20 = 16.8)

end height_10_inches_from_center_l805_805124


namespace vector_equivalence_l805_805695

-- Define the vectors a and b
noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

-- Define the operation 3a - b
noncomputable def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

-- State that for given vectors a and b, the result of the operation equals (4, 2)
theorem vector_equivalence : vector_operation vector_a vector_b = (4, 2) :=
  sorry

end vector_equivalence_l805_805695


namespace find_star_value_l805_805513

theorem find_star_value (x : ℤ) :
  45 - (28 - (37 - (15 - x))) = 58 ↔ x = 19 :=
  by
    sorry

end find_star_value_l805_805513


namespace find_angle_of_inclination_l805_805175

/- Define the function representing the curve -/
def curve (x : ℝ) (m : ℝ) : ℝ := x^3 - 2 * x + m

/- Define the derivative of the function -/
def derivative_of_curve (x : ℝ) : ℝ := 3 * x^2 - 2

/- Find the slope at x = 1 -/
def slope_at_one : ℝ := derivative_of_curve 1

/- Define the angle of inclination, given the slope -/
def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k

theorem find_angle_of_inclination (m : ℝ) : 
  angle_of_inclination (slope_at_one) = Real.pi / 4 := 
sorry

end find_angle_of_inclination_l805_805175


namespace sum_f_1_to_2020_l805_805206

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 3 / 2 then -log (7 - 2 * x) / log 2
  else if x > 3 / 2 then f (x - 3)
  else 0 -- Assuming f(0) = 0 by odd function property

axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)

theorem sum_f_1_to_2020 : ∑ n in Finset.range 2021 \ {0}, f(n) = -log 5 / log 2 :=
  by
  -- Proof would go here.
  sorry

end sum_f_1_to_2020_l805_805206


namespace count_integers_in_range_num_of_integers_l805_805271

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805271


namespace domino_coloring_l805_805731

theorem domino_coloring (m n V : ℕ) (h_even : Even (m * n)) : 
  (∀ (row : Fin m), ∃ (V ≤ n), ∀ (i : Fin n), 
    let red_count := number_of_squares_covered_by_red_dominoes row i,
    let blue_count := number_of_squares_covered_by_blue_dominoes row i in
      red_count ≤ V ∧ blue_count ≤ V) → 
  V = n := sorry

end domino_coloring_l805_805731


namespace probability_of_one_machine_maintenance_l805_805040

theorem probability_of_one_machine_maintenance :
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444 :=
by {
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  show (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444
  sorry
}

end probability_of_one_machine_maintenance_l805_805040


namespace simplify_expression_l805_805423

variable (x : ℝ)

theorem simplify_expression : 2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := 
  sorry

end simplify_expression_l805_805423


namespace count_integer_values_l805_805253

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805253


namespace packets_of_chips_l805_805769

variable (P R M : ℕ)

theorem packets_of_chips (h1: P > 0) (h2: R > 0) (h3: M > 0) :
  ((10 * M * P) / R) = (10 * M * P) / R :=
sorry

end packets_of_chips_l805_805769


namespace parallelogram_base_length_l805_805844

def base_length (area : ℝ) (height_factor : ℝ) : ℝ :=
  let b := sqrt (area / height_factor)
  b

theorem parallelogram_base_length :
  ∀ (area : ℝ) (height_factor : ℝ), area = 242 → height_factor = 2 → base_length area height_factor = 11 :=
by
  intros area height_factor h_area h_height_factor
  -- skipping the proof
  sorry

end parallelogram_base_length_l805_805844


namespace probability_within_three_units_from_origin_l805_805539

-- Define the properties of the square Q is selected from
def isInSquare (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -2 ∧ Q.1 ≤ 2 ∧ Q.2 ≥ -2 ∧ Q.2 ≤ 2

-- Define the condition of being within 3 units from the origin
def withinThreeUnits (Q: ℝ × ℝ) : Prop :=
  (Q.1)^2 + (Q.2)^2 ≤ 9

-- State the problem: Proving the probability is 1
theorem probability_within_three_units_from_origin : 
  ∀ (Q : ℝ × ℝ), isInSquare Q → withinThreeUnits Q := 
by 
  sorry

end probability_within_three_units_from_origin_l805_805539


namespace count_n_integers_l805_805281

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805281


namespace possible_values_of_a_l805_805785

theorem possible_values_of_a (x y a : ℝ)
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) :
  a ∈ {-2, -1, 0, 1, 2} :=
begin
  sorry
end

end possible_values_of_a_l805_805785


namespace percentage_of_copper_buttons_l805_805524

-- Definitions for conditions
def total_items : ℕ := 100
def pin_percentage : ℕ := 30
def button_percentage : ℕ := 100 - pin_percentage
def brass_button_percentage : ℕ := 60
def copper_button_percentage : ℕ := 100 - brass_button_percentage

-- Theorem statement proving the question
theorem percentage_of_copper_buttons (h1 : pin_percentage = 30)
  (h2 : button_percentage = total_items - pin_percentage)
  (h3 : brass_button_percentage = 60)
  (h4 : copper_button_percentage = total_items - brass_button_percentage) :
  (button_percentage * copper_button_percentage) / total_items = 28 := 
sorry

end percentage_of_copper_buttons_l805_805524


namespace polynomial_is_fourth_degree_trinomial_l805_805504

def term1 := (a : ℤ) (b : ℤ) : ℤ := a^4
def term2 := (a : ℤ) (b : ℤ) : ℤ := -2 * (a^2 * b^2)
def term3 := (a : ℤ) (b : ℤ) : ℤ := b^4
def polynomial := term1 a b + term2 a b + term3 a b

theorem polynomial_is_fourth_degree_trinomial :
  (∀ (a b : ℤ), 
    (term1 a b ≠ 0 ∧ term2 a b ≠ 0 ∧ term3 a b ≠ 0) ∧
    (degree (term1 a b) = 4) ∧
    (degree (term1 a b) = degree (term2 a b)) ∧
    (degree (term1 a b) = degree (term3 a b)) ∧
    (polynomial a b = term1 a b + term2 a b + term3 a b)) :=
begin
  sorry,
end

end polynomial_is_fourth_degree_trinomial_l805_805504


namespace pair_b_equal_l805_805141

section FunctionEquality

  -- Define the functions for each pair
  def fA (x : ℝ) := (x - 1)^0
  def gA (x : ℝ) := 1

  def fB (x : ℝ) := abs x
  def gB (x : ℝ) := real.sqrt (x^2)

  def fC (x : ℝ) := x
  def gC (x : ℝ) := (real.sqrt x)^2

  def fD (x : ℝ) := real.sqrt (x-1) * real.sqrt (x+1)
  def gD (x : ℝ) := real.sqrt (x^2 - 1)

  -- Function Equality Proposition 
  theorem pair_b_equal : (∀ x : ℝ, fB x = gB x) :=
  by
    sorry

end FunctionEquality

end pair_b_equal_l805_805141


namespace tan_frac_eq_l805_805963

theorem tan_frac_eq (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
  sorry

end tan_frac_eq_l805_805963


namespace rational_sequence_repetition_l805_805585

theorem rational_sequence_repetition (r s : ℤ) (k : ℕ → ℤ)
  (h_rat : ∃ n : ℕ, ∀ m ≥ n, 10 ^ m * r % s = ∑ i in Finset.range m, k (m - i - 1) * 10 ^ i) :
  ∃ i j : ℕ, i < j ∧ j ≤ s + 1 ∧ (10 ^ i * r - s * ∑ i in Finset.range i, k (i - i - 1) * 10 ^ (i - i - 1)) = 
              (10 ^ j * r - s * ∑ j in Finset.range j, k (j - j - 1) * 10 ^ (j - j - 1)) :=
sorry

end rational_sequence_repetition_l805_805585


namespace binary_addition_l805_805470

theorem binary_addition :
  let x := (1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) in
  let y := (1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0) in
  x + y = 141 :=
by
  sorry

end binary_addition_l805_805470


namespace find_primes_and_integers_l805_805174

theorem find_primes_and_integers (p x y : ℕ) (hp : p.prime) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (p : ℝ)) → (x = p - 1 ∧ y = p * (p - 1)) := by
  sorry

end find_primes_and_integers_l805_805174


namespace result_l805_805521

noncomputable def num_cards : ℕ := 900

def generate_numbers (n : ℕ) : list ℕ :=
(list.range n).map (λ i, i + 1)

def remove_squares (l : list ℕ) : list ℕ :=
l.filter (λ x, ¬ ∃ k : ℕ, k * k = x)

def iterations_to_remove_all_cards : ℕ :=
nat.rec_on num_cards 0 (λ n ih,
  let cards := generate_numbers n in
  let remaining := remove_squares cards in
  if remaining = [] then 0 else ih + 1)

theorem result : iterations_to_remove_all_cards = 59 :=
sorry

end result_l805_805521


namespace fraction_division_l805_805487

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805487


namespace problem_statement_l805_805519

-- Define line and plane as types
variable (Line Plane : Type)

-- Define the perpendicularity and parallelism relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLPlane : Line → Plane → Prop)
variable (perpendicularPPlane : Plane → Plane → Prop)

-- Distinctness of lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Conditions given in the problem
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Statement to be proven
theorem problem_statement :
  perpendicular a b → 
  perpendicularLPlane a α → 
  perpendicularLPlane b β → 
  perpendicularPPlane α β :=
sorry

end problem_statement_l805_805519


namespace number_of_integers_satisfying_cubed_inequality_l805_805301

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805301


namespace count_n_integers_l805_805276

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805276


namespace concurrency_of_lines_l805_805972

noncomputable theory
open_locale classical

variables {A B C C_1 C_2 A_1 A_2 B_1 B_2 A_star B_star C_star : Type*}
variables [circle A B_1 C_1] [circle A B_2 C_2] [circle B C_1 A_1]
          [circle B C_2 A_2] [circle C A_1 B_1] [circle C A_2 B_2]

def segments (p q : Type*) := ∀ a b : p, ∃ c : q, a < b

-- Conditions
variables (hC1C2 : C_1 < C_2) (hA1A2 : A_1 < A_2) (hB1B2 : B_1 < B_2)

variables (hA_star : A_star ∈ intersection_points (A B_1 C_1) (A B_2 C_2) ∧ A_star ≠ A)
variables (hB_star : B_star ∈ intersection_points (B C_1 A_1) (B C_2 A_2) ∧ B_star ≠ B)
variables (hC_star : C_star ∈ intersection_points (C A_1 B_1) (C A_2 B_2) ∧ C_star ≠ C)

-- Goal: Lines concurrency
theorem concurrency_of_lines :
   are_concurrent [line_through A A_star, line_through B B_star, line_through C C_star] :=
sorry

end concurrency_of_lines_l805_805972


namespace root_sum_eq_l805_805217

theorem root_sum_eq
  (a b : ℝ)
  (h1: ∀ x : ℝ, (log (3 : ℝ) / log (3 * x)) + (log (3 * x) / log (27 : ℝ)) = - (4 / 3))
  (h2: a = 3⁻²)
  (h3: b = 3⁻⁴) :
  a + b = 10 / 81 :=
by sorry

end root_sum_eq_l805_805217


namespace complex_magnitude_equivalence_l805_805325

theorem complex_magnitude_equivalence (z : ℂ) (h : |1 + complex.I * z| = |3 + 4 * complex.I|) : |z - complex.I| = 5 :=
sorry

end complex_magnitude_equivalence_l805_805325


namespace tangent_lines_values_l805_805041

noncomputable def tangent_points_meet_at (a b : ℝ) : Prop :=
  ∃ t : ℝ, 2 * t^3 + a * t^2 - 1 = 0

theorem tangent_lines_values (a b : ℝ) (h : b ∈ ℝ)
  (h_tangent : ∃ t₁ t₂ t₃ : ℝ, tangent_points_meet_at a b t₁ ∧ tangent_points_meet_at a b t₂ ∧ tangent_points_meet_at a b t₃) :
  a = 6 ∨ a = 4 :=
sorry

end tangent_lines_values_l805_805041


namespace zero_is_multiple_of_all_primes_l805_805501

theorem zero_is_multiple_of_all_primes :
  ∀ (x : ℕ), (∀ p : ℕ, Prime p → ∃ n : ℕ, x = n * p) ↔ x = 0 := by
sorry

end zero_is_multiple_of_all_primes_l805_805501


namespace relay_race_total_time_l805_805925

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end relay_race_total_time_l805_805925


namespace factor_expression_correct_l805_805602

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_expression_correct (a b c : ℝ) :
  factor_expression a b c = (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_correct_l805_805602


namespace triangle_incircle_touch_equality_l805_805360

theorem triangle_incircle_touch_equality 
  (A B C D E F : Point)     -- points A, B, C, D, E, F
  (O : Circle)              -- the incircle O
  [is_triangle ABC]         -- ABC forms a triangle
  (touches_BC_at_D : O.touches BC D)   -- O touches BC at D
  (diameter_DF : DDiameter DF O)  -- DF is the diameter of O
  (intersects_AF_BC_at_E : Intersects AF BC E)  -- AF intersects BC at E
  : BE = DC := 
by 
  sorry

end triangle_incircle_touch_equality_l805_805360


namespace circle_transformation_l805_805004

theorem circle_transformation (c : ℝ × ℝ) (v : ℝ × ℝ) (h_center : c = (8, -3)) (h_vector : v = (2, -5)) :
  let reflected := (c.2, c.1)
  let translated := (reflected.1 + v.1, reflected.2 + v.2)
  translated = (-1, 3) :=
by
  sorry

end circle_transformation_l805_805004


namespace calculate_f1_plus_f1_deriv_l805_805787

variable (f : ℝ → ℝ)

-- Condition 1: Tangent line at point P(1, f(1)) with equation y = -2x + 10
axiom tangent_eq : ∀ x, f x = -2 * x + 10 ↔ x = 1

-- Condition 2: The derivative of the function is f'(x)
variable (f' : ℝ → ℝ)
axiom derivative_at_1 : ∀ x, deriv f 1 = f' 1

theorem calculate_f1_plus_f1_deriv : f 1 + f' 1 = 6 := by
  have h1 : f 1 = 8 := (tangent_eq 1).mpr rfl
  have h2 : f' 1 = -2 := derivative_at_1 1
  rw [h1, h2]
  norm_num

end calculate_f1_plus_f1_deriv_l805_805787


namespace blue_balls_in_JarB_l805_805459

-- Defining the conditions
def ratio_white_blue (white blue : ℕ) : Prop := white / gcd white blue = 5 ∧ blue / gcd white blue = 3

def white_balls_in_B := 15

-- Proof statement
theorem blue_balls_in_JarB :
  ∃ (blue : ℕ), ratio_white_blue 15 blue ∧ blue = 9 :=
by {
  -- Proof outline (not required, thus just using sorry)
  sorry
}


end blue_balls_in_JarB_l805_805459


namespace ratio_of_triangle_areas_l805_805509

open Real

theorem ratio_of_triangle_areas
  (AB BC AC : ℝ)
  (B C : Point)
  (A B C : Point)
  (D : Point) 
  (angle_BC_is_60 : Angle B C = 60 * pi / 180)
  (AB_is_diameter : is_diameter AB)
  (BC_AC_are_chords : is_chord BC ∧ is_chord AC) :
  area (triangle D C B) / area (triangle D C A) = 1 / 3 :=
sorry

end ratio_of_triangle_areas_l805_805509


namespace median_of_first_twelve_positive_integers_l805_805068

def sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def median (s : List ℕ) : ℚ :=
  if h : (List.length s) % 2 = 0 then
    let k := List.length s / 2
    (s.get (k - 1) + s.get k) / 2
  else
    s.get (List.length s / 2)

theorem median_of_first_twelve_positive_integers :
  median sequence = 6.5 := 
sorry

end median_of_first_twelve_positive_integers_l805_805068


namespace servings_in_box_l805_805518

-- Define amounts
def total_cereal : ℕ := 18
def per_serving : ℕ := 2

-- Define the statement to prove
theorem servings_in_box : total_cereal / per_serving = 9 :=
by
  sorry

end servings_in_box_l805_805518


namespace same_terminal_side_l805_805899

theorem same_terminal_side (angle_set : Set ℝ) (k : ℤ) : 
  angle_set = {α | α = 310 + k * 360} → (-50 ∈ angle_set) :=
by 
  intro h
  unfold angle_set at h
  existsi (-1 : ℤ)
  simp [h]
  sorry

end same_terminal_side_l805_805899


namespace real_roots_of_quadratics_l805_805762

theorem real_roots_of_quadratics {p1 p2 q1 q2 : ℝ} (h : p1 * p2 = 2 * (q1 + q2)) :
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  have D1 := p1^2 - 4 * q1
  have D2 := p2^2 - 4 * q2
  sorry

end real_roots_of_quadratics_l805_805762


namespace racing_game_cost_l805_805722

theorem racing_game_cost (total_spent : ℝ) (basketball_game_cost : ℝ) (racing_game_cost : ℝ)
  (h1 : total_spent = 9.43) (h2 : basketball_game_cost = 5.20) :
  racing_game_cost = total_spent - basketball_game_cost :=
by
  -- Defining local variables
  let total_spent := 9.43
  let basketball_game_cost := 5.20
  let expected_racing_game_cost := 4.23
  
  -- The statement of the theorem
  have h1 : total_spent = 9.43 := rfl
  have h2 : basketball_game_cost = 5.20 := rfl
  have h3 : racing_game_cost = total_spent - basketball_game_cost := by
    rw [h1, h2]
    
  show racing_game_cost = expected_racing_game_cost
  exact h3
sorry

end racing_game_cost_l805_805722


namespace option_A_option_B_option_C_option_D_l805_805665

theorem option_A (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
sorry

theorem option_B (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2^n * a n) : a 5 = 2^10 :=
sorry

theorem option_C (S : ℕ → ℝ) (h₀ : ∀ n, S n = 3^n + 1/2) : ¬(∃ r : ℝ, ∀ n, S n = S 1 * r ^ (n - 1)) :=
sorry

theorem option_D (S : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : S 1 = 1) 
  (h₁ : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h₂ : (S 8) / 8 - (S 4) / 4 = 8) : a 6 = 21 :=
sorry

end option_A_option_B_option_C_option_D_l805_805665


namespace sum_digits_multiplication_and_addition_l805_805149

theorem sum_digits_multiplication_and_addition : 
  let nines := List.replicate 47 (9 : ℕ),
      fours := List.replicate 47 (4 : ℕ),
      num1 := nines.foldl (λ acc d => 10 * acc + d) 0,
      num2 := fours.foldl (λ acc d => 10 * acc + d) 0,
      product := num1 * num2,
      sum_digits_product := (product.digits 10).sum,
      sum_digits_100000 := (100000.digits 10).sum
  in sum_digits_product + sum_digits_100000 = 424 :=
by
  -- The proof will go here
  sorry

end sum_digits_multiplication_and_addition_l805_805149


namespace chemical_reaction_l805_805162

def reaction_balanced (koh nh4i ki nh3 h2o : ℕ) : Prop :=
  koh = nh4i ∧ nh4i = ki ∧ ki = nh3 ∧ nh3 = h2o

theorem chemical_reaction
  (KOH NH4I : ℕ)
  (h1 : KOH = 3)
  (h2 : NH4I = 3)
  (balanced : reaction_balanced KOH NH4I 3 3 3) :
  (∃ (NH3 KI H2O : ℕ),
    NH3 = 3 ∧ KI = 3 ∧ H2O = 3 ∧ 
    NH3 = NH4I - NH4I ∧
    KI = KOH - KOH ∧
    H2O = KOH - KOH) ∧
  (KOH = NH4I) := 
by sorry

end chemical_reaction_l805_805162


namespace probability_closer_to_center_l805_805125

theorem probability_closer_to_center (r_outer r_inner : ℝ) (h_outer : r_outer = 5) (h_inner : r_inner = 2) :
  let A_outer := π * r_outer^2,
      A_inner := π * r_inner^2 in
  (A_inner / A_outer) = 4 / 25 :=
by
  sorry

end probability_closer_to_center_l805_805125


namespace evaluate_f_f2_l805_805657

noncomputable def f : ℝ → ℝ :=
  λ x : ℝ, if x > 0 then log x / log (1/2) else 3^x

theorem evaluate_f_f2 : f (f 2) = 1 / 3 := 
  sorry

end evaluate_f_f2_l805_805657


namespace calc_sample_mean_calc_sample_variance_calc_corrected_sample_variance_l805_805145

open Real

def measurements : List ℝ := [92, 94, 103, 105, 106]

def sample_mean (measurements : List ℝ) : ℝ := 
  measurements.sum / measurements.length

def sample_variance (measurements : List ℝ) : ℝ := 
  let mean := sample_mean measurements
  (measurements.map (λ x => (x - mean) ^ 2)).sum / measurements.length

def corrected_sample_variance (measurements : List ℝ) : ℝ := 
  let variance := sample_variance measurements
  variance * (measurements.length / (measurements.length - 1))

theorem calc_sample_mean :
  sample_mean measurements = 100 :=
by
  sorry

theorem calc_sample_variance :
  sample_variance measurements = 34 :=
by
  sorry

theorem calc_corrected_sample_variance :
  corrected_sample_variance measurements = 42.5 :=
by
  sorry

end calc_sample_mean_calc_sample_variance_calc_corrected_sample_variance_l805_805145


namespace reflection_distance_l805_805875

noncomputable def distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem reflection_distance :
  let P := (3, 5 : ℝ)
  let P' := (-3, 5 : ℝ)
  distance P P' = 6 :=
by
  let P := (3, 5 : ℝ)
  let P' := (-3, 5 : ℝ)
  sorry

end reflection_distance_l805_805875


namespace second_intercept_x_coordinate_l805_805617

-- Define the quadratic function based on the given vertex and x-intercept properties.
noncomputable def quadratic_vertex : ℝ → ℝ := λ x => let a := 1 in a * (x - 5) * (x - 5) + 12

-- Condition: The vertex of the parabola is (5, 12).
def vertex_condition : Prop := (quadratic_vertex 5 = 12)

-- Condition: One x-intercept is (1, 0).
def intercept_condition : Prop := (quadratic_vertex 1 = 0)

-- The theorem to determine the x-coordinate of the other x-intercept.
theorem second_intercept_x_coordinate : vertex_condition ∧ intercept_condition →
  ∃ x : ℝ, quadratic_vertex x = 0 ∧ x ≠ 1 ∧ x = 9 := by
  sorry

end second_intercept_x_coordinate_l805_805617


namespace find_a_l805_805440

-- Conditions in the problem
variable (a b c : ℝ)
axiom vertex_condition : ∃ a b c : ℝ, ∀ x : ℝ, (a (x - 2)^2 + (b - 4a) * (x -2) + (c + b^2 - 4*a): Real) (2) = 0
axiom point_condition : a * (0 - 2)^2 = -50

-- Stating the Result
theorem find_a (a b c : ℝ) (vertex_condition : ∃ a b c : ℝ, ∀ x : ℝ, (a (x - 2)^2 + (b - 4a) * (x -2) + (c + b^2 - 4*a): Real) (2) = 0) (point_condition : a * (0 - 2)^2 = -50 : ℝ) : a = -12.5 :=
sorry

end find_a_l805_805440


namespace first_investment_percentage_l805_805338

variable (P : ℝ)
variable (x : ℝ := 1400)  -- investment amount in the first investment
variable (y : ℝ := 600)   -- investment amount at 8 percent
variable (income_difference : ℝ := 92)
variable (total_investment : ℝ := 2000)
variable (rate_8_percent : ℝ := 0.08)
variable (exceed_by : ℝ := 92)

theorem first_investment_percentage :
  P * x - rate_8_percent * y = exceed_by →
  total_investment = x + y →
  P = 0.10 :=
by
  -- Solution steps can be filled here if needed
  sorry

end first_investment_percentage_l805_805338


namespace calculate_problem_l805_805853

theorem calculate_problem : 
  (0.45 * 2.5 + 4.5 * 0.65 + 0.45 = 4.5) ∧ 
  (let a := 1, d := 2, l := 49 in 
    let n := (l - a) / d + 1 in 
    let S := n / 2 * (a + l) in 
    n = 25 ∧ S = 625) :=
by
  sorry

end calculate_problem_l805_805853


namespace number_of_valid_mappings_l805_805215

def A := {1, 2}
def B := {0, 1, 2, 3, 4}

def valid_mappings (f : A → B) : Prop :=
  f 1 + f 2 = 4

theorem number_of_valid_mappings : 
  (∃ f : A → B, valid_mappings f) → 5 := 
sorry

end number_of_valid_mappings_l805_805215


namespace cube_root_and_power_calc_l805_805911

theorem cube_root_and_power_calc : real.cbrt 8 + (-2)^0 = 3 :=
by
  -- Proof to be inserted here
  sorry

end cube_root_and_power_calc_l805_805911


namespace determine_a_bi_l805_805427

-- Given definitions and conditions
def is_positive_integer (n : ℤ) : Prop := n > 0

def satisfies_conditions (a b : ℤ) : Prop :=
  (is_positive_integer a) ∧
  (is_positive_integer b) ∧
  ((a + b * I)^3 = 2 + 11 * I)

-- Problem statement to be proven
theorem determine_a_bi : ∃ (a b : ℤ), satisfies_conditions a b ∧ (a + b * I = 2 + I) :=
by
  sorry

end determine_a_bi_l805_805427


namespace sequence_sum_50_l805_805632

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -2 ∧ a 2 = 2 ∧ ∀ n, a (n + 2) - a n = 1 + (-1)^n

def S (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in Finset.range n, a (i + 1)

theorem sequence_sum_50 (a : ℕ → ℤ) (h : sequence a) : S a 50 = 600 :=
by
  sorry

end sequence_sum_50_l805_805632


namespace common_roots_cubic_polynomials_l805_805181

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end common_roots_cubic_polynomials_l805_805181


namespace opaque_segments_less_than_one_plus_sqrt_three_l805_805129

noncomputable def is_opaque (segments : Set (Set (ℝ × ℝ))) : Prop :=
  ∀ (line : ℝ × ℝ → bool), ∃ segment ∈ segments, 
  intersects line segment 

def length_of_segments (segments : Set (Set (ℝ × ℝ))) : ℝ :=
  -- The specific implementation to calculate the length of segments is abstract and unnecessary for this statement

theorem opaque_segments_less_than_one_plus_sqrt_three : ∃ (segments : Set (Set (ℝ × ℝ))),
  is_opaque segments ∧ length_of_segments segments < 1 + Real.sqrt 3 :=
by
  -- construction of segments and proof goes here, skipping as per instructions
  sorry

end opaque_segments_less_than_one_plus_sqrt_three_l805_805129


namespace equation_of_line_l805_805009

theorem equation_of_line (θ : ℝ) (b : ℝ) :
  θ = 135 ∧ b = -1 → (∀ x y : ℝ, x + y + 1 = 0) :=
by
  sorry

end equation_of_line_l805_805009


namespace smallest_single_discount_more_advantageous_l805_805190

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  (∀ n, 0 < n -> (1 - (n:ℝ)/100) < 0.64 ∧ (1 - (n:ℝ)/100) < 0.658503 ∧ (1 - (n:ℝ)/100) < 0.63) → 
  n = 38 := 
sorry

end smallest_single_discount_more_advantageous_l805_805190


namespace student_exam_score_l805_805132

theorem student_exam_score
  (points_2_hours : ℝ)
  (points_5_hours : ℝ) :
  points_2_hours = 90 →
  ∀ (h : ℝ) (score_at_time : ℝ → ℝ),
  (∀ t, score_at_time t = t * score_at_time 1) →
  score_at_time 2 = points_2_hours →
  (∀ t, t > 3 → score_at_time t = score_at_time 3 + (t - 3) * 0.1 * score_at_time 3) →
  points_5_hours = 162 :=
by
  intros h pts1 pts3 h1 h2 h3 h4
  -- Assuming "score at 1 hour" to logical deduction should be "how score relates to time"
  -- and using given condition should let us express this relation
  -- hence setting certain equations should derive correct assumption
  sorry

end student_exam_score_l805_805132


namespace extreme_values_of_f_exists_minimum_m_find_minimum_m_l805_805229

noncomputable def f (x : ℝ) : ℝ := real.log x - 2 * x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + (m - 3) * x - 1

theorem extreme_values_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -real.log 2 - 1 :=
sorry

theorem exists_minimum_m (x : ℝ) (m : ℤ) :
  (0 < x) → 
  f x ≤ g ↑m x := 
sorry

theorem find_minimum_m : 
  ∃ (m : ℤ), ∀ x ∈ set.Ioi (0 : ℝ), 
  f x ≤ g m x ∧ (m = 2 ∨ m > 2) := 
sorry

end extreme_values_of_f_exists_minimum_m_find_minimum_m_l805_805229


namespace distinct_ordered_pairs_count_l805_805651

theorem distinct_ordered_pairs_count :
  (∃ (a b : ℕ), 0 < a ∧ a + b = 50 ∧ a % 2 = 0) ∧ cardinal.mk {ab : ℕ × ℕ // 0 < ab.1 ∧ ab.1 + ab.2 = 50 ∧ ab.1 % 2 = 0} = 24 :=
  sorry

end distinct_ordered_pairs_count_l805_805651


namespace problem_statement_l805_805087

-- Define what it means to be a quadratic equation
def is_quadratic (eqn : String) : Prop :=
  -- In the context of this solution, we'll define a quadratic equation as one
  -- that fits the form ax^2 + bx + c = 0 where a, b, c are constants and a ≠ 0.
  eqn = "x^2 - 2 = 0"

-- We need to formulate a theorem that checks the validity of which equation is quadratic.
theorem problem_statement :
  is_quadratic "x^2 - 2 = 0" :=
sorry

end problem_statement_l805_805087


namespace sin_2alpha_minus_pi_over_6_beta_is_pi_over_3_l805_805104

variables (α β : ℝ)

-- Conditions
axiom cos_alpha : cos α = 1 / 7
axiom cos_alpha_minus_beta : cos (α - β) = 13 / 14
axiom alpha_range : 0 < α ∧ α < π
axiom beta_range : 0 < β ∧ β < α

-- 1. Prove that sin (2α - π/6) = 71 / 98
theorem sin_2alpha_minus_pi_over_6 : sin (2 * α - π / 6) = 71 / 98 :=
by sorry 

-- 2. Prove that β = π / 3
theorem beta_is_pi_over_3 : β = π / 3 :=
by sorry

end sin_2alpha_minus_pi_over_6_beta_is_pi_over_3_l805_805104


namespace regular_ngon_multiplicative_l805_805847

def is_multiplicative_triangle (a b c : ℝ) : Prop :=
  a * b = c ∨ b * c = a ∨ c * a = b

def triangle_multiplicative (n : ℕ) (A B : Fin n) : Prop :=
  ∀ (i j : Fin n), 
  i ≠ j → 
  is_multiplicative_triangle 1 (distance i j) (distance j A)

theorem regular_ngon_multiplicative (n : ℕ) (h : n ≥ 3) :
  let vertices := List.range n
  ∀ (A : ℕ) (H : A ∈ vertices), 
  let diagonals_from_A := (vertices.filter (λ v, v ≠ A)).map (λ v, (A, v)),
  ∀ (x y : ℕ), (x, y) ∈ (diagonals_from_A.product diagonals_from_A) 
  → triangle_multiplicative n x y :=
sorry

end regular_ngon_multiplicative_l805_805847


namespace length_of_second_train_is_correct_l805_805522

noncomputable def convert_kmph_to_mps (speed_kmph: ℕ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_lengths_and_time
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_to_cross : ℝ)
  (length_second_train : ℝ) : Prop :=
  let speed_first_train_mps := convert_kmph_to_mps speed_first_train_kmph
  let speed_second_train_mps := convert_kmph_to_mps speed_second_train_kmph
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_to_cross
  total_distance = length_first_train + length_second_train

theorem length_of_second_train_is_correct :
  train_lengths_and_time 260 120 80 9 239.95 :=
by
  sorry

end length_of_second_train_is_correct_l805_805522


namespace circles_intersect_l805_805447

def distance_between_centers (c1 c2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)

def radius_of_circle (coeff : ℝ) : ℝ :=
  real.sqrt coeff

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := radius_of_circle 1
def radius2 : ℝ := radius_of_circle 5

-- The proof statement
theorem circles_intersect : 
  let d := distance_between_centers center1 center2 in
  radius1 < radius2 → radius1 + radius2 > d ∧ d > radius2 - radius1 :=
by {
  let d := distance_between_centers center1 center2,
  let radius1 := radius_of_circle 1,
  let radius2 := radius_of_circle 5,
  sorry
}

end circles_intersect_l805_805447


namespace monic_quartic_polynomial_exists_l805_805935

theorem monic_quartic_polynomial_exists :
  ∃ p : polynomial ℚ, p.monic ∧ p.eval (3 + real.sqrt 5) = 0 ∧ p.eval (3 - real.sqrt 5) = 0 ∧ 
  p.eval (2 - real.sqrt 7) = 0 ∧ p.eval (2 + real.sqrt 7) = 0 ∧ 
  p = polynomial.monic_quotient (polynomial.X^4 - 10 * polynomial.X^3 + 25 * polynomial.X^2 + 2 * polynomial.X - 12) :=
sorry

end monic_quartic_polynomial_exists_l805_805935


namespace any_nat_as_fraction_form_l805_805086

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l805_805086


namespace compute_expression_value_l805_805583

noncomputable def expression := 3 ^ (Real.log 4 / Real.log 3) - 27 ^ (2 / 3) - Real.log 0.01 / Real.log 10 + Real.log (Real.exp 3)

theorem compute_expression_value :
  expression = 0 := 
by
  sorry

end compute_expression_value_l805_805583


namespace number_of_intersection_points_l805_805760

-- Define the prerequisites: two parallel lines and a circle
structure ParallelLinesAndCircle where
  l1 : Line
  l2 : Line
  circle : Circle
  parallel : l1 ∥ l2

-- The mathematical statement: Prove that the possible number of intersection points can be 4, 2, 1, or 0.
theorem number_of_intersection_points (config : ParallelLinesAndCircle) :
  ∃ n : ℕ, n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 := sorry

end number_of_intersection_points_l805_805760


namespace num_integer_solutions_l805_805290

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805290


namespace tangents_intersect_in_planes_l805_805619

-- Variables Definitions
variables {O A B M : Type} {r a b x : ℝ}

-- Conditions
def is_center (O : Type) : Prop := ∃ (r : ℝ), r > 0
def points_on_sphere (A B : Type) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0
def tangent_lengths (M : Type) : Prop := ∃ (x : ℝ), x > 0

-- Theorem statement
theorem tangents_intersect_in_planes 
  (O : Type) [hc : is_center O] 
  (A B : Type) [hp : points_on_sphere A B]
  (M : Type) [ht : tangent_lengths M] :
  ∀ (M : Type), (M ≠ A ∧ M ≠ B) → 
    ∃ (α β γ : ℝ), 
      (α * (a + x) ^ 2 + β * (b + x) ^ 2 + γ * (r^2 + x^2)) = 0 ∧
      (α + β + γ = 0) ∧
      (β * (a + x) ^ 2 + α * (b + x) ^ 2 - (α + β) * (r^2 + x^2) = 0) := sorry

end tangents_intersect_in_planes_l805_805619


namespace indicator_trigger_probability_l805_805763

theorem indicator_trigger_probability (T t : ℝ) (hT : 0 < T) (ht : 0 < t ∧ t < T) :
  let P := (t * (2 * T - t)) / (T * T)
  in P = t * (2 * T - t) / (T * T) :=
by
  sorry

end indicator_trigger_probability_l805_805763


namespace count_n_integers_l805_805278

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805278


namespace min_value_expression_l805_805827

theorem min_value_expression (x y : ℝ) : (x^2 + y^2 - 6 * x + 4 * y + 18) ≥ 5 :=
sorry

end min_value_expression_l805_805827


namespace ellipse_equation_hyperbola_equation_l805_805103

-- Ellipse Problem
theorem ellipse_equation (hx : 3 * 9 * b^2 = 13) (hy : 13 / 9 = y^2) (hmajor_minor : a = 3 * b) :
  (x : ℝ) * (x : ℝ) / 13 + (y : ℝ) * (y : ℝ) / (13 / 9) = 1 := 
sorry

-- Hyperbola Problem
theorem hyperbola_equation (hx : 5 * λ = 10) (hy : 3 * λ = 6) (hfocal_length : 2 * c = 8) :
  (x : ℝ) * (x : ℝ) / 10 - (y : ℝ) * (y : ℝ) / 6 = 1 :=
sorry

end ellipse_equation_hyperbola_equation_l805_805103


namespace area_increase_is_correct_l805_805561

noncomputable def increase_in_area : ℝ :=
  let original_length : ℝ := 40
  let original_width : ℝ := 20
  let original_area := original_length * original_width
  let perimeter := 2 * (original_length + original_width)
  let side_length := perimeter / 3
  let triangle_area := (Math.sqrt 3 / 4) * side_length^2
  triangle_area - original_area

theorem area_increase_is_correct :
  increase_in_area = 400 * (Math.sqrt 3 - 2) :=
by
  sorry

end area_increase_is_correct_l805_805561


namespace solve_farm_l805_805343

def farm_problem (P H L T : ℕ) : Prop :=
  L = 4 * P + 2 * H ∧
  T = P + H ∧
  L = 3 * T + 36 →
  P = H + 36

-- Theorem statement
theorem solve_farm : ∃ P H L T : ℕ, farm_problem P H L T :=
by sorry

end solve_farm_l805_805343


namespace triangle_ABC_angles_l805_805359

theorem triangle_ABC_angles (A B C D M : Point)
  (h1: foot_of_altitude D A B C)
  (h2: midpoint M B C)
  (h3: ∠BAD = ∠DAM ∧ ∠DAM = ∠MAC) :
  is_triangle_with_angles A B C 90 60 30 :=
sorry

end triangle_ABC_angles_l805_805359


namespace perpendicular_vectors_l805_805672

def vec_perp (a b : ℝ × ℝ) : Prop := (2 * a.1 - b.1) * b.1 + (2 * a.2 - b.2) * b.2 = 0

theorem perpendicular_vectors :
  ∀ (x : ℝ), vec_perp (2, 1) (3, x) → (x = -1 ∨ x = 3) :=
begin
  intros x h,
  sorry
end

end perpendicular_vectors_l805_805672


namespace median_of_first_twelve_positive_integers_l805_805056

theorem median_of_first_twelve_positive_integers :
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth (5)).getD 0 + (lst.nth (6)).getD 0 / 2 = 6.5 :=
by
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let median := ((lst.nth (5)).getD 0 + (lst.nth (6)).getD 0) / 2
  show median = 6.5
  sorry

end median_of_first_twelve_positive_integers_l805_805056


namespace janet_fertilizer_spread_per_day_l805_805720

-- Given conditions
def one_horse_fertilizer_per_day := 5
def num_horses := 80
def fertilizer_needed_per_acre := 400
def total_acres := 20
def days_needed := 25

-- Question to prove
theorem janet_fertilizer_spread_per_day :
  (total_acres / days_needed) = 0.8 := by
  sorry

end janet_fertilizer_spread_per_day_l805_805720


namespace part1_part2_l805_805670

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def cos_theta (u v : ℝ × ℝ) : ℝ := dot_product u v / (magnitude u * magnitude v)
def vector_subtract (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def scalar_multiply (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

theorem part1 : cos_theta a b = real.sqrt 65 / 65 := by
  sorry

theorem part2 : magnitude (vector_subtract (scalar_multiply 2 a) b) = real.sqrt 53 := by
  sorry

end part1_part2_l805_805670


namespace find_b_value_l805_805235

-- Define the conditions
variables {a b c : ℝ} -- Define the variables a, b, c as real numbers
axiom triangle_sides (h₁ : a + b > c) (h₂ : a > b + c) -- Define the properties of the triangle sides
axiom absolute_value_property : |a + b - c| + |a - b - c| = 10 -- Given condition

-- Define the theorem that we need to prove
theorem find_b_value : b = 5 := 
sorry -- Proof to be filled later

end find_b_value_l805_805235


namespace symmetric_point_correct_l805_805779

noncomputable def symmetric_point (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let (x, y) := P in (-y, -x)

theorem symmetric_point_correct : 
  symmetric_point (2, 5) (λ (P : ℝ × ℝ), P.1 + P.2 = 0) = (-5, -2) := 
by 
  sorry

end symmetric_point_correct_l805_805779


namespace hyperbola_ecc_l805_805557

noncomputable def sqrt6_div_2 := real.mk (λ _, (sqrt 6 / 2)) sorry

theorem hyperbola_ecc (a b : ℝ) (h : a = 2 * b)
    (hy : ∀ x y : ℝ, (x^2 / a) - (y^2 / b) = 1 -> x ≠ 0 ∧ y ≠ 0) :
    ∀ e : ℝ, e = sqrt6_div_2 -> 
    (1 + b^2 / a^2 = (sqrt e) ^ 2) := sorry

end hyperbola_ecc_l805_805557


namespace kilogram_to_gram_conversion_l805_805106

theorem kilogram_to_gram_conversion (kg_to_g : ℕ) (h1 : kg_to_g = 1000) :
  (5 * kg_to_g = 5000) ∧ (8000 / kg_to_g = 8) :=
by
  rw [h1]
  split
  · norm_num
  · norm_num

end kilogram_to_gram_conversion_l805_805106


namespace minimum_area_of_circle_l805_805357

-- Given conditions
variables (A B : ℝ × ℝ) (x y : ℝ)

-- A lies on the x-axis and B lies on the y-axis
def A_on_x_axis := A.2 = 0
def B_on_y_axis := B.1 = 0

-- Diameter and tangent condition
def AB_is_diameter (C : metric.ball ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (real.dist A B / 2)) :=
  C.radius = real.dist A B / 2

def tangent_line := λ x y, x + y - 4 = 0

-- Midpoint of AB
def midpoint_AB := (A.1 + B.1) / 2, (A.2 + B.2) / 2

-- Distance from O(0,0) to the line x + y - 4 = 0
def perpendicular_distance := (4 / real.sqrt 2)

-- Radius of the circle when distance is minimized
def min_radius := real.sqrt 2

-- Minimum area of the circle
def min_area := real.pi * min_radius^2

-- Theorem statement
theorem minimum_area_of_circle :
  A_on_x_axis A → B_on_y_axis B → tangent_line A.1 A.2 → tangent_line B.1 B.2 →
  min_area = 2 * real.pi :=
by
  intros
  sorry

end minimum_area_of_circle_l805_805357


namespace constant_expression_l805_805758

variable {x y m n : ℝ}

theorem constant_expression (hx : x^2 = 25) (hy : ∀ y : ℝ, (x + y) * (x - 2 * y) - m * y * (n * x - y) = 25) :
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end constant_expression_l805_805758


namespace cone_from_sector_radius_l805_805503

theorem cone_from_sector_radius (r : ℝ) (slant_height : ℝ) : 
  (r = 9) ∧ (slant_height = 12) ↔ 
  (∃ (sector_angle : ℝ) (sector_radius : ℝ), 
    sector_angle = 270 ∧ sector_radius = 12 ∧ 
    slant_height = sector_radius ∧ 
    (2 * π * r = sector_angle / 360 * 2 * π * sector_radius)) :=
by
  sorry

end cone_from_sector_radius_l805_805503


namespace probability_A_in_Omega_l805_805236

noncomputable def region_omega (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < 2

noncomputable def region_A (p : ℝ × ℝ) : Prop :=
  p.1 < 1 ∧ p.2 < 1 ∧ p.1 + p.2 > 1

theorem probability_A_in_Omega : 
  (set_integral measure_space.volume (λ p : ℝ × ℝ, if region_A p then 1 else 0) {p | region_omega p}) /
  (set_integral measure_space.volume (λ p : ℝ × ℝ, 1) {p | region_omega p}) = 
  1 / 4 :=
by
  sorry

end probability_A_in_Omega_l805_805236


namespace valid_combinations_l805_805746

/--
Marty wants to paint a box. 
He can use one of the following colors: blue, green, yellow, black, white.
He can use one of the following painting methods: brush, roller, sponge, spray.
However, if he chooses white paint, he cannot use a sponge.
Prove that the total number of valid combinations of color and painting method is 19.
-/
theorem valid_combinations : 
  let colors := 5
  let methods := 4
  let white_with_sponge := 1
  combinations := colors * methods - white_with_sponge
  combinations = 19 :=
by
  sorry

end valid_combinations_l805_805746


namespace parabola_distance_focus_directrix_l805_805511

theorem parabola_distance_focus_directrix (x y p : ℝ) 
    (h : x^2 = 2 * p * y) : 
    real.sqrt(x^2 + (y - p / 2)^2) = abs(y + p / 2) := 
sorry

end parabola_distance_focus_directrix_l805_805511


namespace three_by_three_grid_prob_odd_sum_l805_805700

noncomputable def prob_odd_sum_in_rows : ℚ :=
  let total_ways := (9!).toNat
  let valid_ways := 40^3
  (valid_ways : ℚ) / total_ways

theorem three_by_three_grid_prob_odd_sum :
  prob_odd_sum_in_rows = 4 / 227 :=
by
  -- stmt: This part computes the exact values and verifies that the ratio is correct
  sorry

end three_by_three_grid_prob_odd_sum_l805_805700


namespace dice_probability_sum_15_l805_805614

theorem dice_probability_sum_15 :
  let dice := fin 5 → fin 6
  let uniform_distribution (x : dice) := 1 / (6^5)
  Pr[∑ i, (dice i + 1) = 15] = 95 / 7776 := 
sorry

end dice_probability_sum_15_l805_805614


namespace fraction_division_l805_805482

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l805_805482


namespace geometric_formula_sum_b_sequence_l805_805233

-- Define the geometric sequence {a_n}
def geometric_sequence (a : ℕ → ℕ) :=
  ∃ q : ℕ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the conditions given
def condition1 (a : ℕ → ℕ) : Prop :=
  2 * a 3 + a 5 = 3 * a 4

def condition2 (a : ℕ → ℕ) : Prop :=
  a 3 + 2 = (a 2 + a 4) / 2

-- Definition of b_n sequence
def b (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  a n / ((a n - 1) * (a (n + 1) - 1))

-- Sum of first n terms of b_n sequence
def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range n, b a k

-- Theorem to prove the general formula for the sequence {a_n}
theorem geometric_formula (a : ℕ → ℕ) 
  (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n, a n = 2 ^ n := 
sorry

-- Theorem to prove the sum of the first n terms of the {b_n} sequence
theorem sum_b_sequence (a : ℕ → ℕ) 
  (h : ∀ n, a n = 2 ^ n) :
  ∀ n, S a n = 1 - 1 / (2 ^ (n + 1) - 1) := 
sorry

end geometric_formula_sum_b_sequence_l805_805233


namespace range_of_a_l805_805999

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^2 - |a| * x
noncomputable def g (x : ℝ) : ℝ := 2 / (1 - x^2)

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f a x > g x) ↔ a < -real.sqrt 2 ∨ a > real.sqrt 2 := 
sorry

end range_of_a_l805_805999


namespace num_planes_four_non_coplanar_l805_805593

-- Define the input points as a set in space.
variables {P : Type*} [metric_space P]

/-- Non-coplanar points are given in a space -/
def non_coplanar (s : set P) : Prop :=
∃ (a b c d : P), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ¬(affine_independent ℝ ![a,b,c,d])

-- The main theorem statement
theorem num_planes_four_non_coplanar (s : set P) (h : finite s) (h_s : s.card = 4) (h_non_coplanar : non_coplanar s) :
  (∃ (n : ℕ), n = 1 ∨ n = 4) :=
sorry

end num_planes_four_non_coplanar_l805_805593


namespace independence_test_confidence_l805_805852

theorem independence_test_confidence (H0 : ¬ related X Y) 
                                    (prob : P(K^2 ≥ 6.635) ≈ 0.01) : 
    confidence_related : confidence > 0.99 :=
sorry

end independence_test_confidence_l805_805852


namespace store_owner_marked_price_l805_805887

theorem store_owner_marked_price (L P S x : ℝ) (hL : L = 100)
 (hP : P = 100 - 0.3 * 100) (hx : x = 121.3) :
 (x / L) * 100 = 121.3 := by
suffices h1 : (P = 70), from sorry,
suffices h2 : (S = 0.75 * x), from sorry,
suffices h3 : (S - P = 0.3 * P), from sorry,
sorry
 
end store_owner_marked_price_l805_805887


namespace expression_value_l805_805582

open Real

theorem expression_value :
  3 + sqrt 3 + 1 / (3 + sqrt 3) + 1 / (sqrt 3 - 3) = 3 + 2 * sqrt 3 / 3 := 
sorry

end expression_value_l805_805582


namespace smallest_integer_sum_to_2020_l805_805612

theorem smallest_integer_sum_to_2020 :
  ∃ B : ℤ, (∃ (n : ℤ), (B * (B + 1) / 2) + ((n * (n + 1)) / 2) = 2020) ∧ (∀ C : ℤ, (∃ (m : ℤ), (C * (C + 1) / 2) + ((m * (m + 1)) / 2) = 2020) → B ≤ C) ∧ B = -2019 :=
by
  sorry

end smallest_integer_sum_to_2020_l805_805612


namespace number_of_integers_satisfying_cubed_inequality_l805_805302

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805302


namespace median_first_twelve_pos_integers_l805_805075

theorem median_first_twelve_pos_integers : 
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = 6.5 := by
  sorry

end median_first_twelve_pos_integers_l805_805075


namespace derivative_correct_1_derivative_correct_2_l805_805944

noncomputable def derivative_1 (x : ℝ) : ℝ := 2 / x

noncomputable def function_1 (x : ℝ) : ℝ := 2 * Real.log x

theorem derivative_correct_1 (x : ℝ) (hx : 0 < x) :
  derivative (λ x, 2 * Real.log x) x = 2 / x := by
  sorry

noncomputable def derivative_2 (x : ℝ) : ℝ := (Real.exp x * x - Real.exp x) / (x * x)

noncomputable def function_2 (x : ℝ) : ℝ := Real.exp x / x

theorem derivative_correct_2 (x : ℝ) (hx : x ≠ 0) :
  derivative (λ x, Real.exp x / x) x = (Real.exp x * x - Real.exp x) / (x * x) := by
  sorry

end derivative_correct_1_derivative_correct_2_l805_805944


namespace correct_propositions_l805_805197

/-- Definitions of lines and planes --/
variables {l m n : Line} {α β γ : Plane}

/-- Proposition ① --/
lemma prop1 (h_m_l : m ∥ l) (h_m_alpha : m ⊥ α) : l ⊥ α := sorry

/-- Proposition ② --/
lemma prop2 (h_m_l : m ∥ l) (h_m_alpha : m ∥ α) : l ∥ α := sorry

/-- Proposition ③ --/
lemma prop3 (h_alpha_beta : α ∩ β = l) (h_beta_gamma : β ∩ γ = m) (h_gamma_alpha : γ ∩ α = n) :
  l ∥ m ∥ n := sorry

/-- Proposition ④ --/
lemma prop4 (h_alpha_gamma : α ∩ γ = m) (h_beta_gamma : β ∩ γ = l) (h_alpha_beta : α ∥ β) :
  m ∥ l := sorry

/-- The main theorem which states that only propositions ① and ④ are true, making the correct choice C --/
theorem correct_propositions :
  (prop1 ∧ prop4) ∧ ¬prop2 ∧ ¬prop3 := sorry

end correct_propositions_l805_805197


namespace median_first_twelve_pos_integers_l805_805076

theorem median_first_twelve_pos_integers : 
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = 6.5 := by
  sorry

end median_first_twelve_pos_integers_l805_805076


namespace solve_eq_l805_805950

theorem solve_eq (x y : ℝ) (h1 : sqrt (8 * x) / sqrt (4 * (y - 2)) = 3) :
  x = (9 * y - 18) / 2 :=
sorry

end solve_eq_l805_805950


namespace sum_of_inscribed_angles_l805_805113

-- Define the circle and its division into arcs.
def circle_division (O : Type) (total_arcs : ℕ) := total_arcs = 16

-- Define the inscribed angles x and y.
def inscribed_angle (O : Type) (arc_subtended : ℕ) := arc_subtended

-- Define the conditions for angles x and y subtending 3 and 5 arcs respectively.
def angle_x := inscribed_angle ℝ 3
def angle_y := inscribed_angle ℝ 5

-- Theorem stating the sum of the inscribed angles x and y.
theorem sum_of_inscribed_angles 
  (O : Type)
  (total_arcs : ℕ)
  (h1 : circle_division O total_arcs)
  (h2 : inscribed_angle O angle_x = 3)
  (h3 : inscribed_angle O angle_y = 5) :
  33.75 + 56.25 = 90 :=
by
  sorry

end sum_of_inscribed_angles_l805_805113


namespace train_pass_time_l805_805889

open Real

def length_of_train : ℝ := 280
def speed_kmh : ℝ := 63
def speed_ms : ℝ := speed_kmh * (1000 / 3600)
def time_to_pass_tree (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_pass_time : time_to_pass_tree length_of_train speed_ms = 16 := by
  -- the proof would go here
  sorry

end train_pass_time_l805_805889


namespace expand_product_correct_l805_805600

noncomputable def expand_product (x : ℝ) : ℝ :=
  (3 / 7) * (7 / x^2 + 6 * x^3 - 2)

theorem expand_product_correct (x : ℝ) (h : x ≠ 0) :
  expand_product x = (3 / x^2) + (18 * x^3 / 7) - (6 / 7) := by
  unfold expand_product
  -- The proof will go here
  sorry

end expand_product_correct_l805_805600


namespace non_congruent_squares_121_l805_805679

def lattice_grid_6x6 : set (ℤ × ℤ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }

-- Define the condition of a square on the grid
def is_square (s : set (ℤ × ℤ)) : Prop :=
  s.size = 4 ∧ ∃ (p q r : (ℤ × ℤ)), s = {p, q, r, (p+q-r)}

noncomputable def total_non_congruent_squares : ℕ :=
  calc
    -- 1x1, 2x2, ..., 5x5 squares
    25 + 16 + 9 + 4 + 1 + 
    -- Diagonal squares (sqrt(2), 2sqrt(2))
    50 + 16

theorem non_congruent_squares_121 : total_non_congruent_squares = 121 := by sorry

end non_congruent_squares_121_l805_805679


namespace square_area_in_circle_l805_805544

theorem square_area_in_circle (r t d S : ℝ)
  (h1 : π * r^2 = 16 * π) 
  (h2 : d = 2 * r) 
  (h3 : t * sqrt 2 = d) 
  : S = t^2 → S = 32 := 
by
  sorry

end square_area_in_circle_l805_805544


namespace matrix_multiplication_zero_l805_805153

theorem matrix_multiplication_zero (d e f : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]
  ]
  let B : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![d^2, d*e, d*f],
    ![d*e, e^2, e*f],
    ![d*f, e*f, f^2]
  ]
  A ⬝ B = 0 :=
by {
  sorry
}

end matrix_multiplication_zero_l805_805153


namespace AF_plus_BE_eq_DF_l805_805730

-- Define the conditions
variable (a : ℝ) -- Side length of the square

-- Coordinates of vertices of the square ABCD
def A := (0, 0)
def B := (a, 0)
def C := (a, a)
def D := (0, a)

-- Coordinate of midpoint E of the side BC
def E := (a, a / 2)

-- Coordinate of point F on AB such that FE ⊥ DE
def F := (3 * a / 4, 0)

-- Distances
def AF := (3 * a / 4 : ℝ)
def BE := (a / 2 : ℝ)
def DF := (5 * a / 4 : ℝ)

-- The proof statement
theorem AF_plus_BE_eq_DF : 
  AF + BE = DF := by
  sorry

end AF_plus_BE_eq_DF_l805_805730


namespace triangle_XYZ_area_l805_805711

theorem triangle_XYZ_area (DE DF : ℝ) (area : ℝ) :
  DE = 13 → DF = 5 → 
  area = 1 / 2 * (84 / 17) * (60 / 17) →
  area = 2520 / 289 :=
by
  intros h1 h2 h3
  rw [h3]
  simp
  norm_num
  sorry

end triangle_XYZ_area_l805_805711


namespace wall_height_l805_805528

noncomputable def brickVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def wallVolume (L W H : ℝ) : ℝ :=
  L * W * H

theorem wall_height (bricks_needed : ℝ) (brick_length_cm brick_width_cm brick_height_cm wall_length wall_width wall_height : ℝ)
  (H1 : bricks_needed = 4094.3396226415093)
  (H2 : brick_length_cm = 20)
  (H3 : brick_width_cm = 13.25)
  (H4 : brick_height_cm = 8)
  (H5 : wall_length = 7)
  (H6 : wall_width = 8)
  (H7 : brickVolume (brick_length_cm / 100) (brick_width_cm / 100) (brick_height_cm / 100) * bricks_needed = wallVolume wall_length wall_width wall_height) :
  wall_height = 0.155 :=
by
  sorry

end wall_height_l805_805528


namespace coin_grid_probability_l805_805885

/--
A square grid is given where the edge length of each smallest square is 6 cm.
A hard coin with a diameter of 2 cm is thrown onto this grid.
Prove that the probability that the coin, after landing, will have a common point with the grid lines is 5/9.
-/
theorem coin_grid_probability :
  let square_edge_cm := 6
  let coin_diameter_cm := 2
  let coin_radius_cm := coin_diameter_cm / 2
  let grid_center_edge_cm := square_edge_cm - coin_diameter_cm
  let non_intersect_area_ratio := (grid_center_edge_cm ^ 2) / (square_edge_cm ^ 2)
  1 - non_intersect_area_ratio = 5 / 9 :=
by
  sorry

end coin_grid_probability_l805_805885


namespace pool_fill_time_l805_805540

noncomputable def fill_time (rates : List ℕ) : ℝ :=
  let invSum := (rates.map (λ r, 1 / r)).sum
  1 / invSum

theorem pool_fill_time :
  let rates := [2, 4, 94, ..., 496] in  -- Note: Fill in correct intermediate rates
  fill_time rates = 2 :=
by
  -- Intermediate step calculations can be checked here if desired
  let rates := [2, 4, ..., 496]  -- Note: Fill in correct intermediate rates
  sorry

end pool_fill_time_l805_805540


namespace taco_truck_beef_purchase_l805_805133

-- Definitions of the conditions:
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5
def total_profit : ℝ := 200

-- Definition of the number of pounds of beef bought:
def B : ℝ := 100

-- The statement to be proven:
theorem taco_truck_beef_purchase :
  ∀ (B : ℝ),
  0.5 * (4 * B) = total_profit → B = 100 :=
begin
  intro B,
  intro h,
  sorry
end

end taco_truck_beef_purchase_l805_805133


namespace smallest_k_is_1_l805_805830

def n : ℕ := 2020
def expression (k : ℕ) : ℕ := (n - 1) * n * (n + 1) * (n + 2) + k

theorem smallest_k_is_1 : ∃ (k : ℕ), (expression k).isSquare ∧ k = 1 :=
by sorry

end smallest_k_is_1_l805_805830


namespace sin_add_pi_over_four_is_sqrt_three_over_two_l805_805216

theorem sin_add_pi_over_four_is_sqrt_three_over_two
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : sin (2 * α) = 1 / 2) :
  sin (α + π / 4) = sqrt (3) / 2 :=
sorry

end sin_add_pi_over_four_is_sqrt_three_over_two_l805_805216


namespace find_rs_l805_805189

theorem find_rs :
  ∃ r s : ℝ, ∀ x : ℝ, 8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 8 * (x - r) ^ 2 * (x - s) * (x - 1) :=
sorry

end find_rs_l805_805189


namespace median_of_first_twelve_positive_integers_l805_805054

theorem median_of_first_twelve_positive_integers :
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth (5)).getD 0 + (lst.nth (6)).getD 0 / 2 = 6.5 :=
by
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let median := ((lst.nth (5)).getD 0 + (lst.nth (6)).getD 0) / 2
  show median = 6.5
  sorry

end median_of_first_twelve_positive_integers_l805_805054


namespace smallest_x_with_24_factors_and_factors_18_28_l805_805014

theorem smallest_x_with_24_factors_and_factors_18_28 :
  ∃ x : ℕ, (∀ n : ℕ, n > 0 → n ∣ x → (nat.factors x).length = 24) ∧ (18 ∣ x ∧ 28 ∣ x) ∧ x = 504 :=
  sorry

end smallest_x_with_24_factors_and_factors_18_28_l805_805014


namespace find_second_year_rate_l805_805606

noncomputable def principal : ℝ := 7000
noncomputable def time_period : ℝ := 2
noncomputable def first_year_rate : ℝ := 0.04
noncomputable def final_amount : ℝ := 7644
noncomputable def amount_after_first_year : ℝ := principal * (1 + first_year_rate)
noncomputable def second_year_rate (r: ℝ) : Prop :=
  final_amount = amount_after_first_year * (1 + r)

theorem find_second_year_rate : ∃ r : ℝ, second_year_rate r ∧ r = 0.05 :=
by {
  sorry,
}

end find_second_year_rate_l805_805606


namespace trapezoid_height_l805_805017

theorem trapezoid_height (AD BC : ℝ) (AB CD : ℝ) (h₁ : AD = 25) (h₂ : BC = 4) (h₃ : AB = 20) (h₄ : CD = 13) : ∃ h : ℝ, h = 12 :=
by
  -- Definitions
  let AD := 25
  let BC := 4
  let AB := 20
  let CD := 13
  
  sorry

end trapezoid_height_l805_805017


namespace kareem_has_largest_final_number_l805_805727

def jose_final : ℕ := (15 - 2) * 4 + 5
def thuy_final : ℕ := (15 * 3 - 3) - 4
def kareem_final : ℕ := ((20 - 3) + 4) * 3

theorem kareem_has_largest_final_number :
  kareem_final > jose_final ∧ kareem_final > thuy_final := 
by 
  sorry

end kareem_has_largest_final_number_l805_805727


namespace quadratic_roots_real_and_equal_l805_805541

open Real

theorem quadratic_roots_real_and_equal :
  ∀ (x : ℝ), x^2 - 4 * x * sqrt 2 + 8 = 0 → ∃ r : ℝ, x = r :=
by
  intro x
  sorry

end quadratic_roots_real_and_equal_l805_805541


namespace min_pipes_needed_l805_805865

theorem min_pipes_needed :
  ∀ (h : ℝ),
    let main_pipe_diameter := 8
    let small_pipe_diameter := 3
    let reduction_factor := 0.9
    let effective_main_diameter := main_pipe_diameter * reduction_factor
    let effective_small_diameter := small_pipe_diameter * reduction_factor
    let main_radius := effective_main_diameter / 2
    let small_radius := effective_small_diameter / 2
    let main_volume := π * (main_radius^2) * h
    let small_volume := π * (small_radius^2) * h
    let min_pipes := main_volume / small_volume
  in (ceil min_pipes) = 8 := 
by sorry

end min_pipes_needed_l805_805865


namespace page_sum_incorrect_l805_805445

theorem page_sum_incorrect (sheets : List (Nat × Nat)) (h_sheets_len : sheets.length = 25)
  (h_consecutive : ∀ (a b : Nat), (a, b) ∈ sheets → (b = a + 1 ∨ a = b + 1))
  (h_sum_eq_2020 : (sheets.map (λ p => p.1 + p.2)).sum = 2020) : False :=
by
  sorry

end page_sum_incorrect_l805_805445


namespace median_is_26_l805_805801

def data_set := [25, 29, 27, 25, 22, 30, 26]

theorem median_is_26 : median data_set = 26 :=
by
  sorry

end median_is_26_l805_805801


namespace function_passes_through_fixed_point_l805_805441

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 5) ∧ ∀ x, f x = a^(x-1) + 4 := 
sorry

end function_passes_through_fixed_point_l805_805441


namespace Martin_can_see_total_area_l805_805400

-- Define the dimensions of the rectangle
def length_of_park : ℝ := 6
def width_of_park : ℝ := 4

-- Define the visibility radius
def visibility_radius : ℝ := 0.5

-- Define the inner rectangle dimensions
def inner_length : ℝ := length_of_park - 2 * visibility_radius
def inner_width : ℝ := width_of_park - 2 * visibility_radius

-- Define the area of the inner, non-visible rectangle
def inner_area : ℝ := inner_length * inner_width

-- Define the total area of the park
def total_park_area : ℝ := length_of_park * width_of_park

-- Define the area Martin can see inside the park
def visible_area_inside : ℝ := total_park_area - inner_area

-- Define the visible regions outside the rectangle (bands and quarter circles)
def bands_along_length : ℝ := 2 * (length_of_park * visibility_radius)
def bands_along_width : ℝ := 2 * (width_of_park * visibility_radius)
def quarter_circles_area : ℝ := 4 * (0.25 * Math.pi * (visibility_radius ^ 2))

-- Define the total visible area
def total_visible_area : ℝ :=
  visible_area_inside + bands_along_length + bands_along_width + quarter_circles_area

-- The proof statement
theorem Martin_can_see_total_area :
  total_visible_area = 20 := by
  sorry

end Martin_can_see_total_area_l805_805400


namespace monkey_climb_ladder_l805_805868

theorem monkey_climb_ladder (n : ℕ) 
  (h1 : ∀ k, (k % 18 = 0 → (k - 18 + 10) % 26 = 8))
  (h2 : ∀ m, (m % 10 = 0 → (m - 10 + 18) % 26 = 18))
  (h3 : ∀ l, (l % 18 = 0 ∧ l % 10 = 0 → l = 0 ∨ l = 26)):
  n = 26 :=
by
  sorry

end monkey_climb_ladder_l805_805868


namespace Terrell_lifting_two_weights_equivalence_l805_805430

theorem Terrell_lifting_two_weights_equivalence :
  (∀ n : ℕ, 2 * 25 * 15 = 750 ∧ 2 * 10 * n = 750 → n = 38 ∨ n = 37) :=
by
  intro n
  have h₁ : 2 * 25 * 15 = 750 := by norm_num
  have h₂ : 2 * 10 * n = 750 := by linarith
  have n_val : n = 750 / 20 := by linarith
  have h₃ := Int.eq_div_of_mul_eq_right (by norm_num : 20 ≠ 0) n_val.symm
  norm_num at h₃
  rw [h₃, Int.of_nat_eq_coe] at ⊢
  exact or.intro_right _ (37 <. 5 38).symm

end Terrell_lifting_two_weights_equivalence_l805_805430


namespace num_int_values_satisfying_ineq_l805_805261

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805261


namespace smallest_positive_integer_l805_805184

theorem smallest_positive_integer :
  ∃ n : ℕ, n + 21 ≡ 0 [MOD 48] ∧
           n + 21 ≡ 0 [MOD 64] ∧
           n + 21 ≡ 0 [MOD 75] ∧
           n + 21 ≡ 0 [MOD 108] ∧
           n = 43179 :=
begin
  sorry
end

end smallest_positive_integer_l805_805184


namespace minimum_value_l805_805979

open Real

variables {A B C M : Type}
variables (AB AC : ℝ) 
variables (S_MBC x y : ℝ)

-- Assume the given conditions
axiom dot_product_AB_AC : AB * AC = 2 * sqrt 3
axiom angle_BAC_30 : (30 : Real) = π / 6
axiom area_MBC : S_MBC = 1/2
axiom area_sum : x + y = 1/2

-- Define the minimum value problem
theorem minimum_value : 
  ∃ m, m = 18 ∧ (∀ x y, (1/x + 4/y) ≥ m) :=
sorry

end minimum_value_l805_805979


namespace range_of_q_l805_805238

noncomputable def f (k : ℤ) (x : ℝ) : ℝ := x ^ (-k^2 + k + 2)
noncomputable def h (k : ℤ) (q : ℝ) (x : ℝ) : ℝ := f k x + (2 * q - 1) * x

theorem range_of_q (k : ℤ) (q : ℝ) :
  (-1 < k ∧ k < 2 ∧ f k 2 < f k 3 ∧ 0 ≤ q ∧ (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 2 → h k q x ≤ h k q (-1)) ∧ ∀ x : ℝ, x ∈ Icc (-1 : ℝ) 2 → h k q x ≤ h k q 2) →
  0 ≤ q ∧ q ≤ 1/4 :=
by
  sorry

end range_of_q_l805_805238


namespace geom_seq_ratio_l805_805982
noncomputable section

theorem geom_seq_ratio (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h₁ : 0 < a_1)
  (h₂ : 0 < a_2)
  (h₃ : 0 < a_3)
  (h₄ : 0 < a_4)
  (h₅ : 0 < a_5)
  (h_seq : a_2 = a_1 * 2)
  (h_seq2 : a_3 = a_1 * 2^2)
  (h_seq3 : a_4 = a_1 * 2^3)
  (h_seq4 : a_5 = a_1 * 2^4)
  (h_ratio : a_4 / a_1 = 8) :
  (a_1 + a_2) * a_4 / ((a_1 + a_3) * a_5) = 3 / 10 := 
by
  sorry

end geom_seq_ratio_l805_805982


namespace median_of_first_twelve_positive_integers_l805_805071

def sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def median (s : List ℕ) : ℚ :=
  if h : (List.length s) % 2 = 0 then
    let k := List.length s / 2
    (s.get (k - 1) + s.get k) / 2
  else
    s.get (List.length s / 2)

theorem median_of_first_twelve_positive_integers :
  median sequence = 6.5 := 
sorry

end median_of_first_twelve_positive_integers_l805_805071


namespace median_first_twelve_integers_l805_805058

theorem median_first_twelve_integers : 
  let lst : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = (6.5 : ℤ) :=
by
  sorry

end median_first_twelve_integers_l805_805058


namespace negation_proposition_l805_805443

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by { sorry }

end negation_proposition_l805_805443


namespace decrement_from_each_observation_l805_805019

theorem decrement_from_each_observation (n : Nat) (mean_original mean_updated decrement : ℝ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 191)
  (h4 : decrement = 9) :
  (mean_original - mean_updated) * (n : ℝ) / n = decrement :=
by
  sorry

end decrement_from_each_observation_l805_805019


namespace letters_containing_dot_not_straight_line_l805_805704

variable (DS S_not_D Total : ℕ)

theorem letters_containing_dot_not_straight_line (h1 : DS = 11) (h2 : S_not_D = 24) (h3 : Total = 40) :
  ∃ D_not_S, D_not_S = 5 :=
by
  let S := DS + S_not_D
  let D := Total - S_not_D
  let D_not_S := D - DS
  have hS : S = 35 := by rw [h1, h2]; norm_num
  have hD : D = 16 := by rw [h2, h3]; norm_num
  have hD_not_S : D_not_S = 5 := by rw [h1]; norm_num
  use 5
  exact hD_not_S

end letters_containing_dot_not_straight_line_l805_805704


namespace products_pairwise_different_residues_modulo_iff_gcd_one_l805_805618

theorem products_pairwise_different_residues_modulo_iff_gcd_one 
  (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (a : Fin n → ℤ) (b : Fin k → ℤ), ∀ i j, 1 ≤ i.1 + 1 ∧ i.1 + 1 ≤ n 
    → 1 ≤ j.1 + 1 ∧ j.1 + 1 ≤ k → (a i * b j) % (n * k) ≠ (a i' * b j') % (n * k) → 
    (i ≠ i' ∨ j ≠ j')) ↔ (Nat.gcd n k = 1) := 
by
  sorry

end products_pairwise_different_residues_modulo_iff_gcd_one_l805_805618


namespace sin_45_l805_805916

theorem sin_45 (Q E : ℝ × ℝ) 
  (h1 : Q.1 = real.sqrt 2 / 2 ∧ Q.2 = real.sqrt 2 / 2) 
  (h2 : E.1 = Q.1 ∧ E.2 = 0) : 
  real.sin (real.pi / 4) = real.sqrt 2 / 2 := 
sorry

end sin_45_l805_805916


namespace contrapositive_of_zero_squared_l805_805777

theorem contrapositive_of_zero_squared {x y : ℝ} :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) →
  (x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by
  intro h1
  intro h2
  sorry

end contrapositive_of_zero_squared_l805_805777


namespace find_side_a_l805_805339

noncomputable def maximum_area (A b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ (b + 2 * c = 8) ∧ 
  ((1 / 2) * b * c * Real.sin (2 * Real.pi / 3) = (Real.sqrt 3 / 2) * c * (4 - c) ∧ 
   (∀ (c' : ℝ), (Real.sqrt 3 / 2) * c' * (4 - c') ≤ 2 * Real.sqrt 3) ∧ 
   c = 2)

theorem find_side_a (A b c a : ℝ) (h : maximum_area A b c) :
  a = 2 * Real.sqrt 7 := 
by
  sorry

end find_side_a_l805_805339


namespace quadratic_real_roots_iff_l805_805203

theorem quadratic_real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 :=
by
  -- Proof is omitted, we only need the statement
  sorry

end quadratic_real_roots_iff_l805_805203


namespace additional_distance_after_modification_l805_805093

noncomputable def initial_mpg : ℝ := 24
noncomputable def tank_capacity : ℝ := 12
noncomputable def modified_fuel_usage : ℝ := 0.75

theorem additional_distance_after_modification :
  let initial_distance := initial_mpg * tank_capacity in
  let modified_mpg := initial_mpg / modified_fuel_usage in
  let modified_distance := modified_mpg * tank_capacity in
  modified_distance - initial_distance = 96 :=
by 
  let initial_distance := initial_mpg * tank_capacity
  let modified_mpg := initial_mpg / modified_fuel_usage
  let modified_distance := modified_mpg * tank_capacity
  show modified_distance - initial_distance = 96
  sorry

end additional_distance_after_modification_l805_805093


namespace sin_double_angle_l805_805717

variable (φ : ℝ)
hypothesis (h : (7/13) + Real.sin φ = Real.cos φ)

theorem sin_double_angle : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l805_805717


namespace distance_from_O_to_AC_l805_805020

theorem distance_from_O_to_AC
  (A B C O : Point)
  (alpha beta : Real)
  (b : Real)
  (M : Point)
  (N : Point)
  (h1 : is_median A M)
  (h2 : is_median C N)
  (h3 : intersect_at O A M C N)
  (h4 : angle BAC = alpha)
  (h5 : angle BCA = beta)
  (h6 : distance A C = b) :
  distance_from_point_to_line O A C = b * sin(alpha) * sin(beta) / (3 * sin(alpha + beta)) :=
sorry

end distance_from_O_to_AC_l805_805020


namespace even_function_phi_value_l805_805688

theorem even_function_phi_value (φ : ℝ) 
  (h : ∀ x : ℝ, sin (2 * (-x) + φ) = sin (2 * x + φ)) : 
  φ = π / 2 :=
sorry

end even_function_phi_value_l805_805688


namespace sum_of_all_possible_values_is_correct_l805_805025

noncomputable def M_sum_of_all_possible_values (a b c M : ℝ) : Prop :=
  M = a * b * c ∧ M = 8 * (a + b + c) ∧ c = a + b ∧ b = 2 * a

theorem sum_of_all_possible_values_is_correct :
  ∃ M, (∃ a b c, M_sum_of_all_possible_values a b c M) ∧ M = 96 * Real.sqrt 2 := by
  sorry

end sum_of_all_possible_values_is_correct_l805_805025


namespace division_of_fractions_l805_805491

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805491


namespace artwork_arrangements_l805_805712

theorem artwork_arrangements :
  ∃ (n : ℕ), n = 96 ∧ 
  ∀ (C1 C2 P1 P2 A : Type),
    (C1 ≠ C2 ∧ P1 ≠ P2) ∧ 
    (C1 ≠ P1 ∧ C1 ≠ P2 ∧ C1 ≠ A ∧ 
     C2 ≠ P1 ∧ C2 ≠ P2 ∧ C2 ≠ A ∧ 
     P1 ≠ P2 ∧ P1 ≠ A ∧ P2 ≠ A) →
  (list.permutations [C1, C2, P1, P2, A].length = 120 ∧
   (∀ l, l ∈ list.permutations [C1, C2, P1, P2, A] →
    adjacent C1 C2 l ∧ ¬ adjacent P1 P2 l)) → n = 96 :=
by
  sorry

end artwork_arrangements_l805_805712


namespace seconds_in_minutes_l805_805318

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l805_805318


namespace marbles_left_l805_805404

-- Definitions and conditions
def marbles_initial : ℕ := 38
def marbles_lost : ℕ := 15

-- Statement of the problem
theorem marbles_left : marbles_initial - marbles_lost = 23 := by
  sorry

end marbles_left_l805_805404


namespace dr_jones_remaining_salary_l805_805596

noncomputable def remaining_salary (salary rent food utilities insurances taxes transport emergency loan retirement : ℝ) : ℝ :=
  salary - (rent + food + utilities + insurances + taxes + transport + emergency + loan + retirement)

theorem dr_jones_remaining_salary :
  remaining_salary 6000 640 385 (1/4 * 6000) (1/5 * 6000) (0.10 * 6000) (0.03 * 6000) (0.02 * 6000) 300 (0.05 * 6000) = 1275 :=
by
  sorry

end dr_jones_remaining_salary_l805_805596


namespace rectangle_vertex_D_l805_805354

theorem rectangle_vertex_D (A B C D : ℂ) (hA : A = 2 + 3 * Complex.i) (hB : B = 3 + 2 * Complex.i) 
  (hC : C = -2 - 3 * Complex.i) 
  (hRect : (B = A + (C - D)) ∧ (C = D + (B - A))) : 
  D = -3 - 2 * Complex.i :=
by
  sorry

end rectangle_vertex_D_l805_805354


namespace box_volume_l805_805497

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l805_805497


namespace part_1_part_2_l805_805653

noncomputable def smallest_positive_period := 4 * Real.pi

noncomputable def f (ω x : ℝ) := 2 * Real.sin (ω * x) * Real.cos (ω * x)

theorem part_1 : ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = f ω (x + smallest_positive_period)) := by
  use 1 / 2
  sorry

theorem part_2 (A B C a b c : ℝ) (h₀ : 2 * b * Real.cos A = a * Real.cos C + c * Real.cos A) (h₁ : ω = 1 / 2) : 
  f ω A = Real.sin (Real.pi / 3) := by
  have h₂ : ω = 1 / 2 := by
    exact h₁
    
  have cos_A_eq_half : Real.cos A = 1 / 2 := by
    sorry

  have angle_A_eq_pi_div_3 : A = Real.pi / 3 := by
    sorry

  show f ω A = Real.sin (Real.pi / 3), by
    sorry

end part_1_part_2_l805_805653


namespace x_alone_days_eq_20_l805_805516

-- The total work is W
-- x can do W_x amount of work per day
-- y can do W_y amount of work per day, and y can complete the work in 12 days

def x_days_to_complete_work : ℕ :=
  let W := 1 -- We consider the total work to be normalized to 1 for simplicity
  let W_y := W / 12
  let total_days_working_together := 10
  let initial_days_x_worked_alone := 4
  
  -- Equation derived from the problem
  let remaining_days_working_together := total_days_working_together - initial_days_x_worked_alone
  let W_x := (W * 2) / 20
  (W / W_x).to_nat -- Conversion to natural number to represent the number of days

theorem x_alone_days_eq_20 : x_days_to_complete_work = 20 := by
  sorry

end x_alone_days_eq_20_l805_805516


namespace median_of_first_twelve_positive_integers_l805_805053

theorem median_of_first_twelve_positive_integers :
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth (5)).getD 0 + (lst.nth (6)).getD 0 / 2 = 6.5 :=
by
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let median := ((lst.nth (5)).getD 0 + (lst.nth (6)).getD 0) / 2
  show median = 6.5
  sorry

end median_of_first_twelve_positive_integers_l805_805053


namespace find_ordered_pair_l805_805182

def cos30 : ℝ := sqrt 3 / 2
def sec30 : ℝ := 2 * sqrt 3 / 3

theorem find_ordered_pair (c d : ℤ)
  (h : sqrt (16 - 12 * cos30) = c + d * sec30) : c = 4 ∧ d = -1 :=
by
  -- Proof placeholder
  sorry

end find_ordered_pair_l805_805182


namespace sum_frac_sq_geq_half_sum_l805_805983

theorem sum_frac_sq_geq_half_sum
  {n : ℕ}
  {x y : Fin n → ℝ}
  (h_posx : ∀ i, 0 < x i)
  (h_posy : ∀ i, 0 < y i)
  (h_eqsum : (∑ i, x i) = ∑ i, y i) :
  ∑ i, ((x i) ^ 2 / (x i + y i)) ≥ (∑ i, x i) / 2 :=
by
  -- The proof needs to be filled in here.
  sorry

end sum_frac_sq_geq_half_sum_l805_805983


namespace boyfriend_picks_up_correct_l805_805562

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end boyfriend_picks_up_correct_l805_805562


namespace no_such_integers_l805_805237

def p (x : ℤ) : ℤ := x^2 + x - 70

theorem no_such_integers : ¬ (∃ m n : ℤ, 0 < m ∧ m < n ∧ n ∣ p m ∧ (n + 1) ∣ p (m + 1)) :=
by
  sorry

end no_such_integers_l805_805237


namespace product_probability_l805_805817

theorem product_probability (a b : ℝ) (h1 : a ∈ set.Icc (-20 : ℝ) 10) (h2 : b ∈ set.Icc (-20 : ℝ) 10) : 
  (∃ (f : measure_theory.measure_space ℝ), measure_theory.probability_measure f) →
  (∃ (f : measure_theory.measure_space ℝ), 
    measure_theory.probability (λ x, (x.fst * x.snd > 0) (prod.mk a b)) = 5 / 9) :=
sorry

end product_probability_l805_805817


namespace sum_of_transformed_numbers_l805_805456

variables (a b x k S : ℝ)

-- Define the condition that a + b = S
def sum_condition : Prop := a + b = S

-- Define the function that represents the final sum after transformations
def final_sum (a b x k : ℝ) : ℝ :=
  k * (a + x) + k * (b + x)

-- The theorem statement to prove
theorem sum_of_transformed_numbers (h : sum_condition a b S) : 
  final_sum a b x k = k * S + 2 * k * x :=
by
  sorry

end sum_of_transformed_numbers_l805_805456


namespace monotonic_intervals_when_a_zero_range_of_a_if_f_positive_l805_805232

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x + a * x

theorem monotonic_intervals_when_a_zero :
    ∃ I1 I2 : Set ℝ, (I1 = { x | 0 < x ∧ x < 1 }) ∧ (I2 = { x | 1 < x }) ∧
        (∀ x y ∈ I1, x < y → f x 0 > f y 0) ∧ (∀ x y ∈ I2, x < y → f x 0 < f y 0) :=
by
  sorry

theorem range_of_a_if_f_positive (a : ℝ) :
    (∀ x > 0, (x - 1) * Real.log x + a * x > 0) → a > 0 :=
by
  sorry

end monotonic_intervals_when_a_zero_range_of_a_if_f_positive_l805_805232


namespace inradius_constant_when_H_is_focus_l805_805740

open Classical

variables {K : Type*} [Field K] [Inhabited K] {point : Type*} [MetricSpace point]
variables {A B C H : point} {Δ : Point → Prop}

-- Assume A, B, and C lie on the parabola Δ
axiom A_on_Δ : Δ A
axiom B_on_Δ : Δ B
axiom C_on_Δ : Δ C

-- Assume the orthocenter H of triangle ABC coincides with the focus of Δ
axiom H_is_focus : is_focus H Δ

-- Define the inradius of any triangle
noncomputable def inradius (A B C : point) : ℝ := sorry

-- Define orthocenter of any triangle
noncomputable def orthocenter (A B C : point) : point := sorry

-- Define H as the orthocenter of triangle ABC
axiom H_is_orthocenter : orthocenter A B C = H

-- Main theorem to be proved
theorem inradius_constant_when_H_is_focus (A B C H : point) (Δ : point → Prop)
    [MetricSpace point]
    (A_on_Δ : Δ A) (B_on_Δ : Δ B) (C_on_Δ : Δ C)
    (H_is_focus : is_focus H Δ)
    (H_is_orthocenter : orthocenter A B C = H) :
    ∀ (A' B' C' : point), Δ A' → Δ B' → Δ C' → orthocenter A' B' C' = H → 
    inradius A B C = inradius A' B' C' :=
sorry

end inradius_constant_when_H_is_focus_l805_805740


namespace least_sum_of_exponents_l805_805683

theorem least_sum_of_exponents : ∃ (a b c : ℕ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (2^a + 2^b + 2^c + ∃d : ℕ, 2^d = 260) ∧ 
  (a + b + c = 10) := 
sorry

end least_sum_of_exponents_l805_805683


namespace net_change_is_12_l805_805839

-- Definitions based on the conditions of the problem

def initial_investment : ℝ := 100
def first_year_increase_percentage : ℝ := 0.60
def second_year_decrease_percentage : ℝ := 0.30

-- Calculate the wealth at the end of the first year
def end_of_first_year_wealth : ℝ := initial_investment * (1 + first_year_increase_percentage)

-- Calculate the wealth at the end of the second year
def end_of_second_year_wealth : ℝ := end_of_first_year_wealth * (1 - second_year_decrease_percentage)

-- Calculate the net change
def net_change : ℝ := end_of_second_year_wealth - initial_investment

-- The target theorem to prove
theorem net_change_is_12 : net_change = 12 := by
  sorry

end net_change_is_12_l805_805839


namespace area_S3_is_1_l805_805597

def S1_area : ℝ := 81

def trisection_side (side_length : ℝ) : ℝ := side_length / 3

noncomputable def side_length_S1 := real.sqrt S1_area

def side_length_S2 := trisection_side side_length_S1

def side_length_S3 := trisection_side side_length_S2

def area (side_length : ℝ) : ℝ := side_length * side_length

theorem area_S3_is_1 : area side_length_S3 = 1 := by
  sorry

end area_S3_is_1_l805_805597


namespace simplify_and_evaluate_expression_l805_805765

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (a - 3) / (a^2 + 6 * a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expression_l805_805765


namespace find_an_l805_805379

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l805_805379


namespace sector_area_l805_805219

-- Definitions from the problem conditions
def central_angle := Real.pi / 6
def arc_length := 2 * Real.pi / 3

-- Goal: Prove the area of the sector
theorem sector_area (r : ℝ) : 
  (r = 4) → 
  (central_angle = Real.pi / 6) → 
  (arc_length = 2 * Real.pi / 3) → 
  (1 / 2 * arc_length * r = (4 * Real.pi) / 3) := by
sorry

end sector_area_l805_805219


namespace pond_uncovered_area_l805_805114

noncomputable def diameter := 16
noncomputable def bridge_width := 4

theorem pond_uncovered_area :
  let r := diameter / 2 in
  let total_area := π * r^2 in
  let strip_height := bridge_width / 2 in
  let chord_distance := real.sqrt (r^2 - strip_height^2) in
  let θ := 2 * real.arccos (chord_distance / r) in
  let sector_area := θ / (2 * π) * total_area in
  let triangle_area := 2 * (1 / 2 * chord_distance * strip_height) in
  let covered_area := sector_area - triangle_area in
  let remaining_area := total_area - covered_area in
  remaining_area = (128 * π / 3) + 32 * real.sqrt 3 :=
by sorry

end pond_uncovered_area_l805_805114


namespace no_tangent_exists_l805_805226

noncomputable def f (x : ℝ) : ℝ := (4 - x) * Real.exp (x - 2)
def is_tangent (m : ℝ) : Prop := ∃ x : ℝ, f x = (3/2)*x + m and ∃ k : ℝ, ∀ y : ℝ, k * y = 3*x - 2*(f x) + m 

theorem no_tangent_exists : ¬ ∃ m : ℝ, is_tangent m :=
by sorry

end no_tangent_exists_l805_805226


namespace part_a_l805_805841

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  |a - b| + |b - c| + |c - a| ≤ 2 * Real.sqrt 2 :=
sorry

end part_a_l805_805841


namespace divisibility_by_3_and_9_l805_805754

theorem divisibility_by_3_and_9 (N : ℕ) (a : ℕ → ℕ) (n : ℕ) 
  (hN : N = ∑ i in Finset.range (n + 1), a i * 10^i) :
  (N % 3 = ∑ i in Finset.range (n + 1), a i % 3) ∧
  (N % 9 = ∑ i in Finset.range (n + 1), a i % 9) := by sorry

end divisibility_by_3_and_9_l805_805754


namespace integer_count_satisfies_inequality_l805_805299

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805299


namespace problem_equivalent_l805_805196

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem problem_equivalent (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by
  sorry

end problem_equivalent_l805_805196


namespace monotonicity_f_log_sum_inequality_l805_805997

def f (a x : ℝ) : ℝ := a * log x + (1 / x) - 1

theorem monotonicity_f (a : ℝ) (x : ℝ) (hx : 0 < x) :
  (a ≤ 0 → ∀ y, y > 0 → (deriv (f a) x) y < 0) ∧ 
  (a > 0 → (∃ c, c = (1 / a) ∧ 
             (∀ y, y ∈ (0, 1 / a) → (deriv (f a) x) y < 0) ∧ 
             (∀ y, y ∈ (1 / a, ∞) → (deriv (f a) x) y > 0))) :=
sorry

theorem log_sum_inequality (n : ℕ) (hn : 2 ≤ n) :
  ∑ i in Finset.range n, (log (i + 1))^2 > ((n - 1)^4 / 4 / n) :=
sorry

end monotonicity_f_log_sum_inequality_l805_805997


namespace boyfriend_picks_pieces_l805_805564

theorem boyfriend_picks_pieces (initial_pieces : ℕ) (cat_steals : ℕ) 
(boyfriend_fraction : ℚ) (swept_fraction : ℚ) 
(h_initial : initial_pieces = 60) (h_swept : swept_fraction = 1 / 2) 
(h_cat : cat_steals = 3) (h_boyfriend : boyfriend_fraction = 1 / 3) : 
ℕ :=
  let swept_pieces := initial_pieces * swept_fraction
  let remaining_pieces := swept_pieces - cat_steals
  let boyfriend_pieces := remaining_pieces * boyfriend_fraction
  by
    have h_swept_pieces : swept_pieces = 30 := by sorry
    have h_remaining_pieces : remaining_pieces = 27 := by sorry
    have h_boyfriend_pieces : boyfriend_pieces = 9 := by sorry
    exact h_boyfriend_pieces

end boyfriend_picks_pieces_l805_805564


namespace find_lambda_l805_805323

-- Condition Definitions
def point_on_hyperbola (P : ℝ × ℝ) (λ : ℝ) : Prop := (P.1 ^ 2) - (P.2 ^ 2) = λ
def foci_distance (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (d : ℝ) : Prop := dist P F2 = d
def perpendicular_to_real_axis (P : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop := P.2 = λ

-- Proof Problem
theorem find_lambda (λ : ℝ) (P F1 F2 : ℝ × ℝ)
  (h_point : point_on_hyperbola P λ)
  (h_foci : foci_distance P F2 6)
  (h_perpendicular : perpendicular_to_real_axis P F1) :
  λ = 4 :=
sorry

end find_lambda_l805_805323


namespace stratified_sampling_city_B_l805_805116

theorem stratified_sampling_city_B (sales_points_A : ℕ) (sales_points_B : ℕ) (sales_points_C : ℕ) (total_sales_points : ℕ) (sample_size : ℕ)
(h_total : total_sales_points = 450)
(h_sample : sample_size = 90)
(h_sales_points_A : sales_points_A = 180)
(h_sales_points_B : sales_points_B = 150)
(h_sales_points_C : sales_points_C = 120) :
  (sample_size * sales_points_B / total_sales_points) = 30 := 
by
  sorry

end stratified_sampling_city_B_l805_805116


namespace initial_numbers_l805_805816

theorem initial_numbers (x : ℕ) (h1 : 2015 > x) (h2 : ∃ (k : ℕ), 2015 - x = 1024 * k) : x = 991 :=
by {
  sorry
}

end initial_numbers_l805_805816


namespace floor_calc_l805_805171

theorem floor_calc : (Int.floor (4 * (7 - 1 / 3))) = 26 := by
  sorry

end floor_calc_l805_805171


namespace sum_of_sixth_powers_less_than_200_l805_805495

theorem sum_of_sixth_powers_less_than_200 : 
  (∑ n in Finset.filter (λ n : ℕ, n^6 < 200) (Finset.range 200), n) = 65 :=
by
  sorry

end sum_of_sixth_powers_less_than_200_l805_805495


namespace quartic_poly_exists_l805_805936

noncomputable def quartic_poly_with_given_roots (p : ℚ[X]) : Prop :=
  p.monic ∧ p.coeff 4 = 1 ∧
  (p.eval (3 + real.sqrt 5) = 0) ∧ (p.eval (3 - real.sqrt 5) = 0) ∧
  (p.eval (2 - real.sqrt 7) = 0) ∧ (p.eval (2 + real.sqrt 7) = 0)

theorem quartic_poly_exists :
  ∃ p : ℚ[X], quartic_poly_with_given_roots p ∧ p = (X^4 - 10*X^3 + 13*X^2 + 18*X - 12) :=
begin
  sorry
end

end quartic_poly_exists_l805_805936


namespace count_integers_in_range_num_of_integers_l805_805265

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805265


namespace inradius_relation_l805_805971

-- Definition of the problem
variables {A B C D E F G O O₁ O₂ : Type*}

-- Conditions of the problem
variables [RightTriangle ABC C] (circumcenter O ABC)
variables (perpendicular CD AB C D)
variables (Circle O₁ BC CD DB E F G r₁)
variables (Circle O₂ AC CD DA E F G r₂)
variables (inradius ABC r)

-- Statement of the proof problem
theorem inradius_relation :
  r = (r₁ + r₂) / 2 :=
by
  sorry

end inradius_relation_l805_805971


namespace total_households_l805_805346

-- Define the conditions
def households_with_neither := 11
def households_with_both := 18
def households_with_car := 44
def households_with_bike_only := 35

-- The statement to prove
theorem total_households : 
  let households_with_car_only := households_with_car - households_with_both in
  let total_households := households_with_car_only + households_with_bike_only + households_with_both + households_with_neither in
  total_households = 90 :=
by
  -- Placeholder for the proof
  sorry

end total_households_l805_805346


namespace binary_addition_to_decimal_l805_805472

theorem binary_addition_to_decimal : nat.ofDigits 2 [1, 0, 1, 0, 1, 0, 1] + nat.ofDigits 2 [1, 1, 1, 0, 0, 0] = 141 := by
  sorry

end binary_addition_to_decimal_l805_805472


namespace austin_more_apples_than_dallas_l805_805918

-- Conditions as definitions
def dallas_apples : ℕ := 14
def dallas_pears : ℕ := 9
def austin_pears : ℕ := dallas_pears - 5
def austin_total_fruit : ℕ := 24

-- The theorem statement
theorem austin_more_apples_than_dallas 
  (austin_apples : ℕ) (h1 : austin_apples + austin_pears = austin_total_fruit) :
  austin_apples - dallas_apples = 6 :=
sorry

end austin_more_apples_than_dallas_l805_805918


namespace tan_of_alpha_l805_805962

noncomputable def sin_alpha : ℝ := 5 / 13
noncomputable def alpha_in_second_quadrant : Prop := true

theorem tan_of_alpha (h1: sin_alpha = 5 / 13) (h2: alpha_in_second_quadrant) : 
  Real.tan (arcsin sin_alpha) = - 5 / 12 :=
by
  sorry

end tan_of_alpha_l805_805962


namespace problem_remaining_integers_l805_805586

def remaining_int_in_set (S : Finset ℕ) : ℕ :=
  let multiples_4 := S.filter (λ x, x % 4 = 0)
  let multiples_5 := S.filter (λ x, x % 5 = 0)
  let remaining := S \ (multiples_4 ∪ multiples_5)
  remaining.card

theorem problem_remaining_integers (T : Finset ℕ) (hT : T = Finset.range 101 \ {0}) :
  remaining_int_in_set T = 60 :=
by
  rw hT
  simp [remaining_int_in_set]
  sorry

end problem_remaining_integers_l805_805586


namespace correct_propositions_l805_805994

open Real

theorem correct_propositions :
  let e_ellipse := (sqrt 5) / 3
  let major_axis_ellipse := 2 * sqrt 3
  let directrix_parabola := -1 / 8
  let asymptote_hyperbola_1 := λ x, (5 / 7) * x
  let asymptote_hyperbola_2 := λ x, -(5 / 7) * x
  let root_ellipse := 1 / 2
  let root_hyperbola := 2
  (e_ellipse ≠ sqrt 3 / 3)
  ∧ (major_axis_ellipse = 2 * sqrt 3)
  ∧ (directrix_parabola = -1 / 8)
  ∧ (asymptote_hyperbola_1 ≠ (7 / 5) * id)
  ∧ (asymptote_hyperbola_2 ≠ -(7 / 5) * id)
  ∧ ((root_ellipse, root_hyperbola) = (1 / 2, 2)) :=
by {
  sorry
}

end correct_propositions_l805_805994


namespace number_of_articles_l805_805329

variables (C S N : ℝ)
noncomputable def gain : ℝ := 3 / 7

-- Cost price of 50 articles is equal to the selling price of N articles
axiom cost_price_eq_selling_price : 50 * C = N * S

-- Selling price is cost price plus gain percentage
axiom selling_price_with_gain : S = C * (1 + gain)

-- Goal: Prove that N = 35
theorem number_of_articles (h1 : 50 * C = N * C * (10 / 7)) : N = 35 := by
  sorry

end number_of_articles_l805_805329


namespace problem_statement_l805_805981

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 4
def g (x : ℝ) : ℝ := 2*x - 1

-- State the theorem and provide the necessary conditions
theorem problem_statement : f (g 5) - g (f 5) = 381 :=
by
  sorry

end problem_statement_l805_805981


namespace hens_count_l805_805535

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := by
  sorry

end hens_count_l805_805535


namespace incorrect_option_c_l805_805622

theorem incorrect_option_c (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a ≤ sqrt b :=
begin
by_contra h,
rw [not_le] at h,
have h3 : a > 0, from gt_of_gt_of_ge h1 (le_of_lt h2),
have sqrt_b_pos : sqrt b > 0, from sqrt_pos.mpr h2,
exact lt_irrefl (sqrt b) (calc
  sqrt b < a : h
  ... = sqrt (a * a) : (sqrt_sq (le_of_lt h3)).symm
  ... ≤ sqrt (b * b) : sqrt_le_sqrt h2 (le_of_lt h1)
  ... = sqrt b : by rw [sqrt_sq, abs_of_pos h2]),
sorry
end

end incorrect_option_c_l805_805622


namespace simplify_log_expression_l805_805420

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l805_805420


namespace projection_is_correct_l805_805626

variables {α : Type*} [inner_product_space ℝ α] 
variables (a e : α) (θ : ℝ)

noncomputable def projection_vector (a e : α) : α :=
  (‖a‖ * real.cos θ) • (e / ‖e‖)

theorem projection_is_correct
  (h₁ : ‖a‖ = 2)
  (h₂ : ‖e‖ = 1)
  (h₃ : θ = 3 * real.pi / 4) :
  projection_vector a e = -real.sqrt 2 • e := 
sorry

end projection_is_correct_l805_805626


namespace tour_guide_groupings_l805_805468

theorem tour_guide_groupings : 
  let n := 8 in 
  let possible_groupings := (finset.range 7).sum (λ k, nat.choose n (k + 1)) in
  possible_groupings = 246 :=
by
  let n := 8
  let possible_groupings := (finset.range 7).sum (λ k, nat.choose n (k + 1))
  have h_poss_groupings : possible_groupings = (8 + 28 + 56 + 70 + 56 + 28)
    ... = 246 := sorry
  exact h_poss_groupings

end tour_guide_groupings_l805_805468


namespace average_speed_for_trip_l805_805857

theorem average_speed_for_trip (t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (total_time : ℝ) 
  (h₁ : t₁ = 6) 
  (h₂ : v₁ = 30) 
  (h₃ : t₂ = 2) 
  (h₄ : v₂ = 46) 
  (h₅ : total_time = t₁ + t₂) 
  (h₆ : total_time = 8) :
  ((v₁ * t₁ + v₂ * t₂) / total_time) = 34 := 
  by 
    sorry

end average_speed_for_trip_l805_805857


namespace quartic_polynomial_with_roots_l805_805932

theorem quartic_polynomial_with_roots :
  ∃ p : Polynomial ℚ, p.monic ∧ p.degree = 4 ∧ (p.eval (3 + Real.sqrt 5) = 0) ∧ (p.eval (2 - Real.sqrt 7) = 0) :=
by
  sorry

end quartic_polynomial_with_roots_l805_805932


namespace integer_values_between_sqrt_inequality_l805_805455

theorem integer_values_between_sqrt_inequality : 
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.to_finset.card = 15 := 
by 
  sorry

end integer_values_between_sqrt_inequality_l805_805455


namespace verify_final_weights_l805_805726

-- Define the initial weights
def initial_bench_press : ℝ := 500
def initial_squat : ℝ := 400
def initial_deadlift : ℝ := 600

-- Define the weight adjustment transformations for each exercise
def transform_bench_press (w : ℝ) : ℝ :=
  let w1 := w * 0.20
  let w2 := w1 * 1.60
  let w3 := w2 * 0.80
  let w4 := w3 * 3
  w4

def transform_squat (w : ℝ) : ℝ :=
  let w1 := w * 0.50
  let w2 := w1 * 1.40
  let w3 := w2 * 2
  w3

def transform_deadlift (w : ℝ) : ℝ :=
  let w1 := w * 0.70
  let w2 := w1 * 1.80
  let w3 := w2 * 0.60
  let w4 := w3 * 1.50
  w4

-- The final calculated weights for verification
def final_bench_press : ℝ := 384
def final_squat : ℝ := 560
def final_deadlift : ℝ := 680.4

-- Statement of the problem: prove that the transformed weights are as calculated
theorem verify_final_weights : 
  transform_bench_press initial_bench_press = final_bench_press ∧ 
  transform_squat initial_squat = final_squat ∧ 
  transform_deadlift initial_deadlift = final_deadlift := 
by 
  sorry

end verify_final_weights_l805_805726


namespace triangle_area_example_l805_805943

def point (α : Type) := (α × α × α)

def vector3D (α : Type) := (α × α × α)

def triangle_area (A B C : point ℝ) : ℝ :=
  let (x1, y1, z1) := A in
  let (x2, y2, z2) := B in
  let (x3, y3, z3) := C in
  let ux := x2 - x1 in
  let uy := y2 - y1 in
  let uz := z2 - z1 in
  let vx := x3 - x1 in
  let vy := y3 - y1 in
  let vz := z3 - z1 in
  let cx := uy * vz - uz * vy in
  let cy := uz * vx - ux * vz in
  let cz := ux * vy - uy * vx in
  (1/2) * (real.sqrt (cx * cx + cy * cy + cz * cz))

theorem triangle_area_example : 
  triangle_area (0, 0, 0) (4, 3, 2) (1, 0, 2) = 4.5 :=
sorry

end triangle_area_example_l805_805943


namespace smaller_angle_at_7_15_45_is_116_l805_805824

-- Define the conditions and question
def degree_measure_of_smaller_angle (hour minute second : ℕ) : ℝ :=
  let minute_angle := (minute * 6 + second * 0.1 : ℝ)
  let hour_angle := (hour * 30 + (minute * 60 + second) * 0.00833 : ℝ)
  let angle_diff := abs (minute_angle - hour_angle)
  min angle_diff (360 - angle_diff)

theorem smaller_angle_at_7_15_45_is_116.125 :
  degree_measure_of_smaller_angle 7 15 45 = 116.125 :=
by
  sorry

end smaller_angle_at_7_15_45_is_116_l805_805824


namespace seconds_in_12_5_minutes_l805_805320

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l805_805320


namespace length_MN_of_circle_l805_805529

def point := ℝ × ℝ

def circle_passing_through (A B C: point) :=
  ∃ (D E F : ℝ), ∀ (p : point), p = A ∨ p = B ∨ p = C →
    (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

theorem length_MN_of_circle (A B C : point) (H : circle_passing_through A B C) :
  A = (1, 3) → B = (4, 2) → C = (1, -7) →
  ∃ M N : ℝ, (A.1 * 0 + N^2 + D * 0 + E * N + F = 0) ∧ (A.1 * 0 + M^2 + D * 0 + E * M + F = 0) ∧
  abs (M - N) = 4 * Real.sqrt 6 := 
sorry

end length_MN_of_circle_l805_805529


namespace count_cyclic_quadrilaterals_l805_805957

-- Definitions of the quintet types of quadrilaterals
structure Quadrilateral :=
(square : Bool)
(rectangle : Bool)
(rhombus : Bool)
(kite : Bool)
(trapezoid : Bool)

def isCyclic (q : Quadrilateral) : Prop :=
  q.square ∨ q.rectangle
  -- Note: This defines cyclic property only for square and rectangle

-- theorem statement
theorem count_cyclic_quadrilaterals : 
  ∀ (qs : List Quadrilateral), (List.countp isCyclic qs) = 2 := by
  sorry

end count_cyclic_quadrilaterals_l805_805957


namespace trigonometric_inequality_l805_805616

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  sin (cos x) < cos x ∧ cos x < cos (sin x) :=
sorry

end trigonometric_inequality_l805_805616


namespace characteristics_differ_l805_805342

noncomputable def range (s : List ℚ) : ℚ := s.maximum - s.minimum
noncomputable def median (s : List ℚ) : ℚ := s.sorted.get (s.length / 2)
noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length
noncomputable def variance (s : List ℚ) : ℚ := (s.map (λ x, (x - mean s)^2)).sum / s.length

theorem characteristics_differ
  (scores : List ℚ)
  (h_len : scores.length = 7)
  (valid_scores : List ℚ)
  (h_valid_scores : valid_scores = scores.erase scores.maximum.erase scores.minimum)
  (h_len_valid : valid_scores.length = 5) :
  (range scores ≠ range valid_scores) ∧ 
  (mean scores ≠ mean valid_scores) ∧ 
  (variance scores ≠ variance valid_scores) :=
by sorry

end characteristics_differ_l805_805342


namespace falling_speed_function_parachute_radius_l805_805183

noncomputable def falling_speed (m k g S: ℝ) (t : ℝ) : ℝ :=
  sqrt ((m * g) / (k * S)) * real.tanh (t * sqrt ((g * k * S) / m))

theorem falling_speed_function (m k g S: ℝ) (t : ℝ) :
  falling_speed m k g S t = sqrt ((m * g) / (k * S)) * real.tanh (t * sqrt ((g * k * S) / m)) :=
sorry

def terminal_velocity (m k v_max: ℝ) : ℝ :=
  m * 9.8 / (k * v_max^2)

theorem parachute_radius (m k v_max: ℝ) (π: ℝ := real.pi) :
  sqrt (terminal_velocity m k v_max / π) = 12.25 :=
sorry

end falling_speed_function_parachute_radius_l805_805183


namespace incorrect_statement_l805_805654

variable {a : ℝ}

def f (x : ℝ) : ℝ := a * sin x - x

theorem incorrect_statement :
  ¬ (∀ (x y : ℝ), (f x = 0 ∧ f y = 0) → x = y → a = 1) := 
sorry

end incorrect_statement_l805_805654


namespace box_width_is_approx_12_2_l805_805526

noncomputable def box_width (total_volume : ℕ) (total_cost : ℕ) (cost_per_box : ℕ) : ℝ :=
  let num_boxes := total_cost / cost_per_box
  let volume_per_box := total_volume / num_boxes
  real.cbrt volume_per_box

-- We are given certain conditions as follows:
def total_volume : ℕ := 1080000  -- 1.08 million cubic inches
def total_cost : ℕ := 480        -- $480 per month
def cost_per_box : ℕ := 8        -- $0.8 per box per month (note: integer representation of 0.8 * 10 = 8)

-- The goal is to prove that the approximation of the width of the boxes is approximately 12.2 inches.
theorem box_width_is_approx_12_2 : |box_width total_volume total_cost cost_per_box - 12.2| < 0.1 :=
begin
  -- The proof will go here.
  sorry
end

end box_width_is_approx_12_2_l805_805526


namespace num_subsets_set_neg1_0_1_l805_805802

theorem num_subsets_set_neg1_0_1 : 
  (finset.powerset {↑-1, ↑0, ↑1}).card = 8 := 
by
  sorry

end num_subsets_set_neg1_0_1_l805_805802


namespace chickens_sold_correct_l805_805412

theorem chickens_sold_correct :
  ∀ (initial_chickens chickens_sold_to_neighbor chickens_left: ℕ),
  initial_chickens = 80 →
  chickens_sold_to_neighbor = 12 →
  chickens_left = 43 →
  initial_chickens - chickens_sold_to_neighbor - chickens_left = 25 :=
begin
  intros,
  /- The proof would go here -/
  sorry,
end

end chickens_sold_correct_l805_805412


namespace area_and_cost_of_path_l805_805094

variables (length_field width_field path_width : ℝ) (cost_per_sq_m : ℝ)

noncomputable def area_of_path (length_field width_field path_width : ℝ) : ℝ :=
  let total_length := length_field + 2 * path_width
  let total_width := width_field + 2 * path_width
  let area_with_path := total_length * total_width
  let area_grass_field := length_field * width_field
  area_with_path - area_grass_field

noncomputable def cost_of_path (area_of_path cost_per_sq_m : ℝ) : ℝ :=
  area_of_path * cost_per_sq_m

theorem area_and_cost_of_path
  (length_field width_field path_width : ℝ)
  (cost_per_sq_m : ℝ)
  (h_length_field : length_field = 75)
  (h_width_field : width_field = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_sq_m : cost_per_sq_m = 10) :
  area_of_path length_field width_field path_width = 675 ∧
  cost_of_path (area_of_path length_field width_field path_width) cost_per_sq_m = 6750 :=
by
  rw [h_length_field, h_width_field, h_path_width, h_cost_per_sq_m]
  simp [area_of_path, cost_of_path]
  sorry

end area_and_cost_of_path_l805_805094


namespace orthocenter_bisects_one_altitude_orthocenter_does_not_bisect_two_altitudes_l805_805912

-- Define the orthocenter and its properties
structure Triangle (α : Type _) [EuclideanGeometry α] :=
(A B C : α)

def orthocenter {α : Type _} [EuclideanGeometry α] (T : Triangle α) : α := sorry
def is_altitude {α : Type _} [EuclideanGeometry α] (T : Triangle α) (line : α → α → Prop) (altitude : α) : Prop := sorry
def is_midpoint {α : Type _} [EuclideanGeometry α] (m a b : α) : Prop := sorry

-- Statement for part (a): Orthocenter bisecting one altitude.
theorem orthocenter_bisects_one_altitude {α : Type _} [EuclideanGeometry α] (T : Triangle α) :
  ∃ (altitude : α) (line1 : α → α → Prop), is_altitude T line1 altitude ∧ is_midpoint (orthocenter T) altitude (T.A) :=
sorry

-- Statement for part (b): Orthocenter cannot bisect two altitudes.
theorem orthocenter_does_not_bisect_two_altitudes {α : Type _} [EuclideanGeometry α] (T : Triangle α) :
  ¬ ∃ (altitude1 altitude2 : α) (line1 line2 : α → α → Prop), 
    (is_altitude T line1 altitude1 ∧ is_midpoint (orthocenter T) altitude1 (T.A)) ∧ 
    (is_altitude T line2 altitude2 ∧ is_midpoint (orthocenter T) altitude2 (T.B)) :=
sorry

end orthocenter_bisects_one_altitude_orthocenter_does_not_bisect_two_altitudes_l805_805912


namespace disk_tangent_position_after_two_cycles_l805_805007

def circle (r : ℝ) := { P : ℝ × ℝ // P.1^2 + P.2^2 = r^2 }
def clock_face := circle 30
def disk := circle 15

def tangent_point {r1 r2 : ℝ} (c1 : circle r1) (c2 : circle r2) : ℝ × ℝ := sorry

theorem disk_tangent_position_after_two_cycles : 
  ∀ (d : disk) (c : clock_face), tangent_point c d = (0, 30) :=
sorry

end disk_tangent_position_after_two_cycles_l805_805007


namespace triangle_radii_l805_805383

-- Define the properties of the scalene triangle
variables (A B C I : Type)
variables  {dAB dBC dCA : ℝ}
variables  (hAB : dAB = 13) (hBC : dBC = 14) (hCA : dCA = 15)
variable  (dIA : ℝ)
variable  (hIA : dIA = 7)

-- Define the function to calculate the circumradius and inradius
def inradius (s K : ℝ) : ℝ := K / s
def circumradius (a b c K : ℝ) : ℝ := (a * b * c) / (4 * K)

-- Define the area using Heron's formula and semi-perimeter
noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def heron (s a b c : ℝ) : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the theorem proof problem statement
theorem triangle_radii (a b c ia : ℝ)
  (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) (h4 : ia = 7) :
  let s := semi_perimeter a b c,
      K := heron s a b c,
      r := inradius s K,
      R := circumradius a b c K
  in r = 4 ∧ R = (65 / 8) :=
sorry

end triangle_radii_l805_805383


namespace fraction_of_salary_on_rent_l805_805122

theorem fraction_of_salary_on_rent
  (S : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (remaining_amount : ℝ) (approx_salary : ℝ)
  (food_fraction_eq : food_fraction = 1 / 5)
  (clothes_fraction_eq : clothes_fraction = 3 / 5)
  (remaining_amount_eq : remaining_amount = 19000)
  (approx_salary_eq : approx_salary = 190000) :
  ∃ (H : ℝ), H = 1 / 10 :=
by
  sorry

end fraction_of_salary_on_rent_l805_805122


namespace num_int_values_satisfying_ineq_l805_805257

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805257


namespace distance_of_point_P_to_plane_alpha_l805_805990

noncomputable def distance_point_to_plane (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) : ℝ :=
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
  let dot := n.1 * AP.1 + n.2 * AP.2 + n.3 * AP.3 in
  let norm := Real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  abs dot / norm

theorem distance_of_point_P_to_plane_alpha :
  distance_point_to_plane (1, 1, 4) (2, 1, 2) (-2, 3, 0) = 4 :=
sorry

end distance_of_point_P_to_plane_alpha_l805_805990


namespace remainder_of_polynomial_division_l805_805828

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 5 * x - 10

-- Prove that the remainder when P(x) is divided by D(x) is -10
theorem remainder_of_polynomial_division : (P 2) = -10 := by
  sorry

end remainder_of_polynomial_division_l805_805828


namespace crushing_load_l805_805169

theorem crushing_load (T H C : ℝ) (L : ℝ) 
  (h1 : T = 5) (h2 : H = 10) (h3 : C = 3)
  (h4 : L = C * 25 * T^4 / H^2) : 
  L = 468.75 :=
by
  sorry

end crushing_load_l805_805169


namespace sin_210_eq_neg_half_l805_805572

theorem sin_210_eq_neg_half : sin 210 = - (1 / 2) :=
by
  sorry

end sin_210_eq_neg_half_l805_805572


namespace second_number_is_72_l805_805808

theorem second_number_is_72 : ∃ (x : ℝ), 
  (let first := 2 * x,
       third := (2 / 3) * first
   in first + x + third = 264) ∧ x = 72 := 
by
  sorry

end second_number_is_72_l805_805808


namespace market_value_of_stock_l805_805091

theorem market_value_of_stock : 
  ∃ V : ℝ, (0.06 * 100 = 0.08 * V) :=
by {
  use (6 / 0.08),
  ring_nf,
  refl,
}

end market_value_of_stock_l805_805091


namespace three_a_ge_two_b_plus_two_l805_805739

theorem three_a_ge_two_b_plus_two (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (a! * b!) % (a! + b!) = 0) :
  3 * a ≥ 2 * b + 2 :=
sorry

end three_a_ge_two_b_plus_two_l805_805739


namespace num_int_values_satisfying_ineq_l805_805256

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805256


namespace even_length_choices_for_triangle_l805_805046

theorem even_length_choices_for_triangle (x : ℕ) (h1 : 5 + 7 > x) (h2 : x > abs (7 - 5)) (hx_even : x % 2 = 0) : 
  ∃ n, n = 4 ∧ x ∈ {4, 6, 8, 10} :=
by {
  sorry
}

end even_length_choices_for_triangle_l805_805046


namespace any_nat_representation_as_fraction_l805_805083

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l805_805083


namespace lambda_sum_l805_805697

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D E : V)
variables (λ1 λ2 : ℝ)
variables (hD : D = A + (2/3) • (C - A))
variables (hE : E = B + (2 : ℝ) • (D - B))
variables (hAE : E - A = λ1 • (B - A) + λ2 • (C - A))

theorem lambda_sum (hD : D = A + (2/3) • (C - A))
  (hE : E = B + (2 : ℝ) • (D - B))
  (hAE : E - A = λ1 • (B - A) + λ2 • (C - A)) :
  λ1 + λ2 = 1/2 :=
sorry

end lambda_sum_l805_805697


namespace equidistant_points_solution_count_l805_805048

noncomputable def solution_count_equidistant_points (l₁ l₂ l₃ : Line) : ℕ :=
if intersect_at_single_point l₁ l₂ l₃ then
  1
else if form_triangle l₁ l₂ l₃ then
  4
else if are_parallel l₁ l₂ ∧ are_parallel l₂ l₃ then
  0
else if two_parallel_one_intersect l₁ l₂ l₃ then
  2
else
  0

-- Helper predicates and definitions; These require correct logical definitions which we outline conceptually.
-- Assume appropriate geometrical definitions for these predicates exist.
def intersect_at_single_point (l₁ l₂ l₃ : Line) : Prop := sorry
def form_triangle (l₁ l₂ l₃ : Line) : Prop := sorry
def are_parallel (l₁ l₂ : Line) : Prop := sorry
def two_parallel_one_intersect (l₁ l₂ l₃ : Line) : Prop := sorry

theorem equidistant_points_solution_count (l₁ l₂ l₃ : Line) :
  solution_count_equidistant_points l₁ l₂ l₃ = 
  if intersect_at_single_point l₁ l₂ l₃ then
    1
  else if form_triangle l₁ l₂ l₃ then
    4
  else if are_parallel l₁ l₂ ∧ are_parallel l₂ l₃ then
    0
  else if two_parallel_one_intersect l₁ l₂ l₃ then
    2
  else
    0 :=
by sorry

end equidistant_points_solution_count_l805_805048


namespace relationship_between_points_l805_805689

theorem relationship_between_points (a b c y1 y2 y3 : ℝ) :
  (∀ x, y = -x^2 - b * x - c) →
  y = -1^2 - b * -1 - c = a →
  y = -3^2 - b * 3 - c = a →
  y = - (-2)^2 - b * (-2) - c = y1 →
  y = - (-(√2))^2 - b * (-(√2)) - c = y2 →
  y = -1^2 - b * 1 - c = y3 →
  y3 < y2 ∧ y2 < y1 :=
sorry

end relationship_between_points_l805_805689


namespace count_ordered_pairs_l805_805680

theorem count_ordered_pairs :
  ∃ (pairs : Finset (ℝ × ℝ)), -- Define a finite set of pairs of real numbers
    (∀ pair ∈ pairs,            -- For every pair in the set
      let x := pair.1,          -- Let x be the first element of the pair
          y := pair.2 in        -- Let y be the second element of the pair
      (-100 * Real.pi ≤ x ∧ x ≤ 100 * Real.pi) ∧  -- Condition 1
      (-100 * Real.pi ≤ y ∧ y ≤ 100 * Real.pi) ∧  -- Condition 2
      (x + y = 20.19) ∧                         -- Condition 3
      (Real.tan x + Real.tan y = 20.19)) ∧       -- Condition 4
    pairs.card = 388 :=                          -- Number of pairs is 388
sorry

end count_ordered_pairs_l805_805680


namespace number_of_clients_visited_garage_l805_805882

theorem number_of_clients_visited_garage
  (num_cars : ℕ)
  (car_selections : ℕ → ℕ)
  (num_colors : ℕ)
  (cars_per_color : ℕ)
  (num_clients : ℕ)
  (selects_3_cars : Π c, (c < num_clients) → fin (car_selections c) → fin num_cars)
  (each_car_selected_3_times : ∀ car, ∑ c in finset.range num_clients, (∃ k : fin (car_selections c), selects_3_cars c (by linarith) k = car) = 3)
  (num_colors_are_equal : (finset.card (finset.filter (λ c, car_colors c = red) (finset.range num_cars)) 
                        = finset.card (finset.filter (λ c, car_colors c = blue) (finset.range num_cars)) 
                        = finset.card (finset.filter (λ c, car_colors c = green) (finset.range num_cars))))
  (num_cars = 15) (num_colors = 3) (car_selections = λ c, 3) 
  (cars_per_color = num_cars / num_colors) :
  num_clients = 15 :=
by {
  sorry
}

end number_of_clients_visited_garage_l805_805882


namespace sin_is_butterfly_function_sqrt_x2_minus_1_is_butterfly_function_both_functions_are_butterfly_functions_l805_805954

def is_butterfly_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, (f x + x) * (f x - x) ≤ 0

theorem sin_is_butterfly_function : is_butterfly_function (λ x, Real.sin x) :=
sorry

theorem sqrt_x2_minus_1_is_butterfly_function : is_butterfly_function (λ x, Real.sqrt (x^2 - 1)) :=
sorry

theorem both_functions_are_butterfly_functions :
  is_butterfly_function (λ x, Real.sin x) ∧ is_butterfly_function (λ x, Real.sqrt (x^2 - 1)) :=
⟨sin_is_butterfly_function, sqrt_x2_minus_1_is_butterfly_function⟩

end sin_is_butterfly_function_sqrt_x2_minus_1_is_butterfly_function_both_functions_are_butterfly_functions_l805_805954


namespace mostSuitableForComprehensiveSurvey_l805_805505

-- Definitions of conditions
def optionA := "Understanding the sleep time of middle school students nationwide"
def optionB := "Understanding the water quality of a river"
def optionC := "Surveying the vision of all classmates"
def optionD := "Surveying the number of fish in a pond"

-- Define the notion of being the most suitable option for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : String) := option = optionC

-- The theorem statement
theorem mostSuitableForComprehensiveSurvey : isSuitableForComprehensiveSurvey optionC := by
  -- This is the Lean 4 statement where we accept the hypotheses
  -- and conclude the theorem. Proof is omitted with "sorry".
  sorry

end mostSuitableForComprehensiveSurvey_l805_805505


namespace Tim_placed_rulers_l805_805813

variable (initial_rulers final_rulers : ℕ)
variable (placed_rulers : ℕ)

-- Given conditions
def initial_rulers_def : initial_rulers = 11 := sorry
def final_rulers_def : final_rulers = 25 := sorry

-- Goal
theorem Tim_placed_rulers : placed_rulers = final_rulers - initial_rulers :=
  by
  sorry

end Tim_placed_rulers_l805_805813


namespace sum_distances_l805_805640

def point := ℝ × ℝ

noncomputable def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * real.cos θ
noncomputable def line_l (P : point) (angle: ℝ) (x y: ℝ) : Prop := y = P.2 + real.sin (angle) * (x - P.1) / real.cos (angle)

theorem sum_distances (P : point) (A B : point) (C_center : point) (radius : ℝ) :
  P = (2,1) → ∀ θ ρ, circle_C ρ θ → A ≠ P → B ≠ P →
  line_l P (3*real.pi/4) A.1 A.2 → line_l P (3*real.pi/4) B.1 B.2 →
  A ≠ B ∧ (A.1 - C_center.1)^2 + A.2^2 = radius^2 ∧
  (B.1 - C_center.1)^2 + B.2^2 = radius^2 →
  C_center = (2, 0) → radius = 2 →
  |((P.1 - A.1)^2 + (P.2 - A.2)^2)^0.5 + ((P.1 - B.1)^2 + (P.2 - B.2)^2)^0.5| = real.sqrt 14 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry 

end sum_distances_l805_805640


namespace sets_equal_l805_805142

def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, abs (-(Real.sqrt 3))}

theorem sets_equal : A = B :=
by 
  sorry

end sets_equal_l805_805142


namespace num_lines_through_A_parallel_to_planes_is_one_l805_805960

-- Definitions
variables {Point : Type*} {Plane : Type*} {Line : Type*}
variables (alpha beta : Plane) (A : Point)

-- Given conditions
variables (IntersectPlanes : ∃ l : Line, ∀ P : Plane, (P = alpha ∨ P = beta) → l ∈ P)
variables (A_not_in_alpha : A ∉ alpha)
variables (A_not_in_beta : A ∉ beta)

-- Prove the statement
theorem num_lines_through_A_parallel_to_planes_is_one :
  (∃! l : Line, (∀ P : Plane, (P = alpha ∨ P = beta) → l ∈ P) ∧ ∀ m : Line, (A ∈ m) ↔ (m = l)) :=
sorry

end num_lines_through_A_parallel_to_planes_is_one_l805_805960


namespace train_pass_tree_in_16_seconds_l805_805891

-- Conditions
def train_length : ℝ := 280
def speed_kmph : ℝ := 63

-- Convert speed from km/hr to m/s
def speed_mps : ℝ := speed_kmph * (1000 / 3600)

-- Question: How many seconds to pass the tree?
def time_to_pass_tree (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Theorem to prove the mathematically equivalent statement
theorem train_pass_tree_in_16_seconds : time_to_pass_tree train_length speed_mps = 16 :=
by
  -- The proof steps will be included here
  sorry

end train_pass_tree_in_16_seconds_l805_805891


namespace inequality_for_log_function_l805_805969

theorem inequality_for_log_function (m n : ℝ) (h₀ : 0 < m) (h₁ : m < n) :
  (f n - f m) ≤ (1 - m) * (log n - log m) :=
by
  let f := λ x : ℝ, log x - (x - 1)
  sorry

end inequality_for_log_function_l805_805969


namespace polynomial_relation_l805_805634

def table : List (ℕ × ℕ) := [
  (1, 5),
  (2, 15),
  (3, 35),
  (4, 69),
  (5, 119)
]

theorem polynomial_relation :
  ∀ (x y : ℕ), (x, y) ∈ table → y = x^3 + 2 * x^2 + x + 1 :=
by
  intros x y h
  cases h
  case inl h' => -- Case when (x, y) = (1, 5)
    have h1 : x = 1 := by cases h'; rfl
    have h2 : y = 5 := by cases h'; rfl
    rw h1
    rw h2
    simp
  case inr hinr =>
    cases hinr
    case inl h' => -- Case when (x, y) = (2, 15)
      have h1 : x = 2 := by cases h'; rfl
      have h2 : y = 15 := by cases h'; rfl
      rw h1
      rw h2
      simp
    case inr hinr =>
      cases hinr
      case inl h' => -- Case when (x, y) = (3, 35)
        have h1 : x = 3 := by cases h'; rfl
        have h2 : y = 35 := by cases h'; rfl
        rw h1
        rw h2
        simp
      case inr hinr =>
        cases hinr
        case inl h' => -- Case when (x, y) = (4, 69)
          have h1 : x = 4 := by cases h'; rfl
          have h2 : y = 69 := by cases h'; rfl
          rw h1
          rw h2
          simp
        case inr hinr =>
          cases hinr
          case inl h' => -- Case when (x, y) = (5, 119)
            have h1 : x = 5 := by cases h'; rfl
            have h2 : y = 119 := by cases h'; rfl
            rw h1
            rw h2
            simp
          case inr hinr =>
            cases hinr  -- No more cases
            contradiction

end polynomial_relation_l805_805634


namespace natalies_father_savings_l805_805405

-- Definition of the conditions given in the problem
def total_savings (T : ℝ) : Prop :=
  let natalie_share := T / 2 in
  let remaining_after_natalie := T - natalie_share in
  let rick_share := 0.6 * remaining_after_natalie in
  let lucy_share := remaining_after_natalie - rick_share in
  lucy_share = 2000

-- The theorem to prove
theorem natalies_father_savings : ∃ T : ℝ, total_savings T ∧ T = 10000 :=
by
  sorry

end natalies_father_savings_l805_805405


namespace no_snow_probability_l805_805799

theorem no_snow_probability :
  let p_snow := 3 / 4
  let p_no_snow := 1 - p_snow
  (p_no_snow ^ 3) = (1 / 64) :=
by
  let p_snow := 3 / 4
  let p_no_snow := 1 - p_snow
  have : p_no_snow = 1 / 4 := by norm_num
  have : (p_no_snow ^ 3) = (1 / 64) := by norm_num
  exact this

end no_snow_probability_l805_805799


namespace median_of_first_twelve_positive_integers_l805_805069

def sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def median (s : List ℕ) : ℚ :=
  if h : (List.length s) % 2 = 0 then
    let k := List.length s / 2
    (s.get (k - 1) + s.get k) / 2
  else
    s.get (List.length s / 2)

theorem median_of_first_twelve_positive_integers :
  median sequence = 6.5 := 
sorry

end median_of_first_twelve_positive_integers_l805_805069


namespace count_integer_values_l805_805254

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805254


namespace quadratic_real_roots_l805_805691

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k^2 * x^2 - (2 * k + 1) * x + 1 = 0 ∧ ∃ x2 : ℝ, k^2 * x2^2 - (2 * k + 1) * x2 + 1 = 0)
  ↔ (k ≥ -1/4 ∧ k ≠ 0) := 
by 
  sorry

end quadratic_real_roots_l805_805691


namespace theorem1_theorem2_l805_805574

theorem theorem1 :
  (cbrt 64) * (100:ℝ)^(-1/2) * (0.25:ℝ)^(-3) * ((16/81):ℝ)^(-3/4) = (432/5:ℝ) := by
  sorry

theorem theorem2 :
  2 * Real.log10 (5/3) - Real.log10 (7/4) + 2 * Real.log10 3 + (1/2) * Real.log10 49 = 2 := by
  sorry

end theorem1_theorem2_l805_805574


namespace box_volume_l805_805500

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l805_805500


namespace same_height_2_2_l805_805790

def height (a h t T : ℝ) : ℝ := a * (t - T) ^ 2 + h

theorem same_height_2_2 (a h : ℝ) (ha : a ≠ 0) : 
  ∃ t, height a h t 1.2 = height a h (t - 2) 1.2 := sorry

/-
Proof goal: show that there exists t such that the heights of the two balls are equal.
-/

end same_height_2_2_l805_805790


namespace correct_result_l805_805322

theorem correct_result (x : ℕ) (h : x + 65 = 125) : x + 95 = 155 :=
sorry

end correct_result_l805_805322


namespace division_of_fractions_l805_805476

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805476


namespace alberto_bjorn_difference_l805_805702

theorem alberto_bjorn_difference :
  ∀ (d_alberto d_bjorn : ℕ), d_alberto = 75 → d_bjorn = 65 → d_alberto - d_bjorn = 10 :=
by
  intros d_alberto d_bjorn H_alberto H_bjorn
  rw [H_alberto, H_bjorn]
  sorry

end alberto_bjorn_difference_l805_805702


namespace calculate_code_count_div_10_l805_805860

/-- Definitions for allowed characters and their occurrence limits. -/
def allowed_characters : List Char := ['C', 'O', 'D', 'E', '1', '3', '3', '7']

/-- Constraint: Each character must appear no more frequently than they are allowed. -/
def valid_code (code : List Char) : Prop :=
  ∀ c, c ∈ allowed_characters →
    code.count c ≤ allowed_characters.count c

/-- The set of all valid codes consists of sequences of 6 characters from the allowed list. -/
def all_valid_codes : Finset (List Char) :=
  { code ∈ (List.replicateM 6 allowed_characters).toFinset | valid_code code }

/-- M is the number of valid codes where each possible code appears exactly once. -/
def M : ℕ := all_valid_codes.card

/-- Final statement: The number of codes divided by 10 is 1044. -/
theorem calculate_code_count_div_10 : M / 10 = 1044 :=
by
  sorry

end calculate_code_count_div_10_l805_805860


namespace num_real_solutions_eq_4_l805_805179

theorem num_real_solutions_eq_4 :
  (λ x : ℝ, (5 * x) / (x^2 + x + 1) + (6 * x) / (x^2 - 6 * x + 2) = -2) →
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    (5 * x₁) / (x₁^2 + x₁ + 1) + (6 * x₁) / (x₁^2 - 6 * x₁ + 2) = -2 ∧
    (5 * x₂) / (x₂^2 + x₂ + 1) + (6 * x₂) / (x₂^2 - 6 * x₂ + 2) = -2 ∧
    (5 * x₃) / (x₃^2 + x₃ + 1) + (6 * x₃) / (x₃^2 - 6 * x₃ + 2) = -2 ∧
    (5 * x₄) / (x₄^2 + x₄ + 1) + (6 * x₄) / (x₄^2 - 6 * x₄ + 2) = -2 
    ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄.
Proof : sorry

end num_real_solutions_eq_4_l805_805179


namespace abs_sub_abs_eq_six_l805_805326

theorem abs_sub_abs_eq_six
  (a b : ℝ)
  (h₁ : |a| = 4)
  (h₂ : |b| = 2)
  (h₃ : a * b < 0) :
  |a - b| = 6 :=
sorry

end abs_sub_abs_eq_six_l805_805326


namespace height_difference_correct_l805_805465

-- Definitions for the problem
def pipe_diameter : ℝ := 8
def crate_A_packing_height (rows : ℕ) : ℝ := rows * pipe_diameter
def crate_B_vertical_spacing : ℝ := pipe_diameter * (real.sqrt 3) / 2 
def crate_B_packing_height (rows : ℕ) : ℝ := crate_B_vertical_spacing * (rows - 1) + (2 * pipe_radius)
def crate_top_layer_height : ℝ := pipe_diameter
def full_row_pipes : ℕ := 10
def additional_pipes : ℕ := 5

-- Total height calculation
def total_height_crate_A (rows : ℕ) : ℝ :=
  crate_A_packing_height rows + crate_top_layer_height
def total_height_crate_B (rows : ℕ) : ℝ :=
  crate_B_packing_height rows + crate_top_layer_height

-- Positive height difference
def height_difference (rows : ℕ) : ℝ :=
  abs( total_height_crate_A rows - total_height_crate_B rows )

-- Main theorem to be proven
theorem height_difference_correct : height_difference 16 = 120 - 60 * real.sqrt 3 := 
by 
  -- Proof omitted
  sorry

end height_difference_correct_l805_805465


namespace calculate_expression_l805_805579

-- Definitions based on conditions
def step1 : Int := 12 - (-18)
def step2 : Int := step1 + (-7)
def final_result : Int := 23

-- Theorem to prove
theorem calculate_expression : step2 = final_result := by
  have h1 : step1 = 12 + 18 := by sorry
  have h2 : step2 = step1 - 7 := by sorry
  rw [h1, h2]
  norm_num
  sorry

end calculate_expression_l805_805579


namespace no_five_consecutive_integers_with_fourth_powers_sum_l805_805951

theorem no_five_consecutive_integers_with_fourth_powers_sum:
  ∀ n : ℤ, n^4 + (n + 1)^4 + (n + 2)^4 + (n + 3)^4 ≠ (n + 4)^4 :=
by
  intros
  sorry

end no_five_consecutive_integers_with_fourth_powers_sum_l805_805951


namespace length_of_other_train_l805_805108

-- Definitions based on the conditions
def length_train1 : ℝ := 360
def speed_train1_kmph : ℝ := 120
def speed_train2_kmph : ℝ := 80
def crossing_time : ℝ := 9

-- Convert speeds from km/h to m/s
def speed_train1_mps : ℝ := speed_train1_kmph * 1000 / 3600
def speed_train2_mps : ℝ := speed_train2_kmph * 1000 / 3600

-- Relative speed when moving in opposite directions
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Distance covered when the trains cross each other
def total_distance : ℝ := relative_speed * crossing_time

-- The length of the other train (L)
theorem length_of_other_train : total_distance = length_train1 + 140.95 :=
by
  have h1 : speed_train1_mps = 120 * 1000 / 3600 := rfl
  have h2 : speed_train2_mps = 80 * 1000 / 3600 := rfl
  have h3 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600) := rfl
  have h4 : total_distance = relative_speed * 9 := rfl
  sorry

end length_of_other_train_l805_805108


namespace correct_statement_is_C_l805_805901

theorem correct_statement_is_C :
  let data1 := [2, 3, 4, 5]
  let data2 := [4, 6, 8, 10]
  let mode_A := [5, 4, 4, 3, 5, 2]
  ¬(mode_A.mode = [4]) ∧ 
  ¬(∀ (d : List ℝ), stddev d = (variance d)^2) ∧
  (stddev data1 = 0.5 * stddev data2) ∧
  ¬(∀ histogram : List ℕ, ∀ (x : ℕ), x ∈ histogram ↔ area_of_rectangle x = frequency x)
  → true :=
by
  sorry

end correct_statement_is_C_l805_805901


namespace median_eq_6point5_l805_805066
open Nat

def median_first_twelve_positive_integers (l : List ℕ) : ℝ :=
  (l !!! 5 + l !!! 6) / 2

theorem median_eq_6point5 : median_first_twelve_positive_integers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 6.5 :=
by
  sorry

end median_eq_6point5_l805_805066


namespace problem_intersecting_lines_l805_805018

theorem problem_intersecting_lines (c d : ℝ) :
  (3 : ℝ) = (1 / 3 : ℝ) * (6 : ℝ) + c ∧ (6 : ℝ) = (1 / 3 : ℝ) * (3 : ℝ) + d → c + d = 6 :=
by
  intros h
  sorry

end problem_intersecting_lines_l805_805018


namespace exists_subgraph_with_min_deg_l805_805385

variables {G : Type*} [Graph G]
variables {d k : ℕ}

theorem exists_subgraph_with_min_deg (h_d_ge_3 : d ≥ 3) (h_delta_G : δ(G) ≥ d) (h_girth_G : g(G) ≥ 8 * k + 3) :
  ∃ (H : subgraph G), δ(H) ≥ d * (d - 1)^k := 
sorry

end exists_subgraph_with_min_deg_l805_805385


namespace radius_inscribed_sphere_quadrilateral_pyramid_l805_805453

noncomputable def radius_of_inscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt 5 - 1) / 4

theorem radius_inscribed_sphere_quadrilateral_pyramid (a : ℝ) :
  r = radius_of_inscribed_sphere a :=
by
  -- problem conditions:
  -- side of the base a
  -- height a
  -- result: r = a * (Real.sqrt 5 - 1) / 4
  sorry

end radius_inscribed_sphere_quadrilateral_pyramid_l805_805453


namespace time_to_cross_approx_l805_805838

-- Define train length, tunnel length, speed in km/hr, conversion factors, and the final equation
def length_of_train : ℕ := 415
def length_of_tunnel : ℕ := 285
def speed_in_kmph : ℕ := 63
def km_to_m : ℕ := 1000
def hr_to_sec : ℕ := 3600

-- Convert speed to m/s
def speed_in_mps : ℚ := (speed_in_kmph * km_to_m) / hr_to_sec

-- Calculate total distance
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Calculate the time to cross the tunnel in seconds
def time_to_cross : ℚ := total_distance / speed_in_mps

theorem time_to_cross_approx : abs (time_to_cross - 40) < 0.1 :=
sorry

end time_to_cross_approx_l805_805838


namespace median_first_twelve_pos_integers_l805_805077

theorem median_first_twelve_pos_integers : 
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (lst.nth 5 + lst.nth 6) / 2 = 6.5 := by
  sorry

end median_first_twelve_pos_integers_l805_805077


namespace PetersMotherAge_l805_805709

namespace AgeProblem

variable (P M : ℕ)
constant Harriet_age : ℕ := 13 -- Harriet's current age

/-- Defining conditions based on the problem statement -/
axiom condition1 : P + 4 = 2 * (Harriet_age + 4) -- In 4 years, Peter's age is twice Harriet's age
axiom condition2 : P = M / 2 -- Peter's age is half of his mother's age

/-- The goal is to prove that Peter's mother's age is 60 -/
theorem PetersMotherAge : M = 60 :=
by
  sorry
  
end AgeProblem

end PetersMotherAge_l805_805709


namespace ribbon_tape_length_l805_805718

theorem ribbon_tape_length
  (one_ribbon: ℝ)
  (remaining_cm: ℝ)
  (num_ribbons: ℕ)
  (total_used: ℝ)
  (remaining_meters: remaining_cm = 0.50)
  (ribbon_meter: one_ribbon = 0.84)
  (ribbons_made: num_ribbons = 10)
  (used_len: total_used = one_ribbon * num_ribbons):
  total_used + 0.50 = 8.9 :=
by
  sorry

end ribbon_tape_length_l805_805718


namespace least_common_multiple_of_812_and_3214_is_correct_l805_805825

def lcm_812_3214 : ℕ :=
  Nat.lcm 812 3214

theorem least_common_multiple_of_812_and_3214_is_correct :
  lcm_812_3214 = 1304124 := by
  sorry

end least_common_multiple_of_812_and_3214_is_correct_l805_805825


namespace division_of_fractions_l805_805477

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805477


namespace find_lengths_of_PC_and_NC_l805_805347

section TriangularPrism
variables (A B C A1 B1 C1 : Point)
variable (P : Point)
variable (AA1_length : ℝ)
variable (AB_length : ℝ)
variable (path_length : ℝ)

def midpoint (A A1 : Point) : Point := sorry -- Implement midpoint calculation
def shortest_path_exists (P : Point) (M : Point) (path_length : ℝ) : Prop := sorry -- Proof for shortest path

theorem find_lengths_of_PC_and_NC
  (h_prism : is_regular_triangular_prism ABC A1 B1 C1)
  (h_AB : AB_length = 3)
  (h_AA1 : AA1_length = 4)
  (h_midpoint : M = midpoint A A1)
  (h_P_on_BC : is_on BC P)
  (h_path : shortest_path_exists P M (sqrt 29)) :
  PC = 2 ∧ NC = 4 / 5 :=
sorry

end TriangularPrism

end find_lengths_of_PC_and_NC_l805_805347


namespace find_constants_l805_805398

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B Q : V)
variable (s v : ℝ)

theorem find_constants
  (h : ∃ (m n : ℝ), m = 5 ∧ n = 2 ∧ Q = (m * B - n * A) / (m - n)) :
  Q = s * A + v * B ↔ s = -2/3 ∧ v = 5/3 := 
by 
  obtain ⟨m, n, hm, hn, hQ⟩ := h
  simp [hm, hn] at hQ
  sorry

end find_constants_l805_805398


namespace pooh_guarantees_win_l805_805819

-- Condition Definitions
def initial_pine_cones := 2012
def winnies_moves := {1, 4}
def eeyores_moves := {1, 3}

-- Theorem Statement
theorem pooh_guarantees_win : winnie_can_guarantee_win initial_pine_cones winnies_moves eeyores_moves := sorry

end pooh_guarantees_win_l805_805819


namespace calculate_p_q_l805_805929

def U_inscribed_in_circle := True -- Given that U is an equilateral triangle inscribed in F
def F_radius : ℝ := 15 -- Circle F has a radius of 15 units
def G : ℝ := 5 -- Circle G with radius 5 units is tangential to F at a vertex of U
def H : ℝ := 3 -- Circle H with radius 3 units is tangential to F at another vertex of U
def I : ℝ := 2 -- Circle I with radius 2 units is tangential to F at another vertex of U
def J_tangent_to_G_H_I := True -- Circles G, H, I are externally tangent to circle J
def J_radius (r : ℝ) := r = 7 -- Calculating J's radius

theorem calculate_p_q : ∃ p q : ℕ, p + q = 8 ∧ gcd p q = 1 ∧ (15 - 5 = 10) ∧ (15 - 3 = 12) ∧ (15 - 2 = 13) := 
by
-- We outline the conditions in a proof
  have h1 := (F_radius - G = 10),
  have h2 := (F_radius - H = 12),
  have h3 := (F_radius - I = 13),
-- J's radius r calculated to be 7 ⇒ p/q equivalent to 7/1
  use [7, 1],
  repeat { sorry },
  exact ⟨rfl, ⟨gcd_one_left 7⟩⟩

end calculate_p_q_l805_805929


namespace compare_abc_l805_805384

noncomputable def a : ℝ := Real.exp (Real.sqrt Real.pi)
noncomputable def b : ℝ := Real.sqrt Real.pi + 1
noncomputable def c : ℝ := (Real.log Real.pi) / Real.exp 1 + 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l805_805384


namespace lim_n_div_x_lim_n_sq_div_x_l805_805128

def sequence (x : ℕ → ℝ) : Prop :=
  x 1 = real.sqrt 6 ∧ ∀ n : ℕ, x (n + 1) = x n + 3 * real.sqrt (x n) + n / real.sqrt (x n)

theorem lim_n_div_x {x : ℕ → ℝ} (hx : sequence x) : 
  tendsto (λ n, (n : ℝ) / x n) at_top (nhds 0) :=
sorry

theorem lim_n_sq_div_x {x : ℕ → ℝ} (hx : sequence x) : 
  tendsto (λ n, (n : ℝ)^2 / x n) at_top (nhds (4 / 9)) :=
sorry

end lim_n_div_x_lim_n_sq_div_x_l805_805128


namespace sand_loss_l805_805550

variable (initial_sand : ℝ) (final_sand : ℝ)

theorem sand_loss (h1 : initial_sand = 4.1) (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
  -- With the given conditions we'll prove this theorem
  sorry

end sand_loss_l805_805550


namespace grid_two_colors_intersection_l805_805033

theorem grid_two_colors_intersection :
  ∀ (color : ℕ → ℕ → bool), 
  (∃ h₁ h₂ v₁ v₂ : ℕ, 
    h₁ ≠ h₂ ∧ v₁ ≠ v₂ ∧ 
    color h₁ v₁ = color h₁ v₂ ∧ 
    color h₁ v₂ = color h₂ v₁ ∧ 
    color h₂ v₁ = color h₂ v₂) :=
by
sor<SIL>Ty

end grid_two_colors_intersection_l805_805033


namespace pizza_slices_left_l805_805507

def TotalSlices : ℕ := 16
def SlicesEatenAtDinner (total : ℕ) : ℕ := total / 4
def SlicesLeftAfterDinner (total : ℕ) (eatenDinner : ℕ) : ℕ := total - eatenDinner
def SlicesEatenByYves (leftAfterDinner : ℕ) : ℕ := leftAfterDinner / 4
def SlicesLeftAfterYves (leftAfterDinner : ℕ) (eatenYves : ℕ) : ℕ := leftAfterDinner - eatenYves
def SlicesEatenBySiblings : ℕ := 2 + 2
def SlicesLeftAfterSiblings (leftAfterYves : ℕ) (eatenSiblings : ℕ) : ℕ := leftAfterYves - eatenSiblings

theorem pizza_slices_left :
  let eatenDinner := SlicesEatenAtDinner TotalSlices,
      leftAfterDinner := SlicesLeftAfterDinner TotalSlices eatenDinner,
      eatenYves := SlicesEatenByYves leftAfterDinner,
      leftAfterYves := SlicesLeftAfterYves leftAfterDinner eatenYves,
      leftAfterSiblings := SlicesLeftAfterSiblings leftAfterYves SlicesEatenBySiblings
  in leftAfterSiblings = 5 := by
  sorry

end pizza_slices_left_l805_805507


namespace num_integer_solutions_l805_805289

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805289


namespace average_age_after_swap_l805_805002

theorem average_age_after_swap :
  let initial_average_age := 28
  let num_people_initial := 8
  let person_leaving_age := 20
  let person_entering_age := 25
  let initial_total_age := initial_average_age * num_people_initial
  let total_age_after_leaving := initial_total_age - person_leaving_age
  let total_age_final := total_age_after_leaving + person_entering_age
  let num_people_final := 8
  initial_average_age / num_people_initial = 28 ->
  total_age_final / num_people_final = 28.625 :=
by
  intros
  sorry

end average_age_after_swap_l805_805002


namespace travelers_same_elevation_l805_805752

variable {ℝ : Type} [LinearOrder ℝ]

-- Let's define the concept of a point having a specific elevation.
variable (A B : ℝ) -- Elevations of points A and B.
variable (trail : List ℝ) -- List of elevation points for the trail.

-- Conditions:
axiom h1 : A = B -- Points A and B are at the same elevation.
axiom h2 : ∀ p ∈ trail, p > A -- Each peak on the trail is higher than points A and B.
axiom h3 : (trail.head? = some A) ∧ (trail.last? = some B)
-- The trail starts at point A and ends at point B.

-- Proof Goal: Can two travelers always maintain the same elevation while moving along the trail?
theorem travelers_same_elevation (A B : ℝ) (trail : List ℝ) :
  A = B → (∀ p ∈ trail, p > A) → (trail.head? = some A) → (trail.last? = some B) →
  ∃ times : List ℝ, ∀ t ∈ times, ∃ (X Y : ℝ), X = Y ∧ X = A ∧ Y = B :=
by
  intros h1 h2 h3 h4
  sorry

end travelers_same_elevation_l805_805752


namespace polygon_diagonals_subtract_sides_l805_805147

theorem polygon_diagonals_subtract_sides (n : ℕ) (h : n = 105) : (n * (n - 3)) / 2 - n = 5250 := by
  -- Here, n is given as 105
  rw [h]
  -- left to prove the statement with n = 105
  sorry

end polygon_diagonals_subtract_sides_l805_805147


namespace sequence_general_formula_l805_805921

theorem sequence_general_formula (a : ℕ → ℕ) (h : ∀ n, n / (a 1 + a 2 + ... + a n : ℕ) = 1 / (2 * n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end sequence_general_formula_l805_805921


namespace candy_remaining_l805_805953

theorem candy_remaining (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) (initial_candy := katie_candy + sister_candy) :
  katie_candy = 10 → sister_candy = 6 → eaten_candy = 9 → initial_candy - eaten_candy = 7 :=
by
  intros h_katie h_sister h_eaten
  rw [h_katie, h_sister, h_eaten]
  simp
  sorry

end candy_remaining_l805_805953


namespace linear_function_iff_l805_805832

variable {x : ℝ} (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x + 4 * x - 5

theorem linear_function_iff (m : ℝ) : 
  (∃ c d, ∀ x, f m x = c * x + d) ↔ m ≠ -6 :=
by 
  sorry

end linear_function_iff_l805_805832


namespace integer_count_satisfies_inequality_l805_805292

theorem integer_count_satisfies_inequality :
  (card {n : ℤ | -100 < n^3 ∧ n^3 < 100}) = 9 :=
by
  sorry

end integer_count_satisfies_inequality_l805_805292


namespace find_a_in_polynomial_l805_805212

noncomputable def polynomial_with_rational_roots : Prop :=
  ∃ (a b : ℚ), (is_root (λ x : ℚ, x^3 + a * x^2 + b * x + 90) (-3 - 5 * real.sqrt 3)) ∧
               a = -15 / 11

theorem find_a_in_polynomial : polynomial_with_rational_roots :=
  sorry

end find_a_in_polynomial_l805_805212


namespace linear_function_through_points_and_value_l805_805220

theorem linear_function_through_points_and_value (x y : ℝ)
  (h1 : ∀ x, y = (3 * x) + 2)
  (h2 : (1, 5))
  (h3 : (-1, -1))
  : ∃ x = 2, y = 8 :=
by
assume (x = 2)
show y = (3 * 2) + 2
from
calc
  y = (3 * 2) + 2 : by { sorry }

end linear_function_through_points_and_value_l805_805220


namespace count_n_integers_l805_805279

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805279


namespace sum_of_x_coordinates_of_points_above_line_l805_805407

theorem sum_of_x_coordinates_of_points_above_line :
  let points := [(3, 11), (7, 19), (13, 32), (18, 38), (21, 48)] in
  let line := λ x, 2 * x + 5 in
  (points.filter (λ p, p.snd > line p.fst)).map (λ p, p.fst).sum = 34 :=
by
  -- Define the line equation
  let points := [(3, 11), (7, 19), (13, 32), (18, 38), (21, 48)] in
  let line := λ x, 2 * x + 5 in
  -- Filter points above the line
  have above_line_points := points.filter (λ p, p.snd > line p.fst) in
  -- Map to x-coordinates and sum
  let sum_x := above_line_points.map (λ p, p.fst).sum in
  -- Prove the sum is 34
  have h_sum : sum_x = 34 := sorry
  exact h_sum
sorry

end sum_of_x_coordinates_of_points_above_line_l805_805407


namespace geometric_sequence_fourth_term_l805_805784

theorem geometric_sequence_fourth_term (a r T4 : ℝ)
  (h1 : a = 1024)
  (h2 : a * r^5 = 32)
  (h3 : T4 = a * r^3) :
  T4 = 128 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l805_805784


namespace angle_A_eq_pi_div_3_area_of_ABC_eq_3sqrt3_l805_805696

/- Problem 1 -/
theorem angle_A_eq_pi_div_3 
  (a c A : ℝ) (C : ℝ) (h : a * sin C = sqrt 3 * c * cos A): 
  A = π / 3 := 
sorry

/- Problem 2 -/
theorem area_of_ABC_eq_3sqrt3 
  (b : ℝ) (h_b : b = 4)
  (a : ℝ) (h_a : a = sqrt 13)
  (c : ℝ) (h_c : c = 3)
  (A : ℝ) (h_A : A = π / 3) :
  1 / 2 * b * c * sin A = 3 * sqrt 3 := 
sorry

end angle_A_eq_pi_div_3_area_of_ABC_eq_3sqrt3_l805_805696


namespace find_original_price_l805_805126

-- Definitions for the conditions mentioned in the problem
variables {P : ℝ} -- Original price per gallon in dollars

-- Proof statement assuming the given conditions
theorem find_original_price 
  (h1 : ∃ P : ℝ, P > 0) -- There exists a positive price per gallon in dollars
  (h2 : (250 / (0.9 * P)) = (250 / P + 5)) -- After a 10% price reduction, 5 gallons more can be bought for $250
  : P = 25 / 4.5 := -- The solution states the original price per gallon is approximately $5.56
by
  sorry -- Proof omitted

end find_original_price_l805_805126


namespace alice_has_winning_strategy_l805_805556

def edge_directed (G: Type) [graph G] (u v: G) (dir: bool) := sorry -- Placeholder for edge direction

structure complete_graph (V : Type) :=
(edges : set (V × V))
(completeness : ∀ u v : V, u ≠ v → (u, v) ∈ edges)

def has_cycle (G: Type) [graph G] : Prop := sorry -- Placeholder for cycle detection

noncomputable def alice_winning_strategy : Prop :=
∀ (G : complete_graph (fin 2014)), ∃ (steps: nat), 
  -- Alice wins if she has a winning strategy within required steps without Bob blocking all edges
  (∀ (turns: steps), ∃ dir: bool, (∀ u v : fin 2014, edge_directed G u v dir)) ∧ 
  has_cycle G

theorem alice_has_winning_strategy : alice_winning_strategy := sorry

end alice_has_winning_strategy_l805_805556


namespace count_int_values_cube_bound_l805_805312

theorem count_int_values_cube_bound : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.finite ∧ {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by
  sorry

end count_int_values_cube_bound_l805_805312


namespace simplify_sqrt_is_cos_20_l805_805764

noncomputable def simplify_sqrt : ℝ :=
  let θ : ℝ := 160 * Real.pi / 180
  Real.sqrt (1 - Real.sin θ ^ 2)

theorem simplify_sqrt_is_cos_20 : simplify_sqrt = Real.cos (20 * Real.pi / 180) :=
  sorry

end simplify_sqrt_is_cos_20_l805_805764


namespace area_between_lines_l805_805922

-- Definitions of the lines
def line1 (x : ℝ) : ℝ := -1/2 * x + 3
def line2 (x : ℝ) : ℝ := -3/10 * x + 5

-- Prove the area between line1 and line2 from x = 0 to x = 5
theorem area_between_lines : 
  (∫ x in 0..5, line2 x - line1 x) = 10 :=
by
  sorry

end area_between_lines_l805_805922


namespace dog_roaming_area_comparison_l805_805399

theorem dog_roaming_area_comparison :
  let r := 10
  let a1 := (1/2) * Real.pi * r^2
  let a2 := (3/4) * Real.pi * r^2 - (1/4) * Real.pi * 6^2 
  a2 > a1 ∧ a2 - a1 = 16 * Real.pi :=
by
  sorry

end dog_roaming_area_comparison_l805_805399


namespace nine_numbers_sum_200_not_always_four_sum_gt_100_l805_805809

theorem nine_numbers_sum_200_not_always_four_sum_gt_100 (nums : Fin 9 → ℕ) (h_sum : (∑ i, nums i) = 200) (h_distinct : Function.Injective nums) :
  ¬ (∀ (A : Finset (Fin 9)), A.card = 4 → (∑ i in A, nums i) > 100) :=
by
  sorry

end nine_numbers_sum_200_not_always_four_sum_gt_100_l805_805809


namespace range_of_m_l805_805643

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → |((x2^2 - m * x2) - (x1^2 - m * x1))| ≤ 9) →
  -5 / 2 ≤ m ∧ m ≤ 13 / 2 :=
sorry

end range_of_m_l805_805643


namespace count_n_integers_l805_805280

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805280


namespace median_eq_6point5_l805_805067
open Nat

def median_first_twelve_positive_integers (l : List ℕ) : ℝ :=
  (l !!! 5 + l !!! 6) / 2

theorem median_eq_6point5 : median_first_twelve_positive_integers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 6.5 :=
by
  sorry

end median_eq_6point5_l805_805067


namespace distinct_floors_l805_805795

theorem distinct_floors (N M : ℕ) (a : ℝ) (h1 : N > 0) (h2 : M > 1) :
  (∀ k l : ℕ, 1 ≤ k ∧ k ≤ N ∧ 1 ≤ l ∧ l ≤ N ∧ k ≠ l → ⌊(k : ℝ) * a⌋ ≠ ⌊(l : ℝ) * a⌋) ∧
  (∀ k l : ℕ, 1 ≤ k ∧ k ≤ M ∧ 1 ≤ l ∧ l ≤ M ∧ k ≠ l → ⌊(k : ℝ) / a⌋ ≠ ⌊(l : ℝ) / a⌋) ↔
  (frac ((N - 1) : ℝ) / N) ≤ a ∧ a ≤ frac (M / (M - 1)) := 
sorry

end distinct_floors_l805_805795


namespace find_multiple_l805_805431

theorem find_multiple (P : ℝ) (hP : P = 38) (A : ℝ) 
    (hA : A = (P / 4) ^ 2) (hA_eq : A = m * P + 14.25) :
    m = 2 :=
by
    have s : ℝ := P / 4
    have hA_val : A = s^2, from calc
        A = (P / 4) ^ 2 : by rw [<- hA]
          = (38 / 4) ^ 2 : by rw hP
          = 9.5 ^ 2 : by norm_num
          = 90.25 : by norm_num
    have h_eq : 90.25 = m * 38 + 14.25, from calc
        90.25 = 9.5 ^ 2 : by rw [<- hA_val]
              = A : by rw [hA]
              = m * 38 + 14.25 : by rw [hA_eq]
    have m_eq : m = 2, from calc
        m * 38 + 14.25 = 90.25 : by rw [<- h_eq]
        m * 38 = 90.25 - 14.25 : by linarith
        m = (90.25 - 14.25) / 38 : by field_simp
        m = 2 : by norm_num
    exact m_eq

end find_multiple_l805_805431


namespace price_difference_in_cents_l805_805993

noncomputable def list_price : ℝ := 49.99
noncomputable def discount_every_penny_counts : ℝ := 7
noncomputable def discount_save_a_lot : ℝ := 0.20

noncomputable def price_every_penny_counts : ℝ := list_price - discount_every_penny_counts
noncomputable def price_save_a_lot : ℝ := list_price * (1 - discount_save_a_lot)

noncomputable def price_difference : ℝ := price_every_penny_counts - price_save_a_lot

theorem price_difference_in_cents : (price_difference * 100).round = 299 := 
by 
  sorry

end price_difference_in_cents_l805_805993


namespace triangle_eq_segments_l805_805372

theorem triangle_eq_segments
  (A B C D E : Type)
  [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C]
  [EuclideanSpace ℝ D] [EuclideanSpace ℝ E]
  -- Given conditions
  (angle_b_eq_120deg : ∠B = 120)
  (d_on_b_angle_bisector : ∀ x, x = D → angle (A, D, B) = 2 * angle(A, C, B))
  (ae_eq_ad : AE = AD)
  -- To prove: EC = ED
  : EC = ED := 
by
  sorry

end triangle_eq_segments_l805_805372


namespace memorable_telephone_numbers_count_l805_805584

theorem memorable_telephone_numbers_count :
  let digits := fin 10
  let is_memorable (d : array 9 digits) := 
    (d[0] = d[6] ∧ d[1] = d[7] ∧ d[2] = d[8]) ∨
    (d[1] = d[6] ∧ d[2] = d[7] ∧ d[3] = d[8]) ∨
    (d[2] = d[6] ∧ d[3] = d[7] ∧ d[4] = d[8])
  finset.card (finset.filter is_memorable (finset.univ : finset (array 9 digits))) = 101000 :=
sorry

end memorable_telephone_numbers_count_l805_805584


namespace tangent_line_at_x1_max_min_values_on_interval_l805_805659

def f (x : ℝ) : ℝ := x * (2 * x^2 - 3 * x - 12) + 5

theorem tangent_line_at_x1 : ∃ C : ℝ, (f 1 = -8) ∧ (f' 1 = -12) ∧ (∀ y, y = f 1 + f' 1 * (1 - x) ↔ C = 12 * x + y - 4) := 
sorry

theorem max_min_values_on_interval : 
  ∃ max min : ℝ, max = 5 ∧ min = -15 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f(x) ≤ max ∧ f(x) ≥ min) :=
sorry

end tangent_line_at_x1_max_min_values_on_interval_l805_805659


namespace average_apples_per_hour_l805_805403

theorem average_apples_per_hour (total_apples : ℝ) (total_hours : ℝ) (h1 : total_apples = 5.0) (h2 : total_hours = 3.0) : total_apples / total_hours = 1.67 :=
  sorry

end average_apples_per_hour_l805_805403


namespace find_x_of_dot_product_l805_805676

theorem find_x_of_dot_product :
  ∀ (x : ℝ), let a := (1, -1 : ℝ × ℝ)
             let b := (2, x : ℝ × ℝ)
             (a.1 * b.1 + a.2 * b.2 = 1) → x = 1 :=
by
  intros x a b
  sorry

end find_x_of_dot_product_l805_805676


namespace stella_pays_1935_l805_805768

theorem stella_pays_1935
  (original_price : ℝ := 50)
  (initial_discount : ℝ := 0.30)
  (further_discount : ℝ := 0.20)
  (store_credit : ℝ := 10)
  (sales_tax_rate : ℝ := 0.075) :
  let discounted_price1 := original_price * (1 - initial_discount),
      discounted_price2 := discounted_price1 * (1 - further_discount),
      price_after_credit := discounted_price2 - store_credit,
      total_price := price_after_credit * (1 + sales_tax_rate) in
  total_price = 19.35 := by
    sorry

end stella_pays_1935_l805_805768


namespace count_integer_values_l805_805252

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805252


namespace ellipse_equation_find_m_l805_805224

theorem ellipse_equation
  (a b : ℝ) 
  (h_ab : a > b) 
  (h_b0 : b > 0) 
  (eccentricity : ℝ) 
  (h_ecc : eccentricity = real.sqrt 2 / 2) 
  (c : ℝ) 
  (h_c : c = 2) 
  (left_focus : ℝ × ℝ) 
  (h_focus : left_focus = (-2, 0)) :
  (∀ (x y : ℝ), (x^2 / (2 * real.sqrt 2)^2 + y^2 / 2^2 = 1) → (x^2 / 8 + y^2 / 4 = 1)) :=
sorry

theorem find_m 
  (m : ℝ)
  (line_eq : ℝ → ℝ)
  (h_line_eq : ∀ x, line_eq x = x + m)
  (ellipse_eq : ℝ → ℝ → Prop)
  (h_ellipse_eq : ∀ x y, ellipse_eq x y = (x^2 / 8 + y^2 / 4 = 1))
  (intersect_points : ℝ × ℝ → ℝ × ℝ)
  (midpoint : ℝ × ℝ)
  (h_midpoint : midpoint = ((λ (A B : ℝ × ℝ), ((A.1 + B.1)/2, (A.2 + B.2)/2)) (intersect_points.1) (intersect_points.2))) 
  (curve_eq : ℝ → ℝ → Prop)
  (h_curve_eq : ∀ x y, curve_eq x y = (x^2 + 2*y = 2)) :
  (m = 3/2 ∨ m = -3) :=
sorry

end ellipse_equation_find_m_l805_805224


namespace log3_applications_count_l805_805160

def tower_function : ℕ → ℕ
| 1     := 3
| (n+1) := 3^(tower_function n)

def A := tower_function 6 ^ tower_function 6
def B := tower_function 6 ^ A

theorem log3_applications_count :
  ∃ n, n = 7 ∧ ∀ m, (0 ≤ m && m < n) → is_defined (fun x => log_base 3 x)^(m) B :=
sorry

end log3_applications_count_l805_805160


namespace danny_marks_in_physics_l805_805159

theorem danny_marks_in_physics (marks_eng : ℕ) (marks_math : ℕ) (marks_chem : ℕ) (marks_bio : ℕ) (avg_marks : ℕ) (num_subjects : ℕ) :
  marks_eng = 76 →
  marks_math = 65 →
  marks_chem = 67 →
  marks_bio = 75 →
  avg_marks = 73 →
  num_subjects = 5 →
  let total_marks := avg_marks * num_subjects in
  let marks_physics := total_marks - (marks_eng + marks_math + marks_chem + marks_bio) in
  marks_physics = 82 :=
by {
  intros,
  let total_marks := avg_marks * num_subjects,
  let marks_physics := total_marks - (marks_eng + marks_math + marks_chem + marks_bio),
  have : total_marks = 365 := by sorry,
  have : marks_eng + marks_math + marks_chem + marks_bio = 283 := by sorry,
  show marks_physics = 82,
  calc 
    marks_physics = total_marks - (marks_eng + marks_math + marks_chem + marks_bio) : by sorry
               ... = 365 - 283 : by sorry
               ... = 82 : by sorry
}

end danny_marks_in_physics_l805_805159


namespace division_of_fractions_l805_805478

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805478


namespace quadratic_roots_sum_product_l805_805335

theorem quadratic_roots_sum_product : 
  ∃ x1 x2 : ℝ, (x1^2 - 2*x1 - 4 = 0) ∧ (x2^2 - 2*x2 - 4 = 0) ∧ 
  (x1 ≠ x2) ∧ (x1 + x2 + x1 * x2 = -2) :=
sorry

end quadratic_roots_sum_product_l805_805335


namespace mean_indicator_not_improved_l805_805527

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def x_bar := 10.0
def y_bar := 10.3
def s1_sq := 0.036
def s2_sq := 0.04

theorem mean_indicator_not_improved :
  y_bar - x_bar < 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := by
  sorry

end mean_indicator_not_improved_l805_805527


namespace common_difference_arithmetic_sequence_l805_805349

-- Problem: In an arithmetic sequence with the initial term 3, the final term 50, and total sum of 265, prove that the common difference is 47/9.
theorem common_difference_arithmetic_sequence (a l S : ℤ) (d : ℚ) (n : ℕ) 
    (ha : a = 3) 
    (hl : l = 50) 
    (hS : S = 265)
    (hn_eq : n = ((2 * S) / (a + l)).to_nat) 
    (hl_eq : l = a + (n - 1) * d):
  d = 47 / 9 :=
by 
  sorry

end common_difference_arithmetic_sequence_l805_805349


namespace three_digit_multiple_of_three_probability_l805_805854

theorem three_digit_multiple_of_three_probability :
  let digits := {1, 2, 3, 4}
  let valid_combinations := [{1, 2, 3}, {2, 3, 4}]
  let total_ways := 4 * 3 * 2
  let valid_ways := 2 * 3!
  (valid_ways : ℚ) / total_ways = 1 / 2 :=
by
  intro digits valid_combinations total_ways valid_ways
  sorry

end three_digit_multiple_of_three_probability_l805_805854


namespace triangle_properties_l805_805340

theorem triangle_properties
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 2)
  (h2 : sin A = 2 * sin C)
  (h3 : cos B = 1 / 4) :
  a = 4 ∧ (1/2 * a * c * sin B = sqrt 15) :=
by
  sorry

end triangle_properties_l805_805340


namespace sachin_is_younger_than_rahul_by_18_years_l805_805416

-- Definitions based on conditions
def sachin_age : ℕ := 63
def ratio_of_ages : ℚ := 7 / 9

-- Assertion that based on the given conditions, Sachin is 18 years younger than Rahul
theorem sachin_is_younger_than_rahul_by_18_years (R : ℕ) (h1 : (sachin_age : ℚ) / R = ratio_of_ages) : R - sachin_age = 18 :=
by
  sorry

end sachin_is_younger_than_rahul_by_18_years_l805_805416


namespace rectangle_area_l805_805356

theorem rectangle_area (a b c d : ℝ) 
  (ha : a = 4) 
  (hb : b = 4) 
  (hc : c = 4) 
  (hd : d = 1) :
  ∃ E F G H : ℝ,
    (E = 0 ∧ F = 3 ∧ G = 4 ∧ H = 0) →
    (a + b + c + d) = 10 :=
by
  intros
  sorry

end rectangle_area_l805_805356


namespace count_integers_in_range_num_of_integers_l805_805270

theorem count_integers_in_range : 
  let count := (finset.Icc (-4) 4).card in
  -100 < n^3 ∧ n^3 < 100 ↔ n ∈ (int.range (-4) 5).toFinset

theorem num_of_integers : ∃ k : ℕ, k = 9 :=
begin
  let problem := { n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100 },
  have range_eq: {n | -100 < n ^ 3 ∧ n ^ 3 < 100} = 
    {n | -4 ≤ n ∧ n ≤ 4},
  { ext,
    simp,
    split; intro h,
    { split; omega, },
    { omega, } },
  let finset := { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
  have range_card : finset.card {n : ℤ | -4 ≤ n ∧ n ≤ 4} = 9,
  { refine finset.card_eq_of_bijective 
    ⟨λ x, ⟨x, _⟩, _, _⟩ ;
    simp },
  use 9,
  tauto
end

example : (finset.Icc (-4) 4).card = 9 := sorry

end count_integers_in_range_num_of_integers_l805_805270


namespace four_digit_factorial_sum_l805_805823

-- Define the factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Define a function to extract digits of a number
def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else 
    let last_digit := n % 10
    in last_digit :: digits (n / 10)

-- Sum of the factorials of the digits
def sum_factorial_digits (n : ℕ) : ℕ :=
  (digits n).map factorial |> List.sum

-- The main statement
theorem four_digit_factorial_sum : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n = sum_factorial_digits n ∧ (digits n).All (λ d => d ≤ 6) :=
by
  -- solution steps go here
  sorry

end four_digit_factorial_sum_l805_805823


namespace point_to_line_distance_l805_805781

-- Define the parameters of the problem
def point : ℝ × ℝ := (1, 1)
def line : ℝ × ℝ × ℝ := (1, 1, -1) -- coefficients (a, b, c) of the line equation ax + by + c = 0

-- The distance formula from a point (x1, y1) to a line ax + by + c = 0 is: 
-- d = |ax1 + by1 + c| / sqrt(a^2 + b^2)
def distance_from_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1) := p
  let (a, b, c) := l
  Abs (a * x1 + b * y1 + c) / Real.sqrt (a^2 + b^2)

-- Prove that the distance is equal to sqrt(2)/2
theorem point_to_line_distance : distance_from_point_to_line point line = Real.sqrt 2 / 2 := 
by
  sorry

end point_to_line_distance_l805_805781


namespace advanced_prime_looking_count_l805_805870

def is_composite (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

def is_advanced_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬(n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0)

def count_advanced_prime_looking (N : ℕ) : ℕ :=
  (Finset.range N).filter (λ n, 0 < n ∧ is_advanced_prime_looking n).card

theorem advanced_prime_looking_count : count_advanced_prime_looking 1200 = 519 := 
by sorry

end advanced_prime_looking_count_l805_805870


namespace min_average_annual_cost_l805_805869

noncomputable def purchase_cost := 10 -- in ten thousand yuan
noncomputable def annual_cost := 0.9 -- in ten thousand yuan

noncomputable def maintenance_cost (x : ℕ) : ℝ := 0.2 + (x - 1) * 0.2 -- in ten thousand yuan

-- Total maintenance cost over x years
noncomputable def total_maintenance_cost (x : ℕ) : ℝ :=
  (x * (0.2 + maintenance_cost x)) / 2

-- Average annual cost
noncomputable def average_annual_cost (x : ℕ) : ℝ :=
  (purchase_cost + annual_cost * x + total_maintenance_cost x) / x

theorem min_average_annual_cost : ∃ x : ℕ, average_annual_cost x = 1 + (10 / 10) + (10 / 10) ∧ x = 10 := 
  sorry

end min_average_annual_cost_l805_805869


namespace slope_angle_of_perpendicular_line_l805_805030

theorem slope_angle_of_perpendicular_line (h : ∀ x, x = (π / 3)) : ∀ θ, θ = (π / 2) := 
by 
  -- Placeholder for the proof
  sorry

end slope_angle_of_perpendicular_line_l805_805030


namespace inscribed_sphere_radius_proof_l805_805774

-- Given the conditions about the cone being divided, we define the parameters.
variables (a r α : ℝ)

-- Assuming the necessary geometric conditions.
axiom cone_geometry : ∀ {h : ℝ}, OC = a / 2 ∧ OD = a / 2 ∧ OP = a * real.sqrt 3 / 2

-- Assume the angle of the intersecting planes with the cone.
def intersecting_planes (α : ℝ) : Prop := α = real.arctan (1 / real.sqrt 2)

-- Define the relationship given above in text form for the radius r in terms of a.
def inscribed_sphere_radius (a r : ℝ) (α: ℝ) : Prop :=
  a = r * (real.sqrt 3 + real.sqrt 2)

-- The theorem statement proving the radius of the inscribed sphere.
theorem inscribed_sphere_radius_proof 
  (a : ℝ) (h1 : cone_geometry) (h2 : intersecting_planes α)  : 
  ∃ (r : ℝ), inscribed_sphere_radius a r α :=
begin
  use (a / (real.sqrt 3 + real.sqrt 2)), 
  sorry
end

end inscribed_sphere_radius_proof_l805_805774


namespace sum_ratio_ge_half_sum_l805_805199

variable (n : ℕ)
variable (a b : Fin n → ℝ)
variable (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
variable (h_sum_eq : (∑ i, a i) = (∑ i, b i))

theorem sum_ratio_ge_half_sum :
  (∑ i, a i ^ 2 / (a i + b i)) ≥ (1 / 2) * (∑ i, a i) :=
by
  sorry

end sum_ratio_ge_half_sum_l805_805199


namespace num_integer_solutions_l805_805283

theorem num_integer_solutions : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.to_finset.card = 9 := 
sorry

end num_integer_solutions_l805_805283


namespace number_of_fours_is_even_l805_805146

theorem number_of_fours_is_even 
  (x y z : ℕ) 
  (h1 : x + y + z = 80) 
  (h2 : 3 * x + 4 * y + 5 * z = 276) : 
  Even y :=
by
  sorry

end number_of_fours_is_even_l805_805146


namespace hyperbola_eccentricity_l805_805662

-- Conditions
variables (a b c : ℝ) (P : ℝ × ℝ)
variables (ha : a > 0) (hb : b > 0)
-- Definition of a hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
-- Point P = (c, 2b)
def point_P : ℝ × ℝ := (c, 2 * b)
-- E is the midpoint of EP
def midpoint_E (E : ℝ × ℝ) : Prop := E = (E.1 + c) / 2 ∧ E.2 = (E.2 + 2 * b) / 2
-- Eccentricity calculation
def eccentricity (x y : ℝ) : ℝ := sqrt (x^2 / a^2 + 1)

theorem hyperbola_eccentricity (hP : hyperbola c (2 * b)) : eccentricity c (2 * b) = sqrt 5 :=
by sorry

end hyperbola_eccentricity_l805_805662


namespace chameleon_cannot_repaint_checkerboard_l805_805111

variables (n : ℕ) (black white : ℕ)
variable (board : ℕ × ℕ → ℕ)

-- Conditions
def is_checkerboard_pattern (board : ℕ × ℕ → ℕ) : Prop :=
∀ (i j : ℕ), (i < 8 ∧ j < 8) → (board (i, j) = board (i+1, j+1) ∧ board (i, j) ≠ board (i+1, j) ∧ board (i, j) ≠ board (i, j+1))

def is_limping_rook_movement (i j : ℕ) : Prop :=
(i < 7 ∨ j < 7) ∧ (i * j = 0)

-- Initial Configuration
def initial_board : ℕ × ℕ → ℕ := λ (i j : ℕ), if (i + j) % 2 = 0 then black else white

def initial_position_color : ℕ := white

-- Proof Problem (Statement)
theorem chameleon_cannot_repaint_checkerboard (n : ℕ) :
  n = 8 → 
  initial_position_color = white →
  initial_board (0, 0) = black →
  ¬ (∃ (board : ℕ × ℕ → ℕ), is_checkerboard_pattern board) :=
begin
  sorry
end

end chameleon_cannot_repaint_checkerboard_l805_805111


namespace yellow_balls_l805_805110

-- Conditions
def total_balls : Nat := 100
def white_balls : Nat := 50
def green_balls : Nat := 30
def red_balls : Nat := 7
def purple_balls : Nat := 3
def probability_neither_red_nor_purple : Float := 0.9

-- Proof that the number of yellow balls is 10
theorem yellow_balls (Y : Nat) : 
  ((white_balls + green_balls + Y) / total_balls : Float) = probability_neither_red_nor_purple → Y = 10 :=
by 
  intro h
  have h1 : (white_balls + green_balls + Y) / total_balls = probability_neither_red_nor_purple := h
  sorry

end yellow_balls_l805_805110


namespace sum_reciprocals_square_l805_805450

theorem sum_reciprocals_square (x y : ℕ) (h : x * y = 11) : (1 : ℚ) / (↑x ^ 2) + (1 : ℚ) / (↑y ^ 2) = 122 / 121 :=
by
  sorry

end sum_reciprocals_square_l805_805450


namespace intervals_of_monotonicity_l805_805995

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem intervals_of_monotonicity :
  (∀ x ∈ set.Ioo (-∞:ℝ) 0, deriv f x > 0) ∧
  (∀ x ∈ set.Ioo (2:ℝ) ∞, deriv f x > 0) ∧
  (∀ x ∈ set.Ioo (0:ℝ) 2, deriv f x < 0) :=
by sorry

end intervals_of_monotonicity_l805_805995


namespace second_number_is_915_l805_805547

theorem second_number_is_915 :
  ∃ (n1 n2 n3 n4 n5 n6 : ℤ), 
    n1 = 3 ∧ 
    n2 = 915 ∧ 
    n3 = 138 ∧ 
    n4 = 1917 ∧ 
    n5 = 2114 ∧ 
    ∃ x: ℤ, 
      (n1 + n2 + n3 + n4 + n5 + x) / 6 = 12 ∧ 
      n2 = 915 :=
by 
  sorry

end second_number_is_915_l805_805547


namespace quadratic_two_distinct_real_roots_l805_805803

theorem quadratic_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, 2 * x^2 - 3 * x - (3 / 2) = 0 → x = x₁ ∨ x = x₂) :=
by
  let a := 2
  let b := -3
  let c := -3/2
  let Δ := b^2 - 4 * a * c
  have hΔ : Δ = 21 := by
    calc
      b^2 - 4 * a * c
      = (-3)^2 - 4 * 2 * (-3/2) : by sorry
      ... = 9 + 12 : by sorry
      ... = 21 : by sorry
  have hΔ_pos : Δ > 0 := by
    rw [hΔ]
    exact zero_lt_one_mul_pos.mpr (by norm_num)
  -- Since Δ > 0, there exist two distinct real roots
  use 0
  use 21
  split
  · sorry -- proof that x₁ ≠ x₂
  · sorry -- proof that x satisfies the equation if and only if it's x₁ or x₂

end quadratic_two_distinct_real_roots_l805_805803


namespace characterize_functions_l805_805939

open Function

noncomputable def f : ℚ → ℚ := sorry
noncomputable def g : ℚ → ℚ := sorry

axiom f_g_condition_1 : ∀ x y : ℚ, f (g (x) - g (y)) = f (g (x)) - y
axiom f_g_condition_2 : ∀ x y : ℚ, g (f (x) - f (y)) = g (f (x)) - y

theorem characterize_functions : 
  (∃ c : ℚ, ∀ x, f x = c * x) ∧ (∃ c : ℚ, ∀ x, g x = x / c) := 
sorry

end characterize_functions_l805_805939


namespace solve_ellipse_and_chord_l805_805395

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ C : set (ℝ × ℝ), ∀ x y : ℝ, C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def line_through_point_with_slope (p : ℝ × ℝ) (m : ℝ) : set (ℝ × ℝ) :=
  { l : ℝ × ℝ | l.2 = m * (l.1 - p.1) + p.2 }

theorem solve_ellipse_and_chord : 
  ∀ (a b : ℝ), a > b → b > 0 → 
  (ellipse_equation a b) → 
  ((0, 4) ∈ { (x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1 }) → 
  (∃ e : ℝ, e = 3/5) →
  (a^2 = b^2 + (e * a)^2) →
  (a = 5 ∧ b = 4) ∧ 
  (∀ (line : ℝ × ℝ → ℝ × ℝ), ((3,0) ∈ line p 4/5) → 
  ∃ l : ℝ, l = 41/5) :=
begin 
  assume a b ha hb hE hecc hrelation,
  have h1 : a = 5, from sorry,
  have h2 : b = 4, from sorry,
  split, 
  { exact ⟨h1, h2⟩ },
  {
    assume line ht,
    exact ⟨41/5, sorry⟩,
    sorry
  },
  sorry 
end

end solve_ellipse_and_chord_l805_805395


namespace min_distance_P_F2_perimeter_PF1QF2_l805_805792

-- Definitions for the ellipse and its properties
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def a := 2
def b := sqrt 3
def c := sqrt (a^2 - b^2)

def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Foci of the ellipse
def F1 := (-c, 0)
def F2 := (c, 0)

-- Point P on the ellipse
variable (P : ℝ × ℝ)
#check P
#check (P.1)
#eval P.1 P.2
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Minimum distance from point P to F2
theorem min_distance_P_F2
  (hP : on_ellipse P) :
  ∃ P : ℝ × ℝ, distance P F1 = a - c := sorry

-- The perimeter of quadrilateral PF1QF2 is 8
theorem perimeter_PF1QF2
  (hP : on_ellipse P)
  (hQ : ∃ Q : ℝ × ℝ, on_ellipse Q ∧ Q ≠ P) :
  distance P F1 + distance F1 Q + distance Q F2 + distance P F2 = 8 := sorry

end min_distance_P_F2_perimeter_PF1QF2_l805_805792


namespace meetings_percentage_l805_805745

/-- Define the total work day in hours -/
def total_work_day_hours : ℕ := 10

/-- Define the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60 -- 1 hour = 60 minutes

/-- Define the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Define the break duration in minutes -/
def break_minutes : ℕ := 30

/-- Define the effective work minutes -/
def effective_work_minutes : ℕ := (total_work_day_hours * 60) - break_minutes

/-- Define the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- The percentage of the effective work day spent in meetings -/
def percent_meetings : ℕ := (total_meeting_minutes * 100) / effective_work_minutes

theorem meetings_percentage : percent_meetings = 24 := by
  sorry

end meetings_percentage_l805_805745


namespace find_number_l805_805807

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l805_805807


namespace minSides_approximation_l805_805826

noncomputable def minSidesForApproximation(r : ℝ, error : ℝ) :=
  inf {n : ℕ | 
    let angle := 2 * Real.pi / n;
    let polygonArea := n / 2 * r^2 * Real.sin(angle);
    let circleArea := Real.pi * r^2;
    abs(circleArea - polygonArea) / circleArea < error
  }

theorem minSides_approximation (r : ℝ) (h : r > 0) :
  minSidesForApproximation r 0.001 = 82 :=
by
  sorry

end minSides_approximation_l805_805826


namespace simplify_and_evaluate_f_l805_805195

noncomputable def f(α : ℝ) := 
    (sin (π + α) * cos (2 * π - α) * sin (3 / 2 * π - α)) / 
    (cos (-π - α) * cos (π / 2 + α))

theorem simplify_and_evaluate_f (α a : ℝ) (h : a ≠ 0) (ha : cos α = (5 / 13)) : 
    f(α) = if 0 < a then (5 / 13) else -(5 / 13) := sorry

end simplify_and_evaluate_f_l805_805195


namespace midpoint_trajectory_of_chord_l805_805650

theorem midpoint_trajectory_of_chord {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 / 3 + A.2^2 = 1) ∧ 
    (B.1^2 / 3 + B.2^2 = 1) ∧ 
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (x, y) ∧ 
    ∃ t : ℝ, ((-1, 0) = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2))) -> 
  x^2 + x + 3 * y^2 = 0 :=
by sorry

end midpoint_trajectory_of_chord_l805_805650


namespace tina_took_away_2_oranges_l805_805043

-- Definition of the problem
def oranges_taken_away (x : ℕ) : Prop :=
  let original_oranges := 5
  let tangerines_left := 17 - 10 
  let oranges_left := original_oranges - x
  tangerines_left = oranges_left + 4 

-- The statement that needs to be proven
theorem tina_took_away_2_oranges : oranges_taken_away 2 :=
  sorry

end tina_took_away_2_oranges_l805_805043


namespace y_intercept_is_2_l805_805684

def equation_of_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def point_P : ℝ × ℝ := (-1, 1)

def y_intercept_of_tangent_line (m c x y : ℝ) : Prop :=
  equation_of_circle x y ∧
  ((y = m * x + c) ∧ (point_P.1, point_P.2) ∈ {(x, y) | y = m * x + c})

theorem y_intercept_is_2 :
  ∃ m c : ℝ, y_intercept_of_tangent_line m c 0 2 :=
sorry

end y_intercept_is_2_l805_805684


namespace sam_return_rate_l805_805417

noncomputable theory
open_locale classical

def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_return_rate :
  let P : ℝ := 10000 in
  let r : ℝ := 0.20 in
  let n : ℕ := 1 in
  let t : ℕ := 3 in
  let A1 : ℝ := compound_interest P r n t in
  let total_investment : ℝ := 3 * P in
  let final_amount : ℝ := 59616 in
  A1 = 17280 ∧ final_amount = total_investment * (1 + 0.9872) :=
begin
  sorry
end

end sam_return_rate_l805_805417


namespace remainder_of_division_l805_805576

def num : ℤ := 1346584
def divisor : ℤ := 137
def remainder : ℤ := 5

theorem remainder_of_division 
  (h : 0 <= divisor) (h' : divisor ≠ 0) : 
  num % divisor = remainder := 
sorry

end remainder_of_division_l805_805576


namespace box_volume_l805_805499

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l805_805499


namespace find_a_l805_805200

noncomputable def is_on_circle (x y a : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = 8

noncomputable def distance_to_line (a : ℝ) : ℝ :=
  Real.abs (a + a) / Real.sqrt (1^2 + 1^2)

theorem find_a (a : ℝ) (x y : ℝ) :
  is_on_circle x y a →
  ∃ a, (distance_to_line a - 2 * Real.sqrt 2) = Real.sqrt 2 →
  a = 3 ∨ a = -3 := by
  sorry

end find_a_l805_805200


namespace trajectory_equation_midpoint_C_equation_of_line_l805_805201

theorem trajectory_equation_midpoint_C (x y x0 y0 : ℝ) (h1 : x = (x0 + 0) / 2) (h2 : y = (y0 - 4) / 2) (h3 : x0^2 + y0^2 = 4) : 
  x^2 + (y + 2)^2 = 1 :=
sorry

theorem equation_of_line (k : ℝ) (B : ℝ × ℝ) (H : B = (-1/2, -1)) (MN_length : ℝ) (H_MN : MN_length = sqrt 3) 
  (d : ℝ) (H_d : d = 1/2) (l1 l2 : ℝ × ℝ) (H_l1 : l1 = (-1/2, 0)) (H_l2_slope : d = (abs (k / 2 + 1)) / sqrt (1 + k^2))
  (H_l2_solved : k = -3/4) :
  (x = -1/2) ∨ (6 * fst B + 8 * snd B + 11 = 0) :=
sorry

end trajectory_equation_midpoint_C_equation_of_line_l805_805201


namespace exists_rhombus_in_parallelogram_l805_805820

open Classical
open Set

variables {A B C D M N K L : Type}
variables [AffineSpace A B C D M N K L]

-- Given condition: ABCD is a parallelogram
noncomputable def is_parallelogram (A B C D : Type) [AffineSpace A B C D] : Prop :=
  parallel (line A B) (line C D) ∧ parallel (line A D) (line B C)

-- Condition that vertices M, N, K, L lie on the sides of the parallelogram
def vertices_on_sides (M N K L A B C D : Type) [AffineSpace M N K L A B C D] : Prop :=
  Collinear M A B ∧ Collinear N B C ∧ Collinear K C D ∧ Collinear L A D

-- Condition that LM is parallel to BD and MN is parallel to AC
def parallel_sides (L M N K A B C D : Type) [AffineSpace L M N K A B C D] : Prop :=
  parallel (line L M) (line B D) ∧ parallel (line M N) (line A C)

-- The proof statement
theorem exists_rhombus_in_parallelogram (A B C D M N K L : Type) [AffineSpace A B C D M N K L] 
  (h_parallelogram : is_parallelogram A B C D)
  (h_vertices : vertices_on_sides M N K L A B C D)
  (h_parallel : parallel_sides L M N K A B C D) :
  ∃ (MNKL : set Type), 
    (MNKL = {M, N, K, L}) ∧ 
    (h_vertices M N K L A B C D) ∧
    (h_parallel L M N K A B C D) := 
  sorry

end exists_rhombus_in_parallelogram_l805_805820


namespace joshua_skittles_l805_805368

theorem joshua_skittles (eggs : ℝ) (skittles_per_friend : ℝ) (friends : ℝ) (h1 : eggs = 6.0) (h2 : skittles_per_friend = 40.0) (h3 : friends = 5.0) : skittles_per_friend * friends = 200.0 := 
by 
  sorry

end joshua_skittles_l805_805368


namespace parallel_vectors_implies_m_value_l805_805671

theorem parallel_vectors_implies_m_value 
  (m : ℝ) : 
  let a := (1, m)
  let b := (-1, real.sqrt 3)
  (a.1 * b.2 - a.2 * b.1 = 0) → (m = -real.sqrt 3) :=
begin
  sorry
end

end parallel_vectors_implies_m_value_l805_805671


namespace frac_x_y_value_l805_805389

theorem frac_x_y_value (x y : ℝ) (h1 : 3 < (2 * x - y) / (x + 2 * y))
(h2 : (2 * x - y) / (x + 2 * y) < 7) (h3 : ∃ (t : ℤ), x = t * y) : x / y = -4 := by
  sorry

end frac_x_y_value_l805_805389


namespace find_a_l805_805591

-- Define the main inequality condition
def inequality_condition (a x : ℝ) : Prop := |x^2 + a * x + 4 * a| ≤ 3

-- Define the condition that there is exactly one solution to the inequality
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (inequality_condition a x) ∧ (∀ y : ℝ, x ≠ y → ¬(inequality_condition a y))

-- The theorem that states the specific values of a
theorem find_a (a : ℝ) : has_exactly_one_solution a ↔ a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13 := 
by
  sorry

end find_a_l805_805591


namespace donation_ratio_is_half_l805_805620

-- Define the conditions as given in part a).
def monthly_income : ℝ := 240
def groceries_expense : ℝ := 20
def remaining_amount : ℝ := 100

-- Define the function to get the amount donated
def donation_amount (monthly_income : ℝ) (groceries_expense : ℝ) (remaining_amount : ℝ) : ℝ :=
  monthly_income - groceries_expense - remaining_amount

-- Define the target ratio
def donation_ratio (donation_amount : ℝ) (monthly_income : ℝ) : ℝ :=
  donation_amount / monthly_income

theorem donation_ratio_is_half :
  donation_ratio (donation_amount monthly_income groceries_expense remaining_amount) monthly_income = 1 / 2 :=
by
  sorry

end donation_ratio_is_half_l805_805620


namespace sum_slope_intercept_l805_805413

def Point := (ℝ × ℝ)
def line_through (C D : Point) : ℝ → ℝ := 
  λ x, ((D.2 - C.2) / (D.1 - C.1)) * x + (C.2 - ((D.2 - C.2) / (D.1 - C.1)) * C.1)

theorem sum_slope_intercept (C D : Point) (hC : C = (3, 7)) (hD : D = (8, 10)) : 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := (C.2 - m * C.1)
  m + b = 29 / 5 := 
by
  sorry

end sum_slope_intercept_l805_805413


namespace sequence_sum_relation_l805_805647

theorem sequence_sum_relation (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, 4 * S n = (a n + 1) ^ 2) →
  (S 1 = a 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  a 2023 = 4045 :=
by
  sorry

end sequence_sum_relation_l805_805647


namespace train_pass_time_l805_805890

open Real

def length_of_train : ℝ := 280
def speed_kmh : ℝ := 63
def speed_ms : ℝ := speed_kmh * (1000 / 3600)
def time_to_pass_tree (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_pass_time : time_to_pass_tree length_of_train speed_ms = 16 := by
  -- the proof would go here
  sorry

end train_pass_time_l805_805890


namespace number_of_integers_satisfying_cubed_inequality_l805_805304

theorem number_of_integers_satisfying_cubed_inequality :
  {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100}.to_finset.card = 9 :=
by
  sorry

end number_of_integers_satisfying_cubed_inequality_l805_805304


namespace line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l805_805358

section BarycentricCoordinates

variables {A1 A2 A3 A4 : Type} 

def barycentric_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 + x3 + x4 = 1

theorem line_A1_A2_condition (x1 x2 x3 x4 : ℝ) : 
  barycentric_condition x1 x2 x3 x4 → (x3 = 0 ∧ x4 = 0) ↔ (x1 + x2 = 1) :=
by
  sorry

theorem plane_A1_A2_A3_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x4 = 0) ↔ (x1 + x2 + x3 = 1) :=
by
  sorry

theorem plane_through_A3_A4_parallel_to_A1_A2_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x1 = -x2 ∧ x3 + x4 = 1) ↔ (x1 + x2 + x3 + x4 = 1) :=
by
  sorry

end BarycentricCoordinates

end line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l805_805358


namespace boyfriend_picks_pieces_l805_805565

theorem boyfriend_picks_pieces (initial_pieces : ℕ) (cat_steals : ℕ) 
(boyfriend_fraction : ℚ) (swept_fraction : ℚ) 
(h_initial : initial_pieces = 60) (h_swept : swept_fraction = 1 / 2) 
(h_cat : cat_steals = 3) (h_boyfriend : boyfriend_fraction = 1 / 3) : 
ℕ :=
  let swept_pieces := initial_pieces * swept_fraction
  let remaining_pieces := swept_pieces - cat_steals
  let boyfriend_pieces := remaining_pieces * boyfriend_fraction
  by
    have h_swept_pieces : swept_pieces = 30 := by sorry
    have h_remaining_pieces : remaining_pieces = 27 := by sorry
    have h_boyfriend_pieces : boyfriend_pieces = 9 := by sorry
    exact h_boyfriend_pieces

end boyfriend_picks_pieces_l805_805565


namespace John_pays_correct_amount_for_hearing_aids_l805_805725

-- Definitions of the variables according to the conditions
def cost_each_hearing_aid_1 : ℕ := 2500
def cost_each_hearing_aid_2 : ℕ := 2500
def cost_each_hearing_aid_3 : ℕ := 3000
def insurance_deductible : ℕ := 500
def insurance_coverage_percentage : ℕ := 80
def insurance_coverage_limit : ℕ := 3500

-- The main theorem
theorem John_pays_correct_amount_for_hearing_aids :
  (let total_cost := cost_each_hearing_aid_1 + cost_each_hearing_aid_2 + cost_each_hearing_aid_3,
       deductible_covered := min (insurance_coverage_limit) ((insurance_coverage_percentage * (total_cost - insurance_deductible)) / 100),
       john_out_of_pocket := total_cost - deductible_covered + insurance_deductible in
   john_out_of_pocket = 4500) :=
begin
  sorry
end

end John_pays_correct_amount_for_hearing_aids_l805_805725


namespace find_cost_price_l805_805548

def selling_price_per_meter : ℝ := 9890 / 92
def profit_per_meter : ℝ := 24
def cost_price_per_meter (selling_price_per_meter profit_per_meter : ℝ) : ℝ := selling_price_per_meter - profit_per_meter

theorem find_cost_price (h1 : selling_price_per_meter = 9890 / 92) 
                        (h2 : profit_per_meter = 24) 
                        (h3 : cost_price_per_meter selling_price_per_meter profit_per_meter = 83.5) : 
  cost_price_per_meter (9890 / 92) 24 = 83.5 := 
by
  rw [h1, h2, h3]
  sorry

end find_cost_price_l805_805548


namespace irene_apples_cost_l805_805361

theorem irene_apples_cost :
  (2 weeks * 7 days/week) * (1 apple/day) * (1/4 pound/apple) * (2 dollars/pound) = 7 dollars :=
by
  let days := 2 * 7
  let apples := days * 1
  let weight := apples * (1/4)
  let cost := weight * 2
  have : cost = 7 := by
    sorry
  sorry

end irene_apples_cost_l805_805361


namespace good_ordered_pairs_count_l805_805560

theorem good_ordered_pairs_count (n : ℕ) : 
  let total_pairs := 4^n in
  let subset_pairs := 3^n in
  let overlap_pairs := 2^n in
  let good_pairs := total_pairs - 2 * subset_pairs + overlap_pairs in
  good_pairs = 4^n - 2 * 3^n + 2^n :=
by
  let total_pairs := 4^n
  let subset_pairs := 3^n
  let overlap_pairs := 2^n
  let good_pairs := total_pairs - 2 * subset_pairs + overlap_pairs
  have h : good_pairs = 4^n - 2 * 3^n + 2^n := sorry
  exact h

#eval good_ordered_pairs_count 2017

end good_ordered_pairs_count_l805_805560


namespace greatest_integer_not_exceeding_perimeter_of_OXYZ_l805_805373

-- Define the conditions for distances
variables (AZ AX BX BZ AY BY: ℝ)
axiom h1 : AZ - AX = 6
axiom h2 : BX - BZ = 9
axiom h3 : AY = 12
axiom h4 : BY = 5

-- Define additional assumptions as needed for the pentagon inscribed in a semicircle
-- using O as the midpoint of AB, and it being a right-angled semicircle etc.

theorem greatest_integer_not_exceeding_perimeter_of_OXYZ
    (O : ℝ) (X Y Z : ℝ) : floor (perimeter_of_quadrilateral O X Y Z) = 23 :=
by
  sorry -- Proof not required

end greatest_integer_not_exceeding_perimeter_of_OXYZ_l805_805373


namespace distance_from_point_P_to_x_axis_l805_805978

theorem distance_from_point_P_to_x_axis 
  (P F1 F2 : ℝ × ℝ) 
  (C : set (ℝ × ℝ))
  (hC_subtype : ∀ {p : ℝ × ℝ}, p ∈ C ↔ (p.1 ^ 2 - p.2 ^ 2 = 1)) 
  (hP_C : P ∈ C)
  (hF1 : F1 = (-sqrt 2, 0))
  (hF2 : F2 = (sqrt 2, 0))
  (h_angle_60 : ∃ θ, θ = 60 ∧ ∀ (m n : ℝ), 
    θ = real.arccos ((P.1 - F1.1)*(P.1 - F2.1) + (P.2 - F1.2)*(P.2 - F2.2)) / 
    ((real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * 
    (real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)))) :
  real.abs P.2 = sqrt 6 / 2 := 
sorry

end distance_from_point_P_to_x_axis_l805_805978


namespace units_digit_of_sum_of_squares_of_first_3005_odd_integers_l805_805081

theorem units_digit_of_sum_of_squares_of_first_3005_odd_integers :
  (List.sum (List.map (λ n : ℕ, (2 * n - 1) ^ 2) (List.range 3005))) % 10 = 3 :=
by
  -- The rest of the proof would go here
  sorry

end units_digit_of_sum_of_squares_of_first_3005_odd_integers_l805_805081


namespace binomial_identity_l805_805100

theorem binomial_identity (n k : ℕ) (h₁ : k ≤ n) (h₂ : (k, n) ≠ (0, 0)) :
  nat.choose n k = nat.choose (n - 1) k + nat.choose (n - 1) (k - 1) :=
by
  sorry

end binomial_identity_l805_805100


namespace task1_task2_task3_l805_805661

noncomputable def f (x a : ℝ) := x^2 - 4 * x + a + 3
noncomputable def g (x m : ℝ) := m * x + 5 - 2 * m

theorem task1 (a m : ℝ) (h₁ : a = -3) (h₂ : m = 0) :
  (∃ x : ℝ, f x a - g x m = 0) ↔ x = -1 ∨ x = 5 :=
sorry

theorem task2 (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem task3 (m : ℝ) :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ 0 = g x₂ m) ↔ m ≤ -3 ∨ 6 ≤ m :=
sorry

end task1_task2_task3_l805_805661


namespace quartic_polynomial_with_roots_l805_805931

theorem quartic_polynomial_with_roots :
  ∃ p : Polynomial ℚ, p.monic ∧ p.degree = 4 ∧ (p.eval (3 + Real.sqrt 5) = 0) ∧ (p.eval (2 - Real.sqrt 7) = 0) :=
by
  sorry

end quartic_polynomial_with_roots_l805_805931


namespace find_1314th_digit_of_fraction_l805_805941

theorem find_1314th_digit_of_fraction:
  (let decimal_expansion := "0.357142857142..."
   in (decimal_expansion.to_list.length > 1314) → (decimal_expansion.to_list[1314 - 1] = '2')) :=
sorry

end find_1314th_digit_of_fraction_l805_805941


namespace least_width_l805_805761

theorem least_width (w : ℝ) (h_nonneg : w ≥ 0) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end least_width_l805_805761


namespace eccentricity_range_l805_805336

noncomputable def hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ (x^2 / a^2 - y^2 / b^2 = 1)}

theorem eccentricity_range (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (x y : ℝ)
  (h : (x, y) ∈ hyperbola a b a_pos b_pos)
  (hx_ge_a : x ≥ a)
  (dist_equal : (x^2 + y^2) = ((x - sqrt(a^2 + b^2))^2 + y^2)) :
  (sqrt (a^2 + b^2) / a > 2) :=
sorry

end eccentricity_range_l805_805336


namespace second_largest_and_second_smallest_sum_l805_805037

theorem second_largest_and_second_smallest_sum :
  ∃ a b c d : ℕ, {a, b, c, d} = {10, 11, 12, 13} ∧
               second_smallest {a, b, c, d} + second_largest {a, b, c, d} = 23 :=
by
  sorry

end second_largest_and_second_smallest_sum_l805_805037


namespace sqrt_inequality_for_natural_l805_805512

theorem sqrt_inequality_for_natural (n : ℕ) : sqrt (↑(n + 1)) + 2 * sqrt (↑n) < sqrt (↑(9 * n + 3)) := sorry

end sqrt_inequality_for_natural_l805_805512


namespace abe_found_4_ants_l805_805553

variable (A : ℝ) -- Number of ants Abe found
variable (Beth CeCe Duke : ℝ) -- Number of ants Beth, CeCe, and Duke found

-- Conditions
def condition1 : Prop := Beth = 1.5 * A
def condition2 : Prop := CeCe = 2 * A
def condition3 : Prop := Duke = 0.5 * A
def condition4 : Prop := A + Beth + CeCe + Duke = 20

-- Proof that Abe found 4 ants
theorem abe_found_4_ants (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : A = 4 := 
sorry

end abe_found_4_ants_l805_805553


namespace quartic_poly_exists_l805_805938

noncomputable def quartic_poly_with_given_roots (p : ℚ[X]) : Prop :=
  p.monic ∧ p.coeff 4 = 1 ∧
  (p.eval (3 + real.sqrt 5) = 0) ∧ (p.eval (3 - real.sqrt 5) = 0) ∧
  (p.eval (2 - real.sqrt 7) = 0) ∧ (p.eval (2 + real.sqrt 7) = 0)

theorem quartic_poly_exists :
  ∃ p : ℚ[X], quartic_poly_with_given_roots p ∧ p = (X^4 - 10*X^3 + 13*X^2 + 18*X - 12) :=
begin
  sorry
end

end quartic_poly_exists_l805_805938


namespace problem_1_problem_2_l805_805194

open Real

-- Part 1
theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) :=
sorry

-- Part 2
theorem problem_2 : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 4 ∧ (4 / (a * b) + a / b = (1 + sqrt 5) / 2) :=
sorry

end problem_1_problem_2_l805_805194


namespace duration_of_each_class_is_3_l805_805589

theorem duration_of_each_class_is_3
    (weeks : ℕ) 
    (x : ℝ) 
    (weekly_additional_class_hours : ℝ) 
    (homework_hours_per_week : ℝ) 
    (total_hours : ℝ) 
    (h1 : weeks = 24)
    (h2 : weekly_additional_class_hours = 4)
    (h3 : homework_hours_per_week = 4)
    (h4 : total_hours = 336) :
    (2 * x + weekly_additional_class_hours + homework_hours_per_week) * weeks = total_hours → x = 3 := 
by 
  sorry

end duration_of_each_class_is_3_l805_805589


namespace golden_horse_cards_count_l805_805699

theorem golden_horse_cards_count : 
  let total_cards := 10000 in
  let cards_without_5_or_8 := 8^4 in
  let golden_horse_cards := total_cards - cards_without_5_or_8
  golden_horse_cards = 5904 :=
by
  sorry

end golden_horse_cards_count_l805_805699


namespace num_int_values_satisfying_ineq_l805_805263

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805263


namespace max_marks_is_400_l805_805095

theorem max_marks_is_400 :
  ∃ M : ℝ, (0.30 * M = 120) ∧ (M = 400) := 
by 
  sorry

end max_marks_is_400_l805_805095


namespace emeralds_division_l805_805894

theorem emeralds_division (E : ℕ) 
  (h1 : E % 8 = 5) 
  (h2 : E % 7 = 6) 
  (h_smallest : ∀ n : ℕ, (n % 8 = 5) → (n % 7 = 6) → (n ≥ 13)) : 
  ∃ n, n = 13 ∧ 13 % 9 = 4 :=
by
  have hE : E = 13 := sorry
  use 13
  split
  . exact hE
  . norm_num
  sorry

end emeralds_division_l805_805894


namespace find_BC_l805_805001

-- Definitions based on given conditions
def area_triangle_ABC := 30 -- in cm^2
def ratio_AD_DC := (2, 3)
def length_perpendicular_DE := 9 -- in cm

-- Statement to prove
theorem find_BC
  (h1 : area_triangle_ABC = 30)
  (h2 : ratio_AD_DC = (2, 3))
  (h3 : length_perpendicular_DE = 9) :
  let BC : ℝ := 4
  in BC = 4 := 
sorry

end find_BC_l805_805001


namespace cartesian_to_polar_l805_805434

theorem cartesian_to_polar
  (x y : ℝ)
  (ρ θ : ℝ)
  (h1 : x^2 + y^2 = ρ^2)
  (h2 : x = ρ * cos θ)
  (h3 : y = ρ * sin θ)
  (h4 : x^2 + y^2 - 2 * x = 0) :
  ρ = 2 * cos θ :=
by
  sorry

end cartesian_to_polar_l805_805434


namespace problem1_problem2_l805_805667

-- Define the set M
def M := {x : ℝ | x^2 - 3*x - 18 ≤ 0}

-- Define the set N with a parameter a
def N (a : ℝ) := {x : ℝ | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Problem part 1: Given a = 3, prove the intersections and complements of sets
theorem problem1 : 
  (N 3 ∩ M) = {x : ℝ | -2 ≤ x ∧ x ≤ 6} ∧ 
  (∀ x, x ∉ N 3 ↔ ((x ∈ Iio (-2)) ∨ (x ∈ Ioi 7))) :=
sorry

-- Problem part 2: Given M ∩ N(a) = N(a), prove the range of a
theorem problem2 : (∀ a, N a ⊆ M → a ≤ 5/2) :=
sorry


end problem1_problem2_l805_805667


namespace find_y_value_l805_805641

noncomputable def find_y (θ : ℝ) (y : ℝ) : Prop :=
  ∃ θ, (P : ℝ × ℝ) (P = (-1, y)) ∧ (sin θ = (2 * real.sqrt 5) / 5)

theorem find_y_value (θ : ℝ) (y : ℝ) (h1 : P = (-1, y)) (h2 : sin θ = (2 * real.sqrt 5) / 5) : y = 2 :=
  by
  sorry

end find_y_value_l805_805641


namespace num_int_values_satisfying_ineq_l805_805259

theorem num_int_values_satisfying_ineq : 
  { n : ℤ | -100 < n^3 ∧ n^3 < 100 }.count == 9 :=
by
  sorry

end num_int_values_satisfying_ineq_l805_805259


namespace tetrahedron_spheres_relationships_l805_805743

-- Define the conditions
variables (A B C D : Type) 
variables (varrho1 varrho2 varrho3 varrho4 varrho12 varrho13 varrho14 m1 m2 m3 m4 : ℝ)

-- State the theorem
theorem tetrahedron_spheres_relationships :
  (A B C D : Type) 
  (varrho1 varrho2 varrho3 varrho4 varrho12 varrho13 varrho14 m1 m2 m3 m4 : ℝ) :
  (pm2over_varrho12_varrho_faces: 
    ∀ varrho12 varrho1 varrho2 varrho3 varrho4, 
    ∃ (sign: ℤ), 
    sign * 2 / varrho12 = 1 / varrho1 + 1 / varrho2 + 1 / varrho3 + 1 / varrho4) ∧
  (pm1over_varrho12_heights: 
    ∀ varrho12 m1 m2 m3 m4, 
    ∃ (sign: ℤ), 
    sign * 1 / varrho12 = 1 / m1 + 1 / m2 + 1 / m3 + 1 / m4)
:= sorry

end tetrahedron_spheres_relationships_l805_805743


namespace digging_cost_correct_l805_805609

noncomputable def total_cost_of_digging_well
  (depth : ℝ)
  (diameter : ℝ)
  (cost_per_cubic_meter : ℝ) 
  : ℝ :=
  let r := diameter / 2 in
  let volume := Real.pi * r^2 * depth in
  volume * cost_per_cubic_meter

theorem digging_cost_correct :
  total_cost_of_digging_well 14 3 18 = 1782 :=
by
  have radius := 3 / 2
  have volume := Real.pi * radius^2 * 14
  have cost := volume * 18
  have cost_rounded := Float.ceil cost
  show total_cost_of_digging_well 14 3 18 = 1782
  sorry

end digging_cost_correct_l805_805609


namespace percentage_error_l805_805123

theorem percentage_error (x : ℝ) (hx : x ≠ 0) :
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 :=
by
  sorry

end percentage_error_l805_805123


namespace division_of_fractions_l805_805480

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l805_805480


namespace part1_part2_l805_805986

theorem part1 (n r : ℕ) (hn : r = 5) (h6th_term : (n - 2 * r) = 10) : n = 10 := by
  sorry

theorem part2 (r : ℕ) (n := 10) (hr : (2 * r) = 4) : 
  let coefficient := ( -1/2) ^ r * (Nat.choose n r) in
  coefficient = 45 / 4 := by
  sorry

end part1_part2_l805_805986


namespace ratio_Rachel_Sara_l805_805414

-- Define Sara's spending
def Sara_shoes_spending : ℝ := 50
def Sara_dress_spending : ℝ := 200

-- Define Rachel's budget
def Rachel_budget : ℝ := 500

-- Calculate Sara's total spending
def Sara_total_spending : ℝ := Sara_shoes_spending + Sara_dress_spending

-- Define the theorem to prove the ratio
theorem ratio_Rachel_Sara : (Rachel_budget / Sara_total_spending) = 2 := by
  -- Proof is omitted (you would fill in the proof here)
  sorry

end ratio_Rachel_Sara_l805_805414


namespace interest_amount_eq_750_l805_805862

-- Definitions
def P : ℕ := 3000
def R : ℕ := 5
def T : ℕ := 5

-- Condition
def interest_less_than_sum := 2250

-- Simple interest formula
def simple_interest (P R T : ℕ) := (P * R * T) / 100

-- Theorem
theorem interest_amount_eq_750 : simple_interest P R T = P - interest_less_than_sum :=
by
  -- We assert that we need to prove the equality holds.
  sorry

end interest_amount_eq_750_l805_805862


namespace median_of_first_twelve_positive_integers_l805_805070

def sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def median (s : List ℕ) : ℚ :=
  if h : (List.length s) % 2 = 0 then
    let k := List.length s / 2
    (s.get (k - 1) + s.get k) / 2
  else
    s.get (List.length s / 2)

theorem median_of_first_twelve_positive_integers :
  median sequence = 6.5 := 
sorry

end median_of_first_twelve_positive_integers_l805_805070


namespace surface_area_of_sphere_l805_805210

-- Translate the conditions and required proof into Lean
theorem surface_area_of_sphere (A B C O : Type)
  [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace O]
  (dist_AB : dist A B = 3) (dist_BC : dist B C = 3) (dist_CA : dist C A = 3)
  (h : ∀ r : ℝ, r > 0 → dist (center_of_sphere O) (plane_ABC A B C) = r / 3) :
  ∃ r : ℝ, 4 * π * r^2 = 27 / 2 * π :=
by
  sorry

end surface_area_of_sphere_l805_805210


namespace train_crosses_bridge_l805_805678

theorem train_crosses_bridge 
  (train_length : ℝ) (bridge_length : ℝ) (speed_kmph : ℝ) (converted_speed : ℝ)
  (total_distance : train_length + bridge_length = 825)
  (speed_conversion : speed_kmph * (1000 / 3600) = converted_speed)
  (time_calculation : total_distance / converted_speed = 33) : 
  train_length = 165 ∧ bridge_length = 660 ∧ speed_kmph = 90 ∧ converted_speed = 25 → 
  ∃ t, t = 33 :=
by
  -- Variables definition
  assume h : train_length = 165 ∧ bridge_length = 660 ∧ speed_kmph = 90 ∧ converted_speed = 25
  use 33
  sorry

end train_crosses_bridge_l805_805678


namespace average_roots_of_quadratic_l805_805876

open Real

theorem average_roots_of_quadratic (a b : ℝ) (h_eq : ∃ x1 x2 : ℝ, a * x1^2 - 2 * a * x1 + b = 0 ∧ a * x2^2 - 2 * a * x2 + b = 0):
  (b = b) → (a ≠ 0) → (h_discriminant : (2 * a)^2 - 4 * a * b ≥ 0) → (x1 + x2) / 2 = 1 :=
by
  sorry

end average_roots_of_quadratic_l805_805876


namespace partA_partB_partC_l805_805449
noncomputable section

def n : ℕ := 100
def p : ℝ := 0.8
def q : ℝ := 1 - p

def binomial_prob (k1 k2 : ℕ) : ℝ := sorry

theorem partA : binomial_prob 70 85 = 0.8882 := sorry
theorem partB : binomial_prob 70 100 = 0.9938 := sorry
theorem partC : binomial_prob 0 69 = 0.0062 := sorry

end partA_partB_partC_l805_805449


namespace magnitude_of_a_plus_b_l805_805959

def vector := (ℝ × ℝ)

def a : vector := (-1, 3)
def b (t : ℝ) : vector := (1, t)
def sub (u v : vector) : vector := (u.1 - v.1, u.2 - v.2)
def scalar_mult (k : ℝ) (v : vector) : vector := (k * v.1, k * v.2)
def dot (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : vector) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_a_plus_b :
  ∀ (t : ℝ), dot (sub a (scalar_mult 2 (b t))) a = 0 → magnitude (a.1 + b t.1, a.2 + b t.2) = 5 :=
by
  intro t h
  sorry

end magnitude_of_a_plus_b_l805_805959


namespace arithmetic_sequence_sum_l805_805353

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h₁ : ∀ n, a (n + 1) = a n + d)
    (h₂ : a 3 + a 5 + a 7 + a 9 + a 11 = 20) : a 1 + a 13 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l805_805353


namespace exists_K4_l805_805520

-- Define the conditions in terms of graph theory constructs
variables (G : SimpleGraph (Fin 1990))

-- Assume that every vertex in G has at least 1327 friends
axiom degree_at_least : ∀ v : Fin 1990, G.degree v ≥ 1327

-- The main theorem statement
theorem exists_K4 : ∃ (H : SimpleGraph (Fin 1990)), (H ≤ G) ∧ H.complete ∧ H.card = 4 :=
by
  sorry

end exists_K4_l805_805520


namespace starting_lineups_count_l805_805915

theorem starting_lineups_count (players : Finset ℕ) (guards : Finset ℕ) (all_stars : Finset ℕ) (H_total : players.card = 15) (H_guards : guards.card = 5)
  (H_all_stars : all_stars.card = 3) (H_all_stars_subset : all_stars ⊆ players) (H_guards_subset : guards ⊆ players) 
  (H_all_stars_lineup : ∀ star ∈ all_stars, star ∈ players) (H_lineup : ∀ player ∈ players, player∈ guards ∨ player ∉ guards) :
  ((players \ all_stars).card.choose 4 * 15.choose 7 = 285) :=
by
  sorry

end starting_lineups_count_l805_805915


namespace cost_of_one_shirt_l805_805570

-- Definitions for the conditions given
variables (J S X : ℝ)
axiom condition1 : 3 * J + 2 * S = X
axiom condition2 : 2 * J + 3 * S = 66
axiom condition3 : X = 99 - 5 / 2 * S

-- The theorem to prove the cost of one shirt
theorem cost_of_one_shirt : S = 13.20 :=
by {
    sorry,
}

end cost_of_one_shirt_l805_805570


namespace number_of_students_l805_805517

noncomputable def is_handshakes_correct (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 
  (1 / 2 : ℚ) * (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) = 1020

theorem number_of_students (m n : ℕ) (h : is_handshakes_correct m n) : m * n = 280 := sorry

end number_of_students_l805_805517


namespace intersecting_line_exists_l805_805101

theorem intersecting_line_exists {n : ℕ} (Δ : Fin n → Set ℝ × ℝ) 
  (h1 : ∀ i : Fin n, ∃ l : Set ℝ × ℝ, l ⊆ Δ i)
  (h2 : ∀ (i j k : Fin n), ∃ l : Set ℝ × ℝ, l ⊆ Δ i ∩ Δ j ∩ Δ k) :
  ∃ l : Set ℝ × ℝ, ∀ i : Fin n, l ⊆ Δ i :=
sorry

end intersecting_line_exists_l805_805101


namespace intersection_sets_l805_805666

open Set

theorem intersection_sets : 
  let M := { x : ℝ | log 2 (x - 1) < 0 }
  let N := { x : ℝ | x ≥ -2 }
  M ∩ N = { x : ℝ | 1 < x ∧ x < 2 } := 
by 
  let M := { x : ℝ | log 2 (x - 1) < 0 }
  let N := { x : ℝ | x ≥ -2 }
  sorry

end intersection_sets_l805_805666


namespace median_of_first_twelve_positive_integers_l805_805072

def sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def median (s : List ℕ) : ℚ :=
  if h : (List.length s) % 2 = 0 then
    let k := List.length s / 2
    (s.get (k - 1) + s.get k) / 2
  else
    s.get (List.length s / 2)

theorem median_of_first_twelve_positive_integers :
  median sequence = 6.5 := 
sorry

end median_of_first_twelve_positive_integers_l805_805072


namespace insulin_pills_per_day_l805_805245

def conditions (I B A : ℕ) : Prop := 
  B = 3 ∧ A = 2 * B ∧ 7 * (I + B + A) = 77

theorem insulin_pills_per_day : ∃ (I : ℕ), ∀ (B A : ℕ), conditions I B A → I = 2 := by
  sorry

end insulin_pills_per_day_l805_805245


namespace color_blocks_probability_at_least_one_box_match_l805_805154

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end color_blocks_probability_at_least_one_box_match_l805_805154


namespace back_seat_people_l805_805703

/-- Define the number of seats on the left side of the bus --/
def left_side_seats : ℕ := 15

/-- Define the number of seats on the right side of the bus (3 fewer because of the rear exit door) --/
def right_side_seats : ℕ := left_side_seats - 3

/-- Define the number of people each seat can hold --/
def people_per_seat : ℕ := 3

/-- Define the total capacity of the bus --/
def total_capacity : ℕ := 90

/-- Define the total number of people that can sit on the regular seats (left and right sides) --/
def regular_seats_people := (left_side_seats + right_side_seats) * people_per_seat

/-- Theorem stating the number of people that can sit at the back seat --/
theorem back_seat_people : (total_capacity - regular_seats_people) = 9 := by
  sorry

end back_seat_people_l805_805703


namespace expression_equals_two_l805_805421

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l805_805421


namespace count_n_integers_l805_805277

theorem count_n_integers (n : ℤ) : 
  set.count {n : ℤ | -100 < n ^ 3 ∧ n ^ 3 < 100} = 9 := 
sorry

end count_n_integers_l805_805277


namespace count_integer_values_l805_805248

theorem count_integer_values (n : ℤ) : (∀ n : ℤ, -100 < n^3 ∧ n^3 < 100 → -4 ≤ n ∧ n ≤ 4) ∧
  (∀ n : ℤ, -4 ≤ n ∧ n ≤ 4 → -100 < n^3 ∧ n^3 < 100)
  → (finset.card (finset.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (finset.range 9.succ)) = 9) :=
by
  sorry

end count_integer_values_l805_805248


namespace number_of_fences_painted_l805_805928

-- Definitions based on the problem conditions
def meter_fee : ℝ := 0.2
def fence_length : ℝ := 500
def total_earnings : ℝ := 5000

-- Target statement
theorem number_of_fences_painted : (total_earnings / (fence_length * meter_fee)) = 50 := by
sorry

end number_of_fences_painted_l805_805928


namespace polynomial_min_degree_l805_805428

noncomputable def f : Polynomial ℚ :=
  (X - C (real.sqrt 5 + 2)) * (X + C (real.sqrt 5 + 2)) *
  (X - C (real.sqrt 5 - 2)) * (X + C (real.sqrt 5 - 2)) *
  (X - C (real.sqrt 8 + 3)) * (X + C (real.sqrt 8 - 3))

theorem polynomial_min_degree :
  degree f = 6 :=
sorry

end polynomial_min_degree_l805_805428


namespace cannot_become_all_isosceles_l805_805863

/-- Definitions used for the problem, starting from acute-angled scalene triangle.

Problem:
Prove that, given an initial acute-angled scalene triangle and the allowed action of cutting along the
median, it is impossible for all resulting triangles to become isosceles after any finite number of actions.
-/
def is_acute_angled (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_scalene (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

def cut_along_median (a b c : ℝ) : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) := sorry

theorem cannot_become_all_isosceles (a b c : ℝ) (A B C : ℝ) :
  is_acute_angled A B C →
  is_scalene a b c →
  is_triangle a b c →
  ∀ (n : ℕ), ¬ (∀ (triangles : list (ℝ × ℝ × ℝ)),
    (triangles.head.fst.triangle.median_cut.triangle.n_times n).all (λ t, is_isosceles t.fst t.snd t.trd)) :=
sorry

end cannot_become_all_isosceles_l805_805863


namespace simplify_expression_l805_805475

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l805_805475


namespace circle_radius_l805_805531

theorem circle_radius 
  {XA XB XC r : ℝ}
  (h1 : XA = 3)
  (h2 : XB = 5)
  (h3 : XC = 1)
  (hx : XA * XB = XC * r)
  (hh : 2 * r = CD) :
  r = 8 :=
by
  sorry

end circle_radius_l805_805531


namespace find_xyz_l805_805466

def satisfies_conditions (x y z n : ℕ) : Prop :=
  let prob := 0.5 in
  let n_val := x - y * Real.sqrt z in
  n = n_val ∧ z ≠ 0 ∧ ¬ ∃ p, nat.prime p ∧ p * p ∣ z

theorem find_xyz : ∃ (x y z : ℕ), satisfies_conditions x y z (60 - 30 * Real.sqrt 2) ∧ x + y + z = 92 :=
by {
  sorry
}

end find_xyz_l805_805466
