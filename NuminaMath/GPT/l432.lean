import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Divisors
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Pi
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Binomial
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Probability
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.FieldTheory.Polynomial
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.MeasureTheory.MeasureSpace
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbTheory
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Topology.Instances.RealVectorSpace

namespace largest_a_good_set_l432_432663

def M : Set ℕ := { x | 1 ≤ x ∧ x ≤ 65 }

def is_good_set (A : Set ℕ) : Prop :=
  ∃ x y ∈ A, x < y ∧ x ∣ y

def A (A : Set ℕ) : Prop :=
  A ⊆ M ∧ Finset.card (A.to_finset) = 33

def largest_good_set_element : ℕ :=
  21

theorem largest_a_good_set:
  ∀ a ∈ M, (∀ A ⊆ M, Finset.card (A.to_finset) = 33 → a ∈ A → is_good_set A) →
  a ≤ 21 :=
sorry

end largest_a_good_set_l432_432663


namespace ratio_of_speeds_l432_432736

noncomputable def speed_of_first_train := 87.5
noncomputable def distance_second_train := 400
noncomputable def time_second_train := 4

theorem ratio_of_speeds :
  (let speed_of_second_train := distance_second_train / time_second_train in
  let ratio := speed_of_first_train / speed_of_second_train in
  ratio = 7 / 8) :=
by
  sorry

end ratio_of_speeds_l432_432736


namespace donald_has_nine_oranges_l432_432452

theorem donald_has_nine_oranges (initial_oranges : Nat) (found_oranges : Nat) :
    initial_oranges = 4 → found_oranges = 5 → initial_oranges + found_oranges = 9 :=
by
  intros h_initial h_found
  rw [h_initial, h_found]
  exact rfl

end donald_has_nine_oranges_l432_432452


namespace students_who_like_both_l432_432198

def total_students : ℕ := 50
def apple_pie_lovers : ℕ := 22
def chocolate_cake_lovers : ℕ := 20
def neither_dessert_lovers : ℕ := 15

theorem students_who_like_both : 
  (apple_pie_lovers + chocolate_cake_lovers) - (total_students - neither_dessert_lovers) = 7 :=
by
  -- Calculation steps (skipped)
  sorry

end students_who_like_both_l432_432198


namespace modulus_product_l432_432473

theorem modulus_product (a b : ℂ) : |a - b * complex.i| * |a + b * complex.i| = 25 := by
  have h1 : complex.norm (4 - 3 * complex.i) = 5 := by
    sorry
  have h2 : complex.norm (4 + 3 * complex.i) = 5 := by
    sorry
  rw [← complex.norm_mul, (4 - 3 * complex.i).mul_conj_self, (4 + 3 * complex.i).mul_conj_self, add_comm] at h1
  rw [mul_comm, mul_comm (complex.norm _), ← mul_assoc, h2, mul_comm, mul_assoc]
  exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).1 h1 

end modulus_product_l432_432473


namespace knights_count_in_meeting_l432_432627

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432627


namespace probability_two_red_marbles_l432_432787

theorem probability_two_red_marbles
  (red_marbles : ℕ)
  (white_marbles : ℕ)
  (total_marbles : ℕ)
  (prob_first_red : ℚ)
  (prob_second_red_after_first_red : ℚ)
  (combined_probability : ℚ) :
  red_marbles = 5 →
  white_marbles = 7 →
  total_marbles = 12 →
  prob_first_red = 5 / 12 →
  prob_second_red_after_first_red = 4 / 11 →
  combined_probability = 5 / 33 →
  combined_probability = prob_first_red * prob_second_red_after_first_red := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_two_red_marbles_l432_432787


namespace exterior_angle_DEF_is_117_l432_432270

noncomputable def pentagon_interior_angle : ℝ := (180 * (5 - 2)) / 5
noncomputable def octagon_interior_angle : ℝ := (180 * (8 - 2)) / 8
noncomputable def exterior_angle_DEF : ℝ := 360 - (pentagon_interior_angle + octagon_interior_angle)

theorem exterior_angle_DEF_is_117 :
  exterior_angle_DEF = 117 := by
  have h1 : pentagon_interior_angle = 108 := by
    unfold pentagon_interior_angle
    norm_num
  have h2 : octagon_interior_angle = 135 := by
    unfold octagon_interior_angle
    norm_num
  unfold exterior_angle_DEF
  rw [h1, h2]
  norm_num
  sorry

end exterior_angle_DEF_is_117_l432_432270


namespace cylindrical_coord_correct_l432_432433

-- Definitions from conditions
def rect_point := (3 : ℝ, -3 * Real.sqrt 3, 2)

def cylindrical_r (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
def cylindrical_theta (x y : ℝ) : ℝ := Real.arctan (y / x)
def cylindrical_z (z : ℝ) : ℝ := z

noncomputable def cylindrical_coords (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (cylindrical_r x y, cylindrical_theta x y, cylindrical_z z)

-- The proof problem
theorem cylindrical_coord_correct :
  cylindrical_coords rect_point = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end cylindrical_coord_correct_l432_432433


namespace max_product_distance_l432_432913

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432913


namespace evaluate_expression_l432_432460

theorem evaluate_expression :
  (3 - 5 * (3 - 4^2)⁻¹)⁻¹ = 13 / 44 :=
by
  sorry

end evaluate_expression_l432_432460


namespace James_selling_percentage_l432_432595

def James_selling_percentage_proof : Prop :=
  ∀ (total_cost original_price return_cost extra_item bought_price out_of_pocket sold_amount : ℝ),
    total_cost = 3000 →
    return_cost = 700 + 500 →
    extra_item = 500 * 1.2 →
    bought_price = 100 →
    out_of_pocket = 2020 →
    sold_amount = out_of_pocket - (total_cost - return_cost + bought_price) →
    sold_amount / extra_item * 100 = 20

theorem James_selling_percentage : James_selling_percentage_proof :=
by
  sorry

end James_selling_percentage_l432_432595


namespace find_x_of_equation_l432_432887

theorem find_x_of_equation (x : ℝ) (hx : x ≠ 0) : (7 * x)^4 = (14 * x)^3 → x = 8 / 7 :=
by
  intro h
  sorry

end find_x_of_equation_l432_432887


namespace conic_section_union_l432_432152

theorem conic_section_union (x y : ℝ) :
  y^4 - 6 * x^4 = 3 * y^2 - 2 ↔ 
  (∃ a b : ℝ, (y^2 - a * x^2) = b ∧ 
               ((a = 3 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) ∨ 
               ((a = -3 ∧ b = 2) ∨ (a = -2 ∧ b = 1))) :=
begin
  sorry
end

end conic_section_union_l432_432152


namespace number_of_movies_l432_432315

theorem number_of_movies (B M : ℕ)
  (h1 : B = 15)
  (h2 : B = M + 1) : M = 14 :=
by sorry

end number_of_movies_l432_432315


namespace g_range_l432_432037

noncomputable def g (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem g_range : set.range g = set.Icc (3 / 4 : ℝ) 1 := by
  sorry

end g_range_l432_432037


namespace number_of_women_l432_432042

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l432_432042


namespace trigonometric_identity_l432_432927

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 :=
by
  sorry

end trigonometric_identity_l432_432927


namespace puppies_start_count_l432_432812

theorem puppies_start_count (x : ℕ) (given_away : ℕ) (left : ℕ) (h1 : given_away = 7) (h2 : left = 5) (h3 : x = given_away + left) : x = 12 :=
by
  rw [h1, h2] at h3
  exact h3

end puppies_start_count_l432_432812


namespace find_roots_of_quadratic_l432_432888

open Complex

theorem find_roots_of_quadratic : 
  let z1 := sqrt 7 - 1 + (sqrt 7 / 2) * Complex.I
      z2 := -sqrt 7 - 1 - (sqrt 7 / 2) * Complex.I in
    (∀ z : ℂ, z^2 + 2 * z = 3 + 7 * Complex.I ↔ (z = z1 ∨ z = z2))
:= sorry

end find_roots_of_quadratic_l432_432888


namespace polynomial_has_roots_l432_432926

noncomputable def omega : ℂ := complex.exp (complex.I * real.pi / 5)

theorem polynomial_has_roots :
  ∀ (x : ℂ),
    (x = omega ∨ x = omega^3 ∨ x = omega^7 ∨ x = omega^9) →
    (x^4 - x^3 + x^2 - x + 1) = 0 :=
by {
  intro x,
  intro h,
  cases h,
  { sorry },
  { cases h,
    { sorry },
    { cases h,
      { sorry },
      { sorry } } }
}

end polynomial_has_roots_l432_432926


namespace audio_space_per_hour_l432_432381

/-
The digital music library holds 15 days of music.
The library occupies 20,000 megabytes of disk space.
The library contains both audio and video files.
Video files take up twice as much space per hour as audio files.
There is an equal number of hours for audio and video.
-/

theorem audio_space_per_hour (total_days : ℕ) (total_space : ℕ) (equal_hours : Prop) (video_space : ℕ → ℕ) 
  (H1 : total_days = 15)
  (H2 : total_space = 20000)
  (H3 : equal_hours)
  (H4 : ∀ x, video_space x = 2 * x) :
  ∃ x, x = 37 :=
by
  sorry

end audio_space_per_hour_l432_432381


namespace simple_interest_rate_l432_432354

-- Define the conditions
def S : ℚ := 2500
def P : ℚ := 5000
def T : ℚ := 5

-- Define the proof problem
theorem simple_interest_rate (R : ℚ) (h : S = P * R * T / 100) : R = 10 := by
  sorry

end simple_interest_rate_l432_432354


namespace real_solutions_satisfying_symmetric_g_l432_432722

noncomputable def g : ℝ → ℝ := λ x, 3 * x^2

lemma g_is_symmetric (x : ℝ) : g x = g (-x) := 
by
  sorry

lemma g_satisfies_condition (x : ℝ) (hx : x ≠ 0) : 
  g x + 2 * g (1 / x) = 6 * x^2 := 
by
  sorry

theorem real_solutions_satisfying_symmetric_g : 
  ∀ x : ℝ, g x = g (-x) := 
by
  exact g_is_symmetric

end real_solutions_satisfying_symmetric_g_l432_432722


namespace polynomial_has_n_real_roots_l432_432495

noncomputable def P_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), (2:ℝ)^k * (Nat.choose (2 * n) (2 * k)) * (x^k) * ((x - 1)^(n - k))

theorem polynomial_has_n_real_roots (n : ℕ) (hn : 0 < n) : ∃ S : Finset ℝ, S.card = n ∧ ∀ x ∈ S, P_n n x = 0 ∧ 0 < x ∧ x < 1 :=
by
  sorry

end polynomial_has_n_real_roots_l432_432495


namespace sequence_50_mod_6_l432_432443

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 9 else 9 ^ sequence (n - 1)

theorem sequence_50_mod_6 : sequence 50 % 6 = 3 := by
  -- Here will be the proof steps
  sorry

end sequence_50_mod_6_l432_432443


namespace rental_cost_per_day_l432_432369

theorem rental_cost_per_day (p m c : ℝ) (d : ℝ) (hc : c = 0.08) (hm : m = 214.0) (hp : p = 46.12) (h_total : p = d + m * c) : d = 29.00 := 
by
  sorry

end rental_cost_per_day_l432_432369


namespace total_children_on_bus_after_stop_l432_432784

theorem total_children_on_bus_after_stop (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 18) (h2 : additional = 7) : total = 25 :=
by sorry

end total_children_on_bus_after_stop_l432_432784


namespace greatest_number_divisible_by_4_and_5_is_980_l432_432330

noncomputable def greatest_three_digit_divisible_by_4_and_5 : ℕ :=
  980

theorem greatest_number_divisible_by_4_and_5_is_980 :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 
  (100 ≤ m ∧ m ≤ 999 ∧ (m % 4 = 0) ∧ (m % 5 = 0)) → m ≤ n :=
begin
  use 980,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    sorry
  }
end

end greatest_number_divisible_by_4_and_5_is_980_l432_432330


namespace complement_A_in_U_l432_432677

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} :=
by 
  sorry

end complement_A_in_U_l432_432677


namespace product_XC_MD_eq_8sqrt5_l432_432426

variables (A B C D M X : Type)
variables [square : Square A B C D 4]
variables [midpoint : Midpoint M B C]
variables [perpendicular : Perpendicular D (Line A M) X]

theorem product_XC_MD_eq_8sqrt5 : length (line_segment X C) * length (line_segment M D) = 8 * real.sqrt 5 :=
sorry

end product_XC_MD_eq_8sqrt5_l432_432426


namespace difference_largest_smallest_geometric_l432_432052

open Nat

noncomputable def is_geometric_sequence (a b c d : ℕ) : Prop :=
  b = a * 2 / 3 ∧ c = a * (2 / 3)^2 ∧ d = a * (2 / 3)^3 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem difference_largest_smallest_geometric : 
  exists (largest smallest : ℕ), 
  (is_geometric_sequence (largest / 1000) ((largest % 1000) / 100) ((largest % 100) / 10) (largest % 10)) ∧ 
  (is_geometric_sequence (smallest / 1000) ((smallest % 1000) / 100) ((smallest % 100) / 10) (smallest % 10)) ∧ 
  largest = 9648 ∧ smallest = 1248 ∧ largest - smallest = 8400 :=
begin
  sorry
end

end difference_largest_smallest_geometric_l432_432052


namespace otimes_neg2_neg1_l432_432070

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432070


namespace cheaper_store_price_in_cents_l432_432946

/-- List price of Book Y -/
def list_price : ℝ := 24.95

/-- Discount at Readers' Delight -/
def readers_delight_discount : ℝ := 5

/-- Discount rate at Book Bargains -/
def book_bargains_discount_rate : ℝ := 0.2

/-- Calculate sale price at Readers' Delight -/
def sale_price_readers_delight : ℝ := list_price - readers_delight_discount

/-- Calculate sale price at Book Bargains -/
def sale_price_book_bargains : ℝ := list_price * (1 - book_bargains_discount_rate)

/-- Difference in price between Book Bargains and Readers' Delight in cents -/
theorem cheaper_store_price_in_cents :
  (sale_price_book_bargains - sale_price_readers_delight) * 100 = 1 :=
by
  sorry

end cheaper_store_price_in_cents_l432_432946


namespace knights_count_l432_432618

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432618


namespace arithmetic_sequence_value_l432_432133

theorem arithmetic_sequence_value (a : ℕ) (h : 2 * a = 12) : a = 6 :=
by
  sorry

end arithmetic_sequence_value_l432_432133


namespace knights_count_in_meeting_l432_432626

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432626


namespace ellipse_parameters_l432_432034

theorem ellipse_parameters :
  ∃ a b h k : ℝ, 
    (a = 8 * Real.sqrt 2 ∧ b = 11 ∧ h = 1 ∧ k = 6) ∧
    let f1 := (1, 3)
    let f2 := (1, 9)
    let p := (10, 1)
    let d1 := Real.sqrt ((10 - 1)^2 + (1 - 3)^2)
    let d2 := Real.sqrt ((10 - 1)^2 + (1 - 9)^2)
    let c := (1, 6)
    let major_axis := 22
    let minor_axis := Real.sqrt (22^2 - 6^2)
    let h1 := (x - 1)^2
    let k1 := (y - 6)^2 in
    ((x - h)^2 / (8 * Real.sqrt 2)^2) + ((y - k)^2 / 11^2) = 1 :=
by
  sorry

end ellipse_parameters_l432_432034


namespace impossible_to_tile_9x9_with_2x1_dominos_possible_to_tile_9x9_with_3x1_triominos_possible_to_tile_9x9_with_L_shaped_polyominoes_l432_432773

/-- Prove that it is impossible to tile a 9 × 9 chessboard using only 2 × 1 dominos. -/
theorem impossible_to_tile_9x9_with_2x1_dominos :
  ¬ (∃ (f : Fin 9 × Fin 9 → Fin 2 × Fin 1), Bijective f) := sorry

/-- Prove that it is possible to tile a 9 × 9 chessboard using 3 × 1 triominos. -/
theorem possible_to_tile_9x9_with_3x1_triominos :
  ∃ (f : Fin 9 × Fin 9 → Fin 3 × Fin 1), Bijective f := sorry

/-- Prove that it is possible to tile a 9 × 9 chessboard using L-shaped polyominoes. -/
theorem possible_to_tile_9x9_with_L_shaped_polyominoes :
  ∃ (f : (Fin 9 × Fin 9 → ℕ) ⊓ (∑ b in Finset.finRange 9 ×ˢ Finset.finRange 9, f b = 81), Bijective f := sorry


end impossible_to_tile_9x9_with_2x1_dominos_possible_to_tile_9x9_with_3x1_triominos_possible_to_tile_9x9_with_L_shaped_polyominoes_l432_432773


namespace hexagon_area_l432_432648

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_convex_equilateral_hexagon (A B C D E F : Point) : Prop :=
  -- Add the conditions that specify a convex equilateral hexagon here
  sorry

def is_parallel (L1 L2 : Line) : Prop :=
  -- Add the condition for parallel lines here
  sorry

def distinct_y_coordinates (points : list Point) (ys : set ℝ) : Prop :=
  points.map (λ p => p.y) = ys.to_list ∧ points.nodup

noncomputable def area_of_hexagon (A B C D E F : Point) : ℝ :=
  -- Calculate the area of the hexagon given vertices A, B, C, D, E, F
  sorry

theorem hexagon_area :
  ∃ (A B C D E F : Point) (c : ℝ) (p q : ℕ),
    A = ⟨0, 0⟩ ∧
    B = ⟨c, 3⟩ ∧
    is_convex_equilateral_hexagon A B C D E F ∧
    ∠FAB = 120 ∧
    is_parallel (Line.mk A B) (Line.mk D E) ∧
    is_parallel (Line.mk B C) (Line.mk E F) ∧
    is_parallel (Line.mk C D) (Line.mk F A) ∧
    distinct_y_coordinates [A, B, C, D, E, F] {0, 3, 6, 9, 12, 15} ∧
    area_of_hexagon A B C D E F = p * real.sqrt q ∧
    p + q = 147 :=
begin
  sorry
end

end hexagon_area_l432_432648


namespace smallest_possible_S_l432_432764

/-- Definition of 8-sided dice roll sum transformation --/
def transformed_sum (n R : ℕ) : ℕ :=
  9 * n - R

/-- The specific problem statement --/
theorem smallest_possible_S (n R S : ℕ) 
  (h_dice_sides : ∀ i, 1 ≤ d_i ≤ 8) 
  (h_sum_1504 : R = 1504) 
  (h_transformed_sum : S = transformed_sum n R) 
  (h_min_n : n = 188) : 
  S = 188 :=
sorry

end smallest_possible_S_l432_432764


namespace seating_arrangements_exactly_two_adjacent_empty_l432_432745

theorem seating_arrangements_exactly_two_adjacent_empty :
  let seats := 6
  let people := 3
  let arrangements := (seats.factorial / (seats - people).factorial)
  let non_adj_non_empty := ((seats - people).choose people * people.factorial)
  let all_adj_empty := ((seats - (people + 1)).choose 1 * people.factorial)
  arrangements - non_adj_non_empty - all_adj_empty = 72 := by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_l432_432745


namespace problem_statement_l432_432530

def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := sin (ω * x + ϕ)

theorem problem_statement (ω ϕ : ℝ) (h1 : ω > 0) (h2 : |ϕ| < π / 2)
    (h3 : f (π / 4) ω ϕ = 1) (h4 : f (7 * π / 12) ω ϕ = -1) :
    (ω = 3) ∧ (ϕ = -π / 4) ∧ (f x ω ϕ = sin (3 * x - π / 4)) := 
sorry

end problem_statement_l432_432530


namespace number_of_divisors_of_215_7_are_perfect_squares_or_cubes_l432_432980

-- Definition of the problem conditions
def factor_215 (n : Nat) : Prop := n = 5 * 43
def power_215_7 (n : Nat) : Prop := n = 215 ^ 7
def divisor_of_215_7 (a b : Nat) : Prop := 0 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 7

-- Statement of the proof problem
theorem number_of_divisors_of_215_7_are_perfect_squares_or_cubes 
  (h1 : ∀ n, factor_215 n → n = 215)
  (h2 : ∀ n, power_215_7 n → n = 215 ^ 7)
  : Σ' (num : Nat), (∀ a b, divisor_of_215_7 a b → Nat := {x : Nat // x = 21}) := 
sorry -- Proof is omitted

end number_of_divisors_of_215_7_are_perfect_squares_or_cubes_l432_432980


namespace decreasing_interval_minimum_value_l432_432528

-- Definitions provided in the problem conditions
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x
def g (a : ℝ) (x : ℝ) : ℝ := f(a, x) + x

-- The first statement to prove
theorem decreasing_interval (a : ℝ) (h : (1 - a) = -1) : 
  (∀ x, 0 < x ∧ x < 2 → deriv (λ x => g a x) x < 0) :=
sorry

-- The second statement to prove
theorem minimum_value (h : ∀ x, 0 < x ∧ x < 0.5 → f(a, x) > 0) : 
  a ≥ 2 - 4 * log 2 :=
sorry

end decreasing_interval_minimum_value_l432_432528


namespace moduli_product_l432_432467

theorem moduli_product (z1 z2 : ℂ) (h1 : z1 = 4 - 3 * complex.I) (h2 : z2 = 4 + 3 * complex.I) : complex.abs z1 * complex.abs z2 = 25 := 
by
  rw [h1, h2]
  -- simplify abs (4 - 3i) * abs (4 + 3i)
  have : |4 - 3*complex.I| * |4 + 3*complex.I| = complex.abs ((4 - 3*complex.I) * (4 + 3*complex.I)) := complex.abs_mul (4 - 3*complex.I) (4 + 3*complex.I)
  rw [this]
  -- (4 - 3i) * (4 + 3i) = 25
  have : (4 - 3*complex.I) * (4 + 3*complex.I) = 25 := by 
    rw [←complex.mul_conj, complex.norm_sq_eq_conj_mul_self]
    simp [complex.norm_sq]
  rw [this]
  -- the modulus of 25 is 25
  rw [complex.abs_assoc, complex.abs_of_real, complex.abs_eq_abs_of_nonneg]
  norm_num
  sorry

end moduli_product_l432_432467


namespace rectangle_coefficient_delta_l432_432756

theorem rectangle_coefficient_delta (DE EF FD : ℝ) (θ : ℝ) (p q : ℤ) :
  DE = 13 → 
  EF = 30 →
  FD = 23 →
  (∀ θ, θ ≠ 0 → ∃ γ δ, (γ θ - δ θ^2 = 0)) →
  γ = 30 * δ →
  ∃ s, s = (DE + EF + FD) / 2 ∧ 
  ∃ A, A = sqrt (s * (s - DE) * (s - EF) * (s - FD)) ∧ 
  Area (triangle DEF) = A ∧ 
  ∀ θ, θ ≠ 0 → Area (rectangle WXYZ) = 1/2 * Area (triangle DEF) →
  δ = A / (225 * θ^2) →
  δ = √2 / 5 →
  p = 1 → 
  q = 5 →
  p + q = 6 := by sorry

end rectangle_coefficient_delta_l432_432756


namespace find_x_value_l432_432098

theorem find_x_value :
  (∀ x: ℚ, 3 / x * (∏ n in finset.Icc 3 120, (1 + 1 / (n: ℚ))) = 11) -> x = 11 / 30 :=
by
  assume h: (∀ x: ℚ, 3 / x * (∏ n in finset.Icc 3 120, (1 + 1 / (n: ℚ))) = 11)
  sorry

end find_x_value_l432_432098


namespace indian_children_percentage_proof_l432_432572

noncomputable def percentage_of_indian_children
  (total_men total_women total_children : ℕ)
  (percentage_indian_men percentage_indian_women percentage_non_indians : ℚ) : ℚ :=
  let total_people := total_men + total_women + total_children in
  let indian_men := percentage_indian_men * ↑total_men in
  let indian_women := percentage_indian_women * ↑total_women in
  let non_indians := percentage_non_indians * ↑total_people in
  let total_indians := ↑total_people - non_indians in
  let indian_children := total_indians - (indian_men + indian_women) in
  indian_children / ↑total_children * 100

theorem indian_children_percentage_proof :
  percentage_of_indian_children 500 300 500 0.10 0.60 0.5538461538461539 = 70 := 
sorry

end indian_children_percentage_proof_l432_432572


namespace janet_gas_usage_l432_432604

variable (distance_dermatologist distance_gynecologist mpg : ℕ)

theorem janet_gas_usage
  (h_distance_dermatologist : distance_dermatologist = 30)
  (h_distance_gynecologist : distance_gynecologist = 50)
  (h_mpg : mpg = 20) :
  (2 * distance_dermatologist + 2 * distance_gynecologist) / mpg = 8 := 
by
  rw [h_distance_dermatologist, h_distance_gynecologist, h_mpg]
  linarith
  sorry

end janet_gas_usage_l432_432604


namespace apollonius_bisector_property_l432_432036

open Real

def point := ℝ × ℝ

noncomputable def distance (p₁ p₂ : point) : ℝ := sqrt ((p₁.1 - p₂.1) ^ 2 + (p₂.2 - p₂.2) ^ 2)

theorem apollonius_bisector_property 
  (A B P : point)
  (hA : A = (-2, 0))
  (hB : B = (4, 0))
  (h_ratio : distance P A / distance P B = 1 / 2)
  (h_nonlinear : ¬ collinear ℝ [A, B, P])
  (O : point)
  (hO : O = (0, 0)) :
  (angle P O A = angle P O B) :=
sorry

end apollonius_bisector_property_l432_432036


namespace liters_to_milliliters_cubic_decimeters_to_cubic_centimeters_cubic_decimeters_to_liters_milliliters_cubic_meters_to_cubic_meters_decimeters_l432_432464

theorem liters_to_milliliters (liters milliliters : ℕ) (h : liters = 4) (h' : milliliters = 25) :
  4 * 1000 + 25 = 4025 :=
by sorry

theorem cubic_decimeters_to_cubic_centimeters (cubic_dm : ℝ) (h : cubic_dm = 6.09) :
  cubic_dm * 1000 = 6090 :=
by sorry

theorem cubic_decimeters_to_liters_milliliters (cubic_dm : ℝ) (h : cubic_dm = 4.9) (L M : ℕ) (hL : L = 4) (hM : M = 900) :
  L * 1000 + M = 4.9 * 1000 :=
by sorry

theorem cubic_meters_to_cubic_meters_decimeters (cubic_m : ℝ) (h : cubic_m = 2.03) (M D : ℕ) (hM : M = 2) (hD : D = 30) :
  M * 1000 + D = 2.03 * 1000 :=
by sorry

end liters_to_milliliters_cubic_decimeters_to_cubic_centimeters_cubic_decimeters_to_liters_milliliters_cubic_meters_to_cubic_meters_decimeters_l432_432464


namespace inverse_function_log_base_l432_432504

noncomputable def f (a x : ℝ) : ℝ := Real.logBase a x

theorem inverse_function_log_base
  (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
  (h₃ : ∀ x, f a x = Real.logBase a x)
  (h₄ : ∀ y, Real.exp (f (1/a) y) = y)
  (h₅ : ∀ y, 2 = (a: ℝ) ^ (-1) → a = (1/2))
  (inv_f_at_neg1 : ∀ (x : ℝ), f a x = -1 → x = 2) :
  ∀ (x : ℝ), (f a (Real.exp x) = x) → Real.exp x = (1/2) ^ x := sorry

end inverse_function_log_base_l432_432504


namespace plate_arrangement_l432_432794

def arrangements_without_restriction : Nat :=
  Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 3)

def arrangements_adjacent_green : Nat :=
  (Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3)) * Nat.factorial 3

def allowed_arrangements : Nat :=
  arrangements_without_restriction - arrangements_adjacent_green

theorem plate_arrangement : 
  allowed_arrangements = 2520 := 
by
  sorry

end plate_arrangement_l432_432794


namespace divisors_91837_l432_432171

/-- 
  The integer 91837 has exactly one divisor in the range {1, 2, 3, 4, 5, 6, 7, 8, 9}, which is 1.
-/
theorem divisors_91837 : ∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → n ∣ 91837 ↔ n = 1 := by
  intro n
  intro hn
  cases hn with
  | inl h => 
    exact ⟨λ h2 => by rfl, by intro _; exact dvd_refl n⟩
  | inr rest => 
    cases rest with
    | inl h => 
      have hmod : 91837 % 2 ≠ 0 := by decide
      have hdiv : ¬2 ∣ 91837 := by
        contrapose! hmod
        exact Nat.dvd_pos_of_ne_zero (by decide)
        simp [Int.natAbs_of_nonneg, hmod]
      exact ⟨λ h2 => False.elim (hdiv h2), λ h2 => False.elim (by decide)⟩
    | inr rest' => 
      cases rest' with
      -- Repeat similar steps for rest' elements 3, 4, 5, 6, 7, 8, and 9 
      -- as per given solution steps to fully define the failed divisibility by 3 through 9
      sorry

end divisors_91837_l432_432171


namespace number_of_correct_statements_l432_432850

theorem number_of_correct_statements : 
  let cond1 := "For two non-intersecting lines, the slopes of the two lines are equal is a necessary but not sufficient condition for the two lines are parallel"
  let cond2 := "The negation of the proposition ∀ x ∈ ℝ, sin x ≤ 1 is ∃ x₀ ∈ ℝ, sin x₀ > 1"
  let cond3 := "p and q are true is a sufficient but not necessary condition for p or q is true"
  let cond4 := "Given lines a, b and plane α, if a ⊥ α and b ∥ α, then a ⊥ b"
  cond1 = false ∧ cond2 = true ∧ cond3 = true ∧ cond4 = true → 
  number_of_correct_statements = 3 :=
by
  sorry

end number_of_correct_statements_l432_432850


namespace range_of_a_l432_432541

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x
noncomputable def g (x a : ℝ) : ℝ := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Set.Icc 0 2 → ∃ x2 : ℝ, x2 ∈ Set.Ioi a ∧ f x1 ≥ g x2 a) →
  a ≤ -1 :=
by
  intro h
  sorry

end range_of_a_l432_432541


namespace distinct_lines_isosceles_not_equilateral_l432_432732

-- Define a structure for an isosceles triangle that is not equilateral
structure IsoscelesButNotEquilateralTriangle :=
  (a b c : ℕ)    -- sides of the triangle
  (h₁ : a = b)   -- two equal sides
  (h₂ : a ≠ c)   -- not equilateral (not all three sides are equal)

-- Define that the number of distinct lines representing altitudes, medians, and interior angle bisectors is 5
theorem distinct_lines_isosceles_not_equilateral (T : IsoscelesButNotEquilateralTriangle) : 
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end distinct_lines_isosceles_not_equilateral_l432_432732


namespace correct_option_is_D_l432_432344

theorem correct_option_is_D :
  (¬("A quadrilateral with opposite sides that are parallel and equal is a parallelogram" ∧ 
     "A quadrilateral with opposite sides that are parallel and equal is not a universal proposition")) ∧
  (¬(∃ x : ℝ, x^2 + x + 4 ≤ 0) ∧ 
     (∀ x : ℝ, x^2 + x + 4 > 0)) ∧
  (¬(("a = b" is a necessary but not sufficient condition for "ac = bc") ∧ 
     ("a = b" is not a sufficient but necessary condition for "ac = bc"))) ∧
  ("a+5 is an irrational number" ↔ "a is an irrational number") :=
begin
  sorry
end

end correct_option_is_D_l432_432344


namespace field_day_difference_l432_432586

def class_students (girls boys : ℕ) := girls + boys

def grade_students 
  (class1_girls class1_boys class2_girls class2_boys class3_girls class3_boys : ℕ) :=
  (class1_girls + class2_girls + class3_girls, class1_boys + class2_boys + class3_boys)

def diff_students (g1 b1 g2 b2 g3 b3 : ℕ) := 
  b1 + b2 + b3 - (g1 + g2 + g3)

theorem field_day_difference :
  let g3_1 := 10   -- 3rd grade first class girls
  let b3_1 := 14   -- 3rd grade first class boys
  let g3_2 := 12   -- 3rd grade second class girls
  let b3_2 := 10   -- 3rd grade second class boys
  let g3_3 := 11   -- 3rd grade third class girls
  let b3_3 :=  9   -- 3rd grade third class boys
  let g4_1 := 12   -- 4th grade first class girls
  let b4_1 := 13   -- 4th grade first class boys
  let g4_2 := 15   -- 4th grade second class girls
  let b4_2 := 11   -- 4th grade second class boys
  let g4_3 := 14   -- 4th grade third class girls
  let b4_3 := 12   -- 4th grade third class boys
  let g5_1 :=  9   -- 5th grade first class girls
  let b5_1 := 13   -- 5th grade first class boys
  let g5_2 := 10   -- 5th grade second class girls
  let b5_2 := 11   -- 5th grade second class boys
  let g5_3 := 11   -- 5th grade third class girls
  let b5_3 := 14   -- 5th grade third class boys
  diff_students (g3_1 + g3_2 + g3_3 + g4_1 + g4_2 + g4_3 + g5_1 + g5_2 + g5_3)
                (b3_1 + b3_2 + b3_3 + b4_1 + b4_2 + b4_3 + b5_1 + b5_2 + b5_3) = 3 :=
by
  sorry

end field_day_difference_l432_432586


namespace evaluate_root_power_l432_432865

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l432_432865


namespace rigid_transformations_l432_432766

open EuclideanGeometry

-- Define points C, C', D, D'
def C : Point := (3, -2)
def C' : Point := (-3, 2)
def D : Point := (4, -5)
def D' : Point := (-4, 5)

-- Define transformations
def reflect_x (p : Point) : Point := (p.1, -p.2)
def rotate_90_clockwise (p : Point) : Point := (p.2, -p.1)
def translate (p : Point) (dx dy : ℝ) : Point := (p.1 + dx, p.2 + dy)
def reflect_y (p : Point) : Point := (-p.1, p.2)
def rotate_180_clockwise (p : Point) : Point := (-p.1, -p.2)

-- Hypotheses: the behavior of transformations on points C and D
def hypothesis1 := translate C (-6) 4 = C' ∧ translate D (-6) 4 = D'
def hypothesis2 := rotate_180_clockwise C = C' ∧ rotate_180_clockwise D = D'

-- Proof problem: Prove that only the translation by 6 units left and 4 units up and the 
-- clockwise rotation about the origin by 180 degrees satisfy the conditions.
theorem rigid_transformations (h1 : hypothesis1) (h2 : hypothesis2) :
  (translate C (-6) 4 = C' ∧ translate D (-6) 4 = D') ∧
  (rotate_180_clockwise C = C' ∧ rotate_180_clockwise D = D') :=
by 
  exact ⟨h1, h2⟩

end rigid_transformations_l432_432766


namespace smallest_n_is_64_l432_432493

noncomputable def smallest_n_satisfying (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∑ i in finset.range n, x i) = 800 ∧ (∑ i in finset.range n, (x i) ^ 4) = 409600

theorem smallest_n_is_64 :
  ∃ n : ℕ, ∀ x : ℕ → ℝ, smallest_n_satisfying n x → n = 64 :=
by
  sorry

end smallest_n_is_64_l432_432493


namespace charge_move_increases_energy_l432_432717

noncomputable def energy_increase_when_charge_moved : ℝ :=
  let initial_energy := 15
  let energy_per_pair := initial_energy / 3
  let new_energy_AB := energy_per_pair
  let new_energy_AC := 2 * energy_per_pair
  let new_energy_BC := 2 * energy_per_pair
  let final_energy := new_energy_AB + new_energy_AC + new_energy_BC
  final_energy - initial_energy

theorem charge_move_increases_energy :
  energy_increase_when_charge_moved = 10 :=
by
  sorry

end charge_move_increases_energy_l432_432717


namespace find_sin_alpha_l432_432106

variable {α : ℝ}

theorem find_sin_alpha (h1 : sin (α - π / 4) = (7 * sqrt 2) / 10) (h2 : cos (2 * α) = 7 / 25) : sin α = 3 / 5 :=
sorry

end find_sin_alpha_l432_432106


namespace factorization_l432_432875

theorem factorization (y : ℝ) : 1 - 4 * y^2 = (1 - 2 * y) * (1 + 2 * y) :=
by sorry

end factorization_l432_432875


namespace janet_gas_usage_l432_432605

variable (distance_dermatologist distance_gynecologist mpg : ℕ)

theorem janet_gas_usage
  (h_distance_dermatologist : distance_dermatologist = 30)
  (h_distance_gynecologist : distance_gynecologist = 50)
  (h_mpg : mpg = 20) :
  (2 * distance_dermatologist + 2 * distance_gynecologist) / mpg = 8 := 
by
  rw [h_distance_dermatologist, h_distance_gynecologist, h_mpg]
  linarith
  sorry

end janet_gas_usage_l432_432605


namespace angle_range_between_vectors_l432_432938

-- Define the data types for vectors and their properties
variables (a b : ℝ^3) (x : ℝ)

-- Define the conditions given in the problem statement
def non_zero_vectors : Prop := a ≠ 0 ∧ b ≠ 0
def magnitude_relation : Prop := ∥a∥ = 2 * ∥b∥
def has_extreme_values (f : ℝ → ℝ) : Prop := ∀ (x : ℝ), ∃ y, f(y) < f(x)

-- Define the function given in the problem
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * ∥a∥ * x^2 + (a ⋅ b) * x + 1

-- Translate the problem statement to a proof statement
theorem angle_range_between_vectors (h1 : non_zero_vectors a b) (h2 : magnitude_relation a b) (h3 : has_extreme_values (f a b)) : 
  ∃ θ, (θ > real.pi / 3 ∧ θ ≤ real.pi) := 
sorry

end angle_range_between_vectors_l432_432938


namespace series_sum_l432_432419

theorem series_sum : (\sum i in Finset.range 1998, (1 : ℚ) / ((i + 1) * (i + 2))) = 1998 / 1999 := 
by
  sorry

end series_sum_l432_432419


namespace sin_half_inequality_l432_432267

-- Define the concept of an arbitrary triangle with sides a, b, c and angles A, B, C.
structure Triangle :=
  (A B C : ℝ) -- angles of the triangle
  (a b c : ℝ) -- sides of the triangle
  (A_pos : 0 < A) (B_pos : 0 < B) (C_pos : 0 < C)
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (angle_sum : A + B + C = π) -- sum of angles in a triangle

noncomputable def sin_half := λ (θ : ℝ), Real.sin (θ / 2)

theorem sin_half_inequality (T : Triangle) : 
  sin_half T.A ≤ T.a / (T.b + T.c) :=
sorry

end sin_half_inequality_l432_432267


namespace billy_questions_third_hour_l432_432045

variable (x : ℝ)
variable (questions_in_first_hour : ℝ := x)
variable (questions_in_second_hour : ℝ := 1.5 * x)
variable (questions_in_third_hour : ℝ := 3 * x)
variable (total_questions_solved : ℝ := 242)

theorem billy_questions_third_hour (h : questions_in_first_hour + questions_in_second_hour + questions_in_third_hour = total_questions_solved) :
  questions_in_third_hour = 132 :=
by
  sorry

end billy_questions_third_hour_l432_432045


namespace ron_total_tax_l432_432436

def car_price : ℝ := 30000
def first_tier_level : ℝ := 10000
def first_tier_rate : ℝ := 0.25
def second_tier_rate : ℝ := 0.15

def first_tier_tax : ℝ := first_tier_level * first_tier_rate
def second_tier_tax : ℝ := (car_price - first_tier_level) * second_tier_rate
def total_tax : ℝ := first_tier_tax + second_tier_tax

theorem ron_total_tax : 
  total_tax = 5500 := by
  -- Proof will be provided here
  sorry

end ron_total_tax_l432_432436


namespace magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432470

noncomputable def magnitude_prod_4_minus_3i_4_plus_3i : ℝ := |complex.abs (4 - 3 * complex.I) * complex.abs (4 + 3 * complex.I)|

theorem magnitude_prod_4_minus_3i_4_plus_3i_eq_25 : magnitude_prod_4_minus_3i_4_plus_3i = 25 :=
by
  sorry

end magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432470


namespace eval_sqrt_pow_l432_432868

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l432_432868


namespace part_I_part_II_l432_432156

noncomputable def f (x : ℝ) := x * (Real.log x - 1) + Real.log x + 1

theorem part_I :
  let f_tangent (x y : ℝ) := x - y - 1
  (∀ x y, f_tangent x y = 0 ↔ y = x - 1) ∧ f_tangent 1 (f 1) = 0 :=
by
  sorry

theorem part_II (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1 / x)) + 1 ≥ 0) → m ≥ -1 :=
by
  sorry

end part_I_part_II_l432_432156


namespace domain_of_sqrt_tan_l432_432092

theorem domain_of_sqrt_tan :
  ∀ x : ℝ, (∃ k : ℤ, k * π ≤ x ∧ x < k * π + π / 2) ↔ 0 ≤ (Real.tan x) :=
sorry

end domain_of_sqrt_tan_l432_432092


namespace crossing_time_correct_l432_432349

def length_of_train : ℝ := 150 -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 72 -- Speed of the train in km/hr
def length_of_bridge : ℝ := 132 -- Length of the bridge in meters

noncomputable def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600 -- Speed of the train in m/s

noncomputable def time_to_cross_bridge : ℝ := (length_of_train + length_of_bridge) / speed_of_train_m_per_s -- Time in seconds

theorem crossing_time_correct : time_to_cross_bridge = 14.1 := by
  sorry

end crossing_time_correct_l432_432349


namespace monotonicity_of_f_range_of_k_for_three_zeros_l432_432535

noncomputable def f (x k : ℝ) := x^3 - k * x + k^2

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_of_f (k : ℝ) :
  (k ≤ 0 → ∀ x y : ℝ, x ≤ y → f x k ≤ f y k) ∧ 
  (k > 0 → (∀ x : ℝ, x < -sqrt (k / 3) → f x k < f (-sqrt (k / 3)) k) ∧ 
            (∀ x : ℝ, x > sqrt (k / 3) → f x k > f (sqrt (k / 3)) k) ∧
            (f (-sqrt (k / 3)) k > f (sqrt (k / 3)) k)) :=
  sorry

-- Problem 2: Range of k for f(x) to have three zeros
theorem range_of_k_for_three_zeros (k : ℝ) : 
  (∃ a b c : ℝ, f a k = 0 ∧ f b k = 0 ∧ f c k = 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) ↔ (0 < k ∧ k < 4 / 27) :=
  sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l432_432535


namespace problem1_problem2_l432_432673

theorem problem1 (x : ℝ) (a : ℝ) (h : a = 1) (hp : a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) : 2 < x ∧ x < 3 := 
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (hp : 0 < a ∧ a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) (hsuff : ∀ (a x : ℝ), (2 < x ∧ x < 3) → a < x ∧ x < 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end problem1_problem2_l432_432673


namespace minimum_value_nS_n_l432_432676

variable (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)

-- Arithmetic sequence definition and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d a1, d = a 2 - a 1 ∧ ∀ n, a (n+1) = a 1 + n * d

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition1 : Prop := S 10 = 40
def condition2 : Prop := a 5 = 3

-- Mathematical problem to solve
theorem minimum_value_nS_n : 
  arithmetic_sequence a → 
  sum_arithmetic_sequence S a → 
  condition1 → 
  condition2 →
  ∃ n, ∀ m, m ≠ n → n * S n ≤ m * S m ∧ n * S n = -32 :=
by
  sorry

end minimum_value_nS_n_l432_432676


namespace park_attraction_children_count_l432_432213

theorem park_attraction_children_count
  (C : ℕ) -- Number of children
  (entrance_fee : ℕ := 5) -- Entrance fee per person
  (kids_attr_fee : ℕ := 2) -- Attraction fee for kids
  (adults_attr_fee : ℕ := 4) -- Attraction fee for adults
  (parents : ℕ := 2) -- Number of parents
  (grandmother : ℕ := 1) -- Number of grandmothers
  (total_cost : ℕ := 55) -- Total cost paid
  (entry_eq : entrance_fee * (C + parents + grandmother) + kids_attr_fee * C + adults_attr_fee * (parents + grandmother) = total_cost) :
  C = 4 :=
by
  sorry

end park_attraction_children_count_l432_432213


namespace fox_cub_distribution_l432_432668

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l432_432668


namespace distribution_plans_correct_l432_432743

-- Defining the problem statement
def outstanding_spots_distribution : ℕ :=
  (Finset.card ((Finset.powersetLen 5 (Finset.range 9)).card))

theorem distribution_plans_correct : outstanding_spots_distribution = 126 := 
  by 
    -- proof will go here
    sorry

end distribution_plans_correct_l432_432743


namespace second_person_time_l432_432757

theorem second_person_time (x : ℝ) (h1 : ∀ t : ℝ, t = 3) 
(h2 : (1/3 + 1/x) = 5/12) : x = 12 := 
by sorry

end second_person_time_l432_432757


namespace team_a_builds_per_day_l432_432748

theorem team_a_builds_per_day (x : ℝ) (h1 : (150 / x = 100 / (2 * x - 30))) : x = 22.5 := by
  sorry

end team_a_builds_per_day_l432_432748


namespace problem1_problem2_problem3_l432_432540

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + 2 * |x + 1|

theorem problem1 :
  {x : ℝ | f x ≤ 5} = set.Icc (-3/2) 1 := sorry

theorem problem2 (m : ℝ) (x₀ : ℝ) :
  f x₀ ≤ 5 + m - m^2 → m ≤ 2 := sorry

theorem problem3 (a b : ℝ) (h : a^3 + b^3 = 2) :
  0 < a + b ∧ a + b ≤ 2 := sorry

end problem1_problem2_problem3_l432_432540


namespace geometry_problem_l432_432423

-- Define the points, circles, and specific conditions
variables 
  (W1 W2 : Type) [MetricSpace W1] [MetricSpace W2]
  (D P A B C M : W1)
  (AB : Line W1) (AD : Line W1)
  (W1_circle : Circle W1) (W2_circle : Circle W2)

-- Conditions of the problem
axioms 
  (h1 : W1_circle ∩ W2_circle = {D, P})
  (h2 : Line.isTangentToCircle AB W1_circle A)
  (h3 : Line.isTangentToCircle AB W2_circle B)
  (h4 : Distance(D, AB) < Distance(P, AB))
  (h5 : AD.intersectsCircleAgain W2_circle C)
  (h6 : Midpoint(M, B, C))

-- The theorem we want to prove
theorem geometry_problem : ∠(D, P, M) = ∠(B, D, C) :=
by 
  sorry

end geometry_problem_l432_432423


namespace movie_ticket_ratio_l432_432247

theorem movie_ticket_ratio
  (cost_Monday : ℕ) (cost_Saturday : ℕ) (total_cost : ℕ)
  (h1 : cost_Monday = 5)
  (h2 : cost_Saturday = 5 * cost_Monday)
  (h3 : total_cost = 35)
  (h4 : ∀ W: ℕ, W + cost_Saturday = total_cost) :
  ∃ W: ℕ, W = 10 ∧ (W / cost_Monday = 2) := 
begin
  use 10,
  split,
  {
    have hW : 10 + cost_Saturday = total_cost,
    calc
      10 + cost_Saturday = 10 + 25 : by rw h2; rw h1
      ... = 35 : by rw add_comm; rw h3,
    exact h4 10
  },
  {
    calc
      10 / cost_Monday = 10 / 5 : by rw h1
      ... = 2 : by norm_num
  }
end

end movie_ticket_ratio_l432_432247


namespace point_in_fourth_quadrant_l432_432210

theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : 
(x > 0) → (y < 0) → (x, y) = (2, -3) → quadrant (2, -3) = 4 :=
by
  sorry

end point_in_fourth_quadrant_l432_432210


namespace soccer_field_illumination_l432_432378

noncomputable def diagonal (l w : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2)

noncomputable def min_ceiling_height (l w : ℝ) : ℝ :=
  Real.ceil ((diagonal l w) / 4 * 10) / 10

theorem soccer_field_illumination :
  min_ceiling_height 90 60 = 27.1 :=
by
  sorry

end soccer_field_illumination_l432_432378


namespace modulus_product_l432_432476

theorem modulus_product (a b : ℂ) : |a - b * complex.i| * |a + b * complex.i| = 25 := by
  have h1 : complex.norm (4 - 3 * complex.i) = 5 := by
    sorry
  have h2 : complex.norm (4 + 3 * complex.i) = 5 := by
    sorry
  rw [← complex.norm_mul, (4 - 3 * complex.i).mul_conj_self, (4 + 3 * complex.i).mul_conj_self, add_comm] at h1
  rw [mul_comm, mul_comm (complex.norm _), ← mul_assoc, h2, mul_comm, mul_assoc]
  exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).1 h1 

end modulus_product_l432_432476


namespace distance_between_X_and_Y_l432_432255

def distance_XY := 31

theorem distance_between_X_and_Y
  (yolanda_rate : ℕ) (bob_rate : ℕ) (bob_walked : ℕ) (time_difference : ℕ) :
  yolanda_rate = 1 →
  bob_rate = 2 →
  bob_walked = 20 →
  time_difference = 1 →
  distance_XY = bob_walked + (bob_walked / bob_rate + time_difference) * yolanda_rate :=
by
  intros hy hb hbw htd
  sorry

end distance_between_X_and_Y_l432_432255


namespace perfect_square_2n_plus_65_l432_432486

theorem perfect_square_2n_plus_65 (n : ℕ) (h : n > 0) : 
  (∃ m : ℕ, m * m = 2^n + 65) → n = 4 ∨ n = 10 :=
by 
  sorry

end perfect_square_2n_plus_65_l432_432486


namespace units_digit_sum_product_l432_432049

theorem units_digit_sum_product :
  let odd_integers := list.range' 1 200 |>.filter (λ n, n % 2 = 1) in
  (lsum odd_integers) % 10 = 0 ∧ (lprod odd_integers) % 10 = 5 :=
by
  let odd_integers := list.range' 1 200 |>.filter (λ n, n % 2 = 1) 
  have : (lsum odd_integers) % 10 = 0 := sorry
  have : (lprod odd_integers) % 10 = 5 := sorry
  exact ⟨this, this_1⟩

end units_digit_sum_product_l432_432049


namespace balls_total_correct_l432_432596

-- Definitions based on the problem conditions
def red_balls_initial : ℕ := 16
def blue_balls : ℕ := 2 * red_balls_initial
def red_balls_lost : ℕ := 6
def red_balls_remaining : ℕ := red_balls_initial - red_balls_lost
def total_balls_after : ℕ := 74
def nonblue_red_balls_remaining : ℕ := red_balls_remaining + blue_balls

-- Goal: Find the number of yellow balls
def yellow_balls_bought : ℕ := total_balls_after - nonblue_red_balls_remaining

theorem balls_total_correct :
  yellow_balls_bought = 32 :=
by
  -- Proof would go here
  sorry

end balls_total_correct_l432_432596


namespace max_plus_min_value_of_y_eq_neg4_l432_432539

noncomputable def y (x : ℝ) : ℝ := (2 * (Real.sin x) ^ 2 + Real.sin (3 * x / 2) - 4) / ((Real.sin x) ^ 2 + 2 * (Real.cos x) ^ 2)

theorem max_plus_min_value_of_y_eq_neg4 (M m : ℝ) (hM : ∃ x : ℝ, y x = M) (hm : ∃ x : ℝ, y x = m) :
  M + m = -4 := sorry

end max_plus_min_value_of_y_eq_neg4_l432_432539


namespace binomial_expectation_variance_l432_432501

noncomputable theory

open ProbabilityTheory

variables {n : ℕ} {p : ℚ} 

theorem binomial_expectation_variance 
  (X : ℕ → ℚ) (hn : X ∼ binomial n p) 
  (hE : 3 * (E[X] : ℚ) - 9 = 27)
  (hD : 9 * (Var[X] : ℚ) = 27) :
  n = 16 ∧ p = 3 / 4 := 
sorry

end binomial_expectation_variance_l432_432501


namespace isosceles_trapezoid_area_l432_432817

noncomputable def area_of_isosceles_trapezoid (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x : ℝ := 10 in
  let y : ℝ := 0.2 * x in
  let h : ℝ := 0.6 * x in
  1 / 2 * (y + longer_base) * h

theorem isosceles_trapezoid_area :
  ∀ (longer_base : ℝ) (base_angle : ℝ),
    (longer_base = 18) → (base_angle = Real.arcsin 0.6) → area_of_isosceles_trapezoid longer_base base_angle = 60 := 
by
  intros longer_base base_angle h_base h_angle
  rw [h_base, h_angle]
  unfold area_of_isosceles_trapezoid
  norm_num
  rw [Real.sin_arcsin (le_of_lt (by norm_num1)) (by norm_num1)]
  norm_num
  sorry

end isosceles_trapezoid_area_l432_432817


namespace bug_final_position_after_8_moves_l432_432368

-- Define the initial position and the sequence of moves
def initial_position : (ℝ × ℝ) := (1, 1)

-- Define the sequence of moves with each turn and movement distance adjustment.
def move_sequence : List (ℝ × ℝ) := [
  (-2, 0),   -- 2 units left
  (0, 1),    -- 1 unit up
  (0.5, 0),  -- 0.5 units right
  (0, -0.25),-- 0.25 units down
  (-0.125, 0), -- 0.125 units left
  (0, 0.0625), -- 0.0625 units up
  (0.03125, 0), -- 0.03125 units right
  (0, -0.015625) -- 0.015625 units down
]

-- Calculate the final position after applying all moves
def final_position (initial : (ℝ × ℝ)) (moves : List (ℝ × ℝ)) : (ℝ × ℝ) :=
  moves.foldl (λ acc, λ move, (acc.1 + move.1, acc.2 + move.2)) initial

theorem bug_final_position_after_8_moves :
  final_position initial_position move_sequence = (-0.59375, 1.796875) :=
by
  sorry

end bug_final_position_after_8_moves_l432_432368


namespace common_ratio_of_geometric_series_l432_432307

theorem common_ratio_of_geometric_series (a r S : Real) (h1 : S = a / (1 - r))
  (h2 : r^2 * S = S / 8) : r = sqrt 2 / 4 ∨ r = -sqrt 2 / 4 := by
  sorry

end common_ratio_of_geometric_series_l432_432307


namespace rectangular_to_cylindrical_conversion_l432_432432

def convert_rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (r, θ, z)

theorem rectangular_to_cylindrical_conversion :
  convert_rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end rectangular_to_cylindrical_conversion_l432_432432


namespace cos_angle_correct_l432_432142

noncomputable def cos_angle_F1PF2 : ℝ :=
  let F₁ := (-2 : ℝ, 0 : ℝ)
  let F₂ := (2 : ℝ, 0 : ℝ)
  let P := (3 * Real.sqrt 2 / 2, Real.sqrt 2 / 2)

  let PF₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let PF₂ := (F₂.1 - P.1, F₂.2 - P.2)

  let dot_product := PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2
  let norm_PF₁ := Real.sqrt ((PF₁.1) * (PF₁.1) + (PF₁.2) * (PF₁.2))
  let norm_PF₂ := Real.sqrt ((PF₁.1) * (PF₂.1) + (PF₁.2) * (PF₂.2))

  dot_product / (norm_PF₁ * norm_PF₂)

theorem cos_angle_correct :
  cos_angle_F1PF2 = 1 / 3 :=
by
  sorry

end cos_angle_correct_l432_432142


namespace AQI_analysis_l432_432822

def AQI_Chaoyang : List ℕ := [167, 61, 79, 78, 97, 153, 59, 179, 85, 209]
def AQI_Nanguan : List ℕ := [74, 54, 47, 47, 43, 43, 59, 104, 119, 251]

def categorize_AQI (AQI: ℕ) : String :=
  if AQI ≤ 50 then "Excellent"
  else if AQI ≤ 100 then "Good"
  else if AQI ≤ 150 then "Mild Pollution"
  else if AQI ≤ 200 then "Moderate Pollution"
  else "Severe Pollution"

def calculate_median (data: List ℕ) : Float :=
  let sorted := data.qsort (≤)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)).toFloat / 2
  else
    sorted.get (sorted.length / 2).toFloat

theorem AQI_analysis :
  let categories_Chaoyang := AQI_Chaoyang.map categorize_AQI
  let counts_Chaoyang := [
    categories_Chaoyang.count (λ c => c = "Excellent"),
    categories_Chaoyang.count (λ c => c = "Good"),
    categories_Chaoyang.count (λ c => c = "Mild Pollution"),
    categories_Chaoyang.count (λ c => c = "Moderate Pollution"),
    categories_Chaoyang.count (λ c => c = "Severe Pollution")
  ]
  let median_Nanguan := calculate_median AQI_Nanguan
  counts_Chaoyang = [0, 6, 0, 3, 1] ∧ median_Nanguan = 56.5 :=
by
  sorry

end AQI_analysis_l432_432822


namespace executiveCommittee_ways_l432_432249

noncomputable def numberOfWaysToFormCommittee (totalMembers : ℕ) (positions : ℕ) : ℕ :=
Nat.choose (totalMembers - 1) (positions - 1)

theorem executiveCommittee_ways : numberOfWaysToFormCommittee 30 5 = 25839 := 
by
  -- skipping the proof as it's not required
  sorry

end executiveCommittee_ways_l432_432249


namespace eccentricity_of_ellipse_equation_of_ellipse_and_point_P_l432_432947

-- Define the given conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ ∀ x y : ℝ, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1
def point_A (b : ℝ) : ℝ × ℝ := (0, b)
def distance_from_origin_to_FA (b : ℝ) : ℝ := (sqrt 2 / 2) * b

-- Define the eccentricity calculation condition
def eccentricity_condition (a b e : ℝ) : Prop :=
  b = sqrt (1 - e ^ 2) * a ∧ e = sqrt 2 / 2

-- Define the conditions for the reflection and coordinates of P
def symmetric_point (a : ℝ) : Prop :=
  (∀ x₀ y₀ : ℝ, 
    (y₀ / (x₀ + (sqrt 2 / 2) * a) = 1 / 2 ∧ 
    2 * (x₀ - (sqrt 2 / 2) * a) / 2 + y₀ / 2 = 0) ∧
    (x₀ ^ 2 + y₀ ^ 2 = 4) ∧
    (x₀ = 3 * sqrt 2 / 10 * a) ∧
    (y₀ = 4 * sqrt 2 / 10 * a) ∧
    (a ^ 2 = 8) ∧
    ((1 - e ^ 2) * a ^ 2 = 4))

-- Final theorem statements
theorem eccentricity_of_ellipse (a b e : ℝ) (h : ellipse a b) 
  (h' : distance_from_origin_to_FA b = (sqrt 2 / 2) * b ∧ eccentricity_condition a b e) :
  e = sqrt 2 / 2 :=
sorry

theorem equation_of_ellipse_and_point_P (a b x₀ y₀ : ℝ) (h : ellipse a b) 
  (h' : symmetric_point a) :
  (x₀ = 6 / 5 ∧ y₀ = 8 / 5) ∧ (∀ x y : ℝ, (x ^ 2) / 8 + (y ^ 2) / 4 = 1) :=
sorry

end eccentricity_of_ellipse_equation_of_ellipse_and_point_P_l432_432947


namespace total_number_of_workers_l432_432353

-- Definitions based on the given conditions
def avg_salary_total : ℝ := 8000
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def num_technicians : ℕ := 7

-- Problem statement in Lean
theorem total_number_of_workers
    (W : ℕ) (N : ℕ)
    (h1 : W * avg_salary_total = num_technicians * avg_salary_technicians + N * avg_salary_non_technicians)
    (h2 : W = num_technicians + N) :
    W = 21 :=
sorry

end total_number_of_workers_l432_432353


namespace number_of_arrangements_l432_432751

-- Define the number of steps and people
def num_steps : ℕ := 6
def num_people : ℕ := 3

-- Define the possible arrangements based on the given conditions
def arrangements_one_person_per_step : ℕ := nat.perm num_steps num_people
def arrangements_two_people_one_step : ℕ := choose num_people 1 * nat.perm (num_steps - 1) (num_people - 1)

-- Sum the arrangements according to the principle of counting by classification
def total_arrangements : ℕ := arrangements_one_person_per_step + arrangements_two_people_one_step

-- The theorem statement
theorem number_of_arrangements : total_arrangements = 210 := by
  -- Proof skipped
  sorry

end number_of_arrangements_l432_432751


namespace perimeter_CBDF_l432_432214

-- Given conditions as definitions
def angle_ABC := 90
def CB_parallel_ED := true
def AB_eq_DF := true
def AD := 24
def AE := 25
def O_center_circle := true

theorem perimeter_CBDF : 
  ∠ABC = 90 ∧ CB_parallel_ED ∧ AB_eq_DF ∧ AD = 24 ∧ AE = 25 ∧ O_center_circle →
  let CB := AE - AD
      DF := AB
      perimeter := CB + AD + DF + (AE - AD) in
  perimeter = 42 :=
by
  sorry

end perimeter_CBDF_l432_432214


namespace convert_polar_to_rectangular_l432_432843

-- Definitions of conversion formulas and given point in polar coordinates
def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Given point in polar coordinates
def givenPolarPoint : ℝ × ℝ := (7, Real.pi / 3)

-- Correct answer in rectangular coordinates
def correctRectangularCoordinates : ℝ × ℝ := (3.5, 7 * Real.sqrt 3 / 2)

-- Proof statement
theorem convert_polar_to_rectangular :
  polarToRectangular (givenPolarPoint.1) (givenPolarPoint.2) = correctRectangularCoordinates :=
by
  sorry

end convert_polar_to_rectangular_l432_432843


namespace mike_spent_l432_432684

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l432_432684


namespace reciprocals_harmonic_progression_of_arithmetic_progression_l432_432266

open Real

theorem reciprocals_harmonic_progression_of_arithmetic_progression
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : b - a = c - b) :
  let a1 := 1 / a,
      b1 := 1 / b,
      c1 := 1 / c
  in 2 * c1 = a1 + b1 + b1 * c1 / a1 :=
by
  sorry

end reciprocals_harmonic_progression_of_arithmetic_progression_l432_432266


namespace remainder_sum_modulo_l432_432488

theorem remainder_sum_modulo :
  (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 :=
by
sorry

end remainder_sum_modulo_l432_432488


namespace no_positive_integer_k_for_rational_solutions_l432_432851

theorem no_positive_integer_k_for_rational_solutions :
  ∀ k : ℕ, k > 0 → ¬ ∃ m : ℤ, 12 * (27 - k ^ 2) = m ^ 2 := by
  sorry

end no_positive_integer_k_for_rational_solutions_l432_432851


namespace reverse_order_number_l432_432494

def is_ordered (i j : ℕ) (a : List ℕ) : Prop :=
  i < j ∧ a[i] < a[j]

def order_number (a : List ℕ) : ℕ :=
  ((List.range a.length).choose 2).count (λ⟨i, j⟩ => is_ordered i j a)

theorem reverse_order_number {a : List ℕ} (h_len : a.length = 5) (h_distinct : a.nodup)
  (h_pos : ∀ x ∈ a, 0 < x) (h_order_num : order_number a = 4) :
  order_number a.reverse = 6 :=
by
  sorry

end reverse_order_number_l432_432494


namespace problem_l432_432937

variables {α : Type*} [Field α] (a b c x : α)

noncomputable def f (x : α) : α := a * x^2 + b * x + c

theorem problem (b_eq_zero : b = 0) : (∀ x : α, f x = a * x^2 + c) ↔ ∀ x : α, f x = f (-x) :=
by 
  sorry

end problem_l432_432937


namespace range_of_mn_l432_432962

theorem range_of_mn (m n : ℝ) (h : m ≠ n) 
  (H : ∀ x y, (m * x + 2 * n * y - 4 = 0 → x^2 + y^2 - 4 * x - 2 * y - 4 = 0)) : 
  {p : ℝ | ∃ m n, p = m * n ∧ m ≠ n ∧ m + n = 2} ⊆ Iio 1 :=
by
  sorry

end range_of_mn_l432_432962


namespace greatest_possible_remainder_l432_432168

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 11 ∧ x % 11 = r ∧ r = 10 :=
by
  exists 10
  sorry

end greatest_possible_remainder_l432_432168


namespace speed_of_train_is_45_km_per_hr_l432_432022

def length_of_train : ℝ := 360
def length_of_platform : ℝ := 340
def time_to_pass_platform : ℝ := 56

def total_distance : ℝ := length_of_train + length_of_platform

def speed_m_per_s : ℝ := total_distance / time_to_pass_platform

def speed_km_per_hr : ℝ := speed_m_per_s * 3.6

theorem speed_of_train_is_45_km_per_hr : speed_km_per_hr = 45 := 
by 
  sorry

end speed_of_train_is_45_km_per_hr_l432_432022


namespace find_side_length_l432_432193

theorem find_side_length (a b c : ℝ) (A : ℝ) 
  (h1 : Real.cos A = 7 / 8) 
  (h2 : c - a = 2) 
  (h3 : b = 3) : 
  a = 2 := by
  sorry

end find_side_length_l432_432193


namespace parallel_line_through_P_perpendicular_line_through_P_l432_432961

-- Define the line l and Point P
def line_l := λ x y : ℝ, 2 * x - y - 2 = 0
def point_P := (1, 2)

-- Prove the parallel line equation
theorem parallel_line_through_P (x y : ℝ) (P : ℝ × ℝ := point_P) :
  2 * P.1 - P.2 + 0 = 0 → (x = P.1) → (y = P.2) → 2 * x - y = 0 := sorry

-- Prove the perpendicular line equation
theorem perpendicular_line_through_P (x y : ℝ) (P : ℝ × ℝ := point_P) :
  P.2 = - (1 / 2) * P.1 + 5 / 2 → (x = P.1) → (y = P.2) → x + 2 * y - 5 = 0 := sorry

end parallel_line_through_P_perpendicular_line_through_P_l432_432961


namespace find_roots_l432_432237

theorem find_roots (p q : ℝ) (z : ℂ) (h1 : z^2 - (16 + 9 * complex.I) * z + (40 + 57 * complex.I) = 0) (hpq : z = (p + 3 * complex.I) ∨ z = (q + 6 * complex.I)) : (p, q) = (9.5 : ℝ, 6.5 : ℝ) :=
sorry

end find_roots_l432_432237


namespace ratio_of_boys_under_6_feet_is_one_fifth_l432_432313

theorem ratio_of_boys_under_6_feet_is_one_fifth (total_students : ℕ) (boys_half_total : total_students / 2 = 50) (boys_under_6_feet : 10) : 
  (boys_under_6_feet : Real) / (total_students / 2 : Real) = 1 / 5 := 
by
  sorry

end ratio_of_boys_under_6_feet_is_one_fifth_l432_432313


namespace tangent_point_is_2_l432_432148

-- Defining the function
def curve (x : ℝ) : ℝ := (x^2) / 2 - 3 * Real.log (2 * x)

-- Defining the derivative of the function
def derivative (x : ℝ) : ℝ := x - 3 / x

-- Statement to prove the slope of the tangent line is 1/2 at x = 2
theorem tangent_point_is_2 (x : ℝ) (h₁ : derivative x = 1 / 2) : x = 2 :=
by
  sorry

end tangent_point_is_2_l432_432148


namespace determine_transportation_mode_l432_432437

def distance : ℝ := 60 -- in kilometers
def time : ℝ := 3 -- in hours
def speed_of_walking : ℝ := 5 -- typical speed in km/h
def speed_of_bicycle_riding : ℝ := 15 -- lower bound of bicycle speed in km/h
def speed_of_driving_a_car : ℝ := 20 -- typical minimum speed in km/h

theorem determine_transportation_mode : (distance / time) = speed_of_driving_a_car ∧ speed_of_driving_a_car ≥ speed_of_walking + speed_of_bicycle_riding - speed_of_driving_a_car := sorry

end determine_transportation_mode_l432_432437


namespace kolya_pays_90_rubles_l432_432409

theorem kolya_pays_90_rubles {x y : ℝ} 
  (h1 : x + 3 * y = 78) 
  (h2 : x + 8 * y = 108) :
  x + 5 * y = 90 :=
by sorry

end kolya_pays_90_rubles_l432_432409


namespace det_A_pow_three_l432_432182

variable {R : Type*} [CommRing R]
variable {A : Matrix R R}

theorem det_A_pow_three (h : det A = 3) : det (A ^ 3) = 27 :=
by
  sorry

end det_A_pow_three_l432_432182


namespace monotonicity_of_f_range_of_k_for_three_zeros_l432_432538

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l432_432538


namespace right_triangle_incenter_bi_length_l432_432998

theorem right_triangle_incenter_bi_length 
  (A B C I D E F : Point)
  (h_triangle : right_triangle A B C)
  (h_ab : dist A B = 12)
  (h_ac : dist A C = 13)
  (h_bc : dist B C = 15)
  (h_incenter : incenter I A B C)
  (h_touch_bc : is_in_circle_touch I D B C)
  (h_touch_ac : is_in_circle_touch I E A C)
  (h_touch_ab : is_in_circle_touch I F A B)
  : dist B I = √58 := 
sorry

end right_triangle_incenter_bi_length_l432_432998


namespace fraction_of_housing_units_with_cable_TV_l432_432218

theorem fraction_of_housing_units_with_cable_TV
  (T : ℝ) -- Total number of housing units
  (C : ℝ) -- Fraction of housing units with cable TV
  (Hvcr : T * (1 / 10)) -- Number of housing units with VCRs
  (Hc_vcr : T * (C * (1 / 4))) -- Number of housing units with both cable TV and VCRs
  (Hneither : T * 0.75) -- Number of housing units with neither cable TV nor VCRs
  (Heqn : T = T * C + T * (1 / 10) + T * 0.75) -- Given equation combining all conditions
  : C = 0.15 :=
sorry

end fraction_of_housing_units_with_cable_TV_l432_432218


namespace function_range_l432_432447

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem function_range : ∀ y : ℝ, (∃ x : ℝ, f(x) = y) ↔ (-1/2 : ℝ) ≤ y ∧ y ≤ (1/2 : ℝ) :=
by
  sorry

end function_range_l432_432447


namespace distance_travelled_l432_432007

theorem distance_travelled (speed time distance : ℕ) 
  (h1 : speed = 25)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 125 :=
by
  sorry

end distance_travelled_l432_432007


namespace sum_of_digits_h2023_eq_6070_l432_432234

def f(x : ℝ) : ℝ := 5^(5*x)
def g(x : ℝ) : ℝ := real.logb 5 (x / 5)
def h_1(x : ℝ) : ℝ := g(f(x))

def h : ℕ → ℝ → ℝ
| 1, x := h_1(x)
| (n+1), x := h_1(h n x)

theorem sum_of_digits_h2023_eq_6070 : 
  ∑ d in (nat.digits 10 (h 2023 1)).to_finset, d = 6070 := sorry

end sum_of_digits_h2023_eq_6070_l432_432234


namespace max_determinant_value_l432_432483

-- Define the matrix and the maximum value as per the problem conditions.
def determinant (θ : Real) : Real := 
  let a := 1
  let b := 1
  let c := 1
  let d := 1
  let e := 1 + sin θ
  let f := 1
  let g := 1 + cos θ
  let h := 1
  let i := 1
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem max_determinant_value : ∃ θ : Real, abs (determinant θ) = 1 / 2 :=
by
  sorry

end max_determinant_value_l432_432483


namespace minimum_stones_l432_432747

theorem minimum_stones (x y : ℕ) : 
  (2 * (x - 100) = y + 100) → 
  ∃ z, (x + z = 5 * (y - z)) → 
  x = 170 ∧ y = 40 :=
begin
  intros h1 h2,
  -- Adding the proofs steps here (would be done by user)
  sorry
end

end minimum_stones_l432_432747


namespace parabola_focus_coordinates_l432_432445

/-- Given the equation of a parabola \(y^2 = -4x\), prove that the coordinates of its focus are (-2, 0) --/
theorem parabola_focus_coordinates : focus_of_parabola (λ (x y : ℝ), y^2 = -4 * x) = (-2, 0) :=
sorry

end parabola_focus_coordinates_l432_432445


namespace solve_P_Q_l432_432099

theorem solve_P_Q :
  ∃ P Q : ℝ, (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 6) + Q / (x * (x - 5)) = (x^2 - 3*x + 15) / (x * (x + 6) * (x - 5)))) ∧
    P = 1 ∧ Q = 5/2 :=
by
  sorry

end solve_P_Q_l432_432099


namespace clara_total_cookies_l432_432833

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l432_432833


namespace arithmetic_sequence_general_term_geometric_sequence_general_term_mixed_sequence_sum_l432_432524

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d a1 : ℕ) (h1 : a 2 = 8) (h2 : S 4 = 40) 
  (h4 : ∀ n : ℕ, S n = n * (2*a1 + (n - 1)*d) / 2) (h5 : ∀ n : ℕ, a n = a1 + (n - 1)*d) :
  a 1 = 4 ∧ d = 4 ∧ (∀ n : ℕ, a n = 4 * n) :=
by 
  sorry

theorem geometric_sequence_general_term (T : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : T 1 = 3 + 2 * b 1) (h2 : ∀ n : ℕ, T n = 2 * b n - 3)
  (h3 : ∀ n > 1, T n - 2 * b n + 3 = 0) :
  b 1 = 3 ∧ ∀ n, b n = 3 * 2^(n-1) :=
by 
  sorry

theorem mixed_sequence_sum (a b c : ℕ → ℕ) (P : ℕ → ℕ)
  (ha : ∀ n, a n = 4 * n) 
  (hb : ∀ n, b n = 3 * 2^(n-1))
  (hc : ∀ n, c n = if (n % 2 = 1) then a n else b n)
  (hP_even : ∀ n, n % 2 = 0 → P n = 2^(n+1) + n^2 - 2)
  (hP_odd : ∀ n, n % 2 = 1 → P n = 2^n + n^2 + 2n - 1) :
  ∀ n, P n = if n % 2 = 0 then 2^(n+1) + n^2 - 2 else 2^n + n^2 + 2n - 1 :=
by 
  sorry

end arithmetic_sequence_general_term_geometric_sequence_general_term_mixed_sequence_sum_l432_432524


namespace evaluate_root_power_l432_432866

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l432_432866


namespace otimes_neg2_neg1_l432_432069

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432069


namespace seating_arrangements_equal_600_l432_432552

-- Definitions based on the problem conditions
def number_of_people : Nat := 4
def number_of_chairs : Nat := 8
def consecutive_empty_seats : Nat := 3

-- Theorem statement
theorem seating_arrangements_equal_600
  (h_people : number_of_people = 4)
  (h_chairs : number_of_chairs = 8)
  (h_consecutive_empty_seats : consecutive_empty_seats = 3) :
  (∃ (arrangements : Nat), arrangements = 600) :=
sorry

end seating_arrangements_equal_600_l432_432552


namespace number_of_non_congruent_triangles_l432_432122

-- Define the set of points
structure Point where
  x : ℝ
  y : ℝ

-- Define the circle condition
def on_circle (points : set Point) : Prop :=
  ∃ (O : Point) (R : ℝ), ∀ p ∈ points, (p.x - O.x)^2 + (p.y - O.y)^2 = R^2

-- Define the inner points condition
def inside_circle (G H I : Point) (O : Point) (R : ℝ) : Prop :=
  ∀ p ∈ {G, H, I}, (p.x - O.x)^2 + (p.y - O.y)^2 < R^2 ∧ ¬(p = O)

-- Define the equilateral condition
def equilateral (G H I : Point) : Prop :=
  (G.x - H.x)^2 + (G.y - H.y)^2 = (H.x - I.x)^2 + (H.y - I.y)^2 ∧
  (H.x - I.x)^2 + (H.y - I.y)^2 = (I.x - G.x)^2 + (I.y - G.y)^2

-- Main theorem statement
theorem number_of_non_congruent_triangles :
  ∀ (A B C D E F G H I : Point) (O : Point) (R : ℝ),
    on_circle {A, B, C, D, E, F} →
    inside_circle G H I O R →
    equilateral G H I →
    ∃ n : ℕ, n = 11 := 
by
  sorry

end number_of_non_congruent_triangles_l432_432122


namespace total_boys_in_camp_l432_432195

theorem total_boys_in_camp (T : ℝ) 
  (h1 : 0.20 * T = number_of_boys_from_school_A)
  (h2 : 0.30 * number_of_boys_from_school_A = number_of_boys_study_science_from_school_A)
  (h3 : number_of_boys_from_school_A - number_of_boys_study_science_from_school_A = 42) :
  T = 300 := 
sorry

end total_boys_in_camp_l432_432195


namespace cyclic_quad_l432_432589

-- Definition of the problem in terms of Lean's logic

variables {A B C D E F I M N : Type} [Inhabited A]


-- Geometry setup for triangle and bisectors
noncomputable def triangle_setup (A B C D E F I M N : Type) [Inhabited A] :=
  ∃ (triangle : A → A → A → Prop), 
    triangle A B C ∧ 
    (∀ (AD : A → A), angle_bisector A B C AD) ∧
    (∀ (BE : A → A), angle_bisector B A C BE) ∧
    (∀ (CF : A → A), angle_bisector C A B CF) ∧
    intersection AD BE I ∧ 
    intersection AD CF I ∧
    let K := midpoint A D in
    (perpendicular_bisector K AD M) ∧ 
    intersection M BE N ∧
    intersection M CF N

-- Statement of the theorem to be proved
theorem cyclic_quad (A B C D E F I M N : Type) [Inhabited A] 
  (h : triangle_setup A B C D E F I M N) : 
  cyclic_quad A I M N :=
sorry

end cyclic_quad_l432_432589


namespace max_product_distances_l432_432918

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432918


namespace triangle_remaining_sides_sum_nearest_tenth_l432_432809

noncomputable def sum_of_remaining_sides
  (A B C : Type)
  [has_measure A] [has_measure B] [has_measure C]
  (angle_A : measure A) (angle_B : measure B) (angle_C : measure C)
  (a : ℝ)
  (h1 : angle_A = 45)
  (h2 : angle_C = 60)
  (h3 : opposite_side angle_C = 9) 
  : ℝ :=
  30.1

theorem triangle_remaining_sides_sum_nearest_tenth 
  (A B C : Type)
  [has_measure A] [has_measure B] [has_measure C]
  (angle_A : measure A) (angle_B : measure B) (angle_C : measure C)
  (a : ℝ)
  (h1 : angle_A = 45)
  (h2 : angle_C = 60)
  (h3 : opposite_side angle_C = 9) :
  sum_of_remaining_sides A B C angle_A angle_B angle_C a h1 h2 h3 = 30.1 :=
sorry

end triangle_remaining_sides_sum_nearest_tenth_l432_432809


namespace max_product_of_distances_l432_432923

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432923


namespace round_4_376_to_hundredth_l432_432699

theorem round_4_376_to_hundredth : round_nearest_hundredth 4.376 = 4.38 := 
by sorry

end round_4_376_to_hundredth_l432_432699


namespace count_integer_n_values_l432_432496

theorem count_integer_n_values :
  let prime_factors_324 := (2 ^ 2) * (3 ^ 4)
  (∃ (n_set : Set ℤ), 
    (∀ n ∈ n_set, prime_factors_324 * (3^n) * (4^(-n)) ∈ ℤ) ∧
    #n_set = 5) :=
by
  let prime_factors_324 := (2 ^ 2) * (3 ^ 4)
  let n_set := {n : ℤ | n ≥ -2 ∧ n ≤ 2}
  have h : ∀ n ∈ n_set, prime_factors_324 * (3^n) * (4^(-n)) ∈ ℤ := sorry
  have cardinality_check : #n_set = 5 := sorry
  exact ⟨n_set, h, cardinality_check⟩

end count_integer_n_values_l432_432496


namespace sqrt_expression_range_l432_432192

theorem sqrt_expression_range (x : ℝ) : x + 3 ≥ 0 ∧ x ≠ 0 ↔ x ≥ -3 ∧ x ≠ 0 :=
by
  sorry

end sqrt_expression_range_l432_432192


namespace table_tennis_arrangements_l432_432813

theorem table_tennis_arrangements : 
  let veteran_players := 2
  let new_players := 3
  let total_players := 5
  let positions := 3
  let n := nat.choose(3, 2) * nat.choose(2, 1) * nat.perm(3, 3) -- For 1 veteran, 2 new
  let m := nat.choose(3, 1) * nat.choose(2, 2) * nat.perm(2, 2) -- For 2 veteran, 1 new
  n + m = 48 :=
by
  sorry

end table_tennis_arrangements_l432_432813


namespace convert_scientific_notation_l432_432715

theorem convert_scientific_notation (a : ℝ) (b : ℤ) (h : a = 6.03 ∧ b = 5) : a * 10^b = 603000 := by
  cases h with
  | intro ha hb =>
    rw [ha, hb]
    sorry

end convert_scientific_notation_l432_432715


namespace cube_vertex_sum_l432_432261

-- Define the problem conditions
variable (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ)
variable (face_sums : ℝ)
variable (sum_faces : face_sums = 2019)

-- Define the proof theorem
theorem cube_vertex_sum (h : 3 * (x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈) = face_sums) :
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ = 673 :=
begin
  rw sum_faces at h,
  linarith,
end

end cube_vertex_sum_l432_432261


namespace correct_student_answer_l432_432996

theorem correct_student_answer :
  (9 - (3^2) / 8 = 9 - (9 / 8)) ∧
  (24 - (4 * (3^2)) = 24 - 36) ∧
  ((36 - 12) / (3 / 2) = 24 * (2 / 3)) ∧
  ((-3)^2 / (1 / 3) * 3 = 9 * 3 * 3) →
  (24 * (2 / 3) = 16) :=
by
  sorry

end correct_student_answer_l432_432996


namespace rounding_test_l432_432345

def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (Real.ceil (x * 100) - 0.5) / 100

theorem rounding_test :
    ¬ (round_to_nearest_hundredth 128.557 = 128.56 ∧
       round_to_nearest_hundredth 128.564 = 128.56 ∧
       round_to_nearest_hundredth 128.5554 = 128.56 ∧
       round_to_nearest_hundredth 128.5599 = 128.56 ∧
       round_to_nearest_hundredth 128.561 = 128.56) :=
sorry

end rounding_test_l432_432345


namespace find_tuple_solution_l432_432480

theorem find_tuple_solution (x1 x2 x3 x4 x5 : ℕ) (hx1 : x1 > x2) (hx2 : x2 > x3) (hx3 : x3 > x4) (hx4 : x4 > x5) (hx5 : x5 > 0)
  (hEq : ⌊(x1 + x2 : ℕ) / 3⌋^2 + ⌊(x2 + x3 : ℕ) / 3⌋^2 + ⌊(x3 + x4 : ℕ) / 3⌋^2 + ⌊(x4 + x5 : ℕ) / 3⌋^2 = 38) :
  (x1, x2, x3, x4, x5) = (7, 4, 5, 6, 1) := 
  sorry

end find_tuple_solution_l432_432480


namespace find_a20_l432_432940

variables {a : ℕ → ℝ} -- sequence a
variable {d : ℝ} -- common difference
variable {a1 : ℝ} -- first term

-- Condition that {a_n} is an arithmetic sequence
definition is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given conditions
axiom sum_terms1 : a 1 + a 3 + a 5 = 105
axiom sum_terms2 : a 2 + a 4 + a 6 = 99

-- Target
theorem find_a20 (h_seq : is_arithmetic_sequence a a1 d) : a 20 = 1 := 
by 
  sorry

end find_a20_l432_432940


namespace set_union_example_l432_432559

theorem set_union_example (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) : M ∪ N = {1, 2, 3} := 
by
  sorry

end set_union_example_l432_432559


namespace max_value_of_x_minus_y_l432_432565

theorem max_value_of_x_minus_y  
  (x y : ℝ)
  (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) : 
  x - y ≤ sqrt(17)/2 :=
sorry

end max_value_of_x_minus_y_l432_432565


namespace conference_attendees_l432_432791

theorem conference_attendees (w m : ℕ) (h1 : w + m = 47) (h2 : 16 + (w - 1) = m) : w = 16 ∧ m = 31 :=
by
  sorry

end conference_attendees_l432_432791


namespace area_of_rhombus_l432_432710

noncomputable def diagonal_length_1 : ℕ := 30
noncomputable def diagonal_length_2 : ℕ := 14

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = diagonal_length_1) (h2 : d2 = diagonal_length_2) : 
  (d1 * d2) / 2 = 210 :=
by 
  rw [h1, h2]
  sorry

end area_of_rhombus_l432_432710


namespace line_through_midpoint_l432_432139

theorem line_through_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hmid : M = (1,1))
  (hellipse : ∀ p, p = A ∨ p = B → 4 * (p.1)^2 + 9 * (p.2)^2 = 36)
  (hintersect : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  ∃ l : affine_line ℝ, l.has_point M ∧ l.has_point A ∧ l.has_point B ∧ l.equation = "4x + 9y - 13 = 0" :=
begin
  sorry
end

end line_through_midpoint_l432_432139


namespace min_f_value_l432_432186

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x ^ 2 - 2 * cos x ^ 2) / (sin x * cos x)

theorem min_f_value : 
  ∀ x, x ∈ Icc (π / 4) (5 * π / 12) → f x = -1 := sorry

end min_f_value_l432_432186


namespace find_a_find_c_range_l432_432674

variable (f : ℝ → ℝ) (a c : ℝ)
noncomputable def f := λ x, x^3 + 3 * a * x^2 - 9 * x + 5

theorem find_a (h_extremum : ∃ x, x = 1 ∧ (deriv f x) = 0) : a = 1 :=
sorry

theorem find_c_range (h_a : a = 1) (h_bound : ∀ x ∈ Icc (-4 : ℝ) 4, f x < c^2) : 
  (-∞ < c ∧ c < -9) ∨ (9 < c ∧ c < ∞) :=
sorry

end find_a_find_c_range_l432_432674


namespace CrucianCarp_culture_days_CrucianCarp_max_profit_l432_432015

-- Part 1
theorem CrucianCarp_culture_days (x : ℕ) (h1 : x ≤ 20) : 
  10 * x * (5 - 10) + (1000 - 10 * x) * x = 8500 → x = 10 := 
begin
  sorry
end

-- Part 2
theorem CrucianCarp_max_profit (x : ℕ) (h1 : x ≤ 20) : 
  let y := -10 * x^2 + 500 * x in 
  y = 6000 :=
begin
  sorry
end

end CrucianCarp_culture_days_CrucianCarp_max_profit_l432_432015


namespace cannot_form_right_triangle_l432_432407

theorem cannot_form_right_triangle (a b c : ℕ) (h₁ : a = 4) (h₂ : b = 5) (h₃ : c = 6) : a^2 + b^2 ≠ c^2 :=
by {
  rw [h₁, h₂, h₃],
  norm_num,
  exact dec_trivial
}

end cannot_form_right_triangle_l432_432407


namespace algebraic_expression_value_l432_432942

-- Definition: x1 and x2 are the roots of the quadratic equation -2x^2 + x + 5 = 0
def is_root_of_quadratic (a b c x : ℝ) :=
  a * x ^ 2 + b * x + c = 0

-- Definition: x1 and x2 are roots of the given quadratic equation
def roots_of_given_quadratic (x1 x2 : ℝ) :=
  is_root_of_quadratic (-2) 1 5 x1 ∧ is_root_of_quadratic (-2) 1 5 x2

-- Problem statement: prove x1^2 * x2 + x1 * x2^2 = -5 / 4 given x1 and x2 are roots of the quadratic equation
theorem algebraic_expression_value (x1 x2 : ℝ) (h : roots_of_given_quadratic x1 x2) :
  x1^2 * x2 + x1 * x2^2 = -5 / 4 :=
begin
  sorry
end

end algebraic_expression_value_l432_432942


namespace part1_part2_l432_432145

-- Definitions based on the problem conditions
def Sn (a_n : ℕ → ℝ) (n : ℕ) : ℝ := n - 2 * a_n n + 20
def an (n : ℕ) : ℝ := 6 * (2/3)^(n-1) + 1

def bn (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (λ k, log (2/3) ((a_n (k+1) - 1) / 9)).sum

-- Mathematical equivalent proofs to be proven
theorem part1 (n : ℕ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, Sn a_n n = n - 2 * a_n n + 20) :
  a_n = an := by
  sorry

theorem part2 (n : ℕ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, Sn a_n n = n - 2 * a_n n + 20) :
  T_n = 2 * n / (n + 1) := by
  let b_n := bn a_n
  let T_n := (List.range n).map (λ k, 1 / b_n (k+1)).sum
  sorry

end part1_part2_l432_432145


namespace max_elements_M_l432_432645

theorem max_elements_M (M : Set ℕ) (hM : ∀ a b c ∈ M, |a + b - c| > 10) : 
  ∃ M' ⊆ { n : ℕ | 1 ≤ n ∧ n ≤ 2021 }, ∀ a b c ∈ M', |a + b - c| > 10 ∧ 
  (∀ N ⊆ { n : ℕ | 1 ≤ n ∧ n ≤ 2021 }, (∀ a b c ∈ N, |a + b - c| > 10) → card N ≤ 1006) → card M' = 1006 :=
sorry

end max_elements_M_l432_432645


namespace find_a_of_tangent_line_at_origin_l432_432525

theorem find_a_of_tangent_line_at_origin :
  ∃ a : ℝ, (∀ (x : ℝ), (a * (Real.exp x - 1) - x) = 0) ∧ ((λ x : ℝ, a * (Real.exp x) - 1) 0 - 1) = 1 → a = 2 :=
by
  sorry

end find_a_of_tangent_line_at_origin_l432_432525


namespace quadratic_term_free_polynomial_l432_432990

theorem quadratic_term_free_polynomial (m : ℤ) (h : 36 + 12 * m = 0) : m^3 = -27 := by
  -- Proof goes here
  sorry

end quadratic_term_free_polynomial_l432_432990


namespace sequence_formula_l432_432585

theorem sequence_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 6)
  (h4 : a 4 = 10)
  (h5 : ∀ n > 0, a (n + 1) - a n = n + 1) :
  ∀ n, a n = n * (n + 1) / 2 :=
by 
  sorry

end sequence_formula_l432_432585


namespace integer_solution_for_x_l432_432491

theorem integer_solution_for_x (x : ℤ) : 
  (∃ y z : ℤ, x = 7 * y + 3 ∧ x = 5 * z + 2) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by
  sorry

end integer_solution_for_x_l432_432491


namespace moduli_product_l432_432466

theorem moduli_product (z1 z2 : ℂ) (h1 : z1 = 4 - 3 * complex.I) (h2 : z2 = 4 + 3 * complex.I) : complex.abs z1 * complex.abs z2 = 25 := 
by
  rw [h1, h2]
  -- simplify abs (4 - 3i) * abs (4 + 3i)
  have : |4 - 3*complex.I| * |4 + 3*complex.I| = complex.abs ((4 - 3*complex.I) * (4 + 3*complex.I)) := complex.abs_mul (4 - 3*complex.I) (4 + 3*complex.I)
  rw [this]
  -- (4 - 3i) * (4 + 3i) = 25
  have : (4 - 3*complex.I) * (4 + 3*complex.I) = 25 := by 
    rw [←complex.mul_conj, complex.norm_sq_eq_conj_mul_self]
    simp [complex.norm_sq]
  rw [this]
  -- the modulus of 25 is 25
  rw [complex.abs_assoc, complex.abs_of_real, complex.abs_eq_abs_of_nonneg]
  norm_num
  sorry

end moduli_product_l432_432466


namespace problem_1_problem_2_l432_432127

noncomputable def circle_C : set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 + 2) ^ 2 = 9 }
def point_P : ℝ × ℝ := (2, 0)

theorem problem_1 :
  ∃ Q : set (ℝ × ℝ), (∀ p : ℝ × ℝ, p ∈ Q ↔ (p.1 - 2) ^ 2 + p.2 ^ 2 = 4) :=
sorry

theorem problem_2 :
  ¬ ∃ a : ℝ, a < 0 ∧ (∀ A B : ℝ × ℝ, A ∈ circle_C ∧ B ∈ circle_C ∧
  line_contains (λ p : ℝ × ℝ, a * p.1 - p.2 + 1 = 0) A ∧
  line_contains (λ p : ℝ × ℝ, a * p.1 - p.2 + 1 = 0) B ∧
  perpendicular_bisector (line_contains (line_from_points point_P (3, -2))) (line_segment A B)) :=
sorry

end problem_1_problem_2_l432_432127


namespace gcd_three_numbers_4557_1953_5115_l432_432726

theorem gcd_three_numbers_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 := 
by 
  sorry

end gcd_three_numbers_4557_1953_5115_l432_432726


namespace minimum_value_of_function_l432_432154

theorem minimum_value_of_function :
  ∀ (x : ℝ) (φ : ℝ),
  (|φ| < π / 2) ∧
  (∀ x, f (x + π / 12) = f x) ∧
  (∀ x, f x = f (-x)) →
  (∀ x ∈ Icc (-π / 2) 0, f x ≥ -sqrt 3) ∧
  (∃ x ∈ Icc (-π / 2) 0, f x = -sqrt 3) :=
by
  let f := λ x, cos (2 * x - φ) - real.sqrt 3 * sin (2 * x - φ)
  -- skipped proof
  sorry

end minimum_value_of_function_l432_432154


namespace find_length_QF_l432_432934

-- Definitions
def parabola_focus : (ℝ × ℝ) := (2, 0)
def directrix (x : ℝ) := x = -2

-- Given Conditions
variables (P Q : ℝ × ℝ)
def on_directrix (P : ℝ × ℝ) := P.1 = -2
def on_parabola (Q : ℝ × ℝ) := Q.2^2 = 8 * Q.1

-- The distance condition
def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def fp_condition (F P : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  dist F P = 4 * dist F Q

-- The goal to prove
theorem find_length_QF (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  on_directrix P →
  on_parabola Q →
  fp_condition parabola_focus P Q →
  dist Q parabola_focus = 3 :=
sorry

end find_length_QF_l432_432934


namespace product_of_roots_of_quadratics_l432_432254

noncomputable def product_of_roots : ℝ :=
  let r1 := 2021 / 2020
  let r2 := 2020 / 2019
  let r3 := 2019
  r1 * r2 * r3

theorem product_of_roots_of_quadratics (b : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, 2020 * x1 * x1 + b * x1 + 2021 = 0 ∧ 2020 * x2 * x2 + b * x2 + 2021 = 0) 
  (h2 : ∃ y1 y2 : ℝ, 2019 * y1 * y1 + b * y1 + 2020 = 0 ∧ 2019 * y2 * y2 + b * y2 + 2020 = 0) 
  (h3 : ∃ z1 z2 : ℝ, z1 * z1 + b * z1 + 2019 = 0 ∧ z1 * z1 + b * z2 + 2019 = 0) :
  product_of_roots = 2021 :=
by
  sorry

end product_of_roots_of_quadratics_l432_432254


namespace distance_A_B_distance_M_N_distance_A_D_l432_432805

-- Define vertices and points on the cube
structure Vertex where
  x : ℝ
  y : ℝ
  z : ℝ

def cube_edge_length : ℝ := 1

-- Vertices
def A : Vertex := ⟨0, 0, 0⟩
def B : Vertex := ⟨1, 0, 0⟩
def D : Vertex := ⟨1, 1, 1⟩
-- Points M and N (assuming as arbitrary points on the cube surface with appropriate coordinates)
def M : Vertex := ⟨0, 0.5, 0⟩
def N : Vertex := ⟨0.5, 1, 0⟩

-- Distance function between two vertices
def distance (v1 v2 : Vertex) : ℝ :=
  real.sqrt ((v1.x - v2.x)^2 + (v1.y - v2.y)^2 + (v1.z - v2.z)^2)

-- Proof statements
theorem distance_A_B : distance A B = 1 :=
by
  sorry

theorem distance_M_N : distance M N = 2 :=
by
  sorry

theorem distance_A_D : distance A D = real.sqrt 5 :=
by
  sorry

end distance_A_B_distance_M_N_distance_A_D_l432_432805


namespace find_f_f_3_l432_432532

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem find_f_f_3 : piecewise_function (piecewise_function 3) = 13 / 9 :=
  by
    sorry

end find_f_f_3_l432_432532


namespace ratio_of_boys_to_total_students_on_playground_l432_432744

theorem ratio_of_boys_to_total_students_on_playground (total_students : ℕ) (fraction_stayed : ℚ) (girls_playground : ℕ) :
  total_students = 20 →
  fraction_stayed = 1/4 →
  girls_playground = 10 →
  let students_playground := total_students * (3 / 4) in
  let boys_playground := students_playground - girls_playground in
  boys_playground / students_playground = 1 / 3 :=
by
  intros h1 h2 h3
  let students_playground := total_students * (3 / 4)
  let boys_playground := students_playground - girls_playground
  sorry

end ratio_of_boys_to_total_students_on_playground_l432_432744


namespace single_digit_pairs_l432_432448

theorem single_digit_pairs
  (a b : ℕ) : 1 < a → a < 10 → 1 < b → b < 10 → 
  (∃ (d : ℕ), d ∈ nat.digits 10 (a * b) ∧ (d = a ∨ d = b)) ↔ 
  (a, b) ∈ [ (5, 3), (5, 5), (5, 7), (5, 9), (6, 2), (6, 4), (6, 6), (6, 8) ] :=
by
  sorry

end single_digit_pairs_l432_432448


namespace num_integers_satisfying_abs_inequality_l432_432170

def numIntegersInSolutionSet : ℕ := 13

theorem num_integers_satisfying_abs_inequality :
  {x : ℤ | |x - 2| ≤ 6.5}.card = numIntegersInSolutionSet :=
by
  sorry

end num_integers_satisfying_abs_inequality_l432_432170


namespace celine_library_payment_l432_432788

-- Definitions based on conditions
def price_per_day (genre : String) (publication_years : Nat) : Float :=
  if genre = "Literature" && publication_years > 5 then 0.50
  else if genre = "History" && publication_years > 5 then 0.50
  else if (genre = "Literature" || genre = "History") && publication_years <= 5 then 0.60
  else if (genre = "Science" || genre = "Mathematics") && publication_years <= 5 then 0.90
  else if (genre = "Science" || genre = "Mathematics") && publication_years > 5 then 0.75
  else 0

def cost_per_book (genre : String) (publication_years : Nat) (days_borrowed : Nat) : Float :=
  (price_per_day genre publication_years) * days_borrowed

def total_cost_before_discount (costs : List Float) : Float :=
  costs.foldl (λ acc x => acc + x) 0

def discount_rate (books_borrowed : Nat) : Float :=
  if books_borrowed >= 5 then 0.15
  else if books_borrowed >= 3 then 0.10
  else 0

def total_cost_after_discount (total_cost : Float) (discount_rate : Float) : Float :=
  total_cost - (total_cost * discount_rate)

-- Given details about the borrowed books
def celine_borrowed_book_details : List (String × Nat × Nat) :=
  [("Literature", 6, 20), ("History", 10, 31), ("Science", 2, 31)]

def celine_total_cost : Float :=
  let costs := celine_borrowed_book_details.map (λ details => 
    cost_per_book details.1 details.2 details.3)
  let total_before_discount := total_cost_before_discount costs
  let discount := discount_rate celine_borrowed_book_details.length
  total_cost_after_discount total_before_discount discount

theorem celine_library_payment : celine_total_cost = 48.06 := by 
  sorry

end celine_library_payment_l432_432788


namespace max_arithmetic_sequence_terms_l432_432382

theorem max_arithmetic_sequence_terms
  (n : ℕ)
  (a1 : ℝ)
  (d : ℝ) 
  (sum_sq_term_cond : (a1 + (n - 1) * d / 2)^2 + (n - 1) * (a1 + d * (n - 1) / 2) ≤ 100)
  (common_diff : d = 4)
  : n ≤ 8 := 
sorry

end max_arithmetic_sequence_terms_l432_432382


namespace smallest_fraction_l432_432250

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) (eqn : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
sorry

end smallest_fraction_l432_432250


namespace angle_T_in_pentagon_l432_432205

theorem angle_T_in_pentagon (P Q R S T : ℝ) 
  (h1 : P = R) (h2 : P = T) (h3 : Q + S = 180) 
  (h4 : P + Q + R + S + T = 540) : T = 120 :=
by
  sorry

end angle_T_in_pentagon_l432_432205


namespace knights_count_l432_432615

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432615


namespace largest_coeff_term_expansion_l432_432310

   open Function

   theorem largest_coeff_term_expansion (x : ℝ) :
     let ex := (x - 1/x)^6 in
     let coeffs := [1, -6, 15, -20, 15, -6, 1] in
     (coeffs 2 * x^4 - coeffs 4 * x^2 * x^2 * (1 / x^2)^2) 
     ≥ coeffs 0 * x^6 - coeffs 6 * (1 / x)^6 :=
   by
     sorry
   
end largest_coeff_term_expansion_l432_432310


namespace exists_n_with_common_divisor_l432_432481

theorem exists_n_with_common_divisor :
  ∃ (n : ℕ), ∀ (k : ℕ), (k ≤ 20) → Nat.gcd (n + k) 30030 > 1 :=
by
  sorry

end exists_n_with_common_divisor_l432_432481


namespace max_product_distances_l432_432917

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432917


namespace arithmetic_sequence_15th_term_eq_53_l432_432337

theorem arithmetic_sequence_15th_term_eq_53 (a1 : ℤ) (d : ℤ) (n : ℕ) (a_15 : ℤ) 
    (h1 : a1 = -3)
    (h2 : d = 4)
    (h3 : n = 15)
    (h4 : a_15 = a1 + (n - 1) * d) : 
    a_15 = 53 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end arithmetic_sequence_15th_term_eq_53_l432_432337


namespace total_cost_of_two_rackets_l432_432256

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l432_432256


namespace inclination_range_l432_432120

-- Definitions of the problem conditions
open Real

noncomputable def curve (x : ℝ) : ℝ := 4 / (exp x + 1)

-- Statement of the mathematically equivalent proof problem
theorem inclination_range :
  ∀ (x : ℝ) (α : ℝ),
  α = atan (-4 / (exp x + 2 + exp (-x))) →
  (3 * π / 4 ≤ α ∧ α < π) :=
sorry

end inclination_range_l432_432120


namespace assertion_a_assertion_b_l432_432229

variables {n : ℕ} (n1 n2 ... ns : ℤ)

def condition (k : ℤ) : Prop :=
  ((list.prod (list.map (λ i, i + k) [n1, n2, ..., ns])) % (list.prod [n1, n2, ..., ns]) = 0)

theorem assertion_a (h : ∀ (k : ℤ), condition [n1, n2, ..., ns] k) : ∃ i, |[n1, n2, ..., ns].nth i| = 1 :=
sorry -- Proof omitted

section positive_condition
  variables (hp : ∀ i, [n1, n2, ..., ns].nth i > 0)

theorem assertion_b (h : ∀ (k : ℤ), condition [n1, n2, ..., ns] k) : [n1, n2, ..., ns] = [1, 2, ..., list.length [n1, n2, ..., ns]] :=
sorry -- Proof omitted
end positive_condition

end assertion_a_assertion_b_l432_432229


namespace parallel_projection_triangle_properties_l432_432566

theorem parallel_projection_triangle_properties (T : Type) [plane_geometry T] (Δ₁ Δ₂ : triangle T) 
  (h₁ : parallel_projection Δ₁ Δ₂) :
  (parallel_projection_altitude Δ₁ Δ₂ → false) ∧ 
  (parallel_projection_median Δ₁ Δ₂) ∧ 
  (parallel_projection_angle_bisector Δ₁ Δ₂ → false) ∧ 
  (parallel_projection_midline Δ₁ Δ₂) 
  ↔ true := 
begin
  sorry
end

end parallel_projection_triangle_properties_l432_432566


namespace at_least_one_not_less_than_two_l432_432239

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := sorry

end at_least_one_not_less_than_two_l432_432239


namespace total_cookies_sold_l432_432831

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l432_432831


namespace range_of_m_l432_432181

theorem range_of_m (m : ℝ) : 
  (∃ x, -2 < x ∧ x < 3 ∧ x^2 + m * x - 2 * m^2 < 0) → m ∈ set.Ici 3 :=
by
  sorry

end range_of_m_l432_432181


namespace starting_odd_number_l432_432308

theorem starting_odd_number (x : ℕ) (n : ℕ) (m : ℕ) 
  (h1 : (∑ i in range (n + 1), 2 * i + 1) = n^2)
  (h2 : (∑ i in (range (n - m)), 2 * i + 1) = m^2) 
  (h3 : 39 = 2 * n - 1) 
  (h4 : n^2 - m^2 = 384) 
  (h5 : nat.succ (2 * m - 1) = x) : 
  x = 9 := 
by
  sorry

end starting_odd_number_l432_432308


namespace cos_double_angle_identity_l432_432146

theorem cos_double_angle_identity (α : ℝ) (y₀ : ℝ) 
  (h1 : (1/2)^2 + y₀^2 = 1) : cos (2 * α) = -1/2 := 
by 
  sorry

end cos_double_angle_identity_l432_432146


namespace school_bought_50_cartons_of_markers_l432_432013

theorem school_bought_50_cartons_of_markers
  (n_puzzles : ℕ := 200)  -- the remaining amount after buying pencils
  (cost_per_carton_marker : ℕ := 4)  -- the cost per carton of markers
  :
  (n_puzzles / cost_per_carton_marker = 50) := -- the theorem to prove
by
  -- Provide skeleton proof strategy here
  sorry  -- details of the proof

end school_bought_50_cartons_of_markers_l432_432013


namespace exists_large_natural_with_high_digit_sum_l432_432697

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem exists_large_natural_with_high_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10 ^ 100 :=
by sorry

end exists_large_natural_with_high_digit_sum_l432_432697


namespace tan_half_sum_l432_432151

variable (a : ℝ) (h_a_gt_1 : 1 < a)
variable (α β : ℝ)
variable (h_α : -real.pi / 2 < α ∧ α < real.pi / 2)
variable (h_β : -real.pi / 2 < β ∧ β < real.pi / 2)
variable (h_roots : polynomial.has_roots (polynomial.C (3 * a + 1) + polynomial.X * polynomial.C (4 * a) + polynomial.X^2) (real.tan α) (real.tan β))

theorem tan_half_sum (h : 1 < a) (h_αβ: (real.tan α + real.tan β) = -4 * a) (h_prod: (real.tan α * real.tan β) = 3 * a + 1) :
  real.tan ((α + β) / 2) = -2 := by
  sorry

end tan_half_sum_l432_432151


namespace find_n_l432_432361

theorem find_n (n : ℕ) (k : ℕ) (a : Fin k → ℚ)
  (h_pos : 0 < n)
  (h_cond1 : k ≥ 2)
  (h_cond2 : (∑ i, a i) = n)
  (h_cond3 : (∏ i, a i) = n)
  (h_pos_rats : ∀ i, 0 < a i) :
  n = 4 ∨ (k ≥ 3 ∧ n > 4 ∧ n ≠ 5) :=
  sorry

end find_n_l432_432361


namespace maxValue_of_MF1_MF2_l432_432903

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432903


namespace a_2009_minus_a_2001_l432_432323

-- Definition of conditions
def digits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ ∀ a, l1.count a = l2.count a

def a_k (k : Nat) : Nat :=
  let perms := digits.permutations
  let sorted_perms := perms.qsort (λ l1 l2 => nat_of_digits l1 < nat_of_digits l2)
  nat_of_digits (sorted_perms.get (k - 1))

def nat_of_digits (l : List Nat) : Nat :=
  l.foldl (λ n d => 10 * n + d) 0

-- Statement of the problem
theorem a_2009_minus_a_2001 : a_k 2009 - a_k 2001 = 180 :=
by
  sorry

end a_2009_minus_a_2001_l432_432323


namespace range_of_m_l432_432115

theorem range_of_m (P Q : ℝ × ℝ) (m : ℝ) (hPQ : P = (-1, 1) ∧ Q = (2, 2)) :
  (∃ (x y : ℝ), (x, y) ∈ set.range (λ t, (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))) ∧ x + m * y + m = 0) →
  -3 < m ∧ m < -2 / 3 := 
  sorry

end range_of_m_l432_432115


namespace find_m_l432_432985

theorem find_m (m : ℝ) :
  let θ := 20 * Real.pi / 180 in
  Real.tan θ + m * Real.sin θ = Real.sqrt 3 → m = 4 :=
by
  intro h,
  sorry

end find_m_l432_432985


namespace marbles_in_jar_l432_432201

theorem marbles_in_jar (g y p : ℕ) (h1 : y + p = 7) (h2 : g + p = 10) (h3 : g + y = 5) :
  g + y + p = 11 :=
sorry

end marbles_in_jar_l432_432201


namespace zero_in_interval_l432_432312

def f (x : ℝ) : ℝ := 2^x + 3*x

theorem zero_in_interval : ∃ x ∈ Ioo (-1 : ℝ) 0, f x = 0 :=
by
  have f_neg_1 := (2 : ℝ)^(-1) + 3*(-1)
  have f_0 := (2 : ℝ)^(0) + 3*(0)
  have h1 : f (-1) < 0 := by
    norm_num at f_neg_1
  have h2 : f 0 > 0 := by
    norm_num at f_0
  exact sorry

end zero_in_interval_l432_432312


namespace max_product_of_distances_l432_432921

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432921


namespace min_ceiling_height_l432_432373

def length : ℝ := 90
def width : ℝ := 60
def diagonal : ℝ := real.sqrt (length^2 + width^2)
def height : ℝ := (1 / 4) * diagonal

theorem min_ceiling_height (h : ℝ) : h = 27.1 → (∃ (r : ℝ), r = h ∧ r ≥ height ∧ (∃ (n : ℝ), n = 0.1 * ⌈r / 0.1⌉₊)) :=
by
  refine ⟨_, _, _, _⟩;
  sorry

end min_ceiling_height_l432_432373


namespace proof_problem_l432_432511

variables {k1 k2 a x y : ℝ}
variables (l1 : ℝ → ℝ := λ x, k1 * x + 3)
variables (l2 : ℝ → ℝ := λ x, k2 * x - 3)

def condition_1 := k1 * k2 = -9/16

def intersects_at_P (P : ℝ × ℝ) : Prop :=
  P = (x, k1 * x + 3) ∧ P = (x, k2 * x - 3)

def trajectory_C := set_of (λ P : ℝ × ℝ, (P.1 ^ 2 / 16) + (P.2 ^ 2 / 9) = 1)

def point_Q := (a, 0)

def PQ_distance_minimized (P : ℝ × ℝ) : Prop :=
  ∀ P, intersects_at_P P → min_dist PQ P = min_dist PQ (x, 0)

def range_of_a := a ≥ 7/4 ∨ a ≤ -7/4

def ratio_of_areas (a : ℝ) (n : ℝ) : ℝ :=
  16 / 25

theorem proof_problem :
  condition_1 →
  (∀ P, intersects_at_P P → P ∈ trajectory_C) ∧
  (PQ_distance_minimized (x, y) ↔ range_of_a) ∧
  (∀ n, 0 < a ∧ a < 4 → ∃ M N E, ratio_of_areas a n = 16 / 25) :=
sorry

end proof_problem_l432_432511


namespace sequence_sum_l432_432503

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def a : ℕ → ℝ
| 1     := 1 / 2
| (n+3) := f (a n)

theorem sequence_sum (h1 : ∀ n, a n > 0) (h2 : a 20 = a 18) : a 2016 + a 2017 = sqrt 2 - 1 / 2 :=
by
  sorry

end sequence_sum_l432_432503


namespace range_of_a_same_solution_set_l432_432735

-- Define the inequality (x-2)(x-5) ≤ 0
def ineq1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the first inequality in the system (x-2)(x-5) ≤ 0
def ineq_system_1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the second inequality in the system x(x-a) ≥ 0
def ineq_system_2 (x a : ℝ) : Prop :=
  x * (x - a) ≥ 0

-- The final proof statement
theorem range_of_a_same_solution_set (a : ℝ) :
  (∀ x : ℝ, ineq_system_1 x ↔ ineq1 x) →
  (∀ x : ℝ, ineq_system_2 x a → ineq1 x) →
  a ≤ 2 :=
sorry

end range_of_a_same_solution_set_l432_432735


namespace minimum_ceiling_height_of_soccer_field_l432_432377

noncomputable def diagonal_length (length width : ℝ) : ℝ := 
  (length^2 + width^2).sqrt

-- Defining the minimum height function
noncomputable def min_height (length width : ℝ) : ℝ :=
  (diagonal_length length width) / 4

-- Function to round up to the nearest multiple of 0.1 meters
noncomputable def round_up_to_nearest_tenth (h : ℝ) : ℝ := 
  ((h * 10).ceil) / 10

theorem minimum_ceiling_height_of_soccer_field : 
  round_up_to_nearest_tenth (min_height 90 60) = 27.1 :=
begin
  -- Proof goes here (omitted)
  sorry,
end

end minimum_ceiling_height_of_soccer_field_l432_432377


namespace hyperbola_equation_final_hyperbola_equation_l432_432143

-- Definitions from conditions a)
def parabola_focus : (ℝ × ℝ) := (5, 0)

def is_asymptote (a b : ℝ) : Prop := 
  ∃ m : ℝ, 
    (b * parabola_focus.1 - m * (b^2 + a^2)^0.5 = 4) ∧
    parabola_focus.2 / b = m / (a^2 + b^2)^0.5

-- Prove the correct answer in part b)
theorem hyperbola_equation (a b : ℝ) (h₀ : a > b) (h₁: b > 0) 
  (hf: parabola_focus = (5, 0)) 
  (has : is_asymptote a b) : 
    (a = 3) ∧ (b = 4) ∧ (a^2 = 9) ∧ (b^2 = 16) := 
by
  -- Assume initial conditions and provide reasoning steps to arrive at (a = 3) and (b = 4)
  sorry

-- Final theorem to confirm the equation of the hyperbola
theorem final_hyperbola_equation : 
  (∃ a b : ℝ, (a = 3) ∧ (b = 4) ∧ ((a > b) ∧ (b > 0)) ∧ (is_asymptote a b)) → 
  (∃ (a b : ℝ), (a = 3) ∧ (b = 4) ∧ (a^2 = 9) ∧ (b^2 = 16) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)) := 
by
  intro h
  cases h with a h
  cases h with b h
  use [a, b]
  exact h

end hyperbola_equation_final_hyperbola_equation_l432_432143


namespace alpha_range_l432_432500

open Real

theorem alpha_range (α : ℝ) (h₀ : 0 ≤ α ∧ α ≤ π)
  (h₁ : ∀ x : ℝ, 8 * x^2 - 8 * sin α * x + cos (2 * α) ≥ 0) :
  (0 ≤ α ∧ α ≤ π / 6) ∨ (5 * π / 6 ≤ α ∧ α ≤ π) :=
by {
  sorry
}

end alpha_range_l432_432500


namespace students_in_second_class_l432_432288

noncomputable def number_of_students_in_second_class (avg1 avg2 avg_comb total1 total2 total_comb : ℝ) : ℝ :=
  let x := (total_comb - total1) / avg2
  x

theorem students_in_second_class :
  number_of_students_in_second_class 50 65 59.23076923076923
    (25 * 50) (25 + number_of_students_in_second_class 50 65 59.23076923076923 (25 * 50) 65 ((25 + number_of_students_in_second_class 50 65 59.23076923076923 (25 * 50) 65 59.23076923076923) * 59.23076923076923)) 65 ((25 + number_of_students_in_second_class 50 65 59.23076923076923 (25 * 50) 65 59.23076923076923) * 59.23076923076923) = (25 + number_of_students_in_second_class 50 65 59.23076923076923 (25 * 50) 65 ((25 + number_of_students_in_second_class 50 65 59.23076923076923 (25 * 50) 65 59.23076923076923) * 59.23076923076923)) :=
sorry

end students_in_second_class_l432_432288


namespace triangle_problem_l432_432219

-- Definitions for angles and side lengths in triangle
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Convert the required angle and side conditions to Lean definitions
def angle_A := 60 * Real.pi / 180
def length_b := 1
def area := Real.sqrt 3

-- Using the existing theorem and facts to set up the proof problem
theorem triangle_problem
  (angle_A : A = 60 * Real.pi / 180)
  (length_b : b = 1)
  (area : 1 / 2 * b * c * Real.sin angle_A = Real.sqrt 3) :
  ∃ (a b c : ℝ), (a = Real.sqrt 13) ∧ (c = 4) ∧ (a + b + c) / (Real.sin angle_A + Real.sin B + Real.sin C) = 2 * (Real.sqrt 13) / (2 * (Real.sqrt 3) / 3) := sorry

end triangle_problem_l432_432219


namespace eval_sqrt_pow_l432_432869

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l432_432869


namespace divisors_not_divisible_by_3_l432_432173

theorem divisors_not_divisible_by_3 (n : ℕ) (h : n = 2^2 * 3^2 * 5) :
  (∃(divisors : ℕ), set.count (λ d, d ∣ n ∧ ¬(d ∣ 3)) divisors = 6) :=
by sorry

end divisors_not_divisible_by_3_l432_432173


namespace cost_of_liter_kerosene_in_cents_l432_432352

variables 
  (cost_per_pound_rice : ℝ)
  (cost_per_dozen_eggs : ℝ)
  (cost_per_half_liter_kerosene : ℝ)
  (cost_per_egg : ℝ)
  (cost_per_liter_kerosene : ℝ)

variables
  (one_dollar_in_cents : ℝ := 100)
  (cost_pound_rice_condition : cost_per_pound_rice = 0.33)
  (dozen_eggs_equals_pound_rice_condition : cost_per_dozen_eggs = cost_per_pound_rice)
  (halfliter_kerosene_equals_4eggs_condition : cost_per_half_liter_kerosene = 4 * (cost_per_dozen_eggs / 12))
  (liter_kerosene_equals_twice_half_liter_condition : cost_per_liter_kerosene = 2 * cost_per_half_liter_kerosene)

theorem cost_of_liter_kerosene_in_cents :
  (cost_per_liter_kerosene * one_dollar_in_cents) = 22 :=
by
  unfold cost_per_dozen_eggs cost_per_half_liter_kerosene cost_per_liter_kerosene
  rw [dozen_eggs_equals_pound_rice_condition, cost_pound_rice_condition]
  have cost_per_egg : cost_per_egg = cost_per_dozen_eggs / 12 := by
    rw dozen_eggs_equals_pound_rice_condition
    ring
  have halfliter_kerosene_cost : cost_per_half_liter_kerosene = 4 * cost_per_egg := by
    rw [halfliter_kerosene_equals_4eggs_condition, cost_per_egg]
  rw [liter_kerosene_equals_twice_half_liter_condition, halfliter_kerosene_cost]
  ring
  -- Convert to cents and check the result
  norm_num
  sorry

end cost_of_liter_kerosene_in_cents_l432_432352


namespace lambda_range_l432_432505

noncomputable def angle_range_obtuse (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  dot_product < 0

theorem lambda_range (λ : ℝ) :
  angle_range_obtuse (1, -1) (λ, 1) ↔ (λ < -1 ∨ (-1 < λ ∧ λ < 1)) :=
by
  sorry

end lambda_range_l432_432505


namespace alcohol_percentage_new_mixture_l432_432783

theorem alcohol_percentage_new_mixture :
  let initial_alcohol_percentage := 0.90
  let initial_solution_volume := 24
  let added_water_volume := 16
  let total_new_volume := initial_solution_volume + added_water_volume
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_percentage
  let new_alcohol_percentage := (initial_alcohol_amount / total_new_volume) * 100
  new_alcohol_percentage = 54 := by
    sorry

end alcohol_percentage_new_mixture_l432_432783


namespace row_convergence_row_11_equals_12_l432_432016

-- Define the sequence transformation function
def sequence_next_row (seq : List ℕ) : List ℕ :=
  seq.map (λ a, seq.count a)

-- Define stabilized predicate
def stabilized (seq : List ℕ) : Prop :=
  seq = sequence_next_row seq

-- Problem statement leaning towards the provided answers
theorem row_convergence (initial_seq : List ℕ) (h_len : initial_seq.length = 1000) :
∃ k : ℕ, k ≤ 11 ∧ stabilized (nth_row initial_seq k) :=
sorry

theorem row_11_equals_12 (initial_seq : List ℕ) (h_len : initial_seq.length = 1000) :
stabilized (nth_row initial_seq 11) :=
sorry

-- Example construction that ensures the 10th row does not equal the 11th row
def example_row : List ℕ :=
[0, 1, 2, 2, 4, 4, 4, 4, 8, 8, (List.repeat 488 256)].join

example example_stabilization :
¬ stabilized (nth_row example_row 10) :=
sorry

example example_correct_next :
stabilized (nth_row example_row 11) :=
sorry

end row_convergence_row_11_equals_12_l432_432016


namespace men_left_hostel_l432_432384

variable (x : ℕ)
variable (h1 : 250 * 36 = (250 - x) * 45)

theorem men_left_hostel : x = 50 :=
by
  sorry

end men_left_hostel_l432_432384


namespace pascals_triangle_101_rows_pascals_triangle_only_101_l432_432177

theorem pascals_triangle_101_rows (n : ℕ) :
  (∃ k, (0 ≤ k) ∧ (k ≤ n) ∧ (Nat.choose n k = 101)) → n = 101 :=
begin
  -- assume that there exists some row n where 101 appears in Pascal's Triangle
  intro h,
  cases h with k hk,
  cases hk with hk0 hk1,
  cases hk1 with hk1 hl,
  
  -- we need to show that n = 101
  have h_prime := Nat.prime_101,
  
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

theorem pascals_triangle_only_101 :
  ∀ n : ℕ, (∀ k, (0 ≤ k) ∧ (k ≤ n) → (Nat.choose n k = 101) → n = 101) :=
begin
  intros n k hkn h,
  have h_prime := Nat.prime_101,
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

end pascals_triangle_101_rows_pascals_triangle_only_101_l432_432177


namespace magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432469

noncomputable def magnitude_prod_4_minus_3i_4_plus_3i : ℝ := |complex.abs (4 - 3 * complex.I) * complex.abs (4 + 3 * complex.I)|

theorem magnitude_prod_4_minus_3i_4_plus_3i_eq_25 : magnitude_prod_4_minus_3i_4_plus_3i = 25 :=
by
  sorry

end magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432469


namespace average_of_four_l432_432560

variable {r s t u : ℝ}

theorem average_of_four (h : (5 / 2) * (r + s + t + u) = 20) : (r + s + t + u) / 4 = 2 := 
by 
  sorry

end average_of_four_l432_432560


namespace isosceles_triangle_apex_angle_l432_432290

theorem isosceles_triangle_apex_angle (base_angle : ℝ) (h_base_angle : base_angle = 42) : 
  180 - 2 * base_angle = 96 :=
by
  sorry

end isosceles_triangle_apex_angle_l432_432290


namespace number_of_women_l432_432043

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l432_432043


namespace total_cost_two_rackets_l432_432258

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l432_432258


namespace least_students_with_brown_eyes_lunchbox_no_glasses_l432_432688

noncomputable def students_with_brown_eyes_and_lunchbox_and_no_glasses (total_students : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ) :=
  ∃ (n : ℕ), n = 3 ∧
    (∀ n, n ≤ brown_eyes) ∧
    (∀ n, n ≤ lunch_box) ∧
    (∀ n, students_with_brown_eyes_and_lunchbox_and_no_glasses ≤ total_students - glasses)

theorem least_students_with_brown_eyes_lunchbox_no_glasses :
  let total_students := 40
  let brown_eyes := 18
  let lunch_box := 25
  let glasses := 16
  students_with_brown_eyes_and_lunchbox_and_no_glasses total_students brown_eyes lunch_box glasses = 3 :=
sorry -- Proof steps omitted

end least_students_with_brown_eyes_lunchbox_no_glasses_l432_432688


namespace imaginary_unit_sum_l432_432986

-- Define that i is the imaginary unit, which satisfies \(i^2 = -1\)
def is_imaginary_unit (i : ℂ) := i^2 = -1

-- The theorem to be proven: i + i^2 + i^3 + i^4 = 0 given that i is the imaginary unit
theorem imaginary_unit_sum (i : ℂ) (h : is_imaginary_unit i) : 
  i + i^2 + i^3 + i^4 = 0 := 
sorry

end imaginary_unit_sum_l432_432986


namespace females_in_coach_class_l432_432251

noncomputable def flight750 := {
  total_passengers : ℕ := 120,
  percent_female : ℝ := 0.55,
  percent_first_class : ℝ := 0.10,
  fraction_male_first_class : ℝ := 1 / 3
}

open flight750

theorem females_in_coach_class (total_passengers percent_female percent_first_class fraction_male_first_class : ℝ) :
  let total_females := (percent_female * total_passengers).toNat
      total_first_class := (percent_first_class * total_passengers).toNat
      males_first_class := (fraction_male_first_class * total_first_class).toNat
      females_first_class := total_first_class - males_first_class in
  total_females - females_first_class = 58 :=
by
  let total_females := (percent_female * total_passengers).toNat
  let total_first_class := (percent_first_class * total_passengers).toNat
  let males_first_class := (fraction_male_first_class * total_first_class).toNat
  let females_first_class := total_first_class - males_first_class
  sorry

end females_in_coach_class_l432_432251


namespace youngest_sibling_age_l432_432705

def consecutive_odd_siblings: Prop :=
  ∃ (a b c d e : ℤ), a ≡ b+2 ∧ b ≡ c+2 ∧ c ≡ d+2 ∧ d ≡ e+2 ∧ a + b + c + d + e = 325

theorem youngest_sibling_age {a b c d e : ℤ} (h : a ≡ b + 2 ∧ b ≡ c + 2 ∧ c ≡ d + 2 ∧ d ≡ e + 2 ∧ a + b + c + d + e = 325) : 
  e = 61 :=
sorry

end youngest_sibling_age_l432_432705


namespace fraction_product_l432_432420

theorem fraction_product :
  ((5/4) * (8/16) * (20/12) * (32/64) * (50/20) * (40/80) * (70/28) * (48/96) : ℚ) = 625/768 := 
by
  sorry

end fraction_product_l432_432420


namespace smallest_positive_period_of_function_l432_432096

theorem smallest_positive_period_of_function :
  (∃ T : ℝ, (∀ x : ℝ, 3 * tan (2 * x + 5 * π / 6) = 3 * tan (2 * (x + T) + 5 * π / 6)) ∧ T > 0) ∧ T = π / 2 := by
  sorry

end smallest_positive_period_of_function_l432_432096


namespace length_of_CE_l432_432994

theorem length_of_CE
  (A B C D E : Type)
  (AE : ℝ)
  (angle_AEB angle_BEC angle_CED : ℝ)
  (AE_30 : AE = 30)
  (angle_AEB_90 : angle_AEB = 90)
  (angle_BEC_90 : angle_BEC = 90)
  (angle_CED_90 : angle_CED = 90)
  (AB BE BC CE : ℝ)
  (right_triangle_ABE : AB^2 + BE^2 = AE^2) : 
  CE = 15 * sqrt 2 :=
by
  sorry

end length_of_CE_l432_432994


namespace minimum_product_of_a_sequence_l432_432233

open Real

theorem minimum_product_of_a_sequence :
  ∀ (a : Fin 2002 → ℝ), (∀ i, 0 < a i) →
  (∑ i, 1 / (2 + a i)) = 1 / 2 →
  (∏ i, a i) ≥ (2002 : ℝ)^2002 :=
by
  intros a pos_a sum_a
  sorry

end minimum_product_of_a_sequence_l432_432233


namespace rounding_proof_l432_432272

-- Definitions based on conditions
def num : ℝ := 3967149.8654321
def rounded_num : ℕ := 3967150

-- Theorem statement
theorem rounding_proof : Int.round num = rounded_num := sorry

end rounding_proof_l432_432272


namespace geometric_progression_l432_432894

theorem geometric_progression (x y z : ℝ) (h_geom : y^2 = x * z) (h_prod : x * y * z = 64) (h_mean : (x + y + z) / 3 = 14 / 3) :
  {x, y, z} = {2, 4, 8} ∨ {x, y, z} = {8, 4, 2} :=
by
  sorry

end geometric_progression_l432_432894


namespace parabola_problem_l432_432933

theorem parabola_problem
  (F : ℝ × ℝ)
  (A : ℝ × ℝ)
  (p : ℝ)
  (h1 : 0 < p)
  (h2 : A = (2, (sqrt (4 * p) * 2)))
  (h3 : dist A F = 3) :
  ∃ (C : ℝ × ℝ → Prop), 
  (∀ x y : ℝ, C (x, y) ↔ y^2 = 4 * x) ∧
  (A = (2, 2 * sqrt(2)) ∨ A = (2, -2 * sqrt(2))) ∧
  (∀ l : ℝ × ℝ → Prop, 
  (∃ (M N : ℝ × ℝ),
   l M ∧ l N ∧ M ≠ N ∧ 
   ( ∃ k1 k2 : ℝ, (M = (M.1, k1 * M.1 + M.2)) ∧ (N = (N.1, k2 * N.1 + N.2)) ∧ k1 * k2 = 2)) →
   l (-2, 0)) :=
sorry

end parabola_problem_l432_432933


namespace AM_GM_inequality_from_Muirhead_l432_432847

theorem AM_GM_inequality_from_Muirhead (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i) :
    (∑ i, a i) / n ≥ (∏ i, a i) ^ (1 / n) := by
  sorry

end AM_GM_inequality_from_Muirhead_l432_432847


namespace triangle_final_position_after_rotation_l432_432020

-- Definitions for the initial conditions
def square_rolls_clockwise_around_octagon : Prop := 
  true -- placeholder definition, assume this defines the motion correctly

def triangle_initial_position : ℕ := 0 -- representing bottom as 0

-- Defining the proof problem
theorem triangle_final_position_after_rotation :
  square_rolls_clockwise_around_octagon →
  triangle_initial_position = 0 →
  triangle_initial_position = 0 :=
by
  intros
  sorry

end triangle_final_position_after_rotation_l432_432020


namespace fraction_livelihood_project_funds_l432_432243

theorem fraction_livelihood_project_funds
  (total_donation : ℕ)
  (community_pantry_fraction : ℚ)
  (crisis_fund_fraction : ℚ)
  (remaining_amount : ℕ)
  (contingency_amount : ℕ)
  (livelihood_fraction : ℚ)
  (community_pantry_amount community_pantry_amount : ℕ)
  (crisis_fund_amount : ℕ)
  (remaining_donation : ℕ) :
  total_donation = 240 →
  community_pantry_fraction = 1/3 →
  crisis_fund_fraction = 1/2 →
  remaining_donation = total_donation - community_pantry_amount - crisis_fund_amount →
  community_pantry_amount = total_donation * community_pantry_fraction →
  crisis_fund_amount = total_donation * crisis_fund_fraction →
  remaining_donation = 40 →
  contingency_amount = 30 →
  remaining_donation - contingency_amount = 10 →
  livelihood_fraction = 1/4 :=
begin
  sorry
end

end fraction_livelihood_project_funds_l432_432243


namespace range_of_m_l432_432188

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 - x - (m + 1) = 0 ∧ x ∈ Icc (-1 : ℝ) 1) → m ∈ Icc (-5/4 : ℝ) 1 :=
by
  sorry

end range_of_m_l432_432188


namespace intercepts_congruence_l432_432359

theorem intercepts_congruence (m : ℕ) (h : m = 29) (x0 y0 : ℕ) (hx : 0 ≤ x0 ∧ x0 < m) (hy : 0 ≤ y0 ∧ y0 < m) 
  (h1 : 5 * x0 % m = (2 * 0 + 3) % m)  (h2 : (5 * 0) % m = (2 * y0 + 3) % m) : 
  x0 + y0 = 31 := by
  sorry

end intercepts_congruence_l432_432359


namespace knights_count_l432_432630

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432630


namespace count_non_negatives_l432_432814

def neg_nums_count : ℤ := 3

theorem count_non_negatives :
  let a := -(-4)
  let b := abs (-1)
  let c := -abs 0
  let d := (-2)^3
  (if a ≥ 0 then 1 else 0) + (if b ≥ 0 then 1 else 0) + (if c ≥ 0 then 1 else 0) + (if d ≥ 0 then 1 else 0) = neg_nums_count :=
by {
  have ha : a = 4 := by norm_num,
  have hb : b = 1 := by norm_num,
  have hc : c = 0 := by norm_num,
  have hd : d = -8 := by norm_num,
  simp [ha, hb, hc, hd],
  norm_num
}

end count_non_negatives_l432_432814


namespace slant_heights_of_cones_l432_432971

-- Define the initial conditions
variables (r r1 x y : Real)

-- Define the surface area condition
def surface_area_condition : Prop :=
  r * Real.sqrt (r ^ 2 + x ^ 2) + r ^ 2 = r1 * Real.sqrt (r1 ^ 2 + y ^ 2) + r1 ^ 2

-- Define the volume condition
def volume_condition : Prop :=
  r ^ 2 * Real.sqrt (x ^ 2 - r ^ 2) = r1 ^ 2 * Real.sqrt (y ^ 2 - r1 ^ 2)

-- Statement of the proof problem: Prove that the slant heights x and y are given by
theorem slant_heights_of_cones
  (h1 : surface_area_condition r r1 x y)
  (h2 : volume_condition r r1 x y) :
  x = (r ^ 2 + 2 * r1 ^ 2) / r ∧ y = (r1 ^ 2 + 2 * r ^ 2) / r1 := 
  sorry

end slant_heights_of_cones_l432_432971


namespace gcd_equation_solution_l432_432880

theorem gcd_equation_solution (x y : ℕ) (h : Nat.gcd x y + x * y / Nat.gcd x y = x + y) : y ∣ x ∨ x ∣ y :=
 by
 sorry

end gcd_equation_solution_l432_432880


namespace constant_term_expansion_l432_432215

theorem constant_term_expansion :
  let f := (x^2 + (4 / x^2) - 4) in
  let g := (x + 3) in
  let expanded := (f^3) * g in
  (∃ c : ℤ, c = -480 ∧ is_constant_term (expanded)) :=
by
  sorry

end constant_term_expansion_l432_432215


namespace sin_pi_div_six_plus_alpha_tan_beta_l432_432516

-- Define the conditions α, β, cos(α), and tan(α + β)
variables (α β : ℝ) (h1 : α ∈ Ioo 0 (π / 2)) (h2 : cos α = 3 / 5) (h3 : tan (α + β) = 3)

-- 1. Prove the value of sin(π/6 + α)
theorem sin_pi_div_six_plus_alpha :
  sin (π / 6 + α) = (3 + 4 * real.sqrt 3) / 10 :=
sorry

-- 2. Prove the value of tan β given tan(α + β) = 3
theorem tan_beta :
  tan β = 5 / 7 :=
sorry

end sin_pi_div_six_plus_alpha_tan_beta_l432_432516


namespace max_product_distance_l432_432911

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432911


namespace bamboo_trunk_cutting_l432_432356

theorem bamboo_trunk_cutting (n : ℕ) (h_n : n = 40) :
  let count_ways := (list.range (n - 1)).bind (λ q, list.range (q - 1)).count
       (λ p, p + q > n - q) in
  count_ways = 171 :=
by sorry

end bamboo_trunk_cutting_l432_432356


namespace intersection_sums_l432_432446

noncomputable def polynomial := λ x : ℝ, x^3 - 2 * x^2 - x + 2

theorem intersection_sums : 
  let curve1 := polynomial
  let curve2 := λ x y : ℝ, 2 * x + 3 * y = 3
  (∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
    curve1 x1 = y1 ∧ 
    curve1 x2 = y2 ∧ 
    curve1 x3 = y3 ∧ 
    curve2 x1 y1 ∧ 
    curve2 x2 y2 ∧ 
    curve2 x3 y3 ∧ 
    x1 + x2 + x3 = 2 ∧ 
    y1 + y2 + y3 = -1) :=
sorry

end intersection_sums_l432_432446


namespace factorial_chain_divisibility_l432_432440

noncomputable def factorial_chain : ℕ → ℕ 
| 0     := n
| 1     := n.factorial
| (k+1) := (factorial_chain k).factorial

theorem factorial_chain_divisibility (n k : ℕ) (h : k ≥ 2) :
  (factorial_chain n k) ∣ (n.factorial * (n-1).factorial * (n.factorial - 1).factorial *
                            (factorial_chain n 2 - 1).factorial * 
                            ... *
                            (factorial_chain n (k-2) - 1).factorial) := 
  sorry

end factorial_chain_divisibility_l432_432440


namespace sum_of_sines_eq_zero_sum_of_cosines_l432_432695

noncomputable def alpha (n : ℕ) : ℝ := (2 * Real.pi) / n

theorem sum_of_sines_eq_zero (n r : ℕ) (h₁ : 0 < n) (h₂ : 0 < r) :
  (Finset.range n).sum (λ k, Real.sin ((r:ℝ) * alpha n * (k + 1))) = 0
:= sorry

theorem sum_of_cosines (n r : ℕ) (h₁ : 0 < n) (h₂ : 0 < r) :
  (Finset.range n).sum (λ k, Real.cos ((r:ℝ) * alpha n * (k + 1))) = 
  if n ∣ r then n else 0
:= sorry

end sum_of_sines_eq_zero_sum_of_cosines_l432_432695


namespace squareFreeCount_l432_432172

-- Define the conditions
def isSquareFree (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

-- Define the range
def validRange (n : ℕ) : Prop :=
  1 < n ∧ n < 200

-- Define the set of integers in the given range
def integersInRange : Finset ℕ :=
  (Finset.range 200).filter (λ n, 1 < n)

-- Define the set of square-free integers in the given range
def squareFreeIntegers : Finset ℕ :=
  integersInRange.filter isSquareFree

-- The main statement
theorem squareFreeCount : (squareFreeIntegers.card = 162) :=
by
  sorry

end squareFreeCount_l432_432172


namespace Jill_arrives_30_minutes_before_Jack_l432_432607

theorem Jill_arrives_30_minutes_before_Jack
  (d : ℝ) (v_J : ℝ) (v_K : ℝ)
  (h₀ : d = 3) (h₁ : v_J = 12) (h₂ : v_K = 4) :
  (d / v_K - d / v_J) * 60 = 30 :=
by
  sorry

end Jill_arrives_30_minutes_before_Jack_l432_432607


namespace intersection_distance_l432_432713

theorem intersection_distance (y : ℝ) (x : ℝ) (h : y = 2016) (hx : y = tan (3 * x)) : 
  ∃ d, d = (π / 3) := 
sorry

end intersection_distance_l432_432713


namespace solution_set_inequality_l432_432943

variable (f : ℝ → ℝ)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

def condition_x_f_prime (f f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x^2 * f' x > 2 * x * f (-x)

-- Main theorem to prove the solution set of inequality
theorem solution_set_inequality (f' : ℝ → ℝ) :
  is_odd_function f →
  derivative f f' →
  condition_x_f_prime f f' →
  ∀ x : ℝ, x^2 * f x < (3 * x - 1)^2 * f (1 - 3 * x) → x < (1 / 4) := 
  by
    intros h_odd h_deriv h_cond x h_ineq
    sorry

end solution_set_inequality_l432_432943


namespace am_gm_inequality_l432_432646

theorem am_gm_inequality (a b : ℕ → ℝ) (n : ℕ) (h_nonneg_a : ∀ i, 0 ≤ a i) (h_nonneg_b : ∀ i, 0 ≤ b i) :
  (∏ i in finset.range n, a i) ^ (1 / n : ℝ) + (∏ i in finset.range n, b i) ^ (1 / n : ℝ) ≤
  (∏ i in finset.range n, (a i + b i)) ^ (1 / n : ℝ) := 
by 
  sorry

end am_gm_inequality_l432_432646


namespace find_set_of_points_l432_432997

noncomputable theory

open_locale classical

variables {C : Type} [metric_space C] [normed_space ℝ C]
variables {L : set C} {T M S : C} (P : set C)

-- Define the properties of the circle, line, and tangent relationship
def is_tangent (C : C) (L : set C) := 
  ∃ (T : C), T ∈ L ∧ dist C T = radius C

-- Define the midpoint relationship
def is_midpoint (M Q R : C) := 
  dist M Q = dist M R

-- Define the condition that C is the incircle of triangle PQR
def is_incircle (C : C) (P Q R : C) :=
  tangent_to_all_sides_of_triangle P Q R C

-- Define the conditions of the problem.
def satisfies_conditions (C : C) (L : set C) (M P : C) :=
  ∃ Q R ∈ L, 
  is_midpoint M Q R ∧ is_incircle C P Q R

-- Define the final proposition in Lean.
theorem find_set_of_points 
  (C : Type) [metric_space C] [normed_space ℝ C]
  (L : set C) (M T S : C) (ray_NS : set C) :
  is_tangent C L →
  M ∈ L →
  ray_NS = {x : C | ∃ k > 0, x = S + k * (S - T)} \ {S} →
  ∀ P : C, satisfies_conditions C L M P ↔ P ∈ ray_NS :=
begin
  sorry
end

end find_set_of_points_l432_432997


namespace angle_AFE_135_l432_432576

-- Define the problem conditions using Lean's structure
structure Square :=
(ABCD : Type)
(A B C D : ABCD)
(E : ABCD → ABCD)
(F : ABCD → ABCD)
(DC : D → C)
(AD : A → D)
(angle_CDE_100 : ∀ {α β γ : ℝ}, α + β + γ = 180)
(angle_DF_perp_DE : (∀ D F E : ABCD, D ≠ F ∧ F ≠ E ∧ D ≠ E → ⟨F, E⟩))

-- Define the equivalence of the angles
theorem angle_AFE_135 {sq : Square} 
(E1 : sq.E = sq.DC)
(F1 : sq.F = sq.AD)
(F2 : sq.F = sq.DC)
(angle_CDE_100 : ∠ sq.DC sq.E = 100)
(angle_DF_perp_DE : ∠ sq.D sq.F = 90)
(angle_DF_perp_DE_E : ∠ sq.D sq.F ∠ sq.F sq.E = 90)
(triangle_DEF_isosceles_right: ∠ sq.D sq.F ∠ sq.F sq.E = 45)
(external_angle_AFE : ∠ sq.F sq.E + ∠ sq.D sq.F = 135) :
∠ sq.A sq.F sq.E = 135 :=
sorry

end angle_AFE_135_l432_432576


namespace probability_at_least_seven_heads_or_tails_l432_432455

open Nat

-- Define the probability of getting at least seven heads or tails in eight coin flips
theorem probability_at_least_seven_heads_or_tails :
  let total_outcomes := 2^8
  let favorable_outcomes := (choose 8 7) + (choose 8 7) + 1 + 1
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 9 / 128 := by
  sorry

end probability_at_least_seven_heads_or_tails_l432_432455


namespace angle_between_b_c_l432_432506

open Real EuclideanSpace

-- Definitions translated from conditions
def vec_a : EuclideanSpace ℝ (Fin 3) := sorry
def vec_b : EuclideanSpace ℝ (Fin 3) := sorry
def vec_c : EuclideanSpace ℝ (Fin 3) := ![-2, 1, 2]

axiom h1 : 3 • vec_a - 2 • vec_b = ![-2, 0, 4]
axiom h2 : vec_a ⋅ vec_c = 2
axiom h3 : ∥vec_b∥ = 4

-- Prove the angle theta
theorem angle_between_b_c : 
  let θ := Real.arccos ((vec_b ⋅ vec_c) / (∥vec_b∥ * ∥vec_c∥)) in
  θ = π - Real.arccos (1 / 4) :=
sorry

end angle_between_b_c_l432_432506


namespace problem_solution_l432_432105

noncomputable def solution_set : Set ℝ :=
  { x : ℝ | x ∈ (Set.Ioo 0 (5 - Real.sqrt 10)) ∨ x ∈ (Set.Ioi (5 + Real.sqrt 10)) }

theorem problem_solution (x : ℝ) : (x^3 - 10*x^2 + 15*x > 0) ↔ x ∈ solution_set :=
by
  sorry

end problem_solution_l432_432105


namespace trajectory_of_M_is_line_segment_l432_432650

-- Define points and distance function
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (p1 p2 : Point) : ℝ :=
Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Define fixed points F1 and F2
noncomputable def F1 : Point := ⟨0, 0⟩
noncomputable def F2 : Point := ⟨6, 0⟩

-- Define moving point M
def M : Point := sorry -- M is a moving point

-- Conditions
axiom dist_F1_F2 : dist F1 F2 = 6
axiom dist_MF1_plus_dist_MF2 (M : Point) : dist M F1 + dist M F2 = 6

-- Theorem to prove 
theorem trajectory_of_M_is_line_segment :
  ∀ M : Point, dist M F1 + dist M F2 = 6 → 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
begin
  sorry -- Proof goes here
end

end trajectory_of_M_is_line_segment_l432_432650


namespace fold_triangle_cover_square_l432_432123

noncomputable theory

def triangle_area (a b c : ℝ) : Prop := 
  ∃ x y z, x = a^2 ∧ y = b^2 ∧ z = c^2 ∧ (a * b) + (b * c) + (c * a) - x^2 - y^2 - z^2 = 2

theorem fold_triangle_cover_square :
  ∃ (a b c : ℝ) (A1 : a > 0) (A2 : b > 0) (A3 : c > 0)
     (T : triangle_area a b c)
     (Sa : (1 / 4)^2 = 1 / 16)
     (Pa : 1 / 2 = (a * b) / 2)
     (I: c ^ 2 = 1),
   True := 
by
  sorry

end fold_triangle_cover_square_l432_432123


namespace problem_statement_l432_432294

variable {f : ℝ → ℝ}
variable {a b : ℝ}
variable h1 : ∀ x, f x = 0 ↔ x = 0
variable h2 : ∀ x y, f (x * y) = f x * f y
variable h3 : ∀ x y, f (x + y) ≤ 2 * max (f x) (f y)

theorem problem_statement (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : f (a + b) ≤ f a + f b :=
by
  sorry

end problem_statement_l432_432294


namespace right_triangle_angles_l432_432031

theorem right_triangle_angles (α : ℝ) (β : ℝ) (A B C K L M : Type)
  [Triangle ABC] [EquilateralTriangle KLM] [RightTriangle ABC] :
  (∀ K L : Side ABC, K ∈ BC ∧ L ∈ AC) ∧
  (M ∈ AB ∧ KL ∥ AB) ∧ (KL = (1/3) * AB) →
  α = 30 ∨ α = 60 →
  ABC.angles = [30, 60, 90] :=
by
  sorry

end right_triangle_angles_l432_432031


namespace magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432472

noncomputable def magnitude_prod_4_minus_3i_4_plus_3i : ℝ := |complex.abs (4 - 3 * complex.I) * complex.abs (4 + 3 * complex.I)|

theorem magnitude_prod_4_minus_3i_4_plus_3i_eq_25 : magnitude_prod_4_minus_3i_4_plus_3i = 25 :=
by
  sorry

end magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432472


namespace product_of_possible_b_values_l432_432852

theorem product_of_possible_b_values :
  let b_values := {b : ℝ | ∃ d : ℝ, (3 * b - 3)^2 + (b + 3)^2 = d^2 ∧ d = 3 * Real.sqrt 13 } in
  b_values.product = -2.1049 :=
by sorry

end product_of_possible_b_values_l432_432852


namespace sum_of_AB_l432_432526

def set_prod {α : Type} (A B : set α) : set α := 
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : set ℕ := {1, 2}
def B : set ℕ := {0, 2}

def sum_set (S : set ℕ) : ℕ :=
  set.fold (λa b, a + b) 0 S

theorem sum_of_AB : sum_set (set_prod A B) = 6 := by
  sorry

end sum_of_AB_l432_432526


namespace largest_side_of_enclosure_l432_432682

-- Definitions for the conditions
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem largest_side_of_enclosure (l w : ℝ) (h_fencing : perimeter l w = 240) (h_area : area l w = 12 * 240) : l = 86.83 ∨ w = 86.83 :=
by {
  sorry
}

end largest_side_of_enclosure_l432_432682


namespace ellipse_slope_product_constant_l432_432508

noncomputable def ellipse_constant_slope_product (a b : ℝ) (P M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (N.1 = -M.1 ∧ N.2 = -M.2) ∧
  (∃ k_PM k_PN : ℝ, k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_PN = (P.2 - N.2) / (P.1 - N.1)) ∧
  ((P.2 - M.2) / (P.1 - M.1) * (P.2 - N.2) / (P.1 - N.1) = -b^2 / a^2)

theorem ellipse_slope_product_constant (a b : ℝ) (P M N : ℝ × ℝ) :
  ellipse_constant_slope_product a b P M N := 
sorry

end ellipse_slope_product_constant_l432_432508


namespace floor_of_2_99_l432_432048

theorem floor_of_2_99 : Int.floor 2.99 = 2 := 
by
  sorry

end floor_of_2_99_l432_432048


namespace all_pairs_are_relatively_prime_l432_432393

def arithmetic_progression (n : ℕ) : ℕ :=
  1 + (n - 1) * Nat.factorial 100

theorem all_pairs_are_relatively_prime :
  ∀ (n m : ℕ), 1 ≤ n → n < m → m ≤ 100 →
  Nat.coprime (arithmetic_progression n) (arithmetic_progression m) :=
by
  sorry

end all_pairs_are_relatively_prime_l432_432393


namespace no_net_gain_after_ten_requests_l432_432811

theorem no_net_gain_after_ten_requests (x : ℕ) (h : 0 ≤ x ∧ x < 1000) : 
  let x10 := (x + 1023000) / 1024 in
  x10 ≤ x :=
by {
  sorry
}

end no_net_gain_after_ten_requests_l432_432811


namespace insurance_coverage_correct_l432_432610

variable (pills_per_day : ℕ) (cost_per_pill dollars_paid_per_month: ℝ)

-- Conditions definitions
def total_pills_per_month := 30 * pills_per_day
def total_cost_without_insurance := total_pills_per_month * cost_per_pill
def insurance_coverage := total_cost_without_insurance - dollars_paid_per_month
def insurance_coverage_percentage := (insurance_coverage / total_cost_without_insurance) * 100

-- Proof statement
theorem insurance_coverage_correct (h_pills_per_day : pills_per_day = 2)
    (h_cost_per_pill : cost_per_pill = 1.5)
    (h_dollars_paid_per_month : dollars_paid_per_month = 54)
    : insurance_coverage_percentage pills_per_day cost_per_pill dollars_paid_per_month = 40 := by
  sorry

end insurance_coverage_correct_l432_432610


namespace tan_sec_cos_l432_432558

theorem tan_sec_cos (A : ℝ) (h : Real.tan A + Real.sec A = 2) : Real.cos A = 4 / 5 := 
  sorry

end tan_sec_cos_l432_432558


namespace solution_to_problem_l432_432557

theorem solution_to_problem (x : ℝ) (h : 12^(Real.log 7 / Real.log 12) = 10 * x + 3) : x = 2 / 5 :=
by sorry

end solution_to_problem_l432_432557


namespace angle_between_vectors_is_30_degrees_l432_432569

-- Define the vectors
def vector_a : ℝ × ℝ := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
def vector_b : ℝ × ℝ := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))

-- Statement of the problem: to prove that the angle between (vector_a + vector_b) and vector_a is 30 degrees
theorem angle_between_vectors_is_30_degrees :
  ∀ a b : ℝ × ℝ, a = vector_a → b = vector_b →
  let sum := (a.1 + b.1, a.2 + b.2) in
  let angle (u v : ℝ × ℝ) :=
    Real.acos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))) * 180 / Real.pi in
  angle sum vector_a = 30 := 
by
  intros a b ha hb
  sorry

end angle_between_vectors_is_30_degrees_l432_432569


namespace sum_even_minus_odd_l432_432418

theorem sum_even_minus_odd :
  (∑ n in Finset.range 100, 2 * (n + 1)) - (∑ n in Finset.range 100, 2 * (n + 1) - 1) = 100 :=
by sorry

end sum_even_minus_odd_l432_432418


namespace geometric_number_difference_l432_432050

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end geometric_number_difference_l432_432050


namespace plane_intersection_properties_l432_432546

theorem plane_intersection_properties 
  (planes : Type) [linear_space planes]
  (α β γ : planes) 
  (a b c : line planes)
  (h_perp : β ⟂ γ)
  (h_int : ∃ l : line planes, l ∈ α ∩ γ ∧ ¬ (α ⟂ γ))
  (h_a_α : a ∈ α)
  (h_b_β : b ∈ β)
  (h_c_γ : c ∈ γ) :
  (∃ a, a ∈ α ∧ a ∦ γ) ∧ (∃ c, c ∈ γ ∧ c ⟂ β) :=
sorry

end plane_intersection_properties_l432_432546


namespace tan_addition_l432_432561

theorem tan_addition (a b : ℝ) 
  (h₁ : Real.tan a + Real.tan b = 15) 
  (h₂ : Real.cot a + Real.cot b = 20) : 
  Real.tan (a + b) = 60 := 
by
  sorry

end tan_addition_l432_432561


namespace event_days_and_medals_l432_432806

-- Defining the conditions as Lean structures and properties
variables (n m : ℕ) (u : ℕ → ℕ)

-- Condition 1: Event lasted more than 1 day
axiom n_gt_1 : n > 1

-- Condition 2: Total medals distributed is m
axiom total_medals : ∃ m, m > 0

-- Condition 3: Medals distribution pattern
axiom medals_distribution_pattern : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    u k = m - ∑ i in finset.range k, (i + 1 + (u i - i - 1) / 7)

-- Condition 4: Remaining medals after each day
axiom remaining_medals : 
  ∀ k : ℕ, 1 ≤ k ∧ k < n → 
    u (k + 1) = u k - (k + 1 + (u k - k - 1) / 7)

-- Condition 5: Initial number of medals
axiom initial_medals : u 1 = m

-- Condition 6: No medals left after the n-th day
axiom no_medals_left : u n = 0

-- Proof statement
theorem event_days_and_medals : n = 6 ∧ m = 36 :=
sorry

end event_days_and_medals_l432_432806


namespace swept_area_square_rotation_l432_432326

-- Define the square with side length 1
def square_side_length : ℝ := 1

-- Define the rotation angle: 90 degrees in radians
def rotation_angle_rad : ℝ := Real.pi / 2

-- Define the expected swept area by side AB during the rotation
def expected_swept_area : ℝ := Real.pi / 4

-- Theorem stating that the swept area by side AB is as expected
theorem swept_area_square_rotation :
  swept_area square_side_length rotation_angle_rad = expected_swept_area :=
sorry

end swept_area_square_rotation_l432_432326


namespace probability_prime_number_l432_432454

def numbers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12]

def is_prime (n : ℕ) : bool :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def prime_number_count : ℕ := numbers.countp (λ n => is_prime n)

def total_number_count : ℕ := numbers.length

theorem probability_prime_number :
  (prime_number_count : ℚ) / total_number_count = 3 / 8 :=
by
  sorry

end probability_prime_number_l432_432454


namespace max_product_distances_l432_432915

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432915


namespace modulus_of_z_conjugate_of_z_l432_432147

-- Define the given complex number z
def z : ℂ := (1 + 2 * Complex.i) * (1 - Complex.i)

-- Prove that the modulus of z is sqrt(10)
theorem modulus_of_z :
  Complex.abs z = Real.sqrt 10 :=
sorry

-- Prove that the complex conjugate of z is 3 - i
theorem conjugate_of_z :
  Complex.conj z = 3 - Complex.i :=
sorry

end modulus_of_z_conjugate_of_z_l432_432147


namespace log_intersection_distance_l432_432795

theorem log_intersection_distance (k a b : ℤ) (hk : k = a + Int.sqrt b) 
  (h_intersect_k : |Real.log 3 k - Real.log 3 (k + 6)| = 1) : a + b = 3 :=
by
  sorry

end log_intersection_distance_l432_432795


namespace sum_of_reciprocals_of_arithmetic_sequence_l432_432124

theorem sum_of_reciprocals_of_arithmetic_sequence :
  ∀ (a : ℕ → ℕ) (n : ℕ),
  (a 5 = 5) →
  (∑ i in (range 5).map (λ i, a i) = 15) →
  (∑ i in finset.range 2016 | (1 : ℚ) / ((a i) * (a (i + 1))) = 2016 / 2017) :=
by
  intros a n ha5 hs5
  sorry

end sum_of_reciprocals_of_arithmetic_sequence_l432_432124


namespace B_midpoint_UV_l432_432038

variables {A B C D E F U V : Point} {circumcircle : Circle}

-- Conditions
axiom AB_lt_AC (AB : LineSegment) (AC : LineSegment) : AB.length < AC.length
axiom reflection_D (A B D : Point) (H : midpoint B A D) : overline BD = overline DA
axiom perp_bisector_CD (CD : LineSegment) (H : bisector E F of CD) : perp_bisector CD = set E.intersection F 
axiom intersect_AE_AF_BC (A E F B C U V : Point) (H1 : line A E intersects line B C at U) (H2 : line A F intersects line B C at V) : true

theorem B_midpoint_UV (A B C D E F U V : Point) (circumcircle : Circle)
  [AB_lt_AC (AB_length : Real) (AC_length : Real)]
  [reflection_D A B D midpoint]
  [perp_bisector_CD CD bisector]
  [intersect_AE_AF_BC A E F B C U V]
  : midpoint B U V := sorry

end B_midpoint_UV_l432_432038


namespace base2_digit_difference_l432_432341

theorem base2_digit_difference (n m : ℕ) (h_n : n = 150) (h_m : m = 750) :
  nat.log2 m + 1 - (nat.log2 n + 1) = 1 :=
by
  rw [h_n, h_m]
  sorry

end base2_digit_difference_l432_432341


namespace negative_root_m_positive_l432_432987

noncomputable def is_negative_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 + m * x - 4 = 0

theorem negative_root_m_positive : ∀ m : ℝ, is_negative_root m → m > 0 :=
by
  intro m
  intro h
  sorry

end negative_root_m_positive_l432_432987


namespace sequence_not_exist_l432_432497

-- Defining the natural number n
def n : ℕ := sorry

-- Sequence conditions
def is_valid_sequence (s : ℕ → ℕ) (n : ℕ) :=
  ∀ r ∈ finset.Icc 1 n, ∃ i j, i < j ∧ j = i + r ∧ s i = r ∧ s j = r

-- The theorem to prove
theorem sequence_not_exist (n : ℕ) (hn : n % 4 = 2 ∨ n % 4 = 3) : 
  ¬∃ s : ℕ → ℕ, is_valid_sequence s n :=
sorry

end sequence_not_exist_l432_432497


namespace radius_of_holes_accurate_l432_432792

noncomputable def radius_of_holes 
  (a b c : ℕ) (flow_velocity_river flow_velocity_holes : ℝ) : ℝ :=
  let h := (c^2 - (b - a)^2 / 4)^0.5
  let A := 1 / 2 * (a + b) * h
  let flow_rate_river := A * flow_velocity_river
  let radius_holes := (flow_rate_river / (4 * flow_velocity_holes * π))^0.5
  radius_holes

theorem radius_of_holes_accurate 
  (a b c : ℕ) (flow_velocity_river flow_velocity_holes : ℝ) :
  a = 10 → b = 16 → c = 5 → flow_velocity_river = π → flow_velocity_holes = 16 →
  radius_of_holes a b c flow_velocity_river flow_velocity_holes = (Real.sqrt 13) / 4 :=
by {
  intros,
  unfold radius_of_holes,
  sorry
}

end radius_of_holes_accurate_l432_432792


namespace hyperbola_focus_coordinates_l432_432849

theorem hyperbola_focus_coordinates :
  (∃ c : ℝ, c = 3 ∧ (0, c) is_a_focus_of_the_hyperbola) :=
begin
  sorry
end

-- Additional definitions necessary for clarity of the statement
def is_a_focus_of_the_hyperbola (coord : ℝ × ℝ) : Prop :=
  let c := coord.snd in
  ∃ a b : ℝ, a^2 = 3 ∧ b^2 = 6 ∧ c = real.sqrt (a^2 + b^2)

end hyperbola_focus_coordinates_l432_432849


namespace trajectory_of_point_P_l432_432519

variable (x y : ℝ)

def point_A : ℝ × ℝ × ℝ := (0, 0, 4)

def |PA| (p : ℝ × ℝ × ℝ) : ℝ :=
  let (px, py, pz) := p
  let (ax, ay, az) := point_A
  Real.sqrt ((px - ax)^2 + (py - ay)^2 + (pz - az)^2)

theorem trajectory_of_point_P (P_on_xy : ∀ P : ℝ × ℝ × ℝ, P.2.1 = 0 → (|PA| P = 5)) :
  x^2 + y^2 = 9 :=
by
  sorry

end trajectory_of_point_P_l432_432519


namespace total_paintings_is_correct_l432_432262

-- Definitions for Philip's schedule and starting number of paintings
def philip_paintings_monday_and_tuesday := 3
def philip_paintings_wednesday := 2
def philip_paintings_thursday_and_friday := 5
def philip_initial_paintings := 20

-- Definitions for Amelia's schedule and starting number of paintings
def amelia_paintings_every_day := 2
def amelia_initial_paintings := 45

-- Calculation of total paintings after 5 weeks
def philip_weekly_paintings := 
  (2 * philip_paintings_monday_and_tuesday) + 
  philip_paintings_wednesday + 
  (2 * philip_paintings_thursday_and_friday)

def amelia_weekly_paintings := 
  7 * amelia_paintings_every_day

def total_paintings_after_5_weeks := 5 * philip_weekly_paintings + philip_initial_paintings + 5 * amelia_weekly_paintings + amelia_initial_paintings

-- Proof statement
theorem total_paintings_is_correct :
  total_paintings_after_5_weeks = 225 :=
  by sorry

end total_paintings_is_correct_l432_432262


namespace mike_stamps_given_l432_432319

theorem mike_stamps_given :
  ∃ (M : ℕ), 
  let H := 2 * M + 10 in
  3000 + M + H = 3061 ∧ M = 17 :=
sorry

end mike_stamps_given_l432_432319


namespace second_discount_given_l432_432018

theorem second_discount_given
  (P : ℚ)
  (h1 : 1.32 * P = P + 0.32 * P)
  (h2 : 1.188 * P = 1.32 * P - 0.10 * 1.32 * P)
  (final_price : 0.98 * P = (1 - D) * 1.188 * P)
  (h3 : D = 1 - (0.98 / 1.188)) :
  D ≈ 17.49 :=
begin
  sorry
end

end second_discount_given_l432_432018


namespace find_a_l432_432534

noncomputable def f : ℝ → ℝ
| x => if h : x > 0 then 2 * x else x + 1

theorem find_a (a : ℝ) (h : f a + f 2 = 0) : a = -5 :=
by
  have f2 : f 2 = 4 := by simp [f]
  rewrite [f2] at h
  sorry

end find_a_l432_432534


namespace find_positive_integers_l432_432478

noncomputable def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem find_positive_integers (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hab_c : is_power_of_two (a * b - c))
  (hbc_a : is_power_of_two (b * c - a))
  (hca_b : is_power_of_two (c * a - b)) :
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) ∨
  (a = 2 ∧ b = 6 ∧ c = 11) :=
sorry

end find_positive_integers_l432_432478


namespace value_of_a_l432_432988

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) → (a = 2 ∨ a = 3 ∨ a = 4) :=
by sorry

end value_of_a_l432_432988


namespace association_satisfaction_tourist_origin_expected_value_of_X_classification_of_attraction_l432_432252

section problem1
-- Question 1: Prove the association between "satisfaction" and "tourist origin"
theorem association_satisfaction_tourist_origin 
  (a b c d n : ℕ) (h1 : a = 90) (h2 : b = 10) (h3 : c = 75) (h4 : d = 25) (h5 : n = 200) :
  a * d - b * c > 0 →  -- This inequality is true for the given data and indicates the association
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)) > 6.635 := sorry
end problem1

section problem2
-- Question 2: Prove the expected value of X is 9/8
theorem expected_value_of_X (p : ℚ) (n : ℕ) (h1 : p = 3 / 8) (h2 : n = 3) :
  n * p = 9 / 8 := sorry
end problem2

section problem3
-- Question 3: Prove classification of attraction based on comprehensive satisfaction rate
def P1 : ℚ := 75 / 100
def P2 : ℚ := 90 / 100
def P0 : ℚ := 165 / 200
def M : ℚ := (P1 - (1 - P2)) / P0

theorem classification_of_attraction (M : ℚ) (h1 : M ≈ 0.788) :
  M < 0.80 := sorry
end problem3

end association_satisfaction_tourist_origin_expected_value_of_X_classification_of_attraction_l432_432252


namespace body_network_construction_ways_l432_432178

-- Definitions of the conditions in the problem
-- These are representing essential decompositions and edges selection.
structure CubeNetwork :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (symmetries : Equiv.Perm Point)
  (length : ℕ)

-- Define an instance of a Cube and its transformation
noncomputable def transformedCubeNetwork : CubeNetwork :=
  -- specifics of vertices, edges and symmetries are abstracted for simplicity
  ⟨{- vertices -}, {- edges -}, {- symmetries -}, 7⟩

-- Prove that there are 18 distinct ways to construct the network
theorem body_network_construction_ways :
  ∃ (configurations : Finset CubeNetwork), configurations.card = 18 :=
by
  sorry

end body_network_construction_ways_l432_432178


namespace total_students_in_class_l432_432571

theorem total_students_in_class (F L B N : ℕ) (hF : F = 26) (hL : L = 20) (hB : B = 17) (hN : N = 9) : F + L - B + N = 38 :=
by
  rw [hF, hL, hB, hN]
  norm_num

end total_students_in_class_l432_432571


namespace p_is_neither_sufficient_nor_necessary_l432_432928

variable {x : ℝ}

def condition_p := (1 / x ≤ 1)
def condition_q := (1 / 3) ^ x ≥ (1 / 2) ^ x

theorem p_is_neither_sufficient_nor_necessary (h : condition_p) : ¬(condition_p ↔ condition_q) :=
by
  sorry

end p_is_neither_sufficient_nor_necessary_l432_432928


namespace magician_performances_l432_432387

noncomputable def number_of_performances : ℕ :=
  let P := 100 in
  let no_reappearance := P / 10 in
  let double_reappearance := P / 5 in
  let single_reappearance := P - no_reappearance - double_reappearance in
  P

theorem magician_performances (P : ℕ) (H1 : P / 10 = P / 10) 
      (H2 : P / 5 * 2 + single_reappearance * 1 = 110) : 
      P = 100 :=
by
  rw [P / 10] at H1
  rw [P / 5 * 2 + single_reappearance = 110] at H2
  sorry

end magician_performances_l432_432387


namespace odd_divisors_10_factorial_l432_432978

theorem odd_divisors_10_factorial : 
  let n := (2^8) * (3^4) * (5^2) * 7 in
  let odd_factors := (3^4) * (5^2) * 7 in
  let count_divisors := (λ k, ∃ a b c, k = (3 ^ a) * (5 ^ b) * (7 ^ c) ∧ 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1) in
  ∑ k in finset.range (n + 1), if count_divisors k then 1 else 0 = 30 := 
by
  sorry

end odd_divisors_10_factorial_l432_432978


namespace blue_parrots_count_l432_432977

theorem blue_parrots_count (P : ℕ) (red green blue : ℕ) (h₁ : red = P / 2) (h₂ : green = P / 4) (h₃ : blue = P - red - green) (h₄ :  P + 30 = 150) : blue = 38 :=
by {
-- We will write the proof here
sorry
}

end blue_parrots_count_l432_432977


namespace A_share_of_profit_l432_432365

-- Define necessary financial terms and operations
def initial_investment_A := 3000
def initial_investment_B := 4000

def withdrawal_A := 1000
def advanced_B := 1000

def duration_initial := 8
def duration_remaining := 4

def total_profit := 630

-- Calculate the equivalent investment duration for A and B
def investment_months_A_first := initial_investment_A * duration_initial
def investment_months_A_remaining := (initial_investment_A - withdrawal_A) * duration_remaining
def investment_months_A := investment_months_A_first + investment_months_A_remaining

def investment_months_B_first := initial_investment_B * duration_initial
def investment_months_B_remaining := (initial_investment_B + advanced_B) * duration_remaining
def investment_months_B := investment_months_B_first + investment_months_B_remaining

-- Prove that A's share of the profit is Rs. 240
theorem A_share_of_profit : 
  let ratio_A : ℚ := 4
  let ratio_B : ℚ := 6.5
  let total_ratio : ℚ := ratio_A + ratio_B
  let a_share : ℚ := (total_profit * ratio_A) / total_ratio
  a_share = 240 := 
by
  sorry

end A_share_of_profit_l432_432365


namespace xiaopang_mom_initial_money_l432_432769

theorem xiaopang_mom_initial_money
  (unit_price : ℝ)
  (wanted_kg : ℝ)
  (short_amount : ℝ)
  (bought_kg : ℝ)
  (remaining_money : ℝ)
  (unit_price_eq : unit_price = 5)
  (wanted_kg_eq : wanted_kg = 5)
  (short_amount_eq : short_amount = 3.5)
  (bought_kg_eq : bought_kg = 4)
  (remaining_money_eq : remaining_money = 1.5)
  : (bought_kg * unit_price + remaining_money = 
     wanted_kg * unit_price - short_amount) →
    (bought_kg * unit_price + remaining_money) = 21.5 :=
by
  intros h1
  rw [unit_price_eq, wanted_kg_eq, short_amount_eq, bought_kg_eq, remaining_money_eq] at h1
  have h2 : 4 * 5 + 1.5 = 21.5 := by norm_num
  exact h2

end xiaopang_mom_initial_money_l432_432769


namespace knights_count_l432_432632

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432632


namespace max_product_distance_l432_432909

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432909


namespace star_4_3_l432_432562

def star (a b : ℕ) : ℕ := a^2 + a * b - b^3

theorem star_4_3 : star 4 3 = 1 := 
by
  -- sorry is used to skip the proof
  sorry

end star_4_3_l432_432562


namespace time_to_cross_bridge_is_179_88_seconds_l432_432399

-- Define the given conditions
def carriages := 24
def carriage_length := 60 -- in meters
def engine_length := 60 -- in meters
def train_speed_kmph := 60 -- in kmph
def bridge_length_km := 1.5 -- in km

-- Convert lengths and speed to consistent units (meters and seconds)
def total_train_length := carriages * carriage_length + engine_length -- in meters
def bridge_length := bridge_length_km * 1000 -- in meters
def total_distance := total_train_length + bridge_length -- in meters

def train_speed_mps := train_speed_kmph * (1000 / 3600) -- in meters per second

-- Calculate the time to cross the bridge
def time_to_cross_bridge := total_distance / train_speed_mps

-- Prove that the time to cross the bridge is approximately 179.88 seconds
theorem time_to_cross_bridge_is_179_88_seconds : abs (time_to_cross_bridge - 179.88) < 0.01 :=
by sorry

end time_to_cross_bridge_is_179_88_seconds_l432_432399


namespace taller_tree_height_l432_432712

variable (T S : ℝ)

theorem taller_tree_height (h1 : T - S = 20)
  (h2 : T - 10 = 3 * (S - 10)) : T = 40 :=
sorry

end taller_tree_height_l432_432712


namespace skilled_new_worker_installation_capacity_number_of_recruitment_plans_l432_432286

-- Define the constants and conditions
variables {x y n m : ℕ}
def skilled_worker_installation_capacity := 2 * x + y = 10
def new_worker_installation_capacity := 3 * x + 2 * y = 16
def total_units := 360
def days := 15

-- Define the constraints
def number_of_units_installed_per_day := 2 * x + y = 10 ∧ 3 * x + 2 * y = 16
def recruitment_constraint1 := 1 < n
def recruitment_constraint2 := n < 8
def recruitment_constraint3 := m > 0
def equipment_completion_constraint := 15 * (4 * n + 2 * m) = 360

-- Prove that each skilled worker can install 4 units and each new worker can install 2 units
theorem skilled_new_worker_installation_capacity : 
  number_of_units_installed_per_day → 
  x = 4 ∧ y = 2 :=
sorry

-- Prove that there are 4 different recruitment plans for the new workers
theorem number_of_recruitment_plans :
  recruitment_constraint1 →
  recruitment_constraint2 →
  (∃ z : ℕ, Σ (n : ℕ), m = 12 - 2 * n ∧ equipment_completion_constraint ∧ recruitment_constraint3) :=
sorry

end skilled_new_worker_installation_capacity_number_of_recruitment_plans_l432_432286


namespace cost_of_gasoline_l432_432642

-- Define all the conditions as variables and constants
variable (total_distance : ℕ)
variable (first_option_cost : ℕ)
variable (second_option_cost : ℕ)
variable (distance_per_liter : ℕ)
variable (savings : ℕ)
variable (gas_cost_per_liter : ℝ)

-- Given conditions for the problem
def conditions := 
  total_distance = 300 ∧
  first_option_cost = 50 ∧
  second_option_cost = 90 ∧
  distance_per_liter = 15 ∧
  savings = 22

-- Define the target cost of gasoline per liter
def answer := gas_cost_per_liter = 3.40

-- The statement we need to prove in Lean 4
theorem cost_of_gasoline (h : conditions) : answer :=
sorry

end cost_of_gasoline_l432_432642


namespace a_minus_b_eq_2_l432_432966

noncomputable def a : ℤ := ((1 + x) + (1 + x)^4).expand exp x 2
noncomputable def b : ℤ := ((1 + x) + (1 + x)^4).expand exp x 3

theorem a_minus_b_eq_2 (x : ℝ): ((1+x) + (1+x)^4 = 2 + 5x + a * x^2 + b * x^3 + x^4) → a - b = 2 :=
by
  sorry

end a_minus_b_eq_2_l432_432966


namespace knights_count_l432_432613

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432613


namespace x_lt_y_l432_432228

variable {a b c d x y : ℝ}

theorem x_lt_y 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (cd)^(y/2)) : 
  x < y :=
by 
  sorry

end x_lt_y_l432_432228


namespace eval_sqrt_pow_l432_432867

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l432_432867


namespace obtuse_triangle_values_count_l432_432738

theorem obtuse_triangle_values_count : 
  ∃ (s : Finset ℕ), 
  (∀ k ∈ s, 
    (let a := 13 in let b := 17 in 
      (b < a + k ∧ k^2 + a^2 < b^2) ∨
      (k < a + b ∧ a^2 + b^2 < k^2)))
  ∧ s.card = 14 :=
by sorry

end obtuse_triangle_values_count_l432_432738


namespace fox_appropriation_l432_432671

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l432_432671


namespace necessary_condition_for_A_l432_432543

variable {x a : ℝ}

def A : Set ℝ := { x | (x - 2) / (x + 1) ≤ 0 }

theorem necessary_condition_for_A (x : ℝ) (h : x ∈ A) (ha : x ≥ a) : a ≤ -1 :=
sorry

end necessary_condition_for_A_l432_432543


namespace number_of_distinct_determinants_l432_432512

theorem number_of_distinct_determinants (numbers : Finset ℕ) (h : numbers.card = 9) : 
  ∃ N : ℕ, N = 10080 :=
begin
  use 10080,
  sorry
end

end number_of_distinct_determinants_l432_432512


namespace knights_count_l432_432638

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432638


namespace fiftieth_term_is_260_l432_432304

def positive_multiples_of_four_with_digit_two (n : ℕ) : Prop :=
  n % 4 = 0 ∧ (∃ d : ℕ, d ∈ (Nat.digits 10 n) ∧ d = 2)

def sequence_term (k : ℕ) : ℕ :=
  (Nat.find_greatest (λ n, n % 4 = 0 ∧ (∃ d : ℕ, d ∈ (Nat.digits 10 n) ∧ d = 2)) k)

theorem fiftieth_term_is_260 : sequence_term 50 = 260 := 
sorry

end fiftieth_term_is_260_l432_432304


namespace sum_series_l432_432837

def S (n : ℕ) : ℚ := 
  (finset.range n).sum (λ k, (2 * (k + 1) + 1) / ((k + 1) ^ 2 * (k + 2) ^ 2))

theorem sum_series (n : ℕ) : S n = 1 - 1 / ((n + 1) ^ 2) :=
by
  sorry

end sum_series_l432_432837


namespace distance_between_sphere_center_and_triangle_plane_l432_432392

open Real

noncomputable def sphere_radius : ℝ := 10
noncomputable def triangle_side_a : ℝ := 13
noncomputable def triangle_side_b : ℝ := 13
noncomputable def triangle_side_c : ℝ := 10
noncomputable def inradius (r : ℝ) : Prop := r = 10/3

theorem distance_between_sphere_center_and_triangle_plane
    (O : Point)
    (triangle_plane : Plane)
    (A B C : Point)
    (h1 : distance O A = sphere_radius)
    (h2 : distance O B = sphere_radius)
    (h3 : distance O C = sphere_radius)
    (h4 : inradius (10/3)) :
  distance_point_plane O triangle_plane = (20 * sqrt 2) / 3 :=
sorry

end distance_between_sphere_center_and_triangle_plane_l432_432392


namespace number_of_correct_statements_is_zero_l432_432032

theorem number_of_correct_statements_is_zero:
  (¬(∀ m: ℝ, ∃ m: ℚ)) ∧ (¬(∀ a b: ℝ, a > b ↔ a^2 > b^2)) ∧ (¬(∀ x: ℝ, x = 3 → x^2 - 2x - 3 = 0)) ∧ (¬(∀ A B: Set, A ∩ B = B → A = ∅)) →
  (0 = 0) :=
by {
  sorry.
}

end number_of_correct_statements_is_zero_l432_432032


namespace simplify_sqrt_expression_l432_432333

theorem simplify_sqrt_expression :
  ( (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175) = 13 / 5 := by
  -- conditions for simplification
  have h1 : Real.sqrt 112 = 4 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 567 = 9 * Real.sqrt 7 := sorry
  have h3 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  
  -- Use the conditions to simplify the expression
  rw [h1, h2, h3]
  -- Further simplification to achieve the result 13 / 5
  sorry

end simplify_sqrt_expression_l432_432333


namespace A_inter_B_l432_432658

-- Define the sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := { abs x | x ∈ A }

-- Statement of the theorem to be proven
theorem A_inter_B :
  A ∩ B = {0, 2} := 
by 
  sorry

end A_inter_B_l432_432658


namespace simplify_complex_division_l432_432277

theorem simplify_complex_division :
  ∀ (a b c d : ℝ), 
    (c ≠ 0 ∨ d ≠ 0) →
    (a * c - b * d = -39 ∧ a * d + b * c = 73 ∧ c^2 + d^2 = 25) →
    (7 + 15 * complex.i) / (3 - 4 * complex.i) = - (39 / 25) + (73 / 25) * complex.i :=
by
  intros a b c d hnonzero hcond
  sorry

end simplify_complex_division_l432_432277


namespace otimes_neg2_neg1_l432_432068

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432068


namespace sum_of_digits_of_N_l432_432656

def d (n : ℕ) : ℕ :=
  Nat.divisors n

def f (n : ℕ) : ℝ :=
  (d n).card / (n:ℝ)^(1/3)

noncomputable def N : ℕ :=
  2^3 * 3^2 * 5 * 7

theorem sum_of_digits_of_N : (N.digits.sum = 9) :=
begin
  sorry
end

end sum_of_digits_of_N_l432_432656


namespace angle_XYZ_in_pentagon_triangle_l432_432010

theorem angle_XYZ_in_pentagon_triangle :
  let pentagon_interior_angle := 108
  let equilateral_triangle_interior_angle := 60
  let triangle_angle_sum := 180
  let common_side (XY : ℝ) := true
  let isosceles_triangle (X Y Z : ℝ) := XY = YZ
  mangleXYZ (X Y Z : ℝ) := 6 := 
  sorry

end angle_XYZ_in_pentagon_triangle_l432_432010


namespace present_population_l432_432300

theorem present_population (P : ℝ)
  (h1 : P + 0.10 * P = 242) :
  P = 220 := 
sorry

end present_population_l432_432300


namespace bleaching_process_percentage_decrease_l432_432396

noncomputable def total_percentage_decrease (L B : ℝ) : ℝ :=
  let area1 := (0.80 * L) * (0.90 * B)
  let area2 := (0.85 * (0.80 * L)) * (0.95 * (0.90 * B))
  let area3 := (0.90 * (0.85 * (0.80 * L))) * (0.92 * (0.95 * (0.90 * B)))
  ((L * B - area3) / (L * B)) * 100

theorem bleaching_process_percentage_decrease (L B : ℝ) :
  total_percentage_decrease L B = 44.92 :=
by
  sorry

end bleaching_process_percentage_decrease_l432_432396


namespace no_real_roots_implies_no_real_roots_of_composite_l432_432731

-- Definitions for monic quadratic polynomials f and g
variables {α : Type*} [linear_ordered_field α]

-- Defining monic quadratic polynomial 
def is_monic_quadratic (p : α → α) : Prop :=
  ∃ a b c : α, a = 1 ∧ p = λ x, a * x^2 + b * x + c

-- Main theorem statement
theorem no_real_roots_implies_no_real_roots_of_composite
  (f g : α → α)
  (hf : is_monic_quadratic f)
  (hg : is_monic_quadratic g)
  (hfg : ∀ x : α, f (g x) ≠ 0)
  (hgf : ∀ x : α, g (f x) ≠ 0)
  : (∀ x : α, g (g x) ≠ 0) ∨ (∀ x : α, f (f x) ≠ 0) :=
sorry

end no_real_roots_implies_no_real_roots_of_composite_l432_432731


namespace max_product_distances_l432_432916

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432916


namespace coefficient_x2_in_expansion_l432_432708

theorem coefficient_x2_in_expansion :
  let C := Nat.choose in
  (4 : ℕ) = 4 →
  ∀ x : ℝ, (x + 2) ^ 4 = (C 4 0) * x^0 * 2^4 + (C 4 1) * x^1 * 2^3 + (C 4 2) * x^2 * 2^2 +
                  (C 4 3) * x^3 * 2^1 + (C 4 4) * x^4 * 2^0 →
  (C 4 2) * 2^2 = 24 :=
by
  intros C h x hx
  sorry

end coefficient_x2_in_expansion_l432_432708


namespace rhombus_iff_area_6_l432_432855

structure Point where
  x : ℝ
  y : ℝ

def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def is_on_curve (p : Point) (a b c : ℝ) : Prop := 
  p.y = f p.x a b c

def equal_length (p1 p2 p3 p4 : Point) : Prop := 
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 =
  (p3.x - p4.x)^2 + (p3.y - p4.y)^2 ∧
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 =
  (p2.x - p4.x)^2 + (p2.y - p4.y)^2

def area (p1 p2 p3 p4 : Point) : ℝ :=
  1/2 * (Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) * Real.sqrt ((p3.x - p4.x)^2 + (p3.y - p4.y)^2))

theorem rhombus_iff_area_6 (a b c : ℝ) (y1 y2 : ℝ)
  (h_y1_y2 : y1 ≠ y2)
  (x1 x2 x3 x4 : ℝ)
  (h1 : is_on_curve ⟨x1, y1⟩ a b c)
  (h2 : is_on_curve ⟨x2, y1⟩ a b c)
  (h3 : is_on_curve ⟨x3, y2⟩ a b c)
  (h4 : is_on_curve ⟨x4, y2⟩ a b c) :
  (equal_length ⟨x1, y1⟩ ⟨x2, y1⟩ ⟨x3, y2⟩ ⟨x4, y2⟩) ↔ (area ⟨x1, y1⟩ ⟨x2, y1⟩ ⟨x3, y2⟩ ⟨x4, y2⟩ = 6) :=
by
  sorry

end rhombus_iff_area_6_l432_432855


namespace max_product_distances_l432_432914

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432914


namespace vector_u_conditions_l432_432492

-- Define the vector and projections
def u := (7 : ℚ, 5, 3)
def v1 := (3 : ℚ, 2, 1)
def proj1 := (57/7 : ℚ, 38/7, 19/7)
def v2 := (1 : ℚ, 3, 2)
def proj2 := (29/7 : ℚ, 87/7, 58/7)

-- Prove that the projection conditions hold for vector u
theorem vector_u_conditions : (orthogonal_projection v1 u = proj1) ∧ (orthogonal_projection v2 u = proj2) := by
  sorry

end vector_u_conditions_l432_432492


namespace point_in_fourth_quadrant_l432_432209

theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : 
(x > 0) → (y < 0) → (x, y) = (2, -3) → quadrant (2, -3) = 4 :=
by
  sorry

end point_in_fourth_quadrant_l432_432209


namespace maxValue_of_MF1_MF2_l432_432904

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432904


namespace arithmetic_geometric_sequence_l432_432125

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ)
  (h₀ : ∀ n, a n = 2^(n-1))
  (h₁ : a 1 = 1)
  (h₂ : a 1 + a 2 + a 3 = 7)
  (h₃ : q > 0) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, S n = 2^n - 1) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l432_432125


namespace time_to_cross_bridge_l432_432808

-- Define the necessary constants for the problem
def length_train : ℝ := 140
def length_bridge : ℝ := 235
def speed_train_kmh : ℝ := 45

-- Define the speed train in m/s (convert from km/hr to m/s)
def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Define the total distance to be covered by the train
def total_distance : ℝ := length_train + length_bridge

-- Define the time taken to cross the bridge
def time_to_cross : ℝ := total_distance / speed_train_ms

-- Prove that the time taken to cross the bridge is 30 seconds
theorem time_to_cross_bridge : time_to_cross = 30 := by
  sorry

end time_to_cross_bridge_l432_432808


namespace vendor_has_correct_liters_of_Pepsi_l432_432793

def Maaza := 80
def Sprite := 368
def least_cans_required := 37

-- Helper function to calculate GCD
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition of the number of liters per can, which is the GCD of Maaza and Sprite
def liters_per_can := gcd Maaza Sprite

-- The number of liters of Pepsi the vendor has
def liters_of_Pepsi := 144

theorem vendor_has_correct_liters_of_Pepsi :
  let cans_for_Maaza := Maaza / liters_per_can,
  let cans_for_Sprite := Sprite / liters_per_can,
  let total_regular_cans := cans_for_Maaza + cans_for_Sprite,
  let cans_for_Pepsi := least_cans_required - total_regular_cans,
  cans_for_Pepsi * liters_per_can = liters_of_Pepsi := by
  sorry

end vendor_has_correct_liters_of_Pepsi_l432_432793


namespace a_100_eq_99_33_l432_432846

noncomputable def a : ℕ → ℝ
| 0          := 1
| (n + 1) := (99^(1/3)) * a n

theorem a_100_eq_99_33 : a 100 = 99 ^ 33 := by
  sorry

end a_100_eq_99_33_l432_432846


namespace geom_cos_sequence_l432_432878

open Real

theorem geom_cos_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) (h : cos (2*b) / cos b = cos (3*b) / cos (2*b)) : b = 180 :=
by
  sorry

end geom_cos_sequence_l432_432878


namespace max_product_of_distances_l432_432924

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432924


namespace find_polynomials_l432_432135

noncomputable def satisfies_conditions (P : ℤ → ℤ) (p : ℤ) : Prop :=
  ∃ c : ℤ,
  (∀ x : ℤ, x > 0 → P x = x + c) ∧ 0 < c ∧
  (∀ m : ℤ, m > 0 → ∃ l : ℕ, (p + l * c) % m = 0)

theorem find_polynomials (p : ℤ) (hp : nat.prime (int.to_nat p)) :
  (∀ P : ℤ → ℤ, (satisfies_conditions P p → 
  (∀ x : ℤ, x > 0 → P x = x + 1) ∨ (∀ x : ℤ, x > 0 → P x = x + p))) :=
sorry

end find_polynomials_l432_432135


namespace sum_sequence_ineq_l432_432240

noncomputable def sequence_a : ℕ → ℝ
| 0       => 1
| (n + 1) => (n + 1) / sequence_a n

theorem sum_sequence_ineq (n : ℕ) : 
  (∑ k in Finset.range (n + 1).succ, (1 / sequence_a k)) ≥ 2 * (Real.sqrt (n + 1) - 1) := 
sorry

end sum_sequence_ineq_l432_432240


namespace find_larger_number_l432_432563

theorem find_larger_number (x y : ℤ) (h1 : 5 * y = 6 * x) (h2 : y - x = 12) : y = 72 :=
sorry

end find_larger_number_l432_432563


namespace monotonicity_of_f_range_of_k_for_three_zeros_l432_432536

noncomputable def f (x k : ℝ) := x^3 - k * x + k^2

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_of_f (k : ℝ) :
  (k ≤ 0 → ∀ x y : ℝ, x ≤ y → f x k ≤ f y k) ∧ 
  (k > 0 → (∀ x : ℝ, x < -sqrt (k / 3) → f x k < f (-sqrt (k / 3)) k) ∧ 
            (∀ x : ℝ, x > sqrt (k / 3) → f x k > f (sqrt (k / 3)) k) ∧
            (f (-sqrt (k / 3)) k > f (sqrt (k / 3)) k)) :=
  sorry

-- Problem 2: Range of k for f(x) to have three zeros
theorem range_of_k_for_three_zeros (k : ℝ) : 
  (∃ a b c : ℝ, f a k = 0 ∧ f b k = 0 ∧ f c k = 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) ↔ (0 < k ∧ k < 4 / 27) :=
  sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l432_432536


namespace exists_polynomial_divisible_by_power_of_x_minus_one_l432_432694

theorem exists_polynomial_divisible_by_power_of_x_minus_one (n : ℕ) :
  ∃ P : Polynomial ℤ,
    (∀ k : ℕ, P.coeff k ∈ {0, -1, 1}) ∧
    P.degree ≤ (2^n - 1) ∧
    (Polynomial.X - 1) ^ n ∣ P :=
by
  sorry

end exists_polynomial_divisible_by_power_of_x_minus_one_l432_432694


namespace cylinder_volume_ratio_l432_432786

theorem cylinder_volume_ratio (hA : ℝ := 9) (cA : ℝ := 6) (hB : ℝ := 6) (cB : ℝ := 9) :
  let rA := cA / (2 * Real.pi)
  let V_A := Real.pi * rA^2 * hA
  let rB := cB / (2 * Real.pi)
  let V_B := Real.pi * rB^2 * hB
  V_B / V_A = (3 : ℚ) / (2 : ℚ) := 
by
  let rA := cA / (2 * Real.pi)
  let V_A := Real.pi * rA^2 * hA
  let rB := cB / (2 * Real.pi)
  let V_B := Real.pi * rB^2 * hB
  sorry

end cylinder_volume_ratio_l432_432786


namespace evaluate_function_l432_432529

def f (x : ℝ) : ℝ := 
  if 0 < x then log x / log 2 else 3 ^ x

theorem evaluate_function :
  f (f (1 / 2)) = 1 / 3 := 
sorry

end evaluate_function_l432_432529


namespace locus_of_P_l432_432975

noncomputable def point_A : Complex := ⟨0, 0⟩
noncomputable def point_B : Complex := ⟨4, -3⟩
noncomputable def circle_radius : ℝ := 3

noncomputable def divides_ratio (P A C : Complex) : Prop :=
  2 * re P = re A + re C ∧ 2 * im P = im A + im C

theorem locus_of_P (P D : Complex) (hD_circle : abs D = circle_radius)
  (hP_division : divides_ratio P point_A (point_B + D)) :
  abs (P - Complex.mk (8 / 3) (-2)) = 2 :=
sorry

end locus_of_P_l432_432975


namespace right_triangle_area_l432_432189

theorem right_triangle_area (h : 5 = 5) (median_hypotenuse : 6 = 6) :
    let hypotenuse := 2 * 6 in
    let area := 1 / 2 * hypotenuse * 5 in
    area = 30 := 
by
  sorry

end right_triangle_area_l432_432189


namespace line_semi_circle_intersection_l432_432730

theorem line_semi_circle_intersection (b : ℝ) :
  (∀ (x y : ℝ), (y = x + b → x = sqrt (1 - y^2) → x ≥ 0) → x = sqrt (1 - y^2) ∧ y = x + b → ∃! (x y : ℝ), y = x + b ∧ x = sqrt (1 - y^2) ∧ x ≥ 0) ↔ (b = -sqrt 2 ∨ b ∈ Icc (-1 : ℝ) 1) :=
by
  sorry

end line_semi_circle_intersection_l432_432730


namespace min_value_l432_432654

open Real

theorem min_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  (\frac{1}{2 * a + b} + \frac{1}{2 * b + c} + \frac{1}{2 * c + a}) ≥ 1 :=
sorry

end min_value_l432_432654


namespace angle_CIH_iff_angle_IDL_l432_432935

open EuclideanGeometry

/-- Given an acute-angled triangle \( \triangle ABC \) with orthocenter \( H \) and incenter \( I \),
where \( AC \neq BC \), \( CH \) and \( CI \) intersect the circumcircle of \( \triangle ABC \)
at points \( D \) and \( L \), respectively, prove that the condition \( \angle CIH = 90^\circ \)
is both necessary and sufficient for \( \angle IDL = 90^\circ \). -/
theorem angle_CIH_iff_angle_IDL (A B C H I D L : Point)
  (hA : acute_angled_triangle A B C)
  (hH : orthocenter H A B C)
  (hI : incenter I A B C)
  (hACneBC : A ≠ B)
  (hC_con_D : chord_intersect_circumcircle C H D)
  (hC_con_L : chord_intersect_circumcircle C I L) :
  (angle_CIH H I C = 90) ↔ (angle_IDL I D L = 90) :=
sorry

end angle_CIH_iff_angle_IDL_l432_432935


namespace clara_total_cookies_l432_432834

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l432_432834


namespace find_distinct_numbers_l432_432100

theorem find_distinct_numbers (k l : ℕ) (h : 64 / k = 4 * (64 / l)) : k = 1 ∧ l = 4 :=
by
  sorry

end find_distinct_numbers_l432_432100


namespace soccer_field_illumination_l432_432380

noncomputable def diagonal (l w : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2)

noncomputable def min_ceiling_height (l w : ℝ) : ℝ :=
  Real.ceil ((diagonal l w) / 4 * 10) / 10

theorem soccer_field_illumination :
  min_ceiling_height 90 60 = 27.1 :=
by
  sorry

end soccer_field_illumination_l432_432380


namespace range_of_f_area_of_triangle_ABC_l432_432533

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 3) + sqrt 3

-- Define the bounds of x
def x_bounds (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π / 6

-- Define the minimum and maximum values
def a := sqrt 3
def b := 2

-- Define the radius of the circumcircle
def r := 3 * sqrt 2 / 4

-- Define the triangle sides based on f(x)
def triangle_sides (f : ℝ → ℝ) : (ℝ × ℝ) := (a, b)

-- Statement to prove the range of f(x)
theorem range_of_f : ∀ x, x_bounds x → f x ∈ set.Icc (sqrt 3) 2 :=
begin
  sorry
end

-- Define the sin, cos of angles A, B, and area of triangle based on given conditions
def angle_A_sin : ℝ := a / (2 * r)
def angle_B_sin : ℝ := b / (2 * r)
def angle_C_sin : ℝ := angle_A_sin * cos (1 / 3) + angle_B_sin * cos (sqrt 3 / 3)
def triangle_area : ℝ := 1 / 2 * a * b * angle_C_sin

-- Statement to prove the area of ΔABC is √2
theorem area_of_triangle_ABC : triangle_area = sqrt 2 :=
begin
  sorry
end

end range_of_f_area_of_triangle_ABC_l432_432533


namespace basketball_weight_l432_432456

variable {b c : ℝ}

theorem basketball_weight (h1 : 8 * b = 4 * c) (h2 : 3 * c = 120) : b = 20 :=
by
  -- Proof omitted
  sorry

end basketball_weight_l432_432456


namespace sum_F_1_to_150_l432_432231

def even_digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ x => x % 2 = 0) |>.sum

def odd_digits_condition (n : ℕ) : ℕ :=
  if n % 5 = 0 then n.digits 10 |>.filter (λ x => x % 2 = 1) |>.sum else 0

def F (n : ℕ) : ℕ :=
  even_digits_sum n + odd_digits_condition n

theorem sum_F_1_to_150 : (∑ n in Finset.range 150, F (n+1)) = 2025 :=
  sorry

end sum_F_1_to_150_l432_432231


namespace polynomial_coefficient_sum_and_a1_l432_432984

theorem polynomial_coefficient_sum_and_a1 :
  (∃ a : Fin 10 → ℤ, (1 - X)^5 * (1 - 2 * X)^4 = ∑ i : Fin 10, a i * X^i) →
  let b := ∑ i : Fin 10, a i in
  b = 0 ∧ a 1 = -13 :=
begin
  intro h_exists,
  obtain ⟨a, h_poly⟩ := h_exists,
  let b := ∑ i : Fin 10, a i,
  have h_sum : b = 0, sorry, -- skips the actual proof of the sum
  have h_a1 : a 1 = -13, sorry, -- skips the actual proof of a1
  tauto,
end

end polynomial_coefficient_sum_and_a1_l432_432984


namespace value_of_Q_l432_432236

theorem value_of_Q :
  (∏ (k : ℕ) in (finRange (12 - 1).succ).map (λ i, i + 2), 1 - (1 : ℚ) / (k ^ 2)) = 13 / 144 := 
sorry

end value_of_Q_l432_432236


namespace henry_initial_games_l432_432590

def initial_games (H N L : ℕ) : Prop :=
  N = 7 ∧ L = 7 ∧ H = 3 * N ∧ (H - 10 = 4 * (N + 6))

theorem henry_initial_games (H N L : ℕ)
  (h : initial_games H N L) : H = 62 :=
by
  -- Definitions provided by problem conditions
  cases h with hN h_rest,
  cases h_rest with hL h_rest2,
  cases h_rest2 with hHN hHEq,
  -- Substitution from conditions
  rw [hN, hL] at *,
  -- Use given conditions to solve for H
  have eq1 : H = 3 * 7 := hHN,
  have eq2 : H - 10 = 4 * (7 + 6) := hHEq,
  rw [eq1] at eq2,
  -- Simplify to find H
  linarith

end henry_initial_games_l432_432590


namespace brian_holds_breath_for_60_seconds_l432_432824

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end brian_holds_breath_for_60_seconds_l432_432824


namespace fiftieth_term_l432_432302

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  d ∈ n.digits 10

def valid_sequence_term (n : ℕ) : Prop :=
  n % 4 = 0 ∧ contains_digit 2 n

def sequence_of_valid_terms : List ℕ :=
  List.filter valid_sequence_term (List.range (4*500)) -- 4*500 is an arbitrary upper bound for illustration

theorem fiftieth_term :
  List.nth sequence_of_valid_terms 49 = some 424 :=
by
  sorry

end fiftieth_term_l432_432302


namespace polar_to_rectangular_l432_432842

-- Define the given polar coordinates
def r : ℝ := 7
def θ : ℝ := Real.pi / 3

-- Define the expected rectangular coordinates
def x_expected : ℝ := 3.5
def y_expected : ℝ := 7 * Real.sqrt 3 / 2

-- State the problem as a theorem
theorem polar_to_rectangular :
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  x = x_expected ∧ y = y_expected := by
  -- The proof would go here
  sorry

end polar_to_rectangular_l432_432842


namespace parity_triple_l432_432073

def sequence (B : ℕ → ℕ) : Prop :=
  B 0 = 0 ∧ B 1 = 1 ∧ B 2 = 1 ∧ (∀ n, n ≥ 3 → B n = B (n - 1) + B (n - 2))

def even (n : ℕ) : Prop := n % 2 = 0
def odd (n : ℕ) : Prop := ¬even n

theorem parity_triple (B : ℕ → ℕ) (h : sequence B) :
  odd (B 2021) ∧ odd (B 2022) ∧ even (B 2023) :=
by
  sorry

end parity_triple_l432_432073


namespace statement_is_true_l432_432184

theorem statement_is_true (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h : ∀ x : ℝ, |x + 2| < b → |(3 * x + 2) + 4| < a) : b ≤ a / 3 :=
by
  sorry

end statement_is_true_l432_432184


namespace tan_theta_perpendicular_vectors_l432_432549

theorem tan_theta_perpendicular_vectors (θ : ℝ) (h : Real.sqrt 3 * Real.cos θ + Real.sin θ = 0) : Real.tan θ = - Real.sqrt 3 :=
sorry

end tan_theta_perpendicular_vectors_l432_432549


namespace ratio_of_hypotenuse_segments_l432_432191

theorem ratio_of_hypotenuse_segments
  (x : ℝ) (h : 0 < x) :
  let AB := 4 * x,
      BC := 3 * x,
      AC := Real.sqrt (AB^2 + BC^2),
      BD := (3 * x) / 4 * (5 * x),
      CD := 4/3 * BD,
      AD := AC - CD
  in (AD / CD) = 1 / 7 :=
by {
  have h1 : AB = 4 * x := rfl,
  have h2 : BC = 3 * x := rfl,
  have h3 : AC = Real.sqrt ((4 * x)^2 + (3 * x)^2) := rfl,
  have h4 : AC = 5 * x := by rw [Real.sqrt_eq_rfl.mpr (sq_succ_s_of_approxExact.le_of),
  have h5 : BD = (3/4) * (5 * x) := by sorry,
  have h6 : CD = 4 / 3 * BD := by sorry,
  have h7 : AD = AC - CD := by sorry,
  have h8 : AD / CD = 1 / 7 := by sorry,
  sorry
}

end ratio_of_hypotenuse_segments_l432_432191


namespace clara_total_cookies_l432_432835

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l432_432835


namespace exists_odd_a_b_and_positive_k_l432_432936

theorem exists_odd_a_b_and_positive_k (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ k > 0 ∧ 2 * m = a^5 + b^5 + k * 2^100 := 
sorry

end exists_odd_a_b_and_positive_k_l432_432936


namespace tan_alpha_plus_beta_eq_neg_one_l432_432131

theorem tan_alpha_plus_beta_eq_neg_one (α β : ℝ) (h1 : ∀ x, x^2 + 3*x - 2 = (x - tan α) * (x - tan β)) : 
  tan (α + β) = -1 := 
by
  sorry -- proof is omitted

end tan_alpha_plus_beta_eq_neg_one_l432_432131


namespace sequence_formula_min_value_Sn_min_value_Sn_completion_l432_432121

-- Define the sequence sum Sn
def Sn (n : ℕ) : ℤ := n^2 - 48 * n

-- General term of the sequence
def an (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Conventionally, sequences start from 1 in these problems
  | (n+1) => 2 * (n + 1) - 49

-- Prove that the general term of the sequence produces the correct sum
theorem sequence_formula (n : ℕ) (h : 0 < n) : an n = 2 * n - 49 := by
  sorry

-- Prove that the minimum value of Sn is -576 and occurs at n = 24
theorem min_value_Sn : ∃ n : ℕ, Sn n = -576 ∧ ∀ m : ℕ, Sn m ≥ -576 := by
  use 24
  sorry

-- Alternative form of the theorem using the square completion form 
theorem min_value_Sn_completion (n : ℕ) : Sn n = (n - 24)^2 - 576 := by
  sorry

end sequence_formula_min_value_Sn_min_value_Sn_completion_l432_432121


namespace find_PQ_l432_432578

-- Define the right triangle and its properties
theorem find_PQ'
  (P Q R : Point)
  (hT : Triangle P Q R)
  (hR : angle P R Q = 90) -- Given \(\angle R = 90^\circ\)
  (hTanP : tan (angle Q P R) = 5/3) -- Given \(\tan{P} = \frac{5}{3}\)
  (hPR : dist P R = 12) -- Given \( PR = 12 \)
  : dist P Q = 4 * sqrt 34 := 
sorry

end find_PQ_l432_432578


namespace solve_diophantine_l432_432451

theorem solve_diophantine : ∃ u v : ℤ, 364 * u + 154 * v = 14 ∧ u = 3 ∧ v = -7 :=
by
  use 3
  use -7
  simp
  sorry

end solve_diophantine_l432_432451


namespace otimes_neg2_neg1_l432_432071

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432071


namespace angle_between_vectors_magnitude_of_sum_projection_of_a_l432_432132

variables {a b : ℝ}

-- Define the magnitudes of vectors a and b
def norm_a := 2
def norm_b := 1

-- Define the dot product condition
def dot_product_condition := (2 * (λ (c : ℝ), c * a) - 3 * (λ (c : ℝ), c * b)) 
                               · (2 * (λ (c : ℝ), c * a) + (λ (c : ℝ), c * b)) = 9

-- Define the angle θ
def theta := real.arccos (1 / 2)

-- Prove that the angle between vectors a and b is π/3
theorem angle_between_vectors : theta = real.pi / 3 :=
by sorry

-- Prove the magnitude of the sum of a and b
theorem magnitude_of_sum : real.sqrt ((λ (c : ℝ), c * a + b)^2) = real.sqrt 7 :=
by sorry

-- Prove the projection of a in the direction of a + b
theorem projection_of_a : 
    (λ (c : ℝ), (λ (d : ℝ), c * a) · (λ (e : ℝ), a + b)) 
    / (λ (f : ℝ), real.sqrt ((λ (g : ℝ), g * a + b)^2)) = 5*real.sqrt(7)/7 :=
by sorry

end angle_between_vectors_magnitude_of_sum_projection_of_a_l432_432132


namespace triangle_height_l432_432024

theorem triangle_height (area : ℝ) (base height : ℝ) 
  (h_base_conversion : base = 60) (h_area : area = 48): height = 1.6 :=
by
  -- Create the equation based on the area formula for a triangle
  have h_eq : base * height / 2 = area,
  { sorry },
  -- Substitute base and area
  rw [h_base_conversion, h_area] at h_eq,
  -- Simplify the equation to find the height
  have h_simplify : height * 60 / 2 = 48,
  { sorry },
  -- Solve the equation for height
  have h_final : height = 96 / 60,
  { sorry },
  -- Simplify 96/60 to 1.6
  have h_simplify_final : 96 / 60 = 1.6,
  { sorry },
  -- Conclude that the height is 1.6
  exact h_simplify_final ▸ h_final

end triangle_height_l432_432024


namespace expected_area_nth_circle_l432_432012

noncomputable def expected_area (a d : ℝ) (n : ℕ) : ℝ :=
  π * n * (d + a ^ 2)

theorem expected_area_nth_circle (a d : ℝ) (n : ℕ) :
  (∀ i : ℕ, i ≤ n → ∃ X_i : ℝ, E[X_i] = a ∧ Var[X_i] = d) →
  E[π * ∑ i in finset.range n, X_i ^ 2] = expected_area(a, d, n) :=
by
  sorry

end expected_area_nth_circle_l432_432012


namespace long_show_episode_duration_is_one_hour_l432_432752

-- Definitions for the given conditions
def total_shows : ℕ := 2
def short_show_length : ℕ := 24
def short_show_episode_duration : ℝ := 0.5
def long_show_episodes : ℕ := 12
def total_viewing_time : ℝ := 24

-- Definition of the length of each episode of the longer show
def long_show_episode_length (L : ℝ) : Prop :=
  (short_show_length * short_show_episode_duration) + (long_show_episodes * L) = total_viewing_time

-- Main statement to prove
theorem long_show_episode_duration_is_one_hour : long_show_episode_length 1 :=
by
  -- Proof placeholder
  sorry

end long_show_episode_duration_is_one_hour_l432_432752


namespace euler_children_mean_age_l432_432782

theorem euler_children_mean_age :
  let ages : List ℕ := [7, 7, 7, 12, 12, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  sum_ages = 74 → number_of_children = 7 → mean_age = 74 / 7 :=
by
  intro hsum hnum
  have hmean : mean_age = 74 / 7 := by
    rw [mean_age, hsum, hnum]
  exact hmean

end euler_children_mean_age_l432_432782


namespace janet_gas_usage_l432_432599

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20
def total_driving_distance : ℕ := (2 * distance_dermatologist) + (2 * distance_gynecologist)
def gas_used : ℝ := total_driving_distance / car_efficiency

theorem janet_gas_usage : gas_used = 8 := by
  sorry

end janet_gas_usage_l432_432599


namespace problem_statement_l432_432965

-- Polynomial definitions and properties.
noncomputable def P (k : ℕ) (c : (fin k) → ℤ) : polynomial ℤ :=
∑ i in (fin.range k), (c i) * (X ^ i)

-- Main theorem statement
theorem problem_statement (n k : ℕ) (c : (fin k) → ℤ) 
  (hk_even : even k) (hcoeffs_odd : ∀ i, c i % 2 = 1)
  (hdiv : (X + 1) ^ n - 1 ∣ P k c) : (k + 1) ∣ n :=
sorry

end problem_statement_l432_432965


namespace transfer_12_liters_l432_432577

-- Define the recurrence relation for the number of ways to transfer n liters of milk.
def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| n + 2 := a (n + 1) + a n

-- Define a proposition that states the number of ways to transfer 12 liters of milk is 233.
theorem transfer_12_liters : a 12 = 233 :=
by {
    -- Skip the proof, just state the theorem.
    sorry
}

end transfer_12_liters_l432_432577


namespace women_attended_l432_432040

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l432_432040


namespace initial_sheep_count_l432_432803

theorem initial_sheep_count : 
  ∃ (n : ℕ), 
    (∀ i : Fin 6, n(i) + 1) * 2 = n(i-value) )  
→ n = 254 := 
sorry

end initial_sheep_count_l432_432803


namespace PQRS_on_circle_l432_432028

-- Definition of an acute-angled triangle
structure AcuteAngledTriangle (A B C : Point) :=
(altitude_from_C : Line)
(altitude_from_B : Line)
(acute : ∀ (a b c : ℝ) (h₁ : a < π / 2) (h₂ : b < π / 2) (h₃ : c < π / 2), a + b + c = π)

-- The problem statement in Lean 4
theorem PQRS_on_circle (A B C P Q R S : Point) 
  (h_triangle : AcuteAngledTriangle A B C)
  (h_circle_AB : Circle (A.midpoint B))
  (h_circle_AC : Circle (A.midpoint C))
  (h_PQ_on_altitude : P ∈ h_triangle.altitude_from_C ∧ Q ∈ h_triangle.altitude_from_C)
  (h_RS_on_altitude : R ∈ h_triangle.altitude_from_B ∧ S ∈ h_triangle.altitude_from_B)
  (h_PQ_on_circle_AB : P ∈ h_circle_AB ∧ Q ∈ h_circle_AB)
  (h_RS_on_circle_AC : R ∈ h_circle_AC ∧ S ∈ h_circle_AC) :
  ∃ O, Circle O ∧ P ∈ Circle O ∧ Q ∈ Circle O ∧ R ∈ Circle O ∧ S ∈ Circle O :=
sorry

end PQRS_on_circle_l432_432028


namespace hexagon_diagonals_sum_l432_432004
open Real

-- Definitions
noncomputable def hexagon_inscribed_in_circle_diagonal_sum : Prop :=
  ∃ (x y z : ℝ),
    x = 66.8 ∧ y = 142.8 ∧ z = 155.8 ∧
    (∀ (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F],
      ∃ (AB BC CD DE EF FA : ℝ), 
      AB = 40 ∧ BC = 100 ∧ CD = 100 ∧ DE = 100 ∧ EF = 100 ∧ FA = 60 ∧
      A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
      (AC = x) ∧ (AD = y) ∧ (AE = z) ∧ 
      x + y + z = 365.4)

theorem hexagon_diagonals_sum : hexagon_inscribed_in_circle_diagonal_sum :=
by
  -- Placeholder for proof
  sorry

end hexagon_diagonals_sum_l432_432004


namespace first_term_exceeds_10000_is_a9_l432_432720

def sequence : ℕ → ℕ
| 0     := 3
| (n+1) := 2 * (Finset.range (n+1)).sum (λ i, sequence i)

theorem first_term_exceeds_10000_is_a9 : sequence 8 = 13122 := 
sorry

end first_term_exceeds_10000_is_a9_l432_432720


namespace time_to_be_100_miles_apart_l432_432404

noncomputable def distance_apart (x : ℝ) : ℝ :=
  Real.sqrt ((12 * x) ^ 2 + (16 * x) ^ 2)

theorem time_to_be_100_miles_apart : ∃ x : ℝ, distance_apart x = 100 ↔ x = 5 :=
by {
  sorry
}

end time_to_be_100_miles_apart_l432_432404


namespace initial_speed_approx_30_52_l432_432799

-- Defining the variables and conditions
variable (V : ℝ)  -- Initial speed from P to Q

def return_speed (V : ℝ) : ℝ := 1.30 * V

def avg_speed (V : ℝ) : ℝ := (2 * V * return_speed V) / (V + return_speed V)

-- Stating the theorem to prove the initial speed
theorem initial_speed_approx_30_52 :
  avg_speed V = 34.5 → V = 30.52 :=
by
  sorry

end initial_speed_approx_30_52_l432_432799


namespace recycling_program_earnings_l432_432225

-- Define conditions
def signup_earning : ℝ := 5.00
def referral_earning_tier1 : ℝ := 8.00
def referral_earning_tier2 : ℝ := 1.50
def friend_earning_signup : ℝ := 5.00
def friend_earning_tier2 : ℝ := 2.00

def initial_friend_count : ℕ := 5
def initial_friend_tier1_referrals_day1 : ℕ := 3
def initial_friend_tier1_referrals_week : ℕ := 2

def additional_friend_count : ℕ := 2
def additional_friend_tier1_referrals : ℕ := 1

-- Calculate Katrina's total earnings
def katrina_earnings : ℝ :=
  signup_earning +
  (initial_friend_count * referral_earning_tier1) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * referral_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * referral_earning_tier2) +
  (additional_friend_count * referral_earning_tier1) +
  (additional_friend_count * additional_friend_tier1_referrals * referral_earning_tier2)

-- Calculate friends' total earnings
def friends_earnings : ℝ :=
  (initial_friend_count * friend_earning_signup) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * friend_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * friend_earning_tier2) +
  (additional_friend_count * friend_earning_signup) +
  (additional_friend_count * additional_friend_tier1_referrals * friend_earning_tier2)

-- Calculate combined total earnings
def combined_earnings : ℝ := katrina_earnings + friends_earnings

-- The proof assertion
theorem recycling_program_earnings : combined_earnings = 190.50 :=
by sorry

end recycling_program_earnings_l432_432225


namespace smallest_rectangles_cover_square_l432_432763

theorem smallest_rectangles_cover_square :
  ∃ (n : ℕ), n = 8 ∧ ∀ (a : ℕ), ∀ (b : ℕ), (a = 2) ∧ (b = 4) → 
  ∃ (s : ℕ), s = 8 ∧ (s * s) / (a * b) = n :=
by
  sorry

end smallest_rectangles_cover_square_l432_432763


namespace sum_tens_units_digits_11_pow_2003_l432_432336

theorem sum_tens_units_digits_11_pow_2003 : 
  let n := 11 ^ 2003 in
  let tens_digit := (n / 10) % 10 in
  let units_digit := n % 10 in
  (tens_digit + units_digit = 4) :=
by
  sorry

end sum_tens_units_digits_11_pow_2003_l432_432336


namespace knights_count_l432_432633

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432633


namespace area_of_OABC_rhombus_l432_432130

noncomputable def area_of_rhombus (A B C : ℝ × ℝ) (O : ℝ × ℝ := (0, 0)) : ℝ :=
  let B := (Real.sqrt 2, 0)
  ∧ let ellipse := ∀ p, p ∈ [A, B, C] → (p.1 ^ 2 / 2 + p.2 ^ 2 = 1)
  ∧ let rhombus := OABC_is_rhombus O A B C
  find_rhombus_area O A B C

theorem area_of_OABC_rhombus
  (A B C : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (B_right_vertex : B = (Real.sqrt 2, 0))
  (OABC_is_rhombus : rhombus A B C O)
  (points_on_ellipse : ∀ p, p ∈ [A, B, C] → (p.1 ^ 2 / 2 + p.2 ^ 2 = 1))
  : area_of_rhombus A B C O = Real.sqrt 6 / 2 := sorry

end area_of_OABC_rhombus_l432_432130


namespace distinct_prime_divisors_l432_432264

theorem distinct_prime_divisors {n : ℕ} : ∃ p : ℕ → Prop, (∀ m < n, ∃ q, p q ∧ prime q) ∧ (p.count (2 ^ (2 ^ n) + 2 ^ (2 ^ (n - 1)) + 1) = n) :=
  sorry

end distinct_prime_divisors_l432_432264


namespace floor_eq_20_l432_432882

theorem floor_eq_20 {x : ℝ} : (⟦x * ⟦x⟧⟧) = 20 ↔ 5 ≤ x ∧ x < 5.25 :=
begin
  sorry
end

end floor_eq_20_l432_432882


namespace perpendicular_line_through_point_l432_432093

theorem perpendicular_line_through_point (x1 y1 : ℝ) (a b c : ℝ) :
  (a = 1) → (b = -2) → (c = 3) → (x1 = -1) → (y1 = 3) →
  ∃ m : ℝ, (m = -2) ∧ (∀ x y : ℝ, (y - y1 = m * (x - x1)) → (2 * x + y - 1 = 0)) :=
by
  intros ha hb hc hx1 hy1
  use -2
  split
  · refl
  · intros x y hxy
    rw [hx1, hy1] at *
    nlinarith

end perpendicular_line_through_point_l432_432093


namespace point_in_fourth_quadrant_l432_432211

theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : 
(x > 0) → (y < 0) → (x, y) = (2, -3) → quadrant (2, -3) = 4 :=
by
  sorry

end point_in_fourth_quadrant_l432_432211


namespace fox_cub_distribution_l432_432669

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l432_432669


namespace volume_of_pyramid_l432_432737

theorem volume_of_pyramid (a α : ℝ) (h₀ : 0 < a) (h₁ : 0 < α ∧ α < π) :
  ∃ V, V = (1 / 3) * a^3 * (sin (α / 2))^4 * sin α := 
sorry

end volume_of_pyramid_l432_432737


namespace new_concentration_l432_432371

theorem new_concentration (Q : ℝ) (f : ℝ) 
  (hf : f = 0.8181818181818181) 
  (hQ: Q > 0) :
  let original_concentration := 0.80
  let replaced_concentration := 0.25
  let new_concentration := (original_concentration * (1 - f) * Q + replaced_concentration * f * Q) / Q
  in new_concentration = 0.35 :=
by
  intro original_concentration replaced_concentration new_concentration
  rw [hf]
  let expr := (((0.80 : ℝ) * (1 - 0.8181818181818181) * Q + (0.25 : ℝ) * 0.8181818181818181 * Q) / Q)
  have h1 : expr = 0.35 := by
    field_simp [Q]
    ring
  exact h1

end new_concentration_l432_432371


namespace alpha_beta_square_l432_432057

theorem alpha_beta_square (α β : ℝ) (h₁ : α^2 = 2*α + 1) (h₂ : β^2 = 2*β + 1) (hαβ : α ≠ β) :
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_l432_432057


namespace number_of_valid_three_digit_numbers_l432_432979

noncomputable def count_valid_numbers : Nat :=
  let count_valid_pairs (a : Nat) : Nat :=
    if 1 ≤ a ∧ a ≤ 9 then
      (List.finRange 10).sum (λ b => (List.finRange 10).count (λ c => a > b + c))
    else
      0
  (List.finRange (10 - 1)).sum (λ a => count_valid_pairs (a + 1))

theorem number_of_valid_three_digit_numbers :
  (List.finRange 900).count (λ n => let a := n / 100
                                    let b := (n / 10) % 10
                                    let c := n % 10
                                    a > b + c) = count_valid_numbers := sorry

end number_of_valid_three_digit_numbers_l432_432979


namespace women_attended_l432_432041

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l432_432041


namespace evaluate_power_l432_432860

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l432_432860


namespace otimes_example_l432_432065

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432065


namespace probability_one_win_one_loss_l432_432574

def P_A_win_B := 0.8
def P_A_win_C := 0.7

def P_A_loss_B := 1 - P_A_win_B
def P_A_loss_C := 1 - P_A_win_C

def P_A_one_win_one_loss := P_A_win_B * P_A_loss_C + P_A_win_C * P_A_loss_B

theorem probability_one_win_one_loss : P_A_one_win_one_loss = 0.38 := by
  have h1 : P_A_win_B * P_A_loss_C = 0.8 * 0.3 := rfl
  have h2 : P_A_win_C * P_A_loss_B = 0.7 * 0.2 := rfl
  have h3 : P_A_one_win_one_loss = (0.8 * 0.3) + (0.7 * 0.2) := by
    unfold P_A_one_win_one_loss; rw [h1, h2]
  calc P_A_one_win_one_loss = (0.8 * 0.3) + (0.7 * 0.2) : by rw [h3]
                            ... = 0.24 + 0.14 : rfl
                            ... = 0.38 : rfl

end probability_one_win_one_loss_l432_432574


namespace calc_exponent_l432_432899

theorem calc_exponent (m n : ℝ) (h1 : 9^m = 3) (h2 : 27^n = 4) : 3^(2*m + 3*n) = 12 := by
  sorry

end calc_exponent_l432_432899


namespace total_money_is_102_l432_432458

-- Defining the amounts of money each person has
def Jack_money : ℕ := 26
def Ben_money : ℕ := Jack_money - 9
def Eric_money : ℕ := Ben_money - 10
def Anna_money : ℕ := Jack_money * 2

-- Defining the total amount of money
def total_money : ℕ := Eric_money + Ben_money + Jack_money + Anna_money

-- Proving the total money is 102
theorem total_money_is_102 : total_money = 102 :=
by
  -- this is where the proof would go
  sorry

end total_money_is_102_l432_432458


namespace pugs_working_together_l432_432370

theorem pugs_working_together (P : ℕ) (H1 : P * 45 = 15 * 12) : P = 4 :=
by {
  sorry
}

end pugs_working_together_l432_432370


namespace cuboids_painted_l432_432461

theorem cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) (num_cuboids : ℕ) 
(h1 : faces_per_cuboid = 6) (h2 : total_faces = 30) (h3 : num_cuboids = total_faces / faces_per_cuboid) :
num_cuboids = 5 :=
by {
  rw [h1, h2, h3], 
  norm_num,
}

end cuboids_painted_l432_432461


namespace evaluate_power_l432_432859

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l432_432859


namespace evaluate_power_l432_432861

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l432_432861


namespace alligators_hiding_correct_l432_432044

def total_alligators := 75
def not_hiding_alligators := 56

def hiding_alligators (total not_hiding : Nat) : Nat :=
  total - not_hiding

theorem alligators_hiding_correct : hiding_alligators total_alligators not_hiding_alligators = 19 := 
by
  sorry

end alligators_hiding_correct_l432_432044


namespace problem_statement_l432_432241

theorem problem_statement (m : ℝ) : 
  let a := (2 : ℝ, 3 : ℝ)
      b := (-3 : ℝ, m)
  in (a.1 * (2 * m - 3) + a.2 * (3 * m + m) = 0) → m = 3 / 8 :=
by
  intros
  sorry

end problem_statement_l432_432241


namespace solve_for_n_l432_432459

variable (n : ℚ)

theorem solve_for_n (h : 22 + Real.sqrt (-4 + 18 * n) = 24) : n = 4 / 9 := by
  sorry

end solve_for_n_l432_432459


namespace parabola_sum_l432_432877

variables (a b c x y : ℝ)

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_sum (h1 : ∀ x, quadratic a b c x = -(x - 3)^2 + 4)
    (h2 : quadratic a b c 1 = 0)
    (h3 : quadratic a b c 5 = 0) :
    a + b + c = 0 :=
by
  -- We assume quadratic(a, b, c, x) = a * x^2 + b * x + c
  -- We assume quadratic(a, b, c, 1) = 0 and quadratic(a, b, c, 5) = 0
  -- We need to prove a + b + c = 0
  sorry

end parabola_sum_l432_432877


namespace otimes_example_l432_432063

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432063


namespace smallest_s_fractional_l432_432739

def is_fractional (s : ℝ) := ∃ (a b : ℤ), b ≠ 0 ∧ s = a / b

def is_triangle_side (a b c : ℝ) := a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_s_fractional :
  ∃ (s : ℝ), is_fractional s ∧ is_triangle_side 7.5 11.5 s ∧
  (∀ t, is_fractional t ∧ is_triangle_side 7.5 11.5 t → s ≤ t) ∧ s = 9 / 2 :=
begin
  sorry
end

end smallest_s_fractional_l432_432739


namespace evaluate_expression_l432_432724

noncomputable def f : ℝ → ℝ := sorry

lemma f_condition (a : ℝ) : f (a + 1) = f a * f 1 := sorry

lemma f_one : f 1 = 2 := sorry

theorem evaluate_expression :
  (f 2018 / f 2017) + (f 2019 / f 2018) + (f 2020 / f 2019) = 6 :=
sorry

end evaluate_expression_l432_432724


namespace new_selling_price_mangoes_new_selling_price_apples_new_selling_price_pears_new_selling_price_bananas_l432_432002

theorem new_selling_price_mangoes : ∀ (SP : ℝ) (Loss : ℝ) (DesiredProfit : ℝ),
  SP = 12 ∧ Loss = 0.15 ∧ DesiredProfit = 0.05 → 
  ((SP / (1 - Loss)) * (1 + DesiredProfit)) ≈ 14.83 := 
by 
  sorry

theorem new_selling_price_apples : ∀ (SP : ℝ) (Loss : ℝ) (DesiredProfit : ℝ),
  SP = 30 ∧ Loss = 0.10 ∧ DesiredProfit = 0.15 → 
  ((SP / (1 - Loss)) * (1 + DesiredProfit)) ≈ 38.33 := 
by 
  sorry

theorem new_selling_price_pears : ∀ (SP : ℝ) (Profit : ℝ) (DesiredProfit : ℝ),
  SP = 40 ∧ Profit = 0.10 ∧ DesiredProfit = 0.20 → 
  ((SP / (1 + Profit)) * (1 + DesiredProfit)) ≈ 43.63 := 
by 
  sorry

theorem new_selling_price_bananas : ∀ (SP : ℝ) (Profit : ℝ) (DesiredProfit : ℝ),
  SP = 20 ∧ Profit = 0.05 ∧ DesiredProfit = 0.12 → 
  ((SP / (1 + Profit)) * (1 + DesiredProfit)) ≈ 21.34 := 
by 
  sorry

end new_selling_price_mangoes_new_selling_price_apples_new_selling_price_pears_new_selling_price_bananas_l432_432002


namespace paperboy_problem_l432_432386

noncomputable def delivery_ways (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else if n = 2 then 4
  else if n = 3 then 8
  else if n = 4 then 15
  else delivery_ways (n - 1) + delivery_ways (n - 2) + delivery_ways (n - 3) + delivery_ways (n - 4)

theorem paperboy_problem : delivery_ways 12 = 2872 :=
  sorry

end paperboy_problem_l432_432386


namespace correct_operation_l432_432343

theorem correct_operation :
  (∀ (b : ℤ), b^2 * b^3 = b^5) ∧ -- for Option A (incorrect)
  (∀ (x y : ℤ), (2 * x + y)^2 = (2 * x)^2 + 2 * (2 * x) * y + y^2) ∧ -- for Option B (incorrect)
  (∀ (x y : ℤ), (-3 * x^2 * y)^3 = (-3)^3 * (x^2)^3 * y^3) ∧ -- for Option C (correct)
  (∀ (x : ℤ), x + x = 2 * x) -- for Option D (incorrect)
  :=
begin
  sorry
end

end correct_operation_l432_432343


namespace tangent_line_at_1_2_l432_432360

noncomputable def tangent_line_eq_at_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), f' p.1 = m ∧ (∀ x, (f x - p.2) = m * (x - p.1) + (f p.1 - p.2))

theorem tangent_line_at_1_2 : 
  tangent_line_eq_at_point (λ x : ℝ, -x^3 + 3*x^2) 
                           (λ x : ℝ, -3*x^2 + 6*x)
                           (1, 2) := 
sorry

end tangent_line_at_1_2_l432_432360


namespace janet_gas_usage_l432_432601

theorem janet_gas_usage :
  ∀ (d_dermatologist d_gynecologist miles_per_gallon : ℕ),
    d_dermatologist = 30 →
    d_gynecologist = 50 →
    miles_per_gallon = 20 →
    (2 * d_dermatologist + 2 * d_gynecologist) / miles_per_gallon = 8 :=
by
  intros d_dermatologist d_gynecologist miles_per_gallon
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end janet_gas_usage_l432_432601


namespace polar_to_rectangular_l432_432841

-- Define the given polar coordinates
def r : ℝ := 7
def θ : ℝ := Real.pi / 3

-- Define the expected rectangular coordinates
def x_expected : ℝ := 3.5
def y_expected : ℝ := 7 * Real.sqrt 3 / 2

-- State the problem as a theorem
theorem polar_to_rectangular :
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  x = x_expected ∧ y = y_expected := by
  -- The proof would go here
  sorry

end polar_to_rectangular_l432_432841


namespace maxValue_of_MF1_MF2_l432_432905

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432905


namespace maximum_xyz_l432_432661

-- Given conditions
variables {x y z : ℝ}

-- Lean 4 statement with the conditions
theorem maximum_xyz (h₁ : x * y + 2 * z = (x + z) * (y + z))
  (h₂ : x + y + 2 * z = 2)
  (h₃ : 0 < x) (h₄ : 0 < y) (h₅ : 0 < z) :
  xyz = 0 :=
sorry

end maximum_xyz_l432_432661


namespace range_of_a_l432_432989

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 4 → deriv (λ x, x^2 + 2 * (a - 1) * x + 2) x ≤ 0) →
  a ≤ -3 := 
by 
  -- placeholder for proof
  sorry

end range_of_a_l432_432989


namespace product_of_min_max_eq_one_l432_432667

theorem product_of_min_max_eq_one {x y : ℝ} (h : 9 * x^2 + 12 * x * y + 4 * y^2 = 1) :
  let k := 3 * x^2 + 4 * x * y + 2 * y^2 in
  let m := k in
  let M := k in
  m * M = 1 :=
by
  sorry

end product_of_min_max_eq_one_l432_432667


namespace min_volume_tetrahedron_l432_432289

/-- Definition of the convex quadrilateral base A, B, C, D of a tetrahedron PABCD with specific intersection and area properties. -/
variables {A B C D O P : Type}
variables [hconvex_quad: ConvexQuadrilateral A B C D]
variables [hintersection: Intersect AC BD O]
variables [harea_AOB: AreaTriangle A O B = 36]
variables [harea_COD: AreaTriangle C O D = 64]
variables [hheight: Height P A B C D = 9]

/-- Minimum volume of the tetrahedron P-ABCD -/
theorem min_volume_tetrahedron : Volume (Tetrahedron P A B C D) = 588 := by
  sorry

end min_volume_tetrahedron_l432_432289


namespace mike_spent_total_l432_432685

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l432_432685


namespace nested_sqrt_bound_l432_432895

theorem nested_sqrt_bound (n : ℕ) (h : 2 ≤ n) : 
  (nat.sqrt_up (2 * nat.sqrt_up (3 * nat.sqrt_up (4 * nat.sqrt_up (nat.prod (list.range (n-1) (λ x, (x + 2)))))))) < 3 :=
sorry

end nested_sqrt_bound_l432_432895


namespace minimum_dimes_to_afford_sneakers_l432_432056

-- Define constants and conditions using Lean
def sneaker_cost : ℝ := 45.35
def ten_dollar_bills_count : ℕ := 3
def quarter_count : ℕ := 4
def dime_value : ℝ := 0.1
def quarter_value : ℝ := 0.25
def ten_dollar_bill_value : ℝ := 10.0

-- Define a function to calculate the total amount based on the number of dimes
def total_amount (dimes : ℕ) : ℝ :=
  (ten_dollar_bills_count * ten_dollar_bill_value) +
  (quarter_count * quarter_value) +
  (dimes * dime_value)

-- The main theorem to be proven
theorem minimum_dimes_to_afford_sneakers (n : ℕ) : total_amount n ≥ sneaker_cost ↔ n ≥ 144 :=
by
  sorry

end minimum_dimes_to_afford_sneakers_l432_432056


namespace max_min_pirate_sum_l432_432164

-- Define the pirate sum operation
def pirate_sum (a b c d : ℕ) : ℕ × ℕ :=
  (a + c, b + d)

-- Statement to prove the maximum and minimum possible value of the last fraction
theorem max_min_pirate_sum (n : ℕ) (h : n ≥ 3) :
  ∃ (max min : ℕ × ℕ),
  (∀ a b, max = (1, 2)) ∧ (∀ a b, min = (1, n - 1)) :=
begin
  sorry,
end

end max_min_pirate_sum_l432_432164


namespace james_points_l432_432222

theorem james_points (x : ℕ) :
  13 * 3 + 20 * x = 79 → x = 2 :=
by
  sorry

end james_points_l432_432222


namespace max_product_of_distances_l432_432920

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432920


namespace div_by_5_mul_diff_l432_432548

theorem div_by_5_mul_diff (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
by
  sorry

end div_by_5_mul_diff_l432_432548


namespace angle_CED_gt_45_l432_432643

-- Definitions of geometric notions for the problem
variables (A B C D E : Type) [EuclideanGeometry A B C D E]

-- Assuming ABC is an acute-angled triangle
axiom acute_triangle (ABC : Triangle A B C) : acute (angle A B C)

-- Definitions of the bisectors and altitudes
axiom bisector_AD (AD : Line A D) : is_bisector (angle A B C) D
axiom altitude_BE (BE : Line B E) : is_altitude (Line B E) (Line A C)

-- The theorem to be proved
theorem angle_CED_gt_45 (ABC : Triangle A B C) (AD : Line A D) (BE : Line B E) (D : Point) (E : Point) :
  acute (angle A B C) → is_bisector (angle A B C) D → is_altitude (Line B E) (Line A C) → 45 < angle C E D :=
by sorry

end angle_CED_gt_45_l432_432643


namespace pictures_left_l432_432767

def zoo_pics : ℕ := 802
def museum_pics : ℕ := 526
def beach_pics : ℕ := 391
def amusement_park_pics : ℕ := 868
def duplicates_deleted : ℕ := 1395

theorem pictures_left : 
  (zoo_pics + museum_pics + beach_pics + amusement_park_pics - duplicates_deleted) = 1192 := 
by
  sorry

end pictures_left_l432_432767


namespace half_angle_in_second_quadrant_l432_432232

def quadrant_of_half_alpha (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : Prop :=
  π / 2 < α / 2 ∧ α / 2 < 3 * π / 4

theorem half_angle_in_second_quadrant (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : quadrant_of_half_alpha α hα1 hα2 hcos :=
sorry

end half_angle_in_second_quadrant_l432_432232


namespace cylindrical_coord_correct_l432_432435

-- Definitions from conditions
def rect_point := (3 : ℝ, -3 * Real.sqrt 3, 2)

def cylindrical_r (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
def cylindrical_theta (x y : ℝ) : ℝ := Real.arctan (y / x)
def cylindrical_z (z : ℝ) : ℝ := z

noncomputable def cylindrical_coords (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (cylindrical_r x y, cylindrical_theta x y, cylindrical_z z)

-- The proof problem
theorem cylindrical_coord_correct :
  cylindrical_coords rect_point = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end cylindrical_coord_correct_l432_432435


namespace flower_beds_fraction_l432_432800

theorem flower_beds_fraction (yard_length yard_width : ℝ)
  (base1 base2 : ℝ) (triangle_legs : ℝ) (flower_beds_area yard_area : ℝ) :
  yard_length = 30 →
  yard_width = 8 →
  base1 = 12 →
  base2 = 22 →
  triangle_legs = (base2 - base1) / 2 →
  flower_beds_area = 2 * (1 / 2 * triangle_legs^2) →
  yard_area = yard_length * yard_width →
  flower_beds_area / yard_area = 5 / 48 :=
begin
  sorry
end

end flower_beds_fraction_l432_432800


namespace quadratic_inequality_solution_l432_432740

theorem quadratic_inequality_solution : 
  ∀ x : ℝ, (2 * x ^ 2 + 7 * x + 3 > 0) ↔ (x < -3 ∨ x > -0.5) :=
by
  sorry

end quadratic_inequality_solution_l432_432740


namespace sufficient_but_not_necessary_condition_l432_432498

theorem sufficient_but_not_necessary_condition 
  (a : ℕ → ℤ) 
  (h : ∀ n, |a (n + 1)| < a n) : 
  (∀ n, a (n + 1) < a n) ∧ 
  ¬(∀ n, a (n + 1) < a n → |a (n + 1)| < a n) := 
by 
  sorry

end sufficient_but_not_necessary_condition_l432_432498


namespace knights_count_l432_432637

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432637


namespace knights_count_l432_432621

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432621


namespace max_value_expression_l432_432666

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end max_value_expression_l432_432666


namespace knights_count_in_meeting_l432_432624

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432624


namespace imaginary_part_of_complex_number_l432_432727

theorem imaginary_part_of_complex_number :
  let z := (1 + Complex.I)^2 * (2 + Complex.I)
  Complex.im z = 4 :=
by
  sorry

end imaginary_part_of_complex_number_l432_432727


namespace knights_count_in_meeting_l432_432628

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432628


namespace solution_to_equation_l432_432075

def star_operation (a b : ℝ) : ℝ := a^2 - 2 * a * b + b^2

theorem solution_to_equation : ∀ (x : ℝ), star_operation (x - 4) 1 = 0 → x = 5 :=
by
  intro x
  assume h
  -- Skipping the proof steps with sorry
  sorry

end solution_to_equation_l432_432075


namespace number_of_combinations_divisible_by_3_l432_432422

theorem number_of_combinations_divisible_by_3 :
  ∃ count : ℕ, count = 1485100 ∧
  ∀ (s : finset ℕ), s ⊆ (finset.range 301).erase 0 ∧ s.card = 3 → 
  (∑ x in s, x) % 3 = 0 → s.card = count :=
begin
  sorry
end

end number_of_combinations_divisible_by_3_l432_432422


namespace knights_count_l432_432616

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432616


namespace combined_swim_time_l432_432439

theorem combined_swim_time 
    (freestyle_time: ℕ)
    (backstroke_without_factors: ℕ)
    (backstroke_with_factors: ℕ)
    (butterfly_without_factors: ℕ)
    (butterfly_with_factors: ℕ)
    (breaststroke_without_factors: ℕ)
    (breaststroke_with_factors: ℕ) :
    freestyle_time = 48 ∧
    backstroke_without_factors = freestyle_time + 4 ∧
    backstroke_with_factors = backstroke_without_factors + 2 ∧
    butterfly_without_factors = backstroke_without_factors + 3 ∧
    butterfly_with_factors = butterfly_without_factors + 3 ∧
    breaststroke_without_factors = butterfly_without_factors + 2 ∧
    breaststroke_with_factors = breaststroke_without_factors - 1 →
    freestyle_time + backstroke_with_factors + butterfly_with_factors + breaststroke_with_factors = 216 :=
by
  sorry

end combined_swim_time_l432_432439


namespace minimum_value_of_x_plus_y_existence_of_minimum_value_l432_432513

theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 :=
sorry

theorem existence_of_minimum_value (x y : ℝ) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 8 ∧ x + y = 2 * Real.sqrt 10 - 3 :=
sorry

end minimum_value_of_x_plus_y_existence_of_minimum_value_l432_432513


namespace knights_count_l432_432612

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432612


namespace volleyball_tournament_ranking_l432_432999

theorem volleyball_tournament_ranking : 
  ∀ E F G H : Type,
  (E ∧ F ∧ G ∧ H) →  -- Teams E, F, G, and H do exist
  (E ≠ F ∧ G ≠ H) → -- Teams E and F are different; Teams G and H are different
  (advantage E F ∧ advantage G H) → -- Team E has an advantage over Team F; Team G has an advantage over Team H
  (no_ties : ∀ (t1 t2 : Type), t1 ≠ t2 → bool) → -- There are no ties
  (adv_win : ∀ (t1 t2 : Type), advantage t1 t2 → wins t1 t2) → -- Advantage translates to a win
  ∃ num_sequences : ℕ, num_sequences = 4 := sorry

end volleyball_tournament_ranking_l432_432999


namespace initial_amount_l432_432690

theorem initial_amount (spent left : ℕ) (h_spent : spent = 38) (h_left : left = 90) : (spent + left = 128) :=
by
  rw [h_spent, h_left]
  exact rfl

end initial_amount_l432_432690


namespace eval_sqrt_pow_l432_432870

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l432_432870


namespace volume_of_cylinder_formed_by_rectangle_rotation_l432_432484

theorem volume_of_cylinder_formed_by_rectangle_rotation
  (length width : ℝ)
  (h_length : length = 20)
  (h_width : width = 10) :
  let r := width / 2
      h := length
      V := π * r^2 * h
  in V = 500 * π :=
by
  sorry

end volume_of_cylinder_formed_by_rectangle_rotation_l432_432484


namespace min_value_complex_op_min_value_op_eq_l432_432441

def complex_mod (a b : ℝ) := real.sqrt (a^2 + b^2)

-- Definition of the operation
def op (z1 z2 : ℂ) := (abs z1 + abs z2) / 2

theorem min_value_complex_op (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  op (complex.of_real a + complex.I * complex.of_real b) (complex.conj (complex.of_real a + complex.I * complex.of_real b)) = 
  complex_mod a b :=
by {
  sorry
}

theorem min_value_op_eq : ∀ (a b : ℝ), 0 < a → 0 < b → a + b = 3 → 
  complex_mod (a / √2) (b / √2) = 3 * √2 / 2 :=
by {
  sorry
}

end min_value_complex_op_min_value_op_eq_l432_432441


namespace skew_lines_intersection_l432_432167

theorem skew_lines_intersection
  (a b : ℝ → ℝ → ℝ) -- skew lines a and b
  (α β : ℝ → ℝ → ℝ) -- planes α and β
  (c : ℝ → ℝ → ℝ) -- line c
  (ha : ∀ p, α p → a p) -- line a is in plane α
  (hb : ∀ p, β p → b p) -- line b is in plane β
  (hc : ∀ p, (α p ∧ β p) → c p) -- line c is the intersection of planes α and β
  (h_skew : ∃ p q, ¬∃ r, a r = b r) -- a and b are skew
  : ∃ p, c p ∨ b p :=
by
  sorry

end skew_lines_intersection_l432_432167


namespace circulation_1961_to_total_ratio_l432_432707

variable (C_1962 : ℤ) (r : ℚ)

def circulation_1961_ratio (average : ℚ) : ℚ :=
  let circulation_1961 := 4 * average
  let circulation_1962_1970 := 9 * average
  let total_circulation := circulation_1961 + circulation_1962_1970
  circulation_1961 / total_circulation

theorem circulation_1961_to_total_ratio (A : ℚ) (h_average : A = 1/9 * ∑ i in naturals.Icc 1 9, C_1962 * (1 + r/100)^i) :
  circulation_1961_ratio C_1962 r A = 4 / 13 := by
    sorry

end circulation_1961_to_total_ratio_l432_432707


namespace value_of_expression_l432_432945

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : x^2 + 3*x - 2 = 0 := 
by {
  -- proof logic will be here
  sorry
}

end value_of_expression_l432_432945


namespace circumcircle_radius_of_isosceles_trapezoid_l432_432706

variables (a b : ℝ) (α : ℝ)

def isosceles_trapezoid_circumcircle_radius (a b α : ℝ) : ℝ :=
  if a > b then
    sqrt (a^2 + b^2 + 2 * a * b * (Real.cos (2 * α))) / (2 * Real.sin (2 * α))
  else
    0

theorem circumcircle_radius_of_isosceles_trapezoid
  (h : a > b) :
  isosceles_trapezoid_circumcircle_radius a b α =
  sqrt (a^2 + b^2 + 2 * a * b * (Real.cos (2 * α))) / (2 * Real.sin (2 * α)) :=
by
  unfold isosceles_trapezoid_circumcircle_radius
  rw if_pos h
  sorry

end circumcircle_radius_of_isosceles_trapezoid_l432_432706


namespace convert_polar_to_rectangular_l432_432844

-- Definitions of conversion formulas and given point in polar coordinates
def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Given point in polar coordinates
def givenPolarPoint : ℝ × ℝ := (7, Real.pi / 3)

-- Correct answer in rectangular coordinates
def correctRectangularCoordinates : ℝ × ℝ := (3.5, 7 * Real.sqrt 3 / 2)

-- Proof statement
theorem convert_polar_to_rectangular :
  polarToRectangular (givenPolarPoint.1) (givenPolarPoint.2) = correctRectangularCoordinates :=
by
  sorry

end convert_polar_to_rectangular_l432_432844


namespace projection_properties_l432_432301

variable (b c : ℝ)

-- Define the vertices of the rectangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (b, 0)
def C : ℝ × ℝ := (b, c)
def D : ℝ × ℝ := (0, c)

-- Define the projection of C on BD
def E : ℝ × ℝ := (b^3 / (b^2 + c^2), c^3 / (b^2 + c^2))

-- Define the projections of E on AB and AD
def F : ℝ × ℝ := (b^3 / (b^2 + c^2), 0)
def G : ℝ × ℝ := (0, c^3 / (b^2 + c^2))

-- Define the distances AF, AG, and AC
def AF : ℝ := (b^3 / (b^2 + c^2))
def AG : ℝ := (c^3 / (b^2 + c^2))
def AC : ℝ := Real.sqrt (b^2 + c^2)

-- Prove the given equation
theorem projection_properties :
  AF ^ (2 / 3) + AG ^ (2 / 3) = AC ^ (2 / 3) :=
by
  sorry

end projection_properties_l432_432301


namespace modulus_product_l432_432475

theorem modulus_product (a b : ℂ) : |a - b * complex.i| * |a + b * complex.i| = 25 := by
  have h1 : complex.norm (4 - 3 * complex.i) = 5 := by
    sorry
  have h2 : complex.norm (4 + 3 * complex.i) = 5 := by
    sorry
  rw [← complex.norm_mul, (4 - 3 * complex.i).mul_conj_self, (4 + 3 * complex.i).mul_conj_self, add_comm] at h1
  rw [mul_comm, mul_comm (complex.norm _), ← mul_assoc, h2, mul_comm, mul_assoc]
  exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).1 h1 

end modulus_product_l432_432475


namespace four_digit_numbers_without_repeated_digits_l432_432169

theorem four_digit_numbers_without_repeated_digits : 
  let digits := {0, 1, 2, 3} in
  let valid_numbers := {x | (∀ d ∈ digits, x has all digits d exactly once) ∧ (fst_digit x ≠ 0) ∧ (1000 ≤ x ∧ x ≤ 9999)} in
  valid_numbers.card = 18 := 
sorry

end four_digit_numbers_without_repeated_digits_l432_432169


namespace total_daily_salary_l432_432200

def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def num_managers : ℕ := 2
def num_clerks : ℕ := 3

theorem total_daily_salary : num_managers * manager_salary + num_clerks * clerk_salary = 16 := by
    sorry

end total_daily_salary_l432_432200


namespace time_to_cross_street_l432_432006

def distance : ℕ := 1080    -- distance in meters
def speed_kmh : ℝ := 5.4    -- speed in km/h
def speed_mmin : ℝ := (speed_kmh * 1000) / 60    -- converted speed in m/min
def expected_time : ℝ := 12    -- expected time in minutes

theorem time_to_cross_street :
  (distance / speed_mmin) = expected_time := by 
  sorry

end time_to_cross_street_l432_432006


namespace remainder_twice_by_x_minus_2_is_5120_l432_432425

variable (x : ℝ)

def p (x : ℝ) : ℝ := x^10

theorem remainder_twice_by_x_minus_2_is_5120 :
  let q1 (x : ℝ) := x^9 + 2*x^8 + 4*x^7 + 8*x^6 + 16*x^5 + 32*x^4 + 64*x^3 + 128*x^2 + 256*x + 512 in
  (q1 2) = 5120 := 
by
  sorry

end remainder_twice_by_x_minus_2_is_5120_l432_432425


namespace find_smallest_integer_l432_432102

theorem find_smallest_integer (r : ℕ) (hr : r > 1) :
  ∃ (h : ℕ), h > 1 ∧ 
  (∀ (partition : fin (h + 1) → fin r), 
   ∃ (a : ℕ) (x y : ℕ), 
     a ≥ 0 
     ∧ 1 ≤ x ∧ x ≤ y 
     ∧ partition ⟨a + x, sorry⟩ = partition ⟨a + y, sorry⟩ 
     ∧ partition ⟨a + x, sorry⟩ = partition ⟨a + x + y, sorry⟩) ∧ h = 2 * r :=
begin
  sorry
end

end find_smallest_integer_l432_432102


namespace positive_difference_l432_432331

def sum_of_squares (n : ℕ) : ℕ := 
  (List.range n).map (λ x => (x+1) * (x+1)).sum

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_of_primes_between (a b : ℕ) : ℕ := 
  ((List.range (b - a - 1)).map (λ x => x + a + 1)).filter is_prime.sum

theorem positive_difference :
  sum_of_squares 10 - sum_of_primes_between 4 49 = 57 :=
by
  sorry

end positive_difference_l432_432331


namespace find_xy_l432_432185

variable (x y : ℝ)

theorem find_xy (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end find_xy_l432_432185


namespace Problem_l432_432982

theorem Problem (x y : ℝ) (h1 : 2*x + 2*y = 10) (h2 : x*y = -15) : 4*(x^2) + 4*(y^2) = 220 := 
by
  sorry

end Problem_l432_432982


namespace rectangular_to_cylindrical_conversion_l432_432430

def convert_rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (r, θ, z)

theorem rectangular_to_cylindrical_conversion :
  convert_rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end rectangular_to_cylindrical_conversion_l432_432430


namespace four_people_fill_pool_together_in_12_minutes_l432_432608

def combined_pool_time (j s t e : ℕ) : ℕ := 
  1 / ((1 / j) + (1 / s) + (1 / t) + (1 / e))

theorem four_people_fill_pool_together_in_12_minutes : 
  ∀ (j s t e : ℕ), j = 30 → s = 45 → t = 90 → e = 60 → combined_pool_time j s t e = 12 := 
by 
  intros j s t e h_j h_s h_t h_e
  unfold combined_pool_time
  rw [h_j, h_s, h_t, h_e]
  have r1 : 1 / 30 = 1 / 30 := rfl
  have r2 : 1 / 45 = 1 / 45 := rfl
  have r3 : 1 / 90 = 1 / 90 := rfl
  have r4 : 1 / 60 = 1 / 60 := rfl
  rw [r1, r2, r3, r4]
  norm_num
  sorry

end four_people_fill_pool_together_in_12_minutes_l432_432608


namespace problem_solution_l432_432163

variables (α β : Plane) (l m : Line)
  (h1 : α ≠ β)
  (h2 : ¬ Intersect l m)
  (h3 : Perpendicular l α)
  (h4 : Perpendicular m β)
  (h5 : Perpendicular α β)

theorem problem_solution (h1 : α ≠ β) (h2 : ¬ Intersect l m) (h3 : Perpendicular l α) (h4 : Perpendicular m β) (h5 : Perpendicular α β) :
  Perpendicular l m :=
sorry

end problem_solution_l432_432163


namespace otimes_example_l432_432061

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432061


namespace dan_dimes_correct_l432_432823

-- Definitions
def value_of_dime : ℝ := 0.10
def barry_total_dimes_value : ℝ := 10.00

def barry_dimes : ℕ := (barry_total_dimes_value / value_of_dime).to_nat
def dan_initial_dimes : ℕ := barry_dimes / 2
def dimes_found_by_dan : ℕ := 2

def dan_total_dimes : ℕ := dan_initial_dimes + dimes_found_by_dan

-- Proof statement
theorem dan_dimes_correct :
  dan_total_dimes = 52 := 
sorry

end dan_dimes_correct_l432_432823


namespace maximum_value_l432_432238

theorem maximum_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)  ≤ 1 :=
sorry

end maximum_value_l432_432238


namespace knights_count_l432_432623

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432623


namespace region_area_l432_432329

theorem region_area (x y: ℝ) :
  (x - 5)^2 + (y - 6)^2 = 21 → y = x - 5 → region_above_area = (21 * real.pi) / 2 := 
by sorry

end region_area_l432_432329


namespace m_squared_plus_reciprocal_squared_l432_432110

theorem m_squared_plus_reciprocal_squared (m : ℝ) (h : m^2 - 2 * m - 1 = 0) : m^2 + 1 / m^2 = 6 :=
by
  sorry

end m_squared_plus_reciprocal_squared_l432_432110


namespace tan_alpha_sq_two_trigonometric_identity_l432_432087

-- Problem 1
theorem tan_alpha_sq_two (α : ℝ) (h : Real.tan α = Real.sqrt 2) :
  1 + Real.sin (2 * α) + (Real.cos α) ^ 2 = (4 + Real.sqrt 2) / 3 :=
by sorry

-- Problem 2
theorem trigonometric_identity : 
  (2 * Real.sin (50 * Real.pi / 180) + Real.sin (80 * Real.pi / 180) *
    (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180))) / 
  Real.sqrt (1 + Real.sin (100 * Real.pi / 180)) = 2 :=
by sorry

end tan_alpha_sq_two_trigonometric_identity_l432_432087


namespace fractional_part_lawn_remainder_l432_432680

def mary_mowing_time := 3 -- Mary can mow the lawn in 3 hours
def tom_mowing_time := 6  -- Tom can mow the lawn in 6 hours
def mary_working_hours := 1 -- Mary works for 1 hour alone

theorem fractional_part_lawn_remainder : 
  (1 - mary_working_hours / mary_mowing_time) = 2 / 3 := 
by
  sorry

end fractional_part_lawn_remainder_l432_432680


namespace K3_3_non_planar_l432_432327

theorem K3_3_non_planar : ¬ (planar (completeBipartiteGraph 3 3)) :=
sorry

end K3_3_non_planar_l432_432327


namespace ratio_of_product_of_composites_l432_432487

theorem ratio_of_product_of_composites :
  let A := [4, 6, 8, 9, 10, 12]
  let B := [14, 15, 16, 18, 20, 21]
  (A.foldl (λ x y => x * y) 1) / (B.foldl (λ x y => x * y) 1) = 1 / 49 :=
by
  -- Proof will be filled here
  sorry

end ratio_of_product_of_composites_l432_432487


namespace second_horse_revolutions_l432_432383

-- Define the parameters and conditions:
def r₁ : ℝ := 30  -- Distance of the first horse from the center
def revolutions₁ : ℕ := 15  -- Number of revolutions by the first horse
def r₂ : ℝ := 5  -- Distance of the second horse from the center

-- Define the statement to prove:
theorem second_horse_revolutions : r₂ * (↑revolutions₁ * r₁⁻¹) * (↑revolutions₁) = 90 := 
by sorry

end second_horse_revolutions_l432_432383


namespace min_disks_needed_l432_432274

theorem min_disks_needed (f1 f2 f3 : ℕ) (d : ℕ) (space_per_disk : ℝ) (f1_size f2_size f3_size : ℝ)
  (h_f1 : f1 = 4) (h_f2 : f2 = 15) (h_f3 : f3 = 14) (h_d : d = 33)
  (h_space_per_disk : space_per_disk = 1.44)
  (h_f1_size : f1_size = 1) (h_f2_size : f2_size = 0.6) (h_f3_size : f3_size = 0.5)
  (h_files : d = f1 + f2 + f3) :
  ∃ n, n = 15 ∧ n = infimum {x : ℕ | ∀ (f₁₁ f₁₂ f₂₁ f₂₂ f₃₁ f₃₂ : ℕ) (hf₁₁ : f₁₁ ≤ f1) 
                                (hf₁₂ : (f1 - f₁₁) ≥ 0) (hf₂₁ : f₂₁ ≤ f2) (hf₂₂ : (f2 - f₂₁) ≥ 0) 
                                (hf₃₁ : f₃₁ ≤ f3) (hf₃₂ : (f3 - f₃₁) ≥ 0) 
                                (hf₁ : f₁₁ + (f1 - f₁₁) = f1) (hf₂ : f₂₁ + (f2 - f₂₁) = f2) 
                                (hf₃ : f₃₁ + (f3 - f₃₁) = f3),
                                f₁₁ * f1_size + f2_size * hf₂₁ + f3_size * hf₃₁ <= space_per_disk} :=
begin
  sorry
end

end min_disks_needed_l432_432274


namespace triangle_angles_l432_432587

noncomputable def degrees_to_radians (d : ℝ) : ℝ :=
  d * (Real.pi / 180)

theorem triangle_angles 
  (A B C : Type) [triangle A B C]
  (C1 : Type) [foot_of_perpendicular C1 C (line A B)]
  (mc : ℝ)
  (h_mc : dist C C1 = mc)
  (right_angle : ∠ ABC = Real.pi / 2)
  (two_angles_equal : ∃ α, ∠ BCA = α ∧ ∠ BAC = α)
  :
  ∃ α β γ, 
  (α = degrees_to_radians 19.7667 ∧ β = degrees_to_radians 70.2333 ∧ γ = degrees_to_radians 90) ∨
  (α = degrees_to_radians 90 ∧ β = degrees_to_radians 23.2167 ∧ γ = degrees_to_radians 66.7833) ∨
  (α = degrees_to_radians 23.2167 ∧ β = degrees_to_radians 23.2167 ∧ γ = degrees_to_radians 133.5667)
  :=
  sorry 

end triangle_angles_l432_432587


namespace chess_tournament_ordering_l432_432227

theorem chess_tournament_ordering (N M : ℕ) (hNM : N > M) (hM1 : M > 1) 
  (P : fin N → fin N → Prop) -- a relation indicating the player at index a beats the player at index b
  (h_played_once : ∀ a b : fin N, a ≠ b → P a b ∨ P b a)
  (h_no_draws : ∀ a b : fin N, ¬(P a b ∧ P b a)) 
  (h_sequence_property : ∀ (players : fin (M + 1)), 
    (∀ i : fin M, P (players i) (players ⟨i + 1, sorry⟩))  → P (players 0) (players M)) :
  ∃ labeling : fin N → fin N, 
  ∀ (a b : fin N), (labeling a) ≥ (labeling b + M - 1) → P a b :=
sorry

end chess_tournament_ordering_l432_432227


namespace evaluate_root_power_l432_432863

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l432_432863


namespace solve_abs_of_linear_system_l432_432647

theorem solve_abs_of_linear_system (x y : ℝ) 
  (h1 : ⌊x⌋ + fract y = 3.7) 
  (h2 : fract x + ⌊y⌋ = 4.2) : 
  |x - 2 * y| = 6.2 := by
  sorry

end solve_abs_of_linear_system_l432_432647


namespace solutions_characterization_l432_432883

noncomputable def find_solutions (n : ℕ) : Set (Fin n → ℝ) :=
  {x | (∀ i, x i ≠ 0) ∧ (∃ j, x j = -1)}

theorem solutions_characterization (n : ℕ) (x : Fin n → ℝ) :
  (1 + ∑ i in Finset.range n, 
  (if i = 0 then 1 else ∏ j in Finset.range i, (x j + 1) / (∏ j in Finset.range (i+1), x j)) = 0) ↔ x ∈ find_solutions n := 
  by 
    sorry

end solutions_characterization_l432_432883


namespace triangle_perimeter_l432_432316

-- Definitions derived from the conditions
def diameter_circle := 18 -- the diameter of the circle is 18 cm
def hypotenuse_triangle := diameter_circle -- the hypotenuse is the diameter of the circle

def x := hypotenuse_triangle / 5 -- since the hypotenuse corresponds to 5x in a 3-4-5 triangle
def side1 := 3 * x -- one of the legs corresponding to 3x
def side2 := 4 * x -- the other leg corresponding to 4x

-- Proving the perimeter
def perimeter := side1 + side2 + hypotenuse_triangle -- perimeter of the triangle

theorem triangle_perimeter : perimeter = 43.2 :=
by 
  sorry

end triangle_perimeter_l432_432316


namespace common_sum_is_zero_l432_432427

-- Define the range of integers from -15 to 15.
def integers_from_neg15_to_15 : List ℤ := List.range' (-15) 31

-- Define the conditions of the 5x5 matrix using the given integers.
def is_magic_square (m : List (List ℤ)) : Prop :=
  m.length = 5 ∧ -- 5 rows
  (∀ row, row ∈ m → row.length = 5) ∧ -- each row has 5 elements
  (∀ i, 0 ≤ i ∧ i < 5 → List.sum (List.map (λ row, row.nth i) m) = 0) ∧ -- sum of each column
  (List.sum (List.head m) = 0) ∧ -- sum of first row
  (List.sum (List.init m.last) = 0) ∧ -- sum of last row
  (List.sum (List.build (λ i, (m.nth i).getOrElse 0 (m.nth i).length)) = 0) -- sum of first diagonal

-- Prove that the common sum is 0.
theorem common_sum_is_zero : 
  ∃ m, is_magic_square m ∧ ∀ i, 0 ≤ i ∧ i < 5 → List.sum (List.nth m i).getOrElse 0 = 0 := 
by
  sorry

end common_sum_is_zero_l432_432427


namespace range_of_a_for_distinct_zeros_l432_432959

-- Define the necessary functions and problem conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x
def g (x : ℝ) : ℝ := Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x
def k (x : ℝ) : ℝ := (Real.log x + x) / (x^2)

-- State the problem as a theorem
theorem range_of_a_for_distinct_zeros (a : ℝ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h a x1 = 0 ∧ h a x2 = 0) ↔ (0 < a ∧ a < 1) :=
sorry

end range_of_a_for_distinct_zeros_l432_432959


namespace math_problem_l432_432693

def cond1 (R r a b c p : ℝ) : Prop := R * r = (a * b * c) / (4 * p)
def cond2 (a b c p : ℝ) : Prop := a * b * c ≤ 8 * p^3
def cond3 (a b c p : ℝ) : Prop := p^2 ≤ (3 * (a^2 + b^2 + c^2)) / 4
def cond4 (m_a m_b m_c R : ℝ) : Prop := m_a^2 + m_b^2 + m_c^2 ≤ (27 * R^2) / 4

theorem math_problem (R r a b c p m_a m_b m_c : ℝ) 
  (h1 : cond1 R r a b c p)
  (h2 : cond2 a b c p)
  (h3 : cond3 a b c p)
  (h4 : cond4 m_a m_b m_c R) : 
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ (27 * R^2) / 2 :=
by 
  sorry

end math_problem_l432_432693


namespace iterate_f_95_l432_432157

noncomputable def f (x : ℝ) : ℝ := 1 / Real.cbrt (1 - x^3)

def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  Nat.recOn n x (λ _ y => f y)

theorem iterate_f_95 (x : ℝ) : iterate_f 95 x = Real.cbrt (1 - 1 / x^3) :=
by
  sorry

#check iterate_f_95 19

end iterate_f_95_l432_432157


namespace convert_square_decimeters_to_square_centimeters_convert_months_to_years_convert_square_decimeters_to_square_meters_convert_hours_to_12_hour_format_l432_432840

theorem convert_square_decimeters_to_square_centimeters 
  (d: ℕ) (h: d = 2) : d * 100 = 200 := by
  rw [h]
  sorry

theorem convert_months_to_years 
  (m: ℕ) (h: m = 24) : m / 12 = 2 := by
  rw [h]
  sorry
  
theorem convert_square_decimeters_to_square_meters 
  (d: ℕ) (h: d = 3000) : d / 100 = 30 := by
  rw [h]
  sorry
  
theorem convert_hours_to_12_hour_format 
  (h: ℕ) (h1: h = 15) : h - 12 = 3 := by
  rw [h1]
  sorry

end convert_square_decimeters_to_square_centimeters_convert_months_to_years_convert_square_decimeters_to_square_meters_convert_hours_to_12_hour_format_l432_432840


namespace f_multiplicative_f_prime_l432_432657

noncomputable def f (a : ℚ) : ℚ :=
  sorry

theorem f_multiplicative (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : 
  f (a * b) = f a + f b := 
  sorry

theorem f_prime (p : ℚ) (hp : Nat.Prime p.numerator) (hp_denom : p.denominator = 1) : 
  f p = p := 
  sorry

example : f (5 / 12) < 0 ∧ f (8 / 15) < 0 := 
by 
  have h1 : f (5 : ℚ) = 5 := sorry
  have h2 : f (2 : ℚ) = 2 := sorry
  have h3 : f (3 : ℚ) = 3 := sorry
  have h4 : f (12 : ℚ) = f (2 * 2 * 3) := sorry
  have h5 : f (8 / 15) = f (8) - f (15) := sorry
  have h6 : f (8) = f (2 * 2 * 2) := sorry
  have h7 : f (15) = f (3 * 5) := sorry

  sorry

end f_multiplicative_f_prime_l432_432657


namespace part1_part2_l432_432107

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x ∈ Ioo (-Real.exp 1) (-1), f x = Real.exp x - a * x) 
  (h2 : ∀ x ∈ Ioo (-Real.exp 1) (-1), f' x < 0) : a > 1 / Real.exp 1 := 
sorry

theorem part2 (a : ℝ) (f F : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.exp x - a * x)
  (h2 : ∀ x, F x = f x - (Real.exp x - 2 * a * x + 2 * Real.log x + a))
  (h3 : ¬ ∃ x ∈ Ioo 0 (1/2 : ℝ), F x = 0) : a ≤ 4 * Real.log 2 :=
sorry

end part1_part2_l432_432107


namespace emily_beads_l432_432856

-- Definitions of the conditions as per step a)
def beads_per_necklace : ℕ := 8
def necklaces : ℕ := 2

-- Theorem statement to prove the equivalent math problem
theorem emily_beads : beads_per_necklace * necklaces = 16 :=
by
  sorry

end emily_beads_l432_432856


namespace find_a1_a7_l432_432144

-- Definitions based on the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a_3_5_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = -6

def a_2_6_condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 6 = 8

-- The theorem we need to prove
theorem find_a1_a7 (a : ℕ → ℝ) (ha : is_geometric_sequence a) (h35 : a_3_5_condition a) (h26 : a_2_6_condition a) :
  a 1 + a 7 = -9 :=
sorry

end find_a1_a7_l432_432144


namespace set_union_inter_eq_l432_432967

open Set

-- Conditions: Definitions of sets M, N, and P
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3, 4, 5}

-- Claim: The result of (M ∩ N) ∪ P equals {1, 2, 3, 4, 5}
theorem set_union_inter_eq :
  (M ∩ N ∪ P) = {1, 2, 3, 4, 5} := 
by
  sorry

end set_union_inter_eq_l432_432967


namespace knights_count_l432_432636

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432636


namespace part_one_part_two_l432_432675

noncomputable def f (a x : ℝ) : ℝ :=
  |x + (1 / a)| + |x - a + 1|

theorem part_one (a : ℝ) (h : a > 0) (x : ℝ) : f a x ≥ 1 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : f a 3 < 11 / 2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part_one_part_two_l432_432675


namespace nancy_picked_l432_432406

variable (total_picked : ℕ) (alyssa_picked : ℕ)

-- Assuming the conditions given in the problem
def conditions := total_picked = 59 ∧ alyssa_picked = 42

-- Proving that Nancy picked 17 pears
theorem nancy_picked : conditions total_picked alyssa_picked → total_picked - alyssa_picked = 17 := by
  sorry

end nancy_picked_l432_432406


namespace negation_equiv_l432_432297

-- Given problem conditions
def exists_real_x_lt_0 : Prop := ∃ x : ℝ, x^2 + 1 < 0

-- Mathematically equivalent proof problem statement
theorem negation_equiv :
  ¬exists_real_x_lt_0 ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l432_432297


namespace find_positive_k_l432_432881

theorem find_positive_k (k a b c : ℕ) (h_pos : 0 < k) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  |(a - b)^3 + (b - c)^3 + (c - a)^3| = 3 * 2^k ↔ ∃ n : ℕ, k = 3 * n + 1 :=
sorry

end find_positive_k_l432_432881


namespace sqrt_7_irrational_l432_432765

theorem sqrt_7_irrational : irrational (real.sqrt 7) := 
sorry

end sqrt_7_irrational_l432_432765


namespace three_digit_odd_sum_count_l432_432033

def countOddSumDigits : Nat :=
  -- Count of three-digit numbers with an odd sum formed by (1, 2, 3, 4, 5)
  24

theorem three_digit_odd_sum_count :
  -- Guarantees that the count of three-digit numbers meeting the criteria is 24
  ∃ n : Nat, n = countOddSumDigits :=
by
  use 24
  sorry

end three_digit_odd_sum_count_l432_432033


namespace knights_count_l432_432620

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432620


namespace total_revenue_from_gerbil_sales_l432_432389

-- The conditions translated into Lean
constant initial_gerbils : ℕ := 450
constant sold_percentage : ℝ := 0.35
constant original_price : ℝ := 12.0
constant discount : ℝ := 0.20

-- The expected revenue from sales
theorem total_revenue_from_gerbil_sales :
  let sold_gerbils := (sold_percentage * initial_gerbils).floor
  let remaining_gerbils := initial_gerbils - sold_gerbils
  let revenue_original_price := sold_gerbils * original_price
  let discounted_price := original_price * (1 - discount)
  let revenue_discounted_price := remaining_gerbils * discounted_price
  let total_revenue := revenue_original_price + revenue_discounted_price
  total_revenue = 4696.80 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end total_revenue_from_gerbil_sales_l432_432389


namespace julios_hours_fishing_l432_432611

-- Define the conditions as constants
def julios_catch_rate : ℕ := 7
def julios_loss : ℕ := 15
def julios_final_fish : ℕ := 48

-- Define the proof of the problem
theorem julios_hours_fishing : ∃ h : ℕ, julios_catch_rate * h - julios_loss = julios_final_fish ∧ h = 9 :=
by {
  use 9,
  split,
  {
    -- Proving that 7h - 15 = 48 when h = 9
    norm_num,
  },
  {
    -- Proving h = 9 is indeed the number of hours
    refl,
  }
}

end julios_hours_fishing_l432_432611


namespace max_product_distance_l432_432910

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432910


namespace angle_BDC_30_l432_432948

/-- 
Proof Problem: Prove that the measure of angle BDC is 30 degrees given the conditions defined below.
-/
theorem angle_BDC_30
  (DB DC Γ : Type) -- Points B, C and circle Γ (types could be more specifically defined for actual geometry)
  (tangent_DB : tangent_to_circle_at DB Γ B) -- DB is tangent to circle Γ at point B
  (tangent_DC : tangent_to_circle_at DC Γ C) -- DC is tangent to circle Γ at point C
  (angle_ABC : ℝ) (h1 : angle_ABC = 62) -- ∠ABC = 62°
  (angle_ACB : ℝ) (h2 : angle_ACB = 43) -- ∠ACB = 43°
  : angle_BDC = 30 := 
sorry

end angle_BDC_30_l432_432948


namespace total_strawberries_weight_is_72_l432_432245

-- Define the weights
def Marco_strawberries_weight := 19
def dad_strawberries_weight := Marco_strawberries_weight + 34 

-- The total weight of their strawberries
def total_strawberries_weight := Marco_strawberries_weight + dad_strawberries_weight

-- Prove that the total weight is 72 pounds
theorem total_strawberries_weight_is_72 : total_strawberries_weight = 72 := by
  sorry

end total_strawberries_weight_is_72_l432_432245


namespace number_of_arrangements_l432_432278

theorem number_of_arrangements (n : ℕ) (h_n : n = 6) : 
  ∃ total : ℕ, total = 90 := 
sorry

end number_of_arrangements_l432_432278


namespace classify_batch_correct_l432_432196

def count_good (s : List String) : Nat :=
  s.count (λ x => x.toLower == "good")

def classify_batch (s : List String) : String :=
  let good_count := count_good s
  if good_count == 0 then "Unsuitable Material"
  else if good_count > 2 then "First-Class Batch"
  else "Second-Class Batch"

theorem classify_batch_correct (s : List String) :
  (count_good s = 0 → classify_batch s = "Unsuitable Material") ∧
  (count_good s > 2 → classify_batch s = "First-Class Batch") ∧
  (1 ≤ count_good s ∧ count_good s ≤ 2 → classify_batch s = "Second-Class Batch") :=
by
  sorry

end classify_batch_correct_l432_432196


namespace rectangular_to_cylindrical_conversion_l432_432431

def convert_rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (r, θ, z)

theorem rectangular_to_cylindrical_conversion :
  convert_rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end rectangular_to_cylindrical_conversion_l432_432431


namespace i_exponent_2016_l432_432450

noncomputable def i : ℂ := complex.I

theorem i_exponent_2016 : i^2016 = 1 :=
by {
  have h1 : i^2 = -1 := complex.I_sq,
  have h2 : i^4 = 1 := (by rw [←pow_two, h1, neg_one_sq]),
  calc
    i^2016 = (i^4)^504 : by rw [pow_mul, div_eq_mul_inv]
         ... = 1^504   : by rw h2
         ... = 1       : one_pow
}

end i_exponent_2016_l432_432450


namespace infer_proportion_l432_432993

theorem infer_proportion (t : ℝ) : 
  let x := [12, 17, 22, 27, 32]
  let y := [10, 18, 20, 30, t]
  let x_mean := (12 + 17 + 22 + 27) / 4
  let y_mean := (10 + 18 + 20 + 30) / 4
  let k := (19.5 + 4.68) / 19.5
  let regression_eq := λ x, k * x - 4.68
  y[4] = regression_eq 32 :=
by
  let x := [12, 17, 22, 27, 32]
  let y := [10, 18, 20, 30, t]
  let x_mean := (12 + 17 + 22 + 27) / 4
  let y_mean := (10 + 18 + 20 + 30) / 4
  let k := (19.5 + 4.68) / 19.5
  let regression_eq := λ x, k * x - 4.68
  have h : regression_eq 32 = 30.84 := sorry
  exact h

end infer_proportion_l432_432993


namespace dwarf_heights_l432_432687

-- Define the heights of the dwarfs.
variables (F J M : ℕ)

-- Given conditions
def condition1 : Prop := J + F = M
def condition2 : Prop := M + F = J + 34
def condition3 : Prop := M + J = F + 72

-- Proof statement
theorem dwarf_heights
  (h1 : condition1 F J M)
  (h2 : condition2 F J M)
  (h3 : condition3 F J M) :
  F = 17 ∧ J = 36 ∧ M = 53 :=
by
  sorry

end dwarf_heights_l432_432687


namespace actual_distance_traveled_l432_432351

theorem actual_distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 15) : D = 40 := 
sorry

end actual_distance_traveled_l432_432351


namespace smallest_prime_sum_l432_432890

def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p > 1 ∧ p ∣ n ∧ ∀ m, m > 1 ∧ m < n → m ∣ n → m = p

def digits_used_once (n1 n2 n3 : ℕ) : Prop :=
  let digits := (n1.digits 10) ++ (n2.digits 10) ++ (n3.digits 10) in
  list.nodup digits ∧ list.sort (≤) digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem smallest_prime_sum : 
  ∃ (n1 n2 n3 : ℕ), is_prime n1 ∧ is_prime n2 ∧ is_prime n3 ∧
  n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
  digits_used_once n1 n2 n3 ∧
  n1 + n2 + n3 = 999 :=
sorry

end smallest_prime_sum_l432_432890


namespace max_min_f_cos_2x0_omega_range_l432_432550

-- Definitions based on the problem conditions.
def vec_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x, -1)
def f (x : ℝ) : ℝ := vec_a x.1 * vec_b x.1 + vec_a x.2 * vec_b x.2

-- Proof problem 1: Proving the maximum and minimum values of f(x) in the given interval.
theorem max_min_f (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ (Real.pi / 4)) : 
  (f (h.1) == 2) ∧ (f (h.2) == 1) := sorry

-- Proof problem 2: Given f(x0) = 6/5, derive cos(2x0) for the given range of x0.
theorem cos_2x0 (x0 : ℝ) (hx0 : f x0 = 6 / 5) (interval : x0 ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) : 
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 := sorry

-- Proof problem 3: Proving the range of omega for the monotonicity condition.
theorem omega_range (ω : ℝ) (mono_inc : (∀ x : ℝ, (Real.pi / 3) ≤ x ∧ x ≤ (2 * Real.pi / 3) → f (ω * x) = f (ω * x))) : 
  0 < ω ∧ ω ≤ 1 / 4 := sorry

end max_min_f_cos_2x0_omega_range_l432_432550


namespace round_trip_average_mileage_l432_432021

theorem round_trip_average_mileage 
  (d1 d2 : ℝ) (m1 m2 : ℝ)
  (h1 : d1 = 150) (h2 : d2 = 150)
  (h3 : m1 = 40) (h4 : m2 = 25) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 30.77 :=
by
  sorry

end round_trip_average_mileage_l432_432021


namespace valid_l_values_l432_432507

theorem valid_l_values
  (m l : ℕ)
  (h1 : 1 ≤ l)
  (h2 : l ≤ m - 1)
  (h3 : even (m * l)) :
  true := sorry

end valid_l_values_l432_432507


namespace triangle_is_acute_triangle_does_not_exist_l432_432772

-- Definitions for the heights given in the conditions
def triangle_heights_a := (h_a h_b h_c : ℕ) (ha : h_a = 4) (hb : h_b = 5) (hc : h_c = 6) :=
  true

def triangle_heights_b := (h_a h_b h_c : ℕ) (ha : h_a = 2) (hb : h_b = 3) (hc : h_c = 6) :=
  true

-- Problem (a): Prove that a triangle with given heights is acute-angled
theorem triangle_is_acute (h_a h_b h_c : ℕ) (ha : h_a = 4) (hb : h_b = 5) (hc : h_c = 6) :
  triangle_heights_a h_a h_b h_c ha hb hc → true := 
begin
  sorry
end

-- Problem (b): Prove that a triangle with given heights does not exist
theorem triangle_does_not_exist (h_a h_b h_c : ℕ) (ha : h_a = 2) (hb : h_b = 3) (hc : h_c = 6) :
  ¬(triangle_heights_b h_a h_b h_c ha hb hc) :=
begin
  sorry
end

end triangle_is_acute_triangle_does_not_exist_l432_432772


namespace minimum_ratio_ge_5_plus_sqrt3_l432_432103

noncomputable def minimum_ratio (P : Fin 4 → EuclideanGeometry.Point ℝ) : ℝ :=
  let distances := (Finset.univ.product Finset.univ).filter (λ p, p.1 < p.2).map (λ p, dist (P p.1) (P p.2))
  (distances.sum) / distances.min' sorry

theorem minimum_ratio_ge_5_plus_sqrt3 (P : Fin 4 → EuclideanGeometry.Point ℝ) :
  minimum_ratio P ≥ 5 + Real.sqrt 3 :=
sorry

end minimum_ratio_ge_5_plus_sqrt3_l432_432103


namespace geometric_sequence_sum_l432_432931

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 3)
  (h_sum : a 1 + a 3 + a 5 = 21) : 
  a 3 + a 5 + a 7 = 42 :=
sorry

end geometric_sequence_sum_l432_432931


namespace ball_bounce_height_l432_432366

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (hₖ : ℕ → ℝ) :
  h₀ = 500 ∧ r = 0.6 ∧ (∀ k, hₖ k = h₀ * r^k) → 
  ∃ k, hₖ k < 3 ∧ k ≥ 22 := 
by
  sorry

end ball_bounce_height_l432_432366


namespace otimes_example_l432_432066

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432066


namespace shaded_area_in_octagon_l432_432584

theorem shaded_area_in_octagon (s r : ℝ) (h_s : s = 4) (h_r : r = s / 2) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_semicircles := 8 * (π * r^2 / 2)
  area_octagon - area_semicircles = 32 * (1 + Real.sqrt 2) - 16 * π := by
  sorry

end shaded_area_in_octagon_l432_432584


namespace length_PQ_l432_432520

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

def P : Point3D := { x := 3, y := 4, z := 5 }

def Q : Point3D := { x := 3, y := 4, z := 0 }

theorem length_PQ : distance P Q = 5 :=
by
  sorry

end length_PQ_l432_432520


namespace nabla_four_seven_l432_432828

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_four_seven : nabla 4 7 = 11 / 29 :=
by
  sorry

end nabla_four_seven_l432_432828


namespace area_of_triangle_PQR_l432_432588

open Real

noncomputable def area_triangle_PQR : ℝ :=
  let PQ := 9
  let PR := 20
  let PS := 13
  let semi_perimeter := (PQ + PR + 26) / 2
  sqrt ((semi_perimeter * (semi_perimeter - PQ) * (semi_perimeter - PR) * (semi_perimeter - 26)))

theorem area_of_triangle_PQR (PQ PR PS : ℝ) (hPQ : PQ = 9) (hPR : PR = 20) (hPS : PS = 13) :
  area_triangle_PQR = 75.63 :=
by
  rw [hPQ, hPR, hPS]
  -- Filled in the computation details for Heron's formula result here.
  sorry

end area_of_triangle_PQR_l432_432588


namespace manhattan_dist_eq_cosine_dist_eq_tan_product_eq_l432_432462

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3/5, 4/5)

-- Manhattan distance definition
def manhattan_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

-- Cosine similarity and cosine distance
def cosine_similarity (p q : ℝ × ℝ) : ℝ :=
  (p.1 / real.sqrt(p.1^2 + p.2^2)) * (q.1 / real.sqrt(q.1^2 + q.2^2)) +
  (p.2 / real.sqrt(p.1^2 + p.2^2)) * (q.2 / real.sqrt(q.1^2 + q.2^2))

def cosine_distance (p q : ℝ × ℝ) : ℝ :=
  1 - cosine_similarity(p, q)

-- Definition of points M, N, Q
def M (α : ℝ) : ℝ × ℝ := (real.sin α, real.cos α)
def N (β : ℝ) : ℝ × ℝ := (real.sin β, real.cos β)
def Q (β : ℝ) : ℝ × ℝ := (real.sin β, -real.cos β)

-- Given conditions
variable (α β : ℝ)
axiom cos_M_N : cosine_similarity (M α) (N β) = 1/5
axiom cos_M_Q : cosine_similarity (M α) (Q β) = 2/5

-- Proof problems
theorem manhattan_dist_eq : manhattan_distance A B = 14 / 5 :=
sorry

theorem cosine_dist_eq : cosine_distance A B = 1 - real.sqrt 5 / 5 :=
sorry

theorem tan_product_eq : real.tan α * real.tan β = -3 :=
sorry

end manhattan_dist_eq_cosine_dist_eq_tan_product_eq_l432_432462


namespace regular_tetrahedron_properties_l432_432415

/-- 
Given a regular tetrahedron, prove the following properties:
1. All edges are equal.
2. The angle between any two edges at the same vertex is equal.
3. All faces are congruent equilateral triangles.
4. The dihedral angle between any two adjacent faces is equal.
-/
theorem regular_tetrahedron_properties :
  ∀ (T : Type) [regular_tetrahedron T],
    (∀ (e1 e2 : edge T), e1 ≠ e2 → length e1 = length e2) ∧
    (∀ (v : vertex T), ∀ (e1 e2 : edge T), e1 ≠ e2 → angle_between_edges v e1 e2 = angle v) ∧
    (∀ (f1 f2 : face T), f1 ≠ f2 → congruent f1 f2) ∧
    (∀ (f1 f2 : face T), adjacent f1 f2 → dihedral_angle f1 f2 = dihedral_angle T) :=
sorry

end regular_tetrahedron_properties_l432_432415


namespace janet_gas_usage_l432_432600

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20
def total_driving_distance : ℕ := (2 * distance_dermatologist) + (2 * distance_gynecologist)
def gas_used : ℝ := total_driving_distance / car_efficiency

theorem janet_gas_usage : gas_used = 8 := by
  sorry

end janet_gas_usage_l432_432600


namespace num_distinct_possible_values_l432_432059

noncomputable def odd_integers_less_than_twenty : list ℕ := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

-- Define a function to compute the distinct possible values of the given expression based on the conditions.
def distinct_possible_values : ℕ :=
(list.finrange 10).map (λ i, odd_integers_less_than_twenty.nth i).bind (λ a, 
(list.finrange 10).map (λ j, odd_integers_less_than_twenty.nth j).bind (λ b,
(list.finrange 10).map (λ k, odd_integers_less_than_twenty.nth k).map (λ c,
match (a, b, c) with
| (some a', some b', some c') => (a' + b' + c' + a' * b' * c')
| _ => 0
end))).to_finset.card

theorem num_distinct_possible_values : distinct_possible_values = 200 := by
  -- sorry is used to skip the proof
  sorry

end num_distinct_possible_values_l432_432059


namespace trig_identity_l432_432502

theorem trig_identity (α β γ : ℝ) 
  (h1 : cos α = tan β)
  (h2 : cos β = tan γ)
  (h3 : cos γ = tan α) :
  sin α ^ 2 = sin β ^ 2 ∧ sin β ^ 2 = sin γ ^ 2 ∧ sin γ ^ 2 = cos α ^ 4 ∧ cos α ^ 4 = cos β ^ 4 ∧ cos β ^ 4 = cos γ ^ 4 ∧ cos γ ^ 4 = 4 * sin 18 ^ 2 :=
by
  sorry

end trig_identity_l432_432502


namespace minimum_distance_AB_l432_432113

theorem minimum_distance_AB (m x y : ℝ) :
  let C := (x - 1)^2 + (y - 2)^2 = 25,
      l := mx - y - 3m + 1 = 0 in 
  (∃ A B : { p : ℝ × ℝ // C p ∧ l p }, ∀ A B, dist A B >= 4 * real.sqrt 5) := sorry

end minimum_distance_AB_l432_432113


namespace find_m_plus_n_l432_432780

-- Definitions for the given problem
variables {X Y Z P Q R O1 O2 O3 : Point}
variables (RY QZ RX PY QX PZ m n : ℝ)

-- Conditions
def triangle_XYZ : Prop := distance X Y = 20 ∧ distance Y Z = 26 ∧ distance X Z = 22
def triangle_PQR_inscribed : Prop := point_on_line P Y Z ∧ point_on_line Q X Z ∧ point_on_line R X Y
def circumcircles_centers : Prop := center O1 (circumcircle P Y Z) ∧ center O2 (circumcircle Q X Z) ∧ center O3 (circumcircle R X Y)
def equal_arcs : Prop := arc_equal R Y Q Z ∧ arc_equal R X P Y ∧ arc_equal Q X P Z
def equal_lengths : Prop := RY = QZ ∧ RX = PY ∧ QX = PZ
def side_equations : Prop := distance X Z = QZ + PZ ∧ distance Y Z = RY + PY ∧ distance X Y = PY + PZ

-- The main theorem we need to prove
theorem find_m_plus_n : 
  triangle_XYZ ∧ triangle_PQR_inscribed ∧ circumcircles_centers ∧ equal_arcs ∧ equal_lengths ∧ side_equations →
  (m = 8) ∧ (n = 1) → m + n = 9 :=
by sorry

end find_m_plus_n_l432_432780


namespace day_18_is_sunday_l432_432750

-- Define that a certain month has three Fridays on even dates
-- Let's use 1 index based to denote days of the month.
constant Month : Type

-- Define a function to determine if a given day in the month is a Friday or not
constant is_friday : ℕ → Month → Prop

-- Define even days in a month
def is_even (n : ℕ) := n % 2 = 0

-- Define the condition that there are exactly three Fridays on even dates in this month
constant three_even_fridays : Month → Prop

-- Define the three known fridays positions
axiom fridays_even_dates (m : Month) (h : three_even_fridays m) : ∃ d1 d2 d3 : ℕ, 
  is_even d1 ∧ is_friday d1 m ∧
  is_even d2 ∧ is_friday d2 m ∧
  is_even d3 ∧ is_friday d3 m ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

-- Assume there is a month in which three Fridays fall on even dates.
constant certain_month : Month
axiom certain_month_conditions : three_even_fridays certain_month

-- Now, prove that the 18th day of the month is a Sunday given the conditions.
theorem day_18_is_sunday : is_friday 2 certain_month → is_friday 16 certain_month → is_friday 30 certain_month → 
  ∀ (d : ℕ), d = 18 → d.mod(7) = 0 := sorry

end day_18_is_sunday_l432_432750


namespace correct_propositions_l432_432293

-- Define the propositions as separate properties
def proposition_1 (line : Type) (point : Type) (plane : Type) := 
  ∀ l : line, ∀ p : point, ∃! pl : plane, is_perpendicular_to l pl ∧ contains pl p

def proposition_2 (point : Type) (plane : Type) := 
  ∀ p1 p2 : point, ∀ pl : plane, (distance_to_plane p1 pl = distance_to_plane p2 pl) 
  → (is_parallel (line_through p1 p2) pl)

def proposition_3 (line : Type) (plane : Type) := 
  ∀ l1 l2 : line, (intersect l1 l2) → (intersect (projection l1 pl) (projection l2 pl))

def proposition_4 (line : Type) (plane1 plane2 : Type) := 
  ∀ l : line, is_perpendicular plane1 plane2 
  → (exists_countlessly (λ l', contains plane1 l ∧ (is_perpendicular l l')))

-- Define the main theorem
theorem correct_propositions {line point plane plane1 plane2 : Type} :
  (proposition_1 line point plane) ∧ (proposition_4 line plane1 plane2) :=
by sorry

end correct_propositions_l432_432293


namespace polygon_sides_l432_432298

theorem polygon_sides (n : ℕ) 
  (H : (n * (n - 3)) / 2 = 3 * n) : n = 9 := 
sorry

end polygon_sides_l432_432298


namespace train_ride_cost_difference_l432_432023

-- Definitions based on the conditions
def bus_ride_cost : ℝ := 1.40
def total_cost : ℝ := 9.65

-- Lemma to prove the mathematical question
theorem train_ride_cost_difference :
  ∃ T : ℝ, T + bus_ride_cost = total_cost ∧ (T - bus_ride_cost) = 6.85 :=
by
  sorry

end train_ride_cost_difference_l432_432023


namespace diff_g_eq_l432_432078

def g (n : ℤ) : ℚ := (1/6) * n * (n+1) * (n+3)

theorem diff_g_eq :
  ∀ (r : ℤ), g r - g (r - 1) = (3/2) * r^2 + (5/2) * r :=
by
  intro r
  sorry

end diff_g_eq_l432_432078


namespace bell_rings_by_geography_class_l432_432593

theorem bell_rings_by_geography_class :
  ∀ (order_of_classes : List String)
    (break : ℕ)
    (class_bell_rings : ℕ),
    order_of_classes = ["Maths", "History", "Geography", "Science", "Music"] →
    break = 15 →
    class_bell_rings = 2 →
    let classes_before_geography := ["Maths", "History"] in
    let bell_rings_up_to_geography := (class_bell_rings * classes_before_geography.length) + (classes_before_geography.length) + 1 in
    bell_rings_up_to_geography = 5
:= by
  intros order_of_classes break class_bell_rings 
  intro h1 h2 h3 
  let classes_before_geography := ["Maths", "History"]
  let bell_rings_up_to_geography := (class_bell_rings * classes_before_geography.length) + (classes_before_geography.length) + 1
  show bell_rings_up_to_geography = 5
  sorry

end bell_rings_by_geography_class_l432_432593


namespace product_is_even_l432_432700

theorem product_is_even (a b c : ℤ) : Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end product_is_even_l432_432700


namespace minimum_ceiling_height_of_soccer_field_l432_432375

noncomputable def diagonal_length (length width : ℝ) : ℝ := 
  (length^2 + width^2).sqrt

-- Defining the minimum height function
noncomputable def min_height (length width : ℝ) : ℝ :=
  (diagonal_length length width) / 4

-- Function to round up to the nearest multiple of 0.1 meters
noncomputable def round_up_to_nearest_tenth (h : ℝ) : ℝ := 
  ((h * 10).ceil) / 10

theorem minimum_ceiling_height_of_soccer_field : 
  round_up_to_nearest_tenth (min_height 90 60) = 27.1 :=
begin
  -- Proof goes here (omitted)
  sorry,
end

end minimum_ceiling_height_of_soccer_field_l432_432375


namespace fox_appropriation_l432_432670

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l432_432670


namespace correct_statements_l432_432408

-- Definitions from conditions
def probability_of_event (A : Event) : ℝ := sorry -- Assume some function to get the probability

-- Assume some frequency-related function
def stable_frequency_value (A : Event) : ℝ := sorry

-- Assume basic event mutual exclusivity
axiom basic_event_exclusivity (E1 E2 : Event) : (E1 ≠ E2) → (disjoint E1 E2)

-- Axioms for probability bounds
axiom prob_bounds (A : Event) : 0 ≤ probability_of_event A ∧ probability_of_event A ≤ 1

-- The proof statements to be demonstrated
theorem correct_statements (A : Event) :
  (probability_of_event A = stable_frequency_value A) ∧
  (∀ E1 E2, E1 ≠ E2 → disjoint E1 E2) ∧
  (¬ (0 < probability_of_event A ∧ probability_of_event A < 1)) :=
by {
  sorry -- placeholder for proof
}

end correct_statements_l432_432408


namespace mans_rate_in_still_water_l432_432771

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water_l432_432771


namespace complement_of_beta_l432_432518

variable (α β : ℝ)
variable (compl : α + β = 180)
variable (alpha_greater_beta : α > β)

theorem complement_of_beta (h : α + β = 180) (h' : α > β) : 90 - β = (1 / 2) * (α - β) :=
by
  sorry

end complement_of_beta_l432_432518


namespace equilateral_triangle_of_cyclic_B_O_G_H_C_l432_432134

noncomputable theory

open_locale euclidean_geometry

variables {A B C O G H : Point}
variables {B_O_G_H_C_cyclic : Cyclic (Set.of [B, O, G, H, C])}
variables {circumcenter_O : Circumcenter O A B C}
variables {centroid_G : Centroid G A B C}
variables {orthocenter_H : Orthocenter H A B C}
variables {acute_ABC : Acute (Triangle A B C)}

theorem equilateral_triangle_of_cyclic_B_O_G_H_C :
  acute_ABC → circumcenter_O → centroid_G → orthocenter_H → B_O_G_H_Cyclic →
  Equilateral (Triangle A B C) :=
begin
  sorry
end

end equilateral_triangle_of_cyclic_B_O_G_H_C_l432_432134


namespace probability_team_B_wins_first_l432_432285

-- We define the conditions and the problem
def team_A_wins_series (games : List Bool) : Prop :=
  games.filter id.length = 4

def team_B_wins_series (games : List Bool) : Prop :=
  games.filter (λ x, not x).length = 4

def wins_the_series (games : List Bool) : Prop :=
  team_A_wins_series games ∨ team_B_wins_series games

-- B wins 3rd game (assuming 0-based indexing, so 3rd game is at index 2)
def third_game_B (games : List Bool) : Prop :=
  games.length > 2 ∧ games.get? 2 = some false

-- Condition: B wins the 3rd game
def condition := ∀ games : List Bool,
  team_A_wins_series games →
  third_game_B games →
  (∃ games' : List Bool, team_A_wins_series games' ∧ third_game_B games' ∧ games'.head = some false ∧
   (games.length = 7 → games' == games))

-- The theorem to prove the probability
theorem probability_team_B_wins_first :
  ∃ p : ℚ, p = 1 / 5 ∧
  (∀ (games : List Bool), team_A_wins_series games → third_game_B games →
  (∑' (games' : List Bool)
    (hA : team_A_wins_series games').filter (λ x, team_A_wins_series x ∧ third_game_B x ∧ x.head = some false) ∑'  (games' : List Bool):= p)
sorry

end probability_team_B_wins_first_l432_432285


namespace correct_propositions_count_l432_432527

-- Define the propositions
def prop1 : Prop := ∀ (A B C : ℝ), (sin A = sin B) → (A = B)
def prop2 : Prop := 
  let P F1 F2 : ℝ × ℝ,
  (constant_sum_dist P F1 F2 8) → (line_segment_trajectory P F1 F2)
def prop3 : Prop := ∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)
def prop4 : ∀ x : ℝ, (x^2 - 3 * x > 0) → (x > 4)
def prop5 : Prop :=
  ∀ m : ℝ, (geometric_sequence 1 m 9) → (eccentricity_conic_section (x^2/m + y^2 = 1) (√6 / 3))

-- The theorem
theorem correct_propositions_count :
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 ∧ ¬prop5) → True :=
sorry

end correct_propositions_count_l432_432527


namespace problem_I_problem_II_problem_III_l432_432957

noncomputable def f (k a x : ℝ) : ℝ := k * a^x - a^(-x)
noncomputable def g (a x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * (a^x - a^(-x))

theorem problem_I (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∃ k : ℝ, k = 1 ∧ ∀ x, a > 1 → f k a x = a^x - a^(-x) ∧ f k a x > 0 :=
sorry

theorem problem_II (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 1 a 1 = 3 / 2) : a = 2 ∧ ∀ x ∈ [-1, 1], 1 ≤ g a x ∧ g a x ≤ 29 / 4 :=
sorry

theorem problem_III (a : ℝ) (h : a = 3) : ∀ x ∈ [1, 2], ∃ λ : ℤ, λ = 10 ∧ f 1 a (3 * x) ≥ λ * f 1 a x :=
sorry

end problem_I_problem_II_problem_III_l432_432957


namespace solve_fraction_problem_l432_432703

theorem solve_fraction_problem :
  let x := 45 / (8 - (3 / 7))
  in x = 315 / 53 := 
by
  sorry

end solve_fraction_problem_l432_432703


namespace cos_neg_23_over_4_pi_l432_432449

theorem cos_neg_23_over_4_pi :
  ∀ (x : ℝ), ∃ n : ℤ, cos x = cos (x + 2 * π * n) → cos (-23/4 * π) = sqrt 2 / 2 :=
by sorry

end cos_neg_23_over_4_pi_l432_432449


namespace ninth_term_arithmetic_sequence_l432_432719

-- Definitions based on conditions:
def first_term : ℚ := 5 / 6
def seventeenth_term : ℚ := 5 / 8

-- Here is the main statement we need to prove:
theorem ninth_term_arithmetic_sequence : (first_term + 8 * ((seventeenth_term - first_term) / 16) = 15 / 16) :=
by
  sorry

end ninth_term_arithmetic_sequence_l432_432719


namespace not_possible_three_lines_intersections_l432_432929

theorem not_possible_three_lines_intersections
  (α : Type) [ordered_field α]
  (vertex : affine_plane α) (V1 V2 : vertex.set)
  (A : α) (h1 : V1 ≠ V2) (hA : A ∈ angle vertex V1 V2) :
  ¬ ∃ (lines : fin 3 → (α → set α)), 
    (∀ (i : fin 3), A ∈ lines i ∧ lines i ≠ ∅ ∧ ∀ (side : vertex.set), 
    ∃ (point_on_side : α), point_on_side ∈ side ∧ point_on_side ∈ lines i) := 
sorry

end not_possible_three_lines_intersections_l432_432929


namespace midpoint_sum_l432_432291

theorem midpoint_sum (x1 y1 x2 y2 : ℕ) (h₁ : x1 = 4) (h₂ : y1 = 7) (h₃ : x2 = 12) (h₄ : y2 = 19) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 21 :=
by
  sorry

end midpoint_sum_l432_432291


namespace sum_of_valid_a_l432_432897

-- Defining the original problem conditions
theorem sum_of_valid_a :
  let
    x y : ℕ := sorry,
    a : ℤ := sorry,
    h : 2 * x + a * y = 16 ∧ x - 2 * y = 0 ∧ x > 0 ∧ y > 0
  in
  ∑ a in {-3, -2, 0, 4, 12}, a = 11 :=
by
  have h₁ : 2 * x + a * y = 16,
  have h₂ : x - 2 * y = 0,
  sorry

end sum_of_valid_a_l432_432897


namespace alex_initial_seat_l432_432276

-- Definitions of the movements according to the conditions.
def seats := Fin 7 -- Seven seats numbered 0 to 6 (Lean's Fin n ranges from 0 to n-1)

def move_bob (s : seats) : seats := (s + Fin.mk 3 (by nativeDecide)) % 7
def move_cara (s : seats) : seats := (s + Fin.mk (7 - 2) (by nativeDecide)) % 7 -- Moving left by 2 is equivalent to moving right by 5 (7 - 2)
def switch_dana_eve (s1 s2 : seats) : (seats × seats) := (s2, s1)
def move_fiona (s : seats) : seats := (s + Fin.mk 1 (by nativeDecide)) % 7

-- Condition: Alex returns to an end seat.
inductive end_seats : seats -> Prop
| left_end : end_seats (Fin.mk 0 (by nativeDecide))
| right_end : end_seats (Fin.mk 6 (by nativeDecide))

-- Final position of Alex must be an end seat. Prove initial position was seat 2.
theorem alex_initial_seat : ∃ s : seats, end_seats s ∧ 
  ∀ s', (move_bob s' = bob_pos) ∧
        (move_cara s' = cara_pos) ∧
        (switch_dana_eve s' = (dana_pos, eve_pos)) ∧
        (move_fiona s' = fiona_pos) →
        s' = Fin.mk 2 (by nativeDecide) :=
begin
  sorry
end

end alex_initial_seat_l432_432276


namespace min_ceiling_height_l432_432372

def length : ℝ := 90
def width : ℝ := 60
def diagonal : ℝ := real.sqrt (length^2 + width^2)
def height : ℝ := (1 / 4) * diagonal

theorem min_ceiling_height (h : ℝ) : h = 27.1 → (∃ (r : ℝ), r = h ∧ r ≥ height ∧ (∃ (n : ℝ), n = 0.1 * ⌈r / 0.1⌉₊)) :=
by
  refine ⟨_, _, _, _⟩;
  sorry

end min_ceiling_height_l432_432372


namespace minimum_ceiling_height_of_soccer_field_l432_432376

noncomputable def diagonal_length (length width : ℝ) : ℝ := 
  (length^2 + width^2).sqrt

-- Defining the minimum height function
noncomputable def min_height (length width : ℝ) : ℝ :=
  (diagonal_length length width) / 4

-- Function to round up to the nearest multiple of 0.1 meters
noncomputable def round_up_to_nearest_tenth (h : ℝ) : ℝ := 
  ((h * 10).ceil) / 10

theorem minimum_ceiling_height_of_soccer_field : 
  round_up_to_nearest_tenth (min_height 90 60) = 27.1 :=
begin
  -- Proof goes here (omitted)
  sorry,
end

end minimum_ceiling_height_of_soccer_field_l432_432376


namespace MoneyDivision_l432_432395

theorem MoneyDivision (w x y z : ℝ)
  (hw : y = 0.5 * w)
  (hx : x = 0.7 * w)
  (hz : z = 0.3 * w)
  (hy : y = 90) :
  w + x + y + z = 450 := by
  sorry

end MoneyDivision_l432_432395


namespace exponential_values_and_inequality_l432_432956

noncomputable def f (a b x : ℝ) := b * a ^ x

theorem exponential_values_and_inequality (A B : ℝ × ℝ)
    (hA : A = (1, 27)) (hB : B = (-1, 3))
    (a b m : ℝ) (ha : a > 0) (hb : b > 0) (ha_ne_1 : a ≠ 1)
    (h_f_A : f a b 1 = 27) (h_f_B : f a b (-1) = 3)
    (h_ineq : ∀ x, x ∈ set.Ici (1:ℝ) → a ^ x + b ^ x ≥ m)
    : a = 3 ∧ b = 9 ∧ m ≤ 12 :=
by
  sorry

end exponential_values_and_inequality_l432_432956


namespace average_stoppage_time_l432_432551

def bus_a_speed_excluding_stoppages := 54 -- kmph
def bus_a_speed_including_stoppages := 45 -- kmph

def bus_b_speed_excluding_stoppages := 60 -- kmph
def bus_b_speed_including_stoppages := 50 -- kmph

def bus_c_speed_excluding_stoppages := 72 -- kmph
def bus_c_speed_including_stoppages := 60 -- kmph

theorem average_stoppage_time :
  (bus_a_speed_excluding_stoppages - bus_a_speed_including_stoppages) / bus_a_speed_excluding_stoppages * 60
  + (bus_b_speed_excluding_stoppages - bus_b_speed_including_stoppages) / bus_b_speed_excluding_stoppages * 60
  + (bus_c_speed_excluding_stoppages - bus_c_speed_including_stoppages) / bus_c_speed_excluding_stoppages * 60
  = 30 / 3 :=
  by sorry

end average_stoppage_time_l432_432551


namespace sum_first_100_terms_l432_432582

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) :=
  n * a₁ + (n * (n - 1) / 2) * d

variables (a₁ d : ℝ)
variables (n : ℕ)
variables (S_odd : ℝ)
hypothesis h_d : d = 1 / 2
hypothesis h_S_odd_99 : S_odd = 60
theorem sum_first_100_terms (h_d : d = 1 / 2) (h_S_odd_99 : S_odd = 60) : sum_arithmetic_sequence a₁ d 100 = 145 :=
sorry

end sum_first_100_terms_l432_432582


namespace hyperbola_equation_l432_432119

theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ m p : ℝ, p > 0 →
      let M := (1, m)
      in y = 4 ∧
        y ^ 2 = 2 * p * x →
        ∃ e : ℝ, e = (a + b) / 2 ∧ 
          ∃ (a b : ℝ), let c := e * sqrt 5 / 2
          in ¬ (a * c = 1) → 
          x^2 - 4 * y^2 = 1 :=
sorry

end hyperbola_equation_l432_432119


namespace solve_inequality_l432_432155

noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.log t (Real.abs (x + 1))

theorem solve_inequality (t : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f x t > 0) →
  Set.Ioo (1/3 : ℝ) (1 : ℝ) = {t | f (8^t - 1) t < f 1 t } :=
by {
  intro h,
  sorry
}

end solve_inequality_l432_432155


namespace expand_expression_l432_432873

theorem expand_expression : ∀ x : ℝ, (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  intro x
  calc
    (x - 3) * (x + 6) = (x - 3) * x + (x - 3) * 6 : by sorry
               ... = x^2 - 3 * x + 6 * x - 18 : by sorry
               ... = x^2 + 3 * x - 18 : by sorry

end expand_expression_l432_432873


namespace pascals_triangle_101_rows_pascals_triangle_only_101_l432_432176

theorem pascals_triangle_101_rows (n : ℕ) :
  (∃ k, (0 ≤ k) ∧ (k ≤ n) ∧ (Nat.choose n k = 101)) → n = 101 :=
begin
  -- assume that there exists some row n where 101 appears in Pascal's Triangle
  intro h,
  cases h with k hk,
  cases hk with hk0 hk1,
  cases hk1 with hk1 hl,
  
  -- we need to show that n = 101
  have h_prime := Nat.prime_101,
  
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

theorem pascals_triangle_only_101 :
  ∀ n : ℕ, (∀ k, (0 ≤ k) ∧ (k ≤ n) → (Nat.choose n k = 101) → n = 101) :=
begin
  intros n k hkn h,
  have h_prime := Nat.prime_101,
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

end pascals_triangle_101_rows_pascals_triangle_only_101_l432_432176


namespace evaluate_root_power_l432_432864

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l432_432864


namespace monotonic_intervals_line_tangent_l432_432951

noncomputable def g (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (∀ x, f x = x * Real.log x ) →
  (∀ x, 0 < x ∧ x < Real.exp (a - 1) → (g f a)' x < 0) ∧
  (∀ x, x > Real.exp (a - 1) → (g f a)' x > 0) :=
sorry

theorem line_tangent (f : ℝ → ℝ) :
  (∀ x, f x = x * Real.log x ) →
  (∃ l : ℝ → ℝ , (∀ x, l x = x - 1) ∧ (l 0 = -1) ∧ (∃ x0, f x0 = x0 * Real.log x0 ∧ y = f x0 ∧ y' = (Real.log x0 + 1))) :=
sorry

end monotonic_intervals_line_tangent_l432_432951


namespace number_101_in_pascals_triangle_l432_432174

/-- Prove that the number 101 appears in exactly one row of Pascal's Triangle, specifically the 101st row. -/
theorem number_101_in_pascals_triangle : 
  ∃! n : ℕ, (∃ k : ℕ, k ≤ n ∧ k ≠ 0 ∧ binom n k = 101) ∧ n = 101 :=
by sorry

end number_101_in_pascals_triangle_l432_432174


namespace extreme_value_sum_l432_432950

-- Define the function f
def f (a b : ℝ) (x : ℝ) := a * x^3 - b * x + 2

-- Define the statement to prove that M + m = 4
theorem extreme_value_sum (a b : ℝ) (M m : ℝ)
  (hM : M = f a b (real.sqrt (b / (3 * a))))
  (hm : m = f a b (-real.sqrt (b / (3 * a)))) :
  M + m = 4 :=
sorry

end extreme_value_sum_l432_432950


namespace knights_count_l432_432622

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432622


namespace CesaroSum151_is_correct_l432_432101

-- Definitions: B = (b_1, b_2, ..., b_150), Cesaro sum for 150-term sequence is 2000.
variable {B : Fin 150 → ℝ}
variable (cesaro_sum_150 : ℝ)
def T (k : Fin 150) (B : Fin 150 → ℝ) : ℝ :=
  ∑ i in Finset.range k.1.succ, B ⟨i, (lt_trans (Fin.is_lt k) (by simp [Nat.lt_succ_self]))⟩

def CesaroSum150 (B : Fin 150 → ℝ) : ℝ :=
  (∑ k in Finset.range 150, T k B) / 150

-- Given condition: Cesaro sum of B is 2000
axiom cesaro_sum_150_valid : CesaroSum150 B = cesaro_sum_150

-- Prove the Cesaro sum for the sequence (100, b_1, b_2, ..., b_150)
def T_extended (k : Fin 151) (B : Fin 150 → ℝ) : ℝ :=
  if k.1 = 0 then 100
  else 100 + ∑ i in Finset.range k.1, B ⟨i, (lt_trans (Fin.is_lt (Fin.castSucc k)) (by simp [Nat.lt_succ_self]))⟩

def CesaroSum151 (B : Fin 150 → ℝ) : ℝ :=
  (∑ k in Finset.range 151, T_extended k B) / 151

theorem CesaroSum151_is_correct (B : Fin 150 → ℝ)
  (cesaro_sum_150_valid : CesaroSum150 B = 2000) :
  abs (CesaroSum151 B - 2086.754966887417) < 1e-9 :=
sorry

end CesaroSum151_is_correct_l432_432101


namespace smallest_number_with_five_different_prime_factors_l432_432335

def has_five_prime_factors (n : ℕ) : Prop :=
  ∃ p1 p2 p3 p4 p5 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5 ∧
  prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ prime p5 ∧ n = p1 * p2 * p3 * p4 * p5

def contains_even_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, prime p ∧ p % 2 = 0 ∧ p ∣ n

theorem smallest_number_with_five_different_prime_factors :
  ∃ n : ℕ, has_five_prime_factors n ∧ contains_even_prime n ∧ (∀ m : ℕ, has_five_prime_factors m ∧ contains_even_prime m → n ≤ m) ∧ n = 2310 :=
by
  sorry

end smallest_number_with_five_different_prime_factors_l432_432335


namespace four_chords_convex_quadrilateral_probability_l432_432457

/-- 
Given eight distinct points on a circle, if four chords joining pairs of the eight points are selected at random, prove that the probability that the four chords form a convex quadrilateral is 2/585. 
-/
theorem four_chords_convex_quadrilateral_probability :
  let total_chords := Nat.choose 8 2,
      total_selections := Nat.choose total_chords 4,
      favorable_outcomes := Nat.choose 8 4 in
  (favorable_outcomes : ℚ) / total_selections = 2 / 585 := by
  sorry

end four_chords_convex_quadrilateral_probability_l432_432457


namespace sum_of_smallest_and_largest_even_l432_432568

theorem sum_of_smallest_and_largest_even (n : ℤ) (h : n + (n + 2) + (n + 4) = 1194) : n + (n + 4) = 796 :=
by
  sorry

end sum_of_smallest_and_largest_even_l432_432568


namespace parabola_problem_l432_432964

-- Define the conditions of the problem using Lean definitions.

def parabola (x y : ℝ) : Prop := y^2 = 16 * x
def is_focus (x y : ℝ) : Prop := x = 4 ∧ y = 0
def line (x : ℝ) : Prop := x = -1

def pointA_on_line_l (a : ℝ) : Prop := line (-1)
def pointB_on_parabola_C (m n : ℝ) : Prop := parabola m n

def FA_FB_relationship (Fx Fy Ax Ay Bx By : ℝ) : Prop :=
  5 * (Bx - Fx) = Ax - Fx ∧ 5 * (By - Fy) = Ay - Fy

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Problem statement (rewriting the mathematical goal to Lean statement)
theorem parabola_problem
  (a m n : ℝ)
  (Ax Ay Bx By : ℝ)
  (FAx FAy : ℝ)
  (h1 : parabola m n)
  (h2 : is_focus FAx FAy)
  (h3 : line Ax)
  (h4 : pointA_on_line_l a)
  (h5 : pointB_on_parabola_C Bx By)
  (h6 : FA_FB_relationship FAx FAy Ax a Bx n)
  : distance Ax a Bx n = 28 :=
sorry

end parabola_problem_l432_432964


namespace num_valid_4x4_arrays_l432_432444

theorem num_valid_4x4_arrays: ∃! (A : Matrix (Fin 4) (Fin 4) ℕ), 
  (∀ i : Fin 4, StrictIncreasing (λ j : Fin 4, A i j)) ∧ 
  (∀ j : Fin 4, StrictIncreasing (λ i : Fin 4, A i j)) ∧ 
  (∀ i j, A i j ∈ Finset.range 16) ∧
  (Finset.univ.image (λ ⟨i, j⟩, A i j) = Finset.range 16) :=
begin
  sorry
end

end num_valid_4x4_arrays_l432_432444


namespace add_2001_1015_l432_432364

theorem add_2001_1015 : 2001 + 1015 = 3016 := 
by
  sorry

end add_2001_1015_l432_432364


namespace rohan_monthly_salary_l432_432271

theorem rohan_monthly_salary (S : ℝ)
  (food_expenditure    : 0.40 * S)
  (rent_expenditure    : 0.20 * S)
  (entertainment_exp   : 0.10 * S)
  (conveyance_exp      : 0.10 * S)
  (savings : ℝ) (h : savings = 0.20 * S) :
  savings = 1500 → S = 7500 :=
by
  intro h₁
  have h₂ : 0.20 * S = 1500 := by rw [← h, h₁]
  have h₃ : S = 1500 / 0.20 := by field_simp [h₂]
  norm_num at h₃
  exact h₃

end rohan_monthly_salary_l432_432271


namespace some_students_not_chess_club_members_l432_432283

variables (Student ChessClub : Type) 
variable (Athlete : Student → Prop)

-- Conditions
variable (H1 : ∃ s : Student, ¬Athlete s) -- Some students are not athletes.
variable (H2 : ∀ c : ChessClub, Athlete (c : Student)) -- All members of the chess club are athletes.

-- Conclusion
theorem some_students_not_chess_club_members : ∃ s : Student, ¬ ∃ c : ChessClub, s = (c : Student) :=
begin
  sorry,
end

end some_students_not_chess_club_members_l432_432283


namespace savanna_more_giraffes_l432_432273

-- Definitions based on conditions
def lions_safari := 100
def snakes_safari := lions_safari / 2
def giraffes_safari := snakes_safari - 10

def lions_savanna := 2 * lions_safari
def snakes_savanna := 3 * snakes_safari

-- Totals given and to calculate giraffes in Savanna
def total_animals_savanna := 410

-- Prove that Savanna has 20 more giraffes than Safari
theorem savanna_more_giraffes :
  ∃ (giraffes_savanna : ℕ), giraffes_savanna = total_animals_savanna - lions_savanna - snakes_savanna ∧
  giraffes_savanna - giraffes_safari = 20 :=
  by
  sorry

end savanna_more_giraffes_l432_432273


namespace ellipse_standard_equation_fixed_point_exists_l432_432510

variable {a b c : ℝ} (hx : a = 2) (hy : b = sqrt 3) (he : c = 1)

theorem ellipse_standard_equation :
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1) :=
sorry

theorem fixed_point_exists :
  (exists D : ℝ × ℝ, D = (-11/8, 0) ∧ (∀ (M N : ℝ × ℝ), M ≠ N ∧ 
  line_passes_through_focus M N (-1, 0) →
  let DM := ((M.1 - D.1), M.2),
      DN := ((N.1 - D.1), N.2)
  in DM.1 * DN.1 + DM.2 * DN.2) = -135/64) :=
sorry

end ellipse_standard_equation_fixed_point_exists_l432_432510


namespace neither_biology_nor_chemistry_l432_432575

def science_club_total : ℕ := 80
def biology_members : ℕ := 50
def chemistry_members : ℕ := 40
def both_members : ℕ := 25

theorem neither_biology_nor_chemistry :
  (science_club_total -
  ((biology_members - both_members) +
  (chemistry_members - both_members) +
  both_members)) = 15 := by
  sorry

end neither_biology_nor_chemistry_l432_432575


namespace outfit_count_l432_432281

theorem outfit_count (shirts pants ties belts : ℕ)
  (h_shirts : shirts = 8)
  (h_pants : pants = 3)
  (h_ties : ties = 6)
  (h_belts : belts = 3) : shirts * pants * ties * belts = 432 :=
by
  rw [h_shirts, h_pants, h_ties, h_belts]
  -- The expected number of outfits is 8 * 3 * 6 * 3 = 432
  -- You would do this multiplication step but we skip with sorry
  sorry

end outfit_count_l432_432281


namespace brit_age_after_vacation_l432_432047

-- Define the given conditions and the final proof question

-- Rebecca's age is 25 years
def rebecca_age : ℕ := 25

-- Brittany is older than Rebecca by 3 years
def brit_age_before_vacation (rebecca_age : ℕ) : ℕ := rebecca_age + 3

-- Brittany goes on a 4-year vacation
def vacation_duration : ℕ := 4

-- Prove that Brittany’s age when she returns from her vacation is 32
theorem brit_age_after_vacation (rebecca_age vacation_duration : ℕ) : brit_age_before_vacation rebecca_age + vacation_duration = 32 :=
by
  sorry

end brit_age_after_vacation_l432_432047


namespace cost_of_720_chocolates_is_216_l432_432367

def cost_of_chocolates
  (candies_per_box : ℕ)
  (cost_per_box : ℕ)
  (discount_threshold : ℕ)
  (discount_percentage : ℕ)
  (total_candies : ℕ)
  : ℕ :=
  let boxes := total_candies / candies_per_box in
  let total_cost := boxes * cost_per_box in
  if boxes > discount_threshold then
    total_cost - (total_cost * discount_percentage / 100)
  else
    total_cost

theorem cost_of_720_chocolates_is_216 :
  cost_of_chocolates 30 10 20 10 720 = 216 :=
by
  sorry

#eval cost_of_chocolates 30 10 20 10 720 -- expected: 216

end cost_of_720_chocolates_is_216_l432_432367


namespace ellipse_equation_l432_432137

-- Definitions based on the conditions.
def ellipse_eq (m n : Real) (x y : Real) : Prop :=
  m * x^2 + n * y^2 = 1

-- Points P and Q
def P : Real × Real := (1, Real.sqrt 3 / 2)
def Q : Real × Real := (2, 0)

-- The final theorem to prove
theorem ellipse_equation :
  ∃ (m n : Real), (ellipse_eq m n P.1 P.2) ∧ (ellipse_eq m n Q.1 Q.2) ∧ (m = (1 / 4) ∧ n = 1) :=
sorry

end ellipse_equation_l432_432137


namespace dividend_50100_l432_432995

theorem dividend_50100 (D Q R : ℕ) (h1 : D = 20 * Q) (h2 : D = 10 * R) (h3 : R = 100) : 
    D * Q + R = 50100 := by
  sorry

end dividend_50100_l432_432995


namespace probability_of_condition_l432_432692

-- Define the vertices of the rectangle
def rectangle_vertices : Set (ℝ × ℝ) :=
  {p | (0 ≤ p.1 ∧ p.1 ≤ 3000) ∧ (0 ≤ p.2 ∧ p.2 ≤ 4500)}

-- Define the condition for the probability question
def condition (p : ℝ × ℝ) : Prop :=
  p.1 < 3 * p.2

-- Define the area of the rectangle
def rectangle_area : ℝ :=
  3000 * 4500

-- Define the area of the region where the condition holds
def region_area : ℝ :=
  (1 / 2) * (1000 + 4500) * 3000

-- Define the probability
def probability : ℝ :=
  region_area / rectangle_area

-- The main statement to prove
theorem probability_of_condition : 
  probability = 11 / 18 := sorry

end probability_of_condition_l432_432692


namespace solve_for_m_l432_432340

theorem solve_for_m (m : ℝ) : 
  (∀ x : ℝ, (x = 2) → ((m - 2) * x = 5 * (x + 1))) → (m = 19 / 2) :=
by
  intro h
  have h1 := h 2
  sorry  -- proof can be filled in later

end solve_for_m_l432_432340


namespace revenue_fall_1999_l432_432992

theorem revenue_fall_1999 :
  ∀ (R R' P P' : ℝ),
    P = 0.10 * R →
    P' = 0.14 * R' →
    P' = 0.98 * P →
    (R' / R  = 0.7) :=
begin
  intros R R' P P' h1 h2 h3,
  sorry,
end

end revenue_fall_1999_l432_432992


namespace janet_gas_usage_l432_432603

theorem janet_gas_usage :
  ∀ (d_dermatologist d_gynecologist miles_per_gallon : ℕ),
    d_dermatologist = 30 →
    d_gynecologist = 50 →
    miles_per_gallon = 20 →
    (2 * d_dermatologist + 2 * d_gynecologist) / miles_per_gallon = 8 :=
by
  intros d_dermatologist d_gynecologist miles_per_gallon
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end janet_gas_usage_l432_432603


namespace problem_equivalence_l432_432958

noncomputable def f (x : ℝ) : ℝ := x + Real.sin (Real.pi * x) - 3

theorem problem_equivalence :
  (finset.sum (finset.range 4038) (λ k, f ((k + 1) / 2019))) = -8074 :=
  sorry

end problem_equivalence_l432_432958


namespace cookies_sold_by_Lucy_l432_432412

theorem cookies_sold_by_Lucy :
  let cookies_first_round := 34
  let cookies_second_round := 27
  cookies_first_round + cookies_second_round = 61 := by
  sorry

end cookies_sold_by_Lucy_l432_432412


namespace blue_first_yellow_second_probability_l432_432403

open Classical

-- Definition of initial conditions
def total_marbles : Nat := 3 + 4 + 9
def blue_marbles : Nat := 3
def yellow_marbles : Nat := 4
def pink_marbles : Nat := 9

-- Probability functions
def probability_first_blue : ℚ := blue_marbles / total_marbles
def probability_second_yellow_given_blue : ℚ := yellow_marbles / (total_marbles - 1)

-- Combined probability
def combined_probability_first_blue_second_yellow : ℚ := 
  probability_first_blue * probability_second_yellow_given_blue

-- Theorem statement
theorem blue_first_yellow_second_probability :
  combined_probability_first_blue_second_yellow = 1 / 20 :=
by
  -- Proof will be provided here
  sorry

end blue_first_yellow_second_probability_l432_432403


namespace record_withdrawal_example_l432_432346

-- Definitions based on conditions
def ten_thousand_dollars := 10000
def record_deposit (amount : ℕ) : ℤ := amount / ten_thousand_dollars
def record_withdrawal (amount : ℕ) : ℤ := -(amount / ten_thousand_dollars)

-- Lean 4 statement to prove the problem
theorem record_withdrawal_example :
  (record_deposit 30000 = 3) → (record_withdrawal 20000 = -2) :=
by
  intro h
  sorry

end record_withdrawal_example_l432_432346


namespace monotonicity_of_f_range_of_k_for_three_zeros_l432_432537

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l432_432537


namespace Alex_runs_faster_l432_432698

def Rick_speed : ℚ := 5
def Jen_speed : ℚ := (3 / 4) * Rick_speed
def Mark_speed : ℚ := (4 / 3) * Jen_speed
def Alex_speed : ℚ := (5 / 6) * Mark_speed

theorem Alex_runs_faster : Alex_speed = 25 / 6 :=
by
  -- Proof is skipped
  sorry

end Alex_runs_faster_l432_432698


namespace two_digit_number_l432_432339

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h1 : x^2 + y^2 = 10*x + y + 11) (h2 : 2*x*y = 10*x + y - 5) :
  10*x + y = 95 ∨ 10*x + y = 15 := 
sorry

end two_digit_number_l432_432339


namespace alpha_plus_beta_l432_432983

theorem alpha_plus_beta (α β : ℝ) (h1 : cos (α - β) = (sqrt 5) / 5) 
  (h2 : cos (2 * α) = (sqrt 10) / 10)
  (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2)
  (h5 : α < β) : α + β = 3 * π / 4 := by
  sorry

end alpha_plus_beta_l432_432983


namespace cheburashkas_balloon_exchange_impossible_l432_432275

theorem cheburashkas_balloon_exchange_impossible :
  ∀ (n : ℕ), n = 7 → (∃ (red yellow : ℕ), red = 7 ∧ yellow = 7 ∧ red % 2 = 1 ∧ yellow % 2 = 1) →
  ¬( ∃ f : fin 7 → fin 7, (∀ i, f i ≠ i ∧ f (f i) = i) ) :=
by sorry

end cheburashkas_balloon_exchange_impossible_l432_432275


namespace ellipse_C_eq_find_m_value_l432_432149

-- Definitions and conditions based on a)
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)
def triangle_perimeter (F1 F2 B : Point) : ℝ := distance F1 F2 + distance F1 B + distance F2 B
def distance_from_point_to_line (F1 : Point) (line : Line) : ℝ
def line_eq (B F2 : Point) : Line := line_through_points B F2

variables a b c : ℝ
variables F1 F2 B : Point

-- Conditions
axiom cond1 : a > b ∧ b > 0
axiom cond2 : 2 * c + 2 * a = 6
axiom cond3 : 2 * c * b = a * b
axiom cond4 : a^2 = b^2 + c^2
axiom cond5 : triangle_perimeter F1 F2 B = 6
axiom cond6 : distance_from_point_to_line F1 (line_eq B F2) = b

-- Question (1) - Find equation of ellipse C
theorem ellipse_C_eq : ellipse_eq 2 (sqrt 3) :=
by
  sorry

-- Definitions for the second part based on the conditions
def ellipse_major_axes (A1 A2 : Point) : ℝ := distance A1 A2
def point_on_ellipse (P : Point) : Prop := ∀ x y : ℝ, (x^2 / (2^2) + y^2 / (3) = 1)

variables A1 A2 P : Point

axiom cond7 : A1 = (-2, 0)
axiom cond8 : A2 = (2, 0)
axiom cond9 : point_on_ellipse P
axiom cond10 : P ≠ A1 ∧ P ≠ A2

-- Question (2) - Find the value of the real number m
theorem find_m_value (m : ℝ) : m = 14 :=
by
  sorry

end ellipse_C_eq_find_m_value_l432_432149


namespace Cathy_total_work_hours_is_180_l432_432055

-- Let's define the conditions as parameters
variables
  (work_hours_per_week : ℕ := 20)
  (weeks_per_month : ℕ := 4)
  (months : ℕ := 2)
  (sick_week_shift_hours : ℕ := 20)

-- Define the function that calculates the total hours Cathy worked
def total_hours_Cathy_works (work_hours_per_week weeks_per_month months sick_week_shift_hours : ℕ) : ℕ :=
  let normal_hours := work_hours_per_week * weeks_per_month * months
  in normal_hours + sick_week_shift_hours

-- The main theorem that needs to be proved
theorem Cathy_total_work_hours_is_180 :
  total_hours_Cathy_works work_hours_per_week weeks_per_month months sick_week_shift_hours = 180 :=
  by sorry

end Cathy_total_work_hours_is_180_l432_432055


namespace min_value_of_fn_l432_432129

noncomputable def f (x : ℝ) : ℝ := 1 / x + 9 / (2 - x)

theorem min_value_of_fn : ∃ x ∈ Ioo 0 2, ∀ y ∈ Ioo 0 2, f x ≤ f y ∧ f x = 8 :=
by
  sorry

end min_value_of_fn_l432_432129


namespace proof_problem_l432_432672

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 - b * x^2 - c * x + 1

axiom a1 : ℕ → ℝ
axiom a1_1 : a1 1 = 1
axiom a2_2 : a1 2 = 2

def bn (n : ℕ) : ℝ := Real.logBase 2 (a1 (2 * n))

axiom extremum_point : (3 : ℝ) * (a1 3 : ℝ) - 2 * (a1 2 : ℝ) - (a1 4 : ℝ) = 0

theorem proof_problem :
  (⌊ ∑ i in (Finset.range 2018).succ, 2018 / ((bn i) * (bn (i + 1))) ⌋ : ℤ) = 1008 :=
sorry

end proof_problem_l432_432672


namespace rectangle_perimeter_ratio_l432_432019

theorem rectangle_perimeter_ratio
  (side_length : ℝ) (fold_height_ratio : ℝ) (small_width_ratio : ℝ) (small_height_ratio : ℝ)
  (h1 : side_length = 6) (h2 : fold_height_ratio = 1 / 2) (h3 : small_width_ratio = 1 / 2) (h4 : small_height_ratio = 1 / 3) :
  let height_unfolded := fold_height_ratio * side_length,
      width_unfolded := side_length,
      height_folded := height_unfolded,
      width_folded := width_unfolded * fold_height_ratio,
      perimeter_largest := 2 * (height_folded + width_unfolded),
      height_small := height_folded,
      width_small := width_folded * small_width_ratio,
      perimeter_small := 2 * (height_small + width_small)
  in perimeter_small / perimeter_largest = 1 / 2 :=
by
  sorry

end rectangle_perimeter_ratio_l432_432019


namespace jamie_coins_l432_432223

theorem jamie_coins : 
  ∀ (x : ℕ), 
    (1 * x + 5 * x + 10 * x = 1200) → 
    x = 75 :=
by
  intro x h
  have h1 : 16 * x = 1200 :=
    calc
      16 * x = 1 * x + 5 * x + 10 * x : by rw [← add_assoc, ← add_assoc, mul_one]
          ... = 1200                 : by rw h
  exact (Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero (by decide)) h1)

end jamie_coins_l432_432223


namespace five_digit_even_digits_probability_l432_432000

noncomputable theory

def five_digit_number : Type := {n : ℕ // n ≥ 10000 ∧ n < 100000}

def is_even (n : ℕ) : Prop := n % 2 = 0

def thousands_digit (n : five_digit_number) : ℕ := (n.val / 1000) % 10

def hundreds_digit (n : five_digit_number) : ℕ := (n.val / 100) % 10

def probability_even_digits (p : ℚ) : Prop :=
  ∀ (n : five_digit_number), 
  (thousands_digit n).is_even ∧ (hundreds_digit n).is_even →
  p = 1 / 4

theorem five_digit_even_digits_probability : probability_even_digits (1 / 4) :=
  sorry

end five_digit_even_digits_probability_l432_432000


namespace eval_sqrt_pow_l432_432871

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l432_432871


namespace total_cost_of_two_rackets_l432_432257

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l432_432257


namespace transitive_defeats_number_of_rules_l432_432314

-- Conditions for the sets of cards
variables {α : Type*} [linear_order α] (A B C : fin 100 → α)

-- Definition: A defeats B if for all i, A(i) > B(i)
def defeats (A B : fin 100 → α) : Prop := ∀ i, A i > B i

-- Transitivity condition: if A defeats B and B defeats C, then A defeats C.
theorem transitive_defeats (hAB : defeats A B) (hBC : defeats B C) : defeats A C := 
by
  intros i,
  exact lt_trans (hAB i) (hBC i)

-- Main statement: there are 100 possible rules to determine the winner
theorem number_of_rules : ∃ n, n = 100 :=
by
  use 100,
  exact rfl

end transitive_defeats_number_of_rules_l432_432314


namespace Mart_income_percentage_of_Juan_l432_432679

variable (J T M : ℝ)

-- Conditions
def Tim_income_def : Prop := T = 0.5 * J
def Mart_income_def : Prop := M = 1.6 * T

-- Theorem to prove
theorem Mart_income_percentage_of_Juan
  (h1 : Tim_income_def T J) 
  (h2 : Mart_income_def M T) : 
  (M / J) * 100 = 80 :=
by
  sorry

end Mart_income_percentage_of_Juan_l432_432679


namespace find_a_parallel_lines_l432_432973

theorem find_a_parallel_lines 
  (a : ℝ) 
  (l1 : ∀ x y : ℝ, ax + 2y + 6 = 0) 
  (l2 : ∀ x y : ℝ, x + (a - 1)y + (a^2 - 1) = 0) 
  (parallel : ∀ {x y : ℝ}, ax + 2y + 6 = 0 → x + (a - 1)y + (a^2 - 1) = 0 → true) : 
  a = -1 := 
sorry 

end find_a_parallel_lines_l432_432973


namespace tetrahedron_volume_le_one_eighth_l432_432263

noncomputable def volume_le_one_eighth (A B C D : Type) (edge_length_CD : ℝ) (edge_length_AB : ℝ) 
    (hCD : edge_length_CD > 1) (hAB : edge_length_AB = x) : Prop :=
  ∃ (V : ℝ), V = volume_of_tetrahedron A B C D ∧ V ≤ 1 / 8

axiom volume_of_tetrahedron : ∀ (A B C D : Type), ℝ

theorem tetrahedron_volume_le_one_eighth (A B C D : Type) 
  (edge_length_CD edge_length_AB : ℝ) (hCD : edge_length_CD > 1) (hAB : edge_length_AB = x) :
  volume_le_one_eighth A B C D edge_length_CD edge_length_AB hCD hAB := 
by 
  sorry

end tetrahedron_volume_le_one_eighth_l432_432263


namespace find_a_of_parabola_l432_432295

theorem find_a_of_parabola
  (a b c : ℝ)
  (h_point : 2 = c)
  (h_vertex : -2 = a * (2 - 2)^2 + b * 2 + c) :
  a = 1 :=
by
  sorry

end find_a_of_parabola_l432_432295


namespace min_rubles_to_mark_all_numbers_l432_432253

theorem min_rubles_to_mark_all_numbers : 
  ( ∀ n ∈ { n : ℕ | n ≥ 2 ∧ n ≤ 30 }, 
    ∃ (marked : finset ℕ), 
      (∀ m ∈ marked, 
        m | n ∨ n | m ∨ n = m ∨ m = 1)) → 
  (marked.card ≤ 5): 
sorry

end min_rubles_to_mark_all_numbers_l432_432253


namespace line_equation_through_point_and_intercepts_l432_432885

theorem line_equation_through_point_and_intercepts (A : ℝ × ℝ) (bx by : ℝ) (hbx : bx = 2 * by) :
  (A = (-2, 3) ∧ (bx = 2 * by)) → (∃ k : ℝ, k ≠ 0 ∧ (A.1 = -2 ∧ A.2 = 3) ∧ 
                                     ((1/k * A.1 + 2/k * A.2 - 4/k = 0) ∨ (3 * A.1 + 2 * A.2 = 0))) :=
by
  sorry

end line_equation_through_point_and_intercepts_l432_432885


namespace map_distance_to_real_distance_l432_432714

theorem map_distance_to_real_distance (map_distance scale_inch scale_mile : ℝ) (h_scale : scale_inch = 0.5) (h_mile : scale_mile = 6) (h_map_distance : map_distance = 12) : 
  let miles_per_inch := scale_mile / scale_inch in
  map_distance * miles_per_inch = 144 :=
by
  sorry

end map_distance_to_real_distance_l432_432714


namespace maxValue_of_MF1_MF2_l432_432906

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432906


namespace pencils_per_student_l432_432014

theorem pencils_per_student (total_pencils : ℤ) (num_students : ℤ) (pencils_per_student : ℤ)
  (h1 : total_pencils = 195)
  (h2 : num_students = 65) :
  total_pencils / num_students = 3 :=
by
  sorry

end pencils_per_student_l432_432014


namespace sum_inequality_l432_432355

theorem sum_inequality {n : ℕ} {x y : Fin n → ℝ}
  (h_sort_x : ∀ i j, i ≤ j → x i ≤ x j) 
  (h_sort_y : ∀ i j, i ≤ j → y i ≥ y j)
  (h_sum_eq : (∑ i, (i + 1) * x i) = ∑ i, (i + 1) * y i) 
  (α : ℝ) :
  (∑ i, x i * ⌊ (i + 1) * α ⌋) ≥ (∑ i, y i * ⌊ (i + 1) * α ⌋) :=
sorry

end sum_inequality_l432_432355


namespace probability_in_shaded_region_l432_432659

noncomputable def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ := nat.choose n k

theorem probability_in_shaded_region :
  ∃ (k : ℕ), k > 0 ∧ binomial_coefficient k 3 * (1 / (k : ℝ))^3 = 1 / 16 ∧
  let intersection_points := {0, 4} in
  let area_S := (1 / 4 : ℝ) * real.pi * 4^2 - (1 / 2) * 4 * 4 in
  let total_area := 4 * 4 in
  (area_S / total_area) = (real.pi / 4 - 1 / 2) :=
by
  sorry

end probability_in_shaded_region_l432_432659


namespace find_a_value_l432_432490

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a_value :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a 0 + f a 1 = a) → a = 1 / 2 :=
by
  sorry

end find_a_value_l432_432490


namespace find_r_sum_b_n_l432_432104

-- Definition and conditions for the geometric sequence and sum of the first n terms
variable {b : ℝ} (a r : ℝ) (n : ℝ) (S : ℕ → ℝ) (b > 0) (b ≠ 1) (S_n = b^n + r)

-- There is a proof that r = -1
theorem find_r : r = -1 := by {
  sorry
}

-- Definitions for b = 2 scenario
variable {a_n b_n T_n : ℕ → ℝ}

-- Geometric sequence with b = 2
def a_n (n : ℕ) : ℝ := 2^(n-1)

def b_n (n : ℕ) : ℝ := (n + 1) / 2^(n+1)

-- Sum of the first n terms of the sequence b_n
def T_n (n : ℕ) : ℝ := ∑ i in range n, b_n i

-- There is a proof that sum of first n terms equals the given expression for T_n
theorem sum_b_n (n : ℕ) : T_n n = (3 / 2) - (1 / 2^n) - ((n + 3) / 2^(n+1)) := by {
  sorry
}

end find_r_sum_b_n_l432_432104


namespace probability_joint_independent_events_conditional_probability_independent_events_l432_432140

theorem probability_joint_independent_events
  (pa pb pc : ℝ)
  (habc : a ∧ b ∧ c)
  (ind : independent a b c)
  (hpa : pa = 5/7)
  (hpb : pb = 2/5)
  (hpc : pc = 3/4) :
  Prob (a ∧ b ∧ c) = 3/14 :=
by
  sorry

theorem conditional_probability_independent_events
  (pa pb pc : ℝ)
  (habc : a ∧ b ∧ c)
  (ind : independent a b c)
  (hpa : pa = 5/7)
  (hpb : pb = 2/5)
  (hpc : pc = 3/4) :
  Prob (a ∧ b | c) = 2/7 :=
by
  sorry

end probability_joint_independent_events_conditional_probability_independent_events_l432_432140


namespace shopkeeper_profit_percentage_l432_432804

-- Define the rates and calculations for mangoes, apples, and oranges
def mango_buy_rate := 10 -- 10 mangoes per rupee
def mango_sell_rate := 4 -- 4 mangoes per rupee

def apple_buy_rate := 5 -- 5 apples per rupee
def apple_sell_rate := 3 -- 3 apples per rupee

def orange_buy_rate := 8 -- 8 oranges per rupee
def orange_sell_rate := 2 -- 2 oranges per rupee

-- Function to calculate profit percentage
def profit_percentage (cp sp : ℚ) : ℚ :=
  ((sp - cp) / cp) * 100

-- Main theorem
theorem shopkeeper_profit_percentage :
  let cp_mangoes := 10,
      sp_mangoes := (cp_mangoes * mango_buy_rate) / mango_sell_rate,
      cp_apples := 10,
      sp_apples := (cp_apples * apple_buy_rate) / apple_sell_rate,
      cp_oranges := 10,
      sp_oranges := (cp_oranges * orange_buy_rate) / orange_sell_rate,
      total_cp := cp_mangoes + cp_apples + cp_oranges,
      total_sp := sp_mangoes + sp_apples + sp_oranges in
  profit_percentage total_cp total_sp = 172.23 := sorry

end shopkeeper_profit_percentage_l432_432804


namespace arithmetic_sequence_terms_l432_432567

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h2 : a 1 + a 2 + a 3 = 34)
  (h3 : a n + a (n-1) + a (n-2) = 146)
  (h4 : S n = 390)
  (h5 : ∀ i j, a i + a j = a (i+1) + a (j-1)) :
  n = 13 :=
sorry

end arithmetic_sequence_terms_l432_432567


namespace trapezoid_isosceles_l432_432202

theorem trapezoid_isosceles 
  (A B C D L : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_bases : parallel (line A B) (line C D))
  (h_nonparallel : ¬ parallel (line A D) (line B C))
  (h_intersection : intersection (line AC) (line BD) = L)
  (h_equidistant : equidistant_from_lines L (line (A, D)) (line (B, C))) :
  is_isosceles_trapezoid A B C D := 
sorry

end trapezoid_isosceles_l432_432202


namespace factorization_roots_l432_432889

theorem factorization_roots (x : ℂ) : 
  (x^3 - 2*x^2 - x + 2) * (x - 3) * (x + 1) = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  -- Note: Proof to be completed
  sorry

end factorization_roots_l432_432889


namespace ratio_is_sqrt_three_l432_432838

-- Definitions based on the conditions
def side_length_of_cube : ℝ := 2

def vertices_of_tetrahedron : set (ℝ × ℝ × ℝ) := 
  {(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)}

-- Calculate side length of the tetrahedron
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def side_length_of_tetrahedron : ℝ := 
  distance (0, 0, 0) (2, 2, 0)

-- Surface area calculations
def surface_area_of_cube (s : ℝ) : ℝ := 
  6 * s^2

def surface_area_of_tetrahedron (a : ℝ) : ℝ := 
  real.sqrt 3 * a^2

-- Ratio of surface areas
def ratio_surface_areas : ℝ := 
  surface_area_of_cube side_length_of_cube / 
  surface_area_of_tetrahedron side_length_of_tetrahedron

-- Theorem statement
theorem ratio_is_sqrt_three : ratio_surface_areas = real.sqrt 3 :=
by 
  unfold ratio_surface_areas surface_area_of_cube 
        surface_area_of_tetrahedron side_length_of_tetrahedron distance
  -- Insert necessary proof steps here
  sorry

end ratio_is_sqrt_three_l432_432838


namespace number_of_divisors_of_16m3_l432_432660

theorem number_of_divisors_of_16m3 (m : ℕ) (h1 : m % 2 = 1) (h2 : (finset.Ico 1 (m + 1)).filter (λ x, m % x = 0).card = 17) : 
  (finset.Ico 1 (16 * m^3 + 1)).filter (λ x, (16 * m^3) % x = 0).card = 245 :=
sorry

end number_of_divisors_of_16m3_l432_432660


namespace car_enters_and_leaves_storm_l432_432789

def car_position (t : ℝ) : ℝ × ℝ :=
  (3 / 4 * t, 0)

def storm_center_position (t : ℝ) : ℝ × ℝ :=
  (- t * sqrt 5 / 3, 130 - t * sqrt 5 / 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def enter_storm_time (t₁ t₂ : ℝ) : Prop :=
  ∃ t, t₁ ≤ t ∧ t ≤ t₂ ∧ distance (car_position t) (storm_center_position t) = 60

theorem car_enters_and_leaves_storm :
  ∃ t₁ t₂, enter_storm_time t₁ t₂ ∧ (1 / 2 * (t₁ + t₂) = 198) :=
sorry

end car_enters_and_leaves_storm_l432_432789


namespace point_on_number_line_l432_432299

theorem point_on_number_line (a : ℤ) (h : abs (a + 3) = 4) : a = 1 ∨ a = -7 := 
sorry

end point_on_number_line_l432_432299


namespace number_of_noncongruent_triangles_with_perimeter_9_l432_432733

theorem number_of_noncongruent_triangles_with_perimeter_9 : ∃ n : ℕ, n = 3 ∧ 
  (∀ t : List ℕ, t.length = 3 ∧ t.sum = 9 ∧ 
                 (∀ a b c : ℕ, Perm (a::b::c::[]) t → a + b > c ∧ a + c > b ∧ b + c > a) →
                 (∃ (u v : List ℕ) (h : u ≠ v), List.Nodup u ∧ List.Nodup v ∧ Perm u t ∧ Perm v t)) :=
  sorry

end number_of_noncongruent_triangles_with_perimeter_9_l432_432733


namespace fiftieth_term_l432_432303

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  d ∈ n.digits 10

def valid_sequence_term (n : ℕ) : Prop :=
  n % 4 = 0 ∧ contains_digit 2 n

def sequence_of_valid_terms : List ℕ :=
  List.filter valid_sequence_term (List.range (4*500)) -- 4*500 is an arbitrary upper bound for illustration

theorem fiftieth_term :
  List.nth sequence_of_valid_terms 49 = some 424 :=
by
  sorry

end fiftieth_term_l432_432303


namespace Elmer_vs_Milton_food_l432_432260

def Penelope_daily_food := 20  -- Penelope eats 20 pounds per day
def Greta_to_Penelope_ratio := 1 / 10  -- Greta eats 1/10 of what Penelope eats
def Milton_to_Greta_ratio := 1 / 100  -- Milton eats 1/100 of what Greta eats
def Elmer_to_Penelope_difference := 60  -- Elmer eats 60 pounds more than Penelope

def Greta_daily_food := Penelope_daily_food * Greta_to_Penelope_ratio
def Milton_daily_food := Greta_daily_food * Milton_to_Greta_ratio
def Elmer_daily_food := Penelope_daily_food + Elmer_to_Penelope_difference

theorem Elmer_vs_Milton_food :
  Elmer_daily_food = 4000 * Milton_daily_food := by
  sorry

end Elmer_vs_Milton_food_l432_432260


namespace part1_part2_l432_432649

-- Definitions for the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Proof statement for the first part
theorem part1 (m : ℝ) (h : m = 4) : A ∪ B m = { x | -2 ≤ x ∧ x ≤ 7 } :=
sorry

-- Proof statement for the second part
theorem part2 (h : ∀ {m : ℝ}, B m ⊆ A) : ∀ m : ℝ, m ∈ Set.Iic 3 :=
sorry

end part1_part2_l432_432649


namespace Abby_sits_in_seat_3_l432_432029

theorem Abby_sits_in_seat_3:
  ∃ (positions : Fin 5 → String),
  (positions 3 = "Abby") ∧
  (positions 4 = "Bret") ∧
  ¬ ((positions 3 = "Dana") ∨ (positions 5 = "Dana")) ∧
  ¬ ((positions 2 = "Erin") ∧ (positions 3 = "Carl") ∨
    (positions 3 = "Erin") ∧ (positions 5 = "Carl")) :=
  sorry

end Abby_sits_in_seat_3_l432_432029


namespace cartesian_line_eq_min_distance_from_C_to_l_l432_432153

section Problem

-- Define the ellipse curve C
def curve_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 5) = 1

-- Define the polar equation of line l
def polar_line_eq (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - π/4) = 2 * Real.sqrt 2

-- Question Ⅰ: Prove that the Cartesian equation of line l is x + y - 4 = 0
theorem cartesian_line_eq (x y : ℝ) (h : polar_line_eq (Real.sqrt (x^2 + y^2)) (Real.atan2 y x)) :
  x + y = 4 :=
sorry

-- Question Ⅱ: Prove that the minimum distance from any point M on curve C to line l is sqrt(2)/2
theorem min_distance_from_C_to_l (θ : ℝ) (h₁ : curve_eq (2 * Real.cos θ) (Real.sqrt 5 * Real.sin θ)) :
  ∃ d, d = Real.sqrt 2 / 2 ∧ ∀ (x y : ℝ), curve_eq x y → d ≤ Real.abs ((x + y - 4) / Real.sqrt 2) :=
sorry

end Problem

end cartesian_line_eq_min_distance_from_C_to_l_l432_432153


namespace coefficient_of_x4_l432_432944

theorem coefficient_of_x4 (n : ℕ) (f : ℕ → ℕ → ℝ)
  (h1 : (2 : ℕ) ^ n = 256) :
  (f 8 4) * (2 : ℕ) ^ 4 = 1120 :=
by
  sorry

end coefficient_of_x4_l432_432944


namespace max_good_pairs_is_197_l432_432704

def distinct_pairs (n : Nat) : Prop :=
  ∀ (i j : Fin n), i ≠ j → (a i ≠ a j ∨ b i ≠ b j)

def good_pair (a b : Fin 100 → Nat) (i j : Fin 100) : Prop :=
  1 ≤ i < j ∧ |a i * b j - a j * b i| = 1

def max_good_pairs (a b : Fin 100 → Nat) : Nat :=
  ∑ i j in range 100, if good_pair a b i j then 1 else 0

theorem max_good_pairs_is_197 (a b : Fin 100 → Nat) :
  distinct_pairs a b → max_good_pairs a b ≤ 197 := sorry

end max_good_pairs_is_197_l432_432704


namespace mf_length_l432_432963

theorem mf_length (x y : ℝ) (F : (ℝ × ℝ)) (M N : (ℝ × ℝ)) 
  (h1 : ∀ x y : ℝ, y^2 = 4 * x → F = (1, 0))
  (h2 : ∃ m : ℝ, ∀ y : ℝ, x = m * y + 1)
  (h3 : (M.Fst = 2 * N.Fst + 2))
  (h4 : ∥D - F∥ = 1)
  : ∥M - F∥ = 2 + sqrt 3 :=
sorry

end mf_length_l432_432963


namespace find_x_if_parallel_l432_432974

-- Define the vectors and the condition of them being parallel

def vector1 : (ℝ × ℝ × ℝ) := (1, 2, 3)
def vector2 (x : ℝ) : (ℝ × ℝ × ℝ) := (x, 4, 6)
def are_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop := 
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2, k * v2.3)

-- Statement to prove the value of x given the condition
theorem find_x_if_parallel (x : ℝ) (h : are_parallel vector1 (vector2 x)) : x = 2 :=
sorry

end find_x_if_parallel_l432_432974


namespace Kaydence_age_l432_432311

theorem Kaydence_age (family_age father_age mother_diff brother_ratio sister_age : ℕ)
  (H1 : family_age = 200)
  (H2 : father_age = 60)
  (H3 : mother_diff = 2)
  (H4 : brother_ratio = 2)
  (H5 : sister_age = 40)
  (Hmother : mother_age = father_age - mother_diff)
  (Hbrother : brother_age = father_age / brother_ratio)
  (Hsum_except_Kaydence : father_age + (father_age - mother_diff) + (father_age / brother_ratio) + sister_age + Kaydence_age = family_age)
  : Kaydence_age = 12 := 
by
  sorry

end Kaydence_age_l432_432311


namespace consecutive_erase_mean_integer_possible_erasable_values_in_100_l432_432785

-- Part (a)
theorem consecutive_erase_mean_integer (n : ℕ) (h : n > 2) :
  ∃ k : ℕ, k ∈ (finset.range (n + 1)).erase (n + 1) ∧ ((finset.range (n + 1)).erase k).sum % (n - 1) = 0 :=
sorry

-- Part (b)
theorem possible_erasable_values_in_100 :
  ∃ k : ℕ, k ∈ {1, 100} ∧ ((finset.range 101).erase k).sum % 99 = 0 :=
sorry

end consecutive_erase_mean_integer_possible_erasable_values_in_100_l432_432785


namespace length_of_train_l432_432348

-- Definition of conditions
def speed_km_per_hr : ℝ := 72
def platform_length_m : ℝ := 250
def crossing_time_s : ℝ := 15

-- Conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Conversion from km/hr to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * km_to_m / hr_to_s

-- Distance covered by the train while crossing the platform
def distance_covered_m : ℝ := speed_m_per_s * crossing_time_s

-- Length of the train
theorem length_of_train : (distance_covered_m = platform_length_m + 50) := by
  sorry

end length_of_train_l432_432348


namespace comparison_among_abc_l432_432941

noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (1/5)^2
noncomputable def c : ℝ := Real.log (1/5) / Real.log 2

theorem comparison_among_abc : a > b ∧ b > c :=
by
  -- Assume the necessary conditions and the conclusion.
  sorry

end comparison_among_abc_l432_432941


namespace probability_sum_die_rolls_odd_l432_432317

theorem probability_sum_die_rolls_odd 
  (h1 : ∀ (c1 c2 c3 : bool), c1 ∨ c2 ∨ c3)
  (h2 : ∀ (num_heads : ℕ), num_heads ≤ 3) : 
  probability (sum_die_rolls_odd h1 h2) = 7 / 16 :=
sorry

-- Definitions required for the theorem
def sum_die_rolls_odd 
  (h1 : ∀ (c1 c2 c3 : bool), c1 ∨ c2 ∨ c3)
  (h2 : ∀ (num_heads : ℕ), num_heads ≤ 3) : Event :=
  -- Here the exact formalization of the conditions goes, which would need to
  -- encompass the scenario of number of heads and calculation of the sum being odd.
  sorry

end probability_sum_die_rolls_odd_l432_432317


namespace center_symmetry_l432_432108

def f (x : ℝ) := Real.sin (x + (Real.pi / 2))
def g (x : ℝ) := Real.sin (Real.pi - x)

theorem center_symmetry : ∃ C : ℝ × ℝ, C = (3 * Real.pi / 4, 0) ∧
  ∀ x : ℝ, f(x) + g(x) = f(2 * (3 * Real.pi / 4) - x) + g(2 * (3 * Real.pi / 4) - x) :=
by
  sorry

end center_symmetry_l432_432108


namespace sequence_sum_l432_432802

theorem sequence_sum (a b : ℕ → ℝ)
  (h0 : ∀ n, (a n.succ, b n.succ) = (real.sqrt 3 * a n - b n, real.sqrt 3 * b n + a n))
  (h1 : a 100 = 2)
  (h2 : b 100 = 4) :
  a 1 + b 1 = 1 / (2 ^ 98) :=
sorry

end sequence_sum_l432_432802


namespace product_third_side_approximation_l432_432321

def triangle_third_side (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

noncomputable def product_of_third_side_lengths : ℝ :=
  Real.sqrt 41 * 3

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 4) (h₂ : b = 5) :
  ∃ (c₁ c₂ : ℝ), triangle_third_side a b c₁ ∧ triangle_third_side a b c₂ ∧
  abs ((c₁ * c₂) - 19.2) < 0.1 :=
sorry

end product_third_side_approximation_l432_432321


namespace parallelepiped_side_lengths_l432_432338

theorem parallelepiped_side_lengths (x y z : ℕ) 
  (h1 : x + y + z = 17) 
  (h2 : 2 * x * y + 2 * y * z + 2 * z * x = 180) 
  (h3 : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 :=
by {
  sorry
}

end parallelepiped_side_lengths_l432_432338


namespace identifyTweedledee_l432_432768

structure House where
  brother1: Prop
  brother2: Prop

def standingNextToAlice (house: House): Prop := 
  -- brother stands silently next to Alice
  sorry -- replace with actual condition representing the brother standing next to Alice

def respondedWithCircle (house: House): Prop :=
  -- brother answers by drawing a circle in the air
  sorry -- replace with actual condition representing brother drawing circle

axiom squareOnSign (house: House): Prop :=
  -- statement representing that a square is drawn on the reverse side of the sign
  sorry -- whether true or false depending upon given condition

-- Lean Statement: Proving which brother is Tweedledee
theorem identifyTweedledee (house: House) (silentNextToAlice: ∀ h: House, standingNextToAlice h) (responseCircle: ∀ h: House, respondedWithCircle h) : Trulalala :=
  sorry

end identifyTweedledee_l432_432768


namespace transformed_parametric_curve_l432_432754
-- Importing Mathlib to bring in necessary libraries.

-- Defining the conditions 
def curve_condition (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 3 ∧ y' = y / 2

-- Stating the proof problem
theorem transformed_parametric_curve (θ : ℝ) :
  (transformation (3 * sqrt(3)/3 * cos θ) (2 * sqrt(2)/2 * sin θ) (sqrt(3)/3 * cos θ) (sqrt(2)/2 * sin θ)) → 
  (curve_condition (3 * sqrt(3)/3 * cos θ) (2 * sqrt(2)/2 * sin θ)) →
  3 * (sqrt(3)/3 * cos θ)^2 + 2 * (sqrt(2)/2 * sin θ)^2 = 1 := by
  sorry

end transformed_parametric_curve_l432_432754


namespace janet_gas_usage_l432_432602

theorem janet_gas_usage :
  ∀ (d_dermatologist d_gynecologist miles_per_gallon : ℕ),
    d_dermatologist = 30 →
    d_gynecologist = 50 →
    miles_per_gallon = 20 →
    (2 * d_dermatologist + 2 * d_gynecologist) / miles_per_gallon = 8 :=
by
  intros d_dermatologist d_gynecologist miles_per_gallon
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end janet_gas_usage_l432_432602


namespace geometry_biology_overlap_diff_l432_432774

theorem geometry_biology_overlap_diff :
  ∀ (total_students geometry_students biology_students : ℕ),
  total_students = 232 →
  geometry_students = 144 →
  biology_students = 119 →
  (max geometry_students biology_students - max 0 (geometry_students + biology_students - total_students)) = 88 :=
by
  intros total_students geometry_students biology_students
  sorry

end geometry_biology_overlap_diff_l432_432774


namespace solution_l432_432292

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def general_term (n r : ℕ) (x : ℚ) : ℚ :=
  (2 ^ r * binomial_coefficient n r * x ^ (5 - 5 * r / 2))

theorem solution (n : ℕ) :
  (∃ (r : ℕ), 
    (∀ k : ℕ, binomial_coefficient n k ≤ binomial_coefficient n 5) ∧ r = 5) ∧
    general_term 10 2 x = 180 :=
by
  existsi 5
  split
  {
    sorry -- Proof that n = 10
  }
  {
    sorry -- Proof for the constant term
  }

end solution_l432_432292


namespace inequality_proof_l432_432265

noncomputable def lean_math_statement (x y : ℝ) (p q : ℕ) : Prop :=
  x ≠ y ∧ x > 0 ∧ y > 0 → 
  (𝔼 : ℚ) (frac_pos: p > 0 ∧ q > 0)
  (x ^ (-(p / q)) - y ^ (p / q) * x ^ (-(2 * p / q)))
  / (x ^ ((1 - 2 * p) / q) - y ^ (1 / q) * x ^ (-(2 * p / q)))
  > (p:ℚ) * (x * y)^(ℚ (p - 1) / (2*v)).

theorem inequality_proof
  (x y : ℝ) (p q : ℕ)
  (hxy : x ≠ y)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hpq_pos : 0 < p ∧ 0 < q) :
  lean_math_statement x y p q :=
sorry

end inequality_proof_l432_432265


namespace janet_gas_usage_l432_432606

variable (distance_dermatologist distance_gynecologist mpg : ℕ)

theorem janet_gas_usage
  (h_distance_dermatologist : distance_dermatologist = 30)
  (h_distance_gynecologist : distance_gynecologist = 50)
  (h_mpg : mpg = 20) :
  (2 * distance_dermatologist + 2 * distance_gynecologist) / mpg = 8 := 
by
  rw [h_distance_dermatologist, h_distance_gynecologist, h_mpg]
  linarith
  sorry

end janet_gas_usage_l432_432606


namespace find_b_plus_m_l432_432180

section MatrixPower

open Matrix

-- Define our matrices
def A (b m : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, b], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3008], 
    ![0, 1, 45], 
    ![0, 0, 1]]

-- The problem statement
noncomputable def power_eq_matrix (b m : ℕ) : Prop :=
  (A b m) ^ m = B

-- The final goal
theorem find_b_plus_m (b m : ℕ) (h : power_eq_matrix b m) : b + m = 283 := sorry

end MatrixPower

end find_b_plus_m_l432_432180


namespace triangles_congruence_one_side_condition_l432_432085

/-- Given two triangles ΔABC and ΔDEF, if at least one side (corresponding segment) of ΔABC is equal
to the corresponding side of ΔDEF, we can conclude with the necessary condition that ΔABC and ΔDEF are congruent. -/
theorem triangles_congruence_one_side_condition (A B C D E F : ℝ) 
  (h1 : A = D ∨ B = E ∨ C = F) : 
  (∃ A B C, A = B ∧ B = C ∧ C = A) :=
sorry

end triangles_congruence_one_side_condition_l432_432085


namespace diff_sum_of_digits_l432_432118

theorem diff_sum_of_digits (N : ℕ) : 
  (∑ d in (digits 10 (N * (N - 1))), d) ≠ (∑ d in (digits 10 ((N + 1)^2)), d) :=
sorry

end diff_sum_of_digits_l432_432118


namespace eccentricity_of_hyperbola_is_two_l432_432058

-- Define the hyperbola and its properties
def hyperbola (a b x y : ℝ) : Prop := 
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the foci of the hyperbola
def foci (a : ℝ) : ℝ × ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in (-c, 0), (c, 0)

-- Define the slopes constraint for perpendicularity of GF₁ and GF₂
def slopes_perpendicular (a b x c : ℝ) : Prop :=
  ((b / a * x / (x + c)) * (b / a * x / (x - c)) = -1)

-- Define the hypotenuse and midpoint conditions
def midpoint_condition (a b c x : ℝ) : Prop :=
  let G := (a, b)
  let H := ((x, -b / a * x))
  (2 * H.1 = a - c) ∧ (2 * H.2 = b)

-- Prove the eccentricity is 2
theorem eccentricity_of_hyperbola_is_two :
  ∀ (a b : ℝ), 
  a > 0 → b > 0 →
  ∃ (c : ℝ), 
    let e := c / a in (hyperbola a b a b) ∧ 
    (∃ (G H : ℝ × ℝ), (slopes_perpendicular a b a c) ∧ (midpoint_condition a b c a) ∧ 
    (e = 2)) :=
by
  -- Continue with actual Lean proof here
  sorry

end eccentricity_of_hyperbola_is_two_l432_432058


namespace area_of_pentagon_l432_432429

theorem area_of_pentagon :
  ∀ (A B C D E : Type) (EA AB BC : ℝ) (CD DE : ℝ)
  (angleA : ℝ) (angleB : ℝ),
  angleA = 120 ∧ angleB = 120 ∧ EA = 3 ∧ AB = 3 ∧ BC = 3 ∧ CD = 2 ∧ DE = 2 →
  let area_triangle := (sqrt 3 / 4) * EA^2
  let area_small_triangle := (sqrt 3 / 4) * CD^2
  let total_area := area_triangle + 2 * area_small_triangle
  total_area = (17 * sqrt 3) / 4 :=
  
by
  intros A B C D E EA AB BC CD DE angleA angleB conditions
  have h1 : (sqrt 3 / 4) * 3^2 = 9 * sqrt 3 / 4, sorry
  have h2 : 2 * (sqrt 3 / 4) * 2^2 = 2 * sqrt 3, sorry
  have total_area := 9 * sqrt 3 / 4 + sqrt 3 * 2, sorry
  have h3 : total_area = 17 * sqrt 3 / 4, sorry
  exact h3

end area_of_pentagon_l432_432429


namespace solve_equation_l432_432090

theorem solve_equation (x : ℝ) (h1 : -1 < x) (h2 : x ≤ 2)
  (h_eq : sqrt (2 - x) + sqrt (2 + 2 * x) = sqrt ((x^4 + 1) / (x^2 + 1)) + (x + 3) / (x + 1)) :
  x = 1 :=
sorry

end solve_equation_l432_432090


namespace cat_litter_container_weight_l432_432597

theorem cat_litter_container_weight :
  (∀ (cost_container : ℕ) (pounds_per_litterbox : ℕ) (cost_total : ℕ) (days : ℕ),
    cost_container = 21 ∧ pounds_per_litterbox = 15 ∧ cost_total = 210 ∧ days = 210 → 
    ∀ (weeks : ℕ), weeks = days / 7 →
    ∀ (containers : ℕ), containers = cost_total / cost_container →
    ∀ (cost_per_container : ℕ), cost_per_container = cost_total / containers →
    (∃ (pounds_per_container : ℕ), pounds_per_container = cost_container / cost_per_container ∧ pounds_per_container = 3)) :=
by
  intros cost_container pounds_per_litterbox cost_total days
  intros h weeks hw containers hc containers_cost hc_cost
  sorry

end cat_litter_container_weight_l432_432597


namespace right_triangle_leg_length_l432_432190

theorem right_triangle_leg_length (a b c : ℕ) (h_c : c = 13) (h_a : a = 12) (h_pythagorean : a^2 + b^2 = c^2) :
  b = 5 := 
by {
  -- Provide a placeholder for the proof
  sorry
}

end right_triangle_leg_length_l432_432190


namespace evaluate_power_l432_432857

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l432_432857


namespace integer_solutions_to_equation_l432_432848

theorem integer_solutions_to_equation :
  ∀ (a b c : ℤ), a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integer_solutions_to_equation_l432_432848


namespace e_nonzero_l432_432725

-- Define the conditions of the problem
variables {a b c d e f : ℝ}
def Q (x : ℝ) : ℝ := x^6 + a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f
variables {p q r s t : ℝ}

-- State the specific conditions of the polynomial
axiom zero_at_zero : Q 0 = 0
axiom six_distinct_roots : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t

-- The theorem to prove
theorem e_nonzero (h : Q = λ x : ℝ, x * (x - p) * (x - q) * (x - r) * (x - s) * (x - t)) : e ≠ 0 :=
sorry

end e_nonzero_l432_432725


namespace no_negative_roots_l432_432083

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 :=
by sorry

end no_negative_roots_l432_432083


namespace equivalent_solutions_l432_432770

noncomputable def problem_statement (x y : ℝ) : Prop :=
  (5 * real.sqrt (x^2 - 3 * y - 88) + real.sqrt (x + 6 * y) = 19) ∧
  (3 * real.sqrt (x^2 - 3 * y - 88) = 1 + 2 * real.sqrt (x + 6 * y))

theorem equivalent_solutions :
  ∃ x y : ℝ, problem_statement x y ∧ ((x = 10 ∧ y = 1) ∨ (x = -21/2 ∧ y = 53/12)) :=
by 
  sorry

end equivalent_solutions_l432_432770


namespace knights_count_l432_432641

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432641


namespace brian_holds_breath_for_60_seconds_l432_432825

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end brian_holds_breath_for_60_seconds_l432_432825


namespace distance_to_right_focus_l432_432514

variable (F1 F2 P : ℝ × ℝ)
variable (a : ℝ)
variable (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / 9) + (P.2^2 / 8) = 1 })
variable (h_foci_dist : (P : ℝ × ℝ) → (F1 : ℝ × ℝ) → (F2 : ℝ × ℝ) → (dist P F1) = 2)
variable (semi_major_axis : a = 3)

theorem distance_to_right_focus (h : dist F1 F2 = 2 * a) : dist P F2 = 4 := 
sorry

end distance_to_right_focus_l432_432514


namespace two_times_koi_minus_X_is_64_l432_432322

-- Definitions based on the conditions
def n : ℕ := 39
def X : ℕ := 14

-- Main proof statement
theorem two_times_koi_minus_X_is_64 : 2 * n - X = 64 :=
by
  sorry

end two_times_koi_minus_X_is_64_l432_432322


namespace solution_l432_432531

def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^5 + p*x^3 + q*x - 8

theorem solution (p q : ℝ) (h : f (-2) p q = 10) : f 2 p q = -26 := by
  sorry

end solution_l432_432531


namespace poly_diff_independent_of_x_l432_432521

theorem poly_diff_independent_of_x (x y: ℤ) (m n : ℤ) 
  (h1 : (1 - n = 0)) 
  (h2 : (m + 3 = 0)) :
  n - m = 4 := by
  sorry

end poly_diff_independent_of_x_l432_432521


namespace moduli_product_l432_432465

theorem moduli_product (z1 z2 : ℂ) (h1 : z1 = 4 - 3 * complex.I) (h2 : z2 = 4 + 3 * complex.I) : complex.abs z1 * complex.abs z2 = 25 := 
by
  rw [h1, h2]
  -- simplify abs (4 - 3i) * abs (4 + 3i)
  have : |4 - 3*complex.I| * |4 + 3*complex.I| = complex.abs ((4 - 3*complex.I) * (4 + 3*complex.I)) := complex.abs_mul (4 - 3*complex.I) (4 + 3*complex.I)
  rw [this]
  -- (4 - 3i) * (4 + 3i) = 25
  have : (4 - 3*complex.I) * (4 + 3*complex.I) = 25 := by 
    rw [←complex.mul_conj, complex.norm_sq_eq_conj_mul_self]
    simp [complex.norm_sq]
  rw [this]
  -- the modulus of 25 is 25
  rw [complex.abs_assoc, complex.abs_of_real, complex.abs_eq_abs_of_nonneg]
  norm_num
  sorry

end moduli_product_l432_432465


namespace solve_tangent_equation_l432_432279

theorem solve_tangent_equation :
  ∃ x : ℝ, (tan x + tan (2 * x) + tan (3 * x) + tan (4 * x) = 0) ↔
  x ∈ {k * (π / 5) | k ∈ (Finset.range 5 : set ℤ)} :=
by
  sorry

end solve_tangent_equation_l432_432279


namespace number_writing_number_reading_l432_432385

def ten_million_place := 10^7
def hundred_thousand_place := 10^5
def ten_place := 10

def ten_million := 1 * ten_million_place
def three_hundred_thousand := 3 * hundred_thousand_place
def fifty := 5 * ten_place

def constructed_number := ten_million + three_hundred_thousand + fifty

def read_number := "ten million and thirty thousand and fifty"

theorem number_writing : constructed_number = 10300050 := by
  -- Sketch of proof goes here based on place values
  sorry

theorem number_reading : read_number = "ten million and thirty thousand and fifty" := by
  -- Sketch of proof goes here for the reading method
  sorry

end number_writing_number_reading_l432_432385


namespace indefinite_integral_partial_fraction_l432_432417

theorem indefinite_integral_partial_fraction :
  ∫ (x : ℝ) in set.univ, (x^3 - 6*x^2 + 11*x - 10) / ((x + 2) * (x - 2)^3) = 
  (λ x, log (real.abs (x + 2)) + 1 / (2 * (x - 2)^2) + C) :=
sorry

end indefinite_integral_partial_fraction_l432_432417


namespace product_discount_l432_432391

theorem product_discount (P : ℝ) (h₁ : P > 0) :
  let price_after_first_discount := 0.7 * P
  let price_after_second_discount := 0.8 * price_after_first_discount
  let total_reduction := P - price_after_second_discount
  let percent_reduction := (total_reduction / P) * 100
  percent_reduction = 44 :=
by
  sorry

end product_discount_l432_432391


namespace gcd_min_value_l432_432183

theorem gcd_min_value {a b c : ℕ} (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (gcd_ab : Nat.gcd a b = 210) (gcd_ac : Nat.gcd a c = 770) : Nat.gcd b c = 10 :=
sorry

end gcd_min_value_l432_432183


namespace max_product_of_distances_l432_432922

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432922


namespace max_area_triangle_l432_432570

theorem max_area_triangle (PQ PR : ℝ → ℝ → ℝ) (h1 : PQ • PR = 7) (h2 : |PQ - PR| = 6) :  
  let m := |PQ|
  let n := |PR|
  S = 1/2 * sqrt ( (m * n)^2 - 49 ) ≤ 12 :=
by sorry

end max_area_triangle_l432_432570


namespace marble_problem_l432_432017

theorem marble_problem : Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7 = 210 := by
  sorry

end marble_problem_l432_432017


namespace smallest_integer_greater_than_100_with_gcd_24_eq_4_l432_432334

theorem smallest_integer_greater_than_100_with_gcd_24_eq_4 :
  ∃ x : ℤ, x > 100 ∧ x % 24 = 4 ∧ (∀ y : ℤ, y > 100 ∧ y % 24 = 4 → x ≤ y) :=
sorry

end smallest_integer_greater_than_100_with_gcd_24_eq_4_l432_432334


namespace least_positive_integer_satisfying_conditions_l432_432761

theorem least_positive_integer_satisfying_conditions : 
  ∃ x : ℕ, (0 < x) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  (x % 9 = 8) ∧ 
  (∀ y : ℕ, (0 < y) ∧ (y % 4 = 3) ∧ (y % 5 = 4) ∧ (y % 7 = 6) ∧ (y % 9 = 8) → x ≤ y) :=
begin
  use 1259,
  split,
  { norm_num }, -- Proving that 1259 > 0
  split,
  { norm_num }, -- Proving that 1259 % 4 = 3
  split,
  { norm_num }, -- Proving that 1259 % 5 = 4
  split,
  { norm_num }, -- Proving that 1259 % 7 = 6
  split,
  { norm_num }, -- Proving that 1259 % 9 = 8
  intros y hy,
  obtain ⟨hy0, hy1, hy2, hy3, hy4⟩ := hy,
  have H : ∀ d : ℕ, (d | 1260) ∧ (d | y - 1259) → d = 1, from sorry, -- gcd and lcm related proof (skipped)
  by_contradiction,
  norm_num at a,
  sorry, -- additional computations or lemmas can be required.
end

end least_positive_integer_satisfying_conditions_l432_432761


namespace evaluate_fractional_parts_l432_432728

def integer_sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ 
  a 2 = 2 ∧ 
  ∀ n, a (n + 2) = 5 * a (n + 1) + a n

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem evaluate_fractional_parts :
  ∀ (a : ℕ → ℕ), 
  integer_sequence a -> 
  (∏ i in (finset.range 2024).map (function.embedding.nat_add 1), fractional_part (a (i + 2) / a (i + 1))) = 1 :=
by
  sorry

end evaluate_fractional_parts_l432_432728


namespace train_length_spec_l432_432398

noncomputable def train_length  {V : ℝ} (h1 : V > 0) (h2 : ∀ L, L = V * 20 → 2 * L = V * 40) : ℝ :=
20 * V

theorem train_length_spec  {V : ℝ} (h1 : V > 0) (h2 : ∀ L, L = V * 20 → 2 * L = V * 40) :
  train_length h1 h2 = 20 * V :=
by {
  unfold train_length,
  sorry
}

end train_length_spec_l432_432398


namespace no_upper_bound_exists_for_expression_l432_432662

theorem no_upper_bound_exists_for_expression 
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_sum : x + y + z = 9) : 
  ∀ M : ℝ, ∃ (x y z : ℝ), 
    0 < x ∧ 0 < y ∧ 0 < z ∧ 
    x + y + z = 9 ∧ 
    (frac_expr := (x^2 + 2 * y^2) / (x + y) + (2 * x^2 + z^2) / (x + z) + (y^2 + 2 * z^2) / (y + z)),
    frac_expr > M := 
begin
  sorry
end

end no_upper_bound_exists_for_expression_l432_432662


namespace calculate_025_percent_l432_432362

theorem calculate_025_percent (n : ℕ) (h : n = 16) : 0.25 / 100 * n = 0.04 := by
  rw [← h]
  suffices 0.25 / 100 * 16 = 0.04 by
    exact this
  rw [div_mul_eq_mul_div, div_self 100, one_mul]
  norm_num
  exact 0.0025 * 16 = 0.04

end calculate_025_percent_l432_432362


namespace max_sum_cos_isosceles_triangle_l432_432762

theorem max_sum_cos_isosceles_triangle :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (2 * Real.cos α + Real.cos (π - 2 * α)) ≤ 1.5 :=
by
  sorry

end max_sum_cos_isosceles_triangle_l432_432762


namespace circle_equation_l432_432930

theorem circle_equation {a b c : ℝ} (hc : c ≠ 0) :
  ∃ D E F : ℝ, 
    (D = -(a + b)) ∧
    (E = - (c + ab / c)) ∧ 
    (F = ab) ∧
    ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 :=
sorry

end circle_equation_l432_432930


namespace moles_of_NH4Cl_combined_l432_432485

-- Define the chemical reaction equation
def reaction (NH4Cl H2O NH4OH HCl : ℕ) := 
  NH4Cl + H2O = NH4OH + HCl

-- Given conditions
def condition1 (H2O : ℕ) := H2O = 1
def condition2 (NH4OH : ℕ) := NH4OH = 1

-- Theorem statement: Prove that number of moles of NH4Cl combined is 1
theorem moles_of_NH4Cl_combined (H2O NH4OH NH4Cl HCl : ℕ) 
  (h1: condition1 H2O) (h2: condition2 NH4OH) (h3: reaction NH4Cl H2O NH4OH HCl) : 
  NH4Cl = 1 :=
sorry

end moles_of_NH4Cl_combined_l432_432485


namespace vectors_collinear_has_solution_l432_432545

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x^2 - 1, 2 + x)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinearity condition (cross product must be zero) as a function
def collinear (x : ℝ) : Prop := (a x).1 * (b x).2 - (b x).1 * (a x).2 = 0

-- The proof statement
theorem vectors_collinear_has_solution (x : ℝ) (h : collinear x) : x = -1 / 2 :=
sorry

end vectors_collinear_has_solution_l432_432545


namespace lisa_kay_clean_room_together_l432_432242

section CleaningTime

variable (lisa_time kay_time : ℝ) (lisa_rate kay_rate combined_rate together_time : ℝ)

-- Define the conditions
def lisa_cleaning_time := lisa_time = 8
def kay_cleaning_time := kay_time = 12
def lisa_cleaning_rate := lisa_rate = 1 / lisa_time
def kay_cleaning_rate := kay_rate = 1 / kay_time
def combined_cleaning_rate := combined_rate = lisa_rate + kay_rate
def time_to_clean_together := together_time = 1 / combined_rate

-- Define the final statement to prove
theorem lisa_kay_clean_room_together :
  ∀ (lisa_time kay_time : ℝ),
    lisa_cleaning_time lisa_time → kay_cleaning_time kay_time →
    ∃ (together_time : ℝ),
      time_to_clean_together lisa_time kay_time (1 / lisa_time) (1 / kay_time) (1 / lisa_time + 1 / kay_time) together_time ∧
      together_time = 4.8 :=
by
  intros lisa_time kay_time hl hk
  use 4.8
  split
  sorry
  sorry

end CleaningTime

end lisa_kay_clean_room_together_l432_432242


namespace factor_expression_l432_432874

theorem factor_expression (x y : ℤ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end factor_expression_l432_432874


namespace form_maltese_cross_l432_432054

-- Define a structure for a four-pointed star
structure FourPointedStar where
  center    : Point    -- Center point where all lines meet
  vertices  : Fin 4 → Point  -- Four vertices defining the star
  deriving Inhabited

-- Define a function to cut the star into four equal parts
def cut_star (star : FourPointedStar) : Fin 4 → Subset Point
| 0 => { point | ∃ line, line passing through star.center delivers the top to bottom cut }
| 1 => { point | ∃ line, line passing through star.center delivers the left to right cut }
| 2 => { point | ∃ line, line passing through star.center delivers the bottom to top cut }
| 3 => { point | ∃ line, line passing through star.center delivers the right to left cut }

-- Define a function to arrange the cut parts in the frame
def arrange_in_frame (parts : Fin 4 → Subset Point) : Frame → Set Point := sorry

-- Define the Maltese cross as a recognizable shape
def is_maltese_cross (points : Set Point) : Prop := sorry

-- Lean 4 statement proving the result
theorem form_maltese_cross (star : FourPointedStar) (frame : Frame) :
  is_maltese_cross (arrange_in_frame (cut_star star) frame) := sorry

end form_maltese_cross_l432_432054


namespace integer_solutions_sum_l432_432893

theorem integer_solutions_sum (x : ℤ) (h : x^4 - 13 * x^2 + 36 = 0) : 
(sum (filter (λ x, x^4 - 13 * x^2 + 36 = 0) (finset.range 7))) = 0 :=
sorry

end integer_solutions_sum_l432_432893


namespace molecular_weight_of_Carbonic_acid_l432_432829

theorem molecular_weight_of_Carbonic_acid :
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  (H_atoms * H_weight + C_atoms * C_weight + O_atoms * O_weight) = 62.024 :=
by 
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  sorry

end molecular_weight_of_Carbonic_acid_l432_432829


namespace ratio_of_ants_Duke_to_Abe_l432_432810

-- Declare the conditions as hypotheses
variables (n_Abe n_Beth n_CeCe n_Duke total_find : ℕ)
hypothesis hAbe : n_Abe = 4
hypothesis hBeth : n_Beth = n_Abe + n_Abe / 2
hypothesis hCeCe : n_CeCe = 2 * n_Abe
hypothesis hTotal : n_Abe + n_Beth + n_CeCe + n_Duke = 20

-- State the theorem with the required ratio
theorem ratio_of_ants_Duke_to_Abe : 
  n_Duke / n_Abe = 1 / 2 :=
by
  sorry

end ratio_of_ants_Duke_to_Abe_l432_432810


namespace sum_first_2016_terms_l432_432891

def sequence (n : ℕ) : ℤ := (-1) ^ n * n

def partial_sum (m : ℕ) : ℤ :=
  (Finset.range m).sum (λ n, sequence (n + 1))

theorem sum_first_2016_terms :
  partial_sum 2016 = 1008 :=
by
  sorry

end sum_first_2016_terms_l432_432891


namespace find_b_l432_432522

-- Given definitions and conditions
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b
def point : ℝ × ℝ := (1, 3)

/-- Prove that the value of b in the curve equation is -3,
    given that the line y = kx + 1 is tangent to the curve 
    y = x^3 + ax + b at the point (1, 3). -/
theorem find_b (k a b : ℝ) 
  (h1 : line k 1 = 3)
  (h2 : curve a b 1 = 3)
  (h3 : deriv (λ x, curve a b x) 1 = k) : 
  b = -3 := 
sorry

end find_b_l432_432522


namespace sum_highest_powers_dividing_factorial_l432_432081

theorem sum_highest_powers_dividing_factorial (n : ℕ) (fact_n : ℕ) (highest_10 highest_8 sum_powers : ℕ)
  (fact_n_def : fact_n = 20!)
  (highest_10_def : highest_10 = min (20 / 2 + 20 / 4 + 20 / 8 + 20 / 16) (20 / 5) )
  (highest_8_def : highest_8 = 18 / 3 )
  (sum_powers_def : sum_powers = highest_10 + highest_8 ) :
  sum_powers = 10 := 
sorry

end sum_highest_powers_dividing_factorial_l432_432081


namespace factorial_division_l432_432564

def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_division : factorial 100 / factorial 98 = 9900 := by
  have h1 : factorial 100 = 100 * 99 * factorial 98 := by sorry
  rw [h1]
  have h2 : (100 * 99 * factorial 98) / factorial 98 = 100 * 99 := by sorry
  rw [h2]
  norm_num
  exact sorry

end factorial_division_l432_432564


namespace quadratic_root_properties_l432_432324

-- Lean statement to show the properties of roots of a quadratic equation.
theorem quadratic_root_properties :
  (∃ (x1 x2 : ℝ), 2 * x1^2 + 3 * x1 - 1 = 0 ∧ 2 * x2^2 + 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  let sum_of_squares := λ x1 x2 : ℝ, x1^2 + x2^2 in
  let sum_of_reciprocals := λ x1 x2 : ℝ, (1 / x1) + (1 / x2) in
  (∀ (x1 x2 : ℝ), sum_of_squares x1 x2 = 13 / 4) ∧
  (∀ (x1 x2 : ℝ), sum_of_reciprocals x1 x2 = -3) :=
by {
  sorry
}

end quadratic_root_properties_l432_432324


namespace radius_of_circle_B_l432_432424

theorem radius_of_circle_B (r_A r_D : ℝ) (r_B : ℝ) (hA : r_A = 2) (hD : r_D = 4) 
  (congruent_BC : r_B = r_B) (tangent_condition : true) -- placeholder conditions
  (center_pass : true) -- placeholder conditions
  : r_B = (4 / 3) * (Real.sqrt 7 - 1) :=
sorry

end radius_of_circle_B_l432_432424


namespace knights_count_l432_432619

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l432_432619


namespace abc_divisibility_l432_432477

theorem abc_divisibility (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) : 
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by {
  sorry  -- proof to be filled in
}

end abc_divisibility_l432_432477


namespace circle_radius_five_d_value_l432_432097

theorem circle_radius_five_d_value :
  ∀ (d : ℝ), (∃ (x y : ℝ), (x - 4)^2 + (y + 5)^2 = 41 - d) → d = 16 :=
by
  intros d h
  sorry

end circle_radius_five_d_value_l432_432097


namespace circle_construction_l432_432165

-- Definitions and conditions
variables {Point Line Circle : Type} [Geometry Point Line Circle]
variables (a b : Line) (P : Point)
variables (on_line_b : Point → Line → Prop)
variables (center_lies_on_b : Circle → Line → Prop)
variables (circle_passes_through : Circle → Point → Prop)
variables (circle_is_tangent_to : Circle → Line → Prop)

-- Intersecting lines condition
axiom intersecting_lines (a b : Line) : ∃ Q : Point, on_line_b Q a ∧ on_line_b Q b

-- Point P is on line b condition
axiom point_on_line_b (P : Point) (b : Line) : on_line_b P b

-- Proof problem statement
theorem circle_construction :
  ∃ O₁ O₂ : Point,
  (on_line_b O₁ b ∧ on_line_b O₂ b) ∧
  (∃ C₁ C₂ : Circle,
    (center_lies_on_b C₁ b ∧ center_lies_on_b C₂ b) ∧ 
    (circle_passes_through C₁ P ∧ circle_passes_through C₂ P) ∧ 
    (circle_is_tangent_to C₁ a ∧ circle_is_tangent_to C₂ a)) :=
sorry

end circle_construction_l432_432165


namespace jar_water_transfer_l432_432776

theorem jar_water_transfer
  (C_x : ℝ) (C_y : ℝ)
  (h1 : C_y = 1/2 * C_x)
  (WaterInX : ℝ)
  (WaterInY : ℝ)
  (h2 : WaterInX = 1/2 * C_x)
  (h3 : WaterInY = 1/2 * C_y) :
  WaterInX + WaterInY = 3/4 * C_x :=
by
  sorry

end jar_water_transfer_l432_432776


namespace initial_apples_correct_l432_432046

def apples_initial (x : ℕ) : Prop := 
  (2 * x) / 3 - 9 = 4

theorem initial_apples_correct : ∃ x : ℕ, apples_initial x := 
begin
  use 22,
  exact calc
    (2 * 22 : ℕ) / 3 - 9 = 44 / 3 - 9 : by simp
                     ...      = 14 - 9 : by norm_num 44 / 3 = 14
                     ...      = 4       : by simp
end

end initial_apples_correct_l432_432046


namespace problem_statement_l432_432655

noncomputable def a : ℝ := -0.5
noncomputable def b : ℝ := (1 + Real.sqrt 3) / 2

theorem problem_statement
  (h1 : a^2 = 9 / 36)
  (h2 : b^2 = (1 + Real.sqrt 3)^2 / 8)
  (h3 : a < 0)
  (h4 : b > 0) :
  ∃ (x y z : ℤ), (a - b)^2 = x * Real.sqrt y / z ∧ (x + y + z = 6) :=
sorry

end problem_statement_l432_432655


namespace find_a_l432_432489

theorem find_a (a : ℝ) (x : ℝ) (hx : x ≠ 0): 
  ((a + 1/x) * (1 + x)^4).coeff 2 = 0 → a = -2/3 :=
by
  sorry

end find_a_l432_432489


namespace star_eq_zero_iff_x_eq_5_l432_432076

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end star_eq_zero_iff_x_eq_5_l432_432076


namespace knights_count_in_meeting_l432_432629

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432629


namespace binary_operation_result_l432_432094

theorem binary_operation_result :
  let b1 := 11011
  let b2 := 101
  let b_sub := 1010
  let dec_b1 := 27  -- decimal equivalent of 11011_2
  let dec_b2 := 5   -- decimal equivalent of 101_2
  let dec_b_sub := 10  -- decimal equivalent of 1010_2
  let product_decimal := dec_b1 * dec_b2
  let result_decimal := product_decimal - dec_b_sub
  let result_binary := Nat.toDigits 2 result_decimal
  result_binary = [1, 1, 1, 1, 1, 0, 1] :=
by
  sorry

end binary_operation_result_l432_432094


namespace eccentricity_of_hyperbola_is_sqrt3_plus_1_l432_432960

-- Define the hyperbola with given conditions
def hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1 }

noncomputable def eccentricity (a b c : ℝ) (e : ℝ) := c / a = e

-- Main theorem to be proved
theorem eccentricity_of_hyperbola_is_sqrt3_plus_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let c := Real.sqrt (a^2 + b^2) in
  ∃ e : ℝ, c / a = e ∧ e = 1 + Real.sqrt 3 :=
sorry

end eccentricity_of_hyperbola_is_sqrt3_plus_1_l432_432960


namespace solution_set_eq_zero_l432_432306

theorem solution_set_eq_zero (x : ℝ) :
  (4^x + 4^(-x)) - 2 * (2^x + 2^(-x)) + 2 = 0 ↔ x = 0 :=
by sorry

end solution_set_eq_zero_l432_432306


namespace inequality_part_a_inequality_part_b_l432_432428

-- Part (a)
theorem inequality_part_a (x y z : ℝ) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * sqrt 2 * sqrt (x^2 + y^2 + z^2) :=
sorry

-- Part (b)
theorem inequality_part_b (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * sqrt (x^2 + y^2 + z^2) :=
sorry

end inequality_part_a_inequality_part_b_l432_432428


namespace square_QP_l432_432320

theorem square_QP 
  (r₁ r₂ d : ℝ)
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 15)
  (h₃ : d = 20)
  (P : {P : ℝ | (P - {C₁ : ℝ | (C₁ - O₁).norm = r₁}).norm = 0 ∧ (P - {C₂ : ℝ | (C₂ - O₂).norm = r₂}).norm = 0})
  (Q R : ℝ)
  (h₄ : (Q - {C₁ : ℝ | (C₁ - O₁).norm = r₁}).norm = 0)
  (h₅ : (R - {C₂ : ℝ | (C₂ - O₂).norm = r₂}).norm = 0)
  (h₆ : tangent (first_circle Q) P)
  (h₇ : tangent (second_circle R) P)
  (h₈ : (Q - P).norm = (R - P).norm) :
  (Q - P).norm ^ 2 = 4375 / 144 := sorry

noncomputable def first_circle (Q : ℝ) := {C₁ : ℝ | (C₁ - O₁).norm = 10}
noncomputable def second_circle (R : ℝ) := {C₂ : ℝ | (C₂ - O₂).norm = 15}

-- Assumes tangent line properties, may need auxiliary lemma definitions for those if necessary.

end square_QP_l432_432320


namespace weight_of_bag_l432_432226

-- Definitions
def chicken_price : ℝ := 1.50
def bag_cost : ℝ := 2
def feed_per_chicken : ℝ := 2
def profit_from_50_chickens : ℝ := 65
def total_chickens : ℕ := 50

-- Theorem
theorem weight_of_bag : 
  (bag_cost / (profit_from_50_chickens - 
               (total_chickens * chicken_price)) / 
               (feed_per_chicken * total_chickens)) = 20 := 
sorry

end weight_of_bag_l432_432226


namespace andy_start_problem_number_l432_432820

theorem andy_start_problem_number : 
  ∀ (n : ℕ), (n + 50 = 125) → (n = 75) :=
by {
  assume n,
  assume h : n + 50 = 125,
  sorry
}

end andy_start_problem_number_l432_432820


namespace change_order_of_integration_l432_432421

-- Define the function f
variable (f : ℝ → ℝ → ℝ)

-- Define the integral I
def I (f : ℝ → ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, ∫ y in 0..(x^2), f x y + ∫ x in 1..(real.sqrt 2), ∫ y in 0..(real.sqrt (2 - x^2)), f x y

-- Define the new integral after changing the order of integration
def I_changed (f : ℝ → ℝ → ℝ) : ℝ :=
  ∫ y in 0..1, ∫ x in (real.sqrt y)..(real.sqrt (2 - y^2)), f x y

-- Mathematically equivalent proof problem
theorem change_order_of_integration :
  I f = I_changed f := sorry

end change_order_of_integration_l432_432421


namespace find_lambda_find_a_l432_432955

open Real

noncomputable def f (ω λ x : ℝ) : ℝ := sin(ω * x) + λ * cos(ω * x)

theorem find_lambda (ω x : ℝ) (λ : ℝ) (h1 : ω = 2) (h2 : x = π / 12) :
  let φ := arctan λ in 
  λ = √3 :=
by
  sorry

noncomputable def g (a : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := a * f x + cos (4 * x - π / 3)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = 2 * sin(2 * x + π / 3)) 
  (hs : ∀ x ∈ Ioo (π / 4) (π / 3), g a f x > g a f (x - 1)) :
  a ≤ -1 :=
by
  sorry

end find_lambda_find_a_l432_432955


namespace vertices_set_a_vertices_set_b_l432_432166

variables {Point : Type} [MetricSpace Point]

-- Define points O and M
variables (O M A : Point)

-- Define circle1 with diameter LM and circle2 with diameter MK
def circle1 : set Point := sorry  -- circle constructed on LM as diameter
def circle2 : set Point := sorry  -- circle constructed on MK as diameter
def circle3 : set Point := sorry  -- circle constructed on LK as diameter

-- Conditions for part (a)
def is_vertex_of_triangle (A : Point) : Prop :=
  (A ∉ circle1) ∧ (A ∉ circle2)

-- Conditions for part (b)
def is_vertex_of_obtuse_triangle (A : Point) : Prop :=
  (A ∈ circle3) ∧ (A ∉ circle1) ∧ (A ∉ circle2)

-- Part (a) statement
theorem vertices_set_a :
  ∃ A, is_vertex_of_triangle O M A :=
sorry

-- Part (b) statement
theorem vertices_set_b :
  ∃ A, is_vertex_of_obtuse_triangle O M A :=
sorry

end vertices_set_a_vertices_set_b_l432_432166


namespace bedroom_light_energy_usage_l432_432689

-- Define the conditions and constants
def noahs_bedroom_light_usage (W : ℕ) : ℕ := W
def noahs_office_light_usage (W : ℕ) : ℕ := 3 * W
def noahs_living_room_light_usage (W : ℕ) : ℕ := 4 * W
def total_energy_used (W : ℕ) : ℕ := 2 * (noahs_bedroom_light_usage W + noahs_office_light_usage W + noahs_living_room_light_usage W)
def energy_consumption := 96

-- The main theorem to be proven
theorem bedroom_light_energy_usage : ∃ W : ℕ, total_energy_used W = energy_consumption ∧ W = 6 :=
by
  sorry

end bedroom_light_energy_usage_l432_432689


namespace min_abs_phi_l432_432721

theorem min_abs_phi {φ : ℝ} (φ_transformed : ℝ) :
  let y := λ x, Real.sin (2 * x + φ)
  let y_translated := λ x, Real.sin (2 * x + φ_transformed)
  (φ_transformed = φ + π/3) -- Function translated by π/6 units to the left, so φ_transformed = φ + π/3
  (∀ x₀ x₁, y_translated x₀ = 0 ∧ y_translated x₁ = 0 ∧ x₀ = -x₁) -- Equidistant zeros
  -> |φ| = π/6 := 
sorry

end min_abs_phi_l432_432721


namespace point_in_fourth_quadrant_l432_432207

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l432_432207


namespace line_intersects_y_axis_l432_432411

theorem line_intersects_y_axis :
  ∃ y : ℝ, 4 * y - 5 * 0 = 20 ∧ (0, y) = (0, 5) :=
by
  use 5
  simp
  split
  { norm_num }
  { refl }

end line_intersects_y_axis_l432_432411


namespace equilateral_triangles_square_math_proof_equility_l432_432711

def is_equilateral_triangle (vertices : List (ℕ × ℕ)) (side_length : ℕ) : Prop :=
  -- Definition of equilateral triangle with given side length
  sorry

def is_square (vertices : List (ℕ × ℕ)) (side_length : ℕ) : Prop :=
  -- Definition of square with given side length
  sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem equilateral_triangles_square (m n : ℕ) (x y : ℝ) (A B : ℝ × ℝ)
  (h1 : distance A B = real.sqrt x + real.sqrt y)
  (h2 : x = 32)
  (h3 : y = 48) : 
  x + y = 80 :=
by
  rw [h2, h3]
  simp
  exact rfl

def problem_conditions : Prop :=
  ∃ (A B : ℝ × ℝ), 
  ∃ (sq_vertices eq1_vertices eq2_vertices : List (ℕ × ℕ)), 
  is_square sq_vertices 4 ∧ 
  is_equilateral_triangle eq1_vertices 4 ∧ 
  is_equilateral_triangle eq2_vertices 4 ∧ 
  distance A B = real.sqrt 32 + real.sqrt 48

theorem math_proof_equility : problem_conditions → 80 =
  32 + 48 :=
by
  intro cond
  cases cond with A cond'
  cases cond' with B cond''
  cases cond'' with sv cond'''
  cases cond''' with ev1 cond''''
  cases cond'''' with ev2 rest
  cases rest with h1 rest'
  cases rest' with h2 rest''
  cases rest'' with h3 dAB
  apply equilateral_triangles_square (32) (48) 32 48 A B
  repeat { assumption } -- Instantiates assumptions to conditions
  exact rfl 
  simp
  exact rfl

end equilateral_triangles_square_math_proof_equility_l432_432711


namespace max_product_distances_l432_432919

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l432_432919


namespace pat_profits_l432_432691

noncomputable def profit (earnings_per_shark : ℕ) (shark_per_minute : ℕ) (fuel_cost_per_hour : ℕ) (hours : ℕ) : ℕ :=
let minutes_per_hour := 60,
    sharks_per_hour := minutes_per_hour / shark_per_minute * hours,
    earnings := sharks_per_hour * earnings_per_shark,
    fuel_cost := fuel_cost_per_hour * hours
in earnings - fuel_cost

theorem pat_profits (earnings_per_shark : ℕ) (shark_per_minute : ℕ) (fuel_cost_per_hour : ℕ) (hours : ℕ) :
  earnings_per_shark = 15 → shark_per_minute = 10 → fuel_cost_per_hour = 50 → hours = 5 →
  profit earnings_per_shark shark_per_minute fuel_cost_per_hour hours = 200 :=
by intros h1 h2 h3 h4; subst_vars; unfold profit; sorry

end pat_profits_l432_432691


namespace geometric_number_difference_l432_432051

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end geometric_number_difference_l432_432051


namespace photo_count_correct_l432_432854

variable (x : ℕ)

variable (Claire : ℕ) (Lisa : ℕ) (Robert : ℕ) (David : ℕ) (Emma : ℕ)
variable hClaire : Claire = x
variable hLisa : Lisa = 3 * x
variable hRobert : Robert = x + 10
variable hDavid : David = 2 * x - 5
variable hEmma : Emma = 2 * (x + 10)
variable hTotalPhotos : Claire + Lisa + Robert + David + Emma = 350

theorem photo_count_correct :
  x + 3*x + (x + 10) + (2*x - 5) + (2*x + 20) = 350 :=
by
  rw [← hClaire, ← hLisa, ← hRobert, ← hDavid, ← hEmma]
  exact hTotalPhotos

end photo_count_correct_l432_432854


namespace distance_C_to_line_AB_l432_432970

-- Definitions of the points A, B, and C in space
def A : ℝ³ := (-1, 0, 0)
def B : ℝ³ := (0, 1, -1)
def C : ℝ³ := (-1, -1, 2)

-- Defining the distance function from point to line
def distance_point_to_line (C A B : ℝ³) : ℝ :=
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let AC_len := real.sqrt (AC.1^2 + AC.2^2 + AC.3^2) in
  let AB_len := real.sqrt (AB.1^2 + AB.2^2 + AB.3^2) in
  let AC_dot_AB := AC.1 * AB.1 + AC.2 * AB.2 + AC.3 * AB.3 in
  real.sqrt (AC_len^2 - (AC_dot_AB / AB_len)^2)

-- Lean 4 statement to prove
theorem distance_C_to_line_AB :
  distance_point_to_line C A B = real.sqrt 2 :=
by
  -- Placeholder for the proof
  sorry

end distance_C_to_line_AB_l432_432970


namespace jack_finished_earlier_l432_432221

theorem jack_finished_earlier (jack_first_half_time : ℕ) (jack_second_half_time : ℕ) (jill_total_time : ℕ) :
    jack_first_half_time = 19 → 
    jack_second_half_time = 6 → 
    jill_total_time = 32 →
    (jill_total_time - (jack_first_half_time + jack_second_half_time) = 7) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end jack_finished_earlier_l432_432221


namespace total_payment_divisible_by_25_l432_432039

theorem total_payment_divisible_by_25 (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) : 
  (2005 + B * 1000) % 25 = 0 :=
by
  sorry

end total_payment_divisible_by_25_l432_432039


namespace set_intersection_l432_432968

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | 0 < 2^x ∧ 2^x < 1}
noncomputable def B := {x : ℝ | Real.log x / Real.log 3 > 0}
noncomputable def complement_U_B := {x : ℝ | ¬(x ∈ B)}

theorem set_intersection :
  A ∩ complement_U_B = {x : ℝ | x < 0} :=
by
  sorry

end set_intersection_l432_432968


namespace sequence_sum_1999_l432_432160

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n+2 => (a n * a (n+1) + 1) / (a n * a (n+1) - 1)

def S (n : ℕ) : ℕ := ∑ i in Finset.range (n+1), a i

theorem sequence_sum_1999 :
  S 1998 = 3997 :=
by 
  sorry

end sequence_sum_1999_l432_432160


namespace product_greater_than_constant_l432_432954

noncomputable def f (x m : ℝ) := Real.log x - (m + 1) * x + (1 / 2) * m * x ^ 2
noncomputable def g (x m : ℝ) := Real.log x - (m + 1) * x

variables {x1 x2 m : ℝ} 
  (h1 : g x1 m = 0)
  (h2 : g x2 m = 0)
  (h3 : x2 > Real.exp 1 * x1)

theorem product_greater_than_constant :
  x1 * x2 > 2 / (Real.exp 1 - 1) :=
sorry

end product_greater_than_constant_l432_432954


namespace infinite_product_eq_9_over_5_l432_432442

noncomputable def a : ℕ → ℚ
| 0       := 2 / 3
| (n + 1) := 1 + (a n - 1)^2

theorem infinite_product_eq_9_over_5 :
  (\(n : ℕ), a n).infinite_product = 9 / 5 :=
by
  sorry

end infinite_product_eq_9_over_5_l432_432442


namespace length_of_segment_XZ_l432_432781

noncomputable def radius_of_circle (C : ℝ) := C / (2 * Real.pi)

theorem length_of_segment_XZ
  (C : ℝ)
  (T : Type) [c : circle T]
  (XY : segment T)
  (h1 : circumference c = 12 * Real.pi)
  (h2 : XY ∈ diameter c)
  (TX TZ : segment T)
  (h3 : ∠ (TX, TZ) = 45) :
  length (XZ) = 6 * Real.sqrt 2 :=
by
  -- Definitions and calculations can be added here if needed.
  sorry

end length_of_segment_XZ_l432_432781


namespace lowest_possible_number_of_students_l432_432347

theorem lowest_possible_number_of_students :
  ∃ n : ℕ, (n % 12 = 0 ∧ n % 24 = 0) ∧ ∀ m : ℕ, ((m % 12 = 0 ∧ m % 24 = 0) → n ≤ m) :=
sorry

end lowest_possible_number_of_students_l432_432347


namespace problem_statement_l432_432162

noncomputable def U : Set Int := {-2, -1, 0, 1, 2}
noncomputable def A : Set Int := {x : Int | -2 ≤ x ∧ x < 0}
noncomputable def B : Set Int := {x : Int | (x = 0 ∨ x = 1)} -- since natural numbers typically include positive integers, adapting B contextually

theorem problem_statement : ((U \ A) ∩ B) = {0, 1} := by
  sorry

end problem_statement_l432_432162


namespace perpendicular_intersect_CD_l432_432194

noncomputable def angle_ABC : ℝ := 150 -- angle ABC is 150 degrees
noncomputable def AB : ℝ := 5 -- length of AB is 5
noncomputable def BC : ℝ := 6 -- length of BC is 6
noncomputable def AC : ℝ := Real.sqrt (25 + 36 + 30 * Real.sqrt 3) -- length of AC using Law of Cosines

-- Function calculating CD
noncomputable def CD : ℝ := (AC * (Real.sqrt 6 - Real.sqrt 2)) / 4 

theorem perpendicular_intersect_CD
  (D : ℝ)
  (H1 : ∃ (A B C : ℝ), angle_ABC = 150 ∧ AB = 5 ∧ BC = 6)
  (H2 : ∃ (perpendicular_from_A_to_AB perpendicular_from_C_to_BC), 
         intersection_at D) : 
  CD = (Real.sqrt (61 + 30 * Real.sqrt 3) * (Real.sqrt 6 - Real.sqrt 2)) / 4 := 
  sorry

end perpendicular_intersect_CD_l432_432194


namespace evaluate_expression_l432_432515

variable (m n p : ℝ)

theorem evaluate_expression 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := 
by 
  sorry

end evaluate_expression_l432_432515


namespace moduli_product_l432_432468

theorem moduli_product (z1 z2 : ℂ) (h1 : z1 = 4 - 3 * complex.I) (h2 : z2 = 4 + 3 * complex.I) : complex.abs z1 * complex.abs z2 = 25 := 
by
  rw [h1, h2]
  -- simplify abs (4 - 3i) * abs (4 + 3i)
  have : |4 - 3*complex.I| * |4 + 3*complex.I| = complex.abs ((4 - 3*complex.I) * (4 + 3*complex.I)) := complex.abs_mul (4 - 3*complex.I) (4 + 3*complex.I)
  rw [this]
  -- (4 - 3i) * (4 + 3i) = 25
  have : (4 - 3*complex.I) * (4 + 3*complex.I) = 25 := by 
    rw [←complex.mul_conj, complex.norm_sq_eq_conj_mul_self]
    simp [complex.norm_sq]
  rw [this]
  -- the modulus of 25 is 25
  rw [complex.abs_assoc, complex.abs_of_real, complex.abs_eq_abs_of_nonneg]
  norm_num
  sorry

end moduli_product_l432_432468


namespace plates_used_l432_432681

theorem plates_used (P : ℕ) (h : 3 * 2 * P + 4 * 8 = 38) : P = 1 := by
  sorry

end plates_used_l432_432681


namespace common_difference_is_minus_two_l432_432741

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
noncomputable def sum_arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem common_difference_is_minus_two
  (a1 d : ℤ)
  (h1 : sum_arith_seq a1 d 5 = 15)
  (h2 : arith_seq a1 d 2 = 5) :
  d = -2 :=
by
  sorry

end common_difference_is_minus_two_l432_432741


namespace parallelogram_construction_l432_432509

theorem parallelogram_construction 
  (α : ℝ) (hα : 0 ≤ α ∧ α < 180)
  (A B : (ℝ × ℝ))
  (in_angle : (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ α ∧ 
               ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ α))
  (C D : (ℝ × ℝ)) :
  ∃ O : (ℝ × ℝ), 
    O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) :=
sorry

end parallelogram_construction_l432_432509


namespace difference_largest_smallest_geometric_l432_432053

open Nat

noncomputable def is_geometric_sequence (a b c d : ℕ) : Prop :=
  b = a * 2 / 3 ∧ c = a * (2 / 3)^2 ∧ d = a * (2 / 3)^3 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem difference_largest_smallest_geometric : 
  exists (largest smallest : ℕ), 
  (is_geometric_sequence (largest / 1000) ((largest % 1000) / 100) ((largest % 100) / 10) (largest % 10)) ∧ 
  (is_geometric_sequence (smallest / 1000) ((smallest % 1000) / 100) ((smallest % 100) / 10) (smallest % 10)) ∧ 
  largest = 9648 ∧ smallest = 1248 ∧ largest - smallest = 8400 :=
begin
  sorry
end

end difference_largest_smallest_geometric_l432_432053


namespace num_primes_le_20_l432_432553

theorem num_primes_le_20 : set.filter nat.prime (set.Icc 1 20) = 8 := by {
  sorry
}

end num_primes_le_20_l432_432553


namespace knights_count_l432_432635

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432635


namespace mike_spent_total_l432_432686

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l432_432686


namespace value_of_M_l432_432981

theorem value_of_M (M : ℝ) (h : (25 / 100) * M = (35 / 100) * 1800) : M = 2520 := 
sorry

end value_of_M_l432_432981


namespace value_of_N_l432_432845

-- Definitions based on conditions
def radius_cylinder_A : ℝ := r
def height_cylinder_A : ℝ := h
def radius_cylinder_B : ℝ := height_cylinder_A
def height_cylinder_B : ℝ := 3 * radius_cylinder_A

-- Volumes based on the radii and heights
def volume_cylinder_A : ℝ := π * radius_cylinder_A^2 * height_cylinder_A
def volume_cylinder_B : ℝ := π * radius_cylinder_B^2 * height_cylinder_B

-- Given condition on volumes
axiom vol_condition : volume_cylinder_A = 3 * volume_cylinder_B

theorem value_of_N (r h : ℝ) (radius_cylinder_A := r) (height_cylinder_A := h)
  (radius_cylinder_B := height_cylinder_A) (height_cylinder_B := 3 * radius_cylinder_A)
  (volume_cylinder_A := π * radius_cylinder_A^2 * height_cylinder_A)
  (volume_cylinder_B := π * radius_cylinder_B^2 * height_cylinder_B)
  (vol_condition : volume_cylinder_A = 3 * volume_cylinder_B) :
  ∃ N : ℝ, volume_cylinder_A = N * π * height_cylinder_A^3 ∧ N = 81 := by
  sorry

end value_of_N_l432_432845


namespace sum_of_third_terms_arithmetic_progressions_l432_432138

theorem sum_of_third_terms_arithmetic_progressions
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∃ d1 : ℕ, ∀ n : ℕ, a (n + 1) = a 1 + n * d1)
  (h2 : ∃ d2 : ℕ, ∀ n : ℕ, b (n + 1) = b 1 + n * d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 5 + b 5 = 35) :
  a 3 + b 3 = 21 :=
by
  sorry

end sum_of_third_terms_arithmetic_progressions_l432_432138


namespace alice_flips_exactly_three_tails_l432_432030

-- Definitions of the conditions
def num_flips : ℕ := 8
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def num_tails : ℕ := 3

-- The main theorem we want to prove
theorem alice_flips_exactly_three_tails :
  let n := num_flips,
      k := num_tails,
      p_t := prob_tails,
      p_h := prob_heads in
  (nat.choose n k : ℚ) * (p_t ^ k) * (p_h ^ (n - k)) = 448 / 177147 := sorry

end alice_flips_exactly_three_tails_l432_432030


namespace fraction_sum_l432_432363

theorem fraction_sum :
  (1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 : ℚ) = -5 / 6 :=
by sorry

end fraction_sum_l432_432363


namespace maximum_possible_value_l432_432651

open List

def maximizeSum (l : List ℕ) : ℕ :=
  match l with
  | [a, b, c, d, e, f] => a * b + b * c + c * d + d * e + e * f + f * a
  | _ => 0

def isMaxPermutation (l : List ℕ) : Prop :=
  l.permutations.any (λ p, maximizeSum p = 76)

theorem maximum_possible_value :
  ∃ M N, M = 76 ∧ N = 12 ∧ (M + N = 88) :=
by
  use 76, 12
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }
  sorry

end maximum_possible_value_l432_432651


namespace star_eq_zero_iff_x_eq_5_l432_432077

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end star_eq_zero_iff_x_eq_5_l432_432077


namespace f_ge_g_l432_432896

noncomputable def f (n : ℕ) : ℕ :=
  (nat.divisors n).filter (λ d, d % 10 = 1 ∨ d % 10 = 9).length

noncomputable def g (n : ℕ) : ℕ :=
  (nat.divisors n).filter (λ d, d % 10 = 3 ∨ d % 10 = 7).length

theorem f_ge_g (n : ℕ) (hn : n > 0) : f(n) ≥ g(n) :=
  sorry

end f_ge_g_l432_432896


namespace problem1_min_value_and_set_problem2_find_a_b_l432_432953

-- Problem 1: Find the minimum value of the function f(x) and the set of values of x when this minimum value is attained.
def f (x : Real) : Real := (√3/2) * Real.sin (2 * x) - Real.cos x ^ 2 - 1 / 2

theorem problem1_min_value_and_set :
  (∃ m : Real, m = -2 ∧ ∀ x : Real, f(x) ≥ m) ∧ 
  (∃ (S : Set Real), S = {x | ∃ k : ℤ, x = k * Real.pi - Real.pi / 6}) :=
sorry

-- Problem 2: Given a triangle with specific side lengths and angles, find a and b.
variable {a b c A B C : Real}
variable (h1 : c = √3) (h2 : sin (2 * C - π / 6) = 1) (h3 : 0 < C ∧ C < π) (h4 : Real.sin B = 2 * Real.sin A)

noncomputable def f_C (C : Real) : Real := (√3/2) * Real.sin (2 * C) - Real.cos C ^ 2 - 1 / 2

theorem problem2_find_a_b :
  f_C(C) = 0 → (B + C = 2 * π) → (a = 1) ∧ (b = 2) :=
sorry

end problem1_min_value_and_set_problem2_find_a_b_l432_432953


namespace fraction_absent_l432_432203

theorem fraction_absent (p : ℕ) (x : ℚ) (h : (W / p) * 1.2 = W / (p * (1 - x))) : x = 1 / 6 :=
by
  sorry

end fraction_absent_l432_432203


namespace probability_triangle_l432_432581

noncomputable def points : List (ℕ × ℕ) := [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2), (3, 3)]

def collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def is_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop := ¬ collinear p1 p2 p3

axiom collinear_ACEF : collinear (0, 0) (1, 1) (2, 2) ∧ collinear (0, 0) (1, 1) (3, 3) ∧ collinear (1, 1) (2, 2) (3, 3)
axiom collinear_BCD : collinear (2, 0) (1, 1) (0, 2)

theorem probability_triangle : 
  let total := 20
  let collinear_ACEF := 4
  let collinear_BCD := 1
  (total - collinear_ACEF - collinear_BCD) / total = 3 / 4 :=
by
  sorry

end probability_triangle_l432_432581


namespace sum_a_k_dot_a_k_plus_1_is_18_sqrt_3_l432_432544

def a (k : ℕ) : ℝ × ℝ := (Real.cos (k * Real.pi / 6), Real.sin (k * Real.pi / 6) + Real.cos (k * Real.pi / 6))

def sum_vectors_dot_products : ℝ :=
  (0 : ℕ).upto 12 |>.toList 
  |>.map (λ k => (a k).fst * (a (k + 1)).fst + (a k).snd * (a (k + 1)).snd) 
  |>.sum

theorem sum_a_k_dot_a_k_plus_1_is_18_sqrt_3 :
  sum_vectors_dot_products = 18 * Real.sqrt 3 :=
sorry

end sum_a_k_dot_a_k_plus_1_is_18_sqrt_3_l432_432544


namespace lines_parallel_if_perpendicular_to_same_plane_l432_432972

variables {Line : Type} {Plane : Type}
variable (a b : Line)
variable (α : Plane)

-- Conditions 
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- Definition for line perpendicular to plane
def lines_parallel (l1 l2 : Line) : Prop := sorry -- Definition for lines parallel

-- Theorem Statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  line_perpendicular_to_plane a α →
  line_perpendicular_to_plane b α →
  lines_parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l432_432972


namespace chocolate_cost_l432_432438

-- Define the necessary conditions as variables or constraints
variables (C B : ℝ)

-- Given conditions
def condition1 : Prop := C + B = 7
def condition2 : Prop := B = C + 4

-- State the goal: Proving the cost of the chocolate (C) is 1.5
theorem chocolate_cost (h1 : condition1) (h2 : condition2) : C = 1.5 :=
by {
  -- Sorry placeholder to skip the proof
  sorry
}

end chocolate_cost_l432_432438


namespace polynomial_degree_l432_432759

def P (x : ℝ) : ℝ := 3 * x^5 + 7 * x^2 + 200 - 3 * Real.pi * x^3 + 4 * Real.sqrt 5 * x^5 - 15

theorem polynomial_degree : polynomial.degree (P : polynomial ℝ) = 5 := by
  sorry

end polynomial_degree_l432_432759


namespace mike_spent_l432_432683

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l432_432683


namespace total_revenue_eq_980_l432_432318

def ticketPrice : ℕ := 20
def firstGroup : ℕ := 10
def secondGroup : ℕ := 20
def totalPeople : ℕ := 56
def discount1 : ℝ := 0.40
def discount2 : ℝ := 0.15

theorem total_revenue_eq_980
  (price : ℕ := ticketPrice)
  (group1 : ℕ := firstGroup)
  (group2 : ℕ := secondGroup)
  (total : ℕ := totalPeople)
  (disc1 : ℝ := discount1)
  (disc2 : ℝ := discount2) :
  let group3 := total - group1 - group2 in
  let revenue1 := group1 * (price - disc1 * price).nat_abs in
  let revenue2 := group2 * (price - disc2 * price).nat_abs in
  let revenue3 := group3 * price in
  revenue1 + revenue2 + revenue3 = 980 := 
by
  sorry

end total_revenue_eq_980_l432_432318


namespace value_of_y_l432_432342

theorem value_of_y (y : ℚ) (h : 3 * y / 7 = 12) : y = 28 :=
begin
  -- Proof goes here
  sorry
end

end value_of_y_l432_432342


namespace three_y_squared_value_l432_432350

theorem three_y_squared_value : ∃ x y : ℤ, 3 * x + y = 40 ∧ 2 * x - y = 20 ∧ 3 * y ^ 2 = 48 :=
by
  sorry

end three_y_squared_value_l432_432350


namespace divisors_product_eq_10p9_l432_432879

theorem divisors_product_eq_10p9 (n : ℕ) : (∏ d in (Finset.filter (λ d, d ∣ n) (Finset.range (n + 1))), d) = 10^9 ↔ n = 100 :=
by
  sorry

end divisors_product_eq_10p9_l432_432879


namespace total_cookies_sold_l432_432830

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l432_432830


namespace M_diff_N_l432_432230

def A : Set ℝ := sorry
def B : Set ℝ := sorry

def M := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Definition of set subtraction
def set_diff (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∉ B}

-- Given problem statement
theorem M_diff_N : set_diff M N = {x : ℝ | -3 ≤ x ∧ x < 0} := 
by
  sorry

end M_diff_N_l432_432230


namespace sum_of_solutions_l432_432080

open Real

theorem sum_of_solutions :
  (∀ x : ℝ, (x^2 - 5*x + 3)^(x^2 - 6*x + 3) = 1 → ∃ S : ℝ, Σ (x : ℝ), 
    (x^2 - 6*x + 3 = 0 → S = 6) ∧ (x^2 - 5*x + 2 = 0 → S = 5)) :=
begin
  sorry
end

end sum_of_solutions_l432_432080


namespace probability_all_digits_distinct_probability_all_digits_odd_l432_432008

-- Definitions to be used in the proof
def total_possibilities : ℕ := 10^5
def all_distinct_possibilities : ℕ := 10 * 9 * 8 * 7 * 6
def all_odd_possibilities : ℕ := 5^5

-- Probabilities
def prob_all_distinct : ℚ := all_distinct_possibilities / total_possibilities
def prob_all_odd : ℚ := all_odd_possibilities / total_possibilities

-- Lean 4 Statements to Prove
theorem probability_all_digits_distinct :
  prob_all_distinct = 30240 / 100000 := by
  sorry

theorem probability_all_digits_odd :
  prob_all_odd = 3125 / 100000 := by
  sorry

end probability_all_digits_distinct_probability_all_digits_odd_l432_432008


namespace extreme_value_M_range_of_m_l432_432949

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := x + 1/x + a * (1/x)
noncomputable def M (x : ℝ) : ℝ := (F x 1) - f x

-- Problem (I): Prove extreme value
theorem extreme_value_M :
  ∃ x, (x = 2) ∧ (∀ x, M(x) ≥ M(2)) :=
sorry

-- Problem (II): Prove range of m
theorem range_of_m (x : ℝ) : 
  (∀ x > 0, (1 / F(x, 0)) ≤ (1 / (2 + m * (f x)^2))) →
  (∀ x > 0, m ∈ Set.Icc 0 1) :=
sorry

end extreme_value_M_range_of_m_l432_432949


namespace range_of_m_l432_432523

noncomputable def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ set.Icc (-2 : ℝ) 2, f (-x) = -f x) ∧
  (∀ x₁ x₂ ∈ set.Icc (-2 : ℝ) 0, x₁ < x₂ → f x₁ > f x₂)

theorem range_of_m (f : ℝ → ℝ) (h_odd_decreasing : odd_decreasing_function f) 
  (h_f_condition : ∀ m : ℝ, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m : ℝ, -1 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l432_432523


namespace red_apples_in_basket_l432_432746

theorem red_apples_in_basket : 
  (total_apples green_apples : ℕ) (h1 : total_apples = 9) (h2 : green_apples = 2) : 
  total_apples - green_apples = 7 := 
by
  sorry

end red_apples_in_basket_l432_432746


namespace cylindrical_coord_correct_l432_432434

-- Definitions from conditions
def rect_point := (3 : ℝ, -3 * Real.sqrt 3, 2)

def cylindrical_r (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
def cylindrical_theta (x y : ℝ) : ℝ := Real.arctan (y / x)
def cylindrical_z (z : ℝ) : ℝ := z

noncomputable def cylindrical_coords (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (cylindrical_r x y, cylindrical_theta x y, cylindrical_z z)

-- The proof problem
theorem cylindrical_coord_correct :
  cylindrical_coords rect_point = (6, 5 * Real.pi / 3, 2) :=
by
  sorry

end cylindrical_coord_correct_l432_432434


namespace knights_count_l432_432634

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432634


namespace range_f_l432_432836

noncomputable def f (x : ℝ) : ℝ := abs (x - 3) - abs (x + 5)

theorem range_f : set.Icc (-8 : ℝ) 8 = set_of (λ y, ∃ x : ℝ, f x = y) :=
by
  sorry

end range_f_l432_432836


namespace cubic_polynomial_root_of_Q_l432_432089

def Q (x : ℚ) : ℚ := x^3 - 3 * x^2 + 3 * x - 5

theorem cubic_polynomial_root_of_Q :
  Q (∛4 + 1) = 0 := 
by sorry

end cubic_polynomial_root_of_Q_l432_432089


namespace modulus_product_l432_432474

theorem modulus_product (a b : ℂ) : |a - b * complex.i| * |a + b * complex.i| = 25 := by
  have h1 : complex.norm (4 - 3 * complex.i) = 5 := by
    sorry
  have h2 : complex.norm (4 + 3 * complex.i) = 5 := by
    sorry
  rw [← complex.norm_mul, (4 - 3 * complex.i).mul_conj_self, (4 + 3 * complex.i).mul_conj_self, add_comm] at h1
  rw [mul_comm, mul_comm (complex.norm _), ← mul_assoc, h2, mul_comm, mul_assoc]
  exact (mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)).1 h1 

end modulus_product_l432_432474


namespace number_of_proper_subsets_A_l432_432969

open Set

-- Define the universal set U and set A based on its complement
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := U \ {2}

-- Define the statement to compute the number of proper subsets.
theorem number_of_proper_subsets_A : Finset.card (Finset.powerset A.to_finset).filter (λ s, s ≠ A.to_finset) = 7 := by
  sorry

end number_of_proper_subsets_A_l432_432969


namespace baseball_card_difference_l432_432678

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) (h1 : marcus_cards = 210) (h2 : carter_cards = 152) : marcus_cards - carter_cards = 58 :=
by {
    --skip the proof
    sorry
}

end baseball_card_difference_l432_432678


namespace susan_first_turn_spaces_l432_432284

theorem susan_first_turn_spaces (x : ℕ) :
  x - 3 + 6 + 37 = 48 → x = 8 :=
by {
    intro h,
    sorry,
}

end susan_first_turn_spaces_l432_432284


namespace angle_between_line_and_plane_correct_l432_432976

variables (m n : Vector3)

noncomputable def angle_between_line_and_plane (m n : Vector3) : ℝ :=
  let cos_theta := -1 / 2
  -- The complementary angle to the one given by the cosine value
  (real.arccos cos_theta)

theorem angle_between_line_and_plane_correct
  (m n : Vector3)
  (h : real.arccos ((m.dot n) / (m.norm * n.norm)) = real.arccos (-1 / 2)) :
  angle_between_line_and_plane m n = real.arccos (-1 / 2) :=
by
  -- Proof is skipped
  sorry

end angle_between_line_and_plane_correct_l432_432976


namespace distance_focus_directrix_l432_432716

-- Given: The equation of the parabola is y = 2x^2
-- Prove: The distance from the focus to the directrix is 1/2.

theorem distance_focus_directrix (x : ℝ) : 
  let y := 2 * x^2 in
  (let focus := (0, 1 / 4) in
   let directrix := -1 / 4 in
   let distance := abs(focus.2 - directrix) in
   distance = 1 / 2) := 
by 
  sorry

end distance_focus_directrix_l432_432716


namespace trapezoid_area_eq_l432_432400

-- Definitions based on conditions
variable {x : ℝ}

def base₁ := 2 * x
def base₂ := 3 * x
def height := x

-- Statement of the problem
theorem trapezoid_area_eq (x : ℝ) : 
  (1 / 2 * (base₁ + base₂) * height) = (5 * x^2) / 2 := by
sorry

end trapezoid_area_eq_l432_432400


namespace geometric_sequence_condition_l432_432991

theorem geometric_sequence_condition {a : ℕ → ℝ} (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) : 
  (a 3 * a 5 = 16) ↔ a 4 = 4 :=
sorry

end geometric_sequence_condition_l432_432991


namespace soccer_field_illumination_l432_432379

noncomputable def diagonal (l w : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2)

noncomputable def min_ceiling_height (l w : ℝ) : ℝ :=
  Real.ceil ((diagonal l w) / 4 * 10) / 10

theorem soccer_field_illumination :
  min_ceiling_height 90 60 = 27.1 :=
by
  sorry

end soccer_field_illumination_l432_432379


namespace exist_circle_with_parallel_chord_l432_432117

variables {Point Line Circle : Type}
variable {O : Point} -- center of \Gamma
variable {Γ : Circle}
variable {ℓ : Line}
variable {P1 P2 : Point}

/-- 
Given a line ℓ, two points P1 and P2, and a circle Γ with center O,
there exists a circle that passes through P1 and P2, 
and has a chord within Γ that is parallel to the line ℓ.
-/
theorem exist_circle_with_parallel_chord (HΓ : Γ.center = O) :
  ∃ (C : Point), (∃ (newCircle : Circle), C = newCircle.center ∧ 
  newCircle.passes_through P1 ∧ newCircle.passes_through P2 ∧ 
  (∃ (chord : Line), chord.parallel ℓ ∧ chord.is_chord_of newCircle Γ)) :=
sorry

end exist_circle_with_parallel_chord_l432_432117


namespace smaller_number_l432_432309

theorem smaller_number (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 16) : y = 4 := by
  sorry

end smaller_number_l432_432309


namespace possible_values_f_30_l432_432555

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 : ∀ x : ℕ+, f x ≤ x^2
axiom f_condition2 : ∀ x y : ℕ+, f(f(f(x)) * f(f(y))) = x * y

theorem possible_values_f_30 : {n : ℕ+ | ∃ f, (∀ x : ℕ+, f x ≤ x^2) ∧ (∀ x y : ℕ+, f(f(f(x)) * f(f(y))) = x * y) ∧ n = f(30)}.card = 24 := sorry

end possible_values_f_30_l432_432555


namespace min_ceiling_height_l432_432374

def length : ℝ := 90
def width : ℝ := 60
def diagonal : ℝ := real.sqrt (length^2 + width^2)
def height : ℝ := (1 / 4) * diagonal

theorem min_ceiling_height (h : ℝ) : h = 27.1 → (∃ (r : ℝ), r = h ∧ r ≥ height ∧ (∃ (n : ℝ), n = 0.1 * ⌈r / 0.1⌉₊)) :=
by
  refine ⟨_, _, _, _⟩;
  sorry

end min_ceiling_height_l432_432374


namespace range_of_m_l432_432158

open Real

noncomputable def f (x : ℝ) : ℝ := 1 + sin (2 * x)
noncomputable def g (x m : ℝ) : ℝ := 2 * (cos x)^2 + m

theorem range_of_m (x₀ : ℝ) (m : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ π / 2) (h₁ : f x₀ ≥ g x₀ m) : m ≤ sqrt 2 :=
by
  sorry

end range_of_m_l432_432158


namespace filipa_coin_position_l432_432463

def move_coin (start_pos : ℕ) (moves : List ℕ) : ℕ :=
  moves.foldl (λ acc move, if move % 2 = 0 then acc + move else acc - move) start_pos

theorem filipa_coin_position :
  let start_pos : ℕ := (15 + 1) / 2 in
  let rolls : List ℕ := [1, 2, 3, 4, 5, 6] in
  move_coin start_pos rolls = 11 :=
by
  -- initial definition for clarity in proof hypothesis
  sorry

end filipa_coin_position_l432_432463


namespace meridian_plane_for_szombathely_meridian_plane_for_szekesfehervar_l432_432116

noncomputable def valparaiso_latitude := -33.1667 -- degree converted to decimal
noncomputable def valparaiso_longitude := -71.5833 -- degree converted to decimal
noncomputable def szombathely_latitude := 47.25 -- degree converted to decimal
noncomputable def szombathely_longitude := 16.6167 -- degree converted to decimal
noncomputable def szekesfehervar_latitude := 47.18333 -- degree converted to decimal
noncomputable def szekesfehervar_longitude := 18.4167 -- degree converted to decimal
noncomputable def rad := λ deg min: deg + min/60

theorem meridian_plane_for_szombathely : 
    ∀ (lamV := rad (-71) (-35)) (phiV := rad (-33) (-10)) 
      (lamS := rad 16 37) (phiS := rad 47 15),
    let x := -40.9167 in let alpha := 52.0333
    in meridian_plane latV lamV latS lamS x ∧ alpha_angle latV lamV latS lamS alpha :=
sorry

theorem meridian_plane_for_szekesfehervar :
    ∀ (lamV := rad (-71) (-35)) (phiV := rad (-33) (-10)) 
      (lamS2 := rad 18 25) (phiS2 := rad 47 11),
    let x := -40.3833 in let alpha := 51.6
    in meridian_plane latV lamV latS2 lamS2 x ∧ alpha_angle latV lamV latS2 lamS2 alpha :=
sorry

end meridian_plane_for_szombathely_meridian_plane_for_szekesfehervar_l432_432116


namespace average_speed_of_car_l432_432187

variables {D : ℝ} (hD : D > 0)

def average_speed (D : ℝ) : ℝ := 
  let time_first_third := D / 240 
  let time_second_third := D / 45
  let time_last_third := D / 144
  let total_time := time_first_third + time_second_third + time_last_third
  D / total_time

theorem average_speed_of_car : average_speed D = 30 :=
by
  sorry

end average_speed_of_car_l432_432187


namespace max_value_of_expression_l432_432235

noncomputable def max_expression_value (x y : ℝ) :=
  x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  max_expression_value x y ≤ 961 / 8 :=
sorry

end max_value_of_expression_l432_432235


namespace otimes_neg2_neg1_l432_432067

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432067


namespace additional_miles_needed_l432_432246

theorem additional_miles_needed :
  ∀ (h : ℝ), (25 + 75 * h) / (5 / 8 + h) = 60 → 75 * h = 62.5 := 
by
  intros h H
  -- the rest of the proof goes here
  sorry

end additional_miles_needed_l432_432246


namespace water_formed_from_reaction_l432_432884

-- Definitions
def mol_mass_water : ℝ := 18.015
def water_formed_grams (moles_water : ℝ) : ℝ := moles_water * mol_mass_water

-- Statement
theorem water_formed_from_reaction (moles_water : ℝ) :
  18 = water_formed_grams moles_water :=
by sorry

end water_formed_from_reaction_l432_432884


namespace midpoint_of_segment_follows_arc_l432_432801

noncomputable def midpoint_trajectory (ABC : Triangle) (M N : Point)
  (MN : ℝ) (h : Angle ABC = π/2) (hMN : dist M N = MN) : 
  Set Point :=
{ O : Point | dist O (vertex_B ABC) = MN / 2 ∧
             (O is the midpoint of segment M N) ∧
             (O slides along the path constrained within the right angle ABC) }

theorem midpoint_of_segment_follows_arc : 
  ∀ (ABC : Triangle) (M N : Point) (MN : ℝ),
  (∠ABC = π / 2) → (dist M N = MN) → 
  ∃ (B : Point), ∀ (O : Point),
  dist O B = MN / 2 ∧
  (O is the midpoint of segment M N) ∧
  (O slides along the path constrained within the right angle ABC) :=
by
  intros
  use B
  split
  sorry
  sorry

end midpoint_of_segment_follows_arc_l432_432801


namespace total_cookies_sold_l432_432832

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l432_432832


namespace discount_difference_correct_l432_432410

-- Define the given conditions
def bill_amount : ℝ := 12000
def single_discount_rate : ℝ := 0.35
def first_successive_discount_rate : ℝ := 0.30
def second_successive_discount_rate : ℝ := 0.06

-- Define the expected answer
def expected_difference : ℝ := 96

-- Prove the expected answer given the conditions
theorem discount_difference_correct : 
  let single_discount_amount := bill_amount * (1 - single_discount_rate),
      first_discount_amount := bill_amount * (1 - first_successive_discount_rate),
      second_discount_amount := first_discount_amount * (1 - second_successive_discount_rate),
      difference := second_discount_amount - single_discount_amount
  in difference = expected_difference :=
by
  sorry

end discount_difference_correct_l432_432410


namespace olivia_card_value_l432_432609

theorem olivia_card_value (x : ℝ) (hx1 : 90 < x ∧ x < 180)
  (h_sin_pos : Real.sin x > 0) (h_cos_neg : Real.cos x < 0) (h_tan_neg : Real.tan x < 0)
  (h_olivia_distinguish : ∀ (a b c : ℝ), 
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (∃! a, a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x)) :
  Real.sin 135 = Real.cos 45 := 
sorry

end olivia_card_value_l432_432609


namespace hall_dimension_difference_l432_432009

theorem hall_dimension_difference
  (L W H : ℝ)
  (hW : W = L / 2)
  (hH : H = W / 3)
  (hV : L * W * H = 600) :
  L - W - H ≈ 6.43 := 
sorry

end hall_dimension_difference_l432_432009


namespace max_product_distance_l432_432908

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432908


namespace license_plate_palindrome_probability_l432_432244

theorem license_plate_palindrome_probability :
  let prob := (1 / 100) * (1 / 676),
      frac := Rat.mkNat 1 67600 in
  prob = frac /\ (1 + 67600 = 67601) :=
by
  sorry

end license_plate_palindrome_probability_l432_432244


namespace cube_edge_length_in_pyramid_l432_432749

variable (m A : ℝ)

theorem cube_edge_length_in_pyramid ( m > 0) (A > 0) :
  ∃ x : ℝ, x = m * Real.sqrt A / (m + Real.sqrt A) :=
by
  sorry

end cube_edge_length_in_pyramid_l432_432749


namespace simplify_expression_correct_l432_432701

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + Real.sqrt 5) - 2 / (2 - Real.sqrt 5)

theorem simplify_expression_correct : simplify_expression = 10 := by
  sorry

end simplify_expression_correct_l432_432701


namespace color_possible_l432_432357

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_odd (n : ℤ) : Prop := n % 2 = 1

def is_blue (p : ℤ × ℤ) : Prop := is_even p.1 ∧ is_even p.2
def is_green (p : ℤ × ℤ) : Prop := is_odd p.1 ∧ is_odd p.2
def is_orange (p : ℤ × ℤ) : Prop := (is_even p.1 ∧ is_odd p.2) ∨ (is_odd p.1 ∧ is_even p.2)

def color (p : ℤ × ℤ) : Type :=
  if is_blue p then Type.blue
  else if is_green p then Type.green
  else if is_orange p then Type.orange
  else Type.none

theorem color_possible :
  (∀ k : ℤ, ∃ p : ℤ × ℤ, p.2 = k ∧ is_blue p) ∧
  (∀ k : ℤ, ∃ p : ℤ × ℤ, p.2 = k ∧ is_green p) ∧
  (∀ k : ℤ, ∃ p : ℤ × ℤ, p.2 = k ∧ is_orange p) ∧
  (∀ (A B C : ℤ × ℤ),
  is_blue A ∧ is_green B ∧ is_orange C → ¬is_collinear A B C) :=
by {
  sorry
}

end color_possible_l432_432357


namespace otimes_neg2_neg1_l432_432072

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l432_432072


namespace min_value_f_l432_432161

-- Define the sequence according to the conditions provided
def a : ℕ → ℝ
| 0     := 0  -- For convenience, let's assume n starts at 1, so a(0) is 0.
| 1     := 8
| (n+2) := a (n + 1) + (n + 1)

-- Define the function f(n) = a(n) / n
noncomputable def f (n : ℕ) : ℝ :=
  if n = 0 then 0 else a n / n

-- State that minimum value of f(n) for n in Natural Numbers is 3.5
theorem min_value_f : ∀ n : ℕ, 0 < n → f n ≥ 3.5 :=
begin
  sorry
end

end min_value_f_l432_432161


namespace complex_modulus_condition_l432_432114

open Complex

theorem complex_modulus_condition (z : ℂ) (h : (z - 1) / (z + 2) = 1 + 3*I) : abs (conj z + 2*I) = sqrt 5 :=
by
  sorry

end complex_modulus_condition_l432_432114


namespace verify_number_of_correct_propositions_l432_432126

variables {a b : Type} {α β γ : Type}

def relation_parallel_lines_planes (a b : Type) : Prop :=
  ∀ α : Type, 
    (a ∣∣ α) → (b ∣∣ α) → (a ∣∣ b) → False

def relation_parallel_planes_intersect (a : Type) (α β : Type) : Prop :=
  (a ∣∣ α) → (a ∣∣ β) → (α ∣∣ β) → False

def relation_parallel_lines_parallel_planes (a : Type) (β γ : Type) : Prop :=
  (a ∣∣ γ) → (β ∣∣ γ) → (a ∣∣ β) → False

def number_of_correct_propositions : ℕ :=
  0

theorem verify_number_of_correct_propositions :
  ¬ (relation_parallel_lines_planes a b) ∧
  ¬ (relation_parallel_planes_intersect a α β) ∧
  ¬ (relation_parallel_lines_parallel_planes a β γ) →
  number_of_correct_propositions = 0 :=
sorry

end verify_number_of_correct_propositions_l432_432126


namespace log_comparison_l432_432082

theorem log_comparison : log 2010 2011 > log 2011 2012 := 
sorry

end log_comparison_l432_432082


namespace sum_of_areas_of_trapezoids_l432_432742
   
   variables {AC BC: ℝ} [NonZero AC] [NonZero BC]
   
   -- Conditions
   def height_from_B := 25
   def side_AC := 24
   def area_of_triangle_ABC := (1 / 2) * side_AC * height_from_B
   
   -- Given conditions provide a total area calculation
   def total_area := 300

   -- Area of individual small triangle
   def small_triangle_area := total_area / 25

   -- Areas of trapezoids KLQR and MNOP
   def area_KLQR := 7 * small_triangle_area
   def area_MNOP := 3 * small_triangle_area

   -- The goal is to show that adding the areas of trapezoids KLQR and MNOP equals 120
   theorem sum_of_areas_of_trapezoids :
     (area_KLQR + area_MNOP) = 120 := 
   by
     sorry
   
end sum_of_areas_of_trapezoids_l432_432742


namespace binary_add_sub_l432_432416

theorem binary_add_sub:
  let a := 0b10110
  let b := 0b1010
  let c := 0b11100
  let d := 0b1110
  a + b - c + d = 0b01110 := by
  sorry

end binary_add_sub_l432_432416


namespace remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l432_432332

theorem remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one :
  ((x - 1) ^ 2028) % (x^2 - x + 1) = 1 :=
by
  sorry

end remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l432_432332


namespace discount_application_l432_432807

theorem discount_application (original_price : ℝ) :
  let sale_price := 0.7 * original_price in
  let coupon_price := 0.75 * sale_price in
  coupon_price = 0.525 * original_price :=
by
  sorry

end discount_application_l432_432807


namespace barbell_squatRack_ratio_l432_432594

-- Defining the given conditions
def squatRackCost : ℝ := 2500
def totalCost : ℝ := 2750

-- Defining the barbell cost based on the given conditions
def barbellCost : ℝ := totalCost - squatRackCost

-- Defining the ratio of barbell cost to squat rack cost
def ratio : ℝ := barbellCost / squatRackCost

-- The proof statement that the ratio is indeed 1/10
theorem barbell_squatRack_ratio : ratio = (1 / 10) :=
by 
  -- The proof goes here (skipped with sorry)
  sorry

end barbell_squatRack_ratio_l432_432594


namespace proof_problem_l432_432580

def line_m (a : Real) : Real → Real → Prop := λ x y, a * x - 3 * y + 2 = 0
def point_M : Real × Real := (3, 1)
def line_n (b : Real) : Real → Real → Prop := λ x y, 4 * x - 6 * y + b = 0

def x_intercept_condition (a : Real) : Prop := ∃ x, line_m a x 0 ∧ x = -2

def parallel_condition (a : Real) : Prop := 2 * a = 12

def intercept_form_m (a : Real) : (Real → Real → Prop) := 
  λ x y, (x / -2) + (y / (2 / 3)) = 1

def distance_between_lines (a b : Real) : Real :=
  let denom := Real.sqrt (a^2 + (-3)^2)
  (Real.abs (4 * 3 - 6 * 1 + b)) / denom

theorem proof_problem : 
  (∃ a, x_intercept_condition a ∧ a = 1
  ∧ (line_m a = intercept_form_m a)
  ∧ ∃ b, line_n b point_M.fst point_M.snd ∧ b = -6
  ∧ (parallel_condition a)
  ∧ (distance_between_lines 2 -6 = 5 * Real.sqrt 13 / 13)) :=
sorry

end proof_problem_l432_432580


namespace evaluate_root_power_l432_432862

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l432_432862


namespace maxValue_of_MF1_MF2_l432_432902

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432902


namespace lies_on_ωA_l432_432664

noncomputable def lies_on_circle (P : Type) [metric_space P] :=
{ C : set P // ∃ x ∈ C, ∀ y ∈ C, dist x y = dist P y }

variables {P : Type} [metric_space P] (A B C D E M : P)
variables (ωA ωC : set P) (lineBC lineAB lineMD lineAC : set P)

def midpoint (x y : P) : P := sorry -- definition for midpoint
def intersection (l1 l2 : set P) : P := sorry -- definition for intersection

noncomputable def is_tangent (circle : set P) (line : set P) (P : P) : Prop :=
circle P ∧ ∃ Q ∈ circle, Q ≠ P ∧ ∀ R ∈ line, dist P Q = dist R Q

axiom circle_through_point (x : P) : ∃ y z, lies_on_circle y ∧ lies_on_circle z

axiom triangle (A B C : P) : Prop

theorem lies_on_ωA
  (h1 : triangle A B C)
  (h2 : lies_on_circle ωA A ∧ is_tangent ωA lineBC B)
  (h3 : lies_on_circle ωC C ∧ is_tangent ωC lineAB B)
  (h4 : ∃ _ : lies_on_circle ωA D, lies_on_circle ωC D)
  (h5 : M = midpoint B C)
  (h6 : E = intersection lineMD lineAC) :
  E ∈ ωA :=
sorry

end lies_on_ωA_l432_432664


namespace sequences_equal_l432_432547

def S (n : ℕ) : ℚ :=
  (List.range (2 * n + 1)).map (λ i => if i % 2 = 0 then - (1 / (i + 1) : ℚ) else (1 / (i + 1) : ℚ)).sum

def T (n : ℕ) : ℚ :=
  (List.range n).map (λ i => 1 / (n + 1 + i : ℚ)).sum

theorem sequences_equal (n : ℕ) (hn : n > 0) : S n = T n := 
  sorry

end sequences_equal_l432_432547


namespace tangent_line_at_origin_l432_432109

def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_origin :
  let f' := fun x => (1 + x) * Real.exp x in
  ( f(0) = 0 ∧ f' 0 = 1 ) → ( ∀ x : ℝ, f x = y ↔ x = y ) :=
by
  intro _ _,
  sorry

end tangent_line_at_origin_l432_432109


namespace initial_thickness_of_blanket_l432_432088

theorem initial_thickness_of_blanket (T : ℝ)
  (h : ∀ n, n = 4 → T * 2^n = 48) : T = 3 :=
by
  have h4 := h 4 rfl
  sorry

end initial_thickness_of_blanket_l432_432088


namespace coeff_sum_equals_4_pow_7_l432_432779

noncomputable def polynomial_coeff_sum : ℤ :=
  let f : polynomial ℤ := (polynomial.C 3 * polynomial.X - polynomial.C 1) ^ 7
  let coeffs := (coeffs := (f.coeffs.drop 1).map abs)
  coeffs.sum

theorem coeff_sum_equals_4_pow_7 :
  polynomial_coeff_sum = 16384 := 
sorry

end coeff_sum_equals_4_pow_7_l432_432779


namespace three_pow_expr_l432_432900

theorem three_pow_expr (m n : ℝ) (h1 : 9^m = 3) (h2 : 27^n = 4) : 3^(2*m + 3*n) = 12 :=
by
  sorry

end three_pow_expr_l432_432900


namespace original_water_amount_l432_432003

theorem original_water_amount (W : ℝ) 
    (evap_rate : ℝ := 0.03) 
    (days : ℕ := 22) 
    (evap_percent : ℝ := 0.055) 
    (total_evap : ℝ := evap_rate * days) 
    (evap_condition : evap_percent * W = total_evap) : W = 12 :=
by sorry

end original_water_amount_l432_432003


namespace selling_price_per_pound_of_mixture_l432_432287

-- Define the base prices of cashews and Brazil nuts.
def price_per_pound_cashews : ℝ := 6.75
def price_per_pound_brazil_nuts : ℝ := 5.00

-- Define the amount of nuts in the mixture.
def weight_cashews : ℝ := 20.0
def weight_brazil_nuts : ℝ := 30.0
def total_weight_mixture : ℝ := 50.0

-- Define the expected selling price per pound of the mixture.
def expected_selling_price_per_pound : ℝ := 5.70

-- Prove that the calculated selling price per pound is as expected.
theorem selling_price_per_pound_of_mixture :
  let total_cost_mixture := (weight_cashews * price_per_pound_cashews) + (weight_brazil_nuts * price_per_pound_brazil_nuts),
      selling_price := total_cost_mixture / total_weight_mixture
  in selling_price = expected_selling_price_per_pound :=
by
  sorry

end selling_price_per_pound_of_mixture_l432_432287


namespace find_ages_of_son_daughter_and_niece_l432_432005

theorem find_ages_of_son_daughter_and_niece
  (S : ℕ) (D : ℕ) (N : ℕ)
  (h1 : ∀ (M : ℕ), M = S + 24) 
  (h2 : ∀ (M : ℕ), 2 * (S + 2) = M + 2)
  (h3 : D = S / 2)
  (h4 : 2 * (D + 6) = 2 * S * 2 / 3)
  (h5 : N = S - 3)
  (h6 : 5 * N = 4 * S) :
  S = 22 ∧ D = 11 ∧ N = 19 := 
by 
  sorry

end find_ages_of_son_daughter_and_niece_l432_432005


namespace calc_g_3_l432_432079

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem calc_g_3 : g (g (g (g 3))) = 1 := by
  sorry

end calc_g_3_l432_432079


namespace find_time_for_compound_interest_l432_432790

noncomputable def compound_interest_time 
  (A P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem find_time_for_compound_interest :
  compound_interest_time 500 453.51473922902494 0.05 1 = 2 :=
sorry

end find_time_for_compound_interest_l432_432790


namespace calc_exponent_l432_432898

theorem calc_exponent (m n : ℝ) (h1 : 9^m = 3) (h2 : 27^n = 4) : 3^(2*m + 3*n) = 12 := by
  sorry

end calc_exponent_l432_432898


namespace locus_of_vertex_C_is_pair_of_distinct_points_l432_432401

-- Assuming conditions provided in the question
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (dist_A_B : dist A B = 4) (median_length : ∀ (D : Type), dist A D = 3)
variables (circle_constraint : ∀ (C : Type), dist B C = 5)

-- Proving the locus of vertex C is a pair of distinct points
theorem locus_of_vertex_C_is_pair_of_distinct_points :
  ∃ C₁ C₂ : Type, C₁ ≠ C₂ ∧ dist B C₁ = 5 ∧ dist B C₂ = 5 ∧ dist A (midpoint C₁ C₂) = 3 :=
sorry

end locus_of_vertex_C_is_pair_of_distinct_points_l432_432401


namespace matrix_N_computation_l432_432652

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) :
  (N.mul_vec ![3, -2] = ![4, 0]) →
  (N.mul_vec ![-4, 6] = ![-2, -2]) →
  (N.mul_vec ![7, 2] = ![16, -4]) :=
by
  intros h1 h2
  sorry

end matrix_N_computation_l432_432652


namespace company_fund_initial_amount_l432_432296

theorem company_fund_initial_amount (n : ℕ) (fund_initial : ℤ) 
  (h1 : ∃ n, fund_initial = 60 * n - 10)
  (h2 : ∃ n, 55 * n + 120 = fund_initial + 130)
  : fund_initial = 1550 := 
sorry

end company_fund_initial_amount_l432_432296


namespace find_g_3_l432_432723

def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 3) : g 3 = 0 := 
by
  sorry

end find_g_3_l432_432723


namespace compare_logarithms_25_75_65_260_finite_steps_log_compare_l432_432709

-- Part (a): Specific comparison using given rules

theorem compare_logarithms_25_75_65_260 :
  ∀ a b c d : ℝ, (a > 1) → (b > 1) → (c > 1) → (d > 1) → 
  (a = 25) → (b = 75) → (c = 65) → (d = 260) → 
  log a b > log c d :=
by 
  intros a b c d ha hb hc hd ha_eq hb_eq hc_eq hd_eq
  rw [ha_eq, hb_eq, hc_eq, hd_eq]
  -- Steps of the device checking and comparison logic can be included here
  sorry

-- Part (b): Proof that any two unequal logarithms can be compared in finite steps

theorem finite_steps_log_compare :
  ∀ a b c d : ℝ, (a > 1) → (b > 1) → (c > 1) → (d > 1) →
  log a b ≠ log c d → 
  ∃ n : ℕ, (comparison_steps a b c d n) :=
sorry

-- Helper function to model comparison steps
def comparison_steps : ℝ → ℝ → ℝ → ℝ → ℕ → Prop
| a b c d 0 := False
| a b c d (n+1) :=
  if b > a ∧ d > c then comparison_steps a (b/a) c (d/c) n
  else if b < a ∧ d < c then comparison_steps d (c/d) b (a/b) n
  else True

end compare_logarithms_25_75_65_260_finite_steps_log_compare_l432_432709


namespace total_cost_two_rackets_l432_432259

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l432_432259


namespace two_roots_range_a_l432_432952

noncomputable def piecewise_func (x : ℝ) : ℝ :=
if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ piecewise_func x1 = a * x1 ∧ piecewise_func x2 = a * x2) ↔ (1/3 < a ∧ a < 1/Real.exp 1) :=
sorry

end two_roots_range_a_l432_432952


namespace max_lambda_value_l432_432112

theorem max_lambda_value (a b c : ℝ) (λ : ℝ) 
  (h1 : 0 < a ∧ a ≤ 1) 
  (h2 : 0 < b ∧ b ≤ 1) 
  (h3 : 0 < c ∧ c ≤ 1) 
  (h4 : (λ x y z : ℝ, (x > 0 ∧ x ≤ 1) → (y > 0 ∧ y ≤ 1) → (z > 0 ∧ z ≤ 1) → λ * (1 - x) * (1 - y) * (1 - z) ≤ 1)): 
  λ ≤ 64/27 := 
sorry

end max_lambda_value_l432_432112


namespace fill_8x8_with_distinct_square_sums_is_square_l432_432592

def sum_is_square (lst : List ℕ) : Prop :=
  ∃ k : ℕ, List.sum lst = k * k

def is_8x8_table_filled_with_distinct_squares_and_sums_are_squares (table : Array (Array ℕ)) : Prop :=
  table.size = 8 ∧
  (∀ row, (table[row]).size = 8) ∧
  (∀ i j, is_square (table[i][j])) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → table[i][j] ≠ table[k][l]) ∧
  (∀ row, sum_is_square (Array.toList (table[row]))) ∧
  (∀ col, sum_is_square (Array.toList (Array.map (λ row => row[col]) table)))

theorem fill_8x8_with_distinct_square_sums_is_square :
  ∃ table : Array (Array ℕ), is_8x8_table_filled_with_distinct_squares_and_sums_are_squares table :=
sorry

end fill_8x8_with_distinct_square_sums_is_square_l432_432592


namespace max_product_of_distances_l432_432925

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l432_432925


namespace find_weight_of_a_l432_432777

variables (a b c d e : ℕ)

-- Conditions
def cond1 : Prop := a + b + c = 252
def cond2 : Prop := a + b + c + d = 320
def cond3 : Prop := e = d + 7
def cond4 : Prop := b + c + d + e = 316

theorem find_weight_of_a (h1 : cond1 a b c) (h2 : cond2 a b c d) (h3 : cond3 d e) (h4 : cond4 b c d e) :
  a = 79 :=
by sorry

end find_weight_of_a_l432_432777


namespace correct_relationship_l432_432517

variables {α : Type} [affine_space α ℝ] 
variables (l m : line ℝ) (P : α) (A : set (affine_subspace ℝ α))

-- Assume l intersects with plane α but is not perpendicular to it.
axiom l_intersects_not_perpendicular (h₁ : l ∩ A ≠ ∅) (h₂ : ∃ v : ℝ, l.direction ≠ v • A.direction) : Prop

-- Let m be a line in space.
variables (m : line ℝ)

-- The correct geometric relationship.
theorem correct_relationship (h₁ : l ∩ A ≠ ∅) (h₂ : ∃ v : ℝ, l.direction ≠ v • A.direction) :
  (m ⊥ l) ∧ (m ∥ A) :=
sorry

end correct_relationship_l432_432517


namespace M_power_4_beta_l432_432111

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![2, 1]]
def beta : Vector (Fin 2) ℤ := ![7, 1]

theorem M_power_4_beta :
  (M ^ 4) ⬝ beta = ![327, 321] :=
by
  sorry

end M_power_4_beta_l432_432111


namespace remainder_of_x_101_div_xplus1_4_l432_432095

theorem remainder_of_x_101_div_xplus1_4 : 
  ∃ r : ℚ[x], degree r < 4 ∧ (x^101) % (x+1)^4 = r ∧ r = 166650 * x^3 - 3520225 * x^2 + 67605570 * x - 1165299255 := 
by sorry

end remainder_of_x_101_div_xplus1_4_l432_432095


namespace ball_count_l432_432573

theorem ball_count (r b y : ℕ) 
  (h1 : b + y = 9) 
  (h2 : r + y = 5) 
  (h3 : r + b = 6) : 
  r + b + y = 10 := 
  sorry

end ball_count_l432_432573


namespace evaluate_power_l432_432858

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l432_432858


namespace even_n_of_points_condition_l432_432001

variable (n : ℕ)

-- Define condition: for each subset S of teams, there exists a team such that
-- the total points obtained by playing all teams in S is odd.
def points_condition (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ S : Finset (Fin n), ∃ t : Fin n, 
    (Finset.sum S (λ i, M t i) % 2 = 1)

-- Define the theorem to prove that n is even
theorem even_n_of_points_condition (M : Matrix (Fin n) (Fin n) ℕ) 
  (hM : Symmetric M) (h_diag : ∀ i, M i i = 0) (h_points : points_condition n M) : 
  Even n :=
sorry

end even_n_of_points_condition_l432_432001


namespace statement_fred_reaches_target_after_40_minutes_l432_432499

/-- Define the starting position and the target position -/
def start_pos : ℕ × ℕ × ℕ × ℕ := (0, 0, 0, 0)
def target_pos : ℕ × ℕ × ℕ × ℕ := (10, 10, 10, 10)

/-- Condition to be satisfied for each move -/
def valid_move (a b : ℕ × ℕ × ℕ × ℕ) :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2 + (a.4 - b.4)^2 = 4 ∧
  abs ((a.1 + a.2 + a.3 + a.4) - (b.1 + b.2 + b.3 + b.4)) = 2

/--
  Theorem statement: Prove the number of ways Fred can reach (10,10,10,10)
  from (0,0,0,0) in exactly 40 minutes, where moves comply with valid_move,
  is equal to the given binomial coefficient calculation.
-/
theorem fred_reaches_target_after_40_minutes :
  (card { f : ℕ → ℕ × ℕ × ℕ × ℕ // f 0 = start_pos ∧ f 40 = target_pos ∧
     (∀ t < 40, valid_move (f t) (f (t+1))) } = (binomial 40 10) * (binomial 40 20)^3) := sorry

end statement_fred_reaches_target_after_40_minutes_l432_432499


namespace village_last_weeks_village_last_5_weeks_l432_432026

theorem village_last_weeks (lead_vampire_drain : ℕ)
  (num_vampires : ℕ) (vampire_drain : ℕ) 
  (alpha_werewolf_drain : ℕ) (num_werewolves : ℕ)
  (werewolf_drain : ℕ) (ghost_drain : ℕ) 
  (village_population : ℕ) : ℕ :=
  let total_consumption_per_week := 
    lead_vampire_drain + 
    num_vampires * vampire_drain +
    alpha_werewolf_drain + 
    num_werewolves * werewolf_drain + 
    ghost_drain
  in 
  (village_population / total_consumption_per_week)

/- Now we define the conditions from the problem -/

def lead_vampire_drain := 5
def num_vampires := 3
def vampire_drain := 5
def alpha_werewolf_drain := 7
def num_werewolves := 2
def werewolf_drain := 5
def ghost_drain := 2
def village_population := 200

theorem village_last_5_weeks : 
  village_last_weeks lead_vampire_drain num_vampires vampire_drain alpha_werewolf_drain num_werewolves werewolf_drain ghost_drain village_population = 5 := 
  by 
    sorry

end village_last_weeks_village_last_5_weeks_l432_432026


namespace scientific_notation_of_274M_l432_432084

theorem scientific_notation_of_274M :
  274000000 = 2.74 * 10^8 := 
by 
  sorry

end scientific_notation_of_274M_l432_432084


namespace number_101_in_pascals_triangle_l432_432175

/-- Prove that the number 101 appears in exactly one row of Pascal's Triangle, specifically the 101st row. -/
theorem number_101_in_pascals_triangle : 
  ∃! n : ℕ, (∃ k : ℕ, k ≤ n ∧ k ≠ 0 ∧ binom n k = 101) ∧ n = 101 :=
by sorry

end number_101_in_pascals_triangle_l432_432175


namespace max_value_of_f_l432_432583

section
  -- Define the new operation "⊕"
  def op (a b : ℝ) : ℝ :=
    if a ≥ b then a else b^2

  -- The function f(x)
  def f (x : ℝ) : ℝ :=
    (op 1 x) * x - (op 2 x)

  -- Prove the maximum value of f(x) on the interval [-2, 2] is 6
  theorem max_value_of_f :
    ∃ x ∈ set.Icc (-2 : ℝ) 2, f x = 6 ∧
    ∀ y ∈ set.Icc (-2 : ℝ) 2, f y ≤ 6 := 
  by
    sorry
end

end max_value_of_f_l432_432583


namespace find_integer_with_prime_factors_l432_432816

theorem find_integer_with_prime_factors : 
  ∃ (a : ℤ), (∃ p1 p2 p3 p4 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ a = p1 * p2 * p3 * p4) ∧ 
             (∃ q1 q2 q3 q4 : ℕ, q1 = p1 * p1 ∧ q2 = p2 * p2 ∧ q3 = p3 * p3 ∧ q4 = p4 * p4 ∧ q1 + q2 + q3 + q4 = 476) ∧ 
             a = 1989 :=
begin
  sorry
end

end find_integer_with_prime_factors_l432_432816


namespace total_investment_is_correct_l432_432035

-- Define principal, rate, and number of years
def principal : ℝ := 8000
def rate : ℝ := 0.04
def years : ℕ := 10

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem total_investment_is_correct :
  compound_interest principal rate years = 11842 :=
by
  sorry

end total_investment_is_correct_l432_432035


namespace nineteen_numbers_sum_non_negative_l432_432204

theorem nineteen_numbers_sum_non_negative (x : Fin 19 → ℤ) 
  (h : ∀ i : Fin 17, x i + x (i + 1) + x (i + 2) > 0) : 
  (∑ i, x i) ≥ 0 := 
sorry

end nineteen_numbers_sum_non_negative_l432_432204


namespace platform_length_l432_432397

def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def distance_covered (speed_mps time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

def length_of_platform (total_distance length_of_train : ℝ) : ℝ :=
  total_distance - length_of_train

theorem platform_length
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (time_in_sec : ℝ)
  (speed_in_mps : ℝ := kmph_to_mps speed_in_kmph)
  (total_distance : ℝ := distance_covered speed_in_mps time_in_sec)
  (platform_length : ℝ := length_of_platform total_distance length_of_train) :
  platform_length = 233.33 :=
sorry

end platform_length_l432_432397


namespace max_min_row_sum_l432_432876

def value_in_grid (grid : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  grid i j ∈ {1, 2, 3, 4, 5}

def correct_distribution (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ n ∈ {1, 2, 3, 4, 5}, (Finset.card (Finset.filter (λ ij, grid ij.1 ij.2 = n) (Finset.product (Finset.range 5) (Finset.range 5)))) = 5

def valid_row (grid : ℕ → ℕ → ℕ) (i : ℕ) : Prop :=
  ∀ j1 j2 < 5, abs (grid i j1 - grid i j2) ≤ 2

def row_sum (grid : ℕ → ℕ → ℕ) (i : ℕ) : ℕ :=
  (Finset.range 5).sum (λ j, grid i j)

def min_row_sum (grid : ℕ → ℕ → ℕ) : ℕ :=
  (Finset.range 5).inf (λ i, row_sum grid i)

theorem max_min_row_sum (grid : ℕ → ℕ → ℕ) :
  (∀ i j < 5, value_in_grid grid i j) →
  (correct_distribution grid) →
  (∀ i < 5, valid_row grid i) →
  min_row_sum grid = 10 :=
sorry

end max_min_row_sum_l432_432876


namespace lateral_surface_area_cylinder_l432_432216

noncomputable def lateralSurfaceAreaOfCylinder {a : ℝ} (h : a > 0) : ℝ :=
  let lateral_edge := (5 / 2) * a
  let x := 2 / 3
  let r := (a * Real.sqrt 2) / 6
  let height := (a * Real.sqrt 23) / 3
  2 * Real.pi * r * height

theorem lateral_surface_area_cylinder (a : ℝ) (h : a > 0) :
  lateralSurfaceAreaOfCylinder h = (Real.pi * a^2 * Real.sqrt 46) / 9 :=
begin
  sorry
end

end lateral_surface_area_cylinder_l432_432216


namespace solution_to_equation_l432_432074

def star_operation (a b : ℝ) : ℝ := a^2 - 2 * a * b + b^2

theorem solution_to_equation : ∀ (x : ℝ), star_operation (x - 4) 1 = 0 → x = 5 :=
by
  intro x
  assume h
  -- Skipping the proof steps with sorry
  sorry

end solution_to_equation_l432_432074


namespace remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l432_432796

-- Definitions of angle types
def obtuse_angle (θ : ℝ) := θ > 90 ∧ θ < 180
def right_angle (θ : ℝ) := θ = 90
def acute_angle (θ : ℝ) := θ > 0 ∧ θ < 90
def straight_angle (θ : ℝ) := θ = 180

-- Proposition 1: Remaining angle when an obtuse angle is cut by a right angle is acute
theorem remaining_angle_obtuse_cut_by_right_is_acute (θ : ℝ) (φ : ℝ) 
    (h1 : obtuse_angle θ) (h2 : right_angle φ) : acute_angle (θ - φ) :=
  sorry

-- Proposition 2: Remaining angle when a straight angle is cut by an acute angle is obtuse
theorem remaining_angle_straight_cut_by_acute_is_obtuse (α : ℝ) (β : ℝ) 
    (h1 : straight_angle α) (h2 : acute_angle β) : obtuse_angle (α - β) :=
  sorry

end remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l432_432796


namespace distance_AC_probability_l432_432818

open Real

theorem distance_AC_probability :
  let A := (0, -10 : ℝ)
  let B := (0, 0 : ℝ)
  let O := (0, -4 : ℝ)
  let C_x (β : ℝ) := 7 * sin β
  let C_y (β : ℝ) := 7 * cos β
  let AC (β : ℝ) := sqrt (C_x β ^ 2 + (C_y β + 10) ^ 2)
  let AO := 6
  ∀ β ∈ Set.Ioo 0 (π/2),
  (AC β < 2 * AO) ↔ β < arc tan(sqrt 371 / 8) := by
  sorry

end distance_AC_probability_l432_432818


namespace area_BCF_result_l432_432644

-- Define the problem conditions
variable (A B C D E F : Type) 
variable [LinearOrderedField A]
variables {AB AC BC : A}
variable [Nonempty (AffinePlane A)]

-- Given conditions
axiom triangle_ABC : Triangle A B C
axiom AB_length : AB = 5
axiom AC_length : AC = 4
axiom reflection_D : D = reflection C across AB
axiom reflection_E : E = reflection B across AC
axiom collinear_DAE : Collinear D A E
axiom intersect_DB_EC_at_F : ∃ F, line D B ∩ line E C = {F}

-- Define the area
noncomputable def area_BCF : A := area (triangle B C F)

-- The statement to prove
theorem area_BCF_result : area_BCF = 60 * √3 / 7 := sorry

end area_BCF_result_l432_432644


namespace distance_from_point_M_to_origin_l432_432579

theorem distance_from_point_M_to_origin : 
  let x : ℝ := 5
  let y : ℝ := -12
  (real.sqrt (x^2 + y^2) = 13) :=
by
  let x := 5
  let y := -12
  have d_eq : real.sqrt (x^2 + y^2) = 13
  sorry

end distance_from_point_M_to_origin_l432_432579


namespace parabola_directrix_l432_432718

theorem parabola_directrix (p : ℝ) (h : 2 * p = 1) : x = - (p / 2) := 
by 
  have p_eq : p = 1 / 2 := by { linarith }
  rw p_eq
  field_simp
  norm_num

end parabola_directrix_l432_432718


namespace cube_volume_from_lateral_surface_area_l432_432778

theorem cube_volume_from_lateral_surface_area 
  (lateral_surface_area : ℝ) 
  (h : lateral_surface_area = 105.6) : 
  ∃ (volume : ℝ), volume ≈ 135.85 :=
by
  sorry

end cube_volume_from_lateral_surface_area_l432_432778


namespace sum_sequence_l432_432217

def a (n : ℕ) : ℚ :=
if n = 0 then 0 else
if n = 1 then 1 / 2 else
0 -- placeholder for recursive definition, but let's define it properly

noncomputable def a_recurrence (n : ℕ) : Prop :=
(n+1 : ℕ) * (n * a n * a (n+1) + a (n+1)) - n * a n = 0

noncomputable def S (n : ℕ) : ℚ :=
∑ i in finset.range n, a i / (i + 2)

theorem sum_sequence (n : ℕ) (h1 : a 1 = 1 / 2)
  (h_recurr : ∀ n, a_recurrence n) : S n = n * (n + 3) / (4 * (n + 1) * (n + 2)) := sorry

end sum_sequence_l432_432217


namespace p_q_sum_l432_432819

-- Define a structure for the geometric properties of the inscribed octagon
structure InscribedOctagon (r : ℝ) (circle_center : (ℝ × ℝ)) :=
  (side_length : ℕ → ℝ)
  (side_length_alternating : ∀ {n}, n % 2 = 0 → side_length n = 4 ∧ n % 2 = 1 → side_length n = 6)
  (total_sides : ∀ n, 0 ≤ n ∧ n < 8)

noncomputable def find_pq_sum {r : ℝ} (O : InscribedOctagon r (0,0)) : ℕ × ℕ :=
sorry

theorem p_q_sum (O: InscribedOctagon r (0,0)) :  let (p, q) := find_pq_sum O in Nat.coprime p q → p + q == 42 := 
sorry

end p_q_sum_l432_432819


namespace range_of_a_for_propositions_p_and_q_l432_432839

theorem range_of_a_for_propositions_p_and_q :
  {a : ℝ | ∃ x, (x^2 + 2 * a * x + 4 = 0) ∧ (3 - 2 * a > 1)} = {a | a ≤ -2} := sorry

end range_of_a_for_propositions_p_and_q_l432_432839


namespace otimes_example_l432_432064

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432064


namespace cos_x_in_terms_of_c_d_l432_432554

-- Define the parameters and the conditions
variables {c d x : ℝ}

-- Assume the conditions
variables (hc : c > d) (hd : d > 0) (hx1 : 0 < x) (hx2 : x < π / 2)
variables (h : tan x = 3 * c * d / (c^2 - d^2))

-- Define the theorem statement
theorem cos_x_in_terms_of_c_d (hc : c > d) (hd : d > 0) (hx1 : 0 < x) (hx2 : x < π / 2)
  (h : tan x = 3 * c * d / (c^2 - d^2)) :
  cos x = (c^2 - d^2) / real.sqrt (c^4 + 7 * c^2 * d^2 + d^4) :=
sorry

end cos_x_in_terms_of_c_d_l432_432554


namespace midpoint_perpendicular_to_chord_l432_432696

theorem midpoint_perpendicular_to_chord {O A B M : Point}
  (hO : O.is_center)
  (hChord : Chord O A B)
  (hMidpoint : Midpoint O A B M) :
  Perpendicular O M A B :=
sorry

end midpoint_perpendicular_to_chord_l432_432696


namespace number_of_subsets_of_set_l432_432734

theorem number_of_subsets_of_set {n : ℕ} (h : n = 2016) :
  (2^2016) = 2^2016 :=
by
  sorry

end number_of_subsets_of_set_l432_432734


namespace integral_cos_eq_zero_l432_432872

open Real

theorem integral_cos_eq_zero : ∫ x in 0..π, cos x = 0 :=
by
  sorry

end integral_cos_eq_zero_l432_432872


namespace sum_base5_eq_l432_432892

theorem sum_base5_eq :
  (432 + 43 + 4 : ℕ) = 1034 :=
by sorry

end sum_base5_eq_l432_432892


namespace largest_lcm_value_l432_432760

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 := by
sorry

end largest_lcm_value_l432_432760


namespace cantor_length_formula_l432_432453

noncomputable def cantor_length : ℕ → ℚ
| 0 => 1
| (n+1) => 2/3 * cantor_length n

theorem cantor_length_formula (n : ℕ) : cantor_length n = (2/3 : ℚ)^(n-1) :=
  sorry

end cantor_length_formula_l432_432453


namespace janet_gas_usage_l432_432598

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20
def total_driving_distance : ℕ := (2 * distance_dermatologist) + (2 * distance_gynecologist)
def gas_used : ℝ := total_driving_distance / car_efficiency

theorem janet_gas_usage : gas_used = 8 := by
  sorry

end janet_gas_usage_l432_432598


namespace find_x_l432_432136

-- Definitions based on provided conditions

def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 7
def rectangle_area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def rectangle_perimeter (x : ℝ) : ℝ := 2 * rectangle_length x + 2 * rectangle_width x

-- Theorem statement
theorem find_x (x : ℝ) (h : rectangle_area x = 2 * rectangle_perimeter x) : x = 1 := 
sorry

end find_x_l432_432136


namespace max_product_distance_l432_432912

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l432_432912


namespace correct_calculation_l432_432179

theorem correct_calculation (x : ℝ) (h : 63 + x = 69) : 36 / x = 6 :=
by
  sorry

end correct_calculation_l432_432179


namespace isabelle_needs_more_weeks_l432_432220

variable (standard_price : ℝ) (saved : ℝ) (earnings_per_week : ℝ)
variable (discount_isabelle : ℝ) (discount_alex : ℝ) (discount_anna_nick : ℝ)

def ticket_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

noncomputable def total_cost 
  (price : ℝ) (discount_isabelle discount_alex discount_anna_nick : ℝ) 
  : ℝ :=
  ticket_price price discount_isabelle +
  ticket_price price discount_alex +
  2 * ticket_price price discount_anna_nick

noncomputable def remaining_amount (total_cost saved : ℝ) : ℝ := total_cost - saved

noncomputable def weeks_needed (remaining_amount earnings_per_week : ℝ) : ℕ :=
  (remaining_amount / earnings_per_week).ceil.to_nat

theorem isabelle_needs_more_weeks
  (standard_price == 50) (saved == 34) (earnings_per_week == 8)
  (discount_isabelle == 0.30) (discount_alex == 0.50) 
  (discount_anna_nick == 0.10) :
  weeks_needed 
    (remaining_amount (total_cost standard_price discount_isabelle discount_alex discount_anna_nick) saved) 
    earnings_per_week 
  = 15 :=
  sorry

end isabelle_needs_more_weeks_l432_432220


namespace intervals_of_monotonicity_l432_432755

open Real

def is_monotonic_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

-- Conditions
def omega : ℝ := 2
def phi : ℝ := π / 6

noncomputable def f (x : ℝ) : ℝ := sin (omega * x + phi)

-- Proof problem
theorem intervals_of_monotonicity :
  let intervals := λ (k : ℤ), Icc (k * π - π / 3) (k * π + π / 6) in
  ∀ k : ℤ, is_monotonic_increasing f (intervals k) :=
sorry

end intervals_of_monotonicity_l432_432755


namespace simple_interest_rate_l432_432394

theorem simple_interest_rate (SI P T : ℝ) (hSI : SI = 1000) (hP : P = 2500) (hT : T = 4) :
  (SI * 100) / (P * T) = 10 :=
by
  rw [hSI, hP, hT]
  norm_num
  sorry

end simple_interest_rate_l432_432394


namespace initial_amount_l432_432388

theorem initial_amount (M : ℝ) 
  (H1 : M * (2/3) * (4/5) * (3/4) * (5/7) * (5/6) = 200) : 
  M = 840 :=
by
  -- Proof to be provided
  sorry

end initial_amount_l432_432388


namespace cube_positive_integers_solution_l432_432479

theorem cube_positive_integers_solution (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∃ k : ℕ, 2^(Nat.factorial a) + 2^(Nat.factorial b) + 2^(Nat.factorial c) = k^3) ↔ 
    ( (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
      (a = 1 ∧ b = 2 ∧ c = 1) ∨ 
      (a = 2 ∧ b = 1 ∧ c = 1) ) :=
by
  sorry

end cube_positive_integers_solution_l432_432479


namespace otimes_example_l432_432062

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l432_432062


namespace sum_first_10_terms_l432_432542

variable (a : ℕ → ℕ)

def condition (p q : ℕ) : Prop :=
  p + q = 11 ∧ p < q

axiom condition_a_p_a_q : ∀ (p q : ℕ), (condition p q) → (a p + a q = 2^p)

theorem sum_first_10_terms (a : ℕ → ℕ) (h : ∀ (p q : ℕ), condition p q → a p + a q = 2^p) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 62) :=
by 
  sorry

end sum_first_10_terms_l432_432542


namespace average_goals_per_game_l432_432248

theorem average_goals_per_game (pizzas slices_per_pizza games : ℕ) (h1 : pizzas = 6) (h2 : slices_per_pizza = 12) (h3 : games = 8) : 
  (pizzas * slices_per_pizza) / games = 9 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end average_goals_per_game_l432_432248


namespace unicorn_rope_problem_l432_432402

-- Define the conditions
def length_of_rope : ℝ := 30
def radius_of_tower : ℝ := 10
def tether_height : ℝ := 5
def distance_to_tower : ℝ := 5

-- Variables to solve for
variables (a b c : ℕ)

-- Define the length of the rope touching the tower using the variables a, b, c
def rope_length_touching_tower (a b c : ℕ) : ℝ := (a - Real.sqrt b) / c

-- State the theorem to prove
theorem unicorn_rope_problem :
  let x := rope_length_touching_tower 90 1500 3 in
  x = (90 - Real.sqrt 1500) / 3 ∧ 3 = 3 →
  a = 90 ∧ b = 1500 ∧ c = 3 →
  a + b + c = 1593 :=
by
  sorry

end unicorn_rope_problem_l432_432402


namespace min_value_exp_sum_l432_432128

theorem min_value_exp_sum (a b : ℝ) (h : a + b = 2) : 3^a + 3^b ≥ 6 :=
by sorry

end min_value_exp_sum_l432_432128


namespace terminal_side_in_third_quadrant_l432_432141

variable (α : ℝ)

-- Given conditions
def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

-- The location of the point P in terms of angle α
def point_P : ℝ × ℝ := (Real.sin α * Real.cos α, 2 * Real.cos α)

-- Proving the statement equivalent to the problem condition
theorem terminal_side_in_third_quadrant
  (h1 : point_in_fourth_quadrant (point_P α)) :
  (Real.sin α < 0 ∧ Real.cos α < 0) :=
sorry

end terminal_side_in_third_quadrant_l432_432141


namespace quadratic_sum_l432_432390

noncomputable def g (x : ℝ) : ℝ := 2 * x^2 + B * x + C

-- Given conditions
def g_at_1 (h : g 1 = 3) := h  -- g(1) = 3
def g_at_2 (h : g 2 = 0) := h  -- g(2) = 0

-- The main theorem to prove
theorem quadratic_sum (B C : ℝ) (h1 : g 1 = 3) (h2 : g 2 = 0) : 2 + B + C + 2 * C = 23 := by
  sorry

end quadratic_sum_l432_432390


namespace cot_sum_condition_l432_432325

noncomputable def points_collinear (A B C A1 B1 C1 : ℂ) := 
  ∃ k : ℂ, A1 = k * B1 + (1 - k) * C1

theorem cot_sum_condition (A B C A1 B1 C1 : ℂ) 
  (h1: A + B + C = 0)
  (h2: A1 = -1/2 * (A + (B - C) * complex.I))
  (h3: B1 = -1/2 * (B + (C - A) * complex.I))
  (h4: C1 = -1/2 * (C + (A - B) * complex.I)):
  points_collinear A B C A1 B1 C1 ↔ real.cot (complex.arg A) + real.cot (complex.arg B) + real.cot (complex.arg C) = 2 := 
sorry

end cot_sum_condition_l432_432325


namespace tangent_line_circumcircle_l432_432815

theorem tangent_line_circumcircle
    (A B C O B' C' : Point)
    (h_acute_isosceles_triangle : acute_isosceles_triangle A B C)
    (h_inscribed_in_circle : inscribed_in_circle A B C O)
    (h_intersections : intersections BO CO AC AB B' C')
    (h_parallel : parallel (line_through C') (line_through C A)) :
    tangent (circumcircle B' O C) (line_through C') :=
by
    sorry

end tangent_line_circumcircle_l432_432815


namespace probability_W_permutation_l432_432798

section WPermutation

open Finset
open Fintype

def is_W_permutation (P : equiv.perm (fin 5)) : Prop :=
  P 0 > P 1 ∧ P 1 < P 2 ∧ P 2 > P 3 ∧ P 3 < P 4

def count_W_permutations : ℕ :=
  univ.filter is_W_permutation.card

theorem probability_W_permutation : (count_W_permutations : ℚ) / (card (univ : finset (equiv.perm (fin 5)))) = 2 / 15 :=
by sorry

end WPermutation

end probability_W_permutation_l432_432798


namespace polynomial_expansion_identity_l432_432556

variable (a0 a1 a2 a3 a4 : ℝ)

theorem polynomial_expansion_identity
  (h : (2 - (x : ℝ))^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a0 - a1 + a2 - a3 + a4 = 81 :=
sorry

end polynomial_expansion_identity_l432_432556


namespace cos_gamma_fraction_sum_l432_432197

theorem cos_gamma_fraction_sum {R : Type} [field R] [algebra ℚ R] 
    (chord_len_5 chord_len_6 chord_len_7 : R)
    (γ δ : ℝ) 
    (h1 : chord_len_5 = (5 : R)) (h2 : chord_len_6 = (6 : R)) (h3 : chord_len_7 = (7 : R))
    (h4 : γ + δ < π) 
    (h5 : ∃ q : ℚ, cos γ = q) 
  : 
  let cos_γ := real.cos γ in
  let num_denom_sum := ((rat.denom (rat.of_real cos_γ)) + (rat.num (rat.of_real cos_γ))) in
  num_denom_sum = 50 := 
begin
  sorry
end

end cos_gamma_fraction_sum_l432_432197


namespace tracy_initial_candies_l432_432753

theorem tracy_initial_candies (x : ℕ) (consumed_candies : ℕ) (remaining_candies_given_rachel : ℕ) (remaining_candies_given_monica : ℕ) (candies_eaten_by_tracy : ℕ) (candies_eaten_by_mom : ℕ) 
  (brother_candies_taken : ℕ) (final_candies : ℕ) (h_consume : consumed_candies = 2 / 5 * x) (h_remaining1 : remaining_candies_given_rachel = 1 / 3 * (3 / 5 * x)) 
  (h_remaining2 : remaining_candies_given_monica = 1 / 6 * (3 / 5 * x)) (h_left_after_friends : 3 / 5 * x - (remaining_candies_given_rachel + remaining_candies_given_monica) = 3 / 10 * x)
  (h_candies_left : 3 / 10 * x - (candies_eaten_by_tracy + candies_eaten_by_mom) = final_candies + brother_candies_taken) (h_eaten_tracy : candies_eaten_by_tracy = 10)
  (h_eaten_mom : candies_eaten_by_mom = 10) (h_final : final_candies = 6) (h_brother_bound : 2 ≤ brother_candies_taken ∧ brother_candies_taken ≤ 6) : x = 100 := 
by 
  sorry

end tracy_initial_candies_l432_432753


namespace atomic_weight_S_is_correct_l432_432886

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end atomic_weight_S_is_correct_l432_432886


namespace books_per_bookshelf_l432_432827

theorem books_per_bookshelf (total_bookshelves total_books books_per_bookshelf : ℕ)
  (h1 : total_bookshelves = 23)
  (h2 : total_books = 621)
  (h3 : total_books = total_bookshelves * books_per_bookshelf) :
  books_per_bookshelf = 27 :=
by 
  -- Proof goes here
  sorry

end books_per_bookshelf_l432_432827


namespace income_recording_l432_432212

theorem income_recording (exp_200 : Int := -200) (income_60 : Int := 60) : exp_200 = -200 → income_60 = 60 →
  (income_60 > 0) :=
by
  intro h_exp h_income
  sorry

end income_recording_l432_432212


namespace knights_count_l432_432639

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432639


namespace fiftieth_term_is_260_l432_432305

def positive_multiples_of_four_with_digit_two (n : ℕ) : Prop :=
  n % 4 = 0 ∧ (∃ d : ℕ, d ∈ (Nat.digits 10 n) ∧ d = 2)

def sequence_term (k : ℕ) : ℕ :=
  (Nat.find_greatest (λ n, n % 4 = 0 ∧ (∃ d : ℕ, d ∈ (Nat.digits 10 n) ∧ d = 2)) k)

theorem fiftieth_term_is_260 : sequence_term 50 = 260 := 
sorry

end fiftieth_term_is_260_l432_432305


namespace Lily_balls_is_3_l432_432826

-- Definitions from conditions
variable (L : ℕ)

def Frodo_balls := L + 8
def Brian_balls := 2 * (L + 8)

axiom Brian_has_22 : Brian_balls L = 22

-- The goal is to prove that Lily has 3 tennis balls
theorem Lily_balls_is_3 : L = 3 :=
by
  sorry

end Lily_balls_is_3_l432_432826


namespace knights_count_l432_432614

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432614


namespace max_value_9_l432_432665

noncomputable def max_ab_ac_bc (a b c : ℝ) : ℝ :=
  max (a * b) (max (a * c) (b * c))

theorem max_value_9 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 27) :
  max_ab_ac_bc a b c = 9 :=
sorry

end max_value_9_l432_432665


namespace sum_of_digits_N_l432_432282

def N : ℕ := (List.range 51).sum (λ i, 10^i + 1)

theorem sum_of_digits_N : (N.digits.sum : ℕ) = 58 := by
  sorry

end sum_of_digits_N_l432_432282


namespace point_in_fourth_quadrant_l432_432206

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l432_432206


namespace find_line_eq_l432_432482

open Real

noncomputable def intersection_point : ℝ × ℝ :=
let (x, y) := ((1 : ℝ), 2) in
(x, y)

noncomputable def is_solution (x y : ℝ) : Prop :=
(x = 1 ∧ y = 2) ∨ (3 * x + 4 * y - 11 = 0)

noncomputable def distance_constraint (x y : ℝ) : Prop :=
(∃ k : ℝ, y = k * (x - 1) + 2 ∧ abs (4 - (2 - k)) / sqrt (k^2 + 1) = 1)

theorem find_line_eq : 
  is_solution 1 2 ∧ (distance_constraint 1 2 ↔ 
  (abs (4 - 2 - (-3/4)) / sqrt ((-3/4)^2 + 1) = 1)) →
  (intermediate_value 1 = 1 ∨ 3 * intermediate_value 1 + 4 * intermediate_value 2 - 11 = 0) :=
by
sorry

end find_line_eq_l432_432482


namespace distance_between_vertices_hyperbola_l432_432091

-- Defining the hyperbola equation and necessary constants
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2) / 64 - (y^2) / 81 = 1

-- Proving the distance between the vertices is 16
theorem distance_between_vertices_hyperbola : ∀ x y : ℝ, hyperbola_eq x y → 16 = 16 :=
by
  intros x y h
  sorry

end distance_between_vertices_hyperbola_l432_432091


namespace regina_total_cost_l432_432269

-- Definitions
def daily_cost : ℝ := 30
def mileage_cost : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 450
def fixed_fee : ℝ := 15

-- Proposition for total cost
noncomputable def total_cost : ℝ := daily_cost * days_rented + mileage_cost * miles_driven + fixed_fee

-- Theorem statement
theorem regina_total_cost : total_cost = 217.5 := by
  sorry

end regina_total_cost_l432_432269


namespace women_in_room_l432_432591

theorem women_in_room (M W : ℕ) 
  (h1 : 9 * M = 7 * W) 
  (h2 : M + 5 = 23) : 
  3 * (W - 4) = 57 :=
by
  sorry

end women_in_room_l432_432591


namespace expected_rainfall_total_correct_l432_432027

def daily_rain := [⟨0.3, 0⟩, ⟨0.35, 3⟩, ⟨0.35, 8⟩] -- Probabilities and inches of rain per day

noncomputable def expected_rain_one_day : ℝ :=
  daily_rain.foldl (λ acc p, acc + p.1 * p.2) 0

def days := 5 -- Number of days (Monday to Friday)

noncomputable def expected_total_rain : ℝ :=
  days * expected_rain_one_day

theorem expected_rainfall_total_correct : expected_total_rain = 19.25 :=
by
  sorry

end expected_rainfall_total_correct_l432_432027


namespace train_speed_is_correct_l432_432729

-- Define the conditions.
def length_of_train : ℕ := 1800 -- Length of the train in meters.
def time_to_cross_platform : ℕ := 60 -- Time to cross the platform in seconds (1 minute).

-- Define the statement that needs to be proved.
def speed_of_train : ℕ := (2 * length_of_train) / time_to_cross_platform

-- State the theorem.
theorem train_speed_is_correct :
  speed_of_train = 60 := by
  sorry -- Proof is not required.

end train_speed_is_correct_l432_432729


namespace rational_product_form_l432_432268

theorem rational_product_form (g : ℕ) (hg : 0 < g) (w : ℚ) (hw : 1 < w) :
  ∃ (k s : ℕ), (g < k) ∧ (w = (∏ i in finset.range (s + 1), (1 : ℚ) + 1/(k + i))) := 
sorry

end rational_product_form_l432_432268


namespace probability_one_letter_remains_l432_432405

/-- 
Given 1001 letters chosen independently and uniformly at random from the set {a, b, c},
the probability that Alice can perform a sequence of moves which results in one letter 
remaining on the blackboard is (3^1000 - 1) / (4 * 3^999). 
-/
theorem probability_one_letter_remains (S : Finset (Fin 3)) (letters : Vector (Fin 3) 1001)
  (random_choice : ∀ i, letters.nth i ∈ S) :
  ∃ p : ℚ, p = (3^1000 - 1) / (4 * 3^999) :=
sorry

end probability_one_letter_remains_l432_432405


namespace Dabbie_spends_99_dollars_l432_432060

noncomputable def total_cost_turkeys (w1 w2 w3 w4 : ℝ) (cost_per_kg : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) * cost_per_kg

theorem Dabbie_spends_99_dollars :
  let w1 := 6
  let w2 := 9
  let w3 := 2 * w2
  let w4 := (w1 + w2 + w3) / 2
  let cost_per_kg := 2
  total_cost_turkeys w1 w2 w3 w4 cost_per_kg = 99 := 
by
  sorry

end Dabbie_spends_99_dollars_l432_432060


namespace number_of_friends_l432_432413

-- Definitions based on the conditions
def totalLemonHeads : ℕ := 72
def lemonHeadsPerFriend : ℕ := 12

-- The theorem stating the problem
theorem number_of_friends : ∃ F : ℕ, totalLemonHeads = lemonHeadsPerFriend * F ∧ F = 6 :=
by
  use 6
  split
  · -- Proof that 72 = 12 * 6
    unfold totalLemonHeads lemonHeadsPerFriend
    rfl
  · -- Proof that F equals 6
    rfl

end number_of_friends_l432_432413


namespace geometric_sequence_S6_l432_432199

theorem geometric_sequence_S6 
  (a : ℕ → ℝ) (a1 : a 1 = 1) (pos_a : ∀ n : ℕ, a n > 0) 
  (S : ℕ → ℝ) (sum_S : ∀ n : ℕ, S n = (∑ i in range n, a (i + 1))) 
  (arith_mean : a 2 = (-a 3 + a 4) / 2) :
  S 6 = 63 :=
sorry

end geometric_sequence_S6_l432_432199


namespace projection_magnitude_ratio_l432_432653

variables {ℝ : Type*} [nondiscrete_normed_field ℝ]

noncomputable def projection (u v : ℝ^n) : ℝ^n := ((inner u v) / (inner v v)) • v

-- Define the vectors
variables (v w p q : ℝ^n)

-- Conditions as definitions
def p_def : p = projection v w := rfl
def q_def : q = projection p v := rfl
def norm_ratio : ∥p∥ / ∥v∥ = 3 / 4 := sorry

-- The proof goal
theorem projection_magnitude_ratio : ∥q∥ / ∥v∥ = 9 / 16 :=
sorry

end projection_magnitude_ratio_l432_432653


namespace point_in_fourth_quadrant_l432_432208

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l432_432208


namespace corn_stalks_per_bushel_l432_432414

theorem corn_stalks_per_bushel (rows : ℕ) (stalks_per_row : ℕ) (bushels : ℕ) (total_stalks : rows * stalks_per_row = 400) (total_bushels : bushels = 50) :
  (rows * stalks_per_row) / bushels = 8 :=
by {
  -- Here we would include the proof, but for now we conclude with "sorry"
  sorry,
}

end corn_stalks_per_bushel_l432_432414


namespace train_pass_telegraph_post_time_l432_432775

theorem train_pass_telegraph_post_time
  (D : ℝ) (S_kmph : ℝ)
  (hD : D = 90)
  (hS : S_kmph = 36) :
  let S := S_kmph * (1000 / 3600) in
  let t := D / S in
  t = 9 := 
by
  sorry

end train_pass_telegraph_post_time_l432_432775


namespace class_trip_contributions_l432_432758

theorem class_trip_contributions (x y : ℕ) :
  (x + 5) * (y + 6) = x * y + 792 ∧ (x - 4) * (y + 4) = x * y - 388 → x = 213 ∧ y = 120 := 
by
  sorry

end class_trip_contributions_l432_432758


namespace seq_an_formula_sum_seq_bn_result_l432_432939

noncomputable def seq_a (n : ℕ) : ℕ → ℝ
| 0 := 1 / 2
| (n + 1) := (1 / 2) ^ (n + 1)

def sum_seq_a (n : ℕ) : ℝ :=
  let partial_sum := ∑ i in range n, seq_a i
  partial_sum

theorem seq_an_formula (n : ℕ) : seq_a n = (1 / 2) ^ n :=
  sorry

noncomputable def seq_T (n : ℕ) : ℝ :=
  ∏ i in range n, seq_a i

noncomputable def seq_bn (n : ℕ) : ℝ :=
  (-1) ^ n * (Real.log (seq_T n) / Real.log 2)

noncomputable def sum_seq_bn (n : ℕ) : ℝ :=
  ∑ i in range n, seq_bn i

theorem sum_seq_bn_result (n : ℕ) : 
  sum_seq_bn n = 
    if Even n then -n
    else (n^2 - n + 2) / 2 :=
  sorry

end seq_an_formula_sum_seq_bn_result_l432_432939


namespace pyramid_height_l432_432011

theorem pyramid_height
  (perimeter : ℝ)
  (apex_vertex_distance : ℝ)
  (perimeter_eq : perimeter = 32)
  (apex_vertex_distance_eq : apex_vertex_distance = 12) :
  ∃ h : ℝ, h = 4 * Real.sqrt 7 :=
by 
  use 4 * Real.sqrt 7
  sorry

end pyramid_height_l432_432011


namespace knights_count_l432_432631

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l432_432631


namespace health_risk_factors_l432_432821

noncomputable def p := 59
noncomputable def q := 76

theorem health_risk_factors {
  p + q = 135
} :=
by
  -- sorry is used to skip the proof
  sorry

end health_risk_factors_l432_432821


namespace three_pow_expr_l432_432901

theorem three_pow_expr (m n : ℝ) (h1 : 9^m = 3) (h2 : 27^n = 4) : 3^(2*m + 3*n) = 12 :=
by
  sorry

end three_pow_expr_l432_432901


namespace ellipse_eccentricity_l432_432150

theorem ellipse_eccentricity :
  ∃ (a : ℝ), (a^2 - 4 = 4) ∧ (∀ e : ℝ, e = 2 / a → e = (ℝ.sqrt 2) / 2) :=
by
  use 2 * ℝ.sqrt 2
  split
  -- proof that a^2 - 4 = 4
  sorry
  -- proof that eccentricity e = (ℝ.sqrt 2) / 2
  intro e he
  rw [he]
  have ha : a = 2 * ℝ.sqrt 2 := by
    sorry
  rw [ha]
  simp
  sorry

end ellipse_eccentricity_l432_432150


namespace knights_count_l432_432617

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l432_432617


namespace magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432471

noncomputable def magnitude_prod_4_minus_3i_4_plus_3i : ℝ := |complex.abs (4 - 3 * complex.I) * complex.abs (4 + 3 * complex.I)|

theorem magnitude_prod_4_minus_3i_4_plus_3i_eq_25 : magnitude_prod_4_minus_3i_4_plus_3i = 25 :=
by
  sorry

end magnitude_prod_4_minus_3i_4_plus_3i_eq_25_l432_432471


namespace knights_count_l432_432640

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l432_432640


namespace find_f_l432_432358

theorem find_f (f : ℕ → ℕ) :
  (∀ a b c : ℕ, ((f a + f b + f c) - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) →
  (∀ n : ℕ, f n = n * n) :=
sorry

end find_f_l432_432358


namespace solve_inequality_l432_432702

theorem solve_inequality (x : ℝ) : 
  (-1 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧ 
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 1) ↔ (1 < x) := 
by 
  sorry

end solve_inequality_l432_432702


namespace evaluate_sqrt_diff_frac_l432_432086

theorem evaluate_sqrt_diff_frac : sqrt (1 / 4 - 1 / 25) = sqrt 21 / 10 :=
by
  sorry

end evaluate_sqrt_diff_frac_l432_432086


namespace median_of_dataset_variance_of_dataset_l432_432159

def dataset : List ℕ := [5, 4, 3, 5, 3, 2, 2, 3, 1, 2]

/-- Prove that the median of the dataset is 3 --/
theorem median_of_dataset :
  let sorted_dataset := [1, 2, 2, 2, 3, 3, 3, 4, 5, 5] in
  (sorted_dataset.nth 4).getOrElse 0 = 3 :=
by sorry

/-- Prove that the variance of the dataset is 8/5 --/
theorem variance_of_dataset :
  let mean := (1 + 2 * 3 + 3 * 3 + 4 + 5 * 2) / 10 in
  let variance := ((1 - mean)^2 + 3 * (2 - mean)^2 + 3 * (3 - mean)^2 + (4 - mean)^2 + 2 * (5 - mean)^2) / 10 in
  variance = 8 / 5 :=
by sorry

end median_of_dataset_variance_of_dataset_l432_432159


namespace knights_count_in_meeting_l432_432625

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l432_432625


namespace maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l432_432853

def pine_tree_height : ℚ := 37 / 4  -- 9 1/4 feet
def maple_tree_height : ℚ := 62 / 4  -- 15 1/2 feet (converted directly to common denominator)
def growth_rate : ℚ := 7 / 4  -- 1 3/4 feet per year

theorem maple_tree_taller_than_pine_tree : maple_tree_height - pine_tree_height = 25 / 4 := 
by sorry

theorem pine_tree_height_in_one_year : pine_tree_height + growth_rate = 44 / 4 := 
by sorry

end maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l432_432853


namespace simplify_fraction_product_l432_432328

theorem simplify_fraction_product :
  12 * ( (1/3) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
by
  sorry

end simplify_fraction_product_l432_432328


namespace largest_possible_perimeter_l432_432025

theorem largest_possible_perimeter (y : ℕ) (h1 : 7 + 9 > y) (h2 : 7 + y > 9) (h3 : 9 + y > 7) :
  y ≤ 15 → (7 + 9 + y) = 31 →
  y = 15 :=
by
  intro hy.
  have h : y = 15 := sorry
  exact h

example : largest_possible_perimeter 15 (by linarith) (by linarith) (by linarith) (by linarith) rfl :=
by
  unfold largest_possible_perimeter
  linarith

end largest_possible_perimeter_l432_432025


namespace maxValue_of_MF1_MF2_l432_432907

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l432_432907


namespace problem_1_problem_2_l432_432932

-- Define the intersection point M of two lines l1 and l2
def M : ℝ×ℝ := (-1, 2)

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 3*x + 4*y - 5 = 0
def l2 (x y : ℝ) : Prop := 2*x - 3*y + 8 = 0

-- Define the line that line l is perpendicular to
def l_perpendicular (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define the point of symmetry
def symmetry_point : ℝ×ℝ := (1, -1)

-- Define the target equations
def target_line_l (x y : ℝ) : Prop := x - 2*y + 5 = 0
def target_line_l_prime (x y : ℝ) : Prop := 3*x + 4*y + 7 = 0

theorem problem_1 :
  (∀ x y : ℝ, (l1 x y ∧ l2 x y) → ∃ (l : ℝ×ℝ→Prop), (∀ x y : ℝ, l x y ↔ target_line_l x y) ∧ l M.1 M.2)
:=
sorry

theorem problem_2 :
  (∀ x y : ℝ, l1 x y → (∃ (l' : ℝ×ℝ→Prop), (∀ x y : ℝ, l' x y ↔ target_line_l_prime x y) ∧ 
    ∃ (x' y' : ℝ), (x' = 2 - x ∧ y' = -2 - y) ∧ l1 x' y'))
:=
sorry

end problem_1_problem_2_l432_432932


namespace solve_polynomial_equation_l432_432280

theorem solve_polynomial_equation :
  ∀ x : ℝ, ((x^3 * 0.76^3) - 0.008) / (x^2 * 0.76^2 + x * 0.76 * 0.2 + 0.04) = 0 →
  x ≈ 0.262 :=
by
  intro x,
  intro h,
  sorry

end solve_polynomial_equation_l432_432280


namespace seokjin_paper_count_l432_432224

theorem seokjin_paper_count :
  ∀ (jimin_paper seokjin_paper : ℕ),
  jimin_paper = 41 →
  jimin_paper = seokjin_paper + 1 →
  seokjin_paper = 40 :=
by
  intros jimin_paper seokjin_paper h_jimin h_relation
  sorry

end seokjin_paper_count_l432_432224


namespace velocity_at_2_l432_432797

def distance (t : ℝ) : ℝ := 3 * t^2 + t

def velocity (t : ℝ) : ℝ := deriv distance t

theorem velocity_at_2 : velocity 2 = 13 := 
by 
  -- this skipped proof is just placeholder
  sorry

end velocity_at_2_l432_432797
