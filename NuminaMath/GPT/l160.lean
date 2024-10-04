import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Contin.CompareIntermediate
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Analysis.Trigonometry.Inverse
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Digits
import Mathlib.Probability.Basic
import Mathlib.Probability.Process.Conditional

namespace fewer_seats_on_right_than_left_l160_160365

theorem fewer_seats_on_right_than_left : 
  ∀ (left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats : ℕ),
    left_seats = 15 →
    back_seat_capacity = 9 →
    people_per_seat = 3 →
    bus_capacity = 90 →
    right_seats = (bus_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat →
    fewer_seats = left_seats - right_seats →
    fewer_seats = 3 :=
by
  intros left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats
  sorry

end fewer_seats_on_right_than_left_l160_160365


namespace distance_of_point_on_parabola_to_focus_l160_160317

theorem distance_of_point_on_parabola_to_focus :
  ∀ (x0 : ℝ), (64 = 8 * x0) -> (let P := (x0, 8 : ℝ × ℝ);
                                let F := (2, 0 : ℝ × ℝ);
                                dist P F = 10) :=
by
  intros x0 h1
  let P := (x0, 8 : ℝ × ℝ)
  let F := (2, 0 : ℝ × ℝ)
  sorry

end distance_of_point_on_parabola_to_focus_l160_160317


namespace profits_to_revenues_ratio_l160_160366

theorem profits_to_revenues_ratio (R P: ℝ) 
    (rev_2009: R_2009 = 0.8 * R) 
    (profit_2009_rev_2009: P_2009 = 0.2 * R_2009)
    (profit_2009: P_2009 = 1.6 * P):
    (P / R) * 100 = 10 :=
by
  sorry

end profits_to_revenues_ratio_l160_160366


namespace problem_statement_l160_160300

noncomputable def S (k : ℕ) : ℚ := sorry

theorem problem_statement (k : ℕ) (a_k : ℚ) :
  S (k - 1) < 10 → S k > 10 → a_k = 6 / 7 :=
sorry

end problem_statement_l160_160300


namespace coupon_probability_l160_160500

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l160_160500


namespace total_frog_eyes_l160_160810

-- Let n be the number of frogs of each species
variable (n : Nat)

-- Define number of eyes for each species
def eyes_A := 2
def eyes_B := 3
def eyes_C := 4
def eyes_D := 6

-- Define the total number of eyes
def total_eyes := (eyes_A + eyes_B + eyes_C + eyes_D) * n

theorem total_frog_eyes (n : Nat) : total_eyes n = 15 * n :=
by 
  -- Calculation based on species eyes and generalize over multiple species
  unfold total_eyes eyes_A eyes_B eyes_C eyes_D
  simp
  sorry

end total_frog_eyes_l160_160810


namespace coupon_probability_l160_160505

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l160_160505


namespace required_run_rate_l160_160006

def initial_run_rate : ℝ := 3.2
def overs_completed : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 50

theorem required_run_rate :
  (target_runs - initial_run_rate * overs_completed) / remaining_overs = 5 := 
by
  sorry

end required_run_rate_l160_160006


namespace solution_l160_160617

noncomputable def polynomial (x m : ℝ) := 3 * x^2 - 5 * x + m

theorem solution (m : ℝ) : (∃ a : ℝ, a = 2 ∧ polynomial a m = 0) -> m = -2 := by
  sorry

end solution_l160_160617


namespace find_x_l160_160962

theorem find_x : ∃ x : ℕ, 16^3 + 16^3 + 16^3 = 2^x ∧ x = 13 :=
by
  exists 13
  have h1 : 16^3 + 16^3 + 16^3 = 3 * 16^3 := by ring
  have h2 : 16 = 2 ^ 4 := by norm_num
  have h3 : 16 ^ 3 = (2 ^ 4) ^ 3 := by rw h2
  rw [h3, pow_mul, mul_comm 3 12, ←pow_add] at h1
  rw h1
  norm_num
  split
  exact rfl
  norm_num
  sorry

end find_x_l160_160962


namespace number_is_209_given_base_value_is_100_l160_160183

theorem number_is_209_given_base_value_is_100 (n : ℝ) (base_value : ℝ) (H : base_value = 100) (percentage : ℝ) (H1 : percentage = 2.09) : n = 209 :=
by
  sorry

end number_is_209_given_base_value_is_100_l160_160183


namespace max_value_of_a_l160_160279

theorem max_value_of_a : 
  (∃ a : ℝ, 
     (∀ x : ℝ, -1/3 * x > 2/3 - x ∧ 1/2 * x - 1 < 1/2 * (a - 2)) ∧ 
     (let sols := {x : ℤ | 1 < ↑x ∧ ↑x < a} in sols.card = 3) 
  ) → a = 5 :=
sorry

end max_value_of_a_l160_160279


namespace inequality_sqrt_sum_gt_sqrt2_abs_diff_l160_160851

theorem inequality_sqrt_sum_gt_sqrt2_abs_diff
  (a b c : ℝ) (h : a ≠ b) : 
  sqrt ((a - c) ^ 2 + b ^ 2) + sqrt (a ^ 2 + (b - c) ^ 2) > sqrt 2 * abs (a - b) := 
by 
  -- Here we would include the proof, but for now we just put a sorry to indicate that the proof is omitted
  sorry

end inequality_sqrt_sum_gt_sqrt2_abs_diff_l160_160851


namespace point_M_trajectory_quadrilateral_ABCD_range_l160_160294

theorem point_M_trajectory :
  (∀ (x y : ℝ), (0.5 * (1 - x)^2 + 0.5 * y^2 = 0.5 * (4 - x)^2 + 0.5 * y^2) ↔ (x^2 + y^2 = 4)) :=
by
  sorry

theorem quadrilateral_ABCD_range :
  (∀ (d1 d2 : ℝ), (d1^2 + d2^2 = 2) →
    (∃ t : ℝ, (0 ≤ t ∧ t ≤ 2) ∧ 
      (4*sqrt((4 - t)*(2 + t)) ∈ Set.Icc (4*sqrt(2)) 6))) :=
by
  sorry

end point_M_trajectory_quadrilateral_ABCD_range_l160_160294


namespace value_of_c_l160_160153

theorem value_of_c (x : ℝ) (c : ℝ) : 
  (∀ x ∈ Ioo (-3/2 : ℝ) (1 : ℝ), x * (4 * x + 2) < c) 
  ∧ (x = -3/2 ∨ x = 1 -> x * (4 * x + 2) = c) 
  → c = 6 := 
sorry

end value_of_c_l160_160153


namespace find_h_l160_160939

theorem find_h (h : ℝ) (r s : ℝ) (h_eq : ∀ x : ℝ, x^2 - 4 * h * x - 8 = 0)
  (sum_of_squares : r^2 + s^2 = 20) (roots_eq : x^2 - 4 * h * x - 8 = (x - r) * (x - s)) :
  h = 1 / 2 ∨ h = -1 / 2 := 
sorry

end find_h_l160_160939


namespace equation_of_ellipse_fixed_point_and_dot_product_l160_160301

-- Definitions and conditions
def ellipse (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) : Prop :=
  ∃ c : ℝ, c = sqrt(3) ∧ a^2 - b^2 = 3

def fixed_point (a b c m : ℝ) (condition_1 : a^2 - b^2 = 3) (condition_2 : c = sqrt(3)) : Prop :=
  ∃ k : ℝ,
    (m = - (9 * sqrt(3)) / 8 ∧
     (4 * m^2 + 8 * sqrt(3) * m + 11) / (m^2 - 4) = 4)

-- Part 1: Prove the equation of ellipse
theorem equation_of_ellipse : ellipse 2 1 2.0_pos 1.0_pos :=
by
  sorry

-- Part 2: Prove existence of fixed point and constant value for dot product
theorem fixed_point_and_dot_product :
  fixed_point 2 1 (sqrt(3)) (-9 * sqrt(3) / 8) (by linarith) (by norm_num [sqrt_eq_rpow] ; linarith) :=
by
  sorry

end equation_of_ellipse_fixed_point_and_dot_product_l160_160301


namespace kangaroo_mob_has_6_l160_160802

-- Define the problem conditions
def mob_of_kangaroos (W : ℝ) (k : ℕ) : Prop :=
  ∃ (two_lightest three_heaviest remaining : ℝ) (n_two n_three n_rem : ℕ),
    two_lightest = 0.25 * W ∧
    three_heaviest = 0.60 * W ∧
    remaining = 0.15 * W ∧
    n_two = 2 ∧
    n_three = 3 ∧
    n_rem = 1 ∧
    k = n_two + n_three + n_rem

-- The theorem to be proven
theorem kangaroo_mob_has_6 (W : ℝ) : ∃ k, mob_of_kangaroos W k ∧ k = 6 :=
by
  sorry

end kangaroo_mob_has_6_l160_160802


namespace paula_candies_l160_160434

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l160_160434


namespace number_of_intersections_l160_160120

-- Definitions of the given curves.
def curve1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

-- Statement of the theorem
theorem number_of_intersections : ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2 := sorry

end number_of_intersections_l160_160120


namespace max_value_l160_160044

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m ≤ 4, ∀ (z w : ℝ), z > 0 → w > 0 → (x + y = z + w) → (z^3 + w^3 ≥ x^3 + y^3 → 
  (z + w)^3 / (z^3 + w^3) ≤ m) :=
sorry

end max_value_l160_160044


namespace probability_all_small_l160_160215

-- Definitions based on conditions
def total_oranges := 8
def large_oranges := 5
def small_oranges := 3
def oranges_chosen := 3

-- Theorem statement
theorem probability_all_small : (choose small_oranges oranges_chosen) / (choose total_oranges oranges_chosen) = 1 / 56 :=
by sorry

end probability_all_small_l160_160215


namespace find_S_2017_l160_160652

open Nat

-- Definitions from the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Sequence a_n has the sum S_n for the first n terms
axiom sum_seq : ∀ n, S n = ∑ i in range 1 (n+1), a i

-- Sequence 2^a_n is geometric
axiom geometric_seq : ∃ q, ∀ n ≥ 1, 2^(a n) = q * 2^(a (n-1))

-- Given condition
axiom given_condition : a 4 + a 1009 + a 2014 = 3/2

-- Goal: Find S_2017
theorem find_S_2017 (a : ℕ → ℝ) (S : ℕ → ℝ) [seq_sum : ∀ n, S n = ∑ i in range 1 (n+1), a i] [geo_seq : ∃ q, ∀ n ≥ 1, 2^(a n) = q * 2^(a (n-1))] [specific_condition : a 4 + a 1009 + a 2014 = 3/2]:
    S 2017 = 2017 / 2 := by
    sorry

end find_S_2017_l160_160652


namespace even_sin_condition_l160_160103

theorem even_sin_condition (φ : ℝ) : 
  (φ = -Real.pi / 2 → ∀ x : ℝ, sin (x + φ) = sin (-(x + φ))) ∧ 
  (∀ x : ℝ, sin (x + φ) = sin (-(x + φ)) → ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2) :=
by
  sorry

end even_sin_condition_l160_160103


namespace irrational_count_l160_160222

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_count :
  let numbers := [22 / 7, 0, 3.1415926, 2.010010001, -Real.sqrt 3, Real.cbrt 343, -Real.pi / 3] in
  count is_irrational numbers = 3 :=
by
  sorry

end irrational_count_l160_160222


namespace same_color_points_unit_distance_l160_160249

def Color := {c : Bool // c = true ∨ c = false} -- True for red, False for blue.

structure Point where
  x : ℕ
  y : ℕ
  h : x ≤ 2 ∧ y ≤ 2

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt (↑((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)))

axiom grid_coloring : ∀ (p : Point), Color

theorem same_color_points_unit_distance : ∃ (p1 p2 : Point), grid_coloring p1 = grid_coloring p2 ∧ distance p1 p2 = 1 :=
by
  sorry

end same_color_points_unit_distance_l160_160249


namespace area_difference_l160_160784

theorem area_difference (r1 d2 : ℝ) (h1 : r1 = 30) (h2 : d2 = 15) : 
  π * r1^2 - π * (d2 / 2)^2 = 843.75 * π :=
by
  sorry

end area_difference_l160_160784


namespace original_price_of_car_l160_160261

theorem original_price_of_car (P : ℝ) 
  (h₁ : 0.561 * P + 200 = 7500) : 
  P = 13012.48 := 
sorry

end original_price_of_car_l160_160261


namespace locus_midpoints_l160_160579

theorem locus_midpoints
  (a x y : ℝ)
  (A : ℝ × ℝ := (x, y))
  (B : ℝ × ℝ := (y, -x))
  (C : ℝ × ℝ := (-x, -y))
  (D : ℝ × ℝ := (-y, x))
  (l : ℝ → Prop := λ y, y = a)
  (Q : ℝ × ℝ := ((x + y) / 2, (y - x) / 2))
  (P : ℝ × ℝ := (-y, a)) :
  ∃ t : ℝ, ∀ M : ℝ × ℝ, M = ((x - y) / 4, a / 2 + (y - x) / 4) ↔ M = (t, -t + a / 2) :=
begin
  sorry
end

end locus_midpoints_l160_160579


namespace investment_value_l160_160227

-- Define the compound interest calculation
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Given values
def P : ℝ := 8000
def r : ℝ := 0.05
def n : ℕ := 7

-- The theorem statement in Lean 4
theorem investment_value :
  round (compound_interest P r n) = 11257 :=
by
  sorry

end investment_value_l160_160227


namespace four_digit_multiples_of_7_l160_160710

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160710


namespace num_chords_from_8_points_l160_160254

theorem num_chords_from_8_points : 
  let n := 8 in 
  nat.choose n 2 = 28 :=
by
  sorry

end num_chords_from_8_points_l160_160254


namespace two_students_cover_all_questions_l160_160173

-- Define the main properties
variables (students : Finset ℕ) (questions : Finset ℕ)
variable (solves : ℕ → ℕ → Prop)

-- Assume the given conditions
axiom total_students : students.card = 8
axiom total_questions : questions.card = 8
axiom each_question_solved_by_min_5_students : ∀ q, q ∈ questions → 
(∃ student_set : Finset ℕ, student_set.card ≥ 5 ∧ ∀ s ∈ student_set, solves s q)

-- The theorem to be proven
theorem two_students_cover_all_questions :
  ∃ s1 s2 : ℕ, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ q ∈ questions, solves s1 q ∨ solves s2 q :=
sorry -- proof to be written

end two_students_cover_all_questions_l160_160173


namespace minimal_changes_needed_to_equalize_sums_l160_160614

def matrix : matrix (fin 3) (fin 3) ℤ :=
![![5, 10, 1], 
  ![7, 2, 8], 
  ![4, 6, 9]]

def initial_row_sums (m : matrix (fin 3) (fin 3) ℤ) : fin 3 → ℤ
| ⟨0, _⟩ := m 0 0 + m 0 1 + m 0 2
| ⟨1, _⟩ := m 1 0 + m 1 1 + m 1 2
| ⟨2, _⟩ := m 2 0 + m 2 1 + m 2 2

def initial_col_sums (m : matrix (fin 3) (fin 3) ℤ) : fin 3 → ℤ
| ⟨0, _⟩ := m 0 0 + m 1 0 + m 2 0
| ⟨1, _⟩ := m 0 1 + m 1 1 + m 2 1
| ⟨2, _⟩ := m 0 2 + m 1 2 + m 2 2

theorem minimal_changes_needed_to_equalize_sums : ∃ changes : ℕ,
  5 ≤ changes ∧
  (∃ m' : matrix (fin 3) (fin 3) ℤ, 
     (∀ i, initial_row_sums m' i = 18) ∧ 
     (∀ j, initial_col_sums m' j = 18)) :=
by
  sorry

end minimal_changes_needed_to_equalize_sums_l160_160614


namespace triangle_ADE_area_l160_160915

theorem triangle_ADE_area (A B C D E : Type)
  [equilateral_triangle : EquilateralTriangle A B C]
  [angle_BAC_60 : Angle A B C = 60]
  [area_ABC_27_sqrt_3 : Area A B C = 27 * Real.sqrt 3]
  [trisect_BAC : Trisects (Angle A B C) D E] :
  Area A D E = 3 * Real.sqrt 3 :=
sorry

end triangle_ADE_area_l160_160915


namespace solve_system_l160_160085

-- Define the variables and conditions
variables {R : Type} [linear_ordered_field R]
variables {n : ℕ} (a : R) (x : fin n → R)

-- Define the system of equations as hypotheses
def eq_system (a : R) (x : fin n → R) : Prop :=
  ∀ i : fin n, x i * |x i| = x ((i + 1) % n) * |x ((i + 1) % n)| + (x i - a) * |(x i - a)|

-- Define the proof statement
theorem solve_system (a_pos : a > 0) (H : eq_system a x) : ∀ i : fin n, x i = a :=
by 
  sorry

end solve_system_l160_160085


namespace probability_of_missing_coupons_l160_160492

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l160_160492


namespace symmetric_points_on_circumcircle_l160_160542

noncomputable def symm_point (H : Point) (A B : Point) : Point :=
  reflection_point(H, Line(A, B))

theorem symmetric_points_on_circumcircle (A B C H : Point) (H1 H2 H3 : Point)
  (H_is_orthocenter : is_orthocenter H A B C)
  (H1_symmetric : H1 = symm_point H B C)
  (H2_symmetric : H2 = symm_point H C A)
  (H3_symmetric : H3 = symm_point H A B) :
  is_on_circumcircle H1 A B C ∧ is_on_circumcircle H2 A B C ∧ is_on_circumcircle H3 A B C :=
by
  sorry

end symmetric_points_on_circumcircle_l160_160542


namespace no_three_progression_in_A_no_infinite_arithmetic_progression_in_B_l160_160891

noncomputable def A : Set ℕ := { n! + n | n : ℕ }

noncomputable def B : Set ℕ := { n | n ∈ (Set.univ : Set ℕ) } \ A

theorem no_three_progression_in_A : ∀ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x < y ∧ y < z → ¬(2 * y = x + z) := 
  sorry

theorem no_infinite_arithmetic_progression_in_B : ¬(∃ (a d : ℕ), ∀ n, a + n * d ∈ B) :=
  sorry

end no_three_progression_in_A_no_infinite_arithmetic_progression_in_B_l160_160891


namespace same_color_points_unit_distance_l160_160250

def Color := {c : Bool // c = true ∨ c = false} -- True for red, False for blue.

structure Point where
  x : ℕ
  y : ℕ
  h : x ≤ 2 ∧ y ≤ 2

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt (↑((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)))

axiom grid_coloring : ∀ (p : Point), Color

theorem same_color_points_unit_distance : ∃ (p1 p2 : Point), grid_coloring p1 = grid_coloring p2 ∧ distance p1 p2 = 1 :=
by
  sorry

end same_color_points_unit_distance_l160_160250


namespace curve_after_scaling_is_ellipse_l160_160683

noncomputable def polar_to_cartesian (ρ θ : ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def scaling_transformation (x y : ℝ) :=
  (x / 2, y / (sqrt 3 / 3))

def original_curve (x y : ℝ) : Prop :=
  (4 * x^2) / 12 + (3 * y^2) / 12 = 1

def scaled_curve (x' y' : ℝ) : Prop :=
  (4 * x'^2) / 3 + (3 * y'^2) / 4 = 1

theorem curve_after_scaling_is_ellipse :
  ∀ x' y' : ℝ, ∃ x y : ℝ, scaling_transformation x y = (x', y') ∧ original_curve x y ↔ scaled_curve x' y' :=
by
  intros
  sorry

end curve_after_scaling_is_ellipse_l160_160683


namespace distance_squared_between_circle_intersections_is_96_l160_160946

def circle1_center : ℝ × ℝ := (3, 2)
def circle1_radius : ℝ := 5
def circle2_center : ℝ × ℝ := (3, -4)
def circle2_radius : ℝ := 7

theorem distance_squared_between_circle_intersections_is_96 :
  ∃ (A B : ℝ × ℝ),
    (A = (3 + 2 * Real.sqrt 6, 1) ∨ A = (3 - 2 * Real.sqrt 6, 1)) ∧
    (B = (3 + 2 * Real.sqrt 6, 1) ∨ B = (3 - 2 * Real.sqrt 6, 1)) ∧
    (A ≠ B) ∧
    let dist := (A.fst - B.fst)^2 + (A.snd - B.snd)^2 in
    dist = 96 :=
by
  sorry

end distance_squared_between_circle_intersections_is_96_l160_160946


namespace range_of_a_l160_160330

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then x / 3 
  else if 1/2 < x ∧ x < 1 then 2 * x^3 / (x + 1)
  else 0  -- Default case added for completeness

def g (a x : ℝ) : ℝ := a * x - a / 2 + 3

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 ∈ Icc (0 : ℝ) 1, ∃ x2 ∈ Icc (0 : ℝ) (1/2), f x1 = g a x2) ↔ (6 ≤ a) :=
begin
  sorry
end

end range_of_a_l160_160330


namespace evaluate_six_applications_problem_solution_l160_160412

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

theorem evaluate_six_applications (θ : ℚ) : 
  r (r (r (r (r (r θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

theorem problem_solution : r (r (r (r (r (r 30))))) = 158 / 253 :=
by
  have h : r (r (r (r (r (r 30))))) = (8 + 5 * 30) / (13 + 8 * 30) := by
    exact evaluate_six_applications 30
  rw [h]
  norm_num

end evaluate_six_applications_problem_solution_l160_160412


namespace solve_equation_l160_160449

theorem solve_equation : ∃ x : ℝ, 4^x - 2^(x + 2) - 12 = 0 ∧ x = 1 + Real.log 3 / Real.log 2 :=
by
  use 1 + Real.log 3 / Real.log 2
  sorry

end solve_equation_l160_160449


namespace tangent_line_eqn_at_one_l160_160113

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_eqn_at_one :
  let k := (Real.exp 1)
  let p := (1, Real.exp 1)
  ∃ m b : ℝ, (m = k) ∧ (b = p.2 - m * p.1) ∧ (∀ x, f x = y → y = m * x + b) :=
sorry

end tangent_line_eqn_at_one_l160_160113


namespace mul_add_distrib_l160_160641

theorem mul_add_distrib :
  15 * 36 + 15 * 24 = 900 := by
  sorry

end mul_add_distrib_l160_160641


namespace min_fish_on_mon_wed_fri_l160_160262

theorem min_fish_on_mon_wed_fri:
  ∀ (a1 a2 a3 a4 a5 : ℕ),
  a1 ≥ a2 →
  a2 ≥ a3 →
  a3 ≥ a4 →
  a4 ≥ a5 →
  a1 + a2 + a3 + a4 + a5 = 100 →
  a1 + a3 + a5 ≥ 50 :=
begin
  intros a1 a2 a3 a4 a5 h1 h2 h3 h4 sum,
  sorry
end

end min_fish_on_mon_wed_fri_l160_160262


namespace all_terms_integers_l160_160894

def sequence (a : ℕ → ℚ) :=
  (a 1 = 1) ∧ (a 2 = 1) ∧ (a 3 = 1) ∧
  (∀ n ≥ 3, a (n + 1) = (1 + a (n - 1) * a n) / a (n - 2))

theorem all_terms_integers (a : ℕ → ℚ) (h : sequence a) : ∀ n, a n ∈ ℤ :=
by
  sorry

end all_terms_integers_l160_160894


namespace correct_op_l160_160644

-- Declare variables and conditions
variables {a b : ℝ} {m n : ℤ}
variable (ha : a > 0)
variable (hb : b ≠ 0)

-- Define and state the theorem
theorem correct_op (ha : a > 0) (hb : b ≠ 0) : (b / a)^m = a^(-m) * b^m :=
sorry  -- Proof omitted

end correct_op_l160_160644


namespace solve_for_x_l160_160350

theorem solve_for_x (x : ℝ) (h : (5 * x - 3)^3 = real.sqrt 64) : x = 1 :=
by
  sorry

end solve_for_x_l160_160350


namespace problem_1_problem_2_l160_160401

-- defining S_n as set of binary vectors of length n
def S (n : ℕ) := { A : Fin n -> ℕ // ∀ i, A i = 0 ∨ A i = 1 }

-- definition of distance function d
def d {n : ℕ} (U V : S n) : ℕ :=
  Finset.card { i : Fin n | U.val i ≠ V.val i }

-- Problem 1: Lean statement
theorem problem_1 : 
  let U : S 6 := ⟨λ _, 1, λ i, or.inr rfl⟩,
  m = finset.card { V : S 6 | d U V = 2 } :=
  sorry

-- Problem 2: Lean statement
theorem problem_2 (U : S n) : 
  let V_set := { V : S n }, 
  (finset.sum V_set (λ V, d U V)) = n * 2^(n-1) :=
  sorry

end problem_1_problem_2_l160_160401


namespace find_values_of_a_and_c_l160_160938

theorem find_values_of_a_and_c
  (a c : ℝ)
  (h1 : ∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ a * x^2 + 5 * x + c > 0) :
  a = -6 ∧ c = -1 :=
by
  sorry

end find_values_of_a_and_c_l160_160938


namespace v_20_eq_b_l160_160615

noncomputable def sequence (b : ℝ) (n : ℕ) : ℝ :=
  Nat.recOn n b (λ n v_n, -1 / (2 * v_n + 1))

theorem v_20_eq_b (b : ℝ) (h : b > 0) : sequence b 20 = b :=
  sorry

end v_20_eq_b_l160_160615


namespace necklace_wire_length_l160_160020

theorem necklace_wire_length
  (spools : ℕ)
  (feet_per_spool : ℕ)
  (total_necklaces : ℕ)
  (h1 : spools = 3)
  (h2 : feet_per_spool = 20)
  (h3 : total_necklaces = 15) :
  (spools * feet_per_spool) / total_necklaces = 4 := by
  sorry

end necklace_wire_length_l160_160020


namespace min_distance_and_distance_from_Glafira_l160_160995

theorem min_distance_and_distance_from_Glafira 
  (U g τ V : ℝ) (h : 2 * U ≥ g * τ) :
  let T := (τ / 2) + (U / g) in
  s T = 0 ∧ (V * T = V * (τ / 2 + U / g)) :=
by
  -- Define the positions y1(t) and y2(t)
  let y1 := λ t, U * t - (g * t^2) / 2
  let y2 := λ t, U * (t - τ) - (g * (t - τ)^2) / 2
  -- Define the distance s(t)
  let s := λ t, |y1 t - y2 t|
  -- Start the proof
  sorry

end min_distance_and_distance_from_Glafira_l160_160995


namespace fraction_irreducible_l160_160078

theorem fraction_irreducible (n : ℤ) : Nat.gcd (18 * n + 3).natAbs (12 * n + 1).natAbs = 1 := 
sorry

end fraction_irreducible_l160_160078


namespace crayons_left_l160_160882

theorem crayons_left (start_crayons lost_crayons left_crayons : ℕ) 
  (h1 : start_crayons = 479) 
  (h2 : lost_crayons = 345) 
  (h3 : left_crayons = start_crayons - lost_crayons) : 
  left_crayons = 134 :=
sorry

end crayons_left_l160_160882


namespace not_exists_cube_in_sequence_l160_160387

-- Lean statement of the proof problem
theorem not_exists_cube_in_sequence : ∀ n : ℕ, ¬ ∃ k : ℤ, 2 ^ (2 ^ n) + 1 = k ^ 3 := 
by 
    intro n
    intro ⟨k, h⟩
    sorry

end not_exists_cube_in_sequence_l160_160387


namespace inversely_proportional_x_y_l160_160453

theorem inversely_proportional_x_y {x y k : ℝ}
    (h_inv_proportional : x * y = k)
    (h_k : k = 75)
    (h_y : y = 45) :
    x = 5 / 3 :=
by
  sorry

end inversely_proportional_x_y_l160_160453


namespace partially_bolded_area_percent_l160_160248

-- Conditions for the areas of the bolded regions in each square
def AreaFirstSquareBold (s : ℝ) : ℝ := (1 / 2) * s^2
def AreaSecondSquareBold (s : ℝ) : ℝ := (1 / 2) * s^2
def AreaThirdSquareBold (s : ℝ) : ℝ := (1 / 8) * s^2
def AreaFourthSquareBold (s : ℝ) : ℝ := (1 / 4) * s^2

-- The total area of a square
def SquareArea (s : ℝ) : ℝ := s^2

-- Prove that the percentage of the total bolded area of the four squares is 25%
theorem partially_bolded_area_percent (s : ℝ) (h : s > 0) :
  (AreaFirstSquareBold s + AreaSecondSquareBold s + AreaThirdSquareBold s + AreaFourthSquareBold s) /
  (4 * SquareArea s) * 100 = 25 :=
begin
  sorry
end

end partially_bolded_area_percent_l160_160248


namespace correct_number_of_statements_is_two_l160_160975

-- Definitions of the geometric conditions
def condition1 (P Q: Plane) (l: Line) : Prop := P ⊥ l ∧ Q ⊥ l → P ∥ Q
def condition2 (l m: Line) (P: Plane) : Prop := l ⊥ P ∧ m ⊥ P → l ∥ m
def condition3 (P Q: Plane) (l: Line) : Prop := P ∥ l ∧ Q ∥ l → P ∥ Q
def condition4 (l m: Line) (P: Plane) : Prop := (l ∥ P) ∧ (m ∥ P) → l ∥ m

-- The main theorem we want to prove
theorem correct_number_of_statements_is_two :
    (∃ (P Q: Plane) (l: Line), condition1 P Q l) = false ∧
    (∃ (l m: Line) (P: Plane), condition2 l m P) = false ∧
    (∃ (P Q: Plane) (l: Line), condition3 P Q l) = true ∧
    (∃ (l m: Line) (P: Plane), condition4 l m P) = true → 
    (number_of_correct_statements = 2) :=
by sorry

end correct_number_of_statements_is_two_l160_160975


namespace least_possible_b_l160_160095

theorem least_possible_b (a b : ℕ) (ha : a.prime) (hb : b.prime) (sum_90 : a + b = 90) (a_greater_b : a > b) : b = 7 :=
by
  sorry

end least_possible_b_l160_160095


namespace num_four_digit_multiples_of_7_l160_160722

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160722


namespace count_four_digit_multiples_of_7_l160_160768

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160768


namespace subsetneq_M_N_l160_160046

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (x < 0) ∨ (x > 1 / 2)}

theorem subsetneq_M_N : M ⊂ N :=
by
  sorry

end subsetneq_M_N_l160_160046


namespace circles_are_separate_l160_160126

-- Definition of the first circle's center and radius
def circle1_center := (1, 0)
def circle1_radius := 1

-- Definition of the second circle's center and radius
def circle2_center := (-3, 2)
def circle2_radius := 2

-- Function to compute the Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Hypothesis stating the correct positional relationship
theorem circles_are_separate :
  euclidean_distance circle1_center circle2_center > (circle1_radius + circle2_radius) :=
sorry

end circles_are_separate_l160_160126


namespace value_20_percent_greater_l160_160164

theorem value_20_percent_greater (x : ℝ) : (x = 88 * 1.20) ↔ (x = 105.6) :=
by
  sorry

end value_20_percent_greater_l160_160164


namespace minimal_distance_l160_160992

noncomputable def y1 (U g t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def y2 (U g t τ : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
noncomputable def s (U g t τ : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem minimal_distance (U g τ : ℝ) (h : 2 * U ≥ g * τ) :
  ∃ t : ℝ, s U g t τ = 0 :=
begin
  use (2 * U / g + τ / 2),
  unfold s,
  sorry
end

end minimal_distance_l160_160992


namespace expectation_of_X_median_of_weights_contingency_table_correct_l160_160225

def probability_of_X (x : ℕ) : ℚ :=
  if x = 0 then 19 / 78
  else if x = 1 then 20 / 39
  else if x = 2 then 19 / 78
  else 0

theorem expectation_of_X : ℚ :=
  0 * (19 / 78) + 1 * (20 / 39) + 2 * (19 / 78)

theorem median_of_weights (control_weights experimental_weights : List ℚ) : ℚ :=
  let sorted_weights := (control_weights ++ experimental_weights).sort
  (sorted_weights.get 19 + sorted_weights.get 20) / 2

theorem contingency_table_correct 
  (control_weights experimental_weights : List ℚ) 
  (m : ℚ) 
  (control_less control_not_less exp_less exp_not_less : ℕ) 
  (K2 : ℚ) :
  (∀ w ∈ control_weights, if w < m then control_less + 1 else control_not_less + 1) →
  (∀ w ∈ experimental_weights, if w < m then exp_less + 1 else exp_not_less + 1) →
  K2 = 6.400 →
  6.400 > 3.841 →
  true :=
begin
  sorry
end

end expectation_of_X_median_of_weights_contingency_table_correct_l160_160225


namespace raspberry_pie_degrees_l160_160800

def total_students : ℕ := 48
def chocolate_preference : ℕ := 18
def apple_preference : ℕ := 10
def blueberry_preference : ℕ := 8
def remaining_students : ℕ := total_students - chocolate_preference - apple_preference - blueberry_preference
def raspberry_preference : ℕ := remaining_students / 2
def pie_chart_degrees : ℕ := (raspberry_preference * 360) / total_students

theorem raspberry_pie_degrees :
  pie_chart_degrees = 45 := by
  sorry

end raspberry_pie_degrees_l160_160800


namespace rotation_preserve_equation_l160_160470

noncomputable def rotation_line_equation (A B: ℝ) : Prop :=
  let l1 := λ x y: ℝ, x - y + A - 1 = 0 in
  let point := (1 : ℝ, real.sqrt 3) in
  let rot_angle := 15 * real.pi / 180 in
  ∃ l2 : ℝ → ℝ → Prop, (∀ x y : ℝ, l2 x y ↔ 3 * x - real.sqrt 3 * y = 0)

theorem rotation_preserve_equation : 
  rotation_line_equation (real.sqrt 3) (real.sqrt 3) → ∃ l2, ∀ x y, l2 x y ↔ 3 * x - real.sqrt 3 * y = 0 :=
    by
    sorry

end rotation_preserve_equation_l160_160470


namespace find_a_and_domain_find_min_value_compare_f2m_f3n_l160_160331

noncomputable theory
open Classical

def f (a : ℝ) (x : ℝ) : ℝ := log a (2 * x - 4) + log a (5 - x)

-- Given conditions
variables (a : ℝ) (2 < x) (x < 5) (P : ℝ × ℝ)
variables (m n t : ℝ) (5/2 < t) (t < 3) (hmt : 2^m = t) (hnt : 3^n = t)
variable hP : P = (3, -2)

-- Equivalent Lean problem statement
theorem find_a_and_domain : f (1/2) 3 = -2 ∧ (2 < x ∧ x < 5) :=
by sorry

theorem find_min_value : ∀ x ∈ set.Icc (3 : ℝ) (9/2), f (1/2) x = 1 - 2 * log 2 3 :=
by sorry

theorem compare_f2m_f3n : f (1/2) (2 * m) < f (1/2) (3 * n) :=
by sorry

end find_a_and_domain_find_min_value_compare_f2m_f3n_l160_160331


namespace solve_x_l160_160273

theorem solve_x (x : ℚ) : (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := 
by
  sorry

end solve_x_l160_160273


namespace product_of_divisors_eq_15625_l160_160934

theorem product_of_divisors_eq_15625 (n : ℕ) (h₁ : 0 < n) (h₂ : ∏ (d : ℕ) in (finset.filter (λ m, m ∣ n) (finset.range (n+1))), d = 15625) :
  n = 125 :=
sorry

end product_of_divisors_eq_15625_l160_160934


namespace coordinates_of_P_tangent_line_equation_l160_160298

-- Define point P and center of the circle
def point_P : ℝ × ℝ := (-2, 1)
def center_C : ℝ × ℝ := (-1, 0)

-- Define the circle equation (x + 1)^2 + y^2 = 2
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the tangent line at point P
def tangent_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Prove the coordinates of point P are (-2, 1) given the conditions
theorem coordinates_of_P (n : ℝ) (h1 : n > 0) (h2 : circle_equation (-2) n) :
  point_P = (-2, 1) :=
by
  -- Proof steps would go here
  sorry

-- Prove the equation of the tangent line to the circle C passing through point P is x - y + 3 = 0
theorem tangent_line_equation :
  tangent_line (-2) 1 :=
by
  -- Proof steps would go here
  sorry

end coordinates_of_P_tangent_line_equation_l160_160298


namespace weight_of_new_person_l160_160465

theorem weight_of_new_person (W : ℝ) (N : ℝ) :
  (∀ (x : ℝ), x = 45 → W + 20 = 8 * (x + 2.5)) → 
  N = 65 :=
by
  -- Given conditions
  assume h : (∀ (x : ℝ), x = 45 → W + 20 = 8 * (x + 2.5)),
  -- Proof
  sorry

end weight_of_new_person_l160_160465


namespace math_problem_l160_160417

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (a + 2^x)

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -(f x)
def isDecreasing (f : ℝ → ℝ) : Prop := ∀ x1 x2, x1 < x2 → f x1 > f x2

theorem math_problem (a b : ℝ):
  (isOdd (f a b)) → 
  (a = 1 ∧ b = 1) ∧
  (isDecreasing (f 1 1)) ∧
  (∀ k : ℝ, (∀ t : ℝ, t ∈ Icc (-1) 3 → f 1 1 (t^2 - 2*t) > f 1 1 (-2*t^2 + k)) → k > 21) :=
by
  sorry

end math_problem_l160_160417


namespace least_number_of_equal_cubes_l160_160191

/-!
  Given a cuboidal block with dimensions 6 cm, 9 cm, and 12 cm,
  prove that the least possible number of equal cubes that can
  exactly fit into the block is 24.
-/
theorem least_number_of_equal_cubes (a b c : ℕ) 
  (ha : a = 6) (hb : b = 9) (hc : c = 12) :
  let gcdabc := Nat.gcd (Nat.gcd a b) c,
      volume_block := a * b * c,
      volume_cube := gcdabc * gcdabc * gcdabc
  in volume_block / volume_cube = 24 :=
by
  let a := 6
  let b := 9
  let c := 12
  let gcdab := Nat.gcd a b
  let gcdabc := Nat.gcd gcdab c
  let volume_block := a * b * c
  let volume_cube := gcdabc * gcdabc * gcdabc
  sorry

end least_number_of_equal_cubes_l160_160191


namespace andrey_gifts_l160_160064

theorem andrey_gifts :
  ∃ (n : ℕ), ∀ (a : ℕ), n(n-2) = a(n-1) + 16 ∧ n = 18 :=
by {
  sorry
}

end andrey_gifts_l160_160064


namespace wizard_concoction_valid_combinations_l160_160585

structure WizardConcoction :=
(herbs : Nat)
(crystals : Nat)
(single_incompatible : Nat)
(double_incompatible : Nat)

def valid_combinations (concoction : WizardConcoction) : Nat :=
  concoction.herbs * concoction.crystals - (concoction.single_incompatible + concoction.double_incompatible)

theorem wizard_concoction_valid_combinations (c : WizardConcoction)
  (h_herbs : c.herbs = 4)
  (h_crystals : c.crystals = 6)
  (h_single_incompatible : c.single_incompatible = 1)
  (h_double_incompatible : c.double_incompatible = 2) :
  valid_combinations c = 21 :=
by
  sorry

end wizard_concoction_valid_combinations_l160_160585


namespace unknown_number_l160_160906

theorem unknown_number 
  (avg1 : ℚ := (14 + 32 + 53) / 3)
  (avg2 : ℚ → ℚ := λ x, (21 + 47 + x) / 3)
  (h : avg1 = avg2 (22) + 3) :
  22 = 22 :=
by
  -- This is where the proof steps would go
  sorry

end unknown_number_l160_160906


namespace Jacob_has_48_graham_crackers_l160_160018

def marshmallows_initial := 6
def marshmallows_needed := 18
def marshmallows_total := marshmallows_initial + marshmallows_needed
def graham_crackers_per_smore := 2

def smores_total := marshmallows_total
def graham_crackers_total := smores_total * graham_crackers_per_smore

theorem Jacob_has_48_graham_crackers (h1 : marshmallows_initial = 6)
                                     (h2 : marshmallows_needed = 18)
                                     (h3 : graham_crackers_per_smore = 2)
                                     (h4 : marshmallows_total = marshmallows_initial + marshmallows_needed)
                                     (h5 : smores_total = marshmallows_total)
                                     (h6 : graham_crackers_total = smores_total * graham_crackers_per_smore) :
                                     graham_crackers_total = 48 :=
by
  sorry

end Jacob_has_48_graham_crackers_l160_160018


namespace trig_identity_l160_160178

theorem trig_identity :
  cos (54 * Real.pi / 180) * cos (24 * Real.pi / 180) + 
  2 * sin (12 * Real.pi / 180) * cos (12 * Real.pi / 180) * 
  sin (126 * Real.pi / 180) = 
  (sqrt 3) / 2 := 
by 
  sorry

end trig_identity_l160_160178


namespace min_value_a1_a7_correct_l160_160369

noncomputable def min_value_a1_a7 (a : ℕ → ℝ) : ℝ :=
  if ∃ r (a : ℝ), (∀ n, a n = a * r ^ n) ∧ a 0 * a 2 = 12 then 4 * real.sqrt 3 else 0

theorem min_value_a1_a7_correct (a : ℕ → ℝ) 
 (h1 : ∀ n, a n > 0) 
 (h2 : a 2 * a 4 = 12) : 
  min_value_a1_a7 a = 4 * real.sqrt 3 :=
begin
  sorry
end

end min_value_a1_a7_correct_l160_160369


namespace bisect_segment_l160_160812

variables {A B C H K : Type} [inhabited A] [inhabited B] [inhabited C] 
  [inhabited H] [inhabited K]

variables {P : Type} [ordered_ring P]

-- Definitions and conditions
def right_triangle (ABC : Type) (angle_ACB : OrderedSemiring P) : Prop := 
  angle_ACB = 90 * degree

def altitude (CH : Type) (C : Type) (AB : Type) : Prop :=
  CH ⊥ AB

def is_angle_equal (CBK : Type) (CAB : Type) : Prop :=
  angle CBK = angle CAB

-- Statement to be proven
theorem bisect_segment (ABC : Type) (CH : Type) (BK : Type) (H : Type) (B : Type) (K : Type) :
  right_triangle ABC (angle ACB) ∧ altitude CH C AB ∧ is_angle_equal CBK CAB →
  CH bisects BK :=
sorry

end bisect_segment_l160_160812


namespace regular_dodecagon_diagonal_relation_l160_160016

noncomputable def diagonal_length (n : ℕ) (r : ℝ) : ℝ :=
  2 * r * (Real.sin (π * n / 12.0))

theorem regular_dodecagon_diagonal_relation (R : ℝ) :
  let d1 := diagonal_length 6 R,
      d2 := diagonal_length 2 R,
      d3 := diagonal_length 10 R
  in d1 = d2 + d3 :=
by
  sorry

end regular_dodecagon_diagonal_relation_l160_160016


namespace selected_bottles_correct_l160_160538

def bottles : List ℕ := List.range 80 
def random_table : List (List ℕ) := 
  [[16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43, 84, 26, 34, 91, 64],
   [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
   [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
   [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54],
   [57, 60, 86, 32, 44, 09, 47, 27, 96, 54, 49, 17, 46, 09, 62, 90, 52, 84, 77, 27, 08, 02, 73, 43, 28]]

def starting_point := random_table[5][4]
def selected_numbers : List ℕ := [77, 39, 49, 54, 43, 17]

theorem selected_bottles_correct : 
  ∀ (b_list : List ℕ) (r_table : List (List ℕ)) (start : ℕ) (selected : List ℕ),
  b_list = bottles →
  r_table = random_table →
  start = starting_point →
  selected = selected_numbers → 
  selected.length = 6 ∧
  ∀ n ∈ selected, n ∈ b_list ∧ 
  r_table[5][4] = 7 ∧
  r_table[5][5] ∈ selected ∧
  r_table[5][6] = 94 → false ∧ -- 94 discarded
  r_table[5][7] ∈ selected ∧ 
  r_table[5][8] ∈ selected ∧ 
  r_table[5][9] ∈ selected ∧ 
  r_table[5][10] = 82 → false ∧
  r_table[5][11] ∈ selected :=
begin
  intros b_list r_table start selected b_list_def r_table_def start_def selected_def,
  rw [b_list_def, r_table_def, start_def, selected_def],
  split,
  sorry,
  sorry,
end

end selected_bottles_correct_l160_160538


namespace num_chords_from_8_points_l160_160255

theorem num_chords_from_8_points : 
  let n := 8 in 
  nat.choose n 2 = 28 :=
by
  sorry

end num_chords_from_8_points_l160_160255


namespace sandwich_combinations_l160_160392

theorem sandwich_combinations (m n k : ℕ) (hm : m = 12) (hn : n = 11) (hk : k = 5) : 
  ∑ (C m 2) * ∑ (C n 2) * k = 18150 :=
by 
  -- Definitions for number of ways to choose 
  let ways_meat := Nat.choose m 2
  let ways_cheese := Nat.choose n 2
  let ways_bread := k
  -- Computation of total combinations
  calc 
    ways_meat * ways_cheese * ways_bread 
      = 66 * 55 * 5 : by { rw [Nat.choose, hm, hn, hk]; 
                          norm_num }
      ... = 18150    : by norm_num

end sandwich_combinations_l160_160392


namespace new_cost_percentage_l160_160544

def cost (t b : ℝ) := t * b^5

theorem new_cost_percentage (t b : ℝ) : 
  let C := cost t b
  let W := cost (3 * t) (2 * b)
  W = 96 * C :=
by
  sorry

end new_cost_percentage_l160_160544


namespace count_four_digit_multiples_of_7_l160_160736

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160736


namespace count_T_diff_S_l160_160778

-- Define a function to check if a digit is in a given number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ i, i < 3 ∧ (n / 10^i) % 10 = d

-- Define a function to check if a three-digit number is valid
def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the set T' of three digit numbers that do not contain a 6
def T_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6

-- Define the set S' of three digit numbers that neither contain a 2 nor a 6
def S_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6 ∧ ¬ contains_digit n 2

-- Define the set of numbers we are interested in, has 2 but not 6
def T_diff_S : {n // is_valid_three_digit n} → Prop := 
  λ n, contains_digit n 2 ∧ ¬ contains_digit n 6

-- Statement to prove
theorem count_T_diff_S : ∃ n, n = 200 ∧ (∀ (x : {n // is_valid_three_digit n}), T_diff_S x) :=
sorry

end count_T_diff_S_l160_160778


namespace four_digit_multiples_of_7_l160_160708

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160708


namespace find_y_log_eq_l160_160626

-- Define the problem in Lean and include the logarithm equivalence and the result we need to prove.
theorem find_y_log_eq : 
  (∃ y : ℝ, y > 0 ∧ y ≠ 1 ∧ log_y 16 = log 64 / log 4) ↔ y = 4096 :=
begin
  -- Given: log_y 16 = log 64 / log 4
  -- Goal: y = 4096

  sorry
end

end find_y_log_eq_l160_160626


namespace monotonically_decreasing_on_interval_l160_160673

open Real

noncomputable def f (x : ℝ) := sin x * cos x

theorem monotonically_decreasing_on_interval :
  ∀ x ∈ Icc (π / 4) (3 * π / 4), 0 ≤ x ∧ x ≤ π → ∃ I : Set ℝ, I = Icc (π / 4) (3 * π / 4) ∧ ∀ x y ∈ I, x < y → f'(x) > f'(y) :=
by
  sorry

end monotonically_decreasing_on_interval_l160_160673


namespace Carol_Final_Position_2304_l160_160610

-- Define the initial setup and movement pattern.
structure HexagonalSpiralPattern :=
(origin : (ℝ × ℝ) := (0, 0))
(directions : List (ℝ × ℝ)) -- List of movement vectors

-- Instantiate the pattern.
noncomputable def Carol'sHexagonalSpiral : HexagonalSpiralPattern := {
  origin := (0, 0),
  directions := [(0, 1), (sqrt 3 / 2, 1 / 2), (sqrt 3 / 2, -1 / 2), (0, -1), (-sqrt 3 / 2, -1 / 2), (-sqrt 3 / 2, 1 / 2)]
}

-- Function to simulate Carol's movement.
noncomputable def CarolPositionAfterSteps (steps : ℕ) (pattern : HexagonalSpiralPattern) : ℝ × ℝ :=
sorry  -- This will contain the logic to calculate Carol's final position based on steps and pattern.

-- The theorem to prove Carol's final position after 2304 moves.
theorem Carol_Final_Position_2304 : 
  CarolPositionAfterSteps 2304 Carol'sHexagonalSpiral = (5 * (sqrt 3) / 2, 23.5) :=
sorry

end Carol_Final_Position_2304_l160_160610


namespace initial_money_l160_160073

theorem initial_money {M : ℝ} (h : (M - 10) - (M - 10) / 4 = 15) : M = 30 :=
sorry

end initial_money_l160_160073


namespace max_value_of_f_l160_160329

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 + 2 * x) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h1 : f 1 a b = 0)
  (h2 : f (-1 / 2) a b = 0)
  (h3 : f (5 / 2) a b = 0) :
  a = -(7/2) ∧ b = (5/2) →
  ∃ x ∈ Icc (-1 : ℝ) 1, 
    f x a b = (3 * sqrt 3 / 2) :=
by
  sorry

end max_value_of_f_l160_160329


namespace hannah_quarters_l160_160340

theorem hannah_quarters :
  ∃ n : ℕ, 40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3 ∧ 
  (n = 171 ∨ n = 339) :=
by
  sorry

end hannah_quarters_l160_160340


namespace three_digit_numbers_containing_2_and_exclude_6_l160_160781

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l160_160781


namespace volume_of_five_tetrahedrons_l160_160516

def volume_tetrahedrons (side_length height_ratio : ℝ) : ℝ :=
  (side_length^2 * height_ratio) / 3

theorem volume_of_five_tetrahedrons (a m : ℝ) : 
  volume_tetrahedrons a m = (a^2 * m) / 3 :=
by
  sorry

end volume_of_five_tetrahedrons_l160_160516


namespace calculate_expression_l160_160234

variable {a : ℝ}

theorem calculate_expression (h₁ : a ≠ 0) (h₂ : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := 
sorry

end calculate_expression_l160_160234


namespace firefly_max_friendships_bound_l160_160459

open Real

noncomputable theory

def max_friendships (n : ℕ) : ℕ :=
  ⌊(n * n) / 3⌋₊

theorem firefly_max_friendships_bound (n : ℕ) :
  let fireflies_in_ℝ3 := 10^7
  let friendship_is_mutual := ∀ (a b : ℕ), (a ≠ b → (a ↔ b) = (b ↔ a))
  let move_preserves_distance := ∀ (a b c : ℕ), (friendship_is_mutual a b → (distance a b = distance a c))
  let initial_distance := ∀ (a b : ℕ), a ≠ b → distance a b ≤ 1
  let final_distance := ∀ (a : ℕ), distance a (initial_position a) ≥ 10^7

  n = fireflies_in_ℝ3 
  → friendship_is_mutual  
  → move_preserves_distance 
  → initial_distance 
  → final_distance 
  → ∀ (friendships : ℕ), friendships ≤ max_friendships n :=
begin
  intros,
  sorry
end

end firefly_max_friendships_bound_l160_160459


namespace hexagon_area_m_plus_n_sum_l160_160917

theorem hexagon_area_m_plus_n_sum :
  ∃ m n : ℕ, (RegularHexagonArea 3 = Real.sqrt m + Real.sqrt n) ∧ (m + n = 756) := by
  sorry

end hexagon_area_m_plus_n_sum_l160_160917


namespace n_lt_p_mul_p_add_1_div_2_l160_160040

theorem n_lt_p_mul_p_add_1_div_2 (p q n : ℕ) (hpp : p < q) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) :
  (∀ k, ∑ k in (Finset.range k), ⌊n / p^k⌋ = ∑ k in Finset.range k, ⌊n / q^k⌋) → n < p * (p + 1) / 2 :=
by
  sorry

end n_lt_p_mul_p_add_1_div_2_l160_160040


namespace option_C_correct_l160_160410

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for parallel and perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def line_parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement based on problem c) translation
theorem option_C_correct (H1 : line_parallel m n) (H2 : perpendicular m α) : perpendicular n α :=
sorry

end option_C_correct_l160_160410


namespace domain_of_f_l160_160109

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x + 1 > 0 ∧ x + 1 ≠ 1} = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by 
  sorry

end domain_of_f_l160_160109


namespace even_sum_probability_l160_160949

-- Conditions
def first_wheel_total_sections := 5
def first_wheel_even_sections := 2
def second_wheel_total_sections := 4
def second_wheel_even_sections := 2

-- Definitions derived from conditions
def first_wheel_even_prob := first_wheel_even_sections / first_wheel_total_sections
def first_wheel_odd_prob := 1 - first_wheel_even_prob
def second_wheel_even_prob := second_wheel_even_sections / second_wheel_total_sections
def second_wheel_odd_prob := 1 - second_wheel_even_prob

-- Compute the probabilities of getting an even sum
def even_sum_prob := (first_wheel_even_prob * second_wheel_even_prob) +
                     (first_wheel_odd_prob * second_wheel_odd_prob)

-- Assertion that the probability of an even sum is 1/2
theorem even_sum_probability :
  even_sum_prob = 1 / 2 :=
sorry

end even_sum_probability_l160_160949


namespace wallys_lock_number_l160_160515

-- Definitions for the conditions
variables {a b c d e : ℕ}
def N : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def M : ℕ := 10000 * a + 1000 * b + 100 * d + 10 * e + c
def P : ℕ := 10000 * a + 1000 * b + 100 * e + 10 * c + d

-- Condition that the digits a, b, c, d, e are distinct
axiom distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Condition that N is divisible by 111
axiom N_divisibility : N % 111 = 0

-- Condition that M is larger than N and divisible by 111
axiom M_conditions : M > N ∧ M % 111 = 0

-- Condition that P is larger than M and divisible by 111
axiom P_conditions : P > M ∧ P % 111 = 0

-- Proof that N is Wally's lock number 74259
theorem wallys_lock_number : N = 74259 :=
by { sorry }

end wallys_lock_number_l160_160515


namespace four_digit_multiples_of_7_l160_160746

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160746


namespace cody_money_l160_160174

theorem cody_money (a b c d : ℕ) (h₁ : a = 45) (h₂ : b = 9) (h₃ : c = 19) (h₄ : d = a + b - c) : d = 35 :=
by
  rw [h₁, h₂, h₃] at h₄
  simp at h₄
  exact h₄

end cody_money_l160_160174


namespace P_minus_Q_correct_l160_160846

open Set

noncomputable def P : Set ℝ := { x | Real.log2 x < 1 }

noncomputable def Q : Set ℝ := { x | 3 < 3^x ∧ 3^x < 9 }

noncomputable def P_minus_Q : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

theorem P_minus_Q_correct :
  (P \ Q) = P_minus_Q :=
by
  sorry

end P_minus_Q_correct_l160_160846


namespace symmetric_sequence_c2_is_19_l160_160192

-- Define the symmetric property for a sequence
def symmetric (n : Nat) (seq : Fin n → Int) : Prop :=
  ∀ i, i < n → seq ⟨i, Nat.le_of_lt (Nat.succ_lt_succ i).trans (Nat.lt.base n)⟩ = seq ⟨n - i - 1, Nat.le_pred_of_lt i⟩

-- Define the arithmetic sequence property
def arithmetic_seq (start diff : Int) (seq : Fin 11 → Int) : Prop :=
  ∀ i, i < 11 → seq ⟨i, Nat.le_of_lt (Nat.lt_succ_of_lt i)⟩ = start + diff * (i : Int)

-- Given the sequence c and its properties, prove c_2 = 19
theorem symmetric_sequence_c2_is_19 ({c : Fin 21 → Int}) (Hsymm : symmetric 21 c)
  (Harith : arithmetic_seq 1 2 (fun i => c ⟨i + 10, by linarith⟩)) : 
  c ⟨1, by decide⟩ = 19 :=
by
  sorry

end symmetric_sequence_c2_is_19_l160_160192


namespace banana_price_l160_160426

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price_l160_160426


namespace possible_values_a1_l160_160177

-- First, define the arithmetic sequence and properties
def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + n * d

-- Define S, the sum of the first 9 terms
def S (a1 d : ℤ) : ℤ := 9 * (a1 + 4 * d)

-- Define the two conditions
def condition1 (a1 d : ℤ) : Prop :=
  (arith_seq a1 d 4) * (arith_seq a1 d 17) > S a1 d - 4

def condition2 (a1 d : ℤ) : Prop :=
  (arith_seq a1 d 12) * (arith_seq a1 d 9) < S a1 d + 60

-- Define the main theorem to prove
theorem possible_values_a1 :
  ∀ (a1 d : ℤ), (d = 1) → (condition1 a1 d) → (condition2 a1 d) → 
    a1 ∈ {-10, -9, -8, -7, -5, -4, -3, -2} :=
by
  intros a1 d h_d h_cond1 h_cond2
  sorry

end possible_values_a1_l160_160177


namespace solve_sqrt_equation_l160_160897

theorem solve_sqrt_equation (x : ℝ) :
  (sqrt(x + 9) - 2 * sqrt(x - 2) + 3 = 0) →
  (x = 8 + 4 * sqrt(2) ∨ x = 8 - 4 * sqrt(2)) :=
by
  sorry

end solve_sqrt_equation_l160_160897


namespace m_eq_n_is_necessary_but_not_sufficient_l160_160102

noncomputable def circle_condition (m n : ℝ) : Prop :=
  m = n ∧ m > 0

theorem m_eq_n_is_necessary_but_not_sufficient 
  (m n : ℝ) :
  (circle_condition m n → mx^2 + ny^2 = 3 → False) ∧
  (mx^2 + ny^2 = 3 → circle_condition m n) :=
by 
  sorry

end m_eq_n_is_necessary_but_not_sufficient_l160_160102


namespace food_remaining_l160_160421

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end food_remaining_l160_160421


namespace term_position_l160_160337

def sequence (n : ℕ) : ℝ := real.sqrt (2 * n - 1)

theorem term_position : ∃ n : ℕ, sequence n = 3 * real.sqrt 5 ∧ n = 23 :=
by {
  sorry
}

end term_position_l160_160337


namespace eight_points_chords_l160_160257

theorem eight_points_chords : (∃ n, n = 8) → (∃ m, m = 28) :=
by
  intro h
  have h1: (∃ n, n = (Nat.choose 8 2)) := by
    use Nat.choose 8 2
    exact Nat.choose_eq 8 2
  cases h1 with m hm
  use m
  exact hm

end eight_points_chords_l160_160257


namespace steve_keeps_1800000_l160_160088

theorem steve_keeps_1800000 (S : ℕ) (P : ℝ) (A : ℝ) (M : ℝ) 
  (hS : S = 1000000) 
  (hP : P = 2) 
  (hA : A = 0.10) 
  (hM : M = S * P - (S * P * A)) : 
  M = 1,800,000 := 
by
  rw [hS, hP, hA, hM]
  norm_num

end steve_keeps_1800000_l160_160088


namespace find_f_19_l160_160360

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the given function

-- Define the conditions
axiom even_function : ∀ x : ℝ, f x = f (-x) 
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x

-- The statement we need to prove
theorem find_f_19 : f 19 = 0 := 
by
  sorry -- placeholder for the proof

end find_f_19_l160_160360


namespace coloring_theorem_l160_160252

namespace ColoringProof

def Vertex := (ℕ × ℕ)

def color := Vertex → Prop -- Assume color is a proposition representing red/blue for simplicity

def unit_distance (v1 v2 : Vertex) : Prop :=
  abs (v1.1 - v2.1) + abs (v1.2 - v2.2) = 1

noncomputable def vertices : list Vertex :=
  [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

theorem coloring_theorem (coloring : Vertex → Prop) :
  ∃ (v1 v2 : Vertex), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 ∧ unit_distance v1 v2 ∧ (coloring v1 = coloring v2) :=
sorry

end ColoringProof

end coloring_theorem_l160_160252


namespace tan_of_sin_cos_l160_160354

variable {θ : ℝ}

theorem tan_of_sin_cos (h₁ : sin θ = 3 / 5) (h₂ : cos θ = -4 / 5) : tan θ = -3 / 4 :=
sorry

end tan_of_sin_cos_l160_160354


namespace count_four_digit_multiples_of_7_l160_160742

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160742


namespace total_trip_time_l160_160139

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l160_160139


namespace four_digit_multiples_of_7_l160_160705

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160705


namespace volume_of_cylinder_with_tetrahedron_inscribed_l160_160205

theorem volume_of_cylinder_with_tetrahedron_inscribed {r h : ℝ} 
  (edge_length : ℝ)
  (tetrahedron_inscribed : ∀ {A B C D : ℝ}, A = B ∧ C = D) :
  edge_length = 2 → r = 1 ∧ h = √2 →
  π * r^2 * h = π * √2 :=
by
  intro h_edge_len h_rh
  sorry

end volume_of_cylinder_with_tetrahedron_inscribed_l160_160205


namespace lisa_needs_additional_marbles_l160_160872

theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 50) :
  ∃ additional_marbles : ℕ, additional_marbles = 78 - marbles ∧ additional_marbles = 28 :=
by
  -- The sum of the first 12 natural numbers is calculated as:
  have h_sum : (∑ i in finset.range (friends + 1), i) = 78 := by sorry
  -- The additional marbles needed:
  use 78 - marbles
  -- It should equal to 28:
  split
  . exact rfl
  . sorry

end lisa_needs_additional_marbles_l160_160872


namespace real_solutions_l160_160627

-- Let [x] be the greatest integer function.
def greatest_integer (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation we need to solve.
def f (x : ℝ) : ℝ := 4 * x^2 - 40 * (greatest_integer x) + 51

-- Provide the statement to prove the set of solutions.
theorem real_solutions :
  {x : ℝ | f x = 0} = {x | x = real.sqrt 29 / 2 ∨ x = real.sqrt 189 / 2 ∨ x = real.sqrt 229 / 2 ∨ x = real.sqrt 269 / 2} :=
sorry

end real_solutions_l160_160627


namespace coupon_probability_l160_160497

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l160_160497


namespace circular_ring_area_theorem_l160_160616

-- Given conditions 
variables {C1 C2 : ℝ} (h : C1 > C2)

-- Definition of area of the circular ring
def circular_ring_area (C1 C2 : ℝ) : ℝ := (C1^2 - C2^2) / (4 * real.pi)

-- Statement of the theorem
theorem circular_ring_area_theorem (h : C1 > C2) : 
  circular_ring_area C1 C2 = (C1^2 - C2^2) / (4 * real.pi) :=
sorry

end circular_ring_area_theorem_l160_160616


namespace find_number_l160_160054

theorem find_number (x : ℝ) : x = 7 ∧ x^2 + 95 = (x - 19)^2 :=
by
  sorry

end find_number_l160_160054


namespace four_digit_multiples_of_7_l160_160711

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160711


namespace garden_ratio_perimeter_l160_160193

theorem garden_ratio_perimeter (L W : ℕ) (hL : L = 23) (hW : W = 15) :
  (W * 76) = 15 * (2 * (L + W)) :=
by
  -- L = 23, W = 15
  rw [hL, hW]
  -- P = 2 * (L + W) = 76
  have hP : 2 * (23 + 15) = 76 := by norm_num
  rw [hP]
  -- show that 15:76 is the ratio by verifying the equation holds
  norm_num
  exact rfl

end garden_ratio_perimeter_l160_160193


namespace red_hat_count_is_3_l160_160134

theorem red_hat_count_is_3 (n_red_hat : ℕ) (n : ℕ) (h1 : n = 8) :
  (∀ i, i < 8 → ((∃ k, k < 8 ∧ k ≠ i ∧ n_red_hat ≥ 3 → (∀ j, j < 8 → (if j ≠ i then k else n_red_hat - 1)))) →
      ((n_red_hat ≥ 3 → (∀ c, c < 8 → ∃ d, d < 8 ∧ d ≠ c → c → (n_red_hat = 3))))) :=
by
  sorry

end red_hat_count_is_3_l160_160134


namespace angle_XYZ_in_isosceles_triangle_l160_160385

theorem angle_XYZ_in_isosceles_triangle (P Q R X Y Z : Point) : 
  Triangle P Q R → 
  (PQ = PR) → 
  (angle P = 100) → 
  (lies_on X QR) → 
  (lies_on Y RP) → 
  (lies_on Z PQ) →
  (RY = RX) → 
  (QZ = QX) →
  angle XYZ = 40 :=
by {
  sorry
}

end angle_XYZ_in_isosceles_triangle_l160_160385


namespace count_four_digit_multiples_of_7_l160_160764

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160764


namespace correct_analogical_conclusions_l160_160668

theorem correct_analogical_conclusions :
    (∀ (a b : ℝ), a - b = 0 → a = b) ∧ (∀ (a b : ℂ), a - b = 0 → a = b) ∧
    (∀ (a b : ℝ), a * b = 0 → a = 0 ∨ b = 0) ∧ (∀ (a b : ℂ), a * b = 0 → a = 0 ∨ b = 0) ∧
    ¬(∀ (a b : ℂ), a - b > 0 → a > b) ∧
    ¬(∀ (a b : ℂ), a ^ 2 + b ^ 2 ≥ 0) :=
by
  sorry

end correct_analogical_conclusions_l160_160668


namespace max_necklaces_with_beads_l160_160182

noncomputable def necklace_problem : Prop :=
  ∃ (necklaces : ℕ),
    let green_beads := 200
    let white_beads := 100
    let orange_beads := 50
    let beads_per_pattern_green := 3
    let beads_per_pattern_white := 1
    let beads_per_pattern_orange := 1
    necklaces = orange_beads ∧
    green_beads / beads_per_pattern_green >= necklaces ∧
    white_beads / beads_per_pattern_white >= necklaces ∧
    orange_beads / beads_per_pattern_orange >= necklaces

theorem max_necklaces_with_beads : necklace_problem :=
  sorry

end max_necklaces_with_beads_l160_160182


namespace catch_up_time_l160_160537

/-!
# Problem Statement
Xiao Yue took a bus for an outing. On the bus, he noticed Xiao Ling walking in the opposite direction.
After 10 seconds, Xiao Yue got off the bus to chase Xiao Ling.
-/

/-- Define the speeds and time conditions -/
constants (x t : ℝ)  -- Definitions for speed of Xiao Yue and the time he takes to catch up

/-- Define the speeds and initial distance calculations based on the problem description -/
def speed_xiao_yue := x
def speed_xiao_ling := (1/2) * x
def speed_bus := 5 * x
def initial_time := 10
def initial_distance := initial_time * speed_bus

/-- Main theorem: time Xiao Yue needs to catch up to Xiao Ling -/
theorem catch_up_time : t = 110 :=
by
  let lhs := speed_xiao_yue * t
  let rhs := initial_distance + speed_xiao_ling * (t + initial_time)
  have : lhs = x * 110 :=
  sorry

end catch_up_time_l160_160537


namespace lisa_additional_marbles_l160_160867

theorem lisa_additional_marbles (n_friends : ℕ) (initial_marbles : ℕ) (h_friends : n_friends = 12) (h_marbles : initial_marbles = 50) :
  let total_marbles_needed := (n_friends * (n_friends + 1)) / 2 in
  total_marbles_needed - initial_marbles = 28 :=
by
  sorry

end lisa_additional_marbles_l160_160867


namespace mark_more_hours_than_kate_l160_160881

theorem mark_more_hours_than_kate {K : ℕ} (h1 : K + 2 * K + 6 * K = 117) :
  6 * K - K = 65 :=
by
  sorry

end mark_more_hours_than_kate_l160_160881


namespace not_prime_1001_base_l160_160276

theorem not_prime_1001_base (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (n^3 + 1) :=
sorry

end not_prime_1001_base_l160_160276


namespace expected_value_xi_probability_event_A_l160_160324

-- Define basic conditions
def probability_of_success : ℝ := 1 / 3
def number_of_experiments : ℕ := 4
def success_or_failure (results : Fin number_of_experiments → Bool) : ℕ :=
  (results.to_list.filter id).length

-- Define the random variable ξ
def xi (results : Fin number_of_experiments → Bool) : ℕ :=
  abs ((success_or_failure results) - (number_of_experiments - (success_or_failure results)))

-- Event A: "The solution set for the inequality ξ x^2 - ξ x + 1 > 0 is the set of all real numbers ℝ"
def satisfies_event_A (ξ : ℕ) : Prop :=
  match ξ with
  | 0 => true
  | 2 => true
  | 4 => false
  | _ => false

-- Expected value of random variable ξ
theorem expected_value_xi (results : Fin number_of_experiments → Bool) : 
  ∑ (xi_value : ℕ) in {0, 2, 4}, xi_value * P(xi = xi_value) = 148 / 81 := sorry

-- Probability of event A
theorem probability_event_A : 
  P(satisfies_event_A ξ) = 64 / 81 := sorry

end expected_value_xi_probability_event_A_l160_160324


namespace probability_of_missing_coupons_l160_160493

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l160_160493


namespace prob_value_set_prob_distribution_correct_l160_160559
open BigOperators

def possible_values : set ℤ := {0, 1, 2, 3}

def probability_distribution (ξ : ℤ) : ℚ :=
  match ξ with
  | 0 => 1/9
  | 1 => 4/9
  | 2 => 2/9
  | 3 => 2/9
  | _ => 0

theorem prob_value_set :
  ∀ x y : ℤ, x ∈ {1, 2, 3} → y ∈ {1, 2, 3} → |x-2| + |y-x| ∈ possible_values :=
by sorry

theorem prob_distribution_correct :
  ∀ ξ : ℤ, ξ ∈ possible_values → probability_distribution ξ = 
    match ξ with
    | 0 => 1/9
    | 1 => 4/9
    | 2 => 2/9
    | 3 => 2/9
    | _ => 0 :=
by sorry

end prob_value_set_prob_distribution_correct_l160_160559


namespace length_CD_l160_160481

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  2 * (1/2 * (4/3) * π * r^3)

theorem length_CD (L : ℝ) (r : ℝ = 4) (total_volume : ℝ = 320 * π) :
  volume_of_cylinder r L + volume_of_hemisphere r = total_volume →
  L = 44 / 3 :=
by sorry

end length_CD_l160_160481


namespace question_correctness_l160_160967

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l160_160967


namespace area_ratio_l160_160830

-- Definitions corresponding to the conditions
variable {A B C P Q R : Type}
variable (t : ℝ)
variable (h_pos : 0 < t) (h_lt_one : t < 1)

-- Define the areas in terms of provided conditions
noncomputable def area_AP : ℝ := sorry
noncomputable def area_BQ : ℝ := sorry
noncomputable def area_CR : ℝ := sorry
noncomputable def K : ℝ := area_AP * area_BQ * area_CR
noncomputable def L : ℝ := sorry -- Area of triangle ABC

-- The statement to be proved
theorem area_ratio (h_pos : 0 < t) (h_lt_one : t < 1) :
  (K / L) = (1 - t + t^2)^2 :=
sorry

end area_ratio_l160_160830


namespace birds_left_after_week_l160_160577

-- Define the initial number of birds
def initial_chickens : ℕ := 300
def initial_turkeys : ℕ := 200
def initial_guinea_fowls : ℕ := 80

-- Define the losses on odd and even days
def chicken_loss_odd_day : ℕ := 20
def turkey_loss_odd_day : ℕ := 8
def guinea_fowl_loss_odd_day : ℕ := 5

def chicken_loss_even_day : ℕ := 15
def turkey_loss_even_day : ℕ := 5
def guinea_fowl_loss_even_day : ℕ := 3

-- Number of days in a week (7 days: 4 odd days, 3 even days)
def odd_days : ℕ := 4
def even_days : ℕ := 3

theorem birds_left_after_week : 
  let chickens_left := initial_chickens - (chicken_loss_odd_day * odd_days + chicken_loss_even_day * even_days) in
  let turkeys_left := initial_turkeys - (turkey_loss_odd_day * odd_days + turkey_loss_even_day * even_days) in
  let guinea_fowls_left := initial_guinea_fowls - (guinea_fowl_loss_odd_day * odd_days + guinea_fowl_loss_even_day * even_days) in
  chickens_left + turkeys_left + guinea_fowls_left = 379 :=
by
  repeat (exact sorry)

end birds_left_after_week_l160_160577


namespace parabola_focus_distance_l160_160682

theorem parabola_focus_distance (m n : ℝ) 
  (h1 : m^2 = 4 * n)
  (h2 : n = 3)
  (h3 : m = -2 * Real.sqrt 3) : 
  ∥(m, n) - (0, 1)∥ = 4 :=
by
  unfold dist
  sorry

end parabola_focus_distance_l160_160682


namespace tan_double_angle_cos_sub_angle_l160_160859

noncomputable def slope_angle (a b c : ℝ) : ℝ := 
  if b ≠ 0 then real.arctan (a / b) else 0

-- Given conditions as definitions in Lean 4
def line_eq (x y : ℝ) := 3 * x - 4 * y + 5 = 0
def α := slope_angle 3 (-4) 5
def tan_α := real.tan α

-- The proof statements
theorem tan_double_angle (h₁ : tan_α = 3 / 4) : real.tan (2 * α) = 24 / 7 :=
sorry

theorem cos_sub_angle (h₂ : tan_α = 3 / 4) : real.cos (π / 6 - α) = (3 + 4 * real.sqrt 3) / 10 :=
sorry

end tan_double_angle_cos_sub_angle_l160_160859


namespace probability_three_draws_exceed_eight_l160_160180

-- Definitions of the conditions in the problem
def chip_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

def sum_exceeds_eight (l: List ℕ) : Bool :=
  (l.sum > 8)

def three_draws_required (draws: List ℕ) : Prop :=
  draws.length = 3 ∧ sum_exceeds_eight draws ∧ sum_exceeds_eight draws.take 2 = false

-- Formal statement of the problem
theorem probability_three_draws_exceed_eight :
  (∑ x in { l : List ℕ | l.length = 3 ∧ sum_exceeds_eight l ∧ sum_exceeds_eight l.take 2 = false }, 1) / 
  (∑ x in { l : List ℕ | l.length = 3 }, 1) = 3 / 5 := sorry

end probability_three_draws_exceed_eight_l160_160180


namespace steve_keeps_amount_l160_160089

def copies_sold : ℕ := 1000000
def advance_copies : ℕ := 100000
def dollars_per_copy : ℕ := 2
def agent_percentage : ℝ := 0.1

theorem steve_keeps_amount : 
  let copies_sold_without_advance := copies_sold - advance_copies in
  let total_money_made := copies_sold_without_advance * dollars_per_copy in
  let agent_cut := total_money_made * agent_percentage in
  let amount_Steve_kept := total_money_made - agent_cut in
  amount_Steve_kept = 1620000 :=
by 
  -- Declaration of variables according to the conditions
  let copies_sold_without_advance := copies_sold - advance_copies
  let total_money_made := copies_sold_without_advance * dollars_per_copy
  let agent_cut := total_money_made * agent_percentage
  let amount_Steve_kept := total_money_made - agent_cut
  show amount_Steve_kept = 1620000
  sorry

end steve_keeps_amount_l160_160089


namespace people_visited_neither_l160_160166

-- Definitions based on conditions
def total_people : ℕ := 60
def visited_iceland : ℕ := 35
def visited_norway : ℕ := 23
def visited_both : ℕ := 31

-- Theorem statement
theorem people_visited_neither :
  total_people - (visited_iceland + visited_norway - visited_both) = 33 :=
by sorry

end people_visited_neither_l160_160166


namespace right_triangle_area_l160_160148

theorem right_triangle_area (a c : ℝ) (h_leg : a = 24) (h_hypotenuse : c = 25) :
  ∃ b A, (b = real.sqrt (c^2 - a^2)) ∧ (A = 0.5 * a * b) ∧ (A = 84) :=
by
  sorry

end right_triangle_area_l160_160148


namespace log_eq_zero_p_l160_160789

theorem log_eq_zero_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hq2 : -2 < q) 
    (h : log p - log q = log (p / (q + 2))) : p = 0 := 
sorry

end log_eq_zero_p_l160_160789


namespace wall_width_8_l160_160549

theorem wall_width_8 (w h l : ℝ) (V : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h) 
  (vol_eq : w * h * l = 129024) : 
  w = 8 := 
by 
  sorry

end wall_width_8_l160_160549


namespace roots_of_equation_l160_160482

theorem roots_of_equation :
  ∀ (x : ℝ), x^2 - 16 = 0 ↔ (x = 4 ∨ x = -4) :=
by
  intro x
  split
  sorry

end roots_of_equation_l160_160482


namespace polynomial_g_correct_l160_160408

noncomputable def polynomial_g : Polynomial ℚ := 
  Polynomial.C (-41 / 2) + Polynomial.X * 41 / 2 + Polynomial.X ^ 2

theorem polynomial_g_correct
  (f g : Polynomial ℚ)
  (h1 : f ≠ 0)
  (h2 : g ≠ 0)
  (hx : ∀ x, f.eval (g.eval x) = (Polynomial.eval x f) * (Polynomial.eval x g))
  (h3 : Polynomial.eval 3 g = 50) :
  g = polynomial_g :=
sorry

end polynomial_g_correct_l160_160408


namespace betty_cookies_and_brownies_difference_l160_160602

-- Definitions based on the conditions
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def cookies_per_day : ℕ := 3
def brownies_per_day : ℕ := 1
def days : ℕ := 7

-- The proof statement
theorem betty_cookies_and_brownies_difference :
  initial_cookies - (cookies_per_day * days) - (initial_brownies - (brownies_per_day * days)) = 36 :=
by
  sorry

end betty_cookies_and_brownies_difference_l160_160602


namespace max_value_abs_diff_PQ_PR_l160_160297

-- Definitions for the points on the given curves
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Statement of the problem as a theorem
theorem max_value_abs_diff_PQ_PR (P Q R : ℝ × ℝ)
(hyp_P : hyperbola P.1 P.2)
(hyp_Q : circle1 Q.1 Q.2)
(hyp_R : circle2 R.1 R.2) :
  max (abs (dist P Q - dist P R)) = 10 :=
sorry

end max_value_abs_diff_PQ_PR_l160_160297


namespace sue_candies_l160_160388

def candies : Type := ℕ

variable (bob mary john sam total sue : candies)
variable (h_bob : bob = 10)
variable (h_mary : mary = 5)
variable (h_john : john = 5)
variable (h_sam : sam = 10)
variable (h_total : total = 50)

theorem sue_candies (bob mary john sam total sue : candies)
  (h_bob : bob = 10) 
  (h_mary : mary = 5) 
  (h_john : john = 5) 
  (h_sam : sam = 10) 
  (h_total : total = 50) : 
  sue = 20 :=
sorry

end sue_candies_l160_160388


namespace remaining_credit_l160_160828

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l160_160828


namespace number_of_correct_conclusions_l160_160221

theorem number_of_correct_conclusions : 
    (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
    (∀ x : ℝ, (x ≠ 0 → x - Real.sin x ≠ 0)) ∧
    (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
    (¬ (∀ x : ℝ, x - Real.log x > 0))
    → 3 = 3 :=
by
  sorry

end number_of_correct_conclusions_l160_160221


namespace area_of_quadrilateral_l160_160892

-- Definitions for given conditions
variables (A B C D E : Type)
variables [EuclideanGeometry A B C D E]
variables (AB : Segment A B) (BC : Segment B C) (CD : Segment C D) (BD : Segment B D)(AC : Segment A C)
noncomputable def angle90 (ABC : Angle A B C) : Prop := ABC = 90
noncomputable def segmentLengthAC (AC : Segment A C) : ℝ := 25
noncomputable def segmentLengthCD (CD : Segment C D) : ℝ := 40
noncomputable def intersectionOfDiagonals (E : Point E) (intersection : IntersectionOfDiagonals AC BD) : Prop := E = intersection
noncomputable def segmentLengthAE (AE : Segment A E) : ℝ := 8

-- The proof statement
theorem area_of_quadrilateral :
  angle90 (angle A B C) ∧ segmentLengthAC AC = 25 ∧ segmentLengthCD CD = 40 ∧ intersectionOfDiagonals E (intersection AC BD) ∧ segmentLengthAE AE = 8
  → areaOfQuadrilateral A B C D = 594 :=
sorry

end area_of_quadrilateral_l160_160892


namespace new_cube_edge_l160_160914

def volume (a: ℝ) := a^3

def total_volume := volume 6 + volume 8 + volume 10 + volume 12 + volume 14

def edge_new_cube := real.cbrt total_volume

theorem new_cube_edge :
  abs (edge_new_cube - 18.39) < 0.01 := 
sorry

end new_cube_edge_l160_160914


namespace triangle_area_gt_half_l160_160509

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l160_160509


namespace minimum_distance_at_meeting_time_distance_glafira_to_meeting_l160_160999

variables (U g τ V : ℝ)
-- assumption: 2 * U ≥ g * τ
axiom h : 2 * U ≥ g * τ

noncomputable def motion_eq1 (t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def motion_eq2 (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2

noncomputable def distance (t : ℝ) : ℝ := 
|motion_eq1 U g t - motion_eq2 U g τ t|

noncomputable def meeting_time : ℝ := (2 * U / g) + (τ / 2)

theorem minimum_distance_at_meeting_time : distance U g τ meeting_time = 0 := sorry

noncomputable def distance_from_glafira_to_meeting : ℝ := 
V * meeting_time

theorem distance_glafira_to_meeting : 
distance_from_glafira_to_meeting U g τ V = V * ((τ / 2) + (U / g)) := sorry

end minimum_distance_at_meeting_time_distance_glafira_to_meeting_l160_160999


namespace count_three_digit_numbers_with_2_without_6_l160_160774

theorem count_three_digit_numbers_with_2_without_6 : 
  let total_without_6 : ℕ := 648
  let total_without_6_and_2 : ℕ := 448
  total_without_6 - total_without_6_and_2 = 200 :=
by 
  have total_without_6 := 8 * 9 * 9
  have total_without_6_and_2 := 7 * 8 * 8
  rw total_without_6
  rw total_without_6_and_2
  exact calc
    8 * 9 * 9 - 7 * 8 * 8 = 648 - 448 := by simp
    ... = 200 := by norm_num

end count_three_digit_numbers_with_2_without_6_l160_160774


namespace probability_of_missing_coupons_l160_160491

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l160_160491


namespace total_go_stones_correct_l160_160133

-- Definitions based on the problem's conditions
def stones_per_bundle : Nat := 10
def num_bundles : Nat := 3
def white_stones : Nat := 16

-- A function that calculates the total number of go stones
def total_go_stones : Nat :=
  num_bundles * stones_per_bundle + white_stones

-- The theorem we want to prove
theorem total_go_stones_correct : total_go_stones = 46 :=
by
  sorry

end total_go_stones_correct_l160_160133


namespace min_distance_and_distance_from_Glafira_l160_160994

theorem min_distance_and_distance_from_Glafira 
  (U g τ V : ℝ) (h : 2 * U ≥ g * τ) :
  let T := (τ / 2) + (U / g) in
  s T = 0 ∧ (V * T = V * (τ / 2 + U / g)) :=
by
  -- Define the positions y1(t) and y2(t)
  let y1 := λ t, U * t - (g * t^2) / 2
  let y2 := λ t, U * (t - τ) - (g * (t - τ)^2) / 2
  -- Define the distance s(t)
  let s := λ t, |y1 t - y2 t|
  -- Start the proof
  sorry

end min_distance_and_distance_from_Glafira_l160_160994


namespace min_k_cells_l_triomino_l160_160150

open Finset

-- Definitions for the problem context
def is_marked (board : Fin 81 → bool) (cell : Fin 81) : Prop :=
  board cell

def l_triomino (cells : Finset (Fin 81)) : Prop :=
  (∃ (c₁ c₂ c₃ : Fin 81), {c₁, c₂, c₃} = cells ∧ 
                         (c₁ / 9 = c₂ / 9 ∧ c₂ / 9 = c₃ / 9 ∧ (c₁ % 9 + 1) % 9 = c₂ % 9 ∧ (c₂ % 9 + 1) % 9 = c₃ % 9) ∨ 
                         (c₁ % 9 = c₂ % 9 ∧ c₂ % 9 = c₃ % 9 ∧ (c₁ / 9 + 1) % 9 = c₂ / 9 ∧ (c₂ / 9 + 1) % 9 = c₃ / 9))

def touches_two_marked_cells (board : Fin 81 → bool) (cells : Finset (Fin 81)) : Prop :=
  ∃ (c₁ c₂ : Fin 81), {c₁, c₂} ⊆ cells ∧ is_marked board c₁ ∧ is_marked board c₂

-- The final proof statement
theorem min_k_cells_l_triomino :
  ∃ (board : Fin 81 → bool), (∀ (cells : Finset (Fin 81)), l_triomino cells → touches_two_marked_cells board cells) ∧ 
                             (∑ i, if is_marked board i then 1 else 0 = 56) :=
  sorry

end min_k_cells_l_triomino_l160_160150


namespace sequence_subtraction_final_value_l160_160899

theorem sequence_subtraction_final_value :
  (2023 - (∑ n in Finset.range 2023 \ {0}, 1 / (n + 1))) = 1 :=
by
  sorry

end sequence_subtraction_final_value_l160_160899


namespace percentage_of_men_in_company_l160_160184

theorem percentage_of_men_in_company 
  (M W : ℝ) 
  (h1 : 0.60 * M + 0.35 * W = 50) 
  (h2 : M + W = 100) : 
  M = 60 :=
by
  sorry

end percentage_of_men_in_company_l160_160184


namespace max_sum_of_first_n_terms_l160_160454

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem max_sum_of_first_n_terms (h1 : a 6 + a 7 + a 8 > 0) (h2 : a 6 + a 9 < 0) (h_arith : is_arithmetic_sequence a) :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≠ 7 → sum (range m) a ≤ sum (range 7) a) :=
by
  sorry

end max_sum_of_first_n_terms_l160_160454


namespace triangle_area_correct_l160_160266

-- Define the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (8, -1)
def C : ℝ × ℝ := (12, 6)

-- Function to compute the area of the triangle given its vertices
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)  -- vector CA
  let w := (B.1 - C.1, B.2 - C.2)  -- vector CB
  (real.abs (v.1 * w.2 - v.2 * w.1)) / 2

-- Statement to prove
theorem triangle_area_correct : triangle_area A B C = 36 :=
by
  -- proof goes here
  sorry

end triangle_area_correct_l160_160266


namespace fill_squares_to_make_equation_true_l160_160381

theorem fill_squares_to_make_equation_true:
  ∃ (x₁ x₂ x₃ : ℕ), (2 < x₁ ∧ 2 < x₂ ∧ 2 < x₃) ∧ (x₁ * (x₂ + x₃))^2 = 8 * x₂ * x₃ * 9 :=
by
  existsi (3, 1, 3)
  -- Since (3 * (1 + 3))^2 = 8 * 1 * 3 * 9
  -- (3 * 4)^2 = 216
  -- 12^2 = 144
  -- Sorry to skip the details
  sorry

end fill_squares_to_make_equation_true_l160_160381


namespace minimum_bottles_needed_l160_160571

theorem minimum_bottles_needed :
  (∃ n : ℕ, n * 45 ≥ 720 - 20 ∧ (n - 1) * 45 < 720 - 20) ∧ 720 - 20 = 700 :=
by
  sorry

end minimum_bottles_needed_l160_160571


namespace symmetrical_circle_eq_l160_160687

variable (a b : ℝ)

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

noncomputable def perpendicular_bisector (p q : ℝ × ℝ) : ℝ := 
  if p.1 = q.1 then p.1 else -(p.2 - q.2) / (p.1 - q.1)

def eq_circle_symmetric (p q : ℝ × ℝ) (k : ℝ) (l : ℝ) : Prop :=
  let midpoint := midpoint p q
  let perp_bisector := perpendicular_bisector p q
  (midpoint.1 - l)^2 + (midpoint.2 - k)^2 = 1
     
theorem symmetrical_circle_eq (a b : ℝ) :
  eq_circle_symmetric (a, b) (3 - b, 3 - a) 2 3 → (1 - 1)^2 + (2 - 2)^2 = 1 :=
sorry

end symmetrical_circle_eq_l160_160687


namespace quadratic_has_two_real_roots_l160_160480

-- Definitions based on conditions
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, b = a * q ∧ c = a * q^2

-- Main statement in Lean 4
theorem quadratic_has_two_real_roots (a b : ℝ) (h : is_geometric_sequence 2 b a) :
  let discriminant := b^2 - 4 * a * (1 / 3) in
  discriminant > 0 :=
by {
  sorry
}

end quadratic_has_two_real_roots_l160_160480


namespace g_of_12_l160_160334

def g (n : ℕ) : ℕ := n^2 - n + 23

theorem g_of_12 : g 12 = 155 :=
by
  sorry

end g_of_12_l160_160334


namespace steve_keeps_amount_l160_160090

def copies_sold : ℕ := 1000000
def advance_copies : ℕ := 100000
def dollars_per_copy : ℕ := 2
def agent_percentage : ℝ := 0.1

theorem steve_keeps_amount : 
  let copies_sold_without_advance := copies_sold - advance_copies in
  let total_money_made := copies_sold_without_advance * dollars_per_copy in
  let agent_cut := total_money_made * agent_percentage in
  let amount_Steve_kept := total_money_made - agent_cut in
  amount_Steve_kept = 1620000 :=
by 
  -- Declaration of variables according to the conditions
  let copies_sold_without_advance := copies_sold - advance_copies
  let total_money_made := copies_sold_without_advance * dollars_per_copy
  let agent_cut := total_money_made * agent_percentage
  let amount_Steve_kept := total_money_made - agent_cut
  show amount_Steve_kept = 1620000
  sorry

end steve_keeps_amount_l160_160090


namespace num_four_digit_multiples_of_7_l160_160718

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160718


namespace volume_of_earth_dug_out_l160_160564

theorem volume_of_earth_dug_out (diameter depth : ℝ) (h_diameter : diameter = 6) (h_depth : depth = 24) :
  let r := diameter / 2 in
  let V := Real.pi * r^2 * depth in
  V = 216 * Real.pi :=
by
  sorry

end volume_of_earth_dug_out_l160_160564


namespace lisa_needs_28_more_marbles_l160_160863

theorem lisa_needs_28_more_marbles :
  ∀ (friends : ℕ) (initial_marbles : ℕ),
  friends = 12 → 
  initial_marbles = 50 →
  (∀ n, 1 ≤ n ∧ n ≤ friends → ∃ (marbles : ℕ), marbles ≥ 1 ∧ ∀ i j, (i ≠ j ∧ i ≠ 0 ∧ j ≠ 0) → (marbles i ≠ marbles j)) →
  ( ∑ k in finset.range (friends + 1), k ) - initial_marbles = 28 :=
by
  intros friends initial_marbles h_friends h_initial_marbles _,
  rw [h_friends, h_initial_marbles],
  sorry

end lisa_needs_28_more_marbles_l160_160863


namespace nice_people_count_l160_160217

theorem nice_people_count
  (barry_count kevin_count julie_count joe_count : ℕ)
  (nice_barry nice_kevin nice_julie nice_joe : ℕ) 
  (H1 : barry_count = 24)
  (H2 : kevin_count = 20)
  (H3 : julie_count = 80)
  (H4 : joe_count = 50)
  (H5 : nice_barry = barry_count)
  (H6 : nice_kevin = kevin_count / 2)
  (H7 : nice_julie = (3 * julie_count) / 4)
  (H8 : nice_joe = joe_count / 10)
  : nice_barry + nice_kevin + nice_julie + nice_joe = 99 :=
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8]
  norm_num
  sorry

end nice_people_count_l160_160217


namespace expected_ties_approx_l160_160510

noncomputable def expected_number_of_ties : ℚ :=
  ∑ k in Finset.range 5 + 1, Nat.choose (2 * k) k / (2^(2 * k))

theorem expected_ties_approx :
  (expected_number_of_ties : ℚ) ≈ 1.707 :=
by
  sorry

end expected_ties_approx_l160_160510


namespace min_value_of_expression_l160_160658

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) :
  2 * a + b ≥ 4 * Real.sqrt 2 - 3 := 
sorry

end min_value_of_expression_l160_160658


namespace count_four_digit_multiples_of_7_l160_160767

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160767


namespace find_R_when_S_is_five_l160_160953

theorem find_R_when_S_is_five (g : ℚ) :
  (∀ (S : ℚ), R = g * S^2 - 5) →
  (R = 25 ∧ S = 3) →
  R = (250 / 3) - 5 :=
by 
  sorry

end find_R_when_S_is_five_l160_160953


namespace third_derivative_l160_160171

noncomputable def y (x : ℝ) : ℝ :=
  (Real.log (3 + x)) / (3 + x)

theorem third_derivative (x : ℝ) :
  deriv^[3] (λ x, (Real.log (3 + x)) / (3 + x)) x = (11 - 6 * Real.log (3 + x)) / (3 + x)^4 :=
by sorry

end third_derivative_l160_160171


namespace combined_stickers_l160_160127

theorem combined_stickers (k j a : ℕ) (h : 7 * j + 5 * a = 54) (hk : k = 42) (hk_ratio : k = 7 * 6) :
  j + a = 54 :=
by
  sorry

end combined_stickers_l160_160127


namespace four_digit_multiples_of_7_l160_160712

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160712


namespace frog_toad_pairing_l160_160132

def frogs := 2017
def toads := 2017
def frog_friend_count := 2

theorem frog_toad_pairing :
  (∃ (N : ℕ) (D : ℕ) (S : ℕ), 
    D = 1009 ∧ 
    S = 2^1009 - 2 ∧ 
    (∀ (f : Fin frogs), 
      ∃ (t1 t2 : Fin toads),
        t1 ≠ t2 ∧
        true -- Just to fill in the friends relationship condition -- sort of placeholder
    )
  ) :=
begin
  sorry -- Proof is omitted
end

end frog_toad_pairing_l160_160132


namespace number_of_smaller_triangles_l160_160952

theorem number_of_smaller_triangles (vertices internal_points : ℕ) (total_points : ℕ) :
  vertices = 3 → internal_points = 7 → total_points = vertices + internal_points →
  ∃ n : ℕ, n = 15 :=
by
  intros h1 h2 h3
  use 15
  exact sorry

end number_of_smaller_triangles_l160_160952


namespace tim_linda_mowing_time_l160_160168

-- Definitions based on given conditions
def tim_time : ℝ := 1.5
def linda_time : ℝ := 2.0
def tim_work_rate : ℝ := 1 / tim_time
def linda_work_rate : ℝ := 1 / linda_time
def combined_work_rate : ℝ := tim_work_rate + linda_work_rate
def combined_time : ℝ := 1 / combined_work_rate

-- The proof problem statement
theorem tim_linda_mowing_time :
  (combined_time * 60) = 360 / 7 :=
sorry

end tim_linda_mowing_time_l160_160168


namespace total_gifts_l160_160058

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l160_160058


namespace proof_AC_time_l160_160535

noncomputable def A : ℝ := 1/10
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 1/30

def rate_A_B (A B : ℝ) := A + B = 1/6
def rate_B_C (B C : ℝ) := B + C = 1/10
def rate_A_B_C (A B C : ℝ) := A + B + C = 1/5

theorem proof_AC_time {A B C : ℝ} (h1 : rate_A_B A B) (h2 : rate_B_C B C) (h3 : rate_A_B_C A B C) : 
  (1 : ℝ) / (A + C) = 7.5 :=
sorry

end proof_AC_time_l160_160535


namespace problem_inequality_l160_160289

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable (A : ℝ)
variable (b : Fin n → ℝ)
variable (x : Fin n → ℝ)

noncomputable def A_def := ∑ i, (a i)^2
noncomputable def b_def := λ i => (a i) / (Real.sqrt A_def)

theorem problem_inequality (H : A ≠ 0) :
  (∑ i, b_def i * (x i - a i)) ≤ (Real.sqrt (∑ i, (x i)^2)) - (Real.sqrt (A_def)) :=
sorry

end problem_inequality_l160_160289


namespace necessary_but_not_sufficient_l160_160175

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 2 → (x^2 - x - 2 >= 0) ∨ (x >= -1 ∧ x < 2)) ∧ ((-1 < x ∧ x < 2) → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_l160_160175


namespace find_value_of_E_l160_160620

variables (Q U I E T Z : ℤ)

theorem find_value_of_E (hZ : Z = 15) (hQUIZ : Q + U + I + Z = 60) (hQUIET : Q + U + I + E + T = 75) (hQUIT : Q + U + I + T = 50) : E = 25 :=
by
  have hQUIZ_val : Q + U + I = 45 := by linarith [hZ, hQUIZ]
  have hQUIET_val : E + T = 30 := by linarith [hQUIZ_val, hQUIET]
  have hQUIT_val : T = 5 := by linarith [hQUIZ_val, hQUIT]
  linarith [hQUIET_val, hQUIT_val]

end find_value_of_E_l160_160620


namespace diameter_expression_l160_160105

noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (r ^ 3)

noncomputable def find_diameter (r1 : ℝ) : ℝ :=
  let volume1 := sphereVolume r1
  let volume2 := 3 * volume1
  let r2 := Real.cbrt (volume2 * 3 / (4 * Real.pi)) -- Solving for r in sphereVolume formula
  2 * r2   -- The diameter is twice the radius

theorem diameter_expression : 
  ∃ (a b : ℕ), b % 5 ≠ 0 ∧ find_diameter 6 = a * Real.cbrt b ∧ a + b = 18 :=
by {
  use [12, 6], -- Providing specific values for a and b
  split, {
    exact dec_trivial,   -- Proving b doesn't have perfect cube factors
  },
  split,
  {   
    have h1 : find_diameter (6 : ℝ) = 2 * Real.cbrt (2592 * 3 / 4),
    sorry,
    have h2 : 2 * Real.cbrt (2592 * 3 / 4) = 12 * Real.cbrt 6,
    sorry
  },
  {
    exact dec_trivial,  -- Directly proving a + b = 18 as a = 12 and b= 6
  }
}

end diameter_expression_l160_160105


namespace value_of_expression_l160_160526

theorem value_of_expression (b : ℚ) (h : b = 1/3) : (3 * b⁻¹ + (b⁻¹ / 3)) / b = 30 :=
by
  rw [h]
  sorry

end value_of_expression_l160_160526


namespace concyclic_points_l160_160379

/-
In the quadrilateral ABCD with perpendicular diagonals AC and BD,
let points M and N lie on BD and be symmetric with respect to AC.
Let X and Y be the reflections of M with respect to lines AB and BC,
and let Z and W be the reflections of N with respect to lines CD and DA.
Prove that points X, Y, Z, and W are concyclic.
-/

variables {A B C D M N X Y Z W : Point}

-- Definitions and assumptions
axiom perpendicular_diagonals : AC ⊥ BD
axiom M_N_on_BD : lies_on M BD ∧ lies_on N BD
axiom M_N_symmetric_AC : symmetric M N AC
axiom X_Y_reflections : reflection M AB X ∧ reflection M BC Y
axiom Z_W_reflections : reflection N CD Z ∧ reflection N DA W

-- Prove points X, Y, Z, W are concyclic
theorem concyclic_points : concyclic X Y Z W := 
by sorry

end concyclic_points_l160_160379


namespace fibonacci_sum_7_fibonacci_sum_2015_m_l160_160094

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def sum_fib (n : ℕ) : ℕ :=
(nat.sum_range_succ' n).sum (λ k, fibonacci k)

theorem fibonacci_sum_7 : sum_fib 7 = 33 :=
sorry

theorem fibonacci_sum_2015_m (m : ℕ) (h : fibonacci 2017 = m) : sum_fib 2015 = m - 1 :=
sorry

end fibonacci_sum_7_fibonacci_sum_2015_m_l160_160094


namespace standard_deviation_is_sqrt_six_l160_160129

-- Given conditions
def num1 := 5
def num2 := 8
def num3 := 11

-- Definitions required for the proof
def average (a b c : ℕ) : ℝ := (a + b + c) / 3
def variance (a b c : ℕ) : ℝ :=
  let av := average a b c
  ( ((a - av) ^ 2) + ((b - av) ^ 2) + ((c - av) ^ 2) ) / 3
def standard_deviation (a b c : ℕ) : ℝ := real.sqrt (variance a b c)

-- The statement that needs to be proved
theorem standard_deviation_is_sqrt_six : standard_deviation num1 num2 num3 = real.sqrt 6 := by
  sorry

end standard_deviation_is_sqrt_six_l160_160129


namespace problem_statement_l160_160791

theorem problem_statement : ∀ (x y : ℝ), |x - 2| + (y + 3)^2 = 0 → (x + y)^2023 = -1 :=
by
  intros x y h
  sorry

end problem_statement_l160_160791


namespace domain_of_function_l160_160913

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (1 - x) / Real.log 2)

theorem domain_of_function :
  { x : ℝ | 1 - x > 0 } ∩ { x : ℝ | Real.log (1 - x) / Real.log 2 ≥ 0 } = { x : ℝ | x ≤ 0 } :=
by
  ext
  simp
  split
  · intro h
    cases h with h1 h2
    linarith
  · intro h
    split
    · linarith
    · rw Real.div_le_iff
      { norm_num,
        exact Real.log_pos_one }
      { exact Real.log_pos_one }

end domain_of_function_l160_160913


namespace exists_numbering_for_nonagon_no_numbering_for_decagon_l160_160172

-- Definitions for the problem setup
variable (n : ℕ) 
variable (A : Fin n → Point)
variable (O : Point)

-- Definition for the numbering function
variable (f : Fin (2 * n) → ℕ)

-- First statement for n = 9
theorem exists_numbering_for_nonagon :
  ∃ (f : Fin 18 → ℕ), (∀ i : Fin 9, f (i : Fin 9) + f (i + 9) + f ((i + 1) % 9) = 15) :=
sorry

-- Second statement for n = 10
theorem no_numbering_for_decagon :
  ¬ ∃ (f : Fin 20 → ℕ), (∀ i : Fin 10, f (i : Fin 10) + f (i + 10) + f ((i + 1) % 10) = 16) :=
sorry

end exists_numbering_for_nonagon_no_numbering_for_decagon_l160_160172


namespace least_d_value_l160_160902

theorem least_d_value (c d : ℕ) (hc : nat.proper_divisors c = {1, p, p^2, c}) (hd : nat.proper_divisors d = {1, q, q^2, ... , d}) (hcd : d % c = 0) :
  d = 24 :=
sorry

end least_d_value_l160_160902


namespace probability_of_four_four_balls_l160_160594

open Nat

def urn_initial_red_balls := 2
def urn_initial_blue_balls := 1
def operations := 5

def final_ball_count := 8

def probability_four_four_balls : ℚ := 8 / 21

theorem probability_of_four_four_balls :
  let final_red_balls := 4
  let final_blue_balls := 4
  let probability := 
    ∑ 
      (sequence : Finset (Finset (List char)))
      ((sequence.card = 10) ∧ (sequence.filter (λ s, s = 'R')).card = 3)
      (sequence.filter (λ s, s = 'B')).card = 2) 
    $\frac{2}{3} \times \frac{3}{4} \times \frac{4}{5} \times \frac{1}{6} \times \frac{2}{7}$ :=
  probability = probability_four_four_balls := sorry

end probability_of_four_four_balls_l160_160594


namespace value_of_factorial_fraction_l160_160961

open Nat

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem value_of_factorial_fraction : (factorial 15) / ((factorial 6) * (factorial 9)) = 4165 := by
  sorry

end value_of_factorial_fraction_l160_160961


namespace sticks_flats_boxes_36_flats_boxes_210_boxes_volume_n_boxes_8_fact_l160_160226

-- Definitions
def is_stick (x y z : ℕ) : Prop := (x = 1 ∧ y = 1)
def is_flat (x y z : ℕ) : Prop := (x = 1 ∧ y > 1)
def is_box (x y z : ℕ) : Prop := (x > 1)

-- Volume 36
theorem sticks_flats_boxes_36 : 
  ∃ sticks flats boxes : ℕ, 
    sticks = 1 ∧ flats = 4 ∧ boxes = 3 ∧ 
    (∀ (x y z : ℕ), x * y * z = 36 → 
     (x ≤ y ∧ y ≤ z) →
       ((is_stick x y z → sticks = 1) ∧ 
       (is_flat x y z → flats = 4) ∧ 
       (is_box x y z → boxes = 3))) := 
by sorry

-- Volume 210
theorem flats_boxes_210 :
  ∃ flats boxes : ℕ, 
    flats = 7 ∧ boxes = 6 ∧ 
    (∀ (x y z : ℕ), x * y * z = 210 → 
     (x ≤ y ∧ y ≤ z) →
      ((is_flat x y z → flats = 7) ∧ 
      (is_box x y z → boxes = 6))) := 
by sorry

-- General formula for volume n
theorem boxes_volume_n (p : ℕ → ℕ) (e : ℕ → ℕ) (k : ℕ) (h : k ≥ 3) :
  ∃ boxes : ℕ, 
    boxes = ∏ i in finset.range k, (e i + 1) / 2 - 1 := 
by sorry

-- Specific case for volume 8!
theorem boxes_8_fact :
  ∃ boxes : ℕ, boxes = 280 := 
by sorry

end sticks_flats_boxes_36_flats_boxes_210_boxes_volume_n_boxes_8_fact_l160_160226


namespace tetrahedron_count_from_cube_vertices_l160_160783

noncomputable def count_tetrahedrons_from_cube_vertices : Nat := 8

def cannot_form_tetrahedron_faces : Nat := 12

theorem tetrahedron_count_from_cube_vertices : 
  (nat.choose count_tetrahedrons_from_cube_vertices 4) - cannot_form_tetrahedron_faces = 58 := 
  by sorry

end tetrahedron_count_from_cube_vertices_l160_160783


namespace sequence_a5_l160_160325

theorem sequence_a5:
  (∀ n : ℕ, 1 + a_n = (1 + a_1) * 2 ^ (n - 1))
  → a_1 = 1
  → a_5 = 31 := 
begin
  intros h₁ h₂,
  sorry,
end

end sequence_a5_l160_160325


namespace polynomial_remainder_l160_160530

theorem polynomial_remainder (x : ℝ) : 
  ∃ q r : ℝ[X], (x^3 - 2) = (q * (x^2 - 2) + r) ∧ degree r < degree (x^2 - 2) ∧ r = 2 * x - 2 :=
by 
  let q := polynomial.C 1 * polynomial.X 
  let r := 2 * polynomial.X - 2 
  use q, r
  split 
  {
    have h : (x^3 - 2) = q * (x^2 - 2) + r 
    from sorry ,
    exact h,
    -- Proof of (x^3 - 2) = q * (x^2 - 2) + r
  }
  split 
  {
    have degree_r_lt_degree_x_squared_minus_two : degree r < degree (x^2 - 2) 
    from sorry,
    exact degree_r_lt_degree_x_squared_minus_two,
    -- Proof that remainder has a smaller degree than the divisor
  }
  exact sorry
  -- Proof that the remainder is equal to 2 * x - 2

end polynomial_remainder_l160_160530


namespace count_four_digit_multiples_of_7_l160_160733

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160733


namespace cos_double_angle_identity_l160_160349

variable (θ : ℝ)

theorem cos_double_angle_identity 
  (h : cos θ + 2 * sin θ = 3 / 2) : cos (2 * θ) = 1 / 2 := 
  sorry

end cos_double_angle_identity_l160_160349


namespace kevin_age_l160_160825

theorem kevin_age (K : ℤ) (h1 : ∀ vanessa_age : ℤ, vanessa_age = 2 → ∃ kevin_age : ℤ, kevin_age = K ∧ K + 5 = 3 * (vanessa_age + 5)) : K = 16 :=
by
  -- Given conditions
  let vanessa_age := 2
  have h2 : vanessa_age = 2 := rfl
  obtain ⟨kevin_age, kevin_eq, h3⟩ := h1 vanessa_age h2
  -- The condition that in 5 years, Kevin's age will be 3 times Vanessa's age
  have h4 : K + 5 = 3 * (vanessa_age + 5) := h3
  -- Simplify to find Kevin's current age
  calc
    K + 5 = 21 := by rw [h4, vanessa_age]; norm_num
    K = 16 := by linarith

end kevin_age_l160_160825


namespace even_sum_probability_correct_l160_160144

-- Definition: Calculate probabilities based on the given wheels
def even_probability_wheel_one : ℚ := 1/3
def odd_probability_wheel_one : ℚ := 2/3
def even_probability_wheel_two : ℚ := 1/4
def odd_probability_wheel_two : ℚ := 3/4

-- Probability of both numbers being even
def both_even_probability : ℚ := even_probability_wheel_one * even_probability_wheel_two

-- Probability of both numbers being odd
def both_odd_probability : ℚ := odd_probability_wheel_one * odd_probability_wheel_two

-- Final probability of the sum being even
def even_sum_probability : ℚ := both_even_probability + both_odd_probability

theorem even_sum_probability_correct : even_sum_probability = 7/12 := 
sorry

end even_sum_probability_correct_l160_160144


namespace correct_model_is_pakistan_traditional_l160_160969

-- Given definitions
def hasPrimitiveModel (country : String) : Prop := country = "Nigeria"
def hasTraditionalModel (country : String) : Prop := country = "India" ∨ country = "Pakistan" ∨ country = "Nigeria"
def hasModernModel (country : String) : Prop := country = "China"

-- The proposition to prove
theorem correct_model_is_pakistan_traditional :
  (hasPrimitiveModel "Nigeria")
  ∧ (hasModernModel "China")
  ∧ (hasTraditionalModel "India")
  ∧ (hasTraditionalModel "Pakistan") →
  (hasTraditionalModel "Pakistan") := by
  intros h
  exact (h.right.right.right)

end correct_model_is_pakistan_traditional_l160_160969


namespace four_digit_multiples_of_7_l160_160706

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160706


namespace area_ratio_of_squares_l160_160399

theorem area_ratio_of_squares (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let E := (s / 2,  -s / 2)
  let F := (3 * s / 2, s / 2)
  let G := (s / 2, 3 * s / 2)
  let H := (-s / 2, s / 2)
  let area_ABCD := s * s
  let side_EFGH := Real.sqrt ((3 * s / 2 - s / 2) ^ 2 + (s / 2 + s / 2) ^ 2)
  let area_EFGH := side_EFGH ^ 2
  in area_EFGH / area_ABCD = 5 / 2 :=
by
  intros
  sorry
  
end area_ratio_of_squares_l160_160399


namespace unit_circle_inequality_l160_160554

theorem unit_circle_inequality 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (habcd : a * b + c * d = 1) 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1^2 + y1^2 = 1)
  (hx2 : x2^2 + y2^2 = 1)
  (hx3 : x3^2 + y3^2 = 1)
  (hx4 : x4^2 + y4^2 = 1) :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2 ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
sorry

end unit_circle_inequality_l160_160554


namespace prove_system_of_inequalities_l160_160086

theorem prove_system_of_inequalities : 
  { x : ℝ | x / (x - 2) ≥ 0 ∧ 2 * x + 1 ≥ 0 } = Set.Icc (-(1:ℝ)/2) 0 ∪ Set.Ioi 2 := 
by
  sorry

end prove_system_of_inequalities_l160_160086


namespace problem_statement_l160_160043

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable theory

theorem problem_statement :
  (∀ n : ℕ, a n * real.floor (a n) = real.exp (log 49 * n) + 2 * n + 1) →
  2 * real.floor (finset.sum (finset.range 2017) (λ n, a (n + 1) / 2)) + 1 =
  7 * (real.exp (log 7 * 2017) - 1) / 6 :=
begin
  sorry
end

end problem_statement_l160_160043


namespace distribute_stickers_l160_160341

-- Definitions based on conditions
def stickers : ℕ := 10
def sheets : ℕ := 5

-- Theorem stating the equivalence of distributing the stickers onto sheets
theorem distribute_stickers :
  (Nat.choose (stickers + sheets - 1) (sheets - 1)) = 1001 :=
by 
  -- Here is where the proof would go, but we skip it with sorry for the purpose of this task
  sorry

end distribute_stickers_l160_160341


namespace lisa_additional_marbles_l160_160865

theorem lisa_additional_marbles (n_friends : ℕ) (initial_marbles : ℕ) (h_friends : n_friends = 12) (h_marbles : initial_marbles = 50) :
  let total_marbles_needed := (n_friends * (n_friends + 1)) / 2 in
  total_marbles_needed - initial_marbles = 28 :=
by
  sorry

end lisa_additional_marbles_l160_160865


namespace area_does_not_depend_on_P_l160_160021

variables (A B C P A' B' C' : Type)
variables [triangle A B C]

-- Let \(A, B, C\) be points of triangle \(ABC\)
-- Let \(A_1, B_1, C_1\) be the midpoints of \(BC, CA, AB\) respectively
variables (A1 B1 C1 : Type)
variables [midpoint A1 B C]
variables [midpoint B1 C A]
variables [midpoint C1 A B]

-- Let \(P\) be a variable point on the circumcircle of \(\triangle ABC\)
variables [onCircumcircle P A B C]

-- Let lines \(PA_1, PB_1, PC_1\) meet the circumcircle again at \(A', B', C'\) respectively
variables [meetCircumcircle PA1 A']
variables [meetCircumcircle PB1 B']
variables [meetCircumcircle PC1 C']

-- Assume that the points \(A, B, C, A', B', C'\) are distinct
variables [distinctPoints A B C]
variables [distinctPoints A' B' C']

-- Assume that lines \(AA', BB', CC'\) form a triangle
variables [formTriangle A A' B B' C C']

-- Prove that the area of this triangle does not depend on the position of \(P\)
theorem area_does_not_depend_on_P :
  ∀ (P : Type), area (triangle AA' BB' CC') = area (triangle AA' BB' CC') :=
sorry

end area_does_not_depend_on_P_l160_160021


namespace opposites_of_each_other_l160_160352

theorem opposites_of_each_other (a b : ℚ) (h : a + b = 0) : a = -b :=
  sorry

end opposites_of_each_other_l160_160352


namespace min_value_fraction_sum_l160_160659

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_eq : 1 = 2 * a + b) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_sum_l160_160659


namespace total_boys_in_groups_l160_160487

-- Definitions of number of groups
def total_groups : ℕ := 35
def groups_with_1_boy : ℕ := 10
def groups_with_at_least_2_boys : ℕ := 19
def groups_with_3_boys_twice_groups_with_3_girls (groups_with_3_boys groups_with_3_girls : ℕ) : Prop :=
  groups_with_3_boys = 2 * groups_with_3_girls

theorem total_boys_in_groups :
  ∃ (groups_with_3_girls groups_with_3_boys groups_with_1_girl_2_boys : ℕ),
    groups_with_1_boy + groups_with_at_least_2_boys + groups_with_3_girls = total_groups
    ∧ groups_with_3_boys_twice_groups_with_3_girls groups_with_3_boys groups_with_3_girls
    ∧ groups_with_1_girl_2_boys + groups_with_3_boys = groups_with_at_least_2_boys
    ∧ (groups_with_1_boy * 1 + groups_with_1_girl_2_boys * 2 + groups_with_3_boys * 3) = 60 :=
sorry

end total_boys_in_groups_l160_160487


namespace max_area_triangle_PAB_l160_160667

-- Define the line equation
def line (x y : ℝ) := 2 * x - y - 2 = 0

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - y^2 = 1

-- Define the points A, B, and P with the intersection conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨5 / 3, 4 / 3⟩

-- Point P on the hyperbola and between A and B
structure PointP (x_3 y_3 : ℝ) where
  on_hyperbola : hyperbola x_3 y_3
  x_bounds : A.x < x_3 ∧ x_3 < B.x
  y_bounds : A.y < y_3 ∧ y_3 < B.y

-- Prove the maximum area of triangle PAB
theorem max_area_triangle_PAB (x_3 y_3 : ℝ) (hP : PointP x_3 y_3) :
  ∃ area, area = (2 - real.sqrt 3) / 3 :=
sorry  -- Placeholder for proof

end max_area_triangle_PAB_l160_160667


namespace max_mondays_in_first_51_days_l160_160519

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l160_160519


namespace coefficient_x4_in_expansion_l160_160466

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem coefficient_x4_in_expansion : 
  ∀ x : ℝ, coeff (x : ℝ)^4 ((1 - x^2) ^ 10) = 45 :=
by {
  intros x,
  -- Coefficients are calculated using binomial theorem
  have h : coeff (x^4) ((1 - x^2)^10) = binom 10 2 * ((-x^2)^2),
  -- Translate steps directly to the binomial coefficient and power of x
  sorry
}

end coefficient_x4_in_expansion_l160_160466


namespace four_digit_multiples_of_7_count_l160_160756

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160756


namespace factorial_division_l160_160958

theorem factorial_division : (15.factorial) / ((6.factorial) * (9.factorial)) = 5005 := by
  sorry

end factorial_division_l160_160958


namespace count_four_digit_multiples_of_7_l160_160770

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160770


namespace books_more_than_action_figures_l160_160019

theorem books_more_than_action_figures :
  ∀ (initial_books initial_figures added_figures : ℕ),
    initial_books = 7 → initial_figures = 3 → added_figures = 2 →
    initial_books - (initial_figures + added_figures) = 2 :=
by
  intros initial_books initial_figures added_figures books_eq figures_eq added_eq
  rw [books_eq, figures_eq, added_eq]
  sorry

end books_more_than_action_figures_l160_160019


namespace smallest_repeating_block_of_8_13_l160_160343

-- Define the repeating block condition
def repeating_block_length (n d : ℕ) : ℕ :=
if h : d ≠ 0 then 
  let dec := (n : ℚ) / d in
  let digits := decimal_period dec in -- hypothetical function for capturing period
  digits.length
else 
  0

-- Claim that the repeating block length of 8/13 is 6
theorem smallest_repeating_block_of_8_13 :
  repeating_block_length 8 13 = 6 :=
by
exactly_digits_period_is_proved_here sorry

end smallest_repeating_block_of_8_13_l160_160343


namespace problem_solution_l160_160258

def parametric_x : ℝ → ℝ := λ α, (√3) * Real.cos α
def parametric_y : ℝ → ℝ := λ α, Real.sin α

def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + (Real.pi / 4)) = 2 * √2

theorem problem_solution :
  (∀ α, 3 * (parametric_x α)^2 + (parametric_y α)^2 = 3) ∧
  (∀ ρ θ, polar_equation ρ θ → (ρ * Real.cos θ) + (ρ * Real.sin θ) = 4) ∧
  ∃ P Q : ℝ × ℝ,
    (P.1 * P.1) / 3 + P.2 * P.2 = 1 ∧
    Q.1 + Q.2 = 4 ∧
    ∀ P' Q' : ℝ × ℝ,
      (P'.1 * P'.1) / 3 + P'.2 * P'.2 = 1 →
      Q'.1 + Q'.2 = 4 →
      ∥P - Q∥ ≤ ∥P' - Q'∥ ∧
      ∥P - Q∥ = √2 ∧
      P = (3 / 2, 1 / 2) :=
by
  sorry

end problem_solution_l160_160258


namespace max_withdrawal_l160_160884

theorem max_withdrawal (initial_balance : ℕ) (withdraw : ℕ) (deposit : ℕ) : ℕ :=
  let max_possible := initial_balance - initial_balance % 6 in
  if 500 = initial_balance ∧ withdraw = 300 ∧ deposit = 198 then
    max_possible
  else 
    sorry

#check max_withdrawal 500 300 198 = 498

end max_withdrawal_l160_160884


namespace isosceles_triangles_count_l160_160371

-- Definitions
variable (n : ℕ) (k : ℕ)

-- Helper function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Main theorem: the number of isosceles triangles with vertices of the same color
theorem isosceles_triangles_count (n k : ℕ) : 
  let total_vertices := 6 * n + 1 in
  let a_0 := 3 * n * total_vertices in
  let a_1 := a_0 - 9 * n in
  let a_k := a_0 - (binomial k 1 * (a_0 - a_1)) + 3 * (binomial k 2) in
  a_k + b_k = 3 * n * total_vertices - 9 * n * k + 3 * k * (k - 1) / 2 :=
sorry -- Proof is omitted

end isosceles_triangles_count_l160_160371


namespace water_evaporation_percentage_l160_160570

-- Define the given conditions as constants
def original_water : ℝ := 12
def total_days : ℕ := 22
def sunny_days : ℕ := 6
def cloudy_days : ℕ := 16
def evap_sunny : ℝ := 0.05
def evap_cloudy : ℝ := 0.02
def added_water : ℝ := 0.03

-- Number of times water is added back
def add_days : ℕ := total_days / 3

-- Calculate the total evaporation and the water added back
def evaporation_sunny : ℝ := sunny_days * evap_sunny
def evaporation_cloudy : ℝ := cloudy_days * evap_cloudy
def total_evaporation : ℝ := evaporation_sunny + evaporation_cloudy
def water_added_back : ℝ := add_days * added_water

-- Calculating the net amount of water evaporated
def net_evaporation : ℝ := total_evaporation - water_added_back
def evaporation_percent : ℝ := (net_evaporation / original_water) * 100

theorem water_evaporation_percentage :
  evaporation_percent ≈ 3.42 := by -- using "≈" to denote approximate equality
  sorry

end water_evaporation_percentage_l160_160570


namespace ratio_of_a_over_5_to_b_over_4_l160_160547

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a/5) / (b/4) = 1 :=
sorry

end ratio_of_a_over_5_to_b_over_4_l160_160547


namespace optimal_strategy_l160_160066

axiom p : ℝ
axiom h_p_pos : 0 < p
axiom h_p_lt_one : p < 1

def prob_2_couriers := 1 - p^2
def prob_3_couriers := 1 - p^2 * (2 - p)
def prob_4_couriers := 1 - p^3 * (4 - 3 * p)

theorem optimal_strategy :
  (0 < p ∧ p < 1/3 → prob_4_couriers > prob_2_couriers ∧ prob_4_couriers > prob_3_couriers) ∧
  (1/3 ≤ p ∧ p < 1 → prob_2_couriers > prob_3_couriers ∧ prob_2_couriers > prob_4_couriers) :=
sorry

end optimal_strategy_l160_160066


namespace compare_sums_l160_160158

noncomputable def S1 := ∑ i in (Finset.range (75 - 25 + 1)).map (λ x, x + 25), (1 : ℝ) / i
noncomputable def S2 := 2 * (∑ j in (Finset.range (125 - 75 + 1)).map (λ x, x + 75), (1 : ℝ) / j)

theorem compare_sums : S1 > S2 := 
by
  sorry

end compare_sums_l160_160158


namespace expression_equality_l160_160030

theorem expression_equality (a b : ℤ) (R S : ℤ) (hR : R = 4^a) (hS : S = 5^b) : 
  20^(a * b) = R^b * S^a :=
by
  sorry

end expression_equality_l160_160030


namespace debit_card_more_advantageous_l160_160458

theorem debit_card_more_advantageous (N : ℕ) (cost_of_tickets : ℝ) (annual_interest_rate : ℝ) (cashback_credit_card : ℝ) (cashback_debit_card : ℝ) (days_per_year : ℝ) (days_per_month : ℝ) :
  cost_of_tickets = 12000 ∧
  annual_interest_rate = 0.06 ∧
  cashback_credit_card = 0.01 ∧
  cashback_debit_card = 0.02 ∧
  days_per_year = 365 ∧
  days_per_month = 30 →
  N ≤ 6 :=
begin
  sorry
end

end debit_card_more_advantageous_l160_160458


namespace coupon_probability_l160_160499

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l160_160499


namespace coupon_probability_l160_160496

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l160_160496


namespace least_number_of_bulbs_l160_160076

theorem least_number_of_bulbs (tulip_packs daffodil_packs : ℕ) (tulip_pack_size daffodil_pack_size : ℕ) 
(h1 : tulip_pack_size = 15) (h2 : daffodil_pack_size = 16) :
  ∃ n : ℕ, n * tulip_pack_size = 240 ∧ n * daffodil_pack_size = 240 :=
by {
  use 16, -- Pack count for tulips (16 packs of 15 bulbs = 240 bulbs)
  use 15, -- Pack count for daffodils (15 packs of 16 bulbs = 240 bulbs)
  sorry
}

end least_number_of_bulbs_l160_160076


namespace prob_5_points_is_0_l160_160557

-- Define the scores Xiao Bai can achieve.
def scores := {1, 2, 3, 4, 5}

-- Define the random variable ξ representing Xiao Bai's score.
def ξ : ℕ → ℝ
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | _ => 0 -- This case is never hit under our conditions, used to complete the function definition.

-- Define the expectation E(ξ).
def E_ξ (p : ℕ → ℝ) : ℝ :=
  (1 * (p 1) + 2 * (p 2) + 3 * (p 3) + 4 * (p 4) + 5 * (p 5))

-- Given condition E(ξ) = 4.2.
axiom expected_value : E_ξ (λ n, if n = 5 then 0.2 else 0.8 / 4) = 4.2

-- Prove that the probability that Xiao Bai scores 5 points is 0.2.
theorem prob_5_points_is_0.2 : (λ n, if n = 5 then 0.2 else 0.8 / 4) 5 = 0.2 :=
sorry

end prob_5_points_is_0_l160_160557


namespace rachel_mark_meeting_time_l160_160072

-- Constants based on the problem conditions
def rateRachel : ℝ := 10 -- Rachel's biking speed in miles per hour.
def rateMark : ℝ := 14 -- Mark's biking speed in miles per hour.
def routeDistance : ℝ := 78 -- Total distance between Calistoga and St. Helena in miles.
def timeMarkStart : ℝ := 0.5 -- Mark starts half an hour (0.5 hours) after Rachel.

-- Time when Rachel and Mark meet
def meetingTime : ℝ := 7 + (85 / 24) -- Rachel leaves at 7 AM, and they meet 85/24 hours later.

-- Theorem to show their meeting time
theorem rachel_mark_meeting_time : meetingTime = 10.5 := 
by
  -- Proof would go here.
  sorry

end rachel_mark_meeting_time_l160_160072


namespace claire_hours_cleaning_l160_160611

-- Definitions of given conditions
def total_hours_in_day : ℕ := 24
def hours_sleeping : ℕ := 8
def hours_cooking : ℕ := 2
def hours_crafting : ℕ := 5
def total_working_hours : ℕ := total_hours_in_day - hours_sleeping

-- Definition of the question
def hours_cleaning := total_working_hours - (hours_cooking + hours_crafting + hours_crafting)

-- The proof goal
theorem claire_hours_cleaning : hours_cleaning = 4 := by
  sorry

end claire_hours_cleaning_l160_160611


namespace rhombus_side_and_inscribed_radius_l160_160203

theorem rhombus_side_and_inscribed_radius
  (K : Point)
  (AC : Line)
  (AB : Line)
  (BC : Line)
  (M : Line)
  (N : Line)
  (ABC : Triangle)
  (r_ABC : ℝ) 
  (h : height of rhombus ABDC)
  (s : side length of rhombus ABDC)
  (r_rhombus: radius circle inscribed rhombus ABDC) 
  (condition1 : K ∈ AC)
  (condition2 : distance K AB = 8)
  (condition3 : distance K BC = 2)
  (condition4 : r_ABC = 3) :
  s = (9/2) * sqrt(5) ∧ r_rhombus = 5 :=
by
  sorry

end rhombus_side_and_inscribed_radius_l160_160203


namespace number_of_four_digit_multiples_of_7_l160_160694

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160694


namespace votes_for_winner_is_744_l160_160986

variable (V : ℝ) -- Total number of votes cast

-- Conditions
axiom two_candidates : True
axiom winner_received_62_percent : True
axiom winner_won_by_288_votes : 0.62 * V - 0.38 * V = 288

-- Theorem to prove
theorem votes_for_winner_is_744 :
  0.62 * V = 744 :=
by
  sorry

end votes_for_winner_is_744_l160_160986


namespace remaining_work_hours_l160_160345

theorem remaining_work_hours (initial_hours_per_week initial_weeks total_earnings first_weeks first_week_hours : ℝ) 
  (hourly_wage remaining_weeks remaining_earnings total_hours_required : ℝ) : 
  15 = initial_hours_per_week →
  15 = initial_weeks →
  4500 = total_earnings →
  3 = first_weeks →
  5 = first_week_hours →
  hourly_wage = total_earnings / (initial_hours_per_week * initial_weeks) →
  remaining_earnings = total_earnings - (first_week_hours * hourly_wage * first_weeks) →
  remaining_weeks = initial_weeks - first_weeks →
  total_hours_required = remaining_earnings / (hourly_wage * remaining_weeks) →
  total_hours_required = 17.5 :=
by
  intros
  sorry

end remaining_work_hours_l160_160345


namespace dot_product_zero_l160_160837

-- Definition of non-zero vectors and given condition
variables {V : Type*} [InnerProductSpace ℝ V] (a b : V)
-- Non-zero vector constraint
noncomputable def non_zero_vectors (a b : V) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0)

-- Given condition |a + b| = |a - b|
noncomputable def equal_magnitudes (a b : V) : Prop :=
  ∥a + b∥ = ∥a - b∥

-- Proof goal: a ⋅ b = 0
theorem dot_product_zero (h1 : non_zero_vectors a b) (h2 : equal_magnitudes a b) : dot_product a b = 0 :=
sorry

end dot_product_zero_l160_160837


namespace four_digit_multiples_of_7_l160_160749

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160749


namespace KM_angle_bisector_of_BKC_l160_160296

-- Definitions based on given conditions
variables (A B C D O K L M : Point) (ω : Circle)
variables [IsParallelogram A B C D]
variables [PerpendicularFrom B O AD]
variables [CircleCenteredAt O PassingThrough ω A B]
variables [IntersectExtension ω AD K]
variables [IntersectSegment BK CD L]
variables [IntersectRay OL ω M]

-- Statement of the theorem to be proved
theorem KM_angle_bisector_of_BKC :
  is_angle_bisector KM (∠ BKC) :=
sorry

end KM_angle_bisector_of_BKC_l160_160296


namespace count_integers_satisfying_inequality_l160_160772

theorem count_integers_satisfying_inequality :
  {m : ℤ | m ≠ 0 ∧ (1 / |m|) ≥ (1 / 10)}.toFinset.card = 20 := by
  sorry

end count_integers_satisfying_inequality_l160_160772


namespace tim_total_trip_time_l160_160138

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l160_160138


namespace lambda_sum_l160_160816

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables {A B C D E : V} {a b : V} {lambda_1 lambda_2 : ℝ}

-- Defining the given conditions
def in_triangle_ABC : Prop :=
E = (A + B) / 2 ∧ A - D = 4 * (C - D) ∧ a = B - A ∧ b = C - B

-- The condition on DE
def DE_condition : Prop :=
lambda_1 • a + lambda_2 • b = D - E

-- The statement to prove
theorem lambda_sum (h1 : in_triangle_ABC) (h2 : DE_condition) : lambda_1 + lambda_2 = -3 / 10 := by
  sorry

end lambda_sum_l160_160816


namespace John_won_by_3_minutes_l160_160821

-- Conditions
def John's_speed : ℝ := 15 -- in mph
def race_distance : ℝ := 5 -- in miles
def next_fastest_time : ℝ := 23 -- in minutes

-- Calculation of John's time in minutes
def John's_time_in_minutes := (race_distance / John's_speed) * 60

-- Theorem statement
theorem John_won_by_3_minutes : next_fastest_time - John's_time_in_minutes = 3 := 
by
  sorry

end John_won_by_3_minutes_l160_160821


namespace count_T_diff_S_l160_160779

-- Define a function to check if a digit is in a given number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ i, i < 3 ∧ (n / 10^i) % 10 = d

-- Define a function to check if a three-digit number is valid
def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the set T' of three digit numbers that do not contain a 6
def T_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6

-- Define the set S' of three digit numbers that neither contain a 2 nor a 6
def S_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6 ∧ ¬ contains_digit n 2

-- Define the set of numbers we are interested in, has 2 but not 6
def T_diff_S : {n // is_valid_three_digit n} → Prop := 
  λ n, contains_digit n 2 ∧ ¬ contains_digit n 6

-- Statement to prove
theorem count_T_diff_S : ∃ n, n = 200 ∧ (∀ (x : {n // is_valid_three_digit n}), T_diff_S x) :=
sorry

end count_T_diff_S_l160_160779


namespace num_12_digit_with_consecutive_ones_l160_160342

theorem num_12_digit_with_consecutive_ones :
  let total := 3^12
  let F12 := 985
  total - F12 = 530456 :=
by
  let total := 3^12
  let F12 := 985
  have h : total - F12 = 530456
  sorry
  exact h

end num_12_digit_with_consecutive_ones_l160_160342


namespace mean_value_theorem_example_l160_160439

noncomputable def f (x : ℝ) : ℝ := x^2 + 3

theorem mean_value_theorem_example : ∃ c ∈ set.Ioo (-1 : ℝ) (2 : ℝ), deriv f c = 1 :=
by
  -- Prove that the conditions of MVT are satisfied for the given function on the interval [-1, 2]
  have f_cont : continuous_on f (set.Icc (-1) 2),
  { exact continuous_on.polynomial continuous_on_const (by norm_num) },
  have f_diff : differentiable_on ℝ f (set.Ioo (-1) 2),
  { exact differentiable_on_polynomial },
  have mvt := exists_deriv_eq_slope f f_cont f_diff (by norm_num) (by norm_num),
  dsimp only [f, deriv] at mvt,
  simp only [sub_neg_eq_add, sub_self, sub_eq_add_neg, add_left_neg, neg_add_self] at mvt,
  exact mvt sorry  -- Deriving the specific value of c

end mean_value_theorem_example_l160_160439


namespace johns_earnings_increase_l160_160394

-- Define the initial and new earnings
def initialEarnings : ℝ := 60
def newEarnings : ℝ := 70

-- Define the formula for percentage increase
def percentageIncrease (newEarnings initialEarnings : ℝ) : ℝ :=
  ((newEarnings - initialEarnings) / initialEarnings) * 100

-- Define the theorem to prove the percentage increase
theorem johns_earnings_increase :
  percentageIncrease newEarnings initialEarnings = 16.67 :=
by
  sorry

end johns_earnings_increase_l160_160394


namespace find_m_range_l160_160674

def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) * x^2

theorem find_m_range (m : ℝ) 
  (h1 : f (Real.log 3 m) - f (Real.log (1 / 3) m) ≤ 2 * f 1) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end find_m_range_l160_160674


namespace isosceles_triangle_count_l160_160199

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def is_isosceles (p1 p2 p3 : Point) : Prop :=
  let d1 := distance p1 p2
  let d2 := distance p2 p3
  let d3 := distance p1 p3
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

def vertices := [Point.mk 0 0, Point.mk 2 4, Point.mk 5 4, Point.mk 7 0, Point.mk 3 1]

noncomputable def count_isosceles_triangles : ℕ :=
  let triangles := List.product (List.product vertices vertices) vertices
  triangles.count (λ ⟨⟨p1, p2⟩, p3⟩, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ is_isosceles p1 p2 p3)

theorem isosceles_triangle_count :
  count_isosceles_triangles = 1 := 
sorry

end isosceles_triangle_count_l160_160199


namespace certain_number_N_l160_160267

theorem certain_number_N (G N : ℕ) (hG : G = 127)
  (h₁ : ∃ k : ℕ, N = G * k + 10)
  (h₂ : ∃ m : ℕ, 2045 = G * m + 13) :
  N = 2042 :=
sorry

end certain_number_N_l160_160267


namespace max_experiments_needed_l160_160811

theorem max_experiments_needed (n : ℕ) (hn : n = 33) : 
  ⌈real.logb 2 n⌉ = 6 :=
by
  rw hn
  sorry

end max_experiments_needed_l160_160811


namespace grid_is_valid_l160_160437

noncomputable def sum (ns : List ℕ) : ℕ := List.sum ns

def central_columns_sum (g : List (List ℕ)) : Prop :=
  sum ([10, 2, 6, 8]) = 26 ∧ sum ([6, 1, 9, 15]) = 26

def central_rows_sum (g : List (List ℕ)) : Prop :=
  sum ([5, 6, 0, 15]) = 26 ∧ sum ([11, 1, 9, 5]) = 26

def roses_sum (g : List (List ℕ)) : Prop :=
  sum ([8, 2, 0, 16]) = 26 

def trilists_sum (g : List (List ℕ)) : Prop :=
  sum ([5, 4, 7, 10]) = 26 

def thistle_sum (g : List (List ℕ)) : Prop :=
  sum ([6, 9, 11]) = 26 

def grid_sum_properties (g : List (List ℕ)) : Prop :=
  central_columns_sum g ∧ 
  central_rows_sum g ∧ 
  roses_sum g ∧ 
  trilists_sum g ∧ 
  thistle_sum g

def grid : List (List ℕ) := [
  [0,   0,  8, 0,  0],
  [0,  10,  2, 0,  0],
  [5,  0,  6,  15,  0],
  [0,  0,  11,   1,  9],
  [0,  4,  7, 0,  0]
]

theorem grid_is_valid : grid_sum_properties grid := 
by {
    unfold grid_sum_properties,
    unfold central_columns_sum,
    unfold central_rows_sum,
    unfold roses_sum,
    unfold trilists_sum,
    unfold thistle_sum,
    sorry
}

end grid_is_valid_l160_160437


namespace sin_2x_from_tan_pi_minus_x_l160_160284

theorem sin_2x_from_tan_pi_minus_x (x : ℝ) (h : Real.tan (Real.pi - x) = 3) : Real.sin (2 * x) = -3 / 5 := by
  sorry

end sin_2x_from_tan_pi_minus_x_l160_160284


namespace four_digit_multiples_of_7_count_l160_160758

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160758


namespace domain_f_abs_l160_160319

-- Given problem conditions
def is_domain_of_f (d : set ℝ) : Prop :=
  d = set.Icc (-1 : ℝ) 2

-- Proof that the domain of f(|x|) is (-2, 2)
theorem domain_f_abs (d : set ℝ) :
  is_domain_of_f d → ∀ x, (x ∈ set.Ioo (-2 : ℝ) 2) ↔ (|x| ∈ d) :=
by
  intro hdomain
  sorry

end domain_f_abs_l160_160319


namespace area_triangle_AEL_l160_160383

open Triangle

theorem area_triangle_AEL (a b c : Point)
  (h_abc_right : ∠ABC = 90)
  (l : Point)
  (h_bl_bisects : angle_bisector B L (A, C))
  (m : Point)
  (h_cm_median : median C (A, B))
  (d : Point)
  (h_bl_cm_intersect : intersects BL CM d)
  (e : Point)
  (h_ad_intersects_bc : intersects AD BC e)
  (x : ℝ)
  (h_el_length : EL = x) :
  area (triangle A E L) = (x^2) / 2 := 
sorry

end area_triangle_AEL_l160_160383


namespace probability_abby_bridget_adjacent_l160_160213

/-
An important point is that though we're not implementing the full combinatorial setup
using Lean's probability and combinatorics libraries (because we are not asked for the proof),
we can still state the problem in terms of what a formal proof would look like.
-/

def totalArrangements : Nat := fact 5

def adjacentArrangements : Nat := 2 * fact 4

theorem probability_abby_bridget_adjacent :
  (adjacentArrangements : ℚ) / (totalArrangements : ℚ) = 2 / 5 :=
sorry

end probability_abby_bridget_adjacent_l160_160213


namespace binomial_expansion_second_third_terms_zero_l160_160528

theorem binomial_expansion_second_third_terms_zero (a b : ℝ) (k n : ℕ) 
    (hn : n ≥ 2) (hab : a ≠ 0) (hb : b ≠ 0) (ha : a = k * b) (hk : k > 0) :
    (a - b)^n.built_expansion().term(1).sum() + (a - b)^n.built_expansion().term(2).sum() = 0 ↔ n = 2 * k + 1 :=
sorry

end binomial_expansion_second_third_terms_zero_l160_160528


namespace largest_prime_divisor_of_36_sq_plus_49_sq_l160_160634

theorem largest_prime_divisor_of_36_sq_plus_49_sq : ∃ (p : ℕ), p = 36^2 + 49^2 ∧ Prime p := 
by
  let n := 36^2 + 49^2
  have h : n = 3697 := by norm_num
  use 3697
  split
  . exact h
  . exact sorry

end largest_prime_divisor_of_36_sq_plus_49_sq_l160_160634


namespace question_correctness_l160_160965

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l160_160965


namespace eight_neg_x_eq_one_fourth_l160_160356

noncomputable def x : ℝ := 2 / 3

theorem eight_neg_x_eq_one_fourth (x : ℝ) (h : 8^(3*x) = 64) : 8^(-x) = 1 / 4 := by
  by_cases h : 8^(3*x) = 64
  sorry

end eight_neg_x_eq_one_fourth_l160_160356


namespace count_four_digit_multiples_of_7_l160_160766

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160766


namespace range_of_a_l160_160277

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  log (x^2 - x + a) / log 2

theorem range_of_a (a : ℝ) : (∀ x, 2 ≤ x → f x a > 0) ↔ a > -1 := by
  sorry

end range_of_a_l160_160277


namespace pascal_diagonal_is_fibonacci_l160_160407

def pascal_diagonal_sum (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), nat.choose n k

def fibonacci : ℕ → ℕ
| 0        := 0
| 1        := 1
| (n + 2)  := fibonacci (n + 1) + fibonacci n

theorem pascal_diagonal_is_fibonacci (n : ℕ) :
  pascal_diagonal_sum n = fibonacci n :=
sorry

end pascal_diagonal_is_fibonacci_l160_160407


namespace minor_characters_count_l160_160819

theorem minor_characters_count : 
  ∀ (\#MC PMC : ℕ) (TPE : ℕ),
    (\#MC = 5) →
    (PMC = 15000) →
    (TPE = 285000) →
    (3 * PMC * \#MC + M * PMC = TPE) →
    M = 4 :=
by
  intros \#MC PMC TPE hMC hPMC hTPE hEq
  sorry

end minor_characters_count_l160_160819


namespace count_four_digit_multiples_of_7_l160_160765

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160765


namespace hyperbola_perimeter_l160_160116

-- Lean 4 statement
theorem hyperbola_perimeter (a b m : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), (x,y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (line_through_F1 : ∀ (x y : ℝ), x = F1.1)
  (A_B_on_hyperbola : (A.1^2/a^2 - A.2^2/b^2 = 1) ∧ (B.1^2/a^2 - B.2^2/b^2 = 1))
  (dist_AB : dist A B = m)
  (dist_relations : dist A F2 + dist B F2 - (dist A F1 + dist B F1) = 4 * a) : 
  dist A F2 + dist B F2 + dist A B = 4 * a + 2 * m :=
sorry

end hyperbola_perimeter_l160_160116


namespace four_digit_multiples_of_7_l160_160707

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160707


namespace actual_distance_between_cities_l160_160875

theorem actual_distance_between_cities :
  ∀ (map_distance a_to_b real_distance a_to_b map_distance_cities_real : ℝ),
  map_distance a_to_b = 3 →
  real_distance a_to_b = 33 →
  map_distance_cities_real = 19 →
  real_distance a_to_b / map_distance a_to_b * map_distance_cities_real = 209 :=
by
  intros map_distance a_to_b real_distance a_to_b map_distance_cities_real
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end actual_distance_between_cities_l160_160875


namespace altered_solution_contains_60_liters_of_detergent_l160_160550

-- Definitions corresponding to the conditions
def initial_ratio_bleach_to_detergent_to_water : ℚ := 2 / 40 / 100
def initial_ratio_bleach_to_detergent : ℚ := 1 / 20
def initial_ratio_detergent_to_water : ℚ := 1 / 5

def altered_ratio_bleach_to_detergent : ℚ := 3 / 20
def altered_ratio_detergent_to_water : ℚ := 1 / 5

def water_in_altered_solution : ℚ := 300

-- We need to find the amount of detergent in the altered solution
def amount_of_detergent_in_altered_solution : ℚ := 20

-- The proportion and the final amount calculation
theorem altered_solution_contains_60_liters_of_detergent :
  (300 / 100) * (20) = 60 :=
by
  sorry

end altered_solution_contains_60_liters_of_detergent_l160_160550


namespace proof_problem_l160_160411

-- Conditions: p and q are solutions to the quadratic equation 3x^2 - 5x - 8 = 0
def is_solution (p q : ℝ) : Prop := (3 * p^2 - 5 * p - 8 = 0) ∧ (3 * q^2 - 5 * q - 8 = 0)

-- Question: Compute the value of (3 * p^2 - 3 * q^2) / (p - q) given the conditions
theorem proof_problem (p q : ℝ) (h : is_solution p q) :
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := sorry

end proof_problem_l160_160411


namespace four_digit_multiples_of_7_l160_160716

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160716


namespace range_magnitude_sum_l160_160306

-- Definitions of vectors and conditions
variables {α : Type*} [inner_product_space ℝ α]

-- Unit vectors a, b, c
variables (a b c : α)
-- Orthogonality condition of a and b
variable (orth_ab : inner_product a b = 0)
-- Unit magnitude condition of a, b, c
variable (unit_a : ∥a∥ = 1)
variable (unit_b : ∥b∥ = 1)
variable (unit_c : ∥c∥ = 1)

-- Prove the range of |a + 2b + c|
theorem range_magnitude_sum :
  ∃ (θ : ℝ), (∥a + 2 • b + c∥) ∈ set.Icc (real.sqrt(5) - 1) (real.sqrt(5) + 1) :=
sorry

end range_magnitude_sum_l160_160306


namespace solve_symbols_values_l160_160170

def square_value : Nat := 423 / 47

def boxminus_and_boxtimes_relation (boxminus boxtimes : Nat) : Prop :=
  1448 = 282 * boxminus + 9 * boxtimes

def boxtimes_value : Nat := 38 / 9

def boxplus_value : Nat := 846 / 423

theorem solve_symbols_values :
  ∃ (square boxplus boxtimes boxminus : Nat),
    square = 9 ∧
    boxplus = 2 ∧
    boxtimes = 8 ∧
    boxminus = 5 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + 9 * boxtimes ∧
    9 * boxtimes = 38 ∧
    423 * boxplus / 3 = 282 := by
  sorry

end solve_symbols_values_l160_160170


namespace multiple_choice_answer_choices_l160_160807

theorem multiple_choice_answer_choices (n : ℕ)
  (h1 : ∃ d > 1, is_divisible d n)
  : (3 * 3) * 6 * n^3 = 384 := by
sorry

end multiple_choice_answer_choices_l160_160807


namespace parallelogram_lambda_l160_160378

variable (V : Type) [AddCommGroup V] [VectorSpace ℝ V]
variables (A B C D O : V)
variables (lambda : ℝ)

-- Conditions
def is_parallelogram (A B C D : V) : Prop :=
  A + C = B + D

def diagonals_intersect_at_O (A C O : V) : Prop :=
  A + C = 2 • O

def vector_equation (A B D O : V) (λ : ℝ) : Prop :=
  A + B + D = λ • O

-- Theorem statement
theorem parallelogram_lambda (h₁ : is_parallelogram A B C D)
                            (h₂ : diagonals_intersect_at_O A C O)
                            (h₃ : vector_equation A B D O lambda) : 
                            λ = 2 := 
by 
  sorry

end parallelogram_lambda_l160_160378


namespace divisors_of_product_primes_l160_160026

open Nat Finset

-- Defining the conditions and translating them into Lean.
def is_even (n : ℕ) : Prop := n % 2 = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem divisors_of_product_primes (k : ℕ) (N a b : ℕ) (primes : Finset ℕ)
  (prime_prod : N = primes.prod id)
  (Hk : is_even k)
  (Ha : a ≤ N) (Hb : b ≤ N)
  (Hp : primes.card = k) 
  (H : ∀ p ∈ primes, Nat.prime p) :
  let S_1 := {d ∈ (Finset.range (N+1)) | d ∣ N ∧ a ≤ d ∧ d ≤ b ∧ is_even (Finset.card (factors d primes))}
  let S_2 := {d ∈ (Finset.range (N+1)) | d ∣ N ∧ a ≤ d ∧ d ≤ b ∧ is_odd (Finset.card (factors d primes))} in
  (S_1.card : ℤ) - (S_2.card : ℤ) ≤ (Nat.choose k (k/2)) :=
sorry

end divisors_of_product_primes_l160_160026


namespace total_gifts_l160_160057

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l160_160057


namespace positive_b_3b_sq_l160_160074

variable (a b c : ℝ)

theorem positive_b_3b_sq (h1 : 0 < a ∧ a < 0.5) (h2 : -0.5 < b ∧ b < 0) (h3 : 1 < c ∧ c < 3) : b + 3 * b^2 > 0 :=
sorry

end positive_b_3b_sq_l160_160074


namespace max_base_angle_is_7_l160_160808

-- Define the conditions and the problem statement
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isosceles_triangle (x : ℕ) : Prop :=
  is_prime x ∧ ∃ y : ℕ, 2 * x + y = 180 ∧ is_prime y

theorem max_base_angle_is_7 :
  ∃ (x : ℕ), isosceles_triangle x ∧ x = 7 :=
by
  sorry

end max_base_angle_is_7_l160_160808


namespace graph_transformation_l160_160679

def transformation_sequence (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  g ((2 - x) / 3)

def reflect (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  g (-x)

def horizontal_stretch (g : ℝ → ℝ) (factor : ℝ) (x : ℝ) : ℝ :=
  g (x / factor)

def horizontal_shift (g : ℝ → ℝ) (shift : ℝ) (x : ℝ) : ℝ :=
  g (x - shift)

theorem graph_transformation (g : ℝ → ℝ) :
  transformation_sequence g = (λ x, reflect (λ y, horizontal_stretch (horizontal_shift g 2) 3 y) x) :=
by
  sorry

end graph_transformation_l160_160679


namespace calculate_difference_l160_160575

theorem calculate_difference :
  let students := 120
  let teachers := 6
  let class_sizes := [40, 30, 30, 10, 5, 5]
  let t := (class_sizes.sum.toReal) / teachers
  let s := (List.sum (class_sizes.map (λ n => (n.toReal * n / students)))) 
  t - s = -9.58 :=
by
  let students := 120
  let teachers := 6
  let class_sizes := [40, 30, 30, 10, 5, 5]
  let t := (class_sizes.sum.toReal) / teachers
  let s := (List.sum (class_sizes.map (λ n => (n.toReal * n / students))))
  show t - s = -9.58
  sorry

end calculate_difference_l160_160575


namespace trees_left_after_typhoon_l160_160586

-- Define the initial count of trees and the number of trees that died
def initial_trees := 150
def trees_died := 24

-- Define the expected number of trees left
def expected_trees_left := 126

-- The statement to be proven: after trees died, the number of trees left is as expected
theorem trees_left_after_typhoon : (initial_trees - trees_died) = expected_trees_left := by
  sorry

end trees_left_after_typhoon_l160_160586


namespace geometric_sum_creates_complete_residue_system_mod_p_l160_160555

theorem geometric_sum_creates_complete_residue_system_mod_p 
  (p : ℕ) (r k : ℕ) 
  (h_prime_p : Nat.Prime p)
  (h_odd_prime_r : Nat.Prime r) 
  (h_odd_r : r % 2 = 1)
  (h_p_value : p = 2 * r * k + 1)
  (a : Fin r → ℕ) 
  (h_geometric_seq : ∃ q : ℕ, ∀ i : Fin r, a (i.succ) = a 0 * q ^ i.val) 
  (h_p_sum : p = ∑ i in (Finset.range r), a i)
  : ∃ b : Fin (2 * k) → ℕ, let A := (Finset.range r).image a,
    let B := (Finset.range (2*k)).image b,
    (A.product B).image (λ (x : ℕ × ℕ), x.1 * x.2 % p) = (Finset.range (p - 1)).image (λ (x : ℕ), x + 1)  := 
by 
  sorry

end geometric_sum_creates_complete_residue_system_mod_p_l160_160555


namespace problem_conclusions_correct_l160_160402

def closest_integer (x : ℝ) : ℕ := 
  if x - x.floor < 0.5 then x.floor else x.ceil

def a_n (n : ℕ) [hn : Fact (n > 0)] : ℝ :=
  1 / closest_integer (Real.sqrt n)

def T_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), a_n k

theorem problem_conclusions_correct (n : ℕ) [hn : Fact (n > 0)] :
    (let k := closest_integer (Real.sqrt n) in 
    Real.sqrt n > k - 0.5 ∧ 
    n ≤ k^2 + k) 
    ∧ T_n 2023 = 88 + 43 / 45 :=
  sorry

end problem_conclusions_correct_l160_160402


namespace gold_to_silver_ratio_is_two_l160_160393

-- Defining condition variables
def silver_weight : ℝ := 1.5
def silver_cost_per_ounce : ℝ := 20
def gold_multiplier : ℝ := 50
def total_spent : ℝ := 3030

-- Defining the cost of silver and gold per the given conditions
def cost_of_silver : ℝ := silver_weight * silver_cost_per_ounce
def cost_per_ounce_gold : ℝ := silver_cost_per_ounce * gold_multiplier
def remaining_spent_on_gold : ℝ := total_spent - cost_of_silver
def gold_weight : ℝ := remaining_spent_on_gold / cost_per_ounce_gold

-- Proving that the ratio of the weights of gold to silver is 2
theorem gold_to_silver_ratio_is_two : gold_weight / silver_weight = 2 :=
  by sorry  -- Proof is to be filled in

end gold_to_silver_ratio_is_two_l160_160393


namespace count_increasing_8digit_no_repeat_more_than_twice_l160_160400

theorem count_increasing_8digit_no_repeat_more_than_twice : 
  let M := (Nat.choose 16 8) - 9 * (Nat.choose 13 5) + (Nat.choose 9 2) * (Nat.choose 10 2) in
  M = 1907 :=
by
  let M := (Nat.choose 16 8) - 9 * (Nat.choose 13 5) + (Nat.choose 9 2) * (Nat.choose 10 2)
  have hM : M = 1907 := sorry
  exact hM

end count_increasing_8digit_no_repeat_more_than_twice_l160_160400


namespace abs_case_inequality_solution_l160_160485

theorem abs_case_inequality_solution (x : ℝ) :
  (|x + 1| + |x - 4| ≥ 7) ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 5) :=
by
  sorry

end abs_case_inequality_solution_l160_160485


namespace fourth_is_20_fewer_than_third_l160_160574

-- Definitions of the number of road signs at each intersection
def first_intersection := 40
def second_intersection := first_intersection + first_intersection / 4
def third_intersection := 2 * second_intersection
def total_signs := 270
def fourth_intersection := total_signs - (first_intersection + second_intersection + third_intersection)

-- Proving the fourth intersection has 20 fewer signs than the third intersection
theorem fourth_is_20_fewer_than_third : third_intersection - fourth_intersection = 20 :=
by
  -- This is a placeholder for the proof
  sorry

end fourth_is_20_fewer_than_third_l160_160574


namespace max_balls_in_cube_l160_160149

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r^3)
noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3
noncomputable def max_balls_fit (v_cube v_ball : ℝ) : ℕ := Int.floor (v_cube / v_ball)

theorem max_balls_in_cube {r s : ℝ} (h_r : r = 3) (h_s : s = 10) :
  max_balls_fit (volume_cube s) (volume r) = 8 :=
by
  sorry

end max_balls_in_cube_l160_160149


namespace least_possible_b_l160_160096

theorem least_possible_b (a b : ℕ) (ha : a.prime) (hb : b.prime) (sum_90 : a + b = 90) (a_greater_b : a > b) : b = 7 :=
by
  sorry

end least_possible_b_l160_160096


namespace sum_g_of_fractions_l160_160842

def g (x : ℝ) : ℝ := 4 / (4^x + 4)

theorem sum_g_of_fractions (f : ℕ → ℝ) (h : ∀ k, f k = g (k / 2001)) :
  (∑ k in Finset.range 2000, f (k + 1)) = 1000 :=
by
  sorry

end sum_g_of_fractions_l160_160842


namespace max_radius_of_ball_l160_160803

theorem max_radius_of_ball
  (a : ℝ)
  (P A B C D : ℝ × ℝ × ℝ)
  (r : ℝ)
  (h_base : distance A B = a ∧ distance B C = a ∧ distance C D = a ∧ distance D A = a)
  (h_PD : distance P D = a)
  (h_PA_PC : distance P A = sqrt 2 * a ∧ distance P C = sqrt 2 * a) :
  r = (1 - (sqrt 2) / 2) * a :=
sorry

end max_radius_of_ball_l160_160803


namespace part1_part2_l160_160857

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1 : {x : ℝ | f x ≤ 4} = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 1} :=
by
  sorry

theorem part2 {a : ℝ} :
  ({x : ℝ | f x ≤ 4} ⊆ {x : ℝ | |x + 3| + |x + a| < x + 6}) ↔ (-4 / 3 < a ∧ a < 2) :=
by
  sorry

end part1_part2_l160_160857


namespace four_digit_multiples_of_7_l160_160702

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160702


namespace vertex_angle_in_isosceles_triangle_l160_160375

-- Definitions for the problem conditions
def is_isosceles_triangle (A B C : Type) [triangle A B C] : Prop := 
  ∃ (a b c : ℝ), isosceles a b c ∧ angle B A C == 30

def height_angle_condition (A B C : Type) [triangle A B C] : Prop := 
  ∃ (h : ℝ), height_forming_angle h == 30

-- Lean 4 statement for the proof
theorem vertex_angle_in_isosceles_triangle (A B C : Type) [triangle A B C] 
  (h : height_angle_condition A B C) (iso : is_isosceles_triangle A B C) : 
  ∃ (θ : ℝ), θ = 60 ∨ θ = 120 :=
by
  sorry

end vertex_angle_in_isosceles_triangle_l160_160375


namespace nice_people_in_crowd_l160_160218

theorem nice_people_in_crowd (Barry Kevin Julie Joe : ℕ)
    (hBarry : Barry = 24)
    (hKevin : Kevin = 20)
    (hJulie : Julie = 80)
    (hJoe : Joe = 50)
    : (Barry + Kevin / 2 + Julie * 3 / 4 + Joe * 10 / 100) = 99 :=
by 
  rw [hBarry, hKevin, hJulie, hJoe]
  norm_num
  sorry

end nice_people_in_crowd_l160_160218


namespace eric_days_waited_l160_160260

def num_chickens := 4
def eggs_per_chicken_per_day := 3
def total_eggs := 36

def eggs_per_day := num_chickens * eggs_per_chicken_per_day
def num_days := total_eggs / eggs_per_day

theorem eric_days_waited : num_days = 3 :=
by
  sorry

end eric_days_waited_l160_160260


namespace pirate_loot_value_l160_160202

def base7_to_base10 (n : ℕ) : ℕ :=
  n.digits 7.reverse.foldl (λ acc d, acc * 7 + d) 0

def total_loot_value : ℕ :=
  base7_to_base10 4516 +
  base7_to_base10 3216 +
  base7_to_base10 654 +
  base7_to_base10 301

theorem pirate_loot_value :
  total_loot_value = 3251 :=
by
  have h1 : base7_to_base10 4516 = 1630 := by sorry
  have h2 : base7_to_base10 3216 = 1140 := by sorry
  have h3 : base7_to_base10 654 = 333 := by sorry
  have h4 : base7_to_base10 301 = 148 := by sorry
  show total_loot_value = 3251
  calc
    total_loot_value = 1630 + 1140 + 333 + 148 := by rw [h1, h2, h3, h4]
                    ... = 3251 := by norm_num

end pirate_loot_value_l160_160202


namespace max_points_in_plane_max_points_in_space_l160_160280

-- Define the context for points being in a plane
def right_triangle_in_plane (points: Finset (ℝ × ℝ)) : Prop :=
  ∀ (a b c : ℝ × ℝ), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → c ≠ a →
    let (ax, ay) := a;
        (bx, by) := b;
        (cx, cy) := c;
        ab := (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
        bc := (cx - bx) * (cx - bx) + (cy - by) * (cy - by);
        ac := (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay)
    in ab + bc = ac ∨ ab + ac = bc ∨ bc + ac = ab

-- Define the context for points being in space
def right_triangle_in_space (points: Finset (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (a b c : ℝ × ℝ × ℝ), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → c ≠ a →
    let (ax, ay, az) := a;
        (bx, by, bz) := b;
        (cx, cy, cz) := c;
        ab := (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az);
        bc := (cx - bx) * (cx - bx) + (cy - by) * (cy - by) + (cz - bz) * (cz - bz);
        ac := (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay) + (cz - az) * (cz - az)
    in ab + bc = ac ∨ ab + ac = bc ∨ bc + ac = ab

theorem max_points_in_plane {points : Finset (ℝ × ℝ)} (h: right_triangle_in_plane points) : points.card ≤ 4 :=
begin
  sorry
end

theorem max_points_in_space {points : Finset (ℝ × ℝ × ℝ)} (h: right_triangle_in_space points) : points.card ≤ 4 :=
begin
  sorry
end

end max_points_in_plane_max_points_in_space_l160_160280


namespace find_8th_result_l160_160907

theorem find_8th_result 
  (S_17 : ℕ := 17 * 24) 
  (S_7 : ℕ := 7 * 18) 
  (S_5_1 : ℕ := 5 * 23) 
  (S_5_2 : ℕ := 5 * 32) : 
  S_17 - S_7 - S_5_1 - S_5_2 = 7 := 
by
  sorry

end find_8th_result_l160_160907


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l160_160636

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l160_160636


namespace cube_to_tetrahedron_l160_160489

noncomputable def is_regular_tetrahedron (A B C D : ℝ × ℝ × ℝ) : Prop :=
  let dist := λ (p q : ℝ × ℝ × ℝ), real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2) in
  dist A B = dist B C ∧ 
  dist B C = dist C D ∧ 
  dist C D = dist D A ∧
  dist A C = dist B D ∧
  dist A C = dist A D

theorem cube_to_tetrahedron
  (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ)
  (a : ℝ)
  (h_cube_edges : dist A B = a ∧ dist A D = a ∧ dist A A1 = a ∧
                  dist B C = a ∧ dist B B1 = a ∧
                  dist C D = a ∧ dist C C1 = a ∧
                  dist D D1 = a ∧ dist A1 B1 = a ∧ dist A1 D1 = a ∧ dist B1 C1 = a ∧ dist C1 D1 = a)
  (T : tetrahedron)
  (h_tetrahedron : T = [A, B1, C, D1]) :
  is_regular_tetrahedron A B1 C D1 ∧
  T.surface_area = 2 * real.sqrt 3 * a^2 ∧
  T.volume = a^3 / 3 := 
sorry

end cube_to_tetrahedron_l160_160489


namespace four_digit_multiples_of_7_l160_160717

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160717


namespace digit_assignment_exists_l160_160110

-- Define a type for the letters
constant Letter : Type

-- Define constants for the digits (0 to 6)
constant digit_A digit_B digit_C digit_D digit_E digit_F digit_G : ℕ

-- Define the condition for the domino arrangement
axiom domino_condition : 
  (digit_A + digit_B + digit_C + digit_D + digit_E + digit_F + digit_G = 24)

theorem digit_assignment_exists : 
  ∃ (A B C D E F G : ℕ),
  (A + B + C + D + E + F + G = 24) ∧
  (A, B, C, D, E, F, G ∈ {0, 1, 2, 3, 4, 5, 6}) ∧
  (A = digit_A) ∧ (B = digit_B) ∧ (C = digit_C) ∧ (D = digit_D) ∧
  (E = digit_E) ∧ (F = digit_F) ∧ (G = digit_G) := 
sorry

end digit_assignment_exists_l160_160110


namespace intersection_M_N_eq_set_l160_160338

universe u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {y | ∃ x, x ∈ M ∧ y = 2 * x + 1}

-- Prove the intersection M ∩ N = {-1, 1}
theorem intersection_M_N_eq_set : M ∩ N = {-1, 1} :=
by
  simp [Set.ext_iff, M, N]
  sorry

end intersection_M_N_eq_set_l160_160338


namespace f_n_at_1_l160_160856

def f (x : ℝ) (h : x > 0) : ℝ := x / (x + 2)

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) := 
  nat.recOn n 
    (fun x => f x (by linarith))
    (fun n f_n_x => fun x => f (f_n_x x) (by linarith))

theorem f_n_at_1 (n : ℕ) (hn : n > 0) : f_n n 1 = 1 / (2^(n + 1) - 1) := 
by sorry

end f_n_at_1_l160_160856


namespace player_A_cannot_win_if_B_plays_correctly_l160_160947

-- Given an even number of cells and the game rules, prove player A cannot win if player B follows the strategy of mirroring A's moves.
theorem player_A_cannot_win_if_B_plays_correctly (n : ℕ) (hn : n % 2 = 0) :
  ∀ (moves : ℕ → Option (String × String)), -- Each move is a pair of cell index and either "O" or "M"
  (∀ k, (k > 0 → moves (k-1) = some ("OMO", "A")) ↔ (k < n → ∃ move, moves k = some move)) →
  (∃ (B_correct_play : ∀ i, moves (i + n/2 % n) = moves i), ¬ ∃ (k < n), moves (k+n) = some ("OMO", "A")) :=
sorry -- proof will be provided here

end player_A_cannot_win_if_B_plays_correctly_l160_160947


namespace distance_to_post_office_l160_160979

theorem distance_to_post_office
  (D : ℝ)
  (travel_rate : ℝ) (walk_rate : ℝ)
  (total_time_hours : ℝ)
  (h1 : travel_rate = 25)
  (h2 : walk_rate = 4)
  (h3 : total_time_hours = 5 + 48 / 60) :
  D = 20 :=
by
  sorry

end distance_to_post_office_l160_160979


namespace count_proper_subsets_l160_160121

def proper_subset_count (s : Set ℕ) : ℕ := 2^s.card - 1

theorem count_proper_subsets {x : ℕ} 
  (h1: ∀ x ∈ Set.univ, -1 ≤ log 10 / - log x ∧ log 10 / - log x < -1 / 2)
  (h2: ∀ x ∈ Set.univ, x ∈ ℕ) :
  proper_subset_count (setOf (λ x, 10 ≤ x ∧ x < 100)) = 2^90 - 1 :=
by
  sorry

end count_proper_subsets_l160_160121


namespace coupon_probability_l160_160494

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l160_160494


namespace vasya_number_exists_l160_160145

open List

-- Define the constraint conditions
def conditions (l : List ℕ) :=
  l.length = 8 ∧
  l.count 1 = 2 ∧
  l.count 2 = 2 ∧
  l.count 3 = 2 ∧
  l.count 4 = 2 ∧
  (l.indexOf 1 < l.indexOfNth 1 1 - 1)
    ∧ (l.indexOfNth 1 1 - l.indexOf 1 = 2) ∧
  (l.indexOf 2 < l.indexOfNth 2 1 - 2)
    ∧ (l.indexOfNth 2 1 - l.indexOf 2 = 3) ∧
  (l.indexOf 3 < l.indexOfNth 3 1 - 3)
    ∧ (l.indexOfNth 3 1 - l.indexOf 3 = 4) ∧
  (l.indexOf 4 < l.indexOfNth 4 1 - 4)
    ∧ (l.indexOfNth 4 1 - l.indexOf 4 = 5)

-- Define the theorem to state the existence of a valid number
theorem vasya_number_exists : ∃ l : List ℕ, conditions l :=
⟨[4, 1, 3, 1, 2, 2, 3, 4], by dec_trivial⟩ -- The answer as an example solution

end vasya_number_exists_l160_160145


namespace percentage_increase_l160_160512

variable (A B C : ℝ)
variable (h1 : A = 0.71 * C)
variable (h2 : A = 0.05 * B)

theorem percentage_increase (A B C : ℝ) (h1 : A = 0.71 * C) (h2 : A = 0.05 * B) : (B - C) / C = 13.2 :=
by
  sorry

end percentage_increase_l160_160512


namespace new_customers_arrived_l160_160584

theorem new_customers_arrived (initial final : ℕ) (h_initial : initial = 3) (h_final : final = 8) : final - initial = 5 := by
  rw [h_initial, h_final]
  norm_num
  sorry

end new_customers_arrived_l160_160584


namespace factorial_division_l160_160956

theorem factorial_division : (15.factorial) / ((6.factorial) * (9.factorial)) = 5005 := by
  sorry

end factorial_division_l160_160956


namespace zero_point_interval_l160_160631

noncomputable def f (x : ℝ) : ℝ := log 10 x + x - 2

theorem zero_point_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 := 
by
  -- We need to prove that the zero point of function f lies in the interval (1, 2).
  sorry

end zero_point_interval_l160_160631


namespace A_B_symmetric_x_axis_l160_160380

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Theorem stating the symmetry relationship between points A and B with respect to the x-axis
theorem A_B_symmetric_x_axis (xA yA xB yB : ℝ) (hA : A = (xA, yA)) (hB : B = (xB, yB)) :
  xA = xB ∧ yA = -yB := by
  sorry

end A_B_symmetric_x_axis_l160_160380


namespace count_valid_n_le_30_l160_160275

def sum_of_integers (n : ℕ) : ℕ := (n * (n + 1)) / 2

definition valid_divisibility (n : ℕ) : Prop :=
  n ≤ 30 ∧ n! % sum_of_integers (n - 1) = 0

theorem count_valid_n_le_30 : { n : ℕ | valid_divisibility n }.to_finset.card = 2 := 
by
  sorry

end count_valid_n_le_30_l160_160275


namespace max_distinct_coin_sums_l160_160569

open Finset

theorem max_distinct_coin_sums : 
  let coin_values := {1, 5, 10, 25, 50}
  let coin_pairs := (coin_values.product coin_values).filter (λ p, p.1 ≤ p.2)
  let all_sums := coin_pairs.image (λ p, p.1 + p.2)
  all_sums.card = 15 :=
by
  let coin_values := {1, 5, 10, 25, 50}
  let coin_pairs := (coin_values.product coin_values).filter (λ p, p.1 ≤ p.2)
  let all_sums := coin_pairs.image (λ p, p.1 + p.2)
  have : all_sums = {2, 6, 11, 26, 51, 10, 15, 30, 55, 20, 35, 60, 50, 75, 100} := by sorry
  have : all_sums.card = 15  := by sorry
  exact this

end max_distinct_coin_sums_l160_160569


namespace range_of_m_n_l160_160035

noncomputable def tangent_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 → (m + 1) * x + (n + 1) * y - 2 = 0

theorem range_of_m_n (m n : ℝ) :
  tangent_condition m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end range_of_m_n_l160_160035


namespace integral_result_l160_160607

open Real

theorem integral_result :
  (∫ x in (0:ℝ)..(π/2), (x^2 - 5 * x + 6) * sin (3 * x)) = (67 - 3 * π) / 27 := by
  sorry

end integral_result_l160_160607


namespace Jasmine_gets_off_work_at_4pm_l160_160820

-- Conditions
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_clean_time : ℕ := 10
def groomer_time : ℕ := 20
def cook_time : ℕ := 90
def dinner_time : ℕ := 19 * 60  -- 7:00 pm in minutes

-- Question to prove
theorem Jasmine_gets_off_work_at_4pm : 
  (dinner_time - cook_time - groomer_time - dry_clean_time - grocery_time - commute_time = 16 * 60) := sorry

end Jasmine_gets_off_work_at_4pm_l160_160820


namespace parabola_positions_and_areas_l160_160613

noncomputable def f1 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def f2 (x : ℝ) : ℝ := x^2 - 3x + 1

-- Vertex calculation for f1: x = -b/2a
noncomputable def vertex_f1 : ℝ := -1 / (2 * 1)
-- Vertex calculation for f2: x = -b/2a
noncomputable def vertex_f2 : ℝ := 3 / (2 * 1)

-- Integration calculation for area under f1 from -1 to 1
noncomputable def area_f1 : ℝ := ∫ x in -1..1, f1 x

-- Integration calculation for area under f2 from -1 to 1
noncomputable def area_f2 : ℝ := ∫ x in -1..1, f2 x

theorem parabola_positions_and_areas :
  vertex_f1 < vertex_f2 ∧ area_f1 > area_f2 :=
by 
  -- Prove vertex positions
  have h1 : vertex_f1 = -1 / 2 := rfl
  have h2 : vertex_f2 = 3 / 2 := rfl
  -- Prove area calculations
  have h3 : area_f1 = ∫ x in -1..1, f1 x := rfl
  have h4 : area_f2 = ∫ x in -1..1, f2 x := rfl
  sorry

end parabola_positions_and_areas_l160_160613


namespace lucas_bus_time_l160_160050

noncomputable def lucas_time_spent_on_bus : ℕ :=
  let time_away_from_home : ℕ := ((5 + 3/4) + 3.5) * 60
  let time_spent_in_school : ℕ := (7 * 45) + 40 + (1.5 * 60)
  time_away_from_home - time_spent_in_school

theorem lucas_bus_time : lucas_time_spent_on_bus = 80 := sorry

end lucas_bus_time_l160_160050


namespace convex_pentagon_parallel_l160_160368

-- Define the given conditions
variables {A B C D E : Type*} [affine_space V ℝ] [add_comm_group V]

-- Definitions of parallel relations
variables {BC AD CD BE DE AC AE BD : V}

-- Given conditions in the problem:
def conditions (BC AD CD BE DE AC AE BD : V) : Prop := 
  (BC ∥ AD) ∧ 
  (CD ∥ BE) ∧ 
  (DE ∥ AC) ∧ 
  (AE ∥ BD)

-- The main theorem to prove
theorem convex_pentagon_parallel (A B C D E : Type*) [affine_space V ℝ] [add_comm_group V]
    (BC AD CD BE DE AC AE BD : V) :
  conditions BC AD CD BE DE AC AE BD → (AB ∥ CE) :=
begin
  -- Proof would go here
  sorry
end

end convex_pentagon_parallel_l160_160368


namespace four_digit_multiples_of_7_count_l160_160762

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160762


namespace a6_range_l160_160047

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem a6_range (a d : ℝ) (h1 : 0 < d) (h2 : arithmetic_seq a d 5 ≤ 6) (h3 : sum_of_first_n_terms a d 3 ≥ 9) :
  3 < arithmetic_seq a d 6 ∧ arithmetic_seq a d 6 ≤ 7 :=
by
  have h4 : d ≤ 1 := sorry
  have h5 : a ≥ 3 - d := sorry
  have h6 : arithmetic_seq a d 6 = a + 5 * d := sorry
  have h7 : 3 + 0 < a + 5 * d := by
    apply lt_add_of_pos_left
    exact h1
  have h8 : a + 5 * d ≤ 7 := sorry
  exact ⟨h7, h8⟩

end a6_range_l160_160047


namespace parallelogram_solution_l160_160253

structure Point where
  x : ℤ
  y : ℤ

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def parallelogram_perimeter (v1 v2 v3 v4 : Point) : ℝ :=
  2 * (distance v1 v2 + distance v2 v3)

def parallelogram_area (v1 v2 v3 v4 : Point) : ℝ :=
  let base := distance v2 v3
  let height := Real.abs (v2.y - v4.y) -- Assuming orthogonality for simplicity
  base * height

def sum_of_perimeter_and_area (v1 v2 v3 v4 : Point) : ℝ :=
  parallelogram_perimeter v1 v2 v3 v4 + parallelogram_area v1 v2 v3 v4

theorem parallelogram_solution :
  let v1 := Point.mk 2 3
  let v2 := Point.mk 5 7
  let v3 := Point.mk 11 7
  let v4 := Point.mk 8 3
  sum_of_perimeter_and_area v1 v2 v3 v4 = 46 :=
by
  sorry

end parallelogram_solution_l160_160253


namespace wrongly_written_height_l160_160462

theorem wrongly_written_height (n : ℕ) (a b h_avg_corr h_avg_wrong h_actual : ℝ)
  (cond1 : n = 35) 
  (cond2 : h_avg_wrong = 183)
  (cond3 : h_actual = 106)
  (cond4 : h_avg_corr = 181) :
  h_avg_wrong * n - h_avg_corr * n = sorry × (x_diff : ℝ) ∧ h_actual + x_diff = 176 :=
by {
  sorry
}

end wrongly_written_height_l160_160462


namespace correct_statements_count_l160_160943

-- Conditions as Lean definitions
def condition_1 := "In the residual plot, if the residual points are evenly distributed within a horizontal band, it indicates that the chosen model is appropriate."
def condition_2 := "The correct interpretation of R^2 is that the larger the value, the better the model fits."
def condition_3 := "The model with a smaller sum of squared residuals has a better fitting effect."

-- Lean statement verifying number of correct conditions
theorem correct_statements_count :
  (if condition_1 then 1 else 0) + 
  (if condition_2 then 1 else 0) + 
  (if condition_3 then 1 else 0) = 2 := 
sorry

end correct_statements_count_l160_160943


namespace every_pos_int_is_sum_of_distinct_fib_numbers_fib_unique_numbers_l160_160265

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := fib n + fib (n + 1)

-- Part (a): Every positive integer can be represented as a sum of distinct Fibonacci numbers
theorem every_pos_int_is_sum_of_distinct_fib_numbers (n : ℕ) (hn : n > 0) :
  ∃ (fib_nums : list ℕ), (∀ (x ∈ fib_nums), (∃ (m : ℕ), fib m = x)) ∧ list.sum fib_nums = n := by
sorry

-- Part (b): Finding all Fib-unique numbers
theorem fib_unique_numbers :
  {n : ℕ // fib n ≠ 1} = {n : ℕ | n > 1 ∧ ∀ (fib_sum1 fib_sum2 : list ℕ), (∀ (x ∈ fib_sum1, ∃ (m : ℕ), fib m = x)) ∧ (∀ (x ∈ fib_sum2, ∃ (m : ℕ), fib m = x)) ∧ list.sum fib_sum1 = n ∧ list.sum fib_sum2 = n → fib_sum1 = fib_sum2} := by
sorry

end every_pos_int_is_sum_of_distinct_fib_numbers_fib_unique_numbers_l160_160265


namespace identify_quadratic_l160_160532

def is_quadratic (eq : String) : Prop :=
  eq = "x^2 - 2x + 1 = 0"

theorem identify_quadratic :
  is_quadratic "x^2 - 2x + 1 = 0" :=
by
  sorry

end identify_quadratic_l160_160532


namespace pillar_D_height_l160_160590

noncomputable def height_of_pillar_D : Prop :=
  let A := (0, 0)
  let B := (10, 0)
  let C := (5, 5 * Real.sqrt 3)
  let P := (0, 0, 15 : ℝ)
  let Q := (10, 0, 12 : ℝ)
  let R := (5, 5 * Real.sqrt 3, 11 : ℝ)
  let D := (0, -10 * Real.sqrt 3)
  let pq := (10, 0, 12 - 15 : ℝ)
  let pr := (5, 5 * Real.sqrt 3, 11 - 15 : ℝ)
  let n := pq.cross_product pr
  let plane_eq := n.1 * x + n.2 * y + n.3 * z = n.1 * A.1 + n.2 * A.2 + n.3 * P.3
  ∃ z, plane_eq (D.1, D.2, z)

theorem pillar_D_height :
  height_of_pillar_D = 19 := 
sorry

end pillar_D_height_l160_160590


namespace four_digit_multiples_of_7_count_l160_160759

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160759


namespace square_sum_zero_real_variables_l160_160950

theorem square_sum_zero_real_variables (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end square_sum_zero_real_variables_l160_160950


namespace max_possible_a_max_possible_a_value_l160_160111

theorem max_possible_a (a x : ℤ) (h_eq : x^2 + a * x = 18) (h_pos : a > 0) : 
  17 ≤ a := sorry

theorem max_possible_a_value : ∃ (a x : ℤ), x^2 + a * x = 18 ∧ a > 0 ∧ ∀ (b : ℤ), (∃ y : ℤ, y^2 + b * y = 18 ∧ b > 0) → b ≤ a := 
begin
  use [17, 1],
  split,
  { norm_num }, -- show that with x = 1 and a = 17, the equation holds
  split,
  { norm_num }, -- show that a = 17 is positive
  { intros b,
    rintros ⟨y, hy, hy_pos⟩,
    obtain h_eq : y^2 + b * y = 18 := hy,
    have h : b ≤ 17 := max_possible_a b y h_eq hy_pos,
    exact h }
end

end max_possible_a_max_possible_a_value_l160_160111


namespace find_coordinates_of_M_l160_160011

-- Definitions of the given points
def A : ℝ × ℝ × ℝ := (1, 0, 2)
def B : ℝ × ℝ × ℝ := (1, -3, 1)

-- M is on the y-axis
def M_y (y : ℝ) : ℝ × ℝ × ℝ := (0, y, 0)

-- Definition of distance between two points in 3D space
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Proof problem
theorem find_coordinates_of_M :
  ∃ y : ℝ, M_y y = (0, -1, 0) ∧ distance (M_y y) A = distance (M_y y) B :=
by 
  -- proof skipped
  sorry

end find_coordinates_of_M_l160_160011


namespace smallest_sum_of_four_consecutive_primes_divisible_by_five_l160_160629

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    b = a + 2 ∧ c = b + 4 ∧ d = c + 2 ∧
    (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_five_l160_160629


namespace calculate_star_value_l160_160646

def star (x y : ℝ) (h : x ≠ y) : ℝ := (x + y) / (x - y)

theorem calculate_star_value : star (star (-2) 3 (ne_of_lt (by norm_num))) (-1/2) (ne_of_lt (by norm_num)) = -7/3 :=
by
  sorry

end calculate_star_value_l160_160646


namespace nickels_eq_100_l160_160141

variables (P D N Q H DollarCoins : ℕ)

def conditions :=
  D = P + 10 ∧
  N = 2 * D ∧
  Q = 4 ∧
  P = 10 * Q ∧
  H = Q + 5 ∧
  DollarCoins = 3 * H ∧
  (P + 10 * D + 5 * N + 25 * Q + 50 * H + 100 * DollarCoins = 2000)

theorem nickels_eq_100 (h : conditions P D N Q H DollarCoins) : N = 100 :=
by {
  sorry
}

end nickels_eq_100_l160_160141


namespace binomial_sum_identity_l160_160409

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem binomial_sum_identity (k n p : ℕ) :
  (∑ j in finset.range (p - n + 1), binomial (n + j) k) = binomial (p + 1) (k + 1) - binomial (n) (k + 1) :=
sorry

end binomial_sum_identity_l160_160409


namespace average_height_corrected_l160_160984

theorem average_height_corrected (students : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ)
  (h1 : students = 20)
  (h2 : incorrect_avg_height = 175)
  (h3 : incorrect_height = 151)
  (h4 : actual_height = 111) :
  (incorrect_avg_height * students - incorrect_height + actual_height) / students = 173 :=
by
  sorry

end average_height_corrected_l160_160984


namespace nested_star_eval_l160_160023

def star (a b : ℤ) : ℤ := a * b + a + b

theorem nested_star_eval : 
  1 ∗ (2 ∗ (3 ∗ (4 ∗ ⋯ (99 ∗ 100) ⋯))) = 101.factorial - 1 :=
by
  sorry

end nested_star_eval_l160_160023


namespace number_of_four_digit_multiples_of_7_l160_160698

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160698


namespace joseph_fewer_heads_than_tails_l160_160523

theorem joseph_fewer_heads_than_tails :
  let p := (∑ i in finset.range 6, (nat.choose 12 i) / (2^12 : ℝ)) in 
  p = 793 / 2048 := 
by
  sorry

end joseph_fewer_heads_than_tails_l160_160523


namespace steve_keeps_1800000_l160_160087

theorem steve_keeps_1800000 (S : ℕ) (P : ℝ) (A : ℝ) (M : ℝ) 
  (hS : S = 1000000) 
  (hP : P = 2) 
  (hA : A = 0.10) 
  (hM : M = S * P - (S * P * A)) : 
  M = 1,800,000 := 
by
  rw [hS, hP, hA, hM]
  norm_num

end steve_keeps_1800000_l160_160087


namespace sum_of_values_of_x_satisfying_sqrt_eq_8_l160_160955

theorem sum_of_values_of_x_satisfying_sqrt_eq_8 :
  (∑ x in { x : ℝ | Real.sqrt ((x - 2)^2) = 8 }, x) = 4 :=
by
  sorry

end sum_of_values_of_x_satisfying_sqrt_eq_8_l160_160955


namespace point_to_plane_distance_l160_160912

theorem point_to_plane_distance :
  let x0 := 0
  let y0 := 1
  let z0 := 3
  let A := 1
  let B := 2
  let C := 3
  let D := 3
  let p := (x0, y0, z0)
  let plane := (A, B, C, D)
  distance p plane = Real.sqrt 14 := by
    sorry

def distance (p : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := p
  let (A, B, C, D) := plane
  abs (A * x0 + B * y0 + C * z0 + D) / Real.sqrt (A^2 + B^2 + C^2)

end point_to_plane_distance_l160_160912


namespace decimal_rational_divisibility_l160_160397

theorem decimal_rational_divisibility (n a b : ℕ) (h_n_pos : n > 0) (h_n_not_div_2_5 : ¬ (2 ∣ n ∨ 5 ∣ n)) 
(h_rational : ∃ (k l : ℕ), (l > k) ∧ (a / b = (1 / n:ℝ) * 10 ^ (-i))) :
  n ∣ b := sorry

end decimal_rational_divisibility_l160_160397


namespace lisa_needs_additional_marbles_l160_160870

theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 50) :
  ∃ additional_marbles : ℕ, additional_marbles = 78 - marbles ∧ additional_marbles = 28 :=
by
  -- The sum of the first 12 natural numbers is calculated as:
  have h_sum : (∑ i in finset.range (friends + 1), i) = 78 := by sorry
  -- The additional marbles needed:
  use 78 - marbles
  -- It should equal to 28:
  split
  . exact rfl
  . sorry

end lisa_needs_additional_marbles_l160_160870


namespace sum_first_19_natural_numbers_l160_160543

theorem sum_first_19_natural_numbers :
  (∑ k in Finset.range 20, k) = 190 :=
by
  sorry

end sum_first_19_natural_numbers_l160_160543


namespace number_of_four_digit_multiples_of_7_l160_160699

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160699


namespace factorial_division_l160_160957

theorem factorial_division : (15.factorial) / ((6.factorial) * (9.factorial)) = 5005 := by
  sorry

end factorial_division_l160_160957


namespace lisa_needs_28_more_marbles_l160_160864

theorem lisa_needs_28_more_marbles :
  ∀ (friends : ℕ) (initial_marbles : ℕ),
  friends = 12 → 
  initial_marbles = 50 →
  (∀ n, 1 ≤ n ∧ n ≤ friends → ∃ (marbles : ℕ), marbles ≥ 1 ∧ ∀ i j, (i ≠ j ∧ i ≠ 0 ∧ j ≠ 0) → (marbles i ≠ marbles j)) →
  ( ∑ k in finset.range (friends + 1), k ) - initial_marbles = 28 :=
by
  intros friends initial_marbles h_friends h_initial_marbles _,
  rw [h_friends, h_initial_marbles],
  sorry

end lisa_needs_28_more_marbles_l160_160864


namespace members_who_play_both_sports_l160_160167

theorem members_who_play_both_sports 
  (N B T Neither BT : ℕ) 
  (h1 : N = 27)
  (h2 : B = 17)
  (h3 : T = 19)
  (h4 : Neither = 2)
  (h5 : BT = B + T - N + Neither) : 
  BT = 11 := 
by 
  have h6 : 17 + 19 - 27 + 2 = 11 := by norm_num
  rw [h2, h3, h1, h4, h6] at h5
  exact h5

end members_who_play_both_sports_l160_160167


namespace max_y_diff_eq_0_l160_160474

-- Definitions for the given conditions
def eq1 (x : ℝ) : ℝ := 4 - 2 * x + x^2
def eq2 (x : ℝ) : ℝ := 2 + 2 * x + x^2

-- Statement of the proof problem
theorem max_y_diff_eq_0 : 
  (∀ x y, eq1 x = y ∧ eq2 x = y → y = (13 / 4)) →
  ∀ (x1 x2 : ℝ), (∃ y1 y2, eq1 x1 = y1 ∧ eq2 x1 = y1 ∧ eq1 x2 = y2 ∧ eq2 x2 = y2) → 
  (x1 = x2) → (y1 = y2) →
  0 = 0 := 
by
  sorry

end max_y_diff_eq_0_l160_160474


namespace log_inequality_a_value_set_l160_160282

theorem log_inequality_a_value_set (a : ℝ) (h : abs (Real.log a⁻¹ 3/4) < 1) : 
  a ∈ set.Ioo 0 (3/4 : ℝ) ∪ set.Ioi (4/3 : ℝ) :=
sorry

end log_inequality_a_value_set_l160_160282


namespace candies_per_friend_l160_160429

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l160_160429


namespace simplify_and_evaluate_division_l160_160448

theorem simplify_and_evaluate_division (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a ^ 3 / (a ^ 2 - 4 * a + 4)) = 1 / 3 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_division_l160_160448


namespace line_symmetric_about_y_eq_x_l160_160664

-- Define the line equation types and the condition for symmetry
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions given
variable (a b c : ℝ)
variable (h_ab_pos : a * b > 0)

-- Definition of the problem in Lean
theorem line_symmetric_about_y_eq_x (h_bisector : ∀ x y : ℝ, line_equation a b c x y ↔ line_equation b a c y x) : 
  ∀ x y : ℝ, line_equation b a c x y := by
  sorry

end line_symmetric_about_y_eq_x_l160_160664


namespace min_value_inequality_l160_160839

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 := 
by
  sorry

end min_value_inequality_l160_160839


namespace coupon_probability_l160_160504

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l160_160504


namespace minimal_distance_l160_160990

noncomputable def y1 (U g t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def y2 (U g t τ : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
noncomputable def s (U g t τ : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem minimal_distance (U g τ : ℝ) (h : 2 * U ≥ g * τ) :
  ∃ t : ℝ, s U g t τ = 0 :=
begin
  use (2 * U / g + τ / 2),
  unfold s,
  sorry
end

end minimal_distance_l160_160990


namespace cannot_repair_l160_160237

-- Define the types of the tiles
inductive Tile
| square2x2
| rectangle4x1

-- Define the bathroom floor and the conditions
structure Bathroom :=
  (width : ℕ)
  (height : ℕ)
  (tiling : list Tile)

-- Define the condition of a broken tile and a replacement spare tile
structure Replacement :=
  (original : Tile)
  (spare : Tile)

-- The main theorem statement
theorem cannot_repair (B : Bathroom) (R : Replacement) : 
  ∃ b∈ B.tiling, b = R.original →
  ∀ new_tiling : list Tile, 
  (sorry : sorry (  (* Here we would specify the matching criteria *)
  new_tiling ≠ B.tiling → 
  (∀ t ∈ new_tiling, t = R.spare → 
  sorry (* Further conditions ensuring the tiling is correct *)) →
  new_tiling ≠ B.tiling → false)) :=
sorry

end cannot_repair_l160_160237


namespace joohyeon_snack_count_l160_160824

theorem joohyeon_snack_count
  (c s : ℕ)
  (h1 : 300 * c + 500 * s = 3000)
  (h2 : c + s = 8) :
  s = 3 :=
sorry

end joohyeon_snack_count_l160_160824


namespace not_divisible_sum_1987_l160_160889

theorem not_divisible_sum_1987 (n : ℕ) : ¬ (n + 2) ∣ ∑ i in Finset.range(n + 1), i ^ 1987 := sorry

end not_divisible_sum_1987_l160_160889


namespace impossible_permutation_of_cubes_l160_160190

theorem impossible_permutation_of_cubes :
  let S := finset.range 27
  let checkerboard_label (x y z : ℕ) := (x + y + z) % 2
  let move (position : ℕ × ℕ × ℕ) : finset (ℕ × ℕ × ℕ) :=
      finset.filter (λ p, (abs (p.1 - position.1) + abs (p.2 - position.2) + abs (p.3 - position.3)) = 1) (finset.univ : finset (ℕ × ℕ × ℕ))
  in
  ∀ (initial_state : ℕ → ℕ × ℕ × ℕ)
    (final_state : ℕ → ℕ × ℕ × ℕ)
    (cube27_moves_back : (initial_state 27) = (final_state 27))
    (required_permutation : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 26 → final_state n = initial_state (27 - n)),
  false :=
by
  sorry

end impossible_permutation_of_cubes_l160_160190


namespace maximize_volume_side_length_of_container_l160_160514

noncomputable def maximizeVolumeSideLength (L W : ℕ) : ℕ :=
  let vol := λ x, (L - 2 * x) * (W - 2 * x) * x
  let candidates := [(0 : ℕ) : λ y, y < min (L/2) (W/2)]
  let optimalSideLength := candidates.max_by (λ x, vol x)
  optimalSideLength

-- Let's assume our conditions
def L : ℕ := 90
def W : ℕ := 48

theorem maximize_volume_side_length_of_container 
    (L W : ℕ) (L = 90) (W = 48) :
  maximizeVolumeSideLength L W = 10 :=
by sorry

end maximize_volume_side_length_of_container_l160_160514


namespace find_b_l160_160130

variable {a b c : ℚ}

theorem find_b (h1 : a + b + c = 117) (h2 : a + 8 = 4 * c) (h3 : b - 10 = 4 * c) : b = 550 / 9 := by
  sorry

end find_b_l160_160130


namespace triangle_pqr_largest_perimeter_l160_160142

noncomputable def largest_perimeter : ℝ :=
  10 * Real.sqrt 7 + 10

theorem triangle_pqr_largest_perimeter
  (PQ PR QR : ℕ)
  (h1 : PQ = PR)
  (h2 : ∃ (S : ℝ × ℝ), PS = 10 ∧ angle QPS = 30)
  (PS : ℝ := 10)
  (QPS_angle : ℝ := 30) :
  PQ + PR + QR = largest_perimeter :=
begin
  sorry
end

end triangle_pqr_largest_perimeter_l160_160142


namespace total_cats_in_academy_is_100_l160_160600

open Finset

def CleverCatAcademy :=
  let J := {1, 2, ..., 60}     -- cats that can jump
  let PD := {61, 62, ..., 95}  -- cats that can play dead
  let F := {96, 97, ..., 135}  -- cats that can fetch
  let JP := {136, 137, ..., 155} -- cats that can jump and play dead
  let PF := {156, 157, ..., 170} -- cats that can play dead and fetch
  let JF := {171, 172, ..., 192} -- cats that can jump and fetch
  let AllThree := {193, ..., 202} -- cats that can do all three tricks
  let None := {203, ..., 214} -- cats that can do none of the tricks
  J ∪ PD ∪ F ∪ JP ∪ PF ∪ JF ∪ AllThree ∪ None

noncomputable def cleverCatAcademySet : Finset ℕ :=
  {1, ..., 214}.erase (197+205) -- removing index to define cats only once across sets

theorem total_cats_in_academy_is_100 : (cleverCatAcademySet.card = 100) := by
  let J := {1, ..., 60} -- 60 cats can jump
  let PD := {61, ..., 95} -- 35 cats can play dead
  let F := {96, ..., 135} -- 40 cats can fetch
  let JP := {136, ..., 155} -- 20 cats can jump and play dead
  let PF := {156, ..., 170} -- 15 cats can play dead and fetch
  let JF := {171, ..., 192} -- 22 cats can jump and fetch
  let AllThree := {193, ..., 202} -- 10 cats can do all three tricks
  let None := {203, ..., 214} -- 12 cats can do none of the tricks
  let Academy := J ∪ PD ∪ F ∪ JP ∪ PF ∪ JF ∪ AllThree ∪ None
  have h1: 60 cats can jump := rfl
  have h2: 35 cats can play dead := rfl
  have h3: 40 cats can fetch := rfl
  have h4: 20 cats can jump and play dead := rfl
  have h5: 15 cats can play dead and fetch := rfl
  have h6: 22 cats can jump and fetch := rfl
  have h7: 10 cats can do all three tricks := rfl
  have h8: 12 cats can do none of the tricks := rfl
  show_finite Academy
  sorry -- Proof skipped

end total_cats_in_academy_is_100_l160_160600


namespace difference_brothers_l160_160212

def aaron_brothers : ℕ := 4
def bennett_brothers : ℕ := 6

theorem difference_brothers : 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end difference_brothers_l160_160212


namespace minimum_distance_at_meeting_time_distance_glafira_to_meeting_l160_160998

variables (U g τ V : ℝ)
-- assumption: 2 * U ≥ g * τ
axiom h : 2 * U ≥ g * τ

noncomputable def motion_eq1 (t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def motion_eq2 (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2

noncomputable def distance (t : ℝ) : ℝ := 
|motion_eq1 U g t - motion_eq2 U g τ t|

noncomputable def meeting_time : ℝ := (2 * U / g) + (τ / 2)

theorem minimum_distance_at_meeting_time : distance U g τ meeting_time = 0 := sorry

noncomputable def distance_from_glafira_to_meeting : ℝ := 
V * meeting_time

theorem distance_glafira_to_meeting : 
distance_from_glafira_to_meeting U g τ V = V * ((τ / 2) + (U / g)) := sorry

end minimum_distance_at_meeting_time_distance_glafira_to_meeting_l160_160998


namespace last_three_digits_of_P_l160_160115

noncomputable def P : ℕ := List.prod (List.filter (λ n, n % 2 = 1) (List.range (2005 + 1)))

theorem last_three_digits_of_P :
  (P % 1000) = 375 := by
  have h1 : P % 8 = 1 := sorry
  have h2 : P % 125 = 0 := sorry
  exact sorry

end last_three_digits_of_P_l160_160115


namespace midpoint_translation_l160_160077

-- Define the points for segment s1
def point_A := (3, -2)
def point_B := (-7, 6)

-- Define the translation vector
def translation := (3, 4)

-- Find the midpoint of a segment given its endpoints
def midpoint (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  ((p1.fst + p2.fst) / 2, (p1.snd + p2.snd) / 2)

-- Apply the translation to a point
def translate (p t : ℤ × ℤ) : ℤ × ℤ :=
  (p.fst + t.fst, p.snd + t.snd)

-- Proof: The midpoint of segment s2 is (1, 6)
theorem midpoint_translation :
  translate (midpoint point_A point_B) translation = (1, 6) :=
by
  -- Point definitions
  let P := midpoint point_A point_B
  let Q := translate P translation
  -- End with a sorry statement to skip the actual proof
  sorry

end midpoint_translation_l160_160077


namespace quadratic_function_decreasing_l160_160573

def quadratic_function_decreasing_condition (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x < 4, ∃ x0 > x, f x0 < f x

theorem quadratic_function_decreasing (a b : ℝ) (h : quadratic_function_decreasing_condition (λ x, x^2 + 2*a*x + b) a b) : 
  a ≤ -4 := 
sorry

end quadratic_function_decreasing_l160_160573


namespace three_digit_numbers_containing_2_and_exclude_6_l160_160780

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l160_160780


namespace f_x1_plus_f_x2_gt_zero_l160_160918

-- Define the function properties and conditions
variables {f : ℝ → ℝ} 

-- Condition 1: f is defined on ℝ (implicit in Lean)
-- Condition 2: f(-x) = -f(x + 4)
axiom f_neg_equiv (x : ℝ) : f (-x) = -f (x + 4)

-- Condition 3: f(x) is monotonically increasing for x ≥ 2
axiom f_monotonic_increasing {x y : ℝ} (hx : x ≥ 2) (hy : y ≥ 2) : x ≤ y → f x ≤ f y

-- Condition 4: x1 + x2 > 4
axiom sum_gt_four {x1 x2 : ℝ} : x1 + x2 > 4

-- Condition 5: (x1 - 2)(x2 - 2) < 0
axiom product_lt_zero {x1 x2 : ℝ} : ((x1 - 2) * (x2 - 2)) < 0

-- Theorem statement: f(x1) + f(x2) > 0
theorem f_x1_plus_f_x2_gt_zero (x1 x2 : ℝ) (h1 : sum_gt_four x1 x2) (h2 : product_lt_zero x1 x2) : 
  f x1 + f x2 > 0 :=
by sorry

end f_x1_plus_f_x2_gt_zero_l160_160918


namespace circumcircle_centers_lie_on_circumcircle_l160_160037

open EuclideanGeometry

theorem circumcircle_centers_lie_on_circumcircle {A B C Q : Point} 
  (hQ : incenter Q A B C):
  let O₁ := circumcenter A Q B
  let O₂ := circumcenter B Q C
  let O₃ := circumcenter A Q C
  in 
  cyclic O₁ O₂ O₃ A B C := 
sorry

end circumcircle_centers_lie_on_circumcircle_l160_160037


namespace reciprocal_of_sum_l160_160954

theorem reciprocal_of_sum : (1 / 2 + 2 / 3)⁻¹ = 6 / 7 := by
  sorry

end reciprocal_of_sum_l160_160954


namespace equiangular_hexagon_side_lengths_relation_l160_160382

theorem equiangular_hexagon_side_lengths_relation
  (A B C D E F : Type)
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_equiangular : ∀ (i : ℕ), (i < 6) → ∀ (j : ℕ), (j < 6) → internal_angle A B C D E F i = internal_angle A B C D E F j)
  : a1 - a4 = a5 - a2 ∧ a5 - a2 = a3 - a6 :=
sorry

end equiangular_hexagon_side_lengths_relation_l160_160382


namespace quadratic_root_zero_l160_160795

theorem quadratic_root_zero (m : ℝ) :
  (∃ x : ℝ, (m-3) * x^2 + x + m^2 - 9 = 0) → m = -3 := 
by {
  intro h,
  rcases h with ⟨x, h_eq⟩,
  have hx : x = 0,
  { -- By plugging x = 0 in the original equation.
    sorry
  },
  have m_eq : m^2 - 9 = 0,
  { -- By substituting x = 0 into (m-3)x^2 + x + m^2 - 9 = 0.
    sorry
  },
  have m_calc : m = 3 ∨ m = -3,
  { -- By solving m^2 - 9 = 0.
    sorry
  },
  have m_ne_3 : m ≠ 3,
  { -- To ensure the quadratic equation is not degenerate, i.e., m - 3 ≠ 0
    sorry
  },
  have m_is_neg3 : m = -3,
  { -- Since m ≠ 3, the only remaining possibility from m = ±3 is m = -3.
    sorry
  },
  exact m_is_neg3,
}

end quadratic_root_zero_l160_160795


namespace semicircle_circumference_approx_19_28_l160_160124

theorem semicircle_circumference_approx_19_28 (length breadth : ℝ) (h_length : length = 9) (h_breadth : breadth = 6) : 
  let perimeter := 2 * (length + breadth),
      side := perimeter / 4,
      diameter := side,
      circumference := (Real.pi * diameter) / 2 + diameter
  in (Float.ofReal circumference).round 2 = 19.28 :=
by sorry

end semicircle_circumference_approx_19_28_l160_160124


namespace find_a_l160_160332

noncomputable def f (a : ℝ) (x : ℝ) := log a (1 - x) + log a (x + 3)

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) (h_min : ∃ x, f a x = -2) : a = 1 / 2 :=
by
  sorry

end find_a_l160_160332


namespace count_four_digit_multiples_of_7_l160_160730

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160730


namespace monotonic_increasing_interval_cos_l160_160928

theorem monotonic_increasing_interval_cos :
  ∀ (k : ℤ), ∃ (a b : ℝ), 
  (a = k * π - π/3) ∧ (b = k * π + π/6) ∧ 
  ∀ (x : ℝ), y = cos (π/3 - 2 * x) → (a <= x ∧ x <= b) := 
begin
  sorry
end

end monotonic_increasing_interval_cos_l160_160928


namespace pet_positions_l160_160008

def dog_positions : Fin 6 → String 
| ⟨0, _⟩ => "Top"
| ⟨1, _⟩ => "Top right"
| ⟨2, _⟩ => "Bottom right"
| ⟨3, _⟩ => "Bottom"
| ⟨4, _⟩ => "Bottom left"
| ⟨5, _⟩ => "Top left"

def rabbit_positions : Fin 12 → String
| ⟨0, _⟩ => "Top center"
| ⟨1, _⟩ => "Top right"
| ⟨2, _⟩ => "Right upper"
| ⟨3, _⟩ => "Right lower"
| ⟨4, _⟩ => "Bottom right"
| ⟨5, _⟩ => "Bottom center"
| ⟨6, _⟩ => "Bottom left"
| ⟨7, _⟩ => "Left lower"
| ⟨8, _⟩ => "Left upper"
| ⟨9, _⟩ => "Top left"
| ⟨10, _⟩ => "Left center"
| ⟨11, _⟩ => "Right center"

theorem pet_positions (n : Nat) :
  dog_positions ⟨n % 6, sorry⟩ = "Top" ∧ rabbit_positions ⟨n % 12, sorry⟩ = "Bottom left" :=
    by
      have h1 : n % 6 = 1 := sorry
      have h2 : n % 12 = 7 := sorry
      exact ⟨h1, h2⟩

end pet_positions_l160_160008


namespace both_hit_exactly_one_hits_at_least_one_hits_l160_160948

noncomputable def prob_A : ℝ := 0.8
noncomputable def prob_B : ℝ := 0.9

theorem both_hit : prob_A * prob_B = 0.72 := by
  sorry

theorem exactly_one_hits : prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 0.26 := by
  sorry

theorem at_least_one_hits : 1 - (1 - prob_A) * (1 - prob_B) = 0.98 := by
  sorry

end both_hit_exactly_one_hits_at_least_one_hits_l160_160948


namespace count_non_negative_integers_in_given_numbers_l160_160223

theorem count_non_negative_integers_in_given_numbers :
  let nums := {2, 7.5, -0.03, -0.4, 0, (1 : ℚ)/6, 10}
  (∑ n in nums, if (∃ k : ℤ, n = k ∧ k ≥ 0) then 1 else 0) = 3 :=
by
  sorry

end count_non_negative_integers_in_given_numbers_l160_160223


namespace tim_total_trip_time_l160_160137

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l160_160137


namespace proof_problem_l160_160678

theorem proof_problem (a b c : ℤ)
  (h1 : a + b + c = 6)
  (h2 : a - b + c = 4)
  (h3 : c = 3) : 3 * a - 2 * b + c = 7 := by
  sorry

end proof_problem_l160_160678


namespace solution_set_of_inequality_l160_160361

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 6

theorem solution_set_of_inequality (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ (-1/3 : ℝ) < m ∧ m < 3 :=
by 
  sorry

end solution_set_of_inequality_l160_160361


namespace pyramid_edge_bound_l160_160441

noncomputable def pyramid (V : Type) (n : ℕ) := 
  {S : V // ∃ A : fin n → V, ∀ i j, (i ≠ j) → A i ≠ A j }  -- Simplified representation of a pyramid

def vertex_angle_sum (p : pyramid V n) : ℝ := sorry  -- Define function that calculates sum of vertex angles

def semiperimeter (p : pyramid V n) : ℝ := sorry      -- Define function that calculates semiperimeter of the base

theorem pyramid_edge_bound (V : Type) (n : ℕ) (p : pyramid V n) (a : fin n → ℝ) (h₁: vertex_angle_sum p > 180) : 
  ∀ i, dist ((p.fst) (i : fin n)) (p.fst 0) < semiperimeter p / 2 := 
sorry

end pyramid_edge_bound_l160_160441


namespace ab_lt_zero_if_fa_fb_lt_zero_l160_160320

theorem ab_lt_zero_if_fa_fb_lt_zero (a b : ℝ) : 
  (∀ x : ℝ, f x = x^3) → f a + f b < 0 → a + b < 0 :=
by
  intros hfa hfb
  sorry

end ab_lt_zero_if_fa_fb_lt_zero_l160_160320


namespace exist_five_natural_numbers_sum_and_product_equal_ten_l160_160240

theorem exist_five_natural_numbers_sum_and_product_equal_ten : 
  ∃ (n_1 n_2 n_3 n_4 n_5 : ℕ), 
  n_1 + n_2 + n_3 + n_4 + n_5 = 10 ∧ 
  n_1 * n_2 * n_3 * n_4 * n_5 = 10 := 
sorry

end exist_five_natural_numbers_sum_and_product_equal_ten_l160_160240


namespace quadratic_solution_exists_l160_160069

theorem quadratic_solution_exists (a b : ℝ) : ∃ (x : ℝ), (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 :=
by
  sorry

end quadratic_solution_exists_l160_160069


namespace a_75_eq_24_l160_160790

variable {a : ℕ → ℤ}

-- Conditions for the problem
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_15_eq_8 : a 15 = 8 := sorry

def a_60_eq_20 : a 60 = 20 := sorry

-- The theorem we want to prove
theorem a_75_eq_24 (d : ℤ) (h_seq : is_arithmetic_sequence a d) (h15 : a 15 = 8) (h60 : a 60 = 20) : a 75 = 24 :=
  by
    sorry

end a_75_eq_24_l160_160790


namespace find_r_plus_s_l160_160925

noncomputable def line_eqn : (ℝ × ℝ) → Prop := 
λ P, P.2 = - (3 / 4) * P.1 + 9

def coordinates_Q : Prop := 
(0, 9) ∈ set_of line_eqn

def coordinates_P : Prop := 
(12, 0) ∈ set_of line_eqn

def point_T (r s : ℝ) := (r, s)

def area_POQ : ℝ := 54

def area_TOP (r s : ℝ) : ℝ := 18

-- Lean 4 statement of the problem
theorem find_r_plus_s (r s : ℝ) (hT_line : point_T r s ∈ set_of line_eqn)
  (ht_area : area_TOP r s = (1 / 3) * area_POQ) : r + s = 11 :=
sorry

end find_r_plus_s_l160_160925


namespace three_digit_numbers_containing_2_and_exclude_6_l160_160782

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l160_160782


namespace count_four_digit_multiples_of_7_l160_160734

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160734


namespace quadrilateral_HIJK_is_square_l160_160327

open Complex

structure Point := (x y : ℝ)

noncomputable def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

noncomputable def center_of_square (p1 p2 p3 p4 : Point) : Point :=
  midpoint (midpoint p1 p3) (midpoint p2 p4)

theorem quadrilateral_HIJK_is_square
  (A B C D E F G : Point)
  (H : Point) (hH : H = center_of_square A B C D)
  (J : Point) (hJ : J = center_of_square A E F G)
  (I : Point) (hI : I = midpoint E D)
  (K : Point) (hK : K = midpoint B G)
  : is_square H I J K :=
sorry

end quadrilateral_HIJK_is_square_l160_160327


namespace least_prime_b_l160_160098

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end least_prime_b_l160_160098


namespace no_2021_inhabitants_no_2021_inhabitants_cannot_have_2021_inhabitants_l160_160056

theorem no_2021_inhabitants (k l : ℕ) (h : k + l = 2021) :
  (k % 2 = 0 ∧ l % 2 = 1) → false :=
by sorry

theorem no_2021_inhabitants' (k l : ℕ) (h : k + l = 2021) :
  (k % 2 = 1 ∧ l % 2 = 0) → false :=
by sorry

theorem cannot_have_2021_inhabitants :
  ¬ ∃ (k l : ℕ), k + l = 2021 ∧ ((k % 2 = 0 ∧ l % 2 = 1) ∨ (k % 2 = 1 ∧ l % 2 = 0)) :=
by {
  intro h,
  cases h with k h,
  cases h with l h,
  cases h with hkl case,
  cases case;
  { apply no_2021_inhabitants; assumption } <|> { apply no_2021_inhabitants'; assumption }
}

#print cannot_have_2021_inhabitants

end no_2021_inhabitants_no_2021_inhabitants_cannot_have_2021_inhabitants_l160_160056


namespace number_of_four_digit_multiples_of_7_l160_160696

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160696


namespace find_relation_prove_inequality_l160_160675

-- Define the functions g(x) and f(x)
def g (x a b : ℝ) := x^3 + a * x^2 + b * x
def f (x a : ℝ) := (x + a) * Real.exp x

-- Condition: The function g(x) has extrema
def hasExtrema (a b : ℝ) := ∃ x, g' x = 0

-- Condition: The extremal points of f(x) are also the extremal points of g(x)
def sameExtremalPoints (a b : ℝ) := ∀ x, f' x = 0 → g' x = 0

-- The first theorem: finding the relationship between a and b
theorem find_relation (a b : ℝ) (h1 : hasExtrema a b) (h2 : sameExtremalPoints a b) :
  b = -a^2 - 4*a - 3 := sorry

-- Define F(x) = f(x) - g(x)
def F (x a b : ℝ) := (f x a) - (g x a b)

-- Define the minimum value of F(x) 
def M (a b : ℝ) := F (-a-1) a b

-- The second theorem: proving the inequality for M(a)
theorem prove_inequality (a : ℝ) (h : 0 < a) :
  ∀ b, (b = -a^2 - 4*a - 3) → M a b < -7/3 := sorry

end find_relation_prove_inequality_l160_160675


namespace num_four_digit_multiples_of_7_l160_160721

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160721


namespace find_interest_rate_l160_160195

-- Definitions based on conditions
def principal : ℝ := 3500
def rate_C : ℝ := 14 / 100
def time : ℝ := 3
def gain_B : ℝ := 420

-- Simple interest calculation
def interest_C (P R T : ℝ) : ℝ := P * R * T

-- B's interest from C
def interest_from_C : ℝ := interest_C principal rate_C time

-- B's interest to A
def interest_to_A (P R T : ℝ) : ℝ := P * R * T

-- Equation relating gain and interests
def gain_equation : Prop := gain_B = interest_from_C - interest_to_A principal (10 / 100) time

theorem find_interest_rate : 10% = sorry

end find_interest_rate_l160_160195


namespace inscribed_square_neq_five_l160_160805

theorem inscribed_square_neq_five (a b : ℝ) 
  (h1 : a - b = 1)
  (h2 : a * b = 1)
  (h3 : a + b = Real.sqrt 5) : a^2 + b^2 ≠ 5 :=
by sorry

end inscribed_square_neq_five_l160_160805


namespace probability_four_people_right_letter_is_zero_l160_160488

theorem probability_four_people_right_letter_is_zero :
  ∀ (letters : List ℕ) (distribution : List ℕ), 
  letters.length = 5 →
  distribution.length = 5 →
  (∀ i, letters.nth i ≠ none → distribution.nth i ≠ none) →
  (∀ i j, i ≠ j → letters.nth i ≠ letters.nth j) →
  (∃ n, List.count (letters.zip distribution) (λ (p : ℕ × ℕ), p.1 = p.2) = n → n ≠ 4) →
  Probability.probability_of_exactly_n_right_letters 4 5 = 0 :=
by sorry

end probability_four_people_right_letter_is_zero_l160_160488


namespace rectangle_distance_sum_l160_160243

theorem rectangle_distance_sum :
  let A := (0 : ℝ, 0 : ℝ),
      B := (3 : ℝ, 0 : ℝ),
      D := (0 : ℝ, 4 : ℝ),
      M := (1.5 : ℝ, 0 : ℝ),
      P := (0 : ℝ, 2 : ℝ),
      dist := λ (p1 p2 : ℝ × ℝ), real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  in dist A M + dist A P = 3.5 := 
by
  sorry

end rectangle_distance_sum_l160_160243


namespace andrey_gifts_l160_160063

theorem andrey_gifts :
  ∃ (n : ℕ), ∀ (a : ℕ), n(n-2) = a(n-1) + 16 ∧ n = 18 :=
by {
  sorry
}

end andrey_gifts_l160_160063


namespace isotomic_lines_cannot_meet_inside_medial_triangle_l160_160071

/-!
# Isotomic Lines and Medial Triangle

Prove that two isotomic lines of a triangle cannot meet inside its medial triangle.
-/

noncomputable def isotomic_lines (Δ : Triangle ℝ) (ℓ ℓ' : Line ℝ) : Prop :=
  -- Define the isotomic condition that each intersection of ℓ and ℓ' with the sides of the triangle are symmetric with respect to the midpoints.
  ∃ (A B C : Point ℝ) (D E D' E' : Point ℝ),
    Δ = ⟨A, B, C⟩ ∧
    is_midpoint A B C (midpoint B C) ∧ 
    is_midpoint B C A (midpoint C A) ∧ 
    is_midpoint C A B (midpoint A B) ∧
    intersects ℓ (line A B) D ∧
    intersects ℓ (line A C) E ∧
    intersects ℓ' (line A B) D' ∧
    intersects ℓ' (line A C) E' ∧
    symmetric D (midpoint B C) D' ∧
    symmetric E (midpoint C A) E'

theorem isotomic_lines_cannot_meet_inside_medial_triangle
  (Δ : Triangle ℝ) (ℓ ℓ' : Line ℝ)
  (h₁ : isotomic_lines Δ ℓ ℓ')
  (h₂ : Δ.medial_triangle = Δ') : ¬ (∃ (P : Point ℝ), inside P Δ' ∧ on_line P ℓ ∧ on_line P ℓ') :=
sorry

end isotomic_lines_cannot_meet_inside_medial_triangle_l160_160071


namespace infinite_series_converges_to_half_l160_160622

noncomputable def series_term (n : ℕ) : ℝ := (n^3 + n^2 - n) / (Real.ofNat (nat.factorial (n + 3)))

theorem infinite_series_converges_to_half :
  (∑' n, series_term n) = 1 / 2 :=
by sorry

end infinite_series_converges_to_half_l160_160622


namespace find_a_l160_160625

theorem find_a (a r s : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 24) (h3 : s^2 = 9) : a = 16 :=
sorry

end find_a_l160_160625


namespace max_min_value_l160_160475

noncomputable def func (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_value : 
  (∀ x ∈ set.Icc (0 : ℝ) 3, func x ≤ 5) ∧ 
  (∃ x ∈ set.Icc (0 : ℝ) 3, func x = 5) ∧
  (∀ x ∈ set.Icc (0 : ℝ) 3, func x ≥ -15) ∧
  (∃ x ∈ set.Icc (0 : ℝ) 3, func x = -15) := 
sorry

end max_min_value_l160_160475


namespace largest_prime_divisor_of_36_sq_plus_49_sq_l160_160633

theorem largest_prime_divisor_of_36_sq_plus_49_sq : ∃ (p : ℕ), p = 36^2 + 49^2 ∧ Prime p := 
by
  let n := 36^2 + 49^2
  have h : n = 3697 := by norm_num
  use 3697
  split
  . exact h
  . exact sorry

end largest_prime_divisor_of_36_sq_plus_49_sq_l160_160633


namespace percentage_both_colors_l160_160976

theorem percentage_both_colors
  (total_flags : ℕ)
  (even_flags : total_flags % 2 = 0)
  (C : ℕ)
  (total_flags_eq : total_flags = 2 * C)
  (blue_percent : ℕ)
  (blue_percent_eq : blue_percent = 60)
  (red_percent : ℕ)
  (red_percent_eq : red_percent = 65) :
  ∃ both_colors_percent : ℕ, both_colors_percent = 25 :=
by
  sorry

end percentage_both_colors_l160_160976


namespace sufficient_condition_for_proposition_l160_160479

theorem sufficient_condition_for_proposition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  assume h : ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0
  sorry

end sufficient_condition_for_proposition_l160_160479


namespace count_three_digit_numbers_with_2_without_6_l160_160776

theorem count_three_digit_numbers_with_2_without_6 : 
  let total_without_6 : ℕ := 648
  let total_without_6_and_2 : ℕ := 448
  total_without_6 - total_without_6_and_2 = 200 :=
by 
  have total_without_6 := 8 * 9 * 9
  have total_without_6_and_2 := 7 * 8 * 8
  rw total_without_6
  rw total_without_6_and_2
  exact calc
    8 * 9 * 9 - 7 * 8 * 8 = 648 - 448 := by simp
    ... = 200 := by norm_num

end count_three_digit_numbers_with_2_without_6_l160_160776


namespace lisa_needs_additional_marbles_l160_160871

theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 50) :
  ∃ additional_marbles : ℕ, additional_marbles = 78 - marbles ∧ additional_marbles = 28 :=
by
  -- The sum of the first 12 natural numbers is calculated as:
  have h_sum : (∑ i in finset.range (friends + 1), i) = 78 := by sorry
  -- The additional marbles needed:
  use 78 - marbles
  -- It should equal to 28:
  split
  . exact rfl
  . sorry

end lisa_needs_additional_marbles_l160_160871


namespace bricks_in_bottom_half_l160_160049

variable (bricksTotal numRows bottomRowsPerRow topRowsPerRow : ℕ)
variable (bricksBottom bricksTop : ℕ)

-- Mathematical conditions as in the problem statement
def bottomRows : ℕ := numRows / 2
def topRows : ℕ := numRows / 2
def bricksTopTotal : ℕ := topRows * topRowsPerRow
def bricksBottomTotal : ℕ := bottomRows * bottomRowsPerRow
def iglooTotalBricks : ℕ := bricksBottomTotal + bricksTopTotal

theorem bricks_in_bottom_half (h1 : numRows = 10)
                             (h2 : bottomRows = 5)
                             (h3 : topRows = 5)
                             (h4 : topRowsPerRow = 8)
                             (h5 : iglooTotalBricks = 100) :
                             bottomRowsPerRow = 12 :=
by
  -- Place the proof here
  sorry

end bricks_in_bottom_half_l160_160049


namespace num_ways_distribute_cards_l160_160888

theorem num_ways_distribute_cards (n : ℕ) (h : n ≥ 1) : 
  (∑ k in Finset.range(n).filter (λ k, 0 < k), Nat.choose n k) = 2 * (2^(n-1) - 1) := 
sorry

end num_ways_distribute_cards_l160_160888


namespace range_of_fx_l160_160940

noncomputable def f (x : ℝ) := x^2

theorem range_of_fx : set.range (f ∘ (λ x, x ∈ Ioc (-1: ℝ) 2)) = set.Icc (0 : ℝ) 4 :=
sorry

end range_of_fx_l160_160940


namespace coupon_probability_l160_160495

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l160_160495


namespace simplify_expression_l160_160080

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 :=
by sorry

end simplify_expression_l160_160080


namespace seats_filled_percentage_l160_160370

theorem seats_filled_percentage (total_seats : ℕ) (vacant_seats : ℕ) (filled_percentage : ℝ)
  (h1 : total_seats = 600)
  (h2 : vacant_seats = 300) :
  filled_percentage = ((total_seats - vacant_seats : ℕ).toReal / total_seats.toReal) * 100 :=
by
  sorry

end seats_filled_percentage_l160_160370


namespace work_days_after_join_l160_160560

noncomputable def timeToFinish (
  initial_workers : ℕ, 
  additional_workers : ℕ,
  total_toys : ℕ, 
  initial_days : ℕ,
  days_until_addition : ℕ → ℕ) : ℕ := sorry

axiom workers_toy_rate 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (total_toys : ℕ) 
  (rate: ℕ) : initial_workers * initial_days * rate = total_toys

theorem work_days_after_join 
  (initial_workers : ℕ) 
  (additional_workers : ℕ)
  (total_toys : ℕ) 
  (initial_days : ℕ) 
  (days_until_addition : ℕ)
  (rate : ℕ)
  (
    h1: initial_workers = 14,
    h2: additional_workers = 14,
    h3: total_toys = 1400,
    h4: initial_days = 5,
    h5: days_until_addition = 1,
    hr: rate = 20,
    total_initial_toy : total_toys % initial_days = 280,
    remaining_toy : total_toys - total_initial_toy = 1120,
    combined_workers : initial_workers + additional_workers = 28,
    combined_rate : rate * combined_workers = 560
  ) : timeToFinish initial_workers additional_workers total_toys initial_days days_until_addition = 2 := sorry

end work_days_after_join_l160_160560


namespace ratio_triangle_EFG_to_square_ABCD_l160_160809

theorem ratio_triangle_EFG_to_square_ABCD 
  (s : ℝ) (A B C D E F G : ℝ × ℝ) 
  (square_ABCD : 
    (AB : ℝ) → (BC : ℝ) → (CD : ℝ) → (DA : ℝ) → 
    (AB = s ∧ BC = s ∧ CD = s ∧ DA = s)) 
  (AE_3EC : (dist A E) = 3 * (dist E C)) 
  (BF_2FB : (dist B F) = 2 * (dist F B)) 
  (G_mid_CD : G = (C + D) / 2) : 
  (let area_square := s ^ 2 in
   let EG := (dist E G) in
   let height_BF := (dist B F) in
   let area_triangle := 0.5 * EG * height_BF in
   area_triangle / area_square = 1 / 24) := 
by
  sorry

end ratio_triangle_EFG_to_square_ABCD_l160_160809


namespace integer_900_in_column_B_l160_160220

def repeating_pattern := ["B", "C", "D", "A", "F", "E", "F", "D", "C", "B", "A", "B"]

def position_in_pattern (n : ℕ) : String :=
  repeating_pattern[(n % 12)]

theorem integer_900_in_column_B : position_in_pattern 898 = "B" :=
by {
  sorry
}

end integer_900_in_column_B_l160_160220


namespace puppies_brought_in_correct_l160_160200

-- Define the initial number of puppies in the shelter
def initial_puppies: Nat := 2

-- Define the number of puppies adopted per day
def puppies_adopted_per_day: Nat := 4

-- Define the number of days over which the puppies are adopted
def adoption_days: Nat := 9

-- Define the total number of puppies adopted after the given days
def total_puppies_adopted: Nat := puppies_adopted_per_day * adoption_days

-- Define the number of puppies brought in
def puppies_brought_in: Nat := total_puppies_adopted - initial_puppies

-- Prove that the number of puppies brought in is 34
theorem puppies_brought_in_correct: puppies_brought_in = 34 := by
  -- proof omitted, filled with sorry to skip the proof
  sorry

end puppies_brought_in_correct_l160_160200


namespace sum_integer_solutions_abs_lt_l160_160152

theorem sum_integer_solutions_abs_lt (I : Set ℤ) (I_eq : I = { n : ℤ | |n| < |n + 1| ∧ |n + 1| < 6 }) :
  (∑ n in I, n) = -27 := by
  sorry

end sum_integer_solutions_abs_lt_l160_160152


namespace determine_digit_l160_160905

theorem determine_digit (d : ℕ) (h1 : d ∈ {0, 2, 4, 6, 8}) (h2 : (8 + 5 + 4 + 3 + d) % 3 = 0) : d = 4 :=
by sorry

end determine_digit_l160_160905


namespace cost_of_pen_is_30_l160_160197

noncomputable def mean_expenditure_per_day : ℕ := 500
noncomputable def days_in_week : ℕ := 7
noncomputable def total_expenditure : ℕ := mean_expenditure_per_day * days_in_week

noncomputable def mon_expenditure : ℕ := 450
noncomputable def tue_expenditure : ℕ := 600
noncomputable def wed_expenditure : ℕ := 400
noncomputable def thurs_expenditure : ℕ := 500
noncomputable def sat_expenditure : ℕ := 550
noncomputable def sun_expenditure : ℕ := 300

noncomputable def fri_notebook_cost : ℕ := 50
noncomputable def fri_earphone_cost : ℕ := 620

noncomputable def total_non_fri_expenditure : ℕ := 
  mon_expenditure + tue_expenditure + wed_expenditure + 
  thurs_expenditure + sat_expenditure + sun_expenditure

noncomputable def fri_expenditure : ℕ := 
  total_expenditure - total_non_fri_expenditure

noncomputable def fri_pen_cost : ℕ := 
  fri_expenditure - (fri_earphone_cost + fri_notebook_cost)

theorem cost_of_pen_is_30 : fri_pen_cost = 30 :=
  sorry

end cost_of_pen_is_30_l160_160197


namespace least_possible_d_l160_160904

def has_n_factors (n : ℕ) (k : ℕ) : Prop := 
  (finset.filter (λ i : ℕ, k % i = 0) (finset.range (k+1))).card = n

theorem least_possible_d (c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) 
  (h3 : has_n_factors 4 c) (h4 : has_n_factors c d) (h5 : d % c = 0) : 
  d = 12 :=
sorry

end least_possible_d_l160_160904


namespace food_remaining_l160_160422

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end food_remaining_l160_160422


namespace solution_set_of_inequality_l160_160937

theorem solution_set_of_inequality:
  {x : ℝ | x^2 - |x-1| - 1 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end solution_set_of_inequality_l160_160937


namespace min_value_of_2a_plus_b_l160_160308

variable {a b : ℝ}

theorem min_value_of_2a_plus_b (h_pos_a : 0 < a) 
                                (h_pos_b : 0 < b)
                                (h_log_eq : log 4 (2 * a + b) = log 2 (sqrt (a * b))) :
    2 * a + b ≥ 8 := 
sorry

end min_value_of_2a_plus_b_l160_160308


namespace vector_combination_l160_160833

noncomputable def midpoint (A B : Type) [Add A] [Mul A] [HasScalar ℝ A] := (1 / 2 : ℝ) • A + (1 / 2 : ℝ) • B

axiom midpoint_def (A B : Type) [Add A] [Mul A] [HasScalar ℝ A] : ∀ A B, midpoint A B = (1 / 2 : ℝ) • A + (1 / 2 : ℝ) • B

theorem vector_combination (A B P C : Type) [Add A] [Mul A] [HasScalar ℝ A] (hC : C = midpoint A B) (hP : P = (1 / 5 : ℝ) • (4 • A + C)) :
  P = (9 / 10 : ℝ) • A + (1 / 10 : ℝ) • B :=
by sorry

#check vector_combination

end vector_combination_l160_160833


namespace movie_theater_ticket_cost_l160_160599

theorem movie_theater_ticket_cost
  (adult_ticket_cost : ℝ)
  (child_ticket_cost : ℝ)
  (total_moviegoers : ℝ)
  (total_amount_paid : ℝ)
  (number_of_adults : ℝ)
  (H_child_ticket_cost : child_ticket_cost = 6.50)
  (H_total_moviegoers : total_moviegoers = 7)
  (H_total_amount_paid : total_amount_paid = 54.50)
  (H_number_of_adults : number_of_adults = 3)
  (H_number_of_children : total_moviegoers - number_of_adults = 4) :
  adult_ticket_cost = 9.50 :=
sorry

end movie_theater_ticket_cost_l160_160599


namespace largest_prime_divisor_of_36_sq_plus_49_sq_l160_160635

theorem largest_prime_divisor_of_36_sq_plus_49_sq : ∃ (p : ℕ), p = 36^2 + 49^2 ∧ Prime p := 
by
  let n := 36^2 + 49^2
  have h : n = 3697 := by norm_num
  use 3697
  split
  . exact h
  . exact sorry

end largest_prime_divisor_of_36_sq_plus_49_sq_l160_160635


namespace intersecting_lines_l160_160511

noncomputable def m_b_sum : ℚ :=
  let m := 11 / 8
  let b := -18
  b + m

theorem intersecting_lines (m b : ℚ) (h₁ : ∀ x y : ℚ, y = m * x + 3 ↔ y = 4 * x + b) (h₂ : m = 11 / 8) (h₃ : b = -18) :
  m_b_sum = (-133 / 8) :=
by
  rw [m_b_sum, h₂, h₃]
  norm_num
  sorry

end intersecting_lines_l160_160511


namespace points_on_a_line_l160_160303

theorem points_on_a_line (n k : ℕ) 
  (points : List ℝ) (len_points : points.length = n) 
  (colors : Finset (Fin k)) :
  (n > k + 1) → 
  ∃ (c₁ c₂ ∈ colors) (P_i P_j P_m P_n : ℝ), 
    (P_i, P_j ∈ points) ∧ (P_m, P_n ∈ points) ∧ 
    c₁ = c₂ ∧ 
    has_common_external_tangent (circle (P_i, P_j)) (circle (P_m, P_n)) := 
  sorry

end points_on_a_line_l160_160303


namespace dryer_runtime_per_dryer_l160_160598

-- Definitions for the given conditions
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℕ := 2
def num_dryers : ℕ := 3
def total_spent : ℝ := 11

-- Statement to prove
theorem dryer_runtime_per_dryer : 
  (2 * washer_cost + ((total_spent - 2 * washer_cost) / dryer_cost_per_10min) * 10) / num_dryers = 40 :=
by
  sorry

end dryer_runtime_per_dryer_l160_160598


namespace average_mpg_l160_160232

noncomputable def initial_odometer : ℕ := 34500
noncomputable def final_odometer : ℕ := 35350
noncomputable def total_gasoline : ℕ := 10 + 10 + 15 + 10

theorem average_mpg :
  let total_distance := final_odometer - initial_odometer in
  let average_mpg := total_distance / total_gasoline.to_rat in
  average_mpg ≈ 18.9 :=
by
  let total_distance := final_odometer - initial_odometer
  let average_mpg := total_distance / total_gasoline.to_rat
  sorry

end average_mpg_l160_160232


namespace elevator_group_distribution_l160_160592

theorem elevator_group_distribution:
  let num_floors := 10
  let num_passengers := 9
  let group_sizes := (2, 3, 4)
  let group_combinations := 
    (finset.card (finset.powerset_len group_sizes.1 (finset.range num_passengers))) *
    (finset.card (finset.powerset_len group_sizes.2 (finset.filter 
      (λ x, x ∉ finset.range(2)) (finset.range num_passengers)))) *
    (finset.card (finset.powerset_len group_sizes.3 (finset.filter 
      (λ x, x ∉ finset.range(2 + 3)) (finset.range num_passengers))))
  let num_ways_permutations := (nat.factorial 10) / (nat.factorial (10 - 3))
  let total_ways := group_combinations * num_ways_permutations
  in total_ways = (nat.factorial 10) / 4 := sorry

end elevator_group_distribution_l160_160592


namespace find_f_neg1_l160_160650

def f : ℝ → ℝ
| x := if x < 6 then f (x + 3) else real.log x / real.log 2

theorem find_f_neg1 : f (-1) = 3 := sorry

end find_f_neg1_l160_160650


namespace f_increasing_range_of_a_S_n_less_than_1_f_of_S_n_less_than_1_l160_160291

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_positive : ∀ x, 0 < x ∧ x < 1 → f x > 0
axiom f_at_1 : f 1 = 1
axiom f_convex : ∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Problem Ⅰ
theorem f_increasing (x1 x2 : ℝ) (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (h : x1 < x2) : f x1 < f x2 :=
sorry

-- Problem Ⅱ
theorem range_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → 4 * f x ^ 2 - 4 * (2 - a) * f x + 5 - 4 * a ≥ 0) : a ≤ 1 :=
sorry

-- Problem Ⅲ
def S (n : ℕ) := ∑ i in range (n + 1), i / 2^(i + 1)

theorem S_n_less_than_1 (n : ℕ) : S n < 1 :=
sorry

theorem f_of_S_n_less_than_1 (n : ℕ) : f (S n) < 1 :=
sorry

end f_increasing_range_of_a_S_n_less_than_1_f_of_S_n_less_than_1_l160_160291


namespace initial_average_age_of_students_l160_160460

theorem initial_average_age_of_students 
(A : ℕ) 
(h1 : 23 * A + 46 = (A + 1) * 24) : 
  A = 22 :=
by
  sorry

end initial_average_age_of_students_l160_160460


namespace find_minimum_x_and_values_l160_160346

theorem find_minimum_x_and_values (x y z w : ℝ) (h1 : y = x - 2003)
  (h2 : z = 2 * y - 2003)
  (h3 : w = 3 * z - 2003)
  (h4 : 0 ≤ x)
  (h5 : 0 ≤ y)
  (h6 : 0 ≤ z)
  (h7 : 0 ≤ w) :
  x ≥ 10015 / 3 ∧ 
  (x = 10015 / 3 → y = 4006 / 3 ∧ z = 2003 / 3 ∧ w = 0) := by
  sorry

end find_minimum_x_and_values_l160_160346


namespace find_S_2n_l160_160286

def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℤ :=
  if Even n then 2 * a n else - (a n)

def S (n : ℕ) : ℤ :=
  Nat.sum (Finset.range n) (λ i, b (i + 1))

theorem find_S_2n (n : ℕ) : S (2 * n) = 2 * n ^ 2 + 3 * n := 
sorry

end find_S_2n_l160_160286


namespace problem1_problem2_l160_160677

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

-- Problem 1: Show that f(x) < 0 for all x ∈ (1, +∞) iff a ≤ 1
theorem problem1 (a : ℝ) : (∀ x : ℝ, 1 < x → f(a, x) < 0) ↔ a ≤ 1 :=
sorry

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1 / x) - 1

-- Problem 2: Show that the maximum value of g(x₂) - g(x₁) when 0 < a ≤ e + 1/e is 4/e
theorem problem2 (a : ℝ) (h₀ : 0 < a) (h₁ : a ≤ Real.exp 1 + 1 / Real.exp 1) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ g a x₂ - g a x₁ = 4 / Real.exp 1 :=
sorry

end problem1_problem2_l160_160677


namespace problem_proof_l160_160299

open_locale big_operators

-- Definitions as per conditions
def seq (n : ℕ) : ℝ → ℝ := sorry -- Placeholder definition for the sequence a_n
def sum_seq (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), seq k

axiom a1 : seq 1 = 1
axiom recurrence (n : ℕ) : seq n + (-1)^n * seq (n + 1) = 1 - n / 2022

-- The main theorem to prove
theorem problem_proof : sum_seq 2023 = 506 :=
by sorry

end problem_proof_l160_160299


namespace spirangle_length_l160_160268

theorem spirangle_length :
  let a := 2
  let d := 2
  let l := 200
  let n := (l - a) / d + 1
  let Sn := n * (a + l) / 2
  let final_segment := Sn + 201
  final_segment = 10301 :=
by
  let a := 2
  let d := 2
  let l := 200
  let n := (l - a) / d + 1
  let Sn := n * (a + l) / 2
  let final_segment := Sn + 201
  exact eq.refl final_segment

end spirangle_length_l160_160268


namespace angle_closer_to_AD_greater_l160_160908

theorem angle_closer_to_AD_greater {A B C D : Type} [has_angle A] [has_angle B] [has_angle C] [has_angle D]
  (h1 : AD = 2 * BC) (h2 : ∠C = (3 / 2) * ∠A) :
  angle_closer_toAD_greater A B C D :=
sorry

end angle_closer_to_AD_greater_l160_160908


namespace solve_equation_l160_160082

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end solve_equation_l160_160082


namespace fuel_consumption_speed_range_l160_160181

theorem fuel_consumption_speed_range (x k : ℝ) (h1 : 60 ≤ x ∧ x ≤ 120)
  (h2 : ∀ x = 120, (1/5) * (120 - k + 4500 / 120) = 11.5)
  (h3 : ∀ x, (1/5) * (x - k + 4500 / x) ≤ 9) : 60 ≤ x ∧ x ≤ 100 :=
begin
  sorry
end

end fuel_consumption_speed_range_l160_160181


namespace lisa_additional_marbles_l160_160868

theorem lisa_additional_marbles (n_friends : ℕ) (initial_marbles : ℕ) (h_friends : n_friends = 12) (h_marbles : initial_marbles = 50) :
  let total_marbles_needed := (n_friends * (n_friends + 1)) / 2 in
  total_marbles_needed - initial_marbles = 28 :=
by
  sorry

end lisa_additional_marbles_l160_160868


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l160_160637

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l160_160637


namespace cricket_innings_l160_160464

theorem cricket_innings (n : ℕ) (h1 : (32 * n + 137) / (n + 1) = 37) : n = 20 :=
sorry

end cricket_innings_l160_160464


namespace largestNumberAcuteAnglesInConvexOctagon_l160_160522

def sumInteriorAngles (n : ℕ) := (n - 2) * 180

theorem largestNumberAcuteAnglesInConvexOctagon (octagon : Type) [polygon octagon] (h : isConvex octagon) 
  (interiorAngleSum : sumInteriorAngles 8 = 1080) :
  ∃ n ≤ 8, (∀ i, i < n → polygon.angle i < 90) ∧ (∀ j, j ≥ n → polygon.angle j > 90) ∧ n = 5 := sorry

end largestNumberAcuteAnglesInConvexOctagon_l160_160522


namespace hyperbola_asymptote_l160_160680

-- Define the conditions
variables (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
          (hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
        
-- The distance condition from the focus to the asymptote
def dist_condition (c : ℝ) : Prop := (b * c / sqrt (a^2 + b^2) = 2 * a)

-- Prove that the equation of the asymptote is y = ±2x
theorem hyperbola_asymptote (c : ℝ) (h_dist : dist_condition a b c) : 
  ∀ x : ℝ, (∀ y : ℝ, y = x -> y = 2*x ∨ y = -2*x) := 
sorry

end hyperbola_asymptote_l160_160680


namespace smallest_possible_value_of_N_l160_160231

open Nat

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, 
    (∀ (n : ℕ), n ∈ (finset.range 60).filter (λ k, (k % 3 = 1)) →
                   (∃ i : ℕ, (n + i) % 60 = k ∨ (n - i + 60) % 60 = k)) 
    ∧ N = 20 := 
sorry

end smallest_possible_value_of_N_l160_160231


namespace find_p_q_r_l160_160305

theorem find_p_q_r
  (t : ℝ)
  (p q r : ℕ)
  (h1 : (1 + sin t) * (1 + cos t) = 3 / 2)
  (h2 : (1 - sin t) * (1 - cos t) = p / q - real.sqrt r)
  (hpq_coprime : nat.coprime p q) :
  p + q + r = 33 :=
sorry

end find_p_q_r_l160_160305


namespace sin_double_angle_l160_160648

theorem sin_double_angle {x : ℝ} (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_l160_160648


namespace minimum_value_of_f_monotonicity_of_F_sum_of_roots_greater_than_a_l160_160672

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def F (x : ℝ) (a : ℝ) : ℝ := x^2 - a * (x + (Real.log x + 1)) + 2 * x

-- Minimum value of f(x) = x * ln x.
theorem minimum_value_of_f : 
  ∃ x_min, f x_min = -1 / Real.exp 1 ∧ ∀ x > 0, f x ≥ f x_min :=
sorry

-- Monotonicity of F(x) = x^2 - a * (x + f'(x)) + 2 * x
theorem monotonicity_of_F (a : ℝ) :
  if a ≤ 0 then
    ∀ x > 0, F x a ≥ F (x - 0.01) a
  else
    (∀ x > a / 2, F x a ≥ F (x - 0.01) a) ∧ (∀ x ≤ a / 2, F x a ≤ F (x + 0.01) a) :=
sorry

-- If F(x) = m has two distinct real roots x1 and x2, prove x1 + x2 > a.
theorem sum_of_roots_greater_than_a (m a : ℝ) (x1 x2 : ℝ) : 
  F x1 a = m → F x2 a = m → x1 ≠ x2 → x1 + x2 > a :=
sorry

end minimum_value_of_f_monotonicity_of_F_sum_of_roots_greater_than_a_l160_160672


namespace integer_solutions_to_abs_equation_l160_160084

theorem integer_solutions_to_abs_equation :
  {p : ℤ × ℤ | abs (p.1 - 2) + abs (p.2 - 1) = 1} =
  {(3, 1), (1, 1), (2, 2), (2, 0)} :=
by
  sorry

end integer_solutions_to_abs_equation_l160_160084


namespace only_common_term_is_one_l160_160936

def x_seq : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 1) := x_seq n + 2 * x_seq (n - 1)

def y_seq : ℕ → ℕ
| 0       := 1
| 1       := 7
| (n + 1) := 2 * y_seq n + 3 * y_seq (n - 1)

theorem only_common_term_is_one : ∀ n m, (x_seq n = y_seq m) → (x_seq n = 1) :=
sorry

end only_common_term_is_one_l160_160936


namespace max_distance_to_line_l160_160117

theorem max_distance_to_line : ∀ (k : ℝ),
  let P := (-1 : ℝ, 3 : ℝ)
  let l (x : ℝ) := k * (x - 2)
  let distance := 3 * Real.sqrt ((1 + (2 * k) / (k^2 + 1)))
  0 ≤ distance ∧ distance ≤ 3 * Real.sqrt 2 :=
sorry

end max_distance_to_line_l160_160117


namespace paula_candies_l160_160436

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l160_160436


namespace common_divisors_sum_diff_l160_160122

theorem common_divisors_sum_diff (A B : ℤ) (h : Int.gcd A B = 1) : 
  {d : ℤ | d ∣ A + B ∧ d ∣ A - B} = {1, 2} :=
sorry

end common_divisors_sum_diff_l160_160122


namespace count_four_digit_multiples_of_7_l160_160728

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160728


namespace four_digit_multiples_of_7_l160_160714

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160714


namespace solve_problem_l160_160797

def min_distance_from_circle_to_line (C : ℝ → ℝ × ℝ → Prop) (l : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, (C (2 : ℝ) ((0 : ℝ), (0 : ℝ))) ∧ (∀ (x y : ℝ), y = x + a) ∧
    ((∀ (x y : ℝ), (x^2 + y^2 = 4 → (abs (a / (√2 : ℝ)) = 4)))

theorem solve_problem : ∃ a : ℝ, min_distance_from_circle_to_line (λ r (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2) (λ x, x + a)
  := sorry

end solve_problem_l160_160797


namespace largest_even_number_in_sequence_of_six_l160_160632

-- Definitions and conditions
def smallest_even_number (x : ℤ) : Prop :=
  x + (x + 2) + (x+4) + (x+6) + (x + 8) + (x + 10) = 540

def sum_of_squares_of_sequence (x : ℤ) : Prop :=
  x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 + (x + 8)^2 + (x + 10)^2 = 97920

-- Statement to prove
theorem largest_even_number_in_sequence_of_six (x : ℤ) (h1 : smallest_even_number x) (h2 : sum_of_squares_of_sequence x) : x + 10 = 95 :=
  sorry

end largest_even_number_in_sequence_of_six_l160_160632


namespace range_of_m_l160_160666

variable {α : Type*} [LinearOrder α]

def f (x : α) : α := sorry

theorem range_of_m (m : α) (h1 : -2 ≤ 1 - m ∧ 1 - m ≤ 2)
                   (h2 : -2 ≤ m ∧ m ≤ 2)
                   (h3 : f(1 - m) < f(m))
                   (h_inc : ∀ x y, -2 ≤ x ∧ x ≤ 2 → -2 ≤ y ∧ y ≤ 2 → x < y → f(x) < f(y)) :
     0.5 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l160_160666


namespace sum_of_repeated_eights_l160_160988

theorem sum_of_repeated_eights (n : ℕ) : 
  let sum := ∑ k in Finset.range (n+1), 8 * ((10^k - 1) / 9)
  in sum = (8 * (10^(n+1) - 10 - 9 * n)) / 81 :=
by
  -- Placeholder for proof steps
  sorry

end sum_of_repeated_eights_l160_160988


namespace nice_people_count_l160_160216

theorem nice_people_count
  (barry_count kevin_count julie_count joe_count : ℕ)
  (nice_barry nice_kevin nice_julie nice_joe : ℕ) 
  (H1 : barry_count = 24)
  (H2 : kevin_count = 20)
  (H3 : julie_count = 80)
  (H4 : joe_count = 50)
  (H5 : nice_barry = barry_count)
  (H6 : nice_kevin = kevin_count / 2)
  (H7 : nice_julie = (3 * julie_count) / 4)
  (H8 : nice_joe = joe_count / 10)
  : nice_barry + nice_kevin + nice_julie + nice_joe = 99 :=
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8]
  norm_num
  sorry

end nice_people_count_l160_160216


namespace probability_of_a_winning_the_match_is_correct_distribution_and_expectation_of_xi_are_correct_l160_160561

noncomputable def probability_a_winning_match : ℝ :=
  let p_a : ℝ := 0.6
  let p_b : ℝ := 0.4
  let p_scenario1 : ℝ := p_a * p_a
  let p_scenario2 : ℝ := 2 * p_a * p_a * p_b
  p_scenario1 + p_scenario2

theorem probability_of_a_winning_the_match_is_correct :
  probability_a_winning_match = 0.648 := by
  sorry

noncomputable def expectation_of_xi : ℝ :=
  let p_a : ℝ := 0.6
  let p_b : ℝ := 0.4
  let p_xi_2 : ℝ := p_a * p_a + p_b * p_b
  let p_xi_3 : ℝ := 2 * p_a * p_a * p_b + 2 * p_b * p_b * p_a
  let expect_xi : ℝ := 2 * p_xi_2 + 3 * p_xi_3
  (p_xi_2, p_xi_3, expect_xi)

theorem distribution_and_expectation_of_xi_are_correct :
  expectation_of_xi = (0.52, 0.48, 2.48) := by
  sorry

end probability_of_a_winning_the_match_is_correct_distribution_and_expectation_of_xi_are_correct_l160_160561


namespace expression_equals_4096_l160_160603

noncomputable def calculate_expression : ℕ :=
  ((16^15 / 16^14)^3 * 8^3) / 2^9

theorem expression_equals_4096 : calculate_expression = 4096 :=
by {
  -- proof would go here
  sorry
}

end expression_equals_4096_l160_160603


namespace island_solution_l160_160581

-- Definitions based on conditions
def is_liar (n : ℕ) (m : ℕ) : Prop := n = m + 2 ∨ n = m - 2
def is_truth_teller (n : ℕ) (m : ℕ) : Prop := n = m

-- Residents' statements
def first_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1001 ∧ is_truth_teller truth_tellers 1002 ∨
  is_liar liars 1001 ∧ is_liar truth_tellers 1002

def second_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1000 ∧ is_truth_teller truth_tellers 999 ∨
  is_liar liars 1000 ∧ is_liar truth_tellers 999

-- Proving the correct number of liars and truth-tellers, and identifying the residents
theorem island_solution :
  ∃ (liars : ℕ) (truth_tellers : ℕ),
    first_resident_statement (liars + 1) (truth_tellers + 1) ∧
    second_resident_statement (liars + 1) (truth_tellers + 1) ∧
    liars = 1000 ∧ truth_tellers = 1000 ∧
    first_resident_statement liars truth_tellers ∧ second_resident_statement liars truth_tellers :=
by
  sorry

end island_solution_l160_160581


namespace traci_trip_distance_l160_160879

theorem traci_trip_distance :
  ∃ (D : ℝ), (1/3) * D + (1/5) * (2/3) * D + (1/4) * (8/15) * D + 400 = D ∧
             (2/5) * D = 400 := 
begin
  use 1000,
  split,
  { calc (1/3) * 1000 + (1/5) * (2/3) * 1000 + (1/4) * (8/15) * 1000 + 400 
      = 1000 : by norm_num },
  { calc (2/5) * 1000 = 400 : by norm_num }
end

end traci_trip_distance_l160_160879


namespace eight_points_chords_l160_160256

theorem eight_points_chords : (∃ n, n = 8) → (∃ m, m = 28) :=
by
  intro h
  have h1: (∃ n, n = (Nat.choose 8 2)) := by
    use Nat.choose 8 2
    exact Nat.choose_eq 8 2
  cases h1 with m hm
  use m
  exact hm

end eight_points_chords_l160_160256


namespace simplify_expression_l160_160081

variable (y : ℝ)

theorem simplify_expression : 
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + 3 * y ^ 8) = 
  15 * y ^ 13 - y ^ 12 + 3 * y ^ 11 + 15 * y ^ 10 - y ^ 9 - 6 * y ^ 8 :=
by
  sorry

end simplify_expression_l160_160081


namespace pure_imaginary_if_and_only_if_a_eq_neg2_l160_160287

open Complex

theorem pure_imaginary_if_and_only_if_a_eq_neg2 (a : ℝ) :
  (Im ((a + I) / (1 + 2 * I)) ≠ 0 ∧ Re ((a + I) / (1 + 2 * I)) = 0) ↔ 
  (a = -2) :=
by
  sorry

end pure_imaginary_if_and_only_if_a_eq_neg2_l160_160287


namespace probability_of_symmetry_line_l160_160806

-- Define the conditions of the problem.
def is_on_symmetry_line (P Q : (ℤ × ℤ)) :=
  (Q.fst = P.fst) ∨ (Q.snd = P.snd) ∨ (Q.fst - P.fst = Q.snd - P.snd) ∨ (Q.fst - P.fst = P.snd - Q.snd)

-- Define the main statement of the theorem to be proved.
theorem probability_of_symmetry_line :
  let grid_size := 11
  let total_points := grid_size * grid_size
  let center : (ℤ × ℤ) := (grid_size / 2, grid_size / 2)
  let other_points := total_points - 1
  let symmetric_points := 40
  /- Here we need to calculate the probability, which is the ratio of symmetric points to other points,
     and this should equal 1/3 -/
  (symmetric_points : ℚ) / other_points = 1 / 3 :=
by sorry

end probability_of_symmetry_line_l160_160806


namespace number_of_four_digit_multiples_of_7_l160_160691

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160691


namespace octagon_diagonals_l160_160344

theorem octagon_diagonals : 
  ∀ n : ℕ, n = 8 → (n * (n - 3)) / 2 = 20 :=
by
  intros n hn
  rw [hn]
  norm_num
  sorry

end octagon_diagonals_l160_160344


namespace min_distance_and_distance_from_Glafira_l160_160996

theorem min_distance_and_distance_from_Glafira 
  (U g τ V : ℝ) (h : 2 * U ≥ g * τ) :
  let T := (τ / 2) + (U / g) in
  s T = 0 ∧ (V * T = V * (τ / 2 + U / g)) :=
by
  -- Define the positions y1(t) and y2(t)
  let y1 := λ t, U * t - (g * t^2) / 2
  let y2 := λ t, U * (t - τ) - (g * (t - τ)^2) / 2
  -- Define the distance s(t)
  let s := λ t, |y1 t - y2 t|
  -- Start the proof
  sorry

end min_distance_and_distance_from_Glafira_l160_160996


namespace cost_milk_powder_proof_l160_160985

noncomputable def cost_milk_powder_in_july (C : ℝ) : ℝ :=
  0.2 * C

theorem cost_milk_powder_proof :
  ∀ (C : ℝ), (4 * 1.5 * C + 0.2 * 1.5 * C = 6.30) → cost_milk_powder_in_july C = 0.20 := by
  intro C h
  have : 4 * 1.5 * C + 0.2 * 1.5 * C = 6.30 := h
  have : 6.3C = 6.30 := by linarith
  have : C = 1 := by linarith
  rw [this, cost_milk_powder_in_july]
  norm_num
  sorry

end cost_milk_powder_proof_l160_160985


namespace remaining_credit_l160_160829

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l160_160829


namespace unique_solution_l160_160414

theorem unique_solution (a : ℝ) (n : ℕ) (x : Fin n → ℝ) (h1 : a > 0) (h2 : n > 4)
    (h3 : ∀ i : Fin n, 0 < x i ∧ 3 * a - 2 * x ((i + 1) % n.succ) > 0)
    (h4 : ∀ i : Fin n, x i * x ((i + 1) % n) * (3 * a - 2 * x ((i + 2) % n)) = a ^ 3) :
    ∀ i : Fin n, x i = a := 
by
  sorry

end unique_solution_l160_160414


namespace rhombus_area_correct_l160_160982

structure Point where
  x : ℝ
  y : ℝ

def rhombus_vertices : List Point := [
  {x := 0, y := 3.5},
  {x := 10, y := 0},
  {x := 0, y := -3.5},
  {x := -10, y := 0}
]

noncomputable def distance (p1 p2 : Point) : ℝ :=
  √((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def area_of_rhombus (v : List Point) : ℝ :=
  let d1 := distance v[0] v[2] -- distance between (0, 3.5) and (0, -3.5)
  let d2 := distance v[1] v[3] -- distance between (10, 0) and (-10, 0)
  (d1 * d2) / 2

theorem rhombus_area_correct :
  area_of_rhombus rhombus_vertices = 70 := by
  sorry

end rhombus_area_correct_l160_160982


namespace max_mondays_in_51_days_l160_160521

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l160_160521


namespace Carlos_hits_11_l160_160895

theorem Carlos_hits_11 (scores : List (String × Nat)) :
  let available_regions := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  let unique_scores := [
    ("Bella", 12),
    ("Diana", 15),
    ("Eva", 18),
    ("Felix", 21),
    ("Carlos", 28),
    ("Alex", 22)
  ]
  (forall player in unique_scores, 
    player.2 = (region1 + region2 + region3) ∧
    region1 ∈ available_regions ∧ 
    region2 ∈ available_regions ∧ 
    region3 ∈ available_regions ∧ 
    region1 ≠ region2 ∧ 
    region2 ≠ region3 ∧ 
    region1 ≠ region3) → scores ∈ unique_scores →
  ∃ player, player = "Carlos" ∧ 11 ∈ (score_regions player)
:=
by
  sorry

end Carlos_hits_11_l160_160895


namespace product_of_possible_values_of_x_l160_160788

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l160_160788


namespace function_increasing_intervals_l160_160472

theorem function_increasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y - x) < δ → f y > f x) ∨ 
  (∀ x : ℝ, ∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, abs (y - x) < δ ∧ f y < f x) :=
sorry

end function_increasing_intervals_l160_160472


namespace coloring_theorem_l160_160251

namespace ColoringProof

def Vertex := (ℕ × ℕ)

def color := Vertex → Prop -- Assume color is a proposition representing red/blue for simplicity

def unit_distance (v1 v2 : Vertex) : Prop :=
  abs (v1.1 - v2.1) + abs (v1.2 - v2.2) = 1

noncomputable def vertices : list Vertex :=
  [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

theorem coloring_theorem (coloring : Vertex → Prop) :
  ∃ (v1 v2 : Vertex), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 ∧ unit_distance v1 v2 ∧ (coloring v1 = coloring v2) :=
sorry

end ColoringProof

end coloring_theorem_l160_160251


namespace not_algorithm_c_l160_160973

-- Definitions based on conditions
def process_a : Prop := (∀ x : ℝ, 2 * x - 6 = 0 → (2 * x - 6 = 0 ∧ x = 3))
def process_b : Prop := true  -- We assume it is true, representing a process with clear steps
def process_c : Prop := ∀ x : ℝ, ¬(2 * x^2 + x - 1 = 0)  -- Representation that there are no steps
def process_d : Prop := (S = π * 3^2 → S = π * 9)

-- The main theorem to be proved, considering that process_c is not an algorithm
theorem not_algorithm_c : process_a ∧ process_b ∧ process_d → process_c := by
  sorry

end not_algorithm_c_l160_160973


namespace enclosed_area_l160_160916

theorem enclosed_area : ∀ (s : ℝ) (arc_length : ℝ),
  s = 3 →
  arc_length = π / 3 →
  (∃ r, arc_length = 2 * π * r / 6 ∧ r = 1 / 2) →
  (hexagon_area = 3 * sqrt 3 / 2 * s^2) →
  (sector_area = π / 24 * 12) →
  (enclosed_area = hexagon_area + sector_area) →

  enclosed_area = π / 2 + 27 * sqrt 3 / 2 :=
by
  intros s arc_length h_s h_arc h_r h_hex h_sector h_total,
  rw [h_s, h_arc, h_r] at *,
  sorry

end enclosed_area_l160_160916


namespace rhombus_side_and_inscribed_radius_l160_160880

variable {A B C D K : Point}
variable (AB AC BC CD DA : Line)
variable (r : ℝ)

-- Conditions
axiom rhombus_ABCD : rhombus ABCD
axiom point_on_AC : K ∈ AC
axiom distance_K_AB : distance K AB = 12
axiom distance_K_BC : distance K BC = 2
axiom radius_inscribed_triangle_ABC : radius_inscribed_triangle A B C = 5

-- Problem statement
theorem rhombus_side_and_inscribed_radius :
  (side_length_rhombus ABCD = (25 * sqrt 21) / 6) ∧ (radius_inscribed_rhombus ABCD = 7) :=
sorry

end rhombus_side_and_inscribed_radius_l160_160880


namespace construct_triangle_l160_160339

-- Define the points and lines
variables {Point : Type} [Geometry Point] 
variables (l1 l2 l3 : Line Point) (A1 O : Point)

-- Conditions given in the problem
axiom intersection_point : Intersect l1 l2 O ∧ Intersect l2 l3 O ∧ Intersect l3 l1 O
axiom A1_on_l1 : OnLine A1 l1
axiom perp_bisectors : PerpBisector l1 = Segment.mk (B, C) ∧ PerpBisector l2 = Segment.mk (C, A) ∧ PerpBisector l3 = Segment.mk (A, B)

-- Statement of the required proof
theorem construct_triangle : 
  ∃ A B C : Point, Midpoint A1 B C ∧ PerpBisector l1 = Segment.mk (B, C) ∧ PerpBisector l2 = Segment.mk (C, A) ∧ PerpBisector l3 = Segment.mk (A, B) 
  := 
sorry

end construct_triangle_l160_160339


namespace value_of_y_square_plus_inverse_square_l160_160347

variable {y : ℝ}
variable (h : 35 = y^4 + 1 / y^4)

theorem value_of_y_square_plus_inverse_square (h : 35 = y^4 + 1 / y^4) : y^2 + 1 / y^2 = Real.sqrt 37 := 
sorry

end value_of_y_square_plus_inverse_square_l160_160347


namespace percentage_of_water_in_mixture_l160_160169

-- Conditions
def percentage_water_LiquidA : ℝ := 0.10
def percentage_water_LiquidB : ℝ := 0.15
def percentage_water_LiquidC : ℝ := 0.25

def volume_LiquidA (v : ℝ) : ℝ := 4 * v
def volume_LiquidB (v : ℝ) : ℝ := 3 * v
def volume_LiquidC (v : ℝ) : ℝ := 2 * v

-- Proof
theorem percentage_of_water_in_mixture (v : ℝ) :
  (percentage_water_LiquidA * volume_LiquidA v + percentage_water_LiquidB * volume_LiquidB v + percentage_water_LiquidC * volume_LiquidC v) / (volume_LiquidA v + volume_LiquidB v + volume_LiquidC v) * 100 = 15 :=
by
  sorry

end percentage_of_water_in_mixture_l160_160169


namespace min_cost_cost_range_l160_160185

noncomputable def total_cost (x : ℝ) : ℝ := 90 * (900 / x) + 9 * x

theorem min_cost (x : ℝ) : 0 < x → 
  (∀ y : ℝ, 0 < y → total_cost x ≤ total_cost y) → x = 30 :=
begin
  sorry
end

theorem cost_range (x : ℝ) : 20 ≤ x ∧ x ≤ 45 ↔ total_cost x ≤ 585 :=
begin
  sorry
end

end min_cost_cost_range_l160_160185


namespace num_four_digit_multiples_of_7_l160_160720

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160720


namespace four_digit_multiples_of_7_count_l160_160760

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160760


namespace fraction_product_l160_160604

theorem fraction_product :
  (2 / 3) * (5 / 7) * (9 / 11) * (4 / 13) = 360 / 3003 := by
  sorry

end fraction_product_l160_160604


namespace sum_of_digits_9ab_l160_160815

theorem sum_of_digits_9ab {a b : ℕ} (h_a : a = 6 * (10^2023 - 1) / 9) (h_b : b = 4 * (10^2023 - 1) / 9) :
  (sum_of_digits (9 * a * b) = 20225) :=
sorry

-- Needed helper function to handle sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  -- Placeholder definition
  -- The real implementation would sum the digits of n in base 10
  sorry

end sum_of_digits_9ab_l160_160815


namespace wire_pieces_lengths_l160_160591

-- Define the length of the wire and the condition of one piece being 2 feet longer than the other
variables (total_length : ℝ) (shorter longer : ℝ)
hypothesis h1 : total_length = 30
hypothesis h2 : longer = shorter + 2

-- Define the sum of the lengths of the two pieces being equal to the total length
hypothesis h3 : shorter + longer = total_length

-- Prove that the lengths are 14 feet and 16 feet, respectively
theorem wire_pieces_lengths : shorter = 14 ∧ longer = 16 :=
by
  sorry

end wire_pieces_lengths_l160_160591


namespace sheila_hours_mon_wed_fri_l160_160445

variables (H : ℕ)  -- the number of hours Sheila works on Monday, Wednesday, and Friday
variables (earnings_per_week : ℕ := 252)
variables (hourly_rate : ℕ := 7)
variables (tue_thu_hours_per_day : ℕ := 6)
variables (work_days_tue_thu : ℕ := 2)

theorem sheila_hours_mon_wed_fri :
  let total_earnings_tue_thu := tue_thu_hours_per_day * work_days_tue_thu * hourly_rate in
  let total_earnings_mon_wed_fri := earnings_per_week - total_earnings_tue_thu in
  total_earnings_mon_wed_fri / hourly_rate = 24 :=
by
  -- placeholder for the actual proof
  sorry

end sheila_hours_mon_wed_fri_l160_160445


namespace tyler_saltwater_animals_l160_160689

theorem tyler_saltwater_animals (aquariums : ℕ) (animals_per_aquarium : ℕ)
  (h1 : aquariums = 8) (h2 : animals_per_aquarium = 64) :
  (aquariums * animals_per_aquarium) = 512 :=
by
  rw [h1, h2]
  norm_num

end tyler_saltwater_animals_l160_160689


namespace range_of_a_l160_160796

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f x ≤ f y) ↔ (-1/4 ≤ a ∧ a ≤ 0) :=
  sorry
  where f (x : ℝ) := a * x^2 + 2 * x - 3

end range_of_a_l160_160796


namespace solitaire_win_l160_160517

theorem solitaire_win (deck : List ℕ) (h_len : deck.length = 13) (h_deck : ∀ x ∈ deck, x ∈ Finset.range 1 14) :
  ∃ seq, transform deck seq = [1] ++ _ :=
sorry

end solitaire_win_l160_160517


namespace solve_for_y_l160_160896

theorem solve_for_y (y : ℤ) (h : 3^(y - 2) = 9^(y + 3)) : y = -8 :=
  sorry

end solve_for_y_l160_160896


namespace triangle_centroid_eq_l160_160028

-- Define the proof problem
theorem triangle_centroid_eq
  (P Q R G : ℝ × ℝ) -- Points P, Q, R, and G (the centroid of the triangle PQR)
  (centroid_eq : G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) -- Condition that G is the centroid
  (gp_sq_gq_sq_gr_sq_eq : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 22) -- Given GP^2 + GQ^2 + GR^2 = 22
  : dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 66 := -- Prove PQ^2 + PR^2 + QR^2 = 66
sorry -- Proof is omitted

end triangle_centroid_eq_l160_160028


namespace no_m_increases_probability_l160_160143
open Set

/-- Define the set T -/
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem no_m_increases_probability :
  ∀ m ∈ T, (∃ a b ∈ T \ {m}, a ≠ b ∧ a + b = 11 → (T \ {m}).InjOn (*)) →
  (0.0 < probability (fun (x y : ℕ) => x ∈ (T \ {m}) ∧ y ∈ (T \ {m}) ∧ x ≠ y ∧ x + y = 11)) :=
sorry

end no_m_increases_probability_l160_160143


namespace summation_identity_is_two_l160_160794

noncomputable def summation_identity (a_k : ℕ → ℂ) : Prop :=
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 3) in
  (ω^3 = 1 ∧ (1 + ω + ω^2 = 0)) ∧
  (∀ x : ℂ, (∑ k in finset.range 4035, a_k k * x^k) = (x^2 + x + 2)^2017) →
  (∑ k in finset.range 1345, (2 * a_k (3 * k) - a_k (3 * k + 1) - a_k (3 * k + 2))) = 2

-- By stating a theorem placeholder, where a_k is a parameter for the function sequence
theorem summation_identity_is_two (a_k : ℕ → ℂ) : summation_identity a_k :=
  sorry

end summation_identity_is_two_l160_160794


namespace cistern_fill_time_l160_160565

variables (C : ℝ) (capacity : ℝ)

-- Assume that Pipe A fills the cistern in 10 hours
def rate_A : ℝ := C / 10

-- Assume that Pipe B empties the cistern in 12 hours
def rate_B : ℝ := C / 12

-- Define the net filling rate when both pipes are open
def net_rate : ℝ := rate_A - rate_B

-- The time it takes to fill the cistern at the net rate
def fill_time : ℝ := C / net_rate

-- Theorems stating that the time to fill the cistern is 60 hours
theorem cistern_fill_time
  (rate_A_def : rate_A = C / 10)
  (rate_B_def : rate_B = C / 12)
  (net_rate_def : net_rate = C / 60)
  (fill_time_def : fill_time = C * (1 / (C / 60))) :
  fill_time = 60 :=
by
  sorry

end cistern_fill_time_l160_160565


namespace probability_B_wins_at_least_one_match_l160_160367

theorem probability_B_wins_at_least_one_match :
  let P_A := 0.5
  let P_B := 0.3
  let P_T := 0.2
  let P_B_not_winning := 1 - P_B
  let P_B_wins_at_least_one := (P_B * P_B) + (P_B * P_B_not_winning) + (P_B_not_winning * P_B)
  P_B_wins_at_least_one = 0.51 :=
by
  let P_A := 0.5
  let P_B := 0.3
  let P_T := 0.2
  let P_B_not_winning := 1 - P_B
  let P_B_wins_at_least_one := (P_B * P_B) + (P_B * P_B_not_winning) + (P_B_not_winning * P_B)
  show P_B_wins_at_least_one = 0.51
  sorry

end probability_B_wins_at_least_one_match_l160_160367


namespace Ganesh_average_speed_l160_160551

noncomputable def average_speed (D : ℝ) : ℝ :=
  let time_xy := D / 43
  let time_yx := D / 35
  let total_distance := 2 * D
  let total_time := time_xy + time_yx
  total_distance / total_time

theorem Ganesh_average_speed :
  (∀ D : ℝ, D > 0 → average_speed D = 71.67) :=
by
  intro D hD
  dsimp [average_speed]
  field_simp
  have h : (2 * D) / (D * (1 / 43 + 1 / 35)) = (43 * 35) / 39 := by
    ring
    linarith
  rw [h]
  norm_num
  exact sorry

#print axioms Ganesh_average_speed

end Ganesh_average_speed_l160_160551


namespace existence_of_solution_l160_160628

theorem existence_of_solution {b : ℝ} : 
  (∀ a : ℝ, ∃ x y : ℝ, x * cos a + y * sin a + 4 ≤ 0 ∧ x^2 + y^2 + 10 * x + 2 * y - b^2 - 8 * b + 10 = 0) ↔ 
  b ≤ -8 - sqrt 26 ∨ b ≥ sqrt 26 := 
sorry

end existence_of_solution_l160_160628


namespace cos_of_angle_through_point_l160_160326

-- Define the point P and the angle α
def P : ℝ × ℝ := (4, 3)
def α : ℝ := sorry  -- α is an angle such that its terminal side passes through P

-- Define the squared distance from the origin to the point P
noncomputable def distance_squared : ℝ := P.1^2 + P.2^2

-- Define cos α
noncomputable def cosα : ℝ := P.1 / (Real.sqrt distance_squared)

-- State the theorem
theorem cos_of_angle_through_point : cosα = 4 / 5 := 
by sorry

end cos_of_angle_through_point_l160_160326


namespace cosine_square_sum_l160_160241

theorem cosine_square_sum :
  (∑ k in finset.filter (λ i, odd i) (finset.range 180), (cos (real.to_rad ((k : ℝ) * 5)))^2) = 21 :=
sorry

end cosine_square_sum_l160_160241


namespace count_four_digit_multiples_of_7_l160_160739

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160739


namespace apple_cost_correct_l160_160898

-- Define all the variables
def Total_cost : ℝ := 25
def Milk_cost : ℝ := 3
def Cereal_cost : ℝ := 3.5
def Banana_cost : ℝ := 0.25
def Num_apples : ℕ := 4
def Cookie_cost : ℝ := 2 * Milk_cost
def Num_cookies : ℕ := 2

-- Calculate intermediate values
def Total_known_cost : ℝ :=
  Milk_cost + (2 * Cereal_cost) + (4 * Banana_cost) + (Num_cookies * Cookie_cost)

def Remaining_cost : ℝ := Total_cost - Total_known_cost

def Cost_per_apple : ℝ := Remaining_cost / Num_apples

-- State the main theorem
theorem apple_cost_correct : Cost_per_apple = 0.5 :=
  by
  sorry

end apple_cost_correct_l160_160898


namespace isosceles_triangle_divide_height_l160_160374

theorem isosceles_triangle_divide_height (A B C D E : Point)
    (h_iso : distance A B = distance B C)
    (h_point_D : ∃ k : ℝ, B + k * (C - B) = D ∧ k = 1 / 5)
    (E_midpoint : E = midpoint A C)
    (BE_perpendicular_AC : ∃ h : line, h = line B E) :
    ∃ k : ℝ, on_line (line AD) E ∧ (distance B A * distance E D)/(distance B E * distance D A) = 1 / 2 :=
by
  sorry

end isosceles_triangle_divide_height_l160_160374


namespace hyperbola_eccentricity_l160_160681

/-- Statement: The eccentricity of a hyperbola with given conditions is \(\frac{\sqrt{37}}{5}\) -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let c := Real.sqrt (a^2 + b^2),
      e := c / a,
      P Q : ℝ × ℝ := sorry in
  ∃ P Q : ℝ × ℝ,
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
  (Q.1^2 / a^2 - Q.2^2 / b^2 = 1) ∧
  (P.1 = Q.1 ∨ P.2 = Q.2) ∧
  (c = Real.sqrt (a^2 + b^2)) ∧
  (|Q.1 - P.1| = 5/12 * |(P.1 + c)|) →
  e = Real.sqrt 37 / 5 :=
by sorry

end hyperbola_eccentricity_l160_160681


namespace least_possible_d_l160_160903

def has_n_factors (n : ℕ) (k : ℕ) : Prop := 
  (finset.filter (λ i : ℕ, k % i = 0) (finset.range (k+1))).card = n

theorem least_possible_d (c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) 
  (h3 : has_n_factors 4 c) (h4 : has_n_factors c d) (h5 : d % c = 0) : 
  d = 12 :=
sorry

end least_possible_d_l160_160903


namespace range_of_a_l160_160155

-- Define the conditions
def f (a x : ℝ) : ℝ := a * x^2 + 4 * (a - 1) * x - 3

def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the property to check if maximum is attained at x=2
def has_max_at_2 (a : ℝ) : Prop := 
  ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), ∀ y ∈ set.Icc (0 : ℝ) (2 : ℝ), f a y ≤ f a x ∧ x = 2

-- Define the final theorem statement
theorem range_of_a : 
  (∀ x, in_interval x → f a x ≤ f a 2) ↔ (a ∈ set.Ici (2/3 : ℝ)) :=
sorry

end range_of_a_l160_160155


namespace rope_lengths_ratio_l160_160921

theorem rope_lengths_ratio (A B C : ℕ) (hC : C = 80) (hSUM : A + C = B + 100) :
  ratio A B C = (B + 20) / B / 80 :=
by {
  sorry
}

end rope_lengths_ratio_l160_160921


namespace max_value_of_xy_l160_160313

noncomputable def max_xy (x y : ℝ) (h1 : x ∈ Set.Ioi 0) (h2 : y ∈ Set.Ioi 0) (h3 : 3 * x + 2 * y = 12) : ℝ :=
  xy := x * y 
  xy

theorem max_value_of_xy : ∃ x y ∈ ℝ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 12 ∧ x * y = 6 :=
by
  sorry

end max_value_of_xy_l160_160313


namespace max_distance_for_g_l160_160473

noncomputable def g (x : ℝ) : ℝ := sin(2 * x + π / 3) + 2

theorem max_distance_for_g (x₁ x₂ : ℝ) (h₁ : x₁ ∈ Icc (-2 * π) (2 * π)) (h₂ : x₂ ∈ Icc (-2 * π) (2 * π)) 
  (h3 : g x₁ * g x₂ = 9) : |x₁ - x₂| ≤ 3 * π :=
sorry

end max_distance_for_g_l160_160473


namespace tg_alpha_over_alpha_lt_tg_beta_over_beta_l160_160440

noncomputable theory -- Necessary because trigonometric functions are involved

open Real -- open the real number namespace for convenience

theorem tg_alpha_over_alpha_lt_tg_beta_over_beta
  {α β : ℝ}
  (hα_pos : 0 < α)
  (hα_acute : α < π / 2)
  (hβ_pos : 0 < β)
  (hβ_acute : β < π / 2)
  (hα_lt_β : α < β) :
  (tan α) / α < (tan β) / β := 
sorry

end tg_alpha_over_alpha_lt_tg_beta_over_beta_l160_160440


namespace remaining_slices_l160_160214

theorem remaining_slices (first_total second_total : nat)
                          (first_given_away second_given_away
                           first_given_family second_given_family
                           first_alex_eats second_alex_eats : nat) :
  first_total = 8 →
  second_total = 12 →
  first_given_away = first_total / 4 →
  second_given_away = second_total / 3 →
  first_given_family = (first_total - first_given_away) / 2 →
  second_given_family = (second_total - second_given_away) / 2 →
  first_alex_eats ≤ first_total - first_given_away - first_given_family →
  second_alex_eats ≤ second_total - second_given_away - second_given_family →
  0 ≤ first_total - first_given_away - first_given_family - first_alex_eats →
  0 ≤ second_total - second_given_away - second_given_family - second_alex_eats →
  (first_total - first_given_away - first_given_family - first_alex_eats) +
  (second_total - second_given_away - second_given_family - second_alex_eats) = 2 :=
by
  intros
  sorry

end remaining_slices_l160_160214


namespace find_principal_amount_l160_160580

theorem find_principal_amount :
  ∃ (P : ℝ), P * Real.exp (0.1) = 5292 ∧ P = 4788.49 :=
by
  have exp_val : Real.exp (0.1) = 1.105170918 := sorry
  exact ⟨4788.49, by simp [exp_val], rfl⟩

end find_principal_amount_l160_160580


namespace collinear_O_P_Q_l160_160582

-- Given: Triangle ABC inscribed in a circle ω with center O
variables 
  (A B C O ω : Type) [MetricSpace O]
  (h0 : ∃ (circumcircle : Circle O), ∀ A B C, IsInscribed O circumcircle (Triangle.mk A B C))
  -- Line AO intersects the circle ω again at point A'
  (A' : O)
  (h1 : Line A O ∩ (Circle O) = {A, A'})
  -- MB and MC are midpoints of AC and AB respectively
  (MB MC : O)
  (h2 : MB = midpoint A C ∧ MC = midpoint A B)
  -- Lines A'M_B and A'M_C intersect the circle ω again at B' and C', and intersect side BC at D_B and D_C
  (B' C' DB DC : O)
  (h3 : Line A' MB ∩ Circle O = {MB, B'} ∧ Line A' MC ∩ Circle O = {MC, C'})
  (h4 : Line A' MB ∩ Segment B C = {DB} ∧ Line A' MC ∩ Segment B C = {DC})
  -- The circumcircles of CD_B B' and BD_C C' intersect at points P and Q
  (P Q : O)
  (h5 : ∃ (circumcircle1 circumcircle2 : Circle O), 
        circumcircle1 = Circumcircle (Triangle.mk C DB B') ∧ 
        circumcircle2 = Circumcircle (Triangle.mk B DC C') ∧ 
        PointsOnBothCircumcircles P circumcircle1 circumcircle2 ∧
        PointsOnBothCircumcircles Q circumcircle1 circumcircle2)

-- To prove: O, P, and Q are collinear
theorem collinear_O_P_Q 
  (h_collinear : Collinear O P Q) : 
  collinear O P Q := sorry

end collinear_O_P_Q_l160_160582


namespace number_divisibility_l160_160079

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility_l160_160079


namespace total_trip_time_l160_160140

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l160_160140


namespace f_prime_at_zero_l160_160007

noncomputable def a : ℕ → ℝ := λ n, 2 ^ (↑n - 1) / 2
def g(x : ℝ) : ℝ := (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)
def f(x : ℝ) : ℝ := x * g(x)
def f_prime(x : ℝ) : ℝ := g(x) + x * (derivative g x)

theorem f_prime_at_zero : f_prime 0 = 2^8 := 
by 
  sorry

end f_prime_at_zero_l160_160007


namespace number_is_165_l160_160534

def is_between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def is_odd (n : ℕ) : Prop := n % 2 = 1
def contains_digit_5 (n : ℕ) : Prop := ∃ k : ℕ, 10^k * 5 ≤ n ∧ n < 10^(k+1) * 5
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem number_is_165 : 
  (is_between 165 144 169) ∧ 
  (is_odd 165) ∧ 
  (contains_digit_5 165) ∧ 
  (is_divisible_by_3 165) :=
by 
  sorry 

end number_is_165_l160_160534


namespace problem_solution_l160_160309

theorem problem_solution
  (a b c : ℕ)
  (h_pos_a : 0 < a ∧ a ≤ 10)
  (h_pos_b : 0 < b ∧ b ≤ 10)
  (h_pos_c : 0 < c ∧ c ≤ 10)
  (h1 : abc % 11 = 2)
  (h2 : 7 * c % 11 = 3)
  (h3 : 8 * b % 11 = 4 + b % 11) : 
  (a + b + c) % 11 = 0 := 
by
  sorry

end problem_solution_l160_160309


namespace apples_per_pie_l160_160101

theorem apples_per_pie
  (total_apples : ℕ) (apples_handed_out : ℕ) (remaining_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 96)
  (h2 : apples_handed_out = 42)
  (h3 : remaining_apples = total_apples - apples_handed_out)
  (h4 : remaining_apples = 54)
  (h5 : number_of_pies = 9) :
  remaining_apples / number_of_pies = 6 := by
  sorry

end apples_per_pie_l160_160101


namespace imaginary_part_of_z_l160_160188

-- Define the problem conditions and what to prove
theorem imaginary_part_of_z (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_z_l160_160188


namespace axis_of_symmetry_l160_160100

theorem axis_of_symmetry (k : ℤ) : ∃ x, y = cos^2 x + sin x * cos x ∧ (x = k * π / 2 + π / 8) :=
begin
  sorry
end

end axis_of_symmetry_l160_160100


namespace vertex_difference_l160_160911

theorem vertex_difference (n m : ℝ) : 
  ∀ x : ℝ, (∀ x, -x^2 + 2*x + n = -((x - m)^2) + 1) → m - n = 1 := 
by 
  sorry

end vertex_difference_l160_160911


namespace cone_b_height_l160_160801

theorem cone_b_height : 
  ∀ (b : ℝ), 
    (∀ h_A h_B : ℝ, h_A = 20 - b → h_B = b → 
    ∀ r : ℝ, r = 3 →
    ∀ V_A V_B : ℝ, V_A = (1/3) * real.pi * r^2 * h_A → V_B = (1/3) * real.pi * r^2 * h_B →
    V_A / V_B = 3 → 
    b = 5) := 
sorry

end cone_b_height_l160_160801


namespace false_proposition_4_l160_160010

namespace ComplexOrder

definition sequence_relation (z1 z2 : ℂ) : Prop :=
  (z1.re > z2.re) ∨ (z1.re = z2.re ∧ z1.im > z2.im)

theorem false_proposition_4 : ∃ (z1 z2 z : ℂ), sequence_relation z1 z2 
                          ∧ sequence_relation z 0 ∧ ¬sequence_relation (z * z1) (z * z2) :=
by 
  let z1 : ℂ := 3 * Complex.I
  let z2 : ℂ := 2 * Complex.I
  let z  : ℂ := 2 * Complex.I
  use z1, z2, z
  split
  · left
    exact by norm_num
  split
  · left
    exact by norm_num
  · intro H
    have H1 : -6.re = 0.re := by norm_num
    have H2 : -4.re = 0.re := by norm_num
    cases H
    · linarith
    · norm_num at H
      contradiction

end false_proposition_4_l160_160010


namespace correct_calculation_is_d_l160_160968

theorem correct_calculation_is_d :
  (-7) + (-7) ≠ 0 ∧
  ((-1 / 10) - (1 / 10)) ≠ 0 ∧
  (0 + (-101)) ≠ 101 ∧
  (1 / 3 + -1 / 2 = -1 / 6) :=
by
  sorry

end correct_calculation_is_d_l160_160968


namespace right_square_prism_surface_area_l160_160578

theorem right_square_prism_surface_area (r h : ℝ) (a : ℝ) 
  (h_r : r = sqrt 6)
  (h_h : h = 4)
  (h_prism_in_sphere : 2 * a^2 + h^2 = (2 * r)^2) :
  2 * a^2 + 4 * a * h = 40 :=
by
  -- Unwrapping the conditions
  have ha : 2 * a^2 + 16 = 24, from by
    rw [h_r, h_h] at h_prism_in_sphere
    exact h_prism_in_sphere
  -- Solving for a directly inferred from the Pythagorean simplification
  have eqn1 : a = 2, from (eq_of_add_eq_add_right (eq_of_sub_eq_zero (ha.trans (by norm_num))))
  -- Substituting back to calculate the surface area
  rw eqn1
  norm_num
  sorry -- This sorry is to indicate the proof is omitted

end right_square_prism_surface_area_l160_160578


namespace initial_average_mark_l160_160463

-- Define the initial conditions
def num_students : ℕ := 9
def excluded_students_avg : ℕ := 44
def remaining_students_avg : ℕ := 80

-- Define the variables for total marks we calculated in the solution
def total_marks_initial := num_students * (num_students * excluded_students_avg / 5 + remaining_students_avg / (num_students - 5) * (num_students - 5))

-- The theorem we need to prove:
theorem initial_average_mark :
  (num_students * (excluded_students_avg * 5 + remaining_students_avg * (num_students - 5))) / num_students = 60 := 
  by
  -- step-by-step solution proof could go here, but we use sorry as placeholder
  sorry

end initial_average_mark_l160_160463


namespace four_digit_multiples_of_7_l160_160703

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160703


namespace sum_of_digits_of_product_l160_160274

theorem sum_of_digits_of_product (n : ℕ) :
  let product := ∏ i in List.range (n + 1), 10^(2^i) - 1
  let final_num := 10^(2^(n+1) - 1) - 1
  (product.digits.sum = 9 * (2^(n+1) - 1)) :=
by
  sorry

end sum_of_digits_of_product_l160_160274


namespace plot_length_l160_160922

variable (b length : ℝ)

theorem plot_length (h1 : length = b + 10)
  (fence_N_cost : ℝ := 26.50 * (b + 10))
  (fence_E_cost : ℝ := 32 * b)
  (fence_S_cost : ℝ := 22 * (b + 10))
  (fence_W_cost : ℝ := 30 * b)
  (total_cost : ℝ := fence_N_cost + fence_E_cost + fence_S_cost + fence_W_cost)
  (h2 : 1.05 * total_cost = 7500) :
  length = 70.25 := by
  sorry

end plot_length_l160_160922


namespace unknown_number_is_six_l160_160643

theorem unknown_number_is_six (n : ℝ) (h : 12 * n^4 / 432 = 36) : n = 6 :=
by 
  -- This will be the placeholder for the proof
  sorry

end unknown_number_is_six_l160_160643


namespace angle_OPQ_30_degrees_l160_160651

-- Definitions used in the theorem
def is_regular_18_gon (O : Point) (V : Fin 18 → Point) : Prop :=
  ∀ i, Distance (V i) O = Distance (V 0) O ∧ ∠ (V i) O (V (i + 1) % 18) = 360 / 18

def midpoint (A B P: Point) : Prop :=
  Distance A P = Distance B P ∧ Distance A B = 2 * Distance A P

-- The theorem statement
theorem angle_OPQ_30_degrees 
(O A B C D P Q : Point)
(h_regular : ∃ V : Fin 18 → Point, V 0 = A ∧ V 1 = B ∧ V 2 = C ∧ V 3 = D ∧ is_regular_18_gon O V)
(h_midpoint_P : midpoint A C P)
(h_midpoint_Q : midpoint O D Q) 
: ∠ O P Q = 30 :=
sorry

end angle_OPQ_30_degrees_l160_160651


namespace equidistant_point_on_y_axis_l160_160813

-- Define the points P, C, and M
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

def P : Point3D := {x := 0, y := 0, z := real.sqrt 3}
def C : Point3D := {x := -1, y := 2, z := 0}

noncomputable def M (y : ℝ) : Point3D := {x := 0, y := y, z := 0}

theorem equidistant_point_on_y_axis :
  ∃ y, y = 1 / 2 ∧ distance (M y) P = distance (M y) C :=
by
  sorry

end equidistant_point_on_y_axis_l160_160813


namespace radius_integer_iff_p_eq_3_l160_160853

theorem radius_integer_iff_p_eq_3 (p : ℕ) (hp : Prime p) (h3 : p ≥ 3) :
  let longer_leg := p^2 - 1,
      shorter_leg := 2 * p,
      hypotenuse := (p^2 + 1) in
  ∃ r : ℕ, semicircle_inscribed_radius longer_leg shorter_leg hypotenuse r → r ∈ ℕ → p = 3 :=
by
  -- Skipping proof steps
  sorry

end radius_integer_iff_p_eq_3_l160_160853


namespace inequality_am_gm_cauchy_schwarz_equality_iff_l160_160447

theorem inequality_am_gm_cauchy_schwarz 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_iff (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) 
  ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_am_gm_cauchy_schwarz_equality_iff_l160_160447


namespace andrey_gifts_l160_160062

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l160_160062


namespace max_value_expr_l160_160405

theorem max_value_expr (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum : a + b + c = 2) :
  ∃ M, (∀ a b c, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 2 → (frac_expr a b c) ≤ M) ∧ (exists a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 ∧ (frac_expr a b c) = M) :=
by
  have frac_expr := (λ a b c, (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c))
  sorry

end max_value_expr_l160_160405


namespace intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l160_160671

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt (x + 2))) + Real.log (3 - x)
def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 1 - m < x ∧ x < 3 * m - 1 }

theorem intersection_of_A_and_B_is_B_implies_m_leq_4_over_3 (m : ℝ) 
    (h : A ∩ B m = B m) : m ≤ 4 / 3 := by
  sorry

end intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l160_160671


namespace four_digit_multiples_of_7_count_l160_160755

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160755


namespace square_pieces_placement_l160_160068

theorem square_pieces_placement (n : ℕ) (H : n = 8) :
  {m : ℕ // m = 17} :=
sorry

end square_pieces_placement_l160_160068


namespace strictly_decreasing_interval_l160_160292

-- Definitions of the given conditions
variable {x : ℝ}
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * (x - 5)^2

-- Statement of the problem where we need to prove where f'(x) < 0 implies x < 2
theorem strictly_decreasing_interval : 
  { x : ℝ | f'(x) < 0 } = { x : ℝ | x < 2 } :=
sorry

end strictly_decreasing_interval_l160_160292


namespace T1_acute_angled_right_triangle_in_sequence_T3_similar_to_T_number_of_non_similar_T_l160_160553

def is_acute_angled (α β γ : ℝ) : Prop :=
  (π / 4 < α ∧ α < π / 2) ∧ (π / 4 < β ∧ β < π / 2) ∧ (π / 4 < γ ∧ γ < π / 2) 
  ∨ (α < π / 4 ∧ β < π / 4 ∧ γ < 3 * π / 4)

def is_right_triangle (α β γ : ℝ) (n : ℕ) : Prop :=
  ∃ s : ℕ, α = π * s / 2^n ∨ β = π * s / 2^n ∨ γ = π * s / 2^n

def is_similar (α β γ α₃ β₃ γ₃ : ℝ) : Prop :=
  (α = α₃ ∧ β = β₃ ∧ γ = γ₃) ∨
  (α = β₃ ∧ β = γ₃ ∧ γ = α₃) ∨
  (α = γ₃ ∧ β = α₃ ∧ γ = β₃)

def number_of_non_similar_triangles (n : ℕ) : ℕ :=
  2^(2*n) - 2^n

theorem T1_acute_angled (α β γ : ℝ) (h₀ : non_right_triangle α β γ) :
  is_acute_angled α β γ :=
sorry

theorem right_triangle_in_sequence (α β γ : ℝ) (n : ℕ) (h₀ : non_right_triangle α β γ) :
  is_right_triangle α β γ n :=
sorry

theorem T3_similar_to_T (α β γ α₃ β₃ γ₃ : ℝ) (h₀ : non_right_triangle α β γ) :
  is_similar α β γ α₃ β₃ γ₃ :=
sorry

theorem number_of_non_similar_T (n : ℕ) :
  number_of_non_similar_triangles n :=
sorry

end T1_acute_angled_right_triangle_in_sequence_T3_similar_to_T_number_of_non_similar_T_l160_160553


namespace no_integer_solutions_l160_160070

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147 :=
by
  sorry

end no_integer_solutions_l160_160070


namespace complex_expression_evaluation_l160_160845

theorem complex_expression_evaluation (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := 
sorry

end complex_expression_evaluation_l160_160845


namespace set_inter_complement_l160_160418

open Set

theorem set_inter_complement (U A B : Set ℕ) (hU : U = {-1, 0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 5}) (hB : B = {0, 1, 2, 3}) :
  B ∩ (U \ A) = {0, 3} :=
by
  sorry

end set_inter_complement_l160_160418


namespace dan_time_second_hour_tshirts_l160_160596

-- Definition of conditions
def t_shirts_in_first_hour (rate1 : ℕ) (time : ℕ) : ℕ := time / rate1
def total_t_shirts (hour1_ts hour2_ts : ℕ) : ℕ := hour1_ts + hour2_ts
def time_per_t_shirt_in_second_hour (time : ℕ) (hour2_ts : ℕ) : ℕ := time / hour2_ts

-- Main theorem statement (without proof)
theorem dan_time_second_hour_tshirts
  (rate1 : ℕ) (hour1_time : ℕ) (total_ts : ℕ) (hour_time : ℕ)
  (hour1_ts := t_shirts_in_first_hour rate1 hour1_time)
  (hour2_ts := total_ts - hour1_ts) :
  rate1 = 12 → 
  hour1_time = 60 → 
  total_ts = 15 → 
  hour_time = 60 →
  time_per_t_shirt_in_second_hour hour_time hour2_ts = 6 :=
by
  intros rate1_eq hour1_time_eq total_ts_eq hour_time_eq
  sorry

end dan_time_second_hour_tshirts_l160_160596


namespace optimal_days_for_debit_card_l160_160456

theorem optimal_days_for_debit_card:
  let ticket_cost: ℝ := 12000
  let cashback_credit: ℝ := 0.01 * ticket_cost
  let cashback_debit: ℝ := 0.02 * ticket_cost
  let annual_interest_rate: ℝ := 0.06
  let daily_interest_rate: ℝ := annual_interest_rate / 365
  let benefit_credit := λ N: ℝ, N * (ticket_cost * daily_interest_rate) + cashback_credit
  let benefit_debit := cashback_debit
  ∀ N: ℝ, benefit_debit ≥ benefit_credit N → N ≤ 6 := 
by {
  sorry
}

end optimal_days_for_debit_card_l160_160456


namespace sibling_grouping_count_l160_160942

theorem sibling_grouping_count : 
  let pairs := 4
  let total_children := 8
  let group_min_size := 2
  ∃ (group_count : ℕ) (group_sizes : List ℕ), 
    group_count = 3 ∧
    (∀ size ∈ group_sizes, size ≥ group_min_size) ∧
    (∑ size in group_sizes, size = total_children) ∧
    (∀ (child_pairs : List (ℕ × ℕ)), 
      length child_pairs = pairs →
      ∀ (grp_assignment : List (Fin group_count)), 
      (∀ (i j : Fin group_count), i ≠ j → grp_assignment[i] ≠ grp_assignment[j]) →
      (∀ (pair ∈ child_pairs), grp_assignment[pair.1] ≠ grp_assignment[pair.2])) → 
    (number_of_groupings total_children group_count group_min_size = 144) :=
by
  sorry

end sibling_grouping_count_l160_160942


namespace satisfies_conditions_l160_160161

noncomputable def f (x : ℝ) : ℝ := -x * Real.exp (abs x)

theorem satisfies_conditions (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x + f (-x) = 0) ∧ 
  (∀ x : ℝ, f' x ≤ 0) :=
sorry

end satisfies_conditions_l160_160161


namespace divides_y_l160_160413

theorem divides_y
  (x y : ℤ)
  (h1 : 2 * x + 1 ∣ 8 * y) : 
  2 * x + 1 ∣ y :=
sorry

end divides_y_l160_160413


namespace find_divided_number_l160_160878

theorem find_divided_number :
  ∃ (Number : ℕ), ∃ (q r d : ℕ), q = 8 ∧ r = 3 ∧ d = 21 ∧ Number = d * q + r ∧ Number = 171 :=
by
  sorry

end find_divided_number_l160_160878


namespace find_shop_width_l160_160929

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end find_shop_width_l160_160929


namespace max_omega_condition_l160_160660

noncomputable def max_omega : ℝ :=
  if h : (∀ x y : ℝ, (0 ≤ x ∧ x ≤ y ∧ y ≤ π/3) → 2 * sin (ω * x) ≤ 2 * sin (ω * y)) then 3 / 2 else 0

theorem max_omega_condition (ω : ℝ) (h : ω > 0) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ y ∧ y ≤ π/3) → 2 * sin (ω * x) ≤ 2 * sin (ω * y))
  → ω ≤ 3 / 2 :=
sorry

end max_omega_condition_l160_160660


namespace r_investment_time_l160_160128

variables (P Q R Profit_p Profit_q Profit_r Tp Tq Tr : ℕ)
variables (h1 : P / Q = 7 / 5)
variables (h2 : Q / R = 5 / 4)
variables (h3 : Profit_p / Profit_q = 7 / 10)
variables (h4 : Profit_p / Profit_r = 7 / 8)
variables (h5 : Tp = 2)
variables (h6 : Tq = t)

theorem r_investment_time (t : ℕ) :
  ∃ Tr : ℕ, Tr = 4 :=
sorry

end r_investment_time_l160_160128


namespace find_coordinates_M_l160_160647

open Real

theorem find_coordinates_M (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℝ) :
  ∃ (xM yM zM : ℝ), 
  xM = (x1 + x2 + x3 + x4) / 4 ∧
  yM = (y1 + y2 + y3 + y4) / 4 ∧
  zM = (z1 + z2 + z3 + z4) / 4 ∧
  (x1 - xM) + (x2 - xM) + (x3 - xM) + (x4 - xM) = 0 ∧
  (y1 - yM) + (y2 - yM) + (y3 - yM) + (y4 - yM) = 0 ∧
  (z1 - zM) + (z2 - zM) + (z3 - zM) + (z4 - zM) = 0 := by
  sorry

end find_coordinates_M_l160_160647


namespace area_union_l160_160208

-- Definitions based on the conditions
def side_length : ℝ := 10
def radius : ℝ := 10

-- Calculations based on the conditions
def area_square : ℝ := side_length^2
def area_circle : ℝ := π * radius^2
def area_overlap : ℝ := (1/4) * area_circle

-- Target statement to prove
theorem area_union (side_length radius : ℝ) (h_side : side_length = 10) (h_radius : radius = 10) : 
  (side_length^2 + π * radius^2 - (1 / 4 * π * radius^2)) = 100 + 75 * π :=
by
  -- Using the given conditions
  rw [h_side, h_radius]
  -- Simplifying the expression
  simp
  -- Resulting assertion
  sorry

end area_union_l160_160208


namespace nice_people_in_crowd_l160_160219

theorem nice_people_in_crowd (Barry Kevin Julie Joe : ℕ)
    (hBarry : Barry = 24)
    (hKevin : Kevin = 20)
    (hJulie : Julie = 80)
    (hJoe : Joe = 50)
    : (Barry + Kevin / 2 + Julie * 3 / 4 + Joe * 10 / 100) = 99 :=
by 
  rw [hBarry, hKevin, hJulie, hJoe]
  norm_num
  sorry

end nice_people_in_crowd_l160_160219


namespace difference_of_squares_l160_160163

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 :=
by
  sorry

end difference_of_squares_l160_160163


namespace calvin_buys_chips_days_per_week_l160_160609

-- Define the constants based on the problem conditions
def cost_per_pack : ℝ := 0.50
def total_amount_spent : ℝ := 10
def number_of_weeks : ℕ := 4

-- Define the proof statement
theorem calvin_buys_chips_days_per_week : 
  (total_amount_spent / cost_per_pack) / number_of_weeks = 5 := 
by
  -- Placeholder proof
  sorry

end calvin_buys_chips_days_per_week_l160_160609


namespace first_generation_tail_length_l160_160389

theorem first_generation_tail_length
  (length_first_gen : ℝ)
  (H : (1.25:ℝ) * (1.25:ℝ) * length_first_gen = 25) :
  length_first_gen = 16 := by
  sorry

end first_generation_tail_length_l160_160389


namespace max_mondays_in_51_days_l160_160520

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l160_160520


namespace ellipse_foci_distance_l160_160224

theorem ellipse_foci_distance
  (a b : ℝ)
  (h_a : a = 7)
  (h_b : b = 3)
  : 2 * real.sqrt (a^2 - b^2) = 4 * real.sqrt 10 :=
by
  rw [h_a, h_b]
  sorry

end ellipse_foci_distance_l160_160224


namespace find_RF_l160_160593

noncomputable def equilateral_triangle_PQR : Type :=
{ side_length : ℝ,
  midpoint_N : ℝ,
  cyclic_quadrilateral_PJNF : Prop,
  area_triangle_JFN : ℝ }

def problem_statement : Prop :=
let P, Q, R, N, J, F : ℝ in
let PQ := 4 in
let PR := 4 in
let QR := 4 in
let N := (QR / 2) in
PJ > PF →
PJNF is cyclic →
area_triangle_JFN = 3 →
RF = 4

theorem find_RF (h : equilateral_triangle_PQR) : problem_statement :=
sorry

end find_RF_l160_160593


namespace trapezoid_area_l160_160206

-- Definitions
noncomputable def diameter : ℝ := 34
noncomputable def AB : ℝ := 10
noncomputable def CD : ℝ := 20
noncomputable def EF : ℝ := 7
noncomputable def FA : ℝ := 7

-- Main theorem to prove the area of trapezoid ABCD
theorem trapezoid_area : 
  (let radius : ℝ := diameter / 2 in
   let height : ℝ := Real.sqrt (radius^2 - (CD - AB)^2 / 4) in
   ((AB + CD) / 2) * height = 244) :=
by
  sorry

end trapezoid_area_l160_160206


namespace problem1_problem2_l160_160608

theorem problem1 : (Real.sqrt 24 - Real.sqrt 18) - Real.sqrt 6 = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : 2 * Real.sqrt 12 * Real.sqrt (1 / 8) + 5 * Real.sqrt 2 = Real.sqrt 6 + 5 * Real.sqrt 2 := by
  sorry

end problem1_problem2_l160_160608


namespace chord_construction_l160_160290

open Real

theorem chord_construction (O P : Point) (r d : ℝ) (h_r : r = 5) (h_d : d = dist O P) :
  (d > 7 → ¬ ∃ A B : Point, dist O A = r ∧ dist O B = r ∧ dist A B = 8 ∧ ∃ F : Point, dist O F = 3 ∧ dist P F = 4 ∧ F = midpoint A B) ∧
  (d = 7 → ∃! F : Point, (∃ A B : Point, dist O A = r ∧ dist O B = r ∧ dist A B = 8 ∧ dist O F = 3 ∧ dist P F = 4 ∧ F = midpoint A B)) ∧
  (d < 7 → ∃ F1 F2 : Point, (F1 ≠ F2 ∧ (∃ A1 B1 A2 B2 : Point, dist O A1 = r ∧ dist O B1 = r ∧ dist A1 B1 = 8 ∧ dist O F1 = 3 ∧ dist P F1 = 4 ∧ F1 = midpoint A1 B1 ∧ dist O F2 = 3 ∧ dist P F2 = 4 ∧ F2 = midpoint A2 B2))) :=
by sorry

end chord_construction_l160_160290


namespace half_angle_in_second_quadrant_l160_160854

theorem half_angle_in_second_quadrant (α : Real) (h1 : 180 < α ∧ α < 270)
        (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
        90 < α / 2 ∧ α / 2 < 180 :=
sorry

end half_angle_in_second_quadrant_l160_160854


namespace max_mondays_in_first_51_days_l160_160518

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l160_160518


namespace value_of_factorial_fraction_l160_160959

open Nat

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem value_of_factorial_fraction : (factorial 15) / ((factorial 6) * (factorial 9)) = 4165 := by
  sorry

end value_of_factorial_fraction_l160_160959


namespace problem_l160_160283

variables (a b : ℝ × ℝ) (c : ℝ × ℝ) (k : ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def projection (v u : ℝ × ℝ) : ℝ :=
  (dot_product v u) / vector_length u

noncomputable def vec_c (b : ℝ × ℝ) (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ :=
  (b.1 - k * a.1, b.2 - k * a.2)

noncomputable def find_k (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (a.1 ^ 2 + a.2 ^ 2)

theorem problem (b : ℝ × ℝ) (a : ℝ × ℝ) (hb : b = (3, 1)) (ha : a = (1, 2)) :
  (projection b a = real.sqrt 5) ∧ find_k a b = 1 ∧ vec_c b (find_k a b) a = (2, -1) :=
by
  split
  -- Proof for the first projection
  · sorry
  split
  -- Proof for finding k
  · sorry
  -- Proof for finding c
  · sorry

end problem_l160_160283


namespace Benjie_is_older_by_5_l160_160601

def BenjieAge : ℕ := 6
def MargoFutureAge : ℕ := 4
def YearsToFuture : ℕ := 3

theorem Benjie_is_older_by_5 :
  BenjieAge - (MargoFutureAge - YearsToFuture) = 5 :=
by
  sorry

end Benjie_is_older_by_5_l160_160601


namespace ratio_is_nine_to_sixteen_l160_160935

-- Define the radius of the base of the cone
variable (r : ℝ)

-- Define the radius of the circumscribed sphere
def radius_of_circumscribed_sphere (r : ℝ) := (2 / Real.sqrt 3) * r

-- Define the surface area of the cone
def surface_area_of_cone (r : ℝ) := 3 * Real.pi * r^2

-- Define the surface area of the circumscribed sphere
def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * (radius_of_circumscribed_sphere r)^2

-- Define the ratio of surface areas
def ratio_of_surface_areas (r : ℝ) := surface_area_of_cone r / surface_area_of_sphere r

-- Prove that the ratio of the surface area of the cone to the surface area of the circumscribed sphere is 9 : 16
theorem ratio_is_nine_to_sixteen (r : ℝ) : ratio_of_surface_areas r = 9 / 16 :=
by
  sorry

end ratio_is_nine_to_sixteen_l160_160935


namespace part_one_part_two_min_profit_part_two_max_profit_l160_160552

/-
Problem Statement:
1. Prove that there are exactly 8 values of n for which buying more than n books is cheaper than buying exactly n books.
2. Prove that the minimum profit the company can make is 302 yuan and the maximum profit is 384 yuan, given that the production cost per book is 5 yuan and two individuals buy a total of 60 books, each buying at least 1 book.
-/

def book_cost (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 24 then 12 * n
  else if 25 ≤ n ∧ n ≤ 48 then 11 * n
  else if n ≥ 49 then 10 * n
  else 0

def number_of_cheaper_values : ℕ := 8 -- From the solution, we concluded there are 8 such values

theorem part_one : (∃ ns : Finset ℕ, (∀ n ∈ ns, book_cost (n+1) < book_cost n) ∧ ns.card = number_of_cheaper_values) := sorry

noncomputable def total_books_cost (a b : ℕ) (h : a + b = 60) : ℕ :=
  book_cost a + book_cost b

def production_cost := 5 * 60

theorem part_two_min_profit :
  ∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ a + b = 60 ∧ 
  602 = total_books_cost a b rfl - production_cost := sorry

theorem part_two_max_profit :
  ∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ a + b = 60 ∧ 
  684 = total_books_cost a b rfl - production_cost := sorry

end part_one_part_two_min_profit_part_two_max_profit_l160_160552


namespace existence_of_solution_largest_unsolvable_n_l160_160619

-- Definitions based on the conditions provided in the problem
def equation (x y z n : ℕ) : Prop := 28 * x + 30 * y + 31 * z = n

-- There exist positive integers x, y, z such that 28x + 30y + 31z = 365
theorem existence_of_solution : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z 365 :=
by
  sorry

-- The largest positive integer n such that 28x + 30y + 31z = n cannot be solved in positive integers x, y, z is 370
theorem largest_unsolvable_n : ∀ (n : ℕ), (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → n ≠ 370) → ∀ (n' : ℕ), n' > 370 → (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n') :=
by
  sorry

end existence_of_solution_largest_unsolvable_n_l160_160619


namespace totalAttendees_l160_160136

def numberOfBuses : ℕ := 8
def studentsPerBus : ℕ := 45
def chaperonesList : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]

theorem totalAttendees : 
    numberOfBuses * studentsPerBus + chaperonesList.sum = 389 := 
by
  sorry

end totalAttendees_l160_160136


namespace count_four_digit_multiples_of_7_l160_160731

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160731


namespace smallest_integer_condition_l160_160640

theorem smallest_integer_condition :
  ∃ (x : ℕ) (d : ℕ) (n : ℕ) (p : ℕ), x = 1350 ∧ d = 1 ∧ n = 450 ∧ p = 2 ∧
  x = 10^p * d + n ∧
  n = x / 19 ∧
  (1 ≤ d ∧ d ≤ 9 ∧ 10^p * d % 18 = 0) :=
sorry

end smallest_integer_condition_l160_160640


namespace four_digit_multiples_of_7_l160_160752

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160752


namespace expression_equality_l160_160471

theorem expression_equality : (81 : ℝ) ^ (-(2 : ℝ) ^ (-2)) = 3 :=
by
  sorry

end expression_equality_l160_160471


namespace total_amount_shared_l160_160587

theorem total_amount_shared (A B C : ℕ) (h1 : 3 * B = 5 * A) (h2 : B = 25) (h3 : 5 * C = 8 * B) : A + B + C = 80 := by
  sorry

end total_amount_shared_l160_160587


namespace last_digit_of_product_is_zero_l160_160017

theorem last_digit_of_product_is_zero : ∃ (B И H П У Х : ℕ), 
  B ≠ И ∧ B ≠ H ∧ B ≠ П ∧ B ≠ У ∧ B ≠ Х ∧ И ≠ H ∧ И ≠ П ∧ И ≠ У ∧ И ≠ Х ∧ 
  H ≠ П ∧ H ≠ У ∧ H ≠ Х ∧ П ≠ У ∧ П ≠ Х ∧ У ≠ Х ∧
  B ∈ {2, 3, 4, 5, 6, 7} ∧ И ∈ {2, 3, 4, 5, 6, 7} ∧ H ∈ {2, 3, 4, 5, 6, 7} ∧
  П ∈ {2, 3, 4, 5, 6, 7} ∧ У ∈ {2, 3, 4, 5, 6, 7} ∧ Х ∈ {2, 3, 4, 5, 6, 7} ∧
  (B * И * H * H * И * П * У * Х) % 10 = 0 := by
  sorry

end last_digit_of_product_is_zero_l160_160017


namespace value_of_factorial_fraction_l160_160960

open Nat

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem value_of_factorial_fraction : (factorial 15) / ((factorial 6) * (factorial 9)) = 4165 := by
  sorry

end value_of_factorial_fraction_l160_160960


namespace length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l160_160639

def hexagon_vertex_to_center_length (a : ℝ) (h : a = 16) (regular_hexagon : Prop) : Prop :=
∃ (O A : ℝ), (a = 16) → (regular_hexagon = true) → (O = 0) ∧ (A = a) ∧ (dist O A = 16)

theorem length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16 :
  hexagon_vertex_to_center_length 16 (by rfl) true :=
sorry

end length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l160_160639


namespace tangent_line_to_parabola_l160_160642

theorem tangent_line_to_parabola : ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 6 * y + k = 0) ∧ (∀ y : ℝ, ∃ x : ℝ, y^2 = 32 * x) ∧ (48^2 - 4 * (1 : ℝ) * 8 * k = 0) := by
  use 72
  sorry

end tangent_line_to_parabola_l160_160642


namespace problem_solution_l160_160029

def P (x : ℝ) : ℝ := 3 * Real.sqrt x
def Q (x : ℝ) : ℝ := x ^ 3

theorem problem_solution : P (Q (P (Q (P (Q 2))))) = 648 * 3^(3/4) :=
by
  sorry

end problem_solution_l160_160029


namespace andrey_gifts_l160_160060

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l160_160060


namespace ratio_ST_SQ_l160_160015

variables {P Q R S T : Type*}
variables [EuclideanGeometry.triangle P Q R] [EuclideanGeometry.triangle P Q S]
variables {PQ : ℝ} {PR : ℝ} {RQ : ℝ} {PS : ℝ} {ST : ℝ} {SQ : ℝ}

noncomputable theory

def triangle_PQR : Prop := 
  EuclideanGeometry.right_angle PR RQ ∧ PR = 5 ∧ RQ = 12

def triangle_PQS : Prop :=
  EuclideanGeometry.right_angle PS PQ ∧ PS = 15

def opposite_sides : Prop :=
  ¬EuclideanGeometry.collinear R S PQ ∧ EuclideanGeometry.collinear P Q R ∧ EuclideanGeometry.collinear P Q S

def parallel_and_extended : Prop :=
  EuclideanGeometry.parallel (EuclideanGeometry.line_through S T) (EuclideanGeometry.line_through P R) ∧
  EuclideanGeometry.collinear R Q T ∧ EuclideanGeometry.extended R Q T

theorem ratio_ST_SQ :
  triangle_PQR ∧ triangle_PQS ∧ opposite_sides ∧ parallel_and_extended →
  (ST / SQ) = 930 / 2197 :=
by sorry

end ratio_ST_SQ_l160_160015


namespace four_digit_multiples_of_7_count_l160_160754

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160754


namespace max_value_expr_l160_160406

theorem max_value_expr (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum : a + b + c = 2) :
  ∃ M, (∀ a b c, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 2 → (frac_expr a b c) ≤ M) ∧ (exists a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 ∧ (frac_expr a b c) = M) :=
by
  have frac_expr := (λ a b c, (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c))
  sorry

end max_value_expr_l160_160406


namespace food_bank_remaining_l160_160420

theorem food_bank_remaining :
  ∀ (f1 f2 : ℕ) (p : ℚ),
  f1 = 40 →
  f2 = 2 * f1 →
  p = 0.7 →
  (f1 + f2) - (p * (f1 + f2)).toNat = 36 :=
by
  intros f1 f2 p h1 h2 h3
  sorry

end food_bank_remaining_l160_160420


namespace max_possible_pieces_l160_160004

-- Definitions for the Lean version of the problem conditions
def grid_size : ℕ := 200

def pieces_are_colored (r : ℕ) (c : ℕ) : Prop :=
(r = 0 ∨ r = 1) ∧ (c = 0 ∨ c = 1)

def sees_exactly_five_of_other_color (grid : ℕ → ℕ → option ℕ) (r c : ℕ) : Prop :=
∃ count : ℕ,
  (count = (if grid r c = some 0 then (list.filter (λ x, x = some 1) ([grid r i | i in finRange 200] ∪ [grid i c | i in finRange 200])).length
            else (list.filter (λ x, x = some 0) ([grid r i | i in finRange 200] ∪ [grid i c | i in finRange 200])).length)) 
    ∧ count = 5

-- The proof problem:
theorem max_possible_pieces : ∀ (grid : ℕ → ℕ → option ℕ),
  (∀ r c, r < grid_size → c < grid_size → pieces_are_colored (grid r c)) →
  (∀ r c, r < grid_size → c < grid_size → sees_exactly_five_of_other_color grid r c) →
  ∃ n : ℕ, n ≤ 3800 :=
by
  sorry

end max_possible_pieces_l160_160004


namespace remaining_balance_on_phone_card_l160_160826

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l160_160826


namespace profit_percent_is_approx_6_point_35_l160_160540

noncomputable def selling_price : ℝ := 2552.36
noncomputable def cost_price : ℝ := 2400
noncomputable def profit_amount : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_percent_is_approx_6_point_35 : abs (profit_percent - 6.35) < 0.01 := sorry

end profit_percent_is_approx_6_point_35_l160_160540


namespace compare_xyz_l160_160595

theorem compare_xyz
  (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d)
  (x : ℝ) (hx : x = (a + b) * (c + d))
  (y : ℝ) (hy : y = (a + c) * (b + d))
  (z : ℝ) (hz : z = (a + d) * (b + c)) :
  x < y ∧ y < z :=
by sorry

end compare_xyz_l160_160595


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l160_160638

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l160_160638


namespace edge_upper_bound_no_4_cycle_l160_160242

variable (G : Type) [Graph G] (n m : ℕ)
variable [graph_of_size G n m]
variable (h1 : ∀ (x y z w : vertex G), ¬(path G [x, y, z, w, x])) -- no 4-cycle

theorem edge_upper_bound_no_4_cycle :
  m ≤ n / 4 * (1 + sqrt (4 * n - 3)) :=
sorry

end edge_upper_bound_no_4_cycle_l160_160242


namespace division_check_l160_160970

def is_divisible (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b

theorem division_check :
  ∀ (a b : ℕ), (a, b) ∈ {(70, 9), (9, 9), (43, 5)} → is_divisible a b ↔ (a = 9 ∧ b = 9) :=
by sorry

end division_check_l160_160970


namespace john_time_for_100_meters_l160_160823

theorem john_time_for_100_meters :
  ∀ (d t additional_distance : ℕ), d = 20 → t = 40 → additional_distance = 100 →
    let speed := d / t
    in additional_distance / speed = 200 :=
by
  assume (d t additional_distance : ℕ) (h_d : d = 20) (h_t : t = 40) (h_additional_distance : additional_distance = 100)
  let speed := d / t
  have h_speed : speed = 0.5 := by sorry
  have h_time : additional_distance / speed = 200 := by sorry
  exact h_time

end john_time_for_100_meters_l160_160823


namespace max_values_sqrt_expr_l160_160288

theorem max_values_sqrt_expr (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (sqrt (a / (2 * a + b)) + sqrt (b / (2 * b + a)) ≤ 2 * sqrt 3 / 3) ∧
  (sqrt (a / (a + 2 * b)) + sqrt (b / (b + 2 * a)) ≤ 2 * sqrt 3 / 3) := 
by
  sorry

end max_values_sqrt_expr_l160_160288


namespace max_element_in_S_n_l160_160025

theorem max_element_in_S_n (d : ℝ) (hd : d > 0) (n : ℕ) : 
  ∃ m ∈ {s : ℝ | ∃ (x : ℕ → ℕ) (hx : ∀ i, x i ∈ ℕ), s = (finset.range n).sum (λ i, (1 / (x i : ℝ)) ∧ s < d)}, 
  ∀ y ∈ {s : ℝ | ∃ (x : ℕ → ℕ) (hx : ∀ i, x i ∈ ℕ), s = (finset.range n).sum (λ i, (1 / (x i : ℝ)) ∧ s < d)}, y ≤ m :=
begin
  sorry
end

end max_element_in_S_n_l160_160025


namespace new_students_joined_l160_160461

theorem new_students_joined (orig_avg_age new_avg_age : ℕ) (decrease_in_avg_age : ℕ) (orig_strength : ℕ) (new_students_avg_age : ℕ) :
  orig_avg_age = 40 ∧ new_avg_age = 36 ∧ decrease_in_avg_age = 4 ∧ orig_strength = 18 ∧ new_students_avg_age = 32 →
  ∃ x : ℕ, ((orig_strength * orig_avg_age) + (x * new_students_avg_age) = new_avg_age * (orig_strength + x)) ∧ x = 18 :=
by
  sorry

end new_students_joined_l160_160461


namespace min_sum_a_b_l160_160649

-- The conditions
variables {a b : ℝ}
variables (h₁ : a > 1) (h₂ : b > 1) (h₃ : ab - (a + b) = 1)

-- The theorem statement
theorem min_sum_a_b : a + b = 2 + 2 * Real.sqrt 2 :=
sorry

end min_sum_a_b_l160_160649


namespace parallelogram_coverage_l160_160123

-- Define the conditions of the problem
variables (A a : ℝ)
variables (triangle_ABD_acute : Prop)

-- Assume the necessary properties of the parallelogram ABCD
axiom angle_BAD_geometric_constraints : triangle_ABD_acute → (∀ {x y z : ℝ}, cos x + sqrt 3 * sin x ≥ 0)

-- Formalize the theorem to be proved
theorem parallelogram_coverage (H : triangle_ABD_acute) : 
  a ≤ cos A + sqrt 3 * sin A :=
by
  apply angle_BAD_geometric_constraints
  exact H
  sorry

end parallelogram_coverage_l160_160123


namespace probability_of_missing_coupons_l160_160490

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l160_160490


namespace fruit_display_fruits_count_l160_160944

theorem fruit_display_fruits_count :
  ∀ (bananas oranges apples lemons : ℕ),
  (bananas = 5) →
  (oranges = 2 * bananas) →
  (apples = 2 * oranges) →
  (lemons = (apples + bananas) / 2) →
  bananas + oranges + apples + lemons = 47 :=
by
  intros bananas oranges apples lemons
  assume h_bananas h_oranges h_apples h_lemons
  sorry

end fruit_display_fruits_count_l160_160944


namespace max_fraction_sum_l160_160404

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_fraction_sum_l160_160404


namespace race_speed_ratio_l160_160981

theorem race_speed_ratio (L v_a v_b : ℝ) (h1 : v_a = v_b / 0.84375) :
  v_a / v_b = 32 / 27 :=
by sorry

end race_speed_ratio_l160_160981


namespace max_BP_squared_l160_160835

theorem max_BP_squared : 
  ∀ (ω : Type) (A B C T P : ω) (r : ℝ),
    diameter ω A B →
    (∀ x : ω, (x = C) → dist A x = 2 * dist A B) →
    (tangent ω T C) →
    (perpendicular A P (line_through C T)) →
    dist A B = 12 →
    (∃ BP : ℝ, BP^2 = 190.08) :=
by 
  intros ω A B C T P r h_diam h_dist h_tangent h_perp h_AB_dist,
  sorry

end max_BP_squared_l160_160835


namespace markup_rate_correct_l160_160822

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def expenses_percentage : ℝ := 0.15
noncomputable def cost (S : ℝ) : ℝ := S - (profit_percentage * S + expenses_percentage * S)
noncomputable def markup_rate (S C : ℝ) : ℝ := (S - C) / C * 100

theorem markup_rate_correct :
  markup_rate selling_price (cost selling_price) = 53.85 := 
by
  sorry

end markup_rate_correct_l160_160822


namespace diameter_expression_l160_160106

noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (r ^ 3)

noncomputable def find_diameter (r1 : ℝ) : ℝ :=
  let volume1 := sphereVolume r1
  let volume2 := 3 * volume1
  let r2 := Real.cbrt (volume2 * 3 / (4 * Real.pi)) -- Solving for r in sphereVolume formula
  2 * r2   -- The diameter is twice the radius

theorem diameter_expression : 
  ∃ (a b : ℕ), b % 5 ≠ 0 ∧ find_diameter 6 = a * Real.cbrt b ∧ a + b = 18 :=
by {
  use [12, 6], -- Providing specific values for a and b
  split, {
    exact dec_trivial,   -- Proving b doesn't have perfect cube factors
  },
  split,
  {   
    have h1 : find_diameter (6 : ℝ) = 2 * Real.cbrt (2592 * 3 / 4),
    sorry,
    have h2 : 2 * Real.cbrt (2592 * 3 / 4) = 12 * Real.cbrt 6,
    sorry
  },
  {
    exact dec_trivial,  -- Directly proving a + b = 18 as a = 12 and b= 6
  }
}

end diameter_expression_l160_160106


namespace number_of_solutions_eq_two_l160_160270

theorem number_of_solutions_eq_two :
  (∃ (x : ℝ), sqrt (9 - x) = x * sqrt (9 - x)) ∧
  (∀ x, sqrt (9 - x) = x * sqrt (9 - x) → (x = 1 ∨ x = 9)) :=
begin
  sorry
end

end number_of_solutions_eq_two_l160_160270


namespace total_problems_done_l160_160209

def recorded_numbers : List ℤ := [-3, 5, -4, 2, -1, 1, 0, -3, 8, 7]

def base_number_of_problems_done : ℤ := 6 * 10

def adjustment : ℤ := recorded_numbers.sum

theorem total_problems_done : base_number_of_problems_done + adjustment = 72 :=
by
  unfold base_number_of_problems_done adjustment
  simp [recorded_numbers]
  calc
    60 + 12 = 72 := by sorry

end total_problems_done_l160_160209


namespace area_of_region_above_line_l160_160146

def circle := set (ℝ × ℝ)
def line := set (ℝ × ℝ)

axiom area_of_circle (center : ℝ × ℝ) (radius : ℝ) : ℝ

theorem area_of_region_above_line 
  (C : circle := {p | (p.1 + 2)^2 + (p.2 - 10)^2 = 36})
  (L : line := {p | p.2 = 2 * p.1 - 4}) :
  (((C : set (ℝ × ℝ)) \ L).area = 36 * Real.pi) :=
sorry

end area_of_region_above_line_l160_160146


namespace count_four_digit_multiples_of_7_l160_160743

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160743


namespace eccentricity_of_hyperbola_l160_160278

variables {a b c : ℝ}

def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def vertex_distance_eq_focal_length_third : Prop := 
  2 * a = (1 / 3) * 2 * c

def eccentricity (e : ℝ) := e = c / a

theorem eccentricity_of_hyperbola (h1 : vertex_distance_eq_focal_length_third) : eccentricity 3 :=
by
  sorry

end eccentricity_of_hyperbola_l160_160278


namespace andrey_gifts_l160_160065

theorem andrey_gifts :
  ∃ (n : ℕ), ∀ (a : ℕ), n(n-2) = a(n-1) + 16 ∧ n = 18 :=
by {
  sorry
}

end andrey_gifts_l160_160065


namespace sugar_per_chocolate_bar_l160_160425

-- Definitions from conditions
def total_sugar : ℕ := 177
def lollipop_sugar : ℕ := 37
def chocolate_bar_count : ℕ := 14

-- Proof problem statement
theorem sugar_per_chocolate_bar : 
  (total_sugar - lollipop_sugar) / chocolate_bar_count = 10 := 
by 
  sorry

end sugar_per_chocolate_bar_l160_160425


namespace sum_first_100_terms_arithmetic_sequence_l160_160887

noncomputable def arithmetic_sum_100 (a1 : ℝ) (a100 : ℝ) (a : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range 100, a i

open function

theorem sum_first_100_terms_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h_vector : ∀ {OA OB OC : ℝ} (h_C : OC = a 0 * OA + a 99 * OB), a1 + a100 = 1) :
  arithmetic_sum_100 (a 0) (a 99) a = 50 :=
by
  -- Definitions and setup
  have h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0) := h_seq
  have h_coeff_sum : (a 0) + (a 99) = 1 := h_vector
  -- Sum calculation from known formulas
  sorry  -- Proof not required, just statement.

end sum_first_100_terms_arithmetic_sequence_l160_160887


namespace fruit_difference_l160_160424

variable (Steven_apples Steven_peaches Mia_apples Mia_peaches Jake_apples Jake_peaches : ℕ)

theorem fruit_difference (
  h1: Steven_apples = 19,
  h2: Steven_peaches = 15,
  h3: Mia_apples = 2 * Steven_apples,
  h4: Mia_peaches = Jake_peaches + 3,
  h5: Jake_peaches = Steven_peaches - 3,
  h6: Jake_apples = Steven_apples + 4)
  : Steven_apples + Steven_peaches + Mia_apples + Mia_peaches + Jake_apples + Jake_peaches = 122 := 
by
  sorry

end fruit_difference_l160_160424


namespace lisa_needs_28_more_marbles_l160_160862

theorem lisa_needs_28_more_marbles :
  ∀ (friends : ℕ) (initial_marbles : ℕ),
  friends = 12 → 
  initial_marbles = 50 →
  (∀ n, 1 ≤ n ∧ n ≤ friends → ∃ (marbles : ℕ), marbles ≥ 1 ∧ ∀ i j, (i ≠ j ∧ i ≠ 0 ∧ j ≠ 0) → (marbles i ≠ marbles j)) →
  ( ∑ k in finset.range (friends + 1), k ) - initial_marbles = 28 :=
by
  intros friends initial_marbles h_friends h_initial_marbles _,
  rw [h_friends, h_initial_marbles],
  sorry

end lisa_needs_28_more_marbles_l160_160862


namespace number_of_four_digit_multiples_of_7_l160_160693

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160693


namespace total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l160_160194

-- Initial conditions
def cost_4_oranges : Nat := 12
def cost_7_oranges : Nat := 28
def total_oranges : Nat := 28

-- Calculate the total cost for 28 oranges
theorem total_cost_28_oranges
  (x y : Nat) 
  (h1 : 4 * x + 7 * y = total_oranges) 
  (h2 : total_oranges = 28) 
  (h3 : x = 7) 
  (h4 : y = 0) : 
  7 * cost_4_oranges = 84 := 
by sorry

-- Calculate the average cost per orange
theorem avg_cost_per_orange 
  (total_cost : Nat) 
  (h1 : total_cost = 84)
  (h2 : total_oranges = 28) : 
  total_cost / total_oranges = 3 := 
by sorry

-- Calculate the cost for 6 oranges
theorem cost_6_oranges 
  (avg_cost : Nat)
  (h1 : avg_cost = 3)
  (n : Nat) 
  (h2 : n = 6) : 
  n * avg_cost = 18 := 
by sorry

end total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l160_160194


namespace phi_value_l160_160333

-- Define the function and its parameters
def f (ω : ℝ) (φ : ℝ) (x : ℝ) := Real.sin (ω * x + φ)

-- Lean 4 statement
theorem phi_value (ω : ℝ) (φ : ℝ) (φ_pos : 0 < φ) (φ_lt_half_pi : φ < Real.pi / 2)
    (ω_pos : 0 < ω) (f0 : f ω φ 0 = -f ω φ (Real.pi / 2))
    (g_symmetric : ∀ x, f ω φ (x - Real.pi / 12) = -f ω φ (-x + Real.pi / 12)) :
    φ = Real.pi / 6 :=
by
  sorry

end phi_value_l160_160333


namespace number_of_solutions_in_interval_l160_160246

noncomputable def num_solutions (f : ℝ → ℝ) (a b : ℝ) : ℕ := (Icc a b).filter (λ x, f x = 0).card

theorem number_of_solutions_in_interval : 
  num_solutions (λ x, 3 * (sin x)^3 - 7 * (sin x)^2 + 3 * (sin x)) 0 (2 * π) = 5 :=
sorry

end number_of_solutions_in_interval_l160_160246


namespace plane_eq_passing_A_perpendicular_BC_l160_160556

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def subtract_points (P Q : Point3D) : Point3D :=
  { x := P.x - Q.x, y := P.y - Q.y, z := P.z - Q.z }

-- Points A, B, and C given in the conditions
def A : Point3D := { x := 1, y := -5, z := -2 }
def B : Point3D := { x := 6, y := -2, z := 1 }
def C : Point3D := { x := 2, y := -2, z := -2 }

-- Vector BC
def BC : Point3D := subtract_points C B

theorem plane_eq_passing_A_perpendicular_BC :
  (-4 : ℝ) * (A.x - 1) + (0 : ℝ) * (A.y + 5) + (-3 : ℝ) * (A.z + 2) = 0 :=
  sorry

end plane_eq_passing_A_perpendicular_BC_l160_160556


namespace find_f3_l160_160852

noncomputable def g (k x : ℕ) : ℕ := (x^3 / 10^(2*k)).floor

def f (k : ℕ) : ℕ := 
  Nat.find (λ n, ∃ x, g k x - g k (x-1) ≥ 2 ∧ n = x)

theorem find_f3 : f 3 = 1 := 
by
  sorry

end find_f3_l160_160852


namespace shaded_area_of_ΔCDE_l160_160147

theorem shaded_area_of_ΔCDE :
  let O := (0, 0)
  let A := (4, 0)
  let B := (16, 0)
  let C := (16, 12)
  let D := (4, 12)
  let E := (4, 4)
  let F := (0, 4)
  let G := (0, 7)
  let EA := 3   -- from solution step
  let DE := 12 - EA  -- from condition 1
  let DC := 12      -- side length
  TriangleArea C D E = 54 :=
by
  sorry

end shaded_area_of_ΔCDE_l160_160147


namespace exists_zero_intersection_sum_l160_160362

-- Define the dimensions of the grid
def m : ℕ := 1980
def n : ℕ := 1981

-- Define the conditions on the grid
def grid_condition (grid : ℕ → ℕ → ℤ) : Prop :=
  (∀ (i : ℕ), i < m → ∀ (j : ℕ), j < n → grid i j ∈ {-1, 0, 1}) ∧
  (∑ i in finset.range m, ∑ j in finset.range n, grid i j = 0)

-- Formulate the main theorem
theorem exists_zero_intersection_sum (grid : ℕ → ℕ → ℤ) (cond_grid : grid_condition grid) :
  ∃ (i1 i2 : ℕ), i1 < m ∧ i2 < m ∧ i1 ≠ i2 ∧ 
  ∃ (j1 j2 : ℕ), j1 < n ∧ j2 < n ∧ j1 ≠ j2 ∧ 
  (grid i1 j1 + grid i1 j2 + grid i2 j1 + grid i2 j2 = 0) :=
begin
  sorry
end

end exists_zero_intersection_sum_l160_160362


namespace four_digit_multiples_of_7_l160_160745

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160745


namespace min_cost_speed_l160_160529

noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

theorem min_cost_speed 
  (v : ℝ) 
  (u : ℝ) 
  (other_costs : ℝ) 
  (h1 : u = (1/200) * v^3) 
  (h2 : u = 40) 
  (h3 : v = 20) 
  (h4 : other_costs = 270) 
  (b : ℝ) 
  : ∃ v_min, v_min = 30 ∧ 
    ∀ (v : ℝ), (0 < v ∧ v ≤ b) → 
    ((fuel_cost v / v + other_costs / v) ≥ (fuel_cost v_min / v_min + other_costs / v_min)) := 
sorry

end min_cost_speed_l160_160529


namespace least_prime_b_l160_160097

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end least_prime_b_l160_160097


namespace max_value_of_n_d_l160_160039

theorem max_value_of_n_d (n : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, i < n → a i > 0 ) 
(h2 : Function.injective a) (h3 : (Finset.range n).sum a = 2014)
(h4 : n > 1):
  ∃ d : ℕ, d = nat.gcd (Finset.range n).prod a → n * d ≤ 530 :=
by
  sorry

end max_value_of_n_d_l160_160039


namespace question_correctness_l160_160966

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l160_160966


namespace reinforcement_correct_l160_160978

-- Conditions
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_before_reinforcement : ℕ := 18
def days_after_reinforcement : ℕ := 20

-- Define the remaining provisions after 18 days
def provisions_left : ℕ := initial_men * (initial_days - days_before_reinforcement)

-- Define reinforcement
def reinforcement : ℕ := 
  sorry -- placeholder for the definition

-- Theorem to prove
theorem reinforcement_correct :
  reinforcement = 1600 :=
by
  -- Use the given conditions to derive the reinforcement value
  let total_provision := initial_men * initial_days
  let remaining_provision := provisions_left
  let men_after_reinforcement := initial_men + reinforcement
  have h := remaining_provision = men_after_reinforcement * days_after_reinforcement
  sorry -- placeholder for the proof

end reinforcement_correct_l160_160978


namespace exists_nat_not_in_any_geometric_progression_l160_160486

theorem exists_nat_not_in_any_geometric_progression (G : Fin 100 → ℕ → ℕ) (hG : ∀ (i : Fin 100), ∃ (a : ℕ) (r : ℕ), r > 1 ∧ G i = λ n, a * r ^ n) :
  ∃ n : ℕ, ∀ (i : Fin 100), ∀ m : ℕ, G i m ≠ n :=
by
  sorry

end exists_nat_not_in_any_geometric_progression_l160_160486


namespace intervals_of_monotonic_increase_l160_160920

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def deriv_y (x : ℝ) : ℝ := Deriv (λ x, x^2 * Real.exp x) x

theorem intervals_of_monotonic_increase :
  {x : ℝ |  x < -2 ∨ x > 0 } = {x : ℝ | deriv_y x > 0 } :=
by
  sorry

end intervals_of_monotonic_increase_l160_160920


namespace paint_problem_l160_160566

theorem paint_problem
  (total_paint : ℕ)
  (blue_paint : ℕ)
  (total_paint_eq : total_paint = 6689)
  (blue_paint_eq : blue_paint = 6029) :
  total_paint - blue_paint = 660 :=
by
  rw [total_paint_eq, blue_paint_eq]
  norm_num

end paint_problem_l160_160566


namespace product_of_roots_eq_neg30_l160_160785

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l160_160785


namespace analytical_method_seeks_sufficient_condition_l160_160228

theorem analytical_method_seeks_sufficient_condition :
  ∀ {P Q : Prop}, (P → Q) → (∃ P, P → Q) :=
by
  intros P Q h
  exact ⟨P, h⟩

end analytical_method_seeks_sufficient_condition_l160_160228


namespace remainder_when_four_times_n_minus_9_divided_by_11_l160_160295

theorem remainder_when_four_times_n_minus_9_divided_by_11 
  (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end remainder_when_four_times_n_minus_9_divided_by_11_l160_160295


namespace lacson_sweet_potatoes_l160_160873

theorem lacson_sweet_potatoes :
  let total := 80 in
  let sold_adams := 20 in
  let unsold := 45 in
  total - sold_adams - unsold = 15 := by
  let total := 80
  let sold_adams := 20
  let unsold := 45
  show total - sold_adams - unsold = 15
  sorry

end lacson_sweet_potatoes_l160_160873


namespace slope_angle_l160_160483

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_angle (θ : ℝ) :
  ∃ (θ : ℝ), (slope_of_line 1 1 (-3) = -1) → (tan θ = -1) ∧ 0 ≤ θ ∧ θ < real.pi → θ = real.pi * (3 / 4) :=
begin
  -- proof omitted
  sorry
end

end slope_angle_l160_160483


namespace inequality_solution_l160_160618

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem inequality_solution (x : ℝ) :
  cube_root x ^ 3 + 2 + 2 / (cube_root x ^ 2 + 3) ≤ 0 ↔ x ∈ Set.Ioo (-∞) (-27) ∪ Set.Ioc (-27) (-1) ∪ Set.Ioo 8 ∞ := by
sorries

end inequality_solution_l160_160618


namespace cards_dealt_evenly_l160_160792

theorem cards_dealt_evenly (total_cards : ℕ) (total_people : ℕ) 
  (h_cards : total_cards = 60) (h_people : total_people = 9) : 
  (∃ (count : ℕ), count = total_people ∧ ∀ (cards : ℕ), cards < 8) :=
by
  have h_quota := total_cards / total_people
  have h_remainder := total_cards % total_people
  have h_quota_val : h_quota = 6, from sorry
  have h_remainder_val : h_remainder = 6, from sorry
  use total_people
  split
  { assumption }
  { intros cards h_cards_lt8
    cases h_cards_lt8
    cases cards
    · sorry
    · sorry }

end cards_dealt_evenly_l160_160792


namespace math_problem_1_math_problem_2_l160_160236

theorem math_problem_1 : 
  real.sqrt 36 - real.cbrt 125 + real.cbrt (7 / 8 - 1) = 1 / 2 := 
by 
  sorry

theorem math_problem_2 : 
  abs (real.sqrt 2 - real.sqrt 3) + 2 * real.sqrt 2 = real.sqrt 3 + real.sqrt 2 := 
by 
  sorry

end math_problem_1_math_problem_2_l160_160236


namespace dx_dt_formula_range_a_extreme_value_volume_of_solid_l160_160024

-- Variables and Functions
variable (a : ℝ) (x : ℝ) (t : ℝ)

-- Definitions based on Conditions
def f (x : ℝ) := (1 - a * Real.cos x) / (1 + Real.sin x)
def g (t : ℝ) := -Real.cos x / (1 + Real.sin x)

-- Statements
theorem dx_dt_formula
  (h₀ : 0 < x) (h₁ : x < π) (hx : t = (-(Real.cos x)) / (1 + Real.sin x)) :
  ∃ y : ℝ, dx / dt = 1 + ( (-t^2 + y) / (t^2 + 1)) := sorry

theorem range_a_extreme_value
  (h₀ : 0 < x) (h₁ : x < π) (hx₁ : -1 < a) (hx₂ : a < 1) :
  ∃ y : ℝ, f (π / 2) = 1 / 2 := sorry

theorem volume_of_solid
  (h₀ : 0 < x) (h₁ : x < π) (hx₁ : -1 < a) (hx₂ : a < 1) (hx : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π) :
  ∃ V : ℝ, V = π * ∫ x in 0..π, (1 - 2 * a * Real.cos x + a^2 * (Real.cos x)^2) / ((1 + Real.sin x)^2) := sorry

end dx_dt_formula_range_a_extreme_value_volume_of_solid_l160_160024


namespace find_f_at_one_l160_160670

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem find_f_at_one :
  (∀ x : ℝ, x ≥ -2 → f x (-16) ≥ f (-2) (-16)) ∧
  (∀ x : ℝ, x ≤ -2 → f x (-16) ≤ f (-2) (-16)) →
  f 1 (-16) = 25 :=
sorry

end find_f_at_one_l160_160670


namespace circle_equation_proof_l160_160187

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define a predicate for the circle being tangent to the y-axis
def tangent_y_axis (center : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r = abs center.1

-- Define the equation of the circle given center and radius
def circle_eqn (center : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2

-- State the theorem
theorem circle_equation_proof :
  tangent_y_axis circle_center →
  ∃ r, r = 1 ∧ circle_eqn circle_center r :=
sorry

end circle_equation_proof_l160_160187


namespace greg_savings_l160_160688

-- Definitions based on the conditions
def scooter_cost : ℕ := 90
def money_needed : ℕ := 33

-- The theorem to prove
theorem greg_savings : scooter_cost - money_needed = 57 := 
by
  -- sorry is used to skip the actual mathematical proof steps
  sorry

end greg_savings_l160_160688


namespace units_digit_sum_factorials_l160_160235

theorem units_digit_sum_factorials :
  (∑ n in Finset.range 50, Nat.factorial (n + 1)) % 10 = 3 := by
  sorry

end units_digit_sum_factorials_l160_160235


namespace find_magnitude_b_l160_160685

open Real

variable {l : ℝ} -- Let l be an arbitrary real number
variable {b_x b_y : ℝ} -- Components of vector b

def a : ℝ × ℝ := (2, l)
def b : ℝ × ℝ := (b_x, b_y)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

theorem find_magnitude_b (h1 : dot_product a b = 10) (h2 : magnitude (vector_add a b) = 5 * sqrt 2) :
  magnitude b = 5 :=
by
  sorry

end find_magnitude_b_l160_160685


namespace nate_walks_past_per_minute_l160_160053

-- Define the conditions as constants
def rows_G := 15
def cars_per_row_G := 10
def rows_H := 20
def cars_per_row_H := 9
def total_minutes := 30

-- Define the problem statement
theorem nate_walks_past_per_minute :
  ((rows_G * cars_per_row_G) + (rows_H * cars_per_row_H)) / total_minutes = 11 := 
sorry

end nate_walks_past_per_minute_l160_160053


namespace volleyball_team_l160_160427

theorem volleyball_team :
  let total_combinations := (Nat.choose 15 6)
  let without_triplets := (Nat.choose 12 6)
  total_combinations - without_triplets = 4081 :=
by
  -- Definitions based on the problem conditions
  let team_size := 15
  let starters := 6
  let triplets := 3
  let total_combinations := Nat.choose team_size starters
  let without_triplets := Nat.choose (team_size - triplets) starters
  -- Identify the proof goal
  have h : total_combinations - without_triplets = 4081 := sorry
  exact h

end volleyball_team_l160_160427


namespace no_two_positive_roots_l160_160328

theorem no_two_positive_roots
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_poly : ∀ x : ℝ, ∑ i in Finset.range n, (a i) * x^(n-i)) = 0) :
  ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → false := 
sorry

end no_two_positive_roots_l160_160328


namespace missing_jar_size_l160_160391

theorem missing_jar_size (x : ℕ) (h₁ : 3 * 16 + 3 * x + 3 * 40 = 252) 
                          (h₂ : 3 + 3 + 3 = 9) : x = 28 := 
by 
  sorry

end missing_jar_size_l160_160391


namespace coefficient_of_x2_in_expansion_l160_160467

theorem coefficient_of_x2_in_expansion :
  (x - (2 : ℤ)/x) ^ 4 = 8 * x^2 := sorry

end coefficient_of_x2_in_expansion_l160_160467


namespace shorter_leg_length_l160_160923

theorem shorter_leg_length (a : ℝ) (h1 : (∃ θ : Geometry.Triangle, θ.is_right ∧ θ.one_leg_length = 2 * θ.other_leg_length ∧ θ.median_hypotenuse_length = 12)) :
  a = 24 * Real.sqrt 5 / 5 := sorry

end shorter_leg_length_l160_160923


namespace transformation_correct_l160_160945

-- Define the original function
def f (x : ℝ) : ℝ := 2 * Real.sin x

-- Define the transformed function
def g (x : ℝ) : ℝ := 2 * Real.sin (x / 3 - Real.pi / 6)

-- State the theorem regarding the transformation
theorem transformation_correct :
  ∀ (x : ℝ), g x = f (3 * (x + Real.pi / 6)) :=
by
  sorry

end transformation_correct_l160_160945


namespace f_positive_f_sub_div_solve_inequality_l160_160312

-- Definitions of the conditions and questions
variable (f : ℝ → ℝ)

-- Condition 1: f is increasing
axiom f_increasing : ∀ x y : ℝ, x < y → f(x) < f(y)

-- Condition 2: f(x) is non-zero
axiom f_nonzero : ∀ x : ℝ, f(x) ≠ 0

-- Condition 3: functional equation f(x + y) = f(x) * f(y)
axiom f_add_mul : ∀ x y : ℝ, f(x + y) = f(x) * f(y)

-- Question (1): Prove f(x) > 0
theorem f_positive : ∀ x : ℝ, f(x) > 0 :=
sorry

-- Question (2): Prove f(x - y) = f(x) / f(y)
theorem f_sub_div (x y : ℝ) : f(x - y) = f(x) / f(y) :=
sorry

-- Question (3): Given f(1) = 2, solve the inequality f(3x) > 4 * f(x)
theorem solve_inequality (x : ℝ) (h_f1 : f 1 = 2) : f(3 * x) > 4 * f x → x > 1 :=
sorry

end f_positive_f_sub_div_solve_inequality_l160_160312


namespace symmetric_line_eq_l160_160112

theorem symmetric_line_eq (P Q : ℝ × ℝ) :
  let L1 := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  let L2 := {p : ℝ × ℝ | p.1 - 2 * p.2 + 2 = 0}
  (P ∈ L1) ∧ (Q ∈ L2) →
  (symmetric_point P Q).1 = 2 / 5 ∧ (symmetric_point P Q).2 = 16 / 5 →
  let M := (6, 4) 
  let P' := (2 / 5, 16 / 5)
  let L3 := λ x y : ℝ, x - 7 * y + 22 = 0 
  L3 (midpoint M P').1 (midpoint M P').2 = 0 :=
sorry

end symmetric_line_eq_l160_160112


namespace four_digit_multiples_of_7_l160_160700

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160700


namespace avg_of_distinct_t_l160_160323

theorem avg_of_distinct_t (t : ℕ) (roots_positive_int : ∀ r1 r2 : ℕ, (r1 + r2 = 6 ∧ r1 * r2 = t) → r1 > 0 ∧ r2 > 0) :
  (let distinct_ts := {t : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = 6 ∧ r1 * r2 = t} in 
   let avg := (5 + 8 + 9) / 3 in 
   avg = 22 / 3) :=
sorry

end avg_of_distinct_t_l160_160323


namespace ann_cate_percentage_eaten_l160_160229

-- Definitions based on conditions
def pizzas_per_person : ℕ := 4
def total_people : ℕ := 4
def total_pizzas_pieces : ℕ := total_people * pizzas_per_person
def pieces_eaten_per_person (percentage: ℕ) (total_pieces: ℕ) : ℕ := (percentage * total_pieces) / 100

-- Phrase the problem conditions
def bill_dale_eaten_percentage : ℕ := 50
def uneaten_pieces : ℕ := 6

-- Proof goal
theorem ann_cate_percentage_eaten :
  let total_pizzas_pieces := total_pizzas_pieces in
  let pieces_eaten_bill_dale := 2 * pieces_eaten_per_person bill_dale_eaten_percentage pizzas_per_person in
  let remaining_pieces := total_pizzas_pieces - pieces_eaten_bill_dale in
  let pieces_eaten_ann_cate := remaining_pieces - uneaten_pieces in
  let total_ann_cate_pieces := 2 * pizzas_per_person in
  100 * pieces_eaten_ann_cate = 75 * total_ann_cate_pieces :=
by
  sorry

end ann_cate_percentage_eaten_l160_160229


namespace candies_per_friend_l160_160430

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l160_160430


namespace domain_shifted_function_l160_160358

theorem domain_shifted_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2016 → f x ∈ set.univ) →
  (∀ x, 0 ≤ x ∧ x ≤ 2015 → f (x + 1) ∈ set.univ) :=
by sorry

end domain_shifted_function_l160_160358


namespace option_A_is_quadratic_l160_160971

def is_quadratic_equation (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (∃ h : a ≠ 0, ∀ x, eq x ↔ a * x^2 + b * x + c = 0)

theorem option_A_is_quadratic :
  is_quadratic_equation (λ x, 3 * x * (x - 4) = 0) :=
sorry

end option_A_is_quadratic_l160_160971


namespace triangle_inequality_example_l160_160849

theorem triangle_inequality_example (a b c : ℝ) (h : a ≠ b) : 
  real.sqrt ((a - c)^2 + b^2) + real.sqrt (a^2 + (b - c)^2) > real.sqrt 2 * |a - b| := 
by 
  sorry

end triangle_inequality_example_l160_160849


namespace triangle_inequality_example_l160_160848

theorem triangle_inequality_example (a b c : ℝ) (h : a ≠ b) : 
  real.sqrt ((a - c)^2 + b^2) + real.sqrt (a^2 + (b - c)^2) > real.sqrt 2 * |a - b| := 
by 
  sorry

end triangle_inequality_example_l160_160848


namespace sin_angle_D_l160_160372

variable {D E F : Type}
variable [RightTriangle DEF] (sin cos : DEF → ℝ)

theorem sin_angle_D (h₁ : angle E = 90) (h₂ : 3 * sin D = 4 * cos D) : sin D = 4 / 5 := sorry

end sin_angle_D_l160_160372


namespace four_digit_multiples_of_7_l160_160751

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160751


namespace prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l160_160013

-- Problem 1: Proof of sin(C - B) = 1 given the trigonometric identity
theorem prove_sin_c_minus_b_eq_one
  (A B C : ℝ)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  : Real.sin (C - B) = 1 := 
sorry

-- Problem 2: Proof of CD/BC given the ratios AB:AD:AC and the trigonometric identity
theorem prove_cd_div_bc_eq
  (A B C : ℝ)
  (AB AD AC BC CD : ℝ)
  (h_ratio : AB / AD = Real.sqrt 3 / Real.sqrt 2)
  (h_ratio_2 : AB / AC = Real.sqrt 3 / 1)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  (h_D_on_BC : True) -- Placeholder for D lies on BC condition
  : CD / BC = (Real.sqrt 5 - 1) / 2 := 
sorry

end prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l160_160013


namespace sequence_divisible_by_13_l160_160773

theorem sequence_divisible_by_13 (n : ℕ) (h : n ≤ 1000) : 
  ∃ m, m = 165 ∧ ∀ k, 1 ≤ k ∧ k ≤ m → (10^(6*k) + 1) % 13 = 0 := 
sorry

end sequence_divisible_by_13_l160_160773


namespace max_value_a_exists_l160_160653

section MaxValue

-- Sequence condition definitions
def seq (a : ℕ → ℤ) (n : ℕ) : Prop :=
  n > 0 ∧ a 1 = 1 ∧ ∀ n : ℕ, n > 0 → n * (a (n + 1) - a n) = a n + 1

-- Inequality condition definition
def ineq (a : ℕ → ℤ) (t : ℤ) (n : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → t ∈ (set.Icc 1 3 : set ℤ) → 
  (a n / n : ℚ) < ((2 * a t) - 1)

-- Problem statement
theorem max_value_a_exists :
  ∃ (a : ℕ → ℤ), seq a ∧ (∃ a_max : ℤ, ineq a a_max ∧ a_max = 1) :=
sorry

end MaxValue

end max_value_a_exists_l160_160653


namespace compare_abc_l160_160840

noncomputable def a : ℝ := 2 * Real.log (3 / 2)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := (1 / 2) ^ (-0.3)

theorem compare_abc : c > a ∧ a > b :=
by
  sorry

end compare_abc_l160_160840


namespace exactly_two_statements_true_l160_160987

noncomputable def f : ℝ → ℝ := sorry -- Definition of f satisfying the conditions

-- Conditions
axiom functional_eq (x : ℝ) : f (x + 3/2) + f x = 0
axiom odd_function (x : ℝ) : f (- x - 3/4) = - f (x - 3/4)

-- Proof statement
theorem exactly_two_statements_true : 
  (¬(∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T = 3/2) ∧
   (∀ (x : ℝ), f (-x - 3/4) = - f (x - 3/4)) ∧
   (¬(∀ (x : ℝ), f x = f (-x)))) :=
sorry

end exactly_two_statements_true_l160_160987


namespace count_four_digit_multiples_of_7_l160_160735

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160735


namespace sum_last_two_digits_of_x2012_l160_160858

def sequence_defined (x : ℕ → ℕ) : Prop :=
  (x 1 = 5 ∨ x 1 = 7) ∧ ∀ k ≥ 1, (x (k+1) = 5^(x k) ∨ x (k+1) = 7^(x k))

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def possible_values : List ℕ :=
  [25, 7, 43]

theorem sum_last_two_digits_of_x2012 {x : ℕ → ℕ} (h : sequence_defined x) :
  List.sum (List.map last_two_digits [25, 7, 43]) = 75 :=
  by
    sorry

end sum_last_two_digits_of_x2012_l160_160858


namespace asha_borrowed_from_mother_l160_160597

def total_money (M : ℕ) : ℕ := 20 + 40 + 70 + 100 + M

def remaining_money_after_spending_3_4 (total : ℕ) : ℕ := total * 1 / 4

theorem asha_borrowed_from_mother : ∃ M : ℕ, total_money M = 260 ∧ remaining_money_after_spending_3_4 (total_money M) = 65 :=
by
  sorry

end asha_borrowed_from_mother_l160_160597


namespace paula_candies_distribution_l160_160433

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l160_160433


namespace problem1_problem2_problem3_l160_160645

-- Definition of f_n(x)
def f_n (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), (nat.choose n k) * f (k / n) * (x ^ k) * ((1 - x) ^ (n - k))

-- Problem 1: Prove for f(x) = 1
theorem problem1 (n : ℕ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  f_n (λ x, 1) n x = 1 := 
sorry

-- Problem 2: Prove for f(x) = x
theorem problem2 (n : ℕ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  f_n (λ x, x) n x = x := 
sorry

-- Problem 3: Prove for f(x) = x^2
theorem problem3 (n : ℕ) (h : 2 ≤ n) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  f_n (λ x, x^2) n x = ((n-1) / n) * x^2 + (x / n) := 
sorry

end problem1_problem2_problem3_l160_160645


namespace new_set_average_and_std_dev_l160_160663

variables {n : ℕ} {x : ℕ → ℝ} (x̄ s : ℝ)
variables (hx : x̄ = (1 / n) * ∑ i, x i) (hs : s = Real.sqrt ((1 / n) * ∑ i, (x i - x̄) ^ 2))

theorem new_set_average_and_std_dev :
  let new_x := (λ i, 2 * x i + 1)
  (new_x̄ : ℝ) (new_s : ℝ)
  (hnew_x̄ : new_x̄ = (1 / n) * ∑ i, new_x i)
  (hnew_s : new_s = Real.sqrt ((1 / n) * ∑ i, (new_x i - new_x̄) ^ 2))
  in new_x̄ = 2 * x̄ + 1 ∧ new_s = 2 * s := 
by
  sorry

end new_set_average_and_std_dev_l160_160663


namespace number_of_always_true_inequalities_l160_160307

theorem number_of_always_true_inequalities (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  (a + c > b + d) ∧
  (¬(a - c > b - d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 - 3 > -2 - (-2))) ∧
  (¬(a * c > b * d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 * 3 > -2 * (-2))) ∧
  (¬(a / c > b / d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 / 3 > (-2) / (-2))) :=
by
  sorry

end number_of_always_true_inequalities_l160_160307


namespace total_books_on_shelves_l160_160135

def num_shelves : ℕ := 520
def books_per_shelf : ℝ := 37.5

theorem total_books_on_shelves : num_shelves * books_per_shelf = 19500 :=
by
  sorry

end total_books_on_shelves_l160_160135


namespace counties_percentage_l160_160201

theorem counties_percentage (a b c : ℝ) (ha : a = 0.2) (hb : b = 0.35) (hc : c = 0.25) :
  a + b + c = 0.8 :=
by
  rw [ha, hb, hc]
  sorry

end counties_percentage_l160_160201


namespace juice_percentage_l160_160189

theorem juice_percentage (c : ℚ) (ju : ℚ) (cups : ℕ) (h1 : ju = 3 / 4) (h2 : cups = 5) :
  (ju / cups) / c * 100 = 15 :=
by
  -- juice in each cup
  have h3 : ju / cups = 3 / 20, sorry
  -- percentage calculation
  have h4 : (ju / cups) / c * 100 = 15, sorry
  exact h4

end juice_percentage_l160_160189


namespace arrangement_exists_l160_160444

-- Define the table and players' directions
def triangular_table : Type := sorry -- Structure representing the table

-- Define the placement of numbers in the table
def arrangement (t : triangular_table) : Prop :=
  sorry -- Definition of placing -1 and 1 in the table

-- Define the sum of products for each player
def sum_of_products (t : triangular_table) (player : Type) [player.direction : Prop] : ℤ := sorry

-- Define players and their directions
inductive player | Sasha | Zhenya | Valya

-- Assertion about the sums for each player
axiom player_direction : player → Prop

-- Define the goal to be proved
theorem arrangement_exists :
  ∃ t : triangular_table, 
  sum_of_products t player.Sasha = 4 ∧
  sum_of_products t player.Zhenya = -4 ∧
  sum_of_products t player.Valya = 0 :=
sorry

end arrangement_exists_l160_160444


namespace cost_of_five_dozens_l160_160230

-- Define cost per dozen given the total cost for two dozen
noncomputable def cost_per_dozen : ℝ := 15.60 / 2

-- Define the number of dozen apples we want to calculate the cost for
def number_of_dozens := 5

-- Define the total cost for the given number of dozens
noncomputable def total_cost (n : ℕ) : ℝ := n * cost_per_dozen

-- State the theorem
theorem cost_of_five_dozens : total_cost number_of_dozens = 39 :=
by
  unfold total_cost cost_per_dozen
  sorry

end cost_of_five_dozens_l160_160230


namespace solve_for_t_l160_160083

theorem solve_for_t (t : ℕ) : 6 * 3^(2 * t) + real.sqrt (4 * 9 * 9^t) = 90 → t = 1 :=
by
  intro h
  sorry

end solve_for_t_l160_160083


namespace amount_after_two_years_is_correct_l160_160264

def annual_increase (amount : ℝ) := amount * (1 / 8)
def apply_inflation (amount : ℝ) := amount * 0.03
def apply_tax (amount : ℝ) := amount * 0.10

noncomputable def amount_after_year (initial_amount : ℝ) : ℝ :=
  let increase := annual_increase initial_amount
  let new_amount := initial_amount + increase
  let inflation := apply_inflation new_amount
  let amount_after_inflation := new_amount - inflation
  let tax := apply_tax amount_after_inflation
  amount_after_inflation - tax

noncomputable def amount_after_two_years (initial_amount : ℝ) : ℝ :=
  let first_year_amount := amount_after_year initial_amount
  amount_after_year first_year_amount

theorem amount_after_two_years_is_correct :
  amount_after_two_years 32000 = 30866.22 := by
  sorry

end amount_after_two_years_is_correct_l160_160264


namespace part1_part2_l160_160841

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 + a*x

theorem part1 (a : ℝ) :
  let f' := (fun x => (1/x) + 2*x + a)
  (f'(1) = 2) -> a = -1 :=
by {
  let f' := fun x => 1/x + 2*x + a,
  sorry,
}

theorem part2 (a : ℝ) :
  if a ≥ -2*Real.sqrt 2 then
    (∀ x > 0, f (x) a <= f (x') a -> x <= x') ∧ 
  if a < -2*Real.sqrt 2 then
    let x1 := (-a - Real.sqrt (a^2 - 8))/4,
    let x2 := (-a + Real.sqrt (a^2 - 8))/4,
    ∀ x > 0,
      (x < x1 -> f (x) a <= f (x') a -> x <= x') ∧
      (x1 < x < x2 -> f (x) a >= f (x') a -> x >= x') ∧
      (x2 < x -> f (x) a <= f (x') a -> x <= x') 
  :=
by {
  let f' := fun x => 1/x + 2*x + a
  let g := fun x => 2*x^2 + a*x + 1,
  let a := a,
  sorry,
}

end part1_part2_l160_160841


namespace count_three_digit_numbers_with_2_without_6_l160_160775

theorem count_three_digit_numbers_with_2_without_6 : 
  let total_without_6 : ℕ := 648
  let total_without_6_and_2 : ℕ := 448
  total_without_6 - total_without_6_and_2 = 200 :=
by 
  have total_without_6 := 8 * 9 * 9
  have total_without_6_and_2 := 7 * 8 * 8
  rw total_without_6
  rw total_without_6_and_2
  exact calc
    8 * 9 * 9 - 7 * 8 * 8 = 648 - 448 := by simp
    ... = 200 := by norm_num

end count_three_digit_numbers_with_2_without_6_l160_160775


namespace problem_statement_l160_160843

def line : Type := sorry
def plane : Type := sorry

def perpendicular (l : line) (p : plane) : Prop := sorry
def parallel (l1 l2 : line) : Prop := sorry

variable (m n : line)
variable (α β : plane)

theorem problem_statement (h1 : perpendicular m α) 
                          (h2 : parallel m n) 
                          (h3 : parallel n β) : 
                          perpendicular α β := 
sorry

end problem_statement_l160_160843


namespace simplify_quadratic_radical_l160_160661

theorem simplify_quadratic_radical (x y : ℝ) (h : x * y < 0) : 
  (x * real.sqrt (- (y / (x^2)))) = real.sqrt (-y) :=
sorry

end simplify_quadratic_radical_l160_160661


namespace least_d_value_l160_160901

theorem least_d_value (c d : ℕ) (hc : nat.proper_divisors c = {1, p, p^2, c}) (hd : nat.proper_divisors d = {1, q, q^2, ... , d}) (hcd : d % c = 0) :
  d = 24 :=
sorry

end least_d_value_l160_160901


namespace number_of_participants_l160_160919

theorem number_of_participants (gloves : ℕ) (h : gloves = 86) : (gloves / 2) = 43 :=
by
  -- Given that we need to arrange a minimum of 86 gloves.
  have h1 : gloves = 86 := h
  -- Each participant needs a pair of gloves (2 gloves per participant).
  have h2 : (gloves / 2) = (86 / 2) := by rw h1
  have h3 : (86 / 2) = 43 := by norm_num
  exact eq.trans h2 h3

end number_of_participants_l160_160919


namespace four_digit_multiples_of_7_l160_160701

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160701


namespace xn_plus_inv_xn_integer_l160_160844

variable (x : ℝ)
variable (hx : x + 1/x ∈ ℤ)

theorem xn_plus_inv_xn_integer (n : ℕ) (hn : n > 0) : (x^n + 1/x^n) ∈ ℤ := by
  sorry

end xn_plus_inv_xn_integer_l160_160844


namespace triangle_at_most_one_obtuse_angle_l160_160157

theorem triangle_at_most_one_obtuse_angle (T : Type) [Triangle T] : 
  (∃ A₁ A₂ A₃ : Angle T, is_obtuse A₁ ∧ is_obtuse A₂) ↔ False := sorry

end triangle_at_most_one_obtuse_angle_l160_160157


namespace incorrect_propositions_l160_160589

theorem incorrect_propositions :
  (∀ (a b : Type) (α : Type) [has_form_equal_angles_with (a b α)], ¬(a ∥ b)) ∧
  (∀ (a b : Type) (α : Type) [has_parallel (a b)] [has_parallel (a α)], ¬(b ∥ α) ∨ (b ∈ α)) ∧
  (∀ (l s : Type) (α : Type) [has_projections_within (l s α) perp_projections], ¬(l ⊥ s) ∨ (l ∈ α)) ∧
  (∀ (A B : Type) (α : Type) [has_equal_distances (A B α)], ¬(AB ∥ α) ∨ (AB ∩ α ≠ ∅) ∨ (AB ∈ α)) :=
by sorry

/- Definitions required for correct type annotations and to ensure the above theorem can be parsed and built successfully can be defined based on standard spatial terminology in mathematical context -/

class has_form_equal_angles_with (a b α : Type) := (equal_angles : Prop)
class has_parallel (a b : Type) := (parallel : Prop)
class has_parallel (a α : Type) := (parallel : Prop)
class has_projections_within (l s α : Type) := (perp_projections : Prop)
class has_equal_distances (A B α : Type) := (equal_distances : Prop)

end incorrect_propositions_l160_160589


namespace BP_over_PE_equals_2_OD_over_DA_equals_1_over_3_l160_160817

open set

variables {A B C D E O P : Type} [point A B C D E O P]

-- Definitions of distances between points as given
variables (AB AC BC BP PE OD DA : ℝ)
variables (Equi : angle_bisectors (A D) and (B E) intersect at (P))
variables (R1 : AB = 8)
variables (R2 : AC = 6)
variables (R3 : BC = 4)
variables (Circumcenter : circumcenter O of_triangle ABC on_line (A D))

-- Required proofs to be filled
theorem BP_over_PE_equals_2 :
  AB = 8 ∧ AC = 6 ∧ BC = 4 ∧ angle_bisectors (A D) and (B E) intersect at (P) → BP / PE = 2 :=
by
  sorry

theorem OD_over_DA_equals_1_over_3 : 
  circumcenter O of_triangle ABC on_line (A D) →
  OD / DA = 1/3 :=
by
  sorry

end BP_over_PE_equals_2_OD_over_DA_equals_1_over_3_l160_160817


namespace other_two_altitudes_intersect_l160_160890

-- Given conditions
variables {A B C D H₁ H₂ H₃ H₄ O : Type}
variables [Tetrahedron A B C D]
variables {AH₁ BH₂ CH₃ DH₄ : Line}

-- Altitudes and their intersections
variables [altitude AH₁ A B C D] [altitude BH₂ B A C D]
variables [altitude CH₃ C A B D] [altitude DH₄ D A B C]
variables (B_intersect_D : ∃ O, O ∈ BH₂ ∧ O ∈ DH₄) -- BH₂ and DH₄ intersect at some point O

-- Theorem statement
theorem other_two_altitudes_intersect :
  (∃ O', O' ∈ AH₁ ∧ O' ∈ CH₃) :=
sorry

end other_two_altitudes_intersect_l160_160890


namespace count_four_digit_multiples_of_7_l160_160727

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160727


namespace group_B_fluctuates_less_l160_160321

-- Conditions
def mean_A : ℝ := 80
def mean_B : ℝ := 90
def variance_A : ℝ := 10
def variance_B : ℝ := 5

-- Goal
theorem group_B_fluctuates_less :
  variance_B < variance_A :=
  by
    sorry

end group_B_fluctuates_less_l160_160321


namespace triangle_cosine_sum_l160_160012

theorem triangle_cosine_sum :
  ∃ (m n : ℕ), ∃ (ABC : Triangle) (BD : ℕ),
  (ABC.angle C = 90) ∧ 
  (BD = 17^3) ∧
  (∃ a b c : ℕ, a = ABC.side_a ∧ b = ABC.side_b ∧ c = ABC.side_c) ∧
  (Nat.gcd m n = 1) ∧ 
  (cosine_of_angle B ABC = m / n) ∧ 
  (m + n = 35) := 
sorry

end triangle_cosine_sum_l160_160012


namespace angle_between_vectors_is_135_degrees_l160_160655

noncomputable theory
open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (θ : ℝ)

hypothesis ha_unit : ∥a∥ = 1
hypothesis hb_norm : ∥b∥ = real.sqrt 6
hypothesis h_dot : inner (2 • a + b) (b - a) = 4 - real.sqrt 3

theorem angle_between_vectors_is_135_degrees :
  real.acos (inner a b / (∥a∥ * ∥b∥)) = real.pi * 3 / 4 :=
sorry

end angle_between_vectors_is_135_degrees_l160_160655


namespace ab_cd_divisible_eq_one_l160_160476

theorem ab_cd_divisible_eq_one (a b c d : ℕ) (h1 : ∃ e : ℕ, e = ab - cd ∧ (e ∣ a) ∧ (e ∣ b) ∧ (e ∣ c) ∧ (e ∣ d)) : ab - cd = 1 :=
sorry

end ab_cd_divisible_eq_one_l160_160476


namespace candies_per_friend_l160_160428

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l160_160428


namespace count_four_digit_multiples_of_7_l160_160744

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160744


namespace count_T_diff_S_l160_160777

-- Define a function to check if a digit is in a given number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ i, i < 3 ∧ (n / 10^i) % 10 = d

-- Define a function to check if a three-digit number is valid
def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the set T' of three digit numbers that do not contain a 6
def T_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6

-- Define the set S' of three digit numbers that neither contain a 2 nor a 6
def S_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6 ∧ ¬ contains_digit n 2

-- Define the set of numbers we are interested in, has 2 but not 6
def T_diff_S : {n // is_valid_three_digit n} → Prop := 
  λ n, contains_digit n 2 ∧ ¬ contains_digit n 6

-- Statement to prove
theorem count_T_diff_S : ∃ n, n = 200 ∧ (∀ (x : {n // is_valid_three_digit n}), T_diff_S x) :=
sorry

end count_T_diff_S_l160_160777


namespace value_of_a1_l160_160656

variable {a_n : ℕ → ℚ} -- the sequence a_n
variable {d : ℚ}       -- common difference

noncomputable def S (n : ℕ) := ∑ i in finset.range n, a_n i

theorem value_of_a1
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)                     -- sequencing condition
  (h_d_nonzero : d ≠ 0)                                         -- non-zero common difference
  (h_condition1 : (a_n 2) * (a_n 3) = (a_n 4) * (a_n 5))        -- given condition a2 a3 = a4 a5
  (h_sum9 : S 9 = 1)                                            -- sum of the first 9 terms equals 1
  : a_n 1 = - (5 : ℚ) / 27 := sorry

end value_of_a1_l160_160656


namespace lisa_additional_marbles_l160_160866

theorem lisa_additional_marbles (n_friends : ℕ) (initial_marbles : ℕ) (h_friends : n_friends = 12) (h_marbles : initial_marbles = 50) :
  let total_marbles_needed := (n_friends * (n_friends + 1)) / 2 in
  total_marbles_needed - initial_marbles = 28 :=
by
  sorry

end lisa_additional_marbles_l160_160866


namespace coupon_probability_l160_160498

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l160_160498


namespace valid_ternary_number_l160_160531

open Set

def ternary_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ nat.digits 10 n → d ∈ ({0, 1, 2} : Set ℕ)

theorem valid_ternary_number : ternary_digits 2012 :=
by
  sorry

end valid_ternary_number_l160_160531


namespace remaining_balance_on_phone_card_l160_160827

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l160_160827


namespace trigonometric_identity_l160_160446

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = 180) :
  ( (sin A + sin B - sin C) / (sin A + sin B + sin C) = tan (A / 2) * tan (B / 2) ) :=
by
  sorry

end trigonometric_identity_l160_160446


namespace four_digit_multiples_of_7_count_l160_160761

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160761


namespace michael_flour_removal_michael_sugar_shortage_l160_160051

def michael_needs_flour : ℕ := 6 + 2
def michael_has_flour : ℕ := 8

def michael_needs_sugar : ℚ := 3.5 + 1.5
def michael_has_sugar : ℚ := 4

theorem michael_flour_removal (has_flour needs_flour : ℕ) :
  has_flour >= needs_flour → ∃ scoops : ℕ, scoops = 0 := by
  intro h
  exists 0
  exact h

theorem michael_sugar_shortage (has_sugar needs_sugar : ℚ) :
  needs_sugar > has_sugar → ∃ shortage : ℚ, shortage = needs_sugar - has_sugar := by
  intro h
  exists needs_sugar - has_sugar
  exact h

-- Usage of the theorems with the specific conditions
example : michael_flour_removal michael_has_flour michael_needs_flour :=
  michael_flour_removal 8 8 (Nat.le_refl 8)

example : michael_sugar_shortage michael_has_sugar michael_needs_sugar :=
  michael_sugar_shortage 4 5 (by norm_num)

end michael_flour_removal_michael_sugar_shortage_l160_160051


namespace condition_parallel_planes_l160_160567

open Set -- Assuming some set-theoretic operations might be needed

noncomputable def condition_to_parallel_planes (α β : Plane) : Prop :=
  ∃ a b : Line, a ⊂ α ∧ b ⊂ β ∧ a ∥ β ∧ b ∥ α

theorem condition_parallel_planes (α β : Plane) :
  (∃ a b : Line, a ⊂ α ∧ b ⊂ β ∧ a ∥ β ∧ b ∥ α) → Parallel α β :=
sorry

end condition_parallel_planes_l160_160567


namespace ny_mets_fans_count_l160_160165

variable (Y M R : ℕ) -- Variables representing number of fans
variable (k j : ℕ)   -- Helper variables for ratios

theorem ny_mets_fans_count :
  (Y = 3 * k) →
  (M = 2 * k) →
  (M = 4 * j) →
  (R = 5 * j) →
  (Y + M + R = 330) →
  (∃ (k j : ℕ), k = 2 * j) →
  M = 88 := sorry

end ny_mets_fans_count_l160_160165


namespace probability_reach_3_0_eight_or_fewer_steps_l160_160450

-- We define the problem conditions and the required proof statement.
theorem probability_reach_3_0_eight_or_fewer_steps : 
  let q := (175 : ℚ) / 16384 in
  let m := 175 in
  let n := 16384 in
  let coprime := Nat.gcd m n = 1 in
  q = 175 / 16384 ∧ m + n = 16559 ∧ coprime := 
by
  sorry

end probability_reach_3_0_eight_or_fewer_steps_l160_160450


namespace find_g_neg2017_l160_160676

noncomputable def g : ℤ → ℝ := sorry
noncomputable def f : ℝ → ℝ :=
  λ x, if 0 < x ∧ x < 2 then log x / log 2 else g x

theorem find_g_neg2017 :
  (∀ x, g (x + 2) = -g x) →
  (∀ x, (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) → f (-x) = f x) →
  (g (-2017) = 0) :=
begin
  intros h1 h2,
  sorry
end

end find_g_neg2017_l160_160676


namespace prove_ratio_l160_160348

variable (M Q N P : ℝ)

def problem_condition1 := M = 0.40 * Q
def problem_condition2 := Q = 0.25 * P
def problem_condition3 := N = 0.60 * P

theorem prove_ratio (h1 : problem_condition1) (h2 : problem_condition2) (h3 : problem_condition3) : M / N = 1 / 6 := 
by
  sorry

end prove_ratio_l160_160348


namespace log_lt_x_l160_160353

theorem log_lt_x (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x := 
sorry

end log_lt_x_l160_160353


namespace star_star_eval_l160_160244

def star (a b : ℚ) : ℚ := a ^ b
def star_star (a b : ℚ) : ℚ := b ^ a

theorem star_star_eval : star_star (star (-3) 2) (-1) = -1 :=
by
  -- Definitions
  have h1 : star (-3) 2 = (-3) ^ 2 := rfl
  have h2 : (-3:ℚ) ^ 2 = 9 := by norm_num
  have h3 : star_star 9 (-1) = (-1) ^ 9 := rfl
  have h4 : (-1:ℚ) ^ 9 = -1 := by norm_num
  -- Combining results
  rw [h1, h2, h3, h4]
  sorry

end star_star_eval_l160_160244


namespace group_identity_elements_l160_160832

variable {G : Type*} [Group G]

theorem group_identity_elements (a b c e : G) (h1 : a⁻¹ * b * a = b ^ 2) 
(h2 : b⁻² * c * b² = c ^ 2) (h3 : c⁻³ * a * c³ = a ^ 2) (he : e = 1) : 
  a = e ∧ b = e ∧ c = e := 
sorry

end group_identity_elements_l160_160832


namespace compound_interest_double_l160_160545

theorem compound_interest_double (t : ℕ) (r : ℝ) (n : ℕ) (P : ℝ) :
  r = 0.15 → n = 1 → (2 : ℝ) < (1 + r)^t → t ≥ 5 :=
by
  intros hr hn h
  sorry

end compound_interest_double_l160_160545


namespace four_digit_multiples_of_7_l160_160704

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l160_160704


namespace ratio_BD_BO_l160_160886

noncomputable def sqrt2 := Real.sqrt 2
noncomputable def sqrt6 := Real.sqrt 6

variables (O A C B D : Point) (r : ℝ) (circle : Circle O r)

-- Condition 1: Points A and C are on a circle centered at O
axiom h1 : circle.contains A
axiom h2 : circle.contains C

-- Condition 2: Lines BA and BC are tangent to the circle
axiom h3 : Tangent circle B A
axiom h4 : Tangent circle B C

-- Condition 3: Triangle ABC is an isosceles right triangle with ∠ACB = 90°
axiom h5 : Angle A C B = 90
axiom h6 : Isosceles A C B
axiom h7 : RightTriangle A C B

-- Condition 4: The circle intersects line BO at D
axiom h8 : circle.intersectLine O B D

-- Prove that BD / BO = 1 - (sqrt(2) + sqrt(6)) / 4
theorem ratio_BD_BO (O A C B D : Point) (r : ℝ) (circle : Circle O r) [h1 : circle.contains A]
  [h2 : circle.contains C] [h3 : Tangent circle B A] [h4 : Tangent circle B C]
  [h5 : Angle A C B = 90] [h6 : Isosceles A C B] [h7 : RightTriangle A C B]
  [h8 : circle.intersectLine O B D] :
  (BD_length / BO_length = 1 - (sqrt2 + sqrt6) / 4) :=
sorry

end ratio_BD_BO_l160_160886


namespace triangle_area_gt_half_l160_160508

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l160_160508


namespace count_four_digit_multiples_of_7_l160_160769

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160769


namespace smallest_real_root_of_equation_l160_160355

theorem smallest_real_root_of_equation :
  ∃ a ∈ ℝ, (∀ x ∈ ℝ, (sqrt (x * (x + 1) * (x + 2) * (x + 3) + 1) = 71 → x ≥ a)) ∧ a = -10 :=
begin
  sorry
end

end smallest_real_root_of_equation_l160_160355


namespace tom_catches_up_with_jerry_after_40_minutes_l160_160507

-- Define the parameter lengths of the loops and the speeds
def m : ℝ := sorry
def v_T : ℝ := m / 10
def v_J : ℝ := m / 20

-- Define the time(s) given in the problem
def initial_time_to_m : ℝ := 20
def additional_time_to_starting_point : ℝ := 10

-- Define the total loop distances
def small_loop_length : ℝ := m
def large_loop_length : ℝ := 2 * m

-- Hypotheses from the problem conditions
axiom same_direction_constant_speed : true
axiom initially_jerry_above_tom : true
axiom tom_above_jerry_after_20_minutes (t : ℝ) (j : ℝ) : t = large_loop_length ∧ j = small_loop_length
axiom tom_at_starting_point_after_30_minutes (t : ℝ) : t = large_loop_length + small_loop_length

-- Define the relative speed and calculate the chase duration
def relative_speed : ℝ := v_T - v_J
def chase_distance : ℝ := 2 * m
def catch_up_time : ℝ := chase_distance / relative_speed

-- The theorem to prove that Tom catches up with Jerry after 40 minutes
theorem tom_catches_up_with_jerry_after_40_minutes : catch_up_time = 40 := 
by sorry

end tom_catches_up_with_jerry_after_40_minutes_l160_160507


namespace count_four_digit_multiples_of_7_l160_160729

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160729


namespace average_water_milk_coffee_l160_160099

theorem average_water_milk_coffee (avg_water_milk : ℕ) (coffee : ℕ) (avg_water_milk = 94) (coffee = 100) :
  (avg_water_milk * 2 + coffee) / 3 = 96 :=
by
sory

end average_water_milk_coffee_l160_160099


namespace chloe_points_lost_l160_160238

def total_points_after_two_rounds (points_first_round: ℕ) (points_second_round : ℕ) : ℕ :=
  points_first_round + points_second_round

def points_lost (total_points: ℕ) (final_points: ℕ) : ℕ :=
  total_points - final_points

theorem chloe_points_lost :
  ∀ (points_first_round points_second_round final_points : ℕ), 
  points_first_round = 40 → points_second_round = 50 → final_points = 86→ 
  points_lost (total_points_after_two_rounds points_first_round points_second_round) final_points = 4 :=
by 
  intros points_first_round points_second_round final_points 
   intros h1 h2 h3 
     rw [h1, h2, h3]
     dsimp 
     sorry

end chloe_points_lost_l160_160238


namespace vehicle_speeds_l160_160263

noncomputable def solve_vehicle_speeds : ℕ × ℕ :=
  let x := 30 in
  let y := x + 35 in
  (x, y)

theorem vehicle_speeds :
  ∃ x y : ℕ, (x = 30 ∧ y = x + 35 ∧ y = 65) ∧
  (∀ (d_eb d_car : ℕ), d_eb = 13 → d_car = 6 → 
  d_eb / x = d_car / y) :=
begin
  use 30,
  use 65,
  split,
  { split,
    { refl },
    { split,
      { simp },
      { refl } }},
  intros d_eb d_car,
  intros h1 h2,
  rw [h1, h2],
  norm_num,
end

end vehicle_speeds_l160_160263


namespace prime_iff_satisfies_condition_l160_160442

def satisfies_condition (n : ℕ) : Prop :=
  if n = 2 then True
  else if 2 < n then ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬ (k ∣ n)
  else False

theorem prime_iff_satisfies_condition (n : ℕ) : Prime n ↔ satisfies_condition n := by
  sorry

end prime_iff_satisfies_condition_l160_160442


namespace coupon_probability_l160_160501

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l160_160501


namespace paula_candies_distribution_l160_160432

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l160_160432


namespace midpoints_coincide_l160_160001

-- Define the elements of the problem
variables {α : ℝ} (A B C D M : Point) -- Points A, B, C, D for the rectangle, M for any midpoint
-- Rectangles and Point definitions
structure Rectangle :=
  (A B C D : Point)
  (side_AB : segment A B)
  (side_CD : segment C D)
  (side_AD : segment A D)
  (side_BC : segment B C)
  (parallel_AB_CD : parallel side_AB side_CD)
  (parallel_AD_BC : parallel side_AD side_BC)

def isosceles_triangle (α : ℝ) (P Q R : Point) : Prop :=
  P ≠ Q ∧ P ≠ R ∧ (∠ P Q R = α ) ∧ (dist P Q = dist P R)

def midpoint (P Q : Point) : Point := sorry -- midpoint computation

variables (rect : Rectangle) (P : Point) (α : ℝ)
  -- Assume we have a point on a segment and construct an isosceles triangle
  (hP : P = rect.A ∨ P = rect.B ∨ P = rect.C ∨ P = rect.D) 
  (Q R : Point) 
  (h_triangle : isosceles_triangle α P Q R)
  (hQ : on_segment rect.side_AB Q ∨ on_segment rect.side_AD Q ∨ on_segment rect.side_BC Q ∨ on_segment rect.side_CD Q)
  (hR : on_segment rect.side_AB R ∨ on_segment rect.side_AD R ∨ on_segment rect.side_BC R ∨ on_segment rect.side_CD R)

theorem midpoints_coincide :
  ∀ (P Q R S T U : Point) (α : ℝ), isosceles_triangle α P Q R → isosceles_triangle α S T U → midpoint Q R = midpoint T U :=
by
  intros P Q R S T U α h1 h2
  sorry -- The proof would be done here.

end midpoints_coincide_l160_160001


namespace four_digit_multiples_of_7_l160_160709

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160709


namespace cost_of_5_8_minute_call_l160_160606

def tariff_function (m : ℝ) : ℝ :=
  1.06 * (3 / 4 * ⌊ m ⌋ + 7 / 4)

theorem cost_of_5_8_minute_call : tariff_function 5.8 = 5.83 := by
  sorry

end cost_of_5_8_minute_call_l160_160606


namespace derivative_of_givenFunction_l160_160989

noncomputable def givenFunction (x : ℝ) : ℝ :=
  real.tan (real.log (1 / 3)) + (1 / 4) * (real.sin (4 * x))^2 / (real.cos (8 * x))

theorem derivative_of_givenFunction (x : ℝ) :
  deriv givenFunction x = (real.tan (8 * x)) / (real.cos (8 * x)) :=
by
  sorry

end derivative_of_givenFunction_l160_160989


namespace max_fraction_sum_l160_160403

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_fraction_sum_l160_160403


namespace smallest_prime_p_q_l160_160311

open Nat

-- Given conditions
axiom prime_p : Nat.prime 53
axiom prime_q : Nat.prime 2
axiom prime_pq_plus_1 : Nat.prime ((53 * 2) + 1)
axiom p_minus_q_gt_40 : (53 - 2) > 40

theorem smallest_prime_p_q : 
  (∀ p q : ℕ, Nat.prime p → Nat.prime q → Nat.prime (p * q + 1) → (p - q > 40) → (p = 53 ∧ q = 2)) :=
by
  intro p q h_prime_p h_prime_q h_prime_pq_plus_1 h_p_minus_q_gt_40
  sorry

end smallest_prime_p_q_l160_160311


namespace count_four_digit_multiples_of_7_l160_160771

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160771


namespace home_run_difference_l160_160588

def hank_aaron_home_runs : ℕ := 755
def dave_winfield_home_runs : ℕ := 465

theorem home_run_difference :
  2 * dave_winfield_home_runs - hank_aaron_home_runs = 175 := by
  sorry

end home_run_difference_l160_160588


namespace roots_reciprocal_sum_l160_160038

theorem roots_reciprocal_sum
  {a b c : ℂ}
  (h_roots : ∀ x : ℂ, (x - a) * (x - b) * (x - c) = x^3 - x + 1) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = -2 :=
by
  sorry

end roots_reciprocal_sum_l160_160038


namespace paula_candies_l160_160435

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l160_160435


namespace four_digit_multiples_of_7_l160_160748

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160748


namespace count_four_digit_multiples_of_7_l160_160732

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160732


namespace diameter_of_triple_sphere_l160_160107

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end diameter_of_triple_sphere_l160_160107


namespace diameter_of_triple_sphere_l160_160108

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end diameter_of_triple_sphere_l160_160108


namespace find_51st_permutation_l160_160951

def digits : List ℕ := [1, 4, 5, 8, 9]

def is_valid_permutation (p : List ℕ) : Prop :=
  (∀ d ∈ p, d ∈ digits) ∧ (∀ d₁ d₂ ∈ p, d₁ = d₂ → d₁ = d₂) ∧ p.length = 5

def nth_permutation (n : ℕ) : List ℕ :=
  (List.permutations digits).nth! (n - 1)

theorem find_51st_permutation :
  nth_permutation 51 = [5, 1, 8, 4, 9] := 
    sorry

end find_51st_permutation_l160_160951


namespace find_missing_term_l160_160605

theorem find_missing_term (a b : ℕ) : ∃ x, (2 * a - b) * x = 4 * a^2 - b^2 :=
by
  use (2 * a + b)
  sorry

end find_missing_term_l160_160605


namespace sum_of_distances_l160_160836

theorem sum_of_distances (AB A'B' AD A'D' x y : ℝ) 
  (h1 : AB = 8)
  (h2 : A'B' = 6)
  (h3 : AD = 3)
  (h4 : A'D' = 1)
  (h5 : x = 2)
  (h6 : x / y = 3 / 2) : 
  x + y = 10 / 3 :=
by
  sorry

end sum_of_distances_l160_160836


namespace coupon_probability_l160_160502

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l160_160502


namespace maximum_value_of_z_l160_160900

noncomputable def satisfy_constraints (x y : ℝ) : Prop :=
  (x - y ≥ 0) ∧ (x + 2y ≤ 3) ∧ (x - 2y ≤ 1)

theorem maximum_value_of_z (x y : ℝ) (hx : satisfy_constraints x y) : 
  x + 6 * y ≤ 7 :=
sorry

end maximum_value_of_z_l160_160900


namespace max_prime_saturated_value_less_than_l160_160198

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : set ℕ :=
  { p ∈ finset.range n | is_prime p ∧ p ∣ n }

def product_of_prime_factors (n : ℕ) : ℕ :=
  finset.prod (finset.filter (λ p, p ∈ prime_factors n) finset.univ) (λ x, x)

def is_prime_saturated (n : ℕ) : Prop :=
  product_of_prime_factors n < n

theorem max_prime_saturated_value_less_than (e : ℕ) (h : is_prime_saturated 96) : product_of_prime_factors 96 < 96 :=
by
  sorry

end max_prime_saturated_value_less_than_l160_160198


namespace range_of_a_l160_160316

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l160_160316


namespace radius_of_circle_l160_160186

-- Conditions
variables (M N : ℝ)
variables (pi : ℝ)
variable (r : ℝ)

-- Given conditions in mathematical terms
def area_of_circle := M = pi * r^2
def circumference_of_circle := N = 2 * pi * r
def ratio_condition := M / N = 40

-- Statement to prove: The radius of the circle equals 80 cm
theorem radius_of_circle (h1 : area_of_circle M N pi r) 
                         (h2 : circumference_of_circle M N pi r) 
                         (h3 : ratio_condition M N pi r)
                         : r = 80 :=
sorry

end radius_of_circle_l160_160186


namespace two_colorable_regions_l160_160876

-- Define what it means for regions formed by n lines to be 2-colorable on the plane
theorem two_colorable_regions (n : ℕ) (h : n ≥ 1) : 
  ∃ (color : Set (Set ℝ) → Bool), 
  ∀ (R1 R2 : Set ℝ), (adjacent R1 R2) → (color R1 ≠ color R2) := by 
sorry

end two_colorable_regions_l160_160876


namespace marilyn_ends_up_with_55_caps_l160_160423

def marilyn_initial_caps := 165
def caps_shared_with_nancy := 78
def caps_received_from_charlie := 23

def remaining_caps (initial caps_shared caps_received: ℕ) :=
  initial - caps_shared + caps_received

def caps_given_away (total_caps: ℕ) :=
  total_caps / 2

def final_caps (initial caps_shared caps_received: ℕ) :=
  remaining_caps initial caps_shared caps_received - caps_given_away (remaining_caps initial caps_shared caps_received)

theorem marilyn_ends_up_with_55_caps :
  final_caps marilyn_initial_caps caps_shared_with_nancy caps_received_from_charlie = 55 :=
by
  sorry

end marilyn_ends_up_with_55_caps_l160_160423


namespace number_of_four_digit_multiples_of_7_l160_160697

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160697


namespace scalar_product_AD_BC_is_1705_l160_160909

-- Define the trapezoid and its properties
variables {A B C D O : Point}
variables (AB CD : ℝ)
variables (a b : Vector)
variables h1 : AB = 55
variables h2 : CD = 31
variables h3 : a ∘ b = 0  -- Diagonals AC and BD are perpendicular

-- Define the scalar product of vectors $\overrightarrow{AD}$ and $\overrightarrow{BC}$
noncomputable def scalar_product_AD_BC : ℝ :=
  let AD := a + (31 / 55) • b in 
  let BC := b + (31 / 55) • a in
  (AD ∘ BC)

-- The proof statement that the scalar product of $\overrightarrow{AD}$ and $\overrightarrow{BC}$ is $1705$
theorem scalar_product_AD_BC_is_1705 : scalar_product_AD_BC a b = 1705 :=
  sorry

end scalar_product_AD_BC_is_1705_l160_160909


namespace line_parallel_to_plane_l160_160322

-- Definitions based on conditions
def normal_vector (n : ℝ × ℝ × ℝ) := n = (2, -2, 4)
def vector_AB (AB : ℝ × ℝ × ℝ) := AB = (-3, 1, 2)

-- Prove that the line AB is parallel to plane α
theorem line_parallel_to_plane (A_not_on_plane : Type) :
  ∀ (n AB : ℝ × ℝ × ℝ),
  normal_vector n →
  vector_AB AB →
  n.1 * AB.1 + n.2 * AB.2 + n.3 * AB.3 = 0 →
  AB∥α :=
by
  intros n AB hn hAB hdot
  sorry

end line_parallel_to_plane_l160_160322


namespace minimum_positive_period_of_given_function_l160_160927

def function_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

def given_function (x : ℝ) : ℝ := 
  (Real.sin x) * (1 + (Real.tan x) * (Real.tan (x / 2)))

theorem minimum_positive_period_of_given_function : function_period given_function (2 * Real.pi) :=
by
  sorry

end minimum_positive_period_of_given_function_l160_160927


namespace find_ellipse_details_l160_160302

noncomputable def ellipse_through_points (A B : (ℝ × ℝ)) : Prop :=
  A = (2, -4 * real.sqrt 5 / 3) ∧ B = (-1, 8 * real.sqrt 2 / 3)

theorem find_ellipse_details :
  ellipse_through_points (2, -4 * real.sqrt 5 / 3) (-1, 8 * real.sqrt 2 / 3) →
  ∃ m n : ℝ,
    m = 1 / 9 ∧ n = 1 / 16 ∧
    ((λ x y : ℝ, m * x^2 + n * y^2 = 1) = (λ x y, x^2 / 9 + y^2 / 16 = 1)) ∧
    (vertices = [(3, 0), (-3, 0), (0, 4), (0, -4)]) ∧
    (foci = [(0, real.sqrt 7), (0, -real.sqrt 7)]) ∧
    (eccentricity = real.sqrt 7 / 4) :=
sorry

end find_ellipse_details_l160_160302


namespace arith_seq_ratio_a7_b7_l160_160686

variable {α : Type*} [OrderedField α] 
variable {a : ℕ → α} {b : ℕ → α}
variable {S T : ℕ → α}

-- Define arithmetic sequences a_n and b_n
def arithmetic_sequence_a (n : ℕ) : α := a n
def arithmetic_sequence_b (n : ℕ) : α := b n

-- Define sums S_n and T_n of the first n terms of the sequences
def sum_S (n : ℕ) : α := S n
def sum_T (n : ℕ) : α := T n

-- Given condition
axiom sum_ratio (n : ℕ) : (sum_S n) / (sum_T n) = (2 * n + 3) / (3 * n - 1)

-- Proof statement
theorem arith_seq_ratio_a7_b7 : (arithmetic_sequence_a 7) / (arithmetic_sequence_b 7) = (29 / 38) := 
sorry

end arith_seq_ratio_a7_b7_l160_160686


namespace correctness_statement_l160_160883

noncomputable def problem_conditions (person : Type) (scores : list ℝ) (mean variance : ℝ) : Prop :=
  list.length scores = 10 ∧
  (list.sum scores / 10) = mean ∧
  (∑ x in scores, (x - mean) ^ 2 / 10) = variance

theorem correctness_statement (A_scores B_scores : list ℝ) :
  problem_conditions personA A_scores 8 1.2 →
  problem_conditions personB B_scores 8 1.6 →
  ¬ (∃ mode_A mode_B, mode_A = mode_B ∧
       mode_A ∈ A_scores ∧ mode_B ∈ B_scores) :=
by
  intros hA hB
  sorry

end correctness_statement_l160_160883


namespace area_trapezoid_DEFG_l160_160003

theorem area_trapezoid_DEFG {A B C D E F G : Type*} 
    (h1 : rectangle ABCD)
    (h2 : midpoint E AD)
    (h3 : segment G CD 1/3 1)
    (h4 : area ABCD = 108)
    (h5 : midpoint F BC) :
    area DEFG = 27 := 
sorry

end area_trapezoid_DEFG_l160_160003


namespace quiz_answer_key_combinations_l160_160377

noncomputable def num_ways_answer_key : ℕ :=
  let true_false_combinations := 2^4
  let valid_true_false_combinations := true_false_combinations - 2
  let multi_choice_combinations := 4 * 4
  valid_true_false_combinations * multi_choice_combinations

theorem quiz_answer_key_combinations : num_ways_answer_key = 224 := 
by
  sorry

end quiz_answer_key_combinations_l160_160377


namespace num_four_digit_multiples_of_7_l160_160725

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160725


namespace roxanne_change_l160_160893

theorem roxanne_change :
  let price_lemonade := 2 * 2
  let price_sandwich := 2 * 2.50
  let price_watermelon := 1.25
  let price_chips := 1.75
  let price_cookies := 3 * 0.75
  let total_cost := price_lemonade + price_sandwich + price_watermelon + price_chips + price_cookies
  let payment := 50
  payment - total_cost = 35.75 :=
by
  let price_lemonade := 2 * 2
  let price_sandwich := 2 * 2.50
  let price_watermelon := 1.25
  let price_chips := 1.75
  let price_cookies := 3 * 0.75
  let total_cost := price_lemonade + price_sandwich + price_watermelon + price_chips + price_cookies
  let payment := 50
  sorry

end roxanne_change_l160_160893


namespace debit_card_more_advantageous_l160_160457

theorem debit_card_more_advantageous (N : ℕ) (cost_of_tickets : ℝ) (annual_interest_rate : ℝ) (cashback_credit_card : ℝ) (cashback_debit_card : ℝ) (days_per_year : ℝ) (days_per_month : ℝ) :
  cost_of_tickets = 12000 ∧
  annual_interest_rate = 0.06 ∧
  cashback_credit_card = 0.01 ∧
  cashback_debit_card = 0.02 ∧
  days_per_year = 365 ∧
  days_per_month = 30 →
  N ≤ 6 :=
begin
  sorry
end

end debit_card_more_advantageous_l160_160457


namespace region_inside_rectangle_outside_circles_l160_160075

/-- Rectangle EFGH has sides EF = 4 and FG = 6. A circle of radius 1.5 is 
centered at E, a circle of radius 2.5 is centered at F, and a circle of 
radius 3.5 is centered at G. Prove that the area inside the rectangle 
but outside all three circles is 7.7. -/
theorem region_inside_rectangle_outside_circles (EF FG : ℝ) (rE rF rG : ℝ)
  (hEF : EF = 4) (hFG : FG = 6) (hrE : rE = 1.5) (hrF : rF = 2.5) (hrG : rG = 3.5) :
  let area_rectangle := EF * FG in
  let area_circles := (π * rE^2) / 4 + (π * rF^2) / 4 + (π * rG^2) / 4 in
  let area_outside_circles := area_rectangle - area_circles in
  area_outside_circles ≈ 7.7 :=
by
  sorry

end region_inside_rectangle_outside_circles_l160_160075


namespace nine_digit_pqrstuvw_l160_160468

theorem nine_digit_pqrstuvw (P Q R S T U V W X : ℕ)
  (h1 : {P, Q, R, S, T, U, V, W, X}.toFinset = {1, 2, 3, 4, 5, 6, 7, 8, 9}.toFinset)
  (h2: (P * 1000 + Q * 100 + R * 10 + S) % 6 = 0)
  (h3: S = 5)
  (h4: R * 100 + S * 10 + T % 5 = 0)
  (h5: (R * 1000 + S * 100 + T * 10 + U) % 9 = 0) :
  P = 7 := by
  sorry

end nine_digit_pqrstuvw_l160_160468


namespace solution_to_problem_l160_160527

def problem_statement : Prop :=
  (2.017 * 2016 - 10.16 * 201.7 = 2017)

theorem solution_to_problem : problem_statement :=
by
  sorry

end solution_to_problem_l160_160527


namespace shaded_area_of_circles_l160_160259

def equilateralTriangle (A B C : Type) := (dist A B = 6) ∧ (dist B C = 6) ∧ (dist C A = 6)

def arithmeticSequence (r_A r_B r_C : ℝ) := (r_A < r_B) ∧ (r_B < r_C) ∧ (r_B = r_A + (r_C - r_A) / 2)

def shortestDistanceCircles (r_A r_B r_C : ℝ) :=
  ((6 - (r_A + r_B) = 3.5) ∧ (6 - (r_A + r_C) = 3))

noncomputable def shadedRegionArea (r_A r_B r_C : ℝ) :=
  (π * r_A^2 / 6 + π * r_B^2 / 6 + π * r_C^2 / 6)

theorem shaded_area_of_circles :
  ∀ (A B C : Type) (r_A r_B r_C : ℝ),
    equilateralTriangle A B C →
    arithmeticSequence r_A r_B r_C →
    shortestDistanceCircles r_A r_B r_C →
    shadedRegionArea r_A r_B r_C = 29 * π / 24 :=
begin
  -- Lean proof steps go here
  sorry
end

end shaded_area_of_circles_l160_160259


namespace percentage_apartments_at_least_two_residents_l160_160364
noncomputable theory

def totalApartments : ℕ := 120
def percentAtLeastOneResident : ℝ := 0.85
def apartmentsOnlyOneResident : ℕ := 30

theorem percentage_apartments_at_least_two_residents 
  (total : ℕ) 
  (percent_one : ℝ) 
  (one_resident : ℕ) 
  (h_total : total = totalApartments) 
  (h_percent_one : percent_one = percentAtLeastOneResident) 
  (h_one_resident : one_resident = apartmentsOnlyOneResident) 
  : (percent_one * total).toNat - one_resident = 72 := sorry

end percentage_apartments_at_least_two_residents_l160_160364


namespace max_abs_diff_2268_l160_160118

theorem max_abs_diff_2268 
  (a b : List ℕ)
  (h1 : 2268 = (a.map Nat.factorial).prod / (b.map Nat.factorial).prod)
  (h2 : (∀ i j : ℕ, i < j → a.get i ≥ a.get j ∧ b.get i ≥ b.get j)) 
  : |a.head.getD 0 - b.head.getD 0| = 7 :=
by 
  sorry

end max_abs_diff_2268_l160_160118


namespace minimum_visible_sum_is_144_l160_160179

-- Define the conditions
def die_opposite_sides_sum_to_seven (x y : ℕ) : Prop :=
  x + y = 7

def corner_cube_visible_face_sum_min : ℕ :=
  1 + 2 + 3

def edge_cube_visible_face_sum_min : ℕ :=
  1 + 2

def face_center_cube_visible_face_sum_min : ℕ :=
  1

-- A 4x4x4 cube made of 64 dice
def large_cube_structure :=
  let corners := 8 in
  let edges := 24 in
  let face_centers := 24 in
  (corners, edges, face_centers)

-- Calcuate the total minimum sum for the visible faces
def minimum_visible_sum : ℕ :=
  let (corners, edges, face_centers) := large_cube_structure in
  (corners * corner_cube_visible_face_sum_min +
   edges * edge_cube_visible_face_sum_min +
   face_centers * face_center_cube_visible_face_sum_min)

-- The proof problem
theorem minimum_visible_sum_is_144 :
  minimum_visible_sum = 144 := by
  -- This line represents the proof which is omitted as per instructions
  sorry

end minimum_visible_sum_is_144_l160_160179


namespace sum_of_interior_angles_convex_polyhedron_l160_160525

-- Define the sum of the interior angles function
def sum_of_interior_angles (V : ℕ) : ℝ := (V - 2) * 180

-- Main theorem statement
theorem sum_of_interior_angles_convex_polyhedron (V : ℕ) (hV : V = 30) : sum_of_interior_angles V = 5040 :=
by
  rw hV
  rw [sum_of_interior_angles]
  sorry

end sum_of_interior_angles_convex_polyhedron_l160_160525


namespace geometry_problem_l160_160386

open Classical
noncomputable theory

variables {A B C D O O₁ O₂ : Type} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty O] [Nonempty O₁] [Nonempty O₂]

-- Given variables
variables (AC : A → C → Prop) (BD : B → D → Prop) (circumcircle_ABC : Triangle → Circle) 
          (tangent_circle_to_segments_AD_BD_circumcircle_ABC : Circle → Prop) 
          (tangent_circle_to_segments_CD_BD_circumcircle_ABC : Circle → Prop)

-- Defining points
variable (D_on_AC : D)
variable (O₁_center : O₁)
variable (O₂_center : O₂)
variable (incenter_O : O)

-- Assuming tangency conditions
variables (tangent1 : tangent_circle_to_segments_AD_BD_circumcircle_ABC O₁_center)
          (tangent2 : tangent_circle_to_segments_CD_BD_circumcircle_ABC O₂_center)

-- Angle variable
variable (varphi : Angle)

-- The theorem to be proved
theorem geometry_problem :
  (∃ line_through_O₁O₂ : Line, line_through_O₁O₂ O₁_center O₂_center incenter_O) ∧
  (∃ ratio_eq_tan_squared : ℝ, ratio_eq_tan_squared = (|O₁_center, incenter_O| : |incenter_O, O₂_center|) = tan^2 (varphi / 2)) :=
by
  sorry

end geometry_problem_l160_160386


namespace max_ab_minus_ba_l160_160930

-- Conditions
def valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10
def distinct (a b : ℕ) : Prop := a ≠ b

-- Problem statement
theorem max_ab_minus_ba (a b : ℕ) (h_a : valid_digit a) (h_b : valid_digit b) (h_distinct : distinct a b) : 
  10 * a + b - (10 * b + a) ≤ 72 :=
begin
  sorry
end

end max_ab_minus_ba_l160_160930


namespace limit_of_derivative_l160_160665

variable {𝕜 : Type*} [NormedField 𝕜] {E : Type*} [NormedSpace 𝕜 E] {f : 𝕜 → E} {a A : 𝕜}

theorem limit_of_derivative (h : HasDerivAt f A a) : 
  filter.tendsto (λ Δx, (f (a + Δx) - f (a - Δx)) / Δx) (nhds_within 0 (set.Ioo (-(1 : 𝕜)) (1 : 𝕜))) (𝓝 (2 * A)) :=
sorry

end limit_of_derivative_l160_160665


namespace four_digit_multiples_of_7_count_l160_160757

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l160_160757


namespace transformed_expression_value_l160_160548

-- Defining the new operations according to the problem's conditions
def new_minus (a b : ℕ) : ℕ := a + b
def new_plus (a b : ℕ) : ℕ := a * b
def new_times (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

-- Problem statement
theorem transformed_expression_value : new_minus 6 (new_plus 9 (new_times 8 (new_div 3 25))) = 5 :=
sorry

end transformed_expression_value_l160_160548


namespace B_completion_time_l160_160562

-- Definitions for work rates and combined rate
def A_work_rate := 1 / 9
def B_work_rate (x : ℝ) := 1 / x
def combined_work_rate (x : ℝ) := A_work_rate + B_work_rate x

-- Condition that combined work rate is the equivalent of finishing the work in 6 days
def combined_work_rate_condition (x : ℝ) := combined_work_rate x = 1 / 6

-- The main theorem stating B's completion time
theorem B_completion_time : ∃ x : ℝ, combined_work_rate_condition x ∧ x = 18 :=
by
  sorry

end B_completion_time_l160_160562


namespace basketball_tournament_impossible_l160_160363

theorem basketball_tournament_impossible (n : ℕ) :
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 →
  let total_matches := (5 * n * (5 * n - 1)) / 2
  in ∀ x : ℕ, total_matches ≠ 7 * x :=
begin
  intros h_cases total_matches x,
  have H : total_matches = (5 * n * (5 * n - 1)) / 2, from rfl,
  cases h_cases,
  { -- Case n = 2
    rw h_cases at H,
    calc (5 * 2 * (5 * 2 - 1)) / 2 = 90 / 2 : by simp
                          ... = 45,
    -- 45 is not divisible by 7
    have : 45 % 7 ≠ 0, by norm_num,
    exact mt (congr_arg (λ x, 45 / x * 7 = 45)) this },
  { -- Case n = 3
    rw h_cases at H,
    calc (5 * 3 * (5 * 3 - 1)) / 2 = 210 / 2 : by simp
                        ... = 105,
    -- 105 is not divisible by 7
    have : 105 % 7 ≠ 0, by norm_num,
    exact mt (congr_arg (λ x, 105 / x * 7 = 105)) this },
  { -- Case n = 4
    rw h_cases at H,
    calc (5 * 4 * (5 * 4 - 1)) / 2 = 380 / 2 : by simp
                        ... = 190,
    -- 190 is not divisible by 7
    have : 190 % 7 ≠ 0, by norm_num,
    exact mt (congr_arg (λ x, 190 / x * 7 = 190)) this },
  { -- Case n = 5
    rw h_cases at H,
    calc (5 * 5 * (5 * 5 - 1)) / 2 = 600 / 2 : by simp
                       ... = 300,
    -- 300 is not divisible by 7
    have : 300 % 7 ≠ 0, by norm_num,
    exact mt (congr_arg (λ x, 300 / x * 7 = 300)) this }
end

end basketball_tournament_impossible_l160_160363


namespace quadrilateral_area_of_isosceles_trapezoid_l160_160042

noncomputable def isosceles_trapezoid_area : ℝ :=
  let AB := 17
  let BC := 25
  let DA := 25
  let CD := 31
  let AP := PQ / 2 := 12.5 -- Not explicitly used in the end but included for completeness
  let PQ := 25 -- Diameter of the circle
  let height := 24 -- Derived using the Pythagorean theorem in the right triangle problem
  let width := 7 -- Half the difference in lengths of AB and CD
  (height * width)

theorem quadrilateral_area_of_isosceles_trapezoid :
  isosceles_trapezoid_area = 168 := by
  sorry

end quadrilateral_area_of_isosceles_trapezoid_l160_160042


namespace sin_equation_necessary_not_sufficient_l160_160359

variable {α β γ : ℝ}

def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
  2 * β = α + γ

theorem sin_equation_necessary_not_sufficient :
  (∀ α β γ, is_arithmetic_sequence α β γ → sin (α + γ) = sin (2 * β)) →
  (∀ α β γ, (sin (α + γ) = sin (2 * β)) → is_arithmetic_sequence α β γ) → False :=
begin
  intro h1,
  intro h2,
  -- example where sin(α + γ) = sin(2 * β), but not necessarily an arithmetic sequence
  have counter_example : ∃ α β γ : ℝ, sin (α + γ) = sin (2 * β) ∧ ¬ is_arithmetic_sequence α β γ,
  { use [0, π/2, -π],
    split,
    { simp, rw[sin_neg, sin_pi_div_two, sin_zero], },
    { unfold is_arithmetic_sequence, linarith, }
  },
  cases counter_example with α hα,
  cases hα with β hα,
  cases hα with γ hα,
  specialize h2 α β γ hα.1,
  contradiction
end

end sin_equation_necessary_not_sufficient_l160_160359


namespace rhombus_diagonals_l160_160125

theorem rhombus_diagonals (p d_sum : ℝ) (h₁ : p = 100) (h₂ : d_sum = 62) :
  ∃ d₁ d₂ : ℝ, (d₁ + d₂ = d_sum) ∧ (d₁^2 + d₂^2 = (p/4)^2 * 4) ∧ ((d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48)) :=
by
  sorry

end rhombus_diagonals_l160_160125


namespace part1_part2_l160_160838

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1 + cos x, 1 + sin x)
def vector_b : ℝ × ℝ := (1, 0)
def vector_c : ℝ × ℝ := (1, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 * u.1 + u.2 * u.2)

theorem part1 (x : ℝ) : dot_product (vector_sub (vector_a x) vector_b) (vector_sub (vector_a x) vector_c) = 0 := by
  sorry

theorem part2 (x : ℝ) (k : ℤ) (h : x = 2 * k * π + π / 4) : magnitude (vector_a x) = sqrt 2 + 1 := by
  sorry

end part1_part2_l160_160838


namespace quadrilateral_area_inequality_l160_160539

theorem quadrilateral_area_inequality
  (a b c d S : ℝ)
  (hS : 0 ≤ S)
  (h : S = (a + b) / 4 * (c + d) / 4)
  : S ≤ (a + b) / 4 * (c + d) / 4 := by
  sorry

end quadrilateral_area_inequality_l160_160539


namespace roots_of_transformed_quadratic_l160_160335

theorem roots_of_transformed_quadratic (a b p q s1 s2 : ℝ)
    (h_quad_eq : s1 ^ 2 + a * s1 + b = 0 ∧ s2 ^ 2 + a * s2 + b = 0)
    (h_sum_roots : s1 + s2 = -a)
    (h_prod_roots : s1 * s2 = b) :
        p = -(a ^ 4 - 4 * a ^ 2 * b + 2 * b ^ 2) ∧ 
        q = b ^ 4 :=
by
  sorry

end roots_of_transformed_quadratic_l160_160335


namespace circumscribed_circle_radius_triangle_753_l160_160583

theorem circumscribed_circle_radius_triangle_753 :
  ∀ (a b c : ℝ), a = 7 ∧ b = 5 ∧ c = 3 → ∃ (R : ℝ), R = 7 * Real.sqrt 3 / 3 := 
by 
  intros a b c h,
  cases h with ha hb,
  cases hb with hb hc,
  use 7 * (Real.sqrt 3) / 3,
  sorry

end circumscribed_circle_radius_triangle_753_l160_160583


namespace sqrt_fourth_root_simplification_l160_160233

theorem sqrt_fourth_root_simplification :
  (∛(∛(32 / 10000))) = ∛(2^5 / 10^4) ∧
  (∛(2^5 / 10^4)) = (∛(2^2.5) / 100) ∧
  (∛(2^2.5 / 100)) = ∛(4 * ∛ 2 / 100) ∧
  (∛(4 * ∛ 2 / 100)) = ∧
  (∛ 2 / 25) = (∛(∛ 2) / ∛ 25)
  
  sorry

end sqrt_fourth_root_simplification_l160_160233


namespace paula_candies_distribution_l160_160431

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l160_160431


namespace four_digit_multiples_of_7_l160_160715

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160715


namespace coupon_probability_l160_160503

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l160_160503


namespace exists_shortest_chord_l160_160293

def shortest_chord_through_points {α : Type*} [metric_space α] (A B E F : α) (e : set α) (circle : α → α → Prop) (line : set α → Prop) :=
  line e →
  (A ∈ e ∨ B ∈ e → False) →
  A ∉ e → B ∉ e →
  ((circle A B) ∧ (E ∈ e) ∧ (F ∈ e) → (circle E F)) →
  ∀ EF, (E ∉ e → F ∉ e → (E, F ∈ circle A B) → 
    (dist E F) = 2 * (sqrt ((dist A E) * (dist B F)))).

theorem exists_shortest_chord {α : Type*} [metric_space α] (A B E F : α) (e : set α) (circle : α → α → Prop) (line : set α → Prop) (h1 : line e) (h2 : ¬(A ∈ e ∨ B ∈ e)) (h3 : A ∉ e) (h4 : B ∉ e) (h5 : (circle A B) ∧ (E ∈ e) ∧ (F ∈ e) → (circle E F)) :
  ∃ EF, E ∉ e → F ∉ e → (E, F ∈ circle A B) → 
    (dist E F) = 2 * (sqrt ((dist A E) * (dist B F))) :=
sorry

end exists_shortest_chord_l160_160293


namespace find_k_l160_160798

theorem find_k (k : ℤ) (x_0 : ℝ) (hk : x_0 ∈ (k : ℝ) + 0, (k + 1): ℝ))
  (hx0 : log x_0 = 5 - 2 * x_0) : k = 2 :=
sorry

end find_k_l160_160798


namespace projection_problem_l160_160415

variables {u z : Vector ℝ 2}

-- Define the known projection
def proj_u_on_z (u z : Vector ℝ 2) (hz : z ≠ 0) : Vector ℝ 2 :=
  let dot_product := (u ⬝ z) / (z ⬝ z)
  in dot_product • z

def given_condition : (proj_u_on_z u z ⬝ z) = 4 ∧ (proj_u_on_z u z ⬝ z) = 3

theorem projection_problem (h : given_condition) : 
  proj_u_on_z (3 • u + 4 • z) z = (12, 9) + 4 • z :=
sorry

end projection_problem_l160_160415


namespace _l160_160041

noncomputable def trapezoid (A B C D : Point) : Prop :=
  parallel (Segment AB) (Segment CD)

noncomputable def bisector_meet_at (A B C D : Point) (P Q R S : Point) (E F G H : Point): Prop :=
  meet (bisector (Angle A D C)) (bisector (Angle D A B)) = E ∧
  meet (bisector (Angle A B C)) (bisector (Angle B C D)) = F ∧
  meet (bisector (Angle B C D)) (bisector (Angle C D A)) = G ∧
  meet (bisector (Angle D A B)) (bisector (Angle A B C)) = H

noncomputable def area_quadrilateral (Q : Quadrilateral) : ℝ :=
  match Q with
  | ⟨E, A, B, F⟩ => 24
  | ⟨E, D, C, F⟩ => 36
  | _ => 0

noncomputable def area_triangle (T : Triangle) : ℝ :=
  match T with
  | ⟨A, B, H⟩ => 25
  | _ => 0

def main_theorem (A B C D E F G H: Point) : Prop :=
  (trapezoid A B C D) ∧
  (bisector_meet_at A B C D E F G H) ∧
  (area_quadrilateral ⟨E, A, B, F⟩ = 24) ∧
  (area_quadrilateral ⟨E, D, C, F⟩ = 36) ∧
  (area_triangle ⟨A, B, H⟩ = 25) →
  area_triangle ⟨C, D, G⟩ = real_root 7 256

example (A B C D E F G H M N : Point) : main_theorem A B C D E F G H :=
by sorry

end _l160_160041


namespace num_ways_4_people_7_steps_l160_160281

theorem num_ways_4_people_7_steps : 
  (∀ (steps : fin 7 → ℕ) (total_people : ℕ), 
   (∀ s, steps s ≤ 3) ∧ (finset.univ.sum steps = total_people) ∧ (total_people = 4)) →
   num_distinct_arrangements steps = 2394 :=
by
  sorry

end num_ways_4_people_7_steps_l160_160281


namespace pow_inequality_l160_160657

theorem pow_inequality (a b : ℝ) (h1: a < b) (h2: b < 0) : 2^a < 2^b :=
by
  sorry

end pow_inequality_l160_160657


namespace num_four_digit_multiples_of_7_l160_160719

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160719


namespace download_speed_scientific_notation_l160_160131

-- Define the theoretical download speed
def theoretical_download_speed : ℕ := 603000000

-- The scientific notation function
def scientific_notation (n : ℕ) : string :=
  let significant_digits := 6.03
  let exponent := 8
  in to_string significant_digits ++ "e" ++ to_string exponent

-- Prove 603,000,000 in scientific notation is 6.03 × 10^8
theorem download_speed_scientific_notation :
  scientific_notation theoretical_download_speed = "6.03e8" :=
sorry

end download_speed_scientific_notation_l160_160131


namespace optimal_days_for_debit_card_l160_160455

theorem optimal_days_for_debit_card:
  let ticket_cost: ℝ := 12000
  let cashback_credit: ℝ := 0.01 * ticket_cost
  let cashback_debit: ℝ := 0.02 * ticket_cost
  let annual_interest_rate: ℝ := 0.06
  let daily_interest_rate: ℝ := annual_interest_rate / 365
  let benefit_credit := λ N: ℝ, N * (ticket_cost * daily_interest_rate) + cashback_credit
  let benefit_debit := cashback_debit
  ∀ N: ℝ, benefit_debit ≥ benefit_credit N → N ≤ 6 := 
by {
  sorry
}

end optimal_days_for_debit_card_l160_160455


namespace count_four_digit_multiples_of_7_l160_160737

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160737


namespace sin_pi_div_2_pow_2011_exact_value_l160_160536

theorem sin_pi_div_2_pow_2011_exact_value :
  sin (π / 2 ^ 2011) = (sqrt (2 - sqrt (2 + ... + sqrt(2)))) / 2 := 
sorry

end sin_pi_div_2_pow_2011_exact_value_l160_160536


namespace tangent_line_equation_at_1_range_of_a_l160_160669

noncomputable def f (x a : ℝ) : ℝ := (x+1) * Real.log x - a * (x-1)

-- (I) Tangent line equation when a = 4
theorem tangent_line_equation_at_1 (x : ℝ) (hx : x = 1) :
  let a := 4
  2*x + f 1 a - 2 = 0 :=
sorry

-- (II) Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_equation_at_1_range_of_a_l160_160669


namespace chord_intersection_square_sum_l160_160376

theorem chord_intersection_square_sum 
  (O : Type) [metric_space O] [semimodule ℝ O]
  (R : ℝ) (CD AB : set O) (P : O)
  (rC : dist O P = R) (rCD : is_chord CD O) (rAB : is_diameter AB O)
  (h_intersect : P ∈ (CD ∩ AB : set O))
  (h_angle : ∠ (line_through P C).direction (line_through P D).direction = π / 4) :
  dist P C ^ 2 + dist P D ^ 2 = 2 * R ^ 2 :=
by
  sorry

end chord_intersection_square_sum_l160_160376


namespace smallest_integer_in_set_l160_160373

theorem smallest_integer_in_set :
  ∃ n : ℤ, n + 6 < 3 * (n + 3) ∧ ∀ m : ℤ, m + 6 < 3 * (m + 3) → n ≤ m := 
begin
  sorry
end

end smallest_integer_in_set_l160_160373


namespace find_sugar_amount_l160_160814

-- Define the variables S, F, B
variables (S F B : ℝ)

-- Define the conditions
def condition1 : Prop := S = F
def condition2 : Prop := F = 10 * B
def condition3 : Prop := F / (B + 60) = 8

-- Define the theorem
theorem find_sugar_amount (h1 : condition1 S F B) (h2 : condition2 S F B) (h3 : condition3 S F B) : S = 2400 := 
by
  sorry

end find_sugar_amount_l160_160814


namespace four_digit_multiples_of_7_l160_160750

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160750


namespace curvilinear_triangle_area_proof_l160_160877

noncomputable def curvilinear_triangle_area (r_sphere : ℝ) (r_circle : ℝ) (num_circles : ℕ) : ℝ :=
  if r_sphere = 2 ∧ r_circle = sqrt 2 ∧ num_circles = 3 then π * (3 * sqrt 2 - 4) else 0

theorem curvilinear_triangle_area_proof :
  curvilinear_triangle_area 2 (sqrt 2) 3 = π * (3 * sqrt 2 - 4) :=
by
  sorry

end curvilinear_triangle_area_proof_l160_160877


namespace right_triangle_angles_l160_160804

def right_triangle_angle_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 2) : Prop :=
  α = 54 ∧ β = 36

theorem right_triangle_angles :
  ∃ α β : ℝ, right_triangle_angle_ratio α β (by norm_num) (by norm_num) :=
sorry

end right_triangle_angles_l160_160804


namespace loaned_books_returned_percentage_l160_160572

theorem loaned_books_returned_percentage
  (b_0 b_f l remaining_loaned_books : ℕ)
  (h_initial : b_0 = 150)
  (h_final : b_f = 122)
  (h_loaned : l = 80)
  (h_remaining : remaining_loaned_books = b_0 - b_f) :
  let P := 100 - (remaining_loaned_books * 100 / l) in
  P = 65 := 
by
  sorry

end loaned_books_returned_percentage_l160_160572


namespace bisector_intersects_at_midpoint_of_arc_l160_160176

variable {Circle : Type} [MetricSpace Circle] [Circumference Circle]
variables {O A B C D : Point Circle}
variables {AB CD : Line Circle}

axiom is_diameter (AB: Line Circle) (O: Point Circle) [Midpoint O AB] : AB.is_diameter
axiom on_circle (C : Point Circle) : C ∈ Circle
axiom perpendicular_chord (CD: Line Circle) (AB: Line Circle) : ⊥ CD AB

theorem bisector_intersects_at_midpoint_of_arc :
  ∀ (C : Point Circle) (hC: C ∈ Circle), 
  let D := perpendicular_from C to AB,
  let bisector_of_OCD := bisector_angle O C D in
  let P := point_of_intersection bisector_of_OCD Circle in
  P.bisects_arc AB :=
begin
  sorry
end

end bisector_intersects_at_midpoint_of_arc_l160_160176


namespace standard_price_of_pizza_l160_160885

theorem standard_price_of_pizza :
  let price_triple (P : ℝ) := (10 / (1 + 1)) * P
  let price_meat (P : ℝ) := (9 * 2 / (2 + 1)) * P
  let total_cost := price_triple 5 + price_meat 5
  total_cost = 55 →
  ∃ P : ℝ, 5 = P :=
by
  intros h
  use 5
  sorry

end standard_price_of_pizza_l160_160885


namespace max_value_frac_norm_x_norm_b_l160_160032

variables {𝕜 : Type*} [IsROrC 𝕜] {V : Type*} [InnerProductSpace 𝕜 V]

def unit_vector (v : V) := ∥v∥ = 1

theorem max_value_frac_norm_x_norm_b
  (e1 e2 : V)
  (x y : 𝕜)
  (h1 : unit_vector e1)
  (h2 : unit_vector e2)
  (h3 : ∃ θ : ℝ, (θ = Real.pi / 6) ∧ ⟪e1, e2⟫ = (IsROrC.ofReal (Real.cos θ)))
  (h4 : x ≠ 0 ∨ y ≠ 0) :
  (∃ b : V, b = x • e1 + y • e2 ∧ ∥x∥ / ∥b∥ ≤ 2) :=
by sorry

end max_value_frac_norm_x_norm_b_l160_160032


namespace intersection_A_B_l160_160027

def set_A : Set ℝ := {x | 1 < 2^x ∧ 2^x < 8}
def set_B : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 - 2*x + 8)}

theorem intersection_A_B :
  { x | x ∈ set_A ∧ x ∈ set_B } = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_A_B_l160_160027


namespace integral_bounds_l160_160022

theorem integral_bounds (P Q : ℝ → ℝ)
  (h1 : ∫ x in 0..2, (P x) ^ 2 = 14)
  (h2 : ∫ x in 0..2, P x = 4)
  (h3 : ∫ x in 0..2, (Q x) ^ 2 = 26)
  (h4 : ∫ x in 0..2, Q x = 2) :
  -8 ≤ ∫ x in 0..2, P x * Q x ∧ ∫ x in 0..2, P x * Q x ≤ 16 :=
by sorry

end integral_bounds_l160_160022


namespace smallest_collinear_distance_equilateral_triangle_l160_160271

theorem smallest_collinear_distance_equilateral_triangle (Δ : Triangle) (P : Point)
  (h₀ : Δ.is_equilateral ∧ Δ.side_length = 1 ∧ P ∈ Δ.interior) :
  ∃ (x : ℝ), ∀ (A B C D E F : Point), A B C D E F ∈ Δ.sides →
  collinear [P, A, B, C, D, E, F] → distance(A, B) ≥ x ∧ distance(C, D) ≥ x ∧ distance(E, F) ≥ x
  := sorry

end smallest_collinear_distance_equilateral_triangle_l160_160271


namespace problem_equivalence_l160_160451

noncomputable def a : ℂ := 2
noncomputable def b : ℂ := complex.I
noncomputable def c : ℂ := -complex.I

theorem problem_equivalence (h1 : a + b + c = 2) (h2 : a * b + a * c + b * c = 3) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = 8 := sorry

end problem_equivalence_l160_160451


namespace geometric_sequence_common_ratio_l160_160031

theorem geometric_sequence_common_ratio
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : S 1 = a 1)
  (h2 : S 2 = a 1 + a 1 * q)
  (h3 : a 2 = a 1 * q)
  (h4 : a 3 = a 1 * q^2)
  (h5 : 3 * S 2 = a 3 - 2)
  (h6 : 3 * S 1 = a 2 - 2) :
  q = 4 :=
sorry

end geometric_sequence_common_ratio_l160_160031


namespace range_of_x_l160_160310

theorem range_of_x (f : ℝ → ℝ) (F : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x) -- f is odd
  (h2 : ∀ x, (x ≠ 0 → ∃ f' : ℝ → ℝ, deriv f x = f' x)) -- derivative exists
  (h3 : ∀ x, x ∈ set.Iic 0 → (x * (deriv f x) < f (-x))) -- xf'(x) < f(-x) for x ≤ 0
  (h4 : ∀ x, F x = x * f x) : set.Ioo (-1 : ℝ) 2 :=
begin
  sorry
end

end range_of_x_l160_160310


namespace meaningful_expression_range_l160_160506

theorem meaningful_expression_range (x : ℝ) :
  (1 ≤ x) ∧ (x ≠ 2) ↔ (sqrt (x - 1) + 1 / (x - 2) = sqrt (x - 1) + 1 / (x - 2)) :=
by
  sorry

end meaningful_expression_range_l160_160506


namespace distance_A_B_l160_160384

-- Define the coordinates of points A and B and the distance formula
def A : ℝ × ℝ × ℝ := (2, 3, 5)
def B : ℝ × ℝ × ℝ := (3, 1, 4)

-- Define the distance formula between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Prove the distance between A and B is sqrt(6)
theorem distance_A_B : distance A B = real.sqrt 6 := by
  sorry

end distance_A_B_l160_160384


namespace measure_of_angle_A_range_of_b2_add_c2_div_a2_l160_160014

variable {A B C a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) : 
  A = 2 * Real.pi / 3 :=
by
  sorry

theorem range_of_b2_add_c2_div_a2
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : A = 2 * Real.pi / 3) : 
  2 / 3 ≤ (b ^ 2 + c ^ 2) / a ^ 2 ∧ (b ^ 2 + c ^ 2) / a ^ 2 < 1 :=
by
  sorry

end measure_of_angle_A_range_of_b2_add_c2_div_a2_l160_160014


namespace number_of_four_digit_multiples_of_7_l160_160695

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160695


namespace geometric_sequence_arithmetic_condition_l160_160315

noncomputable def geometric_sequence_ratio (q : ℝ) : Prop :=
  q > 0

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  2 * a₃ = a₁ + 2 * a₂

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * q ^ n

theorem geometric_sequence_arithmetic_condition
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (q : ℝ)
  (hq : geometric_sequence_ratio q)
  (h_arith : arithmetic_sequence (a 0) (geometric_sequence a q 1) (geometric_sequence a q 2)) :
  (geometric_sequence a q 9 + geometric_sequence a q 10) / 
  (geometric_sequence a q 7 + geometric_sequence a q 8) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_arithmetic_condition_l160_160315


namespace ab_leq_one_l160_160351

theorem ab_leq_one (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

end ab_leq_one_l160_160351


namespace polynomial_evaluation_l160_160285

theorem polynomial_evaluation (a : ℝ) (h : a^2 + 3 * a = 2) : 2 * a^2 + 6 * a - 10 = -6 := by
  sorry

end polynomial_evaluation_l160_160285


namespace combination_equality_l160_160304

theorem combination_equality (x : ℕ) (h: (nat.choose 9 x) = (nat.choose 9 (2 * x - 3))) : x = 3 ∨ x = 4 :=
sorry

end combination_equality_l160_160304


namespace four_digit_multiples_of_7_l160_160747

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160747


namespace total_volume_polyhedron_l160_160834

-- Definitions based on conditions
def S_midpoint_AD (S : Point) : Prop := midpoint S A D
def S1_midpoint_B1C1 (S1 : Point) : Prop := midpoint S1 B1 C1

def rotated_cube (cube : Cube) : Prop := rotated cube

def common_polyhedron (prism : Polyhedron) (pyramid1 pyramid2 : Polyhedron) : Prop :=
  prism.is_regular_quadrilateral_prism ∧
  pyramid1.is_regular_quadrilateral_pyramid ∧
  pyramid2.is_regular_quadrilateral_pyramid

def side_length_base_pyramid : ℝ := 1
def height_pyramid : ℝ := 1 / 2
def volume_prism : ℝ := sqrt 2 - 1

-- Assuming we have the necessary definitions for volume calculations based on these
def volume_pyramid (base_length height : ℝ) : ℝ := (1/3) * base_length * base_length * height
def total_volume (prism_volume : ℝ) (pyramid_volumes : List ℝ) : ℝ := 
  prism_volume + list.sum pyramid_volumes

-- The final theorem statement
theorem total_volume_polyhedron :
  ∀ (S S1 : Point) (rot_cube : Cube) (prism pyramid1 pyramid2 : Polyhedron),
  S_midpoint_AD S →
  S1_midpoint_B1C1 S1 →
  rotated_cube rot_cube →
  common_polyhedron prism pyramid1 pyramid2 →
  volume_pyramid side_length_base_pyramid height_pyramid = 1 / 6 →
  volume_prism = sqrt 2 - 1 →
  total_volume (volume_prism) [1 / 6, 1 / 6] = sqrt 2 - (2 / 3) :=
by
  sorry

end total_volume_polyhedron_l160_160834


namespace triangular_prism_pyramids_count_l160_160690

theorem triangular_prism_pyramids_count : 
  let vertices := 6 in
  let selected_vertices := 4 in
  let invalid_cases := 2 in
  let unique_pyramids := 12 in
  (nat.choose vertices selected_vertices) - invalid_cases = unique_pyramids :=
by
  sorry

end triangular_prism_pyramids_count_l160_160690


namespace midpoint_translation_example_l160_160933

def Point := (ℝ × ℝ)

def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_translation_example :
  let A := (1, 2)
  let L := (7, 2)
  let A' := translate A 3 4
  let L' := translate L 3 4
  midpoint A' L' = (7, 6) :=
by {
  sorry
}

end midpoint_translation_example_l160_160933


namespace total_volume_needed_l160_160156

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 12
def box_cost : ℕ := 50 -- in cents to avoid using floats
def total_spent : ℕ := 20000 -- $200 in cents

def volume_of_box : ℕ := box_length * box_width * box_height
def number_of_boxes : ℕ := total_spent / box_cost

theorem total_volume_needed : number_of_boxes * volume_of_box = 1920000 := by
  sorry

end total_volume_needed_l160_160156


namespace sum_of_special_primes_l160_160272

open Int Nat

theorem sum_of_special_primes :
  (∑ p in (list.range 100).filter (λ n, prime n ∧ (n % 4 = 1) ∧ (n % 5 = 4)), id p) = 139 := by
  sorry

end sum_of_special_primes_l160_160272


namespace count_four_digit_multiples_of_7_l160_160763

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l160_160763


namespace water_remaining_after_3_pourings_l160_160067

-- Define the sequence of remaining fraction after each pouring.
def remaining_fraction (n : ℕ) : ℚ :=
  (∏ k in Finset.range n, ((2 * k + 1) : ℚ) / ((2 * k + 2) : ℚ)) / (2 ^ n)

-- Define the problem statement
theorem water_remaining_after_3_pourings : 
  remaining_fraction 3 = 1 / 8 :=
by
  sorry

end water_remaining_after_3_pourings_l160_160067


namespace cost_price_of_watch_l160_160983

variable (CP : ℕ) 
variable (SP : ℕ)
variable (profit_percentage : ℕ)

-- Conditions
noncomputable def profit := SP - CP
def condition1 : profit_percentage = CP := sorry
def condition2 : CP = 144 := sorry

theorem cost_price_of_watch : CP = 144 :=
by
  have h1 : profit_percentage = CP := condition1
  have h2 : CP = 144 := condition2
  assumption

end cost_price_of_watch_l160_160983


namespace terry_problems_wrong_l160_160093

theorem terry_problems_wrong (R W : ℕ) 
  (h1 : R + W = 25) 
  (h2 : 4 * R - W = 85) : 
  W = 3 := 
by
  sorry

end terry_problems_wrong_l160_160093


namespace sum_sequence_eq_l160_160245

def A : ℕ → ℚ
| 0     := 2
| 1     := 3
| (n+2) := 2 * A (n+1) + A n

theorem sum_sequence_eq :
  (∑' n, (A n) / (5 ^ n)) = 63 / 19 :=
  sorry

end sum_sequence_eq_l160_160245


namespace value_of_b_l160_160033

noncomputable def C (n k : ℕ) : ℕ := nat.choose n k

theorem value_of_b (a : ℕ) (b : ℕ) (h1 : a = ∑ i in finset.range 1 2011, (C 2010 i) * 3^(2*i)) (h2 : b ≡ a [MOD 10]) :
  b = 2009 :=
sorry

end value_of_b_l160_160033


namespace distinct_integers_sum_to_32_l160_160036

theorem distinct_integers_sum_to_32 
  (p q r s t : ℤ)
  (h_diff : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_eq : (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120) : 
  p + q + r + s + t = 32 := 
by 
  sorry

end distinct_integers_sum_to_32_l160_160036


namespace map_distance_scaled_l160_160055

theorem map_distance_scaled (d_map : ℝ) (scale : ℝ) (d_actual : ℝ) :
  d_map = 8 ∧ scale = 1000000 → d_actual = 80 :=
by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end map_distance_scaled_l160_160055


namespace num_four_digit_multiples_of_7_l160_160726

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160726


namespace collinear_ABD_find_k_l160_160048

-- Definitions
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (a b : V) (A B C D : V)

-- Hypotheses
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom not_collinear : ¬ collinear ℝ ({a, b} : set V)

-- Problem 1
axiom AB : B - A = a + b
axiom BC : C - B = 2 • a + 8 • b
axiom CD : D - C = 3 • (a - b)

-- Proof statement for Problem 1
theorem collinear_ABD :
  collinear ℝ ({A, B, D} : set V) := sorry

-- Problem 2
theorem find_k (k : ℝ) :
  collinear ℝ ({k • a + b, a + k • b} : set V) ↔ k = 1 ∨ k = -1 := sorry

end collinear_ABD_find_k_l160_160048


namespace trajectory_line_segment_l160_160654

noncomputable theory

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

def P (x y : ℝ) := (x, y)

def F1 := (-2 : ℝ, 0 : ℝ)
def F2 := (2 : ℝ, 0 : ℝ)

theorem trajectory_line_segment (x y : ℝ) :
  distance (P x y) F1 + distance (P x y) F2 = 4 ↔ 
  ∃ t : ℝ, (P x y) = ((1 - t) * F1.1 + t * F2.1, (1 - t) * F1.2 + t * F2.2) :=
sorry

end trajectory_line_segment_l160_160654


namespace contrapositive_equivalence_l160_160533
-- Importing the necessary libraries

-- Declaring the variables P and Q as propositions
variables (P Q : Prop)

-- The statement that we need to prove
theorem contrapositive_equivalence :
  (P → ¬ Q) ↔ (Q → ¬ P) :=
sorry

end contrapositive_equivalence_l160_160533


namespace log_div_log_eq_neg1_l160_160964

noncomputable def log_div_log {a b : ℝ} (ha : a > 0) (hb : b > 0) (h4: 4 > 0) (hlog1 : Real.log 64 / Real.log 4 = 3) (hlog2 : Real.log (1 / 64) / Real.log 4 = -3) : ℝ :=
  (Real.log 64 / Real.log 4) / (Real.log (1 / 64) / Real.log 4)

theorem log_div_log_eq_neg1 : log_div_log (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) = -1 := 
sorry

end log_div_log_eq_neg1_l160_160964


namespace geometric_special_points_relation_l160_160847

-- Define the necessary points of the triangle ΔABC and the circumradius R
variables {O G K S1 S2 : Type} [metric_space O] [metric_space G] [metric_space K] [metric_space S1] [metric_space S2]
variable (R : ℝ)

-- Define the known conditions
/-!
O is the circumcenter of ΔABC
G is the centroid of ΔABC
K is the symmedian point of ΔABC
S1 is the first isodynamic point of ΔABC
S2 is the second isodynamic point of ΔABC
R is the circumradius of ΔABC
-/

-- Define the distances between points using some distance function (dist : Type -> Type -> ℝ)
variables (dist : ∀ {X : Type} [metric_space X], X → X → ℝ)

-- Statement of the theorem to be proven
theorem geometric_special_points_relation
  (hOS1 : O = circumcenter_of_triangle ABC)
  (hGS1 : G = centroid_of_triangle ABC)
  (hK : K = symmedian_point_of_triangle ABC)
  (hS1 : S1 = first_isodynamic_point_of_triangle ABC)
  (hS2 : S2 = second_isodynamic_point_of_triangle ABC) :
  dist G S1 / dist G S2 = (R / dist O K - real.sqrt (R ^ 2 / (dist O K) ^ 2 - 1)) ^ 3 :=
sorry

end geometric_special_points_relation_l160_160847


namespace trajectory_is_ellipse_l160_160045

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.fst^2 + N.snd^2 = 8 ∧ 
                 ∃ (M : ℝ × ℝ), M.fst = 0 ∧ M.snd = N.snd ∧
                 P.fst = N.fst / 2 ∧ P.snd = N.snd

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_of_P P) : 
  P.fst^2 / 2 + P.snd^2 / 8 = 1 :=
by
  sorry

end trajectory_is_ellipse_l160_160045


namespace complement_fraction_irreducible_l160_160541

theorem complement_fraction_irreducible (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.gcd (b - a) b = 1 :=
sorry

end complement_fraction_irreducible_l160_160541


namespace derivative_zero_extremum_condition_l160_160104

noncomputable def has_extremum (f : ℝ → ℝ) (x : ℝ) :=
  ∃ δ > 0, ∀ y, abs (y - x) < δ → f x ≤ f y ∨ f x ≥ f y

theorem derivative_zero_extremum_condition (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, has_extremum f x → diffable_at f x) ∧
  (∀ x, diffable_at f x → f' x = 0 → has_extremum f x) → False :=
sorry

end derivative_zero_extremum_condition_l160_160104


namespace white_balls_in_bag_l160_160563

theorem white_balls_in_bag:
  ∀ (total balls green yellow red purple : Nat),
  total = 60 →
  green = 18 →
  yellow = 8 →
  red = 5 →
  purple = 7 →
  (1 - 0.8) = (red + purple : ℚ) / total →
  (W + green + yellow = total - (red + purple : ℚ)) →
  W = 22 :=
by
  intros total balls green yellow red purple ht hg hy hr hp hprob heqn
  sorry

end white_balls_in_bag_l160_160563


namespace find_a_10_l160_160624

def a : ℕ → ℕ

-- Given conditions
axiom circle_size : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 15
axiom number_sharing : ∀ (k : ℕ), (a (k - 1) + a (k + 1)) / 2 = k

-- Equations derived from conditions
axiom eq1 : a 9 + a 11 = 20
axiom eq2 : a 10 + a 12 = 22
axiom eq3 : a 11 + a 13 = 24
axiom eq4 : a 8 + a 10 = 18

-- Required to prove
theorem find_a_10 : a 10 = 10 :=
by sorry

end find_a_10_l160_160624


namespace product_of_possible_values_of_x_l160_160787

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l160_160787


namespace count_g_to_2_200_l160_160398

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else n / 2 + 1

def count_g_to_2 (n_range : Finset ℕ) : ℕ :=
  n_range.filter (λ n => ∃ k, g^[k] n = 2).card

theorem count_g_to_2_200 :
  count_g_to_2 (Finset.range 200) = 2 := by
  sorry

end count_g_to_2_200_l160_160398


namespace tetrahedron_circumsphere_radius_proof_l160_160002

noncomputable def tetrahedron_circumsphere_radius (A B C D : ℝ) (angle_ADB angle_BDC angle_CDA : ℝ) (AD BD CD : ℝ) [h1 : angle_ADB = π / 3] [h2 : angle_BDC = π / 3] [h3 : angle_CDA = π / 3] [h4 : AD = 3] [h5 : BD = 3] [h6 : CD = 2] :
  ℝ :=
if h : A = B ∧ B = C ∧ C = D ∧ D = A then
  0  -- By definition, if all points are the same, radius is zero
else
  ∃ R, R^2 = 3 →
    ∀ R', R' = real.sqrt 3

theorem tetrahedron_circumsphere_radius_proof {A B C D : ℝ} {angle_ADB angle_BDC angle_CDA : ℝ} {AD BD CD : ℝ} :
  angle_ADB = π / 3 →
  angle_BDC = π / 3 →
  angle_CDA = π / 3 →
  AD = 3 →
  BD = 3 →
  CD = 2 →
  tetrahedron_circumsphere_radius A B C D angle_ADB angle_BDC angle_CDA AD BD CD = real.sqrt 3 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  dsimp [tetrahedron_circumsphere_radius],
  split_ifs,
  { sorry, }, -- This is where the proof would go
  { sorry, }  -- This is where the proof would go
end

end tetrahedron_circumsphere_radius_proof_l160_160002


namespace chessboard_problem_l160_160092

theorem chessboard_problem (n : ℕ) (k : ℕ) :
  (n = 4 * k) →
  (∀ (i: ℕ) (h_i : i < n), ∃ (f : ℕ → ℕ), 
    (∀ j, j < n → (1 ≤ f j ∧ f j ≤ n) ∧ 
      ∑ (j : ℕ) in (finset.range((n / 2))), 
      if ((i + j) % 2 = 0) then f (2 * j) else f (2 * j + 1) = 
      ∑ (j : ℕ) in (finset.range((n / 2))), 
      if ((i + j) % 2 = 1) then f (2 * j) else f (2 * j + 1)) ∧
      ∀ j, j < n → 
        (∑(i : ℕ) in (finset.range((n / 2))), 
        if ((i + j) % 2 = 0) then f (2 * i) else f (2 * i + 1) = 
        ∑(i : ℕ) in (finset.range((n / 2))), 
        if ((i + j) % 2 = 1) then f (2 * i) else f (2 * i + 1)
      )
  ) := 
sorry

end chessboard_problem_l160_160092


namespace b_finishes_in_15_days_l160_160977

theorem b_finishes_in_15_days (a_rate b_rate : ℝ) (a_days : ℕ) (total_work : ℝ) :
  a_rate = 1 / 20 → b_rate = 1 / 30 → a_days = 10 → total_work = 1 →
  let a_work_done := a_rate * a_days in
  let remaining_work := total_work - a_work_done in
  let b_time_required := remaining_work / b_rate in
  b_time_required = 15 := 
by {
  sorry
}

end b_finishes_in_15_days_l160_160977


namespace smallest_number_divisible_l160_160151

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Using Nat.lcm from Mathlib

theorem smallest_number_divisible (N x : ℕ) (h1 : N - x = 746)
  (h2 : 746 % lcm (lcm 8 14) (lcm 26 28) = 0) :
  N = 1474 ∧ x = 728 :=
by
  sorry

end smallest_number_divisible_l160_160151


namespace inverse_proportional_l160_160452

/-- Given that α is inversely proportional to β and α = -3 when β = -6,
    prove that α = 9/4 when β = 8. --/
theorem inverse_proportional (α β : ℚ) 
  (h1 : α * β = 18)
  (h2 : β = 8) : 
  α = 9 / 4 :=
by
  sorry

end inverse_proportional_l160_160452


namespace sum_of_distinct_products_l160_160119

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10) :
  (3 * H + 8) % 8 = 0 ∧ ((6 + 2 + 8 + G + 4 + 0 + 9 + 3 + H + 8) % 9 = 0) →
  (G * H = 6 ∨ G * H = 48) →
  6 + 48 = 54 :=
by
  intros _ _
  sorry

end sum_of_distinct_products_l160_160119


namespace limit_problem_l160_160269

noncomputable def problem_statement : ℝ :=
  lim (λ x : ℝ, (sin (3 * x) + 2^(8 * x - 3) * cos (Real.pi * x / 3)) / x) 0

theorem limit_problem :
  problem_statement = 3 + Real.log 2 :=
sorry

end limit_problem_l160_160269


namespace intersection_points_eq_two_l160_160239

theorem intersection_points_eq_two
  (param_eqn_C1 : ∀ (a : ℝ), (cos a, 1 + sin a) ∈ set_of (λ (p : ℝ × ℝ), True))
  (polar_eqn_C2 : ∀ (p θ : ℝ), p * (cos θ - sin θ) + 1 = 0 → (p * cos θ, p * sin θ) ∈ set_of (λ (p : ℝ × ℝ), True)):
  ∃ (x y : ℝ), (x = cos) x ∧ (y = 1 + sin) y = 2 := sorry

end intersection_points_eq_two_l160_160239


namespace is_linear_eqD_l160_160972

-- Define the conditions as equations
def eqA (x y : ℝ) : Prop := x * y = 2
def eqB (x y : ℝ) : Prop := (1 / x) + y = 3
def eqC (x y : ℝ) : Prop := 3 * x + y^2 = 1
def eqD (x y : ℝ) : Prop := 2 * x + y = 5

-- Prove that eqD is a linear equation in two variables
theorem is_linear_eqD : ∀ x y : ℝ, eqD x y -> ∀ a b c : ℝ, 2 * x + y = c ↔ a = 2 ∧ b = 1 :=
by
  intros x y h a b c
  split
  sorry

end is_linear_eqD_l160_160972


namespace total_gifts_l160_160059

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l160_160059


namespace geometric_sequence_b_value_l160_160478

theorem geometric_sequence_b_value (b : ℝ) (hb : 0 < b) :
  (∃ r : ℝ, 10 * r = b ∧ b * r = 3 / 4) →
  b = 5 * real.sqrt 3 / 3 :=
by
  intro h
  sorry 

end geometric_sequence_b_value_l160_160478


namespace andrey_gifts_l160_160061

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l160_160061


namespace simplest_root_l160_160974

theorem simplest_root (A B C D : ℝ) (hA : A = Real.sqrt 14)
  (hB : B = Real.sqrt 12) (hC : C = Real.sqrt 8) (hD : D = Real.sqrt (1/3)) :
  A < B ∧ A < C ∧ A < D :=
by
  have hA_simplest: A = Real.sqrt 14 := hA
  have hB_not_simplest: B = 2 * Real.sqrt 3 := by
    rw [hB]
    rw Real.sqrt_mul (4 : ℝ) (3 : ℝ)
    rw Real.sqrt_eq
    ring
  have hC_not_simplest: C = 2 * Real.sqrt 2 := by
    rw [hC]
    rw Real.sqrt_mul (4 : ℝ) (2 : ℝ)
    rw Real.sqrt_eq
    ring
  have hD_not_simplest: D = Real.sqrt 3 / 3 := by
    rw [hD]
    rw Real.sqrt_div (1 : ℝ) (3 : ℝ)
    rw Real.sqrt_eq
    ring
  rw [←hA_simplest, ←hB_not_simplest, ←hC_not_simplest, ←hD_not_simplest]
  exact ⟨lt_trans (by norm_num) (by norm_num), lt_trans (by norm_num) (by norm_num), lt_trans (by norm_num) (by norm_num)⟩

end simplest_root_l160_160974


namespace find_starting_crayons_l160_160390

-- Conditions
variables (end_crayons : ℕ) (eaten_crayons : ℕ)
-- Fixing the given conditions from the problem
def jane_ends_with_80_crayons : Prop := end_crayons = 80
def 7_crayons_eaten_by_hippopotamus : Prop := eaten_crayons = 7

-- Theorem statement
theorem find_starting_crayons (end_crayons eaten_crayons : ℕ) :
  jane_ends_with_80_crayons end_crayons →
  7_crayons_eaten_by_hippopotamus eaten_crayons →
  end_crayons + eaten_crayons = 87 :=
by
  sorry

end find_starting_crayons_l160_160390


namespace min_value_a_plus_2b_l160_160357

theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 20) : a + 2 * b = 4 * Real.sqrt 10 :=
by
  sorry

end min_value_a_plus_2b_l160_160357


namespace evin_loses_count_l160_160623

theorem evin_loses_count : (Finset.filter (λ n : ℕ, n ≤ 1434 ∧ (n % 5 = 0 ∨ n % 5 = 1)) (Finset.range 1435)).card = 573 :=
sorry

end evin_loses_count_l160_160623


namespace total_birds_in_marsh_l160_160009

-- Given conditions
def initial_geese := 58
def doubled_geese := initial_geese * 2
def ducks := 37
def swans := 15
def herons := 22

-- Prove that the total number of birds is 190
theorem total_birds_in_marsh : 
  doubled_geese + ducks + swans + herons = 190 := 
by
  sorry

end total_birds_in_marsh_l160_160009


namespace value_of_a_minus_b_l160_160484

theorem value_of_a_minus_b (a b : ℝ)
  (h1 : ∃ (x : ℝ), x = 3 ∧ (ax / (x - 1)) = 1)
  (h2 : ∀ (x : ℝ), (ax / (x - 1)) < 1 ↔ (x < b ∨ x > 3)) :
  a - b = -1 / 3 :=
by
  sorry

end value_of_a_minus_b_l160_160484


namespace opposite_of_two_thirds_l160_160477

theorem opposite_of_two_thirds : - (2/3) = -2/3 :=
by
  sorry

end opposite_of_two_thirds_l160_160477


namespace min_quadratic_expression_l160_160963

theorem min_quadratic_expression:
  ∀ x : ℝ, x = 3 → (x^2 - 6 * x + 5 = -4) :=
by
  sorry

end min_quadratic_expression_l160_160963


namespace find_c_l160_160910

theorem find_c (a b c : ℝ) (h1 : ∃ a, ∃ b, ∃ c, 
              ∀ y, (∀ x, (x = a * (y-1)^2 + 4) ↔ (x = -2 → y = 3)) ∧
              (∀ y, x = a * y^2 + b * y + c)) : c = 1 / 2 :=
sorry

end find_c_l160_160910


namespace tyler_cd_purchase_l160_160513

theorem tyler_cd_purchase :
  ∀ (initial_cds : ℕ) (given_away_fraction : ℝ) (final_cds : ℕ) (bought_cds : ℕ),
    initial_cds = 21 →
    given_away_fraction = 1 / 3 →
    final_cds = 22 →
    bought_cds = 8 →
    final_cds = initial_cds - initial_cds * given_away_fraction + bought_cds :=
by
  intros
  sorry

end tyler_cd_purchase_l160_160513


namespace solution_greater_iff_l160_160034

variables {c c' d d' : ℝ}
variables (hc : c ≠ 0) (hc' : c' ≠ 0)

theorem solution_greater_iff : (∃ x, x = -d / c) > (∃ x, x = -d' / c') ↔ (d' / c') < (d / c) :=
by sorry

end solution_greater_iff_l160_160034


namespace Roja_speed_is_8_l160_160443

def Pooja_speed : ℝ := 3
def time_in_hours : ℝ := 4
def distance_between_them : ℝ := 44

theorem Roja_speed_is_8 :
  ∃ R : ℝ, R + Pooja_speed = (distance_between_them / time_in_hours) ∧ R = 8 :=
by
  sorry

end Roja_speed_is_8_l160_160443


namespace min_value_PA_plus_PQ_l160_160318

open Real

noncomputable def point_P (y : ℝ) : ℝ × ℝ :=
  ((y^2) / 4, y)

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

def minimum_PA_plus_PQ : ℝ :=
  Type*

theorem min_value_PA_plus_PQ :
  minimum_PA_plus_PQ  = 2 := sorry

end min_value_PA_plus_PQ_l160_160318


namespace correct_statements_l160_160558

open Function

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f at any point
def f'_at (x : ℝ) : ℝ := 3 * x^2

-- Major premise: For a differentiable function f(x), if f'(x_0) = 0, then x = x_0 is an extremum point of f(x)
def major_premise (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ (x : ℝ), Differentiable ℝ f → (f' x0 = 0 → ∃ ε > 0, ∀ δ > 0, (|x - x0| < δ → f(x) ≥ f(x0) ∨ f(x) ≤ f(x0)))

-- Minor premise: The derivative of the function f(x) = x^3 at x = 0 is f'(0) = 0
def minor_premise : Prop := f'_at 0 = 0

-- Form of reasoning: The form of reasoning is correct (modus ponens)
def form_of_reasoning (p q : Prop) : Prop := (p → q) ∧ p → q

-- The final proof problem
theorem correct_statements : 
  (¬major_premise f f'_at 0) ∧ (form_of_reasoning (major_premise f f'_at 0) minor_premise) := 
by
  sorry

end correct_statements_l160_160558


namespace number_of_four_digit_multiples_of_7_l160_160692

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l160_160692


namespace sin_sum_identity_l160_160612

open Real

theorem sin_sum_identity :
  (∑ x in Finset.range 181, (sin (x * (π / 180))) ^ 6) =
    724 / 8 := by
    sorry

end sin_sum_identity_l160_160612


namespace boundary_of_shadow_eq_minus_2_l160_160207

noncomputable def sphere_center := (0, 0, 2)
noncomputable def sphere_radius := 2
noncomputable def light_source := (0, 1, 3)

theorem boundary_of_shadow_eq_minus_2 :
  ∀ x : ℝ, ∃ (g : ℝ → ℝ), g(x) = -2 :=
by
  sorry

end boundary_of_shadow_eq_minus_2_l160_160207


namespace count_four_digit_multiples_of_7_l160_160741

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160741


namespace decreasing_y_for_x_lt_1_l160_160336

theorem decreasing_y_for_x_lt_1 :
  ∀ x : ℝ, y : ℝ, y = 2 * (x - 1)^2 - 3 → (x < 1 → ∀ h : ℝ, (h > x → (2 * (h - 1)^2 - 3 < y))) :=
by
  sorry

end decreasing_y_for_x_lt_1_l160_160336


namespace complex_quadrant_l160_160932

theorem complex_quadrant (z : ℂ) (hz : z = (1 - (-1 : ℂ) * complex.I) / complex.I) : -1 < 0 ∧ -1 < 0 :=
by
  sorry

end complex_quadrant_l160_160932


namespace trader_sold_cloth_l160_160211

-- Define all conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 35
def cost_price_per_meter : ℕ := 70

-- Define our variable
def x (selling_price_per_meter : ℕ) : ℕ :=
  total_selling_price / selling_price_per_meter

-- The theorem we need to prove
theorem trader_sold_cloth (total_selling_price = 8925)
  (profit_per_meter = 35)
  (cost_price_per_meter = 70) :
  let selling_price_per_meter := cost_price_per_meter + profit_per_meter in
  x selling_price_per_meter = 85 :=
by
  -- Proof goes here
  sorry

end trader_sold_cloth_l160_160211


namespace count_four_digit_multiples_of_7_l160_160738

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160738


namespace length_of_gold_part_l160_160793

noncomputable def pencil_length : ℝ := 15
noncomputable def green_ratio : ℝ := 7 / 10
noncomputable def gold_ratio : ℝ := real.sqrt 2 / 2

theorem length_of_gold_part : ∃ (gold_length : ℝ), gold_length ≈ 3.182 ∧ 
  gold_length = (gold_ratio * ((1 - green_ratio) * pencil_length)) :=
by
  let remaining_length := (1 - green_ratio) * pencil_length
  let gold_length := gold_ratio * remaining_length
  use gold_length
  have h1 : remaining_length = 4.5 := by sorry
  have h2 : gold_length = (gold_ratio * 4.5) := by rw [h1]; refl
  have h3 : gold_length ≈ 3.182 := by sorry
  exact ⟨gold_length, h3, rfl⟩

end length_of_gold_part_l160_160793


namespace complex_quadrant_l160_160662

open Complex

-- Let complex number i be the imaginary unit
noncomputable def purely_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem complex_quadrant (z : ℂ) (a : ℂ) (hz : purely_imaginary z) (h : (2 + I) * z = 1 + a * I ^ 3) :
  (a + z).re > 0 ∧ (a + z).im < 0 :=
by 
  sorry

end complex_quadrant_l160_160662


namespace lisa_needs_28_more_marbles_l160_160861

theorem lisa_needs_28_more_marbles :
  ∀ (friends : ℕ) (initial_marbles : ℕ),
  friends = 12 → 
  initial_marbles = 50 →
  (∀ n, 1 ≤ n ∧ n ≤ friends → ∃ (marbles : ℕ), marbles ≥ 1 ∧ ∀ i j, (i ≠ j ∧ i ≠ 0 ∧ j ≠ 0) → (marbles i ≠ marbles j)) →
  ( ∑ k in finset.range (friends + 1), k ) - initial_marbles = 28 :=
by
  intros friends initial_marbles h_friends h_initial_marbles _,
  rw [h_friends, h_initial_marbles],
  sorry

end lisa_needs_28_more_marbles_l160_160861


namespace leap_day_2064_is_saturday_l160_160395

noncomputable def leap_day_weekday_2040 := 6 -- Saturday is typically represented as 6 if Sunday is 0.

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def leap_years_count (start_year end_year : ℕ) : ℕ :=
  (list.range' (end_year - start_year) start_year).countp is_leap_year

def total_days_between (start_year end_year : ℕ) : ℕ :=
  let non_leap_years := end_year - start_year - leap_years_count start_year end_year
  in non_leap_years * 365 + leap_years_count start_year end_year * 366

def weekday_after_days (start_weekday days : ℕ) : ℕ :=
  (start_weekday + days) % 7

theorem leap_day_2064_is_saturday :
  weekday_after_days leap_day_weekday_2040 (total_days_between 2040 2064) = 6 :=
by
  -- Proof details will go here.
  sorry

end leap_day_2064_is_saturday_l160_160395


namespace transformed_sum_l160_160576

theorem transformed_sum (n : ℕ) (s : ℕ) (A : fin n → ℕ) (h : (∑ i, A i) = s) :
     (∑ i, (5 * (A i + 10) - 10)) = 5 * s + 40 * n := 
by 
  sorry

end transformed_sum_l160_160576


namespace projectile_reaches_64ft_at_2sec_l160_160469

theorem projectile_reaches_64ft_at_2sec :
  ∃ t : ℝ, y = 64 ∧ y = -16 * t^2 + 64 * t ∧ t = 2 :=
by
  let y := 64
  have height_eq : y = -16 * 2^2 + 64 * 2 := by
    sorry
  use 2
  split
  · exact rfl
  split
  · exact height_eq
  · exact rfl

end projectile_reaches_64ft_at_2sec_l160_160469


namespace minimal_distance_l160_160993

noncomputable def y1 (U g t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def y2 (U g t τ : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
noncomputable def s (U g t τ : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem minimal_distance (U g τ : ℝ) (h : 2 * U ≥ g * τ) :
  ∃ t : ℝ, s U g t τ = 0 :=
begin
  use (2 * U / g + τ / 2),
  unfold s,
  sorry
end

end minimal_distance_l160_160993


namespace trajectory_midpoint_max_min_distance_origin_l160_160005

-- The parametric equations defining the curve
def curve_eqns (a b : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (a * Real.cos φ, b * Real.sin φ)

-- The coordinates of M, the midpoint of A(a * cos(α), b * sin(α)) and A(a * cos(α + π/2), b * sin(α + π/2))
def mid_pt (a b : ℝ) (α : ℝ) : ℝ × ℝ :=
  ((a * Real.cos α - a * Real.sin α) / 2, (b * Real.sin α + b * Real.cos α) / 2)

-- Trajectory of the midpoint M
theorem trajectory_midpoint (a b : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) :
  (2 * mid_pt a b α).fst^2 / a^2 + (2 * mid_pt a b α).snd^2 / b^2 = 1 := 
sorry

-- Maximum and minimum values of the distance from the origin O to the line AB
theorem max_min_distance_origin (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) :
  ∃ d, d = (a * b) / Real.sqrt (a^2 + b^2) := 
sorry

end trajectory_midpoint_max_min_distance_origin_l160_160005


namespace correct_option_l160_160160

theorem correct_option (a : ℝ) : (a^2)^3 = a^6 :=
by
  calc
    (a^2)^3 = a^(2 * 3) : by rw [pow_mul] 
           ... = a^6    : by norm_num

end correct_option_l160_160160


namespace problem_1_problem_2_l160_160684

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem problem_1 :
  ∀ x, x ∈ (U \ (A ∩ B)) ↔ x ≤ 4 ∨ x > 5 :=
by sorry

theorem problem_2 :
  ∀ a, (A ∪ B ⊆ C a) → a ≥ 6 :=
by sorry

end problem_1_problem_2_l160_160684


namespace correct_statements_count_l160_160416

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def has_rational_a_with_max_val_2 (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℚ), ∀ x : ℝ, f x = 2 * real.cos (((a + 1) * x) / 2) * real.cos (((a - 1) * x) / 2)

def is_periodic_for_irrational_a (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), irrational a ∧ ∀ x t : ℝ, f (x + t) = f x

def number_of_correct_statements (f : ℝ → ℝ) : ℕ :=
  [is_even_function f, has_rational_a_with_max_val_2 f, ¬ is_periodic_for_irrational_a f].count true

theorem correct_statements_count (a : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, real.cos x + real.cos (a * x)) :
  number_of_correct_statements f = 2 :=
sorry

end correct_statements_count_l160_160416


namespace lisa_needs_additional_marbles_l160_160869

theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 50) :
  ∃ additional_marbles : ℕ, additional_marbles = 78 - marbles ∧ additional_marbles = 28 :=
by
  -- The sum of the first 12 natural numbers is calculated as:
  have h_sum : (∑ i in finset.range (friends + 1), i) = 78 := by sorry
  -- The additional marbles needed:
  use 78 - marbles
  -- It should equal to 28:
  split
  . exact rfl
  . sorry

end lisa_needs_additional_marbles_l160_160869


namespace num_four_digit_multiples_of_7_l160_160723

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160723


namespace water_level_decrease_l160_160799

theorem water_level_decrease (increase_notation : ℝ) (h : increase_notation = 2) :
  -increase_notation = -2 :=
by
  sorry

end water_level_decrease_l160_160799


namespace value_of_x_l160_160154

theorem value_of_x :
  let x := (2023^2 - 2023 - 4^2) / 2023 in
  x = 2022 - (16 / 2023) :=
by
  sorry

end value_of_x_l160_160154


namespace triangle_side_lengths_consecutive_l160_160924

theorem triangle_side_lengths_consecutive (n : ℕ) (a b c A : ℕ) 
  (h1 : a = n - 1) (h2 : b = n) (h3 : c = n + 1) (h4 : A = n + 2)
  (h5 : 2 * A * A = 3 * n^2 * (n^2 - 4)) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
sorry

end triangle_side_lengths_consecutive_l160_160924


namespace exists_infinite_arith_prog_exceeding_M_l160_160438

def sum_of_digits(n : ℕ) : ℕ :=
n.digits 10 |> List.sum

theorem exists_infinite_arith_prog_exceeding_M (M : ℝ) :
  ∃ (a d : ℕ), ¬ (10 ∣ d) ∧ (∀ n : ℕ, a + n * d > 0) ∧ (∀ n : ℕ, sum_of_digits (a + n * d) > M) := by
sorry

end exists_infinite_arith_prog_exceeding_M_l160_160438


namespace matrix_inverse_property_l160_160926

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![7, e]]
def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (3 * e - 28)) • ![![e, -4], ![-7, 3]]
def B_squared (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![37, 12 + 4 * e], ![21 + 7 * e, 28 + e^2]]

theorem matrix_inverse_property (m : ℝ) (e : ℝ) (h : B_inv e = m • B_squared e) 
  : (e = -2) ∧ (m = 1 / 11) := by
  sorry

end matrix_inverse_property_l160_160926


namespace minimal_distance_l160_160991

noncomputable def y1 (U g t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def y2 (U g t τ : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
noncomputable def s (U g t τ : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem minimal_distance (U g τ : ℝ) (h : 2 * U ≥ g * τ) :
  ∃ t : ℝ, s U g t τ = 0 :=
begin
  use (2 * U / g + τ / 2),
  unfold s,
  sorry
end

end minimal_distance_l160_160991


namespace average_height_trees_l160_160568

/-- Define the problem conditions and compute the average height of the trees. -/
theorem average_height_trees (h₁ : ℕ) (h₂ : 6) (h₃ h₄ h₅ : ℕ)
  (H1 : h₁ = 2 * h₂ ∨ h₁ = h₂ / 2)
  (H3 : h₃ = 2 * h₂ ∨ h₃ = h₂ / 2)
  (H4 : h₄ = 2 * h₃ ∨ h₄ = h₃ / 2)
  (H5 : h₅ = 2 * h₄ ∨ h₅ = h₄ / 2)
  (H_avg : (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = 13.2) :
  True :=
by
  sorry

end average_height_trees_l160_160568


namespace ben_sue_number_problem_l160_160210

theorem ben_sue_number_problem (n : ℕ) : 
  let ben_thinks := 5 in
  let ben_adds_2 := ben_thinks + 2 in
  let ben_triples := ben_adds_2 * 3 in
  let sue_receives := ben_triples in
  let sue_subtracts_2 := sue_receives - 2 in
  let sue_quadruples := sue_subtracts_2 * 4 in
  sue_quadruples = 76 :=
by
  sorry

end ben_sue_number_problem_l160_160210


namespace exists_46_numbers_with_1000_proper_pairs_l160_160860

-- Define what it means for a pair to be "proper"
def proper_pair (a b : ℕ) : Prop :=
  a < b ∧ b % a = 0

-- Define the problem statement
theorem exists_46_numbers_with_1000_proper_pairs :
  ∃ (S : Finset ℕ), S.card = 46 ∧ (S.pairwise proper_pair).card = 1000 :=
sorry

end exists_46_numbers_with_1000_proper_pairs_l160_160860


namespace log_expression_value_l160_160247

theorem log_expression_value:
  (2 * log 10 2 - log 10 (1 / 25) = 2) := by
  sorry

end log_expression_value_l160_160247


namespace butterfat_milk_mixture_l160_160546

theorem butterfat_milk_mixture :
  ∃ (x : ℝ), 0.10 * x + 0.45 * 8 = 0.20 * (x + 8) ∧ x = 20 := by
  sorry

end butterfat_milk_mixture_l160_160546


namespace food_bank_remaining_l160_160419

theorem food_bank_remaining :
  ∀ (f1 f2 : ℕ) (p : ℚ),
  f1 = 40 →
  f2 = 2 * f1 →
  p = 0.7 →
  (f1 + f2) - (p * (f1 + f2)).toNat = 36 :=
by
  intros f1 f2 p h1 h2 h3
  sorry

end food_bank_remaining_l160_160419


namespace product_of_roots_eq_neg30_l160_160786

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l160_160786


namespace km_to_leaps_l160_160091

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps_l160_160091


namespace comparison_l160_160855

open Real

def a := (sqrt 2) / 2 * (sin (17 * π / 180) + cos (17 * π / 180))
def b := 2 * (cos (13 * π / 180))^2 - 1
def c := (sqrt 3) / 2

theorem comparison : c < a ∧ a < b := by
  have h_a : a = sin (62 * π / 180) := sorry
  have h_b : b = sin (64 * π / 180) := sorry
  have h_c : c = sin (60 * π / 180) := sorry
  -- We apply the monotonicity of the sine function for the intervals given
  have mono : ∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π) → sin x ≤ sin y := sorry
  apply And.intro
  { apply mono
    -- specific proof details are omitted
    sorry }
  { apply mono
    -- specific proof details are omitted
    sorry }
  -- done

end comparison_l160_160855


namespace equal_areas_of_triangles_l160_160831

theorem equal_areas_of_triangles
  (A B C D E F H_a H_b H_c : ℂ)
  (hD : D = (B + C) / 2)
  (hE : E = (C + A) / 2)
  (hF : F = (A + B) / 2)
  (hH_a : H_a = A + B + D - ((A / 2) + ((B - C) / (A - C))))
  (hH_b : H_b = B + C + E - ((B / 2) + ((C - A) / (B - A))))
  (hH_c : H_c = C + A + F - ((C / 2) + ((A - B) / (C - B))))
  : complex.abs ((D * complex.conj E - E * complex.conj D) +
                 (E * complex.conj F - F * complex.conj E) +
                 (F * complex.conj D - D * complex.conj F)) =
    complex.abs ((H_a * complex.conj H_b - H_b * complex.conj H_a) +
                 (H_b * complex.conj H_c - H_c * complex.conj H_b) +
                 (H_c * complex.conj H_a - H_a * complex.conj H_c)) :=
begin
  sorry
end

end equal_areas_of_triangles_l160_160831


namespace train_speed_is_61_l160_160980

noncomputable def speed_of_train (length_of_train_meters : ℕ) (time_seconds : ℕ) (man_speed_kmph : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph:ℝ) * 1000 / 3600
  let relative_speed_mps := length_of_train_meters / (time_seconds:ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := train_speed_mps * 3600 / 1000
  train_speed_kmph.toNat

theorem train_speed_is_61 :
  speed_of_train 110 6 5 = 61 :=
  by
  sorry

end train_speed_is_61_l160_160980


namespace four_digit_multiples_of_7_l160_160753

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160753


namespace count_four_digit_multiples_of_7_l160_160740

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l160_160740


namespace total_games_to_determine_winner_l160_160874

-- Conditions: Initial number of teams in the preliminary round
def initial_teams : ℕ := 24

-- Condition: Preliminary round eliminates 50% of the teams
def preliminary_round_elimination (n : ℕ) : ℕ := n / 2

-- Function to compute the required games for any single elimination tournament
def single_elimination_games (teams : ℕ) : ℕ :=
  if teams = 0 then 0
  else teams - 1

-- Proof Statement: Total number of games to determine the winner
theorem total_games_to_determine_winner (n : ℕ) (h : n = 24) :
  preliminary_round_elimination n + single_elimination_games (preliminary_round_elimination n) = 23 :=
by
  sorry

end total_games_to_determine_winner_l160_160874


namespace visited_both_countries_l160_160000

variables (totalPeople visitedIceland visitedNorway visitedNeither visitedBoth: ℕ)
hypothesis (h_total: totalPeople = 90)
hypothesis (h_iceland: visitedIceland = 55)
hypothesis (h_norway: visitedNorway = 33)
hypothesis (h_neither: visitedNeither = 53)
hypothesis (h_both: visitedBoth = visitedIceland + visitedNorway - (totalPeople - visitedNeither))

theorem visited_both_countries : visitedBoth = 51 :=
by {
  -- Proof would go here, but we're skipping it as per the instructions.
  sorry
}

end visited_both_countries_l160_160000


namespace graph_passes_through_quadrants_l160_160114

theorem graph_passes_through_quadrants :
  ∀ x : ℝ, (4 * x + 2 > 0 → (x > 0)) ∨ (4 * x + 2 > 0 → (x < 0)) ∨ (4 * x + 2 < 0 → (x < 0)) :=
by
  intro x
  sorry

end graph_passes_through_quadrants_l160_160114


namespace inequality_sqrt_sum_gt_sqrt2_abs_diff_l160_160850

theorem inequality_sqrt_sum_gt_sqrt2_abs_diff
  (a b c : ℝ) (h : a ≠ b) : 
  sqrt ((a - c) ^ 2 + b ^ 2) + sqrt (a ^ 2 + (b - c) ^ 2) > sqrt 2 * abs (a - b) := 
by 
  -- Here we would include the proof, but for now we just put a sorry to indicate that the proof is omitted
  sorry

end inequality_sqrt_sum_gt_sqrt2_abs_diff_l160_160850


namespace num_four_digit_multiples_of_7_l160_160724

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l160_160724


namespace series_diverges_l160_160818

open Filter Real

noncomputable def a_n (n : ℕ) : ℝ :=
  (-1)^(n+1) * (∏ k in finset.range n, (3 * (k + 1) - 2)) / (∏ k in finset.range n, (2 * (k + 1) + 1))

theorem series_diverges :
  let a := λ n, (-1)^(n+1) * (∏ k in finset.range n, (3 * (k + 1) - 2)) / (∏ k in finset.range n, (2 * (k + 1) + 1))
  in has_sum (λ n, a n) (diverges) :=
begin
  have h_ratio := λ n, (abs (a (n + 1) / a n)),
  have h_limit := limit_sup (h_ratio),
  have h_lim : tendsto h_ratio at_top (𝓝 (3 / 2)),
  {
    sorry,
  },
  have h_diverge : h_limit > 1,
  {
    rw [h_lim],
    norm_num,
  },
  exact series_test_ratio_diverge (lt_trans (by norm_num : 1 < 3 / 2) h_diverge)
end

end series_diverges_l160_160818


namespace zero_point_in_interval_l160_160941

noncomputable def f (x a : ℝ) : ℝ := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : (∃ x ∈ Ioo 1 2, f x a = 0) ↔ 0 < a ∧ a < 3 := by
  sorry

end zero_point_in_interval_l160_160941


namespace cream_butterfat_percentage_l160_160196

theorem cream_butterfat_percentage (x : ℝ) (h1 : 1 * (x / 100) + 3 * (5.5 / 100) = 4 * (6.5 / 100)) : 
  x = 9.5 :=
by
  sorry

end cream_butterfat_percentage_l160_160196


namespace sector_to_cone_base_radius_and_slant_height_l160_160159

theorem sector_to_cone_base_radius_and_slant_height 
    (r θ : ℝ)
    (hr : r = 12)
    (hθ : θ = 270) :
  let arc_length := (θ / 360) * 2 * Real.pi * r in
  let base_radius := arc_length / (2 * Real.pi) in
  base_radius = 9 ∧ r = 12 :=
by
  sorry

end sector_to_cone_base_radius_and_slant_height_l160_160159


namespace largest_prime_factor_l160_160162

theorem largest_prime_factor : ∃ p : ℕ, nat.prime p ∧ p ≤ 16^4 + 4 * 16^2 + 4 - 15^4 ∧ 
  (∀ q : ℕ, nat.prime q ∧ q ≤ 16^4 + 4 * 16^2 + 4 - 15^4 → q ≤ p) ∧
  p = 23 := 
by
  sorry

end largest_prime_factor_l160_160162


namespace smallest_n_for_2n_3n_5n_conditions_l160_160524

theorem smallest_n_for_2n_3n_5n_conditions : 
  ∃ n : ℕ, 
    (∀ k : ℕ, 2 * n ≠ k^2) ∧          -- 2n is a perfect square
    (∀ k : ℕ, 3 * n ≠ k^3) ∧          -- 3n is a perfect cube
    (∀ k : ℕ, 5 * n ≠ k^5) ∧          -- 5n is a perfect fifth power
    n = 11250 :=
sorry

end smallest_n_for_2n_3n_5n_conditions_l160_160524


namespace first_applicant_earnings_l160_160204

def first_applicant_salary : ℕ := 42000
def first_applicant_training_cost_per_month : ℕ := 1200
def first_applicant_training_months : ℕ := 3
def second_applicant_salary : ℕ := 45000
def second_applicant_bonus_percentage : ℕ := 1
def company_earnings_from_second_applicant : ℕ := 92000
def earnings_difference : ℕ := 850

theorem first_applicant_earnings 
  (salary1 : first_applicant_salary = 42000)
  (train_cost_per_month : first_applicant_training_cost_per_month = 1200)
  (train_months : first_applicant_training_months = 3)
  (salary2 : second_applicant_salary = 45000)
  (bonus_percentage : second_applicant_bonus_percentage = 1)
  (earnings2 : company_earnings_from_second_applicant = 92000)
  (earning_diff : earnings_difference = 850) :
  (company_earnings_from_second_applicant - (second_applicant_salary + (second_applicant_salary * second_applicant_bonus_percentage / 100)) - earnings_difference) = 45700 := 
by 
  sorry

end first_applicant_earnings_l160_160204


namespace ellipse_foci_distance_l160_160630

noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 29

theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 
  (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  distance_between_foci = 2 * Real.sqrt 29 := 
by
  intros x y h
  -- proof goes here (skipped)
  sorry

end ellipse_foci_distance_l160_160630


namespace proof_problem_l160_160314

variable {α β : ℝ}

noncomputable def is_acutes (α β : ℝ) : Prop :=
α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2

theorem proof_problem (h1: is_acutes α β)
            (h2: cos α + cos (2 * β) - cos (α + β) = 3 / 2) : α = π / 3 ∧ β = π / 3 := 
by
  sorry

end proof_problem_l160_160314


namespace min_distance_and_distance_from_Glafira_l160_160997

theorem min_distance_and_distance_from_Glafira 
  (U g τ V : ℝ) (h : 2 * U ≥ g * τ) :
  let T := (τ / 2) + (U / g) in
  s T = 0 ∧ (V * T = V * (τ / 2 + U / g)) :=
by
  -- Define the positions y1(t) and y2(t)
  let y1 := λ t, U * t - (g * t^2) / 2
  let y2 := λ t, U * (t - τ) - (g * (t - τ)^2) / 2
  -- Define the distance s(t)
  let s := λ t, |y1 t - y2 t|
  -- Start the proof
  sorry

end min_distance_and_distance_from_Glafira_l160_160997


namespace michel_operations_distinct_strings_l160_160052

/-- Michel starts with the string "HMMT". An operation consists of either replacing an occurrence
of H with HM, replacing an occurrence of MM with MOM, or replacing an occurrence of T with MT.
Prove that the number of distinct strings Michel can obtain after exactly 10 operations is 370. -/
theorem michel_operations_distinct_strings :
  let initial_string := "HMMT"
  let operation := λ (str: String) =>
    str.replace "H" "HM" ∨ str.replace "MM" "MOM" ∨ str.replace "T" "MT"
  (number_of_distinct_strings_after_operations initial_string operation 10) = 370 :=
sorry

end michel_operations_distinct_strings_l160_160052


namespace afternoon_snack_calories_l160_160621

def ellen_daily_calories : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def dinner_remaining_calories : ℕ := 832

theorem afternoon_snack_calories :
  ellen_daily_calories - (breakfast_calories + lunch_calories + dinner_remaining_calories) = 130 :=
by sorry

end afternoon_snack_calories_l160_160621


namespace multiplicative_inverse_144_mod_941_l160_160931

theorem multiplicative_inverse_144_mod_941 :
  (∃ (a b : ℤ), 144 * a + 941 * b = 1) → ∃ n : ℕ, 0 ≤ n ∧ n < 941 ∧ 144 * n ≡ 1 [MOD 941] :=
by
  assume h
  have h1 : 65^2 + 72^2 = 97^2 := by sorry
  have h2 : 144 * 364 % 941 = 1 := by sorry
  exact ⟨364, by sorry, by sorry, by sorry⟩

end multiplicative_inverse_144_mod_941_l160_160931


namespace antisymmetric_zero_square_implies_zero_l160_160396

variables {n : Type} [fintype n] [decidable_eq n]

def antisymmetric_matrix (A : matrix n n ℝ) : Prop :=
∀ i j, A i j = -A j i

def zero_matrix (A : matrix n n ℝ) : Prop :=
∀ i j, A i j = 0

theorem antisymmetric_zero_square_implies_zero
  (A : matrix n n ℝ)
  (h₁ : antisymmetric_matrix A)
  (h₂ : A * A = 0) :
  zero_matrix A :=
sorry

end antisymmetric_zero_square_implies_zero_l160_160396


namespace four_digit_multiples_of_7_l160_160713

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l160_160713
