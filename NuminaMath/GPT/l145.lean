import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Factorial
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Factorial.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Matrix.PEq
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.GroupSimp
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.LinearIndependent
import Mathlib.MeasureTheory.Constructions.Integrate
import Mathlib.NumberTheory.Divisors
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Topology.Algebra.Affine
import Mathlib.Topology.Algebra.Group
import Real
import mathlib

namespace count_concave_downward_functions_l145_145512

def isConcaveDownward (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 → f((x1 + x2) / 2) < (f x1 + f x2) / 2

noncomputable def f1 : ℝ → ℝ := λ x, 2^x
noncomputable def f2 : ℝ → ℝ := λ x, Real.log x / Real.log 2
noncomputable def f3 : ℝ → ℝ := λ x, x^(1/2)

theorem count_concave_downward_functions :
    (if isConcaveDownward f1 0 1 then 1 else 0) +
    (if isConcaveDownward f2 0 1 then 1 else 0) +
    (if isConcaveDownward f3 0 1 then 1 else 0) = 1 := 
    sorry

end count_concave_downward_functions_l145_145512


namespace column_half_zeroes_l145_145743

theorem column_half_zeroes 
  (n : ℕ) 
  (hn : n > 2) 
  (A : fin n → vector (fin 6) (fin 2)) 
  (h_distinct : function.injective A) 
  (h_pairwise : ∀ i j : fin n, (A i).to_list.zip_with (*) (A j).to_list ∈ (set.range (λ k : fin n, (A k).to_list))) :
  ∃ j : fin 6, (∑ i : fin n, if A i j = 0 then 1 else 0) ≥ n / 2 := 
sorry

end column_half_zeroes_l145_145743


namespace count_two_digit_numbers_with_prime_sum_l145_145881

-- Define a predicate that checks if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

-- Define the sum of digits function for a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the condition that checks if a two-digit number has a prime sum of digits
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ is_prime (sum_of_digits n)

-- Claim to prove
theorem count_two_digit_numbers_with_prime_sum : (finset.filter is_valid_number (finset.range 100)).card = 34 :=
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145881


namespace compare_AM_with_BM_plus_CM_l145_145582

open EuclideanGeometry

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Define the isosceles triangle ABC
variables (A B C O M : α)

-- Assume AB = AC (isosceles triangle)
variables (h_isosceles : dist A B = dist A C)

-- Assume ABC is inscribed in a circle O
variables (h_circ : ∃ (O : α) (r : ℝ), (dist A O = r) ∧ (dist B O = r) ∧ (dist C O = r))

-- Assume M is the midpoint of the arc BC that does not contain A
variables (h_midpoint_arc : is_midpoint_of_arc_not_containing (A B C : set α) O M)

-- Define the segments AM, BM, and CM
variables (AM BM CM : ℝ)
variables (h_AM : AM = dist A M) (h_BM : BM = dist B M) (h_CM : CM = dist C M)

-- The theorem we need to prove
theorem compare_AM_with_BM_plus_CM 
  (h1 : isosceles_with_equal_sides A B C h_isosceles)
  (h2 : inscribed_in_circle A B C O h_circ)
  (h3 : midpoint_of_arc_not_containing_A A B C O M h_midpoint_arc)
  (h4 : define_segment_lengths A B C M AM BM CM h_AM h_BM h_CM) :
  AM > BM + CM := sorry


end compare_AM_with_BM_plus_CM_l145_145582


namespace find_general_term_l145_145844

noncomputable def sequence (n : ℕ) : ℝ := (1/3)^n

theorem find_general_term (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h : ∀ n, S n = 1/2 * (1 - a n)) :
  ∀ n, a n = (1/3)^n :=
by
  sorry

end find_general_term_l145_145844


namespace colten_chickens_l145_145997

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l145_145997


namespace stratified_sampling_condition_l145_145644

-- Define the conditions
def population_small : Prop := ∃ (p : Type), p.nonempty ∧ (p.card ≤ k) for some small k
def population_large : Prop := ∃ (p : Type), p.nonempty ∧ (p.card > k) for some large k
def population_clear_diff : Prop := ∃ (p : Type), ∃ (subparts : p → Prop), ∀ x y, subparts x ≠ subparts y

-- Define the property of being appropriate for stratified sampling
def stratified_sampling_appropriate (p : Type) : Prop :=
  ∀ subset, subset ⊂ p → stratified_sampling.sample subset

-- The theorem stating that the appropriate range for stratified sampling is when the population has clear differences
theorem stratified_sampling_condition :
  population_clear_diff → stratified_sampling_appropriate population_clear_diff :=
by
  sorry

end stratified_sampling_condition_l145_145644


namespace triangulation_count_convex_seven_gon_l145_145003

theorem triangulation_count_convex_seven_gon :
  let number_of_sides := 7
  let number_of_diagonals := 4
  let number_of_triangles := 5
  ∀ (sides : ℕ), sides = number_of_sides →
    ∀ (dia : ℕ), dia = number_of_diagonals →
      ∀ (tri : ℕ), tri = number_of_triangles →
        let n := sides - 2 in
        let C := λ n, (1 / (n + 1 : ℝ) * (nat.choose (2 * n) n)) in
        C n = 42 := 
by 
  intros sides eqs 
  intros dia eqd 
  intros tri eqt 
  sorry

end triangulation_count_convex_seven_gon_l145_145003


namespace binary_equals_octal_l145_145788

-- Define the binary number 1001101 in decimal
def binary_1001101_decimal : ℕ := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the octal number 115 in decimal
def octal_115_decimal : ℕ := 1 * 8^2 + 1 * 8^1 + 5 * 8^0

-- Theorem statement
theorem binary_equals_octal :
  binary_1001101_decimal = octal_115_decimal :=
sorry

end binary_equals_octal_l145_145788


namespace shifted_graph_coefficients_sum_l145_145297

theorem shifted_graph_coefficients_sum :
  let y_orig := λ x, 3 * x^2 + 2 * x + 4
      y_shifted := λ x, 3 * (x + 3)^2 + 2 * (x + 3) + 4
      a := 3
      b := 20
      c := 37
  in a + b + c = 60 := by sorry

end shifted_graph_coefficients_sum_l145_145297


namespace men_build_walls_l145_145895

-- Define the variables
variables (a b d y : ℕ)

-- Define the work rate based on given conditions
def rate := d / (a * b)

-- Theorem to prove that y equals (a * a) / d given the conditions
theorem men_build_walls (h : a * b * y = a * a * d / a) : 
  y = a * a / d :=
by sorry

end men_build_walls_l145_145895


namespace abs_real_part_le_sum_abs_l145_145007

variables {n : ℕ} (x y : ℕ → ℝ)
noncomputable def r : ℝ :=
  abs (real.sqrt (finset.sum (finset.range n) (λ k, (x k)^2 - (y k)^2)) : ℝ)

theorem abs_real_part_le_sum_abs (z : ℕ → ℂ) (h : ∀ k, z k = x k + y k * complex.I):
  let r := abs (complex.re (real.sqrt (finset.sum (finset.range n) (λ k, (z k)^2)))) in
  r ≤ finset.sum (finset.range n) (λ k, abs (x k)) :=
begin
  sorry
end

end abs_real_part_le_sum_abs_l145_145007


namespace total_students_at_concert_l145_145277

theorem total_students_at_concert :
  let schoolA_students := 15 * 30 in
  let schoolB_students := (18 * 7) + (5 * 6) in
  let schoolC_students := (13 * 33) + (10 * 4) in
  let total_students := schoolA_students + schoolB_students + schoolC_students in
  total_students = 1075 :=
by
  let schoolA_students := 15 * 30
  let schoolB_students := (18 * 7) + (5 * 6)
  let schoolC_students := (13 * 33) + (10 * 4)
  let total_students := schoolA_students + schoolB_students + schoolC_students
  show total_students = 1075 from sorry

end total_students_at_concert_l145_145277


namespace find_number_l145_145311

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
sorry

end find_number_l145_145311


namespace solve_quadratic_eq_l145_145230

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 4 * x + 2 = 0 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end solve_quadratic_eq_l145_145230


namespace matrix_determinant_l145_145779

theorem matrix_determinant (x : ℝ) :
  Matrix.det ![![x, x + 2], ![3, 2 * x]] = 2 * x^2 - 3 * x - 6 :=
by
  sorry

end matrix_determinant_l145_145779


namespace general_form_of_numbers_whose_square_ends_with_9_l145_145405

theorem general_form_of_numbers_whose_square_ends_with_9 (x : ℤ) (h : (x^2 % 10 = 9)) :
  ∃ a : ℤ, x = 10 * a + 3 ∨ x = 10 * a + 7 :=
sorry

end general_form_of_numbers_whose_square_ends_with_9_l145_145405


namespace find_point_P_l145_145027

open Real

-- Define the conditions as stated
def parametric_curve (θ : ℝ) : ℝ × ℝ := (3 * cos θ, 4 * sin θ)
def within_interval (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ π
def angle_inclination : ℝ := π / 4
def point_on_line (p : ℝ × ℝ) : Prop := p.2 = p.1
def point_on_curve (p : ℝ × ℝ) : Prop := (p.1^2 / 9 + p.2^2 / 16 = 1) ∧ (p.2 ≥ 0)

-- The proof problem statement
theorem find_point_P : ∃ P : ℝ × ℝ, point_on_curve P ∧ point_on_line P ∧ P = (12 / 5, 12 / 5) :=
by
  sorry

end find_point_P_l145_145027


namespace number_of_nonzero_digits_right_of_decimal_l145_145523

theorem number_of_nonzero_digits_right_of_decimal (n : ℤ) (h : n = 74) : 
  let fraction := (n : ℚ) / (2^2 * 5^3 * 7^2 : ℚ) in
  (∃ digits : ℕ, digits = 5 ∧ fraction.toDecimal.digits.count (≠ 0) = digits) :=
by
  sorry

end number_of_nonzero_digits_right_of_decimal_l145_145523


namespace petya_has_higher_chance_of_winning_l145_145202

noncomputable def vasya_win_probability : ℝ := 0.46
noncomputable def petya_win_probability : ℝ := 1 - vasya_win_probability

theorem petya_has_higher_chance_of_winning 
  (h: vasya_win_probability = 1 - 0.54) : petya_win_probability > 0.5 :=
by
  have hp : petya_win_probability = 1 - vasya_win_probability := rfl
  have h_vasya : vasya_win_probability = 0.46 := h
  rw [h_vasya] at hp
  have h_petya : petya_win_probability = 1 - 0.46 := hp
  simp [h_petya]
  exact by norm_num

end petya_has_higher_chance_of_winning_l145_145202


namespace range_of_x_l145_145659

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a) : 
  ax^2 + (a - 3) * x + (a - 4) > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end range_of_x_l145_145659


namespace prod_mod_6_l145_145379

theorem prod_mod_6 (h1 : 2015 % 6 = 3) (h2 : 2016 % 6 = 0) (h3 : 2017 % 6 = 1) (h4 : 2018 % 6 = 2) : 
  (2015 * 2016 * 2017 * 2018) % 6 = 0 := 
by 
  sorry

end prod_mod_6_l145_145379


namespace intersection_complement_P_C_Z_Q_l145_145168

-- Define all necessary sets and conditions from part (a)
def P : Set ℕ := {x | x ≤ 4 ∧ 0 < x}  -- Here, x ∈ ℕ* means positive natural number ℕ without zero
def Q : Set ℤ := {x | 3 < x}

-- Define the complement of set Q in integers
def C_Z_Q : Set ℤ := {x | ¬ (3 < x)}

-- Translate the proof problem into a Lean statement
theorem intersection_complement_P_C_Z_Q :
  P ∩ C_Z_Q = {1, 2, 3} := sorry

end intersection_complement_P_C_Z_Q_l145_145168


namespace analytical_expression_f_is_correct_range_of_m_l145_145023

noncomputable def f (x : ℝ) : ℝ := (2*x) / (x^2 + 1)

theorem analytical_expression_f_is_correct (a b : ℝ) 
    (hodd : ∀ x : ℝ, f (-x) = -f x) 
    (hval : f (1/2) = 4/5) : f x = (2*x) / (x^2 + 1) :=
sorry

theorem range_of_m (A : set ℝ) (B : set ℝ) : 
    A = set.Icc (-1 : ℝ) (1 : ℝ) → 
    B = set.Icc (1 - m : ℝ) (2 * m : ℝ) → 
    A ⊆ B → 
    ∃ m : ℝ, m ≥ 2 :=
sorry

end analytical_expression_f_is_correct_range_of_m_l145_145023


namespace petya_has_higher_chance_l145_145191

theorem petya_has_higher_chance : 
  let p_chocolate_chocolate : ℝ := 1 - 0.54 in
  p_chocolate_chocolate < 0.5 :=
by
  let p_chocolate_chocolate := 1 - 0.54
  have h1 : p_chocolate_chocolate = 0.46 := by norm_num
  have h2 : 0.46 < 0.5 := by norm_num
  exact h2

end petya_has_higher_chance_l145_145191


namespace find_a_l145_145531

theorem find_a (a : ℝ) (h : coefficient (polynomial.expand (1 + a * x)^5) 3 = -80) : a = -2 :=
sorry

end find_a_l145_145531


namespace part_one_part_two_l145_145503

open Real

variable {a b m : ℝ}

theorem part_one (m : ℝ) : 
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  in Δ > 0 := 
by
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  show Δ > 0 from by {
    simp[Δ],
    linarith
  }

theorem part_two (a b m : ℝ) (h1 : a + b = 2 * m + 1) (h2 : a * b = m^2 + m) (h3 : (2*a + b) * (a + 2*b) = 20) : 
  m = -2 ∨ m = 1 :=
by
  have h4: 9*m^2 + 9*m + 2 = 20 := by {
    rw [h1, h2, ← add_assoc, ← add_assoc],
    linarith
  }
  have h5 := solve_quadratic 9 9 (-18) h4
  cases h5 with h5_left h5_right
  exact or.inl h5_left
  exact or.inr h5_right

end part_one_part_two_l145_145503


namespace petya_numbers_board_l145_145987

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end petya_numbers_board_l145_145987


namespace volume_of_rotated_segment_l145_145623

-- Definitions based on conditions
variables (a h : ℝ)

-- Statement to prove
theorem volume_of_rotated_segment (a h : ℝ) : 
  volume_of_rotated_segment a h = (π * a ^ 2 * h) / 6 := 
sorry

end volume_of_rotated_segment_l145_145623


namespace handshake_count_l145_145769

-- Define the number of team members, referees, and the total number of handshakes
def num_team_members := 7
def num_referees := 3
def num_coaches := 2

-- Calculate the handshakes
def team_handshakes := num_team_members * num_team_members
def player_refhandshakes := (2 * num_team_members) * num_referees
def coach_handshakes := num_coaches * (2 * num_team_members + num_referees)

-- The total number of handshakes
def total_handshakes := team_handshakes + player_refhandshakes + coach_handshakes

-- The proof statement
theorem handshake_count : total_handshakes = 125 := 
by
  -- Placeholder for proof
  sorry

end handshake_count_l145_145769


namespace missing_digit_divisibility_by_nine_l145_145244

theorem missing_digit_divisibility_by_nine (x : ℕ) (h : 0 ≤ x ∧ x < 10) :
  9 ∣ (3 + 5 + 2 + 4 + x) → x = 4 :=
by
  sorry

end missing_digit_divisibility_by_nine_l145_145244


namespace ribbon_leftover_correct_l145_145288

def initial_ribbon : ℕ := 84
def used_ribbon : ℕ := 46
def leftover_ribbon : ℕ := 38

theorem ribbon_leftover_correct : initial_ribbon - used_ribbon = leftover_ribbon :=
by
  sorry

end ribbon_leftover_correct_l145_145288


namespace mean_diff_l145_145831

-- Definitions for the conditions
variables {a1 a2 a3 a4 a5 a6 A : ℝ}

-- Conditions as hypotheses
hypothesis h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A
hypothesis h2 : (a1 + a2 + a3 + a4) / 4 = A + 10
hypothesis h3 : (a3 + a4 + a5 + a6) / 4 = A - 7

-- The proposition we need to prove
theorem mean_diff : (a1 + a2 + a5 + a6) / 4 = A - 3 :=
by
  sorry

end mean_diff_l145_145831


namespace quadratic_discriminant_positive_find_m_l145_145500

-- Define the quadratic equation and its properties
def quadratic (x m : Int) : Int := x^2 - (2 * m + 1) * x + (m^2 + m)

-- Part (1)
theorem quadratic_discriminant_positive (m : Int) : 
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  Δ > 0 := by
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  have h : Δ = 1 := by sorry
  show 1 > 0 from by sorry

-- Part (2)
theorem find_m (a b m : Int) : 
  quadratic a m = 0 ∧ quadratic b m = 0 ∧ (2 * a + b) * (a + 2 * b) = 20 → 
  m = -2 ∨ m = 1 := by
  intro h
  sorry

end quadratic_discriminant_positive_find_m_l145_145500


namespace first_day_of_month_is_sunday_l145_145642

def day_of_week := ℕ  -- Assuming days of the week are represented by natural numbers where 0 = Sunday, 1 = Monday, ..., 6 = Saturday

-- Define a function to find the day of the week by subtracting k days
def subtract_days (d : day_of_week) (k : ℕ) : day_of_week :=
  (d + 7 - (k % 7)) % 7

theorem first_day_of_month_is_sunday (D : day_of_week) (h : subtract_days D 23 = 2) : D = 0 :=
begin
  -- Given that the 24th is a Tuesday (D = 2)
  sorry
end

end first_day_of_month_is_sunday_l145_145642


namespace initial_pens_l145_145985

-- Conditions as definitions
def initial_books := 108
def books_after_sale := 66
def books_sold := 42
def pens_after_sale := 59

-- Theorem statement proving the initial number of pens
theorem initial_pens:
  initial_books - books_after_sale = books_sold →
  ∃ (P : ℕ), P - pens_sold = pens_after_sale ∧ (P = 101) :=
by
  sorry

end initial_pens_l145_145985


namespace fred_seashells_now_l145_145815

def seashells_initial := 47
def seashells_given := 25

theorem fred_seashells_now : seashells_initial - seashells_given = 22 := 
by 
  sorry

end fred_seashells_now_l145_145815


namespace max_value_expression_l145_145426

theorem max_value_expression :
  ∃ (x y : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧
  (sqrt (9 - sqrt 7) * sin x - sqrt (2 * (1 + cos (2 * x))) - 1) *
   (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) = 19 :=
by
  sorry

end max_value_expression_l145_145426


namespace trigonometric_identity_l145_145050

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l145_145050


namespace geom_mean_inequality_l145_145966

theorem geom_mean_inequality (n : ℕ) (h : n > 1) (x : Fin n → ℝ) 
    (pos_x : ∀ i, 0 < x i) 
    (sum_inv : (∑ i, 1 / (1 + x i)) = 1) :
    (∏ i, x i) ≥ (n - 1) ^ n :=
by
  sorry

end geom_mean_inequality_l145_145966


namespace max_expenditure_l145_145840

theorem max_expenditure (b a x : ℝ) (e : ℝ) (hb : b = 0.8) (ha : a = 2) (h_x : x = 10) (h_e : |e| ≤ 0.5) :
  let y := b * x + a + e in y ≤ 10.5 :=
by 
  sorry

end max_expenditure_l145_145840


namespace num_ways_to_select_officers_l145_145750

def members := Fin 24 -- 24 members indexed from 0 to 23
def is_officer (a b : members) : Prop := -- a and b must either both be officers or neither
  (a ∈ {a, b} ∧ b ∈ {a, b}) ∨ (a ∉ {a, b} ∧ b ∉ {a, b})

noncomputable def count_ways : ℕ :=
  let rachel := (0 : members)
  let samuel := (1 : members)
  let tim := (2 : members)
  let uma := (3 : members)
  if is_officer rachel samuel ∧ is_officer tim uma then
    -- Case 1: Neither Rachel and Samuel nor Tim and Uma are officers
    20 * 19 * 18 * 17
  else if is_officer rachel samuel ∧ ¬ is_officer tim uma then
    -- Case 2: Rachel and Samuel are officers, but not Tim and Uma
    6 * 2 * 20 * 19
  else if ¬ is_officer rachel samuel ∧ is_officer tim uma then
    -- Case 3: Tim and Uma are officers, but not Rachel and Samuel
    6 * 2 * 20 * 19
  else
    -- Case 4: Both pairs Rachel & Samuel and Tim & Uma are officers
    6 * 2 * 2

theorem num_ways_to_select_officers :
  count_ways = 126424 := by
  sorry

end num_ways_to_select_officers_l145_145750


namespace quadratic_eq_two_distinct_real_roots_find_m_value_l145_145496

noncomputable def quadratic_eq_discriminant {m : ℝ} : ℝ :=
    (-(2 * m + 1)) ^ 2 - 4 * 1 * (m ^ 2 + m)

theorem quadratic_eq_two_distinct_real_roots (m : ℝ) : quadratic_eq_discriminant m > 0 :=
    by
    sorry

-- Let's define the conditions for the second part
def quadratic_eq_roots (m a b : ℝ) :=
    a + b = 2 * m + 1 ∧ a * b = m ^ 2 + m

def equation_holds (a b : ℝ) :=
    (2 * a + b) * (a + 2 * b) = 20

theorem find_m_value (m a b : ℝ) (h : quadratic_eq_roots m a b) (h_eq : equation_holds a b) :
    m = -2 ∨ m = 1 :=
    by
    sorry

end quadratic_eq_two_distinct_real_roots_find_m_value_l145_145496


namespace evaluate_expression_l145_145580

theorem evaluate_expression (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by 
  rw h
  sorry

end evaluate_expression_l145_145580


namespace maximize_area_of_pasture_l145_145745

theorem maximize_area_of_pasture :
  ∀ (barn_length fence_cost total_cost x : ℝ),
    barn_length = 400 ∧ fence_cost = 5 ∧ total_cost = 1200 →
    (2 * x + (240 - 2 * x) = 240) →
    ∃ (max_y : ℝ), (240 - 2 * max_y) = 120 :=
by
  intros barn_length fence_cost total_cost x hcost htotal
  use 60
  split
  { linarith }
  { sorry }

end maximize_area_of_pasture_l145_145745


namespace find_a_l145_145061

-- Definitions
def point_on_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 + 4*x + a*y - 5 = 0

def symmetric_point (x y : ℝ) : ℝ × ℝ :=
  let m := -2 / 1 in -- slope of the line perpendicular to 2x + y - 1 = 0
  let c := 1 / 1 in   -- y-intercept of the perpendicular line through (x, y)
  let d := (x + m * y) / (1 + m^2) in
  let (x', y') := (2*d - x, 2*d * m - y) in
  (x', y')

-- Theorem
theorem find_a : ∀ (x y : ℝ), point_on_circle x y a → (let (xs, ys) := symmetric_point x y in point_on_circle xs ys a) → a = -10 :=
by sorry

end find_a_l145_145061


namespace books_sold_on_tuesday_l145_145568

theorem books_sold_on_tuesday (total_books books_sold_monday books_sold_wednesday books_sold_thursday books_sold_friday : ℕ)
  (percentage_books_not_sold : ℝ) (h_total_books : total_books = 1300) (h_books_sold_monday : books_sold_monday = 75)
  (h_books_sold_wednesday : books_sold_wednesday = 64) (h_books_sold_thursday : books_sold_thursday = 78)
  (h_books_sold_friday : books_sold_friday = 135) 
  (h_percentage_books_not_sold : percentage_books_not_sold = 69.07692307692308)
  : (total_books * (1 - percentage_books_not_sold / 100)).to_nat
  = books_sold_monday + 50 + books_sold_wednesday + books_sold_thursday + books_sold_friday :=
by sorry

end books_sold_on_tuesday_l145_145568


namespace minimum_value_of_f_l145_145527

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y, y = f x ∧ y = 1 :=
by
  sorry

end minimum_value_of_f_l145_145527


namespace total_turtles_l145_145691

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end total_turtles_l145_145691


namespace find_ratio_l145_145829

variable {x y z : ℝ}

theorem find_ratio
  (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end find_ratio_l145_145829


namespace three_numbers_divisibility_probability_l145_145682

noncomputable def probability_divides (s : Finset ℕ) (n : ℕ) : ℚ :=
  let all_combinations := s.powerset.filter (λ t => t.card = n)
  let valid_combinations := all_combinations.filter (λ t => 
    match t.toList with
    | [] => false
    | [a, b, c] => a ∣ b ∧ a ∣ c
    | _ => false
  )
  (valid_combinations.card : ℚ) / all_combinations.card

theorem three_numbers_divisibility_probability :
  probability_divides ({1, 2, 3, 4, 5, 6}.toFinset) 3 = 1 / 4 :=
  sorry

end three_numbers_divisibility_probability_l145_145682


namespace det_projection_matrix_l145_145586

open Matrix

noncomputable def P_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let a : ℝ := 3
  let b : ℝ := -2
  let norm_sq := a^2 + b^2
  ((λ i j, (if i = 0 then a else b) * (if j = 0 then a else b) / norm_sq) : Matrix (Fin 2) (Fin 2) ℝ)

theorem det_projection_matrix : det P_matrix = 0 := by
  sorry

end det_projection_matrix_l145_145586


namespace product_evaluation_l145_145294

theorem product_evaluation :
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 *
  (1 / 512) * 1024 * (1 / 2048) * 4096 = 64 :=
by
  sorry

end product_evaluation_l145_145294


namespace intersect_lines_and_ratio_l145_145250

theorem intersect_lines_and_ratio (
  (A B C A1 B1 C1 P : Point)
  (l_a l_b l_c : Line)
  (M : Point)
  (h1 : is_median A A1 B C)
  (h2 : is_median B B1 A C)
  (h3 : is_median C C1 A B)
  (h4 : are_concurrent A A1 B B1 C C1 M)
  (h5 : passes_through A l_a)
  (h6 : parallel_to l_a (line_through P A1))
  (h7 : passes_through B l_b)
  (h8 : parallel_to l_b (line_through P B1))
  (h9 : passes_through C l_c)
  (h10 : parallel_to l_c (line_through P C1))
) : 
  ∃ Q : Point, 
  (on_line Q l_a ∧ on_line Q l_b ∧ on_line Q l_c) ∧ 
  collinear P M Q ∧ 
  ratio P M Q 1 2 := 
sorry

end intersect_lines_and_ratio_l145_145250


namespace infinite_series_sum_l145_145776

theorem infinite_series_sum :
  ∑' n : ℕ, (n + 1) * (1 / 1950)^n = 3802500 / 3802601 :=
by
  sorry

end infinite_series_sum_l145_145776


namespace two_digit_prime_sum_count_l145_145888

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_prime_sum_count : 
  let prime_sums := {2, 3, 5, 7, 11, 13, 17}
  ∑ k in (Finset.filter (λ n, is_prime (sum_of_digits n)) (Finset.filter two_digit (Finset.range 100))), 1 = 31 := 
by
  sorry

end two_digit_prime_sum_count_l145_145888


namespace coefficient_of_term_in_expansion_l145_145474

theorem coefficient_of_term_in_expansion (n : ℕ) : 
  (∑ (k : ℕ) in finset.range (n+1), (2:ℕ)^k) = 32 →
  n = 5 →
  (2 * x^2 + x - y)^n = (∑ (p q r : ℕ) in finset.nat.antidiagonal n, 
  (nat.choose n r) * (-y)^r * (nat.choose (n-r) q) * (2 * x^2)^(n-r-q) * x^q) →
  (nat.choose 5 2) * (nat.choose 3 1) * 2^2 = 120 :=
by
  intros hsum hn hcoeff
  sorry

end coefficient_of_term_in_expansion_l145_145474


namespace determine_M_l145_145789

theorem determine_M (M : ℕ) (h : 12 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2) : M = 36 :=
by
  sorry

end determine_M_l145_145789


namespace sinθ_max_value_l145_145460

noncomputable def sinθ_max (a b : EuclideanSpace ℝ (Fin 2)) (θ : ℝ) : ℝ := 
  if (‖a‖ = 1) ∧ (‖b - 2 • a‖ = 1) then Real.sin θ else 0

theorem sinθ_max_value (a b : EuclideanSpace ℝ (Fin 2)) (θ : ℝ) 
                  (h₁ : ‖a‖ = 1) (h₂ : ‖b - 2 • a‖ = 1) 
                  (θ_eq : θ = Real.inner a b) :
  sinθ_max a b θ = 1 / 2 := 
sorry

end sinθ_max_value_l145_145460


namespace average_payment_correct_l145_145110

def first_segment_total (n₁ : ℕ) (p₁ : ℝ) : ℝ := n₁ * p₁
def second_segment_total (n₂ : ℕ) (p₁ p₂ : ℝ) : ℝ := n₂ * (p₁ + p₂)
def third_segment_total (n₃ : ℕ) (p₂ p₃ : ℝ) : ℝ := n₃ * (p₂ + p₃)
def fourth_segment_total (n₄ : ℕ) (p₃ p₄ : ℝ) : ℝ := n₄ * (p₃ + p₄)

def total_amount_paid 
  (n₁ n₂ n₃ n₄ : ℕ) (p₁ p₂ p₃ p₄ : ℝ) : ℝ := 
  first_segment_total n₁ p₁ + 
  second_segment_total n₂ p₁ p₂ + 
  third_segment_total n₃ p₂ p₃ + 
  fourth_segment_total n₄ p₃ p₄

def average_payment 
  (n₁ n₂ n₃ n₄ : ℕ) (p₁ p₂ p₃ p₄ : ℝ) (total_payment : ℝ) : ℝ :=
  total_payment / (n₁ + n₂ + n₃ + n₄)

theorem average_payment_correct :
  average_payment 16 24 32 32 420 75 50 100 (total_amount_paid 16 24 32 32 420 75 50 100) = 544.62 :=
by
  have h1 : first_segment_total 16 420 = 6720 := by sorry
  have h2 : second_segment_total 24 420 75 = 24 * 495 := by sorry
  have h3 : h2 = 11880 := by sorry
  have h4 : third_segment_total 32 495 50 = 32 * 545 := by sorry
  have h5 : h4 = 17440 := by sorry
  have h6 : fourth_segment_total 32 545 100 = 32 * 645 := by sorry
  have h7 : h6 = 20640 := by sorry
  have total_payment : total_amount_paid 16 24 32 32 420 75 50 100 = 6720 + 11880 + 17440 + 20640 := by sorry
  have h8 : total_payment = 56680 := by sorry
  have avg_payment : average_payment 16 24 32 32 420 75 50 100 56680 = 56680 / 104 := by sorry
  have h9 : 56680 / 104 = 544.62 := by sorry
  sorry

end average_payment_correct_l145_145110


namespace garden_plot_width_l145_145771

theorem garden_plot_width
  (length : ℕ)
  (swath_width : ℕ)
  (till_rate : ℕ)
  (time_minutes : ℕ)
  (total_area : ℕ)
  (time_seconds : ℕ := time_minutes * 60)
  (total_feet_tilled : ℕ := time_seconds / till_rate)
  (total_area_tilled : ℕ := total_feet_tilled * swath_width)
  (width : ℕ := total_area / length) :
  length = 120 ∧ swath_width = 2 ∧ till_rate = 2 ∧ time_minutes = 220 ∧ total_area = 13200 → width = 110 :=
by
  intros,
  -- proof will be added here
  sorry

end garden_plot_width_l145_145771


namespace quadratic_function_explicit_formula_l145_145063

theorem quadratic_function_explicit_formula :
  ∃ f : ℝ → ℝ, (∀ x, f(x) = -2 * x^2 + 4 * x + 11) ∧ f(3) = 5 ∧ f(-1) = 5 ∧ ∀ x, f(x) ≤ 13 :=
by
  sorry

end quadratic_function_explicit_formula_l145_145063


namespace centipede_and_earthworm_meeting_time_l145_145272

noncomputable def speed_centipede : ℚ := 5 / 3
noncomputable def speed_earthworm : ℚ := 5 / 2
noncomputable def initial_gap : ℚ := 20

theorem centipede_and_earthworm_meeting_time : 
  ∃ t : ℚ, (5 / 2) * t = initial_gap + (5 / 3) * t ∧ t = 24 := 
by
  sorry

end centipede_and_earthworm_meeting_time_l145_145272


namespace find_b_in_quadratic_l145_145015

theorem find_b_in_quadratic :
  ∃ (b : ℚ), ∃ (m : ℚ), 
  (b < 0) ∧ (m^2 = 1 / 8) ∧ 
  (∀ (x : ℚ), (x^2 + b * x + 1 / 4) = (x + m)^2 + 1 / 8) ∧ 
  b = - (Real.sqrt 2) / 2 :=
by 
  use [- (Real.sqrt 2) / 2, - (Real.sqrt 2) / 4]
  sorry

end find_b_in_quadratic_l145_145015


namespace volume_relation_l145_145637

-- Definitions for points and geometry structures
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

-- Volume function for Tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Given conditions
variable {A B C D D1 A1 B1 C1 : Point3D} 

-- D_1 is the centroid of triangle ABC
axiom centroid_D1 (A B C D1 : Point3D) : D1 = Point3D.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) ((A.z + B.z + C.z) / 3)

-- Line through A parallel to DD_1 intersects plane BCD at A1
axiom A1_condition (A B C D D1 A1 : Point3D) : sorry
-- Line through B parallel to DD_1 intersects plane ACD at B1
axiom B1_condition (A B C D D1 B1 : Point3D) : sorry
-- Line through C parallel to DD_1 intersects plane ABD at C1
axiom C1_condition (A B C D D1 C1 : Point3D) : sorry

-- Volume relation to be proven
theorem volume_relation (t1 t2 : Tetrahedron) (h : t1.A = A ∧ t1.B = B ∧ t1.C = C ∧ t1.D = D ∧
                                                t2.A = A1 ∧ t2.B = B1 ∧ t2.C = C1 ∧ t2.D = D1) :
  volume t1 = 2 * volume t2 := 
sorry

end volume_relation_l145_145637


namespace car_b_speed_l145_145304

def speed_of_car_b (Vb Va : ℝ) (tA tB : ℝ) (dist total_dist : ℝ) : Prop :=
  Va = 3 * Vb ∧ tA = 6 ∧ tB = 2 ∧ dist = 1000 ∧ total_dist = Va * tA + Vb * tB

theorem car_b_speed : ∃ Vb Va tA tB dist total_dist, speed_of_car_b Vb Va tA tB dist total_dist ∧ Vb = 50 :=
by
  sorry

end car_b_speed_l145_145304


namespace circle_equation_correct_l145_145166

theorem circle_equation_correct :
  (∃ M : ℝ × ℝ, M.1 * 2 + M.2 - 1 = 0 ∧
                (M.1 - 3)^2 + (M.2 - 0)^2 = 5 ∧ 
                (M.1 - 0)^2 + (M.2 - 1)^2 = 5) →
  ∃ h k r : ℝ, (h = 1) ∧ (k = -1) ∧ (r = sqrt 5) ∧ 
               (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 5) :=
begin
  sorry
end

end circle_equation_correct_l145_145166


namespace kyungsoo_choice_l145_145577

theorem kyungsoo_choice (pencil_choices : ℕ) (pen_choices : ℕ) (h1 : pencil_choices = 3) (h2 : pen_choices = 5) : 
  pencil_choices * pen_choices = 15 :=
by
  rw [h1, h2]
  norm_num
  sorry

end kyungsoo_choice_l145_145577


namespace abscissa_of_tangent_point_l145_145144

theorem abscissa_of_tangent_point (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = Real.exp x + a * Real.exp (-x))
  (h_odd : ∀ x, (D^[2] f x) = - (D^[2] f (-x)))
  (slope_cond : ∀ x, (D f x) = 3 / 2) : 
  ∃ x ∈ Set.Ioo (-Real.log 2) (Real.log 2), x = Real.log 2 :=
by
  sorry

end abscissa_of_tangent_point_l145_145144


namespace weight_loss_percentage_l145_145299

variables (W : ℝ) (x : ℝ)

def weight_loss_challenge :=
  W - W * x / 100 + W * 2 / 100 = W * 86.7 / 100

theorem weight_loss_percentage (h : weight_loss_challenge W x) : x = 15.3 :=
by sorry

end weight_loss_percentage_l145_145299


namespace tan_alpha_plus_beta_z_squared_plus_u_squared_plus_zu_l145_145477

variables (α β : ℝ)

-- Define the complex numbers z and u
def z := Complex.ofReal (Real.cos α) + Complex.i * (Real.sin α)
def u := Complex.ofReal (Real.cos β) + Complex.i * (Real.sin β)

-- Given condition: z + u = 4/5 + 3/5 i
def condition1 : Prop := (z + u) = Complex.mk (4/5) (3/5)

-- First question: Prove tan(α + β) = 24/7
theorem tan_alpha_plus_beta (h : condition1) : Real.tan (α + β) = 24 / 7 :=
sorry

-- Second question: Prove z^2 + u^2 + z*u = 0
theorem z_squared_plus_u_squared_plus_zu (h : condition1) : z^2 + u^2 + z * u = 0 :=
sorry

end tan_alpha_plus_beta_z_squared_plus_u_squared_plus_zu_l145_145477


namespace ceil_neg_sqrt_eight_l145_145794

theorem ceil_neg_sqrt_eight : Real.ceil (-Real.sqrt 8) = -2 := 
by
  sorry

end ceil_neg_sqrt_eight_l145_145794


namespace center_in_third_quadrant_equiv_l145_145901

noncomputable def circle_center_condition (k : ℝ) : Prop :=
  let centerX := -k in
  let centerY := -2 in
  k > 4

theorem center_in_third_quadrant_equiv (k : ℝ) :
  (¬(k ≥ 4) ∨ ¬(k ∈ Ioi (4 : ℝ))) :=
by
  sorry

end center_in_third_quadrant_equiv_l145_145901


namespace at_least_one_zero_l145_145995

theorem at_least_one_zero (a b : ℝ) : (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0) := by
  intro h
  have h' : ¬ ((a ≠ 0) ∧ (b ≠ 0)) := h
  sorry

end at_least_one_zero_l145_145995


namespace reflection_y_axis_l145_145409

open Matrix

def reflection_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

theorem reflection_y_axis (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M ⬝ ![![1, 0], ![0, 1]]) = reflection_y_axis_matrix :=
by sorry

end reflection_y_axis_l145_145409


namespace sequence_sum_formula_l145_145824

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n / (2 + 3 * a n)

noncomputable def sequence_sum (n : ℕ) : ℝ :=
(finset.range n).sum (λ k, 1 / a k)

theorem sequence_sum_formula (n : ℕ) : 
  sequence_sum n = 2^(n+2) - 4 - 3*n := sorry

end sequence_sum_formula_l145_145824


namespace net_gain_difference_l145_145332

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l145_145332


namespace find_y_coordinate_C_l145_145986

structure Point :=
(x : ℝ)
(y : ℝ)

def area_square (side: ℝ) : ℝ :=
  side * side

def area_triangle (base height: ℝ) : ℝ :=
  (1 / 2) * base * height

def area_pentagon (A B C D E: Point) (area_sq: ℝ) (area_total: ℝ) : Prop :=
  area_sq == (area_square 6) ∧
  area_total == 60 ∧
  ∃ yC, area_triangle 6 (yC - B.y) == area_total - area_sq ->
  ∃ yC, yC == 14

theorem find_y_coordinate_C : 
  ∀ (A B E D C: Point),
  A.x == 0 ∧ A.y == 0 ∧
  B.x == 0 ∧ B.y == 6 ∧
  E.x == 6 ∧ E.y == 0 ∧
  D.x == 6 ∧ D.y == 6 ∧
  area_pentagon A B C D E (area_square 6) 60 ->
  C.y == 14 :=
by
  intros
  sorry

end find_y_coordinate_C_l145_145986


namespace linear_function_intersects_x_axis_at_two_units_l145_145657

theorem linear_function_intersects_x_axis_at_two_units (k : ℝ) :
  (∃ x : ℝ, y = k * x + 2 ∧ y = 0 ∧ |x| = 2) ↔ k = 1 ∨ k = -1 :=
by
  sorry

end linear_function_intersects_x_axis_at_two_units_l145_145657


namespace lee_cookie_problem_l145_145131

theorem lee_cookie_problem :
  (∀ (f1 s1 f2 s2 x : ℕ), 0 < f1 → 0 < s1 → 0 < f2 → 0 < s2 →
    (18 : ℕ) = (2 * 3) →
    (18 : ℕ) = (1 * 1.5 * 18).toNat →
    ((18 / f1 : ℕ) = (x / f2 : ℕ) ∧ (18 / s1 : ℕ) = (x / s2 : ℕ)) →
    x ≅ 27) :=
by
  intros f1 s1 f2 s2 x h₁ h₂ h₃ h₄ hfl hs
  rw [nat.cast_mul, nat.cast_div, nat.cast_div] at *
  sorry

end lee_cookie_problem_l145_145131


namespace total_turtles_30_l145_145688

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)
variable (total_turtles : ℕ)

def Kris_turtle_condition : Prop :=
  Kris_turtles = Kristen_turtles / 4

def Trey_turtle_condition : Prop :=
  Trey_turtles = 5 * Kris_turtles

def total_turtle_condition : Prop :=
  total_turtles = Kristen_turtles + Kris_turtles + Trey_turtles

theorem total_turtles_30
  (h1 : Kristen_turtles = 12)
  (h2 : Kris_turtle_condition)
  (h3 : Trey_turtle_condition)
  (h4 : total_turtle_condition) :
  total_turtles = 30 :=
by
  sorry

end total_turtles_30_l145_145688


namespace max_value_of_function_max_y_value_on_domain_l145_145448

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logBase 3 x
def y (x : ℝ) : ℝ := (f x) ^ 2 + f (x ^ 2)

theorem max_value_of_function :
  ∀ x, x ∈ Set.Icc 1 9 → ∃ t, t = Real.logBase 3 x ∧ 0 ≤ t ∧ t ≤ 1 :=
begin
  intros x hx,
  use Real.logBase 3 x,
  split,
  { refl },
  split,
  { apply Real.logBase_pos,
    exact hx.1 },
  { apply Real.logBase_le_iff_le_3.logBase,
    exacts [hx.2, hx.1] }
end

theorem max_y_value_on_domain :
  ∃ x, x ∈ Set.Icc 1 9 ∧ y x = 13 :=
begin
  use 3,
  split,
  { split,
    { linarith },
    { linarith } },
  { unfold y,
    unfold f,
    calc (2 + Real.logBase 3 3) ^ 2 + 2 + Real.logBase 3 (3 ^ 2)
      = (2 + 1) ^ 2 + 2 + Real.logBase 3 9 : by rw [Real.logBase_pow]
  ... = 3 ^ 2 + 2 + Real.logBase 3 9 : by rw [Real.logBase_self]
  ... = 9 + 2 + 1 : by rw [Real.logBase_3_of_9]
  ... = 13 : by linarith }
end

end max_value_of_function_max_y_value_on_domain_l145_145448


namespace euclidean_algorithm_gcd_gcd_linear_combination_l145_145307

theorem euclidean_algorithm_gcd
  (m0 m1 : ℤ) (h1 : 0 < m1) (h2 : m1 ≤ m0) :
  ∃ (k : ℕ) (a : Fin k → ℤ) (m : Fin k → ℤ),
    (∀ i : Fin (k-1), m i > m (i+1)) ∧ 
    m (0 : Fin k) = m0 ∧ m (1 : Fin k) = m1 ∧
    m (k-1) = m k * a k ∧
    (∀ i : Fin (k-1), m i = m (i+1) * a i + m (i+2)) ∧
    m k = Int.gcd m0 m1 :=
  sorry

theorem gcd_linear_combination
  (m0 m1 : ℤ) (h1 : 0 < m1) (h2 : m1 ≤ m0) :
  ∃ (d : ℤ) (k : ℕ) (a : Fin k → ℤ) (m u v : Fin k → ℤ),
    (∀ s : Fin (k-1), m_s * u_s + m_(s+1) * v_s = d) ∧
    d = Int.gcd m0 m1 ∧
    ∃ u v : ℤ, m0 * u + m1 * v = d :=
  sorry

end euclidean_algorithm_gcd_gcd_linear_combination_l145_145307


namespace scale_readings_poles_greater_l145_145232

-- Define the necessary quantities and conditions
variable (m : ℝ) -- mass of the object
variable (ω : ℝ) -- angular velocity of Earth's rotation
variable (R_e : ℝ) -- radius of the Earth at the equator
variable (g_e : ℝ) -- gravitational acceleration at the equator
variable (g_p : ℝ) -- gravitational acceleration at the poles
variable (F_c : ℝ) -- centrifugal force at the equator
variable (F_g_e : ℝ) -- gravitational force at the equator
variable (F_g_p : ℝ) -- gravitational force at the poles
variable (W_e : ℝ) -- apparent weight at the equator
variable (W_p : ℝ) -- apparent weight at the poles

-- Establish conditions
axiom centrifugal_definition : F_c = m * ω^2 * R_e
axiom gravitational_force_equator : F_g_e = m * g_e
axiom apparent_weight_equator : W_e = F_g_e - F_c
axiom no_centrifugal_force_poles : F_c = 0
axiom gravitational_force_poles : F_g_p = m * g_p
axiom apparent_weight_poles : W_p = F_g_p
axiom gravity_comparison : g_p > g_e

-- Theorem: The readings on spring scales at the poles will be greater than the readings at the equator
theorem scale_readings_poles_greater : W_p > W_e := 
sorry

end scale_readings_poles_greater_l145_145232


namespace solve_inequality_l145_145636

def numerator (x : ℝ) : ℝ := x ^ 2 - 4 * x + 3
def denominator (x : ℝ) : ℝ := (x - 2) ^ 2

theorem solve_inequality : { x : ℝ | numerator x / denominator x < 0 } = { x : ℝ | 1 < x ∧ x < 3 } :=
by
  sorry

end solve_inequality_l145_145636


namespace smallest_nine_l145_145431

def satisfies_conditions (n : ℕ) (x : Fin n → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ n) ∧
  (Finset.univ.sum (λ i => x i) = n * (n + 1) / 2) ∧
  (Finset.univ.prod (λ i => x i) = Nat.factorial n) ∧
  (Multiset.of_finset (Finset.univ.image x) ≠ Multiset.range (n + 1))

def has_solution (n : ℕ) : Prop :=
  ∃ x : Fin n → ℕ, satisfies_conditions n x

theorem smallest_nine : ∃ n, has_solution n ∧ (∀ m, m < n → ¬ has_solution m) :=
  sorry

end smallest_nine_l145_145431


namespace medians_perpendicular_iff_l145_145759

-- Definitions from the conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- A triangle
variables [Nonempty A] [Nonempty B] [Nonempty C]

-- Medians
variables (BD CE : Type) [MetricSpace BD] [MetricSpace CE]

-- Midpoints of sides AC and AB
variables (AC AB : Type) [MetricSpace AC] [MetricSpace AB]
variables (mid_AC : AC → D) (mid_AB : AB → E)

-- Proof statement
theorem medians_perpendicular_iff (BD CE : Type) [MetricSpace BD] [MetricSpace CE]
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (b c a : ℝ) :
  (b^2 + c^2 = 5 * a^2) ↔
  (medians BD CE A B C ∧ mid_AC AC = D ∧ mid_AB AB = E) :=
begin
  sorry
end

end medians_perpendicular_iff_l145_145759


namespace good_triplet_exists_l145_145324

-- Assume n is at least 3 for the existence of a triplet
theorem good_triplet_exists (n : ℕ) (a : Fin n → ℕ) (h_n_ge_3 : n ≥ 3) :
  (∀ i : Fin n, 2 * a i < ∑ j, a j) →
  ∃ (i j k : Fin n), i < j ∧ j < k ∧ a k < a i + a j :=
by
  sorry

end good_triplet_exists_l145_145324


namespace probability_odd_divisor_of_15_fact_l145_145372

theorem probability_odd_divisor_of_15_fact :
  let stirling : ℕ := list.prod (list.map factorial [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
  let total_divisors := stirling.divisor_count
  let odd_divisors := (3^6 * 5^3 * 7 * 11 * 13).divisor_count
  odd_divisors / total_divisors = 1 / 6 :=
by
  let stirling := 1_307_674_368_000
  let total_divisors := stirling.divisor_count
  let odd_divisors := (3^6 * 5^3 * 7 * 11 * 13).divisor_count
  show odd_divisors / total_divisors = 1 / 6
  sorry

end probability_odd_divisor_of_15_fact_l145_145372


namespace sum_fraction_nonnegative_le_one_l145_145594

theorem sum_fraction_nonnegative_le_one 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 2) :
  a * b / (c^2 + 1) + b * c / (a^2 + 1) + c * a / (b^2 + 1) ≤ 1 :=
sorry

end sum_fraction_nonnegative_le_one_l145_145594


namespace students_unable_to_partner_l145_145274

theorem students_unable_to_partner
    (males_class1 : ℕ) (females_class1 : ℕ)
    (males_class2 : ℕ) (females_class2 : ℕ)
    (males_class3 : ℕ) (females_class3 : ℕ)
    (h1 : males_class1 = 17) (h2 : females_class1 = 13)
    (h3 : males_class2 = 14) (h4 : females_class2 = 18)
    (h5 : males_class3 = 15) (h6 : females_class3 = 17) :
    |(females_class1 + females_class2 + females_class3) - (males_class1 + males_class2 + males_class3)| = 2 :=
by 
  sorry

end students_unable_to_partner_l145_145274


namespace find_n_times_s_l145_145141

-- Defining the set S as the positive real numbers
def S := {x : ℝ // 0 < x}

-- Defining the function f with the given condition
def f (x : S) (y : S) : ℝ :=
  (f x) * (f y) = (f (x * y)) + 2005 * ((1 / x) + (1 / y) + 2004)

-- The final theorem stating the result n * s 
theorem find_n_times_s (f : S → ℝ) (condition : ∀ x y : S, f x * f y = f (x * y) + 2005 * (1 / x + 1 / y + 2004)) :
  let n := 1 in
  let s := f ⟨2, by norm_num⟩ in
  n * s = 4011 / 2 :=
begin
  sorry
end

end find_n_times_s_l145_145141


namespace tiles_needed_l145_145089

/--
A rectangular swimming pool is 20m long, 8m wide, and 1.5m deep. 
Each tile used to cover the pool has a side length of 2dm. 
We need to prove the number of tiles required to cover the bottom and all four sides of the pool.
-/
theorem tiles_needed (pool_length pool_width pool_depth : ℝ) (tile_side : ℝ) 
  (h1 : pool_length = 20) (h2 : pool_width = 8) (h3 : pool_depth = 1.5) 
  (h4 : tile_side = 0.2) : 
  (pool_length * pool_width + 2 * pool_length * pool_depth + 2 * pool_width * pool_depth) / (tile_side * tile_side) = 6100 :=
by
  sorry

end tiles_needed_l145_145089


namespace domain_lg_function_l145_145650

noncomputable def domain_of_lg_function : set ℝ :=
  {x : ℝ | x > 1}

theorem domain_lg_function (x : ℝ) : 
  domain_of_lg_function = {x | x > 1} :=
sorry

end domain_lg_function_l145_145650


namespace evaluate_x2_plus_y2_plus_z2_l145_145234

theorem evaluate_x2_plus_y2_plus_z2 (x y z : ℤ) 
  (h1 : x^2 * y + y^2 * z + z^2 * x = 2186)
  (h2 : x * y^2 + y * z^2 + z * x^2 = 2188) 
  : x^2 + y^2 + z^2 = 245 := 
sorry

end evaluate_x2_plus_y2_plus_z2_l145_145234


namespace maximize_profit_max_profit_at_three_l145_145739

def profit_func (x : ℝ) (m : ℕ) : ℝ :=
  x * (1.5 * ((8 + 16 * x) / x)) - (8 + 16 * x + m)

-- Use the given conditions:
def sales_volume (m : ℕ) : ℝ := 3 - (2 / (m + 1))

-- Profit function
noncomputable def y (m : ℕ) : ℝ :=
  28 - (16 / (m + 1)) - m

theorem maximize_profit : ∀ m : ℕ, y m <= 21 :=
begin
  intros m,
  sorry -- proof will go here
end

theorem max_profit_at_three : y 3 = 21 :=
by sorry -- proof will go here

end maximize_profit_max_profit_at_three_l145_145739


namespace find_x_squared_plus_y_squared_l145_145893

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 75) : x^2 + y^2 = 3205 / 121 :=
by
  sorry

end find_x_squared_plus_y_squared_l145_145893


namespace trapezoid_area_correct_l145_145702

noncomputable def trapezoid_area : ℝ := 
  let base1 : ℝ := 8
  let base2 : ℝ := 4
  let height : ℝ := 2
  (1 / 2) * (base1 + base2) * height

theorem trapezoid_area_correct :
  trapezoid_area = 12.0 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end trapezoid_area_correct_l145_145702


namespace living_room_floor_area_l145_145717

variable (Area_carpet : ℝ) (l w : ℝ) (percent_cover : ℝ) (Area_living_room : ℝ)

-- Conditions
def carpet_area (l w : ℝ) : ℝ := l * w
def living_room_area (percent_cover Area_carpet : ℝ) : ℝ := Area_carpet / percent_cover

theorem living_room_floor_area :
  l = 6 →
  w = 12 →
  percent_cover = 0.20 →
  carpet_area l w = 72 →
  living_room_area percent_cover (carpet_area l w) = 360 :=
by
  intros hl hw hp hc
  rw [hl, hw, hp, hc]
  simp
  sorry

end living_room_floor_area_l145_145717


namespace longer_side_of_rectangle_l145_145542

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
noncomputable def rectangle_area : ℝ := 3 * circle_area
noncomputable def shorter_side : ℝ := 2 * circle_radius

theorem longer_side_of_rectangle :
    ∃ (l : ℝ), l = rectangle_area / shorter_side ∧ l = 9 * Real.pi :=
by
  sorry

end longer_side_of_rectangle_l145_145542


namespace trisect_angle_l145_145578

theorem trisect_angle
  (A B C P M K : Type)
  (ABC_isosceles : AC = BC)
  (P_condition1: ∠PAB = ∠PBC)
  (P_condition2: ∠PAC = ∠PCB)
  (M_midpoint : M = midpoint A B)
  (K_intersection : K = intersection BP AC) :
  trisect_angle (∠M P C) AP PK := 
sorry

end trisect_angle_l145_145578


namespace beetle_can_always_return_l145_145114

variables {Cell : Type} [Inhabited Cell]

def Door (c1 c2 : Cell) : Prop := sorry -- Define adjacent cells

def opens (from to : Cell) (opened_doors : Set (Cell × Cell)) : Prop :=
  (from, to) ∈ opened_doors

noncomputable def beetle_can_return_to_start 
  (start : Cell) 
  (opened_doors : Set (Cell × Cell)) 
  (can_open : ∀ c1 c2 : Cell, Door c1 c2 → opens c1 c2 opened_doors) 
  (one_way : ∀ c1 c2 : Cell, opens c1 c2 opened_doors → ¬ opens c2 c1 opened_doors)
  : Prop := 
  ∀ current : Cell, ∃ path_back : List Cell, 
  path_back.head = current ∧ path_back.last = some start ∧ 
  ∀ (from to : Cell), (from, to) ∈ path_back.zip path_back.tail → opens from to opened_doors 

theorem beetle_can_always_return : 
  ∀ (start : Cell)
    (opened_doors : Set (Cell × Cell))
    (can_open : ∀ c1 c2 : Cell, Door c1 c2 → opens c1 c2 opened_doors)
    (one_way : ∀ c1 c2 : Cell, opens c1 c2 opened_doors → ¬ opens c2 c1 opened_doors),
  beetle_can_return_to_start start opened_doors can_open one_way := 
sorry

end beetle_can_always_return_l145_145114


namespace car_sales_total_l145_145724

theorem car_sales_total (a b c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : c = 27):
  a + b + c = 57 :=
by
  repeat {rwa [h1, h2, h3]}
  sorry

end car_sales_total_l145_145724


namespace time_to_fill_with_leak_l145_145735

-- Define the conditions
def fill_time_without_leak := 8 -- hours
def leak_empty_time := 72 -- hours

-- Define the effective rate and solve the problem
theorem time_to_fill_with_leak : 
  (1 / fill_time_without_leak - 1 / leak_empty_time)⁻¹ = 9 :=
by
  sorry

end time_to_fill_with_leak_l145_145735


namespace binomial_coefficient_x3_term_l145_145155

theorem binomial_coefficient_x3_term :
  let a := ∫ x in 0..π, (Real.cos x - Real.sin x) in
  a = -2 →
  let expr := (fun x : ℝ => (x^2 + a / x)^6) in
  let coeff_x3 := (-1)^3 * 2^3 * Nat.choose 6 3 in
  coeff_x3 = -160 :=
by
  intro a ha
  have h : a = -2 := ha
  let expr := (fun x : ℝ => (x^2 + a / x)^6)
  let coeff_x3 := (-1)^3 * 2^3 * Nat.choose 6 3
  show coeff_x3 = -160
  sorry

end binomial_coefficient_x3_term_l145_145155


namespace logan_list_count_l145_145177

theorem logan_list_count : 
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    ∃ n, n = 871 ∧ 
        ∀ k, (k * 30 ≥ smallest_square_multiple ∧ k * 30 ≤ smallest_cube_multiple) ↔ (30 ≤ k ∧ k ≤ 900) :=
by
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    use 871
    sorry

end logan_list_count_l145_145177


namespace petya_numbers_board_l145_145988

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end petya_numbers_board_l145_145988


namespace total_cost_calc_l145_145370

variable (a b : ℝ)

def total_cost (a b : ℝ) := 2 * a + 3 * b

theorem total_cost_calc (a b : ℝ) : total_cost a b = 2 * a + 3 * b := by
  sorry

end total_cost_calc_l145_145370


namespace trees_to_cm_cm_to_trees_l145_145185

-- Define the conditions
def tree_density : ℕ := 40  -- 1 cm represents 40 trees

-- Prove the number of centimeters needed to represent 120 trees is 3 cm
theorem trees_to_cm (trees : ℕ) (density : ℕ) : trees = 120 ∧ density = tree_density → 
  (trees / density) = 3 :=
by
  intros h
  cases h with ht hd
  rw [ht, hd]
  exact (div_eq_of_eq_mul_right (by decide) (by decide)).symm

-- Prove that a bar of length 3.5 cm represents 140 trees
theorem cm_to_trees (length : ℝ) (density : ℕ) : length = 3.5 ∧ density = tree_density → 
  (length * density) = 140 :=
by
  intros h
  cases h with hl hd
  rw [hl, hd]
  norm_num
  exact (mul_eq_of_eq_div_right (by decide) (by decide)).symm


end trees_to_cm_cm_to_trees_l145_145185


namespace P_investment_calculation_l145_145603

variable {P_investment : ℝ}
variable (Q_investment : ℝ := 36000)
variable (total_profit : ℝ := 18000)
variable (Q_profit : ℝ := 6001.89)

def P_profit : ℝ := total_profit - Q_profit

theorem P_investment_calculation :
  P_investment = (P_profit * Q_investment) / Q_profit :=
by
  sorry

end P_investment_calculation_l145_145603


namespace part1_hyperbola_equation_part2_foci_and_cos_angle_part3_eccentricity_range_l145_145075

noncomputable def hyperbola_equation (a b : ℝ) (e : ℝ) (p : ℝ × ℝ) : Prop :=
  let E := ∀ x y, (x / a)^2 - (y / b)^2 = 1
  E p.1 p.2

noncomputable def foci_distance_and_cos (a : ℝ) (foci_dist : ℝ) (dist_mf1 : ℝ) : Prop :=
  let b := sqrt (foci_dist * foci_dist - a * a)
  let E := ∀ x y, (x / a)^2 - (y / b)^2 = 1
  let foci := (sqrt (a*a + b*b), 0)
  let cos_angle := (8^2 + 4^2 - (2 * sqrt (7))^2) / (2 * 8 * 4)
  cos_angle = 13 / 16

noncomputable def eccentricity_range (e : ℝ) (k m : ℝ) (angle_AOB : ℝ) : Prop :=
  let circle := ∀ x y, x^2 + y^2 = 4
  let tangent_line := ∀ x, k * x + m = 0
  π / 2 < e ∧ e < ∞

-- Example math proof problems in Lean 4 statement:
theorem part1_hyperbola_equation (a b : ℝ) (e := sqrt 2) (p := (sqrt 2, 1)) :
  a > 0 → b > 0 → hyperbola_equation a b e p → ∃ x y, x^2 - y^2 = 1 :=
by sorry

theorem part2_foci_and_cos_angle (a := 2) (foci_dist := sqrt 3) (dist_mf1 := 8) :
  foci_distance_and_cos a foci_dist dist_mf1 → cos_angle = 13 / 16 :=
by sorry

theorem part3_eccentricity_range (k m : ℝ) (e : ℝ) (angle_AOB := π / 2) :
  eccentricity_range e k m angle_AOB → (\sqrt 2 < e ∧ e < ∞) :=
by sorry

end part1_hyperbola_equation_part2_foci_and_cos_angle_part3_eccentricity_range_l145_145075


namespace general_formula_l145_145120

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * (a n) + 1

theorem general_formula (a : ℕ → ℕ) (h : sequence a) : 
  ∀ n, a n = 2^n - 1 :=
by
  intro n
  sorry

end general_formula_l145_145120


namespace alice_profit_l145_145355

noncomputable def total_bracelets : ℕ := 52
noncomputable def cost_of_materials : ℝ := 3.00
noncomputable def bracelets_given_away : ℕ := 8
noncomputable def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_remaining := total_bracelets - bracelets_given_away;
      total_revenue := bracelets_remaining * price_per_bracelet;
      profit := total_revenue - cost_of_materials
  in profit = 8.00 := 
by 
  sorry

end alice_profit_l145_145355


namespace separator_theorem_l145_145826

noncomputable def separator_count (A B C D E : Point) : Int :=
  sorry

theorem separator_theorem (h1 : ¬ collinear A B C)
                         (h2 : ¬ collinear A B D)
                         (h3 : ¬ collinear A B E)
                         (h4 : ¬ collinear A C D)
                         (h5 : ¬ collinear A C E)
                         (h6 : ¬ collinear A D E)
                         (h7 : ¬ collinear B C D)
                         (h8 : ¬ collinear B C E)
                         (h9 : ¬ collinear B D E)
                         (h10 : ¬ collinear C D E)
                         (h11 : ¬ cocyclic A B C D)
                         (h12 : ¬ cocyclic A B C E)
                         (h13 : ¬ cocyclic A B D E)
                         (h14 : ¬ cocyclic A C D E)
                         (h15 : ¬ cocyclic B C D E) : 
  separator_count A B C D E = 30 := 
  sorry

end separator_theorem_l145_145826


namespace circle_equation_l145_145161

theorem circle_equation (M : ℝ × ℝ) :
  (∃ a : ℝ, M = (a, 1 - 2 * a) ∧ 2 * M.1 + M.2 - 1 = 0) ∧
  (distance M (3, 0) = distance M (0, 1)) →
  (∃ r : ℝ, (x - 1)^2 + (y + 1)^2 = r^2) :=
begin
  sorry
end

end circle_equation_l145_145161


namespace tourist_distance_l145_145348

theorem tourist_distance :
  let east_miles := 24
  let south_miles := 7
  let west_miles := 15
  let north_miles := 3
  let net_east := east_miles - west_miles
  let net_south := south_miles - north_miles
  let distance := Math.sqrt (net_east^2 + net_south^2)
  distance = Math.sqrt 97 := 
by
  sorry

end tourist_distance_l145_145348


namespace universal_set_eq_l145_145080

def A : Set ℕ := {1, 3, 5}
def complement_U_A : Set ℕ := {2, 4, 6}

theorem universal_set_eq : ∀ (U : Set ℕ), (U = A ∪ complement_U_A) → (U = {1, 2, 3, 4, 5, 6}) :=
by
  intro U
  intro hU
  rw [hU]
  rfl
  sorry

end universal_set_eq_l145_145080


namespace sin_cos_sum_value_l145_145051

theorem sin_cos_sum_value (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) :
  (sin θ + cos θ) = 4 / 3 :=
sorry

end sin_cos_sum_value_l145_145051


namespace probability_b_minus_a_in_A_union_B_l145_145867
-- Import the necessary Lean library

-- Define the sets A and B
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {-1, 0, 1, 2}

-- Define the combined set A ∪ B
def A_union_B : Set ℤ := A ∪ B

-- Define the proof statement to show the probability
theorem probability_b_minus_a_in_A_union_B : 
  (finset.card ((finset.product {a ∈ A} {b ∈ B}).filter (λ (p : ℤ × ℤ), p.2 - p.1 ∈ A_union_B))).to_rat / (finset.card (finset.product {a ∈ A} {b ∈ B})).to_rat = 3 / 4 :=
by {
  -- The actual proof is left as an exercise to the reader.
  sorry
}

end probability_b_minus_a_in_A_union_B_l145_145867


namespace pension_value_l145_145746

-- Conditions
variables (k x c p q : ℝ)
hypothesis h1 : k * Real.sqrt (x + c) = k * Real.sqrt x + 3 * p
hypothesis h2 : k * Real.sqrt (x + 2 * c) = k * Real.sqrt x + 4 * q

-- Conclusion
theorem pension_value : k * Real.sqrt x = (16 * q ^ 2 - 18 * p ^ 2) / (12 * p - 8 * q) :=
sorry

end pension_value_l145_145746


namespace quadratic_roots_d_l145_145670

theorem quadratic_roots_d (d : ℝ) : 
  (∀ x, x^2 + 6 * x + d = 0 → (x = (-6 + real.sqrt d) / 2 ∨ x = (-6 - real.sqrt d) / 2)) →
  d = 36 / 5 :=
  sorry

end quadratic_roots_d_l145_145670


namespace first_problem_number_solved_l145_145765

theorem first_problem_number_solved (last_problem_number number_of_problems_solved : ℕ) 
  (h_last : last_problem_number = 125) 
  (h_solved : number_of_problems_solved = 48) :
  (last_problem_number - number_of_problems_solved) + 1 = 78 :=
by
  rw [h_last, h_solved]
  -- Perform the arithmetic to verify the result
  sorry

end first_problem_number_solved_l145_145765


namespace totalHighlighters_l145_145109

-- Define the number of each type of highlighter
def pinkHighlighters : ℕ := 10
def yellowHighlighters : ℕ := 15
def blueHighlighters : ℕ := 8

-- State the theorem to prove
theorem totalHighlighters :
  pinkHighlighters + yellowHighlighters + blueHighlighters = 33 :=
by
  -- Proof to be filled
  sorry

end totalHighlighters_l145_145109


namespace sum_of_possible_values_l145_145347

theorem sum_of_possible_values (n : ℕ) (h1 : 4 < n) (h2 : n < 18) :
  ∑ i in Finset.Ico 5 18, (i : ℕ) = 143 :=
by
  sorry

end sum_of_possible_values_l145_145347


namespace petya_higher_chance_of_winning_l145_145213

/-- Petya gives Vasya two boxes of candies. Each box contains both chocolate candies and caramel candies.
    In total, there are 25 candies in both boxes. Vasya takes one candy from each box.
    If both candies turn out to be chocolate, Vasya wins. Otherwise, Petya wins.
    The probability that Vasya will get two caramels is 0.54.
    Prove that Petya has a higher chance of winning. --/
theorem petya_higher_chance_of_winning
  (total_candies : ℕ)
  (total_candies = 25 : Prop) 
  (prob_two_caramels : ℝ)
  (prob_two_caramels = 0.54 : Prop) :
  0.46 < 0.54 := 
by 
  sorry

end petya_higher_chance_of_winning_l145_145213


namespace stratified_sampling_first_grade_selection_l145_145329

theorem stratified_sampling_first_grade_selection
  (total_students : ℕ)
  (students_grade1 : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 2000)
  (h_grade1 : students_grade1 = 400)
  (h_sample : sample_size = 200) :
  sample_size * students_grade1 / total_students = 40 := by
  sorry

end stratified_sampling_first_grade_selection_l145_145329


namespace find_inheritance_l145_145940

variable (inheritance : ℝ)
variable (federal_tax_rate : ℝ) (state_tax_rate : ℝ) (total_taxes_paid : ℝ)

-- Define the conditions
def conditions := 
  federal_tax_rate = 0.25 ∧
  state_tax_rate = 0.12 ∧
  total_taxes_paid = 12000 ∧
  total_taxes_paid = (federal_tax_rate * inheritance) + (state_tax_rate * (inheritance - federal_tax_rate * inheritance))

-- The statement of the problem: proving the inheritance amount
theorem find_inheritance (h : conditions) : inheritance ≈ 35294.12 :=
sorry

end find_inheritance_l145_145940


namespace least_value_N_l145_145753

def lcm (a b : ℕ) : ℕ := sorry  -- Model lcm as a dummy definition for now

theorem least_value_N 
  (N : ℕ) 
  (a := 1125) 
  (b := 2925) 
  (c := 1725) 
  (h_linked: lcm a b = lcm c b): 
  N = 2875 := 
sorry

end least_value_N_l145_145753


namespace surface_area_increase_l145_145728

def cube_dimensions : ℝ × ℝ × ℝ := (10, 10, 10)

def number_of_cuts := 3

def initial_surface_area (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  6 * (length * width)

def increase_in_surface_area (cuts : ℕ) (length : ℝ) (width : ℝ) : ℝ :=
  cuts * 2 * (length * width)

theorem surface_area_increase : 
  initial_surface_area 10 10 10 + increase_in_surface_area 3 10 10 = 
  initial_surface_area 10 10 10 + 600 :=
by
  sorry

end surface_area_increase_l145_145728


namespace find_slope_of_line_intersecting_parabola_l145_145859

noncomputable def slope_of_line_intersecting_parabola
  (p : ℝ) 
  (A B : ℝ × ℝ)
  (F : ℝ × ℝ := (p/2, 0)) 
  (line_slope : ℝ) : Prop :=
  let y := (x : ℝ) => line_slope * (x - p / 2) in
  A.1 * A.2 = 2 * p * A.1 ∧ B.1 * B.2 = 2 * p * B.1 ∧ 
  A.2 + B.2 = 2 * p / line_slope ∧ A.2 * B.2 = -p^2 ∧ 
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 3 * ((B.1 - F.1)^2 + (B.2 - F.2)^2) ∧ 
  line_slope = sqrt(3) ∨ line_slope = -sqrt(3)

theorem find_slope_of_line_intersecting_parabola (p : ℝ) :
  ∃ k : ℝ, slope_of_line_intersecting_parabola p A B k :=
begin
  sorry -- proof is not required here
end

end find_slope_of_line_intersecting_parabola_l145_145859


namespace cos_x_eq_fraction_l145_145090

theorem cos_x_eq_fraction (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x ∧ x < π / 2) (h4 : Real.cot x = (a^2 - b^2) / (2 * a * b)) :
  Real.cos x = (a^2 - b^2) / (a^2 + b^2) :=
by
  sorry

end cos_x_eq_fraction_l145_145090


namespace magnitude_difference_l145_145451

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry

-- Conditions
axiom mag_a : ∥a∥ = 2
axiom mag_b : ∥b∥ = 3
axiom orthogonal_ab : IsOrtho a b

-- Theorem statement
theorem magnitude_difference : ∥b - a∥ = Real.sqrt 13 := 
by
  sorry

end magnitude_difference_l145_145451


namespace first_candidate_more_gain_l145_145339

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l145_145339


namespace triangle_is_isosceles_l145_145558

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC_sum : A + B + C = π) 
  (cos_rule : a * Real.cos B + b * Real.cos A = a) :
  a = c :=
by
  sorry

end triangle_is_isosceles_l145_145558


namespace amy_muffins_first_day_l145_145761

theorem amy_muffins_first_day (x : ℕ) :
  (∃ (x : ℕ), 5 * x + 10 = 15) → x = 1 :=
by
  assume h,
  obtain ⟨x', hx'⟩ := h,
  have eqn : 5 * x' + 10 = 15 := hx',
  have solve_x : x' = 1,
  sorry

example : ∃ (x : ℕ), 5 * x + 10 = 15 := begin
  use 1,
  norm_num,
end

end amy_muffins_first_day_l145_145761


namespace distance_from_M0_to_plane_l145_145402

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def Plane (P1 P2 P3 : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  let a := (P2.y - P1.y) * (P3.z - P1.z) - (P2.z - P1.z) * (P3.y - P1.y)
  let b := (P2.z - P1.z) * (P3.x - P1.x) - (P2.x - P1.x) * (P3.z - P1.z)
  let c := (P2.x - P1.x) * (P3.y - P1.y) - (P2.y - P1.y) * (P3.x - P1.x)
  let d := -(a * P1.x + b * P1.y + c * P1.z)
  (a, b, c, d)

def distance_to_plane (P0 : Point3D) (plane : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  let (a, b, c, d) := plane
  (abs (a * P0.x + b * P0.y + c * P0.z + d)) / (sqrt (a^2 + b^2 + c^2))

def M0 := { x := -7, y := 0, z := -1 }
def M1 := { x := -3, y := -1, z := 1 }
def M2 := { x := -9, y := 1, z := -2 }
def M3 := { x := 3, y := -5, z := 4 }

theorem distance_from_M0_to_plane : distance_to_plane M0 (Plane M1 M2 M3) = 0 := by
  sorry

end distance_from_M0_to_plane_l145_145402


namespace irrational_numbers_in_set_l145_145554

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def given_set : List ℝ := [Real.pi / 2, 22 / 7, 0.1414, Real.cbrt 9, Real.sqrt (1 / 2), -5 / 2, Real.sqrt 2 - 1, 0.10100010000100001000010000100001]

theorem irrational_numbers_in_set : (given_set.filter is_irrational).length = 5 := sorry

end irrational_numbers_in_set_l145_145554


namespace Linda_outfits_l145_145175

theorem Linda_outfits (skirts blouses shoes : ℕ) 
  (hskirts : skirts = 5) 
  (hblouses : blouses = 8) 
  (hshoes : shoes = 2) :
  skirts * blouses * shoes = 80 := by
  -- We provide the proof here
  sorry

end Linda_outfits_l145_145175


namespace sum_of_g_72_l145_145146

-- Define the conditions and the properties

def increasing (g : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, g (n + 1) > g n

def multiplicative (g : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, g (m * n) = g m * g n

def special_property (g : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n ∧ m ^ n = n ^ m → (g m = 2 * n ∨ g n = 2 * m)

-- Define the sum of possible values of g(72)

noncomputable def g_sum (g : ℕ+ → ℕ+) : ℕ+ := 294912

-- Main theorem statement

theorem sum_of_g_72 (g : ℕ+ → ℕ+) 
  (h1 : increasing g)
  (h2 : multiplicative g)
  (h3 : special_property g) :
  g 72 = g_sum g :=
sorry

end sum_of_g_72_l145_145146


namespace general_term_of_geom_sequence_sum_of_sequence_b_l145_145841

variable (n : ℕ)
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

noncomputable def geom_sequence (a : ℕ → ℝ) := ∀ n, a n = 1/2 * (-2)^(n-1)

theorem general_term_of_geom_sequence (a : ℕ → ℝ) (h1 : geom_sequence a) :
  a n = -(-2)^(n-2) :=
begin
  sorry
end

def b_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ := n * | a_n n |
noncomputable def sum_sequence (T : ℕ → ℝ) (n : ℕ) : ℝ := 
  1/2 * n * (n-1) * 2^(n-2) + 1/2

theorem sum_of_sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : geom_sequence a) (hb : ∀ n, b n = b_n a n):
  T n = (n-1) * 2^(n-1) + 1/2 :=
begin
  sorry
end

end general_term_of_geom_sequence_sum_of_sequence_b_l145_145841


namespace first_candidate_more_gain_l145_145337

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l145_145337


namespace T_bisects_broken_line_l145_145948

def midpoint_arc {α : Type*} [LinearOrderedField α] (A B C : α) : α := (A + B + C) / 2
def projection_perpendicular {α : Type*} [LinearOrderedField α] (F A B C : α) : α := sorry -- Define perpendicular projection T

theorem T_bisects_broken_line {α : Type*} [LinearOrderedField α]
  (A B C : α) (F := midpoint_arc A B C) (T := projection_perpendicular F A B C) :
  T = (A + B + C) / 2 :=
sorry

end T_bisects_broken_line_l145_145948


namespace sum_c_n_l145_145676

-- Define the geometric sequence {a_n} and its properties
def a_n (n : ℕ) : ℝ := 1 / 3^n

-- Define the sequence {b_n} based on the log base 3 of the {a_n}
def b_n (n : ℕ) : ℝ := -(n * (n + 1) / 2)

-- Define the sequence {c_n} as the sum of a_n and the reciprocal of b_n
def c_n (n : ℕ) : ℝ := a_n n + 1 / b_n n

-- Define the sum of the first n terms of the sequence {c_n}

noncomputable def S_n (n : ℕ) : ℝ := 
  (finset.range (n + 1)).sum (λ k, c_n k)

-- Prove the stated sum of the first n terms equals the given expression
theorem sum_c_n (n : ℕ) : 
  S_n n = (1 / 2) - (1 / (2 * 3^n)) - (2 * n / (n + 1)) :=
  sorry

end sum_c_n_l145_145676


namespace exists_b_for_system_l145_145806

theorem exists_b_for_system (a: ℝ) : 
  (a ≤ -2/3 ∨ a > 0) → 
  ∃ b x y, 
    x = 6/a - |y - a| ∧ 
    x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intros ha
  by_cases h1 : a = 0
  { exfalso, linarith only [h1, ha] }
  sorry

end exists_b_for_system_l145_145806


namespace polygon_sides_l145_145845

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l145_145845


namespace find_d_l145_145793
noncomputable def equilateral_triangle_side := 300
noncomputable def inradius := equilateral_triangle_side * Math.sqrt 3 / 6
noncomputable def circumradius := equilateral_triangle_side * Math.sqrt 3 / 3
noncomputable def distance_from_center_to_vertex := circumradius

theorem find_d {X Y Z R S M : Type}
  (H1 : dist(M, X) = dist(M, Y))
  (H2 : dist(M, X) = dist(M, Z))
  (H3 : dist(M, X) = dist(M, R))
  (H4 : dist(M, X) = dist(M, S))
  (H5 : dist(M, Y) = dist(M, Z))
  (H6 : dist(M, Y) = dist(M, R))
  (H7 : dist(M, Y) = dist(M, S))
  (H8 : dist(M, Z) = dist(M, R))
  (H9 : dist(M, Z) = dist(M, S))
  (H10 : dist(M, R) = dist(M, S))
  (H11 : RX = RY ∧ RY = RZ)
  (H12 : SX = SY ∧ SY = SZ)
  (H13 : angle (plane RXZ) (plane SXZ) = 90)
  (H14 : norm (X - Y) = norm (Y - Z) ∧ norm (Y - Z) = norm (Z - X) ∧ norm (X - Z) = 300)
  : ∀ d : ℝ, dist(M, X) = d ∧ dist(M, Y) = d ∧ dist(M, Z) = d ∧ dist(M, R) = d ∧ dist(M, S) = d -> d = distance_from_center_to_vertex :=
sorry

end find_d_l145_145793


namespace sin_of_angle_in_interval_l145_145816

theorem sin_of_angle_in_interval (x : Real) (h₁ : cos (π + x) = 3 / 5) (h₂ : π < x ∧ x < 2 * π) : sin x = -4 / 5 := 
by 
  sorry

end sin_of_angle_in_interval_l145_145816


namespace sum_first_100_terms_l145_145925

def sequence (a : ℕ → ℤ) := a 1 = -1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

theorem sum_first_100_terms
  (a : ℕ → ℤ)
  (h : sequence a) :
  (∑ i in Finset.range 100, a i.succ) = 5 :=
sorry

end sum_first_100_terms_l145_145925


namespace count_positive_integers_satisfying_inequality_l145_145088

theorem count_positive_integers_satisfying_inequality :
  (∃ s : Finset ℕ, (∀ x ∈ s, 8 < -2 * x + 17) ∧ ∀ x : ℕ, 8 < -2 * x + 17 → x ∈ s) ∧
  (s.card = 4) :=
sorry

end count_positive_integers_satisfying_inequality_l145_145088


namespace find_t_l145_145842

-- Define the power function condition
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ), ∃ (n : ℝ), f = λ x, c * x ^ n

-- Define the function f(x)
def f (t x : ℝ) : ℝ := (t^2 - t + 1) * x * (t + 3) / 5

-- Define the even function condition for symmetry about the y-axis
def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Lean theorem statement
theorem find_t (t : ℝ) :
  is_power_function (f t) →
  is_symmetric_about_y_axis (f t) →
  t = 1 := 
sorry

end find_t_l145_145842


namespace exists_i_j_for_sum_l145_145598

theorem exists_i_j_for_sum (n : ℕ) (h : n > 1) :
  ∃ (i j : ℕ), (i ≥ n-1) ∧ (j ≥ (n-1)) ∧ 
  (∑ k in (Finset.range (j-i+1)).map (Finset.add i), (1:ℚ)/((k)*(k+1))) = (1:ℚ)/n :=
sorry

end exists_i_j_for_sum_l145_145598


namespace ordering_of_a_b_c_l145_145095

noncomputable def a : ℝ := Real.log_base 3 Real.pi
noncomputable def b : ℝ := Real.ln 2
noncomputable def c : ℝ := Real.log10 0.8

theorem ordering_of_a_b_c : a > b ∧ b > c := by
  sorry

end ordering_of_a_b_c_l145_145095


namespace quadratic_discriminant_positive_find_m_l145_145502

-- Define the quadratic equation and its properties
def quadratic (x m : Int) : Int := x^2 - (2 * m + 1) * x + (m^2 + m)

-- Part (1)
theorem quadratic_discriminant_positive (m : Int) : 
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  Δ > 0 := by
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  have h : Δ = 1 := by sorry
  show 1 > 0 from by sorry

-- Part (2)
theorem find_m (a b m : Int) : 
  quadratic a m = 0 ∧ quadratic b m = 0 ∧ (2 * a + b) * (a + 2 * b) = 20 → 
  m = -2 ∨ m = 1 := by
  intro h
  sorry

end quadratic_discriminant_positive_find_m_l145_145502


namespace reflection_y_axis_l145_145411

open Matrix

def reflection_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

theorem reflection_y_axis (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M ⬝ ![![1, 0], ![0, 1]]) = reflection_y_axis_matrix :=
by sorry

end reflection_y_axis_l145_145411


namespace sampling_interval_l145_145687

theorem sampling_interval (total_students sample_size : ℕ) (H_total : total_students = 72) (H_sample : sample_size = 8) : total_students / sample_size = 9 :=
by
  rw [H_total, H_sample]
  norm_num
  sorry

end sampling_interval_l145_145687


namespace nina_earnings_l145_145184

/-- 
Problem: Calculate the total earnings from selling various types of jewelry.
Conditions:
- Necklace price: $25 each
- Bracelet price: $15 each
- Earring price: $10 per pair
- Complete jewelry ensemble price: $45 each
- Number of necklaces sold: 5
- Number of bracelets sold: 10
- Number of earrings sold: 20
- Number of complete jewelry ensembles sold: 2
Question: How much money did Nina make over the weekend?
Answer: Nina made $565.00
-/
theorem nina_earnings
  (necklace_price : ℕ)
  (bracelet_price : ℕ)
  (earring_price : ℕ)
  (ensemble_price : ℕ)
  (necklaces_sold : ℕ)
  (bracelets_sold : ℕ)
  (earrings_sold : ℕ)
  (ensembles_sold : ℕ) :
  necklace_price = 25 → 
  bracelet_price = 15 → 
  earring_price = 10 → 
  ensemble_price = 45 → 
  necklaces_sold = 5 → 
  bracelets_sold = 10 → 
  earrings_sold = 20 → 
  ensembles_sold = 2 →
  (necklace_price * necklaces_sold) + 
  (bracelet_price * bracelets_sold) + 
  (earring_price * earrings_sold) +
  (ensemble_price * ensembles_sold) = 565 := by
  sorry

end nina_earnings_l145_145184


namespace honda_cars_in_city_l145_145111

variable (H N : ℕ)

theorem honda_cars_in_city (total_cars : ℕ)
                         (total_red_car_ratio : ℚ)
                         (honda_red_car_ratio : ℚ)
                         (non_honda_red_car_ratio : ℚ)
                         (total_red_cars : ℕ)
                         (h : total_cars = 9000)
                         (h1 : total_red_car_ratio = 0.6)
                         (h2 : honda_red_car_ratio = 0.9)
                         (h3 : non_honda_red_car_ratio = 0.225)
                         (h4 : total_red_cars = 5400)
                         (h5 : H + N = total_cars)
                         (h6 : honda_red_car_ratio * H + non_honda_red_car_ratio * N = total_red_cars) :
  H = 5000 := by
  -- Proof goes here
  sorry

end honda_cars_in_city_l145_145111


namespace shaded_area_calculation_l145_145919

noncomputable def area_of_largest_circle : ℝ := 100 * Real.pi

def total_shaded_area (area_of_largest_circle : ℝ) : ℝ :=
  let R := Real.sqrt (area_of_largest_circle / Real.pi)
  let area_largest_shaded := area_of_largest_circle / 2
  let radius_second_circle := R / 2
  let area_second_circle := Real.pi * radius_second_circle^2
  let area_second_shaded := area_second_circle / 2
  let radius_third_circle := radius_second_circle / 2
  let area_third_circle := Real.pi * radius_third_circle^2
  let area_third_shaded := area_third_circle / 2
  area_largest_shaded + area_second_shaded + area_third_shaded

theorem shaded_area_calculation : total_shaded_area area_of_largest_circle = 65.625 * Real.pi := 
by
  sorry

end shaded_area_calculation_l145_145919


namespace count_two_digit_multiples_of_six_l145_145517

open Nat

def is_multiple_of_six (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_multiples_of_six :
  (Finset.card (Finset.filter is_multiple_of_six (Finset.filter is_two_digit (Finset.range 100)))) = 15 :=
by
  sorry

end count_two_digit_multiples_of_six_l145_145517


namespace collinear_points_l145_145693

variable {α : Type _}

-- Points A and B which are intersection points of circles w1 and w2
variables (A B : α)

-- Circles w1 and w2
variables (w1 w2 : set α)

-- Tangents to circles w1 and w2 through point A are l1 and l2
variables (l1 l2 : set α)

-- Points K and N
variables (K N : α)

-- Perpendiculars dropped from B to l2 intersect w1 at K
variable (H_K : (perpendicular_from B to l2) ∩ w1 = {K})

-- Perpendiculars dropped from B to l1 intersect w2 at N
variable (H_N : (perpendicular_from B to l1) ∩ w2 = {N})

theorem collinear_points (w1 w2 : set α) (A B K N : α)
  (H_AB_intersection : (w1 ∩ w2) = {A, B})
  (H_tangent_l1 : tangent_at A l1 w1)
  (H_tangent_l2 : tangent_at A l2 w2)
  (H_perpendicular_K : ∃ M, perpendicular_from B l2 M ∧ on_circle w1 M K)
  (H_perpendicular_N : ∃ L, perpendicular_from B l1 L ∧ on_circle w2 L N) :
  collinear K A N :=
by
  sorry

end collinear_points_l145_145693


namespace find_a_for_complex_fraction_real_l145_145892

theorem find_a_for_complex_fraction_real (a : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a + i) / (1 - i) ∈ ℝ) : a = -1 :=
sorry

end find_a_for_complex_fraction_real_l145_145892


namespace train_crossing_time_l145_145520

noncomputable def length_first_train : ℝ := 200  -- meters
noncomputable def speed_first_train_kmph : ℝ := 72  -- km/h
noncomputable def speed_first_train : ℝ := speed_first_train_kmph * (1000 / 3600)  -- m/s

noncomputable def length_second_train : ℝ := 300  -- meters
noncomputable def speed_second_train_kmph : ℝ := 36  -- km/h
noncomputable def speed_second_train : ℝ := speed_second_train_kmph * (1000 / 3600)  -- m/s

noncomputable def relative_speed : ℝ := speed_first_train - speed_second_train -- m/s
noncomputable def total_length : ℝ := length_first_train + length_second_train  -- meters
noncomputable def time_to_cross : ℝ := total_length / relative_speed  -- seconds

theorem train_crossing_time :
  time_to_cross = 50 := by
  sorry

end train_crossing_time_l145_145520


namespace sum_of_three_numbers_l145_145906

theorem sum_of_three_numbers (a b c : ℝ) (h₁ : a + b = 31) (h₂ : b + c = 48) (h₃ : c + a = 59) :
  a + b + c = 69 :=
by
  sorry

end sum_of_three_numbers_l145_145906


namespace count_two_digit_numbers_with_prime_sum_l145_145884

-- Helper predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m < n → n % m ≠ 0

-- Helper function to get the sum of digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  tens + units

-- Define the set of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the set of two-digit numbers with prime digit sum
def two_digit_numbers_with_prime_sum :=
  {n : ℕ | n ∈ two_digit_numbers ∧ is_prime (sum_of_digits n)}

-- The main theorem
theorem count_two_digit_numbers_with_prime_sum : 
  {n : ℕ | n ∈ two_digit_numbers_with_prime_sum }.to_finset.card = 33 :=
by
  -- Placeholder for the actual proof
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145884


namespace area_of_region_l145_145606

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 - p.1)^2 + (abs p.2 - p.2)^2 ≤ 16 ∧ 2 * p.2 + p.1 ≤ 0}

noncomputable def area : ℝ := sorry

theorem area_of_region : area = 5 + Real.pi := by
  sorry

end area_of_region_l145_145606


namespace triangle_angles_60_degrees_l145_145458

theorem triangle_angles_60_degrees
  (α β γ : ℝ)
  (a b c R : ℝ)
  (h1 : 0 < R)
  (h2 : a = 2 * R * sin α)
  (h3 : b = 2 * R * sin β)
  (h4 : c = 2 * R * sin γ)
  (h5 : (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R)) :
  α = π / 3 ∧ β = π / 3 ∧ γ = π / 3 := 
sorry

end triangle_angles_60_degrees_l145_145458


namespace reflectionY_matrix_correct_l145_145418

-- Define the basis vectors e₁ and e₂
def e1 : Vector := ⟨1, 0⟩
def e2 : Vector := ⟨0, 1⟩

-- Define the transformation that reflects over the y-axis
def reflectY : Vector → Vector 
| ⟨x, y⟩ => ⟨-x, y⟩

-- Conditions given in the problem
lemma reflectY_e1 : reflectY e1 = ⟨-1, 0⟩ := sorry
lemma reflectY_e2 : reflectY e2 = ⟨0, 1⟩ := sorry

-- The goal is to find the transformation matrix for reflection over the y-axis
def reflectionMatrixY : Matrix 2 2 ℝ :=
  Matrix.of_vec ([-1, 0, 0, 1])

theorem reflectionY_matrix_correct :
  ∀ (v : Vector), reflectY v = reflectionMatrixY.mul_vec v := sorry

end reflectionY_matrix_correct_l145_145418


namespace sum_of_terms_range_of_m_l145_145064

noncomputable def a (n : ℕ) : ℚ := (1/4)^n
noncomputable def b (n : ℕ) : ℚ := 3 * n - 2
noncomputable def c (n : ℕ) : ℚ := a n * b n

theorem sum_of_terms (n : ℕ) : 
  let S_n := ∑ k in Finset.range n, c (k + 1)
  S_n = 2 / 3 - (3 * n + 2) / 3 * (1 / 4) ^ n := sorry

theorem range_of_m (m : ℚ) :
  (∀ n : ℕ, n > 0 → c n ≤ 1 / 8 * m^2 + 3 / 4 * m - 5 / 8) ↔ (m ≥ 1 ∨ m ≤ -7) := sorry

end sum_of_terms_range_of_m_l145_145064


namespace hard_candy_food_colouring_l145_145316

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end hard_candy_food_colouring_l145_145316


namespace intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l145_145018

noncomputable def setA : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | 1 < x }

theorem intersection_AB : setA ∩ setB = { x : ℝ | 1 < x ∧ x < 2 } := by
  sorry

theorem union_AB : setA ∪ setB = { x : ℝ | -1 < x } := by
  sorry

theorem difference_A_minus_B : setA \ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } := by
  sorry

theorem difference_B_minus_A : setB \ setA = { x : ℝ | 2 ≤ x } := by
  sorry

end intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l145_145018


namespace petya_higher_chance_of_winning_l145_145206

theorem petya_higher_chance_of_winning 
  {total_candies : ℕ} (h_total : total_candies = 25)
  {P_two_caramel : ℝ} (h_P_two_caramel : P_two_caramel = 0.54) :
  let P_two_chocolate := 1 - P_two_caramel in
  P_two_chocolate < 0.5 :=
by {
  sorry
}

end petya_higher_chance_of_winning_l145_145206


namespace racing_problem_l145_145540

theorem racing_problem
  (distance : ℕ) (A_beats_B_by : ℕ) (A_time : ℕ)
  (h1 : distance = 120)
  (h2 : A_beats_B_by = 56)
  (h3 : A_time = 8) :
  let V_A := distance / A_time in
  let distance_B := distance - A_beats_B_by in
  let V_B := distance_B / A_time in
  let T_B := A_beats_B_by / V_B in
  (A_time + T_B) - A_time = 7 :=
by
  sorry

end racing_problem_l145_145540


namespace deficit_percentage_l145_145914

variable (A B : ℝ) -- Actual lengths of the sides of the rectangle
variable (x : ℝ) -- Percentage in deficit
variable (measuredA := A * 1.06) -- One side measured 6% in excess
variable (errorPercent := 0.7) -- Error percent in area
variable (measuredB := B * (1 - x / 100)) -- Other side measured x% in deficit
variable (actualArea := A * B) -- Actual area of the rectangle
variable (calculatedArea := (A * 1.06) * (B * (1 - x / 100))) -- Calculated area with measurement errors
variable (correctArea := actualArea * (1 + errorPercent / 100)) -- Correct area considering the error

theorem deficit_percentage : 
  calculatedArea = correctArea → 
  x = 5 :=
by
  sorry

end deficit_percentage_l145_145914


namespace alice_profit_l145_145354

noncomputable def total_bracelets : ℕ := 52
noncomputable def cost_of_materials : ℝ := 3.00
noncomputable def bracelets_given_away : ℕ := 8
noncomputable def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_remaining := total_bracelets - bracelets_given_away;
      total_revenue := bracelets_remaining * price_per_bracelet;
      profit := total_revenue - cost_of_materials
  in profit = 8.00 := 
by 
  sorry

end alice_profit_l145_145354


namespace running_problem_l145_145684

theorem running_problem 
    (x_speed : ℕ) (f_speed : ℕ) : x_speed = 2 → f_speed = 3 → 
    let total_meetings := 5 in
    let final_meetings := total_meetings - 1 in
    final_meetings = 4 :=
by 
    intros h1 h2 
    let total_meetings := 5
    let final_meetings := total_meetings - 1
    have h3 : final_meetings = 4 ∧ x_speed = 2 ∧ f_speed = 3 := by
      sorry
    exact h3.1

end running_problem_l145_145684


namespace books_sold_on_tuesday_l145_145569

variables (total_books mon tue wed thu fri : ℕ) (unsold_percentage : ℝ)

def unsold_books (total_books : ℕ) (unsold_percentage : ℝ) : ℕ := 
  (total_books * unsold_percentage / 100).round

def books_sold_excluding_tuesday (mon wed thu fri : ℕ) : ℕ := mon + wed + thu + fri

theorem books_sold_on_tuesday :
  total_books = 900 →
  mon = 75 →
  wed = 64 →
  thu = 78 →
  fri = 135 →
  unsold_percentage = 55.333333333333336 →
  tue = total_books - (books_sold_excluding_tuesday mon wed thu fri + unsold_books total_books unsold_percentage) →
  tue = 50 :=
by
  intros h_total h_mon h_wed h_thu h_fri h_unsold h_tue
  -- Proof goes here
  sorry

end books_sold_on_tuesday_l145_145569


namespace find_x_in_PetyaSequence_l145_145989

noncomputable def PetyaSequence (n : ℕ) : Prop :=
n ≥ 2 ∧ 
(∀ k, (0 ≤ k ∧ k < n → ∀ d, d ≠ 4 ∧ (to_list (k / 10 : ℕ).digit.to_string).get 1 d ≠ some '4')) ∧
(∀ d, (to_list ((n - 1) / 10 : ℕ).digit.to_string).get 1 d ≠ some '7) ∧
(∃ a b : ℕ, Prime a ∧ Prime b ∧ a ≠ b ∧ b = a + 4 ∧ (10 ∣ (((a + b) / 2) - 5)) ∧
 ∃ x : ℕ, x = a * b ∧ x % 100 = 21 ∧ x = 2021)

theorem find_x_in_PetyaSequence (n : ℕ) (h : PetyaSequence n) : ∃ x : ℕ, x = 2021 := by
  sorry

end find_x_in_PetyaSequence_l145_145989


namespace moving_point_parabola_l145_145326

-- Define the fixed point F and the line l
structure Point where
  x : Float
  y : Float

def F : Point := {x := 3, y := 0}

def l (p : Point) : Prop := p.x = -1

-- Define the distance formula from a point to a line
def distPointToLine (p : Point) : Float :=
  abs (p.x + 1)

-- Define the distance formula from a point to another point
def dist (p1 p2 : Point) : Float :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the condition and the result
theorem moving_point_parabola (M : Point) :
  dist M F = distPointToLine M + 2 -> M.y^2 = 12 * M.x :=
by
  sorry

end moving_point_parabola_l145_145326


namespace final_total_cost_l145_145345

def initial_spiral_cost : ℝ := 15
def initial_planner_cost : ℝ := 10
def spiral_discount_rate : ℝ := 0.20
def planner_discount_rate : ℝ := 0.15
def num_spirals : ℝ := 4
def num_planners : ℝ := 8
def sales_tax_rate : ℝ := 0.07

theorem final_total_cost :
  let discounted_spiral_cost := initial_spiral_cost * (1 - spiral_discount_rate)
  let discounted_planner_cost := initial_planner_cost * (1 - planner_discount_rate)
  let total_before_tax := num_spirals * discounted_spiral_cost + num_planners * discounted_planner_cost
  let total_tax := total_before_tax * sales_tax_rate
  let total_cost := total_before_tax + total_tax
  total_cost = 124.12 :=
by
  sorry

end final_total_cost_l145_145345


namespace net_gain_difference_l145_145331

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l145_145331


namespace tangent_line_at_point_l145_145818

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * real.exp x * real.log x + (b * real.exp (x - 1)) / x

noncomputable def f'(x : ℝ) (a b : ℝ) : ℝ :=
  a * real.exp x * real.log x + (a / x) * real.exp x - (b / (x ^ 2)) * real.exp (x - 1) + (b / x) * real.exp (x - 1)

theorem tangent_line_at_point {a b : ℝ} (h₁ : f 1 a b = 2) (h₂ : f' 1 a b = real.exp 1) : a + b = 3 := 
  sorry

end tangent_line_at_point_l145_145818


namespace problem_l145_145284

theorem problem (a b : ℕ)
  (ha : a = 2) 
  (hb : b = 121) 
  (h_minPrime : ∀ n, n < a → ¬ (∀ d, d ∣ n → d = 1 ∨ d = n))
  (h_threeDivisors : ∀ n, n < 150 → ∀ d, d ∣ n → d = 1 ∨ d = n → n = 121) :
  a + b = 123 := by
  sorry

end problem_l145_145284


namespace min_value_of_expression_l145_145836

theorem min_value_of_expression {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (1 / a) + (2 / b) >= 8 :=
by
  sorry

end min_value_of_expression_l145_145836


namespace no_hyperbola_with_82_points_l145_145930

noncomputable def S (a : ℝ) : ℕ :=
  ∑ k in finset.range (⌊a⌋₊ + 1), ⌊a / (k + 1)⌋₊

theorem no_hyperbola_with_82_points : 
  ¬ ∃ (a : ℝ) (h : a > 0), S a = 82 := by
  sorry

end no_hyperbola_with_82_points_l145_145930


namespace sum_geq_4k_l145_145133

theorem sum_geq_4k (a b k : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_k : k > 1)
  (h_lcm_gcd : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : a + b ≥ 4 * k := 
by 
  sorry

end sum_geq_4k_l145_145133


namespace find_values_l145_145172

open Real

noncomputable def values_of_xyz (x y z : ℝ) : Prop :=
  x + 2 * y - z = 20 ∧
  y = 5 ∧
  3 * x + 4 * z = 40

theorem find_values :
  ∃ (x y z : ℝ), values_of_xyz x y z ∧ x ≈ 11.42857 ∧ y = 5 ∧ z ≈ 1.42857 := 
by
  sorry

end find_values_l145_145172


namespace distance_S_to_PQR_l145_145216

open euclidean_geometry -- assuming we use some geometry library

theorem distance_S_to_PQR {P Q R S : Point} 
  (h1 : S.distance P = 10) 
  (h2 : S.distance Q = 10)
  (h3 : S.distance R = 8) 
  (h4 : S.perpendicular P Q)
  (h5 : S.perpendicular P R)
  (h6 : S.perpendicular Q R) :
  S.distance_to_plane P Q R = 8 :=
by 
  sorry

end distance_S_to_PQR_l145_145216


namespace bruce_total_cost_l145_145369

def cost_of_grapes : ℕ := 8 * 70
def cost_of_mangoes : ℕ := 11 * 55
def cost_of_oranges : ℕ := 5 * 45
def cost_of_apples : ℕ := 3 * 90
def cost_of_cherries : ℕ := (45 / 10) * 120  -- use rational division and then multiplication

def total_cost : ℕ :=
  cost_of_grapes + cost_of_mangoes + cost_of_oranges + cost_of_apples + cost_of_cherries

theorem bruce_total_cost : total_cost = 2200 := by
  sorry

end bruce_total_cost_l145_145369


namespace quadratic_real_roots_and_m_values_l145_145490

theorem quadratic_real_roots_and_m_values (m : ℝ) :
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  in (Δ > 0) ∧
     (∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1)) :=
by
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  have Δ_pos : Δ > 0 := sorry
  have find_m : ∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1) := sorry
  exact ⟨Δ_pos, find_m⟩

#check quadratic_real_roots_and_m_values

end quadratic_real_roots_and_m_values_l145_145490


namespace quadratic_inequality_solution_l145_145259

theorem quadratic_inequality_solution (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l145_145259


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_l145_145035

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1^2 + (2 - m) * x1 + (1 - m) = 0 ∧
                 x2^2 + (2 - m) * x2 + (1 - m) = 0 :=
by sorry

theorem find_m_for_roots_difference (m x1 x2 : ℝ) (h1 : x1^2 + (2 - m) * x1 + (1 - m) = 0) 
  (h2 : x2^2 + (2 - m) * x2 + (1 - m) = 0) (hm : m < 0) (hd : x1 - x2 = 3) : 
  m = -3 :=
by sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_l145_145035


namespace profit_is_eight_dollars_l145_145351

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l145_145351


namespace quadratic_real_roots_and_m_values_l145_145489

theorem quadratic_real_roots_and_m_values (m : ℝ) :
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  in (Δ > 0) ∧
     (∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1)) :=
by
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  have Δ_pos : Δ > 0 := sorry
  have find_m : ∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1) := sorry
  exact ⟨Δ_pos, find_m⟩

#check quadratic_real_roots_and_m_values

end quadratic_real_roots_and_m_values_l145_145489


namespace perp_AB_CD_l145_145077

noncomputable theory

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def line (b : ℝ) (x y : ℝ) := y = x + b

def circle_through_four_points {A B C D : ℝ × ℝ} 
  (hA : parabola p A.1 A.2) 
  (hB : parabola p B.1 B.2) 
  (hC : parabola p C.1 C.2) 
  (hD : parabola p D.1 D.2) 
  (circ : ∃ (d e f : ℝ), ∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C ∨ (x, y) = D → x^2 + y^2 + d * x + e * y + f = 0) : Prop := 
  true

theorem perp_AB_CD {p : ℝ} (hp : 0 < p) 
  (b : ℝ) {A B C D : ℝ × ℝ}
  (hA: parabola p A.1 A.2) 
  (hB: parabola p B.1 B.2)
  (hAB_line: line b A.1 A.2 ∧ line b B.1 B.2)
  (hC: parabola p C.1 C.2) 
  (hD: parabola p D.1 D.2)
  (h_circle: circle_through_four_points hA hB hC hD) :
  (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0 :=
sorry

end perp_AB_CD_l145_145077


namespace preimage_of_3_1_l145_145486

theorem preimage_of_3_1 (a b : ℝ) (f : ℝ × ℝ → ℝ × ℝ) (h : ∀ (a b : ℝ), f (a, b) = (a + 2 * b, 2 * a - b)) :
  f (1, 1) = (3, 1) :=
by {
  sorry
}

end preimage_of_3_1_l145_145486


namespace total_stamps_correct_l145_145976

def current_stamps : Nat := 40
def percent_increase : Nat := 20
def additional_stamps : Nat := (current_stamps * percent_increase) / 100
def total_stamps : Nat := current_stamps + additional_stamps

theorem total_stamps_correct : total_stamps = 48 := by
  have h1 : additional_stamps = 8 := by
    rw [additional_stamps]
    norm_num
  rw [total_stamps, h1]
  norm_num
  sorry

end total_stamps_correct_l145_145976


namespace maintenance_increase_l145_145725

theorem maintenance_increase (t : ℝ) (p : ℝ) (h₁ : t = 45) (h₂ : p = 1 / 3) : t + t * p = 60 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end maintenance_increase_l145_145725


namespace convert_88_to_base_5_l145_145782

def decimal_to_base (n : ℕ) (b : ℕ) : list ℕ :=
if h : b > 1 then
  let rec aux (n : ℕ) : list ℕ :=
    if n = 0 then [] else
    let (q, r) := n.div_mod b in r :: aux q
  in (aux n).reverse
else []

theorem convert_88_to_base_5 :
  decimal_to_base 88 5 = [3, 2, 3] :=
by sorry

end convert_88_to_base_5_l145_145782


namespace basketball_games_played_l145_145720

theorem basketball_games_played (G : ℕ) (H1 : 35 ≤ G) (H2 : 25 ≥ 0) (H3 : 64 = 100 * (48 / (G + 25))):
  G = 50 :=
sorry

end basketball_games_played_l145_145720


namespace quadratic_real_roots_and_m_values_l145_145492

theorem quadratic_real_roots_and_m_values (m : ℝ) :
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  in (Δ > 0) ∧
     (∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1)) :=
by
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  have Δ_pos : Δ > 0 := sorry
  have find_m : ∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1) := sorry
  exact ⟨Δ_pos, find_m⟩

#check quadratic_real_roots_and_m_values

end quadratic_real_roots_and_m_values_l145_145492


namespace extremum_value_l145_145596

def f (x : ℝ) : ℝ := 1 - x * sin x

def has_extremum (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

theorem extremum_value (x₀ : ℝ) 
  (h : has_extremum x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 := 
sorry

end extremum_value_l145_145596


namespace standard_deviation_transformed_l145_145869

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable (s^2 : ℝ)

theorem standard_deviation_transformed (h : s^2 = 4) : 
  Real.sqrt (9 * s^2) = 6 := by
  sorry

end standard_deviation_transformed_l145_145869


namespace sequence_an_general_formula_sequence_bn_sum_l145_145256

theorem sequence_an_general_formula
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3) :
  ∀ n, a n = 3 ^ n := sorry

theorem sequence_bn_sum
  (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3)
  (h3 : ∀ n, b n = a (n + 1) / (S n * S (n + 1))) :
  ∀ n, T n = (2 / 3) * (1 / 2 - 1 / (3 ^ (n + 1) - 1)) := sorry

end sequence_an_general_formula_sequence_bn_sum_l145_145256


namespace velocity_at_t3_l145_145651

-- Define the equation of motion s = 1/t^4
def equation_of_motion (t : ℝ) : ℝ := 1 / t^4

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := -4 / t^5

-- Theorem stating the velocity of the particle at t = 3s
theorem velocity_at_t3 : velocity 3 = -4 / 243 :=
by
  -- Proof omitted
  sorry

end velocity_at_t3_l145_145651


namespace lambda_range_l145_145082

open Real

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def vector_eq (M C D : ℝ × ℝ) (λ : ℝ) : Prop :=
  (D.1 - M.1, D.2 - M.2) = (λ * (C.1 - M.1), λ * (C.2 - M.2))

theorem lambda_range
  (C D : ℝ × ℝ) (M : ℝ × ℝ) (λ : ℝ)
  (h1 : M = (0, 2))
  (h2 : ellipse C.1 C.2)
  (h3 : ellipse D.1 D.2)
  (h4 : vector_eq M C D λ) :
  λ ∈ set.Icc (1 / 3 : ℝ) 3 :=
sorry

end lambda_range_l145_145082


namespace circumcenter_of_triangle_l145_145561

/-- In triangle ABC, ∠B = 2∠C. -/
def angle_relation (A B C : Point) : Prop :=
  ∠B = 2 * ∠C

/-- P and Q lie on the perpendicular bisector of the side CB. -/
def perpendicular_bisector (C B P Q : Point) : Prop :=
  (P ≠ Q) ∧ (dist C P = dist B P) ∧ (dist C Q = dist B Q)

/-- ∠CAP = ∠PAQ = ∠QAB = 1/3∠A. -/
def angle_conditions (A P Q : Point) (A_angle : Angle) : Prop :=
  ∠CAP = 1/3 * A_angle ∧ ∠PAQ = 1/3 * A_angle ∧ ∠QAB = 1/3 * A_angle

theorem circumcenter_of_triangle (A B C P Q : Point) (A_angle : Angle)
  (h1 : angle_relation A B C) 
  (h2 : perpendicular_bisector C B P Q) 
  (h3 : angle_conditions A P Q A_angle) :
  is_circumcenter_of_triangle Q C P B :=
sorry

end circumcenter_of_triangle_l145_145561


namespace parallel_lines_concurrent_l145_145946

-- Definitions and assumptions
variables (A B C O H : Point)
variable (circumcircle : Circle)
variable (abc_triangle : Triangle A B C)

-- Given conditions
axiom circumcentre (h : isCircumcenter O (Triangle A B C))
axiom orthocentre (h : isOrthocenter H (Triangle A B C))
axiom reflection_A'_BC : reflectOverLine H (Line BC) = A'
axiom reflection_B'_CA : reflectOverLine H (Line CA) = B'
axiom reflection_C'_AB : reflectOverLine H (Line AB) = C'

-- The theorem to prove the concurrent lines
theorem parallel_lines_concurrent :
  areConcurrent
    (Line.parallelThroughPoint (Line.fromPoints O A) A')
    (Line.parallelThroughPoint (Line.fromPoints O B) B')
    (Line.parallelThroughPoint (Line.fromPoints O C) C') :=
sorry

end parallel_lines_concurrent_l145_145946


namespace find_k_root_interval_l145_145902

theorem find_k_root_interval 
  (f : ℝ → ℝ)
  (k : ℤ)
  (k_nonzero : k ≠ 0)
  (interval : ∀ x, 2 < x ∧ x < 3 → f(x) = log 2 x + x - k) : 
  ∃ root, 2 < root ∧ root < 3 ∧ f(root) = 0 → k = 4 :=
by 
  sorry

end find_k_root_interval_l145_145902


namespace smallest_prime_dividing_large_sum_is_5_l145_145293

-- Definitions based on the conditions
def large_sum : ℕ := 4^15 + 7^12

-- Prime number checking function
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Check for the smallest prime number dividing the sum
def smallest_prime_dividing_sum (n : ℕ) : ℕ := 
  if n % 2 = 0 then 2 
  else if n % 3 = 0 then 3 
  else if n % 5 = 0 then 5 
  else 2 -- Since 2 is a placeholder, theoretical logic checks can replace this branch

-- Final theorem to prove
theorem smallest_prime_dividing_large_sum_is_5 : smallest_prime_dividing_sum large_sum = 5 := 
  sorry

end smallest_prime_dividing_large_sum_is_5_l145_145293


namespace parameterized_line_l145_145248

noncomputable def g (t : ℝ) : ℝ := 9 * t + 10

theorem parameterized_line (t : ℝ) :
  let x := g t
  let y := 18 * t - 10
  y = 2 * x - 30 :=
by
  sorry

end parameterized_line_l145_145248


namespace equilateral_triangle_side_length_l145_145143

theorem equilateral_triangle_side_length
  (A B C P : Type)
  (s : ℝ)
  (h_equilateral : distance A B = s ∧ distance B C = s ∧ distance C A = s)
  (h_AP : distance A P = 1)
  (h_BP : distance B P = √3)
  (h_CP : distance C P = 2) :
  s = √7 :=
begin
  sorry
end

end equilateral_triangle_side_length_l145_145143


namespace minimum_period_f_monotonically_decreasing_interval_range_when_x_in_interval_l145_145482

noncomputable def f : ℝ → ℝ := λ x, 2 * Real.sin (2 * x - Real.pi / 6) - 1

theorem minimum_period_f : ∃ p > 0, ∀ x ∈ ℝ, f (x + p) = f x :=
  sorry

theorem monotonically_decreasing_interval : 
  ∀ k : ℤ, (∀ x y, k * Real.pi + Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ 2 * k * Real.pi + 5 * Real.pi / 3 → f y ≤ f x) :=
  sorry

theorem range_when_x_in_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → ( ∃ y, y = f x ∧ -2 ≤ y ∧ y ≤ 1 ) :=
  sorry

end minimum_period_f_monotonically_decreasing_interval_range_when_x_in_interval_l145_145482


namespace problem_incorrect_property_l145_145483

theorem problem_incorrect_property :
  ∀ x : ℝ, - (5 / 6) * π ≤ x ∧ x ≤ π / 6 →
    ¬(∀ x1 x2 : ℝ, x1 < x2 → x + π ≤ x1 ∧ x1 ≤ x + π ∧ x2 ≤ x + 2 * π →
      (√2) * sin (x1 + π + π/3) < (√2) * sin (x2 + π + π/3)) :=
by
  sorry

end problem_incorrect_property_l145_145483


namespace quadratic_real_roots_and_m_values_l145_145491

theorem quadratic_real_roots_and_m_values (m : ℝ) :
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  in (Δ > 0) ∧
     (∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1)) :=
by
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  have Δ_pos : Δ > 0 := sorry
  have find_m : ∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1) := sorry
  exact ⟨Δ_pos, find_m⟩

#check quadratic_real_roots_and_m_values

end quadratic_real_roots_and_m_values_l145_145491


namespace determine_e_l145_145958

def p (x: ℝ) : ℝ := 5 * x - 17
def q (x: ℝ) : ℝ := 4 * x - e

theorem determine_e (e : ℝ)
  (h : p (q 3) = 23) : e = 4 :=
by sorry

end determine_e_l145_145958


namespace sum_of_squares_of_roots_of_quadratic_l145_145708

noncomputable def sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) : Prop :=
  a^2 + b^2 = 4 * p^2 - 6 * q

theorem sum_of_squares_of_roots_of_quadratic
  (p q a b : ℝ)
  (h1 : a + b = 2 * p / 3)
  (h2 : a * b = q / 3)
  (h3 : a * a + b * b = 4 * p^2 - 6 * q) :
  sum_of_squares_of_roots p q a b :=
by
  sorry

end sum_of_squares_of_roots_of_quadratic_l145_145708


namespace tangent_parallel_AB_find_k_perp_l145_145455

theorem tangent_parallel_AB (k : ℝ) :
  (∃ x1 y1 x2 y2 M N : ℝ,
    (y1 = k * (x1 - 2) ∧ y2 = k * (x2 - 2)) ∧
    (y1^2 = 1/2 * x1 ∧ y2^2 = 1/2 * x2) ∧
    (M = (x1 + x2) / 2 ∧ N = (M - 2)^2 / k) ∧
    -- the tangent slope condition
    (tangent_slope y1 y2 M N k)
  ) :=
sorry

theorem find_k_perp (k : ℝ) :
  (∃ x1 y1 x2 y2 M N : ℝ,
    (y1 = k * (x1 - 2) ∧ y2 = k * (x2 - 2)) ∧
    (y1^2 = 1/2 * x1 ∧ y2^2 = 1/2 * x2) ∧
    (M = (x1 + x2) / 2 ∧ N = (M - 2)^2 / k) ∧
    -- condition NA . NB = 0 leads to k = ± 1/2
    (tangent_slope y1 y2 M N k)
  ) → (k = 1/2 ∨ k = -1/2) :=
sorry

end tangent_parallel_AB_find_k_perp_l145_145455


namespace total_time_proof_l145_145652

-- Define constants and conditions
def total_length : ℝ := 300
def segment_length : ℝ := 100
def stop_interval : ℝ := 50
def stop_duration : ℝ := 5
def resume_time : ℝ := 2

def exe_speed_segment1 : ℝ := 30
def exe_speed_segment2 : ℝ := 20
def exe_speed_segment3 : ℝ := 40

def person_speed : ℝ := 10

def stops_per_segment : ℕ := segment_length / stop_interval

-- Function to compute the combined speed in each segment
def combined_speed (e_speed : ℝ) : ℝ :=
  e_speed + person_speed

-- Function to compute the time to cover each segment without stops
def time_without_stops (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

-- Function to compute the total time lost due to stops in each segment
def time_lost (stops : ℕ) : ℝ :=
  stops * (stop_duration + resume_time)

-- Calculations for total time
def total_time : ℝ :=
  let segment1_time := (time_without_stops segment_length (combined_speed exe_speed_segment1)) + 
                       (time_lost stops_per_segment),
      segment2_time := (time_without_stops segment_length (combined_speed exe_speed_segment2)) + 
                       (time_lost stops_per_segment),
      segment3_time := (time_without_stops segment_length (combined_speed exe_speed_segment3)) +
                       (time_lost stops_per_segment)
  in segment1_time + segment2_time + segment3_time

-- The theorem to prove the total time is approximately 49.83 seconds
theorem total_time_proof : total_time = 49.83 := 
by
  sorry

end total_time_proof_l145_145652


namespace petya_has_higher_chance_of_winning_l145_145201

noncomputable def vasya_win_probability : ℝ := 0.46
noncomputable def petya_win_probability : ℝ := 1 - vasya_win_probability

theorem petya_has_higher_chance_of_winning 
  (h: vasya_win_probability = 1 - 0.54) : petya_win_probability > 0.5 :=
by
  have hp : petya_win_probability = 1 - vasya_win_probability := rfl
  have h_vasya : vasya_win_probability = 0.46 := h
  rw [h_vasya] at hp
  have h_petya : petya_win_probability = 1 - 0.46 := hp
  simp [h_petya]
  exact by norm_num

end petya_has_higher_chance_of_winning_l145_145201


namespace candidate_net_gain_difference_l145_145336

theorem candidate_net_gain_difference :
  let salary1 := 42000
      revenue1 := 93000
      training_cost_per_month := 1200
      training_months := 3
      salary2 := 45000
      revenue2 := 92000
      hiring_bonus_percent := 1 / 100 in
  let total_training_cost1 := training_cost_per_month * training_months in
  let hiring_bonus2 := salary2 * hiring_bonus_percent in
  let net_gain1 := revenue1 - salary1 - total_training_cost1 in
  let net_gain2 := revenue2 - salary2 - hiring_bonus2 in
  net_gain1 - net_gain2 = 850 :=
by
  sorry

end candidate_net_gain_difference_l145_145336


namespace car_service_30th_month_l145_145796

theorem car_service_30th_month :
  ∀ (n : ℕ), (n = 30) →
  ∃ (month : ℕ), month = ((7 * (n - 1) % 12 + 1) % 12) + 1 ∧ month = 12 :=
by {
  intro n hn,
  use ((7 * (n - 1) % 12 + 1) % 12) + 1,
  split,
  {
    simp [hn],
    norm_num,
  },
  {
    simp [hn],
    norm_num,
  },
  sorry,
}

end car_service_30th_month_l145_145796


namespace ellipse_standard_eq_line_eq_l145_145837

-- Define conditions
def center_origin : Prop := true  -- condition for center of ellipse at origin
def foci_x_axis : Prop := true  -- condition for foci on x-axis
def major_axis_length : Prop := ∃ a : ℝ, 2*a = 2*real.sqrt 3  -- length of major axis is 2√3
def eccentricity : Prop := ∃ e : ℝ, e = real.sqrt 6 / 3  -- eccentricity is √6/3
def line_passing_through_point : Prop := true  -- line l passes through (1,0)
def line_intersects_ellipse : Prop := true  -- line l intersects ellipse C at points A and B
def circle_diameter_passes_origin : Prop := true  -- circle with diameter AB passes through the origin

-- Given the conditions, prove the standard equation of the ellipse
theorem ellipse_standard_eq :
  (center_origin ∧ foci_x_axis ∧ major_axis_length ∧ eccentricity) →
  (∃ a b: ℝ, a > b ∧ (ellipse_eq : ∀ (x y: ℝ), (x^2/a^2 + y^2/b^2 = 1) ↔ (x^2) / 3 + y^2 = 1)) :=
by sorry

-- Given the conditions, prove the equation of line l
theorem line_eq :
  (center_origin ∧ foci_x_axis ∧ major_axis_length ∧ eccentricity ∧ line_passing_through_point ∧ line_intersects_ellipse ∧ circle_diameter_passes_origin) →
  (∃ m: ℝ, m = real.sqrt 3 / 3 ∨ m = -real.sqrt 3 / 3 ∧ ∀ (x y: ℝ), x = m * y + 1) :=
by sorry

end ellipse_standard_eq_line_eq_l145_145837


namespace train_crossing_time_l145_145086

noncomputable def train_length := 375 / 1 -- length of train in meters
noncomputable def bridge_length := 450 / 1 -- length of bridge in meters
noncomputable def total_distance := train_length + bridge_length -- total distance to cross in meters

noncomputable def speed_kph := 78.5 -- speed of the train in km/hr
noncomputable def speed_mps := speed_kph * (1000 / 1) / 3600 -- speed of the train in m/s

noncomputable def time_to_cross := total_distance / speed_mps -- time to cross in seconds

theorem train_crossing_time : time_to_cross ≈ 37.83 := by sorry

end train_crossing_time_l145_145086


namespace max_min_values_l145_145091

noncomputable def y (α β : ℝ) : ℝ := cos β - 6 * sin α

theorem max_min_values (α β : ℝ) (h : 2 * α + β = π) : 
  -5 ≤ y α β ∧ y α β ≤ 7 :=
by
  intro α β
  assume h : 2 * α + β = π 
  sorry

end max_min_values_l145_145091


namespace coeff_x7_in_expansion_l145_145921

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coeff_x7_in_expansion : 
  let general_term := ∀ (r : ℕ), binomial_coefficient 8 r * (1 / 2)^r * x^(16 - 3 * r)
  in 7 = binomial_coefficient 8 3 * (1 / 2)^3 :=
by
  sorry

end coeff_x7_in_expansion_l145_145921


namespace find_phi_l145_145152

theorem find_phi :
  (∃ (Q s : ℂ) (φ : ℝ), s > 0 ∧ 0 ≤ φ ∧ φ < 360 ∧ 
    (Q = s * (complex.cos (real.pi * φ / 180) + complex.sin (real.pi * φ / 180) * complex.I)) ∧
    (Q = ∏ (z : ℂ) in {z | z^6 = 1 ∧ z ≠ 1 ∧ z.im > 0}, z)) →
  φ = 180 :=
by sorry

end find_phi_l145_145152


namespace Suraj_new_average_l145_145237

theorem Suraj_new_average (A : ℕ) (h1 : 16 * A + 112 = 17 * (A + 6)) : A + 6 = 16 :=
by
  have hA : A = 10 := by
    linarith
  rw hA
  linarith

end Suraj_new_average_l145_145237


namespace inscribed_circle_radius_l145_145292

-- Define the sides of triangle DEF
def DE : ℝ := 26
def DF : ℝ := 15
def EF : ℝ := 17

-- Define the semiperimeter s
def s : ℝ := (DE + DF + EF) / 2

-- Define the area K using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius r of the inscribed circle
def r : ℝ := K / s

-- The theorem stating the radius of the circle inscribed in triangle DEF
theorem inscribed_circle_radius : r = Real.sqrt 6 := 
  sorry

end inscribed_circle_radius_l145_145292


namespace circle_equation_correct_l145_145165

theorem circle_equation_correct :
  (∃ M : ℝ × ℝ, M.1 * 2 + M.2 - 1 = 0 ∧
                (M.1 - 3)^2 + (M.2 - 0)^2 = 5 ∧ 
                (M.1 - 0)^2 + (M.2 - 1)^2 = 5) →
  ∃ h k r : ℝ, (h = 1) ∧ (k = -1) ∧ (r = sqrt 5) ∧ 
               (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 5) :=
begin
  sorry
end

end circle_equation_correct_l145_145165


namespace natural_number_representation_l145_145632

theorem natural_number_representation (n : ℕ) : ∃ (x y z : ℕ), n = nat.choose x 1 + nat.choose y 2 + nat.choose z 3 ∧ (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) :=
sorry

end natural_number_representation_l145_145632


namespace find_number_of_girls_l145_145115

-- Definitions for the number of candidates
variables (B G : ℕ)
variable (total_candidates : B + G = 2000)

-- Definitions for the percentages of passed candidates
variable (pass_rate_boys : ℝ := 0.34)
variable (pass_rate_girls : ℝ := 0.32)
variable (pass_rate_total : ℝ := 0.331)

-- Hypotheses based on the conditions
variables (P_B P_G : ℝ)
variable (pass_boys : P_B = pass_rate_boys * B)
variable (pass_girls : P_G = pass_rate_girls * G)
variable (pass_total_eq : P_B + P_G = pass_rate_total * 2000)

-- Goal: Prove that the number of girls (G) is 1800
theorem find_number_of_girls (B G : ℕ)
  (total_candidates : B + G = 2000)
  (pass_rate_boys : ℝ := 0.34)
  (pass_rate_girls : ℝ := 0.32)
  (pass_rate_total : ℝ := 0.331)
  (P_B P_G : ℝ)
  (pass_boys : P_B = pass_rate_boys * (B : ℝ))
  (pass_girls : P_G = pass_rate_girls * (G : ℝ))
  (pass_total_eq : P_B + P_G = pass_rate_total * 2000) : G = 1800 :=
sorry

end find_number_of_girls_l145_145115


namespace num_blocks_in_fort_l145_145390

-- Define the dimensions of the fort and the thickness of the walls and floor
def fort_length := 15
def fort_width := 12
def fort_height := 6
def thickness := 2

-- Define the volumes
def volume_original : ℕ := fort_length * fort_width * fort_height
def volume_interior : ℕ := (fort_length - 2 * thickness) * (fort_width - 2 * thickness) * (fort_height - thickness)
def volume_blocks : ℕ := volume_original - volume_interior

-- Prove the number of blocks in the fort is 728
theorem num_blocks_in_fort : volume_blocks = 728 := 
by
  rw [volume_original, volume_interior, volume_blocks],
  norm_num,
  sorry

end num_blocks_in_fort_l145_145390


namespace number_of_b_values_l145_145680

theorem number_of_b_values (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), ∀ (x : ℤ), x^2 + b * x + 6 ≤ 0 ↔ x = x1 ∨ x = x2 ∨ x = x3) ↔ (b = -6 ∨ b = -5 ∨ b = 5 ∨ b = 6) :=
by
  sorry

end number_of_b_values_l145_145680


namespace bike_price_l145_145181

variable (p : ℝ)

def percent_upfront_payment : ℝ := 0.20
def upfront_payment : ℝ := 200

theorem bike_price (h : percent_upfront_payment * p = upfront_payment) : p = 1000 := by
  sorry

end bike_price_l145_145181


namespace a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l145_145467

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ := (3 * n * (n + 1) / 2) + (2^n - 1)

theorem a_n_is_arithmetic_sequence (n : ℕ) :
  (a_n 1 = 3) ∧ (a_n 4 = 12) ∧ (∀ n : ℕ, a_n n = 3 * n) :=
by
  sorry

theorem b_n_is_right_sequence (n : ℕ) :
  (b_n 1 = 4) ∧ (b_n 4 = 20) ∧ (∀ n : ℕ, b_n n = 3 * n + 2^(n - 1)) ∧ 
  (∀ n : ℕ, b_n n - a_n n = 2^(n - 1)) :=
by
  sorry

theorem sum_first_n_terms_b_n (n : ℕ) :
  S_n n = 3 * (n * (n + 1) / 2) + 2^n - 1 :=
by
  sorry

end a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l145_145467


namespace greatest_divisor_540_180_under_60_l145_145290

theorem greatest_divisor_540_180_under_60 : ∃ d, d ∣ 540 ∧ d ∣ 180 ∧ d < 60 ∧ ∀ k, k ∣ 540 → k ∣ 180 → k < 60 → k ≤ d :=
by
  sorry

end greatest_divisor_540_180_under_60_l145_145290


namespace part1_part2_l145_145508

variables (a b c x x1 x2 x0 : ℝ)

-- Conditions as definitions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def roots (a b c x1 x2 : ℝ) : Prop :=
x1 ≠ x2 ∧ 0 < x1 ∧ x1 < x2 ∧ x2 < (1 / a) ∧ quadratic_function a b c x1 = x1 ∧ quadratic_function a b c x2 = x2

-- First part of the proof
theorem part1 (h1 : a > 0) (h2 : roots a b c x1 x2) (h3 : x ∈ set.Ioo 0 x1) : 
  quadratic_function a b c x < x1 := 
sorry

-- Second part of the proof
theorem part2 (h1 : a > 0) (h2 : roots a b c x1 x2) : 
  -b / (2 * a) < x1 / 2 := 
sorry

end part1_part2_l145_145508


namespace prove_inequality_l145_145220

variables {a b c A B C k : ℝ}

-- Define the conditions
def conditions (a b c A B C k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ k > 0 ∧
  a + A = k ∧ b + B = k ∧ c + C = k

-- Define the theorem to be proven
theorem prove_inequality (a b c A B C k : ℝ) (h : conditions a b c A B C k) :
  a * B + b * C + c * A ≤ k^2 :=
sorry

end prove_inequality_l145_145220


namespace pyramid_edges_sum_l145_145747

theorem pyramid_edges_sum (a h : ℝ) (hb : a = 12) (hh : h = 15) :
  (let d := 12 * Real.sqrt 2 in
   let s := Real.sqrt (15^2 + (6 * Real.sqrt 2)^2) in
   4 * a + 4 * s = 48 + 12 * Real.sqrt 33) :=
by
  sorry

end pyramid_edges_sum_l145_145747


namespace circle_excluding_points_l145_145026

theorem circle_excluding_points (z ω : ℂ) 
  (h1 : |z - complex.I| = 1) 
  (h2 : z ≠ 0) 
  (h3 : z ≠ 2 * complex.I)
  (h4 : ∀ z ω : ℂ, (ω / (ω - 2 * complex.I)) * ((z - 2 * complex.I) / z) ∈ ℝ) :   
  {z : ℂ | |z - complex.I| = 1 ∧ z ≠ 0 ∧ z ≠ 2 * complex.I} =
  {z : ℂ | (z - complex.I).abs = 1 ∧ z ≠ 0 ∧ z ≠ 2 * complex.I} := 
begin
  sorry
end

end circle_excluding_points_l145_145026


namespace monotonically_decreasing_interval_l145_145069

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

noncomputable def f'' (x : ℝ) : ℝ := -Real.cos (x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f x + f'' x

theorem monotonically_decreasing_interval :
  ∀ (x : ℝ), -7 * Real.pi / 12 ≤ x ∧ x ≤ 5 * Real.pi / 12 → (g x ≤ g (x + δ)) :=
begin
  sorry
end

end monotonically_decreasing_interval_l145_145069


namespace part_a_part_b_l145_145944

open Function

noncomputable def f : ℝ → ℝ := sorry
axiom f_continuous : ContinuousOn f (Ici 0)
axiom f_nonnegative : ∀ x, 0 ≤ f x
axiom f_noninc : ∀ {x y : ℝ}, 0 ≤ x → x < y → f y < f x
axiom f_pos_at_zero : 0 < f 0

theorem part_a : ∃! α : ℝ, 0 < α ∧ f (f α) = α :=
sorry

def seq (x₁ : ℝ) : ℕ → ℝ
| 0     => x₁
| (s + 1) => f (seq s)

theorem part_b (x₁ : ℝ) (hx₁ : 0 ≤ x₁) : 
  ∃ L, Tendsto (seq x₁) atTop (nhds L) :=
sorry

end part_a_part_b_l145_145944


namespace number_of_sequences_l145_145875

theorem number_of_sequences : 
  let letters := ['T', 'R', 'O', 'U', 'B', 'L', 'E']
  -- Define the set of all possible sequences
  let sequences := { s : List Char // ∀ c ∈ s, c ∈ letters ∧ List.Nodup s.toList ∧ List.length s = 4 }
  -- Define the subset of sequences that start with T and do not end with E
  let valid_sequences := { s ∈ sequences // s.head = 'T' ∧ s.last ≠ 'E' }
  -- Prove that the number of valid sequences is 30
  in valid_sequences.card = 30
:= sorry

end number_of_sequences_l145_145875


namespace part1_i_part1_ii_part2_l145_145855

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (m n x : ℝ) := m * x + n
noncomputable def h (m n x : ℝ) := f x - g m n x
noncomputable def r (m x : ℝ) := (1 / f x) + (4 * x / (4 + x))

-- Problem 1 (i)
theorem part1_i (m n : ℝ) (h_tangent : ∀ x : ℝ, h m n x = Real.exp x - (m * x) - n)
    (passes_through : ∀ x y : ℝ, y = (1 - n) + (1 - m) * x → (1, 0)) :
    m + n = 2 := sorry

-- Problem 1 (ii)
theorem part1_ii (m : ℝ) (h_tangent : ∀ x : ℝ, h m 0 x = Real.exp x - (m * x))
    (no_zeros : ∀ x : ℝ, -1 < x → ∃ y : ℝ, h m 0 y ≠ 0) :
    m ∈ Set.Ico (-1 / Real.exp 1) Real.exp 1 := sorry

-- Problem 2
theorem part2 (m : ℝ) (h_r : ∀ x : ℝ, r m x = (1 / Real.exp x) + (4 * x / (x + 4)))
    (m_pos : 0 < m) :
    ∀ x : ℝ, 0 ≤ x → r m x ≥ 1 := sorry

end part1_i_part1_ii_part2_l145_145855


namespace ones_digits_of_numbers_divisible_by_8_count_of_possible_ones_digits_divisible_by_8_l145_145980

theorem ones_digits_of_numbers_divisible_by_8 :
  ∀ (n : ℕ), (∃ k : ℕ, n = k * 8) → 
    (∃ d : ℕ, d ∈ {0, 2, 4, 6, 8} ∧ (n % 10 = d)) :=
sorry

theorem count_of_possible_ones_digits_divisible_by_8 :
  (∃ (count : ℕ), count = 5) :=
sorry

end ones_digits_of_numbers_divisible_by_8_count_of_possible_ones_digits_divisible_by_8_l145_145980


namespace number_of_possible_sets_l145_145464

theorem number_of_possible_sets :
  let U := {1, 2, 3, 4, 5, 6}
  let condition (M : Set ℕ) : Prop := ∅ ⊆ M ∧ M ⊆ U ∧ ∀ a ∈ M, (6 - a) ∈ M
  (count (filter condition (powerset U)) = 7) :=
by 
  let U := {1, 2, 3, 4, 5, 6}
  let condition (M : Set ℕ) : Prop := ∅ ⊆ M ∧ M ⊆ U ∧ ∀ a ∈ M, (6 - a) ∈ M
  have h : (count (filter condition (powerset U)) = 7)
  exact h
  sorry

end number_of_possible_sets_l145_145464


namespace exist_l_N_l145_145943

def sequence (a : ℕ → ℝ) (s : ℕ) := 
  ∀ n, n > s → a n = (finset.range (n - 1)).sup (λ k, a k.succ + a (n - k))

theorem exist_l_N (a : ℕ → ℝ) (s : ℕ) 
  (h : sequence a s) :
  ∃ ℓ N, 0 < ℓ ∧ ℓ ≤ s ∧ ∀ n, n ≥ N → a n = a ℓ + a (n - ℓ) :=
sorry

end exist_l_N_l145_145943


namespace intersection_complement_l145_145081

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 > 4}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 3) / (x + 1) < 0}

-- Complement of N in U
def complement_N : Set ℝ := {x : ℝ | x <= -1} ∪ {x : ℝ | x >= 3}

-- Final proof to show intersection
theorem intersection_complement :
  M ∩ complement_N = {x : ℝ | x < -2} ∪ {x : ℝ | x >= 3} :=
by
  sorry

end intersection_complement_l145_145081


namespace molecular_weight_of_CH3COOH_l145_145291

def MolecularWeight (total_mass : ℝ) (num_moles : ℝ) : ℝ :=
  total_mass / num_moles

theorem molecular_weight_of_CH3COOH :
  (MolecularWeight 360 6) = 60 := 
by
  sorry

end molecular_weight_of_CH3COOH_l145_145291


namespace gnome_problem_l145_145265

theorem gnome_problem : 
  ∀ (total_gnomes red_hats big_noses blue_big_noses red_big_noses red_small_noses : ℕ),
  total_gnomes = 28 →
  red_hats = (total_gnomes * 3) / 4 →
  big_noses = total_gnomes / 2 →
  blue_big_noses = 6 →
  red_big_noses = big_noses - blue_big_noses →
  red_small_noses = red_hats - red_big_noses →
  red_small_noses = 13 :=
by
  intros total_gnomes red_hats big_noses blue_big_noses red_big_noses red_small_noses
  assume h_total h_red_hats h_big_noses h_blue_big_noses h_red_big_noses h_red_small_noses
  sorry

end gnome_problem_l145_145265


namespace average_between_10_and_86_div_by_9_l145_145773

noncomputable def average_of_divisible_by_9 (a b : ℕ) (cond : ℕ → ℕ → Prop) : ℚ :=
  let nums := {n : ℕ | n > a ∧ n < b ∧ cond n 9} 
  if nums ≠ ∅ then nums.sum / nums.card else 0

theorem average_between_10_and_86_div_by_9 : average_of_divisible_by_9 10 86 (λ n d, n % d = 0) = 49.5 := 
by 
  -- Proof goes here
  sorry

end average_between_10_and_86_div_by_9_l145_145773


namespace part1_part2_l145_145484

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem part1 : {x : ℝ | f x < 2} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, f x ≤ a - a^2 / 2) → (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l145_145484


namespace profit_is_eight_dollars_l145_145352

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l145_145352


namespace proof_triangle_c_value_l145_145777

-- Definition and condition for the existence of the triangle
def triangle_exists (a b c : ℝ) (A B C : ℝ) : Prop :=
  sin A = sqrt 3 * sin B ∧
  C = π / 6 ∧
  ((a * c = sqrt 3) ∨ (c * sin A = 3) ∨ (c = sqrt 3 * b))

-- Proof statement for the value of c
theorem proof_triangle_c_value : 
  ∃ (c : ℝ), 
    (∃ (a b A B C : ℝ), triangle_exists a b c A B C) → 
    (c = 1 ∨ c = 2*sqrt 3) :=
sorry

end proof_triangle_c_value_l145_145777


namespace circle_equation_l145_145160

theorem circle_equation (M : ℝ × ℝ) :
  (∃ a : ℝ, M = (a, 1 - 2 * a) ∧ 2 * M.1 + M.2 - 1 = 0) ∧
  (distance M (3, 0) = distance M (0, 1)) →
  (∃ r : ℝ, (x - 1)^2 + (y + 1)^2 = r^2) :=
begin
  sorry
end

end circle_equation_l145_145160


namespace find_complex_number_l145_145808

-- Define what it means for a complex number z to satisfy the given conditions
def satisfies_conditions (z : ℂ) : Prop :=
  complex.abs (z - 2) = complex.abs (z + 4) ∧ complex.abs (z + 4) = complex.abs (z + 2 * complex.I)

-- Define the complex number -1 + i
def z_answer : ℂ := -1 + complex.I

-- State the main theorem that we need to prove
theorem find_complex_number : ∃ z : ℂ, satisfies_conditions z ∧ z = z_answer :=
  sorry

end find_complex_number_l145_145808


namespace interest_percent_correct_l145_145709

-- Defining the conditions given in the problem
def purchase_price : ℝ := 127
def down_payment : ℝ := 27
def monthly_payment : ℝ := 10
def number_of_payments : ℕ := 12

-- Defining the total amount paid using the payment plan
def total_paid : ℝ := down_payment + (monthly_payment * number_of_payments)

-- Defining the interest paid
def interest_paid : ℝ := total_paid - purchase_price

-- Defining the interest percent
def interest_percent : ℝ := (interest_paid / purchase_price) * 100

-- The rounded interest percent to the nearest tenth
def interest_percent_rounded : ℝ := Float.round(interest_percent * 10) / 10

-- Proof problem: prove that the interest percent paid is approximately 15.7% after rounding to the nearest tenth
theorem interest_percent_correct : interest_percent_rounded = 15.7 := by
  sorry

end interest_percent_correct_l145_145709


namespace arctan_sum_l145_145894

theorem arctan_sum (a b : ℝ) (h1 : a = 1/3) (h2 : (a + 1) * (b + 1) = 3) : 
  Real.arctan a + Real.arctan b = Real.arctan (19 / 7) :=
by
  sorry

end arctan_sum_l145_145894


namespace perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l145_145591

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

def set_of_perfect_squares : set ℕ :=
  {n | is_perfect_square n}

theorem perfect_squares_closed_under_multiplication :
  ∀ a b ∈ set_of_perfect_squares, (a * b) ∈ set_of_perfect_squares :=
by
  sorry

theorem perfect_squares_not_closed_under_addition :
  ∃ a b ∈ set_of_perfect_squares, ¬ (a + b) ∈ set_of_perfect_squares :=
by
  sorry

theorem perfect_squares_not_closed_under_subtraction :
  ∃ a b ∈ set_of_perfect_squares, ¬ (a - b) ∈ set_of_perfect_squares :=
by
  sorry

theorem perfect_squares_not_closed_under_division :
  ∃ a b ∈ set_of_perfect_squares, ¬ ∃ q : ℕ, a = q * b :=
by
  sorry

end perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l145_145591


namespace arithmetic_sequence_S6_by_S4_l145_145463

-- Define the arithmetic sequence and the sum function
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def S1 : ℕ := 1
def r (S2 S4 : ℕ) : Prop := S4 / S2 = 4

-- Proof statement
theorem arithmetic_sequence_S6_by_S4 :
  ∀ (a d : ℕ), 
  (sum_arithmetic_sequence a d 1 = S1) → (r (sum_arithmetic_sequence a d 2) (sum_arithmetic_sequence a d 4)) → 
  (sum_arithmetic_sequence a d 6 / sum_arithmetic_sequence a d 4 = 9 / 4) := 
by
  sorry

end arithmetic_sequence_S6_by_S4_l145_145463


namespace f_2016_equals_cos_l145_145954

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.cos x
| (n + 1) := λ x, (f n)' x

theorem f_2016_equals_cos (x : ℝ) : f 2016 x = Real.cos x := by
  sorry

end f_2016_equals_cos_l145_145954


namespace line_intersects_extension_of_segment_l145_145076

theorem line_intersects_extension_of_segment
  (A B C x1 y1 x2 y2 : ℝ)
  (hnz : A ≠ 0 ∨ B ≠ 0)
  (h1 : (A * x1 + B * y1 + C) * (A * x2 + B * y2 + C) > 0)
  (h2 : |A * x1 + B * y1 + C| > |A * x2 + B * y2 + C|) :
  ∃ t : ℝ, t ≥ 0 ∧ l * (t * (x2 - x1) + x1) + m * (t * (y2 - y1) + y1) = 0 :=
sorry

end line_intersects_extension_of_segment_l145_145076


namespace Petya_higher_chance_l145_145197

theorem Petya_higher_chance :
  ∀ (P_vasya P_petya : ℝ), P_vasya = 0.46 → P_petya = 0.54 → P_vasya < P_petya :=
by
  intros P_vasya P_petya h1 h2
  rw [h1, h2]
  linarith

end Petya_higher_chance_l145_145197


namespace inclination_angle_of_line_l145_145245

theorem inclination_angle_of_line : 
  let line_eq (x y : ℝ) := x + sqrt 3 * y - 1 = 0
  let slope := - (sqrt 3) / 3
  let α := real.arctan slope
  α = 5 * real.pi / 6 := 
begin
  sorry
end

end inclination_angle_of_line_l145_145245


namespace total_distance_thrown_l145_145605

theorem total_distance_thrown (D : ℝ) (total_distance : ℝ) 
  (h1 : total_distance = 20 * D + 60 * D) : 
  total_distance = 1600 := 
by
  sorry

end total_distance_thrown_l145_145605


namespace geometric_sequence_ratio_l145_145032

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
(h_arith : 2 * a 1 * q = a 0 + a 0 * q * q) :
  q = 2 + Real.sqrt 3 ∨ q = 2 - Real.sqrt 3 := 
by
  sorry

end geometric_sequence_ratio_l145_145032


namespace centroids_coincide_l145_145546

-- Define vertices of the hexagon
variables {A1 A2 A3 A4 A5 A6 : Point}

-- Define midpoints of the sides
def M1 := midpoint A1 A2
def M2 := midpoint A2 A3
def M3 := midpoint A3 A4
def M4 := midpoint A4 A5
def M5 := midpoint A5 A6
def M6 := midpoint A6 A1

-- Define triangles formed by these midpoints
def triangle1 := triangle M1 M3 M5
def triangle2 := triangle M2 M4 M6

-- Define centroids of the triangles
def centroid1 := centroid triangle1
def centroid2 := centroid triangle2

-- Prove that the centroids of the two triangles are the same
theorem centroids_coincide : centroid1 = centroid2 :=
sorry

end centroids_coincide_l145_145546


namespace num_multiples_of_3_labels_l145_145552

theorem num_multiples_of_3_labels : 
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ {m n : ℕ}, (m, n) ∈ pairs ↔ 
                  m ≠ n ∧ m + n ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ (m + n) % 3 = 0) ∧
    pairs.card = 10 :=
sorry

end num_multiples_of_3_labels_l145_145552


namespace trajectory_point_P_ellipse_l145_145646

noncomputable def pointO : Type := sorry
noncomputable def pointQ : pointO := sorry
noncomputable def pointA : pointO := sorry

def circle : set pointO := sorry
def is_center (O : pointO) (c : set pointO) : Prop := sorry
def on_circumference (A : pointO) (c : set pointO) : Prop := sorry
def fixed_point (Q : pointO) (c : set pointO) : Prop := sorry
def across_fold (A Q O : pointO) : Prop := sorry
def find_trajectory (P A : pointO) (l : list pointO) : pointO := sorry

theorem trajectory_point_P_ellipse
  (O Q : pointO)
  (circ : set pointO)
  (hO : is_center O circ)
  (hQ : fixed_point Q circ)
  (hA : ∀ (A : pointO), on_circumference A circ) :
  find_trajectory P A (circ.fold_list) = ellipse :=
by
  sorry

end trajectory_point_P_ellipse_l145_145646


namespace profit_percentage_l145_145315

theorem profit_percentage (SP CP : ℝ) (h₁ : SP = 300) (h₂ : CP = 250) : ((SP - CP) / CP) * 100 = 20 := by
  sorry

end profit_percentage_l145_145315


namespace geometrical_shapes_OABC_l145_145459

/-- Given distinct points A(x₁, y₁), B(x₂, y₂), and C(2x₁ - x₂, 2y₁ - y₂) on a coordinate plane
    and the origin O(0,0), determine the possible geometrical shapes that the figure OABC can form
    among these three possibilities: (1) parallelogram (2) straight line (3) rhombus.
    
    Prove that the figure OABC can form either a parallelogram or a straight line,
    but not a rhombus.
-/
theorem geometrical_shapes_OABC (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂) ∧ (x₂, y₂) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂)) :
  (∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁) ∨
  (2 * x₁ = x₁ + x₂ ∧ 2 * y₁ = y₁ + y₂) :=
sorry

end geometrical_shapes_OABC_l145_145459


namespace intersection_A_N_is_012_l145_145866

def A : Set ℝ := {x | 3^x ≤ 10}
def N : Set ℝ := {n | ∃ (m : ℕ), n = m}

theorem intersection_A_N_is_012 : A ∩ N = {0, 1, 2} := by
  sorry

end intersection_A_N_is_012_l145_145866


namespace y_intercept_of_line_l145_145673

theorem y_intercept_of_line (slope : ℝ) (x_intercept : ℝ × ℝ) : x_intercept = (7, 0) → slope = 3 → ∃ y_intercept : ℝ × ℝ, y_intercept = (0, -21) :=
by
  intros hx hs
  use (0, -21)
  sorry

end y_intercept_of_line_l145_145673


namespace reflect_y_axis_matrix_l145_145423

theorem reflect_y_axis_matrix : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, (
    (∀ v : (Fin 2 → ℝ), v = ![1, 0] → A.mulVec v = ![-1, 0]) ∧ 
    (∀ v : (Fin 2 → ℝ), v = ![0, 1] → A.mulVec v = ![0, 1])
  ) ∧ A = ![![-1, 0], ![0, 1]] :=
begin
  sorry
end

end reflect_y_axis_matrix_l145_145423


namespace triangle_angle_impossibility_l145_145306

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∀ (A B C D E : Point) 
  (h : ¬collinear A B C),
  between A D C ∧ between A E C ∧ dist A D = dist D E ∧ dist D E = dist E C →
  ¬ (angle_eq (angle A B D) (angle D B E) ∧ angle_eq (angle D B E) (angle E B C))

-- considering the triangle and conditions as described
theorem triangle_angle_impossibility (A B C D E : Point) (h_noncollinear: ¬collinear A B C) :
  between A D C → between A E C → dist A D = dist D E → dist D E = dist E C →
  ¬ (angle_eq (angle A B D) (angle D B E) ∧ angle_eq (angle D B E) (angle E B C)) :=
sorry

end triangle_angle_impossibility_l145_145306


namespace largest_number_in_set_l145_145897

theorem largest_number_in_set (b : ℤ) (h : b = -3) :
  ∃ x ∈ ({-3 * b, 5 * b, 36 / b, b ^ 2, 2}.map (λ x, x.to_int)), 
    (∀ y ∈ ({-3 * b, 5 * b, 36 / b, b ^ 2, 2}.map (λ x, x.to_int)), y ≤ x) ∧ (x = -3 * b ∨ x = b ^ 2) := 
by
  sorry

end largest_number_in_set_l145_145897


namespace exists_tangential_quadrilateral_on_circle_l145_145601

theorem exists_tangential_quadrilateral_on_circle 
    (K : Circle) 
    (A B C : K.Point) 
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A) : 
    ∃ D : K.Point, (ABCD : ConvexQuadrilateral K) ∧ is_tangential_ABCD (ABCD) :=
by
  sorry

end exists_tangential_quadrilateral_on_circle_l145_145601


namespace prime_gt_100_exists_ab_l145_145993

theorem prime_gt_100_exists_ab (p : ℕ) (r : ℤ) (hp : 100 < p) (prime_p : Nat.Prime p) : 
  ∃ a b : ℤ, p ∣ (a^2 + b^5 - r) :=
sorry

end prime_gt_100_exists_ab_l145_145993


namespace capital_left_fraction_l145_145322

theorem capital_left_fraction (C : ℝ) :
  let after_raw_material := C - (1/4) * C in
  let after_machinery := after_raw_material - (1/10) * after_raw_material in
  after_machinery = (27/40) * C := 
by sorry

end capital_left_fraction_l145_145322


namespace quadratic_eq_two_distinct_real_roots_find_m_value_l145_145494

noncomputable def quadratic_eq_discriminant {m : ℝ} : ℝ :=
    (-(2 * m + 1)) ^ 2 - 4 * 1 * (m ^ 2 + m)

theorem quadratic_eq_two_distinct_real_roots (m : ℝ) : quadratic_eq_discriminant m > 0 :=
    by
    sorry

-- Let's define the conditions for the second part
def quadratic_eq_roots (m a b : ℝ) :=
    a + b = 2 * m + 1 ∧ a * b = m ^ 2 + m

def equation_holds (a b : ℝ) :=
    (2 * a + b) * (a + 2 * b) = 20

theorem find_m_value (m a b : ℝ) (h : quadratic_eq_roots m a b) (h_eq : equation_holds a b) :
    m = -2 ∨ m = 1 :=
    by
    sorry

end quadratic_eq_two_distinct_real_roots_find_m_value_l145_145494


namespace equal_abc_l145_145009

theorem equal_abc {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ 
       b^2 * (c + a - b) = c^2 * (a + b - c)) : a = b ∧ b = c :=
by
  sorry

end equal_abc_l145_145009


namespace bus_passenger_count_l145_145722

-- Definition of the function f representing the number of passengers per trip
def passengers (n : ℕ) : ℕ :=
  120 - 2 * n

-- The total number of trips is 18 (from 9 AM to 5:30 PM inclusive)
def total_trips : ℕ := 18

-- Sum of passengers over all trips
def total_passengers : ℕ :=
  List.sum (List.map passengers (List.range total_trips))

-- Problem statement
theorem bus_passenger_count :
  total_passengers = 1854 :=
sorry

end bus_passenger_count_l145_145722


namespace petya_higher_chance_of_winning_l145_145211

/-- Petya gives Vasya two boxes of candies. Each box contains both chocolate candies and caramel candies.
    In total, there are 25 candies in both boxes. Vasya takes one candy from each box.
    If both candies turn out to be chocolate, Vasya wins. Otherwise, Petya wins.
    The probability that Vasya will get two caramels is 0.54.
    Prove that Petya has a higher chance of winning. --/
theorem petya_higher_chance_of_winning
  (total_candies : ℕ)
  (total_candies = 25 : Prop) 
  (prob_two_caramels : ℝ)
  (prob_two_caramels = 0.54 : Prop) :
  0.46 < 0.54 := 
by 
  sorry

end petya_higher_chance_of_winning_l145_145211


namespace no_solution_l145_145231

theorem no_solution : ¬∃ x : ℝ, x^3 - 8*x^2 + 16*x - 32 / (x - 2) < 0 := by
  sorry

end no_solution_l145_145231


namespace wheel_speed_l145_145590

theorem wheel_speed (r : ℝ) (t : ℝ) :
  let circumference_in_miles := (13 : ℝ) / 5280
  let time_conversion := 3600 : ℝ
  let time_decrease := 1 / 14400

  (r * t = circumference_in_miles * time_conversion) ∧
  ((r + 3) * (t - time_decrease) = circumference_in_miles * time_conversion) →
  r = 12 :=
by
  sorry

end wheel_speed_l145_145590


namespace find_x_in_PetyaSequence_l145_145990

noncomputable def PetyaSequence (n : ℕ) : Prop :=
n ≥ 2 ∧ 
(∀ k, (0 ≤ k ∧ k < n → ∀ d, d ≠ 4 ∧ (to_list (k / 10 : ℕ).digit.to_string).get 1 d ≠ some '4')) ∧
(∀ d, (to_list ((n - 1) / 10 : ℕ).digit.to_string).get 1 d ≠ some '7) ∧
(∃ a b : ℕ, Prime a ∧ Prime b ∧ a ≠ b ∧ b = a + 4 ∧ (10 ∣ (((a + b) / 2) - 5)) ∧
 ∃ x : ℕ, x = a * b ∧ x % 100 = 21 ∧ x = 2021)

theorem find_x_in_PetyaSequence (n : ℕ) (h : PetyaSequence n) : ∃ x : ℕ, x = 2021 := by
  sorry

end find_x_in_PetyaSequence_l145_145990


namespace set_intersection_l145_145868

-- Define set A
def setA : Set ℝ := { y : ℝ | y = Real.sin x → x ∈ ℝ }

-- Define set B
def setB : Set ℝ := { x : ℝ | (1/9) < (1/3)^x ∧ (1/3)^x < 3 }

-- Define the intersection of set A and set B and the expected result
theorem set_intersection : 
  (∃ y, y ∈ setA ∧ y ∈ setB) = (Icc (-1.0) 1.0) :=
sorry

end set_intersection_l145_145868


namespace first_candidate_more_gain_l145_145338

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l145_145338


namespace no_such_function_exists_l145_145123

open Set Real

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ContinuousOn f (Icc 0 1) ∧
  (∀ x ∈ Icc 0 1, (x ∈ ℚ → f x ∉ ℚ) ∧ (x ∉ ℚ → f x ∈ ℚ)) ∧
  (∀ x ∈ Icc 0 1, f x ∈ Icc 0 1) :=
by
  sorry

end no_such_function_exists_l145_145123


namespace cubic_polynomial_sum_l145_145323

theorem cubic_polynomial_sum (q : ℝ → ℝ) 
  (h_cubic : ∃ a b c d, ∀ x, q x = a*x^3 + b*x^2 + c*x + d)
  (h_q1 : q 1 = 6)
  (h_q5 : q 5 = 10)
  (h_q10 : q 10 = -2)
  (h_q14 : q 14 = 2) : 
  (∑ i in Finset.range 16, q i) = 64 :=
by
  sorry

end cubic_polynomial_sum_l145_145323


namespace part1_part2_part3_l145_145169
noncomputable theory

-- Given conditions
def f (x : ℝ) (a k : ℝ) := k * a^x - a^(-x)
def g (x : ℝ) (a : ℝ) := a^(2*x) + a^(-2*x) - 2 * (a^x - a^(-x))

-- Problem: Under the given conditions prepare the structure for the proofs
theorem part1 (a : ℝ) (h1 : 0 < a ∧ a ≠ 1) (h2 : ∀ x, f x a 1 = -f (-x) a 1) : ∃ k, k = 1 :=
by sorry

theorem part2 (a : ℝ) (h1 : 0 < a ∧ a ≠ 1) (h3 : 4 = a) (h4 : 4 - 4^(-1) = 15 / 4) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 1 ≤ g x a ∧ g x a ≤ 137 / 16 :=
by sorry

theorem part3 (a : ℝ) (h3 : 4 = a) :
  ¬ ∃ λ : ℕ, ∀ x, -1 / 2 ≤ x ∧ x ≤ 1 / 2 → f (2 * x) a 1 ≥ λ * f x a 1 :=
by sorry

end part1_part2_part3_l145_145169


namespace range_of_quadratic_not_contains_neg2_l145_145443

def quadratic_has_no_real_solution {a b c : ℝ} (h : ℝ) : Prop :=
  b^2 - 4 * a * c < h

theorem range_of_quadratic_not_contains_neg2 (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + 2) ≠ -2) ↔ b ∈ Ioo (-4 : ℝ) (4 : ℝ) :=
by 
  sorry

end range_of_quadratic_not_contains_neg2_l145_145443


namespace det_of_mat_is_neg52_l145_145774

-- Define the matrix
def mat : matrix (fin 3) (fin 3) ℤ :=
  ![[2, -6, 6],
    [0, 6, -2],
    [3, -1, 2]]

-- Define the target theorem statement
theorem det_of_mat_is_neg52 : mat.det = -52 := 
sorry

end det_of_mat_is_neg52_l145_145774


namespace count_valid_arrangements_l145_145349

-- Definitions based on conditions
def employee_on_adjacent_days (schedule : list nat) (A B : nat) : Prop :=
  (schedule.index_of A + 1 = schedule.index_of B) ∨ (schedule.index_of B + 1 = schedule.index_of A)

def does_not_work_on_first_day (schedule : list nat) (C : nat) : Prop :=
  schedule.head? ≠ some C

def does_not_work_on_seventh_day (schedule : list nat) (D : nat) : Prop :=
  schedule.get? 6 ≠ some D

noncomputable def number_of_valid_arrangements : nat :=
  -- The process to calculate this is non-trivial and combinatorial, hence noncomputable.
  sorry

theorem count_valid_arrangements : 
  number_of_valid_arrangements = 1008 :=
begin
  sorry -- The proof would demonstrate the combinatorial counting resulting in 1008
end

end count_valid_arrangements_l145_145349


namespace minimum_attendees_for_cheaper_second_caterer_l145_145187

theorem minimum_attendees_for_cheaper_second_caterer (x : ℕ) :
  (50 + 18 * x > 150 + 15 * x) ↔ (x ≥ 34) := 
begin
  sorry
end

end minimum_attendees_for_cheaper_second_caterer_l145_145187


namespace new_ratio_cooks_waiters_l145_145768

theorem new_ratio_cooks_waiters
  (initial_ratio : ℕ → ℕ → Prop)
  (cooks waiters : ℕ) :
  initial_ratio 9 24 → 
  12 + waiters = 36 →
  initial_ratio 3 8 →
  9 * 4 = 36 :=
by
  intros h1 h2 h3
  sorry

end new_ratio_cooks_waiters_l145_145768


namespace complex_root_exists_l145_145621

noncomputable def polynomial (a : ℕ → ℝ) (n : ℕ) : polynomial ℝ :=
polynomial.X ^ n + ∑ i in finset.range n, polynomial.C (a i) * polynomial.X ^ (n - 1 - i)

theorem complex_root_exists
  (a : ℕ → ℝ)
  (n : ℕ)
  (h_cond : (∑ i in finset.range (n // 2), (-1) ^ i * a (2 * i + 1))^2 +
            (1 + ∑ i in finset.range ((n - 1) // 2), (-1) ^ (i + 1) * a (2 * i + 2))^2 ≤ 1)
  (h_an_nonzero : a n ≠ 0) :
  ∃ z : ℂ, polynomial.eval z (polynomial a n) = 0 ∧ z.im ≠ 0 := sorry

end complex_root_exists_l145_145621


namespace locus_of_midpoints_single_point_l145_145140

theorem locus_of_midpoints_single_point (K : Type) (P : K) (O : K) (r : ℝ) (diam : Set (line_P O r)) :
  (P ∈ diam) → (∀ chord K, chord.contains P → midpoint chord P = P) := 
sorry

end locus_of_midpoints_single_point_l145_145140


namespace find_s_l145_145952

noncomputable def s_value (m : ℝ) : ℝ := m + 16.25

theorem find_s (a b m s : ℝ)
  (h1 : a + b = m) (h2 : a * b = 4) :
  s = s_value m :=
by
  sorry

end find_s_l145_145952


namespace total_turtles_30_l145_145689

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)
variable (total_turtles : ℕ)

def Kris_turtle_condition : Prop :=
  Kris_turtles = Kristen_turtles / 4

def Trey_turtle_condition : Prop :=
  Trey_turtles = 5 * Kris_turtles

def total_turtle_condition : Prop :=
  total_turtles = Kristen_turtles + Kris_turtles + Trey_turtles

theorem total_turtles_30
  (h1 : Kristen_turtles = 12)
  (h2 : Kris_turtle_condition)
  (h3 : Trey_turtle_condition)
  (h4 : total_turtle_condition) :
  total_turtles = 30 :=
by
  sorry

end total_turtles_30_l145_145689


namespace Petya_higher_chance_l145_145196

theorem Petya_higher_chance :
  ∀ (P_vasya P_petya : ℝ), P_vasya = 0.46 → P_petya = 0.54 → P_vasya < P_petya :=
by
  intros P_vasya P_petya h1 h2
  rw [h1, h2]
  linarith

end Petya_higher_chance_l145_145196


namespace shaded_triangle_equilateral_l145_145281

theorem shaded_triangle_equilateral
  (triangle1 triangle2 : Type)
  [is_right_angled_triangle triangle1]
  [is_right_angled_triangle triangle2]
  (identical : triangle1 ≃ triangle2)
  (vertex_placed_on_side : ∃ (A B C : Type) 
                                (right_angle_vertex_of_triangle2 : A) 
                                (side_vertex_of_triangle1 : line B C),
                             vertex_placed_on_right_angle_of_triangle2 triangle1 triangle2 A B C)
   : let shaded_triangle : triangle := by {
       obtain ⟨A, B, C, right_angle_vertex_of_triangle2, side_vertex_of_triangle1⟩ := vertex_placed_on_side,
       exact ⟨A, B, C⟩ }
   in is_equilateral shaded_triangle := by {
     exact sorry
   }

end shaded_triangle_equilateral_l145_145281


namespace quadratic_real_roots_and_m_values_l145_145488

theorem quadratic_real_roots_and_m_values (m : ℝ) :
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  in (Δ > 0) ∧
     (∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1)) :=
by
  let Δ := (2 * m + 1) ^ 2 - 4 * (m ^ 2 + m)
  have Δ_pos : Δ > 0 := sorry
  have find_m : ∀ (a b : ℝ), (a + b = 2 * m + 1) → (a * b = m ^ 2 + m) → ((2 * a + b) * (a + 2 * b) = 20) → (m = -2 ∨ m = 1) := sorry
  exact ⟨Δ_pos, find_m⟩

#check quadratic_real_roots_and_m_values

end quadratic_real_roots_and_m_values_l145_145488


namespace unique_real_solution_l145_145802

theorem unique_real_solution (b : ℝ) : 
    ∀ b : ℝ, (∃! x : ℝ, x^3 - b * x^2 - 3 * b * x + b^2 - 2 = 0) :=
begin
  sorry
end

end unique_real_solution_l145_145802


namespace quadratic_root_a_value_l145_145764

theorem quadratic_root_a_value (a k : ℝ) (h1 : k = 65) (h2 : a * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : a = 2 :=
by
  sorry

end quadratic_root_a_value_l145_145764


namespace solution_l145_145825

-- Define the conditions
variable (f : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (f_periodic : ∀ x, f (x + 1) = f (1 - x))
variable (f_cubed : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x ^ 3)

-- Define the goal
theorem solution : f 2019 = -1 :=
by sorry

end solution_l145_145825


namespace injured_player_age_l145_145239

noncomputable def average_age_full_team := 22
noncomputable def number_of_players := 11
noncomputable def average_age_remaining_players := 21
noncomputable def number_of_remaining_players := 10
noncomputable def total_age_full_team := number_of_players * average_age_full_team
noncomputable def total_age_remaining_players := number_of_remaining_players * average_age_remaining_players

theorem injured_player_age :
  (number_of_players * average_age_full_team) -
  (number_of_remaining_players * average_age_remaining_players) = 32 :=
by
  sorry

end injured_player_age_l145_145239


namespace quadratic_roots_eq_d_l145_145669

theorem quadratic_roots_eq_d (d : ℝ) (h : (∀ x, x^2 + 6 * x + d = 0 ↔ x = (-6 + real.sqrt d) / 2 ∨ x = (-6 - real.sqrt d) / 2)) :
  d = 36 / 5 :=
sorry

end quadratic_roots_eq_d_l145_145669


namespace line_through_midpoint_l145_145066

open Real

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

def midpoint (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

def chord_eq (x1 y1 x2 y2 : ℝ) (k : ℝ) : Prop :=
  y1 = k * x1 + b ∧ y2 = k * x2 + b

theorem line_through_midpoint 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse x1 y1) 
  (h2 : ellipse x2 y2) 
  (h3 : midpoint x1 y1 x2 y2) : 
  x - 2 * y - 4 = 0 :=
sorry

end line_through_midpoint_l145_145066


namespace base_angle_in_isosceles_triangle_l145_145547

theorem base_angle_in_isosceles_triangle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_angle_70 : A = 70 ∨ B = 70 ∨ C = 70) :
  (A = 55 ∨ B = 55 ∨ C = 55) ∨ (A = 70 ∨ B = 70).

end base_angle_in_isosceles_triangle_l145_145547


namespace gcd_228_1995_l145_145697

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l145_145697


namespace isosceles_triangle_of_parallel_altitudes_and_bisectors_l145_145960

noncomputable def triangle (A B C H_a H_b L_a L_b : Type) := 
  (is_altitude A H_a C) ∧
  (is_altitude B H_b C) ∧
  (is_angle_bisector A L_a C B) ∧
  (is_angle_bisector B L_b C A) ∧
  (H_aH_b ∥ L_aL_b)

theorem isosceles_triangle_of_parallel_altitudes_and_bisectors
  {A B C H_a H_b L_a L_b : Type} 
  (h : triangle A B C H_a H_b L_a L_b) : AC = BC :=
  sorry

end isosceles_triangle_of_parallel_altitudes_and_bisectors_l145_145960


namespace number_of_valid_x_l145_145585

def product_of_digits (n : ℕ) : ℕ :=
  let f x := x.digits.reverse in
  f n |>.foldr (*) 1

noncomputable def valid_count : ℕ :=
  (List.range' 10 90).countp (λ x,
    let p := product_of_digits x
    in product_of_digits p = 6)

theorem number_of_valid_x : valid_count = 6 :=
  sorry

end number_of_valid_x_l145_145585


namespace covered_area_of_two_congruent_squares_l145_145280

theorem covered_area_of_two_congruent_squares :
  let side_length := 12
  let area_of_one_square := side_length * side_length
  let total_area_without_overlap := 2 * area_of_one_square
  let overlap_side_length := side_length / 2
  let overlap_area := overlap_side_length * overlap_side_length
  let covered_area := total_area_without_overlap - overlap_area
  covered_area = 252 :=
by
  let side_length := 12
  let area_of_one_square := side_length * side_length
  let total_area_without_overlap := 2 * area_of_one_square
  let overlap_side_length := side_length / 2
  let overlap_area := overlap_side_length * overlap_side_length
  let covered_area := total_area_without_overlap - overlap_area
  show covered_area = 252 from sorry

end covered_area_of_two_congruent_squares_l145_145280


namespace point_on_circle_l145_145843

axiom circle (O A : Type) (dist : O → A → ℝ) 
def radius : ℝ := 4
def length_OA : ℝ := 4

theorem point_on_circle {O A : Type} (dist : O → A → ℝ)
  (h₁ : radius = 4) (h₂ : length_OA = 4) : 
  dist O A = radius := by
  sorry

end point_on_circle_l145_145843


namespace smallest_three_digit_sum_l145_145937

/-- Defining the digits of a number for usage in conditions. -/
def digits (n : ℕ) : List ℕ := 
  if n < 10 then [n] else digits (n / 10) ++ [n % 10]

/-- Checking if all digits in a list are unique. -/
def all_unique (l : List ℕ) : Prop := 
  ∀ a ∈ l, ∀ b ∈ l, a = b → a = b

/-- Main theorem stating that under the given conditions, the smallest sum of the three leftmost digits of the sum is 10. -/
theorem smallest_three_digit_sum :
  ∀ x y : ℕ, 
    (100 ≤ x ∧ x ≤ 999) ∧ (100 ≤ y ∧ y ≤ 999) ∧ 
    (digits x).length = 3 ∧ (digits y).length = 3 ∧ 
    (∃ d, d ∈ digits x ∧ d = 0) ∧ 
    (all_unique ((digits x) ++ (digits y))) ∧ 
    (999 < x + y) ∧ (x + y < 10000) → 
    (∃ T : ℕ, T = x + y ∧ 
    let three_leftmost := (digits T).take 3
    in (three_leftmost.sum = 10)) :=
by {
  sorry
}

end smallest_three_digit_sum_l145_145937


namespace quadrilateral_area_correct_l145_145624

-- Define the characteristics of the quadrilateral
structure Quadrilateral :=
(AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
(ABC_angle : ℝ) (ACD_angle : ℝ)

def given_quad : Quadrilateral :=
{ AB := 0, -- unknown
  BC := 0, -- unknown
  CD := 18,
  DA := 0, -- unknown
  AC := 24,
  BD := 0, -- unknown
  ABC_angle := 90,
  ACD_angle := 90
}

-- Define the known relationships in the quadrilateral
noncomputable def quadrilateral_area (q : Quadrilateral) : ℝ :=
let ACD_area := (1 / 2) * q.AC * q.CD in
let [AB, _, BC] := sorry_proof_of_Pythagorean_theorem q.AC in -- Use Pythagorean theorem hypothetically
let ABC_area := (1 / 2 * AB * BC) in
ABC_area + ACD_area

-- Statement to prove
theorem quadrilateral_area_correct (q : Quadrilateral)
  (h1 : q.AC = 24) (h2 : q.CD = 18) (h3 : q.ABC_angle = 90) (h4 : q.ACD_angle = 90) 
(h5 : sorry_proof_that_diagonals_intersect_at_point_E)
: quadrilateral_area q = 351 :=
by sorry 

end quadrilateral_area_correct_l145_145624


namespace total_pages_read_l145_145984

theorem total_pages_read (days : ℕ)
  (deshaun_books deshaun_pages_per_book lilly_percent ben_extra eva_factor sam_pages_per_day : ℕ)
  (lilly_percent_correct : lilly_percent = 75)
  (ben_extra_correct : ben_extra = 25)
  (eva_factor_correct : eva_factor = 2)
  (total_break_days : days = 80)
  (deshaun_books_correct : deshaun_books = 60)
  (deshaun_pages_per_book_correct : deshaun_pages_per_book = 320)
  (sam_pages_per_day_correct : sam_pages_per_day = 150) :
  deshaun_books * deshaun_pages_per_book +
  (lilly_percent * deshaun_books * deshaun_pages_per_book / 100) +
  (deshaun_books * (100 + ben_extra) / 100) * 280 +
  (eva_factor * (deshaun_books * (100 + ben_extra) / 100 * 280)) +
  (sam_pages_per_day * days) = 108450 := 
sorry

end total_pages_read_l145_145984


namespace equal_magnitudes_necessary_not_sufficient_l145_145648

-- Non-zero vectors definition
variables {V : Type*} [InnerProductSpace ℝ V]

-- If two non-zero vectors have equal magnitudes, this condition is necessary but not sufficient for these vectors to be equal.
theorem equal_magnitudes_necessary_not_sufficient (u v : V) (hu : u ≠ 0) (hv : v ≠ 0) :
  (∥u∥ = ∥v∥) ↔ (necessary_condition_for_vectors_to_be_equal u v ∧ ¬sufficient_condition_for_vectors_to_be_equal u v) :=
sorry

-- Definitions for necessary and sufficient conditions based on the problem context
def necessary_condition_for_vectors_to_be_equal (u v : V) : Prop := ∥u∥ = ∥v∥
def sufficient_condition_for_vectors_to_be_equal (u v : V) : Prop := u = v

end equal_magnitudes_necessary_not_sufficient_l145_145648


namespace positive_sum_inequality_l145_145151

open Real

theorem positive_sum_inequality
  (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ i, 0 < x i) : 
  (∑ i : Fin n, x i ^ 2 / x (if h : i + 1 < n then Fin.mk (i + 1) h else Fin.mk 0 (by linarith))) 
  ≥ ∑ i : Fin n, x i :=
by
  sorry

end positive_sum_inequality_l145_145151


namespace coin_flip_sequences_l145_145442

theorem coin_flip_sequences (n : ℕ) (h_pos : 0 < n) (p : ℚ) (h_p : 0 < p ∧ p < 1) :
  (∃ a : ℕ, ∃ b : ℕ, a < b ∧ b ≠ 0 ∧ p = a / b ∧ gcd(a, b) = 1) ∧
  (∃ a_r : fin (n + 1) → ℕ, (∀ r, 0 ≤ a_r r ∧ a_r r ≤ nat.choose n r) ∧
     (∑ r in finset.range (n + 1), (a_r r) * (p ^ r) * ((1 - p) ^ (n - r))) = 1 / 2)
    → p = 1 / 2 :=
begin
  sorry
end

end coin_flip_sequences_l145_145442


namespace farmer_ploughing_problem_l145_145305

theorem farmer_ploughing_problem (A D : ℕ) (h1 : A = 120 * D) (h2 : A - 40 = 85 * (D + 2)) : 
  A = 720 ∧ D = 6 :=
by
  sorry

end farmer_ploughing_problem_l145_145305


namespace smallest_value_condition_l145_145962

theorem smallest_value_condition (n : ℕ) (h : n > 3) :
  (∀ (A B : set ℕ) (hAB : S_n = A ∪ B) (hdisjoint : A ∩ B = ∅), 
    ∃ (a b c ∈ A) (a_eq_b : a = b), a * b = c ∨ 
    ∃ (a b c ∈ B) (a_eq_b : a = b), a * b = c) 
  ↔ n = 243 :=
sorry

end smallest_value_condition_l145_145962


namespace intersection_eq_expected_result_l145_145139

def M := { x : ℝ | x - 2 > 0 }
def N := { x : ℝ | (x - 3) * (x - 1) < 0 }
def expected_result := { x : ℝ | 2 < x ∧ x < 3 }

theorem intersection_eq_expected_result : M ∩ N = expected_result := 
by
  sorry

end intersection_eq_expected_result_l145_145139


namespace find_norm_b_l145_145084

open Real

noncomputable def sqrt : ℝ → ℝ := λ x, sorry

variables (a b : ℝ → ℝ)
variables (v w : ℝ)
variables (θ : ℝ)
variables (norm_a : ℝ)
variables (norm_2a_sub_b : ℝ)
variables (cos_θ : ℝ)
variables (dot_product : ℝ → ℝ → ℝ)

-- Conditions
def vec_angle : Prop := θ = π / 4
def norm_a_is_1 : Prop := ‖a‖ = 1
def norm_2a_sub_b_is_sqrt10 : Prop := ‖2 • a - b‖ = sqrt 10

-- Question
def norm_b : ℝ := ‖b‖

-- Correct Answer
def target : Prop := norm_b = 3 * sqrt 2

theorem find_norm_b (h1 : vec_angle) (h2 : norm_a_is_1) (h3 : norm_2a_sub_b_is_sqrt10) : target :=
sorry

end find_norm_b_l145_145084


namespace value_of_k_parallel_vectors_l145_145049

theorem value_of_k_parallel_vectors :
  (a : ℝ × ℝ) → (b : ℝ × ℝ) → (k : ℝ) →
  a = (2, 1) → b = (-1, k) → 
  (a.1 * b.2 - a.2 * b.1 = 0) →
  k = -(1/2) :=
by
  intros a b k ha hb hab_det
  sorry

end value_of_k_parallel_vectors_l145_145049


namespace saved_percent_correct_l145_145723

noncomputable def price_kit : ℝ := 144.20
noncomputable def price1 : ℝ := 21.75
noncomputable def price2 : ℝ := 18.60
noncomputable def price3 : ℝ := 23.80
noncomputable def price4 : ℝ := 29.35

noncomputable def total_price_individual : ℝ := 2 * price1 + 2 * price2 + price3 + 2 * price4
noncomputable def amount_saved : ℝ := total_price_individual - price_kit
noncomputable def percent_saved : ℝ := 100 * (amount_saved / total_price_individual)

theorem saved_percent_correct : percent_saved = 11.64 := by
  sorry

end saved_percent_correct_l145_145723


namespace sum_of_real_roots_is_correct_l145_145434

noncomputable def sum_of_real_roots_of_equation : ℝ :=
  let f := λ x : ℝ, x ^ (Real.log x / Real.log 2)
  if h : 0 < 2 then ∑ x in {x : ℝ | f x = 2}.to_finset, x else 0

theorem sum_of_real_roots_is_correct :
  sum_of_real_roots_of_equation = 2.5 := by sorry

end sum_of_real_roots_is_correct_l145_145434


namespace number_of_valid_functions_l145_145134

-- Define the condition for the function f
def valid_function (n : ℕ) (f : Fin n → Fin 5) : Prop :=
  ∀ k : Fin (n-1), |f (k + 1) - f k| ≥ 3

-- Define the number of valid functions recursively
noncomputable def num_valid_functions : ℕ → ℕ
| 2     := 6
| 3     := 10
| (n+1) := num_valid_functions n + num_valid_functions (n - 1)

-- The theorem which states that the number of valid functions follows a specific pattern
theorem number_of_valid_functions (n : ℕ) (h : n ≥ 2) :
  ∃ x : ℕ, x = num_valid_functions n ∧ (
  (x = num_valid_functions (n - 1) + num_valid_functions (n - 2)) ∧
  (num_valid_functions 2 = 6 ∧ num_valid_functions 3 = 10)) :=
begin
  existsi num_valid_functions n,
  split,
  { reflexivity },
  { split,
    { exact rfl },
    { split; norm_num } },
  sorry
end

end number_of_valid_functions_l145_145134


namespace reasoning_is_deductive_l145_145287

-- Define the lines and their parallel relationships
variables (a b c : Type) [parallel : a ∣∣ b] [parallel : b ∣∣ c]

-- State that the reasoning of parallel lines is deductive reasoning
theorem reasoning_is_deductive : deductive reasoning :=
sorry

end reasoning_is_deductive_l145_145287


namespace distance_polar_to_line_is_half_l145_145118

noncomputable def polar_point : ℝ × ℝ := (real.sqrt 2, real.pi / 4)

noncomputable def polar_line (ρ θ : ℝ) : Prop := ρ * real.sin(θ - real.pi / 3) = -real.sqrt 3 / 2

noncomputable def cartesian_point : ℝ × ℝ := (1, 1)

noncomputable def cartesian_line (x y : ℝ) : Prop := 3 * x - real.sqrt 3 * y - 3 = 0

theorem distance_polar_to_line_is_half : 
  let d := 
    abs (3 * cartesian_point.1 - real.sqrt 3 * cartesian_point.2 - 3) / 
    real.sqrt (3^2 + (real.sqrt 3)^2) 
  in d = 1 / 2 :=
by
  sorry

end distance_polar_to_line_is_half_l145_145118


namespace minimum_value_f_l145_145781

noncomputable def f (x : ℝ) : ℝ := max (3 - x) (x^2 - 4 * x + 3)

theorem minimum_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < m + ε) ∧ m = 0 := 
sorry

end minimum_value_f_l145_145781


namespace truck_capacity_bounds_l145_145278

theorem truck_capacity_bounds {n x : ℝ} :
  (∃ (n : ℝ), n trips are taken to transport 40 tons and 
  ∃ (n_plus_5 : ℝ), (n + 5) trips are taken to transport 80 tons and
  ∀ last_trip_loaded, (on all trips, except perhaps the last one in each transport, the truck was fully loaded)) →
  (7 + 3 / 11 ≤ x ∧ x < 8 + 8 / 9) :=
sorry

end truck_capacity_bounds_l145_145278


namespace number_of_equilateral_triangles_l145_145658

noncomputable def count_equilateral_triangles : ℕ :=
  let count_pairs (limit : ℤ) : ℕ := (limit + 1) - (0 : ℤ)
  let total_pairs : ℕ := (10 - (-10) + 1 : ℕ) * count_pairs 10
  2 * total_pairs

theorem number_of_equilateral_triangles :
  count_equilateral_triangles = 660 := by
  sorry

end number_of_equilateral_triangles_l145_145658


namespace no_integer_triplets_satisfying_eq_l145_145800

theorem no_integer_triplets_satisfying_eq (x y z : ℤ) : 3 * x^2 + 7 * y^2 ≠ z^4 := 
by {
  sorry
}

end no_integer_triplets_satisfying_eq_l145_145800


namespace alice_profit_l145_145356

noncomputable def total_bracelets : ℕ := 52
noncomputable def cost_of_materials : ℝ := 3.00
noncomputable def bracelets_given_away : ℕ := 8
noncomputable def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_remaining := total_bracelets - bracelets_given_away;
      total_revenue := bracelets_remaining * price_per_bracelet;
      profit := total_revenue - cost_of_materials
  in profit = 8.00 := 
by 
  sorry

end alice_profit_l145_145356


namespace product_of_remainders_one_is_one_l145_145619

theorem product_of_remainders_one_is_one (a b : ℕ) (h1 : a % 3 = 1) (h2 : b % 3 = 1) : (a * b) % 3 = 1 :=
sorry

end product_of_remainders_one_is_one_l145_145619


namespace staircase_problem_l145_145625

theorem staircase_problem :
  ∃ (n : ℕ), (n > 20) ∧ (n % 5 = 4) ∧ (n % 6 = 3) ∧ (n % 7 = 5) ∧ n = 159 :=
by sorry

end staircase_problem_l145_145625


namespace book_arrangements_l145_145736

theorem book_arrangements (total_books : ℕ) (at_least_in_library : ℕ) (at_least_checked_out : ℕ) 
  (h_total : total_books = 10) (h_at_least_in : at_least_in_library = 2) 
  (h_at_least_out : at_least_checked_out = 3) : 
  ∃ arrangements : ℕ, arrangements = 6 :=
by
  sorry

end book_arrangements_l145_145736


namespace sum_of_perimeters_l145_145607

-- Define points A, B, C, D with given conditions
variables (A B C D : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB BC AD CD : ℕ)
variables (x y : ℕ)

-- Define the conditions on points and distances
axiom AB_eq_8 : AB = 8
axiom BC_eq_22 : BC = 22
axiom AD_eq_CD : AD = CD
axiom AD_cd_int : (∃ x : ℕ, AD = x) ∧ (∃ y : ℕ, y = y)

-- Define the problem
theorem sum_of_perimeters (h : ∀ {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D],
  x^2 - y^2 = 420 ) :
  let x1 := 71 in let x2 := 106 in let x3 := 72 in
  3 * 30 + 2 * (x1 + x2 + x3) = 588 := by sorry

end sum_of_perimeters_l145_145607


namespace gcd_of_factorials_l145_145001

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_of_factorials :
  Nat.gcd (factorial 8) ((factorial 6)^2) = 1440 := by
  sorry

end gcd_of_factorials_l145_145001


namespace sheers_cost_l145_145130

-- defining the conditions
def number_of_windows : ℕ := 3
def drapes_cost : ℝ := 60
def total_cost : ℝ := 300

-- statement of the proof problem
theorem sheers_cost (S : ℝ) :
  (number_of_windows * (S + drapes_cost) = total_cost) → S = 40 :=
by 
suffices h : S = 40 from h
  intro h1
  sorry

end sheers_cost_l145_145130


namespace hexagon_colors_count_l145_145389

def hexagon_coloring (vertices : Fin 6 → Fin 7) : Prop :=
  ∀ (i j : Fin 6), (i ≠ j) →
    (adjacent i j ∨ diagonal i j) -> vertices i ≠ vertices j

/-- The number of ways to color the vertices of a convex hexagon,
    given 7 colors and the condition that no two adjacent or diagonally
    connected vertices share the same color, is 5040. -/
theorem hexagon_colors_count : ∃ (count : ℕ), count = 5040 ∧
  ∃ (vertices : Fin 6 → Fin 7), hexagon_coloring vertices := by
  sorry

noncomputable def adjacent (i j : Fin 6) : Prop :=
  sorry  -- Replace with actual adjacency condition

noncomputable def diagonal (i j : Fin 6) : Prop :=
  sorry  -- Replace with actual diagonal condition

end hexagon_colors_count_l145_145389


namespace no_such_partition_exists_l145_145388

-- Defining the required elements from the problem
def partition_property (n : ℕ) (S : set (set ℕ)) : Prop :=
  (∀ i, S i ≠ ∅) ∧ -- nonempty subsets
  (∀ T ⊂ finset.filter (λ x, x ∈ S) ⊤, T.card = n - 1 → 
    (∃ A ∈ S, disjoint A T ∧ (∑ i in T, i) ∈ A))

-- The problem statement
theorem no_such_partition_exists (n : ℕ) (h : n > 1) : 
  ¬ ∃ S : set (set ℕ), partition_property n S :=
by
  sorry

end no_such_partition_exists_l145_145388


namespace find_y1_l145_145048

theorem find_y1 
  (y1 y2 y3 : ℝ) 
  (h₀ : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h₁ : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = (2 * Real.sqrt 2 - 1) / (2 * Real.sqrt 2) :=
by
  sorry

end find_y1_l145_145048


namespace part_a_part_b_l145_145711

noncomputable theory

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (ξ ζ : Ω → ℝ) 
-- conditions for part (a)
variable [h1 : Independent ξ ζ]
variable [h2 : IdentDistrib ξ ζ]
variable [h3 : HasExpectation ξ]

theorem part_a :
  ∀ (ω : Ω), 
  E[ξ | ξ + ζ] = E[ζ | ξ + ζ] :=  
begin
  sorry
end

-- conditions for part (b)
variable [h4 : HasFiniteExpectation (ξ^2)]
variable [h5 : HasFiniteExpectation (ζ^2)]
variable [h6 : IdentDistrib ξ (-ξ)]

theorem part_b :
  ∀ (ω : Ω), 
  E[(ξ + ζ)^2 | ξ^2 + ζ^2] = ξ^2 + ζ^2 := 
begin
  sorry
end

end part_a_part_b_l145_145711


namespace intersection_range_l145_145904

theorem intersection_range (k : ℝ) :
  (∃ x y : ℝ, y = k * x + k + 2 ∧ y = -2 * x + 4 ∧ x > 0 ∧ y > 0) ↔ -2/3 < k ∧ k < 2 :=
by
  sorry

end intersection_range_l145_145904


namespace range_of_a_l145_145102

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x - 1 else x ^ 2 + 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (3 / 2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l145_145102


namespace variance_transformed_l145_145038

variables (x1 x2 x3 : ℝ)

def variance (x1 x2 x3 : ℝ) : ℝ :=
  (1 / 3) * ((x1 - (x1 + x2 + x3) / 3)^2 + (x2 - (x1 + x2 + x3) / 3)^2 + (x3 - (x1 + x2 + x3) / 3)^2)

theorem variance_transformed (h : variance x1 x2 x3 = 3) : variance (2 * x1 + 3) (2 * x2 + 3) (2 * x3 + 3) = 12 :=
by
  sorry

end variance_transformed_l145_145038


namespace angle_ABC_possible_values_l145_145469

theorem angle_ABC_possible_values (A B C D E I : Type*) 
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup I]
  (triangle_ABC : ∃ (AD BE : Type*), ∀ I, (AD ∧ BE) ∧ (I ≠ A) ∧ (I ≠ B) ∧ (I ≠ C) ∧ (I ≠ D) ∧ (I ≠ E))
  (angle_CAB_eq_80 : ∀ (A B C : Type*), angle A B C = 80)
  (ID_Eq_IE : ∀ (I D E : Type*), I = D ↔ I = E) :
  ∃ (ABC : ℝ), (angle A B C = 40 ∨ angle A B C = 80) :=
by sorry

end angle_ABC_possible_values_l145_145469


namespace find_d1_l145_145956

-- Define the function F(n) that counts the quadruples
def F (n : ℕ) : ℕ := 
  if n ≥ 3 then 
    (count { (b1, b2, b3, b4) : ℕ × ℕ × ℕ × ℕ |
      b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ b1 ≠ b4 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
      1 ≤ b1 ∧ b1 ≤ n ∧ 1 ≤ b2 ∧ b2 ≤ n ∧ 1 ≤ b3 ∧ b3 ≤ n ∧ 1 ≤ b4 ∧ b4 ≤ n ∧ 
      n ∣ (b1^2 + b2^2 + b3^2 + b4^2) }) 
  else 0

-- Define the polynomial p(x) = d3 * x^3 + d2 * x^2 + d1 * x + d0
def p (x : ℕ) : ℕ := d3 * x^3 + d2 * x^2 + d1 * x + d0

-- The theorem we aim to prove
theorem find_d1 : ∃ d3 d2 d1 d0, (∀ n ≥ 3, F n = p n) ∧ d1 = 8 := 
sorry

end find_d1_l145_145956


namespace summer_camp_sampling_l145_145715

theorem summer_camp_sampling :
  let num_students := 600,
      sample_size := 50,
      first_camp := {x // 1 ≤ x ∧ x ≤ 300},
      second_camp := {x // 301 ≤ x ∧ x ≤ 495},
      third_camp := {x // 496 ≤ x ∧ x ≤ 600},
      y_formula := λ n, 12 * n - 9
  in
  let first_camp_count := ∑ n in Finset.range (50 + 1), if y_formula n ∈ first_camp then 1 else 0,
      second_camp_count := ∑ n in Finset.range (50 + 1), if y_formula n ∈ second_camp then 1 else 0,
      third_camp_count := ∑ n in Finset.range (50 + 1), if y_formula n ∈ third_camp then 1 else 0
  in
  first_camp_count = 25 ∧ second_camp_count = 17 ∧ third_camp_count = 8 :=
by
  sorry

end summer_camp_sampling_l145_145715


namespace central_angle_is_2_radians_l145_145530

namespace CircleAngle

def radius : ℝ := 2
def arc_length : ℝ := 4

theorem central_angle_is_2_radians : arc_length / radius = 2 := by
  sorry

end CircleAngle

end central_angle_is_2_radians_l145_145530


namespace combined_average_age_l145_145645

theorem combined_average_age :
  (8 * 35 + 6 * 30) / (8 + 6) = 33 :=
by
  sorry

end combined_average_age_l145_145645


namespace angle_DAH_eq_angle_MAO_l145_145138

variables {A B C D M O H X : Type*}
variables [geometry ℝ ℝ²]

-- Defining the conditions
def is_triangle (ABC : Set ℝ²) := ∃ A B C, ∆ABC = {A, B, C}

def is_orthocenter (H : ℝ²) (A B C : ℝ²) : Prop :=
  is_altitude H A B C -- H is orthocenter means it is where altitudes meet

def is_circumcenter (O : ℝ²) (A B C : ℝ²) : Prop :=
  is_center_circumscribed_circle O A B C -- O is center of circumcircle

def is_altitude (AH : ℝ²) (A B C : ℝ²) : Prop :=
  line_through_perp AH BC

def perpendicular (P Q : ℝ²) (L : ℝ²) : Prop :=
  right_angle (line_through P Q) L

def parallel (L1 L2 : ℝ²) : Prop :=
  parallel_lines L1 L2

-- Now, let's state the theorem
theorem angle_DAH_eq_angle_MAO 
  (H : ℝ²)
  (A B C : ℝ²)
  (O : ℝ²)
  (AH BC OX DAH MAO : ℝ²) :
  is_orthocenter H A B C →
  is_circumcenter O A B C →
  is_altitude AH A B C →
  perpendicular AH BC →
  perpendicular OX BC →
  parallel AH OX →
  ∠DAH = ∠MAO :=
sorry -- proof omitted

end angle_DAH_eq_angle_MAO_l145_145138


namespace find_x_values_l145_145017

theorem find_x_values (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 ∧ x2 = 3 / 5 ∧ x3 = 2 / 5 ∧ x4 = 1 / 5 :=
by
  sorry

end find_x_values_l145_145017


namespace range_of_k_l145_145065

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ 1 ≤ k ∧ k < 9 :=
by
  sorry

end range_of_k_l145_145065


namespace relationship_among_a_b_c_l145_145833

noncomputable def a : ℕ := 33
noncomputable def b : ℕ := 5 * 6^1 + 2
noncomputable def c : ℕ := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  have ha : a = 33 := rfl
  have hb : b = 5 * 6^1 + 2 := rfl
  have hc : c = 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 := rfl
  rw [ha, hb, hc]
  split
  · exact Nat.lt_of_add_le_left (by rfl)
  · exact Nat.lt_of_add_le_left (by rfl)

end relationship_among_a_b_c_l145_145833


namespace initial_boys_correct_l145_145929

variable (p : ℕ) -- the initial total number of people in the group
variable (boys_initial : ℕ) -- the initial number of boys
variable (boys_leave : ℕ := 3) -- the number of boys who leave
variable (girls_join : ℕ := 5) -- the number of girls who join

-- Conditions
def initial_boys : Prop := boys_initial = 6 * p / 10
def total_after_changes : Prop := p + 2
def boys_after_changes : Prop := boys_initial - 3

-- Main Statement
theorem initial_boys_correct : 
  initial_boys p boys_initial → 
  p + 2 = total_after_changes p boys_initial → 
  boys_initial - 3 = boys_after_changes p boys_initial → 
  (boys_initial = 6 * p / 10) / (p + 2) = 1 / 2 → 
  boys_initial = 24 :=
sorry

end initial_boys_correct_l145_145929


namespace cotangent_sum_l145_145005

def a_n (n : ℕ) : ℝ := 1 + n + n^2

theorem cotangent_sum :
  10 * Real.cot (Real.arccot 3 + Real.arccot 7 + Real.arccot 13 + Real.arccot 21) = 15 :=
by
  -- Proof goes here
  sorry

end cotangent_sum_l145_145005


namespace maximize_expr_approx_l145_145425

noncomputable def expr (x y : ℝ) : ℝ := 
  (sqrt (9 - sqrt 7) * Real.sin x - sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
  (3 + 2 * sqrt (13 - sqrt 7) * Real.cos y - Real.cos (2 * y))

noncomputable def maximum_value : ℝ := 17

theorem maximize_expr_approx :
  ∃ x y : ℝ, (-1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ∧ 
             (-1 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧ 
             (-1 ≤ Real.sin y ∧ Real.sin y ≤ 1) ∧ 
             (-1 ≤ Real.cos y ∧ Real.cos y ≤ 1) ∧ 
             abs ((expr x y) - maximum_value) < 1 :=
by
  sorry

end maximize_expr_approx_l145_145425


namespace rational_numbers_inequality_l145_145588

theorem rational_numbers_inequality (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 :=
sorry

end rational_numbers_inequality_l145_145588


namespace compound_interest_principal_amount_l145_145000

theorem compound_interest_principal_amount :
  ∀ (r : ℝ) (n : ℕ) (t : ℕ) (CI : ℝ) (P : ℝ),
    r = 0.04 ∧ n = 1 ∧ t = 2 ∧ CI = 612 →
    (CI = P * (1 + r / n) ^ (n * t) - P) →
    P = 7500 :=
by
  intros r n t CI P h_conditions h_CI
  -- Proof not needed
  sorry

end compound_interest_principal_amount_l145_145000


namespace reciprocal_G_is_R_reciprocal_G_on_unit_circle_l145_145551

def G : ℂ := (3 / 5) + (4 / 5) * complex.I
def reciprocal_G : ℂ := complex.conj G / (complex.abs G * complex.abs G) -- by definition of the reciprocal in complex numbers
def R : ℂ := (3 / 5) - (4 / 5) * complex.I

theorem reciprocal_G_is_R : reciprocal_G = R :=
sorry

theorem reciprocal_G_on_unit_circle : complex.abs reciprocal_G = 1 :=
sorry

end reciprocal_G_is_R_reciprocal_G_on_unit_circle_l145_145551


namespace Petya_higher_chance_l145_145198

theorem Petya_higher_chance :
  ∀ (P_vasya P_petya : ℝ), P_vasya = 0.46 → P_petya = 0.54 → P_vasya < P_petya :=
by
  intros P_vasya P_petya h1 h2
  rw [h1, h2]
  linarith

end Petya_higher_chance_l145_145198


namespace reflection_proof_l145_145692

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

noncomputable def initial_point : ℝ × ℝ := (3, -3)
noncomputable def reflected_over_y_axis := reflect_y initial_point
noncomputable def reflected_over_x_axis := reflect_x reflected_over_y_axis

theorem reflection_proof : reflected_over_x_axis = (-3, 3) :=
  by
    -- proof goes here
    sorry

end reflection_proof_l145_145692


namespace trig_identity_example_l145_145791

theorem trig_identity_example :
  sin (43 * (Real.pi / 180)) * cos (17 * (Real.pi / 180)) + 
  cos (43 * (Real.pi / 180)) * sin (17 * (Real.pi / 180)) = 
  (Real.sqrt 3) / 2 :=
by
  sorry

end trig_identity_example_l145_145791


namespace find_num_valid_n_l145_145810

noncomputable def num_solutions : Nat :=
  {n : ℕ // n ≤ 1200 ∧ ∃ k : ℕ, 18 * n = k ^ 2}.toFinset.card

theorem find_num_valid_n : num_solutions = 5 := by
  sorry

end find_num_valid_n_l145_145810


namespace initial_percentage_of_milk_l145_145271

theorem initial_percentage_of_milk 
  (initial_solution_volume : ℝ)
  (extra_water_volume : ℝ)
  (desired_percentage : ℝ)
  (new_total_volume : ℝ)
  (initial_percentage : ℝ) :
  initial_solution_volume = 60 →
  extra_water_volume = 33.33333333333333 →
  desired_percentage = 54 →
  new_total_volume = initial_solution_volume + extra_water_volume →
  (initial_percentage / 100 * initial_solution_volume = desired_percentage / 100 * new_total_volume) →
  initial_percentage = 84 := 
by 
  intros initial_volume_eq extra_water_eq desired_perc_eq new_volume_eq equation
  -- proof steps here
  sorry

end initial_percentage_of_milk_l145_145271


namespace no_more_than_n_diagonals_l145_145620

theorem no_more_than_n_diagonals {n : ℕ} (n_gt_2 : n > 2) :
  ∀ (P : Polygon) (convex : P.isConvex) (hn : P.numVertices = n),
  ¬ ∃ (diagonals : Finset (Diagonal P)), (diagonals.card > n) ∧
    ∀ d1 d2 ∈ diagonals, d1 ≠ d2 → (Diagonal.commonVertex d1 d2) :=
by
  sorry

end no_more_than_n_diagonals_l145_145620


namespace petya_has_higher_chance_l145_145190

theorem petya_has_higher_chance : 
  let p_chocolate_chocolate : ℝ := 1 - 0.54 in
  p_chocolate_chocolate < 0.5 :=
by
  let p_chocolate_chocolate := 1 - 0.54
  have h1 : p_chocolate_chocolate = 0.46 := by norm_num
  have h2 : 0.46 < 0.5 := by norm_num
  exact h2

end petya_has_higher_chance_l145_145190


namespace min_buses_needed_l145_145748

theorem min_buses_needed (n : ℕ) (h1 : 45 * n ≥ 500) (h2 : n ≥ 2) : n = 12 :=
sorry

end min_buses_needed_l145_145748


namespace standard_eq_ellipse_foci_axes_l145_145433

theorem standard_eq_ellipse_foci_axes 
  (A : ℝ) (B : ℝ) (hA : A = (sqrt 3, -2)) (hB : B = (-2 * sqrt 3, 1)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ A = (sqrt 3, -2) ∧ B = (-2 * sqrt 3, 1) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
by {
  sorry
}

end standard_eq_ellipse_foci_axes_l145_145433


namespace petya_higher_chance_of_winning_l145_145210

/-- Petya gives Vasya two boxes of candies. Each box contains both chocolate candies and caramel candies.
    In total, there are 25 candies in both boxes. Vasya takes one candy from each box.
    If both candies turn out to be chocolate, Vasya wins. Otherwise, Petya wins.
    The probability that Vasya will get two caramels is 0.54.
    Prove that Petya has a higher chance of winning. --/
theorem petya_higher_chance_of_winning
  (total_candies : ℕ)
  (total_candies = 25 : Prop) 
  (prob_two_caramels : ℝ)
  (prob_two_caramels = 0.54 : Prop) :
  0.46 < 0.54 := 
by 
  sorry

end petya_higher_chance_of_winning_l145_145210


namespace number_of_valid_subsets_l145_145787

theorem number_of_valid_subsets : 
  let S := {1, 2, 3, 4, 5, 6}
  let subset_X (X : Set ℕ) := {1, 2, 3} ⊆ X ∧ ¬ (6 ∈ X) ∧ X ⊆ S
  (Finset.filter subset_X (Finset.powerset S)).card = 4 :=
by
  sorry

end number_of_valid_subsets_l145_145787


namespace range_of_m_l145_145456

-- Given definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 2 else n * 2^n

def vec_a : ℕ × ℤ := (2, -1)

def vec_b (n : ℕ) : ℕ × ℤ := (sequence_a n + 2^n, sequence_a (n + 1))

def orthogonal (v1 v2 : ℕ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Translate the proof problem
theorem range_of_m (n : ℕ) (m : ℝ) (h1 : orthogonal vec_a (vec_b n))
  (h2 : ∀ n : ℕ, n > 0 → (sequence_a n) / (n * (n + 1)^2) > (m^2 - 3 * m) / 9) :
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l145_145456


namespace number_of_possible_values_l145_145300

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

def area_of_rhombus (s : ℝ) (θ : ℝ) : ℝ := s^2 * real.sin θ

-- Conditions from the problem
axiom h1 : area_of_rhombus x y = real.sqrt 2006
axiom h2 : ∀ (θ : ℝ), 0 ≤ θ → θ ≤ 2 * real.pi → real.sin θ = 2 * real.sin (θ / 2) * real.cos (θ / 2)

noncomputable def K : ℝ := 2 * area_of_rhombus x y * real.cos y

-- Main statement to prove
theorem number_of_possible_values (h_pos : 0 ≤ K) (h_int : K ∈ ℕ) :
  ∃ n : ℕ, n = 89 :=
sorry

end number_of_possible_values_l145_145300


namespace number_of_candies_l145_145106

theorem number_of_candies (n : ℕ) (h1 : 11 ≤ n) (h2 : n ≤ 100) (h3 : n % 18 = 0) (h4 : n % 7 = 1) : n = 36 :=
by
  sorry

end number_of_candies_l145_145106


namespace quadrilateral_sides_and_angles_rectangle_opposite_sides_square_all_sides_equal_l145_145741

-- Definitions
def is_quadrilateral (sides angles: ℕ) : Prop := (sides = 4 ∧ angles = 4)
def rectangle_opposite_sides_equal := ∀ (a b : ℕ), (a = b) -- simplified example
def square_sides_equal := ∀ (side sides: ℕ), (sides = 4 ∧ side * 4 = sides * side)

-- Statements to prove
theorem quadrilateral_sides_and_angles (sides angles: ℕ) (a b side: ℕ) :
  is_quadrilateral sides angles → sides = 4 ∧ angles = 4 :=
by
    intro h,
    exact h

theorem rectangle_opposite_sides (a b : ℕ) : rectangle_opposite_sides_equal :=
by
    intros a b,
    sorry

theorem square_all_sides_equal (side sides: ℕ) : square_sides_equal :=
by
    intros side sides,
    sorry

end quadrilateral_sides_and_angles_rectangle_opposite_sides_square_all_sides_equal_l145_145741


namespace minimum_value_expression_l145_145592

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 2) :
  (x + y) * (y + 3 * z) * (2 * x * z + 1) ≥ 16 * real.sqrt 6 :=
sorry

end minimum_value_expression_l145_145592


namespace value_of_a5_l145_145046

theorem value_of_a5 {a_1 a_3 a_5 : ℤ} (n : ℕ) (hn : n = 8) (h1 : (1 - x)^n = 1 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) (h_ratio : a_1 / a_3 = 1 / 7) :
  a_5 = -56 := 
sorry

end value_of_a5_l145_145046


namespace investment_proof_l145_145183

noncomputable def invest {α β γ : Type} (initial_A initial_B initial_C : ℝ)
  (y1_A_inc y1_B_dec y2_A_inc y2_B_inc y2_C_dec : ℝ)
  : ℝ × ℝ × ℝ :=
  let A1 := initial_A * (1 + y1_A_inc)
  let B1 := initial_B * (1 - y1_B_dec)
  let C1 := initial_C
  let A2 := A1 * (1 + y2_A_inc)
  let B2 := B1 * (1 + y2_B_inc)
  let C2 := C1 * (1 - y2_C_dec)
  (A2, B2, C2)

theorem investment_proof :
  let (A, B, C) := invest 150 150 150 0.10 0.30 0.15 0.50 0.10 in
  A = 189.75 ∧ B = 157.5 ∧ C = 135 ∧ A > B ∧ B > C :=
by
  sorry

end investment_proof_l145_145183


namespace parallel_vectors_perpendicular_vectors_l145_145951

open Real

variables (e1 e2 : ℝ × ℝ) 
variables (a b : ℝ × ℝ) (λ : ℝ)

-- Conditions: e1 and e2 are mutually perpendicular unit vectors
def perpendicular_unit_vectors (e1 e2 : ℝ × ℝ) : Prop :=
  (e1.1 ^ 2 + e1.2 ^ 2 = 1) ∧ (e2.1 ^ 2 + e2.2 ^ 2 = 1) ∧ (e1.1 * e2.1 + e1.2 * e2.2 = 0)

-- Definitions of vectors a and b
def vector_a (e1 e2 : ℝ × ℝ) : ℝ × ℝ := 
  (-2 * e1.1 - e2.1, -2 * e1.2 - e2.2)

def vector_b (e1 e2 : ℝ × ℝ) (λ : ℝ) : ℝ × ℝ := 
  (e1.1 - λ * e2.1, e1.2 - λ * e2.2)

-- Proof problem 1: If a parallel b, then λ = -1/2
theorem parallel_vectors (h : perpendicular_unit_vectors e1 e2) 
  (h_parallel : vector_a e1 e2 = -2 * vector_b e1 e2 (-1/2)) : λ = -1/2 := sorry

-- Proof problem 2: If a perpendicular b, then λ = 2
theorem perpendicular_vectors (h : perpendicular_unit_vectors e1 e2) 
  (h_perpendicular : (vector_a e1 e2).fst * (vector_b e1 e2 λ).fst + 
                    (vector_a e1 e2).snd * (vector_b e1 e2 λ).snd = 0) : λ = 2 := sorry

end parallel_vectors_perpendicular_vectors_l145_145951


namespace eq_square_sum_five_l145_145047

theorem eq_square_sum_five (a b : ℝ) (i : ℂ) (h : i * i = -1) (h_eq : (a - 2 * i) * i^2013 = b - i) : a^2 + b^2 = 5 :=
by
  -- Proof will be filled in later
  sorry

end eq_square_sum_five_l145_145047


namespace petya_higher_chance_of_winning_l145_145207

theorem petya_higher_chance_of_winning 
  {total_candies : ℕ} (h_total : total_candies = 25)
  {P_two_caramel : ℝ} (h_P_two_caramel : P_two_caramel = 0.54) :
  let P_two_chocolate := 1 - P_two_caramel in
  P_two_chocolate < 0.5 :=
by {
  sorry
}

end petya_higher_chance_of_winning_l145_145207


namespace find_n_l145_145435

noncomputable def e : ℝ := Real.exp 1

-- lean cannot compute non-trivial transcendental solutions, this would need numerical methods
theorem find_n (n : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3) (h2 : y = 27) :
  Real.log n ^ (n / (2 * Real.sqrt (Real.pi + x))) = y :=
by
  rw [h1, h2]
  sorry

end find_n_l145_145435


namespace clock_angle_3_05_l145_145701

/-- The clock is divided into 12 hours and 60 minutes. Each hour marks 30 degrees and 
each minute marks 6 degrees. We need to prove the angle between hour and minute hands is 62.5 degrees at 3:05. -/
theorem clock_angle_3_05 : 
  let degrees_per_hour := 30
  let degrees_per_minute := 6
  let hour_hand_at_3 := 3 * degrees_per_hour
  let additional_hour_hand_movement := (5 * 0.5)
  let hour_hand_movement := hour_hand_at_3 + additional_hour_hand_movement
  let minute_hand_movement := 5 * degrees_per_minute
  abs (hour_hand_movement - minute_hand_movement) = 62.5 := 
by 
  sorry

end clock_angle_3_05_l145_145701


namespace hall_width_length_ratio_l145_145264

theorem hall_width_length_ratio 
  (w l : ℝ) 
  (h1 : w * l = 128) 
  (h2 : l - w = 8) : 
  w / l = 1 / 2 := 
by sorry

end hall_width_length_ratio_l145_145264


namespace vector_difference_squared_l145_145872

-- Defining the vectors a and b.
def a (α : ℝ) : ℝ × ℝ := (2 * Real.sin α, Real.cos α)
def b : ℝ × ℝ := (1, -1)

-- Condition: a is perpendicular to b.
def a_perp_b (α : ℝ) : Prop := (2 * Real.sin α - Real.cos α = 0)

-- Proving the squared magnitude of the vector difference.
theorem vector_difference_squared (α : ℝ) (h : a_perp_b α) :
  (a α).fst - b.fst) ^ 2 + ((a α).snd - b.snd) ^ 2 = 18 / 5 := 
sorry

end vector_difference_squared_l145_145872


namespace parallel_lines_d_l145_145295

theorem parallel_lines_d (d : ℝ) : (∀ x : ℝ, -3 * x + 5 = (-6 * d) * x + 10) → d = 1 / 2 :=
by sorry

end parallel_lines_d_l145_145295


namespace expand_capital_options_l145_145908

-- Defining the conditions in Lean
variables {City : Type} (is_connected : City → City → Prop)
variable (Ben_Song : City)
variables (n k : ℕ)

-- The main theorem: proving the maximal number of ways to expand the capital.
theorem expand_capital_options (expanded_capital : set City) 
  (h1 : Ben_Song ∈ expanded_capital)
  (h2 : ∀ (x y : City), x ∈ expanded_capital → y ∈ expanded_capital → is_connected x y)
  (h3 : ∃ (k_out : set City), k_out.card = k ∧ ∀ x ∈ k_out, ∃ y ∈ expanded_capital, is_connected x y)
  (hn : expanded_capital.card = n + 1) :
  card (set_to_subsets expanded_capital) ≤ nat.choose (n + k) k :=
sorry

end expand_capital_options_l145_145908


namespace solution_set_of_inequality_l145_145028

variable (f : ℝ → ℝ)

def g (x : ℝ) : ℝ := f x - x - 1

theorem solution_set_of_inequality (h₁ : f 1 = 2) (h₂ : ∀ x, (deriv f x) < 1) :
  { x : ℝ | f x < x + 1 } = { x | 1 < x } :=
by
  sorry

end solution_set_of_inequality_l145_145028


namespace point_divide_ratio_trapezoid_l145_145555

theorem point_divide_ratio_trapezoid
  (A B C D M : Type*)
  (angle : Type*)
  [is_right_angle : Π (X Y Z : Type*), Prop]
  (len_AB len_AD len_CD : ℝ)
  (M_on_AD : A → D → Prop)
  (α : angle)
  (∡AM_A : angle)
  (∡CD_M : angle)
  (h1 : is_right_angle A D B)
  (h2 : is_right_angle D A M)
  (h3 : len_AB = 1)
  (h4 : len_CD = 4)
  (h5 : len_AD = 5)
  (h6 : M_on_AD A D)
  (h7 : ∡CD_M = 2 * ∡AM_A)
  : ∀ (AM MD : ℝ), AM / MD = 2 / 3 :=
sorry

end point_divide_ratio_trapezoid_l145_145555


namespace find_complex_number_l145_145807

-- Define what it means for a complex number z to satisfy the given conditions
def satisfies_conditions (z : ℂ) : Prop :=
  complex.abs (z - 2) = complex.abs (z + 4) ∧ complex.abs (z + 4) = complex.abs (z + 2 * complex.I)

-- Define the complex number -1 + i
def z_answer : ℂ := -1 + complex.I

-- State the main theorem that we need to prove
theorem find_complex_number : ∃ z : ℂ, satisfies_conditions z ∧ z = z_answer :=
  sorry

end find_complex_number_l145_145807


namespace exists_consecutive_primes_l145_145792

-- Define the function S(n) which counts the primes in the interval [n, n+2015]
def S (n : ℕ) : ℕ := (n..(n + 2015)).count is_prime

-- Condition: The difference between S(n) and S(n + 1) is at most 1
axiom S_diff_le_one (n : ℕ) : abs ((S n : ℤ) - S (n + 1) : ℤ) ≤ 1

-- Condition: S(2017! + 2) = 0
axiom S_factorial_plus_two : S (2017.factorial + 2) = 0

-- Condition: S(1) > 16
axiom S_one_gt_sixteen : S 1 > 16

-- Theorem stating there exists m such that S m = 16
theorem exists_consecutive_primes : ∃ m : ℕ, S m = 16 := by
  sorry

end exists_consecutive_primes_l145_145792


namespace john_eggs_per_week_l145_145573

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l145_145573


namespace art_museum_visitors_l145_145710

theorem art_museum_visitors 
  (V : ℕ)
  (H1 : ∃ (d : ℕ), d = 130)
  (H2 : ∃ (e u : ℕ), e = u)
  (H3 : ∃ (x : ℕ), x = (3 * V) / 4)
  (H4 : V = (3 * V) / 4 + 130) :
  V = 520 :=
sorry

end art_museum_visitors_l145_145710


namespace sum_of_arithmetic_series_is_correct_l145_145653

variable (t : ℝ)

-- Defining the arithmetic series
def first_term : ℝ := t^2 + 3
def common_difference : ℝ := 1
def number_of_terms : ℝ := 3t + 2

-- Sum formula for the arithmetic series
def sum_arithmetic_series (n : ℝ) (a₁ : ℝ) (d : ℝ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

-- The sum of the series with the given first term and common difference
theorem sum_of_arithmetic_series_is_correct :
  sum_arithmetic_series (3t + 2) (t^2 + 3) 1 = (3t + 2) * (t^2 + 1.5 * t + 3.5) :=
sorry

end sum_of_arithmetic_series_is_correct_l145_145653


namespace nile_family_p1_eq_p2_nile_family_p2_gt_p3_gt_p10_l145_145726

-- Part a: Prove that p1 = p2
theorem nile_family_p1_eq_p2 :
  (each_crocodile_probability_eq_0.1 : ∀ (i : Nat), i ∈ [1..10] → i.probability = 0.1) →
  ∀ (k : Nat), k = 1 →
  ∀ (p1 p2: Probability), p1 = (P(B_1)) ∧ p2 = (P(B_2)) →
  p1 = p2 :=
sorry

-- Part b: Prove that p2 > p3 > p4 > ... > p10
theorem nile_family_p2_gt_p3_gt_p10 :
  (each_crocodile_probability_eq_0.1 : ∀ (i : Nat), i ∈ [1..10] → i.probability = 0.1) →
  ∀ (k : Nat), 2 ≤ k ∧ k ≤ 10 →
  ∀ (p : Probability), (p2 k) >= (p2 k+1) :=
sorry

end nile_family_p1_eq_p2_nile_family_p2_gt_p3_gt_p10_l145_145726


namespace area_of_triangle_l145_145107

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_triangle_l145_145107


namespace inverse_relationship_l145_145678

variables {x y : ℝ} 

theorem inverse_relationship (h1 : y * real.sqrt x = 4 * real.sqrt 2) (h2 : x = 2) (hy : y = 1) :
  x = 32 :=
by sorry

end inverse_relationship_l145_145678


namespace count_triangles_in_extended_figure_l145_145380

theorem count_triangles_in_extended_figure: 
  let base_row := 4
  let num_rows := 4
  let num_small_triangles := base_row + (base_row - 1) + (base_row - 2) + (base_row - 3)
  let num_composite_triangles := 2 + 1  -- Composite triangles made of 4 small triangles
  let num_largest_triangle := 1  -- The entire figure as one large triangle
  num_small_triangles + num_composite_triangles + num_largest_triangle = 14 := 
by
  let base_row := 4
  let num_rows := 4
  let num_small_triangles := base_row + (base_row - 1) + (base_row - 2) + (base_row - 3)
  let num_composite_triangles := 2 + 1
  let num_largest_triangle := 1
  have h1 : num_small_triangles = 10 := rfl
  have h2 : num_composite_triangles = 3 := rfl
  have h3 : num_largest_triangle = 1 := rfl
  show num_small_triangles + num_composite_triangles + num_largest_triangle = 14,
    calc
      10 + 3 + 1 = 14 : rfl

end count_triangles_in_extended_figure_l145_145380


namespace reflect_over_y_axis_matrix_l145_145412

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l145_145412


namespace smallest_positive_period_cos_squared_l145_145258

theorem smallest_positive_period_cos_squared :
  ∃ p > 0, (∀ x, f(x + p) = f(x)) ∧ 
        (∀ q > 0, (∀ x, f(x + q) = f(x)) → q ≥ p) :=
by
  let f := λ x : ℝ, (Real.cos x) ^ 2
  have h1 : f = λ x, (1 + Real.cos (2 * x)) / 2 := by
    -- This is using the double-angle formula for cosine
    sorry 
  have period_cos2x : ∃ p > 0, (∀ x, Real.cos(2 * (x + p)) = Real.cos(2 * x)) := 
    by sorry
  have period_cos_squared : ∀ x, f(x + (π)) = f(x) := by
    -- Use the period of cos(2x) to show this
    sorry
  have minimality : ∀ q > 0, (∀ x, f(x + q) = f(x)) → q ≥ π := by
    sorry
  exact ⟨π, Real.pi_pos, period_cos_squared, minimality⟩

end smallest_positive_period_cos_squared_l145_145258


namespace petya_has_higher_chance_l145_145193

theorem petya_has_higher_chance : 
  let p_chocolate_chocolate : ℝ := 1 - 0.54 in
  p_chocolate_chocolate < 0.5 :=
by
  let p_chocolate_chocolate := 1 - 0.54
  have h1 : p_chocolate_chocolate = 0.46 := by norm_num
  have h2 : 0.46 < 0.5 := by norm_num
  exact h2

end petya_has_higher_chance_l145_145193


namespace petya_has_higher_chance_l145_145192

theorem petya_has_higher_chance : 
  let p_chocolate_chocolate : ℝ := 1 - 0.54 in
  p_chocolate_chocolate < 0.5 :=
by
  let p_chocolate_chocolate := 1 - 0.54
  have h1 : p_chocolate_chocolate = 0.46 := by norm_num
  have h2 : 0.46 < 0.5 := by norm_num
  exact h2

end petya_has_higher_chance_l145_145192


namespace inequality_sum_sqrt_ge_sum_l145_145817

variable {n : ℕ}
variables {a : ℕ → ℝ}

theorem inequality_sum_sqrt_ge_sum 
  (h1 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a k ≥ 0) 
  (h2 : a (n + 1) = a 1) :
  ∑ k in finset.range n, 
      real.sqrt (a k ^ 2 - a k * a (k + 1) + a (k + 1) ^ 2) 
    ≥ ∑ k in finset.range n, a k :=
sorry

end inequality_sum_sqrt_ge_sum_l145_145817


namespace find_x_l145_145476

noncomputable def z1 : ℂ := 3 + 2 * I
noncomputable def z2 (x : ℝ) : ℂ := 4 - x * I

def purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_x (x : ℝ) (hx : purely_imaginary (z1 / z2 x)) : x = 6 := 
  sorry

end find_x_l145_145476


namespace max_k_l145_145031

noncomputable def sum_geometric (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  a₁ * (1 - q^n) / (1 - q)

open Real

theorem max_k (a₁ : ℝ) (h₁ : a₁ < 0) (q : ℝ) (h₂ : q = 1 / 2)
  (h_arith_seq : 2 * (a₁ * q^3) = (a₁ * q^2)) :
  ∃ k : ℕ, S_k < 5 * S_{k-4} → k ≤ 4 := 
by
  let S := sum_geometric a₁ q
  have S_k := S k
  have S_k_minus_4 := S (k - 4)
  sorry

end max_k_l145_145031


namespace mn_difference_l145_145462

theorem mn_difference (m n : ℕ) 
  (M : set ℕ) (hM : M = {1, 2, 3, m}) 
  (N : set ℕ) (hN : N = {4, 7, n^4, n^2 + 3n}) 
  (f : ℕ → ℕ) (hf : f = λ x, 3 * x + 1) 
  (hmn : ∃ x ∈ M, f x ∈ N) : m - n = 3 :=
begin
  sorry
end

end mn_difference_l145_145462


namespace cheburashka_time_l145_145377
-- Import the necessary library

-- Define the given conditions
variables (g_rate c_rate : ℚ) (difference : ℚ) (amount_cake : ℚ)

-- Define the conditions using the variables
def conditions :=
  (g_rate > 0) ∧
  (c_rate = g_rate / 2) ∧
  (difference = 1) ∧
  (amount_cake = g_rate * 2)

-- Define the expected result or proof goal
theorem cheburashka_time :
  conditions g_rate c_rate difference amount_cake →
  (amount_cake / c_rate) = 4 :=
begin
  intro h,
  sorry
end

end cheburashka_time_l145_145377


namespace teal_sold_pumpkin_pies_l145_145913

def pies_sold 
  (pumpkin_pie_slices : ℕ) (pumpkin_pie_price : ℕ) 
  (custard_pie_slices : ℕ) (custard_pie_price : ℕ) 
  (custard_pies_sold : ℕ) (total_revenue : ℕ) : ℕ :=
  total_revenue / (pumpkin_pie_slices * pumpkin_pie_price)

theorem teal_sold_pumpkin_pies : 
  pies_sold 8 5 6 6 5 340 = 4 := 
by 
  sorry

end teal_sold_pumpkin_pies_l145_145913


namespace compute_expression_l145_145378

theorem compute_expression :
  18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 :=
by
  sorry

end compute_expression_l145_145378


namespace total_dots_not_visible_l145_145273

noncomputable def total_dots_on_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
noncomputable def total_dice : ℕ := 3
noncomputable def total_visible_faces : ℕ := 5

def visible_faces : List ℕ := [1, 2, 3, 3, 4]

theorem total_dots_not_visible :
  (total_dots_on_die * total_dice) - (visible_faces.sum) = 50 := by
  sorry

end total_dots_not_visible_l145_145273


namespace system_solution_exists_l145_145310

theorem system_solution_exists :
  ∃ (x y z : ℝ), 
  x^2 - y * z = 1 ∧
  y^2 - x * z = 2 ∧
  z^2 - x * y = 3 ∧
  x = 5 * Real.sqrt 2 / 6 ∧
  y = -Real.sqrt 2 / 6 ∧
  z = -7 * Real.sqrt 2 / 6 :=
by
  use 5 * Real.sqrt 2 / 6, -Real.sqrt 2 / 6, -7 * Real.sqrt 2 / 6
  split
  { sorry }
  split
  { sorry }
  ratio
  { sorry }
  split
  { refl }
  split
  { refl }
  { refl }

end system_solution_exists_l145_145310


namespace digit_sum_of_4_digit_number_l145_145308

theorem digit_sum_of_4_digit_number (abcd : ℕ) (H1 : 1000 ≤ abcd ∧ abcd < 10000) (erased_digit: ℕ) (H2: erased_digit < 10) (H3 : 100*(abcd / 1000) + 10*(abcd % 1000 / 100) + (abcd % 100 / 10) + erased_digit = 6031): 
    (abcd / 1000 + abcd % 1000 / 100 + abcd % 100 / 10 + abcd % 10 = 20) :=
sorry

end digit_sum_of_4_digit_number_l145_145308


namespace monotonicity_of_f_inequality_of_f_l145_145071

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

theorem monotonicity_of_f {a : ℝ}:
(a ≥ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f x a ≤ f y a) ∧
(a < 0 → ∀ x y : ℝ, 0 < x ∧ x < y ∧ x ≥ -1 + Real.sqrt (1 - 2 * a) → f x a ≤ f y a 
∨ 0 < x ∧ x < -1 + Real.sqrt (1 - 2 * a) → f x a ≥ f y a) := sorry

theorem inequality_of_f {a : ℝ} (h : t ≥ 1) :
(f (2*t-1) a ≥ 2 * f t a - 3) ↔ (a ≤ 2) := sorry

end monotonicity_of_f_inequality_of_f_l145_145071


namespace number_of_ordered_pairs_l145_145135

noncomputable def numberOfPairs (p : ℕ) (q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (cond : p = 2 * q + 1) (hcond : p > 13) : ℕ :=
  q - 1

theorem number_of_ordered_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hcond1 : p = 2 * q + 1) (hcond2 : p > 13) : 
  (∃ m n : ℕ, 0 ≤ m ∧ m < n ∧ n < p - 1 ∧ (3^m + (-12)^m) % p = (3^n + (-12)^n) % p) →
  numberOfPairs p q hp hq hcond1 hcond2 = q - 1 :=
by
  sorry

end number_of_ordered_pairs_l145_145135


namespace factorial_eq_l145_145397

theorem factorial_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (Nat.factorial a) * (Nat.factorial b) = (Nat.factorial a) + (Nat.factorial b) + (Nat.factorial c) → 
  (a = 3 ∧ b = 3 ∧ c = 4) := by
  sorry

end factorial_eq_l145_145397


namespace smallest_M_exists_l145_145432

theorem smallest_M_exists :
  ∃ M : ℕ, M = 249 ∧
  (∃ k1 : ℕ, (M + k1 = 8 * k1 ∨ M + k1 + 1 = 8 * k1 ∨ M + k1 + 2 = 8 * k1)) ∧
  (∃ k2 : ℕ, (M + k2 = 27 * k2 ∨ M + k2 + 1 = 27 * k2 ∨ M + k2 + 2 = 27 * k2)) ∧
  (∃ k3 : ℕ, (M + k3 = 125 * k3 ∨ M + k3 + 1 = 125 * k3 ∨ M + k3 + 2 = 125 * k3)) :=
by
  sorry

end smallest_M_exists_l145_145432


namespace colten_chickens_l145_145999

variable (C Q S : ℕ)

-- Conditions
def condition1 : Prop := Q + S + C = 383
def condition2 : Prop := Q = 2 * S + 25
def condition3 : Prop := S = 3 * C - 4

-- Theorem to prove
theorem colten_chickens : condition1 C Q S ∧ condition2 C Q S ∧ condition3 C Q S → C = 37 := by
  sorry

end colten_chickens_l145_145999


namespace maximum_area_triangle_l145_145827

noncomputable def maximum_area_of_triangle_ABC 
(O A B C : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C] 
(distance_OA : dist O A = 4) 
(distance_OB : dist O B = 3) 
(distance_OC : dist O C = 2) 
(dot_OB_OC : inner_product_space O B * inner_product_space O C = 3) : ℝ :=
2 * (real.sqrt 7) + 3 * (real.sqrt 3) / 2

theorem maximum_area_triangle 
(O A B C : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C] 
(distance_OA : dist O A = 4) 
(distance_OB : dist O B = 3) 
(distance_OC : dist O C = 2) 
(dot_OB_OC : inner_product_space O B * inner_product_space O C = 3) :
maximum_area_of_triangle_ABC O A B C = 2 * (real.sqrt 7) + 3 * (real.sqrt 3) / 2 := 
sorry

end maximum_area_triangle_l145_145827


namespace Lewis_earnings_without_overtime_l145_145597

theorem Lewis_earnings_without_overtime
  (weeks : ℕ) (overtime_per_week : ℕ) (total_earnings : ℕ) (work_overtime_every_week : Prop)
  (h1 : weeks = 1091) (h2 : overtime_per_week = 939) (h3 : total_earnings = 1_054_997) 
  (h4 : work_overtime_every_week) :
  let total_overtime_earnings := overtime_per_week * weeks in
  let total_without_overtime := total_earnings - total_overtime_earnings in
  let weekly_without_overtime := total_without_overtime / weeks in
  weekly_without_overtime = 2761 / 100 := -- since 27.61 is represented as 2761 / 100 in integer arithmetic
by
  sorry

end Lewis_earnings_without_overtime_l145_145597


namespace exists_k_lt_n_tk_tkp1_ge_two_l145_145664

theorem exists_k_lt_n_tk_tkp1_ge_two
  (n : ℕ) (t : Fin n → ℝ) (hpos : ∀ i, 0 < t i) (hprod : (∏ i, t i) = 1) :
  ∃ k : Fin n, t k * (t k.succ % n + 1) ≥ 2 := by
  sorry

end exists_k_lt_n_tk_tkp1_ge_two_l145_145664


namespace max_n_value_l145_145041

variable {a : ℤ} (a_1 a_8 a_9 : ℤ)
def arithmetic_seq (n : ℕ) : ℤ := a_1 + n * (a_8 - a_1)

def b_seq (n : ℕ) : ℤ := n * a + n^2

theorem max_n_value (h1 : a_9 < 0) (h2 : a_8 > |a_9|)
  (h_seq : ∀ n : ℕ, arithmetic_seq a_1 a_8 n = a_1 + n * (a_8 - a_1))
  : ∃ n : ℕ, n ≤ 8 ∧ (∑ i in range n, b_seq a i) = 1 :=
sorry

end max_n_value_l145_145041


namespace neither_proportional_A_D_neither_proportional_l145_145522

theorem neither_proportional_A_D (x y : ℝ) : 
  ¬ ((∃ k : ℝ, y = k * x) ∨ (∃ k : ℝ, x * y = k)) :=
by
  sorry

theorem neither_proportional (x y : ℝ) :
  (2 * x + 3 * y = 6 ∧ 5 * x + 2 * y = 3) →
  (¬ (∃ k : ℝ, x / y = k) ∧ ¬ (∃ k : ℝ, x * y = k)) :=
by
  intros h
  cases h with hA hD
  split
  · intro h1
    rcases h1 with ⟨k, hk⟩
    revert hk -- contradicts
    exact neither_proportional_A_D x y
  · intro h2
    rcases h2 with ⟨k, hk⟩
    revert hk -- contradicts
    exact neither_proportional_A_D x y

end neither_proportional_A_D_neither_proportional_l145_145522


namespace minor_premise_l145_145783

variables (A B C : Prop)

theorem minor_premise (hA : A) (hB : B) (hC : C) : B := 
by
  exact hB

end minor_premise_l145_145783


namespace product_of_x_coords_l145_145608

theorem product_of_x_coords : 
  ∃ (x1 x2 : ℝ), (∀ x, (x = x1 ∨ x = x2) ↔ 
    ((x, 4) = (x1, 4) ∨ (x, 4) = (x2, 4)) ∧ 
    (real.sqrt ((-2 - x) ^ 2 + 49) = 15)) ∧ 
    (x1 * x2 = -172) :=
sorry

end product_of_x_coords_l145_145608


namespace petya_has_higher_chance_of_winning_l145_145204

noncomputable def vasya_win_probability : ℝ := 0.46
noncomputable def petya_win_probability : ℝ := 1 - vasya_win_probability

theorem petya_has_higher_chance_of_winning 
  (h: vasya_win_probability = 1 - 0.54) : petya_win_probability > 0.5 :=
by
  have hp : petya_win_probability = 1 - vasya_win_probability := rfl
  have h_vasya : vasya_win_probability = 0.46 := h
  rw [h_vasya] at hp
  have h_petya : petya_win_probability = 1 - 0.46 := hp
  simp [h_petya]
  exact by norm_num

end petya_has_higher_chance_of_winning_l145_145204


namespace min_sum_abc_l145_145667

theorem min_sum_abc (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1020) : a + b + c = 33 :=
sorry

end min_sum_abc_l145_145667


namespace hexagon_chord_length_sum_l145_145732

theorem hexagon_chord_length_sum :
  ∀ {m' n' : ℕ}, 
    ∃ (hexagon : Type) 
        (inscribed_in_circle : Prop) 
        (sides_4 : Prop)
        (sides_6 : Prop)
        (chord_divides_hexagon : Prop),
      rel_prime m' n' → 
      inscribed_in_circle → 
      sides_4 → 
      sides_6 → 
      chord_divides_hexagon → 
      (m' + n' = 491) :=
by
  intros
  sorry

end hexagon_chord_length_sum_l145_145732


namespace reflect_over_y_axis_matrix_l145_145415

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l145_145415


namespace candidate_net_gain_difference_l145_145334

theorem candidate_net_gain_difference :
  let salary1 := 42000
      revenue1 := 93000
      training_cost_per_month := 1200
      training_months := 3
      salary2 := 45000
      revenue2 := 92000
      hiring_bonus_percent := 1 / 100 in
  let total_training_cost1 := training_cost_per_month * training_months in
  let hiring_bonus2 := salary2 * hiring_bonus_percent in
  let net_gain1 := revenue1 - salary1 - total_training_cost1 in
  let net_gain2 := revenue2 - salary2 - hiring_bonus2 in
  net_gain1 - net_gain2 = 850 :=
by
  sorry

end candidate_net_gain_difference_l145_145334


namespace area_S_le_pi_center_S_on_y_equals_x_l145_145584

noncomputable def fractional_part (t : ℝ) : ℝ := t - Real.floor t

noncomputable def S (t : ℝ) : Set (ℝ × ℝ) :=
  let T := fractional_part t
  {p : ℝ × ℝ | (p.1 - T) ^ 2 + (p.2 - T) ^ 2 ≤ T ^ 2}

theorem area_S_le_pi (t : ℝ) (ht : t ≥ 0) : 0 ≤ (real.pi * (fractional_part t) ^ 2) ∧ (real.pi * (fractional_part t) ^ 2) ≤ real.pi :=
  sorry

theorem center_S_on_y_equals_x (t : ℝ) (ht : t ≥ 0) : ∃ (T : ℝ), T = fractional_part t ∧ ∀ (x y : ℝ), ((x, y) ∈ S t) → x = y :=
  sorry

end area_S_le_pi_center_S_on_y_equals_x_l145_145584


namespace female_officers_on_duty_percentage_l145_145602

theorem female_officers_on_duty_percentage:
  ∀ (total_on_duty : ℕ) (female_on_duty : ℕ) (total_female : ℕ)
  (h1 : total_on_duty = 500)
  (h2 : female_on_duty = total_on_duty / 2)
  (h3 : total_female = 1000),
  (female_on_duty : ℕ) = 25% :=
by
  sorry

end female_officers_on_duty_percentage_l145_145602


namespace sum_of_squares_of_roots_l145_145382

theorem sum_of_squares_of_roots :
  let a := 2
  let b := 5
  let c := -12
  let sum_of_roots := -(b / a)
  let product_of_roots := (c / a)
  (sum_of_roots ^ 2 - 2 * product_of_roots) = (73/4) := by
  let x₁ := -(b / a)
  let x₂ := (c / a)
  have h₁: (x₁ ^ 2 - 2 * x₂) = (73/4) := sorry
  exact h₁

end sum_of_squares_of_roots_l145_145382


namespace Petya_higher_chance_l145_145195

theorem Petya_higher_chance :
  ∀ (P_vasya P_petya : ℝ), P_vasya = 0.46 → P_petya = 0.54 → P_vasya < P_petya :=
by
  intros P_vasya P_petya h1 h2
  rw [h1, h2]
  linarith

end Petya_higher_chance_l145_145195


namespace reflectionY_matrix_correct_l145_145417

-- Define the basis vectors e₁ and e₂
def e1 : Vector := ⟨1, 0⟩
def e2 : Vector := ⟨0, 1⟩

-- Define the transformation that reflects over the y-axis
def reflectY : Vector → Vector 
| ⟨x, y⟩ => ⟨-x, y⟩

-- Conditions given in the problem
lemma reflectY_e1 : reflectY e1 = ⟨-1, 0⟩ := sorry
lemma reflectY_e2 : reflectY e2 = ⟨0, 1⟩ := sorry

-- The goal is to find the transformation matrix for reflection over the y-axis
def reflectionMatrixY : Matrix 2 2 ℝ :=
  Matrix.of_vec ([-1, 0, 0, 1])

theorem reflectionY_matrix_correct :
  ∀ (v : Vector), reflectY v = reflectionMatrixY.mul_vec v := sorry

end reflectionY_matrix_correct_l145_145417


namespace function_meets_conditions_l145_145030

theorem function_meets_conditions :
  (∃ f : ℝ → ℝ, (f 0 * f 2 < 0) ∧ (∀ x, f x ≠ 0) ∧ (∀ x, f x = 1 / (x - 1))) :=
by
  have exists_f : ∃ f : ℝ → ℝ, (f = λ x, 1 / (x - 1)) := ⟨λ x, 1 / (x - 1), rfl⟩
  cases exists_f with f hf
  use f
  split
  { calc f 0 * f 2 = (1 / (0 - 1)) * (1 / (2 - 1)) : by rw hf
    ... = (-1) * (1) : by norm_num
    ... < 0 : by norm_num }
  split
  { intro x,
    rw hf,
    exact one_div_ne_zero (sub_ne_zero.mpr (ne_of_gt (lt_of_le_of_ne (le_of_eq rfl) (ne_of_not (λ h, subtype.mk_inj.2 h)))) )}
  { exact hf }

-- sorry

end function_meets_conditions_l145_145030


namespace no_politics_reporters_percentage_l145_145907

def percent_covering_politics (x y both total covering_local covering_other_covering_politics : ℝ): ℝ :=
  covering_local + covering_other_covering_politics

def percent_not_covering_any_politics (total covering_politics : ℝ): ℝ :=
  total - covering_politics

theorem no_politics_reporters_percentage :  
  (total x y both covering_local covering_other_covering_politics not_covering_politics : ℝ) 
  (hx : x = 30 / 100) 
  (hy : y = 10 / 100) 
  (hxy : both = 10 / 100) 
  (hlocal : covering_local = x + y - both) 
  (hother : covering_other_covering_politics = 25 / 100) 
  (htotal : total = 1) 
  (hcovering_politics : covering_local + covering_other_covering_politics = 55 / 100)
  (hnot_covering_politics : percent_not_covering_any_politics total (covering_local + covering_other_covering_politics) = 0.45)
  : not_covering_politics = percent_not_covering_any_politics total (x + y - both + covering_other_covering_politics)  :=
  sorry

end no_politics_reporters_percentage_l145_145907


namespace sum_of_integer_ks_l145_145536

theorem sum_of_integer_ks (k : ℤ) (x : ℚ) (y : ℚ) 
    (h1 : -2 * (x - 2) - x < 2) 
    (h2 : (k - x) / 2 ≥ -1 / 2 + x) 
    (h3 : 3 * (y - 1) - 2 * (y - k) = 7)
    (h4 : y ≤ 0) : 
    ∑ i in {k | 5 ≤ k ∧ k < 8}.to_finset, i = 18 := 
by 
  sorry

end sum_of_integer_ks_l145_145536


namespace find_t_l145_145982

theorem find_t (t : ℕ) : 
  t > 3 ∧ (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) → t = 6 := 
by
  intro h
  have h1 : t > 3 := h.1
  have h2 : (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) := h.2
  sorry

end find_t_l145_145982


namespace exists_divisible_by_11_l145_145614

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem exists_divisible_by_11 (n : ℕ) :
  ∃ k ∈ (List.range' n 39), sum_of_digits k % 11 = 0 :=
by
  sorry

end exists_divisible_by_11_l145_145614


namespace quadratic_roots_eq_d_l145_145668

theorem quadratic_roots_eq_d (d : ℝ) (h : (∀ x, x^2 + 6 * x + d = 0 ↔ x = (-6 + real.sqrt d) / 2 ∨ x = (-6 - real.sqrt d) / 2)) :
  d = 36 / 5 :=
sorry

end quadratic_roots_eq_d_l145_145668


namespace minimum_area_correct_l145_145924

-- Define the geometric setup and conditions as Lean structures and assumptions
variables (a : ℝ) -- Defining the side length as a real number

noncomputable def minimum_area_of_cross_section (A B C P D : ℝ → ℝ → ℝ) : Prop :=
  let PA := 3 * a
  let AB := 2 * a
  let AC := a
  let PC := Real.sqrt (PA^2 + AC^2)
  let AD := (PA * AC) / PC
  let S_ABC := (1 / 2) * AB * AD in
  S_ABC = (3 / Real.sqrt 10) * a^2

-- Assuming the required conditions for the points
axiom (A B C P D : ℝ → ℝ → ℝ)
  (h1 : PA ⊥ plane_ABC)
  (h2 : AB ⊥ AC)
  (h3 : AB intersects PC at D)
  (h4 : A ≠ B)
  (h5 : A ≠ C)
  (h6 : B ≠ C)

theorem minimum_area_correct : minimum_area_of_cross_section A B C P D :=
sorry

end minimum_area_correct_l145_145924


namespace magnitude_angle_A_max_sin_B_plus_sin_C_l145_145108

def triangle := Σ a b c : ℝ, Prop

variables {a b c : ℝ}
variables (A B C : ℝ) (cosA : ℝ) (cosB : ℝ) (vector_m : ℝ × ℝ) (vector_n : ℝ × ℝ)

-- Conditions
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

def angle_cosine (A B : ℝ) : ℝ × ℝ := (cos A, cos B)

def side_vector (a b c : ℝ) : ℝ × ℝ := (a, 2 * c - b)

-- Questions as theorem statements
theorem magnitude_angle_A (cosA cosB : ℝ) (a b c : ℝ) (h_triangle : is_triangle a b c)
  (h_parallel : vectors_parallel (angle_cosine A B) (side_vector a b c))
  : A = π / 3 :=
sorry

theorem max_sin_B_plus_sin_C (A B C : ℝ) (a b c : ℝ)
  (h_triangle : is_triangle a b c) (h_A : A = π / 3)
  : ∃ (B C : ℝ), 
    (sin B + sin C = √3) ∧ (angle_cosine A B = angle_cosine B A) :=
sorry

end magnitude_angle_A_max_sin_B_plus_sin_C_l145_145108


namespace tax_diminished_by_32_percent_l145_145262

noncomputable def diminished_tax_percentage (T C : ℝ) : ℝ :=
  let R := T * C
  let R' := (T - (32 / 100) * T) * (C + (12 / 100) * C)
  R' = 0.7616 * R

theorem tax_diminished_by_32_percent (T C : ℝ) :
  diminished_tax_percentage T C :=
by
  let one := 1
  let X := 32 / 100
  let twelve := 12 / 100
  have h1 : (T - X * T) * (C + twelve * C) = 0.7616 * T * C := _
  sorry


end tax_diminished_by_32_percent_l145_145262


namespace gnome_problem_l145_145266

theorem gnome_problem : 
  ∀ (total_gnomes red_hats big_noses blue_big_noses red_big_noses red_small_noses : ℕ),
  total_gnomes = 28 →
  red_hats = (total_gnomes * 3) / 4 →
  big_noses = total_gnomes / 2 →
  blue_big_noses = 6 →
  red_big_noses = big_noses - blue_big_noses →
  red_small_noses = red_hats - red_big_noses →
  red_small_noses = 13 :=
by
  intros total_gnomes red_hats big_noses blue_big_noses red_big_noses red_small_noses
  assume h_total h_red_hats h_big_noses h_blue_big_noses h_red_big_noses h_red_small_noses
  sorry

end gnome_problem_l145_145266


namespace area_square_slopes_asymmetric_lines_relationship_between_a_b_l145_145043

open Real

-- (1)
theorem area_square (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  parallelogram_is_square a b →
  parallelogram_area a b = 4 * a^2 * b^2 / (a^2 + b^2) :=
sorry

-- (2)
theorem slopes_asymmetric_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  symmetric_lines_about_y_axis →
  ∀ P : ℝ × ℝ, point_on_ellipse P a b →
  let d₁ := distance_to_line P (line_slope b a)
  let d₂ := distance_to_line P (neg_line_slope b a) in
    slopes_of_lines l₁ l₂ = (b / a, -b / a) ∧ 
    fixed_value_d1_d2 a b = 2 * a^2 * b^2 / (a^2 + b^2) :=
sorry

-- (3)
theorem relationship_between_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  rhombus_contains_unit_circle a b →
  1 / a^2 + 1 / b^2 = 1 :=
sorry

end area_square_slopes_asymmetric_lines_relationship_between_a_b_l145_145043


namespace find_m_and_f_explicit_form_find_range_of_a_l145_145478

-- Definitions
def f (x : ℝ) (m : ℤ) : ℝ := x ^ (m^2 - 4 * m)

-- Problem 1: Find m such that f(x) = x^(m^2 - 4m) and conditions hold
theorem find_m_and_f_explicit_form :
  (∃ m : ℤ, 0 < m ∧ m < 4 ∧ is_even (m^2 - 4 * m) ∧ ∀ x > 0, ∀ y > x, f(x, m) > f(y, m))
  → ∃ m : ℤ, m = 2 ∧ ∀ x : ℝ, f(x, m) = x ^ (-4) :=
by
  sorry

-- Problem 2: Find the range of a such that f(a+2) < f(1-2a) for given f(x)
theorem find_range_of_a (a : ℝ) :
  (∀ a, f(a+2, 2) < f(1 - 2a, 2))
  → ∃ (s1 s2 : set ℝ), s1 = set.Ioo (-1/3) (1/2) ∧ s2 = set.Ioo (1/2) 3 
  ∧ a ∈ s1 ∪ s2 :=
by
  sorry

end find_m_and_f_explicit_form_find_range_of_a_l145_145478


namespace magnitude_of_z_l145_145453

def z : ℂ := (5 * complex.I) / (1 - 2 * complex.I)

theorem magnitude_of_z : complex.abs z = real.sqrt 5 := by
  sorry

end magnitude_of_z_l145_145453


namespace wallenstein_wins_tilly_wins_l145_145014

theorem wallenstein_wins (N M : ℕ) (n : Fin M → ℕ) (h_sum : (∑ i, n i) = N) :
  (N - M) % 2 = 0 → loses Tilly :=
sorry

theorem tilly_wins (N M : ℕ) (n : Fin M → ℕ) (h_sum : (∑ i, n i) = N) :
  (N - M) % 2 = 1 → loses Wallenstein :=
sorry

end wallenstein_wins_tilly_wins_l145_145014


namespace charge_per_meter_is_correct_l145_145391

-- Define the given conditions
def number_of_fences := 50
def length_of_each_fence := 500
def total_earnings := 5000

-- Calculate total meters painted
def total_meters_painted := number_of_fences * length_of_each_fence

-- Calculate the charge per meter
def charge_per_meter := total_earnings / total_meters_painted

-- Theorem stating that the charge per meter is $0.20
theorem charge_per_meter_is_correct : charge_per_meter = 0.20 := 
by 
  have h1: total_meters_painted = 25000, from sorry,
  have h2: charge_per_meter = (5000.0 / 25000.0), from sorry,
  exact h2

end charge_per_meter_is_correct_l145_145391


namespace determine_PS_l145_145556

noncomputable def length_PS (PQ QR PR RS : ℝ) (cos_phi : ℝ) : ℝ :=
  real.sqrt ((PR^2 + RS^2 - 2 * PR * RS * cos_phi) / 96)

theorem determine_PS :
  ∀ (PQ QR PR RS : ℝ)
    (H1 : PQ = 6)
    (H2 : QR = 8)
    (H3 : PR = 7)
    (H4 : RS = 10)
    (cos_phi : ℝ),
    cos_phi = 51 / 96 →
    length_PS PQ QR PR RS cos_phi = real.sqrt (7116 / 96) :=
by
  intros PQ QR PR RS H1 H2 H3 H4 cos_phi h_cos
  simp [length_PS, h_cos]
  sorry

end determine_PS_l145_145556


namespace triangle_ratio_l145_145171

variable (α β γ λ μ ν : ℝ) -- Define the side lengths α, β, γ, and distances λ, μ, ν

theorem triangle_ratio (h : α^2 + β^2 + γ^2 = 6 * (λ^2 + μ^2 + ν^2)) :
  (α^2 + β^2 + γ^2) / (λ^2 + μ^2 + ν^2) = 6 :=
begin
  rw h,
  field_simp,
  ring,
end

end triangle_ratio_l145_145171


namespace candy_boxes_count_l145_145600

def num_boxes : ℕ := 2020

-- The label function returns the statement on the i-th box.
def label (i : ℕ) : String :=
  if i = 1 then "All boxes are empty."
  else if i = 2 then "At least 2019 boxes are empty."
  else if i = 3 then "At least 2018 boxes are empty."
  else if i = num_boxes then "At least one box is empty."
  else "At least {num_boxes + 1 - i} boxes are empty."

-- Definition of empty and candy boxes
def is_empty (i : ℕ) : Prop := sorry -- Definition to capture that i-th box is empty
def has_candy (i : ℕ) : Prop := ¬(is_empty i) -- A box has candy if it is not empty

-- Definition of label truthfulness:
def label_is_true (i : ℕ) : Prop :=
  if has_candy i then
    match i with
    | 1 => false
    | i => ∀ j, j ≥ i → is_empty j
  else
    false

def label_is_false (i : ℕ) : Prop :=
  ¬(label_is_true i)

-- The key condition: empty boxes have false labels and candy boxes have true labels
axiom empty_boxes_false : ∀ i, is_empty i → label_is_false i
axiom candy_boxes_true : ∀ i, has_candy i → label_is_true i

-- Final statement to prove
theorem candy_boxes_count : ∃ k, (k = 1010 ∧ ∃ L, L = 2020 - k ∧ 
 (∀ i, i ≥ L → has_candy i) ∧ (∀ i, i < L → is_empty i)) :=
sorry

end candy_boxes_count_l145_145600


namespace min_value_modulus_squared_l145_145186

noncomputable def z (x y : ℝ) : ℂ := x + y * complex.I
noncomputable def zi (x y : ℝ) : ℂ := -y + x * complex.I

theorem min_value_modulus_squared (x y : ℝ) (hx : 0 < x) (area_condition : abs (x ^ 2 + y ^ 2) = 40) :
  abs ((x - y) + (x + y) * complex.I) ^ 2 = 80 :=
by
  sorry

end min_value_modulus_squared_l145_145186


namespace area_of_given_triangle_l145_145400

-- Definitions of vertices
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (8, 6)
def C : ℝ × ℝ := (14, 2)

-- Function to calculate the area of the triangle given three vertices
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let v := (fst C - fst A, snd C - snd A)
  let w := (fst C - fst B, snd C - snd B)
  -- Cross product formula for 2D vectors gives the area of the parallelogram
  let cross_product := v.1 * w.2 - v.2 * w.1
  -- Area of the triangle is half the area of the parallelogram
  (1 / 2) * |cross_product|

-- The proof statement
theorem area_of_given_triangle : area_of_triangle A B C = 21 :=
by
  sorry

end area_of_given_triangle_l145_145400


namespace total_turtles_l145_145690

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end total_turtles_l145_145690


namespace find_d_l145_145099

theorem find_d (c : ℕ) (d : ℕ) : 
  (∀ n : ℕ, c = 3 ∧ ∀ k : ℕ, k ≠ 30 → ((1 : ℚ) * (29 / 30) * (28 / 30) = 203 / 225) → d = 203) := 
by
  intros
  sorry

end find_d_l145_145099


namespace maximum_triangle_area_within_circles_l145_145279

noncomputable def radius1 : ℕ := 71
noncomputable def radius2 : ℕ := 100
noncomputable def largest_triangle_area : ℕ := 24200

theorem maximum_triangle_area_within_circles : 
  ∃ (L : ℕ), L = largest_triangle_area ∧ 
             ∀ (r1 r2 : ℕ), r1 = radius1 → 
                             r2 = radius2 → 
                             L ≥ (r1 * r1 + 2 * r1 * r2) :=
by
  sorry

end maximum_triangle_area_within_circles_l145_145279


namespace triangle_equality_l145_145758

theorem triangle_equality (α : ℝ) (A B C D M N : Point ℝ) (AB AC CD : ℝ) 
  (hAB_AC : AB < AC)
  (hA_eq_2C : ∠A = 2 * ∠C)
  (hCD : CD = AB)
  (hL_parallel_AC : line_through B ∥ AC)
  (hL_meet_ext_bisector_A_at_M : L ⟂ ext_bisector ∠A ∧ M ∈ L)
  (hline_through_C_parallel_AB: line_through C ∥ AB ∧ N ∈ (line_through C))
  : distance M D = distance N D := 
sorry

end triangle_equality_l145_145758


namespace range_of_r_l145_145856

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the condition for the circle with variable radius r
def circle (x y r : ℝ) : Prop := (x - 3)^2 + y^2 = r^2

-- Define the foci of the hyperbola (F1 and F2)
def foci (x : ℝ) : ℝ × ℝ :=
  if x = 1 then (-2, 0) 
  else if x = 2 then (2, 0) 
  else (0, 0)

-- Define the dot product condition
def orthogonal (F1 F2 P : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

-- Main theorem statement:
theorem range_of_r (r : ℝ) (h : r > 0) :
  (∃ P : ℝ × ℝ, circle P.1 P.2 r ∧ orthogonal (foci 1) (foci 2) P) ↔ (1 ≤ r ∧ r ≤ 3) := sorry

end range_of_r_l145_145856


namespace unique_ids_div_10_l145_145321

noncomputable def num_unique_ids (n : ℕ) : ℕ :=
  let no_repeats := nat.factorial 7 / nat.factorial (7 - 5)
  let repeats_with_zero := 5 * (nat.choose 6 4) * nat.factorial 4
  no_repeats + repeats_with_zero

theorem unique_ids_div_10 : num_unique_ids 5 / 10 = 432 := by
  sorry

end unique_ids_div_10_l145_145321


namespace intersection_complement_A_B_l145_145513

def Universe : Set ℝ := Set.univ

def A : Set ℝ := {x | abs (x - 1) > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_complement_A_B :
  (Universe \ A) ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l145_145513


namespace gcd_228_1995_l145_145695

-- Define the gcd function according to the Euclidean algorithm
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a
  else gcd b (a % b)

-- Prove that the gcd of 228 and 1995 is 57
theorem gcd_228_1995 : gcd 228 1995 = 57 :=
by
  sorry

end gcd_228_1995_l145_145695


namespace quadrilateral_possible_conditions_l145_145383

variables (a b c d : ℝ) (a_le_d : a ≤ d) (AC_angle_bisector : ∀ (A B C : ℝ), True)
-- The actual proof requires explicit angle bisector functionality

theorem quadrilateral_possible_conditions :
  (a = d ∧ b = c) ∨
  (a ≠ d ∧ -- additional non-computable conditions proving uniqueness
   ∃ (AD AB CB_1 DB_1 : ℝ), (AD = d) ∧ (AB = a) ∧ (CB_1 = b) ∧ (DB_1 = d - a) ∧
   ∃ (triangle_CBD_exists : (True))
  ) ∨
  (a = d ∧ c ≠ b → False) :=
by
  sorry

end quadrilateral_possible_conditions_l145_145383


namespace dikembe_floss_problem_l145_145386

noncomputable def total_cost_and_leftover_floss
  (students_class1 : ℕ) (floss_per_student_class1 : ℚ)
  (students_class2 : ℕ) (floss_per_student_class2 : ℚ)
  (students_class3 : ℕ) (floss_per_student_class3 : ℚ)
  (meters_per_packet : ℚ) (packet_cost : ℚ)
  (discount_after_two_packets : ℚ) : (ℚ × ℚ) :=
let total_floss_needed_yards :=
  (students_class1 * floss_per_student_class1) +
  (students_class2 * floss_per_student_class2) +
  (students_class3 * floss_per_student_class3) in
let total_floss_needed_meters := total_floss_needed_yards * 0.9144 in
let packets_needed := (total_floss_needed_meters / meters_per_packet).ceil in
let normal_packets := min packets_needed 2 in
let discounted_packets := packets_needed - normal_packets in
let total_cost := normal_packets * packet_cost + discounted_packets * (packet_cost * (1 - discount_after_two_packets)) in
let total_floss_bought := packets_needed * meters_per_packet in
let leftover_floss := total_floss_bought - total_floss_needed_meters in
(total_cost, leftover_floss)

theorem dikembe_floss_problem :
  total_cost_and_leftover_floss 20 1.5 25 1.75 30 2 50 5 0.10 = (14.50, 27.737) :=
  sorry

end dikembe_floss_problem_l145_145386


namespace log_relationship_l145_145524

theorem log_relationship (a b : ℝ) (h1 : log a 2 < log b 2) (h2 : log b 2 < 0) : 0 < b ∧ b < a ∧ a < 1 :=
by
  sorry

end log_relationship_l145_145524


namespace train_length_approx_l145_145303

-- Given conditions
def speed_km_per_hr : ℝ := 40
def time_sec : ℝ := 27
def conversion_factor : ℝ := 5 / 18

-- Converting speed to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * conversion_factor

-- The main theorem: calculating the length of the train
theorem train_length_approx :
  (speed_m_per_s * time_sec ≈ 299.97) := by
  sorry

end train_length_approx_l145_145303


namespace range_of_a_l145_145903

theorem range_of_a (a : ℝ) 
  (h1 : a > 0) 
  (h2 : (∃ x : ℝ, x^2 * log a - 2 * x + 1 = 0) ∧ (∃ y : ℝ, y^2 * log a - 2 * y + 1 = 0) ∧ x ≠ y) :
  (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 10) := sorry

end range_of_a_l145_145903


namespace water_difference_l145_145228

variables (S H : ℝ)

theorem water_difference 
  (h_diff_after : S - 0.43 - (H + 0.43) = 0.88)
  (h_seungmin_more : S > H) :
  S - H = 1.74 :=
by
  sorry

end water_difference_l145_145228


namespace domain_of_composed_function_l145_145101

theorem domain_of_composed_function (f : ℝ → ℝ) (h : ∀ x, x ∈ Icc (-1 : ℝ) 1 → f x = f x) :
  ∀ x, x ∈ Icc (-real.sqrt 2) (real.sqrt 2) → (f (x^2 - 1)) = (f (x^2 - 1)) :=
by
  intros x hx
  sorry

end domain_of_composed_function_l145_145101


namespace neither_sufficient_nor_necessary_l145_145024

-- Definitions based on given conditions
def propA (a b : ℕ) : Prop := a + b ≠ 4
def propB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem statement (proof not required)
theorem neither_sufficient_nor_necessary (a b : ℕ) :
  ¬ (propA a b → propB a b) ∧ ¬ (propB a b → propA a b) := 
sorry

end neither_sufficient_nor_necessary_l145_145024


namespace find_N_l145_145092

theorem find_N : (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / (N : ℚ) → N = 1991 := by
sorry

end find_N_l145_145092


namespace quadrilateral_area_min_l145_145945

noncomputable def parabola_focus : Point := ⟨1, 0⟩  -- Example definition of the focus at (1, 0)

structure Point where
  x : ℝ
  y : ℝ

def on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

def dot_product (FA FB : Point) : ℝ :=
  FA.x * FB.x + FA.y * FB.y

def ext_intersect (A B : Point) : Point :=
  -- Placeholder definition for intersection, real definition requires more geometry
  ⟨1, 1⟩

theorem quadrilateral_area_min :
  ∀ (A B C D : Point),
    on_parabola A →
    on_parabola B →
    A ≠ ⟨0, 0⟩ →
    B ≠ ⟨0, 0⟩ →
    dot_product ⟨A.x - parabola_focus.x, A.y - parabola_focus.y⟩
                ⟨B.x - parabola_focus.x, B.y - parabola_focus.y⟩ = 0 →
    C = ext_intersect A parabola_focus →
    D = ext_intersect B parabola_focus →
    ∃ (area : ℝ), area = 32 :=
by
  sorry

end quadrilateral_area_min_l145_145945


namespace gcd_228_1995_l145_145698

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l145_145698


namespace count_correct_propositions_l145_145083

def proposition1 (m n : Line) (α : Plane) : Prop := (m ⟂ n ∧ m ⟂ α) → (n ∥ α)
def proposition2 (m n : Line) (α β : Plane) : Prop := (m ⟂ α ∧ n ⟂ β ∧ m ∥ n) → (α ∥ β)
def proposition3 (m n : Line) (α β : Plane) : Prop := (isSkew m n ∧ m ∈ α ∧ n ∈ β ∧ m ∥ β ∧ n ∥ α) → (α ∥ β)
def proposition4 (m n : Line) (α β : Plane) : Prop := (α ⟂ β ∧ α ∩ β = m ∧ n ∈ β ∧ n ⟂ m) → (n ⟂ α)

theorem count_correct_propositions 
  (m n : Line) (α β : Plane) 
  (p1 : ¬ proposition1 m n α) 
  (p2 : proposition2 m n α β) 
  (p3 : proposition3 m n α β) 
  (p4 : proposition4 m n α β) : 
  (p1 = false ∧ p2 = true ∧ p3 = true ∧ p4 = true) → 
  num_correct = 3 :=
by sorry

end count_correct_propositions_l145_145083


namespace atomic_weight_Cl_approx_l145_145002

noncomputable def mol_weight_CCl4 : ℝ := 152
noncomputable def atomic_weight_C : ℝ := 12.01

theorem atomic_weight_Cl_approx : 
  ∃ (atomic_weight_Cl : ℝ), mol_weight_CCl4 = atomic_weight_C + 4 * atomic_weight_Cl ∧ atomic_weight_Cl ≈ 35 :=
by
  sorry

end atomic_weight_Cl_approx_l145_145002


namespace find_triplets_of_real_numbers_l145_145803

theorem find_triplets_of_real_numbers (x y z : ℝ) :
  (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z) ∧ 
  (3 * x^2 + 2 * y^2 + z^2 = 240) → 
  (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) := 
sorry

end find_triplets_of_real_numbers_l145_145803


namespace window_width_l145_145176

theorem window_width : 
  ( ∀ (w h : ℕ) (k : ℕ),
    (w/h = 3/4) ∧ 
    (w = 3 * k) ∧ 
    (h = 4 * k) ∧ 
    (∃ t b l r : ℕ, 
      t = 3 ∧ b = 3 ∧ l = 3 ∧ r = 3 ∧
    (3 ≤ x) ∧ (x ≤ h)) ) → 
    ( width = (4 * 3 * k + 5 * 3) → 
      width = 51 ) ) 
:= 
by sorry

end window_width_l145_145176


namespace negation_of_proposition_l145_145663

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, sin x + |x| ≥ 0)) ↔ (∃ x : ℝ, sin x + |x| < 0) :=
by
  sorry

end negation_of_proposition_l145_145663


namespace jerry_money_left_l145_145127

example (money : ℕ) (mustard_oil_liters : ℕ) (mustard_oil_cost_per_liter : ℕ)
        (penne_pasta_pounds : ℕ) (penne_pasta_cost_per_pound : ℕ)
        (pasta_sauce_pounds : ℕ) (pasta_sauce_cost_per_pound : ℕ)
        (total_money_available : ℕ) : ℕ :=
  let cost_mustard_oil := mustard_oil_liters * mustard_oil_cost_per_liter
  let cost_penne_pasta := penne_pasta_pounds * penne_pasta_cost_per_pound
  let cost_pasta_sauce := pasta_sauce_pounds * pasta_sauce_cost_per_pound
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  total_money_available - total_cost

theorem jerry_money_left (h1 : mustard_oil_liters = 2)
                         (h2 : mustard_oil_cost_per_liter = 13)
                         (h3 : penne_pasta_pounds = 3)
                         (h4 : penne_pasta_cost_per_pound = 4)
                         (h5 : pasta_sauce_pounds = 1)
                         (h6 : pasta_sauce_cost_per_pound = 5)
                         (h7 : total_money_available = 50) :
  example 50 2 13 3 4 1 5 total_money_available = 7 :=
by {
  simp [example, h1, h2, h3, h4, h5, h6, h7],
  sorry
}

end jerry_money_left_l145_145127


namespace circle_equation_l145_145159

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l145_145159


namespace sum_alternating_series_l145_145898

def S (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ k, (k + 1) * (-1 : ℤ)^(k + 1))

theorem sum_alternating_series :
  S 1992 + S 1993 = 1 :=
sorry

end sum_alternating_series_l145_145898


namespace problem1_problem2_l145_145712

def triangle_valid (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_third_sides1 : List ℕ :=
  [5, 7, 9]

theorem problem1 : 
  ∃ n, 
    n = (valid_third_sides1.length) ∧
    valid_third_sides1.All (λ x, triangle_valid 7 3 x ∧ x % 2 = 1 ∧ 4 < x ∧ x < 10) :=
sorry

noncomputable def cost_per_cm : ℕ := 8

noncomputable def total_length : ℕ := 
  [15, 17, 19].sum

theorem problem2 : 
  ∃ cost_needed,
    cost_needed = (total_length * cost_per_cm) ∧
    cost_needed = 408 :=
sorry

end problem1_problem2_l145_145712


namespace evaluate_x_l145_145795

variable {R : Type*} [LinearOrderedField R]

theorem evaluate_x (m n k x : R) (hm : m ≠ 0) (hn : n ≠ 0) (h : m ≠ n) (h_eq : (x + m)^2 - (x + n)^2 = k * (m - n)^2) :
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 :=
by
  sorry

end evaluate_x_l145_145795


namespace no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l145_145714

-- Problem 1: There is no value of m that makes the lines parallel.
theorem no_solution_for_parallel_lines (m : ℝ) :
  ¬ ∃ m, (2 * m^2 + m - 3) / (m^2 - m) = 1 := sorry

-- Problem 2: The values of a that make the lines perpendicular.
theorem values_of_a_for_perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔ (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) := sorry

end no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l145_145714


namespace lower_bound_fraction_sum_l145_145054

open Real

theorem lower_bound_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (1 / (3 * a) + 3 / b) ≥ 8 / 3 :=
by 
  sorry

end lower_bound_fraction_sum_l145_145054


namespace time_after_9876_seconds_l145_145124

noncomputable def currentTime : Nat := 2 * 3600 + 45 * 60 + 0
noncomputable def futureDuration : Nat := 9876
noncomputable def resultingTime : Nat := 5 * 3600 + 29 * 60 + 36

theorem time_after_9876_seconds : 
  (currentTime + futureDuration) % (24 * 3600) = resultingTime := 
by 
  sorry

end time_after_9876_seconds_l145_145124


namespace february_first_2015_is_sunday_l145_145126

theorem february_first_2015_is_sunday : 
  (day_of_week : ℕ) → (january_first_2015 : ℕ) 
  (thursday : nat.day_of_week = 4) 
  (january_first_2015 = thursday) → 
  (days_in_january : ℕ) → (days_in_january = 31) → 
  (february_first_2015 : ℕ) → 
  (february_first_2015 = (january_first_2015 + 30) % 7) → 
  (february_first_2015 = 0) :=
by
  sorry

end february_first_2015_is_sunday_l145_145126


namespace proof_problem_g_4971657808_l145_145150

def g : ℕ → ℕ := sorry 

theorem proof_problem_g_4971657808 :
  (∃ (n s : ℕ), n * s = 4971657808 ∧
    (∀ a b : ℕ, 3 * g (a^2 + b^2 + 1) = g a ^ 3 + g b ^ 3) ∧ 
      n = {x : ℕ | ∃ z : ℕ, g(10) = z}.to_finset.card ∧
      s = {x : ℕ | ∃ z : ℕ, g(10) = z}.to_finset.sum id) :=
sorry

end proof_problem_g_4971657808_l145_145150


namespace area_of_square_l145_145247

noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle_given_length_and_breadth (L B : ℝ) : ℝ := L * B

theorem area_of_square (r : ℝ) (B : ℝ) (A : ℝ) 
  (h_length : length_of_rectangle r = (2 / 5) * r) 
  (h_breadth : B = 10) 
  (h_area : A = 160) 
  (h_rectangle_area : area_of_rectangle_given_length_and_breadth ((2 / 5) * r) B = 160) : 
  r = 40 → (r ^ 2 = 1600) := 
by 
  sorry

end area_of_square_l145_145247


namespace crossing_time_approx_11_16_seconds_l145_145285

noncomputable def length_train_1 : ℝ := 140 -- length of the first train in meters
noncomputable def length_train_2 : ℝ := 170 -- length of the second train in meters
noncomputable def speed_train_1_km_hr : ℝ := 60 -- speed of the first train in km/hr
noncomputable def speed_train_2_km_hr : ℝ := 40 -- speed of the second train in km/hr

noncomputable def speed_conversion_factor : ℝ := 5 / 18 -- conversion factor from km/hr to m/s

-- convert speeds from km/hr to m/s
noncomputable def speed_train_1_m_s : ℝ := speed_train_1_km_hr * speed_conversion_factor
noncomputable def speed_train_2_m_s : ℝ := speed_train_2_km_hr * speed_conversion_factor

-- calculate relative speed in m/s (since they are moving in opposite directions)
noncomputable def relative_speed_m_s : ℝ := speed_train_1_m_s + speed_train_2_m_s

-- total distance to be covered
noncomputable def total_distance : ℝ := length_train_1 + length_train_2

-- calculate the time to cross each other
noncomputable def crossing_time : ℝ := total_distance / relative_speed_m_s

theorem crossing_time_approx_11_16_seconds : abs (crossing_time - 11.16) < 0.01 := by
    sorry

end crossing_time_approx_11_16_seconds_l145_145285


namespace petya_has_higher_chance_of_winning_l145_145200

noncomputable def vasya_win_probability : ℝ := 0.46
noncomputable def petya_win_probability : ℝ := 1 - vasya_win_probability

theorem petya_has_higher_chance_of_winning 
  (h: vasya_win_probability = 1 - 0.54) : petya_win_probability > 0.5 :=
by
  have hp : petya_win_probability = 1 - vasya_win_probability := rfl
  have h_vasya : vasya_win_probability = 0.46 := h
  rw [h_vasya] at hp
  have h_petya : petya_win_probability = 1 - 0.46 := hp
  simp [h_petya]
  exact by norm_num

end petya_has_higher_chance_of_winning_l145_145200


namespace time_to_cross_bridge_l145_145873

noncomputable def train_length := 300  -- in meters
noncomputable def train_speed_kmph := 72  -- in km/h
noncomputable def bridge_length := 1500  -- in meters

-- Define the conversion from km/h to m/s
noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600  -- in m/s

-- Define the total distance to be traveled
noncomputable def total_distance := train_length + bridge_length  -- in meters

-- Define the time to cross the bridge
noncomputable def time_to_cross := total_distance / train_speed_mps  -- in seconds

theorem time_to_cross_bridge : time_to_cross = 90 := by
  -- skipping the proof
  sorry

end time_to_cross_bridge_l145_145873


namespace num_selections_is_72_l145_145915

-- Define the volunteers
inductive Volunteer
| A | B | C | D | E

-- Define the tasks
inductive Task
| translation | tour_guide | etiquette | driving

-- Define the constraints
def can_undertake (v : Volunteer) (t : Task) : Prop :=
  match v, t with
  | .A, .driving => False
  | .B, .driving => False
  | _, _ => True

-- Definition that translates the problem statement
noncomputable def num_selection_schemes : ℕ :=
  if h : (∀ (assign : Volunteer → Task), 
            -- Each task is assigned to one volunteer
            (∀ t1 t2 : Task, t1 ≠ t2 → ∃ v1 v2 : Volunteer, assign v1 = t1 ∧ assign v2 = t2 ∧ v1 ≠ v2) ∧
            -- Each volunteer takes at most one task
            (∀ (v : Volunteer), ∃ t : Task, assign v = t) ∧
            -- Only valid volunteers get invalid tasks
            (∀ (v : Volunteer) (t : Task), can_undertake v t ∨ assign v ≠ t)) 
    then 72 else 0

-- Theorem stating the number of different selection schemes
theorem num_selections_is_72 : num_selection_schemes = 72 := 
by 
  -- Not actually constructing the proof details here, marking with 'sorry'.
  sorry

end num_selections_is_72_l145_145915


namespace positive_int_solutions_l145_145436

theorem positive_int_solutions : 
  { (x, y, z) ∈ ℕ × ℕ × ℕ | 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / x + 2 / y - 3 / z = 1) } =
  { (2, 1, 2), (k, 2, 3 * k), (2, 3, 18), (1, 2 * k, 3 * k), (2, 2, 6) | k ∈ ℕ } :=
sorry

end positive_int_solutions_l145_145436


namespace min_points_distance_square_l145_145545

noncomputable def square_partition := 
  1/2 * sqrt 32 < sqrt 5

theorem min_points_distance_square : 
  ∀ (n : ℕ), n ≥ 7 → 
  ∃ p : fin n → (ℝ × ℝ), 
  (∀ (i j : fin n), i ≠ j → 
    dist (p i) (p j) ≤ sqrt 5) :=
by sorry

end min_points_distance_square_l145_145545


namespace problem1_problem2_l145_145374

theorem problem1 : (-0.125) ^ 2023 * (8:ℝ) ^ 2024 = -8 := 
by sorry

theorem problem2 : (∛(-27) + (5:ℝ) - (2/3) * sqrt(9/4)) = 1 := 
by sorry

end problem1_problem2_l145_145374


namespace unique_perpendicular_plane_through_skew_line_l145_145994

open EuclideanGeometry

def skew_lines (a b : Line) : Prop := 
  ¬ ∃ (P : Point), (P ∈ a) ∧ (P ∈ b)

theorem unique_perpendicular_plane_through_skew_line (a b : Line) (h_skew : skew_lines a b) (h_perp : a ⊥ b) :
  ∃! (α : Plane), (b ⊂ α) ∧ (α ⊥ a) :=
sorry

end unique_perpendicular_plane_through_skew_line_l145_145994


namespace problem_statement_l145_145784

-- Definition of the function f with the given condition
def satisfies_condition (f : ℝ → ℝ) := ∀ (α β : ℝ), f (α + β) - (f α + f β) = 2008

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ (x : ℝ), f (-x) = -f x

-- Main statement to prove in Lean
theorem problem_statement (f : ℝ → ℝ) (h : satisfies_condition f) : is_odd (fun x => f x + 2008) :=
sorry

end problem_statement_l145_145784


namespace circumcenter_PQH_lies_on_median_l145_145132

variable {O A B C H P Q : Type} [IsTriangle O A B C] [IsCircumcenter O A B C]
variable [IsOrthocenter H A B C]
variable [Intersects OA (altitudeThrough B) P] [Intersects OA (altitudeThrough C) Q]

theorem circumcenter_PQH_lies_on_median :
  ∃ M : Point, IsMedian M A B C ∧ LiesOnCircumcenterOfPQH M :=
sorry

end circumcenter_PQH_lies_on_median_l145_145132


namespace geom_mean_inequality_l145_145965

theorem geom_mean_inequality (n : ℕ) (h : n > 1) (x : Fin n → ℝ) 
    (pos_x : ∀ i, 0 < x i) 
    (sum_inv : (∑ i, 1 / (1 + x i)) = 1) :
    (∏ i, x i) ≥ (n - 1) ^ n :=
by
  sorry

end geom_mean_inequality_l145_145965


namespace sum_super_cool_rectangle_areas_eq_84_l145_145340

theorem sum_super_cool_rectangle_areas_eq_84 :
  ∀ (a b : ℕ), 
  (a * b = 3 * (a + b)) → 
  ∃ (S : ℕ), 
  S = 84 :=
by
  sorry

end sum_super_cool_rectangle_areas_eq_84_l145_145340


namespace shortest_ribbon_length_l145_145565

theorem shortest_ribbon_length :
  ∃ (L : ℕ), (∀ (n : ℕ), n = 2 ∨ n = 5 ∨ n = 7 → L % n = 0) ∧ L = 70 :=
by
  sorry

end shortest_ribbon_length_l145_145565


namespace circle_equation_l145_145162

theorem circle_equation (M : ℝ × ℝ) :
  (∃ a : ℝ, M = (a, 1 - 2 * a) ∧ 2 * M.1 + M.2 - 1 = 0) ∧
  (distance M (3, 0) = distance M (0, 1)) →
  (∃ r : ℝ, (x - 1)^2 + (y + 1)^2 = r^2) :=
begin
  sorry
end

end circle_equation_l145_145162


namespace cylinder_lateral_area_l145_145246

-- Define the cylindrical lateral area calculation
noncomputable def lateral_area_of_cylinder (d h : ℝ) : ℝ := (2 * Real.pi * (d / 2)) * h

-- The statement of the problem in Lean 4.
theorem cylinder_lateral_area : lateral_area_of_cylinder 4 4 = 16 * Real.pi := by
  sorry

end cylinder_lateral_area_l145_145246


namespace piastres_in_6th_purse_l145_145936

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end piastres_in_6th_purse_l145_145936


namespace evaluate_expression_l145_145392

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) :
  a^2 * b^3 * c = 5 / 256 :=
by
  rw [ha, hb, hc]
  norm_num

end evaluate_expression_l145_145392


namespace necessary_sufficient_d_n_Sn_value_l145_145466

-- Definitions
def seq_condition (a : ℕ → ℕ) : Prop :=
a 1 = 3 ∧ ∀ n ≥ 2, a n = 3^n - 1

def max_val_A (a : ℕ → ℕ) (n : ℕ) : ℕ :=
list.max (list.of_fn (λ i, a (i+1)) n)

def min_val_B (a : ℕ → ℕ) (n : ℕ) : ℕ :=
list.minimum (list.of_fn (λ i, a (i + n + 1)) (list.length (list.of_fn (λ i, a (i + n + 1)) (nat.succ n))))

def d_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
max_val_A a n - min_val_B a n

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem necessary_sufficient_d_n (a : ℕ → ℕ) (d : ℕ) :
  (∀ n, d_n a n = -d) ↔ is_arithmetic_sequence a d :=
sorry

-- General term for sum Sn
def a_n := λ n : ℕ, 2^n

def d_n_for_a_n (n : ℕ) : ℤ :=
((max_val_A a_n n : ℤ) - (min_val_B a_n n))

def Sn (n : ℕ) : ℤ :=
(∑ i in finset.range n, (-i^2) / d_n_for_a_n i : ℕ)

theorem Sn_value (n : ℕ) :
  Sn n = 6 - (4 * n + 6 + n^2) / 2^n :=
sorry

end necessary_sufficient_d_n_Sn_value_l145_145466


namespace coupon_X_comparison_l145_145751

noncomputable def discountX (P : ℝ) := 0.20 * P
noncomputable def discountY (P : ℝ) := 40.0
noncomputable def discountZ (P : ℝ) := if P > 150 then 0.30 * (P - 150) else 0

theorem coupon_X_comparison (P : ℝ) (h : P > 100) :
  let a := 200
  let b := 450
  (∀ P, (discountX P) ≥ discountY P ∨ (discountX P) ≥ discountZ P) →
  b - a = 250 :=
by
  intros
  sorry

end coupon_X_comparison_l145_145751


namespace hyperbola_point_distance_l145_145074

-- Define the hyperbola with given parameters
def is_hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci distances from the center of the hyperbola
def foci_distances (a c : ℝ) : ℝ := sqrt(c^2 - a^2)

-- Define the conditions given in the problem statement
def hyperbola_conditions (P: ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  is_hyperbola P.1 P.2 ∧ dist P F1 = 3 ∧ dist P F2 = 9

theorem hyperbola_point_distance (P F1 F2 : ℝ × ℝ) :
  hyperbola_conditions P F1 F2 → dist P F2 = 9 := 
sorry

end hyperbola_point_distance_l145_145074


namespace shaded_region_is_hyperbolas_l145_145105

theorem shaded_region_is_hyperbolas (T : ℝ) (hT : T > 0) :
  (∃ (x y : ℝ), x * y = T / 4) ∧ (∃ (x y : ℝ), x * y = - (T / 4)) :=
by
  sorry

end shaded_region_is_hyperbolas_l145_145105


namespace numbers_meeting_conditions_l145_145180

def is_abb_pattern (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (digits.nth 0 % 2 = 1) ∧ -- first digit is odd (A)
  (digits.nth 1 % 2 = 0) ∧ -- second digit is even (B)
  (digits.nth 2 % 2 = 0) -- third digit is even (B)

def is_vvg_pattern (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (digits.nth 0 % 3 = 0) ∧ -- first digit is divisible by 3 (V)
  (digits.nth 1 % 3 = 0) ∧ -- second digit is divisible by 3 (V)
  (digits.nth 2 % 3 ≠ 0) -- third digit is not divisible by 3 (G)

theorem numbers_meeting_conditions :
  {n : ℕ // 500 < n ∧ n < 1000 ∧ is_abb_pattern n ∧ is_vvg_pattern n} =
  ({902, 904, 908, 962, 964, 968} : set ℕ) :=
by sorry

end numbers_meeting_conditions_l145_145180


namespace lateral_surface_area_of_frustum_l145_145820

theorem lateral_surface_area_of_frustum (slant_height : ℝ) (ratio : ℕ × ℕ) (central_angle_deg : ℝ)
  (h_slant_height : slant_height = 10) 
  (h_ratio : ratio = (2, 5)) 
  (h_central_angle_deg : central_angle_deg = 216) : 
  ∃ (area : ℝ), area = (252 * Real.pi / 5) := 
by 
  sorry

end lateral_surface_area_of_frustum_l145_145820


namespace speed_of_mans_train_l145_145325

-- Conditions
def goods_train_length : ℝ := 280 -- in meters
def passing_time : ℝ := 9 -- in seconds
def goods_train_speed_kmph : ℝ := 62 -- in km/h

-- Conversion factor
def kmph_to_mps (speed : ℝ) := speed / 3.6

-- Goods train speed in m/s
def goods_train_speed_mps : ℝ := kmph_to_mps goods_train_speed_kmph

-- Relative speed in m/s
def relative_speed : ℝ := goods_train_length / passing_time

-- Question: speed of man's train in km/h
def mans_train_speed_kmph := (relative_speed - goods_train_speed_mps) * 3.6

theorem speed_of_mans_train :
  mans_train_speed_kmph = 50 :=
  by
  sorry

end speed_of_mans_train_l145_145325


namespace inequality_holds_l145_145616

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 :=
by
  sorry

end inequality_holds_l145_145616


namespace lambda_mu_squared_l145_145119

noncomputable def triangle_ABC (A B C : ℝ³) :=
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) ∧
  dist A B = dist A C ∧
  (B - A) ⬝ (C - A) = 0

theorem lambda_mu_squared (A B C E F M : ℝ³) (λ μ : ℝ) (hABC : triangle_ABC A B C)
    (hAE : E = λ • (B - A))
    (hAF : F = μ • (C - A))
    (hM : M = (E + F) / 2)
    (hAM : dist A M = (1 / 2) * dist A B) :
    λ^2 + μ^2 = 1 :=
begin
  sorry
end

end lambda_mu_squared_l145_145119


namespace find_d1_over_d2_l145_145235

variables {k c1 c2 d1 d2 : ℝ}
variables (c1_nonzero : c1 ≠ 0) (c2_nonzero : c2 ≠ 0) 
variables (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0)
variables (h1 : c1 * d1 = k) (h2 : c2 * d2 = k)
variables (h3 : c1 / c2 = 3 / 4)

theorem find_d1_over_d2 : d1 / d2 = 4 / 3 :=
sorry

end find_d1_over_d2_l145_145235


namespace Smarties_remainder_l145_145395

theorem Smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end Smarties_remainder_l145_145395


namespace find_sum_principal_l145_145706

theorem find_sum_principal (P R : ℝ) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 → P = 300 :=
by
  sorry

end find_sum_principal_l145_145706


namespace div_by_11_l145_145932

theorem div_by_11 (x y : ℤ) (k : ℤ) (h : 14 * x + 13 * y = 11 * k) : 11 ∣ (19 * x + 9 * y) :=
by
  sorry

end div_by_11_l145_145932


namespace area_of_figure_l145_145923

theorem area_of_figure 
  (S Q : ℝ) 
  (α : ℝ) 
  (H1 : ∃ (T : ℝ), S = T * cos (2 * α) ∧ Q = T * cos α) :
  ∃ T : ℝ, T = 1 / 2 * (sqrt(S^2 + 8 * Q^2) - S) :=
by 
  sorry

end area_of_figure_l145_145923


namespace should_not_call_l145_145396

-- Definitions of conditions and corresponding Lean properties
def received_sms (Alexander : Type) : Prop :=
  ∃ number : String, ∃ message : String,
    message = "Alexander, your mobile number has participated in a prize draw. You can learn the details of claiming your prize by calling the phone number " ++ number

-- The main theorem to be proved
theorem should_not_call (Alexander : Type) (h : received_sms Alexander) : ¬ (call_number Alexander) :=
sorry

end should_not_call_l145_145396


namespace two_digit_prime_sum_count_l145_145887

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_prime_sum_count : 
  let prime_sums := {2, 3, 5, 7, 11, 13, 17}
  ∑ k in (Finset.filter (λ n, is_prime (sum_of_digits n)) (Finset.filter two_digit (Finset.range 100))), 1 = 31 := 
by
  sorry

end two_digit_prime_sum_count_l145_145887


namespace johns_eggs_per_week_l145_145571

noncomputable def total_emus (pens : ℕ) (emus_per_pen : ℕ) : ℕ :=
  pens * emus_per_pen

noncomputable def female_emus (total : ℕ) : ℕ :=
  total / 2

noncomputable def eggs_per_week (females : ℕ) (days_per_week : ℕ) : ℕ :=
  females * days_per_week

theorem johns_eggs_per_week :
  let pens := 4 in
  let emus_per_pen := 6 in
  let days_per_week := 7 in
  let total := total_emus pens emus_per_pen in
  let females := female_emus total in
  eggs_per_week females days_per_week = 84 :=
by
  sorry

end johns_eggs_per_week_l145_145571


namespace police_coverage_l145_145251

def intersection := {A, B, C, D, E, F, G, H, I, J, K}
def street_set := { "horizontal1" := {"A", "B", "C", "D"}
                  , "horizontal2" := {"E", "F", "G"}
                  , "horizontal3" := {"H", "I", "J", "K"}
                  , "vertical1"   := {"A", "E", "H"}
                  , "vertical2"   := {"B", "F", "I"}
                  , "vertical3"   := {"D", "G", "J"}
                  , "diagonal1"   := {"H", "F", "C"}
                  , "diagonal2"   := {"C", "G", "K"} }

def placements := {"B", "G", "H"}

theorem police_coverage (s : street_set) (i : intersection) (p : placements) :
  (B ∈ p) ∧ (G ∈ p) ∧ (H ∈ p) → ∀ (str ∈ s), ∃ (inter ∈ str), inter ∈ p := sorry

end police_coverage_l145_145251


namespace tangent_line_eq_a_two_range_of_a_l145_145072

open Real

-- Condition definitions
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * (a^2 - a) * log x
def g (x : ℝ) (a : ℝ) : ℝ := 2 * a^2 * log x

-- Problem statement 1: Tangent line equation for a = 2
theorem tangent_line_eq_a_two :
  ∃ (m c : ℝ), (∀ (x : ℝ), m * x + c = -2 * x + 3) :=
sorry

-- Problem statement 2: Range of a for f(x) > 2g(x)
theorem range_of_a (a : ℝ) (x : ℝ) (h : 1 < x) (ha : a ≤ 1/2) :
  f x a > 2 * g x a → (1 - sqrt(1 + 12*exp 1))/6 < a ∧ a <= 1/2 :=
sorry

end tangent_line_eq_a_two_range_of_a_l145_145072


namespace minimum_sum_of_reciprocals_l145_145154

open BigOperators

theorem minimum_sum_of_reciprocals (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i)
    (h_sum : ∑ i, b i = 1) :
    ∑ i, 1 / (b i) ≥ 225 := sorry

end minimum_sum_of_reciprocals_l145_145154


namespace lucas_purchase_l145_145178

-- Define the variables and assumptions.
variables (a b c : ℕ)
variables (h1 : a + b + c = 50) (h2 : 50 * a + 400 * b + 500 * c = 10000)

-- Goal: Prove that the number of 50-cent items (a) is 30.
theorem lucas_purchase : a = 30 :=
by sorry

end lucas_purchase_l145_145178


namespace fourth_of_may_is_sunday_l145_145899

theorem fourth_of_may_is_sunday (x : ℕ) (h : 5 * x + 70 = 80) : 
  nat.succ (nat.succ (x)) = 4 ∧ nat.succ (nat.succ (x)) % 7 = 0 :=
sorry

end fourth_of_may_is_sunday_l145_145899


namespace subset_to_union_eq_l145_145830

open Set

variable {α : Type*} (A B : Set α)

theorem subset_to_union_eq (h : A ∩ B = A) : A ∪ B = B :=
by
  sorry

end subset_to_union_eq_l145_145830


namespace kevin_ends_with_604_cards_l145_145938

theorem kevin_ends_with_604_cards : 
  ∀ (initial_cards found_cards : ℕ), initial_cards = 65 → found_cards = 539 → initial_cards + found_cards = 604 :=
by
  intros initial_cards found_cards h_initial h_found
  sorry

end kevin_ends_with_604_cards_l145_145938


namespace max_sum_eq_double_n_l145_145045

theorem max_sum_eq_double_n (n : ℕ) (x : ℕ → ℕ) 
    (h1 : 2 ≤ n) 
    (h2 : ∀ i j, i < j → x i ≤ x j)
    (h3 : (∑ i in Finset.range n, x i) = (∏ i in Finset.range n, x i)) : 
    (∑ i in Finset.range n, x i) = 2 * n :=
sorry

end max_sum_eq_double_n_l145_145045


namespace correct_proposition_l145_145056

variables {α β : Type} [plane α] [plane β]
variable (l : line)

-- Definitions for propositions
def prop1 (l : line) (α β : plane) : Prop := (l ⊆ β ∧ α ⊥ β) → (l ⊥ α)
def prop2 (l : line) (α β : plane) : Prop := (l ⊥ β ∧ α ∥ β) → (l ⊥ α)
def prop3 (l : line) (α β : plane) : Prop := (l ⊥ β ∧ α ⊥ β) → (l ∥ α)
def prop4 (m : line) (l : line) (α β : plane) : Prop := (α ∩ β = m ∧ l ∥ m) → (l ∥ α)

-- Theorem statement
theorem correct_proposition (l : line) (α β : plane) :
  prop2 l α β :=
sorry

end correct_proposition_l145_145056


namespace minutes_in_hours_l145_145087

theorem minutes_in_hours (h : ℝ) (m : ℝ) (H : h = 3.5) (M : m = 60) : h * m = 210 := by
  sorry

end minutes_in_hours_l145_145087


namespace triangle_inequality_l145_145633

open Real

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) :
  sin A * cos C + A * cos B > 0 :=
by
  sorry

end triangle_inequality_l145_145633


namespace palindromic_primes_sum_eq_99_l145_145604

def is_prime (n : ℕ) : Prop := nat.prime n
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

def palindromic_prime_condition (p : ℕ) : Prop :=
  is_prime p ∧ p >= 10 ∧ p < 100 ∧
  is_prime (reverse_digits p) ∧
  is_prime (p + 2)

theorem palindromic_primes_sum_eq_99 :
  ∑ p in {p | palindromic_prime_condition p}.to_finset, p = 99 := 
sorry

end palindromic_primes_sum_eq_99_l145_145604


namespace exists_two_participants_with_disjoint_solutions_l145_145694

theorem exists_two_participants_with_disjoint_solutions
  (candidates : Finset ℕ) (num_candidates : candidates.card = 200)
  (problems : Finset ℕ) (num_problems : problems.card = 6)
  (solved : ℕ → Finset ℕ) 
  (hsolved : ∀ p ∈ problems, (solved p).card < 80) :
  ∃ (a b ∈ candidates), (a ≠ b) ∧ (∀ p ∈ problems, ¬(a ∈ solved p ∧ b ∈ solved p)) :=
by
  sorry

end exists_two_participants_with_disjoint_solutions_l145_145694


namespace avg_sum_sub_eq_avg_sum_avg_sub_l145_145270

variable {X Y Z : Type}
variable {x1 x2 x3 : X}
variable {y1 y2 y3 : Y}
variable {z1 z2 z3 : Z}
variable {a b c : ℝ}
variable [Add X] [Add Y] [Add Z] [Sub Z] [HasSmul ℝ X] [HasSmul ℝ Y] [HasSmul ℝ Z] [HasSmul ℝ ℝ]
variable [Nonempty X] [Nonempty Y] [Nonempty Z]

theorem avg_sum_sub_eq_avg_sum_avg_sub (H1 : 1/3 • ((x1 + x2 + x3) : X) = a)
                                       (H2 : 1/3 • ((y1 + y2 + y3) : Y) = b)
                                       (H3 : 1/3 • ((z1 + z2 + z3) : Z) = c) :
  1/3 • ((x1 + y1 - z1) + (x2 + y2 - z2) + (x3 + y3 - z3)) = a + b - c :=
sorry

end avg_sum_sub_eq_avg_sum_avg_sub_l145_145270


namespace common_chord_length_l145_145870

theorem common_chord_length : 
  let c1 := { x | x.1^2 + x.2^2 + x.1 - 2 * x.2 - 20 = 0 }
  let c2 := { x | x.1^2 + x.2^2 = 25 }
  ∃ (chord_length : ℝ), chord_length = 4 * Real.sqrt 5 :=
by
  -- Define the circles
  let c1 := { x : ℝ × ℝ | x.1^2 + x.2^2 + x.1 - 2 * x.2 - 20 = 0 }
  let c2 := { x : ℝ × ℝ | x.1^2 + x.2^2 = 25 }

  -- The correct answer length of the common chord
  refine ⟨4 * Real.sqrt 5, _⟩
  sorry

end common_chord_length_l145_145870


namespace largest_integer_remainder_l145_145406

theorem largest_integer_remainder :
  ∃ (a : ℤ), a < 61 ∧ a % 6 = 5 ∧ ∀ b : ℤ, b < 61 ∧ b % 6 = 5 → b ≤ a :=
by
  sorry

end largest_integer_remainder_l145_145406


namespace a_b_c_sum_l145_145639

-- Definitions of the conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

theorem a_b_c_sum (a b c : ℝ) :
  (∀ x : ℝ, f (x + 4) a b c = 4 * x^2 + 9 * x + 5) ∧ (∀ x : ℝ, f x a b c = a * x^2 + b * x + c) →
  a + b + c = 14 :=
by
  intros h
  sorry

end a_b_c_sum_l145_145639


namespace no_triangles_if_all_horizontal_removed_l145_145444

/-- 
Given a figure that consists of 40 identical toothpicks, making up a symmetric figure with 
additional rows on the top and bottom. We need to prove that removing all 40 horizontal toothpicks 
ensures there are no remaining triangles in the figure.
-/
theorem no_triangles_if_all_horizontal_removed
  (initial_toothpicks : ℕ)
  (horizontal_toothpicks_in_figure : ℕ) 
  (rows : ℕ)
  (top_row : ℕ)
  (second_row : ℕ)
  (third_row : ℕ)
  (fourth_row : ℕ)
  (bottom_row : ℕ)
  (additional_rows : ℕ)
  (triangles_for_upward : ℕ)
  (triangles_for_downward : ℕ):
  initial_toothpicks = 40 →
  horizontal_toothpicks_in_figure = top_row + second_row + third_row + fourth_row + bottom_row →
  rows = 5 →
  top_row = 5 →
  second_row = 10 →
  third_row = 10 →
  fourth_row = 10 →
  bottom_row = 5 →
  additional_rows = 2 →
  triangles_for_upward = 15 →
  triangles_for_downward = 10 →
  horizontal_toothpicks_in_figure = 40 → 
  ∀ toothpicks_removed, toothpicks_removed = 40 →
  no_triangles_remain :=
by
  intros
  sorry

end no_triangles_if_all_horizontal_removed_l145_145444


namespace part1_part2_l145_145716

-- Proof for part 1
theorem part1 (x : ℤ) : (x - 1 ∣ x - 3 ↔ (x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3)) :=
by sorry

-- Proof for part 2
theorem part2 (x : ℤ) : (x + 2 ∣ x^2 + 3 ↔ (x = -9 ∨ x = -3 ∨ x = -1 ∨ x = 5)) :=
by sorry

end part1_part2_l145_145716


namespace solve_problem_l145_145905

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) / 2

noncomputable def problem_statement : Prop :=
  let A := (0, 12) in
  let B := (5, 0) in
  let C := (0, 0) in
  let O := ((5 / 2), 6) in
  let I := (2, 2) in
  let M := (4, 4) in
  area_triangle M O I = 7 / 2

theorem solve_problem : problem_statement := 
by
  sorry

end solve_problem_l145_145905


namespace net_gain_difference_l145_145333

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l145_145333


namespace geometric_sequence_a6_l145_145042

noncomputable def a_sequence (n : ℕ) : ℝ := 1 * 2^(n-1)

theorem geometric_sequence_a6 (S : ℕ → ℝ)
  (h1 : S 10 = 3 * S 5)
  (h2 : ∀ n, S n = (1 - 2^n) / (1 - 2))
  (h3 : a_sequence 1 = 1) :
  a_sequence 6 = 2 := by
  sorry

end geometric_sequence_a6_l145_145042


namespace cost_price_is_2800_l145_145721

-- Define the cost price of the book
def CostPrice : ℝ := 2800

-- Conditions
variable C : ℝ
hypothesis h1 : (1.10 : ℝ) * C + (140 : ℝ) = (1.15 : ℝ) * C

-- Theorem: Prove that the cost price of the book is $2800
theorem cost_price_is_2800 (h : h1) : C = CostPrice :=
by
  sorry

end cost_price_is_2800_l145_145721


namespace volume_ratio_l145_145036

-- Definitions based on conditions:
def regular_triangular_pyramid := Type
def height (P: regular_triangular_pyramid) := ℝ
def midpoint (PO : ℝ) := ℝ

-- Problem conditions:
axiom height_PO : ℝ
axiom M_midpoint_PO : midpoint height_PO

-- Prove statement:
theorem volume_ratio (PABC : regular_triangular_pyramid) 
  (PO : height PABC) 
  (M : midpoint PO) 
  (plane_AM_parallel_BC : Prop) 
  (volume_ratio_parts : Rat) :
  plane_AM_parallel_BC → 
  volume_ratio_parts = 4 / 21 := 
sorry

end volume_ratio_l145_145036


namespace distance_S_to_PQR_l145_145217

open euclidean_geometry -- assuming we use some geometry library

theorem distance_S_to_PQR {P Q R S : Point} 
  (h1 : S.distance P = 10) 
  (h2 : S.distance Q = 10)
  (h3 : S.distance R = 8) 
  (h4 : S.perpendicular P Q)
  (h5 : S.perpendicular P R)
  (h6 : S.perpendicular Q R) :
  S.distance_to_plane P Q R = 8 :=
by 
  sorry

end distance_S_to_PQR_l145_145217


namespace range_of_x2_plus_y2_l145_145655

noncomputable def f : ℝ → ℝ := sorry

lemma increasing_function (a b : ℝ) (h : a < b) : f(a) < f(b) := sorry

lemma symmetric_about_origin (x : ℝ) : f(x - 2010) = -f(-(x - 2010)) := sorry

theorem range_of_x2_plus_y2 (x y : ℝ) 
    (h_sym : ∀ x, f(x - 2010) = -f(- (x - 2010)))
    (h_inc : ∀ a b, a < b → f(a) < f(b))
    (h_ineq : f(x^2 - 6x) + f(y^2 - 8y + 24) < 0) :
    16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end range_of_x2_plus_y2_l145_145655


namespace sum_not_prime_30_l145_145330

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_not_prime_30 (p1 p2 : ℕ) (hp1 : is_prime p1) (hp2 : is_prime p2) (h : p1 + p2 = 30) : false :=
sorry

end sum_not_prime_30_l145_145330


namespace Q_trajectory_l145_145471

-- Declaring conditions
def point_P : ℝ × ℝ := (3, 4)
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 64
def angle_APB (A P B : ℝ × ℝ) : Prop := ∠APB = π/2

-- Statement of the proof problem
theorem Q_trajectory :
  ∀ (A B Q : ℝ × ℝ),
    circle_equation A.1 A.2 →
    circle_equation B.1 B.2 →
    angle_APB A point_P B →
    (Q = (A.1 + B.1 - 3, A.2 + B.2 - 4)) →
    circle_equation Q.1 Q.2 =
    (Q.1^2 + Q.2^2 = 103) :=
by
  intro A B Q hA hB hAPB hQ
  sorry

end Q_trajectory_l145_145471


namespace det_S_l145_145949

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], ![0, 1]]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![ ![real.sqrt 2 / 2, -real.sqrt 2 / 2],
     ![real.sqrt 2 / 2, real.sqrt 2 / 2] ]

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ := A ⬝ B

theorem det_S : S.det = 2 :=
sorry

end det_S_l145_145949


namespace petya_has_higher_chance_of_winning_l145_145203

noncomputable def vasya_win_probability : ℝ := 0.46
noncomputable def petya_win_probability : ℝ := 1 - vasya_win_probability

theorem petya_has_higher_chance_of_winning 
  (h: vasya_win_probability = 1 - 0.54) : petya_win_probability > 0.5 :=
by
  have hp : petya_win_probability = 1 - vasya_win_probability := rfl
  have h_vasya : vasya_win_probability = 0.46 := h
  rw [h_vasya] at hp
  have h_petya : petya_win_probability = 1 - 0.46 := hp
  simp [h_petya]
  exact by norm_num

end petya_has_higher_chance_of_winning_l145_145203


namespace part_a_part_b_part_c_l145_145611

variables {α : Type*}
variables {A B C A' B' C' P : α}
variables [InnerProductSpace ℝ α]

-- Conditions: A', B', and C' are symmetric to P with respect to sides BC, CA, and AB of ΔABC.
def is_symmetric (P A B : α) : Prop :=
  ∃ M, M = midpoint P A ∧ M = midpoint B P

-- Hypotheses
variables 
  (hA' : is_symmetric P A B)
  (hB' : is_symmetric P B C)
  (hC' : is_symmetric P C A)

-- Propositions
theorem part_a : 
  ∃ X, X ∈ circumcircle A B' C' ∧ X ∈ circumcircle A' B C' ∧ X ∈ circumcircle A' B' C ∧ X ∈ circumcircle A B C :=
by sorry

theorem part_b :
  ∃ Q, Q ∈ circumcircle A' B' C' ∧ Q ∈ circumcircle A' B C ∧ Q ∈ circumcircle A B' C ∧ Q ∈ circumcircle A B C' :=
by sorry

variables (I J K O Q : α)
-- Centers of circumcircles of respective triangles
variables 
  (circI : circumcenter A' B C I)
  (circJ : circumcenter A B' C J)
  (circK : circumcenter A B C' K)
  (circO : circumcenter A' B' C' O)

-- Part (c): Proving the ratio QI:OI = QJ:OJ = QK:OK
theorem part_c :
  dist Q I / dist O I = dist Q J / dist O J ∧ dist Q I / dist O I = dist Q K / dist O K :=
by sorry

end part_a_part_b_part_c_l145_145611


namespace selection_schemes_count_l145_145454

theorem selection_schemes_count :
  let people := {A, B, C, D, E}
  let cities := {Paris, London, Washington, Sydney}
  ∃ (f : cities → people), function.injective f ∧ f Paris ≠ A ∧ fintype.card {f | function.injective f ∧ f Paris ≠ A} = 96 :=
sorry

end selection_schemes_count_l145_145454


namespace siblings_pizza_order_l145_145437

-- Define the fractions each sibling consumed
def Alex := 1 / 6
def Beth := 1 / 4
def Cyril := 1 / 3
def Daria := 1 / 8

-- Define the total fraction consumed except Ed
def TotalConsumed : ℚ := Alex + Beth + Cyril + Daria

-- Define the fraction left for Ed
def Ed : ℚ := 1 - TotalConsumed

-- The sequence in decreasing order for verification
def expected_sequence : List (String × ℚ) := [
  ("Cyril", Cyril),
  ("Beth", Beth),
  ("Alex", Alex),
  ("Daria", Daria),
  ("Ed", Ed)
]

-- The proof to check if the calculated sequence is correct
theorem siblings_pizza_order :
  List.sort (λ x y => Prod.snd y < Prod.snd x) [("Alex", Alex), ("Beth", Beth), ("Cyril", Cyril), ("Daria", Daria), ("Ed", Ed)] = expected_sequence :=
by
  sorry

end siblings_pizza_order_l145_145437


namespace francesca_lemonade_calories_l145_145012

def lemonJuiceWeight := 150 -- grams
def sugarWeight := 50 -- grams
def waterWeight := 400 -- grams
def honeyWeight := 100 -- grams

def lemonJuiceCaloriesPer100g := 25
def sugarCaloriesPer100g := 386
def honeyCaloriesPer100g := 300
def waterCaloriesPer100g := 0

def lemonJuiceTotalCalories := lemonJuiceWeight * lemonJuiceCaloriesPer100g / 100
def sugarTotalCalories := sugarWeight * sugarCaloriesPer100g / 100
def honeyTotalCalories := honeyWeight * honeyCaloriesPer100g / 100
def waterTotalCalories := waterWeight * waterCaloriesPer100g / 100

def totalCalories := lemonJuiceTotalCalories + sugarTotalCalories + honeyTotalCalories + waterTotalCalories
def totalWeight := lemonJuiceWeight + sugarWeight + waterWeight + honeyWeight

def caloricDensity := totalCalories / totalWeight

def caloriesIn300g := 300 * caloricDensity

theorem francesca_lemonade_calories : caloriesIn300g = 227.35 := by
  sorry

end francesca_lemonade_calories_l145_145012


namespace difference_largest_smallest_3_digit_l145_145521

theorem difference_largest_smallest_3_digit :
  let digits := [1, 2, 6, 7, 8]
  ∧ (∀ n : ℕ, ∀ d1 d2 d3 : ℕ, (d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 →
  (100 * d1 + 10 * d2 + d3 = n) →
  (876 ≥ n ∧ n ≥ 126)) )
  → 876 - 126 = 750 :=
by
  sorry

end difference_largest_smallest_3_digit_l145_145521


namespace ken_got_1750_l145_145182

theorem ken_got_1750 (K : ℝ) (h : K + 2 * K = 5250) : K = 1750 :=
sorry

end ken_got_1750_l145_145182


namespace last_score_entered_is_70_l145_145979

def scores : List ℕ := [62, 65, 70, 73, 85, 90]

noncomputable def is_integer_average (scores: List ℕ) (sums: List ℕ) : Prop :=
  ∀ (n : ℕ), n < (length sums) → n < (length scores) → (∑ i in (Finset.range n).attach, (sums.nth_le i sorry) / ↑(i+1) : ℤ) = (∑ i in (Finset.range n).attach, (sums.nth_le i sorry) / ↑(i+1) : ℚ).den
  
def total_sum (s : List ℕ) : ℕ := s.sum

def is_divisible_by_6 (n: ℕ) : Prop := n % 6 = 0

noncomputable def sum_of_scores (s : List ℕ) (n : ℕ) : ℕ :=
 if h : n < s.length then (List.take n s).sum else 0

theorem last_score_entered_is_70
  (h1 : total_sum scores = 445)
  (h2 : is_divisible_by_6 (total_sum scores))
  (h3 : is_integer_average scores (List.scanl Nat.add 0 scores)) :
  last (List.permutations scores).head = 70 :=
sorry

end last_score_entered_is_70_l145_145979


namespace eccentricity_of_ellipse_l145_145472

noncomputable def hyperbola_conditions (m n : ℝ): Prop :=
  m > 0 ∧ n > 0 ∧
  ((∃ e : ℝ, ∃ x y : ℝ, (m * x^2 - n * y^2 = 1) ∧ e = 2 ∧ (x^2/n + y^2/m - 1 ≤ 0)))

def ellipse_equation (m n : ℝ): Prop :=
  ∃ x y : ℝ, m * x^2 + n * y^2 = 1

theorem eccentricity_of_ellipse (m n : ℝ) (h : hyperbola_conditions m n) :
  ellipse_equation m n → ∃ e : ℝ, e = 2 ∧ e = (sqrt 6 / 3) :=
by
  sorry

end eccentricity_of_ellipse_l145_145472


namespace sum_of_edges_at_least_3d_l145_145622

def convex_polyhedron := sorry   -- Here, we assume the definition of a convex polyhedron. 
                                 -- This would typically include vertices and edges.

def maximum_distance (P : convex_polyhedron) : ℝ := sorry -- Definition of maximum distance between vertices in the polyhedron.

def sum_of_edge_lengths (P : convex_polyhedron) : ℝ := sorry -- Definition of the sum of the lengths of the edges of the polyhedron.

theorem sum_of_edges_at_least_3d (P : convex_polyhedron) (d : ℝ) 
  (h_max_dist : d = maximum_distance P)
  (h_convex : is_convex P) : sum_of_edge_lengths P ≥ 3 * d :=
by
  sorry

end sum_of_edges_at_least_3d_l145_145622


namespace max_value_expression_l145_145033

theorem max_value_expression (n : ℕ) (h : 2 ≤ n) (a : ℕ → ℝ) 
  (h_nonneg : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i)
  (h_sum : (∑ i in Finset.range (n + 1), a i) = 1) :
  (∑ i in Finset.range (n + 1), i * a i) *
  (∑ i in Finset.range (n + 1), a i / i) /
  (∑ i in Finset.range (n + 1), a i) ^ 2 ≤ (n + 1) ^ 2 / (4 * n) :=
by
  sorry

end max_value_expression_l145_145033


namespace cos_to_sin_shift_l145_145685

theorem cos_to_sin_shift (x : ℝ) : 
  graph_transform_right y = λx, sin(2 * x) by (π / 6) = graph_transform_right y = λx, cos(2 * x - π / 6) :=
sorry

end cos_to_sin_shift_l145_145685


namespace log_of_product_of_arithmetic_sequence_l145_145917

open Real

noncomputable def an_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a(n + 1) = a(n) + d

theorem log_of_product_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : an_arithmetic_sequence a) (h_sum : a 5 + a 6 = 4) :
  log 2 (∏ i in (Finset.range 10).image (λ i, 2^(a (i+1))) ) = 20 :=
by
  sorry

end log_of_product_of_arithmetic_sequence_l145_145917


namespace distance_from_origin_l145_145912

-- Lean definition and statement of the problem
theorem distance_from_origin (x y z : ℝ) : 
  x = 8 ∧ y = -15 ∧ z = 17 → real.sqrt (x^2 + y^2 + z^2) = real.sqrt 578 := 
by
  assume h: x = 8 ∧ y = -15 ∧ z = 17
  sorry

end distance_from_origin_l145_145912


namespace triangle_inequality_shortest_side_l145_145224

theorem triangle_inequality_shortest_side (a b c : ℝ) (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c ≤ a ∧ c ≤ b :=
sorry

end triangle_inequality_shortest_side_l145_145224


namespace quadratic_discriminant_positive_find_m_l145_145499

-- Define the quadratic equation and its properties
def quadratic (x m : Int) : Int := x^2 - (2 * m + 1) * x + (m^2 + m)

-- Part (1)
theorem quadratic_discriminant_positive (m : Int) : 
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  Δ > 0 := by
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  have h : Δ = 1 := by sorry
  show 1 > 0 from by sorry

-- Part (2)
theorem find_m (a b m : Int) : 
  quadratic a m = 0 ∧ quadratic b m = 0 ∧ (2 * a + b) * (a + 2 * b) = 20 → 
  m = -2 ∨ m = 1 := by
  intro h
  sorry

end quadratic_discriminant_positive_find_m_l145_145499


namespace valid_condition_C_l145_145360

-- Definitions of the conditions
def condition_A (a b : ℝ) : Prop := real.sqrt (a ^ 2) + real.sqrt (b ^ 2) = a + b
def condition_B (a b : ℝ) : Prop := real.sqrt (a * b) = real.sqrt (a) * real.sqrt (b)
def condition_C (a b : ℝ) : Prop := real.sqrt (-a ^ 2 * b ^ 2) = -a * b
def condition_D (a : ℝ) : Prop := real.sqrt (-1 * a) = - real.sqrt (-a)

-- The problem statement
theorem valid_condition_C (a b : ℝ) (h : a * b = 0) : condition_C a b :=
sorry

end valid_condition_C_l145_145360


namespace triangle_angle_iff_orthic_triangle_right_angled_l145_145222

noncomputable def is_orthic_triangle_right_angled (A B C : Point) : Prop :=
  ∃ H D E F : Point,
    is_orthocenter (triangle A B C) H ∧
    is_perpendicular_from H A D ∧
    is_perpendicular_from H B E ∧
    is_perpendicular_from H C F ∧
    is_right_angled_triangle H D E F

theorem triangle_angle_iff_orthic_triangle_right_angled (A B C : Point) :
  (∠A = 45° ∨ ∠A = 135°) ↔ is_orthic_triangle_right_angled A B C := 
by sorry

end triangle_angle_iff_orthic_triangle_right_angled_l145_145222


namespace f_value_at_25pi_over_6_smallest_positive_period_of_f_max_value_in_interval_min_value_in_interval_l145_145481

def f (x : ℝ) : ℝ := - (Real.sqrt 3) * (Real.sin x) ^ 2 + (Real.sin x) * (Real.cos x)

theorem f_value_at_25pi_over_6 : f (25 * Real.pi / 6) = 0 := 
sorry

theorem smallest_positive_period_of_f : ∀ x, f (x + Real.pi) = f x :=
sorry

theorem max_value_in_interval : ∃ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x = (2 - Real.sqrt 3) / 2 :=
sorry

theorem min_value_in_interval : ∃ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 :=
sorry

end f_value_at_25pi_over_6_smallest_positive_period_of_f_max_value_in_interval_min_value_in_interval_l145_145481


namespace find_x_y_l145_145016

theorem find_x_y (x y : ℝ) (h : (2 * x - 1) + 1 * Complex.i = y - (2 - y) * Complex.i) : x = 2 ∧ y = 3 :=
by
  sorry

end find_x_y_l145_145016


namespace total_valid_arrangements_l145_145255

-- Define the subjects and arrange them according to the given conditions
def subjects := ["Chinese", "Mathematics", "English", "Science Comprehensive"]

-- Define the constraints of the problem
def valid_arrangement (arrangement : List String) : Bool :=
  arrangement.length = 3 ∧
  ∀ i, (arrangement.nth i != some "Mathematics" ∨ arrangement.nth i != some "Science Comprehensive")

-- Prove the total number of different valid arrangements is 24
theorem total_valid_arrangements : ∃ n, n = 24 ∧
  (finset.univ.filter valid_arrangement).card = n := 
sorry

end total_valid_arrangements_l145_145255


namespace problem_statement_l145_145543

-- Define the points A, B, C, D, and E
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (12, 0)
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the slope of the line through points C and D
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept of the line through points C and D
def y_intercept (p1 p2 : ℝ × ℝ) : ℝ :=
  p2.2 - slope p1 p2 * p2.1

-- The sum of the slope and the y-intercept
def sum_slope_intercept (p1 p2 : ℝ × ℝ) : ℝ :=
  slope p1 p2 + y_intercept p1 p2

-- Given conditions that D is the midpoint of AB,
-- and the expected result for the combination of slope and y-intercept
theorem problem_statement : 
  D = (0, 5) ∧ 
  E = (6, 0) ∧
  sum_slope_intercept C D = 55 / 12 :=
by
  -- proof is omitted for brevity
  sorry

end problem_statement_l145_145543


namespace positive_real_solution_unique_l145_145876

theorem positive_real_solution_unique :
  (∃! x : ℝ, 0 < x ∧ x^12 + 5 * x^11 - 3 * x^10 + 2000 * x^9 - 1500 * x^8 = 0) :=
sorry

end positive_real_solution_unique_l145_145876


namespace nature_of_roots_of_quadratic_l145_145440

theorem nature_of_roots_of_quadratic (k : ℝ) (h1 : k > 0) (h2 : 3 * k^2 - 2 = 10) :
  let a := 1
  let b := -(4 * k - 3)
  let c := 3 * k^2 - 2
  let Δ := b^2 - 4 * a * c
  Δ < 0 :=
by
  sorry

end nature_of_roots_of_quadratic_l145_145440


namespace max_min_S_l145_145236

theorem max_min_S (x y : ℝ) (h : (x - 1)^2 + (y + 2)^2 = 4) : 
  (∃ S_max S_min : ℝ, S_max = 4 + 2 * Real.sqrt 5 ∧ S_min = 4 - 2 * Real.sqrt 5 ∧ 
  (∀ S : ℝ, (∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 4 ∧ S = 2 * x + y) → S ≤ S_max ∧ S ≥ S_min)) :=
sorry

end max_min_S_l145_145236


namespace number_of_cars_l145_145113

variable (C B : ℕ)

-- Define the conditions
def number_of_bikes : Prop := B = 2
def total_number_of_wheels : Prop := 4 * C + 2 * B = 44

-- State the theorem
theorem number_of_cars (hB : number_of_bikes B) (hW : total_number_of_wheels C B) : C = 10 := 
by 
  sorry

end number_of_cars_l145_145113


namespace flat_rate_65_l145_145733

noncomputable def flat_rate_first_night (f n : ℝ) : Prop := 
  (f + 4 * n = 245) ∧ (f + 9 * n = 470)

theorem flat_rate_65 :
  ∃ (f n : ℝ), flat_rate_first_night f n ∧ f = 65 := 
by
  sorry

end flat_rate_65_l145_145733


namespace money_left_for_fun_l145_145973

theorem money_left_for_fun (winnings : ℕ) (tax_rate : ℝ) (mortgage_fraction : ℝ) 
  (retirement_fraction : ℝ) (college_fraction : ℝ) (savings_amount : ℕ) :
  winnings = 20000 →
  tax_rate = 0.55 →
  mortgage_fraction = 1/2 →
  retirement_fraction = 1/3 →
  college_fraction = 1/4 →
  savings_amount = 1200 →
  let amount_after_taxes := winnings - (tax_rate * winnings).to_nat in
  let amount_after_mortgage := amount_after_taxes - (mortgage_fraction * amount_after_taxes).to_nat in
  let amount_after_retirement := amount_after_mortgage - (retirement_fraction * amount_after_mortgage).to_nat in
  let amount_after_college := amount_after_retirement - (college_fraction * amount_after_retirement).to_nat in
  let amount_after_savings := amount_after_college - savings_amount in 
  amount_after_savings = 1050 :=
by sorry

end money_left_for_fun_l145_145973


namespace johns_eggs_per_week_l145_145572

noncomputable def total_emus (pens : ℕ) (emus_per_pen : ℕ) : ℕ :=
  pens * emus_per_pen

noncomputable def female_emus (total : ℕ) : ℕ :=
  total / 2

noncomputable def eggs_per_week (females : ℕ) (days_per_week : ℕ) : ℕ :=
  females * days_per_week

theorem johns_eggs_per_week :
  let pens := 4 in
  let emus_per_pen := 6 in
  let days_per_week := 7 in
  let total := total_emus pens emus_per_pen in
  let females := female_emus total in
  eggs_per_week females days_per_week = 84 :=
by
  sorry

end johns_eggs_per_week_l145_145572


namespace perimeter_of_equilateral_triangle_l145_145737

theorem perimeter_of_equilateral_triangle : 
  ∃ (L : ℝ → ℝ), L 0 = 0 ∧ (∃ x1, L x1 = (1 + real.sqrt 3 / 3 * x1)) ∧ 
  (∃ y1, y1 = 1 + real.sqrt 3 / 3) ∧ 
  (∃ y2, y2 = -real.sqrt 3 / 3) ∧ 
  (∀ x, L x = -real.sqrt 3 / 3 * x) ∧ 
  (∀ x1 x2 y1 y2, x1 = x2 ∧ x1 = 1 ∧ 
  L x1 = y2 ∧ y1 = y2 + 1 + real.sqrt 3 / 3 ∧ 
  3 * (1 + 2 * real.sqrt 3 / 3) = 3 + 2 * real.sqrt 3) :=
begin
  sorry
end

end perimeter_of_equilateral_triangle_l145_145737


namespace derivative_at_pi_over_six_l145_145852

-- Define the function f(x) = cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem: the derivative of f at π/6 is -1/2
theorem derivative_at_pi_over_six : deriv f (Real.pi / 6) = -1 / 2 :=
by sorry

end derivative_at_pi_over_six_l145_145852


namespace maria_total_stamps_l145_145975

variable (currentStamps : ℕ) (percentageIncrease : ℕ) (additionalStamps : ℕ) (totalStamps : ℕ)

axiom h1 : currentStamps = 40
axiom h2 : percentageIncrease = 20
axiom h3 : additionalStamps = (percentageIncrease * currentStamps) / 100
axiom h4 : totalStamps = currentStamps + additionalStamps

theorem maria_total_stamps :
  totalStamps = 48 :=
by
  rw [h4, h3, h2, h1]
  sorry

end maria_total_stamps_l145_145975


namespace trajectory_of_M_l145_145328

-- Define the circle equation
def circle_eq (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in 
  x^2 + y^2 = 1

-- Define the condition that point M is on the line PD such that PM = 1/3 PD
def point_on_PD (M P : ℝ × ℝ) : Prop :=
  let (mx, my) := M in
  let (px, py) := P in
  mx = px ∧ my = (2/3) * py

-- Define the trajectory equation for point M
def trajectory_eq (M : ℝ × ℝ) : Prop := 
  let (x, y) := M in 
  x^2 + (9/4) * y^2 = 1

theorem trajectory_of_M :
  ∀ (P M : ℝ × ℝ), circle_eq P → point_on_PD M P → trajectory_eq M :=
by
  intros P M h1 h2
  sorry

end trajectory_of_M_l145_145328


namespace part_one_part_two_l145_145504

open Real

variable {a b m : ℝ}

theorem part_one (m : ℝ) : 
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  in Δ > 0 := 
by
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  show Δ > 0 from by {
    simp[Δ],
    linarith
  }

theorem part_two (a b m : ℝ) (h1 : a + b = 2 * m + 1) (h2 : a * b = m^2 + m) (h3 : (2*a + b) * (a + 2*b) = 20) : 
  m = -2 ∨ m = 1 :=
by
  have h4: 9*m^2 + 9*m + 2 = 20 := by {
    rw [h1, h2, ← add_assoc, ← add_assoc],
    linarith
  }
  have h5 := solve_quadratic 9 9 (-18) h4
  cases h5 with h5_left h5_right
  exact or.inl h5_left
  exact or.inr h5_right

end part_one_part_two_l145_145504


namespace algebraic_expression_is_200_l145_145700

-- Define the condition
def satisfies_ratio (x : ℕ) : Prop :=
  x / 10 = 20

-- The proof problem statement
theorem algebraic_expression_is_200 : ∃ x : ℕ, satisfies_ratio x ∧ x = 200 :=
by
  -- Providing the necessary proof infrastructure
  use 200
  -- Assuming the proof is correct
  sorry


end algebraic_expression_is_200_l145_145700


namespace value_of_expression_l145_145672

variable (a b : ℝ)

-- Conditions required by the problem
axiom h1 : a > b
axiom h2 : 3^(2 * a) - 10 * 3^(a + 1) + 81 = 0
axiom h3 : 3^(2 * b) - 10 * 3^(b + 1) + 81 = 0

-- The main theorem asserting the desired outcome
theorem value_of_expression (a b : ℝ) (h1 : a > b) 
    (h2 : 3^(2 * a) - 10 * 3^(a + 1) + 81 = 0)
    (h3 : 3^(2 * b) - 10 * 3^(b + 1) + 81 = 0) 
    : 20 * a^2 + 18 * b^2 = 198 := 
sorry

end value_of_expression_l145_145672


namespace reflection_y_axis_l145_145408

open Matrix

def reflection_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

theorem reflection_y_axis (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M ⬝ ![![1, 0], ![0, 1]]) = reflection_y_axis_matrix :=
by sorry

end reflection_y_axis_l145_145408


namespace non_gray_squares_count_l145_145731

-- Define the dimensions of the grid strip
def width : ℕ := 5
def length : ℕ := 250

-- Define the repeating pattern dimensions and color distribution
def pattern_columns : ℕ := 4
def pattern_non_gray_squares : ℕ := 13
def pattern_total_squares : ℕ := width * pattern_columns

-- Define the number of complete patterns in the grid strip
def complete_patterns : ℕ := length / pattern_columns

-- Define the number of additional columns and additional non-gray squares
def additional_columns : ℕ := length % pattern_columns
def additional_non_gray_squares : ℕ := 6

-- Calculate the total non-gray squares
def total_non_gray_squares : ℕ := complete_patterns * pattern_non_gray_squares + additional_non_gray_squares

theorem non_gray_squares_count : total_non_gray_squares = 812 := by
  sorry

end non_gray_squares_count_l145_145731


namespace parallel_lines_sufficient_not_necessary_l145_145240

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  (∀ x y : ℝ, a = 3 → ax + 2*y + 1 = 0 → 3*x + (a-1)*y - 2 = 0) →
  (∀ x y : ℝ, (ax + 2*y + 1 = 0 ∧ 3*x + (a-1)*y - 2 = 0) ↔ a = 3 ∨ a = -2) →
  "sufficient but not necessary" :=
sorry

end parallel_lines_sufficient_not_necessary_l145_145240


namespace find_values_of_x_and_y_l145_145174

theorem find_values_of_x_and_y (x y : ℝ) :
  (2.5 * x = y^2 + 43) ∧ (2.1 * x = y^2 - 12) → (x = 137.5 ∧ y = Real.sqrt 300.75) :=
by
  sorry

end find_values_of_x_and_y_l145_145174


namespace zero_in_interval_l145_145385

def f (x : ℝ) : ℝ := (0.5) ^ x - x

theorem zero_in_interval : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
begin
  sorry
end

end zero_in_interval_l145_145385


namespace z_squared_minus_w_squared_l145_145148

noncomputable def z : ℝ := 3000 ^ 1501 - 3000 ^ (-1501)
noncomputable def w : ℝ := 3000 ^ 1501 + 3000 ^ (-1501)

theorem z_squared_minus_w_squared :
  z ^ 2 - w ^ 2 = -4 :=
by
  sorry

end z_squared_minus_w_squared_l145_145148


namespace problem_result_l145_145296

-- Define the components of the problem
def term1 := 3^2 * 20.4
def term2 := 5100 / 102
def term3 := nat.factorial 4

-- Combine terms into the final expression:
def expression := term1 - term2 + term3

-- Provide the theorem stating the result
theorem problem_result : expression = 157.6 :=
  by sorry

end problem_result_l145_145296


namespace petya_higher_chance_of_winning_l145_145208

theorem petya_higher_chance_of_winning 
  {total_candies : ℕ} (h_total : total_candies = 25)
  {P_two_caramel : ℝ} (h_P_two_caramel : P_two_caramel = 0.54) :
  let P_two_chocolate := 1 - P_two_caramel in
  P_two_chocolate < 0.5 :=
by {
  sorry
}

end petya_higher_chance_of_winning_l145_145208


namespace problem_statement_l145_145094

noncomputable def a : ℕ := by
  -- The smallest positive two-digit multiple of 3
  let a := Finset.range 100 \ Finset.range 10
  let multiples := a.filter (λ n => n % 3 = 0)
  exact multiples.min' ⟨12, sorry⟩

noncomputable def b : ℕ := by
  -- The smallest positive three-digit multiple of 4
  let b := Finset.range 1000 \ Finset.range 100
  let multiples := b.filter (λ n => n % 4 = 0)
  exact multiples.min' ⟨100, sorry⟩

theorem problem_statement : a + b = 112 := by
  sorry

end problem_statement_l145_145094


namespace part1_intervals_and_extreme_part2_min_value_l145_145850

-- Define the functions and conditions
def f (x : ℝ) (a : ℝ) := log x - a * x

-- Statement for Part (1)
theorem part1_intervals_and_extreme
  (x : ℝ) (h₀ : x > 0) (h₁ : x ≠ 2) :
  let a := 1 / 2 in
  (0 < x ∧ x < 2 → deriv (f x a) > 0) ∧
  (x > 2 → deriv (f x a) < 0) ∧
  f 2 a = log 2 - 1 :=
sorry

-- Statement for Part (2)
theorem part2_min_value
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (1 ≤ a⁻¹ → ∀ x, (1 ≤ x ∧ x ≤ 2) → f x a ≥ -a) ∧
  (a⁻¹ < 2 → a < log 2 → ∀ x, (1 ≤ x ∧ x ≤ 2) → f x a ≥ -a) ∧
  (a⁻¹ < 2 → log 2 ≤ a → ∀ x, (1 ≤ x ∧ x ≤ 2) → f x a ≥ log 2 - 2 * a) :=
sorry

end part1_intervals_and_extreme_part2_min_value_l145_145850


namespace only_triple_l145_145399

theorem only_triple (a b c : ℕ) (h1 : (a * b + 1) % c = 0)
                                (h2 : (a * c + 1) % b = 0)
                                (h3 : (b * c + 1) % a = 0) :
    (a = 1 ∧ b = 1 ∧ c = 1) :=
by
  sorry

end only_triple_l145_145399


namespace parallel_vectors_perpendicular_vectors_l145_145058

variables (a b : ℝ^3)
variables (k : ℝ)
variables (c d : ℝ^3)

-- Conditions
def norm_a : ℝ := 2
def norm_b : ℝ := 3
def cos_angle_ab : ℝ := 1/2

-- Vectors
def vec_c : ℝ^3 := 5 • a + 3 • b
def vec_d : ℝ^3 := 3 • a + k • b

-- Given dot product of a and b
def dot_ab : ℝ := norm_a * norm_b * cos_angle_ab

-- Prove k = 9/5 when c is parallel to d
theorem parallel_vectors (h_parallel : vec_c = (5/3 : ℝ) • vec_d) : 
  k = 9/5 :=
sorry

-- Prove k = -29/14 when c is perpendicular to d
theorem perpendicular_vectors (h_perpendicular : inner (5 • a + 3 • b) (3 • a + k • b) = 0) :
  k = -29/14 :=
sorry

end parallel_vectors_perpendicular_vectors_l145_145058


namespace acute_triangle_inequality_l145_145762

theorem acute_triangle_inequality 
  (A B C H P Q X Y : Type) 
  [AcuteTriangle A B C] 
  [H = orthocenter A B C] 
  (AC_ne_BC : AC ≠ BC) 
  (alt_A_meets : meets_alt A H) 
  (alt_B_meets : meets_alt B H) 
  (P_Q_eq : PX = QY) 
  : AP + BQ ≥ 2 * CH :=
sorry

end acute_triangle_inequality_l145_145762


namespace alice_profit_l145_145357

def total_bracelets : ℕ := 52
def cost_materials : ℝ := 3.0
def bracelets_given_away : ℕ := 8
def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_sold := total_bracelets - bracelets_given_away in
  let total_revenue := bracelets_sold * price_per_bracelet in
  let profit := total_revenue - cost_materials in
  profit = 8.00 :=
by
  sorry

end alice_profit_l145_145357


namespace f_2019_eq_neg_2_l145_145055

/-- Define the function f with the given properties -/
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry

/-- f is odd -/
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

/-- f is periodic with period 4 -/
axiom f_periodic : ∀ x : ℝ, f (x + 4) = f x

/-- Given facts, prove that f(2019) = -2 -/
theorem f_2019_eq_neg_2 : f 2019 = -2 :=
sorry

end f_2019_eq_neg_2_l145_145055


namespace letter_arrangements_count_l145_145515

theorem letter_arrangements_count :
  (∑ k in Finset.range 6, (Nat.choose 5 k)^2 * (Nat.choose 5 (5 - k))) =
  (∑ k in Finset.range 6, Nat.mul (Nat.mul (Nat.choose 5 k) (Nat.choose 5 k)) (Nat.choose 5 (5 - k))) := by
sorry

end letter_arrangements_count_l145_145515


namespace product_geq_n_minus_one_pow_n_l145_145964

open Real

theorem product_geq_n_minus_one_pow_n
  (n : ℕ)
  (x : Fin n → ℝ)
  (hx : ∀ i, 0 < x i)
  (h : ∑ i, 1 / (1 + x i) = 1) :
  (∏ i, x i) ≥ (n - 1) ^ n :=
by
  -- proof goes here
  sorry

end product_geq_n_minus_one_pow_n_l145_145964


namespace batsman_average_after_21st_inning_l145_145313

theorem batsman_average_after_21st_inning (A : ℝ) :
  let total_runs := 20 * A,
      new_total_runs := total_runs + 89,
      new_average := (new_total_runs) / 21,
      increased_average := A + 2
  in (20 * A + 89) = 21 * (A + 2) → new_average = 49 := 
by
  sorry

end batsman_average_after_21st_inning_l145_145313


namespace false_propositions_count_l145_145654

theorem false_propositions_count :
  let cond1 := ¬ (∃ x : ℝ, x = 0 ∧ ∀ ε > 0, ∃ δ > 0, |x| < δ → |x| = 0) -- False, as explained in the solution
  let cond2 := ∀ (a b c : ℝ), (b^2 - 3 * a * c > 0 ↔ ∃ x : ℝ, 3 * a * x^2 + 2 * b * x + c = 0) -- True
  let cond3 := ∀ (m n : ℝ), (∀ x : ℝ, x ∈ set.Ioo (-(4:ℝ)) 4 → 3 * x^2 - 48 < 0 ∧ m = 1 ∧ n = 0) -- True
  (↑1 = cond1 + cond2 + cond3) = 1 :=
by
  sorry

end false_propositions_count_l145_145654


namespace largest_coefficient_in_binomial_expansion_l145_145675

theorem largest_coefficient_in_binomial_expansion :
  let a := 2 in
  let b := 3 * x in
  let n := 10 in
  let T (k : ℕ) := Nat.choose n k * a^(n-k) * (b^k) in
  T 5 = (6^5) * (Nat.choose 10 5) * (x^5) :=
sorry

end largest_coefficient_in_binomial_expansion_l145_145675


namespace ammonium_hydroxide_amount_l145_145428

-- Defining the entities involved
variables (NH4Cl NaOH NH4OH NaCl : Type)

-- Defining stoichiometry function which verifies the reaction
def Stoichiometry (reactants products : Type) : Prop := 
  ∃ NH4Cl, ∃ NaOH, (NH4Cl + NaOH) = (NH4OH + NaCl)

-- Defining the amount function to calculate moles
def Amount (compound : Type) : Nat := 1

-- The main theorem statement
theorem ammonium_hydroxide_amount :
  (Stoichiometry (NH4Cl + NaOH) (NH4OH + NaCl)) → (Amount NH4OH = 1) :=
sorry

end ammonium_hydroxide_amount_l145_145428


namespace probability_ant_ends_at_C_l145_145763

section AntProbability
variable (A C : ℕ × ℕ)
variable (red blue : ℕ × ℕ → Prop)
variable (moves_to : (ℕ × ℕ) → (ℕ × ℕ) → Prop)
variable [DecidablePred red]
variable [DecidablePred blue]

-- Initial conditions
def ant_starts_at_A : red A := sorry
def ant_moves_to_neighbor (x y : ℕ × ℕ) : moves_to x y → 
  (|x.1 - y.1| + |x.2 - y.2| = 1) := sorry

def ant_on_blue_after_odd_moves (x : ℕ × ℕ) (n : ℕ) : Prop := 
  ∃ (m : ℕ), n = 2 * m + 1 ∧ red x → blue x := sorry

def ant_7_moves_ends_at_blue (x : ℕ × ℕ) : Prop := 
  ant_on_blue_after_odd_moves x 7 → blue x := sorry
  
-- Conditions and desired conclusion
def C_coordinates := (0, 2)

theorem probability_ant_ends_at_C :
  ant_starts_at_A A ∧
  ant_7_moves_ends_at_blue C →
  (∃ (k : ℕ), k ≠ 0 ∧ is_prob (C k) = 1 / 4) :=
sorry
end AntProbability

end probability_ant_ends_at_C_l145_145763


namespace calc_edges_of_R_l145_145727

open Classical

noncomputable def edges_of_new_polyhedron (n : ℕ) (e : ℕ) (intersections : ℕ) : ℕ :=
  4 * e

theorem calc_edges_of_R (n : ℕ) (e : ℕ) (intersections : ℕ) (h1 : e = 150) (h2 : intersections = 3)
  : edges_of_new_polyhedron n e intersections = 600 := by
  dsimp [edges_of_new_polyhedron]
  rw [h1]
  exact rfl

#check calc_edges_of_R

end calc_edges_of_R_l145_145727


namespace area_of_polygon_intersection_l145_145289

theorem area_of_polygon_intersection :
  (let points := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16 ∧ (p.1 - 6)^2 + 4*p.2^2 = 36},
       vertices := Finset.mk (points.to_list.filter (λ p, ∀ v ∈ points.to_list, p ≠ v))
   in 1/2 * |(Finset.univ.sum (λ ⟨i, j⟩, i.1 * j.2 - j.1 * i.2))) = 12 :=
by
  sorry

end area_of_polygon_intersection_l145_145289


namespace total_school_population_l145_145312

theorem total_school_population (x B : ℕ) 
  (h1 : 162 = (x / 100) * B) 
  (h2 : B = (0.5 * x).toNat) : 
  x = 180 := 
by 
  sorry

end total_school_population_l145_145312


namespace general_term_arithmetic_sequence_sum_first_n_terms_l145_145052

noncomputable def a_n (n : ℕ) : ℤ :=
  3 * n - 1

def b_n (n : ℕ) (b : ℕ → ℚ) : Prop :=
  (b 1 = 1) ∧ (b 2 = 1 / 3) ∧ ∀ n : ℕ, a_n n * b (n + 1) = n * b n

def sum_b_n (n : ℕ) (b : ℕ → ℚ) : ℚ :=
  (3 / 2) - (1 / (2 * (3 ^ (n - 1))))

theorem general_term_arithmetic_sequence (n : ℕ) :
  a_n n = 3 * n - 1 := by sorry

theorem sum_first_n_terms (n : ℕ) (b : ℕ → ℚ) (h : b_n n b) :
  sum_b_n n b = (3 / 2) - (1 / (2 * (3 ^ (n - 1)))) := by sorry

end general_term_arithmetic_sequence_sum_first_n_terms_l145_145052


namespace log_prime_values_l145_145103

theorem log_prime_values (p q : ℕ) (hp : p.prime) (hq : q.prime) (h_eq : 3 * p + 5 * q = 31) :
  log 2 (p / (3 * q + 1) : ℝ) = -3 ∨ log 2 (p / (3 * q + 1) : ℝ) = 0 :=
sorry

end log_prime_values_l145_145103


namespace max_value_expression_l145_145427

theorem max_value_expression :
  ∃ (x y : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧
  (sqrt (9 - sqrt 7) * sin x - sqrt (2 * (1 + cos (2 * x))) - 1) *
   (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) = 19 :=
by
  sorry

end max_value_expression_l145_145427


namespace find_x_l145_145704

-- Definitions for the median and mean calculations
def mean (a b c d e : ℝ) : ℝ :=
  (a + b + c + d + e) / 5

def median (a b c d e : ℝ) : ℝ :=
  let sorted_list := List.sort [a, b, c, d, e]
  sorted_list.nthLe 2 sorry -- second index is the median in zero-indexed list

theorem find_x :
  ∃ x : ℝ, median 3 7 12 21 x = 9 + (mean 3 7 12 21 x) ^ (1/4) ∧ x = 362 :=
sorry

end find_x_l145_145704


namespace mean_median_mode_comparison_l145_145626

noncomputable def fish : List ℕ := [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

/- 
  Helper definitions: mean, median, and mode.
-/
def mean (l : List ℕ) : ℚ :=
  l.sum / l.length

def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (λ a b => a ≤ b)
  if l.length % 2 = 0
  then (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2
  else sorted.get! (l.length / 2)

def mode (l : List ℕ) : ℕ :=
  let freq := List.foldr (λ n m => m.insertWith (+) n 1) (RBMap.empty ℕ ℕ compare) l
  freq.foldl (λ acc k v => if v > acc.2 then (k, v) else acc) (0, 0) |>.1

theorem mean_median_mode_comparison :
  let μ := mean fish
  let med := median fish
  let mod := mode fish
  μ < med ∧ med < mod := by
    let μ : ℚ := 1.75
    let med : ℚ := 2
    let mod : ℕ := 3
    sorry

end mean_median_mode_comparison_l145_145626


namespace lines_perpendicular_to_same_plane_are_parallel_l145_145282

-- Let l1 and l2 be lines and π be a plane.
variable (l1 l2 : Type) (π : Type)

-- Assume there are two lines l1 and l2.
variable [linear_algebra_space l1] [linear_algebra_space l2]

-- Assume both lines are perpendicular to the same plane π.
variable (H1 : perpendicular l1 π)
variable (H2 : perpendicular l2 π)

-- Prove that l1 is parallel to l2.
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel l1 l2 :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l145_145282


namespace distance_between_trees_l145_145350

/-
A yard is in the shape of a trapezoid with an upper base of 160 meters, 
a lower base of 200 meters, and a height of 180 meters. 
If 11 trees are planted at equal distances along the perimeter, one tree being at each end of the yard, 
prove that the distance between two consecutive trees is approximately 72.22 meters.
-/

noncomputable def trapezoid_perimeter (upper_base : ℝ) (lower_base : ℝ) (height : ℝ) : ℝ :=
  let base_diff := lower_base - upper_base
  let half_base_diff := base_diff / 2
  let leg := Real.sqrt (height^2 + half_base_diff^2)
  in upper_base + lower_base + 2 * leg

noncomputable def tree_distance (upper_base lower_base height : ℝ) (trees : ℕ) : ℝ :=
  (trapezoid_perimeter upper_base lower_base height) / (trees - 1)

theorem distance_between_trees :
  tree_distance 160 200 180 11 ≈ 72.22 := sorry

end distance_between_trees_l145_145350


namespace sandwich_total_calories_l145_145368

-- Given conditions
def bacon_calories := 2 * 125
def bacon_percentage := 20 / 100

-- Statement to prove
theorem sandwich_total_calories :
  bacon_calories / bacon_percentage = 1250 := 
sorry

end sandwich_total_calories_l145_145368


namespace probability_is_two_thirds_l145_145538

-- Definitions for the problem

def set := {3, 4, 6, 8 : ℕ}

def is_multiple_of_8 (n : ℕ) : Prop := 8 ∣ n

def chosen_pairs := {(3,4), (3,6), (3,8), (4,6), (4,8), (6,8) : ℕ×ℕ}

def favorable_pairs : Finset (ℕ × ℕ) := 
  {(3,8), (4,6), (4,8), (6,8)}

def probability_of_favorable_event (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Statement of the problem in Lean 4
theorem probability_is_two_thirds :
  probability_of_favorable_event favorable_pairs.card chosen_pairs.card = 2 / 3 := 
sorry

end probability_is_two_thirds_l145_145538


namespace piastres_in_6th_purse_l145_145933

-- We define the amounts and constraints
variables (x : ℕ) -- Number of piastres in the first purse
variables (piastres : ℕ) -- Total number of piastres
variables (purse6 : ℕ) -- Number of piastres in the 6th purse

-- Given conditions as variables
axiom total_piastres : 150 = piastres
axiom num_purses : 10
axiom increasing_sequence : ∀ i j : ℕ, i < j → (x + i) < (x + j)
axiom first_last_condition : x ≥ (x + 9) / 2

-- Prove the number of piastres in the 6th purse
theorem piastres_in_6th_purse : purse6 = 16 :=
by
  -- placeholder for proof
  sorry

end piastres_in_6th_purse_l145_145933


namespace inequality_for_sequences_l145_145865

noncomputable def a : ℕ → ℝ
| 0       := (Real.sqrt 2) / 2
| (n + 1) := (Real.sqrt 2) / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
| 0       := 1
| (n + 1) := (Real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem inequality_for_sequences (n : ℕ) : 
  2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n :=
sorry

end inequality_for_sequences_l145_145865


namespace simplify_expression_l145_145635

theorem simplify_expression (x : ℝ) (hx : x > 0) :
  ( (x - 1) / (x^(3/4) + x^(1/2)) * (x^(1/2) + x^(1/4)) / (x^(1/2) + 1) * x^(1/4) + 1 ) = sqrt x :=
by
  sorry

end simplify_expression_l145_145635


namespace max_red_points_800_l145_145452

def maxRedPoints (n : ℕ) : ℕ :=
  if n ≠ 0 then
    let s := {k : ℕ | ∃ m, m < n ∧ (odd m ∧ (2^k * m) % n = k % n)}
    s.card
  else 0

theorem max_red_points_800 :
  maxRedPoints 800 = 25 := by
  sorry

end max_red_points_800_l145_145452


namespace jenny_prints_7_copies_l145_145562

theorem jenny_prints_7_copies :
  let cost_per_page := 0.10
  let pages_per_essay := 25
  let cost_per_pen := 1.50
  let num_pens := 7
  let total_money := 2 * 20
  let change := 12
  let total_cost_pens := num_pens * cost_per_pen
  let total_spent := total_money - change
  let cost_per_essay := pages_per_essay * cost_per_page
  let money_spent_on_printing := total_spent - total_cost_pens
  let num_copies := money_spent_on_printing / cost_per_essay
  num_copies = 7 := 
by {
  -- Introduce the variables
  let cost_per_page := 0.10
  let pages_per_essay := 25
  let cost_per_pen := 1.50
  let num_pens := 7
  let total_money := 2 * 20
  let change := 12
  let total_cost_pens := num_pens * cost_per_pen
  let total_spent := total_money - change
  let cost_per_essay := pages_per_essay * cost_per_page
  let money_spent_on_printing := total_spent - total_cost_pens
  have h1 : money_spent_on_printing = 28 - 10.50, by ring_nf,
  have h2 : cost_per_essay = 25 * 0.10, by ring_nf,
  have h3 : money_spent_on_printing = 17.50, from h1,
  have h4 : cost_per_essay = 2.50, from h2,
  have h5 : num_copies = 17.50 / 2.50, by ring_nf,
  show num_copies = 7, from h5
} sorry

end jenny_prints_7_copies_l145_145562


namespace probability_four_blue_l145_145227

theorem probability_four_blue:
  let p_blue := (10 / 15:ℚ) in
  let p_red := (5 / 15:ℚ) in
  let exact_scenario := ((10.choose 4) * (p_blue ^ 4) * (p_red ^ 6)) in
  exact_scenario = (210 * 16 / 59049) :=
by 
  let p_blue : ℚ := 10 / 15
  let p_red : ℚ := 5 / 15
  let exact_scenario := ((10.choose 4) * (p_blue ^ 4) * (p_red ^ 6)) 
  have : exact_scenario = (210 * 16 / 59049), by sorry
  exact ‹exact_scenario = (210 * 16 / 59049)›

end probability_four_blue_l145_145227


namespace find_A_cos_identity_range_perimeter_acute_l145_145121

theorem find_A_cos_identity (A B C a b c : ℝ) 
  (h1 : b = 2)
  (h2 : ∀ A B C : ℝ, is_triangle real.triangle_ABC A B C)
  (h3 : ∀ α β γ : ℝ, cos (α - β) = sqrt 3 * sin β - cos γ) :
  A = π / 3 ∨ A = 2 * π / 3 :=
sorry

theorem range_perimeter_acute (A B C a b c : ℝ) 
  (h1 : b = 2)
  (h2 : A = π / 3)
  (h3 : ∀ A B C : ℝ, is_triangle real.triangle_ABC A B C) 
  (h4 : triangle_acute real.triangle_ABC) :
  3 + sqrt 3 < a + b + c ∧ a + b + c < 6 + 2 * sqrt 3 :=
sorry

end find_A_cos_identity_range_perimeter_acute_l145_145121


namespace question_1_solution_question_2_solution_l145_145851

def f (m x : ℝ) := m*x^2 - (m^2 + 1)*x + m

theorem question_1_solution (x : ℝ) :
  (f 2 x ≤ 0) ↔ (1 / 2 ≤ x ∧ x ≤ 2) :=
sorry

theorem question_2_solution (x m : ℝ) :
  (m > 0) → 
  ((0 < m ∧ m < 1 → f m x > 0 ↔ x < m ∨ x > 1 / m) ∧
  (m = 1 → f m x > 0 ↔ x ≠ 1) ∧
  (m > 1 → f m x > 0 ↔ x < 1 / m ∨ x > m)) :=
sorry

end question_1_solution_question_2_solution_l145_145851


namespace paper_area_difference_l145_145519

def area (length width : ℕ) : ℕ := length * width

def combined_area (length width : ℕ) : ℕ := 2 * (area length width)

def sq_inch_to_sq_ft (sq_inch : ℕ) : ℕ := sq_inch / 144

theorem paper_area_difference :
  sq_inch_to_sq_ft (combined_area 15 24 - combined_area 12 18) = 2 :=
by
  sorry

end paper_area_difference_l145_145519


namespace larger_solution_exists_l145_145957

noncomputable def larger_value (y : ℝ) : Prop :=
  ∃ (r s : ℤ), y = r + Real.sqrt s ∧ r + s = 87

theorem larger_solution_exists :
  ∀ y : ℝ, (∃ p q : ℝ, p = Real.cbrt y ∧ q = Real.cbrt (26 - y) ∧ p + q = 3) →
  larger_value y :=
by
  assume y hyp,
  sorry

end larger_solution_exists_l145_145957


namespace steven_erasers_l145_145233

theorem steven_erasers (skittles erasers groups items_per_group total_items : ℕ)
  (h1 : skittles = 4502)
  (h2 : groups = 154)
  (h3 : items_per_group = 57)
  (h4 : total_items = groups * items_per_group)
  (h5 : total_items - skittles = erasers) :
  erasers = 4276 :=
by
  sorry

end steven_erasers_l145_145233


namespace sqrt_integer_part_equality_l145_145613

namespace Proof

def intPart (x : ℝ) : ℤ := ⌊x⌋

theorem sqrt_integer_part_equality (n : ℕ) : 
  intPart (Real.sqrt n + Real.sqrt (n+1)) = intPart (Real.sqrt (n+1) + Real.sqrt (n+2)) :=
by
  sorry

end Proof

end sqrt_integer_part_equality_l145_145613


namespace find_f_pi_plus_2_l145_145449

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * sin (2 * x) + b * tan x + 1

theorem find_f_pi_plus_2 (a b : ℝ) (h : f (-2) a b = 4) : f (π + 2) a b = -2 :=
by 
  sorry

end find_f_pi_plus_2_l145_145449


namespace inverse_proportion_quadrants_l145_145857

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ y = k / x) →
  (∀ x : ℝ, x ≠ 0 → ( (x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0) ) ) :=
by
  sorry

end inverse_proportion_quadrants_l145_145857


namespace base_n_divisible_by_13_l145_145813

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 7 + 3 * n + 5 * n^2 + 6 * n^3 + 3 * n^4 + 5 * n^5

-- The main theorem stating the result
theorem base_n_divisible_by_13 : 
  (∃ ns : Finset ℕ, ns.card = 16 ∧ ∀ n ∈ ns, 3 ≤ n ∧ n ≤ 200 ∧ f n % 13 = 0) :=
sorry

end base_n_divisible_by_13_l145_145813


namespace ball_probability_p6_l145_145011

noncomputable def p : ℕ → ℚ
| 0       := 1  -- initial condition (ball starts with A)
| 1       := 0  -- p_1 = 0
| n + 1 := 1/3 * (1 - p n)

theorem ball_probability_p6 : p 6 = 61/243 := by
  sorry

end ball_probability_p6_l145_145011


namespace find_number_of_partners_l145_145734

noncomputable def law_firm_partners (P A : ℕ) : Prop :=
  (P / A = 3 / 97) ∧ (P / (A + 130) = 1 / 58)

theorem find_number_of_partners (P A : ℕ) (h : law_firm_partners P A) : P = 5 :=
  sorry

end find_number_of_partners_l145_145734


namespace find_f2_plus_g2_l145_145835

-- Functions f and g are defined
variable (f g : ℝ → ℝ)

-- Conditions based on the problem
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function : Prop := ∀ x : ℝ, g (-x) = g x
def function_equation : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

-- Lean Theorem Statement
theorem find_f2_plus_g2 (h1 : even_function f) (h2 : odd_function g) (h3 : function_equation f g) :
  f 2 + g 2 = -4 :=
by
  sorry

end find_f2_plus_g2_l145_145835


namespace percentage_volume_occupied_is_100_l145_145744

-- Define the dimensions of the box and cube
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 2

-- Define the volumes
def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the number of cubes that fit in each dimension
def cubes_along_length : ℕ := box_length / cube_side
def cubes_along_width : ℕ := box_width / cube_side
def cubes_along_height : ℕ := box_height / cube_side

-- Define the total number of cubes and the volume they occupy
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height
def volume_occupied_by_cubes : ℕ := total_cubes * cube_volume

-- Define the percentage of the box volume occupied by the cubes
def percentage_volume_occupied : ℕ := (volume_occupied_by_cubes * 100) / box_volume

-- Statement to prove
theorem percentage_volume_occupied_is_100 : percentage_volume_occupied = 100 := by
  sorry

end percentage_volume_occupied_is_100_l145_145744


namespace container_ratio_l145_145179

theorem container_ratio (A B : ℝ) (h : (4 / 5) * A = (2 / 3) * B) : (A / B) = (5 / 6) :=
by
  sorry

end container_ratio_l145_145179


namespace monotonic_intervals_l145_145068

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem monotonic_intervals :
  (∀ x ∈ Ioi 1, ∃ ε > 0, ∀ y ∈ Ioo (x - ε) (x + ε), f y < f x) ∧
  (∀ x ∈ Iio 0 ∪ Ioo 0 1, ∃ ε > 0, ∀ y ∈ Ioo (x - ε) (x + ε), f y > f x) := by
sorry

end monotonic_intervals_l145_145068


namespace tangent_lines_to_circle_radius_of_circle_l145_145847

-- Problem 1: Tangent lines from point P to the circle
theorem tangent_lines_to_circle (x y : ℝ) (a : ℝ) (h_a : a = -8) (P : ℝ × ℝ) (h_P : P = (4, 5)) :
  (x^2 + y^2 - 2*x - 8 = 0 ∧ (∃ k : ℝ, y - 5 = k * (x - 4) ∧ ∃ x y, 8*x - 15*y + 43 = 0))
  ∨ (x = 4) := sorry

-- Problem 2: Radius of the circle given dot product condition
theorem radius_of_circle (x y a : ℝ) (AB_diameter : Prop) (h_dot_product : x * y = -6) :
  ∃ r : ℝ, r = sqrt 7:= sorry

end tangent_lines_to_circle_radius_of_circle_l145_145847


namespace cats_not_eating_either_l145_145544

/-- In a shelter with 80 cats, 15 cats like tuna, 60 cats like chicken, 
and 10 like both tuna and chicken, prove that 15 cats do not eat either. -/
theorem cats_not_eating_either (total_cats : ℕ) (like_tuna : ℕ) (like_chicken : ℕ) (like_both : ℕ)
    (h1 : total_cats = 80) (h2 : like_tuna = 15) (h3 : like_chicken = 60) (h4 : like_both = 10) :
    (total_cats - (like_tuna - like_both + like_chicken - like_both + like_both) = 15) := 
by
    sorry

end cats_not_eating_either_l145_145544


namespace sum_series_T_l145_145775

theorem sum_series_T :
  (∑ k in finset.range 50, (3 + (k + 1) * 5) / (3 ^ (51 - (k + 1)))) = 125.25 :=
by
  sorry

end sum_series_T_l145_145775


namespace sum_of_six_marked_angles_l145_145006

theorem sum_of_six_marked_angles (rect : Type) [IsRectangle rect] (segments : Fin 5 → Segment) (angles : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → (segments i).interior ∩ (segments j).interior = ∅)
  (h2 : ∀ i, angle (segments i) = angles i)
  (h_rect : sum_of_rectangle_angles rect = 4 * 90) :
  sum angles = 1080 := by
  sorry

end sum_of_six_marked_angles_l145_145006


namespace time_to_cross_stationary_train_l145_145756

theorem time_to_cross_stationary_train (t_pole : ℝ) (speed_train : ℝ) (length_stationary_train : ℝ) 
  (t_pole_eq : t_pole = 5) (speed_train_eq : speed_train = 64.8) (length_stationary_train_eq : length_stationary_train = 360) :
  (t_pole * speed_train + length_stationary_train) / speed_train = 10.56 := 
by
  rw [t_pole_eq, speed_train_eq, length_stationary_train_eq]
  norm_num
  sorry

end time_to_cross_stationary_train_l145_145756


namespace length_AE_l145_145628

theorem length_AE 
  (A B C D E : Point)
  (r : ℝ) (h : Triangle A B C)
  (h1 : Segment A B = 4)
  (h2 : IsoscelesTriangle A B C)
  (h3 : VertexAngle A B C = 120)
  (h4 : CircleIntersectPointsWithDiameter A B A C D B E)
  (h5 : ∠ A E B = 90) :
  Segment A E = 2 :=
sorry

end length_AE_l145_145628


namespace exists_infinitely_many_n_l145_145223

def divisor_count (n : ℕ) : ℕ :=
  if h : n ≠ 0 then
  ((List.range n).filter (λ d, d > 0 ∧ d ∣ n)).length + 1 -- including n itself and 1
  else 0

theorem exists_infinitely_many_n (k : ℕ) :
  ∃ᶠ n in Filter.atTop, ∀ r : ℕ, r ∈ Finset.range (k + 1) \ {0} → (n - divisor_count (n ^ r)) % r = 0 :=
sorry

end exists_infinitely_many_n_l145_145223


namespace area_of_tangent_segments_l145_145025

noncomputable def circle_radius: ℝ := 3
noncomputable def segment_length: ℝ := 6
noncomputable def area_of_region: ℝ := 9 * Real.pi

theorem area_of_tangent_segments :
  ∀ (O : Type*) [metric_space O] [normed_group O] [normed_space ℝ O],
  ∃ (circle : set O) (radius : ℝ) (line_segments : set (set O)),
    (radius = circle_radius) ∧
    (∀ segment ∈ line_segments, 
      ∃ (A B P : O), 
        segment = {x | dist A x + dist B x = dist A B} ∧
        dist A B = segment_length ∧
        ∃ (O : O), P ∈ circle ∧ dist O P = radius ∧ 
        is_tangent O segment P) ∧
    (measure_theory.measure theoretic_measure [metric_space O] [measure_space O]).measure (⋃₀ line_segments) =
    area_of_region := 
sorry

end area_of_tangent_segments_l145_145025


namespace quadratic_discriminant_positive_find_m_l145_145501

-- Define the quadratic equation and its properties
def quadratic (x m : Int) : Int := x^2 - (2 * m + 1) * x + (m^2 + m)

-- Part (1)
theorem quadratic_discriminant_positive (m : Int) : 
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  Δ > 0 := by
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  have h : Δ = 1 := by sorry
  show 1 > 0 from by sorry

-- Part (2)
theorem find_m (a b m : Int) : 
  quadratic a m = 0 ∧ quadratic b m = 0 ∧ (2 * a + b) * (a + 2 * b) = 20 → 
  m = -2 ∨ m = 1 := by
  intro h
  sorry

end quadratic_discriminant_positive_find_m_l145_145501


namespace evaluate_expression_l145_145393

-- Define the main statement we want to prove:
theorem evaluate_expression (t : ℝ) : 
    sqrt (9 * t^4 + 4 * t^2 + 4 * t) = abs t * sqrt ((3 * t^2 + 2 * t) * (3 * t^2 + 2 * t + 2)) :=
    sorry

end evaluate_expression_l145_145393


namespace range_of_a_l145_145487

theorem range_of_a (a : ℝ) :
  (∀ x, a * x^2 - x + (1 / 16 * a) > 0 → a > 2) →
  (0 < a - 3 / 2 ∧ a - 3 / 2 < 1 → 3 / 2 < a ∧ a < 5 / 2) →
  (¬ ((∀ x, a * x^2 - x + (1 / 16 * a) > 0) ∧ (0 < a - 3 / 2 ∧ a - 3 / 2 < 1))) →
  ((3 / 2 < a) ∧ (a ≤ 2)) ∨ (a ≥ 5 / 2) :=
by
  sorry

end range_of_a_l145_145487


namespace speed_of_current_l145_145260

theorem speed_of_current 
  (speed_in_still_water : ℝ) 
  (distance_downstream : ℝ) 
  (time_downstream_seconds : ℝ) 
  (speed_in_still_water_value : speed_in_still_water = 6) 
  (distance_downstream_value : distance_downstream = 110 / 1000) 
  (time_downstream_hours : time_downstream_seconds = 44 / 3600) :
  let speed_of_current := (distance_downstream / (time_downstream_seconds / 3600)) - speed_in_still_water
  in speed_of_current = 3.016 :=
by
  unfold speed_of_current
  rw [speed_in_still_water_value, distance_downstream_value, time_downstream_hours]
  sorry

end speed_of_current_l145_145260


namespace Part1_Part2_l145_145480

def f (x m : ℝ) : ℝ := m * x^2 + 2 * m * x - 1

def g (x m : ℝ) : ℝ := (3 * f x m + 4) / (x - 2)

noncomputable def minimum_g_when_x_gt_2 (x m : ℝ) (hx : x > 2) (h_eq : g x m = 12) : Prop :=
  g x m = 12

theorem Part1 (m : ℝ) (hm : m > 0) (h_sol : ∀ x : ℝ, f x m < 0 ↔ x ∈ -[3, 1]) : 
  ∃ (x : ℝ), x > 2 ∧ minimum_g_when_x_gt_2 x m := sorry

theorem Part2 (m : ℝ) (hm : m > 0) :
  (∃ x1 x2 : ℝ, x1 ∈ -[3, 0] ∧ x2 ∈ -[3, 0] ∧ |f x1 m - f x2 m| ≥ 4) ↔ m ∈ [1, +∞) := sorry

end Part1_Part2_l145_145480


namespace inner_v_w_l145_145950

variables (u v w : EuclideanSpace ℝ (Fin 3)) 

axiom norm_u : ∥u∥ = 1
axiom norm_v : ∥v∥ = 1
axiom norm_u_v_sum : ∥u + v∥ = 2
axiom w_eq : w = 2 * u + 3 * v + 2 * (u × v)

theorem inner_v_w : ⟪v, w⟫ = 5 :=
sorry

end inner_v_w_l145_145950


namespace arithmetic_sequence_properties_l145_145587

-- Definitions of the arithmetic sequence, sum of first n terms, and the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)

-- Main theorem to prove
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : sum_of_terms a S)
  (h_a1_lt_0 : a 1 < 0)
  (h_S1999_eq_S2023 : S 1999 = S 2023) :
  d > 0 ∧ S 4022 = 0 :=
by
  sorry

end arithmetic_sequence_properties_l145_145587


namespace root_expression_value_l145_145053

-- Define the root condition
def is_root (a : ℝ) : Prop := 2 * a^2 - 3 * a - 5 = 0

-- The main theorem statement
theorem root_expression_value {a : ℝ} (h : is_root a) : -4 * a^2 + 6 * a = -10 := by
  sorry

end root_expression_value_l145_145053


namespace suff_not_nec_l145_145832

variables (a b : ℝ)
def P := (a = 1) ∧ (b = 1)
def Q := (a + b = 2)

theorem suff_not_nec : P a b → Q a b ∧ ¬ (Q a b → P a b) :=
by
  sorry

end suff_not_nec_l145_145832


namespace negative_range_a_l145_145430

theorem negative_range_a (a : ℝ) (h_neg : a < 0) :
  (∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 > 1 + cos x) → a ≤ -2 :=
by
  sorry

end negative_range_a_l145_145430


namespace trip_time_l145_145514

theorem trip_time (T : ℝ) (x : ℝ) : 
  (150 / 4 = 50 / 30 + (x - 50) / 4 + (150 - x) / 30) → (T = 37.5) :=
by
  sorry

end trip_time_l145_145514


namespace constant_term_in_expansion_l145_145021

noncomputable def integral_result : ℝ := ∫ x in 0..π, Real.sin x

theorem constant_term_in_expansion :
  integral_result = 2 →
  let a := integral_result in
  let expr := (Real.sqrt x - a / x)^6 in
  -- The formula for the term T_(r+1) in binomial expansion could be 
  -- more formally defined, but we'll leverage the fact statement:
  -- The r such that 3 - 3r/2 = 0 is r = 2, hence we choose T_3
  -- which simplifies to binom 6 2 * (-2)^2 = 60
  (binom 6 2 * (-2)^2 = 60 :=
  sorry

end constant_term_in_expansion_l145_145021


namespace length_of_major_axis_of_ellipse_l145_145754

theorem length_of_major_axis_of_ellipse :
  ∀ (p1 p2 p3 p4 p5 : ℝ × ℝ),
    p1 = (0, 0) →
    p2 = (2, 2) →
    p3 = (-2, 2) →
    p4 = (4, 0) →
    p5 = (4, 4) →
    (∃ a b x0 y0, 
      (x0 = 2 ∧ y0 = 0) ∧
      ∀ (x y : ℝ), 
        ((x - x0) ^ 2) / (a ^ 2) + (y - y0) ^ 2 / (b ^ 2) = 1 → 
        (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 ∨ (x, y) = p4 ∨ (x, y) = p5) → 
    ∃ a, 2 * a = 4 :=
begin
  sorry
end

end length_of_major_axis_of_ellipse_l145_145754


namespace percentage_decrease_stock_l145_145376

theorem percentage_decrease_stock (F J M : ℝ)
  (h1 : J = F - 0.10 * F)
  (h2 : M = J - 0.20 * J) :
  (F - M) / F * 100 = 28 := by
sorry

end percentage_decrease_stock_l145_145376


namespace solution_proof_l145_145861

variable (t : ℝ)

def parametric_eq_line_l : ℝ × ℝ := (-1 - (Real.sqrt 2)/2 * t, 1 + (Real.sqrt 2)/2 * t)

def cartesian_eq_line_l : Prop := ∀ x y : ℝ, parametric_eq_line_l t = (x, y) → x + y = 0

def polar_eq_curve_C (θ : ℝ) : ℝ := 4 * Real.sin θ

def cartesian_eq_curve_C : Prop := ∀ ρ θ : ℝ, ρ = polar_eq_curve_C θ → (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2 - 4 * (ρ * Real.sin θ) = 0

noncomputable def length_segment_AB : ℝ :=
  let t_A := Real.sqrt 2
  let t_B := -Real.sqrt 2
  Real.abs (t_A - t_B)

theorem solution_proof : 
  ∃ x y : ℝ, cartesian_eq_line_l ∧ cartesian_eq_curve_C ∧ length_segment_AB = 2 * Real.sqrt 2 :=
by
  sorry

end solution_proof_l145_145861


namespace parabola_distance_l145_145062

def parabola (x y : ℝ) : Prop := y^2 = 8 * x

theorem parabola_distance (x y : ℝ) (h : parabola x y) :
  let F := (2, 0)
  let P := (x, y)
  let E := (0, y)
  let dist (a b : ℝ × ℝ) := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  dist P F - dist P E = 2 := 
by
  sorry

end parabola_distance_l145_145062


namespace parallelogram_height_l145_145809

theorem parallelogram_height
  (area : ℝ)
  (base : ℝ)
  (h_area : area = 375)
  (h_base : base = 25) :
  (area / base) = 15 :=
by
  sorry

end parallelogram_height_l145_145809


namespace two_digit_prime_sum_count_l145_145886

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_prime_sum_count : 
  let prime_sums := {2, 3, 5, 7, 11, 13, 17}
  ∑ k in (Finset.filter (λ n, is_prime (sum_of_digits n)) (Finset.filter two_digit (Finset.range 100))), 1 = 31 := 
by
  sorry

end two_digit_prime_sum_count_l145_145886


namespace maria_total_stamps_l145_145974

variable (currentStamps : ℕ) (percentageIncrease : ℕ) (additionalStamps : ℕ) (totalStamps : ℕ)

axiom h1 : currentStamps = 40
axiom h2 : percentageIncrease = 20
axiom h3 : additionalStamps = (percentageIncrease * currentStamps) / 100
axiom h4 : totalStamps = currentStamps + additionalStamps

theorem maria_total_stamps :
  totalStamps = 48 :=
by
  rw [h4, h3, h2, h1]
  sorry

end maria_total_stamps_l145_145974


namespace count_two_digit_numbers_with_prime_sum_l145_145885

-- Helper predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m < n → n % m ≠ 0

-- Helper function to get the sum of digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  tens + units

-- Define the set of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the set of two-digit numbers with prime digit sum
def two_digit_numbers_with_prime_sum :=
  {n : ℕ | n ∈ two_digit_numbers ∧ is_prime (sum_of_digits n)}

-- The main theorem
theorem count_two_digit_numbers_with_prime_sum : 
  {n : ℕ | n ∈ two_digit_numbers_with_prime_sum }.to_finset.card = 33 :=
by
  -- Placeholder for the actual proof
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145885


namespace quadratic_eq_two_distinct_real_roots_find_m_value_l145_145495

noncomputable def quadratic_eq_discriminant {m : ℝ} : ℝ :=
    (-(2 * m + 1)) ^ 2 - 4 * 1 * (m ^ 2 + m)

theorem quadratic_eq_two_distinct_real_roots (m : ℝ) : quadratic_eq_discriminant m > 0 :=
    by
    sorry

-- Let's define the conditions for the second part
def quadratic_eq_roots (m a b : ℝ) :=
    a + b = 2 * m + 1 ∧ a * b = m ^ 2 + m

def equation_holds (a b : ℝ) :=
    (2 * a + b) * (a + 2 * b) = 20

theorem find_m_value (m a b : ℝ) (h : quadratic_eq_roots m a b) (h_eq : equation_holds a b) :
    m = -2 ∨ m = 1 :=
    by
    sorry

end quadratic_eq_two_distinct_real_roots_find_m_value_l145_145495


namespace john_eggs_per_week_l145_145575

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l145_145575


namespace locus_of_C_l145_145846

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2

theorem locus_of_C (x y : ℝ) :
  (∃ x1 : ℝ, y = 2 * x1^2 ∧
  (x = x1 / 2 + (real.sqrt 3) / 2 * 2 * x1^2 ∨ x = x1 / 2 - (real.sqrt 3) / 2 * 2 * x1^2) ∧
  (y = y / 2 + (real.sqrt 3) / 2 * x1 ∨ y = y / 2 - (real.sqrt 3) / 2 * x1)) →
  (x^2 + 2 * real.sqrt 3 * x * y + 3 * y^2 + real.sqrt 3 * x - y = 0 ∨ 
   x^2 - 2 * real.sqrt 3 * x * y + 3 * y^2 - real.sqrt 3 * x - y = 0) :=
by
  sorry

end locus_of_C_l145_145846


namespace adoption_cost_l145_145931

theorem adoption_cost :
  let cost_cat := 50
  let cost_adult_dog := 100
  let cost_puppy := 150
  let num_cats := 2
  let num_adult_dogs := 3
  let num_puppies := 2
  (num_cats * cost_cat + num_adult_dogs * cost_adult_dog + num_puppies * cost_puppy) = 700 :=
by
  sorry

end adoption_cost_l145_145931


namespace piastres_in_6th_purse_l145_145935

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end piastres_in_6th_purse_l145_145935


namespace piastres_in_6th_purse_l145_145934

-- We define the amounts and constraints
variables (x : ℕ) -- Number of piastres in the first purse
variables (piastres : ℕ) -- Total number of piastres
variables (purse6 : ℕ) -- Number of piastres in the 6th purse

-- Given conditions as variables
axiom total_piastres : 150 = piastres
axiom num_purses : 10
axiom increasing_sequence : ∀ i j : ℕ, i < j → (x + i) < (x + j)
axiom first_last_condition : x ≥ (x + 9) / 2

-- Prove the number of piastres in the 6th purse
theorem piastres_in_6th_purse : purse6 = 16 :=
by
  -- placeholder for proof
  sorry

end piastres_in_6th_purse_l145_145934


namespace divisibility_by_5_l145_145309

theorem divisibility_by_5 (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divisibility_by_5_l145_145309


namespace interval_contains_integer_l145_145805

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, (3 * a < n) ∧ (n < 5 * a - 2)) ↔ (1.2 < a ∧ a < 4 / 3) ∨ (7 / 5 < a) :=
by sorry

end interval_contains_integer_l145_145805


namespace distance_from_S_to_PQR_eq_l145_145218

noncomputable def distance_S_to_face_PQR (S P Q R : ℝ³) (h_perpendicular1 : is_perpendicular S P Q) (h_perpendicular2 : is_perpendicular S Q R) (h_perpendicular3 : is_perpendicular S R P)
  (h_SP : dist S P = 10) (h_SQ : dist S Q = 10) (h_SR : dist S R = 8) : ℝ :=
  (40 * real.sqrt 19) / 57

-- here is the statement to prove the problem
theorem distance_from_S_to_PQR_eq 
  (S P Q R : ℝ³) (h_perpendicular1 : is_perpendicular S P Q) (h_perpendicular2 : is_perpendicular S Q R) (h_perpendicular3 : is_perpendicular S R P)
  (h_SP : dist S P = 10) (h_SQ : dist S Q = 10) (h_SR : dist S R = 8) : 
  distance_S_to_face_PQR S P Q R h_perpendicular1 h_perpendicular2 h_perpendicular3 h_SP h_SQ h_SR = (40 * real.sqrt 19) / 57 :=
begin
  sorry
end

end distance_from_S_to_PQR_eq_l145_145218


namespace sweets_total_l145_145681

theorem sweets_total 
  (red_sweets : ℕ)
  (green_sweets : ℕ)
  (neither_sweets : ℕ) :
  red_sweets = 49 →
  green_sweets = 59 →
  neither_sweets = 177 →
  red_sweets + green_sweets + neither_sweets = 285 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sweets_total_l145_145681


namespace find_naturals_for_divisibility_l145_145801

theorem find_naturals_for_divisibility (n : ℕ) (h1 : 3 * n ≠ 1) :
  (∃ k : ℤ, 7 * n + 5 = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := 
by
  sorry

end find_naturals_for_divisibility_l145_145801


namespace interval_of_monotonicity_minimum_value_of_a_plus_b_proof_of_inequality_l145_145849

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def g (a b x : ℝ) : ℝ := a * x + b

-- Problem (1) related definitions and assumptions
def F (b x : ℝ) : ℝ := f x - (2 * x + b)
def F_prime (b x : ℝ) : ℝ := 1 / x + 1 / (x ^ 2) - 2

theorem interval_of_monotonicity (b : ℝ) :
  (∀ x > 0, F_prime b x > 0 ↔ 0 < x ∧ x < 1) ∧ 
  (∀ x > 0, F_prime b x < 0 ↔ x > 1) :=
sorry

-- Problem (2) related definitions and assumptions
noncomputable def tangent_line_a (x0 : ℝ) : ℝ := 1 / x0 + 1 / (x0 ^ 2)
noncomputable def tangent_line_b (x0 : ℝ) : ℝ := Real.log x0 - 2 / x0 - 1
noncomputable def a_plus_b (x0 : ℝ) : ℝ := Real.log x0 + 1 / (x0 ^ 2) - 1 / x0 - 1

theorem minimum_value_of_a_plus_b : ∀ (x0 > 0), a_plus_b x0 ≥ -1 :=
sorry

-- Problem (3) related definitions and assumptions
noncomputable def P (x : ℝ) : ℝ := 2 * Real.exp (x - 5 / 2) - Real.log x + 1 / x

theorem proof_of_inequality : ∀ x > 0, P x > 0 :=
sorry

end interval_of_monotonicity_minimum_value_of_a_plus_b_proof_of_inequality_l145_145849


namespace trigonometric_identity_proof_l145_145811

theorem trigonometric_identity_proof :
    (sin 24 * cos 16 + cos 156 * sin 66) / (sin 28 * cos 12 + cos 152 * sin 72) = 1 / sin 80 :=
by
  sorry

end trigonometric_identity_proof_l145_145811


namespace lloyd_hourly_rate_l145_145970

variable (R : ℝ)  -- Lloyd's regular hourly rate

-- Conditions
def lloyd_works_regular_hours_per_day : Prop := R > 0
def lloyd_earns_excess_rate : Prop := 1.5 * R > 0
def lloyd_worked_hours : Prop := 10.5 > 7.5
def lloyd_earned_amount : Prop := 7.5 * R + 3 * 1.5 * R = 66

-- Theorem statement
theorem lloyd_hourly_rate (hr_pos : lloyd_works_regular_hours_per_day R)
                           (excess_rate : lloyd_earns_excess_rate R)
                           (worked_hours : lloyd_worked_hours)
                           (earned_amount : lloyd_earned_amount R) : 
    R = 5.5 :=
by sorry

end lloyd_hourly_rate_l145_145970


namespace total_isosceles_triangles_l145_145365

open Classical

noncomputable def is_isosceles_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := 
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d13 := (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  d12 = d13 ∨ d12 = d23 ∨ d13 = d23

/-- Prove total number of isosceles triangles formed by the given 9 grid points is 36. -/
theorem total_isosceles_triangles (p : ℕ → (ℝ × ℝ)) (h : set.range p = {
  (0, 0), (1, 0), (2, 0),
  (0, 1), (1, 1), (2, 1),
  (0, 2), (1, 2), (2, 2)
}) : 
  (finset.univ.filter (λ t : finset (ℕ × (ℝ × ℝ)), t.card = 3 ∧ is_isosceles_triangle (p t.1) (p t.2) (p t.3))).card = 36 := sorry

end total_isosceles_triangles_l145_145365


namespace sphere_to_hemisphere_l145_145344

noncomputable def initial_sphere_radius (r: ℝ) : ℝ :=
  r

theorem sphere_to_hemisphere (r : ℝ) (_ : r = 4 * real.cbrt 2) : 
  initial_sphere_radius r = 4 * real.cbrt 2 :=
by
  -- considering the provided conditions and statement
  sorry

end sphere_to_hemisphere_l145_145344


namespace find_salary_l145_145302

-- Define the conditions
variables (S : ℝ) -- S is the man's monthly salary

def saves_25_percent (S : ℝ) : ℝ := 0.25 * S
def expenses (S : ℝ) : ℝ := 0.75 * S
def increased_expenses (S : ℝ) : ℝ := 0.75 * S + 0.10 * (0.75 * S)
def monthly_savings_after_increase (S : ℝ) : ℝ := S - increased_expenses S

-- Define the problem statement
theorem find_salary
  (h1 : saves_25_percent S = 0.25 * S)
  (h2 : increased_expenses S = 0.825 * S)
  (h3 : monthly_savings_after_increase S = 175) :
  S = 1000 :=
sorry

end find_salary_l145_145302


namespace value_of_each_walmart_gift_card_l145_145563

variable (best_buy_value : ℕ) (best_buy_count : ℕ) (walmart_count : ℕ) (points_sent_bb : ℕ) (points_sent_wm : ℕ) (total_returnable : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  best_buy_value = 500 ∧
  best_buy_count = 6 ∧
  walmart_count = 9 ∧
  points_sent_bb = 1 ∧
  points_sent_wm = 2 ∧
  total_returnable = 3900

-- Result to prove
theorem value_of_each_walmart_gift_card : conditions best_buy_value best_buy_count walmart_count points_sent_bb points_sent_wm total_returnable →
  (total_returnable - ((best_buy_count - points_sent_bb) * best_buy_value)) / (walmart_count - points_sent_wm) = 200 :=
by
  intros h
  rcases h with
    ⟨hbv, hbc, hwc, hsbb, hswm, htr⟩
  sorry

end value_of_each_walmart_gift_card_l145_145563


namespace meanScore_is_91_666_l145_145564

-- Define Jane's quiz scores
def janesScores : List ℕ := [85, 88, 90, 92, 95, 100]

-- Define the total sum of Jane's quiz scores
def sumScores (scores : List ℕ) : ℕ := scores.foldl (· + ·) 0

-- The number of Jane's quiz scores
def numberOfScores (scores : List ℕ) : ℕ := scores.length

-- Define the mean of Jane's quiz scores
def meanScore (scores : List ℕ) : ℚ := sumScores scores / numberOfScores scores

-- The theorem to be proven
theorem meanScore_is_91_666 (h : janesScores = [85, 88, 90, 92, 95, 100]) :
  meanScore janesScores = 91.66666666666667 := by 
  sorry

end meanScore_is_91_666_l145_145564


namespace circle_equation_l145_145158

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l145_145158


namespace am_gm_inequality_l145_145953

theorem am_gm_inequality {a1 a2 a3 : ℝ} (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) :
  (a1 * a2 / a3) + (a2 * a3 / a1) + (a3 * a1 / a2) ≥ a1 + a2 + a3 := 
by 
  sorry

end am_gm_inequality_l145_145953


namespace gnomes_red_hats_small_noses_l145_145268

theorem gnomes_red_hats_small_noses :
  ∀ (total_gnomes red_hats blue_hats big_noses_blue_hats : ℕ),
  total_gnomes = 28 →
  red_hats = (3 * total_gnomes) / 4 →
  blue_hats = total_gnomes - red_hats →
  big_noses_blue_hats = 6 →
  (total_gnomes / 2) - big_noses_blue_hats = 8 →
  red_hats - 8 = 13 :=
by
  intros total_gnomes red_hats blue_hats big_noses_blue_hats
  intros h1 h2 h3 h4 h5
  sorry

end gnomes_red_hats_small_noses_l145_145268


namespace center_is_correct_and_point_not_on_circle_l145_145242

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def circle_center_and_radius (p1 p2: ℝ × ℝ) :=
  let center := midpoint p1 p2 in
  let radius := distance center p1 in
  (center, radius)

theorem center_is_correct_and_point_not_on_circle :
  let center := (-3, 1 : ℝ)
  let radius := 3 * real.sqrt 5
  center = midpoint (3, -2) (-9, 4) ∧
  distance center (0, 1) ≠ radius :=
by
  -- We provide the goal with the required conditions and correct answers
  let center := (-3, 1 : ℝ)
  let radius := 3 * real.sqrt 5
  have h_center : center = midpoint (3, -2) (-9, 4), by sorry
  have h_point_not_on_circle : distance center (0, 1) ≠ radius, by sorry
  exact ⟨h_center, h_point_not_on_circle⟩

end center_is_correct_and_point_not_on_circle_l145_145242


namespace petya_higher_chance_of_winning_l145_145214

/-- Petya gives Vasya two boxes of candies. Each box contains both chocolate candies and caramel candies.
    In total, there are 25 candies in both boxes. Vasya takes one candy from each box.
    If both candies turn out to be chocolate, Vasya wins. Otherwise, Petya wins.
    The probability that Vasya will get two caramels is 0.54.
    Prove that Petya has a higher chance of winning. --/
theorem petya_higher_chance_of_winning
  (total_candies : ℕ)
  (total_candies = 25 : Prop) 
  (prob_two_caramels : ℝ)
  (prob_two_caramels = 0.54 : Prop) :
  0.46 < 0.54 := 
by 
  sorry

end petya_higher_chance_of_winning_l145_145214


namespace find_m_in_function_l145_145070

noncomputable def f (m : ℝ) (x : ℝ) := (1 / 3) * x^3 - x^2 - x + m

theorem find_m_in_function {m : ℝ} (h : ∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f m x ≥ (1/3)) :
  m = 2 :=
sorry

end find_m_in_function_l145_145070


namespace part_one_part_two_l145_145507

open Real

variable {a b m : ℝ}

theorem part_one (m : ℝ) : 
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  in Δ > 0 := 
by
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  show Δ > 0 from by {
    simp[Δ],
    linarith
  }

theorem part_two (a b m : ℝ) (h1 : a + b = 2 * m + 1) (h2 : a * b = m^2 + m) (h3 : (2*a + b) * (a + 2*b) = 20) : 
  m = -2 ∨ m = 1 :=
by
  have h4: 9*m^2 + 9*m + 2 = 20 := by {
    rw [h1, h2, ← add_assoc, ← add_assoc],
    linarith
  }
  have h5 := solve_quadratic 9 9 (-18) h4
  cases h5 with h5_left h5_right
  exact or.inl h5_left
  exact or.inr h5_right

end part_one_part_two_l145_145507


namespace sum_of_forbidden_units_digits_is_22_l145_145060

def units_digit (n : ℕ) : ℕ := n % 10

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

def forbidden_units_digits : list ℕ := [2, 4, 7, 9]

theorem sum_of_forbidden_units_digits_is_22 :
  let sum := forbidden_units_digits.sum 
  in sum = 22 := by
  -- since we assume the problem and correct answer
  -- we directly state the outcome
  sorry

end sum_of_forbidden_units_digits_is_22_l145_145060


namespace yoga_to_exercise_ratio_l145_145972

-- Define parameters and ratio
variable (Y : ℕ) -- Time spent on yoga
def bike_time : ℕ := 18 -- Time spent riding the bicycle
def gym_ratio : ℕ := 2
def bike_ratio : ℕ := 3

-- Define the time spent on gym based on the given ratio
def gym_time : ℕ := (gym_ratio * bike_time) / bike_ratio

-- Define total exercise time
def total_exercise_time : ℕ := gym_time + bike_time

-- The theorem statement that we expect to prove
theorem yoga_to_exercise_ratio : Y : total_exercise_time = Y : 30 :=
by
  sorry

end yoga_to_exercise_ratio_l145_145972


namespace circle_equation_l145_145163

theorem circle_equation (M : ℝ × ℝ) :
  (∃ a : ℝ, M = (a, 1 - 2 * a) ∧ 2 * M.1 + M.2 - 1 = 0) ∧
  (distance M (3, 0) = distance M (0, 1)) →
  (∃ r : ℝ, (x - 1)^2 + (y + 1)^2 = r^2) :=
begin
  sorry
end

end circle_equation_l145_145163


namespace hexagon_interior_angles_and_regular_l145_145142

def is_sum_of_interior_angles (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : Prop :=
  ∑ i in Finset.range n, a i = 6 * 360

def is_regular_hexagon (a : ℝ) (b : ℝ) : Prop :=
  a = 6 * b

theorem hexagon_interior_angles_and_regular (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h_sum_interior : is_sum_of_interior_angles a b 6)
  (h_equal_relation : ∀ i ∈ Finset.range n, a i = 6 * b i) :
  (∑ i in Finset.range 6, a i = 2160) ∧ (∀ i j ∈ Finset.range 6, a i = a j) :=
sorry

end hexagon_interior_angles_and_regular_l145_145142


namespace sequence_general_term_l145_145864

theorem sequence_general_term (a : ℕ → ℤ) :
  a 1 = -1 ∧ 
  a 2 > a 1 ∧ 
  ∀ n : ℕ, n > 0 → |a (n + 1) - a n| = 2 ^ n ∧ 
  (∀ n : ℕ, n > 0 → a (2 * n - 1) > a (2 * (n - 1) + 1)) ∧ 
  (∀ n : ℕ, n > 0 → a (2 * n) > a (2 * (n - 1) + 2)) →
  ∀ n : ℕ, n > 0 → a n = (Int.ofNat (-2) ^ n - 1) / 3 :=
by
  sorry

end sequence_general_term_l145_145864


namespace circle_equation_l145_145156

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l145_145156


namespace vertex_of_parabola_is_max_and_correct_l145_145786

theorem vertex_of_parabola_is_max_and_correct (x y : ℝ) (h : y = -3 * x^2 + 6 * x + 1) :
  (x, y) = (1, 4) ∧ ∃ ε > 0, ∀ z : ℝ, abs (z - x) < ε → y ≥ -3 * z^2 + 6 * z + 1 :=
by
  sorry

end vertex_of_parabola_is_max_and_correct_l145_145786


namespace find_a_l145_145117

noncomputable def circle (a : ℝ) : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0

variables {x y x1 y1 x2 y2 x0 y0 : ℝ}

def passes_through_P (a : ℝ) : Prop := circle a (-5) a

def tangents_intersect_circle (a : ℝ) : Prop := 
  circle a x1 y1 ∧ circle a x2 y2

def slope_condition (a : ℝ) : Prop := 
  let k₁ := (y2 - y1) / (x2 - x1) in
  let k₂ := (x1 + x2 - 2) / (y1 + y2) in
  k₁ + k₂ = 0

theorem find_a :
  ∀ (a : ℝ), 
    passes_through_P a → 
    tangents_intersect_circle a → 
    slope_condition a → 
    (a = 3 ∨ a = -2) :=
by sorry

end find_a_l145_145117


namespace complex_modulus_l145_145660

open Complex

noncomputable def modulus_of_complex : ℂ :=
  (1 - 2 * Complex.I) * (1 - 2 * Complex.I) / Complex.I

theorem complex_modulus : Complex.abs modulus_of_complex = 5 :=
  sorry

end complex_modulus_l145_145660


namespace vaishali_hats_l145_145286

theorem vaishali_hats :
  -- Definitions
  let x := number of hats with three stripes:
  let hatsWithThreeStripes (x : ℕ) := x * 3
  let hatsWithFourStripes := 3 * 4
  let hatsWithNoStripes := 6 * 0
  let hatsWithFiveStripes := 2 * 5
  -- Condition
  hatsWithThreeStripes x + hatsWithFourStripes + hatsWithNoStripes + hatsWithFiveStripes = 34
  -- Conclusion
  → x = 4 := 
sorry

end vaishali_hats_l145_145286


namespace area_CGE_correct_l145_145766

-- Definitions for the geometric elements
variables (A B C D O E F : Type) [geometry.shape A B C D]
variables (in_trapezoid : geometry.is_trapezoid A B C D)
variables (intersect_at_O : geometry.diagonals_intersect_at A B C D O)
variables (extended_AC : geometry.extended A C E)
variables (extended_DB : geometry.extended D B F)
variables (E_O_conditions : geometry.equal_length C E A O)
variables (F_O_conditions : geometry.equal_length B F D O)

-- Given area information
variables (area_BFG : geometry.area (geometry.Triangle B F G) = 2015)

-- Goal: Prove the area of triangle CGE
theorem area_CGE_correct :
  geometry.area (geometry.Triangle C G E) = 2015 :=
sorry

end area_CGE_correct_l145_145766


namespace separator_is_comma_l145_145922

-- Define the condition as a predicate on statements
def multiple_variables_input (statement : String) : Prop :=
  -- Assume some function that checks if multiple variables are entered at the same time
  sorry

-- Define the function separator that returns the separator used
def separator (statement : String) : String :=
  -- Function implementation is assumed to exist
  sorry

-- The theorem statement
theorem separator_is_comma (statement : String) (h : multiple_variables_input(statement)) : separator(statement) = "," :=
  sorry

end separator_is_comma_l145_145922


namespace Jack_miles_per_four_months_l145_145125

theorem Jack_miles_per_four_months 
    (years_driving : ℕ) (total_miles : ℕ) (months_in_year : ℕ) (months_per_period : ℕ) 
    (total_periods : ℕ) (miles_per_period : ℕ)
    (H1 : years_driving = 9)
    (H2 : total_miles = 999000)
    (H3 : months_in_year = 12)
    (H4 : months_per_period = 4) 
    (H5 : total_periods = (years_driving * months_in_year) / months_per_period)
    : miles_per_period = total_miles / total_periods := 
by
  have H6 : total_periods = (9 * 12) / 4 := by
    rw [H1, H3, H4]
  have H7 : total_periods = 27 := by
    norm_num at H6
    exact H6
  have H8 : miles_per_period = 999000 / 27 := by
    rw [H2, H7]
  norm_num at H8
  exact H8

end Jack_miles_per_four_months_l145_145125


namespace proof_problem_l145_145445

def A_n_4_equals_40C_n_5 (n : ℕ) : Prop :=
  n.choose 4 = 40 * (n.choose 5)

def f (x : ℚ) (n : ℕ) : ℚ :=
  (x - 1/(3*x))^n

theorem proof_problem (n r : ℕ) (x : ℚ) (f : ℚ → ℕ → ℚ) 
  (H1 : A_n_4_equals_40C_n_5 n)
  (H2 : n = 7)
  (H3 : f x 7 = (x - 1/(3*x))^7) :
  n = 7 ∧
  let expansion_terms := [0, 3, 6] in
  (count_rational_terms (expansion_terms.length) = 3) ∧
  (largest_coefficient_term expansion_terms = 4) ∧
  (smallest_coefficient_term expansion_terms = 3) := by {
  sorry
}

-- Helper functions to count rational terms, find largest and smallest coefficient terms
def count_rational_terms (terms : ℕ) : ℕ := sorry

def largest_coefficient_term (terms : list ℕ) : ℕ := sorry

def smallest_coefficient_term (terms : list ℕ) : ℕ := sorry

end proof_problem_l145_145445


namespace sum_of_digits_is_twenty_l145_145100

theorem sum_of_digits_is_twenty (a b c d : ℕ) (h1 : c + b = 9) (h2 : a + d = 10) 
  (H1 : a ≠ b) (H2 : a ≠ c) (H3 : a ≠ d) 
  (H4 : b ≠ c) (H5 : b ≠ d) (H6 : c ≠ d) :
  a + b + c + d = 20 := 
sorry

end sum_of_digits_is_twenty_l145_145100


namespace range_of_m_l145_145822

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define the interval
def interval (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

-- Prove the range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, interval x → f x > 2 * x + m) ↔ m < - 5 / 4 :=
by
  -- This is the theorem statement, hence the proof starts here
  sorry

end range_of_m_l145_145822


namespace shirley_ends_with_106_l145_145229

-- Define the initial number of eggs and the number bought
def initialEggs : Nat := 98
def additionalEggs : Nat := 8

-- Define the final count as the sum of initial eggs and additional eggs
def finalEggCount : Nat := initialEggs + additionalEggs

-- State the theorem with the correct answer
theorem shirley_ends_with_106 :
  finalEggCount = 106 :=
by
  sorry

end shirley_ends_with_106_l145_145229


namespace square_partition_exists_integer_N_l145_145634

theorem square_partition_exists_integer_N :
  ∃ N : ℤ, ∀ n : ℤ, n ≥ N → ∃ k : ℕ, ∃ f : fin k → ℕ, ∃ a : fin k → ℕ, (∑ i, f i) = n ∧ (∀ i, a i * a i = f i) :=
sorry

end square_partition_exists_integer_N_l145_145634


namespace tin_in_new_mixture_l145_145718

/-- A proof to determine the total amount of tin in a new mixture of alloys. -/
theorem tin_in_new_mixture :
  let w_A := 90
  let w_B := 140
  let w_C := 50
  let ratio_A := (3, 4) -- lead to tin
  let ratio_B := (2, 5) -- tin to copper
  let ratio_C := (3, 2, 5) -- tin to copper to iron
  let tin_A := (4 / (3 + 4)) * w_A
  let tin_B := (2 / (2 + 5)) * w_B
  let tin_C := (3 / (3 + 2 + 5)) * w_C
  let total_tin := tin_A + tin_B + tin_C
  total_tin ≈ 106.42857 := by
sorry

end tin_in_new_mixture_l145_145718


namespace evaluate_combinations_l145_145394

theorem evaluate_combinations (n : ℕ) (h1 : 0 ≤ 5 - n) (h2 : 5 - n ≤ n) (h3 : 0 ≤ 10 - n) (h4 : 10 - n ≤ n + 1) (h5 : n > 0) :
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 :=
sorry

end evaluate_combinations_l145_145394


namespace parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l145_145871

section vector

variables {k : ℝ}
def a : ℝ × ℝ := (6, 2)
def b : ℝ × ℝ := (-2, k)

-- Parallel condition
theorem parallel_vectors : 
  (∀ c : ℝ, (6, 2) = -2 * (c * k, c)) → k = -2 / 3 :=
by 
  sorry

-- Perpendicular condition
theorem perpendicular_vectors : 
  6 * (-2) + 2 * k = 0 → k = 6 :=
by 
  sorry

-- Obtuse angle condition
theorem obtuse_angle_vectors : 
  6 * (-2) + 2 * k < 0 ∧ k ≠ -2 / 3 → k < 6 ∧ k ≠ -2 / 3 :=
by 
  sorry

end vector

end parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l145_145871


namespace gcd_228_1995_l145_145696

-- Define the gcd function according to the Euclidean algorithm
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a
  else gcd b (a % b)

-- Prove that the gcd of 228 and 1995 is 57
theorem gcd_228_1995 : gcd 228 1995 = 57 :=
by
  sorry

end gcd_228_1995_l145_145696


namespace find_general_term_formula_l145_145078

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a : ℕ → ℕ), (∀ n, a (n+1) = 2 * a n + 3 * 5^n) ∧ (a 1 = 6)

theorem find_general_term_formula :
  ∀ a : ℕ → ℕ, sequence a → ∀ n, a n = 5^n + 2^(n-1) :=
by
  intros
  sorry

end find_general_term_formula_l145_145078


namespace monkeys_and_bananas_l145_145638

theorem monkeys_and_bananas (m1 m2 t b1 b2 : ℕ) (h1 : m1 = 8) (h2 : t = 8) (h3 : b1 = 8) (h4 : b2 = 3) : m2 = 3 :=
by
  -- Here we will include the formal proof steps
  sorry

end monkeys_and_bananas_l145_145638


namespace product_of_segments_l145_145927

variables (A B C D E F : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [has_incircle ABC] [has_incircle EFC]
variables {ABC EFC : triangle}

-- Define the properties of ABC and EFC
def isosceles_triangle (ABC : triangle) := 
  (ABC.side CA = ABC.side CB) ∧
  midpoint D ABC.side AB ∧
  incircle_same ABC EFC D EF

-- State the metric property to be proved
theorem product_of_segments {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  (h : isosceles_triangle ABC) :
  DE * DF = DA * DA :=
sorry

end product_of_segments_l145_145927


namespace unique_sequence_exists_l145_145640

theorem unique_sequence_exists (a : ℕ → ℝ) (B A : ℕ → ℝ) (k r : ℕ) (B1_ne_0 : B 1 ≠ 0) (Bk_ne_0 : B k ≠ 0) (A0_ne_0 : A 0 ≠ 0) (sum_B_ne_1 : ∑ i in range k, B i ≠ 1) : 
  ∃! (b : ℕ → ℝ), 
    (∃ E : ℕ → ℝ, E 0 ≠ 0 ∧ ∀ n : ℕ, b n = ∑ i in range (r+1), E i * n^(r-i)) ∧
    (∀ n : ℕ, 
      (a (n + k) - b (n + k)) - ∑ i in range (k + 1), B i * (a (n + k - i) - b (n + k - i)) = 0) :=
sorry

end unique_sequence_exists_l145_145640


namespace a_4_value_l145_145823

-- Define the sequence and sum function
def S : ℕ+ → ℤ
| ⟨n, h⟩ := 2 * n ^ 2 - 3 * n

-- Define the term function using the sum function
def a : ℕ+ → ℤ
| ⟨1, _⟩ := S ⟨1, _⟩
| ⟨n + 1, _⟩ := S ⟨n + 1, _⟩ - S ⟨n, by simp [Nat.succ_pos]⟩

-- Prove that a_4 = 11
theorem a_4_value : a ⟨4, by simp⟩ = 11 := sorry

end a_4_value_l145_145823


namespace incircle_circumcircle_relation_incircle_circumcircle_inequality_l145_145647

structure Triangle :=
(A B C O : Point)
(incircle_center : ∃ (O : Point), center of incircle at O)
(incircle_radius : ℝ)
(circumcircle_radius : ℝ)

theorem incircle_circumcircle_relation 
  (ABC : Triangle)
  (O_center : ABC.incircle_center = some O)
  (varrho r : ℝ)
  (h_varrho : ABC.incircle_radius = varrho)
  (h_r : ABC.circumcircle_radius = r) :
  let OA := dist O ABC.A 
  let OB := dist O ABC.B 
  let OC := dist O ABC.C 
  in OA * OB * OC = 4 * varrho^2 * r :=
sorry

theorem incircle_circumcircle_inequality 
  (ABC : Triangle)
  (O_center : ABC.incircle_center = some O)
  (varrho r : ℝ)
  (h_varrho : ABC.incircle_radius = varrho)
  (h_r : ABC.circumcircle_radius = r) :
  let OA := dist O ABC.A 
  let OB := dist O ABC.B 
  let OC := dist O ABC.C 
  in OA + OB + OC >= 6 * varrho 
     ∧ (OA = OB ∧ OB = OC → r = 2 * varrho) :=
sorry

end incircle_circumcircle_relation_incircle_circumcircle_inequality_l145_145647


namespace ellipse_properties_triangle_area_l145_145044

open Real

noncomputable def ellipse_equation (a b : ℝ) : Prop := 
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a > b) 
  (ecc : b / sqrt (a^2 - b^2) = (√3) / 2) 
  (vertex_directrix : a = 2 ∧ b = 1) :
  ellipse_equation 2 1 :=
sorry

theorem triangle_area 
  (a b k₁ k₂ m : ℝ) 
  (h1 : k₁ * k₂ = -1 / 4) 
  (h2 : a = 2) 
  (h3 : b = 1) :
  area_triangle_MON k₁ k₂ m = 1 :=
sorry

end ellipse_properties_triangle_area_l145_145044


namespace find_number_of_children_l145_145599

theorem find_number_of_children (N : ℕ) (B : ℕ) 
    (h1 : B = 2 * N) 
    (h2 : B = 4 * (N - 160)) 
    : N = 320 := 
by
  sorry

end find_number_of_children_l145_145599


namespace reflected_ray_equation_l145_145742

noncomputable def equation_of_reflected_ray 
  (light_emitted_point : ℝ × ℝ)
  (line_parallel_slope : ℝ)
  (intersection_y_axis : ℝ × ℝ) 
  : Prop :=
  let reflected_point := (-light_emitted_point.1, light_emitted_point.2) in
  (intersection_y_axis.1 + 2 * intersection_y_axis.2 - 4 = 0)

theorem reflected_ray_equation :
  ∀ (light_emitted_point : ℝ × ℝ) 
  (line_parallel_slope : ℝ)
  (intersection_y_axis : ℝ × ℝ),
  light_emitted_point = (2, 3) ∧
  line_parallel_slope = 1/2 ∧
  intersection_y_axis = (0, 2) →
  equation_of_reflected_ray light_emitted_point line_parallel_slope intersection_y_axis
:= 
by
  intros,
  sorry

end reflected_ray_equation_l145_145742


namespace bucket_capacities_l145_145263

theorem bucket_capacities (a b c : ℕ) 
  (h1 : a + b + c = 1440) 
  (h2 : a + b / 5 = c) 
  (h3 : b + a / 3 = c) : 
  a = 480 ∧ b = 400 ∧ c = 560 := 
by 
  sorry

end bucket_capacities_l145_145263


namespace distance_from_A_to_focus_l145_145860

noncomputable def distance (a : ℝ) (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_from_A_to_focus
  (a : ℝ)
  (h_passes_through_A : 1^2 = a * (1/4))
  (focus_x focus_y : ℝ)
  (h_focus_coordinates : focus_x = 1 ∧ focus_y = 0)
  :
  distance a (1/4) 1 1 0 = 5/4 :=
by
  sorry

end distance_from_A_to_focus_l145_145860


namespace box_with_most_balls_l145_145269

theorem box_with_most_balls (A B C D : ℕ) :
  -- Initial conditions
  (A + B + C + D = 64) ∧ 
  ((A + A_div2 + A_div2 = 16) ∧ ((B + B_div2 + B_div2 = 16) ∧ (C + C_div2 + C_div2 = 16) ∧ (D + D_div2 + D_div2 = 16)
) :=
  (A = 33) sorry

end box_with_most_balls_l145_145269


namespace chocolate_bar_original_price_l145_145128

-- Definitions of conditions
def num_chocolate_bars := 10
def num_gummy_bears := 10
def num_chocolate_chips := 20
def discount_chocolate_bars := 0.15
def discount_gummy_bears := 0.05
def discount_chocolate_chips := 0.10
def total_cost_after_discounts := 150
def original_cost_gummy_bears := 2
def original_cost_chocolate_chips := 5

-- Required proof
theorem chocolate_bar_original_price :
  let total_cost_gummy_bears_before := num_gummy_bears * original_cost_gummy_bears,
      total_cost_gummy_bears_after := total_cost_gummy_bears_before * (1 - discount_gummy_bears),
      total_cost_chocolate_chips_before := num_chocolate_chips * original_cost_chocolate_chips,
      total_cost_chocolate_chips_after := total_cost_chocolate_chips_before * (1 - discount_chocolate_chips),
      other_total_cost_after := total_cost_gummy_bears_after + total_cost_chocolate_chips_after,
      total_cost_chocolate_bars_after := total_cost_after_discounts - other_total_cost_after,
      x := total_cost_chocolate_bars_after / (num_chocolate_bars * (1 - discount_chocolate_bars))
  in
    x = 4.82 :=
sorry

end chocolate_bar_original_price_l145_145128


namespace midline_le_half_side_l145_145225

open Set Real

variable {A B C M₁ M₂ : Point}

-- Conditions: Definition of a triangle's midline
def is_midpoint (P Q : Point) (M : Point) : Prop :=
  dist P M = dist Q M ∧ dist P Q = 2 * dist P M

def is_midline (P Q R M₁ M₂ : Point) : Prop :=
  is_midpoint P Q M₁ ∧ is_midpoint P R M₂ ∧ collinear {M₁, M₂, centroid P Q R}

-- Proof problem: Proving the inequality for the midline not being longer than half the side
theorem midline_le_half_side (A B C M₁ M₂ : Point) :
  is_midline A B C M₁ M₂ → dist M₁ M₂ ≤ 1 / 2 * dist A B :=
by
  sorry

end midline_le_half_side_l145_145225


namespace probability_meeting_P_Q_l145_145243

-- Definition of the grid and positions
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨4, 3⟩

-- Definition of shortest paths
def num_paths (start : Point) (end : Point) : ℕ :=
  nat.choose ((end.x - start.x) + (end.y - start.y), end.y - start.y)

-- Definition that P and Q meet
def meet (P_path : list Point) (Q_path : list Point) : Prop :=
  ∃ t : ℕ, (t < P_path.length) ∧ (P_path.nth t = Q_path.nth t)

-- The final theorem statement
theorem probability_meeting_P_Q : 
  (num_paths A B) = 35 →
  (num_paths B A) = 35 →
  (∃ P_path Q_path, meet P_path Q_path) →
  ∃ (prob : ℚ), prob = 1/5 :=
begin
  intros h1 h2 h3,
  use (1/5 : ℚ),
  sorry,
end

end probability_meeting_P_Q_l145_145243


namespace petya_higher_chance_of_winning_l145_145205

theorem petya_higher_chance_of_winning 
  {total_candies : ℕ} (h_total : total_candies = 25)
  {P_two_caramel : ℝ} (h_P_two_caramel : P_two_caramel = 0.54) :
  let P_two_chocolate := 1 - P_two_caramel in
  P_two_chocolate < 0.5 :=
by {
  sorry
}

end petya_higher_chance_of_winning_l145_145205


namespace sum_x_and_f_values_l145_145004

def f (x : ℝ) : ℝ := 2*x + 3

def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 12

def linear (x : ℝ) : ℝ := 4 - x

theorem sum_x_and_f_values : 
  let xs := {x : ℝ | quadratic x = linear x ∧ x ≤ 2 ∨ quadratic x = linear x ∧ x ≥ 6} in
  let sum_x := ∑ x in xs, x in
  let sum_fx := ∑ x in xs, f x in
  sum_x = 9 ∧ sum_fx = 24 :=
by
  sorry

end sum_x_and_f_values_l145_145004


namespace smallest_union_cardinality_l145_145630

open Finset

theorem smallest_union_cardinality (A B C : Finset α) [Fintype α] 
  (hA : card A = 30) (hB : card B = 25) (hC : card C = 10) 
  (hCA : C ⊆ A) (hCB : C ⊆ B) : card (A ∪ B) = 30 :=
by
  sorry

end smallest_union_cardinality_l145_145630


namespace domain_of_f_l145_145403

def f (x : ℝ) : ℝ := (x - 2) / (x^3 + 8 * x^2 + 19 * x + 12)

theorem domain_of_f :
  (∀ x : ℝ, x ∉ {x : ℝ | (x + 4) * (x + 3) * (x + 1) = 0}) →
  ∀ x : ℝ, x ∈ 
    Set.Ioo (-∞) (-4) ∪ Set.Ioo (-4) (-3) ∪ Set.Ioo (-3) (-1) ∪ Set.Ioo (-1) ∞ :=
  sorry

end domain_of_f_l145_145403


namespace quadratic_roots_d_l145_145671

theorem quadratic_roots_d (d : ℝ) : 
  (∀ x, x^2 + 6 * x + d = 0 → (x = (-6 + real.sqrt d) / 2 ∨ x = (-6 - real.sqrt d) / 2)) →
  d = 36 / 5 :=
  sorry

end quadratic_roots_d_l145_145671


namespace coprime_arith_seq_l145_145188

def sequence (a d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ k => a + k * d)

theorem coprime_arith_seq :
  ∃ (a d : ℕ) (n : ℕ), 
  n = 100 ∧
  (∀ i j : ℕ, i ≠ j ∧ i < n ∧ j < n -> Nat.coprime (a + i * d) (a + j * d)) :=
by
  -- it suffices to consider a = 1 and d = 99! to show the existence
  -- sorry used to indicate proof is skipped
  sorry

end coprime_arith_seq_l145_145188


namespace circle_equation_l145_145157

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l145_145157


namespace quadratic_discriminant_positive_find_m_l145_145498

-- Define the quadratic equation and its properties
def quadratic (x m : Int) : Int := x^2 - (2 * m + 1) * x + (m^2 + m)

-- Part (1)
theorem quadratic_discriminant_positive (m : Int) : 
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  Δ > 0 := by
  let Δ := (-(2 * m + 1))^2 - 4 * (m^2 + m)
  have h : Δ = 1 := by sorry
  show 1 > 0 from by sorry

-- Part (2)
theorem find_m (a b m : Int) : 
  quadratic a m = 0 ∧ quadratic b m = 0 ∧ (2 * a + b) * (a + 2 * b) = 20 → 
  m = -2 ∨ m = 1 := by
  intro h
  sorry

end quadratic_discriminant_positive_find_m_l145_145498


namespace projection_displacement_result_l145_145429

/-- Definition of vector type to simplify statements -/
def vec3 := ℝ × ℝ × ℝ

/-- Plane condition -/
def plane_condition (x y z : ℝ) : Prop := x + 2 * y - 3 * z = 1

/-- Definitions of given vectors -/
def v : vec3 := (4, 2, 6)
def d : vec3 := (-1, 1, 2)
def n : vec3 := (1, 2, -3)

/-- Projection function (this would normally be defined, but it is skipped here) -/
noncomputable def projection (v n : vec3) : vec3 := 
  let dot_product (a b : vec3) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar := (dot_product v n) / (dot_product n n)
  let scale_vector (a : ℝ) (b : vec3) : vec3 := (a * b.1, a * b.2, a * b.3)
  let subtract_vector (a b : vec3) : vec3 := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
  subtract_vector v (scale_vector scalar n)

/-- Displacement function -/
def displace (p d : vec3) : vec3 := (p.1 + d.1, p.2 + d.2, p.3 + d.3)

/-- Theorem to prove: Given the plane and vectors v and d, the resulting displaced projection is
 the expected vector p_d -/
theorem projection_displacement_result :
  ∃ p, p = projection v n ∧ displace p d = (26 / 7, 31 / 7, 41 / 7) :=
sorry

end projection_displacement_result_l145_145429


namespace cube_dot_path_length_l145_145729

noncomputable def length_of_path_traced_by_dot_in_terms_of_d (d : ℝ) : Prop :=
  ∃ c, c = 2 * real.sqrt 2 ∧ d = 2 * real.sqrt 2

theorem cube_dot_path_length :
  ∀ (cube_length : ℝ) (dot_position : (ℝ × ℝ)) (condition1 : cube_length = 2)
  (condition2 : dot_position = (1, 1)) (d : ℝ),
    length_of_path_traced_by_dot_in_terms_of_d d :=
by sorry

end cube_dot_path_length_l145_145729


namespace smallest_n_divides_999_l145_145439

/-- 
Given \( 1 \leq n < 1000 \), \( n \) divides 999, and \( n+6 \) divides 99,
prove that the smallest possible value of \( n \) is 27.
 -/
theorem smallest_n_divides_999 (n : ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : n < 1000) 
  (h3 : n ∣ 999) 
  (h4 : n + 6 ∣ 99) : 
  n = 27 :=
  sorry

end smallest_n_divides_999_l145_145439


namespace unique_value_for_a_l145_145446

noncomputable def set_values_for_a (a : ℝ) : Prop :=
  let M := {1, 2, a^2 - 3*a - 1}
  let N := {-1, a, 3}
  let intersection := M ∩ N
  intersection = {3}

theorem unique_value_for_a : ∀ (a : ℝ), set_values_for_a a → a = 4 :=
by
  intros a h
  sorry

end unique_value_for_a_l145_145446


namespace correct_equation_l145_145549

theorem correct_equation (x : ℕ) :
  (30 * x + 8 = 31 * x - 26) := by
  sorry

end correct_equation_l145_145549


namespace divisor_is_13_l145_145890

theorem divisor_is_13 (N D : ℕ) (h1 : N = 32) (h2 : (N - 6) / D = 2) : D = 13 := by
  sorry

end divisor_is_13_l145_145890


namespace find_x_l145_145798

theorem find_x (x : ℝ) (h : 9^(Real.log x / Real.log 8) = 81) : x = 64 := by
  sorry

end find_x_l145_145798


namespace question_1_question_2_l145_145862

def curve_is_ellipse (m : ℝ) : Prop :=
  (3 - m > 0) ∧ (m - 1 > 0) ∧ (3 - m > m - 1)

def domain_is_R (m : ℝ) : Prop :=
  m^2 < (9 / 4)

theorem question_1 (m : ℝ) :
  curve_is_ellipse m → 1 < m ∧ m < 2 :=
sorry

theorem question_2 (m : ℝ) :
  (curve_is_ellipse m ∧ domain_is_R m) → 1 < m ∧ m < (3 / 2) :=
sorry

end question_1_question_2_l145_145862


namespace gnomes_red_hats_small_noses_l145_145267

theorem gnomes_red_hats_small_noses :
  ∀ (total_gnomes red_hats blue_hats big_noses_blue_hats : ℕ),
  total_gnomes = 28 →
  red_hats = (3 * total_gnomes) / 4 →
  blue_hats = total_gnomes - red_hats →
  big_noses_blue_hats = 6 →
  (total_gnomes / 2) - big_noses_blue_hats = 8 →
  red_hats - 8 = 13 :=
by
  intros total_gnomes red_hats blue_hats big_noses_blue_hats
  intros h1 h2 h3 h4 h5
  sorry

end gnomes_red_hats_small_noses_l145_145267


namespace correct_conclusions_l145_145828

variables {R : Type*} [Real R]
variable (f g : R → R)
variable (f' g' : R → R)

-- Assume functions and their derivatives
variable (hf : ∀ x, f' x = derivative (f x))
variable (hg : ∀ x, g' x = derivative (g x))

-- Conditions given in the problem
def f_even_at_2 := ∀ x, f (x + 2) = f (-x + 2)
def g_odd_at_shift := ∀ x, g (-x + 1) - 2 = -(g (x + 1) - 2)
def f_and_g_relation := ∀ x, f (3 - x) + g (x - 1) = 2

theorem correct_conclusions (h1 : f_even_at_2 f)
                           (h2 : g_odd_at_shift g)
                           (h3 : f_and_g_relation f g) :
  f' 2022 = 0 ∧ (∀ x, g' x = -(g' (-x))) :=
  sorry

end correct_conclusions_l145_145828


namespace trigonometric_identity_l145_145093

-- Definitions according to the conditions
variable (α : ℝ)
axiom sin_add_cos_eq : sin α + cos α = (2 * real.sqrt 3) / 3
axiom is_acute_angle : 0 < α ∧ α < real.pi / 2

-- The statement to prove
theorem trigonometric_identity :
  (sin α + cos α = (2 * real.sqrt 3) / 3) ∧ (0 < α ∧ α < real.pi / 2) →
  (1 + tan α) / (sin (2 * α) - cos (2 * α) + 1) = 3 := 
by
  intros h
  sorry

end trigonometric_identity_l145_145093


namespace tetrahedron_projection_is_orthocenter_l145_145104

-- Define the conditions and the theorem to be proved
theorem tetrahedron_projection_is_orthocenter
  (T : Type)
  (V1 V2 V3 V4 : T)
  (L1 L2 L3 : T → T)
  (is_perpendicular : ∀ {a b c : T}, (L1 a = L2 b) → (L2 b = L3 c) → (L3 c = L1 a) → True)
  (is_tetrahedron : ∀ {P Q R S : T}, True) :
  ∀ {Proj : T}, (Proj V1 = V2 ∧ Proj V3 = V4) → Proj = orthocenter V1 V2 V3 :=
sorry

end tetrahedron_projection_is_orthocenter_l145_145104


namespace complex_on_real_axis_implies_b_zero_l145_145550

-- Definitions
variables (a b : ℝ) (z : ℂ)

-- Problem statement
theorem complex_on_real_axis_implies_b_zero (h : z = a + b * complex.I) (hz : z.im = 0) : b = 0 :=
sorry

end complex_on_real_axis_implies_b_zero_l145_145550


namespace max_sum_x_y_l145_145612

theorem max_sum_x_y {x y a b : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a * x + b * y = 1) : 
  x + y ≤ 2 :=
sorry

end max_sum_x_y_l145_145612


namespace PO_is_circumcenter_l145_145991

-- Definitions of the conditions:
variables {P A B C O : Type}
variable [metric_space O]

-- Condition: Point P is outside the plane of triangle ABC.
-- This can be assumed as the points are distinct and showing P is not coplanar with A, B, C.
-- Condition: PO is perpendicular to plane ABC at foot O.
variable (PO_perp_ABC: ∀ {A B C : Point}, ∃ O : Point, (P ≠ O) ∧ (PO ⊥ plane A B C))
-- Condition: PA = PB = PC.
variable (PA_eq_PB_eq_PC : dist P A = dist P B ∧ dist P B = dist P C)

-- Goal: Prove that point O is the circumcenter.
theorem PO_is_circumcenter 
(PO_perp_ABC : PO ⊥ plane A B C)
(PA_eq_PB_eq_PC : dist P A = dist P B ∧ dist P B = dist P C) :
  dist O A = dist O B ∧ dist O B = dist O C := 
sorry

end PO_is_circumcenter_l145_145991


namespace max_intersections_convex_polygons_l145_145911

noncomputable def max_intersections (m1 m2 : ℕ) (h : m2 ≥ m1 + 3) : ℕ :=
  m1 * m2

theorem max_intersections_convex_polygons (m1 m2 : ℕ) (h : m2 ≥ m1 + 3) 
  (Q1 Q2 : Type) [convex_polygon Q1 m1] [convex_polygon Q2 m2]
  (h_non_shared : ¬(∃ (e : edge Q1) (f : edge Q2), e = f)) :
  max_intersections m1 m2 h = m1 * m2 :=
begin
  sorry
end

end max_intersections_convex_polygons_l145_145911


namespace fuchsia_to_mauve_l145_145767

def fuchsia_to_mauve_amount (F : ℝ) : Prop :=
  let blue_in_fuchsia := (3 / 8) * F
  let red_in_fuchsia := (5 / 8) * F
  blue_in_fuchsia + 14 = 2 * red_in_fuchsia

theorem fuchsia_to_mauve (F : ℝ) (h : fuchsia_to_mauve_amount F) : F = 16 :=
by
  sorry

end fuchsia_to_mauve_l145_145767


namespace MD_geq_MA_l145_145629

variables {A B C D M : Point}
variables {ℝ : Type*} [metric_space ℝ] [semi_normed_add_comm_group ℝ] [normed_space ℝ ℝ]

-- Define the existing conditions
def Segment (x y : Point) : Set Point := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • x + t • y}

def intersection (u v : Set Point) : Set Point := {p | p ∈ u ∧ p ∈ v}

def angle_eq_90 (x y z : Point) : Prop := 
  let v₁ := y - x in
  let v₂ := y - z in
  ∥v₁∥ * ∥v₂∥ = ∥v₁∥ * ∥v₂∥ * cos (π / 2)

-- The main theorem
theorem MD_geq_MA (h_intersect : M ∈ intersection (Segment A C) (Segment B D)) 
                  (h_equal : ∥A - B∥ = ∥C - D∥) 
                  (h_angle : angle_eq_90 A C D) : 
  ∥M - D∥ ≥ ∥M - A∥ := 
begin
  sorry
end

end MD_geq_MA_l145_145629


namespace aero_flight_tees_per_package_l145_145770

theorem aero_flight_tees_per_package {A : ℕ} :
  (∀ (num_people : ℕ), num_people = 4 → 20 * num_people ≤ A * 28 + 2 * 12) →
  A * 28 ≥ 56 →
  A = 2 :=
by
  intros h1 h2
  sorry

end aero_flight_tees_per_package_l145_145770


namespace area_fraction_of_square_hole_l145_145346

theorem area_fraction_of_square_hole (A B C M N : ℝ)
  (h1 : B = C)
  (h2 : M = 0.5 * A)
  (h3 : N = 0.5 * A) :
  (M * N) / (B * C) = 1 / 4 :=
by
  sorry

end area_fraction_of_square_hole_l145_145346


namespace Q_value_l145_145149

theorem Q_value (a b c P Q : ℝ) (h1 : a + b + c = 0)
    (h2 : (a^2 / (2 * a^2 + b * c)) + (b^2 / (2 * b^2 + a * c)) + (c^2 / (2 * c^2 + a * b)) = P - 3 * Q) : 
    Q = 8 := 
sorry

end Q_value_l145_145149


namespace find_a_plus_b_l145_145853

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 4

theorem find_a_plus_b
  (a b : ℝ)
  (f_even : ∀ x : ℝ, f x a = f (-x) a)
  (x_domain : ∀ x : ℝ, x ∈ set.Icc b 3) :
  a + b = -3 :=
by
  sorry

end find_a_plus_b_l145_145853


namespace minimum_value_of_a_l145_145022

variables {f : ℝ → ℝ} {a : ℝ}

-- Define f(x) as an odd function
def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the given function f(x) with the condition when x > 0
def f_pos (a : ℝ) (x : ℝ) : Prop :=
  x > 0 → f x = exp x + a

-- Define the derived function f(x) when x < 0 using the odd property
def f_neg (a : ℝ) (x : ℝ) : Prop :=
  x < 0 → f x = -exp (-x) - a

-- Define the monotonicity condition on ℝ
def monotonic (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- The mathematical proof problem in Lean
theorem minimum_value_of_a
  (h_odd : odd_function f)
  (h_f_pos : f_pos a)
  (h_f_neg : f_neg a)
  (h_monotonic : monotonic f) :
  a ≥ -1 / 2 :=
sorry

end minimum_value_of_a_l145_145022


namespace find_AP_CP_l145_145947

variable (P A B C D : Point)
variable (O1 O2 : Circle → Point)
variable (s : ℝ) (h : 0 < s)
variable (angle_O1PO2 : ℝ)
variable (AP CP : ℝ) 

hypothesis h_square : square A B C D
hypothesis h_side_length : side_length A B C D = 10
hypothesis h_point_on_diagonal : P ∈ line A C
hypothesis h_ratio : AP = 3 * CP
hypothesis h_circumcenter_O1 : O1 (circumcircle A B P) = circumcenter A B P
hypothesis h_circumcenter_O2 : O2 (circumcircle C D P) = circumcenter C D P
hypothesis h_right_angle : angle_O1PO2 = 90

theorem find_AP_CP
    (h_sqrt : s = 10)
    (h_AP : AP = s * 3 / (s + s * 3))
    (h_CP : CP = s / (s + s * 3)) :
    (AP = 3 * (5 * Real.sqrt 2 / 2) ∧ CP = 5 * Real.sqrt 2 / 2) :=
by
  sorry

end find_AP_CP_l145_145947


namespace largest_rhombus_in_rectangle_l145_145085

theorem largest_rhombus_in_rectangle (l w : ℝ) :
  ∃ (d1 d2 : ℝ), 
    d1 = sqrt (l^2 + w^2) ∧ 
    d2 = w ∧ 
    ∀ (d1' d2' : ℝ), d1' * d2' ≤ d1 * d2 :=
begin
  sorry
end

end largest_rhombus_in_rectangle_l145_145085


namespace solution_set_of_inequality_l145_145674

theorem solution_set_of_inequality : {x : ℝ | x^2 < 2 * x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l145_145674


namespace books_in_final_category_l145_145226

-- Define the number of initial books
def initial_books : ℕ := 400

-- Define the number of divisions
def num_divisions : ℕ := 4

-- Define the iterative division process
def final_books (initial : ℕ) (divisions : ℕ) : ℕ :=
  initial / (2 ^ divisions)

-- State the theorem
theorem books_in_final_category : final_books initial_books num_divisions = 25 := by
  sorry

end books_in_final_category_l145_145226


namespace probability_of_nonzero_product_of_dice_l145_145298

/-- The probability that the product (a-2)*(b-1)*(c-1) is not zero when three standard dice are tossed. -/
theorem probability_of_nonzero_product_of_dice : 
  (∃ a b c : ℕ, 
    (a ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (b ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (c ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    ((a-2)*(b-1)*(c-1) ≠ 0) ) → 
    ∃ p : ℚ, p = 125 / 216 := 
sorry

end probability_of_nonzero_product_of_dice_l145_145298


namespace slope_angle_at_origin_l145_145533

noncomputable def f (f' : ℝ -> ℝ) (x : ℝ) : ℝ :=
  (1/3) * x^3 + (1/2) * f' 1 * x^2 - f' 2 * x + 3

noncomputable def f_prime (f' : ℝ -> ℝ) (x : ℝ) : ℝ :=
  x^2 + f' 1 * x - f' 2

theorem slope_angle_at_origin (f' : ℝ -> ℝ)
  (h1 : f_prime f' 1 = 0)
  (h2 : f_prime f' 2 = 1) :
  ∃ α : ℝ, tan α = -1 ∧ α = 3 / 4 * Real.pi :=
  sorry

end slope_angle_at_origin_l145_145533


namespace range_of_f_inequality_l145_145848

def f (x : ℝ) : ℝ := 
  if x ≤ -1 then (x + 1) ^ 2 
  else 2 * x + 2

theorem range_of_f_inequality :
  { x : ℝ | f x > 1 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > -1 / 2 } :=
by
  sorry

end range_of_f_inequality_l145_145848


namespace inequality_solution_l145_145010

def inequality_holds (x : ℝ) : Prop :=
  (4 * x^2) / (1 - sqrt(1 + 2 * x))^2 < 2 * x + 9

def domain_condition (x : ℝ) : Prop :=
  (x >= -1 / 2) ∧ (x ≠ 0)

theorem inequality_solution (x : ℝ) :
  domain_condition x → 
  inequality_holds x ↔ (-1 / 2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45 / 8) :=
by sorry

end inequality_solution_l145_145010


namespace equal_segments_l145_145136

open_locale classical

variables {A B C D E U V : Type*}

-- Definitions of points and properties
variables [has_mem A B C D E U V] [has_coe ℝ ℂ] -- Points are variables that can be represented in ℂ
variables (CA CB AB AE : set (set ℂ)) -- segments are sets of ℂ
variables [right_triangle ABC C] [isosceles_right_triangle A B C] 
variables (U V : ℂ) 

-- Conditions
hypothesis (h1 : isosceles_right_triangle ABC)
hypothesis (h2 : CD = CE)
hypothesis (h3 : D ∈ [CA])
hypothesis (h4 : E ∈ [CB])
hypothesis (h5 : U ∈ [AB])
hypothesis (h6 : V ∈ [AB])
hypothesis (h7 : DU ⊥ AE)
hypothesis (h8 : CV ⊥ AE)

-- Theorem
theorem equal_segments : UV = VB :=
sorry

end equal_segments_l145_145136


namespace min_value_l145_145097

theorem min_value (x y : ℝ) (h1 : xy > 0) (h2 : x + 4 * y = 3) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, xy > 0 → x + 4 * y = 3 → (1 / x + 1 / y) ≥ 3 := sorry

end min_value_l145_145097


namespace unique_strictly_increasing_sequence_l145_145398

theorem unique_strictly_increasing_sequence :
  ∃! (a : ℕ → ℕ), 
    (strict_mono a) ∧ 
    (a 1 = 1) ∧ 
    (∀ n : ℕ, 3 * (∑ i in range n, a (i + 1)) = ∑ i in range n, a (n + i + 1)) ∧ 
    (∀ n : ℕ, a n = 2 * n - 1) := sorry

end unique_strictly_increasing_sequence_l145_145398


namespace triangle_inequality_l145_145617

theorem triangle_inequality (ABC : Type) [triangle ABC] (S R r : ℝ) :
  S = area ABC ∧ R = circumradius ABC ∧ r = inradius ABC → S > 2 * R^(1/2) * r^(3/2) :=
by
  sorry

end triangle_inequality_l145_145617


namespace reflect_over_y_axis_matrix_l145_145413

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l145_145413


namespace meet_on_altitude_l145_145941

variable {α : Type*} [LinearOrder α] [Field α]

def midpoint (A B : α × α) : α × α :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

variables (A B C A1 B1 C0 I : α × α)
variables (hC0 : C0 = midpoint A B)
variables (hAA1 : ∃ A2, line_through A A1 = angle_bisector_triangle A B C)
variables (hBB1 : ∃ B2, line_through B B1 = angle_bisector_triangle B A C)
variables (hI : I = incenter A B C)

theorem meet_on_altitude (h : ∃ P, altitude_from C A B = line_through P (C0, I) ∧ altitude_from C A B = line_through P (A1, B1)) :
  ∃ P, P = intersection (line_through C0 I) (line_through A1 B1) :=
sorry

end meet_on_altitude_l145_145941


namespace length_of_equal_sides_of_triangle_l145_145609

-- Define the points on the given curve and the isosceles right triangle condition
def on_curve (A B : ℝ × ℝ) : Prop :=
  ∃ a : ℝ, A = (a, -2 * a ^ 2) ∧ B = (-a, -2 * a ^ 2)

def isosceles_right_triangle_at_origin (A B O : ℝ × ℝ) : Prop :=
  O = (0, 0) ∧ (A.fst = -B.fst ∧ A.snd = B.snd) ∧ 
  (dist O A = dist O B)

-- Main statement proving the length of the equal sides of ΔABO
theorem length_of_equal_sides_of_triangle (A B O : ℝ × ℝ) 
  (h_curve : on_curve A B) 
  (h_iso : isosceles_right_triangle_at_origin A B O) : dist O A = sqrt 5 := 
by 
  sorry

end length_of_equal_sides_of_triangle_l145_145609


namespace quadrilateral_conditions_l145_145618

/-- If the sum of the cosines of the angles of a quadrilateral is zero,
    then it must be a parallelogram, trapezoid, or an inscribed quadrilateral. -/
theorem quadrilateral_conditions (α β γ δ : ℝ) (h_sum : α + β + γ + δ = 2 * Real.pi) 
(h_cos_sum : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) : 
  is_parallelogram α β γ δ ∨ is_trapezoid α β γ δ ∨ is_inscribed α β γ δ := 
sorry

def is_parallelogram (α β γ δ : ℝ) : Prop := sorry
def is_trapezoid (α β γ δ : ℝ) : Prop := sorry
def is_inscribed (α β γ δ : ℝ) : Prop := sorry

end quadrilateral_conditions_l145_145618


namespace smallest_d_for_inequality_l145_145790

theorem smallest_d_for_inequality :
  ∃ (d : ℝ), 0 < d ∧ (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y →
    (real.sqrt(x^2 * y^2) + d * |x^2 - y^2| + x + y ≥ x^2 + y^2)) ∧
    (∀ d' > 0, d' < d → ∃ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧
      real.sqrt(x^2 * y^2) + d' * |x^2 - y^2| + x + y < x^2 + y^2)) :=
  sorry

end smallest_d_for_inequality_l145_145790


namespace part1_part2_l145_145510

def M := {-3, -2, -1, 0, 1, 2}

/-- Part 1: Number of different quadratic functions y = ax^2 + bx + c where a, b, c ∈ M and a ≠ 0 -/
theorem part1 {a b c : ℤ} (h1 : a ≠ 0) (h2 : a ∈ M) (h3 : b ∈ M) (h4 : c ∈ M) : 
  (∃ f : ℤ × ℤ × ℤ → ℤ, ∀ (a b c : ℤ), f (a, b, c) = a * x^2 + b * x + c) ↔ (M.filter (λ x, x ≠ 0)).card * M.card * M.card = 180 := 
sorry

/-- Part 2: Number of quadratic functions y = ax^2 + bx + c with a > 0 -/
theorem part2 {a b c : ℤ} (h1 : a > 0) (h2 : a ∈ M) (h3 : b ∈ M) (h4 : c ∈ M) : 
  (∃ f : ℤ × ℤ × ℤ → ℤ, ∀ (a b c : ℤ), f (a, b, c) = a * x^2 + b * x + c) ↔ (M.filter (λ x, 0 < x)).card * M.card * M.card = 72 := 
sorry

end part1_part2_l145_145510


namespace triangle_side_lengths_l145_145318

theorem triangle_side_lengths 
  (r : ℝ) (CD : ℝ) (DB : ℝ) 
  (h_r : r = 4) 
  (h_CD : CD = 8) 
  (h_DB : DB = 10) :
  ∃ (AB AC : ℝ), AB = 14.5 ∧ AC = 12.5 :=
by
  sorry

end triangle_side_lengths_l145_145318


namespace partition_grid_square_l145_145189

theorem partition_grid_square (n : ℕ) (s : ℕ) (k : ℕ) (h_n : n = 8) (h_s : s = 4) (h_k : k = 4) :
  ∃ (sizes : list ℕ) (counts : list ℕ), 
  sizes = [s, s, s, s] ∧ counts = [4, 4, 4, 4] ∧ 
  (∀ idx, sizes.nth idx = counts.nth idx) := 
by
  -- here will be the proof
  sorry

end partition_grid_square_l145_145189


namespace win_sector_area_l145_145319

theorem win_sector_area (r : ℝ) (P : ℝ) (h0 : r = 8) (h1 : P = 3 / 8) :
    let area_total := Real.pi * r ^ 2
    let area_win := P * area_total
    area_win = 24 * Real.pi :=
by 
  sorry

end win_sector_area_l145_145319


namespace colten_chickens_l145_145998

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l145_145998


namespace alice_profit_l145_145358

def total_bracelets : ℕ := 52
def cost_materials : ℝ := 3.0
def bracelets_given_away : ℕ := 8
def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_sold := total_bracelets - bracelets_given_away in
  let total_revenue := bracelets_sold * price_per_bracelet in
  let profit := total_revenue - cost_materials in
  profit = 8.00 :=
by
  sorry

end alice_profit_l145_145358


namespace max_a_dot_c_l145_145595

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def angle_between (u v : V) : real.angle :=
  real.angle_of u v (inner_product_space.real_inner_product_space_topology.has_continuous_inner)

variables (a b c : V)
variables (angle_ab : angle_between a b = real.pi / 3)
variables (angle_ca_cb : angle_between (c - a) (c - b) = 2 * real.pi / 3)
variables (dist_ab : dist a b = 5)
variables (dist_ca : dist c a = 2 * real.sqrt 3)

theorem max_a_dot_c : 
  let dot := inner_product_space.is_R_or_C.inner in
  ∃ max_val, (∀ c, a.inner c ≤ max_val) ∧ max_val = 24 :=
begin
  sorry
end

end max_a_dot_c_l145_145595


namespace total_readers_l145_145910

theorem total_readers (S L B T : ℕ) (hS : S = 180) (hL : L = 88) (hB : B = 18) (hT : T = S + L - B) : T = 250 :=
by {
  rw [hS, hL, hB] at hT,
  exact hT,
}

end total_readers_l145_145910


namespace delta_x_range_l145_145534

theorem delta_x_range (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + x) (Δx : ℝ) (hΔx_pos : Δx > 0) 
  (h_avg_rate : (f (2 + Δx) - f 2) / Δx ≤ -1) : Δx ∈ set.Ioi (0 : ℝ) :=
by {
  sorry
}

end delta_x_range_l145_145534


namespace algebraic_expression_value_l145_145059

theorem algebraic_expression_value (a b c : ℝ) (h1 : a - b = 2) (h2 : a - c = real.cbrt 7) :
  ((c - b) * ((a - b)^2 + (a - b) * (a - c) + (a - c)^2)) = 1 :=
by
  -- Proof will go here.
  sorry

end algebraic_expression_value_l145_145059


namespace volume_of_TABC_l145_145610

noncomputable def volume_pyramid_TABC : ℝ :=
  let TA : ℝ := 15
  let TB : ℝ := 15
  let TC : ℝ := 5 * Real.sqrt 3
  let area_ABT : ℝ := (1 / 2) * TA * TB
  (1 / 3) * area_ABT * TC

theorem volume_of_TABC :
  volume_pyramid_TABC = 187.5 * Real.sqrt 3 :=
sorry

end volume_of_TABC_l145_145610


namespace average_velocity_instantaneous_velocity_l145_145067

-- Define the equation of motion for a free-falling object
def s (t : ℝ) : ℝ := 5 * t^2

-- State the theorems to prove
theorem average_velocity (Δt : ℝ) : 
  Δt ≠ 0 → (s(2 + Δt) - s 2) / Δt = 5 * Δt + 20 :=
by
  sorry

theorem instantaneous_velocity : 
  (∀ (Δt : ℝ), Δt ≠ 0 → (s(2 + Δt) - s 2) / Δt = 5 * Δt + 20) →
  ∀ ε > 0, ∃ δ > 0, ∀ Δt, 0 < |Δt| ∧ |Δt| < δ → |((s(2 + Δt) - s 2) / Δt) - 20| < ε :=
by
  sorry

end average_velocity_instantaneous_velocity_l145_145067


namespace angle_equality_l145_145560

variables {A B C D E F : Type} [RealField ℝ] {a b : ℝ}

-- Definitions of points and lines involved
def is_right_triangle (A B C : Type) [HasAngle (A B C)] :=
  angle A B C = π / 2

def isosceles_right_triangle (A B C : Type) [HasLength A B a] [HasLength B C a] :=
  is_right_triangle A B C ∧ a = b

def median (B D C : Type) [IsMidpoint D A C] :=
  line B D

def perpendicular (C E D : Type) [IsPerpendicular C E D] :=
  line C E ∩ line D B

theorem angle_equality 
  (hABC : isosceles_right_triangle A B C) 
  (hMed : median B D C) 
  (hPerp : perpendicular C E D ∧ intersects C E A B F) :
  angle B D C = angle A D F :=
sorry

end angle_equality_l145_145560


namespace sphere_volume_after_cut_l145_145343

noncomputable def remaining_sphere_volume (R : ℝ) (h : ℝ) (V_cap : ℝ) : ℝ :=
  (4 / 3) * π * R^3 - V_cap

theorem sphere_volume_after_cut :
  let R := 1 in
  let h := 0.5 in
  let V_cap := (π * h^2 / 3) * (3 * R - h) in
  remaining_sphere_volume R h V_cap = (59 * π) / 48 :=
by
  sorry

end sphere_volume_after_cut_l145_145343


namespace ratio_area_triangles_to_square_l145_145548

theorem ratio_area_triangles_to_square (x : ℝ) :
  let A := (0, x)
  let B := (x, x)
  let C := (x, 0)
  let D := (0, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let area_AMN := 1/2 * ((M.1 - A.1) * (N.2 - A.2) - (M.2 - A.2) * (N.1 - A.1))
  let area_MNP := 1/2 * ((N.1 - M.1) * (P.2 - M.2) - (N.2 - M.2) * (P.1 - M.1))
  let total_area_triangles := area_AMN + area_MNP
  let area_square := x * x
  total_area_triangles / area_square = 1/4 := 
by
  sorry

end ratio_area_triangles_to_square_l145_145548


namespace sum_S_5_l145_145079

open BigOperators

-- Definitions from the conditions
def a (n : ℕ+) : ℚ := 1 / (n * (n + 1))

def S (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), a (k + 1)

-- Statement of the proof
theorem sum_S_5 : S 5 = 5 / 6 := sorry

end sum_S_5_l145_145079


namespace mats_length_l145_145320

open Real

theorem mats_length (r : ℝ) (n : ℤ) (w : ℝ) (y : ℝ) (h₁ : r = 6) (h₂ : n = 8) (h₃ : w = 1):
  y = 6 * sqrt (2 - sqrt 2) :=
sorry

end mats_length_l145_145320


namespace quadratic_inequality_solution_l145_145804

theorem quadratic_inequality_solution (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end quadratic_inequality_solution_l145_145804


namespace similar_triangles_l145_145583

variable {Point : Type} [InnerProductSpace ℝ Point]
variable {Triangle : Type} [AffineSpace Point ℝ]

variables {A0 B0 C0 : Point}
variables {P : Point}

def projection_of_point_on_line (P A B : Point) : Point := sorry -- Implementation detail necessary

def A1 := projection_of_point_on_line P B0 C0
def B1 := projection_of_point_on_line P A0 C0
def C1 := projection_of_point_on_line P A0 B0

def A2 := projection_of_point_on_line P B1 C1
def B2 := projection_of_point_on_line P A1 C1
def C2 := projection_of_point_on_line P A1 B1

def A3 := projection_of_point_on_line P B2 C2
def B3 := projection_of_point_on_line P A2 C2
def C3 := projection_of_point_on_line P A2 B2

theorem similar_triangles (A0 B0 C0 A3 B3 C3 : Point) (hA3 : A3 = projection_of_point_on_line P B2 C2) (hB3 : B3 = projection_of_point_on_line P A2 C2) (hC3 : C3 = projection_of_point_on_line P A2 B2) :
  ∆ A0 B0 C0 ∼ ∆ A3 B3 C3 := sorry

end similar_triangles_l145_145583


namespace min_checkout_counters_l145_145367

variable (n : ℕ)
variable (x y : ℝ)

-- Conditions based on problem statement
axiom cond1 : 40 * y = 20 * x + n
axiom cond2 : 36 * y = 12 * x + n

theorem min_checkout_counters (m : ℕ) (h : 6 * m * y > 6 * x + n) : m ≥ 6 :=
  sorry

end min_checkout_counters_l145_145367


namespace digit_0_in_382_pages_l145_145364

-- Definition for counting occurrences of a digit in page numbers
def count_digit_0_appears (n : ℕ) : ℕ :=
  ((List.range (n + 1)).map (λ m, String.toNat! (m.toString.filter (λ c, c = '0')).length)).sum

-- Problem statement: Given the book has 382 pages, prove the number of times the digit 0 appears in the page numbers is 68.
theorem digit_0_in_382_pages : count_digit_0_appears 382 = 68 := by sorry

end digit_0_in_382_pages_l145_145364


namespace max_possible_salary_l145_145342

-- Definitions based on conditions
def num_players : Nat := 25
def min_salary : Nat := 15000
def total_salary_cap : Nat := 850000

-- Problem statement
theorem max_possible_salary 
  (h1 : ∀ (i : Nat), i < 24 → 15000 ≤ min_salary)
  (h2 : ∑ i in (Finset.range 24), min_salary = 360000) :
  ∃ x : Nat, (360000 + x = 850000) ∧ (x <= total_salary_cap) ∧ (x = 490000) :=
sorry

end max_possible_salary_l145_145342


namespace eightfold_application_l145_145147

def s (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem eightfold_application (θ : ℝ) : s (s (s (s (s (s (s (s θ)))))))) = 30 :=
by sorry

end eightfold_application_l145_145147


namespace count_two_digit_numbers_with_prime_sum_l145_145878

-- Define a predicate that checks if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

-- Define the sum of digits function for a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the condition that checks if a two-digit number has a prime sum of digits
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ is_prime (sum_of_digits n)

-- Claim to prove
theorem count_two_digit_numbers_with_prime_sum : (finset.filter is_valid_number (finset.range 100)).card = 34 :=
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145878


namespace count_good_mappings_l145_145457

noncomputable def good_number_pairs_count (f : Fin 6 → Fin 6) : ℕ :=
  Fintype.card { x : Fin 6 | x + f x = 6 }

def is_good_mapping (f : Fin 6 → Fin 6) : Prop :=
  Bijective f ∧ good_number_pairs_count f = 3

theorem count_good_mappings : 
  Fintype.card { f : Fin 6 → Fin 6 // is_good_mapping f } = 40 := 
by
  sorry

end count_good_mappings_l145_145457


namespace odd_function_def_l145_145096

noncomputable def f (x : ℝ) : ℝ := 
if h : x > 0 then log (x + 1) else - log (1 - x)

theorem odd_function_def {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x) (h_pos : ∀ x, 0 < x → f x = log (x + 1)) :
  ∀ x, x < 0 → f x = - log (1 - x) :=
begin
  assume x hx_neg,
  have hx_pos : -x > 0, from neg_pos.mpr hx_neg,
  calc
  f x = f (-(-x)) : by rw neg_neg
     ... = - f (-x) : by rw h_odd
     ... = - log (1 - x) : by rw h_pos (-x) hx_pos,
end

end odd_function_def_l145_145096


namespace angle_x_value_l145_145918

theorem angle_x_value 
  (AB CD : Prop) -- AB and CD are straight lines
  (angle_AXB angle_AXZ angle_BXY angle_CYX : ℝ) -- Given angles in the problem
  (h1 : AB) (h2 : CD)
  (h3 : angle_AXB = 180)
  (h4 : angle_AXZ = 60)
  (h5 : angle_BXY = 50)
  (h6 : angle_CYX = 120) : 
  ∃ x : ℝ, x = 50 := by
sorry

end angle_x_value_l145_145918


namespace solve_inequality_l145_145020

theorem solve_inequality (a x : ℝ) :
  (a - x) * (x - 1) < 0 ↔
  (a > 1 ∧ (x < 1 ∨ x > a)) ∨
  (a < 1 ∧ (x < a ∨ x > 1)) ∨
  (a = 1 ∧ x ≠ 1) :=
by
  sorry

end solve_inequality_l145_145020


namespace vec_op_not_comm_l145_145785

open Real

-- Define the operation ⊙
def vec_op (a b: ℝ × ℝ) : ℝ :=
  (a.1 * b.2) - (a.2 * b.1)

-- Define a predicate to check if two vectors are collinear
def collinear (a b: ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the proof theorem
theorem vec_op_not_comm (a b: ℝ × ℝ) : vec_op a b ≠ vec_op b a :=
by
  -- The contents of the proof will go here. Insert 'sorry' to skip.
  sorry

end vec_op_not_comm_l145_145785


namespace math_equivalence_l145_145896

theorem math_equivalence (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) (hbc : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c := 
by 
  sorry

end math_equivalence_l145_145896


namespace dark_chocolate_bars_after_change_l145_145253

theorem dark_chocolate_bars_after_change :
  ∀ (W0 D0 W1 D1 : ℕ),
  (W0 : ℝ) / (D0 : ℝ) = 4 / 3 → 
  W1 = 1.5 * W0 →
  W1 = 30 →
  D1 = 0.75 * D0 →
  D1 = 11 :=
by
  intros W0 D0 W1 D1 h1 h2 h3 h4
  sorry

end dark_chocolate_bars_after_change_l145_145253


namespace monotonic_increasing_interval_l145_145662

noncomputable def f (x : ℝ) : ℝ := log 2 (x ^ 2 - 4)

theorem monotonic_increasing_interval : (∀ x : ℝ, x > 2 → monotone_on f (Ioi 2)) :=
sorry

end monotonic_increasing_interval_l145_145662


namespace cos_squared_value_l145_145465

theorem cos_squared_value (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 :=
sorry

end cos_squared_value_l145_145465


namespace count_two_digit_numbers_with_prime_sum_l145_145880

-- Define a predicate that checks if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

-- Define the sum of digits function for a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the condition that checks if a two-digit number has a prime sum of digits
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ is_prime (sum_of_digits n)

-- Claim to prove
theorem count_two_digit_numbers_with_prime_sum : (finset.filter is_valid_number (finset.range 100)).card = 34 :=
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145880


namespace triangle_area_l145_145473

/-- The function f(x) = x^2 - 5x - 6 --/
def f (x : ℝ) : ℝ := x^2 - 5 * x - 6

/-- Coordinates of Intersection Points --
A = (-1, 0)
B = (6, 0)
C = (0, -6) --/

theorem triangle_area :
  ∃ A B C : ℝ × ℝ, A = (-1, 0) ∧ B = (6, 0) ∧ C = (0, -6) ∧
  let base := ∥(A.1 - B.1)∥,
      height := ∥C.2∥ in
  (1 / 2) * base * height = 21 :=
by
  sorry

end triangle_area_l145_145473


namespace find_d_l145_145539

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)  
  (h1 : α = c) 
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : α + d + β + γ = 180) :
  d = 42 :=
by
  sorry

end find_d_l145_145539


namespace angle_bisector_inequality_part1_angle_bisector_inequality_part2_l145_145643

-- Define the theorem for the first inequality
theorem angle_bisector_inequality_part1 :
  ∀ {A B C A' B' C' D E F D' E' F' : Point},
    (angle_bisectors_intersect_circumcircle A B C A' B' C') →
    (angle_bisectors_intersect_sides A B C D E F) →
    (angle_bisectors_intersect_triangle_sides A' B' C' D' E' F') →
    (orthocenter_config A B C A' D D' E B' E' F C' F') →
  (A'D / AD' + B'E / BE' + C'F / CF' ≥ 3) := sorry

-- Define the theorem for the second inequality
theorem angle_bisector_inequality_part2 :
  ∀ {A B C A' B' C' D E F D' E' F' : Point},
    (angle_bisectors_intersect_circumcircle A B C A' B' C') →
    (angle_bisectors_intersect_sides A B C D E F) →
    (angle_bisectors_intersect_triangle_sides A' B' C' D' E' F') →
    (orthocenter_config A B C A' D D' E B' E' F C' F') →
  (AD' / A'D + BE' / B'E + CF' / C'F ≥ 3) := sorry

end angle_bisector_inequality_part1_angle_bisector_inequality_part2_l145_145643


namespace relationship_abc_l145_145968

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end relationship_abc_l145_145968


namespace sequence_general_term_and_sum_bound_l145_145037

theorem sequence_general_term_and_sum_bound (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n : ℕ, n > 0 → S n + 2 = 2 * a n)
  (h2 : ∀ n : ℕ, n > 0 → b n = 2 * Real.log2 (a n))
  (h3 : ∀ n : ℕ, T n = (∑ i in Finset.range n, 1 / (b i * b (i + 1)))) :
  (∀ n : ℕ, n > 0 → a n = 2^n) ∧ (∀ n : ℕ, n > 0 → T n < 1 / 4) := by
  sorry

end sequence_general_term_and_sum_bound_l145_145037


namespace min_value_arith_prog_sum_l145_145039

noncomputable def arithmetic_progression_sum (x y : ℝ) (n : ℕ) : ℝ :=
  (x + 2 * y + 1) * 3^n + (x - y - 4)

theorem min_value_arith_prog_sum (x y : ℝ)
  (hx : x > 0) (hy : y > 0)
  (h_sum : ∀ n, arithmetic_progression_sum x y n = (x + 2 * y + 1) * 3^n + (x - y - 4)) :
  (∀ x y, 2 * x + y = 3 → 1/x + 2/y ≥ 8/3) :=
by sorry

end min_value_arith_prog_sum_l145_145039


namespace reflect_y_axis_matrix_l145_145420

theorem reflect_y_axis_matrix : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, (
    (∀ v : (Fin 2 → ℝ), v = ![1, 0] → A.mulVec v = ![-1, 0]) ∧ 
    (∀ v : (Fin 2 → ℝ), v = ![0, 1] → A.mulVec v = ![0, 1])
  ) ∧ A = ![![-1, 0], ![0, 1]] :=
begin
  sorry
end

end reflect_y_axis_matrix_l145_145420


namespace total_stamps_correct_l145_145977

def current_stamps : Nat := 40
def percent_increase : Nat := 20
def additional_stamps : Nat := (current_stamps * percent_increase) / 100
def total_stamps : Nat := current_stamps + additional_stamps

theorem total_stamps_correct : total_stamps = 48 := by
  have h1 : additional_stamps = 8 := by
    rw [additional_stamps]
    norm_num
  rw [total_stamps, h1]
  norm_num
  sorry

end total_stamps_correct_l145_145977


namespace union_cardinality_l145_145461

open Set

theorem union_cardinality {A B : Set ℕ} (hA : A = {1, 3, 5}) (hB : B = {2, 3}) :
  Finset.card (Finset.union (Finset.of_set A) (Finset.of_set B)) = 4 := by
  sorry

end union_cardinality_l145_145461


namespace no_such_alpha_exists_l145_145387

open real

theorem no_such_alpha_exists :
  ¬ ∃ (α : ℝ), irrational (cos α) ∧ rational (cos (2 * α)) ∧ rational (cos (3 * α)) ∧ rational (cos (4 * α)) ∧ rational (cos (5 * α)) :=
by
  sorry

end no_such_alpha_exists_l145_145387


namespace part_one_part_two_l145_145505

open Real

variable {a b m : ℝ}

theorem part_one (m : ℝ) : 
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  in Δ > 0 := 
by
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  show Δ > 0 from by {
    simp[Δ],
    linarith
  }

theorem part_two (a b m : ℝ) (h1 : a + b = 2 * m + 1) (h2 : a * b = m^2 + m) (h3 : (2*a + b) * (a + 2*b) = 20) : 
  m = -2 ∨ m = 1 :=
by
  have h4: 9*m^2 + 9*m + 2 = 20 := by {
    rw [h1, h2, ← add_assoc, ← add_assoc],
    linarith
  }
  have h5 := solve_quadratic 9 9 (-18) h4
  cases h5 with h5_left h5_right
  exact or.inl h5_left
  exact or.inr h5_right

end part_one_part_two_l145_145505


namespace m_range_l145_145834

noncomputable def f (x : ℝ) : ℝ := - (1/2) * x^2 + 6 * x - 8 * Real.log x

def critical_points (x : ℝ) : ℝ := - (x - 2) * (x - 4) / x

def is_not_monotonic (m : ℝ) : Prop := ∃ (x : ℝ), m < x ∧ x < m + 1 ∧ critical_points x = 0

theorem m_range (m : ℝ) : is_not_monotonic m ↔ m ∈ set.Ioo 1 2 ∪ set.Ioo 3 4 :=
by
  sorry

end m_range_l145_145834


namespace find_ax5_by5_l145_145529

variable (a b x y : ℝ)

theorem find_ax5_by5 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_by5_l145_145529


namespace range_of_m_l145_145854
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1))^2

noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

theorem range_of_m {x : ℝ} (m : ℝ) (h1 : 1 / 16 ≤ x) (h2 : x ≤ 1 / 4) 
  (h3 : ∀ (x : ℝ), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)): 
  -1 < m ∧ m < 5 / 4 :=
sorry

end range_of_m_l145_145854


namespace range_of_g_area_of_triangle_ABC_l145_145479

-- Part (1)
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
  0 ≤ (Real.sin (2 * x + π / 3) + Real.sqrt 3 / 2) ∧ (Real.sin (2 * x + π / 3) + Real.sqrt 3 / 2) ≤ (Real.sqrt 3 / 2 + 1) :=
by sorry

-- Part (2)
theorem area_of_triangle_ABC (a b c : ℝ) (A : ℝ) :
  A = π / 3 ∧ a = 4 ∧ (b + c = 5) ∧ Real.sin A = Real.sqrt 3 / 2 →
  let area := 1 / 2 * b * c * Real.sin A in
  area = 9 * Real.sqrt 3 / 4 :=
by sorry

end range_of_g_area_of_triangle_ABC_l145_145479


namespace fixed_point_l145_145656

def passes_through_fixed_point (a : ℝ) (x y : ℝ) : Prop :=
  y = 3 + a^(x-1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  passes_through_fixed_point a 1 4 :=
begin
  sorry
end

end fixed_point_l145_145656


namespace find_radius_larger_ball_l145_145254

noncomputable def volume_of_ball (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem find_radius_larger_ball :
  let volume_small := volume_of_ball 2
  let total_volume_small := 10 * volume_small
  let volume_smallest := volume_of_ball 1
  let total_volume_smallest := 2 * volume_smallest
  ∃ r : ℝ, (volume_of_ball r = total_volume_small - total_volume_smallest) ∧ r = Real.cbrt 78 :=
by
  sorry

end find_radius_larger_ball_l145_145254


namespace AF_leq_one_third_AB_l145_145559

theorem AF_leq_one_third_AB 
  (A B C D E F M : Point) 
  (h_triangle_ABC : IsTriangle A B C)
  (h_AB_eq_AC : AB = AC)
  (h_CD_perp_AB : Perpendicular CD AB D)
  (h_M_midpoint_CD : Midpoint M CD)
  (h_AE_perp_BM : Perpendicular AE BM E)
  (h_AF_perp_CE : Perpendicular AF CE F) : 
  AF ≤ (1 / 3) * AB := 
by
  sorry

end AF_leq_one_third_AB_l145_145559


namespace extra_fee_count_l145_145665

-- Defining the envelopes and their dimensions
structure Envelope where
  length : ℕ
  height : ℕ

def E : Envelope := { length := 7, height := 5 }
def F : Envelope := { length := 10, height := 4 }
def G : Envelope := { length := 5, height := 5 }
def H : Envelope := { length := 14, height := 5 }

-- Function to calculate length to height ratio
def ratio (env : Envelope) : ℚ := env.length / env.height

-- Conditions for requiring an additional fee
def requires_extra_fee (env : Envelope) : Prop := ratio env < 1.5 ∨ ratio env > 2.8

-- List of envelopes
def envelopes := [E, F, G, H]

-- Calculating how many envelopes require the additional fee
def count_extra_fee : ℕ := (envelopes.filter requires_extra_fee).length

-- Final statement: Prove that the count is 2
theorem extra_fee_count : count_extra_fee = 2 :=
by
  sorry

end extra_fee_count_l145_145665


namespace weierstrass_limit_l145_145615

theorem weierstrass_limit (a_n : ℕ → ℝ) (M : ℝ) :
  (∀ n m, n ≤ m → a_n n ≤ a_n m) → 
  (∀ n, a_n n ≤ M ) → 
  ∃ c, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n n - c| < ε :=
by
  sorry

end weierstrass_limit_l145_145615


namespace largest_alpha_exists_triangle_l145_145942

-- Definitions and conditions
variable (A B C G : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited G]
variable [HasCentroid A B C G]

-- Statement
theorem largest_alpha_exists_triangle 
  (∀ (ABC : Triangle) (G : Centroid ABC), 
    ∃ (α α₀ : ℝ), α = α₀ ∧ (α₀ = arcsin (sqrt 3 / 3)) ∧ 
    at_least_three_angles_geq_alpha ABC G α₀) : 
  ∃ (α : ℝ), α = arcsin (sqrt 3 / 3) :=
sorry

end largest_alpha_exists_triangle_l145_145942


namespace cos_graph_symmetry_l145_145073

noncomputable def is_symmetric_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
∀ x : ℝ, f (2 * p.1 - x) = f x

theorem cos_graph_symmetry {φ : ℝ} 
  (h : ∃ x : ℝ, y = sin (2 * x + φ) ∧ y = 1)
  : is_symmetric_point (λ x, cos (2 * x + φ)) (π / 6, 0) :=
sorry

end cos_graph_symmetry_l145_145073


namespace induction_example_l145_145221

theorem induction_example (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end induction_example_l145_145221


namespace animals_saved_l145_145314

noncomputable def initial_counts := ⟨20, 10, 14, 8, 12, 6, 15⟩ -- (sheep, cows, dogs, pigs, chickens, rabbits, ducks)

theorem animals_saved :
  ∀ (drowned_sheep: ℕ) (drowned_cows: ℕ) (drowned_dogs: ℕ) (drowned_pigs: ℕ) (drowned_chickens: ℕ) (drowned_rabbits: ℕ) (saved_ducks: ℕ),
    drowned_sheep = 6 ∧
    drowned_cows = 4 ∧
    drowned_dogs = 3 ∧
    drowned_pigs = 2 ∧
    drowned_chickens = 2 ∧
    drowned_rabbits = 1 ∧
    saved_ducks = 10 → 
    initial_counts.1 - drowned_sheep = 14 ∧
    initial_counts.2 - drowned_cows = 6 ∧
    initial_counts.3 - (initial_counts.3 - drowned_dogs) = 11 ∧
    initial_counts.4 - drowned_pigs = 6 ∧
    initial_counts.5 - drowned_chickens = 10 ∧
    initial_counts.6 - drowned_rabbits = 5 ∧
    initial_counts.7 - 5 = saved_ducks :=
by sorry

end animals_saved_l145_145314


namespace max_value_f_l145_145470

def f (x : ℝ) : ℝ := min (-x + 6) (-2 * x^2 + 4 * x + 6)

theorem max_value_f : ∃ x : ℝ, f x = 6 :=
sorry

end max_value_f_l145_145470


namespace reflect_y_axis_matrix_l145_145421

theorem reflect_y_axis_matrix : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, (
    (∀ v : (Fin 2 → ℝ), v = ![1, 0] → A.mulVec v = ![-1, 0]) ∧ 
    (∀ v : (Fin 2 → ℝ), v = ![0, 1] → A.mulVec v = ![0, 1])
  ) ∧ A = ![![-1, 0], ![0, 1]] :=
begin
  sorry
end

end reflect_y_axis_matrix_l145_145421


namespace circle_equation_correct_l145_145164

theorem circle_equation_correct :
  (∃ M : ℝ × ℝ, M.1 * 2 + M.2 - 1 = 0 ∧
                (M.1 - 3)^2 + (M.2 - 0)^2 = 5 ∧ 
                (M.1 - 0)^2 + (M.2 - 1)^2 = 5) →
  ∃ h k r : ℝ, (h = 1) ∧ (k = -1) ∧ (r = sqrt 5) ∧ 
               (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 5) :=
begin
  sorry
end

end circle_equation_correct_l145_145164


namespace angle_A1C1O_eq_30_l145_145926

-- Define a structure for a triangle
structure Triangle (A B C : Type) :=
  (A : A)
  (B : B)
  (C : C)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

noncomputable def triangle_ABC : Triangle ℝ := {
  A := 120, -- Using degrees
  B := sorry,
  C := sorry,
  angle_A := 120,
  angle_B := sorry,
  angle_C := sorry
}

-- Define the angle bisectors and point of intersection
def angle_bisector_A : sorry := sorry
def angle_bisector_B : sorry := sorry
def angle_bisector_C : sorry := sorry
def O : sorry := sorry

-- Define the points A1 B1 C1 on the bisectors
def A1 : sorry := sorry
def B1 : sorry := sorry
def C1 : sorry := sorry

-- Lean statement for the given proof problem
theorem angle_A1C1O_eq_30 :
  ∠ A1 C1 O = 30 :=
sorry

end angle_A1C1O_eq_30_l145_145926


namespace number_of_solutions_of_trig_eq_l145_145518

open Real

/-- The number of values of θ in the interval 0 < θ ≤ 4π that satisfy the equation 
    2 + 4 * sin (2 * θ) - 3 * cos (4 * θ) + 2 * tan(θ) = 0 is 16. -/
theorem number_of_solutions_of_trig_eq : 
  (set_of (λ θ : ℝ, 0 < θ ∧ θ ≤ 4 * π ∧ 2 + 4 * sin (2 * θ) - 3 * cos (4 * θ) + 2 * tan θ = 0)).finite.to_finset.card = 16 :=
sorry

end number_of_solutions_of_trig_eq_l145_145518


namespace quadratic_complete_square_l145_145699

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 + 2 * x + 3) = ((x + 1)^2 + 2) :=
by
  intro x
  sorry

end quadratic_complete_square_l145_145699


namespace profit_is_eight_dollars_l145_145353

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l145_145353


namespace rooks_on_chessboard_l145_145116

theorem rooks_on_chessboard : 
  (∑ s in {(s : Finset (Fin (8 !))) | ∀ i j, i ≠ j → i < |s| → j < |s| → s i ≠ s j}, 1) = 8! :=
sorry

end rooks_on_chessboard_l145_145116


namespace geometric_sequence_a3_l145_145909

theorem geometric_sequence_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 1)
  (h5 : a 5 = 4)
  (geo_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l145_145909


namespace hyperbolic_derivative_theorem_l145_145401

noncomputable def hyperbolic_derivative (x : ℝ) : ℝ :=
  let y := (2 / 3) * Real.coth x - (Real.cosh x) / (3 * (Real.sinh x)^3)
  deriv y

theorem hyperbolic_derivative_theorem (x : ℝ) :
  hyperbolic_derivative x = 1 / (Real.sinh x)^4 := 
by
  sorry

end hyperbolic_derivative_theorem_l145_145401


namespace angle_mnp_is_120_l145_145778

-- Definitions based on the conditions
variable (α β γ : ℝ)  -- angles of the triangle DEF
variable (θ : ℝ)      -- angle MNP
variable (in_DEFEF : Prop) -- Omega is the incircle of DEF and circumcircle of MNP
variable (M := M_on_EF : Prop) -- M lies on EF
variable (N := N_on_DF : Prop) -- N lies on DF
variable (P := P_on_DE : Prop) -- P lies on DE

-- Given conditions
axiom angle_d : α = 50   -- Angle D
axiom angle_e : β = 70   -- Angle E
axiom angle_f : γ = 60   -- Angle F
axiom sum_angles : α + β + γ = 180 -- Sum of angles in triangle DEF
axiom incircle_circumcircle : in_DEFEF  -- Omega is incircle of DEF and circumcircle of MNP

-- The goal is to prove that angle MNP equals 120 degrees.
noncomputable def measure_angle_mnp (α β γ θ : ℝ) (in_DEFEF : Prop) (M_on_EF : Prop)
  (N_on_DF : Prop) (P_on_DE : Prop) : Prop :=
  θ = α + β

theorem angle_mnp_is_120 (angle_d : α = 50) (angle_e : β = 70) (angle_f : γ = 60) 
  (sum_angles : α + β + γ = 180) (incircle_circumcircle : in_DEFEF) 
  (M_on_EF : M) (N_on_DF : N) (P_on_DE : P) : measure_angle_mnp α β γ θ in_DEFEF M_on_EF N_on_DF P_on_DE :=
sorry

end angle_mnp_is_120_l145_145778


namespace count_two_digit_numbers_with_prime_sum_l145_145883

-- Helper predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m < n → n % m ≠ 0

-- Helper function to get the sum of digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  tens + units

-- Define the set of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the set of two-digit numbers with prime digit sum
def two_digit_numbers_with_prime_sum :=
  {n : ℕ | n ∈ two_digit_numbers ∧ is_prime (sum_of_digits n)}

-- The main theorem
theorem count_two_digit_numbers_with_prime_sum : 
  {n : ℕ | n ∈ two_digit_numbers_with_prime_sum }.to_finset.card = 33 :=
by
  -- Placeholder for the actual proof
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145883


namespace max_excellent_squares_l145_145821

theorem max_excellent_squares (n : ℕ) (h : 2004 < n) : 
  ∃ maxExcellentSquares : ℕ, maxExcellentSquares = n * (n - 2004) := 
begin
  use n * (n - 2004),
  sorry
end

end max_excellent_squares_l145_145821


namespace distance_from_S_to_PQR_eq_l145_145219

noncomputable def distance_S_to_face_PQR (S P Q R : ℝ³) (h_perpendicular1 : is_perpendicular S P Q) (h_perpendicular2 : is_perpendicular S Q R) (h_perpendicular3 : is_perpendicular S R P)
  (h_SP : dist S P = 10) (h_SQ : dist S Q = 10) (h_SR : dist S R = 8) : ℝ :=
  (40 * real.sqrt 19) / 57

-- here is the statement to prove the problem
theorem distance_from_S_to_PQR_eq 
  (S P Q R : ℝ³) (h_perpendicular1 : is_perpendicular S P Q) (h_perpendicular2 : is_perpendicular S Q R) (h_perpendicular3 : is_perpendicular S R P)
  (h_SP : dist S P = 10) (h_SQ : dist S Q = 10) (h_SR : dist S R = 8) : 
  distance_S_to_face_PQR S P Q R h_perpendicular1 h_perpendicular2 h_perpendicular3 h_SP h_SQ h_SR = (40 * real.sqrt 19) / 57 :=
begin
  sorry
end

end distance_from_S_to_PQR_eq_l145_145219


namespace focus_of_parabola_l145_145404

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  let f := k + 1 / (4 * a)
  in (h, f)

theorem focus_of_parabola : parabola_focus 2 1 0 = (1, 1 / 8) :=
  by sorry

end focus_of_parabola_l145_145404


namespace reflectionY_matrix_correct_l145_145419

-- Define the basis vectors e₁ and e₂
def e1 : Vector := ⟨1, 0⟩
def e2 : Vector := ⟨0, 1⟩

-- Define the transformation that reflects over the y-axis
def reflectY : Vector → Vector 
| ⟨x, y⟩ => ⟨-x, y⟩

-- Conditions given in the problem
lemma reflectY_e1 : reflectY e1 = ⟨-1, 0⟩ := sorry
lemma reflectY_e2 : reflectY e2 = ⟨0, 1⟩ := sorry

-- The goal is to find the transformation matrix for reflection over the y-axis
def reflectionMatrixY : Matrix 2 2 ℝ :=
  Matrix.of_vec ([-1, 0, 0, 1])

theorem reflectionY_matrix_correct :
  ∀ (v : Vector), reflectY v = reflectionMatrixY.mul_vec v := sorry

end reflectionY_matrix_correct_l145_145419


namespace cyclic_sum_inequality_l145_145961

theorem cyclic_sum_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 :=
by
  -- TODO: Provide proof here
  sorry

end cyclic_sum_inequality_l145_145961


namespace domain_of_f_log2x_l145_145839

def domain_f_2x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

theorem domain_of_f_log2x :
  ∀ f : (ℝ → ℝ), (∀ x, domain_f_2x x → (0 ≤ x ∧ x ≤ 1)) →
  (∀ x, 2 ≤ x ∧ x ≤ 4 ↔ ∃ y, 1 ≤ log (2:ℝ) y ∧ log (2:ℝ) y ≤ 2 ∧ x = y) :=
by
  sorry

end domain_of_f_log2x_l145_145839


namespace police_coverage_l145_145553

-- Given Intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K
deriving DecidableEq

open Intersection

-- Given streets
def streets : List (List Intersection) := [
  [A, B, C, D],  -- Horizontal 1
  [E, F, G],     -- Horizontal 2
  [H, I, J, K],  -- Horizontal 3
  [A, E, H],     -- Vertical 1
  [B, F, I],     -- Vertical 2
  [D, G, J],     -- Vertical 3
  [H, F, C],     -- Diagonal 1
  [C, G, K]      -- Diagonal 2
]

-- Define the proof problem
theorem police_coverage (P1 P2 P3 : Intersection) (c1 : List Intersection) (c2 : List Intersection) (c3 : List Intersection) :
  c1 = [B, G, H] →
  (∀ s ∈ streets, ∃ p ∈ [P1, P2, P3], p ∈ s) :=
by {
  intros,
  sorry
}

end police_coverage_l145_145553


namespace man_l145_145301

theorem man's_speed_against_current :
  ∀ (v : ℝ), 
  (v + 2.5 = 15) → 
  (v - 2.5 = 10) :=
by {
  intros v hv,
  rwa [← sub_eq_iff_eq_add] at hv,
  exact eq_sub_of_add_eq hv,
}

end man_l145_145301


namespace two_digit_prime_sum_count_l145_145889

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_prime_sum_count : 
  let prime_sums := {2, 3, 5, 7, 11, 13, 17}
  ∑ k in (Finset.filter (λ n, is_prime (sum_of_digits n)) (Finset.filter two_digit (Finset.range 100))), 1 = 31 := 
by
  sorry

end two_digit_prime_sum_count_l145_145889


namespace reflection_y_axis_l145_145410

open Matrix

def reflection_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

theorem reflection_y_axis (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M ⬝ ![![1, 0], ![0, 1]]) = reflection_y_axis_matrix :=
by sorry

end reflection_y_axis_l145_145410


namespace bus_stops_for_6_point_98_min_per_hour_l145_145797

noncomputable def minutes_stopped_per_hour 
  (speed_without_stoppages : ℝ) 
  (speed_with_stoppages : ℝ) : ℝ :=
  let distance_loss := speed_without_stoppages - speed_with_stoppages
  distance_loss / (speed_without_stoppages / 60)

theorem bus_stops_for_6_point_98_min_per_hour :
  minutes_stopped_per_hour 86 76 ≈ 6.98 :=
by
  sorry

end bus_stops_for_6_point_98_min_per_hour_l145_145797


namespace pipe_q_drain_portion_l145_145276

-- Define the times it takes for each pipe in terms of T_q
def T_p (T_q : ℝ) := (3/4) * T_q
def T_r (T_q : ℝ) := 3 * T_p T_q
def T_s (T_q : ℝ) := (1/2) * T_r T_q
def T_t (T_q : ℝ) := (1/3) * T_s T_q

-- Define the rates for each pipe
def R_q (T_q : ℝ) := 1 / T_q
def R_p (T_q : ℝ) := R_q T_q / (3/4)
def R_r (T_q : ℝ) := R_q T_q / (9/4)
def R_s (T_q : ℝ) := R_q T_q / (9/8)
def R_t (T_q : ℝ) := R_q T_q / (3/8)

-- Define the total rate when all pipes are operating simultaneously
def total_rate (T_q : ℝ) := R_q T_q + R_p T_q + R_r T_q + R_s T_q + R_t T_q

-- The portion of the liquid that pipe q drains
def portion_q (T_q : ℝ) := R_q T_q / total_rate T_q

-- State the theorem to prove that q drains 3/19 of the liquid
theorem pipe_q_drain_portion (T_q : ℝ) : portion_q T_q = 3 / 19 := by
  sorry

end pipe_q_drain_portion_l145_145276


namespace linear_function_parallel_through_point_l145_145738

theorem linear_function_parallel_through_point 
(k b : ℝ) 
(h_parallel : ∀ x: ℝ, y = kx + b → y = 2x + 1) 
(h_point : ∀ x= -3, y=4 →  ∃ b, y = kx + b) : 
(λ x, 2x + 10) := 
by 
have h1 : k = 2 := sorry 
have h2 : b = 10 := sorry
use h1,
use h2,
triv

end linear_function_parallel_through_point_l145_145738


namespace investment_time_R_l145_145252

theorem investment_time_R (x t : ℝ) 
  (h1 : 7 * 5 * x / (5 * 7 * x) = 7 / 9)
  (h2 : 3 * t * x / (5 * 7 * x) = 4 / 9) : 
  t = 140 / 27 :=
by
  -- Placeholder for the proof, which is not required in this step.
  sorry

end investment_time_R_l145_145252


namespace linear_independent_exists_k_l145_145008

open Function

noncomputable def P (i n : ℕ) (v : Fin n → ℝ) : Fin n → ℝ :=
  λ j, if j = i then 0 else v j

theorem linear_independent_exists_k {n : ℕ} (hn : n ≥ 2)
    (v : Fin (n - 1) → Fin n → ℝ)
    (h_independent : LinearIndependent ℝ v) :
    ∃ k : Fin n, LinearIndependent ℝ (λ i, P k.val n (v i)) := sorry

end linear_independent_exists_k_l145_145008


namespace sum_of_first_6n_integers_l145_145261

theorem sum_of_first_6n_integers (n : ℕ) :
  (∑ i in finset.range (6 * n + 1), i) = 666 :=
by
  -- Declare the conditions
  have h1 : (∑ i in finset.range (5 * n + 1), i) = (∑ i in finset.range (2 * n + 1), i) + 425 :=
    sorry -- needs complex setup and verification
  -- translate the sum using known formula
  have h2 : (5 * n * (5 * n + 1)) / 2 = ((2 * n * (2 * n + 1)) / 2) + 425 :=
    sorry -- needs simplification

  -- Derive n = 6 from conditions
  have h3 : n = 6 :=
    sorry -- needs solving the equation

  -- Calculate and prove the sum of the first 6n positive integers
  calc
    (∑ i in finset.range (6 * n + 1), i)
        = (6 * 6 * (6 * 6 + 1)) / 2 : 
        by {
          -- use the formula for sum of the first k positive integers
          sorry 
        }
    ... = 666 :
        by {
          rw h3,
          -- compute the exact sum
          sorry
        }

end sum_of_first_6n_integers_l145_145261


namespace tailor_cut_difference_l145_145752

def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

theorem tailor_cut_difference : skirt_cut - pants_cut = 0.25 :=
by
  sorry

end tailor_cut_difference_l145_145752


namespace trigonometric_identity_l145_145525

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 2 + α) = - 1 / 3 := 
by
  sorry

end trigonometric_identity_l145_145525


namespace john_eggs_per_week_l145_145574

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l145_145574


namespace glue_drops_3x3x3_cube_glue_drops_10x10x10_cube_l145_145707

-- Problem a) Lean 4 Statement
theorem glue_drops_3x3x3_cube : 
  (number_of_drops (3: ℕ) (3: ℕ) (3: ℕ) = 54) := 
sorry

-- Problem b) Lean 4 Statement
theorem glue_drops_10x10x10_cube : 
  (number_of_drops (10: ℕ) (10: ℕ) (10: ℕ) = 2700) := 
sorry

end glue_drops_3x3x3_cube_glue_drops_10x10x10_cube_l145_145707


namespace determine_function_l145_145485

theorem determine_function (f : ℝ → ℝ)
    (h1 : f 1 = 0)
    (h2 : ∀ x y : ℝ, |f x - f y| = |x - y|) :
    (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end determine_function_l145_145485


namespace length_CF_l145_145137

/-- Given a point C not on line AF, a point D on line AF such that CD is perpendicular to AF,
a point B on line CF such that AB is perpendicular to CF, where AB = 6, CD = 10, and AF = 3,
prove that the length of CF is 5. -/
theorem length_CF {C D B : Point} {A F : Point} 
  (h1 : ¬ same_line C F A F) 
  (h2 : on_line D A F) 
  (h3 : perpendicular (line C F) (line A B)) 
  (h4 : AB = 6) 
  (h5 : CD = 10) 
  (h6 : AF = 3) 
  : length_CF = 5 :=
by sorry

end length_CF_l145_145137


namespace cos_B_geq_three_fourths_angle_B_eq_pi_over_six_l145_145122

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Condition: ac = 2b^2
axiom ac_eq_2b2 : a * c = 2 * b^2

-- Proof problem part I: Prove that cos B ≥ 3/4 given ac = 2b^2.
theorem cos_B_geq_three_fourths (h₁ : a * c = 2 * b^2) : real.cos B ≥ 3 / 4 :=
by
  sorry

-- Condition: cos(A - C) + cos B = 1
axiom cos_A_minus_C_plus_cos_B_eq_1 : real.cos (A - C) + real.cos B = 1

-- Proof problem part II: Determine the size of angle B given ac = 2b^2 and cos(A - C) + cos B = 1.
theorem angle_B_eq_pi_over_six (h₁ : a * c = 2 * b^2) (h₂ : real.cos (A - C) + real.cos B = 1) : B = real.pi / 6 :=
by
  sorry

end cos_B_geq_three_fourths_angle_B_eq_pi_over_six_l145_145122


namespace machine_B_makes_more_l145_145971

theorem machine_B_makes_more {rate_A rate_B total_B time : ℕ} 
  (h_rate_A : rate_A = 8) 
  (h_rate_B : rate_B = 10) 
  (h_total_B : total_B = 100) 
  (h_time : time = total_B / rate_B) : 
  let prod_A := rate_A * time in
  let prod_B := total_B in
  prod_B - prod_A = 20 :=
by 
  sorry

end machine_B_makes_more_l145_145971


namespace prove_tangents_intersect_on_circle_l145_145959

noncomputable def problem_statement (AB : ℝ) (BC : ℝ) (AD : ℝ) (CD : ℝ) 
  (quad : convex_quadrilateral ABCD) (incircle_ABC : circle ABC) 
  (incircle_ADC : circle ADC) (circle_omega : circle (angle ABC)) 
  (omega_tangent_AD_CD : tangent_to_extensions AD CD circle_omega) 
  (AB_ne_BC : AB ≠ BC) : Prop :=
  ∃ P Q, 
    common_external_tangents incircle_ABC incircle_ADC 
    ∧ point_on_circle (extension_tangent_intersection P Q) circle_omega

-- Problem needs the proof part denoted by sorry
theorem prove_tangents_intersect_on_circle 
  (AB : ℝ) (BC : ℝ) (AD : ℝ) (CD : ℝ) 
  (quad : convex_quadrilateral ABCD) 
  (incircle_ABC : circle ABC) 
  (incircle_ADC : circle ADC) 
  (circle_omega : circle (angle ABC)) 
  (omega_tangent_AD_CD : tangent_to_extensions AD CD circle_omega) 
  (AB_ne_BC : AB ≠ BC) : problem_statement AB BC AD CD quad incircle_ABC incircle_ADC circle_omega omega_tangent_AD_CD AB_ne_BC := sorry

end prove_tangents_intersect_on_circle_l145_145959


namespace part_one_part_two_l145_145506

open Real

variable {a b m : ℝ}

theorem part_one (m : ℝ) : 
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  in Δ > 0 := 
by
  let Δ := (2 * m + 1)^2 - 4 * (m^2 + m)
  show Δ > 0 from by {
    simp[Δ],
    linarith
  }

theorem part_two (a b m : ℝ) (h1 : a + b = 2 * m + 1) (h2 : a * b = m^2 + m) (h3 : (2*a + b) * (a + 2*b) = 20) : 
  m = -2 ∨ m = 1 :=
by
  have h4: 9*m^2 + 9*m + 2 = 20 := by {
    rw [h1, h2, ← add_assoc, ← add_assoc],
    linarith
  }
  have h5 := solve_quadratic 9 9 (-18) h4
  cases h5 with h5_left h5_right
  exact or.inl h5_left
  exact or.inr h5_right

end part_one_part_two_l145_145506


namespace petya_higher_chance_of_winning_l145_145209

theorem petya_higher_chance_of_winning 
  {total_candies : ℕ} (h_total : total_candies = 25)
  {P_two_caramel : ℝ} (h_P_two_caramel : P_two_caramel = 0.54) :
  let P_two_chocolate := 1 - P_two_caramel in
  P_two_chocolate < 0.5 :=
by {
  sorry
}

end petya_higher_chance_of_winning_l145_145209


namespace centroid_trajectory_hyperbola_l145_145215

/-- Proof that the trajectory of the centroid of the triangle 
    formed by the foci and a point moving on a hyperbola is an equation. -/
theorem centroid_trajectory_hyperbola :
  ∀ (x y : ℝ),
  (∀ (m n : ℝ), 
    (dfrac m 3) = x ∧
    (dfrac n 3) = y ∧
    (dfrac (m ^ 2) 16 - dfrac (n ^ 2) 9 = 1)) →
  (dfrac (x ^ 2) (16 / 9) - y ^ 2 = 1 ∧ y ≠ 0) :=
by
  sorry

end centroid_trajectory_hyperbola_l145_145215


namespace smallest_C_for_root_on_unit_circle_l145_145719

def is_mirrored_polynomial (f : Polynomial ℝ) : Prop :=
  (Polynomial.degree f = 100) ∧
  (f.coeff 50 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → Polynomial.eval (x) f = x^100 * Polynomial.eval (1/x) f)

theorem smallest_C_for_root_on_unit_circle :
  ∀ (f : Polynomial ℝ), is_mirrored_polynomial f → f.eval 1 ≥ 51 →
  ∃ (z : ℂ), Polynomial.eval z f = 0 ∧ Complex.abs z = 1 :=
sorry

end smallest_C_for_root_on_unit_circle_l145_145719


namespace calculate_change_l145_145939

def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def dessert_cost : ℝ := 3
def dessert_discount : ℝ := 0.5
def side_dish_cost : ℝ := 1.5
def side_dish_discount : ℝ := 0.25
def tip_percentage : ℝ := 0.2
def payment_in_euros : ℝ := 50
def exchange_rate : ℝ := 1.2

theorem calculate_change :
  let total_cost := meal_cost + drink_cost + dessert_cost + side_dish_cost,
      discount := dessert_cost * dessert_discount + side_dish_cost * side_dish_discount,
      cost_with_discounts := total_cost - discount,
      tip := cost_with_discounts * tip_percentage,
      final_total_in_usd := cost_with_discounts + tip,
      final_total_in_euros := final_total_in_usd / exchange_rate,
      change := payment_in_euros - final_total_in_euros
  in change = 34.875 := by
  sorry

end calculate_change_l145_145939


namespace sum_odd_probability_zero_l145_145438

def is_prime (n : ℕ) : Prop := (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n) ∧ n > 1

def roll_results (d1 d2 d3 d4 d5 : ℕ) : Prop := 
  d1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d4 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d5 ∈ {1, 2, 3, 4, 5, 6}

theorem sum_odd_probability_zero (d1 d2 d3 d4 d5 : ℕ) (h1 : roll_results d1 d2 d3 d4 d5) (h2 : is_prime (d1 * d2 * d3 * d4 * d5)) : 
  -- probability of their sum being odd
  ∃ p : ℚ, p = 0 := 
sorry

end sum_odd_probability_zero_l145_145438


namespace three_gorges_dam_capacity_scientific_notation_l145_145677

theorem three_gorges_dam_capacity_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (16780000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.678 ∧ n = 7 :=
by
  sorry

end three_gorges_dam_capacity_scientific_notation_l145_145677


namespace cost_of_frozen_yoghurt_l145_145375

theorem cost_of_frozen_yoghurt (y : ℝ) :
  (10 * 4) - (40 - y * 4) = 36 → y = 1 :=
by
  intro h
  linear_combination 4 * (-10 * 4 + 36) + 40
  sorry

end cost_of_frozen_yoghurt_l145_145375


namespace find_values_of_f1_and_fn1_determine_parity_of_f_l145_145241

variable (f : ℝ → ℝ)
variable (D : Set ℝ)
variable (hD : D = {x | x ≠ 0})
variable (h_property : ∀ (x1 x2 : ℝ), x1 ∈ D → x2 ∈ D → f(x1 * x2) = f(x1) + f(x2))

theorem find_values_of_f1_and_fn1 (hD : D = {x | x ≠ 0}) 
    (h_property : ∀ (x1 x2 : ℝ), x1 ∈ D → x2 ∈ D → f(x1 * x2) = f(x1) + f(x2)) :
    f(1) = 0 ∧ f(-1) = 0 := 
sorry

theorem determine_parity_of_f (hD : D = {x | x ≠ 0}) 
    (h_property : ∀ (x1 x2 : ℝ), x1 ∈ D → x2 ∈ D → f(x1 * x2) = f(x1) + f(x2))
    (hf1 : f(1) = 0) (hfn1 : f(-1) = 0) 
    (hx : ∀ x ∈ D, f(-x) = f(x)) :
    ∀ x ∈ D, f(-x) = f(x) :=
sorry

end find_values_of_f1_and_fn1_determine_parity_of_f_l145_145241


namespace alice_profit_l145_145359

def total_bracelets : ℕ := 52
def cost_materials : ℝ := 3.0
def bracelets_given_away : ℕ := 8
def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_sold := total_bracelets - bracelets_given_away in
  let total_revenue := bracelets_sold * price_per_bracelet in
  let profit := total_revenue - cost_materials in
  profit = 8.00 :=
by
  sorry

end alice_profit_l145_145359


namespace total_taxi_trip_cost_l145_145129

def initial_fee : ℝ := 2.25
def normal_rate : ℝ := 0.3
def peak_hour_rate_increase : ℝ := 0.2
def off_peak_hour_rate_decrease : ℝ := 0.15
def miles_during_peak : ℝ := 1.8
def miles_during_off_peak : ℝ := 1.8
def increment_distance : ℝ := 2 / 5

def peak_hour_rate : ℝ := normal_rate * (1 + peak_hour_rate_increase)
def off_peak_hour_rate : ℝ := normal_rate * (1 - off_peak_hour_rate_decrease)

def increments (miles : ℝ) : ℝ := miles / increment_distance
def cost (miles : ℝ) (rate : ℝ) : ℝ := increments miles * rate

def total_charge : ℝ :=
  initial_fee + cost miles_during_peak peak_hour_rate + cost miles_during_off_peak off_peak_hour_rate

theorem total_taxi_trip_cost : total_charge = 5.02 := by
  sorry

end total_taxi_trip_cost_l145_145129


namespace triangle_min_perimeter_l145_145757

-- Definitions of points A, B, and C and the conditions specified in the problem.
def pointA : ℝ × ℝ := (3, 2)
def pointB (t : ℝ) : ℝ × ℝ := (t, t)
def pointC (c : ℝ) : ℝ × ℝ := (c, 0)

-- Main theorem which states that the minimum perimeter of triangle ABC is sqrt(26).
theorem triangle_min_perimeter : 
  ∃ (B C : ℝ × ℝ), B = pointB (B.1) ∧ C = pointC (C.1) ∧ 
  ∀ (B' C' : ℝ × ℝ), B' = pointB (B'.1) ∧ C' = pointC (C'.1) →
  (dist pointA B + dist B C + dist C pointA ≥ dist (2, 3) (3, -2)) :=
by 
  sorry

end triangle_min_perimeter_l145_145757


namespace range_of_a_l145_145537

theorem range_of_a :
  (∃ (x y a : ℝ), (x - a)^2 + (y - a)^2 = 4) →
  (∀ (a : ℝ), √2 * |a| = 2 → -2 * √2 < a ∧ a < 2 * √2 ∧ a ≠ 0) :=
by
  intros h a h1
  -- Here will be the proof
  sorry

end range_of_a_l145_145537


namespace jacob_subtract_l145_145686

/-- To compute 49^2, Jacob subtracts 99 from 50^2. Given that 50^2 - 99 = 49^2. -/ 
theorem jacob_subtract (a b : ℕ) (h : 50^2 - b = 49^2) : b = 99 :=
by {
  -- Setup known squares
  have h50 : 50^2 = 2500 := by norm_num,
  have h49 : 49^2 = 2401 := by norm_num,
  
  -- Substitute h50 and h49 in the assumption h
  rw [h50, h49] at h,
  
  -- Solve for b
  linarith,
}

end jacob_subtract_l145_145686


namespace quadratic_eq_two_distinct_real_roots_find_m_value_l145_145497

noncomputable def quadratic_eq_discriminant {m : ℝ} : ℝ :=
    (-(2 * m + 1)) ^ 2 - 4 * 1 * (m ^ 2 + m)

theorem quadratic_eq_two_distinct_real_roots (m : ℝ) : quadratic_eq_discriminant m > 0 :=
    by
    sorry

-- Let's define the conditions for the second part
def quadratic_eq_roots (m a b : ℝ) :=
    a + b = 2 * m + 1 ∧ a * b = m ^ 2 + m

def equation_holds (a b : ℝ) :=
    (2 * a + b) * (a + 2 * b) = 20

theorem find_m_value (m a b : ℝ) (h : quadratic_eq_roots m a b) (h_eq : equation_holds a b) :
    m = -2 ∨ m = 1 :=
    by
    sorry

end quadratic_eq_two_distinct_real_roots_find_m_value_l145_145497


namespace M_inter_N_eq_singleton_l145_145511

def M (x y : ℝ) : Prop := x + y = 2
def N (x y : ℝ) : Prop := x - y = 4

theorem M_inter_N_eq_singleton :
  {p : ℝ × ℝ | M p.1 p.2} ∩ {p : ℝ × ℝ | N p.1 p.2} = { (3, -1) } :=
by
  sorry

end M_inter_N_eq_singleton_l145_145511


namespace johns_eggs_per_week_l145_145570

noncomputable def total_emus (pens : ℕ) (emus_per_pen : ℕ) : ℕ :=
  pens * emus_per_pen

noncomputable def female_emus (total : ℕ) : ℕ :=
  total / 2

noncomputable def eggs_per_week (females : ℕ) (days_per_week : ℕ) : ℕ :=
  females * days_per_week

theorem johns_eggs_per_week :
  let pens := 4 in
  let emus_per_pen := 6 in
  let days_per_week := 7 in
  let total := total_emus pens emus_per_pen in
  let females := female_emus total in
  eggs_per_week females days_per_week = 84 :=
by
  sorry

end johns_eggs_per_week_l145_145570


namespace g_sum_eq_2_l145_145812

def g (n : ℕ) : ℝ := Math.log (n^2) / Math.log 3003

theorem g_sum_eq_2 : g 3 + g 11 + g 13 = 2 := by
  sorry

end g_sum_eq_2_l145_145812


namespace petya_has_higher_chance_l145_145194

theorem petya_has_higher_chance : 
  let p_chocolate_chocolate : ℝ := 1 - 0.54 in
  p_chocolate_chocolate < 0.5 :=
by
  let p_chocolate_chocolate := 1 - 0.54
  have h1 : p_chocolate_chocolate = 0.46 := by norm_num
  have h2 : 0.46 < 0.5 := by norm_num
  exact h2

end petya_has_higher_chance_l145_145194


namespace reflectionY_matrix_correct_l145_145416

-- Define the basis vectors e₁ and e₂
def e1 : Vector := ⟨1, 0⟩
def e2 : Vector := ⟨0, 1⟩

-- Define the transformation that reflects over the y-axis
def reflectY : Vector → Vector 
| ⟨x, y⟩ => ⟨-x, y⟩

-- Conditions given in the problem
lemma reflectY_e1 : reflectY e1 = ⟨-1, 0⟩ := sorry
lemma reflectY_e2 : reflectY e2 = ⟨0, 1⟩ := sorry

-- The goal is to find the transformation matrix for reflection over the y-axis
def reflectionMatrixY : Matrix 2 2 ℝ :=
  Matrix.of_vec ([-1, 0, 0, 1])

theorem reflectionY_matrix_correct :
  ∀ (v : Vector), reflectY v = reflectionMatrixY.mul_vec v := sorry

end reflectionY_matrix_correct_l145_145416


namespace constant_term_expansion_l145_145920

-- Define the polynomial expressions
def poly1 : ℝ[X] := 1 + X^2
def poly2 : ℝ[X] := (1 - X⁻¹)^5

-- State the theorem
theorem constant_term_expansion : (constant_term ((1 + X^2) * (1 - X⁻¹)^5) = 11) :=
sorry

end constant_term_expansion_l145_145920


namespace find_b_l145_145891

theorem find_b 
  (a b c d : ℚ) 
  (h1 : a = 2 * b + c) 
  (h2 : b = 2 * c + d) 
  (h3 : 2 * c = d + a - 1) 
  (h4 : d = a - c) : 
  b = 2 / 9 :=
by
  -- Proof is omitted (the proof steps would be inserted here)
  sorry

end find_b_l145_145891


namespace intersection_M_N_l145_145509

def M : Set ℕ := { y | y < 6 }
def N : Set ℕ := {2, 3, 6}

theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l145_145509


namespace train_passes_man_in_1_2727_seconds_l145_145755

def train_length : ℝ := 140     -- Train's length in meters
def man's_speed_kmph : ℝ := 6   -- Man's speed in km/h
def train_speed_kmph : ℝ := 77.993280537557 -- Train's speed in km/h
def conversion_factor : ℝ := 5 / 18  -- Conversion factor from km/h to m/s

-- Converting speeds from km/h to m/s
def man's_speed_mps : ℝ := man's_speed_kmph * conversion_factor
def train_speed_mps : ℝ := train_speed_kmph * conversion_factor

-- Man and train are moving in opposite directions, hence their relative speed is additive
def relative_speed : ℝ := man's_speed_mps + train_speed_mps

def desired_time : ℝ := train_length / relative_speed

theorem train_passes_man_in_1_2727_seconds : abs (desired_time - 1.2727) < 1e-4 :=
by
  sorry

end train_passes_man_in_1_2727_seconds_l145_145755


namespace triangle_ratio_l145_145928

theorem triangle_ratio
  {A B C : Type*} [RealNormedField A]
  (hC_trisect : ∃ D M, isAltitude ∠C D ∧ isMedian ∠C M ∧ ∠A + ∠B + ∠C = π / 2 ∧ ∠A = π / 6 ∧ ∠B = π / 3) :
  (a / b / c = 1 / sqrt 3 / 2) :=
sorry

end triangle_ratio_l145_145928


namespace circle_center_and_radius_l145_145649

-- The original equation of the circle
def original_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x = 0

-- The standard form of the circle equation
def standard_circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

theorem circle_center_and_radius :
  (∀ x y : ℝ, original_circle_equation x y ↔ standard_circle_equation x y) →
  (center radius : ℝ × ℝ) radius (center = (3,0) ∧ radius = 3) :=
by
  intro h
  have : ∀ x y : ℝ, original_circle_equation x y ↔ standard_circle_equation x y := h
  sorry

end circle_center_and_radius_l145_145649


namespace sum_solutions_congruence_l145_145703

theorem sum_solutions_congruence : 
  (∑ x in Finset.filter (λ x, x ≤ 25) (Finset.filter (λ x, 7 * (5 * x - 3) % 9 = 35 % 9) (Finset.range 26))) = 48 := by
sorry

end sum_solutions_congruence_l145_145703


namespace b_k_divisible_by_11_is_5_l145_145589

def b (n : ℕ) : ℕ :=
  -- Function to concatenate numbers from 1 to n
  let digits := List.join (List.map (λ x => Nat.digits 10 x) (List.range' 1 n.succ))
  digits.foldl (λ acc d => acc * 10 + d) 0

def g (n : ℕ) : ℤ :=
  let digits := Nat.digits 10 n
  digits.enum.foldl (λ acc ⟨i, d⟩ => if i % 2 = 0 then acc + Int.ofNat d else acc - Int.ofNat d) 0

def isDivisibleBy11 (n : ℕ) : Bool :=
  g n % 11 = 0

def count_b_k_divisible_by_11 : ℕ :=
  List.length (List.filter isDivisibleBy11 (List.map b (List.range' 1 51)))

theorem b_k_divisible_by_11_is_5 : count_b_k_divisible_by_11 = 5 := by
  sorry

end b_k_divisible_by_11_is_5_l145_145589


namespace age_difference_l145_145730

theorem age_difference (father_age : ℕ) (son_age_5_years_ago : ℕ) : father_age = 38 → son_age_5_years_ago = 14 → father_age - (son_age_5_years_ago + 5) = 19 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end age_difference_l145_145730


namespace find_original_number_l145_145407

theorem find_original_number (N : ℕ) (h : ∃ k : ℕ, N - 5 = 13 * k) : N = 18 :=
sorry

end find_original_number_l145_145407


namespace complement_union_l145_145173

open Set

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 2}
def N : Set ℕ := {0, 2, 3}

theorem complement_union :
  compl (M ∪ N) = {1} :=
by
  sorry

end complement_union_l145_145173


namespace composite_proof_l145_145576

theorem composite_proof (K L M N : ℕ) (h1 : K > L) (h2 : L > M) (h3 : M > N)
  (h4 : K > 0) (h5 : L > 0) (h6 : M > 0) (h7 : N > 0)
  (h8 : K * M + L * N = (K + L - M + N) * (-K + L + M + N)) :
  ∃ a b, a > 1 ∧ b > 1 ∧ KL + MN = a * b :=
sorry

end composite_proof_l145_145576


namespace spadesuit_eval_l145_145441

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval_l145_145441


namespace percent_within_one_standard_deviation_l145_145317

variable (m d : ℝ)
variable (distribution : ℝ → ℝ)
variable (symmetric_about_mean : ∀ x, distribution (m + x) = distribution (m - x))
variable (percent_less_than_m_plus_d : distribution (m + d) = 0.84)

theorem percent_within_one_standard_deviation :
  distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end percent_within_one_standard_deviation_l145_145317


namespace parabola_translation_l145_145327

theorem parabola_translation (a b x : ℝ) : 
  let f (x : ℝ) := a * x^2 + b * (x - 4)
      g (x : ℝ) := a * x^2 - b * (x + 4)
  in (f(x) + g(x)) = 2 * a * x^2 - 8 * b :=
by
  sorry

end parabola_translation_l145_145327


namespace monotonic_decreasing_interval_l145_145661

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = 0 ∧ b = Real.exp (-1) ∧ Ioo a b = {x : ℝ | y'.derivative_at x ≤ 0} := 
sorry

end monotonic_decreasing_interval_l145_145661


namespace circle_equation_correct_l145_145167

theorem circle_equation_correct :
  (∃ M : ℝ × ℝ, M.1 * 2 + M.2 - 1 = 0 ∧
                (M.1 - 3)^2 + (M.2 - 0)^2 = 5 ∧ 
                (M.1 - 0)^2 + (M.2 - 1)^2 = 5) →
  ∃ h k r : ℝ, (h = 1) ∧ (k = -1) ∧ (r = sqrt 5) ∧ 
               (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 5) :=
begin
  sorry
end

end circle_equation_correct_l145_145167


namespace product_geq_n_minus_one_pow_n_l145_145963

open Real

theorem product_geq_n_minus_one_pow_n
  (n : ℕ)
  (x : Fin n → ℝ)
  (hx : ∀ i, 0 < x i)
  (h : ∑ i, 1 / (1 + x i) = 1) :
  (∏ i, x i) ≥ (n - 1) ^ n :=
by
  -- proof goes here
  sorry

end product_geq_n_minus_one_pow_n_l145_145963


namespace counting_numbers_dividing_48_with_remainder_7_l145_145874

theorem counting_numbers_dividing_48_with_remainder_7 :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n > 7 ∧ 48 % n = 0 :=
by
  sorry

end counting_numbers_dividing_48_with_remainder_7_l145_145874


namespace vanya_goal_l145_145983

-- Define the height types and initial conditions
def Row := ℕ    -- Natural number representation for rows

-- Define the grid
def grid := finset (ℕ × ℕ)  -- A set of pairs representing rows and columns.

-- Define initial and goal positions for Vanya's and Seryozha's pieces
structure InitialSetup :=
  (vanya_black_1 : ℕ × ℕ)   -- Vanya's first black piece position
  (vanya_black_2 : ℕ × ℕ)   -- Vanya's second black piece position
  (seryozha_white_1 : ℕ × ℕ)  -- Seryozha's first white piece position
  (seryozha_white_2 : ℕ × ℕ)  -- Seryozha's second white piece position

axiom initial_conditions : InitialSetup :=
  ⟨(4, 1), (4, 2), (1, 1), (1, 6)⟩

-- Define a move
structure Move :=
  (src : ℕ × ℕ)  -- Source position
  (dst : ℕ × ℕ)  -- Destination position

-- Define a captured condition
def is_captured (piece : ℕ × ℕ) (white_1 white_2 : ℕ × ℕ) : Prop :=
  (piece.2 = white_1.2 ∧ piece.2 = white_2.2 ∧ piece.1 = (white_1.1 + white_2.1) / 2) ∨
  (piece.1 = white_1.1 - 1 ∧ piece.1 = white_2.1 - 1 ∧ |piece.2 - white_1.2| = |piece.2 - white_2.2|)

-- Define the goal: Both black pieces reaching row 1 without being captured
theorem vanya_goal : ∀ (moves : list Move), (∀ m ∈ moves, (vanya_turn m.src → ¬ is_captured m.dst)) → ¬ ∃ (moves_seryozha : list Move), (∀ m ∈ moves_seryozha, seryozha_turn m.src → is_captured m.dst) :=
begin
  sorry  -- Proof required
end


end vanya_goal_l145_145983


namespace trajectory_is_parabola_l145_145838

theorem trajectory_is_parabola
  (P : ℝ × ℝ) : 
  (dist P (0, P.2 + 1) < dist P (0, 2)) -> 
  (P.1^2 = 8 * (P.2 + 2)) :=
by
  sorry

end trajectory_is_parabola_l145_145838


namespace quadratic_inequality_solution_l145_145814

theorem quadratic_inequality_solution (x : ℝ) : 16 ≤ x ∧ x ≤ 20 → x^2 - 36 * x + 323 ≤ 3 :=
by
  sorry

end quadratic_inequality_solution_l145_145814


namespace sufficient_condition_inequalities_l145_145450

theorem sufficient_condition_inequalities (x a : ℝ) :
  (¬ (a-4 < x ∧ x < a+4) → ¬ (1 < x ∧ x < 2)) ↔ -2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end sufficient_condition_inequalities_l145_145450


namespace radius_of_circle_touching_PT_and_passing_through_S_l145_145557

theorem radius_of_circle_touching_PT_and_passing_through_S
  (P T S : Type) [fintype PTS]
  (angle_P : ∠ P = 60) -- angle at P is 60 degrees
  (PT PS : ℝ) -- lengths PT and PS
  (r : ℝ) -- the radius of the circle touching PT at T and passing through S
  (h : r = (PT^2 + PS^2 - PT * PS) / (sqrt 3 * PS)) : Prop := 
by 
  sorry

end radius_of_circle_touching_PT_and_passing_through_S_l145_145557


namespace probability_X_eq_4_l145_145541

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

example : binom 7 4 * binom 9 6 / binom 16 10 = (binom 7 4 * binom 9 6) / binom 16 10 := by
  sorry

-- Define the conditions
def totalSections := 16
def congestedSections := 7
def selectedSections := 10
def X := 4

-- Define the probability using hypergeometric distribution
def P (X : ℕ) :=
  (binom congestedSections X * binom (totalSections - congestedSections) (selectedSections - X)) / binom totalSections selectedSections

theorem probability_X_eq_4 :
  P 4 = (binom 7 4 * binom 9 6) / binom 16 10 :=
by
  sorry

end probability_X_eq_4_l145_145541


namespace Gugu_problem_l145_145579

noncomputable def q_allowed_values : set ℝ := {0, 2, -2}

theorem Gugu_problem (q : ℝ)
    (napkin_numbers : fin 10 → ℝ)  
    (line1 : set ℝ := {a - b | a b | (∃ i j, a = napkin_numbers i ∧ b = napkin_numbers j)}) 
    (line2 : set ℝ := {qab | a b ∈ line1}) 
    (line3 : set ℝ := {a^2 + b^2 - c^2 - d^2 | a b c d ∈ line1}):
    (∀ x ∈ line2, x ∈ line3) ↔ q ∈ q_allowed_values :=
by
  sorry

end Gugu_problem_l145_145579


namespace f_increasing_l145_145013

noncomputable def f (x : Real) : Real := (2 * Real.exp x) / (1 + Real.exp x) + 1/2

theorem f_increasing : ∀ x y : Real, x < y → f x < f y := 
by
  -- the proof goes here
  sorry

end f_increasing_l145_145013


namespace b_n_bound_l145_145034

open Nat

theorem b_n_bound (a : ℕ → ℝ) (h : ∀ n, a n > 0) (rec_def : ∀ n, a (n + 1) = 1 + ∏ i in range (n + 1), a i) :
  ∀ (n : ℕ), (∑ i in range (n + 1), 1 / a (i + 1)) < 2 / a 1 := sorry

end b_n_bound_l145_145034


namespace jason_needs_87_guppies_per_day_l145_145566

def guppies_needed_per_day (moray_eel_guppies : Nat)
  (betta_fish_number : Nat) (betta_fish_guppies : Nat)
  (angelfish_number : Nat) (angelfish_guppies : Nat)
  (lionfish_number : Nat) (lionfish_guppies : Nat) : Nat :=
  moray_eel_guppies +
  betta_fish_number * betta_fish_guppies +
  angelfish_number * angelfish_guppies +
  lionfish_number * lionfish_guppies

theorem jason_needs_87_guppies_per_day :
  guppies_needed_per_day 20 5 7 3 4 2 10 = 87 := by
  sorry

end jason_needs_87_guppies_per_day_l145_145566


namespace expression_divisible_by_13_l145_145153

theorem expression_divisible_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a ^ 2007 + b ^ 2007 + c ^ 2007 + 2 * 2007 * a * b * c) % 13 = 0 := 
by 
  sorry

end expression_divisible_by_13_l145_145153


namespace simplify_fraction_l145_145373

theorem simplify_fraction (x : ℝ) (hx : x ≠ 1) : (x^2 / (x-1)) - (1 / (x-1)) = x + 1 :=
by 
  sorry

end simplify_fraction_l145_145373


namespace percentage_increase_l145_145528

variable (E : ℝ) (P : ℝ)
variable (h1 : 1.36 * E = 495)
variable (h2 : (1 + P) * E = 454.96)

theorem percentage_increase :
  P = 0.25 :=
by
  sorry

end percentage_increase_l145_145528


namespace find_f_prime_at_2_l145_145532

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f' 2 - Real.log x

theorem find_f_prime_at_2 (f' : ℝ → ℝ) (h : ∀ x, deriv (f f') x = f' x) :
  f' 2 = -7 / 2 :=
by
  have H := h 2
  sorry

end find_f_prime_at_2_l145_145532


namespace probability_not_pulling_prize_l145_145535

theorem probability_not_pulling_prize (pulling_odds : ℕ × ℕ) (h : pulling_odds = (3, 4)) :
  let total_attempts := pulling_odds.1 + pulling_odds.2,
      not_pull_attempts := pulling_odds.2
  in (not_pull_attempts : ℚ) / total_attempts = 4 / 7 :=
by
  sorry

end probability_not_pulling_prize_l145_145535


namespace quadratic_equation_roots_l145_145799

theorem quadratic_equation_roots {x y : ℝ}
  (h1 : x + y = 10)
  (h2 : |x - y| = 4)
  (h3 : x * y = 21) : (x - 7) * (x - 3) = 0 ∨ (x - 3) * (x - 7) = 0 :=
by
  sorry

end quadratic_equation_roots_l145_145799


namespace max_interval_length_l145_145967

-- Define the function f(x).
def f (x : ℝ) (m : ℝ) := (1 / 12) * x^4 - (1 / 6) * m * x^3 - (3 / 2) * x^2

-- Define the second derivative of the function f(x).
def f'' (x : ℝ) (m : ℝ) := x^2 - m * x - 3

-- Problem statement: prove the maximum value of (b - a) is 2 given the conditions.
theorem max_interval_length (a b m : ℝ) (h : ∀ x ∈ Set.Ioo a b, f'' x m < 0) (hm : |m| ≤ 2) : b - a ≤ 2 := 
sorry

end max_interval_length_l145_145967


namespace can_write_one_if_odd_can_write_one_if_even_can_write_one_if_arbitrary_can_write_two_if_arbitrary_can_write_any_if_x_exists_l145_145740

-- Condition: We start with a positive rational number on the board.
def positive_rational_number := { r : ℚ // r > 0 }

-- Condition: For any two numbers a and b on the board, we can write the numbers a + 2b, ab^2, and a / b^2.
variables (a b : ℚ) (ha : a ∈ positive_rational_number) (hb: b ∈ positive_rational_number)

# Theorem stating it's always possible to write 1 if we start with an odd natural number.
theorem can_write_one_if_odd (n : ℕ) (hn : n % 2 = 1) : ∃ r : ℚ, r = 1 :=
  sorry

# Theorem stating it's always possible to write 1 if we start with an even natural number.
theorem can_write_one_if_even (n : ℕ) (hn : n % 2 = 0) : ∃ r : ℚ, r = 1 :=
  sorry

# Theorem stating it's always possible to write 1 if we start with any positive rational number.
theorem can_write_one_if_arbitrary (r : ℚ) (hr : r > 0) : ∃ r : ℚ, r = 1 :=
  sorry

# Theorem stating it's always possible to write 2 if we start with any positive rational number.
theorem can_write_two_if_arbitrary (r : ℚ) (hr : r > 0) : ∃ r : ℚ, r = 2 :=
  sorry

# Theorem stating there exists a number x such that if x is written on the board, any other positive rational number can be written.
theorem can_write_any_if_x_exists : ∃ x : ℚ, ∀ y : ℚ, y > 0 → (∃ r : ℚ, r = y) :=
  sorry

end can_write_one_if_odd_can_write_one_if_even_can_write_one_if_arbitrary_can_write_two_if_arbitrary_can_write_any_if_x_exists_l145_145740


namespace set_C_is_different_from_others_l145_145361

-- Define sets A, B, C, and D
def setA : Set ℝ := {x | x = 0}
def setB : Set ℝ := {a | a ^ 2 ≠ 0}
def setC : Set (Set Prop) := {a = 0}
def setD : Set ℝ := {0}

-- Statement representing the proof problem
theorem set_C_is_different_from_others :
  setC ≠ setA ∧ setC ≠ setB ∧ setC ≠ setD :=
sorry

end set_C_is_different_from_others_l145_145361


namespace sequence_sign_changes_l145_145257

theorem sequence_sign_changes 
  (a : ℕ → ℝ) (m n : ℕ) 
  (h_nonzero : ∀ i, 1 ≤ i ∧ i ≤ m → a i ≠ 0)
  (h_sum_zero : ∀ k, k ≤ n → ∑ i in Finset.range m, a (i + 1) * (i + 1)^k = 0)
  (h_cond : n < m - 1) : 
  ∃ i_list : Finset ℕ, i_list.card ≥ n + 1 ∧ (∀ i ∈ i_list, a i * a (i + 1) < 0) :=
sorry

end sequence_sign_changes_l145_145257


namespace planes_formed_through_three_lines_l145_145275

theorem planes_formed_through_three_lines (L1 L2 L3 : ℝ × ℝ × ℝ → Prop) (P : ℝ × ℝ × ℝ) :
  (∀ (x : ℝ × ℝ × ℝ), L1 x → L2 x → L3 x → x = P) →
  (∃ n : ℕ, n = 1 ∨ n = 3) :=
sorry

end planes_formed_through_three_lines_l145_145275


namespace bandit_showdown_l145_145112

theorem bandit_showdown :
  ∃ b : ℕ, b ≥ 8 ∧ b < 50 ∧
         ∀ i j : ℕ, i ≠ j → (i < 50 ∧ j < 50) →
         ∃ k : ℕ, k < 50 ∧
         ∀ b : ℕ, b < 50 → 
         ∃ l m : ℕ, l ≠ m ∧ l < 50 ∧ m < 50 ∧ l ≠ b ∧ m ≠ b :=
sorry

end bandit_showdown_l145_145112


namespace count_two_digit_numbers_with_prime_sum_l145_145879

-- Define a predicate that checks if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

-- Define the sum of digits function for a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the condition that checks if a two-digit number has a prime sum of digits
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ is_prime (sum_of_digits n)

-- Claim to prove
theorem count_two_digit_numbers_with_prime_sum : (finset.filter is_valid_number (finset.range 100)).card = 34 :=
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145879


namespace cross_section_area_correct_l145_145341

-- Define the problem statement and conditions in Lean
noncomputable def pyramid_cross_section_area : Prop :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (1, (sqrt 3), 0)
  let D := (1, (sqrt 3) / 3, 4)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, 0)
  let N := ((A.1 + C.1) / 2, (A.2 + C.2) / 2, 0)
  let P := ((D.1 + A.1) / 2, (D.2 + A.2) / 2, (D.3 + 0) / 2)
  let K := ((A.1 + C.1) / 2, (A.2 + C.2) / 2, 0)
  let h := 2 -- height from K down to the plane
  let base := 1 -- base MN of the triangle
  
  -- The cross-section area of the plane intersecting the pyramid is 1.5
  let cross_section_area := (base * h) / 2 = 1.5

  cross_section_area

-- Prove the cross-section area theorem condition
theorem cross_section_area_correct :
  pyramid_cross_section_area :=
by 
  sorry

end cross_section_area_correct_l145_145341


namespace largest_possible_value_l145_145526

noncomputable def largest_log_expression (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) : ℝ := 
  Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b

theorem largest_possible_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) (h3 : a = b) : 
  largest_log_expression a b h1 h2 = 0 :=
by
  sorry

end largest_possible_value_l145_145526


namespace probability_different_suits_l145_145363

theorem probability_different_suits {cards : Finset ℕ} {suits : Finset (Finset ℕ)}
  (h_cards : cards.card = 60) (h_suits : suits.card = 5)
  (h_each_suit : ∀ suit ∈ suits, suit.card = 12)
  (h_union : suits.bUnion id = cards) :
  ∃ p : ℚ, p = 48/59 := by
  sorry

end probability_different_suits_l145_145363


namespace probability_linda_luis_in_picture_l145_145969

-- Define conditions 
def lap_time_linda := 120 -- seconds
def lap_time_luis := 75 -- seconds
def start_time := 0 -- initial start time in seconds
def snapshot_time_start := 15 * 60 -- 15 minutes in seconds
def snapshot_time_end := 16 * 60 -- 16 minutes in seconds
def track_fraction_pictured := 1 / 3

-- Define what we are proving
theorem probability_linda_luis_in_picture :
  (snapshot_time_end - snapshot_time_start) / lap_time_linda = 7.5 ∧
  (snapshot_time_end - snapshot_time_start) / lap_time_luis = 12 →
  -- Calculate time intervals
  let interval_linda_start := snapshot_time_start - (track_fraction_pictured * lap_time_linda) in
  let interval_linda_end := snapshot_time_start + (track_fraction_pictured * lap_time_linda) in
  let interval_luis_start := snapshot_time_start - (track_fraction_pictured * lap_time_luis) in
  let interval_luis_end := snapshot_time_start + (track_fraction_pictured * lap_time_luis) in
  -- Overlap interval
  let overlap_start := max interval_linda_start interval_luis_start in
  let overlap_end := min interval_linda_end interval_luis_end in
  -- Probability calculation
  (overlap_end - overlap_start) / (snapshot_time_end - snapshot_time_start) = 5 / 6 :=
sorry

end probability_linda_luis_in_picture_l145_145969


namespace polygon_area_eq_192_l145_145683

theorem polygon_area_eq_192
  (side_length : ℝ)
  (rotation1 : ℝ)
  (rotation2 : ℝ)
  (h1 : side_length = 8)
  (h2 : rotation1 = 20)
  (h3 : rotation2 = 50) :
  area_of_polygon_formed_by_rotations side_length rotation1 rotation2 = 192 := sorry

end polygon_area_eq_192_l145_145683


namespace Petya_higher_chance_l145_145199

theorem Petya_higher_chance :
  ∀ (P_vasya P_petya : ℝ), P_vasya = 0.46 → P_petya = 0.54 → P_vasya < P_petya :=
by
  intros P_vasya P_petya h1 h2
  rw [h1, h2]
  linarith

end Petya_higher_chance_l145_145199


namespace initial_percentage_increase_l145_145666

theorem initial_percentage_increase (P : ℝ) (x : ℝ) : 
  (P * (1 + x / 100) * 0.9 = P * 1.125) → (x = 25) :=
by
  intro h
  -- introduce intermediate steps leading to the conclusion for clarity
  have h1 : 1 + x / 100 = 1.125 / 0.9,
  { rw [← h, mul_assoc, mul_comm P, mul_assoc], simp },
  have h2 : x / 100 = 0.25,
  { field_simp [h1], ring_nf },
  have h3 : x = 25,
  { field_simp [h2], ring_nf },
  exact h3

end initial_percentage_increase_l145_145666


namespace expr_eval_l145_145679

theorem expr_eval : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end expr_eval_l145_145679


namespace regular_pentagon_to_rectangle_l145_145992

-- Statement
theorem regular_pentagon_to_rectangle : 
  ∀ (pentagon : Type) (is_regular : (∀ (a b : pentagon), dist a b = dist b c) ∧ (∀ (x : pentagon), angle x = 108)) , 
  ∃ (parts : list (set pentagon)), 
  (∀ (p1 p2 ∈ parts, p1 ≠ p2 → disjoint p1 p2)) ∧ 
  (⋃ (p ∈ parts), p = pentagon) ∧ 
  (∃ (rectangle : set pentagon), is_rectangle rectangle ∧ (⋃ (p ∈ parts), p = rectangle)) :=
sorry

end regular_pentagon_to_rectangle_l145_145992


namespace reflect_over_y_axis_matrix_l145_145414

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l145_145414


namespace taoqi_has_higher_utilization_rate_l145_145238

noncomputable def area_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius * radius

noncomputable def utilization_rate (cut_area : ℝ) (original_area : ℝ) : ℝ :=
  cut_area / original_area

noncomputable def tao_qi_utilization_rate : ℝ :=
  let side_length := 9
  let square_area := area_square side_length
  let radius := side_length / 2
  let circle_area := area_circle radius
  utilization_rate circle_area square_area

noncomputable def xiao_xiao_utilization_rate : ℝ :=
  let diameter := 9
  let radius := diameter / 2
  let large_circle_area := area_circle radius
  let small_circle_radius := diameter / 6
  let small_circle_area := area_circle small_circle_radius
  let total_small_circles_area := 7 * small_circle_area
  utilization_rate total_small_circles_area large_circle_area

-- Theorem statement reflecting the proof problem:
theorem taoqi_has_higher_utilization_rate :
  tao_qi_utilization_rate > xiao_xiao_utilization_rate := by sorry

end taoqi_has_higher_utilization_rate_l145_145238


namespace power_increase_factor_l145_145760

variables (F k v v_new : ℝ)
hypothesis h1 : F = k * v^2
hypothesis h2 : 2 * F = k * v_new^2

theorem power_increase_factor : (2 * F * v_new) / (F * v) = 2 * real.sqrt 2 :=
by
  sorry

end power_increase_factor_l145_145760


namespace michelle_will_have_four_crayons_l145_145978

def michelle_crayons (m j : ℕ) : ℕ := m + j

theorem michelle_will_have_four_crayons (H₁ : michelle_crayons 2 2 = 4) : michelle_crayons 2 2 = 4 :=
by
  sorry

end michelle_will_have_four_crayons_l145_145978


namespace subset_condition_l145_145581

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_condition_l145_145581


namespace problem_f_12_l145_145029

variable {α : Type} {f : α → ℕ} {p q : ℕ}
variable (h1 : ∀ a b : α, f (a * b) = f a + f b)
variable (h2 : f 2 = p)
variable (h3 : f 3 = q)

theorem problem_f_12 : f 12 = 2 * p + q :=
  sorry

end problem_f_12_l145_145029


namespace integral_arctan_x_div_x_squared_l145_145371

theorem integral_arctan_x_div_x_squared (C : ℝ) : 
  ∫ (x : ℝ) in ℝ, (4 * Real.arctan x - x) / (1 + x^2) dx 
  = (1 / 2) * (4 * (Real.arctan x)^2 - Real.log (1 + x^2)) + C := by
sorry

end integral_arctan_x_div_x_squared_l145_145371


namespace pizza_store_total_sales_l145_145981

theorem pizza_store_total_sales (pepperoni bacon cheese : ℕ) (h1 : pepperoni = 2) (h2 : bacon = 6) (h3 : cheese = 6) :
  pepperoni + bacon + cheese = 14 :=
by sorry

end pizza_store_total_sales_l145_145981


namespace colten_chickens_l145_145996

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l145_145996


namespace candidate_net_gain_difference_l145_145335

theorem candidate_net_gain_difference :
  let salary1 := 42000
      revenue1 := 93000
      training_cost_per_month := 1200
      training_months := 3
      salary2 := 45000
      revenue2 := 92000
      hiring_bonus_percent := 1 / 100 in
  let total_training_cost1 := training_cost_per_month * training_months in
  let hiring_bonus2 := salary2 * hiring_bonus_percent in
  let net_gain1 := revenue1 - salary1 - total_training_cost1 in
  let net_gain2 := revenue2 - salary2 - hiring_bonus2 in
  net_gain1 - net_gain2 = 850 :=
by
  sorry

end candidate_net_gain_difference_l145_145335


namespace induction_step_l145_145468

-- Define the proposition P(n) for even n
def P (n : ℕ) : Prop := sorry -- Replace with the actual proposition P

-- Define the proposition holds for k
variables (k : ℕ) (h_even_k : even k) (h_k_ge_2 : k ≥ 2) (h_Pk : P k)

-- State that if P(k) is true for even k, then P(k + 2) must also hold
theorem induction_step (h_induction : ∀ k, (k ≥ 2) → even k → P k → P (k + 2)) : P (k + 2) :=
begin
  exact h_induction k h_k_ge_2 h_even_k h_Pk,
end

end induction_step_l145_145468


namespace seq_a_expression_l145_145863

noncomputable def seq_a : ℕ → ℤ
| 1     := 3
| (n+2) := seq_a (n+1) + 3^(n+2)

theorem seq_a_expression (n : ℕ) (hn : n ≥ 1) : seq_a (n + 1) = (3^(n+2) - 3) / 2 :=
by sorry

end seq_a_expression_l145_145863


namespace theta_in_second_quadrant_l145_145955

def z (θ : ℝ) : Complex := Complex.mk (Real.cos θ) (Real.sin θ)

theorem theta_in_second_quadrant (θ : ℝ) : 
    (z θ).re < 0 ∧ (z θ).im > 0 ↔ (Real.cos θ < 0 ∧ Real.sin θ > 0) := by
  sorry

end theta_in_second_quadrant_l145_145955


namespace petya_higher_chance_of_winning_l145_145212

/-- Petya gives Vasya two boxes of candies. Each box contains both chocolate candies and caramel candies.
    In total, there are 25 candies in both boxes. Vasya takes one candy from each box.
    If both candies turn out to be chocolate, Vasya wins. Otherwise, Petya wins.
    The probability that Vasya will get two caramels is 0.54.
    Prove that Petya has a higher chance of winning. --/
theorem petya_higher_chance_of_winning
  (total_candies : ℕ)
  (total_candies = 25 : Prop) 
  (prob_two_caramels : ℝ)
  (prob_two_caramels = 0.54 : Prop) :
  0.46 < 0.54 := 
by 
  sorry

end petya_higher_chance_of_winning_l145_145212


namespace sum_bn_l145_145040

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

def geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 1 = a 0 * r ∧ a 2 = a 1 * r

-- Given S_5 = 35
def S5_property (S : ℕ → ℕ) := S 5 = 35

-- a_1, a_4, a_{13} is a geometric sequence
def a1_a4_a13_geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 4 = a 1 * r ∧ a 13 = a 4 * r

-- Define the sequence b_n and conditions
def bn_prop (a b : ℕ → ℕ) := ∀ n : ℕ, b n = a n * (2^(n-1))

-- Main theorem
theorem sum_bn {a b : ℕ → ℕ} {S T : ℕ → ℕ} (h_a : arithmetic_sequence a 2) (h_S5 : S5_property S) (h_geo : a1_a4_a13_geometric_sequence a) (h_bn : bn_prop a b)
  : ∀ n : ℕ, T n = 1 + (2 * n - 1) * 2^n := sorry

end sum_bn_l145_145040


namespace trapezoids_are_parallelograms_l145_145713

variables {Point : Type} [affine_space Point] (A B C D A' B' C' D' : Point)

-- Define the segment lengths
variable (AB A'B' BC B'C' CD C'D' AD A'D' : ℝ)
variable (parallel_AB_CD parallel_BC_AD' : Prop)

-- Define the equality of corresponding sides
hypothesis h1 : AB = A'B'
hypothesis h2 : BC = B'C'
hypothesis h3 : CD = C'D'
hypothesis h4 : AD = A'D'

-- Parallelism conditions
hypothesis h5 : parallel_AB_CD
hypothesis h6 : parallel_BC_AD'

-- Statement to prove the two trapezoids are parallelograms
theorem trapezoids_are_parallelograms :
  parallelogram A B C D ∧ parallelogram A' B' C' D' :=
sorry

end trapezoids_are_parallelograms_l145_145713


namespace sum_3x_4y_l145_145098

theorem sum_3x_4y (x y N : ℝ) (H1 : 3 * x + 4 * y = N) (H2 : 6 * x - 4 * y = 12) (H3 : x * y = 72) : 3 * x + 4 * y = 60 := 
sorry

end sum_3x_4y_l145_145098


namespace quadratic_eq_two_distinct_real_roots_find_m_value_l145_145493

noncomputable def quadratic_eq_discriminant {m : ℝ} : ℝ :=
    (-(2 * m + 1)) ^ 2 - 4 * 1 * (m ^ 2 + m)

theorem quadratic_eq_two_distinct_real_roots (m : ℝ) : quadratic_eq_discriminant m > 0 :=
    by
    sorry

-- Let's define the conditions for the second part
def quadratic_eq_roots (m a b : ℝ) :=
    a + b = 2 * m + 1 ∧ a * b = m ^ 2 + m

def equation_holds (a b : ℝ) :=
    (2 * a + b) * (a + 2 * b) = 20

theorem find_m_value (m a b : ℝ) (h : quadratic_eq_roots m a b) (h_eq : equation_holds a b) :
    m = -2 ∨ m = 1 :=
    by
    sorry

end quadratic_eq_two_distinct_real_roots_find_m_value_l145_145493


namespace value_of_x_l145_145900

theorem value_of_x (x α β : ℝ) (h1 : x = Real.log 3 (Real.tan α))
  (h2 : x = -Real.log 3 (Real.tan β)) (h3 : α - β = Real.pi / 6) :
  x = 1 / 2 := by
  sorry

end value_of_x_l145_145900


namespace fraction_of_trunks_l145_145366

theorem fraction_of_trunks (h1 : 0.38 ≤ 1) (h2 : 0.63 ≤ 1) : 
  0.63 - 0.38 = 0.25 :=
by
  sorry

end fraction_of_trunks_l145_145366


namespace diagonals_in_octagon_l145_145516

/-- The formula to calculate the number of diagonals in a polygon -/
def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

/-- The number of sides in an octagon -/
def sides_of_octagon : Nat := 8

/-- The number of diagonals in an octagon is 20. -/
theorem diagonals_in_octagon : number_of_diagonals sides_of_octagon = 20 :=
by
  sorry

end diagonals_in_octagon_l145_145516


namespace compound_pyramid_edge_length_l145_145780

noncomputable def hexagon_side := 8 -- cm
noncomputable def hexagon_height := 15 -- cm
noncomputable def pentagon_side := 5 -- cm
noncomputable def small_pyramid_height := 10 -- cm

theorem compound_pyramid_edge_length :
    let hexagon_edges := 6 * hexagon_side
    let hex_slant_height := Math.sqrt (hexagon_height ^ 2 + (hexagon_side / (2 * Math.tan (Real.pi / 6))) ^ 2)
    let hexagon_slant_edges := 6 * hex_slant_height
    let pentagon_edges := 5 * pentagon_side
    let pent_slant_height := Math.sqrt (small_pyramid_height ^ 2 + (pentagon_side / (2 * Math.tan (Real.pi / 5))) ^ 2)
    let pentagon_slant_edges := 5 * pent_slant_height
    in hexagon_edges + hexagon_slant_edges + pentagon_edges + pentagon_slant_edges ≈ 220 := 
begin
  sorry
end

end compound_pyramid_edge_length_l145_145780


namespace shekar_average_marks_l145_145631

-- Define the scores for each subject
def mathematics := 76
def science := 65
def social_studies := 82
def english := 67
def biology := 55
def computer_science := 89
def history := 74
def geography := 63
def physics := 78
def chemistry := 71

-- Define the total number of subjects
def number_of_subjects := 10

-- State the theorem to prove the average marks
theorem shekar_average_marks :
  (mathematics + science + social_studies + english + biology +
   computer_science + history + geography + physics + chemistry) 
   / number_of_subjects = 72 := 
by
  -- Proof is omitted
  sorry

end shekar_average_marks_l145_145631


namespace slope_correct_l145_145381

-- Coordinates of the vertices of the polygon
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (6, 2)
def vertex_F := (6, 0)

-- Define the total area of the polygon
def total_area : ℝ := 20

-- Define the slope of the line through the origin dividing the area in half
def slope_line_dividing_area (slope : ℝ) : Prop :=
  ∃ l : ℝ, l = 5 / 3 ∧
  ∃ area_divided : ℝ, area_divided = total_area / 2

-- Prove the slope is 5/3
theorem slope_correct :
  slope_line_dividing_area (5 / 3) :=
by
  sorry

end slope_correct_l145_145381


namespace count_two_digit_numbers_with_prime_sum_l145_145882

-- Helper predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m < n → n % m ≠ 0

-- Helper function to get the sum of digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  tens + units

-- Define the set of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the set of two-digit numbers with prime digit sum
def two_digit_numbers_with_prime_sum :=
  {n : ℕ | n ∈ two_digit_numbers ∧ is_prime (sum_of_digits n)}

-- The main theorem
theorem count_two_digit_numbers_with_prime_sum : 
  {n : ℕ | n ∈ two_digit_numbers_with_prime_sum }.to_finset.card = 33 :=
by
  -- Placeholder for the actual proof
  sorry

end count_two_digit_numbers_with_prime_sum_l145_145882


namespace course_selection_scheme_count_l145_145749

noncomputable def number_of_course_selection_schemes 
  (total_courses : ℕ) 
  (choose_courses : ℕ) 
  (A : ℕ) 
  (B : ℕ) : ℕ :=
  let neither_AB := nat.choose (total_courses - 2) (choose_courses) in
  let one_of_AB := nat.choose 2 1 * nat.choose (total_courses - 2) (choose_courses - 1) in
  neither_AB + one_of_AB

theorem course_selection_scheme_count : 
  number_of_course_selection_schemes 8 5 1 1 = 36 :=
by
  sorry

end course_selection_scheme_count_l145_145749


namespace calc_div_of_exponentiation_l145_145772

theorem calc_div_of_exponentiation : (3^12 / (27^2) = 729) := by
  have h₁ : 27 = 3^3 := rfl
  have h₂ : 27^2 = (3^3)^2 := by rw [h₁]
  have h₃ : (3^3)^2 = 3^(3*2) := pow_mul 3 3 2
  have h₄ : 3^(3*2) = 3^6 := rfl
  have h₅ : 3^12 / 3^6 = 3^(12-6) := div_eq_pow_sub 3 12 6
  have h₆ : 3^(12-6) = 3^6 := rfl
  have h₇ : 3^6 = 729 := (3^6 : ℕ)
  rw [h₂, h₃, h₄, h₅, h₆, h₇]

end calc_div_of_exponentiation_l145_145772


namespace three_digit_numbers_ending_in_6_divisible_by_9_l145_145877

theorem three_digit_numbers_ending_in_6_divisible_by_9 :
  ∃! n ≥ 100, n ≤ 999 ∧ n % 10 = 6 ∧ n % 9 = 0 :=
begin
  -- Omitted the proof part
  sorry
end

end three_digit_numbers_ending_in_6_divisible_by_9_l145_145877


namespace prism_total_surface_area_l145_145475

theorem prism_total_surface_area (P V : ℝ) (hP : P > 0) :
  let H := V / P in
  let a := Real.sqrt P in
  let surface_area := 2 * P + 4 * V / Real.sqrt P in
  surface_area = 2 * P + 4 * V / Real.sqrt P := 
by
  sorry

end prism_total_surface_area_l145_145475


namespace sum_of_coefficients_l145_145057

noncomputable def n : ℝ := ∫ x in (0:ℝ)..(real.pi / 2), 4 * real.sin x

theorem sum_of_coefficients : 
  n = ∫ x in (0:ℝ)..(real.pi / 2), 4 * real.sin x →
  (x + 2 / x) * (x - 2 / x) ^ n = 3 :=
by
  intro h
  rw [h]
  -- Proof steps go here, but we're omitting them
  sorry

end sum_of_coefficients_l145_145057


namespace reflect_y_axis_matrix_l145_145422

theorem reflect_y_axis_matrix : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, (
    (∀ v : (Fin 2 → ℝ), v = ![1, 0] → A.mulVec v = ![-1, 0]) ∧ 
    (∀ v : (Fin 2 → ℝ), v = ![0, 1] → A.mulVec v = ![0, 1])
  ) ∧ A = ![![-1, 0], ![0, 1]] :=
begin
  sorry
end

end reflect_y_axis_matrix_l145_145422


namespace trig_identity_l145_145019

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 :=
by
  sorry

end trig_identity_l145_145019


namespace min_value_proof_l145_145593

noncomputable def min_value (x : Fin 50 → ℝ) : ℝ :=
  ∑ i, x i / (2 - (x i)^2)

theorem min_value_proof (x : Fin 50 → ℝ) (h1 : ∀ i, 0 < x i) (h2 : ∑ i, (x i)^2 = 2) :
  min_value x = 3 * Real.sqrt 3 / 2 :=
sorry

end min_value_proof_l145_145593


namespace limit_derivative_l145_145447

variable {α β : Type}
variables (f : α → β) (x₀ : α)
variable [normed_field α]
variable [normed_group β]
variable [normed_space α β]
variable [complete_space β]

theorem limit_derivative (h : α) (hf : has_deriv_at f (-(3 : ℝ)) x₀) :
  tendsto (λ h, (f (x₀ + h) - f (x₀ - h)) / h) (𝓝 0) (𝓝 (-6 : ℝ)) :=
by 
  -- This is where the proof would go
  sorry

end limit_derivative_l145_145447


namespace find_difference_of_roots_l145_145170

-- Define the conditions for the given problem
def larger_root_of_eq_1 (a : ℝ) : Prop :=
  (1998 * a) ^ 2 - 1997 * 1999 * a - 1 = 0

def smaller_root_of_eq_2 (b : ℝ) : Prop :=
  b ^ 2 + 1998 * b - 1999 = 0

-- Define the main problem with the proof obligation
theorem find_difference_of_roots (a b : ℝ) (h1: larger_root_of_eq_1 a) (h2: smaller_root_of_eq_2 b) : a - b = 2000 :=
sorry

end find_difference_of_roots_l145_145170


namespace sachin_age_l145_145627

variable S R : ℝ

def condition1 := S = R - 9
def condition2 := S / R = 7 / 9

theorem sachin_age :
  (condition1 → condition2 → S = 31.5) :=
by
  intros h1 h2
  sorry

end sachin_age_l145_145627


namespace sculptures_not_on_display_count_l145_145362

noncomputable def total_art_pieces : ℕ := 1800
noncomputable def pieces_on_display : ℕ := total_art_pieces / 3
noncomputable def pieces_not_on_display : ℕ := total_art_pieces - pieces_on_display
noncomputable def sculptures_on_display : ℕ := pieces_on_display / 6
noncomputable def sculptures_not_on_display : ℕ := pieces_not_on_display * 2 / 3

theorem sculptures_not_on_display_count : sculptures_not_on_display = 800 :=
by {
  -- Since this is a statement only as requested, we use sorry to skip the proof
  sorry
}

end sculptures_not_on_display_count_l145_145362


namespace maximum_distance_l145_145916

theorem maximum_distance (m : ℝ) (h : m > 0) : 
  let l := line {(x, y) | y - (m / 2) = (-m / 4) * (x - 1)} in
  let dist := (2 - 1) ^ 2 + ((-1) - 0) ^ 2 in
  Real.sqrt dist = Real.sqrt 2 :=
by 
  let tangent_slope := -m / 4
  let P := (1, m / 2)
  let Q := (2, -1)
  let equation_l := 4 * y + m * x - 3 * m = 0
  sorry

end maximum_distance_l145_145916


namespace total_bill_l145_145641

theorem total_bill (m : ℝ) (h1 : m = 10 * (m / 10 + 3) - 27) : m = 270 :=
by
  sorry

end total_bill_l145_145641


namespace f_plus_2011_is_odd_l145_145384

def f (x : ℝ) : ℝ

constant condition : ∀ α β : ℝ, f (α + β) - (f α + f β) = 2011

theorem f_plus_2011_is_odd : ∀ x : ℝ, f(x) + 2011 = -(f(-x) + 2011) :=
by
  intro x
  sorry

end f_plus_2011_is_odd_l145_145384


namespace find_binomial_parameters_l145_145819

noncomputable def binomial_parameters (n p : ℝ) : Prop :=
  let E_X := n * p in
  let D_X := n * p * (1 - p) in
  E_X = 3 ∧ D_X = 2

theorem find_binomial_parameters : ∃ (n p : ℝ), binomial_parameters n p ∧ n = 9 ∧ p = 1 / 3 :=
sorry

end find_binomial_parameters_l145_145819


namespace sum_of_interior_angles_of_regular_polygon_l145_145249

theorem sum_of_interior_angles_of_regular_polygon (theta : ℝ) (h₁ : theta = 24) (h₂ : 360 / theta = 15) : 
  180 * (15 - 2) = 2340 :=
by
  rw [h₁, h₂]
  sorry

end sum_of_interior_angles_of_regular_polygon_l145_145249


namespace fraction_of_paper_per_book_l145_145567

theorem fraction_of_paper_per_book (total_fraction_used : ℚ) (num_books : ℕ) (h1 : total_fraction_used = 5 / 8) (h2 : num_books = 5) : 
  (total_fraction_used / num_books) = 1 / 8 :=
by
  sorry

end fraction_of_paper_per_book_l145_145567


namespace number_and_sum_of_g3_is_5_l145_145145

theorem number_and_sum_of_g3_is_5 :
  let g : ℤ → ℤ := sorry in
  (∀ m n : ℤ, g (m + n) + g (mn - 1) = g m * g n - 1) →
  g 0 = 1 →
  g (-1) = 0 →
  g 3 = 5 →
  let x := 1 in  -- Only one possible integer value for g(3)
  let y := 5 in  -- The sum of the possible values for g(3) is 5
  x * y = 5 :=
sorry

end number_and_sum_of_g3_is_5_l145_145145


namespace complex_number_in_second_quadrant_l145_145858

noncomputable def z_in_second_quadrant : Prop :=
  ∃ z : ℂ, (z, 1 - 2 * complex.I) = 0 ∧ 
           z.re < 0 ∧ 
           z.im > 0

theorem complex_number_in_second_quadrant : z_in_second_quadrant :=
  sorry

end complex_number_in_second_quadrant_l145_145858


namespace distance_is_functional_l145_145705

/-- Conditions -/
def heightWeightRelation : Prop := ∃ (p : Person), p.height ≠ p.weight
def snowfallAccidentRelation : Prop := ∃ (s : Snowfall), s.rate ≠ trafficAccidentRate s
def fertilizerYieldRelation : Prop := ∃ (f : Fertilizer), f.amount ≠ yieldPerAcre f

/-- Distance traveled by a vehicle moving at a constant speed and time is a deterministic relationship -/
def distanceTimeRelation : Prop := ∀ (v : Vehicle), ∀ (s : Speed), ∀ (t : Time), v.distance = s * t

theorem distance_is_functional :
  ¬ heightWeightRelation ∧
  ¬ snowfallAccidentRelation ∧
  ¬ fertilizerYieldRelation ∧
  distanceTimeRelation :=
sorry

end distance_is_functional_l145_145705


namespace maximize_expr_approx_l145_145424

noncomputable def expr (x y : ℝ) : ℝ := 
  (sqrt (9 - sqrt 7) * Real.sin x - sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
  (3 + 2 * sqrt (13 - sqrt 7) * Real.cos y - Real.cos (2 * y))

noncomputable def maximum_value : ℝ := 17

theorem maximize_expr_approx :
  ∃ x y : ℝ, (-1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ∧ 
             (-1 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧ 
             (-1 ≤ Real.sin y ∧ Real.sin y ≤ 1) ∧ 
             (-1 ≤ Real.cos y ∧ Real.cos y ≤ 1) ∧ 
             abs ((expr x y) - maximum_value) < 1 :=
by
  sorry

end maximize_expr_approx_l145_145424


namespace two_cats_can_catch_mouse_mouse_always_escapes_l145_145283

-- Lean statement for part (a)
theorem two_cats_can_catch_mouse
    (mouse_pos : ℕ × ℕ)
    (h_non_edge : 1 ≤ mouse_pos.1 ∧ mouse_pos.1 < 8 ∧ 1 ≤ mouse_pos.2 ∧ mouse_pos.2 < 8)
    (cat1_pos : ℕ × ℕ)
    (cat2_pos : ℕ × ℕ)
    (h_cats_on_edge : (cat1_pos.1 = 0 ∨ cat1_pos.1 = 7 ∨ cat1_pos.2 = 0 ∨ cat1_pos.2 = 7) ∧ 
                      (cat2_pos.1 = 0 ∨ cat2_pos.1 = 7 ∨ cat2_pos.2 = 0 ∨ cat2_pos.2 = 7)) :
  ∃ (steps : ℕ) (cat1_moves : list (ℕ × ℕ)) (cat2_moves : list (ℕ × ℕ)),
    (∀ step ∈ finset.range(steps), cat1_moves[step] = mouse_pos ∨ cat2_moves[step] = mouse_pos) :=
sorry

-- Lean statement for part (b)
theorem mouse_always_escapes
    (mouse_pos : ℕ × ℕ)
    (cat1_pos : ℕ × ℕ)
    (cat2_pos : ℕ × ℕ)
    (cat3_pos : ℕ × ℕ)
    (mouse_can_move_two_steps_first : bool := true) :
  ∃ (steps : ℕ) (mouse_moves : list (ℕ × ℕ)),
    (steps ≥ 1 ∧ ∃ step ∈ finset.range(steps), (mouse_moves[step].1 = 0 ∨ mouse_moves[step].1 = 7 ∨ mouse_moves[step].2 = 0 ∨ mouse_moves[step].2 = 7) ∧
      ∀ cat_pos ∈ [cat1_pos, cat2_pos, cat3_pos], mouse_moves[step] ≠ cat_pos) :=
sorry

end two_cats_can_catch_mouse_mouse_always_escapes_l145_145283
