import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMass
import Mathlib.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace find_sin_E_floor_l596_596662

variable {EF GH EH FG : ℝ}
variable (E G : ℝ)

-- Conditions from the problem
def is_convex_quadrilateral (EF GH EH FG : ℝ) : Prop := true
def angles_congruent (E G : ℝ) : Prop := E = G
def sides_equal (EF GH : ℝ) : Prop := EF = GH ∧ EF = 200
def sides_not_equal (EH FG : ℝ) : Prop := EH ≠ FG
def perimeter (EF GH EH FG : ℝ) : Prop := EF + GH + EH + FG = 800

-- The theorem to be proved
theorem find_sin_E_floor (h_convex : is_convex_quadrilateral EF GH EH FG)
                         (h_angles : angles_congruent E G)
                         (h_sides : sides_equal EF GH)
                         (h_sides_ne : sides_not_equal EH FG)
                         (h_perimeter : perimeter EF GH EH FG) :
  ⌊ 1000 * Real.sin E ⌋ = 0 := by
  sorry

end find_sin_E_floor_l596_596662


namespace abs_neg_2023_l596_596482

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596482


namespace math_problem_l596_596085

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596085


namespace convert_exp_to_rectangular_form_l596_596541

theorem convert_exp_to_rectangular_form : exp (13 * π * complex.I / 2) = complex.I :=
by
  sorry

end convert_exp_to_rectangular_form_l596_596541


namespace find_y_intercept_of_tangent_line_l596_596366

theorem find_y_intercept_of_tangent_line :
  let f := λ x : ℝ, x^3 + 11
  ∃ b : ℝ, (∃ m : ℝ, ∃ c : ℝ, c = 1 ∧ m = (deriv f c) ∧ b = f c - m * c) ∧ b = 9 := 
by
  sorry

end find_y_intercept_of_tangent_line_l596_596366


namespace alice_instructors_l596_596000

noncomputable def num_students : ℕ := 40
noncomputable def num_life_vests_Alice_has : ℕ := 20
noncomputable def percent_students_with_their_vests : ℕ := 20
noncomputable def num_additional_life_vests_needed : ℕ := 22

-- Constants based on calculated conditions
noncomputable def num_students_with_their_vests : ℕ := (percent_students_with_their_vests * num_students) / 100
noncomputable def num_students_without_their_vests : ℕ := num_students - num_students_with_their_vests
noncomputable def num_life_vests_needed_for_students : ℕ := num_students_without_their_vests - num_life_vests_Alice_has
noncomputable def num_life_vests_needed_for_instructors : ℕ := num_additional_life_vests_needed - num_life_vests_needed_for_students

theorem alice_instructors : num_life_vests_needed_for_instructors = 10 := 
by
  sorry

end alice_instructors_l596_596000


namespace tan_arccot_3_5_l596_596048

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596048


namespace tan_arccot_eq_5_div_3_l596_596029

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596029


namespace incircle_inequality_l596_596806

variables {A B C A1 B1 : Point}
variables {AC BC: Real}
variables {t : Triangle}

-- Definitions
def incircle_touches_at (t: Triangle) (P Q: Point): Prop :=
  t.incircle.touches t.side AC P ∧ t.incircle.touches t.side BC Q

-- Problem statement
theorem incircle_inequality (h1 : incircle_touches_at t A1 B1) (h2 : AC > BC) : 
  distance A A1 > distance B B1 :=
sorry

end incircle_inequality_l596_596806


namespace smallest_set_size_l596_596674

noncomputable def smallest_num_elements (s : Multiset ℝ) : ℕ :=
  s.length

theorem smallest_set_size (s : Multiset ℝ) :
  (∀ a b c : ℝ, s = {a, b, 3, 6, 6, c}) →
  (s.median = 3) →
  (s.mean = 5) →
  (∀ x, s.count x < 3 → x ≠ 6) →
  smallest_num_elements s = 6 :=
by
  intros _ _ _ _
  sorry

end smallest_set_size_l596_596674


namespace length_of_AX_l596_596055

theorem length_of_AX (A B C X : Type) [LineSegment AB BC BX AX] (AB_length : AB = 45) (BC_length : BC = 32) (BX_length : BX = 20) (X_on_AB : X ∈ AB) (CX_bisects_C : CX bisects ∠ACB) : AX = 25 :=
sorry

end length_of_AX_l596_596055


namespace tan_arccot_l596_596022

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596022


namespace b_minus_a_l596_596182

theorem b_minus_a :
  ∃ (a b : ℝ), (2 + 4 = -a) ∧ (2 * 4 = b) ∧ (b - a = 14) :=
by
  use (-6 : ℝ)
  use (8 : ℝ)
  simp
  sorry

end b_minus_a_l596_596182


namespace find_b_perpendicular_lines_l596_596108

/-- Given the direction vectors of two lines in 3D, find the value of b so that the lines are perpendicular. -/
theorem find_b_perpendicular_lines 
  (b : ℝ) 
  (v1 : ℝ × ℝ × ℝ) 
  (v2 : ℝ × ℝ × ℝ)
  (h1 : v1 = (b, -3, 2)) 
  (h2 : v2 = (2, 3, 4)) 
  (perp : v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0)
  : b = 1 / 2 := by 
    -- The proof will be written here.
    sorry

end find_b_perpendicular_lines_l596_596108


namespace area_of_bounded_figures_l596_596474

-- Conditions
def r_rose (phi : ℝ) : ℝ := 6 * Real.sin (3 * phi)
def r_circle : ℝ := 3

-- Statement of the theorem
theorem area_of_bounded_figures :
  (3 * Real.pi + (9 * Real.sqrt 3) / 2) = 3 * (1/2) * ∫ (phi in Real.pi/18)..(5 * Real.pi/18), ((r_rose phi)^2 - (r_circle)^2) :=
by
  sorry

end area_of_bounded_figures_l596_596474


namespace range_of_m_l596_596340

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 ∈ set.Icc (1 : ℝ) 3, (x1 ≤ x2 → (x1^2 - 2 * m * x1 + 3) ≤ (x2^2 - 2 * m * x2 + 3)) ∨
                              (x1 ≥ x2 → (x1^2 - 2 * m * x1 + 3) ≥ (x2^2 - 2 * m * x2 + 3))) ↔
  (m ≤ 1 ∨ m ≥ 3) :=
sorry

end range_of_m_l596_596340


namespace probability_divisible_by_5_l596_596823

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l596_596823


namespace quadratic_inequality_solution_l596_596361

theorem quadratic_inequality_solution (f : ℝ → ℝ) (h_quad : ∃ a b c : ℝ, a > 0 ∧ f = λ x, a * x^2 + b * x + c)
  (h_symmetry : ∀ x : ℝ, f x = f (4 - x)) :
  (∀ x : ℝ, f(1 - 2 * x^2) < f(1 + 2 * x - x^2) ↔ -2 < x ∧ x < 0) :=
sorry

end quadratic_inequality_solution_l596_596361


namespace total_sailors_count_l596_596444

noncomputable def number_of_sailors (inexperienced_sailors : ℕ) (inexperienced_hourly_wage : ℕ) 
(inexperienced_hours_per_week : ℕ) (monthly_earnings_experienced : ℕ) (experience_pay_increase_factor : ℚ) 
(monthly_weeks : ℕ) (total_earnings_experienced : ℕ) : ℕ :=
  let inexperienced_weekly_earnings := inexperienced_hourly_wage * inexperienced_hours_per_week
  let experienced_hourly_wage := inexperienced_hourly_wage + (inexperienced_hourly_wage * experience_pay_increase_factor)
  let experienced_weekly_earnings := experienced_hourly_wage * inexperienced_hours_per_week
  let experienced_monthly_earnings := experienced_weekly_earnings * monthly_weeks
  let num_experienced_sailors := total_earnings_experienced / experienced_monthly_earnings
  inexperienced_sailors + num_experienced_sailors.to_nat

theorem total_sailors_count : number_of_sailors 5 10 60 34560 (1 / 5) 4 34560 = 17 := 
by 
  apply rfl

end total_sailors_count_l596_596444


namespace sum_of_first_100_digits_l596_596393

def repeating_decimal_expansion : ℝ := 1 / 10101

def first_100_digits_sum (n : ℕ) : ℤ :=
  let block := [0, 0, 0, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9]
  let full_blocks := n / 40
  let remaining_digits := n % 40
  (full_blocks * 180) + (block.take remaining_digits).sum

theorem sum_of_first_100_digits : first_100_digits_sum 100 = 450 :=
by
  sorry

end sum_of_first_100_digits_l596_596393


namespace Vitya_catchup_mom_in_5_l596_596889

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596889


namespace e_to_13pi_2_eq_i_l596_596534

-- Define the problem in Lean 4
theorem e_to_13pi_2_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I :=
by
  sorry

end e_to_13pi_2_eq_i_l596_596534


namespace combination_sum_l596_596473

noncomputable def combination (n k : ℕ) : ℕ :=
nat.choose n k -- Lean's built-in function for combinations

theorem combination_sum {n : ℕ} (h1 : 0 ≤ 17 - n)
                         (h2 : 17 - n ≤ 2 * n)
                         (h3 : 0 ≤ 3 * n)
                         (h4 : 3 * n ≤ 13 + n)
                         (hn : n = 6) :
  combination (2 * n) (17 - n) + combination (13 + n) (3 * n) = 31 :=
by {
  rw hn,
  norm_num,
  sorry
}

end combination_sum_l596_596473


namespace proof_problem_l596_596178

-- Definitions from the problem conditions
def sym_plus (a b : ℕ) := a * b
def sym_minus (a b : ℕ) := a + b
def sym_mul (a b : ℕ) := a / b
def sym_div (a b : ℕ) := a - b
def sym_pow (a b : ℕ) := a - b
def sym_paren (a b : ℕ) := a + b

-- The problem expressed using the new symbolic meanings
def transformed_expr : ℕ :=
  sym_plus 6 (sym_plus 5 (sym_pow 2) + sym_minus 9 (sym_mul 8 (sym_paren 2 3)) - sym_div 25 (sym_pow 4 2))

-- The statement to be proven
theorem proof_problem : transformed_expr = 34 := by sorry

end proof_problem_l596_596178


namespace monotonic_decreasing_interval_l596_596349

noncomputable def f (x : ℝ) : ℝ :=
  x / 4 + 5 / (4 * x) - Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = 5) ∧ (∀ x, 0 < x ∧ x < 5 → (deriv f x < 0)) :=
by
  sorry

end monotonic_decreasing_interval_l596_596349


namespace company_employees_l596_596259

theorem company_employees :
  (∃ (M W : ℕ), 0.60 * M = 176 ∧ M + W = 1200) → ∃ (W : ℕ), W = 907 :=
by
  intro h
  obtain ⟨M, W, h1, h2⟩ := h
  -- Conversion of floating-point to integer in proof requires further steps
  have : W = 907 := 
  sorry
  exact ⟨W, this⟩

end company_employees_l596_596259


namespace closest_perfect_square_to_528_l596_596912

theorem closest_perfect_square_to_528 : 
  ∃ (n : ℕ), n^2 = 529 ∧ 
  (∀ (m : ℕ), m^2 ≠ 528 ∧ m^2 ≠ 529 → (abs (528 - n^2) < abs (528 - m^2))) :=
by
  sorry

end closest_perfect_square_to_528_l596_596912


namespace valerie_money_left_l596_596369

theorem valerie_money_left
  (small_bulb_cost : ℕ)
  (large_bulb_cost : ℕ)
  (num_small_bulbs : ℕ)
  (num_large_bulbs : ℕ)
  (initial_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  num_small_bulbs = 3 →
  num_large_bulbs = 1 →
  initial_money = 60 →
  initial_money - (num_small_bulbs * small_bulb_cost + num_large_bulbs * large_bulb_cost) = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end valerie_money_left_l596_596369


namespace total_books_written_l596_596919

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l596_596919


namespace cosine_value_of_angle_between_AB_and_CD_in_tetrahedron_l596_596357

-- Define the vertices of the parallelogram formed by 6 equilateral triangles
variables {A B C D E F : Type*}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F]

-- Stating the main problem: the cosine value of the angle between AB and CD in the folded structure is 1/2
theorem cosine_value_of_angle_between_AB_and_CD_in_tetrahedron (h: six_equilateral_triangles A B C D E F):
  cos (angle_between AB CD) = 1 / 2 :=
sorry

end cosine_value_of_angle_between_AB_and_CD_in_tetrahedron_l596_596357


namespace angle_of_inclination_of_line_l596_596147

theorem angle_of_inclination_of_line (θ : ℝ) (m : ℝ) (h : |m| = 1) :
  θ = 45 ∨ θ = 135 :=
sorry

end angle_of_inclination_of_line_l596_596147


namespace equal_focal_distances_l596_596593

theorem equal_focal_distances (k : ℝ) (k_pos : 0 < k) (k_less_9 : k < 9) :
    (let c1 := real.sqrt (25 + (9 - k)) in
     let c2 := real.sqrt ((25 - k) + 9) in
     c1 = c2) :=
by {
  sorry
}

end equal_focal_distances_l596_596593


namespace moles_of_NH4I_used_l596_596567

variables (KOH KI NH4I NH3 H2O : Type) (one : ℕ)

-- Chemical reaction equation KOH + NH4I → KI + NH3 + H2O
def reaction := (KOH → KI + NH3 + H2O) = (NH4I → KI + NH3 + H2O)

-- Assuming conversion: 1 mole KOH produces 1 mole KI
axiom reaction_ratio : reaction → (one : ℕ) = one

-- Prove the number of moles of NH4I used
theorem moles_of_NH4I_used (one : ℕ) : (reaction_ratio (one : ℕ)) → (one : ℕ) = one :=
sorry

end moles_of_NH4I_used_l596_596567


namespace probability_second_even_given_first_even_l596_596940

/-- A fair six-sided die is rolled twice in succession. 
    Given that the outcome of the first roll is an even number, 
    what is the probability that the second roll also results in an even number? -/
theorem probability_second_even_given_first_even :
  (P := @classical.Probability (fin 6) (fun i => (i + 1) ∈ {1, 2, 3, 4, 5, 6})) →
  let A := {i : (fin 6) | (i.val % 2) = 0} in
  let B := {j : (fin 6) | (j.val % 2) = 0} in
  ∀ i (hA : i ∈ A), Classical.Probability.toReal (B ∣ A) = 1 / 2 :=
begin
  intros,
  sorry,
end

end probability_second_even_given_first_even_l596_596940


namespace abs_neg_2023_l596_596485

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596485


namespace quadrilateral_area_inequality_equality_condition_l596_596289

theorem quadrilateral_area_inequality 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d) 
  : S ≤ 0.5 * (a * c + b * d) :=
sorry

theorem equality_condition 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d)
  (h_perpendicular : ∃ (α β : ℝ), α = 90 ∧ β = 90) 
  : S = 0.5 * (a * c + b * d) :=
sorry

end quadrilateral_area_inequality_equality_condition_l596_596289


namespace dot_product_of_vectors_l596_596141

variable (a b : Vector ℝ 3)
variable (magnitude_a : ℝ := 6)
variable (magnitude_b : ℝ := 3 * Real.sqrt 3)
variable (theta : ℝ := Real.pi / 6) -- 30 degrees in radians

-- Define the magnitudes
axiom magnitude_definition_a : Real.norm(a) = magnitude_a
axiom magnitude_definition_b : Real.norm(b) = magnitude_b

-- Define the cosine of the angle between vectors
axiom cosine_angle : Real.cos theta = Real.sqrt(3) / 2

theorem dot_product_of_vectors : a • b = 27 :=
by
  sorry

end dot_product_of_vectors_l596_596141


namespace carter_road_trip_l596_596493

theorem carter_road_trip
  (road_trip_hours : ℕ)
  (stretch_interval_hours : ℕ)
  (additional_stops_food : ℕ)
  (additional_stops_gas : ℕ)
  (stop_duration_minutes : ℕ)
  (minutes_per_hour : ℕ) :
  road_trip_hours = 14 →
  stretch_interval_hours = 2 →
  additional_stops_food = 2 →
  additional_stops_gas = 3 →
  stop_duration_minutes = 20 →
  minutes_per_hour = 60 →
  let stretch_stops := road_trip_hours / stretch_interval_hours in
  let total_stops := stretch_stops + additional_stops_food + additional_stops_gas in
  let total_stop_time_minutes := total_stops * stop_duration_minutes in
  let additional_hours := total_stop_time_minutes / minutes_per_hour in
  road_trip_hours + additional_hours = 18 :=
by
  intros
  sorry

end carter_road_trip_l596_596493


namespace maximum_N_value_l596_596228

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596228


namespace find_b_l596_596273

noncomputable def triangle_side_length {a b c : ℝ} (h1 : 2 * b = a + c) (B : ℝ) (h2 : B = π/6) (area : ℝ) (h3 : area = 3/2) : ℝ :=
  b

theorem find_b {a b c : ℝ} (h1 : 2 * b = a + c) (h2 : (π / 6 : ℝ) = π / 6) (h3 : (1/2 * a * c * real.sin (π/6)) = 3 / 2) :
  b = real.sqrt(3) + 1 :=
  sorry

end find_b_l596_596273


namespace total_earmuffs_l596_596471

theorem total_earmuffs {a b c : ℕ} (h1 : a = 1346) (h2 : b = 6444) (h3 : c = a + b) : c = 7790 := by
  sorry

end total_earmuffs_l596_596471


namespace election_votes_l596_596264

theorem election_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℝ) 
  (candidate_A_percentage : ℝ) 
  (candidate_B_percentage : ℝ) 
  (candidate_C_percentage : ℝ) 
  (candidate_D_percentage : ℝ) 
  (valid_percentage : ℝ) 
  (valid_votes : ℕ) 
  (votes_A : ℕ) 
  (votes_B : ℕ) 
  (votes_C : ℕ) 
  (votes_D : ℕ)
  (h1 : total_votes = 800000)
  (h2 : invalid_percentage = 0.20)
  (h3 : candidate_A_percentage = 0.45)
  (h4 : candidate_B_percentage = 0.30)
  (h5 : candidate_C_percentage = 0.12)
  (h6 : candidate_D_percentage = 0.10)
  (h7 : valid_percentage = 0.80)
  (h8 : valid_votes = (valid_percentage * total_votes).toNat)
  (h9 : votes_A = (candidate_A_percentage * valid_votes).toNat)
  (h10 : votes_B = (candidate_B_percentage * valid_votes).toNat)
  (h11 : votes_C = (candidate_C_percentage * valid_votes).toNat)
  (h12 : votes_D = (candidate_D_percentage * valid_votes).toNat) :

  votes_A = 288000 ∧ votes_B = 192000 ∧ votes_C = 76800 ∧ votes_D = 64000 :=
by
  sorry

end election_votes_l596_596264


namespace probability_eta_neg_l596_596746

open MeasureTheory

noncomputable def xi_pdf (λ : ℝ) : ℝ → ℝ := 
λ x => if x < 0 then 0 else λ * exp (-λ * x)

noncomputable def eta (ξ : ℝ) : ℝ := cos ξ

theorem probability_eta_neg (λ : ℝ) (hλ : 0 < λ) :
  let P_eta_neg := ∫ (x : ℝ) in (Set.Ioi 0), xi_pdf λ x ∂(MeasureTheory.volume) * indicator (fun x => cos x < 0) x
  P_eta_neg = (exp(-λ * (π / 2)) - exp(-3 * λ * (π / 2))) / (1 - exp(-2 * λ * π)) :=
by
  sorry

end probability_eta_neg_l596_596746


namespace f_2013_eq_zero_l596_596545

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then Real.log2 (1 - x) else f (x - 1) - f (x - 2)

theorem f_2013_eq_zero : f 2013 = 0 := 
by 
  sorry

end f_2013_eq_zero_l596_596545


namespace ratio_twelfth_term_geometric_sequence_l596_596740

theorem ratio_twelfth_term_geometric_sequence (G H : ℕ → ℝ) (n : ℕ) (a r b s : ℝ)
  (hG : ∀ n, G n = a * (r^n - 1) / (r - 1))
  (hH : ∀ n, H n = b * (s^n - 1) / (s - 1))
  (ratio_condition : ∀ n, G n / H n = (5 * n + 3) / (3 * n + 17)) :
  (a * r^11) / (b * s^11) = 2 / 5 :=
by 
  sorry

end ratio_twelfth_term_geometric_sequence_l596_596740


namespace circle_through_fixed_point_l596_596694

-- Given conditions
def pointE (x y : ℝ) : Prop :=
  dist (x, y) (1, 0) = dist (x, y) ((-1 : ℝ), y)

-- The curve C as a parabola with focus (1,0) and directrix x = -1
def curveC (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- The line l described by y = kx + b that is tangent to the curve
def tangentLine (k b x y : ℝ) : Prop :=
  y = k * x + b ∧ y^2 = 4 * x

-- The intersection point Q
def pointQ (k : ℝ) : ℝ × ℝ :=
  (-1, -k + 1 / k)

-- The tangent point P
def pointP (k : ℝ) : ℝ × ℝ :=
  (1 / k^2, 2 / k)

-- The fixed point on the x-axis
def fixedPoint : ℝ × ℝ :=
  (1, 0)

-- Prove that the circle with PQ as diameter passes through (1,0)
theorem circle_through_fixed_point (k : ℝ) (hk : k ≠ 0) :
  let Q := pointQ k,
      P := pointP k,
      M := fixedPoint in
  dist Q M * dist P M = (dist Q P / 2)^2 :=
begin
  sorry -- Proof required
end

end circle_through_fixed_point_l596_596694


namespace ceiling_and_floor_calculation_l596_596089

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596089


namespace Cab_is_rhombus_with_no_right_angles_l596_596628

-- Define the sets A and B
def A : Set := { x | x is_rhombus ∨ x is_rectangle }
def B : Set := { x | x is_rectangle }

-- Define the condition for a rhombus with no right angles
def is_rhombus_with_no_right_angles (x : shape) : Prop :=
  is_rhombus x ∧ ¬ is_rectangle x

-- Main theorem statement
theorem Cab_is_rhombus_with_no_right_angles : (C A B) = { x | is_rhombus_with_no_right_angles x } :=
by
  sorry

end Cab_is_rhombus_with_no_right_angles_l596_596628


namespace min_elements_with_properties_l596_596672

noncomputable def median (s : List ℝ) : ℝ :=
if h : s.length % 2 = 1 then
  s.nthLe (s.length / 2) (by simp [h])
else
  (s.nthLe (s.length / 2 - 1) (by simp [Nat.sub_right_comm, h, *]) + s.nthLe (s.length / 2) (by simp [h])) / 2

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

noncomputable def mode (s : List ℝ) : ℝ :=
s.groupBy (· = ·)
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (·.snd |>.value)

theorem min_elements_with_properties :
  ∃ s : List ℝ, 
    s.length ≥ 6 ∧ 
    median s = 3 ∧ 
    mean s = 5 ∧ 
    ∃! m, mode s = m ∧ m = 6 :=
by
  sorry

end min_elements_with_properties_l596_596672


namespace find_y_point_l596_596614

theorem find_y_point (θ : ℝ) (y : ℝ) (h1 : sin θ = - (2 * real.sqrt 5) / 5)
                     (h2 : ∃ y, P = (4, y)) : y = -8 :=
sorry

end find_y_point_l596_596614


namespace calculate_savings_l596_596006

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l596_596006


namespace sum_of_roots_l596_596520

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l596_596520


namespace no_valid_permutation_l596_596491

-- Define the problem conditions and question in Lean 4

def isValidPosition (a : Fin 20 → Fin 21) : Prop :=
  (∑ i, a i = 210) ∧
  (∑ i in Finset.range 10, |a i - a (i + 10)| = 55)

theorem no_valid_permutation (a : Fin 20 → Fin 21) : ¬ isValidPosition a := by
  sorry

end no_valid_permutation_l596_596491


namespace gcd_a_m_a_k_eq_a_gcd_m_k_l596_596364

noncomputable def a : ℕ → ℕ
| 0       := 0
| (n + 1) := P (a n)

def P : (ℕ → ℕ) := sorry  -- P is a polynomial with positive integer coefficients

theorem gcd_a_m_a_k_eq_a_gcd_m_k 
  (m k d : ℕ) 
  (h_gcd : Nat.gcd m k = d) : 
  Nat.gcd (a m) (a k) = a d := 
sorry

end gcd_a_m_a_k_eq_a_gcd_m_k_l596_596364


namespace sum_of_roots_l596_596514

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l596_596514


namespace polynomial_p_at_5_l596_596286

noncomputable def polynomial_p (x : ℝ) : ℝ := x^4 + (some : ℝ) * x^3 + (some : ℝ) * x^2 + (some : ℝ) * x + (some : ℝ)

axiom h1 : polynomial_p 1 = 1
axiom h2 : polynomial_p 2 = 2
axiom h3 : polynomial_p 3 = 3
axiom h4 : polynomial_p 4 = 4

theorem polynomial_p_at_5 : polynomial_p 5 = 29 :=
by sorry

end polynomial_p_at_5_l596_596286


namespace vitya_catch_up_time_l596_596861

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596861


namespace max_possible_cities_traversed_l596_596204

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596204


namespace equation_of_line_BC_l596_596591

variables {p x y b c : ℝ}

def parabola := ∀ (x y : ℝ), y^2 = 2 * p * x
def pointA := (1, 2)
def pointF := (p, 0)

def points_on_parabola (x y b c : ℝ) := (y^2 = 2 * p * x) ∧ ((b + c) = -1) ∧ (b^2 + c^2 = 2)

theorem equation_of_line_BC (p : ℝ) 
  (A : pointA)
  (F : pointF)
  (h_parabola : parabola)
  (h_points : points_on_parabola x y b c) : 
  2 * x + y - 1 = 0 :=
sorry

end equation_of_line_BC_l596_596591


namespace calculate_savings_l596_596007

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l596_596007


namespace sin_angle_RPD_l596_596756

-- Problem conditions
variable (A B C D P Q R : Type)
variable [Trapezoid ABCD]
variable (AB CD : ℝ) (BC DA : ℝ)
variable (P : PointIntersection AC BD)
variable (Q : FootOfAltitude D BC)
variable (R : Intersection PQ AB)

-- Given side lengths
variable (h1 : AB = 1)
variable (h2 : BC = 5)
variable (h3 : DA = 5)
variable (h4 : CD = 7)

-- Prove that sin ∠RPD = 4/5
theorem sin_angle_RPD : sin (angle R P D) = 4 / 5 := 
sorry

end sin_angle_RPD_l596_596756


namespace area_BPCQ_computed_l596_596282

-- Given conditions
variables {A B C P Q : Point}
variable {t : ℕ}
variable {I_t : ℕ → Point}

axiom ABC_scalene : scalene_triangle A B C
axiom I0_eq_A : I_t 0 = A
axiom I_t_incenter : ∀ (n : ℕ), incenter (I_t (n - 1)) B C = I_t n
axiom points_on_hyperbola : ∀ (n : ℕ), lies_on_hyperbola (I_t n)
axiom asymptotes_ell1_ell2 : is_asymptote ℓ1 ℓ2
axiom perpendicular_A_BC : is_perpendicular (line_through A) BC ℓ1 P ℓ2 Q
axiom AC2_eq : AC^2 = (12/7)*(AB^2) + 1

-- To prove
theorem area_BPCQ_computed :
  ∃ (j k l m n : ℕ),
  area_quad B P C Q = (j * real.sqrt k + l * real.sqrt m) / n ∧
  nat.gcd j l n = 1 ∧
  10000 * j + 1000 * k + 100 * l + 10 * m + n = 317142 :=
by
-- Proof skipped
  sorry

end area_BPCQ_computed_l596_596282


namespace even_function_value_for_negative_x_l596_596175

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_value_for_negative_x (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_pos : ∀ (x : ℝ), 0 < x → f x = 10^x) :
  ∀ x : ℝ, x < 0 → f x = 10^(-x) :=
by
  sorry

end even_function_value_for_negative_x_l596_596175


namespace max_N_value_l596_596222

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596222


namespace smallest_n_for_condition_l596_596307

def has_invalid_digits (k : ℕ) : Prop :=
  (k.to_digits 10).all (λ d, d ≠ 0 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9)

theorem smallest_n_for_condition : ∃ (n : ℕ), n = 56 ∧ ¬ has_invalid_digits (1994 * n) :=
by {
  existsi 56,
  split,
  refl,
  -- Here we would proceed with the proof that 1994 * 56 does not contain invalid digits
  sorry
}

end smallest_n_for_condition_l596_596307


namespace pattern_black_percentage_l596_596658

theorem pattern_black_percentage 
    (n : ℕ := 7) 
    (r : ℕ → ℕ)
    (radius_seq : ∀ k, (r (k + 1) = r k + (if even k then 3 else 1)))
    (initial_radius : r 0 = 3)
    (colors : ℕ → Bool)
    (alternating_colors : ∀ k, colors k = (k % 2 = 0))
    (black_area : ℝ := 
        pi * (r 6 ^ 2) - pi * (r 5 ^ 2) +
        pi * (r 4 ^ 2) - pi * (r 3 ^ 2) +
        pi * (r 2 ^ 2) - pi * (r 1 ^ 2) +
        pi * (r 0 ^ 2))
    (total_area : ℝ := pi * (r (n-1) ^ 2)) :
    (black_area / total_area) ≈ 0.32 := sorry

end pattern_black_percentage_l596_596658


namespace a_100_result_l596_596132

-- Define the parameters of the arithmetic sequence
variables (a : ℕ → ℤ) (d a1 : ℤ)

-- Sum of the first 9 terms is 27
axiom sum_first_9_terms : (∑ i in range 9, a i) = 27

-- The 10th term is 8
axiom a10_is_8 : a 10 = 8

-- General formula for arithmetic sequence
axiom arithmetic_sequence : ∀ n, a n = a1 + (n * d)

theorem a_100_result : a 100 = 98 :=
by
  sorry

end a_100_result_l596_596132


namespace monotonic_intervals_slope_and_range_l596_596620

noncomputable def f (a x : ℝ) : ℝ := a * log x - a * x - 3

theorem monotonic_intervals (a : ℝ) :
  ( ∀ x : ℝ, 0 < x ∧ x < 1 → deriv (f a) x > 0) ∧ 
  ( ∀ x : ℝ, 1 < x → deriv (f a) x < 0) :=
by sorry

noncomputable def g (a m x : ℝ) : ℝ := a * log x - a * x - 3 - m / x

theorem slope_and_range (m : ℝ) :
  (deriv (f (-1)) 2 = 1 / 2) ∧ 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → deriv (g (-1) m) x ≥ 0) ↔
  (0 ≤ m) :=
by sorry

end monotonic_intervals_slope_and_range_l596_596620


namespace no_real_solutions_l596_596099

theorem no_real_solutions :
  ¬ ∃ (a b c d : ℝ), 
  (a^3 + c^3 = 2) ∧ 
  (a^2 * b + c^2 * d = 0) ∧ 
  (b^3 + d^3 = 1) ∧ 
  (a * b^2 + c * d^2 = -6) := 
by
  sorry

end no_real_solutions_l596_596099


namespace abs_neg_2023_l596_596481

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l596_596481


namespace ceil_floor_difference_l596_596072

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596072


namespace total_books_written_l596_596920

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l596_596920


namespace geom_seq_problem_l596_596288

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions for the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)

def arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

-- The main theorem to prove
theorem geom_seq_problem (h_geom : is_geometric_sequence a)
    (h_sum : sum_first_n_terms S a)
    (h_S3 : S 3 = 14)
    (h_arith : arithmetic_sequence (a 1 + 8) (3 * a 2) (a 3 + 6)) :
    a 1 * a 3 = 16 := by
  sorry

end geom_seq_problem_l596_596288


namespace median_square_length_l596_596272

theorem median_square_length 
  {A B C O : Point}
  (hAOMedian : is_median A O B C)
  (hAC : |AC| = b)
  (hAB : |AB| = c)
  (m_a : ℝ) : 
  m_a = |AO| → m_a^2 = (1/2) * b^2 + (1/2) * c^2 - (1/4) * (|BC|^2) := 
by
  sorry

end median_square_length_l596_596272


namespace people_in_line_l596_596318

theorem people_in_line (front_last_between : ℕ) (people_in_between : front_last_between = 5) : 
    ∃ total_people : ℕ, total_people = 7 :=
by
  let total_people := front_last_between + 2
  have total_eq : total_people = 7 := by
    rw [people_in_between]
    simp
  exact ⟨ total_people, total_eq ⟩

end people_in_line_l596_596318


namespace floral_arrangement_carnations_percentage_l596_596663

theorem floral_arrangement_carnations_percentage :
  ∀ (F : ℕ),
  (1 / 4) * (7 / 10) * F + (2 / 3) * (3 / 10) * F = (29 / 40) * F :=
by
  sorry

end floral_arrangement_carnations_percentage_l596_596663


namespace charlie_keeps_two_lollipops_for_himself_l596_596495

/- Charlie has 57 cherry lollipops, 128 wintergreen lollipops, 14 grape lollipops,
   246 shrimp cocktail lollipops, and 12 raspberry lollipops which he plans to
   distribute equally among his 13 best friends. Prove that Charlie keeps 2 lollipops
   for himself. -/
theorem charlie_keeps_two_lollipops_for_himself :
  let total_lollipops := 57 + 128 + 14 + 246 + 12 in
  let number_of_friends := 13 in
  total_lollipops % number_of_friends = 2 := 
by
  let total_lollipops := 57 + 128 + 14 + 246 + 12
  let number_of_friends := 13
  sorry

end charlie_keeps_two_lollipops_for_himself_l596_596495


namespace vitya_catchup_time_l596_596854

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596854


namespace max_possible_N_in_cities_l596_596242

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596242


namespace circles_intersection_distance_squared_l596_596522

open Real

-- Definitions of circles
def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 25

def circle2 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 9

-- Theorem to prove
theorem circles_intersection_distance_squared :
  ∃ A B : (ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B) ∧
  (dist A B)^2 = 675 / 49 :=
sorry

end circles_intersection_distance_squared_l596_596522


namespace bulbs_in_bedroom_l596_596323

theorem bulbs_in_bedroom :
  ∃ B : ℕ,
  let total := 12 in
  let bathroom := 1 in
  let kitchen := 1 in
  let basement := 4 in
  let garage := (B + bathroom + kitchen + basement) / 2 in
  B + bathroom + kitchen + basement + garage = total ∧ B = 2 :=
sorry

end bulbs_in_bedroom_l596_596323


namespace find_f_at_3_l596_596056

theorem find_f_at_3 : 
  (∀ x, (x ^ (3^5 - 1) - 1) * f x = (x + 1) * (x^2 + 1) * (x^3 + 1) * (x^4 + 1) * ... * (x^(3^4) + 1) - 1) → f 3 = 3 :=
by 
  intro h
  sorry

end find_f_at_3_l596_596056


namespace ceiling_and_floor_calculation_l596_596095

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596095


namespace total_feathers_needed_l596_596114

theorem total_feathers_needed 
  (animals_group1 : ℕ) (feathers_group1 : ℕ)
  (animals_group2 : ℕ) (feathers_group2 : ℕ) 
  (total_feathers : ℕ) :
  animals_group1 = 934 →
  feathers_group1 = 7 →
  animals_group2 = 425 →
  feathers_group2 = 12 →
  total_feathers = 11638 :=
by sorry

end total_feathers_needed_l596_596114


namespace area_MOI_l596_596657

/-!
Given the triangle ABC with vertices at coordinates A = (0,0),
B = (0,5), and C = (12,0), and given the circumcenter O = (6, 2.5)
and incenter I = (2, 2), and point M is such that the circle centered at
M is tangent to sides AC, BC, and the circumcircle of triangle ABC.
Then, the area of triangle MOI is 7/2.
-/
noncomputable def point : Type := { x : ℝ // true }

def A : point := ⟨0,0⟩
def B : point := ⟨0,5⟩
def C : point := ⟨12,0⟩

def O : point := ⟨6,2.5⟩
def I : point := ⟨2,2⟩
def M : point := ⟨4,4⟩

def area (a b c : point) : ℝ :=
  abs (a.1 * b.2 + b.1 * c.2 + c.1 * a.2 - a.2 * b.1 - b.2 * c.1 - c.2 * a.1) / 2

theorem area_MOI :
  area M O I = 7 / 2 := 
sorry

end area_MOI_l596_596657


namespace parallelogram_area_72_l596_596327

def parallelogram_area (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_72 :
  parallelogram_area 12 6 = 72 :=
by
  sorry

end parallelogram_area_72_l596_596327


namespace q_sum_l596_596296

noncomputable def q (x : ℝ) := sorry

theorem q_sum :
  (∃ q : ℝ → ℝ, monic (q : polynomial ℝ) ∧ polynomial.degree q = 5 ∧
    q 1 = -3 ∧ q 2 = -6 ∧ q 3 = -9 ∧ q 4 = -12) →
  q 0 + q 5 = 124 :=
sorry

end q_sum_l596_596296


namespace graph_of_equation_is_two_lines_l596_596397

-- define the condition
def equation_condition (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

-- state the theorem
theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, equation_condition x y → (x = 0) ∨ (y = 0) :=
by
  intros x y h
  -- proof here
  sorry

end graph_of_equation_is_two_lines_l596_596397


namespace basketball_game_first_half_score_l596_596661

theorem basketball_game_first_half_score :
    ∃ (a b e r : ℕ), 
    let sa1 := a in 
    let sa2 := ar in 
    let sb1 := b in 
    let sb2 := b + e in 
    a = b ∧
    a * (1 + r + r^2 + r^3) = 4 * b + 6 * e + 2 ∧ 
    a + ar + b + (b + e) = 41 ∧
    a + ar + ar^2 + ar^3 ≤ 80 ∧ 
    b + (b + e) + (b + 2 * e) + (b + 3 * e) ≤ 80 := sorry

end basketball_game_first_half_score_l596_596661


namespace ella_strawberries_l596_596636

theorem ella_strawberries :
  ∃ x : ℕ, (x - 8) = (x / 3) ∧ x = 12 :=
begin
  sorry
end

end ella_strawberries_l596_596636


namespace minimal_elements_for_conditions_l596_596681

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, Nat.odd_iff_not_even.mpr (List.length_pos_of_ne_nil sorted).2])
  else
    let a := sorted.nth_le (sorted.length / 2 - 1) (by simp [Nat.sub_pos_of_lt (Nat.div_lt_self (List.length_pos_of_ne_nil sorted)].)
    let b := sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, List.length_pos_of_ne_nil sorted])
    (a + b) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def mode (s : List ℝ) : Option ℝ :=
  s.group_by id (· = ·).max_by (·.length).map (·.head)

def satisfies_conditions (s : List ℝ) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = some 6

theorem minimal_elements_for_conditions : ∀ s : List ℝ, satisfies_conditions s → s.length ≥ 6 :=
by
  intro s h
  sorry

end minimal_elements_for_conditions_l596_596681


namespace monotonically_decreasing_interval_cos_l596_596565

theorem monotonically_decreasing_interval_cos (k : ℤ) : 
    ∃ (a b : ℝ), 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b → 
    ∀ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → cos (π / 3 - 2 * x₁) ≥ cos (π / 3 - 2 * x₂)) ∧ 
    a = k * π + π / 6 ∧ b = k * π + 2 * π / 3 :=
by
  sorry

end monotonically_decreasing_interval_cos_l596_596565


namespace hyperbola_ellipse_equations_l596_596946

theorem hyperbola_ellipse_equations 
  (F1 F2 P : ℝ × ℝ) 
  (hF1 : F1 = (0, -5))
  (hF2 : F2 = (0, 5))
  (hP : P = (3, 4)) :
  (∃ a b : ℝ, a^2 = 40 ∧ b^2 = 16 ∧ 
    ∀ x y : ℝ, (y^2 / 40 + x^2 / 15 = 1 ↔ y^2 / a^2 + x^2 / (a^2 - 25) = 1) ∧
    (y^2 / 16 - x^2 / 9 = 1 ↔ y^2 / b^2 - x^2 / (25 - b^2) = 1)) :=
sorry

end hyperbola_ellipse_equations_l596_596946


namespace distance_to_destination_l596_596305

theorem distance_to_destination (x : ℕ) 
    (condition_1 : True)  -- Manex is a tour bus driver. Ignore in the proof.
    (condition_2 : True)  -- Ignores the fact that the return trip is using a different path.
    (condition_3 : x / 30 + (x + 10) / 30 + 2 = 6) : 
    x = 55 :=
sorry

end distance_to_destination_l596_596305


namespace tower_count_l596_596936

theorem tower_count : 
  let total_red := 2
  let total_blue := 4
  let total_green := 5
  let total_height := 7
  ∃ count, 
  (total_red + total_blue+ total_green = 11) ∧
  (count = 420) :=
begin
  let total_red := 2,
  let total_blue := 4,
  let total_green := 5,
  let tower_height := 7,
  
  -- number of distinct ways to build a tower of 7 cubes
  let num_choices := (finchoose 11 7),
  have h : num_choices = 330,
  
  sorry
end

end tower_count_l596_596936


namespace incorrect_description_l596_596158

def inverse_proportion_function (x : ℝ) : ℝ := 6 / x

theorem incorrect_description (x : ℝ): 
  ¬∀ x > 0, ((inverse_proportion_function x) < (inverse_proportion_function (x + 1))) :=
sorry

end incorrect_description_l596_596158


namespace ratio_boys_to_girls_l596_596353

def total_students : Nat := 68
def girls : Nat := 28
def boys : Nat := total_students - girls

theorem ratio_boys_to_girls : (boys : girls) = (10 : 7) :=
by
  have h1 : boys = total_students - girls := rfl
  have h2 : boys = 40 := by rw [h1]; norm_num
  have h3 : Int.gcd 40 28 = 4 := by norm_num
  have h4 : (40 / 4 : 28 / 4) = (10 : 7) := by congr; norm_num
  rw [h2] at h4
  exact h4

end ratio_boys_to_girls_l596_596353


namespace vitya_catches_up_in_5_minutes_l596_596847

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596847


namespace highest_possible_salary_l596_596261

theorem highest_possible_salary 
    (n : ℕ) (min_salary max_team_salary : ℕ) 
    (team_size : n = 18) 
    (min_wage : min_salary = 20000) 
    (salary_cap : max_team_salary = 800000) : 
    ∃ s, s = 460000 ∧ 
          (∀ p ∈ (finset.range 17).map (λ _, min_salary), p >= min_salary) ∧ 
          (s + (finset.sum (finset.range 17).map (λ _, min_salary)) ≤ max_team_salary) := 
sorry

end highest_possible_salary_l596_596261


namespace statement_A_statement_B_statement_D_l596_596400

theorem statement_A (x : ℝ) (hx : x > 1) : 
  ∃(y : ℝ), y = 3 * x + 1 / (x - 1) ∧ y = 2 * Real.sqrt 3 + 3 := 
  sorry

theorem statement_B (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃(z : ℝ), z = 1 / (x + 1) + 2 / y ∧ z = 9 / 2 := 
  sorry

theorem statement_D (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃(k : ℝ), k = (x^2 + y^2 + z^2) / (3 * x * y + 4 * y * z) ∧ k = 2 / 5 := 
  sorry

end statement_A_statement_B_statement_D_l596_596400


namespace darnell_fabric_left_l596_596984

/-- Darnell has 1000 square feet of fabric initially. 
He makes several types of flags with specified dimensions. 
Given the number of each type of flag made, we want to prove 
the remaining fabric left. 
-/
theorem darnell_fabric_left (f_total : ℕ) (sq_count : ℕ) (sq_area : ℕ) (wide_count : ℕ) (wide_area : ℕ) (tall_count : ℕ) (tall_area : ℕ) :
  f_total = 1000 → 
  sq_area = 16 → 
  wide_area = 15 → 
  tall_area = 15 → 
  sq_count = 16 → 
  wide_count = 20 → 
  tall_count = 10 → 
  let fabric_used := (sq_count * sq_area) + (wide_count * wide_area) + (tall_count * tall_area) in
    let fabric_left := f_total - fabric_used in
      fabric_left = 294 :=
by intros; sorry

end darnell_fabric_left_l596_596984


namespace prob_both_correct_prob_at_least_one_correct_prob_exactly_3_correct_l596_596843

-- Define probabilities for satellite A and B
axiom P_A : ℚ
axiom P_B : ℚ
axiom independent_A_B : IndepEvents P_A P_B
axiom P_A_val : P_A = 4/5
axiom P_B_val : P_B = 3/4

-- Problem I
theorem prob_both_correct : P(⋂₁ (λ(ω : event_univ), A ω ∧ B ω)) = 3/5
  := by
     sorry

-- Problem II
theorem prob_at_least_one_correct : 1 - (1 - P_A) * (1 - P_B) = 19/20
  := by
     sorry

-- Define binomial probability conditions
axiom p : ℚ
axiom n : ℕ
axiom k : ℕ
axiom trials_independence : IndepTrials p n
axiom p_val : p = 4/5
axiom n_val : n = 4
axiom k_val : k = 3

-- Problem III
theorem prob_exactly_3_correct : binomial_pmf n k p = 256/625
  := by
     sorry

end prob_both_correct_prob_at_least_one_correct_prob_exactly_3_correct_l596_596843


namespace eight_diamond_five_l596_596356

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem eight_diamond_five : diamond 8 5 = 160 :=
by sorry

end eight_diamond_five_l596_596356


namespace sum_of_roots_l596_596518

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l596_596518


namespace vitya_catch_up_l596_596884

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596884


namespace projection_of_w_l596_596953

-- Define the given vectors
def u : ℝ × ℝ := (2, -3)
def v : ℝ × ℝ := (1, -3 / 2)
def w : ℝ × ℝ := (3, -1)

-- Define the given condition
def is_projection (x y : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, y = k • x

-- State the theorem
theorem projection_of_w :
  is_projection u v →
  ∃ k : ℝ, (k • u) = (18 / 13, -27 / 13) :=
sorry

end projection_of_w_l596_596953


namespace value_of_a_for_perfect_square_trinomial_l596_596177

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) (x y : ℝ) :
  (∃ b : ℝ, (x + b * y) ^ 2 = x^2 + a * x * y + y^2) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l596_596177


namespace num_mappings_P_to_Q_l596_596596

def P : Set := {a, b}
def Q : Set := {-1, 0, 1}

theorem num_mappings_P_to_Q : ∃(n : ℕ), n = 9 ∧ ∃(f : P → Q), bijective f := 󠅓.their sorry

end num_mappings_P_to_Q_l596_596596


namespace product_of_card_sums_even_l596_596413

theorem product_of_card_sums_even :
  ∃ (cards : Fin 99 → ℕ × ℕ),
    (∀ (i : Fin 99), cards i.1 ∈ Finset.range 100 ∧ cards i.2 ∈ Finset.range 100) →
    (∃ (sums : Fin 99 → ℕ), 
      (∀ (i : Fin 99), sums i = cards i.1 + cards i.2) ∧ 
      (even (∏ i, sums i))) :=
sorry

end product_of_card_sums_even_l596_596413


namespace jebb_take_home_pay_l596_596723

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l596_596723


namespace product_sequence_eq_670_l596_596905

theorem product_sequence_eq_670 : 
  let seq := (fun n => (n + 4)/(n + 3)) in
  (finset.range 2007).prod (λ k, seq k) = 670 := 
by 
  sorry

end product_sequence_eq_670_l596_596905


namespace imaginary_part_of_z_l596_596126

variable {z : ℂ}

def on_line (z : ℂ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ z = x + 2 * x * I

theorem imaginary_part_of_z (h1 : on_line z) (h2 : |z| = real.sqrt 5) : z.im = 2 := by
  sorry

end imaginary_part_of_z_l596_596126


namespace ferry_tourists_total_l596_596941

theorem ferry_tourists_total :
  let initial_time := 9
  let last_time := 15
  let initial_tourists := 120
  let decrement_per_trip := 3
  let number_of_trips := last_time - initial_time + 1
  let tourists_in_nth_trip (n : ℕ) := initial_tourists - (n * decrement_per_trip)
  let total_tourists := ∑ i in Finset.range number_of_trips, tourists_in_nth_trip i
  in total_tourists = 777 :=
by
  sorry

end ferry_tourists_total_l596_596941


namespace distance_between_lines_l596_596335

theorem distance_between_lines : 
  let A := 3
  let B := 4
  let C1 := -12
  let C2 := 3
  let distance := Real.abs (C2 - C1) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = 3 :=
by
  sorry

end distance_between_lines_l596_596335


namespace quadratic_discriminant_eq_l596_596363

theorem quadratic_discriminant_eq (a b c n : ℤ) (h_eq : a = 3) (h_b : b = -8) (h_c : c = -5)
  (h_discriminant : b^2 - 4 * a * c = n) : n = 124 := 
by
  -- proof skipped
  sorry

end quadratic_discriminant_eq_l596_596363


namespace maximum_N_value_l596_596226

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596226


namespace geometric_sequence_find_a_n_l596_596130

noncomputable def a : ℕ → ℝ
| 0     := 2 / 3
| (n+1) := 2 * a n / (a n + 1)

-- (1) Prove that the sequence {1/a_n - 1} is a geometric sequence.
theorem geometric_sequence :
  ∃ r b : ℝ, b = (1 / a 0 - 1) ∧ ∀ n, (1 / a (n+1) - 1) = r * (1 / a n - 1) :=
begin
  sorry
end

-- (2) Find a_n.
theorem find_a_n (n : ℕ) :
  a n = 2^n / (2^n + 1) :=
begin
  sorry
end

end geometric_sequence_find_a_n_l596_596130


namespace ratio_correct_l596_596768

def my_age : ℕ := 35
def son_age_next_year : ℕ := 8
def son_age_now : ℕ := son_age_next_year - 1
def ratio_of_ages : ℕ := my_age / son_age_now

theorem ratio_correct : ratio_of_ages = 5 :=
by
  -- Add proof here
  sorry

end ratio_correct_l596_596768


namespace stratified_sampling_11th_grade_representatives_l596_596935

theorem stratified_sampling_11th_grade_representatives 
  (students_10th : ℕ)
  (students_11th : ℕ)
  (students_12th : ℕ)
  (total_rep : ℕ)
  (total_students : students_10th + students_11th + students_12th = 5000)
  (Students_10th : students_10th = 2500)
  (Students_11th : students_11th = 1500)
  (Students_12th : students_12th = 1000)
  (Total_rep : total_rep = 30) : 
  (9 : ℕ) = (3 : ℚ) / (10 : ℚ) * (30 : ℕ) :=
sorry

end stratified_sampling_11th_grade_representatives_l596_596935


namespace inequality_does_not_hold_l596_596173

theorem inequality_does_not_hold (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬(a + b < 2 * real.sqrt (a * b)) :=
sorry

end inequality_does_not_hold_l596_596173


namespace find_angle_BAC_l596_596700

open Real
open Nat

theorem find_angle_BAC (A B C D E : Type) [is_triangle A B C] [is_triangle E B D]
  (h1 : angle D A E = 37) (h2 : angle D E A = 37) (h3 : congruent A B C E B D) 
  : angle B A C = 7 :=
sorry

end find_angle_BAC_l596_596700


namespace vector_relation_l596_596151

variables {V : Type*} [add_comm_group V] [module ℝ V] 
variables (a b : V) 

theorem vector_relation
  (h1 : ∀ k : ℝ, a ≠ k • b)
  (h2 : ∀ AB BC CD AD : V, AB = a + 2 • b → BC = -4 • a - b → CD = -5 • a - 3 • b → AD = 2 • BC → AD = AB + BC + CD) :
  ∃ AD, AD = 2 • (-4 • a - b) := 
by { sorry }

end vector_relation_l596_596151


namespace max_possible_value_l596_596255

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596255


namespace maximum_N_value_l596_596232

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596232


namespace max_possible_value_l596_596252

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596252


namespace min_value_of_expression_l596_596140

theorem min_value_of_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 1 < b) (h₃ : a + b = 2) :
  4 / a + 1 / (b - 1) = 9 := 
sorry

end min_value_of_expression_l596_596140


namespace part_one_solution_set_part_two_m_range_l596_596622

theorem part_one_solution_set (m : ℝ) (x : ℝ) (h : m = 0) : ((m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

theorem part_two_m_range (m : ℝ) : (∀ x : ℝ, (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
by
  sorry

end part_one_solution_set_part_two_m_range_l596_596622


namespace closest_perfect_square_to_528_l596_596913

theorem closest_perfect_square_to_528 : 
  ∃ (n : ℕ), n^2 = 529 ∧ 
  (∀ (m : ℕ), m^2 ≠ 528 ∧ m^2 ≠ 529 → (abs (528 - n^2) < abs (528 - m^2))) :=
by
  sorry

end closest_perfect_square_to_528_l596_596913


namespace normal_intersection_constant_l596_596811

-- Define the parabola and points
def parabola (x : ℝ) : ℝ := x ^ 2

-- Define a point on the parabola
def point_on_parabola (x0 : ℝ) : ℝ × ℝ := (x0, parabola x0)

-- Define the y-coordinate of the intersection of the normal with the y-axis
def normal_intersect_y_axis (x0 : ℝ) : ℝ :=
  let y0 := parabola x0 in
  (1 / 2) + y0

-- Theorem to prove the difference is constant
theorem normal_intersection_constant (x0 : ℝ) :
  let y0 := parabola x0
  let y1 := normal_intersect_y_axis x0
  in
  y1 - y0 = (1 / 2) :=
by
  sorry

end normal_intersection_constant_l596_596811


namespace option_B_is_basis_l596_596577

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (a b c : V)

-- Given basis
axiom basis_set : LinearIndependent ℝ ![a, b, c]

-- Verify Option B is also a basis
theorem option_B_is_basis :
  LinearIndependent ℝ ![a + 2•b, b, a - c] ∧ Submodule.span ℝ ![a + 2•b, b, a - c] = ⊤ :=
by
  sorry

end option_B_is_basis_l596_596577


namespace complete_the_square_l596_596160

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end complete_the_square_l596_596160


namespace minimum_value_l596_596747

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x : ℝ, 
    (x = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2) ∧ 
    (∀ y, y = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2 → x ≤ y) ∧ 
    x = 7 :=
by 
  sorry

end minimum_value_l596_596747


namespace abs_neg_2023_l596_596489

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596489


namespace parametric_plane_equiv_l596_596951

/-- Define the parametric form of the plane -/
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + s - t, 2 - s, 3 - 2*s + 2*t)

/-- Define the equation of the plane in standard form -/
def plane_equation (x y z : ℝ) : Prop :=
  2 * x + z - 5 = 0

/-- The theorem stating that the parametric form corresponds to the given plane equation -/
theorem parametric_plane_equiv :
  ∃ x y z s t,
    (x, y, z) = parametric_plane s t ∧ plane_equation x y z :=
by
  sorry

end parametric_plane_equiv_l596_596951


namespace PQRS_product_l596_596606

def P : ℝ := (Real.sqrt 2010 + Real.sqrt 2011)
def Q : ℝ := (-Real.sqrt 2010 - Real.sqrt 2011)
def R : ℝ := (Real.sqrt 2010 - Real.sqrt 2011)
def S : ℝ := (Real.sqrt 2011 - Real.sqrt 2010)

theorem PQRS_product : P * Q * R * S = 1 := by
  sorry

end PQRS_product_l596_596606


namespace vitya_catch_up_l596_596886

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596886


namespace melanie_total_dimes_l596_596764

theorem melanie_total_dimes (original_dimes : ℕ) (dimes_from_dad : ℕ) (dimes_from_mom : ℕ) (total_dimes : ℕ) :
  original_dimes = 19 → dimes_from_dad = 39 → dimes_from_mom = 25 → total_dimes = 83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end melanie_total_dimes_l596_596764


namespace probability_2x_less_y_equals_one_over_eight_l596_596437

noncomputable def probability_2x_less_y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 3 * 1.5
  let area_rectangle : ℚ := 6 * 3
  area_triangle / area_rectangle

theorem probability_2x_less_y_equals_one_over_eight :
  probability_2x_less_y_in_rectangle = 1 / 8 :=
by
  sorry

end probability_2x_less_y_equals_one_over_eight_l596_596437


namespace exists_ai_l596_596281

open BigOperators

variable {n : ℕ} (x : Fin n → ℝ)

theorem exists_ai (x_nonneg : ∀ i, 0 ≤ x i) (x_sum_one : ∑ i, x i = 1) :
  ∃ (a : Fin n → ℕ), 2 ≤ ∑ i, a i * x i ∧ ∑ i, a i * x i ≤ 2 + 2 / (3 ^ n - 1) ∧
                      ∀ i, a i ∈ {0, 1, 2, 3, 4} ∧ ∃ i, a i ≠ 2 := by
  sorry

end exists_ai_l596_596281


namespace incenter_of_triangle_GQM_coincides_with_D_l596_596294

-- Define the basic elements and conditions of the problem
variables {A B C I D E F P M G Q : Point}
variable {ABC : Triangle}

-- Given conditions as hypotheses
hypotheses
(h1 : scalene_triangle ABC)
(h2 : incenter ABC I)
(h3 : incircle_touches ABC D E F)
(h4 : P = foot_altitude D (segment E F))
(h5 : M = midpoint (segment B C))
(h6 : AP = ray_intersect_circumcircle (segment A P) ABC G)
(h7 : IP = ray_intersect_circumcircle (segment I P) ABC Q)

-- Proof goal
theorem incenter_of_triangle_GQM_coincides_with_D :
  incenter (triangle G Q M) = D :=
sorry

end incenter_of_triangle_GQM_coincides_with_D_l596_596294


namespace min_elements_with_conditions_l596_596668

theorem min_elements_with_conditions (s : Finset ℝ) 
  (h_median : (s.sort (≤)).median = 3) 
  (h_mean : s.mean = 5)
  (h_mode : s.mode = 6) : 
  s.card >= 6 :=
sorry

end min_elements_with_conditions_l596_596668


namespace max_possible_value_l596_596257

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596257


namespace parabola_intersection_with_y_axis_l596_596797

theorem parabola_intersection_with_y_axis :
  (∃ y, (0, y) ∧ (y = 2)) ∧ (0^2 - 3*0 + 2 = 2) :=
by
  sorry

end parabola_intersection_with_y_axis_l596_596797


namespace scalar_p_l596_596990

variable (ℝ : Type) [field ℝ]

variables (u v w : ℝ × ℝ × ℝ)

theorem scalar_p (p : ℝ) : 
  (∀ (u v w : ℝ × ℝ × ℝ), u + v + w = (0, 0, 0) → 
  p * (v × u) + 2 * (v × w) + 3 * (w × u) = (0, 0, 0)) → 
  p = 5 := sorry

end scalar_p_l596_596990


namespace max_possible_N_in_cities_l596_596244

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596244


namespace max_N_value_l596_596224

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596224


namespace ceil_floor_diff_l596_596080

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596080


namespace domain_f_real_l596_596155

noncomputable def f (a x : ℝ) : ℝ := log (x^2 - 2*(2*a-1)*x + 8) / log (1/2)

theorem domain_f_real (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*(2*a-1)*x + 8 > 0) ↔ (1/2 - sqrt 2 < a ∧ a < 1/2 + sqrt 2) := by
  sorry

end domain_f_real_l596_596155


namespace zero_point_exists_l596_596564

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - (1 / 2) ^ x

theorem zero_point_exists :
  (0 < 1) ∧ (f 0 < 0) ∧ (f 1 > 0) → ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end zero_point_exists_l596_596564


namespace angle_equality_l596_596295

variables {A B C D O L M N : Point}

-- Assuming O is the center of the circumsphere of tetrahedron ABCD.
def isCircumcenter (O A B C D : Point) : Prop := 
sorry  -- Definition of O being the circumcenter

-- Defining the midpoint conditions
def isMidpoint (X Y M : Point) : Prop := 
sorry  -- M is the midpoint of line segment XY

-- Given conditions in the problem
axiom AB_BC_eq_AD_CD : (distance AB + distance BC = distance AD + distance CD)
axiom BC_CA_eq_BD_AD : (distance BC + distance CA = distance BD + distance AD)
axiom CA_AB_eq_CD_BD : (distance CA + distance AB = distance CD + distance BD)

-- The main proof goal
theorem angle_equality
    (O_is_circumcenter : isCircumcenter O A B C D)
    (L_midpoint_BC : isMidpoint B C L)
    (M_midpoint_CA : isMidpoint C A M)
    (N_midpoint_AB : isMidpoint A B N)
    (h1 : AB + BC = AD + CD)
    (h2 : BC + CA = BD + AD)
    (h3 : CA + AB = CD + BD) :
    angle L O M = angle M O N ∧ angle M O N = angle N O L :=
sorry

end angle_equality_l596_596295


namespace tan_arccot_3_5_l596_596049

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596049


namespace total_nails_needed_l596_596116

-- Given conditions
def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

-- Prove the total number of nails required
theorem total_nails_needed : nails_per_plank * number_of_planks = 32 :=
by
  sorry

end total_nails_needed_l596_596116


namespace angle_between_planes_is_pi_div_4_l596_596558

noncomputable def normal_vector (a b c d : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def cos_angle_between (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem angle_between_planes_is_pi_div_4 :
  let n1 := normal_vector 3 (-1) 0 5,
      n2 := normal_vector 2 1 0 3,
      cos_phi := cos_angle_between n1 n2
  in Real.arccos cos_phi = Real.pi / 4 :=
by
  let n1 := normal_vector 3 (-1) 0 5
  let n2 := normal_vector 2 1 0 3
  let cos_phi := cos_angle_between n1 n2
  show Real.arccos cos_phi = Real.pi / 4 
  sorry

end angle_between_planes_is_pi_div_4_l596_596558


namespace sum_of_roots_l596_596521

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l596_596521


namespace Vitya_catches_mother_l596_596868

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596868


namespace max_N_value_l596_596218

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596218


namespace tan_arccot_eq_5_div_3_l596_596026

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596026


namespace pair_with_15_l596_596813

theorem pair_with_15 (s : List ℕ) (h : s = [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]) :
  ∃ (t : List (ℕ × ℕ)), (∀ (x y : ℕ), (x, y) ∈ t → x + y = 62) ∧ (15, 47) ∈ t := by
  sorry

end pair_with_15_l596_596813


namespace flagpole_height_l596_596431

theorem flagpole_height :
  ∃ (AB AC AD DE DC : ℝ), 
    AC = 5 ∧
    AD = 3 ∧ 
    DE = 1.8 ∧
    DC = AC - AD ∧
    AB = (DE * AC) / DC ∧
    AB = 4.5 :=
by
  exists 4.5, 5, 3, 1.8, 2
  simp
  sorry

end flagpole_height_l596_596431


namespace barry_smallest_total_amount_l596_596492

theorem barry_smallest_total_amount 
  (coin_values : list ℝ := [2.00, 1.00, 0.25, 0.10, 0.05])
  (total_coins : ℕ := 12)
  (at_least_one_each : ∀ (v : ℝ), v ∈ coin_values → ∃ n : ℕ, n ≥ 1) :
  real :=
  -- Given the conditions, prove that the smallest total amount is 3.75
begin
  have h₁ : real := coin_values.sum,
  have h₂ : ℕ := total_coins - coin_values.length,
  have h₃ : real := h₂ * 0.05,
  exact h₁ + h₃,
  sorry
end

#eval barry_smallest_total_amount == 3.75

end barry_smallest_total_amount_l596_596492


namespace range_of_m_l596_596185

-- Definitions
def positive_reals := {x : ℝ // 0 < x}

-- Problem conditions
variables {x y m : ℝ}
variables (hx : x ∈ positive_reals) (hy : y ∈ positive_reals)
variables (h_condition : 1/x + 4/y = 2)

-- Proof goal
theorem range_of_m (m : ℝ) (hx : 0 < x) (hy : 0 < y) (h_condition : 1/x + 4/y = 2) :
  (x + y / 4 < m^2 - m) ↔ (m < -1 ∨ 2 < m) :=
sorry

end range_of_m_l596_596185


namespace e_to_13pi_2_eq_i_l596_596536

-- Define the problem in Lean 4
theorem e_to_13pi_2_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I :=
by
  sorry

end e_to_13pi_2_eq_i_l596_596536


namespace roots_of_quadratic_l596_596928

theorem roots_of_quadratic (b c : ℝ) (K L M : ℝ × ℝ)
  (h_intersect_x : K = (0, 0) ∧ L = (a, 0) ∧ M = (b, 0)) 
  (h_KL_KM_eq : dist K L = dist K M)
  (h_angle_120 : angle L K M = 120): 
  {r : ℝ // (r = 1/2) ∨ (r = 3/2)} :=
sorry

end roots_of_quadratic_l596_596928


namespace circle_and_tangent_line_l596_596587

noncomputable def standard_eq_circle : (ℝ × ℝ) → ℝ → (ℝ → ℝ → Bool) :=
  λ C r x y => (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2
  
noncomputable def is_tangent (l: ℝ → ℝ → Bool) (C: ℝ × ℝ) (r: ℝ) : Prop :=
  ∃ p: ℝ × ℝ, l p.1 p.2 ∧ (standard_eq_circle C r p.1 p.2) ∧ 
  ∃ m: ℝ, (m = -(p.1 - C.1)/(p.2 - C.2)) ∧ ∀ x y, l x y → y = m * (x - p.1) + p.2

theorem circle_and_tangent_line:
  ∃ C: ℝ × ℝ, ∃ r: ℝ, 
    (C.1 - 1)^2 + (C.2 - 1)^2 = r^2 ∧
    (C.1 - 2)^2 + (C.2 + 2)^2 = r^2 ∧
    C.1 - C.2 + 1 = 0 ∧
    standard_eq_circle C r = λ x y, (x + 3) ^ 2 + (y + 2) ^ 2 = 25 ∧
    is_tangent (λ x y, 4 * x + 3 * y - 7 = 0) C 5 :=
sorry

end circle_and_tangent_line_l596_596587


namespace neg_p_l596_596626

def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p : 
  (¬ (∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∃ x : ℝ, f a x = 0)) ↔ 
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f a x ≠ 0) := 
by sorry

end neg_p_l596_596626


namespace minimum_value_of_f_l596_596416

noncomputable def f (x a : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + a

theorem minimum_value_of_f (a : ℝ) (h_max : f 0 a = 3) :
  ∃ x ∈ set.Icc (-2 : ℝ) 2, f x a = -37 :=
begin
  sorry
end

end minimum_value_of_f_l596_596416


namespace arman_two_weeks_earnings_l596_596736

theorem arman_two_weeks_earnings :
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  total_earnings = 770 := 
by
  -- Definitions based on conditions
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  sorry

end arman_two_weeks_earnings_l596_596736


namespace grade_point_average_ratio_l596_596341

theorem grade_point_average_ratio (A B : ℕ) 
  (h₁ : A.average = 60) 
  (h₂ : B.average = 66) 
  (h₃ : (A + B).average = 64) : 
  A / (A + B) = 1 / 3 := 
sorry

end grade_point_average_ratio_l596_596341


namespace rhombus_area_l596_596956

theorem rhombus_area (Q : ℝ) : 30° = π / 6 ∧ (exists r : ℝ, Q = π * r ^ 2) → S = 8 * Q / π := sorry

end rhombus_area_l596_596956


namespace tan_arccot_l596_596020

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596020


namespace round_robin_teams_l596_596375

theorem round_robin_teams (x : ℕ) (h : x ≠ 0) :
  (x * (x - 1)) / 2 = 15 → ∃ n : ℕ, x = n :=
by
  sorry

end round_robin_teams_l596_596375


namespace friend_cutoff_fraction_l596_596840

-- Definitions based on problem conditions
def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def days_biking : ℕ := 1
def days_bus : ℕ := 3
def days_friend : ℕ := 1
def total_weekly_commuting_time : ℕ := 160

-- Lean theorem statement
theorem friend_cutoff_fraction (F : ℕ) (hF : days_biking * biking_time + days_bus * bus_time + days_friend * F = total_weekly_commuting_time) :
  (biking_time - F) / biking_time = 2 / 3 :=
by
  sorry

end friend_cutoff_fraction_l596_596840


namespace percentage_of_annual_decrease_is_10_l596_596788

-- Define the present population and future population
def P_present : ℕ := 500
def P_future : ℕ := 450 

-- Calculate the percentage decrease
def percentage_decrease (P_present P_future : ℕ) : ℕ :=
  ((P_present - P_future) * 100) / P_present

-- Lean statement to prove the percentage decrease is 10%
theorem percentage_of_annual_decrease_is_10 :
  percentage_decrease P_present P_future = 10 :=
by
  unfold percentage_decrease
  sorry

end percentage_of_annual_decrease_is_10_l596_596788


namespace part1_part2_l596_596597

variable {α : Type*}
def A : Set ℝ := {x | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1)
theorem part1 : B 5 ∩ A = {x | 6 ≤ x ∧ x < 9} := 
sorry

-- Part (2)
theorem part2 (m : ℝ): A ∩ B m = B m ↔ m < 5 :=
sorry

end part1_part2_l596_596597


namespace trajectory_parabola_l596_596180

open Real

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def point_on_line (P : ℝ × ℝ) (x_line : ℝ) : ℝ :=
  abs (P.1 - x_line)

theorem trajectory_parabola (P : ℝ × ℝ) (F : ℝ × ℝ) (line_x : ℝ) (h : distance P F = point_on_line P line_x) :
  ∃ a b c : ℝ, (P.1 * P.1 + a * P.1 + b = P.2 * P.2 + c) :=
sorry

end trajectory_parabola_l596_596180


namespace solve_system_l596_596781

theorem solve_system : 
  ∃ x y : ℚ, (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ∧ x = 1/2 ∧ y = -3 :=
by
  sorry

end solve_system_l596_596781


namespace no_integer_solution_l596_596989

theorem no_integer_solution (x : ℤ) : ¬ (x + 12 > 15 ∧ -3 * x > -9) :=
by {
  sorry
}

end no_integer_solution_l596_596989


namespace painted_faces_cube_eq_54_l596_596430

def painted_faces (n : ℕ) : ℕ :=
  if n = 5 then (3 * 3) * 6 else 0

theorem painted_faces_cube_eq_54 : painted_faces 5 = 54 := by {
  sorry
}

end painted_faces_cube_eq_54_l596_596430


namespace longest_side_of_triangle_l596_596188

theorem longest_side_of_triangle :
  ∀ (A B C a b : ℝ),
    B = 2 * π / 3 →
    C = π / 6 →
    a = 5 →
    A = π - B - C →
    (b / (Real.sin B) = a / (Real.sin A)) →
    b = 5 * Real.sqrt 3 :=
by
  intros A B C a b hB hC ha hA h_sine_ratio
  sorry

end longest_side_of_triangle_l596_596188


namespace e_to_13pi_2_eq_i_l596_596537

-- Define the problem in Lean 4
theorem e_to_13pi_2_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I :=
by
  sorry

end e_to_13pi_2_eq_i_l596_596537


namespace length_of_AP_in_right_triangle_l596_596685

theorem length_of_AP_in_right_triangle 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 2))
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0 ∧ M.2 = 0)
  (inc : ℝ × ℝ)
  (hinc : inc = (1, 1)) :
  ∃ P : ℝ × ℝ, (P.1 = 0 ∧ P.2 = 1) ∧ dist A P = 1 := by
  sorry

end length_of_AP_in_right_triangle_l596_596685


namespace convert_e_to_rectangular_l596_596528

-- Definitions and assumptions based on conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x
def periodicity_cos (x : ℝ) : ∀ (k : ℤ), complex.cos (x + 2 * real.pi * k) = complex.cos x
def periodicity_sin (x : ℝ) : ∀ (k : ℤ), complex.sin (x + 2 * real.pi * k) = complex.sin x

-- Problem statement
theorem convert_e_to_rectangular:
  complex.exp (complex.I * 13 * real.pi / 2) = complex.I :=
by
  sorry

end convert_e_to_rectangular_l596_596528


namespace xiaoming_additional_games_l596_596916

variable (total_games games_won target_percentage : ℕ)

theorem xiaoming_additional_games :
  total_games = 20 →
  games_won = 95 * total_games / 100 →
  target_percentage = 96 →
  ∃ additional_games, additional_games = 5 ∧
    (games_won + additional_games) / (total_games + additional_games) = target_percentage / 100 :=
by
  sorry

end xiaoming_additional_games_l596_596916


namespace difference_of_extremes_l596_596370

theorem difference_of_extremes :
  let nums := [25, 17, 21, 34, 32] in
  let largest := List.maximum nums in
  let smallest := List.minimum nums in
  largest - smallest = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end difference_of_extremes_l596_596370


namespace triangle_sides_length_l596_596343

theorem triangle_sides_length 
  (h b s : ℕ) 
  (height_eq : h = 24) 
  (base_eq : b = 28)
  (sum_of_sides_eq : s = 56) 
  (area : ℕ) 
  (area_eq : area = (1 / 2) * b * h) :
  ∃ (x y : ℕ), x + y = s ∧ x = 26 ∧ y = 30 :=
by
  have h1 : b * h / 2 = area, from calc
      b * h / 2 = 28 * 24 / 2 : by rw [height_eq, base_eq]
          ... = 336 : by norm_num,
  sorry  -- Proof steps are not required as per instruction

end triangle_sides_length_l596_596343


namespace find_lambda_collinear_opposite_l596_596166

variable (a b : ℝ × ℝ)
variable (λ : ℝ)
variable (m : ℝ)

-- Define vector addition and scalar multiplication
instance : Add (ℝ × ℝ) := ⟨λ u v, (u.1 + v.1, u.2 + v.2)⟩
instance : SMul ℝ (ℝ × ℝ) := ⟨λ c u, (c * u.1, c * u.2)⟩

-- Define vectors c and d
def c := λ a + b
def d := a + (2 * λ - 1) • b

-- Define collinear_opposite
def collinear_opposite (u v : ℝ × ℝ) := ∃ m : ℝ, m < 0 ∧ u = m • v

-- Given conditions
axiom not_collinear : ¬ ∃ k : ℝ, k ≠ 0 ∧ b = k • a

theorem find_lambda_collinear_opposite :
  collinear_opposite (λ a + b) (a + (2 * λ - 1) • b) ↔ λ = -1/2 := 
sorry

end find_lambda_collinear_opposite_l596_596166


namespace problem1_problem2_l596_596617

def f (x k : ℝ) : ℝ := |x + 1| + |2 - x| - k

-- Problem 1
theorem problem1 (x : ℝ) : f x 4 < 0 → x > -3/2 ∧ x < 5/2 := by
  sorry

-- Problem 2
theorem problem2 (x k : ℝ) : (∀ x, f x k ≥ real.sqrt (k + 3)) ↔ k ≤ 1 := by
  sorry

end problem1_problem2_l596_596617


namespace tan_arccot_of_frac_l596_596035

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596035


namespace perimeter_of_semicircle_l596_596410

noncomputable def π : Real := 3.141592653589793

theorem perimeter_of_semicircle (r : Real) (h : r = 6.7) : 
  let d := 2 * r,
      half_circumference := π * r,
      P := half_circumference + d in
  P ≈ 34.438 :=
by
  admit

end perimeter_of_semicircle_l596_596410


namespace r20_moves_to_r30_probability_l596_596590

noncomputable def sequence : List ℝ := sorry

def isDistinct (l : List ℝ) : Prop :=
  l.Nodup

def swapOperation (l : List ℝ) : List ℝ :=
  List.foldl (λ l' i =>
    if i < l'.length - 1 ∧ l'.nthLe i sorry > l'.nthLe (i + 1) sorry then
      l'.updateNth i (l'.nthLe (i + 1) sorry)
        |> List.updateNth (i + 1) (l'.nthLe i sorry)
    else l') l (List.range (l.length - 1))

def probability : ℚ :=
  1 / 930

theorem r20_moves_to_r30_probability (l : List ℝ) (h : isDistinct l) (h_length : l.length = 40) :
  let result := swapOperation l
  classical.some (Rat.mk_eq probability (1 / 930)) + classical.some (Rat.mk_eq probability (1 / 930)) = 931 := 
  sorry

end r20_moves_to_r30_probability_l596_596590


namespace sum_of_roots_l596_596506

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l596_596506


namespace minimum_degree_of_g_l596_596384

noncomputable def f (x : ℕ) : ℕ := sorry -- Some polynomial
noncomputable def g (x : ℕ) : ℕ := sorry -- Some polynomial
noncomputable def h (x : ℕ) : ℕ := sorry -- Some polynomial

-- Given conditions
def condition1 : Prop := 2 * f = h - 5 * g
def condition2 : Prop := ∃ n : ℕ, deg f = 7
def condition3 : Prop := ∃ m : ℕ, deg h = 10

-- The goal to prove
theorem minimum_degree_of_g : condition1 → condition2 → condition3 → (∃ k : ℕ, deg g = k ∧ k ≥ 10) :=
by
  intros h₁ h₂ h₃
  sorry

end minimum_degree_of_g_l596_596384


namespace positive_divisors_of_n_squared_plus_one_l596_596914

theorem positive_divisors_of_n_squared_plus_one :
  (∀ n : ℤ, ∀ d ∈ {2, 4, 6, 8}, ∃ k : ℤ, n^2 + 1 = k ∧ divisors k = d) :=
by sorry

end positive_divisors_of_n_squared_plus_one_l596_596914


namespace volume_ratio_of_insphere_and_circumsphere_l596_596955

noncomputable def volume_ratio_of_spheres (a : ℝ) [fact (a > 0)] : ℝ :=
  let r1 := a / (2 * Real.sqrt 6)
  let r2 := a / (Real.sqrt 2)
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3)

theorem volume_ratio_of_insphere_and_circumsphere (a : ℝ) [fact (a > 0)] :
  volume_ratio_of_spheres a = 1 / 27 :=
sorry

end volume_ratio_of_insphere_and_circumsphere_l596_596955


namespace abs_neg_2023_l596_596484

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596484


namespace lake_width_l596_596728

theorem lake_width
  (W : ℝ)
  (janet_speed : ℝ) (sister_speed : ℝ) (wait_time : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : wait_time = 3)
  (h4 : W / sister_speed = W / janet_speed + wait_time) :
  W = 60 := 
sorry

end lake_width_l596_596728


namespace max_value_of_function_l596_596602

theorem max_value_of_function (x : ℝ) (h : x < 5 / 4) :
    (∀ y, y = 4 * x - 2 + 1 / (4 * x - 5) → y ≤ 1):=
sorry

end max_value_of_function_l596_596602


namespace mildred_oranges_l596_596310

theorem mildred_oranges (initial_oranges : ℝ) (eaten_oranges : ℝ) (remaining_oranges : ℝ) :
  initial_oranges = 77.0 → 
  eaten_oranges = 2.0 → 
  remaining_oranges = 75.0 → 
  initial_oranges - eaten_oranges = remaining_oranges := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end mildred_oranges_l596_596310


namespace percentage_increase_l596_596325

theorem percentage_increase (P Q R : ℝ) (x y : ℝ) 
  (h1 : P > 0) (h2 : Q > 0) (h3 : R > 0)
  (h4 : P = (1 + x / 100) * Q)
  (h5 : Q = (1 + y / 100) * R)
  (h6 : P = 2.4 * R) :
  x + y = 140 :=
sorry

end percentage_increase_l596_596325


namespace tan_arccot_of_frac_l596_596034

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596034


namespace nice_count_l596_596113

def is_k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ N = ∏ (i : ℕ) in (finset.range a.succ), (i + 1) 

def is_k_nice_5 (N : ℕ) : Prop := is_k_nice N 5

def is_k_nice_6 (N : ℕ) : Prop := is_k_nice N 6

def not_5_or_6_nice (x : ℕ) : Prop :=
  ¬ is_k_nice_5 x ∧ ¬ is_k_nice_6 x

theorem nice_count :
  finset.card ((finset.range 1000).filter (not_5_or_6_nice)) = 666 := 
sorry

end nice_count_l596_596113


namespace domain_of_f_l596_596988

def f (x : ℝ) : ℝ := log (x - 2)

theorem domain_of_f : ∀ x : ℝ, (∃ y, f y) ↔ x > 2 := by
  sorry

end domain_of_f_l596_596988


namespace A_plus_B_eq_zero_l596_596749

variables {A B : ℝ}

def f (x : ℝ) : ℝ := A * x^2 + B
def g (x : ℝ) : ℝ := B * x^2 + A
def fg (x : ℝ) : ℝ := f(g(x))
def gf (x : ℝ) : ℝ := g(f(x))

theorem A_plus_B_eq_zero (h1 : A ≠ B) (h2 : ∀ x : ℝ, fg(x) - gf(x) = 3 * (B - A)) : A + B = 0 :=
sorry

end A_plus_B_eq_zero_l596_596749


namespace find_angle_C_find_b_minus_a_range_l596_596189

variables (A B C a b c : ℝ)
variables (p q : ℝ × ℝ)

-- Conditions
def condition1 : p = (2 * Real.sin A, Real.cos (A - B)) := sorry
def condition2 : q = (Real.sin B, -1) := sorry
def condition3 : p.1 * q.1 + p.2 * q.2 = 1 / 2 := sorry
def condition4 : c = Real.sqrt 3 := sorry
def condition5 : C = Real.arccos (1 / 2) := by sorry

-- Proof goals
theorem find_angle_C (h1 : p = (2 * Real.sin A, Real.cos (A - B)))
                     (h2 : q = (Real.sin B, -1))
                     (h3 : p.1 * q.1 + p.2 * q.2 = 1 / 2) :
                     C = π / 3 := sorry

theorem find_b_minus_a_range (h4 : c = Real.sqrt 3) (h5 : C = π / 3) :
                              b - a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
begin
  sorry
end

end find_angle_C_find_b_minus_a_range_l596_596189


namespace length_of_common_chord_l596_596651

noncomputable def common_chord_length (x y : ℝ) : ℝ :=
if (x^2 + y^2 = 4) ∧ (x^2 + y^2 + 2*y - 6 = 0) then 2 * Real.sqrt 3 else 0

theorem length_of_common_chord (x y : ℝ) : 
  (x^2 + y^2 = 4) → 
  (x^2 + y^2 + 2*y - 6 = 0) → 
  common_chord_length x y = 2 * Real.sqrt 3 :=
by
  intros h1 h2
  rw [common_chord_length, if_pos (and.intro h1 h2)]
  sorry

end length_of_common_chord_l596_596651


namespace largest_angle_of_triangle_l596_596710

theorem largest_angle_of_triangle
  (p q r : ℝ)
  (h1 : p + 3 * q + 3 * r = p^2)
  (h2 : p + 3 * q - 3 * r = -1) :
  ∃ R, R = 120 :=
by
  use 120
  sorry

end largest_angle_of_triangle_l596_596710


namespace picture_center_distance_l596_596576

theorem picture_center_distance :
  ∀ (wall_width picture_width : ℕ) (pictures_count : ℕ),
  wall_width = 4800 ∧ picture_width = 420 ∧ pictures_count = 4 →
  (∃ x : ℕ, 
    (2 * x + picture_width/2 + picture_width/2 + 2 * x = wall_width) ∧ 
    x = 730) :=
by
  assume wall_width picture_width pictures_count,
  assume h : wall_width = 4800 ∧ picture_width = 420 ∧ pictures_count = 4,
  sorry

end picture_center_distance_l596_596576


namespace max_N_value_l596_596220

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596220


namespace hyperbola_asymptotes_find_a_l596_596790

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) : Prop :=
  ∃ (O A B : ℝ × ℝ), ∃ (c : ℝ),
  a > 0 ∧ b > 0 ∧
  (∏ (x : ℝ), (y = (b / a) * x) ∧ (y = -(b / a) * x))
  ∧ |dist O B - dist O A| = 0 ∧ c ^ 2 = a ^ 2 + b ^ 2 
  ∧ |dist A B| = 2 ∧
  |O = (0, 0)| ∧
  dist (midpoint ℝ O B) c = 1

theorem hyperbola_asymptotes_find_a (a b : ℝ) :
  hyperbola_asymptotes a b → a = 3 / 2 :=
by
  sorry

end hyperbola_asymptotes_find_a_l596_596790


namespace ceil_floor_difference_l596_596074

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596074


namespace vitya_catchup_time_l596_596855

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596855


namespace max_possible_cities_traversed_l596_596205

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596205


namespace complex_number_solution_l596_596152

-- Define that z is a complex number and the condition given in the problem.
theorem complex_number_solution (z : ℂ) (hz : (i / (z + i)) = 2 - i) : z = -1/5 - 3/5 * i :=
sorry

end complex_number_solution_l596_596152


namespace length_of_AC_l596_596265

-- Definitions for the conditions
def point : Type := ℝ × ℝ

variables (A B C D E : point)
variables (d1 d2 d3 : ℝ)
variable h_1 : (A, B) = (0, 0, 9, 18)
variable h_2 : (A, D) = (0, 0, 9, 0)
variable h_3 : (D, C) = (9, 0, 33, 18)
variable h_4: d1 = 15
variable h_5: d2=24 
variable h_6: d3=9

#check (A,B)

-- Distance function for the Euclidean distance
def dist (p1 p2 : point) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The main theorem statement
theorem length_of_AC (A B C D : point)
  (h₁ : dist A B = 15)
  (h₂ : dist D C = 24)
  (h₃ : dist A D = 9) :
  dist A C = 31.7 :=
by
  sorry

end length_of_AC_l596_596265


namespace ratio_of_a_b_l596_596783

theorem ratio_of_a_b (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
(h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h4 : a + b = 36) :
  a / b ≈ 4 :=
sorry

end ratio_of_a_b_l596_596783


namespace initial_processing_capacity_l596_596465

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l596_596465


namespace max_possible_cities_traversed_l596_596208

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596208


namespace Alexis_mangoes_l596_596460

-- Define the variables for the number of mangoes each person has.
variable (A D Ash : ℕ)

-- Conditions given in the problem.
axiom h1 : A = 4 * (D + Ash)
axiom h2 : A + D + Ash = 75

-- The proof goal.
theorem Alexis_mangoes : A = 60 :=
sorry

end Alexis_mangoes_l596_596460


namespace maximum_possible_value_of_N_l596_596215

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596215


namespace angle_PQR_is_90_degrees_l596_596266

-- Definitions for the conditions
def straight_line (a b c : ℝ) := a + b = c
def angle_QSP := 80

-- Theorem statement
theorem angle_PQR_is_90_degrees : 
  ∀ (P Q R S : ℝ), straight_line (angle P Q) (angle Q S) 180 → angle_SQP = angle_QSP → angle P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_degrees_l596_596266


namespace subtract_fifteen_result_l596_596187

theorem subtract_fifteen_result (x : ℕ) (h : x / 10 = 6) : x - 15 = 45 :=
by
  sorry

end subtract_fifteen_result_l596_596187


namespace geometric_sequence_a4_l596_596699

theorem geometric_sequence_a4 {a_2 a_6 a_4 : ℝ} 
  (h1 : ∃ a_1 r : ℝ, a_2 = a_1 * r ∧ a_6 = a_1 * r^5) 
  (h2 : a_2 * a_6 = 64) 
  (h3 : a_2 = a_1 * r)
  (h4 : a_6 = a_1 * r^5)
  : a_4 = 8 :=
by
  sorry

end geometric_sequence_a4_l596_596699


namespace ratio_Sydney_to_Sherry_l596_596320

variable (Randolph_age Sydney_age Sherry_age : ℕ)

-- Conditions
axiom Randolph_older_than_Sydney : Randolph_age = Sydney_age + 5
axiom Sherry_age_is_25 : Sherry_age = 25
axiom Randolph_age_is_55 : Randolph_age = 55

-- Theorem to prove
theorem ratio_Sydney_to_Sherry : (Sydney_age : ℝ) / (Sherry_age : ℝ) = 2 := by
  sorry

end ratio_Sydney_to_Sherry_l596_596320


namespace sunny_lead_l596_596659

-- Define the context of the race
variables {s m : ℝ}  -- s: Sunny's speed, m: Misty's speed
variables (distance_first : ℝ) (distance_ahead_first : ℝ)
variables (additional_distance_sunny_second : ℝ) (correct_answer : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first = 400 ∧
  distance_ahead_first = 20 ∧
  additional_distance_sunny_second = 40 ∧
  correct_answer = 20 

-- The math proof problem in Lean 4
theorem sunny_lead (h : conditions distance_first distance_ahead_first additional_distance_sunny_second correct_answer) :
  ∀ s m : ℝ, s / m = (400 / 380 : ℝ) → 
  (s / m) * 400 + additional_distance_sunny_second = (m / s) * 440 + correct_answer :=
sorry

end sunny_lead_l596_596659


namespace vitya_catch_up_time_l596_596862

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596862


namespace trapezoid_area_is_correct_l596_596453

-- Define necessary values for the trapezoid
def upper_base : ℝ := 5
def lower_base : ℝ := 10
def leg1 : ℝ := 3
def leg2 : ℝ := 4

-- Function to calculate the area of the trapezoid
def area_of_trapezoid (b1 b2 h : ℝ) : ℝ := 0.5 * (b1 + b2) * h

-- Height computation assuming it is derived correctly
def height : ℝ := 2.4

-- Prove that the area of the trapezoid is 18
theorem trapezoid_area_is_correct : area_of_trapezoid upper_base lower_base height = 18 := sorry

end trapezoid_area_is_correct_l596_596453


namespace max_possible_value_l596_596251

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596251


namespace length_of_segment_AB_l596_596708

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l596_596708


namespace log_base_49_of_x_l596_596172

-- Given condition as a definition
def given_condition (x : ℝ) : Prop := log 7 (x + 6) = 2

-- The problem statement to be proved
theorem log_base_49_of_x (x : ℝ) (h : given_condition x) :
  log 49 x = 0.99 :=
sorry

end log_base_49_of_x_l596_596172


namespace calculate_savings_l596_596015

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l596_596015


namespace length_of_wall_l596_596446

theorem length_of_wall (side_mirror length_wall width_wall : ℕ) 
  (mirror_area wall_area : ℕ) (H1 : side_mirror = 54) 
  (H2 : mirror_area = side_mirror * side_mirror) 
  (H3 : wall_area = 2 * mirror_area) 
  (H4 : width_wall = 68) 
  (H5 : wall_area = length_wall * width_wall) : 
  length_wall = 86 :=
by
  sorry

end length_of_wall_l596_596446


namespace greatest_diff_survives_l596_596690

-- Definitions based on conditions
inductive Population
| largestIndiv
| greatestDiff
| smallestIndiv
| leastDiff

-- Statement equivalent to the provided problem and solution
theorem greatest_diff_survives :
  ∀ (Ecosystem : Type) (P : Ecosystem → Population),
  (∀ x : Ecosystem, P x = Population.greatestDiff) →
  ∀ (x : Ecosystem), survival_chance x = highest :=
begin
  sorry
end

end greatest_diff_survives_l596_596690


namespace budget_for_equipment_l596_596937

theorem budget_for_equipment 
    (transportation_p : ℝ := 20)
    (r_d_p : ℝ := 9)
    (utilities_p : ℝ := 5)
    (supplies_p : ℝ := 2)
    (salaries_degrees : ℝ := 216)
    (total_degrees : ℝ := 360)
    (total_budget : ℝ := 100)
    :
    (total_budget - (transportation_p + r_d_p + utilities_p + supplies_p +
    (salaries_degrees / total_degrees * total_budget))) = 4 := 
sorry

end budget_for_equipment_l596_596937


namespace smallest_set_size_l596_596678

noncomputable def smallest_num_elements (s : Multiset ℝ) : ℕ :=
  s.length

theorem smallest_set_size (s : Multiset ℝ) :
  (∀ a b c : ℝ, s = {a, b, 3, 6, 6, c}) →
  (s.median = 3) →
  (s.mean = 5) →
  (∀ x, s.count x < 3 → x ≠ 6) →
  smallest_num_elements s = 6 :=
by
  intros _ _ _ _
  sorry

end smallest_set_size_l596_596678


namespace each_child_plays_equally_l596_596835

theorem each_child_plays_equally (total_time : ℕ) (num_children : ℕ)
  (play_group_size : ℕ) (play_time : ℕ) :
  num_children = 6 ∧ play_group_size = 3 ∧ total_time = 120 ∧ play_time = (total_time * play_group_size) / num_children →
  play_time = 60 :=
by
  intros h
  sorry

end each_child_plays_equally_l596_596835


namespace min_elements_with_conditions_l596_596665

theorem min_elements_with_conditions (s : Finset ℝ) 
  (h_median : (s.sort (≤)).median = 3) 
  (h_mean : s.mean = 5)
  (h_mode : s.mode = 6) : 
  s.card >= 6 :=
sorry

end min_elements_with_conditions_l596_596665


namespace area_of_triangle_correct_l596_596559

def area_of_triangle : ℝ :=
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  (det / 2)

theorem area_of_triangle_correct : area_of_triangle = 15 := by
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  have h : area_of_triangle = (det / 2) := rfl
  rw [h]
  norm_num
  sorry

end area_of_triangle_correct_l596_596559


namespace math_problem_l596_596086

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596086


namespace shem_wage_multiple_kem_l596_596775

-- Define the hourly wages and conditions
def kem_hourly_wage : ℝ := 4
def shem_daily_wage : ℝ := 80
def shem_workday_hours : ℝ := 8

-- Prove the multiple of Shem's hourly wage compared to Kem's hourly wage
theorem shem_wage_multiple_kem : (shem_daily_wage / shem_workday_hours) / kem_hourly_wage = 2.5 := by
  sorry

end shem_wage_multiple_kem_l596_596775


namespace min_elements_with_conditions_l596_596666

theorem min_elements_with_conditions (s : Finset ℝ) 
  (h_median : (s.sort (≤)).median = 3) 
  (h_mean : s.mean = 5)
  (h_mode : s.mode = 6) : 
  s.card >= 6 :=
sorry

end min_elements_with_conditions_l596_596666


namespace number_is_negative_l596_596181

theorem number_is_negative (x : ℝ) : -x > x → x < 0 :=
by
  intro h
  have h1 : 0 > 2 * x := by linarith
  exact lt_of_mul_neg_left h1 zero_lt_two


end number_is_negative_l596_596181


namespace binom_sum_mod_l596_596753

open Nat

theorem binom_sum_mod (p : ℕ) (hprime : Nat.Prime p) (hgt : p > 3) :
  (∑ i in Finset.range (p + 1), Nat.choose (i * p) p * Nat.choose ((p - i + 1) * p) p) % p^2 =
    if p % 6 = 1 then
      let k := (p - 1) / 6 in (4 * k + 1) * p % p^2
    else
      let k := (p - 5) / 6 in (2 * k + 2) * p % p^2 :=
sorry

end binom_sum_mod_l596_596753


namespace convert_e_to_rectangular_l596_596529

-- Definitions and assumptions based on conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x
def periodicity_cos (x : ℝ) : ∀ (k : ℤ), complex.cos (x + 2 * real.pi * k) = complex.cos x
def periodicity_sin (x : ℝ) : ∀ (k : ℤ), complex.sin (x + 2 * real.pi * k) = complex.sin x

-- Problem statement
theorem convert_e_to_rectangular:
  complex.exp (complex.I * 13 * real.pi / 2) = complex.I :=
by
  sorry

end convert_e_to_rectangular_l596_596529


namespace vasya_fractions_l596_596339

theorem vasya_fractions (n : ℕ) (fractions : list ℚ) (h1 : fractions = list.map (λ k, k / (n - k + 1)) (list.range n.succ))
  (h2 : (list.filter (λ x, ∃ k : ℕ, x = 1 / k) (list.map (λ p, p.1 - p.2) (fractions.zip fractions.tail))).length = 10000) :
  (list.filter (λ x, ∃ k : ℕ, x = 1 / k) (list.map (λ p, p.1 - p.2) (fractions.zip fractions.tail))).length ≥ 15000 :=
by
  sorry

end vasya_fractions_l596_596339


namespace Vitya_catches_mother_l596_596867

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596867


namespace flash_runs_distance_l596_596455

variables {z v t : ℝ} (h_z : 0 < z) (h_v : 0 < v)

theorem flash_runs_distance : 
  let ace_speed := v in
  let flash_speed := 3 * v in
  let head_start := 3 * z in
  let time_to_catch := (3 * z) / (2 * v) in
  3 * flash_speed * time_to_catch = (9 * z) / 2 :=
sorry

end flash_runs_distance_l596_596455


namespace min_elements_with_conditions_l596_596664

theorem min_elements_with_conditions (s : Finset ℝ) 
  (h_median : (s.sort (≤)).median = 3) 
  (h_mean : s.mean = 5)
  (h_mode : s.mode = 6) : 
  s.card >= 6 :=
sorry

end min_elements_with_conditions_l596_596664


namespace correct_growth_equation_l596_596192

-- Define the parameters
def initial_income : ℝ := 2.36
def final_income : ℝ := 2.7
def growth_period : ℕ := 2

-- Define the growth rate x
variable (x : ℝ)

-- The theorem we want to prove
theorem correct_growth_equation : initial_income * (1 + x)^growth_period = final_income :=
sorry

end correct_growth_equation_l596_596192


namespace king_of_qi_wins_probability_l596_596373

/-- Tian Ji and the King of Qi had a horse race. We know the following:
1. Tian Ji's top horse is better than the King of Qi's middle horse but worse than the King of Qi's top horse.
2. Tian Ji's middle horse is better than the King of Qi's bottom horse but worse than the King of Qi's middle horse.
3. Tian Ji's bottom horse is worse than the King of Qi's bottom horse.
We need to show that if a horse is randomly selected from each side for a race, the probability of the King of Qi's horse winning is 2/3.
-/
theorem king_of_qi_wins_probability :
  let A := "King of Qi's top horse",
      B := "King of Qi's middle horse",
      C := "King of Qi's bottom horse",
      a := "Tian Ji's top horse",
      b := "Tian Ji's middle horse",
      c := "Tian Ji's bottom horse" in
  (strength A > strength a ∧ strength a > strength B ∧ 
   strength B > strength b ∧ strength b > strength C ∧ 
   strength C > strength c ∧ strength a > strength b ∧ 
   strength b > strength c) →
  (prob_king_of_qi_wins = 2 / 3) :=
by
  sorry

end king_of_qi_wins_probability_l596_596373


namespace math_problem_l596_596082

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596082


namespace luis_finish_fourth_task_l596_596762

-- Define the starting and finishing times
def start_time : ℕ := 540  -- 9:00 AM is 540 minutes from midnight
def finish_third_task : ℕ := 750  -- 12:30 PM is 750 minutes from midnight
def duration_one_task : ℕ := (750 - 540) / 3  -- Time for one task

-- Define the problem statement
theorem luis_finish_fourth_task :
  start_time = 540 →
  finish_third_task = 750 →
  3 * duration_one_task = finish_third_task - start_time →
  finish_third_task + duration_one_task = 820 :=
by
  -- You can place the proof for the theorem here
  sorry

end luis_finish_fourth_task_l596_596762


namespace tan_arccot_3_5_l596_596046

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596046


namespace max_N_value_l596_596223

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596223


namespace tan_arccot_eq_5_div_3_l596_596027

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596027


namespace log_expression_evaluation_l596_596477

theorem log_expression_evaluation :
  let log_10 := Real.logb 10
  in (log_10 5)^2 + (log_10 2) * (log_10 50) = 1 := by
  have log_10 : ℝ → ℝ := Real.logb 10
  have log_50 : log_10 50 = 1 + log_10 5 := by sorry
  sorry

end log_expression_evaluation_l596_596477


namespace amount_of_b_l596_596921

variable (A B : ℝ)

theorem amount_of_b (h₁ : A + B = 2530) (h₂ : (3 / 5) * A = (2 / 7) * B) : B = 1714 :=
sorry

end amount_of_b_l596_596921


namespace sum_of_roots_l596_596515

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l596_596515


namespace problem_solution_l596_596654

noncomputable def am_gm_inequality (a b c : ℝ) : Prop :=
  (a + b + c) / 3 ≥ real.cbrt (a * b * c)

lemma specific_condition_false (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h₃ : c = 16) :
  (a + b + c) / 3 - real.cbrt (a * b * c) ≠ 1 := 
by {
  rw [h₁, h₂, h₃],
  norm_num,
  have h : (4 + 9 + 16) / 3 - real.cbrt (4 * 9 * 16) = (29 / 3) - real.cbrt 576,
  norm_num,
  -- Continue with the numerical simplifications and show ≠ 1
  sorry
}

-- The main theorem combines both parts.
theorem problem_solution (a b c : ℝ) :
  am_gm_inequality a b c ∧ specific_condition_false a b c (by norm_num : a = 4) (by norm_num : b = 9) (by norm_num : c = 16) :=
sorry

end problem_solution_l596_596654


namespace rhombus_and_rectangle_diags_bisect_l596_596359

-- Define the basic properties of a rhombus
structure Rhombus where
  sides_equal : ∀ a b : ℝ, true  -- we're primarily focusing on the diagonal properties here
  diags_bisect : ∀ {p q : ℝ}, true

-- Define the basic properties of a rectangle
structure Rectangle where
  angles_right : ∀ a b c d : ℝ, true  -- we're primarily focusing on the diagonal properties here
  diags_bisect : ∀ {p q : ℝ}, true

-- Proof goal stating that both a rhombus and rectangle have diagonals that bisect each other
theorem rhombus_and_rectangle_diags_bisect : 
  (R : Rhombus) → (rec : Rectangle) → (R.diags_bisect = rec.diags_bisect) :=
by
  sorry

end rhombus_and_rectangle_diags_bisect_l596_596359


namespace f_diff_ineq_l596_596303

variable {f : ℝ → ℝ}
variable (deriv_f : ∀ x > 0, x * (deriv f x) > 1)

theorem f_diff_ineq (h : ∀ x > 0, x * (deriv f x) > 1) : f 2 - f 1 > Real.log 2 := by 
  sorry

end f_diff_ineq_l596_596303


namespace number_of_trailing_zeros_l596_596904

theorem number_of_trailing_zeros (n : ℕ) (h : n = 10^10 - 3) : 
  nat.trail_zeros (n^2) = 10 := 
by 
  sorry

end number_of_trailing_zeros_l596_596904


namespace infinite_pseudoprimes_l596_596066

-- Definition of a pseudoprime to base a
def isPseudoprime (a n : ℕ) : Prop := 
  ¬ n.prime ∧ a^(n-1) ≡ 1 [MOD n]

-- The statement to prove there are infinitely many pseudoprimes to base 2
theorem infinite_pseudoprimes : ∃ᶠ n in at_top, isPseudoprime 2 n :=
sorry

end infinite_pseudoprimes_l596_596066


namespace correct_statements_count_l596_596750

theorem correct_statements_count 
  (x y : ℝ) (a : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (num_correct_statements x y a = 2) :=
begin
  -- Definitions of operations to be analyzed
  let stmt1 := (if x < 0 then false else log a (x^2) = 3 * log a x),
  let stmt2 := (log a |xy| = log a |x| + log a |y|),
  let stmt3 := (e = real.log x) → (x = real.exp (e^2)),
  let stmt4 := (real.log 10 (real.log y) = 0) → (y = real.exp 1),
  let stmt5 := (2^(1 + real.log 4 x) = 16) → (x = 64),
  
  -- Count how many statements above are true
  let num_correct_statements := (stmt1 as Prop) + (stmt2 as Prop) + (stmt3 as Prop) + (stmt4 as Prop) + (stmt5 as Prop),
  sorry
end

end correct_statements_count_l596_596750


namespace take_home_pay_l596_596719

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l596_596719


namespace brian_books_chapters_l596_596976

variable (x : ℕ)

theorem brian_books_chapters (h1 : 1 ≤ x) (h2 : 20 + 2 * x + (20 + 2 * x) / 2 = 75) : x = 15 :=
sorry

end brian_books_chapters_l596_596976


namespace Iggy_miles_on_Monday_l596_596655

theorem Iggy_miles_on_Monday 
  (tuesday_miles : ℕ)
  (wednesday_miles : ℕ)
  (thursday_miles : ℕ)
  (friday_miles : ℕ)
  (monday_minutes : ℕ)
  (pace : ℕ)
  (total_hours : ℕ)
  (total_minutes : ℕ)
  (total_tuesday_to_friday_miles : ℕ)
  (total_tuesday_to_friday_minutes : ℕ) :
  tuesday_miles = 4 →
  wednesday_miles = 6 →
  thursday_miles = 8 →
  friday_miles = 3 →
  pace = 10 →
  total_hours = 4 →
  total_minutes = total_hours * 60 →
  total_tuesday_to_friday_miles = tuesday_miles + wednesday_miles + thursday_miles + friday_miles →
  total_tuesday_to_friday_minutes = total_tuesday_to_friday_miles * pace →
  monday_minutes = total_minutes - total_tuesday_to_friday_minutes →
  (monday_minutes / pace) = 3 := sorry

end Iggy_miles_on_Monday_l596_596655


namespace quadratic_inequality_solution_l596_596365

theorem quadratic_inequality_solution : {x : ℝ | x^2 - x - 6 < 0} = set.Ioo (-2 : ℝ) 3 :=
by
  sorry

end quadratic_inequality_solution_l596_596365


namespace maximum_N_value_l596_596227

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596227


namespace minimal_elements_for_conditions_l596_596683

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, Nat.odd_iff_not_even.mpr (List.length_pos_of_ne_nil sorted).2])
  else
    let a := sorted.nth_le (sorted.length / 2 - 1) (by simp [Nat.sub_pos_of_lt (Nat.div_lt_self (List.length_pos_of_ne_nil sorted)].)
    let b := sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, List.length_pos_of_ne_nil sorted])
    (a + b) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def mode (s : List ℝ) : Option ℝ :=
  s.group_by id (· = ·).max_by (·.length).map (·.head)

def satisfies_conditions (s : List ℝ) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = some 6

theorem minimal_elements_for_conditions : ∀ s : List ℝ, satisfies_conditions s → s.length ≥ 6 :=
by
  intro s h
  sorry

end minimal_elements_for_conditions_l596_596683


namespace last_two_nonzero_digits_80_factorial_l596_596352

noncomputable def last_two_nonzero_digits_of_factorial (n : ℕ) : ℕ :=
  let fac := (nat.factorial n) in
  -- Helper function to remove trailing zeros and get last two non-zero digits
  let rec remove_trailing_zeros (x : ℕ) :=
    if x % 10 = 0 then remove_trailing_zeros (x / 10) else x in
  let without_zeros := remove_trailing_zeros fac in
  without_zeros % 100

theorem last_two_nonzero_digits_80_factorial : last_two_nonzero_digits_of_factorial 80 = 72 :=
  sorry

end last_two_nonzero_digits_80_factorial_l596_596352


namespace cos_B_in_triangle_ABC_l596_596709

theorem cos_B_in_triangle_ABC
  (A B C : Type)
  (BC AC : ℝ)
  (angleA : ℝ)
  (hBC : BC = 15)
  (hAC : AC = 10)
  (hAngleA : angleA = real.pi / 3) :
  ∃ (cosB : ℝ), cosB = real.sqrt(6) / 3 :=
by sorry

end cos_B_in_triangle_ABC_l596_596709


namespace class_average_history_test_l596_596428

theorem class_average_history_test :
  let total_students := 30
  let group1_students := 25
  let group2_students := 5
  let group1_average := 75
  let group2_average := 40
  let total_score := (group1_students * group1_average) + (group2_students * group2_average)
  let class_average := total_score / total_students
  ∃ avg, avg = round class_average ∧ avg = 69 :=
by
  sorry

end class_average_history_test_l596_596428


namespace max_possible_N_in_cities_l596_596246

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596246


namespace probability_top_two_hearts_and_third_spade_l596_596054

open Classical

-- Definitions related to the standard deck of cards with the described conditions.
def deck : Finset (Fin 52) := Finset.univ

def heartsuits : Finset (Fin 52) := (Finset.range 13).image (λ n, n + 0)
def spadesuits : Finset (Fin 52) := (Finset.range 13).image (λ n, n + 39)

-- The main problem statement.
theorem probability_top_two_hearts_and_third_spade :
  (13 * 12 * 13 : ℚ) / (52 * 51 * 50) = 13 / 850 := by
  sorry

end probability_top_two_hearts_and_third_spade_l596_596054


namespace cos_alpha_plus_pi_div_3_l596_596121

theorem cos_alpha_plus_pi_div_3 (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 2 * sin β - cos α = 1) (h4 : sin α + 2 * cos β = sqrt 3) : 
  cos (α + π / 3) = -1 / 4 :=
sorry

end cos_alpha_plus_pi_div_3_l596_596121


namespace ceil_floor_diff_l596_596078

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596078


namespace convert_e_to_rectangular_l596_596531

-- Definitions and assumptions based on conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x
def periodicity_cos (x : ℝ) : ∀ (k : ℤ), complex.cos (x + 2 * real.pi * k) = complex.cos x
def periodicity_sin (x : ℝ) : ∀ (k : ℤ), complex.sin (x + 2 * real.pi * k) = complex.sin x

-- Problem statement
theorem convert_e_to_rectangular:
  complex.exp (complex.I * 13 * real.pi / 2) = complex.I :=
by
  sorry

end convert_e_to_rectangular_l596_596531


namespace smallest_n_divisible_by_100_million_l596_596302

noncomputable def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

noncomputable def nth_term (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^(n - 1)

theorem smallest_n_divisible_by_100_million :
  ∀ (a1 a2 : ℚ), a1 = 5/6 → a2 = 25 → 
  ∃ n : ℕ, nth_term a1 (common_ratio a1 a2) n % 100000000 = 0 ∧ n = 9 :=
by
  intros a1 a2 h1 h2
  have r := common_ratio a1 a2
  have a9 := nth_term a1 r 9
  sorry

end smallest_n_divisible_by_100_million_l596_596302


namespace tan_arccot_of_frac_l596_596033

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596033


namespace plane_equation_l596_596803

theorem plane_equation (x y z : ℝ) (A B C D : ℤ) (h1 : A = 9) (h2 : B = -6) (h3 : C = 4) (h4 : D = -133) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) : 
  A * x + B * y + C * z + D = 0 :=
sorry

end plane_equation_l596_596803


namespace max_value_of_N_l596_596196

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596196


namespace janet_more_siblings_than_carlos_l596_596727

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l596_596727


namespace parallel_vectors_l596_596165

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, -2)
def vector_b := (m, -1)

theorem parallel_vectors (h : vector_a ∥ vector_b) : m = 1 / 2 := by
  -- Proof omitted
  sorry

end parallel_vectors_l596_596165


namespace take_home_pay_l596_596720

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l596_596720


namespace magnitude_two_vec_a_sub_vec_b_l596_596168

noncomputable def vec_a := sorry -- Assume definition for vec_a
noncomputable def vec_b := sorry -- Assume definition for vec_b

def angle_between_vec_a_vec_b : ℝ := real.pi / 3 -- 60 degrees in radians
axiom norm_vec_a : ∥vec_a∥ = 2
axiom norm_vec_b : ∥vec_b∥ = 3
axiom vec_a_dot_vec_b : vec_a • vec_b = ∥vec_a∥ * ∥vec_b∥ * real.cos angle_between_vec_a_vec_b

theorem magnitude_two_vec_a_sub_vec_b : ∥2 • vec_a - vec_b∥ = real.sqrt 13 := by
  sorry

end magnitude_two_vec_a_sub_vec_b_l596_596168


namespace contribution_rate_correct_l596_596183

def sumOfSquaredResiduals := 325
def totalSumOfSquares := 923
def contributionRate := sumOfSquaredResiduals.toFloat / totalSumOfSquares.toFloat

theorem contribution_rate_correct :
  contributionRate ≈ 0.352 := by
  sorry

end contribution_rate_correct_l596_596183


namespace iesha_books_l596_596640

theorem iesha_books : 
  ∀ (totalBooks schoolBooks : ℕ), 
  totalBooks = 344 → 
  schoolBooks = 136 → 
  totalBooks - schoolBooks = 208 :=
  by
  intros totalBooks schoolBooks ht hs
  rw [ht, hs]
  exact rfl

end iesha_books_l596_596640


namespace positive_sum_geometric_mean_ineq_l596_596142

variable (n : ℕ) (k : ℝ) (a : Fin n → ℝ)

noncomputable def sum_except {α : Type*} [AddGroup α] (f : Fin n → α) (i : Fin n) : α :=
  (∑ j, f j) - f i

theorem positive_sum_geometric_mean_ineq 
  (hn : n ≥ 2) 
  (hk : k ≥ 1) 
  (ha_pos : ∀ i : Fin n, 0 < a i) : 
  (∑ i : Fin n, (a i / sum_except a i) ^ k) ≥ n / (n - 1 : ℝ)^k := 
by
  sorry

end positive_sum_geometric_mean_ineq_l596_596142


namespace median_length_squared_l596_596270

theorem median_length_squared 
  (A B C O : Point)
  (AO_is_median : midpoint O B C)
  (m_a : ℝ) (b c a : ℝ)
  (h_m_a: m_a = dist A O)
  (h_b: dist A C = b)
  (h_c: dist A B = c)
  (h_a: dist B C = a) :
  m_a^2 = (1/2) * b^2 + (1/2) * c^2 - (1/4) * a^2 := 
sorry

end median_length_squared_l596_596270


namespace min_tan_sum_l596_596688

theorem min_tan_sum (a b C : ℝ) (A B : ℝ)
  (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) (h5 : 0 < C) (h6 : C < π / 2)
  (h7 : a = 2 * b * Real.sin C) :
  (∃ A B C : ℝ, tan A + tan B + tan C = 8) :=
sorry

end min_tan_sum_l596_596688


namespace mariela_cards_after_home_l596_596308

theorem mariela_cards_after_home (cards_in_hospital : ℕ) (total_cards : ℕ) (h₁ : cards_in_hospital = 403) (h₂ : total_cards = 690) : 
  (total_cards - cards_in_hospital = 287) :=
by
  rw [h₁, h₂]
  sorry

end mariela_cards_after_home_l596_596308


namespace fractions_sum_to_one_l596_596836

theorem fractions_sum_to_one :
  ∃ (a b c : ℕ), (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ ((a, b, c) = (2, 3, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (3, 6, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (6, 3, 2)) :=
by
  sorry

end fractions_sum_to_one_l596_596836


namespace impossible_to_draw_1006_2012gons_l596_596713

theorem impossible_to_draw_1006_2012gons :
  ¬ ∃ G : Finset (Finset (Fin 2012)),
    G.card = 1006 ∧
    ∀ polygon ∈ G, polygon.card = 2012 ∧
    (∀ p1 p2 ∈ G, p1 ≠ p2 → p1 ∩ p2 ⊆ ∅) :=
sorry

end impossible_to_draw_1006_2012gons_l596_596713


namespace sum_of_roots_l596_596504

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l596_596504


namespace crackers_initial_count_l596_596309

theorem crackers_initial_count (friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) :
  (friends = 4) → (crackers_per_friend = 2) → (total_crackers = friends * crackers_per_friend) → total_crackers = 8 :=
by intros h_friends h_crackers_per_friend h_total_crackers
   rw [h_friends, h_crackers_per_friend] at h_total_crackers
   exact h_total_crackers

end crackers_initial_count_l596_596309


namespace restroom_students_l596_596018

theorem restroom_students (R : ℕ) (h1 : 4 * 6 = 24) (h2 : (2/3 : ℚ) * 24 = 16)
  (h3 : 23 = 16 + (3 * R - 1) + R) : R = 2 :=
by
  sorry

end restroom_students_l596_596018


namespace infinite_product_equality_l596_596067

theorem infinite_product_equality :
  (∏ n in set.univ, 2^(n / (2^n : ℝ))) = 4 :=
by sorry

end infinite_product_equality_l596_596067


namespace sum_of_six_smallest_multiples_of_8_l596_596910

theorem sum_of_six_smallest_multiples_of_8 :
  let multiples := [8, 16, 24, 32, 40, 48] in
  (multiples.sum) = 168 :=
by
  let multiples := [8, 16, 24, 32, 40, 48]
  have h : multiples = [8, 16, 24, 32, 40, 48] := rfl
  exact sorry

end sum_of_six_smallest_multiples_of_8_l596_596910


namespace least_common_multiple_e_n_is_230_l596_596344

-- Define the conditions for e and n
variables {e : ℕ} (h_pos_e : e > 0) (n : ℕ) (h_3digits : 100 ≤ n ∧ n < 1000) 
variables (h_not_div_by_3 : ¬(3 ∣ n)) (h_not_div_2_e : ¬(2 ∣ e))
variables (h_n_is_230 : n = 230)

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

-- Statement we need to prove
theorem least_common_multiple_e_n_is_230 : lcm e 230 = 230 := 
sorry

end least_common_multiple_e_n_is_230_l596_596344


namespace part_a_equidistant_lines_in_plane_part_b_equidistant_lines_parallel_to_plane_part_c_equidistant_lines_parallel_to_line_l596_596062

variables {A B C : Point} (S : Plane) (i : Line)
           (ABC_noncollinear : ¬Collinear [A, B, C])
           (i_intersects_S : Intersects i S)
           (e : Line)

-- Part a
theorem part_a_equidistant_lines_in_plane :
  ∀ (e : Line), (liesInPlane e S ∧ equidistantFromPoints e [A, B, C]) ↔ (isMidlineOfTriangle e A B C) := 
sorry

-- Part b
theorem part_b_equidistant_lines_parallel_to_plane :
  ∀ (e : Line), (parallelTo e S ∧ equidistantFromPoints e [A, B, C]) ↔ (isVerticalTranslationOfMidline e A B C S) := 
sorry

-- Part c
theorem part_c_equidistant_lines_parallel_to_line :
  ∀ (e : Line), (parallelTo e i ∧ equidistantFromPoints e [A, B, C]) ↔ (passesThroughCircumcenterOfProjections e A B C i) := 
sorry

end part_a_equidistant_lines_in_plane_part_b_equidistant_lines_parallel_to_plane_part_c_equidistant_lines_parallel_to_line_l596_596062


namespace distance_proof_l596_596411

namespace IntersectionDistance

-- Define the linear function as a given (with a and b as unknowns)
def linear_function (a b x : ℝ) : ℝ := a * x + b

-- Define the distance calculation on the plane for the given conditions
def distance_between_intersections (a b : ℝ) : ℝ :=
  real.sqrt ((a^2 + 1) * (a^2 + 4 * b))

-- State the conditions given
axiom condition1 (a b : ℝ) : distance_between_intersections a b = 2 * real.sqrt 3
axiom condition2 (a b : ℝ) : real.sqrt ((a^2 + 1) * (a^2 + 4 * b + 12)) = real.sqrt 60

-- Calculate the required distance
def distance_solution: ℝ := 2 * real.sqrt 11

-- The main theorem to prove the final distance
theorem distance_proof (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) :
  distance_between_intersections a (b+2) = distance_solution := 
sorry

end IntersectionDistance

end distance_proof_l596_596411


namespace convert_negative_150_degrees_to_radians_l596_596980

def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem convert_negative_150_degrees_to_radians :
  degrees_to_radians (-150) = - (5/6) * Real.pi := 
by
  sorry

end convert_negative_150_degrees_to_radians_l596_596980


namespace time_invested_q_l596_596838

noncomputable def investment_time_q (P Q R Profit_p Profit_q Profit_r : ℝ) (Investment_p Investment_r : ℝ) : ℝ :=
  let ratio_investment := P / Q / R = 7 / 5.00001 / 3.99999
  let ratio_profit := Profit_p / Profit_q / Profit_r = 7.00001 / 10 / 6
  let investment_p := Investment_p = 5
  let investment_r := Investment_r = 8
  T

theorem time_invested_q
  (P Q R Profit_p Profit_q Profit_r : ℝ)
  (Investment_p Investment_r : ℝ)
  (h1 : P / Q / R = 7 / 5.00001 / 3.99999)
  (h2 : Profit_p / Profit_q / Profit_r = 7.00001 / 10 / 6)
  (h3 : Investment_p = 5)
  (h4 : Investment_r = 8) :
  investment_time_q P Q R Profit_p Profit_q Profit_r Investment_p Investment_r = 13.39 := 
sorry

end time_invested_q_l596_596838


namespace convert_exp_to_rectangular_form_l596_596540

theorem convert_exp_to_rectangular_form : exp (13 * π * complex.I / 2) = complex.I :=
by
  sorry

end convert_exp_to_rectangular_form_l596_596540


namespace cos_double_angle_second_quadrant_l596_596603

theorem cos_double_angle_second_quadrant (α : ℝ)
  (h1 : α > π / 2 ∧ α < π)
  (h2 : sin α + cos α = sqrt (3) / 3) :
  cos (2 * α) = - sqrt (5) / 3 :=
sorry

end cos_double_angle_second_quadrant_l596_596603


namespace ceil_floor_diff_l596_596076

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596076


namespace find_angle_A_find_tan_C_l596_596598

-- Define the conditions
variables (A B C : ℝ) -- internal angles of the triangle
variables (m n : ℝ × ℝ) -- vectors

-- Given vectors
def m : ℝ × ℝ := (Real.cos A + 1, Real.sqrt 3)
def n : ℝ × ℝ := (Real.sin A, 1)

-- Condition: vector m is parallel to vector n
def vectors_parallel : Prop := ∃ k, n = (k * (Real.cos A + 1), k * Real.sqrt 3)

-- Condition: equation for B
def equation_B : Prop := (1 + Real.sin (2 * B)) / (Real.cos B ^ 2 - Real.sin B ^ 2) = -3

-- Proof statements
theorem find_angle_A (h_parallel : vectors_parallel) : A = π / 3 :=
sorry

theorem find_tan_C (h_parallel : vectors_parallel) (h_B : equation_B) : Real.tan C = (8 + 5 * Real.sqrt 3) / 11 :=
sorry

end find_angle_A_find_tan_C_l596_596598


namespace b_is_arithmetic_sequence_a_general_formula_l596_596129

-- Sequence {a_n} defined
def a : ℕ → ℝ
| 0       := 1   -- Lean uses 0-indexing naturally; consider 0 as a base case for a_1 = 1
| (n + 1) := a n / (2 * a n + 1)

-- Sequence {b_n} defined
def b (n : ℕ) : ℝ := 1 / a (n + 1)

-- Proof problem 1: b_n is an arithmetic sequence with first term 1 and common difference 2
theorem b_is_arithmetic_sequence : ∀ n : ℕ, b n = 1 + 2 * n :=
sorry

-- Proof problem 2: General formula for a_n
theorem a_general_formula : ∀ n : ℕ, a (n + 1) = 1 / (2 * (n + 1) - 1) :=
sorry

end b_is_arithmetic_sequence_a_general_formula_l596_596129


namespace intersection_eq_one_l596_596629

def A : set ℝ := { x | x ≤ 1 }
def B : set ℝ := { y | ∃ x, y = x^2 + 2*x + 2 }

theorem intersection_eq_one : A ∩ B = {1} := 
by
  have B_transformed : B = { y | y ≥ 1 } :=
    sorry
  have A_eq : A = { x | x ≤ 1 } := 
    sorry
  show A ∩ B = {1} from
    sorry

end intersection_eq_one_l596_596629


namespace minimal_elements_for_conditions_l596_596680

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, Nat.odd_iff_not_even.mpr (List.length_pos_of_ne_nil sorted).2])
  else
    let a := sorted.nth_le (sorted.length / 2 - 1) (by simp [Nat.sub_pos_of_lt (Nat.div_lt_self (List.length_pos_of_ne_nil sorted)].)
    let b := sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, List.length_pos_of_ne_nil sorted])
    (a + b) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def mode (s : List ℝ) : Option ℝ :=
  s.group_by id (· = ·).max_by (·.length).map (·.head)

def satisfies_conditions (s : List ℝ) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = some 6

theorem minimal_elements_for_conditions : ∀ s : List ℝ, satisfies_conditions s → s.length ≥ 6 :=
by
  intro s h
  sorry

end minimal_elements_for_conditions_l596_596680


namespace remaining_sweet_potatoes_l596_596767

def harvested_sweet_potatoes : ℕ := 80
def sold_sweet_potatoes_mrs_adams : ℕ := 20
def sold_sweet_potatoes_mr_lenon : ℕ := 15
def traded_sweet_potatoes : ℕ := 10
def donated_sweet_potatoes : ℕ := 5

theorem remaining_sweet_potatoes :
  harvested_sweet_potatoes - (sold_sweet_potatoes_mrs_adams + sold_sweet_potatoes_mr_lenon + traded_sweet_potatoes + donated_sweet_potatoes) = 30 :=
by
  sorry

end remaining_sweet_potatoes_l596_596767


namespace max_value_of_N_l596_596198

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596198


namespace janet_more_siblings_than_carlos_l596_596725

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l596_596725


namespace F_2_f_3_equals_341_l596_596058

def f (a : ℕ) : ℕ := a^2 - 2
def F (a b : ℕ) : ℕ := b^3 - a

theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end F_2_f_3_equals_341_l596_596058


namespace ceiling_and_floor_calculation_l596_596092

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596092


namespace domain_of_log_function_l596_596336

theorem domain_of_log_function : 
  { x : ℝ | x < 1 ∨ x > 2 } = { x : ℝ | 0 < x^2 - 3 * x + 2 } :=
by sorry

end domain_of_log_function_l596_596336


namespace irreducible_polynomial_l596_596284

noncomputable def polynomial_irreducible (a n : ℤ) (p : ℕ) (hp : p.prime) (h : p > (|a| + 1)) : Prop :=
  ¬(∃ g h : Polynomial ℤ, g.degree < (Polynomial.X^n + Polynomial.C a * Polynomial.X + Polynomial.C p).degree ∧ h.degree < (Polynomial.X^n + Polynomial.C a * Polynomial.X + Polynomial.C p).degree ∧ g * h = (Polynomial.X^n + Polynomial.C a * Polynomial.X + Polynomial.C p))

theorem irreducible_polynomial (a n : ℤ) (p : ℕ) (hp : p.prime) (h : p > (|a| + 1)) : 
  polynomial_irreducible a n p hp h :=
by
  sorry

end irreducible_polynomial_l596_596284


namespace min_value_of_sequence_l596_596592

noncomputable def geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  ( ∀ k, a k > 0 )  -- positive sequence
  ∧ ( ∃ q, ∀ k, a (k+1) = q * a k )  -- geometric sequence
  ∧ ( a 7 = a 6 + 2 * a 5 )
  ∧ ( sqrt (a m * a n) = 4 * a 1 )
  ∧ ( n ≠ 0 ∧ m ≠ 0 )

theorem min_value_of_sequence (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence_min_value a m n → (1 / m + 5 / n) = 7 / 4 :=
by {
  -- Proof goes here.
  sorry
}

end min_value_of_sequence_l596_596592


namespace maximum_possible_value_of_N_l596_596211

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596211


namespace matrix_multiplication_result_l596_596298

-- Definitions of matrices P and Q
variables (P Q : Matrix (Fin 2) (Fin 2) ℝ)

-- Conditions
axiom cond1 : P + Q = P * Q
axiom cond2 : P * Q = λ (i j : Fin 2), [ [4, 0], [-2, 3] ] i j

-- Theorem to prove
theorem matrix_multiplication_result : Q * P = λ (i j : Fin 2), [ [4, 0], [-2, 3] ] i j := 
sorry

end matrix_multiplication_result_l596_596298


namespace missing_digit_in_20th_rising_number_l596_596441

def is_rising_number (n : ℕ) : Prop :=
  let digits := n.digits in
  (digits.length = 4) ∧ (∀ i j, (0 ≤ i ∧ i < j ∧ j < digits.length) → digits.nth i < digits.nth j)

def count_rising_numbers : ℕ :=
  nat.choose 9 4

def nth_rising_number (n : ℕ) : ℕ :=
  sorry -- Placeholder logic to get the nth rising number

theorem missing_digit_in_20th_rising_number :
  (count_rising_numbers = 126) →
  (20 < 126) →
  ¬ 3 ∈ (nth_rising_number 20).digits :=
by
  intros hcount hbound
  sorry

end missing_digit_in_20th_rising_number_l596_596441


namespace points_concyclic_l596_596287

structure Triangle (ABC : Type) :=
  (A B C : ABC) 

structure CollinearPoints (ABC : Type) (BC : Set ABC) :=
  (A1 A2 : ABC)
  (collinear : ∀ (B C : ABC), A1 ∈ BC ∧ B ∈ BC ∧ C ∈ BC ∧ A2 ∈ BC)

structure EqualSegments (ABC : Type) :=
  (A B C A1 A2 B1 B2 C1 C2 : ABC) 
  (A1B_eq_AC : Distance ABC A1 B = Distance ABC A C)
  (CA2_eq_AB : Distance ABC C A2 = Distance ABC A B)
  (B1A_eq_BC : Distance ABC B1 A = Distance ABC B C)
  (AB2_eq_BC : Distance ABC A B2 = Distance ABC B C)
  (C1A_eq_AB : Distance ABC C1 A = Distance ABC A B)
  (BC2_eq_CA : Distance ABC B C2 = Distance ABC C A)

theorem points_concyclic (ABC : Type) [MetricSpace ABC] 
  (triangle : Triangle ABC)
  (BC AC AB : Set ABC) 
  (collinear_A : CollinearPoints ABC BC)
  (collinear_B : CollinearPoints ABC AC)
  (collinear_C : CollinearPoints ABC AB)
  (equal_segments : EqualSegments ABC) : 
  Concyclic ABC (Set.ofList [A1, A2, B1, B2, C1, C2]) := 
sorry

end points_concyclic_l596_596287


namespace parabola_equation_l596_596805

open Real

theorem parabola_equation : ∃ (a b c d e f : ℤ),
  (a = 0) ∧ (b = 0) ∧ (c = 1) ∧ (d = -8) ∧ (e = -8) ∧ (f = 16) ∧
  (c > 0) ∧ gcd (gcd (gcd (gcd (gcd a b) c) d) e) f = 1 ∧
  ∀ (x y : ℝ), (y^2 - 8*x - 8*y + 16 = 0 ↔
  ((x = 2 ∧ y = 8) ∨ (∃ k : ℝ, y = 4 + k ∧ x = 1/8 * (y - 4)^2))) :=
by
  existsi [0, 0, 1, -8, -8, 16]
  repeat { split }
  any_goals { linarith }
  simp only [gcd_mul_left, gcd_comm (|a|) (|b|), Int.gcd_eq_zero_iff]
  sorry

end parabola_equation_l596_596805


namespace calculate_savings_l596_596005

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l596_596005


namespace vectors_perpendicular_l596_596634

def vector (ℝ : Type) (n : Type) := vector ℝ n

variable {ℝ : Type} [field ℝ] -- Assuming ℝ is a field (e.g., real numbers)
variable (a b : vector ℝ (fin 2))

noncomputable def a' : vector ℝ (fin 2) := vector.of_fn ![2, -3]
noncomputable def b' : vector ℝ (fin 2) := vector.of_fn ![3, 2]

theorem vectors_perpendicular : dot_product a' b' = 0 → perpendicular a' b' :=
sorry

end vectors_perpendicular_l596_596634


namespace angle_XYZ_45_degrees_l596_596337

-- Axioms for hexagon properties
def regular_hexagon_angle : ℝ := 120
def square_angle : ℝ := 90
def triangle_internal_sum : ℝ := 180

-- Variables for points representing vertices
variables (X Y Z D E : Point)

-- Assumption: Regular hexagon with side-angle conditions
axiom hexagon_consecutive_vertices 
    (X Y Z : Point) : 
    angle Y D Z = regular_hexagon_angle - square_angle ∧
    angle D Y Z = angle D Z Y

-- Proof that angle XYZ is 45 degrees
theorem angle_XYZ_45_degrees 
    (X Y Z of_hexagon : Point)
    (shared_side: square):
    angle XYZ = 45 :=
by
  -- Establish the conditions from our abstract definitions
  have h1 : angle Y D Z = 30 := 
    calc angle Y D Z = 120 - 90 : by admit sorry
    ... triv
  
  -- Calculate angles in the isosceles triangle
  have h2 : 2 * angle1 = 150 :=
    calc sum(angles) = 180 :
    ... form ? actual_proof
  ... triv

  -- Calculate the final angle
  trivial sorry
  exact 45 sorry 

#align angle_XYZ_45_degrees

end angle_XYZ_45_degrees_l596_596337


namespace line_l_l596_596623

theorem line_l'_equation 
  (l : ∀ (x y : ℝ), x - (sqrt 3) * y + 6 = 0)
  (point : ℝ × ℝ)
  (hl : point = (0, 1))
  (hangle : 2 * atan (1 / (sqrt 3)) = atan (sqrt 3)) :
  ∃ k : ℝ, (k * x - y + 1 = 0) ∧ (k = sqrt 3) :=
by
  sorry

end line_l_l596_596623


namespace complex_number_in_third_quadrant_l596_596601

theorem complex_number_in_third_quadrant :
  let i := complex.I in
  let z := (1 - 2 * i) / i in
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_in_third_quadrant_l596_596601


namespace vitya_catches_up_in_5_minutes_l596_596849

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596849


namespace darnell_fabric_left_l596_596985

/-- Darnell has 1000 square feet of fabric initially. 
He makes several types of flags with specified dimensions. 
Given the number of each type of flag made, we want to prove 
the remaining fabric left. 
-/
theorem darnell_fabric_left (f_total : ℕ) (sq_count : ℕ) (sq_area : ℕ) (wide_count : ℕ) (wide_area : ℕ) (tall_count : ℕ) (tall_area : ℕ) :
  f_total = 1000 → 
  sq_area = 16 → 
  wide_area = 15 → 
  tall_area = 15 → 
  sq_count = 16 → 
  wide_count = 20 → 
  tall_count = 10 → 
  let fabric_used := (sq_count * sq_area) + (wide_count * wide_area) + (tall_count * tall_area) in
    let fabric_left := f_total - fabric_used in
      fabric_left = 294 :=
by intros; sorry

end darnell_fabric_left_l596_596985


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l596_596824

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l596_596824


namespace set_intersection_complement_l596_596631

def U := ℝ
def A := { x : ℝ | 2^x > 1 / 2 }
def B := { x : ℝ | log x / log 3 < 1 }

theorem set_intersection_complement (x : ℝ) :
  x ∈ A ∩ (U \ B) ↔ (-1 < x ∧ x ≤ 0) ∨ (x ≥ 3) :=
sorry

end set_intersection_complement_l596_596631


namespace take_home_pay_is_correct_l596_596716

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l596_596716


namespace find_y_of_set_with_mean_l596_596604

theorem find_y_of_set_with_mean (y : ℝ) (h : ((8 + 15 + 20 + 6 + y) / 5 = 12)) : y = 11 := 
by 
    sorry

end find_y_of_set_with_mean_l596_596604


namespace transformed_graph_correct_l596_596804

def g (x : ℝ) : ℝ :=
  if x ∈ set.Icc (-4 : ℝ) (0 : ℝ) then -x
  else if x ∈ set.Icc (0 : ℝ) (4 : ℝ) then x - 2
  else 0 -- Assuming g(x) is 0 outside given ranges for well-definedness

theorem transformed_graph_correct :
  (∀ x, x ∈ set.Icc (-4 : ℝ) (0 : ℝ) → (1 / 3 * g x - 2 = -x / 3 - 2)) ∧
  (∀ x, x ∈ set.Icc (0 : ℝ) (4 : ℝ) → (1 / 3 * g x - 2 = x / 3 - 8 / 3)) :=
by {
  sorry,
}

end transformed_graph_correct_l596_596804


namespace g_triple_3_eq_31_l596_596757

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 1 else 2 * n - 3

theorem g_triple_3_eq_31 : g (g (g 3)) = 31 := by
  sorry

end g_triple_3_eq_31_l596_596757


namespace susan_age_l596_596784

theorem susan_age (S J B : ℝ) 
  (h1 : S = 2 * J)
  (h2 : S + J + B = 60) 
  (h3 : B = J + 10) : 
  S = 25 := sorry

end susan_age_l596_596784


namespace average_age_increase_l596_596258

theorem average_age_increase (A : ℝ) :
  let ages_of_men := [26, 30]
  let ages_of_women := [42, 42]
  let num_of_people := 7
  let total_age_original := num_of_people * A
  let total_age_men_removed := total_age_original - ages_of_men.sum
  let total_age_with_women := total_age_men_removed + ages_of_women.sum
  let new_average_age := total_age_with_women / num_of_people
  new_average_age - A = 4 :=
by
  sorry

end average_age_increase_l596_596258


namespace tan_arccot_eq_5_div_3_l596_596028

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596028


namespace sum_of_roots_of_poly_eq_14_over_3_l596_596499

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l596_596499


namespace vitya_catchup_time_l596_596878

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596878


namespace max_possible_cities_traversed_l596_596207

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596207


namespace math_problem_l596_596087

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596087


namespace sum_of_roots_l596_596507

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l596_596507


namespace new_concentration_is_correct_l596_596427

-- Let Q be the initial quantity of the 70% solution.
variable (Q : ℝ)

-- Define the initial concentration (70%) and replacement fraction (7/9).
def initial_concentration : ℝ := 0.70
def replaced_fraction : ℝ := 7 / 9

-- Define the concentration of the replacing solution (25%).
def replacing_concentration : ℝ := 0.25

-- Define the remaining fraction of the original solution.
def remaining_fraction : ℝ := 1 - replaced_fraction

-- Calculate the new concentration of the solution.
def new_concentration : ℝ := 
  (remaining_fraction * initial_concentration + replaced_fraction * replacing_concentration)

-- The theorem stating that the new concentration is 35%.
theorem new_concentration_is_correct : new_concentration Q = 0.35 :=
by
  sorry

end new_concentration_is_correct_l596_596427


namespace avery_tom_work_time_l596_596714

theorem avery_tom_work_time :
  let t := 1 in 
  (5 / 6) * t + (1 / 2) * (20.000000000000007 / 60) = 1 :=
by
  sorry

end avery_tom_work_time_l596_596714


namespace maximum_candies_l596_596371

/-
Define the stations and candy distribution.
Say Station A gives 5 candies.
Say Stations C and E give 3 candies each.
Say Stations B, D, and F give 1 candy each.
-/
def Station := Type
def candies : Station → ℕ
| "A" => 5
| "B" => 1
| "C" => 3
| "D" => 1
| "E" => 3
| "F" => 1
| _   => 0  -- Default case

/-
Define the path based on the given conditions.
Note: This path representation may vary depending on how paths are defined in the actual problem statement.
-/
def path : List Station := ["A", "B", "C", "D", "E", "C", "A", "E", "F", "A"]

/-
Define a function to count the number of times Jirka visits each station.
-/
def visitCounts : List Station → Station → ℕ
| [], _        => 0
| (s::rest), t => (if s = t then 1 else 0) + visitCounts rest t

/-
Calculate the total number of candies based on the visits.
-/
def totalCandies (p: List Station) : ℕ :=
  visitCounts p "A" * candies "A" + 
  visitCounts p "B" * candies "B" + 
  visitCounts p "C" * candies "C" + 
  visitCounts p "D" * candies "D" + 
  visitCounts p "E" * candies "E" + 
  visitCounts p "F" * candies "F"

/-
Statement to prove the maximum number of candies Jirka could receive.
-/
theorem maximum_candies : totalCandies path = 30 :=
by
  -- Assuming the calculations based on the paths and the given conditions
  -- Actual proof steps will be filled here.
  sorry

end maximum_candies_l596_596371


namespace max_possible_value_l596_596253

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596253


namespace area_of_triangle_ABC_l596_596561

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end area_of_triangle_ABC_l596_596561


namespace vitya_catch_up_time_l596_596860

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596860


namespace letter_distribution_in_each_row_l596_596550

-- Define the board dimensions
def board_rows : Nat := 2021
def board_cols : Nat := 2022

-- Define the total appearance condition for each letter
def total_appearance : Nat := board_rows * 674

-- Define properties
def no_adjacent_same_letter (board : List (List Char)) : Prop := sorry
def two_by_two_contains_all_letters (board : List (List Char)) : Prop := sorry
def letter_count_per_row (board : List (List Char)) (letter : Char) : Nat := sorry

theorem letter_distribution_in_each_row (board : List (List Char)) 
  (h1 : List.length board = board_rows)
  (h2 : ∀ row, List.length row = board_cols)
  (h3 : ∑ row in board, (λ row, letter_count_per_row row 'T') = total_appearance)
  (h4 : ∑ row in board, (λ row, letter_count_per_row row 'M') = total_appearance)
  (h5 : ∑ row in board, (λ row, letter_count_per_row row 'O') = total_appearance)
  (h6 : no_adjacent_same_letter board)
  (h7 : two_by_two_contains_all_letters board) 
  : ∀ row, letter_count_per_row row 'T' = 674 ∧ letter_count_per_row row 'M' = 674 ∧ letter_count_per_row row 'O' = 674 := sorry

end letter_distribution_in_each_row_l596_596550


namespace initial_processing_capacity_l596_596466

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l596_596466


namespace calculate_savings_l596_596013

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l596_596013


namespace simplify_expression_l596_596995

theorem simplify_expression (a : ℝ) (h : a / 2 - 2 / a = 3) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 :=
by
  sorry

end simplify_expression_l596_596995


namespace geom_seq_sum_l596_596267

theorem geom_seq_sum (n : ℕ) (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1/2)
                     (h2 : a 4 = 4) (h_geom : ∀ k, a (k+1) = a k * q) :
  ∑ i in finset.range n, a (i+1) = 2^(n-1) - 1/2 :=
sorry

end geom_seq_sum_l596_596267


namespace Arman_total_earnings_two_weeks_l596_596738

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end Arman_total_earnings_two_weeks_l596_596738


namespace abs_neg_2023_l596_596487

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596487


namespace Vitya_catches_mother_l596_596873

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596873


namespace minimal_elements_for_conditions_l596_596682

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, Nat.odd_iff_not_even.mpr (List.length_pos_of_ne_nil sorted).2])
  else
    let a := sorted.nth_le (sorted.length / 2 - 1) (by simp [Nat.sub_pos_of_lt (Nat.div_lt_self (List.length_pos_of_ne_nil sorted)].)
    let b := sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, List.length_pos_of_ne_nil sorted])
    (a + b) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def mode (s : List ℝ) : Option ℝ :=
  s.group_by id (· = ·).max_by (·.length).map (·.head)

def satisfies_conditions (s : List ℝ) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = some 6

theorem minimal_elements_for_conditions : ∀ s : List ℝ, satisfies_conditions s → s.length ≥ 6 :=
by
  intro s h
  sorry

end minimal_elements_for_conditions_l596_596682


namespace vaishali_total_stripes_l596_596845

theorem vaishali_total_stripes
  (hats1 : ℕ) (stripes1 : ℕ)
  (hats2 : ℕ) (stripes2 : ℕ)
  (hats3 : ℕ) (stripes3 : ℕ)
  (hats4 : ℕ) (stripes4 : ℕ)
  (total_stripes : ℕ) :
  hats1 = 4 → stripes1 = 3 →
  hats2 = 3 → stripes2 = 4 →
  hats3 = 6 → stripes3 = 0 →
  hats4 = 2 → stripes4 = 5 →
  total_stripes = (hats1 * stripes1) + (hats2 * stripes2) + (hats3 * stripes3) + (hats4 * stripes4) →
  total_stripes = 34 := by
  sorry

end vaishali_total_stripes_l596_596845


namespace convert_to_canonical_form_l596_596981

def quadratic_eqn (x y : ℝ) : ℝ :=
  8 * x^2 + 4 * x * y + 5 * y^2 - 56 * x - 32 * y + 80

def canonical_form (x2 y2 : ℝ) : Prop :=
  (x2^2 / 4) + (y2^2 / 9) = 1

theorem convert_to_canonical_form (x y : ℝ) :
  quadratic_eqn x y = 0 → ∃ (x2 y2 : ℝ), canonical_form x2 y2 :=
sorry

end convert_to_canonical_form_l596_596981


namespace parallelepiped_volume_l596_596569

-- Definitions according to conditions
noncomputable def side_length : ℝ := 1
noncomputable def acute_angle : ℝ := 60 * Real.pi / 180  -- Converting degrees to radians
noncomputable def sin_acute_angle : ℝ := Real.sin acute_angle

-- Area of the rhombus
noncomputable def rhombus_area : ℝ := side_length ^ 2 * sin_acute_angle

-- The height of the parallelepiped is equal to the side length of the square faces
noncomputable def height : ℝ := side_length

-- Volume of the parallelepiped
noncomputable def volume : ℝ := rhombus_area * height

-- Statement to prove
theorem parallelepiped_volume : volume = (Real.sqrt 3) / 2 := by
  sorry

end parallelepiped_volume_l596_596569


namespace smallest_set_size_l596_596675

noncomputable def smallest_num_elements (s : Multiset ℝ) : ℕ :=
  s.length

theorem smallest_set_size (s : Multiset ℝ) :
  (∀ a b c : ℝ, s = {a, b, 3, 6, 6, c}) →
  (s.median = 3) →
  (s.mean = 5) →
  (∀ x, s.count x < 3 → x ≠ 6) →
  smallest_num_elements s = 6 :=
by
  intros _ _ _ _
  sorry

end smallest_set_size_l596_596675


namespace product_of_roots_l596_596743

noncomputable def polynomial_minimal_root_625 : Polynomial ℚ :=
  Polynomial.sum (Polynomial.monomial 4 1) (Polynomial.monomial 0 (-5))

theorem product_of_roots :
  let P : Polynomial ℚ := polynomial_minimal_root_625 in
  (∏ root in P.roots, root) = -1600 :=
sorry

end product_of_roots_l596_596743


namespace collete_age_ratio_l596_596770

theorem collete_age_ratio (Ro R C : ℕ) (h1 : R = 2 * Ro) (h2 : Ro = 8) (h3 : R - C = 12) :
  C / Ro = 1 / 2 := by
sorry

end collete_age_ratio_l596_596770


namespace range_of_m_l596_596647

noncomputable def f (x : ℝ) : ℝ := sorry -- to be defined as an odd, decreasing function

theorem range_of_m 
  (hf_odd : ∀ x, f (-x) = -f x) -- f is odd
  (hf_decreasing : ∀ x y, x < y → f y < f x) -- f is strictly decreasing
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m, (0 < m ∧ m < 1) :=
sorry

end range_of_m_l596_596647


namespace infinite_valuation_increase_l596_596627

noncomputable def sequence (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), nat.factorial i

def p_power_in_factorization (p d : ℕ) : ℕ := nat.count p (nat.factors d)

theorem infinite_valuation_increase (p : ℕ) (hp : nat.prime p) :
  ∃ (a : ℕ) (infinitely_many : ∀ N : ℕ, ∃ a > N, p_power_in_factorization p (sequence (a + 1)) ≥ p_power_in_factorization p (sequence a)) := 
sorry

end infinite_valuation_increase_l596_596627


namespace binom_n_n_minus_2_l596_596897

theorem binom_n_n_minus_2 (n : ℕ) (h : n > 0) : nat.choose n (n-2) = n * (n-1) / 2 :=
by sorry

end binom_n_n_minus_2_l596_596897


namespace sqrt_series_solution_l596_596443

-- Define the sequence and the recurrence relation.
def sequence (a : ℕ → ℝ) := ∀ n > 2, a (n + 1) * (a (n - 1))^5 = (a n)^4 * (a (n - 2))^2

-- Initial values for the sequence
def initial_values (a : ℕ → ℝ) := a 1 = 8 ∧ a 2 = 64 ∧ a 3 = 1024

-- The main theorem to be proved
theorem sqrt_series_solution (a : ℕ → ℝ) 
  (h_seq : sequence a) 
  (h_ini : initial_values a) :
  sqrt (a 1 + sqrt (a 2 + sqrt (a 3 + ..))) = 3 * sqrt 2 := 
  sorry

end sqrt_series_solution_l596_596443


namespace new_mixture_concentration_l596_596922

def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.30
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40
def total_volume : ℝ := 8
def expected_concentration : ℝ := 37.5

theorem new_mixture_concentration :
  ((vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration) / total_volume) * 100 = expected_concentration :=
by
  sorry

end new_mixture_concentration_l596_596922


namespace x_days_worked_l596_596925

theorem x_days_worked (W : ℝ) :
  let x_work_rate := W / 20
  let y_work_rate := W / 24
  let y_days := 12
  let y_work_done := y_work_rate * y_days
  let total_work := W
  let work_done_by_x := (W - y_work_done) / x_work_rate
  work_done_by_x = 10 := 
by
  sorry

end x_days_worked_l596_596925


namespace minimum_value_1_minimum_value_2_l596_596583

noncomputable section

open Real -- Use the real numbers

theorem minimum_value_1 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + y^2 + z^2 >= 8 / 3 :=
by
  sorry  -- Proof omitted
 
theorem minimum_value_2 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + (y - 1)^2 + z^2 >= 6 :=
by
  sorry  -- Proof omitted

end minimum_value_1_minimum_value_2_l596_596583


namespace binary_computation_l596_596978

def b1101 := nat.of_digits 2 [1, 1, 0, 1]
def b0111 := nat.of_digits 2 [1, 1, 1]
def b1001 := nat.of_digits 2 [1, 0, 0, 1]
def b1010 := nat.of_digits 2 [1, 0, 1, 0]
def b10111 := nat.of_digits 2 [1, 0, 1, 1, 1]

theorem binary_computation :
  b1101 + b0111 - b1001 + b1010 = b10111 := by
  sorry

end binary_computation_l596_596978


namespace fibonacci_product_value_l596_596739

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

noncomputable def fibonacci_ratio_product : ℚ :=
  ∏ k in (Finset.range 148).filter (λ k, 3 ≤ k + 3).map (λ k, k + 3), 
    (fibonacci k / fibonacci (k - 2) - fibonacci k / fibonacci (k + 2))

theorem fibonacci_product_value :
  fibonacci_ratio_product = fibonacci 150 / fibonacci 152 := by
sorry

end fibonacci_product_value_l596_596739


namespace frequency_of_sixth_group_l596_596701

theorem frequency_of_sixth_group 
    (students_total : ℕ)
    (students_group_1 : ℕ)
    (students_group_2 : ℕ)
    (students_group_3 : ℕ)
    (students_group_4 : ℕ)
    (frequency_group_5 : ℚ) : 
    students_total = 40 → 
    students_group_1 = 10 → 
    students_group_2 = 5 → 
    students_group_3 = 7 → 
    students_group_4 = 6 → 
    frequency_group_5 = 0.2 → 
    (students_total - (students_group_1 + students_group_2 + students_group_3 + students_group_4) - (students_total * frequency_group_5)).to_rational / students_total = 0.1 :=
by
  intros h_total h_group1 h_group2 h_group3 h_group4 h_freq_group5
  sorry

end frequency_of_sixth_group_l596_596701


namespace area_of_isosceles_right_triangle_l596_596692

theorem area_of_isosceles_right_triangle (X Y Z : Type) 
  (X90 : angle X Y Z = 90)
  (XY_eq_XZ : XY = XZ)
  (YZ_hypotenuse : YZ = 10) : 
  area_of_triangle XY XZ YZ = 25 :=
by
  -- Proof omitted
  sorry

end area_of_isosceles_right_triangle_l596_596692


namespace no_super_sudoku_grids_l596_596051

-- A type to represent a 9x9 grid filled with numbers from 1 to 9
def Grid := Fin 9 → Fin 9 → Fin 9

-- Function to check if a given grid is a super-sudoku
def is_super_sudoku (g : Grid) : Prop :=
  (∀ i : Fin 9, ∃ perm : Fin 9 → Fin 9, bijective perm ∧ ∀ j : Fin 9, g i j = perm j) ∧
  (∀ j : Fin 9, ∃ perm : Fin 9 → Fin 9, bijective perm ∧ ∀ i : Fin 9, g i j = perm i) ∧
  (∀ i j, ∃ perm : Fin 9 → Fin 9, bijective perm ∧ ∀ k l, i / 3 * 3 + k < 9 ∧ j / 3 * 3 + l < 9 →
    g (i / 3 * 3 + k) (j / 3 * 3 + l) = perm (3 * k + l))

-- Statement that no super-sudoku grid exists
theorem no_super_sudoku_grids : ¬ ∃ g : Grid, is_super_sudoku g :=
by sorry

end no_super_sudoku_grids_l596_596051


namespace smallest_set_size_l596_596676

noncomputable def smallest_num_elements (s : Multiset ℝ) : ℕ :=
  s.length

theorem smallest_set_size (s : Multiset ℝ) :
  (∀ a b c : ℝ, s = {a, b, 3, 6, 6, c}) →
  (s.median = 3) →
  (s.mean = 5) →
  (∀ x, s.count x < 3 → x ≠ 6) →
  smallest_num_elements s = 6 :=
by
  intros _ _ _ _
  sorry

end smallest_set_size_l596_596676


namespace abs_neg_2023_l596_596479

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l596_596479


namespace ellipse_standard_eq_and_fixed_point_l596_596135

theorem ellipse_standard_eq_and_fixed_point (a b : ℝ) (h : a > b ∧ b > 0) (F₁ F₂ P : ℝ × ℝ) 
(hPF₁PF₂ : |P.1 - F₁.1| + |P.1 - F₂.1| = 4 * sqrt 2) (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1) :
  (∃ a b : ℝ, a = 2 * sqrt 2 ∧ b^2 = 2 ∧ (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
  (∀ M N : ℝ × ℝ, (M ≠ P ∧ N ≠ P ∧ (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧ (N.1^2 / a^2 + N.2^2 / b^2 = 1) 
  → ∃ (fixed_pt : ℝ × ℝ), fixed_pt = (6 / 5, -3 / 5) ∧ 
  (∀ (x : ℝ), ∃ (y : ℝ), y = M.2 + (x - M.1) * (N.2 - M.2) / (N.1 - M.1) → y = fixed_pt.2)) :=
begin
  sorry
end

end ellipse_standard_eq_and_fixed_point_l596_596135


namespace problem_f_2017_sum_l596_596582

def f : ℕ → (ℝ → ℝ)
| 0       := λ x, sin x + cos x
| (n + 1) := λ x, (deriv (f n)) x

theorem problem_f_2017_sum :
  (∑ i in Finset.range 2017, f i x) = sin x + cos x := by
  sorry

end problem_f_2017_sum_l596_596582


namespace donny_total_cost_eq_45_l596_596019

-- Definitions for prices of each type of apple
def price_small : ℝ := 1.5
def price_medium : ℝ := 2
def price_big : ℝ := 3

-- Quantities purchased by Donny
def count_small : ℕ := 6
def count_medium : ℕ := 6
def count_big : ℕ := 8

-- Total cost calculation
def total_cost (count_small count_medium count_big : ℕ) : ℝ := 
  (count_small * price_small) + (count_medium * price_medium) + (count_big * price_big)

-- Theorem stating the total cost
theorem donny_total_cost_eq_45 : total_cost count_small count_medium count_big = 45 := by
  sorry

end donny_total_cost_eq_45_l596_596019


namespace length_of_AB_l596_596704

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l596_596704


namespace speed_of_train_l596_596452

def distance : ℝ := 80
def time : ℝ := 6
def expected_speed : ℝ := 13.33

theorem speed_of_train : distance / time = expected_speed :=
by
  sorry

end speed_of_train_l596_596452


namespace problem_statement_l596_596391

theorem problem_statement (x y : ℤ) (h1 : x = 8) (h2 : y = 3) :
  (x - 2 * y) * (x + 2 * y) = 28 :=
by
  sorry

end problem_statement_l596_596391


namespace investment_amount_l596_596374

noncomputable def calculate_principal (A : ℕ) (r t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount (A : ℕ) (r t : ℝ) (n P : ℕ) :
  A = 70000 → r = 0.08 → t = 5 → n = 12 →
  P = 46994 →
  calculate_principal A r t n = P :=
by
  intros hA hr ht hn hP
  rw [hA, hr, ht, hn, hP]
  sorry

end investment_amount_l596_596374


namespace point_on_line_unique_u_l596_596096

theorem point_on_line_unique_u (u : ℝ) :
  (∃ a b : ℝ, (a ≠ b ∧ (u, -4) = (a, b))) →
  (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧ (u = x1 ∧ -4 = y1) ∧ (10 = x2 ∧ 3 = y2) ∧ (2 = x1 ∧ -1 = y1) ∧ (slope x1 y1 x2 y2 = (1/2))) →
  u = -4 :=
by
  sorry

end point_on_line_unique_u_l596_596096


namespace collinear_incenter_condition_l596_596752

open_locale real

/-- Let I be the incenter of triangle ABC, and M and N be the midpoints of the arcs ABC and BAC
of the circumcircle. Prove that the points M, I, and N are collinear if and only if AC + BC = 3AB. -/
theorem collinear_incenter_condition (A B C I M N : Type*) [euclidean_geometry A B C I M N] 
  (hI : incenter A B C I) 
  (hM : midpoint_of_arc A B C M) 
  (hN : midpoint_of_arc B A C N) : 
  (collinear M I N) ↔ (dist A C + dist B C = 3 * dist A B) := 
by sorry

end collinear_incenter_condition_l596_596752


namespace minimum_colors_needed_l596_596971

-- Definitions corresponding to the conditions of the problem
variables (regions : Fin 8 → Prop)
variables (adjacent : (Fin 8) → (Fin 8) → Prop)
variables (color : Fin 8 → Fin 4)
variables (diff_colors : ∀ (i j : Fin 8), adjacent i j → color i ≠ color j)

-- The proof problem in Lean 4
theorem minimum_colors_needed (h : ∀ (color : Fin 8 → Fin 3), 
                              ∃ (i j : Fin 8), 
                              adjacent i j ∧ color i = color j) : 
                              ∃ (color : Fin 8 → Fin 4), 
                              ∀ (i j : Fin 8), 
                              adjacent i j → color i ≠ color j :=
begin
  sorry
end

end minimum_colors_needed_l596_596971


namespace exponential_to_rectangular_form_l596_596526

theorem exponential_to_rectangular_form : 
  (Real.exp (Complex.i * (13 * Real.pi / 2))) = Complex.i :=
by
  sorry

end exponential_to_rectangular_form_l596_596526


namespace probability_divisible_by_five_l596_596819

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l596_596819


namespace length_of_AB_l596_596705

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l596_596705


namespace smallest_odd_n_l596_596387

theorem smallest_odd_n (n : ℕ) (odd : n % 2 = 1) :
  (2 : ℝ) ^ ((1 + 3 + List.sum (List.map (λ k, 2*k+1) (List.range n)) : ℝ) / 9) > 5000 ↔ n = 10 :=
by admit

end smallest_odd_n_l596_596387


namespace solve_for_x_values_for_matrix_l596_596063

def matrix_equals_neg_two (x : ℝ) : Prop :=
  let a := 3 * x
  let b := x
  let c := 4
  let d := 2 * x
  (a * b - c * d = -2)

theorem solve_for_x_values_for_matrix : 
  ∃ (x : ℝ), matrix_equals_neg_two x ↔ (x = (4 + Real.sqrt 10) / 3 ∨ x = (4 - Real.sqrt 10) / 3) :=
sorry

end solve_for_x_values_for_matrix_l596_596063


namespace abigail_time_to_finish_l596_596963

noncomputable def words_total : ℕ := 1000
noncomputable def words_per_30_min : ℕ := 300
noncomputable def words_already_written : ℕ := 200
noncomputable def time_per_word : ℝ := 30 / words_per_30_min

theorem abigail_time_to_finish :
  (words_total - words_already_written) * time_per_word = 80 :=
by
  sorry

end abigail_time_to_finish_l596_596963


namespace real_solution_set_l596_596997

theorem real_solution_set (x : ℝ) :
  (x - 2) / (x - 4) ≥ 3 → x ≠ 2 → x ∈ set.Ioo 4 5 ∨ x = 5 :=
by
  sorry

end real_solution_set_l596_596997


namespace grocer_display_proof_l596_596945

-- Define the arithmetic sequence conditions
def num_cans_in_display (n : ℕ) : Prop :=
  let a := 1
  let d := 2
  (n * n = 225) 

-- Prove the total weight is 1125 kg
def total_weight_supported (weight_per_can : ℕ) (total_cans : ℕ) : Prop :=
  (total_cans * weight_per_can = 1125)

-- State the main theorem combining the two proofs.
theorem grocer_display_proof (n weight_per_can total_cans : ℕ) :
  num_cans_in_display n → total_weight_supported weight_per_can total_cans → 
  n = 15 ∧ total_cans * weight_per_can = 1125 :=
by {
  sorry
}

end grocer_display_proof_l596_596945


namespace problem_solution_l596_596100

theorem problem_solution (x : ℝ) :
    (x^2 / (x - 2) ≥ (3 / (x + 2)) + (7 / 5)) →
    (x ∈ Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi (2 : ℝ)) :=
by
  intro h
  sorry

end problem_solution_l596_596100


namespace frog_climbing_time_l596_596943

-- Defining the conditions as Lean definitions
def well_depth : ℕ := 12
def climb_distance : ℕ := 3
def slip_distance : ℕ := 1
def climb_time : ℚ := 1 -- time in minutes for the frog to climb 3 meters
def slip_time : ℚ := climb_time / 3
def total_time_per_cycle : ℚ := climb_time + slip_time
def total_climbed_at_817 : ℕ := well_depth - 3 -- 3 meters from the top means it climbed 9 meters

-- The equivalent proof statement in Lean:
theorem frog_climbing_time : 
  ∃ (T : ℚ), T = 22 ∧ 
    (well_depth = 9 + 3) ∧
    (∀ (cycles : ℕ), cycles = 4 → 
         total_time_per_cycle * cycles + 2 = T) :=
by 
  sorry

end frog_climbing_time_l596_596943


namespace max_N_value_l596_596225

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596225


namespace calculate_savings_l596_596012

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l596_596012


namespace circle_radius_symmetry_l596_596146

theorem circle_radius_symmetry (m : ℝ) :
  (symmetric_points : ∀ M N : ℝ × ℝ, 
    (M.1^2 + M.2^2 - 2*M.1 + m * M.2 - 4 = 0) ∧ 
    (N.1^2 + N.2^2 - 2*N.1 + m * N.2 - 4 = 0) ∧ 
    (2*M.1 + M.2 = 0) ∧ (2*N.1 + N.2 = 0)) → 
  ((2*1 + (-m/2) = 0) → 
  (real_radius : ℝ) :
    real_radius = 3) :=
sorry

end circle_radius_symmetry_l596_596146


namespace length_of_AB_l596_596703

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l596_596703


namespace inequality_antichain_sum_antichain_size_bound_l596_596299

open Finset Nat

variable {α : Type*} [DecidableEq α] (n : ℕ) (A : Finset (Finset α))
  (hA : ∀ B C ∈ A, B ≠ C → ¬ B ⊆ C)

-- Define the main theorem statements

theorem inequality_antichain_sum :
  (∑ B in A, (1 : ℝ) / (choose (B.card) n)) ≤ 1 := by
  sorry

theorem antichain_size_bound :
  A.card ≤ choose n (n / 2) := by
  sorry

end inequality_antichain_sum_antichain_size_bound_l596_596299


namespace problem1_proof_ellipse_equation_problem2_proof_collinearity_problem3_max_area_l596_596136

noncomputable def ellipse_equation (a : ℝ) (h : a > real.sqrt 2) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / 2) = 1

theorem problem1_proof_ellipse_equation :
  ellipse_equation √6 (by norm_num) x y = (x^2 / 6) + (y^2 / 2) = 1 := 
sorry

noncomputable def eccentricity_relation (a e : ℝ) : Prop :=
  e = real.sqrt ((a^2 - 2) / a^2)

theorem problem2_proof_collinearity (a : ℝ) (ha : a = real.sqrt 6) (c : ℝ) 
  (hc : c = (real.sqrt 6 * real.sqrt 6) / 3) : 
  ∀ (A P Q M F : point), 
  P = rotate_about_y A M ∧ M = reflect P 
  ∧ collinear F P Q := 
sorry

theorem problem3_max_area (a : ℝ) 
  (ha : a = real.sqrt 6) (c : ℝ) 
  (hc : c = (a * 2) / 3) : 
  ∀ (F P Q : point),
  maximized_area_triangle F P Q → line_eq_PQ x y := 
sorry

end problem1_proof_ellipse_equation_problem2_proof_collinearity_problem3_max_area_l596_596136


namespace least_repeating_block_length_l596_596798

theorem least_repeating_block_length (n d : ℚ) (h1 : n = 7) (h2 : d = 13) (h3 : (n / d).isRepeatingDecimal) : 
  ∃ k : ℕ, k = 6 ∧ ∃ m : ℕ, lenRecBlock (fractionToDecimal (n / d)) m k := 
by 
  sorry

end least_repeating_block_length_l596_596798


namespace sum_of_roots_of_poly_eq_14_over_3_l596_596500

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l596_596500


namespace parametric_line_eq_l596_596412

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_from_points (p1 p2 : Point3D) : Point3D :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def cross_product (v1 v2 : Point3D) : Point3D :=
  ⟨v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x⟩

def parametric_eq (M n : Point3D) (t : ℝ) : Point3D :=
  ⟨M.x + n.x * t, M.y + n.y * t, M.z + n.z * t⟩

theorem parametric_line_eq :
  let M := Point3D.mk (-2) 0 3 in
  let A := Point3D.mk (-3) 0 1 in
  let P := Point3D.mk (-1) 2 5 in
  let Q := Point3D.mk 3 (-4) 1 in
  let AP := vector_from_points A P in
  let AQ := vector_from_points A Q in
  let n := cross_product AP AQ in
  let n_simplified := Point3D.mk 4 6 (-5) in
  ∀ t : ℝ, parametric_eq M n_simplified t = ⟨-2 + 4 * t, 6 * t, 3 - 5 * t⟩ :=
by
  sorry

end parametric_line_eq_l596_596412


namespace sum_g_h_l596_596476

theorem sum_g_h (d g h : ℝ) 
  (h1 : (8 * d^2 - 4 * d + g) * (4 * d^2 + h * d + 7) = 32 * d^4 + (4 * h - 16) * d^3 - (14 * d^2 - 28 * d - 56)) :
  g + h = -8 :=
sorry

end sum_g_h_l596_596476


namespace sampling_method_correct_l596_596944

variables {Grade : Type} [Fintype Grade] 
variables (classes : Finset Grade) (n_students : ℕ) (student_number : Grade → ℕ)

def systematic_sampling (classes : Finset Grade) (student_number : Grade → ℕ) : Prop :=
  ∀ c ∈ classes, student_number c = 40

theorem sampling_method_correct : 
  (∃ (n_classes : ℕ) (n_students : ℕ) (n_selected : ℕ) (classes : Finset Grade) (student_number : Grade → ℕ), 
    n_classes = 12 ∧ 
    n_students = 50 ∧ 
    n_selected = 1 ∧ 
    ∀ c ∈ classes, student_number c ∈ finset.range n_students.succ ∧ 
    systematic_sampling classes student_number) → 
  "Systematic Sampling Method" = "Systematic Sampling Method" :=
by 
  intros h,
  sorry

end sampling_method_correct_l596_596944


namespace train_speed_l596_596451

theorem train_speed (length : ℝ) (time_seconds : ℝ) (speed : ℝ) :
  length = 320 → time_seconds = 16 → speed = 72 :=
by 
  sorry

end train_speed_l596_596451


namespace vitya_catchup_time_l596_596877

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596877


namespace abs_neg_2023_l596_596486

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596486


namespace find_p_l596_596947

def point := (ℝ × ℝ)

noncomputable def line_through_point_with_slope (P : point) (m : ℝ) : (ℝ → ℝ) :=
  λ x, m * (x - P.1) + P.2

def parabola (p : ℝ) := {P : point | P.2 ^ 2 = 2 * p * P.1}

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem find_p (p : ℝ) (hP : p > 0) :
  let M : point := (2, 1),
      L := line_through_point_with_slope M 1,
      A B : point := sorry in  -- We assume we have points A and B such that...
    L A.1 = A.2 ∧ L B.1 = B.2 ∧
    (A ∈ parabola p) ∧ (B ∈ parabola p) ∧ is_midpoint M A B
  → p = 1 :=
by
  intro hP
  -- skip the proof
  sorry

end find_p_l596_596947


namespace sum_of_ratios_geq_sum_l596_596584

theorem sum_of_ratios_geq_sum (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ i : Fin n, x i > 0) : 
  ∑ i, (x i) ^ 2 / (x (i + 1) % n) ≥ ∑ i, x i :=
sorry

end sum_of_ratios_geq_sum_l596_596584


namespace third_trial_point_l596_596064

noncomputable def interval : set ℝ := set.Icc 2 4
noncomputable def golden_ratio : ℝ := 0.618
noncomputable def x1 : ℝ := 2 + golden_ratio * (4 - 2)
noncomputable def x2 : ℝ := 2 + (4 - x1)
noncomputable def x3 : ℝ := 4 - golden_ratio * (4 - x1)

theorem third_trial_point :
  x3 = 3.528 := sorry

end third_trial_point_l596_596064


namespace gold_copper_ratio_one_to_one_l596_596635

variable (G C : ℝ)
variable (heaviness_gold heaviness_copper heaviness_alloy : ℝ)

def heaviness_gold := 10
def heaviness_copper := 6
def heaviness_alloy := 8

theorem gold_copper_ratio_one_to_one (h : (heaviness_gold * G + heaviness_copper * C) / (G + C) = heaviness_alloy) : 
  G = C :=
by
  sorry

end gold_copper_ratio_one_to_one_l596_596635


namespace distinct_integer_condition_l596_596996

theorem distinct_integer_condition (n : ℕ) (hn : n > 0) : 
  (∃ (a : Fin n → ℤ), Function.Injective a ∧ 
   (∑ i in Finset.univ, (i + 1 : ℕ) / (a i) = 
    (∑ i in Finset.univ, a i) / 2)) -> 
  n ≥ 3 := 
sorry

end distinct_integer_condition_l596_596996


namespace sum_of_roots_l596_596505

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l596_596505


namespace max_roads_city_condition_l596_596236

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596236


namespace large_jar_capacity_l596_596368

theorem large_jar_capacity :
  ∀ (total_jars small_jar_capacity small_jars total_volume large_jars_capacity : ℕ),
    total_jars = 100 →
    small_jar_capacity = 3 →
    small_jars = 62 →
    total_volume = 376 →
    large_jars_capacity = 5 :=
begin
  intros _ _ _ _ _,
  sorry
end

end large_jar_capacity_l596_596368


namespace ceiling_and_floor_calculation_l596_596094

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596094


namespace tax_percentage_first_tier_l596_596543

theorem tax_percentage_first_tier
  (car_price : ℝ)
  (total_tax : ℝ)
  (first_tier_level : ℝ)
  (second_tier_rate : ℝ)
  (first_tier_tax : ℝ)
  (T : ℝ)
  (h_car_price : car_price = 30000)
  (h_total_tax : total_tax = 5500)
  (h_first_tier_level : first_tier_level = 10000)
  (h_second_tier_rate : second_tier_rate = 0.15)
  (h_first_tier_tax : first_tier_tax = (T / 100) * first_tier_level) :
  T = 25 :=
by
  sorry

end tax_percentage_first_tier_l596_596543


namespace sufficient_not_necessary_condition_for_positive_quadratic_l596_596645

variables {a b c : ℝ}

theorem sufficient_not_necessary_condition_for_positive_quadratic 
  (ha : a > 0)
  (hb : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x ^ 2 + b * x + c > 0) 
  ∧ ¬ (∀ x : ℝ, ∃ a b c : ℝ, a > 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (a * x ^ 2 + b * x + c > 0)) :=
by
  sorry

end sufficient_not_necessary_condition_for_positive_quadratic_l596_596645


namespace students_arrangements_l596_596830

theorem students_arrangements (n : ℕ) (k : ℕ) (h₁ : n = 5) (h₂ : k = 1) : 
  (nat.factorial n) - (nat.factorial (n - k)) = 96 := 
by 
  -- sorry allows us to skip the proof
  sorry

end students_arrangements_l596_596830


namespace y_payment_is_approximately_272_73_l596_596924

noncomputable def calc_y_payment : ℝ :=
  let total_payment : ℝ := 600
  let percent_x_to_y : ℝ := 1.2
  total_payment / (percent_x_to_y + 1)

theorem y_payment_is_approximately_272_73
  (total_payment : ℝ)
  (percent_x_to_y : ℝ)
  (h1 : total_payment = 600)
  (h2 : percent_x_to_y = 1.2) :
  calc_y_payment = 272.73 :=
by
  sorry

end y_payment_is_approximately_272_73_l596_596924


namespace projection_matrix_property_l596_596470

variable (x y : ℚ)

def Q : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 21 / 49], ![y, 35 / 49]]

def Q_squared : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.mul Q Q

theorem projection_matrix_property :
  Q = ![![x, 21 / 49], ![y, 35 / 49]] →
  Q_squared = Q →
  x = 666 / 2401 ∧ y = (49 * 2401) / 1891 := by
  intros hQ hQ_squared
  sorry

end projection_matrix_property_l596_596470


namespace angle_between_vectors_l596_596167

variables {α β : ℝ}

/-- Given vectors a = (cos α, sin α) and b = (cos β, sin β), where α - β = 2π/3, 
    prove that the angle between a and (a + b) is π/3. --/
theorem angle_between_vectors (α β : ℝ) (h : α - β = (2 * Real.pi) / 3) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.cos β, Real.sin β)
  in vector.angle a (a.1 + b.1, a.2 + b.2) = Real.pi / 3 :=
begin
  sorry
end

end angle_between_vectors_l596_596167


namespace fg_of_3_l596_596179

open Real

def f (x : ℝ) : ℝ := 4 - sqrt x
def g (x : ℝ) : ℝ := -3 * x + 3 * x^2

theorem fg_of_3 : f (g 3) = 4 - 3 * sqrt 2 := by
  sorry

end fg_of_3_l596_596179


namespace no_three_distinct_integers_solving_polynomial_l596_596404

theorem no_three_distinct_integers_solving_polynomial (p : ℤ → ℤ) (hp : ∀ x, ∃ k : ℕ, p x = k • x + p 0) :
  ∀ a b c : ℤ, a ≠ b → b ≠ c → c ≠ a → p a = b → p b = c → p c = a → false :=
by
  intros a b c hab hbc hca hpa_hp pb_pc_pc
  sorry

end no_three_distinct_integers_solving_polynomial_l596_596404


namespace given_expression_equality_l596_596122

theorem given_expression_equality (x : ℝ) (A ω φ b : ℝ) (hA : 0 < A)
  (h : 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) :
  A = Real.sqrt 2 ∧ b = 1 :=
sorry

end given_expression_equality_l596_596122


namespace Vitya_catches_mother_l596_596871

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596871


namespace percentage_cut_is_20_l596_596367

-- Define the conditions
def original_budget : ℝ := 940.00
def new_budget : ℝ := 752.00

-- Define the amount of cut
def amount_of_cut : ℝ := original_budget - new_budget

-- Define the percentage cut calculation
def percentage_cut : ℝ := (amount_of_cut / original_budget) * 100

-- The proof statement
theorem percentage_cut_is_20 : percentage_cut = 20 := by
  -- amount_of_cut = 188
  have h1 : amount_of_cut = 940.00 - 752.00 := rfl
  -- percentage_cut = (amount_of_cut / 940.00) * 100
  have h2 : percentage_cut = (188 / 940) * 100 := by
    rw [h1]
    sorry
  -- percentage_cut = 20
  sorry

end percentage_cut_is_20_l596_596367


namespace cyclic_sum_le_one_div_2014_l596_596124

open List

def cyclic_sum {α : Type*} [Semiring α] (l : List α) : α :=
match l with
| [] => 0
| h::t => (h * t.headI) + cyclic_sum (t ++ [h])

theorem cyclic_sum_le_one_div_2014 (a : Fin 2014 → ℝ) (h₁ : ∀ i, 0 ≤ a i) (h₂ : (Finset.univ.sum a) = 1) :
  ∃ p : List (Fin 2014), p.perm (Finset.univ.attach.map (λ i, ↑i)) ∧ cyclic_sum (p.map a) ≤ 1 / 2014 := by
sorry

end cyclic_sum_le_one_div_2014_l596_596124


namespace maximum_possible_value_of_N_l596_596214

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596214


namespace two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l596_596828

-- Define the number of boys and girls
def boys : ℕ := 2
def girls : ℕ := 3
def total_people : ℕ := boys + girls

-- Define assumptions about arrangements
def arrangements_in_two_rows : ℕ := sorry
def arrangements_with_person_A_not_head_tail : ℕ := sorry
def arrangements_with_girls_together : ℕ := sorry
def arrangements_with_boys_not_adjacent : ℕ := sorry

-- State the mathematical equivalence proof problems
theorem two_rows_arrangement : arrangements_in_two_rows = 60 := 
  sorry

theorem person_A_not_head_tail_arrangement : arrangements_with_person_A_not_head_tail = 72 := 
  sorry

theorem girls_together_arrangement : arrangements_with_girls_together = 36 := 
  sorry

theorem boys_not_adjacent_arrangement : arrangements_with_boys_not_adjacent = 72 := 
  sorry

end two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l596_596828


namespace tan_arccot_of_frac_l596_596037

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596037


namespace number_of_true_propositions_l596_596463

theorem number_of_true_propositions 
    (P1 : ∀ (p : ∃ l : line, ∀ P : point, ∃! l' : line, l' ⊥ l ∧ passes_through l' P), Prop)
    (P2 : ∀ {x : ℝ}, x^(1/2) = x^(1/3) → x = 0 ∨ x = 1)
    (P3 : ∀ {a b c : line}, a ⊥ b → b ⊥ c → ¬(a ⊥ c))
    (P4 : ∀ (A : point) (l : line), (∀ P : point, ¬(passes_through P l) → dist(A, l) ≥ 5) → dist(A, l) = 5)
    (P5 : ∀ (x : ℝ), is_irrational x → ¬(x = 0)) :
    (P1 true) ∧ 
    (P2 true) ∧ 
    (P3 false) ∧ 
    (P4 true) ∧ 
    (P5 false) →
    true_propositions = 2 := 
by 
  -- Proof is omitted
  sorry

end number_of_true_propositions_l596_596463


namespace maximum_N_value_l596_596229

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596229


namespace degree_d_l596_596439

def f (x : ℝ) := -- Polynomial of degree 15.
  sorry

def q (x : ℝ) := -- Quotient polynomial of degree 9.
  sorry

def r (x : ℝ) := 5 * x^4 + 3 * x^3 - 2 * x + 8 -- Remainder.

theorem degree_d (deg_f : ∀ x : ℝ, degree(f x) = 15)
  (deg_q : ∀ x : ℝ, degree(q x) = 9)
  (deg_r : ∀ x : ℝ, degree(r x) = 4) :
  ∀ d : ℝ → ℝ, (∀ x : ℝ, f x = d x * q x + r x) → degree(d) = 6 := by
  sorry

end degree_d_l596_596439


namespace quadrilateral_sides_quadrilateral_diagonals_quadrilateral_area_l596_596575

noncomputable def quadrilateral_properties (R : ℝ) := 
  let A := (R, 0)
  let B := (R * Real.cos (π / 3), R * Real.sin (π / 3))
  let C := (R * Real.cos (3 * π / 4), R * Real.sin (3 * π / 4))
  let D := (R * Real.cos (5 * π / 3), R * Real.sin (5 * π / 3)) in
  let AB := dist A B
  let BC := dist B C
  let CD := dist C D
  let DA := dist D A in
  let AC := dist A C
  let BD := dist B D in
  let area := R^2 * (Real.sqrt 3 / 2) * Real.sqrt 2 in
  (AB = R) ∧ (BC = R * Real.sqrt 2) ∧ (CD = R * Real.sqrt 3) ∧ (DA = R * Real.sqrt 2)
  ∧ ∠(AC, BD) = 90 
  ∧ area_quadrilateral ABCD = area

/-- First, the distances for each side of quadrangulal. --/
theorem quadrilateral_sides (R : ℝ) :
  let (AB, BC, CD, DA) := quadrilateral_properties R in
  AB = R ∧ BC = R * Real.sqrt 2 ∧ CD = R * Real.sqrt 3 ∧ DA = R * Real.sqrt 2 := sorry

/-- Next, we show the quadrilateral diagonals are perpendicular to each other --/
theorem quadrilateral_diagonals (R : ℝ) :
  let (_, _, _, _, ∠ACBD) := quadrilateral_properties R in
  ∠ACBD = 90 := sorry

/-- Finally, we compute the area of this quadrilateral and show it matches expectations --/  
theorem quadrilateral_area (R : ℝ) :
  let (_, _, _, _, _, area) := quadrilateral_properties R in
  area = R^2 * (Real.sqrt 3 / 2) * Real.sqrt 2 := sorry

end quadrilateral_sides_quadrilateral_diagonals_quadrilateral_area_l596_596575


namespace chord_length_l596_596938

open Real

theorem chord_length (radius : ℝ) (dist_to_chord : ℝ) (h_radius : radius = 5) (h_dist_to_chord : dist_to_chord = 4) :
  ∃ EF : ℝ, EF = 6 :=
by
  have OG : ℝ := 4
  have OE : ℝ := 5
  have GE : ℝ := sqrt (OE^2 - OG^2)
  have h_GE : GE = 3
  use 2 * GE
  simp [h_GE]
  norm_num
  exact sorry

end chord_length_l596_596938


namespace josie_substitution_l596_596730

variables (a b c d e : ℤ)

theorem josie_substitution :
  a = 2 → b = 1 → c = -1 → d = 3 →
  ( a - b + c^2 - d + e = a - (b - (c^2 - (d + e))) ) →
  e = 0 :=
by
  intros ha hb hc hd heq
  change a with 2 at ha
  change b with 1 at hb
  change c with -1 at hc
  change d with 3 at hd
  rw [ha, hb, hc, hd] at heq
  -- The expressions become:
  -- - 1 + e = -1 - e if we directly plug-in the values
  -- Hence, simplifying that would directly be used.
  sorry

end josie_substitution_l596_596730


namespace sum_of_roots_l596_596509

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l596_596509


namespace alcohol_quantity_l596_596809

theorem alcohol_quantity (A W : ℕ) (h1 : 4 * W = 3 * A) (h2 : 4 * (W + 8) = 5 * A) : A = 16 := 
by
  sorry

end alcohol_quantity_l596_596809


namespace coefficient_x3_l596_596795

theorem coefficient_x3 (C : ℕ → ℕ → ℕ) (hC : ∀ n k, C n k = Nat.choose n k) :
  ∃ c : ℕ, c = 54 ∧ ∃ f : (ℕ → ℕ → ℕ), f = (λ n k, 3^k * C n k) ∧ 
  ∃ r : ℕ, r = 2 ∧
  ∃ n : ℕ, n = 4 ∧
  ∃ x : ℕ, x = 3 ∧ 
 (x * (1 + 3 * x)^4).third_coeff = 54 := sorry

end coefficient_x3_l596_596795


namespace binomial_n_choose_n_sub_2_l596_596898

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end binomial_n_choose_n_sub_2_l596_596898


namespace two_pow_n_plus_one_square_or_cube_l596_596061

theorem two_pow_n_plus_one_square_or_cube (n : ℕ) :
  (∃ a : ℕ, 2^n + 1 = a^2) ∨ (∃ a : ℕ, 2^n + 1 = a^3) → n = 3 :=
by
  sorry

end two_pow_n_plus_one_square_or_cube_l596_596061


namespace cyclist_encounters_l596_596927

theorem cyclist_encounters (n : ℕ) (S : ℝ) (v : Fin (2 * n) → ℝ) 
  (distinct_speeds : ∀ i j, i ≠ j → v i ≠ v j)
  (meet_at_least_once : ∀ i j, i ≠ j → ∃ t : ℝ, t ≤ 12 * 3600 ∧ (S * t) % (v j - v i) = 0)
  (no_three_meet_simultaneously : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ∀ t : ℝ, (S * t) % (v j - v i) = 0 → (S * t) % (v k - v j) ≠ 0) 
  : ∀ i, ∃ meets : ℕ, meets ≥ n^2 := 
by
  sorry

end cyclist_encounters_l596_596927


namespace simplify_sqrt_product_l596_596777

theorem simplify_sqrt_product : 
    real.sqrt (9 * 4) * real.sqrt (3 ^ 3 * 4 ^ 3) = 144 * real.sqrt 3 :=
by
  sorry

end simplify_sqrt_product_l596_596777


namespace find_f_sum_of_roots_l596_596154

noncomputable def f (x : ℝ) : ℝ :=
if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_f_sum_of_roots (b c x1 x2 x3 x4 x5 : ℝ) :
  f x1 = 1 ∧
  f x2 = Real.log (12 - 2) ∧
  f x3 = Real.log ((10^(1 - b) + 2) - 2) ∧
  f x4 = Real.log (2 - (2 - 10^(1 - b))) ∧
  f x5 = Real.log (abs (-8 - 2)) ∧
  f x1 ^ 2 + b * f x1 + c = 0 ∧
  f x2 ^ 2 + b * f x2 + c = 0 ∧
  f x3 ^ 2 + b * f x3 + c = 0 ∧
  f x4 ^ 2 + b * f x4 + c = 0 ∧
  f x5 ^ 2 + b * f x5 + c = 0 ∧
  x1 + x2 + x3 + x4 + x5 = 10 :=
  f(10) = 3 * Real.log 2 := by sorry

end find_f_sum_of_roots_l596_596154


namespace length_of_segment_AB_l596_596707

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l596_596707


namespace vitya_catch_up_l596_596882

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596882


namespace max_pencils_to_buy_l596_596965

-- Definition of costs and budget
def pin_cost : ℕ := 3
def pen_cost : ℕ := 4
def pencil_cost : ℕ := 9
def total_budget : ℕ := 72

-- Minimum purchase required: one pin and one pen
def min_purchase : ℕ := pin_cost + pen_cost

-- Remaining budget after minimum purchase
def remaining_budget : ℕ := total_budget - min_purchase

-- Maximum number of pencils can be bought with the remaining budget
def max_pencils := remaining_budget / pencil_cost

-- Theorem stating the maximum number of pencils Alice can purchase
theorem max_pencils_to_buy : max_pencils = 7 :=
by
  -- Proof would go here
  sorry

end max_pencils_to_buy_l596_596965


namespace find_angle_YWZ_l596_596304

/-
Given:
1. ∠XYZ = 75°
2. ∠YXZ = 68°
3. ∠ YWZ is desired

Prove:
∠YWZ = 18.5°
-/
noncomputable def angle_XYZ := 75
noncomputable def angle_YXZ := 68
noncomputable def angle_YZX := 180 - 75 - 68

theorem find_angle_YWZ (XYZ : Type) (X Y Z W : XYZ)
  (h1 : ∠XYZ Y Z X = angle_XYZ)
  (h2 : ∠YXZ X Y Z = angle_YXZ)
  (h3 : ∠YZX X Z Y = angle_YZX)
  (h4 : bisects W (∠XYZ Y Z X))
  (h5 : bisects W (∠YZX X Z Y)) :
  ∠YWZ Y W Z = 18.5 :=
sorry

end find_angle_YWZ_l596_596304


namespace ceil_floor_diff_l596_596079

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596079


namespace initial_lizards_l596_596731

theorem initial_lizards (d c l n p : ℕ)
  (H1 : d = 30) 
  (H2 : c = 28)
  (H3 : n = 13)
  (H4 : p = 65)
  (H5 : 0.5 * d = 15)
  (H6 : 0.25 * c = 7)
  (H7 : 0.8 * l = 16)
  : l = 20 := 
by
  sorry

end initial_lizards_l596_596731


namespace smallest_base_not_sum_27_l596_596906

theorem smallest_base_not_sum_27 :
  ∃ b : ℕ, (b > 1) ∧ (let digits := (nat.digits b 1331) in digits.sum ≠ 27) ∧
  ∀ k : ℕ, (k > 1) ∧ (k < b) → (let digits := (nat.digits k 1331) in digits.sum = 27) :=
by
  sorry

end smallest_base_not_sum_27_l596_596906


namespace smallest_y_l596_596907

theorem smallest_y (y : ℕ) : (27^y > 3^24) ↔ (y ≥ 9) :=
sorry

end smallest_y_l596_596907


namespace P_X_gt_3_l596_596613

noncomputable section

open ProbabilityTheory

-- Define normal distribution
def normalDist (μ σ : ℝ) : Measure ℝ := Measure.normal μ σ

-- Define X as a random variable
axiom X : ℝ → ℝ

-- Given assumptions
axiom X_normal : ∀ t, ∫ x in Set.Iic (X t), x = real_density_normal 2 1
axiom P_X_gt_1 : P (λ x, X x > 1) = 0.8413

-- The proof goal
theorem P_X_gt_3 : P (λ x, X x > 3) = 0.1587 :=
  sorry

end P_X_gt_3_l596_596613


namespace maria_final_result_liam_final_result_aisha_final_result_aisha_has_largest_final_l596_596763

theorem maria_final_result (start : ℕ) : start = 15 → ((start - 2) * 3 + 5) = 44 :=
by
  intro h
  rw [h]
  calc
    (15 - 2) * 3 + 5 = 13 * 3 + 5   : by norm_num
                    ...  = 39 + 5    : by norm_num
                    ...  = 44        : by norm_num

theorem liam_final_result (start : ℕ) : start = 15 → ((start * 3 - 2) + 5) = 48 :=
by
  intro h
  rw [h]
  calc
    (15 * 3 - 2) + 5 = 45 - 2 + 5 : by norm_num
                    ...  = 43 + 5  : by norm_num
                    ...  = 48      : by norm_num

theorem aisha_final_result (start : ℕ) : start = 15 → (((start - 2) + 5) * 3) = 54 :=
by
  intro h
  rw [h]
  calc
    ((15 - 2) + 5) * 3 = (13 + 5) * 3 : by norm_num
                      ...  = 18 * 3    : by norm_num
                      ...  = 54        : by norm_num

theorem aisha_has_largest_final (start : ℕ) : start = 15 →
  let maria_result := (start - 2) * 3 + 5
  let liam_result := (start * 3 - 2) + 5
  let aisha_result := ((start - 2) + 5) * 3
  max (max maria_result liam_result) aisha_result = aisha_result :=
by
  intro h
  simp [h]
  calc
    max (max ((15 - 2) * 3 + 5) ((15 * 3 - 2) + 5)) (((15 - 2) + 5) * 3)
      = max (max 44 48) 54 : by simp [maria_final_result, liam_final_result, aisha_final_result]
  ... = max 48 54 : by norm_num
  ... = 54       : by norm_num

end maria_final_result_liam_final_result_aisha_final_result_aisha_has_largest_final_l596_596763


namespace maximize_value_l596_596392

def f (x : ℝ) : ℝ := -3 * x^2 - 8 * x + 18

theorem maximize_value : ∀ x : ℝ, f x ≤ f (-4/3) :=
by sorry

end maximize_value_l596_596392


namespace complete_the_square_l596_596159

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end complete_the_square_l596_596159


namespace ratio_of_a_to_c_l596_596193

theorem ratio_of_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3) 
  (h3 : d / b = 1 / 5) : a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l596_596193


namespace map_distance_to_actual_distance_l596_596800

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_inches : ℝ)
  (scale_miles : ℝ)
  (actual_distance : ℝ)
  (h_scale : scale_inches = 0.5)
  (h_scale_miles : scale_miles = 10)
  (h_map_distance : map_distance = 20) :
  actual_distance = 400 :=
by
  sorry

end map_distance_to_actual_distance_l596_596800


namespace vendor_sales_first_day_l596_596454

theorem vendor_sales_first_day (A S: ℝ) (h1: S = S / 100) 
  (h2: 0.20 * A * (1 - S / 100) = 0.42 * A - 0.50 * A * (0.80 * (1 - S / 100)))
  (h3: 0 < S) (h4: S < 100) : 
  S = 30 := 
by
  sorry

end vendor_sales_first_day_l596_596454


namespace weight_of_each_bag_is_7_l596_596445

-- Defining the conditions
def morning_bags : ℕ := 29
def afternoon_bags : ℕ := 17
def total_weight : ℕ := 322

-- Defining the question in terms of proving a specific weight per bag
def bags_sold := morning_bags + afternoon_bags
def weight_per_bag (w : ℕ) := total_weight = bags_sold * w

-- Proving the question == answer under the given conditions
theorem weight_of_each_bag_is_7 :
  ∃ w : ℕ, weight_per_bag w ∧ w = 7 :=
by
  sorry

end weight_of_each_bag_is_7_l596_596445


namespace find_exponent_l596_596642

theorem find_exponent
  (a b : ℝ)
  (h1 : 30^a = 2)
  (h2 : 30^b = 3) :
  6^((1 - a - b) / (2 * (1 - b))) = sqrt 5 :=
  sorry

end find_exponent_l596_596642


namespace distance_from_C_to_A_is_8_l596_596402

-- Define points A, B, and C as real numbers representing positions
def A : ℝ := 0  -- Starting point
def B : ℝ := A - 15  -- 15 meters west from A
def C : ℝ := B + 23  -- 23 meters east from B

-- Prove that the distance from point C to point A is 8 meters
theorem distance_from_C_to_A_is_8 : abs (C - A) = 8 :=
by
  sorry

end distance_from_C_to_A_is_8_l596_596402


namespace total_time_is_120_l596_596435

-- Definitions for the given conditions
def v_boat : ℝ := 9 -- speed of the boat in standing water (km/h)
def v_stream : ℝ := 6 -- speed of the stream (km/h)
def distance : ℝ := 300 -- distance between points A and B (km)

-- Downstream and upstream speeds calculation
def v_down : ℝ := v_boat + v_stream
def v_up : ℝ := v_boat - v_stream

-- Time calculations for downstream and upstream
def time_down : ℝ := distance / v_down
def time_up : ℝ := distance / v_up

-- Total time calculation
def total_time : ℝ := time_down + time_up

-- Theorem: Prove that the total time taken for the entire journey is 120 hours.
theorem total_time_is_120 : total_time = 120 := by
  -- The proof will be provided here
  sorry

end total_time_is_120_l596_596435


namespace total_painter_cost_l596_596424

-- Definitions for the conditions
def n : ℕ := 25
def a_east : ℕ := 5
def d_east : ℕ := 7
def a_west : ℕ := 6
def d_west : ℕ := 7
def cost_per_digit : ℕ := 1

-- Function to calculate the last term of an arithmetic sequence
def last_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Lists of house numbers on each side
def east_house_numbers : list ℕ :=
  list.range n |>.map (λ i, a_east + i * d_east)

def west_house_numbers : list ℕ :=
  list.range n |>.map (λ i, a_west + i * d_west)

-- Function to calculate the digits of a number
def num_digits (x : ℕ) : ℕ :=
  if x == 0 then 1 else nat.floor (real.log10 (x : ℝ)) + 1

-- Function to calculate the total cost of painting all house numbers
def total_cost (numbers : list ℕ) : ℕ :=
  numbers.foldl (λ acc x, acc + (num_digits x) * cost_per_digit) 0

-- Theorem statement (no proof required)
theorem total_painter_cost : total_cost east_house_numbers + total_cost west_house_numbers = 122 :=
by sorry

end total_painter_cost_l596_596424


namespace problem_solution_l596_596319

noncomputable def ratio_of_areas (A B C D : Point)
  (h_isosceles : distance A B = distance A C)
  (angle_BAC : angle A B C = 100)
  (angle_DBC : angle D B C = 30)
  (D_on_AC : lies_on_segment D A C) : ℚ :=
  let area_ADB := area_triangle A D B
  let area_CDB := area_triangle C D B
  in area_ADB / area_CDB

theorem problem_solution 
  (A B C D : Point)
  (h_isosceles : distance A B = distance A C)
  (angle_BAC : angle A B C = 100)
  (angle_DBC : angle D B C = 30)
  (D_on_AC : lies_on_segment D A C) :
  ratio_of_areas A B C D h_isosceles angle_BAC angle_DBC D_on_AC = 1 / 3 :=
sorry

end problem_solution_l596_596319


namespace solve_for_x_l596_596395

theorem solve_for_x (x : ℚ) : (3 * x / 7 - 2 = 12) → (x = 98 / 3) :=
by
  intro h
  sorry

end solve_for_x_l596_596395


namespace fabric_left_after_flags_l596_596983

theorem fabric_left_after_flags :
  let initial_fabric := 1000
  let square_flag_area := 4 * 4
  let wide_flag_area := 5 * 3
  let tall_flag_area := 3 * 5
  let number_of_square_flags := 16
  let number_of_wide_flags := 20
  let number_of_tall_flags := 10
  let fabric_used_for_square_flags := number_of_square_flags * square_flag_area
  let fabric_used_for_wide_flags := number_of_wide_flags * wide_flag_area
  let fabric_used_for_tall_flags := number_of_tall_flags * tall_flag_area
  let total_fabric_used := fabric_used_for_square_flags + fabric_used_for_wide_flags + fabric_used_for_tall_flags
  initial_fabric - total_fabric_used = 294 := by
  let initial_fabric := 1000
  let square_flag_area := 4 * 4
  let wide_flag_area := 5 * 3
  let tall_flag_area := 3 * 5
  let number_of_square_flags := 16
  let number_of_wide_flags := 20
  let number_of_tall_flags := 10
  let fabric_used_for_square_flags := number_of_square_flags * square_flag_area
  let fabric_used_for_wide_flags := number_of_wide_flags * wide_flag_area
  let fabric_used_for_tall_flags := number_of_tall_flags * tall_flag_area
  let total_fabric_used := fabric_used_for_square_flags + fabric_used_for_wide_flags + fabric_used_for_tall_flags
  have : total_fabric_used = 706, by sorry
  show initial_fabric - total_fabric_used = 294 from sorry

end fabric_left_after_flags_l596_596983


namespace card_X_le_4_l596_596414

variables {X : Type*} {k : ℕ} 
variable (F : finset (finset X))
variable (X1 X2 : finset X)
variable [decidable_eq X]

-- Assume F is a family of 3-subsets of set X
def is_family_of_3_subsets (F : finset (finset X)) : Prop :=
  ∀ A ∈ F, A.card = 3

-- Assume every two distinct elements of X are in exactly k elements of F
def pairs_in_exactly_k_elements (F : finset (finset X)) (k : ℕ) : Prop :=
  ∀ {a b : X}, a ≠ b → (finset.filter (λ A, {a, b} ⊆ A) F).card = k

-- Assume there exists a partition of F into sets X1 and X2 such that each element of F has non-empty intersection with both X1 and X2
def exists_partition_with_nonempty_intersections (F X1 X2 : finset X) : Prop :=
  ∃ Y1 Y2 : finset (finset X), F = Y1 ∪ Y2 ∧ Y1 ∩ Y2 = ∅ ∧ 
    ∀ {A}, A ∈ F → (A ∩ X1).nonempty ∧ (A ∩ X2).nonempty

-- The main theorem
theorem card_X_le_4 (hF : is_family_of_3_subsets F) 
  (hk : pairs_in_exactly_k_elements F k) 
  (hX : exists_partition_with_nonempty_intersections F X1 X2) : 
  (F.image finset.univ).card ≤ 4 :=
sorry

end card_X_le_4_l596_596414


namespace locus_P_eq_l596_596948

noncomputable def locus_of_P : Real → Real → Prop :=
  λ x y, 4 * x + y = (2 / 3) * sqrt (5 - (y - x) ^ 2) ∧ abs (y - x) < sqrt 5

theorem locus_P_eq :
  ∃ A B P : ℝ × ℝ,
  let line_eq := λ x, x + (A.2 - A.1),
      ellipse_eq := λ (x y : ℝ), x^2 + (y^2 / 4) = 1 in 
  line_eq A.1 = A.2 ∧ 
  line_eq B.1 = B.2 ∧
  ellipse_eq A.1 A.2 ∧ 
  ellipse_eq B.1 B.2 ∧ 
  ∃ t : ℝ, P = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2) ∧
  (t = 2 / 3) →
  locus_of_P P.1 P.2 :=
sorry

end locus_P_eq_l596_596948


namespace quadrilateral_is_parallelogram_l596_596317

variables {Point : Type} [AddGroup Point] [VectorSpace ℝ Point]

structure Triangle :=
(A B C : Point)

def midpoint (P Q : Point) : Point :=
(P + Q) / 2

structure Quadrilateral :=
(A B C D : Point)

def median (T : Triangle) : Point :=
  midpoint T.B T.C

def extend_median (T : Triangle) (M : Point) : Point :=
  M + (M - T.A)

theorem quadrilateral_is_parallelogram 
  (T : Triangle) 
  (M : Point) 
  (hM : M = midpoint T.B T.C)
  (D : Point)
  (hD : D = extend_median T M) :
  ∃ Q : Quadrilateral, Q.A = T.A ∧ Q.B = T.B ∧ Q.C = T.C ∧ Q.D = D ∧ 
    (midpoint Q.A Q.C = midpoint Q.B Q.D) :=
begin
  sorry,
end

end quadrilateral_is_parallelogram_l596_596317


namespace mr_kishore_savings_l596_596457

def total_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 2000 + 2500

def monthly_salary : ℝ := total_expenses / 0.9

def savings : ℝ := 0.1 * monthly_salary

theorem mr_kishore_savings : savings ≈ 2333.33 := by
  have h1 : total_expenses = 5000 + 1500 + 4500 + 2500 + 2000 + 2500 := rfl
  have h2 : savings = 0.1 * (total_expenses / 0.9) := rfl
  sorry

end mr_kishore_savings_l596_596457


namespace percentage_shaded_l596_596960

def area_rect (width height : ℝ) : ℝ := width * height

def overlap_area (side_length : ℝ) (width_rect : ℝ) (length_rect: ℝ) (length_total: ℝ) : ℝ :=
  (side_length - (length_total - length_rect)) * width_rect

theorem percentage_shaded (sqr_side length_rect width_rect total_length total_width : ℝ) (h1 : sqr_side = 12) (h2 : length_rect = 9) (h3 : width_rect = 12)
  (h4 : total_length = 18) (h5 : total_width = 12) :
  (overlap_area sqr_side width_rect length_rect total_length) / (area_rect total_width total_length) * 100 = 12.5 :=
by
  sorry

end percentage_shaded_l596_596960


namespace count_20_tuples_l596_596566

theorem count_20_tuples :
  (finset.univ.filter (λ (xy : (fin 10 → ℕ) × (fin 10 → ℕ)),
    (∀ i : fin 10, 1 ≤ xy.1 i ∧ xy.1 i ≤ 10 ∧ 1 ≤ xy.2 i ∧ xy.2 i ≤ 10) ∧
    (∀ i : fin 9, xy.1 i ≤ xy.1 i.succ) ∧
    (∀ i : fin 9, xy.1 i = xy.1 i.succ → xy.2 i ≤ xy.2 i.succ))).card =
  nat.choose 109 10 :=
sorry

end count_20_tuples_l596_596566


namespace green_caps_percentage_l596_596461

variable (total_caps : ℕ) (red_caps : ℕ)

def green_caps (total_caps red_caps: ℕ) : ℕ :=
  total_caps - red_caps

def percentage_of_green_caps (total_caps green_caps: ℕ) : ℕ :=
  (green_caps * 100) / total_caps

theorem green_caps_percentage :
  (total_caps = 125) →
  (red_caps = 50) →
  percentage_of_green_caps total_caps (green_caps total_caps red_caps) = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry  -- The proof is omitted 

end green_caps_percentage_l596_596461


namespace symmetric_line_eqn_l596_596164

noncomputable def circle1 : (ℝ → ℝ → Prop) :=
  λ x y, x^2 + y^2 + 2*x - 2*y + 1 = 0

noncomputable def circle2 : (ℝ → ℝ → Prop) :=
  λ x y, x^2 + y^2 - 4*x + 4*y + 7 = 0

theorem symmetric_line_eqn : 
  (∀ x y : ℝ, circle1 x y ↔ circle2 x y) →
  ∃ l : (ℝ → ℝ → Prop), (∀ x y : ℝ, l x y ↔ x - y - 1 = 0) :=
by
  sorry

end symmetric_line_eqn_l596_596164


namespace max_roads_city_condition_l596_596235

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596235


namespace parallel_lines_distance_l596_596610

theorem parallel_lines_distance
  (x y : ℝ)
  (line1 : x + 2 * y - 1 = 0)
  (line2 : 2 * x + m * y + 4 = 0)
  (parallel : line1.parallel line2)
  (line2_sim : x + 2 * y + 2 = 0) :
  distance_in_parallel_lines line1.line_eq line2.line_eq = 3 / sqrt 5 := 
sorry

end parallel_lines_distance_l596_596610


namespace Vitya_catchup_mom_in_5_l596_596890

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596890


namespace sum_of_roots_of_poly_eq_14_over_3_l596_596497

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l596_596497


namespace average_chemistry_math_l596_596961

variables {T : ℝ} {P : ℝ} {C : ℝ} {M : ℝ}

-- Conditions
def physics_scaled (T : ℝ) : ℝ := 0.30 * T
def chemistry_scaled (T : ℝ) : ℝ := 0.25 * T
def mathematics_scaled (T : ℝ) : ℝ := 0.35 * T
def total_marks_condition (T : ℝ) (P : ℝ) : Prop := T = P + 110

-- Proof problem
theorem average_chemistry_math (T : ℝ) (P : ℝ) (C : ℝ) (M : ℝ) 
  (h1 : P = physics_scaled T)
  (h2 : C = chemistry_scaled T)
  (h3 : M = mathematics_scaled T)
  (h4 : total_marks_condition T P) :
  (C + M) / 2 = 47.145 :=
begin
  sorry
end

end average_chemistry_math_l596_596961


namespace parallelepiped_tetrahedron_area_equality_l596_596926

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

noncomputable def area_square (u v : V) : ℝ :=
  ∥u × v∥^2

theorem parallelepiped_tetrahedron_area_equality 
  (a b c : V) : 
  area_square a b + area_square a c + area_square b c = 
  area_square a b + area_square b c + area_square c a + area_square (a + b) c :=
sorry

end parallelepiped_tetrahedron_area_equality_l596_596926


namespace student_exam_score_l596_596449

-- Define the conditions
def student_score_direct_proportion (S T : ℝ) (score_per_time hour : ℝ) :=
  score_per_time = S / T ∧ hour * score_per_time 

-- Function to cap the score at 100 points maximum
def cap_score (grade : ℝ) : ℝ :=
  if grade > 100 then 100 else grade

-- Theorem to prove the student's score after 5 hours, given the conditions
theorem student_exam_score
  (S1 : ℝ) (T1 : ℝ) (S_max : ℝ) (T2 : ℝ) :
  (S1 = 84) → (T1 = 2) → (S_max = 100) → (T2 = 5) →
  (student_score_direct_proportion S1 T1 (S1 / T1) T2 * (S1 / T1) ≤ S_max) →
  cap_score (T2 * (S1 / T1)) = 100 := by
  sorry

end student_exam_score_l596_596449


namespace circumcircle_area_of_triangle_l596_596711

-- Lean expression of the conditions for triangle ABC.
variables {A B C : ℝ}   -- Angles
variables {a b c : ℝ}  -- Sides opposite to angles A, B, C (respectively)
variable h_geom_seq : b^2 = a * c
variable h_a : a = 6
variable h_equation : (a + b) * sin C = (a - c) * (sin A + sin C)

-- Lean statement to prove the area of the circumcircle is 12π.
theorem circumcircle_area_of_triangle : 
  b^2 = a * c → a = 6 → (a + b) * sin C = (a - c) * (sin A + sin C) → 
  ∃ (R : ℝ), R = 2 * sqrt 3 ∧ (π * R^2 = 12 * π) :=
by sorry

end circumcircle_area_of_triangle_l596_596711


namespace max_value_of_N_l596_596194

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596194


namespace vitya_catch_up_time_l596_596865

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596865


namespace simplified_expression_value_substitute_x_value_l596_596050

theorem simplified_expression_value (x : ℝ) : 
  (x^8 + 16 * x^4 + 64) / (x^4 + 8) = x^4 + 8 := sorry

theorem substitute_x_value : 
  (3^8 + 16 * 3^4 + 64) / (3^4 + 8) = 89 := by
have h : (3^4 + 8) ≠ 0 := by norm_num
rw [←simplified_expression_value 3, div_eq_mul_inv, mul_inv_cancel h, one_mul]
norm_num

end simplified_expression_value_substitute_x_value_l596_596050


namespace abs_neg_2023_l596_596478

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l596_596478


namespace books_written_l596_596917

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l596_596917


namespace largest_number_l596_596787

theorem largest_number (HCF : ℕ) (factors : List ℕ):
  HCF = 154 ∧ factors = [19, 23, 37] → 
  ∃ largest, largest = 154 * 19 * 23 * 37 ∧ largest = 2493726 :=
by
  intro h
  -- creating variables to store the conditions 
  let hcf := 154
  let factors := [19, 23, 37]
  existsi 154 * 19 * 23 * 37
  split
  -- Proving the definition of largest number based on given conditions
  { rfl }
  -- Check if the resulting largest number equals 2493726
  { exact (19 * 154) * (23 * 37) == 2493726 } -- further steps would be calculated in the proof attempt
sorry

end largest_number_l596_596787


namespace smallest_sum_of_squares_l596_596333

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 217) : 
  x^2 + y^2 ≥ 505 :=
sorry

end smallest_sum_of_squares_l596_596333


namespace sandy_paid_twenty_dollars_l596_596017

-- Define variables and conditions
def price_cappuccino := 2
def price_iced_tea := 3
def price_cafe_latte := 1.5
def price_espresso := 1
def quantity_cappuccino := 3
def quantity_iced_tea := 2
def quantity_cafe_latte := 2
def quantity_espresso := 2
def change_received := 3

-- Calculate individual costs
def cost_cappuccino := price_cappuccino * quantity_cappuccino
def cost_iced_tea := price_iced_tea * quantity_iced_tea
def cost_cafe_latte := price_cafe_latte * quantity_cafe_latte
def cost_espresso := price_espresso * quantity_espresso

-- Calculate total cost
def total_cost := cost_cappuccino + cost_iced_tea + cost_cafe_latte + cost_espresso

-- Calculate the amount paid
def amount_paid := total_cost + change_received

-- Prove the amount Sandy paid with is $20
theorem sandy_paid_twenty_dollars : amount_paid = 20 := sorry

end sandy_paid_twenty_dollars_l596_596017


namespace range_of_a1_l596_596844

theorem range_of_a1 (
    a1 : ℝ
) (h_prob : (∀ (a3 : ℝ), (a3 = 2 * a1 - 6 ∨ a3 = a1 / 2 + 6) → a3 > a1)
): a1 ∈ Iic 6 ∨ a1 ∈ Ici 12 ↔ (∃ p, p = 3/4) :=
sorry

end range_of_a1_l596_596844


namespace max_possible_value_l596_596256

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596256


namespace circle_radius_5_l596_596573

theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10 * x + y^2 + 2 * y + c = 0) → 
  (∀ x y : ℝ, (x + 5)^2 + (y + 1)^2 = 25) → 
  c = 51 :=
sorry

end circle_radius_5_l596_596573


namespace multiple_of_a_l596_596321

theorem multiple_of_a's_share (A B : ℝ) (x : ℝ) (h₁ : A + B + 260 = 585) (h₂ : x * A = 780) (h₃ : 6 * B = 780) : x = 4 :=
sorry

end multiple_of_a_l596_596321


namespace sum_of_roots_l596_596503

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l596_596503


namespace simplify_and_evaluate_expr_l596_596778

noncomputable def problem_expr (x : ℝ) :=
  (4 - x) / (x - 2) / (x + 2 - 12 / (x - 2))

def x_value : ℝ := Real.sqrt 3 - 4

theorem simplify_and_evaluate_expr :
  problem_expr x_value = - (Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expr_l596_596778


namespace min_disks_to_have_twelve_same_label_l596_596734

theorem min_disks_to_have_twelve_same_label :
    ∀ (n : ℕ), (n = 60) →
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n → ∃! (label : ℕ), label = i ∧ count_disks label = i)) →
    ∀ (draws : ℕ), (∃ (k : ℕ), draws = k ∧ draws ≥ 606) →
    ∃ (label : ℕ), (1 ≤ label ∧ label ≤ n ∧ at_least_twelve_disks_same_label draws label) :=
by
  sorry

end min_disks_to_have_twelve_same_label_l596_596734


namespace max_possible_cities_traversed_l596_596203

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596203


namespace minimum_value_sum_l596_596834

theorem minimum_value_sum {z : ℕ → ℂ} (h1 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → 
  |z k * complex.I ^ k + z (k+1) * complex.I ^ (k+1)| = 
  |z (k+1) * complex.I ^ k + z k * complex.I ^ (k+1)|)
  (h2 : |z 1| = 9)
  (h3 : |z 2| = 29)
  (h4 : ∀ n : ℕ, 3 ≤ n ∧ n ≤ 10 → |z n| = |z (n-1) + z (n-2)|) :
  ∑ n in (finset.range 10).map (finset.range.succ 1), complex.abs (z n) = 183 :=
begin
  sorry
end

end minimum_value_sum_l596_596834


namespace max_roads_city_condition_l596_596239

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596239


namespace eliza_tom_difference_l596_596994

theorem eliza_tom_difference (q : ℕ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := (7 * q + 3) - (2 * q + 8)
  let nickel_value := 5
  let groups_of_5 := quarter_difference / 5
  let difference_in_cents := nickel_value * groups_of_5
  difference_in_cents = 5 * (q - 1) := by
  sorry

end eliza_tom_difference_l596_596994


namespace binom_n_n_minus_2_l596_596896

theorem binom_n_n_minus_2 (n : ℕ) (h : n > 0) : nat.choose n (n-2) = n * (n-1) / 2 :=
by sorry

end binom_n_n_minus_2_l596_596896


namespace problem_statement_l596_596145

theorem problem_statement :
  ∀ {A B : ℝ} (t₁ t₂ : ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ), 
  (P = (2, 0)) →
  (A = t₁) →
  (B = t₂) →
  (t₁ + t₂ = 15 / 8) →
  (t₁ * t₂ = -25 / 4) →
  (M = ((P.1 + (t₁ + t₂) / 2) / 2, (P.2 + (t₁ * t₂) / 2) / 2)) →

  -- Parametric equation of line l
  (∀ t : ℝ, (2 + 3 / 5 * t, 4 / 5 * t)) →

  -- Length of segment PM
  |PM.1 - M.1| = 15 / 16 →
  
  -- Length of segment AB
  |t₁ - t₂| = 5 * sqrt 73 / 8 :=
sorry

end problem_statement_l596_596145


namespace min_elements_with_conditions_l596_596667

theorem min_elements_with_conditions (s : Finset ℝ) 
  (h_median : (s.sort (≤)).median = 3) 
  (h_mean : s.mean = 5)
  (h_mode : s.mode = 6) : 
  s.card >= 6 :=
sorry

end min_elements_with_conditions_l596_596667


namespace incircle_tangent_distance_l596_596275

theorem incircle_tangent_distance (a b c : ℝ) (M : ℝ) (BM : ℝ) (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : BM = y1 + z1)
  (h2 : BM = y2 + z2)
  (h3 : x1 + y1 = x2 + y2)
  (h4 : x1 + z1 = c)
  (h5 : x2 + z2 = a) :
  |y1 - y2| = |(a - c) / 2| := by 
  sorry

end incircle_tangent_distance_l596_596275


namespace maximum_possible_value_of_N_l596_596216

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596216


namespace magnitude_P1P2_l596_596751

variables (e₁ e₂ : ℝ^3)
variables (OP₁ OP₂ : ℝ^3)
variables (x y : ℝ)

-- Conditions
def angle_60 : Prop := inner e₁ e₂ = (1 : ℝ)/2
def unit_vectors : Prop := (∥e₁∥ = 1) ∧ (∥e₂∥ = 1)
def coordinates_P1 : Prop := OP₁ = 2 * e₁ + 3 * e₂
def coordinates_P2 : Prop := OP₂ = 3 * e₁ + 2 * e₂

-- The problem statement
theorem magnitude_P1P2 :
  angle_60 e₁ e₂ →
  unit_vectors e₁ e₂ →
  coordinates_P1 e₁ e₂ OP₁ →
  coordinates_P2 e₁ e₂ OP₂ →
  ∥OP₂ - OP₁∥ = 1 :=
by
  intros,
  sorry

end magnitude_P1P2_l596_596751


namespace calculate_savings_l596_596004

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l596_596004


namespace parametric_curve_to_ordinary_l596_596358

noncomputable theory

def parametric_to_cartesian (t : ℝ) (h : t ≠ 0) : ℝ × ℝ :=
  let x := 1 - 1/t in
  let y := 1 - t^2 in
  (x, y)

theorem parametric_curve_to_ordinary :
  (∀ (t : ℝ) (ht : t ≠ 0), let xy := parametric_to_cartesian t ht in xy.2 = (xy.1 * (xy.1 - 2)) / (1 - xy.1)^2) :=
by
  assume t ht
  have h1 : t ≠ 0 := ht
  have hx : 1 - 1/t ≠ 1 := by
    intro h
    have h2 := congr_arg (fun z => z * t) h
    simp [one_mul, sub_eq_zero] at h2
    contradiction
  sorry

end parametric_curve_to_ordinary_l596_596358


namespace triangle_angles_proof_l596_596263

-- Definitions of given conditions:
variables (A B C D E I : Type)
variables (angle_BDE : ℝ) (angle_CED : ℝ)
variables (alpha beta gamma : ℝ)

-- Given data
def triangle_ABC := Type

-- Given angles for the bisectors
def angle_BDE_value : Prop := angle_BDE = 24
def angle_CED_value : Prop := angle_CED = 18

-- To prove: angles of the triangle
def angles_sum_to_180 : Prop := alpha + beta + gamma = 180
def angle_A_value : Prop := alpha = 96
def angle_B_value : Prop := beta = 42
def angle_C_value : Prop := gamma = 42

-- Proof problem statement in Lean 4
theorem triangle_angles_proof
  (h1 : angle_BDE_value)
  (h2 : angle_CED_value)
  (h3 : angles_sum_to_180)
  : angle_A_value ∧ angle_B_value ∧ angle_C_value := by
  sorry

end triangle_angles_proof_l596_596263


namespace coeff_of_x5_in_expansion_l596_596794

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the polynomial expansion
noncomputable def poly (x : ℝ) : ℝ := (1 + x^3) * ((1 - 2 * x)^6)

-- Coefficient extraction function specifically for x^5
noncomputable def coeff_of_x5 (x : ℝ) : ℝ :=
  (-1)^5 * (binom 6 5) * (2^5) + (1 * (binom 6 2) * (2^2))

-- Main theorem statement
theorem coeff_of_x5_in_expansion : coeff_of_x5 (-132) = -132 :=
  by
    -- The actual proof steps go here
    sorry

end coeff_of_x5_in_expansion_l596_596794


namespace problem_statement_l596_596260

noncomputable def distance_from_line_to_point (a b : ℝ) : ℝ :=
  abs (1 / 2) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem problem_statement (a b : ℝ) (h1 : a = (1 - 2 * b) / 2) (h2 : b = 1 / 2 - a) :
  distance_from_line_to_point a b ≤ Real.sqrt 2 := 
sorry

end problem_statement_l596_596260


namespace probability_divisible_by_5_l596_596821

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l596_596821


namespace problem1_problem2_l596_596415

-- Problem 1: Prove that |-\sqrt{3}| + 2 * cos(45°) - tan(60°) = √2
theorem problem1 : abs (-sqrt 3) + 2 * real.cos (real.pi / 4) - real.tan (real.pi / 3) = sqrt 2 := 
sorry

-- Problem 2: Prove that the solutions to (x-7)^2 = 3(7-x) are x=7 and x=4
theorem problem2 (x : ℝ) : (x - 7)^2 = 3 * (7 - x) ↔ x = 7 ∨ x = 4 :=
sorry

end problem1_problem2_l596_596415


namespace find_k_of_direct_proportion_l596_596334

theorem find_k_of_direct_proportion (k : ℝ) (h : (1, 3) ∈ set_of (λ p : ℝ × ℝ, p.snd = k * p.fst)) : k = 3 := 
by
  sorry

end find_k_of_direct_proportion_l596_596334


namespace solution_to_inequality_l596_596115

theorem solution_to_inequality (x m : ℝ) (h : (∀ x, (x + 2) / 2 ≥ (2 * x + m) / 3 + 1 → x ≤ 8)) :
  2 ^ m = 1 / 16 :=
sorry

end solution_to_inequality_l596_596115


namespace moles_H2_produced_l596_596998

-- Given the balanced reaction: Zn + H2SO4 → ZnSO4 + H2

-- Define the number of moles of H2SO4 and Zn as constants
constant H2SO4_moles : ℕ := 3
constant Zn_moles : ℕ := 3

-- Define the function that calculates moles of H2 formed based on the reaction Zn + H2SO4 → ZnSO4 + H2
def moles_of_H2_formed (H2SO4_moles Zn_moles : ℕ) : ℕ :=
  if H2SO4_moles = Zn_moles then H2SO4_moles else 0

-- State the theorem to prove
theorem moles_H2_produced : moles_of_H2_formed H2SO4_moles Zn_moles = 3 :=
sorry

end moles_H2_produced_l596_596998


namespace original_is_891134_l596_596552

-- Definition of the encryption transformation method as outlined in the problem
def encrypt_digit (n : ℕ) : ℕ :=
  let d := (n * 7) % 10
  in 10 - d

-- Hypothesis that 473392 results from encrypting the original number
axiom encrypted_original : ∃ (original : ℕ), 
  (digit_at original 0 |> encrypt_digit) = 2 ∧
  (digit_at original 1 |> encrypt_digit) = 9 ∧
  (digit_at original 2 |> encrypt_digit) = 1 ∧
  (digit_at original 3 |> encrypt_digit) = 9 ∧
  (digit_at original 4 |> encrypt_digit) = 3 ∧
  (digit_at original 5 |> encrypt_digit) = 4 ∧
  -- converted back to the number 473392
  original = 891134

theorem original_is_891134 : ∃ (original : ℕ) , encrypted_original →
  original = 891134 :=
by
  sorry

end original_is_891134_l596_596552


namespace solution_valid_l596_596390

noncomputable def x₁ : ℝ := -671.56
noncomputable def x₂ : ℝ := 2015.56

theorem solution_valid (x : ℝ) (h : x = x₁ ∨ x = x₂) : (2016 + x)^2 = 4 * x^2 :=
by {
    rcases h with rfl | rfl,
    -- Both cases for x = x₁ and x = x₂ need to be proved
    sorry -- For x = x₁,
    sorry -- For x = x₂
}

end solution_valid_l596_596390


namespace abc_inequality_l596_596292

theorem abc_inequality (x y z : ℝ) (a b c : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a = (x * (y - z) ^ 2) ^ 2) (h2 : b = (y * (z - x) ^ 2) ^ 2) (h3 : c = (z * (x - y) ^ 2) ^ 2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) :=
by {
  sorry
}

end abc_inequality_l596_596292


namespace probability_one_basket_made_l596_596660

noncomputable def estimate_basket_probability (sets : List (List ℕ)) : ℚ :=
  let successful := sets.count (λ s => 
    s.count (λ x => x ≤ 3) = 1
  )
  successful / sets.length

theorem probability_one_basket_made :
  let sets := [[9,7,7], [8,6,4], [1,9,1], [9,2,5], [2,7,1], [9,3,2], [8,1,2], 
               [4,5,8], [5,6,9], [6,8,3], [4,3,1], [2,5,7], [3,9,4], [0,2,7], 
               [5,5,6], [4,8,8], [7,3,0], [1,1,3], [5,3,7], [9,0,8]] in
  estimate_basket_probability sets = 0.3 :=
by
  sorry

end probability_one_basket_made_l596_596660


namespace func_zero_l596_596546

def f (x : ℝ) : ℝ := sorry

axiom f_condition1 : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * (f x) * (f y)
axiom f_condition2 : filter.tendsto f filter.at_top (𝓝 0)

theorem func_zero (x : ℝ) : f x = 0 := sorry

end func_zero_l596_596546


namespace complement_20_18_is_69_7_l596_596643

theorem complement_20_18_is_69_7 :
  let alpha := 20 + 18 / 60
  in (90 - alpha = 69 + 7 / 10) :=
by
  sorry

end complement_20_18_is_69_7_l596_596643


namespace find_m_l596_596817

variable {a : Nat → ℝ}
variable {m : ℝ}

-- Define Sn as the sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℝ := 3^(n-2) + m

-- State the theorem to find the value of m
theorem find_m (n : ℕ) (Sn_def : ∀ n, Sn n = 3^(n-2) + m) : m = -1/9 :=
  sorry

end find_m_l596_596817


namespace no_ordered_triples_l596_596999

open Real

theorem no_ordered_triples (x y z : ℝ) (h1 : x + y = 4) (h2 : xy - 9 * z^2 = -5) : 
  (false : Prop) :=
by
  have h3 : xy = -5 + 9 * z^2 := by
    rw [mul_sub, mul_neg, neg_sub]
    sorry
  sorry

end no_ordered_triples_l596_596999


namespace molecular_weight_one_mole_of_AlPO4_l596_596903

theorem molecular_weight_one_mole_of_AlPO4
  (molecular_weight_4_moles : ℝ)
  (h : molecular_weight_4_moles = 488) :
  molecular_weight_4_moles / 4 = 122 :=
by
  sorry

end molecular_weight_one_mole_of_AlPO4_l596_596903


namespace total_packs_of_groceries_l596_596761

-- Definitions for the conditions
def packs_of_cookies : ℕ := 2
def packs_of_cake : ℕ := 12

-- Theorem stating the total packs of groceries
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake = 14 :=
by sorry

end total_packs_of_groceries_l596_596761


namespace Vitya_catchup_mom_in_5_l596_596891

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596891


namespace ceil_floor_difference_l596_596073

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596073


namespace minimum_dot_product_l596_596608

-- Definitions of points A and B
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 0)

-- Definition of condition that P lies on the line x - y + 1 = 0
def onLineP (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Definition of dot product between vectors PA and PB
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Lean 4 theorem statement
theorem minimum_dot_product (P : ℝ × ℝ) (hP : onLineP P) : 
  dotProduct P pointA pointB = 0 := 
sorry

end minimum_dot_product_l596_596608


namespace area_of_triangle_correct_l596_596560

def area_of_triangle : ℝ :=
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  (det / 2)

theorem area_of_triangle_correct : area_of_triangle = 15 := by
  let A := (4, -3)
  let B := (-1, 2)
  let C := (2, -7)
  let v := (2 : ℝ, 4 : ℝ)
  let w := (-3 : ℝ, 9 : ℝ)
  let det := (2 * 9 + 3 * 4 : ℝ)
  have h : area_of_triangle = (det / 2) := rfl
  rw [h]
  norm_num
  sorry

end area_of_triangle_correct_l596_596560


namespace max_possible_N_in_cities_l596_596249

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596249


namespace num_candidates_l596_596448

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l596_596448


namespace abs_neg_2023_l596_596483

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596483


namespace combine_like_terms_l596_596399

theorem combine_like_terms (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := 
  sorry

end combine_like_terms_l596_596399


namespace tan_arccot_l596_596023

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596023


namespace largest_constant_C_l596_596102

theorem largest_constant_C (C : ℝ) : C = 2 / Real.sqrt 3 ↔ ∀ (x y z : ℝ), x^2 + y^2 + 2 * z^2 + 1 ≥ C * (x + y + z) :=
by
  sorry

end largest_constant_C_l596_596102


namespace books_written_l596_596918

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l596_596918


namespace arman_two_weeks_earnings_l596_596735

theorem arman_two_weeks_earnings :
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  total_earnings = 770 := 
by
  -- Definitions based on conditions
  let hourly_rate := 10
  let last_week_hours := 35
  let this_week_hours := 40
  let increase := 0.5
  let first_week_earnings := last_week_hours * hourly_rate
  let new_hourly_rate := hourly_rate + increase
  let second_week_earnings := this_week_hours * new_hourly_rate
  let total_earnings := first_week_earnings + second_week_earnings
  sorry

end arman_two_weeks_earnings_l596_596735


namespace find_deepaks_age_l596_596362

variable (R D : ℕ)

theorem find_deepaks_age
  (h1 : R / D = 4 / 3)
  (h2 : R + 2 = 26) :
  D = 18 := by
  sorry

end find_deepaks_age_l596_596362


namespace quadratic_inequality_solution_set_l596_596816

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x - 14 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 7} :=
by
  -- proof to be filled here
  sorry

end quadratic_inequality_solution_set_l596_596816


namespace triangle_inequality_condition_l596_596442

theorem triangle_inequality_condition (a b : ℝ) (h : a + b = 1) (ha : a ≥ 0) (hb : b ≥ 0) :
    a + b > 1 → a + 1 > b ∧ b + 1 > a := by
  sorry

end triangle_inequality_condition_l596_596442


namespace task_D_is_suitable_l596_596915

-- Definitions of the tasks
def task_A := "Investigating the age distribution of your classmates"
def task_B := "Understanding the ratio of male to female students in the eighth grade of your school"
def task_C := "Testing the urine samples of athletes who won championships at the Olympics"
def task_D := "Investigating the sleeping conditions of middle school students in Lishui City"

-- Definition of suitable_for_sampling_survey condition
def suitable_for_sampling_survey (task : String) : Prop :=
  task = task_D

-- Theorem statement
theorem task_D_is_suitable : suitable_for_sampling_survey task_D := by
  -- the proof is omitted
  sorry

end task_D_is_suitable_l596_596915


namespace no_carry_pairs_count_l596_596571

theorem no_carry_pairs_count : 
  let valid_pair (n : ℕ) := (1000 ≤ n ∧ n < 2000) ∧ 
                             (n % 10 < 9) ∧ -- Units place
                             ((n / 10) % 10 < 9) ∧ -- Tens place
                             ((n / 100) % 10 < 9) ∧ -- Hundreds place
                             (n / 1000 = 1) -- Thousands place
  in (Finset.filter valid_pair (Finset.range 1000 2000)).card = 729 :=
by
  sorry

end no_carry_pairs_count_l596_596571


namespace tangent_line_eqn_l596_596586

theorem tangent_line_eqn (r x0 y0 : ℝ) (h : x0^2 + y0^2 = r^2) : 
  ∃ a b c : ℝ, a = x0 ∧ b = y0 ∧ c = r^2 ∧ (a*x + b*y = c) :=
sorry

end tangent_line_eqn_l596_596586


namespace units_place_digit_expression_l596_596107

theorem units_place_digit_expression :
  let units_place_3_exp (n : ℕ) := [3, 9, 7, 1]; 
  let units_place_7_exp (n : ℕ) := [7, 9, 3, 1]; 
  let units_place_5_exp (n : ℕ) := [5]; 
  (units_place_3_exp 34 % 10) * (units_place_7_exp 21 % 10) + (units_place_5_exp 17 % 10) % 10 = 8 :=
by
  sorry

end units_place_digit_expression_l596_596107


namespace tens_digit_of_factorial_sum_l596_596106

theorem tens_digit_of_factorial_sum : 
  let sum := (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10! + ... + 100!) in  
  (sum / 10) % 10 = 0 :=
by sorry

end tens_digit_of_factorial_sum_l596_596106


namespace sum_of_roots_l596_596513

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l596_596513


namespace cube_volume_ratio_l596_596639

theorem cube_volume_ratio
  (s1_m : ℝ) (s2_cm : ℝ)
  (conversion : s2_cm = 100) (m_to_cm: 100 = 1*0.01)
  (side1 : s1_m = 2) (side2 : s2_cm = 100) :
  let V1 := s1_m^3 in
  let s2_m := s2_cm * 0.01 in
  let V2 := s2_m^3 in
  V1 / V2 = 8 :=
by
  sorry

end cube_volume_ratio_l596_596639


namespace total_travel_ways_l596_596117

-- Define the number of car departures
def car_departures : ℕ := 3

-- Define the number of train departures
def train_departures : ℕ := 4

-- Define the number of ship departures
def ship_departures : ℕ := 2

-- The total number of ways to travel from location A to location B
def total_ways : ℕ := car_departures + train_departures + ship_departures

-- The theorem stating the total number of ways to travel given the conditions
theorem total_travel_ways :
  total_ways = 9 :=
by
  -- Proof goes here
  sorry

end total_travel_ways_l596_596117


namespace Vitya_catchup_mom_in_5_l596_596888

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596888


namespace fabric_left_after_flags_l596_596982

theorem fabric_left_after_flags :
  let initial_fabric := 1000
  let square_flag_area := 4 * 4
  let wide_flag_area := 5 * 3
  let tall_flag_area := 3 * 5
  let number_of_square_flags := 16
  let number_of_wide_flags := 20
  let number_of_tall_flags := 10
  let fabric_used_for_square_flags := number_of_square_flags * square_flag_area
  let fabric_used_for_wide_flags := number_of_wide_flags * wide_flag_area
  let fabric_used_for_tall_flags := number_of_tall_flags * tall_flag_area
  let total_fabric_used := fabric_used_for_square_flags + fabric_used_for_wide_flags + fabric_used_for_tall_flags
  initial_fabric - total_fabric_used = 294 := by
  let initial_fabric := 1000
  let square_flag_area := 4 * 4
  let wide_flag_area := 5 * 3
  let tall_flag_area := 3 * 5
  let number_of_square_flags := 16
  let number_of_wide_flags := 20
  let number_of_tall_flags := 10
  let fabric_used_for_square_flags := number_of_square_flags * square_flag_area
  let fabric_used_for_wide_flags := number_of_wide_flags * wide_flag_area
  let fabric_used_for_tall_flags := number_of_tall_flags * tall_flag_area
  let total_fabric_used := fabric_used_for_square_flags + fabric_used_for_wide_flags + fabric_used_for_tall_flags
  have : total_fabric_used = 706, by sorry
  show initial_fabric - total_fabric_used = 294 from sorry

end fabric_left_after_flags_l596_596982


namespace max_possible_cities_traversed_l596_596206

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596206


namespace abs_neg_2023_l596_596480

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l596_596480


namespace find_x_such_that_ceil_mul_x_eq_168_l596_596555

theorem find_x_such_that_ceil_mul_x_eq_168 (x : ℝ) (h_pos : x > 0)
  (h_eq : ⌈x⌉ * x = 168) (h_ceil: ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) :
  x = 168 / 13 :=
by
  sorry

end find_x_such_that_ceil_mul_x_eq_168_l596_596555


namespace final_elements_not_equal_l596_596842

open Nat List

-- Define a sequence in arithmetic progression
def arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- First list sequence
def first_list : List ℕ := List.map (arith_seq 1 5) (List.range 10).map (λ x => x + 1)

-- Second list sequence
def second_list : List ℕ := List.map (arith_seq 4 5) (List.range 10).map (λ x => x + 1)

-- Function to perform the given operation
def operation (l1 l2 : List ℕ) (x y : ℕ) (h : x ∈ l1 ∧ y ∈ l1) : (List ℕ × List ℕ) :=
  ((l1.erase x).erase y, (l2 ++ [ (x + y) / 3]))

-- Proof Statement
theorem final_elements_not_equal :
  ∀ (l1 l2 : List ℕ), l1 = first_list → l2 = second_list →
  ∃ l1' l2', (iter.filter l1).length = 1 ∧ (iter.filter l2).length = 1 → 
  (List.head l1' ≠ List.head l2') := 
  sorry

end final_elements_not_equal_l596_596842


namespace intersection_M_N_l596_596162

open Set

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} :=
  sorry

end intersection_M_N_l596_596162


namespace sector_radius_l596_596328

-- Definition for area of the sector
def sector_area (l r : ℝ) : ℝ := (l * r) / 2

-- Problem statement in Lean 4
theorem sector_radius : ∀ (l a : ℝ), l = 3.5 ∧ a = 7 → ∃ r : ℝ, sector_area l r = a ∧ r = 4 :=
by
  intros l a h
  have : l = 3.5 := h.1
  have : a = 7 := h.2
  use (4 : ℝ)
  split
  {
    unfold sector_area
    calc
      (3.5 * 4) / 2 = 14 / 2 : by ring
      ... = 7 : by norm_num
  }
  { norm_num }

end sector_radius_l596_596328


namespace heptagon_largest_angle_l596_596354

theorem heptagon_largest_angle (angles : Fin 7 → ℝ) (h_convex : ∀ i : Fin 7, 0 ≤ angles i ∧ angles i < 180) 
  (h_consecutive : ∃ y : ℝ, angles 0 = y - 6 ∧ angles 1 = y - 4 ∧ angles 2 = y - 2 ∧ angles 3 = y ∧
                                 angles 4 = y + 2 ∧ angles 5 = y + 4 ∧ angles 6 = y + 6) 
  (h_sum : ∑ i, angles i = 900) : 
  ∃ y : ℝ, angles 6 = 128.57142857142858 + 6 :=
begin
  sorry
end

end heptagon_largest_angle_l596_596354


namespace maximum_N_value_l596_596230

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596230


namespace B_is_1_and_2_number_of_sets_A_l596_596600

open Set

variable (A B : Set ℝ)
variable (f : ℝ → ℝ)

-- Define the function
noncomputable def f (x : ℝ) : ℝ := abs x + 1

-- Condition: A = {-1, 0, 1}
def cond_A : A = ({-1, 0, 1} : Set ℝ) := rfl

-- Condition: B has exactly 2 elements
def cond_B_two_elements : B.Card = 2 := sorry

-- Prove that if A = {-1, 0, 1} and B has 2 elements, then B = {1, 2}
theorem B_is_1_and_2 (hA : A = {-1, 0, 1}) (hb : B.Card = 2) : B = {1, 2} :=
  sorry

-- Prove number of sets A that map to {1, 2} under the function f
theorem number_of_sets_A :
  let B := {1, 2}
  in (card (filter (λ (s : Set ℝ), image (f) s = B) (powerset {-1, 0, 1}))) = 7 :=
  sorry

end B_is_1_and_2_number_of_sets_A_l596_596600


namespace missing_digit_divisibility_l596_596338

theorem missing_digit_divisibility (d : ℕ) (h_d : d ∈ {0, 3, 6, 9}) : (2460 + d * 10 + 9) % 3 = 0 :=
by
  sorry

end missing_digit_divisibility_l596_596338


namespace eric_age_l596_596380

theorem eric_age (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 :=
by
  sorry

end eric_age_l596_596380


namespace find_logarithmic_solutions_log10_l596_596568

theorem find_logarithmic_solutions_log10 (x : ℝ) :
  log 10 (x^2 - 20 * x) = 3 ↔ x = 10 + real.sqrt 1100 ∨ x = 10 - real.sqrt 1100 :=
by
  sorry

end find_logarithmic_solutions_log10_l596_596568


namespace geometric_sequence_sum_l596_596268

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : a 1 + a 3 = 8)
  (h2 : a 5 + a 7 = 4)
  (geometric_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 9 + a 11 + a 13 + a 15 = 3 :=
by
  sorry

end geometric_sequence_sum_l596_596268


namespace sum_of_digits_of_product_l596_596105

theorem sum_of_digits_of_product :
  let nines := list.repeat 9 94
  let fours := list.repeat 4 94
  let product := -- this computes the decimal digits of the product of the two numbers
    (nines.foldl (λ acc d, 10 * acc + d) 0) * (fours.foldl (λ acc d, 10 * acc + d) 0)
  let sum_of_digits := product.digits.sum
  sum_of_digits = 846 := sorry

end sum_of_digits_of_product_l596_596105


namespace ceil_floor_difference_l596_596071

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596071


namespace find_x_l596_596942

noncomputable def area_of_figure (x : ℝ) : ℝ :=
  let A_rectangle := 3 * x * 2 * x
  let A_square1 := x ^ 2
  let A_square2 := (4 * x) ^ 2
  let A_triangle := (3 * x * 2 * x) / 2
  A_rectangle + A_square1 + A_square2 + A_triangle

theorem find_x (x : ℝ) : area_of_figure x = 1250 → x = 6.93 :=
  sorry

end find_x_l596_596942


namespace negation_proposition_l596_596401

-- Define the proposition as a Lean function
def quadratic_non_negative (x : ℝ) : Prop := x^2 - 2*x + 1 ≥ 0

-- State the theorem that we need to prove
theorem negation_proposition : ∀ x : ℝ, quadratic_non_negative x :=
by 
  sorry

end negation_proposition_l596_596401


namespace prove_relationship_l596_596632

noncomputable def f (x : ℝ) : ℝ := 2^x + x
noncomputable def g (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := Real.log x / Real.log 2 + x

theorem prove_relationship (a b c : ℝ)
  (h1 : ∃ a, f a = 0)
  (h2 : ∃ b, g b = 0)
  (h3 : ∃ c, h c = 0)
  (h4 : a ∈ Ioo (-1 : ℝ) 0)
  (h5 : b = 2)
  (h6 : c ∈ Ioo (1/2 : ℝ) 1)
  (h7 : StrictMono f)
  (h8 : StrictMono g)
  (h9 : StrictMono h)
  : a < c ∧ c < b := sorry

end prove_relationship_l596_596632


namespace sum_of_areas_of_circles_l596_596383

theorem sum_of_areas_of_circles (n : ℕ) (α r1 : ℝ) :
  let q := (1 + Real.sin (α / 2)) / (1 - Real.sin (α / 2))
  let geometric_sum := (q ^ (2 * n) - 1) / (q ^ 2 - 1)
  in
  ∑ i in Finset.range n, (r1 * q ^ i)^2 * Real.pi = r1^2 * Real.pi * geometric_sum :=
by
  sorry

end sum_of_areas_of_circles_l596_596383


namespace part_a_part_b_part_b_parallel_l596_596285

noncomputable def point_distance_preserving (f : ℝ × ℝ → ℝ × ℝ) : Prop :=
∀ A B : ℝ × ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = (f A).1 - (f B).1)^2 + ((f A).2 - (f B).2)^2

def on_line (A B X : ℝ × ℝ) : Prop :=
∃ t : ℝ, X = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)

theorem part_a (f : ℝ × ℝ → ℝ × ℝ) (h_preserve : point_distance_preserving f)
  (C D X : ℝ × ℝ) (hX : on_line C D X) : on_line (f C) (f D) (f X) :=
sorry

theorem part_b (f : ℝ × ℝ → ℝ × ℝ) (h_preserve : point_distance_preserving f)
  (C D E F : ℝ × ℝ) (α : ℝ)
  (h_angle : ∃ I : ℝ × ℝ, on_line C D I ∧ on_line E F I ∧ some_angle_measure I = α) :
  ∃ J : ℝ × ℝ, on_line (f C) (f D) J ∧ on_line (f E) (f F) J ∧ some_angle_measure J = α :=
sorry

theorem part_b_parallel (f : ℝ × ℝ → ℝ × ℝ) (h_preserve : point_distance_preserving f)
  (C D E F : ℝ × ℝ) 
  (h_parallel : ∀ X : ℝ × ℝ, ¬ (on_line C D X ∧ on_line E F X)) :
  ∀ Z : ℝ × ℝ, ¬ (on_line (f C) (f D) Z ∧ on_line (f E) (f F) Z) :=
sorry

end part_a_part_b_part_b_parallel_l596_596285


namespace find_roots_l596_596563

noncomputable def quartic_equation_roots : Prop :=
  let root_set := {x : ℂ | 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0} in
  ∃ (r1 r2 r3 r4 : ℂ),
    r1 ∈ root_set ∧ r2 ∈ root_set ∧ r3 ∈ root_set ∧ r4 ∈ root_set ∧
    (∀ x ∈ root_set, x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4)

theorem find_roots : quartic_equation_roots :=
sorry

end find_roots_l596_596563


namespace part1_part2_part3_l596_596618

noncomputable def f (a: ℝ) (x: ℝ) := (a * Real.exp x / x) + x

theorem part1 (a: ℝ) (f := f a) (x := 1) :
  ∀ a, (Deriv.deriv (f a) x = 1) → 
       (f a 1 - 1) = 0 →
       a = 1 / Real.exp 1 := 
sorry

theorem part2 (a: ℝ) (f := f a) :
  a < 0 → 
  ∀ x < 0, Deriv.deriv (f a) x > 0 →
  ∀ x > 0, Deriv.deriv (f a) x = 0 → 
  p > 0 → False := 
sorry

theorem part3 (a: ℝ) (f := f a) :
  a > 0 →
  ∃ x, x ∈ (0, 1) ∧ Deriv.deriv (f a) x = 0 
  ∧ f a x > 0 ∧
  ∃ y, y ∈ (-∞, 0) ∧
  Deriv.deriv (f a) y = 0 ∧ 
  f a y < 0 :=
sorry

end part1_part2_part3_l596_596618


namespace simplify_expression_l596_596776

variable (q : Int) -- condition that q is an integer

theorem simplify_expression (q : Int) : 
  ((7 * q + 3) - 3 * q * 2) * 4 + (5 - 2 / 4) * (8 * q - 12) = 40 * q - 42 :=
  by
  sorry

end simplify_expression_l596_596776


namespace convert_exp_to_rectangular_form_l596_596542

theorem convert_exp_to_rectangular_form : exp (13 * π * complex.I / 2) = complex.I :=
by
  sorry

end convert_exp_to_rectangular_form_l596_596542


namespace median_square_length_l596_596271

theorem median_square_length 
  {A B C O : Point}
  (hAOMedian : is_median A O B C)
  (hAC : |AC| = b)
  (hAB : |AB| = c)
  (m_a : ℝ) : 
  m_a = |AO| → m_a^2 = (1/2) * b^2 + (1/2) * c^2 - (1/4) * (|BC|^2) := 
by
  sorry

end median_square_length_l596_596271


namespace parallel_lines_a_value_l596_596633

theorem parallel_lines_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
      (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l596_596633


namespace monotonic_intervals_of_f_min_integer_value_of_a_l596_596156

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2

theorem monotonic_intervals_of_f (a x : ℝ) : 
  (a ≤ 0 → ∀ x > 0, 0 < f a x) ∧ 
  (a > 0 → (∀ x, 0 < x ∧ x < sqrt (1 / a) → 0 < f a x) ∧ (∀ x, x > sqrt (1 / a) → f a x < 0)) :=
begin
  sorry -- Proof will go here
end

theorem min_integer_value_of_a (a x : ℝ) : 
  (∀ x, f a x ≤ (a - 1) * x - 1) → a ≥ 2 :=
begin
  sorry -- Proof will go here
end

end monotonic_intervals_of_f_min_integer_value_of_a_l596_596156


namespace number_of_valid_n_l596_596163

noncomputable def is_arithmetic_seq (seq : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n > 0 → seq n = seq (n - 1) + d for some d : ℝ

noncomputable def sum_first_n_terms (seq : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * seq 1 + (n - 1) * d) for some d : ℝ

theorem number_of_valid_n (a b : ℕ → ℝ) 
  (A B : ℕ → ℝ)
  (h₁ : is_arithmetic_seq a)
  (h₂ : is_arithmetic_seq b)
  (h₃ : ∀ n, A n = sum_first_n_terms a n)
  (h₄ : ∀ n, B n = sum_first_n_terms b n)
  (h₅ : ∀ n, A n / B n = (7 * n + 45) / (n + 3)) :
  { n : ℕ | n > 0 ∧ (a n / b n).denom = 1 }.to_finset.card = 5 :=
sorry

end number_of_valid_n_l596_596163


namespace max_possible_value_l596_596254

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596254


namespace tan_arccot_l596_596039

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596039


namespace vegetable_bins_takeup_l596_596549

theorem vegetable_bins_takeup : 
  let T := 0.75 
  let S := 0.125 
  let P := 0.5 
  let V := T - S - P 
  V = 0.125 := 
by
  let T := 0.75
  let S := 0.125
  let P := 0.5
  let V := T - S - P
  calc
    V = T - S - P : by rfl
    ... = 0.75 - 0.125 - 0.5 : by rfl
    ... = 0.125 : by norm_num
  
sorry

end vegetable_bins_takeup_l596_596549


namespace basil_pots_count_l596_596469

theorem basil_pots_count (B : ℕ) (h1 : 9 * 18 + 6 * 30 + 4 * B = 354) : B = 3 := 
by 
  -- This is just the signature of the theorem. The proof is omitted.
  sorry

end basil_pots_count_l596_596469


namespace zero_in_tens_place_l596_596697

variable {A B : ℕ} {m : ℕ}

-- Define the conditions
def condition1 (A : ℕ) (B : ℕ) (m : ℕ) : Prop :=
  ∀ A B : ℕ, ∀ m : ℕ, A * 10^(m+1) + B = 9 * (A * 10^m + B)

theorem zero_in_tens_place (A B : ℕ) (m : ℕ) :
  condition1 A B m → m = 1 :=
by
  intro h
  sorry

end zero_in_tens_place_l596_596697


namespace revised_lemonade_calories_l596_596574

def lemonade (lemon_grams sugar_grams water_grams lemon_calories_per_50grams sugar_calories_per_100grams : ℕ) :=
  let lemon_cals := lemon_calories_per_50grams
  let sugar_cals := (sugar_grams / 100) * sugar_calories_per_100grams
  let water_cals := 0
  lemon_cals + sugar_cals + water_cals

def lemonade_weight (lemon_grams sugar_grams water_grams : ℕ) :=
  lemon_grams + sugar_grams + water_grams

def caloric_density (total_calories : ℕ) (total_weight : ℕ) := (total_calories : ℚ) / total_weight

def calories_in_serving (density : ℚ) (serving : ℕ) := density * serving

theorem revised_lemonade_calories :
  let lemon_calories := 32
  let sugar_calories := 579
  let total_calories := lemonade 50 150 300 lemon_calories sugar_calories
  let total_weight := lemonade_weight 50 150 300
  let density := caloric_density total_calories total_weight
  let serving_calories := calories_in_serving density 250
  serving_calories = 305.5 := sorry

end revised_lemonade_calories_l596_596574


namespace ellipse_foci_coordinates_l596_596796

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
    x^2 / 16 + y^2 / 25 = 1 → (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3) :=
by
  sorry

end ellipse_foci_coordinates_l596_596796


namespace max_roads_city_condition_l596_596237

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596237


namespace part1_part2_l596_596712

variables {A B C : Type} [metric_space A]
variables (a b c : ℝ)
variables [triangle : triangle A B C]
variable (angle_A : angle A B C)
variable (angle_B : angle B C A)
variable (angle_C : angle C A B)

-- Part 1
theorem part1 (h1 : ∠A = 3 * ∠B) : 
  (a^2 - b^2) * (a - b) = b * c^2 := 
sorry

-- Part 2
theorem part2 (h2 : (a^2 - b^2) * (a - b) = b * c^2) : 
  ∠A = 3 * ∠B ∨ ∠A ≠ 3 * ∠B :=
sorry

end part1_part2_l596_596712


namespace reseating_ways_l596_596786

noncomputable def S : ℕ → ℕ
| 0        := 1
| 1        := 1
| (n + 2)  := S n + S (n + 1)

/-- Ten friends are divided into two groups of five and sit in two separate 
rows of 5 seats each. They all get up and then reseat themselves in their 
respective rows, each sitting in the seat they were in before or a seat next 
to the one they occupied before. Prove that the number of ways the friends 
can be reseated is 64. -/
theorem reseating_ways : S 5 * S 5 = 64 :=
by
  -- calculation placeholder
  sorry


end reseating_ways_l596_596786


namespace find_marks_in_english_l596_596544

theorem find_marks_in_english (math_marks: ℝ) (physics_marks: ℝ) (chemistry_marks: ℝ) 
(biology_marks: ℝ) (num_subjects: ℕ) (average_marks: ℝ):
  math_marks = 65 →
  physics_marks = 82 →
  chemistry_marks = 67 →
  biology_marks = 90 →
  num_subjects = 5 →
  average_marks = 75.6 →
  ∃ E : ℝ, E = 74 :=
by
  intros
  use 74
  sorry

end find_marks_in_english_l596_596544


namespace part1_part2_l596_596157

open Real

noncomputable def part1_statement (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0

noncomputable def part2_statement (x : ℝ) : Prop := 
  ∀ (m : ℝ), |m| ≤ 1 → (m * x^2 - 2 * m * x - 1 < 0)

theorem part1 : part1_statement m ↔ (-1 < m ∧ m ≤ 0) :=
sorry

theorem part2 : part2_statement x ↔ ((1 - sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + sqrt 2)) :=
sorry

end part1_part2_l596_596157


namespace max_value_of_a_l596_596648

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem max_value_of_a (f : ℝ → ℝ) :
  is_odd f →
  is_decreasing f →
  (∀ x : ℝ, f (cos (2 * x) + sin x) + f (sin x - a) ≤ 0) →
  a ≤ -3 :=
by
  sorry

end max_value_of_a_l596_596648


namespace min_value_of_f_on_interval_l596_596348

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4*x - 2

theorem min_value_of_f_on_interval : 
  (∃ x ∈ set.Icc 1 4, ∀ y ∈ set.Icc 1 4, f(x) ≤ f(y)) ∧ f 1 = -2 :=
by 
  sorry

end min_value_of_f_on_interval_l596_596348


namespace max_possible_N_in_cities_l596_596243

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596243


namespace count_two_digit_nums_div_by_7_l596_596101

theorem count_two_digit_nums_div_by_7  : 
  let two_digit_numbers := {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b};
  let valid_nums := {n ∈ two_digit_numbers | ∃ a b : ℕ, n = 10 * a + b ∧ (8 * a - b) % 7 = 0};
  valid_nums.card = SOME_NUMBER :=
by sorry

end count_two_digit_nums_div_by_7_l596_596101


namespace eccentricity_hyperbola_l596_596139

-- Define the focus points and conditions for the hyperbola
variable {a b c : ℝ} (h1 : a > 0) (h2 : b > 0)
variable {F1 F2 O : ℝ × ℝ}

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Conditions for the foci and the area of the triangle
variable (r : ℝ) (hf1 : r = |OF1|) (intersects_at : (ℝ × ℝ))
variable (area_triangle : ℝ) (h3 : area_triangle = a^2)
variable (hf1_dist : |intersects_at - F1|^2 + |intersects_at - F2|^2 = 4*c^2)
variable (hf2_dist : |intersects_at - F1| - |intersects_at - F2| = 2*a)
variable (c_def : c = sqrt 2 * a)

-- Prove eccentricity
theorem eccentricity_hyperbola : ∃ (e : ℝ), e = sqrt 2 := sorry

end eccentricity_hyperbola_l596_596139


namespace sum_evaluation_l596_596548

-- Define the sum S as given in the problem
def S : ℝ := ∑ n in Finset.range (9999) \ {0, 1}, 1 / (Real.sqrt (n + 2 + Real.sqrt ((n + 2) ^ 2 - 4)))

-- Define the variables a, b, c
def a : ℕ := 139
def b : ℕ := 49
def c : ℕ := 2

-- Provide the final proof statement
theorem sum_evaluation : a + b + c = 190 :=
by
  -- Here we would give the proof based on the conditions and solution steps
  sorry

end sum_evaluation_l596_596548


namespace factorize_expression_l596_596554

theorem factorize_expression (a b : ℝ) : 3 * a ^ 2 - 3 * b ^ 2 = 3 * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l596_596554


namespace solve_matrix_equation_l596_596382

def M : Matrix (Fin 2) (Fin 2) ℚ := !![5, 2; 4, 1]
def N : Matrix (Fin 2) (Fin 1) ℚ := !![5; 8]

theorem solve_matrix_equation :
  let X := !![(11 : ℚ) / 3; (-20 : ℚ) / 3]
  M.mul_vec X = N :=
by
  sorry

end solve_matrix_equation_l596_596382


namespace distance_Esha_behind_Anusha_l596_596968

-- Given conditions
variables (A_d B_d E_d : ℕ)
hypothesis h1 : A_d = B_d + 10
hypothesis h2 : B_d = E_d + 10

-- Prove the distance Esha is behind Anusha when Anusha reaches the finish line
theorem distance_Esha_behind_Anusha : E_d = A_d - 20 :=
by {
  have h3 : A_d = E_d + 20, from eq.trans h1 (by rw h2),
  linarith,
}

end distance_Esha_behind_Anusha_l596_596968


namespace max_possible_N_in_cities_l596_596245

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596245


namespace independence_events_exactly_one_passing_l596_596930

-- Part 1: Independence of Events

def event_A (die1 : ℕ) : Prop :=
  die1 % 2 = 1

def event_B (die1 die2 : ℕ) : Prop :=
  (die1 + die2) % 3 = 0

def P_event_A : ℚ :=
  1 / 2

def P_event_B : ℚ :=
  1 / 3

def P_event_AB : ℚ :=
  1 / 6

theorem independence_events : P_event_AB = P_event_A * P_event_B :=
by
  sorry

-- Part 2: Probability of Exactly One Passing the Assessment

def probability_of_hitting (p : ℝ) : ℝ :=
  1 - (1 - p)^2

def P_A_hitting : ℝ :=
  0.7

def P_B_hitting : ℝ :=
  0.6

def probability_one_passing : ℝ :=
  (probability_of_hitting P_A_hitting) * (1 - probability_of_hitting P_B_hitting) + (1 - probability_of_hitting P_A_hitting) * (probability_of_hitting P_B_hitting)

theorem exactly_one_passing : probability_one_passing = 0.2212 :=
by
  sorry

end independence_events_exactly_one_passing_l596_596930


namespace city_budget_allocation_l596_596793

theorem city_budget_allocation (annual_budget : ℕ) (pct_infrastructure pct_healthcare : ℕ) (transp_fund : ℕ)
  (h_budget : annual_budget = 80_000_000)
  (h_pct_infrastructure : pct_infrastructure = 30)
  (h_transp_fund : transp_fund = 10_000_000)
  (h_pct_healthcare : pct_healthcare = 15) :
  annual_budget - (pct_infrastructure * annual_budget / 100 + transp_fund + pct_healthcare * annual_budget / 100) = 34_000_000 := by
  sorry

end city_budget_allocation_l596_596793


namespace vitya_catches_up_in_5_minutes_l596_596852

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596852


namespace spherical_coords_negate_xy_l596_596438

theorem spherical_coords_negate_xy :
  ∀ (x y z : ℝ),
    -- Conditions
    x = 3 * real.sin (π / 4) * real.cos (5 * π / 6) →
    y = 3 * real.sin (π / 4) * real.sin (5 * π / 6) →
    z = 3 * real.cos (π / 4) →
    -- Question: Prove the spherical coordinates of (-x, -y, z)
    ∃ ρ θ ϕ : ℝ, ρ = 3 ∧ θ = 11 * π / 6 ∧ ϕ = π / 4 :=
by
  intros x y z hx hy hz
  -- Placeholder for proof, this is skipped
  sorry

end spherical_coords_negate_xy_l596_596438


namespace calculate_savings_l596_596014

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l596_596014


namespace tan_arccot_l596_596042

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596042


namespace import_tax_paid_l596_596394

theorem import_tax_paid (V : ℝ) (hV : V = 2250) : 
  let excess_value := V - 1000
  let tax_rate := 0.07 
  let T := tax_rate * excess_value
  T = 87.50 := by
    -- condition V = 2250
    rw hV
    -- definition of excess_value, tax_rate, and T
    let excess_value := 2250 - 1000
    let tax_rate := 0.07
    let T := tax_rate * excess_value
    have h1: excess_value = 1250 := by norm_num
    have h2: T = 0.07 * 1250 := by rw h1
    norm_num at h2
    exact h2.symm

end import_tax_paid_l596_596394


namespace exponential_to_rectangular_form_l596_596523

theorem exponential_to_rectangular_form : 
  (Real.exp (Complex.i * (13 * Real.pi / 2))) = Complex.i :=
by
  sorry

end exponential_to_rectangular_form_l596_596523


namespace pumping_time_l596_596423

-- Define the conditions
def length_of_basement := 30 -- feet
def width_of_basement := 40 -- feet
def depth_of_water := 24 -- inches
def pumps := 4
def pump_rate := 10 -- gallons per minute per pump
def gallon_per_cubic_foot := 7.5 -- gallons

-- Convert depth from inches to feet
def depth_of_water_feet := depth_of_water / 12

-- Calculate the volume of water in cubic feet
def volume_of_water := depth_of_water_feet * length_of_basement * width_of_basement

-- Calculate the volume of water in gallons
def total_gallons_of_water := volume_of_water * gallon_per_cubic_foot

-- Calculate the total pump rate
def total_pump_rate := pumps * pump_rate

-- Calculate the time required to pump out all the water
def time_to_empty_basement := total_gallons_of_water / total_pump_rate

-- The theorem: the time required is 450 minutes
theorem pumping_time : time_to_empty_basement = 450 := by
  sorry

end pumping_time_l596_596423


namespace yarn_ball_ratio_l596_596278

open Real

theorem yarn_ball_ratio:
  ∀ (size_first size_second size_third : ℝ),
  size_first = 1 / 2 * size_second →
  size_third = 27 →
  size_second = 18 →
  size_third / size_first = 3 :=
by
  intros size_first size_second size_third h1 h2 h3
  have h4 : size_first = 9 := by
    rw [h1, h3]
    norm_num
  rw [h4] at h2
  rw [h2]
  rw [div_eq_mul_inv, mul_inv_cancel] 
  norm_num
  sorry

end yarn_ball_ratio_l596_596278


namespace cashier_total_value_l596_596934

theorem cashier_total_value (total_bills : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ)
  (h1 : total_bills = 30) (h2 : ten_bills = 27) (h3 : twenty_bills = 3) :
  (10 * ten_bills + 20 * twenty_bills) = 330 :=
by
  sorry

end cashier_total_value_l596_596934


namespace max_ATM_cards_l596_596422

-- Define the maximum number of passwords problem
theorem max_ATM_cards (total_passwords : ℕ := 1000000) (no_three_consecutive_digits : ℕ := 36910): 
  total_passwords - no_three_consecutive_digits = 963090 :=
by
  let total_passwords := 1000000
  let no_three_consecutive_digits := 36910
  have calc : total_passwords - no_three_consecutive_digits = 1000000 - 36910, from rfl
  have step1 : 1000000 - 36910 = 963090, by exact I_am_a_machine
  exact step1
  sorry


end max_ATM_cards_l596_596422


namespace Arman_total_earnings_two_weeks_l596_596737

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end Arman_total_earnings_two_weeks_l596_596737


namespace vitya_catch_up_l596_596883

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596883


namespace solution_set_l596_596153

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if x > 0 then -2 else x^2 + b*x + c

-- State the necessary conditions
def condition1 := f (-4) = f 0
def condition2 := f (-2) = 0

-- Define the main statement to prove
theorem solution_set (b c : ℝ) (h1 : condition1) (h2 : condition2) :
  { x : ℝ | f x ≤ 1 } = { x : ℝ | x > 0 ∨ (-3 ≤ x ∧ x ≤ -1) } :=
sorry

end solution_set_l596_596153


namespace true_propositions_count_l596_596360

theorem true_propositions_count (A B C : Type) (h : ∀ P Q R : triangle, angle PQR = 90 → is_right_triangle PQR) :
  count_true_propositions = 2 :=
by
  sorry

end true_propositions_count_l596_596360


namespace geo_seq_sum_l596_596599

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_sum (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 0 + a 1 = 30) (h4 : a 3 + a 4 = 120) :
  a 6 + a 7 = 480 :=
sorry

end geo_seq_sum_l596_596599


namespace probability_divisible_by_five_l596_596818

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l596_596818


namespace largest_class_students_l596_596923

theorem largest_class_students (x : ℕ) 
  (h1 : let num_students := [x, x - 2, x - 4, x - 6, x - 8] in 
        num_students.sum = 115) : 
  x = 27 := 
by
  sorry

end largest_class_students_l596_596923


namespace product_of_roots_l596_596744

noncomputable def polynomial_minimal_root_625 : Polynomial ℚ :=
  Polynomial.sum (Polynomial.monomial 4 1) (Polynomial.monomial 0 (-5))

theorem product_of_roots :
  let P : Polynomial ℚ := polynomial_minimal_root_625 in
  (∏ root in P.roots, root) = -1600 :=
sorry

end product_of_roots_l596_596744


namespace percentage_exceed_l596_596186

theorem percentage_exceed (x y : ℝ) (h : y = x + (0.25 * x)) : (y - x) / x * 100 = 25 :=
by
  sorry

end percentage_exceed_l596_596186


namespace triangle_inequality_l596_596283

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  sqrt (a^2 + a*b + b^2) + sqrt (b^2 + b*c + c^2) + sqrt (c^2 + c*a + a^2) 
  ≤ sqrt (5*a^2 + 5*b^2 + 5*c^2 + 4*a*b + 4*b*c + 4*c*a) := 
sorry

end triangle_inequality_l596_596283


namespace tan_arccot_l596_596021

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596021


namespace take_home_pay_is_correct_l596_596715

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l596_596715


namespace smallest_n_for_g_eq_nine_l596_596112

def sum_of_digits_base_five (n : ℕ) : ℕ :=
  (n.digits 5).sum

def sum_of_digits_base_nine (n : ℕ) : ℕ :=
  (n.digits 9).sum

def f (n : ℕ) : ℕ :=
  sum_of_digits_base_five n

def g (n : ℕ) : ℕ :=
  sum_of_digits_base_nine (f n)

theorem smallest_n_for_g_eq_nine :
  ∃ n, g(n) = 9 ∧ n % 100 = 44 := by
  sorry

end smallest_n_for_g_eq_nine_l596_596112


namespace min_value_interval_l596_596347

namespace MinValueProof

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem min_value_interval : ∃ x ∈ set.Icc (-1 : ℝ) 0, ∀ y ∈ set.Icc (-1 : ℝ) 0, f(y) ≥ f(x) ∧ f(x) = 2 :=
by
  existsi (0 : ℝ)
  simp
  intros y hy
  sorry

end MinValueProof

end min_value_interval_l596_596347


namespace find_worst_competitor_l596_596950

structure Competitor :=
  (name : String)
  (gender : String)
  (generation : String)

-- Define the competitors
def man : Competitor := ⟨"man", "male", "generation1"⟩
def wife : Competitor := ⟨"wife", "female", "generation1"⟩
def son : Competitor := ⟨"son", "male", "generation2"⟩
def sister : Competitor := ⟨"sister", "female", "generation1"⟩

-- Conditions
def opposite_genders (c1 c2 : Competitor) : Prop :=
  c1.gender ≠ c2.gender

def different_generations (c1 c2 : Competitor) : Prop :=
  c1.generation ≠ c2.generation

noncomputable def worst_competitor : Competitor :=
  sister

def is_sibling (c1 c2 : Competitor) : Prop :=
  (c1 = man ∧ c2 = sister) ∨ (c1 = sister ∧ c2 = man)

-- Theorem statement
theorem find_worst_competitor (best_competitor : Competitor) :
  (opposite_genders worst_competitor best_competitor) ∧
  (different_generations worst_competitor best_competitor) ∧
  ∃ (sibling : Competitor), (is_sibling worst_competitor sibling) :=
  sorry

end find_worst_competitor_l596_596950


namespace ellipse_problem_l596_596120

open Real

theorem ellipse_problem (a b : ℝ) (h : a > b ∧ b > 0)
    (h1 : sqrt 3 ^ 2 / a ^ 2 + (1 / 2) ^ 2 / b ^ 2 = 1)
    (k1 k2 : ℝ) (h2 : k1 * k2 ≠ 0)
    (h3 : ∀ (P : ℝ × ℝ) (hx : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1),
        |(b * P.2)/(a * (1 - P.1/a))| + |(b * P.2)/(a * (1 + P.1/a))| ≥ 2 * b / a ∧ 
        |(b * P.2)/(a * (1 - P.1/a))| + |(b * P.2)/(a * (1 + P.1/a))| = 1 ) :
    (1 = 2 * b / a) → 
    (x y : ℝ) (h4 : x = sqrt 3 ∧ y = 1 / 2) :
    ∀ (a b : ℝ), (3 / a^2 + 1 / (4 * b ^ 2) = 1) → 
    (∃ (a b : ℝ), (a = 2 ∧ b = 1)) → 
    (∃ (a b : ℝ), ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1 ↔ y = sqrt (1 - 3/4))) :=
    sorry

end ellipse_problem_l596_596120


namespace sum_of_roots_l596_596502

theorem sum_of_roots : 
  let equation := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7) = 0
  in (root1, root2 : ℚ) (h1 : (3 * root1 + 4) = 0 ∨ (2 * root1 - 12) = 0) 
    (h2 : (3 * root2 + 4) = 0 ∨ (2 * root2 - 12) = 0) :
    root1 + root2 = 14 / 3
by 
  sorry

end sum_of_roots_l596_596502


namespace roger_piles_of_quarters_l596_596772

theorem roger_piles_of_quarters (Q : ℕ) 
  (h₀ : ∃ Q : ℕ, True) 
  (h₁ : ∀ p, (p = Q) → True)
  (h₂ : ∀ c, (c = 7) → True) 
  (h₃ : Q * 14 = 42) : 
  Q = 3 := 
sorry

end roger_piles_of_quarters_l596_596772


namespace equal_sum_sequence_definition_l596_596468

/-- Defining an arithmetic sequence. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ n : ℕ, a (n + 1) - a n = k

/-- Defining an equal sum sequence. -/
def is_equal_sum_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n

/-- Theorem stating that an equal sum sequence is defined as having each term and its preceding term having an equal sum from the second term. -/
theorem equal_sum_sequence_definition :
  is_equal_sum_sequence = (λ a, ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n) :=
sorry

end equal_sum_sequence_definition_l596_596468


namespace amount_spent_on_tumbler_l596_596765

def initial_amount : ℕ := 50
def spent_on_coffee : ℕ := 10
def amount_left : ℕ := 10
def total_spent : ℕ := initial_amount - amount_left

theorem amount_spent_on_tumbler : total_spent - spent_on_coffee = 30 := by
  sorry

end amount_spent_on_tumbler_l596_596765


namespace number_of_integers_for_negative_polynomial_l596_596625

theorem number_of_integers_for_negative_polynomial :
  let polynomial := λ x : ℤ, x^4 - 61 * x^2 + 60 in 
  (finset.card $ { x | polynomial x < 0 }.to_finset) = 12 :=
by
  let polynomial := λ x : ℤ, x^4 - 61 * x^2 + 60
  sorry

end number_of_integers_for_negative_polynomial_l596_596625


namespace count_not_5nice_6nice_l596_596572

def is_nice (M j : ℕ) : Prop := ∃ b : ℕ, b > 0 ∧ nat.divisors_count (b^j) = M

def num_less_than_500 (n : ℕ) : ℕ := (500 / n)

def num_5nice : ℕ := num_less_than_500 5 - num_less_than_500 25
def num_6nice : ℕ := num_less_than_500 6 - num_less_than_500 36
def num_30nice : ℕ := num_less_than_500 30 - num_less_than_500 900

def count_not_5nice_and_not_6nice : ℕ := 500 - (num_5nice + num_6nice - num_30nice)

theorem count_not_5nice_6nice : count_not_5nice_and_not_6nice = 333 := by
  sorry

end count_not_5nice_6nice_l596_596572


namespace count_valid_colorings_l596_596378

-- Define the condition that for all pairs of balls a, b such that a + b < 13, color propagation rules apply
def color_propagation_rule (is_red : ℕ → Bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a < 13 → b < 13 →
  (is_red a ∧ is_red b → is_red (a + b)) ∧
  (¬is_red a ∧ ¬is_red b → ¬is_red (a + b))

-- Define the statement of the theorem
theorem count_valid_colorings : 
  ∃ (f : ℕ → Bool), color_propagation_rule f ∧ 
  (∃ g : ℕ → Bool, color_propagation_rule g → f = g) ∧ 
  (finset.univ.filter f).card = 6 :=
sorry

end count_valid_colorings_l596_596378


namespace find_ns_l596_596109

-- Define d(n) for number of positive divisors of n
def d (n : ℕ) : ℕ := (n.divisors).card

theorem find_ns (n : ℕ) (hn : 0 < n) : 
  (d(n)^3 = 4*n) ↔ (n = 2 ∨ n = 128 ∨ n = 2000) := 
by
  sorry

end find_ns_l596_596109


namespace smallest_digit_change_l596_596326

theorem smallest_digit_change :
  ∃ d : ℕ, (d = 6) ∧ (∃ a b c : ℕ, a = 639 ∧ b = 964 ∧ c = 852 ∧ a + b + c = 2457) :=
begin
  sorry
end

end smallest_digit_change_l596_596326


namespace smallest_possible_n_l596_596293

theorem smallest_possible_n (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∀ i, i < n → |x i| < 1)
  (h2 : ∑ i in finset.range n, |x i| = 17 + |∑ i in finset.range n, x i|) :
  n = 18 :=
by sorry

end smallest_possible_n_l596_596293


namespace area_of_circle_from_polar_eq_l596_596342

-- Define the polar equation as a function
def polar_eq (θ : ℝ) : ℝ :=
  3 * Real.cos θ - 4 * Real.sin θ

-- Define the main theorem statement
theorem area_of_circle_from_polar_eq :
  ∀ θ, polar_eq θ = 3 * Real.cos θ - 4 * Real.sin θ →
  ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    (center = (3/2, -2)) ∧
    (radius = 5/2) ∧
    (π * radius^2 = 25/4 * π) :=
by
  sorry

#eval area_of_circle_from_polar_eq

end area_of_circle_from_polar_eq_l596_596342


namespace even_distinct_probability_l596_596467

def is_between_100_and_999 (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  digits.fst ≠ digits.snd ∧ digits.fst ≠ digits.snd.snd ∧ digits.snd ≠ digits.snd.snd

def satisfies_conditions (n : ℕ) : Prop :=
  is_between_100_and_999 n ∧ is_even n ∧ has_distinct_digits n

theorem even_distinct_probability :
  ∃ (p : ℚ), satisfies_conditions ∧ p = 82 / 225 :=
sorry

end even_distinct_probability_l596_596467


namespace seat_number_X_l596_596831

theorem seat_number_X (X : ℕ) (h1 : 42 - 30 = X - 6) : X = 18 :=
by
  sorry

end seat_number_X_l596_596831


namespace initial_numbers_is_five_l596_596791

theorem initial_numbers_is_five : 
  ∀ (n S : ℕ), 
    (12 * n = S) →
    (10 * (n - 1) = S - 20) → 
    n = 5 := 
by sorry

end initial_numbers_is_five_l596_596791


namespace legendre_symbol_neg3_l596_596291

theorem legendre_symbol_neg3 (p : ℕ) (h_prime : Nat.Prime p) :
  ( ∃ k : ℤ, p = 6 * k + 1 ∧ (legendre (-3) p = 1) ) ∨ 
  ( ∃ k : ℤ, p = 6 * k - 1 ∧ (legendre (-3) p = -1) ) := 
sorry

end legendre_symbol_neg3_l596_596291


namespace log_sum_geometric_seq_l596_596127

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem log_sum_geometric_seq (a : ℕ → ℝ) (n : ℕ) 
  (h1 : is_geometric_sequence a) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a n > 0) 
  (h3 : ∀ n : ℕ, n ≥ 3 → a 5 * a (2 * n - 5) = 2 ^ (2 * n)) 
  (h4 : n ≥ 1) :
  ∑ i in range n, Real.log2 (a (2 * i + 1)) = n^2 :=
sorry

end log_sum_geometric_seq_l596_596127


namespace exponential_to_rectangular_form_l596_596524

theorem exponential_to_rectangular_form : 
  (Real.exp (Complex.i * (13 * Real.pi / 2))) = Complex.i :=
by
  sorry

end exponential_to_rectangular_form_l596_596524


namespace triangle_BD_length_l596_596698

noncomputable def triangle_length_BD : ℝ :=
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  BD

theorem triangle_BD_length : triangle_length_BD = 63 :=
by
  -- Definitions and assumptions
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)

  -- Formal proof logic corresponding to solution steps
  sorry

end triangle_BD_length_l596_596698


namespace problem_statement_l596_596174

theorem problem_statement (a : ℝ) (h : a^2 - 2 * a + 1 = 0) : 4 * a - 2 * a^2 + 2 = 4 := 
sorry

end problem_statement_l596_596174


namespace price_when_x2_y2_is_3_l596_596993

-- Conditions: price of HMMT is directly proportional to x and inversely proportional to y
def price (k x y : ℝ) : ℝ := k * (x / y)

-- Given values
def x1 : ℝ := 8
def y1 : ℝ := 4
def h1 : ℝ := 12

-- New values that we need to determine the price for
def x2 : ℝ := 4
def y2 : ℝ := 8

-- Theorem stating that the price for x = 4 and y = 8 is 3 dollars
theorem price_when_x2_y2_is_3 (k : ℝ) (h : ℝ) (h2 : ℝ) :
  (price k x1 y1 = h1) → (k = 6) → (price k x2 y2 = 3) :=
by
  intros h12 k6
  sorry

end price_when_x2_y2_is_3_l596_596993


namespace tan_arccot_l596_596041

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596041


namespace tan_double_angle_gt_double_tan_l596_596110

theorem tan_double_angle_gt_double_tan (α : ℝ) 
  (h1 : 0 < α) (h2 : α < real.pi / 4) 
  (h3 : 0 < real.tan α) (h4 : real.tan α < 1) : 
  real.tan (2 * α) > 2 * real.tan α :=
by
  sorry

end tan_double_angle_gt_double_tan_l596_596110


namespace convert_exp_to_rectangular_form_l596_596538

theorem convert_exp_to_rectangular_form : exp (13 * π * complex.I / 2) = complex.I :=
by
  sorry

end convert_exp_to_rectangular_form_l596_596538


namespace cyclic_quadrilateral_of_isosceles_triangle_and_circle_l596_596972

open Geometry

variables [EuclideanGeometry] {A B C D E F : Point}

-- Given Definitions and Conditions
def is_isosceles_triangle (ABC : Triangle) : Prop := AB = AC
def ratio_AD_DC (AD DC BC CE : ℝ) : Prop := AD / DC = BC / (2 * CE)
def intersects_at (ω : Circle) (DE : Line) (F : Point) : Prop := F ∈ ω ∧ F ∈ DE

theorem cyclic_quadrilateral_of_isosceles_triangle_and_circle
  (h_iso : is_isosceles_triangle ABC)
  (h_ratio : ratio_AD_DC AD DC BC CE)
  (h_intersect : intersects_at ω DE F)
  : cyclic_quadrilateral B C F D := by sorry

end cyclic_quadrilateral_of_isosceles_triangle_and_circle_l596_596972


namespace smallest_product_in_numSet_l596_596059

-- Define the set of numbers
def numSet : Set Int := {-8, -6, -2, 2, 4, 5}

-- Define a function to compute the set of all products obtained by multiplying two numbers from numSet
def products (s : Set Int) : Set Int := 
  {p | ∃ a b ∈ s, p = a * b}

-- Define the smallest element function for non-empty set
def smallest (s : Set Int) [h : Nonempty s] : Int :=
  Set.fold (λ (x y : Int), if x < y then x else y) h.some s

-- Proposition stating the smallest product
theorem smallest_product_in_numSet : smallest (products numSet) = -40 := 
by
  -- Proof to be done
  sorry

end smallest_product_in_numSet_l596_596059


namespace probability_divisible_by_five_l596_596820

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l596_596820


namespace vitya_catchup_time_l596_596879

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596879


namespace ratio_of_areas_of_inscribed_squares_l596_596104

-- Define the structures
variables (R : ℝ)

-- Define the side lengths of the inscribed squares in the segments
def side_length_semicircle (R : ℝ) : ℝ := 2 * R / Real.sqrt 5
def side_length_quadrant (R : ℝ) : ℝ := R * Real.sqrt 2 / 5

-- Define the problem statement
theorem ratio_of_areas_of_inscribed_squares :
  let x := side_length_semicircle R,
      y := side_length_quadrant R in
  (x / y)^2 = 10 :=
by sorry

end ratio_of_areas_of_inscribed_squares_l596_596104


namespace behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l596_596057

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 5 * x ^ 2 + 4

theorem behavior_of_g_as_x_approaches_infinity_and_negative_infinity :
  (∀ ε > 0, ∃ M > 0, ∀ x > M, g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x < -N, g x > ε) :=
by
  sorry

end behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l596_596057


namespace games_draw_fraction_l596_596262

-- Definitions from the conditions in the problems
def ben_win_fraction : ℚ := 4 / 9
def tom_win_fraction : ℚ := 1 / 3

-- The theorem we want to prove
theorem games_draw_fraction : 1 - (ben_win_fraction + (1 / 3)) = 2 / 9 := by
  sorry

end games_draw_fraction_l596_596262


namespace prime_factorization_of_expression_l596_596769

theorem prime_factorization_of_expression :
  ∃ a b c, Prime a ∧ Prime b ∧ Prime c ∧ (5 * 13 * 31 - 2 = a * b * c) :=
by
  use 3, 11, 61
  constructor
  exact Prime 3
  constructor
  exact Prime 11
  constructor
  exact Prime 61
  exact (5 * 13 * 31 - 2 = 3 * 11 * 61)
  sorry

end prime_factorization_of_expression_l596_596769


namespace inequality_l596_596641

variable (a b : ℝ)
variables (h : 0 < a) (hb : a ≤ b)

def LHS : ℝ := (2 / Real.sqrt 3) * Real.arctan ((2 * (b^2 - a^2)) / ((a^2 + 2) * (b^2 + 2)))
def RHS : ℝ := (4 / Real.sqrt 3) * Real.arctan (((b - a) * Real.sqrt 3) / (a + b + 2 * (1 + a * b)))
def integrand (x : ℝ) : ℝ := ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))
noncomputable def integral : ℝ := ∫ x in a..b, integrand x

theorem inequality : LHS a b ≤ integral a b ∧ integral a b ≤ RHS a b := by
  sorry

end inequality_l596_596641


namespace tan_arccot_l596_596038

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596038


namespace maximum_possible_value_of_N_l596_596210

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596210


namespace fill_time_of_combined_pumps_l596_596839

-- Definitions of the given conditions
def small_pump_rate := (1 / 4 : ℝ)  -- rate for small pump (tanks per hour)
def large_pump_rate := (2 : ℝ)      -- rate for large pump (tanks per hour)
def medium_pump_rate := (1 / 2 : ℝ) -- rate for medium pump (tanks per hour)

-- Combined rate calculation
def combined_rate := small_pump_rate + large_pump_rate + medium_pump_rate

-- Theorem statement
theorem fill_time_of_combined_pumps : combined_rate ≠ 0 → 1 / combined_rate = 4 / 11 :=
by
  intro h
  sorry

end fill_time_of_combined_pumps_l596_596839


namespace exam_correct_answers_l596_596691

theorem exam_correct_answers (C W : ℕ) 
  (h1 : C + W = 60)
  (h2 : 4 * C - W = 160) : 
  C = 44 :=
sorry

end exam_correct_answers_l596_596691


namespace tan_arccot_l596_596043

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596043


namespace find_k_l596_596684

namespace MathProof

-- Define the vector OA
def OA : ℝ × ℝ := (-3, 1)

-- Define the vector OB parameterized by k
def OB (k : ℝ) : ℝ × ℝ := (-2, k)

-- Define the condition that OA is perpendicular to AB and write it in Lean
def is_perpendicular_to (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define AB in terms of OA and OB
def AB (k : ℝ) : ℝ × ℝ := (OB k).map2 (λ x y => x - y) OA

-- Statement of the problem in Lean
theorem find_k (k : ℝ) (h : is_perpendicular_to OA (AB k)) : k = 4 := 
sorry

end MathProof

end find_k_l596_596684


namespace same_savings_weeks_l596_596322

def sara_initial_savings : ℤ := 4100
def sara_weekly_savings : ℤ := 10
def jim_weekly_savings : ℤ := 15

theorem same_savings_weeks : ∃ w : ℤ, 4100 + 10 * w = 15 * w :=
by {
  use 820,
  sorry
}

end same_savings_weeks_l596_596322


namespace min_frac_inv_l596_596621

variable (a m n x : ℝ)

def function_y (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 3) / Real.log a - 1

def point_A := (-2, -1 : ℝ) -- Coordinates of point A

def line_eq (m n : ℝ) (x y : ℝ) : ℝ := m * x + n * y + 1

def condition (m n : ℝ) : Prop := 2 * m + n = 1

theorem min_frac_inv (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m * n > 0) :
  1 / m + 2 / n = 8 :=
by
  sorry

end min_frac_inv_l596_596621


namespace min_ratio_T2_T1_l596_596456

-- Definitions for geometric elements
variables {A B C D E F: Type*}

variables (ABC : A → B → C → Triangleₓ)
variables (D E F : Pointₓ ABC)
variables (Dc Db Ea Ec Fa Fb : Pointₓ)

-- Conditions
def acute_triangle (ABC : Triangleₓ) : Prop :=  -- define that it's an acute triangle
  acute (angle A B C) ∧ acute (angle B C A) ∧ acute (angle C A B)

def is_altitude (D : Pointₓ) (ABC : Triangleₓ) : Prop :=
  -- Points D, E, F are the feet of the altitudes of the triangle
  altitude D A B C ∧ altitude E B A C ∧ altitude F C A B

def is_projection (D : Pointₓ) (Db Dc : Pointₓ) : Prop :=
  -- D_c, D_b are projections of points D, E, F on specific sides
  projection D Dc A B ∧ projection D Db A C ∧
  projection E Ea B C ∧ projection E Ec A B ∧ 
  projection F Fa B C ∧ projection F Fb A C

-- Areas of triangles formed by lines
def area_T1 (ABC : Triangleₓ) (Dc Db Ea Ec Fa Fb : Pointₓ) : Real :=
  -- T1: Area of the triangle formed by D_bD_c, E_cE_a, and F_aF_b
  area (triangle (segment D_b D_c) (segment E_c E_a) (segment F_a F_b))

def area_T2 (ABC : Triangleₓ) (Dc Db Ea Ec Fa Fb : Pointₓ) : Real :=
  -- T2: Area of the triangle formed by E_cF_b, D_bE_a, and F_aD_c
  area (triangle (segment E_c F_b) (segment D_b E_a) (segment F_a D_c))
  
theorem min_ratio_T2_T1 (ABC : Triangleₓ) (D E F : Pointₓ ABC) (Dc Db Ea Ec Fa Fb : Pointₓ) :
  acute_triangle ABC →
  is_altitude D ABC →
  is_projection D Dc Db →
  is_projection E Ea Ec →
  is_projection F Fa Fb →
  area_T2 ABC Dc Db Ea Ec Fa Fb / area_T1 ABC Dc Db Ea Ec Fa Fb = 25 :=
sorry

end min_ratio_T2_T1_l596_596456


namespace iso_triangle_partition_l596_596098

theorem iso_triangle_partition (m n : ℕ) (m_gt_1 : m > 1) (n_ge_3 : n ≥ 3)
    (partition : ∃ (mints : ℕ → Prop), (∀ i, mints i → isosceles_triangle i) ∧ (set.partition (set.univ : set (polygon n)) (mints '' (set.univ : set (fin m)))) (polygon n)) :
  (∃ (a b : ℕ), a ≠ b ∧ side_length (polygon n) a = side_length (polygon n) b) ↔ n = m + 2 := 
sorry

end iso_triangle_partition_l596_596098


namespace dot_product_computation_l596_596148

variables (a b : ℝ^3) (theta : ℝ)

-- Condition definitions
def angle_a_b := theta = 120 * (Real.pi / 180)
def norm_a := ‖a‖ = 2
def norm_b := ‖b‖ = 5

-- Theorem statement
theorem dot_product_computation (h1 : angle_a_b) (h2 : norm_a a) (h3 : norm_b b) : 
  (2 • a - b) • a = 13 := 
sorry

end dot_product_computation_l596_596148


namespace all_statements_correct_l596_596314

theorem all_statements_correct :
  (∀ (b h : ℝ), (3 * b * h = 3 * (b * h))) ∧
  (∀ (b h : ℝ), (1/2 * b * (1/2 * h) = 1/2 * (1/2 * b * h))) ∧
  (∀ (r : ℝ), (π * (2 * r) ^ 2 = 4 * (π * r ^ 2))) ∧
  (∀ (r : ℝ), (π * (3 * r) ^ 2 = 9 * (π * r ^ 2))) ∧
  (∀ (s : ℝ), ((2 * s) ^ 2 = 4 * (s ^ 2)))
  → False := 
by 
  intros h
  sorry

end all_statements_correct_l596_596314


namespace decreasing_sequence_inequality_l596_596939

theorem decreasing_sequence_inequality (x : ℕ → ℝ) (h_decreasing : ∀ n m, n ≤ m → x n ≥ x m)
  (h_positive : ∀ n, x n > 0)
  (h_condition : ∀ n, ∑ i in finset.range n, x (i*i + 1) / (i + 1) ≤ 1) 
  (n : ℕ) :
  ∑ i in finset.range n, x (i + 1) / (i + 1) ≤ 3 :=
sorry

end decreasing_sequence_inequality_l596_596939


namespace angle_between_skew_lines_l596_596799

-- Define the Lean statement based on the conditions and the question-answer tuple.
theorem angle_between_skew_lines (α β : Plane) (a b : Line) (h1 : dihedral_angle α l β = 60) (h2 : a ⊥ α) (h3 : b ⊥ β) : angle a b = 60 := 
sorry

end angle_between_skew_lines_l596_596799


namespace simplify_expression_l596_596389

theorem simplify_expression : 3 - (-3)⁻³ = 82 / 27 :=
by sorry

end simplify_expression_l596_596389


namespace length_of_segment_AB_l596_596706

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l596_596706


namespace min_elements_with_properties_l596_596673

noncomputable def median (s : List ℝ) : ℝ :=
if h : s.length % 2 = 1 then
  s.nthLe (s.length / 2) (by simp [h])
else
  (s.nthLe (s.length / 2 - 1) (by simp [Nat.sub_right_comm, h, *]) + s.nthLe (s.length / 2) (by simp [h])) / 2

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

noncomputable def mode (s : List ℝ) : ℝ :=
s.groupBy (· = ·)
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (·.snd |>.value)

theorem min_elements_with_properties :
  ∃ s : List ℝ, 
    s.length ≥ 6 ∧ 
    median s = 3 ∧ 
    mean s = 5 ∧ 
    ∃! m, mode s = m ∧ m = 6 :=
by
  sorry

end min_elements_with_properties_l596_596673


namespace comic_books_stack_order_count_l596_596311

theorem comic_books_stack_order_count :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let total_comics := spiderman_books + archie_books + garfield_books
  let group_order_count := factorial spiderman_books
                           * factorial archie_books
                           * factorial garfield_books
                           * factorial 3
  in total_comics = 18 → group_order_count = 248832000 :=
by
  intros
  sorry

end comic_books_stack_order_count_l596_596311


namespace customers_left_tip_l596_596992

theorem customers_left_tip
  (morning_customers evening_customers : ℕ)
  (percent_no_tip : ℝ)
  (total_customers := morning_customers + evening_customers)
  (no_tip_customers := (percent_no_tip * total_customers).to_nat)
  : (morning_customers = 60) → 
    (evening_customers = 75) → 
    (percent_no_tip = 0.70) → 
    (total_customers = 135) → 
    ((total_customers - no_tip_customers) = 41) := 
by
  intros h1 h2 h3 h4
  -- Proof would go here
  sorry

end customers_left_tip_l596_596992


namespace length_of_AB_l596_596650

theorem length_of_AB 
  (A B C : Type)
  (area_ABC : Real := sqrt 3)
  (BC : Real := 2)
  (angle_C : Real := 60)
  (AB : Real) :
  sqrt 3 = 1/2 * BC * AB * Real.sin (angle_C * Real.pi / 180) ->
  AB = 2 :=
by
  sorry

end length_of_AB_l596_596650


namespace not_speaking_hindi_is_32_l596_596313

-- Definitions and conditions
def total_diplomats : ℕ := 120
def spoke_french : ℕ := 20
def percent_neither : ℝ := 0.20
def percent_both : ℝ := 0.10

-- Number of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi := (percent_neither * total_diplomats : ℝ)

-- Number of diplomats who spoke both French and Hindi
def both_french_and_hindi := (percent_both * total_diplomats : ℝ)

-- Number of diplomats who spoke only French
def only_french := (spoke_french - both_french_and_hindi : ℝ)

-- Number of diplomats who did not speak Hindi
def not_speaking_hindi := (only_french + neither_french_nor_hindi : ℝ)

theorem not_speaking_hindi_is_32 :
  not_speaking_hindi = 32 :=
by
  -- Provide proof here
  sorry

end not_speaking_hindi_is_32_l596_596313


namespace max_possible_cities_traversed_l596_596209

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596209


namespace vitya_catch_up_l596_596887

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596887


namespace tan_arccot_eq_5_div_3_l596_596031

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596031


namespace sum_of_roots_l596_596511

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l596_596511


namespace vectors_form_basis_l596_596464

-- Define the vectors in set B
def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (3, 7)

-- Define a function that checks if two vectors form a basis
def form_basis (v1 v2 : ℝ × ℝ) : Prop :=
  let det := v1.1 * v2.2 - v1.2 * v2.1
  det ≠ 0

-- State the theorem that vectors e1 and e2 form a basis
theorem vectors_form_basis : form_basis e1 e2 :=
by
  -- Add the proof here
  sorry

end vectors_form_basis_l596_596464


namespace calculate_savings_l596_596010

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l596_596010


namespace max_roads_city_condition_l596_596241

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596241


namespace smallest_possible_area_of_2020th_square_l596_596447

theorem smallest_possible_area_of_2020th_square :
  ∃ A : ℕ, (∃ n : ℕ, n * n = 2019 + A) ∧ A ≠ 1 ∧
  ∀ A' : ℕ, A' > 0 ∧ (∃ n : ℕ, n * n = 2019 + A') ∧ A' ≠ 1 → A ≤ A' :=
by
  sorry

end smallest_possible_area_of_2020th_square_l596_596447


namespace simplify_f_cos_2α_plus_π_over_4_l596_596580

noncomputable def f (α : ℝ) : ℝ :=
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (π / 2 + α)) / Real.cos (-α - π)

-- First part: Prove that f(α) = sin α
theorem simplify_f (α : ℝ) : f α = Real.sin α :=
  sorry

-- Second part: Given sin α = 4/5 and α is in the second quadrant
theorem cos_2α_plus_π_over_4 (α : ℝ)
  (h1 : Real.sin α = 4 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos (2 * α + π / 4) = (17 * Real.sqrt 2) / 50 :=
  sorry

end simplify_f_cos_2α_plus_π_over_4_l596_596580


namespace exponent_multiplication_l596_596396

variable x : ℝ

theorem exponent_multiplication : x^2 * x^3 = x^5 :=
by sorry

end exponent_multiplication_l596_596396


namespace car_mileage_proof_l596_596426

variable (distance : ℝ) (gallons : ℝ) (mileage : ℝ)

-- Given conditions
def car_distance := 210
def car_gallons := 5.25

-- Question: Prove that the car's mileage is 40 kilometers per gallon
theorem car_mileage_proof :
  ∀ (distance = car_distance) (gallons = car_gallons), mileage = distance / gallons → mileage = 40 := 
by
  sorry

end car_mileage_proof_l596_596426


namespace number_of_pipes_l596_596433

noncomputable def large_tank_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

noncomputable def small_pipe_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem number_of_pipes (d_L D_L d_S D_S : ℝ) (n : ℕ) :
  let r_L := d_L / 2,
      r_S := d_S / 2,
      V_L := large_tank_volume r_L D_L,
      V_S := small_pipe_volume r_S D_S
  in V_L = n * V_S → n = 108 :=
by
  intros
  sorry

end number_of_pipes_l596_596433


namespace alice_life_vests_needed_l596_596966

noncomputable def students_per_class : ℕ := 40
noncomputable def instructors_per_class : ℕ := 10
noncomputable def number_of_classes : ℕ := 4
noncomputable def student_life_vest_probability : ℝ := 0.40
noncomputable def instructor_life_vest_probability : ℝ := 0.70
noncomputable def life_vest_damage_probability : ℝ := 0.10

theorem alice_life_vests_needed :
  let total_students := students_per_class * number_of_classes
  let total_instructors := instructors_per_class * number_of_classes
  let total_participants := total_students + total_instructors
  let expected_students_with_vests := student_life_vest_probability * total_students
  let expected_instructors_with_vests := instructor_life_vest_probability * total_instructors
  let total_expected_with_vests := expected_students_with_vests + expected_instructors_with_vests
  let expected_lost_vests := life_vest_damage_probability * total_expected_with_vests
  let total_lost_vests := (expected_lost_vests).ceil.to_nat
in total_participants - total_expected_with_vests.to_nat - total_lost_vests = 98 :=
by
  sorry

end alice_life_vests_needed_l596_596966


namespace scuba_diver_time_l596_596957

theorem scuba_diver_time
  (rate_of_descent : ℕ)
  (total_depth : ℕ)
  (rate_eq : rate_of_descent = 80)
  (depth_eq : total_depth = 4000) :
  total_depth / rate_of_descent = 50 :=
by
  rw [depth_eq, rate_eq]
  norm_num

end scuba_diver_time_l596_596957


namespace find_s5_l596_596607

variables {a b x y : ℝ}

def s1 := a * x + b * y = 3
def s2 := a * x^2 + b * y^2 = 7
def s3 := a * x^3 + b * y^3 = 16
def s4 := a * x^4 + b * y^4 = 42

theorem find_s5 (h1 : s1) (h2 : s2) (h3 : s3) (h4 : s4) : a * x^5 + b * y^5 = 20 := 
sorry

end find_s5_l596_596607


namespace sum_neg_one_powers_l596_596895

theorem sum_neg_one_powers : ∑ k in Finset.range 2007, (-1) ^ (k + 1) = 1002 := by
  sorry

end sum_neg_one_powers_l596_596895


namespace sum_x_coordinates_Q3_l596_596421

theorem sum_x_coordinates_Q3 (Q1 Q2 Q3 : Type) [add_comm_group Q1]
  [add_comm_group Q2] [add_comm_group Q3] 
  (vertices_Q1 : list Q1)
  (vertices_Q2 : list Q2)
  (vertices_Q3 : list Q3)
  (sum_x_Q1 : Q1)
  (sum_x_Q2 : Q2)
  (sum_x_Q3 : Q3)
  (n : ℕ)
  (h1 : n = 44)
  (h2 : sum_x_Q1 = 132)
  (h3 : ∀ i ∈ list.range n, sum_x_Q2 = sum_x_Q1)
  (h4 : ∀ i ∈ list.range n, sum_x_Q3 = sum_x_Q2) :
  sum_x_Q3 = 132 :=
sorry

end sum_x_coordinates_Q3_l596_596421


namespace probability_of_black_ball_is_correct_l596_596191

def number_of_balls := 100
def number_of_red_balls := 45
def probability_of_white_ball := 0.23
def number_of_white_balls := probability_of_white_ball * number_of_balls
def number_of_black_balls := number_of_balls - (number_of_red_balls + number_of_white_balls)
def probability_of_black_ball := number_of_black_balls / number_of_balls

theorem probability_of_black_ball_is_correct :
  probability_of_black_ball = 0.32 :=
by
  sorry

end probability_of_black_ball_is_correct_l596_596191


namespace functions_equivalent_l596_596967

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := real.sqrt (x^2)

-- Proof of equivalence
theorem functions_equivalent : f = g :=
by
  sorry

end functions_equivalent_l596_596967


namespace complex_number_quadrant_l596_596331

theorem complex_number_quadrant :
  let z := (-1 + complex.I) / (1 + complex.I)
  Im z > 0 ∧ Re z < 0 → "second" := sorry

end complex_number_quadrant_l596_596331


namespace tan_arccot_l596_596025

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596025


namespace dividend_value_l596_596407

theorem dividend_value :
  ∀ (R D Q V : ℕ), 
  R = 6 →
  D = 5 * Q →
  D = 3 * R + 2 →
  V = D * Q + R →
  V = 86 := 
by
  intros R D Q V R_eq six R_eq divisor_eq V_eq  
  sorry

end dividend_value_l596_596407


namespace fraction_of_audience_for_second_band_l596_596970

-- Definitions of the conditions
variable (f : ℝ)
variable (totalAudience : ℝ) (condition1 : (0.4 * 0.5 * f * totalAudience = 20)) (condition2 : totalAudience = 150)

theorem fraction_of_audience_for_second_band 
  (h1 : condition1)
  (h2 : condition2) : 
  f = 2 / 3 :=
sorry

end fraction_of_audience_for_second_band_l596_596970


namespace degree_of_g_l596_596952

noncomputable theory
open Polynomial

def polynomial_degree_17 (p : Polynomial ℝ) : Prop := degree p = 17
def polynomial_quotient_degree_9 (s : Polynomial ℝ) : Prop := degree s = 9
def polynomial_remainder (r : Polynomial ℝ) : Prop := r = (5 * X^5 + 2 * X^3 - 3 * X + 7)

theorem degree_of_g (p g s r : Polynomial ℝ) 
  (hp : polynomial_degree_17 p)
  (hs : polynomial_quotient_degree_9 s)
  (hr : polynomial_remainder r)
  (h_div : p = g * s + r) : degree g = 8 :=
sorry

end degree_of_g_l596_596952


namespace tan_arccot_l596_596040

noncomputable def arccot (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem tan_arccot (x : ℝ) (h : x = 3/5) : tan (arccot x) = 5/3 :=
by
  have h1 : arccot x = arccot (3/5) := by rw [h]
  have h2 : arccot (3/5) = θ := sorry
  have h3 : tan θ = 5/3 := sorry
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end tan_arccot_l596_596040


namespace perfect_square_divisors_of_product_of_factorials_l596_596016

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : ℕ := (List.range (10 + 1)).map factorial |>.foldl (*) 1 * 144

def count_perfect_square_divisors (n : ℕ) : ℕ := sorry

theorem perfect_square_divisors_of_product_of_factorials :
  count_perfect_square_divisors product_of_factorials = 2640 := sorry

end perfect_square_divisors_of_product_of_factorials_l596_596016


namespace fish_ratio_bobby_sarah_l596_596472

-- Defining the conditions
variables (bobby sarah tony billy : ℕ)

-- Condition: Billy has 10 fish.
def billy_has_10_fish : billy = 10 := by sorry

-- Condition: Tony has 3 times as many fish as Billy.
def tony_has_3_times_billy : tony = 3 * billy := by sorry

-- Condition: Sarah has 5 more fish than Tony.
def sarah_has_5_more_than_tony : sarah = tony + 5 := by sorry

-- Condition: All 4 people have 145 fish together.
def total_fish : bobby + sarah + tony + billy = 145 := by sorry

-- The theorem we want to prove
theorem fish_ratio_bobby_sarah : (bobby : ℚ) / sarah = 2 / 1 := by
  -- You can write out the entire proof step by step here, but initially, we'll just put sorry.
  sorry

end fish_ratio_bobby_sarah_l596_596472


namespace abs_eq_two_iff_l596_596649

theorem abs_eq_two_iff (a : ℝ) : |a| = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_two_iff_l596_596649


namespace minimum_value_of_f_l596_596754

noncomputable def f (x y z : ℝ) : ℝ := (1 / (x + y)) + (1 / (x + z)) + (1 / (y + z)) - (x * y * z)

theorem minimum_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → f x y z = 1 / 2 :=
by
  sorry

end minimum_value_of_f_l596_596754


namespace playground_area_l596_596792

theorem playground_area (L B : ℕ) (h1 : B = 6 * L) (h2 : B = 420)
  (A_total A_playground : ℕ) (h3 : A_total = L * B) 
  (h4 : A_playground = A_total / 7) :
  A_playground = 4200 :=
by sorry

end playground_area_l596_596792


namespace calculate_savings_l596_596008

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l596_596008


namespace vitya_catches_up_in_5_minutes_l596_596848

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596848


namespace domain_of_function_l596_596801

theorem domain_of_function (x : ℝ) (k : ℤ) : 
  2 * sin x - 1 ≥ 0 → x ∈ Set.Icc (2 * k * Real.pi + Real.pi / 6) (2 * k * Real.pi + 5 * Real.pi / 6) :=
by
  intro h
  sorry

end domain_of_function_l596_596801


namespace BoatCrafters_total_canoes_l596_596974

def canoe_production (n : ℕ) : ℕ :=
  if n = 0 then 5 else 3 * canoe_production (n-1) - 1

theorem BoatCrafters_total_canoes : 
  (canoe_production 0 - 1) + (canoe_production 1 - 1) + (canoe_production 2 - 1) + (canoe_production 3 - 1) = 196 := 
by
  sorry

end BoatCrafters_total_canoes_l596_596974


namespace sum_of_roots_l596_596512

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l596_596512


namespace point_below_line_l596_596612

theorem point_below_line (m : ℝ) : 
  ((-∞ < m ∧ m < -3) ∨ (0 < m ∧ m < ∞)) ↔ (-2 + m * (-1) - 1 < 0) :=
sorry

end point_below_line_l596_596612


namespace problem_proof_l596_596176

noncomputable section

variables {x : ℝ} (hx : x ∈ set.Ioi (Real.exp 1 - 1) ∩ set.Iio 1)
def a : ℝ := Real.log x
def b : ℝ := 2 * Real.log x
def c : ℝ := Real.log (3 * x)

theorem problem_proof : (b < a) ∧ (a < c) :=
by
  sorry

end problem_proof_l596_596176


namespace symmetric_coordinates_l596_596332

structure Point :=
  (x : Int)
  (y : Int)

def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_coordinates (P : Point) (h : P = Point.mk (-1) 2) :
  symmetric_about_origin P = Point.mk 1 (-2) :=
by
  sorry

end symmetric_coordinates_l596_596332


namespace exactly_one_line_through_two_points_l596_596372

-- Assume the existence of a point and line
variables {Point : Type} [h : nonempty Point]
include h

-- Assume the given condition: the axiom of lines
axiom exists_unique_line_through_two_points
  (P Q : Point) (h : P ≠ Q) : ∃! (l : Set Point), ∀ (x : Point), x ∈ l ↔ (x = P ∨ x = Q)

theorem exactly_one_line_through_two_points (P Q : Point) (h : P ≠ Q) : 
  ∃! (l : Set Point), ∀ (x : Point), x ∈ l ↔ (x = P ∨ x = Q) :=
exists_unique_line_through_two_points P Q h

end exactly_one_line_through_two_points_l596_596372


namespace johns_weekly_earnings_percentage_increase_l596_596729

theorem johns_weekly_earnings_percentage_increase (initial final : ℝ) :
  initial = 30 →
  final = 50 →
  ((final - initial) / initial) * 100 = 66.67 :=
by
  intros h_initial h_final
  rw [h_initial, h_final]
  norm_num
  sorry

end johns_weekly_earnings_percentage_increase_l596_596729


namespace simplify_f_value_of_f_at_specific_alpha_l596_596579

noncomputable def f (α : ℝ) : ℝ :=
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (2 * sin (3 * π + α) * sin (-π - α) * sin (9 * π / 2 + α))

theorem simplify_f (α : ℝ) : f α = - (1 / 2) * sin α := by
  sorry

theorem value_of_f_at_specific_alpha : f (-25 * π / 4) = sqrt 2 / 4 := by
  sorry

end simplify_f_value_of_f_at_specific_alpha_l596_596579


namespace max_possible_intersections_l596_596379

theorem max_possible_intersections : 
  let num_x := 12
  let num_y := 6
  let intersections := (num_x * (num_x - 1) / 2) * (num_y * (num_y - 1) / 2)
  intersections = 990 := 
by 
  sorry

end max_possible_intersections_l596_596379


namespace centroid_and_area_ratio_l596_596702

theorem centroid_and_area_ratio 
  (A B C M A_1 B_1 C_1 : Point)
  (h1 : ∀ (P : Point), (IsPerpendicular (line A M) (line A P)) ∧ (dist M A_1 = dist A B) ∨ (IsPerpendicular (line B M) (line B P)) ∧ (dist M B_1 = dist B C) ∨ (IsPerpendicular (line C M) (line C P)) ∧ (dist M C_1 = dist C A)) :
  (Centroid M A_1 B_1 C_1) ∧ (area_ratio (Triangle.mk A B C) (Triangle.mk A_1 B_1 C_1) = 1/3) :=
by
  sorry

end centroid_and_area_ratio_l596_596702


namespace minimum_disks_needed_to_store_files_l596_596760

-- Definitions of the conditions
def totalFiles : Nat := 35
def diskCapacity : Float := 1.6
def filesSize0_9 : List Float := List.replicate 5 0.9
def filesSize0_8 : List Float := List.replicate 10 0.8
def remainingFilesSize : List Float := List.replicate (totalFiles - filesSize0_9.length - filesSize0_8.length) 0.5

-- Sum of file sizes
def totalSize (files : List Float) : Float := files.foldl (· + ·) 0

noncomputable def totalFileSize : Float := totalSize (filesSize0_9 ++ filesSize0_8 ++ remainingFilesSize)

-- Let's state the problem
theorem minimum_disks_needed_to_store_files : Nat :=
  let disks_needed : Nat := Nat.ceil (totalFileSize / diskCapacity)
  have correct_disks : disks_needed = 15 := sorry
  disks_needed

#eval minimum_disks_needed_to_store_files -- This will evaluate to 15 which is the correct answer

end minimum_disks_needed_to_store_files_l596_596760


namespace ceil_floor_difference_l596_596069

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596069


namespace min_elements_with_properties_l596_596670

noncomputable def median (s : List ℝ) : ℝ :=
if h : s.length % 2 = 1 then
  s.nthLe (s.length / 2) (by simp [h])
else
  (s.nthLe (s.length / 2 - 1) (by simp [Nat.sub_right_comm, h, *]) + s.nthLe (s.length / 2) (by simp [h])) / 2

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

noncomputable def mode (s : List ℝ) : ℝ :=
s.groupBy (· = ·)
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (·.snd |>.value)

theorem min_elements_with_properties :
  ∃ s : List ℝ, 
    s.length ≥ 6 ∧ 
    median s = 3 ∧ 
    mean s = 5 ∧ 
    ∃! m, mode s = m ∧ m = 6 :=
by
  sorry

end min_elements_with_properties_l596_596670


namespace max_roads_city_condition_l596_596234

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596234


namespace a5_equals_2_l596_596689

variable {a : ℕ → ℝ}  -- a_n represents the nth term of the arithmetic sequence

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a 1 + n * d 

-- Given condition
axiom arithmetic_condition (h : is_arithmetic_sequence a) : a 1 + a 5 + a 9 = 6

-- The goal is to prove a_5 = 2
theorem a5_equals_2 (h : is_arithmetic_sequence a) (h_cond : a 1 + a 5 + a 9 = 6) : a 5 = 2 := 
by 
  sorry

end a5_equals_2_l596_596689


namespace chord_length_l596_596329

-- Define the line equation
def line (x y : ℝ) : Prop := 3 * x - 4 * y = 9

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Define the proof problem to determine the chord length
theorem chord_length : ∀ (x y : ℝ), circle x y → line x y → 2 * 3 = 6 :=
by sorry

end chord_length_l596_596329


namespace sum_of_roots_l596_596508

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l596_596508


namespace remaining_cubes_l596_596052

theorem remaining_cubes (n : ℕ) (h : n = 12) : 
  let full_cuboid := n * n * n,
      inner_cuboid := (n - 2) * (n - 2) * (n - 2),
      hollow_shell_cuboid := (n - 4) * (n - 4) * (n - 4) in
  full_cuboid - (inner_cuboid - hollow_shell_cuboid) = 488 :=
by {
  have full_cuboid_eq : full_cuboid = 12 * 12 * 12, from rfl,
  have inner_cuboid_eq : inner_cuboid = 10 * 10 * 10, from rfl,
  have hollow_shell_cuboid_eq : hollow_shell_cuboid = 8 * 8 * 8, from rfl,
  rw [full_cuboid_eq, mul_left_comm 12, ←pow_succ', pow_two, pow_two, add_pow_two, add_pow_succ],
  rw [inner_cuboid_eq, mul_left_comm 10, ←pow_succ', pow_two, pow_two, add_pow_two, add_pow_succ],
  rw [hollow_shell_cuboid_eq, mul_left_comm 8, ←pow_succ', pow_two, pow_two, add_pow_two, add_pow_succ],
  norm_num,
}

end remaining_cubes_l596_596052


namespace vitya_catchup_time_l596_596857

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596857


namespace line_equation_l596_596802

theorem line_equation (m b : ℝ) (h_slope : m = 3) (h_intercept : b = 4) :
  3 * x - y + 4 = 0 :=
by
  sorry

end line_equation_l596_596802


namespace num_divisors_8_factorial_l596_596547

-- Defining the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Defining the function to count the number of positive divisors using the prime factorization
noncomputable def num_divisors (n : ℕ) : ℕ :=
  let factors := nat.factorization n
  factors.fold (1 : ℕ) (λ p e acc, acc * (e + 1))

-- The problem statement
theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 :=
  sorry

end num_divisors_8_factorial_l596_596547


namespace girls_more_than_boys_l596_596190

-- Given conditions
def ratio_boys_girls : ℕ := 3
def ratio_girls_boys : ℕ := 4
def total_students : ℕ := 42

-- Theorem statement
theorem girls_more_than_boys : 
  let x := total_students / (ratio_boys_girls + ratio_girls_boys)
  let boys := ratio_boys_girls * x
  let girls := ratio_girls_boys * x
  girls - boys = 6 := by
  sorry

end girls_more_than_boys_l596_596190


namespace take_home_pay_is_correct_l596_596717

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l596_596717


namespace tan_arccot_of_frac_l596_596036

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596036


namespace find_escalator_rate_l596_596001

-- Given conditions
def escalator_length : ℕ := 140 -- length of the escalator in feet
def walking_speed : ℕ := 3 -- walking speed of the person in feet per second
def time_to_cover : ℕ := 10 -- time taken to cover the entire length in seconds

-- The rate at which the escalator moves
def escalator_rate : ℕ :=
  let effective_speed := walking_speed + escalator_rate
  let total_distance := effective_speed * time_to_cover
  if total_distance = escalator_length then
    escalator_rate
  else
    sorry

-- Proof statement
theorem find_escalator_rate : escalator_rate = 11 := by
  sorry

end find_escalator_rate_l596_596001


namespace binomial_square_b_value_l596_596646

theorem binomial_square_b_value (b : ℝ) (h : ∃ c : ℝ, (9 * x^2 + 24 * x + b) = (3 * x + c) ^ 2) : b = 16 :=
sorry

end binomial_square_b_value_l596_596646


namespace sum_of_roots_l596_596519

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l596_596519


namespace conjugate_of_z_l596_596696

def z : ℂ := complex.mk (-1) (real.sqrt 3)

theorem conjugate_of_z :
  complex.conj z = complex.mk (-1) (-real.sqrt 3) :=
by
  -- Proof skipped
  sorry

end conjugate_of_z_l596_596696


namespace minimum_value_of_z_l596_596758

theorem minimum_value_of_z
  (x y : ℝ)
  (h1 : 3 * x + y - 6 ≥ 0)
  (h2 : x - y - 2 ≤ 0)
  (h3 : y - 3 ≤ 0) :
  ∃ z, z = 4 * x + y ∧ z = 7 :=
sorry

end minimum_value_of_z_l596_596758


namespace M_greater_than_N_l596_596301

-- Definitions based on the problem's conditions
def M (x : ℝ) : ℝ := (x - 3) * (x - 7)
def N (x : ℝ) : ℝ := (x - 2) * (x - 8)

-- Statement to prove
theorem M_greater_than_N (x : ℝ) : M x > N x := by
  -- Proof is omitted
  sorry

end M_greater_than_N_l596_596301


namespace BKINGTON_appears_first_on_eighth_line_l596_596345

-- Define the cycle lengths for letters and digits
def cycle_letters : ℕ := 8
def cycle_digits : ℕ := 4

-- Define the problem statement
theorem BKINGTON_appears_first_on_eighth_line :
  Nat.lcm cycle_letters cycle_digits = 8 := by
  sorry

end BKINGTON_appears_first_on_eighth_line_l596_596345


namespace probability_of_U_l596_596386

def pinyin : List Char := ['S', 'H', 'U', 'X', 'U', 'E']
def total_letters : Nat := 6
def u_count : Nat := 2

theorem probability_of_U :
  ((u_count : ℚ) / (total_letters : ℚ)) = (1 / 3) :=
by
  sorry

end probability_of_U_l596_596386


namespace halfthink_set_last_three_digits_l596_596119

theorem halfthink_set_last_three_digits :
  let S := {1, 2, 3, ..., 1984},
      sum_S := 1984 * 1985 / 2,
      half_sum_S := 985360 in
  ∀ (A : Set ℕ), (A ⊆ S ∧ A.card = 992 ∧ A.sum = half_sum_S ∧ ∃! n m, n ∈ A ∧ m ∈ A ∧ n + 1 = m) →
  (∀ n, n ∉ A ∨ (n + 1) ∉ A) →
  (let nk := [989, 991, 993, 995],
       product_n := nk.foldl (*) 1 in
   product_n % 1000 = 465) := by sorry

end halfthink_set_last_three_digits_l596_596119


namespace sum_of_roots_l596_596517

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l596_596517


namespace vitya_catchup_time_l596_596858

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596858


namespace sum_odd_multiples_of_five_between_200_and_800_l596_596388

theorem sum_odd_multiples_of_five_between_200_and_800 :
  (∑ k in finset.filter (λ k, k % 5 = 0 ∧ k % 2 = 1) (finset.Icc 200 800), k) = 30000 := 
sorry

end sum_odd_multiples_of_five_between_200_and_800_l596_596388


namespace max_possible_N_in_cities_l596_596247

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596247


namespace vitya_catchup_time_l596_596880

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596880


namespace triangle_side_length_l596_596656

theorem triangle_side_length
  (a : ℝ) (A B : ℝ) (A_eq_60 : A = 60) (B_eq_45 : B = 45) (a_eq_4 : a = 4) :
  ∃ b : ℝ, b = 4 * sqrt 6 / 3 :=
by
  sorry

end triangle_side_length_l596_596656


namespace max_possible_value_l596_596250

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l596_596250


namespace vitya_catch_up_time_l596_596866

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596866


namespace only_one_fruit_remains_last_remaining_fruit_is_banana_not_possible_zero_fruits_l596_596315

section MagicalAppleTree

variables (B O : ℕ) -- B symbolizes bananas, O symbolizes oranges

-- Initial state conditions
def initial_bananas := 15
def initial_oranges := 20

-- Picking rules
def pick_one (n : ℕ) := n + 1
def pick_two_same_bananas (B : ℕ) (O : ℕ) := (B - 2, O + 1)
def pick_two_same_oranges (B : ℕ) (O : ℕ) := (B, O)
def pick_two_different (B : ℕ) (O : ℕ) := (B + 1, O - 1)

-- Conditions for part (a)
theorem only_one_fruit_remains :
  ∃ (f : ℕ), ∀ (B O : ℕ), B = 15 → O = 20 →
    (λ (steps : List (ℕ × ℕ)),
      steps.foldl (λ st step,
        if step.1 = 1 then (pick_one st.1, pick_one st.2)
        else if step.1 = 2 then pick_two_same_bananas st.1 st.2
        else pick_two_different st.1 st.2) (B, O) = (f, 0)) sorry

-- Conditions for part (b)
theorem last_remaining_fruit_is_banana :
  ∀ (B O : ℕ), B = 15 → O = 20 →
    (∃ f, (λ (steps : List (ℕ × ℕ)),
      steps.foldl (λ st step,
        if step.1 = 1 then (pick_one st.1, pick_one st.2)
        else if step.1 = 2 then pick_two_same_bananas st.1 st.2
        else pick_two_different st.1 st.2) (B, O) = (f, 0)) → f = 1) sorry

-- Conditions for part (c)
theorem not_possible_zero_fruits :
  ∀ (B O : ℕ), B = 15 → O = 20 → ¬(∃ (steps : List (ℕ × ℕ)),
    (steps.foldl (λ st step,
        if step.1 = 1 then (pick_one st.1, pick_one st.2)
        else if step.1 = 2 then pick_two_same_bananas st.1 st.2
        else pick_two_different st.1 st.2) (B, O) = (0, 0))) sorry

end MagicalAppleTree

end only_one_fruit_remains_last_remaining_fruit_is_banana_not_possible_zero_fruits_l596_596315


namespace binomial_n_choose_n_sub_2_l596_596899

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end binomial_n_choose_n_sub_2_l596_596899


namespace max_roads_city_condition_l596_596240

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596240


namespace maximum_N_value_l596_596233

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596233


namespace calculate_savings_l596_596009

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l596_596009


namespace line_eq_passes_through_M_and_center_l596_596144

noncomputable theory

-- Definitions of the given conditions
def point (x y : ℝ) : Type := {x := x, y := y}

def line (a b c : ℝ) : Type := {a := a, b := b, c := c}

def circle (h k r : ℝ) : Prop := ∀ (x y : ℝ), (x - h) ^ 2 + (y + k) ^ 2 = r ^ 2

-- Given conditions
def M := point 2 3

def circle_eq := circle 2 (-3) 9

-- Prove that the equation of line l is x = 2
theorem line_eq_passes_through_M_and_center :
  (∃ l : line, l.a = 1 ∧ l.b = 0 ∧ l.c = -2) :=
begin
  sorry
end

end line_eq_passes_through_M_and_center_l596_596144


namespace ellipse_standard_equation_line_passes_fixed_point_l596_596134

theorem ellipse_standard_equation : 
  ∃ (a b : ℝ),
  (a > 0) ∧ (b > 0) ∧ (∀ (x y : ℝ), 
  (x, y) = (1, 3/2) → (x^2 / a^2 + y^2 / b^2 = 1)) ∧ ((1 - (focus x)^2/a^2 = (1/2)^2)) :=
sorry

theorem line_passes_fixed_point :
  ∀ (k m : ℝ) (x1 y1 x2 y2 : ℝ),
  (x1^2 / 4 + y1^2 / 3 = 1) ∧
  (x2^2 / 4 + y2^2 / 3 = 1) ∧
  (y1 = k * x1 + m) ∧ 
  (y2 = k * x2 + m) ∧
  (circle_with_diameter_AB_passes_through_right_vertex x1 y1 x2 y2) 
  → 
  (∀ k, m = -2*k ∨ m = -2*k/7 → (line l passing through AB passes through (2/7, 0))) :=
sorry

end ellipse_standard_equation_line_passes_fixed_point_l596_596134


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l596_596825

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l596_596825


namespace grocery_shop_sale_l596_596432

theorem grocery_shop_sale 
  (sale2 sale3 sale4 sale5 sale6 average total_sale: ℝ)
  (h1 : sale2 = 6927) 
  (h2 : sale3 = 6855) 
  (h3 : sale4 = 7230) 
  (h4 : sale5 = 6562) 
  (h5 : sale6 = 4891) 
  (h_avg : average = 6500)
  (h_total : total_sale = 6 * average): 
  ∃ sale1, sale1 = total_sale - (sale2 + sale3 + sale4 + sale5 + sale6) := 
by 
  let sale1 := total_sale - (sale2 + sale3 + sale4 + sale5 + sale6)
  use sale1
  sorry

end grocery_shop_sale_l596_596432


namespace teagan_saved_200_pennies_l596_596785

-- Given definitions
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330
def total_savings : ℝ := 40

-- The value of one nickel in dollars
def nickel_value : ℝ := 0.05

-- The value of one dime in dollars
def dime_value : ℝ := 0.10

-- The value of one penny in dollars
def penny_value : ℝ := 0.01

-- Total amount saved by Rex
def rex_savings : ℝ := rex_nickels * nickel_value

-- Total amount saved by Toni
def toni_savings : ℝ := toni_dimes * dime_value

-- Combined savings from Rex and Toni
def combined_savings : ℝ := rex_savings + toni_savings

-- Total savings the kids have minus Rex's and Toni's savings gives Teagan's savings
def teagan_savings : ℝ := total_savings - combined_savings

-- Convert Teagan's savings to the number of pennies
def teagan_pennies : ℕ := (teagan_savings / penny_value).to_nat

-- Theorem stating that Teagan saved 200 pennies
theorem teagan_saved_200_pennies : teagan_pennies = 200 := by
  sorry

end teagan_saved_200_pennies_l596_596785


namespace probability_three_out_of_four_odd_dice_l596_596003

theorem probability_three_out_of_four_odd_dice :
  let p_odd := (4 / 8 : ℚ) in
  let p_even := 1 - p_odd in
  let ways := Nat.choose 4 3 in
  (ways * (p_odd^3) * (p_even^1) = (1 / 4 : ℚ)) :=
by
  sorry

end probability_three_out_of_four_odd_dice_l596_596003


namespace vitya_catch_up_time_l596_596864

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596864


namespace binom_30_3_squared_l596_596977

-- Define the binomial coefficient binom function
noncomputable def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Define the specific case of binom(30, 3)
def binom_30_3 := binom 30 3

theorem binom_30_3_squared :
  (binom_30_3) ^ 2 = 16483600 :=
by
  sorry

end binom_30_3_squared_l596_596977


namespace area_of_triangle_ABC_l596_596562

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end area_of_triangle_ABC_l596_596562


namespace arithmetic_sequence_x_values_l596_596097

theorem arithmetic_sequence_x_values {x : ℝ} (h_nonzero : x ≠ 0) (h_arith_seq : ∃ (k : ℤ), x - k = 1/2 ∧ x + 1 - (k + 1) = (k + 1) - 1/2) (h_lt_four : x < 4) :
  x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5 :=
by
  sorry

end arithmetic_sequence_x_values_l596_596097


namespace rectangle_area_l596_596954

/-- Define a rectangle with its length being three times its breadth, and given diagonal length d = 20.
    Prove that the area of the rectangle is 120 square meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) (d : ℝ) (h1 : l = 3 * b) (h2 : d = 20) (h3 : l^2 + b^2 = d^2) : l * b = 120 :=
by
  sorry

end rectangle_area_l596_596954


namespace prob1_prob2_prob3_l596_596161

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2*x + 3
noncomputable def g (x : ℝ) : ℝ := f (Real.sin x)

theorem prob1 : ∀ x : ℝ, f x = -x^2 + 2*x + 3 :=
by sorry

theorem prob2 : { x : ℝ | f x ≤ 3 } = { x : ℝ | x ≤ 0 ∨ x ≥ 2 } :=
by sorry

theorem prob3 : set.range g = set.Icc 0 4 :=
by sorry

end prob1_prob2_prob3_l596_596161


namespace max_value_of_N_l596_596195

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596195


namespace ceil_floor_diff_l596_596075

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596075


namespace tan_arccot_eq_5_div_3_l596_596030

theorem tan_arccot_eq_5_div_3 : tan (arccot (3 / 5)) = 5 / 3 :=
sorry

end tan_arccot_eq_5_div_3_l596_596030


namespace members_playing_both_sports_l596_596409

theorem members_playing_both_sports 
    (N : ℕ) (B : ℕ) (T : ℕ) (D : ℕ)
    (hN : N = 30) (hB : B = 18) (hT : T = 19) (hD : D = 2) :
    N - D = 28 ∧ B + T = 37 ∧ B + T - (N - D) = 9 :=
by
  sorry

end members_playing_both_sports_l596_596409


namespace f_5m_eq_neg_2_over_5_l596_596986

-- Define the function f with the specified properties
noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then x - m
else if 0 ≤ x ∧ x < 1 then |x - (2/5)|
else f (x - 2) m

-- State the theorem to be proven
theorem f_5m_eq_neg_2_over_5 {m : ℝ} (h1 : ∀ x, f (x + 2) m = f x m)
  (h2 : f (-5/2) m = f (9/2) m) : f (5 * m) m = -2/5 :=
sorry

end f_5m_eq_neg_2_over_5_l596_596986


namespace problem_1_problem_2_l596_596417

-- First Proof Problem
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x^2 + 1) : 
  f x = 2 * x^2 - 4 * x + 3 :=
sorry

-- Second Proof Problem
theorem problem_2 {a b : ℝ} (f : ℝ → ℝ) (hf : ∀ x, f x = x / (a * x + b))
  (h1 : f 2 = 1) (h2 : ∃! x, f x = x) : 
  f x = 2 * x / (x + 2) :=
sorry

end problem_1_problem_2_l596_596417


namespace original_number_of_players_l596_596832

theorem original_number_of_players 
    (n : ℕ) (W : ℕ)
    (h1 : W = n * 112)
    (h2 : W + 110 + 60 = (n + 2) * 106) : 
    n = 7 :=
by
  sorry

end original_number_of_players_l596_596832


namespace max_value_of_N_l596_596200

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596200


namespace circle_equation_l596_596125

-- Conditions definitions
def circle (x y : ℝ) (r : ℝ) := ∀ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = r^2

def tangent_to_y_axis (x : ℝ) := abs x = 4

def center_line (x y : ℝ) := x - 3 * y = 0

-- Theorem statement
theorem circle_equation (x y : ℝ) (r : ℝ) (hx : tangent_to_y_axis x) (hy : center_line x y) (hr : r = 4) :
  circle x y 4 = (λ p : ℝ × ℝ, (p.1 - 4)^2 + (p.2 - (4/3))^2 = 16) ∨ 
  circle x y 4 = (λ p : ℝ × ℝ, (p.1 + 4)^2 + (p.2 + (4/3))^2 = 16) := 
sorry

end circle_equation_l596_596125


namespace seq_properties_l596_596131

-- Conditions for the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * a n + 1

-- The statements to prove given the sequence definition
theorem seq_properties (a : ℕ → ℝ) (h : seq a) :
  (∀ n, a (n + 1) ≥ 2 * a n) ∧
  (∀ n, a (n + 1) / a n ≥ a n) ∧
  (∀ n, a n ≥ n * n - 2 * n + 2) :=
by
  sorry

end seq_properties_l596_596131


namespace hyperbola_E_equation_hyperbola_C_equation_l596_596002

-- Proof for Problem 1
theorem hyperbola_E_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : (∀ x y : ℝ, ((x, y) = (2, sqrt 2) → (x^2 / a^2 - y^2 / b^2 = 1)))
  (h4 : sqrt 2 = sqrt 2) :
  (a = sqrt 2) → (b = sqrt 2) → (a^2 = 2) → (b^2 = 2) → (∀ x y : ℝ, x^2 / 2 - y^2 / 2 = 1) :=
sorry

-- Proof for Problem 2
theorem hyperbola_C_equation (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a = 1) (h5 : (2a = 2 → a = 1)) 
  (h6 : (c = 2)) 
  (h7 : c^2 = a^2 + b^2) :
  (4 = 1 + b^2) → (b^2 = 3) → (a = 1) → (∀ x y : ℝ, x^2 - y^2 / 3 = 1) :=
sorry

end hyperbola_E_equation_hyperbola_C_equation_l596_596002


namespace closest_log_7_a2019_l596_596987

-- Define necessary operations and functions
def bigcirc (a b : ℝ) : ℝ := a ^ (Real.log b / Real.log 7)
def closedown b : ℝ := Real.log b / Real.log 7 -- note Real.log b is log base e (ln)
def tensor (a b : ℝ) : ℝ := a ^ (1 / (Real.log b / Real.log 7)^6)
def seq (a n : ℕ) (h : n ≥ 4) : ℝ :=
  if n = 3 then tensor 3 2 else bigcirc (tensor (n:ℝ) (n-1:ℝ)) (seq a (n-1) sorry)

-- Define the actual problem statement
theorem closest_log_7_a2019 : 
    let a2019 := seq 0 2019 sorry
    in abs (Real.log 2019 / Real.log 2 - 11) < 1 := sorry

end closest_log_7_a2019_l596_596987


namespace vitya_catchup_time_l596_596856

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596856


namespace seating_arrangement_l596_596169

theorem seating_arrangement (desks : ℕ) (students : ℕ) 
  (condition : desks = 6 ∧ students = 2) :
  (count_ways (desks, students) = 9) := sorry

end seating_arrangement_l596_596169


namespace max_value_of_vector_dot_product_l596_596137

theorem max_value_of_vector_dot_product :
  ∀ (x y : ℝ), (-2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2) → (2 * x - y ≤ 4) :=
by
  intros x y h
  sorry

end max_value_of_vector_dot_product_l596_596137


namespace product_of_roots_Q_l596_596742

noncomputable def Q (x : ℝ) : ℝ := (x - 5)^4 - 5

theorem product_of_roots_Q :
  (∃ x : ℝ, Q x = 0 ∧ (∃ u : ℝ, u^4 = 5 ∧ x = u + 5)) →
  (∑ (r : ℝ) in (Roots (Q (x))), r) = (620) :=
by
  sorry

end product_of_roots_Q_l596_596742


namespace min_elements_with_properties_l596_596671

noncomputable def median (s : List ℝ) : ℝ :=
if h : s.length % 2 = 1 then
  s.nthLe (s.length / 2) (by simp [h])
else
  (s.nthLe (s.length / 2 - 1) (by simp [Nat.sub_right_comm, h, *]) + s.nthLe (s.length / 2) (by simp [h])) / 2

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

noncomputable def mode (s : List ℝ) : ℝ :=
s.groupBy (· = ·)
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (·.snd |>.value)

theorem min_elements_with_properties :
  ∃ s : List ℝ, 
    s.length ≥ 6 ∧ 
    median s = 3 ∧ 
    mean s = 5 ∧ 
    ∃! m, mode s = m ∧ m = 6 :=
by
  sorry

end min_elements_with_properties_l596_596671


namespace ellipse_eq_1_ellipse_eq_2a_ellipse_eq_2b_l596_596419

-- Problem 1
theorem ellipse_eq_1 (x y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (e : ℝ) :
  a = 6 → 
  e = 1 / 3 → 
  c = 2 → 
  b = sqrt (a ^ 2 - c ^ 2) → 
  (x^2 / a^2) + (y^2 / b^2) = 1 ↔ 
  (x^2 / 36) + (y^2 / 32) = 1 :=
by
  intros
  sorry

-- Problem 2 (foci on x-axis)
theorem ellipse_eq_2a (x y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :
  2 * a = 10 → 
  2 * c = 6 → 
  a = 5 → 
  c = 3 → 
  b = sqrt (a ^ 2 - c ^ 2) → 
  (x^2 / a^2) + (y^2 / b^2) = 1 ↔ 
  (x^2 / 25) + (y^2 / 16) = 1 :=
by
  intros
  sorry

-- Problem 2 (foci on y-axis)
theorem ellipse_eq_2b (x y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :
  2 * a = 10 → 
  2 * c = 6 → 
  a = 5 → 
  c = 3 → 
  b = sqrt (a ^ 2 - c ^ 2) → 
  (y^2 / a^2) + (x^2 / b^2) = 1 ↔ 
  (x^2 / 16) + (y^2 / 25) = 1 :=
by
  intros
  sorry

end ellipse_eq_1_ellipse_eq_2a_ellipse_eq_2b_l596_596419


namespace length_of_AD_l596_596687

theorem length_of_AD (AB BC AC AD DC : ℝ)
    (h1 : AB = BC)
    (h2 : AD = 2 * DC)
    (h3 : AC = AD + DC)
    (h4 : AC = 27) : AD = 18 := 
by
  sorry

end length_of_AD_l596_596687


namespace vitya_catch_up_l596_596881

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596881


namespace area_triangle_ABF_area_triangle_ADF_l596_596789

-- Define the square with area 300 cm^2
noncomputable def square_area : ℝ := 300

-- Define points and conditions
structure Point (α : Type) := (x : α) (y : α)
def A := Point ℝ
def B := Point ℝ
def C := Point ℝ
def D := Point ℝ
def M := Point ℝ
def F := Point ℝ

-- Define the conditions
axiom condition1 : (A, B, C, D : Point ℝ)
axiom condition2 : (M.x = (C.x + D.x) / 2) ∧ (M.y = (C.y + D.y) / 2)
axiom condition3 : ∃ (k : ℝ), F.x = B.x + k * (C.x - B.x) ∧ F.y = B.y + k * (C.y - B.y)

-- Prove the area of triangles
theorem area_triangle_ABF : ∃ (ABF_area : ℝ), ABF_area = 300 := by
  sorry

theorem area_triangle_ADF : ∃ (ADF_area : ℝ), ADF_area = 150 := by
  sorry

end area_triangle_ABF_area_triangle_ADF_l596_596789


namespace pumpkin_pie_degrees_l596_596312

theorem pumpkin_pie_degrees (total_students : ℕ) (peach_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ)
                               (pumpkin_pie : ℕ) (banana_pie : ℕ)
                               (h_total : total_students = 40)
                               (h_peach : peach_pie = 14)
                               (h_apple : apple_pie = 9)
                               (h_blueberry : blueberry_pie = 7)
                               (h_remaining : pumpkin_pie = banana_pie)
                               (h_half_remaining : 2 * pumpkin_pie = 40 - (peach_pie + apple_pie + blueberry_pie)) :
  (pumpkin_pie * 360) / total_students = 45 := by
sorry

end pumpkin_pie_degrees_l596_596312


namespace edward_boxes_l596_596929

theorem edward_boxes (initial_games sold_games games_per_box : ℕ) (h_initial : initial_games = 35) (h_sold : sold_games = 19) (h_games_per_box : games_per_box = 8) :
  (initial_games - sold_games) / games_per_box = 2 :=
by {
  rw [h_initial, h_sold, h_games_per_box],
  norm_num,
  sorry
}

end edward_boxes_l596_596929


namespace set_equality_l596_596184

def U := {1, 2, 3, 4, 5, 6}
def M := {1, 4}
def N := {2, 3}
def C_U (S : Set ℕ) : Set ℕ := U \ S

theorem set_equality :
  {5, 6} = (C_U M) ∩ (C_U N) := by
  sorry

end set_equality_l596_596184


namespace ratio_of_boys_to_girls_l596_596406

theorem ratio_of_boys_to_girls {T G B : ℕ} (h1 : (2/3 : ℚ) * G = (1/4 : ℚ) * T) (h2 : T = G + B) : (B : ℚ) / G = 5 / 3 :=
by
  sorry

end ratio_of_boys_to_girls_l596_596406


namespace sin_double_alpha_l596_596171

theorem sin_double_alpha (α : ℝ) (h : cos (α - π / 4) = sqrt 3 / 3) : sin (2 * α) = -1 / 3 :=
sorry

end sin_double_alpha_l596_596171


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l596_596826

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l596_596826


namespace position_of_8_over_9_in_sequence_l596_596979

theorem position_of_8_over_9_in_sequence :
  let position_sum k = (k * (k + 1)) / 2 in
  let total_terms_up_to_n n = position_sum n in
  let numerator := 8 in
  let denominator := 9 in
  let target_sum := numerator + denominator in
  let position_in_target_sum := numerator in
  total_terms_up_to_n (target_sum - 2) + position_in_target_sum = 128 :=
by
  sorry

end position_of_8_over_9_in_sequence_l596_596979


namespace smallest_set_size_l596_596677

noncomputable def smallest_num_elements (s : Multiset ℝ) : ℕ :=
  s.length

theorem smallest_set_size (s : Multiset ℝ) :
  (∀ a b c : ℝ, s = {a, b, 3, 6, 6, c}) →
  (s.median = 3) →
  (s.mean = 5) →
  (∀ x, s.count x < 3 → x ≠ 6) →
  smallest_num_elements s = 6 :=
by
  intros _ _ _ _
  sorry

end smallest_set_size_l596_596677


namespace zero_points_l596_596150

def f1 (x : ℝ) : ℝ := log 4 x - (1 / 4) ^ x
def f2 (x : ℝ) : ℝ := log (1 / 4) x - (1 / 4) ^ x

theorem zero_points (x1 x2 : ℝ)
    (hx1 : f1 x1 = 0)
    (hx2 : f2 x2 = 0)
    (hx2_range : 0 < x2)
    (hx2_less_1 : x2 < 1)
    (hx1_greater_1 : 1 < x1) :
    0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end zero_points_l596_596150


namespace math_problem_l596_596088

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596088


namespace tan_arccot_of_frac_l596_596032

noncomputable theory

-- Given the problem involves trigonometric identities specifically relating to arccot and tan
def tan_arccot (x : ℝ) : ℝ :=
  Real.tan (Real.arccot x)

theorem tan_arccot_of_frac (a b : ℝ) (h : b ≠ 0) :
  tan_arccot (a / b) = b / a :=
by
  sorry

end tan_arccot_of_frac_l596_596032


namespace Vitya_catches_mother_l596_596870

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596870


namespace moon_speed_conversion_l596_596351

theorem moon_speed_conversion :
  ∀ (moon_speed_kps : ℝ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ),
  moon_speed_kps = 0.9 →
  seconds_in_minute = 60 →
  minutes_in_hour = 60 →
  (moon_speed_kps * (seconds_in_minute * minutes_in_hour) = 3240) := by
  sorry

end moon_speed_conversion_l596_596351


namespace blue_pill_cost_l596_596458

theorem blue_pill_cost (days : ℕ) (total_cost : ℝ) (cost_diff : ℝ) (daily_cost : ℝ) (cost_red_pill : ℝ) (cost_blue_pill : ℝ) :
  days = 21 →
  total_cost = 903 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  cost_red_pill = cost_blue_pill - cost_diff →
  2 * cost_blue_pill - cost_diff = daily_cost →
  cost_blue_pill = 22.5 :=
begin
  intros h_days h_total h_diff h_daily h_cost_red h_eq,
  rw h_days at h_daily,
  rw h_total at h_daily,
  norm_num at h_daily,
  have h_eq' := calc
    2 * cost_blue_pill - cost_diff = 43 : by rw [h_eq, h_daily]
    ... = 2 * 22.5 - 2 : by norm_num, 
  linarith,
end

end blue_pill_cost_l596_596458


namespace ceil_floor_diff_l596_596077

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596077


namespace angle_sum_proof_l596_596276

-- Define angles and triangle properties
variables {A B C D : Type*}
variables (x y : ℝ) -- Define the angles x and y

-- The conditions given in the problem
def is_isosceles_triangle (T: Type*) (a b : ℝ) : Prop :=
∃ A B C : T, angle A B C = a ∧ angle B A C = b ∧ angle C A B = a

def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
a + b + c = 180

-- Assume that BDA and CDA are isosceles 
axiom isosceles_BDA : is_isosceles_triangle A x x
axiom isosceles_CDA : is_isosceles_triangle A y y

-- Assume the sum of angles in triangle ABC
axiom sum_angle_ABC : sum_of_angles_in_triangle x y (x + y)

theorem angle_sum_proof : x + y = 90 :=
by sorry

end angle_sum_proof_l596_596276


namespace find_p_and_q_l596_596630

theorem find_p_and_q (p q : ℝ)
    (M : Set ℝ := {x | x^2 + p * x - 2 = 0})
    (N : Set ℝ := {x | x^2 - 2 * x + q = 0})
    (h : M ∪ N = {-1, 0, 2}) :
    p = -1 ∧ q = 0 :=
sorry

end find_p_and_q_l596_596630


namespace problem_solution_l596_596637

theorem problem_solution :
  (∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3) =
  ∑ k, (Nat.choose 5 k) ^ 3 :=
by
  sorry

end problem_solution_l596_596637


namespace boat_travel_time_downstream_l596_596932

theorem boat_travel_time_downstream
  (v c: ℝ)
  (h1: c = 1)
  (h2: 24 / (v - c) = 6): 
  24 / (v + c) = 4 := 
by
  sorry

end boat_travel_time_downstream_l596_596932


namespace sum_of_roots_l596_596516

theorem sum_of_roots :
  ∑ (x : ℚ) in ({ -4 / 3, 6 } : Finset ℚ), x = 14 / 3 :=
by
  -- Initial problem statement
  let poly := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  
  -- Extract the roots from the factored form
  have h1 : ∀ x, (3 * x + 4) = 0 → x = -4 / 3, by sorry
  have h2 : ∀ x, (2 * x - 12) = 0 → x = 6, by sorry

  -- Define the set of roots
  let roots := { -4 / 3, 6 }

  -- Compute the sum of the roots
  have sum_roots : ∑ (x : ℚ) in roots, x = 14 / 3, by sorry

  -- Final assertion
  exact sum_roots

end sum_of_roots_l596_596516


namespace ceil_floor_difference_l596_596068

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596068


namespace necessary_sufficient_condition_l596_596440

open Real

-- Define the conditions for the problem
def pyramid_base (a : ℝ) : Prop :=
  ∃ eq_triangle (side_length : ℝ), side_length = a ∧ eq_triangle = true

def is_equilateral_triangle (t : ℝ) : Prop := 
  t = (sqrt 3) / 2

-- Statement of the theorem
theorem necessary_sufficient_condition (a H : ℝ) (p : pyramid_base a) :
  (H = a * sqrt 3 / 3) ↔ (∃ r, r > 0 ∧ r = H / sqrt 3) :=
by
  sorry

end necessary_sufficient_condition_l596_596440


namespace vitya_catch_up_l596_596885

theorem vitya_catch_up (s : ℝ) : 
  let distance := 20 * s in
  let relative_speed := 4 * s in
  let t := distance / relative_speed in
  t = 5 :=
by
  let distance := 20 * s;
  let relative_speed := 4 * s;
  let t := distance / relative_speed;
  -- to complete the proof:
  sorry

end vitya_catch_up_l596_596885


namespace janet_more_siblings_than_carlos_l596_596724

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l596_596724


namespace ceiling_and_floor_calculation_l596_596091

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596091


namespace tan_arccot_3_5_l596_596045

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596045


namespace math_problem_l596_596084

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596084


namespace wall_length_l596_596933

def brick_volume_cm (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  length * width * height

def cm_to_m (volume_cm : ℕ) : ℚ :=
  volume_cm / (100^3)

noncomputable def total_volume_of_bricks_m (num_bricks : ℕ) (brick_volume_m : ℚ) : ℚ :=
  num_bricks * brick_volume_m

noncomputable def wall_volume (L : ℚ) (height : ℚ) (width : ℚ) : ℚ :=
  L * height * width

theorem wall_length (brick_length_cm : ℕ) (brick_width_cm : ℕ) (brick_height_cm : ℕ)
  (num_bricks : ℕ) (wall_height_m : ℚ) (wall_width_m : ℚ) :
  let brick_volume_m := cm_to_m (brick_volume_cm brick_length_cm brick_width_cm brick_height_cm) in
  let total_bricks_vol := total_volume_of_bricks_m num_bricks brick_volume_m in
  ∃ (L : ℚ), wall_volume L wall_height_m wall_width_m = total_bricks_vol ∧ L = 250 :=
begin
  sorry
end

#eval wall_length 20 10 7.5 25000 2 0.75

end wall_length_l596_596933


namespace abs_neg_2023_l596_596488

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l596_596488


namespace ceiling_and_floor_calculation_l596_596093

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596093


namespace random_sampling_not_in_proving_methods_l596_596398

inductive Method
| Comparison
| RandomSampling
| SyntheticAndAnalytic
| ProofByContradictionAndScaling

open Method

def proving_methods : List Method :=
  [Comparison, SyntheticAndAnalytic, ProofByContradictionAndScaling]

theorem random_sampling_not_in_proving_methods : 
  RandomSampling ∉ proving_methods :=
sorry

end random_sampling_not_in_proving_methods_l596_596398


namespace ninth_term_arithmetic_seq_l596_596133

theorem ninth_term_arithmetic_seq (a : ℕ → ℕ) (d : ℕ) (n : ℕ) 
  (h1 : ∀ k, a (k + 1) = a k + d) 
  (h2 : d = 2) 
  (h3 : (finset.range 9).sum a = 81) : 
  a 8 = 17 :=
begin
  sorry
end

end ninth_term_arithmetic_seq_l596_596133


namespace tail_heavy_permutations_count_is_28_l596_596624

def is_tail_heavy (b : Fin 5 → ℕ) : Prop :=
  b 0 + b 1 < b 3 + b 4 ∧ b 2 > 2

noncomputable def count_tail_heavy_permutations : ℕ :=
  Fintype.card {b // Perm b ∧ is_tail_heavy b}

theorem tail_heavy_permutations_count_is_28 : count_tail_heavy_permutations = 28 :=
by
  sorry

end tail_heavy_permutations_count_is_28_l596_596624


namespace vitya_catchup_time_l596_596874

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596874


namespace num_valid_divisors_karlanna_marbles_l596_596732

theorem num_valid_divisors (m : ℕ) (h : 1 < m ∧ m < 720 ∧ 720 % m = 0) : ∃ n, n > 1 ∧ 720 = m * n := sorry

theorem karlanna_marbles : 
  let divisors := finset.filter (λ m, 1 < m ∧ m < 720 ∧ 720 % m = 0) (finset.Icc 1 720) in
  divisors.card = 28 := sorry

end num_valid_divisors_karlanna_marbles_l596_596732


namespace negative_fractions_in_list_l596_596462

-- Definitions of the given numbers and the set of negative fractions.
def given_numbers : List ℚ := [5, -1, 0, -6, 125.73, 0.3, -7/2, -72/100, 5.25]

def is_negative_fraction (x : ℚ) := x < 0 ∧ is_fraction x

-- The statement of the problem in Lean.
theorem negative_fractions_in_list :
  { x ∈ given_numbers | is_negative_fraction x } = { -7/2, -72/100 } :=
sorry

end negative_fractions_in_list_l596_596462


namespace midpoint_of_segment_l596_596902

def A : ℝ × ℝ × ℝ := (10, -3, 5)
def B : ℝ × ℝ × ℝ := (-2, 7, -4)

theorem midpoint_of_segment :
  let M_x := (10 + -2 : ℝ) / 2
  let M_y := (-3 + 7 : ℝ) / 2
  let M_z := (5 + -4 : ℝ) / 2
  (M_x, M_y, M_z) = (4, 2, 0.5) :=
by
  let M_x : ℝ := (10 + -2) / 2
  let M_y : ℝ := (-3 + 7) / 2
  let M_z : ℝ := (5 + -4) / 2
  show (M_x, M_y, M_z) = (4, 2, 0.5)
  repeat { sorry }

end midpoint_of_segment_l596_596902


namespace max_gcd_thirteen_sum_1988_l596_596900

theorem max_gcd_thirteen_sum_1988 (a : Fin 13 → ℕ) (h_sum : ∑ i, a i = 1988) : ∃ d, d = 142 ∧ (∀ i, d ∣ a i) := 
sorry

end max_gcd_thirteen_sum_1988_l596_596900


namespace minimum_sum_of_distances_l596_596695

noncomputable theory

def point (x y : ℝ) := (x, y)

def A := point 1 2
def B := point 1 5
def C := point 3 6
def D := point 7 (-1)
def P := point 2 4

theorem minimum_sum_of_distances :
  ∃ P, P = point 2 4 ∧
    (∀ Q, (sum_distances A B C D Q) ≥ (sum_distances A B C D P)) :=
sorry

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def sum_distances (A B C D P : (ℝ × ℝ)) : ℝ :=
distance A P + distance B P + distance C P + distance D P

#check minimum_sum_of_distances

end minimum_sum_of_distances_l596_596695


namespace intersection_of_sets_l596_596138

def setA : Set ℝ := {x | x^2 < 8}
def setB : Set ℝ := {x | 1 - x ≤ 0}
def setIntersection : Set ℝ := {x | x ∈ setA ∧ x ∈ setB}

theorem intersection_of_sets :
    setIntersection = {x | 1 ≤ x ∧ x < 2 * Real.sqrt 2} :=
by
  sorry

end intersection_of_sets_l596_596138


namespace tan_arccot_3_5_l596_596044

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596044


namespace friend_selling_price_correct_l596_596436

-- Definition of the original cost price
def original_cost_price : ℕ := 50000

-- Definition of the loss percentage
def loss_percentage : ℕ := 10

-- Definition of the gain percentage
def gain_percentage : ℕ := 20

-- Definition of the man's selling price after loss
def man_selling_price : ℕ := original_cost_price - (original_cost_price * loss_percentage / 100)

-- Definition of the friend's selling price after gain
def friend_selling_price : ℕ := man_selling_price + (man_selling_price * gain_percentage / 100)

theorem friend_selling_price_correct : friend_selling_price = 54000 := by
  sorry

end friend_selling_price_correct_l596_596436


namespace convert_e_to_rectangular_l596_596532

-- Definitions and assumptions based on conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x
def periodicity_cos (x : ℝ) : ∀ (k : ℤ), complex.cos (x + 2 * real.pi * k) = complex.cos x
def periodicity_sin (x : ℝ) : ∀ (k : ℤ), complex.sin (x + 2 * real.pi * k) = complex.sin x

-- Problem statement
theorem convert_e_to_rectangular:
  complex.exp (complex.I * 13 * real.pi / 2) = complex.I :=
by
  sorry

end convert_e_to_rectangular_l596_596532


namespace probability_divisible_by_5_l596_596822

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l596_596822


namespace triangle_inradius_l596_596814

theorem triangle_inradius (p A : ℝ) (h_p : p = 20) (h_A : A = 30) : 
  ∃ r : ℝ, r = 3 ∧ A = r * p / 2 :=
by
  sorry

end triangle_inradius_l596_596814


namespace leonardo_nap_duration_l596_596280

theorem leonardo_nap_duration (h : (1 : ℝ) / 5 * 60 = 12) : (1 / 5 : ℝ) * 60 = 12 :=
by 
  exact h

end leonardo_nap_duration_l596_596280


namespace inscribed_circle_radius_l596_596053

def quadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  dist B A = 10 ∧ dist C B = 11 ∧ dist D C = 8 ∧ dist A D = 13

theorem inscribed_circle_radius {A B C D O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O] 
    (h : quadrilateral A B C D) 
    (center_of_circle : O) 
    (radius_of_circle : ℝ) 
    (circle_condition : ∀ p ∈ ({A, B, C, D} : Finset O), dist p center_of_circle = radius_of_circle) 
    : radius_of_circle = Real.sqrt 4.8 := 
sorry

end inscribed_circle_radius_l596_596053


namespace vitya_catches_up_in_5_minutes_l596_596851

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596851


namespace valid_four_digit_numbers_solution_l596_596638

noncomputable def count_valid_pairs : ℕ :=
  have valid_pairs : List (ℕ × ℕ) := 
    [(4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (5, 9), (6, 5), (6, 6), 
     (6, 7), (6, 8), (6, 9), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), 
     (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 4), (9, 5), (9, 6), 
     (9, 7), (9, 8), (9, 9)],
  valid_pairs.length

theorem valid_four_digit_numbers : ℕ :=
let first_digit_choices := 7 in
let middle_digit_pairs := count_valid_pairs in
let last_digit_choices := 10 in
first_digit_choices * middle_digit_pairs * last_digit_choices = 2100

theorem solution : valid_four_digit_numbers  :=
begin
  unfold valid_four_digit_numbers,
  have fd : 7 = 7 := rfl,
  have md : count_valid_pairs = 30 := rfl,
  have ld : 10 = 10 := rfl,
  rw [fd, md, ld],
  norm_num,
end

end valid_four_digit_numbers_solution_l596_596638


namespace collinearity_of_A_P_Q_l596_596588

noncomputable def equation_of_curve_E : (x y : ℝ) → Prop :=
λ x y, y^2 = 4 * x

theorem collinearity_of_A_P_Q (k b : ℝ) (hk : k ≠ 0) :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (equation_of_curve_E x₁ y₁ ∧ equation_of_curve_E x₂ y₂) →
  (y₁ = k * x₁ + b ∧ y₂ = k * x₂ + b) →
  let P := (x₂, -y₂) in
  let C := (-b / k, 0) in
  let Q := (b / k, 0) in
  collinear ℝ ({(x₁, y₁), P, Q}) :=
sorry

end collinearity_of_A_P_Q_l596_596588


namespace max_roads_city_condition_l596_596238

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l596_596238


namespace cookies_per_person_l596_596975

variable (x y z : ℕ)
variable (h_pos_z : z ≠ 0) -- Ensure z is not zero to avoid division by zero

theorem cookies_per_person (h_cookies : x * y / z = 35) : 35 / 5 = 7 := by
  sorry

end cookies_per_person_l596_596975


namespace least_four_digit_perfect_square_and_cube_l596_596385

theorem least_four_digit_perfect_square_and_cube :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ m1 : ℕ, n = m1^2) ∧ (∃ m2 : ℕ, n = m2^3) ∧ n = 4096 := sorry

end least_four_digit_perfect_square_and_cube_l596_596385


namespace max_N_value_l596_596219

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596219


namespace sam_distance_l596_596306

theorem sam_distance
  (marguerite_distance : ℝ)
  (marguerite_time : ℝ)
  (sam_time : ℝ)
  (speed_increase : ℝ)
  (marguerite_drove : marguerite_distance = 150)
  (marguerite_drove_time : marguerite_time = 3)
  (sam_drove_time : sam_time = 4)
  (speed_increase_ratio : speed_increase = 1.20)
  (marguerite_speed : ℝ) (sam_speed : ℝ) (sam_distance : ℝ) :
  marguerite_speed = marguerite_distance / marguerite_time →
  sam_speed = marguerite_speed * speed_increase →
  sam_distance = sam_speed * sam_time →
  sam_distance = 240 :=
begin
  intros,
  rw [marguerite_drove, marguerite_drove_time] at *,
  have marguerite_speed_calc : marguerite_speed = 150 / 3 := by assumption,
  rw marguerite_speed_calc,
  have marguerite_speed_val : marguerite_speed = 50 := by norm_num,
  rw marguerite_speed_val at *,
  rw speed_increase_ratio,
  have sam_speed_calc : sam_speed = 50 * 1.20 := by assumption,
  rw sam_speed_calc,
  have sam_speed_val : sam_speed = 60 := by norm_num,
  rw sam_drove_time at *,
  have sam_distance_calc : sam_distance = 60 * 4 := by assumption,
  rw sam_distance_calc,
  have sam_distance_val : sam_distance = 240 := by norm_num,
  exact sam_distance_val
end

end sam_distance_l596_596306


namespace smallest_positive_integer_l596_596908

theorem smallest_positive_integer (n : ℕ) : 13 * n ≡ 567 [MOD 5] ↔ n = 4 := by
  sorry

end smallest_positive_integer_l596_596908


namespace probability_opening_small_file_l596_596827

theorem probability_opening_small_file :
  let first_folder := { files := 16, small_files := 4 }
  let second_folder := { files := 20, small_files := 5 }
  let transfer_file := true
  ∃ (P : ℚ), P = (1 / 4) := 
by 
  sorry

end probability_opening_small_file_l596_596827


namespace ceil_floor_difference_l596_596070

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l596_596070


namespace Vitya_catches_mother_l596_596872

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596872


namespace sum_of_roots_of_poly_eq_14_over_3_l596_596501

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l596_596501


namespace arithmetic_sequence_condition_geometric_sequence_condition_any_term_as_sum_of_two_others_l596_596594

def sequence (a n : ℕ) : ℝ := n / (n + a)

theorem arithmetic_sequence_condition (a : ℕ) (h1 : a ≠ 0) :
  (2 * sequence a 2 = sequence a 1 + sequence a 4) → a = 2 :=
by
  sorry

theorem geometric_sequence_condition (a k : ℕ) (h1 : a ≠ 0) (h2 : k ≥ 10) :
  (sequence a 1 * sequence a k = (sequence a 3) ^ 2) →
  k ∈ {10, 11, 12, 13, 15, 18, 21, 27, 45} :=
by
  sorry

theorem any_term_as_sum_of_two_others (a n : ℕ) (h1 : a ≠ 0) (h2 : n ≠ 0) :
  sequence a n = sequence a (2 * n + a) - sequence a (2 * n) :=
by
  sorry

end arithmetic_sequence_condition_geometric_sequence_condition_any_term_as_sum_of_two_others_l596_596594


namespace sum_of_roots_l596_596510

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l596_596510


namespace two_students_same_initial_l596_596829

theorem two_students_same_initial
  (students : Finset String)
  (alphabet : Finset Char)
  (h_students : ∃ n, students.card = 35)
  (h_alphabet : ∃ m, alphabet.card = 33) :
  ∃ a ∈ alphabet, ∃ s1 s2 ∈ students, s1 ≠ s2 ∧ s1.head! = a ∧ s2.head! = a :=
by
  sorry

end two_students_same_initial_l596_596829


namespace probability_at_least_one_odd_probability_outside_or_on_circle_l596_596773

-- Define the sample space for rolling a die twice
def sample_space : finset (ℕ × ℕ) :=
  finset.product (finset.range 1 7) (finset.range 1 7)

-- Event: at least one number is odd
def event_B : finset (ℕ × ℕ) :=
  sample_space.filter (λ (xy : ℕ × ℕ), (xy.fst % 2 = 1) ∨ (xy.snd % 2 = 1))

-- Event: both numbers are even
def event_not_B : finset (ℕ × ℕ) :=
  sample_space.filter (λ (xy : ℕ × ℕ), (xy.fst % 2 = 0) ∧ (xy.snd % 2 = 0))

-- Probability that at least one number is odd
def probability_B : ℚ :=
  event_B.card.to_rat / sample_space.card.to_rat

-- Circle condition x^2 + y^2 <= 15
def event_C : finset (ℕ × ℕ) :=
  sample_space.filter (λ (xy : ℕ × ℕ), xy.fst * xy.fst + xy.snd * xy.snd < 15)

-- Probability that point lies inside the circle
def probability_C : ℚ :=
  event_C.card.to_rat / sample_space.card.to_rat

theorem probability_at_least_one_odd :
  probability_B = 3 / 4 :=
sorry

theorem probability_outside_or_on_circle :
  1 - probability_C = 7 / 9 :=
sorry

end probability_at_least_one_odd_probability_outside_or_on_circle_l596_596773


namespace sum_a_k_is_correct_l596_596143

noncomputable def a_k (k : ℕ) : ℝ := Real.tan k * Real.tan (k - 1)

theorem sum_a_k_is_correct :
  ∑ k in Finset.range 1994, a_k k = (Real.tan 1994 / Real.tan 1) - 1994 :=
sorry

end sum_a_k_is_correct_l596_596143


namespace highway_length_is_105_l596_596381

-- Define the speeds of the two cars
def speed_car1 : ℝ := 15
def speed_car2 : ℝ := 20

-- Define the time they travel for
def time_travelled : ℝ := 3

-- Define the distances covered by the cars
def distance_car1 : ℝ := speed_car1 * time_travelled
def distance_car2 : ℝ := speed_car2 * time_travelled

-- Define the total length of the highway
def length_highway : ℝ := distance_car1 + distance_car2

-- The theorem statement
theorem highway_length_is_105 : length_highway = 105 :=
by
  -- Skipping the proof for now
  sorry

end highway_length_is_105_l596_596381


namespace vitya_catchup_time_l596_596876

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596876


namespace sum_of_possible_values_of_x_l596_596841

-- Define the concept of an isosceles triangle with specific angles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the angle sum property of a triangle
def angle_sum_property (a b c : ℝ) : Prop := 
  a + b + c = 180

-- State the problem using the given conditions and the required proof
theorem sum_of_possible_values_of_x :
  ∀ (x : ℝ), 
    is_isosceles_triangle 70 70 x ∨
    is_isosceles_triangle 70 x x ∨
    is_isosceles_triangle x 70 70 →
    angle_sum_property 70 70 x →
    angle_sum_property 70 x x →
    angle_sum_property x 70 70 →
    (70 + 55 + 40) = 165 :=
  by
    sorry

end sum_of_possible_values_of_x_l596_596841


namespace snow_probability_at_most_3_days_l596_596815

-- Definition of binomial probability
noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * p^k * (1 - p)^(n - k)

-- Definition of the problem's conditions
def probability_snow (n : ℕ) (days_with_snow : ℕ → ℚ) : ℚ :=
  days_with_snow 0 + days_with_snow 1 + days_with_snow 2 + days_with_snow 3

-- Definition of the specific problem for the given conditions
noncomputable def problem :=
  let p := (1 : ℚ) / 5 in
  let n := 31 in
  let days_with_snow := binomial_probability n in
  probability_snow n days_with_snow

-- Main theorem stating the conclusion
theorem snow_probability_at_most_3_days :
  problem ≈ 0.415 :=
sorry

end snow_probability_at_most_3_days_l596_596815


namespace minimum_value_l596_596595

variable {x y : ℝ}

theorem minimum_value : x + 2 * y + 3 = 0 → ∃ t : ℝ, (t = (√ 5)) ∧ 
  ∀ a : ℝ, (sqrt ((-2 * y - 3)^2 + y^2 - 2 * y + 1) = t) := by
  -- Here we state the problem without providing the proof
  sorry

end minimum_value_l596_596595


namespace jebb_take_home_pay_l596_596722

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l596_596722


namespace words_written_first_two_hours_l596_596733

def essay_total_words : ℕ := 1200
def words_per_hour_first_two_hours (W : ℕ) : ℕ := 2 * W
def words_per_hour_next_two_hours : ℕ := 2 * 200

theorem words_written_first_two_hours (W : ℕ) (h : words_per_hour_first_two_hours W + words_per_hour_next_two_hours = essay_total_words) : W = 400 := 
by 
  sorry

end words_written_first_two_hours_l596_596733


namespace max_value_xyz_eq_l596_596300

noncomputable def max_value_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 3) : ℝ :=
  x^3 * y^3 * z^2

theorem max_value_xyz_eq :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ max_value_xyz x y z (by linarith) (by linarith) (by linarith) (by norm_num) = 4782969 / 390625 :=
sorry

end max_value_xyz_eq_l596_596300


namespace minimal_elements_for_conditions_l596_596679

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, Nat.odd_iff_not_even.mpr (List.length_pos_of_ne_nil sorted).2])
  else
    let a := sorted.nth_le (sorted.length / 2 - 1) (by simp [Nat.sub_pos_of_lt (Nat.div_lt_self (List.length_pos_of_ne_nil sorted)].)
    let b := sorted.nth_le (sorted.length / 2) (by simp [Nat.div_lt_self, List.length_pos_of_ne_nil sorted])
    (a + b) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def mode (s : List ℝ) : Option ℝ :=
  s.group_by id (· = ·).max_by (·.length).map (·.head)

def satisfies_conditions (s : List ℝ) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = some 6

theorem minimal_elements_for_conditions : ∀ s : List ℝ, satisfies_conditions s → s.length ≥ 6 :=
by
  intro s h
  sorry

end minimal_elements_for_conditions_l596_596679


namespace monotonic_decreasing_interval_l596_596350

/-- The monotonic decreasing interval of the function y = ln(sin(-2 * x + π / 3)) 
is [kπ - π / 12, kπ + π / 6), where k ∈ ℤ. -/
theorem monotonic_decreasing_interval 
  (k : ℤ) : 
  ∃ (x : ℝ), (k * real.pi - real.pi / 12 ≤ x ∧ x < k * real.pi + real.pi / 6) → 
  continuous_on (λ x, real.log (real.sin(-2 * x + real.pi / 3))) {x | k * real.pi - real.pi / 12 ≤ x ∧ x < k * real.pi + real.pi / 6} :=
sorry


end monotonic_decreasing_interval_l596_596350


namespace purely_imaginary_k_l596_596170

theorem purely_imaginary_k (k : ℝ) : 
  ((2 * k^2 - 3 * k - 2) + (k^2 - 2 * k) * complex.I).im = 0 ∧ ((2 * k^2 - 3 * k - 2) + (k^2 - 2 * k) * complex.I).re = 0 → k = -1 / 2 :=
by
  sorry

end purely_imaginary_k_l596_596170


namespace min_elements_with_properties_l596_596669

noncomputable def median (s : List ℝ) : ℝ :=
if h : s.length % 2 = 1 then
  s.nthLe (s.length / 2) (by simp [h])
else
  (s.nthLe (s.length / 2 - 1) (by simp [Nat.sub_right_comm, h, *]) + s.nthLe (s.length / 2) (by simp [h])) / 2

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

noncomputable def mode (s : List ℝ) : ℝ :=
s.groupBy (· = ·)
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (·.snd |>.value)

theorem min_elements_with_properties :
  ∃ s : List ℝ, 
    s.length ≥ 6 ∧ 
    median s = 3 ∧ 
    mean s = 5 ∧ 
    ∃! m, mode s = m ∧ m = 6 :=
by
  sorry

end min_elements_with_properties_l596_596669


namespace sum_of_f_l596_596581

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 :=
by
  sorry

end sum_of_f_l596_596581


namespace sum_of_y_coordinates_l596_596619

-- Define the given function f(x)
def f (x : ℝ) : ℝ := (x + 1) / x

-- Define the odd function g(x) such that y = g(x) - 1
def g (x : ℝ) : ℝ

-- Assume g(x) is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g (x)

-- Intersection points of f(x) and g(x)
variables (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
axiom intersections : f x1 = g x1 ∧ f x2 = g x2 ∧ f x3 = g x3 ∧ f x4 = g x4
axiom intersection_points : y1 = f x1 ∧ y2 = f x2 ∧ y3 = f x3 ∧ y4 = f x4

-- Prove that the sum of y-coordinates is equal to 4
theorem sum_of_y_coordinates : y1 + y2 + y3 + y4 = 4 :=
by sorry

end sum_of_y_coordinates_l596_596619


namespace e_to_13pi_2_eq_i_l596_596535

-- Define the problem in Lean 4
theorem e_to_13pi_2_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I :=
by
  sorry

end e_to_13pi_2_eq_i_l596_596535


namespace total_weight_on_scale_l596_596459

-- Define the weights of Alexa and Katerina
def alexa_weight : ℕ := 46
def katerina_weight : ℕ := 49

-- State the theorem to prove the total weight on the scale
theorem total_weight_on_scale : alexa_weight + katerina_weight = 95 := by
  sorry

end total_weight_on_scale_l596_596459


namespace sharon_trip_distance_l596_596774

theorem sharon_trip_distance
  (h1 : ∀ (d : ℝ), (180 * d) = 1 ∨ (d = 0))  -- Any distance traveled in 180 minutes follows 180d=1 (usual speed)
  (h2 : ∀ (d : ℝ), (276 * (d - 20 / 60)) = 1 ∨ (d = 0))  -- With reduction in speed due to snowstorm too follows a similar relation
  (h3: ∀ (total_time : ℝ), total_time = 276 ∨ total_time = 0)  -- Total time is 276 minutes
  : ∃ (x : ℝ), x = 135 := sorry

end sharon_trip_distance_l596_596774


namespace convert_exp_to_rectangular_form_l596_596539

theorem convert_exp_to_rectangular_form : exp (13 * π * complex.I / 2) = complex.I :=
by
  sorry

end convert_exp_to_rectangular_form_l596_596539


namespace Vitya_catchup_mom_in_5_l596_596893

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596893


namespace vitya_catches_up_in_5_minutes_l596_596850

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596850


namespace janet_more_siblings_than_carlos_l596_596726

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l596_596726


namespace sum_of_roots_of_poly_eq_14_over_3_l596_596498

-- Define the polynomial
def poly (x : ℚ) : ℚ := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- Define the statement to prove
theorem sum_of_roots_of_poly_eq_14_over_3 :
  (∑ x in ([(-4/3), 6] : list ℚ), x) = 14 / 3 :=
by
  -- stating the polynomial equation
  have h_poly_eq_zero : poly = (3 * (3 * x + 4) * (x - 6)) by {
    sorry
  }
  
  -- roots of the polynomial
  have h_roots : {x : ℚ | poly x = 0} = {(-4/3), 6} by {
    sorry
  }

  -- sum of the roots
  sorry

end sum_of_roots_of_poly_eq_14_over_3_l596_596498


namespace ratio_of_areas_eq_9_over_25_l596_596782

theorem ratio_of_areas_eq_9_over_25 (x : ℝ) :
  let area_c := (3 * x) * (3 * x),
      area_d := (5 * x) * (5 * x)
  in area_c / area_d = 9 / 25 :=
by
  let area_c := (3 * x) * (3 * x)
  let area_d := (5 * x) * (5 * x)
  suffices area_c / area_d = 9/25 by
    exact this
  sorry

end ratio_of_areas_eq_9_over_25_l596_596782


namespace Julian_numbers_l596_596324

/--
We have a set of numbers from 1 to 9. Define the draws of three people: Pedro, Ana, and Julian,
such that:
1. Pedro draws three consecutive numbers whose product is 5 times their sum.
2. Ana draws three numbers without prime numbers, two of which are consecutive, and their product is 4 times their sum.
3. Identify the numbers Julian draws.
-/
theorem Julian_numbers
  (n : ℕ)
  (h1 : n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h2 : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 5)
  (h3 : n ≠ 1 ∧ n ≠ 8 ∧ n ≠ 9) :
  {2, 6, 7} = {m | m ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ m ≠ 3 ∧ m ≠ 4 ∧ m ≠ 5 ∧ m ≠ 1 ∧ m ≠ 8 ∧ m ≠ 9} :=
by {
  sorry
}

end Julian_numbers_l596_596324


namespace part_a_part_b_l596_596418

theorem part_a (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℤ), a < m * α - n ∧ m * α - n < b :=
sorry

theorem part_b (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end part_a_part_b_l596_596418


namespace factorization_correct_l596_596553

noncomputable def factor_polynomial : Polynomial ℝ :=
  Polynomial.X^6 - 64

theorem factorization_correct : 
  factor_polynomial = 
  (Polynomial.X - 2) * 
  (Polynomial.X + 2) * 
  (Polynomial.X^4 + 4 * Polynomial.X^2 + 16) :=
by
  sorry

end factorization_correct_l596_596553


namespace latoya_initial_payment_l596_596279

variable (cost_per_minute : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) 
variable (initial_credit : ℝ)

theorem latoya_initial_payment : 
  ∀ (cost_per_minute call_duration remaining_credit initial_credit : ℝ),
  cost_per_minute = 0.16 →
  call_duration = 22 →
  remaining_credit = 26.48 →
  initial_credit = (cost_per_minute * call_duration) + remaining_credit →
  initial_credit = 30 :=
by
  intros cost_per_minute call_duration remaining_credit initial_credit
  sorry

end latoya_initial_payment_l596_596279


namespace max_of_2xy_l596_596585

theorem max_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 2 * x * y ≤ 8 :=
by
  sorry

end max_of_2xy_l596_596585


namespace vitya_catches_up_in_5_minutes_l596_596846

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l596_596846


namespace simplify_expr1_simplify_expr2_l596_596779

theorem simplify_expr1 (h1 : sin (135 * Real.pi / 180) = sqrt 2 / 2) (h2 : cos (135 * Real.pi / 180) = -sqrt 2 / 2) :
  (sqrt (1 - 2 * sin (135 * Real.pi / 180) * cos (135 * Real.pi / 180)) / (sin (135 * Real.pi / 180) + sqrt (1 - (sin (135 * Real.pi / 180))^2))) = 1 :=
sorry

theorem simplify_expr2 (θ : ℝ) :
  (sin (θ - 5 * Real.pi) * cos (-Real.pi / 2 - θ) * cos (8 * Real.pi - θ) / 
  (sin (θ - 3 * Real.pi / 2) * sin (-θ - 4 * Real.pi))) = -sin (θ - 5 * Real.pi) :=
sorry

end simplify_expr1_simplify_expr2_l596_596779


namespace solid_views_same_shape_and_size_l596_596931

theorem solid_views_same_shape_and_size (solid : Type) (sphere triangular_pyramid cube cylinder : solid)
  (views_same_shape_and_size : solid → Bool) : 
  views_same_shape_and_size cylinder = false :=
sorry

end solid_views_same_shape_and_size_l596_596931


namespace vitya_catchup_time_l596_596859

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596859


namespace find_a_statement_l596_596556

noncomputable def find_a : set ℝ := { a | ∃ (n : ℕ) (A : ℕ → set ℤ),
  (∀ i, set.infinite (A i)) ∧          -- Each A_i is infinite
  (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧    -- A_i are pairwise disjoint
  (⋃ i, A i = set.univ) ∧             -- Union of A_i covers all integers
  (∀ i, ∀ b c ∈ A i, b > c → b - c ≥ a^i) }  -- Condition on the difference of elements

theorem find_a_statement : find_a = {a | a > 0 ∧ a < 2} :=
begin
  sorry
end

end find_a_statement_l596_596556


namespace determine_b_l596_596991

theorem determine_b (b : ℝ) : (∀ x : ℝ, (-x^2 + b * x + 1 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by sorry

end determine_b_l596_596991


namespace krish_remaining_money_l596_596759

variable (initial_amount sweets stickers friends each_friend charity : ℝ)

theorem krish_remaining_money :
  initial_amount = 200.50 →
  sweets = 35.25 →
  stickers = 10.75 →
  friends = 4 →
  each_friend = 25.20 →
  charity = 15.30 →
  initial_amount - (sweets + stickers + friends * each_friend + charity) = 38.40 :=
by
  intros h_initial h_sweets h_stickers h_friends h_each_friend h_charity
  sorry

end krish_remaining_money_l596_596759


namespace angle_of_inclination_of_line_l596_596611

theorem angle_of_inclination_of_line
  (a b c : ℝ)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0)
  (h_sym_axis : ∀ x, f x = f (π - x))
  (h_f_def : ∀ x, f x = a * real.sin x - b * real.cos x) :
  (∃ θ, θ = π / 4 ∧ ∃ k, - (a / b) = k ∧ k = 1) :=
sorry

end angle_of_inclination_of_line_l596_596611


namespace exponential_to_rectangular_form_l596_596525

theorem exponential_to_rectangular_form : 
  (Real.exp (Complex.i * (13 * Real.pi / 2))) = Complex.i :=
by
  sorry

end exponential_to_rectangular_form_l596_596525


namespace vitya_catch_up_time_l596_596863

theorem vitya_catch_up_time
  (s : ℝ)  -- speed of Vitya and his mom in meters per minute
  (t : ℝ)  -- time in minutes to catch up
  (h : t = 5) : 
  let distance := 20 * s in   -- distance between Vitya and his mom after 10 minutes
  let relative_speed := 4 * s in  -- relative speed of Vitya with respect to his mom
  distance / relative_speed = t  -- time to catch up is distance divided by relative speed
:=
  by sorry

end vitya_catch_up_time_l596_596863


namespace bowls_remaining_l596_596973

def initial_bowls : ℕ := 250

def customers_purchases : List (ℕ × ℕ) :=
  [(5, 7), (10, 15), (15, 22), (5, 36), (7, 46), (8, 0)]

def reward_ranges (bought : ℕ) : ℕ :=
  if bought >= 5 && bought <= 9 then 1
  else if bought >= 10 && bought <= 19 then 3
  else if bought >= 20 && bought <= 29 then 6
  else if bought >= 30 && bought <= 39 then 8
  else if bought >= 40 then 12
  else 0

def total_free_bowls : ℕ :=
  List.foldl (λ acc (n, b) => acc + n * reward_ranges b) 0 customers_purchases

theorem bowls_remaining :
  initial_bowls - total_free_bowls = 1 := by
  sorry

end bowls_remaining_l596_596973


namespace Vitya_catchup_mom_in_5_l596_596892

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596892


namespace vitya_catchup_time_l596_596875

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l596_596875


namespace largest_prime_divisor_of_base7_number_l596_596103

def base7_to_base10 (n : Nat) : Nat := 
  2 * 7^6 + 1 * 7^5 + 0 * 7^4 + 2 * 7^3 + 0 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem largest_prime_divisor_of_base7_number :
  let n := base7_to_base10 2102012 in n = 252800 ∧ largest_prime_factor n = 13 :=
by
  sorry

end largest_prime_divisor_of_base7_number_l596_596103


namespace max_value_of_N_l596_596199

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596199


namespace exists_a_solution_iff_l596_596557

theorem exists_a_solution_iff (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1 / 4 := 
by 
  sorry

end exists_a_solution_iff_l596_596557


namespace trajectory_of_center_l596_596128

-- Define the fixed circle C as x^2 + (y + 3)^2 = 1
def fixed_circle (p : ℝ × ℝ) : Prop :=
  (p.1)^2 + (p.2 + 3)^2 = 1

-- Define the line y = 2
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2

-- The main theorem stating the trajectory of the center of circle M is x^2 = -12y
theorem trajectory_of_center :
  ∀ (M : ℝ × ℝ), 
  tangent_line M → (∃ r : ℝ, fixed_circle (M.1, M.2 - r) ∧ r > 0) →
  (M.1)^2 = -12 * M.2 :=
sorry

end trajectory_of_center_l596_596128


namespace smallest_integer_n_satisfying_inequality_l596_596111

theorem smallest_integer_n_satisfying_inequality 
  (x y z : ℝ) : 
  (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

end smallest_integer_n_satisfying_inequality_l596_596111


namespace maximum_N_value_l596_596231

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l596_596231


namespace arrangement_ways_l596_596551

def green_marbles : Nat := 7
noncomputable def N_max_blue_marbles : Nat := 924

theorem arrangement_ways (N : Nat) (blue_marbles : Nat) (total_marbles : Nat)
  (h1 : total_marbles = green_marbles + blue_marbles) 
  (h2 : ∃ b_gap, b_gap = blue_marbles - (total_marbles - green_marbles - 1))
  (h3 : blue_marbles ≥ 6)
  : N = N_max_blue_marbles := 
sorry

end arrangement_ways_l596_596551


namespace evie_shells_left_l596_596065

theorem evie_shells_left :
  (collect_per_day : ℕ) (days_collected : ℕ) (given_away : ℕ) (total_collected : ℕ) (remaining : ℕ),
  collect_per_day = 10 →
  days_collected = 6 →
  given_away = 2 →
  total_collected = collect_per_day * days_collected →
  remaining = total_collected - given_away →
  remaining = 58 :=
by
  intros collect_per_day days_collected given_away total_collected remaining
  assume h1 : collect_per_day = 10
  assume h2 : days_collected = 6
  assume h3 : given_away = 2
  assume h4 : total_collected = collect_per_day * days_collected
  assume h5 : remaining = total_collected - given_away
  rw [h1, h2, h3] at h4
  rw [h4, h3] at h5
  simp at h5
  exact h5


end evie_shells_left_l596_596065


namespace max_value_of_N_l596_596197

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596197


namespace highest_numbered_street_l596_596969

theorem highest_numbered_street (L : ℕ) (d : ℕ) (H : L = 15000 ∧ d = 500) : 
    (L / d) - 2 = 28 :=
by
  sorry

end highest_numbered_street_l596_596969


namespace percentage_error_edge_percentage_error_edge_l596_596962

open Real

-- Define the main context, E as the actual edge and E' as the calculated edge
variables (E E' : ℝ)

-- Condition: Error in calculating the area is 4.04%
axiom area_error : E' * E' = E * E * 1.0404

-- Statement: To prove that the percentage error in edge calculation is 2%
theorem percentage_error_edge : (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

-- Alternatively, include variable and condition definitions in the actual theorem statement
theorem percentage_error_edge' (E E' : ℝ) (h : E' * E' = E * E * 1.0404) : 
    (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

end percentage_error_edge_percentage_error_edge_l596_596962


namespace find_geometric_sequence_values_l596_596605

theorem find_geometric_sequence_values :
  ∃ (a b c : ℤ), (∃ q : ℤ, q ≠ 0 ∧ 2 * q ^ 4 = 32 ∧ a = 2 * q ∧ b = 2 * q ^ 2 ∧ c = 2 * q ^ 3)
                 ↔ ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end find_geometric_sequence_values_l596_596605


namespace three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l596_596475

def three_digit_odd_nums (digits : Finset ℕ) : ℕ :=
  let odd_digits := digits.filter (λ n => n % 2 = 1)
  let num_choices_for_units_place := odd_digits.card
  let remaining_digits := digits \ odd_digits
  let num_choices_for_hundreds_tens_places := remaining_digits.card * (remaining_digits.card - 1)
  num_choices_for_units_place * num_choices_for_hundreds_tens_places

theorem three_digit_odd_nums_using_1_2_3_4_5_without_repetition :
  three_digit_odd_nums {1, 2, 3, 4, 5} = 36 :=
by
  -- Proof is skipped
  sorry

end three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l596_596475


namespace e_to_13pi_2_eq_i_l596_596533

-- Define the problem in Lean 4
theorem e_to_13pi_2_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I :=
by
  sorry

end e_to_13pi_2_eq_i_l596_596533


namespace wrapping_paper_area_l596_596429

theorem wrapping_paper_area (w h : ℝ) : 
  let l := 2 * w in
  A = (4 * w + h) * (2 * w + h) -> 
  A = 8 * w^2 + 6 * w * h + h^2 :=
by
  sorry

end wrapping_paper_area_l596_596429


namespace max_area_triangle_has_max_area_l596_596377

noncomputable def max_area_triangle (PQ QR_ratio RP_ratio : ℝ) : ℝ :=
  -- constants based on given problem
  let PQ := 15
  let QR_ratio := 5
  let RP_ratio := 9
  -- let y be a positive real representing segments QR and RP
  ∀ (y : ℝ), 3 < y ∧ y < 5 → 
  -- lengths of the triangle sides QR and RP
  let QR := QR_ratio * y
  let RP := RP_ratio * y
  -- semi-perimeter s
  let s := (PQ + QR + RP) / 2
  -- area calculation using Heron's formula
  let area_squared := s * (s - PQ) * (s - QR) * (s - RP)
  -- calculate upper bound
  max_area = 612.5

theorem max_area_triangle_has_max_area :
  max_area_triangle 15 5 9 = 612.5 := sorry  

end max_area_triangle_has_max_area_l596_596377


namespace common_tangents_angle_l596_596808

theorem common_tangents_angle
  {r : ℝ} (h_pos : r > 0)
  (A B : \(\mathbb{R}^2\)) (h_AB_len : dist A B = 2 * r)
  (circle1 : set \( \mathbb{R}^2\)) (h_circle1 : circle1 = sphere A r)
  (circle2 : set \( \mathbb{R}^2\)) (h_circle2 : ∃ C, dist B C = r ∧ circle2 = sphere C r) :
  ∃ β, β = 36 + 52 / 60 :=
by
  sorry

end common_tangents_angle_l596_596808


namespace find_population_2003_l596_596812

def population_in_2003 (P : ℕ → ℕ) : Prop :=
  (P 2001 = 50) ∧ (P 2002 = 80) ∧ (P 2004 = 170) ∧
  (∀ n, P(n + 2) - P(n) = k * P(n + 1)) →
  (P 2003 = 120)

theorem find_population_2003 (P : ℕ → ℕ) (k : ℤ) 
  (h_prop : population_in_2003 P) : 
  P 2003 = 120 :=
by 
  sorry

end find_population_2003_l596_596812


namespace max_number_of_lines_l596_596118

theorem max_number_of_lines (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5}) :
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, p.1 < p.2) (s.product s))) = 10 :=
by
  -- intro
  rw h
  -- apply finset_ext
  sorry

end max_number_of_lines_l596_596118


namespace distance_between_A_and_B_l596_596316

def scale : ℕ := 20000
def map_distance : ℕ := 6
def actual_distance_cm : ℕ := scale * map_distance
def actual_distance_m : ℕ := actual_distance_cm / 100

theorem distance_between_A_and_B : actual_distance_m = 1200 := by
  sorry

end distance_between_A_and_B_l596_596316


namespace exponential_to_rectangular_form_l596_596527

theorem exponential_to_rectangular_form : 
  (Real.exp (Complex.i * (13 * Real.pi / 2))) = Complex.i :=
by
  sorry

end exponential_to_rectangular_form_l596_596527


namespace area_of_triangle_ABC_l596_596149

theorem area_of_triangle_ABC :
  let circle_eq := (fun x y => (x - 3)^2 + (y - 4)^2 = 25)
  let line_eq := (fun x y => 3 * x + 4 * y - 5 = 0)
  let center := (3, 4)
  let r := 5
  let d := abs ((3 * 3) + (4 * 4) - 5) / real.sqrt (3^2 + 4^2)
  let AB := 2 * real.sqrt (r^2 - d^2)
  let S_ABC := (1 / 2) * AB * d
  in S_ABC = 12 :=
by
  sorry

end area_of_triangle_ABC_l596_596149


namespace no_injective_function_l596_596405

theorem no_injective_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m * n) = f m + f n) : ¬ Function.Injective f := 
sorry

end no_injective_function_l596_596405


namespace part1_part2_l596_596745

-- Definitions and conditions
variables {A B C a b c : ℝ}
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A)) -- Given condition

-- Part (1): If A = 2B, then find C
theorem part1 (h2 : A = 2 * B) : C = (5 / 8) * π := by
  sorry

-- Part (2): Prove that 2a² = b² + c²
theorem part2 : 2 * a^2 = b^2 + c^2 := by
  sorry

end part1_part2_l596_596745


namespace constant_seq_arith_geo_l596_596958

def is_arithmetic_sequence (s : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n + d

def is_geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n * r

theorem constant_seq_arith_geo (s : ℕ → ℝ) (d r : ℝ) :
  is_arithmetic_sequence s d →
  is_geometric_sequence s r →
  (∃ c : ℝ, ∀ n : ℕ, s n = c) ∧ r = 1 :=
by
  sorry

end constant_seq_arith_geo_l596_596958


namespace product_of_largest_and_second_largest_l596_596833

theorem product_of_largest_and_second_largest (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  (max (max a b) c * (max (min a (max b c)) (min b (max a c)))) = 132 :=
by
  sorry

end product_of_largest_and_second_largest_l596_596833


namespace tax_rate_correct_l596_596450

def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 89.6

def taxable_amount : ℝ := total_value - non_taxable_amount

theorem tax_rate_correct : (tax_paid / taxable_amount) * 100 = 8 := by
  sorry

end tax_rate_correct_l596_596450


namespace moe_mowing_time_l596_596766

-- Define constants 
def lawn_length : ℝ := 120
def lawn_width : ℝ := 180
def swath_width_inches : ℝ := 30
def overlap_inches : ℝ := 4
def walking_rate : ℝ := 4000
def inches_to_feet (inches : ℝ) : ℝ := inches / 12

-- Effective swath width in feet
def effective_swath_width : ℝ := inches_to_feet (swath_width_inches - overlap_inches)

-- Number of strips needed to cover the lawn width
def num_strips : ℝ := lawn_width / effective_swath_width

-- Total distance Moe will cover
def total_distance : ℝ := num_strips * lawn_length

-- Time taken to mow the lawn
def mowing_time : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  mowing_time = 2.49 :=
  by
    -- The proof will involve carrying out the necessary computations using the definitions above
    sorry

end moe_mowing_time_l596_596766


namespace ceil_floor_diff_l596_596081

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l596_596081


namespace max_possible_cities_traversed_l596_596202

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l596_596202


namespace calculate_savings_l596_596011

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l596_596011


namespace negate_universal_proposition_l596_596810

open Classical

def P (x : ℝ) : Prop := x^3 - 3*x > 0

theorem negate_universal_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end negate_universal_proposition_l596_596810


namespace readers_both_l596_596408

-- Define the given conditions
def total_readers : ℕ := 250
def readers_S : ℕ := 180
def readers_L : ℕ := 88

-- Define the proof statement
theorem readers_both : (readers_S + readers_L - total_readers = 18) :=
by
  -- Proof is omitted
  sorry

end readers_both_l596_596408


namespace negation_of_proposition_l596_596578

-- Definitions of the conditions
variables (a b c : ℝ) 

-- Prove the mathematically equivalent statement:
theorem negation_of_proposition :
  (a + b + c ≠ 1) → (a^2 + b^2 + c^2 > 1 / 9) :=
sorry

end negation_of_proposition_l596_596578


namespace max_N_value_l596_596221

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l596_596221


namespace vitya_catchup_time_l596_596853

theorem vitya_catchup_time (s : ℝ) (h1 : s > 0) : 
  let distance := 20 * s,
      relative_speed := 4 * s in
  distance / relative_speed = 5 := by
  sorry

end vitya_catchup_time_l596_596853


namespace coeff_x3_term_l596_596330

theorem coeff_x3_term (n : ℕ) (a b : ℕ) (r : ℕ) (x : ℕ) (coeff : ℕ) :
  (a = 1) → (b = 1) → (n = 10) → (r = 3) → (coeff = (-1) ^ r * (Nat.choose n r) * x ^ r) →
  coeff = -120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp [Nat.choose] at h5
  sorry

end coeff_x3_term_l596_596330


namespace nested_rectangles_l596_596693

noncomputable section

variables {n : ℕ} (a b : Fin n → ℝ)

def is_sorted (v : Fin n → ℝ) : Prop :=
  ∀ i j : Fin n, i ≤ j → v i ≥ v j

def sum_equal (a b : Fin n → ℝ) : Prop :=
  ∑ i, a i = ∑ i, b i

def can_be_nested (a b : ℝ) : Prop :=
  a ≥ b ∨ b ≥ a

theorem nested_rectangles (a_sorted : is_sorted a)
                           (b_sorted : is_sorted b)
                           (sum_cond : sum_equal a b) :
  ∀ (i j k l : Fin n), can_be_nested (a i) (a k) →
                        can_be_nested (b j) (b l) :=
by
  intros i j k l h1 h2
  sorry

end nested_rectangles_l596_596693


namespace area_triangle_MND_l596_596755

variables {A B C D M N : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ M] [InnerProductSpace ℝ N]
variables (parallelogram_area : ℝ)

-- Define the parallelogram and its properties
def is_parallelogram (A B C D : Type) : Prop :=
∃ f1 f2 : A, line f1 A B ∧ line f2 C D ∧ A ≠ C ∧ B ≠ D ∧ A ≠ D

-- Define the segment
def on_segment (M B D : Type) (k : ℝ) : Prop :=
k > 0 ∧ k < 1 ∧ M = k • B + (1 - k) • D

-- Define the intersection point
def is_intersection (AM CB : Type) (N : Type) : Prop :=
∃ f : AM, line f AM ∧ line f CB ∧ belongs_to_line N AM ∧ belongs_to_line N CB

-- Hypothesis
variable (hpar : is_parallelogram A B C D)
variable (harea : parallelogram_area A B C D = 1)
variable (hM : on_segment M B D (3 / 4))
variable (hN : is_intersection (AM : Line A M) (CB : Line C B) N)

-- Proof statement
theorem area_triangle_MND {A B C D M N : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ M] [InnerProductSpace ℝ N]
  (hpar : is_parallelogram A B C D)
  (harea : parallelogram_area A B C D = 1)
  (hM : on_segment M B D (3 / 4))
  (hN : is_intersection (AM : Line A M) (CB : Line C B) N) :
  triangle_area M N D = 1/8 := 
sorry

end area_triangle_MND_l596_596755


namespace numeral_in_150th_decimal_place_of_13_div_14_l596_596911

theorem numeral_in_150th_decimal_place_of_13_div_14 : (decimal_representation (13/14)).digit_at 150 = 1 :=
by sorry

end numeral_in_150th_decimal_place_of_13_div_14_l596_596911


namespace maximum_possible_value_of_N_l596_596212

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596212


namespace total_amount_l596_596837

noncomputable def initial_amounts (a j t : ℕ) := (t = 24)
noncomputable def redistribution_amounts (a j t a' j' t' : ℕ) :=
  a' = 3 * (2 * (a - 2 * j - 24)) ∧
  j' = 3 * (3 * j - (a - 2 * j - 24 + 48)) ∧
  t' = 144 - (6 * (a - 2 * j - 24) + 9 * j - 3 * (a - 2 * j - 24 + 48))

theorem total_amount (a j t a' j' t' : ℕ) (h1 : t = 24)
  (h2 : redistribution_amounts a j t a' j' t')
  (h3 : t' = 24) : 
  a + j + t = 72 :=
sorry

end total_amount_l596_596837


namespace solution_set_l596_596780

noncomputable def solve_equation (x : ℝ) : Prop :=
  sqrt (log x ^ 2 + log x ^ 2 + 1) + log x + 1 = 0

theorem solution_set : {x : ℝ | x > 0 ∧ solve_equation x} = {x : ℝ | 0 < x ∧ x ≤ 1 / 10} :=
by sorry

end solution_set_l596_596780


namespace octagon_sequences_l596_596346

theorem octagon_sequences (n : ℕ) (h_conditions : 
  ∀ a : ℕ, ∃ b : ℕ, b < 20 ∧ 
  ∀ d : ℕ, ¬(7 * d % 2 = 1) ∧ 
  0 < (270 - 7 * d) ∧ 
  270 - 6 * d ≤ 160 ∧ 
  (d <= 18) ) :
  n = 9 :=
begin
  sorry
end

end octagon_sequences_l596_596346


namespace convert_e_to_rectangular_l596_596530

-- Definitions and assumptions based on conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x
def periodicity_cos (x : ℝ) : ∀ (k : ℤ), complex.cos (x + 2 * real.pi * k) = complex.cos x
def periodicity_sin (x : ℝ) : ∀ (k : ℤ), complex.sin (x + 2 * real.pi * k) = complex.sin x

-- Problem statement
theorem convert_e_to_rectangular:
  complex.exp (complex.I * 13 * real.pi / 2) = complex.I :=
by
  sorry

end convert_e_to_rectangular_l596_596530


namespace camilla_country_drive_time_is_38_minutes_l596_596490

section
variable (x : ℝ) -- Camilla's speed in the city in miles per hour

-- Time taken to drive 20 miles in the city
def city_time (x : ℝ) : ℝ := 20 / x

-- Time taken to drive 40 miles in the country
def country_time (x : ℝ) : ℝ := 40 / (x + 20)

-- Total time for the trip is 1 hour
def total_trip_time (x : ℝ) : Prop := city_time x + country_time x = 1

-- Time spent driving in the country
def country_time_minutes (x : ℝ) : ℝ := (country_time x) * 60

-- Prove that the time Camilla drove in the country rounded to the nearest minute is 38 minutes
theorem camilla_country_drive_time_is_38_minutes (x : ℝ) (h : total_trip_time x) :
  round (country_time_minutes x) = 38 := sorry
end

end camilla_country_drive_time_is_38_minutes_l596_596490


namespace least_integer_satisfying_conditions_l596_596901

theorem least_integer_satisfying_conditions :
  ∃ (N : ℕ), N ≡ 9 [MOD 10] ∧ N ≡ 10 [MOD 11] ∧ N ≡ 11 [MOD 12] ∧ N ≡ 12 [MOD 13] ∧ N = 8579 := by
  use 8579
  split
  sorry
  split
  sorry
  split
  sorry
  split
  sorry

end least_integer_satisfying_conditions_l596_596901


namespace popcorn_distance_l596_596277

def total_distance : ℝ := 5000
def remaining_kernels : ℕ := 150
def proportion_remaining : ℝ := 3 / 4
def original_kernels (r : ℕ) (p : ℝ) : ℕ := r * 4 / 3
def distance_per_kernel (td : ℝ) (k : ℕ) : ℝ := td / k

theorem popcorn_distance :
  distance_per_kernel total_distance (original_kernels remaining_kernels proportion_remaining) = 25 :=
by
  sorry

end popcorn_distance_l596_596277


namespace determinant_condition_l596_596644

variable (p q r s : ℝ)

theorem determinant_condition (h: p * s - q * r = 5) :
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 20 :=
by
  sorry

end determinant_condition_l596_596644


namespace math_problem_l596_596083

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l596_596083


namespace find_m_find_min_and_x_values_l596_596616

-- Defining the function
def f (m : ℝ) (x : ℝ) : ℝ := m * (1 + Real.sin (2 * x)) + Real.cos (2 * x)

-- Condition: The function passes through (π/4, 2)
def condition (m : ℝ) : Prop := f m (π / 4) = 2

-- The value of m
theorem find_m : ∃ m : ℝ, condition m := 
  sorry

-- The minimum value and the set of x values where the minimum occurs
def f_min (x : ℝ) : Prop := f 1 x = 1 - Real.sqrt 2
def x_set (x : ℝ) (k : ℤ) : Prop := x = k * π - (3 * π) / 8

theorem find_min_and_x_values : 
  ∃ y : ℝ, (y = 1 - Real.sqrt 2) ∧ (∀ x : ℝ, f_min x → ∃ k : ℤ, x_set x k) :=
  sorry

end find_m_find_min_and_x_values_l596_596616


namespace max_possible_N_in_cities_l596_596248

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l596_596248


namespace volume_Q2_m_plus_n_l596_596403

noncomputable def volume_seq (i : ℕ) : ℚ :=
  match i with
  | 0     => 1
  | (n+1) => let v := volume_seq n in
             v + (8 * (1 / (2 ^ (3 * (n + 1)))))

theorem volume_Q2 : volume_seq 2 = 3 := 
by sorry

theorem m_plus_n : (3 : ℚ).numerator + (3 : ℚ).denominator = 4 := 
by norm_num

end volume_Q2_m_plus_n_l596_596403


namespace find_hyperbola_eq_M_is_on_circle_area_of_triangle_l596_596589

def ecc := Real.sqrt 2
def c := Real.sqrt 6

def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 6

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola_eq x y

axiom hyperbola_passes_through : point_on_hyperbola 4 (-Real.sqrt 10)

theorem find_hyperbola_eq :
    ∀ (x y : ℝ), (point_on_hyperbola x y ↔ hyperbola_eq x y) := sorry

theorem M_is_on_circle (m : ℝ) :
    point_on_hyperbola 3 m →
    (m = Real.sqrt 3 ∨ m = -Real.sqrt 3) ∧
    (let F₁ := (Real.sqrt 6, 0)
     let F₂ := (-Real.sqrt 6, 0)
     let MF₁ := (Real.sqrt 6 - 3, -m)
     let MF₂ := (-(Real.sqrt 6) - 3, -m)
     MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0
    ) := sorry

theorem area_of_triangle :
    ∀ (m : ℝ), point_on_hyperbola 3 m →
    (Real.abs (3 * Real.sqrt 2)) = 3 * Real.sqrt 2 := sorry

end find_hyperbola_eq_M_is_on_circle_area_of_triangle_l596_596589


namespace stockpile_defective_percentage_l596_596686

def percentage_defective (products : ℕ) :=
  let m1_contribution := 0.30 * products
  let m2_contribution := 0.20 * products
  let m3_contribution := 0.15 * products
  let m4_contribution := 0.25 * products
  let m5_contribution := 0.10 * products

  let m1_defective := 0.04 * m1_contribution
  let m2_defective := 0.02 * m2_contribution
  let m3_defective := 0.03 * m3_contribution
  let m4_defective := 0.05 * m4_contribution
  let m5_defective := 0.02 * m5_contribution

  let total_defective := m1_defective + m2_defective + m3_defective + m4_defective + m5_defective
  (total_defective / products) * 100

theorem stockpile_defective_percentage : percentage_defective 100 = 3.5 := by
  sorry

end stockpile_defective_percentage_l596_596686


namespace tan_arccot_3_5_l596_596047

theorem tan_arccot_3_5 : Real.tan (Real.arccot (3/5)) = 5/3 :=
by
  sorry

end tan_arccot_3_5_l596_596047


namespace maximum_possible_value_of_N_l596_596213

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596213


namespace ceiling_and_floor_calculation_l596_596090

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l596_596090


namespace a_n_odd_l596_596748

def distinct_ratios (xs : List ℕ) (n : ℕ) : Prop :=
  xs.length = n ∧ (List.range 1 (n + 1)).All (λ k, xs.nth k ≠ none ∧ xs.nth k ≠ some k ∧ ∀ j ≠ k, xs.nth (k - 1) / k ≠ xs.nth (j - 1) / j)

def a_n (n : ℕ) : ℕ :=
  Nat.card { xs : List ℕ // distinct_ratios xs n }

theorem a_n_odd : ∀ n : ℕ, 1 ≤ n → Odd (a_n n) := by
  intros n hn
  sorry

end a_n_odd_l596_596748


namespace solve_for_y_l596_596771

theorem solve_for_y (x y : ℝ) (h : 3 * x + y = 17) : y = -3 * x + 17 :=
by {
  sorry,
}

end solve_for_y_l596_596771


namespace range_m_range_h_l596_596615

-- Problem 1
theorem range_m (m : ℝ) : 
  (∀ x > 2, (x - 4) * real.exp (x - 2) + m * x ≥ 0) ↔ (m ∈ set.Ici 1) :=
sorry

-- Problem 2
theorem range_h (a : ℝ) (h : ℝ) (g : ℝ → ℝ) :
  (0 ≤ a ∧ a < 1 ∧ ∀ x > 2, g x = (real.exp (x - 2) - a * x + a) / (x - 2)^2
     ∧ ∃ x₀ > 2, g(x₀) = h ∧ ∀ x > 2, g(x) ≥ g(x₀)) ↔ (h ∈ set.Icc (1/2) (real.exp 2 / 4)) :=
sorry

end range_m_range_h_l596_596615


namespace infinite_series_sum_l596_596496

theorem infinite_series_sum :
  (∑' n : ℕ, if (n = 0) then 0 else (n + 1) / (n^2 * (n + 2))) = (3/8) + (Real.pi^2 / 24) :=
by
  sorry

end infinite_series_sum_l596_596496


namespace maximum_possible_value_of_N_l596_596217

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l596_596217


namespace take_home_pay_l596_596718

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l596_596718


namespace systematic_sampling_interval_l596_596376

-- Define the total number of students and sample size
def N : ℕ := 1200
def n : ℕ := 40

-- Define the interval calculation for systematic sampling
def k : ℕ := N / n

-- Prove that the interval k is 30
theorem systematic_sampling_interval : k = 30 := by
sorry

end systematic_sampling_interval_l596_596376


namespace geometry_problem_l596_596290

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem geometry_problem (k p : ℝ) (A B C : ℝ × ℝ) 
  (h1 : midpoint B C = (k, 0))
  (h2 : midpoint A C = (0, p))
  (h3 : midpoint A B = (0, 0)) :
  (let AB_sq := (A.1 - B.1)^2 + (A.2 - B.2)^2,
       AC_sq := (A.1 - C.1)^2 + (A.2 - C.2)^2,
       BC_sq := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
    (AB_sq + AC_sq + BC_sq) / (k^2 + p^2) = 8) :=
by
  sorry

end geometry_problem_l596_596290


namespace millet_more_than_half_l596_596494

def daily_millet (n : ℕ) : ℝ :=
  1 - (0.7)^n

theorem millet_more_than_half (n : ℕ) : daily_millet 2 > 0.5 :=
by {
  sorry
}

end millet_more_than_half_l596_596494


namespace find_x_values_l596_596653

def piecewise_function (x : ℝ) : ℝ :=
if x ≤ 3 then x^2 + 3 else 3 * x

theorem find_x_values (x : ℝ) (y : ℝ) (h : y = 15) :
  (piecewise_function x = y) ↔ (x = -2 * Real.sqrt 3 ∨ x = 5) :=
by
  sorry

end find_x_values_l596_596653


namespace tan_arccot_l596_596024

theorem tan_arccot (x : ℝ) (h : x = 3/5) : Real.tan (Real.arccot x) = 5/3 :=
by 
  sorry

end tan_arccot_l596_596024


namespace man_age_twice_son_age_l596_596434

theorem man_age_twice_son_age (S M : ℕ) (h1 : M = S + 24) (h2 : S = 22) : 
  ∃ Y : ℕ, M + Y = 2 * (S + Y) ∧ Y = 2 :=
by 
  sorry

end man_age_twice_son_age_l596_596434


namespace volume_region_l596_596570

theorem volume_region (x y z : ℝ) (hx : x ≤ 4) (hy : y ≤ 4) (hz : z ≥ 0)
  (h : |x + y + z| + |x + y - z| ≤ 12) :
  volume x y z = 48 :=
sorry

end volume_region_l596_596570


namespace complex_number_negative_y_axis_l596_596652

theorem complex_number_negative_y_axis (a : ℝ) : 
  (∀ z : ℂ, z = (a + complex.i) ^ 2 → z.im < 0) → a = -1 := 
by
  intro h
  have key : ((a + complex.i) ^ 2).re = a^2 - 1 := by sorry
  have imag_term : ((a + complex.i) ^ 2).im = 2 * a := by sorry
  specialize h ((a + complex.i) ^ 2) (by sorry)
  have a_square : a^2 = 1 := by sorry
  have a_negative : a < 0 := by sorry
  linarith

end complex_number_negative_y_axis_l596_596652


namespace area_of_rectangle_l596_596807

-- Definitions and conditions
def side_of_square : ℕ := 50
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 5
def breadth_of_rectangle : ℕ := 10

-- Theorem statement
theorem area_of_rectangle :
  (length_of_rectangle * breadth_of_rectangle = 200) := by
  sorry

end area_of_rectangle_l596_596807


namespace max_value_of_N_l596_596201

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l596_596201


namespace initial_apples_l596_596060

theorem initial_apples (C : ℝ) (h : C + 7.0 = 27) : C = 20.0 := by
  sorry

end initial_apples_l596_596060


namespace smallest_solution_l596_596909

noncomputable def equation (x : ℝ) := x^4 - 40 * x^2 + 400

theorem smallest_solution : ∃ x : ℝ, equation x = 0 ∧ ∀ y : ℝ, equation y = 0 → -2 * Real.sqrt 5 ≤ y :=
by
  sorry

end smallest_solution_l596_596909


namespace Vitya_catches_mother_l596_596869

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l596_596869


namespace Adam_current_money_is_8_l596_596964

variable (Adam_initial : ℕ) (spent_on_game : ℕ) (allowance : ℕ)

def money_left_after_spending (initial : ℕ) (spent : ℕ) := initial - spent
def current_money (money_left : ℕ) (allowance : ℕ) := money_left + allowance

theorem Adam_current_money_is_8 
    (h1 : Adam_initial = 5)
    (h2 : spent_on_game = 2)
    (h3 : allowance = 5) :
    current_money (money_left_after_spending Adam_initial spent_on_game) allowance = 8 := 
by sorry

end Adam_current_money_is_8_l596_596964


namespace part_one_part_two_l596_596609

-- 1. Prove that 1 + 2x^4 >= 2x^3 + x^2 for all real numbers x
theorem part_one (x : ℝ) : 1 + 2 * x^4 ≥ 2 * x^3 + x^2 := sorry

-- 2. Given x + 2y + 3z = 6, prove that x^2 + y^2 + z^2 ≥ 18 / 7
theorem part_two (x y z : ℝ) (h : x + 2 * y + 3 * z = 6) : x^2 + y^2 + z^2 ≥ 18 / 7 := sorry

end part_one_part_two_l596_596609


namespace Vitya_catchup_mom_in_5_l596_596894

variables (s t : ℝ)

-- Defining the initial conditions
def speeds_equal : Prop := 
  ∀ t, (t ≥ 0 ∧ t ≤ 10) → (Vitya_Distance t + Mom_Distance t = 20 * s)

def Vitya_Distance (t : ℝ) : ℝ := 
  if t ≤ 10 then s * t else s * 10 + 5 * s * (t - 10)

def Mom_Distance (t : ℝ) : ℝ := 
  s * t

-- Main theorem
theorem Vitya_catchup_mom_in_5 (s : ℝ) : 
  speeds_equal s → (Vitya_Distance s 15 - Vitya_Distance s 10 = Mom_Distance s 15 - Mom_Distance s 10) :=
by
  sorry

end Vitya_catchup_mom_in_5_l596_596894


namespace triangle_angle_XYZ_l596_596274

theorem triangle_angle_XYZ
  (O : Point) (X Y Z : Triangle)
  (h1 : incenter O XYZ)
  (h2 : angle XYZ = 75°)
  (h3 : angle YZO = 40°) :
  angle YXZ = 25° :=
sorry

end triangle_angle_XYZ_l596_596274


namespace cars_between_black_and_white_l596_596420

theorem cars_between_black_and_white :
  ∀ (n : ℕ), n = 20 →
  ∀ (black : ℕ), black = 16 →
  ∀ (white : ℕ), white = 11 →
  ∃ (cars_between : ℕ), cars_between = 5 := 
begin
  intros n hn black hb white hw,
  use (white - (n - black)),
  have h_black_left : nat.succ (n - black) = 5,
  { rw [<- nat.sub_assoc, nat.sub_self, nat.succ_zero],
    exact nat.sub_le n black },
  have h_cars_between := nat.sub (white - (n - black) - 1) 0,
  rw [nat.add_sub_cancel_left, hw, h_black_left],
  refl,
end

end cars_between_black_and_white_l596_596420


namespace train_length_correct_l596_596949

-- Define the conditions
def bridge_length : ℝ := 180
def train_speed : ℝ := 15
def time_to_cross_bridge : ℝ := 20
def time_to_cross_man : ℝ := 8

-- Define the length of the train
def length_of_train : ℝ := 120

-- Proof statement
theorem train_length_correct :
  (train_speed * time_to_cross_man = length_of_train) ∧
  (train_speed * time_to_cross_bridge = length_of_train + bridge_length) :=
by
  sorry

end train_length_correct_l596_596949


namespace num_real_roots_eq_two_l596_596355

theorem num_real_roots_eq_two : 
  ∀ x : ℝ, (∃ r : ℕ, r = 2 ∧ (abs (x^2 - 1) = 1/10 * (x + 9/10) → x = r)) := sorry

end num_real_roots_eq_two_l596_596355


namespace jebb_take_home_pay_l596_596721

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l596_596721


namespace median_length_squared_l596_596269

theorem median_length_squared 
  (A B C O : Point)
  (AO_is_median : midpoint O B C)
  (m_a : ℝ) (b c a : ℝ)
  (h_m_a: m_a = dist A O)
  (h_b: dist A C = b)
  (h_c: dist A B = c)
  (h_a: dist B C = a) :
  m_a^2 = (1/2) * b^2 + (1/2) * c^2 - (1/4) * a^2 := 
sorry

end median_length_squared_l596_596269


namespace T_depends_on_d_and_n_l596_596297

variable (a d n : ℕ)

def sum_arith_series (k : ℕ) : ℕ := k * (2 * a + (k - 1) * d) / 2

def s1 : ℕ := sum_arith_series n
def s4 : ℕ := sum_arith_series (4 * n)
def s5 : ℕ := sum_arith_series (5 * n)

def T : ℕ := s5 - s4 - s1

theorem T_depends_on_d_and_n : T = 14 * d * n^2 :=
by
  -- proof to be filled in
  sorry

end T_depends_on_d_and_n_l596_596297


namespace projection_vector_l596_596123

variable (a b : V)
variable (c : ℝ)

-- Assumptions based on the problem statement
axiom NormA : ∥a∥ = 2
axiom NormB : ∥b∥ = 3
axiom AngleAB : Real.angleBetween a b = Real.pi * (3/4)  -- 135 = (3/4)π radians

-- Desired projection
theorem projection_vector (a b : V) (NormA : ∥a∥ = 2) (NormB : ∥b∥ = 3) (AngleAB : Real.angleBetween a b = Real.pi * (3/4)) :
  (Real.proj b a) = -Real.sqrt 2 / 3 • b :=
sorry

end projection_vector_l596_596123


namespace product_of_roots_Q_l596_596741

noncomputable def Q (x : ℝ) : ℝ := (x - 5)^4 - 5

theorem product_of_roots_Q :
  (∃ x : ℝ, Q x = 0 ∧ (∃ u : ℝ, u^4 = 5 ∧ x = u + 5)) →
  (∑ (r : ℝ) in (Roots (Q (x))), r) = (620) :=
by
  sorry

end product_of_roots_Q_l596_596741


namespace society_position_assignments_l596_596959

theorem society_position_assignments (n : ℕ) (h : n = 12) : 
  let ways := n * (n-1) * (n-2) * (n-3) * (n-4) in
  ways = 95040 :=
by
  simp only [h],
  have : 12 * 11 * 10 * 9 * 8 = 95040 by norm_num,
  exact this

end society_position_assignments_l596_596959


namespace final_result_after_operations_l596_596425

theorem final_result_after_operations (x : ℝ) (n : ℕ) (h : x ≠ 0) : y = x ^ ((-2) ^ n) := 
sorry

end final_result_after_operations_l596_596425
