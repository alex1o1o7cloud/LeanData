import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Conditional
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Real

namespace trapezoid_AD_equal_l272_272692

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272692


namespace trig_problems_l272_272984

theorem trig_problems
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hcos : cos (α + π / 4) = -3 / 5) :
  tan α = 7 ∧ sin (2 * α) - cos (2 * α) + cos α ^ 2 = 63 / 50 :=
by
  sorry

end trig_problems_l272_272984


namespace angles_in_arithmetic_sequence_max_area_of_triangle_l272_272715

variables (A B C : ℝ) (a b c : ℝ)
axiom triangle_conditions : a * cos C + c * cos A = 2 * b * cos B ∧ b = sqrt 3

theorem angles_in_arithmetic_sequence : A + B + C = π ∧ a * cos C + c * cos A = 2 * b * cos B ∧ b = sqrt 3 → 
  ∃ k : ℝ, A = k - π/3 ∧ B = π/3 ∧ C = k + π/3 :=
by sorry

theorem max_area_of_triangle : a * cos C + c * cos A = 2 * b * cos B ∧ b = sqrt 3 → 
  ∃ max_area : ℝ, max_area = 3*sqrt(3) / 4 :=
by sorry

end angles_in_arithmetic_sequence_max_area_of_triangle_l272_272715


namespace no_such_function_exists_l272_272937

theorem no_such_function_exists (f : ℕ → ℕ) (h : ∀ n, f (f n) = n + 2019) : false :=
sorry

end no_such_function_exists_l272_272937


namespace crayons_lost_or_given_away_l272_272309

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end crayons_lost_or_given_away_l272_272309


namespace total_distance_covered_l272_272040

theorem total_distance_covered (d : ℝ) :
  (d / 5 + d / 10 + d / 15 + d / 20 + d / 25 = 15 / 60) → (5 * d = 375 / 137) :=
by
  intro h
  -- proof will go here
  sorry

end total_distance_covered_l272_272040


namespace equal_areas_of_medians_divided_triangles_l272_272793

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  in (s * (s - a) * (s - b) * (s - c)).sqrt

theorem equal_areas_of_medians_divided_triangles :
  let a := 13.0
  let b := 14.0
  let c := 15.0
  let area_abc := heron_area a b c
  let smaller_triangle_area := area_abc / 6
  in smaller_triangle_area = 14 :=
  sorry

end equal_areas_of_medians_divided_triangles_l272_272793


namespace mckenna_work_hours_l272_272746

theorem mckenna_work_hours:
  ∀ (start_office end_office start_meeting end_meeting work_after_meeting: ℕ),
  start_office = 8 ∧ 
  end_office = 11 ∧ 
  start_meeting = 11 ∧ 
  end_meeting = 13 ∧
  work_after_meeting = 2 →
  (end_office - start_office) + 
  (end_meeting - start_meeting) + 
  work_after_meeting = 7 := 
by
  intros start_office end_office start_meeting end_meeting work_after_meeting
  intro h
  cases h
  cases h_left
  cases h_right
  rw [h_left_left, h_left_right, h_right_left_left, h_right_left_right, h_right_right]
  norm_num

end mckenna_work_hours_l272_272746


namespace trapezoid_AD_value_l272_272686

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272686


namespace max_sundays_in_first_45_days_l272_272824

theorem max_sundays_in_first_45_days : 
  ∃ d : ℕ, (d ≤ 6) ∧ 
  (∀ n : ℕ, (n < 7) → (d + if n < 3 then 1 else 0 = 7)) := 
sorry

end max_sundays_in_first_45_days_l272_272824


namespace probability_bob_wins_l272_272229

theorem probability_bob_wins (P_lose : ℝ) (P_tie : ℝ) (h1 : P_lose = 5/8) (h2 : P_tie = 1/8) :
  (1 - P_lose - P_tie) = 1/4 :=
by
  sorry

end probability_bob_wins_l272_272229


namespace nina_has_9_times_more_reading_homework_l272_272747

theorem nina_has_9_times_more_reading_homework
  (ruby_math_homework : ℕ)
  (ruby_reading_homework : ℕ)
  (nina_total_homework : ℕ)
  (nina_math_homework_factor : ℕ)
  (h1 : ruby_math_homework = 6)
  (h2 : ruby_reading_homework = 2)
  (h3 : nina_total_homework = 48)
  (h4 : nina_math_homework_factor = 4) :
  nina_total_homework - (ruby_math_homework * (nina_math_homework_factor + 1)) = 9 * ruby_reading_homework := by
  sorry

end nina_has_9_times_more_reading_homework_l272_272747


namespace Uncle_Bradley_bills_l272_272818

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end Uncle_Bradley_bills_l272_272818


namespace round_robin_games_l272_272866

theorem round_robin_games (x : ℕ) (h : 45 = (1 / 2) * x * (x - 1)) : (1 / 2) * x * (x - 1) = 45 :=
sorry

end round_robin_games_l272_272866


namespace bad_arrangement_count_l272_272385

theorem bad_arrangement_count:
  let nums := [2, 3, 4, 5, 6]
  let is_bad (arrangement : List ℕ) : Prop :=
    ¬ ∀ n, 2 ≤ n ∧ n ≤ 20 → ∃ subset, subset ⊆ arrangement ∧ subset.sum = n

  ∃! bad_arrangements, bad_arrangements.card = 3 ∧ 
    ∀ arrangement, arrangement ∈ bad_arrangements ↔ is_bad arrangement :=
sorry

end bad_arrangement_count_l272_272385


namespace problem_part1_problem_part2_problem_part3_l272_272626

noncomputable theory

open Real

def f (x : ℝ) := log x
def g (x : ℝ) (m n : ℝ) := m * (x + n) / (x + 1)

theorem problem_part1 {m n : ℝ} (h1 : g 1 m n = 0) (h2 : deriv (λ x => g x m n) 1 = 1) : m = 2 := 
sorry

theorem problem_part2 {m n : ℝ} (h_non_mono : ¬monotonic (λ x => f x - g x m n)) : m - n > 3 := 
sorry

theorem problem_part3 {m : ℝ} (h : ∀ x > 0, abs (f x) > - abs (g x m 1)) : m ≤ 2 := 
sorry

end problem_part1_problem_part2_problem_part3_l272_272626


namespace blu_ray_movies_returned_l272_272877

theorem blu_ray_movies_returned (D B x : ℕ)
  (h1 : D / B = 17 / 4)
  (h2 : D + B = 378)
  (h3 : D / (B - x) = 9 / 2) :
  x = 4 := by
  sorry

end blu_ray_movies_returned_l272_272877


namespace number_of_sections_l272_272723

theorem number_of_sections (pieces_per_section : ℕ) (cost_per_piece : ℕ) (total_cost : ℕ)
  (h1 : pieces_per_section = 30)
  (h2 : cost_per_piece = 2)
  (h3 : total_cost = 480) :
  total_cost / (pieces_per_section * cost_per_piece) = 8 := by
  sorry

end number_of_sections_l272_272723


namespace magician_knows_numbers_l272_272875

def circle_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def valid_selection (A D : ℕ) (picked_by_A : A ∈ circle_numbers) (picked_by_D : D ∈ circle_numbers) : Prop :=
  is_even A ∧ is_even D ∧ A ≠ D

theorem magician_knows_numbers :
  ∃ (A D : ℕ), valid_selection A D (by trivial) (by trivial) ∧ A * D = 120 :=
begin
  have A : ℕ := 10,
  have D : ℕ := 12,
  use [A, D],
  split,
  { split,
    { rw is_even,
      exact rfl },
    { split,
      { rw is_even,
        exact rfl },
      { exact ne_of_lt (by norm_num) } } },
  { simp [A, D] }
end

end magician_knows_numbers_l272_272875


namespace ratio_an_bn_l272_272569

def a_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), (1 : ℚ) / Nat.choose n k
def b_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), (k ^ 2 : ℚ) / Nat.choose n k

theorem ratio_an_bn (n : ℕ) (hn : 0 < n) : a_n n / b_n n = n^2 / 2 := 
by sorry

end ratio_an_bn_l272_272569


namespace seventh_rack_dvds_l272_272004

def rack_dvds : ℕ → ℕ
| 0 => 3
| 1 => 4
| n + 2 => ((rack_dvds (n + 1)) - (rack_dvds n)) * 2 + (rack_dvds (n + 1))

theorem seventh_rack_dvds : rack_dvds 6 = 66 := 
by
  sorry

end seventh_rack_dvds_l272_272004


namespace derivative_y_l272_272555

variable (α x : ℝ)

def y := x * cos α + sin α * log (sin (x - α))

theorem derivative_y :
  deriv (fun x => x * cos α + sin α * log (sin (x - α))) x = (sin x) / (sin (x - α)) :=
by
  sorry

end derivative_y_l272_272555


namespace line_and_circle_separate_l272_272637

section

variable (α β : Real)

def vector_m : Real × Real := (2 * Real.cos α, 2 * Real.sin α)
def vector_n : Real × Real := (3 * Real.cos β, 3 * Real.sin β)
def angle_between_vectors : Real := 60

def line : Real × Real → Real := 
  λ (x y), x * Real.cos α - y * Real.sin α + 1 / 2

def circle_center : Real × Real := (Real.cos β, -Real.sin β)
def circle_radius : Real := Real.sqrt 2 / 2
def circle : Real × Real → Real := 
  λ (x y), (x - Real.cos β) ^ 2 + (y + Real.sin β) ^ 2 - 1 / 2

theorem line_and_circle_separate : 
  ∃ d : Real, d = ((Real.cos α * Real.cos β + Real.sin α * Real.sin β) + 1 / 2) ∧ 
  d > Real.sqrt 2 / 2 :=
sorry

end

end line_and_circle_separate_l272_272637


namespace area_of_square_inscribed_in_ellipse_l272_272490

noncomputable def areaSquareInscribedInEllipse : ℝ :=
  let a := 4
  let b := 8
  let s := (2 * Real.sqrt (b / 3)).toReal in
  (2 * s) ^ 2

theorem area_of_square_inscribed_in_ellipse :
  (areaSquareInscribedInEllipse) = 32 / 3 :=
sorry

end area_of_square_inscribed_in_ellipse_l272_272490


namespace jasper_initial_candies_l272_272749

def initial_candies : ℕ := 537

theorem jasper_initial_candies 
  (x : ℕ) 
  (hx1 : let y1 := x * 3/4 - 3 in y1 * 4/5 - 5 * 5/6 - 2 = 10) 
  (hx2 : let y2 := 4/5 (x * 3/4 - 3) - 5 in y2 * 5/6 - 2 = 10)
  (hx3 : let y3 := (4 * (x - 3) - 3) - 6 = 60) 
  (hx4 : x - 3 - 5 = 62) :
  x = initial_candies := 
sorry

end jasper_initial_candies_l272_272749


namespace number_of_polynomials_l272_272928

-- Define each coefficient and their possible values
def coeffs_valid (b_0 b_1 b_2 b_3 b_4 b_5 : ℕ) : Prop :=
  b_0 ∈ {0, 1, 2} ∧
  b_1 ∈ {0, 1, 2} ∧
  b_2 ∈ {0, 1, 2} ∧
  b_3 ∈ {0, 1, 2} ∧
  b_4 ∈ {0, 1, 2} ∧
  b_5 ∈ {0, 1, 2}

-- Define the polynomial condition for x = 2 to be an integer root
def polynomial_cond (b_0 b_1 b_2 b_3 b_4 b_5 : ℕ) : Prop :=
  2^6 + b_5 * 2^5 + b_4 * 2^4 + b_3 * 2^3 + b_2 * 2^2 + b_1 * 2 + b_0 = 0

-- Define the main theorem
theorem number_of_polynomials : ∃! (n : ℕ), n = 21 ∧ 
  (∃ (b_0 b_1 b_2 b_3 b_4 b_5 : ℕ), coeffs_valid b_0 b_1 b_2 b_3 b_4 b_5 ∧ 
  polynomial_cond b_0 b_1 b_2 b_3 b_4 b_5) := 
by { sorry }

end number_of_polynomials_l272_272928


namespace ratio_of_areas_l272_272449

-- Define the ellipses and their properties
variables {x y k m : ℝ}
def ellipse1 (x y : ℝ) := (x^2 / 4) + (y^2) = 1
def ellipse2 (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1

-- Given point P on the first ellipse
variables {x₀ y₀ : ℝ}
axiom P_on_ellipse1 : ellipse1 x₀ y₀

-- Line y = kx + m passing through P intersects second ellipse at A and B
axiom line_through_P : y₀ = k * x₀ + m
axiom line_intersects_ellipse2_at_A_B :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), ellipse2 x₁ y₁ ∧ ellipse2 x₂ y₂ ∧ 
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m

-- Ray PO intersects the second ellipse at Q
axiom ray_PO_intersects_ellipse2_at_Q : 
  ∃ (λ : ℝ) (x₃ y₃ : ℝ), x₃ = -λ * x₀ ∧ y₃ = -λ * y₀ ∧ ellipse2 x₃ y₃

-- The main theorem to prove
theorem ratio_of_areas : 
  ∃ (λ : ℝ), λ = 4 → 
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    ellipse2 x₁ y₁ ∧  ellipse2 x₂ y₂ ∧ ellipse2 x₃ y₃ ∧
    y₁ = k * x₁ + m ∧  y₂ = k * x₂ + m ∧
    x₃ = -λ * x₀ ∧ y₃ = -λ * y₀ → 
  ∑ (|PQ| / |PO| = λ) :=
λx₀ y₀, sorry

end ratio_of_areas_l272_272449


namespace probability_all_heads_or_tails_l272_272348

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l272_272348


namespace salt_percentage_in_first_solution_l272_272892

variable (S : ℚ)
variable (H : 0 ≤ S ∧ S ≤ 100)  -- percentage constraints

theorem salt_percentage_in_first_solution (h : 0.75 * S / 100 + 7 = 16) : S = 12 :=
by { sorry }

end salt_percentage_in_first_solution_l272_272892


namespace proof_divisible_by_13_l272_272959

theorem proof_divisible_by_13 :
  let a := 9174532
  let b := 119268916
  b = a * 13 →
  119268903 % 13 = 0 :=
by
  intros h
  have h1 : 119268903 = b - 13, from
    calc
      119268903 = 119268916 - 13 : by sorry
  rw [h1]
  have h2 : (b - 13) % 13 = 0, from
    calc
      (b - 13) % 13 = (13 * a - 13) % 13 : by rw [h]
               ... = 13 * (a - 1) % 13 : by sorry
  exact h2

end proof_divisible_by_13_l272_272959


namespace parabola_focus_l272_272944

theorem parabola_focus :
  let a := 4
  let h := -1
  let k := -5
  let vertex_form_focus (a h k : ℝ) := (h, k + 1 / (4 * a))
  ∃ (focus : ℝ × ℝ), focus = vertex_form_focus a h k
  and focus = (-1, -79 / 16) :=
by
  -- Given parabola equation y = 4x^2 + 8x - 1
  let a := 4
  let h := -1
  let k := -5
  let focus := vertex_form_focus a h k
  use focus
  have : vertex_form_focus a h k = (-1, -79 / 16) := by
    -- This would be derived from the steps in the natural language proof
    sorry
  exact this

end parabola_focus_l272_272944


namespace polynomial_product_result_l272_272654

theorem polynomial_product_result
  (k j : ℤ)
  (P : ℤ → ℤ := λ e, 8 * e^2 - 4 * e + k)
  (Q : ℤ → ℤ := λ e, 4 * e^2 + j * e - 9)
  (R : ℤ → ℤ := λ e, 32 * e^4 - 52 * e^3 + 23 * e^2 + 6 * e - 27) :
  ((P * Q) = R → k + j = -7) := sorry

end polynomial_product_result_l272_272654


namespace part_a_l272_272851

theorem part_a (α : ℝ) (n : ℕ) (hα : α > 0) (hn : n > 1) : (1 + α)^n > 1 + n * α :=
sorry

end part_a_l272_272851


namespace tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces_l272_272534

/-- A theorem to prove that a regular tetrahedron can be divided into equal polyhedra,
each with six faces. -/
theorem tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces
  (T : Type) [regular_tetrahedron T] :
  ∃ (P : Type), equal_polyhedra_with_six_faces P :=
sorry

end tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces_l272_272534


namespace wendy_percentage_accounting_related_jobs_l272_272413

noncomputable def wendy_accountant_years : ℝ := 25.5
noncomputable def wendy_accounting_manager_years : ℝ := 15.5 -- Including 6 months as 0.5 years
noncomputable def wendy_financial_consultant_years : ℝ := 10.25 -- Including 3 months as 0.25 years
noncomputable def wendy_tax_advisor_years : ℝ := 4
noncomputable def wendy_lifespan : ℝ := 80

theorem wendy_percentage_accounting_related_jobs :
  ((wendy_accountant_years + wendy_accounting_manager_years + wendy_financial_consultant_years + wendy_tax_advisor_years) / wendy_lifespan) * 100 = 69.0625 :=
by
  sorry

end wendy_percentage_accounting_related_jobs_l272_272413


namespace number_of_sides_of_regular_polygon_l272_272213

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l272_272213


namespace removing_two_elements_mean_8_l272_272644

theorem removing_two_elements_mean_8 :
  {A : Finset ℕ // A ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ A.card = 2 ∧ ((∑ x in ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} \ A), id) / 13) = 8}.card = 7 :=
by
  sorry

end removing_two_elements_mean_8_l272_272644


namespace john_uber_profit_l272_272724

theorem john_uber_profit
  (P0 : ℝ) (T : ℝ) (P : ℝ)
  (hP0 : P0 = 18000)
  (hT : T = 6000)
  (hP : P = 18000) :
  P + (P0 - T) = 30000 :=
by
  sorry

end john_uber_profit_l272_272724


namespace circle_radius_l272_272030

-- Define the problem parameters
def is_square (ABCD : Type) := exists (a : ℝ), true -- this is just a placeholder to indicate ABCD is a square
def is_tangent (circle : Type) (line_segment : Type) := exists (point_of_tangency : Type), true -- placeholder

theorem circle_radius (ABCD : Type) (circle : Type) (R : ℝ) 
  (A B C D : ABCD) (AB AD BC CD : line_segment) :
  is_square ABCD →
  is_tangent circle AB → 
  is_tangent circle AD →
  tangent_segment_length AB 8 →
  tangent_segment_length AD 8 →
  intersection_segment_length BC 4 →
  intersection_segment_length CD 2 →
  R = 10 :=
by
  sorry

end circle_radius_l272_272030


namespace yuna_has_biggest_number_l272_272844

theorem yuna_has_biggest_number (yoongi : ℕ) (jungkook : ℕ) (yuna : ℕ) (hy : yoongi = 7) (hj : jungkook = 6) (hn : yuna = 9) :
  yuna = 9 ∧ yuna > yoongi ∧ yuna > jungkook :=
by 
  sorry

end yuna_has_biggest_number_l272_272844


namespace painter_can_color_black_l272_272036

/-- Given conditions:
  1. The board is a grid of size 2012 × 2013.
  2. The board starts with cells colored in black and white.
  3. The painter starts at a corner cell.
  4. Each time the painter leaves a cell, that cell changes its color.
-/

def painter_to_black (grid : list (list bool)) (start : (nat × nat)) : Prop :=
  -- grid is 2012 x 2013
  grid.length = 2012 ∧ (∀ row, row ∈ grid → row.length = 2013) ∧
  -- start cell is at a corner
  (start = (0, 0) ∨ start = (0, 2012) ∨ start = (2011, 0) ∨ start = (2011, 2012)) ∧
  -- initial condition: grid is colored black and white
  (∀ row col, grid[row][col] = tt ∨ grid[row][col] = ff) ∧
  -- every cell can be black
  (∃ path,  ∀ row col, path -> grid[row][col] = tt)

theorem painter_can_color_black : 
  ∀ (grid : list (list bool)) (start : (nat × nat)), painter_to_black grid start :=
sorry

end painter_can_color_black_l272_272036


namespace average_of_numbers_eq_80x_l272_272768

theorem average_of_numbers_eq_80x (x : ℚ) : (finset.sum (finset.range 150) id + x) / 150 = 80 * x → x = 11175 / 11999 :=
by
  sorry

end average_of_numbers_eq_80x_l272_272768


namespace square_inscribed_in_ellipse_area_l272_272496

theorem square_inscribed_in_ellipse_area :
  (∃ t : ℝ, (t > 0) ∧ (t^2 = 8/3)) →
  ∃ A : ℝ, A = (2 * sqrt (8/3))^2 ∧ A = 32/3 :=
by {
  intro ht,
  cases ht with t ht_props,
  use (2 * sqrt (8 / 3))^2,
  split,
  { 
    -- First part of proof: showing the computed area matches the calculation
    have area_computed : (2 * sqrt (8 / 3))^2 = 4 * (8 / 3),
    { 
      calc (2 * sqrt (8 / 3))^2 = 4 * (sqrt (8 / 3))^2 : by ring
      ... = 4 * (8 / 3) : by rw [Real.sqrt_sq (show 8 / 3 ≥ 0, by norm_num)]
    },
    exact area_computed,
  },
  { 
    -- Second part of proof: showing the area equals 32/3
    have area_value : 4 * (8 / 3) = 32 / 3,
    { 
      calc 4 * (8 / 3) = 32 / 3 : by ring,
    },
    exact area_value,
  }
}

end square_inscribed_in_ellipse_area_l272_272496


namespace line_divides_perimeter_area_in_equal_ratios_through_incenter_l272_272323

noncomputable def dividesEqualRatios (ABC : Triangle) (M N : Point) (incenterABC : Point) : Prop :=
MN.passesThrough incenterABC → 
  dividesPerimeterEqually MN ABC ↔ dividesAreaEqually MN ABC

theorem line_divides_perimeter_area_in_equal_ratios_through_incenter 
  (ABC : Triangle) (M N : Point) (incenterABC : Point) :
  dividesEqualRatios ABC M N incenterABC :=
sorry

end line_divides_perimeter_area_in_equal_ratios_through_incenter_l272_272323


namespace find_width_of_river_l272_272883

theorem find_width_of_river
    (total_distance : ℕ)
    (river_width : ℕ)
    (prob_find_item : ℚ)
    (h1 : total_distance = 500)
    (h2 : prob_find_item = 4/5)
    : river_width = 100 :=
by
    sorry

end find_width_of_river_l272_272883


namespace compound_interest_rate_l272_272010

theorem compound_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Annual interest rate in decimal
  (A2 A3 : ℝ)  -- Amounts after 2 and 3 years
  (h2 : A2 = P * (1 + r)^2)
  (h3 : A3 = P * (1 + r)^3) :
  A2 = 17640 → A3 = 22932 → r = 0.3 := by
  sorry

end compound_interest_rate_l272_272010


namespace gcd_lcm_sum_18_30_45_l272_272282

theorem gcd_lcm_sum_18_30_45 :
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 :=
by
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  have hA : A = 3 := by sorry -- Proof of GCD computation
  have hB : B = 90 := by sorry -- Proof of LCM computation
  rw [hA, hB]
  norm_num

end gcd_lcm_sum_18_30_45_l272_272282


namespace find_A_minus_B_l272_272977

theorem find_A_minus_B (A B a b c : ℝ) (h1 : b = a * tan B) (h2 : A > π / 2) :
  A - B = π / 2 := 
sorry

end find_A_minus_B_l272_272977


namespace f_range_of_a_l272_272931

-- Define the conditions
variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Assume the function definition and given conditions
axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f(x) + f(y) = f(x * y)
axiom f_neg_for_x_gt_1 : ∀ x : ℝ, 1 < x → f(x) < 0
axiom sqrt_inequality : ∀ x y : ℝ, 0 < x → 0 < y → real.sqrt (x^2 + y^2) ≥ a * real.sqrt (x * y)

theorem f_range_of_a : 0 < a ∧ a ≤ real.sqrt 2 :=
sorry

end f_range_of_a_l272_272931


namespace y_intercept_of_line_l272_272097

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (0, 4) = (0, y) :=
by { intro h,
     have y_eq : y = 4,
     { 
       sorry
     },
     have : (0, y) = (0, 4),
     { 
       sorry 
     },
     exact this }

end y_intercept_of_line_l272_272097


namespace derivative_at_pi_div_2_l272_272994

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (x + Real.pi / 2)

theorem derivative_at_pi_div_2 :
  Deriv.deriv f (Real.pi / 2) = - (Real.pi / 2) :=
sorry

end derivative_at_pi_div_2_l272_272994


namespace quadrilateral_prism_properties_l272_272384

theorem quadrilateral_prism_properties :
  ∀ (P : prism), P.is_quadrilateral_prism → P.num_vertices = 8 ∧ P.num_edges = 12 ∧ P.num_faces = 6 :=
by
  intros P h
  sorry

end quadrilateral_prism_properties_l272_272384


namespace correct_equation_l272_272399

theorem correct_equation (x : ℝ) :
  232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l272_272399


namespace magic_square_y_value_l272_272663

noncomputable def magic_square (y a b e : ℕ) : Prop :=
  let row1 := y + 23 + 84
  let row2 := 5 + (y - 79) + (2 * y - 163)
  row1 = row2

theorem magic_square_y_value : ∃ (y : ℕ), (∃ (a b e : ℕ), (magic_square y a b e)) ∧ y = 172 :=
by
  use 172
  -- It is sufficient to show that there exist a, b, e satisfying the given relationship.
  use (172 - 79), (2*172 - 163), 0
  unfold magic_square
  -- Simplifying the definition
  dsimp only
  -- Rewriting with actual values
  rw [nat.add_sub_cancel_left, nat.add_sub_assoc, nat.mul_sub]
  sorry

end magic_square_y_value_l272_272663


namespace tan_alpha_eq_one_seventh_l272_272168

theorem tan_alpha_eq_one_seventh (α : ℝ) (h1 : 0 < α) (h2 : α < π / 4) (h3 : sin (α + π/4) = 4/5) : tan α = 1/7 :=
sorry

end tan_alpha_eq_one_seventh_l272_272168


namespace problem_statement_l272_272734

theorem problem_statement
  (g : ℝ → ℝ)
  (h0 : g 1 = -1)
  (h1 : ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x - g y)) :
  let m := 1 in 
  let t := -1 in 
  m * t = -1 :=
by
  let m := 1
  let t := -1
  show m * t = -1
  calc
    m * t = 1 * (-1) := by rfl
        ... = -1 := by rfl
  sorry

end problem_statement_l272_272734


namespace josiah_total_expenditure_l272_272572

noncomputable def cookies_per_day := 2
noncomputable def cost_per_cookie := 16
noncomputable def days_in_march := 31

theorem josiah_total_expenditure :
  (cookies_per_day * days_in_march * cost_per_cookie) = 992 :=
by sorry

end josiah_total_expenditure_l272_272572


namespace lateral_surface_of_prism_is_parallelogram_l272_272785

-- Definitions based on conditions
def is_right_prism (P : Type) : Prop := sorry
def is_oblique_prism (P : Type) : Prop := sorry
def is_rectangle (S : Type) : Prop := sorry
def is_parallelogram (S : Type) : Prop := sorry
def lateral_surface (P : Type) : Type := sorry

-- Condition 1: The lateral surface of a right prism is a rectangle
axiom right_prism_surface_is_rectangle (P : Type) (h : is_right_prism P) : is_rectangle (lateral_surface P)

-- Condition 2: The lateral surface of an oblique prism can either be a rectangle or a parallelogram
axiom oblique_prism_surface_is_rectangle_or_parallelogram (P : Type) (h : is_oblique_prism P) :
  is_rectangle (lateral_surface P) ∨ is_parallelogram (lateral_surface P)

-- Lean 4 statement for the proof problem
theorem lateral_surface_of_prism_is_parallelogram (P : Type) (p : is_right_prism P ∨ is_oblique_prism P) :
  is_parallelogram (lateral_surface P) :=
by
  sorry

end lateral_surface_of_prism_is_parallelogram_l272_272785


namespace evaluate_expression_l272_272941

theorem evaluate_expression : 
  let expr := (15 / 8) ^ 2
  let ceil_expr := Nat.ceil expr
  let mult_expr := ceil_expr * (21 / 5)
  Nat.floor mult_expr = 16 := by
  sorry

end evaluate_expression_l272_272941


namespace six_coins_all_heads_or_tails_probability_l272_272352

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l272_272352


namespace ella_seventh_test_score_l272_272093

theorem ella_seventh_test_score 
  (scores : Fin 8 → ℤ)
  (h_distinct : ∀ i j, i ≠ j → scores i ≠ scores j)
  (h_range : ∀ i, 88 ≤ scores i ∧ scores i ≤ 97)
  (h_avg_int_after_each_test : ∀ n, 1 ≤ n → n ≤ 8 → (∑ i in Finset.range n, scores ⟨i, sorry⟩) % n = 0)
  (h_eighth_test_score : scores 7 = 90) :
  scores 6 = 95 := sorry

end ella_seventh_test_score_l272_272093


namespace find_a_b_range_of_a_l272_272192

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

-- Problem 1
theorem find_a_b (a b : ℝ) :
  f a 1 = 0 ∧ f a b = 0 ∧ (∀ x, f a x > 0 ↔ x < 1 ∨ x > b) → a = 1 ∧ b = 2 := sorry

-- Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → (0 ≤ a ∧ a < 8/9) := sorry

end find_a_b_range_of_a_l272_272192


namespace value_of_y_l272_272210

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 :=
sorry

end value_of_y_l272_272210


namespace malesWithCollegeDegreesOnly_l272_272227

-- Define the parameters given in the problem
def totalEmployees : ℕ := 180
def totalFemales : ℕ := 110
def employeesWithAdvancedDegrees : ℕ := 90
def employeesWithCollegeDegreesOnly : ℕ := totalEmployees - employeesWithAdvancedDegrees
def femalesWithAdvancedDegrees : ℕ := 55

-- Define the question as a theorem
theorem malesWithCollegeDegreesOnly : 
  totalEmployees = 180 →
  totalFemales = 110 →
  employeesWithAdvancedDegrees = 90 →
  employeesWithCollegeDegreesOnly = 90 →
  femalesWithAdvancedDegrees = 55 →
  ∃ (malesWithCollegeDegreesOnly : ℕ), 
    malesWithCollegeDegreesOnly = 35 := 
by
  intros
  sorry

end malesWithCollegeDegreesOnly_l272_272227


namespace adults_in_milburg_l272_272794

theorem adults_in_milburg :
  ∀ (total_population children : ℕ),
  total_population = 5256 → children = 2987 → total_population - children = 2269 :=
by
  intros total_population children h1 h2
  rw [h1, h2]
  -- 5256 - 2987 = 2269
  sorry

end adults_in_milburg_l272_272794


namespace actual_time_when_watch_reads_5PM_l272_272266

-- Definitions corresponding to given conditions
def set_time_correctly_at : ℕ := 8 * 60 -- Time in minutes (8:00 AM)
def actual_time_10AM : ℕ := 10 * 60 -- Time in minutes (10:00 AM)
def watch_time_at_10AM : ℕ := 9 * 60 + 48 -- Time in minutes (9:48 AM)
def watch_loss_rate (actual_interval watch_interval : ℕ) : ℚ :=
  watch_interval.toRat / actual_interval.toRat
def target_watch_time : ℕ := 17 * 60 -- Time in minutes (5:00 PM)

-- Main theorem statement
theorem actual_time_when_watch_reads_5PM (rate : ℚ) :
  rate = watch_loss_rate (actual_time_10AM - set_time_correctly_at) 
                        (watch_time_at_10AM - set_time_correctly_at) →
  ∃ actual_time_when_watch_reads : ℕ,
    actual_time_when_watch_reads = 10 * 60 + target_watch_time * (rate⁻¹ : ℚ) :=
sorry -- proof is not required

end actual_time_when_watch_reads_5PM_l272_272266


namespace asymptotic_expansion_l272_272106

noncomputable def f (x : ℝ) : ℝ := ∫ t in x..+∞, t⁻¹ * exp (x - t)

def S_n (x : ℝ) (n : ℕ) : ℝ := ∑ m in finset.range (n + 1), (-1)^m * (m.factorial / x^(m + 1))

theorem asymptotic_expansion {x : ℝ} {n : ℕ} (hx : x > 0) : 
  ∃ (R_n : ℝ → ℝ), 
    (∀ (x : ℝ), x > 0 → |f(x) - S_n x n| ≤ R_n x) ∧ 
    (∀ (x : ℝ), x > 2 * n → R_n x < 1 / (2^(n+1) * n^2)) ∧
    (∀ (x : ℝ), lim (λ x, x^n * |f x - S_n x (n - 1)|) = 0) :=
sorry

end asymptotic_expansion_l272_272106


namespace sequence_general_formula_sum_sequence_l272_272601

open BigOperators

theorem sequence_general_formula (S : ℕ+ → ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, 2 * a n = S n + 2) :
  ∀ n : ℕ+, a n = 2 ^ (n : ℕ) :=
begin
  sorry
end

theorem sum_sequence (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) (T : ℕ+ → ℝ) 
  (h₁ : ∀ n : ℕ+, a n = 2 ^ (n : ℕ))
  (h₂ : ∀ n : ℕ+, b n = log 2 (a n)) :
  ∀ n : ℕ+, T n = (∑ i in Finset.range n, 1 / ((b i) * (b (i + 2))) ) = (3 / 4) - (1 / (2 * n + 2)) - (1 / (2 * n + 4)) :=
begin
  sorry
end

end sequence_general_formula_sum_sequence_l272_272601


namespace smallest_sum_of_two_perfect_squares_l272_272377

theorem smallest_sum_of_two_perfect_squares (x y : ℕ) (h : x^2 - y^2 = 143) :
  x + y = 13 ∧ x - y = 11 → x^2 + y^2 = 145 :=
by
  -- Add this placeholder "sorry" to skip the proof, as required.
  sorry

end smallest_sum_of_two_perfect_squares_l272_272377


namespace min_area_after_fold_l272_272051

theorem min_area_after_fold (A : ℝ) (h_A : A = 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) : 
  ∃ (m : ℝ), m = min_area ∧ m = 2 / 3 :=
by
  sorry

end min_area_after_fold_l272_272051


namespace intersection_not_in_first_quadrant_l272_272974

theorem intersection_not_in_first_quadrant (m : ℝ) :
  let x := -((m + 4) / 2)
  let y := (m / 2) - 2
  ¬ (x > 0 ∧ y > 0) :=
by
  let x := -((m + 4) / 2)
  let y := (m / 2) - 2
  simp
  apply or.intro_left
  sorry

end intersection_not_in_first_quadrant_l272_272974


namespace find_t_l272_272197

def vector (α : Type*) := (α × α)

variables (a b : vector ℤ)
variable t : ℤ

-- Definitions from conditions
def a := (1, 2) : vector ℤ
def b := (3, t)

-- Dot product of vectors
def dot_product (u v : vector ℤ) : ℤ :=
  u.1 * v.1 + u.2 * v.2

-- Sum of vectors
def vector_add (u v : vector ℤ) : vector ℤ :=
  (u.1 + v.1, u.2 + v.2)
  
-- The condition
def condition := dot_product (vector_add a b) a = 0

-- Translate condition to what needs to be proven
theorem find_t : condition → t = -4 :=
  sorry

end find_t_l272_272197


namespace probability_all_heads_or_tails_l272_272343

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l272_272343


namespace find_a_l272_272646

theorem find_a (a : ℝ) (h : 2 * a + 3 = -3) : a = -3 := 
by 
  sorry

end find_a_l272_272646


namespace inequality_holds_for_positive_x_l272_272761

theorem inequality_holds_for_positive_x (x : ℝ) (h : x > 0) : 
  x^8 - x^5 - 1/x + 1/(x^4) ≥ 0 := 
sorry

end inequality_holds_for_positive_x_l272_272761


namespace part_a_part_b_l272_272441

-- Part (a) Equivalent Proof Problem
theorem part_a (k : ℤ) : 
  ∃ a b c : ℤ, 3 * k - 2 = a ^ 2 + b ^ 3 + c ^ 3 := 
sorry

-- Part (b) Equivalent Proof Problem
theorem part_b (n : ℤ) : 
  ∃ a b c d : ℤ, n = a ^ 2 + b ^ 3 + c ^ 3 + d ^ 3 := 
sorry

end part_a_part_b_l272_272441


namespace find_AD_l272_272681

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272681


namespace flower_bed_area_l272_272787

noncomputable def area_of_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1/2) * a * b

theorem flower_bed_area : 
  area_of_triangle 6 8 10 (by norm_num) = 24 := 
sorry

end flower_bed_area_l272_272787


namespace number_of_packages_l272_272300

-- Given conditions
def totalMarkers : ℕ := 40
def markersPerPackage : ℕ := 5

-- Theorem: Calculate the number of packages
theorem number_of_packages (totalMarkers: ℕ) (markersPerPackage: ℕ) : totalMarkers / markersPerPackage = 8 :=
by 
  sorry

end number_of_packages_l272_272300


namespace probability_all_heads_or_tails_l272_272346

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l272_272346


namespace sum_flips_equal_1012_l272_272821

theorem sum_flips_equal_1012 :
  let board_length := 2022 in
  let sequence_condition := {a | ∀ i j, i < j → a i < a j ∧ a (board_length - 1) = board_length} in
  let flip (board : Fin board_length → Bool) (pos : Fin board_length) : Fin board_length → Bool :=
    fun idx => if idx.val < pos.val then not (board idx) else board idx in
  let S := fun seq => (List.range board_length).map (fun i => seq i) in
  exists a b : ℕ,
    gcd a b = 1 ∧
    (∑ i in Finset.range board_length, ite (i < board_length) (1 : ℤ) 0) / 2 = a / b ∧
    a + b = 1012 := sorry

end sum_flips_equal_1012_l272_272821


namespace correct_operation_l272_272841

theorem correct_operation :
  (∀ a b : ℝ, (-2 * a * b^2)^3 = -8 * a^3 * b^6) ∧ 
  ¬ (sqrt 3 + sqrt 2 = sqrt 5) ∧ 
  (∀ a b : ℝ, (a + b)^2 ≠ a^2 + b^2) ∧ 
  (- (1 / 2) ^ (-2) ≠ - (1 / 4)) :=
by
  sorry

end correct_operation_l272_272841


namespace smallest_number_with_2020_divisors_l272_272147

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272147


namespace number_of_int_solution_is_one_l272_272081

theorem number_of_int_solution_is_one : 
  let equation := λ x y : ℤ, x^2024 + y^3 = 3 * y in
  (∃! (p : ℤ × ℤ), equation p.1 p.2) :=
by
  let equation := λ x y : ℤ, x^2024 + y^3 = 3 * y
  have h : ∃! (p : ℤ × ℤ), equation p.1 p.2
  { use (0, 0)
    split
    { exact dec_trivial }
    { intros y h
      cases y with x y
      obtain ⟨hx, hy⟩ := congr_arg equation (prod.ext_iff.mp h).1
      finish } }
  exact h

end number_of_int_solution_is_one_l272_272081


namespace hannah_dog_food_l272_272638

def dog_food_consumption : Prop :=
  let dog1 : ℝ := 1.5 * 2
  let dog2 : ℝ := (1.5 * 2) * 1
  let dog3 : ℝ := (dog2 + 2.5) * 3
  let dog4 : ℝ := 1.2 * (dog2 + 2.5) * 2
  let dog5 : ℝ := 0.8 * 1.5 * 4
  let total_food := dog1 + dog2 + dog3 + dog4 + dog5
  total_food = 40.5

theorem hannah_dog_food : dog_food_consumption :=
  sorry

end hannah_dog_food_l272_272638


namespace variance_and_stddev_of_transformed_data_l272_272795

-- Define the variance of the original dataset
variable {α : Type*} [nonempty α] [fintype α]
variable {f : α → ℝ}
variable (S : ℝ) 
hypothesis (h : (∑ i, (f i - (∑ i, f i) / fintype.card α)^2) / fintype.card α = S^2)

-- Define the transformation
variable (k : ℝ)

-- Define the transformed data
noncomputable def transformed_data (x : α → ℝ) : α → ℝ := λ i, k * x i - 5

theorem variance_and_stddev_of_transformed_data :
  (variance (transformed_data f) = k^2 * S^2) ∧ (stddev (transformed_data f) = k * S) := 
by
  sorry

end variance_and_stddev_of_transformed_data_l272_272795


namespace arithmetic_series_sum_l272_272952

theorem arithmetic_series_sum :
  let a1 := 15
  let an := 31.2
  let d := 0.4
  let n := Int.ofNat (1 + Float.toInt 40.5)
  n ≠ 42 → (2 * 42 * (a1 + an)) / 2 = 970.2 := by {
    let a1 := 15
    let an := 31.2
    let d := 0.4
    let n := Int.ofNat (1 + Float.toInt 40.5)
    have n_eq : n = 42 := by sorry,
    have sum_eq := (42 * (15 + 31.2)) = 970.2 := by sorry,
    exact sum_eq
}

end arithmetic_series_sum_l272_272952


namespace variance_half_X_l272_272219

noncomputable def X : ℕ → probability_mass_function ℕ := 
  probability_mass_function.binomial 8 (3/5)

theorem variance_half_X (X : ℕ → ℙ ℕ) (hX : X = probability_mass_function.binomial 8 (3/5)) :
  D (1/2 * X) = 12 / 25 :=
by sorry

end variance_half_X_l272_272219


namespace fewest_occupied_seats_l272_272797

/-- 
If there are 200 seats in a row, then the fewest number of seats that must be occupied so that the next person to be seated must sit next to someone is 50. 
-/
theorem fewest_occupied_seats (seats : ℕ) (h : seats = 200) : ∃ p, p = 50 ∧ (∀ next_seat, next_seat > 50 → sits_next_to p next_seat) :=
by
  sorry


/-- Helper definition: sits_next_to -/
def sits_next_to (p : ℕ) (next_seat : ℕ) : Prop :=
  ∃ j, (next_seat = j + 1 ∨ next_seat = j - 1) ∧ 1 ≤ j ∧ j ≤ p

end fewest_occupied_seats_l272_272797


namespace six_coins_heads_or_tails_probability_l272_272360

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l272_272360


namespace proper_divisors_sum_of_729_l272_272433

theorem proper_divisors_sum_of_729 : (∑ d in {1, 3, 9, 27, 81, 243}, d) = 364 :=
by
  sorry

end proper_divisors_sum_of_729_l272_272433


namespace derivative_at_zero_l272_272593

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x * f' s !(@ Object. Lean. specific))

theorem derivative_at_zero :
  f' 0 = -12 := by
  sorry

end derivative_at_zero_l272_272593


namespace find_AD_l272_272684

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272684


namespace range_of_m_equiv_l272_272157

def f (m x : ℝ) : ℝ := m + 2^x

def locally_odd_function (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ x ∈ set.Icc (-1 : ℝ) 2, f (-x) = -f (x)

def g (m x : ℝ) : ℝ := x^2 + (5 * m + 1) * x + 1

def intersect_x_axis_seconds (m : ℝ) : Prop :=
  ((5 * m + 1)^2 - 4) > 0

variable (m : ℝ)

theorem range_of_m_equiv :
  ((locally_odd_function (f m) (-1)) ∧ intersect_x_axis_seconds m = false) ∧ 
  (locally_odd_function (f m) ∨ intersect_x_axis_seconds m = true) →
  m < -5/4 ∨ -1 < m < -3/5 ∨ m > 1/5 :=
by
  sorry

end range_of_m_equiv_l272_272157


namespace participation_arrangements_l272_272153

def num_students : ℕ := 5
def num_competitions : ℕ := 3
def eligible_dance_students : ℕ := 4

def arrangements_singing : ℕ := num_students
def arrangements_chess : ℕ := num_students
def arrangements_dance : ℕ := eligible_dance_students

def total_arrangements : ℕ := arrangements_singing * arrangements_chess * arrangements_dance

theorem participation_arrangements :
  total_arrangements = 100 := by
  sorry

end participation_arrangements_l272_272153


namespace solve_inequality_l272_272765

theorem solve_inequality (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) → x ≥ -2 :=
by
  intros h
  sorry

end solve_inequality_l272_272765


namespace triangle_combinations_count_l272_272056

open Set

theorem triangle_combinations_count :
  {s : Finset ℕ | s.card = 3 ∧ 
    (∀ x ∈ s, 1 ≤ x ∧ x ≤ 9) ∧ 
    (∀ ⦃a b c⦄, a ∈ s → b ∈ s → c ∈ s → a < b → b < c → a + b > c)}.card = 34 :=
by
  sorry

end triangle_combinations_count_l272_272056


namespace stationery_cost_l272_272501

theorem stationery_cost (cost_per_pencil cost_per_pen : ℕ)
    (boxes : ℕ)
    (pencils_per_box pens_offset : ℕ)
    (total_cost : ℕ) :
    cost_per_pencil = 4 →
    boxes = 15 →
    pencils_per_box = 80 →
    pens_offset = 300 →
    cost_per_pen = 5 →
    total_cost = (boxes * pencils_per_box * cost_per_pencil) +
                 ((2 * (boxes * pencils_per_box + pens_offset)) * cost_per_pen) →
    total_cost = 18300 :=
by
  intros
  sorry

end stationery_cost_l272_272501


namespace min_filtrations_l272_272452

theorem min_filtrations (n : ℕ) (h1 : 1.2 * (0.8^n) ≤ 0.2) : 
  n ≥ 8 :=
sorry

end min_filtrations_l272_272452


namespace max_sundays_in_45_days_l272_272826

theorem max_sundays_in_45_days (days_in_year_start: ∀ year: Nat, [1 ≤ year ∧ year ≤ 365], total_days: Nat) :
  total_days = 45 ->
  (∃sundays : Nat, (∀days_of_week: Nat, days_of_week = 7) ∧ (days_in_year_start 1) -> (sundays ≤ total_days) ∧ (sundays ≤ 7)) :=
by
  sorry

end max_sundays_in_45_days_l272_272826


namespace smallest_n_with_divisors_2020_l272_272141

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272141


namespace absolute_value_properties_not_true_option_l272_272837

theorem absolute_value_properties_not_true_option :
  (|-1.5| = 1.5) ∧
  (|-1.5| = |1.5|) ∧
  (-|1.5| = -|-1.5|) ∧
  ¬(-|-1.5| = -(-1.5)) :=
by
  sorry

end absolute_value_properties_not_true_option_l272_272837


namespace max_net_income_l272_272370

-- Define the conditions
def a := 0.6
def b := 1
def c := 0.8

def total_tickets := a + b + c = 2.4
def product_tickets := a * b = 0.6
def revenue := 3 * a + 5 * b + 8 * c

-- Statement of the proof problem
theorem max_net_income :
  total_tickets → product_tickets → 
  (∀ a b c, total_tickets ∧ product_tickets → revenue ≤ 13.2) ∧ revenue = 13.2 :=
sorry

end max_net_income_l272_272370


namespace measure_angle_EHG_l272_272247

-- Given problem conditions
variable (EFGH : Type) [Parallelogram EFGH]
variable (angleEFG angleFGH : ℝ)
variable (H1 : angleEFG = 2 * angleFGH)
variable (H2 : angleEFG + angleFGH = 180)

-- Conclude the measure of angle EHG
theorem measure_angle_EHG : angleEFG = 120 :=
by
  -- Proof omitted
  sorry

end measure_angle_EHG_l272_272247


namespace find_x_l272_272510

variable {A B C D E : Type}
variable {AD DB BE CE : ℝ}
variable {x : ℝ}

-- Given conditions
def triangle_acute (triangle : Type) (ac: A) (bc: B) : Prop := true 
def segment_lengths (AD DB BE CE : ℝ) (x : ℝ) : Prop :=
  AD = 6 ∧ DB = 4 ∧ CE = 3 ∧ BE = x

-- The condition stating that these segments form an acute triangle
axiom acute_triangle : triangle_acute A B

-- The theorem we are proving
theorem find_x (h : segment_lengths AD DB BE CE x) : 
  x = 31 / 3 := by
  sorry

end find_x_l272_272510


namespace cannot_use_square_of_binomial_l272_272003

theorem cannot_use_square_of_binomial (x y : ℝ) :
  ¬ (∃ a b : ℝ, (-x + y) * (x - y) = (a + b)^2 ∨ (-x + y) * (x - y) = (a - b)^2) := sorry

end cannot_use_square_of_binomial_l272_272003


namespace raised_bed_area_l272_272064

theorem raised_bed_area (length width : ℝ) (total_area tilled_area remaining_area raised_bed_area : ℝ) 
(h_len : length = 220) (h_wid : width = 120)
(h_total_area : total_area = length * width)
(h_tilled_area : tilled_area = total_area / 2)
(h_remaining_area : remaining_area = total_area / 2)
(h_raised_bed_area : raised_bed_area = (2 / 3) * remaining_area) : raised_bed_area = 8800 :=
by
  have h1 : total_area = 220 * 120, from by rw [h_total_area, h_len, h_wid]
  have h2 : tilled_area = 26400 / 2, from by rw [h_tilled_area, h1]
  have h3 : remaining_area = 26400 / 2, from by rw [h_remaining_area, h1]
  have h4 : raised_bed_area = (2 / 3) * 13200, from by rw [h_raised_bed_area, h3]
  have h5 : raised_bed_area = 8800, from by rwa [← h_raised_bed_area, h4]
  exact h5

end raised_bed_area_l272_272064


namespace total_number_of_sheep_l272_272256

theorem total_number_of_sheep (a₁ a₂ a₃ a₄ a₅ a₆ a₇ d : ℤ)
    (h1 : a₂ = a₁ + d)
    (h2 : a₃ = a₁ + 2 * d)
    (h3 : a₄ = a₁ + 3 * d)
    (h4 : a₅ = a₁ + 4 * d)
    (h5 : a₆ = a₁ + 5 * d)
    (h6 : a₇ = a₁ + 6 * d)
    (h_sum : a₁ + a₂ + a₃ = 33)
    (h_seven: 2 * a₂ + 9 = a₇) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 133 := sorry

end total_number_of_sheep_l272_272256


namespace percentage_reduction_in_oil_price_l272_272479

theorem percentage_reduction_in_oil_price (R : ℝ) (P : ℝ) (hR : R = 48) (h_quantity : (800/R) - (800/P) = 5) : 
    ((P - R) / P) * 100 = 30 := 
    sorry

end percentage_reduction_in_oil_price_l272_272479


namespace rectangle_area_l272_272381

-- Define length and width
def width : ℕ := 6
def length : ℕ := 3 * width

-- Define area of the rectangle
def area (length width : ℕ) : ℕ := length * width

-- Statement to prove
theorem rectangle_area : area length width = 108 := by
  sorry

end rectangle_area_l272_272381


namespace square_area_in_ellipse_l272_272483

theorem square_area_in_ellipse (t : ℝ) (ht : 0 < t) :
  (t ^ 2 / 4 + t ^ 2 / 8 = 1) →
  let side_length := 2 * t in
  let area := side_length ^ 2 in
  area = 32 / 3 := 
sorry

end square_area_in_ellipse_l272_272483


namespace sin_675_eq_neg_sqrt2_div_2_l272_272918

axiom angle_reduction (a : ℝ) : (a - 360 * (floor (a / 360))) * π / 180 = a * π / 180 - 2 * π * (floor (a / 360))

theorem sin_675_eq_neg_sqrt2_div_2 : real.sin (675 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1: real.sin (675 * real.pi / 180) = real.sin (315 * real.pi / 180),
  { rw [← angle_reduction 675, show floor(675 / 360:ℝ) = 1 by norm_num, int.cast_one, sub_self, zero_mul, add_zero] },
  have h2: 315 = 360 - 45,
  { norm_num },
  have h3: real.sin (315 * real.pi / 180) = - real.sin (45 * real.pi / 180),
  { rw [eq_sub_of_add_eq $ show real.sin (2 * π - 45 * real.pi / 180) = -real.sin (45 * real.pi / 180) by simp] },
  rw [h1, h3, real.sin_pi_div_four],
  norm_num

end sin_675_eq_neg_sqrt2_div_2_l272_272918


namespace peter_vacation_saving_l272_272313

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l272_272313


namespace committees_share_members_l272_272458

theorem committees_share_members {n m t : ℕ} (h₁ : n = 1600) (h₂ : m = 16000) (h₃ : t = 80)
  (k : Fin n → ℕ) (h₄ : ∑ i, k i = m * t) :
  ∃ (A B : Set (Fin n)), card A ≥ 4 ∧ card B ≥ 4 ∧ card (A ∩ B) ≥ 4 :=
by
  sorry

end committees_share_members_l272_272458


namespace triangle_angle_condition_triangle_parallel_condition_triangle_angle_iff_parallel_l272_272260

variables {α : Type*} [linear_ordered_field α]
variables (A B C M I A1 B1 G : EuclideanGeometry.Point α)
variables (h₁ : EuclideanGeometry.in_triangle A B C)
variables (h₂ : EuclideanGeometry.is_median_intersection M A B C)
variables (h₃ : EuclideanGeometry.is_incenter I A B C)
variables (h₄ : EuclideanGeometry.is_tangent_point A1 (EuclideanGeometry.Side BC))
variables (h₅ : EuclideanGeometry.is_tangent_point B1 (EuclideanGeometry.Side AC))
variables (h₆ : EuclideanGeometry.is_intersection G (EuclideanGeometry.Line AA1) (EuclideanGeometry.Line BB1))

theorem triangle_angle_condition (h_parallel : EuclideanGeometry.parallel_line G M (EuclideanGeometry.Line AB)) :
  EuclideanGeometry.angle_eq (EuclideanGeometry.langle C G I) 90 :=
sorry

theorem triangle_parallel_condition (h_angle : EuclideanGeometry.angle_eq (EuclideanGeometry.langle C G I) 90) :
  EuclideanGeometry.parallel_line G M (EuclideanGeometry.Line AB) :=
sorry

theorem triangle_angle_iff_parallel :
  EuclideanGeometry.angle_eq (EuclideanGeometry.langle C G I) 90 ↔
  EuclideanGeometry.parallel_line G M (EuclideanGeometry.Line AB) :=
⟨triangle_angle_condition, triangle_parallel_condition⟩

end triangle_angle_condition_triangle_parallel_condition_triangle_angle_iff_parallel_l272_272260


namespace distance_covered_l272_272556

def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

theorem distance_covered : 
  let time_in_minutes := 42 in
  let speed_in_kmh := 10 in
  let time_in_hours := minutes_to_hours time_in_minutes in
  let distance := speed_in_kmh * time_in_hours in
  distance = 7 :=
by
  sorry

end distance_covered_l272_272556


namespace smallest_number_with_2020_divisors_l272_272119

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272119


namespace children_left_birthday_l272_272304

theorem children_left_birthday 
  (total_guests : ℕ := 60)
  (women : ℕ := 30)
  (men : ℕ := 15)
  (remaining_guests : ℕ := 50)
  (initial_children : ℕ := total_guests - women - men)
  (men_left : ℕ := men / 3)
  (total_left : ℕ := total_guests - remaining_guests)
  (children_left : ℕ := total_left - men_left) :
  children_left = 5 :=
by
  sorry

end children_left_birthday_l272_272304


namespace rowing_time_to_place_and_back_l272_272876

open Real

/-- Definitions of the problem conditions -/
def rowing_speed_still_water : ℝ := 5
def current_speed : ℝ := 1
def distance_to_place : ℝ := 2.4

/-- Proof statement: the total time taken to row to the place and back is 1 hour -/
theorem rowing_time_to_place_and_back :
  (distance_to_place / (rowing_speed_still_water + current_speed)) + 
  (distance_to_place / (rowing_speed_still_water - current_speed)) =
  1 := by
  sorry

end rowing_time_to_place_and_back_l272_272876


namespace find_AD_l272_272679

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272679


namespace point_on_ellipse_l272_272986

variables {a b x y : ℝ}

def is_ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def is_foci (a b : ℝ) : Prop :=
  ∃ c : ℝ, c = sqrt (a^2 - b^2)

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (a^2 - b^2) / a

theorem point_on_ellipse (a b : ℝ) (θ : ℝ) (e : ℝ) (h_ellipse : is_ellipse a b x y) 
(h_foci: is_foci a b) (he : e = eccentricity a b) (hθ : 0 < θ ∧ θ < π) :
∃ P : ℝ × ℝ, ∃ θ1 θ2 ∈ set.Ioo 0 π, θ1 + θ2 = θ ∧ sin (θ / 2) ≤ e := sorry

end point_on_ellipse_l272_272986


namespace sugar_needed_l272_272888

theorem sugar_needed (sugar_per_24 : ℕ) (muffins_per_24 : ℕ) (maria_muffins : ℕ) (sugar_needed : ℕ) : 
  sugar_per_24 = 3 → muffins_per_24 = 24 → maria_muffins = 72 → sugar_needed = 9 → 
  3 * maria_muffins / muffins_per_24 = sugar_needed :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end sugar_needed_l272_272888


namespace midpoints_form_regular_dodecagon_l272_272176

theorem midpoints_form_regular_dodecagon (A B C D K L M N : Point)
  (h₁ : is_square A B C D)
  (h₂ : is_equilateral_triangle A B K)
  (h₃ : is_equilateral_triangle B C L)
  (h₄ : is_equilateral_triangle C D M)
  (h₅ : is_equilateral_triangle D A N)
  (midpoint : Point → Point → Point) :
  is_regular_dodecagon 
    (midpoint K L) (midpoint L M) (midpoint M N) (midpoint N K)
    (midpoint A K) (midpoint B K) (midpoint B L) (midpoint C L)
    (midpoint C M) (midpoint D M) (midpoint D N) (midpoint A N) := 
sorry

end midpoints_form_regular_dodecagon_l272_272176


namespace bulb_positions_97_100_l272_272805

def bulb_color : ℕ → Prop
| 1 := true  -- First bulb is yellow (Y)
| 2 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 3 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 4 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 5 := true -- Fifth bulb is yellow (Y)
| n := sorry  -- Placeholder for the general case to be defined

-- Condition that among any five consecutive bulbs, exactly two are yellow and exactly three are blue.
def valid_sequence (n : ℕ) : Prop :=
  (bulb_color n ∧ bulb_color (n + 4)) ∧
  (¬bulb_color (n + 1) ∧ ¬bulb_color (n + 2) ∧ ¬bulb_color (n + 3))

theorem bulb_positions_97_100 :
  bulb_color 97 = true ∧ bulb_color 98 = false ∧ bulb_color 99 = false ∧ bulb_color 100 = true :=
by
  -- This is the place where the proof for the sequence constraints and result should be filled in
  sorry

end bulb_positions_97_100_l272_272805


namespace comm_group_cardinality_l272_272729

variable (G : Type) [CommGroup G]
variable (k : ℕ) [Fintype G] [Fintype.card_eq G k]

theorem comm_group_cardinality (x : G) : k • x = 0 := 
  sorry

end comm_group_cardinality_l272_272729


namespace sin_double_angle_tan_double_angle_l272_272019

-- Step 1: Define the first problem in Lean 4.
theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 12 / 13) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (2 * α) = -120 / 169 := 
sorry

-- Step 2: Define the second problem in Lean 4.
theorem tan_double_angle (α : ℝ) (h1 : Real.tan α = 1 / 2) :
  Real.tan (2 * α) = 4 / 3 := 
sorry

end sin_double_angle_tan_double_angle_l272_272019


namespace julie_simple_interest_earned_l272_272725

-- Definitions based on the conditions
def initial_savings : ℝ := 1200
def simple_savings : ℝ := initial_savings / 2
def compound_savings : ℝ := initial_savings / 2
def compound_earned : ℝ := 126
def time_years : ℝ := 2

-- The main theorem to prove
theorem julie_simple_interest_earned :
  ∃ r : ℝ, 
    compound_earned = compound_savings * ((1 + r)^time_years - 1) ∧ 
    simple_savings * r * time_years = 120 :=
by 
  sorry

end julie_simple_interest_earned_l272_272725


namespace averages_and_variances_l272_272460

def scores_A : List ℕ := [82, 82, 79, 95, 87]
def scores_B : List ℕ := [95, 75, 80, 90, 85]

noncomputable def average (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / scores.length

noncomputable def variance (scores : List ℕ) : ℝ :=
  let avg := average scores
  (scores.map (λ s => (s : ℝ - avg) ^ 2)).sum / scores.length

theorem averages_and_variances :
  average scores_A = 85 ∧
  average scores_B = 85 ∧
  variance scores_A = 31.6 ∧
  variance scores_B = 50 ∧
  variance scores_A < variance scores_B :=
by
  sorry

end averages_and_variances_l272_272460


namespace sum_products_digits_three_digits_sum_products_digits_four_digits_l272_272850

-- Statement for part (a)
theorem sum_products_digits_three_digits : 
  (∑ (a : ℕ) in {1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ (b : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ (c : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, a * b * c) = 45 ^ 3 := sorry

-- Statement for part (b)
theorem sum_products_digits_four_digits :
  (∑ (a : ℕ) in {1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ (b : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ (c : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ (d : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, a * b * c * d) = 45 ^ 4 := sorry

end sum_products_digits_three_digits_sum_products_digits_four_digits_l272_272850


namespace problem1_problem2_problem3_l272_272411

-- Define the set of digits
def D : set ℕ := {1, 2, 3, 4}

-- Theorem for Problem 1
theorem problem1 : ∃ S : set (ℕ × ℕ × ℕ), (∀ x ∈ S, x.1 ∈ D ∧ x.2.1 ∈ D ∧ x.2.2 ∈ D) ∧ S.card = 64 :=
by
  sorry

-- Defining conditions for non-repeating digits formation
def non_repeating (a b c : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

-- Theorem for Problem 2
theorem problem2 : ∃ S : set (ℕ × ℕ × ℕ), (∀ x ∈ S, x.1 ∈ D ∧ x.2.1 ∈ D ∧ x.2.2 ∈ D ∧ non_repeating x.1 x.2.1 x.2.2) ∧ S.card = 24 :=
by
  sorry

-- Defining conditions for ordered non-repeating digits formation
def ordered_non_repeating (a b c : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a > b ∧ b > c

-- Theorem for Problem 3
theorem problem3 : ∃ S : set (ℕ × ℕ × ℕ), (∀ x ∈ S, x.1 ∈ D ∧ x.2.1 ∈ D ∧ x.2.2 ∈ D ∧ ordered_non_repeating x.1 x.2.1 x.2.2) ∧ S.card = 4 :=
by
  sorry

end problem1_problem2_problem3_l272_272411


namespace tan_alpha_value_l272_272162

theorem tan_alpha_value (α : ℝ) (h : (sin α - 2 * cos α) / (3 * sin α + 5 * cos α) = -5) 
(h1 : cos α ≠ 0) :
  tan α = -23 / 16 :=
sorry

end tan_alpha_value_l272_272162


namespace probability_even_product_l272_272938

-- Conditions
def box := {1, 2, 4}
def draw := List.product box box box

-- The Problem: Prove the probability that the product of numbers on three drawn chips is even equals 26/27.
theorem probability_even_product :
  let total_outcomes := 27
  let favorable_outcomes := 26
  favorable_outcomes / total_outcomes = (26/27 : ℚ) :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end probability_even_product_l272_272938


namespace tan_addition_identity_l272_272165

theorem tan_addition_identity (α β : ℝ) (h : 2 * tan α = 3 * tan β) :
  tan (α + β) = 5 * sin (2 * β) / (5 * cos (2 * β) - 1) :=
by
  sorry

end tan_addition_identity_l272_272165


namespace log_sum_correct_l272_272084

theorem log_sum_correct : log 3 9 + log 3 27 = 5 := 
sorry

end log_sum_correct_l272_272084


namespace transformed_function_equivalency_l272_272110

-- Definitions of transformations
def reflect_about_y_equals_x (f : ℝ → ℝ) : ℝ → ℝ := fun x =>
  let y := f x
  y

def reflect_about_origin (f : ℝ → ℝ) : ℝ → ℝ := fun x =>
  let y := f x
  -y

def translate_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x =>
  let y := f (x + shift)
  y

-- The original function y = -1 - 2^x
def f (x : ℝ) : ℝ := -1 - 2^x

-- The transformation definitions
def c1 := reflect_about_y_equals_x f
def c2 := reflect_about_origin c1
def c3 := translate_left c2 1

-- The target final function
def target_c3 (x : ℝ) : ℝ := Math.log2 (- (1/x))

-- The equivalence statement we want to prove
theorem transformed_function_equivalency : ∀ x, c3 x = target_c3 x := by
  sorry

end transformed_function_equivalency_l272_272110


namespace proof_a_value_l272_272650

noncomputable def a_value : ℝ :=
  classical.some (exists_a (λ (a : ℝ), ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - ax - 2) ≥ 0))

theorem proof_a_value :
  ∃ a : ℝ, (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - ax - 2) ≥ 0) ∧ a = 1 :=
begin
  use 1,
  split,
  { intros x hx,
    have h1 : x - 1 + 2 = x + 1 := by linarith,
    have h2 : x^2 - x - 2 = (x - 2) * (x + 1) := by ring,
    rw [h1, h2],
    apply mul_nonneg; linarith [hx] },
  { refl }
end

end proof_a_value_l272_272650


namespace cylinder_volume_l272_272953

-- Define the conditions
def length : ℝ := 16
def width : ℝ := 8

-- Define the radius and height based on the conditions
def radius : ℝ := width / 2
def height : ℝ := length

-- Statement that we need to prove
theorem cylinder_volume : 
  (∀ length width : ℝ, radius = width / 2 ∧ height = length → 
  π * radius^2 * height = 256 * π) :=
by
  intros length width h
  sorry

end cylinder_volume_l272_272953


namespace measure_of_angle_ehg_l272_272245

open Real

noncomputable def measure_angle_of_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : ℝ :=
  if h : FGH = 60 then 120 else sorry

theorem measure_of_angle_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : 
  ∃ (EHG : ℝ), EHG = 120 :=
begin
  use 120,
  sorry
end

end measure_of_angle_ehg_l272_272245


namespace sum_of_absolute_values_of_binomial_coefficients_l272_272167

theorem sum_of_absolute_values_of_binomial_coefficients :
  let a := λ k, if k % 2 = 0 then Nat.choose 8 k else -Nat.choose 8 k in
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8|) = 256 :=
by
  sorry

end sum_of_absolute_values_of_binomial_coefficients_l272_272167


namespace volume_of_sphere_formula_l272_272518

noncomputable def volume_of_sphere (R : ℝ) : ℝ :=
  ∫ x in -R..R, π * (R^2 - x^2)

theorem volume_of_sphere_formula (R : ℝ) :
  volume_of_sphere R = (4 * π * R^3) / 3 :=
by
  sorry

end volume_of_sphere_formula_l272_272518


namespace BD_squared_correct_l272_272041

noncomputable def BD_squared (AB BC CD DA : ℝ) (h1 : AB = 12) (h2: BC = 9) (h3: CD = 20) (h4: DA = 25) (h5 : ∃ A B C D : ℝ × ℝ, ∠ABC = 90) : ℝ :=
  let AC := Real.sqrt (AB^2 + BC^2) in
  let sin_theta := AB / AC in
  let cos_theta := BC / AC in
  let cos_BCD := -sin_theta in
  let BD_squared := BC^2 + CD^2 - 2 * BC * CD * cos_BCD in
  BD_squared

theorem BD_squared_correct (h1 : AB = 12) (h2: BC = 9) (h3: CD = 20) (h4: DA = 25) (h5 : ∃ A B C D : ℝ × ℝ, ∠ABC = 90) : BD_squared 12 9 20 25 h1 h2 h3 h4 h5 = 769 := sorry

end BD_squared_correct_l272_272041


namespace range_of_a_l272_272617

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

theorem range_of_a {a : ℝ} : is_monotonic (f a) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l272_272617


namespace least_real_number_K_l272_272571

theorem least_real_number_K (x y z K : ℝ) (h_cond1 : -2 ≤ x ∧ x ≤ 2) (h_cond2 : -2 ≤ y ∧ y ≤ 2) (h_cond3 : -2 ≤ z ∧ z ≤ 2) (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  (∀ x y z : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2 ∧ x^2 + y^2 + z^2 + x * y * z = 4 → z * (x * z + y * z + y) / (x * y + y^2 + z^2 + 1) ≤ K) → K = 4 / 3 :=
by
  sorry

end least_real_number_K_l272_272571


namespace pairs_solution_l272_272932

theorem pairs_solution (x y : ℝ) :
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := 
by
  sorry

end pairs_solution_l272_272932


namespace new_weight_l272_272770

-- Conditions
def avg_weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase
def weight_replacement (initial_weight : ℝ) (total_increase : ℝ) : ℝ := initial_weight + total_increase

-- Problem Statement: Proving the weight of the new person
theorem new_weight {n : ℕ} {avg_increase initial_weight W : ℝ} 
  (h_n : n = 8) (h_avg_increase : avg_increase = 2.5) (h_initial_weight : initial_weight = 65) (h_W : W = 85) :
  weight_replacement initial_weight (avg_weight_increase n avg_increase) = W :=
by 
  rw [h_n, h_avg_increase, h_initial_weight, h_W]
  sorry

end new_weight_l272_272770


namespace arithmetic_sequence_a11_l272_272604

theorem arithmetic_sequence_a11 (a : ℕ → ℤ) (h_arithmetic : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_a3 : a 3 = 4) (h_a5 : a 5 = 8) : a 11 = 12 :=
by
  sorry

end arithmetic_sequence_a11_l272_272604


namespace length_of_train_l272_272048

/-- Given the conditions:
  1. The train is running at a speed of 72 kmph.
  2. The train crosses a platform in 20 seconds.
  3. The length of the platform is approximately 150.03 meters,
  prove that the length of the train is approximately 249.97 meters.
-/
noncomputable def train_length (speed_kmph time_sec platform_length : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) * time_sec - platform_length

theorem length_of_train : train_length 72 20 150.03 ≈ 249.97 := by
  -- Definitions from the problem
  let speed_mps := 72 * (1000 / 3600)
  let distance_covered := speed_mps * 20
  let train_length := distance_covered - 150.03
  -- Conclude the theorem
  show train_length ≈ 249.97 from sorry

end length_of_train_l272_272048


namespace germs_per_dish_l272_272253

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end germs_per_dish_l272_272253


namespace find_digits_l272_272791

theorem find_digits (a b c d : ℕ) 
  (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : 0 ≤ c ∧ c ≤ 9)
  (h₃ : 0 ≤ d ∧ d ≤ 9)
  (h₄ : (10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) :
  1000 * a + 100 * b + 10 * c + d = 2315 :=
by
  sorry

end find_digits_l272_272791


namespace measure_angle_EHG_l272_272248

-- Given problem conditions
variable (EFGH : Type) [Parallelogram EFGH]
variable (angleEFG angleFGH : ℝ)
variable (H1 : angleEFG = 2 * angleFGH)
variable (H2 : angleEFG + angleFGH = 180)

-- Conclude the measure of angle EHG
theorem measure_angle_EHG : angleEFG = 120 :=
by
  -- Proof omitted
  sorry

end measure_angle_EHG_l272_272248


namespace part1_range_of_m_part2_min_value_h_part3_no_real_m_n_l272_272992

noncomputable def f (x : ℝ) : ℝ := (1 / 3)^x
noncomputable def g (x : ℝ) : ℝ := -Math.logb 3 x

noncomputable def h (a x : ℝ) : ℝ :=
  let t := (1 / 3)^x
  t^2 - 2 * a * t + 3

theorem part1_range_of_m (m : ℝ) : g (m * x^2 + 2 * x + 1) = ℝ → 1 < m :=
sorry

theorem part2_min_value_h (a : ℝ) (x : ℝ) (hx : x ∈ Icc (-1) 1) :
  h a x = Icc (min (λ t, (t - a)^2 + 3 - a^2 [if a ≤ 1/3 then (28 - 6 * a) / 9, -a*a + 3, -6 * a + 12)) :=
sorry

theorem part3_no_real_m_n (m n : ℝ) (hmn : m > n ∧ n > 3) :
  ¬ (∃ n m, (∀ x ∈ Icc n m, h x = -6x + 12 ∧ Icc n m ⊆ range (λ x, x^2))) :=
sorry

end part1_range_of_m_part2_min_value_h_part3_no_real_m_n_l272_272992


namespace complex_square_eq_l272_272608

theorem complex_square_eq (a b : ℝ) (h : (a + b * complex.I)^2 = 3 + 4 * complex.I) : 
  a^2 + b^2 = 5 ∧ a * b = 2 :=
by
  sorry

end complex_square_eq_l272_272608


namespace find_AD_l272_272711

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272711


namespace general_term_formula_sum_reverse_formula_valid_values_n_l272_272602

-- Given a sequence {a_n}, S_n denotes the sum of its first n terms, and it is known that S_n = n(n+1)/2.
constants {a_n S_n T_n R_n : ℕ → ℕ}
axiom h1 : ∀ n, S_n = n * (n + 1) / 2

-- (1) Prove that the general term formula a_n = n
theorem general_term_formula (n : ℕ) : a_n = n :=
sorry

-- (2) Prove that when n ≥ 2 and n ∈ ℕ*, R_{n-1} = n(T_n - 1)
theorem sum_reverse_formula (n : ℕ) (h2 : n ≥ 2) : R_{n-1} = n * T_n - n :=
sorry

-- (3) Prove that the only values of n that satisfy 3^n + 4^n + ... + (n+2)^n = (a_n+3)^a_n are n = 2 and n = 3
theorem valid_values_n (n : ℕ) (h3 : (3^n + 4^n + ... + (n+2)^n) = (n+3)^n) : n = 2 ∨ n = 3 :=
sorry

end general_term_formula_sum_reverse_formula_valid_values_n_l272_272602


namespace compute_expression_l272_272526

theorem compute_expression :
  let t := real.tan (real.pi / 4) ^ 2
  let c := real.cos (real.pi / 4) ^ 2
  t = 1 →
  c = 1 / 2 →
  (t - c) / (t * c) = 1 :=
by
  intros t_eq c_eq
  sorry

end compute_expression_l272_272526


namespace total_arrangements_l272_272903

-- Defining players
inductive Player :=
| A | B | C | D

-- Total matches are 5
def totalMatches := 5

-- Conditions for doubles match and no consecutive matches.
def isValidArrangement (arr : List Player) : Prop :=
  arr.length = totalMatches ∧
  (arr.nth 2 ≠ some Player.A) ∧ -- Player A not in doubles match
  (arr.nth 0 ≠ arr.nth 1) ∧
  (arr.nth 3 ≠ arr.nth 4) ∧
  -- Each player plays exactly two matches
  (List.count arr Player.A = 2) ∧ 
  (List.count arr Player.B = 2) ∧
  (List.count arr Player.C = 2) ∧
  (List.count arr Player.D = 2)

-- Defining the proof statement
theorem total_arrangements : ∃ (arrangements : List (List Player)), 
  (∀ arr ∈ arrangements, isValidArrangement arr) ∧ arrangements.length = 48 := 
sorry

end total_arrangements_l272_272903


namespace fourth_power_square_prime_l272_272001

noncomputable def fourth_smallest_prime := 7

theorem fourth_power_square_prime :
  (fourth_smallest_prime ^ 2) ^ 4 = 5764801 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end fourth_power_square_prime_l272_272001


namespace expression1_eq_expression2_eq_l272_272152

-- Problem 1: Prove that the given expression evaluates to 8
theorem expression1_eq : 2^(-1) + 16^(3/4) - (Real.sqrt 2 + 1)^0 + log 3 (Real.sqrt 3) = 8 := by
  sorry

-- Problem 2: Prove that the given expression evaluates to 3
theorem expression2_eq : (Real.log 5)^2 - (Real.log 2)^2 + 2*Real.log 2 + log 2 9 * log 3 2 = 3 := by
  sorry

end expression1_eq_expression2_eq_l272_272152


namespace intersection_M_N_l272_272293

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | ∃ k : ℕ, x = 2 * k}

theorem intersection_M_N :
  M ∩ N = {2, 4, 8} :=
by sorry

end intersection_M_N_l272_272293


namespace problem_expression_l272_272073

theorem problem_expression : 
  (1 / 4 : ℝ)⁻¹ + | - real.sqrt 3 | - (real.pi - 3)^0 + 3 * real.tan (real.pi / 6) = 3 + 2 * real.sqrt 3 :=
by sorry

end problem_expression_l272_272073


namespace smaller_angle_at_7_30_l272_272416

def clock_angle_deg_per_hour : ℝ := 30 

def minute_hand_angle_at_7_30 : ℝ := 180

def hour_hand_angle_at_7_30 : ℝ := 225

theorem smaller_angle_at_7_30 : 
  ∃ angle : ℝ, angle = 45 ∧ 
  (angle = |hour_hand_angle_at_7_30 - minute_hand_angle_at_7_30|) :=
begin
  sorry
end

end smaller_angle_at_7_30_l272_272416


namespace smallest_number_with_2020_divisors_l272_272142

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272142


namespace six_coins_heads_or_tails_probability_l272_272362

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l272_272362


namespace european_stamp_costs_l272_272269

theorem european_stamp_costs :
  let P_Italy := 0.07
  let P_Germany := 0.03
  let N_Italy := 9
  let N_Germany := 15
  N_Italy * P_Italy + N_Germany * P_Germany = 1.08 :=
by
  sorry

end european_stamp_costs_l272_272269


namespace max_cylinder_volume_l272_272828

theorem max_cylinder_volume (A V : ℝ) (r h : ℝ) (π : Real.pi) :
  (A = 2 * π * r^2 + 2 * π * r * h) →
  (A = 1) →
  (V = π * r^2 * h) →
  (r = 1 / (sqrt (6 * π))) ∧ (h = 1 / (sqrt (6 * π))) :=
  sorry

end max_cylinder_volume_l272_272828


namespace Uncle_Bradley_bills_l272_272819

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end Uncle_Bradley_bills_l272_272819


namespace MK_less_than_half_AD_plus_BC_l272_272250

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]
variables (A B C D M K : V)
variable (midpoint : V → V → V)
variable (dist : V → V → ℝ)

-- Given Conditions
def AB_CD_not_in_same_plane : Prop := ¬ ∃ (u v : ℝ), ∀ (t : ℝ), A + t * u = C + t * v

def M_midpoint_AB : Prop := M = midpoint A B

def K_midpoint_CD : Prop := K = midpoint C D

-- Theorem to prove
theorem MK_less_than_half_AD_plus_BC
  (h1 : AB_CD_not_in_same_plane A B C D)
  (h2 : M_midpoint_AB A M B)
  (h3 : K_midpoint_CD C K D) :
  dist M K < (1 / 2) * (dist A D + dist B C) :=
sorry

end MK_less_than_half_AD_plus_BC_l272_272250


namespace trapezoid_AD_CF_BC_CE_l272_272702

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272702


namespace cos_B_value_l272_272659

theorem cos_B_value (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 6) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  in cos_B = 29 / 36 :=
by
  sorry

end cos_B_value_l272_272659


namespace probability_mia_meets_bus_l272_272463

open ProbabilityTheory

-- Definitions for the problem conditions
def bus_arrival := Uniform 0 60
def mia_arrival := Uniform 0 60
def bus_wait_time := 15

-- Define events for bus and Mia arrival and waiting period
def bus_at_stop (y : ℝ) := Ioc y (y + bus_wait_time)
def mia_in_time (x : ℝ) (y : ℝ) := x ∈ bus_at_stop y

-- The theorem statement
theorem probability_mia_meets_bus : 
  ∀ (x y : ℝ), x ∈ Icc 0 60 → y ∈ Icc 0 45 → 
  (∃ x ∈ Icc 0 60, x ∈ bus_at_stop y) ↔ 
  (62.5 / 3600) = (25 / 128) := 
begin
  sorry,
end

end probability_mia_meets_bus_l272_272463


namespace coordinates_of_A_l272_272613

theorem coordinates_of_A 
  (a : ℝ)
  (h1 : (a - 1) = 3 + (3 * a - 2)) :
  (a - 1, 3 * a - 2) = (-2, -5) :=
by
  sorry

end coordinates_of_A_l272_272613


namespace sin_675_eq_neg_sqrt2_over_2_l272_272920

theorem sin_675_eq_neg_sqrt2_over_2 :
  sin (675 * Real.pi / 180) = - (Real.sqrt 2 / 2) := 
by
  -- problem states that 675° reduces to 315°
  have h₁ : (675 : ℝ) ≡ 315 [MOD 360], by norm_num,
  
  -- recognize 315° as 360° - 45°
  have h₂ : (315 : ℝ) = 360 - 45, by norm_num,

  -- in the fourth quadrant, sin(315°) = -sin(45°)
  have h₃ : sin (315 * Real.pi / 180) = - (sin (45 * Real.pi / 180)), by
    rw [Real.sin_angle_sub_eq_sin_add, Real.sin_angle_eq_sin_add],
    
  -- sin(45°) = sqrt(2)/2
  have h₄ : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by
    -- As an assumed known truth for this problem
    exact Real.sin_pos_of_angle,

  -- combine above facts
  rw [h₃, h₄],
  norm_num
  -- sorry is needed if proof steps aren't complete
  sorry

end sin_675_eq_neg_sqrt2_over_2_l272_272920


namespace sin_675_eq_neg_sqrt2_div_2_l272_272922

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l272_272922


namespace raised_bed_area_l272_272065

theorem raised_bed_area (length width : ℝ) (total_area tilled_area remaining_area raised_bed_area : ℝ) 
(h_len : length = 220) (h_wid : width = 120)
(h_total_area : total_area = length * width)
(h_tilled_area : tilled_area = total_area / 2)
(h_remaining_area : remaining_area = total_area / 2)
(h_raised_bed_area : raised_bed_area = (2 / 3) * remaining_area) : raised_bed_area = 8800 :=
by
  have h1 : total_area = 220 * 120, from by rw [h_total_area, h_len, h_wid]
  have h2 : tilled_area = 26400 / 2, from by rw [h_tilled_area, h1]
  have h3 : remaining_area = 26400 / 2, from by rw [h_remaining_area, h1]
  have h4 : raised_bed_area = (2 / 3) * 13200, from by rw [h_raised_bed_area, h3]
  have h5 : raised_bed_area = 8800, from by rwa [← h_raised_bed_area, h4]
  exact h5

end raised_bed_area_l272_272065


namespace polygon_has_more_than_56_vertices_l272_272801

theorem polygon_has_more_than_56_vertices :
  ∀ (n : ℕ) (polygons : ℕ → List (ℝ × ℝ)),
    n = 57 →
    (∀ i, i < n → (∃ l, length l = 57 ∧ polygons i = l)) →
    (∀ rope : List (ℝ × ℝ), (∃ vertices, length vertices > 56 ∧ is_taut_enclosure vertices polygons rope)) →
    True :=
by 
  intros n polygons h_n h_polygons rope h_rope 
  sorry

end polygon_has_more_than_56_vertices_l272_272801


namespace line_tangent_circle_not_second_quadrant_find_line_l2_l272_272965

-- Define the line l with the conditions given
def line_l (a : ℝ) : Affine ℝ :=
  {x | a * x - y - 4 = 0}

-- Define the circle
def circle : Affine ℝ :=
  {p | (p.x)^2 + (p.y - 1)^2 = 5}

-- Define the line l_1 with the conditions given
def line_l1 : Affine ℝ :=
  {p | 2 * p.x - p.y - 7 = 0}

-- Define the line l_2 with the conditions given
def line_l2 : Affine ℝ :=
  {p | 2 * p.x + p.y - 9 = 0}

theorem line_tangent_circle_not_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), a * x - y - 4 = 0 → ¬((x < 0) ∧ (y > 0)))
  → ∀ (x y : ℝ), (x^2 + (y-1)^2 = 5 ∧ a * x - y - 4 = 0) 
  → a = 2 ∧ ∀ (x y : ℝ), 2 * x - y - 4 = 0 :=
sorry

theorem find_line_l2 :
  ∀ (p : Affine ℝ), l1 3 (-1) && l2 -symm_p_parallel_line1 (2*x - y - 7 = 0) 
  → ∀ (p : Affine ℝ), y - 1 = -2 * (x - 4) 
  → ∀ (p : Affine ℝ), 2*x + y - 9 = 0 :=
sorry

end line_tangent_circle_not_second_quadrant_find_line_l2_l272_272965


namespace smallest_four_digit_divisible_by_45_l272_272428

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def divisible_by_45 (n : ℕ) : Prop := n % 45 = 0
def last_digit_zero (n : ℕ) : Prop := n % 10 = 0
def sum_of_digits_divisible_by_9 (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.foldr (· + ·) 0 % 9 = 0

theorem smallest_four_digit_divisible_by_45 : ∃ n : ℕ, is_four_digit n ∧ divisible_by_45 n ∧ last_digit_zero n ∧ sum_of_digits_divisible_by_9 n ∧ 
  ∀ m : ℕ, is_four_digit m ∧ divisible_by_45 m ∧ last_digit_zero m ∧ sum_of_digits_divisible_by_9 m → n ≤ m :=
  ∃ n : ℕ, n = 1008 ∧ is_four_digit n ∧ divisible_by_45 n ∧ last_digit_zero n ∧ sum_of_digits_divisible_by_9 n ∧ 
    ∀ m : ℕ, is_four_digit m ∧ divisible_by_45 m ∧ last_digit_zero m ∧ sum_of_digits_divisible_by_9 m → n ≤ m :=
begin
  sorry
end

end smallest_four_digit_divisible_by_45_l272_272428


namespace smallest_number_with_2020_divisors_l272_272146

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272146


namespace total_handshakes_l272_272233

theorem total_handshakes (ages : Finset ℕ) 
  (h_ages : ages = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
  (even_condition : ∀ (age ∈ ages), age % 2 = 0 → 
    ∀ (other ∈ ages), other = age + 2 ∨ other = age - 2 → (∃ (x, y : ℕ), (age = x ∧ other = y)))
  (odd_condition : ∀ (age ∈ ages), age % 2 ≠ 0 → 
    ∀ (other ∈ ages), other = age + 1 ∨ other = age - 1 → (∃ (x, y : ℕ), (age = x ∧ other = y))) :
  ∃ n, n = 26 := 
sorry

end total_handshakes_l272_272233


namespace trapezoid_AD_value_l272_272687

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272687


namespace part1_part2_i_part2_ii_l272_272623

variable (f : ℝ → ℝ)
variable (k : ℝ)

def function_f (x : ℝ) : ℝ := log x - k * (x - 1) / (x + 1)

theorem part1 (x : ℝ) (hpos : x > 0) (hk : k = 2) : 
  ∃ y > 0, ∀ x > y, (function_f 2) x > 0 :=
sorry

theorem part2_i (x : ℝ) (hpos : x > 1) : 
  k ≤ 2 → (function_f k) x > 0 :=
sorry

theorem part2_ii (n : ℕ) (hn : n > 0) : 
  (∑ i in range (2 * n), 1 / (n + 1 + i)) < log 2 :=
sorry

end part1_part2_i_part2_ii_l272_272623


namespace equal_segments_if_and_only_if_equilateral_l272_272972

theorem equal_segments_if_and_only_if_equilateral (A B C : Point)
  (A' B' C' N P Q R S M : Point) 
  (hA' : midpoint_of_arc BC' A' A)
  (hB' : midpoint_of_arc CA' B' B)
  (hC' : midpoint_of_arc AB' C' C)
  (hN : intersection A' B' N)
  (hP : intersection B' C' P)
  (hQ : intersection C' A' Q)
  (hR : intersection A' B' R)
  (hS : intersection B' C' S)
  (hM : intersection C' A' M)
  : (MN = PQ ∧ PQ = RS) ↔ (equilateral ABC) := 
sorry

end equal_segments_if_and_only_if_equilateral_l272_272972


namespace parabola_equation_coordinates_A_l272_272981

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

def line (x y : ℝ) := x + 2 * y - 2 = 0

def parabola (p y : ℝ) : ℝ := x ^ 2 = 2 * p * y

theorem parabola_equation (p : ℝ) (h₀ : parabola_focus p ∈ {p | line 0 1}) :
  parabola 2 y = x ^ 2 / 4 y :=
sorry

theorem coordinates_A (m : ℝ) (area : ℝ) (C : set (ℝ × ℝ)) 
  (h₀ : A ∈ C)
  (h₁ : A ∈ {p | ∃ k₁ k₂, k₁ * k₂ = -1 ∧ (k₁ + k₂ = m)})
  (h₂ : |m - (-m)| * 1/2 = area) :
  A = (1, -1) ∨ A = (-1, -1) :=
sorry

end parabola_equation_coordinates_A_l272_272981


namespace sum_of_ratios_is_integer_l272_272307

theorem sum_of_ratios_is_integer (a b c d e : ℕ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ d) (h₃ : d ≠ e) (h₄ : e ≠ a) 
  (h₅ : a ≠ c) (h₆ : a ≠ d) (h₇ : a ≠ e) (h₈ : b ≠ d) (h₉ : b ≠ e) (h₁₀ : c ≠ e) :=
  ∃ k : ℤ, (a / b + b / c + c / d + d / e + e / a : ℚ) = k := 
sorry

end sum_of_ratios_is_integer_l272_272307


namespace arc_length_of_curve_l272_272014

noncomputable def arc_length_parametric (x y : ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ :=
  ∫ t in t₁ .. t₂, sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_of_curve :
  let x := λ t : ℝ, 2 * (cos t + t * sin t)
  let y := λ t : ℝ, 2 * (sin t - t * cos t) in
  arc_length_parametric x y 0 (π / 2) = π^2 / 4 :=
by
  sorry

end arc_length_of_curve_l272_272014


namespace sum_of_digits_square_1111111_l272_272831

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_square_1111111 :
  sum_of_digits (1111111 * 1111111) = 49 :=
sorry

end sum_of_digits_square_1111111_l272_272831


namespace smallest_solution_exists_l272_272148

noncomputable def equation := 
  λ x : ℝ, 1/((x - 3)^2) + 1/((x - 5)^2) = 4/((x - 4)^2)

theorem smallest_solution_exists : 
  ∃ x : ℝ, equation x ∧ ∀ y : ℝ, equation y → x <= y :=
sorry

end smallest_solution_exists_l272_272148


namespace new_weight_l272_272769

-- Conditions
def avg_weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase
def weight_replacement (initial_weight : ℝ) (total_increase : ℝ) : ℝ := initial_weight + total_increase

-- Problem Statement: Proving the weight of the new person
theorem new_weight {n : ℕ} {avg_increase initial_weight W : ℝ} 
  (h_n : n = 8) (h_avg_increase : avg_increase = 2.5) (h_initial_weight : initial_weight = 65) (h_W : W = 85) :
  weight_replacement initial_weight (avg_weight_increase n avg_increase) = W :=
by 
  rw [h_n, h_avg_increase, h_initial_weight, h_W]
  sorry

end new_weight_l272_272769


namespace prob_representative_error_lt_3mm_l272_272391

/--
Given:
1. The root mean square deviation (RMSD) of 10 measurements is 10 mm.
2. The measurements are samples from a normal distribution.
3. Use the $t$-distribution with sample size n = 10.

Prove that the probability that the representativeness error in absolute value is less than 3 mm is approximately 0.608.
-/
theorem prob_representative_error_lt_3mm
  (RMSD : ℝ)
  (n : ℕ)
  (h_RMSD : RMSD = 10)
  (h_n : n = 10) :
  let t_dist := t_distribution (n - 1)
  in probability (|representative_error| < 3) t_dist = 0.608 := by
  sorry

end prob_representative_error_lt_3mm_l272_272391


namespace range_of_m_l272_272621

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x 
else if 1 < x ∧ x ≤ 2 then Real.log (x - 1) 
else 0 -- function is not defined outside the given range

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 
  (x ≤ 1 → 2^x ≤ 4 - m * x) ∧ 
  (1 < x ∧ x ≤ 2 → Real.log (x - 1) ≤ 4 - m * x)) → 
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_of_m_l272_272621


namespace speed_in_still_water_is_50_l272_272008

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 80
def speed_in_still_water : ℝ := (upstream_speed + downstream_speed) / 2

theorem speed_in_still_water_is_50 : speed_in_still_water = 50 := by
  sorry

end speed_in_still_water_is_50_l272_272008


namespace domain_f_even_f_increasing_f_l272_272647

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp (2*x) + 1) - x

theorem domain_f : ∀ x : ℝ, True := 
by -- The domain is all real numbers
  trivial

theorem even_f (x : ℝ) : f (-x) = f (x) := 
by 
  sorry -- proof to be filled in

theorem increasing_f (x y : ℝ) (hx : 0 ≤ x) (hy : x ≤ y) : f x ≤ f y :=
by 
  sorry -- proof to be filled in

end domain_f_even_f_increasing_f_l272_272647


namespace only_one_is_true_l272_272929

theorem only_one_is_true (b x y : ℝ) (log_b : ℝ → ℝ) :
  (∀ b x y, b * (x + y) = b * x + b * y) ∧ 
  ¬ (∀ b x y, b^(x + y) = b * x + b * y) ∧
  ¬ (∀ b x y, log_b (x + y) = log_b x + log_b y) ∧
  ¬ (∀ b x y, log_b x * log_b y = log_b (x * y)) ∧
  ¬ (∀ b x y, b * (x / y) = b * x / b * y) :=
by {
  sorry
}

end only_one_is_true_l272_272929


namespace find_time_ball_hits_ground_l272_272775

theorem find_time_ball_hits_ground :
  ∃ t : ℝ, (-16 * t^2 + 40 * t + 30 = 0) ∧ (t = (5 + 5 * Real.sqrt 22) / 4) := 
by
  sorry

end find_time_ball_hits_ground_l272_272775


namespace quadratic_geometric_sequence_root_l272_272390

theorem quadratic_geometric_sequence_root {a b c : ℝ} (r : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b = a * r) 
  (h3 : c = a * r^2)
  (h4 : a ≥ b) 
  (h5 : b ≥ c) 
  (h6 : c ≥ 0) 
  (h7 : (a * r)^2 - 4 * a * (a * r^2) = 0) : 
  -b / (2 * a) = -1 / 8 := 
sorry

end quadratic_geometric_sequence_root_l272_272390


namespace trapezoid_AD_value_l272_272689

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272689


namespace ellipse_equation_exists_l272_272951

theorem ellipse_equation_exists :
  ∃ m n : ℝ, ∃ a^2 : ℝ, m > 0 ∧ n > 0 ∧
  4 * m + 2 * n = 1 ∧ 6 * m + n = 1 ∧
  3 ≤ a^2 ∧ a^2 ≤ 15 ∧
  (9 / a^2) + (4 / (a^2 - 5)) = 1 ∧
  (m = 1/8 ∧ n = 1/4 ∧ a^2 = 15) :=
by
  sorry

end ellipse_equation_exists_l272_272951


namespace Sara_team_wins_l272_272337

theorem Sara_team_wins (total_games losses wins : ℕ) (h1 : total_games = 12) (h2 : losses = 4) (h3 : wins = total_games - losses) :
  wins = 8 :=
by
  sorry

end Sara_team_wins_l272_272337


namespace cistern_leak_time_l272_272845

theorem cistern_leak_time (R : ℝ) (L : ℝ) (eff_R : ℝ) : 
  (R = 1/5) → 
  (eff_R = 1/6) → 
  (eff_R = R - L) → 
  (1 / L = 30) :=
by
  intros hR heffR heffRate
  sorry

end cistern_leak_time_l272_272845


namespace gcd_40_120_80_l272_272426

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end gcd_40_120_80_l272_272426


namespace triangle_J_divides_BC_l272_272657

theorem triangle_J_divides_BC (A B C F H J : Point) (r1 : F ∈ LineSegment A C ∧ dist A F / dist F C = 1/3) (r2: H ∈ LineSegment A B ∧ dist A H = dist H B) (intersect : J ∈ LineSegment B C ∧ J ∈ Line HF H F) : 
  ∃ k : ℝ, k = 3 ∧ dist B J / dist J C = 1 / k := 
sorry

end triangle_J_divides_BC_l272_272657


namespace other_diagonal_of_rhombus_l272_272808

theorem other_diagonal_of_rhombus 
  (area : ℚ) (d1 : ℚ) (h1 : area = 64 / 5) (h2 : d1 = 64 / 9) : 
  ∃ d2 : ℚ, d2 = 18 / 5 ∧ (area = (d1 * d2) / 2) :=
begin
  sorry
end

end other_diagonal_of_rhombus_l272_272808


namespace simplify_expression_l272_272910

variable (x y : ℝ)

theorem simplify_expression : (-(3 * x * y - 2 * x ^ 2) - 2 * (3 * x ^ 2 - x * y)) = (-4 * x ^ 2 - x * y) :=
by
  sorry

end simplify_expression_l272_272910


namespace smallest_prime_factor_of_62_or_64_is_2_l272_272760

def is_prime (n : ℕ) : Prop := nat.prime n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if n = 1 then 1
  else ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

def set_C := {62, 64, 65, 69, 71}

theorem smallest_prime_factor_of_62_or_64_is_2 :
  ∀ x ∈ set_C, (x = 62 ∨ x = 64) → smallest_prime_factor x = 2 :=
by
  intros x hx h
  cases h
  . rw [h]; sorry  -- proving the statement for x = 62
  . rw [h]; sorry  -- proving the statement for x = 64

end smallest_prime_factor_of_62_or_64_is_2_l272_272760


namespace polynomial_zero_candidates_l272_272884

def correct_zeros (z : ℂ) : Prop :=
  ∃ (a b p : ℤ), (∀ x : ℝ, (x - p) * (x^2 + a * x + b) = 0 → x = p) ∧
    a^2 - 4 * b < 0 ∧ ((x^2 + a * x + b).roots).count z = 1

theorem polynomial_zero_candidates :
  correct_zeros (-1 + complex.I * sqrt(3)) ∨
  correct_zeros (-1 + 2 * complex.I) ∨
  correct_zeros (-1 + complex.I * sqrt(5)) :=
sorry

end polynomial_zero_candidates_l272_272884


namespace sin_675_eq_neg_sqrt2_div_2_l272_272923

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l272_272923


namespace concert_with_highest_attendance_l272_272302

namespace ConcertAttendance

theorem concert_with_highest_attendance
  (attend_first : ℕ := 65899) 
  (left_first : ℕ := 375) 
  (initial_diff_second : ℕ := 119)
  (left_second : ℕ := 498)
  (attend_third : ℕ := 80453)
  (left_third : ℕ := 612):
  let remain_first := attend_first - left_first in
  let attend_second := attend_first + initial_diff_second in
  let remain_second := attend_second - left_second in
  let remain_third := attend_third - left_third in
  remain_first = 65524 ∧ 
  remain_second = 65520 ∧ 
  remain_third = 79841 ∧ 
  remain_third = max remain_first (max remain_second remain_third) := by
  sorry

end ConcertAttendance

end concert_with_highest_attendance_l272_272302


namespace evaluate_expression_at_neg_two_l272_272339

noncomputable def complex_expression (a : ℝ) : ℝ :=
  (1 - (a / (a + 1))) / (1 / (1 - a^2))

theorem evaluate_expression_at_neg_two :
  complex_expression (-2) = sorry :=
sorry

end evaluate_expression_at_neg_two_l272_272339


namespace calculate_cost_price_l272_272891

noncomputable def cost_price (SP : ℝ) (profit_percentage : ℝ) : ℝ :=
  SP / (1 + profit_percentage)

theorem calculate_cost_price :
  cost_price 100 0.45 ≈ 68.97 :=
by
  -- Here will reside the proof regarding the approximate nature of this calculation
  sorry

end calculate_cost_price_l272_272891


namespace a_1000_is_2666_l272_272230

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2000 ∧
  a 2 = 2008 ∧
  (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem a_1000_is_2666 (a : ℕ → ℤ) (h : sequence a) : a 1000 = 2666 :=
sorry

end a_1000_is_2666_l272_272230


namespace greta_is_oldest_l272_272053

-- Define the people
variables (A D M G J : ℕ)

-- Conditions given:
def cond1 : Prop := A < D
def cond2 : Prop := M < G
def cond3 : Prop := J > D
def cond4 : Prop := M = J

-- Prove that Greta is the oldest
theorem greta_is_oldest (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
  ∀ x, x ∈ {A, D, M, J} → x < G :=
by {
  sorry
}

end greta_is_oldest_l272_272053


namespace range_of_f_l272_272618

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem range_of_f :
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f x ∧ f x ≤ 3) := sorry

end range_of_f_l272_272618


namespace range_of_a_l272_272987

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, real.exp(2 * x) + real.exp(x) - a = 0) : 0 < a :=
sorry

end range_of_a_l272_272987


namespace trapezoid_AD_CF_BC_CE_l272_272698

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272698


namespace James_final_assets_correct_l272_272721

/-- Given the following initial conditions:
- James starts with 60 gold bars.
- He pays 10% in tax.
- He loses half of what is left in a divorce.
- He invests 25% of the remaining gold bars in a stock market and earns an additional gold bar.
- On Monday, he exchanges half of his remaining gold bars at a rate of 5 silver bars for 1 gold bar.
- On Tuesday, he exchanges half of his remaining gold bars at a rate of 7 silver bars for 1 gold bar.
- On Wednesday, he exchanges half of his remaining gold bars at a rate of 3 silver bars for 1 gold bar.

We need to determine:
- The number of silver bars James has,
- The number of remaining gold bars James has, and
- The number of gold bars worth from the stock investment James has after these transactions.
-/
noncomputable def James_final_assets (init_gold : ℕ) : ℕ × ℕ × ℕ :=
  let tax := init_gold / 10
  let gold_after_tax := init_gold - tax
  let gold_after_divorce := gold_after_tax / 2
  let invest_gold := gold_after_divorce * 25 / 100
  let remaining_gold_after_invest := gold_after_divorce - invest_gold
  let gold_after_stock := remaining_gold_after_invest + 1
  let monday_gold_exchanged := gold_after_stock / 2
  let monday_silver := monday_gold_exchanged * 5
  let remaining_gold_after_monday := gold_after_stock - monday_gold_exchanged
  let tuesday_gold_exchanged := remaining_gold_after_monday / 2
  let tuesday_silver := tuesday_gold_exchanged * 7
  let remaining_gold_after_tuesday := remaining_gold_after_monday - tuesday_gold_exchanged
  let wednesday_gold_exchanged := remaining_gold_after_tuesday / 2
  let wednesday_silver := wednesday_gold_exchanged * 3
  let remaining_gold_after_wednesday := remaining_gold_after_tuesday - wednesday_gold_exchanged
  let total_silver := monday_silver + tuesday_silver + wednesday_silver
  (total_silver, remaining_gold_after_wednesday, invest_gold)

theorem James_final_assets_correct : James_final_assets 60 = (99, 3, 6) := 
sorry

end James_final_assets_correct_l272_272721


namespace natasha_mosaic_l272_272305

theorem natasha_mosaic (tiles_1x1 tiles_1x2 : ℕ) (cells_2 cells_0 cells_1 : ℕ) :
  tiles_1x1 = 4 → tiles_1x2 = 24 → 
  cells_2 = 13 → cells_0 = 18 → cells_1 = 8 →
  (∃ n, n = 6517) :=
by
  intros h1 h2 h3 h4 h5
  use 6517
  sorry

end natasha_mosaic_l272_272305


namespace sally_earnings_last_month_l272_272759

theorem sally_earnings_last_month (x : ℝ) (hx : x + 1.10 * x = 2100) : x = 1000 :=
by
  rw [← add_mul, ← one_mul x] at hx
  have : (1 + 1.10) * x = 2100 := hx
  rw [mul_comm, ← div_eq_iff (1 + 1.10 != 0 : by norm_num)] at this
  have : x = 2100 / 2.10 := this
  norm_num at this
  exact this

end sally_earnings_last_month_l272_272759


namespace bulb_pos_97_to_100_l272_272806

/-- Define the sequence of bulbs meeting the given conditions --/
def bulb (n : Nat) : Prop :=
  if n % 5 == 1 ∨ n % 5 == 0 then "Y" else "B"

/-- The problem to prove the color and order of bulbs at positions 97, 98, 99, and 100 --/
theorem bulb_pos_97_to_100 :
  bulb 97 = "Y" ∧
  bulb 98 = "B" ∧
  bulb 99 = "B" ∧
  bulb 100 = "Y" := by
  sorry

end bulb_pos_97_to_100_l272_272806


namespace g_g_g_of_3_eq_neg_6561_l272_272736

def g (x : ℤ) : ℤ := -x^2

theorem g_g_g_of_3_eq_neg_6561 : g (g (g 3)) = -6561 := by
  sorry

end g_g_g_of_3_eq_neg_6561_l272_272736


namespace smallest_number_has_2020_divisors_l272_272133

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272133


namespace find_AD_l272_272708

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272708


namespace six_coins_heads_or_tails_probability_l272_272356

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l272_272356


namespace find_negative_integer_l272_272559

theorem find_negative_integer (N : ℤ) (h : N^2 + N = -12) : N = -4 := 
by sorry

end find_negative_integer_l272_272559


namespace original_number_correct_l272_272833

-- Definitions for the problem conditions
/-
Let N be the original number.
X is the number to be subtracted.
We are given that X = 8.
We need to show that (N - 8) mod 5 = 4, (N - 8) mod 7 = 4, and (N - 8) mod 9 = 4.
-/

-- Declaration of variables
variable (N : ℕ) (X : ℕ)

-- Given conditions
def conditions := (N - X) % 5 = 4 ∧ (N - X) % 7 = 4 ∧ (N - X) % 9 = 4

-- Given the subtracted number X is 8.
def X_val : ℕ := 8

-- Prove that N = 326 meets the conditions
theorem original_number_correct (h : X = X_val) : ∃ N, conditions N X ∧ N = 326 := by
  sorry

end original_number_correct_l272_272833


namespace triangles_with_red_blue_sides_l272_272664

theorem triangles_with_red_blue_sides :
  ∃ (m n : ℕ), 
    m = 204 ∧ 
    n = 240 ∧ 
    ∀ (points : Fin 18 → ℝ × ℝ) 
      (red_blue : (Fin 18 → Fin 18 → Bool)) 
      (A : Fin 18), 
      ¬∃ i j : Fin 18, i ≠ j ∧ points i = points j ∧ points i = points A ∧ red_blue A i ≠ red_blue A j ∧ 
      ∃ k,
        (red_blue k i = tt ∧ red_blue k j = tt ∧ red_blue k k = tt) ∨ 
        (red_blue k i = tt ∧ red_blue k j = tt ∧ red_blue k k = ff) :=
begin
  sorry
end

end triangles_with_red_blue_sides_l272_272664


namespace solution_unique_l272_272553

def satisfies_equation (x y : ℝ) : Prop :=
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1 / 3

theorem solution_unique (x y : ℝ) :
  satisfies_equation x y ↔ x = 7 + 1/3 ∧ y = 8 - 1/3 :=
by {
  sorry
}

end solution_unique_l272_272553


namespace sphere_radius_l272_272185

theorem sphere_radius (A : ℝ) (k1 k2 k3 : ℝ) (h : A = 64 * Real.pi) : ∃ r : ℝ, r = 4 := 
by 
  sorry

end sphere_radius_l272_272185


namespace cards_per_box_l272_272336

-- Define the conditions
def total_cards : ℕ := 75
def cards_not_in_box : ℕ := 5
def boxes_given_away : ℕ := 2
def boxes_left : ℕ := 5

-- Calculating the total number of boxes initially
def initial_boxes : ℕ := boxes_given_away + boxes_left

-- Define the number of cards in each box
def num_cards_per_box (number_of_cards : ℕ) (number_of_boxes : ℕ) : ℕ :=
  (number_of_cards - cards_not_in_box) / number_of_boxes

-- The proof problem statement
theorem cards_per_box :
  num_cards_per_box total_cards initial_boxes = 10 :=
by
  -- Proof is omitted with sorry
  sorry

end cards_per_box_l272_272336


namespace sequence_2013th_term_l272_272018

theorem sequence_2013th_term :
  (∀ n : ℕ, (∃ k : ℕ, n = 2 * k + 1 → sequence_term n = - (1 / n)) ∧ 
    (∃ k : ℕ, n = 2 * k → sequence_term n = (1 / n)))
  → sequence_term 2013 = - (1 / 2013) :=
by
  sorry

end sequence_2013th_term_l272_272018


namespace reflection_of_P_across_l_reflection_of_l_across_A_l272_272996

open Real

-- Definitions for reflections
def reflect_point (P : Point ℝ) (l : Line ℝ) : Point ℝ := sorry
def reflect_line (l : Line ℝ) (A : Point ℝ) : Line ℝ := sorry

-- Coordinates of P and reflected P'
def P : Point ℝ := (-2, -1)
def P' : Point ℝ := (2 / 5, 19 / 5)

-- Line l and its reflection about point A
def l : Line ℝ := (1, 2, -2) -- x + 2y - 2 = 0
def A : Point ℝ := (1, 1)
def l' : Line ℝ := (1, 2, -4) -- x + 2y - 4 = 0

-- Theorem statements
theorem reflection_of_P_across_l : reflect_point P l = P' := 
by sorry

theorem reflection_of_l_across_A : reflect_line l A = l' := 
by sorry

end reflection_of_P_across_l_reflection_of_l_across_A_l272_272996


namespace trapezoid_AD_equal_l272_272691

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272691


namespace weng_earnings_l272_272822

theorem weng_earnings (hourly_rate : ℝ) (minutes_in_hour : ℕ) (minutes_worked : ℕ) (h_rate : hourly_rate = 12) (h_minutes_in_hour : minutes_in_hour = 60) (h_minutes_worked : minutes_worked = 50) : 
  hourly_rate / minutes_in_hour * minutes_worked = 10 :=
by
  subst h_rate
  subst h_minutes_in_hour
  subst h_minutes_worked
  norm_num
  sorry

end weng_earnings_l272_272822


namespace original_speed_of_person_B_l272_272410

-- Let v_A and v_B be the speeds of person A and B respectively
variable (v_A v_B : ℝ)

-- Conditions for problem
axiom initial_ratio : v_A / v_B = (5 / 4 * v_A) / (v_B + 10)

-- The goal: Prove that v_B = 40
theorem original_speed_of_person_B : v_B = 40 := 
  sorry

end original_speed_of_person_B_l272_272410


namespace equation_solution_l272_272394

theorem equation_solution :
  ∃ x : ℝ, (3 * (x + 2) = x * (x + 2)) ↔ (x = -2 ∨ x = 3) :=
by
  sorry

end equation_solution_l272_272394


namespace probability_of_no_intersection_l272_272756

noncomputable def circle_line_no_common_points_probability (m : ℝ) : ℝ :=
if 0 < m ∧ m < real.sqrt 18 then 1 else 0

theorem probability_of_no_intersection :
  (∫ m in 0..8, circle_line_no_common_points_probability m) / 8 = real.sqrt 18 / 8 :=
sorry

end probability_of_no_intersection_l272_272756


namespace sequence_exists_l272_272279

theorem sequence_exists (a b : ℤ) (ha : 2 < a) (hb : 2 < b) :
  ∃ (k : ℕ) (n : Fin k → ℕ), 
    n 0 = a ∧ n (k - 1) = b ∧ ∀ i : Fin (k - 1), (n i) * (n (i + 1)) % ((n i) + (n (i + 1))) = 0 :=
by
  sorry

end sequence_exists_l272_272279


namespace area_of_M_l272_272860

def K : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10}

def int_part (a : ℝ) : ℤ := int.floor a

def M : set (ℝ × ℝ) :=
  {p | p ∈ K ∧ int_part p.1 < int_part p.2}

theorem area_of_M : (measure_theory.measure.scale (2*S).measure M) / 100 = 0.45 :=
sorry

end area_of_M_l272_272860


namespace trapezoid_AD_equal_l272_272695

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272695


namespace testing_methods_576_l272_272582

theorem testing_methods_576 :
  let genuine_items := 6
  let defective_items := 4
  (∀ sequence : List (Fin (genuine_items + defective_items)),
    (∀ k, k ∈ sequence → k.1 < genuine_items + defective_items) →
    (sequence.length ≥ defective_items + 1) →
    (∀ i < defective_items,
      (∀ j, j < i → ((sequence.nth j).getD 0 < genuine_items)) →
      ((sequence.nth i).getD 0 ≥ genuine_items)) →
    (sequence.nth (defective_items)).getD 0 ≥ genuine_items) →
  ((4.choose 1) * (6.choose 1) * (3.choose 3) * (4.factorial) = 576) :=
by
  sorry

end testing_methods_576_l272_272582


namespace discount_difference_l272_272058

def original_amount : ℚ := 20000
def single_discount_rate : ℚ := 0.30
def first_discount_rate : ℚ := 0.25
def second_discount_rate : ℚ := 0.05

theorem discount_difference :
  (original_amount * (1 - single_discount_rate)) - (original_amount * (1 - first_discount_rate) * (1 - second_discount_rate)) = 250 := by
  sorry

end discount_difference_l272_272058


namespace smallest_positive_period_max_value_and_set_of_x_interval_of_monotonic_decreasing_l272_272625

-- Define the function f
def f (x : ℝ) : ℝ := (sin x)^2 + sqrt 3 * sin x * cos x + 1 / 2

-- Prove the smallest positive period of f is π
theorem smallest_positive_period : ∃ (T > 0), T = π ∧ ∀ x, f (x + T) = f x :=
sorry

-- Prove the maximum value of f and the values of x where it occurs
theorem max_value_and_set_of_x : 
  (max_value : ℝ) → (max_value = 2) ∧ (∃ (x : ℝ), f x = max_value ∧ ∀ k : ℤ, x = k * π + π / 3) :=
sorry

-- Prove the interval of monotonic decreasing for f
theorem interval_of_monotonic_decreasing : ∀ k : ℤ, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → 
  ∀ x₁ x₂, k * π + π / 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + 5 * π / 6 → f x₁ > f x₂ :=
sorry

end smallest_positive_period_max_value_and_set_of_x_interval_of_monotonic_decreasing_l272_272625


namespace smallest_number_of_students_l272_272037

noncomputable def smallest_n (n : ℕ) : Prop :=
  n % 3 = 2 ∧ n % 6 = 5 ∧ n % 8 = 7

theorem smallest_number_of_students : ∃ n, smallest_n n ∧ ∀ m, smallest_n m → n ≤ m :=
by
  use 23
  split
  · unfold smallest_n
    exact ⟨by norm_num, by norm_num, by norm_num⟩
  · intro m
    unfold smallest_n
    intro h
    cases h
    sorry

end smallest_number_of_students_l272_272037


namespace sum_of_100th_group_l272_272085

-- Definitions of the sequence and the grouping function
def sequence (n : ℕ) : ℕ := 2 * n + 1

def group_size (n : ℕ) : ℕ :=
  if n % 4 = 0 then 4 else n % 4

def group_start (n : ℕ) : ℕ :=
  let cycle_len := (n - 1) / 4
  (cycle_len * 10) + (1 + (cycle_len * 4) + (n % 4))

def group_elements (n : ℕ) : List ℕ :=
  let size := group_size n
  List.map sequence (List.range' (group_start n) size)

-- Proposition to prove the required sum of the 100th group
theorem sum_of_100th_group : (group_elements 100).sum = 1992 :=
by sorry

end sum_of_100th_group_l272_272085


namespace sum_of_solutions_zero_l272_272150

noncomputable def f (x : ℝ) : ℝ := 2^|x| + 3*|x|

theorem sum_of_solutions_zero :
  ∃ solutions : set ℝ, (∀ x ∈ solutions, f x = 18) ∧
  (solutions = solutions.Image (λ x, -x)) ∧
  (solutions.Sum id = 0) :=
sorry

end sum_of_solutions_zero_l272_272150


namespace no_such_function_l272_272327

theorem no_such_function : ¬ (∃ f : ℤ → Fin 3, ∀ x y : ℤ, (|x - y| = 2 ∨ |x - y| = 3 ∨ |x - y| = 5) → f x ≠ f y) :=
by sorry

end no_such_function_l272_272327


namespace complex_modulus_range_l272_272979

noncomputable def complex_mod_range (z : ℂ) (hz : abs z = 1) : Set ℝ :=
  {y | 0 ≤ y ∧ y ≤ 3 * Real.sqrt 3}

/-- If the modulus of a complex number z is 1, then the range of the modulus of the expression
    |(z-2)(z+1)^2| is [0, 3√3] -/
theorem complex_modulus_range (z : ℂ) (hz : abs z = 1) :
  (abs ((z-2) * (z+1)^2)) ∈ complex_mod_range z hz :=
sorry

end complex_modulus_range_l272_272979


namespace sum_not_prime_l272_272156

theorem sum_not_prime 
  (a b c x y z : ℕ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : a * x * y = b * y * z ∧ b * y * z = c * z * x) :
  ¬ prime (a + b + c + x + y + z) :=
sorry

end sum_not_prime_l272_272156


namespace length_of_crease_eq_area_of_triangle_eq_l272_272879

noncomputable def triangle5_12_13 (A B C : Type) (AC BC AB AC BC AB : ℝ) : Prop :=
  AC = 5 ∧ BC = 12 ∧ AB = 13 

theorem length_of_crease_eq :=
  ∀ (A B C : Type) (AC BC AB : ℝ),
    triangle5_12_13 A B C AC BC AB ->
    let D := mid_point A C in
    let DC := 5 / 2 in
    let DB_squared := DC^2 + BC^2 in
    DB_squared = 601 / 4 →
    let DB := sqrt (601 / 4) in
    DB = sqrt 601 / 2 :=
sorry

theorem area_of_triangle_eq :=
  ∀ (A B C : Type) (AC BC AB : ℝ),
    triangle5_12_13 A B C AC BC AB ->
    let area := (1 / 2) * AC *BC in
    area = 30 :=
sorry

end length_of_crease_eq_area_of_triangle_eq_l272_272879


namespace smallest_in_decomposition_of_eight_cube_nth_cube_decomposition_l272_272155

theorem smallest_in_decomposition_of_eight_cube :
  let n := 8 in (n^3 = (2 * 28 + 1) + (2 * 28 + 3) + (2 * 28 + 5) + (2 * 28 + 7) +
  (2 * 28 + 9) + (2 * 28 + 11) + (2 * 28 + 13) + (2 * 28 + 15)) :=
by
  have n_eq : 8^3 = 512 := rfl
  have decomp : 512 = 57 + 59 + 61 + 63 + 65 + 67 + 69 + 71 := by 
    calc
      512 = 8 * 64 : by norm_num
      ... = 57 + 59 + 61 + 63 + 65 + 67 + 69 + 71 : sorry
  exact decomp

theorem nth_cube_decomposition (n : ℕ) :
  (n + 1) ≥ 2 →
  (n + 1)^3 = (n^2 + n + 1) + (n^2 + n + 3) + ... + (n^2 + 3*n + 1) :=
by
  intros h
  sorry

end smallest_in_decomposition_of_eight_cube_nth_cube_decomposition_l272_272155


namespace triangle_angle_B_triangle_sides_a_c_l272_272658

theorem triangle_angle_B {A B C a b c : ℝ}
  (h₁ : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

theorem triangle_sides_a_c {A B C a b c : ℝ}
  (h₁ : b = 3)
  (h₂ : sin C = 2 * sin A)
  (h₃ : b * sin A = sqrt 3 * a * cos B) : 
  a = sqrt 3 ∧ c = 2 * sqrt 3 :=
sorry

end triangle_angle_B_triangle_sides_a_c_l272_272658


namespace find_AD_l272_272707

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272707


namespace find_ellipse_equation_l272_272184

-- Definitions based on the given conditions
def parabola_focus (x y : ℝ) : Prop := (y^2 = 4 * x) ∧ (x = 1) ∧ (y = 0)
def ellipse_eccentricity (c a : ℝ) : Prop := (c = 1) ∧ (a = sqrt 2) ∧ (c / a = sqrt 2 / 2)

-- The goal is to find the equation of the ellipse
theorem find_ellipse_equation (x y : ℝ) (c a b : ℝ) 
  (h_parabola_focus : parabola_focus 1 0) 
  (h_eccentricity : ellipse_eccentricity c a) 
  (h_b : b^2 = a^2 - c^2)
  : (x^2 / 2 + y^2 = 1) :=
sorry

end find_ellipse_equation_l272_272184


namespace probability_independent_conditional_eq_conditional_prob_eq_iff_independent_total_probability_l272_272635

theorem probability_independent_conditional_eq {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  (Probability.Independent P A B → P(B | A) = P(B)) :=
by sorry

theorem conditional_prob_eq_iff_independent {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  (P(B | A) = P(B) → P(A | B) = P(A)) :=
by sorry

theorem total_probability {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  P(A ∩ B) + P(Aᶜ ∩ B) = P(B) :=
by sorry

end probability_independent_conditional_eq_conditional_prob_eq_iff_independent_total_probability_l272_272635


namespace vector_computation_l272_272737

def p : ℝ³ := ⟨4, -2, 5⟩
def q : ℝ³ := ⟨1, 2, -3⟩
def r : ℝ³ := ⟨-1, 6, 2⟩

theorem vector_computation :
  let pq := p - q
  let qr := q - r
  let rp := r - p
  pq ⋅ (qr × rp) = 0 :=
by
  sorry

end vector_computation_l272_272737


namespace uncle_bradley_bills_l272_272813

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end uncle_bradley_bills_l272_272813


namespace find_length_AB_l272_272676

-- Definitions for the given conditions
def right_angled_at (A B C : Type) (C : B) := ∃r, r ∈ ℝ ∧ C = 90
def length_BD (x : ℝ) := 2 * x
def length_DC (x : ℝ) := x
def angle_ADC_equivalent (angle_ABC : ℝ) := 2 * angle_ABC

-- Statement for the problem
theorem find_length_AB (A B C D : Type) (x : ℝ) (hABC : right_angled_at A B C C)
  (hBD : length_BD x = 2 * x) (hDC : length_DC x = x) (hAngleADC : angle_ADC_equivalent θ = 2 * θ) : 
  length_AB 2 (sqrt 3 * x) := sorry

end find_length_AB_l272_272676


namespace starting_number_for_product_l272_272573

theorem starting_number_for_product (n k : ℕ) (h7 : k = 7) (h_mult : ∀ m, m ≥ n ∧ m ≤ k → (n * ∏ r in range (k - n + 1), (n + r) = 315)) : n = 3 :=
sorry

end starting_number_for_product_l272_272573


namespace max_minus_min_on_interval_l272_272217

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_on_interval (a : ℝ) :
  let M := max (f 0 a) (f 3 a)
  let N := f 1 a
  M - N = 20 :=
by
  sorry

end max_minus_min_on_interval_l272_272217


namespace second_layer_tiling_l272_272475

theorem second_layer_tiling (n m : ℕ) (first_layer : fin (2 * n * 2 * m) → fin 2) :
  ∃ second_layer : fin (2 * n * 2 * m) → fin 2,
    (∀ i, second_layer i ≠ first_layer i) := 
sorry

end second_layer_tiling_l272_272475


namespace side_length_of_square_l272_272498

theorem side_length_of_square (total_length : ℝ) (sides : ℝ) (side_length : ℝ) : 
  total_length = 34.8 → sides = 4 → side_length = total_length / sides → side_length = 8.7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end side_length_of_square_l272_272498


namespace candies_josh_wants_to_eat_l272_272267

/-
Define the known quantities from the problem conditions.
-/
def initial_candies : ℕ := 100
def siblings : ℕ := 3
def candies_per_sibling : ℕ := 10
def candies_shared_with_others : ℕ := 19

/-
State the goal: Prove that the number of candies Josh wants to eat is 16.
-/
theorem candies_josh_wants_to_eat :
  let total_given_to_siblings := siblings * candies_per_sibling in
  let remaining_after_siblings := initial_candies - total_given_to_siblings in
  let given_to_best_friend := remaining_after_siblings / 2 in
  let remaining_after_best_friend := remaining_after_siblings - given_to_best_friend in
  remaining_after_best_friend - candies_shared_with_others = 16 := by
  sorry

end candies_josh_wants_to_eat_l272_272267


namespace evaluate_rr2_l272_272077

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := (x - 3) * (x - 2)

theorem evaluate_rr2 : r (r 2) = 6 :=
by
  -- proof goes here
  sorry

end evaluate_rr2_l272_272077


namespace smallest_n_mult_y_perfect_cube_l272_272039

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_n_mult_y_perfect_cube : ∃ n : ℕ, (∀ m : ℕ, y * n = m^3 → n = 1500) :=
sorry

end smallest_n_mult_y_perfect_cube_l272_272039


namespace shift_sine_graph_l272_272406

theorem shift_sine_graph :
  ∀ (x : ℝ), (∀ (x : ℝ), sin (2 * (x + π / 6)) = sin (2 * x + π / 3)) := 
  by
    sorry

end shift_sine_graph_l272_272406


namespace quadratic_roots_m_value_l272_272612

noncomputable def quadratic_roots_condition (m : ℝ) (x1 x2 : ℝ) : Prop :=
  (∀ a b c : ℝ, a = 1 ∧ b = 2 * (m + 1) ∧ c = m^2 - 1 → x1^2 + b * x1 + c = 0 ∧ x2^2 + b * x2 + c = 0) ∧ 
  (x1 - x2)^2 = 16 - x1 * x2

theorem quadratic_roots_m_value (m : ℝ) (x1 x2 : ℝ) (h : quadratic_roots_condition m x1 x2) : m = 1 :=
sorry

end quadratic_roots_m_value_l272_272612


namespace raised_bed_section_area_l272_272067

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l272_272067


namespace car_can_drive_on_slope_l272_272045

noncomputable def tan : ℝ → ℝ := sorry -- Placeholder for the tangent function

theorem car_can_drive_on_slope (slope_gradient : ℝ) (max_climbing_angle_deg : ℝ) :
  slope_gradient = 1.5 → max_climbing_angle_deg = 60 →
  (slope_gradient <= tan (max_climbing_angle_deg * (Real.pi / 180))) :=
begin
  intros h1 h2,
  rw [h1, h2],
  sorry,
end

end car_can_drive_on_slope_l272_272045


namespace smallest_number_with_2020_divisors_l272_272145

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272145


namespace find_m_l272_272732

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m + 1) * x + m = 0}

theorem find_m (m : ℝ) : B m ⊆ A → (m = 1 ∨ m = 2) :=
sorry

end find_m_l272_272732


namespace vector_inequality_condition_l272_272326

variable {ℝ : Type} [LinearOrderedField ℝ]

theorem vector_inequality_condition (a b c : ℝ) (f g : ℝ) :
  (∀ (f g : ℝ), a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end vector_inequality_condition_l272_272326


namespace Taehyung_age_l272_272367

variable (T U : Nat)

-- Condition 1: Taehyung is 17 years younger than his uncle
def condition1 : Prop := U = T + 17

-- Condition 2: Four years later, the sum of their ages is 43
def condition2 : Prop := (T + 4) + (U + 4) = 43

-- The goal is to prove that Taehyung's current age is 9, given the conditions above
theorem Taehyung_age : condition1 T U ∧ condition2 T U → T = 9 := by
  sorry

end Taehyung_age_l272_272367


namespace fractional_expression_value_l272_272988

theorem fractional_expression_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 2 * x - 3 * y - z = 0)
  (h2 : x + 3 * y - 14 * z = 0) :
  (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by sorry

end fractional_expression_value_l272_272988


namespace sqrt2_mul_sqrt8_add_sqrt10_between_8_and_9_l272_272550

theorem sqrt2_mul_sqrt8_add_sqrt10_between_8_and_9 : 
  8 < Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) ∧ 
  Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) < 9 := 
by
  sorry

end sqrt2_mul_sqrt8_add_sqrt10_between_8_and_9_l272_272550


namespace purple_gumdrops_after_replacement_l272_272471

def total_gumdrops : Nat := 200
def orange_percentage : Nat := 40
def purple_percentage : Nat := 10
def yellow_percentage : Nat := 25
def white_percentage : Nat := 15
def black_percentage : Nat := 10

def initial_orange_gumdrops := (orange_percentage * total_gumdrops) / 100
def initial_purple_gumdrops := (purple_percentage * total_gumdrops) / 100
def orange_to_purple := initial_orange_gumdrops / 3
def final_purple_gumdrops := initial_purple_gumdrops + orange_to_purple

theorem purple_gumdrops_after_replacement : final_purple_gumdrops = 47 := by
  sorry

end purple_gumdrops_after_replacement_l272_272471


namespace maximum_value_of_f_l272_272788

def f (x : Float) : Float := Float.sin (x + 10 * Float.pi / 180) + Float.cos (x - 20 * Float.pi / 180)

theorem maximum_value_of_f : ∃ (x : Float), f(x) = Float.sqrt 3 := by
  sorry

end maximum_value_of_f_l272_272788


namespace area_percentage_decrease_42_l272_272215

def radius_decrease_factor : ℝ := 0.7615773105863908

noncomputable def area_percentage_decrease : ℝ :=
  let k := radius_decrease_factor
  100 * (1 - k^2)

theorem area_percentage_decrease_42 :
  area_percentage_decrease = 42 := by
  sorry

end area_percentage_decrease_42_l272_272215


namespace pulley_distance_l272_272902

def r1 : ℝ := 10
def r2 : ℝ := 6
def d : ℝ := 20

theorem pulley_distance :
  let BE := r1 - r2 in
  let CD := d in
  let AB := Real.sqrt (BE^2 + CD^2) in
  AB = 2 * Real.sqrt 104 := 
by
  sorry

end pulley_distance_l272_272902


namespace extreme_value_a_eq_2_monotonically_decreasing_g_implies_range_of_a_number_of_zeros_of_f_when_a_gt_0_l272_272989

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x + 1 / x

-- (1) When a=2, find the extreme values of the function y=f(x)
theorem extreme_value_a_eq_2 :
  ∀ x > 0, ∃ (x : ℝ), x = 1 / 2 ∧ f x 2 = 2 - 2 * Real.log 2 :=
sorry

-- (2) If g(x) = f(x) - 2x is monotonically decreasing on (0, +∞), find the range of a
def g (x : ℝ) (a : ℝ) : ℝ := f x a - 2 * x

theorem monotonically_decreasing_g_implies_range_of_a :
  ∀ x > 0, ∀ a : ℝ, (∀ x : ℝ, x > 0 → Deriv.deriv (g x a) < 0) → a < 2 * Real.sqrt 2 :=
sorry

-- (3) When a > 0, discuss the number of zeros of the function y=f(x)
theorem number_of_zeros_of_f_when_a_gt_0 :
  ∀ a > 0, 
    (a = Real.exp 1 → ∃! x : ℝ, x > 0 ∧ f x a = 0) ∧
    (a > Real.exp 1 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ∧
    (0 < a ∧ a < Real.exp 1 → ∀ x : ℝ, x > 0 → f x a ≠ 0) :=
sorry

end extreme_value_a_eq_2_monotonically_decreasing_g_implies_range_of_a_number_of_zeros_of_f_when_a_gt_0_l272_272989


namespace fill_grid_impossible_l272_272521

theorem fill_grid_impossible : 
  ∀ (n m : ℕ) (grid : ℕ → ℕ → ℕ), 
    n = 1000 → 
    m = 1000 → 
    (∀ i, 1 ≤ grid i / 1000 + 1 ≤ 200000) → 
    (∀ j i, grid (i / 1000 + 1) (i % 1000 + 1) ≤ (i / 1000 + 1) * (i % 1000 + 1)) → 
    (∀ k, (count k (λ i, grid (i / 1000 + 1) (i % 1000 + 1)) = 5)) → 
    false := 
by sorry

end fill_grid_impossible_l272_272521


namespace cup_receives_percentage_l272_272201

theorem cup_receives_percentage (C : ℝ) (hC_pos : 0 < C) :
  let pineapple_juice := 1 / 2 * C,
      orange_juice := 1 / 4 * C,
      total_juice := pineapple_juice + orange_juice,
      juice_per_cup := total_juice / 4
  in (juice_per_cup / C) * 100 = 18.75 := by
  sorry

end cup_receives_percentage_l272_272201


namespace geometric_series_sum_l272_272908

theorem geometric_series_sum :
  let a := -1
  let r := -3
  let n := 8
  let S := (a * (r ^ n - 1)) / (r - 1)
  S = 1640 :=
by 
  sorry 

end geometric_series_sum_l272_272908


namespace smallest_n_with_2020_divisors_l272_272129

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272129


namespace proper_divisors_sum_of_729_l272_272432

theorem proper_divisors_sum_of_729 : (∑ d in {1, 3, 9, 27, 81, 243}, d) = 364 :=
by
  sorry

end proper_divisors_sum_of_729_l272_272432


namespace max_cardinality_A_l272_272170

-- Definitions and conditions from the problem
variable (p : ℕ) [Fact (Nat.Prime p)] (n : ℕ) (hpn : p ≥ n) (hn3 : 3 ≤ n)

-- A is the set of sequences of length n from {1, 2, ..., p-1}
def A := {s : Fin n → Fin (p-1) // ∀ (x y : Fin n → Fin (p-1)), x ≠ y → ∃ k l m : Fin n, k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ x k ≠ y k ∧ x l ≠ y l ∧ x m ≠ y m}

-- The theorem statement
theorem max_cardinality_A : Fintype.card A = (p-1)^(n-2) :=
sorry

end max_cardinality_A_l272_272170


namespace intersection_of_sets_l272_272454

theorem intersection_of_sets :
  let M := ({-1, 0, 1} : Set ℤ),
      N := ({0, 1, 2} : Set ℤ) in
  M ∩ N = ({0, 1} : Set ℤ) :=
by
  sorry

end intersection_of_sets_l272_272454


namespace zion_dad_age_difference_in_10_years_l272_272437

/-
Given:
1. Zion's age is 8 years.
2. Zion's dad's age is 3 more than 4 times Zion's age.
Prove:
In 10 years, the difference in age between Zion's dad and Zion will be 27 years.
-/

theorem zion_dad_age_difference_in_10_years :
  let zion_age := 8
  let dad_age := 4 * zion_age + 3
  (dad_age + 10) - (zion_age + 10) = 27 := by
  sorry

end zion_dad_age_difference_in_10_years_l272_272437


namespace triangle_tangent_value_l272_272259

theorem triangle_tangent_value {A B C : Type*} [metric_space M R] [metric_space V W]
  (hypotenuse : R) (side : V) (BC_sqrt : h = real.sqrt (R^2 + V^2)) (AC : V = 4):
  tan B = 4 :=
by
-- Hypotheses
hypothesis h : ∃ angle A B C, angle R
hypothesis angle A : right_angle A
hypothesis metric_space_h [h.metric_space R V]
sorry

end triangle_tangent_value_l272_272259


namespace factory_production_system_l272_272469

theorem factory_production_system (x y : ℕ) (h1 : x + y = 95)
    (h2 : 8*x - 22*y = 0) :
    16*x - 22*y = 0 :=
by
  sorry

end factory_production_system_l272_272469


namespace vinegar_remaining_l272_272316

open Nat

theorem vinegar_remaining (jars cucumbers : ℕ) (vinegar : ℕ) (pickles_per_cucumber pickles_per_jar vinegar_per_jar : ℕ)
  (h_jars : jars = 4) (h_cucumbers : cucumbers = 10) (h_vinegar : vinegar = 100) 
  (h_pickles_per_cucumber : pickles_per_cucumber = 6) (h_pickles_per_jar : pickles_per_jar = 12) 
  (h_vinegar_per_jar : vinegar_per_jar = 10) : 
  vinegar - ((jars * vinegar_per_jar) min (vinegar / vinegar_per_jar) * vinegar_per_jar) = 60 := 
by 
  sorry

end vinegar_remaining_l272_272316


namespace peter_erasers_l272_272315

theorem peter_erasers (initial_erasers : ℕ) (extra_erasers : ℕ) (final_erasers : ℕ)
  (h1 : initial_erasers = 8) (h2 : extra_erasers = 3) : final_erasers = 11 :=
by
  sorry

end peter_erasers_l272_272315


namespace average_and_variance_of_original_data_l272_272044

theorem average_and_variance_of_original_data (μ σ_sq : ℝ)
  (h1 : 2 * μ - 80 = 1.2)
  (h2 : 4 * σ_sq = 4.4) :
  μ = 40.6 ∧ σ_sq = 1.1 :=
by
  sorry

end average_and_variance_of_original_data_l272_272044


namespace find_a4_l272_272196

noncomputable def quadratic_eq (t : ℝ) := t^2 - 36 * t + 288 = 0

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∃ a1 : ℝ, a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 2 = -1
def condition2 (a : ℕ → ℝ) := a 1 - a 3 = -3

theorem find_a4 :
  ∃ (a : ℕ → ℝ) (q : ℝ), quadratic_eq q ∧ geometric_sequence a q ∧ condition1 a ∧ condition2 a ∧ a 4 = -8 :=
by
  sorry

end find_a4_l272_272196


namespace x_sq_between_75_and_85_l272_272207

-- Define the conditions
def satisfies_equation (x : ℝ) : Prop :=
  real.cbrt (x + 9) - real.cbrt (x - 9) = 3

-- Prove the statement
theorem x_sq_between_75_and_85 (x : ℝ) (h : satisfies_equation x) : 75 < x^2 ∧ x^2 < 85 :=
  sorry

end x_sq_between_75_and_85_l272_272207


namespace maximum_cable_connections_l272_272057

theorem maximum_cable_connections 
  (employees : ℕ)
  (brand_A_computers : ℕ)
  (brand_B_computers : ℕ)
  (condition : ∀ b ∈ finset.range brand_B_computers, finset.card (finset.univ.filter (λ a, connection a b)) ≥ 2)
  (connection : ℕ → ℕ → Prop) :
  employees = 40 ∧ brand_A_computers = 25 ∧ brand_B_computers = 15 →
  (∃ (max_connections : ℕ), max_connections = 375) :=
by 
  sorry

end maximum_cable_connections_l272_272057


namespace percentage_markup_l272_272386

theorem percentage_markup 
  (selling_price : ℝ) 
  (cost_price : ℝ) 
  (h1 : selling_price = 8215)
  (h2 : cost_price = 6625)
  : ((selling_price - cost_price) / cost_price) * 100 = 24 := 
  by
    sorry

end percentage_markup_l272_272386


namespace equivalent_statement_l272_272536

-- Definitions based on the conditions
def radius_cylinder_A (r : ℝ) (h : ℝ) : ℝ := r
def height_cylinder_A (h : ℝ) : ℝ := h
def height_cylinder_B (r : ℝ) : ℝ := r
def radius_cylinder_B (h : ℝ) : ℝ := h

-- Volumes of cylinders
def volume_cylinder_A (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h
def volume_cylinder_B (h : ℝ) (r : ℝ) : ℝ := π * h^2 * r

-- Given condition: volume of cylinder A is three times volume of cylinder B
def volume_relation (r : ℝ) (h : ℝ) : Prop := volume_cylinder_A r h = 3 * volume_cylinder_B h r

-- Statement to be proven
theorem equivalent_statement (r h : ℝ) (H : volume_relation r h) : volume_cylinder_A r h = 9 * π * h^3 :=
by
  sorry

end equivalent_statement_l272_272536


namespace perimeter_of_rectangle_l272_272927

-- Define the given conditions and quantities
def large_square_side_length (a : ℝ) : ℝ := 2 * a
def small_square_side_length (b : ℝ) : ℝ := b

-- Define the perimeter calculation for one of the four congruent rectangles
theorem perimeter_of_rectangle (a b : ℝ) :
  let width := b in
  let length := (2 * a) - b in
  (2 * length) + (2 * width) = 4 * a :=
by
  sorry

end perimeter_of_rectangle_l272_272927


namespace candy_mixture_cost_l272_272031

/-- 
A club mixes 15 pounds of candy worth $8.00 per pound with 30 pounds of candy worth $5.00 per pound.
We need to find the cost per pound of the mixture.
-/
theorem candy_mixture_cost :
    (15 * 8 + 30 * 5) / (15 + 30) = 6 := 
by
  sorry

end candy_mixture_cost_l272_272031


namespace final_volume_of_syrup_l272_272515

-- Definitions based on conditions extracted from step a)
def quarts_to_cups (q : ℚ) : ℚ := q * 4
def reduce_volume (v : ℚ) : ℚ := v / 12
def add_sugar (v : ℚ) (s : ℚ) : ℚ := v + s

theorem final_volume_of_syrup :
  let initial_volume_in_quarts := 6
  let sugar_added := 1
  let initial_volume_in_cups := quarts_to_cups initial_volume_in_quarts
  let reduced_volume := reduce_volume initial_volume_in_cups
  add_sugar reduced_volume sugar_added = 3 :=
by
  sorry

end final_volume_of_syrup_l272_272515


namespace equation_of_line_l272_272456

noncomputable def vector := (Real × Real)
noncomputable def point := (Real × Real)

def line_equation (x y : Real) : Prop := 
  let v1 : vector := (-1, 2)
  let p : point := (3, -4)
  let lhs := (v1.1 * (x - p.1) + v1.2 * (y - p.2)) = 0
  lhs

theorem equation_of_line (x y : Real) :
  line_equation x y ↔ y = (1/2) * x - (11/2) := 
  sorry

end equation_of_line_l272_272456


namespace problem_statement_l272_272961

variable (x : ℝ)

theorem problem_statement : 
  let S := (x - 2)^4 + 4 * (x - 2)^3 + 6 * (x - 2)^2 + 4 * (x - 2) + 1
  in S = (x - 1)^4 :=
by 
  sorry

end problem_statement_l272_272961


namespace ab_eq_one_l272_272611

theorem ab_eq_one 
  (h : ∀ (x y : ℝ), x^(2 * a) + y^(b - 1) = 5 → 2 * a = 1 ∧ b - 1 = 1) :
  ∃ a b : ℝ, a * b = 1 :=
by
  have ha_1 : 2 * a = 1 := h 0 0 rfl
  have hb_1 : b - 1 = 1 := h 0 0 rfl
  sorry

end ab_eq_one_l272_272611


namespace smallest_number_with_2020_divisors_l272_272118

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272118


namespace isosceles_right_triangle_l272_272674

theorem isosceles_right_triangle {m : ℝ} (h₀ : 0 < m) (h₁ : m < 2) :
  (m = 4/3) ∨ (m = 1) :=
by
  -- Define the coordinates of vertices
  let A := (-1 : ℝ, 0 : ℝ)
  let B := (3 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, 2 : ℝ)

  -- Define the points of intersection D and E
  let D := (m/2 - 1, m)
  let E := (-3*m/2 + 3, m)

  -- Set conditions for isosceles right triangle ΔDEP
  -- conditions and calculations go here
  sorry

end isosceles_right_triangle_l272_272674


namespace powers_sum_l272_272907

theorem powers_sum : (-3)^4 + (-3)^2 + (-3)^1 + 3^1 - 3^4 + 3^2 = 18 := 
by
  have h1 : (-3)^4 = 3^4 := by sorry -- because 4 is even
  have h2 : (-3)^2 = 3^2 := by sorry -- because 2 is even
  have h3 : (-3)^1 = -3^1 := by sorry -- because 1 is odd

  calc (-3)^4 + (-3)^2 + (-3)^1 + 3^1 - 3^4 + 3^2
      = 3^4 + 3^2 + (-3) + 3 - 3^4 + 3^2 : by rw [h1, h2, h3]
  ... = 81 + 9 - 3 + 3 - 81 + 9 : by norm_num
  ... = 18 : by norm_num

end powers_sum_l272_272907


namespace largest_m_factorial_l272_272587

theorem largest_m_factorial (m : ℕ) (h : ∃ n : ℕ, m! * 2022! = n!) : m = ↑(2022!) - 1 := by
  sorry

end largest_m_factorial_l272_272587


namespace segments_arrangements_equality_l272_272803

theorem segments_arrangements_equality (n k : ℕ) (hn : n > 0) (hk : 1 ≤ k ∧ k ≤ n) :
  let f := λ m : ℕ, if m % 2 = 0 then 
                      let l := m / 2 in choose (n - 1) (l - 1) * choose (n - 1) (l - 1)
                    else 
                      let l := m / 2 in choose (n - 1) l * choose (n - 1) (l - 1) 
  in f (n + k) = f (n - k + 2) := by
  sorry

end segments_arrangements_equality_l272_272803


namespace anne_speed_ratio_l272_272071

variable (B A A' : ℝ)

theorem anne_speed_ratio (h1 : A = 1 / 12)
                        (h2 : B + A = 1 / 4)
                        (h3 : B + A' = 1 / 3) : 
                        A' / A = 2 := 
by
  -- Proof is omitted
  sorry

end anne_speed_ratio_l272_272071


namespace remainder_division_polynomial_l272_272209

noncomputable theory

def polynomial_division_quotient_remainder (f g : ℚ[X]) : (ℚ[X] × ℚ[X]) := Polynomial.div_mod_by_monic f g

theorem remainder_division_polynomial (x : ℚ) (hx : x = 1 / 3):
  let p1_r := polynomial_division_quotient_remainder (Polynomial.X ^ 9) (Polynomial.C x - Polynomial.X) in
  let s1 := p1_r.snd.eval x in
  let p2_r := polynomial_division_quotient_remainder p1_r.fst (Polynomial.C x - Polynomial.X) in
  let s2 := p2_r.snd.eval x in
  s1 = 1 / 19683 ∧ s2 = 1 / 2 := 
sorry

end remainder_division_polynomial_l272_272209


namespace square_inscribed_in_ellipse_area_l272_272494

theorem square_inscribed_in_ellipse_area :
  (∃ t : ℝ, (t > 0) ∧ (t^2 = 8/3)) →
  ∃ A : ℝ, A = (2 * sqrt (8/3))^2 ∧ A = 32/3 :=
by {
  intro ht,
  cases ht with t ht_props,
  use (2 * sqrt (8 / 3))^2,
  split,
  { 
    -- First part of proof: showing the computed area matches the calculation
    have area_computed : (2 * sqrt (8 / 3))^2 = 4 * (8 / 3),
    { 
      calc (2 * sqrt (8 / 3))^2 = 4 * (sqrt (8 / 3))^2 : by ring
      ... = 4 * (8 / 3) : by rw [Real.sqrt_sq (show 8 / 3 ≥ 0, by norm_num)]
    },
    exact area_computed,
  },
  { 
    -- Second part of proof: showing the area equals 32/3
    have area_value : 4 * (8 / 3) = 32 / 3,
    { 
      calc 4 * (8 / 3) = 32 / 3 : by ring,
    },
    exact area_value,
  }
}

end square_inscribed_in_ellipse_area_l272_272494


namespace vinegar_remaining_l272_272317

open Nat

theorem vinegar_remaining (jars cucumbers : ℕ) (vinegar : ℕ) (pickles_per_cucumber pickles_per_jar vinegar_per_jar : ℕ)
  (h_jars : jars = 4) (h_cucumbers : cucumbers = 10) (h_vinegar : vinegar = 100) 
  (h_pickles_per_cucumber : pickles_per_cucumber = 6) (h_pickles_per_jar : pickles_per_jar = 12) 
  (h_vinegar_per_jar : vinegar_per_jar = 10) : 
  vinegar - ((jars * vinegar_per_jar) min (vinegar / vinegar_per_jar) * vinegar_per_jar) = 60 := 
by 
  sorry

end vinegar_remaining_l272_272317


namespace students_who_failed_correct_l272_272400

noncomputable def num_students := 240
noncomputable def students_with_A := 0.4 * num_students
noncomputable def remaining_students := num_students - students_with_A
noncomputable def students_with_B_or_C := remaining_students / 3
noncomputable def students_with_D := 29 -- We directly use 29 instead of rounding in Lean for simplicity.
noncomputable def students_failed := num_students - (students_with_A + students_with_B_or_C + students_with_D)

theorem students_who_failed_correct :
  students_failed = 67 :=
by
  have hnum_students : num_students = 240 :=
  by rfl
  have hstudents_with_A : students_with_A = 0.4 * 240 :=
  by rfl

  have hremaining_students: remaining_students = 144 :=
  by simp [remaining_students, students_with_A, hnum_students, hstudents_with_A]
  
  have hstudents_with_B_or_C : students_with_B_or_C = 144 / 3 :=
  by simp [students_with_B_or_C, remaining_students, hremaining_students]
  
  have hstudents_with_B_or_C_value : students_with_B_or_C = 48 :=
  by simp [hstudents_with_B_or_C]
  
  have hstudents_with_D_value : students_with_D = 29 :=
  by rfl
  
  have expected_total_used := students_with_A + students_with_B_or_C + students_with_D
    
  have expected_total : expected_total_used = 173 :=
  by simp [students_with_A, hstudents_with_Students_num_students, hstudents_with_A_value, hstudents_with_B_or_C_value, hstudents_with_D_value]

  have hstudents_failed : students_failed = num_students - expected_total_used :=
  by simp [students_failed, hnum_students, expected_total]
  
  have hstudents_failed_value : students_failed = 67 :=
  by simp [hstudents_failed]

  leave hstudents_failed_value -- To conclude students_who_failed_correct as mentioned.

  sorry

end students_who_failed_correct_l272_272400


namespace six_coins_heads_or_tails_probability_l272_272357

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l272_272357


namespace triangle_area_l272_272784

def right_triangle_area (hypotenuse leg1 : ℕ) : ℕ :=
  if (hypotenuse ^ 2 - leg1 ^ 2) > 0 then (1 / 2) * leg1 * (hypotenuse ^ 2 - leg1 ^ 2).sqrt else 0

theorem triangle_area (hypotenuse leg1 : ℕ) (h_hypotenuse : hypotenuse = 13) (h_leg1 : leg1 = 5) :
  right_triangle_area hypotenuse leg1 = 30 :=
by
  rw [h_hypotenuse, h_leg1]
  sorry

end triangle_area_l272_272784


namespace trapezoid_AD_CF_BC_CE_l272_272699

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272699


namespace percentage_of_male_students_l272_272662

noncomputable def percentage_male_students := 40

def condition1 (M F : ℝ) : Prop := M + F = 100
def condition2 (M : ℝ) : Prop := 0.60 * M / 100
def condition3 (F : ℝ) : Prop := 0.70 * F / 100
def condition4 (M F : ℝ) : Prop := (0.60 * M + 0.70 * F) / 100 = 0.66

theorem percentage_of_male_students :
  ∃ M F : ℝ, condition1 M F ∧ condition2 M ∧ condition3 F ∧ condition4 M F ∧ M = percentage_male_students :=
sorry

end percentage_of_male_students_l272_272662


namespace DEF_iso_right_l272_272052

variable {A B C D E F : Type*}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable {triangle : A → B → C → Prop}
variable {midpoint : A → B → C}
variable {isosceles_right_triangle : A → B → C → Prop}

-- Conditions:
-- 1. ABC is any triangle
-- 2. D and E are constructed such that ABD and ACE are right-angled isosceles triangles
-- 3. F is the midpoint of BC
axiom ABC_tri : triangle A B C
axiom ABD_iso_right : isosceles_right_triangle A B D
axiom ACE_iso_right : isosceles_right_triangle A C E
axiom F_midpoint : midpoint B C = F

-- Question: Show that DEF is a right-angled isosceles triangle
theorem DEF_iso_right : isosceles_right_triangle D E F :=
  sorry

end DEF_iso_right_l272_272052


namespace percentage_unloaded_at_second_store_l272_272504

theorem percentage_unloaded_at_second_store
  (initial_weight : ℝ)
  (percent_unloaded_first : ℝ)
  (remaining_weight_after_deliveries : ℝ)
  (remaining_weight_after_first : ℝ)
  (weight_unloaded_second : ℝ)
  (percent_unloaded_second : ℝ) :
  initial_weight = 50000 →
  percent_unloaded_first = 0.10 →
  remaining_weight_after_deliveries = 36000 →
  remaining_weight_after_first = initial_weight * (1 - percent_unloaded_first) →
  weight_unloaded_second = remaining_weight_after_first - remaining_weight_after_deliveries →
  percent_unloaded_second = (weight_unloaded_second / remaining_weight_after_first) * 100 →
  percent_unloaded_second = 20 :=
by
  intros _
  sorry

end percentage_unloaded_at_second_store_l272_272504


namespace ticket_price_increase_l272_272549

-- Definitions as per the conditions
def old_price : ℝ := 85
def new_price : ℝ := 102
def percent_increase : ℝ := (new_price - old_price) / old_price * 100

-- Statement to prove the percent increase is 20%
theorem ticket_price_increase : percent_increase = 20 := by
  sorry

end ticket_price_increase_l272_272549


namespace sum_proper_divisors_729_l272_272430

def proper_divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d, d ∣ n ∧ d ≠ n)

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

theorem sum_proper_divisors_729 : sum_proper_divisors 729 = 364 := by
  sorry

end sum_proper_divisors_729_l272_272430


namespace y_intercept_of_line_l272_272101

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l272_272101


namespace purple_tetrahedron_volume_l272_272870

theorem purple_tetrahedron_volume (s : ℝ) (hs : s = 6) : volume_of_tetrahedron (tetrahedron_with_purple_vertices (cube s)) = 72 := 
by
  sorry

end purple_tetrahedron_volume_l272_272870


namespace pearl_grandchildren_count_l272_272310

noncomputable def stockings_cost (price_per_stocking: ℝ) (discount: ℝ) (monogramming_cost: ℝ) : ℝ :=
  (price_per_stocking * (1 - discount / 100)) + monogramming_cost

noncomputable def total_stockings (total_cost: ℝ) (cost_per_stocking: ℝ) : ℕ :=
  (total_cost / cost_per_stocking).to_nat

noncomputable def number_of_grandchildren (total_stockings: ℕ) (children: ℕ) : ℕ :=
  total_stockings - children

theorem pearl_grandchildren_count : 
  let price_per_stocking := 20.00
  let discount := 10.0
  let monogramming_cost := 5.00
  let total_cost := 1035.00
  let children := 4 in
  number_of_grandchildren (total_stockings total_cost (stockings_cost price_per_stocking discount monogramming_cost)) children = 41 :=
  by
    -- The proof would go here
    sorry

end pearl_grandchildren_count_l272_272310


namespace train_length_is_330_meters_l272_272895

noncomputable def train_speed : ℝ := 60 -- in km/hr
noncomputable def man_speed : ℝ := 6    -- in km/hr
noncomputable def time : ℝ := 17.998560115190788  -- in seconds

noncomputable def relative_speed_km_per_hr : ℝ := train_speed + man_speed
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed_m_per_s : ℝ := 
  relative_speed_km_per_hr * conversion_factor

theorem train_length_is_330_meters : 
  (relative_speed_m_per_s * time) = 330 := 
sorry

end train_length_is_330_meters_l272_272895


namespace parallelogram_larger_angle_l272_272225

theorem parallelogram_larger_angle (a b : ℕ) (h₁ : b = a + 50) (h₂ : a = 65) : b = 115 := 
by
  -- Use the conditions h₁ and h₂ to prove the statement.
  sorry

end parallelogram_larger_angle_l272_272225


namespace choose_signs_inequality_l272_272291

theorem choose_signs_inequality (n : ℕ) (h_n : 2 ≤ n) (a : ℕ → ℝ) :
  ∃ ε : ℕ → ℤ, (∀ i, ε i = 1 ∨ ε i = -1) ∧ 
  ((∑ i in Finset.range n, a i) ^ 2 + (∑ i in Finset.range n, ε i * a i) ^ 2 ≤ 
  (n + 1) * ∑ i in Finset.range n, (a i) ^ 2) :=
sorry

end choose_signs_inequality_l272_272291


namespace gcd_40_120_80_l272_272427

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end gcd_40_120_80_l272_272427


namespace triangle_congruence_l272_272261

theorem triangle_congruence (a b c d e f : ℝ) 
  (h1 : a ≠ d) 
  (h2 : b ≠ e) 
  (h3 : c ≠ f) 
  (h4 : a = e) 
  (h5 : b = f) 
  (h6 : c = d) 
  : ∀ (A B C D E F : fin 3 → ℝ) (hABC : Δ A B C) (hDEF : Δ D E F), 
    (A, B, C) ≃ (D, E, F) := 
by 
  sorry

end triangle_congruence_l272_272261


namespace find_sum_of_arithmetic_sequences_l272_272186

variable {a b : ℕ → ℝ}
variable (a1 b1 : ℝ)
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem find_sum_of_arithmetic_sequences
  (ha : is_arithmetic_seq a)
  (hb : is_arithmetic_seq b)
  (h₁ : a 0 = 1)
  (h₂ : b 0 = 7)
  (h₃ : a 1 + b 1 = 12) :
  let S (n : ℕ) := ∑ i in finset.range n, (a i + b i) in
  S 20 = 920 :=
by
  sorry

end find_sum_of_arithmetic_sequences_l272_272186


namespace vinegar_left_l272_272319

-- Define the initial quantities and conversion factors
def initial_jars := 4
def initial_cucumbers := 10
def initial_vinegar := 100
def pickles_per_cucumber := 6
def pickles_per_jar := 12
def vinegar_per_jar := 10

-- The main theorem to prove
theorem vinegar_left (jars : ℕ) (cucumbers : ℕ) (vinegar : ℕ) 
  (pickles_per_cucumber : ℕ) (pickles_per_jar : ℕ) (vinegar_per_jar : ℕ) : 
  (pickles_per_cucumber * cucumbers) ≥ (pickles_per_jar * jars) →
  vinegar - (jars * vinegar_per_jar) = 60 :=
by
  sorry

-- Substitute the given conditions and facts into the theorem
example : vinegar_left initial_jars initial_cucumbers initial_vinegar 
  pickles_per_cucumber pickles_per_jar vinegar_per_jar :=
by
  simp [initial_jars, initial_cucumbers, initial_vinegar, pickles_per_cucumber, pickles_per_jar, vinegar_per_jar]
  let jars := initial_jars
  let cucumbers := initial_cucumbers
  let vinegar := initial_vinegar
  let pickles_per_cucumber := pickles_per_cucumber
  let pickles_per_jar := pickles_per_jar
  let vinegar_per_jar := vinegar_per_jar

  have h1 : pickles_per_cucumber * cucumbers ≥ pickles_per_jar * jars := by
    calc
      60 = pickles_per_cucumber * cucumbers := by norm_num
      48 = pickles_per_jar * jars := by norm_num
      60 ≥ 48 := by norm_num
    
  apply vinegar_left
  assumption

end vinegar_left_l272_272319


namespace monotonicity_f_range_b_l272_272193

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + (a / x)
noncomputable def h (x : ℝ) : ℝ := x + (4 / x) - 8
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := -x - 2 * b

-- Statement to prove the monotonicity of f(x)
theorem monotonicity_f {a : ℝ} (ha : a > 0) :
  ((∀ x : ℝ, 0 < x → x ≤ sqrt a → ∀ x₁ x₂, x₁ < x₂ → f x₁ a > f x₂ a) ∧
   (∀ x : ℝ, x ≥ sqrt a → f x a ≥ f (sqrt a) a)) ∧
  ((∀ x : ℝ, x < 0 → -sqrt a ≤ x → ∀ x₁ x₂, x₁ < x₂ → f x₁ a > f x₂ a) ∧
   (∀ x : ℝ, x ≤ -sqrt a → f x a ≤ f (-sqrt a) a)) :=
sorry

-- Statement to prove the range of b
theorem range_b :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 3 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 3 ∧ g x₂ b = h x₁) → 1/2 ≤ b ∧ b ≤ 1 :=
sorry

end monotonicity_f_range_b_l272_272193


namespace jessica_saves_l272_272722

-- Define the costs based on the conditions given
def basic_cost : ℕ := 15
def movie_cost : ℕ := 12
def sports_cost : ℕ := movie_cost - 3
def bundle_cost : ℕ := 25

-- Define the total cost when the packages are purchased separately
def separate_cost : ℕ := basic_cost + movie_cost + sports_cost

-- Define the savings when opting for the bundle
def savings : ℕ := separate_cost - bundle_cost

-- The theorem that states the savings are 11 dollars
theorem jessica_saves : savings = 11 :=
by
  sorry

end jessica_saves_l272_272722


namespace probability_of_sum_ge_5_l272_272160

def balls : Finset ℕ := {1, 2, 3, 4}

def selected_balls := {pair ∈ balls.powerset.filter (λ s, s.card = 2) | (s : ℕ).sum ≥ 5}

theorem probability_of_sum_ge_5 :
  (selected_balls.card : ℚ) / (balls.powerset.filter (λ s, s.card = 2)).card = 2 / 3 := 
by 
  -- Add proof here 
  sorry

end probability_of_sum_ge_5_l272_272160


namespace raised_bed_area_correct_l272_272062

def garden_length : ℝ := 220
def garden_width : ℝ := 120
def garden_area : ℝ := garden_length * garden_width
def tilled_land_area : ℝ := garden_area / 2
def remaining_area : ℝ := garden_area - tilled_land_area
def trellis_area : ℝ := remaining_area / 3
def raised_bed_area : ℝ := remaining_area - trellis_area

theorem raised_bed_area_correct : raised_bed_area = 8800 := by
  sorry

end raised_bed_area_correct_l272_272062


namespace existence_of_subconvex_polygons_minimum_value_of_m_l272_272159

theorem existence_of_subconvex_polygons (n : ℕ) (h_n_odd : n % 2 = 1) (h_n_ge_5 : n ≥ 5) :
  ∃ m : ℕ, 
    (∀ (S : list (list ℕ)), 
      (S.length = m) ∧ 
      (∀ sub_poly ∈ S, 
        ∀ (vtx1 vtx2 : ℕ), 
          (vtx1 ∈ sub_poly ∧ vtx2 ∈ sub_poly → vtx1 ≠ vtx2) ) ∧
      (∀ edge : ℕ, 
        ∃ sub_poly ∈ S, 
          (∀ vtx1 vtx2, (edge = vtx1 + vtx2) → 
            (vtx1 ∈ sub_poly ∧ vtx2 ∈ sub_poly) ) ) ) :=
sorry

theorem minimum_value_of_m (n : ℕ) (h_n_odd : n % 2 = 1) (h_n_ge_5 : n ≥ 5) :
  ∃ m : ℕ, m = (n - 1) * (n + 1) / 8 :=
sorry

end existence_of_subconvex_polygons_minimum_value_of_m_l272_272159


namespace telomerase_structure_l272_272369

theorem telomerase_structure :
  (∀ (E_coli : Type) (nucleoid_DNA : E_coli → Prop), ¬ nucleoid_DNA (E_coli := E_coli)) →   -- E. coli DNA does not contain telomeres
  (∀ (protein_telo : Type) (is_RNAP : protein_telo → Prop), ¬ is_RNAP (protein_telo := protein_telo)) →   -- Telomerase protein is not RNA polymerase
  (∀ (human_cell : Type) (chromosome_ends : human_cell → Prop), chromosome_ends (human_cell := human_cell)) →   -- Chromosome ends in human cells contain telomere DNA
  (∀ (somatic_cell : Type) (telomere_division : somatic_cell → Prop), ¬ telomere_division (somatic_cell := somatic_cell)) →   -- Telomere DNA shortens with cell division
  True := 
begin
  sorry
end

end telomerase_structure_l272_272369


namespace cos_Y_relatively_prime_l272_272717

variables {X Y Z W : Type} [EuclideanGeometry X] [EuclideanGeometry Y] [EuclideanGeometry Z] [EuclideanGeometry W]

def triangle_XYZ (X Y Z : Type) [EuclideanGeometry X] [EuclideanGeometry Y] [EuclideanGeometry Z] : Prop :=
triangle X Y Z ∧ is_right_angle Z ∧ is_hypotenuse XY ∧ altitude_from Z_meets_XY_in W 

def length_YW (Y W : Type) [HasLength Y W] : ℕ := 17^3

theorem cos_Y_relatively_prime (X Y Z W : Type) [EuclideanGeometry X] [EuclideanGeometry Y] [EuclideanGeometry Z] [EuclideanGeometry W]
  (h_triangle : triangle_XYZ X Y Z)
  (h_length_YW : length_YW Y W)
  (h_cos_Y : ∃ r s : ℕ, r.gcd s = 1 ∧ cos Y = r / s) :
  ∃ r s : ℕ, r.gcd s = 1 ∧ cos Y = r / s ∧ r + s = 162 := 
by
  sorry

end cos_Y_relatively_prime_l272_272717


namespace tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces_l272_272535

/-- A theorem to prove that a regular tetrahedron can be divided into equal polyhedra,
each with six faces. -/
theorem tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces
  (T : Type) [regular_tetrahedron T] :
  ∃ (P : Type), equal_polyhedra_with_six_faces P :=
sorry

end tetrahedron_can_be_divided_into_equal_polyhedra_with_six_faces_l272_272535


namespace average_weight_of_Arun_l272_272444

theorem average_weight_of_Arun :
  ∃ avg_weight : Real,
    (avg_weight = (65 + 68) / 2) ∧
    ∀ w : Real, (65 < w ∧ w < 72) ∧ (60 < w ∧ w < 70) ∧ (w ≤ 68) → avg_weight = 66.5 :=
by
  -- we will fill the details of the proof here
  sorry

end average_weight_of_Arun_l272_272444


namespace least_pos_int_N_l272_272827

theorem least_pos_int_N :
  ∃ N : ℕ, (N > 0) ∧ (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ 
  (∀ m : ℕ, (m > 0) ∧ (m % 4 = 3) ∧ (m % 5 = 4) ∧ (m % 6 = 5) ∧ (m % 7 = 6) → N ≤ m) ∧ N = 419 :=
by
  sorry

end least_pos_int_N_l272_272827


namespace sin_equations_solution_l272_272457

theorem sin_equations_solution {k : ℤ} (hk : k ≤ 1 ∨ k ≥ 5) : 
  (∃ x : ℝ, 2 * x = π * k ∧ x = (π * k) / 2) ∨ x = 7 * π / 4 :=
by
  sorry

end sin_equations_solution_l272_272457


namespace number_of_moles_of_water_formed_l272_272560

def balanced_combustion_equation : Prop :=
  ∀ (CH₄ O₂ CO₂ H₂O : ℕ), (CH₄ + 2 * O₂ = CO₂ + 2 * H₂O)

theorem number_of_moles_of_water_formed
  (CH₄_initial moles_of_CH₄ O₂_initial moles_of_O₂ : ℕ)
  (h_CH₄_initial : CH₄_initial = 3)
  (h_O₂_initial : O₂_initial = 6)
  (h_moles_of_H₂O : moles_of_CH₄ * 2 = 2 * moles_of_H₂O) :
  moles_of_H₂O = 6 :=
by
  sorry

end number_of_moles_of_water_formed_l272_272560


namespace major_axis_equals_10_l272_272946

noncomputable def major_axis_length (x y : ℝ) (φ : ℝ) (h1 : x = 3 * cos φ) (h2 : y = 5 * sin φ) : ℝ :=
2 * 5

theorem major_axis_equals_10 (x y : ℝ) (φ : ℝ) (h1 : x = 3 * cos φ) (h2 : y = 5 * sin φ) : 
  major_axis_length x y φ h1 h2 = 10 :=
by {
  sorry -- The actual proof is omitted
}

end major_axis_equals_10_l272_272946


namespace clock_angle_7_30_l272_272423

theorem clock_angle_7_30 :
  let hour_mark_angle := 30
  let minute_mark_angle := 6
  let hour_hand_angle := 7 * hour_mark_angle + (30 * hour_mark_angle / 60)
  let minute_hand_angle := 30 * minute_mark_angle
  let angle_diff := abs (hour_hand_angle - minute_hand_angle)
  angle_diff = 45 := by
  sorry

end clock_angle_7_30_l272_272423


namespace triangle_area_correct_l272_272409

noncomputable def triangle_area_problem : ℝ := 6.3

theorem triangle_area_correct:
  let line1 := (λ x : ℝ, (3/2) * x - 1/2),
      line2 := (λ x : ℝ, (1/3) * x + 2/3),
      intersect_point := (1, 1 : ℝ),
      line3 := (λ x, 8 - x),
      point_A := (1, 1 : ℝ),
      point_B := (3.4, 4.6: ℝ),
      point_C := (5.5, 2.5: ℝ) in
  point_A = intersect_point ∧
  line1 point_A.1 = point_A.2 ∧
  line2 point_A.1 = point_A.2 ∧
  line1 point_B.1 = point_B.2 ∧
  line3 point_B.1 = point_B.2 ∧
  line2 point_C.1 = point_C.2 ∧
  line3 point_C.1 = point_C.2 ∧
  (1 / 2) * | point_A.1 * (point_B.2 - point_C.2) +
             point_B.1 * (point_C.2 - point_A.2) +
             point_C.1 * (point_A.2 - point_B.2) | =
  triangle_area_problem :=
by sorry

end triangle_area_correct_l272_272409


namespace expression_evaluation_l272_272909

theorem expression_evaluation : 
  54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by 
  sorry

end expression_evaluation_l272_272909


namespace smallest_n_with_2020_divisors_l272_272128

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272128


namespace find_S5_l272_272289

variable (x : ℝ) (m : ℕ)

def Sm (x : ℝ) (m : ℕ) := x^m + x^(-m)

theorem find_S5 (h1 : x + x⁻¹ = 5) : Sm x 5 = 2520 := sorry

end find_S5_l272_272289


namespace compute_div_mul_l272_272527

noncomputable def a : ℚ := 0.24
noncomputable def b : ℚ := 0.006

theorem compute_div_mul : ((a / b) * 2) = 80 := by
  sorry

end compute_div_mul_l272_272527


namespace derivative_at_1_is_neg2_l272_272774

-- Define the function y = (x-2)^2
def my_function (x : ℝ) : ℝ := (x - 2) ^ 2

-- Define its derivative
noncomputable def my_function_deriv (x : ℝ) : ℝ := (derivative my_function x)

-- Theorem stating that the derivative of the function at x = 1 is -2
theorem derivative_at_1_is_neg2 : my_function_deriv 1 = -2 :=
by {
  -- The proof goes here
  sorry
}

end derivative_at_1_is_neg2_l272_272774


namespace determine_n_l272_272540

theorem determine_n (n : ℕ) (h : 3^n = 3^2 * 9^4 * 81^3) : n = 22 := 
by
  sorry

end determine_n_l272_272540


namespace chelsea_initial_sugar_l272_272522

variable (S : Real)

def remaining_sugar (S : Real) : Real := S - S / 8

theorem chelsea_initial_sugar
  (h1 : remaining_sugar S = 21) :
  S = 24 :=
  sorry

end chelsea_initial_sugar_l272_272522


namespace inscribed_square_area_l272_272486

noncomputable def area_of_inscribed_square : ℝ :=
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let t := (sqrt (8 / 3))
  let square_side := 2 * t
  let square_area := (square_side^2)
  square_area

theorem inscribed_square_area :
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let square_area := (2 * sqrt (8 / 3))^2
  square_area = 32 / 3 :=
by
  sorry

end inscribed_square_area_l272_272486


namespace largest_angle_is_90_degrees_l272_272616

noncomputable def is_largest_angle_90_degrees (M F₁ F₂ : Type) [metric_space M] (s₁ s₂ : ℝ) :=
  ∃ (|MF₁| |MF₂| : ℝ) (C : conic_section) (point_on_C : M), 
    (C = ellipse 16 12) ∧ 
    (foci C = (F₁, F₂)) ∧ 
    (point_on_C ∈ C) ∧ 
    (|MF₁| - |MF₂| = 2) ∧ 
    (largest_angle_in_triangle (|MF₁|, |MF₂|, distance F₁ F₂) = 90)

theorem largest_angle_is_90_degrees (M F₁ F₂ : Type) [metric_space M] (s₁ s₂ : ℝ) : 
  is_largest_angle_90_degrees M F₁ F₂ s₁ s₂ :=
sorry

end largest_angle_is_90_degrees_l272_272616


namespace vector_dot_product_example_l272_272607

noncomputable def vector_dot_product (e1 e2 : ℝ) : ℝ :=
  let c := e1 * (-3 * e1)
  let d := (e1 * (2 * e2))
  let e := (e2 * (2 * e2))
  c + d + e

theorem vector_dot_product_example (e1 e2 : ℝ) (unit_vectors : e1^2 = 1 ∧ e2^2 = 1) :
  (e1 - e2) * (e1 - e2) = 1 ∧ (e1 * e2 = 1 / 2) → 
  vector_dot_product e1 e2 = -5 / 2 := by {
  sorry
}

end vector_dot_product_example_l272_272607


namespace smaller_angle_at_7_30_l272_272417

def clock_angle_deg_per_hour : ℝ := 30 

def minute_hand_angle_at_7_30 : ℝ := 180

def hour_hand_angle_at_7_30 : ℝ := 225

theorem smaller_angle_at_7_30 : 
  ∃ angle : ℝ, angle = 45 ∧ 
  (angle = |hour_hand_angle_at_7_30 - minute_hand_angle_at_7_30|) :=
begin
  sorry
end

end smaller_angle_at_7_30_l272_272417


namespace ratio_female_democrats_l272_272802

theorem ratio_female_democrats (total_participants male_participants female_participants total_democrats female_democrats : ℕ)
  (h1 : total_participants = 750)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : total_democrats = total_participants / 3)
  (h4 : female_democrats = 125)
  (h5 : total_democrats = male_participants / 4 + female_democrats) :
  (female_democrats / female_participants : ℝ) = 1 / 2 :=
sorry

end ratio_female_democrats_l272_272802


namespace chef_already_cooked_potatoes_l272_272466

theorem chef_already_cooked_potatoes :
  ∀ (total_potatoes remaining_time cooking_time_per_potato : ℕ),
    total_potatoes = 15 →
    remaining_time = 72 →
    cooking_time_per_potato = 8 →
    total_potatoes - (remaining_time / cooking_time_per_potato) = 6 :=
by
  intros total_potatoes remaining_time cooking_time_per_potato
  intros h_total h_remaining h_cooking
  rw [h_total, h_remaining, h_cooking]
  norm_num
  sorry

end chef_already_cooked_potatoes_l272_272466


namespace range_of_m_l272_272590

theorem range_of_m (m : ℝ) (P : Prop) (Q : Prop) : 
  (P ∨ Q) ∧ ¬(P ∧ Q) →
  (P ↔ (m^2 - 4 > 0)) →
  (Q ↔ (16 * (m - 2)^2 - 16 < 0)) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l272_272590


namespace max_sin_square_sum_l272_272597

theorem max_sin_square_sum (n : ℕ) (h : n > 1) (x : ℕ → ℝ) 
  (non_neg : ∀ i, 0 ≤ x i) (sum_eq_pi : (∑ i in finset.range n, x i) = real.pi) : 
  (∑ i in finset.range n, (real.sin (x i))^2) ≤ if n = 2 then 2 else 9 / 4 := 
by {
  -- Additional hypotheses needed for a complete proof
  sorry
}

end max_sin_square_sum_l272_272597


namespace calories_in_serving_l272_272404

/-- Define the conditions given in the problem. --/
def total_servings_in_block : ℕ := 16
def servings_eaten : ℕ := 5
def remaining_calories : ℕ := 1210

/-- Define the number of remaining servings --/
def remaining_servings : ℕ := total_servings_in_block - servings_eaten

/-- Define the number of calories per serving --/
def calories_per_serving : ℕ := remaining_calories / remaining_servings

/-- The proof statement --/
theorem calories_in_serving (total_servings_in_block = 16)
    (servings_eaten = 5)
    (remaining_calories = 1210)
    (remaining_servings = total_servings_in_block - servings_eaten)
    (calories_per_serving = remaining_calories / remaining_servings) :
  calories_per_serving = 110 := sorry

end calories_in_serving_l272_272404


namespace raised_bed_section_area_l272_272069

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l272_272069


namespace geometric_probability_l272_272158

theorem geometric_probability (f : ℝ → ℝ) :
  (∀ x, f x = -x^2 + 2 * x) →
  (∀ x, x ∈ Icc (-1 : ℝ) 3) →
  (∃ p : ℚ, p = 1 / 2 ∧ 
  (∀ x0 : ℝ, x0 ∈ Icc (-1 : ℝ) 3 → f x0 ≥ 0 → x0 ∈ Icc 0 2) → 
  p = (2 - 0) / (3 - (-1))) :=
by
  sorry

end geometric_probability_l272_272158


namespace tangent_perpendicular_l272_272985

theorem tangent_perpendicular
  (a : ℝ)
  (curve : ℝ → ℝ)
  (line : ℝ → ℝ → Prop)
  (h_curve : curve = λ x, (2 - real.cos x) / real.sin x)
  (h_tangent_point : curve (real.pi / 2) = 2)
  (h_line : line x y = (x + a * y + 1 = 0)) :
  ∃ a : ℝ, a = 1 → 
  ∀ x y : ℝ, x = real.pi / 2 ∧ y = 2 →
  ∀ m₁ m₂ : ℝ, m₁ = 1 ∧ m₂ = - 1 / a →
  m₁ * m₂ = -1 := 
sorry

end tangent_perpendicular_l272_272985


namespace problem_result_l272_272911

theorem problem_result : (-1)^2022 + abs (1 - real.sqrt 2) + real.cbrt (-27) - real.sqrt ((-2)^2) = real.sqrt 2 - 5 := by
  sorry

end problem_result_l272_272911


namespace six_coins_all_heads_or_tails_probability_l272_272353

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l272_272353


namespace employee_payment_proof_l272_272440

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price as 20 percent above the wholesale cost
def retail_price (C_w : ℝ) : ℝ := C_w + 0.2 * C_w

-- Define the employee discount on the retail price
def employee_discount (C_r : ℝ) : ℝ := 0.15 * C_r

-- Define the amount paid by the employee
def amount_paid_by_employee (C_w : ℝ) : ℝ :=
  let C_r := retail_price C_w
  let D_e := employee_discount C_r
  C_r - D_e

-- Main theorem to prove the employee paid $204
theorem employee_payment_proof : amount_paid_by_employee wholesale_cost = 204 :=
by
  sorry

end employee_payment_proof_l272_272440


namespace time_to_tenth_pole_l272_272798

-- Definitions for the conditions
def num_poles : ℕ := 10
def poles_distance_equal : Bool := true
def time_to_sixth_pole : ℝ := 6.6
def constant_speed : Bool := true

-- The theorem to be proved
theorem time_to_tenth_pole : ℝ :=
  if poles_distance_equal ∧ constant_speed then
    let intervals : ℕ := 9
    let interval_time : ℝ := time_to_sixth_pole / 5
    intervals * interval_time
  else
    sorry

example : time_to_tenth_pole = 11.88 := by
  have : time_to_tenth_pole = (if poles_distance_equal ∧ constant_speed then 11.88 else sorry) :=
    by simp [time_to_tenth_pole]; sorry
  simp [this]

end time_to_tenth_pole_l272_272798


namespace range_of_g_l272_272930

def g (x : ℝ) : ℝ := ⌊2 * x⌋ - 2 * x

theorem range_of_g : set.Ico (-1 : ℝ) 0 = set.range g :=
sorry

end range_of_g_l272_272930


namespace binom_subtract_l272_272915

theorem binom_subtract :
  (Nat.choose 7 4) - 5 = 30 :=
by
  -- proof goes here
  sorry

end binom_subtract_l272_272915


namespace freq_percentage_in_neg_inf_to_50_l272_272871

-- Definitions of intervals and their frequencies
def freq_10_20 := 2
def freq_20_30 := 3
def freq_30_40 := 4
def freq_40_50 := 5
def total_samples := 20

-- Theorem statement to prove the frequency percentage in (-∞, 50] is 70%
theorem freq_percentage_in_neg_inf_to_50 : 
  (freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) / total_samples = 0.7 := 
by
  sorry

end freq_percentage_in_neg_inf_to_50_l272_272871


namespace train_length_l272_272050

/-
  Given:
  - Speed of the train is 78 km/h
  - Time to pass an electric pole is 5.0769230769230775 seconds
  We need to prove that the length of the train is 110 meters.
-/

def speed_kmph : ℝ := 78
def time_seconds : ℝ := 5.0769230769230775
def expected_length_meters : ℝ := 110

theorem train_length :
  (speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters :=
by {
  -- Proof goes here
  sorry
}

end train_length_l272_272050


namespace base_value_l272_272238

theorem base_value (b : ℕ) : (b - 1)^2 * (b - 2) = 256 → b = 17 :=
by
  sorry

end base_value_l272_272238


namespace solve_for_x_l272_272002

theorem solve_for_x (x : ℤ) (h : (3012 + x)^2 = x^2) : x = -1506 := 
sorry

end solve_for_x_l272_272002


namespace airplane_average_speed_l272_272511

theorem airplane_average_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 1584) (h_time : time = 24) : distance / time = 66 :=
by
  rw [h_distance, h_time]
  exact rfl

end airplane_average_speed_l272_272511


namespace six_coins_heads_or_tails_probability_l272_272355

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l272_272355


namespace probability_all_heads_or_tails_l272_272345

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l272_272345


namespace natalie_needs_12_bushes_for_60_zucchinis_l272_272091

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end natalie_needs_12_bushes_for_60_zucchinis_l272_272091


namespace find_varphi_l272_272191

noncomputable theory

def f (x ϕ a : ℝ) := sin (2 * x + ϕ) + a * cos (2 * x + ϕ)

theorem find_varphi (a ϕ : ℝ) (h_max : ∀ x, f x ϕ a ≤ 2) (h_sym : ∀ x, f x ϕ a = f ((π / 2) - x) ϕ a) (h_range : 0 < ϕ ∧ ϕ < π) :
  ϕ = π / 3 ∨ ϕ = 2 * π / 3 :=
sorry

end find_varphi_l272_272191


namespace probability_all_heads_or_tails_l272_272344

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l272_272344


namespace smallest_n_with_2020_divisors_l272_272126

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272126


namespace area_ABC_l272_272388

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨3, 4⟩
def B : Point := ⟨-3, 4⟩
def C : Point := ⟨-4, -3⟩

def dist (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def height := 7
def base := dist B C

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * base * height

theorem area_ABC : area_triangle A B C = 35 * Real.sqrt 2 / 2 := 
sorry

end area_ABC_l272_272388


namespace log_inequality_l272_272730

theorem log_inequality (a b c : ℝ) (h1 : 1 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  log a b + log b c + log c a ≤ log b a + log c b + log a c :=
sorry

end log_inequality_l272_272730


namespace smallest_number_has_2020_divisors_l272_272131

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272131


namespace sum_of_floors_eq_neg_squared_l272_272567

theorem sum_of_floors_eq_neg_squared : 
  (∑ k in Finset.range (1989 * 1990 + 1), (Int.floor (Real.sqrt k) + Int.floor (Real.sqrt (-k)))) = -1989^2 := by
  sorry

end sum_of_floors_eq_neg_squared_l272_272567


namespace problem_statement_l272_272278

variable (a b : Type) [LinearOrder a] [LinearOrder b]
variable (α β : Type) [LinearOrder α] [LinearOrder β]

-- Given conditions
def line_perpendicular_to_plane (l : Type) (p : Type) [LinearOrder l] [LinearOrder p] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

def lines_parallel (l1 : Type) (l2 : Type) [LinearOrder l1] [LinearOrder l2] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

theorem problem_statement (a b α : Type) [LinearOrder a] [LinearOrder b] [LinearOrder α]
(val_perp1 : line_perpendicular_to_plane a α)
(val_perp2 : line_perpendicular_to_plane b α)
: lines_parallel a b :=
sorry

end problem_statement_l272_272278


namespace log_inequality_l272_272206

theorem log_inequality (m n : ℝ) (h : m > n) (pos_n : n > 0) : 0.3 ^ m < 0.3 ^ n :=
sorry

end log_inequality_l272_272206


namespace base9_to_base10_l272_272506

theorem base9_to_base10 (c: ℕ) (d: ℕ) (e: ℕ)
  (h1 : c = 5) (h2 : d = 4) (h3 : e = 7) :
  c * 9^2 + d * 9^1 + e * 9^0 = 448 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end base9_to_base10_l272_272506


namespace order_of_abc_l272_272163

noncomputable def a : ℝ := 2⁻³
noncomputable def b : ℝ := Real.logBase 3 5
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = 2⁻³ := rfl
  have hb : b = Real.logBase 3 5 := rfl
  have hc : c = Real.cos (100 * Real.pi / 180) := rfl
  have ha_pos : 0 < a := by
    simp only [ha, inv_pos, pow_pos]
    norm_num
  have hb_pos : 1 < b := by
    simp only [hb, Real.logBase]
    exact Real.log_pos (by norm_num) (by norm_num)
  have hc_neg : c < 0 := by
    simp only [hc, Real.cos]
    exact Real.cos_pos_of_angle_pi_le (by norm_num)
  sorry

end order_of_abc_l272_272163


namespace find_m_l272_272164

variable (a : ℝ) (m : ℝ)

theorem find_m (h : a^(m + 1) * a^(2 * m - 1) = a^9) : m = 3 := 
by
  sorry

end find_m_l272_272164


namespace number_of_balls_l272_272237

noncomputable def frequency_of_yellow (n : ℕ) : ℚ := 9 / n

theorem number_of_balls (n : ℕ) (h1 : frequency_of_yellow n = 0.30) : n = 30 :=
by sorry

end number_of_balls_l272_272237


namespace find_blue_highlighters_l272_272223

theorem find_blue_highlighters
(h_pink : P = 9)
(h_yellow : Y = 8)
(h_total : T = 22)
(h_sum : P + Y + B = T) :
  B = 5 :=
by
  -- Proof would go here
  sorry

end find_blue_highlighters_l272_272223


namespace temperature_difference_in_fahrenheit_l272_272779

-- Define the conversion formula from Celsius to Fahrenheit as a function
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperatures in Boston and New York
variables (C_B C_N : ℝ)

-- Condition: New York is 10 degrees Celsius warmer than Boston
axiom temp_difference : C_N = C_B + 10

-- Goal: The temperature difference in Fahrenheit
theorem temperature_difference_in_fahrenheit : celsius_to_fahrenheit C_N - celsius_to_fahrenheit C_B = 18 :=
by sorry

end temperature_difference_in_fahrenheit_l272_272779


namespace original_book_pages_l272_272843

theorem original_book_pages (n k : ℕ) (h1 : (n * (n + 1)) / 2 - (2 * k + 1) = 4979)
: n = 100 :=
by
  sorry

end original_book_pages_l272_272843


namespace num_intersections_l272_272076

noncomputable def intersection_points (A : ℝ) (hA : 0 < A) :=
  {p : ℝ × ℝ | p.2 = A * p.1^2 ∧ p.1^2 + 5 = p.2^2 + 6 * p.2}

theorem num_intersections (A : ℝ) (hA : 0 < A) : 
  #((intersection_points A hA).to_finset) = 4 := by
  sorry

end num_intersections_l272_272076


namespace planes_not_perpendicular_l272_272678

-- Definition of the quadrilateral pyramid
structure Pyramid (P A B C D : Type) := 
(base_rect : Rectangle ABCD)
(perpendicular_PD : is_perpendicular PD ABCD)
(equals_PD_AD : PD = AD)
(equals_PD_a : PD = a)

-- Statement of proof problem
theorem planes_not_perpendicular (P A B C D : Type) [Pyramid P A B C D] : 
  ¬ is_perpendicular_plane PBA PBC := 
sorry

end planes_not_perpendicular_l272_272678


namespace adjacent_cells_diff_at_least_n_l272_272551

theorem adjacent_cells_diff_at_least_n (n : ℕ) (hn : 2 ≤ n) (f : Fin n.succ → Fin n.succ → ℕ)
  (hf : ∀ i j, f i j ∈ Finset.range (n * n + 1)) :
  ∃ (i1 i2 : Fin n.succ) (j1 j2 : Fin n.succ), 
    (abs ((f i1 j1) - (f i2 j2)) ≥ n ∧ 
     (((i1 = i2) ∧ (j1 = j2 + 1)) ∨ 
      ((i1 = i2) ∧ (j1 + 1 = j2)) ∨ 
      ((i1 + 1 = i2) ∧ (j1 = j2)) ∨ 
      ((i1 = i2 + 1) ∧ (j1 = j2)))) 
  := sorry

end adjacent_cells_diff_at_least_n_l272_272551


namespace correct_operation_l272_272838

theorem correct_operation :
  (∀ a b : ℝ, (sqrt 3 + sqrt 2 ≠ sqrt 5) ∧ 
              ((a + b) ^ 2 ≠ a^2 + b^2) ∧ 
              ((-2 * a * b^2) ^ 3 = -8 * a^3 * b^6) ∧ 
              ((-1 / 2) ^ (-2 : ℤ) ≠ -(1 / 4))) :=
by {
  intros a b,
  split,
  -- proof for sqrt 3 + sqrt 2 ≠ sqrt 5
  sorry,
  
  split,
  -- proof for (a + b) ^ 2 ≠ a^2 + b^2
  sorry,
  
  split,
  -- proof for (-2 * a * b^2) ^ 3 = -8 * a^3 * b^6
  sorry,
  
  -- proof for (-1 / 2) ^ (-2 : ℤ) ≠ -(1 / 4)
  sorry
}

end correct_operation_l272_272838


namespace square_inscribed_in_ellipse_area_l272_272495

theorem square_inscribed_in_ellipse_area :
  (∃ t : ℝ, (t > 0) ∧ (t^2 = 8/3)) →
  ∃ A : ℝ, A = (2 * sqrt (8/3))^2 ∧ A = 32/3 :=
by {
  intro ht,
  cases ht with t ht_props,
  use (2 * sqrt (8 / 3))^2,
  split,
  { 
    -- First part of proof: showing the computed area matches the calculation
    have area_computed : (2 * sqrt (8 / 3))^2 = 4 * (8 / 3),
    { 
      calc (2 * sqrt (8 / 3))^2 = 4 * (sqrt (8 / 3))^2 : by ring
      ... = 4 * (8 / 3) : by rw [Real.sqrt_sq (show 8 / 3 ≥ 0, by norm_num)]
    },
    exact area_computed,
  },
  { 
    -- Second part of proof: showing the area equals 32/3
    have area_value : 4 * (8 / 3) = 32 / 3,
    { 
      calc 4 * (8 / 3) = 32 / 3 : by ring,
    },
    exact area_value,
  }
}

end square_inscribed_in_ellipse_area_l272_272495


namespace solve_differential_eq_l272_272565

noncomputable def solution (x : ℝ) : ℝ := Real.arccos (1 / x^2)

theorem solve_differential_eq (y : ℝ → ℝ) (y' : ℝ → ℝ)
  (eq_diff : ∀ x, (x^3) * Real.sin (y x) * (y' x) = 2)
  (boundary_cond : filter.tendsto y at_top (nhds (Real.pi / 2)))
  (sol : ∀ x, y x = solution x) : 
  ∀ x, y x = Real.arccos (1 / x^2) :=
by
  sorry

end solve_differential_eq_l272_272565


namespace max_points_on_circle_d_l272_272321

open Real

/--
Problem: Given:
- Point Q is 7 cm away from the center of circle D.
- The radius of circle D is 4 cm.

Prove that the maximum number of points on circle D that are exactly 5 cm from point Q is 2.
-/
theorem max_points_on_circle_d (Q D : Point) (rD : ℝ) (dQD : ℝ) (rQ : ℝ) :
  rD = 4 → dQD = 7 → rQ = 5 → 
  ∃ max_points : ℕ, max_points = 2 :=
by
  intro h1 h2 h3
  use 2
  sorry

end max_points_on_circle_d_l272_272321


namespace largest_possible_perimeter_l272_272957

-- Definitions and conditions
def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

def sum_of_angles (a b c d : ℕ) : ℝ :=
  interior_angle a + interior_angle b + interior_angle c + interior_angle d

def total_sides_le_30 (a b c d : ℕ) : Prop := a + b + c + d ≤ 30

def non_overlapping (a b c d : ℕ) : Prop := True -- Placeholder, as non-overlapping condition is geometry-related and we are concerned only with calculations here

-- Main theorem
theorem largest_possible_perimeter (a b c d : ℕ) (ha : a = b) (hb : b = c) (habcd : sum_of_angles a b c d = 360) (hsides : total_sides_le_30 a b c d) : 
  (4 * a) = 24 → (4 * 1) = 4 → 20 := sorry

end largest_possible_perimeter_l272_272957


namespace find_AD_l272_272683

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272683


namespace train_length_l272_272894

theorem train_length (L V : ℝ) (h1 : L = V * 40) (h2 : L + 159.375 = V * 55) : L = 425 :=
by {
  sorry,
}

end train_length_l272_272894


namespace expression_value_l272_272595

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 :=
by
  sorry

end expression_value_l272_272595


namespace g_solution_l272_272781

def g (x : ℝ) : ℝ := 4^x - 3^x

theorem g_solution (x y : ℝ) (hg1 : g 1 = 1) (h : ∀ x y, g (x + y) = 4^y * g x + 3^x * g y) :
  g x = 4^x - 3^x := by
  sorry

end g_solution_l272_272781


namespace calculate_perimeters_of_isosceles_triangles_KLX_l272_272172

noncomputable def perimeter_isosceles_triangles_KLX (KL ML : ℝ) (X lies_on : X ∈ ℝ) : List ℝ :=
  if KL = 6 ∧ ML = 4 then
    [16.3, 16.3, 16]
  else
    []

theorem calculate_perimeters_of_isosceles_triangles_KLX :
  perimeter_isosceles_triangles_KLX 6 4 (anything_on_MN) = [16.3, 16.3, 16] :=
by
  sorry

end calculate_perimeters_of_isosceles_triangles_KLX_l272_272172


namespace num_ways_to_express_1000000_l272_272205

theorem num_ways_to_express_1000000 :
  let n := 1000000
  in ∃ (a b c : ℕ), 
    1 < a ∧ 1 < b ∧ 1 < c ∧ a * b * c = n ∧ 
    (∃! a' b' c' : ℕ, (a'=a ∨ a'=b ∨ a'=c) ∧ (b'=a ∨ b'=b ∨ b'=c) ∧ (c'=a ∨ c'=b ∨ c'=c)) :=
by
  sorry

end num_ways_to_express_1000000_l272_272205


namespace new_volume_l272_272477

theorem new_volume (l w h : ℝ) 
  (h1: l * w * h = 3000) 
  (h2: l * w + w * h + l * h = 690) 
  (h3: l + w + h = 40) : 
  (l + 2) * (w + 2) * (h + 2) = 4548 := 
  sorry

end new_volume_l272_272477


namespace pages_revised_twice_correct_l272_272757

-- Given conditions
variables 
  (pages : ℕ) -- total number of pages
  (rev_once : ℕ) -- number of pages revised once
  (rev_twice : ℕ) -- number of pages revised twice
  (cost_first_time : ℕ) -- cost per page for the first time it is typed
  (cost_revise_once : ℕ) -- cost per page for each revision
  (total_cost : ℕ) -- total cost 

-- Assumptions based on the conditions
def conditions := 
  pages = 100 ∧ 
  rev_once = 30 ∧ 
  (cost_first_time = 5) ∧ 
  (cost_revise_once = 3) ∧ 
  (total_cost = 710)

-- To prove: number of pages revised twice is 20
theorem pages_revised_twice_correct :
  conditions → rev_twice = 20 :=
by 
  sorry

end pages_revised_twice_correct_l272_272757


namespace sum_of_a_b_l272_272592

theorem sum_of_a_b (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : a + b = 7 ∨ a + b = -7 := 
by 
  sorry

end sum_of_a_b_l272_272592


namespace student_count_l272_272306

theorem student_count (rank_top rank_bottom : ℕ) (h1 : rank_top = 24) (h2 : rank_bottom = 34) : rank_top + rank_bottom - 1 = 57 :=
by
  rw [h1, h2]
  exact rfl

end student_count_l272_272306


namespace coord_point_P_l272_272320

theorem coord_point_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (2 * a - 1, a + 2))
  (h2 : P.2 = 0 ∨ P.1 = 0) : P = (-5, 0) ∨ P = (0, 2.5) := by
  cases h2
  case inl =>
    have : a + 2 = 0 := h2
    have : a = -2 := by linarith
    have : P.1 = (2 * a - 1) := by simpa using congr_arg Prod.fst h1
    have : P.1 = -5 := by linarith
    exact Or.inl (Prod.ext this h2)
  case inr =>
    have : 2 * a - 1 = 0 := h2
    have : a = 0.5 := by linarith
    have : P.2 = a + 2 := by simpa using congr_arg Prod.snd h1
    have : P.2 = 2.5 := by linarith
    exact Or.inr (Prod.ext h2 this)

end coord_point_P_l272_272320


namespace area_increase_percentage_l272_272856

def length_increase_square_a (s : ℝ) : ℝ := 2 * s

def length_increase_square_b (s : ℝ) : ℝ := 3.2 * (2 * s)

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def sum_areas (area_a area_b : ℝ) : ℝ := area_a + area_b

def percentage_increase (new_area original_area : ℝ) : ℝ := (new_area - original_area) / original_area * 100

theorem area_increase_percentage (s : ℝ) :
  let side_b := length_increase_square_a s in
  let side_c := length_increase_square_b s in
  let area_a := area_square s in
  let area_b := area_square side_b in
  let area_c := area_square side_c in
  let sum_ab := sum_areas area_a area_b in
  percentage_increase area_c sum_ab = 104.8 :=
by
  sorry

end area_increase_percentage_l272_272856


namespace six_coins_heads_or_tails_probability_l272_272358

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l272_272358


namespace three_digit_diff_no_repeated_digits_l272_272000

theorem three_digit_diff_no_repeated_digits :
  let largest := 987
  let smallest := 102
  largest - smallest = 885 := by
  sorry

end three_digit_diff_no_repeated_digits_l272_272000


namespace triangle_area_example_l272_272967

-- Define the right triangle DEF with angle at D being 45 degrees and DE = 8 units
noncomputable def area_of_45_45_90_triangle (DE : ℝ) (angle_d : ℝ) (h_angle : angle_d = 45) (h_DE : DE = 8) : ℝ :=
  1 / 2 * DE * DE

-- State the theorem to prove the area
theorem triangle_area_example {DE : ℝ} {angle_d : ℝ} (h_angle : angle_d = 45) (h_DE : DE = 8) :
  area_of_45_45_90_triangle DE angle_d h_angle h_DE = 32 := 
sorry

end triangle_area_example_l272_272967


namespace ratio_of_ages_l272_272334

theorem ratio_of_ages (sandy_future_age : ℕ) (sandy_years_future : ℕ) (molly_current_age : ℕ)
  (h1 : sandy_future_age = 42) (h2 : sandy_years_future = 6) (h3 : molly_current_age = 27) :
  (sandy_future_age - sandy_years_future) / gcd (sandy_future_age - sandy_years_future) molly_current_age = 
    4 / 3 :=
by
  sorry

end ratio_of_ages_l272_272334


namespace replacement_parts_l272_272403

theorem replacement_parts (num_machines : ℕ) (parts_per_machine : ℕ) (week1_fail_rate : ℚ) (week2_fail_rate : ℚ) (week3_fail_rate : ℚ) :
  num_machines = 500 ->
  parts_per_machine = 6 ->
  week1_fail_rate = 0.10 ->
  week2_fail_rate = 0.30 ->
  week3_fail_rate = 0.60 ->
  (num_machines * parts_per_machine) * week1_fail_rate +
  (num_machines * parts_per_machine) * week2_fail_rate +
  (num_machines * parts_per_machine) * week3_fail_rate = 3000 := by
  sorry

end replacement_parts_l272_272403


namespace part_I_part_II_l272_272962

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, -12)
noncomputable def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude_sum := magnitude vector_sum
noncomputable def magnitude_a := magnitude vector_a
noncomputable def magnitude_b := magnitude vector_b
noncomputable def cos_theta := dot_product vector_a vector_b / (magnitude_a * magnitude_b)

-- Prove the magnitude of the sum of vectors is 9√2
theorem part_I : vector_magnitude_sum = 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the cosine of the angle between the vectors is -16/65
theorem part_II : cos_theta = -16 / 65 :=
by
  sorry

end part_I_part_II_l272_272962


namespace simplify_expression_l272_272516

variable (x : ℝ)

theorem simplify_expression :
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 = -x^2 + 23 * x - 3 :=
sorry

end simplify_expression_l272_272516


namespace Problem1_part_a_Problem1_part_b_l272_272455

theorem Problem1_part_a (m l : ℕ) (hm : 0 < m) (hl : m ≤ l) :
  ∃ (n : ℕ) (x y : Fin n → ℕ),
    (∀ k ∈ {1, 2, ..., m-1} ∪ {m+1, ..., l}, ∑ i, (x i)^k = ∑ i, (y i)^k) ∧
    (∑ i, (x i)^m ≠ ∑ i, (y i)^m) :=
sorry

theorem Problem1_part_b (m l : ℕ) (hm : 0 < m) (hl : m ≤ l) :
  ∃ (n : ℕ) (x y : Fin n → ℕ),
    (∀ k ∈ {1, 2, ..., m-1} ∪ {m+1, ..., l}, ∑ i, (x i)^k = ∑ i, (y i)^k) ∧
    (∑ i, (x i)^m ≠ ∑ i, (y i)^m) ∧
    (Function.Injective x) ∧
    (Function.Injective y) :=
sorry

end Problem1_part_a_Problem1_part_b_l272_272455


namespace boys_at_park_l272_272398

theorem boys_at_park (girls parents groups people_per_group : ℕ) 
  (h_girls : girls = 14) 
  (h_parents : parents = 50)
  (h_groups : groups = 3) 
  (h_people_per_group : people_per_group = 25) : 
  (groups * people_per_group) - (girls + parents) = 11 := 
by 
  -- Not providing the proof, only the statement
  sorry

end boys_at_park_l272_272398


namespace matt_assignment_problems_l272_272719

theorem matt_assignment_problems (P : ℕ) (h : 5 * P - 2 * P = 60) : P = 20 :=
by
  sorry

end matt_assignment_problems_l272_272719


namespace smallest_n_for_cube_T_n_l272_272574

def greatest_power_of_3 (x : Nat) : Nat :=
  sorry

def T_n (n : Nat) : Nat :=
  Finset.sum (Finset.range (3^n + 1)) (λ k, greatest_power_of_3 (3 * k))

theorem smallest_n_for_cube_T_n : ∃ n : Nat, T_n n = (fun t => t ^ 3) 1 :=
  sorry

end smallest_n_for_cube_T_n_l272_272574


namespace intersection_AB_union_AB_complement_union_l272_272631

open Set

variable {U : Type} [hU : Nonempty U]

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 3 < x ∧ x < 10}
noncomputable def complement (S : Set ℝ) : Set ℝ := {x : ℝ | x ∉ S}

theorem intersection_AB : A ∩ B = {x : ℝ | 3 < x ∧ x < 7} :=
by
  sorry
  
theorem union_AB : A ∪ B = {x : ℝ | 2 ≤ x ∧ x < 10} :=
by
  sorry
  
theorem complement_union : complement A ∪ complement B = {x : ℝ | x ≤ 3 ∨ x ≥ 7} :=
by
  sorry

end intersection_AB_union_AB_complement_union_l272_272631


namespace surface_area_of_sphere_l272_272042

theorem surface_area_of_sphere :
  let a := 3
  let b := 4
  let c := 5
  let space_diagonal := Math.sqrt(a^2 + b^2 + c^2)
  let r := space_diagonal / 2
  let surface_area := 4 * Real.pi * r^2
  surface_area = 50 * Real.pi := by
  sorry

end surface_area_of_sphere_l272_272042


namespace number_of_solutions_eq_4_l272_272539

noncomputable def num_solutions := 
  ∃ n : ℕ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → (3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x = 0) → n = 4)

-- To state the above more clearly, we can add an abbreviation function for the equation.
noncomputable def equation (x : ℝ) : ℝ := 3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x

theorem number_of_solutions_eq_4 :
  (∃ n, n = 4 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → equation x = 0 → true) := sorry

end number_of_solutions_eq_4_l272_272539


namespace find_x_i_l272_272726

-- Given problem with conditions and solution
theorem find_x_i (n : ℕ) (h : 0 < n) (x : Fin n → ℝ) :
  (∑ i in Finset.range n, (i + 1) * sqrt (x (⟨i, Nat.lt_of_lt_succ (Finset.mem_range.mpr (Nat.lt_succ_self i))⟩) - (i+1)^2)) =
  (1 / 2) * (∑ i in Finset.range n, x (⟨i, Nat.lt_of_lt_succ (Finset.mem_range.mpr (Nat.lt_succ_self i))⟩)) →
  ∀ i : Fin n, x i = 2 * (@Fin.val ℕ n i + 1)^2 :=
  sorry

end find_x_i_l272_272726


namespace count_valid_n_l272_272514

def is_valid_n (N : ℕ) : Prop :=
  N >= 10 ∧ N < 100 ∧ 
  ∃ a b c d : ℕ, 
    0 ≤ a ∧ a < 4 ∧ 0 ≤ b ∧ b < 4 ∧ 
    0 ≤ c ∧ c < 7 ∧ 0 ≤ d ∧ d < 7 ∧ 
    N = 4 * a + b ∧ N = 7 * c + d ∧ 
    10 * (a + c) + (b + d) = a + b + c + d

theorem count_valid_n : {N : ℕ | is_valid_n N}.card = 5 :=
by
  sorry

end count_valid_n_l272_272514


namespace max_digit_sum_of_watch_display_l272_272872

-- Define the problem conditions
def valid_hour (h : ℕ) : Prop := 0 ≤ h ∧ h < 24
def valid_minute (m : ℕ) : Prop := 0 ≤ m ∧ m < 60
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the proof problem
theorem max_digit_sum_of_watch_display : 
  ∃ h m : ℕ, valid_hour h ∧ valid_minute m ∧ (digit_sum h + digit_sum m = 24) :=
sorry

end max_digit_sum_of_watch_display_l272_272872


namespace point_coordinates_in_second_quadrant_l272_272675

theorem point_coordinates_in_second_quadrant
    (P : ℝ × ℝ)
    (h1 : P.1 < 0)
    (h2 : P.2 > 0)
    (h3 : |P.2| = 4)
    (h4 : |P.1| = 5) :
    P = (-5, 4) :=
sorry

end point_coordinates_in_second_quadrant_l272_272675


namespace count_positive_integers_l272_272204

theorem count_positive_integers (n : ℤ) : 
  (∃ k, (k > 0) ∧ ∃ s, s = finset.count (λ n, (n + 10) * (n - 5) * (n - 15) < 0) (finset.range (k + 1)) ∧ s = 9) :=
begin
  sorry
end

end count_positive_integers_l272_272204


namespace problem_I_problem_II_problem_III_l272_272281

section Problem1

variables {R : Type*} [Field R] 
variable n : ℕ
variables (a1 a2 a3 : fin n → R)
variable λ : R

def A := { a | ∃ (x : fin n → R), a = x }

def eq (a b : fin n → R) : Prop := 
  ∀ k : fin n, a k = b k

def add (a b : fin n → R) : fin n → R := 
  λ k, a k + b k

def scalar_mul (λ : R) (a : fin n → R) : fin n → R := 
  λ k, λ * a k

def perfect_subset (B : set (fin n → R)) : Prop := 
  ∀ λ1 λ2 λ3, 
  (λ λ1 (B.to_finset.nth 0 hd λ 0) + λ λ2 (B.to_finset.nth 1 hd λ 0) + λ λ3 (B.to_finset.nth 2 hd λ 0) = 0) ↔ 
  λ1 = 0 ∧ λ2 = 0 ∧ λ3 = 0

def B1 := {vec_cons (1:R) (vec_cons 0 (vec_cons 0 vec_nil)), vec_cons 0 (vec_cons 1 (vec_cons 0 vec_nil)), vec_cons 0 (vec_cons 0 (vec_cons 1 vec_nil))} 
def B2 := {vec_cons (1:R) (vec_cons 2 (vec_cons 3 vec_nil)), vec_cons 2 (vec_cons 3 (vec_cons 4 vec_nil)), vec_cons 4 (vec_cons 5 (vec_cons 6 vec_nil))} 

-- Problem I proof problem
theorem problem_I : 
  perfect_subset 3 B1 ∧ ¬ perfect_subset B2 := sorry 

-- Define set B for problem II as described
def set_B (m : R) : set (fin 3 → R) :=
  {vec_cons (2 * m) (vec_cons m (vec_cons (m - 1) vec_nil)), 
   vec_cons m (vec_cons (2 * m) (vec_cons (m - 1) vec_nil)), 
   vec_cons m (vec_cons (m - 1) (vec_cons (2 * m) vec_nil))}

-- Problem II proof problem
theorem problem_II (h : ¬ perfect_subset (set_B (1/4:R))) : 
  ∃ m : R, m = 1/4 := sorry

-- Define set B and condition for problem III
def B_III := {vec_cons (x11 : R) (vec_cons x12 (vec_cons x13 vec_nil)), vec_cons x21 (vec_cons x22 (vec_cons x23 vec_nil)), vec_cons x31 (vec_cons x32 (vec_cons x33 vec_nil))}

def condition (a b c : R) := 2 * |a| > |a| + |b| + |c|

-- Problem III proof problem
theorem problem_III (h1: condition x11 x21 x31) (h2: condition x12 x22 x32) (h3: condition x13 x23 x33) :
  perfect_subset n B_III := sorry

end Problem1

end problem_I_problem_II_problem_III_l272_272281


namespace sequence_formula_l272_272558

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : a 3 = 7) (h4 : a 4 = 15) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end sequence_formula_l272_272558


namespace sample_points_on_line_have_correlation_coefficient_negative_one_l272_272666
noncomputable def sampleCorrelationCoefficient (n : ℕ) (x y : fin n → ℝ) : ℝ := sorry

theorem sample_points_on_line_have_correlation_coefficient_negative_one
  (n : ℕ) (hn : n ≥ 2) 
  (x y : fin n → ℝ) 
  (h_not_all_equal : ¬ (∀ i j, x i = x j)) 
  (h_line : ∀ i, y i = -3 * x i + 1) 
  : sampleCorrelationCoefficient n x y = -1 :=
by
  sorry

end sample_points_on_line_have_correlation_coefficient_negative_one_l272_272666


namespace expected_value_area_stddev_area_l272_272513

noncomputable theory

open MeasureTheory Probability

variables (X Y : ℝ)

/-- Expected value of the area of the resulting rectangle is 2 square meters. -/
theorem expected_value_area (hX : E X = 2) (hY : E Y = 1) (hindep : Independent X Y) :
  E (X * Y) = 2 := sorry

/-- Standard deviation of the area of the resulting rectangle is 50 square centimeters. -/
theorem stddev_area (hX : E X = 2) (hY : E Y = 1) 
  (varX : Var X = (0.003)^2) (varY : Var Y = (0.002)^2) (hindep : Independent X Y) :
  sqrt (Var (X * Y)) = 0.005 * 10000 := sorry

end expected_value_area_stddev_area_l272_272513


namespace find_unknown_rate_l272_272847

theorem find_unknown_rate :
    let n := 7 -- total number of blankets
    let avg_price := 150 -- average price of the blankets
    let total_price := n * avg_price
    let cost1 := 3 * 100
    let cost2 := 2 * 150
    let remaining := total_price - (cost1 + cost2)
    remaining / 2 = 225 :=
by sorry

end find_unknown_rate_l272_272847


namespace triangle_construction_two_solutions_l272_272531

theorem triangle_construction_two_solutions
  (a h_b m_b : ℝ) 
  (H1 : a > 0)
  (H2 : h_b > 0)
  (H3 : m_b > 0) :
  ∃ (Δ1 Δ2 : Type),
    (is_triangle Δ1 ∧
    is_triangle Δ2 ∧
    side_length Δ1 = a ∧
    side_length Δ2 = a ∧
    height_to_side Δ1 = h_b ∧
    height_to_side Δ2 = h_b ∧
    median_to_side Δ1 = m_b ∧
    median_to_side Δ2 = m_b) ∧
    Δ1 ≠ Δ2 ∧
    symmetric_wrt_height Δ1 Δ2 :=
by
  sorry

-- Definitions of the used predicates
def is_triangle (Δ : Type) : Prop := sorry
def side_length (Δ : Type) : ℝ := sorry
def height_to_side (Δ : Type) : ℝ := sorry
def median_to_side (Δ : Type) : ℝ := sorry
def symmetric_wrt_height (Δ1 Δ2 : Type) : Prop := sorry

end triangle_construction_two_solutions_l272_272531


namespace infinitely_many_n_l272_272859

theorem infinitely_many_n (h : ℤ) : ∃ (S : Set ℤ), S ≠ ∅ ∧ ∀ n ∈ S, ∃ k : ℕ, ⌊n * Real.sqrt (h^2 + 1)⌋ = k^2 :=
by
  sorry

end infinitely_many_n_l272_272859


namespace max_lassis_l272_272913

/-- Prove that given the lassi production ratios, the maximum number of lassis Caroline can make
    with 12 mangoes and 20 coconuts is 55. --/
theorem max_lassis (mangoes coconuts : ℕ) (lassis_per_2_mangoes lassis_per_4_coconuts : ℕ) :
  mangoes = 12 →
  coconuts = 20 →
  lassis_per_2_mangoes = 11 →
  lassis_per_4_coconuts = 11 →
  max_lassis mangoes coconuts lassis_per_2_mangoes lassis_per_4_coconuts = 55 :=
by
  sorry

end max_lassis_l272_272913


namespace opposite_of_fraction_l272_272790

def opposite_of (x : ℚ) : ℚ := -x

theorem opposite_of_fraction :
  opposite_of (1/2023) = - (1/2023) :=
by
  sorry

end opposite_of_fraction_l272_272790


namespace option_b_correct_l272_272973

variables {a b c : ℝ^2}

-- Option B conditions 
variable (a_nonzero : a ≠ 0)
variable (b_ne_c : b ≠ c)
variable (dot_equal : a.dot b = a.dot c)

-- Theorem to be proved
theorem option_b_correct : a.dot (b - c) = 0 ↔ a ⊥ (b - c) :=
sorry

end option_b_correct_l272_272973


namespace count_true_statements_l272_272648

theorem count_true_statements (x : ℝ) (h : x > -3) :
  (if (x > -3 → x > -6) then 1 else 0) +
  (if (¬ (x > -3 → x > -6)) then 1 else 0) +
  (if (x > -6 → x > -3) then 1 else 0) +
  (if (¬ (x > -6 → x > -3)) then 1 else 0) = 2 :=
sorry

end count_true_statements_l272_272648


namespace A_plus_B_eq_93_l272_272285

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end A_plus_B_eq_93_l272_272285


namespace sophomores_selected_l272_272029

variables (total_students freshmen sophomores juniors selected_students : ℕ)
def high_school_data := total_students = 2800 ∧ freshmen = 970 ∧ sophomores = 930 ∧ juniors = 900 ∧ selected_students = 280

theorem sophomores_selected (h : high_school_data total_students freshmen sophomores juniors selected_students) :
  (930 / 2800 : ℚ) * 280 = 93 := by
  sorry

end sophomores_selected_l272_272029


namespace infinite_6s_in_sequence_l272_272017

-- Define the sequence generator as described
def generate_sequence (initial_seq : List ℕ) : Stream ℕ := sorry

-- Define the property that a sequence contains infinitely many 6's
def infinitely_many_6s (s : Stream ℕ) : Prop :=
  ∀ n, ∃ m > n, s.get m = 6

-- Initial data given in the problem
def initial_sequence : List ℕ := [7, 7, 4, 9, 2, 8, 3, 6, 1, 8, 1, 6, 2, 4, 1, 8, 6, 8, 8, 6, 1, 2, 8]

theorem infinite_6s_in_sequence :
  infinitely_many_6s (generate_sequence initial_sequence) :=
  sorry

end infinite_6s_in_sequence_l272_272017


namespace smallest_number_with_2020_divisors_l272_272122

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272122


namespace problem_geometry_l272_272408

/-- Two points P and Q lie on the same side of line XY such that triangles XYP and XYQ are congruent.
    Given XY = 12, YP = QX = 13, and PX = QY = 20, the intersection of the triangular regions has 
    area p / q, where p and q are relatively prime positive integers. Prove that p + q = 4594. --/
theorem problem_geometry (XY YP QX PX QY : ℝ) (hXY: XY = 12) (hYP: YP = 13) (hQX: QX = 13) (hPX: PX = 20) (hQY: QY = 20) :
  ∃ (p q : ℕ), gcd p q = 1 ∧ p + q = 4594 ∧ ((XY * 19.1) / (YP * 20)) = (p : ℝ) / (q : ℝ) :=
begin
  sorry,
end

end problem_geometry_l272_272408


namespace ratio_difference_l272_272852

variables (p q r : ℕ) (x : ℕ)
noncomputable def shares_p := 3 * x
noncomputable def shares_q := 7 * x
noncomputable def shares_r := 12 * x

theorem ratio_difference (h1 : shares_q - shares_p = 2400) : shares_r - shares_q = 3000 :=
by sorry

end ratio_difference_l272_272852


namespace can_be_factored_using_formulas_l272_272842

   variable (x y : ℝ)
   def P1 := x^2 + 4
   def P2 := x^2 - x + 1/4
   def P3 := x^2 + 2x + 4
   def P4 := x^2 - 4 * y

   theorem can_be_factored_using_formulas : (∃ a : ℝ, ∃ b : ℝ, P2 = (x - a) * (x - a)) :=
   by
     exists 1/2
     exists 1/2
     sorry
   
end can_be_factored_using_formulas_l272_272842


namespace marble_prob_l272_272862

def total_marbles := 20
def blue_marbles := 6
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

def prob_red_or_white := (red_marbles + white_marbles) / total_marbles.to_float

theorem marble_prob : prob_red_or_white = 7 / 10 := by 
  sorry

end marble_prob_l272_272862


namespace count_positive_integers_count_of_positive_integers_l272_272579

theorem count_positive_integers (n : ℕ) :
  (150 ≤ n^2 ∧ n^2 ≤ 300) → n ∈ {13, 14, 15, 16, 17} :=
begin
  sorry
end

theorem count_of_positive_integers :
  {n : ℕ | 150 ≤ n^2 ∧ n^2 ≤ 300}.card = 5 :=
begin
  sorry
end

end count_positive_integers_count_of_positive_integers_l272_272579


namespace cost_of_one_pencil_l272_272301

def total_spent : ℕ := 74
def cost_of_notebook : ℕ := 35
def cost_of_ruler : ℕ := 18
def number_of_pencils : ℕ := 3

theorem cost_of_one_pencil : (total_spent - (cost_of_notebook + cost_of_ruler)) / number_of_pencils = 7 :=
by
  -- Lean will calculate the above expression to ensure it equals 7
  let cost_of_pencils := total_spent - (cost_of_notebook + cost_of_ruler)
  have h1 : cost_of_pencils = 21 := by decide
  have h2 : 21 / number_of_pencils = 7 := by decide
  rw [h1, h2]
  exact rfl

end cost_of_one_pencil_l272_272301


namespace square_area_in_ellipse_l272_272482

theorem square_area_in_ellipse (t : ℝ) (ht : 0 < t) :
  (t ^ 2 / 4 + t ^ 2 / 8 = 1) →
  let side_length := 2 * t in
  let area := side_length ^ 2 in
  area = 32 / 3 := 
sorry

end square_area_in_ellipse_l272_272482


namespace trigonometric_identity_l272_272007

theorem trigonometric_identity (α : ℝ) :
  (tan (4 * α) + sec (4 * α) = (cos (2 * α) + sin (2 * α)) / (cos (2 * α) - sin (2 * α))) :=
by
  sorry

end trigonometric_identity_l272_272007


namespace malcolm_needs_more_lights_l272_272298

def red_lights := 12
def blue_lights := 3 * red_lights
def green_lights := 6
def white_lights := 59

def colored_lights := red_lights + blue_lights + green_lights
def need_more_lights := white_lights - colored_lights

theorem malcolm_needs_more_lights :
  need_more_lights = 5 :=
by
  sorry

end malcolm_needs_more_lights_l272_272298


namespace can_partition_to_equal_hexafaced_polyhedra_l272_272533

-- Define a RegularTetrahedron type and avail an instance of it
structure RegularTetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ
  is_regular : ∀ i j k l, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
                  dist (vertices i) (vertices j) = dist (vertices i) (vertices k)

-- Define the main theorem to prove the partitioning
theorem can_partition_to_equal_hexafaced_polyhedra (T : RegularTetrahedron) :
    ∃ P : ℕ → (ℝ × ℝ × ℝ) set, (∀ i, (convex_hull ℝ (P i)).card = 6) ∧
    (∀ i j, i ≠ j → disjoint (P i) (P j)) ∧
    (⋃ i, P i) = convex_hull ℝ (set.range T.vertices) := 
sorry

end can_partition_to_equal_hexafaced_polyhedra_l272_272533


namespace Luca_fruit_baskets_unique_ratios_l272_272739

theorem Luca_fruit_baskets_unique_ratios :
  let apples := 7
  let oranges := 5
  let basket_compositions (x y : ℕ) := x ≥ 1 ∧ x ≤ apples ∧ y ≥ 1 ∧ y ≤ oranges ∧ Nat.gcd x y = 1
  {xy | ∃ x y, basket_compositions x y}.card = 27 :=
by
  let apples := 7
  let oranges := 5
  let basket_compositions (x y : ℕ) := x ≥ 1 ∧ x ≤ apples ∧ y ≥ 1 ∧ y ≤ oranges ∧ Nat.gcd x y = 1
  sorry

end Luca_fruit_baskets_unique_ratios_l272_272739


namespace game_players_exceed_30000_l272_272026

theorem game_players_exceed_30000 :
  ∀ (R: ℕ → ℝ) (k R₀ : ℝ),
  (R 0 = 100) →
  (R 5 = 1000) →
  (∀ t, R t = R₀ * exp (k * t)) →
  (∃ t: ℕ, t ≥ 13 ∧ R t > 30000) := by
  sorry

end game_players_exceed_30000_l272_272026


namespace smallest_n_with_divisors_2020_l272_272138

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272138


namespace least_xy_l272_272180

noncomputable def condition (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 1 / 7)

theorem least_xy (x y : ℕ) (h : condition x y) : x * y = 98 :=
sorry

end least_xy_l272_272180


namespace max_min_diff_z_l272_272735

theorem max_min_diff_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 29) :
  let z_max := max_val z, z_min := min_val z in
  z_max - z_min = 20 / 3 :=
sorry

end max_min_diff_z_l272_272735


namespace min_distance_le_one_l272_272366

noncomputable def complex_numbers_modulus (z : ℂ) : ℝ := complex.abs z

variables (z1 z2 z3 w1 w2 : ℂ)

-- Conditions
def modulus_not_greater_than_one (z : ℂ) : Prop := complex_numbers_modulus z ≤ 1

def is_root_of_equation (z w1 w2 : ℂ) : Prop := 
  (z - z1) * (z - z2) + (z - z2) * (z - z3) + (z - z3) * (z - z1) = 0

axiom (h1 : modulus_not_greater_than_one z1)
axiom (h2 : modulus_not_greater_than_one z2)
axiom (h3 : modulus_not_greater_than_one z3)

axiom (hw1 : is_root_of_equation w1 z1 z2 z3)
axiom (hw2 : is_root_of_equation w2 z1 z2 z3)

theorem min_distance_le_one : ∀ j : ℕ, (j = 1 ∨ j = 2 ∨ j = 3) →
  (complex.dist (if j = 1 then z1 else if j = 2 then z2 else z3) w1 ≃ 1 ∨
   complex.dist (if j = 1 then z1 else if j = 2 then z2 else z3) w2 ≃ 1) ∨
  (complex.dist (if j = 1 then z1 else if j = 2 then z2 else z3) w1 ≤ 1 ∨
   complex.dist (if j = 1 then z1 else if j = 2 then z2 else z3) w2 ≤ 1) :=
sorry

end min_distance_le_one_l272_272366


namespace sum_of_sequence_l272_272603

def a_n (n : ℕ) : ℕ := (2 * n + 1) * 2^(n - 1)

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a_n (i + 1)

theorem sum_of_sequence (n : ℕ) : S n = 2 * n * 2^n - 1 := by
  sorry

end sum_of_sequence_l272_272603


namespace solve_equation_1_solve_equation_2_solve_equation_3_l272_272365

theorem solve_equation_1 : ∀ x : ℝ, (4 * (x + 3) = 25) ↔ (x = 13 / 4) :=
by
  sorry

theorem solve_equation_2 : ∀ x : ℝ, (5 * x^2 - 3 * x = x + 1) ↔ (x = -1 / 5 ∨ x = 1) :=
by
  sorry

theorem solve_equation_3 : ∀ x : ℝ, (2 * (x - 2)^2 - (x - 2) = 0) ↔ (x = 2 ∨ x = 5 / 2) :=
by
  sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l272_272365


namespace complex_properties_l272_272169

variable (z : ℂ)

noncomputable def z_def : ℂ :=
  Complex.ofReal 1 - Complex.I

theorem complex_properties
  (h : z * Complex.I = 1 + Complex.I) :
  -- Prove that the conjugate of z equals 1 + i
  Complex.conj z = 1 + Complex.I ∧
  -- Prove that the point corresponding to z lies in the fourth quadrant
  (z.re > 0 ∧ z.im < 0) ∧
  -- Prove that z satisfies the quadratic equation z^2 - 2z + 2 = 0
  z^2 - 2 * z + 2 = 0 := by
  sorry

end complex_properties_l272_272169


namespace smallest_number_with_2020_divisors_is_correct_l272_272115

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272115


namespace raised_bed_section_area_l272_272068

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l272_272068


namespace permutation_of_tourists_l272_272226

-- Definition of the problem setup.
def num_cinemas : ℕ
def num_tourists : ℕ

-- Condition in the problem where the number of cinemas equals the number of tourists.
axiom h_eq : num_cinemas = num_tourists

-- Theorem to state the number of permutations (ways to distribute tourists).
theorem permutation_of_tourists : ∃ n : ℕ, ∀ n_cinemas n_tourists, 
  (n_cinemas = n_tourists) → (num_tourists = n) → (n.factorial = num_cinemas.factorial) := 
by 
  sorry

end permutation_of_tourists_l272_272226


namespace sum_first_10_terms_l272_272174

noncomputable def a_n (n : ℕ) : ℝ :=
if n % 2 = 1 then 2 / (n * (n + 2))
else real.log ((n + 2) / n)

theorem sum_first_10_terms :
  (∑ n in finset.range 10, a_n (n + 1)) = (10 / 11) + real.log 6 :=
by
  sorry

end sum_first_10_terms_l272_272174


namespace max_rectangle_area_l272_272387

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 48) : x * y ≤ 144 :=
by
  sorry

end max_rectangle_area_l272_272387


namespace area_AMC_eq_half_l272_272016

-- Definitions of conditions
variables (A B C M : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ M]
variable (area_ABC : ℝ)
variable (is_angle_bisector : ∀ (x : B), on_angle_bisector M x C)
variable (perpendicular : ∀ (x : B), is_perpendicular M x)

-- Statement of the theorem
theorem area_AMC_eq_half :
  area_ABC = 1 →
  is_angle_bisector M C → 
  perpendicular M B →
  area AMC = 1 / 2 :=
by
  sorry

end area_AMC_eq_half_l272_272016


namespace new_quadratic_coeff_l272_272998

theorem new_quadratic_coeff (r s p q : ℚ) 
  (h1 : 3 * r^2 + 4 * r + 2 = 0)
  (h2 : 3 * s^2 + 4 * s + 2 = 0)
  (h3 : r + s = -4 / 3)
  (h4 : r * s = 2 / 3) 
  (h5 : r^3 + s^3 = - p) :
  p = 16 / 27 :=
by
  sorry

end new_quadratic_coeff_l272_272998


namespace needed_supplemental_tanks_l272_272904

-- Given conditions represented as definitions
def primary_tank_duration : ℕ := 2
def total_diving_time : ℕ := 8
def duration_per_supplemental_tank : ℕ := 1

-- Proof statement that calculates the number of supplemental tanks needed
theorem needed_supplemental_tanks : 
  total_diving_time - primary_tank_duration = 6 →
  6 / duration_per_supplemental_tank = 6 :=
by
  intro h
  simp [duration_per_supplemental_tank]
  exact h

#check needed_supplemental_tanks

end needed_supplemental_tanks_l272_272904


namespace find_question_mark_l272_272414

-- Define variables
def a := 47 / 100 * 1442
def b := 36 / 100 * 1412
def diff := a - b
def result := 252

-- Prove that ? (question mark) is equal to 82.58
theorem find_question_mark : (diff + 82.58 = result) := by
  -- Sorry as the placeholder for the proof
  sorry

end find_question_mark_l272_272414


namespace find_AD_l272_272705

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272705


namespace square_inscribed_in_ellipse_area_l272_272497

theorem square_inscribed_in_ellipse_area :
  (∃ t : ℝ, (t > 0) ∧ (t^2 = 8/3)) →
  ∃ A : ℝ, A = (2 * sqrt (8/3))^2 ∧ A = 32/3 :=
by {
  intro ht,
  cases ht with t ht_props,
  use (2 * sqrt (8 / 3))^2,
  split,
  { 
    -- First part of proof: showing the computed area matches the calculation
    have area_computed : (2 * sqrt (8 / 3))^2 = 4 * (8 / 3),
    { 
      calc (2 * sqrt (8 / 3))^2 = 4 * (sqrt (8 / 3))^2 : by ring
      ... = 4 * (8 / 3) : by rw [Real.sqrt_sq (show 8 / 3 ≥ 0, by norm_num)]
    },
    exact area_computed,
  },
  { 
    -- Second part of proof: showing the area equals 32/3
    have area_value : 4 * (8 / 3) = 32 / 3,
    { 
      calc 4 * (8 / 3) = 32 / 3 : by ring,
    },
    exact area_value,
  }
}

end square_inscribed_in_ellipse_area_l272_272497


namespace find_length_of_side_of_triangle_ABO_l272_272752

noncomputable def length_of_side_of_isosceles_right_triangle : ℝ :=
  let graph_func : ℝ → ℝ := λ x, -x^2
  let A : ℝ × ℝ := (1, graph_func 1)
  let B : ℝ × ℝ := (-1, graph_func (-1))
  let O : ℝ × ℝ := (0, 0)
  let side_length := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)
  in 2

theorem find_length_of_side_of_triangle_ABO :
  ∃ (side_length : ℝ), side_length = length_of_side_of_isosceles_right_triangle :=
by
  use 2
  sorry

end find_length_of_side_of_triangle_ABO_l272_272752


namespace natalie_bushes_l272_272089

theorem natalie_bushes (bush_yield : ℕ) (containers_per_zucchini : ℕ → ℕ) (desired_zucchinis : ℕ):
  (bush_yield = 10) →
  (containers_per_zucchini 1 = 2) →
  (desired_zucchinis = 60) →
  ∃ bushes_needed : ℕ, bushes_needed = 12 :=
by
  intros h_bush_yield h_containers_per_zucchini h_desired_zucchinis
  use 12
  sorry

end natalie_bushes_l272_272089


namespace clock_angle_7_30_l272_272422

theorem clock_angle_7_30 :
  let hour_mark_angle := 30
  let minute_mark_angle := 6
  let hour_hand_angle := 7 * hour_mark_angle + (30 * hour_mark_angle / 60)
  let minute_hand_angle := 30 * minute_mark_angle
  let angle_diff := abs (hour_hand_angle - minute_hand_angle)
  angle_diff = 45 := by
  sorry

end clock_angle_7_30_l272_272422


namespace find_x_l272_272766

theorem find_x (a b x : ℝ) (h : ∀ a b, a * b = a + 2 * b) (H : 3 * (4 * x) = 6) : x = -5 / 4 :=
by
  sorry

end find_x_l272_272766


namespace fraction_of_fresh_berries_to_keep_l272_272262

def Iris_berry_problem_blueberries : ℕ := 30
def Iris_berry_problem_cranberries : ℕ := 20
def Iris_berry_problem_raspberries : ℕ := 10
def Iris_berry_problem_fraction_rotten : ℚ := 1 / 3
def Iris_berry_problem_berries_to_sell : ℕ := 20

theorem fraction_of_fresh_berries_to_keep :
  let total_berries := Iris_berry_problem_blueberries + Iris_berry_problem_cranberries + Iris_berry_problem_raspberries in
  let rotten_berries := Iris_berry_problem_fraction_rotten * total_berries in
  let fresh_berries := total_berries - rotten_berries.natAbs in
  let berries_to_keep := fresh_berries - Iris_berry_problem_berries_to_sell in
  (berries_to_keep : ℚ) / fresh_berries = 1 / 2 :=
by
  sorry

end fraction_of_fresh_berries_to_keep_l272_272262


namespace angle_AFB_108_degrees_l272_272926

-- Define a regular pentagon and its properties
structure regular_pentagon (P : Type) [euclidean_geometry P] :=
(A B C D E : P)
(all_sides_equal : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A)
(all_angles_equal : ∀ a : P, angle A B C = 108 ∧ angle B C D = 108 ∧ angle C D E = 108 ∧ angle D E A = 108 ∧ angle E A B = 108)

-- Define the intersection point F of diagonals CA and EB
def intersection_point (P : Type) [euclidean_geometry P] (p : regular_pentagon P) : P :=
  let ⟨A, B, C, D, E, _, _⟩ := p in
  euclidean_geometry.inter_point P (line C A) (line E B)

-- Define the theorem we need to prove
theorem angle_AFB_108_degrees (P : Type) [euclidean_geometry P] (p : regular_pentagon P) :
  let F := intersection_point P p in
  ∃ F, angle (pointA p) F (pointB p) = 108 := by sorry

end angle_AFB_108_degrees_l272_272926


namespace ticket_price_increase_l272_272548

-- Definitions as per the conditions
def old_price : ℝ := 85
def new_price : ℝ := 102
def percent_increase : ℝ := (new_price - old_price) / old_price * 100

-- Statement to prove the percent increase is 20%
theorem ticket_price_increase : percent_increase = 20 := by
  sorry

end ticket_price_increase_l272_272548


namespace common_tangent_circumcircles_l272_272249

variable (A B C D M : Type)
variable [IsParallelogram A B C D]
variable (AC BD : Type)
variable [IsLongerDiagonal A B C D AC BD]
variable [IsCircumscribedQuadrilateral B C D M]

theorem common_tangent_circumcircles :
  IsCommonTangent BD (Circumcircle (A B M)) (Circumcircle (A D M)) :=
by
  sorry

end common_tangent_circumcircles_l272_272249


namespace factory_shirts_production_l272_272901

theorem factory_shirts_production :
  let A_shirts := 7 * 20 in
  let B_shirts := (5 / 2) * 30 in
  let C_shirts := (3 / 3) * 40 in
  A_shirts + B_shirts + C_shirts = 255 :=
by
  -- Definitions based on the problem statement
  let A_shirts := 7 * 20
  let B_shirts := (5 / 2) * 30
  let C_shirts := (3 / 3) * 40
  -- Assumption placeholders for Lean's requirement
  have hA : A_shirts = 140 := by sorry
  have hB : B_shirts = 75 := by sorry
  have hC : C_shirts = 40 := by sorry
  -- Total shirts calculation
  have total := A_shirts + B_shirts + C_shirts
  -- Final verification
  have : total = 255 := eq.trans (eq.trans (congrArg (+ B_shirts) hA) (congrArg (+ C_shirts) hB)) (congrArg (A_shirts + B_shirts +) hC)
  exact this

end factory_shirts_production_l272_272901


namespace length_of_PQ_l272_272727

open EuclideanGeometry

theorem length_of_PQ
  (ω : circle) (A B C D E P Q : Point)
  (h_diameter : diameter ω A B)
  (h_on_circle : C ∈ ω ∧ C ≠ A ∧ C ≠ B)
  (h_perpendicular : ∃ D, is_perpendicular C D ∧ AB.contains D)
  (h_intersect : ∃ E, ω.perpendicular_intersection C AB E ∧ E ≠ C)
  (h_circle : ∃ P Q, circle.centered C (C.distance D).radius ω.intersects P Q)
  (h_perimeter : 2 * (C.distance P) + (P.distance Q) = 24) : P.distance Q = 8 :=
by
  sorry

end length_of_PQ_l272_272727


namespace distribution_value_l272_272395

def standard_deviation := 2
def mean := 51

theorem distribution_value (x : ℝ) (hx : x < 45) : (mean - 3 * standard_deviation) > x :=
by
  -- Provide the statement without proof
  sorry

end distribution_value_l272_272395


namespace single_digit_A_exists_l272_272873

theorem single_digit_A_exists :
  ∃ (A : ℕ), (A < 10) ∧ (3.4 < 3 + A / 10) ∧ (3 + A / 10 < 4) ∧ (4 / 3 > 4 / A) ∧ (4 / A > 2 / 3) ∧ A = 5 :=
by sorry

end single_digit_A_exists_l272_272873


namespace assign_grades_l272_272885

-- Definitions based on the conditions:
def num_students : ℕ := 12
def num_grades : ℕ := 4

-- Statement of the theorem
theorem assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end assign_grades_l272_272885


namespace area_of_square_inscribed_in_ellipse_l272_272492

noncomputable def areaSquareInscribedInEllipse : ℝ :=
  let a := 4
  let b := 8
  let s := (2 * Real.sqrt (b / 3)).toReal in
  (2 * s) ^ 2

theorem area_of_square_inscribed_in_ellipse :
  (areaSquareInscribedInEllipse) = 32 / 3 :=
sorry

end area_of_square_inscribed_in_ellipse_l272_272492


namespace general_formula_sum_of_first_three_terms_l272_272605

-- Definitions
variable {a b : ℕ → ℤ}
variable {S T : ℕ → ℤ}
variable (d : ℤ) (q : ℤ)

-- Conditions
def initial_conditions : Prop :=
  a 1 = 2 ∧ b 1 = 1 ∧ b 3 = 3 + a 2 ∧ b 2 = -2 * a 4

def geometric_sequence_condition : Prop :=
  T 3 = 13

-- Statements to prove
theorem general_formula (h : initial_conditions d q) : 
  ∀ n, b n = 2 ^ (n - 1) := sorry

theorem sum_of_first_three_terms (h : initial_conditions d q) (hT : geometric_sequence_condition d q) :
  S 3 = 18 := sorry

end general_formula_sum_of_first_three_terms_l272_272605


namespace angle_between_clock_hands_at_7_30_l272_272419

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l272_272419


namespace correct_statement_is_d_l272_272900

theorem correct_statement_is_d (a b : ℝ) : 
  ¬(|a - b| = 3) ∧
  (∀ (s : Type), is_square s → (axes_of_symmetry s = 4)) ∧
  (∀ (t : Type), is_isosceles_triangle t → 
    (is_acute_triangle t ∨ is_right_triangle t ∨ is_obtuse_triangle t)) ∧
  (∀ (t : Type), is_isosceles_triangle t → is_axisymmetric_figure t) →
  (∀ (t : Type), is_isosceles_triangle t → is_axisymmetric_figure t) :=
by 
  sorry

end correct_statement_is_d_l272_272900


namespace linear_regression_change_l272_272966

theorem linear_regression_change (x : ℝ) :
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  y2 - y1 = -1.5 := by
  -- y1 = 2 - 1.5 * x
  -- y2 = 2 - 1.5 * x - 1.5
  -- Δ y = y2 - y1
  sorry

end linear_regression_change_l272_272966


namespace geometric_seq_20th_term_l272_272778

theorem geometric_seq_20th_term (a r : ℕ)
  (h1 : a * r ^ 4 = 5)
  (h2 : a * r ^ 11 = 1280) :
  a * r ^ 19 = 2621440 :=
sorry

end geometric_seq_20th_term_l272_272778


namespace clock_angle_7_30_l272_272421

theorem clock_angle_7_30 :
  let hour_mark_angle := 30
  let minute_mark_angle := 6
  let hour_hand_angle := 7 * hour_mark_angle + (30 * hour_mark_angle / 60)
  let minute_hand_angle := 30 * minute_mark_angle
  let angle_diff := abs (hour_hand_angle - minute_hand_angle)
  angle_diff = 45 := by
  sorry

end clock_angle_7_30_l272_272421


namespace probability_all_heads_or_tails_l272_272347

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l272_272347


namespace proof_f_value_and_a_l272_272990

noncomputable def f : ℝ → ℝ := λ x,
  if h : x < 0 then log 3 (-x) else 3^(x-2)

theorem proof_f_value_and_a (a : ℝ) (h : f a = 3) :
  f 2 = 1 ∧ (a = -27 ∨ a = 3) :=
by
  have h1 : f 2 = 1, from sorry,
  split
  · exact h1
  have h2 : a = -27 ∨ a = 3, from sorry,
  exact h2

end proof_f_value_and_a_l272_272990


namespace scientific_notation_of_18860000_l272_272338

theorem scientific_notation_of_18860000 : 
  ∃ a : ℝ, ∃ b : ℤ, (18_860_000 = a * 10^b) ∧ (a = 1.886) ∧ (b = 7) :=
by
  exists 1.886
  exists 7
  split
  -- Placeholder proofs
  . exact sorry
  split
  . exact rfl
  . exact rfl

end scientific_notation_of_18860000_l272_272338


namespace sin_675_eq_neg_sqrt2_over_2_l272_272919

theorem sin_675_eq_neg_sqrt2_over_2 :
  sin (675 * Real.pi / 180) = - (Real.sqrt 2 / 2) := 
by
  -- problem states that 675° reduces to 315°
  have h₁ : (675 : ℝ) ≡ 315 [MOD 360], by norm_num,
  
  -- recognize 315° as 360° - 45°
  have h₂ : (315 : ℝ) = 360 - 45, by norm_num,

  -- in the fourth quadrant, sin(315°) = -sin(45°)
  have h₃ : sin (315 * Real.pi / 180) = - (sin (45 * Real.pi / 180)), by
    rw [Real.sin_angle_sub_eq_sin_add, Real.sin_angle_eq_sin_add],
    
  -- sin(45°) = sqrt(2)/2
  have h₄ : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by
    -- As an assumed known truth for this problem
    exact Real.sin_pos_of_angle,

  -- combine above facts
  rw [h₃, h₄],
  norm_num
  -- sorry is needed if proof steps aren't complete
  sorry

end sin_675_eq_neg_sqrt2_over_2_l272_272919


namespace sum_of_100th_group_l272_272086

-- Definitions of the sequence and the grouping function
def sequence (n : ℕ) : ℕ := 2 * n + 1

def group_size (n : ℕ) : ℕ :=
  if n % 4 = 0 then 4 else n % 4

def group_start (n : ℕ) : ℕ :=
  let cycle_len := (n - 1) / 4
  (cycle_len * 10) + (1 + (cycle_len * 4) + (n % 4))

def group_elements (n : ℕ) : List ℕ :=
  let size := group_size n
  List.map sequence (List.range' (group_start n) size)

-- Proposition to prove the required sum of the 100th group
theorem sum_of_100th_group : (group_elements 100).sum = 1992 :=
by sorry

end sum_of_100th_group_l272_272086


namespace count_positive_integers_satisfying_condition_l272_272576

-- Definitions
def is_between (x: ℕ) : Prop := 30 < x^2 + 8 * x + 16 ∧ x^2 + 8 * x + 16 < 60

-- Theorem statement
theorem count_positive_integers_satisfying_condition :
  {x : ℕ | is_between x}.card = 2 := 
sorry

end count_positive_integers_satisfying_condition_l272_272576


namespace sufficient_but_not_necessary_condition_for_root_l272_272447

theorem sufficient_but_not_necessary_condition_for_root (m : ℝ) :
  (∃ x : ℝ, x ≥ 1 ∧ m + log x = 0) → (m < 0) :=
sorry

end sufficient_but_not_necessary_condition_for_root_l272_272447


namespace sum_of_qi_neg1_l272_272271

-- Definitions based on conditions
def poly := x^6 + x^4 - x^3 - 1
def q1 := x - 1
def q2 := x^2 + 1
def q3 := x^2 + x + 1
def factorized_poly := q1 * q2 * q3

-- Main theorem statement
theorem sum_of_qi_neg1 :
  (q1 * q2 * q3 = poly) → 
  (q1.mononic ∧ q2.mononic ∧ q3.mononic) → 
  (∀ i, coe_fn (coe_fn polynomial.eval) i : ℤ) → 
  (∀ i, irreducible (poly i)) → 
  q1 (-1) + q2 (-1) + q3 (-1) = 1 :=
  by
  sorry

end sum_of_qi_neg1_l272_272271


namespace bouquet_carnations_l272_272461

def proportion_carnations (P : ℚ) (R : ℚ) (PC : ℚ) (RC : ℚ) : ℚ := PC + RC

theorem bouquet_carnations :
  let P := (7 / 10 : ℚ)
  let R := (3 / 10 : ℚ)
  let PC := (1 / 2) * P
  let RC := (2 / 3) * R
  let C := proportion_carnations P R PC RC
  (C * 100) = 55 :=
by
  sorry

end bouquet_carnations_l272_272461


namespace positive_difference_between_median_and_mode_l272_272830

-- Definition of the data as provided in the stem and leaf plot
def data : List ℕ := [
  21, 21, 21, 24, 25, 25,
  33, 33, 36, 37,
  40, 43, 44, 47, 49, 49,
  52, 56, 56, 58, 
  59, 59, 60, 63
]

-- Definition of mode and median calculations
def mode (l : List ℕ) : ℕ := 49  -- As determined, 49 is the mode
def median (l : List ℕ) : ℚ := (43 + 44) / 2  -- Median determined from the sorted list

-- The main theorem to prove
theorem positive_difference_between_median_and_mode (l : List ℕ) :
  abs (median l - mode l) = 5.5 := by
  sorry

end positive_difference_between_median_and_mode_l272_272830


namespace same_order_as_x_higher_order_than_x_lower_order_than_x_l272_272649

open Real

def f (x : ℝ) := 3 * x
def g (x : ℝ) := x^2
def h (x : ℝ) := sqrt x
def i (x : ℝ) := x^3
def j (x : ℝ) := (1/2) * x

theorem same_order_as_x :
  (∀ ε > 0, ∃ δ > 0, ∀ x, (0 < |x| ∧ |x| < δ) → |f x - 3 * x| / |x| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, (0 < |x| ∧ |x| < δ) → |j x - (1/2) * x| / |x| < ε) :=
sorry

theorem higher_order_than_x :
  (∀ ε > 0, ∃ δ > 0, ∀ x, (0 < |x| ∧ |x| < δ) → |g x| / |x| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, (0 < |x| ∧ |x| < δ) → |i x| / |x| < ε) :=
sorry

theorem lower_order_than_x :
  (∀ M > 0, ∃ δ > 0, ∀ x, (0 < |x| ∧ |x| < δ) → |h x| / |x| > M) :=
sorry

end same_order_as_x_higher_order_than_x_lower_order_than_x_l272_272649


namespace angle_BQD_in_triangle_BED_l272_272716

theorem angle_BQD_in_triangle_BED
  (BED : Triangle)
  (angle_EBD_eq_angle_EDB : BED.angle EBD = BED.angle EDB)
  (angle_BED : BED.angle BED = 40)
  (bisects_DQ_angle_EDB : bisects_angle BED EBD DQ) :
  BED.angle BQD = 110 := sorry

end angle_BQD_in_triangle_BED_l272_272716


namespace ratio_of_segments_of_hypotenuse_l272_272173

theorem ratio_of_segments_of_hypotenuse
  (a b c r s : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 2 / 5)
  (h_r : r = (a^2) / c) 
  (h_s : s = (b^2) / c) : 
  r / s = 4 / 25 := sorry

end ratio_of_segments_of_hypotenuse_l272_272173


namespace probability_all_heads_or_tails_l272_272349

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l272_272349


namespace A_plus_B_eq_93_l272_272284

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end A_plus_B_eq_93_l272_272284


namespace f_odd_function_f_increasing_l272_272190

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem f_odd_function : ∀ x : ℝ, x ∈ Ioo (-1 : ℝ) 1 → f (-x) = -f x := 
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 ∈ Ioo (-1 : ℝ) 1 → x2 ∈ Ioo (-1 : ℝ) 1 → x1 < x2 → f x1 < f x2 := 
by
  sorry

end f_odd_function_f_increasing_l272_272190


namespace master_zhang_reading_hours_l272_272745

theorem master_zhang_reading_hours :
  ∃ (hours : set ℕ), hours = {3, 4, 5} ∧
  (∀ h ∈ hours, 1 ≤ h ∧ h ≤ 12) ∧
  (∑ h in hours, h = 12) ∧
  ∀ t : ℝ, t ∈ real.line_segment 0 1 → 
    hour_hand_position t == minute_hand_position t :=
sorry

end master_zhang_reading_hours_l272_272745


namespace radius_of_sphere_tangent_to_edges_l272_272393

noncomputable def radius_of_tangent_sphere 
  (a b : ℝ) : ℝ := 
  (2 * b - a) / 2 * (a * real.sqrt 3 / 2) / real.sqrt (b ^ 2 - 3 * a ^ 2 / 4)

theorem radius_of_sphere_tangent_to_edges 
  (a b : ℝ) (h : b > real.sqrt (3 / 4) * a) : 
  radius_of_tangent_sphere a b = 
    (2 * b - a) / 2 * (a * real.sqrt 3 / 2) / real.sqrt (b ^ 2 - 3 * a ^ 2 / 4) :=
sorry

end radius_of_sphere_tangent_to_edges_l272_272393


namespace joanne_trip_l272_272264

theorem joanne_trip (a b c x : ℕ) (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 100 * c + 10 * a + b - (100 * a + 10 * b + c) = 60 * x) : 
  a^2 + b^2 + c^2 = 51 :=
by
  sorry

end joanne_trip_l272_272264


namespace avg_speed_eq_inst_speed_l272_272023

noncomputable def position (v b : ℝ) : ℝ → ℝ := 
  λ t, v * t + b

theorem avg_speed_eq_inst_speed (v b : ℝ) : 
  ∀ t1 t2 : ℝ, t1 ≠ t2 → 
    (position v b t1 - position v b t2) / (t1 - t2) = v :=
by sorry

end avg_speed_eq_inst_speed_l272_272023


namespace players_exceed_30000_l272_272027

theorem players_exceed_30000
  (R : ℕ → ℝ)
  (R0 : ℝ)
  (k : ℝ)
  (R_formula : ∀ t, R(t) = R0 * Real.exp(k * t))
  (R_at_0 : R(0) = 100)
  (R_at_5 : R(5) = 1000)
  (approx_lg3 : Real.log 3 ≈ 0.4771) :
  ∃ t : ℕ, t ≥ 13 ∧ R(t) > 30000 :=
by
  have R0_eq_100 : R0 = 100 := by 
    sorry
  have k_eq : k = Real.log 10 / 5 := by
    sorry
  existsi 13
  split
  · exact Nat.le_refl 13
  · rw [R_formula, R0_eq_100, k_eq]
    have exp_inequality : 10 ^ (13 / 5) > 300 := by
      sorry
    exact exp_inequality

end players_exceed_30000_l272_272027


namespace six_coins_all_heads_or_tails_probability_l272_272354

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l272_272354


namespace bulb_pos_97_to_100_l272_272807

/-- Define the sequence of bulbs meeting the given conditions --/
def bulb (n : Nat) : Prop :=
  if n % 5 == 1 ∨ n % 5 == 0 then "Y" else "B"

/-- The problem to prove the color and order of bulbs at positions 97, 98, 99, and 100 --/
theorem bulb_pos_97_to_100 :
  bulb 97 = "Y" ∧
  bulb 98 = "B" ∧
  bulb 99 = "B" ∧
  bulb 100 = "Y" := by
  sorry

end bulb_pos_97_to_100_l272_272807


namespace total_time_to_travel_600_meters_l272_272882

theorem total_time_to_travel_600_meters :
  let d := 0.6 / 3 in
  let t1 := d / 2 in
  let t2 := d / 4 in
  let t3 := d / 6 in
  (t1 * 60) + (t2 * 60) + (t3 * 60) = 11 :=
by
  sorry

end total_time_to_travel_600_meters_l272_272882


namespace characteristics_of_function_l272_272991

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem characteristics_of_function :
  (∀ x : ℝ, x ≠ 0 → deriv f x = -1 / x^2) ∧
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x < y → f x > f y) ∧
  (∀ L : ℝ, ∃ x : ℝ, x > 0 ∧ f x > L) ∧
  (∀ L : ℝ, ∃ x : ℝ, x < 0 ∧ f x < -L) :=
by
  sorry

end characteristics_of_function_l272_272991


namespace angle_between_clock_hands_at_7_30_l272_272420

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l272_272420


namespace max_singular_words_l272_272232

theorem max_singular_words (alphabet_length : ℕ) (word_length : ℕ) (strip_length : ℕ) 
  (num_non_overlapping_pieces : ℕ) (h_alphabet : alphabet_length = 25)
  (h_word_length : word_length = 17) (h_strip_length : strip_length = 5^18)
  (h_non_overlapping : num_non_overlapping_pieces = 5^16) : 
  ∃ max_singular_words, max_singular_words = 2 * 5^17 :=
by {
  -- proof to be completed
  sorry
}

end max_singular_words_l272_272232


namespace area_of_square_inscribed_in_ellipse_l272_272493

noncomputable def areaSquareInscribedInEllipse : ℝ :=
  let a := 4
  let b := 8
  let s := (2 * Real.sqrt (b / 3)).toReal in
  (2 * s) ^ 2

theorem area_of_square_inscribed_in_ellipse :
  (areaSquareInscribedInEllipse) = 32 / 3 :=
sorry

end area_of_square_inscribed_in_ellipse_l272_272493


namespace find_AD_l272_272709

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272709


namespace cookies_per_child_l272_272542

def num_adults : ℕ := 4
def num_children : ℕ := 6
def cookies_jar1 : ℕ := 240
def cookies_jar2 : ℕ := 360
def cookies_jar3 : ℕ := 480

def fraction_eaten_jar1 : ℚ := 1 / 4
def fraction_eaten_jar2 : ℚ := 1 / 3
def fraction_eaten_jar3 : ℚ := 1 / 5

theorem cookies_per_child :
  let eaten_jar1 := fraction_eaten_jar1 * cookies_jar1
  let eaten_jar2 := fraction_eaten_jar2 * cookies_jar2
  let eaten_jar3 := fraction_eaten_jar3 * cookies_jar3
  let remaining_jar1 := cookies_jar1 - eaten_jar1
  let remaining_jar2 := cookies_jar2 - eaten_jar2
  let remaining_jar3 := cookies_jar3 - eaten_jar3
  let total_remaining_cookies := remaining_jar1 + remaining_jar2 + remaining_jar3
  let cookies_each_child := total_remaining_cookies / num_children
  cookies_each_child = 134 := by
  sorry

end cookies_per_child_l272_272542


namespace rectangle_area_ratio_l272_272445

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 :=
by
  sorry

end rectangle_area_ratio_l272_272445


namespace distinct_cube_colorings_l272_272642

-- Defining what it means to color a cube with three red faces and three blue faces, 
-- considering rotational symmetries.

def number_of_distinct_colorings (nRed nBlue : ℕ) : ℕ :=
  if nRed = 3 ∧ nBlue = 3 then 2 else sorry -- The criterion we are given.

-- The main statement
theorem distinct_cube_colorings :
  number_of_distinct_colorings 3 3 = 2 :=
by {
  unfold number_of_distinct_colorings,
  simp,
  done
}

end distinct_cube_colorings_l272_272642


namespace number_of_intersections_l272_272999

noncomputable def f (x : ℝ) (t : ℝ) : ℝ := t * x^2 - x + 1
noncomputable def g (x : ℝ) (t : ℝ) : ℝ := 2 * t * x - 1

theorem number_of_intersections (t : ℝ) :
    let f (x : ℝ) := t * x^2 - x + 1
    let g (x : ℝ) := 2 * t * x - 1
    ∃ n : ℕ, (n = 1 ∨ n = 2) ∧ ∃ x : set ℝ, (∀ y ∈ x, f y = g y) ∧ set.card x = n := by
  sorry

end number_of_intersections_l272_272999


namespace f_2017_eq_l272_272596

def f (x : ℝ) : ℝ := (1 + x) / (1 - 3 * x)

noncomputable def iter_f (n : ℕ) : ℝ → ℝ
| 0     := id
| (n+1) := f ∘ iter_f n

theorem f_2017_eq: iter_f 2017 (-2) = 3 / 5 := by
  sorry

end f_2017_eq_l272_272596


namespace binary_sum_correct_l272_272528

-- Definitions for the binary numbers involved
def binary_1010 : nat := 2^3 + 2^1
def binary_111 : nat := 2^2 + 2^1 + 2^0
def binary_1001 : nat := 2^3 + 2^0
def binary_1011 : nat := 2^3 + 2^1 + 2^0

-- Correct answer in decimal form
def correct_answer : nat := 2^4 + 2^2 + 2^1 + 2^0 -- 10111_2

-- Theorem that proves the binary sum and subtraction equals correct answer
theorem binary_sum_correct :
  (binary_1010 + binary_111 - binary_1001 + binary_1011) = correct_answer :=
by 
  -- We skip the proof here, just stating the theorem
  sorry

end binary_sum_correct_l272_272528


namespace nutritional_content_l272_272405

-- Define the protein and iron content of type A and type B foods
variables (x y : ℝ)

-- Condition definitions
def iron_content_A := 2 * x
def iron_content_B := (4 / 7) * y

-- Define the system of equations for the patient's needs
noncomputable def equation_1 := 28 * x + 30 * y = 35
noncomputable def equation_2 := 56 * x + 120 / 7 * y = 40

-- Proof statement
theorem nutritional_content (h1 : equation_1) (h2 : equation_2) : x = 0.5 ∧ iron_content_A = 1 :=
by 
  sorry

end nutritional_content_l272_272405


namespace exists_rectangle_with_perimeter_divisible_by_4_l272_272499

-- Define the problem conditions in Lean
def square_length : ℕ := 2015

-- Define what it means to cut the square into rectangles with integer sides
def is_rectangle (a b : ℕ) := 1 ≤ a ∧ a ≤ square_length ∧ 1 ≤ b ∧ b ≤ square_length

-- Define the perimeter condition
def perimeter_divisible_by_4 (a b : ℕ) := (2 * a + 2 * b) % 4 = 0

-- Final theorem statement
theorem exists_rectangle_with_perimeter_divisible_by_4 :
  ∃ (a b : ℕ), is_rectangle a b ∧ perimeter_divisible_by_4 a b :=
by {
  sorry -- The proof itself will be filled in to establish the theorem
}

end exists_rectangle_with_perimeter_divisible_by_4_l272_272499


namespace area_triangle_intersection_point_l272_272251

-- Define the Cartesian form of the curve for m=1 and n=1
def curve1 (x y : ℝ) : Prop := (x^2/4) + y^2 = 1

-- Define the Cartesian form of the curve for m=1 and n=2
def curve2 (x y : ℝ) : Prop := x^2 + (y/2)^2 = 1

-- Define the parametric form of the line l
def line_l (t : ℝ) (x y : ℝ) : Prop := x = t - real.sqrt 3 ∧ y = real.sqrt 3 * t + 1

-- First assertion: area of triangle AOB
theorem area_triangle (m n : ℝ) (h1 : m = 1) (h2 : n = 1):
  ∃ (OA OB : ℝ),
  -- Specified points where curve intersects given rays
  curve1 (2 * m * real.cos (real.pi / 4)) (n * real.sin (real.pi / 4)) ∧ 
  curve1 (2 * m * real.cos (-real.pi / 4)) (n * real.sin (-real.pi / 4)) ∧
  -- The area of the triangle formed by these points and the origin
  let OA := real.sqrt (8 / 5) in
  let OB := real.sqrt (8 / 5) in
  1 /2 * OA * OB = 4 / 5 := sorry

-- Second assertion: coordinates of intersection of curve and line l
theorem intersection_point (m n : ℝ) (h1 : m = 1) (h2 : n = 2):
  ∃ (x y : ℝ),
  curve2 x y ∧
  line_l 0 x y ∧
  x = -real.sqrt 3 ∧ y = 1 := sorry

end area_triangle_intersection_point_l272_272251


namespace minimum_area_of_triangle_OAB_BC_parallel_y_axis_l272_272627

open Real

noncomputable def parabola_equation : ℝ → ℝ :=
λ x, x^2 / 4

noncomputable def focus : ℝ × ℝ := (0, 1)

noncomputable def line_through_focus (k : ℝ) : ℝ → ℝ :=
λ x, k * x + 1

noncomputable def area_OAB_minimum : ℝ :=
2

theorem minimum_area_of_triangle_OAB :
  ∃ (A B : ℝ × ℝ) (k : ℝ), ∀ (O : ℝ × ℝ), O = (0, 0) →
  (A.2 = parabola_equation A.1) ∧ (B.2 = parabola_equation B.1) ∧
  (A.2 = line_through_focus k A.1) ∧ (B.2 = line_through_focus k B.1) ∧
  let S := abs (4 * (1 + k^2)) * (1 / sqrt (1 + k^2)) in S = 2 :=
sorry

noncomputable def intersection_AO_directrix (x1 : ℝ) : ℝ × ℝ :=
(-4 / x1, -1)

theorem BC_parallel_y_axis :
  ∀ (A B C : ℝ × ℝ) (x1 : ℝ),
  (A.2 = parabola_equation A.1) ∧ (B.2 = parabola_equation B.1) ∧
  (A.1 = x1) ∧ (B.1 = -4 / x1) ∧ (C = intersection_AO_directrix x1) →
  B.1 = C.1 :=
sorry

end minimum_area_of_triangle_OAB_BC_parallel_y_axis_l272_272627


namespace james_final_weight_l272_272720

noncomputable def initial_weight : ℝ := 120
noncomputable def muscle_gain : ℝ := 0.20 * initial_weight
noncomputable def fat_gain : ℝ := muscle_gain / 4
noncomputable def final_weight (initial_weight muscle_gain fat_gain : ℝ) : ℝ :=
  initial_weight + muscle_gain + fat_gain

theorem james_final_weight :
  final_weight initial_weight muscle_gain fat_gain = 150 :=
by
  sorry

end james_final_weight_l272_272720


namespace calculate_unoccupied_volume_l272_272478

def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def water_volume : ℕ := tank_volume / 3
def ice_cube_volume : ℕ := 1
def ice_cubes_count : ℕ := 12
def total_ice_volume : ℕ := ice_cubes_count * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume

def unoccupied_volume : ℕ := tank_volume - occupied_volume

theorem calculate_unoccupied_volume : unoccupied_volume = 628 := by
  sorry

end calculate_unoccupied_volume_l272_272478


namespace exists_valid_board_l272_272005

-- Definitions representing the conditions given in the problem
def board : Type := matrix (fin 3) (fin 3) ℕ

def valid_numbers : set ℕ := {1, 2, 3, 4, 6, 8, 9}

def valid_board (b : board) : Prop :=
  (∀ i j, b i j ∈ valid_numbers) ∧
  (list.prod [b 0 0, b 0 1, b 0 2] = list.prod [b 1 0, b 1 1, b 1 2]) ∧
  (list.prod [b 0 0, b 1 0, b 2 0] = list.prod [b 0 0, b 0 1, b 0 2])

-- Theorem stating that such a valid board exists
theorem exists_valid_board : ∃ (b : board), valid_board b := by
  sorry

end exists_valid_board_l272_272005


namespace sixSquaresPackable_l272_272753

noncomputable def squaresPackable (a : Fin 6 → ℝ) : Prop :=
    a 0 ^ 2 + a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 2 ∧
    (∀ i, 0 < a i) ∧
    (∀ i j, i ≠ j → disjoint (square a i) (square a j)) ∧ -- Ensures non-overlapping
    (∀ i, inSquare 2 (square a i)) -- Fits inside the 2×2 square

theorem sixSquaresPackable (a : Fin 6 → ℝ)
  (h1 : a 0 ^ 2 + a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 2)
  (h2 : ∀ i, 0 < a i) :
  squaresPackable a :=
by
  sorry

end sixSquaresPackable_l272_272753


namespace email_count_first_day_l272_272464

theorem email_count_first_day (E : ℕ) 
  (h1 : ∃ E, E + E / 2 + E / 4 + E / 8 = 30) : E = 16 :=
by
  sorry

end email_count_first_day_l272_272464


namespace correct_operation_l272_272840

theorem correct_operation :
  (∀ a b : ℝ, (-2 * a * b^2)^3 = -8 * a^3 * b^6) ∧ 
  ¬ (sqrt 3 + sqrt 2 = sqrt 5) ∧ 
  (∀ a b : ℝ, (a + b)^2 ≠ a^2 + b^2) ∧ 
  (- (1 / 2) ^ (-2) ≠ - (1 / 4)) :=
by
  sorry

end correct_operation_l272_272840


namespace probability_all_heads_or_tails_l272_272342

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l272_272342


namespace negation_of_universal_statement_l272_272383

def P (x : ℝ) : Prop := x^3 - x^2 + 1 ≤ 0

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by {
  sorry
}

end negation_of_universal_statement_l272_272383


namespace largest_positive_integer_n_l272_272945

 

theorem largest_positive_integer_n (n : ℕ) :
  (∀ p : ℕ, Nat.Prime p ∧ 2 < p ∧ p < n → Nat.Prime (n - p)) →
  ∀ m : ℕ, (∀ q : ℕ, Nat.Prime q ∧ 2 < q ∧ q < m → Nat.Prime (m - q)) → n ≥ m → n = 10 :=
by
  sorry

end largest_positive_integer_n_l272_272945


namespace minimum_value_of_f_l272_272975

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ x ∈ (set.Icc (-2 : ℝ) (1 / 2)), f x = 5 / 2 :=
by {
  use 1 / 2,
  split,
  linarith,
  linarith,
  have h : |2 * (1 / 2) - 1| = 0, by norm_num,
  have h' : |(1 / 2) + 2| = 5 / 2, by norm_num,
  simp [f, h, h']
}

end minimum_value_of_f_l272_272975


namespace guitar_center_discount_is_correct_l272_272762

-- Define the suggested retail price
def retail_price : ℕ := 1000

-- Define the shipping fee of Guitar Center
def shipping_fee : ℕ := 100

-- Define the discount percentage offered by Sweetwater
def sweetwater_discount_rate : ℕ := 10

-- Define the amount saved by buying from the cheaper store
def savings : ℕ := 50

-- Define the discount offered by Guitar Center
def guitar_center_discount : ℕ :=
  retail_price - ((retail_price * (100 - sweetwater_discount_rate) / 100) + savings - shipping_fee)

-- Theorem: Prove that the discount offered by Guitar Center is $150
theorem guitar_center_discount_is_correct : guitar_center_discount = 150 :=
  by
    -- The proof will be filled in based on the given conditions
    sorry

end guitar_center_discount_is_correct_l272_272762


namespace median_of_consecutive_integers_l272_272396

theorem median_of_consecutive_integers (n : ℕ) (h1 : n = 49) (sum : ℕ) (h2 : sum = 7^5) : 
  let median := sum / n in median = 343 := by
  -- By definitions and conditions
  sorry

end median_of_consecutive_integers_l272_272396


namespace max_sundays_in_45_days_l272_272825

theorem max_sundays_in_45_days (days_in_year_start: ∀ year: Nat, [1 ≤ year ∧ year ≤ 365], total_days: Nat) :
  total_days = 45 ->
  (∃sundays : Nat, (∀days_of_week: Nat, days_of_week = 7) ∧ (days_in_year_start 1) -> (sundays ≤ total_days) ∧ (sundays ≤ 7)) :=
by
  sorry

end max_sundays_in_45_days_l272_272825


namespace quadratic_inequality_solution_l272_272763

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end quadratic_inequality_solution_l272_272763


namespace randy_feeds_per_day_l272_272330

theorem randy_feeds_per_day
  (pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ)
  (h1 : pigs = 2) (h2 : total_feed_per_week = 140) (h3 : days_per_week = 7) :
  total_feed_per_week / pigs / days_per_week = 10 :=
by
  sorry

end randy_feeds_per_day_l272_272330


namespace right_triangle_congruence_l272_272435

theorem right_triangle_congruence : 
  ∀ (Δ1 Δ2 : Triangle) 
    (rightΔ1 : Δ1.angle_sum = 180 ∧ Δ1.angles.0 = 90)
    (rightΔ2 : Δ2.angle_sum = 180 ∧ Δ2.angles.0 = 90)
    (acute1 : Δ1.angles.1 = Δ2.angles.1)
    (acute2 : Δ1.angles.2 = Δ2.angles.2), 
  ¬ (Δ1 ≅ Δ2) :=
by sorry

end right_triangle_congruence_l272_272435


namespace prove_sandwiches_ratio_l272_272905

def sandwiches_problem 
  (billy_sandwiches : ℕ) 
  (katelyn_more : ℕ) 
  (total_sandwiches : ℕ) 
  (chloe_ratio : ℚ) : Prop :=
  let katelyn_sandwiches := billy_sandwiches + katelyn_more
  let chloe_sandwiches := total_sandwiches - billy_sandwiches - katelyn_sandwiches
  ∀ (billy_sandwiches = 49) (katelyn_more = 47) (total_sandwiches = 169),
  chloe_ratio = (chloe_sandwiches : ℚ) / (katelyn_sandwiches : ℚ) → chloe_ratio = 1 / 4

theorem prove_sandwiches_ratio : sandwiches_problem 49 47 169 (1 / 4) :=
sorry

end prove_sandwiches_ratio_l272_272905


namespace hexagon_angle_E_l272_272239

theorem hexagon_angle_E (A N G L E S : ℝ) 
  (h1 : A = G) 
  (h2 : G = E) 
  (h3 : N + S = 180) 
  (h4 : L = 90) 
  (h_sum : A + N + G + L + E + S = 720) : 
  E = 150 := 
by 
  sorry

end hexagon_angle_E_l272_272239


namespace sin_675_eq_neg_sqrt2_div_2_l272_272924

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end sin_675_eq_neg_sqrt2_div_2_l272_272924


namespace least_number_of_tiles_l272_272869

-- Definitions for classroom dimensions
def classroom_length : ℕ := 624 -- in cm
def classroom_width : ℕ := 432 -- in cm

-- Definitions for tile dimensions
def rectangular_tile_length : ℕ := 60
def rectangular_tile_width : ℕ := 80
def triangular_tile_base : ℕ := 40
def triangular_tile_height : ℕ := 40

-- Definition for the area calculation
def area (length width : ℕ) : ℕ := length * width
def area_triangular_tile (base height : ℕ) : ℕ := (base * height) / 2

-- Define the area of the classroom and tiles
def classroom_area : ℕ := area classroom_length classroom_width
def rectangular_tile_area : ℕ := area rectangular_tile_length rectangular_tile_width
def triangular_tile_area : ℕ := area_triangular_tile triangular_tile_base triangular_tile_height

-- Define the number of tiles required
def number_of_rectangular_tiles : ℕ := (classroom_area + rectangular_tile_area - 1) / rectangular_tile_area -- ceiling division in lean
def number_of_triangular_tiles : ℕ := (classroom_area + triangular_tile_area - 1) / triangular_tile_area -- ceiling division in lean

-- Define the minimum number of tiles required
def minimum_number_of_tiles : ℕ := min number_of_rectangular_tiles number_of_triangular_tiles

-- The main theorem establishing the least number of tiles required
theorem least_number_of_tiles : minimum_number_of_tiles = 57 := by
    sorry

end least_number_of_tiles_l272_272869


namespace domain_of_function_is_correct_l272_272380

def function_domain (x : ℝ) : Set ℝ :=
  {x | x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0}

theorem domain_of_function_is_correct :
  function_domain = { x | (0 < x ∧ x < 1/2) ∨ (x > 2) } :=
by 
  sorry

end domain_of_function_is_correct_l272_272380


namespace exterior_angle_DEG_l272_272333

noncomputable def interior_angle_regular_polygon (n : ℕ) : ℝ :=
  180 * (n - 2) / n

def exterior_angle_deg (hept_int_angle oct_int_angle : ℝ) : ℝ :=
  360 - hept_int_angle - oct_int_angle

theorem exterior_angle_DEG :
  let hept_int_angle := interior_angle_regular_polygon 7 in
  let oct_int_angle := interior_angle_regular_polygon 8 in
  exterior_angle_deg hept_int_angle oct_int_angle = 96.43 :=
by
  -- Assuming the computed decimal values for the specific problem
  let hept_int_angle := 180 * (7 - 2) / 7
  let oct_int_angle := 180 * (8 - 2) / 8
  have h_hept : hept_int_angle = 900 / 7 := by sorry
  have h_oct : oct_int_angle = 135 := by sorry
  unfold exterior_angle_deg
  rw [h_hept, h_oct]
  sorry

end exterior_angle_DEG_l272_272333


namespace vertex_on_line_l272_272628

theorem vertex_on_line (n : ℝ) : 
  let y := λ x : ℝ, x^2 + (2*n + 1)*x + n^2 - 1 in
  ∃ (x_v y_v : ℝ), y_v = (x_v - 3/4) ∧ y_v = (x_v + (2*n + 1)/2)^2 - (4*n + 5)/4  :=
sorry

end vertex_on_line_l272_272628


namespace eccentricity_of_ellipse_equation_of_ellipse_equation_of_line_l272_272294

variables {a b c : ℝ} (x y : ℝ) (A B F1 F2 O P M N : ℝ × ℝ)

-- Conditions
def ellipse (x y : ℝ) (a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def AB_condition (A B F1 F2 : ℝ × ℝ) := abs (A.1 - B.1)^2 + (A.2 - B.2)^2 = (sqrt 2 / 2) * (2 * F1.1 * F2.1)^2
def distance_origin_to_AB (O A B : ℝ × ℝ) := abs (A.2 * O.1 - B.1 * O.2) / sqrt (A.2^2 + B.1^2) = 3 * sqrt 3 / 2
def midpoint_condition (P M N : ℝ × ℝ) := P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Question (1)
theorem eccentricity_of_ellipse :
  ellipse x y a b →
  AB_condition A B F1 F2 →
  a > 0 → b > 0 → b < a → 
  sqrt (a^2 - b^2) / a = sqrt 6 / 3 := sorry

-- Question (2)
theorem equation_of_ellipse :
  ellipse x y a b →
  AB_condition A B F1 F2 →
  distance_origin_to_AB O A B →
  a = 3 * sqrt 3 →
  b = 3 →
  ellipse x y a b = (x^2 / 27 + y^2 / 9 = 1) := sorry

-- Question (3)
theorem equation_of_line :
  ellipse x y a b →
  AB_condition A B F1 F2 →
  distance_origin_to_AB O A B →
  midpoint_condition P M N →
  P = (-2, 1) →
  x ≠ y →
  equation_of_ellipse x y a b →
  (M.1^2 / 27 + M.2^2 / 9 = 1 ∧ N.1^2 / 27 + N.2^2 / 9 = 1) →
  2 * x - 3 * y + 7 = 0 := sorry

end eccentricity_of_ellipse_equation_of_ellipse_equation_of_line_l272_272294


namespace arcsin_arccos_interval_l272_272552

open Real
open Set

theorem arcsin_arccos_interval (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ t ∈ Icc (-3 * π / 2) (π / 2), 2 * arcsin x - arccos y = t := 
sorry

end arcsin_arccos_interval_l272_272552


namespace solve_trig_eq_l272_272438

open Real Int

-- Definition of the problem condition
def condition (x : ℝ) : Prop :=
  sin x ^ 2 + 2 * sin x * cos x = 3 * cos x ^ 2

-- The problem statement to be proven
theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  cos x ≠ 0 → (condition x) ↔ (x = (π / 4) + π * k ∨ x = -arctan 3 + π * k) :=
by
  intros h_cos_ne_zero
  have : cos x ≠ 0 := h_cos_ne_zero
  sorry

end solve_trig_eq_l272_272438


namespace germs_per_dish_l272_272252

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end germs_per_dish_l272_272252


namespace largest_n_with_integer_solutions_l272_272111

theorem largest_n_with_integer_solutions : ∃ n, ∀ x y1 y2 y3 y4, 
 ( ((x + 1)^2 + y1^2) = ((x + 2)^2 + y2^2) ∧  ((x + 2)^2 + y2^2) = ((x + 3)^2 + y3^2) ∧ 
  ((x + 3)^2 + y3^2) = ((x + 4)^2 + y4^2)) → (n = 3) := sorry

end largest_n_with_integer_solutions_l272_272111


namespace ticket_price_increase_l272_272546

-- Define the initial price and the new price
def last_year_price : ℝ := 85
def this_year_price : ℝ := 102

-- Define the percent increase calculation
def percent_increase (initial : ℝ) (new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

-- Statement to prove
theorem ticket_price_increase (initial : ℝ) (new : ℝ) (h_initial : initial = last_year_price) (h_new : new = this_year_price) :
  percent_increase initial new = 20 :=
by
  sorry

end ticket_price_increase_l272_272546


namespace probability_of_selecting_double_l272_272890

/-- Define the set of all possible pairs of integers from 0 to 12 -/
def all_pairs : Finset (ℕ × ℕ) :=
  (Finset.range 13).product (Finset.range 13)

/-- Define what a double is, i.e., a pair where both elements are equal -/
def is_double (p : ℕ × ℕ) : Prop := p.1 = p.2

/-- Compute the total number of doubles in the set of pairs -/
def count_doubles : ℕ :=
  (all_pairs.filter is_double).card

/-- The total number of dominoes in the set -/
def total_dominoes : ℕ :=
  all_pairs.card / 2

/-- The probability of selecting a double from the set of dominoes -/
def double_probability : ℚ :=
  count_doubles / total_dominoes

/-- The final probability calculation -/
theorem probability_of_selecting_double :
  double_probability = 13 / 91 :=
by
  sorry

end probability_of_selecting_double_l272_272890


namespace total_number_of_bills_l272_272817

theorem total_number_of_bills (total_money : ℕ) (fraction_for_50_bills : ℚ) (fifty_bill_value : ℕ) (hundred_bill_value : ℕ) :
  total_money = 1000 →
  fraction_for_50_bills = 3 / 10 →
  fifty_bill_value = 50 →
  hundred_bill_value = 100 →
  let money_for_50_bills := total_money * fraction_for_50_bills in
  let num_50_bills := money_for_50_bills / fifty_bill_value in
  let rest_money := total_money - money_for_50_bills in
  let num_100_bills := rest_money / hundred_bill_value in
  num_50_bills + num_100_bills = 13 :=
by
  intros h1 h2 h3 h4
  let money_for_50_bills := 1000 * (3 / 10)
  have h5 : money_for_50_bills = 300 := by sorry
  have h6 : 300 / 50 = 6 := by sorry
  let rest_money := 1000 - 300
  have h7 : rest_money = 700 := by sorry
  have h8 : 700 / 100 = 7 := by sorry
  have total_bills := 6 + 7
  show total_bills = 13 from eq.refl 13

end total_number_of_bills_l272_272817


namespace find_value_of_k_find_max_profit_l272_272224

noncomputable def daily_cost (x : ℝ) : ℝ := x + 5

noncomputable def daily_sales_revenue (x : ℝ) (k : ℝ) : ℝ :=
  if (0 < x ∧ x < 6) then 3*x + k / (x - 8) + 7
  else 16

noncomputable def daily_profit (x : ℝ) (k : ℝ) : ℝ :=
  daily_sales_revenue x k - daily_cost x

theorem find_value_of_k : daily_profit 2 k = 3 → k = 18 :=
by
  intro h
  sorry

theorem find_max_profit :
  (∀ x k, 0 < x ∧ x < 6 → L = 2*x + k/(x-8) + 2) → 
  (∀ x k, x ≥ 6 → L = 11 - x) → 
  (∃ x, daily_profit x 18 = 6 ∧ x = 5) :=
by
  intros h1 h2
  sorry

end find_value_of_k_find_max_profit_l272_272224


namespace positive_integer_pairs_l272_272095

theorem positive_integer_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : (n + (n + 1) + (n + 2) + ... + (n + m)) = 1000) :
  (m = 15 ∧ n = 55) ∨ (m = 24 ∧ n = 28) ∨ (m = 4 ∧ n = 198) :=
sorry

end positive_integer_pairs_l272_272095


namespace max_min_diff_l272_272780

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 + 2

theorem max_min_diff (M m : ℝ) 
  (hM : ∀ x ∈ Icc (-1 : ℝ) 1, f x ≤ M)
  (hM_max : ∃ x ∈ Icc (-1 : ℝ) 1, f x = M)
  (hm : ∀ x ∈ Icc (-1 : ℝ) 1, f x ≥ m)
  (hm_min : ∃ x ∈ Icc (-1 : ℝ) 1, f x = m) :
  M - m = 4 :=
by
  sorry

end max_min_diff_l272_272780


namespace no_fraternity_member_is_club_member_l272_272512

variable {U : Type} -- Domain of discourse, e.g., the set of all people at the school
variables (Club Member Student Honest Fraternity : U → Prop)

theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, Club x → Student x)
  (h2 : ∀ x, Club x → ¬ Honest x)
  (h3 : ∀ x, Fraternity x → Honest x) :
  ∀ x, Fraternity x → ¬ Club x := 
sorry

end no_fraternity_member_is_club_member_l272_272512


namespace minimum_length_of_segment_PQ_l272_272615

theorem minimum_length_of_segment_PQ:
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y + 1 = 0) → 
              (xy >= 2) → 
              (x - y >= 0) → 
              (y <= 1) → 
              ℝ) :=
sorry

end minimum_length_of_segment_PQ_l272_272615


namespace rita_money_left_l272_272335

theorem rita_money_left :
  let initial_amount : ℝ := 400
  let cost_short_dresses : ℝ := 5 * (20 - 0.1 * 20)
  let cost_pants : ℝ := 2 * 15
  let cost_jackets : ℝ := 2 * (30 - 0.15 * 30) + 2 * 30
  let cost_skirts : ℝ := 2 * 18 * 0.8
  let cost_tshirts : ℝ := 2 * 8
  let cost_transportation : ℝ := 5
  let total_spent : ℝ := cost_short_dresses + cost_pants + cost_jackets + cost_skirts + cost_tshirts + cost_transportation
  let money_left : ℝ := initial_amount - total_spent
  money_left = 119.2 :=
by 
  sorry

end rita_money_left_l272_272335


namespace solve_x_correct_l272_272948

open Complex Real

noncomputable def solve_x : ℝ :=
  let x := sqrt (1822 / 29) in x

theorem solve_x_correct :
  |(solve_x : ℂ) + (⟨0, sqrt 7⟩ : ℂ)| * |3 - (⟨0, 2 * sqrt 5⟩ : ℂ)| = 45 ∧
  solve_x > 0 ∧
  solve_x ≈ 7.93 :=
by
  let x := solve_x
  have hx1 : |(x : ℂ) + (⟨0, sqrt 7⟩)| * |3 - (⟨0, 2 * sqrt 5⟩)| = 45 := sorry
  have hx2 : x > 0 := sorry
  have hx3 : x ≈ 7.93 := sorry
  exact ⟨hx1, hx2, hx3⟩

end solve_x_correct_l272_272948


namespace mode_and_median_of_set_l272_272234

theorem mode_and_median_of_set :
  let S := [3, 5, 1, 4, 6, 5] in
  (mode S = 5 ∧ median S = 4.5) :=
by
  sorry

end mode_and_median_of_set_l272_272234


namespace max_excellent_lessons_l272_272670

structure VideoLesson :=
  (view_count : ℕ)
  (expert_rating : ℕ)

def not_inferior (A B : VideoLesson) : Prop :=
  A.view_count >= B.view_count ∨ A.expert_rating >= B.expert_rating

def excellent (A : VideoLesson) (others : List VideoLesson) : Prop :=
  ∀ B ∈ others, not_inferior A B

theorem max_excellent_lessons : ∃ (lessons : List VideoLesson), 
  lessons.length = 5 ∧ 
  (∃ (excellent_lessons : List VideoLesson), 
  excellent_lessons.length = 5 ∧ 
  ∀ A ∈ excellent_lessons, excellent A (lessons.erase A)) :=
sorry

end max_excellent_lessons_l272_272670


namespace jack_cycling_ratio_l272_272263

variable (distance_home_store: ℝ) (speed: ℝ)
variable (distance_store_peter: ℝ := 50) (total_distance: ℝ := 250)

-- Total distance condition
def total_distance_condition := distance_home_store + 2 * distance_store_peter + distance_store_peter = total_distance

-- Time calculation
def time_home_store := distance_home_store / speed
def time_store_peter := distance_store_peter / speed

-- Ratio condition
def ratio_condition := time_home_store / time_store_peter = 2

theorem jack_cycling_ratio (h_total : total_distance_condition) : ratio_condition :=
sorry

end jack_cycling_ratio_l272_272263


namespace calc_value_l272_272832

theorem calc_value :
  (2525 - 2424)^2 + 100 = 10301 → 10301 / 225 = 46 → (2525 - 2424)^2 + 100 / 225 = 46 :=
by
  intros h1 h2
  simp
  exact h1.trans h2

end calc_value_l272_272832


namespace trapezoid_AD_equal_l272_272696

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272696


namespace smallest_pos_int_terminating_decimal_with_9_l272_272429

theorem smallest_pos_int_terminating_decimal_with_9 : ∃ n : ℕ, (∃ m k : ℕ, n = 2^m * 5^k ∧ (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9)) ∧ n = 4096 :=
by {
    sorry
}

end smallest_pos_int_terminating_decimal_with_9_l272_272429


namespace find_AD_l272_272704

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272704


namespace initial_milk_quarts_l272_272024

theorem initial_milk_quarts (M : ℝ)
    (h1 : 0.04 * M = 0.03 * (M - 50) + 0.23 * 50) :
    M = 1000 := 
begin
  sorry
end

end initial_milk_quarts_l272_272024


namespace f_f_minus_one_range_of_a_l272_272188

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then (1/2)^x else 1 - 3*x

theorem f_f_minus_one :
  f (f (-1)) = -5 :=
sorry

theorem range_of_a (a : ℝ) :
  f (2*a^2 - 3) > f (5*a) → -1/2 < a ∧ a < 3 :=
sorry

end f_f_minus_one_range_of_a_l272_272188


namespace angle_between_m_n_is_135_l272_272199

open Real

-- Define the vectors
def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (9, 12)
def vec_c : ℝ × ℝ := (4, -3)
def vec_m : ℝ × ℝ := (2 * 3 - 9, 2 * 4 - 12)  -- equivalent to 2 * vec_a - vec_b
def vec_n : ℝ × ℝ := (3 + 4, 4 - 3)           -- equivalent to vec_a + vec_c

-- Dot product of two vectors
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Norm (magnitude) of a vector
def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Cosine of the angle between two vectors
def cos_angle (v1 v2 : ℝ × ℝ) : ℝ := dot_prod v1 v2 / (norm v1 * norm v2)

-- Angle between two vectors in degrees
def angle_degrees (v1 v2 : ℝ × ℝ) : ℝ := arccos (cos_angle v1 v2) * (180 / π)

-- Theorem declaration to prove the angle between vec_m and vec_n is 135 degrees
theorem angle_between_m_n_is_135 :
  angle_degrees vec_m vec_n = 135 := by
  sorry

end angle_between_m_n_is_135_l272_272199


namespace B_alone_completion_time_l272_272846

variables (A B : Type) [workman A] [workman B]

-- Define the function work_done which takes the number of days and the
-- amount of work done in a day and produces the total amount of work done
def work_done (days : ℕ) (work_per_day : ℕ) : ℕ := days * work_per_day

-- A is twice as good a workman as B
axiom A_is_twice_b : ∀ (Wb : ℕ), work_done 1 (work_per_day B) = 2 * work_done 1 (work_per_day A)

-- A and B together took 10 days to complete the work
axiom A_and_B_together : ∀ (Wb : ℕ), work_done 10 (work_per_day A + work_per_day B) = 30 * work_done 1 (work_per_day B)

-- Prove that B alone can do the work in 30 days
theorem B_alone_completion_time : ∀ (Wb : ℕ), work_done 30 (work_per_day B) = 30 * work_per_day B :=
by
  sorry

end B_alone_completion_time_l272_272846


namespace estimate_probability_l272_272021

/--  A bag contains four balls, each labeled with one of the characters "美", "丽", "惠", "州". 
Balls are drawn with replacement until both "惠" and "州" are drawn. 
The integer values 0, 1, 2, and 3 represent "惠", "州", "美", and "丽" respectively. 
Given 16 groups of three random numbers each, estimate the probability that the drawing stops exactly on the third draw. --/
def groups : list (list ℕ) := [[2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0],
                                [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [3, 3, 1], [3, 2, 0], [1, 2, 2], [2, 3, 3]]

def is_valid_group (group : list ℕ) : bool :=
  (group.head = some 0 ∨ group.head = some 1 ∨ group.tail.head = some 0 ∨ group.tail.head = some 1) ∧
  (group.drop 2).head = some 0 ∨ (group.drop 2).head = some 1

def valid_groups := groups.filter is_valid_group

theorem estimate_probability : (valid_groups.length : ℚ) / groups.length = 1/8 :=
sorry

end estimate_probability_l272_272021


namespace new_person_weight_l272_272771

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (initial_person_weight : ℝ) 
  (weight_increase : ℝ) (final_person_weight : ℝ) : 
  avg_increase = 2.5 ∧ num_persons = 8 ∧ initial_person_weight = 65 ∧ 
  weight_increase = num_persons * avg_increase ∧ final_person_weight = initial_person_weight + weight_increase 
  → final_person_weight = 85 :=
by 
  intros h
  sorry

end new_person_weight_l272_272771


namespace variance_of_data_set_is_4_l272_272187

/-- The data set for which we want to calculate the variance --/
def data_set : List ℝ := [2, 4, 5, 6, 8]

/-- The mean of the data set --/
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Calculation of the variance of a list given its mean
noncomputable def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_of_data_set_is_4 :
  variance data_set (mean data_set) = 4 :=
by
  sorry

end variance_of_data_set_is_4_l272_272187


namespace collinear_F_D_E_G_l272_272968

-- Definitions based on the given conditions
variables (A B C H M D E F G : Point)

-- Conditions
variables (h_acute : is_acute_triangle A B C)
variables (h_orthocenter : orthocenter H A B C)
variables (h_circumcenter_H : circle H M A B C)
variables (h_tangent_D : tangent_at D (circle H M) B)
variables (h_tangent_E : tangent_at E (circle H M) C)
variables (h_altitude_F : altitude CF A B C)
variables (h_altitude_G : altitude BG A B C)

-- Proof problem
theorem collinear_F_D_E_G : collinear F D E G :=
sorry

end collinear_F_D_E_G_l272_272968


namespace sum_is_integer_l272_272728

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 :=
  sorry

end sum_is_integer_l272_272728


namespace stationery_cost_l272_272503

theorem stationery_cost :
  let 
    pencil_price := 4
    pen_price := 5
    boxes := 15
    pencils_per_box := 80
    pencils_ordered := boxes * pencils_per_box
    total_cost_pencils := pencils_ordered * pencil_price
    pens_ordered := 2 * pencils_ordered + 300
    total_cost_pens := pens_ordered * pen_price
  in 
  total_cost_pencils + total_cost_pens = 18300 :=
by 
  sorry

end stationery_cost_l272_272503


namespace find_AD_l272_272682

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272682


namespace ap_bp_cp_leq_one_plus_x_sq_one_minus_x_l272_272773

variable {A B C O P : Type} [inner_product_space ℝ P]
variable [metric_space P]

variables (A B C O P : P)
variables (r x : ℝ)
variables (h_circumcircle : (dist O A = r) ∧ (dist O B = r) ∧ (dist O C = r))
variables (h_radius : r = 1)
variables (h_center : ∀ P : P, (dist O P = x))

theorem ap_bp_cp_leq_one_plus_x_sq_one_minus_x (O : P) (x : ℝ)
  (h1 : O ∈ metric.ball O (1 + x)) : 
  let A B C P : P in
  (dist A P * dist B P * dist C P) ≤ (1 + x)^2 * (1 - x) :=
sorry

end ap_bp_cp_leq_one_plus_x_sq_one_minus_x_l272_272773


namespace solution_set_of_inequality_l272_272079

variable {ℝ : Type*} [LinearOrderedField ℝ] (f : ℝ → ℝ) (x : ℝ)

def odd_function (h : ℝ -> ℝ) := ∀ x, h (-x) = -h x

def decreasing_function (g : ℝ → ℝ) := ∀ x y, x < y → g x > g y

noncomputable def g (f : ℝ → ℝ) := λ x, f x / exp x

theorem solution_set_of_inequality (hf_deriv : ∀ x, f x > deriv f x)
  (hf_odd : odd_function (λ x, f x + 2017)) :
  { x : ℝ | f x + 2017 * exp x < 0 } = set.Ioi 0 :=
by sorry

end solution_set_of_inequality_l272_272079


namespace radius_of_other_circle_l272_272811

theorem radius_of_other_circle (R r x : ℝ)
  (h1 : 2 * r < R)
  (h2 : 2 * x < R)
  (h3 : (sqrt (R^2 - 2 * R * r) + sqrt (R^2 - 2 * R * x)) = 2 * sqrt (r * x)) :
  x = (r * R * (3 * R - 2 * r + 2 * sqrt (2 * R * (R - 2 * r)))) / (R + 2 * r)^2 :=
by
  sorry

end radius_of_other_circle_l272_272811


namespace y_intercept_l272_272103

theorem y_intercept (x y : ℝ) (h : 4 * x + 7 * y = 28) : x = 0 → y = 4 :=
by
  intro hx
  rw [hx, zero_mul, add_zero] at h
  have := eq_div_of_mul_eq (by norm_num : 7 ≠ 0) h
  rw [eq_comm, div_eq_iff (by norm_num : 7 ≠ 0), mul_comm] at this
  exact this

end y_intercept_l272_272103


namespace fixer_used_30_percent_kitchen_l272_272470

def fixer_percentage (x : ℝ) : Prop :=
  let initial_nails := 400
  let remaining_after_kitchen := initial_nails * ((100 - x) / 100)
  let remaining_after_fence := remaining_after_kitchen * 0.3
  remaining_after_fence = 84

theorem fixer_used_30_percent_kitchen : fixer_percentage 30 :=
by
  exact sorry

end fixer_used_30_percent_kitchen_l272_272470


namespace example_problem_l272_272198

variables {Line Plane : Type} [AffineSpace Line Plane]

def perpendicular (l₁ : Line) (p₁ : Plane) : Prop := sorry -- define perpendicular relationship
def parallel (l₁ : Line) (p₁ : Plane) : Prop := sorry -- define parallel relationship 
def perpendicular_planes (p₁ p₂ : Plane) : Prop := sorry -- define perpendicular planes relationship

-- Given two different lines and planes
variables (m n : Line) (α β : Plane)

-- Prove that given conditions on lines and planes imply perpendicular planes
theorem example_problem (hmα : perpendicular m α) (hmβ : parallel m β) : perpendicular_planes α β :=
sorry

end example_problem_l272_272198


namespace smallest_number_with_2020_divisors_is_correct_l272_272114

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272114


namespace smallest_root_equation_l272_272564

theorem smallest_root_equation :
  ∃ x : ℝ, (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 ∧ ∀ y, (3 * y) / (y - 2) + (2 * y^2 - 28) / y = 11 → x ≤ y ∧ x = (-1 - Real.sqrt 17) / 2 :=
sorry

end smallest_root_equation_l272_272564


namespace roots_of_quadratic_l272_272591

theorem roots_of_quadratic (a b : ℝ) (h : ab ≠ 0) : 
  (a + b = -2 * b) ∧ (a * b = a) → (a = -3 ∧ b = 1) :=
by
  sorry

end roots_of_quadratic_l272_272591


namespace smallest_number_with_2020_divisors_is_correct_l272_272116

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272116


namespace line_in_slope_intercept_form_l272_272472

variable (x y : ℝ)

def line_eq (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 1) = 0

theorem line_in_slope_intercept_form (x y : ℝ) (h: line_eq x y) :
  y = (3 / 4) * x - 5 / 2 :=
sorry

end line_in_slope_intercept_form_l272_272472


namespace uncle_bradley_bills_l272_272812

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end uncle_bradley_bills_l272_272812


namespace train_passes_in_two_minutes_l272_272849

noncomputable def time_to_pass_through_tunnel : ℕ := 
  let train_length := 100 -- Length of the train in meters
  let train_speed := 72 * 1000 / 60 -- Speed of the train in m/min (converted)
  let tunnel_length := 2300 -- Length of the tunnel in meters (converted from 2.3 km to meters)
  let total_distance := train_length + tunnel_length -- Total distance to travel
  total_distance / train_speed -- Time in minutes (total distance divided by speed)

theorem train_passes_in_two_minutes : time_to_pass_through_tunnel = 2 := 
  by
  -- proof would go here, but for this statement, we use 'sorry'
  sorry

end train_passes_in_two_minutes_l272_272849


namespace starting_weight_of_labrador_puppy_l272_272661

theorem starting_weight_of_labrador_puppy :
  ∃ L : ℝ,
    (L + 0.25 * L) - (12 + 0.25 * 12) = 35 ∧ 
    L = 40 :=
by
  use 40
  sorry

end starting_weight_of_labrador_puppy_l272_272661


namespace sum_of_coefficients_of_nonzero_y_terms_l272_272434

theorem sum_of_coefficients_of_nonzero_y_terms :
  let p := (5 * X + 3 * Y + 2) * (2 * X + 5 * Y + 3) in
  (coeff p (X * Y) + coeff p (Y^2) + coeff p Y) = 65 := by
  sorry

end sum_of_coefficients_of_nonzero_y_terms_l272_272434


namespace distinct_values_of_products_l272_272020

theorem distinct_values_of_products (n : ℤ) (h : 1 ≤ n) :
  ¬ ∃ a b c d : ℤ, n^2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 ∧ ad = bc :=
sorry

end distinct_values_of_products_l272_272020


namespace percent_of_workday_in_meetings_l272_272743

theorem percent_of_workday_in_meetings (h1 : 9 > 0) (m1 m2 : ℕ) (h2 : m1 = 45) (h3 : m2 = 2 * m1) : 
  (135 / 540 : ℚ) * 100 = 25 := 
by
  -- Just for structure, the proof should go here
  sorry

end percent_of_workday_in_meetings_l272_272743


namespace min_distance_parabola_line_l272_272978

theorem min_distance_parabola_line :
  ∃ P : ℝ × ℝ, (∃ (y : ℝ), P = (y^2 / 4, y)) ∧
  ∀ Q : ℝ × ℝ, (∃ (y : ℝ), Q = (y^2 / 4, y)) →
  distance_to_line Q (3, 4, 15) ≥ (29 / 15) :=
begin
  sorry
end

-- Additional definitions for distance and distance_to_line
noncomputable def distance_to_line (P : ℝ × ℝ) (lin : ℝ × ℝ × ℝ) : ℝ :=
  let (a, b, c) := lin in
  abs (a * P.1 + b * P.2 + c) / real.sqrt (a * a + b * b)

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

end min_distance_parabola_line_l272_272978


namespace radius_of_circle_l272_272563

theorem radius_of_circle (x y : ℝ) (h : x^2 - 2*x + y^2 - 4*y = 12) : ∃ r : ℝ, r = sqrt 17 :=
by
  sorry

end radius_of_circle_l272_272563


namespace average_age_decrease_l272_272767

theorem average_age_decrease :
  let avg_original := 40
  let new_students := 15
  let avg_new_students := 32
  let original_strength := 15
  let total_age_original := original_strength * avg_original
  let total_age_new_students := new_students * avg_new_students
  let total_strength := original_strength + new_students
  let total_age := total_age_original + total_age_new_students
  let avg_new := total_age / total_strength
  avg_original - avg_new = 4 :=
by
  sorry

end average_age_decrease_l272_272767


namespace quadratic_interval_length_l272_272554

theorem quadratic_interval_length (a : ℝ) :
  (∀ x ∈ set.Icc (-4 : ℝ) 0, abs (a * x^2 + 4 * a * x - 1) ≤ 4) →
  (set.Icc (-5 / 4) 0 ∪ set.Icc 0 (3 / 4)).length = 2 :=
begin
  sorry
end

end quadratic_interval_length_l272_272554


namespace sum_divisible_by_100_l272_272754

theorem sum_divisible_by_100 (n : ℕ) : 
  (∑ k in finset.range (12 * n + 1), 93 ^ k) % 100 = 0 :=
sorry

end sum_divisible_by_100_l272_272754


namespace problem1_problem2_l272_272222

theorem problem1 (A B C : ℝ) (a b c q : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h4 : q > 0) (angle_prog : 2 * B = A + C) (angle_sum : A + B + C = π)
  (geo_prog : b^2 = a * c) : q = 1 := 
sorry

theorem problem2 (a b c q : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h4 : q > 0) (geo_prog : b^2 = a * c) (ineq1 : 1 + q > q^2)
  (ineq2 : 1 + q^2 > q) (ineq3 : q + q^2 > 1)
  (broot1 : ∃ q1 q2 : ℝ, q1 = (-1 + Real.sqrt 5) / 2 ∧ q2 = (1 + Real.sqrt 5) / 2 ∧ 
  (q1 < q ∧ q < q2)) : Fraction (-1 + Real.sqrt 5) / 2 < q ∧ q < Fraction (1 + Real.sqrt 5) / 2 :=
sorry

end problem1_problem2_l272_272222


namespace six_coins_all_heads_or_tails_probability_l272_272350

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l272_272350


namespace plane_equation_l272_272943

theorem plane_equation (A B C D : ℤ)
  (h1 : ∃ (P₁ P₂ : ℝ × ℝ × ℝ), P₁ = (0,0,0) ∧ P₂ = (2,-2,2))
  (h2 : A = 2 ∧ B = -1 ∧ C = 1 ∧ D = 0)
  (h3 : P₃ : ℝ × ℝ × ℝ, P₄ : ℝ × ℝ × ℝ, plane_eq : ∀ x y z, A * x + B * y + C * z + D = 0 → P₃ = (x,y,z) → P₄ = (2*x-2*y+2*z, -x+3*y+z, x-3*y+2*z))
  : A * x + B * y + C * z + D = 0 :=
sorry

end plane_equation_l272_272943


namespace sum_of_terms_7_8_9_l272_272295

namespace ArithmeticSequence

-- Define the sequence and its properties
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 0 + n * (n - 1) / 2 * (a 1 - a 0)

def condition3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def condition5 (S : ℕ → ℤ) : Prop :=
  S 5 = 30

-- Main statement to prove
theorem sum_of_terms_7_8_9 :
  is_arithmetic_sequence a →
  (∀ n, S n = sum_first_n_terms a n) →
  condition3 S →
  condition5 S →
  a 7 + a 8 + a 9 = 63 :=
by
  sorry

end ArithmeticSequence

end sum_of_terms_7_8_9_l272_272295


namespace combined_resistance_parallel_l272_272854

-- Define resistances x and y
def x : ℝ := 5
def y : ℝ := 6

-- Combined resistance in parallel r
noncomputable def r : ℝ := 30 / 11

-- Prove that the reciprocal of r is the sum of the reciprocals of x and y
theorem combined_resistance_parallel (x y : ℝ) (hx : x = 5) (hy : y = 6) : 
  (1 / r) = (1 / x) + (1 / y) := 
by
  rw [hx, hy]
  have : x ^ (-1) + y ^ (-1) = (1/5) + (1/6), by sorry
  have : 1/30 * 6 + 1/30 * 5 = 11/30, by sorry
  show 1 / r = 11 / 30, by sorry
  sorry

end combined_resistance_parallel_l272_272854


namespace find_initial_value_l272_272530

theorem find_initial_value (v : ℝ) 
  (h1 : ∀ (k : ℕ), k > 0 → v_k = v_(k-1) - v_(k-1) / 3) 
  (h2 : v_1 = 20) (h3 : v_2 = 12) : 
  v = 30 := 
by 
  sorry

end find_initial_value_l272_272530


namespace dogwood_trees_in_another_part_of_park_l272_272660

theorem dogwood_trees_in_another_part_of_park :
  ∀ (initial_trees_one_part : ℝ) (trees_to_cut_down : ℝ) (number_workers : ℝ) (remaining_trees : ℝ) (trees_other_part : ℝ),
  initial_trees_one_part = 5.0 →
  trees_to_cut_down = 7.0 →
  number_workers = 8.0 →
  remaining_trees = 2.0 →
  (initial_trees_one_part - trees_to_cut_down + trees_other_part = remaining_trees) →
  trees_other_part = 2 :=
begin
  intros,
  sorry
end

end dogwood_trees_in_another_part_of_park_l272_272660


namespace part1_part2_l272_272610

section

-- Definitions and conditions for Part 1
def P : ℕ → ℕ
| 234 := 9
| _ := 0  -- placeholder for other values

theorem part1 : P 234 = 9 :=
by
  exact rfl

-- Definitions and conditions for Part 2
variables {a b : ℕ}
def x (a b : ℕ) : ℕ := 100 * a + 10 * b + 3
def y (b : ℕ) : ℕ := 400 + 10 * b + 5

def P_x (a b : ℕ) : ℕ := a + b + 3
def P_y (b : ℕ) : ℕ := b + 9

theorem part2 (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) :
  P_x a b + P_y b = 20 → max (x a b + y b) (x (8 - 2 * b) b + y b) = 1028 :=
by
  intros h
  sorry  -- proof skipped

end

end part1_part2_l272_272610


namespace sum_proper_divisors_729_l272_272431

def proper_divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d, d ∣ n ∧ d ≠ n)

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

theorem sum_proper_divisors_729 : sum_proper_divisors 729 = 364 := by
  sorry

end sum_proper_divisors_729_l272_272431


namespace number_of_points_in_region_l272_272382

-- Define the lines and region
def y1 (x : ℝ) : ℝ := x
def y2 (x : ℝ) : ℝ := x^2 / 2^0.1

-- Define the point condition to be inside the region
def point_in_region (m n : ℕ) : Prop :=
  let x := 2^m
  let y := 2^n
  x > 0 ∧ y > 0 ∧ y < x ∧ y < 2^(2*m - (10 * Real.log2(Real.exp 1)) / 100)

-- Define the main theorem statement
theorem number_of_points_in_region : 
  (Finset.card (Finset.filter (λ mn : ℕ × ℕ, point_in_region mn.1 mn.2) 
    (Finset.product (Finset.range 100) (Finset.range 100)))) = 2401 := sorry

end number_of_points_in_region_l272_272382


namespace max_sundays_in_first_45_days_l272_272823

theorem max_sundays_in_first_45_days : 
  ∃ d : ℕ, (d ≤ 6) ∧ 
  (∀ n : ℕ, (n < 7) → (d + if n < 3 then 1 else 0 = 7)) := 
sorry

end max_sundays_in_first_45_days_l272_272823


namespace subset_closed_under_addition_eventually_linear_l272_272893

noncomputable def eventually_linear (S : set ℕ) :=
  ∃ (k N : ℕ), ∀ n, n > N → (n ∈ S ↔ k ∣ n)

theorem subset_closed_under_addition_eventually_linear
  (S : set ℕ) (hS₁ : S ⊆ ℕ) (hS₂ : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S) :
  eventually_linear S :=
sorry

end subset_closed_under_addition_eventually_linear_l272_272893


namespace difference_of_decimal_and_fraction_l272_272789

theorem difference_of_decimal_and_fraction :
  0.127 - (1 / 8) = 0.002 := 
by
  sorry

end difference_of_decimal_and_fraction_l272_272789


namespace smallest_number_with_2020_divisors_l272_272123

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272123


namespace number_of_terms_in_list_l272_272643

theorem number_of_terms_in_list : 
  let a := -48
  let l := 72
  let d := 6
  ∃ n : ℕ, n = ((l - a) / d) + 1 ∧ n = 21 :=
by
  sorry

end number_of_terms_in_list_l272_272643


namespace cylinder_ratio_l272_272653

theorem cylinder_ratio (h r : ℝ) (h_eq : h = 2 * Real.pi * r) : 
  h / r = 2 * Real.pi := 
by 
  sorry

end cylinder_ratio_l272_272653


namespace smallest_n_with_divisors_2020_l272_272136

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272136


namespace smallest_n_with_divisors_2020_l272_272139

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272139


namespace percentage_of_work_day_in_meetings_is_25_l272_272740

-- Define the conditions
def workDayHours : ℕ := 9
def firstMeetingMinutes : ℕ := 45
def secondMeetingMinutes : ℕ := 2 * firstMeetingMinutes
def totalMeetingMinutes : ℕ := firstMeetingMinutes + secondMeetingMinutes
def workDayMinutes : ℕ := workDayHours * 60

-- Define the percentage calculation
def percentageOfWorkdaySpentInMeetings : ℕ := (totalMeetingMinutes * 100) / workDayMinutes

-- The theorem to be proven
theorem percentage_of_work_day_in_meetings_is_25 :
  percentageOfWorkdaySpentInMeetings = 25 :=
sorry

end percentage_of_work_day_in_meetings_is_25_l272_272740


namespace sum_of_m_for_equal_area_triangles_l272_272897

theorem sum_of_m_for_equal_area_triangles (m : ℝ) :
  let A := (0, 0)
  let B := (2, 2)
  let C := (9 * m, 0)
  let line := \y = 2 * m * x\
  (line divides triangle A B C into two triangles of equal area) →
  ∃ m1 m2 : ℝ, 9 * m1^2 + 2 * m1 - 1 = 0 ∧ 9 * m2^2 + 2 * m2 - 1 = 0 ∧ m1 + m2 = -2 / 9 :=
sorry

end sum_of_m_for_equal_area_triangles_l272_272897


namespace part_one_part_two_l272_272624

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log x - a * x

theorem part_one (a : ℝ) (h : a = 1) :
  ∀ x ∈ set.Icc (1 : ℝ) real.exp (1 : ℝ), deriv (λ x, f x a) x < 0 :=
by sorry

theorem part_two (h : ∃ a > 0, ∀ x ∈ set.Icc (1 : ℝ) real.exp, f x a ≤ -4 ∧ (∃ c ∈ set.Icc (1 : ℝ) real.exp, f c a = -4)) :
  ∃ a = 4, ∀ x, f x 4 = real.log x - 4 * x :=
by sorry

end part_one_part_two_l272_272624


namespace shortest_chord_eq_l272_272971

-- Define the center of the circle D and point P
def D := (1 : ℝ, 0 : ℝ)
def P := (2 : ℝ, -1 : ℝ)

-- Define the circle equation 
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define a function to find the perpendicular line passing through P with the desired properties
def line_eq (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the main proof problem
theorem shortest_chord_eq :
  (∀ A B : ℝ × ℝ, (circle_eq A.1 A.2 → circle_eq B.1 B.2 → (∃ l : ℝ × ℝ → Prop, (∀ P : ℝ × ℝ, l P) ∧ l (2, -1)))) →
  line_eq P.1 P.2 :=
  sorry

end shortest_chord_eq_l272_272971


namespace smallest_n_with_divisors_2020_l272_272140

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272140


namespace problem_statement_l272_272609

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 + 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 3

-- Statement to prove: f(g(3)) - g(f(3)) = 61
theorem problem_statement : f (g 3) - g (f 3) = 61 := by
  sorry

end problem_statement_l272_272609


namespace nine_pow_y_eq_three_pow_fourteen_l272_272208

theorem nine_pow_y_eq_three_pow_fourteen (y : ℝ) (h : 9^y = 3^14) : y = 7 :=
by {
  sorry,
}

end nine_pow_y_eq_three_pow_fourteen_l272_272208


namespace binary_arithmetic_l272_272914

theorem binary_arithmetic 
  : (0b10110 + 0b1011 - 0b11100 + 0b11101 = 0b100010) :=
by
  sorry

end binary_arithmetic_l272_272914


namespace trapezoid_AD_value_l272_272688

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272688


namespace count_valid_rods_l272_272268

theorem count_valid_rods : 
  ∑ i in range(1, 41), 1 - 3 = 26 :=
by
  sorry

end count_valid_rods_l272_272268


namespace train_crosses_platform_in_20_seconds_l272_272049

theorem train_crosses_platform_in_20_seconds 
  (t : ℝ) (lp : ℝ) (lt : ℝ) (tp : ℝ) (sp : ℝ) (st : ℝ) 
  (pass_time : st = lt / tp) (lc : lp = 267) (lc_train : lt = 178) (cross_time : t = sp / st) : 
  t = 20 :=
by
  sorry

end train_crosses_platform_in_20_seconds_l272_272049


namespace slices_per_person_l272_272899

/-!
# Proof that each person ate 2 slices of pizza

Given:
- The total number of pizza slices originally is 16.
- The number of leftover slices is 4.
- The number of people who ate the pizza is 6.

Prove:
- The number of slices each person ate is 2.
-/

theorem slices_per_person
  (total_slices : ℕ)
  (leftover_slices : ℕ)
  (num_people : ℕ)
  (h_total : total_slices = 16)
  (h_leftover : leftover_slices = 4)
  (h_num_people : num_people = 6) :
  let slices_eaten := total_slices - leftover_slices in
  let slices_per_person := slices_eaten / num_people in
  slices_per_person = 2 :=
by
  sorry

end slices_per_person_l272_272899


namespace tangent_line_at_point_A_l272_272109

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

def point : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point_A :
  ∃ m b : ℝ, (∀ x : ℝ, (curve x - (m * x + b))^2 = 0) ∧  
  m = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_at_point_A_l272_272109


namespace circle_line_intersection_l272_272673

noncomputable def circle_intersects_line_condition (k : ℝ) : Prop :=
  let d := (abs (6 * k - 2)) / (sqrt (4 * k^2 + 1))
  d ≤ 2

theorem circle_line_intersection (k : ℝ) :
  (x y : ℝ) (hx : x^2 + y^2 - 6 * x + 8 = 0) (hl : y = 2 * k * x - 2) → 
  (∃ x y : ℝ, (x - 3)^2 + y^2 = 1 ∧ y = 2 * k * x - 2) ↔ (k ∈ set.Icc 0 (6/5)) :=
by
  sorry

end circle_line_intersection_l272_272673


namespace sum_in_A_n_iff_power_of_prime_l272_272570

-- Define a predicate for being a power of a single prime
def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), p.prime ∧ k > 0 ∧ n = p^k

-- Define the set A_n
def A_n (n : ℕ) : set ℕ :=
  {x | ¬ nat.coprime x n}

-- The main theorem
theorem sum_in_A_n_iff_power_of_prime (n : ℕ) :
  (∀ x y, x ∈ A_n n → y ∈ A_n n → x + y ∈ A_n n) ↔ is_power_of_prime n :=
by
  -- Proof is not required, so we use sorry
  sorry

end sum_in_A_n_iff_power_of_prime_l272_272570


namespace distance_between_parallel_lines_l272_272379

theorem distance_between_parallel_lines (A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_eqn1: A1 = 3) (h_eqn2: B1 = -2) (h_eqn3: C1 = -5) 
  (h_eqn4: A2 = 6) (h_eqn5: B2 = -4) (h_eqn6: C2 = 3) :
  let d := (|C1 - C2|) / (Real.sqrt (A1^2 + B1^2))
  in d = Real.sqrt 13 / 2 :=
by
  sorry

end distance_between_parallel_lines_l272_272379


namespace prove_f_f_1_over_27_l272_272620

def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 3
  else (1 / 2) ^ x

theorem prove_f_f_1_over_27 : f (f (1 / 27)) = 8 := by
  sorry

end prove_f_f_1_over_27_l272_272620


namespace smallest_n_with_2020_divisors_l272_272125

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272125


namespace initial_carrots_l272_272034

variable (C : ℕ)

-- Definitions based on conditions
def before_lunch_used := (2/5 : ℝ) * C
def remaining_after_lunch := (3/5 : ℝ) * C
def after_lunch_used := (3/5 : ℝ) * remaining_after_lunch
def remaining_at_end_of_day := (2/5 : ℝ) * remaining_after_lunch

-- Statement to prove the question
theorem initial_carrots (h: remaining_at_end_of_day = 72) : C = 300 :=
sorry

end initial_carrots_l272_272034


namespace corvette_trip_average_rate_l272_272011

theorem corvette_trip_average_rate :
  ∀ (t d f s : ℝ), 
    d = 640 ∧ 
    f = (d / 2) ∧ 
    s = 80 ∧ 
    t = (f / s) ∧ 
    (3 * t) + t = 16 →
    (d / 16) = 40 :=
by
  intros t d f s
  rintro ⟨hd, hf, hs, ht, ht_total⟩
  have h_total_distance : d = 640 := hd
  have hf_eq : f = d / 2 := hf
  have hs_eq : s = 80 := hs
  have ht_eq : t = f / s := ht
  have ht_total_eq : (3 * t) + t = 16 := ht_total
  rw [hf_eq, hs_eq, ht_eq] at ht_total_eq
  sorry

end corvette_trip_average_rate_l272_272011


namespace markup_percentage_l272_272857

variables (C M S : ℝ)

theorem markup_percentage (h1 : M = 0.10 * C) (h2 : S = C + M) :
  (M / S) * 100 = 9.09 :=
by
  have hS : S = 1.10 * C := by rw [h1, h2]; ring
  calc
    (M / S) * 100
        = (0.10 * C / (1.10 * C)) * 100 : by rw [h1, hS]
    ... = (0.10 / 1.10) * 100         : by rw [div_mul_cancel _ (ne_of_gt (by norm_num : 1.10 > 0))]
    ... = 9.09                        : by norm_num

end markup_percentage_l272_272857


namespace total_seeds_eaten_l272_272059

def first_seeds := 78
def second_seeds := 53
def third_seeds := second_seeds + 30

theorem total_seeds_eaten : first_seeds + second_seeds + third_seeds = 214 := by
  -- Sorry, placeholder for proof
  sorry

end total_seeds_eaten_l272_272059


namespace problem_part1_problem_part2_l272_272189

noncomputable def f (x a : ℝ) : ℝ := 0.5 * x^2 - (2 * a + 2) * x + (2 * a + 1) * (Real.log x)

theorem problem_part1 (a : ℝ) (h : 1/2 < a) :
  ((∀ x: ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x > 0) ∧ (∀ x: ℝ, x > 2 * a + 1 → deriv (λ x, f x a) x > 0)) :=
begin
  sorry
end

theorem problem_part2 (x1 x2 : ℝ) (a : ℝ) (λ : ℝ) (h₁ : 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) (h₂ : 3/2 ≤ a ∧ a ≤ 5/2) (h₃ : λ ≥ 8) :
  (|f x1 a - f x2 a| < λ * |1/x1 - 1/x2|) :=
begin
  sorry
end

end problem_part1_problem_part2_l272_272189


namespace sin_675_eq_neg_sqrt2_div_2_l272_272916

axiom angle_reduction (a : ℝ) : (a - 360 * (floor (a / 360))) * π / 180 = a * π / 180 - 2 * π * (floor (a / 360))

theorem sin_675_eq_neg_sqrt2_div_2 : real.sin (675 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1: real.sin (675 * real.pi / 180) = real.sin (315 * real.pi / 180),
  { rw [← angle_reduction 675, show floor(675 / 360:ℝ) = 1 by norm_num, int.cast_one, sub_self, zero_mul, add_zero] },
  have h2: 315 = 360 - 45,
  { norm_num },
  have h3: real.sin (315 * real.pi / 180) = - real.sin (45 * real.pi / 180),
  { rw [eq_sub_of_add_eq $ show real.sin (2 * π - 45 * real.pi / 180) = -real.sin (45 * real.pi / 180) by simp] },
  rw [h1, h3, real.sin_pi_div_four],
  norm_num

end sin_675_eq_neg_sqrt2_div_2_l272_272916


namespace trapezoid_AD_equal_l272_272693

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272693


namespace custom_op_example_l272_272537

def custom_op (a b : ℕ) : ℕ := (a + 1) / b

theorem custom_op_example : custom_op 2 (custom_op 3 4) = 3 := 
by
  sorry

end custom_op_example_l272_272537


namespace range_of_a_l272_272181

/-- Proposition p: ∀ x ∈ [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

/-- Proposition q: ∃ x₀ ∈ ℝ, x + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop := 
  ∃ x₀ : ℝ, ∃ x : ℝ, x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : prop_p a ∧ prop_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l272_272181


namespace number_of_sides_of_regular_polygon_l272_272214

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l272_272214


namespace pastries_average_per_day_l272_272022

theorem pastries_average_per_day :
  let monday_sales := 2
  let tuesday_sales := monday_sales + 1
  let wednesday_sales := tuesday_sales + 1
  let thursday_sales := wednesday_sales + 1
  let friday_sales := thursday_sales + 1
  let saturday_sales := friday_sales + 1
  let sunday_sales := saturday_sales + 1
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
  let days := 7
  total_sales / days = 5 := by
  sorry

end pastries_average_per_day_l272_272022


namespace mark_saves_5_dollars_l272_272481

def cost_per_pair : ℤ := 50

def promotionA_total_cost (cost : ℤ) : ℤ :=
  cost + (cost / 2)

def promotionB_total_cost (cost : ℤ) : ℤ :=
  cost + (cost - 20)

def savings (totalB totalA : ℤ) : ℤ :=
  totalB - totalA

theorem mark_saves_5_dollars :
  savings (promotionB_total_cost cost_per_pair) (promotionA_total_cost cost_per_pair) = 5 := by
  sorry

end mark_saves_5_dollars_l272_272481


namespace smallest_number_with_2020_divisors_l272_272120

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272120


namespace nine_point_circle_intersection_l272_272287

theorem nine_point_circle_intersection
  (O A B C : Point)
  (circumCircle : Circle O)
  (l : Line)
  (A1 B1 C1 : Point)
  (hL : l.passes_through O)
  (proj_A : projection_on_line A l A1)
  (proj_B : projection_on_line B l B1)
  (proj_C : projection_on_line C l C1)
  (perp_A1 : Line)
  (perp_B1 : Line)
  (perp_C1 : Line)
  (hA1 : perp_A1.passes_through A1 ∧ perp_A1.perpendicular_to (line_through B C))
  (hB1 : perp_B1.passes_through B1 ∧ perp_B1.perpendicular_to (line_through A C))
  (hC1 : perp_C1.passes_through C1 ∧ perp_C1.perpendicular_to (line_through A B))
  : ∃ P : Point, intersect_three_lines perp_A1 perp_B1 perp_C1 P ∧ lies_on_nine_point_circle P (triangle.mk A B C) :=
sorry

end nine_point_circle_intersection_l272_272287


namespace new_person_weight_l272_272772

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (initial_person_weight : ℝ) 
  (weight_increase : ℝ) (final_person_weight : ℝ) : 
  avg_increase = 2.5 ∧ num_persons = 8 ∧ initial_person_weight = 65 ∧ 
  weight_increase = num_persons * avg_increase ∧ final_person_weight = initial_person_weight + weight_increase 
  → final_person_weight = 85 :=
by 
  intros h
  sorry

end new_person_weight_l272_272772


namespace find_p_l272_272887

noncomputable def binomial_p (n : ℝ) : ℝ :=
  let p := 300 / n in
  have h1 : n = 300 / p, by sorry
  have h2 : 200 = (300 / p) * p * (1 - p), by sorry
  p

theorem find_p : ∃ p : ℝ, p = 1 / 3 ∧ (ξ : ℝ) ∼ binomial_p (200 / 3) := 
  sorry

end find_p_l272_272887


namespace line_through_center_of_circle_parallel_l272_272776

noncomputable def center_of_circle : ℝ × ℝ := (2, 0)
noncomputable def slope_of_parallel_line : ℝ := 2
noncomputable def line_equation (p : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop :=
  λ (q : ℝ × ℝ), q.2 - p.2 = m * (q.1 - p.1)

theorem line_through_center_of_circle_parallel 
  (C : ℝ × ℝ) (m : ℝ) (H : C = center_of_circle ∧ m = slope_of_parallel_line) :
  ∃ k : ℝ, ∀ p : ℝ × ℝ, line_equation C m p = (2 * p.1 - p.2 - 4 = 0) :=
sorry

end line_through_center_of_circle_parallel_l272_272776


namespace six_coins_all_heads_or_tails_probability_l272_272351

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l272_272351


namespace function_domain_l272_272557

noncomputable def f : ℝ → ℝ := λ x, (x^2 - 3*x + 2) / (x^2 - 5*x + 6)

theorem function_domain :
  (∀ x : ℝ, x ∉ {2, 3} → ∃ y : ℝ, y = f x) ∧ 
  (∀ x : ℝ, x ∈ {2, 3} → ¬ ∃ y : ℝ, y = f x) :=
sorry

end function_domain_l272_272557


namespace min_value_at_x_zero_l272_272956

noncomputable def f (x : ℝ) := Real.sqrt (x^2 + (x + 1)^2) + Real.sqrt (x^2 + (x - 1)^2)

theorem min_value_at_x_zero : ∀ x : ℝ, f x ≥ f 0 := by
  sorry

end min_value_at_x_zero_l272_272956


namespace quadr_pyramid_edge_sum_is_36_l272_272397

def sum_edges_quad_pyr (hex_sum_edges : ℕ) (hex_num_edges : ℕ) (quad_num_edges : ℕ) : ℕ :=
  let length_one_edge := hex_sum_edges / hex_num_edges
  length_one_edge * quad_num_edges

theorem quadr_pyramid_edge_sum_is_36 :
  sum_edges_quad_pyr 81 18 8 = 36 :=
by
  -- We defer proof
  sorry

end quadr_pyramid_edge_sum_is_36_l272_272397


namespace smallest_number_with_2020_divisors_is_correct_l272_272112

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272112


namespace ticket_price_increase_l272_272544

-- Define the initial price and the new price
def last_year_price : ℝ := 85
def this_year_price : ℝ := 102

-- Define the percent increase calculation
def percent_increase (initial : ℝ) (new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

-- Statement to prove
theorem ticket_price_increase (initial : ℝ) (new : ℝ) (h_initial : initial = last_year_price) (h_new : new = this_year_price) :
  percent_increase initial new = 20 :=
by
  sorry

end ticket_price_increase_l272_272544


namespace largest_y_l272_272275

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

theorem largest_y (x y : ℕ) (hx : x ≥ y) (hy : y ≥ 3) 
  (h : (interior_angle x * 28) = (interior_angle y * 29)) :
  y = 57 :=
by
  sorry

end largest_y_l272_272275


namespace probability_all_heads_or_tails_l272_272340

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l272_272340


namespace smallest_number_has_2020_divisors_l272_272135

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272135


namespace abs_z_bounds_l272_272976

open Complex

theorem abs_z_bounds (z : ℂ) (h : abs (z + 1/z) = 1) : 
  (Real.sqrt 5 - 1) / 2 ≤ abs z ∧ abs z ≤ (Real.sqrt 5 + 1) / 2 := 
sorry

end abs_z_bounds_l272_272976


namespace range_m_l272_272738

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 1/2 * x^2 - 2 * x + 5

theorem range_m (m : ℝ) : (∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), f x < m) → 7 < m :=
by
  sorry

end range_m_l272_272738


namespace sqrt_subtraction_compound_squares_l272_272517

-- Problem 1
theorem sqrt_subtraction : real.sqrt 18 - real.sqrt 32 + real.sqrt 2 = 0 :=
by
  sorry

-- Problem 2
theorem compound_squares : 
  (real.sqrt 3 + 2) ^ 2022 * (real.sqrt 3 - 2) ^ 2021 * (real.sqrt 3 - 3) = 3 + real.sqrt 3 :=
by
  sorry

end sqrt_subtraction_compound_squares_l272_272517


namespace problem_solution_l272_272543

noncomputable def parametric_line (t ϕ : ℝ) : ℝ × ℝ :=
  (t * Real.cos ϕ, -2 + t * Real.sin ϕ)

def curve_c (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def midpoint_trajectory (ϕ : ℝ) : ℝ × ℝ :=
  (Real.sin (2 * ϕ), -1 - Real.cos (2 * ϕ))

theorem problem_solution :
  (∀ t x y ϕ, 0 ≤ ϕ ∧ ϕ < Real.pi → parametric_line t ϕ = (x, y) → curve_c x y →
    (ϕ ∈ Set.Ioo (Real.pi / 3) (2 * Real.pi / 3))) ∧
  (∀ ϕ, ϕ ∈ Set.Ioo (Real.pi / 3) (2 * Real.pi / 3) →
    (∃ t1 t2, parametric_line t1 ϕ = (x1, y1) ∧ parametric_line t2 ϕ = (x2, y2) ∧
      midpoint_trajectory ϕ = ((x1 + x2) / 2, (y1 + y2) / 2))) :=
by
  sorry

end problem_solution_l272_272543


namespace Sam_distance_l272_272744

theorem Sam_distance (d_m t_m : ℝ) (t_s t_stop : ℝ)
  (h1 : d_m = 150) (h2 : t_m = 3) (h3 : t_s = 4) (h4 : t_stop = 0.5) :
  let avg_speed := d_m / t_m in
  let actual_driving_time := t_s - t_stop in
  let distance_s := avg_speed * actual_driving_time in
  distance_s = 175 :=
by
  sorry

end Sam_distance_l272_272744


namespace find_AD_l272_272710

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272710


namespace kingfisher_catch_difference_l272_272473

def pelicanFish : Nat := 13
def fishermanFish (K : Nat) : Nat := 3 * (pelicanFish + K)
def fishermanConditionFish : Nat := pelicanFish + 86

theorem kingfisher_catch_difference (K : Nat) (h1 : K > pelicanFish)
  (h2 : fishermanFish K = fishermanConditionFish) :
  K - pelicanFish = 7 := by
  sorry

end kingfisher_catch_difference_l272_272473


namespace two_vertical_asymptotes_l272_272935

-- Define the polynomial in the numerator and denominator
def numerator (d : ℝ) (x : ℝ) : ℝ := x^2 - 3 * x + d
def denominator (x : ℝ) : ℝ := x^2 - 2 * x - 8

-- Define the function g(x)
def g (d : ℝ) (x : ℝ) : ℝ := numerator d x / denominator x

-- State that g(x) has exactly two vertical asymptotes
theorem two_vertical_asymptotes (d : ℝ) : 
  (∀ x : ℝ, (denominator x = 0) → numerator d x ≠ 0) ↔ (d ≠ -4 ∧ d ≠ -10) := 
begin
  sorry
end

end two_vertical_asymptotes_l272_272935


namespace sum_of_digits_of_largest_n_l272_272280

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def single_digit_primes : List ℕ := [2, 3, 5, 7]

def valid_d : List ℕ := [3, 5, 7]

def possible_combinations : List (ℕ × ℕ × ℕ) :=
  valid_d.bind (λ d, single_digit_primes.bind (λ e,
    let p := 9 * d + e in
    if is_prime p then [(d, e, p)] else []))

def product_of_triplet (triplet : ℕ × ℕ × ℕ) : ℕ :=
  let (d, e, p) := triplet in d * e * p

def largest_n : ℕ :=
  (possible_combinations.map product_of_triplet).maximum.getD 0 

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_largest_n : sum_of_digits largest_n = 11 := sorry

end sum_of_digits_of_largest_n_l272_272280


namespace consecutive_integers_sum_150_count_valid_n_values_l272_272561

theorem consecutive_integers_sum_150 : 
  ∃ (n a : ℕ), (n ≥ 2) ∧ (n * (2 * a + n - 1) = 300) :=
by sorry

theorem count_valid_n_values : 
  (∃ n a : ℕ, (n ≥ 2) ∧ (n * (2 * a + n - 1) = 300) ∧ (2 * a = 300 / n - n + 1)) ∧
  finset.card { n : ℕ | ∃ a : ℕ, (n ≥ 2) ∧ (n * (2 * a + n - 1) = 300) ∧ (2 * a = 300 / n - n + 1) } = 3 :=
by sorry

end consecutive_integers_sum_150_count_valid_n_values_l272_272561


namespace statements_④_and_⑤_are_correct_l272_272509

def correct_statements : Finset ℕ := { 4, 5 }

def statement_4_valid (z : ℂ) : z = 1 - I → (2 / z + z^2).abs = real.sqrt 2 :=
by
  intro h
  rw h
  simp
  norm_num

def statement_5_valid (z : ℂ) : z = 1 / I → (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0 :=
by
  intro h
  rw h
  simp
  norm_num

theorem statements_④_and_⑤_are_correct :
  let z4 := 1 - Complex.I in
  let z5 := 1 / Complex.I in
  (statement_4_valid z4 rfl ∧ statement_5_valid z5 rfl) →
  ∀ n ∈ correct_statements, n = 4 ∨ n = 5 :=
by 
  intros _ h
  simp [correct_statements] at h
  exact finset.mem_singleton.1 h

end statements_④_and_⑤_are_correct_l272_272509


namespace locus_area_l272_272525

noncomputable def circle (centre : ℝ × ℝ) (radius : ℝ) :=
  { p | (p.1 - centre.1) ^ 2 + (p.2 - centre.2) ^ 2 = radius ^ 2 }

theorem locus_area 
  (B_center : ℝ × ℝ) 
  (B_radius : ℝ := 6 * real.sqrt 7)
  (A_center : ℝ × ℝ) 
  (A_radius : ℝ := real.sqrt 7)
  (H_contains : ∀ x y : ℝ, circle B_center B_radius (A_center.1 + x, A_center.2 + y) → circle B_center B_radius (A_center.1, A_center.2))
  (H_boundary : ∀ D : ℝ × ℝ, circle B_center B_radius D → ∃ X Y : ℝ × ℝ, circle B_center B_radius X ∧ circle B_center B_radius Y ∧ (tangent_from_points X Y A_center A_radius))
  : (π * (real.sqrt 168) ^ 2 = 168 * π) := by
  sorry

end locus_area_l272_272525


namespace matching_shoes_probability_l272_272462

noncomputable def number_of_ways_to_select_two (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem matching_shoes_probability (h : 14 = 7 * 2) :
  let total_ways := number_of_ways_to_select_two 14 in
  let matching_pairs := 7 in
  (matching_pairs : ℚ) / total_ways = 1 / 13 := by
  sorry

end matching_shoes_probability_l272_272462


namespace cricketer_total_score_l272_272468

theorem cricketer_total_score : 
    ∃ (T : ℝ), T = 132 ∧ 
    let boundaries := 12 * 4 in 
    let sixes := 2 * 6 in 
    let runs_between_wickets := 0.5454545454545454 * T in 
    T = runs_between_wickets + (boundaries + sixes) :=
by
  sorry

end cricketer_total_score_l272_272468


namespace cost_price_correct_l272_272505

variables (sp : ℕ) (profitPerMeter : ℕ) (metersSold : ℕ)

def total_profit (profitPerMeter metersSold : ℕ) : ℕ := profitPerMeter * metersSold
def total_cost_price (sp total_profit : ℕ) : ℕ := sp - total_profit
def cost_price_per_meter (total_cost_price metersSold : ℕ) : ℕ := total_cost_price / metersSold

theorem cost_price_correct (h1 : sp = 8925) (h2 : profitPerMeter = 10) (h3 : metersSold = 85) :
  cost_price_per_meter (total_cost_price sp (total_profit profitPerMeter metersSold)) metersSold = 95 :=
by
  rw [h1, h2, h3];
  sorry

end cost_price_correct_l272_272505


namespace shape_is_sphere_l272_272671

noncomputable def spherical_coordinates_shape (c : ℝ) (h : c > 0) : Prop :=
  ∀ (ρ θ φ : ℝ), ρ = c ∧ (0 ≤ φ ∧ φ ≤ π) ∧ (0 ≤ θ ∧ θ < 2 * π) → ∃ (x y z : ℝ), 
  (x, y, z) = (c * sin φ * cos θ, c * sin φ * sin θ, c * cos φ) ∧ (x^2 + y^2 + z^2 = c^2)

theorem shape_is_sphere (c : ℝ) (h : c > 0) : spherical_coordinates_shape c h := sorry

end shape_is_sphere_l272_272671


namespace triangle_has_two_acute_angles_l272_272896

theorem triangle_has_two_acute_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : ∀ {x}, x ∈ {A, B, C} → x ≤ 90) :
  ∃ a b, a ∈ {A, B, C} ∧ b ∈ {A, B, C} ∧ a < 90 ∧ b < 90 :=
by sorry

end triangle_has_two_acute_angles_l272_272896


namespace not_necessarily_square_l272_272886

structure Quadrilateral (A B C D : Type) :=
(equal_diagonals : ∀ AC BD : ℝ, AC = BD)
(right_angle_intersection : ∀ O : Type, ∠AOB = 90 ∧ ∠BOC = 90 ∧ ∠COD = 90 ∧ ∠DOA = 90)

theorem not_necessarily_square (A B C D O : Type) [Quadrilateral A B C D] :
  ¬(Quadrilateral A B C D → (AC = BD ∧ ∠AOB = 90 ∧ ∠BOC = 90 ∧ ∠COD = 90 ∧ ∠DOA = 90) → is_square A B C D) :=
sorry

end not_necessarily_square_l272_272886


namespace largest_m_for_factorial_product_l272_272586

theorem largest_m_for_factorial_product (m n : ℕ) (h : fact (m! * 2022!) = fact n) :
  m = 2022! - 1 :=
sorry

end largest_m_for_factorial_product_l272_272586


namespace find_AD_l272_272712

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272712


namespace measure_angle_EHG_l272_272240

theorem measure_angle_EHG (EFGH : Parallelogram) (x : ℝ)
  (h1 : EFGH.ang_EFG = 2 * EFGH.ang_FGH)
  (h2 : EFGH.ang_EFG + EFGH.ang_FGH = 180) :
  EFGH.ang_EHG = 60 :=
by
  sorry

end measure_angle_EHG_l272_272240


namespace conditional_probability_l272_272032

def prob_event_A : ℚ := 7 / 8 -- Probability of event A (at least one occurrence of tails)
def prob_event_AB : ℚ := 3 / 8 -- Probability of both events A and B happening (at least one occurrence of tails and exactly one occurrence of heads)

theorem conditional_probability (prob_A : ℚ) (prob_AB : ℚ) 
  (h1: prob_A = 7 / 8) (h2: prob_AB = 3 / 8) : 
  (prob_AB / prob_A) = 3 / 7 := 
by
  rw [h1, h2]
  norm_num

end conditional_probability_l272_272032


namespace line_always_passes_fixed_point_l272_272874

theorem line_always_passes_fixed_point (m : ℝ) : ∃ x y : ℝ, (x = -2) ∧ (y = 1) ∧ (y = m * x + (2 * m + 1)) :=
by
  use [-2, 1]
  simp
  sorry

end line_always_passes_fixed_point_l272_272874


namespace find_y_l272_272566

theorem find_y (y : ℚ) : (16 : ℚ)^(-3) = (2 : ℚ)^(80 / y) / ((2 : ℚ)^(50 / y) * (16 : ℚ)^(26 / y)) → y = 37 / 6 :=
by
  sorry

end find_y_l272_272566


namespace container_volume_ratio_l272_272508

theorem container_volume_ratio (A B : ℚ) (h : (2 / 3 : ℚ) * A = (1 / 2 : ℚ) * B) : A / B = 3 / 4 :=
by sorry

end container_volume_ratio_l272_272508


namespace feb_leap_year_rel_prime_count_l272_272480

def is_rel_prime (m d : ℕ) : Prop :=
  Nat.gcd m d = 1

def is_rel_prime_date_in_february (d : ℕ) : Prop :=
  is_rel_prime 2 d

theorem feb_leap_year_rel_prime_count : ∃ n, n = 15 ∧ ∀ d, (1 ≤ d ∧ d ≤ 29) → (is_rel_prime 2 d ↔ d ∉ Set.range (λ i, 2 * i)) :=
by
  sorry

end feb_leap_year_rel_prime_count_l272_272480


namespace six_coins_heads_or_tails_probability_l272_272363

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l272_272363


namespace calories_difference_l272_272465

variable (slices_cake slices_brownies : Nat)
variable (calories_per_slice_cake calories_per_brownie : Nat)

-- Given conditions
def slices_of_cake := 8
def calories_per_slice_of_cake := 347
def slices_of_brownies := 6
def calories_per_brownie := 375

-- Definition to calculate total calories in cake
def total_calories_cake : Nat :=
  slices_of_cake * calories_per_slice_of_cake

-- Definition to calculate total calories in brownies
def total_calories_brownies : Nat :=
  slices_of_brownies * calories_per_brownie

-- Statement to be proved
theorem calories_difference :
  total_calories_cake slices_of_cake calories_per_slice_of_cake - total_calories_brownies slices_of_brownies calories_per_brownie = 526 := by
  sorry

end calories_difference_l272_272465


namespace six_coins_heads_or_tails_probability_l272_272359

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l272_272359


namespace find_ordered_triple_l272_272777

variables (x y u v p : ℝ)

def equation1 := x = y^3
def equation2 := x + y^2 = 1

noncomputable def distance_between_intersections : ℝ :=
  let y_sol := [. . .] -- solutions to y^2(y + 1) = 1
  let points := y_sol.map (λ y, (y^3, y))
  let distances := [ distances between each pair in points ]
  distances.foldl min distances.head -- taking minimum pairwise distance

theorem find_ordered_triple : 
    ∃ u v p : ℝ, 
      u ≥ 0 ∧ v ≥ 0 ∧ p ≥ 0 ∧
      (distance_between_intersections x y u v p)^2 = u + v * real.sqrt p :=
sorry

end find_ordered_triple_l272_272777


namespace probability_sum_18_l272_272220

def is_valid_roll (d1 d2 d3 : ℕ) : Prop := 
  d1 >= 1 ∧ d1 <= 8 ∧ 
  d2 >= 1 ∧ d2 <= 8 ∧ 
  d3 >= 1 ∧ d3 <= 8

def valid_combinations : List (ℕ × ℕ × ℕ) :=
[(8, 8, 2), (8, 7, 3), (8, 6, 4), (8, 5, 5)]

def count_combinations (comb : (ℕ × ℕ × ℕ)) : ℕ :=
  if comb = (8, 8, 2) ∨ comb = (8, 5, 5) then 3
  else 6

theorem probability_sum_18 : 
  ( ∑ comb in valid_combinations, count_combinations comb * (1 / 8) ^ 3 ) = 9 / 256 := 
sorry

end probability_sum_18_l272_272220


namespace income_remaining_percentage_l272_272439

variable (income : ℝ)
variable (spent_on_food : ℝ)
variable (spent_on_education : ℝ)
variable (spent_on_rent : ℝ)
variable (remaining_income : ℝ)

-- Conditions
def conditions (income spent_on_food spent_on_education spent_on_rent remaining_income : ℝ) : Prop :=
  spent_on_food = 0.5 * income ∧
  spent_on_education = 0.15 * income ∧
  let remaining_after_food := income - spent_on_food in
  let remaining_after_education := remaining_after_food - spent_on_education in
  spent_on_rent = 0.5 * remaining_after_education ∧
  remaining_income = remaining_after_education - spent_on_rent

-- Question and Answer
def percent_left (income remaining_income : ℝ) : ℝ :=
  (remaining_income / income) * 100

-- Theorem
theorem income_remaining_percentage
  (income : ℝ)
  (spent_on_food : ℝ)
  (spent_on_education : ℝ)
  (spent_on_rent : ℝ)
  (remaining_income : ℝ)
  (h : conditions income spent_on_food spent_on_education spent_on_rent remaining_income) :
  percent_left income remaining_income = 17.5 :=
by
  sorry

end income_remaining_percentage_l272_272439


namespace trapezoid_angles_and_area_ratio_l272_272451

-- Definitions of given conditions
variables {A B C D K M P : Type} [Geometry Type]  
variable {BC AP BK : ℝ}
variable [BK_proof : ∀ (x : ℝ), x > 0 -> x == BK]
variable [AB_proof : ∀ (x : ℝ), x > 0 -> 2 * x == AB]

-- Stating the formal problem to prove
theorem trapezoid_angles_and_area_ratio :
  ∀ {AB AP BK BM KM BC α},  
  BC * 3 = AP ->
  AB = 2 * BC -> 
  α = Real.arctan (2 / Real.sqrt 5) -> 
  let S_ABCD := (AB * BC) in
  let S_ABKM := (32 * BC^2 * (2 / Real.sqrt(5)) * (Real.sqrt(5) + 1)/5) in
  (ABCD, ABKM) == (Real.angle A B M, Real.ratio S_ABCD S_ABKM) ->
  (Real.angle A B M = Real.arctan (2 / Real.sqrt 5)) ∧
  (S_ABCD / S_ABKM = 3 / (1 + 2 * Real.sqrt 2)) := 
by intros;
sorry

end trapezoid_angles_and_area_ratio_l272_272451


namespace reassemble_tetrahedron_l272_272324

-- This definition encapsulates any tetrahedron T
structure Tetrahedron :=
  (vertices : fin 4 → ℝ × ℝ × ℝ) -- A tetrahedron has 4 vertices in 3D space

-- The theorem statement that corresponds to the proof problem
theorem reassemble_tetrahedron (T : Tetrahedron) :
  ∃ (P : Plane) (parts : Unit → Tetrahedron), 
    (ereassembling parts' T1 ∧ reassemble parts T2 T) :=
sorry

end reassemble_tetrahedron_l272_272324


namespace smallest_A_divided_by_6_has_third_of_original_factors_l272_272474

theorem smallest_A_divided_by_6_has_third_of_original_factors:
  ∃ A: ℕ, A > 0 ∧ (∃ a b: ℕ, A = 2^a * 3^b ∧ (a + 1) * (b + 1) = 3 * a * b) ∧ A = 12 :=
by
  sorry

end smallest_A_divided_by_6_has_third_of_original_factors_l272_272474


namespace vector_coordinates_l272_272997

-- Define the given vectors.
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

-- Define the proof goal.
theorem vector_coordinates :
  -2 • a - b = (-3, -1) :=
by
  sorry -- Proof not required.

end vector_coordinates_l272_272997


namespace not_monotonic_range_l272_272993

def f (a x : ℝ) : ℝ := a * Real.log x + x^2 + (a - 6) * x

theorem not_monotonic_range (a : ℝ) :
  (∀ x ∈ Ioo 0 3, (∀ y ∈ Ioo (0 : ℝ) (3 : ℝ), (f a x) ≤ (f a y))
   ∨ (∀ y ∈ Ioo (0 : ℝ) (3 : ℝ), (f a x) ≥ (f a y))) ↔ 0 < a ∧ a < 2 :=
sorry

end not_monotonic_range_l272_272993


namespace Peter_vacation_l272_272311

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l272_272311


namespace Uncle_Bradley_bills_l272_272820

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end Uncle_Bradley_bills_l272_272820


namespace arithmetic_sequence_variance_l272_272969

variable (a1 d : ℝ)
def arithmetic_sequence (n : ℕ) : ℝ := a1 + (n - 1) * d

def mean_of_first_five_terms : ℝ := (arithmetic_sequence a1 d 1 + arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 3 + arithmetic_sequence a1 d 4 + arithmetic_sequence a1 d 5) / 5

def variance_of_first_five_terms : ℝ := ((arithmetic_sequence a1 d 1 - mean_of_first_five_terms a1 d)^2 + (arithmetic_sequence a1 d 2 - mean_of_first_five_terms a1 d)^2 + (arithmetic_sequence a1 d 3 - mean_of_first_five_terms a1 d)^2 + (arithmetic_sequence a1 d 4 - mean_of_first_five_terms a1 d)^2 + (arithmetic_sequence a1 d 5 - mean_of_first_five_terms a1 d)^2) / 5

theorem arithmetic_sequence_variance : variance_of_first_five_terms a1 d = 2 → d = 1 ∨ d = -1 := by
  sorry

end arithmetic_sequence_variance_l272_272969


namespace correct_options_l272_272632

variable {Ω : Type} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variable {A B : Set Ω}

open MeasureTheory

theorem correct_options (hA : P A > 0) (hB : P B > 0) :
  (Independent A B → P[B|A] = P B) ∧
  (P[B|A] = P B → P[A|B] = P A) ∧
  (P (A ∩ B) + P (Aᶜ ∩ B) = P B) :=
by
  sorry

end correct_options_l272_272632


namespace trapezoid_AD_value_l272_272690

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272690


namespace probability_three_common_books_l272_272303

-- Defining the total number of books
def total_books : ℕ := 12

-- Defining the number of books each of Harold and Betty chooses
def books_per_person : ℕ := 6

-- Assertion that the probability of choosing exactly 3 common books is 50/116
theorem probability_three_common_books :
  ((Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)) /
  ((Nat.choose 12 6) * (Nat.choose 12 6)) = 50 / 116 := by
  sorry

end probability_three_common_books_l272_272303


namespace all_cards_moved_from_original_position_ace_of_spades_not_next_to_empty_space_l272_272015

-- Definitions based on the problem's conditions
noncomputable def card := ℕ
noncomputable def empty_space : card := 0
noncomputable def deck := list card
noncomputable def shuffled_deck : deck := generate_shuffled_deck 52
noncomputable def vrungel_moves (deck : deck) (empty_space : card) (called_card : card) : deck := 
  if (called_card ≠ empty_space ∧ (next_to called_card empty_space deck)) then 
    move_to_empty space deck called_card 
  else 
    deck

-- Part (a): Lean statement for proving that every card has moved from its original position
theorem all_cards_moved_from_original_position : ∃ (fuks_strategy : ℕ → card), 
  ∀ t, ∃ n ≤ 52 * 52, (∃ m, vrungel_moves (step_moves initial_deck fuks_strategy n) empty_space (fuks_strategy m) 
        ≠ initial_deck) :=
sorry

-- Part (b): Lean statement for proving that the Ace of Spades cannot be ensured to never be next to the empty space
theorem ace_of_spades_not_next_to_empty_space : ¬∃ (fuks_strategy : ℕ → card),
  ∀ n, ¬ next_to 1 empty_space (step_moves shuffled_deck fuks_strategy n) :=
sorry

end all_cards_moved_from_original_position_ace_of_spades_not_next_to_empty_space_l272_272015


namespace divide_54_degree_angle_l272_272598

theorem divide_54_degree_angle :
  ∃ (angle_div : ℝ), angle_div = 54 / 3 :=
by
  sorry

end divide_54_degree_angle_l272_272598


namespace inequality_proof_l272_272963

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / real.sqrt (x + y) + y / real.sqrt (y + 1) + 1 / real.sqrt (x + 1)) ≥ 
  (y / real.sqrt (x + y) + x / real.sqrt (x + 1) + 1 / real.sqrt (y + 1)) :=
by
  sorry

end inequality_proof_l272_272963


namespace inscribed_square_area_l272_272488

noncomputable def area_of_inscribed_square : ℝ :=
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let t := (sqrt (8 / 3))
  let square_side := 2 * t
  let square_area := (square_side^2)
  square_area

theorem inscribed_square_area :
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let square_area := (2 * sqrt (8 / 3))^2
  square_area = 32 / 3 :=
by
  sorry

end inscribed_square_area_l272_272488


namespace range_of_m_l272_272171

theorem range_of_m (m : ℝ) :
  (∃ (x1 x2 : ℝ), (2*x1^2 - 2*x1 + 3*m - 1 = 0 ∧ 2*x2^2 - 2*x2 + 3*m - 1 = 0) ∧ (x1 * x2 > x1 + x2 - 4)) →
  -5/3 < m ∧ m ≤ 1/2 :=
by
  sorry

end range_of_m_l272_272171


namespace find_a_l272_272655

-- Define the condition that the polynomial product has no linear term of y
def no_linear_term (a : ℝ) : Prop := 
  let poly := (λ y : ℝ, (y + 2 * a) * (5 - y)) in
  ∀ y : ℝ, poly y = -y^2 + 10 * a → poly y

-- The theorem we need to prove
theorem find_a (a : ℝ) : no_linear_term a → a = 5 / 2 :=
by
  sorry

end find_a_l272_272655


namespace trays_needed_to_refill_l272_272087

theorem trays_needed_to_refill (initial_ice_cubes used_ice_cubes tray_capacity : ℕ)
  (h_initial: initial_ice_cubes = 130)
  (h_used: used_ice_cubes = (initial_ice_cubes * 8 / 10))
  (h_tray_capacity: tray_capacity = 14) :
  (initial_ice_cubes + tray_capacity - 1) / tray_capacity = 10 :=
by
  sorry

end trays_needed_to_refill_l272_272087


namespace vinegar_left_l272_272318

-- Define the initial quantities and conversion factors
def initial_jars := 4
def initial_cucumbers := 10
def initial_vinegar := 100
def pickles_per_cucumber := 6
def pickles_per_jar := 12
def vinegar_per_jar := 10

-- The main theorem to prove
theorem vinegar_left (jars : ℕ) (cucumbers : ℕ) (vinegar : ℕ) 
  (pickles_per_cucumber : ℕ) (pickles_per_jar : ℕ) (vinegar_per_jar : ℕ) : 
  (pickles_per_cucumber * cucumbers) ≥ (pickles_per_jar * jars) →
  vinegar - (jars * vinegar_per_jar) = 60 :=
by
  sorry

-- Substitute the given conditions and facts into the theorem
example : vinegar_left initial_jars initial_cucumbers initial_vinegar 
  pickles_per_cucumber pickles_per_jar vinegar_per_jar :=
by
  simp [initial_jars, initial_cucumbers, initial_vinegar, pickles_per_cucumber, pickles_per_jar, vinegar_per_jar]
  let jars := initial_jars
  let cucumbers := initial_cucumbers
  let vinegar := initial_vinegar
  let pickles_per_cucumber := pickles_per_cucumber
  let pickles_per_jar := pickles_per_jar
  let vinegar_per_jar := vinegar_per_jar

  have h1 : pickles_per_cucumber * cucumbers ≥ pickles_per_jar * jars := by
    calc
      60 = pickles_per_cucumber * cucumbers := by norm_num
      48 = pickles_per_jar * jars := by norm_num
      60 ≥ 48 := by norm_num
    
  apply vinegar_left
  assumption

end vinegar_left_l272_272318


namespace ball_reaches_top_left_pocket_l272_272476

-- Definitions based on the given problem
def table_width : ℕ := 26
def table_height : ℕ := 1965
def pocket_start : (ℕ × ℕ) := (0, 0)
def pocket_end : (ℕ × ℕ) := (0, table_height)
def angle_of_release : ℝ := 45

-- The goal is to prove that the ball will reach the top left pocket after reflections
theorem ball_reaches_top_left_pocket :
  ∃ reflections : ℕ, (reflections * table_width, reflections * table_height) = pocket_end :=
sorry

end ball_reaches_top_left_pocket_l272_272476


namespace num_integers_between_sqrt28_sqrt65_l272_272203

theorem num_integers_between_sqrt28_sqrt65 : 
  ∃ n : ℕ, ∀ x : ℕ, (⌊Real.sqrt 28⌋ < x ∧ x < ⌈Real.sqrt 65⌉) ↔ x ∈ {6, 7, 8} ∧ n = 3 :=
sorry

end num_integers_between_sqrt28_sqrt65_l272_272203


namespace product_of_roots_dodecagon_l272_272043

/-- A regular dodecagon with certain points defined in the coordinate plane -/
structure RegularDodecagon :=
(Q : Fin 12 → ℂ)
(hQ1 : Q 0 = 1)
(hQ7 : Q 6 = -1)
(center_origin : ∀ i : Fin 12, abs (Q i) = 1)

theorem product_of_roots_dodecagon (D : RegularDodecagon) :
  ∏ i in Finset.range 12, D.Q ⟨i, (Fin.is_lt i)⟩ = 1 := sorry

end product_of_roots_dodecagon_l272_272043


namespace pure_imaginary_condition_l272_272651

variable (a : ℝ)

def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition :
  isPureImaginary (a - 17 / (4 - (i : ℂ))) → a = 4 := 
by
  sorry

end pure_imaginary_condition_l272_272651


namespace fudge_solution_exists_l272_272373

noncomputable def fudge_amounts_eq (M L V : ℕ) (y : ℕ) (k : ℤ) : Prop :=
  (M = 9 * k) ∧ (L = 7 * k) ∧ (V = y * k) ∧ (M + L + V = 248) ∧ (M - L = 16)

theorem fudge_solution_exists : 
  ∃ (M L V : ℕ) (y k : ℤ), y = 15 ∧ fudge_amounts_eq M L V y k :=
begin
  sorry
end

end fudge_solution_exists_l272_272373


namespace mutually_exclusive_event_at_least_one_head_l272_272881

theorem mutually_exclusive_event_at_least_one_head :
  let event_A := "at most one head"
  let event_B := "both tosses are heads"
  let event_C := "exactly one head"
  let event_D := "both tosses are tails"
  let at_least_one_head := λ (throws : string), (throws.contains 'H')
  ∀ (throws : string), (event_D = "both tosses are tails") → ¬at_least_one_head throws :=
by
  sorry

end mutually_exclusive_event_at_least_one_head_l272_272881


namespace trapezoid_AD_CF_BC_CE_l272_272697

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272697


namespace correct_statements_l272_272980

def f (x : ℝ) := Real.log x / Real.log 2

def h (x : ℝ) := f (1 - |x|)

theorem correct_statements :
  (∀ x, h x = h (-x)) ∧
  (∀ x, h x ≤ 0) :=
by
  sorry

end correct_statements_l272_272980


namespace hyperbola_eccentricity_l272_272195

theorem hyperbola_eccentricity
  (a : ℝ)
  (h_hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / 9 = 1))
  (focus_coincide : ∃ c : ℝ, c = 4) :
  let b := 3 in
  let c := 4 in
  let a_squared := c^2 - b^2 in
  let e := c / real.sqrt a_squared in
  e = 4 * real.sqrt 7 / 7 := 
suffices h : a_squared = 7, 
by sorry,
begin
  sorry
end

end hyperbola_eccentricity_l272_272195


namespace smallest_number_with_2020_divisors_l272_272144

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272144


namespace smallest_number_with_2020_divisors_l272_272121

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l272_272121


namespace trapezoid_AD_CF_BC_CE_l272_272701

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272701


namespace h_in_terms_of_f_l272_272782

-- Definitions based on conditions in a)
def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) := f (-x)
def shift_left (f : ℝ → ℝ) (x : ℝ) (c : ℝ) := f (x + c)

-- Express h(x) in terms of f(x) based on conditions
theorem h_in_terms_of_f (f : ℝ → ℝ) (x : ℝ) :
  reflect_y_axis (shift_left f 2) x = f (-x - 2) :=
by
  sorry

end h_in_terms_of_f_l272_272782


namespace area_of_region_ABCDEFGHIJ_l272_272332

/-- 
  Given:
  1. Region ABCDEFGHIJ consists of 13 equal squares.
  2. Region ABCDEFGHIJ is inscribed in rectangle PQRS.
  3. Point A is on line PQ, B is on line QR, E is on line RS, and H is on line SP.
  4. PQ has length 28 and QR has length 26.

  Prove that the area of region ABCDEFGHIJ is 338 square units.
-/
theorem area_of_region_ABCDEFGHIJ 
  (squares : ℕ)             -- Number of squares in region ABCDEFGHIJ
  (len_PQ len_QR : ℕ)       -- Lengths of sides PQ and QR
  (area : ℕ)                 -- Area of region ABCDEFGHIJ
  (h1 : squares = 13)
  (h2 : len_PQ = 28)
  (h3 : len_QR = 26)
  : area = 338 :=
sorry

end area_of_region_ABCDEFGHIJ_l272_272332


namespace sum_f_ge_three_halves_l272_272995

noncomputable def f (x : ℝ) := 3^x / (3^x + 1)

theorem sum_f_ge_three_halves 
  (a b c : ℝ)
  (h : a + b + c = 0) : 
  f(a) + f(b) + f(c) ≥ 3 / 2 := 
sorry

end sum_f_ge_three_halves_l272_272995


namespace ratio_of_blue_to_red_marbles_l272_272906

theorem ratio_of_blue_to_red_marbles 
    (B R : ℕ) 
    (h1: B = R + 24) 
    (h2: R = 6) 
    : B = 30 ∧ (B / R = 5) 
by
  sorry

end ratio_of_blue_to_red_marbles_l272_272906


namespace magnitude_of_z_l272_272599

theorem magnitude_of_z (z : ℂ) (h : z + 2 * conj z = 3 - I) : |z| = Real.sqrt 2 :=
by sorry

end magnitude_of_z_l272_272599


namespace find_original_number_l272_272668

theorem find_original_number (a b c : ℕ) (h : 100 * a + 10 * b + c = 390) 
  (N : ℕ) (hN : N = 4326) : a = 3 ∧ b = 9 ∧ c = 0 :=
by 
  sorry

end find_original_number_l272_272668


namespace exists_special_set_l272_272325

theorem exists_special_set (n : ℕ) (hn : n ≥ 4) : 
  ∃ (S : Finset ℕ), 
    (S.card = n) ∧ 
    (∀ x ∈ S, x < 2 ^ (1 / n : ℝ)) ∧ 
    (∀ A B : Finset ℕ, A ≠ B → A ⊆ S → B ⊆ S → A.nonempty → B.nonempty → A.sum id ≠ B.sum id) := 
  sorry

end exists_special_set_l272_272325


namespace determine_positive_integers_l272_272933

theorem determine_positive_integers (x y z : ℕ) (h : x^2 + y^2 - 15 = 2^z) :
  (x = 0 ∧ y = 4 ∧ z = 0) ∨ (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 4 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 1) :=
sorry

end determine_positive_integers_l272_272933


namespace absolute_value_zero_l272_272541

theorem absolute_value_zero (x : ℝ) (h : |4 * x + 6| = 0) : x = -3 / 2 :=
sorry

end absolute_value_zero_l272_272541


namespace measure_of_angle_ehg_l272_272244

open Real

noncomputable def measure_angle_of_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : ℝ :=
  if h : FGH = 60 then 120 else sorry

theorem measure_of_angle_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : 
  ∃ (EHG : ℝ), EHG = 120 :=
begin
  use 120,
  sorry
end

end measure_of_angle_ehg_l272_272244


namespace find_solution_l272_272562

theorem find_solution (x y : ℝ) (h1 : y = (x + 2)^2) (h2 : x * y + y = 2) : 
  (x = real.cbrt 2 - 2 ∧ y = real.cbrt 4) :=
by
  sorry

end find_solution_l272_272562


namespace rachel_lunch_problems_l272_272755

theorem rachel_lunch_problems (problems_per_minute minutes_before_bed total_problems : ℕ) 
    (h1 : problems_per_minute = 5)
    (h2 : minutes_before_bed = 12)
    (h3 : total_problems = 76) : 
    (total_problems - problems_per_minute * minutes_before_bed) = 16 :=
by
    sorry

end rachel_lunch_problems_l272_272755


namespace new_sequence_69th_term_is_18th_l272_272257

theorem new_sequence_69th_term_is_18th (a : ℕ → ℕ) :
  let new_seq (n : ℕ) := if n % 4 = 0 then a (n / 4) else arbitrary ℕ in
  new_seq 68 = a 17 :=
by
  intro a
  let new_seq (n : ℕ) := if n % 4 = 0 then a (n / 4) else arbitrary ℕ
  exact eq.refl (a 17)

end new_sequence_69th_term_is_18th_l272_272257


namespace restore_example_l272_272228

theorem restore_example (x : ℕ) (y : ℕ) :
  (10 ≤ x * 8 ∧ x * 8 < 100) ∧ (100 ≤ x * 9 ∧ x * 9 < 1000) ∧ y = 98 → x = 12 ∧ x * y = 1176 :=
by
  sorry

end restore_example_l272_272228


namespace expected_value_is_minus_one_half_l272_272863

def prob_heads := 1 / 4
def prob_tails := 2 / 4
def prob_edge := 1 / 4
def win_heads := 4
def win_tails := -3
def win_edge := 0

theorem expected_value_is_minus_one_half :
  (prob_heads * win_heads + prob_tails * win_tails + prob_edge * win_edge) = -1 / 2 :=
by
  sorry

end expected_value_is_minus_one_half_l272_272863


namespace find_equation_of_ellipse_max_area_of_quadrilateral_l272_272606

variables {a b c : ℝ}

-- Definition of the equation of the ellipse and its conditions
def equation_of_ellipse : Prop := (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b * c = 1) ∧ (b = c)

-- The theorem stating the equation of the ellipse
theorem find_equation_of_ellipse (h : equation_of_ellipse) : 
  (∃ b, (b = 1) ∧ (a = √2) ∧ (∀ x y : ℝ, x^2 / 2 + y^2 = 1)) :=
sorry

-- The theorem stating the maximum area of quadrilateral ABCD
theorem max_area_of_quadrilateral (h : equation_of_ellipse) : 
  (∃ area, (area = 2*√2)) :=
sorry

end find_equation_of_ellipse_max_area_of_quadrilateral_l272_272606


namespace regular_polygon_sides_l272_272211

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l272_272211


namespace distance_between_points_l272_272982

theorem distance_between_points (m : ℝ) :
  (2: ℝ) = (45: ℝ) →
  let A := (m, 2)
  let B := (-m, m - 1)
  (forall slope : ℝ, 
    slope = 1 →
    (m - 3) / (-2 * m) = 1 →
    ∃ (m : ℝ), m = 1) ∧
  let A' := (1, 2)
  let B' := (-1, 0)
  ∃ (d : ℝ), d = (2: ℝ) * real.sqrt (2: ℝ) := 
sorry

end distance_between_points_l272_272982


namespace frog_toad_problem_l272_272799

theorem frog_toad_problem :
  let frogs := 2017
  let toads := 2017
  let pairings := Set (Set (ℕ × ℕ)) -- Represent the set of valid pairings
  let N := ∀ (frogs toads : ℕ), (pairings frogs toads) -> ℕ -- Number of pairings function
  let D := ∀ (frogs toads : ℕ), (pairings frogs toads) -> ℕ -- Number of distinct values of N
  let S := ∀ (frogs toads : ℕ), (pairings frogs toads) -> ℕ -- Sum of all distinct values of N
  ∃ (frogs toads pairings),
  frogs = 2017 ∧
  toads = 2017 ∧
  (∀ (f : ℕ), f < frogs → ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 < toads ∧ t2 < toads) → -- Each frog is friends with exactly 2 distinct toads
  D = 1009 ∧
  S = 2^1009 - 2 :=
begin
  sorry
end

end frog_toad_problem_l272_272799


namespace raised_bed_area_l272_272066

theorem raised_bed_area (length width : ℝ) (total_area tilled_area remaining_area raised_bed_area : ℝ) 
(h_len : length = 220) (h_wid : width = 120)
(h_total_area : total_area = length * width)
(h_tilled_area : tilled_area = total_area / 2)
(h_remaining_area : remaining_area = total_area / 2)
(h_raised_bed_area : raised_bed_area = (2 / 3) * remaining_area) : raised_bed_area = 8800 :=
by
  have h1 : total_area = 220 * 120, from by rw [h_total_area, h_len, h_wid]
  have h2 : tilled_area = 26400 / 2, from by rw [h_tilled_area, h1]
  have h3 : remaining_area = 26400 / 2, from by rw [h_remaining_area, h1]
  have h4 : raised_bed_area = (2 / 3) * 13200, from by rw [h_raised_bed_area, h3]
  have h5 : raised_bed_area = 8800, from by rwa [← h_raised_bed_area, h4]
  exact h5

end raised_bed_area_l272_272066


namespace find_a_l272_272583

noncomputable def f (x : ℝ) (a : ℝ) := x^3 + a * x^2 - 9 * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x - 9

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (f x a).deriv = f_prime x a) →
  (∀ x : ℝ, 12 * x + (f x a) = 6) →
  a = 3 ∨ a = -3 :=
by
  -- provided conditions
  intros _ _
  sorry

end find_a_l272_272583


namespace largest_m_for_factorial_product_l272_272585

theorem largest_m_for_factorial_product (m n : ℕ) (h : fact (m! * 2022!) = fact n) :
  m = 2022! - 1 :=
sorry

end largest_m_for_factorial_product_l272_272585


namespace six_coins_heads_or_tails_probability_l272_272364

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l272_272364


namespace polygon_info_l272_272276

noncomputable def sum_interior_angles (P : Type) [polygon P] : ℝ :=
  by sorry

def is_regular (P : Type) [polygon P] : Prop :=
  by sorry

theorem polygon_info (P : Type) [polygon P] (a b : P → ℝ)
  (h1 : ∀ p, a p = 9 * b p)
  (h2 : ∑ p in vertices P, b p = 360) :
  sum_interior_angles P = 3240 ∧ (is_regular P ∨ ¬ is_regular P) :=
  by sorry

end polygon_info_l272_272276


namespace find_number_l272_272836

theorem find_number (N x : ℕ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 :=
by
  sorry

end find_number_l272_272836


namespace peter_vacation_saving_l272_272314

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l272_272314


namespace area_of_overlap_l272_272075

open Real

theorem area_of_overlap (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : cos α = 3/5) :
  let side_length := 2
  let area := (side_length * side_length * (atan 1 - atan 1/2)) / cos(π/4 - α/2) in
  area = 4 / 3 :=
by
  sorry

end area_of_overlap_l272_272075


namespace calculate_expression_l272_272519

theorem calculate_expression :
  |(-Real.sqrt 3)| - (1/3)^(-1/2 : ℝ) + 2 / (Real.sqrt 3 - 1) - 12^(1/2 : ℝ) = 1 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l272_272519


namespace product_sum_inequality_l272_272288

open BigOperators

theorem product_sum_inequality (n : ℕ) (x : Fin n → ℝ)
  (h₁ : 0 < n)
  (h₂ : ∀ i, 0 < x i)
  (h₃ : (∏ i in Finset.univ, x i) = 1) :
  (∑ i in Finset.range n, x i * Real.sqrt (∑ j in Finset.range (i + 1), (x j) ^ 2)) ≥ (n + 1) / 2 * Real.sqrt n :=
begin
  sorry
end

end product_sum_inequality_l272_272288


namespace odds_of_blue_marble_l272_272912

def total_marbles : ℕ := 120
def yellow_marbles : ℕ := 30
def green_marbles : ℕ := yellow_marbles / 3
def red_marbles : ℕ := 2 * yellow_marbles
def blue_marbles : ℕ := total_marbles - (yellow_marbles + green_marbles + red_marbles)

theorem odds_of_blue_marble : (blue_marbles : ℚ) / total_marbles * 100 = 16.67 := by
  have h1 : green_marbles = yellow_marbles / 3 := rfl
  have h2 : red_marbles = 2 * yellow_marbles := rfl
  have h3 : yellow_marbles + green_marbles + red_marbles = 100 := by ipsub
  have h4 : blue_marbles = 20 := by sorry
  have h5 : (20 : ℚ) / 120 * 100 = 16.67 := by simp [div_eq_mul_inv]
  exact h5

end odds_of_blue_marble_l272_272912


namespace find_a_values_l272_272096

open Complex Polynomial

noncomputable def roots_form_parallelogram (a : ℝ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ), is_root (X^4 - 4*X^3 + 9*a*X^2 - 2*(a^2 + 2*a - 5)*X + 2 : Polynomial ℂ) z₁ ∧
                         is_root (X^4 - 4*X^3 + 9*a*X^2 - 2*(a^2 + 2*a - 5)*X + 2 : Polynomial ℂ) z₂ ∧
                         is_root (X^4 - 4*X^3 + 9*a*X^2 - 2*(a^2 + 2*a - 5)*X + 2 : Polynomial ℂ) z₃ ∧
                         is_root (X^4 - 4*X^3 + 9*a*X^2 - 2*(a^2 + 2*a - 5)*X + 2 : Polynomial ℂ) z₄ ∧
                         (z₁ + z₂ + z₃ + z₄ = 4 ∧
                          (z₁ - 1) + (z₂ - 1) + (z₃ - 1) + (z₄ - 1) = 0 ∧
                          ∃ (w₁ w₃ : ℂ), z₁ = 1 + w₁ ∧ z₂ = 1 - w₁ ∧ z₃ = 1 + w₃ ∧ z₄ = 1 - w₃)

theorem find_a_values :
  ∀ (a : ℝ), roots_form_parallelogram a → a = -1 + Real.sqrt 6 ∨ a = -1 - Real.sqrt 6 := 
sorry

end find_a_values_l272_272096


namespace quadrilateral_AMFD_area_l272_272783

structure Trapezoid :=
(height : ℝ)
(BC AD : ℝ)
(BE : ℝ)
(CD_mid : ℝ)
(intersection : ℝ)

def quadrilateral_area (trap : Trapezoid) : ℝ :=
  let S_AMN := (1 / 2) * 8 * 4
  let S_FDN := (1 / 2) * 3 * (5 / 2)
  S_AMN - S_FDN

theorem quadrilateral_AMFD_area (trap : Trapezoid) : quadrilateral_area trap = 49 / 4 :=
  have h1 : trap.height = 5, from rfl,
  have h2 : trap.BC = 3, from rfl,
  have h3 : trap.AD = 5, from rfl,
  have h4 : trap.BE = 2, from rfl,
  have h5 : trap.CD_mid = 2.5, from rfl,
  have h6 : trap.intersection = 4, from rfl,
  sorry

end quadrilateral_AMFD_area_l272_272783


namespace pictures_per_album_l272_272861

theorem pictures_per_album (phone_pics camera_pics albums pics_per_album : ℕ)
  (h1 : phone_pics = 7) (h2 : camera_pics = 13) (h3 : albums = 5)
  (h4 : pics_per_album * albums = phone_pics + camera_pics) :
  pics_per_album = 4 :=
by
  sorry

end pictures_per_album_l272_272861


namespace stationery_cost_l272_272500

theorem stationery_cost (cost_per_pencil cost_per_pen : ℕ)
    (boxes : ℕ)
    (pencils_per_box pens_offset : ℕ)
    (total_cost : ℕ) :
    cost_per_pencil = 4 →
    boxes = 15 →
    pencils_per_box = 80 →
    pens_offset = 300 →
    cost_per_pen = 5 →
    total_cost = (boxes * pencils_per_box * cost_per_pencil) +
                 ((2 * (boxes * pencils_per_box + pens_offset)) * cost_per_pen) →
    total_cost = 18300 :=
by
  intros
  sorry

end stationery_cost_l272_272500


namespace magnitude_vector_sub_eq_sqrt6_l272_272200

open Real

noncomputable def vector_sub_magnitude (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  ‖a - b‖

theorem magnitude_vector_sub_eq_sqrt6
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 2)
  (hab : inner a b = 1) : 
  vector_sub_magnitude a b = sqrt 6 :=
by
  sorry

end magnitude_vector_sub_eq_sqrt6_l272_272200


namespace spending_representation_l272_272523

theorem spending_representation :
  ∀ (x : Int), (x > 0 → -x < 0) →
  (spending_300 : Int) (receiving_500 : Int) (denote_receiving : receiving_500 = 500) (denote_spending : spending_300 = -300),
  spending_300 = -300 := 
by
  intro x h spending_300 receiving_500 denote_receiving denote_spending
  exact denote_spending
  sorry

end spending_representation_l272_272523


namespace area_of_square_inscribed_in_ellipse_l272_272491

noncomputable def areaSquareInscribedInEllipse : ℝ :=
  let a := 4
  let b := 8
  let s := (2 * Real.sqrt (b / 3)).toReal in
  (2 * s) ^ 2

theorem area_of_square_inscribed_in_ellipse :
  (areaSquareInscribedInEllipse) = 32 / 3 :=
sorry

end area_of_square_inscribed_in_ellipse_l272_272491


namespace probability_all_heads_or_tails_l272_272341

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l272_272341


namespace sin_675_eq_neg_sqrt2_div_2_l272_272917

axiom angle_reduction (a : ℝ) : (a - 360 * (floor (a / 360))) * π / 180 = a * π / 180 - 2 * π * (floor (a / 360))

theorem sin_675_eq_neg_sqrt2_div_2 : real.sin (675 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1: real.sin (675 * real.pi / 180) = real.sin (315 * real.pi / 180),
  { rw [← angle_reduction 675, show floor(675 / 360:ℝ) = 1 by norm_num, int.cast_one, sub_self, zero_mul, add_zero] },
  have h2: 315 = 360 - 45,
  { norm_num },
  have h3: real.sin (315 * real.pi / 180) = - real.sin (45 * real.pi / 180),
  { rw [eq_sub_of_add_eq $ show real.sin (2 * π - 45 * real.pi / 180) = -real.sin (45 * real.pi / 180) by simp] },
  rw [h1, h3, real.sin_pi_div_four],
  norm_num

end sin_675_eq_neg_sqrt2_div_2_l272_272917


namespace y_intercept_l272_272104

theorem y_intercept (x y : ℝ) (h : 4 * x + 7 * y = 28) : x = 0 → y = 4 :=
by
  intro hx
  rw [hx, zero_mul, add_zero] at h
  have := eq_div_of_mul_eq (by norm_num : 7 ≠ 0) h
  rw [eq_comm, div_eq_iff (by norm_num : 7 ≠ 0), mul_comm] at this
  exact this

end y_intercept_l272_272104


namespace number_of_valid_pairs_l272_272082

theorem number_of_valid_pairs :
  (∃! S : ℕ, S = 1250 ∧ ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1000) →
  (3^n < 4^m ∧ 4^m < 4^(m+1) ∧ 4^(m+1) < 3^(n+1))) :=
sorry

end number_of_valid_pairs_l272_272082


namespace price_difference_l272_272072

-- Define the conditions
def dress_cost_after_discount := 71.4
def discount_percent := 0.15
def increase_percent := 0.25

-- Define the properties
def original_price (p : ℝ) : Prop :=
  0.85 * p = dress_cost_after_discount

def final_price (p : ℝ) : ℝ :=
  dress_cost_after_discount * (1 + increase_percent)

-- The theorem to be proven
theorem price_difference (p : ℝ) (h : original_price p) : 
  final_price p - p = 5.25 :=
by
  sorry

end price_difference_l272_272072


namespace sin_675_eq_neg_sqrt2_over_2_l272_272921

theorem sin_675_eq_neg_sqrt2_over_2 :
  sin (675 * Real.pi / 180) = - (Real.sqrt 2 / 2) := 
by
  -- problem states that 675° reduces to 315°
  have h₁ : (675 : ℝ) ≡ 315 [MOD 360], by norm_num,
  
  -- recognize 315° as 360° - 45°
  have h₂ : (315 : ℝ) = 360 - 45, by norm_num,

  -- in the fourth quadrant, sin(315°) = -sin(45°)
  have h₃ : sin (315 * Real.pi / 180) = - (sin (45 * Real.pi / 180)), by
    rw [Real.sin_angle_sub_eq_sin_add, Real.sin_angle_eq_sin_add],
    
  -- sin(45°) = sqrt(2)/2
  have h₄ : sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by
    -- As an assumed known truth for this problem
    exact Real.sin_pos_of_angle,

  -- combine above facts
  rw [h₃, h₄],
  norm_num
  -- sorry is needed if proof steps aren't complete
  sorry

end sin_675_eq_neg_sqrt2_over_2_l272_272921


namespace length_of_grassy_plot_l272_272889

theorem length_of_grassy_plot (L : ℝ) : 
    (width_grassy_plot : ℝ) 
    (width_gravel_path : ℝ) 
    (cost_per_m2 : ℝ) 
    (total_cost : ℝ) 
    :=
    width_grassy_plot = 65 → 
    width_gravel_path = 2.5 → 
    cost_per_m2 = 0.5 → 
    total_cost = 425 → 
    70 * (L + 5) - 65 * L = total_cost / cost_per_m2 → 
    L = 100 :=
sorry

end length_of_grassy_plot_l272_272889


namespace inscribed_square_area_l272_272487

noncomputable def area_of_inscribed_square : ℝ :=
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let t := (sqrt (8 / 3))
  let square_side := 2 * t
  let square_area := (square_side^2)
  square_area

theorem inscribed_square_area :
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let square_area := (2 * sqrt (8 / 3))^2
  square_area = 32 / 3 :=
by
  sorry

end inscribed_square_area_l272_272487


namespace total_number_of_bills_l272_272815

theorem total_number_of_bills (total_money : ℕ) (fraction_for_50_bills : ℚ) (fifty_bill_value : ℕ) (hundred_bill_value : ℕ) :
  total_money = 1000 →
  fraction_for_50_bills = 3 / 10 →
  fifty_bill_value = 50 →
  hundred_bill_value = 100 →
  let money_for_50_bills := total_money * fraction_for_50_bills in
  let num_50_bills := money_for_50_bills / fifty_bill_value in
  let rest_money := total_money - money_for_50_bills in
  let num_100_bills := rest_money / hundred_bill_value in
  num_50_bills + num_100_bills = 13 :=
by
  intros h1 h2 h3 h4
  let money_for_50_bills := 1000 * (3 / 10)
  have h5 : money_for_50_bills = 300 := by sorry
  have h6 : 300 / 50 = 6 := by sorry
  let rest_money := 1000 - 300
  have h7 : rest_money = 700 := by sorry
  have h8 : 700 / 100 = 7 := by sorry
  have total_bills := 6 + 7
  show total_bills = 13 from eq.refl 13

end total_number_of_bills_l272_272815


namespace line_intersects_circle_l272_272614

theorem line_intersects_circle
    (r : ℝ) (d : ℝ)
    (hr : r = 6) (hd : d = 5) : d < r :=
by
    rw [hr, hd]
    exact by norm_num

end line_intersects_circle_l272_272614


namespace gcd_lcm_sum_18_30_45_l272_272283

theorem gcd_lcm_sum_18_30_45 :
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 :=
by
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  have hA : A = 3 := by sorry -- Proof of GCD computation
  have hB : B = 90 := by sorry -- Proof of LCM computation
  rw [hA, hB]
  norm_num

end gcd_lcm_sum_18_30_45_l272_272283


namespace smallest_number_with_2020_divisors_is_correct_l272_272113

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272113


namespace bob_final_total_score_l272_272070

theorem bob_final_total_score 
  (points_per_correct : ℕ := 5)
  (points_per_incorrect : ℕ := 2)
  (correct_answers : ℕ := 18)
  (incorrect_answers : ℕ := 2) :
  (points_per_correct * correct_answers - points_per_incorrect * incorrect_answers) = 86 :=
by 
  sorry

end bob_final_total_score_l272_272070


namespace problem_statement_l272_272656

-- Define the sequence and its properties
def seq_pos_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, 0 < a n

def sqrt_sum_eq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in finset.range (n+1), real.sqrt (a i)) = n^2 + n

-- Theorem statement
theorem problem_statement (a : ℕ → ℝ) (h_pos : seq_pos_terms a) (h_sum : ∀ n, sqrt_sum_eq a n) :
  ∀ n, (∑ i in finset.range (n+1), (a i / (i + 1))) = 2*n^2 + 2*n :=
sorry

end problem_statement_l272_272656


namespace sqrt_seventeen_subtract_decimal_part_l272_272216

theorem sqrt_seventeen_subtract_decimal_part :
  let a := (√17 - 4) in a - √17 = -4 :=
by
  let a := (Real.sqrt 17 - 4)
  have h : a - Real.sqrt 17 = -4 := sorry
  exact h

end sqrt_seventeen_subtract_decimal_part_l272_272216


namespace remainder_div_by_7_l272_272835

theorem remainder_div_by_7 (n : ℤ) (k m : ℤ) (r : ℤ) (h₀ : n = 7 * k + r) (h₁ : 3 * n = 7 * m + 3) (hrange : 0 ≤ r ∧ r < 7) : r = 1 :=
by
  sorry

end remainder_div_by_7_l272_272835


namespace ratio_of_radii_l272_272374

variable {a : ℝ} -- The length of the base of the isosceles triangle
variable {φ : ℝ} -- Base angle of the isosceles triangle in radians

-- Definition of the circumradius R
def circumradius (a : ℝ) (φ : ℝ) : ℝ :=
  a / (2 * sin (2 * φ))

-- Definition of the inradius r
def inradius (a : ℝ) (φ : ℝ) : ℝ :=
  (a / 2) * tan (φ / 2)

-- The required ratio r/R
def ratio_r_R (a : ℝ) (φ : ℝ) : ℝ :=
  let r := inradius a φ
  let R := circumradius a φ
  r / R

-- Theorem stating the required ratio
theorem ratio_of_radii (a : ℝ) (φ : ℝ) : ratio_r_R a φ = tan (φ / 2) * 2 * sin φ * cos φ := by
  sorry

end ratio_of_radii_l272_272374


namespace derivative_of_f_l272_272594

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x - 1) * exp (2 - x)

theorem derivative_of_f :
  (deriv f) = (λ x, (3 - x^2) * exp(2 - x)) :=
by
  sorry

end derivative_of_f_l272_272594


namespace count_variant_monotonic_l272_272080

def is_decreasing_seq (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i, i < digits.length - 1 → digits[i] > digits[i+1]

def is_variant_monotonic (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i, i < digits.length - 1 → (digits[i] = 2 * digits[i+1] ∨ digits[i] > digits[i+1])

def count_variant_monotonic_up_to_4_digits : ℕ :=
  Finset.filter is_variant_monotonic (Finset.range 10000).card

theorem count_variant_monotonic : count_variant_monotonic_up_to_4_digits = 14 := sorry

end count_variant_monotonic_l272_272080


namespace interval_monotonic_decrease_min_value_g_l272_272733

noncomputable def a (x : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.sin x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := let (a1, a2) := a x; let (b1, b2) := b x; a1 * b1 + a2 * b2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x + m

theorem interval_monotonic_decrease (x : ℝ) (k : ℤ) :
  0 ≤ x ∧ x ≤ Real.pi ∧ (2 * x + Real.pi / 6) ∈ [Real.pi/2 + 2 * (k : ℝ) * Real.pi, 3 * Real.pi/2 + 2 * (k : ℝ) * Real.pi] →
  x ∈ [Real.pi / 6 + (k : ℝ) * Real.pi, 2 * Real.pi / 3 + (k : ℝ) * Real.pi] := sorry

theorem min_value_g (x : ℝ) :
  x ∈ [- Real.pi / 3, Real.pi / 3] →
  ∃ x₀, g x₀ 1 = -1/2 ∧ x₀ = - Real.pi / 3 := sorry

end interval_monotonic_decrease_min_value_g_l272_272733


namespace quadrilateral_AD_length_l272_272665

-- Given conditions for the quadrilateral
variables (B O D A C : ℝ)
variable h_BO : dist B O = 6
variable h_OD : dist O D = 8
variable h_AO : dist A O = 10
variable h_OC : dist O C = 4
variable h_AB : dist A B = 7

-- Definition of the quadrilateral
def quadrilateral_ABCD : Prop :=
  ∃ (A B C D : Point) (O : Point), 
    B ≠ O ∧ O ≠ D ∧ A ≠ O ∧ O ≠ C ∧ A ≠ B ∧
    dist B O = 6 ∧
    dist O D = 8 ∧
    dist A O = 10 ∧
    dist O C = 4 ∧
    dist A B = 7

-- The statement to be proven
theorem quadrilateral_AD_length 
  (h : quadrilateral_ABCD) : 
  ∃ AD : ℝ, AD = sqrt 222 :=
sorry

end quadrilateral_AD_length_l272_272665


namespace ticket_price_increase_l272_272545

-- Define the initial price and the new price
def last_year_price : ℝ := 85
def this_year_price : ℝ := 102

-- Define the percent increase calculation
def percent_increase (initial : ℝ) (new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

-- Statement to prove
theorem ticket_price_increase (initial : ℝ) (new : ℝ) (h_initial : initial = last_year_price) (h_new : new = this_year_price) :
  percent_increase initial new = 20 :=
by
  sorry

end ticket_price_increase_l272_272545


namespace well_diameter_l272_272868

theorem well_diameter (V h : ℝ) (pi : ℝ) (r : ℝ) :
  h = 8 ∧ V = 25.132741228718345 ∧ pi = 3.141592653589793 ∧ V = pi * r^2 * h → 2 * r = 2 :=
by
  sorry

end well_diameter_l272_272868


namespace counting_numbers_dividing_56_greater_than_2_l272_272641

theorem counting_numbers_dividing_56_greater_than_2 :
  (∃ (A : Finset ℕ), A = {n ∈ (Finset.range 57) | n > 2 ∧ 56 % n = 0} ∧ A.card = 5) :=
sorry

end counting_numbers_dividing_56_greater_than_2_l272_272641


namespace smallest_n_with_2020_divisors_l272_272124

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272124


namespace max_value_f_l272_272183

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * (4 : ℝ) * x + 2

theorem max_value_f :
  ∃ x : ℝ, -f x = -18 ∧ (∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end max_value_f_l272_272183


namespace largest_m_factorial_l272_272588

theorem largest_m_factorial (m : ℕ) (h : ∃ n : ℕ, m! * 2022! = n!) : m = ↑(2022!) - 1 := by
  sorry

end largest_m_factorial_l272_272588


namespace exists_subset_fourth_power_l272_272286

theorem exists_subset_fourth_power (M : Finset ℕ) (hM : M.card = 1985)
  (h : ∀ n ∈ M, ∀ p ∈ (Nat.factors n), p ≤ 26) :
  ∃ a b c d ∈ M, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ, a * b * c * d = k ^ 4 := 
sorry

end exists_subset_fourth_power_l272_272286


namespace speed_ratio_l272_272270

variables {D T : ℝ}
variables (Distance_1 Time_1 Distance_2 Time_2 : ℝ)
variables (v1 v2 : ℝ)

-- Define conditions
def condition1 : Prop := Distance_1 = 0.1 * D
def condition2 : Prop := Time_1 = 0.2 * T
def condition3 : Prop := Distance_2 = 0.9 * D
def condition4 : Prop := Time_2 = 0.8 * T
def speed1 : Prop := v1 = Distance_1 / Time_1
def speed2 : Prop := v2 = Distance_2 / Time_2

-- State the Lean theorem
theorem speed_ratio : 
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ speed1 ∧ speed2 → v1 / v2 = 4 / 9 :=
by
  sorry

end speed_ratio_l272_272270


namespace real_solutions_exist_l272_272589

noncomputable
def valid_params (a b : ℝ) : Prop :=
  (a ≥ 0 ∧ -a ≤ b ∧ b ≤ 0) ∨ (a ≤ 0 ∧ -a/9 < b ∧ b ≤ 0)

theorem real_solutions_exist (a b x : ℝ) :
  (√(2 * a + b + 2 * x) + √(10 * a + 9 * b - 6 * x) = 2 * √(2 * a + b - 2 * x)) →
  ((x = √(a * (a + b)) ∨ x = -√(a * (a + b))) → valid_params a b) :=
by
  sorry

end real_solutions_exist_l272_272589


namespace regular_polygon_sides_l272_272212

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l272_272212


namespace count_positive_integers_satisfying_condition_l272_272575

-- Definitions
def is_between (x: ℕ) : Prop := 30 < x^2 + 8 * x + 16 ∧ x^2 + 8 * x + 16 < 60

-- Theorem statement
theorem count_positive_integers_satisfying_condition :
  {x : ℕ | is_between x}.card = 2 := 
sorry

end count_positive_integers_satisfying_condition_l272_272575


namespace largest_possible_s_l272_272731

-- Define the conditions
def is_regular_polygon (n : ℕ) : Prop :=
  n ≥ 3

def interior_angle (n : ℕ) : ℚ :=
  if is_regular_polygon n then (n - 2) * 180 / n else 0

-- Define the problem statement
theorem largest_possible_s (s : ℕ) (r : ℕ) 
  (h1 : is_regular_polygon r)
  (h2 : is_regular_polygon s)
  (h3 : r ≥ s)
  (h4 : interior_angle r = (61 / 60) * interior_angle s) :
  s = 121 :=
sorry

end largest_possible_s_l272_272731


namespace second_set_length_is_20_l272_272265

-- Define the lengths
def length_first_set : ℕ := 4
def length_second_set : ℕ := 5 * length_first_set

-- Formal proof statement
theorem second_set_length_is_20 : length_second_set = 20 :=
by
  sorry

end second_set_length_is_20_l272_272265


namespace third_quadrant_probability_l272_272329

theorem third_quadrant_probability :
  ∀ (a b : ℚ),
  (a ∈ ({1 / 3, 1 / 2, 2, 3} : Finset ℚ)) →
  (b ∈ ({-1, 1, -2, 2} : Finset ℚ)) →
  (finset.filter (λ ab : ℚ × ℚ, (ab.1 > 1 ∧ ab.2 < 0)) 
    (({1 / 3, 1 / 2, 2, 3} : Finset ℚ).product ({-1, 1, -2, 2} : Finset ℚ))).card 
  = 6 → (16) → ((6: ℚ)/(16: ℚ) = (3: ℚ)/(8: ℚ)) := sorry

end third_quadrant_probability_l272_272329


namespace solution_set_ineq_l272_272149

theorem solution_set_ineq (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ (x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1)) :=
sorry

end solution_set_ineq_l272_272149


namespace measure_angle_EHG_l272_272242

theorem measure_angle_EHG (EFGH : Parallelogram) (x : ℝ)
  (h1 : EFGH.ang_EFG = 2 * EFGH.ang_FGH)
  (h2 : EFGH.ang_EFG + EFGH.ang_FGH = 180) :
  EFGH.ang_EHG = 60 :=
by
  sorry

end measure_angle_EHG_l272_272242


namespace Peter_vacation_l272_272312

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l272_272312


namespace sum_infinite_series_l272_272925

theorem sum_infinite_series : 
  ∑' n : ℕ, (3 * (n + 1) - 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3)) = 73 / 12 := 
by sorry

end sum_infinite_series_l272_272925


namespace technician_completed_percentage_round_trip_l272_272047

-- Definitions based on the conditions
def task_time_first_stop : ℕ := 30
def task_time_second_stop : ℕ := 45
def task_time_third_stop : ℕ := 15
def task_time_service_center : ℕ := 60

def total_task_time : ℕ := task_time_first_stop + task_time_second_stop + task_time_third_stop + task_time_service_center

def time_spent_first_stop : ℕ := 0.5 * task_time_first_stop
def time_spent_second_stop : ℕ := 0.8 * task_time_second_stop
def time_spent_third_stop : ℕ := 0.6 * task_time_third_stop
def time_spent_service_center : ℕ := task_time_service_center

def total_time_spent_on_tasks : ℕ := time_spent_first_stop + time_spent_second_stop + time_spent_third_stop + time_spent_service_center

-- Theorem to prove
theorem technician_completed_percentage_round_trip :
  (total_time_spent_on_tasks / total_task_time * 100) = 80 :=
by
  -- Proof will be added here
  sorry

end technician_completed_percentage_round_trip_l272_272047


namespace test_mode_l272_272792

def scores : List ℕ := [61, 61, 61, 72, 75, 83, 85, 85, 85, 87, 87, 90, 92, 92, 94, 96, 96, 101, 101, 101, 103, 110, 110]

def mode (l : List ℕ) : ℕ :=
  l.group_by id
  |> List.map (λ g => (g.head, g.length))
  |> List.max_by (λ x => x.snd)
  |> Prod.fst

theorem test_mode : mode scores = 85 := 
  sorry

end test_mode_l272_272792


namespace count_positive_integers_count_of_positive_integers_l272_272578

theorem count_positive_integers (n : ℕ) :
  (150 ≤ n^2 ∧ n^2 ≤ 300) → n ∈ {13, 14, 15, 16, 17} :=
begin
  sorry
end

theorem count_of_positive_integers :
  {n : ℕ | 150 ≤ n^2 ∧ n^2 ≤ 300}.card = 5 :=
begin
  sorry
end

end count_positive_integers_count_of_positive_integers_l272_272578


namespace octal_subtraction_conversion_base4_l272_272950

noncomputable def octal_subtract_and_convert_to_base4 (a b : ℕ) : ℕ := 
  let result := a - b -- Subtraction in base 8, interpreted as natural numbers
  result -- Convert result to base 4, assumed predefined function exists (e.g., for educational purpose)

theorem octal_subtraction_conversion_base4 (a b expected_result_in_base4 : ℕ)
  (ha : a = 6 * 8^2 + 4 * 8 + 3)
  (hb : b = 2 * 8^2 + 5 * 8 + 7)
  (h_exp : expected_result_in_base4 = 3 * 4^4 + 3 * 4^3 + 1 * 4^1 + 0 * 4^0) :
  octal_subtract_and_convert_to_base4 a b = expected_result_in_base4 := 
by {
  have hresult : a - b = 3 * 8^2 + 6 * 8 + 4, {
    -- Expected result computed
    sorry
  },
  -- Convert result to decimal and then base 4
  have hconvert : convert_to_base4 (a - b) = expected_result_in_base4, {
    -- expected conversion steps
    sorry
  },
  exact Eq.trans hresult hconvert
}

end octal_subtraction_conversion_base4_l272_272950


namespace tangent_perpendicular_line_l272_272108

/-- Given a line 2x - 6y + 1 = 0 and a curve f(x) = x^3 + 3x^2 - 1,
    prove that the equation of the line that is perpendicular to the
    given line and tangent to the curve is 3x + y + 2 = 0. -/
theorem tangent_perpendicular_line :
  ∃ m : ℝ, ∃ n : ℝ,
    (∀ x y : ℝ, 2 * x - 6 * y + 1 = 0 → (y = -3 * x + m)) ∧
    (∀ x : ℝ, f(x) = x^3 + 3 * x^2 - 1) ∧
    (∀ n : ℝ, 3 * n^2 + 6 * n = -3) ∧
    (tangent_point = (n, f(n))) ∧
    (equation := y = -3 * n + m) ∧
    (m = -2) →
    (3 * x + y + 2 = 0) :=
by
  sorry

end tangent_perpendicular_line_l272_272108


namespace measure_of_angle_ehg_l272_272243

open Real

noncomputable def measure_angle_of_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : ℝ :=
  if h : FGH = 60 then 120 else sorry

theorem measure_of_angle_ehg (EFGH : Type) [Parallelogram EFGH] (EFG FGH : ℝ) (h1 : EFG = 2 * FGH) : 
  ∃ (EHG : ℝ), EHG = 120 :=
begin
  use 120,
  sorry
end

end measure_of_angle_ehg_l272_272243


namespace expression_max_value_l272_272954

open Real

theorem expression_max_value (x : ℝ) : ∃ M, M = 1/7 ∧ (∀ y : ℝ, y = x -> (y^3) / (y^6 + y^4 + y^3 - 3*y^2 + 9) ≤ M) :=
sorry

end expression_max_value_l272_272954


namespace large_kangaroos_count_l272_272235

theorem large_kangaroos_count (total_kangaroos : ℕ) (empty_pouches : ℕ) (pouch_capacity : ℕ) (total_kangaroos = 100) (empty_pouches = 77) (pouch_capacity = 3) : 
  ∃ large_kangaroos : ℕ, large_kangaroos = 31 := 
by 
  let full_pouches := total_kangaroos - empty_pouches 
  let small_kangaroos_in_pouches := pouch_capacity * full_pouches 
  let large_kangaroos := total_kangaroos - small_kangaroos_in_pouches 
  use large_kangaroos 
  have eq_full_pouches : full_pouches = 23 
  have eq_small_kangaroos_in_pouches : small_kangaroos_in_pouches = 69 
  have eq_large_kangaroos : large_kangaroos = 31 
  simp [eq_full_pouches, eq_small_kangaroos_in_pouches, eq_large_kangaroos]
  trivial

end large_kangaroos_count_l272_272235


namespace total_number_of_bills_l272_272816

theorem total_number_of_bills (total_money : ℕ) (fraction_for_50_bills : ℚ) (fifty_bill_value : ℕ) (hundred_bill_value : ℕ) :
  total_money = 1000 →
  fraction_for_50_bills = 3 / 10 →
  fifty_bill_value = 50 →
  hundred_bill_value = 100 →
  let money_for_50_bills := total_money * fraction_for_50_bills in
  let num_50_bills := money_for_50_bills / fifty_bill_value in
  let rest_money := total_money - money_for_50_bills in
  let num_100_bills := rest_money / hundred_bill_value in
  num_50_bills + num_100_bills = 13 :=
by
  intros h1 h2 h3 h4
  let money_for_50_bills := 1000 * (3 / 10)
  have h5 : money_for_50_bills = 300 := by sorry
  have h6 : 300 / 50 = 6 := by sorry
  let rest_money := 1000 - 300
  have h7 : rest_money = 700 := by sorry
  have h8 : 700 / 100 = 7 := by sorry
  have total_bills := 6 + 7
  show total_bills = 13 from eq.refl 13

end total_number_of_bills_l272_272816


namespace rate_of_interest_is_20_l272_272446

-- Definitions of the given conditions
def principal := 400
def simple_interest := 160
def time := 2

-- Definition of the rate of interest based on the given formula
def rate_of_interest (P SI T : ℕ) : ℕ := (SI * 100) / (P * T)

-- Theorem stating that the rate of interest is 20% given the conditions
theorem rate_of_interest_is_20 :
  rate_of_interest principal simple_interest time = 20 := by
  sorry

end rate_of_interest_is_20_l272_272446


namespace range_of_a_l272_272218

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a ∈ Iio (-2) ∪ Ioi 2) :=
by
  sorry

end range_of_a_l272_272218


namespace number_of_assignment_schemes_l272_272401

-- We assume roles and members as natural numbers for simplicity
def valid_roles := {p : ℕ | p ≥ 1 ∧ p ≤ 5}
def valid_members := {m : ℕ | m ≥ 1 ∧ m ≤ 5}

-- Given conditions
def constraints (A B : ℕ) (roles : ℕ → ℕ) : Prop :=
  roles B = 2 ∧
  (roles A ≠ 1 ∧ roles A ≠ 2) ∧
  ∀(x : ℕ), x ∈ valid_members → roles x ∈ valid_roles

theorem number_of_assignment_schemes : 
  ∀ (roles : ℕ → ℕ),
  (∃ A B, A ≠ B ∧ constraints A B roles) → 
  (∑ x in valid_members, roles x) = 18 :=
by
  intro roles
  intro h
  sorry

end number_of_assignment_schemes_l272_272401


namespace triangle_area_eq_l272_272855

/--
Given:
1. The base of the triangle is 4 meters.
2. The height of the triangle is 5 meters.

Prove:
The area of the triangle is 10 square meters.
-/
theorem triangle_area_eq (base height : ℝ) (h_base : base = 4) (h_height : height = 5) : 
  (base * height / 2) = 10 := by
  sorry

end triangle_area_eq_l272_272855


namespace find_AD_l272_272713

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272713


namespace trapezoid_AD_CF_BC_CE_l272_272700

theorem trapezoid_AD_CF_BC_CE (A B C D E F : Point) (x y : ℝ)
  (h: AC = 1)
  (h1: AD = CF)
  (h2: BC = CE)
  (h3: Perpendicular AE CD)
  (h4: Perpendicular CF AB)
  (h5: AC_perpendicular_height : height_of_trapezoid AC = 1) :
  AD = √(√2 - 1) :=
sorry

end trapezoid_AD_CF_BC_CE_l272_272700


namespace correct_operation_l272_272839

theorem correct_operation :
  (∀ a b : ℝ, (sqrt 3 + sqrt 2 ≠ sqrt 5) ∧ 
              ((a + b) ^ 2 ≠ a^2 + b^2) ∧ 
              ((-2 * a * b^2) ^ 3 = -8 * a^3 * b^6) ∧ 
              ((-1 / 2) ^ (-2 : ℤ) ≠ -(1 / 4))) :=
by {
  intros a b,
  split,
  -- proof for sqrt 3 + sqrt 2 ≠ sqrt 5
  sorry,
  
  split,
  -- proof for (a + b) ^ 2 ≠ a^2 + b^2
  sorry,
  
  split,
  -- proof for (-2 * a * b^2) ^ 3 = -8 * a^3 * b^6
  sorry,
  
  -- proof for (-1 / 2) ^ (-2 : ℤ) ≠ -(1 / 4)
  sorry
}

end correct_operation_l272_272839


namespace y_intercept_of_line_l272_272100

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l272_272100


namespace raised_bed_area_correct_l272_272061

def garden_length : ℝ := 220
def garden_width : ℝ := 120
def garden_area : ℝ := garden_length * garden_width
def tilled_land_area : ℝ := garden_area / 2
def remaining_area : ℝ := garden_area - tilled_land_area
def trellis_area : ℝ := remaining_area / 3
def raised_bed_area : ℝ := remaining_area - trellis_area

theorem raised_bed_area_correct : raised_bed_area = 8800 := by
  sorry

end raised_bed_area_correct_l272_272061


namespace sum_first_20_terms_arithmetic_seq_l272_272970

theorem sum_first_20_terms_arithmetic_seq :
  ∃ (a d : ℤ) (S_20 : ℤ), d > 0 ∧
  (a + 2 * d) * (a + 6 * d) = -12 ∧
  (a + 3 * d) + (a + 5 * d) = -4 ∧
  S_20 = 20 * a + (20 * 19 / 2) * d ∧
  S_20 = 180 :=
by
  sorry

end sum_first_20_terms_arithmetic_seq_l272_272970


namespace find_x_l272_272636

def vector (T : Type) := T × T

def a : vector ℝ := (1, 2)
def b (x : ℝ) : vector ℝ := (x, -2)

def vector_add (v1 v2 : vector ℝ) : vector ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : vector ℝ) : vector ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) : 
  dot_product (vector_add a (b x)) (vector_sub a (b x)) = 0 ↔ (x = 1 ∨ x = -1) :=
begin
  sorry
end

end find_x_l272_272636


namespace final_distance_back_to_start_l272_272078

theorem final_distance_back_to_start :
  let north := (0:ℝ, 2:ℝ),
      northeast := (2 * (Real.sqrt 2) / 2, 2 * (Real.sqrt 2) / 2),
      southeast := (3 * (Real.sqrt 2) / 2, -(3 * (Real.sqrt 2) / 2)),
      west := (-2:ℝ, 0:ℝ)
  in
  let final_displacement_x := north.1 + northeast.1 + southeast.1 + west.1,
      final_displacement_y := north.2 + northeast.2 + southeast.2 + west.2
  in
  Real.sqrt (final_displacement_x^2 + final_displacement_y^2) = Real.sqrt ((5 * Real.sqrt 2 / 2 - 2)^2 + (2 - Real.sqrt 2 / 2)^2) :=
by
  let north := (0:ℝ, 2:ℝ)
  let northeast := (2 * (Real.sqrt 2) / 2, 2 * (Real.sqrt 2) / 2)
  let southeast := (3 * (Real.sqrt 2) / 2, -(3 * (Real.sqrt 2) / 2))
  let west := (-2:ℝ, 0:ℝ)
  let final_displacement_x := north.1 + northeast.1 + southeast.1 + west.1
  let final_displacement_y := north.2 + northeast.2 + southeast.2 + west.2
  have hx : final_displacement_x = (5 * Real.sqrt 2 / 2 - 2) := sorry
  have hy : final_displacement_y = (2 - Real.sqrt 2 / 2) := sorry
  rw [hx, hy]
  apply congr_arg
  sorry

end final_distance_back_to_start_l272_272078


namespace problem_a_not_tilable_problem_b_tilable_l272_272009

def square : Type := (ℕ × ℕ)

-- Definitions for Problem (a)
def problem_a_board : List square := [(3, 5), (7, 2)]
def problem_a_remaining_black_cells (squares : List square) : ℕ :=
  squares.filter (λ cell, (cell.fst + cell.snd) % 2 = 0).length - 2

def problem_a_remaining_white_cells (squares : List square) : ℕ :=
  squares.filter (λ cell, (cell.fst + cell.snd) % 2 = 1).length

-- Definitions for Problem (b)
def problem_b_board : List square := [(3, 6), (7, 2)]
def problem_b_remaining_black_cells (squares : List square) : ℕ :=
  squares.filter (λ cell, (cell.fst + cell.snd) % 2 = 0).length - 1

def problem_b_remaining_white_cells (squares : List square) : ℕ :=
  squares.filter (λ cell, (cell.fst + cell.snd) % 2 = 1).length - 1

-- The proofs for the problems start here.
theorem problem_a_not_tilable :
  problem_a_remaining_black_cells [(i, j) | i <- List.range 8, j <- List.range 8] ≠
  problem_a_remaining_white_cells [(i, j) | i <- List.range 8, j <- List.range 8] := 
sorry

theorem problem_b_tilable :
  problem_b_remaining_black_cells [(i, j) | i <- List.range 8, j <- List.range 8] =
  problem_b_remaining_white_cells [(i, j) | i <- List.range 8, j <- List.range 8] := 
sorry

end problem_a_not_tilable_problem_b_tilable_l272_272009


namespace avg_mpg_is_15_point_2_l272_272060

-- Definitions from conditions
def initial_odometer := 35200
def gallons_at_start := 10
def additional_gallons_1 := 15
def odometer_after_first_fill := 35480
def end_odometer := 35960
def gallons_at_end := 25

-- Definition of total distance traveled
def total_distance_traveled := end_odometer - initial_odometer

-- Definition of total gasoline used
def total_gasoline_used := gallons_at_start + additional_gallons_1 + gallons_at_end

-- Proof that the average miles-per-gallon equals 15.2
theorem avg_mpg_is_15_point_2 : total_distance_traveled / total_gasoline_used = 15.2 :=
by
  sorry

end avg_mpg_is_15_point_2_l272_272060


namespace foldable_topless_cubical_box_count_l272_272600

def isFoldable (placement : Char) : Bool :=
  placement = 'C' ∨ placement = 'E' ∨ placement = 'G'

theorem foldable_topless_cubical_box_count :
  (List.filter isFoldable ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']).length = 3 :=
by
  sorry

end foldable_topless_cubical_box_count_l272_272600


namespace inscribed_square_area_l272_272489

noncomputable def area_of_inscribed_square : ℝ :=
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let t := (sqrt (8 / 3))
  let square_side := 2 * t
  let square_area := (square_side^2)
  square_area

theorem inscribed_square_area :
  let ellipse_eq := (λ x y : ℝ, (x^2) / 4 + (y^2) / 8 = 1)
  let square_area := (2 * sqrt (8 / 3))^2
  square_area = 32 / 3 :=
by
  sorry

end inscribed_square_area_l272_272489


namespace range_of_a_l272_272964

noncomputable def f (a : ℝ) (x : ℝ) := - (1 / 3) * (Real.cos (2 * x)) - a * (Real.sin x - Real.cos x)

theorem range_of_a {a : ℝ} :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 1) ↔ a ∈ set.Icc (-(Real.sqrt 2 / 6)) (Real.sqrt 2 / 6) :=
begin
  sorry
end

end range_of_a_l272_272964


namespace cone_water_volume_percentage_l272_272033

theorem cone_water_volume_percentage
  (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  abs (percentage - 29.6296) < 0.0001 :=
by
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  sorry

end cone_water_volume_percentage_l272_272033


namespace find_AD_l272_272680

-- Definitions inferred from the problem conditions
def is_trapezoid (ABCD : Type) (A B C D : ABCD) : Prop := sorry
def is_diagonal_equal_height (A C : ABCD) (AC : ℝ) : Prop := AC = 1
def perpendiculars_drawn (A C E F : ABCD) (AE CF : ℝ) : Prop := sorry
def equal_sides (AD CF : ℝ) : Prop := AD = CF
def equal_sides_2 (BC CE : ℝ) : Prop := BC = CE

-- Problem statement in Lean 4
theorem find_AD (ABCD : Type) [is_trapezoid ABCD] (A B C D E F : ABCD) (AC AD CF BC CE AE : ℝ)
  [is_diagonal_equal_height A C AC] 
  [perpendiculars_drawn A C E F AE CF] 
  [equal_sides AD CF] 
  [equal_sides_2 BC CE] : 
  AD = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_AD_l272_272680


namespace angle_FCE_eq_angle_ADE_angle_FEC_eq_angle_BDC_l272_272290

variables {Point : Type} [MetricSpace Point]
variables (A B C D E F : Point)

-- Definitions of conditions
def CD_eq_DE (CD DE : ℝ) : Prop := CD = DE
def angles_90 (BC D EA : Point) : Prop := angle B C D = 90 ∧ angle D E A = 90
def ratio_AF_AE_BF_BC (A E F B C : Point) (AE BF BC: ℝ) : Prop := AF / AE = BF / BC

-- Prove the equal angles
theorem angle_FCE_eq_angle_ADE 
  (h1 : CD_eq_DE CD DE)
  (h2 : angles_90 B C D E A)
  (h3 : ratio_AF_AE_BF_BC A E F B C AE BF BC)
  : angle F C E = angle A D E :=
sorry

theorem angle_FEC_eq_angle_BDC 
  (h1 : CD_eq_DE CD DE)
  (h2 : angles_90 B C D E A)
  (h3 : ratio_AF_AE_BF_BC A E F B C AE BF BC)
  : angle F E C = angle B D C :=
sorry

end angle_FCE_eq_angle_ADE_angle_FEC_eq_angle_BDC_l272_272290


namespace natalie_needs_12_bushes_for_60_zucchinis_l272_272090

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end natalie_needs_12_bushes_for_60_zucchinis_l272_272090


namespace loe_speed_l272_272368

-- Define the conditions as given in the problem
def teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_in_hours : ℝ := 1.5 -- 90 minutes in hours
def distance_ahead : ℝ := 15

-- Define the problem equivalence in Lean
theorem loe_speed :
  let L := (teena_speed * time_in_hours + initial_distance_behind - distance_ahead) / time_in_hours in
  L = 50 :=
by
  -- Placeholder for the proof
  sorry

end loe_speed_l272_272368


namespace spanish_teams_in_final_probability_l272_272372

noncomputable def probability_of_spanish_teams_in_final : ℚ :=
  let teams := 16
  let spanish_teams := 3
  let non_spanish_teams := teams - spanish_teams
  -- Probability calculation based on given conditions and solution steps
  1 - 7 / 15 * 6 / 14

theorem spanish_teams_in_final_probability :
  probability_of_spanish_teams_in_final = 4 / 5 :=
sorry

end spanish_teams_in_final_probability_l272_272372


namespace tan_15_expression_equals_one_l272_272083

theorem tan_15_expression_equals_one :
  (∃ (ϴ : ℝ), ϴ = 15 * real.pi / 180) → 
  (∃ (φ : ℝ), φ = 60 * real.pi / 180) → 
  (∃ (ψ : ℝ), ψ = 45 * real.pi / 180) → 
  ((sqrt 3 * real.tan (ϴ) + 1) / (sqrt 3 - real.tan (ϴ)) = 1) :=
by
  intros hϴ hφ hψ
  sorry

end tan_15_expression_equals_one_l272_272083


namespace incenter_of_transformed_triangle_l272_272272

variables {α : Type*} [euclidean_space α] 

noncomputable def orthocenter (A B C : α) : α := sorry

noncomputable def circumcenter (A B C : α) : α := sorry

noncomputable def circumradius (A B C : α) : ℝ := sorry

noncomputable def is_incenter (O : α) (A' B' C' : α) : Prop := sorry

theorem incenter_of_transformed_triangle 
  (A B C A' B' C' : α)
  (H : α := orthocenter A B C)
  (O : α := circumcenter A B C) 
  (R : ℝ := circumradius A B C)
  (h1 : dist A H * dist A A' = R ^ 2)
  (h2 : dist B H * dist B B' = R ^ 2)
  (h3 : dist C H * dist C C' = R ^ 2) :
  is_incenter O A' B' C' :=
sorry

end incenter_of_transformed_triangle_l272_272272


namespace measure_angle_EHG_l272_272246

-- Given problem conditions
variable (EFGH : Type) [Parallelogram EFGH]
variable (angleEFG angleFGH : ℝ)
variable (H1 : angleEFG = 2 * angleFGH)
variable (H2 : angleEFG + angleFGH = 180)

-- Conclude the measure of angle EHG
theorem measure_angle_EHG : angleEFG = 120 :=
by
  -- Proof omitted
  sorry

end measure_angle_EHG_l272_272246


namespace angle_at_450_is_155_degrees_l272_272829

noncomputable def acute_angle_at_time (h m : ℕ) : ℝ :=
  let hour_position := (h % 12) * 30 + m * 0.5
  let minute_position := m * 6
  let angle := abs (minute_position - hour_position)
  if angle > 180 then 360 - angle else angle

theorem angle_at_450_is_155_degrees :
  acute_angle_at_time 4 50 = 155 :=
sorry

end angle_at_450_is_155_degrees_l272_272829


namespace triangle_AOM_side_lengths_l272_272322

open Real

/-- A Lean 4 definition that computes the side lengths
    of triangle AOM in a square. -/
def sides_of_triangle_AOM (A O M : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let AO := sqrt ((O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2)
  let AM := sqrt ((M.1 - A.1) ^ 2 + (M.2 - A.2) ^ 2)
  let OM := sqrt ((M.1 - O.1) ^ 2 + (M.2 - O.2) ^ 2)
  (AO, AM, OM)

/-- The side lengths of triangle AOM -/
theorem triangle_AOM_side_lengths :
  let A := (0, 0)
  let O := (3, 3)
  let M := (4, 6)
  sides_of_triangle_AOM A O M = (3 * sqrt 2, 2 * sqrt 13, sqrt 10) :=
by
  -- Placeholder for proof
  sorry

end triangle_AOM_side_lengths_l272_272322


namespace sufficient_not_necessary_l272_272453

theorem sufficient_not_necessary (a : ℝ) :
  a > 1 → (a^2 > 1) ∧ (∀ a : ℝ, a^2 > 1 → a = -1 ∨ a > 1 → false) :=
by {
  sorry
}

end sufficient_not_necessary_l272_272453


namespace count_positive_integers_l272_272581

theorem count_positive_integers (count : ℕ) :
  count = (List.filter (λ x : ℕ, 150 ≤ x^2 ∧ x^2 ≤ 300) (List.range 19)).length := by
  sorry

end count_positive_integers_l272_272581


namespace find_room_dimension_l272_272378

noncomputable def unknown_dimension_of_room 
  (cost_per_sq_ft : ℕ)
  (total_cost : ℕ)
  (w : ℕ)
  (l : ℕ)
  (h : ℕ)
  (door_h : ℕ)
  (door_w : ℕ)
  (window_h : ℕ)
  (window_w : ℕ)
  (num_windows : ℕ) : ℕ := sorry

theorem find_room_dimension :
  unknown_dimension_of_room 10 9060 25 15 12 6 3 4 3 3 = 25 :=
sorry

end find_room_dimension_l272_272378


namespace game_probability_l272_272639

def game_state := ℕ → ℕ → ℕ -- Representation for states (Hawkins, Dustin, Lucas)

-- Initial condition: Each player starts with $2
def initial_state : game_state := λ h d, 2

-- Condition: Bell rings every 10 seconds for 2021 times
def bell_rings : ℕ := 2021

-- Condition: Probability that a player keeps their $1 when they only have $1 left
def keep_probability : ℚ := 1 / 3

-- Question: Probability that each player will still have $1 after the bell has rung 2021 times
theorem game_probability :
  ∀ (s : game_state),
  s 1 1 1 →  -- Final state condition that each player has $1
  s bell_rings = initial_state →
  prob_final_state = 1/4 :=  -- Probability that final state where each player has $1
sorry

end game_probability_l272_272639


namespace alice_current_age_l272_272507

def alice_age_twice_eve (a b : Nat) : Prop := a = 2 * b

def eve_age_after_10_years (a b : Nat) : Prop := a = b + 10

theorem alice_current_age (a b : Nat) (h1 : alice_age_twice_eve a b) (h2 : eve_age_after_10_years a b) : a = 20 := by
  sorry

end alice_current_age_l272_272507


namespace overlap_per_connection_is_4_cm_l272_272436

-- Condition 1: There are 24 tape measures.
def number_of_tape_measures : Nat := 24

-- Condition 2: Each tape measure is 28 cm long.
def length_of_one_tape_measure : Nat := 28

-- Condition 3: The total length of all connected tape measures is 580 cm.
def total_length_with_overlaps : Nat := 580

-- The question to prove: The overlap per connection is 4 cm.
theorem overlap_per_connection_is_4_cm 
  (n : Nat) (length_one : Nat) (total_length : Nat) 
  (h_n : n = number_of_tape_measures)
  (h_length_one : length_one = length_of_one_tape_measure)
  (h_total_length : total_length = total_length_with_overlaps) :
  ((n * length_one - total_length) / (n - 1)) = 4 := 
by 
  sorry

end overlap_per_connection_is_4_cm_l272_272436


namespace montoya_family_budget_l272_272371

theorem montoya_family_budget :
  let groceries := 0.6
  let total_food := 0.8
  total_food - groceries = 0.2 :=
by
  let groceries := 0.6
  let total_food := 0.8
  show total_food - groceries = 0.2 from sorry

end montoya_family_budget_l272_272371


namespace Olivia_earnings_l272_272748

def Monday_hours : Nat := 4
def Wednesday_hours : Nat := 3
def Friday_hours : Nat := 6
def Saturday_hours : Nat := 5

def Monday_rate : Nat := 9
def Wednesday_rate : Nat := 9
def Friday_rate : Nat := 12
def Saturday_rate : Nat := 15

def expenses : Nat := 45
def tax_rate : Rat := 0.10

def earnings_before_expenses : Nat :=
  (Monday_hours * Monday_rate) +
  (Wednesday_hours * Wednesday_rate) +
  (Friday_hours * Friday_rate) +
  (Saturday_hours * Saturday_rate)

def earnings_after_expenses : Nat := earnings_before_expenses - expenses
def taxes : Nat := (tax_rate * (earnings_before_expenses : Rat)).natCeil
def net_earnings : Nat := earnings_after_expenses - taxes

theorem Olivia_earnings :
  net_earnings = 144 :=
by
  sorry

end Olivia_earnings_l272_272748


namespace y_intercept_of_line_l272_272098

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (0, 4) = (0, y) :=
by { intro h,
     have y_eq : y = 4,
     { 
       sorry
     },
     have : (0, y) = (0, 4),
     { 
       sorry 
     },
     exact this }

end y_intercept_of_line_l272_272098


namespace ticket_price_increase_l272_272547

-- Definitions as per the conditions
def old_price : ℝ := 85
def new_price : ℝ := 102
def percent_increase : ℝ := (new_price - old_price) / old_price * 100

-- Statement to prove the percent increase is 20%
theorem ticket_price_increase : percent_increase = 20 := by
  sorry

end ticket_price_increase_l272_272547


namespace quadrilateral_XYZV_is_square_l272_272672

theorem quadrilateral_XYZV_is_square
  ( A B C D P Q R S X Y Z V : Point )
  ( hSquare_ABCD : is_square A B C D)
  ( hAP_BQ_CR_DS : AP = BQ ∧ BQ = CR ∧ CR = DS )
  ( hX_on_AB : is_on_side X A B )
  ( hPX_int_BC_Y : is_intersection_point P X B C Y )
  ( hQY_int_CD_Z : is_intersection_point Q Y C D Z )
  ( hRZ_int_DA_V : is_intersection_point R Z D A V )
  ( hSV_int_AB_X' : is_intersection_point S V A B X' )
  ( hX_prime_eq_X : X = X' ) :
  is_square X Y Z V :=
by
  sorry

end quadrilateral_XYZV_is_square_l272_272672


namespace transform_large_number_l272_272177

def can_transform_to_less_than_10_squared (n : ℕ) (digits : Finset ℕ) : Prop :=
  digits ⊆ {4, 5, 6, 7, 8, 9} ∧ n < 10^2

theorem transform_large_number (number : ℕ) 
  (h_digits : ∀ d ∈ (Finset.range 2019).bind (λ i, { (number / 10^i % 10) }), d ∈ {4, 5, 6, 7, 8, 9}) 
  (h_len : 10^(2019-1) ≤ number ∧ number < 10^2019) :
  ∃ m < 10 ^ 2, can_transform_to_less_than_10_squared m {d | ∃ i < 2019, d = number / 10^i % 10} :=
sorry

end transform_large_number_l272_272177


namespace general_term_formula_l272_272392

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0
  else 3 * sequence (n - 1) + real.sqrt (8 * (sequence (n - 1))^2 + 1)

theorem general_term_formula (n : ℕ) :
  sequence n = (real.sqrt 2 / 8) * (3 + 2 * real.sqrt 2)^n -
               (real.sqrt 2 / 8) * (3 - 2 * real.sqrt 2)^n :=
sorry

end general_term_formula_l272_272392


namespace profit_percentage_l272_272880

def CP : ℝ := 600
def SP : ℝ := 648
def Profit := SP - CP
def Percentage_Profit := (Profit / CP) * 100

theorem profit_percentage :
  Percentage_Profit = 8 := by
  sorry

end profit_percentage_l272_272880


namespace average_after_modifications_l272_272375

theorem average_after_modifications (S : ℕ) (sum_initial : S = 1080)
  (sum_after_removals : S - 80 - 85 = 915)
  (sum_after_additions : 915 + 75 + 75 = 1065) :
  (1065 / 12 : ℚ) = 88.75 :=
by sorry

end average_after_modifications_l272_272375


namespace larger_root_exceeds_smaller_root_by_l272_272853

theorem larger_root_exceeds_smaller_root_by (A B C : ℝ) (hA : A = 2) (hB : B = 5) (hC : C = -12) :
  let discriminant := B^2 - 4 * A * C,
      root1 := (-B + real.sqrt discriminant) / (2 * A),
      root2 := (-B - real.sqrt discriminant) / (2 * A)
  in (max root1 root2) - (min root1 root2) = 5.5 :=
by
  have hDiscriminant : discriminant = 25 + 96 := sorry
  have hSqrtDiscriminant : real.sqrt discriminant = 11 := sorry
  have root1_value : root1 = 1.5 := sorry
  have root2_value : root2 = -4 := sorry
  have hMaxMinDifference : (max root1 root2) - (min root1 root2) = 5.5 := sorry
  exact hMaxMinDifference

end larger_root_exceeds_smaller_root_by_l272_272853


namespace length_of_room_l272_272786

noncomputable def room_length (width cost rate : ℝ) : ℝ :=
  let area := cost / rate
  area / width

theorem length_of_room :
  room_length 4.75 38475 900 = 9 := by
  sorry

end length_of_room_l272_272786


namespace solve_inequality_l272_272764

theorem solve_inequality (x : ℝ) (h : 0 < x) :
    27 * Real.sqrt (Real.log x / Real.log 3) 
    - 33 * Real.sqrt (4 * Real.log x / Real.log 3) 
    + 40 * x * Real.sqrt (ℝ.log 3 / Real.log x) 
    ≤ 48 
    → x ∈ Ioc 1 3 ∨ x = 4 ^ Real.log 4 / Real.log 3 := 
sorry

end solve_inequality_l272_272764


namespace measure_angle_EHG_l272_272241

theorem measure_angle_EHG (EFGH : Parallelogram) (x : ℝ)
  (h1 : EFGH.ang_EFG = 2 * EFGH.ang_FGH)
  (h2 : EFGH.ang_EFG + EFGH.ang_FGH = 180) :
  EFGH.ang_EHG = 60 :=
by
  sorry

end measure_angle_EHG_l272_272241


namespace GreenFunction_boundary_value_problem_l272_272942

theorem GreenFunction_boundary_value_problem 
  (k : ℝ)
  (y G: ℝ → ℝ)
  (x ξ : ℝ)
  (h1: ∀ (x : ℝ), y x = G x ξ)
  (h2 : 0 ≤ x ∧ x ≤ ξ ∨ ξ ≤ x ∧ x ≤ 1) :
  (y'' x + k^2 * y x = 0) ∧ (y 0 = 0) ∧ (y 1 = 0) ∧
  G(x, ξ) =  if x ≤ ξ then (sin (k * (ξ - 1)) * sin (k * x)) / (k * sin k)
                        else (sin (k * ξ) * sin (k * (x - 1))) / (k * sin k) :=
sorry

end GreenFunction_boundary_value_problem_l272_272942


namespace find_AD_l272_272706

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272706


namespace sum_of_powers_mod_7_l272_272012

theorem sum_of_powers_mod_7 :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7 = 1) := by
  sorry

end sum_of_powers_mod_7_l272_272012


namespace k_times_value_function_range_l272_272584

open Real

def is_k_times_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  (a ≠ b) ∧ (f a = k * a) ∧ (f b = k * b)

theorem k_times_value_function_range
  (f : ℝ → ℝ)
  (domain : Set ℝ)
  (h_f : ∀ x ∈ domain, f x = log x + x)
  (k : ℝ)
  (k_pos : k > 0)
  (h_monotone : ∀ x y ∈ domain, x < y → f x < f y)
  (a b : ℝ)
  (h_a : a ∈ domain)
  (h_b : b ∈ domain)
  (h_ab : a ≠ b) :
  is_k_times_value_function f k a b ↔ (1 < k ∧ k < 1 + 1 / exp 1) :=
sorry

end k_times_value_function_range_l272_272584


namespace smallest_number_has_2020_divisors_l272_272130

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272130


namespace probability_independent_conditional_eq_conditional_prob_eq_iff_independent_total_probability_l272_272634

theorem probability_independent_conditional_eq {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  (Probability.Independent P A B → P(B | A) = P(B)) :=
by sorry

theorem conditional_prob_eq_iff_independent {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  (P(B | A) = P(B) → P(A | B) = P(A)) :=
by sorry

theorem total_probability {Ω : Type*} {P : Probability.ProbabilitySpace Ω} 
  (A B : set Ω) (hA : P(A) > 0) (hB : P(B) > 0) : 
  P(A ∩ B) + P(Aᶜ ∩ B) = P(B) :=
by sorry

end probability_independent_conditional_eq_conditional_prob_eq_iff_independent_total_probability_l272_272634


namespace next_palindrome_after_2010_is_2020_product_of_digits_of_2020_is_0_product_of_digits_of_next_palindrome_after_2010_l272_272796

-- Define what it means for a number to be a palindrome.
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

-- Define the function to find the next palindrome year after a given year.
noncomputable def next_palindromic_year (year : ℕ) : ℕ :=
  Nat.find (λ y => y > year ∧ is_palindrome y)

-- The year 2010 is given as a palindrome.
example : is_palindrome 2010 := sorry

-- Prove that the next palindromic year after 2010 is 2020 and the product of its digits is 0
theorem next_palindrome_after_2010_is_2020 : next_palindromic_year 2010 = 2020 := sorry

theorem product_of_digits_of_2020_is_0 :
  (2 * 0 * 2 * 0 : ℕ) = 0 := by norm_num

theorem product_of_digits_of_next_palindrome_after_2010 :
  ∃ (n : ℕ), next_palindromic_year 2010 = n ∧ (n.toDigits 10).prod = 0 :=
begin
  use 2020,
  split,
  { exact next_palindrome_after_2010_is_2020, },
  { rw [←Finset.prod_eq_multiset_prod, Multiset.toDigits, Multiset.prod_cons, Multiset.prod_cons, Multiset.prod_cons, Multiset.prod_cons],
    exact product_of_digits_of_2020_is_0, },
end

end next_palindrome_after_2010_is_2020_product_of_digits_of_2020_is_0_product_of_digits_of_next_palindrome_after_2010_l272_272796


namespace angle_between_clock_hands_at_7_30_l272_272418

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l272_272418


namespace percent_of_workday_in_meetings_l272_272742

theorem percent_of_workday_in_meetings (h1 : 9 > 0) (m1 m2 : ℕ) (h2 : m1 = 45) (h3 : m2 = 2 * m1) : 
  (135 / 540 : ℚ) * 100 = 25 := 
by
  -- Just for structure, the proof should go here
  sorry

end percent_of_workday_in_meetings_l272_272742


namespace Ptolemys_theorem_l272_272328

theorem Ptolemys_theorem {A B C D : Point}
  (hCyclic : CyclicQuadrilateral A B C D) :
  length (diagonal A C) * length (diagonal B D) =
    length (side A B) * length (side C D) +
    length (side A D) * length (side B C) := 
sorry

end Ptolemys_theorem_l272_272328


namespace commission_selection_possible_l272_272092

-- Representation of the Problem
variable (G : Type) [Fintype G] [DirectedGraph G] [∀ v : G, ∃ u : G, v ⟶ u] [FiniteGraph G]

def parliament := { g : G // vertices G = 450 }

-- Definition for checking if it's possible to select a commission of 150 members
def can_select_commission_of_150 (G : parliament) : Prop :=
  ∃ (S : Finset G), S.card = 150 ∧ ∀ v₁ v₂ ∈ S, ¬ (v₁ ⟶ v₂ ∨ v₂ ⟶ v₁)

theorem commission_selection_possible (G : parliament) : can_select_commission_of_150 G :=
by {
  sorry
}

end commission_selection_possible_l272_272092


namespace y_intercept_l272_272105

theorem y_intercept (x y : ℝ) (h : 4 * x + 7 * y = 28) : x = 0 → y = 4 :=
by
  intro hx
  rw [hx, zero_mul, add_zero] at h
  have := eq_div_of_mul_eq (by norm_num : 7 ≠ 0) h
  rw [eq_comm, div_eq_iff (by norm_num : 7 ≠ 0), mul_comm] at this
  exact this

end y_intercept_l272_272105


namespace students_from_second_grade_l272_272389

theorem students_from_second_grade (r1 r2 r3 : ℕ) (total_students sample_size : ℕ) (h_ratio: r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ r1 + r2 + r3 = 10) (h_sample_size: sample_size = 50) : 
  (r2 * sample_size / (r1 + r2 + r3)) = 15 :=
by
  sorry

end students_from_second_grade_l272_272389


namespace inverse_functions_symmetric_l272_272619

def f (x : ℝ) : ℝ := 2^(-x)
def g (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem inverse_functions_symmetric {x y : ℝ} :
  (y = f x) ↔ (x = g y) :=
sorry

end inverse_functions_symmetric_l272_272619


namespace longest_line_segment_in_quarter_circle_l272_272038

theorem longest_line_segment_in_quarter_circle (diameter : ℝ) (h_diameter : diameter = 16) :
  let r := diameter / 2 in
  let l := r * real.sqrt 2 in
  l^2 = 128 :=
by
  sorry

end longest_line_segment_in_quarter_circle_l272_272038


namespace germs_per_dish_l272_272255

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l272_272255


namespace mateen_backyard_area_l272_272669

theorem mateen_backyard_area :
  (∀ (L : ℝ), 30 * L = 1200) →
  (∀ (P : ℝ), 12 * P = 1200) →
  (∃ (L W : ℝ), 2 * L + 2 * W = 100 ∧ L * W = 400) := by
  intros hL hP
  use 40
  use 10
  apply And.intro
  sorry
  sorry

end mateen_backyard_area_l272_272669


namespace soccer_ball_max_height_l272_272046

theorem soccer_ball_max_height :
  ∃ a : ℝ, (∀ t : ℝ, (h t = a * t^2 + 19.6 * t) ) ∧ (h 4 = 0) ∧ 
  (∀ h : ℝ, (t = 2 → h = -4.9 * t^2 + 19.6 * t) → h = 19.6) :=
sorry

end soccer_ball_max_height_l272_272046


namespace pattern_B_forms_pyramid_l272_272179

-- Definitions for the patterns
inductive Pattern where
  | A
  | B
  | C
  | D

open Pattern

-- The proposition to prove is that pattern B can be folded to form a pyramid with a square base.
theorem pattern_B_forms_pyramid : ∃ p : Pattern, p = B :=
by
  use B
  sorry

end pattern_B_forms_pyramid_l272_272179


namespace technicians_in_workshop_l272_272236

theorem technicians_in_workshop 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_tech : ℕ) 
  (avg_salary_rest : ℕ) 
  (total_salary : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (h1 : total_workers = 14) 
  (h2 : avg_salary_all = 8000) 
  (h3 : avg_salary_tech = 10000) 
  (h4 : avg_salary_rest = 6000) 
  (h5 : total_salary = total_workers * avg_salary_all) 
  (h6 : T + R = 14)
  (h7 : total_salary = 112000) 
  (h8 : total_salary = avg_salary_tech * T + avg_salary_rest * R) :
  T = 7 := 
by {
  -- Proof goes here
  sorry
} 

end technicians_in_workshop_l272_272236


namespace find_AD_l272_272714

universe u

variables (A B C D E F : Type u) [trapezoid ABCD]
variables (h1 : AC = 1) (h2 : height ABCD = AC)
variables (h3 : AD = CF) (h4 : BC = CE)
variables [perpendicular AE CD] [perpendicular CF AB]

theorem find_AD : AD = sqrt (sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272714


namespace smallest_number_has_2020_divisors_l272_272132

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272132


namespace solve_for_unknown_l272_272459

theorem solve_for_unknown :
  ∃ (x : ℝ), 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 :=
by
  use 2
  split
  -- Placeholder for the equality proof
  sorry
  -- Placeholder for the solution confirmation
  rfl

end solve_for_unknown_l272_272459


namespace probability_sum_of_four_selected_is_odd_l272_272094

def first_fifteen_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the problem condition in Lean
def probability_sum_odd_of_four_selected : ℚ := 
  let total_ways := Nat.choose 15 4
  let ways_with_2 := Nat.choose 14 3
  ways_with_2 / total_ways

-- Assign the simplified probability to the variable
def answer : ℚ := 4 / 15

-- The theorem we need to prove
theorem probability_sum_of_four_selected_is_odd :
  probability_sum_odd_of_four_selected = answer :=
sorry

end probability_sum_of_four_selected_is_odd_l272_272094


namespace alice_bob_meeting_point_l272_272898

def meet_same_point (turns : ℕ) : Prop :=
  ∃ n : ℕ, turns = 2 * n ∧ 18 ∣ (7 * n - (7 * n + n))

theorem alice_bob_meeting_point :
  meet_same_point 36 :=
by
  sorry

end alice_bob_meeting_point_l272_272898


namespace angle_between_mirrors_l272_272448

variable (EC' DA : Type)

def beam_coincidence_condition (ψ_total φ α : ℝ) : Prop :=
  ψ_total = 2 * φ ∧ ψ_total = α

theorem angle_between_mirrors 
  (ψ_total φ α : ℝ)
  (h_coincidence : beam_coincidence_condition ψ_total φ α)
  (h_alpha : α = 30) :
  φ = 15 :=
by 
  cases h_coincidence with h1 h2
  rw [←h2, h_alpha, two_mul] at h1 
  linarith

end angle_between_mirrors_l272_272448


namespace correct_options_l272_272633

variable {Ω : Type} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variable {A B : Set Ω}

open MeasureTheory

theorem correct_options (hA : P A > 0) (hB : P B > 0) :
  (Independent A B → P[B|A] = P B) ∧
  (P[B|A] = P B → P[A|B] = P A) ∧
  (P (A ∩ B) + P (Aᶜ ∩ B) = P B) :=
by
  sorry

end correct_options_l272_272633


namespace quadratic_inequality_solution_l272_272934

theorem quadratic_inequality_solution (m : ℝ) (h : m ≠ 0) : 
  (∃ x : ℝ, m * x^2 - x + 1 < 0) ↔ (m ∈ Set.Iio 0 ∨ m ∈ Set.Ioo 0 (1 / 4)) :=
by
  sorry

end quadratic_inequality_solution_l272_272934


namespace total_distance_6_seconds_is_5_metres_l272_272864

def velocity (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t < 1 then 2
  else if 1 ≤ t ∧ t < 2 then 1
  else if 2 ≤ t ∧ t < 4 then 0
  else if 4 ≤ t ∧ t < 5 then 2
  else if 5 ≤ t ∧ t ≤ 6 then 0
  else 0

def distance_traveled : ℝ :=
  ∫ t in 0..6, velocity t

theorem total_distance_6_seconds_is_5_metres : distance_traveled = 5 := by
  sorry

end total_distance_6_seconds_is_5_metres_l272_272864


namespace parallelogram_sides_l272_272074

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 3 * x + 6 = 15) 
  (h2 : 10 * y - 2 = 12) :
  x + y = 4.4 := 
sorry

end parallelogram_sides_l272_272074


namespace sum_of_integers_60_to_80_l272_272221

theorem sum_of_integers_60_to_80 : 
  let x := ∑ i in finset.range (80 - 60 + 1), (i + 60) in
  x = 1470 :=
by
  sorry

end sum_of_integers_60_to_80_l272_272221


namespace smallest_number_with_2020_divisors_l272_272143

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l272_272143


namespace triangle_min_side_bounds_l272_272407

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def min_side_values (a b c : ℝ) : Set ℝ :=
  {x : ℝ | x = min a (min b c)}

theorem triangle_min_side_bounds (a b c : ℝ) (h : a + b + c = 1) (h₁ : is_triangle a b c) : 
  ∃ (min_side : ℝ), 
  min_side ∈ min_side_values a b c ∧ min_side ∈ Ioc ( (3 - Real.sqrt 5) / 4 ) ( 1 / 3 ) :=
sorry

end triangle_min_side_bounds_l272_272407


namespace trapezoid_AD_equal_l272_272694

/-- In trapezoid ABCD, with AC = 1 (and it is also the height), AD = CF, and BC = CE.
    Prove that AD = sqrt(sqrt(2) - 1). -/
theorem trapezoid_AD_equal (A B C D E F : Point)
  (AC_eq_1 : dist A C = 1)
  (AC_height : ∃ h, h = 1)
  (AD_eq_CF : dist A D = dist C F)
  (BC_eq_CE : dist B C = dist C E)
  (perp_AE_CD : perpendicular A E C D)
  (perp_CF_AB : perpendicular C F A B)
  : dist A D = Real.sqrt (Real.sqrt 2 - 1) := 
sorry

end trapezoid_AD_equal_l272_272694


namespace find_S10_value_l272_272983

noncomputable def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, 4 * S n = n * (a n + a (n + 1))

theorem find_S10_value (a S : ℕ → ℕ) (h1 : a 4 = 7) (h2 : sequence_sum a S) :
  S 10 = 100 :=
sorry

end find_S10_value_l272_272983


namespace minimum_distance_is_sqrt3_l272_272947

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

theorem minimum_distance_is_sqrt3 (a : ℝ) :
  ∃ (a : ℝ), (1, a, 0) = (1, a, 0) ∧ (1 - a, 2, 1) = (1 - a, 2, 1) ∧ distance (1, a, 0) (1 - a, 2, 1) = real.sqrt 3 :=
sorry

end minimum_distance_is_sqrt3_l272_272947


namespace valid_lineups_count_l272_272308

theorem valid_lineups_count :
  ∃ n : ℕ, n = 11220 ∧
  (∃ A B C D : Fin 16, 
     (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) ∧ 
     (4 ≤ (A :: B :: C :: D :: []).length) ∧
     (n = Nat.choose 16 7 - Nat.choose 12 3)
  ) :=
begin
  sorry,
end

end valid_lineups_count_l272_272308


namespace percent_less_than_p_percent_more_than_q_l272_272442

variables (Q P : ℝ)

def percent_less (p q : ℝ) : ℝ := ((p - q) / p) * 100
def percent_more (p q : ℝ) : ℝ := ((p - q) / q) * 100

theorem percent_less_than_p (h : P = 1.5 * Q) :
  percent_less P Q = 33.33 :=
by sorry

theorem percent_more_than_q (h : P = 1.5 * Q) :
  percent_more P Q = 50 :=
by sorry

end percent_less_than_p_percent_more_than_q_l272_272442


namespace AP_solution_l272_272524

variable (Γ : Type) [MetricSpace Γ]  -- We represent the circle and its properties
variable (O A B C D P L : Point Γ)
variable (h_circle : ∀ {X : Point Γ}, dist O X = dist O A → X ∈ Circle O (dist O A))
variable (h_chord_AB : dist A B = 6)
variable (h_chord_CD : dist C D = 6)
variable (h_ext_intersect : ∃ P, LineThrough O A ∧ LineThrough O D)
variable (h_PO_intersect_AC : L ∈ Segment A C ∧ Ratio AL : LC = 1 : 2)

theorem AP_solution :
  dist A P = 6 :=
sorry

end AP_solution_l272_272524


namespace divisor_in_second_division_is_19_l272_272834

theorem divisor_in_second_division_is_19 (n d : ℕ) (h1 : n % 25 = 4) (h2 : (n + 15) % d = 4) : d = 19 :=
sorry

end divisor_in_second_division_is_19_l272_272834


namespace part_a_part_b_part_c_part_d_l272_272450

-- Definition of the polynomial and conditions
noncomputable def f (x : ℝ) (p q : ℝ) := x^3 + p * x + q

-- part (a)
theorem part_a (p q : ℝ) (h : p ≥ 0) :
  ∀ b : ℝ, ∃! x : ℝ, f x p q = b :=
by sorry

-- part (b)
theorem part_b (p q : ℝ) (h : p < 0) :
  ∃ b : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ p q = b ∧ f x₂ p q = b ∧ f x₃ p q = b :=
by sorry

-- part (c)
theorem part_c (p q : ℝ) (h : p < 0) :
  ∃ x_min x_max : ℝ, x_min ≠ x_max ∧ is_local_min_on (f x p q) x_min ∧ is_local_max_on (f x p q) x_max :=
by sorry

-- part (d)
theorem part_d (p q : ℝ) (h : p < 0) :
  ∀ (x_min x_max : ℝ), is_local_min_on (f x p q) x_min → is_local_max_on (f x p q) x_max → x_min = -x_max :=
by sorry

end part_a_part_b_part_c_part_d_l272_272450


namespace natalie_bushes_l272_272088

theorem natalie_bushes (bush_yield : ℕ) (containers_per_zucchini : ℕ → ℕ) (desired_zucchinis : ℕ):
  (bush_yield = 10) →
  (containers_per_zucchini 1 = 2) →
  (desired_zucchinis = 60) →
  ∃ bushes_needed : ℕ, bushes_needed = 12 :=
by
  intros h_bush_yield h_containers_per_zucchini h_desired_zucchinis
  use 12
  sorry

end natalie_bushes_l272_272088


namespace crank_slider_properties_l272_272529

-- Define constants
def ω : ℝ := 10 -- Angular velocity in rad/s

-- Definitions related to the dimensions
def OA : ℝ := 90 -- cm
def AB : ℝ := 90 -- cm
def AM : ℝ := (2/3) * AB -- cm

-- Position functions
def position_A (t : ℝ) : ℝ × ℝ :=
  (OA * Real.cos(ω * t), OA * Real.sin(ω * t)) -- coordinates of point A

def position_B (t : ℝ) : ℝ × ℝ := 
  let (A_x, A_y) := position_A t in
  (A_x + AB, 0) -- coordinates of point B

def position_M (t : ℝ) : ℝ × ℝ := 
  let (A_x, A_y) := position_A t in
  (30 * Real.cos(ω * t) + 60, 60 * Real.sin(ω * t)) -- coordinates of point M

-- Equation of the trajectory of point M
def trajectory_eq_of_M (M_x M_y : ℝ) : Prop :=
  ((M_y)^2 / 3600 + (M_x - 60)^2 / 900 = 1)

-- Velocity of point M
def velocity_M (t : ℝ) : ℝ × ℝ :=
  (-300 * (Real.sin(ω * t)), 600 * (Real.cos(ω * t)))

-- Main theorem to prove the corresponding properties
theorem crank_slider_properties :
  ∀ (t: ℝ),
  let (M_x, M_y) := position_M t in
  position_A t = (OA * Real.cos(ω * t), OA * Real.sin(ω * t)) ∧
  position_B t = (OA * Real.cos(ω * t) + AB, 0) ∧
  M_x = 30 * Real.cos(ω * t) + 60 ∧ M_y = 60 * Real.sin(ω * t) ∧
  trajectory_eq_of_M M_x M_y ∧
  velocity_M t = (-300 * (Real.sin(ω * t)), 600 * (Real.cos(ω * t))) :=
by {
  sorry
}

end crank_slider_properties_l272_272529


namespace smaller_angle_at_7_30_l272_272415

def clock_angle_deg_per_hour : ℝ := 30 

def minute_hand_angle_at_7_30 : ℝ := 180

def hour_hand_angle_at_7_30 : ℝ := 225

theorem smaller_angle_at_7_30 : 
  ∃ angle : ℝ, angle = 45 ∧ 
  (angle = |hour_hand_angle_at_7_30 - minute_hand_angle_at_7_30|) :=
begin
  sorry
end

end smaller_angle_at_7_30_l272_272415


namespace smallest_number_with_2020_divisors_is_correct_l272_272117

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l272_272117


namespace maximal_distance_from_point_on_circumsphere_to_face_l272_272630

noncomputable def maximal_distance (P A B C : Point) (Q : Point) : Real :=
  if h1 : PA = 1 ∧ PB = 2 ∧ PC = 3 ∧ PA ⊥ PB ∧ PB ⊥ PC ∧ PC ⊥ PA then
    (3 / 7) + (sqrt 14 / 2)
  else
    0 -- Default value when preconditions are not met.

theorem maximal_distance_from_point_on_circumsphere_to_face 
  (P A B C : Point) (Q : Point) 
  (h1 : PA = 1) 
  (h2 : PB = 2) 
  (h3 : PC = 3) 
  (h4 : PA ⊥ PB) 
  (h5 : PB ⊥ PC) 
  (h6 : PC ⊥ PA) :
  maximal_distance P A B C Q = (3 / 7) + (sqrt 14 / 2) :=
by sorry

end maximal_distance_from_point_on_circumsphere_to_face_l272_272630


namespace rounding_of_7362_499_l272_272758

theorem rounding_of_7362_499 :
  Real.round 7362.499 = 7362 :=
by
  sorry

end rounding_of_7362_499_l272_272758


namespace surface_area_circumsphere_M_BCD_l272_272013

noncomputable def surface_area_circumsphere_tetra (A B C D G M : Point) 
  (h1 : regular_tetrahedron A B C D)
  (h2 : centroid_triangle G B C D)
  (h3 : midpoint_segment M A G) : ℝ :=
  3 / 2 * π

theorem surface_area_circumsphere_M_BCD (A B C D G M : Point)
  (h1 : regular_tetrahedron A B C D)
  (h2 : centroid_triangle G B C D)
  (h3 : midpoint_segment M A G) :
  surface_area_circumsphere_tetra A B C D G M h1 h2 h3 = 3 / 2 * π :=
sorry

end surface_area_circumsphere_M_BCD_l272_272013


namespace number_of_prime_looking_numbers_l272_272520

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n
def is_prime_looking (n : ℕ) : Prop := is_composite n ∧ ¬ (is_divisible n 2 ∨ is_divisible n 3 ∨ is_divisible n 5)

theorem number_of_prime_looking_numbers (h_primes : ∀ P, is_prime P → P < 1000 → ∃! P, count P 168 )
  : ∃ k : ℕ, k = 100 ∧ ∀ n : ℕ, n < 1000 → is_prime_looking n ↔ (n > 1 ∧ is_composite n ∧ ¬ (is_divisible n 2 ∨ is_divisible n 3 ∨ is_divisible n 5)) :=
sorry

end number_of_prime_looking_numbers_l272_272520


namespace percent_difference_l272_272645

theorem percent_difference :
  (0.90 * 40) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_l272_272645


namespace raised_bed_area_correct_l272_272063

def garden_length : ℝ := 220
def garden_width : ℝ := 120
def garden_area : ℝ := garden_length * garden_width
def tilled_land_area : ℝ := garden_area / 2
def remaining_area : ℝ := garden_area - tilled_land_area
def trellis_area : ℝ := remaining_area / 3
def raised_bed_area : ℝ := remaining_area - trellis_area

theorem raised_bed_area_correct : raised_bed_area = 8800 := by
  sorry

end raised_bed_area_correct_l272_272063


namespace square_area_in_ellipse_l272_272485

theorem square_area_in_ellipse (t : ℝ) (ht : 0 < t) :
  (t ^ 2 / 4 + t ^ 2 / 8 = 1) →
  let side_length := 2 * t in
  let area := side_length ^ 2 in
  area = 32 / 3 := 
sorry

end square_area_in_ellipse_l272_272485


namespace omega_cannot_be_3_over_4_l272_272622

theorem omega_cannot_be_3_over_4 (ω : ℝ) (h₁ : ω > 0) 
  (h₂ : ∀ x y : ℝ, -π/2 < x → x < π/2 → -π/2 < y → y < π/2 → x < y → (cos(ω * x) - sin(ω * x)) ≥ (cos(ω * y) - sin(ω * y))) : ω ≠ 3/4 :=
by
  sorry

end omega_cannot_be_3_over_4_l272_272622


namespace players_exceed_30000_l272_272028

theorem players_exceed_30000
  (R : ℕ → ℝ)
  (R0 : ℝ)
  (k : ℝ)
  (R_formula : ∀ t, R(t) = R0 * Real.exp(k * t))
  (R_at_0 : R(0) = 100)
  (R_at_5 : R(5) = 1000)
  (approx_lg3 : Real.log 3 ≈ 0.4771) :
  ∃ t : ℕ, t ≥ 13 ∧ R(t) > 30000 :=
by
  have R0_eq_100 : R0 = 100 := by 
    sorry
  have k_eq : k = Real.log 10 / 5 := by
    sorry
  existsi 13
  split
  · exact Nat.le_refl 13
  · rw [R_formula, R0_eq_100, k_eq]
    have exp_inequality : 10 ^ (13 / 5) > 300 := by
      sorry
    exact exp_inequality

end players_exceed_30000_l272_272028


namespace roots_of_quadratic_polynomial_l272_272273

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_progression (roots : List α) : Prop :=
∀ (i j k : ℕ), i < j → j < k → i < roots.length → k < roots.length → 
roots.get i + roots.get k = 2 * roots.get j

theorem roots_of_quadratic_polynomial {P : polynomial ℚ} (hdeg : P.degree ≥ 2) 
(hroots : ∃ (roots : List ℚ), roots.length = P.nat_degree ∧ 
                           ∀ r, r ∈ roots → (polynomial.eval r P = 0) ∧ 
                           roots.pairwise (≠) ∧ 
                           is_arithmetic_progression roots) : 
∃ (a b : ℚ), a ≠ b ∧ ∃ Q : polynomial ℚ, Q.degree = 2 ∧ 
                                polynomial.eval a Q = 0 ∧ 
                                polynomial.eval b Q = 0 :=
sorry

end roots_of_quadratic_polynomial_l272_272273


namespace price_of_hot_water_bottle_l272_272412

-- Definitions of the conditions
def thermometer_price : ℕ := 2
def total_sales : ℕ := 1200
def hot_water_bottles_sold : ℕ := 60
def thermometer_sold := 7 * hot_water_bottles_sold

-- The proof statement 
theorem price_of_hot_water_bottle : ∃ (P : ℕ), 840 + 60 * P = total_sales ∧ P = 6 :=
by
  exists 6
  split
  sorry

end price_of_hot_water_bottle_l272_272412


namespace systematic_sample_first_segment_number_l272_272231

theorem systematic_sample_first_segment_number :
  ∃ a_1 : ℕ, ∀ d k : ℕ, k = 5 → a_1 + (59 - 1) * k = 293 → a_1 = 3 :=
by
  sorry

end systematic_sample_first_segment_number_l272_272231


namespace lucas_product_l272_272274

def lucas (n : ℕ) : ℕ
| 1       := 1
| 2       := 3
| (n + 1) := lucas n + lucas (n - 1)

theorem lucas_product : 
  (∏ k in finset.range 49 \ finset.singleton 0, 
     (lucas (k + 2)) / (lucas (k + 1)) - (lucas (k + 2)) / (lucas (k + 3))) = 
  (lucas 50) / (lucas 51) :=
sorry

end lucas_product_l272_272274


namespace different_picture_size_is_correct_l272_272750

-- Define constants and conditions
def memory_card_picture_capacity := 3000
def single_picture_size := 8
def different_picture_capacity := 4000

-- Total memory card capacity in megabytes
def total_capacity := memory_card_picture_capacity * single_picture_size

-- The size of each different picture
def different_picture_size := total_capacity / different_picture_capacity

-- The theorem to prove
theorem different_picture_size_is_correct :
  different_picture_size = 6 := 
by
  -- We include 'sorry' here to bypass actual proof
  sorry

end different_picture_size_is_correct_l272_272750


namespace yuna_candy_days_l272_272006

theorem yuna_candy_days (total_candies : ℕ) (daily_candies_week : ℕ) (days_week : ℕ) (remaining_candies : ℕ) (daily_candies_future : ℕ) :
  total_candies = 60 →
  daily_candies_week = 6 →
  days_week = 7 →
  remaining_candies = total_candies - (daily_candies_week * days_week) →
  daily_candies_future = 3 →
  remaining_candies / daily_candies_future = 6 :=
by
  intros h_total h_daily_week h_days_week h_remaining h_daily_future
  sorry

end yuna_candy_days_l272_272006


namespace can_partition_to_equal_hexafaced_polyhedra_l272_272532

-- Define a RegularTetrahedron type and avail an instance of it
structure RegularTetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ
  is_regular : ∀ i j k l, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
                  dist (vertices i) (vertices j) = dist (vertices i) (vertices k)

-- Define the main theorem to prove the partitioning
theorem can_partition_to_equal_hexafaced_polyhedra (T : RegularTetrahedron) :
    ∃ P : ℕ → (ℝ × ℝ × ℝ) set, (∀ i, (convex_hull ℝ (P i)).card = 6) ∧
    (∀ i j, i ≠ j → disjoint (P i) (P j)) ∧
    (⋃ i, P i) = convex_hull ℝ (set.range T.vertices) := 
sorry

end can_partition_to_equal_hexafaced_polyhedra_l272_272532


namespace sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l272_272202

theorem sum_of_consecutive_natural_numbers_eq_three_digit_same_digits :
  ∃ n : ℕ, (1 + n) * n / 2 = 111 * 6 ∧ n = 36 :=
by
  sorry

end sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l272_272202


namespace probability_mass_range_l272_272958

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end probability_mass_range_l272_272958


namespace smallest_number_of_students_l272_272667

theorem smallest_number_of_students:
  ∃ n : ℕ, n ≡ 1 [MOD 6] ∧ n ≡ 2 [MOD 8] ∧ n ≡ 4 [MOD 10] ∧
  ∀ m : ℕ, (m ≡ 1 [MOD 6] ∧ m ≡ 2 [MOD 8] ∧ m ≡ 4 [MOD 10]) → m ≥ n :=
begin
  use 274,
  split,
  { exact nat.modeq.modeq_of_dvd' 1 },
  split,
  { exact nat.modeq.modeq_of_dvd' 2 },
  split,
  { exact nat.modeq.modeq_of_dvd' 4 },
  { intros m hm,
    have h274: 274 = 2 * (3 * 40 + 3) + 4, by norm_num,
    have h3m: m = 2 * (3 * (m / 6) + m % 6) + 4, by norm_num,
    norm_num at hm,
    sorry
  } 
end

end smallest_number_of_students_l272_272667


namespace compare_p_q_l272_272277

variable (a : ℝ)

def P : ℝ := (sqrt (a + 41)) - (sqrt (a + 40))
def Q : ℝ := (sqrt (a + 39)) - (sqrt (a + 38))

theorem compare_p_q (h : a > -38) : P a < Q a := by
  sorry

end compare_p_q_l272_272277


namespace second_container_price_is_correct_l272_272035

/-
  A cylindrical container of salsa that is 2 inches in diameter and 5 inches high sells for $0.50.
  The price for a container that is 4 inches in diameter and 10 inches high
  should be proportional to its volume, maintaining the same price per unit volume.
-/

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h

def first_container_diameter : ℝ := 2
def first_container_height : ℝ := 5
def first_container_price : ℝ := 0.50

def second_container_diameter : ℝ := 4
def second_container_height : ℝ := 10

def price_second_container : ℝ := first_container_price * 
    (volume (second_container_diameter / 2) second_container_height) / 
    (volume (first_container_diameter / 2) first_container_height)

theorem second_container_price_is_correct : 
  price_second_container = 4 := by
  sorry

end second_container_price_is_correct_l272_272035


namespace uncle_bradley_bills_l272_272814

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end uncle_bradley_bills_l272_272814


namespace y_intercept_of_line_l272_272099

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (0, 4) = (0, y) :=
by { intro h,
     have y_eq : y = 4,
     { 
       sorry
     },
     have : (0, y) = (0, 4),
     { 
       sorry 
     },
     exact this }

end y_intercept_of_line_l272_272099


namespace remainder_sum_div7_l272_272949

theorem remainder_sum_div7 (a b c : ℕ) (h1 : a * b * c ≡ 2 [MOD 7])
  (h2 : 3 * c ≡ 4 [MOD 7])
  (h3 : 4 * b ≡ 2 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_sum_div7_l272_272949


namespace square_area_in_ellipse_l272_272484

theorem square_area_in_ellipse (t : ℝ) (ht : 0 < t) :
  (t ^ 2 / 4 + t ^ 2 / 8 = 1) →
  let side_length := 2 * t in
  let area := side_length ^ 2 in
  area = 32 / 3 := 
sorry

end square_area_in_ellipse_l272_272484


namespace percentage_of_work_day_in_meetings_is_25_l272_272741

-- Define the conditions
def workDayHours : ℕ := 9
def firstMeetingMinutes : ℕ := 45
def secondMeetingMinutes : ℕ := 2 * firstMeetingMinutes
def totalMeetingMinutes : ℕ := firstMeetingMinutes + secondMeetingMinutes
def workDayMinutes : ℕ := workDayHours * 60

-- Define the percentage calculation
def percentageOfWorkdaySpentInMeetings : ℕ := (totalMeetingMinutes * 100) / workDayMinutes

-- The theorem to be proven
theorem percentage_of_work_day_in_meetings_is_25 :
  percentageOfWorkdaySpentInMeetings = 25 :=
sorry

end percentage_of_work_day_in_meetings_is_25_l272_272741


namespace good_points_area_l272_272677

noncomputable def hyperbola_Γ (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 = 1

def Ω_P (P : ℝ × ℝ) : set (ℝ → ℝ) :=
  {l | ∃ (k : ℝ) (c : ℝ), l = (λ x, k * x + c) ∧
        ∃ M N : ℝ × ℝ,
          M ≠ P ∧ N ≠ P ∧
          hyperbola_Γ M.1 M.2 ∧ hyperbola_Γ N.1 N.2 ∧
          M.2 = l M.1 ∧ N.2 = l N.1}

noncomputable def f_P (P : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  let M := classical.some (Ω_P P l).1 in
  let N := classical.some (Ω_P P l).2 in
  real.dist P.1 M.1 * real.dist P.2 N.2

def good_point (P : ℝ × ℝ) : Prop :=
  ∃ (l₀ ∈ Ω_P P), (∃ M N, 
      M.2 = (l₀ M.1) ∧ N.2 = (l₀ N.1) ∧
      M.1 * N.1 < 0) ∧
  ∀ l ∈ Ω_P P, l ≠ l₀ → f_P P l > f_P P l₀

theorem good_points_area : 
  let region := {P : ℝ × ℝ | good_point P}
  ∀ r : region, area r = 4 :=
sorry

end good_points_area_l272_272677


namespace ellipse_foci_on_y_axis_l272_272178

theorem ellipse_foci_on_y_axis (k : ℝ) (h1 : 5 + k > 3 - k) (h2 : 3 - k > 0) (h3 : 5 + k > 0) : -1 < k ∧ k < 3 :=
by 
  sorry

end ellipse_foci_on_y_axis_l272_272178


namespace smallest_number_has_2020_divisors_l272_272134

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l272_272134


namespace prime_bounds_l272_272166

noncomputable def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem prime_bounds (n : ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ k, 0 ≤ k → k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) : 
  ∀ k, 0 ≤ k → k ≤ n - 2 → is_prime (k^2 + k + n) :=
by
  sorry

end prime_bounds_l272_272166


namespace sum_of_possible_values_of_cardinality_l272_272955

noncomputable def sum_possible_cardinalities (A : Set ℝ) 
  (h : (Set.Image2 (λ a b => a - b) A A).Finite ∧ (Set.Image2 (λ a b => a - b) A A).toFinset.card = 25) : ℕ := by
  sorry

theorem sum_of_possible_values_of_cardinality (A : Set ℝ)
  (h : (Set.Image2 (λ a b => a - b) A A).Finite ∧ (Set.Image2 (λ a b => a - b) A A).toFinset.card = 25):
  sum_possible_cardinalities A h = 76 := by
  sorry

end sum_of_possible_values_of_cardinality_l272_272955


namespace CK_eq_AX_l272_272467

-- Given a right triangle ABC with ∠ACB = 90°
variables {A B C M K N X : Point}
variables {r : ℝ}
variables {hABC : is_right_triangle A B C ∧ ∠ A C B = 90}
variables {inscribed_circle : is_inscribed_circle A B C M K N}
variables {line_through_K : perpendicular_line_through K M N meets AC at X}

theorem CK_eq_AX (hABC : is_right_triangle A B C ∧ ∠ A C B = 90)
  (inscribed_circle : is_inscribed_circle A B C M K N)
  (line_through_K : perpendicular_line_through K M N meets AC at X) :
  distance C K = distance A X := 
sorry

end CK_eq_AX_l272_272467


namespace x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l272_272960

theorem x_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 6 * x (k - 1) - x (k - 2) := 
by sorry

theorem x_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 34 * x (k - 2) - x (k - 4) := 
by sorry

theorem x_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 198 * x (k - 3) - x (k - 6) := 
by sorry

theorem y_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 6 * y (k - 1) - y (k - 2) := 
by sorry

theorem y_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 34 * y (k - 2) - y (k - 4) := 
by sorry

theorem y_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 198 * y (k - 3) - y (k - 6) := 
by sorry

end x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l272_272960


namespace notebook_cost_l272_272878

theorem notebook_cost
  (n c : ℝ)
  (h1 : n + c = 2.20)
  (h2 : n = c + 2) :
  n = 2.10 :=
by
  sorry

end notebook_cost_l272_272878


namespace election_votes_l272_272443

/-!
Problem: 
In an election, candidate A got 75% of the total valid votes. 
If 15% of the total votes were declared invalid, and the total number of votes is 560,000, 
find the number of valid votes polled in favor of candidate A.

We need to prove that the number of valid votes polled in favor of candidate A is 357,000.
-/

theorem election_votes (total_votes : ℕ) (invalid_percentage : ℝ) (valid_percentage_A : ℝ)
    (h1 : total_votes = 560000)
    (h2 : invalid_percentage = 0.15)
    (h3 : valid_percentage_A = 0.75) :
    let valid_votes := (1 - invalid_percentage) * total_votes in
    let votes_A := valid_percentage_A * valid_votes in
    votes_A = 357000 := by
  -- Calculation of valid votes
  have valid_votes_calc : valid_votes = (1 - invalid_percentage) * total_votes := rfl
  -- Calculation of votes for candidate A
  have votes_A_calc : votes_A = valid_percentage_A * valid_votes := rfl
  have calc : votes_A = 357000, sorry
  exact calc

end election_votes_l272_272443


namespace angle_BOC_is_112_5_l272_272258

-- Let triangle ABC have sides AB, BC, and AC; O is the incenter of the triangle.
def triangle_ABC :=
  ∃ (A B C O : Type) [metric_space A] [metric_space B] [metric_space C],
  dist A B = real.sqrt 2 ∧
  dist B C = real.sqrt 5 ∧
  dist A C = 3 ∧
  is_incenter O A B C

-- Define the angle comparison
def angle_comparison :=
  ∀ {A B C O : Type} [metric_space A] [metric_space B] [metric_space C],
  triangle_ABC → ∠BOC = 112.5

-- Conjecture to be proved
theorem angle_BOC_is_112_5:
  angle_comparison :=
  sorry

end angle_BOC_is_112_5_l272_272258


namespace game_players_exceed_30000_l272_272025

theorem game_players_exceed_30000 :
  ∀ (R: ℕ → ℝ) (k R₀ : ℝ),
  (R 0 = 100) →
  (R 5 = 1000) →
  (∀ t, R t = R₀ * exp (k * t)) →
  (∃ t: ℕ, t ≥ 13 ∧ R t > 30000) := by
  sorry

end game_players_exceed_30000_l272_272025


namespace diameter_of_sphere_l272_272424

theorem diameter_of_sphere (r : ℝ) (π : ℝ) (h : r = 7) (V : ℝ → ℝ) 
  (V_eq : V r = (4 / 3) * π * (r^3)) : 
  ∃ a b : ℝ, a = 14 ∧ b = 3 ∧ b^(1 / 3) ≠ floor (b^(1 / 3)) ∧ ((a + b) = 17) :=
by 
  let volume_original := (4 / 3) * π * (7^3)
  have volume_triple := 3 * volume_original
  let radius_new := (volume_triple * (3 / 4 / π))^(1 / 3)
  have radius_cube := radius_new^3
  have h_r3 : radius_cube = 1323 := by omega
  let diameter_new := 2 * radius_new
  existsi (14 : ℝ)
  existsi (3 : ℝ)
  have h_d : diameter_new = 14 * (3)^(1/3)
  split; try {sorry}

end diameter_of_sphere_l272_272424


namespace smallest_n_with_divisors_2020_l272_272137

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l272_272137


namespace compute_B_plus_D_l272_272331

open Complex Polynomial

def conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩
def g (z : ℂ) : ℂ := -2 * I * conjugate z

noncomputable def P : Polynomial ℂ := Polynomial.mk [2, -3, 6, -2, 1]
noncomputable def z1 : ℂ := (Classical.some (Polynomial.exists_root P))
noncomputable def z2 : ℂ := (Classical.some (Polynomial.exists_root P.erase z1))
noncomputable def z3 : ℂ := (Classical.some (Polynomial.exists_root P.erase z1.erase z2))
noncomputable def z4 : ℂ := (Classical.some (Polynomial.exists_root P.erase z1.erase z2.erase z3))

noncomputable def Q : Polynomial ℂ := Polynomial.map g P

theorem compute_B_plus_D :
  let B := Q.coeff 2
  let D := Q.coeff 0
  B + D = 56 := by
  sorry

end compute_B_plus_D_l272_272331


namespace true_proposition_is_one_l272_272054
-- Lean 4 code

theorem true_proposition_is_one :
  (∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ∧
  (¬ (∀ A B : ℝ, A > B → sin A > sin B)) ∧
  (¬ (∀ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = (a (n + 1))^2) ↔ (∀ n, a n * a (n + 2) = (a (n + 1))^2 ∧ a n ≠ 0 ∧ a (n + 1) ≠ 0))) ∧
  (¬ (∀ x : ℝ, (x > 0) → lg x + 1 / lg x = 2)) :=
by
  split; 
  { intros,
    apply sorry }

-- Note: The proof sections (apply sorry) can be expanded with actual proof tactics as needed.

end true_proposition_is_one_l272_272054


namespace geometric_sequence_sum_of_sequence_l272_272629

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 1) := (2 * (n + 1) * a n / n) + (n + 1)

def geo_seq (n : ℕ) : ℕ := a n / n + 1

def S (n : ℕ) : ℕ := (n - 1) * 2 ^ (n + 1) - n * (n + 1) / 2 + 2

theorem geometric_sequence (n : ℕ) : 
  geo_seq 1 = 2 ∧ ∀ n > 0, geo_seq (n + 1) = 2 * geo_seq n := 
  sorry

theorem sum_of_sequence : ∀ n, (Σ i in range n, a (i + 1)) = S n := 
  sorry

end geometric_sequence_sum_of_sequence_l272_272629


namespace divisor_properties_of_420_l272_272538

theorem divisor_properties_of_420 :
  let n := 420
  ( ∑ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))), d = 1344 ) ∧
  ( (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card = 24 ) :=
by
  sorry

end divisor_properties_of_420_l272_272538


namespace money_distribution_probability_l272_272809

/-- Three players (Carl, Maya, and Lily) each start with $2. Every 30 seconds, a bell rings and each player can independently and randomly give $1 to another player or opt not to give, if they have at least $1. Prove that the probability that after the bell rings 1000 times, each player will have $2 is 1/9. -/
theorem money_distribution_probability :
  let carl_start := 2
  let maya_start := 2
  let lily_start := 2
  let num_rings := 1000
  let probability_end := 1 / 9
  ∀ (distribution: ℕ → (ℕ × ℕ × ℕ)),
  (distribution 0 = (carl_start, maya_start, lily_start)) →
  (∀ t, 0 ≤ distribution t ∧ distribution t.1 + distribution t.2 + distribution t.3 = 6) →
  (∀ t, 
    distribution (t+1) =
    match distribution t with
    | (c, m, l) => 
      -- hypothetical transition logic placeholder
      sorry -- transition logic is not detailed in this problem statement

    end) →
  -- after num_rings times, the probability that each player has $2
  ∃ t_final = num_rings,
  (distribution t_final = (2, 2, 2)) →
  true :=
sorry

end money_distribution_probability_l272_272809


namespace bulb_positions_97_100_l272_272804

def bulb_color : ℕ → Prop
| 1 := true  -- First bulb is yellow (Y)
| 2 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 3 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 4 := false  -- Placeholder for demonstrational purposes, actual implementation needed
| 5 := true -- Fifth bulb is yellow (Y)
| n := sorry  -- Placeholder for the general case to be defined

-- Condition that among any five consecutive bulbs, exactly two are yellow and exactly three are blue.
def valid_sequence (n : ℕ) : Prop :=
  (bulb_color n ∧ bulb_color (n + 4)) ∧
  (¬bulb_color (n + 1) ∧ ¬bulb_color (n + 2) ∧ ¬bulb_color (n + 3))

theorem bulb_positions_97_100 :
  bulb_color 97 = true ∧ bulb_color 98 = false ∧ bulb_color 99 = false ∧ bulb_color 100 = true :=
by
  -- This is the place where the proof for the sequence constraints and result should be filled in
  sorry

end bulb_positions_97_100_l272_272804


namespace smallest_n_with_2020_divisors_l272_272127

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l272_272127


namespace minimum_one_by_one_squares_l272_272936

theorem minimum_one_by_one_squares :
  ∀ (x y z : ℕ), 9 * x + 4 * y + z = 49 → (z = 3) :=
  sorry

end minimum_one_by_one_squares_l272_272936


namespace probability_of_selecting_exactly_2_female_students_l272_272867

noncomputable def choose (n k : ℕ) : ℕ := n.choose k

def number_of_ways_to_select_2_students : ℕ := choose 5 2
def number_of_ways_to_select_2_females : ℕ := choose 3 2
def probability_of_selecting_2_females : ℝ := number_of_ways_to_select_2_females / number_of_ways_to_select_2_students

theorem probability_of_selecting_exactly_2_female_students : probability_of_selecting_2_females = 0.3 :=
by
  have h1 : number_of_ways_to_select_2_students = 10 := by norm_num
  have h2 : number_of_ways_to_select_2_females = 3 := by norm_num
  have h3 : probability_of_selecting_2_females = 3 / 10 := by 
    rw [h1, h2]
    norm_num
  exact h3

end probability_of_selecting_exactly_2_female_students_l272_272867


namespace y_intercept_of_line_l272_272102

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l272_272102


namespace increased_percentage_l272_272376

theorem increased_percentage (P : ℝ) (N : ℝ) (hN : N = 80) 
  (h : (N + (P / 100) * N) - (N - (25 / 100) * N) = 30) : P = 12.5 := 
by 
  sorry

end increased_percentage_l272_272376


namespace stationery_cost_l272_272502

theorem stationery_cost :
  let 
    pencil_price := 4
    pen_price := 5
    boxes := 15
    pencils_per_box := 80
    pencils_ordered := boxes * pencils_per_box
    total_cost_pencils := pencils_ordered * pencil_price
    pens_ordered := 2 * pencils_ordered + 300
    total_cost_pens := pens_ordered * pen_price
  in 
  total_cost_pencils + total_cost_pens = 18300 :=
by 
  sorry

end stationery_cost_l272_272502


namespace unripe_oranges_zero_l272_272640

def oranges_per_day (harvest_duration : ℕ) (ripe_oranges_per_day : ℕ) : ℕ :=
  harvest_duration * ripe_oranges_per_day

theorem unripe_oranges_zero
  (harvest_duration : ℕ)
  (ripe_oranges_per_day : ℕ)
  (total_ripe_oranges : ℕ)
  (h1 : harvest_duration = 25)
  (h2 : ripe_oranges_per_day = 82)
  (h3 : total_ripe_oranges = 2050)
  (h4 : oranges_per_day harvest_duration ripe_oranges_per_day = total_ripe_oranges) :
  ∀ unripe_oranges_per_day, unripe_oranges_per_day = 0 :=
by
  sorry

end unripe_oranges_zero_l272_272640


namespace black_white_area_ratio_l272_272568

theorem black_white_area_ratio :
  ∃ (radii : ℕ → ℝ) (black_indices : set ℕ) (white_indices : set ℕ),
    let black_area := ∑ i in black_indices, π * (radii i)^2 - if i > 0 then π * (radii (i - 1))^2 else 0
    let white_area := ∑ i in white_indices, π * (radii i)^2 - if i > 0 then π * (radii (i - 1))^2 else 0
    radii 0 = 1 ∧ radii 1 = 3 ∧ radii 2 = 5 ∧ radii 3 = 7 ∧ radii 4 = 9 ∧
    black_indices = {0, 2, 4} ∧ 
    white_indices = {1, 3} ∧
    (black_area / white_area = 49 / 32) :=
begin
  sorry
end

end black_white_area_ratio_l272_272568


namespace consecutive_integers_in_list_F_l272_272297

theorem consecutive_integers_in_list_F (F : Set Int) (H1 : ∀ a b, a ∈ F → b ∈ F → (a ≤ b → ∀ x, a ≤ x ∧ x ≤ b → x ∈ F)) 
  (H2 : -4 ∈ F) (H3 : ∃ n, 1 ≤ n ∧ n ∈ F ∧ ∀ m, 1 ≤ m → m ∈ F → m ≤ n ∧ ∀ k ∈ F, k > n → k - n ≤ 6) : 
  F.card = 12 :=
sorry

end consecutive_integers_in_list_F_l272_272297


namespace part1_part2_l272_272194

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x - 4) - abs (x + 2)

-- Statement for Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, f x - a^2 + 5 * a ≥ 0) → (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

-- Statement for Part (2)
theorem part2 (a b c : ℝ) (h : a + b + c = 6) : 
  let M := 6 in 
  (sqrt (a + 1) + sqrt (b + 2) + sqrt (c + 3) ≤ 6) :=
by
  sorry

end part1_part2_l272_272194


namespace average_of_ABC_l272_272800

theorem average_of_ABC (A B C : ℝ) 
  (h1 : 2002 * C - 1001 * A = 8008) 
  (h2 : 2002 * B + 3003 * A = 7007) 
  (h3 : A = 2) : (A + B + C) / 3 = 2.33 := 
by 
  sorry

end average_of_ABC_l272_272800


namespace four_digit_number_count_l272_272161

theorem four_digit_number_count :
  ∃ n : ℕ, n = 36 ∧
  (∑ x y z w : ℕ, {1, 2, 3, 4}.erase 1 s.length.w. = n ∧
    x = 1 ∧ z = z ∧ (x + y + z + w = 10))
  :=
sorry

end four_digit_number_count_l272_272161


namespace total_weight_of_batch_of_rice_l272_272865

theorem total_weight_of_batch_of_rice 
  (total_weight : ℝ)
  (consumed_day_one : total_weight * (3 / 10))
  (remaining_after_day_one : total_weight - consumed_day_one)
  (consumed_day_two : remaining_after_day_one * (2 / 5))
  (remaining_after_day_two : remaining_after_day_one - consumed_day_two)
  (remaining_weight : remaining_after_day_two = 210) :
  total_weight = 500 :=
by
  sorry

end total_weight_of_batch_of_rice_l272_272865


namespace eval_expression_l272_272940

theorem eval_expression : 
  3000^3 - 2998 * 3000^2 - 2998^2 * 3000 + 2998^3 = 23992 := 
by 
  sorry

end eval_expression_l272_272940


namespace triangle_angle_at_least_120_l272_272182

theorem triangle_angle_at_least_120 (points : Fin 6 → ℝ × ℝ) 
  (h_distinct : Function.injective points)
  (h_no_three_collinear : ∀ a b c : Fin 6, a ≠ b → a ≠ c → b ≠ c → 
     ¬IsCollinear points a b c) :
  ∃ (a b c : Fin 6), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
    ∃ (angle : ℝ), angle ≥ 120 ∧ angle = angle_of_triangle points a b c :=
sorry

end triangle_angle_at_least_120_l272_272182


namespace num_acute_integer_x_l272_272154

def is_acute_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

theorem num_acute_integer_x : 
  {x : ℕ | 18 < x ∧ x < 54 ∧ is_acute_triangle 18 36 x}.card = 9 := by
  sorry

end num_acute_integer_x_l272_272154


namespace medians_equal_in_length_l272_272751

-- Given:
variables {A B C M : Point} -- Points on the plane such that M lies on segment BC
variable [Field k]          -- Consider k as the field for coordinates
variables [AddGroup B] [DistributionField C] [Field AddGroup] [DistributionField AddGroup]-- Define Field

-- Define the centroids of the triangles
def centroid_ABM (A B M : Point) : Point := (A + B + M) / 3
def centroid_ACM (A C M : Point) : Point := (A + C + M) / 3

-- Define circumcircles
def circumcircle (A B C : Point) : Set Point := sorry
-- Definition of circumcircle is complex and involves constructing the circle passing through all three points

-- Define Points lying on circumcircle
def lies_on_circumcircle (P : Point) (circum : Set Point) : Prop := P ∈ circum

-- Assume M is on BC
axiom M_on_BC : ∃ t : k, M = t • B + (1 - t) • C

-- Given conditions
axiom centroid_Tc_on_circumcircle_ACM : lies_on_circumcircle (centroid_ABM A B M) (circumcircle A C M)
axiom centroid_Tb_on_circumcircle_ABM : lies_on_circumcircle (centroid_ACM A C M) (circumcircle A B M)

-- Prove that the medians of the triangles ABM and ACM from M are of the same length
theorem medians_equal_in_length :
  let median_ABM := dist M ((A + B) / 2) in
  let median_ACM := dist M ((A + C) / 2) in
  median_ABM = median_ACM :=
sorry

end medians_equal_in_length_l272_272751


namespace count_positive_integers_l272_272580

theorem count_positive_integers (count : ℕ) :
  count = (List.filter (λ x : ℕ, 150 ≤ x^2 ∧ x^2 ≤ 300) (List.range 19)).length := by
  sorry

end count_positive_integers_l272_272580


namespace distance_from_center_to_point_l272_272425

def originEq : String := "x^2 + y^2 = 4x + 6y + 3"

theorem distance_from_center_to_point : 
  let center_x := 2
  let center_y := 3
  let point_x := 10
  let point_y := 5
  sqrt ((point_x - center_x)^2 + (point_y - center_y)^2) = 2 * sqrt 17 := 
by
  sorry

end distance_from_center_to_point_l272_272425


namespace trader_profit_percentage_l272_272848

variables {P : ℝ} (hP : P > 0)

-- The price the trader bought the car
def purchase_price := 0.80 * P

-- The price the trader sold the car for
def selling_price := purchase_price * 1.70

-- The profit made
def profit := selling_price - P

-- The profit percentage on the original price
def profit_percentage := (profit / P) * 100

theorem trader_profit_percentage :
  profit_percentage = 36 := by
  unfold profit_percentage profit selling_price purchase_price
  field_simp
  have h : (1.36 - 1) * 100 = 36 := by norm_num
  exact h

end trader_profit_percentage_l272_272848


namespace lex_reads_in_12_days_l272_272296

theorem lex_reads_in_12_days
  (total_pages : ℕ)
  (pages_per_day : ℕ)
  (h1 : total_pages = 240)
  (h2 : pages_per_day = 20) :
  total_pages / pages_per_day = 12 :=
by
  sorry

end lex_reads_in_12_days_l272_272296


namespace six_coins_heads_or_tails_probability_l272_272361

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l272_272361


namespace solution_to_inequality_l272_272055

theorem solution_to_inequality : ∃ x : ℤ, (x ∈ {-3, 0, 2, 4}) ∧ (x > 3) ∧ (x = 4) := by
  sorry

end solution_to_inequality_l272_272055


namespace magnitude_of_a_plus_i_l272_272652

theorem magnitude_of_a_plus_i (a : ℝ) (h1 : (a-2=0)) : complex.abs (a + complex.i) = real.sqrt 5 :=
by {
  sorry
}

end magnitude_of_a_plus_i_l272_272652


namespace marco_wins_with_optimal_strategy_l272_272299

theorem marco_wins_with_optimal_strategy
  (m : ℕ)
  (h1 : 9 < m)
  (initial_choice : ℝ)
  (h2 : 0 ≤ initial_choice ∧ initial_choice ≤ m)
  (valid_choice : ∀ x y : ℝ, |x - y| ≥ 1.5) :
  ∃ strategy_for_marco : (ℕ → ℝ) → bool,
    ∀ strategy_for_lisa : (ℕ → ℝ) → bool,
      marco_wins strategy_for_marco strategy_for_lisa :=
sorry

end marco_wins_with_optimal_strategy_l272_272299


namespace exists_square_with_digit_sum_2002_l272_272718

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_square_with_digit_sum_2002 :
  ∃ (n : ℕ), sum_of_digits (n^2) = 2002 :=
sorry

end exists_square_with_digit_sum_2002_l272_272718


namespace hyperbola_vertex_distance_l272_272107

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

-- Statement: The distance between the vertices of the hyperbola is 1
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 2 * (1 / 2) = 1 :=
by
  intros x y H
  sorry

end hyperbola_vertex_distance_l272_272107


namespace some_number_value_l272_272151

theorem some_number_value (x : ℕ) (some_number : ℕ) : x = 5 → ((x / 5) + some_number = 4) → some_number = 3 :=
by
  intros h1 h2
  sorry

end some_number_value_l272_272151


namespace asymptotic_stability_l272_272858

theorem asymptotic_stability (f : ℝ → ℝ → ℝ) (x : ℝ → ℝ) :
  (∀ t, (deriv x t = 1 + t - x t)) → 
  (x 0 = 0) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x0 < δ, ∀ t ≥ 0, abs (x0 * exp (-t)) < ε) → 
  (tendsto (λ t, abs (x t - t)) at_top (nhds 0)) → 
  ∃ t, (x t = t ∧ (tendsto (λ t, abs (x t - t)) at_top (nhds 0))) := 
by 
  sorry

end asymptotic_stability_l272_272858


namespace part1_part2_l272_272939

noncomputable section

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 4 ↔ f x a < 4 - 2 * a) →
  a = 0 := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x 1 - f (-2 * x) 1 ≤ x + m) →
  2 ≤ m :=
sorry

end part1_part2_l272_272939


namespace radius_equals_6_sqrt_2_l272_272175

-- Define the relevant geometric entities and properties
variables (A B C D O E : ℝ) (side length : ℝ)
-- side length of the square
def side : Prop := side length = 12
def is_square : Prop := ∀ (x y : ℝ), (x - y)^2 = 12^2
def passes_through_A_B : Prop := passes_through O A ∧ passes_through O B
def is_tangent_to_CD : Prop := tangent_to O CD

-- Define the requirement in terms of the radius
def radius_of_circle : ℝ := 6 * real.sqrt 2
def circle_radius_condition : Prop := ∃ (E : ℝ), E = radius_of_circle

-- The proof statement translating to the original problem: proving the radius
theorem radius_equals_6_sqrt_2 (A B C D O : ℝ) (side_length : ℝ) :
  is_square A B C D side_length →
  passes_through_A_B A B O →
  is_tangent_to_CD O →
  circle_radius_condition side_length :=
sorry

end radius_equals_6_sqrt_2_l272_272175


namespace germs_per_dish_l272_272254

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l272_272254


namespace shift_sine_graph_l272_272810

theorem shift_sine_graph : ∀ x, 3 * sin (2 * x - real.pi / 4) = 3 * sin (2 * (x - real.pi / 8)) :=
by
  intro x
  -- proof goes here
  sorry

end shift_sine_graph_l272_272810


namespace count_positive_integers_satisfying_condition_l272_272577

-- Definitions
def is_between (x: ℕ) : Prop := 30 < x^2 + 8 * x + 16 ∧ x^2 + 8 * x + 16 < 60

-- Theorem statement
theorem count_positive_integers_satisfying_condition :
  {x : ℕ | is_between x}.card = 2 := 
sorry

end count_positive_integers_satisfying_condition_l272_272577


namespace find_AD_l272_272703

variable (A B C D E F : Type) [Trapezoid A B C D]
variable (h1 : Diagonal A C = 1)
variable (h2 : Height_Of_Trapezoid A C)
variable (h3 : Perpendicular A E C D)
variable (h4 : Perpendicular C F A B)
variable (h5 : Side A D = Side C F)
variable (h6 : Side B C = Side C E)

theorem find_AD : Side A D = Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry

end find_AD_l272_272703


namespace trapezoid_AD_value_l272_272685

theorem trapezoid_AD_value (ABCD is a trapezoid) 
  (AC_height : ∀ (A C ∈ ABCD), ∃ (h : ℝ), AC = h ∧ h = 1)
  (AD_eq_CF : AD = CF) 
  (BC_eq_CE : BC = CE)
  (AE_perp_CD : ∀ (A E C D ∈ ABCD), is_perpendicular AE CD)
  (CF_perp_AB : ∀ (C F A B ∈ ABCD), is_perpendicular CF AB) 
  : AD = sqrt (sqrt (2) - 1) := 
sorry

end trapezoid_AD_value_l272_272685


namespace range_of_a_l272_272292

-- Definitions for propositions
def p (a : ℝ) : Prop :=
  (1 - 4 * (a^2 - 6 * a) > 0) ∧ (a^2 - 6 * a < 0)

def q (a : ℝ) : Prop :=
  (a - 3)^2 - 4 ≥ 0

-- Proof statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (a ≤ 0 ∨ 1 < a ∧ a < 5 ∨ a ≥ 6) :=
by 
  sorry

end range_of_a_l272_272292


namespace arrangement_count_l272_272402

theorem arrangement_count (m f : ℕ) (h_m : m = 5) (h_f : f = 4) :
  let gaps := m + 1 in
  m! * (gaps.choose f) = nat.factorial m * (nat.choose gaps f) :=
by
  sorry

end arrangement_count_l272_272402
