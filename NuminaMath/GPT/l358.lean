import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.NumInstances.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Real.Sqrt
import Mathlib.Analysis.Calculus.Monotonic
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Graph.Coloring
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Enumerate
import Mathlib.LinearAlgebra.QuadraticForm.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Numbertheory.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.SetTheory.Set.Basic
import Mathlib.Tactic

namespace dave_deleted_apps_l358_358123

-- Definitions based on problem conditions
def original_apps : Nat := 16
def remaining_apps : Nat := 5

-- Theorem statement for proving how many apps Dave deleted
theorem dave_deleted_apps : original_apps - remaining_apps = 11 :=
by
  sorry

end dave_deleted_apps_l358_358123


namespace sum_of_digits_base2_315_l358_358763

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358763


namespace find_a4_l358_358266

variable (a_1 d : ℝ)

def a_n (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

axiom condition1 : (a_n a_1 d 2 + a_n a_1 d 6) / 2 = 5 * Real.sqrt 3
axiom condition2 : (a_n a_1 d 3 + a_n a_1 d 7) / 2 = 7 * Real.sqrt 3

theorem find_a4 : a_n a_1 d 4 = 5 * Real.sqrt 3 :=
by
  -- Proof should go here, but we insert "sorry" to mark it as incomplete.
  sorry

end find_a4_l358_358266


namespace expected_participants_2003_l358_358269

noncomputable def participants (year : ℕ) : ℝ :=
  if year = 1998 then 500 else participants (year - 1) * 1.6

theorem expected_participants_2003 :
  participants 2003 = 5243 :=
by
  sorry

end expected_participants_2003_l358_358269


namespace tan_x_solution_l358_358620

theorem tan_x_solution (a b : ℝ) (x : ℝ) 
  (hx1: tan x = a / b) 
  (hx2: tan (2 * x) = (b + 1) / (a + b)) : 
  tan x = 1 / 3 :=
sorry

end tan_x_solution_l358_358620


namespace sufficient_not_necessary_for_inverse_l358_358693

def f (x b : ℝ) : ℝ := x^2 - 2 * b * x

theorem sufficient_not_necessary_for_inverse (b : ℝ) :
  (∀ x ∈ set.Ici (1 : ℝ), ∀ y ∈ set.Ici (1 : ℝ), f x b = f y b → x = y) →
  b < 1 :=
sorry

end sufficient_not_necessary_for_inverse_l358_358693


namespace quadrangular_prism_volume_l358_358924

theorem quadrangular_prism_volume
  (perimeter : ℝ)
  (side_length : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (H1 : perimeter = 32)
  (H2 : side_length = perimeter / 4)
  (H3 : height = side_length)
  (H4 : volume = side_length * side_length * height) :
  volume = 512 := by
    sorry

end quadrangular_prism_volume_l358_358924


namespace find_principal_l358_358049

-- Define the conditions
variables (P R : ℝ)
-- The principal (P) was put at a simple interest rate (R) for 5 years.
-- If it had been put at a 2% higher rate, it would have fetched Rs. 250 more.

-- Given condition as equations:
def si_original : ℝ := (P * R * 5) / 100
def si_increased : ℝ := (P * (R + 2) * 5) / 100
def condition : Prop := si_increased = si_original + 250

-- Proof objective
theorem find_principal (h : condition P R) : P = 2500 :=
sorry

end find_principal_l358_358049


namespace percentage_increase_correct_l358_358421

-- Defining the initial and final numbers
def initial_number : ℝ := 1500
def final_number : ℝ := 1800

-- Defining the percentage increase calculation
def percentage_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

-- Theorem statement that the percentage increase is 20%
theorem percentage_increase_correct : percentage_increase initial_number final_number = 20 :=
  sorry

end percentage_increase_correct_l358_358421


namespace gcd_36_54_l358_358721

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l358_358721


namespace cone_rotation_ratio_l358_358827

theorem cone_rotation_ratio (r h : ℝ) 
  (h_pos : 0 < h) (r_pos : 0 < r)
  (rotations : 15) 
  (arc_circumference : 30 * Real.pi * r) :
  ∃ (m n : ℕ), (h / r = m * Real.sqrt n) ∧ (m + n = 18) := by
  sorry

end cone_rotation_ratio_l358_358827


namespace find_two_digit_number_l358_358922

theorem find_two_digit_number : ∃ A B : ℕ, A ≠ B ∧ 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧
  let n := 10 * A + B in
  10 ≤ n ∧ n ≤ 99 ∧ n^2 = (A + B)^3 ∧ n = 27 :=
by
  sorry

end find_two_digit_number_l358_358922


namespace second_tap_emptying_time_l358_358064

-- Definitions
def fill_rate_first_tap : ℝ := 1 / 5
def net_fill_rate : ℝ := 1 / 13.333333333333332

-- Statement: Prove the emptying rate of the second tap
theorem second_tap_emptying_time
  (F : ℝ) (E : ℝ)
  (h1 : F = fill_rate_first_tap)
  (h2 : F - E = net_fill_rate) :
  E = 1 / 8 :=
by sorry

end second_tap_emptying_time_l358_358064


namespace gcd_of_36_and_54_l358_358726

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l358_358726


namespace mass_percentage_C_in_CO_l358_358147

theorem mass_percentage_C_in_CO :
  let atomic_mass_C := 12.01
  let atomic_mass_O := 16.00
  let molecular_mass_CO := atomic_mass_C + atomic_mass_O
  (atomic_mass_C / molecular_mass_CO) * 100 ≈ 42.91 :=
by
  sorry

end mass_percentage_C_in_CO_l358_358147


namespace correctness_of_statements_l358_358400

open Classical

-- Definitions based on conditions
def proposition_p : Prop := ∃ x : ℝ, 2^x = 1
def statement1 : Prop := ¬proposition_p = ∃ x : ℝ, 2^x ≠ 1
def line_parallel (l m : Type) [linear_order l] [linear_order m] : Prop := sorry  -- specify parallel definition in Lean
def plane_containment (l : Type) (α : Type) : Prop := sorry  -- specify containment relation in Lean
def statement2 (l m α : Type) [linear_order l] [linear_order m] [linear_order α] : Prop :=
  line_parallel l m ∧ line_parallel m α → line_parallel l α

-- Probability calculation
def uniform_random_variable_between_0_and_1 : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}
def event_3a_minus_1_gt_0 (a : ℝ) := 3 * a - 1 > 0
def probability_3a_minus_1_gt_0 : ℝ := ∫ (fun a : ℝ => if event_3a_minus_1_gt_0 a then 1 else 0) uniform_random_variable_between_0_and_1

def statement3 : Prop := probability_3a_minus_1_gt_0 = 2/3

-- Inequality property
def statement4 (a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 → (a/b + b/a ≥ 2)) ∧ (¬(a > 0 ∧ b > 0) → a/b + b/a ≥ 2)

-- Combined statement for correctness of statements 3 and 4, falsity of 1 and 2
def math_problem : Prop :=
  ¬statement1 ∧ ¬statement2 (Type) (Type) (Type) ∧ statement3 ∧ statement4 ∃ x : ℝ, 2^x = 1.

theorem correctness_of_statements : math_problem := sorry

end correctness_of_statements_l358_358400


namespace sum_of_digits_base2_315_l358_358757

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358757


namespace moon_nighttime_temperature_l358_358670

theorem moon_nighttime_temperature :
  ∀ (daytime_temp nighttime_temp : ℝ),
  daytime_temp = 126 → nighttime_temp = -150 →
  (nighttime_temp = -150) :=
by
  intros daytime_temp nighttime_temp h_day h_night
  rw h_night
  rfl

end moon_nighttime_temperature_l358_358670


namespace union_M_N_l358_358947

def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

theorem union_M_N : M ∪ N = {x | x < -5 ∨ x > -3} := sorry

end union_M_N_l358_358947


namespace find_third_root_l358_358869

-- Define the polynomial
def poly (a b x : ℚ) : ℚ := a * x^3 + 2 * (a + b) * x^2 + (b - 2 * a) * x + (10 - a)

-- Define the roots condition
def is_root (a b x : ℚ) : Prop := poly a b x = 0

-- Given conditions and required proof
theorem find_third_root (a b : ℚ) (ha : a = 350 / 13) (hb : b = -1180 / 13) :
  is_root a b (-1) ∧ is_root a b 4 → 
  ∃ r : ℚ, is_root a b r ∧ r ≠ -1 ∧ r ≠ 4 ∧ r = 61 / 35 :=
by sorry

end find_third_root_l358_358869


namespace math_problem_proof_l358_358958

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ) (a : ℝ), 
    (a < 0) ∧ (θ ≠ 0) ∧
    (∃ (P : ℝ × ℝ), P = (4 * a, 3 * a)) ∧ 
    (sin θ = -3 / 5) ∧ (cos θ = -4 / 5) ∧
    (1 + 2 * sin (π + θ) * cos (2023 * π - θ)) / (sin (π / 2 + θ)^2 - cos (5 * π / 2 - θ)^2) = 7

theorem math_problem_proof : problem_statement := 
  by sorry

end math_problem_proof_l358_358958


namespace number_without_digit_two_323_l358_358983

def contains_digit_two (n : Nat) : Bool :=
  (toString n).any (λ ch, ch = '2')

def count_numbers_without_digit_two (n : Nat) : Nat :=
  (List.range n).count (λ k, ¬ contains_digit_two k)

theorem number_without_digit_two_323 :
  count_numbers_without_digit_two 500 = 323 :=
by
  sorry

end number_without_digit_two_323_l358_358983


namespace sum_of_three_digit_numbers_l358_358031

-- Define the arithmetic sum of natural numbers from 100 to 999
theorem sum_of_three_digit_numbers : (Nat.sum (Finset.range 999 \ Finset.range 100)) = 494550 := 
by sorry

end sum_of_three_digit_numbers_l358_358031


namespace inequality_true_l358_358951

variables {a b : ℝ}
variables (c : ℝ)

theorem inequality_true (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 :=
by sorry

end inequality_true_l358_358951


namespace distributor_profit_percentage_l358_358815

theorem distributor_profit_percentage 
  (P : ℝ) (C : ℝ) (D : ℝ) (S : ℝ) (Π : ℝ) (Π_perc : ℝ)
  (hP : P = 28.5)
  (hC : C = 0.2)
  (hD : D = 19)
  (hS : S = P / (1 - C))
  (hΠ : Π = S - D)
  (hΠ_perc : Π_perc = (Π / D) * 100) :
  Π_perc = 87.5 :=
by
  sorry

end distributor_profit_percentage_l358_358815


namespace distance_to_y_axis_eq_reflection_across_x_axis_eq_l358_358674

-- Definitions based on the conditions provided
def point_P : ℝ × ℝ := (4, -2)

-- Statements we need to prove
theorem distance_to_y_axis_eq : (abs (point_P.1) = 4) := 
by
  sorry  -- Proof placeholder

theorem reflection_across_x_axis_eq : (point_P.1 = 4 ∧ -point_P.2 = 2) :=
by
  sorry  -- Proof placeholder

end distance_to_y_axis_eq_reflection_across_x_axis_eq_l358_358674


namespace range_of_a_l358_358298

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1 / 2) * x - 1
  else 1 / x

theorem range_of_a (a : ℝ) : f a > a → a ∈ set.Ioo (-1) 0 :=
sorry

end range_of_a_l358_358298


namespace num_true_props_equals_two_l358_358352

-- Definitions based on the conditions
def original_prop (x : ℝ) : Prop := (x = 2) → (x^2 + x - 6 = 0)
def converse_prop (x : ℝ) : Prop := (x^2 + x - 6 = 0) → (x = 2)
def contrapositive_prop (x : ℝ) : Prop := (¬ (x^2 + x - 6 = 0)) → (x ≠ 2)
def negation_prop (x : ℝ) : Prop := ¬ ((x = 2) → (x^2 + x - 6 = 0))

-- Theorem stating the number of true propositions
theorem num_true_props_equals_two :
  (original_prop 2 ∧ contrapositive_prop 2) ∧
  (¬ converse_prop 2 ∧ ¬ negation_prop 2) → 
  2 = 2 :=
by 
  intros,
  sorry

end num_true_props_equals_two_l358_358352


namespace carpet_area_l358_358127

theorem carpet_area (length_ft : ℕ) (width_ft : ℕ) (ft_per_yd : ℕ) (A_y : ℕ) 
  (h_length : length_ft = 15) (h_width : width_ft = 12) (h_ft_per_yd : ft_per_yd = 9) :
  A_y = (length_ft * width_ft) / ft_per_yd := 
by sorry

#check carpet_area

end carpet_area_l358_358127


namespace probability_no_three_points_form_obtuse_l358_358511

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let uniform_distribution := (0 : ℝ) → Icc 0 (2 * π)
  let chosen_points := choose 4 uniform_distribution
  let obtuse_triangle_condition := 
      ∀ (i j k : ℕ) (h1 : i ≠ j) (h2 : j ≠ k) (h3 : i ≠ k),
      angle_at_center chosen_points[i] chosen_points[j] ≤ π / 2 
      ∧ angle_at_center chosen_points[j] chosen_points[k] ≤ π / 2 
      ∧ angle_at_center chosen_points[k] chosen_points[i] ≤ π / 2
  let probability := calculate_probability chosen_points obtuse_triangle_condition
  probability

theorem probability_no_three_points_form_obtuse : probability_no_obtuse_triangle = π^2 / 64 := 
sorry

end probability_no_three_points_form_obtuse_l358_358511


namespace quadratic_inequality_solution_l358_358472

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - 4 * x - 21 < 0) ↔ (-3 < x ∧ x < 7) :=
sorry

end quadratic_inequality_solution_l358_358472


namespace total_wicks_is_20_l358_358454

variable (x y : ℕ)
variable (six_wick twelve_wick total_wick : ℕ)
variable (total_length : ℕ)
variable (spool_length ft_to_inch : ℕ)

-- Condition that Amy has a 15-foot spool of string.
def spool_length := 15

-- Definition to convert feet to inches.
def ft_to_inch := 12
def total_length := spool_length * ft_to_inch

-- Condition that the total string is cut into equal number of 6-inch and 12-inch wicks
def six_wick := 6
def twelve_wick := 12

-- Condition that the total length of string used matches the total length of spool.
def length_equality := six_wick * x + twelve_wick * y = total_length

-- Condition that Amy cuts an equal number of 6-inch and 12-inch wicks.
def equal_wicks := x = y

-- Given all conditions, we need to prove the total number of wicks.
theorem total_wicks_is_20 (h1 : length_equality) (h2 : equal_wicks) : total_wick = 20 :=
by {
  -- Proof steps to be filled in later.
  sorry
}

end total_wicks_is_20_l358_358454


namespace pentagon_isosceles_triangle_l358_358371

theorem pentagon_isosceles_triangle 
    (α β : ℝ)
    (h1 : 3 * α + 2 * β = 540)
    (h2 : ∀ A B C D E : ℝ, A = α ∨ A = β ∧ B = α ∨ B = β ∧ C = α ∨ C = β ∧ D = α ∨ D = β ∧ E = α ∨ E = β):
  ∃ θ : ℝ, θ ∈ {α, β} ∧
  (  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (a = b ∨ b = c ∨ c = a)) := 
sorry

end pentagon_isosceles_triangle_l358_358371


namespace students_qualifying_percentage_l358_358422

theorem students_qualifying_percentage (N B G : ℕ) (boy_percent : ℝ) (girl_percent : ℝ) :
  N = 400 →
  G = 100 →
  B = N - G →
  boy_percent = 0.60 →
  girl_percent = 0.80 →
  (boy_percent * B + girl_percent * G) / N * 100 = 65 :=
by
  intros hN hG hB hBoy hGirl
  simp [hN, hG, hB, hBoy, hGirl]
  sorry

end students_qualifying_percentage_l358_358422


namespace kids_on_lake_pleasant_l358_358702

theorem kids_on_lake_pleasant (K : ℕ) : 
  (K / 8 = 5) → (K = 40) := 
begin
  -- Assume the given condition
  intro h,
  -- We need to prove K = 40
  sorry
end

end kids_on_lake_pleasant_l358_358702


namespace number_of_samples_from_retired_l358_358880

def ratio_of_forms (retired current students : ℕ) : Prop :=
retired = 3 ∧ current = 7 ∧ students = 40

def total_sampled_forms := 300

theorem number_of_samples_from_retired :
  ∃ (xr : ℕ), ratio_of_forms 3 7 40 → xr = (300 / (3 + 7 + 40)) * 3 :=
sorry

end number_of_samples_from_retired_l358_358880


namespace triangle_area_min_area_BDC_l358_358081

noncomputable def triangle {α : Type} [LinearOrderedField α] 
  (A B C : Point α) : ℚ := 25 * Real.sqrt 3

theorem triangle_area (A B C : Point α) 
    (bisector : Line α) (A L : Point α) (D: Point α)
    (is_on_bisector : D ∈ line_extends (segment AB bisector))
    (AD_eq_10 : real AD = 10)
    (angle_BDC_eq_60 : angle ABC D = 60) :
    triangle A B C = 25 * Real.sqrt 3 :=
sorry

theorem min_area_BDC (A B C : Point α) 
    (bisector : Line α) (A L : Point α) (D: Point α)
    (is_on_bisector : D ∈ line_extends (segment AB bisector))
    (AD_eq_10 : real AD = 10)
    (angle_BDC_eq_60 : angle ABC D = 60) :
    triangle B D C := 75 * Real.sqrt 3 :=
sorry

end triangle_area_min_area_BDC_l358_358081


namespace find_min_n_l358_358417

theorem find_min_n (k : ℕ) : ∃ n, 
  (∀ (m : ℕ), (k = 2 * m → n = 100 * (m + 1)) ∨ (k = 2 * m + 1 → n = 100 * (m + 1) + 1)) ∧
  (∀ n', (∀ (m : ℕ), (k = 2 * m → n' ≥ 100 * (m + 1)) ∨ (k = 2 * m + 1 → n' ≥ 100 * (m + 1) + 1)) → n' ≥ n) :=
by {
  sorry
}

end find_min_n_l358_358417


namespace proof_of_sum_correct_l358_358230

noncomputable def proof_of_sum : Prop :=
  ∀ (x y z : ℝ), 
  x > 0 → y > 0 → z > 0 → 
  (x * y = 32) → (x * z = 64) → (y * z = 96) → 
  x + y + z = (44 * real.sqrt 3) / 3

theorem proof_of_sum_correct : proof_of_sum :=
by
  intros x y z hx hy hz hxy hxz hyz
  sorry

end proof_of_sum_correct_l358_358230


namespace mika_bought_stickers_l358_358637

noncomputable def stickers_mika_bought 
    (initial: ℝ) (birthday: ℝ) (sister: ℝ) (mother: ℝ) (total: ℝ): ℝ :=
  let received := birthday + sister + mother
  in total - received

theorem mika_bought_stickers 
    (initial: ℝ) (birthday: ℝ) (sister: ℝ) (mother: ℝ) (total: ℝ): 
  initial = 20 ∧ birthday = 20 ∧ sister = 6 ∧ mother = 58 ∧ total = 130 → 
  stickers_mika_bought initial birthday sister mother total = 46 :=
by
  intro h
  obtain ⟨h_init, h_bday, h_sis, h_mom, h_total⟩ := h
  simp [stickers_mika_bought, h_init, h_bday, h_sis, h_mom, h_total] at *
  sorry

end mika_bought_stickers_l358_358637


namespace molecular_weight_boric_acid_l358_358029

theorem molecular_weight_boric_acid :
  let H := 1.008  -- atomic weight of Hydrogen in g/mol
  let B := 10.81  -- atomic weight of Boron in g/mol
  let O := 16.00  -- atomic weight of Oxygen in g/mol
  let H3BO3 := 3 * H + B + 3 * O  -- molecular weight of H3BO3
  H3BO3 = 61.834 :=  -- correct molecular weight of H3BO3
by
  sorry

end molecular_weight_boric_acid_l358_358029


namespace hexagon_perimeter_is_42_l358_358790

-- Define the side length of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) : ℕ :=
  num_sides * side_length

-- The theorem to prove
theorem hexagon_perimeter_is_42 : hexagon_perimeter side_length num_sides = 42 :=
by
  sorry

end hexagon_perimeter_is_42_l358_358790


namespace solution_set_of_inequality_l358_358361

theorem solution_set_of_inequality :
  {x : ℝ | -1 < x ∧ x < 2} = {x : ℝ | (x - 2) / (x + 1) < 0} :=
sorry

end solution_set_of_inequality_l358_358361


namespace fg_at_3_l358_358993

-- Define the functions f and g according to the conditions
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2)^2

theorem fg_at_3 : f (g 3) = 103 :=
by
  sorry

end fg_at_3_l358_358993


namespace modulo_inverse_5_mod_23_l358_358910

/-- The modular inverse of 5 modulo 23 is 14. -/
theorem modulo_inverse_5_mod_23 : ∃ b : ℕ, b ∈ set.Ico 0 23 ∧ (5 * b) % 23 = 1 :=
  by
  use 14
  split
  · exact Nat.le_of_lt (by norm_num : (14 < 23))
  · change (5 * 14) % 23 = 1
    norm_num
    sorry

end modulo_inverse_5_mod_23_l358_358910


namespace magnitude_of_z_l358_358135

-- Definitions based on conditions
def real_part := (7 / 4 : ℝ)
def imag_part := (-3 : ℝ)
def z := real_part + imag_part * I
def magnitude (x : ℂ) : ℝ := complex.abs x

-- Statement of the proof problem
theorem magnitude_of_z : magnitude z = (real.sqrt 193) / 4 :=
by sorry

end magnitude_of_z_l358_358135


namespace percent_not_local_politics_l358_358137

variables (total_reporters : ℕ) (perc_local_politics : ℝ) (perc_no_politics : ℝ)
variables (num_local_politics : ℕ) (num_no_politics : ℕ) (num_politics : ℕ) (num_not_local_politics: ℕ)

def percentage_of_reporters_not_local 
  (total_reporters = 100) 
  (perc_local_politics = 0.18) 
  (perc_no_politics = 0.70) 
  (num_local_politics : = total_reporters * perc_local_politics) 
  (num_no_politics : = total_reporters * perc_no_politics) 
  (num_politics : = total_reporters - num_no_politics) 
  (num_not_local_politics : = num_politics - num_local_politics) 
  : ℝ := (num_not_local_politics / num_politics) * 100

theorem percent_not_local_politics : 
  percentage_of_reporters_not_local total_reporters perc_local_politics perc_no_politics num_local_politics num_no_politics num_politics num_not_local_politics = 40 :=
sorry

end percent_not_local_politics_l358_358137


namespace gcd_36_54_l358_358730

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l358_358730


namespace only_exprD_is_always_defined_l358_358100

variable (a x : ℝ)

def exprA := (x + a) / (|x| - 2)
def exprB := x / (2 * x + 1)
def exprC := (3 * x + 1) / (x^2)
def exprD := (x^2) / (2 * x^2 + 1)

theorem only_exprD_is_always_defined (h : ∀ x, |x| - 2 ≠ 0 ∧ 2 * x + 1 ≠ 0 ∧ x^2 ≠ 0 ∧ (2 * x^2 + 1) ≠ 0) : ∀ x : ℝ, exprD a x ≠ 0 :=
by
  intro x
  -- proof left as an exercise
  sorry

end only_exprD_is_always_defined_l358_358100


namespace sum_of_ages_is_59_l358_358281

variable (juliet maggie ralph nicky lucy lily alex : ℕ)

def juliet_age := 10
def maggie_age := juliet_age - 3
def ralph_age := juliet_age + 2
def nicky_age := ralph_age / 2
def lucy_age := ralph_age + 1
def lily_age := ralph_age + 1
def alex_age := lucy_age - 5

theorem sum_of_ages_is_59 :
  maggie_age + ralph_age + nicky_age + lucy_age + lily_age + alex_age = 59 :=
by
  let maggie := 7
  let ralph := 12
  let nicky := 6
  let lucy := 13
  let lily := 13
  let alex := 8
  show maggie + ralph + nicky + lucy + lily + alex = 59
  sorry

end sum_of_ages_is_59_l358_358281


namespace correct_statement_l358_358101

-- Definitions of the statements
def statement_A : Prop := ∀ (P : Type), ∀ (r : P → ℝ) (S : P → ℝ), ¬ ∃ p, r p = S p
def statement_B : Prop := ∀ (P : Type), ∀ (f : P → ℝ) (Q : P → Prop), (∀ p, Q p → f p ≠ 0) → false
def statement_C : Prop := ∃ (P : Type), ∃ (S : P → ℝ) (n m : ℕ), n < m ∧ S n < S m ∧ (∀ k, n < k ∧ k < m → S k = k)
def statement_D : Prop := ∀ (P : Type), ∃ (r : P → ℝ), ∀ p, r p = ρ ∧ ρ > 0 ∧ r p = 0 → false

-- Main theorem statement to prove that statement C is correct
theorem correct_statement : statement_C :=
by {
    sorry
}

end correct_statement_l358_358101


namespace find_m_value_l358_358568

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_m_value :
  let a := vec 4 4
  let b := λ m : ℝ, vec 5 m
  let c := vec 1 3
  let diff_vec := vec (a.1 - 2 * c.1) (a.2 - 2 * c.2)
  ∀ m : ℝ, dot_product diff_vec (b m) = 0 → m = 5 :=
by
  intros a b c diff_vec m hv
  -- proof steps can be filled here
  sorry

end find_m_value_l358_358568


namespace baron_munchausen_correct_l358_358108

theorem baron_munchausen_correct : 
  let original_rectangle := (15, 9) in
  ∃ methods : List (Nat × Nat) → (Nat × Nat) → Nat × Nat → ℕ, 
    (methods ((15, 5), (15 / 3, 9)) = 28) ∧ 
    (methods ((15, 3), (15, 9 / 3)) = 36) ∧ 
    (methods ((15, 0.75), (15, 9 - 0.75)) = 31.5) ∧ 
    (methods ((9, 5.25), (9, 15 - 5.25)) = 28.5) ∧ 
    True := 
by 
  sorry

end baron_munchausen_correct_l358_358108


namespace oranges_in_bin_l358_358090

theorem oranges_in_bin (initial_oranges thrown_out new_oranges : ℕ) (h1 : initial_oranges = 34) (h2 : thrown_out = 20) (h3 : new_oranges = 13) :
  (initial_oranges - thrown_out + new_oranges = 27) :=
by
  sorry

end oranges_in_bin_l358_358090


namespace mean_subtract_const_variance_subtract_const_l358_358996

variable {n : ℕ} (x : Fin n → ℝ) (a : ℝ)

noncomputable def mean (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

noncomputable def variance (x : Fin n → ℝ) : ℝ :=
  (∑ i, (x i - mean x) ^ 2) / n

theorem mean_subtract_const (x : Fin n → ℝ) (a : ℝ) :
  mean (fun i => x i - a) = mean x - a :=
by
  sorry

theorem variance_subtract_const (x : Fin n → ℝ) (a : ℝ) :
  variance (fun i => x i - a) = variance x :=
by
  sorry

end mean_subtract_const_variance_subtract_const_l358_358996


namespace result_of_operation_given_y_l358_358354

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem result_of_operation_given_y :
  ∀ (y : ℤ), y = 11 → operation y 10 = 90 :=
by
  intros y hy
  rw [hy]
  show operation 11 10 = 90
  sorry

end result_of_operation_given_y_l358_358354


namespace part_I_part_II_l358_358601

variable {α : Type*} [LinearOrderedField α]

-- Assume necesssary conditions
variables (A B C a b c : α)
variables (h_triangle_acute : A + B + C = π)
variables (h_b : b = (a / 2) * sin C)
variables (h_A_pos : 0 < A ∧ A < π / 2)
variables (h_B_pos : 0 < B ∧ B < π / 2)
variables (h_C_pos : 0 < C ∧ C < π / 2)

-- Part (I)
theorem part_I : (1 / tan A) + (1 / tan C) = (1 / 2) :=
by sorry

-- Part (II)
theorem part_II : Sup {tan B | 0 < A ∧ A < π / 2 ∧ A + B + C = π ∧ b = (a / 2) * sin C ∧ B = π - A - C} = (8 / 15) :=
by sorry

end part_I_part_II_l358_358601


namespace prime_angle_triangle_l358_358544

theorem prime_angle_triangle (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_sum : a + b + c = 180) : a = 2 ∨ b = 2 ∨ c = 2 :=
sorry

end prime_angle_triangle_l358_358544


namespace vertex_and_symmetry_decreasing_interval_l358_358355

def parabola := ∀ x : ℝ, -2 * x^2 + 8 * x - 6

theorem vertex_and_symmetry (x : ℝ) :
  (∃ h k : ℝ, parabola x = -2 * (x - h)^2 + k ∧ h = 2 ∧ k = 2) ∧
  (x - 2 = 0) :=
sorry

theorem decreasing_interval (x : ℝ) :
  (∀ (x1 x2 : ℝ), x1 ≥ 2 ∧ x2 ≥ 2 ∧ x1 < x2 → parabola x1 ≥ parabola x2) :=
sorry

end vertex_and_symmetry_decreasing_interval_l358_358355


namespace survey_not_suitable_for_sampling_l358_358846

theorem survey_not_suitable_for_sampling :
  (¬ suitable_for_sampling "reviewing typographical errors in an article") :=
by
  sorry

-- Definitions for conditions
def suitable_for_sampling (survey_type : String) : Prop :=
  match survey_type with
  | "understanding the accuracy of a batch of shells" => true
  | "investigating the internet usage of middle school students nationwide" => true
  | "reviewing typographical errors in an article" => false
  | "examining the growth status of a certain crop" => true
  | _ => false

end survey_not_suitable_for_sampling_l358_358846


namespace count_noncongruent_int_triangles_l358_358573

-- Define the conditions for the triangle
noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

-- Define noncongruent triangles
def noncongruent_int_triangles : Nat :=
  let triangles := (List.range (19 * 19)).map (λ idx, let (a, b, c) := (idx / 19 / 19, (idx / 19) % 19, idx % 19) in (a + 1, b + 1, c + 1))
  triangles.filter (λ ⟨a, b, c⟩, is_valid_triangle a b c) |>.dedupLength

theorem count_noncongruent_int_triangles :
  noncongruent_int_triangles = 15 :=
by
  sorry

end count_noncongruent_int_triangles_l358_358573


namespace largest_value_of_b_l358_358623

theorem largest_value_of_b :
  ∃ b : ℚ, (3 * b + 4) * (b - 2) = 9 * b ∧ b = (11 + Real.sqrt 217) / 6 :=
by
  use (11 + Real.sqrt 217) / 6
  split
  · sorry
  · rfl

end largest_value_of_b_l358_358623


namespace pattern_C_not_foldable_without_overlap_l358_358175

-- Define the four patterns, denoted as PatternA, PatternB, PatternC, and PatternD.
inductive Pattern
| A : Pattern
| B : Pattern
| C : Pattern
| D : Pattern

-- Define a predicate for a pattern being foldable into a cube without overlap.
def foldable_into_cube (p : Pattern) : Prop := sorry

theorem pattern_C_not_foldable_without_overlap : ¬ foldable_into_cube Pattern.C := sorry

end pattern_C_not_foldable_without_overlap_l358_358175


namespace total_age_l358_358045

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l358_358045


namespace prime_squared_difference_divisible_by_24_l358_358419

theorem prime_squared_difference_divisible_by_24 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) :
  24 ∣ (p^2 - q^2) :=
sorry

end prime_squared_difference_divisible_by_24_l358_358419


namespace smallest_next_divisor_l358_358634

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor_l358_358634


namespace conic_section_equation_l358_358877

def is_ellipse (f1 f2 : ℝ × ℝ) (c : ℝ) (p : ℝ × ℝ) : Prop :=
  (real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)) = c

theorem conic_section_equation :
  is_ellipse (2, 0) (-2, 0) 12 := sorry

end conic_section_equation_l358_358877


namespace union_of_A_and_B_l358_358512

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_A_and_B_l358_358512


namespace problem_solution_l358_358555

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else (1 / 2) ^ x

theorem problem_solution :
  {a : ℝ | f(a) = 1} = {0, 2} :=
by
  sorry

end problem_solution_l358_358555


namespace find_constant_term_l358_358195

/-- Theorem stating the relationship between the constant term 'c' in the equation 
q' = 3q + c and the given condition that (6')' equals 210. -/
theorem find_constant_term
  (c : ℝ)
  (h₁ : ∀ (q : ℝ), derivative (λ q, 3 * q + c) = (λ q', 3 * q'))
  (h₂ : derivative (λ q, 3 * 6 + c) = 210) : 
  c = 165 := 
begin
  sorry
end

end find_constant_term_l358_358195


namespace field_length_l358_358414

-- Definitions of the conditions
def pond_area : ℝ := 25  -- area of the square pond
def width_to_length_ratio (w l : ℝ) : Prop := l = 2 * w  -- length is double the width
def pond_to_field_ratio (pond_area field_area : ℝ) : Prop := pond_area = (1/8) * field_area  -- pond area is 1/8 of field area

-- Statement to prove
theorem field_length (w l : ℝ) (h1 : width_to_length_ratio w l) (h2 : pond_to_field_ratio pond_area (l * w)) : l = 20 :=
by sorry

end field_length_l358_358414


namespace find_two_digit_number_l358_358919

theorem find_two_digit_number :
  ∃ (n : ℕ),
  10 ≤ n ∧ n ≤ 99 ∧
  (let A := n / 10, B := n % 10 in A ≠ B ∧ n ^ 2 = (A + B) ^ 3) ∧
  n = 27 :=
by
  sorry

end find_two_digit_number_l358_358919


namespace domain_of_sqrt_x_plus_1_over_x_l358_358675

noncomputable def domain_of_function (f : ℝ → ℝ) : set ℝ :=
  { x : ℝ | ∃ y : ℝ, f x = y }

def f (x : ℝ) : ℝ :=
  real.sqrt (x + 1) / x

theorem domain_of_sqrt_x_plus_1_over_x :
  domain_of_function f = {x : ℝ | x >= -1} \ {0} :=
by
  sorry

end domain_of_sqrt_x_plus_1_over_x_l358_358675


namespace maximize_pmf_l358_358950

noncomputable def zeta : Type := sorry
instance : Distribution zeta where
  pmf : ℕ → ℝ := λ k, ((nat.choose 100 k) * (1 / 2)^100)

theorem maximize_pmf :
  ∃ k : ℕ, (∀ j : ℕ, pmf k ≥ pmf j) → k = 50 := by
  sorry

end maximize_pmf_l358_358950


namespace cubic_has_three_natural_roots_l358_358885

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l358_358885


namespace area_enclosed_by_intersecting_cylinders_l358_358791

noncomputable def A (k : ℝ) : ℝ :=
  ∫ x in 0..1, real.sqrt (1 - x^2) * real.sqrt (1 - k^2 * x^2)⁻¹

noncomputable def B (k : ℝ) : ℝ :=
  ∫ x in 0..1, real.sqrt (1 - x^2)⁻¹ * real.sqrt (1 - k^2 * x^2)⁻¹

noncomputable def C (k : ℝ) : ℝ :=
  ∫ x in 0..1, real.sqrt (1 - x^2)⁻¹ * real.sqrt (1 - k^2 * x^2)

def k (r2 r1 : ℝ) : ℝ :=
  r2 / r1

theorem area_enclosed_by_intersecting_cylinders (r1 r2 : ℝ) (h : r1 > r2) :
  let k := k r2 r1 in
  let S := 8 * r2^2 * A k - 8 * (r1^2 - r2^2) * B k in
  S = 8 * r2^2 * A k - 8 * (r1^2 - r2^2) * B k :=
by sorry

end area_enclosed_by_intersecting_cylinders_l358_358791


namespace seq_a_bound_l358_358206

def seq_a : ℕ → ℝ
| 0 => 1
| (n + 1) => seq_a n / (seq_a n ^ 3 + 1)

theorem seq_a_bound (n : ℕ) (hn : 0 < n) : seq_a n > 1 / (↑3 * n + Real.log n + 14/9) ^ (1/3) := by
  sorry

end seq_a_bound_l358_358206


namespace profit_ratio_l358_358668

noncomputable def lion_king_cost : ℝ := 10
noncomputable def lion_king_revenue : ℝ := 200
noncomputable def star_wars_cost : ℝ := 25
noncomputable def star_wars_revenue : ℝ := 405

noncomputable def lion_king_profit : ℝ :=
  lion_king_revenue - lion_king_cost

noncomputable def star_wars_profit : ℝ :=
  star_wars_revenue - star_wars_cost

theorem profit_ratio :
  (lion_king_profit / star_wars_profit) = (1 / 2) :=
  by 
    sorry

end profit_ratio_l358_358668


namespace part1_part2_l358_358553

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part1 (a : ℝ) (x : ℝ) (hx1 : 1 ≤ x) (hx2 : x ≤ Real.exp 1) :
  a = 1 →
  (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 + (Real.exp 1)^2 / 2) ∧ (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 / 2) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part2 (a : ℝ) :
  (-1/2 ≤ a ∧ a ≤ 1/2) ↔
  ∀ x, 1 < x → g a x < 0 :=
sorry

end part1_part2_l358_358553


namespace range_of_g_l358_358476

def g (A : ℝ) : ℝ :=
  (sin A * (5 * (cos A)^2 + (cos A)^6 + 2 * (sin A)^2 + 2 * (sin A)^4 * (cos A)^2)) /
  (tan A * (sec A - 2 * sin A * tan A))

theorem range_of_g :
  ∀ A : ℝ, (A ≠ nπ / 2) → (g A > 5 ∧ ∀ y, y > 5 → ∃ A : ℝ, g A = y) :=
begin
  sorry -- proof is not required
end

end range_of_g_l358_358476


namespace solution_set_f_greater_than_xf1_l358_358539

variable {f : ℝ → ℝ}

-- Define the conditions
axiom h1 : ∀ x > 0, f(x) = ∂² (f x)
axiom h2 : ∀ x > 0, f(x) > x * ∂² (f x)

-- Define the theorem statement
theorem solution_set_f_greater_than_xf1 (f : ℝ → ℝ) (h1 : ∀ x > 0, f(x) = ∂² (f x)) (h2 : ∀ x > 0, f(x) > x * ∂² (f x)) : 
  {x | x > 0 ∧ f(x) > x * f(1)} = {x | 0 < x ∧ x < 1} := 
sorry

end solution_set_f_greater_than_xf1_l358_358539


namespace common_ratio_is_2_l358_358619

noncomputable def common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : ℝ :=
(a1 + 2 * d) / a1

theorem common_ratio_is_2 (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : 
    common_ratio a1 d h1 h2 = 2 :=
by
  -- Proof would go here
  sorry

end common_ratio_is_2_l358_358619


namespace quadratic_min_value_correct_l358_358158

theorem quadratic_min_value_correct : ∀ x : ℝ, ∃ y_min : ℝ, y_min = 2 ∧ (∃ x0 : ℝ, y_min = (x0 - 1)^2 + 2) :=  
by {
  intro x,
  use 2,
  split,
  { reflexivity, },
  { use 1,
    reflexivity, }
}

end quadratic_min_value_correct_l358_358158


namespace maximum_value_of_g_l358_358469

def g : ℕ+ → ℕ
| ⟨n, h⟩ := if hlt : n < 12 then n + 12 else g ⟨n - 7, Nat.sub_pos_of_lt hlt⟩ + 3

theorem maximum_value_of_g : ∃ m : ℕ+, g m = 26 := sorry

end maximum_value_of_g_l358_358469


namespace ratio_areas_ABR_ACS_eq_one_l358_358644

theorem ratio_areas_ABR_ACS_eq_one 
    {A B C M D P Q R S : Type}
    [triangle : ∀ (t : Type), t = triangle ABC]
    [obtuse_angle_C : ¬acute C]
    [point_M_on_BC : on_segment M BC]
    [acute_triangle_BCD : ∀(BCD : Type), BCD = triangle BCD ∧ acute ∠BCD]
    [opposite_sides_AD : opposite_sides A D (line BC)]
    [circumscribed_BMD {ω_B} : circumscribed ω_B (triangle BMD)]
    [circumscribed_CMD {ω_C} : circumscribed ω_C (triangle CMD)]
    [AB_intersects_ωB_at_P : AB ∩ ω_B = {P}]
    [AC_intersects_ωC_at_Q : ray AC ∩ ω_C = {Q}]
    [PD_intersects_ωC_at_R : segment PD ∩ ω_C = {R}]
    [QD_intersects_ωB_at_S : ray QD ∩ ω_B = {S}] :
  area (triangle ABR) / area (triangle ACS) = 1 := 
sorry

end ratio_areas_ABR_ACS_eq_one_l358_358644


namespace second_group_work_days_l358_358581

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l358_358581


namespace probability_P8_eq_P_l358_358080

-- Definition of the problem
def P_center_of_square (P : Point) (A B C D : Point) : Prop :=
  is_center P (square A B C D)

def random_reflection_sequence (P : Point) (A B C D : Point) (n : ℕ) : Prop :=
  ∀ i, i < n → reflect_over_side P (square A B C D)

-- The main theorem we want to prove
theorem probability_P8_eq_P (P : Point) (A B C D : Point) (P_center : P_center_of_square P A B C D) :
  let prob : ℚ := 1225 / 16384
  in random_reflection_sequence P A B C D 8 → P_8 = P → prob
:= by
  sorry

end probability_P8_eq_P_l358_358080


namespace sum_of_binary_digits_of_315_l358_358749

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358749


namespace determine_a_k_and_max_profit_l358_358959

-- Assume a and k are real numbers, and k is less than 0.
-- Also assume functions y1 and y2 for each interval of x.

def daily_sales_volume (x : ℝ) (a : ℝ) (k : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then a * (x - 4) ^ 2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then k * x + 7
  else 0

-- Conditions
def condition1 := daily_sales_volume 3 1 (-1) = 4
def condition2 := (∃ x : ℝ, 3 < x ∧ x ≤ 5 ∧ daily_sales_volume x 1 (-1) = 2)

-- Proof goals
theorem determine_a_k_and_max_profit :
  ∀ (x : ℝ), 1 < x ∧ x ≤ 5 →
  (daily_sales_volume x 1 (-1) = if 1 < x ∧ x ≤ 3 then (x - 4) ^ 2 + 6 / (x - 1) else -x + 7) ∧
  x = 2 :=
  
begin
  assume x hx,
  split,
  {
    by_cases (1 < x ∧ x ≤ 3),
    {
      simp [daily_sales_volume, h],
    },
    {
      by_cases (3 < x ∧ x ≤ 5),
      {
        simp [daily_sales_volume, h],
      },
      {
        exfalso,
        linarith,
      }
    }
  },
  sorry
end

end determine_a_k_and_max_profit_l358_358959


namespace PQ_over_F1F2_l358_358848

-- Definitions based on conditions
def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def equilateral_triangle (P Q R : ℝ × ℝ) := 
  let s := dist P Q in
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- Conditions
constants (a b PQ F1F2 : ℝ)
constants (P Q R F1 F2 : ℝ × ℝ)
constants (foci_line_prl_x_axis : Prop)
constants (foci_cond: Prop)

-- The conditions in terms of definitions:
axiom PQR_on_ellipse : ellipse (P.1 - Q.1) (P.2 - Q.2) a b
axiom Q_coord : Q = (0, b)
axiom PR_parallel : foci_line_prl_x_axis -- Q is at (0, b) and ΠΡ is parallel to x-axis
axiom F1_on_QR : F1 = ((0, b) + R) / 2
axiom F2_on_PQ : F2 = (P + (0, b)) / 2
axiom dist_foci_equal : dist F1 F2 = 2

-- The actual proof problem
theorem PQ_over_F1F2 : @equilateral_triangle ℝ P Q R → 
                       a = 2 → 
                       b = sqrt 3 → 
                       PQR_on_ellipse → 
                       Q_coord → 
                       PR_parallel → 
                       F1_on_QR → 
                       F2_on_PQ → 
                       dist_foci_equal →
                       PQ / dist F1 F2 = 8 / 5 := 
by sorry

end PQ_over_F1F2_l358_358848


namespace gymnast_scores_difference_l358_358096

theorem gymnast_scores_difference
  (s1 s2 s3 s4 s5 : ℝ)
  (h1 : (s2 + s3 + s4 + s5) / 4 = 9.46)
  (h2 : (s1 + s2 + s3 + s4) / 4 = 9.66)
  (h3 : (s2 + s3 + s4) / 3 = 9.58)
  : |s5 - s1| = 8.3 :=
sorry

end gymnast_scores_difference_l358_358096


namespace cards_sum_probability_l358_358008

theorem cards_sum_probability :
  let Ω := Finset.product (Finset.range 6) (Finset.range 6)
  let favorable_outcomes := Ω.filter (λ (p : ℕ × ℕ), p.1 + p.2 > 7)
  (favorable_outcomes.card : ℝ) / (Ω.card : ℝ) = 1 / 6 :=
by
  let Ω := Finset.product (Finset.range 6) (Finset.range 6)
  let favorable_outcomes := Ω.filter (λ (p : ℕ × ℕ), p.1 + p.2 > 7)
  have h : (Ω.card : ℝ) = 36 := by simp only [Finset.card_product, Finset.card_range]; norm_num
  have favorable_cases : favorable_outcomes.card = 6 := by sorry
  rw [h, favorable_cases]
  simp
  norm_num

end cards_sum_probability_l358_358008


namespace find_a_of_complex_modulus_l358_358935

theorem find_a_of_complex_modulus (a : ℝ) (h : a < 0) (modulus : abs (complex.mk 3 a * complex.i) = 5) : a = -4 :=
sorry

end find_a_of_complex_modulus_l358_358935


namespace numbers_without_digit_2_1_to_500_l358_358980

def count_numbers_without_digit_2 (n : ℕ) : ℕ :=
  n.digits 10 |>.all (λ d, d ≠ 2)

theorem numbers_without_digit_2_1_to_500 : 
  (Finset.range 500).filter count_numbers_without_digit_2 |> Finset.card = 323 := 
by
  sorry

end numbers_without_digit_2_1_to_500_l358_358980


namespace intersection_of_S_and_T_l358_358209

def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | 0 < x}

theorem intersection_of_S_and_T : S ∩ T = {x | 1 ≤ x} := by
  sorry

end intersection_of_S_and_T_l358_358209


namespace derived_sequence_div_count_geq_l358_358624

-- Definitions for the sets and derived sequences
def derived_sequence (A : Finset ℤ) : Finset ℤ := 
  {d | ∃ (x y : ℤ), x ∈ A ∧ y ∈ A ∧ y > x ∧ d = y - x}.to_finset

def divisible_count (s : Finset ℤ) (m : ℤ) : ℕ := 
  s.count (λ x, x % m = 0)

-- The main theorem statement
theorem derived_sequence_div_count_geq (m n : ℤ) (A : Finset ℤ) :
  2 ≤ m → 2 ≤ n → A.card = n →
  divisible_count (derived_sequence A) m ≥ divisible_count (derived_sequence (Finset.range n)) m :=
  sorry

end derived_sequence_div_count_geq_l358_358624


namespace percentage_weight_loss_l358_358445

def weight_before_processing : ℝ := 892.31
def weight_after_processing : ℝ := 580.00

def weight_lost (weight_before weight_after : ℝ) : ℝ := weight_before - weight_after

theorem percentage_weight_loss :
  weight_lost weight_before_processing weight_after_processing / weight_before_processing * 100 ≈ 34.99 :=
by
  sorry

end percentage_weight_loss_l358_358445


namespace general_formula_T_bound_l358_358192

noncomputable def a (n : ℕ) : ℕ := 2 ^ n

def Sn (n : ℕ) : ℕ := (finset.range n).sum (λ k => a k)

def d (n : ℕ) : ℕ := a (n + 1) - a n

def T (n : ℕ) : ℝ := (finset.range n).sum (λ k => (k + 1 : ℝ) / (2 ^ (k + 1)))

theorem general_formula (n : ℕ) : a n = 2 ^ n :=
sorry

theorem T_bound (n : ℕ) : 1 ≤ T n ∧ T n < 3 :=
sorry

end general_formula_T_bound_l358_358192


namespace probability_of_multiples_l358_358450

-- Definitions based on problem conditions
def card_numbers : Finset ℕ := Finset.range 201

def is_multiple_of (n k : ℕ) : Prop := ∃ m, k = n * m

-- Main statement
theorem probability_of_multiples :
  (card (card_numbers.filter (λ k, is_multiple_of 4 k ∨ is_multiple_of 5 k ∨ is_multiple_of 7 k)) : ℕ) = 97 :=
sorry

end probability_of_multiples_l358_358450


namespace pizza_slice_angle_l358_358063

theorem pizza_slice_angle (A : ℝ) (θ : ℝ) (h : (θ / 360) * A / A = 1 / 8) : θ = 45 :=
by
  rw [div_mul_cancel (θ / 360) (ne_of_gt (real.pi_pos))] at h
  rw [div_eq_one_iff_eq] at h
  linarith
  norm_num
  simp at *
  rwa [mul_comm]
  sorry

end pizza_slice_angle_l358_358063


namespace domain_subsets_l358_358559

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := real.sqrt ((x - 1) / (x + 1))
def g (x a : ℝ) : ℝ := real.log ((x - a) * (x - 1))

-- Conditions for the domains
def A : set ℝ := {x | x < -1 ∨ x ≥ 1}
def B (a : ℝ) : set ℝ := {x | x > 1 ∨ x < a}

-- The problem statement
theorem domain_subsets (a : ℝ) (ha : a < 1) :
  B a ⊆ A ↔ a ≤ -1 :=
sorry

end domain_subsets_l358_358559


namespace increasing_if_and_only_if_l358_358197

noncomputable def f (x a : ℝ) : ℝ := Real.exp (abs (x - a))

theorem increasing_if_and_only_if (a : ℝ) : (∀ x ∈ Icc 1 (⊤ : ℝ), 0 ≤ (f x a).diff x) ↔ a ≤ 1 :=
by { sorry }

end increasing_if_and_only_if_l358_358197


namespace earnings_of_r_l358_358789

theorem earnings_of_r (P Q R : ℕ) (h1 : 9 * (P + Q + R) = 1710) (h2 : 5 * (P + R) = 600) (h3 : 7 * (Q + R) = 910) : 
  R = 60 :=
by
  -- proof will be provided here
  sorry

end earnings_of_r_l358_358789


namespace number_of_ways_to_distribute_66_coins_l358_358500

theorem number_of_ways_to_distribute_66_coins : ∃ n : ℕ, n = 315 ∧
  (∃ a b c : ℕ, 0 < a ∧ a < b ∧ b < c ∧ a + b + c = 66) :=
by sorry

end number_of_ways_to_distribute_66_coins_l358_358500


namespace dot_product_correct_l358_358117

open Matrix

def vec1 : Fin 3 → ℝ
| ⟨0, _⟩ := 3
| ⟨1, _⟩ := -2
| ⟨2, _⟩ := 4

def vec2 : Fin 3 → ℝ
| ⟨0, _⟩ := -1
| ⟨1, _⟩ := 5
| ⟨2, _⟩ := -3

theorem dot_product_correct : dotProduct vec1 vec2 = -25 :=
by
  sorry

end dot_product_correct_l358_358117


namespace total_marbles_l358_358223

theorem total_marbles (g : ℕ) (h : g = 36) : 
  ∃ t : ℕ, t = 72 :=
by
  have r : ℕ := (g / 3) * 1 -- red marbles
  have b : ℕ := (g / 3) * 2 -- blue marbles
  have t : ℕ := r + b + g -- total marbles
  use t
  calc 
    t = r + b + g : by sorry
    ... = (g / 3) * 1 + (g / 3) * 2 + g : by sorry
    ... = (36 / 3) * 1 + (36 / 3) * 2 + 36 : by rw [h]
    ... = 12 * 1 + 12 * 2 + 36 : by sorry
    ... = 12 + 24 + 36 : by sorry
    ... = 72 : by sorry

end total_marbles_l358_358223


namespace sum_of_digits_base2_315_l358_358755

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358755


namespace proof_CF_distance_l358_358894

noncomputable def distance_CF (A B D E F C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)

def a_b_sum_CF := 19

-- Square vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (17, 0)
def D : ℝ × ℝ := (17, 17)
def E : ℝ × ℝ := (0, 17)

-- Coordinates for points F and C based on given conditions
def F : ℝ × ℝ := (-120 / 17, 225 / 17)
def C : ℝ × ℝ := (17 + 120 / 17, 64 / 17)

theorem proof_CF_distance :
  distance_CF A B D E F C = 17 * Real.sqrt 2 ∧ a_b_sum_CF = 19 :=
by
  sorry

end proof_CF_distance_l358_358894


namespace probability_calc_l358_358853

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let pairs_count := 169
  let valid_pairs_count := 17
  1 - (valid_pairs_count / pairs_count : ℚ)

theorem probability_calc :
  probability_no_distinct_positive_real_roots = 152 / 169 := by sorry

end probability_calc_l358_358853


namespace round_9_6654_l358_358649

theorem round_9_6654 :
  (Float.roundToDecimal 9.6654 2 = 9.67) ∧ (Float.roundToInt 9.6654 = 10) := by
  sorry

end round_9_6654_l358_358649


namespace time_escalator_upwards_l358_358859

variables (L T_up T_down T_stationary x y: ℝ)

-- Let the length of the escalator be 1 unit (L = 1)
axiom length_escalator: L = 1

-- Running conditions when the escalator is not working.
-- It takes Vasya 6 minutes to run up and down.
axiom stationary_condition: T_stationary = 6

-- Running conditions when the escalator is moving down, taking 13.5 minutes.
axiom moving_down_condition: T_down = 13.5

-- Speed relationships: running down (x) and half up (x/2)
axiom speed_relationship: ∀ x y, T_stationary = (1 / x) + (2 / x)

-- Effective speeds and times with the escalator moving down.
axiom effective_speed_down: ∀ x y, T_down = (1 / (x + y)) + (1 / ((x / 2) - y))

-- The final time calculation when the escalator moves upward.
-- Result: It takes Vasya 324 seconds to run up and down.
theorem time_escalator_upwards (x y: ℝ) : (T_up * 60) = 324 :=
sorry

end time_escalator_upwards_l358_358859


namespace sum_of_binary_digits_of_315_l358_358750

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358750


namespace triangle_ABC_is_right_triangle_l358_358276

-- Define the triangle and the given conditions
variable (a b c : ℝ)
variable (h1 : a + c = 2*b)
variable (h2 : c - a = 1/2*b)

-- State the problem
theorem triangle_ABC_is_right_triangle : c^2 = a^2 + b^2 :=
by
  sorry

end triangle_ABC_is_right_triangle_l358_358276


namespace gcd_of_36_and_54_l358_358725

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l358_358725


namespace find_y_l358_358992

theorem find_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end find_y_l358_358992


namespace FN_bisects_MD_l358_358294

open EuclideanGeometry

variables {A B C D E F M N O : Point}

/-- Conditions -/
axiom circumcenter_O : Circumcenter O A B C
axiom acute_triangle_ABC : acute (triangle A B C)
axiom AB_gt_AC : AB > AC
axiom tangents_at_B_and_C_meet_at_D : TangentsMeet O B C D
axiom AO_extends_to_E : Extends AO BC E
axiom M_midpoint_BC : Midpoint M B C
axiom N_second_intersection_AM_with_circ_O : SecondIntersection AM (circO A B C) N
axiom F_is_second_intersect_circ_AEM_with_circO : SecondIntersection (circ A E M) (circO A B C) F

/-- Theorem -/
theorem FN_bisects_MD :
  Bisects FN MD :=
sorry

end FN_bisects_MD_l358_358294


namespace translation_coordinates_l358_358953

theorem translation_coordinates
  (a b : ℝ)
  (h₁ : 4 = a + 2)
  (h₂ : -3 = b - 6) :
  (a, b) = (2, 3) :=
by
  sorry

end translation_coordinates_l358_358953


namespace probability_of_third_ball_white_is_seven_thirteen_l358_358808

noncomputable def probability_third_ball_white : ℚ :=
let white_balls : ℕ := 8
let black_balls : ℕ := 7
let removed_white_balls : ℕ := 1
let removed_black_balls : ℕ := 1
let remaining_white_balls := white_balls - removed_white_balls
let remaining_black_balls := black_balls - removed_black_balls
let total_remaining_balls := remaining_white_balls + remaining_black_balls in
remaining_white_balls / total_remaining_balls

theorem probability_of_third_ball_white_is_seven_thirteen :
  probability_third_ball_white = 7 / 13 :=
by
  unfold probability_third_ball_white
  norm_num
  sorry

end probability_of_third_ball_white_is_seven_thirteen_l358_358808


namespace find_expression_and_range_l358_358547

-- Definition stating the condition given in the problem
def g (x : ℝ) : ℝ := 2^x

-- The point P(3, 8) condition
axiom g_passes_through_P : g 3 = 8

-- Main theorem that encapsulates the problem's conclusion
theorem find_expression_and_range (x : ℝ) (hx : ∀ y, g y = 2^y) :
  (∃ a, g a = 2^a ∧ g 3 = 8) ∧ (g (2*x^2 - 3*x + 1) > g (x^2 + 2*x - 5) → (x < 2 ∨ x > 3)) :=
by
  sorry

end find_expression_and_range_l358_358547


namespace angle_of_projection_for_min_speed_l358_358402

noncomputable def find_projection_angle (A B g : ℝ) : ℝ :=
  let C := real.sqrt (A^2 + B^2) in
  real.arctan ((B + C) / A)

theorem angle_of_projection_for_min_speed
  (A B : ℝ)
  (hA : A = 40)
  (hB : B = 30)
  (g : ℝ) :
  find_projection_angle 40 30 g ≈ real.arctan 2 :=
by
  sorry

end angle_of_projection_for_min_speed_l358_358402


namespace photographs_to_reach_target_l358_358132
noncomputable def photographsTaken18HoursAgo := 100
noncomputable def percentageReduction := 0.20
noncomputable def todayPhotographs := photographsTaken18HoursAgo * (1 - percentageReduction)
noncomputable def targetPhotographs := 300

theorem photographs_to_reach_target :
  todayPhotographs + photographsTaken18HoursAgo + 120 = targetPhotographs := by
  sorry

end photographs_to_reach_target_l358_358132


namespace fence_cost_square_plot_l358_358391

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) 
  (h_area : area = 289) (h_price : price_per_foot = 59) : 
  let side_length := Real.sqrt area in
  let perimeter := 4 * side_length in
  perimeter * price_per_foot = 4012 := 
by 
  sorry

end fence_cost_square_plot_l358_358391


namespace prob_at_least_one_wrong_l358_358832

-- Defining the conditions in mathlib
def prob_wrong : ℝ := 0.1
def num_questions : ℕ := 3

-- Proving the main statement
theorem prob_at_least_one_wrong : 1 - (1 - prob_wrong) ^ num_questions = 0.271 := by
  sorry

end prob_at_least_one_wrong_l358_358832


namespace compound_interest_calculation_l358_358650

-- Define the given conditions
def principal : ℝ := 15000
def annual_interest_rate : ℝ := 0.10
def times_compounded_per_year : ℕ := 2
def time_in_years : ℝ := 1

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem that we need to prove
theorem compound_interest_calculation :
  compound_interest principal annual_interest_rate times_compounded_per_year time_in_years = 16537.50 :=
by {
  // This is where the proof would go
  sorry
}

end compound_interest_calculation_l358_358650


namespace point_P_coordinates_l358_358194

noncomputable def P_coordinates (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 4 * Real.sin θ)

theorem point_P_coordinates : 
  ∀ θ, (0 ≤ θ ∧ θ ≤ Real.pi ∧ 1 = (4 / 3) * Real.tan θ) →
  P_coordinates θ = (12 / 5, 12 / 5) :=
by
  intro θ h
  sorry

end point_P_coordinates_l358_358194


namespace store_profit_l358_358077

theorem store_profit (C : ℝ) : 
    let SP1 := 1.20 * C
    let SP2 := 1.25 * SP1
    let SPF := 0.91 * SP2
    SPF - C = 0.365 * C :=
by 
    let SP1 := 1.20 * C
    let SP2 := 1.25 * SP1
    let SPF := 0.91 * SP2
    have h1 : SP1 = 1.20 * C := rfl
    have h2 : SP2 = 1.25 * SP1 := rfl
    have h3 : SPF = 0.91 * SP2 := rfl
    calc
    SPF - C = (0.91 * (1.25 * (1.20 * C))) - C : by rw [h3, h2, h1]
          ... = 1.365 * C - C : by simp [mul_assoc]
          ... = 0.365 * C : by ring

end store_profit_l358_358077


namespace quadratic_real_roots_l358_358239

-- Define the quadratic equation and the necessary conditions for real roots
def quadratic_equation_real_roots (k : ℝ) : Prop :=
  kx^2 - 2x + 1 = 0 ∧ (k ≤ 1 ∧ k ≠ 0)

-- Prove that the equation has real roots if and only if the conditions on k are satisfied
theorem quadratic_real_roots (k : ℝ) :
  quadratic_equation_real_roots k ↔ (k ≤ 1 ∧ k ≠ 0) := 
sorry

end quadratic_real_roots_l358_358239


namespace monotonic_lambda_range_exp_diff_inequality_l358_358203

noncomputable def f (λ : ℝ) (x : ℝ) : ℝ :=
  λ * Real.log x - Real.exp (-x)

-- Equivalent proof problem for Question 1
theorem monotonic_lambda_range (λ : ℝ) (x : ℝ) (h_pos : 0 < x) :
  (∀ x > 0, f λ x = λ * Real.log x - Real.exp (-x)) →
  (λ ≥ 0 ∨ λ ≤ -1 / Real.exp 1) :=
sorry

-- Equivalent proof problem for Question 2
theorem exp_diff_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  Real.exp (1 - x2) - Real.exp (1 - x1) > 1 - x2 / x1 :=
sorry

end monotonic_lambda_range_exp_diff_inequality_l358_358203


namespace problem_solution_l358_358989

theorem problem_solution (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 2 ^ a + 2 ^ -a = 4 * Real.sqrt 3 / 3 := 
by
sorry

end problem_solution_l358_358989


namespace no_hamiltonian_path_exists_eulerian_path_no_eulerian_circuit_l358_358801

def graph_vertices : Type := sorry -- Define the type for vertices
def graph_edges : graph_vertices → graph_vertices → Prop := sorry -- Define the edge relation

-- Given Conditions for Hamiltonian Path
def is_bipartite (G : graph_vertices → graph_vertices → Prop) : Prop := sorry
axiom bipartite_graph : is_bipartite graph_edges
def vertex_count := ∃ (V : list graph_vertices), V.length = 12 ∧ V.filter (λ v, true).length = 5  -- example counting vertices filter by a property (like color)
axiom color_partition : vertex_count

-- Given Conditions for Eulerian Path
def degree (v : graph_vertices) : ℕ := sorry -- Define vertex degree function
def odd_degree_vertices := {v : graph_vertices | degree v % 2 = 1}
axiom graph_odd_degree : odd_degree_vertices.to_list.length = 2

-- Questions translated to proof problems

-- Question 1: Hamiltonian Path
theorem no_hamiltonian_path : ¬ ∃ (p : list graph_vertices), (∀ v ∈ p, v ∈ graph_vertices ) ∧ 
  (∀ i < p.length - 1, graph_edges (p.nth i) (p.nth (i + 1))) ∧ (p.nodup = true) := sorry

-- Question 2: Eulerian Path
theorem exists_eulerian_path : ∃ (p : list (graph_vertices × graph_vertices)), 
  (∀ e ∈ p, graph_edges e.1 e.2) ∧ (∀ v ∈ graph_vertices, (p.count (λ e, e.1 = v) + p.count (λ e, e.2 = v)) % 2 = 0 → degree v % 2 = 0) ∧
  (odd_degree_vertices.to_list.all (λ v, vertex_in_path v p)) := sorry

-- Question 3: Eulerian Circuit
theorem no_eulerian_circuit : ¬ ∃ (p : list (graph_vertices × graph_vertices)), 
  (∀ e ∈ p, graph_edges e.1 e.2) ∧ (∀ v ∈ graph_vertices, p.count (λ e, e.1 = v) + p.count (λ e, e.2 = v) = degree v) ∧ 
  (odd_degree_vertices.to_list = []) := sorry

end no_hamiltonian_path_exists_eulerian_path_no_eulerian_circuit_l358_358801


namespace inequality_and_equality_condition_l358_358290

theorem inequality_and_equality_condition
  (a b : Fin n → ℝ) 
  (h_pos : ∑ i, (a i)^2 > 0) :
  (∑ i, (a i)^2) * (∑ i, (b i)^2) ≤ (∑ i, (a i * b i))^2 ∧ 
  (∑ i, (a i)^2) * (∑ i, (b i)^2) = (∑ i, (a i * b i))^2 → ∀ i, b i = 0 → a i = 0 :=
sorry

end inequality_and_equality_condition_l358_358290


namespace train_distance_l358_358783

theorem train_distance :
  ∃ D : ℝ, D = 480 ∧
  (∀ t₁ t₂ : ℝ, t₁ = D / 160 → t₂ = D / 120 → t₂ = t₁ + 1) :=
by
  let D := 480
  use D
  split
  · rfl
  · intros t₁ t₂ h₁ h₂
    rw [h₁, h₂]   -- use the provided equalities
    sorry -- This part would typically involve showing t₂ = t₁ + 1 which requires algebraic manipulation.

end train_distance_l358_358783


namespace primes_in_interval_l358_358979

theorem primes_in_interval (C : ℕ) (H : C = 25560) :
  ∃ (S : finset ℕ), (∀ p ∈ S, prime p ∧ 2 ≤ p ∧ p ≤ 2^30 ∧ p % 2017 = 1) ∧ S.card = C :=
by
  let S := {p ∈ finset.range (2^30 + 1) | prime p ∧ p ≥ 2 ∧ p % 2017 = 1}
  have HS : ∀ p ∈ S, prime p ∧ 2 ≤ p ∧ p ≤ 2^30 ∧ p % 2017 = 1, {
    intros p hp,
    simp only [finset.mem_filter, finset.mem_range] at hp,
    exact ⟨hp.1.prime, hp.2.1, hp.2.2, hp.1.1.symm⟩
  }
  use S,
  split,
  { exact HS },
  {
    sorry  -- Proof omitted
  }

end primes_in_interval_l358_358979


namespace sum_of_first_11_terms_is_132_l358_358264

variable {a : ℕ → ℤ} -- arithmetic sequence {a_n}
variable {d : ℤ} -- common difference of the sequence

noncomputable def aₙ (n : ℕ) := a 1 + (n - 1) * d -- general formula for arithmetic sequence

def arithmetic_sequence_condition : Prop :=
  2 * aₙ 9 = aₙ 12 + 12

def sum_of_first_11_terms : ℤ :=
  ∑ i in Finset.range 11, aₙ (i + 1)

theorem sum_of_first_11_terms_is_132 (h : arithmetic_sequence_condition) :
  sum_of_first_11_terms = 132 :=
sorry

end sum_of_first_11_terms_is_132_l358_358264


namespace f_60_eq_11_25_l358_358677

theorem f_60_eq_11_25
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) / y)
  (h2 : f 45 = 15) :
  f 60 = 11.25 :=
by
  sorry

end f_60_eq_11_25_l358_358677


namespace find_two_digit_number_l358_358921

theorem find_two_digit_number : ∃ A B : ℕ, A ≠ B ∧ 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧
  let n := 10 * A + B in
  10 ≤ n ∧ n ≤ 99 ∧ n^2 = (A + B)^3 ∧ n = 27 :=
by
  sorry

end find_two_digit_number_l358_358921


namespace power_calculation_l358_358114

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := 
by 
  -- Simplifying (8^3 / 8^2)
  have h1 : 8^3 / 8^2 = 8 := by 
    calc
      8^3 / 8^2 = 8^(3 - 2) : by rw [nat.pow_sub (show 2 ≤ 3, by norm_num)]
      ... = 8^1 : by norm_num
      ... = 8 : by norm_num
  -- Rewriting 8 as 2^3
  have h2 : 8 = 2^3 := by norm_num
  -- We now have (8^3 / 8^2) * 2^10 = 8 * 2^10
  rw [h1, h2], simp
  -- Simplifying 2^3 * 2^10 = 2^(3+10)
  calc 8 * 2^10 = 2^3 * 2^10 : by {rw h2}
  ... = 2^(3 + 10) : by rw [pow_add]
  ... = 2^13 : by norm_num
  ... = 8192 : by norm_num

end power_calculation_l358_358114


namespace perpendiculars_form_regular_ngon_l358_358429

-- Definitions related to the problem conditions
def circle_divided_by_equal_arcs (n : ℕ) (O : point) : Prop :=
  -- Definition for a circle divided into n equal arcs by diameters
  ∃ diameters : fin n → line,
    (∀ i, (diameters i).contains O) ∧ 
    (∀ i j, i ≠ j → is_perpendicular (diameters i) (diameters j))

def perpendicular_foot (M : point) (d : line) : point :=
  -- Definition for the foot of the perpendicular dropped from M to line d
  let p := Foot (M, d) in p

def regular_n_gon (vertices : list point) (n : ℕ) : Prop :=
  -- Definition for a regular n-gon given a list of vertices
  (∀ i j : fin n, ∃ d : ℝ, vertices.nth i = vertices.nth j → dist vertices.nth i vertices.nth j = d)

-- Proof statement
theorem perpendiculars_form_regular_ngon (n : ℕ) (O M : point) (h : circle_divided_by_equal_arcs n O) :
  ∃ vertices : list point, (∀ i, vertices.nth i = perpendicular_foot M (h.1 i)) ∧ regular_n_gon vertices n :=
by
  sorry

end perpendiculars_form_regular_ngon_l358_358429


namespace roadRepairDays_l358_358813

-- Definitions from the conditions
def dailyRepairLength1 : ℕ := 6
def daysToFinish1 : ℕ := 8
def totalLengthOfRoad : ℕ := dailyRepairLength1 * daysToFinish1
def dailyRepairLength2 : ℕ := 8
def daysToFinish2 : ℕ := totalLengthOfRoad / dailyRepairLength2

-- Theorem to be proven
theorem roadRepairDays :
  daysToFinish2 = 6 :=
by
  sorry

end roadRepairDays_l358_358813


namespace math_problem_l358_358226

noncomputable def proof_problem (n : ℝ) (A B : ℝ) : Prop :=
  A = n^2 ∧ B = n^2 + 1 ∧ (1 * n^4 + 2 * n^2 + 3 + 2 * (n^2 + 1) + 1 = 5 * (2 * n^2 + 1)) → 
  A + B = 7 + 4 * Real.sqrt 2

theorem math_problem (n : ℝ) (A B : ℝ) :
  proof_problem n A B :=
sorry

end math_problem_l358_358226


namespace not_possible_scores_l358_358642

theorem not_possible_scores : 
  let score_1 := 117
  let score_2 := 119
  ∀ (correct unanswered incorrect : ℕ), 
  (correct + unanswered + incorrect = 30) → 
  (4 * correct + 2 * unanswered + 0 * incorrect ≠ score_1
  ∧ 4 * correct + 2 * unanswered + 0 * incorrect ≠ score_2) :=
by
  intros correct unanswered incorrect hsum
  have h1 : 4 * correct + 2 * unanswered ≠ 117 := sorry
  have h2 : 4 * correct + 2 * unanswered ≠ 119 := sorry
  exact ⟨h1, h2⟩

end not_possible_scores_l358_358642


namespace evaluate_powers_of_i_l358_358480

noncomputable def i : ℂ := complex.I

theorem evaluate_powers_of_i : i^15 + i^20 + i^25 + i^30 + i^35 + i^40 = 2 :=
by
  have h : i^4 = 1 := by simp [i]
  sorry

end evaluate_powers_of_i_l358_358480


namespace circle_equations_l358_358496

-- Given conditions: the circle passes through points O(0,0), A(1,1), B(4,2)
-- Prove the general equation of the circle and the standard equation 

theorem circle_equations : 
  ∃ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
                      (x, y) = (0, 0) ∨ (x, y) = (1, 1) ∨ (x, y) = (4, 2)) ∧
  (D = -8) ∧ (E = 6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y = 0 ↔ (x - 4)^2 + (y + 3)^2 = 25) :=
sorry

end circle_equations_l358_358496


namespace sum_of_digits_of_binary_315_is_6_l358_358743
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358743


namespace students_just_passed_total_l358_358602

noncomputable def total_students := 500
noncomputable def first_division_math := 0.30 * total_students
noncomputable def second_division_math := 0.50 * total_students
noncomputable def just_passed_math := total_students - (first_division_math + second_division_math)

noncomputable def first_division_sci := 0.20 * total_students
noncomputable def second_division_sci := 0.60 * total_students
noncomputable def just_passed_sci := total_students - (first_division_sci + second_division_sci)

noncomputable def first_division_hist := 0.35 * total_students
noncomputable def second_division_hist := 0.45 * total_students
noncomputable def just_passed_hist := total_students - (first_division_hist + second_division_hist)

noncomputable def first_division_lit := 0.25 * total_students
noncomputable def second_division_lit := 0.55 * total_students
noncomputable def just_passed_lit := total_students - (first_division_lit + second_division_lit)

noncomputable def total_just_passed := just_passed_math + just_passed_sci + just_passed_hist + just_passed_lit

theorem students_just_passed_total : total_just_passed = 400 := by sorry

end students_just_passed_total_l358_358602


namespace probability_of_detecting_non_conforming_l358_358428

noncomputable def prob_detecting_non_conforming (total_cans non_conforming_cans selected_cans : ℕ) : ℚ :=
  let total_outcomes := Nat.choose total_cans selected_cans
  let outcomes_with_one_non_conforming := Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) (selected_cans - 1)
  let outcomes_with_two_non_conforming := Nat.choose non_conforming_cans 2
  (outcomes_with_one_non_conforming + outcomes_with_two_non_conforming) / total_outcomes

theorem probability_of_detecting_non_conforming :
  prob_detecting_non_conforming 5 2 2 = 7 / 10 :=
by
  -- Placeholder for the actual proof
  sorry

end probability_of_detecting_non_conforming_l358_358428


namespace smallest_k_l358_358302

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 19}
def satisfies_condition (A : Set ℕ) : Prop :=
  ∀ b ∈ M, ∃ a_i a_j ∈ A, a_i = b ∨ a_i + a_j = b ∨ a_i - a_j = b

theorem smallest_k :
  ∃ (k : ℕ), (∀ A ⊆ M, (∃ B ⊆ A, ∃ h : satisfies_condition B, B.card = k)) ∧
  (∀ j : ℕ, j < k → ¬∃ A ⊆ M, ∃ B ⊆ A, satisfies_condition B ∧ B.card = j) :=
begin
  use 5,
  sorry
end

end smallest_k_l358_358302


namespace ages_total_l358_358047

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l358_358047


namespace transformed_data_variance_l358_358171

noncomputable def original_data_variance {α : Type*} [has_variance α] (s : ℝ) := s^2 = 9

theorem transformed_data_variance {α : Type*} [has_variance α] (x : α) (s : ℝ) (h : original_data_variance s) :
  variance (2 * x + 1) = 36 := 
sorry

end transformed_data_variance_l358_358171


namespace red_fraction_after_tripling_l358_358248

-- Define the initial and final conditions
theorem red_fraction_after_tripling (x : ℕ) :
  let blue_initial := 4 / 7
  let red_initial := 1 - blue_initial
  let blue_count := blue_initial * x
  let red_count := red_initial * x
  let red_new_count := 3 * red_count
  let total_new_count := blue_count + red_new_count
  let new_red_fraction := red_new_count / total_new_count
  new_red_fraction = 9 / 13 :=
by
  -- Assumption and Conditions
  have h1 : blue_initial = 4 / 7 := rfl
  have h2 : red_initial = 3 / 7 := by linarith
  have h3 : blue_count = 4 * x / 7 := by norm_num [blue_initial, mul_div_assoc, mul_comm]
  have h4 : red_count = 3 * x / 7 := by norm_num [red_initial, mul_div_assoc, mul_comm]
  have h5 : red_new_count = 9 * x / 7 := by norm_num [red_count, three_mul_eq_succ_two_mul_twice]
  have h6 : total_new_count = 13 * x / 7 := by linarith [blue_count, red_new_count]
  have h7 : new_red_fraction = (9 * x / 7) / (13 * x / 7) := by rw [div_div_eq_div_mul, div_self (mul_ne_zero (nsmul_ne_zero_of_pos zero_lt_3 x.ne_zero))]
  -- Conclusion
  show new_red_fraction = 9 / 13 := by norm_num [h7]

sorry -- Omit the proof

end red_fraction_after_tripling_l358_358248


namespace pigeon_burrito_expected_time_l358_358467

/-- Given a 10x10 grid, a burrito dropped in the top left square, and a pigeon 
looking for food, where the pigeon either eats 10% of the burrito and moves it 
to a random square or moves toward an adjacent square to the burrito, 
prove that the expected number of minutes before the pigeon has eaten 
the entire burrito is 71.8 --/
theorem pigeon_burrito_expected_time : 
  let grid_size := 10
  let initial_burrito_pos := (0, 0)
  let initial_pigeon_pos := (0, 0)
  let eat_rate := 0.1
  let move_time := 1
  -- E[d] is the expected travel distance
  let travel_distance_expectation := 2 * 4.5 -- 4.5 being the average initial travel distance
  -- total steps to eat fully the burrito
  let total_steps := (9: Nat)
  -- summing the minutess 
  let total_minutes := (total_steps * travel_distance_expectation) + (total_steps + 1)  + total_steps * move_time
  total_minutes = 71.8 := 
sorry

end pigeon_burrito_expected_time_l358_358467


namespace shaded_region_area_nearest_integer_l358_358379
noncomputable def area_shaded_region : ℤ :=
let radius := 10 in
let BC := 20 in
let AC := radius * Real.sqrt 3 in
let triangle_area := (1/2 : ℚ) * radius * AC in
let sector_area := (1/6 : ℚ) * Real.pi * radius ^ 2 in
let total_sector_area := 2 * sector_area in
let shaded_area := triangle_area - total_sector_area in
Int.nearest (Real.toFloat shaded_area)

theorem shaded_region_area_nearest_integer :
  ∀ (radius : ℝ), radius = 10 →
  ∀ (BC AC triangle_area sector_area total_sector_area shaded_area : ℝ),
    BC = 2 * radius →
    AC = radius * Real.sqrt 3 →
    triangle_area = (1/2) * radius * AC →
    sector_area = (1/6) * Real.pi * radius ^ 2 →
    total_sector_area = 2 * sector_area →
    shaded_area = triangle_area - total_sector_area →
    Int.nearest (Real.toFloat shaded_area) = 8 :=
by
  intros radius radius_eq 
        BC BC_eq AC AC_eq 
        triangle_area tri_area_eq 
        sector_area sector_area_eq
        total_sector_area total_sec_area_eq
        shaded_area shaded_area_eq
  rw [radius_eq, BC_eq, AC_eq, tri_area_eq, sector_area_eq, total_sec_area_eq, shaded_area_eq]
  sorry

end shaded_region_area_nearest_integer_l358_358379


namespace haylee_has_36_guppies_l358_358219

variables (H J C N : ℝ)
variables (total_guppies : ℝ := 84)

def jose_has_half_of_haylee := J = H / 2
def charliz_has_third_of_jose := C = J / 3
def nicolai_has_four_times_charliz := N = 4 * C
def total_guppies_eq_84 := H + J + C + N = total_guppies

theorem haylee_has_36_guppies 
  (hJ : jose_has_half_of_haylee H J)
  (hC : charliz_has_third_of_jose J C)
  (hN : nicolai_has_four_times_charliz C N)
  (htotal : total_guppies_eq_84 H J C N) :
  H = 36 := 
  sorry

end haylee_has_36_guppies_l358_358219


namespace circles_intersect_l358_358213

def circle (a b r : ℝ) := { p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2 }

def C1 : set (ℝ × ℝ) := circle (-1.0) (-4.0) 5.0
def C2 : set (ℝ × ℝ) := circle 2.0 2.0 (real.sqrt 10)

-- The distance between the centers of the circles
def d : ℝ := real.sqrt ((-1.0 - 2.0)^2 + (-4.0 - 2.0)^2)

-- The radii of the circles
def r1 : ℝ := 5.0
def r2 : ℝ := real.sqrt 10

-- The condition for the circles to intersect
theorem circles_intersect : |r1 - r2| < d ∧ d < r1 + r2 :=
by
  sorry

end circles_intersect_l358_358213


namespace profit_share_difference_l358_358781

noncomputable def investment_a : ℕ := 8000
noncomputable def investment_b : ℕ := 10000
noncomputable def investment_c : ℕ := 12000

noncomputable def profit_share_b : ℕ := 3000

theorem profit_share_difference :
  let ratio_a := 4
      ratio_b := 5
      ratio_c := 6
      total_parts := ratio_a + ratio_b + ratio_c
      part_value := profit_share_b / ratio_b
      profit_share_a := ratio_a * part_value
      profit_share_c := ratio_c * part_value
  in profit_share_c - profit_share_a = 1200 :=
by
  sorry

end profit_share_difference_l358_358781


namespace gcd_36_54_l358_358723

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l358_358723


namespace rational_square_of_one_minus_product_l358_358236

theorem rational_square_of_one_minus_product (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
  ∃ (q : ℚ), 1 - x * y = q^2 := 
by 
  sorry

end rational_square_of_one_minus_product_l358_358236


namespace lifestyle_risk_probability_l358_358843

theorem lifestyle_risk_probability (p q : ℕ) (hpq : Nat.coprime p q) :
  (∀ A B C D : Prop,
    (Prob (A ∧ ¬B ∧ ¬C ∧ ¬D) = 0.05) ∧
    (Prob (A ∧ B ∧ C ∧ ¬D) = 0.08) ∧
    (Prob (A ∧ B ∧ C ∧ D | A ∧ B ∧ C) = 1/4) ∧
    (Prob (¬A ∧ ¬B ∧ ¬C ∧ ¬D | ¬A) = p/q)) →
  p + q = 157 :=
by
  intros A B C D h
  sorry

end lifestyle_risk_probability_l358_358843


namespace new_average_is_15_l358_358671

-- Definitions corresponding to the conditions
def avg_10_consecutive (seq : List ℤ) : Prop :=
  seq.length = 10 ∧ seq.sum = 200

def new_seq (seq : List ℤ) : List ℤ :=
  List.mapIdx (λ i x => x - ↑(9 - i)) seq

-- Statement of the proof problem
theorem new_average_is_15
  (seq : List ℤ)
  (h_seq : avg_10_consecutive seq) :
  (new_seq seq).sum = 150 := sorry

end new_average_is_15_l358_358671


namespace construct_triangle_with_given_ratios_l358_358121

-- Definitions for the points M, N, P based on the given ratios
variable {A B C M N P : Type*}
variable [metric_space A] [metric_space B] [metric_space C]
variable (ratios : ℝ) (m n p : ℝ)
variable (H1 : ∃ (M on BC), ∃ (N on CA), ∃ (P on AB),
  (MB / MC = m) ∧ (NA / NB = n) ∧ (PC / PA = p))

-- Proving the existence of triangle
theorem construct_triangle_with_given_ratios
  (m n p : ℝ)
  (H1 : ∃ {M N P : Type*} [metric_space M] [metric_space N] [metric_space P],
    (MB / MC = m) ∧ (NA / NB = n) ∧ (PC / PA = p)) :
  ∃ (A B C : Type*) [metric_space A] [metric_space B] [metric_space C],
    true :=
by
  sorry

end construct_triangle_with_given_ratios_l358_358121


namespace minimum_positive_period_monotonically_increasing_interval_length_of_side_c_l358_358218

-- Definitions of the given vectors and the function f(x)
def a (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin x, sqrt 3 * cos (x + π / 2) + 1)

def b (x : ℝ) : ℝ × ℝ :=
  (cos x, sqrt 3 * cos (x + π / 2) - 1)

def f (x : ℝ) : ℝ :=
  let (a1, a2) := a x
  let (b1, b2) := b x
  a1 * b1 + a2 * b2

-- Prove the minimum positive period of f(x) is π
theorem minimum_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

-- Prove the monotonically increasing interval of f(x)
theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x, x ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
  ∃ ε > 0, ∀ y ∈ set.Icc x (x + ε), f y > f x :=
sorry

-- Prove the length of side c given the conditions in △ABC
theorem length_of_side_c :
  ∀ a b C,
  a = 2 * sqrt 2 →
  b = sqrt 2 →
  (sqrt 3 * sin (2 * C - π / 3) + 1 / 2 = 2) →
  C ∈ set.Icc 0 π →
  ∃ c, c = sqrt 6 ∨ c = sqrt 10 :=
sorry

end minimum_positive_period_monotonically_increasing_interval_length_of_side_c_l358_358218


namespace arithmetic_problem_l358_358330

theorem arithmetic_problem : 
  let x := 512.52 
  let y := 256.26 
  let diff := x - y 
  let result := diff * 3 
  result = 768.78 := 
by 
  sorry

end arithmetic_problem_l358_358330


namespace current_speed_is_1_3_l358_358405

noncomputable def rate_of_current (rowing_speed : ℝ) (h : 0 < rowing_speed) (time_downstream : ℝ) (time_upstream : ℝ) 
  (h_time : time_upstream = 2 * time_downstream) : ℝ :=
  let current_speed := (rowing_speed * (time_upstream - time_downstream)) / (2 * time_downstream) in
  current_speed

theorem current_speed_is_1_3
  (rowing_speed : ℝ)
  (h : rowing_speed = 3.9)
  (time_downstream : ℝ)
  (time_upstream : ℝ)
  (h_time : time_upstream = 2 * time_downstream) :
  rate_of_current rowing_speed (by linarith) time_downstream time_upstream h_time = 1.3 :=
by
  -- Let's define c as the rate of the current according to the given solution
  let c := 1.3
  -- Assert and prove that c is equal to 1.3 km/hr based on the problem conditions
  have : rate_of_current rowing_speed (by linarith) time_downstream time_upstream h_time = c := 
  by sorry
  -- Conclude with the proven assertion
  exact this

end current_speed_is_1_3_l358_358405


namespace average_expr_value_l358_358672

-- Define the set of numbers
def numbers : Finset ℤ := {1, 2, 3, 11, 12, 13, 14}

-- Define the expression
def expr (a b c d e f g : ℤ) : ℤ :=
  (a - b)^2 + (b - c)^2 + (c - d)^2 + (d - e)^2 + (e - f)^2 + (f - g)^2

-- The main theorem stating the average value
theorem average_expr_value :
  let perms := (Finset.univ : Finset (Fin 7)).permutations in
  (∑ σ in perms, expr (numbers σ[0]) (numbers σ[1]) (numbers σ[2]) (numbers σ[3]) (numbers σ[4]) (numbers σ[5]) (numbers σ[6])) / perms.card = 392 :=
sorry

end average_expr_value_l358_358672


namespace find_train_speed_l358_358835

def train_speed (v t_pole t_stationary d_stationary : ℕ) : ℕ := v

theorem find_train_speed (v : ℕ) (t_pole : ℕ) (t_stationary : ℕ) (d_stationary : ℕ) :
  t_pole = 5 →
  t_stationary = 25 →
  d_stationary = 360 →
  25 * v = 5 * v + d_stationary →
  v = 18 :=
by intros h1 h2 h3 h4; sorry

end find_train_speed_l358_358835


namespace smallest_k_for_sixty_four_gt_four_nineteen_l358_358741

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen_l358_358741


namespace domain_of_g_l358_358474

noncomputable def g (x : ℝ) := (x - 5) / Real.sqrt (x^2 - 5*x - 6)

theorem domain_of_g :
  {x : ℝ | (x^2 - 5*x - 6 > 0)} = {x | x < -1} ∪ {x | x > 6} :=
by
  ext
  simp only [mem_set_of_eq, set.mem_union_eq, gt_iff_lt]
  constructor
  · intro h
    cases lt_or_ge x (-1)
    · left; exact h_1
    cases le_or_gt x 6
    · exfalso; linarith
    right; exact h_2
  · intro h
    cases h
    · linarith
    linarith

end domain_of_g_l358_358474


namespace sum_powers_of_i_l358_358418

theorem sum_powers_of_i : 
  let i := Complex.I in
  ∑ k in Finset.range (2023 + 1), i^k = -1 := sorry

end sum_powers_of_i_l358_358418


namespace quadratic_minimum_value_l358_358156

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end quadratic_minimum_value_l358_358156


namespace complex_number_quadrant_l358_358267

noncomputable def z_coordinate : ℂ :=
  (-1 + 3 * complex.I) / 2

theorem complex_number_quadrant :
  ∃ (z : ℂ), z * (1 - complex.I) = (1 + 2 * complex.I) * complex.I ∧
  z_coordinate.re < 0 ∧ z_coordinate.im > 0 :=
by
  use (-1 + 3 * complex.I) / 2
  split
  · sorry  -- Placeholder for the proof of the equation
  · split
    · sorry  -- Placeholder for the proof that the real part of z is negative
    · sorry  -- Placeholder for the proof that the imaginary part of z is positive

end complex_number_quadrant_l358_358267


namespace room_width_is_12_l358_358676

variable (w : ℝ)

def length_of_room : ℝ := 20
def width_of_veranda : ℝ := 2
def area_of_veranda : ℝ := 144

theorem room_width_is_12 :
  24 * (w + 4) - 20 * w = 144 → w = 12 := by
  sorry

end room_width_is_12_l358_358676


namespace number_difference_l358_358804

theorem number_difference:
  ∀ (number : ℝ), 0.30 * number = 63.0000000000001 →
  (3 / 7) * number - 0.40 * number = 6.00000000000006 := by
  sorry

end number_difference_l358_358804


namespace Xa_has_density_l358_358621

-- Definitions of i.i.d. random variables and series convergence

noncomputable def Xi (n : ℕ): ℝ := -- Placeholder definition for i.i.d. random variable
sorry

def lambda (n : ℕ): ℝ := -- Placeholder definition for lambda_n
sorry

-- Condition: Xi are i.i.d. with specified probabilities
axiom Xi_iid : ∀ n, Xi n = 1 ∨ Xi n = -1
axiom Xi_prob : ∀ n, probability (Xi n = 1) = 1/2 ∧ probability (Xi n = -1) = 1/2

-- Convergence of the series is given by the condition
axiom series_converges (alpha : ℝ) (h_alpha : alpha > 1/2) : 
  ∑ (n : ℕ), lambda n * Xi n = some (X : ℝ)

-- The main statement to prove
theorem Xa_has_density (alpha : ℝ) (h_alpha : alpha > 1/2) :
  has_density (series_converges alpha h_alpha) :=
sorry

end Xa_has_density_l358_358621


namespace trig_expression_value_l358_358190

theorem trig_expression_value (θ : ℝ) (h1 : ∀ x : ℝ, 3 * Real.sin x + 4 * Real.cos x ≤ 5)
(h2 : 3 * Real.sin θ + 4 * Real.cos θ = 5)
(h3 : Real.sin θ = 3 / 5)
(h4 : Real.cos θ = 4 / 5) :
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / (Real.cos (2 * θ)) = 15 / 7 := 
sorry

end trig_expression_value_l358_358190


namespace three_digit_even_numbers_count_l358_358718

theorem three_digit_even_numbers_count :
  ∃ n, n = 288 ∧ 
    let available_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ∀ (units hundreds tens : ℕ),
    units ∈ available_digits ∧ hundreds ∈ available_digits ∧ tens ∈ available_digits ∧
    units ≠ hundreds ∧ units ≠ tens ∧ hundreds ≠ tens ∧
    (units % 2 = 0) ∧
    (hundreds ≠ 0) ∧
    (hundreds * 100 + tens * 10 + units) ∈ (100..999) -> 
    (units * hundreds * tens) = n := sorry

end three_digit_even_numbers_count_l358_358718


namespace remainder_of_37_div_8_is_5_l358_358956

theorem remainder_of_37_div_8_is_5 : ∃ A B : ℤ, 37 = 8 * A + B ∧ 0 ≤ B ∧ B < 8 ∧ B = 5 := 
by
  sorry

end remainder_of_37_div_8_is_5_l358_358956


namespace modular_inverse_of_5_mod_23_l358_358904

theorem modular_inverse_of_5_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 := 
begin
  use 14,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end modular_inverse_of_5_mod_23_l358_358904


namespace lamp_pricing_problem_l358_358809

theorem lamp_pricing_problem
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (sales_decrease_rate : ℝ)
  (desired_profit : ℝ) :
  purchase_price = 30 →
  initial_selling_price = 40 →
  initial_sales_volume = 600 →
  sales_decrease_rate = 10 →
  desired_profit = 10000 →
  (∃ (selling_price : ℝ) (sales_volume : ℝ), selling_price = 50 ∧ sales_volume = 500) :=
by
  intros h_purchase h_initial_selling h_initial_sales h_sales_decrease h_desired_profit
  sorry

end lamp_pricing_problem_l358_358809


namespace old_clock_slower_by_144_minutes_l358_358852

-- The minute and hour hands of an old clock overlap every 66 minutes in standard time.
def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def overlap_interval_old_clock : ℝ := 66
def standard_day_minutes : ℕ := 24 * 60

theorem old_clock_slower_by_144_minutes :
  let relative_speed := minute_hand_speed - hour_hand_speed in
  let alignment_time := 360 / relative_speed in
  let k := overlap_interval_old_clock / alignment_time in
  let old_clock_minutes := standard_day_minutes * (12 / 11) in  
  old_clock_minutes = 1584 :=
by
  sorry

end old_clock_slower_by_144_minutes_l358_358852


namespace treasure_in_region_A_l358_358307

def point := ℝ × ℝ

structure Region :=
  (contains : point → Prop)

noncomputable def hedge : ℝ := 0
noncomputable def tree : point := (10, 0)

def at_least_5m_from_hedge (p : point) : Prop :=
  abs (p.1 - hedge) ≥ 5

def at_most_5m_from_tree (p : point) : Prop :=
  (p.1 - tree.1)^2 + (p.2 - tree.2)^2 ≤ 25

def region_A : Region :=
  { contains := λ p, at_least_5m_from_hedge p ∧ at_most_5m_from_tree p }

theorem treasure_in_region_A : 
  ∀ p : point, at_least_5m_from_hedge p ∧ at_most_5m_from_tree p → region_A.contains p :=
by {
  intro p,
  intro H,
  exact H,
  sorry
}

end treasure_in_region_A_l358_358307


namespace boatman_distance_downstream_l358_358427

noncomputable def speed_stationary_boat : ℝ := 1.5 -- 3 km in 2 hours
noncomputable def speed_upstream : ℝ := 1 -- 3 km in 3 hours
noncomputable def speed_current : ℝ := speed_stationary_boat - speed_upstream
noncomputable def speed_downstream : ℝ := speed_stationary_boat + speed_current
noncomputable def time_downstream : ℝ := 0.5 -- 30 minutes in hours

theorem boatman_distance_downstream :
  let D := speed_downstream * time_downstream in
  D = 1 :=
by
  sorry

end boatman_distance_downstream_l358_358427


namespace old_clock_144_minutes_slower_l358_358849

/-- Define the condition: the old clock hand overlap period -/
def hand_overlap_period_old_clock : ℝ := 66

/-- Define the standard hand overlap period using given rates -/
def hand_overlap_period_standard : ℝ := 360 / (6 - 0.5)

/-- Define the relative speed constant k -/
def k : ℝ := 66 / hand_overlap_period_standard

/-- Define the total minutes in 24 hours in the old clock -/
def total_minutes_old_clock : ℝ := 24 * 60 * (1 / k)

/-- Define the standard 24 hours in minutes -/
def total_minutes_standard : ℝ := 24 * 60

/-- Prove that the old clock is 144 minutes slower than the standard 24 hours -/
theorem old_clock_144_minutes_slower : total_minutes_standard - total_minutes_old_clock = 144 :=
by
  sorry

end old_clock_144_minutes_slower_l358_358849


namespace statistics_statements_correctness_l358_358552

theorem statistics_statements_correctness :
  (∀ (stmt1 stmt2 stmt3 stmt4 : Prop), 
  (stmt1 ↔ (¬ (∃ (linear_regression : Prop), ¬ linear_regression → linear_regression))) ∧
  (stmt2 ↔ (∀ (R : ℝ), 0 ≤ R ∧ R ≤ 1 → (R → ^2) ~ 1)) ∧
  (stmt3 ↔ (∀ (X Y : Type) (K : Type), ∃ (k : ℕ), (k > 0) → (K = X → Y))) ∧
  (stmt4 ↔ (∀ (r : ℝ), -1 ≤ r ∧ r ≤ 1 → ((abs r = r) ↔ (|r|)) → (r == 0))) →
  stmt1 ∧ stmt2 ∧ stmt3 ∧ stmt4 → 4
)$ 

end statistics_statements_correctness_l358_358552


namespace taxi_fare_l358_358697

-- Define the necessary values and functions based on the problem conditions
def starting_price : ℝ := 6
def additional_charge_per_km : ℝ := 1.5
def distance (P : ℝ) : Prop := P > 6

-- Lean proposition to state the problem
theorem taxi_fare (P : ℝ) (hP : distance P) : 
  (starting_price + additional_charge_per_km * (P - 6)) = 1.5 * P - 3 := 
by 
  sorry

end taxi_fare_l358_358697


namespace sum_of_digits_of_binary_315_is_6_l358_358742
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358742


namespace custom_op_identity_l358_358578

def custom_op (x y : ℕ) : ℕ := x * y - 3 * x + 1

theorem custom_op_identity : custom_op 8 5 - custom_op 5 8 = -9 :=
by
  have h1 : custom_op 8 5 = 8 * 5 - 3 * 8 + 1 := rfl
  have h2 : custom_op 5 8 = 5 * 8 - 3 * 5 + 1 := rfl
  rw [h1, h2]
  norm_num
  exact rfl

end custom_op_identity_l358_358578


namespace problem1_problem2_l358_358282

-- Proof problem for the first part
theorem problem1 (n : ℕ) (h : n = 11) 
  (a : ℕ → ℤ)
  (h_eq : (1 - x) ^ n = ∑ i in range (n + 1), a i * x ^ i) :
  |a 6| + |a 7| + |a 8| + |a 9| + |a 10| + |a 11| = 1024 := sorry

-- Proof problem for the second part
theorem problem2 (n : ℕ) (a : ℕ → ℤ) (b : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_eq : (1 - x) ^ n = ∑ i in range (n + 1), a i * x ^ i)
  (h_b : ∀ k, k ≤ n - 1 → b k = (k + 1) / (n - k) * a (k + 1))
  (h_S : ∀ m, m ≤ n - 1 → S m = ∑ i in range (m + 1), b i):
  ∀ m, m ≤ n - 1 -> (|S m / C (n - 1) m| : ℤ) = 1 := sorry

end problem1_problem2_l358_358282


namespace find_PF2_l358_358534

-- Statement of the problem

def hyperbola_1 (x y: ℝ) := (x^2 / 16) - (y^2 / 20) = 1

theorem find_PF2 (x y PF1 PF2: ℝ) (a : ℝ)
    (h_hyperbola : hyperbola_1 x y)
    (h_a : a = 4) 
    (h_dist_PF1 : PF1 = 9) :
    abs (PF1 - PF2) = 2 * a → PF2 = 17 :=
by
  intro h1
  sorry

end find_PF2_l358_358534


namespace identify_a_and_b_l358_358041

open Int

theorem identify_a_and_b : 
    ∃ (a b : ℚ), 
    (19 + 2/3) * (20 + 1/3) = (a - b) * (a + b) ∧ a = 20 ∧ b = 1/3 :=
by {
  use [20, 1/3],
  split,
  sorry, -- The proof would demonstrate the equivalence of the expressions
  split; refl, -- Verifies the assigned values for a and b are correct
}

end identify_a_and_b_l358_358041


namespace largest_composite_in_five_consecutive_ints_l358_358507

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l358_358507


namespace volume_intersection_of_pyramid_and_sphere_equals_2pi_over_9_l358_358172

noncomputable def volume_intersection (sphere_radius : ℝ) (pyramid_slant_height : ℝ) 
  (pyramid_base_side : ℝ) : ℝ :=
  sorry

theorem volume_intersection_of_pyramid_and_sphere_equals_2pi_over_9 
  (h1 : ∃ O A B C D : Type, OA = √3)
  (h2 : BC = 2)
  (h3 : sphere_radius = 1)
  :
  volume_intersection 1 √3 2 = (2 * π) / 9 :=
sorry

end volume_intersection_of_pyramid_and_sphere_equals_2pi_over_9_l358_358172


namespace solution_A_property_l358_358891

noncomputable def candidate_A := 2 + Real.sqrt 3

theorem solution_A_property : 
  ∃ (A : ℝ), 
  (A = candidate_A) ∧ 
  ∀ (n : ℕ), 
    let ceil_An := Real.ceil (A^n) in
    ∃ k : ℤ, abs (ceil_An - k^2) = 2 :=
by
  exists candidate_A
  split
  · refl
  · intro n
    let ceil_An := Real.ceil (candidate_A^n)
    let k := (Real.ceil (candidate_A^n)) - 2 -- Rough estimate
    use k
    sorry

end solution_A_property_l358_358891


namespace least_n_divisibility_l358_358027

def f (n : ℕ) : ℕ := n^2 - n

theorem least_n_divisibility :
  ∃ n : ℕ, n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n+1 ∧ (f(n) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n →
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ m+1 → ((f(m) % k = 0) ↔ k = m ∨ k = m-1)) ∧
    (∃ k : ℕ, 1 ≤ k ∧ k ≤ m+1 ∧ f(m) % k ≠ 0) :=
by
  sorry

end least_n_divisibility_l358_358027


namespace cut_into_no_more_than_four_parts_to_form_square_cut_into_no_more_than_five_triangular_parts_to_form_square_l358_358641

-- Define the conditions for the figure and its area
def figureArea : ℕ := 64
def targetSquareSide : ℕ := 8

-- Proof problem 1: Cutting into no more than four parts to form a square
theorem cut_into_no_more_than_four_parts_to_form_square 
    (figure : Type) 
    (area : figure → ℕ)
    (H_area : ∃ f, area f = figureArea) :
    ∃ parts : list figure, parts.length ≤ 4 ∧ pieces_form_square parts targetSquareSide :=
sorry

-- Proof problem 2: Cutting into no more than five triangular parts to form a square
theorem cut_into_no_more_than_five_triangular_parts_to_form_square 
    (figure : Type) 
    (area : figure → ℕ) 
    (is_triangle : figure → Prop)
    (H_area : ∃ f, area f = figureArea)
    (H_triangle : ∀ f, is_triangle f) :
    ∃ parts : list figure, parts.length ≤ 5 ∧ pieces_form_square parts targetSquareSide :=
sorry

end cut_into_no_more_than_four_parts_to_form_square_cut_into_no_more_than_five_triangular_parts_to_form_square_l358_358641


namespace quasi_symmetric_point_l358_358562

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4*Real.log x

def tangent_line (f : ℝ → ℝ) (x0 : ℝ) : ℝ → ℝ :=
  λ x, (2 * x0 + 4 / x0 - 6) * (x - x0) + x0^2 - 6 * x0 + 4 * Real.log x0

def m (x0 : ℝ) (x : ℝ) : ℝ :=
  f x - tangent_line f x0 x

theorem quasi_symmetric_point :
  ∀ x0 ∈ Set.Ioi 0, (∀ x ≠ x0, (m x0 x) / (x - x0) > 0) ↔ x0 = Real.sqrt 2 := 
by
  intro x0 hx0
  constructor
  -- Left-to-right direction
  { intro H
    sorry
  }
  -- Right-to-left direction
  { intro H
    rw H
    sorry
  }

end quasi_symmetric_point_l358_358562


namespace maximum_f_1989_count_f_1989_l358_358818

open Nat

def f : ℕ → ℕ
| 1 := 1
| (2 * n) := f n
| (2 * n + 1) := f (2 * n) + 1

theorem maximum_f_1989 : (u : ℕ) (1 ≤ u ∧ u ≤ 1989) → f u ≤ 10 :=
by sorry

theorem count_f_1989 : (u : ℕ) (1 ≤ u ∧ u ≤ 1989) → f u = 10 → ∃!(x : ℕ), x = u :=
by sorry

end maximum_f_1989_count_f_1989_l358_358818


namespace probability_of_drawing_red_ball_l358_358259

theorem probability_of_drawing_red_ball :
  let red_balls := 1
  let white_balls := 4
  let total_balls := red_balls + white_balls
  let probability_red := red_balls / total_balls
  probability_red = 0.2 :=
by {
  -- Definitions
  let red_balls : ℝ := 1
  let white_balls : ℝ := 4
  let total_balls := red_balls + white_balls

  -- Calculation
  let probability_red := red_balls / total_balls

  -- Proof
  have : probability_red = 0.2 :=
    by calc
      red_balls / total_balls
        = 1 / (1 + 4) : by sorry
      ... = 1 / 5 : by sorry
      ... = 0.2 : by sorry

  -- Concluding the theorem
  exact this
}

end probability_of_drawing_red_ball_l358_358259


namespace correct_statements_l358_358339

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l358_358339


namespace largest_perfect_square_factor_and_perfect_cube_l358_358026

noncomputable def largest_perfect_square_factor (n : ℕ) : ℕ :=
  let factors := [2, 3, 7]
  in factors.product

theorem largest_perfect_square_factor_and_perfect_cube (n : ℕ)
  (h : n = 1764) : 
  largest_perfect_square_factor n = 1764 ∧ ∀ m ∣ n, ¬ ∃ k : ℕ, k^3 = m :=
by
  sorry

end largest_perfect_square_factor_and_perfect_cube_l358_358026


namespace cos_even_function_l358_358876

theorem cos_even_function : ∀ x : ℝ, Real.cos (-x) = Real.cos x := 
by 
  sorry

end cos_even_function_l358_358876


namespace range_of_m_value_of_m_l358_358966

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end range_of_m_value_of_m_l358_358966


namespace math_problem_l358_358179

theorem math_problem (x y : ℝ) (h : (x + 2 * y) ^ 3 + x ^ 3 + 2 * x + 2 * y = 0) : x + y - 1 = -1 := 
sorry

end math_problem_l358_358179


namespace complement_of_M_in_U_l358_358303

-- Definition of the universal set U
def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }

-- Definition of the set M
def M : Set ℝ := { 1 }

-- The statement to prove
theorem complement_of_M_in_U : (U \ M) = {x | 1 < x ∧ x ≤ 5} :=
by
  sorry

end complement_of_M_in_U_l358_358303


namespace three_digit_even_numbers_count_l358_358719

theorem three_digit_even_numbers_count :
  ∃ n, n = 288 ∧ 
    let available_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ∀ (units hundreds tens : ℕ),
    units ∈ available_digits ∧ hundreds ∈ available_digits ∧ tens ∈ available_digits ∧
    units ≠ hundreds ∧ units ≠ tens ∧ hundreds ≠ tens ∧
    (units % 2 = 0) ∧
    (hundreds ≠ 0) ∧
    (hundreds * 100 + tens * 10 + units) ∈ (100..999) -> 
    (units * hundreds * tens) = n := sorry

end three_digit_even_numbers_count_l358_358719


namespace digit_three_more_than_seven_l358_358131

/-- 
Given a book with pages numbered from 1 to 567, each page number appearing exactly once, 
prove that the digit '3' appears two times more than the digit '7' in the entire page number set.
-/
theorem digit_three_more_than_seven : 
  let count_digit (d n : Nat) := (Nat.digits 10 n).count d in
  let count_in_book (d : Nat) := (List.range 1 568).sum (count_digit d) in
  count_in_book 3 = count_in_book 7 + 2 :=
  sorry

end digit_three_more_than_seven_l358_358131


namespace distance_P_to_P_prime_l358_358712

noncomputable def P := (2 : ℤ, -4 : ℤ)
noncomputable def P_prime := (-2 : ℤ, -4 : ℤ)

def euclidean_distance (x1 y1 x2 y2 : ℤ) : ℝ :=
  real.sqrt (((x2 - x1) ^ 2 : ℝ) + ((y2 - y1) ^ 2 : ℝ))

theorem distance_P_to_P_prime :
  euclidean_distance (P.1) (P.2) (P_prime.1) (P_prime.2) = 4 :=
by
  sorry

end distance_P_to_P_prime_l358_358712


namespace intersection_complement_l358_358971

variable {M : Set ℝ := {-1, 0, 1, 3}}

def N : Set ℝ := {x | x^2 - x - 2 ≥ 0}

def complement_N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_complement :
  M ∩ complement_N = {0, 1} :=
  sorry

end intersection_complement_l358_358971


namespace problem_statement_l358_358646

theorem problem_statement (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
sorry

end problem_statement_l358_358646


namespace modular_inverse_of_5_mod_23_l358_358902

theorem modular_inverse_of_5_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 := 
begin
  use 14,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end modular_inverse_of_5_mod_23_l358_358902


namespace min_total_bananas_l358_358009

noncomputable def total_bananas_condition (b1 b2 b3 : ℕ) : Prop :=
  let m1 := (5/8 : ℚ) * b1 + (5/16 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m2 := (3/16 : ℚ) * b1 + (3/8 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m3 := (3/16 : ℚ) * b1 + (5/16 : ℚ) * b2 + (1/24 : ℚ) * b3
  (((m1 : ℚ) * 4) = ((m2 : ℚ) * 3)) ∧ (((m1 : ℚ) * 4) = ((m3 : ℚ) * 2))

theorem min_total_bananas : ∃ (b1 b2 b3 : ℕ), b1 + b2 + b3 = 192 ∧ total_bananas_condition b1 b2 b3 :=
sorry

end min_total_bananas_l358_358009


namespace sale_price_of_one_bottle_l358_358160

-- Define the relevant conditions.
def loads_per_bottle : ℕ := 80
def cost_per_load_in_cents : ℕ := 25
def number_of_bottles : ℕ := 2

-- Translate cost per load in cents to cost per load in dollars for easier calculation.
def cost_per_load_in_dollars : ℝ := cost_per_load_in_cents / 100.0

-- Define the total cost when buying 2 bottles.
def total_cost : ℝ := loads_per_bottle * number_of_bottles * cost_per_load_in_dollars

-- Define the expected sale price per bottle.
def sale_price_per_bottle : ℝ := total_cost / number_of_bottles

-- The theorem statement that asserts the sale price per bottle is $20.00.
theorem sale_price_of_one_bottle : sale_price_per_bottle = 20.0 :=
by
  -- Skipping the proof.
  sorry

end sale_price_of_one_bottle_l358_358160


namespace proof_simplify_expression_l358_358320

noncomputable def simplify_expression (θ : ℝ) : Prop :=
  (cos θ - complex.i * sin θ)^8 * (1 + complex.i * tan θ)^5 / ((cos θ + complex.i * sin θ)^2 * (tan θ + complex.i)) 
  = - (sin (4 * θ) + complex.i * cos (4 * θ)) / cos(θ)^4

theorem proof_simplify_expression (θ : ℝ) : simplify_expression θ :=
  sorry

end proof_simplify_expression_l358_358320


namespace simplify_trig_expression_l358_358321

theorem simplify_trig_expression :
  (sin (15 * real.pi / 180) + sin (30 * real.pi / 180) + sin (45 * real.pi / 180) + 
   sin (60 * real.pi / 180) + sin (75 * real.pi / 180)) / 
  (cos (10 * real.pi / 180) * cos (20 * real.pi / 180) * cos (30 * real.pi / 180)) = 
  (√2 * (4 * (cos (22.5 * real.pi / 180)) * (cos (7.5 * real.pi / 180)) + 1)) / 
  (2 * (cos (10 * real.pi / 180)) * (cos (20 * real.pi / 180)) * (cos (30 * real.pi / 180))) :=
sorry

end simplify_trig_expression_l358_358321


namespace max_cross_section_area_is_260_l358_358498

noncomputable def max_cross_section_area (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) : ℝ :=
  max (a * Real.sqrt (b^2 + c^2)) (max (b * Real.sqrt (a^2 + c^2)) (c * Real.sqrt (a^2 + b^2)))

theorem max_cross_section_area_is_260 :
  max_cross_section_area 5 12 20 (and.intro (by norm_num) (by norm_num)) = 260 :=
by
  sorry

end max_cross_section_area_is_260_l358_358498


namespace work_rate_proof_l358_358404

theorem work_rate_proof (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : C = 1 / 60) : 
  1 / (A + B + C) = 12 :=
by
  sorry

end work_rate_proof_l358_358404


namespace f_2002_eq_0_l358_358679

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_2_eq_0 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem f_2002_eq_0 : f 2002 = 0 :=
by
  sorry

end f_2002_eq_0_l358_358679


namespace quadratic_solution_condition_sufficient_but_not_necessary_l358_358475

theorem quadratic_solution_condition_sufficient_but_not_necessary (m : ℝ) :
  (m < -2) → (∃ x : ℝ, x^2 + m * x + 1 = 0) ∧ ¬(∀ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0 → m < -2) :=
by 
  sorry

end quadratic_solution_condition_sufficient_but_not_necessary_l358_358475


namespace cylinder_volume_correct_l358_358128

noncomputable def volume_cylinder_in_yards (d h : ℝ) : ℝ :=
  let r := d / 2
  let volume_ft := π * r^2 * h
  volume_ft / 27

theorem cylinder_volume_correct :
  volume_cylinder_in_yards 25 12 = 69.444 * π :=
by
  sorry

end cylinder_volume_correct_l358_358128


namespace part_I_part_II_l358_358199

-- Given definitions

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + 3| - m + 1

-- Part (I): Prove the value of m
theorem part_I (m : ℝ) (h : m > 0) (h_sol : ∀ x, f (x - 3) m ≥ 0 ↔ x ∈ Icc (-2 : ℝ) 2 ∪ Ioc (2 : ℝ) ∞ ∪ Ioc (-∞ : ℝ) (-2)) :
  m = 3 :=
sorry

-- Part (II): Range of t given the condition
theorem part_II (t : ℝ) (h : ∃ x : ℝ, f x 2 ≥ |2 * x - 1| - t^2 + (5 / 2) * t) :
  t ≤ 1 ∨ t ≥ 3 / 2 :=
sorry

end part_I_part_II_l358_358199


namespace max_not_expressed_as_linear_comb_l358_358622

theorem max_not_expressed_as_linear_comb {a b c : ℕ} (h_coprime_ab : Nat.gcd a b = 1)
                                        (h_coprime_bc : Nat.gcd b c = 1)
                                        (h_coprime_ca : Nat.gcd c a = 1) :
    Nat := sorry

end max_not_expressed_as_linear_comb_l358_358622


namespace number_of_ordered_triples_l358_358085

theorem number_of_ordered_triples:
  let possible_triples := 
    { (a, b, c) : ℕ × ℕ × ℕ | 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ 2 * a * b * c = 4 * (a * b + b * c + c * a) }
  in possible_triples.count = 11 := sorry

end number_of_ordered_triples_l358_358085


namespace average_income_PQ_l358_358332

/-
Conditions:
1. The average monthly income of Q and R is Rs. 5250.
2. The average monthly income of P and R is Rs. 6200.
3. The monthly income of P is Rs. 3000.
-/

def avg_income_QR := 5250
def avg_income_PR := 6200
def income_P := 3000

theorem average_income_PQ :
  ∃ (Q R : ℕ), ((Q + R) / 2 = avg_income_QR) ∧ ((income_P + R) / 2 = avg_income_PR) ∧ 
               (∀ (p q : ℕ), p = income_P → q = (Q + income_P) / 2 → q = 2050) :=
by
  sorry

end average_income_PQ_l358_358332


namespace find_c_l358_358217

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -3)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_c (c : vector) : 
  (is_parallel (c.1 + a.1, c.2 + a.2) b) ∧ (is_perpendicular c (a.1 + b.1, a.2 + b.2)) → 
  c = (-7 / 9, -20 / 9) := 
by
  sorry

end find_c_l358_358217


namespace count_valid_triples_l358_358499

-- Definitions for the conditions
def is_prime (n : ℕ) : Prop := nat.prime n
def product_of_two_primes (n : ℕ) : Prop := ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n
def no_squared_prime_factors (n : ℕ) : Prop := ∀ p : ℕ, is_prime p → p * p ∣ n → false

-- Condition definitions
def condition_a (a b : ℕ) : Prop := is_prime (a * b)
def condition_b (b c : ℕ) : Prop := product_of_two_primes (b * c)
def condition_c (a b c : ℕ) : Prop := no_squared_prime_factors (a * b * c)
def condition_d (a b c : ℕ) : Prop := a * b * c ≤ 30

-- Set of valid triples
def valid_triples : set (ℕ × ℕ × ℕ) :=
  { (a, b, c) | a > 0 ∧ b > 0 ∧ c > 0 ∧ 
                condition_a a b ∧ 
                condition_b b c ∧ 
                condition_c a b c ∧ 
                condition_d a b c }

-- Main theorem to be proven
theorem count_valid_triples : finset.card (finset.filter valid_triples (finset.range 31 ×ˢ finset.range 31 ×ˢ finset.range 31)) = 21 := 
  sorry

end count_valid_triples_l358_358499


namespace largest_difference_l358_358295

def U : ℕ := 2 * 1002 ^ 1003
def V : ℕ := 1002 ^ 1003
def W : ℕ := 1001 * 1002 ^ 1002
def X : ℕ := 2 * 1002 ^ 1002
def Y : ℕ := 1002 ^ 1002
def Z : ℕ := 1002 ^ 1001

theorem largest_difference : (U - V) = 1002 ^ 1003 ∧ 
  (V - W) = 1002 ^ 1002 ∧ 
  (W - X) = 999 * 1002 ^ 1002 ∧ 
  (X - Y) = 1002 ^ 1002 ∧ 
  (Y - Z) = 1001 * 1002 ^ 1001 ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 999 * 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1001 * 1002 ^ 1001) :=
by {
  sorry
}

end largest_difference_l358_358295


namespace sequence_product_eq_81_l358_358482

theorem sequence_product_eq_81 :
  let seq : ℕ → ℚ := λ n, if n % 2 = 0 then 1 / 3 ^ (n / 2) else 3 ^ (n / 2 + 1)
  (∏ n in finset.range 8, seq n) = 81 := by 
  sorry

end sequence_product_eq_81_l358_358482


namespace solve_train_passenger_problem_l358_358078

def train_passenger_problem : Prop :=
  ∃ (x : ℕ), 
    (let trips := 4 in
     let passengers_per_trip_one_way := 100 in
     let total_passengers := 640 in
     400 + 4 * x = total_passengers) 
    ∧ x = 60

theorem solve_train_passenger_problem : train_passenger_problem :=
sorry

end solve_train_passenger_problem_l358_358078


namespace eight_digit_integers_l358_358975

theorem eight_digit_integers : ∀ (n : ℕ), (∃ (f : Fin 8 → Fin 10), (f 0) ∈ {2, 3, 4, 5, 6, 7, 8, 9} ∧ (∀ (i : Fin 7), f (i + 1) ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})) → n = 80000000 := by
    sorry

end eight_digit_integers_l358_358975


namespace calvin_wig_goal_l358_358462

theorem calvin_wig_goal :
  (∀ (X : ℕ), 
   (∀ (haircut_percentages : ℕ → ℕ), 
    (haircut_percentages 1 = 70 ∧ 
     haircut_percentages 2 = 50 ∧ 
     haircut_percentages 3 = 25) →
     (∀ (haircuts_done haircuts_needed : ℕ), 
      (haircuts_done = 8 ∧ 
       haircuts_needed = 10) →
       (∀ (dog : ℕ), 
        (dog = 1 ∨ 
         dog = 2 ∨ 
         dog = 3) →
         (haircuts_done * haircut_percentages dog) / haircuts_needed = 80)))) :=
by
  intros X haircut_percentages hp haircuts haircuts_done haircuts_needed dog dcond
  cases dcond; sorry

end calvin_wig_goal_l358_358462


namespace sum_of_digits_base2_315_l358_358758

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358758


namespace jia_opened_physical_store_l358_358705

-- Define the possible shop types
inductive ShopType
| Taobao
| WeChat
| PhysicalStore

open ShopType

-- Assume we have three people Jia, Yi, and Bing, with their shops
variable (jia_shop yi_shop bing_shop : ShopType)

-- Define their statements
def jia_statement : Prop := (jia_shop = Taobao ∧ yi_shop = WeChat) ∨ (jia_shop ≠ Taobao ∧ yi_shop ≠ WeChat)
def yi_statement : Prop := (jia_shop = WeChat ∧ bing_shop = Taobao) ∨ (jia_shop ≠ WeChat ∧ bing_shop ≠ Taobao)
def bing_statement : Prop := (jia_shop = PhysicalStore ∧ yi_shop = Taobao) ∨ (jia_shop ≠ PhysicalStore ∧ yi_shop ≠ Taobao)

-- Define the condition that each person's statement is only half correct
axiom jia_half_correct : jia_statement
axiom yi_half_correct : yi_statement
axiom bing_half_correct : bing_statement

-- Define the theorem to prove the solution
theorem jia_opened_physical_store : jia_shop = PhysicalStore := 
sorry

end jia_opened_physical_store_l358_358705


namespace problem_statement_l358_358970

noncomputable def U : Set ℝ := set.univ

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 6}

noncomputable def B : Set ℝ := {x | - x^2 + 3 * x + 4 ≤ 0}

theorem problem_statement :
  (A ∩ set.compl U ∩ B) = {1, 2, 3} := by sorry

end problem_statement_l358_358970


namespace find_largest_number_l358_358364

noncomputable def largest_number (a b c : ℚ) : ℚ :=
  if a + b + c = 77 ∧ c - b = 9 ∧ b - a = 5 then c else 0

theorem find_largest_number (a b c : ℚ) 
  (h1 : a + b + c = 77) 
  (h2 : c - b = 9) 
  (h3 : b - a = 5) : 
  c = 100 / 3 := 
sorry

end find_largest_number_l358_358364


namespace sum_of_seven_numbers_l358_358590

theorem sum_of_seven_numbers 
  (average_eight : ℝ) 
  (num_known : ℝ) 
  (total_eight : average_eight * 8 = 41.6) 
  (one_of_eight : num_known = 7) : 
  ∑ sum_seven = 34.6 := 
by 
  sorry

end sum_of_seven_numbers_l358_358590


namespace problem1_problem2_l358_358930

def expansion_coeffs (n : ℕ) : (ℕ → ℕ) :=
  -- Typically, one would derive the coefficient directly using combinatorics.
  λ k, match k with
  | 0   => 1
  | 1   => -3 * n
  | 2   => (9 * n * (n - 1)) / 2
  | _   => 0 -- Placeholder for general coefficients

theorem problem1 (n : ℕ) (h : expansion_coeffs n 2 = 15 * expansion_coeffs n 0 - 13 * expansion_coeffs n 1) : n = 10 :=
  sorry

theorem problem2 (n : ℕ) (n2023 : n = 2023) (A B : ℤ) 
  (hA : A = ∑ i in finset.range 1012, expansion_coeffs n (2*i))
  (hB : B = ∑ i in finset.range 1012, expansion_coeffs n (2*i+1) + expansion_coeffs n 2023) : 
  (A - B)/(A + B) = -2^2023 :=
  sorry

end problem1_problem2_l358_358930


namespace figure_100_squares_l358_358254

-- Define the initial conditions as given in the problem
def squares_in_figure (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | 2 => 25
  | 3 => 45
  | _ => sorry

-- Define the quadratic formula assumed from the problem conditions
def quadratic_formula (n : ℕ) : ℕ :=
  3 * n^2 + 5 * n + 3

-- Theorem: For figure 100, the number of squares is 30503
theorem figure_100_squares :
  squares_in_figure 100 = quadratic_formula 100 :=
by
  sorry

end figure_100_squares_l358_358254


namespace solution_A_property_l358_358892

noncomputable def candidate_A := 2 + Real.sqrt 3

theorem solution_A_property : 
  ∃ (A : ℝ), 
  (A = candidate_A) ∧ 
  ∀ (n : ℕ), 
    let ceil_An := Real.ceil (A^n) in
    ∃ k : ℤ, abs (ceil_An - k^2) = 2 :=
by
  exists candidate_A
  split
  · refl
  · intro n
    let ceil_An := Real.ceil (candidate_A^n)
    let k := (Real.ceil (candidate_A^n)) - 2 -- Rough estimate
    use k
    sorry

end solution_A_property_l358_358892


namespace sum_mod_12_l358_358739

def remainder_sum_mod :=
  let nums := [10331, 10333, 10335, 10337, 10339, 10341, 10343]
  let sum_nums := nums.sum
  sum_nums % 12 = 7

theorem sum_mod_12 : remainder_sum_mod :=
by
  sorry

end sum_mod_12_l358_358739


namespace area_of_circle_through_K_L_l358_358524

noncomputable def triangle_KLM := 
  {KM : ℝ, KL : ℝ, FM : ℝ, KM_FM : Prop, F_on_KM : Prop}
  (KM = sqrt 3 / 2)
  (KL = 1)
  (FM = sqrt 3 / 6)
  (KM_FM : KM = FM + KF)
  (F_on_KM : F ∈ KM)

theorem area_of_circle_through_K_L (K L M F : triangle_KLM) :
  ∃ r, r = sqrt(6) / 4 ∧ area_of_circle r = 3 * pi / 8 := sorry

end area_of_circle_through_K_L_l358_358524


namespace sum_of_digits_in_binary_representation_of_315_l358_358767

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358767


namespace Chuck_area_proof_l358_358465

noncomputable def area_available_for_Chuck (radius_leash radius_sector1 radius_sector2 : ℝ) : ℝ :=
  (3 / 4) * (Real.pi) * (radius_leash ^ 2) + (1 / 4) * (Real.pi) * (radius_sector2 ^ 2)

theorem Chuck_area_proof :
  let radius_leash := 5
      radius_sector1 := 5
      radius_sector2 := 2
      expected_area := 19.75 * Real.pi
  in area_available_for_Chuck radius_leash radius_sector1 radius_sector2 = expected_area := 
by
  sorry

end Chuck_area_proof_l358_358465


namespace P_zero_eq_zero_l358_358825

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry

axiom distinct_roots : ∃ y : Fin 17 → ℝ, Function.Injective y ∧ ∀ i, P (y i ^ 2) = 0

theorem P_zero_eq_zero : P 0 = 0 :=
by
  sorry

end P_zero_eq_zero_l358_358825


namespace part1_part2_l358_358210

open Vector3

-- Define points A, B, C
def A : ℝ × ℝ × ℝ := (0, 2, 3)
def B : ℝ × ℝ × ℝ := (-2, 1, 6)
def C : ℝ × ℝ × ℝ := (1, -1, 5)

-- Define vectors AB and AC
def AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Define magnitude function for a vector
noncomputable def magnitude(v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Part (1)
theorem part1 (a : ℝ × ℝ × ℝ) :
  magnitude(a) = √3 ∧ (AB.1 * a.1 + AB.2 * a.2 + AB.3 * a.3 = 0) ∧
  (AC.1 * a.1 + AC.2 * a.2 + AC.3 * a.3 = 0) →
  (a = (1, 1, 1) ∨ a = (-1, -1, -1)) :=
sorry

-- Define vector addition and scalar multiplication
def vec_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

-- Part (2)
theorem part2 :
  magnitude(vec_add (vec_scalar_mul 2 AB) AC) = 7 * real.sqrt 2 :=
sorry

end part1_part2_l358_358210


namespace distance_QR_eq_25_l358_358648

-- Definitions of the points and triangle properties.
variables (D E F Q R G : Type) [MetricSpace D E F Q R G]
variables (DE EF DF QA RA : ℝ)

-- Conditions
hypothesis h_DE : DE = 9
hypothesis h_EF : EF = 12
hypothesis h_DF : DF = 15

-- Definitions of the circles and their properties.
def circle_centered_at_Q_tangent_to_DE_and_passing_through_F : Prop :=
  sorry -- omitted specific geometric properties for brevity

def circle_centered_at_R_tangent_to_EF_and_passing_through_D : Prop :=
  sorry -- omitted specific geometric properties for brevity

-- Conditions related to the circles
hypothesis h_Q : circle_centered_at_Q_tangent_to_DE_and_passing_through_F
hypothesis h_R : circle_centered_at_R_tangent_to_EF_and_passing_through_D

-- The distance result to be proven.
theorem distance_QR_eq_25 : dist Q R = 25 :=
by
  -- Proof is omitted, use sorry to indicate missing proof body.
  sorry

end distance_QR_eq_25_l358_358648


namespace monotone_decreasing_func_l358_358182

theorem monotone_decreasing_func {f : ℝ → ℝ} (hf' : ∀ x > 0, x^2 * (deriv f x) + x * f x = log x)
  (hf_e : f (exp 1) = 1 / (exp 1)) :
  ∀ x > 0, deriv f x ≤ 0 :=
sorry

end monotone_decreasing_func_l358_358182


namespace fizz_preference_count_l358_358598

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count_l358_358598


namespace double_derivative_eq_function_l358_358631

noncomputable def f : ℝ → ℝ := λ x, Real.exp x  -- This is f(x) = e^x

theorem double_derivative_eq_function :
  (∀ x, deriv (deriv f) x = f x) :=
by
  sorry

end double_derivative_eq_function_l358_358631


namespace relationship_among_a_b_c_l358_358124

variable {f : ℝ → ℝ}
variable {a b c : ℝ}

-- Conditions
def differentiable_on_f_on_R : Prop :=
  ∀ x ∈ Ioi (1 : ℝ), differentiable_at ℝ f x

def condition1 (x : ℝ) (h : x ∈ Ioi (1 : ℝ)) : Prop :=
  (x - 1) * deriv f x - f x > 0

-- Definitions for a, b, c
def def_a : a = f 2 := rfl
def def_b : b = (1 / 2) * f 3 := rfl
def def_c : c = (1 / (real.sqrt 2 - 1)) * f (real.sqrt 2) := rfl

-- The statement to prove
theorem relationship_among_a_b_c (h1 : differentiable_on_f_on_R)
  (h2 : ∀ x (hx : x ∈ Ioi (1 : ℝ)), condition1 x hx)
  (ha : a = f 2)
  (hb : b = (1 / 2) * f 3)
  (hc : c = (1 / (real.sqrt 2 - 1)) * f (real.sqrt 2)) :
  c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l358_358124


namespace original_number_l358_358386

theorem original_number (x : ℕ) : 
  (∃ y : ℕ, y = x + 28 ∧ (y % 5 = 0) ∧ (y % 6 = 0) ∧ (y % 4 = 0) ∧ (y % 3 = 0)) → x = 32 :=
by
  sorry

end original_number_l358_358386


namespace volume_ratio_octahedron_tetrahedron_l358_358252

theorem volume_ratio_octahedron_tetrahedron 
  (regular_tetrahedron : Type) 
  (midpoints_form_octahedron : ∀ (v : regular_tetrahedron), (∃ (edge : regular_tetrahedron → regular_tetrahedron), (midpoint (edge v))) → (∃ (octahedron : Type), octahedron)
) : 
  let volume_tetrahedron := (√2^3 / (6 * √2))
  let volume_octahedron := (1 / 3 * (side_oct⁴)^2 * √3) / 4 * (√6) / 4
  let side_tetrahedron := √2
  let side_oct := √2 / 2
  volume_octahedron / volume_tetrahedron = 3 / 16 := sorry

end volume_ratio_octahedron_tetrahedron_l358_358252


namespace binary_mul_correct_l358_358148

def bin_to_nat (l : List ℕ) : ℕ :=
  l.foldl (λ n b => 2 * n + b) 0

def p : List ℕ := [1,0,1,1,0,1]
def q : List ℕ := [1,1,0,1]
def r : List ℕ := [1,0,0,0,1,0,0,0,1,1]

theorem binary_mul_correct :
  bin_to_nat p * bin_to_nat q = bin_to_nat r := by
  sorry

end binary_mul_correct_l358_358148


namespace third_class_duration_l358_358873

theorem third_class_duration :
  ∃ (x : ℝ), (24 * ((2 * 3) + x + 4)) = 336 ∧ x = 4 :=
begin
  use 4,
  split,
  { sorry, }, -- Prove the total hours calculation matches
  { sorry, }  -- Prove the third class is indeed 4 hours
end

end third_class_duration_l358_358873


namespace work_done_in_days_l358_358583

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l358_358583


namespace jimmy_wins_bet_l358_358872

theorem jimmy_wins_bet :
  ∃ t₀ : ℝ, ∀ k : ℕ, k < 4 → 
    let θ := k * (2 * Real.pi / 4),
        ω := 100 * Real.pi,
        t_aligned := t₀ + θ / ω
    in (t_aligned * ω) % (2 * Real.pi) = θ :=
by
  sorry

end jimmy_wins_bet_l358_358872


namespace magazine_ad_extra_cost_l358_358346

/--
The cost of purchasing a laptop through a magazine advertisement includes four monthly 
payments of $60.99 each and a one-time shipping and handling fee of $19.99. The in-store 
price of the laptop is $259.99. Prove that purchasing the laptop through the magazine 
advertisement results in an extra cost of 396 cents.
-/
theorem magazine_ad_extra_cost : 
  let in_store_price := 259.99
  let monthly_payment := 60.99
  let num_payments := 4
  let shipping_handling := 19.99
  let total_magazine_cost := (num_payments * monthly_payment) + shipping_handling
  (total_magazine_cost - in_store_price) * 100 = 396 := 
by
  sorry

end magazine_ad_extra_cost_l358_358346


namespace sum_of_digits_base_2_315_l358_358775

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358775


namespace task_completion_time_l358_358231

theorem task_completion_time (A B : ℝ) : 
  (14 * A / 80 + 10 * B / 96) = (20 * (A + B)) →
  (1 / (14 * A / 80 + 10 * B / 96)) = 480 / (84 * A + 50 * B) :=
by
  intros h
  sorry

end task_completion_time_l358_358231


namespace shortest_line_lateral_face_l358_358446

noncomputable def base_edge : ℝ := 40
noncomputable def height : ℝ := 40

-- The slant height of the pyramid
noncomputable def slant_height : ℝ := real.sqrt (base_edge^2 * 2 / 4 + height^2)

-- The perpendicular distance to the middle of the base
noncomputable def perpendicular_distance : ℝ := real.sqrt (slant_height^2 - (base_edge / 2)^2)

-- The shortest path on the lateral face
noncomputable def shortest_path : ℝ := 2 * base_edge * perpendicular_distance / slant_height

-- Numerically approximate the shortest path and compare
noncomputable def shortest_path_approx : ℝ := 80 * real.sqrt (5 / 6)

theorem shortest_line_lateral_face :
  real.abs (shortest_path - 73.03) < 0.01 := by
  sorry

end shortest_line_lateral_face_l358_358446


namespace number_of_divisors_of_3b_plus_15_l358_358665

theorem number_of_divisors_of_3b_plus_15 (a b : ℤ) (h : 4 * b = 10 - 5 * a) : 
  {d ∈ {1, 2, 3, 4, 5, 6} | d ∣ (3 * b + 15)}.toFinset.card = 3 := 
sorry

end number_of_divisors_of_3b_plus_15_l358_358665


namespace edge_ratio_of_cubes_l358_358051

theorem edge_ratio_of_cubes (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 :=
by
  sorry

end edge_ratio_of_cubes_l358_358051


namespace problem_I_problem_II_l358_358560

-- Definition of f(x) and the condition for problem (I)
def f (k : ℝ) (x : ℝ) : ℝ := abs (k * x - 1)
def solutionSet (k : ℝ) := setOf (fun x : ℝ => f k x ≤ 3)

-- Theorem for problem (I)
theorem problem_I (k : ℝ) :
  (solutionSet k = set.interval [-2, 1]) → (k = -2) :=
begin
  sorry
end

-- Definition of f(x) for problem (II), specifically k=1
def f2 (x : ℝ) : ℝ := abs (x - 1)
def h (x : ℝ) : ℝ := f2 (x + 2) - f2 (2 * x + 1)

-- Theorem for problem (II)
theorem problem_II (m : ℝ) :
  (∀ x : ℝ, h x ≤ 3 - 2 * m) → (m ≤ 1) :=
begin
  sorry
end

end problem_I_problem_II_l358_358560


namespace largest_x_value_l358_358292

noncomputable def largest_x : ℝ :=
  (13 + real.sqrt 160) / 6

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) :
  x ≤ largest_x := sorry

end largest_x_value_l358_358292


namespace sum_of_digits_of_binary_315_is_6_l358_358744
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358744


namespace trent_bus_blocks_to_library_l358_358710

-- Define the given conditions
def total_distance := 22
def walking_distance := 4

-- Define the function to determine bus block distance
def bus_ride_distance (total: ℕ) (walk: ℕ) : ℕ :=
  (total - (walk * 2)) / 2

-- The theorem we need to prove
theorem trent_bus_blocks_to_library : 
  bus_ride_distance total_distance walking_distance = 7 := by
  sorry

end trent_bus_blocks_to_library_l358_358710


namespace F_sum_of_coordinates_is_6_l358_358268

noncomputable def F : ℝ × ℝ :=
  let A := (0 : ℝ, 10 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let C := (10 : ℝ, 0 : ℝ)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let x_F := (10 - 2 * E.1) / (1 / 2 + 2)
  (x_F, 1 / 2 * x_F)

theorem F_sum_of_coordinates_is_6 : (F.1 + F.2) = 6 :=
by
  let A := (0 : ℝ, 10 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let C := (10 : ℝ, 0 : ℝ)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let x_F := (10 - 2 * E.1) / (1 / 2 + 2)
  have : F = (4, 2), from sorry
  sorry

end F_sum_of_coordinates_is_6_l358_358268


namespace old_clock_slower_by_144_minutes_l358_358851

-- The minute and hour hands of an old clock overlap every 66 minutes in standard time.
def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def overlap_interval_old_clock : ℝ := 66
def standard_day_minutes : ℕ := 24 * 60

theorem old_clock_slower_by_144_minutes :
  let relative_speed := minute_hand_speed - hour_hand_speed in
  let alignment_time := 360 / relative_speed in
  let k := overlap_interval_old_clock / alignment_time in
  let old_clock_minutes := standard_day_minutes * (12 / 11) in  
  old_clock_minutes = 1584 :=
by
  sorry

end old_clock_slower_by_144_minutes_l358_358851


namespace ratio_volumes_l358_358149

noncomputable def V_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def V_cone (r : ℝ) : ℝ := (1 / 3) * Real.pi * r^3

theorem ratio_volumes (r : ℝ) (hr : r > 0) : 
  (V_cone r) / (V_sphere r) = 1 / 4 :=
by
  sorry

end ratio_volumes_l358_358149


namespace rate_definition_l358_358351

-- Definitions based on conditions
def frequency (num_times : ℕ) : ℕ := num_times

def rate (num_times : ℕ) (total_times : ℕ) : ℚ :=
  num_times / total_times

-- Theorem statement
theorem rate_definition (num_times total_times : ℕ) (h_total_pos : total_times > 0) : 
  let freq := frequency num_times in
  rate freq total_times = num_times / total_times := sorry

end rate_definition_l358_358351


namespace range_of_ratio_l358_358188

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_ratio (x y : ℝ) 
  (h1 : ∀ a b : ℝ, a ≤ b → f(a) ≤ f(b))
  (h2 : ∀ z : ℝ, f(3 + z) = -f(3 - z))
  (h3 : f(x^2 - 2 * sqrt(3) * x + 9) + f(y^2 - 2 * y) ≤ 0) :
  0 ≤ y / x ∧ y / x ≤ sqrt(3) :=
sorry

end range_of_ratio_l358_358188


namespace shortest_distance_between_points_is_line_segment_l358_358398

theorem shortest_distance_between_points_is_line_segment (P Q : ℝ²) : 
  ∀ (P Q : ℝ²), (∃ l : set ℝ², line P Q l ∧ ∀ R ∈ l, dist P R + dist R Q = dist P Q) :=
sorry

end shortest_distance_between_points_is_line_segment_l358_358398


namespace remove_zeros_in_decimal_add_sub_does_not_change_value_l358_358035

theorem remove_zeros_in_decimal_add_sub_does_not_change_value :
  ∀ (a b : ℝ), (∃ c : ℝ, c = a + b ∨ c = a - b) → 
  (∀ (x : ℝ), x = c ∧ decimal_part_zero_removal x = x) -> 
  False :=
by
  sorry

end remove_zeros_in_decimal_add_sub_does_not_change_value_l358_358035


namespace sin_A_eq_side_a_eq_l358_358211

-- Given Conditions
variable (A B C a b c : ℝ)
variable (s : ℝ)
variable (h1 : A < (Real.pi / 2))
variable (h2 : Real.sin (A - Real.pi / 4) = (Real.sqrt 2) / 10)
variable (h3 : s = 24)
variable (h4 : b = 10)

-- Prove \( \sin A = \frac{4}{5} \)
theorem sin_A_eq : Real.sin A = 4 / 5 := by
  sorry

-- Prove \( a = 8 \)
theorem side_a_eq {c : ℝ} (h5 : c = (2 * s) / (b * (4 / 5))) : a = 8 := by
  let h6 := by
    rcalc [eq rfl] : 
    h1 => Reals.rt 64 eq_side_eq (cToa_sum r sm (ident.transpose.piecework)
                                  
all insert args unwrapping
                                )
only calc eqeq).(√-operationsetermined later. )
exact (Calculable 
  sorry

end sin_A_eq_side_a_eq_l358_358211


namespace parabola_and_ratio_proof_l358_358681

open Real

/-- Prove the equation of the parabola and the ratio of line segments given certain conditions -/
theorem parabola_and_ratio_proof 
  (A M N B P Q : Point)
  (p : ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hp : p > 0)
  (hn : |distance M N| = 3 * sqrt 2)
  (equiv_slope : slope_of_line A M = π / 4) 
  (hMN : ∃l, Line.Equiv l A M N)
  (hMB : ∃l, Line.Equiv l M B)
  (hPN : ∃l, Line.Equiv l P N)
  (hq : Q.x = 0): 
  (∃C, Parabola.Equiv C (λ c => y^2 = x) ∧ ∃ratio, ratio = 2 / 3) :=
sorry

end parabola_and_ratio_proof_l358_358681


namespace exists_real_number_A_l358_358890

-- Given the real number A = 2 + sqrt(3)
def A : ℝ := 2 + Real.sqrt 3

-- Main theorem stating that for any natural number n
-- the distance from the ceiling of A^n to the nearest square of an integer is 2
theorem exists_real_number_A :
  ∀ n : ℕ, ∃ k : ℤ, (Int.ceil (A ^ n) - k^2) = 2 :=
by sorry

end exists_real_number_A_l358_358890


namespace second_candidate_votes_l358_358258

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end second_candidate_votes_l358_358258


namespace lambda_value_l358_358011

open Real

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

def right_focus : ℝ × ℝ := (sqrt 3, 0)

def line_through_focus (m : ℝ) : ℝ → ℝ := λ x, m * (x - sqrt 3)

theorem lambda_value :
  (∃ l : ℝ → ℝ, l = line_through_focus) →
  (∃ A B : ℝ × ℝ, hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ A ≠ B ∧ ∃ λ : ℝ, |λ| = 4) →
  (∃ n : ℕ, n = 3) →
  λ = 4 :=
by
  sorry

end lambda_value_l358_358011


namespace find_quadruples_l358_358489

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l358_358489


namespace base10_to_base9_conversion_l358_358871

theorem base10_to_base9_conversion : 
  let n := 256
  let base := 9
  n = 256 ∧ base = 9 ∧ (256 / 81 = 3) ∧ (256 % 81 = 13) ∧ (13 / 9 = 1) ∧ (13 % 9 = 4)
  → show Nat, from 256 = 3 * 81 + 1 * 9 + 4 
:= by
  intros
  sorry

end base10_to_base9_conversion_l358_358871


namespace revenue_correct_l358_358603

def calculate_revenue : Real :=
  let pumpkin_pie_revenue := 4 * 8 * 5
  let custard_pie_revenue := 5 * 6 * 6
  let apple_pie_revenue := 3 * 10 * 4
  let pecan_pie_revenue := 2 * 12 * 7
  let cookie_revenue := 15 * 2
  let red_velvet_revenue := 6 * 8 * 9
  pumpkin_pie_revenue + custard_pie_revenue + apple_pie_revenue + pecan_pie_revenue + cookie_revenue + red_velvet_revenue

theorem revenue_correct : calculate_revenue = 1090 :=
by
  sorry

end revenue_correct_l358_358603


namespace max_marked_cells_no_shared_vertices_l358_358365

theorem max_marked_cells_no_shared_vertices (N : ℕ) (cube_side : ℕ) (total_cells : ℕ) (total_vertices : ℕ) :
  cube_side = 3 →
  total_cells = cube_side ^ 3 →
  total_vertices = 8 + 12 * 2 + 6 * 4 →
  ∀ (max_cells : ℕ), (4 * max_cells ≤ total_vertices) → (max_cells ≤ 14) :=
by
  sorry

end max_marked_cells_no_shared_vertices_l358_358365


namespace range_of_a_minus_abs_b_l358_358225

theorem range_of_a_minus_abs_b (a b : ℝ) (h1: 1 < a) (h2: a < 3) (h3: -4 < b) (h4: b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l358_358225


namespace sheila_weekly_earnings_is_288_l358_358655

-- Define the conditions as constants.
def sheilaWorksHoursPerDay (d : String) : ℕ :=
  if d = "Monday" ∨ d = "Wednesday" ∨ d = "Friday" then 8
  else if d = "Tuesday" ∨ d = "Thursday" then 6
  else 0

def hourlyWage : ℕ := 8

-- Calculate total weekly earnings based on conditions.
def weeklyEarnings : ℕ :=
  (sheilaWorksHoursPerDay "Monday" + sheilaWorksHoursPerDay "Wednesday" + sheilaWorksHoursPerDay "Friday") * hourlyWage +
  (sheilaWorksHoursPerDay "Tuesday" + sheilaWorksHoursPerDay "Thursday") * hourlyWage

-- The Lean statement for the proof.
theorem sheila_weekly_earnings_is_288 : weeklyEarnings = 288 :=
  by
    sorry

end sheila_weekly_earnings_is_288_l358_358655


namespace find_m_l358_358985

theorem find_m (m : ℝ) (a a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (1 + m)^6 = a + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 = 63)
  (h3 : a = 1) : m = 1 ∨ m = -3 := 
by
  sorry

end find_m_l358_358985


namespace gcd_of_36_and_54_l358_358729

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l358_358729


namespace sum_of_digits_base_2_315_l358_358773

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358773


namespace mod_inverse_exists_l358_358905

theorem mod_inverse_exists :
  ∃ a : ℕ, 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 :=
  by {
    use 14,
    split,
    { norm_num },
    split,
    { norm_num },
    exact rfl,
    sorry
  }

end mod_inverse_exists_l358_358905


namespace mod_inverse_exists_l358_358907

theorem mod_inverse_exists :
  ∃ a : ℕ, 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 :=
  by {
    use 14,
    split,
    { norm_num },
    split,
    { norm_num },
    exact rfl,
    sorry
  }

end mod_inverse_exists_l358_358907


namespace largest_n_satisfies_conditions_l358_358898

theorem largest_n_satisfies_conditions :
  ∃ (n m a : ℤ), n = 313 ∧ n^2 = (m + 1)^3 - m^3 ∧ ∃ (k : ℤ), 2 * n + 103 = k^2 :=
by
  sorry

end largest_n_satisfies_conditions_l358_358898


namespace sum_youngest_oldest_l358_358682

variables {a1 a2 a3 a4 a5 : ℕ}

def mean_age (x y z u v : ℕ) : ℕ := (x + y + z + u + v) / 5
def median_age (x y z u v : ℕ) : ℕ := z

theorem sum_youngest_oldest
  (h_mean: mean_age a1 a2 a3 a4 a5 = 10) 
  (h_median: median_age a1 a2 a3 a4 a5 = 7)
  (h_sorted: a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  a1 + a5 = 23 :=
sorry

end sum_youngest_oldest_l358_358682


namespace runners_meet_fractions_l358_358378

theorem runners_meet_fractions (l V₁ V₂ : ℝ)
  (h1 : l / V₂ - l / V₁ = 10)
  (h2 : 720 * V₁ - 720 * V₂ = l) :
  (1 / V₁ = 1 / 80 ∧ 1 / V₂ = 1 / 90) ∨ (1 / V₁ = 1 / 90 ∧ 1 / V₂ = 1 / 80) :=
sorry

end runners_meet_fractions_l358_358378


namespace hyperbola_center_l358_358141

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l358_358141


namespace compare_speed_is_half_l358_358806

def total_time_watched (x : ℝ) : ℝ :=
  (6 * 100 / (2 * x)) + (6 * 100 / x)

theorem compare_speed_is_half (x : ℝ) (h : total_time_watched x = 900) : x = 1 / 2 := by
  sorry

end compare_speed_is_half_l358_358806


namespace find_a_l358_358557

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then (1 / 2) * x - 1 else 1 / x

theorem find_a (a : ℝ) : f (f a) = -1 / 2 → a = 4 ∨ a = -1 / 2 :=
by
  intro h
  have : f(f(a)) = -1 / 2, from h
  sorry

end find_a_l358_358557


namespace commute_solution_l358_358308

noncomputable def commute_problem : Prop :=
  let t : ℝ := 1                -- 1 hour from 7:00 AM to 8:00 AM
  let late_minutes : ℝ := 5 / 60  -- 5 minutes = 5/60 hours
  let early_minutes : ℝ := 4 / 60 -- 4 minutes = 4/60 hours
  let speed1 : ℝ := 30          -- 30 mph
  let speed2 : ℝ := 70          -- 70 mph
  let d1 : ℝ := speed1 * (t + late_minutes)
  let d2 : ℝ := speed2 * (t - early_minutes)

  ∃ (speed : ℝ), d1 = d2 ∧ speed = d1 / t ∧ speed = 32.5

theorem commute_solution : commute_problem :=
by sorry

end commute_solution_l358_358308


namespace Lisa_pay_per_hour_is_15_l358_358974

-- Given conditions:
def Greta_hours : ℕ := 40
def Greta_pay_per_hour : ℕ := 12
def Lisa_hours : ℕ := 32

-- Define Greta's earnings based on the given conditions:
def Greta_earnings : ℕ := Greta_hours * Greta_pay_per_hour

-- The main statement to prove:
theorem Lisa_pay_per_hour_is_15 (h1 : Greta_earnings = Greta_hours * Greta_pay_per_hour) 
                                (h2 : Greta_earnings = Lisa_hours * L) :
  L = 15 :=
by sorry

end Lisa_pay_per_hour_is_15_l358_358974


namespace find_ellipse_equation_find_locus_of_Q_l358_358942

-- Define the given conditions for the ellipse and focus
def ellipse_conditions (x y a b : ℝ) : Prop := 
  (a > b ∧ b > 0) ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (x = sqrt 2 ∧ y = 1) ∧
  (x = -sqrt 2 ∧ y = 0)

-- Statement for part (1)
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 / 2 = 1)

theorem find_ellipse_equation (a b : ℝ) (h : ellipse_conditions (sqrt 2) 1 a b) :
  ellipse_equation (sqrt 2) 1 :=
by 
  sorry

-- Define the condition for the moving line and point Q
def moving_line_conditions (x y k : ℝ) : Prop :=
  ∃ m : ℝ, (y - 1 = m * (x - 4)) ∧
  (x^2 / 4 + (k * x + 1 - 4 * k)^2 / 2 = 1)

-- Statement for part (2)
def locus_of_Q (x y : ℝ) : Prop :=
  (2 * x + y - 2 = 0)

theorem find_locus_of_Q (x y k : ℝ) (h : moving_line_conditions x y k) :
  locus_of_Q x y :=
by 
  sorry

end find_ellipse_equation_find_locus_of_Q_l358_358942


namespace smaller_circle_radius_l358_358810

theorem smaller_circle_radius (A1 A2 : ℝ) 
  (h1 : A1 + 2 * A2 = 25 * Real.pi) 
  (h2 : ∃ d : ℝ, A1 + d = A2 ∧ A2 + d = A1 + 2 * A2) : 
  ∃ r : ℝ, r^2 = 5 ∧ Real.pi * r^2 = A1 :=
by
  sorry

end smaller_circle_radius_l358_358810


namespace number_of_common_tangents_l358_358527

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0 → (C₁ = (1, 0)) ∧ (r₁ = 1))
  (h₂ : ∀ (x y : ℝ), x^2 + y^2 - 4 * y + 3 = 0 → (C₂ = (0, 2)) ∧ (r₂ = 1))
  (d : distance C₁ C₂ = Real.sqrt 5) :
  4 = 4 := 
by sorry

end number_of_common_tangents_l358_358527


namespace sum_of_two_smallest_prime_factors_of_294_l358_358032

theorem sum_of_two_smallest_prime_factors_of_294 : ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ p1 < p2 ∧ (p1 * p2 ∣ 294) ∧ (p1 + p2 = 5) := 
by {
  sorry
}

end sum_of_two_smallest_prime_factors_of_294_l358_358032


namespace number_of_parakeets_per_cage_l358_358437

noncomputable def parakeets_per_cage : Nat :=
  let num_cages := 9
  let total_birds := 36
  -- Let P = parakeets per cage, Q = parrots per cage
  -- And we know P = Q
  let P := (total_birds / num_cages) / 2
  P

theorem number_of_parakeets_per_cage : parakeets_per_cage = 2 :=
by
  let num_cages := 9
  let total_birds := 36
  have h1 : total_birds / num_cages = 4 := Nat.div_eq_of_eq_eval (by norm_num : 36 = 9 * 4)
  have h2 : 4 / 2 = 2 := rfl
  show parakeets_per_cage = 2 from
    by simp [parakeets_per_cage, h1, h2]
  sorry

end number_of_parakeets_per_cage_l358_358437


namespace find_n_e_l358_358434

theorem find_n_e (N E : ℕ) (h_condition : N < E) : (N = 3 ∧ E = 7) → N / E = 0.428571 :=
by
  sorry

end find_n_e_l358_358434


namespace number_of_saplings_l358_358240

theorem number_of_saplings (r s : ℝ) (h_r : r = 8) (h_s : s = 2) : 
  (2 * real.pi * r / s).round = 25 :=
by
  rw [h_r, h_s]
  have C := 2 * real.pi * r
  have H := C / s
  rw [← h_r, ← h_s, real.pi]
  sorry

end number_of_saplings_l358_358240


namespace percentage_blue_and_red_l358_358060

theorem percentage_blue_and_red (F : ℕ) (h_even: F % 2 = 0)
  (h1: ∃ C, 50 / 100 * C = F / 2)
  (h2: ∃ C, 60 / 100 * C = F / 2)
  (h3: ∃ C, 40 / 100 * C = F / 2) :
  ∃ C, (50 / 100 * C + 60 / 100 * C - 100 / 100 * C) = 10 / 100 * C :=
sorry

end percentage_blue_and_red_l358_358060


namespace sum_of_digits_base_2_315_l358_358776

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358776


namespace reciprocal_altitude_intersection_l358_358662

-- Definitions of points and geometric constructions
variable (A B C D F H O : Type)

-- Conditions given in the problem
variable [RightTriangle A B C]

variable (CD_sq BF_sq : Square)
variable (h_inter_AH : Intersects CD B F AH)

-- Statement to prove
theorem reciprocal_altitude_intersection
  (h1 : SquaresConstructedOnLegs A B C CD_sq BF_sq)
  (h2 : Intersects CD BF AH O) :
  1 / AO = 1 / AH + 1 / BC := 
sorry

end reciprocal_altitude_intersection_l358_358662


namespace white_cells_in_contour_modulus_l358_358625

theorem white_cells_in_contour_modulus (n : ℕ) (hn : 1 < n) : ∀ moves : List (ℤ × ℤ), 
(∀ move ∈ moves, abs move.fst = n ∨ abs move.snd = n) → 
let path := List.scanl (λ (prev : ℤ × ℤ) (move : ℤ × ℤ), (prev.fst + move.fst, prev.snd + move.snd)) (0, 0) moves in
(List.last path).getD (0, 0) = (0, 0) →
(∀ i j : ℤ, (i, j) ∈ path → (i, j) ∈ (Set.range (λ k : ℤ, (k*n, 0)) ∪ Set.range (λ k : ℤ, (0, k*n)))) →
(∑ i in (path.eraseDuplicates).filter (λ p, p.1 ≠ 0 ∧ p.2 ≠ 0), (1 : ℤ)) % n = 1 :=
by 
  intros
  sorry

end white_cells_in_contour_modulus_l358_358625


namespace range_of_m_l358_358692

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 5x + 1 ∧ x - m > 1 → x > 1) → m ≤ 0 :=
by 
  intro h,
  sorry

end range_of_m_l358_358692


namespace minimum_ticket_cost_l358_358129

theorem minimum_ticket_cost (students : ℕ) (single_ticket_cost group_ticket_cost : ℕ) 
  (group_size : ℕ) (h_students : students = 48) (h_single_ticket : single_ticket_cost = 10)
  (h_group_ticket : group_ticket_cost = 70) (h_group_size : group_size = 10) :
  ∃ cost : ℕ, cost = 350 :=
by {
  have h1 : 4 * group_ticket_cost + 8 * single_ticket_cost = 360 :=
    by rw [h_group_ticket, h_single_ticket]; norm_num,
  have h2 : 5 * group_ticket_cost = 350 :=
    by rw [h_group_ticket]; norm_num,
  use 350,
  exact h2,
  sorry
}

end minimum_ticket_cost_l358_358129


namespace urn_contains_three_red_three_blue_after_five_operations_l358_358103

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5

noncomputable def calculate_probability (initial_red: ℕ) (initial_blue: ℕ) (operations: ℕ) : ℚ :=
  sorry

theorem urn_contains_three_red_three_blue_after_five_operations :
  calculate_probability initial_red_balls initial_blue_balls total_operations = 8 / 105 :=
by sorry

end urn_contains_three_red_three_blue_after_five_operations_l358_358103


namespace sum_of_digits_base_2_315_l358_358777

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358777


namespace complex_quadrant_l358_358997

theorem complex_quadrant (z : ℂ) (hz : z = (2 - 1*ℂ.I) / (1 + ℂ.I)) : 
  let coord := (⟨1/2, -(3/2)⟩ : ℂ) in 
  z.re > 0 ∧ z.im < 0 :=
by 
  sorry

end complex_quadrant_l358_358997


namespace largest_common_divisor_l358_358025

theorem largest_common_divisor (a b : ℕ) (h1 : a = 360) (h2 : b = 315) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ ∀ e : ℕ, (e ∣ a ∧ e ∣ b) → e ≤ d ∧ d = 45 :=
by
  sorry

end largest_common_divisor_l358_358025


namespace value_of_n_l358_358073

noncomputable def bundle_cost : ℝ := 0.50 + 3 * 1.00 + 2 * 0.25
noncomputable def bundle_selling_price : ℝ := 4.60
noncomputable def profit_per_bundle : ℝ := bundle_selling_price - bundle_cost
noncomputable def nth_bundle_cost : ℝ := bundle_cost + 1.00
noncomputable def nth_bundle_selling_price : ℝ := 2.00
noncomputable def nth_bundle_loss : ℝ := nth_bundle_cost - nth_bundle_selling_price

theorem value_of_n : ∃ n : ℕ, n = 6 :=
  by
    let n := (nth_bundle_loss / profit_per_bundle).to_nat + 1
    use n
    have h : (n-1) * profit_per_bundle = nth_bundle_loss :=
      sorry
    linarith

end value_of_n_l358_358073


namespace monotonicity_intervals_l358_358201

noncomputable def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

theorem monotonicity_intervals (a b c : ℝ) (h_a : a > 0)
  (h1 : Deriv (f a b c) (-3) = 0)
  (h2 : Deriv (f a b c) 0 = 0)
  (h_min : ∃ x_min : ℝ, x_min = 0 ∧ f a b c x_min = -1) :
  (∀ x, x < -3 → Deriv (f a b c) x > 0) ∧
  (∀ x, -3 < x ∧ x < 0 → Deriv (f a b c) x < 0) ∧
  (∀ x, x > 0 → Deriv (f a b c) x > 0) ∧
  ∃ max_val,
  max_val = f 1 1 (-1) (-3) ∧ 
  max_val = 5 / Real.exp 3 :=
sorry

end monotonicity_intervals_l358_358201


namespace total_profit_l358_358708

theorem total_profit (Tom_investment : ℕ) (Jose_investment : ℕ) (Jose_share : ℕ) 
  (Tom_time : ℕ) (Jose_time : ℕ) : 
  Tom_investment = 30000 → 
  Jose_investment = 45000 → 
  Jose_share = 35000 → 
  Tom_time = 12 → 
  Jose_time = 10 → 
  (9 * (Jose_share / 5)) = 63000 := 
by
  intros hTom_investment hJose_investment hJose_share hTom_time hJose_time
  rw [hTom_investment, hJose_investment, hJose_share, hTom_time, hJose_time]
  exact eq.refl 63000

end total_profit_l358_358708


namespace rounding_problem_l358_358401

def round_nearest_hundredth (x : ℝ) : ℝ :=
  (Float.ofReal <| Real.round (100 * x) / 100)

theorem rounding_problem :
  ¬(round_nearest_hundredth 56.475999 = 56.47) ∧
  (round_nearest_hundredth 56.4657 = 56.47) ∧
  (round_nearest_hundredth 56.4705 = 56.47) ∧
  (round_nearest_hundredth 56.4695 = 56.47) ∧
  (round_nearest_hundredth 56.47345 = 56.47) :=
by
  sorry

end rounding_problem_l358_358401


namespace car_storm_distance_30_l358_358061

noncomputable def car_position (t : ℝ) : ℝ × ℝ :=
  (0, 3/4 * t)

noncomputable def storm_center (t : ℝ) : ℝ × ℝ :=
  (150 - (3/4 / Real.sqrt 2) * t, -(3/4 / Real.sqrt 2) * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem car_storm_distance_30 :
  ∃ (t : ℝ), distance (car_position t) (storm_center t) = 30 :=
sorry

end car_storm_distance_30_l358_358061


namespace find_a_div_b_theorem_l358_358335

noncomputable def find_a_div_b (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (a * x^2 + b * y^2 = 1) ∧ (y = 1 - x) →
  let k := (a / (a + b)) / (b / (a + b)) in
  k = (sqrt 3 / 2) →
  a / b = sqrt 3 / 2

theorem find_a_div_b_theorem (a b : ℝ) : find_a_div_b a b := 
sorry

end find_a_div_b_theorem_l358_358335


namespace arrangement_girls_boys_arrangement_teacher_students_l358_358799

-- Problem 1: Arrangement of 2 girls and 4 boys with the condition that girls must be together
theorem arrangement_girls_boys (girls boys : ℕ) (h_girls : girls = 2) (h_boys : boys = 4) :
  let total_people := girls + boys - 1 in
  let girl_permutations := 2! in
  (total_people)! * girl_permutations = 240 := by
  -- The detailed proof steps would go here
  sorry

-- Problem 2: Arrangement of a teacher and 4 students with the condition that the teacher cannot be at the ends
theorem arrangement_teacher_students (teacher students : ℕ) (h_teacher : teacher = 1) (h_students : students = 4) :
  let valid_positions_for_teacher := 3 in
  valid_positions_for_teacher * (students!) = 72 := by
  -- The detailed proof steps would go here
  sorry

end arrangement_girls_boys_arrangement_teacher_students_l358_358799


namespace three_digit_even_numbers_count_l358_358717

theorem three_digit_even_numbers_count : 
  ∃ (n : ℕ), 
  n = 360 ∧ (∀ (numbers : list ℕ), 
  list.length numbers = 3 ∧ 
  (∀ x ∈ numbers, x ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ 
  (numbers.nth 2).get_or_else 0 % 2 = 0 ∧ 
  (list.nodup numbers) 
  → n = 360) := 
sorry

end three_digit_even_numbers_count_l358_358717


namespace bertolli_farm_corn_l358_358857

theorem bertolli_farm_corn
  (tomatoes : ℕ)
  (onions : ℕ)
  (differential : ℕ)
  (h_tomatoes : tomatoes = 2073)
  (h_onions : onions = 985)
  (h_diff : onions = tomatoes + differential - 5200) :
  differential = 4039 :=
by {
  subst h_tomatoes,
  subst h_onions,
  rw [←h_diff],
  sorry
}

end bertolli_farm_corn_l358_358857


namespace total_age_l358_358046

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l358_358046


namespace total_travel_time_l358_358836

theorem total_travel_time
  (d : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (h_d : d = 120)
  (h_r1 : r1 = 50)
  (h_r2 : r2 ≈ 38.71) :
  (d / r1) + (d / r2) ≈ 5.5 :=
by 
  sorry

end total_travel_time_l358_358836


namespace seq_a_perfect_square_l358_358357

-- Definition of the sequence (a_n)
def seq_a : ℕ → ℤ 
| 0       := 1
| 1       := 1
| (n + 2) := 14 * seq_a (n + 1) - seq_a n

-- Theorem statement: stating that 2a_n - 1 is a perfect square for any n ≥ 0
theorem seq_a_perfect_square (n : ℕ) : 
  ∃ k : ℤ, 2 * (seq_a n) - 1 = k^2 := 
sorry

end seq_a_perfect_square_l358_358357


namespace sum_of_consecutive_integers_with_product_506_l358_358687

theorem sum_of_consecutive_integers_with_product_506 :
  ∃ x : ℕ, (x * (x + 1) = 506) → (x + (x + 1) = 45) :=
by
  sorry

end sum_of_consecutive_integers_with_product_506_l358_358687


namespace evaluate_f_f_neg_2_l358_358936

def f (x : ℝ) : ℝ :=
  if x >= 0 then
    x + 1
  else
    x^2 - 4

theorem evaluate_f_f_neg_2 : f (f (-2)) = 1 :=
by
  -- to be proven
  sorry

end evaluate_f_f_neg_2_l358_358936


namespace sum_with_permutation_not_all_ones_l358_358242

-- We need to define the concept of a number with no zeros
def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

-- We define the main theorem statement
theorem sum_with_permutation_not_all_ones (n : ℕ) (h : no_zero_digits n) :
  ¬ ∃ σ : List ℕ, (σ.permutes n.digits 10) ∧ (n + digits_to_nat 10 σ = repeat 1 (nat_length n)) :=
sorry

end sum_with_permutation_not_all_ones_l358_358242


namespace garments_fraction_l358_358856

theorem garments_fraction (bikini_fraction : ℝ) (trunk_fraction : ℝ) :
  bikini_fraction = 0.38 ∧ trunk_fraction = 0.25 → bikini_fraction + trunk_fraction = 0.63 :=
by
  intros h,
  cases h,
  sorry

end garments_fraction_l358_358856


namespace passing_percentage_l358_358447

theorem passing_percentage
  (marks_obtained : ℕ)
  (marks_failed_by : ℕ)
  (max_marks : ℕ)
  (h_marks_obtained : marks_obtained = 92)
  (h_marks_failed_by : marks_failed_by = 40)
  (h_max_marks : max_marks = 400) :
  (marks_obtained + marks_failed_by) / max_marks * 100 = 33 := 
by
  sorry

end passing_percentage_l358_358447


namespace fifth_term_in_expansion_l358_358392

theorem fifth_term_in_expansion (a x : ℝ) (h1 : x ≠ 0) : 
  (∑ k in Finset.range 9, (Nat.choose 8 k : ℝ) * (a / Real.sqrt x) ^ (8 - k) * (-Real.sqrt x / a^2) ^ k) = ∑ k in Finset.range 9, 70 / a^4 := 
sorry

end fifth_term_in_expansion_l358_358392


namespace probability_factor_of_6_l358_358436

theorem probability_factor_of_6! : 
  let nums := Finset.range 121 \ {0}
  let factors := {n ∈ nums | n ∣ (720 : ℕ)}
  ((factors.card : ℚ) / (nums.card : ℚ)) = 1 / 4 :=
by
  sorry

end probability_factor_of_6_l358_358436


namespace infinite_M_not_expressible_l358_358614

noncomputable def d (k : ℕ) : ℕ := sorry -- placeholder for the divisor function

theorem infinite_M_not_expressible :
  ∃ᶠ M in at_top, ¬ ∃ n : ℕ, M = (2 * (int.sqrt n) / d n)^2 := sorry

end infinite_M_not_expressible_l358_358614


namespace juniors_in_club_l358_358253

theorem juniors_in_club
  (j s x y : ℝ)
  (h1 : x = 0.4 * j)
  (h2 : y = 0.25 * s)
  (h3 : j + s = 36)
  (h4 : x = 2 * y) :
  j = 20 :=
by
  sorry

end juniors_in_club_l358_358253


namespace relationship_l358_358515

noncomputable def a : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def c : ℝ := Real.logb (3 / 5) (2 / 5)

theorem relationship : a < b ∧ b < c :=
by
  -- proof will go here
  sorry


end relationship_l358_358515


namespace triangle_area_l358_358837

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end triangle_area_l358_358837


namespace area_of_triangle_ABC_l358_358273

noncomputable def area_of_triangle {a b c : ℝ} (tanC : ℝ) (dot_CA_CB : ℝ) (sum_ab : ℝ) : ℝ :=
  if tanC = 3 * real.sqrt 7 ∧ dot_CA_CB = 5 / 2 ∧ sum_ab = 9 then
    let sinC := 3 * real.sqrt 7 / 8
    let cosC := 1 / 8
    let ab := 20
    in (1 / 2) * ab * sinC
  else
    0

theorem area_of_triangle_ABC
  {a b c : ℝ}
  (h1: a + b = 9)
  (h2: ∏ p in {a, b, c}, p > 0)
  (h3: (ab : ℝ) ∧ a * b * real.sqrt (1 - 1 / 64) = 20)
  (h4: tanC = 3 * real.sqrt 7)
  (h5: dot_CA_CB = 5 / 2)
  : area_of_triangle h4 h5 h1 = 15 * real.sqrt 7 / 4 := by
  sorry

end area_of_triangle_ABC_l358_358273


namespace velocity_zero_times_l358_358824

theorem velocity_zero_times :
  ∀ t : ℝ, (1 / 4 * t^4 - 5 / 3 * t^3 + 2 * t^2)' = 0 ↔ t = 0 ∨ t = 1 ∨ t = 4 :=
by
  sorry

end velocity_zero_times_l358_358824


namespace perimeter_of_square_with_area_36_l358_358408

theorem perimeter_of_square_with_area_36 : 
  ∀ (A : ℝ), A = 36 → (∃ P : ℝ, P = 24 ∧ (∃ s : ℝ, s^2 = A ∧ P = 4 * s)) :=
by
  sorry

end perimeter_of_square_with_area_36_l358_358408


namespace area_of_triangle_l358_358439

variables (yellow_area green_area blue_area : ℝ)
variables (is_equilateral_triangle : Prop)
variables (centered_at_vertices : Prop)
variables (radius_less_than_height : Prop)

theorem area_of_triangle (h_yellow : yellow_area = 1000)
                        (h_green : green_area = 100)
                        (h_blue : blue_area = 1)
                        (h_triangle : is_equilateral_triangle)
                        (h_centered : centered_at_vertices)
                        (h_radius : radius_less_than_height) :
  ∃ (area : ℝ), area = 150 :=
by
  sorry

end area_of_triangle_l358_358439


namespace both_participation_correct_l358_358423

-- Define the number of total participants
def total_participants : ℕ := 50

-- Define the number of participants in Chinese competition
def chinese_participants : ℕ := 30

-- Define the number of participants in Mathematics competition
def math_participants : ℕ := 38

-- Define the number of people who do not participate in either competition
def neither_participants : ℕ := 2

-- Define the number of people who participate in both competitions
def both_participants : ℕ :=
  chinese_participants + math_participants - (total_participants - neither_participants)

-- The theorem we want to prove
theorem both_participation_correct : both_participants = 20 :=
by
  sorry

end both_participation_correct_l358_358423


namespace possible_values_of_a_l358_358369

-- Define the main condition as a def
def condition (a b c x : ℤ) : Prop :=
  (x - a) * (x - 10) + 1 = (x + b) * (x + c)

-- State the theorem to prove
theorem possible_values_of_a :
  ∃ a : ℤ, (a = 8 ∨ a = 12) ∧ ∃ b c : ℤ, ∀ x : ℤ, condition a b c x :=
begin
  sorry
end

end possible_values_of_a_l358_358369


namespace Glenn_total_spent_l358_358638

theorem Glenn_total_spent :
  let monday_ticket := 5
  let wednesday_ticket := 2 * monday_ticket
  let saturday_ticket := 5 * monday_ticket
  let wednesday_discount := 0.10 * wednesday_ticket
  let wednesday_price := wednesday_ticket - wednesday_discount
  let saturday_extras := 7
  let saturday_price := saturday_ticket + saturday_extras
  wednesday_price + saturday_price = 41 := by
  -- Definitions
  let monday_ticket := 5
  let wednesday_ticket := 2 * monday_ticket
  let saturday_ticket := 5 * monday_ticket
  let wednesday_discount := 0.10 * wednesday_ticket
  let wednesday_price := wednesday_ticket - wednesday_discount
  let saturday_extras := 7
  let saturday_price := saturday_ticket + saturday_extras
  
  -- Calculation to check correctness
  have hw : wednesday_price = 9 := by
    calc
      wednesday_price = wednesday_ticket - wednesday_discount : by rfl
      ... = 10 - 1 : by rfl
      ... = 9 : rfl
  
  have hs : saturday_price = 32 := by
    calc
      saturday_price = saturday_ticket + saturday_extras : by rfl
      ... = 25 + 7 : by rfl
      ... = 32 : rfl
  
  show 9 + 32 = 41 from by
    calc
      9 + 32 = 41 : by rfl
    
  sorry

end Glenn_total_spent_l358_358638


namespace lattice_points_at_distance_4_l358_358255

theorem lattice_points_at_distance_4 :
  {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2 ^ 2 + p.3 ^ 2 = 16}.to_finset.card = 86 :=
sorry

end lattice_points_at_distance_4_l358_358255


namespace tile_floor_covering_l358_358040

theorem tile_floor_covering (n : ℕ) (h1 : 10 < n) (h2 : n < 20) (h3 : ∃ x, 9 * x = n^2) : n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end tile_floor_covering_l358_358040


namespace muirhead_inequality_prove_inequalities_l358_358647

def majorizes (α β : List ℕ) : Prop :=
  ∃ (n : ℕ), α.take n ∑ λ i => α.get i ≥ β.take n ∑ λ i => β.get i ∧ α.sum = β.sum

noncomputable def symmetric_mean (α : List ℕ) (a b c : ℝ) : ℝ := 
  sorry -- Implementation of the symmetric mean T_α(a, b, c)

theorem muirhead_inequality 
  (α β : List ℕ) (a b c : ℝ)
  (h_majorize: majorizes α β) :
  symmetric_mean α a b c ≥ symmetric_mean β a b c :=
sorry

theorem prove_inequalities 
  (a b c : ℝ) 
  (α := [2, 1, 1]) (β := [3, 1, 0]) (γ := [4, 0, 0])
  (h1 : majorizes α β)
  (h2 : majorizes β γ) 
: symmetric_mean α a b c ≤ symmetric_mean β a b c ∧ symmetric_mean β a b c ≤ symmetric_mean γ a b c :=
  by
  have h1' := muirhead_inequality α β a b c h1
  have h2' := muirhead_inequality β γ a b c h2
  exact ⟨h1', h2'⟩

end muirhead_inequality_prove_inequalities_l358_358647


namespace fraction_of_boxes_loaded_by_day_crew_l358_358784

-- Definitions based on the conditions
variables (D W : ℕ)  -- Day crew per worker boxes (D) and number of workers (W)

-- Helper Definitions
def boxes_day_crew : ℕ := D * W  -- Total boxes by day crew
def boxes_night_crew : ℕ := (3 * D / 4) * (3 * W / 4)  -- Total boxes by night crew
def total_boxes : ℕ := boxes_day_crew D W + boxes_night_crew D W  -- Total boxes by both crews

-- The main theorem
theorem fraction_of_boxes_loaded_by_day_crew :
  (boxes_day_crew D W : ℚ) / (total_boxes D W : ℚ) = 16/25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l358_358784


namespace greatest_possible_remainder_l358_358973

theorem greatest_possible_remainder (x : ℕ) (h: x % 7 ≠ 0) : (∃ r < 7, r = x % 7) ∧ x % 7 ≤ 6 := by
  sorry

end greatest_possible_remainder_l358_358973


namespace students_play_both_l358_358249

theorem students_play_both (total : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ) :
  total = 35 → F = 26 → L = 20 → N = 6 → B = F + L - (total - N) → B = 17 :=
by
  intros total_eq F_eq L_eq N_eq B_def
  rw [total_eq, F_eq, L_eq, N_eq] at B_def
  unfold total F L N B at *
  sorry

end students_play_both_l358_358249


namespace min_t_value_l358_358198

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
  ∀ (x y : ℝ), x ∈ Set.Icc (-3 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-3 : ℝ) (2 : ℝ)
  → |f (x) - f (y)| ≤ 20 :=
by
  sorry

end min_t_value_l358_358198


namespace infinite_geometric_series_first_term_l358_358455

theorem infinite_geometric_series_first_term (a r : ℝ) (h_sum : a / (1 - r) = 2020)
  (h_arith_seq : 2 * a * (r ^ 2) = a + a * (r ^ 3)) (h_r_abs : |r| < 1) : a = 1010 * (1 + real.sqrt 5) :=
by
  sorry

end infinite_geometric_series_first_term_l358_358455


namespace truncated_pyramid_base_sides_l358_358680

noncomputable def upper_base_side (S α β : ℝ) : ℝ :=
  sqrt (S * cos α * sin(α + β)^2 / (2 * sin α * sin (2 * β)))

noncomputable def lower_base_side (S α β : ℝ) : ℝ :=
  sin (α - β) * sqrt (S / (2 * sin α * sin (2 * β)))

theorem truncated_pyramid_base_sides (S α β a b : ℝ) :
  let a := upper_base_side S α β,
      b := lower_base_side S α β
  in a = sqrt (S * cos α * sin(α + β)^2 / (2 * sin α * sin (2 * β))) ∧
     b = sin (α - β) * sqrt (S / (2 * sin α * sin (2 * β))) := 
by  -- proof is omitted
  sorry

end truncated_pyramid_base_sides_l358_358680


namespace infinite_pairs_divisibility_l358_358615

theorem infinite_pairs_divisibility {n : ℕ} (hn : n ≥ 2)
  (a : ℕ → ℕ)
  (P : ℕ → ℕ)
  (hP : ∀ X, P X = X^n + (n-1).sum (λ k, a k * X^k) + 1)
  (ha_sym : ∀ k, k ∈ {1, 2, ..., n-1} → a k = a (n - k)) :
  ∃ (infinitely_many_pairs : ℕ × ℕ → Prop),
    (∀ x y, infinitely_many_pairs (x, y) → (x ≠ 0 ∧ y ≠ 0 ∧ x | P y ∧ y | P x))
:= sorry

end infinite_pairs_divisibility_l358_358615


namespace area_of_triangle_l358_358093

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (9, 0)
def C : (ℝ × ℝ) := (5, 7)

-- Function to calculate the area of a triangle given its vertices
def area_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Statement to prove
theorem area_of_triangle :
  area_triangle A B C = 25.5 :=
  sorry

end area_of_triangle_l358_358093


namespace cross_product_with_scalar_and_addition_l358_358576

def a_cross_b : ℝ^3 := ⟨2, -3, 4⟩
def c : ℝ^3 := ⟨1, 0, 1⟩

theorem cross_product_with_scalar_and_addition (a b : ℝ^3) (ha : a × b = a_cross_b) :
  a × (5 • b) + c = ⟨11, -15, 21⟩ := 
by sorry

end cross_product_with_scalar_and_addition_l358_358576


namespace mary_younger_than_albert_l358_358097

variable (A M B : ℕ)

noncomputable def albert_age := 4 * B
noncomputable def mary_age := A / 2
noncomputable def betty_age := 4

theorem mary_younger_than_albert (h1 : A = 2 * M) (h2 : A = 4 * 4) (h3 : 4 = 4) :
  A - M = 8 :=
sorry

end mary_younger_than_albert_l358_358097


namespace range_of_a_l358_358300

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 0.5 * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ -1 < a ∧ a < 0 := by
  sorry

end range_of_a_l358_358300


namespace number_of_students_absent_l358_358003

def classes := 18
def students_per_class := 28
def students_present := 496
def students_absent := (classes * students_per_class) - students_present

theorem number_of_students_absent : students_absent = 8 := 
by
  sorry

end number_of_students_absent_l358_358003


namespace intersection_with_x_axis_l358_358737

theorem intersection_with_x_axis : ∃ (x y : ℝ), (5 * y - 3 * x = 15) ∧ (y = 0) ∧ (x = -5) :=
by
  use [-5, 0]
  split
  { sorry }
  split
  { sorry }
  { sorry }

end intersection_with_x_axis_l358_358737


namespace parallel_A2B2_AB_l358_358257

-- Definitions
variables {A B C A_1 B_1 O A_2 B_2 : Type}
variable [plane_geometry: geometry A B C A_1 B_1 O A_2 B_2]

-- Conditions
axiom acute_triangle (h_acute : ∀ a b c, triangle a b c → acute a b c)
axiom altitude_AA1 (h_AA1 : altitude A A_1 C B O)
axiom altitude_BB1 (h_BB1 : altitude B B_1 A C O)
axiom altitude_A1A2 (h_A1A2 : altitude A_1 A_2 O B)
axiom altitude_B1B2 (h_B1B2 : altitude B_1 B_2 O A)

-- Theorem
theorem parallel_A2B2_AB (h_acute : acute_triangle A B C) 
                          (h_AA1 : altitude_AA1 A A_1 C B O)
                          (h_BB1 : altitude_BB1 B B_1 A C O)
                          (h_A1A2 : altitude_A1A2 A_1 A_2 O B)
                          (h_B1B2 : altitude_B1B2 B_1 B_2 O A) :
  parallel A_2 B_2 A B :=
sorry

end parallel_A2B2_AB_l358_358257


namespace original_number_l358_358821

theorem original_number (x : ℤ) (h : 5 * x - 9 = 51) : x = 12 :=
sorry

end original_number_l358_358821


namespace last_two_digits_of_sequence_l358_358384

theorem last_two_digits_of_sequence : 
  let last_two_digits := (8 + 2007 * 88) % 100 in
  last_two_digits = 24 := 
by
  -- Proof omitted
  sorry

end last_two_digits_of_sequence_l358_358384


namespace books_distribution_ways_l358_358070

theorem books_distribution_ways (num_books : ℕ) (library_books : ℕ) :
  num_books = 8 → 2 ≤ library_books → library_books ≤ 6 →
  ∃ (distribution_ways : ℕ), distribution_ways = 5 :=
begin
  intros h_books h_min_library h_max_library,
  use 5,
  sorry -- Proof omitted
end

end books_distribution_ways_l358_358070


namespace vector_projection_is_two_l358_358952

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (m n : ℝ)
variables (θ : ℝ)

noncomputable def vector_projection (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (a + b) ⬝ a / (∥a∥ * ∥a + b∥)

theorem vector_projection_is_two
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (angle_ab : real.angle (a : EuclideanSpace ℝ (Fin 2)) b = real.pi / 3) :
  vector_projection a b = 2 :=
sorry

end vector_projection_is_two_l358_358952


namespace evaluate_expression_l358_358481

theorem evaluate_expression : 
  (1 / 10 : ℝ) + (2 / 20 : ℝ) - (3 / 60 : ℝ) = 0.15 :=
by
  sorry

end evaluate_expression_l358_358481


namespace companyB_sells_350_bottles_l358_358380

def companyA_revenue := 300 * 4
def companyB_revenue(x : ℕ) := x * 3.5

theorem companyB_sells_350_bottles :
  ∃ x : ℕ, (companyA_revenue = companyB_revenue x + 25 ∨ companyB_revenue x = companyA_revenue + 25) → x = 350 :=
by
  sorry

end companyB_sells_350_bottles_l358_358380


namespace current_velocity_l358_358707

def distance_sound_travel (x y : ℝ) : ℝ :=
  Math.sqrt (x^2 + y^2)

def speed_of_motorboat := 20 -- 20 km/h in m/s

-- Given conditions translated directly from the problem:
def D := 100 -- Distance from the bank to the bridge in meters
def initial_x := 0
def initial_y := 50
def current_x := 62 -- Position of the current
noncomputable def sound_speed := 343 -- Speed of sound in air, in m/s

-- Problem to solve
theorem current_velocity (current : ℝ) : 
  ∀ (t : ℝ), 
  let siren_distance := distance_sound_travel (initial_x + D) 0 
  let sound_travelled := siren_distance / sound_speed
  let motorboat_distance := distance_sound_travel initial_x current_x 
  let motorboat_time := motorboat_distance / speed_of_motorboat 
  sound_travelled = motorboat_time → 
  current = 12.4 := 
sorry

end current_velocity_l358_358707


namespace sum_of_binary_digits_of_315_l358_358752

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358752


namespace sum_of_digits_in_binary_representation_of_315_l358_358766

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358766


namespace sqrt_of_16_eq_pm4_l358_358000

theorem sqrt_of_16_eq_pm4 : (sqrt 16 = 4 ∨ sqrt 16 = -4) :=
sorry

end sqrt_of_16_eq_pm4_l358_358000


namespace number_of_equilateral_triangles_in_lattice_l358_358865

-- Definitions representing the conditions of the problem
def is_unit_distance (a b : ℕ) : Prop :=
  true -- Assume true as we are not focusing on the definition

def expanded_hexagonal_lattice (p : ℕ) : Prop :=
  true -- Assume true as the specific construction details are abstracted

-- The target theorem statement
theorem number_of_equilateral_triangles_in_lattice 
  (lattice : ℕ → Prop) (dist : ℕ → ℕ → Prop) 
  (h₁ : ∀ p, lattice p → dist p p) 
  (h₂ : ∀ p, (expanded_hexagonal_lattice p) ↔ lattice p ∧ dist p p) : 
  ∃ n, n = 32 :=
by 
  existsi 32
  sorry

end number_of_equilateral_triangles_in_lattice_l358_358865


namespace range_of_m_l358_358531

def f (m x : ℝ) : ℝ := (1/3) * m * x^3 + x^2 + x
def g (m x : ℝ) : ℝ := 4 * Real.log (x + 1) + (1/2) * x^2 - (m - 1) * x

def f'_strictly_increasing (m x : ℝ) : Prop := (m * x^2 + 2 * x + 1) > 0
def g'_greater_than_one (m x : ℝ) : Prop := (4 / (x + 1) + x - m + 1) > 1

def prop_p (m : ℝ) : Prop := ∀ x : ℝ, 1 < x ∧ x < 2 → f'_strictly_increasing m x
def prop_q (m : ℝ) : Prop := ∀ x : ℝ, x > -1 → g'_greater_than_one m x

theorem range_of_m (m : ℝ) (h : prop_p m ∨ ¬ prop_q m) : (¬ prop_p m ∨ prop_q m) :=
sorry

end range_of_m_l358_358531


namespace cost_of_one_package_of_berries_l358_358633

noncomputable def martin_daily_consumption : ℚ := 1 / 2

noncomputable def package_content : ℚ := 1

noncomputable def total_period_days : ℚ := 30

noncomputable def total_spent : ℚ := 30

theorem cost_of_one_package_of_berries :
  (total_spent / (total_period_days * martin_daily_consumption / package_content)) = 2 :=
sorry

end cost_of_one_package_of_berries_l358_358633


namespace sin_36_eq_formula_l358_358656

theorem sin_36_eq_formula : sin (36 * (Real.pi / 180)) = (1 / 4) * Real.sqrt (10 - 2 * Real.sqrt 5) :=
by
  sorry

end sin_36_eq_formula_l358_358656


namespace final_expression_l358_358995

theorem final_expression (y : ℝ) : (3 * (1 / 2 * (12 * y + 3))) = 18 * y + 4.5 :=
by
  sorry

end final_expression_l358_358995


namespace shortest_distance_proof_l358_358397

-- Given conditions definitions

-- Non-intersecting lines definition
def non_intersecting_lines (l1 l2 : Line) : Prop :=¬ intersect l1 l2

-- Perpendicular segment definition
def perp_segment_length (p : Point) (l : Line) : Length := 
  perpendicular_segment_length p l

-- Shortest distance definition
def shortest_distance (p1 p2 : Point) : LineSegment :=
  line_segment p1 p2

-- Perpendicular through point
def unique_perpendicular (p : Point) (l : Line) : Prop :=
  ∃! m : Line, perpendicular p m l

-- Proof problem definition
theorem shortest_distance_proof (p1 p2 : Point) :
  shortest_distance p1 p2 = line_segment p1 p2 := 
  by
    sorry

end shortest_distance_proof_l358_358397


namespace candy_ratio_l358_358847

theorem candy_ratio (chocolate_bars M_and_Ms marshmallows total_candies : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : M_and_Ms = 7 * chocolate_bars)
  (h3 : total_candies = 25 * 10)
  (h4 : marshmallows = total_candies - chocolate_bars - M_and_Ms) :
  marshmallows / M_and_Ms = 6 :=
by
  sorry

end candy_ratio_l358_358847


namespace rosemary_leaves_count_l358_358104

-- Define the number of pots for each plant type
def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6

-- Define the number of leaves each plant type has
def basil_leaves : ℕ := 4
def thyme_leaves : ℕ := 30
def total_leaves : ℕ := 354

-- Prove that the number of leaves on each rosemary plant is 18
theorem rosemary_leaves_count (R : ℕ) (h : basil_pots * basil_leaves + rosemary_pots * R + thyme_pots * thyme_leaves = total_leaves) : R = 18 :=
by {
  -- Following steps are within the theorem's proof
  sorry
}

end rosemary_leaves_count_l358_358104


namespace find_value_of_m_l358_358577

theorem find_value_of_m (m : ℝ) (h : ∃ (x : ℝ), x = 3 ∧ x^2 - 2 * x + m = 0) : m = -3 :=
by
  obtain ⟨x, hx, hroot⟩ := h
  havesub x = 3 by exact hx
  have eq : x^2 - 2 * x + m = 0 := hroot
  rw havesub at eq
  norm_num at eq
  linarith

end find_value_of_m_l358_358577


namespace greatest_divisor_of_arithmetic_sequence_l358_358024

theorem greatest_divisor_of_arithmetic_sequence (x c : ℕ) : ∃ d, d = 15 ∧ ∀ S, S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_l358_358024


namespace number_smaller_than_neg2023_l358_358452

theorem number_smaller_than_neg2023 :
  let a := -2024
  let b := -2022
  let c := -2022.5
  let d := 0
  in a < -2023 := by
  sorry

end number_smaller_than_neg2023_l358_358452


namespace solution_x_chemical_b_l358_358323

theorem solution_x_chemical_b (percentage_x_a percentage_y_a percentage_y_b : ℝ) :
  percentage_x_a = 0.3 →
  percentage_y_a = 0.4 →
  percentage_y_b = 0.6 →
  (0.8 * percentage_x_a + 0.2 * percentage_y_a = 0.32) →
  (100 * (1 - percentage_x_a) = 70) :=
by {
  sorry
}

end solution_x_chemical_b_l358_358323


namespace rectangle_dimensions_l358_358353

theorem rectangle_dimensions (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w * l) = 2 * (2 * w + w)) :
  w = 6 ∧ l = 12 := 
by sorry

end rectangle_dimensions_l358_358353


namespace sum_of_digits_in_binary_representation_of_315_l358_358769

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358769


namespace domain_of_function_l358_358385

theorem domain_of_function (x : ℝ) : (∃ (f : ℝ → ℝ), f x = log 3 (log 4 (log 5 (log 6 x)))) 
    → x > 6 ^ (5 ^ 64) :=
by
    sorry

end domain_of_function_l358_358385


namespace infinite_n_for_sequence_l358_358314

theorem infinite_n_for_sequence :
  ∃ (S : Set ℕ), (∀ n ∈ S, 
    n > 0 ∧ 
    (∃ a b c : Finₓ n → ℕ,
      (∀ i : Finₓ n, a i + b i + c i ≡ 0 [MOD 6]) ∧
      (Finset.sum (Finset.univ : Finset (Finₓ n)) a ≡ 0 [MOD 6]) ∧
      (Finset.sum (Finset.univ : Finset (Finₓ n)) b ≡ 0 [MOD 6]) ∧
      (Finset.sum (Finset.univ : Finset (Finₓ n)) c ≡ 0 [MOD 6]))) ∧
  Set.Infinite S :=
sorry

end infinite_n_for_sequence_l358_358314


namespace point_on_unit_circle_after_arc_length_l358_358823

theorem point_on_unit_circle_after_arc_length (P Q : ℝ × ℝ) (θ : ℝ) (h_start : P = (1, 0)) (h_arc_length : θ = 2 * π / 3) :
  Q = (-1/2, sqrt(3)/2) :=
by
  sorry

end point_on_unit_circle_after_arc_length_l358_358823


namespace triangle_area_l358_358838

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end triangle_area_l358_358838


namespace smallest_x_abs_eq_9_l358_358150

theorem smallest_x_abs_eq_9 : ∃ x : ℝ, |x - 4| = 9 ∧ ∀ y : ℝ, |y - 4| = 9 → x ≤ y :=
by
  -- Prove there exists an x such that |x - 4| = 9 and for all y satisfying |y - 4| = 9, x is the minimum.
  sorry

end smallest_x_abs_eq_9_l358_358150


namespace closest_value_l358_358118

theorem closest_value
  (A B C D : ℝ)
  (optionA : A = 1500)
  (optionB : B = 1600)
  (optionC : C = 1700)
  (optionD : D = 1800)
  (x : ℝ)
  (hx : x = (0.00056 * 5210362) / 2) :
  B = 1600 ∧ abs (x - B) < abs (x - A) ∧ abs (x - B) < abs (x - C) ∧ abs (x - B) < abs (x - D) :=
begin
  sorry
end

end closest_value_l358_358118


namespace determine_sequence_once_l358_358639

-- Define the sequence and conditions
def sequence (n : ℕ) := list ℕ
def initial_red (r : ℕ) (b : ℕ) : Prop := r > b

-- Transformation rules
def transformation (k : ℕ) (r : ℕ) (b : ℕ) : ℕ × ℕ :=
(r * k + b, r)

-- Question statement
theorem determine_sequence_once (k : sequence n) (r_0 b_0 : ℕ) (h_0 : initial_red r_0 b_0) :
  ∃ (submissions : ℕ), submissions = 1 :=
by
  use 1
  sorry

end determine_sequence_once_l358_358639


namespace number_of_short_trees_planted_today_l358_358367

variable (short_trees_before short_trees_after short_trees_planted : ℕ)
variable (h1 : short_trees_before = 31)
variable (h2 : short_trees_after = 95)
variable (h3 : short_trees_planted = short_trees_after - short_trees_before)

theorem number_of_short_trees_planted_today : short_trees_planted = 64 :=
by
  rw [h1, h2, h3]
  norm_num

end number_of_short_trees_planted_today_l358_358367


namespace counting_4digit_integers_l358_358526

theorem counting_4digit_integers (x y : ℕ) (a b c d : ℕ) :
  (x = 1000 * a + 100 * b + 10 * c + d) →
  (y = 1000 * d + 100 * c + 10 * b + a) →
  (y - x = 3177) →
  (1 ≤ a) → (a ≤ 6) →
  (0 ≤ b) → (b ≤ 7) →
  (c = b + 2) →
  (d = a + 3) →
  ∃ n : ℕ, n = 48 := 
sorry

end counting_4digit_integers_l358_358526


namespace parallelogram_side_length_l358_358075

theorem parallelogram_side_length (s : ℝ) (h30 : real.sin (real.pi / 6) = 0.5)
  (area_eq : s * (s * 0.5) = 12 * real.sqrt 3) : s = 2 * real.sqrt 6 :=
sorry

end parallelogram_side_length_l358_358075


namespace ab_value_l358_358089

theorem ab_value (a b : ℝ) 
  (h1: a + b = 3 * real.sqrt 2)
  (ha : a = 4 * real.sqrt 2 * real.cos (real.pi / 6))
  (hb : b = 4 * real.sqrt 2 * real.sin (real.pi / 6))
  : a * b = 8 * real.sqrt 3 :=
by {
  sorry,
}

end ab_value_l358_358089


namespace sum_fraction_inequality_l358_358167

theorem sum_fraction_inequality (n : ℕ) (x : ℕ → ℝ)
  (h₁ : 2 ≤ n)
  (h₂ : ∑ i in Finset.range n, |x i| = 1)
  (h₃ : ∑ i in Finset.range n, x i = 0) :
  |∑ i in Finset.range n, x i / (i + 1)| ≤ 0.5 - 0.5 / n :=
sorry

end sum_fraction_inequality_l358_358167


namespace length_of_mn_is_sixteen_l358_358565

noncomputable def parabola (x y : ℝ) : Prop :=
  x^2 = 8 * y

noncomputable def line (x y : ℝ) : Prop :=
  y = x + 2

noncomputable def intersection_points : set (ℝ × ℝ) :=
  { p | ∃ x y, parabola x y ∧ line x y ∧ p = ⟨x, y⟩ }

noncomputable def chord_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt (2) * real.abs (p2.1 - p1.1)

theorem length_of_mn_is_sixteen :
  ∃ (M N : ℝ × ℝ), M ∈ intersection_points ∧ N ∈ intersection_points ∧ chord_length M N = 16 :=
sorry

end length_of_mn_is_sixteen_l358_358565


namespace distance_planes_eq_l358_358895

variable (x y z : ℝ)

def plane1 (x y z : ℝ) : ℝ := x - 3 * y + 2 * z + 4
def plane2 (x y z : ℝ) : ℝ := 2 * x - 6 * y + 4 * z + 9

def distance_between_planes : ℝ := (Real.abs ((1:ℝ) * (0:ℝ) + (-3:ℝ) * (0:ℝ) + 2 * (-2:ℝ) + 9 / 2)) / Real.sqrt (1 ^ 2 + (-3) ^ 2 + 2 ^ 2)

theorem distance_planes_eq (h₁ : plane1 0 0 (-2) = 0)
                           (h₂ : plane2 2 (-6) 4 = 9) :
    distance_between_planes = (Real.sqrt 14) / 28 := by
  sorry

end distance_planes_eq_l358_358895


namespace angle_between_vectors_l358_358241

open Complex

theorem angle_between_vectors 
  (z1 z2 : ℂ) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0)
  (h : abs (z1 + z2) = abs (z1 - z2)) : 
  angle z1 z2 = π / 2 :=
sorry

end angle_between_vectors_l358_358241


namespace quadratic_monotonically_increasing_l358_358592

open Interval

theorem quadratic_monotonically_increasing (m : ℝ) :
  (∀ x y : ℝ, 2 < x → x < y → y < +∞ → f x ≤ f y) ↔ m ≥ -4 :=
by
  let f := λ x : ℝ, x^2 + m*x - 2
  sorry

end quadratic_monotonically_increasing_l358_358592


namespace total_profit_Q2_is_correct_l358_358368

-- Conditions as definitions
def profit_Q1_A := 1500
def profit_Q1_B := 2000
def profit_Q1_C := 1000

def profit_Q2_A := 2500
def profit_Q2_B := 3000
def profit_Q2_C := 1500

def profit_Q3_A := 3000
def profit_Q3_B := 2500
def profit_Q3_C := 3500

def profit_Q4_A := 2000
def profit_Q4_B := 3000
def profit_Q4_C := 2000

-- The total profit calculation for the second quarter
def total_profit_Q2 := profit_Q2_A + profit_Q2_B + profit_Q2_C

-- Proof statement
theorem total_profit_Q2_is_correct : total_profit_Q2 = 7000 := by
  sorry

end total_profit_Q2_is_correct_l358_358368


namespace length_of_GH_l358_358606

variable (S_A S_C S_E S_F : ℝ)
variable (AB FE CD GH : ℝ)

-- Given conditions
axiom h1 : AB = 11
axiom h2 : FE = 13
axiom h3 : CD = 5

-- Relationships between the sizes of the squares
axiom h4 : S_A = S_C + AB
axiom h5 : S_C = S_E + CD
axiom h6 : S_E = S_F + FE
axiom h7 : GH = S_A - S_F

theorem length_of_GH : GH = 29 :=
by
  -- This is where the proof would go
  sorry

end length_of_GH_l358_358606


namespace california_vs_texas_license_plates_l358_358866

theorem california_vs_texas_license_plates :
  (26^4 * 10^4) - (26^3 * 10^3) = 4553200000 :=
by
  sorry

end california_vs_texas_license_plates_l358_358866


namespace find_length_AC_l358_358272

variable (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]

variable [HasDistance A B] [HasDistance B C] [HasDistance A M] [HasDistance B M] [HasDistance C M] [HasDistance A C] [HasDistance C B]

-- Condition: AB = 7
def AB : ℝ := 7

-- Condition: BC = 10
def BC : ℝ := 10

-- Condition: The length of median AM is 5
def AM : ℝ := 5

theorem find_length_AC (hAB : dist A B = AB) (hBC : dist B C = BC) (hAM : dist A M = AM) (hM_midpoint : dist B M = dist C M) : dist A C = Real.sqrt 51 := 
sorry

end find_length_AC_l358_358272


namespace trajectory_eq_l358_358698

theorem trajectory_eq {x y m : ℝ} (h : x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0) :
  x - 2 * y - 1 = 0 ∧ x ≠ 1 :=
sorry

end trajectory_eq_l358_358698


namespace slope_angle_of_tangent_line_expx_at_0_l358_358360

theorem slope_angle_of_tangent_line_expx_at_0 :
  let f := fun x : ℝ => Real.exp x 
  let f' := fun x : ℝ => Real.exp x
  ∀ x : ℝ, f' x = Real.exp x → 
  (∃ α : ℝ, Real.tan α = 1) →
  α = Real.pi / 4 :=
by
  intros f f' h_deriv h_slope
  sorry

end slope_angle_of_tangent_line_expx_at_0_l358_358360


namespace computer_price_increase_l358_358999

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : d * 1.2 = 351 := by
  sorry

end computer_price_increase_l358_358999


namespace ashok_subjects_l358_358106

variables (n T : ℕ)
variables (h1 : T = n * 70) (h2 : T = 370 + 50)

theorem ashok_subjects : n = 6 :=
by 
  have h_370_50 : 370 + 50 = 420 := rfl
  rw h_370_50 at h2
  rw [h1, h2]
  sorry

end ashok_subjects_l358_358106


namespace total_seven_flights_time_l358_358864

def time_for_nth_flight (n : ℕ) : ℕ :=
  25 + (n - 1) * 8

def total_time_for_flights (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => time_for_nth_flight (k + 1))

theorem total_seven_flights_time :
  total_time_for_flights 7 = 343 :=
  by
    sorry

end total_seven_flights_time_l358_358864


namespace bottle_nosed_dolphin_air_frequency_l358_358643

theorem bottle_nosed_dolphin_air_frequency :
  (∀ (time_beluga : ℝ) (time_dolphin : ℝ),
    time_beluga = 6 ∧ 
    (24*60 / time_dolphin) = (1.5 * (24*60 / time_beluga)) → 
    time_dolphin = 2.4) :=
by 
  intros time_beluga time_dolphin h
  cases h with h1 h2
  sorry

end bottle_nosed_dolphin_air_frequency_l358_358643


namespace correct_operation_l358_358036

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ (a^10 / a^2 = a^5) ∧
  ¬ ((2 * b^2)^3 = 6 * b^6) ∧
  (a^2 * a^(-4) = 1 / a^2) :=
by sorry

end correct_operation_l358_358036


namespace optimal_profit_game_l358_358715

noncomputable def optimal_profit : ℕ → ℕ
| n := 2^n

theorem optimal_profit_game (n : ℕ) : 
  let S := {x : ℕ | x < 4^n} in
  -- Players remove numbers alternately ensuring A maximizes and B minimizes the profit.
  ∃ a b ∈ S, a < b ∧ 
  ∀ (A_remove B_remove : finset ℕ), 
    A_remove.card = 2^(2*n-1) ∧ B_remove.card = 2^(2*n-2) →
    (b - a = optimal_profit n) :=
sorry

end optimal_profit_game_l358_358715


namespace find_a_l358_358207

theorem find_a (a : ℝ) (h : {0, -1, 2 * a} = {a - 1, -|a|, a + 1}) : a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l358_358207


namespace parallel_perpendicular_implies_perpendicular_l358_358628

-- Define parallel and perpendicular notation
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Conditions: m and n are lines, α is a plane.
-- Problem: Prove (m ∥ n ∧ n ⟂ α) → m ⟂ α.
theorem parallel_perpendicular_implies_perpendicular 
    (h1 : parallel m n) 
    (h2 : perpendicular n α) 
    : perpendicular m α := 
sorry

end parallel_perpendicular_implies_perpendicular_l358_358628


namespace median_family_size_l358_358597

open List

/-
Problem:
In Mr. Smith's history class, there are 15 students. Each student comes from a family with a different number of children. 
Assuming the 8th value from the ordered list of family sizes is 3 children, prove that 3 is the median number of children in the families.
-/

theorem median_family_size {n : ℕ} {l : List ℕ} (hn : n = 15) (hl : length l = 15) (hl_diff : nodup l) 
                          (l_sorted : sorted (≤) l) (h8 : nth l 7 = some 3) : 
  (median l).getD 0 = 3 :=
by
  sorry

end median_family_size_l358_358597


namespace problem_solution_l358_358390

theorem problem_solution :
  let x := (2023^2 - 2023 + 1) / 2023
  in x = 2022 + 1 / 2023 :=
by 
  let x := (2023^2 - 2023 + 1) / 2023
  sorry

end problem_solution_l358_358390


namespace sugar_for_90_cupcakes_l358_358068

theorem sugar_for_90_cupcakes : (∃ sugar_needed_for_90 : ℚ, sugar_needed_for_90 = 6) :=
by
  let sugar_per_cupcake : ℚ := 3 / 45
  let sugar_needed_for_90 := sugar_per_cupcake * 90
  have sugar_expression : sugar_needed_for_90 = 6 := by
    calc
      sugar_needed_for_90 = (3 / 45) * 90 : by rfl
      ... = 6 : by sorry
  exact ⟨sugar_needed_for_90, sugar_expression⟩

end sugar_for_90_cupcakes_l358_358068


namespace tim_trip_time_l358_358012

theorem tim_trip_time (driving_time: ℕ) (traffic_time: ℕ) (total_trip_time: ℕ)
  (h1: driving_time = 5)
  (h2: traffic_time = 2 * driving_time) 
  (h3: total_trip_time = driving_time + traffic_time) : 
  total_trip_time = 15 := 
by 
  intro driving_time traffic_time total_trip_time h1 h2 h3
  sorry

end tim_trip_time_l358_358012


namespace minimize_f_l358_358034

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 9

theorem minimize_f : ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ f x :=
begin
  use 3,
  split,
  {
    -- f(3) = 0
    unfold f,
    norm_num,
  },
  {
    -- ∀ y : ℝ, f y ≥ f 3
    intro y,
    calc f y = (y - 3) ^ 2 : by {ring}
        ... ≥ 0 : by {apply pow_two_nonneg},
  }
end

end minimize_f_l358_358034


namespace range_of_m_l358_358540

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h1 : 1/x + 1/y = 1) (h2 : x + y > m) : m < 4 := 
sorry

end range_of_m_l358_358540


namespace pq_sum_is_21_l358_358208

def M (p : ℝ) : set ℝ := {x | x^2 - p * x + 6 = 0}
def N (q : ℝ) : set ℝ := {x | x^2 + 6 * x - q = 0}

theorem pq_sum_is_21 (p q : ℝ) (hM : M p) (hN : N q) (h_inter : {x | x ∈ M p ∩ N q} = {2}) : p + q = 21 := by
  sorry

end pq_sum_is_21_l358_358208


namespace log_sum_value_l358_358389

theorem log_sum_value : 
  log 10 3 + 3 * log 10 4 + 2 * log 10 5 + 4 * log 10 2 + log 10 9 = 5.8399 :=
by sorry

end log_sum_value_l358_358389


namespace volume_of_cylinder_l358_358911

-- Define the problem conditions
def rectangle_length : ℝ := 20
def rectangle_width : ℝ := 10
def cylinder_height : ℝ := rectangle_width
def cylinder_radius : ℝ := rectangle_length / 2

-- State the theorem
theorem volume_of_cylinder :
  ∃ (V : ℝ), V = π * cylinder_radius^2 * cylinder_height ∧ V = 1000 * π := by
  sorry

end volume_of_cylinder_l358_358911


namespace selection_methods_count_l358_358435

theorem selection_methods_count :
  ∃ (classes : ℕ) (students : ℕ) (min_students : ℕ), classes = 10 ∧ students = 23 ∧ min_students = 2 ∧
  (∑ i in range classes, 2) + (multichoose students - (∑ i in range classes, min_students)) = 220 :=
by {
  sorry
}

end selection_methods_count_l358_358435


namespace boys_ages_l358_358694

theorem boys_ages (a b : ℕ) (h1 : a = b) (h2 : a + b + 11 = 29) : a = 9 :=
by
  sorry

end boys_ages_l358_358694


namespace hilary_ears_per_stalk_l358_358572

-- Define the given conditions
def num_stalks : ℕ := 108
def kernels_per_ear_half1 : ℕ := 500
def kernels_per_ear_half2 : ℕ := 600
def total_kernels_to_shuck : ℕ := 237600

-- Define the number of ears of corn per stalk as the variable to prove
def ears_of_corn_per_stalk : ℕ := 4

-- The proof problem statement
theorem hilary_ears_per_stalk :
  (54 * ears_of_corn_per_stalk * kernels_per_ear_half1) + (54 * ears_of_corn_per_stalk * kernels_per_ear_half2) = total_kernels_to_shuck :=
by
  sorry

end hilary_ears_per_stalk_l358_358572


namespace union_inter_complement_l358_358502

open Set

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | abs (x - 2) > 3})
variable (B : Set ℝ := {x | x * (-2 - x) > 0})

theorem union_inter_complement 
  (C_U_A : Set ℝ := compl A)
  (A_def : A = {x | abs (x - 2) > 3})
  (B_def : B = {x | x * (-2 - x) > 0})
  (C_U_A_def : C_U_A = compl A) :
  (A ∪ B = {x : ℝ | x < 0} ∪ {x : ℝ | x > 5}) ∧ 
  ((C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 0}) :=
by
  sorry

end union_inter_complement_l358_358502


namespace largest_triangle_area_l358_358699

noncomputable def area (base : ℝ) (height : ℝ) : ℝ :=
  (base * height) / 2

theorem largest_triangle_area :
  let A_base : ℝ := 8
  let A_height : ℝ := 6
  let B_base : ℝ := 5
  let B_height : ℝ := 9
  let C_base : ℝ := 7
  let C_height : ℝ := 7
  let AreaA := area A_base A_height
  let AreaB := area B_base B_height
  let AreaC := area C_base C_height
  AreaC > AreaA ∧ AreaC > AreaB
  sorry

end largest_triangle_area_l358_358699


namespace find_diameter_of_field_l358_358893

-- Definitions for the conditions
def cost_per_meter : ℝ := 1.50
def total_cost : ℝ := 122.52211349000194
def pi_approx : ℝ := 3.14159

-- Prove the diameter given the conditions
theorem find_diameter_of_field : 
  let circumference := total_cost / cost_per_meter in
  let diameter := circumference / pi_approx in
  abs (diameter - 26) < 1 :=
by
  sorry

end find_diameter_of_field_l358_358893


namespace Parallelogram_Area_Permimeter_Sum_l358_358076

theorem Parallelogram_Area_Permimeter_Sum :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b →
  let θ := (45 : ℝ) * (Real.pi / 180) in
  let A := a * b * Real.sin θ in
  let P := 2 * a + 2 * b in
  A + P ≠ 106 :=
by sorry

end Parallelogram_Area_Permimeter_Sum_l358_358076


namespace line_intersects_y_axis_at_origin_l358_358819

theorem line_intersects_y_axis_at_origin : 
  ∀ (x1 y1 x2 y2 : ℤ), (x1 = 5 ∧ y1 = 25) ∧ (x2 = -5 ∧ y2 = 5) →
  ∃ (y : ℤ), (x = 0) ∧ y = 15 :=
by {
  intros x1 y1 x2 y2 h,
  rcases h with ⟨⟨hx1, hy1⟩, ⟨hx2, hy2⟩⟩,
  have slope : (y2 - y1) / (x2 - x1) = 2,
  {
    -- proof of the slope goes here
    sorry,
  },
  use 15,
  split,
  {
    -- proof that x = 0
    sorry,
  },
  {
    -- proof that y = 15 given the slope and point (5, 25)
    sorry,
  }
}

end line_intersects_y_axis_at_origin_l358_358819


namespace sum_groups_is_250_l358_358460

-- Definitions based on the conditions
def group1 := [3, 13, 23, 33, 43]
def group2 := [7, 17, 27, 37, 47]

-- The proof problem
theorem sum_groups_is_250 : (group1.sum + group2.sum) = 250 :=
by
  sorry

end sum_groups_is_250_l358_358460


namespace sum_of_binary_digits_of_315_l358_358751

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358751


namespace find_roots_l358_358490

theorem find_roots (x : ℝ) :
  (x - 1) * (x - 2) * (x - 4) = x^3 - 7x^2 + 14x - 8 :=
by
  sorry

end find_roots_l358_358490


namespace length_of_de_l358_358043

variables (Point : Type) [LinearOrder Point]

-- Definition of 5 consecutive points on a straight line
variables (a b c d e : Point)
variables (ab bc cd de ae ac : ℕ)

-- Given conditions
def consecutive_points : Prop :=
  ab = 5 ∧
  ac = 11 ∧
  ae = 20 ∧
  bc = 3 * cd

-- The theorem to prove that the length of de is 7
theorem length_of_de (h : consecutive_points a b c d e ab bc cd de ae ac) : de = 7 :=
by {
  sorry
}

end length_of_de_l358_358043


namespace division_of_eggs_l358_358005

theorem division_of_eggs (students eggs : ℕ) (h_students : students = 7) (h_eggs : eggs = 56) : 
  eggs / students = 8 := by
  rw [h_students, h_eggs]
  norm_num
  sorry

end division_of_eggs_l358_358005


namespace calculate_area_of_triangle_PQR_l358_358275

structure Triangle (α β γ : Type) :=
(PQ : α)
(PR : α)
(angle_PQR : β)

noncomputable def triangle_area {α : Type} [LinearOrderedField α]
  {β : Type} [LinearOrderedSemiring β]
  [Algebra α β]
  (t : Triangle α β α)
  (angle_PQR_in_degrees : α) : β :=
  0.5 * t.PQ * t.PR * real.sin (π * angle_PQR_in_degrees / 180)

theorem calculate_area_of_triangle_PQR : ∀ (PQ PR : ℝ) (angle_PQR : ℝ),
  PQ = 30 → PR = 24 → angle_PQR = 60 → triangle_area {PQ := PQ, PR := PR, angle_PQR := angle_PQR} 60 = 180 * real.sqrt 3 :=
by
  intros PQ PR angle_PQR hPQ hPR hangle_PQR 
  rw [hPQ, hPR, hangle_PQR]
  unfold triangle_area
  rw [← div_mul_cancel ((30 : ℝ) * 24 * real.sin (π * (60 : ℝ) / 180)) (2 : ℝ)]
  rw [real.sin_pi_div_three]
  norm_num
  sorry

end calculate_area_of_triangle_PQR_l358_358275


namespace find_rate_of_current_l358_358362

open Real

theorem find_rate_of_current (c : ℝ) : (42 + c) * (44 / 60) = 36.67 -> c = 8 :=
by
  intro h
  have h1 : (42 + c) = 36.67 / (44 / 60) := by linarith
  have h2 : (36.67 / (44 / 60)) = 50 := by norm_num
  linarith

end find_rate_of_current_l358_358362


namespace domain_g_l358_358494

noncomputable def g (x : ℝ) := Real.tan (Real.arccos (x ^ 3))

theorem domain_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end domain_g_l358_358494


namespace cricket_team_members_l358_358370

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ := 27)
  (wk_age : ℕ := captain_age + 1)
  (total_avg_age : ℕ := 23)
  (remaining_avg_age : ℕ := total_avg_age - 1)
  (total_age : ℕ := n * total_avg_age)
  (captain_and_wk_age : ℕ := captain_age + wk_age)
  (remaining_age : ℕ := (n - 2) * remaining_avg_age) : n = 11 := 
by
  sorry

end cricket_team_members_l358_358370


namespace bike_rides_total_l358_358585

noncomputable def total_rides (B J M A : ℝ) : ℝ :=
  B + J + M + A

theorem bike_rides_total : 
  let Billy := 17
  let John := 2 * Billy
  let Mother := John + 10
  let SumJohnBilly := John + Billy
  let Amy := 3 * Real.sqrt SumJohnBilly
  in total_rides Billy John Mother Amy = 116 :=
by
  let Billy := 17
  let John := 2 * Billy
  let Mother := John + 10
  let SumJohnBilly := John + Billy
  let Amy := 3 * Real.sqrt SumJohnBilly
  show total_rides Billy John Mother Amy = 116 from sorry

end bike_rides_total_l358_358585


namespace sum_of_binary_digits_of_315_l358_358748

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358748


namespace better_packing_quality_l358_358212

noncomputable def packing_quality (X1 X2 : ℝ) : string :=
  if h : V(X1) > V(X2) = true then "Beta" else "Alpha"

theorem better_packing_quality (X1 X2 : ℝ) (E1 E2 : ℝ) :
  E(X1) = E(X2) ∧ V(X1) > V(X2) → packing_quality X1 X2 = "Beta" :=
by
  sorry

end better_packing_quality_l358_358212


namespace magnitude_of_complex_num_l358_358134

-- Definition of the complex number as a hypothesis
def complex_num : ℂ := (3 / 5 : ℝ) - (5 / 4 : ℝ) * complex.I

-- The statement to prove
theorem magnitude_of_complex_num :
  complex_num.abs = real.sqrt 769 / 20 :=
by sorry

end magnitude_of_complex_num_l358_358134


namespace relationship_among_a_b_c_l358_358538

def f (x : Real) : Real := Real.cos x
def a : Real := f (Real.log 2)
def b : Real := f (Real.log Real.pi)
def c : Real := f (Real.log (1 / 3))

theorem relationship_among_a_b_c : a > c ∧ c > b :=
by
  unfold f a b c
  -- Proof goes here
  sorry

end relationship_among_a_b_c_l358_358538


namespace max_value_y_l358_358579

theorem max_value_y (x : ℝ) : (y = -x^2 + 5) → (∀ x : ℝ, y ≤ 5) :=
by
  intro h
  have h₁ : ∀ x : ℝ, -x^2 ≤ 0,
  from assume x, (neg_nonpos.mpr (pow_two_nonneg _))
  have h₂ : ∀ x : ℝ, -x^2 + 5 ≤ 0 + 5,
  from assume x, add_le_add_right (h₁ x) 5
  exact h₂

end max_value_y_l358_358579


namespace number_of_lattice_points_l358_358432

open Real

def is_lattice_point (x y : ℤ) : Prop := y = |(x : ℝ)| ∨ y = -x^2 + 4

def region_lattice_points_count : ℕ :=
  (if is_lattice_point (-2, 2) then 1 else 0) +
  (if is_lattice_point (-1, 1) ∨ is_lattice_point (-1, 3) then 2 else 0) +
  (if is_lattice_point (0, 0) ∨ is_lattice_point (0, 4) then 2 else 0) +
  (if is_lattice_point (1, 1) ∨ is_lattice_point (1, 3) then 2 else 0)

theorem number_of_lattice_points : region_lattice_points_count = 7 :=
  sorry

end number_of_lattice_points_l358_358432


namespace alphametic_puzzle_l358_358610

theorem alphametic_puzzle (I D A M E R O : ℕ) 
  (h1 : R = 0) 
  (h2 : D + E = 10)
  (h3 : I + M + 1 = O)
  (h4 : A = D + 1) :
  I + 1 + M + 10 + 1 = O + 0 + A := sorry

end alphametic_puzzle_l358_358610


namespace compare_three_numbers_l358_358874

open Real -- Open the Real namespace for real number functions

theorem compare_three_numbers (a b c : ℝ) (h1 : a = 3^0.7) (h2 : b = 0.7^3) (h3 : c = log 3 0.7) :
  a > b ∧ b > c :=
by
  sorry

end compare_three_numbers_l358_358874


namespace trajectory_of_P_l358_358178

-- Definitions based on the conditions
def point (A : Type) : Type := A × A
def line (A : Type) : Type := { l // ∃ a b : A, ∀ x, l x = a * x + b }
def trajectory_equation (α : Type) [Field α] := α → α → Prop

-- Given conditions
variable (α : Type) [Field α]
variable (A : point α)
variable (l : line α)

-- Define points A and line l according to the problem
noncomputable def A := (1 : α, 0 : α)
noncomputable def l : α → α := λ x, 2 * x - 4

-- Main statement to prove
theorem trajectory_of_P : ∀ P, (α → α → Prop) :=
  λ P, P x y ↔ y = 2 * x

end trajectory_of_P_l358_358178


namespace part_I_part_II_l358_358191

-- Part (I) 
theorem part_I (a b : ℝ) : (∀ x : ℝ, x^2 - 5 * a * x + b > 0 ↔ (x > 4 ∨ x < 1)) → 
(a = 1 ∧ b = 4) :=
by { sorry }

-- Part (II) 
theorem part_II (x y : ℝ) (a b : ℝ) (h : x + y = 2 ∧ a = 1 ∧ b = 4) : 
x > 0 → y > 0 → 
(∃ t : ℝ, t = a / x + b / y ∧ t ≥ 9 / 2) :=
by { sorry }

end part_I_part_II_l358_358191


namespace largest_of_five_consecutive_composite_integers_under_40_l358_358506

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l358_358506


namespace count_valid_sets_l358_358616

def is_isolated_element (A : set ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k-1 ∉ A ∧ k+1 ∉ A

def no_isolated_elements (A : set ℤ) : Prop :=
  ∀ k, is_isolated_element A k → false

def S : set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def set_of_3_elements : finset (set ℤ) := 
  finset.image (λ t : finset ℤ, t.to_set) (finset.powerset_len 3 S.to_finset)

def valid_sets (sets : finset (set ℤ)) : finset (set ℤ) := 
  finset.filter no_isolated_elements sets

theorem count_valid_sets : (valid_sets set_of_3_elements).card = 6 := 
  by
    sorry

end count_valid_sets_l358_358616


namespace thomas_training_days_proof_l358_358703

variable (x : ℕ)

-- Conditions
def training_hours_per_day : ℕ := 5
def additional_days : ℕ := 12
def total_training_hours_after_additional_days : ℕ := 210

-- Proposition
def thomas_training_days := 
  5 * x + 5 * additional_days = total_training_hours_after_additional_days → x = 30

-- Skipping the actual proof
theorem thomas_training_days_proof : thomas_training_days :=
  sorry

end thomas_training_days_proof_l358_358703


namespace power_div_multiply_l358_358111

theorem power_div_multiply (a b c : ℕ) (h₁ : a = 8^3) (h₂ : b = 8^2) (h₃ : c = 2^{10}) :
  (a / b) * c = 8192 := by
  sorry

end power_div_multiply_l358_358111


namespace largest_composite_in_five_consecutive_ints_l358_358508

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l358_358508


namespace area_of_region_l358_358571

noncomputable def condition (x y : ℝ) := x^2 + y^2 + 4 * (x - |y|) ≤ 0

theorem area_of_region :
  (∫∫ (condition x y), 1) = 12 * Real.pi + 8 :=
sorry

end area_of_region_l358_358571


namespace area_ratio_l358_358138

noncomputable def A := (1 : ℝ, 0 : ℝ, 0 : ℝ)
noncomputable def B := (0 : ℝ, 1 : ℝ, 0 : ℝ)
noncomputable def C := (0 : ℝ, 0 : ℝ, 1 : ℝ)
noncomputable def D := (0 : ℝ, 13 / 20 : ℝ, 7 / 20 : ℝ)
noncomputable def E := (5 / 8 : ℝ, 0 : ℝ, 3 / 8 : ℝ)

theorem area_ratio :
  let area := λ (P Q R : ℝ × ℝ × ℝ), (1 / 2) * abs (P.1 * (Q.2 * R.3 - Q.3 * R.2) + P.2 * (Q.3 * R.1 - Q.1 * R.3) + P.3 * (Q.1 * R.2 - Q.2 * R.1))
  in (area C D (7/19 : ℝ, 39/95 : ℝ, 21/95 : ℝ)) / (area C E (7/19 : ℝ, 39/95 : ℝ, 21/95 : ℝ)) = 14 / 15 :=
by
  sorry

end area_ratio_l358_358138


namespace compare_abc_l358_358554

noncomputable def f (x : ℝ) : ℝ := 2 ^ (|x|)

noncomputable def a : ℝ := f (Real.logb 0.5 3)
noncomputable def b : ℝ := Real.log 2 5
noncomputable def c : ℝ := f 0

theorem compare_abc : c < b ∧ b < a := 
by {
  have h1 : a = f (Real.logb 0.5 3) := rfl,
  have h2 : b = Real.log 2 5 := rfl,
  have h3 : c = f 0 := rfl,
  -- Proof needs to be added here
  sorry
}

end compare_abc_l358_358554


namespace problem1_problem2_problem3_l358_358520

-- Given function definition
def f (x : ℝ) (m : ℝ) : ℝ := m - (2 / (5^x + 1))

-- Problem 1: Monotonicity of the function
theorem problem1 (m : ℝ) : 
  ∀ x y : ℝ, x < y → f x m < f y m := sorry

-- Problem 2: Odd function implies m = 1
theorem problem2 (m : ℝ) (h : ∀ x : ℝ, f x m + f (-x) m = 0) : 
  m = 1 := sorry

-- Problem 3: Range of m given D ⊂ [-3, 1]
theorem problem3 (m : ℝ) (hD : ∀ y : ℝ, (∃ x : ℝ, y = f x m) → y ∈ set.Icc (-3 : ℝ) 1) : 
  -1 ≤ m ∧ m ≤ 1 := sorry

end problem1_problem2_problem3_l358_358520


namespace distance_interval_l358_358098

-- Define the conditions based on the false statements
def Alice_statement_false (d : ℝ) : Prop := d < 8
def Bob_statement_false (d : ℝ) : Prop := d > 7
def Charlie_statement_false (d : ℝ) : Prop := d > 6

-- Formal statement of the problem to prove
theorem distance_interval (d : ℝ) 
  (hA : Alice_statement_false d) 
  (hB : Bob_statement_false d) 
  (hC : Charlie_statement_false d) :
  7 < d ∧ d < 8 :=
by
  split
  · exact hB
  · exact hA

end distance_interval_l358_358098


namespace distance_AB_l358_358262

def C1_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.cos θ

def C2_polar (ρ θ : Real) : Prop :=
  ρ^2 * (1 + (Real.sin θ)^2) = 2

def ray_polar (θ : Real) : Prop :=
  θ = Real.pi / 6

theorem distance_AB :
  let ρ1 := 2 * Real.cos (Real.pi / 6)
  let ρ2 := Real.sqrt 10 * 2 / 5
  |ρ1 - ρ2| = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  sorry

end distance_AB_l358_358262


namespace prob_X_greater_than_4_l358_358957

noncomputable def normalDist (μ σ : ℝ) : Measure ℝ := sorry

variable (μ σ : ℝ) [isNormalDist : isProbabilityDistribution (normalDist μ σ)]
variable (X : ℝ → ℝ)
variable (h1 : μ = 2)
variable (h2 : ∃ σ, X ~ normalDist μ σ)
variable (h3 : P (fun x => 0 < X x ∧ X x < 4) = 0.8)

-- We need to express the aim of the problem
theorem prob_X_greater_than_4 : P (λ x => X x > 4) = 0.1 :=
by
  sorry

end prob_X_greater_than_4_l358_358957


namespace jesse_tiles_needed_l358_358279

-- Define the length and width of the room, and the area of each tile
def room_length : ℕ := 2
def room_width : ℕ := 12
def tile_area : ℕ := 4

-- Calculate the area of the room
def room_area : ℕ := room_length * room_width

-- Calculate the number of tiles needed
def number_of_tiles_needed : ℕ := room_area / tile_area

-- Statement of the proof problem
theorem jesse_tiles_needed : number_of_tiles_needed = 6 := by
  def room_length : ℕ := 2
  def room_width : ℕ := 12
  def tile_area : ℕ := 4
  def room_area : ℕ := 2 * 12
  def number_of_tiles_needed : ℕ := room_area / tile_area
  sorry

end jesse_tiles_needed_l358_358279


namespace correct_area_l358_358596

-- Define the main elements involved in the problem
structure Triangle :=
  (A B C : (ℚ × ℚ))

def circumcenter (T : Triangle) : (ℚ × ℚ) :=
  let ⟨ax, ay⟩ := T.A
  let ⟨bx, by⟩ := T.B
  let ⟨cx, cy⟩ := T.C
  ((bx + cx) / 2, (by + cy) / 2)

def incenter (T : Triangle) : (ℚ × ℚ) :=
  let ⟨ax, ay⟩ := T.A
  let ⟨bx, by⟩ := T.B
  let ⟨cx, cy⟩ := T.C
  let a := (bx - cx)^2 + (by - cy)^2
  let b := (ax - cx)^2 + (ay - cy)^2
  let c := (ax - bx)^2 + (ay - by)^2
  ((a * ax + b * bx + c * cx) / (a + b + c), (a * ay + b * by + c * cy) / (a + b + c))

def external_tangent_circle_center (T : Triangle) : (ℚ × ℚ) :=
  (2, 3.75) -- Coordinates determined from problem statement

def area_of_triangle (P Q R : (ℚ × ℚ)) : ℚ :=
  (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def problem :=
  let T : Triangle := {A := (0, 0), B := (8, 0), C := (0, 15)}
  let O := circumcenter T
  let I := incenter T
  let M := external_tangent_circle_center T
  area_of_triangle M O I = 4.5
  
theorem correct_area : problem := by
  sorry

end correct_area_l358_358596


namespace transformation_correct_l358_358039

theorem transformation_correct (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
sorry

end transformation_correct_l358_358039


namespace total_worth_of_travelers_checks_l358_358448

theorem total_worth_of_travelers_checks 
  (x y : ℕ) 
  (h1 : x + y = 30) 
  (h2 : x ≥ 24) 
  (remaining_checks_count : x - 24 + y = 6)
  (average_value_is_100 : (x - 24) * 50 + y * 100 = 600) : 
  (x * 50 + y * 100) = 1800 := 
begin
  sorry
end

end total_worth_of_travelers_checks_l358_358448


namespace broadcasting_methods_count_l358_358669

-- Defining the given conditions
def num_commercials : ℕ := 4 -- number of different commercial advertisements
def num_psa : ℕ := 2 -- number of different public service advertisements
def total_slots : ℕ := 6 -- total number of slots for commercials

-- The assertion we want to prove
theorem broadcasting_methods_count : 
  (num_psa * (total_slots - num_commercials - 1) * (num_commercials.factorial)) = 48 :=
by sorry

end broadcasting_methods_count_l358_358669


namespace terminal_side_of_angle_l358_358589

theorem terminal_side_of_angle (α : ℝ) (h1 : sin α < 0) (h2 : cos α > 0) : 
  -- α is in the fourth quadrant
  α ∈ set_of (λ α : ℝ, sin α < 0 ∧ cos α > 0) :=
by {
  split,
  assumption,
  assumption,
}

end terminal_side_of_angle_l358_358589


namespace find_cost_price_l358_358782

theorem find_cost_price
  (C : ℝ) 
  (h1 : let SP := 1.25 * C in SP = SP) 
  (h2 : let NCP := 0.80 * C in NCP = NCP) 
  (h3 : let NSP := 1.25 * C - 8.40 in NSP = NSP)
  (h4 : let G := 0.30 * (0.80 * C) in NSP = 0.80 * C + G) :
  C = 40 :=
by
  sorry

end find_cost_price_l358_358782


namespace necessary_but_not_sufficient_for_positive_roots_l358_358845

theorem necessary_but_not_sufficient_for_positive_roots 
  (a b c : ℝ) : 
  (b^2 - 4 * a * c ≥ 0) ∧ (a * c > 0) ∧ (a * b < 0) 
  → (∀ x : ℝ, a * x^2 + b * x + c = 0 → x > 0) :=
by {
  intro h,
  sorry
}

end necessary_but_not_sufficient_for_positive_roots_l358_358845


namespace range_of_a_l358_358688

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + 6 * a * x + 1 < 0) : 
  a ∈ set.Icc (-1 / 3) (1 / 3) :=
by 
  sorry

end range_of_a_l358_358688


namespace fourth_student_in_sample_of_systematic_sampling_l358_358244

theorem fourth_student_in_sample_of_systematic_sampling
  (students : List ℕ)
  (sample : List ℕ)
  (num_students : ℕ)
  (sample_size : ℕ)
  (systematic_sampling : Prop)
  (student_nums : students = List.range (num_students + 1))
  (sample_nums : sample = [6, 34, 48, 20]) :
  sample.nth 3 = some 20 := by
  sorry

end fourth_student_in_sample_of_systematic_sampling_l358_358244


namespace total_amount_spent_l358_358855

def cost_per_dozen_apples : ℕ := 40
def cost_per_dozen_pears : ℕ := 50
def dozens_apples : ℕ := 14
def dozens_pears : ℕ := 14

theorem total_amount_spent : (dozens_apples * cost_per_dozen_apples + dozens_pears * cost_per_dozen_pears) = 1260 := 
  by
  sorry

end total_amount_spent_l358_358855


namespace division_by_ab_plus_one_is_perfect_square_l358_358296

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end division_by_ab_plus_one_is_perfect_square_l358_358296


namespace equation_of_circle_l358_358495

-- Definitions directly based on conditions 
noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)
noncomputable def directrix_of_parabola : ℝ × ℝ -> Prop
  | (x, _) => x = -1

-- The statement of the problem: equation of the circle with given conditions
theorem equation_of_circle : ∃ (r : ℝ), (∀ (x y : ℝ), (x - 1)^2 + y^2 = r^2) ∧ r = 2 :=
sorry

end equation_of_circle_l358_358495


namespace solve_eq_l358_358327

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l358_358327


namespace probability_is_one_tenth_l358_358858

-- Defining the universe of digits and letters
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def primes := {2, 3, 5, 7}
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def lettersAC := {'A', 'B', 'C'}
def evens := {0, 2, 4, 6, 8}

-- Probability calculation definitions
noncomputable def probability_prime : ℚ := ↑(primes.toFinset.card) / ↑(digits.toFinset.card)
noncomputable def probability_letterAC : ℚ := ↑(lettersAC.toFinset.card) / ↑(letters.toFinset.card)
noncomputable def probability_even : ℚ := ↑(evens.toFinset.card) / ↑(digits.toFinset.card)

-- Combined probability
noncomputable def combined_probability : ℚ := probability_prime * probability_letterAC * probability_even

-- The statement to be proved
theorem probability_is_one_tenth :
  combined_probability = 1 / 10 :=
sorry

end probability_is_one_tenth_l358_358858


namespace cost_of_fencing_each_side_l358_358234

theorem cost_of_fencing_each_side (total_cost : ℕ) (x : ℕ) (h : total_cost = 276) (hx : 4 * x = total_cost) : x = 69 :=
by {
  sorry
}

end cost_of_fencing_each_side_l358_358234


namespace cat_moves_on_circular_arc_l358_358431

theorem cat_moves_on_circular_arc (L : ℝ) (x y : ℝ)
  (h : x^2 + y^2 = L^2) :
  (x / 2)^2 + (y / 2)^2 = (L / 2)^2 :=
  by sorry

end cat_moves_on_circular_arc_l358_358431


namespace find_positive_real_solutions_l358_358492

open Real

theorem find_positive_real_solutions 
  (x : ℝ) 
  (h : (1/3 * (4 * x^2 - 2)) = ((x^2 - 60 * x - 15) * (x^2 + 30 * x + 3))) :
  x = 30 + sqrt 917 ∨ x = -15 + (sqrt 8016) / 6 :=
by sorry

end find_positive_real_solutions_l358_358492


namespace age_double_condition_l358_358072

theorem age_double_condition (S M X : ℕ) (h1 : S = 44) (h2 : M = S + 46) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end age_double_condition_l358_358072


namespace intersection_A_B_l358_358969

namespace MathProof

open Set

def A := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2 * x + 6}

theorem intersection_A_B : A ∩ B = Icc (-1 : ℝ) 7 :=
by
  sorry

end MathProof

end intersection_A_B_l358_358969


namespace translation_even_property_l358_358013

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x - cos x

theorem translation_even_property (m : ℝ) (h : m > 0) :
  ∀ x : ℝ, f (x - m) = f (-x + m) ↔ m = π / 3 := by
  sorry

end translation_even_property_l358_358013


namespace required_butter_l358_358441

-- Define the given conditions
variables (butter sugar : ℕ)
def recipe_butter : ℕ := 25
def recipe_sugar : ℕ := 125
def used_sugar : ℕ := 1000

-- State the theorem
theorem required_butter (h1 : butter = recipe_butter) (h2 : sugar = recipe_sugar) :
  (used_sugar * recipe_butter) / recipe_sugar = 200 := 
by 
  sorry

end required_butter_l358_358441


namespace quadratic_min_value_correct_l358_358157

theorem quadratic_min_value_correct : ∀ x : ℝ, ∃ y_min : ℝ, y_min = 2 ∧ (∃ x0 : ℝ, y_min = (x0 - 1)^2 + 2) :=  
by {
  intro x,
  use 2,
  split,
  { reflexivity, },
  { use 1,
    reflexivity, }
}

end quadratic_min_value_correct_l358_358157


namespace count_3digit_numbers_l358_358220

theorem count_3digit_numbers (digits : Finset ℕ) (length_digits : digits.card = 5) (no_repeats : ∀ x ∈ digits, ∀ y ∈ digits, x ≠ y → x ≠ y)
  (min_digit_greater_than_two : ∀ d ∈ digits, d > 2 → ∃ x ∈ digits, x = 3 ∨ x = 5 ∨ x = 6):
    (∃ n : ℕ, ∃ l : List ℕ, List.Perm (digits.to_list) l ∧ List.length l = 3 ∧ l.nth_le 0 sorry > 2 ∧ (l.nth_le 1 sorry ≠ l.nth_le 0 sorry) ∧ (l.nth_le 2 sorry ≠ l.nth_le 1 sorry) ∧ (l.nth_le 2 sorry ≠ l.nth_le 0 sorry)) → 
    ∃ lsts : List (List ℕ), lsts.length = 36 :=
sorry

end count_3digit_numbers_l358_358220


namespace intersection_point_polar_coordinates_l358_358270

theorem intersection_point_polar_coordinates :
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ 
    ∀ ρ : ℝ, ρ = cos θ + sin θ ∧ ρ * sin (θ - π/4) = sqrt 2 / 2 → 
    ρ = 1 ∧ θ = π/2) :=
begin
  sorry
end

end intersection_point_polar_coordinates_l358_358270


namespace quadratic_minimum_value_l358_358155

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end quadratic_minimum_value_l358_358155


namespace perpendicular_lines_necessary_not_sufficient_l358_358001

variables (l : Line) (α : Plane)
variables (countless_perpendicular_lines : ∀ (m : Line), m ∈ α → m ⟂ l)

theorem perpendicular_lines_necessary_not_sufficient : 
  (∀ m : Line, m ∈ α → m ⟂ l) ↔ (∃ n : Line, n ∈ α ∧ n ⟂ l) := 
sorry

end perpendicular_lines_necessary_not_sufficient_l358_358001


namespace men_work_in_80_days_l358_358237

theorem men_work_in_80_days (x : ℕ) (work_eq_20men_56days : x * 80 = 20 * 56) : x = 14 :=
by 
  sorry

end men_work_in_80_days_l358_358237


namespace x_pow_twelve_l358_358991

theorem x_pow_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 = 322 :=
sorry

end x_pow_twelve_l358_358991


namespace sum_of_digits_base2_315_l358_358756

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358756


namespace perimeter_of_remaining_area_l358_358079

def paper_width : ℕ := 12
def paper_length : ℕ := 16
def margin : ℕ := 2

def remaining_width : ℕ := paper_width - 2 * margin
def remaining_length : ℕ := paper_length - 2 * margin

def perimeter (w l : ℕ) : ℕ := 2 * (w + l)

theorem perimeter_of_remaining_area : perimeter remaining_width remaining_length = 40 := 
by
  -- conditions given in the problem
  have h1 : remaining_width = 8 := by
    have h2 : 2 * margin = 4 := by rfl
    calc
      remaining_width = paper_width - 2 * margin := by rfl
      ... = 12 - 4 := by rw [← h2]
      ... = 8 := by norm_num
  have h3 : remaining_length = 12 := by
    have h4 : 2 * margin = 4 := by rfl
    calc
      remaining_length = paper_length - 2 * margin := by rfl
      ... = 16 - 4 := by rw [← h4]
      ... = 12 := by norm_num
  -- using the calculated dimensions for the final proof
  calc 
    perimeter remaining_width remaining_length = perimeter 8 12 := by rw [h1, h3]
    ... = 2 * (8 + 12) := by rfl
    ... = 2 * 20 := by norm_num
    ... = 40 := by norm_num

end perimeter_of_remaining_area_l358_358079


namespace find_common_difference_l358_358667

-- Define the arithmetic series sum formula
def arithmetic_series_sum (a₁ : ℕ) (d : ℚ) (n : ℕ) :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the first day's production, total days, and total fabric
def first_day := 5
def total_days := 30
def total_fabric := 390

-- The proof statement
theorem find_common_difference : 
  ∃ d : ℚ, arithmetic_series_sum first_day d total_days = total_fabric ∧ d = 16 / 29 :=
by
  sorry

end find_common_difference_l358_358667


namespace more_than_200_marbles_day_l358_358800

noncomputable def numMarbles (n : ℕ) : ℕ :=
  3 * 2^n

theorem more_than_200_marbles_day :
  ∃ n : ℕ, numMarbles n > 200 ∧ ∀ m < n, numMarbles m ≤ 200 :=
by
  have nat_four_seven := (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ nat.zero_le)))))))
  have p0 : numMarbles 7 = 384 := by norm_num
  have p1 : 200 < numMarbles 7 := by norm_num
  exact ⟨7, p1, sorry⟩

end more_than_200_marbles_day_l358_358800


namespace solve_equation_l358_358797

theorem solve_equation :
  ∀ y : ℤ, 4 * (y - 1) = 1 - 3 * (y - 3) → y = 2 :=
by
  intros y h
  sorry

end solve_equation_l358_358797


namespace average_of_remaining_two_numbers_l358_358333

theorem average_of_remaining_two_numbers 
  (avg_6 : ℝ) (avg1_2 : ℝ) (avg2_2 : ℝ)
  (n1 n2 n3 : ℕ)
  (h_avg6 : n1 = 6 ∧ avg_6 = 4.60)
  (h_avg1_2 : n2 = 2 ∧ avg1_2 = 3.4)
  (h_avg2_2 : n3 = 2 ∧ avg2_2 = 3.8) :
  ∃ avg_rem2 : ℝ, avg_rem2 = 6.6 :=
by {
  sorry
}

end average_of_remaining_two_numbers_l358_358333


namespace ellipse_solution_l358_358941

noncomputable def ellipse_equation_and_max_intercept (a b c : ℝ) (P m n : ℝ) := 
  let ε := real.sqrt 2 / 2 in
  let C := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 in
  let focus_cond := ∀ F1 F2 : ℝ, F1 = c ∧ F2 = -c in
  let point_cond := |(m, n)| = real.sqrt 7 / 2 ∧ (c - m)^2 + n^2 = 3 / 4 in
  let max_intercept := ∀ k : ℝ, 
    let S := (0, -1/3) in
    let intercept := if k = 0 then 0 else (k / (3 * (2 * k^2 + 1))) in
    max_intercept = (real.sqrt 2 / 12) in
  C ∧ focus_cond ∧ point_cond ∧ max_intercept

theorem ellipse_solution : 
  ∃ a b c : ℝ, ∃ P m n : ℝ, ellipse_equation_and_max_intercept a b c P m n :=
by
  exists 1
  exists 1
  exists 1
  exists 2
  exists 2
  exists 2
  sorry

end ellipse_solution_l358_358941


namespace dot_product_a_b_l358_358541

open Real

noncomputable def cos_deg (x : ℝ) := cos (x * π / 180)
noncomputable def sin_deg (x : ℝ) := sin (x * π / 180)

theorem dot_product_a_b :
  let a_magnitude := 2 * cos_deg 15
  let b_magnitude := 4 * sin_deg 15
  let angle_ab := 30
  a_magnitude * b_magnitude * cos_deg angle_ab = sqrt 3 :=
by
  -- proof omitted
  sorry

end dot_product_a_b_l358_358541


namespace x_days_to_finish_work_l358_358415

variable (W : ℝ)
variable (W_x : ℝ)
variable (W_y : ℝ)
variable (D_x : ℝ)
variable (D_y : ℝ := 15)
variable (t_y : ℝ := 10)
variable (t_x : ℝ := 10.000000000000002)

theorem x_days_to_finish_work :
  W / D_x = W_x → 
  W_y = W / D_y → 
  (t_y * (W / D_y)) + (t_x * (W / D_x)) = W →
  D_x = 30 := 
by
  intro h1 h2 h3
  have h4 : W_x = W / D_x, from h1
  have h5 : W_y = W / D_y, from h2
  have h6 : t_y * W_y = (2/3) * W, from by norm_num; exact h2 ▸ rfl
  have h7 : W - (t_y * W_y) = (1/3) * W, from by norm_num; exact h6 ▸ rfl
  have h8 : W_x = (1/3) * W / t_x, from by norm_num; exact h7 ▸ rfl
  have h9 : (1/3) * W / t_x = W / D_x, from by exact h4; exact h8 ▸ rfl
  have h10 : (1/3) * W / 10.000000000000002 = W / D_x, from by norm_num; exact h8 ▸ rfl
  have h11 : D_x = 10.000000000000002 * 3, from calc
    D_x = W / ((1/3) * W / t_x) : by exact eq.symm h9
        ... = W / ((1/3) * W / 10.000000000000002) : by exact eq.symm h10
        ... = 10.000000000000002 * 3 : by norm_num
  exact h11

end x_days_to_finish_work_l358_358415


namespace arithmetic_sequence_problem_l358_358265

variable {α : Type*} [LinearOrder α] [AddSemigroup α] [MulSemigroup α]

noncomputable def arithmetic_sequence (n : ℕ) (a d : α) : α :=
  a + n * d

theorem arithmetic_sequence_problem
  (a d : ℕ) (h : a + 3 * (a + 7 * d) + (a + 14 * d) = 60) :
  2 * (a + 8 * d) - (a + 9 * d) = 12 :=
by
  sorry

end arithmetic_sequence_problem_l358_358265


namespace cubic_roots_natural_numbers_l358_358887

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l358_358887


namespace years_older_l358_358044

variable (A B C : ℕ)

-- given conditions
def condition1 : Prop := B = 18
def condition2 : Prop := B = 2 * C
def condition3 : Prop := A + B + C = 47

-- theorem to prove
theorem years_older (h1 : condition1) (h2 : condition2) (h3 : condition3) : A - B = 2 := by
  sorry

end years_older_l358_358044


namespace tetrahedron_volume_le_one_l358_358092

open Real

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := A
  let (x1, y1, z1) := B
  let (x2, y2, z2) := C
  let (x3, y3, z3) := D
  abs ((x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) -
       (x2 - x0) * ((y1 - y0) * (z3 - z0) - (y3 - y0) * (z1 - z0)) +
       (x3 - x0) * ((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0))) / 6

theorem tetrahedron_volume_le_one (A B C D : ℝ × ℝ × ℝ)
  (h1 : dist A B ≤ 2) (h2 : dist A C ≤ 2) (h3 : dist A D ≤ 2)
  (h4 : dist B C ≤ 2) (h5 : dist B D ≤ 2) (h6 : dist C D ≤ 2) :
  volume_tetrahedron A B C D ≤ 1 := by
  sorry

end tetrahedron_volume_le_one_l358_358092


namespace range_of_m_l358_358164

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ : 0 ≤ θ) (hθ_pi2 : θ ≤ Real.pi / 2)
  (hf : f(m * Real.sin θ) + f(1 - m) > 0) :
  m < 1 :=
sorry

end range_of_m_l358_358164


namespace curved_surface_area_of_cone_l358_358359

-- Define the conditions
def slant_height := 14 -- Slant height in cm
def radius := 12 -- Radius in cm
def pi_value := 3.14159 -- Approximation of π

-- Define the formula for curved surface area (CSA)
def curved_surface_area (r : ℝ) (l : ℝ) : ℝ :=
  pi_value * r * l

-- Problem statement in Lean 4
theorem curved_surface_area_of_cone :
  curved_surface_area radius slant_height ≈ 528.672 :=
by
  sorry

end curved_surface_area_of_cone_l358_358359


namespace six_digit_palindromes_l358_358822

-- Definition of palindrome condition
def isPalindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

-- The problem statement in Lean
theorem six_digit_palindromes : 
  {n : ℕ // 100000 ≤ n ∧ n < 1000000 ∧ isPalindrome n}.toFinset.card = 900 := by
  sorry  -- Omit the proof.

end six_digit_palindromes_l358_358822


namespace tap_B_filling_time_l358_358700

theorem tap_B_filling_time : 
  ∀ (r_A r_B : ℝ), 
  (r_A + r_B = 1 / 30) → 
  (r_B * 40 = 2 / 3) → 
  (1 / r_B = 60) := 
by
  intros r_A r_B h₁ h₂
  sorry

end tap_B_filling_time_l358_358700


namespace greatest_three_digit_not_divisor_l358_358735

theorem greatest_three_digit_not_divisor :
  ∃ (n : ℕ), n = 996 ∧ ∀ m, m > 996 → m ≤ 999 → (let S_m := m * (m + 1) / 2;
                                                  let P_m := Nat.factorial m in
                                                  ¬ S_m ∣ P_m) :=
by
  sorry

end greatest_three_digit_not_divisor_l358_358735


namespace area_of_isosceles_triangle_l358_358711

theorem area_of_isosceles_triangle (a b c : ℝ) (h1 : a = 16) (h2 : b = 16) (h3 : c = 30) :
  let s := (a + b + c) / 2 in
  ∃ h : ℝ, 2 * h = 30 → (a ^ 2 - (c / 2) ^ 2) = h ^ 2 → 
  area_triangle = 15 * Real.sqrt 31 :=
by
  sorry

end area_of_isosceles_triangle_l358_358711


namespace calculate_length_of_escalator_l358_358015

noncomputable def length_of_escalator (time_boy : ℝ) (time_girl : ℝ) (speed_boy : ℝ) (speed_girl : ℝ) (length : ℝ) :=
  ∃ (L : ℝ),
    L = length ∧
    (∀ (v : ℝ), (L = (speed_boy + v) * time_boy) ∧ (L = (speed_girl - v) * time_girl)) 

theorem calculate_length_of_escalator :
  length_of_escalator 100 300 3 2 150 :=
begin
  sorry
end

end calculate_length_of_escalator_l358_358015


namespace value_of_b_l358_358458

noncomputable def amplitude := 4

noncomputable def period := (2 * Real.pi) / 4

theorem value_of_b (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h_amplitude : a = amplitude) (h_period : (2 * Real.pi) / b = period) :
  (b = 4) ∧ (∀ c, (2 * Real.pi) / b = period) :=
by
  sorry

end value_of_b_l358_358458


namespace largest_perfect_square_of_three_one_digit_numbers_l358_358736

theorem largest_perfect_square_of_three_one_digit_numbers :
  ∃ a b c : ℕ, (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ∈ {1, 2, 3, 4, 6, 8, 9}) ∧ (b ∈ {1, 2, 3, 4, 6, 8, 9}) ∧ (c ∈ {1, 2, 3, 4, 6, 8, 9}) ∧ (a * b * c = 144) :=
sorry

end largest_perfect_square_of_three_one_digit_numbers_l358_358736


namespace cubic_roots_natural_numbers_l358_358888

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l358_358888


namespace temperature_after_fall_initial_velocity_to_melting_point_l358_358424

-- Definitions of the given conditions.
def specific_heat_lead : ℝ := 0.0315
def mechanical_equivalent_heat : ℝ := 425
def melting_point_lead : ℝ := 335
def initial_temperature : ℝ := 20
def gravitational_acceleration : ℝ := 9.80
def height : ℝ := 100

-- The temperature of the ball immediately after the fall
theorem temperature_after_fall : 
  let Δt := (gravitational_acceleration * height) / (mechanical_equivalent_heat * specific_heat_lead) 
  in initial_temperature + Δt = 93.2 := 
by sorry

-- The initial velocity required for the ball to reach the melting point
theorem initial_velocity_to_melting_point : 
  let ΔT := melting_point_lead - initial_temperature 
  let velocity_sq := 2 * (ΔT * mechanical_equivalent_heat * specific_heat_lead - gravitational_acceleration * height)
  in real.sqrt velocity_sq = 80.75 := 
by sorry

end temperature_after_fall_initial_velocity_to_melting_point_l358_358424


namespace range_of_function_l358_358550

theorem range_of_function (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y = if x < 1 then (1-2*a)*x + 3*a else log x) →
  -1 ≤ a ∧ a < 1/2 :=
by
  sorry

end range_of_function_l358_358550


namespace negation_of_p_l358_358946

theorem negation_of_p :
  (¬ (∀ x : ℝ, x ∈ (0, π) → sin x + (1 / sin x) > 2)) ↔
  (∃ x₀ : ℝ, x₀ ∈ (0, π) ∧ sin x₀ + (1 / sin x₀) ≤ 2) :=
by sorry

end negation_of_p_l358_358946


namespace domain1_domain2_domain3_l358_358493

-- Problem 1
theorem domain1 (x : ℝ) : (
  x + 1 ≥ 0 ∧ x - 2 ≠ 0
) ↔ (x ≥ -1 ∧ x ≠ 2) :=
sorry

-- Problem 2
theorem domain2 (x : ℝ) : (
  1 - (1 / 3)^x ≥ 0
) ↔ (x ≥ 0) :=
sorry

-- Problem 3
theorem domain3 (x : ℝ) : (
  log 2 (x - 1) > 0 ∧ x - 1 > 0
) ↔ (x > 2) :=
sorry

end domain1_domain2_domain3_l358_358493


namespace perp_lines_l358_358528

theorem perp_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ ((3 - a) * x - y + a = 0) → 
  (a = 1 ∨ a = 2)) :=
by
  intros x y h
  let k1 := -a / 2
  let k2 := 3 - a
  have h_perp : k1 * k2 = -1, from sorry,
  -- Solution step: Substitute the slopes connected to given lines
  have h_eq : (3 - a) * (-a / 2) = -1, from sorry,
  -- Show that this h_eq completes the demonstration that a is in {1, 2}
  sorry

end perp_lines_l358_358528


namespace point_P_in_third_quadrant_l358_358263

def point_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_third_quadrant :
  point_in_third_quadrant (-3) (-2) :=
by
  sorry -- Proof of the statement, as per the steps given.

end point_P_in_third_quadrant_l358_358263


namespace shipping_cost_per_unit_leq_one_point_six_seven_l358_358066

variable (S : ℝ)

-- Conditions as definitions in Lean 4
def production_cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℝ := 191.67

-- The theorem to prove the shipping cost per unit
theorem shipping_cost_per_unit_leq_one_point_six_seven :
  let production_cost := components_per_month * production_cost_per_component
  let shipping_cost := components_per_month * S
  let total_revenue := components_per_month * selling_price_per_component
  total_revenue ≥ production_cost + shipping_cost + fixed_monthly_cost → S ≤ 1.67 :=
by
  sorry

end shipping_cost_per_unit_leq_one_point_six_seven_l358_358066


namespace distance_between_points_l358_358861

-- Define the points
def point1 := (-3: ℝ, 1: ℝ, -2: ℝ)
def point2 := (4: ℝ, -3: ℝ, 2: ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- The theorem stating the distance between the given points
theorem distance_between_points : distance point1 point2 = 9 := 
by
  -- Proof would go here
  sorry

end distance_between_points_l358_358861


namespace shortest_distance_between_points_is_line_segment_l358_358399

theorem shortest_distance_between_points_is_line_segment (P Q : ℝ²) : 
  ∀ (P Q : ℝ²), (∃ l : set ℝ², line P Q l ∧ ∀ R ∈ l, dist P R + dist R Q = dist P Q) :=
sorry

end shortest_distance_between_points_is_line_segment_l358_358399


namespace circumscribed_sphere_around_prism_inscribed_sphere_in_prism_l358_358407

-- Definitions of cyclic_polygon, tangential_polygon, right_prism, and incircle_radius for illustration purposes. 
-- These would typically be defined in appropriate mathematical terms.
def cyclic_polygon (P : Type) : Prop := sorry
def tangential_polygon (P : Type) : Prop := sorry
def right_prism (B : Type) (H : ℝ) : Prop := sorry
def incircle_radius (P : Type) : ℝ := sorry

-- Problem 1: A sphere can be circumscribed around a right prism if its base is a cyclic polygon
theorem circumscribed_sphere_around_prism (B : Type) (H : ℝ) (P : set B) :
  cyclic_polygon P → right_prism P H → ∃ S : Type, (∀ v ∈ P, v ∈ S) :=
sorry
    
-- Problem 2: A sphere can be inscribed in a right prism if its base is a tangential polygon and the height of the prism equals the diameter of the incircle
theorem inscribed_sphere_in_prism (B : Type) (H : ℝ) (P : set B) :
  tangential_polygon P → right_prism P H → H = 2 * incircle_radius P → ∃ S : Type, (∀ f ∈ faces P, S ∩ f ≠ ∅) :=
sorry

end circumscribed_sphere_around_prism_inscribed_sphere_in_prism_l358_358407


namespace range_of_g_l358_358689

def g (x : ℕ) : ℕ := x^2 + x

theorem range_of_g : {g 1, g 2} = {2, 6} :=
by 
  -- proof will be filled in here
  sorry

end range_of_g_l358_358689


namespace good_pair_bound_all_good_pairs_l358_358416

namespace good_pairs

-- Definition of a "good" pair
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a : Fin r → ℤ) (b : Fin s → ℤ),
  (∀ i j : Fin r, i ≠ j → a i ≠ a j) ∧
  (∀ i j : Fin s, i ≠ j → b i ≠ b j) ∧
  (∀ i : Fin r, P (a i) = 2) ∧
  (∀ j : Fin s, P (b j) = 5)

-- (a) Show that for every good pair (r, s), r, s ≤ 3
theorem good_pair_bound (r s : ℕ) (h : is_good_pair r s) : r ≤ 3 ∧ s ≤ 3 :=
sorry

-- (b) Determine all good pairs
theorem all_good_pairs (r s : ℕ) : is_good_pair r s ↔ (r ≤ 3 ∧ s ≤ 3 ∧ (
  (r = 1 ∧ s = 1) ∨ (r = 1 ∧ s = 2) ∨ (r = 1 ∧ s = 3) ∨
  (r = 2 ∧ s = 1) ∨ (r = 2 ∧ s = 2) ∨ (r = 2 ∧ s = 3) ∨
  (r = 3 ∧ s = 1) ∨ (r = 3 ∧ s = 2))) :=
sorry

end good_pairs

end good_pair_bound_all_good_pairs_l358_358416


namespace domain_width_of_f_l358_358229

noncomputable def h : ℝ → ℝ := sorry -- Placeholder definition for h

def domain_of_h := Icc (-10 : ℝ) 10

def g (x : ℝ) := h (x / 2)

def f (x : ℝ) := g (3 * x)

theorem domain_width_of_f :
  let domain_f := {x : ℝ | (-20 : ℝ)/3 ≤ x ∧ x ≤ 20/3} in 
  (sup domain_f - inf domain_f = 40 / 3) :=
by
  sorry

end domain_width_of_f_l358_358229


namespace student_can_miss_and_pass_l358_358107

-- Define the test parameters
def total_questions : ℕ := 40
def passing_score_percentage : ℝ := 0.75

-- Definition of the greatest number of questions a student can miss and still pass
def max_missed_questions : ℕ := total_questions - (total_questions : ℝ) * passing_score_percentage

-- The theorem to be proved
theorem student_can_miss_and_pass :
  max_missed_questions = 10 :=
by
  sorry

end student_can_miss_and_pass_l358_358107


namespace digit_in_101st_place_of_decimal_of_3_div_7_l358_358986

theorem digit_in_101st_place_of_decimal_of_3_div_7 : 
  let repeating_block := [4, 2, 8, 5, 7, 1]
      n := 101
  in repeating_block[((n - 1) % repeating_block.length) + 1] = 7 :=
sorry

end digit_in_101st_place_of_decimal_of_3_div_7_l358_358986


namespace center_of_hyperbola_l358_358143

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l358_358143


namespace relationship_abcd_l358_358287

noncomputable def a := Real.log 0.32
noncomputable def b := Real.log 0.33
def c := 20.3
def d := 0.32

theorem relationship_abcd : b < a ∧ a < d ∧ d < c :=
  sorry

end relationship_abcd_l358_358287


namespace unique_complex_z_l358_358884

theorem unique_complex_z (x y : ℤ) (c : ℤ) (hx : 0 < x) (hy : 0 < y) :
  ((x + y * Complex.i)^3 = -107 + c * Complex.i) → (x = 1 ∧ y = 6) := by
  sorry

end unique_complex_z_l358_358884


namespace disproves_johns_assertion_l358_358504

def is_consonant (c : Char) : Prop :=
  c ∈ ['R', 'S']

def is_prime (n : ℕ) : Prop :=
  n ≠ 1 ∧ n ≠ 0 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

def card : ℕ → Char := sorry  -- Assume we have a function associating numbers and characters.

theorem disproves_johns_assertion (cards : List (ℕ × Char)) : 
  (∃ n c, (n = 8 ∧ is_consonant (card n) ∧ ¬ is_prime n)) :=
sorry

end disproves_johns_assertion_l358_358504


namespace exactly_two_toads_l358_358608

universe u

structure Amphibian where
  brian : Bool
  julia : Bool
  sean : Bool
  victor : Bool

def are_same_species (x y : Bool) : Bool := x = y

-- Definitions of statements by each amphibian
def Brian_statement (a : Amphibian) : Bool :=
  are_same_species a.brian a.sean

def Julia_statement (a : Amphibian) : Bool :=
  a.victor

def Sean_statement (a : Amphibian) : Bool :=
  ¬ a.julia

def Victor_statement (a : Amphibian) : Bool :=
  (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2

-- Conditions translated to Lean definition
def valid_statements (a : Amphibian) : Prop :=
  (a.brian → Brian_statement a) ∧
  (¬ a.brian → ¬ Brian_statement a) ∧
  (a.julia → Julia_statement a) ∧
  (¬ a.julia → ¬ Julia_statement a) ∧
  (a.sean → Sean_statement a) ∧
  (¬ a.sean → ¬ Sean_statement a) ∧
  (a.victor → Victor_statement a) ∧
  (¬ a.victor → ¬ Victor_statement a)

theorem exactly_two_toads (a : Amphibian) (h : valid_statements a) : 
( (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2 ) :=
sorry

end exactly_two_toads_l358_358608


namespace distances_inequality_l358_358313

variable (T : Triangle) (P : Point T) (x y z r : ℝ)

def is_acute (T : Triangle) : Prop :=
  T.angleA < π / 2 ∧ T.angleB < π / 2 ∧ T.angleC < π / 2

def circumradius (T : Triangle) (r : ℝ) : Prop :=
  r = T.circumradius

def distance_from_point (P : Point T) (x y z : ℝ) : Prop :=
  x = P.distance_to_side T.sideBC ∧
  y = P.distance_to_side T.sideCA ∧
  z = P.distance_to_side T.sideAB

theorem distances_inequality :
  ∀ (T : Triangle) (P : Point T) (x y z r : ℝ),
  is_acute T →
  circumradius T r →
  distance_from_point P x y z →
  sqrt x + sqrt y + sqrt z ≤ 3 * sqrt (r / 2) :=
sorry

end distances_inequality_l358_358313


namespace graph_paper_fold_proof_l358_358438

theorem graph_paper_fold_proof (m n : ℝ) :
  let midpoint_p := (2, 1)
  let slope_line := - (1 / 2)
  let slope_perpendicular := 2
  let fold_line := (λ (x: ℝ), 2 * x - 3)
  let midpoint_q := ((7 + m) / 2, (3 + n) / 2)
  (n - 3) / (m - 7) = - (1 / 2) ∧ midpoint_q.2 = fold_line midpoint_q.1 → 
  m + n = 6.8 :=
by
  sorry

end graph_paper_fold_proof_l358_358438


namespace unique_gift_box_combinations_l358_358007

def giftPrices : List ℕ := [2, 5, 8, 11, 14]
def boxPrices : List ℕ := [3, 6, 9, 12, 15]

theorem unique_gift_box_combinations : 
  (Set.univ : Set ℕ).toFinset.filter (λ n, ∃ gp ∈ giftPrices, ∃ bp ∈ boxPrices, gp + bp = n).card = 9 := 
by sorry

end unique_gift_box_combinations_l358_358007


namespace johns_total_spent_l358_358280

def silver_ounces := 2.5
def silver_cost_per_ounce := 25 -- USD

def gold_ounces := 3.5
def gold_cost_multiplier := 60

def platinum_ounces := 4.5
def platinum_cost_per_ounce_gbp := 80 -- GBP
def usd_to_gbp_conversion := 1.3

def total_spent_usd : Real :=
  (silver_ounces * silver_cost_per_ounce) +
  (gold_ounces * (silver_cost_per_ounce * gold_cost_multiplier)) +
  (platinum_ounces * (platinum_cost_per_ounce_gbp * usd_to_gbp_conversion))

theorem johns_total_spent :
  total_spent_usd = 5780.5 :=
by
  sorry

end johns_total_spent_l358_358280


namespace question_1_question_2_l358_358537

open Real

theorem question_1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab < m / 2 → m > 2 := sorry

theorem question_2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : 9 / a + 1 / b ≥ |x - 1| + |x + 2|) :
  -9/2 ≤ x ∧ x ≤ 7/2 := sorry

end question_1_question_2_l358_358537


namespace uncle_wang_withdrawal_l358_358017

variable (principal : ℝ) (rate : ℝ) (time : ℝ)

-- Uncle Wang's conditions
def condition_principal := principal = 20000
def condition_rate := rate = 0.0325
def condition_time := time = 2

-- Calculation function
def total_amount (principal rate time : ℝ) := principal + (principal * rate * time)

-- Theorem to prove
theorem uncle_wang_withdrawal :
  principal = 20000 → rate = 0.0325 → time = 2 → total_amount principal rate time = 21300 :=
by
  intros
  rw [condition_principal, condition_rate, condition_time]
  sorry

end uncle_wang_withdrawal_l358_358017


namespace average_price_of_remaining_cans_l358_358611

theorem average_price_of_remaining_cans (price_all price_returned : ℕ) (average_all average_returned : ℚ) 
    (h1 : price_all = 6) (h2 : average_all = 36.5) (h3 : price_returned = 2) (h4 : average_returned = 49.5) : 
    (price_all - price_returned) ≠ 0 → 
    4 * 30 = 6 * 36.5 - 2 * 49.5 :=
by
  intros hne
  sorry

end average_price_of_remaining_cans_l358_358611


namespace parallelogram_area_calculation_l358_358618

noncomputable def parallelogram_area (p q : ℝ^3) : ℝ :=
  ∥(5 • q - 2 • p) × (2 • p + 5 • q)∥ / 4

theorem parallelogram_area_calculation (p q : ℝ^3)
  (hp : ∥p∥ = 1) (hq : ∥q∥ = 1) (angle_pq : inner p q = real.cos (π / 4)) :
  parallelogram_area p q = 25 * real.sqrt 2 / 4 := by
sorry

end parallelogram_area_calculation_l358_358618


namespace find_curve_equation_find_constant_a_l358_358522

variable {a t θ : ℝ}

def line_parametric (a t θ : ℝ) : ℝ × ℝ :=
  (a + t * cos θ, t * sin θ)

-- Polar equation of the curve
def curve_polar (ρ θ : ℝ) := ρ - ρ * cos θ ^ 2 - 4 * cos θ = 0

theorem find_curve_equation (θ : ℝ) (ρ : ℝ) :
    curve_polar ρ θ →
    ∃ (x y : ℝ), y^2 = 4*x ∧ x = ρ * cos θ ∧ y = ρ * sin θ :=
    by 
      sorry

theorem find_constant_a (a : ℝ) (θ : ℝ)
    (intersects_curve : ∃ t1 t2 : ℝ, (a + t1 * cos θ)^2 - t1^2 * sin^2 θ - 4 * (a + t2 * cos θ) = 0 ∧
                                       (a + t2 * cos θ)^2 - t2^2 * sin^2 θ - 4 * (a + t2 * cos θ) = 0) :
    a = 2 → ( 1 / (a + t1 * cos θ)^2 + 1 / (a + t2 * cos θ)^2 = 1 / 4 )
    :=
    by
      sorry

end find_curve_equation_find_constant_a_l358_358522


namespace avg_score_for_C_l358_358833

def avg_score_class_C (avg_U avg_B ratio_U ratio_B ratio_C combined_avg : ℕ) : ℕ :=
  let u := ratio_U * 1 in
  let b := ratio_B * 1 in
  let c := ratio_C * 1 in
  let total_students := u + b + c in
  let total_score_U := avg_U * u in
  let total_score_B := avg_B * b in
  let combined_total_score := combined_avg * total_students in
  let total_score_C := combined_total_score - total_score_U - total_score_B in
  total_score_C / c

theorem avg_score_for_C :
  avg_score_class_C 65 80 4 6 5 75 = 77 :=
by
  sorry

end avg_score_for_C_l358_358833


namespace perimeter_of_trapezoid_WXYZ_l358_358271

-- Define a trapezoid and its properties
structure Trapezoid :=
(WX YZ WZ XY : ℝ)
(height : ℝ)
(WZ_XY_parallel : WX = YZ)
(WX_parallel_XY : true)

-- Given conditions
noncomputable def trapezoid_WXYZ : Trapezoid :=
{ WX := real.sqrt (5 ^ 2 + 4 ^ 2),
  YZ := real.sqrt (5 ^ 2 + 4 ^ 2),
  WZ := 10,
  XY := 18,
  height := 5,
  WZ_XY_parallel := by rfl,
  WX_parallel_XY := by trivial }

-- Assert the perimeter calculation
theorem perimeter_of_trapezoid_WXYZ :
  trapezoid_WXYZ.WZ + trapezoid_WXYZ.XY + 2 * trapezoid_WXYZ.WX = 28 + 2 * real.sqrt 41 :=
by sorry

end perimeter_of_trapezoid_WXYZ_l358_358271


namespace bisection_method_next_interval_l358_358019

noncomputable def f : ℝ → ℝ := λ x, 2 * real.log x / real.log 5 - 1

theorem bisection_method_next_interval :
  (f 2 < 0) → (f 3 > 0) → f (2 + (3-2)/2) > 0 → ∃ a b, a = 2 ∧ b = 2.5 ∧ (∀ x, a < x ∧ x < b → f x = 2 * real.log x / real.log 5 - 1) :=
by
  intros h1 h2 h3
  use [2, 2.5]
  split; try {refl}
  intros x hx
  rw f
  sorry

end bisection_method_next_interval_l358_358019


namespace max_halls_l358_358062

theorem max_halls (n : ℕ) (hall : ℕ → ℕ) (H : ∀ n, hall n = hall (3 * n + 1) ∧ hall n = hall (n + 10)) :
  ∃ (m : ℕ), m = 3 :=
by
  sorry

end max_halls_l358_358062


namespace equal_alternating_sums_l358_358525

-- Given: 
def inscribed_polygon (angles : List ℝ) (n : ℕ) : Prop :=
  angles.length = 2 * n

-- We have a 2n-gon with angles β_1, β_2, ..., β_2n
def given_2n_gon (angles : List ℝ) (β : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i < 2 * n → angles.get i = β i

-- The condition that perpendicular bisectors intersect at the center
def bisectors_intersect_at_center (n : ℕ) : Prop := 
  -- replace with actual condition logic if necessary
  true

-- We need to prove:
theorem equal_alternating_sums (angles : List ℝ) (β : ℕ → ℝ) (n : ℕ) 
  (h1 : inscribed_polygon angles n)
  (h2 : given_2n_gon angles β n)
  (h3 : bisectors_intersect_at_center n) :
  ∑ i in List.range n, β (2 * i + 1) = ∑ i in List.range n, β (2 * i) :=
by sorry

end equal_alternating_sums_l358_358525


namespace solve_for_x_l358_358659

theorem solve_for_x : ∃ x : ℚ, 0.05 * x + 0.09 * (30 + x) = 12 ∧ x = 465 / 7 :=
by
  use 465 / 7
  split
  sorry

end solve_for_x_l358_358659


namespace find_a6_l358_358180

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (h1 : ∀ n ≥ 2, S n = 2 * a n)
variable (h2 : S 5 = 8)

theorem find_a6 : a 6 = 8 :=
by
  sorry

end find_a6_l358_358180


namespace PM_dot_PN_l358_358185

variables {A B C P M N : Point}
variable {area : ℝ}
variable {angle_BAC : ℝ}

-- Conditions: 
-- area of triangle ABC is 4
-- angle BAC is 120 degrees
-- point P such that vector BP = 3 * vector PC

axiom area_ABC : area = 4
axiom angle_BAC_eq : angle_BAC = 120 * (π / 180)    -- converting degrees to radians
axiom point_P_relation : vector B P = 3 • (vector P C)

-- Question: Find dot product of vectors PM and PN
theorem PM_dot_PN :
  (vector P M) ⬝ (vector P N) = 3 * sqrt 3 / 8 :=
sorry

end PM_dot_PN_l358_358185


namespace translation_symmetry_example_l358_358343

def g (x : ℝ) : ℝ := exp (-x)

def translated_function (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

theorem translation_symmetry_example : 
  (translated_function g 1) = (fun x => exp (-x - 1)) :=
by
  sorry

end translation_symmetry_example_l358_358343


namespace max_y_coordinate_l358_358901

theorem max_y_coordinate (θ : ℝ) : 
  let r := 2 * sin (2 * θ)
  let y := r * sin θ
  y ≤ 8 * real.sqrt 3 / 9 := 
sorry

end max_y_coordinate_l358_358901


namespace count_valid_rearrangements_l358_358222

-- Define the alphabet adjacency condition
def is_adjacent (c1 c2 : Char) : Prop :=
  (c1 = 'e' ∧ c2 = 'f') ∨ (c1 = 'f' ∧ c2 = 'e') ∨
  (c1 = 'f' ∧ c2 = 'g') ∨ (c1 = 'g' ∧ c2 = 'f') ∨
  (c1 = 'g' ∧ c2 = 'h') ∨ (c1 = 'h' ∧ c2 = 'g')

-- Define a condition on a list of characters
def valid_rearrangement (s : List Char) : Prop :=
  ∀ i, i < s.length - 1 → ¬ is_adjacent (s.get i) (s.get (i + 1))

-- List all possible characters to use
def char_list : List Char := ['e', 'f', 'g', 'h']

-- Main theorem to prove the count of valid rearrangements
theorem count_valid_rearrangements : 
  (List.permutations char_list).filter valid_rearrangement).length = 6 :=
by sorry

end count_valid_rearrangements_l358_358222


namespace ordered_pairs_count_l358_358978

theorem ordered_pairs_count : 
  (∃ (M N : ℕ), M > 0 ∧ N > 0 ∧ (M / 8 = 8 / N)) ↔ 7 := 
  sorry

end ordered_pairs_count_l358_358978


namespace smallest_points_to_exceed_mean_l358_358443

theorem smallest_points_to_exceed_mean (X y : ℕ) (h_scores : 24 + 17 + 25 = 66) 
  (h_mean_9_gt_mean_6 : X / 6 < (X + 66) / 9) (h_mean_10_gt_22 : (X + 66 + y) / 10 > 22) 
  : y ≥ 24 := by
  sorry

end smallest_points_to_exceed_mean_l358_358443


namespace gcd_36_54_l358_358724

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l358_358724


namespace find_a_l358_358288

theorem find_a (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) 
(h3 : a^2 - b^2 - c^2 + 2 * a * b = 2315) 
(h4 : a^2 + 2 * b^2 + 2 * c^2 - 2 * a * b - a * c - b * c = -2213) : a = 255 :=
by
  sorry

end find_a_l358_358288


namespace haley_marbles_l358_358245

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (h_boys : boys = 13) (h_marbles_per_boy : marbles_per_boy = 2) :
  boys * marbles_per_boy = 26 := 
by 
  sorry

end haley_marbles_l358_358245


namespace range_of_lg_x_l358_358183

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_lg_x {f : ℝ → ℝ} (h_even : is_even f)
    (h_decreasing : is_decreasing_on_nonneg f)
    (h_condition : f (Real.log x) > f 1) :
    x ∈ Set.Ioo (1/10 : ℝ) (10 : ℝ) :=
  sorry

end range_of_lg_x_l358_358183


namespace number_of_zeros_in_sequence_l358_358336

theorem number_of_zeros_in_sequence :
  ∃ (count_zero : ℕ), 
    let seq : Fin 2015 → ℤ := λ n, if n.1 < 1000 then 1 else if n.1 < 1015 then 0 else -1 in
    (∑ n, seq n) = 427 ∧ 
    (∑ n, (seq n + 1) ^ 2) = 3869 ∧ 
    count_zero = (Finset.univ.filter (λ n, seq n = 0)).card :=
  sorry

end number_of_zeros_in_sequence_l358_358336


namespace positive_integers_appear_on_board_l358_358372

theorem positive_integers_appear_on_board :
  ∀ n ∈ ℕ, ∃ a b ∈ ℕ, (a = 1 ∧ b + 1 ∣ a ^ 2 + b ^ 2 + 1 → n ∈ ℕ) :=
begin
  sorry
end

end positive_integers_appear_on_board_l358_358372


namespace group_most_vs_least_dumplings_difference_class_total_dumplings_l358_358331

theorem group_most_vs_least_dumplings_difference :
  let diff := [(-8), 7, 4, -3, 1, 5, -2]
  in (List.maximum diff).getD 0 - (List.minimum diff).getD 0 = 15 :=
by
  let diff := [(-8), 7, 4, -3, 1, 5, -2]
  have h_max_min_diff : (List.maximum diff).getD 0 - (List.minimum diff).getD 0 = 15
    := by sorry
  exact h_max_min_diff

theorem class_total_dumplings :
  let diff := [(-8), 7, 4, -3, 1, 5, -2]
  let total_difference := diff.sum
  let num_groups := 7
  let benchmark := 100
  in total_difference + num_groups * benchmark = 704 :=
by
  let diff := [(-8), 7, 4, -3, 1, 5, -2]
  let total_difference := diff.sum
  let num_groups := 7
  let benchmark := 100
  have h_total_dumplings : total_difference + num_groups * benchmark = 704
    := by sorry
  exact h_total_dumplings

end group_most_vs_least_dumplings_difference_class_total_dumplings_l358_358331


namespace megan_math_problems_l358_358928

theorem megan_math_problems (num_spelling_problems num_problems_per_hour num_hours total_problems num_math_problems : ℕ) 
  (h1 : num_spelling_problems = 28)
  (h2 : num_problems_per_hour = 8)
  (h3 : num_hours = 8)
  (h4 : total_problems = num_problems_per_hour * num_hours)
  (h5 : total_problems = num_spelling_problems + num_math_problems) :
  num_math_problems = 36 := 
by
  sorry

end megan_math_problems_l358_358928


namespace max_distinct_natural_numbers_l358_358006

open Nat

/-- There are n pairwise distinct natural numbers, and the sum of the pairwise products of these
    numbers is 239. The maximum value of n is 7. -/
theorem max_distinct_natural_numbers (n : ℕ) (a : Fin n → ℕ)
  (h0 : ∀ i j, i < j → a i ≠ a j)
  (h1 : (Finset.univ.filter (λ i, i.1 < n)).sum (λ i, (Finset.univ.filter (λ j, j.1 < n ∧ i.1 < j.1)).sum (λ j, a i * a j)) = 239) :
  n ≤ 7 :=
  sorry

end max_distinct_natural_numbers_l358_358006


namespace largest_lattice_solution_l358_358497

theorem largest_lattice_solution :
  ∃ x y₁ y₂ y₃ : ℤ, (x + 1)^2 + y₁^2 = (x + 2)^2 + y₂^2 ∧ (x + 2)^2 + y₂^2 = (x + 3)^2 + y₃^2 ∧
  (∀ n > 3, ∃ x (y : fin n → ℤ), ¬(∀ i j, i < j → (x + i + 1)^2 + y i ^ 2 = (x + j + 1)^2 + y j ^ 2)) :=
  sorry

end largest_lattice_solution_l358_358497


namespace average_price_of_5_pillows_l358_358412

-- Define the conditions as constants
constant num_pillows_1 : ℕ := 4
constant avg_cost_1 : ℝ := 5
constant cost_pillow_5 : ℝ := 10

-- Define the total cost of the first 4 pillows and the fifth pillow
def total_cost_4_pillows : ℝ := num_pillows_1 * avg_cost_1
def total_cost_5_pillows : ℝ := total_cost_4_pillows + cost_pillow_5

-- Average cost calculation
def avg_cost_5_pillows : ℝ := total_cost_5_pillows / (num_pillows_1 + 1)

-- The theorem to be proved
theorem average_price_of_5_pillows : avg_cost_5_pillows = 6 := by
  sorry

end average_price_of_5_pillows_l358_358412


namespace count_difference_525_pages_l358_358130

-- Define the function to count the occurrences of a digit in a given range of page numbers
def count_digit_occurrences (digit lower upper : ℕ) : ℕ :=
  (list.range (upper - lower + 1)).map (λ n, (lower + n).digits.count (λ d, d = digit)).sum

-- Define the main theorem to prove the difference between counts of 2's and 5's
theorem count_difference_525_pages : 
  count_digit_occurrences 2 1 525 - count_digit_occurrences 5 1 525 = 72 :=
by
  -- The actual proof code should go here, but we'll use sorry to indicate it's omitted.
  sorry

end count_difference_525_pages_l358_358130


namespace range_of_a_l358_358205

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (f a x) * (f a (1 - x)) ≥ 1) ↔ (1 ≤ a) ∨ (a ≤ - (1/4)) := 
by
  sorry

end range_of_a_l358_358205


namespace domain_of_f_l358_358473

noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : 
  {x : ℝ | Real.sqrt (x^2 - 5 * x + 6) ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l358_358473


namespace factorial_base16_trailing_zeros_eq_2_l358_358684

theorem factorial_base16_trailing_zeros_eq_2 :
  let k := (15.factorial).trailing_zeros_base 16 in k = 2 :=
by
  sorry

end factorial_base16_trailing_zeros_eq_2_l358_358684


namespace gcd_36_54_l358_358722

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l358_358722


namespace biologist_fish_population_calc_l358_358426

def fish_population_on_May_1 (tagged_on_may_1: ℕ) (caught_on_sep_1: ℕ) 
(tagged_in_sep_sample: ℕ) (percent_tagged_remain: ℚ) 
(percent_fish_same: ℚ) : ℕ :=
  let tagged_remaining := tagged_on_may_1 * percent_tagged_remain
  let fish_same_in_sample := caught_on_sep_1 * percent_fish_same
  let ratio_tagged := tagged_in_sep_sample / fish_same_in_sample
  let total_fish_on_may_1 := (1 / ratio_tagged) * tagged_remaining
  total_fish_on_may_1

theorem biologist_fish_population_calc : fish_population_on_May_1 60 70 3 (3/4 : ℚ) (6/10 : ℚ) = 630 :=
by
  sorry

end biologist_fish_population_calc_l358_358426


namespace gcd_of_36_and_54_l358_358727

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l358_358727


namespace ordered_pair_unique_l358_358875

theorem ordered_pair_unique (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x, y) = (1, 14) :=
by
  sorry

end ordered_pair_unique_l358_358875


namespace west_3m_if_east_2m_is_pos2m_l358_358587

theorem west_3m_if_east_2m_is_pos2m :
  (∀ (m : ℝ), 3m = -3m) ↔ (∀ (n : ℝ), east n = +n) := by
  sorry

end west_3m_if_east_2m_is_pos2m_l358_358587


namespace problem1_l358_358882

theorem problem1 : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by 
  sorry

end problem1_l358_358882


namespace length_PQ_leq_quarter_perimeter_l358_358057

open Real EuclideanGeometry

variables {A B C D M S P Q : Point}
variables {a b c : ℝ} -- sides A, B, and C
variables (triangle_ABC : Triangle A B C)
variables (footD : OrthocenterFoot A B C D)
variables (midpointM : Midpoint B C M)
variables (pointS_onDM : OnSegment D M S)
variables (projP_onAB : ProjectedPoint S A B P)
variables (projQ_onAC : ProjectedPoint S A C Q)

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem length_PQ_leq_quarter_perimeter :
  lengthPQ ≤ (perimeter a b c) / 4 :=
sorry

end length_PQ_leq_quarter_perimeter_l358_358057


namespace problem_statement_l358_358342

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l358_358342


namespace roots_of_quadratic_l358_358949

variable {γ δ : ℝ}

theorem roots_of_quadratic (hγ : γ^2 - 5*γ + 6 = 0) (hδ : δ^2 - 5*δ + 6 = 0) : 
  8*γ^5 + 15*δ^4 = 8425 := 
by
  sorry

end roots_of_quadratic_l358_358949


namespace Roberto_outfits_count_l358_358317

theorem Roberto_outfits_count :
  ∃ (trousers shirts jackets shoes : ℕ), 
  trousers = 4 ∧
  shirts = 8 ∧
  jackets = 3 ∧
  shoes = 5 ∧
  (∀ (j : ℕ), j < jackets → ∃ (t1 t2 : ℕ), t1 < trousers ∧ t2 < trousers) ∧
  let combinations_shirts_shoes := shirts * shoes,
      combinations_jackets_trousers := jackets * 2,
      total_outfits := combinations_shirts_shoes * combinations_jackets_trousers
  in total_outfits = 240 :=
begin
  sorry
end

end Roberto_outfits_count_l358_358317


namespace puppies_left_l358_358099

namespace AlyssaPuppies

def initPuppies : ℕ := 12
def givenAway : ℕ := 7
def remainingPuppies : ℕ := 5

theorem puppies_left (initPuppies givenAway remainingPuppies : ℕ) : 
  initPuppies - givenAway = remainingPuppies :=
by
  sorry

end AlyssaPuppies

end puppies_left_l358_358099


namespace volume_of_cylinder_l358_358912

-- Define the problem conditions
def rectangle_length : ℝ := 20
def rectangle_width : ℝ := 10
def cylinder_height : ℝ := rectangle_width
def cylinder_radius : ℝ := rectangle_length / 2

-- State the theorem
theorem volume_of_cylinder :
  ∃ (V : ℝ), V = π * cylinder_radius^2 * cylinder_height ∧ V = 1000 * π := by
  sorry

end volume_of_cylinder_l358_358912


namespace problem1_problem2_problem3_l358_358056

-- Definitions of functions
noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

-- Problem 1: Prove the intersection of ranges when A = [1,2]
theorem problem1 (A : set ℝ) (hA : A = set.Icc 1 2) :
    let S := set.image f A
    let T := set.image g A
    S ∩ T = {5} :=
by sorry

-- Problem 2: Prove the value of m when A = [0, m] and S = T
theorem problem2 (A : set ℝ) (m : ℝ) (hA : A = set.Icc 0 m) 
    (hST : set.image f A = set.image g A) : 
    m = 4 :=
by sorry

-- Problem 3: Prove the set A when f(x) = g(x) for all x in A
theorem problem3 (A : set ℝ) 
    (hfg : ∀ x ∈ A, f x = g x) : 
    A = {0} ∨ A = {4} ∨ A = {0, 4} :=
by sorry

end problem1_problem2_problem3_l358_358056


namespace quadratic_has_real_roots_find_m_l358_358964

theorem quadratic_has_real_roots (m : ℝ) :
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5) in
  discriminant ≥ 0 ↔ m ≥ 1 / 2 :=
by
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5)
  split
  { intro h
    sorry -- This proof would show that if the discriminant is non-negative, then m ≥ 1/2
  }
  { intro h
    sorry -- This proof would show that if m ≥ 1/2, then the discriminant is non-negative
  }

theorem find_m (m : ℝ) (x1 x2 : ℝ) :
  (x1 + x2 = 4) →
  (x1 * x2 = -2 * m + 5) →
  (x1 * x2 + x1 + x2 = m ^ 2 + 6) →
  m ≥ 1 / 2 →
  m = 1 :=
by
  intros h1 h2 h3 h4
  sorry -- This proof would show that given the conditions, m must be 1

end quadratic_has_real_roots_find_m_l358_358964


namespace power_calculation_l358_358113

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := 
by 
  -- Simplifying (8^3 / 8^2)
  have h1 : 8^3 / 8^2 = 8 := by 
    calc
      8^3 / 8^2 = 8^(3 - 2) : by rw [nat.pow_sub (show 2 ≤ 3, by norm_num)]
      ... = 8^1 : by norm_num
      ... = 8 : by norm_num
  -- Rewriting 8 as 2^3
  have h2 : 8 = 2^3 := by norm_num
  -- We now have (8^3 / 8^2) * 2^10 = 8 * 2^10
  rw [h1, h2], simp
  -- Simplifying 2^3 * 2^10 = 2^(3+10)
  calc 8 * 2^10 = 2^3 * 2^10 : by {rw h2}
  ... = 2^(3 + 10) : by rw [pow_add]
  ... = 2^13 : by norm_num
  ... = 8192 : by norm_num

end power_calculation_l358_358113


namespace total_amount_l358_358994

theorem total_amount (a b c total first : ℕ)
  (h1 : a = 1 / 2) (h2 : b = 2 / 3) (h3 : c = 3 / 4)
  (h4 : first = 204)
  (ratio_sum : a * 12 + b * 12 + c * 12 = 23)
  (first_ratio : a * 12 = 6) :
  total = 23 * (first / 6) → total = 782 :=
by 
  sorry

end total_amount_l358_358994


namespace budget_equality_year_l358_358243

theorem budget_equality_year :
  ∀ Q R V W : ℕ → ℝ,
  Q 0 = 540000 ∧ R 0 = 660000 ∧ V 0 = 780000 ∧ W 0 = 900000 ∧
  (∀ n, Q (n+1) = Q n + 40000 ∧ 
         R (n+1) = R n + 30000 ∧ 
         V (n+1) = V n - 10000 ∧ 
         W (n+1) = W n - 20000) →
  ∃ n : ℕ, 1990 + n = 1995 ∧ 
  Q n + R n = V n + W n := 
by 
  sorry

end budget_equality_year_l358_358243


namespace common_point_sufficient_condition_l358_358002

theorem common_point_sufficient_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) → k ≤ -2 * Real.sqrt 2 :=
by
  -- Proof will go here
  sorry

end common_point_sufficient_condition_l358_358002


namespace california_more_license_plates_l358_358304

theorem california_more_license_plates :
  let CA_format := 26^4 * 10^2
  let NY_format := 26^3 * 10^3
  CA_format - NY_format = 28121600 := by
  let CA_format : Nat := 26^4 * 10^2
  let NY_format : Nat := 26^3 * 10^3
  have CA_plates : CA_format = 45697600 := by sorry
  have NY_plates : NY_format = 17576000 := by sorry
  calc
    CA_format - NY_format = 45697600 - 17576000 := by rw [CA_plates, NY_plates]
                    _ = 28121600 := by norm_num

end california_more_license_plates_l358_358304


namespace marble_prob_red_or_white_l358_358807

def marble_bag_prob (total_marbles : ℕ) (blue_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) : ℚ :=
  (red_marbles + white_marbles : ℚ) / total_marbles

theorem marble_prob_red_or_white :
  let total_marbles := 20
  let blue_marbles := 5
  let red_marbles := 7
  let white_marbles := total_marbles - (blue_marbles + red_marbles)
  marble_bag_prob total_marbles blue_marbles red_marbles white_marbles = 3 / 4 :=
by
  sorry

end marble_prob_red_or_white_l358_358807


namespace maximize_profit_l358_358334

def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0  -- This case is redundant by the problem constraints but needed for completeness.

noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2000
  else if x ≥ 40 then -x - 10000 / x + 2500
  else 0  -- This case is redundant by the problem constraints but needed for completeness.

theorem maximize_profit :
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, 0 < y → (L y ≤ L 100)) ∧ L 100 = 2300 :=
begin
  sorry
end

end maximize_profit_l358_358334


namespace quadratic_solution_sum_l358_358868

theorem quadratic_solution_sum (m n p : ℤ) (h : gcd m (gcd n p) = 1)
    (sols : ∀ x : ℝ, 5 * x^2 - 11 * x + 4 = 0 ↔ (x = (m + real.sqrt n) / p) ∨ (x = (m - real.sqrt n) / p)) :
    m + n + p = 62 := 
sorry

end quadratic_solution_sum_l358_358868


namespace odd_func_value_at_neg_two_l358_358189

-- Define the function f and its properties
def f (x : ℝ) : ℝ := if x > 0 then 2^x else -2^(-x)

-- The theorem 
theorem odd_func_value_at_neg_two
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 < x → f x = 2^x) : 
  f (-2) = -4 := 
  sorry

end odd_func_value_at_neg_two_l358_358189


namespace find_AX_l358_358607

theorem find_AX (A B C X : Type) [HasLength A B 68] [AngleBisector A C X X C B] [HasLength A C 20] [HasLength B C 40] :
    AX = 68 / 3 := by
  sorry

end find_AX_l358_358607


namespace total_amount_earned_l358_358433

variables (price_first_cycle : ℝ) (loss_first_cycle_perc : ℝ)
          (gain_second_cycle_perc : ℝ)
          (gain_third_cycle_perc : ℝ)

def calculate_selling_price (cost : ℝ) (perc : ℝ) (is_loss : Bool) : ℝ :=
  if is_loss then cost - (cost * perc) else cost + (cost * perc)

theorem total_amount_earned (h : price_first_cycle = 1600)
    (h1 : loss_first_cycle_perc = 0.12)
    (h2 : gain_second_cycle_perc = 0.15)
    (h3 : gain_third_cycle_perc = 0.20) :
  let price_after_first_sale := calculate_selling_price price_first_cycle loss_first_cycle_perc true,
      price_after_second_sale := calculate_selling_price price_after_first_sale gain_second_cycle_perc false,
      price_after_third_sale := calculate_selling_price price_after_second_sale gain_third_cycle_perc false
  in price_after_third_sale = 1943.04 :=
by
  sorry

end total_amount_earned_l358_358433


namespace triangle_area_l358_358839

theorem triangle_area (a b c : ℕ) (h : a * a + b * b = c * c) : 1 / 2 * a * b = 84 := by
  have ha : a = 7 := rfl
  have hb : b = 24 := rfl
  have hc : c = 25 := rfl
  rw [ha, hb, hc] at h
  norm_num at h
  norm_num
  sorry

end triangle_area_l358_358839


namespace no_b_satisfies_condition_for_f_l358_358159

theorem no_b_satisfies_condition_for_f (b : ℝ) :
  ∀ x : ℝ, x^2 + b * x - 1 = 1 :=
begin
  sorry
end

end no_b_satisfies_condition_for_f_l358_358159


namespace find_a_b_l358_358105

-- Definitions based on the problem conditions
variables (AC BC HE HD a b : Real)
hypothesis h1 : AC = 16.25
hypothesis h2 : BC = 13.75
hypothesis h3 : HE = 6
hypothesis h4 : HD = 3
hypothesis h5 : b - a = 5

-- The Lean 4 statement to prove a + b = 15 under given conditions
theorem find_a_b (AC BC HE HD a b : Real)
  (h1 : AC = 16.25) (h2 : BC = 13.75) (h3 : HE = 6) (h4 : HD = 3) (h5 : b - a = 5) :
  a + b = 15 :=
sorry

end find_a_b_l358_358105


namespace required_rate_for_team4_is_316_l358_358509

-- Define all the conditions provided in the problem
def total_pieces := 500
def deadline_hours := 3
def team1_pieces := 189
def team1_hours := 1
def team2_pieces := 131
def team2_hours := 1.5
def team3_rate := 45
def remaining_hours := deadline_hours - (team1_hours + team2_hours)
def team3_pieces := team3_rate * remaining_hours  -- This assumes an integer representation of pieces

-- Calculate the total pieces made by the first three teams
def total_made_by_first_three_teams := team1_pieces + team2_pieces + team3_pieces

-- The number of pieces the fourth team needs to complete
def pieces_needed_by_team4 := total_pieces - total_made_by_first_three_teams

-- The fourth team has the remaining time to complete their part
def team4_hours := remaining_hours

-- The required production rate for the fourth team
def required_production_rate_for_team4 := pieces_needed_by_team4 / team4_hours

-- Our goal is to prove that the required production rate for the fourth team is 316 pieces per hour
theorem required_rate_for_team4_is_316 : 
  required_production_rate_for_team4 = 316 := 
by
  sorry

end required_rate_for_team4_is_316_l358_358509


namespace count_terms_expansion_l358_358574

/-
This function verifies that the number of distinct terms in the expansion
of (a + b + c)(a + d + e + f + g) is equal to 15.
-/

theorem count_terms_expansion : 
    (a b c d e f g : ℕ) → 
    3 * 5 = 15 :=
by 
    intros a b c d e f g
    sorry

end count_terms_expansion_l358_358574


namespace circle_equation_fixed_point_exists_l358_358519

-- Definition of the problem's conditions
def A := (6: ℝ, 0: ℝ)
def B := (1: ℝ, 5: ℝ)
def l (x y : ℝ) := 2*x - 7*y + 8 = 0
def M := (1: ℝ, 2: ℝ)

-- Statements to be proven
theorem circle_equation (h : ∃ x y : ℝ, l x y ∧ (x - 3)^2 + (y - 2)^2 = 13) :
  ∀ C : ℝ × ℝ, C ∈ { p | (p.1 - 3)^2 + (p.2 - 2)^2 = 13 } ⇔ C = A ∨ C = B := sorry

theorem fixed_point_exists (h : ∃ x y : ℝ, l x y ∧ (x - 3)^2 + (y - 2)^2 = 13) :
  ∃ N : ℝ × ℝ, N = (-7/2, 2) ∧ ∀ A B : ℝ × ℝ, A ∈ {(6, 0)} ∧ B ∈ {(1, 5)} → 
  (A.2 - 2)/(A.1 + 7/2) + (B.2 - 2)/(B.1 + 7/2) = 0 := sorry

end circle_equation_fixed_point_exists_l358_358519


namespace length_PR_PS_eq_AF_l358_358053

variables (A B C D P S R Q F: Type) [Rectangle A B C D] 

variables [Segment P S] [Segment P R] [Segment A F] [Segment P Q] [Segment B D] [Segment A C] 

-- Conditions
variable (h1 : on_line A B P) -- P is any point on line segment AB.
variable (h2 : perp P S B D) -- PS is perpendicular to BD.
variable (h3 : perp P R A C) -- PR is perpendicular to AC.
variable (h4 : perp A F B D) -- AF is perpendicular to BD.
variable (h5 : perp P Q A F) -- PQ is perpendicular to AF.

-- Proof statement
theorem length_PR_PS_eq_AF : length (segment P R) + length (segment P S) = length (segment A F) :=
sorry

end length_PR_PS_eq_AF_l358_358053


namespace equal_roots_m_eq_minus_half_l358_358605

theorem equal_roots_m_eq_minus_half (x m : ℝ) 
  (h_eq: ∀ x, ( (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m )) :
  m = -1/2 := by 
  sorry

end equal_roots_m_eq_minus_half_l358_358605


namespace point_on_graph_l358_358037

variable (x y : ℝ)

-- Define the condition for a point to be on the graph of the function y = 6/x
def is_on_graph (x y : ℝ) : Prop :=
  x * y = 6

-- State the theorem to be proved
theorem point_on_graph : is_on_graph (-2) (-3) :=
  by
  sorry

end point_on_graph_l358_358037


namespace simplify_expression_l358_358658

theorem simplify_expression : (1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3))) = (1 / 39) :=
by
  sorry

end simplify_expression_l358_358658


namespace similarity_of_triangles_area_min_max_conditions_l358_358161

open EuclideanGeometry

variables {A B C P: Point}
variables {A1 B1 C1: Point}
variables (Ω: Circle)

/-- Conditions -/
axiom H1 : Ω.circumscribes △ABC
axiom H2 : P ∈ Ω
axiom H3 : A1 is_perpendicular_foot_from ▸ (P, altitude_from A (△ABC))
axiom H4 : B1 is_perpendicular_foot_from ▸ (P, altitude_from B (△ABC))
axiom H5 : C1 is_perpendicular_foot_from ▸ (P, altitude_from C (△ABC))

/-- Prove similarity -/
theorem similarity_of_triangles :
  similar (△A1 B1 C1) (△ABC) :=
sorry

/-- Area minimization and maximization -/
theorem area_min_max_conditions :
  ∃ O : Point, (O ∈ Ω.center) ∧ 
  (area_min_condition : area(△A1 B1 C1) minimizes at P when P lies on the closest point to O on diameter) ∧
  (area_max_condition : area(△A1 B1 C1) maximizes at P when P lies on the farthest point to O on diameter) :=
sorry

end similarity_of_triangles_area_min_max_conditions_l358_358161


namespace true_discount_correct_l358_358593

noncomputable def sum_due : ℝ := 768
noncomputable def interest_rate : ℝ := 0.14
noncomputable def time_period : ℝ := 3

noncomputable def present_value (S : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  S / (1 + r) ^ n

noncomputable def true_discount (S : ℝ) (PV : ℝ) : ℝ :=
  S - PV

theorem true_discount_correct :
  true_discount sum_due (present_value sum_due interest_rate time_period) ≈ 249.705 := by
  sorry

end true_discount_correct_l358_358593


namespace find_parallel_slope_l358_358238

theorem find_parallel_slope (a : ℝ) :
    (∀ x y : ℝ, (ax + 4y + 1 = 0) → (2x + y - 2 = 0) → (-a / 4 = -2)) → (a = 8) := by
  sorry

end find_parallel_slope_l358_358238


namespace parabola_slope_line_parabola_midpoint_l358_358564

theorem parabola_slope_line (M : ℝ × ℝ) (F : ℝ × ℝ) (d : ℝ) (k : ℝ) :
  M = (4, 0) → F = (1, 0) → d = sqrt 3 →
  |3 * k / real.sqrt (1 + k^2)| = sqrt 3 →
  k = sqrt 2 / 2 ∨ k = -sqrt 2 / 2 :=
by sorry

theorem parabola_midpoint (M : ℝ × ℝ) (A B : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (4, 0) →
  A.1^2 = 4 * A.2 →
  B.1^2 = 4 * B.2 →
  N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  N.1 = 2 :=
by sorry

end parabola_slope_line_parabola_midpoint_l358_358564


namespace least_prime_factor_of_11_pow4_minus_11_pow3_l358_358028

open Nat

theorem least_prime_factor_of_11_pow4_minus_11_pow3 : 
  Nat.minFac (11^4 - 11^3) = 2 :=
  sorry

end least_prime_factor_of_11_pow4_minus_11_pow3_l358_358028


namespace root_calculation_l358_358860

theorem root_calculation :
  (Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5)) ^ (1 / 4))) = 0.6687 :=
by
  sorry

end root_calculation_l358_358860


namespace product_even_of_permutation_algebraic_sum_not_2003_l358_358055

-- Problem 1
theorem product_even_of_permutation (a : Fin 9 → Fin 9) (h_perm : Function.Bijective a) : 
  ∃ n, (∏ i, (a i).val + 1 - (i + 1)) = 2 * n := 
sorry

-- Problem 2
theorem algebraic_sum_not_2003 (b : Fin 2003 → Bool) : 
  let sequence := (fun i => if b i then 1 else -1) * i.val^(i.val)  
  ∑ i, sequence i ≠ 2003 := 
sorry

end product_even_of_permutation_algebraic_sum_not_2003_l358_358055


namespace zoe_correct_percentage_l358_358246

variable (t : ℝ) -- total number of problems

-- Conditions
variable (chloe_solved_fraction : ℝ := 0.60)
variable (zoe_solved_fraction : ℝ := 0.40)
variable (chloe_correct_percentage_alone : ℝ := 0.75)
variable (chloe_correct_percentage_total : ℝ := 0.85)
variable (zoe_correct_percentage_alone : ℝ := 0.95)

theorem zoe_correct_percentage (h1 : chloe_solved_fraction = 0.60)
                               (h2 : zoe_solved_fraction = 0.40)
                               (h3 : chloe_correct_percentage_alone = 0.75)
                               (h4 : chloe_correct_percentage_total = 0.85)
                               (h5 : zoe_correct_percentage_alone = 0.95) :
  (zoe_correct_percentage_alone * zoe_solved_fraction * 100 + (chloe_correct_percentage_total - chloe_correct_percentage_alone * chloe_solved_fraction) * 100 = 78) :=
sorry

end zoe_correct_percentage_l358_358246


namespace solve_for_x_l358_358461

theorem solve_for_x (x : ℝ) (h : (1/3 : ℝ) * (x + 8 + 5*x + 3 + 3*x + 4) = 4*x + 1) : x = 4 :=
by {
  sorry
}

end solve_for_x_l358_358461


namespace ian_remaining_money_l358_358984

def total_hours : ℕ := 8
def hourly_rate_first_4_hours : ℕ := 18
def hourly_rate_next_4_hours : ℕ := 22
def expense_fraction : ℝ := 0.5
def tax_rate : ℝ := 0.10
def monthly_expense : ℕ := 50

theorem ian_remaining_money :
  let earnings := (4 * hourly_rate_first_4_hours) + (4 * hourly_rate_next_4_hours) in
  let spend := (earnings : ℝ) * expense_fraction in
  let taxes := (earnings : ℝ) * tax_rate in
  let deductions := spend + taxes + (monthly_expense : ℝ) in
  (earnings : ℝ) - deductions = 14 :=
by
  sorry

end ian_remaining_money_l358_358984


namespace second_group_work_days_l358_358580

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l358_358580


namespace sum_of_digits_in_binary_representation_of_315_l358_358770

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358770


namespace find_imag_part_of_complex_l358_358897

def complex_imag_part (z : ℂ) : ℂ := z.im

theorem find_imag_part_of_complex:
  let z := (3 * complex.I) / (1 - complex.I) * complex.I in
  complex_imag_part z = 3 / 2 :=
by
  sorry

end find_imag_part_of_complex_l358_358897


namespace teachers_students_probability_l358_358594

theorem teachers_students_probability :
  let total_arrangements := 4!
  let invalid_arrangements := 2! * 2!
  let probability := invalid_arrangements / total_arrangements
  probability = 1 / 6 :=
by
  -- Here we skip the proof for now
  sorry

end teachers_students_probability_l358_358594


namespace gcd_of_powers_of_two_l358_358023

def m : ℕ := 2^2100 - 1
def n : ℕ := 2^2000 - 1

theorem gcd_of_powers_of_two :
  Nat.gcd m n = 2^100 - 1 := sorry

end gcd_of_powers_of_two_l358_358023


namespace cyclists_meet_fourth_time_l358_358510

theorem cyclists_meet_fourth_time 
  (speed1 speed2 speed3 speed4 : ℕ)
  (len : ℚ)
  (t_start : ℕ)
  (h_speed1 : speed1 = 6)
  (h_speed2 : speed2 = 9)
  (h_speed3 : speed3 = 12)
  (h_speed4 : speed4 = 15)
  (h_len : len = 1 / 3)
  (h_t_start : t_start = 12 * 60 * 60)
  : 
  (t_start + 4 * (20 * 60 + 40)) = 12 * 60 * 60 + 1600  :=
sorry

end cyclists_meet_fourth_time_l358_358510


namespace find_a_plus_b_l358_358284

variables {x y z a b : ℝ}

noncomputable def log_base (a b : ℝ) := log b / log a

def satisfies_conditions (x y z : ℝ) : Prop :=
  log_base 2 (x + y) = z ∧ log_base 2 (x^2 + y^2) = z + 2

theorem find_a_plus_b (h : ∀ (x y z : ℝ), satisfies_conditions x y z → x^3 + y^3 = a * 2^(3 * z) + b * 2^(2 * z)) :
  a + b = 6.5 :=
sorry

end find_a_plus_b_l358_358284


namespace trigonometric_identity_l358_358513

noncomputable theory

theorem trigonometric_identity
  (x : ℝ)
  (h : Real.sin (x - (5 * Real.pi / 12)) = 1 / 3) :
  Real.cos ((2021 * Real.pi / 6) - 2 * x) = 7 / 9 :=
by
  sorry

end trigonometric_identity_l358_358513


namespace check_statements_l358_358176

theorem check_statements :
  ( (∀ p, (¬ p → q) → ¬ q) ∧
    (∀ p q, ¬ (p ∨ q) → ¬ p ∧ ¬ q) ∧
    (∀ x, (x > 2 → x > 1) ∧ (¬ ∀ x, x > 1 → x > 2)) ∧
    (¬ (∀ p, (p → "All team members in A are from Beijing") → ¬ ("All team members in A are not from Beijing"))) ) :=
by
  sorry

end check_statements_l358_358176


namespace find_two_digit_number_l358_358920

theorem find_two_digit_number :
  ∃ (n : ℕ),
  10 ≤ n ∧ n ≤ 99 ∧
  (let A := n / 10, B := n % 10 in A ≠ B ∧ n ^ 2 = (A + B) ^ 3) ∧
  n = 27 :=
by
  sorry

end find_two_digit_number_l358_358920


namespace part1_part2_l358_358563

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 2 * a + 3)

theorem part1 (x : ℝ) : f x 2 ≤ 9 ↔ -2 ≤ x ∧ x ≤ 4 :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ Set.Iic (-2 / 3) ∪ Set.Ici (14 / 3) :=
by sorry

end part1_part2_l358_358563


namespace scientific_notation_of_virus_diameter_l358_358691

theorem scientific_notation_of_virus_diameter :
  0.000000102 = 1.02 * 10 ^ (-7) :=
  sorry

end scientific_notation_of_virus_diameter_l358_358691


namespace highlighter_total_l358_358411

theorem highlighter_total 
  (pink_highlighters : ℕ)
  (yellow_highlighters : ℕ)
  (blue_highlighters : ℕ)
  (h_pink : pink_highlighters = 4)
  (h_yellow : yellow_highlighters = 2)
  (h_blue : blue_highlighters = 5) :
  pink_highlighters + yellow_highlighters + blue_highlighters = 11 :=
by
  sorry

end highlighter_total_l358_358411


namespace f_lt_g_for_n_ge_5_l358_358934

def f (n : ℕ) := n^2 + n
def g (n : ℕ) := 2^n

theorem f_lt_g_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : f(n) < g(n) :=
by sorry

end f_lt_g_for_n_ge_5_l358_358934


namespace sqrt_sum_eq_2_sqrt_7_l358_358881

theorem sqrt_sum_eq_2_sqrt_7 : 
  sqrt (10 - 2 * sqrt 21) + sqrt (10 + 2 * sqrt 21) = 2 * sqrt 7 :=
by
  sorry

end sqrt_sum_eq_2_sqrt_7_l358_358881


namespace centroid_fixed_min_area_fraction_l358_358798

variable {A B C D E F G : Type}
variable {x1 x2 x3 y1 y2 y3 : ℝ}
variable (t : ℝ)

noncomputable def point_A : point := ⟨x1, y1⟩
noncomputable def point_B : point := ⟨x2, y2⟩
noncomputable def point_C : point := ⟨x3, y3⟩
noncomputable def point_D : point := ⟨(1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2⟩
noncomputable def point_E : point := ⟨(1 - t) * x2 + t * x3, (1 - t) * y2 + t * y3⟩
noncomputable def point_F : point := ⟨(1 - t) * x3 + t * x1, (1 - t) * y3 + t * y1⟩

noncomputable def centroid_triangle (A B C : point) : point :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

theorem centroid_fixed :
  centroid_triangle (point_D t) (point_E t) (point_F t) = centroid_triangle point_A point_B point_C :=
sorry

noncomputable def area_triangle (A B C : point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

theorem min_area_fraction :
  ∃ t : ℝ, t ∈ Ioo 0 1 ∧ (area_triangle (point_D t) (point_E t) (point_F t)) * 4 = area_triangle point_A point_B point_C :=
sorry

end centroid_fixed_min_area_fraction_l358_358798


namespace subset_A_l358_358968

open Set

theorem subset_A (A : Set ℝ) (h : A = { x | x > -1 }) : {0} ⊆ A :=
by
  sorry

end subset_A_l358_358968


namespace gcd_36_54_l358_358733

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l358_358733


namespace prob_at_least_two_in_september_is_l358_358030

noncomputable def prob_at_least_two_in_september (n : ℕ) (prob_month : ℚ) : ℚ :=
let v1 := (11/12)^13 in
let v2 := 13 * (1/12) * (11/12)^12 in
1 - v1 - v2

theorem prob_at_least_two_in_september_is (n : ℕ) (prob_month : ℚ) :
  prob_at_least_two_in_september 13 (1/12) ≈ 0.296 :=
by
  sorry

end prob_at_least_two_in_september_is_l358_358030


namespace correct_proposition_l358_358394

noncomputable def plane (α : Type) : α → Prop := sorry
noncomputable def line (a : Type) : a → Prop := sorry
noncomputable def perpendicular (l : Type) (p : Type) : l → p → Prop := sorry
noncomputable def parallel (l : Type) (p : Type) : l → p → Prop := sorry

theorem correct_proposition
    (α1 α2 l: Type)
    (p1 : α1)
    (p2 : α2)
    (a : l) 
    (h1 : perpendicular l α1 a p1)
    (h2 : perpendicular l α2 a p2) :
  parallel α1 α2 :=
sorry

end correct_proposition_l358_358394


namespace length_of_highway_l358_358714

theorem length_of_highway 
  (speed_car1 : ℕ) (speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 40)
  (h_speed_car2 : speed_car2 = 60)
  (h_time : time = 5) : 
  (speed_car1 * time + speed_car2 * time = 500) :=
by
  rw [h_speed_car1, h_speed_car2, h_time]
  exact rfl

end length_of_highway_l358_358714


namespace planting_schemes_count_l358_358600

universe u

def hexagon_graph : SimpleGraph (Fin 6) :=
  SimpleGraph.cycle (Fin 6)

def num_plants : ℕ := 4

theorem planting_schemes_count : chromaticPolynomial hexagon_graph num_plants = 732 := by
  sorry

end planting_schemes_count_l358_358600


namespace probability_diamond_then_ace_l358_358713

/-- Assume we have a deck with two combined standard decks (104 cards total). 
There are 26 diamonds and 8 aces in this deck. -/
axiom total_cards : ℕ
axiom total_diamonds : ℕ
axiom total_aces : ℕ

/-- The probability that the first card is a diamond and the second card is an ace is given by: -/
theorem probability_diamond_then_ace :
  total_cards = 104 →
  total_diamonds = 26 →
  total_aces = 8 →
  let probability := ((2 / 104) * (7 / 103)) + ((24 / 104) * (8 / 103)) in
  (probability = 103 / 5356) :=
by
  intro h1 h2 h3
  let probability := ((2 / 104) * (7 / 103)) + ((24 / 104) * (8 / 103))
  have h : probability = 103 / 5356 := sorry
  exact h

end probability_diamond_then_ace_l358_358713


namespace smallest_value_of_a1_l358_358867

noncomputable def a : ℕ → ℝ
| 1      := x
| (n+1) := 9 * (a n) - n

theorem smallest_value_of_a1 : (∃ (x : ℝ), x ≥ 0 ∧ (∀ n > 1, a (n+1) = 9 * (a n) - n) ∧ x = 17 / 64) :=
sorry

end smallest_value_of_a1_l358_358867


namespace clothes_washer_final_price_l358_358065

theorem clothes_washer_final_price
  (P : ℝ) (d1 d2 d3 : ℝ)
  (hP : P = 500)
  (hd1 : d1 = 0.10)
  (hd2 : d2 = 0.20)
  (hd3 : d3 = 0.05) :
  (P * (1 - d1) * (1 - d2) * (1 - d3)) / P = 0.684 :=
by
  sorry

end clothes_washer_final_price_l358_358065


namespace inscribed_sphere_radius_correct_l358_358939

noncomputable def radius_of_inscribed_sphere (M A B C D : Point) (square_base: IsSquareBase A B C D) 
  (MA_eq_MD : MA = MD) (MA_perp_AB : isPerpendicular MA AB) (triangle_AMD_area : area (triangle M A D) = 1) : ℝ :=
  let radius := (sqrt 2) - 1 in
  radius

theorem inscribed_sphere_radius_correct (M A B C D : Point) 
  (square_base: IsSquareBase A B C D) (MA_eq_MD : MA = MD) 
  (MA_perp_AB : isPerpendicular MA AB) (triangle_AMD_area : area (triangle M A D) = 1) : 
  radius_of_inscribed_sphere M A B C D square_base MA_eq_MD MA_perp_AB triangle_AMD_area = sqrt 2 - 1 :=
by
  sorry

end inscribed_sphere_radius_correct_l358_358939


namespace selection_assignment_schemes_l358_358652

noncomputable def number_of_selection_schemes (males females : ℕ) : ℕ :=
  if h : males + females < 3 then 0
  else
    let total3 := Nat.choose (males + females) 3
    let all_males := if hM : males < 3 then 0 else Nat.choose males 3
    let all_females := if hF : females < 3 then 0 else Nat.choose females 3
    total3 - all_males - all_females

theorem selection_assignment_schemes :
  number_of_selection_schemes 4 3 = 30 :=
by sorry

end selection_assignment_schemes_l358_358652


namespace part_a_part_b_l358_358154

def largest_integer_le (x : ℝ) : ℤ := Int.floor x

def is_arithmetic_progression (seq : List ℤ) : Prop :=
  ∃ (b : ℤ), ∀ i, i < seq.length - 1 → (seq.get ⟨i+1, sorry⟩ - seq.get ⟨i, sorry⟩) = b

def set_S (α : ℝ) : Set ℤ := { z | ∃ n : ℤ, z = largest_integer_le (n * α)}

noncomputable def α := 2.5  -- Example irrational number > 2

theorem part_a (m : ℕ) (hm : 3 ≤ m) : 
  ∃ (seq : List ℤ), (seq.length = m ∧ (∀ i j, i < j → (seq.get ⟨i, sorry⟩ ≠ seq.get ⟨j, sorry⟩)) ∧ is_arithmetic_progression seq) := 
sorry

theorem part_b : 
  ¬ ∃ f : ℕ → ℤ, (∀ i, f i ∈ set_S α) ∧ is_arithmetic_progression (List.ofFn f) := 
sorry

end part_a_part_b_l358_358154


namespace months_b_after_a_started_business_l358_358831

theorem months_b_after_a_started_business
  (A_initial : ℝ)
  (B_initial : ℝ)
  (profit_ratio : ℝ)
  (A_investment_time : ℕ)
  (B_investment_time : ℕ)
  (investment_ratio : A_initial * A_investment_time / (B_initial * B_investment_time) = profit_ratio) :
  B_investment_time = 6 :=
by
  -- Given:
  -- A_initial = 3500
  -- B_initial = 10500
  -- profit_ratio = 2 / 3
  -- A_investment_time = 12 months
  -- B_investment_time = 12 - x months
  -- We need to prove that x = 6 months such that investment ratio matches profit ratio.
  sorry

end months_b_after_a_started_business_l358_358831


namespace tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l358_358202

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

-- First proof problem: Equation of the tangent line at (2, 7)
theorem tangent_line_at_2_is_12x_minus_y_minus_17_eq_0 :
  ∀ x y : ℝ, y = f x → (x = 2) → y = 7 → (∃ (m b : ℝ), (m = 12) ∧ (b = -17) ∧ (∀ x, 12 * x - y - 17 = 0)) :=
by
  sorry

-- Second proof problem: Range of m for three distinct real roots
theorem range_of_m_for_three_distinct_real_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) → -3 < m ∧ m < -2 :=
by 
  sorry

end tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l358_358202


namespace three_planes_divide_space_into_at_most_eight_parts_l358_358010

theorem three_planes_divide_space_into_at_most_eight_parts :
  ∃ (planes : list (set ℝ^3)), planes.length = 3 ∧
  (∀ i j k, i < j ∧ j < k → intersection (intersection (planes.nth i) (planes.nth j)) (planes.nth k) ≠ ∅) ∧
  ∃ partition, partition.count = 8 :=
sorry

end three_planes_divide_space_into_at_most_eight_parts_l358_358010


namespace sum_of_binary_digits_of_315_l358_358753

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l358_358753


namespace quadrilateral_perimeter_sum_l358_358826

theorem quadrilateral_perimeter_sum (a b : ℤ) :
  let d1 := real.sqrt ((4 - 1) ^ 2 + (5 - 2) ^ 2),
      d2 := real.sqrt ((5 - 4) ^ 2 + (4 - 5) ^ 2),
      d3 := real.sqrt ((4 - 5) ^ 2 + (1 - 4) ^ 2),
      d4 := real.sqrt ((1 - 4) ^ 2 + (2 - 1) ^ 2)
  in d1 + d2 + d3 + d4 = (real.sqrt 2 * a + real.sqrt 10 * b) → (a + b) = 6 :=
by 
  -- Insert detailed proof steps here.
  sorry

end quadrilateral_perimeter_sum_l358_358826


namespace volume_of_rotated_rectangle_l358_358913

-- Define the problem conditions
def length : ℝ := 20
def width : ℝ := 10

-- Define the radius and height of the cylinder
def radius : ℝ := width / 2
def height : ℝ := length

-- Volume of the cylinder formula
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Statement of the theorem
theorem volume_of_rotated_rectangle :
  volume_cylinder radius height = 500 * π := 
by
  -- Proof skipped
  sorry

end volume_of_rotated_rectangle_l358_358913


namespace triangle_area_l358_358840

theorem triangle_area (a b c : ℕ) (h : a * a + b * b = c * c) : 1 / 2 * a * b = 84 := by
  have ha : a = 7 := rfl
  have hb : b = 24 := rfl
  have hc : c = 25 := rfl
  rw [ha, hb, hc] at h
  norm_num at h
  norm_num
  sorry

end triangle_area_l358_358840


namespace max_modulus_l358_358586

open Complex

theorem max_modulus (z : ℂ) (hz : abs z = 1) : 
  (∀ z, abs z = 1 → 
  ∃ M : ℝ, M = sqrt 7 + sqrt 5 ∧ ∀ ε > 0, abs (abs ((sqrt 3 * I - z) / (sqrt 2 - z)) - M) < ε) :=
by
  sorry

end max_modulus_l358_358586


namespace triangle_is_isosceles_right_triangle_l358_358549

variables {V : Type*} [InnerProductSpace ℝ V]
variables (A B C : V)
variables (AB AC BC CB : V)

noncomputable def is_isosceles_right_triangle (A B C : V) : Prop :=
  let AB := B - A in
  let AC := C - A in
  let BC := C - B in
  AB ≠ 0 ∧ AC ≠ 0 ∧ 
  (AB / ∥AB∥ + AC / ∥AC∥) ⋅ BC = 0 ∧
  (AB ⋅ (A - B) = ∥AB∥ ^ 2)

theorem triangle_is_isosceles_right_triangle
  (A B C : V)
  (h1 : (AB / ∥AB∥ + AC / ∥AC∥) ⋅ BC = 0)
  (h2 : AB ⋅ (A - B) = ∥AB∥ ^ 2) : is_isosceles_right_triangle A B C :=
sorry

end triangle_is_isosceles_right_triangle_l358_358549


namespace number_of_ways_to_fill_grid_l358_358854

noncomputable def totalWaysToFillGrid (S : Finset ℕ) : ℕ :=
  S.card.choose 5

theorem number_of_ways_to_fill_grid : totalWaysToFillGrid ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 6 :=
by
  sorry

end number_of_ways_to_fill_grid_l358_358854


namespace fraction_division_l358_358459

theorem fraction_division : 
  ((8 / 4) * (9 / 3) * (20 / 5)) / ((10 / 5) * (12 / 4) * (15 / 3)) = (4 / 5) := 
by
  sorry

end fraction_division_l358_358459


namespace equilateral_triangle_area_is_correct_l358_358014

def side_length : ℝ := 8

def area_equilateral_triangle (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s^2

theorem equilateral_triangle_area_is_correct :
  area_equilateral_triangle side_length = 16 * real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_is_correct_l358_358014


namespace congruent_angles_with_parallel_sides_l358_358588

-- Definitions for the conditions.
def congruent_angles (α β : ℝ) : Prop := α = β
def parallel (u v : Prop) : Prop := u = v

-- The proof problem statement.
theorem congruent_angles_with_parallel_sides (α β : ℝ) (u v : Prop) (h1 : congruent_angles α β) 
  (h2 : parallel u v) : 
  ¬(∀ w, (parallel u w ∨ ¬(parallel u w) ∨ perpendicular u w)) :=
sorry

end congruent_angles_with_parallel_sides_l358_358588


namespace mix_solutions_l358_358322

variables (Vx : ℚ)

def alcohol_content_x (Vx : ℚ) : ℚ := 0.10 * Vx
def alcohol_content_y : ℚ := 0.30 * 450
def final_alcohol_content (Vx : ℚ) : ℚ := 0.22 * (Vx + 450)

theorem mix_solutions (Vx : ℚ) (h : 0.10 * Vx + 0.30 * 450 = 0.22 * (Vx + 450)) :
  Vx = 300 :=
sorry

end mix_solutions_l358_358322


namespace part_1_part_2_equality_case_l358_358168

variables {m n : ℝ}

-- Definition of positive real numbers and given condition m > n and n > 1
def conditions_1 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m > n ∧ n > 1

-- Prove that given conditions, m^2 + n > mn + m
theorem part_1 (m n : ℝ) (h : conditions_1 m n) : m^2 + n > m * n + m :=
  by sorry

-- Definition of the condition m + 2n = 1
def conditions_2 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m + 2 * n = 1

-- Prove that given conditions, (2/m) + (1/n) ≥ 8
theorem part_2 (m n : ℝ) (h : conditions_2 m n) : (2 / m) + (1 / n) ≥ 8 :=
  by sorry

-- Prove that the minimum value is obtained when m = 2n = 1/2
theorem equality_case (m n : ℝ) (h : conditions_2 m n) : 
  (2 / m) + (1 / n) = 8 ↔ m = 1/2 ∧ n = 1/4 :=
  by sorry

end part_1_part_2_equality_case_l358_358168


namespace sum_of_digits_in_binary_representation_of_315_l358_358771

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358771


namespace option_a_option_d_l358_358933

theorem option_a (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m = Nat.choose n (n - m) := 
sorry

theorem option_d (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m := 
sorry

end option_a_option_d_l358_358933


namespace constant_term_expansion_l358_358162

theorem constant_term_expansion : 
  let a := (2 / Real.pi) * (∫ x in -1..1, Real.sqrt (1 - x^2))
  (x + (a / x))^6 = 
  15 :=
by
  sorry

end constant_term_expansion_l358_358162


namespace number_of_random_events_is_3_l358_358451

def event1 := "Tossing the same dice twice in a row and getting a 2 both times"
def event2 := "It raining tomorrow"
def event3 := "Someone winning the lottery"
def event4 := "Selecting two elements from the set {1, 2, 3}, and their sum being greater than 2"
def event5 := "Water boiling when heated to 90°C under standard atmospheric pressure"

-- Define what it means for an event to be random
def is_random_event (event : String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 ∨
  (event ≠ event4 ∧ event ≠ event5)

-- Define the list of events
def events := [event1, event2, event3, event4, event5]

-- Calculate the number of random events
def number_of_random_events : ℕ :=
  (events.count (is_random_event)) -- count the random events

-- main theorem
theorem number_of_random_events_is_3 : number_of_random_events = 3 :=
by sorry

end number_of_random_events_is_3_l358_358451


namespace y_in_terms_of_x_l358_358627

variable {p : ℝ}
def x (p : ℝ) := 3 + 3^p
def y (p : ℝ) := 3 + 3^(-p)

theorem y_in_terms_of_x (h : x p = 3 + 3^p) : y p = 3 + 3^(-p) := by
  rw [←h]
  -- placeholder for the proof steps
  sorry

end y_in_terms_of_x_l358_358627


namespace maximum_allied_subset_size_l358_358215

def is_league_pair (a b : ℕ) : Prop :=
  ∃ d, d > 1 ∧ d ∣ a ∧ d ∣ b ∧ ¬(a ∣ b) ∧ ¬(b ∣ a)

theorem maximum_allied_subset_size : ∀ A ⊆ {i | 1 ≤ i ∧ i ≤ 2014},
  (∀ a b ∈ A, is_league_pair a b) → A.card ≤ 504 :=
sorry

end maximum_allied_subset_size_l358_358215


namespace cubic_geometric_progression_l358_358955

theorem cubic_geometric_progression (a b c : ℝ) (α β γ : ℝ) 
    (h_eq1 : α + β + γ = -a) 
    (h_eq2 : α * β + α * γ + β * γ = b) 
    (h_eq3 : α * β * γ = -c) 
    (h_gp : ∃ k q : ℝ, α = k / q ∧ β = k ∧ γ = k * q) : 
    a^3 * c - b^3 = 0 :=
by
  sorry

end cubic_geometric_progression_l358_358955


namespace sum_of_digits_base2_315_l358_358762

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358762


namespace more_pencils_than_pens_l358_358690

theorem more_pencils_than_pens : 
  ∀ (P L : ℕ), L = 30 → (P / L: ℚ) = 5 / 6 → ((L - P) = 5) := by
  intros P L hL hRatio
  sorry

end more_pencils_than_pens_l358_358690


namespace perpendicular_bisector_point_eq_triangle_circumcenter_exists_quadrilateral_circumcircle_exists_l358_358793

-- Part (a)
theorem perpendicular_bisector_point_eq {X Y P : Type} [MetricSpace P] (M : P) 
  (h_midpoint : dist M X = dist M Y) (h_perp : dist M X = dist M Y) :
  dist P X = dist P Y := 
sorry

-- Part (b)
theorem triangle_circumcenter_exists {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] : 
  ∃ O : Type, dist O A = dist O B ∧ dist O B = dist O C :=
sorry

-- Part (c)
theorem quadrilateral_circumcircle_exists {A B C D : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] (P : Type) 
  (h1 : dist P A = dist P B) (h2 : dist P A = dist P C) (h3 : dist P A = dist P D) : 
    ∃ R : Type, ∀ Q ∈ {A, B, C, D}, dist P Q = R :=
sorry

end perpendicular_bisector_point_eq_triangle_circumcenter_exists_quadrilateral_circumcircle_exists_l358_358793


namespace ribbon_tape_needed_l358_358654

theorem ribbon_tape_needed 
  (total_length : ℝ) (num_boxes : ℕ) (ribbon_per_box : ℝ)
  (h1 : total_length = 82.04)
  (h2 : num_boxes = 28)
  (h3 : total_length / num_boxes = ribbon_per_box)
  : ribbon_per_box = 2.93 :=
sorry

end ribbon_tape_needed_l358_358654


namespace fraction_power_calc_l358_358863

theorem fraction_power_calc : 
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
sorry

end fraction_power_calc_l358_358863


namespace smaller_circle_radius_l358_358830

theorem smaller_circle_radius (r1 r2 : ℝ) (A1 A2 : ℝ) (hlarger_radius : r1 = 5) (harea_condition : 2 * A2 = A1 + 25 * real.pi) (harea_relation : A1 + A2 = 25 * real.pi) : r2 = 5 * real.sqrt 2 / 2 :=
by
  -- Define the area of the larger circle
  let A_larger := real.pi * r1^2
  -- Define the area difference as A2
  let A_diff := A_larger - A1
  -- Use the given conditions to derive the radius of the smaller circle
  have A2_eq : A2 = 25 * real.pi / 2,
    from (eq_div_of_mul_eq harea_condition).mpr (by simp [harea_relation, hlarger_radius, add_comm])
  -- Calculate the radius of the smaller circle from A1
  have A1_eq : A1 = (25 * real.pi) - A2_eq,
    by rw [harea_relation, A2_eq, sub_eq_add_neg]
  -- Therefore, r2^2 should equal A1 / pi
  exact real.sqrt_modeq_of_sq A1_eq sorry /-

end smaller_circle_radius_l358_358830


namespace imaginary_part_of_complex_z_l358_358345

noncomputable def complex_z : ℂ := (1 + Complex.I) / (1 - Complex.I) + (1 - Complex.I) ^ 2

theorem imaginary_part_of_complex_z : complex_z.im = -1 := by
  sorry

end imaginary_part_of_complex_z_l358_358345


namespace cubic_has_three_natural_roots_l358_358886

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l358_358886


namespace part_one_part_two_l358_358960

noncomputable def f (x a : ℝ) : ℝ :=
  Real.log (1 + x) + a * Real.cos x

noncomputable def g (x : ℝ) : ℝ :=
  f x 2 - 1 / (1 + x)

theorem part_one (a : ℝ) : 
  (∀ x, f x a = Real.log (1 + x) + a * Real.cos x) ∧ 
  f 0 a = 2 ∧ 
  (∀ x, x + f (0:ℝ) a = x + 2) → 
  a = 2 := 
sorry

theorem part_two : 
  (∀ x, g x = Real.log (1 + x) + 2 * Real.cos x - 1 / (1 + x)) →
  (∃ y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) ∧ 
  (∀ x, -1 < x ∧ x < (Real.pi / 2) → g x ≠ 0) →
  (∃! y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) :=
sorry

end part_one_part_two_l358_358960


namespace resulting_vector_after_rotation_l358_358366

-- Define the initial vector and the rotation transformation
def initial_vector : ℝ × ℝ × ℝ := (2, 1, 3)

def rotate_180 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-v.1, -v.2, -v.3)

theorem resulting_vector_after_rotation :
  rotate_180 initial_vector = (-2, -1, -3) := 
by
  sorry

end resulting_vector_after_rotation_l358_358366


namespace sum_series_equality_l358_358862

theorem sum_series_equality : 
  (∑ n in Finset.range 2015, (n + 1) / ((n + 2)! : ℝ)) = 1 - 1 / (2016! : ℝ) := 
by
  sorry

end sum_series_equality_l358_358862


namespace yasmine_chocolate_beverage_l358_358042

theorem yasmine_chocolate_beverage :
  ∃ (m s : ℕ), (∀ k : ℕ, k > 0 → (∃ n : ℕ, 4 * n = 7 * k) → (m, s) = (7 * k, 4 * k)) ∧
  (2 * 7 * 1 + 1.4 * 4 * 1) = 19.6 := by
sorry

end yasmine_chocolate_beverage_l358_358042


namespace train_speed_conversion_l358_358834

def km_per_hour_to_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 108) :
  km_per_hour_to_m_per_s speed_kmph = 30 :=
by
  rw [h]
  sorry

end train_speed_conversion_l358_358834


namespace number_without_digit_two_323_l358_358982

def contains_digit_two (n : Nat) : Bool :=
  (toString n).any (λ ch, ch = '2')

def count_numbers_without_digit_two (n : Nat) : Nat :=
  (List.range n).count (λ k, ¬ contains_digit_two k)

theorem number_without_digit_two_323 :
  count_numbers_without_digit_two 500 = 323 :=
by
  sorry

end number_without_digit_two_323_l358_358982


namespace problem1_problem2_problem3_problem4_l358_358794

-- SI. 1
theorem problem1 (b m n : ℕ) (A : ℝ) (h1 : b = 4) (h2 : m = 1) (h3 : n = 1) (h4 : A = (b^m)^n + b^(m+n)) :
  A = 20 := 
by 
  simp [h1, h2, h3, h4]; 
  sorry

-- SI. 2
theorem problem2 (A B : ℝ) (h1 : A = 20) (h2 : 2^A = B^10) (h3 : 0 < B) :
  B = 4 := 
by 
  simp [h1, h2, h3]; 
  sorry

-- SI. 3
theorem problem3 (B C : ℝ) (h1 : B = 4) (h2 : (sqrt((20 * B + 45) / C)) = C) :
  C = 5 := 
by 
  simp [h1, h2];
  sorry

-- SI. 4
theorem problem4 (C D : ℝ) (h1 : C = 5) (h2 : D = C * (Real.sin (30 : ℝ) * (Real.pi / 180))) :
  D = 2.5 := 
by 
  simp [h1, h2];
  sorry

end problem1_problem2_problem3_problem4_l358_358794


namespace probability_alex_meets_train_l358_358842

-- Define the conditions as functions and types in Lean
def train_arrival (t : ℝ) : Prop := t ∈ Icc (0 : ℝ) 90
def alex_arrival (a : ℝ) : Prop := a ∈ Icc (0 : ℝ) 90

-- Define the event where Alex meets the train
def meets (t a : ℝ) : Prop := a ∈ Icc t (t + 15)

-- The main theorem stating the probability
theorem probability_alex_meets_train : 
  (∫ x in Icc 0 75, ∫ y in Icc x (x + 15), 1) / (90 * 90) = 7 / 9 := 
by
  sorry

end probability_alex_meets_train_l358_358842


namespace ab_eq_zero_l358_358347

theorem ab_eq_zero (a b : ℤ) (h : ∀ (m n : ℕ), ∃ k : ℕ, am^2 + b n^2 = k^2) : a * b = 0 :=
sorry

end ab_eq_zero_l358_358347


namespace AB_perp_AD_rectangle_ABCD_cosine_angle_l358_358972

-- Define the points A, B, and D as given.
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (-1, 4)

-- Define the vector AB and vector AD
def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)

-- Prove that vectors AB and AD are orthogonal
theorem AB_perp_AD : vector_AB.1 * vector_AD.1 + vector_AB.2 * vector_AD.2 = 0 :=
  sorry

-- Define point C as having coordinates that make ABCD a rectangle.
def C : ℝ × ℝ := (0, 5)

-- Define the vectors AC and BD
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vector_BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Prove the coordinates of C and the cosine value of the acute angle between the diagonals
theorem rectangle_ABCD_cosine_angle : 
  C = (0, 5) ∧ (vector_AC.1 * vector_BD.1 + vector_AC.2 * vector_BD.2) / (real.sqrt (vector_AC.1 ^ 2 + vector_AC.2 ^ 2) * real.sqrt (vector_BD.1 ^ 2 + vector_BD.2 ^ 2)) = 4 / 5 :=
  sorry

end AB_perp_AD_rectangle_ABCD_cosine_angle_l358_358972


namespace range_of_a_l358_358297

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1 / 2) * x - 1
  else 1 / x

theorem range_of_a (a : ℝ) : f a > a → a ∈ set.Ioo (-1) 0 :=
sorry

end range_of_a_l358_358297


namespace unique_x1_exists_l358_358926

noncomputable def sequence (x₁ : ℝ) : ℕ → ℝ
| 0 => x₁
| (n + 1) => let x_n := sequence n in x_n * (x_n + (1 / (n + 1)))
  
theorem unique_x1_exists :
  ∃! x₁ : ℝ, ∀ n : ℕ, 0 < sequence x₁ n ∧ sequence x₁ n < sequence x₁ (n + 1) ∧ sequence x₁ n < 1 :=
sorry

end unique_x1_exists_l358_358926


namespace product_identity_l358_358116

def e (x : ℝ) : ℂ := Complex.exp (2 * Real.pi * Complex.I * x)

theorem product_identity :
  (∏ k in Finset.range 8, ∏ j in Finset.range 6, (e (j / 7) - e (k / 9))) = 1 :=
  sorry

end product_identity_l358_358116


namespace cube_vertices_faces_edges_l358_358686

theorem cube_vertices_faces_edges (V F E : ℕ) (hv : V = 8) (hf : F = 6) (euler : V - E + F = 2) : E = 12 :=
by
  sorry

end cube_vertices_faces_edges_l358_358686


namespace sum_of_digits_of_binary_315_is_6_l358_358747
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358747


namespace find_ratio_of_OM_to_radius_l358_358706

noncomputable def OM_to_circumcircle_ratio (O M H1 H2 H3 : Point) (l1 l2 l3 : Line)
                                           (H1_perp_l1 : Perpendicular M l1 H1)
                                           (H2_perp_l2 : Perpendicular M l2 H2)
                                           (H3_perp_l3 : Perpendicular M l3 H3)
                                           (l1_intersect_l2_l3 : IntersectAt l1 l2 l3 O) : ℝ :=
  let OM := dist O M in
  let R := circumradius H1 H2 H3 in
  OM / R

theorem find_ratio_of_OM_to_radius (O M H1 H2 H3 : Point) (l1 l2 l3 : Line)
                                   (H1_perp_l1 : Perpendicular M l1 H1)
                                   (H2_perp_l2 : Perpendicular M l2 H2)
                                   (H3_perp_l3 : Perpendicular M l3 H3)
                                   (l1_intersect_l2_l3 : IntersectAt l1 l2 l3 O) :
  OM_to_circumcircle_ratio O M H1 H2 H3 l1 l2 l3 H1_perp_l1 H2_perp_l2 H3_perp_l3 l1_intersect_l2_l3 = 2 := 
sorry

end find_ratio_of_OM_to_radius_l358_358706


namespace sequence_missing_number_l358_358803

theorem sequence_missing_number : 
  ∃ x, (x - 21 = 7 ∧ 37 - x = 9) ∧ x = 28 := by
  sorry

end sequence_missing_number_l358_358803


namespace solve_eq_l358_358326

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l358_358326


namespace average_five_students_l358_358653

/-- Definition of individual and average scores --/
def average(scores : List ℝ) : ℝ := scores.sum / scores.length

/-- Conditions given in the problem --/
def scores_condition_1 := average [92, 92, 92] = 92
def scores_condition_2 := average [90] = 90
def scores_condition_3 := average [95] = 95

/-- The main theorem to solve --/
theorem average_five_students : average [92, 92, 92, 90, 95] = 92.2 :=
by 
  sorry

end average_five_students_l358_358653


namespace olympic_triathlon_total_distance_l358_358256

theorem olympic_triathlon_total_distance (x : ℝ) (L S : ℝ)
  (hL : L = 4 * x)
  (hS : S = (3 / 80) * x)
  (h_diff : L - S = 8.5) :
  x + L + S = 51.5 := by
  sorry

end olympic_triathlon_total_distance_l358_358256


namespace solve_for_x_l358_358666

def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x (x : ℝ) : (custom_mul 3 (custom_mul 6 x) = 2) → (x = 19 / 2) :=
sorry

end solve_for_x_l358_358666


namespace quadractic_b_value_l358_358929

def quadratic_coefficients (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem quadractic_b_value :
  ∀ (a b c : ℝ), quadratic_coefficients 1 (-2) (-3) (x : ℝ) → 
  b = -2 := by
  sorry

end quadractic_b_value_l358_358929


namespace solve_for_x_over_z_l358_358196

variables (x y z : ℝ)

theorem solve_for_x_over_z
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y = 6 * z) :
  x / z = 5 :=
sorry

end solve_for_x_over_z_l358_358196


namespace yellow_yellow_pairs_count_l358_358457

def num_blue_students : ℕ := 75
def num_yellow_students : ℕ := 105
def total_pairs : ℕ := 90
def blue_blue_pairs : ℕ := 30

theorem yellow_yellow_pairs_count :
  -- number of pairs where both students are wearing yellow shirts is 45.
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 :=
by
  sorry

end yellow_yellow_pairs_count_l358_358457


namespace max_cross_section_area_l358_358083

noncomputable def prism_cross_section_area : ℝ :=
  let z_axis_parallel := true
  let square_base := 8
  let plane := ∀ x y z, 3 * x - 5 * y + 2 * z = 20
  121.6

theorem max_cross_section_area :
  prism_cross_section_area = 121.6 :=
sorry

end max_cross_section_area_l358_358083


namespace unique_tangency_point_l358_358613

theorem unique_tangency_point (A B C X : Type) (a b : ℕ) [geometry.Point A] [geometry.Point B] [geometry.Point C] [geometry.Point X]
  (h1 : dist A B = 7) (h2 : dist B C = 8) (h3 : dist C A = 9)
  (h4 : dist X B = dist X C) 
  (h5 : tangent (circumcircle A B C) (A X)) 
  (h6 : XA = a / b) 
  (h7 : nat.coprime a b) :
  a + b = 61 :=
sorry

end unique_tangency_point_l358_358613


namespace monotonic_intervals_exists_a_gt_e2_div_2e_sub_1_l358_358962

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 2 * a
noncomputable def g (a x : ℝ) : ℝ := x + a / x

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → (∀ x, 0 < x → g a x > g a 0)) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < Real.sqrt a ∨ x > Real.sqrt a → g a x > g a (Real.sqrt a) ∧ 
            ∀ x, 0 < x ∧ x < Real.sqrt a ∨ x > Real.sqrt a → g a x < g a 0)) :=
sorry

theorem exists_a_gt_e2_div_2e_sub_1 (a : ℝ) :
  a > 0 → (∀ x1 x2 ∈ set.Icc 1 Real.exp, f a x1 - g a x2 > 0) ↔ 
  a ∈ set.Ioi (Real.exp ^ 2 / (2 * Real.exp - 1)) :=
sorry

end monotonic_intervals_exists_a_gt_e2_div_2e_sub_1_l358_358962


namespace gcd_36_54_l358_358732

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l358_358732


namespace path_length_l358_358074

theorem path_length (scale_ratio : ℕ) (map_path_length : ℝ) 
  (h1 : scale_ratio = 500)
  (h2 : map_path_length = 3.5) : 
  (map_path_length * scale_ratio = 1750) :=
sorry

end path_length_l358_358074


namespace slower_train_speed_l358_358382

theorem slower_train_speed (faster_speed : ℝ) (time_passed : ℝ) (train_length : ℝ) (slower_speed: ℝ) :
  faster_speed = 50 ∧ time_passed = 15 ∧ train_length = 75 →
  slower_speed = 32 :=
by
  intro h
  sorry

end slower_train_speed_l358_358382


namespace janet_litter_change_frequency_l358_358278

-- Define the function how_often_changes with the given conditions
def how_often_changes (container_weight : ℕ) (container_price : ℕ) (portion_weight : ℕ) (total_cost : ℕ) (total_days : ℕ) : ℕ :=
  let portions_per_container := container_weight / portion_weight in
  let containers := total_cost / container_price in
  let total_portions := containers * portions_per_container in
  total_days / total_portions

-- Specify the proof problem
theorem janet_litter_change_frequency :
  how_often_changes 45 21 15 210 210 = 7 :=
by
  sorry

end janet_litter_change_frequency_l358_358278


namespace analytical_expression_monotonic_intervals_and_range_l358_358954

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

theorem analytical_expression (x : ℝ) : f x = if x ≤ 0 then x^2 + 2*x else x^2 - 2*x :=
by rfl

theorem monotonic_intervals_and_range :
  (∀ x, x ∈ (-∞, -1) → decreasing_on f (Ioo (-(1:ℝ)) (-1))) ∧
  (∀ x, x ∈ (-1, 0) → increasing_on f (Ioo (-1) (0))) ∧
  (∀ x, x ∈ (0, 1) → decreasing_on f (Ioo (0) (1))) ∧
  (∀ x, x ∈ (1, ∞) → increasing_on f (Ioo (1) (∞))) ∧
  (∀ y, y ∈ range f → y ≥ -1) :=
begin
  sorry,
end

end analytical_expression_monotonic_intervals_and_range_l358_358954


namespace minimum_AB_l358_358260

noncomputable def shortest_AB (a : ℝ) : ℝ :=
  let x := (Real.sqrt 3) / 4 * a
  x

theorem minimum_AB (a : ℝ) : ∃ x, (x = (Real.sqrt 3) / 4 * a) ∧ ∀ y, (y = (Real.sqrt 3) / 4 * a) → shortest_AB a = x :=
by
  sorry

end minimum_AB_l358_358260


namespace sum_of_squares_ends_in_9_l358_358696

open Nat

-- Defining the two prime numbers p1 and p2
def p1 := 2
def p2 := 5

-- Stating that p1 and p2 are prime
def p1_is_prime : Prime p1 := by sorry
def p2_is_prime : Prime p2 := by sorry

-- Stating our main theorem involving the sum of squares condition
theorem sum_of_squares_ends_in_9 :
  (p1^2 + p2^2) % 10 = 9 := by
  sorry

end sum_of_squares_ends_in_9_l358_358696


namespace no_valid_pairs_l358_358485

theorem no_valid_pairs (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ¬(1000 * a + 100 * b + 32) % 99 = 0 :=
by
  sorry

end no_valid_pairs_l358_358485


namespace incircle_area_of_triangle_l358_358617

noncomputable def hyperbola_params : Type :=
  sorry

noncomputable def point_on_hyperbola (P : hyperbola_params) : Prop :=
  sorry

noncomputable def in_first_quadrant (P : hyperbola_params) : Prop :=
  sorry

noncomputable def distance_ratio (PF1 PF2 : ℝ) : Prop :=
  PF1 / PF2 = 4 / 3

noncomputable def distance1_is_8 (PF1 : ℝ) : Prop :=
  PF1 = 8

noncomputable def distance2_is_6 (PF2 : ℝ) : Prop :=
  PF2 = 6

noncomputable def distance_between_foci (F1F2 : ℝ) : Prop :=
  F1F2 = 10

noncomputable def incircle_area (area : ℝ) : Prop :=
  area = 4 * Real.pi

theorem incircle_area_of_triangle (P : hyperbola_params) 
  (hP : point_on_hyperbola P) 
  (h1 : in_first_quadrant P)
  (PF1 PF2 : ℝ)
  (h2 : distance_ratio PF1 PF2)
  (h3 : distance1_is_8 PF1)
  (h4 : distance2_is_6 PF2)
  (F1F2 : ℝ) 
  (h5 : distance_between_foci F1F2) :
  ∃ r : ℝ, incircle_area (Real.pi * r^2) :=
by
  sorry

end incircle_area_of_triangle_l358_358617


namespace chord_length_circle_line_l358_358899

-- Define the conditions
def circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def line (x y : ℝ) : Prop := 3*x - 4*y - 4 = 0

-- Define the proof problem
theorem chord_length_circle_line
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : circle x₁ y₁)
  (h₂ : circle x₂ y₂)
  (h₃ : line x₁ y₁)
  (h₄ : line x₂ y₂) :
  dist (x₁, y₁) (x₂, y₂) = 4 * real.sqrt 2 :=
sorry

end chord_length_circle_line_l358_358899


namespace sum_of_digits_base2_315_l358_358765

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358765


namespace numbers_without_digit_2_1_to_500_l358_358981

def count_numbers_without_digit_2 (n : ℕ) : ℕ :=
  n.digits 10 |>.all (λ d, d ≠ 2)

theorem numbers_without_digit_2_1_to_500 : 
  (Finset.range 500).filter count_numbers_without_digit_2 |> Finset.card = 323 := 
by
  sorry

end numbers_without_digit_2_1_to_500_l358_358981


namespace tax_budget_level_correct_l358_358477

-- Definitions for tax types and their corresponding budget levels
inductive TaxType where
| property_tax_organizations : TaxType
| federal_tax : TaxType
| profit_tax_organizations : TaxType
| tax_subjects_RF : TaxType
| transport_collecting : TaxType
deriving DecidableEq

inductive BudgetLevel where
| federal_budget : BudgetLevel
| subjects_RF_budget : BudgetLevel
deriving DecidableEq

def tax_to_budget_level : TaxType → BudgetLevel
| TaxType.property_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.federal_tax => BudgetLevel.federal_budget
| TaxType.profit_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.tax_subjects_RF => BudgetLevel.subjects_RF_budget
| TaxType.transport_collecting => BudgetLevel.subjects_RF_budget

theorem tax_budget_level_correct :
  tax_to_budget_level TaxType.property_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.federal_tax = BudgetLevel.federal_budget ∧
  tax_to_budget_level TaxType.profit_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.tax_subjects_RF = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.transport_collecting = BudgetLevel.subjects_RF_budget :=
by
  sorry

end tax_budget_level_correct_l358_358477


namespace tangent_parallel_to_AB_exists_a_l358_358566

variables {a : ℝ} (a_pos : a > 0)

def parabola (x : ℝ) : ℝ := a * x ^ 2

def line (x : ℝ) : ℝ := x + 2

-- Points of intersection
variables {x1 x2 : ℝ}
variables (h_intersection : parabola a x1 = line x1)
variables (h_intersection2 : parabola a x2 = line x2)

-- Midpoint M coordinates
def midpoint_x := (x1 + x2) / 2
def midpoint_y := (parabola a x1 + parabola a x2) / 2

-- Point N coordinates (intersection of the parabola with vertical line through M)
def point_N_x := midpoint_x
def point_N_y := parabola a point_N_x

-- Formulating the assertions
theorem tangent_parallel_to_AB : 
  let k1 := (2 * a * point_N_x) -- slope of the tangent line at N
  let k2 := 1 -- slope of line AB (y = x + 2 has slope 1)
  in k1 = k2 := sorry

theorem exists_a : 
  ∃ a : ℝ, a > 0 ∧ let x1 := some_x1_proof in -- provide the proof of x1 existence
                  let x2 := some_x2_proof in -- provide the proof of x2 existence
                  let Mx := (x1 + x2) / 2 in
                  let My := (parabola a x1 + parabola a x2) / 2 in
                  let Nx := Mx in
                  let Ny := parabola a Nx in
                  (Nx - x1) * (Nx - x2) + (Ny - parabola a x1) * (Ny - parabola a x2) = 0 :=
begin
  -- Proof required to verify the existence and value of a
  sorry
end

end tangent_parallel_to_AB_exists_a_l358_358566


namespace sequence_length_div_by_four_l358_358358

theorem sequence_length_div_by_four (a : ℕ) (h0 : a = 11664) (H : ∀ n, a = (4 ^ n) * b → b ≠ 0 ∧ n ≤ 3) : 
  ∃ n, n + 1 = 4 :=
by
  sorry

end sequence_length_div_by_four_l358_358358


namespace equation_of_perpendicular_line_passing_through_point_l358_358349

theorem equation_of_perpendicular_line_passing_through_point :
  ∃ c : ℝ, ∃ (l : ℝ → ℝ → Prop), 
  (∀ x y : ℝ, l x y ↔ 3 * x + 2 * y + c = 0) 
  ∧ l (-1) 2 
  ∧ (∃ m : ℝ, ∀ x y : ℝ, (2 * x - 3 * y + 8 = 0) ↔ (m = -3/2)) 
  ∧ (decidual.equal l (λ (x y : ℝ), 3 * x + 2 * y - 1 = 0)) := sorry

end equation_of_perpendicular_line_passing_through_point_l358_358349


namespace necessary_but_not_sufficient_l358_358350

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

-- State the necessary but not sufficient condition proof statement
theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x y : ℝ, quadratic_eq a x = 0 ∧ quadratic_eq a y = 0 ∧ x > 0 ∧ y < 0) → a < 1 :=
sorry

end necessary_but_not_sufficient_l358_358350


namespace triangle_area_l358_358274

-- Define the conditions as given in the original problem
def angle_sum (A B C : ℝ) := A + B + C = real.pi

def arithmetic_sequence (A B C : ℝ) := A + C = 2 * B

def law_of_sines (a b : ℝ) (A B : ℝ) :=
  a / real.sin A = b / real.sin B

-- Define the sides and angles
def side_a := 1
def side_b := real.sqrt 3

-- Define the problem statement to prove the area of the triangle
theorem triangle_area (A B C : ℝ)
  (h_angle_sum : angle_sum A B C)
  (h_arith_sequence : arithmetic_sequence A B C)
  (h_sine_a : real.sin A = 0.5)
  (h_side_a : side_a = 1)
  (h_side_b : side_b = real.sqrt 3) :
  (1 / 2) * side_a * side_b = real.sqrt 3 / 2 := 
sorry

end triangle_area_l358_358274


namespace triangular_prism_edges_after_cut_l358_358841

theorem triangular_prism_edges_after_cut :
  let original_edges := 9
  let vertices := 6
  let new_edges_per_vertex := 3
  let total_new_edges := new_edges_per_vertex * vertices
  in original_edges + total_new_edges = 27 :=
by
  sorry

end triangular_prism_edges_after_cut_l358_358841


namespace prove_angle_BSC_eq_2_angle_BAC_l358_358795

-- Definitions and conditions
variables {A B C D E S : Type*} [linear_ordered_field A] [metric_space A]
variables {a b c d e s : A} -- Points in space
variables {AB AC BC : A} -- Lengths of segments
variable (angle : A → A → A → A)
variables (a1 a2 : A) -- Angles

-- Conditions
axiom triangle_ABC (h: AB ≤ AC)
axiom point_D (bc_circumcircle_not_containing_a : Type*) 
(point D_on_arc_not_containing_A : (BC → bc_circumcircle_not_containing_a))
(point C_on_arc_not_containing_A : (BC → C))
axiom point_E (E_on_BC : Type*) (E : (BC → E_on_BC))
axiom angle_BAD_eq_angle_CAE_lt_half_angle_BAC : (angle A B D = angle A C E ∧ angle A B D < a1 / 2)
axiom midpoint_S (M : Type*) (midpoint_S_of_AD : (AD → M))
axiom angle_ADE_eq_angle_ABC_minus_angle_ACB : angle A D E = angle A B C - angle A C B

-- Theorem to prove
theorem prove_angle_BSC_eq_2_angle_BAC :
  (AB ≤ AC) → 
  D_on_arc_not_containing_A → 
  E_on_BC → 
  (angle A B D = angle A C E ∧ angle A B D < a1 / 2) →
  midpoint_S_of_AD → 
  (angle A D E = angle A B C - angle A C B) →
  angle B S C = 2 * angle A B C :=
begin
  sorry
end

end prove_angle_BSC_eq_2_angle_BAC_l358_358795


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l358_358629

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : x < -1) : 2 * x ^ 2 + x - 1 > 0 :=
by sorry

theorem not_necessary_condition (h2 : 2 * x ^ 2 + x - 1 > 0) : x > 1/2 ∨ x < -1 :=
by sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l358_358629


namespace percentage_both_questions_correct_l358_358410

-- Definitions for the conditions in the problem
def percentage_first_question_correct := 85
def percentage_second_question_correct := 65
def percentage_neither_question_correct := 5
def percentage_one_or_more_questions_correct := 100 - percentage_neither_question_correct

-- Theorem stating that 55 percent answered both questions correctly
theorem percentage_both_questions_correct :
  percentage_first_question_correct + percentage_second_question_correct - percentage_one_or_more_questions_correct = 55 :=
by
  sorry

end percentage_both_questions_correct_l358_358410


namespace participant_can_compare_with_median_l358_358247

variable (Scores : Fin 19 → ℝ) (i : Fin 19)
-- Ensure uniqueness of scores
def unique_scores : Prop := ∀ i j, i ≠ j → Scores i ≠ Scores j

-- Define the participant's own score
variable (participant_score : ℝ)

-- Hypothesis that the participant's score is in the Scores set
def in_scores : Prop := ∃ (j : Fin 19), Scores j = participant_score

-- Define the median
def median (a : Fin 19 → ℝ) : ℝ := 
  let sorted_scores := List.sort (Fin.val ∘ a).toList in 
  sorted_scores[9] -- Median in 0-based index for 19 (10th element).

theorem participant_can_compare_with_median
  (h : unique_scores Scores) (h₁ : in_scores Scores participant_score) : 
  ∃ j : Fin 19, Scores j > median Scores ↔ participant_score > median Scores := 
sorry

end participant_can_compare_with_median_l358_358247


namespace part1_part2_l358_358945

variables {x : ℝ} {a b : ℝ × ℝ}

def a := (-x, x)
def b := (2x + 3, 1)

theorem part1 (h : a.1 * b.1 + a.2 * b.2 = 0) : x = -1 :=
sorry

theorem part2 (h : ¬ ∃ k : ℝ, a = k • b) : ‖(a.1 - b.1, a.2 - b.2)‖ = 3 * real.sqrt 2 :=
sorry

end part1_part2_l358_358945


namespace measure_smaller_angle_east_northwest_l358_358817

/-- A mathematical structure for a circle with 12 rays forming congruent central angles. -/
structure CircleWithRays where
  rays : Finset (Fin 12)  -- There are 12 rays
  congruent_angles : ∀ i, i ∈ rays

/-- The measure of the central angle formed by each ray is 30 degrees (since 360/12 = 30). -/
def central_angle_measure : ℝ := 30

/-- The measure of the smaller angle formed between the ray pointing East and the ray pointing Northwest is 150 degrees. -/
theorem measure_smaller_angle_east_northwest (c : CircleWithRays) : 
  ∃ angle : ℝ, angle = 150 := by
  sorry

end measure_smaller_angle_east_northwest_l358_358817


namespace cos_alpha_minus_sin_alpha_l358_358558

theorem cos_alpha_minus_sin_alpha 
  (f : ℝ → ℝ) (α : ℝ)
  (h1 : ∀ x, f(x) = Real.sin (3 * x + Real.pi / 4))
  (h2 : α > Real.pi / 2 ∧ α < Real.pi)
  (h3 : f(α / 3) = 4 / 5 * Real.cos (α + Real.pi / 4) * Real.cos (2 * α)) :
  (Real.cos α - Real.sin α = - Real.sqrt 5 / 2) ∨ (Real.cos α - Real.sin α = - Real.sqrt 2) := 
sorry

end cos_alpha_minus_sin_alpha_l358_358558


namespace jess_double_cards_l358_358316

theorem jess_double_cards (rob_total_cards jess_doubles : ℕ) 
    (one_third_rob_cards_doubles : rob_total_cards / 3 = rob_total_cards / 3)
    (jess_times_rob_doubles : jess_doubles = 5 * (rob_total_cards / 3)) :
    rob_total_cards = 24 → jess_doubles = 40 :=
  by
  sorry

end jess_double_cards_l358_358316


namespace exists_k_fractional_parts_in_interval_l358_358184

theorem exists_k_fractional_parts_in_interval 
  (a b c : ℕ) (h1 : b > 2 * a) (h2 : c > 2 * b) : 
  ∃ k : ℝ, (frac (k * a) > 1 / 3) ∧ (frac (k * a) ≤ 2 / 3) ∧ 
           (frac (k * b) > 1 / 3) ∧ (frac (k * b) ≤ 2 / 3) ∧ 
           (frac (k * c) > 1 / 3) ∧ (frac (k * c) ≤ 2 / 3) := 
sorry

end exists_k_fractional_parts_in_interval_l358_358184


namespace variance_of_scores_l358_358088

def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (List.sum (List.map (λ x => (x - m) ^ 2) xs)) / (xs.length)

theorem variance_of_scores : variance scores = 40 / 9 :=
by
  sorry

end variance_of_scores_l358_358088


namespace total_fish_in_pond_equals_1500_l358_358786

-- Define initial conditions
variables (TotalTaggedFish : ℕ) (SampleSize : ℕ) (TaggedInSample : ℕ) (TotalFishInPond : ℕ)

-- Initial values as given in the problem
def initial_conditions :=
  (TotalTaggedFish = 60) ∧
  (SampleSize = 50) ∧
  (TaggedInSample = 2)

-- Proportion condition: The percent of tagged fish in sample approximates percent of tagged fish in pond
def proportion_condition (TotalFishInPond : ℕ) :=
  (TaggedInSample : ℚ) / (SampleSize : ℚ) = (TotalTaggedFish : ℚ) / (TotalFishInPond : ℚ)

-- The statement to prove the total number of fish in the pond is 1500, given the conditions
theorem total_fish_in_pond_equals_1500 (h : initial_conditions) : ∃ N : ℕ, proportion_condition N ∧ N = 1500 :=
begin
  -- sorry, the detailed proof construction would go here
  sorry
end

end total_fish_in_pond_equals_1500_l358_358786


namespace coin_flip_probability_l358_358376

theorem coin_flip_probability (n : ℕ) : 
  let p : ℕ → ℝ := λ n, 1 / 3 * (2 + (-1/2) ^ n) in 
  True :=
sorry

end coin_flip_probability_l358_358376


namespace sin_theta_value_l358_358533

theorem sin_theta_value (θ : ℝ) 
  (h : sin (π / 4 - θ / 2) = 2 / 3) : sin θ = 1 / 9 := by
  sorry

end sin_theta_value_l358_358533


namespace range_of_m_value_of_m_l358_358967

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end range_of_m_value_of_m_l358_358967


namespace eating_cereal_time_l358_358306

theorem eating_cereal_time:
  let rate_Fat := 1 / 20 in
  let rate_Thin := 1 / 30 in
  let rate_Medium := 1 / 40 in
  let combined_rate := rate_Fat + rate_Thin + rate_Medium in
  let total_cereal := 4 in
  let time := total_cereal / combined_rate in
  time = 480 / 13 :=
by
  sorry

end eating_cereal_time_l358_358306


namespace power_div_multiply_l358_358112

theorem power_div_multiply (a b c : ℕ) (h₁ : a = 8^3) (h₂ : b = 8^2) (h₃ : c = 2^{10}) :
  (a / b) * c = 8192 := by
  sorry

end power_div_multiply_l358_358112


namespace solar_eclipse_coverage_l358_358879

theorem solar_eclipse_coverage (R : ℝ) (O1 O2 : Point)
    (h1 : O2 ∈ circle (R : ℝ) O1)
    (h2 : dist O1 O2 = R) :
    let sector_area := (1 / 3) * π * R^2 in
    let segment_area := 2 * ((π * R^2 / 6) - (√3 * R^2 / 4)) in
    (sector_area + segment_area) / (π * R^2) ≈ 0.39 := sorry

end solar_eclipse_coverage_l358_358879


namespace sum_of_digits_base2_315_l358_358764

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358764


namespace polynomial_identity_l358_358119

theorem polynomial_identity :
  ∀ (A B C D : ℝ), 
  (g : ℝ → ℝ) →
  (g = λ x, A * x^3 + B * x^2 + C * x + D) →
  (g 3 = 1) →
  A = -1 →
  B = 1 →
  C = -1 →
  D = 1 →
  12 * A - 6 * B + 3 * C - D = -22 :=
by
  intros A B C D g g_def g_at_3 A_val B_val C_val D_val
  sorry

end polynomial_identity_l358_358119


namespace horner_operations_count_l358_358383

def horner_operations (x : ℝ) : ℕ :=
  let f := λ x, ((((3 * x + 4) * x + 5) * x + 6) * x + 7) * x + 8
  f x -- To ensure the function is used and calculations performed

theorem horner_operations_count :
  let f := λ x, ((((3 * x + 4) * x + 5) * x + 6) * x + 7) * x + 8
  let num_operations := 12
  (horner_operations 0.4 = num_operations) := 
sorry

end horner_operations_count_l358_358383


namespace sum_of_digits_in_binary_representation_of_315_l358_358768

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l358_358768


namespace modulo_inverse_5_mod_23_l358_358909

/-- The modular inverse of 5 modulo 23 is 14. -/
theorem modulo_inverse_5_mod_23 : ∃ b : ℕ, b ∈ set.Ico 0 23 ∧ (5 * b) % 23 = 1 :=
  by
  use 14
  split
  · exact Nat.le_of_lt (by norm_num : (14 < 23))
  · change (5 * 14) % 23 = 1
    norm_num
    sorry

end modulo_inverse_5_mod_23_l358_358909


namespace adam_simon_distance_apart_l358_358095

theorem adam_simon_distance_apart
  (t : ℕ) : 
  (∀ (a_dist s_dist total_dist : ℝ),
    a_dist = 12 * t ∧ s_dist = 9 * t ∧ 
    total_dist = real.sqrt (a_dist ^ 2 + s_dist ^ 2) ∧ 
    total_dist = 90) → 
  t = 6 :=
by 
  sorry

end adam_simon_distance_apart_l358_358095


namespace proof_equivalent_answer_l358_358395

-- Definitions and properties used
variables {a b c : ℝ}

-- Distributive Property
def distributive_property (a b c : ℝ) : Prop := a * (b + c) = a * b + a * c

-- Exponentiation Property
def exponent_property (a b c : ℝ) : Prop := a ^ (b + c) = (a ^ b) * (a ^ c)

-- Logarithm Sum Property (known to be false in context)
def log_sum_property (b c : ℝ) : Prop := log (b + c) = log b + log c

-- Logarithm Division Property (known to be false in context)
def log_div_property (b c : ℝ) : Prop := log b / log c = log b + log c

-- Exponentiation Distributive Property (known to be false in context)
def exp_distributive_property (a b c : ℝ) : Prop := (a * b) ^ c = a ^ c + b ^ c

-- Theorem to prove the problem is equivalent to the solution
theorem proof_equivalent_answer : 
  distributive_property a b c ∧ 
  exponent_property a b c ∧ 
  ¬log_sum_property b c ∧ 
  ¬log_div_property b c ∧ 
  ¬exp_distributive_property a b c :=
by
  -- The proof is omitted
  sorry

end proof_equivalent_answer_l358_358395


namespace solved_smallest_a_l358_358925

noncomputable def smallest_a (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  if h : (∃ a > 0, 
          (sqrt a / cos θ + sqrt a / sin θ > 1) ∧
          (∃ x ∈ set.Icc (1 - sqrt a / sin θ) (sqrt a / cos θ), 
            ((1 - x) * sin θ - sqrt (a - x^2 * cos θ^2))^2 + (x * cos θ - sqrt (a - (1 - x)^2 * sin θ^2))^2 <= a)) 
    then classical.some h else 0

theorem solved_smallest_a (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  smallest_a θ hθ = (sin² θ * cos² θ) / (1 + sqrt 3 * sin θ * cos θ) :=
by
  sorry

end solved_smallest_a_l358_358925


namespace product_of_solutions_l358_358501

theorem product_of_solutions (t : ℝ) (h : t^2 = 64) : t * (-t) = -64 :=
sorry

end product_of_solutions_l358_358501


namespace condition_on_a_and_b_l358_358224

theorem condition_on_a_and_b (a b p q : ℝ) 
    (h1 : (∀ x : ℝ, (x + a) * (x + b) = x^2 + p * x + q))
    (h2 : p > 0)
    (h3 : q < 0) :
    (a < 0 ∧ b > 0 ∧ b > -a) ∨ (a > 0 ∧ b < 0 ∧ a > -b) :=
by
  sorry

end condition_on_a_and_b_l358_358224


namespace sum_of_squares_distances_constant_l358_358938

theorem sum_of_squares_distances_constant {P A B C : Point} (h_circle : is_circle O r)
  (h_triangle : is_equilateral_triangle ℓ A B C O r)
  (h_P0 : is_on_plane P O)
  : ∃ k : ℝ, PA.to(A).dist^2 + PB.to(B).dist^2 + PC.to(C).dist^2 = k :=
sorry

end sum_of_squares_distances_constant_l358_358938


namespace pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l358_358091

section pencil_case_problem

variables (x m : ℕ)

-- Part 1: The cost prices of each $A$ type and $B$ type pencil cases.
def cost_price_A (x : ℕ) : Prop := 
  (800 : ℝ) / x = (1000 : ℝ) / (x + 2)

-- Part 2.1: Maximum quantity of $B$ type pencil cases.
def max_quantity_B (m : ℕ) : Prop := 
  3 * m - 50 + m ≤ 910

-- Part 2.2: Number of different scenarios for purchasing the pencil cases.
def profit_condition (m : ℕ) : Prop := 
  4 * (3 * m - 50) + 5 * m > 3795

theorem pencil_case_solution_part1 (hA : cost_price_A x) : 
  x = 8 := 
sorry

theorem pencil_case_solution_part2_1 (hB : max_quantity_B m) : 
  m ≤ 240 := 
sorry

theorem pencil_case_solution_part2_2 (hB : max_quantity_B m) (hp : profit_condition m) : 
  236 ≤ m ∧ m ≤ 240 := 
sorry

end pencil_case_problem

end pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l358_358091


namespace gcd_36_54_l358_358734

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l358_358734


namespace infinite_t_with_equal_digit_sum_l358_358315

def digit_sum (n : ℕ) : ℕ :=
  -- Function to calculate the sum of the digits of a natural number n
  (n.digits).sum

theorem infinite_t_with_equal_digit_sum (k : ℕ) :
  ∃∞ t : ℕ, (∀ d ∈ t.digits, d ≠ 0) ∧ digit_sum t = digit_sum (k * t) :=
by sorry

end infinite_t_with_equal_digit_sum_l358_358315


namespace relationship_among_abc_l358_358932

theorem relationship_among_abc (a b c : ℝ) (h1 : a = 2^(-2)) (h2 : b = 1) (h3 : c = (-1)^3) : c < a ∧ a < b := by
  rw [h1, h2, h3]
  sorry

end relationship_among_abc_l358_358932


namespace correct_statements_l358_358340

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l358_358340


namespace problem1_problem2_problem3_l358_358551

noncomputable section

open Classical

variable {X : Type}

def P (E : Set X) : ℝ := 
  -- Probability function. P(X = x) should be defined based on the problem.
  sorry

def a : ℝ := 1 / 15

theorem problem1 (h : ∑ k in ({1, 2, 3, 4, 5} : Set ℕ), P ({ x | x = k / 5 }) = 1) : 
  a = 1 / 15 :=
sorry

theorem problem2 {X : ℝ} (h : ∀ k : ℕ, k ∈ {1, 2, 3, 4, 5} → P ({x | x = k / 5}) = k / 15) : 
  P ({x | x ≥ 3 / 5}) = 4 / 5 :=
sorry

theorem problem3 {X : ℝ} (h : ∀ k : ℕ, k ∈ {1, 2, 3, 4, 5} → P ({x | x = k / 5}) = k / 15) : 
  P ({x | 1 / 10 < x ∧ x < 7 / 10}) = 2 / 5 :=
sorry

end problem1_problem2_problem3_l358_358551


namespace range_of_m_for_inequality_l358_358796

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 := 
sorry

end range_of_m_for_inequality_l358_358796


namespace min_fraction_sum_l358_358227

theorem min_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  ∃ m, (∀ (a b : ℝ), (0 < a) → (0 < b) → (a + b = 2) → (m ≤ (1 / a + 9 / b))) ∧ m = 8 := 
by 
  use 8
  intros x y hx hy hxy
  sorry

end min_fraction_sum_l358_358227


namespace problem_statement_l358_358341

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l358_358341


namespace abc_not_all_positive_l358_358163

theorem abc_not_all_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ac > 0) (h3 : abc > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := 
by 
sorry

end abc_not_all_positive_l358_358163


namespace shop_discount_l358_358829

theorem shop_discount
  (initial_discount : ℝ := 0.25)
  (second_discount : ℝ := 0.15)
  (third_discount : ℝ := 0.1)
  (claimed_total_discount : ℝ := 0.50) :
  (let first_price := 1 - initial_discount,
       second_price := first_price * (1 - second_discount),
       final_price := second_price * (1 - third_discount),
       actual_total_discount := 1 - final_price,
       difference := claimed_total_discount - actual_total_discount
   in final_price = 0.57375 ∧ actual_total_discount = 0.42625 ∧ difference = 0.07375) :=
by
  sorry

end shop_discount_l358_358829


namespace tourists_count_l358_358377

theorem tourists_count :
  ∃ (n : ℕ), (1 / 2 * n + 1 / 3 * n + 1 / 4 * n = 39) :=
by
  use 36
  sorry

end tourists_count_l358_358377


namespace germination_percentage_l358_358409

theorem germination_percentage (s1 s2 t1 t2 : ℕ) (p1 p2 : ℝ)
  (h1 : s1 = 300) (h2 : s2 = 200)
  (h3 : p1 = 0.20) (h4 : p2 = 0.35)
  (h5 : t1 = s1 * 0.20) (h6 : t2 = s2 * 0.35):
  (t1 + t2) / (s1 + s2) * 100 = 26 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  -- Additional algebraic steps to simplify can be added here, if desired
  sorry

end germination_percentage_l358_358409


namespace count_seven_appearances_in_range_l358_358277

theorem count_seven_appearances_in_range (count_seven : ℕ → ℕ) :
    (∀ n, count_seven n = (if (n / 10 = 7) ∨ (n % 10 = 7) then 1 else 0)) →
    (Finset.range 190).sum (λ n, count_seven (n + 10)) = 39 :=
by
  intro h
  sorry

end count_seven_appearances_in_range_l358_358277


namespace solve_equation_l358_358324

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l358_358324


namespace count_distinct_integer_sums_of_special_fractions_l358_358125

-- Define the special fraction condition
def is_special_fraction (a b : ℕ) : Prop :=
  a + b = 18 ∧ a > 0 ∧ b > 0

-- Define the set of special fractions
def special_fractions : set (ℚ) :=
  {q | ∃ a b : ℕ, is_special_fraction a b ∧ q = (a : ℚ) / (b : ℚ)}

-- Define the set of sums of two special fractions
def sum_of_two_special_fractions : set ℚ :=
  {s | ∃ x y ∈ special_fractions, s = x + y}

-- Define the set of sums that are integers
def integer_sums : set ℤ :=
  {n | (n : ℚ) ∈ sum_of_two_special_fractions}

-- The problem statement in Lean 4
theorem count_distinct_integer_sums_of_special_fractions : 
  finset.card (finset.image (coe : ℤ → ℚ) (finset.filter (λ x, x ∈ integer_sums) (finset.range 100))) = 5 :=
sorry

end count_distinct_integer_sums_of_special_fractions_l358_358125


namespace linear_function_quadrants_l358_358071

theorem linear_function_quadrants : 
  ∀ (x y : ℝ), y = -5 * x + 3 
  → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by 
  intro x y h
  sorry

end linear_function_quadrants_l358_358071


namespace probability_of_different_value_and_suit_l358_358738

theorem probability_of_different_value_and_suit :
  let total_cards := 52
  let first_card_choices := 52
  let remaining_cards := 51
  let different_suits := 3
  let different_values := 12
  let favorable_outcomes := different_suits * different_values
  let total_outcomes := remaining_cards
  let probability := favorable_outcomes / total_outcomes
  probability = 12 / 17 := 
by
  sorry

end probability_of_different_value_and_suit_l358_358738


namespace problem_statement_l358_358235

variables {Point : Type} {Line Plane : Type}
variable [IncidenceGeometry Point Line Plane]

-- Define the lines and planes involved
variables (l m : Line) (α β γ : Plane)

-- Defining the conditions
axiom h1 : β ∩ γ = l
axiom h2 : l ∥ α
axiom h3 : m ⊆ α
axiom h4 : m ⊥ γ

-- Prove the required statements
theorem problem_statement : α ⊥ γ ∧ l ⊥ m :=
by {
  sorry
}

end problem_statement_l358_358235


namespace problem_statement_l358_358033

theorem problem_statement :
  (1296 ^ (log 40 / log 4)) ^ (1 / 4) = 6 ^ (log 40 / log 4) :=
by sorry

end problem_statement_l358_358033


namespace number_of_special_m_gons_correct_l358_358214

noncomputable def number_of_special_m_gons (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose n (m - 1) + Nat.choose (n + 1) (m - 1))

theorem number_of_special_m_gons_correct (m n : ℕ) (h1 : 4 < m) (h2 : m < n) :
  number_of_special_m_gons m n h1 h2 =
  (2 * n + 1) * (Nat.choose n (m - 1) + Nat.choose (n + 1) (m - 1)) :=
by
  unfold number_of_special_m_gons
  sorry

end number_of_special_m_gons_correct_l358_358214


namespace shortest_distance_proof_l358_358396

-- Given conditions definitions

-- Non-intersecting lines definition
def non_intersecting_lines (l1 l2 : Line) : Prop :=¬ intersect l1 l2

-- Perpendicular segment definition
def perp_segment_length (p : Point) (l : Line) : Length := 
  perpendicular_segment_length p l

-- Shortest distance definition
def shortest_distance (p1 p2 : Point) : LineSegment :=
  line_segment p1 p2

-- Perpendicular through point
def unique_perpendicular (p : Point) (l : Line) : Prop :=
  ∃! m : Line, perpendicular p m l

-- Proof problem definition
theorem shortest_distance_proof (p1 p2 : Point) :
  shortest_distance p1 p2 = line_segment p1 p2 := 
  by
    sorry

end shortest_distance_proof_l358_358396


namespace quadruples_positive_integers_l358_358486

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l358_358486


namespace longest_side_of_triangle_l358_358094

-- Define the vertices of the triangle
def A := (3 : ℤ, 3 : ℤ)
def B := (8 : ℤ, 9 : ℤ)
def C := (9 : ℤ, 3 : ℤ)

-- Define the distance function between two points
def dist (p1 p2 : ℤ × ℤ) := real.sqrt (((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) : ℝ)

-- Define the distances between each pair of vertices
def d_AB := dist A B
def d_BC := dist B C
def d_CA := dist C A

-- State the problem: Prove that the longest side of the triangle has length sqrt(61)
theorem longest_side_of_triangle :
  max d_AB (max d_BC d_CA) = real.sqrt 61 :=
by sorry

end longest_side_of_triangle_l358_358094


namespace max_value_of_quadratic_l358_358567

def quadratic_func (x : ℝ) : ℝ := -3 * (x - 2) ^ 2 - 3

theorem max_value_of_quadratic : 
  ∃ x : ℝ, quadratic_func x = -3 :=
by
  sorry

end max_value_of_quadratic_l358_358567


namespace intersection_of_A_and_B_range_of_a_l358_358870

open Set

namespace ProofProblem

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 ≤ x ∧ x < 3} := 
sorry

theorem range_of_a (a : ℝ) :
  (B ∪ C a) = C a → a ≤ 3 :=
sorry

end ProofProblem

end intersection_of_A_and_B_range_of_a_l358_358870


namespace custom_operation_example_l358_358471

-- Define the custom operation
def custom_operation (a b : ℕ) : ℕ := a * b + (a - b)

-- State the theorem
theorem custom_operation_example : custom_operation (custom_operation 3 2) 4 = 31 :=
by
  -- the proof will go here, but we skip it for now
  sorry

end custom_operation_example_l358_358471


namespace sum_of_digits_base_2_315_l358_358774

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358774


namespace proof_A_equals_B_find_area_of_triangle_l358_358595

namespace TriangleProof

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (h1 : a = b)
variable (h2 : A = B)
variable (h_angle_values : A = 7 * π / 24)
variable (h_a_value : a = sqrt 6)

theorem proof_A_equals_B (h : sin (A - B) = (a / (a + b)) * sin A * cos B - (b / (a + b)) * sin B * cos A) : A = B :=
by
  sorry

noncomputable def area_of_triangle (A B : ℝ) (a b : ℝ) : ℝ :=
  have h : B = A := by sorry
  let c := 2 * b * cos A
  1 / 2 * b * c * sin A

theorem find_area_of_triangle : area_of_triangle (7 * π / 24) (7 * π / 24) (sqrt 6) (sqrt 6) = (3 * (sqrt 2 + sqrt 6)) / 4 :=
by
  sorry

end TriangleProof

end proof_A_equals_B_find_area_of_triangle_l358_358595


namespace correct_geometric_statement_l358_358779

theorem correct_geometric_statement (P: Type) [NormedAddCommGroup P] [MetricSpace P] [NormedSpace ℝ P]
  (point : P) (line : Set P) (h : ∃ ! perp_line, perp_line ⊆ P ∧ perp_line ∩ line = ∅ ∧ perp_line ≠ ∅) :
  ∃ unique_perp_line : Set P, unique_perp_line ⊆ P ∧ unique_perp_line ∩ line = ∅ ∧ unique_perp_line ≠ ∅ :=
by
  exact h

end correct_geometric_statement_l358_358779


namespace exists_x_satisfying_equation_l358_358923

theorem exists_x_satisfying_equation :
  ∃ x : ℝ, (x = (-62 / 29)) ∧ (sqrt (7 * x + 1) / sqrt (4 * (x + 2) - 1) = 3) :=
by
  use -62 / 29
  split
  · rfl
  · sorry

end exists_x_satisfying_equation_l358_358923


namespace old_clock_144_minutes_slower_l358_358850

/-- Define the condition: the old clock hand overlap period -/
def hand_overlap_period_old_clock : ℝ := 66

/-- Define the standard hand overlap period using given rates -/
def hand_overlap_period_standard : ℝ := 360 / (6 - 0.5)

/-- Define the relative speed constant k -/
def k : ℝ := 66 / hand_overlap_period_standard

/-- Define the total minutes in 24 hours in the old clock -/
def total_minutes_old_clock : ℝ := 24 * 60 * (1 / k)

/-- Define the standard 24 hours in minutes -/
def total_minutes_standard : ℝ := 24 * 60

/-- Prove that the old clock is 144 minutes slower than the standard 24 hours -/
theorem old_clock_144_minutes_slower : total_minutes_standard - total_minutes_old_clock = 144 :=
by
  sorry

end old_clock_144_minutes_slower_l358_358850


namespace not_critical_point_for_inflection_l358_358661

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem not_critical_point_for_inflection :
  (∀ x0 : ℝ, deriv f x0 = 0 → ¬ is_critical_point f x0) :=
begin
  intros x0 h,
  have h_deriv : deriv f x0 = 0 := h,
  unfold is_critical_point,
  -- The reasoning that x_0 is a critical point is incorrect,
  -- since the definition of a critical point requires further criteria,
  -- such as a change in sign of the derivative around x_0
  sorry
end

end not_critical_point_for_inflection_l358_358661


namespace gcd_884_1071_l358_358018

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end gcd_884_1071_l358_358018


namespace kelly_apples_total_l358_358612

def initial_apples : ℕ := 56
def second_day_pick : ℕ := 105
def third_day_pick : ℕ := 84
def apples_eaten : ℕ := 23

theorem kelly_apples_total :
  initial_apples + second_day_pick + third_day_pick - apples_eaten = 222 := by
  sorry

end kelly_apples_total_l358_358612


namespace pet_insurance_cost_per_month_l358_358709

theorem pet_insurance_cost_per_month 
  (months : ℕ) (procedure_cost : ℝ) (coverage_percentage : ℝ) (savings : ℝ) (insurance_pays : ℝ) (monthly_cost : ℝ) :
  months = 24 →
  procedure_cost = 5000 →
  coverage_percentage = 0.8 →
  savings = 3520 →
  insurance_pays = procedure_cost * coverage_percentage →
  insurance_pays - savings = 480 →
  480 / months = 20 →
  monthly_cost = 20 :=
by
  intros,
  sorry

end pet_insurance_cost_per_month_l358_358709


namespace find_k_b_integral_xexp_l358_358516

noncomputable def f (k b x : ℝ) := (k * x + b) * Real.exp x
noncomputable def f' (k b x : ℝ) := (k * x + k + b) * Real.exp x

theorem find_k_b :
  (f 1 1 = (f' 1 (0 0 1) = 1))

-- theorem find_k_b:
-- ( ∀ f, y → y = e →  f (1 * e ** x) = 0 )  :=
-- sorry

theorem integral_xexp (k:ℝ) b:=
  ∀ x:ℝ,  int (x + fx - 1 * e ** x |= 1 -1 :=x - 1 → x *e  } =
 sorry



end find_k_b_integral_xexp_l358_358516


namespace log_equation_solution_l358_358987

theorem log_equation_solution (x : ℝ) (h : log 4 x + 2 * log 8 x = 7) : x = 64 :=
sorry

end log_equation_solution_l358_358987


namespace coordinates_of_M_are_3_3_l358_358187

theorem coordinates_of_M_are_3_3 (a : ℝ) (h : abs (2 - a) = abs (3 * a + 6)) :
  (2 - a = 3) ∧ (3 * a + 6 = 3) :=
begin
  sorry
end

end coordinates_of_M_are_3_3_l358_358187


namespace factory_correct_decision_prob_l358_358816

def prob_correct_decision (p : ℝ) : ℝ :=
  let prob_all_correct := p * p * p
  let prob_two_correct_one_incorrect := 3 * p * p * (1 - p)
  prob_all_correct + prob_two_correct_one_incorrect

theorem factory_correct_decision_prob : prob_correct_decision 0.8 = 0.896 :=
by
  sorry

end factory_correct_decision_prob_l358_358816


namespace correctAnswer_is_B_l358_358337

noncomputable def P1_false : Prop :=
  ¬(∃ n0 : ℕ, n0^2 ≤ 2^n0)

noncomputable def P2_true : Prop :=
  ∀ m n : ℤ, let a := (m, 1); let b := (1, -n) in (a.1 * b.1 + a.2 * b.2 = 0 ↔ m = n)

noncomputable def P3_true : Prop :=
  ∀ A B : ℝ, A > B → sin A > sin B → A ≤ B → sin A ≤ sin B

noncomputable def P4_false : Prop :=
  ¬(¬(p ∧ q) → p)

def correctAnswer (P1_false : Prop) (P2_true : Prop) (P3_true : Prop) (P4_false : Prop) : Prop :=
  (¬P1_false) ∧ P2_true ∧ P3_true ∧ (¬P4_false)

theorem correctAnswer_is_B : correctAnswer P1_false P2_true P3_true P4_false = true :=
  by
    sorry

end correctAnswer_is_B_l358_358337


namespace fixed_point_f_l358_358338

theorem fixed_point_f (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  let f : ℝ → ℝ := λ x, 3 + a^(x - 1) in 
  f 1 = 4 :=
by
  simp [f]
  sorry

end fixed_point_f_l358_358338


namespace hyperbola_eccentricity_is_2_l358_358948

noncomputable theory

-- Condition: Definitions based on the problem description
variables (a b c : ℝ) (e : ℝ)
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def focus1 : ℝ × ℝ := (-c, 0)
def focus2 : ℝ × ℝ := (c, 0)
def asymptote (x y : ℝ) : Prop := y = (b / a) * x
def symmetric_point_on_circle (p : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop := 
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Question/Proposition to prove: The eccentricity of the hyperbola is 2
theorem hyperbola_eccentricity_is_2
  (h_hyperbola : ∀ x y, hyperbola a b x y)
  (h_focus1 : focus1 c = (-c, 0))
  (h_focus2 : focus2 c = (c, 0))
  (h_asymptote_distance : (c * b) / (real.sqrt(b^2 + a^2)) = b)
  (h_symmetric : symmetric_point_on_circle (c, 0) (-c, 0) (real.abs c)) :
  e = 2 :=
sorry

end hyperbola_eccentricity_is_2_l358_358948


namespace inequality_of_negatives_l358_358181

theorem inequality_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_of_negatives_l358_358181


namespace necessary_but_not_sufficient_l358_358054

-- Definitions from conditions
def abs_gt_2 (x : ℝ) : Prop := |x| > 2
def x_lt_neg_2 (x : ℝ) : Prop := x < -2

-- Statement to prove
theorem necessary_but_not_sufficient : 
  ∀ x : ℝ, (abs_gt_2 x → x_lt_neg_2 x) ∧ (¬(x_lt_neg_2 x → abs_gt_2 x)) := 
by 
  sorry

end necessary_but_not_sufficient_l358_358054


namespace cricket_team_members_eq_11_l358_358701

-- Definitions based on conditions:
def captain_age : ℕ := 26
def wicket_keeper_age : ℕ := 31
def avg_age_whole_team : ℕ := 24
def avg_age_remaining_players : ℕ := 23

-- Definition of n based on the problem conditions
def number_of_members (n : ℕ) : Prop :=
  n * avg_age_whole_team = (n - 2) * avg_age_remaining_players + (captain_age + wicket_keeper_age)

-- The proof statement:
theorem cricket_team_members_eq_11 : ∃ n, number_of_members n ∧ n = 11 := 
by
  use 11
  unfold number_of_members
  sorry

end cricket_team_members_eq_11_l358_358701


namespace inequality_a_gt_b_gt_c_l358_358289

-- Define the variables a, b, c with the given conditions
def a : ℝ := 2 ^ 0.6
def b : ℝ := Real.log 3 / Real.log π
def c : ℝ := Real.log (Real.sin 2) / Real.log 2

-- State the main theorem
theorem inequality_a_gt_b_gt_c : a > b ∧ b > c := by
  sorry

end inequality_a_gt_b_gt_c_l358_358289


namespace a_le_3n_l358_358626

-- Define the divisor function d(n)
def d (n : Nat) : Nat :=
  if n = 0 then 1
  else (List.range n).count (λ k => k > 0 ∧ n % k = 0) + 1

-- Define the sequence a_n using recursion
def a : ℕ → ℕ 
| 0 => 1
| n + 1 => (List.range (n + 1)).foldr (λ i acc => acc + (d^[i + 1] a i)) 0

-- The theorem to be proven
theorem a_le_3n (n : ℕ) : 1 ≤ n → a n ≤ 3 * n := by
  sorry

end a_le_3n_l358_358626


namespace sum_of_digits_base_2_315_l358_358772

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l358_358772


namespace minimal_abs_difference_l358_358228

theorem minimal_abs_difference (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : a * b - 8 * a + 7 * b = 395) : 
  |a - b| = 15 :=
sorry

end minimal_abs_difference_l358_358228


namespace hyperbola_eccentricity_l358_358348

-- Define the conditions given in the problem.
variables {a b c : ℝ} (h1 : a > 0) (h2 : b > 0)
-- Define the equation of the hyperbola and the conditions on foci and asymptotes.
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Assume the relationship between a, b, and c.
def c_eq_two_a : Prop := c = 2 * a
def eccentricity (c a : ℝ) : ℝ := c / a

-- The proof problem to show the eccentricity.
theorem hyperbola_eccentricity (h1 : a > 0) (h2 : b > 0) (h3 : c = 2 * a) : eccentricity c a = 2 :=
by
  sorry

end hyperbola_eccentricity_l358_358348


namespace daya_percentage_more_than_emma_l358_358805

variable (Emma Daya Jeff Brenda : ℕ)

-- Conditions
def emma_has_8 : Emma = 8 := sorry
def brenda_has_8 : Brenda = 8 := sorry
def brenda_has_4_more_than_jeff : Brenda = Jeff + 4 := sorry
def jeff_has_2_5_of_daya : Jeff = 2 * Daya / 5 := sorry

-- Proof statement
theorem daya_percentage_more_than_emma :
  let Daya := 10 in -- Daya's money derived from solution
  Daya = 8 * 5 / 4 →
  Daya - Emma = 2 →
  ((Daya - Emma).toRat / Emma.toRat) * 100 = 25 :=
by
  sorry

end daya_percentage_more_than_emma_l358_358805


namespace sandwich_cost_l358_358927

def cost_of_bagel : ℝ := 0.95
def cost_of_orange_juice : ℝ := 0.85
def cost_of_milk : ℝ := 1.15
def cost_of_breakfast : ℝ := cost_of_bagel + cost_of_orange_juice
def cost_difference : ℝ := 4

theorem sandwich_cost : ∃ S : ℝ, S = (cost_of_breakfast + cost_difference) - cost_of_milk :=
by
  use (cost_of_breakfast + cost_difference) - cost_of_milk
  split
  sorry

end sandwich_cost_l358_358927


namespace find_f_2013_l358_358126

noncomputable def f : ℝ → ℝ := sorry
axiom functional_eq : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2
axiom f_1_ne_0 : f 1 ≠ 0

theorem find_f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 :=
sorry

end find_f_2013_l358_358126


namespace non_intersecting_paths_l358_358944

def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

theorem non_intersecting_paths {m n p q : ℕ}
  (h_pm : p < m) (h_qn : q < n) :
  let S := binom (m + n) m * binom (m + q - p) q - binom (m + q) m * binom (m + n - p) n in
  S =
    (nat.choose (m + n) m) * (nat.choose (m + q - p) q) -
    (nat.choose (m + q) m) * (nat.choose (m + n - p) n) :=
by {
  sorry
}

end non_intersecting_paths_l358_358944


namespace license_plates_count_l358_358976

theorem license_plates_count :
  (20 * 6 * 20 * 10 * 26 = 624000) :=
by
  sorry

end license_plates_count_l358_358976


namespace liam_annual_income_l358_358363

theorem liam_annual_income (q : ℝ) (I : ℝ) (T : ℝ) 
  (h1 : T = (q + 0.5) * 0.01 * I) 
  (h2 : I > 50000) 
  (h3 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000)) : 
  I = 56000 :=
by
  sorry

end liam_annual_income_l358_358363


namespace inequalities_correct_l358_358988

variable (a b c : ℝ)

theorem inequalities_correct (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  ( (c / a > c / b) ∧
    ¬(ln (a + c) > ln (b + c)) ∧
    ((a - c) ^ c < (b - c) ^ c) ∧
    (b * exp a > a * exp b) ) :=
by
  sorry

end inequalities_correct_l358_358988


namespace problem_correct_options_l358_358961

-- Define the function f(x)
def f (ω φ x : ℝ) : ℝ := sqrt 3 * sin (ω * x + φ) - cos (ω * x + φ)

-- Define the conditions
variables {ω φ : ℝ} (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (h_symmetry : ∀ x, f ω φ (x + π / ω) = f ω φ x)

-- Theorem stating the correct interpretations
theorem problem_correct_options (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_increasing : ∀ ⦃a b : ℝ⦄, 0 ≤ a → a < b → b ≤ π / 6 → f ω φ a ≤ f ω φ b)
  (h_sym_center : f ω φ (-π / 12) = 0) :
  (ω = 2) ∧ (φ = 2 * π / 3) ∧ (φ ≤ π / 3) ∧ False :=
begin
  sorry
end

end problem_correct_options_l358_358961


namespace count_functions_with_given_range_l358_358998

noncomputable def num_functions_with_range : ℕ :=
  let D := {x : ℝ // x^2 ∈ {0, 1, 2, 3, 4, 5}};
  243

theorem count_functions_with_given_range :
  ∃ (f : ℝ → ℝ) (D : Set ℝ), 
    (∀ x ∈ D, f x = x^2) ∧ (Set.range f = {0, 1, 2, 3, 4, 5}) ∧ num_functions_with_range = 243 :=
begin
  use (λ x => x^2),
  use {0, 1, -1, sqrt 2, -sqrt 2, sqrt 3, -sqrt 3, 2, -2, sqrt 5, -sqrt 5},
  split,
  { intros x hx,
    simp, },
  split,
  { ext y,
    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff],
    split; intro h,
    { rcases h with ⟨x, rfl⟩,
      finish },
    { finish }
  },
  { exact rfl }
end

end count_functions_with_given_range_l358_358998


namespace part1_part2_l358_358570

-- Define the vectors in Lean
def a : ℝ × ℝ := (1, 2)
def b (λ : ℝ) : ℝ × ℝ := (2, λ)
def c : ℝ × ℝ := (-3, 2)

-- Statement 1: Proving the value of λ when a is parallel to b
theorem part1 (λ : ℝ) : (∀ (v : ℝ × ℝ), v = a → λ = 4) :=
sorry

-- Statement 2: Proving the value of k when k * a + c is perpendicular to a - 2 * c
theorem part2 (k : ℝ) : (∀ (v w : ℝ × ℝ), v = (k - 3, 2 * k + 2) → w = (7, -2) → k = 25 / 3) :=
sorry

end part1_part2_l358_358570


namespace scientific_notation_of_smallest_transistor_l358_358122

theorem scientific_notation_of_smallest_transistor :
  (0.00000004 : ℝ) = 4 * 10^(-8) := 
sorry

end scientific_notation_of_smallest_transistor_l358_358122


namespace decimal_to_binary_addition_l358_358468

theorem decimal_to_binary_addition : 
  let n1 := 45
  let n2 := 3
  let b1 := 101101  -- binary of 45
  let b2 := 11      -- binary of 3
  let result_bin := b1 + b2  -- binary addition result
  let result_dec := 48  -- expected decimal result
  in result_bin = 110000 ∧ result_dec = 48 := 
sorry

end decimal_to_binary_addition_l358_358468


namespace share_of_C_l358_358449

-- Definitions of investments
def investment_B (x : ℝ) : ℝ := x
def investment_A (x : ℝ) : ℝ := 3 * x
def investment_C (x : ℝ) : ℝ := 3 * x * (3 / 2)

-- Total profit
def total_profit : ℝ := 66000

-- Total investment
def total_investment (x : ℝ) : ℝ := (3 * x + x + 3 * x * (3 / 2)) / 2

-- C's share of the profit
def share_C (x : ℝ) : ℝ := (3 * x * (3 / 2) / (3 * x + x + 3 * x * (3 / 2))) * total_profit

theorem share_of_C (x : ℝ) (h : x > 0) : share_C x = 594000 / 17 := by
  admit  -- admit is used here temporarily to indicate unfinished proof

end share_of_C_l358_358449


namespace total_cats_in_meow_and_paw_l358_358464

-- Define the conditions
def CatsInCatCafeCool : Nat := 5
def CatsInCatCafePaw : Nat := 2 * CatsInCatCafeCool
def CatsInCatCafeMeow : Nat := 3 * CatsInCatCafePaw

-- Define the total number of cats in Cat Cafe Meow and Cat Cafe Paw
def TotalCats : Nat := CatsInCatCafeMeow + CatsInCatCafePaw

-- The theorem stating the problem
theorem total_cats_in_meow_and_paw : TotalCats = 40 :=
by
  sorry

end total_cats_in_meow_and_paw_l358_358464


namespace percent_more_l358_358584

variable (B_height : ℕ) (A_height : ℕ)
hypothesis (hA : A_height = (B_height * 65) / 100)

theorem percent_more (hA : A_height = (B_height * 65) / 100) : 
  ((B_height - A_height) * 100 / A_height : ℚ) ≈ 53.846 := 
by
  sorry

end percent_more_l358_358584


namespace minimize_cylinder_surface_area_l358_358151

open Real

variable (V : ℝ) (r h : ℝ)

theorem minimize_cylinder_surface_area (hV : V > 0) :
  (∃ r h, (V = π * r^2 * h) ∧ (S = 2 * π * r^2 + 2 * π * r * h) ∧ 
     (r = real.cbrt (V / (2 * π))) ∧ (h = 2 * real.cbrt (V / (2 * π)))) :=
sorry

end minimize_cylinder_surface_area_l358_358151


namespace steve_popsicle_sticks_l358_358318

theorem steve_popsicle_sticks (S Sid Sam : ℕ) (h1 : Sid = 2 * S) (h2 : Sam = 3 * Sid) (h3 : S + Sid + Sam = 108) : S = 12 :=
by
  sorry

end steve_popsicle_sticks_l358_358318


namespace hyperbola_center_l358_358140

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l358_358140


namespace smallest_number_digit_sum_2017_l358_358917

/-- 
  Prove that the smallest natural number whose digit sum is 2017 has 
  its first digit (from the left) multiplied by the number of its digits equal to 225.
-/
theorem smallest_number_digit_sum_2017 :
  ∃ (n : ℕ), (nat.digits 10 n).sum = 2017 ∧ (nat.digits 10 n).head! * (nat.digits 10 n).length = 225 :=
sorry

end smallest_number_digit_sum_2017_l358_358917


namespace mod_inverse_exists_l358_358906

theorem mod_inverse_exists :
  ∃ a : ℕ, 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 :=
  by {
    use 14,
    split,
    { norm_num },
    split,
    { norm_num },
    exact rfl,
    sorry
  }

end mod_inverse_exists_l358_358906


namespace matrix_inverse_l358_358548

noncomputable def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, a], ![3, b]]

def eigenvalue := -1 : ℝ

def e : Vector (Fin 2) ℝ := ![1, -1]

theorem matrix_inverse (a b : ℝ) (h_eigenvalue : M a b ⬝ e = eigenvalue • e) :
  let detM := (1 : ℝ) * b - a * 3 in
  detM ≠ 0 →
  let adjM := Matrix.adjugate (M a b) in
  (Matrix.inverse (M a b)) = (1 / detM) • adjM :=
  sorry

end matrix_inverse_l358_358548


namespace total_games_played_l358_358250

-- Definitions based on conditions.
variable (total_participants : ℕ) (num_instructors : ℕ)
variable (h1 : total_participants = 15) (h2 : num_instructors = 5)

-- Theorem statement.
theorem total_games_played : 
  let games_per_instructor := total_participants - 1 in
  let total_games := num_instructors * games_per_instructor in
  total_games = 70 :=
by
  sorry

end total_games_played_l358_358250


namespace axis_of_symmetry_l358_358673

-- Define the given parabolic function
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Define the axis of symmetry property for the given parabola
theorem axis_of_symmetry : ∀ x : ℝ, ((2 - x) * x) = -((x - 1)^2) + 1 → (∃ x_sym : ℝ, x_sym = 1) :=
by
  sorry

end axis_of_symmetry_l358_358673


namespace solve_system_of_equations_l358_358329

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (1 / x + 1 / y = 6) ∧ 
    (1 / y + 1 / z = 4) ∧ 
    (1 / z + 1 / x = 5) ∧ 
    (x = 2 / 7) ∧ 
    (y = 2 / 5) ∧ 
    (z = 2 / 3) :=
  by 
    use [2 / 7, 2 / 5, 2 / 3]
    sorry

end solve_system_of_equations_l358_358329


namespace arithmetic_sequence_properties_l358_358286

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (d : ℤ) :
  a 1 = 3 ∧ (∀ n : ℕ, ∃ m : ℕ, (finset.range n).sum (a ∘ nat.succ) = a m) →
  (d = 1 ∨ d = -3 ∨ d = 3) ∧ ∃ d_set : finset ℤ, {1, -3, 3} ⊆ d_set ∧ ∀ d', d' ∈ d_set → (3 / (d' - 3) ∈ ℤ) := 
sorry

end arithmetic_sequence_properties_l358_358286


namespace no_prime_satisfies_condition_l358_358152

theorem no_prime_satisfies_condition (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, (Real.sqrt (p + n) + Real.sqrt n) = k :=
by
  sorry

end no_prime_satisfies_condition_l358_358152


namespace equal_area_l358_358293

noncomputable def acute_triangle (A B C : Type) [triangle A B C] : Prop :=
  acute C ∧ acute B ∧ acute A

def is_projection (P Q R : Type) [triangle P Q R] : Prop :=
  ∃ H, perpendicular (line_through P Q) (line_through H R)

theorem equal_area (A B C L N K M : Type) [triangle A B C]
  (h_acute : acute_triangle A B C)
  (h_internal_bis : is_bisector (angle A B C) A L N)
  (h_K_proj : is_projection L A K)
  (h_M_proj : is_projection L C M)
  :
  area_quadrilateral A K N M = area_triangle A B C :=
sorry

end equal_area_l358_358293


namespace shaded_region_area_l358_358251

theorem shaded_region_area : 
  let rect1_area := 4 * 12 in
  let rect2_area := 5 * 9 in
  let overlap_area := 4 * 5 in
     (rect1_area + rect2_area - overlap_area) = 73 :=
by
  -- Definitions
  let rect1_area := 4 * 12
  let rect2_area := 5 * 9
  let overlap_area := 4 * 5

  -- Main statement
  have eq : (rect1_area + rect2_area - overlap_area) = 73 :=
    by sorry

  exact eq

end shaded_region_area_l358_358251


namespace solve_equation_l358_358325

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l358_358325


namespace alpha_plus_2beta_eq_pi_div_2_l358_358542

theorem alpha_plus_2beta_eq_pi_div_2
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h4 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
by
  sorry

end alpha_plus_2beta_eq_pi_div_2_l358_358542


namespace f_2008_equals_6_l358_358678

noncomputable def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x < 5 then x^2 - x else f (x % 5)

theorem f_2008_equals_6 :
  f 2008 = 6 :=
by sorry

end f_2008_equals_6_l358_358678


namespace quadruples_positive_integers_l358_358487

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l358_358487


namespace div_pow_two_sub_one_l358_358291

theorem div_pow_two_sub_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  (3^k ∣ 2^n - 1) ↔ (∃ m : ℕ, n = 2 * 3^(k-1) * m) :=
by
  sorry

end div_pow_two_sub_one_l358_358291


namespace recurring_decimal_36_exceeds_decimal_35_l358_358110

-- Definition of recurring decimal 0.36...
def recurring_decimal_36 : ℚ := 36 / 99

-- Definition of 0.35 as fraction
def decimal_35 : ℚ := 7 / 20

-- Statement of the math proof problem
theorem recurring_decimal_36_exceeds_decimal_35 :
  recurring_decimal_36 - decimal_35 = 3 / 220 := by
  sorry

end recurring_decimal_36_exceeds_decimal_35_l358_358110


namespace ages_total_l358_358048

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l358_358048


namespace concert_total_payment_l358_358305

def hourly_rate : ℕ → ℝ
| 1 := 25
| 2 := 35
| 3 := 20
| 4 := 30
| 5 := 40
| 6 := 50
| 7 := 45
| 8 := 30
| 9 := 20
| _ := 0

def hours_worked : ℕ → ℝ
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 2.5
| 5 := 3
| 6 := 2
| 7 := 1.5
| 8 := 1
| 9 := 3
| _ := 0

def tip_percentage : ℕ → ℝ
| 1 := 0.15
| 2 := 0.20
| 3 := 0.25
| 4 := 0.18
| 5 := 0.12
| 6 := 0.10
| 7 := 0.15
| 8 := 0.20
| 9 := 0.25
| _ := 0

def total_payment (n : ℕ) : ℝ :=
let base_payment := hours_worked n * hourly_rate n in
let tip := base_payment * tip_percentage n in
base_payment + tip

def final_total_payment : ℝ :=
(total_payment 1) + (total_payment 2) + (total_payment 3) + (total_payment 4) + 
(total_payment 5) + (total_payment 6) + (total_payment 7) + (total_payment 8) + 
(total_payment 9)

theorem concert_total_payment : final_total_payment = 805.03 :=
by
  sorry

end concert_total_payment_l358_358305


namespace coconut_grove_l358_358599

theorem coconut_grove (x : ℕ) :
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) = 300 * x → x = 2 :=
by
  intro h
  -- We can leave the proof part to prove this later.
  sorry

end coconut_grove_l358_358599


namespace sin_eq_neg_inv_arcsin_l358_358536

theorem sin_eq_neg_inv_arcsin (x : Real) (h1 : sin x = -1/3) (h2 : x ∈ Set.Ioo (-Real.pi / 2) 0) : 
  x = -Real.arcsin (1 / 3) :=
by
  sorry

end sin_eq_neg_inv_arcsin_l358_358536


namespace domain_of_expression_l358_358144

theorem domain_of_expression : {x : ℝ | 2 ≤ x ∧ x < 9} = {x : ℝ | ∃y, y = f(x)} :=
by
  let f (x : ℝ) := (sqrt (3 * x - 6)) / (sqrt (9 - x))
  sorry

end domain_of_expression_l358_358144


namespace cannot_make_all_cells_equal_l358_358792

def initial_table : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![0, 1, 0, 0],
    ![0, 0, 0, 0],
    ![0, 0, 0, 0],
    ![0, 0, 0, 0]]

def add_one_row (m : Matrix (Fin 4) (Fin 4) ℕ) (r : Fin 4) : Matrix (Fin 4) (Fin 4) ℕ :=
  ![ if i = r then (m i j + 1) else (m i j) | i, j : Fin 4]

def add_one_col (m : Matrix (Fin 4) (Fin 4) ℕ) (c : Fin 4) : Matrix (Fin 4) (Fin 4) ℕ :=
  ![ if j = c then (m i j + 1) else (m i j) | i, j : Fin 4]

def add_one_diagonal (m : Matrix (Fin 4) (Fin 4) ℕ) (d : (Fin 4 × Fin 4)) : Matrix (Fin 4) (Fin 4) ℕ :=
  -- needs proper diagonal definition; placeholder here:
  sorry

theorem cannot_make_all_cells_equal : ¬∃ m : Matrix (Fin 4) (Fin 4) ℕ, 
  (∃ steps : list (Matrix (Fin 4) (Fin 4) ℕ → Matrix (Fin 4) (Fin 4) ℕ), 
  foldl (λ mat f, f mat) initial_table steps = m) ∧ 
  ∀ i j, m i j = m 0 0 :=
sorry

end cannot_make_all_cells_equal_l358_358792


namespace dot_product_AB_AC_l358_358200

-- Defining the vectors AB and AC
def vecAB : ℝ × ℝ := (-π / 2, 2)
def vecAC : ℝ × ℝ := (π / 2, 2)

-- Calculating the dot product of AB and AC
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

-- Proving the final dot product
theorem dot_product_AB_AC : dot_product vecAB vecAC = 4 - (π^2 / 4) := 
  by sorry

end dot_product_AB_AC_l358_358200


namespace intersection_height_of_poles_l358_358016

theorem intersection_height_of_poles (hA hB d H: ℝ) (hA_val: hA = 20) (hB_val: hB = 80) (d_val: d = 100) : 
    H = 16 :=
by
  -- Given heights of the poles and distance apart
  have hA := 20 : ℝ
  have hB := 80 : ℝ
  have d := 100 : ℝ

  -- Goal: height of intersection point H is 16
  sorry

end intersection_height_of_poles_l358_358016


namespace cevian_intersection_l358_358630

noncomputable theory

open_locale classical

variables {A B C A1 B1 C1 : Type} 
          {BC AC AB : Type}
          (BC_contains_A1 : BC A1)
          (AC_contains_B1 : AC B1)
          (AB_contains_C1 : AB C1)

theorem cevian_intersection (h : (AB₁ / B₁C) * (CA₁ / A₁B) * (BC₁ / C₁A) = 1) : 
  ∃ M, lines_intersection AA₁ BB₁ CC₁ M :=
sorry

end cevian_intersection_l358_358630


namespace plane_divided_into_four_regions_l358_358120

theorem plane_divided_into_four_regions :
  ∃ regions : ℕ, 
    regions = 4 ∧ 
    (∀ (x y : ℝ), (y = 3 * x) ∨ (y = (1 / 3) * x) → 
      (y ≠ 3 * x ∨ y ≠ (1 / 3) * x ∨ (x = y))). 
sorry

end plane_divided_into_four_regions_l358_358120


namespace dihedral_angle_right_angle_iff_l358_358604

-- Definitions for the problem conditions
variable (α β : Type) -- Define two types for the planes
variable (l : Type) -- Define a type for the edge
variable (a b : Type) -- Define types for the lines
variable (in_plane : α → Type) -- a belongs to plane α
variable (in_plane : β → Type) -- b belongs to plane β
variable (not_perpendicular_to_edge : α → β → Prop) -- neither plane is perpendicular to the edge l
variable (right_dihedral_angle : Prop) -- define right dihedral angle

-- The statement of the problem, proving the correct answer
theorem dihedral_angle_right_angle_iff (h : right_dihedral_angle)
  (ha : in_plane a) 
  (hb : in_plane b) 
  (na : not_perpendicular_to_edge a l) 
  (nb : not_perpendicular_to_edge b l) : 
  ∃ (h_parallel : Prop) (h_not_perpendicular : Prop), 
    h_parallel a b ∧ ¬ h_not_perpendicular a b := 
sorry

end dihedral_angle_right_angle_iff_l358_358604


namespace find_overhead_expenses_l358_358442

noncomputable def overhead_expenses : ℝ := 35.29411764705882 / (1 + 0.1764705882352942)

theorem find_overhead_expenses (cost_price selling_price profit_percent : ℝ) (h_cp : cost_price = 225) (h_sp : selling_price = 300) (h_pp : profit_percent = 0.1764705882352942) :
  overhead_expenses = 30 :=
by
  sorry

end find_overhead_expenses_l358_358442


namespace perimeter_of_sector_l358_358543

def central_angle : ℝ := (2/3) * Real.pi
def area_of_sector : ℝ := 3 * Real.pi

theorem perimeter_of_sector (θ : ℝ) (S : ℝ) (hθ : θ = central_angle) (hS : S = area_of_sector) :
  let R := Real.sqrt (2 * S / θ)
  let l := θ * R
  l + 2 * R = 6 + 2 * Real.pi :=
by
  sorry

end perimeter_of_sector_l358_358543


namespace modulo_inverse_5_mod_23_l358_358908

/-- The modular inverse of 5 modulo 23 is 14. -/
theorem modulo_inverse_5_mod_23 : ∃ b : ℕ, b ∈ set.Ico 0 23 ∧ (5 * b) % 23 = 1 :=
  by
  use 14
  split
  · exact Nat.le_of_lt (by norm_num : (14 < 23))
  · change (5 * 14) % 23 = 1
    norm_num
    sorry

end modulo_inverse_5_mod_23_l358_358908


namespace work_duration_l358_358052

theorem work_duration (X_full_days : ℕ) (Y_full_days : ℕ) (Y_worked_days : ℕ) (R : ℚ) :
  X_full_days = 18 ∧ Y_full_days = 15 ∧ Y_worked_days = 5 ∧ R = (2 / 3) →
  (R / (1 / X_full_days)) = 12 :=
by
  intros h
  sorry

end work_duration_l358_358052


namespace arithmetic_expression_l358_358020

theorem arithmetic_expression : 5 + 12 / 3 - 3 ^ 2 + 1 = 1 := by
  sorry

end arithmetic_expression_l358_358020


namespace sector_area_l358_358086

theorem sector_area (r : ℝ) (alpha : ℝ) (h : r = 2) (h2 : alpha = π / 3) : 
  1/2 * alpha * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l358_358086


namespace simson_line_condition_l358_358940

variables {A B C P P_A P_B P_C : Point} {triangle : Triangle}

-- Assume we have definitions for orthogonal projections, collinear points, and circumcircle
def is_orthogonal_projection (P P' : Point) (line : Line) : Prop := sorry -- This should define the orthogonal projection condition
def are_collinear (P1 P2 P3 : Point) : Prop := sorry -- This should define the collinearity condition
def on_circumcircle (P : Point) (triangle : Triangle) : Prop := sorry -- This should define the condition for point P being on the circumcircle of a given triangle

-- The theorem to prove
theorem simson_line_condition (h_pa : is_orthogonal_projection P P_A (line B C))
                             (h_pb : is_orthogonal_projection P P_B (line A C))
                             (h_pc : is_orthogonal_projection P P_C (line A B)) :
                             are_collinear P_A P_B P_C ↔ on_circumcircle P triangle :=
sorry

end simson_line_condition_l358_358940


namespace modular_inverse_of_5_mod_23_l358_358903

theorem modular_inverse_of_5_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (5 * a) % 23 = 1 := 
begin
  use 14,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end modular_inverse_of_5_mod_23_l358_358903


namespace largest_divisor_of_n_cube_minus_n_minus_six_l358_358146

theorem largest_divisor_of_n_cube_minus_n_minus_six (n : ℤ) : 6 ∣ (n^3 - n - 6) :=
by sorry

end largest_divisor_of_n_cube_minus_n_minus_six_l358_358146


namespace quadratic_has_real_roots_find_m_l358_358965

theorem quadratic_has_real_roots (m : ℝ) :
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5) in
  discriminant ≥ 0 ↔ m ≥ 1 / 2 :=
by
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5)
  split
  { intro h
    sorry -- This proof would show that if the discriminant is non-negative, then m ≥ 1/2
  }
  { intro h
    sorry -- This proof would show that if m ≥ 1/2, then the discriminant is non-negative
  }

theorem find_m (m : ℝ) (x1 x2 : ℝ) :
  (x1 + x2 = 4) →
  (x1 * x2 = -2 * m + 5) →
  (x1 * x2 + x1 + x2 = m ^ 2 + 6) →
  m ≥ 1 / 2 →
  m = 1 :=
by
  intros h1 h2 h3 h4
  sorry -- This proof would show that given the conditions, m must be 1

end quadratic_has_real_roots_find_m_l358_358965


namespace range_of_a_l358_358299

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 0.5 * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ -1 < a ∧ a < 0 := by
  sorry

end range_of_a_l358_358299


namespace sum_of_digits_of_binary_315_is_6_l358_358745
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358745


namespace sum_of_y_when_x_in_range_l358_358165

theorem sum_of_y_when_x_in_range :
  let y (x : ℕ) := sqrt (x - 4) ^ 2 - x + 5
  ∑ x in Finset.range 2024 | 0 < x, y x = 2036 :=
by
  let y : ℕ → ℝ := λ x, abs (x - 4) - x + 5
  have key : ∀ x, 0 < x → y x = if x < 4 then 9 - 2 * x else 1 := by
    intro x hx
    by_cases h : x < 4
    · rw [if_pos h, abs_of_lt (show x - 4 < 0 by linarith)]
      linarith
    · rw [if_neg h, abs_of_nonneg (show x - 4 ≥ 0 by linarith)]
      linarith
  have : ∑ x in Finset.range 2024 | 0 < x, (if x < 4 then 9 - 2 * x else 1) = 2036 := by
    simp [sum_congr, {inner := 3, Finset.sum, Finset.sum_const, Nat.card}]
  exact this.trans (sum_congr rfl key)

end sum_of_y_when_x_in_range_l358_358165


namespace a5_is_3_l358_358170

section
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_a2 : a 2 = Real.sqrt 3)
variable (h_recursive : ∀ n ≥ 2, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2)

theorem a5_is_3 : a 5 = 3 :=
  by
  sorry
end

end a5_is_3_l358_358170


namespace parallel_condition_l358_358931

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_condition (x : ℝ) : 
  let a := (2, 1)
  let b := (3 * x ^ 2 - 1, x)
  (x = 1 → are_parallel a b) ∧ 
  ∃ x', x' ≠ 1 ∧ are_parallel a (3 * x' ^ 2 - 1, x') :=
by
  sorry

end parallel_condition_l358_358931


namespace divided_number_l358_358820

theorem divided_number (x y : ℕ) (h1 : 7 * x + 5 * y = 146) (h2 : y = 11) : x + y = 24 :=
sorry

end divided_number_l358_358820


namespace vector_subtraction_proof_l358_358915

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (1, -6)
def scalar1 : ℝ := 2
def scalar2 : ℝ := 3

theorem vector_subtraction_proof :
  v1 - (scalar2 • (scalar1 • v2)) = (-3, 32) := by
  sorry

end vector_subtraction_proof_l358_358915


namespace simplify_sqrt_9800_l358_358657

noncomputable def square_root_100 : ℝ := real.sqrt 100
noncomputable def square_root_49 : ℝ := real.sqrt 49

theorem simplify_sqrt_9800 :
  real.sqrt 9800 = 70 * real.sqrt 2 :=
by
  have h1 : 9800 = 100 * 98 := by norm_num
  have h2 : 98 = 2 * 49 := by norm_num
  have h3 : square_root_100 = 10 := by simp [square_root_100, real.sqrt, nat.cast_bit1, nat.cast_bit0, nat.cast_one]; norm_num
  have h4 : square_root_49 = 7 := by simp [square_root_49, real.sqrt, nat.cast_bit1, nat.cast_bit0, nat.cast_one]; norm_num
  sorry

end simplify_sqrt_9800_l358_358657


namespace arc_length_eq_4pi_area_of_sector_eq_12pi_l358_358186

namespace SectorGeometry

-- Definitions
def α : ℝ := 2 / 3 * Real.pi
def r : ℝ := 6

-- Theorems
theorem arc_length_eq_4pi : (α * r) = 4 * Real.pi := by
  sorry

theorem area_of_sector_eq_12pi : (1 / 2 * α * r^2) = 12 * Real.pi := by
  sorry

end SectorGeometry

end arc_length_eq_4pi_area_of_sector_eq_12pi_l358_358186


namespace initial_total_cards_l358_358430

theorem initial_total_cards (x y : ℕ) (h1 : x / (x + y) = 1 / 3) (h2 : x / (x + y + 4) = 1 / 4) : x + y = 12 := 
sorry

end initial_total_cards_l358_358430


namespace average_yield_and_quote_l358_358640

def stock_Yield_Quote :=
  (div_yield : ℝ) × (quote_percentage : ℝ)

def stockA : stock_Yield_Quote := (0.15, 0.12)
def stockB : stock_Yield_Quote := (0.10, -0.05)
def stockC : stock_Yield_Quote := (0.08, 0.18)

def average (x y z : ℝ) : ℝ := (x + y + z) / 3

theorem average_yield_and_quote :
  average stockA.1 stockB.1 stockC.1 = 0.11 ∧
  average stockA.2 stockB.2 stockC.2 = 25 / 3 / 100 := sorry

end average_yield_and_quote_l358_358640


namespace solidAngle_rightCircularCone_l358_358900

-- Definition of the solid angle of a right circular cone.
def solidAngle (α : ℝ) : ℝ := 4 * π * (Real.sin (α / 4)) ^ 2

-- Theorem statement that proves the magnitude of the solid angle.
theorem solidAngle_rightCircularCone (α : ℝ) : 
solidAngle α = 4 * π * (Real.sin (α / 4)) ^ 2 := by
  sorry

end solidAngle_rightCircularCone_l358_358900


namespace probability_of_selecting_same_color_shoes_l358_358788

-- Define conditions as necessary
variables (pairs : Finset (Fin 9))        -- Represent 9 pairs of shoes
variables (shoes : Finset (Fin 18))       -- Represent 18 shoes
variables (distinct_colors : ∀ (p1 p2 : Fin 9), p1 ≠ p2 → shoe_color p1 ≠ shoe_color p2)  -- Different colors
variables (select_without_replacement : ∀ (s1 s2 : Fin 18), s1 ≠ s2)  -- Selection without replacement

-- Define the color function for the shoes (helper function)
noncomputable def shoe_color (s : Fin 18) : Fin 9 :=
sorry  -- Color mapping can be defined here if necessary

-- Theorem stating the problem and the answer
theorem probability_of_selecting_same_color_shoes :
  probability_of_selecting_same_color_shoes pairs shoes distinct_colors select_without_replacement = 9 / 2601 := 
sorry

end probability_of_selecting_same_color_shoes_l358_358788


namespace ellipse_with_foci_on_y_range_l358_358591

theorem ellipse_with_foci_on_y_range (k : ℝ) :
  (∀ x y : ℝ, x^2 + k * y^2 = 2 → ∃ a b : ℝ, a ≠ b ∧ (a^2 + b^2 = 1 ∧ k ∈ (0, 1))) :=
sorry

end ellipse_with_foci_on_y_range_l358_358591


namespace solve_problem_1_solve_problem_2_l358_358561

noncomputable def problem_1_statement (b : ℝ) : Prop :=
  (∀ x : ℝ, (0 < x ∧ x < 4 → x^b > 0)) ∧ (∀ x : ℝ, (4 < x → x^b > 0)) ∧ b = 4

noncomputable def problem_2_statement (c : ℝ) (x : ℝ) : Prop :=
  (1 ≤ x ∧ x ≤ 2 ∧ 1 < c ∧ c < 4 →
  ∀ minimum_value, (minimum_value = 2^c) ∧
  ∀ maximum_value, 
  (if 1 < c ∧ c < 2 then maximum_value = 2^c + c
  else if c = 2 then maximum_value = 3
  else if 2 < c ∧ c < 4 then maximum_value = 1^c + c
  else false))

theorem solve_problem_1 : ∀ b : ℝ, problem_1_statement b := 
by sorry

theorem solve_problem_2 : ∀ c x : ℝ, problem_2_statement c x :=
by sorry

end solve_problem_1_solve_problem_2_l358_358561


namespace largest_base_number_l358_358453

theorem largest_base_number :
  let n1 := 31 in  -- 11111 in base 2 converted to decimal
  let n2 := 52 in  -- 1221 in base 3 converted to decimal
  let n3 := 54 in  -- 312 in base 4 converted to decimal
  let n4 := 46 in  -- 56 in base 8 converted to decimal
  n3 > n1 ∧ n3 > n2 ∧ n3 > n4 :=
by {
  let n1 := 31,
  let n2 := 52,
  let n3 := 54,
  let n4 := 46,
  sorry
}

end largest_base_number_l358_358453


namespace solve_floor_equation_l358_358285

theorem solve_floor_equation (x : ℚ) 
  (h : ⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) : 
  x = 7 / 15 ∨ x = 4 / 5 := 
sorry

end solve_floor_equation_l358_358285


namespace volume_of_rotated_rectangle_l358_358914

-- Define the problem conditions
def length : ℝ := 20
def width : ℝ := 10

-- Define the radius and height of the cylinder
def radius : ℝ := width / 2
def height : ℝ := length

-- Volume of the cylinder formula
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Statement of the theorem
theorem volume_of_rotated_rectangle :
  volume_cylinder radius height = 500 * π := 
by
  -- Proof skipped
  sorry

end volume_of_rotated_rectangle_l358_358914


namespace count_ordered_pairs_l358_358221

noncomputable def count_solutions : Nat :=
  let solutions := {p : ℝ × ℝ | 2 * p.1 - p.2 = 2 ∧ abs (abs p.1 - 2 * abs p.2) = 2}
  solutions.to_finset.card

theorem count_ordered_pairs : count_solutions = 4 :=
  sorry

end count_ordered_pairs_l358_358221


namespace domain_of_f_l358_358896

def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1) + 1 / (x - 2)

theorem domain_of_f :
  { x : ℝ | 0 ≤ x ∧ x ≠ 2 } = { x : ℝ | 0 ≤ x ∨ (2 < x ∧ x ≠ 2)} :=
by sorry

end domain_of_f_l358_358896


namespace molecular_weight_of_3_moles_HBrO3_l358_358387

-- Definitions from the conditions
def mol_weight_H : ℝ := 1.01  -- atomic weight of H
def mol_weight_Br : ℝ := 79.90  -- atomic weight of Br
def mol_weight_O : ℝ := 16.00  -- atomic weight of O

-- Definition of molecular weight of HBrO3
def mol_weight_HBrO3 : ℝ := mol_weight_H + mol_weight_Br + 3 * mol_weight_O

-- The goal: The molecular weight of 3 moles of HBrO3 is 386.73 grams
theorem molecular_weight_of_3_moles_HBrO3 : 3 * mol_weight_HBrO3 = 386.73 :=
by
  -- We will insert the proof here later
  sorry

end molecular_weight_of_3_moles_HBrO3_l358_358387


namespace distance_between_cities_l358_358780

variable (D : ℝ) -- D is the distance between City A and City B
variable (time_AB : ℝ) -- Time from City A to City B
variable (time_BA : ℝ) -- Time from City B to City A
variable (saved_time : ℝ) -- Time saved per trip
variable (avg_speed : ℝ) -- Average speed for the round trip with saved time

theorem distance_between_cities :
  time_AB = 6 → time_BA = 4.5 → saved_time = 0.5 → avg_speed = 90 →
  D = 427.5 :=
by
  sorry

end distance_between_cities_l358_358780


namespace angle_between_planes_l358_358139

theorem angle_between_planes :
  let n1 := (1, 2, 2) in
  let n2 := (2, -1, 2) in
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3 in
  let magnitude_n1 := Real.sqrt (n1.1 ^ 2 + n1.2 ^ 2 + n1.3 ^ 2) in
  let magnitude_n2 := Real.sqrt (n2.1 ^ 2 + n2.2 ^ 2 + n2.3 ^ 2) in
  let cos_phi := dot_product / (magnitude_n1 * magnitude_n2) in
  let phi := Real.arccos cos_phi in
  phi = Real.arccos (4 / 9) :=
by
  let n1 := (1, 2, 2)
  let n2 := (2, -1, 2)
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude_n1 := Real.sqrt (n1.1 ^ 2 + n1.2 ^ 2 + n1.3 ^ 2)
  let magnitude_n2 := Real.sqrt (n2.1 ^ 2 + n2.2 ^ 2 + n2.3 ^ 2)
  let cos_phi := dot_product / (magnitude_n1 * magnitude_n2)
  let phi := Real.arccos cos_phi
  sorry

end angle_between_planes_l358_358139


namespace probability_three_twos_given_sum_six_l358_358704

open ProbabilityTheory

-- Definitions of the conditions
def balls : List ℕ := [1, 2, 3]
def urn := finset.univ : Finset ℕ).image (λ _, balls)

-- The event space
def events := urn.product urn.product urn

-- The condition that the sum of the draws is 𝟞
def total_sum_six (e : ℕ × (ℕ × ℕ)) : Prop :=
  e.1 + e.2.1 + e.2.2 = 6

-- The favorable event where all draws are 𝟚
def all_twos (e : ℕ × (ℕ × ℕ)) : Prop :=
  e.1 = 2 ∧ e.2.1 = 2 ∧ e.2.2 = 2

-- The main statement to prove
theorem probability_three_twos_given_sum_six :
  (∑' (e : ℕ × (ℕ × ℕ)) in events.filter all_twos, 1 / events.card) /
  (∑' (e : ℕ × (ℕ × ℕ)) in events.filter total_sum_six, 1 / events.card)
  = 1 / 7 :=
sorry

end probability_three_twos_given_sum_six_l358_358704


namespace no_such_functions_exist_l358_358878

theorem no_such_functions_exist (f g : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1) :=
sorry

end no_such_functions_exist_l358_358878


namespace participants_all_ten_correct_l358_358059

noncomputable def num_participants := 60
noncomputable def num_questions := 10
noncomputable def total_correct_answers := 452
noncomputable def min_correct_answers_per_participant := 6
noncomputable def participants_six_correct := 21
noncomputable def participants_eight_correct := 12

theorem participants_all_ten_correct :
  ∃ n, n = 7 ∧
  ∃ (num_participants num_questions total_correct_answers min_correct_answers_per_participant participants_six_correct participants_eight_correct),
  num_participants = 60 ∧
  num_questions = 10 ∧
  total_correct_answers = 452 ∧
  min_correct_answers_per_participant = 6 ∧
  participants_six_correct = 21 ∧
  participants_eight_correct = 12 ∧
  ∀ (participants_seven_correct participants_nine_correct : ℕ), 
    participants_seven_correct = participants_nine_correct ∧
    21 * 6 + 12 * 8 + participants_seven_correct * 7 + participants_nine_correct * 9 + n * 10 = 452 :=
sorry


end participants_all_ten_correct_l358_358059


namespace find_certain_number_l358_358802

theorem find_certain_number : 
  ∃ (certain_number : ℕ), 1038 * certain_number = 173 * 240 ∧ certain_number = 40 :=
by
  sorry

end find_certain_number_l358_358802


namespace emily_necklaces_for_friends_l358_358133

theorem emily_necklaces_for_friends (n b B : ℕ)
  (h1 : n = 26)
  (h2 : b = 2)
  (h3 : B = 52)
  (h4 : n * b = B) : 
  n = 26 :=
by
  sorry

end emily_necklaces_for_friends_l358_358133


namespace determine_p_l358_358169

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end determine_p_l358_358169


namespace domain_of_h_l358_358022

noncomputable def h (x : ℝ) : ℝ := (3 * x + 1) / (x ^ 2 - 9)

theorem domain_of_h :
  ∀ x : ℝ, (∃ y, h x = y) ↔ x ∈ ((set.Ioo (real.is_real_neg_infty) (-3)) ∪ (set.Ioo (-3) 3) ∪ (set.Ioo 3 (real.is_real_pos_infty))) :=
by 
  sorry

end domain_of_h_l358_358022


namespace fruits_in_good_condition_l358_358406

def percentage_good_fruits (num_oranges num_bananas pct_rotten_oranges pct_rotten_bananas : ℕ) : ℚ :=
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (pct_rotten_oranges * num_oranges) / 100
  let rotten_bananas := (pct_rotten_bananas * num_bananas) / 100
  let good_fruits := total_fruits - (rotten_oranges + rotten_bananas)
  (good_fruits * 100) / total_fruits

theorem fruits_in_good_condition :
  percentage_good_fruits 600 400 15 8 = 87.8 := sorry

end fruits_in_good_condition_l358_358406


namespace max_x_proof_l358_358883

-- Conditions
def unique_numbers (s : set ℕ) : Prop :=
  s = {1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_neighbors (x : ℕ) (left right : ℕ) (s : set ℕ) : Prop :=
  sum s % x = 0

noncomputable def max_x : ℕ :=
  max_value

theorem max_x_proof :
  ∀ (s : set ℕ) (left right : ℕ),
    unique_numbers s → 
    left = 4 → 
    right = 5 → 
    valid_neighbors max_x left right s → 
    (∀ x ∈ s, valid_neighbors x left right s → x ≤ max_x) →
  max_x = 6 :=
by
  sorry

end max_x_proof_l358_358883


namespace problem_statement1_problem_statement2_l358_358530

variable (α β : ℝ)
variable (A : Set ℝ)

-- Define proposition p
def p : Prop := ∀ α β, (α = β ↔ tan α = tan β)

-- Define proposition q
def q : Prop := ∅ ⊆ A

-- Theorem statements
theorem problem_statement1 : p α β ∨ q A :=
by sorry

theorem problem_statement2 : ¬ p α β :=
by sorry

end problem_statement1_problem_statement2_l358_358530


namespace find_matrix_solution_l358_358491

variables {α : ℝ} {c1 c2 c3 : ℝ}
variables (M : Matrix (Fin 3) (Fin 3) ℝ) 
variables (i j k : Matrix (Fin 3) (Fin 1) ℝ)

noncomputable def is_solution_matrix (M : Matrix (Fin 3) (Fin 3) ℝ) (c1 c2 c3 α : ℝ) : Prop :=
  (M ⬝ i = (c1 • i)) ∧ 
  (M ⬝ j = (c2 • j + α • k)) ∧ 
  (M ⬝ k = (c3 • k))

theorem find_matrix_solution (M : Matrix (Fin 3) (Fin 3) ℝ)
  (h1 : M ⬝ i = (c1 • i))
  (h2 : M ⬝ j = (c2 • j + α • k))
  (h3 : M ⬝ k = (c3 • k)) :
  M = ![![c1, 0, 0], ![0, c2, α], ![0, 0, c3]] :=
  sorry

end find_matrix_solution_l358_358491


namespace find_m_l358_358990

theorem find_m (m : ℤ) : m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 :=
sorry

end find_m_l358_358990


namespace find_last_number_l358_358413

theorem find_last_number (A B C D : ℝ) (h1 : A + B + C = 18) (h2 : B + C + D = 9) (h3 : A + D = 13) : D = 2 :=
by
sorry

end find_last_number_l358_358413


namespace shirt_final_price_is_correct_l358_358828

noncomputable def final_price_percentage (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * 0.80
  let second_discount := first_discount * 0.90
  let anniversary_addition := second_discount * 1.05
  let final_price := anniversary_addition * 1.15
  final_price / initial_price * 100

theorem shirt_final_price_is_correct (initial_price : ℝ) : final_price_percentage initial_price = 86.94 := by
  sorry

end shirt_final_price_is_correct_l358_358828


namespace isosceles_triangle_solution_l358_358660

noncomputable def isosceles_triangle_properties (t : ℝ) (α : ℝ) : Prop :=
  let l := Math.sqrt(512 / Real.sin α)
  ∧ let a := 2 * l * Real.sin(α / 2)
  ∧ let β := 90 - α / 2
  t = 256 ∧ α = (48 + 15 / 60 + 20 / 3600) * (π / 180) →  -- 48 degrees, 15 minutes, 20 seconds in radians
  l = 26.19 ∧ a = 21.41 ∧ β = (65 + 52 / 60 + 20 / 3600) * (π / 180) 

-- Here we declare the conjecture we want to check as a theorem (statement only).
theorem isosceles_triangle_solution : isosceles_triangle_properties 256 (48 + 15 / 60 + 20 / 3600) * (π / 180) := 
sorry -- Proof omitted

end isosceles_triangle_solution_l358_358660


namespace polynomials_with_roots_l358_358484

theorem polynomials_with_roots (x : ℝ) :
  (x = sqrt 2 + sqrt 3 ∨ x = sqrt 2 + real.cbrt 3) →
  (x^4 - 10 * x^2 + 1) * (x^6 - 6 * x^4 - 6 * x^3 + 12 * x^2 - 36 * x + 1) = 0 :=
by
  sorry

end polynomials_with_roots_l358_358484


namespace gcd_of_36_and_54_l358_358728

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l358_358728


namespace fraction_power_seven_l358_358021

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := 
by
  sorry

end fraction_power_seven_l358_358021


namespace average_of_five_integers_l358_358943

theorem average_of_five_integers (a1 a2 a3 a4 a5 : ℤ) 
  (h : (a1 + a2) + (a1 + a3) + (a1 + a4) + (a1 + a5) + (a2 + a3) + (a2 + a4) + (a2 + a5) + (a3 + a4) + (a3 + a5) + (a4 + a5) = 2020) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 101 := 
begin
  sorry
end

end average_of_five_integers_l358_358943


namespace circle_theorem_l358_358523

theorem circle_theorem {O A B X Y : Type} [metric_space O]
  (h1 : ∃ (c : circle), intersect_plane_sphere O c)
  (H_a : distance A B = a)
  (H_b : distance A X = b ∧ distance A Y = b)
  (H_AO_perp : perpendicular O A (plane_through O A))
  (H_XY_plane_through_AB : plane_through A B ∩ circ = {X, Y}) :
  distance B X * distance B Y = a^2 - b^2 :=
sorry

end circle_theorem_l358_358523


namespace problem_1_max_min_value_problem_2_monotonic_interval_l358_358556

def f (a x : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem problem_1_max_min_value :
  ∀ (x : ℝ), x ∈ set.Icc (-2 : ℝ) 5 →
    f (-2) x ≤ 13 ∧ f (-2) x ≥ -3 :=
by sorry

theorem problem_2_monotonic_interval :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 ∈ set.Icc (-2 : ℝ) 5 ∧ x2 ∈ set.Icc (-2 : ℝ) 5 → 
    x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (a ≤ -5 ∨ a ≥ 2) :=
by sorry

end problem_1_max_min_value_problem_2_monotonic_interval_l358_358556


namespace sum_bn_lt_half_l358_358937

section ArithmeticSequence

variables {a_n : ℕ → ℕ}

/-- Define the sequence {a_n} with a_n = 2n + 5 --/
def a (n : ℕ) : ℕ := 2 * n + 5

/-- Define the sequence {b_n} with b_n = 1 / ((a_n - 6) * (a_n - 4)) --/
def b (n : ℕ) : ℚ := 1 / ((a n - 6) * (a n - 4))

/-- Conditions: sum of the first 5 terms of {a_n} is 55 and a_2, sqrt(a_6 + a_7), a_4 - 9 form a geometric sequence --/
def conditions : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 55) ∧
  ((a 2 : ℚ) * (a 4 - 9) = (√((a 6 + a 7))))^2

/-- Prove that for the sequence {a_n}, the sum of the first n terms of the sequence {b_n} is less than 1/2 --/
theorem sum_bn_lt_half (n : ℕ) : conditions → (∑ k in Finset.range n, b (k + 1)) < (1 / 2) :=
by
  sorry

end ArithmeticSequence

end sum_bn_lt_half_l358_358937


namespace work_done_in_days_l358_358582

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l358_358582


namespace no_rect_with_exactly_20_cells_l358_358310

theorem no_rect_with_exactly_20_cells (exists_40_marked_cells : ∃ (grid : ℕ → ℕ → bool), (∑ i j, if grid i j then 1 else 0 = 40)) :
  ¬ ∀ (grid : ℕ → ℕ → bool) (r1 r2 c1 c2 : ℕ), 
    ∑ i in finset.range (r2 - r1 + 1), ∑ j in finset.range (c2 - c1 + 1), if grid (i + r1) (j + c1) then 1 else 0 = 20 := 
sorry

end no_rect_with_exactly_20_cells_l358_358310


namespace rectangular_prism_volume_l358_358695

theorem rectangular_prism_volume (w : ℝ) (w_pos : 0 < w) 
    (h_edges_sum : 4 * w + 8 * (2 * w) + 4 * (w / 2) = 88) :
    (2 * w) * w * (w / 2) = 85184 / 343 :=
by
  sorry

end rectangular_prism_volume_l358_358695


namespace bella_steps_to_meet_ella_l358_358109

open Nat Real

noncomputable def steps_bella_takes (distance_miles : ℝ) (feet_per_mile : ℕ) (bella_speed_factor : ℕ) (ella_speed_factor : ℕ) (step_length : ℕ) : ℕ :=
  let distance_feet := distance_miles * feet_per_mile
  let total_speed := bella_speed_factor + ella_speed_factor
  let time_to_meet := distance_feet / total_speed
  let distance_bella_covers := bella_speed_factor * time_to_meet
  distance_bella_covers / step_length

theorem bella_steps_to_meet_ella :
  steps_bella_takes 3 5280 1 4 3 = 1056 :=
by
  sorry

end bella_steps_to_meet_ella_l358_358109


namespace weight_of_third_piece_l358_358312

-- Define the weights of the first two pieces
def w1 : ℝ := 0.3333333333333333
def w2 : ℝ := 0.3333333333333333

-- Define the total weight of all pieces
def total_weight : ℝ := 0.75

-- Define the weight of the third piece
def w3 : ℝ := total_weight - (w1 + w2)

-- Prove that the weight of the third piece is 0.08333333333333337
theorem weight_of_third_piece : w3 = 0.08333333333333337 := by
  simp [w3, total_weight, w1, w2]
  norm_num

end weight_of_third_piece_l358_358312


namespace sqrt_log_addition_l358_358479

def log_base_change (b a : ℝ) (h : a > 0) (h1 : b > 0) : ℝ := real.log a / real.log b

theorem sqrt_log_addition :
  sqrt (log_base_change 4 8 _ _ + log_base_change 8 16 _ _) = sqrt (17 / 6) :=
begin
  -- Definitions of logarithmic change of base properties
  have log2_8 : real.log 8 = real.log 2 ^ 3 := by norm_num,
  have log2_4 : real.log 4 = real.log 2 ^ 2 := by norm_num,
  have log2_16 : real.log 16 = real.log 2 ^ 4 := by norm_num,

  sorry
end

end sqrt_log_addition_l358_358479


namespace simplify_neg_neg_l358_358393

theorem simplify_neg_neg (a b : ℝ) : -(-a - b) = a + b :=
sorry

end simplify_neg_neg_l358_358393


namespace center_dispersion_correct_covariance_matrix_correct_l358_358082

noncomputable def center_of_dispersion (ξ η : ℝ) (D : Set (ℝ × ℝ))
  [IsProbabilityMeasure (uniformMeasure D)] : (ℝ × ℝ) :=
  (4 / (3 * π), 4 / (3 * π))

noncomputable def covariance_matrix (ξ η : ℝ) (D : Set (ℝ × ℝ))
  [IsProbabilityMeasure (uniformMeasure D)] : Matrix (Fin 2) (Fin 2) ℝ :=
  (λ i j, if (i,j) = (0,0) then (1 / 4 - 16 / (9 * π^2))
          else if (i,j) = (1,1) then (1 / 4 - 16 / (9 * π^2))
          else if (i,j) = (0,1) then (1 / (2 * π) - 16 / (9 * π^2))
          else if (i,j) = (1,0) then (1 / (2 * π) - 16 / (9 * π^2))
          else 0)

theorem center_dispersion_correct (ξ η : ℝ) (D : Set (ℝ × ℝ))
  [IsProbabilityMeasure (uniformMeasure D)] :
  D = {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ p.1^2 + p.2^2 ≤ 1} →
  center_of_dispersion ξ η D = (4 / (3 * π), 4 / (3 * π)) :=
sorry

theorem covariance_matrix_correct (ξ η : ℝ) (D : Set (ℝ × ℝ))
  [IsProbabilityMeasure (uniformMeasure D)] :
  D = {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ p.1^2 + p.2^2 ≤ 1} →
  covariance_matrix ξ η D = 
  (λ i j, if (i,j) = (0,0) then (1 / 4 - 16 / (9 * π^2))
          else if (i,j) = (1,1) then (1 / 4 - 16 / (9 * π^2))
          else if (i,j) = (0,1) then (1 / (2 * π) - 16 / (9 * π^2))
          else if (i,j) = (1,0) then (1 / (2 * π) - 16 / (9 * π^2))
          else 0) :=
sorry

end center_dispersion_correct_covariance_matrix_correct_l358_358082


namespace number_of_bricks_to_build_wall_l358_358050

-- Define the dimensions of the brick
def brick_length_cm : ℝ := 20
def brick_height_cm : ℝ := 13.25
def brick_width_cm : ℝ := 8

-- Define the dimensions of the wall in cm
def wall_length_cm : ℝ := 700
def wall_height_cm : ℝ := 800
def wall_width_cm : ℝ := 1550

-- Calculate the volume of the brick and the wall
def volume_brick : ℝ := brick_length_cm * brick_height_cm * brick_width_cm
def volume_wall : ℝ := wall_length_cm * wall_height_cm * wall_width_cm

-- Calculate the number of bricks needed
def number_of_bricks : ℝ := volume_wall / volume_brick

-- The proof statement
theorem number_of_bricks_to_build_wall : number_of_bricks.toNat = 409434 :=
by sorry

end number_of_bricks_to_build_wall_l358_358050


namespace radius_ratio_l358_358381

-- Define the various elements in the problem.
variables (R1 R2 : ℝ) (O1 O2 B C E : Type) [metric_space O1] [metric_space O2] [metric_space B]
[metric_space C] [metric_space E]

-- Define the ratio of the areas as per the conditions.
def area_ratio_condition (S_BO1CO2 S_O2BE : ℝ) : Prop :=
  S_BO1CO2 / S_O2BE = 5 / 4

-- Given the above conditions, prove the required ratio of radii.
theorem radius_ratio (h : area_ratio_condition (R1 * R2 * Real.sin α) (R2^2 * Real.sin α * Real.cos α)) :
  R2 / R1 = 6 / 5 :=
sorry

end radius_ratio_l358_358381


namespace stock_yield_is_10_percent_l358_358425

-- Define the necessary financial terms and conditions
def par_value (stock : Type) : ℝ := 100
def market_value (stock : Type) := 50 -- dollars
def dividend_rate (stock : Type) := 0.05 -- 5%

-- Define the annual dividend based on par value and dividend rate
def annual_dividend (stock : Type) : ℝ :=
  dividend_rate stock * par_value stock

-- Define the yield calculation based on annual dividend and market value
def stock_yield (stock : Type) : ℝ :=
  (annual_dividend stock / market_value stock) * 100

-- Define the theorem we want to prove
theorem stock_yield_is_10_percent (stock : Type) : stock_yield stock = 10 :=
by
  sorry

end stock_yield_is_10_percent_l358_358425


namespace smallest_n_condition_l358_358177

theorem smallest_n_condition (k : ℕ) (hk : k ≥ 2) :
  ∃ n, (∀ (points : Finset (ℝ × ℝ)), points.card = n →
    (∃ (subset : Finset (ℝ × ℝ)), subset ⊆ points ∧ subset.card = k ∧
    (∀ (p q ∈ subset), p ≠ q → (dist p q ≤ 2 ∨ 1 < dist p q))) 
    ) ∧ n = k^2 - 2*k + 2 := 
begin
  sorry
end

end smallest_n_condition_l358_358177


namespace log3_monotonic_increasing_l358_358778

theorem log3_monotonic_increasing (a b : ℝ) (h : 0 < a ∧ 0 < b ∧ a < b) : 
  Real.log 3 a < Real.log 3 b :=
sorry

end log3_monotonic_increasing_l358_358778


namespace num_intersections_eq_two_l358_358685

noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem num_intersections_eq_two : 
  {x : ℝ | f x = g x}.finite ∧ {x : ℝ | f x = g x}.to_finset.card = 2 :=
by 
  sorry

end num_intersections_eq_two_l358_358685


namespace tiles_needed_l358_358635

def ft_to_inch (x : ℕ) : ℕ := x * 12

def height_ft : ℕ := 10
def length_ft : ℕ := 15
def tile_size_sq_inch : ℕ := 1

def height_inch : ℕ := ft_to_inch height_ft
def length_inch : ℕ := ft_to_inch length_ft
def area_sq_inch : ℕ := height_inch * length_inch

theorem tiles_needed : 
  height_ft = 10 ∧ length_ft = 15 ∧ tile_size_sq_inch = 1 →
  area_sq_inch = 21600 :=
by
  intro h
  exact sorry

end tiles_needed_l358_358635


namespace clock_resale_price_l358_358811

theorem clock_resale_price
    (C : ℝ)  -- original cost of the clock to the store
    (H1 : 0.40 * C = 100)  -- condition: difference between original cost and buy-back price is $100
    (H2 : ∀ (C : ℝ), resell_price = 1.80 * (0.60 * C))  -- store sold the clock again with a 80% profit on buy-back
    : resell_price = 270 := 
by
  sorry

end clock_resale_price_l358_358811


namespace eccentricity_of_ellipse_l358_358261

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0 ∧ 2 * (sqrt (a^2 - b^2)) * a = b^2) : ℝ :=
  if h_cond : a > b ∧ b > 0 ∧ 2 * (sqrt (a^2 - b^2)) * a = b^2 then sqrt 2 - 1 else 0

theorem eccentricity_of_ellipse :
  ∀ {a b : ℝ}, a > b → b > 0 → (∃ c, c = sqrt (a^2 - b^2) ∧ 2 * c * a = b^2) →
  ellipse_eccentricity a b (and.intro ‹a > b› (and.intro ‹b > 0› ‹∃ c, c = sqrt (a^2 - b^2) ∧ 2 * c * a = b^2›)) = sqrt 2 - 1 :=
by
  intros a b a_gt_b b_gt_0 ⟨c, c_eq, cond⟩
  unfold ellipse_eccentricity
  simp only [if_pos, and.intro]
  intro h
  apply and.intro
  assumption
  apply and.intro
  assumption
  have c_ineq : c = sqrt (a^2 - b^2), from c_eq
  have cond_rearranged : 2 * c * a = b^2, from cond
  apply exists.intro c
  apply and.intro <;> assumption
  sorry

end eccentricity_of_ellipse_l358_358261


namespace smallest_number_digit_sum_2017_l358_358918

/-- 
  Prove that the smallest natural number whose digit sum is 2017 has 
  its first digit (from the left) multiplied by the number of its digits equal to 225.
-/
theorem smallest_number_digit_sum_2017 :
  ∃ (n : ℕ), (nat.digits 10 n).sum = 2017 ∧ (nat.digits 10 n).head! * (nat.digits 10 n).length = 225 :=
sorry

end smallest_number_digit_sum_2017_l358_358918


namespace slope_of_asymptotes_l358_358916

noncomputable def hyperbola_asymptote_slope (x y : ℝ) : Prop :=
  (x^2 / 144 - y^2 / 81 = 1)

theorem slope_of_asymptotes (x y : ℝ) (h : hyperbola_asymptote_slope x y) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -3 / 4 :=
sorry

end slope_of_asymptotes_l358_358916


namespace solve_for_x_l358_358233

noncomputable def log_2 : ℝ := 0.3010
noncomputable def log_3 : ℝ := 0.4771
noncomputable def value_of_x : ℝ := 1.18

theorem solve_for_x (x : ℝ) (h1 : log_3 * (x + 3) = log_3 * 2 + log 11) : x = value_of_x :=
by
  sorry

end solve_for_x_l358_358233


namespace tetrahedron_inradii_sum_l358_358173

-- Definitions based on problem conditions
def is_tetrahedron (A B C D : Type) : Prop :=
  -- Add necessary conditions for A B C D forming a tetrahedron
  sorry -- placeholder for tetrahedron definition

def sum_opposite_edges (A B C D : Type) (edge_length : A × B → ℝ) : Prop :=
  -- Condition: sum of lengths of any two opposite edges is 1
  -- Implement specifics here
  sorry -- placeholder for edge sum condition

def inradii_sum_leq (r_A r_B r_C r_D : ℝ) : Prop :=
  r_A + r_B + r_C + r_D ≤ sqrt 3 / 3

def is_regular_tetrahedron (A B C D : Type) : Prop :=
  -- Implement conditions for a tetrahedron being regular
  sorry -- placeholder for regular tetrahedron definition

-- The theorem statement
theorem tetrahedron_inradii_sum (A B C D : Type) (edge_length : A × B → ℝ)
  (r_A r_B r_C r_D : ℝ) (h_tetra : is_tetrahedron A B C D)
  (h_edge_sum : sum_opposite_edges A B C D edge_length) :
  inradii_sum_leq r_A r_B r_C r_D :=
sorry

end tetrahedron_inradii_sum_l358_358173


namespace find_w_l358_358483

theorem find_w (w : ℝ) (h : 10^3 * 10^w = 1000) : w = 0 := by
  sorry

end find_w_l358_358483


namespace arithmetic_sequence_formula_product_sequence_sum_l358_358174

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (sum_a : ℕ → ℕ) 
  (h1 : a 5 = 5) 
  (h2 : sum_a 7 = 28)
  (h3 : ∀ n, sum_a n = n * (a 1) + n * (n - 1) * (a 2 - a 1) / 2) :
  ∀ n, a n = n := 
by sorry

theorem product_sequence_sum (a b : ℕ → ℕ) (sum_a sum_b sum_ab : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : sum_b = λ n, 2^n)
  (h3 : ∀ n, b n = ite (n = 1) 2 (2^(n-1))) 
  (h4 : ∀ n, sum_ab n = (n-1) * 2^n + 2) :
  ∀ n, sum_ab n = (n-1) * 2^n + 2 :=
by sorry

end arithmetic_sequence_formula_product_sequence_sum_l358_358174


namespace identify_variables_l358_358663

def temperature (T : ℝ) (d : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, ∀ d, T = f(d) ∧ ∀ d₁ d₂, d₁ < d₂ → f(d₁) < f(d₂)

/-- Given that the temperature T rises with the date d, prove that the independent variable 
    is the date d and the dependent variable is the temperature T. -/
theorem identify_variables (T : ℝ) (d : ℝ) (h : temperature T d) :
  (independent_variable : ℝ) × (dependent_variable : ℝ) :=
by
  let independent_variable := d
  let dependent_variable := T
  exact (independent_variable, dependent_variable)
  sorry

end identify_variables_l358_358663


namespace cone_in_sphere_less_half_volume_l358_358463

theorem cone_in_sphere_less_half_volume
  (R r m : ℝ)
  (h1 : m < 2 * R)
  (h2 : r <= R) :
  (1 / 3 * Real.pi * r^2 * m < 1 / 2 * 4 / 3 * Real.pi * R^3) :=
by
  sorry

end cone_in_sphere_less_half_volume_l358_358463


namespace problem1_problem2_l358_358166

-- Problem 1: Prove w == -1 - i given w = z^2 + 3 * conjugate(z) - 4 and z = 1 + i
theorem problem1 (z : ℤ) (w : ℤ) (hz : z = 1 + I) (hw : w = z^2 + 3 * conj z - 4) : w = -1 - I :=
by sorry

-- Problem 2: Prove a == -1 and b == 2 given (z^2 + az + b) / (z^2 - z + 1) = 1 - i and z = 1 + i
theorem problem2 (z : ℤ) (a b : ℤ) (hz : z = 1 + I) 
  (h_eq : (z^2 + a * z + b) / (z^2 - z + 1) = 1 - I) : a = -1 ∧ b = 2 :=
by sorry

end problem1_problem2_l358_358166


namespace find_quadruples_l358_358488

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l358_358488


namespace height_of_parallelogram_l358_358145

-- Define the base of the parallelogram
def base : ℝ := 24
-- Define the area of the parallelogram
def area : ℝ := 384

-- Define the function to compute the height
def height (A b : ℝ) : ℝ := A / b

-- State the proposition that the height of the parallelogram is 16 cm
theorem height_of_parallelogram : height area base = 16 := 
by {
  sorry -- proof to be filled in
}

end height_of_parallelogram_l358_358145


namespace three_digit_even_numbers_count_l358_358716

theorem three_digit_even_numbers_count : 
  ∃ (n : ℕ), 
  n = 360 ∧ (∀ (numbers : list ℕ), 
  list.length numbers = 3 ∧ 
  (∀ x ∈ numbers, x ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ 
  (numbers.nth 2).get_or_else 0 % 2 = 0 ∧ 
  (list.nodup numbers) 
  → n = 360) := 
sorry

end three_digit_even_numbers_count_l358_358716


namespace sin3_sum_eq_zero_l358_358609

theorem sin3_sum_eq_zero (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 60):
  sin (3 * A) + sin (3 * B) + sin (3 * C) = 0 :=
by sorry

end sin3_sum_eq_zero_l358_358609


namespace gcd_36_54_l358_358731

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l358_358731


namespace inequality_solution_l358_358328

theorem inequality_solution (x : ℝ) : 
  (1/2)^(x^2 - 2 * x + 3) < (1/2)^(2 * x^2 + 3 * x - 3) ↔ -6 < x ∧ x < 1 := sorry

end inequality_solution_l358_358328


namespace find_f_prime_at_2_l358_358545

-- Define the function f based on the given condition
def f (x : ℝ) : ℝ := 2 * x * f' 2 + x^3

-- Assume the condition and state the desired equality
theorem find_f_prime_at_2 : (2 : ℝ) * f'(2) + (2 ^ 3) = (f' 2 : ℝ) → f' 2 = -12 :=
by sorry

end find_f_prime_at_2_l358_358545


namespace sum_of_digits_base2_315_l358_358754

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358754


namespace largest_of_five_consecutive_composite_integers_under_40_l358_358505

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l358_358505


namespace lemonade_price_hot_day_l358_358651

theorem lemonade_price_hot_day
  (profit : ℝ := 200)
  (cost_per_cup : ℝ := 0.75)
  (cups_per_day : ℕ := 32)
  (total_days : ℕ := 10)
  (hot_days : ℕ := 4)
  (total_cost : ℝ := 10 * 32 * 0.75)
  (P : ℝ) -- regular price of 1 cup of lemonade
  (P_hot : ℝ := 1.25 * P) -- price of 1 cup on a hot day
  (revenue_regular : ℝ := 6 * 32 * P)
  (revenue_hot : ℝ := 4 * 32 * P_hot)
  (total_revenue : ℝ := revenue_regular + revenue_hot) :
  (total_revenue - total_cost = profit) →
  (P = 1.25) →
  (P_hot = 1.5625) := by
  intros h1 h2
  exact h2

end lemonade_price_hot_day_l358_358651


namespace jane_output_increase_l358_358787

noncomputable def output_increase_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) : ℝ :=
  ((1.8B / (0.9H)) - (B / H)) / (B / H) * 100

theorem jane_output_increase (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  output_increase_with_assistant B H hB hH = 100 :=
by
  -- Proof to be filled in
  sorry

end jane_output_increase_l358_358787


namespace sum_of_digits_of_binary_315_is_6_l358_358746
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l358_358746


namespace exp_function_unique_range_of_x_l358_358546

-- Definition of the exponential function passing through (2, 4)
def exp_function (f : ℝ → ℝ) := ∀ x, f x = 2 ^ x

-- Given that f(x) the graph passes through the point (2, 4), prove that f(x) = 2^x.
theorem exp_function_unique (f : ℝ → ℝ) (h : f 2 = 4) : exp_function f := by
  sorry

-- Given that f(x) = 2^x and f(x - 1) < 1, prove that x ∈ (-∞, 1).
theorem range_of_x (f : ℝ → ℝ) (h : exp_function f) (hx : f (x - 1) < 1) : x ∈ Iio 1 := by
  sorry

end exp_function_unique_range_of_x_l358_358546


namespace min_weighings_to_find_heaviest_l358_358518

-- Given conditions
variable (n : ℕ) (hn : n > 2)
variables (coins : Fin n) -- Representing coins with distinct masses
variables (scales : Fin n) -- Representing n scales where one is faulty

-- Theorem statement: Minimum number of weighings to find the heaviest coin
theorem min_weighings_to_find_heaviest : ∃ m, m = 2 * n - 1 := 
by
  existsi (2 * n - 1)
  rfl

end min_weighings_to_find_heaviest_l358_358518


namespace smallest_n_satisfying_condition_l358_358388

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n = 250001) ∧ (∀ m : ℕ, m < n → (sqrt m - sqrt (m - 1) < 0.001) = false) :=
by
  sorry

end smallest_n_satisfying_condition_l358_358388


namespace projectile_area_eq_l358_358084

-- Define the parameters and the given conditions
variables (k u g : ℝ) (θ : ℝ)
axiom k_pos : 0 < k
axiom k_lt_one : k < 1

-- Given parametric equations
def x := λ (t : ℝ), k * u * t * cos θ
def y := λ (t : ℝ), k * u * t * sin θ - (1 / 2) * g * t^2

-- Statement to prove the area of the closed curve
theorem projectile_area_eq : 
  ∃ (A : ℝ), (A = (π / 8) * ((k * u)^4 / g^2)) :=
sorry

end projectile_area_eq_l358_358084


namespace find_number_l358_358058

theorem find_number (x : ℝ) (h : 0.2 * x = 0.3 * 120 + 80) : x = 580 :=
by
  sorry

end find_number_l358_358058


namespace ratio_lateral_surface_area_cone_to_base_area_l358_358067

-- Define the conditions
def angle_PSR : ℝ := 90
def angle_SQR : ℝ := 45
def angle_PSQ : ℝ := 105

-- Define the statement
theorem ratio_lateral_surface_area_cone_to_base_area (A : ℝ) :
  let
    -- Define the given angles
    α := angle_PSR,
    β := angle_SQR,
    γ := angle_PSQ
  in
  (α = 90) ∧ (β = 45) ∧ (γ = 105) →
  -- Conclusion: The ratio of lateral surface area to base area is A
  ∃ (ratio : ℝ), ratio = A :=
by
  sorry

end ratio_lateral_surface_area_cone_to_base_area_l358_358067


namespace problem1_problem2_l358_358420

-- Defining the given function
def f (α : ℝ) : ℝ :=
  (sin (α + 3 * Real.pi / 2) * sin (-α + Real.pi) * cos (α + Real.pi / 2)) / 
  (cos (-α - Real.pi) * cos (α - Real.pi / 2) * tan (α + Real.pi)) 

-- The first proof problem stating that f(α) == -cos(α)
theorem problem1 (α : ℝ) : f α = -cos α :=
  sorry

-- The second proof problem stating that the given trigonometric expression is equal to -1
theorem problem2 : tan (675 * Real.pi / 180) + sin (-330 * Real.pi / 180) + cos (960 * Real.pi / 180) = -1 :=
  sorry

end problem1_problem2_l358_358420


namespace total_absent_students_l358_358309

-- Definitions from the conditions
def total_students : ℕ := 280
def absent_day3 : ℕ := total_students / 7
def absent_day2 : ℕ := 2 * absent_day3
def present_day1 : ℕ := total_students - absent_day2
def absent_day1 : ℕ := total_students - present_day1

-- Statement of the proof problem
theorem total_absent_students (T : ℕ) 
  (h1 : T = 280) 
  (h2 : ∀ (A3 : ℕ), A3 = T / 7) 
  (h3 : ∀ (A2 : ℕ), A2 = 2 * h2 A3) 
  (h4 : ∀ (P1 : ℕ), P1 = T - h3 A2) 
  (h5 : ∀ (A1 : ℕ), A1 = T - h4 P1) : 
  h5 A1 + h3 A2 + h2 A3 = 200 :=
sorry

end total_absent_students_l358_358309


namespace sum_of_digits_base2_315_l358_358761

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358761


namespace find_N_l358_358102

variable (N : ℚ)
variable (p : ℚ)

def ball_probability_same_color 
  (green1 : ℚ) (total1 : ℚ) 
  (green2 : ℚ) (blue2 : ℚ) 
  (p : ℚ) : Prop :=
  (green1/total1) * (green2 / (green2 + blue2)) + 
  ((total1 - green1) / total1) * (blue2 / (green2 + blue2)) = p

theorem find_N :
  p = 0.65 → 
  ball_probability_same_color 5 12 20 N p → 
  N = 280 / 311 := 
by
  sorry

end find_N_l358_358102


namespace gasoline_price_increase_l358_358344

theorem gasoline_price_increase :
  ∀ (p_low p_high : ℝ), p_low = 14 → p_high = 23 → 
  ((p_high - p_low) / p_low) * 100 = 64.29 :=
by
  intro p_low p_high h_low h_high
  rw [h_low, h_high]
  sorry

end gasoline_price_increase_l358_358344


namespace marble_arrangement_count_l358_358403
noncomputable def countValidMarbleArrangements : Nat := 
  let totalArrangements := 120
  let restrictedPairsCount := 24
  totalArrangements - restrictedPairsCount

theorem marble_arrangement_count :
  countValidMarbleArrangements = 96 :=
  by
    sorry

end marble_arrangement_count_l358_358403


namespace maximum_sum_in_three_circles_l358_358456

theorem maximum_sum_in_three_circles : ∃ S : ℕ, 
  (∀ A B C : set ℕ, 
     A ∩ B ∩ C = {7} ∧
     (A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7}) ∧
     (∀ a b c d e f g h i j k l m n o p q r s t u v w x y z, 
       A = {a, b, c, d} ∧ 
       B = {e, f, g, h} ∧
       C = {i, j, k, l} ∧
       (a + b + c + d = e + f + g + h) ∧
       (e + f + g + h = i + j + k + l) ∧
       A ∪ B ∪ C = {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z} →
   S = 19)) 

end maximum_sum_in_three_circles_l358_358456


namespace projection_vector_correct_l358_358569

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖v‖ = 1

def angle_between (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  real.arccos ((u ⬝ v) / (‖u‖ * ‖v‖))

theorem projection_vector_correct (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hangle : angle_between a b = π / 4) :
  let proj := (λ u v : EuclideanSpace ℝ (Fin 2), ((u ⬝ v) / ‖v‖^2) • v)
  in proj (a - b) b = ((-2 + sqrt 2) / 2) • b := by
sorry

end projection_vector_correct_l358_358569


namespace midpoint_param_l358_358311
-- Importing Mathlib

-- Defining the parametric equations and points B and C
def parametric_eqs (a b θ : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a + t * Real.cos θ, b + t * Real.sin θ)

-- Defining midpoint parameter calculation
theorem midpoint_param (a b θ t₁ t₂ : ℝ) :
  let B := parametric_eqs a b θ t₁
      C := parametric_eqs a b θ t₂
      M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  in (2 * (M.1 - a)) / Real.cos θ = t₁ + t₂ ∧
     (2 * (M.2 - b)) / Real.sin θ = t₁ + t₂ →
     (t₁ + t₂) / 2 = (t₁ + t₂) / 2 := 
sorry

end midpoint_param_l358_358311


namespace compare_game_A_and_C_l358_358812

-- Probability definitions for coin toss
def p_heads := 2/3
def p_tails := 1/3

-- Probability of winning Game A
def prob_win_A := (p_heads^3) + (p_tails^3)

-- Probability of winning Game C
def prob_win_C := (p_heads^3 + p_tails^3)^2

-- Theorem statement to compare chances of winning Game A to Game C
theorem compare_game_A_and_C : prob_win_A - prob_win_C = 2/9 := by sorry

end compare_game_A_and_C_l358_358812


namespace compute_y_geometric_series_l358_358466

theorem compute_y_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3)^n)
  let S2 := (∑' n : ℕ, (-1)^n * (1 / 3)^n)
  (S1 * S2 = ∑' n : ℕ, (1 / 9)^n) → 
  S1 = 3 / 2 →
  S2 = 3 / 4 →
  (∑' n : ℕ, (1 / y)^n) = 9 / 8 →
  y = 9 := 
by
  intros S1 S2 h₁ h₂ h₃ h₄
  sorry

end compute_y_geometric_series_l358_358466


namespace rational_function_nonnegative_l358_358136

noncomputable def rational_function (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3)

theorem rational_function_nonnegative :
  ∀ x, 0 ≤ x ∧ x < 3 → 0 ≤ rational_function x :=
sorry

end rational_function_nonnegative_l358_358136


namespace total_baseball_cards_l358_358636
-- Import the Mathlib library to bring all necessary mathematical structures

-- Declare the conditions of the problem
variable (n : ℕ) (cards_per_person : ℕ)
variable (p1 p2 p3 p4 p5 p6 : ℕ)

-- Assume each person has 8 baseball cards
axiom p1_cards : p1 = 8
axiom p2_cards : p2 = 8
axiom p3_cards : p3 = 8
axiom p4_cards : p4 = 8
axiom p5_cards : p5 = 8
axiom p6_cards : p6 = 8

-- Declare the total number of people
axiom num_people : n = 6

-- Declare the number of cards per person
axiom cards_each : cards_per_person = 8

-- Prove that the total number of baseball cards they have is 48
theorem total_baseball_cards : (p1 + p2 + p3 + p4 + p5 + p6) = 48 := by
  simp [p1_cards, p2_cards, p3_cards, p4_cards, p5_cards, p6_cards]
  -- Alternatively, you can also use
  -- exact 48
  sorry

end total_baseball_cards_l358_358636


namespace science_book_multiple_l358_358683

theorem science_book_multiple (history_pages novel_pages science_pages : ℕ)
  (H1 : history_pages = 300)
  (H2 : novel_pages = history_pages / 2)
  (H3 : science_pages = 600) :
  science_pages / novel_pages = 4 := 
by
  -- Proof will be filled out here
  sorry

end science_book_multiple_l358_358683


namespace sum_of_digits_base2_315_l358_358760

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l358_358760


namespace hyperbola_centered_at_origin_sharing_focus_with_ellipse_l358_358521

noncomputable def hyperbola_equation : Prop :=
  let a_ellipse := Real.sqrt 2
  let b_ellipse := 1
  let c := 1         -- Focus distance for both ellipse and hyperbola
  let e_ellipse := c / a_ellipse
  let e_hyperbola := Real.sqrt 2
  let a_hyperbola := c / e_hyperbola
  let b_hyperbola := Real.sqrt (c^2 - a_hyperbola^2)
  (2 * x^2 - 2 * y^2 = 1)

theorem hyperbola_centered_at_origin_sharing_focus_with_ellipse :
  hyperbola_equation := 
sorry

end hyperbola_centered_at_origin_sharing_focus_with_ellipse_l358_358521


namespace exists_real_number_A_l358_358889

-- Given the real number A = 2 + sqrt(3)
def A : ℝ := 2 + Real.sqrt 3

-- Main theorem stating that for any natural number n
-- the distance from the ceiling of A^n to the nearest square of an integer is 2
theorem exists_real_number_A :
  ∀ n : ℕ, ∃ k : ℤ, (Int.ceil (A ^ n) - k^2) = 2 :=
by sorry

end exists_real_number_A_l358_358889


namespace rope_for_second_post_l358_358115

theorem rope_for_second_post 
(r1 r2 r3 r4 : ℕ) 
(h_total : r1 + r2 + r3 + r4 = 70)
(h_r1 : r1 = 24)
(h_r3 : r3 = 14)
(h_r4 : r4 = 12) 
: r2 = 20 := 
by 
  sorry

end rope_for_second_post_l358_358115


namespace comparison_of_fractions_l358_358232

theorem comparison_of_fractions 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 2014 / 2015^2) 
  (hb : b = 2015 / 2016^2) 
  (hc : c = 2016 / 2017^2) : c < b < a := 
sorry

end comparison_of_fractions_l358_358232


namespace statement_II_must_be_true_l358_358444

-- Defining the facts related to the problem
variable (d : ℕ)
variable (H_statements : (d = 5 ∨ d ≠ 6 ∨ d = 7 ∨ d ≠ 8 ∨ d ≠ 9))

-- The Lean statement to prove
theorem statement_II_must_be_true (H : (d = 5 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) ∨ 
                                  (d ≠ 5 ∧ d ≠ 6 ∧ d = 7 ∧ d ≠ 8 ∧ d ≠ 9) ∨ 
                                  (H_statements ∧ (d ≠ 5 ∨ d ≠ 7))) :
  d ≠ 6 :=
sorry

end statement_II_must_be_true_l358_358444


namespace cargo_loaded_in_bahamas_l358_358087

def initial : ℕ := 5973
def final : ℕ := 14696
def loaded : ℕ := final - initial

theorem cargo_loaded_in_bahamas : loaded = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l358_358087


namespace apex_angle_of_cones_l358_358373

theorem apex_angle_of_cones :
  ∃ α : ℝ, (∀ (cone : ℕ), cone = 1 ∨ cone = 2 → apex_angle cone = 2 * α ) ∧ 
  ∃ γ β : ℝ, (γ = π / 3) ∧ (β = 5 * π / 12) ∧ 
  ∀ (fourth_cone : ℕ), apex_angle fourth_cone = 5 * π / 6 →
  2 * real.arctan (real.sqrt 3 - 1) = apex_angle 1 :=
by
  sorry

end apex_angle_of_cones_l358_358373


namespace center_of_hyperbola_l358_358142

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l358_358142


namespace regression_equation_estimation_at_40_percent_l358_358664

-- Define the necessary constants and sums.
def k : ℕ := 10
def x_bar : ℝ := 0.69
def y_bar : ℝ := 0.28
def sum_xi_yi : ℝ := 1.9951
def sum_xi_squared : ℝ := 4.9404

-- Define the regression coefficients using the provided empirical formulae.
noncomputable def b : ℝ := 
  (sum_xi_yi - k * x_bar * y_bar) / (sum_xi_squared - (k * x_bar^2))

noncomputable def a : ℝ := 
  y_bar - b * x_bar

-- Define the regression function.
def regression_line (x : ℝ) : ℝ :=
  b * x + a

-- Prove the specific results requested: the coefficients and the estimation.
theorem regression_equation :
  b ≈ 0.35 ∧ a ≈ 0.04 :=
by
  sorry

theorem estimation_at_40_percent :
  regression_line 0.4 ≈ 0.18 :=
by
  sorry

end regression_equation_estimation_at_40_percent_l358_358664


namespace card_numbers_ratio_2018_l358_358004

theorem card_numbers_ratio_2018:
  ∃ (digits : Fin 20 → ℕ),
    (∀ n, digits n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
    (∀ d, ∃! (m₁ m₂ : Fin 20), digits m₁ = d ∧ digits m₂ = d) ∧
    ∃ (m₁ : Fin 20), digits m₁ = 1 ∧
    (∃ (a b : ℕ), 
      (∃ (cards : Fin 19 → ℕ),
        (∀ n, cards n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
        ∀ n, ∃! (k : Fin 20), digits k = cards n) ∧
      a + b = cards_sum cards ∧
      a = 2018 * b) → False :=
by 
  sorry

end card_numbers_ratio_2018_l358_358004


namespace sum_of_digits_base2_315_l358_358759

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l358_358759


namespace comb_23_5_eq_33649_l358_358535

theorem comb_23_5_eq_33649 :
  (∃ (c21_3 c21_4 c21_5 : ℕ), c21_3 = 1330 ∧ c21_4 = 5985 ∧ c21_5 = 20349) →
  (nat.choose 23 5 = 33649) :=
by
  intro h
  obtain ⟨c21_3, c21_4, c21_5, h1, h2, h3⟩ := h
  have h4 : nat.choose 21 3 = c21_3 := h1
  have h5 : nat.choose 21 4 = c21_4 := h2
  have h6 : nat.choose 21 5 = c21_5 := h3
  sorry

end comb_23_5_eq_33649_l358_358535


namespace triangle_circle_sum_l358_358844

theorem triangle_circle_sum :
  ∃ Δ ∘ : ℚ, (Δ + 3 * ∘ = 18) ∧ (2 * Δ + ∘ = 14) ∧ (Δ + 2 * ∘ = 68 / 5) :=
by
  use [24 / 5, 22 / 5]
  split
  · calc
      (24 / 5) + 3 * (22 / 5) = (24 / 5) + 66 / 5 := by simp
                        ... = 18 := by norm_num
  split
  · calc
      2 * (24 / 5) + (22 / 5) = 48 / 5 + 22 / 5 := by simp
                         ... = 70 / 5 := by norm_num
                         ... = 14 := by norm_num
  · calc
      (24 / 5) + 2 * (22 / 5) = (24 / 5) + 44 / 5 := by simp
                         ... = 68 / 5 := by norm_num

end triangle_circle_sum_l358_358844


namespace mark_study_hours_l358_358632

theorem mark_study_hours (h1 h2 h3 h4 : ℕ) (avg : ℕ) (goal_avg : ℕ) :
  h1 = 10 → h2 = 14 → h3 = 9 → h4 = 13 → avg = 12 → goal_avg = 5 →
  (h1 + h2 + h3 + h4 + 14) / 5 = avg :=
by
  intros h1_eq h2_eq h3_eq h4_eq avg_eq goal_avg_eq
  -- Applying conditions directly
  rw [h1_eq, h2_eq, h3_eq, h4_eq]
  -- Plug in the average value and goal average duration
  rw [avg_eq, goal_avg_eq]
  -- Simplifying the arithmetic
  dsimp
  -- The final proof is simplified to basic arithmetic already evident from the conditions
  sorry

end mark_study_hours_l358_358632


namespace DG_HF_concyclic_l358_358283

variables (A B C D E F G H : Point)

-- Definitions and conditions as variables and hypotheses
def is_cyclic_quad (A B C D : Point) : Prop := -- definition of cyclic quadrilateral
sorry

def intersect_diag (A C B D E : Point) : Prop := -- diagonals AC and BD intersect at E
sorry

def intersect_dside (D A B C F : Point) : Prop := -- DA and BC intersect at F
sorry

def reflection (E A D H : Point) : Prop := -- H is reflection of E across AD
sorry

def form_parallelogram (D E C G : Point) : Prop := -- DECG forms a parallelogram
sorry

theorem DG_HF_concyclic 
  (h1 : is_cyclic_quad A B C D) 
  (h2 : intersect_diag A C B D E)
  (h3 : intersect_dside D A B C F)
  (h4 : reflection E A D H)
  (h5 : form_parallelogram D E C G) : 
  concyclic D G H F :=
sorry

end DG_HF_concyclic_l358_358283


namespace giant_kite_area_72_l358_358153

-- Definition of the vertices of the medium kite
def vertices_medium_kite : List (ℕ × ℕ) := [(1,6), (4,9), (7,6), (4,1)]

-- Given condition function to check if the giant kite is created by doubling the height and width
def double_coordinates (c : (ℕ × ℕ)) : (ℕ × ℕ) := (2 * c.1, 2 * c.2)

def vertices_giant_kite : List (ℕ × ℕ) := vertices_medium_kite.map double_coordinates

-- Function to calculate the area of the kite based on its vertices
def kite_area (vertices : List (ℕ × ℕ)) : ℕ := sorry -- The way to calculate the kite area can be complex

-- Theorem to prove the area of the giant kite
theorem giant_kite_area_72 :
  kite_area vertices_giant_kite = 72 := 
sorry

end giant_kite_area_72_l358_358153


namespace count_integers_between_200_and_400_with_digit_sum_14_l358_358977

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem count_integers_between_200_and_400_with_digit_sum_14 : 
  (finset.filter (λ n, sum_of_digits n = 14)
  (finset.Icc 200 400)).card = 15 := 
by
  sorry

end count_integers_between_200_and_400_with_digit_sum_14_l358_358977


namespace spadesuit_example_l358_358470

def spadesuit (a b : ℕ) : ℕ := abs (a - b)

theorem spadesuit_example : 5 * (spadesuit 2 (spadesuit 6 9)) = 5 :=
by
  sorry

end spadesuit_example_l358_358470


namespace range_of_a_l358_358963

variables (a : ℝ)

def p : Prop := (a - 2) * (6 - a) > 0
def q : Prop := 4 - a > 1
def r : Set ℝ := {a | a ≤ 2 ∨ 3 ≤ a ∧ a < 6}

theorem range_of_a : (p ∨ q) ∧ ¬ (p ∧ q) → a ∈ r := 
by
  sorry

end range_of_a_l358_358963


namespace exists_logarithmic_base_l358_358216

variable {n : ℕ} (a1 q b1 d : ℝ)

-- Conditions:
-- (1) The sequences are geometric and arithmetic respectively
-- (2) d > 0

def geom_seq (n : ℕ) : ℝ := a1 * q^(n - 1)
def arith_seq (n : ℕ) : ℝ := b1 + d * (n - 1)
def exists_log_base (n : ℕ) (a1 q b1 d : ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, log k (geom_seq n) - arith_seq n = log k (geom_seq 1) - arith_seq 1

theorem exists_logarithmic_base
  (h_geom : ∀ n : ℕ, geom_seq n = a1 * q^(n - 1))
  (h_arith : ∀ n : ℕ, arith_seq n = b1 + d * (n - 1))
  (h_pos_d : d > 0) :
  exists_log_base n a1 q b1 d := 
sorry

end exists_logarithmic_base_l358_358216


namespace find_a_pow_b_l358_358532

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b_l358_358532


namespace infinite_isosceles_triangles_of_same_color_l358_358356

theorem infinite_isosceles_triangles_of_same_color
    (circle : Type)
    (points : set circle)
    (color : points → ℕ)
    (color_range : set.range color ⊆ {1, 2, 3})
    (is_circle : ∀ (x y : points), x ≠ y → ∃ (z : circle), z ≠ x ∧ z ≠ y ∧ z ∈ points ∧ collinear x.val y.val z) : 
  ∃ (tris : set (triangle circle)), (∀ t ∈ tris, is_isosceles t ∧ ∃ c ∈ {1, 2, 3}, ∀ v ∈ triangle.verts t, color v = c) ∧ ∀ n, ∃ t₁ t₂ ∈ tris, dist t₁ t₂ > n := 
by sorry

end infinite_isosceles_triangles_of_same_color_l358_358356


namespace irrational_sum_root_l358_358645

theorem irrational_sum_root
  (α : ℝ) (hα : Irrational α)
  (n : ℕ) (hn : 0 < n) :
  Irrational ((α + (α^2 - 1).sqrt)^(1/n : ℝ) + (α - (α^2 - 1).sqrt)^(1/n : ℝ)) := sorry

end irrational_sum_root_l358_358645


namespace tommy_initial_balloons_l358_358375

theorem tommy_initial_balloons (initial_balloons balloons_added total_balloons : ℝ)
  (h1 : balloons_added = 34.5)
  (h2 : total_balloons = 60.75)
  (h3 : total_balloons = initial_balloons + balloons_added) :
  initial_balloons = 26.25 :=
by sorry

end tommy_initial_balloons_l358_358375


namespace real_part_of_z_l358_358193

theorem real_part_of_z (z : ℂ) (h : (1 - complex.i) * z = 2) : complex.re z = 1 := 
sorry

end real_part_of_z_l358_358193


namespace solution_of_system_l358_358301

variables (a b c : ℝ) (n : ℕ) (x : fin n → ℝ)

def system_of_equations := ∀ i : fin n, a * (x i)^2 + b * (x i) + c = x (i + 1)

def Delta := (b - 1)^2 - 4 * a * c

theorem solution_of_system (a_ne_zero : a ≠ 0) (sys : system_of_equations a b c n x) :
  (Delta a b c < 0 → ¬∃ x : fin n → ℝ, system_of_equations a b c n x) ∧
  (Delta a b c = 0 → ∃! x : fin n → ℝ, system_of_equations a b c n x) ∧
  (Delta a b c > 0 → ∃ x : fin n → ℝ, (system_of_equations a b c n x ∧ (∃ i, x i ≠ x 0))) :=
by
  sorry

end solution_of_system_l358_358301


namespace time_per_student_l358_358575

-- Given Conditions
def total_students : ℕ := 18
def groups : ℕ := 3
def minutes_per_group : ℕ := 24

-- Mathematical proof problem
theorem time_per_student :
  (minutes_per_group / (total_students / groups)) = 4 := by
  -- Proof not required, adding placeholder
  sorry

end time_per_student_l358_358575


namespace impossible_to_form_3x3_in_upper_left_or_right_l358_358517

noncomputable def initial_positions : List (ℕ × ℕ) := 
  [(6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3)]

def sum_vertical (positions : List (ℕ × ℕ)) : ℕ :=
  positions.foldr (λ pos acc => pos.1 + acc) 0

theorem impossible_to_form_3x3_in_upper_left_or_right
  (initial_positions_set : List (ℕ × ℕ) := initial_positions)
  (initial_sum := sum_vertical initial_positions_set)
  (target_positions_upper_left : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
  (target_positions_upper_right : List (ℕ × ℕ) := [(1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8)])
  (target_sum_upper_left := sum_vertical target_positions_upper_left)
  (target_sum_upper_right := sum_vertical target_positions_upper_right) : 
  ¬ (initial_sum % 2 = 1 ∧ target_sum_upper_left % 2 = 0 ∧ target_sum_upper_right % 2 = 0) := sorry

end impossible_to_form_3x3_in_upper_left_or_right_l358_358517


namespace cos_2pi_minus_alpha_tan_alpha_minus_7pi_l358_358514

open Real

variables (α : ℝ)
variables (h1 : sin (π + α) = -1 / 3) (h2 : π / 2 < α ∧ α < π)

-- Statement for the problem (Ⅰ)
theorem cos_2pi_minus_alpha :
  cos (2 * π - α) = -2 * sqrt 2 / 3 :=
sorry

-- Statement for the problem (Ⅱ)
theorem tan_alpha_minus_7pi :
  tan (α - 7 * π) = -sqrt 2 / 4 :=
sorry

end cos_2pi_minus_alpha_tan_alpha_minus_7pi_l358_358514


namespace problem_statement_l358_358038

theorem problem_statement :
  ¬(∀ n : ℤ, n ≥ 0 → n = 0) ∧
  ¬(∀ q : ℚ, q ≠ 0 → q > 0 ∨ q < 0) ∧
  ¬(∀ a b : ℝ, abs a = abs b → a = b) ∧
  (∀ a : ℝ, abs a = abs (-a)) :=
by
  sorry

end problem_statement_l358_358038


namespace tom_initial_balloons_l358_358374

noncomputable def initial_balloons (x : ℕ) : ℕ :=
  if h₁ : x % 2 = 1 ∧ (x / 3) + 10 = 45 then x else 0

theorem tom_initial_balloons : initial_balloons 105 = 105 :=
by {
  -- Given x is an odd number and the equation (x / 3) + 10 = 45 holds, prove x = 105.
  -- These conditions follow from the problem statement directly.
  -- Proof is skipped.
  sorry
}

end tom_initial_balloons_l358_358374


namespace sam_distance_walked_l358_358785

variable (d : ℝ := 40) -- initial distance between Fred and Sam
variable (v_f : ℝ := 4) -- Fred's constant speed in miles per hour
variable (v_s : ℝ := 4) -- Sam's constant speed in miles per hour

theorem sam_distance_walked :
  (d / (v_f + v_s)) * v_s = 20 :=
by
  sorry

end sam_distance_walked_l358_358785


namespace part_I_part_II_l358_358204
noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem part_I (h : ∀ x > 1, f'(-2) x > 0) : 
  ∀ x > 1, f'(-2) x > 0 := 
sorry

theorem part_II (a : ℝ) (h_pos : a ≥ -2 ∨ (-2 * exp 2 < a ∧ a < -2) ∨ a ≤ -2 * exp 2) :
  (∃ m x : ℝ, 
    (a ≥ -2 ∧ m = 1 ∧ x = 1) ∨ 
    ((-2 * exp 2 < a ∧ a < -2) ∧ m = (a / 2) * log (-(a / 2)) - (a / 2) ∧ x = sqrt (-(a / 2))) ∨ 
    (a ≤ -2 * exp 2 ∧ m = a + exp 2 ∧ x = exp 1)) :=
sorry

#check part_I
#check part_II

end part_I_part_II_l358_358204


namespace max_alpha_red_squares_l358_358529

theorem max_alpha_red_squares (a b N : ℕ) (ha : a > 0) (hb : b > 0) (hab : a < b ∧ b < 2 * a)
  (H : ∀ (x y : ℕ), (x = a ∧ y = b ∨ x = b ∧ y = a) → (∃ (i j : ℕ), i < x ∧ j < y ∧ red_square (i, j))) :
  ∃ α : ℚ, α = (N : ℝ) / ((N - 1) : ℝ) ∧ (∃ s : ℚ, s = α * (N^2 : ℚ) ∧ ∀ (x y : ℕ), (x = N ∧ y = N)
      → ∃ (i j : ℕ), i < x ∧ j < y ∧ red_square (i, j)) :=
by sorry

end max_alpha_red_squares_l358_358529


namespace hyperbola_ellipse_foci_product_l358_358069

theorem hyperbola_ellipse_foci_product
  (m n a b : ℝ)
  (h_m : 0 < m)
  (h_n : 0 < n)
  (h_a : 0 < a)
  (h_b : 0 < b)
  (h_a_gt_b : a > b)
  (foci_shared : ℝ × ℝ)
  (M : ℝ × ℝ)
  (h_Hyperbola : (M.1 ^ 2) / m - (M.2 ^ 2) / n = 1)
  (h_Ellipse : (M.1 ^ 2) / a + (M.2 ^ 2) / b = 1) :
  (dist M foci_shared.1) * (dist M foci_shared.2) = a - m := 
sorry

end hyperbola_ellipse_foci_product_l358_358069


namespace greatest_k_dividing_n_l358_358440

theorem greatest_k_dividing_n (n : ℕ) (k : ℕ) :
  (∃ m : ℕ, n = 7^k * m ∧ (∀ p : ℕ, prime p → p ∣ m → ¬(p = 7))) ∧
  (nat.factors.count n = 72) ∧
  (nat.factors.count (7 * n) = 90) →
  k = 3 :=
by
  sorry

end greatest_k_dividing_n_l358_358440


namespace gcd_36_54_l358_358720

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l358_358720


namespace inequality_one_solution_inequality_two_solution_l358_358503

theorem inequality_one_solution (x : ℝ) :
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3 / 2) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x + 1) / (x - 2) ≤ 2 ↔ (x < 2 ∨ x ≥ 5) :=
sorry

end inequality_one_solution_inequality_two_solution_l358_358503


namespace square_distance_B_to_center_is_50_l358_358814

noncomputable def distance_square_from_B_to_center 
  (radius : ℝ := real.sqrt 98)
  (AB BC : ℝ := 8)
  (angle_ABC_right : ∀ (A B C : ℝ × ℝ), B - A = (0, 8) ∧ C - B = (3, 0) ∧ angle A B C = real.pi / 2) : 
  ℝ :=
  50

theorem square_distance_B_to_center_is_50 : 
  distance_square_from_B_to_center = 50 := 
sorry

end square_distance_B_to_center_is_50_l358_358814


namespace initial_oranges_l358_358478

theorem initial_oranges (x : ℕ) (additional_oranges : ℕ) (total_oranges : ℕ) (h₁ : additional_oranges = 5) (h₂ : total_oranges = 9) (h₃ : x + additional_oranges = total_oranges) : x = 4 :=
by
  subst h₁
  subst h₂
  rw [nat.add_comm x 5, nat.add_sub_cancel_left]
  exact h₃

end initial_oranges_l358_358478


namespace am_gm_inequality_l358_358319

theorem am_gm_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - (Real.sqrt (a * b)) ∧ 
  (a + b) / 2 - (Real.sqrt (a * b)) < (a - b)^2 / (8 * b) := 
sorry

end am_gm_inequality_l358_358319


namespace remainder_sum_1_to_12_div_9_l358_358740

-- Define the sum of the first n natural numbers
def sum_natural (n : Nat) : Nat := n * (n + 1) / 2

-- Define the sum of the numbers from 1 to 12
def sum_1_to_12 := sum_natural 12

-- Define the remainder function
def remainder (a b : Nat) : Nat := a % b

-- Prove that the remainder when the sum of the numbers from 1 to 12 is divided by 9 is 6
theorem remainder_sum_1_to_12_div_9 : remainder sum_1_to_12 9 = 6 := by
  sorry

end remainder_sum_1_to_12_div_9_l358_358740
