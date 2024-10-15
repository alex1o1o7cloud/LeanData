import Mathlib

namespace NUMINAMATH_GPT_triangle_projection_inequality_l2390_239069

variable (a b c t r μ : ℝ)
variable (h1 : AC_1 = 2 * t * AB)
variable (h2 : BA_1 = 2 * r * BC)
variable (h3 : CB_1 = 2 * μ * AC)
variable (h4 : AB = c)
variable (h5 : AC = b)
variable (h6 : BC = a)

theorem triangle_projection_inequality
  (h1 : AC_1 = 2 * t * AB)  -- condition AC_1 = 2t * AB
  (h2 : BA_1 = 2 * r * BC)  -- condition BA_1 = 2r * BC
  (h3 : CB_1 = 2 * μ * AC)  -- condition CB_1 = 2μ * AC
  (h4 : AB = c)             -- side AB
  (h5 : AC = b)             -- side AC
  (h6 : BC = a)             -- side BC
  : (a^2 / b^2) * (t / (1 - 2 * t))^2 
  + (b^2 / c^2) * (r / (1 - 2 * r))^2 
  + (c^2 / a^2) * (μ / (1 - 2 * μ))^2 
  + 16 * t * r * μ ≥ 1 := 
  sorry

end NUMINAMATH_GPT_triangle_projection_inequality_l2390_239069


namespace NUMINAMATH_GPT_marcus_batches_l2390_239004

theorem marcus_batches (B : ℕ) : (5 * B = 35) ∧ (35 - 8 = 27) → B = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_marcus_batches_l2390_239004


namespace NUMINAMATH_GPT_loads_ratio_l2390_239024

noncomputable def loads_wednesday : ℕ := 6
noncomputable def loads_friday (T : ℕ) : ℕ := T / 2
noncomputable def loads_saturday : ℕ := loads_wednesday / 3
noncomputable def total_loads_week (T : ℕ) : ℕ := loads_wednesday + T + loads_friday T + loads_saturday

theorem loads_ratio (T : ℕ) (h : total_loads_week T = 26) : T / loads_wednesday = 2 := 
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_loads_ratio_l2390_239024


namespace NUMINAMATH_GPT_find_letter_l2390_239045

def consecutive_dates (A B C D E F G : ℕ) : Prop :=
  B = A + 1 ∧ C = A + 2 ∧ D = A + 3 ∧ E = A + 4 ∧ F = A + 5 ∧ G = A + 6

theorem find_letter (A B C D E F G : ℕ) 
  (h_consecutive : consecutive_dates A B C D E F G) 
  (h_condition : ∃ y, (B + y = 2 * A + 6)) :
  y = F :=
by
  sorry

end NUMINAMATH_GPT_find_letter_l2390_239045


namespace NUMINAMATH_GPT_train_crosses_signal_pole_in_12_seconds_l2390_239020

noncomputable def time_to_cross_signal_pole (length_train : ℕ) (time_to_cross_platform : ℕ) (length_platform : ℕ) : ℕ :=
  let distance_train_platform := length_train + length_platform
  let speed_train := distance_train_platform / time_to_cross_platform
  let time_to_cross_pole := length_train / speed_train
  time_to_cross_pole

theorem train_crosses_signal_pole_in_12_seconds :
  time_to_cross_signal_pole 300 39 675 = 12 :=
by
  -- expected proof in the interactive mode
  sorry

end NUMINAMATH_GPT_train_crosses_signal_pole_in_12_seconds_l2390_239020


namespace NUMINAMATH_GPT_quadratic_vertex_l2390_239080

theorem quadratic_vertex (x : ℝ) :
  ∃ (h k : ℝ), (h = -3) ∧ (k = -5) ∧ (∀ y, y = -2 * (x + h) ^ 2 + k) :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_l2390_239080


namespace NUMINAMATH_GPT_number_of_students_only_taking_AMC8_l2390_239087

def total_Germain := 13
def total_Newton := 10
def total_Young := 12

def olympiad_Germain := 3
def olympiad_Newton := 2
def olympiad_Young := 4

def number_only_AMC8 :=
  (total_Germain - olympiad_Germain) +
  (total_Newton - olympiad_Newton) +
  (total_Young - olympiad_Young)

theorem number_of_students_only_taking_AMC8 :
  number_only_AMC8 = 26 := by
  sorry

end NUMINAMATH_GPT_number_of_students_only_taking_AMC8_l2390_239087


namespace NUMINAMATH_GPT_number_exceeds_part_l2390_239005

theorem number_exceeds_part (x : ℝ) (h : x = (5 / 9) * x + 150) : x = 337.5 := sorry

end NUMINAMATH_GPT_number_exceeds_part_l2390_239005


namespace NUMINAMATH_GPT_num_five_digit_ints_l2390_239057

open Nat

theorem num_five_digit_ints : 
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  num_ways = 10 :=
by
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  sorry

end NUMINAMATH_GPT_num_five_digit_ints_l2390_239057


namespace NUMINAMATH_GPT_problem_equiv_l2390_239041

theorem problem_equiv (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4 * a + 5 > 0) ∧ (a^2 + b^2 ≥ 2 * (a - b - 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_equiv_l2390_239041


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l2390_239066

theorem sufficient_and_necessary_condition (a : ℝ) : 
  (0 < a ∧ a < 4) ↔ ∀ x : ℝ, (x^2 - a * x + a) > 0 :=
by sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l2390_239066


namespace NUMINAMATH_GPT_example_theorem_l2390_239095

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end NUMINAMATH_GPT_example_theorem_l2390_239095


namespace NUMINAMATH_GPT_system_of_equations_solution_l2390_239038

theorem system_of_equations_solution
  (a b c d e f g : ℝ)
  (x y z : ℝ)
  (h1 : a * x = b * y)
  (h2 : b * y = c * z)
  (h3 : d * x + e * y + f * z = g) :
  (x = g * b * c / (d * b * c + e * a * c + f * a * b)) ∧
  (y = g * a * c / (d * b * c + e * a * c + f * a * b)) ∧
  (z = g * a * b / (d * b * c + e * a * c + f * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2390_239038


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2390_239048

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, abs (x - 1) < 3 → (x + 2) * (x + a) < 0) ∧ 
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ ¬(abs (x - 1) < 3)) →
  a < -4 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2390_239048


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l2390_239028

/-- An infinite geometric series with common ratio -1/3 has a sum of 24.
    Prove that the first term of the series is 32. -/
theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 24) 
  (h3 : S = a / (1 - r)) : 
  a = 32 := 
sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l2390_239028


namespace NUMINAMATH_GPT_find_common_difference_l2390_239026

variable {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) = a n + d)
variable (a7_minus_2a4_eq_6 : a 7 - 2 * a 4 = 6)
variable (a3_eq_2 : a 3 = 2)

theorem find_common_difference (d : ℝ) : d = 4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_common_difference_l2390_239026


namespace NUMINAMATH_GPT_locus_of_midpoint_l2390_239047

theorem locus_of_midpoint {P Q M : ℝ × ℝ} (hP_on_circle : P.1^2 + P.2^2 = 13)
  (hQ_perpendicular_to_y_axis : Q.1 = P.1) (h_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1^2 / (13 / 4)) + (M.2^2 / 13) = 1 := 
sorry

end NUMINAMATH_GPT_locus_of_midpoint_l2390_239047


namespace NUMINAMATH_GPT_range_of_a_l2390_239055

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, a * x ^ 2 + 2 * a * x + 1 ≤ 0) →
  0 ≤ a ∧ a < 1 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_range_of_a_l2390_239055


namespace NUMINAMATH_GPT_find_S_30_l2390_239029

variable (S : ℕ → ℚ)
variable (a : ℕ → ℚ)
variable (d : ℚ)

-- Definitions based on conditions
def arithmetic_sum (n : ℕ) : ℚ := (n / 2) * (a 1 + a n)
def a_n (n : ℕ) : ℚ := a 1 + (n - 1) * d

-- Given conditions
axiom h1 : S 10 = 20
axiom h2 : S 20 = 15

-- Required Proof (the final statement to be proven)
theorem find_S_30 : S 30 = -15 := sorry

end NUMINAMATH_GPT_find_S_30_l2390_239029


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_x_eq_1_l2390_239090

theorem necessary_and_sufficient_condition_x_eq_1
    (x : ℝ) :
    (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_x_eq_1_l2390_239090


namespace NUMINAMATH_GPT_total_pencils_l2390_239065

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end NUMINAMATH_GPT_total_pencils_l2390_239065


namespace NUMINAMATH_GPT_circle_area_greater_than_hexagon_area_l2390_239097

theorem circle_area_greater_than_hexagon_area (h : ℝ) (r : ℝ) (π : ℝ) (sqrt3 : ℝ) (ratio : ℝ) : 
  (h = 1) →
  (r = sqrt3 / 2) →
  (π > 3) →
  (sqrt3 > 1.7) →
  (ratio = (π * sqrt3) / 6) →
  ratio > 0.9 :=
by
  intros h_eq r_eq pi_gt sqrt3_gt ratio_eq
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_circle_area_greater_than_hexagon_area_l2390_239097


namespace NUMINAMATH_GPT_minimize_theta_l2390_239003

theorem minimize_theta (K : ℤ) : ∃ θ : ℝ, -495 = K * 360 + θ ∧ |θ| ≤ 180 ∧ θ = -135 :=
by
  sorry

end NUMINAMATH_GPT_minimize_theta_l2390_239003


namespace NUMINAMATH_GPT_find_positive_n_for_quadratic_l2390_239002

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

-- Define the condition: the quadratic equation has exactly one real root if its discriminant is zero
def has_one_real_root (a b c : ℝ) : Prop := discriminant a b c = 0

-- The specific quadratic equation y^2 + 6ny + 9n
def my_quadratic (n : ℝ) : Prop := has_one_real_root 1 (6 * n) (9 * n)

-- The statement to be proven: for the quadratic equation y^2 + 6ny + 9n to have one real root, n must be 1
theorem find_positive_n_for_quadratic : ∃ (n : ℝ), my_quadratic n ∧ n > 0 ∧ n = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_positive_n_for_quadratic_l2390_239002


namespace NUMINAMATH_GPT_find_time_l2390_239093

variables (V V_0 S g C : ℝ) (t : ℝ)

-- Given conditions.
axiom eq1 : V = 2 * g * t + V_0
axiom eq2 : S = (1 / 3) * g * t^2 + V_0 * t + C * t^3

-- The statement to prove.
theorem find_time : t = (V - V_0) / (2 * g) :=
sorry

end NUMINAMATH_GPT_find_time_l2390_239093


namespace NUMINAMATH_GPT_range_of_c_l2390_239001

variable {a b c : ℝ} -- Declare the variables

-- Define the conditions
def triangle_condition (a b : ℝ) : Prop :=
|a + b - 4| + (a - b + 2)^2 = 0

-- Define the proof problem
theorem range_of_c {a b c : ℝ} (h : triangle_condition a b) : 2 < c ∧ c < 4 :=
sorry -- Proof to be completed

end NUMINAMATH_GPT_range_of_c_l2390_239001


namespace NUMINAMATH_GPT_find_circle_equation_l2390_239073

noncomputable def center (m : ℝ) := (3 * m, m)

def radius (m : ℝ) : ℝ := 3 * m

def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3 * m)^2 + (y - m)^2 = (radius m)^2

def point_A : ℝ × ℝ := (6, 1)

theorem find_circle_equation (m : ℝ) :
  (radius m = 3 * m ∧ center m = (3 * m, m) ∧ 
   point_A = (6, 1) ∧
   circle_eq m 6 1) →
  (circle_eq 1 x y ∨ circle_eq 37 x y) :=
by
  sorry

end NUMINAMATH_GPT_find_circle_equation_l2390_239073


namespace NUMINAMATH_GPT_specific_gravity_cylinder_l2390_239036

noncomputable def specific_gravity_of_cylinder (r m : ℝ) : ℝ :=
  (1 / 3) - (Real.sqrt 3 / (4 * Real.pi))

theorem specific_gravity_cylinder
  (r m : ℝ) 
  (cylinder_floats : r > 0 ∧ m > 0)
  (submersion_depth : r / 2 = r / 2) :
  specific_gravity_of_cylinder r m = 0.1955 :=
sorry

end NUMINAMATH_GPT_specific_gravity_cylinder_l2390_239036


namespace NUMINAMATH_GPT_triangular_prism_sliced_faces_l2390_239091

noncomputable def resulting_faces_count : ℕ :=
  let initial_faces := 5 -- 2 bases + 3 lateral faces
  let additional_faces := 3 -- from the slices
  initial_faces + additional_faces

theorem triangular_prism_sliced_faces :
  resulting_faces_count = 8 := by
  sorry

end NUMINAMATH_GPT_triangular_prism_sliced_faces_l2390_239091


namespace NUMINAMATH_GPT_highest_score_not_necessarily_12_l2390_239031

-- Define the structure of the round-robin tournament setup
structure RoundRobinTournament :=
  (teams : ℕ)
  (matches_per_team : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (points_draw : ℕ)

-- Tournament conditions
def tournament : RoundRobinTournament :=
  { teams := 12,
    matches_per_team := 11,
    points_win := 2,
    points_loss := 0,
    points_draw := 1 }

-- The statement we want to prove
theorem highest_score_not_necessarily_12 (T : RoundRobinTournament) :
  ∃ team_highest_score : ℕ, team_highest_score < 12 :=
by
  -- Provide a proof here
  sorry

end NUMINAMATH_GPT_highest_score_not_necessarily_12_l2390_239031


namespace NUMINAMATH_GPT_units_digit_29_pow_8_pow_7_l2390_239006

/-- The units digit of 29 raised to an arbitrary power follows a cyclical pattern. 
    For the purposes of this proof, we use that 29^k for even k ends in 1.
    Since 8^7 is even, we prove the units digit of 29^(8^7) is 1. -/
theorem units_digit_29_pow_8_pow_7 : (29^(8^7)) % 10 = 1 :=
by
  have even_power_cycle : ∀ k, k % 2 = 0 → (29^k) % 10 = 1 := sorry
  have eight_power_seven_even : (8^7) % 2 = 0 := by norm_num
  exact even_power_cycle (8^7) eight_power_seven_even

end NUMINAMATH_GPT_units_digit_29_pow_8_pow_7_l2390_239006


namespace NUMINAMATH_GPT_correct_properties_l2390_239078

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_properties_l2390_239078


namespace NUMINAMATH_GPT_kendall_tau_correct_l2390_239084

-- Base Lean setup and list of dependencies might go here

structure TestScores :=
  (A : List ℚ)
  (B : List ℚ)

-- Constants from the problem
def scores : TestScores :=
  { A := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
  , B := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70] }

-- Function to calculate the Kendall rank correlation coefficient
noncomputable def kendall_tau (scores : TestScores) : ℚ :=
  -- the method of calculating Kendall tau could be very complex
  -- hence we assume the correct coefficient directly for the example
  0.51

-- The proof problem
theorem kendall_tau_correct : kendall_tau scores = 0.51 :=
by
  sorry

end NUMINAMATH_GPT_kendall_tau_correct_l2390_239084


namespace NUMINAMATH_GPT_room_width_to_perimeter_ratio_l2390_239011

theorem room_width_to_perimeter_ratio (L W : ℕ) (hL : L = 25) (hW : W = 15) :
  let P := 2 * (L + W)
  let ratio := W / P
  ratio = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_room_width_to_perimeter_ratio_l2390_239011


namespace NUMINAMATH_GPT_power_of_negative_base_l2390_239052

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end NUMINAMATH_GPT_power_of_negative_base_l2390_239052


namespace NUMINAMATH_GPT_sqrt_16_eq_pm_4_l2390_239092

theorem sqrt_16_eq_pm_4 (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_GPT_sqrt_16_eq_pm_4_l2390_239092


namespace NUMINAMATH_GPT_minimize_quadratic_l2390_239023

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l2390_239023


namespace NUMINAMATH_GPT_greatest_integer_of_set_is_152_l2390_239081

-- Define the conditions
def median (s : Set ℤ) : ℤ := 150
def smallest_integer (s : Set ℤ) : ℤ := 140
def consecutive_even_integers (s : Set ℤ) : Prop := 
  ∀ x ∈ s, ∃ y ∈ s, x = y ∨ x = y + 2

-- The main theorem
theorem greatest_integer_of_set_is_152 (s : Set ℤ) 
  (h_median : median s = 150)
  (h_smallest : smallest_integer s = 140)
  (h_consecutive : consecutive_even_integers s) : 
  ∃ greatest : ℤ, greatest = 152 := 
sorry

end NUMINAMATH_GPT_greatest_integer_of_set_is_152_l2390_239081


namespace NUMINAMATH_GPT_last_three_digits_7_pow_103_l2390_239056

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end NUMINAMATH_GPT_last_three_digits_7_pow_103_l2390_239056


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2390_239088

theorem area_of_triangle_ABC (BD CE : ℝ) (angle_BD_CE : ℝ) (BD_len : BD = 9) (CE_len : CE = 15) (angle_BD_CE_deg : angle_BD_CE = 60) : 
  ∃ area : ℝ, 
    area = 90 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2390_239088


namespace NUMINAMATH_GPT_three_digit_numbers_with_repeated_digits_l2390_239018

theorem three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  total_three_digit_numbers - without_repeats = 252 := by
{
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  show total_three_digit_numbers - without_repeats = 252
  sorry
}

end NUMINAMATH_GPT_three_digit_numbers_with_repeated_digits_l2390_239018


namespace NUMINAMATH_GPT_find_K_l2390_239096

theorem find_K (K m n : ℝ) (p : ℝ) (hp : p = 0.3333333333333333)
  (eq1 : m = K * n + 5)
  (eq2 : m + 2 = K * (n + p) + 5) : 
  K = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_K_l2390_239096


namespace NUMINAMATH_GPT_area_of_triangle_l2390_239013

noncomputable def findAreaOfTriangle (a b : ℝ) (cosAOF : ℝ) : ℝ := sorry

theorem area_of_triangle (a b cosAOF : ℝ)
  (ha : a = 15 / 7)
  (hb : b = Real.sqrt 21)
  (hcos : cosAOF = 2 / 5) :
  findAreaOfTriangle a b cosAOF = 6 := by
  rw [ha, hb, hcos]
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2390_239013


namespace NUMINAMATH_GPT_age_difference_l2390_239015

variable (A B : ℕ)

-- Given conditions
def B_is_95 : Prop := B = 95
def A_after_30_years : Prop := A + 30 = 2 * (B - 30)

-- Theorem to prove
theorem age_difference (h1 : B_is_95 B) (h2 : A_after_30_years A B) : A - B = 5 := 
by
  sorry

end NUMINAMATH_GPT_age_difference_l2390_239015


namespace NUMINAMATH_GPT_BoatCrafters_l2390_239014

/-
  Let J, F, M, A represent the number of boats built in January, February,
  March, and April respectively.

  Conditions:
  1. J = 4
  2. F = J / 2
  3. M = F * 3
  4. A = M * 3

  Goal:
  Prove that J + F + M + A = 30.
-/

def BoatCrafters.total_boats_built : Nat := 4 + (4 / 2) + ((4 / 2) * 3) + (((4 / 2) * 3) * 3)

theorem BoatCrafters.boats_built_by_end_of_April : 
  BoatCrafters.total_boats_built = 30 :=   
by 
  sorry

end NUMINAMATH_GPT_BoatCrafters_l2390_239014


namespace NUMINAMATH_GPT_inequality_solution_l2390_239016

theorem inequality_solution (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3)
:=
sorry

end NUMINAMATH_GPT_inequality_solution_l2390_239016


namespace NUMINAMATH_GPT_a_fraction_of_capital_l2390_239042

theorem a_fraction_of_capital (T : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (profit_A : ℝ) (total_profit : ℝ)
  (h1 : B = T * (1 / 4))
  (h2 : C = T * (1 / 5))
  (h3 : D = T - (T * (1 / 4) + T * (1 / 5) + T * x))
  (h4 : profit_A = 805)
  (h5 : total_profit = 2415) :
  x = 161 / 483 :=
by
  sorry

end NUMINAMATH_GPT_a_fraction_of_capital_l2390_239042


namespace NUMINAMATH_GPT_min_value_quadratic_l2390_239046

theorem min_value_quadratic (x : ℝ) : x = -1 ↔ (∀ y : ℝ, x^2 + 2*x + 4 ≤ y) := by
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l2390_239046


namespace NUMINAMATH_GPT_probability_exactly_three_heads_in_seven_tosses_l2390_239008

def combinations (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) : ℚ :=
  (combinations n k) / (2^n : ℚ)

theorem probability_exactly_three_heads_in_seven_tosses :
  binomial_probability 7 3 = 35 / 128 := 
by 
  sorry

end NUMINAMATH_GPT_probability_exactly_three_heads_in_seven_tosses_l2390_239008


namespace NUMINAMATH_GPT_part1_part2_l2390_239071

-- Part (1)
theorem part1 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) (opposite : m * n < 0) :
  m + n = -3 ∨ m + n = 3 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (m - n) ≤ 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2390_239071


namespace NUMINAMATH_GPT_train_crossing_time_l2390_239007

-- Defining basic conditions
def train_length : ℕ := 150
def platform_length : ℕ := 100
def time_to_cross_post : ℕ := 15

-- The time it takes for the train to cross the platform
theorem train_crossing_time :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 25 := 
sorry

end NUMINAMATH_GPT_train_crossing_time_l2390_239007


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2390_239064

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a^2 + 1 > 0) (h2 : -1 - b^2 < 0) : 
  (a^2 + 1 > 0 ∧ -1 - b^2 < 0) ∧ (0 < a^2 + 1) ∧ (-1 - b^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2390_239064


namespace NUMINAMATH_GPT_polynomial_abc_l2390_239032

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end NUMINAMATH_GPT_polynomial_abc_l2390_239032


namespace NUMINAMATH_GPT_compound_analysis_l2390_239039

noncomputable def molecular_weight : ℝ := 18
noncomputable def atomic_weight_nitrogen : ℝ := 14.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.01

theorem compound_analysis :
  ∃ (n : ℕ) (element : String), element = "hydrogen" ∧ n = 4 ∧
  (∃ remaining_weight : ℝ, remaining_weight = molecular_weight - atomic_weight_nitrogen ∧
   ∃ k, remaining_weight / atomic_weight_hydrogen = k ∧ k = n) :=
by
  sorry

end NUMINAMATH_GPT_compound_analysis_l2390_239039


namespace NUMINAMATH_GPT_return_trip_time_l2390_239067

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end NUMINAMATH_GPT_return_trip_time_l2390_239067


namespace NUMINAMATH_GPT_line_through_intersection_and_parallel_l2390_239037

theorem line_through_intersection_and_parallel
  (x y : ℝ)
  (l1 : 3 * x + 4 * y - 2 = 0)
  (l2 : 2 * x + y + 2 = 0)
  (l3 : ∃ k : ℝ, k * x + y + 2 = 0 ∧ k = -(4 / 3)) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 4 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end NUMINAMATH_GPT_line_through_intersection_and_parallel_l2390_239037


namespace NUMINAMATH_GPT_solve_for_y_l2390_239089

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l2390_239089


namespace NUMINAMATH_GPT_first_consecutive_odd_number_l2390_239085

theorem first_consecutive_odd_number :
  ∃ k : Int, 2 * k - 1 + 2 * k + 1 + 2 * k + 3 = 2 * k - 1 + 128 ∧ 2 * k - 1 = 61 :=
by
  sorry

end NUMINAMATH_GPT_first_consecutive_odd_number_l2390_239085


namespace NUMINAMATH_GPT_group_members_l2390_239076

theorem group_members (n : ℕ) (hn : n * n = 1369) : n = 37 :=
by
  sorry

end NUMINAMATH_GPT_group_members_l2390_239076


namespace NUMINAMATH_GPT_opposite_of_neg_two_l2390_239070

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end NUMINAMATH_GPT_opposite_of_neg_two_l2390_239070


namespace NUMINAMATH_GPT_div_decimal_l2390_239079

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_div_decimal_l2390_239079


namespace NUMINAMATH_GPT_determine_m_l2390_239074

-- Define f and g according to the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

-- Define the value of x
def x := 5

-- State the main theorem we need to prove
theorem determine_m 
  (h : 3 * f x m = 2 * g x m) : m = 10 / 7 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_determine_m_l2390_239074


namespace NUMINAMATH_GPT_negation_of_proposition_l2390_239033

-- Define the proposition P(x)
def P (x : ℝ) : Prop := x + Real.log x > 0

-- Translate the problem into lean
theorem negation_of_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l2390_239033


namespace NUMINAMATH_GPT_xy_value_l2390_239022

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end NUMINAMATH_GPT_xy_value_l2390_239022


namespace NUMINAMATH_GPT_percentage_40_number_l2390_239044

theorem percentage_40_number (x y z P : ℝ) (hx : x = 93.75) (hy : y = 0.40 * x) (hz : z = 6) (heq : (P / 100) * y = z) :
  P = 16 :=
sorry

end NUMINAMATH_GPT_percentage_40_number_l2390_239044


namespace NUMINAMATH_GPT_correct_proposition_l2390_239059

theorem correct_proposition :
  (∃ x₀ : ℤ, x₀^2 = 1) ∧ ¬(∃ x₀ : ℤ, x₀^2 < 0) ∧ ¬(∀ x : ℤ, x^2 ≤ 0) ∧ ¬(∀ x : ℤ, x^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_l2390_239059


namespace NUMINAMATH_GPT_sum_radical_conjugates_l2390_239010

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_radical_conjugates_l2390_239010


namespace NUMINAMATH_GPT_intersection_of_sets_l2390_239083

def setA : Set ℝ := {x | x^2 - 1 ≥ 0}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets : (setA ∩ setB) = {x | 1 ≤ x ∧ x < 4} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2390_239083


namespace NUMINAMATH_GPT_amount_of_benzene_l2390_239060

-- Definitions of the chemical entities involved
def Benzene := Type
def Methane := Type
def Toluene := Type
def Hydrogen := Type

-- The balanced chemical equation as a condition
axiom balanced_equation : ∀ (C6H6 CH4 C7H8 H2 : ℕ), C6H6 + CH4 = C7H8 + H2

-- The proof problem: Prove the amount of Benzene required
theorem amount_of_benzene (moles_methane : ℕ) (moles_toluene : ℕ) (moles_hydrogen : ℕ) :
  moles_methane = 2 → moles_toluene = 2 → moles_hydrogen = 2 → 
  ∃ moles_benzene : ℕ, moles_benzene = 2 := by
  sorry

end NUMINAMATH_GPT_amount_of_benzene_l2390_239060


namespace NUMINAMATH_GPT_shaded_region_area_eq_l2390_239051

noncomputable def areaShadedRegion : ℝ :=
  let side_square := 14
  let side_triangle := 18
  let height := 14
  let H := 9 * Real.sqrt 3
  let BF := (side_square + side_triangle, height - H)
  let base_BF := BF.1 - 0
  let height_BF := BF.2
  let area_triangle_BFH := 0.5 * base_BF * height_BF
  let total_triangle_area := 0.5 * side_triangle * height
  let area_half_BFE := 0.5 * total_triangle_area
  area_half_BFE - area_triangle_BFH

theorem shaded_region_area_eq :
  areaShadedRegion = 9 * Real.sqrt 3 :=
by 
 sorry

end NUMINAMATH_GPT_shaded_region_area_eq_l2390_239051


namespace NUMINAMATH_GPT_points_not_all_odd_distance_l2390_239086

open Real

theorem points_not_all_odd_distance (p : Fin 4 → ℝ × ℝ) : ∃ i j : Fin 4, i ≠ j ∧ ¬ Odd (dist (p i) (p j)) := 
by
  sorry

end NUMINAMATH_GPT_points_not_all_odd_distance_l2390_239086


namespace NUMINAMATH_GPT_percentage_apples_basket_l2390_239054

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_apples_basket_l2390_239054


namespace NUMINAMATH_GPT_fraction_of_rotten_is_one_third_l2390_239061

def total_berries (blueberries cranberries raspberries : Nat) : Nat :=
  blueberries + cranberries + raspberries

def fresh_berries (berries_to_sell berries_to_keep : Nat) : Nat :=
  berries_to_sell + berries_to_keep

def rotten_berries (total fresh : Nat) : Nat :=
  total - fresh

def fraction_rot (rotten total : Nat) : Rat :=
  (rotten : Rat) / (total : Rat)

theorem fraction_of_rotten_is_one_third :
  ∀ (blueberries cranberries raspberries berries_to_sell : Nat),
    blueberries = 30 →
    cranberries = 20 →
    raspberries = 10 →
    berries_to_sell = 20 →
    fraction_rot (rotten_berries (total_berries blueberries cranberries raspberries) 
                  (fresh_berries berries_to_sell berries_to_sell))
                  (total_berries blueberries cranberries raspberries) = 1 / 3 :=
by
  intros blueberries cranberries raspberries berries_to_sell
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_fraction_of_rotten_is_one_third_l2390_239061


namespace NUMINAMATH_GPT_common_factor_of_right_triangle_l2390_239021

theorem common_factor_of_right_triangle (d : ℝ) 
  (h_triangle : (2*d)^2 + (4*d)^2 = (5*d)^2) 
  (h_side : 2*d = 45 ∨ 4*d = 45 ∨ 5*d = 45) : 
  d = 9 :=
sorry

end NUMINAMATH_GPT_common_factor_of_right_triangle_l2390_239021


namespace NUMINAMATH_GPT_z_value_l2390_239049

theorem z_value (x y z : ℝ) (h : 1 / x + 1 / y = 2 / z) : z = (x * y) / 2 :=
by
  sorry

end NUMINAMATH_GPT_z_value_l2390_239049


namespace NUMINAMATH_GPT_factorize_expression_l2390_239058

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2390_239058


namespace NUMINAMATH_GPT_find_divisor_l2390_239019

theorem find_divisor (d q r : ℕ) (h1 : d = 265) (h2 : q = 12) (h3 : r = 1) :
  ∃ x : ℕ, d = (x * q) + r ∧ x = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_divisor_l2390_239019


namespace NUMINAMATH_GPT_tangent_line_at_point_l2390_239072

noncomputable def f : ℝ → ℝ := λ x => 2 * Real.log x + x^2 

def tangent_line_equation (x y : ℝ) : Prop :=
  4 * x - y - 3 = 0 

theorem tangent_line_at_point {x y : ℝ} (h : f 1 = 1) : 
  tangent_line_equation 1 1 ∧
  y = 4 * (x - 1) + 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2390_239072


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_l2390_239077

theorem polynomial_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 160 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_l2390_239077


namespace NUMINAMATH_GPT_students_interested_in_both_l2390_239000

theorem students_interested_in_both (A B C Total : ℕ) (hA : A = 35) (hB : B = 45) (hC : C = 4) (hTotal : Total = 55) :
  A + B - 29 + C = Total :=
by
  -- Assuming the correct answer directly while skipping the proof.
  sorry

end NUMINAMATH_GPT_students_interested_in_both_l2390_239000


namespace NUMINAMATH_GPT_simplify_fraction_l2390_239017

theorem simplify_fraction (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2390_239017


namespace NUMINAMATH_GPT_union_set_subset_range_intersection_empty_l2390_239068

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

-- Question 1: When m = -1, prove A ∪ B = { x | -2 < x < 3 }
theorem union_set (m : ℝ) (h : m = -1) : A ∪ B m = { x | -2 < x ∧ x < 3 } := by
  sorry

-- Question 2: If A ⊆ B, prove m ∈ (-∞, -2]
theorem subset_range (m : ℝ) (h : A ⊆ B m) : m ∈ Set.Iic (-2) := by
  sorry

-- Question 3: If A ∩ B = ∅, prove m ∈ [0, +∞)
theorem intersection_empty (m : ℝ) (h : A ∩ B m = ∅) : m ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_GPT_union_set_subset_range_intersection_empty_l2390_239068


namespace NUMINAMATH_GPT_largest_common_number_in_range_l2390_239025

theorem largest_common_number_in_range (n1 d1 n2 d2 : ℕ) (h1 : n1 = 2) (h2 : d1 = 4) (h3 : n2 = 5) (h4 : d2 = 6) :
  ∃ k : ℕ, k ≤ 200 ∧ (∀ n3 : ℕ, n3 = n1 + d1 * k) ∧ (∀ n4 : ℕ, n4 = n2 + d2 * k) ∧ n3 = 190 ∧ n4 = 190 := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_common_number_in_range_l2390_239025


namespace NUMINAMATH_GPT_chromium_first_alloy_percentage_l2390_239030

-- Defining the conditions
def percentage_chromium_first_alloy : ℝ := 10 
def percentage_chromium_second_alloy : ℝ := 6
def mass_first_alloy : ℝ := 15
def mass_second_alloy : ℝ := 35
def percentage_chromium_new_alloy : ℝ := 7.2

-- Proving the percentage of chromium in the first alloy is 10%
theorem chromium_first_alloy_percentage : percentage_chromium_first_alloy = 10 :=
by
  sorry

end NUMINAMATH_GPT_chromium_first_alloy_percentage_l2390_239030


namespace NUMINAMATH_GPT_cannot_determine_x_l2390_239027

theorem cannot_determine_x
  (n m : ℝ) (x : ℝ)
  (h1 : n + m = 8) 
  (h2 : n * x + m * (1/5) = 1) : true :=
by {
  sorry
}

end NUMINAMATH_GPT_cannot_determine_x_l2390_239027


namespace NUMINAMATH_GPT_pairs_bought_after_donation_l2390_239098

-- Definitions from conditions
def initial_pairs : ℕ := 80
def donation_percentage : ℕ := 30
def post_donation_pairs : ℕ := 62

-- The theorem to be proven
theorem pairs_bought_after_donation : (initial_pairs - (donation_percentage * initial_pairs / 100) + 6 = post_donation_pairs) :=
by
  sorry

end NUMINAMATH_GPT_pairs_bought_after_donation_l2390_239098


namespace NUMINAMATH_GPT_problem_statement_l2390_239082

theorem problem_statement :
  (3 = 0.25 * x) ∧ (3 = 0.50 * y) → (x - y = 6) ∧ (x + y = 18) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2390_239082


namespace NUMINAMATH_GPT_value_of_a_b_c_l2390_239062

theorem value_of_a_b_c 
  (a b c : ℤ) 
  (h1 : x^2 + 12*x + 35 = (x + a)*(x + b)) 
  (h2 : x^2 - 15*x + 56 = (x - b)*(x - c)) : 
  a + b + c = 20 := 
sorry

end NUMINAMATH_GPT_value_of_a_b_c_l2390_239062


namespace NUMINAMATH_GPT_range_of_k_l2390_239035
noncomputable def quadratic_nonnegative (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 - 4 * x + 3 ≥ 0

theorem range_of_k (k : ℝ) : quadratic_nonnegative k ↔ k ∈ Set.Ici (4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2390_239035


namespace NUMINAMATH_GPT_find_ab_sum_l2390_239053

theorem find_ab_sum
  (a b : ℝ)
  (h₁ : a^3 - 3 * a^2 + 5 * a - 1 = 0)
  (h₂ : b^3 - 3 * b^2 + 5 * b - 5 = 0) :
  a + b = 2 := by
  sorry

end NUMINAMATH_GPT_find_ab_sum_l2390_239053


namespace NUMINAMATH_GPT_balance_test_l2390_239075

variable (a b h c : ℕ)

theorem balance_test
  (h1 : 4 * a + 2 * b + h = 21 * c)
  (h2 : 2 * a = b + h + 5 * c) :
  b + 2 * h = 11 * c :=
sorry

end NUMINAMATH_GPT_balance_test_l2390_239075


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2390_239099

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 * x + m = 0 ∧ y^2 - 2 * y + m = 0) ↔ m < 1 :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2390_239099


namespace NUMINAMATH_GPT_haily_cheapest_salon_l2390_239094

def cost_Gustran : ℕ := 45 + 22 + 30
def cost_Barbara : ℕ := 40 + 30 + 28
def cost_Fancy : ℕ := 30 + 34 + 20

theorem haily_cheapest_salon : min (min cost_Gustran cost_Barbara) cost_Fancy = 84 := by
  sorry

end NUMINAMATH_GPT_haily_cheapest_salon_l2390_239094


namespace NUMINAMATH_GPT_difference_of_x_values_l2390_239063

theorem difference_of_x_values : 
  ∀ x y : ℝ, ( (x + 3) ^ 2 / (3 * x + 29) = 2 ∧ (y + 3) ^ 2 / (3 * y + 29) = 2 ) → |x - y| = 14 := 
sorry

end NUMINAMATH_GPT_difference_of_x_values_l2390_239063


namespace NUMINAMATH_GPT_max_value_of_y_no_min_value_l2390_239034

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem max_value_of_y_no_min_value :
  (∃ x, -2 < x ∧ x < 2 ∧ function_y x = 5) ∧
  (∀ y, ∃ x, -2 < x ∧ x < 2 ∧ function_y x >= y) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_y_no_min_value_l2390_239034


namespace NUMINAMATH_GPT_red_tickets_for_one_yellow_l2390_239043

-- Define the conditions given in the problem
def yellow_needed := 10
def red_for_yellow (R : ℕ) := R -- This function defines the number of red tickets for one yellow
def blue_for_red := 10

def toms_yellow := 8
def toms_red := 3
def toms_blue := 7
def blue_needed := 163

-- Define the target function that converts the given conditions into a statement.
def red_tickets_for_yellow_proof : Prop :=
  ∀ R : ℕ, (2 * R = 14) → (R = 7)

-- Statement for proof where the condition leads to conclusion
theorem red_tickets_for_one_yellow : red_tickets_for_yellow_proof :=
by
  intros R h
  rw [← h, mul_comm] at h
  sorry

end NUMINAMATH_GPT_red_tickets_for_one_yellow_l2390_239043


namespace NUMINAMATH_GPT_sanjay_homework_fraction_l2390_239012

theorem sanjay_homework_fraction :
  let original := 1
  let done_on_monday := 3 / 5
  let remaining_after_monday := original - done_on_monday
  let done_on_tuesday := 1 / 3 * remaining_after_monday
  let remaining_after_tuesday := remaining_after_monday - done_on_tuesday
  remaining_after_tuesday = 4 / 15 :=
by
  -- original := 1
  -- done_on_monday := 3 / 5
  -- remaining_after_monday := 1 - 3 / 5
  -- done_on_tuesday := 1 / 3 * (1 - 3 / 5)
  -- remaining_after_tuesday := (1 - 3 / 5) - (1 / 3 * (1 - 3 / 5))
  sorry

end NUMINAMATH_GPT_sanjay_homework_fraction_l2390_239012


namespace NUMINAMATH_GPT_compute_expression_l2390_239009

theorem compute_expression :
  (5 + 7)^2 + 5^2 + 7^2 = 218 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2390_239009


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2390_239050

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (a - 3) / (a^2 + 6 * a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2390_239050


namespace NUMINAMATH_GPT_monotonic_increasing_intervals_l2390_239040

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x^2 + 4*x + 3)

theorem monotonic_increasing_intervals :
  ∀ x, f' x > 0 ↔ (x < -3 ∨ x > -1) :=
by
  intro x
  -- proof omitted
  sorry

end NUMINAMATH_GPT_monotonic_increasing_intervals_l2390_239040
